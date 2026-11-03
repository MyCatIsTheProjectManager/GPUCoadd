"""Warp for GPU stacking as one CUDA kernel: for each output pixel of a tile and each light of a
batch it computes the source position in double precision, takes bicubic (cubic convolution
A = -0.75, as torch's grid_sample) or bilinear taps with zero padding, masks samples outside the
crop or within `edge` px of the sensor edge with NaN, applies the light's brightness scale and
offset, and writes into the stack. That replaces a float64 coordinate grid, grid_sample,
masked_fill, the normalization and an indexed scatter (each a pass over the whole batch).

The tap position and its fraction come straight from the double source position. grid_sample
instead goes through a normalized float32 coordinate scaled by the crop width, which moves the
position by up to ~1e-5 px depending on where the tile's crop starts; results then depended on the
tile size and the framed area (M 42: 547 pixels changed by more than 10 ADU between 512 and 256 px
tiles). Crop origins are whole pixels, so here the same output pixel always reads the same taps
with the same weights, whatever the tile. warp_torch is the same arithmetic in torch: the
fallback when the kernel cannot be built, and the reference the kernel is tested against."""

import numpy as np
import torch

from cuda_kernels import Double, Module

SOURCE = r"""
__device__ __forceinline__ float cc1(float x, float A) { return ((A + 2.f) * x - (A + 3.f)) * x * x + 1.f; }
__device__ __forceinline__ float cc2(float x, float A) { return ((A * x - 5.f * A) * x + 8.f * A) * x - 4.f * A; }
__device__ __forceinline__ float cubic(float x0, float x1, float x2, float x3, float t) {
    const float A = -0.75f;
    float c0 = cc2(t + 1.f, A), c1 = cc1(t, A), c2 = cc1(1.f - t, A), c3 = cc2((1.f - t) + 1.f, A);
    return x0 * c0 + x1 * c1 + x2 * c2 + x3 * c3;
}
__device__ __forceinline__ float tap(const float* p, int x, int y, int W, int H) {
    return (x >= 0 && x < W && y >= 0 && y < H) ? p[(long long)y * W + x] : 0.f;
}

// rgb (B, 3, Hm, Wm); mats (B, 9) dst->src; box (B, 4) a, b, c, d of each crop; wh (B, 2) sensor
// size; x0, y0: the tile's position in the matrices' output coordinates; sc, of (B, 3); gr (B, 3, 2)
// per-light plane in ADU per output pixel or null; slot (B) row of each light in the stack
// (n, 3, R, C); bicubic: 1 bicubic, 0 bilinear
extern "C" __global__ void warp(const float* rgb, int B, int Hm, int Wm, const double* mats, const double* box,
                                const double* wh, double edge, double x0, double y0, int R, int C,
                                const float* sc, const float* of, const float* gr, const long long* slot, float* stack,
                                int bicubic) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long per = (long long)R * C;
    if (idx >= per * B) return;
    int b = (int)(idx / per);
    int p = (int)(idx % per), yy = p / C, xx = p % C;
    const double* M = mats + 9 * b;
    double gx = x0 + xx, gy = y0 + yy;
    double a = box[4 * b], bb = box[4 * b + 1], c = box[4 * b + 2], d = box[4 * b + 3];
    // separate rounding per operation, as warp_torch
    double den = __dadd_rn(__dadd_rn(__dmul_rn(M[6], gx), __dmul_rn(M[7], gy)), M[8]);
    double sx = __dadd_rn(__ddiv_rn(__dadd_rn(__dadd_rn(__dmul_rn(M[0], gx), __dmul_rn(M[1], gy)), M[2]), den), -c);
    double sy = __dadd_rn(__ddiv_rn(__dadd_rn(__dadd_rn(__dmul_rn(M[3], gx), __dmul_rn(M[4], gy)), M[5]), den), -a);
    double W = wh[2 * b], H = wh[2 * b + 1];
    double lox = fmax(__dadd_rn(edge, -c), 0.0), hix = fmin(__dadd_rn(__dadd_rn(d, -c), -1.0), __dadd_rn(__dadd_rn(__dadd_rn(W, -1.0), -edge), -c));
    double loy = fmax(__dadd_rn(edge, -a), 0.0), hiy = fmin(__dadd_rn(__dadd_rn(bb, -a), -1.0), __dadd_rn(__dadd_rn(__dadd_rn(H, -1.0), -edge), -a));
    bool valid = sx >= lox && sx <= hix && sy >= loy && sy <= hiy;
    long long o = ((long long)slot[b] * 3) * per + p;
    if (!valid) {
        float nan = __int_as_float(0x7fc00000);
        stack[o] = nan; stack[o + per] = nan; stack[o + 2 * per] = nan;
        return;
    }
    // whole crop origins: sx - floor(sx) is the same fraction in every tile (exact in double)
    double fx = floor(sx), fy = floor(sy);
    float tx = (float)__dadd_rn(sx, -fx), ty = (float)__dadd_rn(sy, -fy);
    int x_ = (int)fx, y_ = (int)fy;
    for (int ch = 0; ch < 3; ++ch) {
        const float* src = rgb + ((long long)b * 3 + ch) * Hm * Wm;
        float v;
        if (bicubic) {
            float r[4];
            for (int i = 0; i < 4; ++i) {
                int y = y_ - 1 + i;
                r[i] = cubic(tap(src, x_ - 1, y, Wm, Hm), tap(src, x_, y, Wm, Hm), tap(src, x_ + 1, y, Wm, Hm), tap(src, x_ + 2, y, Wm, Hm), tx);
            }
            v = cubic(r[0], r[1], r[2], r[3], ty);
        } else {
            float wx = __fadd_rn(1.f, -tx), wy = __fadd_rn(1.f, -ty);
            v = __fadd_rn(__fadd_rn(__fmul_rn(__fmul_rn(wx, wy), tap(src, x_, y_, Wm, Hm)),
                                    __fmul_rn(__fmul_rn(tx, wy), tap(src, x_ + 1, y_, Wm, Hm))),
                          __fadd_rn(__fmul_rn(__fmul_rn(wx, ty), tap(src, x_, y_ + 1, Wm, Hm)),
                                    __fmul_rn(__fmul_rn(tx, ty), tap(src, x_ + 1, y_ + 1, Wm, Hm))));
        }
        v = __fadd_rn(__fmul_rn(v, sc[3 * b + ch]), of[3 * b + ch]);
        if (gr) v = __fadd_rn(v, (float)__dadd_rn(__dmul_rn((double)gr[6 * b + 2 * ch], gx), __dmul_rn((double)gr[6 * b + 2 * ch + 1], gy)));  // gradient matching
        stack[o + ch * per] = v;
    }
}
"""

_modules = {}


def module(fmad=True):
    if fmad not in _modules:
        _modules[fmad] = Module(SOURCE, fmad=fmad)
    return _modules[fmad]


def warp(rgb, mats, boxes, sizes, edge, x0, y0, rows, cols, sc, of, slots, stack, fmad=True, planes=None, interp="bicubic"):
    """rgb (B, 3, Hm, Wm) float32; mats (B, 3, 3), boxes (B, 4), sizes (B, 2) numpy; sc, of (B, 3)
    numpy float32; slots: rows of `stack` (n, 3, rows, cols) to write; planes: optional (B, 3, 2)
    per-light plane (gradient matching), ADU per output pixel in x and y, added after scale and
    offset; interp: "bicubic" or "bilinear"."""
    dev = rgb.device
    B, _, Hm, Wm = rgb.shape
    t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a, dtype=dt)).to(dev)
    args = [
        rgb.contiguous(),
        B,
        Hm,
        Wm,
        t(np.reshape(mats, (B, 9)), np.float64),
        t(boxes, np.float64),
        t(sizes, np.float64),
        Double(edge),
        Double(x0),
        Double(y0),
        rows,
        cols,
        t(sc, np.float32),
        t(of, np.float32),
        None if planes is None else t(np.reshape(planes, (B, 6)), np.float32),
        t(slots, np.int64),
        stack,
        1 if interp == "bicubic" else 0,
    ]
    total = B * rows * cols
    module(fmad).launch("warp", (total + 255) // 256, 256, args)


def _cubic(x0, x1, x2, x3, t):
    A = -0.75
    cc1 = lambda x: ((A + 2) * x - (A + 3)) * x * x + 1
    cc2 = lambda x: ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
    return x0 * cc2(t + 1) + x1 * cc1(t) + x2 * cc1(1 - t) + x3 * cc2((1 - t) + 1)


def warp_torch(rgb, mats, boxes, sizes, edge, x0, y0, rows, cols, sc, of, slots, stack, planes=None, interp="bicubic"):
    """warp() in torch, operation for operation (the kernel may fuse multiply-adds, so the two agree
    to float rounding): the fallback when the kernel cannot be built, and its test reference."""
    dev = rgb.device
    B, _, Hm, Wm = rgb.shape
    t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a, dtype=dt)).to(dev)
    gy, gx = torch.meshgrid(
        torch.arange(rows, device=dev, dtype=torch.float64) + y0,
        torch.arange(cols, device=dev, dtype=torch.float64) + x0,
        indexing="ij",
    )
    mv, box, wh = t(mats, np.float64), t(boxes, np.float64), t(sizes, np.float64)
    e = lambda r, k: mv[:, r, k, None, None]
    a, b, c, d = (box[:, k, None, None] for k in range(4))
    den = e(2, 0) * gx + e(2, 1) * gy + e(2, 2)
    sx = (e(0, 0) * gx + e(0, 1) * gy + e(0, 2)) / den - c
    sy = (e(1, 0) * gx + e(1, 1) * gy + e(1, 2)) / den - a
    Wd, Hd = wh[:, 0, None, None], wh[:, 1, None, None]
    valid = (
        (sx >= (edge - c).clamp(min=0))
        & (sx <= torch.minimum(d - c - 1, Wd - 1 - edge - c))
        & (sy >= (edge - a).clamp(min=0))
        & (sy <= torch.minimum(b - a - 1, Hd - 1 - edge - a))
    )
    fx, fy = sx.floor(), sy.floor()
    tx, ty = (sx - fx).float()[:, None], (sy - fy).float()[:, None]
    ix, iy = fx.long(), fy.long()
    flat = rgb.reshape(B, 3, Hm * Wm)

    def tap(dx, dy):
        x, y = ix + dx, iy + dy
        inside = (x >= 0) & (x < Wm) & (y >= 0) & (y < Hm)
        index = (y.clamp(0, Hm - 1) * Wm + x.clamp(0, Wm - 1)).reshape(B, 1, -1).expand(B, 3, -1)
        values = flat.gather(2, index).reshape(B, 3, rows, cols)
        return values * inside[:, None]

    if interp == "bicubic":
        r = [_cubic(tap(-1, k), tap(0, k), tap(1, k), tap(2, k), tx) for k in (-1, 0, 1, 2)]
        out = _cubic(*r, ty)
    else:
        wx, wy = 1 - tx, 1 - ty
        out = (wx * wy * tap(0, 0) + tx * wy * tap(1, 0)) + (wx * ty * tap(0, 1) + tx * ty * tap(1, 1))
    out = out * t(sc, np.float32)[:, :, None, None] + t(of, np.float32)[:, :, None, None]
    if planes is not None:  # gradient matching: each light's plane over the output coordinates
        g = t(planes, np.float32).double()
        out = out + (g[:, :, 0, None, None] * gx + g[:, :, 1, None, None] * gy).float()
    out.masked_fill_(~valid[:, None], float("nan"))
    stack[t(slots, np.int64)] = out
