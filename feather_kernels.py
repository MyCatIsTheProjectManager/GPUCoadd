"""Feather weights: for each light and output pixel of a tile, a weight that rises from the light's
(trimmed) sensor edge to 1 at `radius` pixels inside it, so the mean blends lights in across their
edges instead of stepping where coverage changes. The ramp is smootherstep, t^3 (6t^2 - 15t + 10)
with t = distance / radius. The distance is measured in the
light's own pixels to the edge of its trimmed sensor area (the area the warp keeps), from the same
source position as the warp (double precision). Outside that area the weight is 0 (the sample is
NaN anyway); inside it never drops below FLOOR, so a pixel that only one light's edge covers keeps
that light's value instead of becoming 0/0.

Three implementations with the same arithmetic: a CUDA kernel (one thread per light and pixel),
torch (if the kernel cannot be built) and numpy (CPU backend)."""

import numpy as np

from cuda_kernels import Double, Module

FLOOR = 1e-3

SOURCE = r"""
extern "C" __global__ void feather_weights(const double* mats, const double* wh, int B, double edge, double x0,
                                           double y0, int R, int C, double radius, float floor_w,
                                           const long long* slot, float* out) {
    // grid: (pixel blocks, B), one row of blocks per light
    int b = blockIdx.y;
    long long per = (long long)R * C;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= per) return;
    int yy = p / C, xx = p - yy * C;
    const double* M = mats + 9 * b;
    double gx = x0 + xx, gy = y0 + yy;
    double den = M[6] * gx + M[7] * gy + M[8];
    double sx = (M[0] * gx + M[1] * gy + M[2]) / den;
    double sy = (M[3] * gx + M[4] * gy + M[5]) / den;
    double W = wh[2 * b], H = wh[2 * b + 1];
    double d = fmin(fmin(sx - edge, W - 1.0 - edge - sx), fmin(sy - edge, H - 1.0 - edge - sy));
    float w = 0.f;
    if (d >= 0.0) {
        double t = fmin(d / radius, 1.0);
        w = fmaxf((float)(t * t * t * (t * (6.0 * t - 15.0) + 10.0)), floor_w);
    }
    out[slot[b] * per + p] = w;
}
"""

_module = None


def module():
    global _module
    if _module is None:
        _module = Module(SOURCE, fmad=False)
    return _module


def ramp(t):
    """Smootherstep on 0..1 (numpy or torch)."""
    return t * t * t * (t * (6 * t - 15) + 10)


def feather_cuda(torch, out, slots, mats, sizes, edge, x0, y0, radius):
    """out: (n, rows, cols) float32 CUDA tensor; slots: its rows to fill, one per light; mats (B, 3, 3)
    canvas -> sensor; sizes (B, 2) sensor width, height."""
    B = len(slots)
    rows, cols = out.shape[1], out.shape[2]
    t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a, dtype=dt)).to(out.device)
    args = [
        t(np.reshape(mats, (B, 9)), np.float64),
        t(sizes, np.float64),
        B,
        Double(edge),
        Double(x0),
        Double(y0),
        rows,
        cols,
        Double(radius),
        float(FLOOR),
        t(slots, np.int64),
        out,
    ]  # Python float: packed as float32
    if B:
        module().launch("feather_weights", ((rows * cols + 255) // 256, B), 256, args)


def distance(xp, gx, gy, m, w, h, edge):
    """Distance (light pixels) from canvas pixels (gx, gy) to the trimmed sensor edge; negative outside."""
    den = m[2, 0] * gx + m[2, 1] * gy + m[2, 2]
    sx = (m[0, 0] * gx + m[0, 1] * gy + m[0, 2]) / den
    sy = (m[1, 0] * gx + m[1, 1] * gy + m[1, 2]) / den
    return xp.minimum(xp.minimum(sx - edge, w - 1.0 - edge - sx), xp.minimum(sy - edge, h - 1.0 - edge - sy))


def weights_from_distance(xp, d, radius):
    w = xp.clip(ramp(xp.clip(d / radius, 0, 1)), FLOOR, 1)
    return xp.where(d >= 0, w, 0)


def feather_torch(torch, out, slots, mats, sizes, edge, x0, y0, radius):
    rows, cols = out.shape[1], out.shape[2]
    gy, gx = torch.meshgrid(
        torch.arange(y0, y0 + rows, device=out.device, dtype=torch.float64),
        torch.arange(x0, x0 + cols, device=out.device, dtype=torch.float64),
        indexing="ij",
    )
    for s, m, (w, h) in zip(slots, mats, sizes):
        d = distance(torch, gx, gy, torch.as_tensor(np.asarray(m, np.float64), device=out.device), w, h, edge)
        out[s] = weights_from_distance(torch, d, radius).float()


def feather_numpy(out, slots, mats, sizes, edge, x0, y0, radius):
    rows, cols = out.shape[1], out.shape[2]
    gy, gx = np.mgrid[y0 : y0 + rows, x0 : x0 + cols].astype(np.float64)
    for s, m, (w, h) in zip(slots, mats, sizes):
        out[s] = weights_from_distance(np, distance(np, gx, gy, np.asarray(m, np.float64), w, h, edge), radius)
