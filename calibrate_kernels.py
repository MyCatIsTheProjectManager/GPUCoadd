"""Calibrate and reflect-pad a batch of sensor crops in one CUDA launch.

The unsigned raw bits, float32 pedestal subtraction and flat multiplication follow
GPUBackend.calibrate exactly. Each crop reflects at its own boundary before it
is placed in the batch. Frame analysis keeps the existing calibration path.
"""

import numpy as np
import torch
from cuda_kernels import Module

SOURCE = r"""
// grid: (pixel blocks, B): one row of blocks per crop, 32-bit pixel arithmetic
extern "C" __global__ void calibrate_pad(const long long* desc, float* out, int B, int H, int W, int unit) {
    int q = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * W) return;
    long long p = (long long)q * H * W + i;
    int y = i / W, x = i - y * W;
    const long long* d = desc + q * 11;
    const short* raw = (const short*)d[0];
    const float* flat = (const float*)d[1];
    int a = d[6], b = d[7], c = d[8], e = d[9];
    int py = 2 * (b - a - 1), px = 2 * (e - c - 1);
    int iy = y % py, ix = x % px;
    iy = a + min(iy, py - iy);
    ix = c + min(ix, px - ix);
    float value = (float)((unsigned short)raw[(long long)iy * d[2] + (long long)ix * d[3]]);
    float pedestal = __int_as_float((int)d[10]);
    float corrected = __fsub_rn(value, pedestal);
    float v = __fmul_rn(corrected, flat[(long long)iy * d[4] + (long long)ix * d[5]]);
    // unit: RCD's input, (calibrated + pedestal) / 65535 as torch computes it on the GPU: the sum,
    // then times float32(1 / 65535) (0x37800080; torch divides by a scalar with its reciprocal)
    out[p] = unit ? __fmul_rn(__fadd_rn(v, pedestal), __int_as_float(0x37800080)) : v;
}
"""

_module = None


def module():
    global _module
    if _module is None:
        _module = Module(SOURCE)
    return _module


def calibrate_pad(raws, flats, pedestals, boxes, out, unit=False):
    """Write (B,H,W) float32 out; raws are int16 bit views of unsigned sensor data.

    Raw/flat tensors may be strided. Metadata contains device pointers and strides,
    crop bounds and the exact float32 pedestal bits. All work uses the current stream.
    unit: write (calibrated + pedestal) / 65535 instead, the staged RCD's input, bit for bit as
    torch computes it afterwards on the GPU (the sum times the float32 reciprocal of 65535).
    """
    B, H, W = out.shape
    if not (len(raws) == len(flats) == len(pedestals) == len(boxes) == B):
        raise ValueError("Calibration batch lengths differ")
    if out.dtype != torch.float32 or not out.is_cuda or not out.is_contiguous():
        raise ValueError("Calibration output must be contiguous CUDA float32")
    desc = np.empty((B, 11), dtype=np.int64)
    for q, (raw, flat, ped, box) in enumerate(zip(raws, flats, pedestals, boxes)):
        a, b, c, d = map(int, box)
        if (
            raw.dtype != torch.int16
            or flat.dtype != torch.float32
            or raw.device != out.device
            or flat.device != out.device
            or raw.ndim != 2
            or flat.shape != raw.shape
        ):
            raise ValueError("Calibration inputs must be matching CUDA int16 raw / float32 flat images")
        if not (
            0 <= a < b <= raw.shape[0]
            and 0 <= c < d <= raw.shape[1]
            and b - a >= 2
            and d - c >= 2
            and H >= b - a
            and W >= d - c
        ):
            raise ValueError("Invalid calibration crop or padded output size")
        bits = np.asarray(ped, dtype=np.float32).view(np.int32).item()
        desc[q] = [raw.data_ptr(), flat.data_ptr(), *raw.stride(), *flat.stride(), a, b, c, d, bits]
    if B:
        metadata = torch.from_numpy(desc).to(out.device)
        module().launch("calibrate_pad", ((H * W + 255) // 256, B), 256, [metadata, out, B, H, W, 1 if unit else 0])
