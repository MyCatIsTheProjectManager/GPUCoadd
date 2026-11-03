"""Formatting a light for FITS as one CUDA kernel: stack_pixels' arithmetic (offset, scale, the
floor that keeps a covered pixel off 0), its handling of NaN and infinities, the FITS row order and
colour planes and big-endian bytes, all in one pass. The light stays on the GPU from warping to
here, and only the finished bytes cross to the host, into pinned memory.

The bytes are the same as the CPU path's for every float32 value; tests compare them on random bit
patterns. Each operation rounds to nearest in float32, as numpy does. When the kernel can't be built
or the buffers don't fit, the export falls back to the CPU path."""

import numpy as np
import torch

from cuda_kernels import Module

SOURCE = r"""
extern "C" __global__ void encode(const float* src, unsigned* out, int h, int w, float offset, float scale, float low) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = h * w;
    if (i >= n * 3) return;
    float v = __fdiv_rn(__fadd_rn(src[i], offset), scale);
    unsigned bits = __float_as_uint(v);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        if (bits & 0x007fffffu) bits = 0;                   /* NaN: no data */
        else bits = (bits & 0x80000000u) | 0x7f7fffffu;     /* infinity: nan_to_num's largest float */
    } else {
        if (v < low) v = low;
        if (v > 1.f) v = 1.f;
        bits = __float_as_uint(v);
    }
    unsigned be = (bits >> 24) | ((bits >> 8) & 0x0000ff00u) | ((bits << 8) & 0x00ff0000u) | (bits << 24);
    int p = i / 3;
    out[(i % 3) * n + (h - 1 - p / w) * w + p % w] = be;
}
"""

SCRATCH = 512 << 20  # GPU memory left for rendering; a tile that still doesn't fit is split
_module = None


def module():
    global _module
    if _module is None:
        _module = Module(SOURCE)
    return _module


def encode(source, output, height, width, offset, scale, low):
    """Launch the kernel: source (height, width, 3) float32 values to output's FITS bytes."""
    count = height * width * 3
    module().launch("encode", (count + 255) // 256, 256, [source, output, height, width, offset, scale, low])


class LightEncoder:
    """A canvas and formatted bytes on the GPU, and pinned host buffers the writer takes in turn.
    pixels: the largest light. All CUDA calls stay on the thread that made it."""

    def __init__(self, backend, pixels, hosts, scale, low):
        self.scale, self.low = float(scale), float(low)
        self.device = torch.empty(pixels * 3, dtype=torch.float32, device=backend.dev)
        self.packed = torch.empty(pixels * 3, dtype=torch.int32, device=backend.dev)
        self.pinned = [torch.empty(pixels * 3, dtype=torch.int32, pin_memory=True) for _ in range(hosts)]

    def canvas(self, height, width):
        return self.device[: height * width * 3].view(height, width, 3)

    def encode(self, height, width, offset, slot):
        """The canvas as FITS bytes in host buffer `slot`, which the caller must not be writing."""
        count = height * width * 3
        encode(self.device, self.packed, height, width, float(offset), self.scale, self.low)
        host = self.pinned[slot][:count]
        host.copy_(self.packed[:count], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return host.numpy().view(np.uint8)


def open_encoder(backend, pixels, hosts, scale, low, log):
    """A LightEncoder, or None with the reason logged, so the CPU path takes over."""
    from resource_guard import check_ram

    device = pixels * 24  # canvas and formatted bytes
    if pixels * 3 >= 2**31:
        reason = "the framing is too large for it"
    else:
        backend.torch.cuda.empty_cache()
        if backend.free_vram() < device + backend.VRAM_RESERVE + SCRATCH:
            reason = "GPU memory is short"
        elif not check_ram(pixels * 12 * hosts, "light export formatting"):
            reason = "RAM is short"
        else:
            try:
                module()
                return LightEncoder(backend, pixels, hosts, scale, low)
            except (RuntimeError, OSError, MemoryError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
    log(f"Formatting light exports on the CPU: {reason}. The files are the same.")
    return None
