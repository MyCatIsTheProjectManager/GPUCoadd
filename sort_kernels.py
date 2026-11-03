"""Warp/register bitonic sorting for GPU samples. NaNs last; ties by input index.

Indices remain paired with samples. Generated networks use shuffle exchanges and
register pairs, without a full sample array in shared memory.
"""


def source(capacity):
    registers = capacity // 32
    lines = [
        r"""
__device__ bool less(float a, int ai, float b, int bi) {
    bool an = isnan(a), bn = isnan(b);
    if (an != bn) return !an;
    if (!an && a != b) return a < b;
    return ai < bi;
}
extern "C" __global__ void sort_samples(const float* x, float* y, long long* order,
                                       int n, int M, int stride, int pixelstride,
                                       int outframe, int outpixel) {
    int lane = threadIdx.x & 31;
    int m = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (m >= M) return;
"""
    ]
    for r in range(registers):
        lines.append(
            f"int i{r} = lane + {32 * r}; float v{r} = i{r} < n ? x[(long long)i{r}*stride+(long long)m*pixelstride] : __int_as_float(0x7fc00000);"
        )
    k = 2
    while k <= capacity:
        j = k // 2
        while j:
            if j < 32:
                for r in range(registers):
                    lines.append(
                        f"{{ float b = __shfl_xor_sync(0xffffffff, v{r}, {j}); int bi = __shfl_xor_sync(0xffffffff, i{r}, {j}); "
                        f"bool low = (((lane+{32 * r}) & {k}) == 0) == (((lane+{32 * r}) & {j}) == 0); "
                        f"if (low ? less(b,bi,v{r},i{r}) : less(v{r},i{r},b,bi)) {{v{r}=b;i{r}=bi;}} }}"
                    )
            else:
                for r in range(registers):
                    other = r ^ (j // 32)
                    if r < other:
                        # For k >= 32, direction is constant for this pair of registers.
                        ascending = (32 * r & k) == 0
                        cond = (
                            f"less(v{other},i{other},v{r},i{r})" if ascending else f"less(v{r},i{r},v{other},i{other})"
                        )
                        lines.append(
                            f"if ({cond}) {{float t=v{r};v{r}=v{other};v{other}=t;int ti=i{r};i{r}=i{other};i{other}=ti;}}"
                        )
            j //= 2
        k *= 2
    for r in range(registers):
        lines.append(
            f"if (lane+{32 * r}<n) {{long long at=(long long)(lane+{32 * r})*outframe+(long long)m*outpixel;y[at]=v{r};order[at]=i{r};}}"
        )
    return "\n".join(lines) + "\n}"


MODULES = {}


def prepare(max_frames):
    """Compile supported sizes before timed stacking; larger sorts stay with torch."""
    from cuda_kernels import Module

    capacity = 32
    limit = min(512, max(32, 1 << (max_frames - 1).bit_length()))
    while capacity <= limit:
        if capacity not in MODULES:
            MODULES[capacity] = Module(source(capacity))
        capacity *= 2


def sort(x):
    import torch

    if x.shape[0] > 512:
        return torch.sort(x, dim=0)
    values, indices = sort_samples(x)
    return torch.return_types.sort((values, indices))


def sort_samples(x):
    import torch
    from cuda_kernels import Module

    n, m = x.shape
    capacity = max(32, 1 << (n - 1).bit_length())
    if x.dtype != torch.float32 or not x.is_cuda or x.stride(1) != 1 or n < 1 or m < 1 or capacity > 1024:
        raise ValueError("Expected CUDA float32, contiguous columns, 1..1024 frames")
    if capacity not in MODULES:
        MODULES[capacity] = Module(source(capacity))
    values = torch.empty((n, m), device=x.device, dtype=torch.float32)
    indices = torch.empty((n, m), device=x.device, dtype=torch.int64)
    MODULES[capacity].launch("sort_samples", (m + 3) // 4, 128, [x, values, indices, n, m, x.stride(0), 1, m, 1])
    return values, indices
