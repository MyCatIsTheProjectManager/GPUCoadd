"""Warp/register bitonic sorting for GPU samples. NaNs last.

sort_samples keeps each sample's frame index paired with it (ties by index). sort_values sorts
values only: equal values may come out in any frame order, which leaves the sorted values
unchanged, so it serves rejection that gathers nothing by frame. Generated networks use shuffle
exchanges and register pairs. A block sorts a few neighbouring pixels: their samples pass through
shared memory in whole rows, so global reads and writes use full memory sectors.
"""

PAD = 4  # shared-memory row padding: staging and per-warp reads without bank conflicts


def columns(capacity):
    """Pixels per block (one warp each), within 48 KB of shared memory for values and indices."""
    return 8 if capacity <= 512 else 4

PRELUDE = r"""
__device__ bool less(float a, int ai, float b, int bi) {
    bool an = isnan(a), bn = isnan(b);
    if (an != bn) return !an;
    if (!an && a != b) return a < b;
    return ai < bi;
}
__device__ bool vless(float a, float b) {
    bool an = isnan(a), bn = isnan(b);
    if (an != bn) return !an;
    return !an && a < b;
}
"""


def _kernel(capacity, indices):
    """One generated bitonic network; indices=False drops the index registers and output."""
    registers = capacity // 32

    def lt(a, b):
        return f"less(v{a},i{a},v{b},i{b})" if indices else f"vless(v{a},v{b})"

    name, order = ("sort_samples", "long long* order, ") if indices else ("sort_values", "")
    cols, row = columns(capacity), capacity + PAD
    staged = f"__shared__ float sv[{cols * row}];" + (f" __shared__ int si[{cols * row}];" if indices else "")
    lines = [
        f"""extern "C" __global__ void {name}(const float* x, float* y, {order}
                                       int n, int M, int stride, int pixelstride,
                                       int outframe, int outpixel) {{
    {staged}
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32, m0 = blockIdx.x * {cols};
    for (int t = threadIdx.x; t < n * {cols}; t += blockDim.x) {{
        int f = t / {cols}, c = t % {cols};
        sv[c * {row} + f] = m0 + c < M ? x[(long long)f * stride + (long long)(m0 + c) * pixelstride] : 0.f;
    }}
    __syncthreads();
    const float* col = sv + warp * {row};"""
    ]
    for r in range(registers):
        index = f"int i{r} = lane + {32 * r}; " if indices else ""
        lines.append(f"{index}float v{r} = lane + {32 * r} < n ? col[lane + {32 * r}] : __int_as_float(0x7fc00000);")
    k = 2
    while k <= capacity:
        j = k // 2
        while j:
            if j < 32:
                for r in range(registers):
                    low = f"bool low = (((lane+{32 * r}) & {k}) == 0) == (((lane+{32 * r}) & {j}) == 0); "
                    if indices:
                        lines.append(
                            f"{{ float b = __shfl_xor_sync(0xffffffff, v{r}, {j}); int bi = __shfl_xor_sync(0xffffffff, i{r}, {j}); "
                            + low
                            + f"if (low ? less(b,bi,v{r},i{r}) : less(v{r},i{r},b,bi)) {{v{r}=b;i{r}=bi;}} }}"
                        )
                    else:
                        lines.append(
                            f"{{ float b = __shfl_xor_sync(0xffffffff, v{r}, {j}); "
                            + low
                            + f"if (low ? vless(b,v{r}) : vless(v{r},b)) v{r}=b; }}"
                        )
            else:
                for r in range(registers):
                    other = r ^ (j // 32)
                    if r < other:
                        # For k >= 32, direction is constant for this pair of registers.
                        ascending = (32 * r & k) == 0
                        cond = lt(other, r) if ascending else lt(r, other)
                        swap = f"float t=v{r};v{r}=v{other};v{other}=t;"
                        if indices:
                            swap += f"int ti=i{r};i{r}=i{other};i{other}=ti;"
                        lines.append(f"if ({cond}) {{{swap}}}")
            j //= 2
        k *= 2
    lines.append("__syncthreads();")
    for r in range(registers):
        at = f"warp * {row} + lane + {32 * r}"
        lines.append(f"if (lane + {32 * r} < n) {{ sv[{at}] = v{r};" + (f" si[{at}] = i{r};" if indices else "") + " }")
    store = f"y[at] = sv[c * {row} + f];" + (f" order[at] = si[c * {row} + f];" if indices else "")
    lines.append(f"""__syncthreads();
    for (int t = threadIdx.x; t < n * {cols}; t += blockDim.x) {{
        int f = t / {cols}, c = t % {cols};
        if (m0 + c < M) {{ long long at = (long long)f * outframe + (long long)(m0 + c) * outpixel; {store} }}
    }}""")
    return "\n".join(lines) + "\n}"


def source(capacity):
    return PRELUDE + _kernel(capacity, True) + "\n" + _kernel(capacity, False)


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


def sort_values(x):
    """Sorted values only (dim 0, NaN last), for rejection that needs no frame indices."""
    import torch

    if x.shape[0] > 512:
        return torch.sort(x, dim=0).values
    return _launch(x, False)


def sort_samples(x):
    return _launch(x, True)


def _launch(x, indices):
    import torch
    from cuda_kernels import Module

    n, m = x.shape
    capacity = max(32, 1 << (n - 1).bit_length())
    if x.dtype != torch.float32 or not x.is_cuda or x.stride(1) != 1 or n < 1 or m < 1 or capacity > 1024:
        raise ValueError("Expected CUDA float32, contiguous columns, 1..1024 frames")
    if capacity not in MODULES:
        MODULES[capacity] = Module(source(capacity))
    values = torch.empty((n, m), device=x.device, dtype=torch.float32)
    cols = columns(capacity)
    grid, block = (m + cols - 1) // cols, 32 * cols
    if not indices:
        MODULES[capacity].launch("sort_values", grid, block, [x, values, n, m, x.stride(0), 1, m, 1])
        return values
    order = torch.empty((n, m), device=x.device, dtype=torch.int64)
    MODULES[capacity].launch("sort_samples", grid, block, [x, values, order, n, m, x.stride(0), 1, m, 1])
    return values, order
