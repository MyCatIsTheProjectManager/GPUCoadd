"""Fused Overlap matching normal products/reductions for the common 20000-sample batch.

Reproduces torch 2.11 Reduce.cuh's float32 vector accumulators and x/y sum tree
for >=18 output rows. Other sizes retain torch. Small systems solve in float64.
"""

import torch
from cuda_kernels import Module

SOURCE = r"""
extern "C" __global__ void normal_reduce(const long long* cols, const int* terms,
    const float* w, const float* target, double* N, double* rhs, int rows, int P, int k) {
    int row=blockIdx.x%rows, t=blockIdx.x/rows, tid=threadIdx.x;
    int i=terms[2*t], j=terms[2*t+1];
    const long long* ci=cols+4*i;
    const float* a=(const float*)ci[0];
    const long long* cj=cols+4*(j<0?0:j);
    const float* b=(const float*)cj[0];
    float acc[4]={0.f,0.f,0.f,0.f};
    for (int v=tid; v*4+3<P; v+=512) {
        #pragma unroll
        for (int z=0; z<4; ++z) {
            int p=v*4+z;
            float wi=w[(long long)row*P+p]*a[(row/3)*ci[1]+(row%3)*ci[2]+p*ci[3]];
            float bj=j<0?target[(long long)row*P+p]:b[(row/3)*cj[1]+(row%3)*cj[2]+p*cj[3]];
            acc[z] += wi*bj;
        }
    }
    float value=((acc[0]+acc[1])+acc[2])+acc[3];
    // Torch Reduce.cuh: x reduction (32 lanes), then y reduction (16 warps).
    for (int off=16;off>0;off>>=1) value += __shfl_down_sync(0xffffffff,value,off);
    __shared__ float shared[512];
    shared[tid]=value;
    for (int off=8;off>0;off>>=1) {
        __syncthreads();
        if (tid/32<off) {value += shared[tid+off*32];shared[tid]=value;}
    }
    if (tid==0) {
        double v=value;
        if(j<0) rhs[row*k+i]=v;
        else {N[(row*k+i)*k+j]=v;N[(row*k+j)*k+i]=v;}
    }
}
"""

_module = None
_terms = {}


def module():
    global _module
    if _module is None:
        _module = Module(SOURCE)
    return _module


def normal(cols, W, target):
    k = len(cols)
    C, channels, P = W.shape
    if k not in (3, 4) or channels != 3 or W.dtype != torch.float32 or not W.is_cuda:
        raise ValueError("Expected 3/4 columns and CUDA float32 weights (pairs,3,pixels)")
    if C < 6 or P != 20000:
        raise ValueError("Fused reduction requires >=6 pairs and 20000 samples")
    if target.shape != W.shape or target.dtype != W.dtype or target.device != W.device:
        raise ValueError("Target must match the weights")
    if any(
        c.ndim != 3
        or c.shape[0] != C
        or c.shape[1] not in (1, 3)
        or c.shape[2] != P
        or c.device != W.device
        or c.dtype != W.dtype
        for c in cols
    ):
        raise ValueError("Columns must broadcast to the weights on the same device and dtype")
    W, target = W.contiguous(), target.contiguous()
    desc = torch.tensor(
        [[c.data_ptr(), c.stride(0), c.stride(1) if c.shape[1] != 1 else 0, c.stride(2)] for c in cols],
        device=W.device,
        dtype=torch.int64,
    )
    key = (k, W.device)
    if key not in _terms:
        _terms[key] = torch.tensor(
            [(i, j) for i in range(k) for j in [-1, *range(i, k)]], device=W.device, dtype=torch.int32
        )
    terms = _terms[key]
    nt, rows = terms.shape[0], C * 3
    mod = module()
    N = torch.empty((C, 3, k, k), device=W.device, dtype=torch.float64)
    rhs = torch.empty((C, 3, k), device=W.device, dtype=torch.float64)
    mod.launch("normal_reduce", nt * rows, 512, [desc, terms, W, target, N, rhs, rows, P, k])
    return N, rhs
