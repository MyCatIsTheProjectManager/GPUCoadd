"""Rejection for GPU integration as single CUDA kernels (one thread per pixel and channel),
replacing the many whole-tensor mask and sum operations of sigma_clip_sorted_torch and
noise_clip_torch (which took longer than the sort itself; 3-5x for winsorized and noise-aware).
The samples are still sorted by torch (identical input). Each kernel follows its torch
counterpart statement by statement in float32, compiled without fused multiply-add so every
operation rounds alike; sums are accumulated in double, so a result can differ from torch's
float32 sums in the last bits, and a decision only where a sample lies within that rounding of a
threshold (about one pixel in ten million on real data)."""

import torch

from cuda_kernels import Module

SOURCE = r"""
#define S(i) s[(long long)(i) * M + m]
#define W(i) w[(long long)(i) * M + m]
#define SG(i) sg[(long long)(i) * M + m]

__device__ int finite_count(const float* s, int n, int M, int m) {  // NaN sorted last
    int a = 0, b = n;
    while (a < b) { int mid = (a + b) / 2; if (isnan(S(mid))) b = mid; else a = mid + 1; }
    return a;
}

// sigma_clip_sorted_torch: survivors are the run [lo, hi) of the sorted samples
extern "C" __global__ void clip_sorted(const float* s, const float* w, int n, int M, float kappa, int iters,
                                       int winsor_iters, float* out, int* kept) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    int nf = finite_count(s, n, M, m);
    int lo = 0, hi = nf;
    float mean = 0.f;
    for (int it = 0; it <= iters; ++it) {
        int cnt = max(hi - lo, 1);
        double acc = 0.0;
        for (int i = lo; i < hi; ++i) acc += S(i);
        mean = (float)acc / (float)cnt;
        if (it == iters) {
            if (w) {
                double sx = 0.0, sw = 0.0;
                for (int i = lo; i < hi; ++i) { sx += S(i) * W(i); sw += W(i); }
                mean = (float)sx / fmaxf((float)sw, 1e-12f);
            }
            break;
        }
        int i1 = min(lo + (cnt - 1) / 2, n - 1), i2 = min(lo + cnt / 2, n - 1);
        float med = 0.5f * (S(i1) + S(i2));
        double v = 0.0;
        for (int i = lo; i < hi; ++i) { float d = S(i) - mean; v += d * d; }
        float sd = sqrtf((float)v / (float)cnt);
        for (int k = 0; k < winsor_iters; ++k) {
            float a = 1.5f * sd, lw = med - a, hw = med + a;
            double sa = 0.0;
            for (int i = lo; i < hi; ++i) sa += fminf(fmaxf(S(i), lw), hw);
            float mw = (float)sa / (float)cnt;
            double sb = 0.0;
            for (int i = lo; i < hi; ++i) { float d = fminf(fmaxf(S(i), lw), hw) - mw; sb += d * d; }
            sd = 1.134f * sqrtf((float)sb / (float)cnt);
        }
        float ks = kappa * sd, tl = med - ks, th = med + ks;
        int a = 0, b = nf;  // count of samples < tl
        while (a < b) { int mid = (a + b) / 2; if (S(mid) < tl) a = mid + 1; else b = mid; }
        int nl = a;
        a = 0; b = nf;      // count of samples <= th
        while (a < b) { int mid = (a + b) / 2; if (S(mid) <= th) a = mid + 1; else b = mid; }
        int nh = a;
        int new_lo = max(lo, nl), new_hi = min(hi, nh);
        if (new_hi - new_lo >= 3) { lo = new_lo; hi = new_hi; }
    }
    out[m] = hi > lo ? mean : __int_as_float(0x7fc00000);
    if (kept) kept[m] = max(hi - lo, 0);
}

// noise_clip_torch: membership is "all finite" or |x - cu| <= ku * sig (cu, ku from the pass that set it)
extern "C" __global__ void clip_noise(const float* s, const float* sg, const float* w, int n, int M, float kappa,
                                      int iters, float* out, int* kept) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    int nf = finite_count(s, n, M, m);
    int i1 = min(max((nf - 1) / 2, 0), n - 1), i2 = min(max(nf / 2, 0), n - 1);
    float c = nf > 0 ? 0.5f * (S(i1) + S(i2)) : 0.f;
    bool all = true;
    float cu = 0.f, ku = 0.f;
#define SIG(i) fmaxf(SG(i), 1e-12f)
#define MEMBER(i) (all || fabsf(S(i) - cu) <= ku * SIG(i))
    for (int it = 0; it < iters; ++it) {
        double rr = 0.0; int k = 0;
        for (int i = 0; i < nf; ++i) if (MEMBER(i)) { float r = (S(i) - c) / SIG(i); rr += r * r; ++k; }
        float spread = sqrtf((float)rr / (float)max(k, 1));
        float ks = kappa * spread;
        int cn = 0;
        for (int i = 0; i < nf; ++i) if (fabsf(S(i) - c) <= ks * SIG(i)) ++cn;
        if (cn >= 3) { all = false; cu = c; ku = ks; }
        double num = 0.0, den = 0.0;
        for (int i = 0; i < nf; ++i) if (MEMBER(i)) { float sg2 = SIG(i) * SIG(i); float iv = 1.f / sg2; num += S(i) * iv; den += iv; }
        c = (float)num / fmaxf((float)den, 1e-30f);
    }
    int k = 0; double sx = 0.0, sw = 0.0;
    for (int i = 0; i < nf; ++i) if (MEMBER(i)) { ++k; if (w) { sx += S(i) * W(i); sw += W(i); } else sx += S(i); }
    float mean = w ? (float)sx / fmaxf((float)sw, 1e-12f) : (float)sx / (float)max(k, 1);
    out[m] = k > 0 ? mean : __int_as_float(0x7fc00000);
    if (kept) kept[m] = k;
}
"""

_module = None


def module():
    global _module
    if _module is None:
        _module = Module(SOURCE)
    return _module


def _launch(name, s, extra, kept):
    n, M = s.shape
    out = torch.empty(M, device=s.device, dtype=torch.float32)
    cnt = torch.empty(M, device=s.device, dtype=torch.int32) if kept else None
    module().launch(name, (M + 127) // 128, 128, [s, *extra(n, M), out, cnt])
    return (out, cnt) if kept else out


def clip_sorted(s, kappa, iters, w=None, kept=False, winsorize=False, winsor_iters=5):
    """As sigma_clip_sorted_torch. s: (n, M) float32 sorted along dim 0, NaN last."""
    s = s.contiguous()
    w = None if w is None else w.to(torch.float32).contiguous()
    return _launch(
        "clip_sorted", s, lambda n, M: [w, n, M, float(kappa), int(iters), winsor_iters if winsorize else 0], kept
    )


def clip_noise(s, sig, kappa, iters, w=None, kept=False):
    """As noise_clip_torch. s, sig: (n, M) float32, sorted by s along dim 0 (NaN last)."""
    s = s.contiguous()
    sig = sig.to(torch.float32).contiguous()
    w = None if w is None else w.to(torch.float32).contiguous()
    return _launch("clip_noise", s, lambda n, M: [sig, w, n, M, float(kappa), int(iters)], kept)
