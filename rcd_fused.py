"""RCD demosaic for CUDA with fused kernels (RCD is by Luis Sanz Rodriguez, see
THIRD-PARTY-NOTICES.txt): the same algorithm and the same padding as
gpucoadd.rcd_demosaic (the reference, also used on the CPU), but each group of elementwise
operations runs as one kernel (torch jiterator, compiled at run time with the NVRTC library that
ships with torch, cached on disk: the first use ever compiles for ~20 s, later workers load the
kernels in ~0.1 s). The reference launches ~300 small kernels per batch, each reading and writing
whole images; this one ~45. Results differ from the reference only by float rounding
(operation order, fused multiply-add)."""

import atexit
import os
import re

from torch.cuda.jiterator import _create_jit_fn as _jit

# Compiled kernels are cached on disk by torch, but on Windows torch leaves each file under a
# temporary name (".._tmp_<pid>": the final rename fails), so every new worker compiled them again
# (~11 s). Use a fixed cache folder and give those files their final names at start and exit:
# later workers then load the kernels in ~0.04 s.
CACHE = os.environ.setdefault(
    "PYTORCH_KERNEL_CACHE_PATH",
    os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "GPUCoadd", "kernel_cache"),
)


def repair_cache(folder=CACHE):
    try:
        names = os.listdir(folder)
    except OSError:
        return
    for name in names:
        m = re.fullmatch(r"(.+)_tmp_\d+", name)
        if not m:
            continue
        src, dst = os.path.join(folder, name), os.path.join(folder, m.group(1))
        try:
            if os.path.exists(dst):
                os.remove(src)
            else:
                os.replace(src, dst)
        except OSError:  # still being written by another worker: the next start repairs it
            pass


os.makedirs(CACHE, exist_ok=True)
repair_cache()
atexit.register(repair_cache)

P = 4  # reflect padding, as in the reference (largest offset used is 4)

_K = {}


def _k(name, code):
    if name not in _K:
        _K[name] = _jit(code)
    return _K[name]


EPS = "T(1e-5)"
EPSSQ = "T(1e-10)"


def _kernels():
    k = {}
    # 1. / 4.1 directional discrimination: (a3 - a1 - b1 + b3 - 3 (a2 + b2) + 6 c)^2
    k["band"] = _k(
        "band",
        """template <typename T> T band(T a3, T a1, T b1, T b3, T a2, T b2, T c) {
        T d = a3 - a1 - b1 + b3 - T(3) * (a2 + b2) + T(6) * c; return d * d; }""",
    )
    # vs / (vs + hs) with vs = max(v- + v + v+, epssq), hs likewise
    k["ratio"] = _k(
        "ratio",
        f"""template <typename T> T ratio(T vm, T v, T vp, T hm, T h, T hp) {{
        T vs = vm + v + vp; vs = vs < {EPSSQ} ? {EPSSQ} : vs;
        T hs = hm + h + hp; hs = hs < {EPSSQ} ? {EPSSQ} : hs; return vs / (vs + hs); }}""",
    )
    # use the diagonal neighbourhood's direction where it is more decisive
    k["disc"] = _k(
        "disc",
        """template <typename T> T disc(T d, T a, T b, T c, T e) {
        T avg = T(0.25) * (a + b + c + e);
        T x = T(0.5) - d; x = x < T(0) ? -x : x; T y = T(0.5) - avg; y = y < T(0) ? -y : y;
        return x < y ? avg : d; }""",
    )
    k["sum4"] = _k("sum4", """template <typename T> T sum4(T a, T b, T c, T d) { return a + b + c + d; }""")
    k["lpf"] = _k(
        "lpf",
        """template <typename T> T lpf(T c, T a, T b, T d, T e, T diag) {
        return c + T(0.5) * (a + b + d + e) + T(0.25) * diag; }""",
    )
    # green gradient: eps + |a - b| + |c - d| + |n - e| + |d - f| (n: this side's neighbour)
    k["ggrad"] = _k(
        "ggrad",
        f"""template <typename T> T ggrad(T a, T b, T c, T d, T n, T e, T f) {{
        T x = a - b; T y = c - d; T z = n - e; T w = d - f;
        return {EPS} + (x < T(0) ? -x : x) + (y < T(0) ? -y : y) + (z < T(0) ? -z : z) + (w < T(0) ? -w : w); }}""",
    )
    # (w1 * n1 * (1 + (l - l1) / (eps + l + l1)) + w2 * n2 * (1 + (l - l2) / (eps + l + l2))) / (w1 + w2)
    k["gest"] = _k(
        "gest",
        f"""template <typename T> T gest(T w1, T n1, T l1, T w2, T n2, T l2, T l) {{
        T r1 = T(1) + (l - l1) / ({EPS} + l + l1); T r2 = T(1) + (l - l2) / ({EPS} + l + l2);
        return (w1 * n1 * r1 + w2 * n2 * r2) / (w2 + w1); }}""",
    )
    k["gsel"] = _k(
        "gsel",
        """template <typename T> T gsel(T gm, T c, T disc, T h, T v) {
        return gm > T(0) ? c : disc * (h - v) + v; }""",
    )
    # diagonal gradient: eps + |a - b| + |a - c| + |d - e|
    k["dgrad"] = _k(
        "dgrad",
        f"""template <typename T> T dgrad(T a, T b, T c, T d, T e) {{
        T x = a - b; T y = a - c; T z = d - e;
        return {EPS} + (x < T(0) ? -x : x) + (y < T(0) ? -y : y) + (z < T(0) ? -z : z); }}""",
    )
    # (w1 * (a1 - g1) + w2 * (a2 - g2)) / (w1 + w2)
    k["dest"] = _k(
        "dest",
        """template <typename T> T dest(T w1, T a1, T g1, T w2, T a2, T g2) {
        return (w1 * (a1 - g1) + w2 * (a2 - g2)) / (w1 + w2); }""",
    )
    k["other"] = _k(
        "other", """template <typename T> T other(T g, T disc, T q, T p) { return g + disc * (q - p) + p; }"""
    )
    # this colour's own sites keep the CFA value, the opposite colour's sites get `other`, green 0
    k["rbsel"] = _k(
        "rbsel",
        """template <typename T> T rbsel(T own, T opp, T c, T o) {
        return own > T(0) ? c : (opp > T(0) ? o : T(0)); }""",
    )
    # red/blue gradient at green sites: eps + |g - g2| + |c1 - c2| + |c1 - c3|
    k["cgrad"] = _k(
        "cgrad",
        f"""template <typename T> T cgrad(T g, T g2, T c1, T c2, T c3) {{
        T x = g - g2; T y = c1 - c2; T z = c1 - c3;
        return {EPS} + (x < T(0) ? -x : x) + (y < T(0) ? -y : y) + (z < T(0) ? -z : z); }}""",
    )
    k["csel"] = _k(
        "csel",
        """template <typename T> T csel(T gm, T g, T disc, T h, T v, T ch) {
        return gm > T(0) ? g + disc * (h - v) + v : ch; }""",
    )
    return k


def rcd_demosaic_fused(torch, F, cfa, mr, mg, mb):
    """Same contract as gpucoadd.rcd_demosaic: cfa (H, W) or (B, H, W) on CUDA, float32, scaled
    to roughly 0..1; masks (H, W). Returns (3, H, W) or (B, 3, H, W)."""
    K = _kernels()
    H, W = cfa.shape[-2:]

    def nb(a):
        ap = F.pad(a.reshape(-1, 1, H, W), (P, P, P, P), mode="reflect").reshape(*a.shape[:-2], H + 2 * P, W + 2 * P)
        return lambda dy, dx: ap[..., P + dy : P + dy + H, P + dx : P + dx + W]

    def disc(d):
        n = nb(d)
        return K["disc"](d, n(-1, -1), n(-1, 1), n(1, -1), n(1, 1))

    s = nb(cfa)
    c = cfa
    gm, rm, bm = mg.to(cfa.dtype), mr.to(cfa.dtype), mb.to(cfa.dtype)

    # 1. vertical/horizontal discrimination
    bv = K["band"](s(-3, 0), s(-1, 0), s(1, 0), s(3, 0), s(-2, 0), s(2, 0), c)
    bh = K["band"](s(0, -3), s(0, -1), s(0, 1), s(0, 3), s(0, -2), s(0, 2), c)
    v, h = nb(bv), nb(bh)
    vh_disc = disc(K["ratio"](v(-1, 0), bv, v(1, 0), h(0, -1), bh, h(0, 1)))

    # 2. low pass filter
    lpf = K["lpf"](c, s(-1, 0), s(1, 0), s(0, -1), s(0, 1), K["sum4"](s(-1, -1), s(-1, 1), s(1, -1), s(1, 1)))
    L = nb(lpf)

    # 3. green at red and blue sites
    ng = K["ggrad"](s(-1, 0), s(1, 0), c, s(-2, 0), s(-1, 0), s(-3, 0), s(-4, 0))
    sg = K["ggrad"](s(-1, 0), s(1, 0), c, s(2, 0), s(1, 0), s(3, 0), s(4, 0))
    wg = K["ggrad"](s(0, -1), s(0, 1), c, s(0, -2), s(0, -1), s(0, -3), s(0, -4))
    eg = K["ggrad"](s(0, -1), s(0, 1), c, s(0, 2), s(0, 1), s(0, 3), s(0, 4))
    v_est = K["gest"](sg, s(-1, 0), L(-2, 0), ng, s(1, 0), L(2, 0), lpf)
    h_est = K["gest"](wg, s(0, 1), L(0, 2), eg, s(0, -1), L(0, -2), lpf)
    g = K["gsel"](gm, c, vh_disc, h_est, v_est)
    G = nb(g)

    # 4.1 diagonal discrimination
    pd = K["band"](s(-3, -3), s(-1, -1), s(1, 1), s(3, 3), s(-2, -2), s(2, 2), c)
    qd = K["band"](s(-3, 3), s(-1, 1), s(1, -1), s(3, -3), s(-2, 2), s(2, -2), c)
    p, q = nb(pd), nb(qd)
    pq_disc = disc(K["ratio"](p(-1, -1), pd, p(1, 1), q(-1, 1), qd, q(1, -1)))

    # 4.2 red at blue sites and blue at red sites
    nwg = K["dgrad"](s(-1, -1), s(1, 1), s(-3, -3), g, G(-2, -2))
    neg = K["dgrad"](s(-1, 1), s(1, -1), s(-3, 3), g, G(-2, 2))
    swg = K["dgrad"](s(1, -1), s(-1, 1), s(3, -3), g, G(2, -2))
    seg = K["dgrad"](s(1, 1), s(-1, -1), s(3, 3), g, G(2, 2))
    p_est = K["dest"](nwg, s(1, 1), G(1, 1), seg, s(-1, -1), G(-1, -1))
    q_est = K["dest"](neg, s(1, -1), G(1, -1), swg, s(-1, 1), G(-1, 1))
    other = K["other"](g, pq_disc, q_est, p_est)
    r = K["rbsel"](rm, bm, c, other)
    b = K["rbsel"](bm, rm, c, other)

    # 4.3 red and blue at green sites
    rb = []
    for ch in (r, b):
        C = nb(ch)
        ngc = K["cgrad"](g, G(-2, 0), C(-1, 0), C(1, 0), C(-3, 0))
        sgc = K["cgrad"](g, G(2, 0), C(1, 0), C(-1, 0), C(3, 0))
        wgc = K["cgrad"](g, G(0, -2), C(0, -1), C(0, 1), C(0, -3))
        egc = K["cgrad"](g, G(0, 2), C(0, 1), C(0, -1), C(0, 3))
        v_c = K["dest"](ngc, C(1, 0), G(1, 0), sgc, C(-1, 0), G(-1, 0))
        h_c = K["dest"](egc, C(0, -1), G(0, -1), wgc, C(0, 1), G(0, 1))
        rb.append(K["csel"](gm, g, vh_disc, h_c, v_c, ch))
    return torch.stack([rb[0], g, rb[1]], dim=-3)
