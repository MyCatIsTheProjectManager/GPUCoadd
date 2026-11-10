"""Overlap-based brightness normalization ("Overlap matching", described in the README's expert section).

Lights are matched on the sky they actually share, so real nebula and stars cancel. Computed
once per heat-map analysis (every retained light loaded) at 1/8 resolution and cached with the
alignment; a stack then applies the cached per-light, per-channel (scale, offset).

Pipeline (validated against synthetic ground truth and held-out real pairs, see the design doc;
tools/overlap_norm.py is the reference prototype this module reproduces in batched form):

1. Low-res lights: CFA superpixels (sensor-edge band masked) warped onto a half-resolution union
   canvas, 4 x 4 median on the canvas, common Gaussian blur; residual stars masked from a
   provisional consensus.
2. Pairs: k largest overlaps + random substantial partners + maximum spanning tree + mosaic
   registration links (largest overlaps alone split a field into islands of time-neighbours).
3. Pair fits, all pairs batched on the GPU: d = y - x = k (m - m0) + delta + plane, robustly, with
   m the noise-weighted mean (unbiased when one light is noisier), each light's effective noise
   from pair residuals, and the slope corrected for attenuation by noise in m.
4. Network: a log-scale per light shared by channels plus a per-channel deviation, per-channel
   offsets; sparse weighted least squares, reference fixed, priors (photometric) pulling only
   where pairs carry no information, Huber down-weighting of bad pairs, continuous pair weights.
   Noise frozen after NOISE_UPDATES iterations and weights after WEIGHT_UPDATES (otherwise weak
   channels jitter or flip), sign-flip damping as a safeguard.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sky_store import stored

VERSION = 1  # bump when results change: cached normalizations are rejected
BIN = 8  # low-res pixel = 8 x 8 sensor pixels
PRIOR_SIGMA = 0.05  # uncertainty of a prior log-scale
COLOUR_SIGMA = 0.02  # prior width of a per-channel deviation from the shared scale
SCALE_SE_MAX = 0.05  # reporting only ("scale from prior" when no pair is more precise)
GROUP_SIGMA = 0.15  # prior width (log) of a light group's scale per channel around its star prior
LAM_TAPER = (0.25, 0.5)  # pairs whose regressor is mostly noise fade out smoothly
HUBER = 2.5  # robust-sigma units beyond which a pair is down-weighted
MAX_PIXELS = 20000  # pixels per pair fit (fixed random subset per pair)
NOISE_UPDATES = 3
WEIGHT_UPDATES = 5
CORRELATION = 2.0  # standard-error inflation for correlated low-res pixels (measured 1.9-2.15)
CHUNK = 48  # pairs fitted together on the GPU
MIN_SHARED = 100  # low-res pixels for a pair


# --------------------------------------------------------------------------- low-res lights


def superpixels(cfa, edge):
    """(3, h/2, w/2) RGB from a calibrated GRBG CFA; superpixels touching the sensor-edge band -> NaN."""
    rgb = torch.stack([cfa[0::2, 1::2], 0.5 * (cfa[0::2, 0::2] + cfa[1::2, 1::2]), cfa[1::2, 0::2]]).float()
    k = -(-edge // 2)
    if k:
        rgb[:, :k], rgb[:, -k:], rgb[:, :, :k], rgb[:, :, -k:] = (float("nan"),) * 4
    return rgb


def smooth(img, sigma=1.0):
    """Gaussian blur (sigma in low-res px) of a (3, h, w) image; pixels within the kernel radius of
    a NaN become NaN. A common blur much wider than any interpolation blur makes lights with
    different sub-pixel phases (the reference has none) comparable."""
    r = 2
    k = torch.exp(-(torch.arange(-r, r + 1, device=img.device, dtype=torch.float32) ** 2) / (2 * sigma**2))
    k = k / k.sum()
    ok = torch.isfinite(img).all(0, keepdim=True).float()
    x = torch.nan_to_num(img) * ok
    blur = lambda z: F.conv2d(
        F.conv2d(F.pad(z[None], (r, r, r, r)), k.view(1, 1, 1, -1).expand(z.shape[0], 1, 1, -1), groups=z.shape[0]),
        k.view(1, 1, -1, 1).expand(z.shape[0], 1, -1, 1),
        groups=z.shape[0],
    )[0]
    full = F.max_pool2d(1 - ok[None], 2 * r + 1, 1, r)[0] == 0  # whole kernel footprint valid
    return blur(x).masked_fill_(~full.expand_as(img), float("nan"))


def warp_lowres(sp, minv, H8, W8):
    """Warp superpixels onto the half-resolution canvas (bilinear, NaN outside), then 4 x 4 median
    on the canvas and the common blur -> (3, H8, W8). Binning after warping gives every light, the
    reference included, the same effective resolution."""
    dev = sp.device
    h2, w2 = sp.shape[-2:]
    m = torch.from_numpy(minv).to(dev)
    corners = np.linalg.inv(minv) @ np.array([[0, 2 * w2, 0, 2 * w2], [0, 0, 2 * h2, 2 * h2], [1, 1, 1, 1.0]])
    cx, cy = corners[0] / corners[2] / 2, corners[1] / corners[2] / 2
    U0, V0 = max(0, int(np.floor(cx.min())) // 4 - 1), max(0, int(np.floor(cy.min())) // 4 - 1)
    U1, V1 = min(W8, int(np.ceil(cx.max())) // 4 + 2), min(H8, int(np.ceil(cy.max())) // 4 + 2)
    out = torch.full((3, H8, W8), float("nan"), device=dev)
    if U1 <= U0 or V1 <= V0:
        return out
    V, U = torch.meshgrid(
        torch.arange(4 * V0, 4 * V1, device=dev, dtype=torch.float64),
        torch.arange(4 * U0, 4 * U1, device=dev, dtype=torch.float64),
        indexing="ij",
    )
    X, Y = 2 * U + 0.5, 2 * V + 0.5  # canvas coordinates of half-res pixel centres
    den = m[2, 0] * X + m[2, 1] * Y + m[2, 2]
    u = ((m[0, 0] * X + m[0, 1] * Y + m[0, 2]) / den - 0.5) / 2  # superpixel coordinates
    v = ((m[1, 0] * X + m[1, 1] * Y + m[1, 2]) / den - 0.5) / 2
    grid = torch.stack([2 * u / (w2 - 1) - 1, 2 * v / (h2 - 1) - 1], -1).float()[None]
    half = F.grid_sample(sp[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    half.masked_fill_(~((u >= 0) & (u <= w2 - 1) & (v >= 0) & (v <= h2 - 1))[None], float("nan"))
    nb, nu = V1 - V0, U1 - U0
    blocks = half.reshape(3, nb, 4, nu, 4).permute(0, 1, 3, 2, 4).reshape(3, nb, nu, 16)
    binned = blocks.median(-1).values
    binned[~torch.isfinite(blocks).all(-1)] = float("nan")  # partial blocks are not used
    out[:, V0:V1, U0:U1] = smooth(binned)
    return out


def robust(x):
    med = x.median()
    return med, 1.4826 * (x - med).abs().median()


def star_mask(L, valid, a_g, b_g):
    """Residual stars: high-pass of a provisional consensus (median of the lights with the given
    green scale/offset) > 5 sigma, dilated by one low-res pixel. Only excludes pixels."""
    n, _, H8, W8 = L.shape
    if stored(L):
        L.owner.require(H8 * W8 * 8, "consensus canvas")
    a0, b0 = a_g[:, None, None], b_g[:, None, None]
    prov = torch.empty((H8, W8), device=L.device)
    step = L.owner.consensus_rows(n, H8, W8) if stored(L) else 32
    for r0 in range(0, H8, step):
        if stored(L):
            L.owner.phase("star consensus", r0, H8)
        chunk = (L[:, 1, r0 : r0 + step] * a0 + b0).masked_fill_(~valid[:, r0 : r0 + step], float("nan"))
        prov[r0 : r0 + step] = torch.nanmedian(chunk, dim=0).values
    filled = torch.nan_to_num(prov, nan=float(torch.nanmedian(prov)))
    patches = F.pad(filled[None, None], (3, 3, 3, 3), mode="replicate").unfold(2, 7, 1).unfold(3, 7, 1)
    if stored(L):
        L.owner.require(H8 * W8 * 32 + min(H8, 32) * W8 * 49 * 4, "star mask")
        hp = torch.empty_like(filled)
        for r0 in range(0, H8, 32):
            L.owner.phase("star mask", r0, H8)
            hp[r0 : r0 + 32] = filled[r0 : r0 + 32] - patches[0, 0, r0 : r0 + 32].reshape(-1, W8, 49).median(-1).values
    else:
        hp = filled - patches.reshape(H8, W8, 49).median(-1).values
    hp_med, hp_sd = robust(hp[torch.isfinite(prov)])
    return F.max_pool2d(((hp - hp_med) > 5 * hp_sd).float()[None, None], 3, 1, 1)[0, 0] > 0


# --------------------------------------------------------------------------- pairs


def select_pairs(overlap, k=4, extra=4, required=(), min_shared=MIN_SHARED, seed=0):
    """Each light's k largest overlaps, `extra` random partners among the lights it overlaps by at
    least half its largest overlap, a maximum spanning tree and the required pairs."""
    n = overlap.shape[0]
    rng = np.random.default_rng(seed)
    pairs = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in required}
    for a in range(n):
        o = overlap[a].copy()
        o[a] = 0
        order = np.argsort(o)[::-1]
        for b in order[:k]:
            if o[b] >= min_shared:
                pairs.add((min(a, int(b)), max(a, int(b))))
        pool = [int(b) for b in order[k:] if o[b] >= max(min_shared, 0.5 * o[order[0]])]
        for b in rng.permutation(pool)[:extra]:
            pairs.add((min(a, int(b)), max(a, int(b))))
    inside, best, parent = np.zeros(n, bool), np.full(n, -1.0), np.full(n, -1)
    for start in range(n):  # maximum spanning tree (Prim)
        if inside[start]:
            continue
        best[start] = np.inf
        while True:
            cand = np.where(~inside & (best >= 0), best, -1.0)
            v = int(np.argmax(cand))
            if cand[v] < 0:
                break
            inside[v] = True
            if parent[v] >= 0:
                pairs.add((min(v, int(parent[v])), max(v, int(parent[v]))))
            upd = ~inside & (overlap[v] >= min_shared) & (overlap[v] > best)
            best[upd], parent[upd] = overlap[v][upd], v
    return sorted(p for p in pairs if overlap[p] >= min_shared)


def sample_pixels(usable_flat, pairs, seed=0):
    """(n_pairs, P) flat indices of up to MAX_PIXELS shared usable pixels per pair, -1 padded; a
    fixed random subset per pair (deterministic), so every iteration fits the same pixels."""
    npix = usable_flat.shape[1]
    P = min(MAX_PIXELS, npix)
    out = (
        usable_flat.owner.create("samples", (len(pairs), P), np.int64)
        if stored(usable_flat)
        else torch.empty((len(pairs), P), dtype=torch.long, device=usable_flat.device)
    )
    pi = torch.tensor([p[0] for p in pairs], device=usable_flat.device)
    pj = torch.tensor([p[1] for p in pairs], device=usable_flat.device)
    gen = torch.Generator(device=usable_flat.device).manual_seed(seed)
    for s in range(0, len(pairs), 16):
        if stored(usable_flat):
            usable_flat.owner.require(16 * npix * 16 + 16 * P * 16, "exact sample selection")
            usable_flat.owner.phase("sample selection", s, len(pairs))
        shared = usable_flat[pi[s : s + 16]] & usable_flat[pj[s : s + 16]]
        keys = torch.rand(shared.shape, generator=gen, device=usable_flat.device)
        keys[~shared] = -1
        vals, ids = keys.topk(P, dim=1)
        ids[vals < 0] = -1
        out[s : s + 16] = ids
    return out


# --------------------------------------------------------------------------- batched pair fits

_pair_kernels = None
_pair_failure = None


def pair_kernels():
    """Reference override or cached runtime kernel, with a safe compile-failure fallback."""
    global _pair_kernels, _pair_failure
    if os.environ.get("GPUCOADD_OVERLAP") == "torch":
        return None, "torch (GPUCOADD_OVERLAP)"
    if not torch.__version__.split("+")[0].startswith("2.11."):
        return None, "torch (pair kernel supports the torch 2.11 sum order)"
    if _pair_kernels is None and _pair_failure is None:
        try:
            import overlap_kernels

            overlap_kernels.module()
            _pair_kernels = overlap_kernels
        except Exception as exc:
            _pair_failure = f"{type(exc).__name__}: {exc}"
    return (
        (_pair_kernels, "CUDA normal systems; torch for other sizes")
        if _pair_kernels is not None
        else (None, f"torch (pair kernel unavailable: {_pair_failure})")
    )


def _normal(cols, W, target):
    if W.is_cuda and W.dtype == torch.float32 and W.shape[0] >= 6 and W.shape[-1] == 20000:
        kernels, _ = pair_kernels()
        if kernels is not None:
            return kernels.normal(cols, W, target)
    return _normal_torch(cols, W, target)


def _normal_torch(cols, W, target):
    """Normal equations of a weighted least-squares fit with a few regressor columns (each
    broadcastable to (C, 3, P)), as direct sums: much faster than matrix products for 3-4
    unknowns over 20,000 pixels. Returns N (C, 3, k, k) and rhs (C, 3, k) in float64."""
    k = len(cols)
    shape = W.shape[:-1]
    N = torch.empty(shape + (k, k), device=W.device, dtype=torch.float64)
    rhs = torch.empty(shape + (k,), device=W.device, dtype=torch.float64)
    for i in range(k):
        Wi = W * cols[i]
        rhs[..., i] = (Wi * target).sum(-1).double()
        for j in range(i, k):
            N[..., i, j] = N[..., j, i] = (Wi * cols[j]).sum(-1).double()
    return N, rhs


def fit_pairs(Lr, XY, idx, pairs, a, b, noise, correlation=CORRELATION, dtype=torch.float32):
    """All pair fits. Lr: (n, 3, npix) lights (own units; a view of the canvas stack, no copy:
    a pixel-major copy doubled memory and pushed a 12 GB GPU into paging); XY: (npix, 2) low-res canvas
    coordinates; idx: sample_pixels output; a, b, noise: (n, 3) numpy. Returns arrays (n_pairs, 3):
    k, se_k, lam, delta, m0, sigma, n; and the pair's residual plane px, py (normalized ADU per
    1000 low-res px, about the overlap centroid cx, cy) with the fitted pixels' coordinate spread
    sxx, syy (the slopes' information) for gradient matching (solve_planes).

    The large sums run in float32 (consumer GPUs run float64 at 1/64 speed): the brightness
    regressor is centred and divided by its spread so all columns are O(1) and the 4 x 4 normal
    equations are well conditioned; only those small systems are solved in float64."""
    dev = Lr.device
    npairs = len(pairs)
    ch = torch.arange(3, device=dev)[None, :, None]
    pi = torch.tensor([p[0] for p in pairs], device=dev)
    pj = torch.tensor([p[1] for p in pairs], device=dev)
    at = torch.tensor(a, device=dev, dtype=dtype)
    bt = torch.tensor(b, device=dev, dtype=dtype)
    st = torch.tensor(noise * a, device=dev, dtype=torch.float64)
    keys = ("k", "se_k", "lam", "delta", "m0", "sigma", "n", "px", "py", "cx", "cy", "sxx", "syy")
    out = {k: np.zeros((npairs, 3)) for k in keys}
    eye4 = 1e-9 * torch.eye(4, device=dev, dtype=torch.float64)
    eye3 = 1e-9 * torch.eye(3, device=dev, dtype=torch.float64)
    nan = float("nan")
    for s in range(0, npairs, CHUNK):
        if stored(Lr):
            Lr.owner.require(min(CHUNK, npairs - s) * 3 * idx.shape[1] * 4 * 100, "pair fits")
            Lr.owner.phase("pair fits", s, npairs)
        I, J, ix = pi[s : s + CHUNK], pj[s : s + CHUNK], idx[s : s + CHUNK]
        valid = ix >= 0
        ixc = ix.clamp(min=0)
        # gathered directly as contiguous (C, 3, P); padded slots point at pixel 0, which may be
        # NaN: zero them (NaN * 0 weight is still NaN)
        v3 = valid[:, None, :]
        xs = torch.where(v3, Lr[I[:, None, None], ch, ixc[:, None, :]].to(dtype), 0.0)
        ys = torch.where(v3, Lr[J[:, None, None], ch, ixc[:, None, :]].to(dtype), 0.0)
        x = xs * at[I][..., None] + bt[I][..., None]
        y = ys * at[J][..., None] + bt[J][..., None]
        vx64, vy64 = (st[I] ** 2)[..., None], (st[J] ** 2)[..., None]
        vx, vy = vx64.to(dtype), vy64.to(dtype)
        m = (vy * x + vx * y) / (vx + vy)
        d = y - x
        w0 = valid[:, None, :].expand_as(d).to(dtype)
        cnt = w0.sum(-1, keepdim=True).clamp(min=1)
        m0 = (m * w0).sum(-1, keepdim=True) / cnt
        mc = (m - m0) * w0
        sm = ((mc**2).sum(-1, keepdim=True) / cnt).sqrt().clamp(min=1e-12)  # regressor spread
        xy = XY[ixc].to(dtype)  # (C, P, 2)
        wv = valid.to(dtype)[..., None]
        cxy = (xy * wv).sum(1, keepdim=True) / wv.sum(1, keepdim=True).clamp(min=1)
        Xc = ((xy[..., 0] - cxy[..., 0]) / 1000)[:, None]  # (C, 1, P): same for all channels
        Yc = ((xy[..., 1] - cxy[..., 1]) / 1000)[:, None]
        u0 = mc / sm  # standardized brightness regressor, O(1) like the others
        one = torch.ones_like(Xc)
        cols = [u0, Xc, Yc, one]
        use = valid[:, None, :].expand_as(d).clone()
        for _ in range(4):  # robust: clip at 3 sigma and refit
            W = use.to(dtype)
            N, rhs = _normal(cols, W, d)
            coef = torch.linalg.solve(N + eye4, rhs).to(dtype)
            r = d - (coef[..., 0:1] * u0 + coef[..., 1:2] * Xc + coef[..., 2:3] * Yc + coef[..., 3:4])
            sig = 1.4826 * torch.nanmedian(torch.where(use, r.abs(), torch.full_like(r, nan)), dim=-1).values.double()
            use = valid[:, None, :] & (r.abs() <= 3 * sig.to(dtype)[..., None])
        coef = coef.double()
        W = use.to(dtype)
        N, _ = _normal(cols, W, d)
        smd = sm[..., 0].double()
        cov00 = torch.linalg.inv(N + eye4)[..., 0, 0] / smd**2  # back to brightness units
        slope = coef[..., 0] / smd
        # signal variance of m after removing what the plane and offset explain
        NQ, rq = _normal([Xc, Yc, one], W, u0)
        cq = torch.linalg.solve(NQ + eye3, rq).to(dtype)
        resid = u0 - (cq[..., 0:1] * Xc + cq[..., 1:2] * Yc + cq[..., 2:3])
        nu = W.sum(-1).double()
        var_tot = (resid**2 * W).sum(-1).double() / (nu - 1).clamp(min=1) * smd**2
        var_noise = (vx64 * vy64 / (vx64 + vy64))[..., 0]
        var_sig = (var_tot - var_noise).clamp(min=1e-30)
        lam = (var_sig / var_tot.clamp(min=1e-30)).clamp(min=1e-6)
        res = {
            "k": slope / lam,
            "se_k": correlation * sig * cov00.clamp(min=0).sqrt() / lam,
            "lam": lam,
            "delta": coef[..., 3],
            "m0": m0[..., 0].double(),
            "sigma": sig,
            "n": nu,
            "px": coef[..., 1],
            "py": coef[..., 2],
            "cx": cxy[:, 0, 0].double()[:, None].expand(-1, 3),
            "cy": cxy[:, 0, 1].double()[:, None].expand(-1, 3),
            "sxx": (Xc**2 * W).sum(-1).double() / nu.clamp(min=1),
            "syy": (Yc**2 * W).sum(-1).double() / nu.clamp(min=1),
        }
        for key in keys:
            out[key][s : s + CHUNK] = res[key].cpu().numpy()
    return out


# --------------------------------------------------------------------------- network


def noise_from_pairs(n, pairs, sigma, current, groups=None):
    """Per-light effective noise (normalized units) from pair residuals sigma_ij^2 = v_i + v_j,
    by median polish (robust to a few bad pairs). Lights without pairs keep `current`.
    groups: (n,) labels; a light then uses only partners of its own group when it has any. With
    a ~9x noisier partner (another sensor), 1% error in the partner's variance is ~10% of the
    light's own, and the pair fits' regressor weights (and so their scales) depend on it."""
    out = current.copy()
    partners = [[] for _ in range(n)]
    for q, (i, j) in enumerate(pairs):
        partners[i].append((j, q))
        partners[j].append((i, q))
    if groups is not None:
        for k in range(n):
            own = [(j, q) for j, q in partners[k] if groups[j] == groups[k]]
            if own:
                partners[k] = own
    seen = np.array([len(p) > 0 for p in partners])
    if not seen.any():
        return out
    s2 = sigma**2
    v = np.zeros((n, 3))
    for k in range(n):
        if partners[k]:
            v[k] = 0.5 * np.median(s2[[q for _, q in partners[k]]], 0)
    for _ in range(20):
        new = v.copy()
        for k in range(n):
            if partners[k]:
                js = [j for j, _ in partners[k]]
                qs = [q for _, q in partners[k]]
                new[k] = np.median(s2[qs] - v[js], 0)
        floor = 0.05 * np.median(new[seen], 0)
        new[seen] = np.maximum(new[seen], floor)
        if np.allclose(new, v, rtol=1e-4):
            v = new
            break
        v = new
    out[seen] = np.sqrt(v[seen])
    return out


def pixel_noise(L, usable):
    """Initial per-light noise (own units) from horizontal second differences; one light at a
    time (whole-stack temporaries would be several times the size of L)."""
    if stored(L):
        L.owner.require(L.shape[-1] * L.shape[-2] * 60, "pixel noise")
    out = np.zeros((L.shape[0], 3))
    for k in range(L.shape[0]):
        if stored(L):
            L.owner.phase("initial noise", k, L.shape[0])
        d = L[k, :, :, :-2] - 2 * L[k, :, :, 1:-1] + L[k, :, :, 2:]
        ok = usable[k, :, :-2] & usable[k, :, 1:-1] & usable[k, :, 2:]
        for c in range(3):
            v = d[c][ok]
            out[k, c] = float(1.4826 * (v - v.median()).abs().median() / np.sqrt(6)) if v.numel() > 10 else np.nan
    return out


def overlap_counts(flat, block=1 << 18):
    """(n, n) shared usable pixels between lights, accumulated over pixel blocks (a float copy of
    the whole mask would cost 4 bytes x lights x pixels)."""
    if stored(flat):
        from sky_store import overlap_counts as bounded_counts

        return bounded_counts(flat, block)
    n, npix = flat.shape
    ov = torch.zeros((n, n), device=flat.device, dtype=torch.float32)
    for s in range(0, npix, block):
        f = flat[:, s : s + block].float()
        ov += f @ f.T
    return ov.cpu().numpy()


def own_noise_floor(L, usable, noise, correlation=CORRELATION, max_pixels=50000):
    """(n, 3) smallest log-scale uncertainty a light's own noise allows (pairs share it, so they
    do not average it down)."""
    n, _, H, W = L.shape
    if stored(L):
        L.owner.require(H * W * 40, "uncertainty floor")
    YY, XX = torch.meshgrid(
        torch.arange(H, device=L.device, dtype=torch.float64),
        torch.arange(W, device=L.device, dtype=torch.float64),
        indexing="ij",
    )
    gen = torch.Generator(device=L.device).manual_seed(1)
    var = torch.full((n, 3), float("nan"), device=L.device, dtype=torch.float64)
    counts = np.zeros(n)
    for k in range(n):
        if stored(L):
            L.owner.phase("uncertainty floor", k, n)
        u = usable[k]
        cnt = int(u.sum())
        if cnt < 100:
            continue
        counts[k] = cnt
        sel = torch.nonzero(u.reshape(-1))[:, 0]
        if sel.numel() > max_pixels:
            sel = sel[torch.randperm(sel.numel(), generator=gen, device=L.device)[:max_pixels]]
        Q = torch.stack(
            [XX.reshape(-1)[sel], YY.reshape(-1)[sel], torch.ones(sel.numel(), device=L.device, dtype=torch.float64)], 1
        )
        # the three channels share the design matrix: one solve, read back once after the loop
        V = L[k].reshape(3, -1)[:, sel].double().T
        var[k] = (V - Q @ torch.linalg.lstsq(Q, V).solution).var(0)
    var = var.cpu().numpy()
    out = np.full((n, 3), np.inf)
    ok = counts > 0
    var_sig = np.maximum(var[ok] - np.asarray(noise, np.float64)[ok] ** 2, 1e-30)
    out[ok] = correlation * np.asarray(noise, np.float64)[ok] / np.sqrt(counts[ok, None] * var_sig)
    return out


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def scale_weights(fits):
    """(n_pairs, 3) continuous weights of the scale measurements: precision, tapered by signal share."""
    lo, hi = LAM_TAPER
    taper = np.clip((fits["lam"] - lo) / (hi - lo), 0, 1) ** 2
    se = fits["se_k"]
    ok = np.isfinite(se) & (se > 0)
    return np.where(ok, taper / np.where(ok, se, 1) ** 2, 0.0)


def _sparse_solve(rows, cols, vals, rhs, wts, nunk, fixed, want_cov=False):
    """Weighted least squares sum w (A u - rhs)^2 with unknowns in `fixed` held at 0. Returns u and
    the posterior covariance (dense, only when want_cov and small enough; else None)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    keep = np.ones(nunk, bool)
    keep[list(fixed)] = False
    remap = -np.ones(nunk, int)
    remap[keep] = np.arange(keep.sum())
    m = np.asarray(cols) >= 0
    rows, cols, vals = np.asarray(rows)[m], remap[np.asarray(cols)[m]], np.asarray(vals)[m]
    good = cols >= 0
    A = sp.csr_matrix((vals[good], (rows[good], cols[good])), shape=(len(rhs), int(keep.sum())))
    sw = np.sqrt(np.asarray(wts))
    Aw = sp.diags(sw) @ A
    Nm = (Aw.T @ Aw).tocsc()
    sol = spsolve(Nm, Aw.T @ (np.asarray(rhs) * sw))
    u = np.zeros(nunk)
    u[keep] = sol
    cov = None
    if want_cov and Nm.shape[0] <= 6000:
        from resource_guard import check_ram

        check_ram(nunk * nunk * 8 * 5, "Overlap matching exact covariance")
        cov = np.zeros((nunk, nunk))
        cov[np.ix_(keep, keep)] = np.linalg.inv(Nm.toarray())
    return u, cov


def solve_scales(n, ref, pairs, fits, weights, target, group=None, gcum=None, want_cov=False):
    """Shared log-scale + per-channel deviation corrections, sparse. Returns corrections (n, 3)
    and their variance (n, 3) (NaN when not asked for, or too large to invert densely: the dense
    inverse costs ~1 s for 600 lights, so normalize() asks only for the final system's).
    group: (n,) index of each light's group term, -1 for the reference's group (none). A group
    term (per channel) moves all of its lights together; the lights' priors (`target`) then only
    steer their deviations from it, so a systematic prior error of a whole group (another sensor's
    colour response: its stars and its sky do not scale alike) is set by the overlaps instead.
    gcum: (G, 3) group terms applied so far (their prior pulls the total weakly towards 0).
    Returns, with groups, also the group-term corrections (G, 3) and their variance."""
    ng = 0 if group is None else int(group.max()) + 1
    ncols = 4 * n + 3 * ng
    sw = scale_weights(fits) * weights
    # Pair rows, in the order of a loop over pairs then channels; each row's entries in a fixed
    # order (-1 = absent, dropped by _sparse_solve), so the matrix is built exactly as before.
    P = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    qq, cc = np.nonzero(sw > 0)
    i, j = P[qq, 0], P[qq, 1]
    cols = np.full((len(qq), 6), -1, np.int64)
    vals = np.zeros((len(qq), 6), np.int64)
    cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3] = i, n + 3 * i + cc, j, n + 3 * j + cc
    vals[:, :4] = (1, 1, -1, -1)
    if ng:
        cross = group[i] != group[j]
        gi, gj = cross & (group[i] >= 0), cross & (group[j] >= 0)
        cols[gi, 4], vals[gi, 4] = 4 * n + 3 * group[i[gi]] + cc[gi], 1
        cols[gj, 5], vals[gj, 5] = 4 * n + 3 * group[j[gj]] + cc[gj], -1
    rows = np.repeat(np.arange(len(qq)), 6)
    rhs, wts = [fits["k"][qq, cc]], [sw[qq, cc]]
    # Prior rows per light: shared scale, then its three channel deviations
    shared = target.mean(1)
    k = np.arange(n)
    pc = np.stack([k, n + 3 * k, n + 3 * k + 1, n + 3 * k + 2], 1)
    rows = np.concatenate([rows, len(qq) + np.arange(4 * n)])
    cols = np.concatenate([cols.ravel(), pc.ravel()])
    vals = np.concatenate([vals.ravel(), np.ones(4 * n, np.int64)])
    rhs.append(np.concatenate([shared[:, None], target - shared[:, None]], 1).ravel())
    wts.append(np.tile([1 / PRIOR_SIGMA**2] + [1 / COLOUR_SIGMA**2] * 3, n))
    if ng:  # group-term prior rows
        r0 = len(qq) + 4 * n
        rows = np.concatenate([rows, r0 + np.arange(3 * ng)])
        cols = np.concatenate([cols, 4 * n + np.arange(3 * ng)])
        vals = np.concatenate([vals, np.ones(3 * ng, np.int64)])
        rhs.append(-gcum.ravel())
        wts.append(np.full(3 * ng, 1 / GROUP_SIGMA**2))
    fixed = {ref, n + 3 * ref, n + 3 * ref + 1, n + 3 * ref + 2}
    u, cov = _sparse_solve(rows, cols, vals, np.concatenate(rhs), np.concatenate(wts), ncols, fixed, want_cov=want_cov)
    A = u[:n, None] + u[n : 4 * n].reshape(n, 3)
    if ng:
        T = u[4 * n :].reshape(ng, 3)
        A = A + np.where(group[:, None] >= 0, T[np.maximum(group, 0)], 0)
    if cov is None:
        var = np.full((n, 3), np.nan)
        tvar = np.full((ng, 3), np.nan)
    else:
        k = k[:, None]
        g = n + 3 * k + np.arange(3)[None]
        var = cov[k, k] + cov[g, g] + 2 * cov[k, g]
        if ng:
            t = 4 * n + 3 * np.maximum(group, 0)[:, None] + np.arange(3)[None]
            has = (group >= 0)[:, None]
            var = var + np.where(has, cov[t, t] + 2 * cov[k, t] + 2 * cov[g, t], 0)
            tg = 4 * n + 3 * np.arange(ng)[:, None] + np.arange(3)[None]
            tvar = cov[tg, tg]
    if ng:
        return A, var, T, tvar
    return A, var


def solve_offsets(n, ref, pairs, fits, weights, A, c):
    P = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    w = weights[:, c] * fits["n"][:, c] / np.maximum(fits["sigma"][:, c], 1e-30) ** 2
    q = np.flatnonzero(w > 0)
    i, j = P[q, 0], P[q, 1]
    rows = np.concatenate([np.repeat(np.arange(len(q)), 2), len(q) + np.arange(n)])
    cols = np.concatenate([np.stack([i, j], 1).ravel(), np.arange(n)])
    vals = np.concatenate([np.tile([1, -1], len(q)), np.ones(n, np.int64)])
    rhs = np.concatenate([fits["delta"][q, c] - (A[i, c] - A[j, c]) * fits["m0"][q, c], np.zeros(n)])
    # negligible ridge: lights without offset information stay at 0
    wts = np.concatenate([w[q], np.full(n, 1e-12)])
    u, _ = _sparse_solve(rows, cols, vals, rhs, wts, n, {ref})
    return u


def solve_planes(n, ref, pairs, fits, weights, c):
    """Gradient matching: per-light plane and offset corrections (channel c) so that overlapping
    lights agree everywhere on their shared sky, not only at their overlap centroids. The pair fits
    measure, for each pair, the plane by which light j still differs from light i after the a, b
    normalization (px, py about the centroid cx, cy; sky structure is in both lights and cancels).
    Per light, in normalized units per 1000 low-res px: g_i - g_j = (px, py); beta_i - beta_j =
    delta - px cx - py cy. Gauge: a plane common to every light is indistinguishable from sky
    in the overlaps, so the solved planes are shifted to mean 0 over the lights with pair
    information, and so are the offset corrections: the stack keeps the background level and the
    average gradient it has without the option, and only the disagreement between lights is removed;
    lights without pairs keep 0, the mean (negligible ridge).
    Returns beta (n,), gx (n,), gy (n,) and diagnostics: the pair slopes' RMS, their median standard
    error and the RMS left after the solve (normalized ADU per 1000 low-res px)."""
    P = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    w = weights[:, c] * fits["n"][:, c] / np.maximum(fits["sigma"][:, c], 1e-30) ** 2
    q = np.flatnonzero(w > 0)
    i, j = P[q, 0], P[q, 1]
    rows = np.concatenate([np.repeat(np.arange(len(q)), 2), len(q) + np.arange(n)])
    cols = np.concatenate([np.stack([i, j], 1).ravel(), np.arange(n)])
    vals = np.concatenate([np.tile([1, -1], len(q)), np.ones(n, np.int64)])
    ridge = np.full(n, 1e-12)
    cx, cy = fits["cx"][q, c] / 1000, fits["cy"][q, c] / 1000
    px, py = fits["px"][q, c], fits["py"][q, c]
    # the slopes' precision grows with the fitted pixels' coordinate spread
    gx, _ = _sparse_solve(rows, cols, vals, np.concatenate([px, np.zeros(n)]),
                          np.concatenate([w[q] * fits["sxx"][q, c], ridge]), n, {ref})
    gy, _ = _sparse_solve(rows, cols, vals, np.concatenate([py, np.zeros(n)]),
                          np.concatenate([w[q] * fits["syy"][q, c], ridge]), n, {ref})
    rhs = fits["delta"][q, c] - px * cx - py * cy
    beta, _ = _sparse_solve(rows, cols, vals, np.concatenate([rhs, np.zeros(n)]), np.concatenate([w[q], ridge]), n, {ref})
    measured = np.zeros(n, bool)
    measured[i], measured[j], measured[ref] = True, True, True
    # the whole correction field (offset and plane) has mean zero over the measured lights: the stack keeps
    # the background level and the average gradient it has without the option
    beta[measured] -= beta[measured].mean()
    gx[measured] -= gx[measured].mean()
    gy[measured] -= gy[measured].mean()
    se = fits["sigma"][q, c] / np.sqrt(np.maximum(fits["n"][q, c], 1))
    diag = {
        "px_rms": float(np.sqrt(np.mean(px**2))) if len(q) else 0.0,
        "py_rms": float(np.sqrt(np.mean(py**2))) if len(q) else 0.0,
        "px_se_median": float(np.median(se / np.sqrt(np.maximum(fits["sxx"][q, c], 1e-30)))) if len(q) else 0.0,
        "py_se_median": float(np.median(se / np.sqrt(np.maximum(fits["syy"][q, c], 1e-30)))) if len(q) else 0.0,
        "px_residual_rms": float(np.sqrt(np.mean((px - (gx[i] - gx[j])) ** 2))) if len(q) else 0.0,
        "py_residual_rms": float(np.sqrt(np.mean((py - (gy[i] - gy[j])) ** 2))) if len(q) else 0.0,
    }
    return beta, gx, gy, diag


def huber(res, scale):
    return np.minimum(1.0, HUBER / np.maximum(np.abs(res) / max(scale, 1e-12), 1e-12))


def normalize(
    L,
    usable,
    ref,
    prior_scale,
    init_offset=None,
    pairs=None,
    required=(),
    iterations=12,
    seed=0,
    correlation=CORRELATION,
    log=None,
    groups=None,
):
    """L: (n, 3, H, W) lights on the low-res canvas (NaN outside); usable: (n, H, W) bool.
    prior_scale / init_offset: (n, 3). Returns a, b (n, 3), scale_sigma, noise, flags, diagnostics.
    groups: (n,) group label per light (e.g. telescope); groups other than the reference's get a
    free scale term per channel (solve_scales), reported as "group_terms" (factor vs the prior)."""
    t0 = time.perf_counter()
    n, _, H, W = L.shape
    dev = L.device
    flat = usable.reshape(n, -1)
    if pairs is None:
        pairs = select_pairs(overlap_counts(flat), required=required, seed=seed)
    Lr = L.reshape(n, 3, -1)  # view, no copy
    YY, XX = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    XY = torch.stack([XX.reshape(-1), YY.reshape(-1)], 1)
    idx = sample_pixels(flat, pairs, seed)
    noise = pixel_noise(L, usable)
    prior = np.array(prior_scale, float)
    a = prior.copy()
    b = np.zeros((n, 3)) if init_offset is None else np.array(init_offset, float).copy()
    a[ref], b[ref] = 1.0, 0.0
    comp = components(n, pairs)
    disconnected = [i for i in range(n) if comp[i] != comp[ref]]
    npairs = len(pairs)
    w_scale, w_off = np.ones((npairs, 3)), np.ones((npairs, 3))
    pi = np.array([p[0] for p in pairs], int)
    pj = np.array([p[1] for p in pairs], int)
    history = []
    prev_A, flips = np.zeros((n, 3)), np.zeros((n, 3))
    var = np.full((n, 3), np.nan)
    gidx, glabels = None, []
    if groups is not None and len(set(groups)) > 1:
        glabels = [g for g in dict.fromkeys(groups) if g != groups[ref]]
        gidx = np.array([glabels.index(g) if g in glabels else -1 for g in groups])
    gcum = np.zeros((len(glabels), 3))
    tvar = np.full((len(glabels), 3), np.nan)
    shift = lambda: np.where(gidx[:, None] >= 0, gcum[np.maximum(gidx, 0)], 0) if gidx is not None else 0
    for it in range(iterations):
        fits = fit_pairs(Lr, XY, idx, pairs, a, b, noise, correlation)
        if it < NOISE_UPDATES:
            noise = noise_from_pairs(n, pairs, fits["sigma"], noise * a, None if gidx is None else gidx) / a
        target = np.log(prior / a) + shift()  # the priors move with their group's term
        for rnd in range(3 if it < WEIGHT_UPDATES else 1):
            last = (fits, w_scale, target, None if gidx is None else gcum.copy())
            if gidx is None:
                Acorr, _ = solve_scales(n, ref, pairs, fits, w_scale, target)
            else:
                Acorr, _, T, _ = solve_scales(n, ref, pairs, fits, w_scale, target, gidx, gcum)
            if it >= WEIGHT_UPDATES:
                break
            active = scale_weights(fits) > 0
            res = (Acorr[pi] - Acorr[pj] - fits["k"]) / np.where(active, fits["se_k"], 1)
            s = 1.4826 * np.median(np.abs(res[active])) if active.any() else 1.0
            w_scale = np.where(active, huber(res, s), w_scale)
        beta = np.zeros((n, 3))
        for c in range(3):
            for rnd in range(3 if it < WEIGHT_UPDATES else 1):
                bc = solve_offsets(n, ref, pairs, fits, w_off, Acorr, c)
                if it >= WEIGHT_UPDATES:
                    break
                res = bc[pi] - bc[pj] - (fits["delta"][:, c] - (Acorr[pi, c] - Acorr[pj, c]) * fits["m0"][:, c])
                s = 1.4826 * np.median(np.abs(res)) if npairs else 1.0
                w_off[:, c] = huber(res, s)
            beta[:, c] = bc
        Acorr[disconnected], beta[disconnected] = 0, 0
        flip = (np.sign(Acorr) * np.sign(prev_A) < 0) & (np.abs(Acorr) > 1e-4)
        damp = np.where(flip, 0.5 ** np.minimum(flips + 1, 20), 1.0)
        flips = np.where(flip, flips + 1, flips)
        Acorr *= damp
        beta *= damp
        if gidx is not None:  # a group term follows its (damped) lights: mean damping of its members
            for gi in range(len(glabels)):
                m = gidx == gi
                gcum[gi] += T[gi] * damp[m].mean(0)
        prev_A = Acorr.copy()
        a = a * np.exp(Acorr)  # new normalized value = exp(A) * (a L + b) + beta
        b = b * np.exp(Acorr) + beta
        a[ref], b[ref] = 1.0, 0.0
        history.append(float(np.abs(Acorr).max()))
        if log:
            log(f"overlap normalization iteration {it + 1}: largest scale correction {history[-1]:.2%}")
        if history[-1] < 1e-4 and it >= WEIGHT_UPDATES:
            break
    # uncertainties of the last scale system solved (the same system again, now with its inverse)
    if gidx is None:
        _, var = solve_scales(n, ref, pairs, last[0], last[1], last[2], want_cov=True)
    else:
        _, var, _, tvar = solve_scales(n, ref, pairs, last[0], last[1], last[2], gidx, last[3], want_cov=True)
    measured = np.zeros(n, int)
    good = (fits["se_k"] < SCALE_SE_MAX).any(1)
    np.add.at(measured, pi, good)
    np.add.at(measured, pj, good)
    lw, cnt = np.zeros(n), np.zeros(n)
    wpair = 0.5 * (w_off.mean(1) + w_scale.mean(1))
    np.add.at(lw, pi, wpair), np.add.at(lw, pj, wpair)
    np.add.at(cnt, pi, 1), np.add.at(cnt, pj, 1)
    floor = own_noise_floor(L, usable, noise, correlation)
    sigma = np.sqrt(var + floor**2 * (np.arange(n) != ref)[:, None])
    sigma[disconnected] = np.nan
    # Gradient matching (applied only when a stack asks for it): one more pair fit at the final
    # a, b gives each pair's residual plane; per-light planes and the offsets consistent with them
    # are solved once, apart from a and b, so the result without the option is unchanged.
    final = fit_pairs(Lr, XY, idx, pairs, a, b, noise, correlation)
    plane, b_plane, plane_pairs = np.zeros((n, 3, 2)), b.copy(), []
    for c in range(3):
        beta_c, gx_c, gy_c, diag = solve_planes(n, ref, pairs, final, w_off, c)
        plane[:, c, 0], plane[:, c, 1] = gx_c, gy_c
        b_plane[:, c] = b[:, c] + beta_c
        plane_pairs.append(diag)
    plane[disconnected], b_plane[disconnected] = 0, b[disconnected]
    extent = np.zeros((n, 3))  # largest plane excursion across each light's usable footprint, ADU
    for k in range(n):
        u = usable[k]
        rows_k, cols_k = torch.nonzero(u.any(1))[:, 0], torch.nonzero(u.any(0))[:, 0]
        if len(rows_k) and len(cols_k):
            ex, ey = float(cols_k.max() - cols_k.min()) / 1000, float(rows_k.max() - rows_k.min()) / 1000
            extent[k] = np.abs(plane[k, :, 0]) * ex + np.abs(plane[k, :, 1]) * ey
    return {
        "a": a,
        "b": b,
        "b_plane": b_plane,
        "plane": plane,
        "plane_extent": extent,
        "plane_pairs": plane_pairs,
        "scale_sigma": sigma,
        "noise": noise,
        "pairs": pairs,
        "fits": fits,
        "group_terms": {
            g: {"factor": np.exp(gcum[k]).tolist(), "sigma": np.sqrt(tvar[k]).tolist()} for k, g in enumerate(glabels)
        },
        "disconnected": disconnected,
        "scale_from_prior": [i for i in range(n) if i != ref and measured[i] == 0],
        "light_weight": lw / np.maximum(cnt, 1),
        "iterations": len(history),
        "max_correction": history,
        "seconds": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- engine entry


def lowres_lights(calibrate, frames, mats, w, h, width, height, edge, device, workspace=None):
    """(n, 3, H8, W8) low-res lights on the union canvas (mats already shifted to it), calibrated
    units. calibrate(i) -> CFA (torch or numpy)."""
    H8, W8 = -(-height // BIN), -(-width // BIN)
    L = (
        workspace.create("lights", (len(frames), 3, H8, W8), np.float32)
        if workspace is not None
        else torch.empty((len(frames), 3, H8, W8), device=device)
    )
    for k, i in enumerate(frames):
        cfa = calibrate(i)
        if not torch.is_tensor(cfa):
            cfa = torch.from_numpy(np.ascontiguousarray(cfa)).to(device)
        L[k] = warp_lowres(superpixels(cfa, edge), mats[i], H8, W8)
        del cfa
    if workspace is not None:
        workspace.flush()
    return L


def build(calibrate, frames, mats, w, h, edge, device, workspace=None):
    """Low-res lights of `frames` on their union canvas. mats: dst->src matrices in reference
    coordinates (unshifted); w/h: one sensor size or per-light sizes (gpucoadd.frame_size).
    Returns L (n, 3, H8, W8) and the canvas (width, height, origin)."""
    from gpucoadd import frame_canvas  # canvas rules shared with stacking

    width, height, origin = frame_canvas(mats, frames, w, h, "union")
    shift = np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1.0]])
    L = lowres_lights(
        calibrate, frames, {i: mats[i] @ shift for i in frames}, w, h, width, height, edge, device, workspace
    )
    return L, (width, height, origin)


def solve(L, frames, ref, prior, init_offset, spread_g, edge, links=(), log=None, groups=None, origin=(0, 0)):
    """Overlap normalization of the low-res lights `L` of `frames` (global indices, ref among them).
    prior / init_offset: {frame: (3,)} photometric scale / offset; spread_g: {frame: (scale, offset)}
    of green, for the provisional star mask. groups: {frame: label} (see normalize).
    origin: the low-res canvas origin in reference-frame pixels (gpucoadd.output_canvas), so the
    gradient-matching planes can be stored per reference pixel ("plane", "plane_origin").
    Returns the cacheable result (JSON-friendly)."""
    t0 = time.perf_counter()
    # per light: isfinite() on the whole stack builds a full-size float temporary (abs) and
    # pushed a 12 GB GPU past its memory on IC 1396
    if stored(L):
        valid = L.owner.create("valid", (L.shape[0], *L.shape[-2:]), np.bool_)
        for k in range(L.shape[0]):
            L.owner.phase("valid pixels", k, L.shape[0])
            valid[k] = torch.isfinite(L[k]).all(0)
    else:
        valid = torch.stack([torch.isfinite(L[k]).all(0) for k in range(L.shape[0])])
    ag = torch.tensor([spread_g[i][0] for i in frames], device=L.device, dtype=torch.float32)
    bg = torch.tensor([spread_g[i][1] for i in frames], device=L.device, dtype=torch.float32)
    stars = star_mask(L, valid, ag, bg)
    if stored(L):
        usable = L.owner.create("usable", valid.shape, np.bool_)
        for k in range(L.shape[0]):
            usable[k] = valid[k] & ~stars
    else:
        usable = valid & ~stars
    pos = {i: k for k, i in enumerate(frames)}
    req = [(pos[i], pos[j]) for i, j in links if i in pos and j in pos]
    res = normalize(
        L,
        usable,
        pos[ref],
        np.stack([prior[i] for i in frames]),
        np.stack([init_offset[i] for i in frames]),
        required=req,
        log=log,
        groups=None if groups is None else [str(groups[i]) for i in frames],
    )
    key = lambda k: str(frames[k])
    finite = lambda v: None if not np.isfinite(v) else float(v)
    return {
        "version": VERSION,
        "method": "overlap",
        "reference": int(ref),
        "frames": [int(i) for i in frames],
        "edge_px": int(edge),
        "bin": BIN,
        "canvas_lowres": [int(L.shape[-1]), int(L.shape[-2])],
        "a": {key(k): res["a"][k].tolist() for k in range(len(frames))},
        "b": {key(k): res["b"][k].tolist() for k in range(len(frames))},
        # gradient matching: offset and plane per light and channel, ADU per reference-frame pixel
        # about plane_origin (the centre of low-res pixel (0, 0)); mean zero over the solved lights
        "b_plane": {key(k): res["b_plane"][k].tolist() for k in range(len(frames))},
        "plane": {key(k): (res["plane"][k] / (1000.0 * BIN)).tolist() for k in range(len(frames))},
        "plane_origin": [float(origin[0]) + BIN / 2, float(origin[1]) + BIN / 2],
        "plane_extent": {key(k): res["plane_extent"][k].tolist() for k in range(len(frames))},
        "plane_gauge": "mean zero over lights with pair information",
        # per channel, ADU per reference pixel: how large the pair planes are, their noise, what the solve leaves
        "plane_pairs": [{k: v / (1000.0 * BIN) for k, v in d.items()} for d in res["plane_pairs"]],
        "scale_sigma": {key(k): [finite(v) for v in res["scale_sigma"][k]] for k in range(len(frames))},
        "weight": {key(k): float(res["light_weight"][k]) for k in range(len(frames))},
        "noise": {key(k): res["noise"][k].tolist() for k in range(len(frames))},
        "disconnected": [frames[k] for k in res["disconnected"]],
        "scale_from_prior": [frames[k] for k in res["scale_from_prior"]],
        **(
            {
                "group_terms": {
                    g: {"factor": t["factor"], "sigma": [finite(v) for v in t["sigma"]]}
                    for g, t in res["group_terms"].items()
                }
            }
            if res["group_terms"]
            else {}
        ),
        "pairs": len(res["pairs"]),
        "iterations": res["iterations"],
        "max_correction": res["max_correction"],
        "pair_fit_backend": (
            pair_kernels()[1]
            if L.is_cuda and L.shape[-1] * L.shape[-2] >= 20000 and len(res["pairs"]) >= 6
            else "torch (CPU or small batch)"
        ),
        "star_mask_fraction": float(stars.float().mean()),
        "solve_s": time.perf_counter() - t0,
    }
