"""Overlap-based brightness normalization: the prototype solver.

Input: lights already on one low-resolution canvas (tools/normalization_metrics.py builds them
from real data; tests/test_overlap_norm.py from a synthetic sky), a usable-pixel mask (stars and
defects excluded), the reference index and a prior scale per light (e.g. 1 / star flux ratio).
Output: per light and channel (a, b) so that a * light + b matches the reference's brightness,
plus the uncertainty of each scale and diagnostics.

Only differences between lights on shared sky are used, so real nebula and stars cancel. For each
overlapping pair, in current normalized units x (light i, noise sx) and y (light j, noise sy):

    d = y - x = k * (m - m0) + delta + plane(X, Y),    m = (sy^2 x + sx^2 y) / (sx^2 + sy^2)

fitted robustly. m is the noise-weighted mean: its noise is uncorrelated with the noise of d, so
k is not biased when one light is much noisier (low gain, haze, another instrument); the plain
mean would bias it. The noise used is each light's effective noise from the pair residuals
(sigma_ij^2 = v_i + v_j), not a pixel-difference estimate: on blurred low-res images that badly
under-states it. k is then corrected for the attenuation by noise in m (signal / total variance). k ~ log(scale_i / scale_j) needs brightness range in the overlap. The plane
is a nuisance term so a sky-gradient difference cannot read as scale; it is never applied.

Network: per light a log-scale shared by the channels (transparency, gain and exposure act on
all channels alike) plus a small per-channel deviation (colour-dependent extinction), and per
channel an offset:
    (alpha_i + g_ic) - (alpha_j + g_jc) = k_ijc         (pairs with a measurable k)
    beta_ic - beta_jc = delta_ijc - (A_ic - A_jc) m0     (A = alpha + g, solved first)
Weighted least squares, the reference fixed; priors pull alpha to the prior scale (PRIOR_SIGMA)
and g to the prior's channel pattern (COLOUR_SIGMA), so a light without measurable scale keeps
its prior; pairs are re-weighted (Huber) so clouds/trails/bad lights lose influence instead of
bending the solution. Repeated, re-fitting pairs in the updated units. Lights in a group not
connected to the reference keep their initial coefficients and are flagged.
"""

import numpy as np
import torch

PRIOR_SIGMA = 0.05  # uncertainty of a prior log-scale (star flux ratio ~ a few %)
COLOUR_SIGMA = 0.02  # allowed per-channel deviation from the shared scale
SCALE_SE_MAX = 0.05  # reporting only: a light with no pair more precise than this has its
# scale "from prior". The solve itself weights every pair by 1/se^2 with
# no cutoff: a hard cutoff let borderline pairs switch in and out between
# iterations and made weak (blue) scales oscillate (SH2-142).
LAM_TAPER = (0.25, 0.5)  # pairs whose regressor is mostly noise (signal share lam below 0.5) fade
# out smoothly, to zero at 0.25: their attenuation correction is unstable
HUBER = 2.5  # residual (robust sigma units) beyond which a pair is down-weighted
MAX_PIXELS = 20000  # per pair fit (random subset)
NOISE_UPDATES = 3  # iterations that re-estimate the effective noise; then frozen, since
# re-estimating it every round makes the solution jitter (~0.5%)
WEIGHT_UPDATES = 5  # iterations that re-decide Huber pair weights; then frozen, since a pair
# switching between down-weighted and full weight made weak (blue)
# scales oscillate between two states (SH2-142, +/-1.8%)
CORRELATION = 2.0  # default standard-error inflation for correlated neighbouring low-res
# pixels (binning, blur, interpolation); to be measured on real data (5c)


def overlapping_pairs(usable, k=4, extra=4, required=(), min_shared=100, seed=0):
    """Pairs to fit: each light's k largest overlaps, `extra` random partners among the lights it
    overlaps by at least half its largest overlap, a maximum spanning tree of the overlap graph
    and the required pairs (e.g. registration links). Largest overlaps alone are not enough:
    consecutive lights point almost identically, so they pick each other and the network splits
    into islands of time-neighbours; random partners and the spanning tree keep it connected
    whenever the lights overlap at all. Returns (pairs, overlap matrix)."""
    n = usable.shape[0]
    flat = usable.reshape(n, -1).float()
    overlap = (flat @ flat.T).cpu().numpy()
    return select_pairs(overlap, k, extra, required, min_shared, seed), overlap


def select_pairs(overlap, k=4, extra=4, required=(), min_shared=100, seed=0):
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
    # maximum spanning tree (Prim) over overlaps >= min_shared
    inside, best, parent = np.zeros(n, bool), np.full(n, -1.0), np.full(n, -1)
    for start in range(n):
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


def pixel_noise(L, usable):
    """(n, 3) robust per-pixel noise of each light (its own units) from horizontal second
    differences L[x-1] - 2 L[x] + L[x+1] on usable pixels: they cancel smooth sky and nebula
    gradients, which first differences would count as noise."""
    d = L[..., :-2] - 2 * L[..., 1:-1] + L[..., 2:]
    ok = usable[..., :-2] & usable[..., 1:-1] & usable[..., 2:]
    out = np.zeros((L.shape[0], 3))
    for k in range(L.shape[0]):
        for c in range(3):
            v = d[k, c][ok[k]]
            out[k, c] = float(1.4826 * (v - v.median()).abs().median() / np.sqrt(6)) if v.numel() > 10 else np.nan
    return out


def own_noise_floor(L, usable, noise, correlation=CORRELATION):
    """(n, 3) smallest log-scale uncertainty a light's own noise allows (against a noiseless
    partner): noise / sqrt(pixels x signal variance), signal variance with a plane removed.
    Every pair a light is in shares this noise, so pairs do not average it down; the network
    solve alone would treat them as independent and under-report the uncertainty."""
    n, _, H, W = L.shape
    YY, XX = torch.meshgrid(
        torch.arange(H, device=L.device, dtype=torch.float64),
        torch.arange(W, device=L.device, dtype=torch.float64),
        indexing="ij",
    )
    out = np.full((n, 3), np.inf)
    for k in range(n):
        u = usable[k]
        if int(u.sum()) < 100:
            continue
        Q = torch.stack([XX[u], YY[u], torch.ones_like(XX[u])], 1)
        for c in range(3):
            v = L[k, c][u].double()
            resid = v - Q @ torch.linalg.lstsq(Q, v[:, None]).solution[:, 0]
            mean = float(v.mean())
            var_sig = max(float(resid.var()) - noise[k, c] ** 2, 1e-30)
            # d(log scale) from the brightness range: noise / (sqrt(N) * std) relative to the level
            out[k, c] = correlation * noise[k, c] / np.sqrt(int(u.sum()) * var_sig) if mean else np.inf
    return out


def fit_pair(x, y, sx, sy, X, Y, seed=0, correlation=CORRELATION):
    """x, y: (3, P) normalized brightness of two lights on shared usable pixels; sx, sy: (3,)
    their noise in those units; X, Y: (P,) canvas coordinates. Per channel: k, se_k, delta, m0,
    robust sigma of residuals, pixels used. The pixel subset depends only on `seed`, so a pair
    is fitted on the same pixels in every iteration (needed for convergence)."""
    P = x.shape[1]
    if P > MAX_PIXELS:
        idx = torch.randperm(P, generator=torch.Generator().manual_seed(seed))[:MAX_PIXELS].to(x.device)
        x, y, X, Y = x[:, idx], y[:, idx], X[idx], Y[idx]
    x, y = x.double(), y.double()
    vx = torch.tensor(np.asarray(sx, float) ** 2, device=x.device)[:, None]
    vy = torch.tensor(np.asarray(sy, float) ** 2, device=x.device)[:, None]
    m = (vy * x + vx * y) / (vx + vy)
    d = y - x
    m0 = m.mean(1, keepdim=True)
    Xc, Yc = (X - X.mean()).double() / 1000, (Y - Y.mean()).double() / 1000
    A = torch.stack([m - m0, Xc.expand_as(m), Yc.expand_as(m), torch.ones_like(m)], -1)  # (3, P, 4)
    use = torch.ones_like(d, dtype=torch.bool)
    for _ in range(4):  # robust: clip at 3 sigma and refit
        w = use.double()[..., None]
        coef = torch.linalg.lstsq(A * w, (d * use.double())[..., None]).solution[..., 0]  # (3, 4)
        r = d - (A @ coef[..., None])[..., 0]
        sig = 1.4826 * torch.stack([r[c][use[c]].abs().median() for c in range(3)])
        use = r.abs() <= 3 * sig[:, None]
    cov = torch.linalg.inv((A * use.double()[..., None]).transpose(1, 2) @ A)
    # Only real sky structure in m tells scales apart: noise in m does not (it is not scaled by
    # the lights' scales). Use the signal part of m's variance, after removing what the plane
    # and offset already explain, so a flat overlap reads as "scale unmeasurable" (se -> inf),
    # not as evidence that the scales are equal.
    var_noise = (vx * vy / (vx + vy))[:, 0]
    Q = A[..., 1:]  # plane + offset columns
    var_tot, var_sig = [], []
    for c in range(3):
        u = use[c]
        mc, Qc = A[c, u, 0], Q[c, u]
        resid = mc - Qc @ torch.linalg.lstsq(Qc, mc[:, None]).solution[:, 0]
        vt = resid.var()
        var_tot.append(vt)
        var_sig.append(torch.clamp(vt - var_noise[c], min=1e-30))
    # Errors in variables: noise in m attenuates the slope by lam = signal / total variance of m
    # (method-of-moments correction); the standard error grows by the same factor.
    lam = torch.clamp(torch.stack(var_sig) / torch.stack(var_tot), min=1e-6)
    se_k = correlation * sig * cov[:, 0, 0].sqrt() / lam
    k = coef[:, 0] / lam
    return {
        "k": k.cpu().numpy(),
        "se_k": se_k.cpu().numpy(),
        "lam": lam.cpu().numpy(),
        "delta": coef[:, 3].cpu().numpy(),
        "m0": m0[:, 0].cpu().numpy(),
        "sigma": sig.cpu().numpy(),
        "n": use.sum(1).cpu().numpy(),
    }


def weighted_lstsq(rows, rhs, wts, ncols, fixed):
    """Solve sum w (row . u - rhs)^2 with the columns in `fixed` held at 0. Returns u and the
    posterior covariance (ncols x ncols, zero rows/columns for fixed unknowns)."""
    keep = [c for c in range(ncols) if c not in fixed]
    A = np.array(rows)[:, keep] * np.sqrt(np.array(wts))[:, None]
    r = np.array(rhs) * np.sqrt(np.array(wts))
    N = A.T @ A
    sol = np.linalg.solve(N, A.T @ r)
    u, cov = np.zeros(ncols), np.zeros((ncols, ncols))
    u[keep] = sol
    cov[np.ix_(keep, keep)] = np.linalg.inv(N)
    return u, cov


def noise_from_pairs(n, fits, current):
    """Per-light effective noise (normalized units) from pair residuals: sigma_ij^2 = v_i + v_j.
    It includes everything that differs between lights besides scale and offset (blurred,
    correlated noise; residual structure), which is what the regression has to account for.
    Solved robustly (median polish: v_i = median over its pairs of sigma_ij^2 - v_j), so a few
    cloudy or otherwise bad pairs cannot shift a clean light's estimate. Lights without pairs
    keep `current`."""
    out = current.copy()
    partners = [[] for _ in range(n)]
    for (i, j), f in fits.items():
        partners[i].append((j, f["sigma"] ** 2))
        partners[j].append((i, f["sigma"] ** 2))
    seen = np.array([len(p) > 0 for p in partners])
    if not seen.any():
        return out
    v = np.zeros((n, 3))
    for k in range(n):
        if partners[k]:
            v[k] = 0.5 * np.median([s for _, s in partners[k]], 0)
    for _ in range(20):
        new = v.copy()
        for k in range(n):
            if partners[k]:
                new[k] = np.median([s - v[j] for j, s in partners[k]], 0)
        floor = 0.05 * np.median(new[seen], 0)
        new[seen] = np.maximum(new[seen], floor)
        if np.allclose(new, v, rtol=1e-4):
            v = new
            break
        v = new
    out[seen] = np.sqrt(v[seen])
    return out


def components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def scale_weight(f, c):
    """Continuous weight of a pair's scale measurement: precision, tapered by signal share."""
    lo, hi = LAM_TAPER
    taper = float(np.clip((f["lam"][c] - lo) / (hi - lo), 0, 1)) ** 2
    se = f["se_k"][c]
    return taper / se**2 if np.isfinite(se) and se > 0 else 0.0


def solve_scales(n, ref, fits, weights, target):
    """Shared log-scale + per-channel deviation corrections. target: (n, 3) prior log-scale
    corrections (log(prior / current)). Returns corrections A (n, 3) and their variance."""
    # unknowns: alpha_0..n-1, then g_(k,c) at n + 3k + c
    ncols = 4 * n
    rows, rhs, wts = [], [], []
    for (i, j), f in fits.items():
        for c in range(3):
            w = weights[(i, j)][c] * scale_weight(f, c)
            if w > 0:
                r = np.zeros(ncols)
                r[i], r[n + 3 * i + c], r[j], r[n + 3 * j + c] = 1, 1, -1, -1
                rows.append(r), rhs.append(f["k"][c]), wts.append(w)
    shared = target.mean(1)
    for k in range(n):
        r = np.zeros(ncols)
        r[k] = 1
        rows.append(r), rhs.append(shared[k]), wts.append(1 / PRIOR_SIGMA**2)
        for c in range(3):
            r = np.zeros(ncols)
            r[n + 3 * k + c] = 1
            rows.append(r), rhs.append(target[k, c] - shared[k]), wts.append(1 / COLOUR_SIGMA**2)
    fixed = {ref, n + 3 * ref, n + 3 * ref + 1, n + 3 * ref + 2}
    u, cov = weighted_lstsq(rows, rhs, wts, ncols, fixed)
    A = u[:n, None] + u[n:].reshape(n, 3)
    k = np.arange(n)[:, None]
    g = n + 3 * k + np.arange(3)[None]  # variance of alpha_k + g_kc, with their covariance
    var = cov[k, k] + cov[g, g] + 2 * cov[k, g]
    return A, var


def solve_offsets(n, ref, fits, weights, A, c):
    rows, rhs, wts = [], [], []
    for (i, j), f in fits.items():
        w = weights[(i, j)][c] * f["n"][c] / f["sigma"][c] ** 2
        if w > 0:
            r = np.zeros(n)
            r[i], r[j] = 1, -1
            rows.append(r), rhs.append(f["delta"][c] - (A[i, c] - A[j, c]) * f["m0"][c]), wts.append(w)
    for k in range(n):  # negligible ridge: keeps lights without offset information at 0
        r = np.zeros(n)
        r[k] = 1
        rows.append(r), rhs.append(0.0), wts.append(1e-12)
    u, _ = weighted_lstsq(rows, rhs, wts, n, {ref})
    return u


def normalize(
    L, usable, ref, prior_scale, init_offset=None, pairs=None, iterations=12, seed=0, correlation=CORRELATION
):
    """L: (n, 3, H, W) lights on the canvas (NaN outside); usable: (n, H, W) bool (inside, not star/defect).
    prior_scale: (n, 3) multiplicative prior (1 for the reference). init_offset: (n, 3) or None (0).
    Returns dict with a, b (n, 3), scale_sigma (n, 3, relative) and diagnostics."""
    n = L.shape[0]
    dev = L.device
    if pairs is None:
        pairs, _ = overlapping_pairs(usable)
    H, W = L.shape[-2:]
    YY, XX = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    noise = pixel_noise(L, usable)  # own units; replaced by pair-residual noise after the first fits
    prior = np.array(prior_scale, float)
    a = prior.copy()
    b = np.zeros((n, 3)) if init_offset is None else np.array(init_offset, float).copy()
    a[ref], b[ref] = 1.0, 0.0
    comp = components(n, pairs)
    disconnected = [i for i in range(n) if comp[i] != comp[ref]]
    w_scale = {p: np.ones(3) for p in pairs}
    w_off = {p: np.ones(3) for p in pairs}
    history = []
    prev_A, flips = np.zeros((n, 3)), np.zeros((n, 3))
    for it in range(iterations):
        at = torch.tensor(a, device=dev, dtype=torch.float32)[:, :, None]
        bt = torch.tensor(b, device=dev, dtype=torch.float32)[:, :, None]
        fits = {}
        for i, j in pairs:
            shared = usable[i] & usable[j]
            x = L[i][:, shared] * at[i] + bt[i]
            y = L[j][:, shared] * at[j] + bt[j]
            fits[(i, j)] = fit_pair(
                x,
                y,
                noise[i] * a[i],
                noise[j] * a[j],
                XX[shared],
                YY[shared],
                int(seed * 100003 + i * 1009 + j),
                correlation,
            )
        # effective noise from the pair residuals, back to each light's own units
        if it < NOISE_UPDATES:
            noise = noise_from_pairs(n, fits, noise * a) / a
        target = np.log(prior / a)
        for _ in range(3 if it < WEIGHT_UPDATES else 1):  # Huber re-weighting of scale pairs
            Acorr, var = solve_scales(n, ref, fits, w_scale, target)
            if it >= WEIGHT_UPDATES:
                break
            res = [
                (p, c, (Acorr[p[0], c] - Acorr[p[1], c] - f["k"][c]) / f["se_k"][c])
                for p, f in fits.items()
                for c in range(3)
                if scale_weight(f, c) > 0
            ]
            s = 1.4826 * np.median([abs(r) for _, _, r in res]) if res else 1.0
            for p, c, r in res:
                w_scale[p][c] = min(1.0, HUBER / max(abs(r) / max(s, 1e-12), 1e-12))
        beta = np.zeros((n, 3))
        for c in range(3):
            for _ in range(3 if it < WEIGHT_UPDATES else 1):  # Huber re-weighting of offset pairs
                bc = solve_offsets(n, ref, fits, w_off, Acorr, c)
                if it >= WEIGHT_UPDATES:
                    break
                res = {
                    p: bc[p[0]] - bc[p[1]] - (f["delta"][c] - (Acorr[p[0], c] - Acorr[p[1], c]) * f["m0"][c])
                    for p, f in fits.items()
                }
                s = 1.4826 * np.median(np.abs(list(res.values()))) if res else 1.0
                for p, r in res.items():
                    w_off[p][c] = min(1.0, HUBER / max(abs(r) / max(s, 1e-12), 1e-12))
            beta[:, c] = bc
        Acorr[disconnected], beta[disconnected] = 0, 0
        # Damping: a scale correction that reverses sign from the last iteration is overshooting
        # (weak, noise-dominated channels, where the attenuation correction can over-correct ~2x
        # and lock into a two-state flip). Halve it each time it reverses; offsets follow.
        flip = (np.sign(Acorr) * np.sign(prev_A) < 0) & (np.abs(Acorr) > 1e-4)
        damp = np.where(flip, 0.5 ** np.minimum(flips + 1, 20), 1.0)
        flips = np.where(flip, flips + 1, flips)
        Acorr *= damp
        beta *= damp
        prev_A = Acorr.copy()
        a = a * np.exp(Acorr)  # new normalized value = exp(A) * (a L + b) + beta
        b = b * np.exp(Acorr) + beta
        a[ref], b[ref] = 1.0, 0.0
        history.append(float(np.abs(Acorr).max()))
        if history[-1] < 1e-4 and it >= WEIGHT_UPDATES:
            break
    measured = np.zeros(n, int)
    for (i, j), f in fits.items():
        good = int((f["se_k"] < SCALE_SE_MAX).any())
        measured[i] += good
        measured[j] += good
    lw, cnt = np.zeros(n), np.zeros(n)
    for p in pairs:
        w = 0.5 * (w_off[p].mean() + w_scale[p].mean())
        lw[list(p)] += w
        cnt[list(p)] += 1
    floor = own_noise_floor(L, usable, noise, correlation)  # with the effective noise
    sigma = np.sqrt(var + floor**2 * (np.arange(n) != ref)[:, None])
    sigma[disconnected] = np.nan
    return {
        "a": a,
        "b": b,
        "scale_sigma": sigma,
        "pairs": pairs,
        "fits": fits,
        "pair_weight": w_off,
        "pair_scale_weight": w_scale,
        "noise": noise,
        "disconnected": disconnected,
        "scale_from_prior": [i for i in range(n) if i != ref and measured[i] == 0],
        "light_weight": lw / np.maximum(cnt, 1),
        "iterations": len(history),
        "max_correction": history,
    }
