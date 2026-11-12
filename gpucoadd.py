"""GPUCoadd's stacking engine for raw colour lights (Seestar S50 and S50 Pro).

GPUBackend does the pixel work on an NVIDIA GPU. CPUBackend runs the same pipeline on the CPU; it
is kept as a reference for tests and comparisons:

  camera offset -> star detection -> registration -> (frame selection) -> normalization ->
  debayer + warp (per output tile) -> rejection + mean

There is no dark, flat or bias calibration: only each light's camera offset is subtracted. In this
code, "calibrate" and "calibrated" mean that offset subtraction and nothing more.

Registration (triangle matching with astroalign, optionally refined to a homography) always
runs on the CPU: it works on a few hundred star positions per frame, so there is nothing for
a GPU to speed up.

Memory: the registered stack (N frames x H x W x 3 float32) is far too big to hold at once
(351 S50 Pro frames = 35 GB), so the output is built in square tiles. For each tile, every
frame is calibrated, debayered and warped again from its raw data, only over the source
pixels that tile needs. Square tiles keep that re-work small even with alt-az field
rotation, where a full-width strip of output maps to a tall slanted band of each source frame.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import tempfile
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

import astroalign as aa
import cv2
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

from fits_input import Layout, read_light
from tileplan import INTEGRATION_BYTES, RESERVE as GPU_RESERVE

# GRBG Bayer, for crops that start on an even row and column
K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
# source pixels per warp_tile call with RCD (lights per call: this over the crop size, at most 32).
# NGC 2175, 2048 px region, 322 lights: 2 M 7.53 s, 4 M 6.69 s, 8 M 6.45 s, 16 M 6.32 s; peak GPU
# memory +0.29 GiB at 8 M over 2 M, well inside tileplan's RCD scratch estimate; pixels identical.
RCD_BATCH_PIXELS = 8_000_000
N_DETECT = 200  # stars kept per frame, brightest first
N_MATCH = 60  # brightest stars used for triangle matching
MAX_FEATHER = 1000  # feather distance limit, px
MIN_LINK_STARS = 10  # fewer stars cannot link a light group to the reference
ANCHOR_TRIES = 3  # group anchor candidates tried before the group is left to neighbour alignment
# Overlap matching steps, in order, as the dialog numbers them. With its data in RAM or on disk
# every step reports; on the GPU only the first and "solving" do.
SKY_STEPS = (
    "low-resolution lights",
    "valid pixels",
    "star consensus",
    "star mask",
    "overlap counts",
    "sample selection",
    "initial noise",
    "pair fits",
    "solving",
    "uncertainty floor",
)
SKY_STEPS_GPU = ("low-resolution lights", "solving")
BG_CELL = 32  # background mesh cell, in half-res pixels
MARGIN = 12  # extra source pixels around each crop: RCD reads 10 px out, bicubic 2 more
WINSOR_ITERS = 5  # winsorized spread re-estimates per clipping pass (converges to ~0.1% by then)
# Pixels from the sensor edge whose demosaiced value depends on (reflected) data beyond it,
# measured by varying that data (tests/test_edges.py), and interpolation taps beyond a sample.
DEMOSAIC_REACH = {"bilinear": 1, "rcd": 10}
INTERP_REACH = {"bilinear": 1, "bicubic": 2}
AP_R = 5  # star aperture radius for photometry, half-res px (10 px at full res)
SAT = 50000  # calibrated ADU above which a star's peak may be non-linear: no photometry


def bayer_masks(h, w):
    """GRBG masks (red, green, blue), float32 0/1, (h, w) each: red at even rows and odd columns,
    blue at odd rows and even columns, green elsewhere. The 2 x 2 cell tiled (full-size
    coordinate grids took about 0.5 s per S50 Pro job)."""
    reps = ((h + 1) // 2, (w + 1) // 2)
    cells = ([[0, 1], [0, 0]], [[1, 0], [0, 1]], [[0, 0], [1, 0]])
    return [np.ascontiguousarray(np.tile(np.array(c, np.float32), reps)[:h, :w]) for c in cells]


def rcd_demosaic(torch, F, cfa, mr, mg, mb):
    """RCD v2.3 demosaic (Luis Sanz Rodriguez, GPL-3.0; see THIRD-PARTY-NOTICES.txt), vectorised with torch so the
    same code runs on GPU or CPU. cfa: (H, W) or a batch (B, H, W), positive, scaled to roughly
    0..1; masks (H, W). Returns (3, H, W) or (B, 3, H, W)."""
    eps, epssq = 1e-5, 1e-10
    H, W = cfa.shape[-2:]
    P = 4

    def nb(a):
        """Neighbour accessor: nb(a)(dy, dx)[y, x] == a[y + dy, x + dx], reflecting at the edges
        (reflection keeps the Bayer phase, since -k and k have the same parity)."""
        ap = F.pad(a.reshape(-1, 1, H, W), (P, P, P, P), mode="reflect").reshape(*a.shape[:-2], H + 2 * P, W + 2 * P)
        return lambda dy, dx: ap[..., P + dy : P + dy + H, P + dx : P + dx + W]

    def disc(d):
        """Use the diagonal neighbourhood's direction where it is more decisive than the centre's."""
        n = nb(d)
        avg = 0.25 * (n(-1, -1) + n(-1, 1) + n(1, -1) + n(1, 1))
        return torch.where((0.5 - d).abs() < (0.5 - avg).abs(), avg, d)

    s = nb(cfa)
    gmask = mg > 0

    # 1. vertical/horizontal discrimination
    bv = (s(-3, 0) - s(-1, 0) - s(1, 0) + s(3, 0) - 3 * (s(-2, 0) + s(2, 0)) + 6 * cfa) ** 2
    bh = (s(0, -3) - s(0, -1) - s(0, 1) + s(0, 3) - 3 * (s(0, -2) + s(0, 2)) + 6 * cfa) ** 2
    v, h = nb(bv), nb(bh)
    vs = (v(-1, 0) + bv + v(1, 0)).clamp_min(epssq)
    hs = (h(0, -1) + bh + h(0, 1)).clamp_min(epssq)
    vh_disc = disc(vs / (vs + hs))

    # 2. low pass filter
    lpf = cfa + 0.5 * (s(-1, 0) + s(1, 0) + s(0, -1) + s(0, 1)) + 0.25 * (s(-1, -1) + s(-1, 1) + s(1, -1) + s(1, 1))
    L = nb(lpf)

    def ratio(dy, dx):
        l2 = L(2 * dy, 2 * dx)
        return 1 + (lpf - l2) / (eps + lpf + l2)

    # 3. green at red and blue sites
    ng = (
        eps
        + (s(-1, 0) - s(1, 0)).abs()
        + (cfa - s(-2, 0)).abs()
        + (s(-1, 0) - s(-3, 0)).abs()
        + (s(-2, 0) - s(-4, 0)).abs()
    )
    sg = (
        eps + (s(-1, 0) - s(1, 0)).abs() + (cfa - s(2, 0)).abs() + (s(1, 0) - s(3, 0)).abs() + (s(2, 0) - s(4, 0)).abs()
    )
    wg = (
        eps
        + (s(0, -1) - s(0, 1)).abs()
        + (cfa - s(0, -2)).abs()
        + (s(0, -1) - s(0, -3)).abs()
        + (s(0, -2) - s(0, -4)).abs()
    )
    eg = (
        eps + (s(0, -1) - s(0, 1)).abs() + (cfa - s(0, 2)).abs() + (s(0, 1) - s(0, 3)).abs() + (s(0, 2) - s(0, 4)).abs()
    )
    v_est = (sg * s(-1, 0) * ratio(-1, 0) + ng * s(1, 0) * ratio(1, 0)) / (ng + sg)
    h_est = (wg * s(0, 1) * ratio(0, 1) + eg * s(0, -1) * ratio(0, -1)) / (eg + wg)
    g = torch.where(gmask, cfa, vh_disc * (h_est - v_est) + v_est)
    G = nb(g)

    # 4.1 diagonal discrimination
    pd = (s(-3, -3) - s(-1, -1) - s(1, 1) + s(3, 3) - 3 * (s(-2, -2) + s(2, 2)) + 6 * cfa) ** 2
    qd = (s(-3, 3) - s(-1, 1) - s(1, -1) + s(3, -3) - 3 * (s(-2, 2) + s(2, -2)) + 6 * cfa) ** 2
    p, q = nb(pd), nb(qd)
    ps = (p(-1, -1) + pd + p(1, 1)).clamp_min(epssq)
    qs = (q(-1, 1) + qd + q(1, -1)).clamp_min(epssq)
    pq_disc = disc(ps / (ps + qs))

    # 4.2 red at blue sites and blue at red sites (their diagonal neighbours are the other colour)
    nwg = eps + (s(-1, -1) - s(1, 1)).abs() + (s(-1, -1) - s(-3, -3)).abs() + (g - G(-2, -2)).abs()
    neg = eps + (s(-1, 1) - s(1, -1)).abs() + (s(-1, 1) - s(-3, 3)).abs() + (g - G(-2, 2)).abs()
    swg = eps + (s(-1, 1) - s(1, -1)).abs() + (s(1, -1) - s(3, -3)).abs() + (g - G(2, -2)).abs()
    seg = eps + (s(-1, -1) - s(1, 1)).abs() + (s(1, 1) - s(3, 3)).abs() + (g - G(2, 2)).abs()
    p_est = (nwg * (s(1, 1) - G(1, 1)) + seg * (s(-1, -1) - G(-1, -1))) / (nwg + seg)
    q_est = (neg * (s(1, -1) - G(1, -1)) + swg * (s(-1, 1) - G(-1, 1))) / (neg + swg)
    other = g + pq_disc * (q_est - p_est) + p_est
    zero = torch.zeros_like(cfa)
    r = torch.where(mr > 0, cfa, torch.where(mb > 0, other, zero))
    b = torch.where(mb > 0, cfa, torch.where(mr > 0, other, zero))

    # 4.3 red and blue at green sites (their cardinal neighbours now have both)
    rb = []
    for ch in (r, b):
        C = nb(ch)
        ng = eps + (g - G(-2, 0)).abs() + (C(-1, 0) - C(1, 0)).abs() + (C(-1, 0) - C(-3, 0)).abs()
        sg = eps + (g - G(2, 0)).abs() + (C(1, 0) - C(-1, 0)).abs() + (C(1, 0) - C(3, 0)).abs()
        wg = eps + (g - G(0, -2)).abs() + (C(0, -1) - C(0, 1)).abs() + (C(0, -1) - C(0, -3)).abs()
        eg = eps + (g - G(0, 2)).abs() + (C(0, 1) - C(0, -1)).abs() + (C(0, 1) - C(0, 3)).abs()
        v_est = (ng * (C(1, 0) - G(1, 0)) + sg * (C(-1, 0) - G(-1, 0))) / (ng + sg)
        h_est = (eg * (C(0, -1) - G(0, -1)) + wg * (C(0, 1) - G(0, 1))) / (eg + wg)
        rb.append(torch.where(gmask, g + vh_disc * (h_est - v_est) + v_est, ch))
    return torch.stack([rb[0], g, rb[1]], dim=-3)


def gpu_sort(max_frames):
    """Warp/register sort through 512 frames; torch above that or if compilation fails."""
    if os.environ.get("GPUCOADD_SORT") == "torch":
        return None, "torch (GPUCOADD_SORT)"
    try:
        import sort_kernels

        sort_kernels.prepare(max_frames)
        return sort_kernels, "warp <=512 / torch >512"
    except Exception as exc:
        return None, f"torch (sort unavailable: {type(exc).__name__}: {exc})"


def gpu_clip():
    """(kernels, description) for rejection on the GPU: integrate_kernels (one CUDA kernel per
    mode, 5-12x faster than the torch mask-and-sum passes; sigma and winsorized decisions identical,
    noise-aware identical but for samples within float rounding of a threshold) when they can be
    built here, else None (torch). GPUCOADD_CLIP=torch forces the torch path."""
    if os.environ.get("GPUCOADD_CLIP") == "torch":
        return None, "torch (GPUCOADD_CLIP)"
    try:
        import integrate_kernels

        integrate_kernels.module()  # compiled once, then loaded from the disk cache
        return integrate_kernels, "cuda kernels"
    except Exception as exc:
        return None, f"torch (kernels unavailable: {type(exc).__name__}: {exc})"


def gpu_warp():
    """(kernels, description) for the warp: warp_kernels (coordinates, taps, edge mask, brightness
    and scatter in one kernel, the same arithmetic as warp_kernels.warp_torch, ~3x faster) when they
    can be built here, else None (warp_torch). GPUCOADD_WARP=torch forces the torch path."""
    if os.environ.get("GPUCOADD_WARP") == "torch":
        return None, "torch (GPUCOADD_WARP)"
    try:
        import warp_kernels

        warp_kernels.module()
        return warp_kernels, "cuda kernel"
    except Exception as exc:
        return None, f"torch (kernel unavailable: {type(exc).__name__}: {exc})"


def gpu_calibrate():
    """Fused tile calibration/padding, or the original torch path when unavailable."""
    if os.environ.get("GPUCOADD_CALIBRATE") == "torch":
        return None, "torch (GPUCOADD_CALIBRATE)"
    try:
        import calibrate_kernels

        calibrate_kernels.module()
        return calibrate_kernels, "cuda kernel"
    except Exception as exc:
        return None, f"torch (kernel unavailable: {type(exc).__name__}: {exc})"


def gpu_rcd():
    """Five-stage CUDA RCD, then Jiterator, then torch if compilation is unavailable.

    GPUCOADD_RCD=jiterator selects the previous fused implementation;
    GPUCOADD_RCD=reference selects the original torch implementation.
    """
    if os.environ.get("GPUCOADD_RCD") == "reference":
        return rcd_demosaic, "reference (GPUCOADD_RCD)"
    staged_error = ""
    if os.environ.get("GPUCOADD_RCD") != "jiterator":
        try:
            from rcd_kernels import module, rcd_demosaic_staged

            module()  # Compile/load before loading the lights; fallback is explicit in the log.
            return rcd_demosaic_staged, "staged CUDA"
        except Exception as exc:
            staged_error = f" (staged CUDA unavailable: {type(exc).__name__}: {exc})"
    try:
        from rcd_fused import _kernels, rcd_demosaic_fused

        _kernels()  # compiled once ever (~20 s), later loaded from torch's disk cache (~0.1 s)
        return rcd_demosaic_fused, "fused" + staged_error
    except Exception as exc:  # no NVRTC, unsupported GPU, ...: the reference works everywhere
        return rcd_demosaic, f"reference (fused kernels unavailable: {type(exc).__name__}: {exc})" + staged_error


class Timer:
    def __init__(self, sync=None):
        self.t = defaultdict(float)
        self.sync = sync or (lambda: None)

    @contextmanager
    def __call__(self, name):
        self.sync()
        t0 = time.perf_counter()
        yield
        self.sync()
        self.t[name] += time.perf_counter() - t0


def frame_size(w, h, i):
    """(width, height) of light i: w/h are either one size for all lights (single sensor) or
    per-light mappings/sequences (lights of different sensors in one stack)."""
    if isinstance(w, (int, np.integer)):
        return int(w), int(h)
    return int(w[i]), int(h[i])


def per_light(values, n, name):
    """A per-loaded-light list from one value or a sequence of n values."""
    if np.ndim(values) == 0:
        return [float(values)] * n
    values = [float(v) for v in values]
    if len(values) != n:
        raise ValueError(f"Expected {n} {name} values, got {len(values)}")
    return values


def sensor_edge(debayer, interp):
    """Warped samples closer than this to a light's sensor edge are discarded: their
    interpolation taps would read demosaiced pixels that depend on data beyond the sensor,
    which are unreliable and, on the GPU, also depend on how crops are batched."""
    return DEMOSAIC_REACH[debayer] + INTERP_REACH[interp]


def tile_source_box(minv, x0, x1, y0, y1, w, h):
    """Source crop (a, b, c, d) = rows [a, b), cols [c, d) that output tile [y0, y1) x [x0, x1)
    needs, with a and c even. None if the tile is outside the frame. minv may be projective;
    a homography maps the tile's edges to straight lines, so the corners bound it."""
    corners = np.array([[x0, y0, 1], [x1 - 1, y0, 1], [x0, y1 - 1, 1], [x1 - 1, y1 - 1, 1]], np.float64).T
    p = minv @ corners
    sx, sy = p[0] / p[2], p[1] / p[2]
    a = max(0, int(np.floor(sy.min())) - MARGIN) & ~1
    b = min(h, int(np.ceil(sy.max())) + MARGIN + 1)
    c = max(0, int(np.floor(sx.min())) - MARGIN) & ~1
    d = min(w, int(np.ceil(sx.max())) + MARGIN + 1)
    if b - a < 16 or d - c < 16:
        return None
    return a, b, c, d


def norm_coeffs(stats, ref, flux_ratio=None, rgb_equal=False, green_ratio=None):
    """Per-channel (scale, offset) onto the reference frame. The background median is always
    matched. Scale: with flux_ratio (star flux vs the reference, per channel), stars come out
    equally bright ("photometric"); without it, the background spread (MAD) is matched, which
    follows additive+scaling normalization.
    Spread is mostly noise, so that equalizes noise instead of signal: hazy frames end up
    with a dimmer nebula. rgb_equal matches every channel to the reference's green median
    and spread; only supported for spread normalization. This changes colour, not calibration.
    green_ratio: the reference's green spread over each channel's spread, as measured on
    demosaiced data (rgb_spread_ratio); without it, the ratio of the sampled CFA MADs."""
    med, mad = stats[:2]
    rmed, rmad = ref[:2]
    if rgb_equal:
        if flux_ratio is not None:
            raise ValueError("RGB equalization requires Background spread normalization")
        if not (
            np.isfinite(med).all()
            and np.isfinite(mad).all()
            and (mad > 0).all()
            and np.isfinite(rmed[1])
            and np.isfinite(rmad[1])
            and rmad[1] > 0
        ):
            raise ValueError("RGB equalization needs finite background levels and positive channel spreads")
        if green_ratio is None:
            rmed, rmad = np.full_like(rmed, rmed[1]), np.full_like(rmad, rmad[1])
        else:
            green_ratio = np.asarray(green_ratio, np.float32)
            if not (np.isfinite(green_ratio).all() and (green_ratio > 0).all()):
                raise ValueError("RGB equalization needs finite background levels and positive channel spreads")
            # each channel onto the same reference channel, then the reference's colour ratio:
            # the green scale stays exactly what it is without RGB equalization
            a = (rmad / mad * green_ratio).astype(np.float32)
            return a, (rmed[1] - med * a).astype(np.float32)
    a = rmad / mad if flux_ratio is None or not np.all(np.isfinite(flux_ratio)) else 1 / flux_ratio
    return a.astype(np.float32), (rmed - med * a).astype(np.float32)


def ikss(x):
    """IKSS location and scale of a sample: clip to median +- 4 sigma, sigma being the
    square root of the biweight midvariance, until sigma changes by less than 1e-5 (relative)."""
    x = np.sort(np.asarray(x, np.float64).ravel())
    lo, hi, s0 = 0, len(x), np.inf
    while hi > lo:
        v = x[lo:hi]
        m = 0.5 * (v[(len(v) - 1) // 2] + v[len(v) // 2])
        mad = np.median(np.abs(v - m))
        if not mad > 0:
            return m, 0.0
        u = (v - m) / (9 * mad)
        k = np.abs(u) < 1
        u2 = u[k] ** 2
        s = np.sqrt(len(v) * np.sum((v[k] - m) ** 2 * (1 - u2) ** 4) / np.sum((1 - u2) * (1 - 5 * u2)) ** 2)
        if (s0 - s) / s < 1e-5:
            return m, 0.991 * s
        s0 = s
        lo = max(lo, int(np.searchsorted(x, m - 4 * s, "left")))
        hi = min(hi, int(np.searchsorted(x, m + 4 * s, "right")))
    return float("nan"), float("nan")


def rgb_spread_ratio(backend, i):
    """Green over red/green/blue IKSS spread of light i, measured after demosaicing.
    Demosaicing averages red and blue samples, so their spread on the raw CFA
    samples is larger (IC 63, RCD: green/red 0.59 on the CFA, 0.653 demosaiced)."""
    s = np.array([ikss(c)[1] for c in backend.demosaiced_samples(i)])
    ratio = s[1] / s
    ratio[1] = 1.0
    return ratio


def strips(h, w, budget=2e6, margin=16):
    """Row strips for demosaicing a whole light within a pixel budget: (a, b) rows read, which
    reach `margin` rows past the kept rows (ka, kb) so the strip edge cannot change them.
    Strip starts are even, keeping the Bayer phase."""
    rows = max(2, (int(budget // w) - 2 * margin) // 2 * 2)
    for ka in range(0, h, rows):
        kb = min(h, ka + rows)
        yield max(0, ka - margin), min(h, kb + margin), ka, kb


def star_fwhm(win, ox, oy, tot, cx, cy, xp):
    """Frame FWHM in full-res px: median second-moment size of the detected stars, skipping the
    brightest few (saturated cores read wide). Only used to rank frames against each other."""
    mxx = (win * ox * ox).sum((1, 2)) / tot - cx**2
    myy = (win * oy * oy).sum((1, 2)) / tot - cy**2
    fw = 2.3548 * 2 * xp.sqrt(((mxx + myy) / 2).clip(min=0))
    return float(xp.median(fw[5:N_MATCH])) if len(fw) > 10 else float("nan")


def light_background(cfa):
    """Sky level of a calibrated light: median of every fourth green pixel (GRBG: green at the
    first pixel). Only compared between lights of the same sky to choose the reference."""
    s = cfa[0::8, 0::8]
    return float(np.median(s)) if isinstance(s, np.ndarray) else float(s.median())


def star_transparency(stars):
    """Summed green aperture flux of the detected stars ranked 6th to 60th by brightness: lower
    through haze or cloud, and lower again when fewer stars are found. The brightest five are
    skipped (saturated cores). Only compared between lights of the same sky."""
    return float(np.sum(stars[5:N_MATCH, 3])) if len(stars) else 0.0


REFERENCE_CANDIDATES = 10  # at least this many lights (or 5% of them) are weighed for the reference
REFERENCE_BACKGROUND = 1.1  # the reference's sky may be at most this much above the candidates' median
REFERENCE_TRANSPARENCY = 0.9  # and its star flux at least this share of their median
REFERENCE_SKY = 1.5  # and its sky at most this much above the median of all lights (a twilight panel)


def choose_reference(pool, fwhm, stars, background, pointings=None, field_deg=None):
    """The automatic reference light, from measurements (GPUCOADD_REFERENCE=session restores the earlier
    choice by session order, for comparisons).

    1. Sky position: with pointings and a known field size, the candidates are the lights within a
       quarter of the field's short side of the light nearest the centre of the covered sky,
       widened to the nearest REFERENCE_CANDIDATES lights (or 5%) when fewer are that close: the
       central part of a mosaic, every light of a single field.
    2. Quality limits, against the candidates' median: sky background at most
       REFERENCE_BACKGROUND times it (no twilight, moonlight or light-pollution peaks), star flux at
       least REFERENCE_TRANSPARENCY times it (no haze or cloud), and a measured FWHM. The sky must
       also stay within REFERENCE_SKY times the median of all lights, so a central panel taken under
       a bright sky does not pass on its own median (nebula raises a panel's level far less). When
       no candidate meets them, every light of the pool is weighed the same way.
    3. The sharpest of those; ties go to the light nearest the centre, then nearest mid-session.
    Returns (index, a summary of the choice for the log and the report)."""
    pool = [int(i) for i in pool]
    fw = np.array([fwhm[i] if np.isfinite(fwhm[i]) else np.inf for i in pool])
    flux_all = np.array([star_transparency(stars[i]) for i in pool])
    bg_all = np.array([background.get(i, np.nan) for i in pool], dtype=float)
    dist = np.zeros(len(pool))
    choice = {"rule": "measured", "pool": len(pool)}
    if pointings is not None and field_deg:
        sky = np.array([pointings[i] for i in pool], dtype=float)
        if np.isfinite(sky).all():
            ra, dec = np.deg2rad(sky).T
            v = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)
            c = v.sum(0)
            c /= np.linalg.norm(c)
            east = np.cross([0.0, 0.0, 1.0], c)
            east = east / np.linalg.norm(east) if np.linalg.norm(east) > 1e-9 else np.array([1.0, 0.0, 0.0])
            north = np.cross(c, east)
            along = v @ c
            x, y = np.rad2deg((v @ east) / along), np.rad2deg((v @ north) / along)  # gnomonic, degrees
            xs, ys = np.percentile(x, [2, 98]), np.percentile(y, [2, 98])  # extent, ignoring strays
            dist = np.hypot(x - xs.mean(), y - ys.mean())
            choice.update(sky_extent_deg=[round(float(xs[1] - xs[0]), 3), round(float(ys[1] - ys[0]), 3)],
                          field_deg=round(float(field_deg), 3))
    sky = np.nanmedian(bg_all) if np.isfinite(bg_all).any() else np.inf
    need = min(len(pool), max(REFERENCE_CANDIDATES, int(np.ceil(0.05 * len(pool)))))
    radius = max(dist.min() + 0.25 * field_deg if field_deg else np.inf, np.sort(dist)[need - 1])

    def meets(cand):
        flux, bg = flux_all[cand], bg_all[cand]
        ok = np.isfinite(fw[cand]) & (flux >= REFERENCE_TRANSPARENCY * np.median(flux))
        if np.isfinite(bg).any():
            bg = np.nan_to_num(bg, nan=np.inf)
            ok &= (bg <= REFERENCE_BACKGROUND * np.nanmedian(bg_all[cand])) & (bg <= REFERENCE_SKY * sky)
        return ok

    cand = np.flatnonzero(dist <= radius)
    ok = meets(cand)
    widened = not ok.any()
    if widened:
        cand = np.arange(len(pool))
        ok = meets(cand)
    if not ok.any():  # nothing meets the limits: the sharpest measured light
        ok = np.isfinite(fw[cand]) if np.isfinite(fw[cand]).any() else np.ones(len(cand), bool)
    keep = cand[ok]
    middle = len(pool) // 2
    best = int(min(keep, key=lambda k: (fw[k], dist[k], abs(k - middle))))
    i = pool[best]
    med_flux = np.median(flux_all[cand])
    med_bg = np.nanmedian(bg_all[cand]) if np.isfinite(bg_all[cand]).any() else None
    med_fw = np.median(fw[cand][np.isfinite(fw[cand])]) if np.isfinite(fw[cand]).any() else None
    choice.update(
        index=i,
        candidates=len(cand),
        widened_to_all=widened,
        qualified=len(keep),
        fwhm_px=round(float(fw[best]), 3),
        fwhm_median_px=None if med_fw is None else round(float(med_fw), 3),
        transparency_vs_median=round(float(flux_all[best] / med_flux), 3) if med_flux > 0 else None,
        background_vs_median=None if not med_bg else round(float(bg_all[best] / med_bg), 3),
        centre_offset_deg=round(float(dist[best]), 3),
    )
    choice["summary"] = (
        f"light #{i + 1} of {len(pool)}: {len(cand)} candidates "
        f"{'(all lights: none near the centre met the limits)' if widened else 'near the centre of the covered sky'}, "
        f"{len(keep)} within the background and star-flux limits; FWHM {choice['fwhm_px']} px "
        f"(median {choice['fwhm_median_px']}), star flux x{choice['transparency_vs_median']}, "
        f"background x{choice['background_vs_median']} of the candidates' median, "
        f"{choice['centre_offset_deg']} deg from the centre"
    )
    return i, choice


# --------------------------------------------------------------------------- CPU


class CPUBackend:
    name = "cpu"

    def __init__(self, raws, pedestal, threads, interp="bilinear", debayer="bilinear"):
        """raws: one array per light (sizes may differ between lights); pedestal: one value or
        one per light."""
        self.raws = raws
        self.interp = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}[interp]
        self.debayer = debayer
        self.edge = sensor_edge(debayer, interp)
        self.n = len(raws)
        self.pedestals = per_light(pedestal, self.n, "pedestal")
        self.h, self.w = raws[0].shape
        self.sizes = {i: (r.shape[1], r.shape[0]) for i, r in enumerate(raws)}  # (width, height)
        self.threads = threads
        self.timer = Timer()
        cv2.setNumThreads(threads)
        # No flat correction yet: an array of ones per sensor size keeps the multiply (and its cost)
        # where a master flat will go. The pedestal is the camera offset, not a dark frame.
        self.flat_inv = {shape: np.ones(shape, np.float32) for shape in {r.shape for r in raws}}
        hmax, wmax = max(r.shape[0] for r in raws), max(r.shape[1] for r in raws)
        self.masks = bayer_masks(hmax, wmax)  # GRBG is periodic: one mask fits every sensor
        self.pool = ThreadPoolExecutor(threads)
        if debayer == "rcd":  # RCD is written once, in torch; on the CPU it runs on torch's CPU threads
            import torch
            import torch.nn.functional as F

            torch.set_num_threads(threads)
            self.torch, self.F = torch, F
            self.tmasks = [torch.from_numpy(m) for m in self.masks]

    def calibrate(self, i, a=0, b=None, c=0, d=None):
        raw = self.raws[i]
        b = raw.shape[0] if b is None else b
        d = raw.shape[1] if d is None else d
        return (raw[a:b, c:d].astype(np.float32) - np.float32(self.pedestals[i])) * self.flat_inv[raw.shape][a:b, c:d]

    def detect(self, cfa):
        lum = cfa[0::2, 0::2] + cfa[0::2, 1::2] + cfa[1::2, 0::2] + cfa[1::2, 1::2]
        hh, ww = lum.shape
        small = cv2.resize(lum, (ww // BG_CELL, hh // BG_CELL), interpolation=cv2.INTER_AREA)
        d = lum - cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)
        sm = cv2.GaussianBlur(d, (9, 9), 1.5)
        samp = sm[::4, ::4]
        med = np.median(samp)
        sig = 1.4826 * np.median(np.abs(samp - med))
        peak = (sm == cv2.dilate(sm, np.ones((7, 7), np.uint8))) & (sm > med + 5 * sig)
        peak[:8], peak[-8:], peak[:, :8], peak[:, -8:] = False, False, False, False
        ys, xs = np.nonzero(peak)
        top = np.argsort(sm[ys, xs])[::-1][:N_DETECT]
        ys, xs = ys[top], xs[top]
        oy, ox = np.mgrid[-3:4, -3:4]
        win = np.clip(d[ys[:, None, None] + oy, xs[:, None, None] + ox], 0, None)
        tot = win.sum((1, 2)) + 1e-9
        cy = (win * oy).sum((1, 2)) / tot
        cx = (win * ox).sum((1, 2)) / tot
        fwhm = star_fwhm(win, ox, oy, tot, cx, cy, np)
        xy = np.stack([xs + cx, ys + cy], 1) * 2 + 0.5  # half-res -> full-res pixel coords
        return np.c_[xy, self.photometry(cfa, ys, xs)], fwhm

    def photometry(self, cfa, ys, xs):
        """Per star: background-subtracted aperture flux in R, G, B (half-res planes) and the
        peak calibrated value in its 4x4 full-res block. Returns (n, 4)."""
        oy, ox = np.mgrid[-AP_R : AP_R + 1, -AP_R : AP_R + 1]
        ap = (ox**2 + oy**2 <= AP_R**2).astype(np.float32)
        cols = []
        for pl in (cfa[0::2, 1::2], 0.5 * (cfa[0::2, 0::2] + cfa[1::2, 1::2]), cfa[1::2, 0::2]):
            hh, ww = pl.shape
            small = cv2.resize(pl, (ww // BG_CELL, hh // BG_CELL), interpolation=cv2.INTER_AREA)
            d = pl - cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)
            cols.append((d[ys[:, None, None] + oy, xs[:, None, None] + ox] * ap).sum((1, 2)))
        fy, fx = np.mgrid[-1:3, -1:3]
        cols.append(cfa[2 * ys[:, None, None] + fy, 2 * xs[:, None, None] + fx].max((1, 2)))
        return np.stack(cols, 1)

    def stats(self, cfa):
        chans = [cfa[0::2, 1::2], cfa[0::2, 0::2], cfa[1::2, 0::2]]  # R, G, B
        med, mad = [], []
        for c in chans:
            s = c[::4, ::4]
            m = np.median(s)
            med.append(m)
            mad.append(np.median(np.abs(s - m)))
        # pixel-scale noise per channel: differences of neighbouring same-colour pixels barely see
        # stars or gradients; every other row of each colour (R and G on even rows, B on odd rows)
        rgb = []
        for c in (cfa[0::4, 1::2], cfa[0::4, 0::2], cfa[1::4, 0::2]):
            dc = c[:, 1:] - c[:, :-1]
            rgb.append(1.4826 * np.median(np.abs(dc - np.median(dc))) / np.sqrt(2))
        rgb = np.array(rgb, np.float32)
        return np.array(med, np.float32), np.array(mad, np.float32), float(rgb[1]), rgb

    def demosaic(self, cal, ped):
        """Calibrated CFA crop starting on an even row and column -> (h, w, 3)."""
        h, w = cal.shape
        if self.debayer == "rcd":
            torch = self.torch
            x = (torch.from_numpy(cal) + ped) / 65535
            ms = [m[:h, :w] for m in self.tmasks]
            return (rcd_demosaic(torch, self.F, x, *ms) * 65535 - ped).permute(1, 2, 0).contiguous().numpy()
        mr, mg, mb = (m[:h, :w] for m in self.masks)
        bt = cv2.BORDER_REFLECT_101
        return cv2.merge(
            [
                cv2.filter2D(cal * mr, -1, K_RB, borderType=bt),
                cv2.filter2D(cal * mg, -1, K_G, borderType=bt),
                cv2.filter2D(cal * mb, -1, K_RB, borderType=bt),
            ]
        )

    def demosaiced_samples(self, i, step=3, budget=2e6):
        """Every step-th demosaiced pixel of light i in both directions (odd steps sample all
        Bayer phases alike), per channel: (3, N) float64. Demosaiced in row strips."""
        h, w = self.raws[i].shape
        out = []
        for a, b, ka, kb in strips(h, w, budget):
            rgb = self.demosaic(self.calibrate(i, a, b), self.pedestals[i])
            out.append(rgb[ka - a + (-ka) % step : kb - a : step, ::step].reshape(-1, 3))
        return np.concatenate(out).T.astype(np.float64)

    def alloc_stack(self, n, rows, cols):
        return np.empty((n, rows, cols, 3), np.float32)

    def alloc_feather(self, n, rows, cols):
        return np.empty((n, rows, cols), np.float32)

    def feather_tile(self, out, frames, mats, x0, y0, radius):
        import feather_kernels

        feather_kernels.feather_numpy(
            out,
            list(range(len(frames))),
            np.stack([mats[i] for i in frames]),
            np.array([self.sizes[i] for i in frames], np.float64),
            self.edge,
            x0,
            y0,
            radius,
        )

    def warp_tile(self, stack, frames, mats, coefs, x0, x1, y0, y1):
        for j, i in enumerate(frames):
            self.warp_into(stack, j, i, mats[i], x0, x1, y0, y1, coefs[i])

    def warp_into(self, stack, j, i, minv, x0, x1, y0, y1, coef):
        T = self.timer
        w, h = self.sizes[i]
        box = tile_source_box(minv, x0, x1, y0, y1, w, h)
        if box is None:
            stack[j] = np.nan
            return
        a, b, c, d = box
        with T("calibrate"):
            cal = self.calibrate(i, a, b, c, d)
        with T("debayer"):
            rgb = self.demosaic(cal, self.pedestals[i])
        with T("warp"):
            # tile pixel -> full output -> source -> source crop
            m = np.array([[1, 0, -c], [0, 1, -a], [0, 0, 1]]) @ minv @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]])
            kw = dict(
                flags=self.interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan,) * 3
            )
            if np.allclose(m[2], [0, 0, 1]):
                stack[j] = cv2.warpAffine(rgb, m[:2], (x1 - x0, y1 - y0), **kw)
            else:
                stack[j] = cv2.warpPerspective(rgb, m, (x1 - x0, y1 - y0), **kw)
            # discard samples too close to the sensor edge (see sensor_edge)
            gy, gx = np.mgrid[y0:y1, x0:x1]
            p = minv @ np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
            sx, sy = (p[0] / p[2]).reshape(gx.shape), (p[1] / p[2]).reshape(gx.shape)
            e = self.edge
            stack[j][~((sx >= e) & (sx <= w - 1 - e) & (sy >= e) & (sy <= h - 1 - e))] = np.nan
        with T("normalize"):
            s, o = coef[:2]
            stack[j] *= s
            stack[j] += o
            if len(coef) > 2:  # gradient matching: the light's plane over the output canvas
                g = np.asarray(coef[2], np.float64)
                stack[j] += (gx[..., None] * g[None, None, :, 0] + gy[..., None] * g[None, None, :, 1]).astype(
                    np.float32
                )

    def integrate(self, stack, kappa, iters, weights=None, rejection="sigma", noise=None, feather=None):
        """rejection: "sigma" | "winsorized" | "noise" (noise: (frames, 3) per-light noise in
        normalized units, required for "noise"); see sigma_clip_sorted_np / noise_clip_torch.
        feather: (frames, rows, cols) per-sample weights multiplied into the light weights."""
        n = stack.shape[0]
        flat = stack.reshape(n, -1)
        fw = None if feather is None else feather.reshape(n, -1)  # (n, rows, cols, 3) layout: pixel = column // 3

        def sample_weights(m0, m1):  # (n, m1 - m0) per-sample weights, or None
            if fw is None:
                return None
            f = fw[:, np.arange(m0, m1) // 3]
            if weights is None:
                return f
            return f * (weights[:, np.arange(m0, m1) % 3] if weights.ndim == 2 else weights[:, None])

        out = np.empty(flat.shape[1], np.float32)
        chunk = max(2048, (1 << 20) // n)  # ~1M values per chunk keeps it cache friendly
        if rejection == "noise":
            import torch

            chan = np.arange(flat.shape[1]) % 3  # (n, rows, cols, 3) layout
            nz = torch.from_numpy(np.asarray(noise, np.float32))
            wt = None if weights is None else torch.from_numpy(weights.astype(np.float32))

        def work(m0):
            sw = sample_weights(m0, min(m0 + chunk, flat.shape[1]))
            if rejection == "noise":
                x = torch.from_numpy(np.ascontiguousarray(flat[:, m0 : m0 + chunk]))
                srt = torch.sort(x, dim=0)  # NaN sort last
                sig = nz[:, torch.from_numpy(chan[m0 : m0 + chunk])]
                if sw is not None:
                    ws = torch.gather(torch.from_numpy(np.ascontiguousarray(sw, np.float32)), 0, srt.indices)
                elif wt is None:
                    ws = None
                elif wt.ndim == 2:  # per-channel weights: (n, 3) indexed by each column's channel
                    ws = wt[srt.indices, torch.from_numpy(chan[m0 : m0 + chunk])]
                else:
                    ws = wt[srt.indices]
                out[m0 : m0 + chunk] = noise_clip_torch(
                    torch, srt.values, torch.gather(sig, 0, srt.indices), kappa, iters, ws
                ).numpy()
                return
            v = np.ascontiguousarray(flat[:, m0 : m0 + chunk].T)
            win = rejection == "winsorized"
            if weights is None and sw is None:
                out[m0 : m0 + chunk] = sigma_clip_sorted_np(
                    np.sort(v, axis=1), kappa, iters, winsorize=win
                )  # NaN sort last
            else:
                order = np.argsort(v, axis=1)
                if sw is not None:
                    ws = np.take_along_axis(np.ascontiguousarray(sw.T), order, 1)
                elif weights.ndim == 2:  # per-channel weights: rows are samples, (n, rows, cols, 3) layout
                    ws = weights[order, (np.arange(m0, m0 + len(v)) % 3)[:, None]]
                else:
                    ws = weights[order]
                out[m0 : m0 + chunk] = sigma_clip_sorted_np(np.take_along_axis(v, order, 1), kappa, iters, ws, win)

        list(self.pool.map(work, range(0, flat.shape[1], chunk)))
        return out.reshape(stack.shape[1:])


def sigma_clip_sorted_np(s, kappa, iters, w=None, winsorize=False):
    """s: (m, N) sorted along axis 1, NaN last. Survivors of sigma clipping are always a
    contiguous run [lo, hi) of the sorted values, so clipping just narrows that range.
    w: optional per-value frame weights in the same order. Rejection is unweighted; the weights
    only apply to the final mean of the survivors.
    winsorize: estimate the spread after pulling survivors beyond median +/- 1.5 sigma in to that
    limit (sigma x 1.134, iterated; winsorized sigma clipping), so a few strong
    outliers cannot inflate it; rejection is still median +/- kappa sigma."""
    m, n = s.shape
    idx = np.arange(n)[None, :]
    lo = np.zeros(m, np.int64)
    hi = (~np.isnan(s)).sum(1)
    rows = np.arange(m)
    for it in range(iters + 1):
        cnt = np.maximum(hi - lo, 1)
        inr = (idx >= lo[:, None]) & (idx < hi[:, None])
        mean = np.where(inr, s, 0).sum(1) / cnt
        if it == iters:
            if w is not None:
                mean = np.where(inr, s * w, 0).sum(1) / np.maximum(np.where(inr, w, 0).sum(1), 1e-12)
            break
        med = 0.5 * (s[rows, np.minimum(lo + (cnt - 1) // 2, n - 1)] + s[rows, np.minimum(lo + cnt // 2, n - 1)])
        sd = np.sqrt(np.where(inr, (s - mean[:, None]) ** 2, 0).sum(1) / cnt)
        if winsorize:
            for _ in range(WINSOR_ITERS):
                xw = np.clip(s, (med - 1.5 * sd)[:, None], (med + 1.5 * sd)[:, None])
                mw = np.where(inr, xw, 0).sum(1) / cnt
                sd = 1.134 * np.sqrt(np.where(inr, (xw - mw[:, None]) ** 2, 0).sum(1) / cnt)
        new_lo = np.maximum(lo, (s < (med - kappa * sd)[:, None]).sum(1))
        new_hi = np.minimum(hi, (s <= (med + kappa * sd)[:, None]).sum(1))
        keep = (new_hi - new_lo) >= 3
        lo = np.where(keep, new_lo, lo)
        hi = np.where(keep, new_hi, hi)
    return np.where(hi > lo, mean, np.nan).astype(np.float32)


# --------------------------------------------------------------------------- GPU


class GPUBackend:
    name = "gpu"

    def __init__(
        self,
        raws,
        pedestal,
        n=None,
        interp="bilinear",
        debayer="bilinear",
        indices=None,
        paths=None,
        sizes=None,
        storage=None,
        log=None,
        loading_progress=None,
        storage_notice=None,
    ):
        """raws yields the lights in order. indices: their positions among all n lights when only
        some are loaded (e.g. those reaching a crop); frame numbers elsewhere stay global.
        Each light is stored on its own (lights may come from sensors of different sizes);
        pedestal: one value or one per loaded light."""
        import torch
        import torch.nn.functional as F

        self.torch, self.F = torch, F
        self.interp = interp
        self.debayer = debayer
        self.edge = sensor_edge(debayer, interp)
        self.dev = torch.device("cuda")
        self.rcd, self.rcd_kind = gpu_rcd() if debayer == "rcd" else (rcd_demosaic, "not used")
        self.clip, self.clip_kind = gpu_clip()
        self.warp, self.warp_kind = gpu_warp()
        self.calibration, self.calibration_kind = gpu_calibrate()
        self.timer = Timer(None if os.environ.get("GPUCOADD_SYNC_TIMING") == "0" else torch.cuda.synchronize)
        self.n = n = n or 1
        self.sort, self.sort_kind = gpu_sort(n)
        self.loaded = list(range(n)) if indices is None else [int(i) for i in indices]
        if len(set(self.loaded)) != len(self.loaded) or not all(0 <= i < n for i in self.loaded):
            raise ValueError("Loaded light indices must be distinct and within the input list")
        self.slot = None if indices is None else {i: k for k, i in enumerate(self.loaded)}
        self.pedestals = per_light(pedestal, len(self.loaded), "pedestal")
        from raw_storage import plan, DiskLight

        requested = storage or os.environ.get("GPUCOADD_STORAGE", "auto")
        if paths is not None and (len(paths) != len(self.loaded) or sizes is None or len(sizes) != len(paths)):
            raise ValueError("Storage paths and sensor sizes must match loaded lights")
        if sizes is None:
            raws = iter(raws)
            first = next(raws)
            sizes = [tuple(first.shape)] * len(self.loaded)
            raws = itertools.chain([first], raws)
        free, total = torch.cuda.mem_get_info(self.dev)
        self.storage_plan = plan(sizes, free, total, mode=requested, has_paths=paths is not None)
        self.storage_mode = self.storage_plan["mode"]
        if storage_notice:
            storage_notice(
                "raw",
                self.storage_mode,
                "normal" if self.storage_mode == "gpu" else "capacity" if requested == "auto" else "forced",
                bytes=self.storage_plan["raw_bytes"],
            )
        self.h, self.w = map(int, sizes[0])
        self.transfer_bytes = 0
        self.read_ahead = True  # file-backed crops read by background threads (warp_tile); same pixels
        if log:
            place = {
                "gpu": "GPU memory",
                "ram": "ordinary RAM; GPU crops uploaded as needed",
                "disk": "original files; GPU crops uploaded as needed",
            }[self.storage_mode]
            log(
                f"Raw storage: {self.storage_plan['raw_bytes'] / 2**30:.2f} GiB; using {place}. "
                f"GPU memory kept free for the desktop and working space: "
                f"{self.storage_plan['gpu_workspace_reserve_bytes'] / 2**30:.2f} GiB."
            )
        if self.storage_mode == "disk":
            raws = (DiskLight(p, shape) for p, shape in zip(paths, sizes))
        self.paths = None if paths is None else [str(p) for p in paths]
        self.log, self.storage_notice, self.released = log, storage_notice, 0
        # one tensor per light (uint16 has no fast torch path: keep the raw bits as int16 and fix
        # up on use). External storage retains only metadata or CPU tensors; in RAM mode a light
        # can also be a DiskLight (read from its file per tile) when RAM runs short.
        self.raw = []
        raw_bytes = 0
        for r in raws:
            if len(self.raw) == len(self.loaded):
                raise ValueError(f"More lights than the {len(self.loaded)} expected")
            raw_bytes += int(r.shape[0]) * int(r.shape[1]) * 2
            budget = self.storage_plan.get(self.storage_mode + "_raw_budget_bytes")
            if budget is not None and raw_bytes > budget:
                raise MemoryError("Raw lights exceeded the planned storage budget; use disk storage.")
            if self.storage_mode == "disk":
                self.raw.append(r)
            else:
                from resource_guard import check_ram, check_gpu

                light_bytes = int(r.shape[0]) * int(r.shape[1]) * 2
                if self.storage_mode == "ram" and self.paths and not check_ram(light_bytes, "loading lights"):
                    # RAM ran short while loading: this light is read from its file per tile instead
                    self.raw.append(DiskLight(self.paths[len(self.raw)], tuple(r.shape)))
                    self.note_released(1, "while loading", log=False)
                else:
                    if self.storage_mode == "gpu":
                        check_gpu(
                            self, light_bytes + self.storage_plan["gpu_workspace_reserve_bytes"], "raw-light upload"
                        )
                    raw = torch.from_numpy(np.ascontiguousarray(r).view(np.int16))
                    if self.storage_mode == "gpu":
                        with self.timer("upload"):
                            raw = raw.to(self.dev)
                    self.raw.append(raw)
            if loading_progress and (len(self.raw) % 32 == 0 or len(self.raw) == len(self.loaded)):
                loading_progress("loading lights", len(self.raw), len(self.loaded))
        if len(self.raw) != len(self.loaded):
            raise ValueError(f"Expected {len(self.loaded)} lights, received {len(self.raw)}")
        if self.released:
            self.note_released(0, "while loading")
        self.sizes = {i: (int(r.shape[1]), int(r.shape[0])) for i, r in zip(self.loaded, self.raw)}  # (width, height)
        del r
        # ones where a master flat will go (no flat correction yet); the pedestal is the camera offset
        self.flat_inv = {s: torch.ones(s, device=self.dev) for s in {tuple(r.shape) for r in self.raw}}
        self.k_db = torch.from_numpy(np.stack([K_RB, K_G, K_RB])[:, None]).to(self.dev)
        g = cv2.getGaussianKernel(9, 1.5).astype(np.float32).ravel()
        self.k_gauss = torch.from_numpy(g).to(self.dev)
        hmax, wmax = max(r.shape[0] for r in self.raw), max(r.shape[1] for r in self.raw)
        self.masks = torch.from_numpy(np.stack(bayer_masks(hmax, wmax))).to(self.dev)  # GRBG is periodic

    def keep_only(self, keep):
        """Free the loaded lights not in `keep` (frame indices)."""
        keep = set(keep)
        ks = [k for k, i in enumerate(self.loaded) if i in keep]
        self.raw = [self.raw[k] for k in ks]
        if self.paths is not None:
            self.paths = [self.paths[k] for k in ks]
        self.released = self.file_backed()
        self.pedestals = [self.pedestals[k] for k in ks]
        self.loaded = [self.loaded[k] for k in ks]
        self.slot = {i: k for k, i in enumerate(self.loaded)}
        self.sizes = {i: self.sizes[i] for i in self.loaded}
        shapes = {tuple(r.shape) for r in self.raw}
        self.flat_inv = {s: f for s, f in self.flat_inv.items() if s in shapes}
        self.torch.cuda.empty_cache()

    def calibrate(self, i, a=0, b=None, c=0, d=None, ahead=None):
        k = i if self.slot is None else self.slot[i]
        shape = self.raw[k].shape
        b = shape[0] if b is None else b
        d = shape[1] if d is None else d
        raw = self.raw_crop(k, (a, b, c, d), ahead)
        x = raw.to(self.torch.int32).bitwise_and_(0xFFFF).float()
        return (x - self.pedestals[k]) * self.flat_inv[tuple(shape)][a:b, c:d]

    def release_ram(self, need):
        """Read RAM-held lights again from their files instead, until `need` bytes are freed. The
        raw bits are identical either way; only reading them for each tile takes longer."""
        if self.storage_mode != "ram" or not self.paths:
            return 0
        from raw_storage import DiskLight

        # One way only: nothing moves a light back into RAM during the job, so releasing cannot
        # start a release / refill cycle. Each call frees just what is asked for.
        freed = count = 0
        for k in reversed(range(len(self.raw))):
            if freed >= need:
                break
            r = self.raw[k]
            if isinstance(r, self.torch.Tensor):
                self.raw[k] = DiskLight(self.paths[k], tuple(r.shape))
                freed += r.numel() * r.element_size()
                count += 1
            r = None  # drop the last reference, so the RAM returns to Windows now
        if count:
            self.note_released(count, "because other programs needed RAM")
        return freed

    def restore_ram(self):
        """Between jobs only (a kept set reused by the next job): read lights that were released to
        file reads back into RAM while RAM allows, one at a time, stopping at the first shortage and
        never releasing anything to make room. Only within RAM mode: lights on the GPU are never
        released, so nothing moves between GPU and RAM. Same raw bits as the first load (the file
        is checked against its recorded state as it is read)."""
        if self.storage_mode != "ram" or not self.released:
            return 0
        from resource_guard import check_ram

        restored = 0
        for k, r in enumerate(self.raw):
            if isinstance(r, self.torch.Tensor):
                continue
            if not check_ram(int(r.shape[0]) * int(r.shape[1]) * 2, "reading lights back into RAM"):
                break
            self.raw[k] = self.torch.from_numpy(np.ascontiguousarray(r.crop()).view(np.int16))
            restored += 1
        self.released = self.file_backed()
        return restored

    def file_backed(self):
        """Lights of a RAM-mode set that are read from their files (released or short while loading)."""
        if self.storage_mode != "ram":
            return 0
        return sum(not isinstance(r, self.torch.Tensor) for r in self.raw)

    def note_released(self, count, why, log=True):
        if not self.released and count and self.storage_notice:
            self.storage_notice("raw", "mixed", "capacity", bytes=self.storage_plan["raw_bytes"])
        self.released += count
        if log and self.log:
            self.log(
                f"RAM is short {why}: {self.released} lights are now read from their files for each tile "
                "instead of being held in RAM. Same pixels; stacking takes a little longer."
            )

    def raw_crop(self, k, box, ahead=None):
        """Upload only the requested crop in external-storage modes. ahead: the same crop, already
        read from the file (ReadAhead), or None to read it here."""
        a, b, c, d = box
        r = self.raw[k]
        if not isinstance(r, self.torch.Tensor):  # a DiskLight: disk mode, or released from RAM
            crop = r.crop(a, b, c, d) if ahead is None else ahead
            raw = self.torch.from_numpy(np.ascontiguousarray(crop).view(np.int16))
        else:
            raw = r[a:b, c:d]
        if raw.device != self.dev and not raw.is_cuda:
            self.transfer_bytes += raw.numel() * raw.element_size()
            raw = raw.to(self.dev)
        return raw

    def storage_info(self):
        raw_bytes = sum(int(r.shape[0]) * int(r.shape[1]) * 2 for r in self.raw if isinstance(r, self.torch.Tensor))
        return {
            **self.storage_plan,
            "available_lights": len(self.loaded),
            "resident_lights": len(self.loaded) if self.storage_mode == "gpu" else 0,
            "host_raw_bytes": raw_bytes if self.storage_mode == "ram" else 0,
            "transfer_bytes": self.transfer_bytes,
            "lights_read_from_files_after_ram_ran_short": self.released,
        }

    def detect(self, cfa):
        torch, F = self.torch, self.F
        lum = cfa[0::2, 0::2] + cfa[0::2, 1::2] + cfa[1::2, 0::2] + cfa[1::2, 1::2]
        hh, ww = lum.shape
        x = lum[None, None]
        small = F.adaptive_avg_pool2d(x, (hh // BG_CELL, ww // BG_CELL))
        d = x - F.interpolate(small, size=(hh, ww), mode="bilinear", align_corners=False)
        kg = self.k_gauss
        sm = F.conv2d(F.pad(d, (4, 4, 0, 0), mode="reflect"), kg.view(1, 1, 1, 9))
        sm = F.conv2d(F.pad(sm, (0, 0, 4, 4), mode="reflect"), kg.view(1, 1, 9, 1))
        samp = sm[0, 0, ::4, ::4]
        med = samp.median()
        sig = 1.4826 * (samp - med).abs().median()
        peak = (sm == F.max_pool2d(sm, 7, 1, 3)) & (sm > med + 5 * sig)
        peak = peak[0, 0]
        peak[:8], peak[-8:], peak[:, :8], peak[:, -8:] = False, False, False, False
        ys, xs = torch.nonzero(peak, as_tuple=True)
        smv = sm[0, 0, ys, xs]
        top = torch.argsort(smv, descending=True)[:N_DETECT]
        ys, xs = ys[top], xs[top]
        o = torch.arange(-3, 4, device=self.dev)
        oy, ox = torch.meshgrid(o, o, indexing="ij")
        win = d[0, 0, ys[:, None, None] + oy, xs[:, None, None] + ox].clamp_min(0)
        tot = win.sum((1, 2)) + 1e-9
        cy = (win * oy).sum((1, 2)) / tot
        cx = (win * ox).sum((1, 2)) / tot
        fwhm = star_fwhm(*(t.cpu().numpy().astype(np.float64) for t in (win, ox, oy, tot, cx, cy)), np)
        xy = torch.stack([xs + cx, ys + cy], 1) * 2 + 0.5
        return torch.cat([xy, self.photometry(cfa, ys, xs)], 1).cpu().numpy(), fwhm

    def photometry(self, cfa, ys, xs):
        """Same as CPUBackend.photometry."""
        torch, F = self.torch, self.F
        o = torch.arange(-AP_R, AP_R + 1, device=self.dev)
        oy, ox = torch.meshgrid(o, o, indexing="ij")
        ap = ((ox**2 + oy**2) <= AP_R**2).float()
        cols = []
        for pl in (cfa[0::2, 1::2], 0.5 * (cfa[0::2, 0::2] + cfa[1::2, 1::2]), cfa[1::2, 0::2]):
            hh, ww = pl.shape
            small = F.adaptive_avg_pool2d(pl[None, None], (hh // BG_CELL, ww // BG_CELL))
            d = pl - F.interpolate(small, size=(hh, ww), mode="bilinear", align_corners=False)[0, 0]
            cols.append((d[ys[:, None, None] + oy, xs[:, None, None] + ox] * ap).sum((1, 2)))
        f = torch.arange(-1, 3, device=self.dev)
        fy, fx = torch.meshgrid(f, f, indexing="ij")
        cols.append(cfa[2 * ys[:, None, None] + fy, 2 * xs[:, None, None] + fx].amax((1, 2)))
        return torch.stack(cols, 1)

    def stats(self, cfa):
        med, mad = [], []
        for c in (cfa[0::2, 1::2], cfa[0::2, 0::2], cfa[1::2, 0::2]):
            s = c[::4, ::4].reshape(-1)
            m = s.median()
            med.append(m)
            mad.append((s - m).abs().median())
        rgb = []  # pixel-scale noise per channel, as CPUBackend.stats
        for c in (cfa[0::4, 1::2], cfa[0::4, 0::2], cfa[1::4, 0::2]):
            dc = (c[:, 1:] - c[:, :-1]).reshape(-1)
            rgb.append(1.4826 * (dc - dc.median()).abs().median() / np.sqrt(2))
        rgb = self.torch.stack(rgb).cpu().numpy().astype(np.float32)
        return (
            self.torch.stack(med).cpu().numpy().astype(np.float32),
            self.torch.stack(mad).cpu().numpy().astype(np.float32),
            float(rgb[1]),
            rgb,
        )

    def demosaiced_samples(self, i, step=3, budget=2e6):
        """Every step-th demosaiced pixel of light i in both directions (odd steps sample all
        Bayer phases alike), per channel: (3, N) float64. Demosaiced in row strips."""
        torch, F = self.torch, self.F
        k = i if self.slot is None else self.slot[i]
        h, w = self.raw[k].shape
        ped = self.pedestals[k]
        out = []
        for a, b, ka, kb in strips(h, w, budget):
            cal = self.calibrate(i, a, b)[None]
            ms = self.masks[:, : b - a, :w]
            if self.debayer == "rcd":
                rgb = self.rcd(torch, F, (cal + ped) / 65535, *ms) * 65535 - ped
            else:
                rgb = F.conv2d(F.pad(cal[:, None] * ms[None], (1, 1, 1, 1), mode="reflect"), self.k_db, groups=3)
            out.append(rgb[0, :, ka - a + (-ka) % step : kb - a : step, ::step].reshape(3, -1))
        return torch.cat(out, 1).double().cpu().numpy()

    # integration temporaries per light and pixel, measured peak (sort values + indices, clipping
    # state, weights; noise mode also the gathered per-sample noise), bytes
    INTEGRATE_BYTES = INTEGRATION_BYTES
    VRAM_RESERVE = GPU_RESERVE  # left for the CUDA context and the display

    def free_vram(self):
        """Bytes this process can still use: free on the device plus what torch holds unused."""
        torch = self.torch
        free = torch.cuda.mem_get_info(self.dev)[0]
        # the same counters as memory_reserved / memory_allocated, without flattening every
        # allocator statistic (0.43 ms per call while loading, called once per light)
        stats = torch.cuda.memory_stats_as_nested_dict(self.dev)
        reserved = stats.get("reserved_bytes", {}).get("all", {}).get("current", 0)
        allocated = stats.get("allocated_bytes", {}).get("all", {}).get("current", 0)
        return free + reserved - allocated

    def alloc_stack(self, n, rows, cols):
        """A tile's (n, 3, rows, cols) stack: a view of the buffer from reserve_stack when it is
        large enough. Tiles stack different numbers of lights, and fresh tensors of ever-changing
        sizes fragment the caching allocator (SH2-142: 8.4 -> 16 GiB reserved, paging)."""
        k = n * 3 * rows * cols
        buf = getattr(self, "_stack_buf", None)
        if buf is not None and buf.numel() >= k:
            return buf[:k].view(n, 3, rows, cols)
        return self.torch.empty((n, 3, rows, cols), device=self.dev)

    def reserve_stack(self, n, rows, cols, feather=False):
        """One buffer for every tile's stack (n: most lights any tile stacks), and one for its
        feather weights when feathering; n = 0 releases them."""
        self._stack_buf = self._feather_buf = None
        if n:
            self._stack_buf = self.torch.empty(n * 3 * rows * cols, device=self.dev)
            if feather:
                self._feather_buf = self.torch.empty(n * rows * cols, device=self.dev)

    def alloc_feather(self, n, rows, cols):
        """A tile's (n, rows, cols) feather weights, from the reserve_stack buffer when it fits."""
        k = n * rows * cols
        buf = getattr(self, "_feather_buf", None)
        if buf is not None and buf.numel() >= k:
            return buf[:k].view(n, rows, cols)
        return self.torch.empty((n, rows, cols), device=self.dev)

    def feather_tile(self, out, frames, mats, x0, y0, radius):
        """Feather weights of `frames` (rows of `out`) on the tile at (x0, y0): feather_kernels'
        CUDA kernel, built on first use, or its torch version if it cannot be built."""
        import feather_kernels

        if getattr(self, "_feather_fn", None) is None:
            try:
                feather_kernels.module()
                self._feather_fn = feather_kernels.feather_cuda
            except Exception:
                self._feather_fn = feather_kernels.feather_torch
        self._feather_fn(
            self.torch,
            out,
            list(range(len(frames))),
            np.stack([mats[i] for i in frames]),
            np.array([self.sizes[i] for i in frames], np.float64),
            self.edge,
            x0,
            y0,
            radius,
        )

    def warp_tile(self, stack, frames, mats, coefs, x0, x1, y0, y1):
        """Calibrate, debayer and warp frames onto one output tile, several frames per GPU call:
        RCD is ~200 small elementwise ops, so per-frame calls would be dominated by launch overhead.
        Crops differ slightly in size per frame, so each batch is padded to its largest crop.
        Batches are sized by pixel count: RCD holds dozens of crop-sized temporaries, and running
        past VRAM makes Windows page GPU memory to system RAM, which is many times slower."""
        torch, F, T = self.torch, self.F, self.timer
        boxes = [tile_source_box(mats[i], x0, x1, y0, y1, *self.sizes[i]) for i in frames]
        sizes = [(b - a) * (d - c) for a, b, c, d in (bx for bx in boxes if bx is not None)] or [1]
        batch = max(1, min(32, int((RCD_BATCH_PIXELS if self.debayer == "rcd" else 16e6) // max(sizes))))
        # Tile-local capacity only: reused within the unchanged batches, then released
        # before integration. Batch locals must die before allocating the next RGB image.
        calibration_buffer = None
        # Lights read from their files (disk mode, or released from RAM): their crops are read by
        # background threads in the order the batches below use them (raw_storage.ReadAhead).
        ahead = None
        if self.storage_mode != "gpu" and self.read_ahead:
            from raw_storage import DiskLight, ReadAhead

            items = []
            for j, i in enumerate(frames):
                r = self.raw[i if self.slot is None else self.slot[i]]
                if boxes[j] is not None and isinstance(r, DiskLight):
                    items.append((j, r, tuple(int(v) for v in boxes[j])))
            if len(items) > 1:
                ahead = ReadAhead(items)

        def read(j):
            return None if ahead is None else ahead.take(j)

        def warp_batch(js):
            nonlocal calibration_buffer
            bx = np.array([boxes[j] for j in js])  # a, b, c, d
            hmax, wmax = int((bx[:, 1] - bx[:, 0]).max()), int((bx[:, 3] - bx[:, 2]).max())
            # RCD's scaling, (calibrated + pedestal) / 65535 before and x 65535 - pedestal after,
            # done inside the calibration and RCD kernels when both are in use (same roundings)
            fold = self.debayer == "rcd" and self.calibration is not None and getattr(self.rcd, "folds_scale", False)
            with T("calibrate"):
                needed = len(js) * hmax * wmax
                if calibration_buffer is None or calibration_buffer.numel() < needed:
                    calibration_buffer = None  # release the old capacity before growing
                    calibration_buffer = torch.empty(needed, device=self.dev)
                cal = calibration_buffer[:needed].view(len(js), hmax, wmax)
                if self.calibration is not None:
                    slots = [frames[j] if self.slot is None else self.slot[frames[j]] for j in js]
                    if self.storage_mode == "gpu":
                        self.calibration.calibrate_pad(
                            [self.raw[k] for k in slots],
                            [self.flat_inv[tuple(self.raw[k].shape)] for k in slots],
                            [self.pedestals[k] for k in slots],
                            bx,
                            cal,
                            unit=fold,
                        )
                    else:
                        crops = [self.raw_crop(k, box, read(j)) for j, k, box in zip(js, slots, bx)]
                        flats = [
                            self.flat_inv[tuple(self.raw[k].shape)][a:b, c:d] for k, (a, b, c, d) in zip(slots, bx)
                        ]
                        local = np.array([[0, b - a, 0, d - c] for a, b, c, d in bx])
                        self.calibration.calibrate_pad(
                            crops, flats, [self.pedestals[k] for k in slots], local, cal, unit=fold
                        )
                        del crops, flats
                else:
                    for q, j in enumerate(js):
                        a, b, c, d = boxes[j]
                        crop = self.calibrate(frames[j], a, b, c, d, read(j))
                        # Each crop reflects at its own boundary before batching.
                        iy = torch.arange(hmax, device=self.dev) % (2 * (b - a - 1))
                        ix = torch.arange(wmax, device=self.dev) % (2 * (d - c - 1))
                        iy = torch.minimum(iy, 2 * (b - a - 1) - iy)
                        ix = torch.minimum(ix, 2 * (d - c - 1) - ix)
                        cal[q] = crop[iy[:, None], ix[None, :]]
                        del crop, iy, ix
            with T("debayer"):
                ms = self.masks[:, :hmax, :wmax]
                if self.debayer == "rcd":
                    peds = [self.pedestals[frames[j] if self.slot is None else self.slot[frames[j]]] for j in js]
                    if fold:
                        rgb = self.rcd(torch, F, cal, *ms, pedestals=torch.tensor(peds, device=self.dev))
                    elif len(set(peds)) == 1:  # one sensor: scalar path, as before
                        rgb = self.rcd(torch, F, (cal + peds[0]) / 65535, *ms) * 65535 - peds[0]
                    else:
                        ped = torch.tensor(peds, device=self.dev)[:, None, None]
                        rgb = self.rcd(torch, F, (cal + ped) / 65535, *ms) * 65535 - ped[:, None]
                else:
                    rgb = F.conv2d(F.pad(cal[:, None] * ms[None], (1, 1, 1, 1), mode="reflect"), self.k_db, groups=3)
            # one fused kernel: coordinates, taps, edge mask, brightness, scatter (warp_kernels);
            # warp_torch is the same arithmetic when the kernel cannot be built
            import warp_kernels

            with T("warp"):
                (self.warp.warp if self.warp is not None else warp_kernels.warp_torch)(
                    rgb,
                    np.stack([mats[frames[j]] for j in js]),
                    bx,
                    [self.sizes[frames[j]] for j in js],
                    self.edge,
                    x0,
                    y0,
                    y1 - y0,
                    x1 - x0,
                    np.stack([coefs[frames[j]][0] for j in js]),
                    np.stack([coefs[frames[j]][1] for j in js]),
                    js,
                    stack,
                    planes=np.stack([coefs[frames[j]][2] for j in js]) if len(coefs[frames[js[0]]]) > 2 else None,
                    interp=self.interp,
                )

        try:
            for k0 in range(0, len(frames), batch):
                js = [j for j in range(k0, min(k0 + batch, len(frames))) if boxes[j] is not None]
                for j in range(k0, min(k0 + batch, len(frames))):
                    if boxes[j] is None:
                        stack[j] = float("nan")
                if js:
                    warp_batch(js)
        finally:
            if ahead is not None:
                ahead.close()

    def integrate(self, stack, kappa, iters, weights=None, kept=False, rejection="sigma", noise=None, feather=None):
        """Clipped mean per pixel and channel; kept=True also returns how many samples survived
        rejection (diagnostics only; the mean is identical either way). rejection: "sigma" |
        "winsorized" | "noise" (noise: (frames, 3) per-light noise in normalized units).
        feather: (frames, rows, cols) per-sample weights (feather_tile), multiplied into the light
        weights of the mean; rejection itself is unchanged."""
        torch = self.torch
        n = stack.shape[0]
        flat = stack.reshape(n, -1)
        out = torch.empty(flat.shape[1], device=self.dev)
        count = torch.empty(flat.shape[1], device=self.dev, dtype=torch.int32) if kept else None
        wt = None if weights is None else torch.from_numpy(weights.astype(np.float32)).to(self.dev)
        clip = getattr(self, "clip", None)  # CUDA rejection kernels (None: torch)
        # noise mode holds several (frames x chunk) temporaries: smaller chunks keep VRAM in bounds
        chunk = 1 << (17 if rejection == "noise" else 18)
        # Many lights (e.g. two telescopes: 587 on SH2-142) make even that exceed the free VRAM,
        # and Windows then pages GPU memory (integration 874 s instead of ~90). Halve the chunk
        # until its temporaries use at most half of what is free: debayering the next tile needs
        # room too, and cached blocks fragment. The result is identical for any chunk (per-pixel
        # work), and smaller chunks are no slower (587 lights: 1.2 s per 512 px tile either way).
        per = n * self.INTEGRATE_BYTES[rejection]
        budget = (self.free_vram() - self.VRAM_RESERVE) / 2
        while chunk > 4096 and per * chunk > budget:
            chunk //= 2
        plane = stack.shape[2] * stack.shape[3]  # (n, 3, rows, cols) layout: channel = column // plane
        if rejection == "noise":
            nz = torch.from_numpy(np.asarray(noise, np.float32)).to(self.dev)
        fw = None if feather is None else feather.reshape(n, plane)
        sorter = getattr(self, "sort", None)
        # Frame indices are needed only to gather weights, feather weights or per-sample noise in
        # sorted order; without them a values-only sort gives the same sorted values, faster.
        values_only = wt is None and fw is None and rejection != "noise" and hasattr(sorter, "sort_values")
        for m0 in range(0, flat.shape[1], chunk):
            if values_only:
                values = sorter.sort_values(flat[:, m0 : m0 + chunk])
                res = (
                    clip.clip_sorted(values, kappa, iters, None, kept, rejection == "winsorized", WINSOR_ITERS)
                    if clip is not None
                    else sigma_clip_sorted_torch(torch, values, kappa, iters, None, kept, rejection == "winsorized")
                )
                if kept:
                    out[m0 : m0 + chunk], count[m0 : m0 + chunk] = res
                else:
                    out[m0 : m0 + chunk] = res
                del values, res
                continue
            srt = (
                sorter.sort(flat[:, m0 : m0 + chunk])
                if sorter is not None
                else torch.sort(flat[:, m0 : m0 + chunk], dim=0)
            )
            chan = None
            if rejection == "noise" or (wt is not None and wt.ndim == 2):
                chan = torch.arange(m0, m0 + srt.values.shape[1], device=self.dev) // plane
            if wt is None:
                ws = None
            elif wt.ndim == 2:  # per-channel weights (n, 3): each column's channel
                ws = wt[srt.indices, chan[None, :]]
            else:
                ws = wt[srt.indices]
            if fw is not None:  # each sample's feather weight, in the sorted order
                pix = torch.arange(m0, m0 + srt.values.shape[1], device=self.dev) % plane
                f = torch.gather(fw[:, pix], 0, srt.indices)
                ws = f if ws is None else ws * f
                del f
            if rejection == "noise":
                sig = torch.gather(nz[:, chan], 0, srt.indices)
            values = srt.values
            del srt  # indices are no longer needed by rejection
            if rejection == "noise":
                if clip is not None:
                    res = clip.clip_noise(values, sig, kappa, iters, ws, kept)
                else:
                    res = noise_clip_torch(torch, values, sig, kappa, iters, ws, kept)
            elif clip is not None:
                res = clip.clip_sorted(values, kappa, iters, ws, kept, rejection == "winsorized", WINSOR_ITERS)
            else:
                res = sigma_clip_sorted_torch(torch, values, kappa, iters, ws, kept, rejection == "winsorized")
            if kept:
                out[m0 : m0 + chunk], count[m0 : m0 + chunk] = res
            else:
                out[m0 : m0 + chunk] = res
            del values, ws, res
            if rejection == "noise":
                del sig
            del chan
            if fw is not None:
                del pix
        # (n, 3, rows, cols) layout -> (rows, cols, 3) to match the CPU output
        img = out.reshape(stack.shape[1:]).permute(1, 2, 0).cpu().numpy()
        if kept:
            return img, count.reshape(stack.shape[1:]).permute(1, 2, 0).cpu().numpy()
        return img


def sigma_clip_sorted_torch(torch, s, kappa, iters, w=None, kept=False, winsorize=False):
    """Same algorithm as sigma_clip_sorted_np, with frames on dim 0. kept=True also returns the
    number of surviving samples (0 where the result is NaN)."""
    n, m = s.shape
    idx = torch.arange(n, device=s.device)[:, None]
    lo = torch.zeros(m, dtype=torch.long, device=s.device)
    hi = (~torch.isnan(s)).sum(0)
    cols = torch.arange(m, device=s.device)
    for it in range(iters + 1):
        cnt = (hi - lo).clamp_min(1)
        inr = (idx >= lo) & (idx < hi)
        mean = torch.where(inr, s, 0).sum(0) / cnt
        if it == iters:
            if w is not None:
                mean = torch.where(inr, s * w, 0).sum(0) / torch.where(inr, w, 0).sum(0).clamp_min(1e-12)
            break
        med = 0.5 * (s[(lo + (cnt - 1) // 2).clamp_max(n - 1), cols] + s[(lo + cnt // 2).clamp_max(n - 1), cols])
        sd = torch.sqrt(torch.where(inr, (s - mean) ** 2, 0).sum(0) / cnt)
        if winsorize:
            for _ in range(WINSOR_ITERS):
                xw = torch.minimum(torch.maximum(s, med - 1.5 * sd), med + 1.5 * sd)
                mw = torch.where(inr, xw, 0).sum(0) / cnt
                sd = 1.134 * torch.sqrt(torch.where(inr, (xw - mw) ** 2, 0).sum(0) / cnt)
        new_lo = torch.maximum(lo, (s < med - kappa * sd).sum(0))
        new_hi = torch.minimum(hi, (s <= med + kappa * sd).sum(0))
        keep = (new_hi - new_lo) >= 3
        lo = torch.where(keep, new_lo, lo)
        hi = torch.where(keep, new_hi, hi)
    result = torch.where(hi > lo, mean, float("nan"))
    if kept:
        return result, (hi - lo).clamp_min(0).to(torch.int32)
    return result


def noise_clip_torch(torch, s, sig, kappa, iters, w=None, kept=False):
    """Noise-aware sigma clipping. s: (n, m) samples sorted along dim 0 (NaN last); sig: (n, m)
    each sample's own light noise (normalized units). A sample is rejected when its residual from
    the centre exceeds kappa x (spread of the noise-standardized residuals) x its own noise, so a
    light that is noisier (e.g. hazy, scaled up by brightness matching) is judged against its own
    noise instead of being clipped more often. Centre: median first, then the inverse-variance
    mean of the survivors. Survivors need not be contiguous in sorted order, so masks are used.
    Same output conventions as sigma_clip_sorted_torch."""
    n, m = s.shape
    cols = torch.arange(m, device=s.device)
    fin = ~torch.isnan(s)
    nf = fin.sum(0)
    x = torch.where(fin, s, 0)
    sig = sig.clamp_min(1e-12)
    c = 0.5 * (s[((nf - 1) // 2).clamp(0, n - 1), cols] + s[(nf // 2).clamp(0, n - 1), cols])
    c = torch.where(nf > 0, c, 0)
    use = fin
    for _ in range(iters):
        r = torch.where(use, (x - c) / sig, 0)
        k = use.sum(0).clamp_min(1)
        spread = torch.sqrt((r * r).sum(0) / k)
        new = fin & ((x - c).abs() <= kappa * spread * sig)
        use = torch.where((new.sum(0) >= 3)[None], new, use)
        iv = torch.where(use, 1 / sig**2, 0)
        c = (x * iv).sum(0) / iv.sum(0).clamp_min(1e-30)
    k = use.sum(0)
    if w is None:
        mean = torch.where(use, x, 0).sum(0) / k.clamp_min(1)
    else:
        mean = torch.where(use, x * w, 0).sum(0) / torch.where(use, w, 0).sum(0).clamp_min(1e-12)
    result = torch.where(k > 0, mean, float("nan"))
    if kept:
        return result, k.to(torch.int32)
    return result


# --------------------------------------------------------------------------- pipeline


def _pool_ready():
    return os.getpid()


def start_registration_pool(workers):
    """A process pool for register(), started before the lights load. Windows starts workers
    with spawn, and each imports the engine on its first task (1.65 s over 8 workers, measured);
    a no-op task per worker moves that import to the time the lights load and stars are
    detected. register() shuts the pool down when it is done; a caller that does not reach
    register() shuts it down itself."""
    pool = ProcessPoolExecutor(workers)
    for _ in range(workers):
        pool.submit(_pool_ready)
    return pool


def match_to_ref(job):
    """Triangle-match the brightest stars for a similarity transform, then use it to pair up
    every detected star: the pairs give the frame's transparency (median star flux ratio vs the
    reference, per channel, unsaturated stars only) and, for a homography, its fit.
    Star rows are [x, y, flux R, flux G, flux B, peak]. Returns (dst->src matrix, flux ratio).
    job[4], expect: the src -> ref scale expected from the optics (src pixel scale / ref pixel
    scale, e.g. ~1.04 for an S50 light onto an S50 Pro reference); the fit must be within 5%."""
    src, ref, transform = job[:3]
    limit = job[3] if len(job) > 3 else N_MATCH
    expect = job[4] if len(job) > 4 else 1.0
    lo, hi = 0.95 * expect, 1.05 * expect
    # batched triangle preparation, same values as astroalign's (GPUCOADD_REGISTRATION=reference: astroalign's own)
    import registration_fast

    find = (
        aa.find_transform
        if os.environ.get("GPUCOADD_REGISTRATION") == "reference"
        else registration_fast.find_transform
    )
    try:
        t, _ = find(src[:limit, :2], ref[:limit, :2], max_control_points=limit)
    except aa.MaxIterError:
        return None  # no consistent triangle match: the light does not align
    except (ValueError, TypeError):
        # astroalign's answers for star lists it cannot match: fewer than 3 stars (ValueError),
        # none (TypeError), duplicate or non-finite positions (ValueError from its KD-tree).
        # The same errors from usable star lists are faults and must not look like a bad light.
        if degenerate_stars(src[:limit, :2]) or degenerate_stars(ref[:limit, :2]):
            return None
        raise
    return fit_star_pairs(src, ref, transform, t(src[:, :2]), t.scale, lo, hi)


def degenerate_stars(xy):
    """Star positions astroalign cannot match: fewer than 3, any non-finite or any repeated."""
    xy = np.asarray(xy, dtype=float)
    return len(xy) < 3 or not np.isfinite(xy).all() or len(np.unique(xy, axis=0)) < len(xy)


def fit_star_pairs(src, ref, transform, mapped, guess_scale, lo, hi, radius=2.0):
    """Pair every src star with the reference star nearest its `mapped` position (an initial
    transform's guess), then fit and validate as described in match_to_ref."""
    dist, k = cKDTree(ref[:, :2]).query(mapped)
    good = dist < radius
    # A chance triangle or several source stars paired to one target is not an
    # acceptable mosaic link. Require independent, spatially distributed pairs.
    used = set()
    for j in np.argsort(dist):
        if good[j] and int(k[j]) not in used:
            used.add(int(k[j]))
        else:
            good[j] = False
    if good.sum() < 10 or not lo <= guess_scale <= hi:
        return None
    pts = src[good, :2]
    if np.min(np.ptp(pts, axis=0)) < 100:
        return None
    affine, inliers = cv2.estimateAffinePartial2D(
        pts,
        ref[k[good], :2],
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=2000,
        confidence=0.999,
        refineIters=10,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 10:
        return None
    forward = np.eye(3)
    forward[:2] = affine
    transformed = (forward @ np.c_[src[:, :2], np.ones(len(src))].T).T[:, :2]
    errors = np.linalg.norm(transformed[good] - ref[k[good], :2], axis=1)
    fitted = inliers.ravel().astype(bool)
    rms = float(np.sqrt(np.mean(errors[fitted] ** 2)))
    scale = float(np.sqrt(abs(np.linalg.det(affine[:, :2]))))
    if rms > 1.0 or not lo <= scale <= hi:
        return None
    keep = np.flatnonzero(good)[fitted]
    good[:] = False
    good[keep] = True
    phot = good & (src[:, 5] < SAT) & (ref[k, 5] < SAT) & (src[:, 2:5] > 0).all(1) & (ref[k, 2:5] > 0).all(1)
    ratio = np.median(src[phot, 2:5] / ref[k[phot], 2:5], 0) if phot.sum() >= 10 else np.full(3, np.nan)
    minv = np.linalg.inv(forward)
    details = {"transform": "similarity", "matched_stars": int(good.sum()), "rms_px": rms, "scale": scale}
    if transform == "homography":
        hm, found = validated_homography(src[:, :2], src[good, :2], ref[k[good], :2], forward)
        if hm is None:
            details["homography_rejected"] = found
        else:
            minv = np.linalg.inv(hm)
            details.update(found)
    return minv, ratio, details


HOMOGRAPHY_SPAN = 0.6  # matched stars span this share of the detected star field on each axis
HOMOGRAPHY_DEVIATION = 0.02  # largest departure from the similarity at the field's corners, of its diagonal


def validated_homography(stars, src, dst, forward):
    """A homography (src -> dst) refining a validated similarity `forward`, or (None, reason).
    It is accepted only where it interpolates: at least 12 RANSAC inliers with RMS <= 1 px,
    spanning HOMOGRAPHY_SPAN of the detected star field (`stars`, all source stars, standing in
    for the sensor) on each axis, a positive projective denominator over that field, and at its
    corners within HOMOGRAPHY_DEVIATION of the field diagonal from the similarity. Stars matched
    in one region (a mosaic overlap, clouds) otherwise give a perspective that is unconstrained
    elsewhere and was accepted with corner errors of hundreds of pixels. Returns (matrix,
    diagnostics for it)."""
    if len(src) < 12:
        return None, "fewer than 12 matched stars"
    hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
    if hm is None or mask is None or not np.isfinite(hm).all() or abs(np.linalg.det(hm)) < 1e-12:
        return None, "no homography fit"
    inl = mask.ravel().astype(bool)
    if inl.sum() < 12:
        return None, "fewer than 12 homography inliers"
    hn = hm / hm[2, 2] if hm[2, 2] != 0 else hm  # checks only: the fit itself is returned unchanged
    lo, hi = stars.min(0), stars.max(0)
    extent = hi - lo
    if (extent <= 0).any() or (np.ptp(src[inl], axis=0) < HOMOGRAPHY_SPAN * extent).any():
        return None, "matched stars cover too little of the field"
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])

    def apply(m, pts):
        p = m @ np.c_[pts, np.ones(len(pts))].T
        return p[:2] / p[2], p[2]

    mapped, w = apply(hn, np.r_[corners, src[inl]])
    if not (np.isfinite(mapped).all() and (w > 0).all()):
        return None, "perspective folds within the field"
    rms = float(np.sqrt(np.mean(np.sum((mapped[:, 4:].T - dst[inl]) ** 2, 1))))
    if rms > 1.0:
        return None, "homography residuals above 1 px"
    deviation = float(np.max(np.hypot(*(mapped[:, :4] - apply(forward, corners)[0]))))
    if deviation > HOMOGRAPHY_DEVIATION * float(np.hypot(*extent)):
        return None, f"departs {deviation:.0f} px from the similarity at the field corners"
    centre = (lo + hi) / 2
    J = (apply(hn, np.array([centre + [1, 0], centre + [0, 1]]))[0].T - apply(hn, centre[None])[0].T).T
    return hm, {
        "transform": "homography",
        "matched_stars": int(inl.sum()),
        "rms_px": rms,
        "scale": float(np.sqrt(abs(np.linalg.det(J)))),
        "homography_departure_px": deviation,
    }


def guided_match(job):
    """Match through a predicted dst->src matrix (no triangle search): fast, and it uses every
    detected star, so it also refines a prediction composed of several links."""
    src, ref, transform, minv, expect = job
    forward = np.linalg.inv(minv)
    p = forward @ np.c_[src[:, :2], np.ones(len(src))].T
    guess = float(np.sqrt(abs(np.linalg.det(forward[:2, :2]))))
    return fit_star_pairs(src, ref, transform, (p[:2] / p[2]).T, guess, 0.95 * expect, 1.05 * expect, radius=3.0)


def shift_match(src, ref, transform, expect, search=300.0):
    """Lights close in time differ mostly by a shift (drift, dithering, a reframe): vote for the
    offset over all star pairs within `search` px (2 px bins), then refine with guided_match.
    Milliseconds, against seconds for a failing triangle search. None if there is no clear
    peak or the refinement does not validate (e.g. rotated): the caller falls back."""
    a, b = src[:, :2] * expect, ref[:, :2]
    d = (b[None, :, :] - a[:, None, :]).reshape(-1, 2)
    d = d[(np.abs(d) < search).all(1)]
    if len(d) < 10:
        return None
    bins = int(search)  # 2 px wide over [-search, search)
    h, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=bins, range=[[-search, search], [-search, search]])
    ix, iy = np.unravel_index(int(h.argmax()), h.shape)
    if h[ix, iy] < 10:
        return None
    near = (np.abs(d[:, 0] - (xe[ix] + 1)) <= 2) & (np.abs(d[:, 1] - (ye[iy] + 1)) <= 2)
    dx, dy = np.median(d[near], 0)
    forward = np.array([[expect, 0, dx], [0, expect, dy], [0, 0, 1.0]])
    return guided_match((src, ref, transform, np.linalg.inv(forward), expect))


def star_windows(src, ref, expect, min_stars=15):
    """Subsets of the reference's stars, in overlapping circles big enough to hold the whole src
    field at any rotation. A small sensor's field covers only part of a large one, so the large
    one's brightest stars are mostly outside it and whole-list triangle matching fails."""
    c = src[:, :2].mean(0)
    r = expect * np.linalg.norm(src[:, :2] - c, axis=1).max()
    step = r / 3
    radius = r + 0.71 * step  # every possible field centre is within 0.71 step of a window centre
    lo, hi = ref[:, :2].min(0), ref[:, :2].max(0)
    centres = [(x, y) for y in np.arange(lo[1], hi[1] + step, step) for x in np.arange(lo[0], hi[0] + step, step)]
    # centre outwards: lights of one target are framed on it, so the field is usually central
    centres.sort(key=lambda c: float(np.hypot(c[0] - (lo[0] + hi[0]) / 2, c[1] - (lo[1] + hi[1]) / 2)))
    out = []
    for x, y in centres:
        sel = np.linalg.norm(ref[:, :2] - (x, y), axis=1) <= radius
        if sel.sum() >= min_stars:
            out.append(ref[sel])
    return out


def link_group(ex, src, ref, transform, expect, batch=8):
    """Star link between two lights of different telescopes (src: a group's anchor light, ref:
    the reference): triangle matching in the reference windows, centre outwards in batches
    (a failing triangle search costs ~2 s), stopping at the first batch with a validated match;
    the best match of that batch refined against all reference stars. None if nothing validates."""
    if len(src) < MIN_LINK_STARS or len(ref) < MIN_LINK_STARS:
        return None
    jobs = [(src, w, transform, N_MATCH, expect) for w in star_windows(src, ref, expect)]
    batch = max(1, batch)
    for k in range(0, len(jobs), batch):
        found = [r for r in ex.map(match_to_ref, jobs[k : k + batch]) if r is not None]
        if found:
            best = max(found, key=lambda r: r[2]["matched_stars"])
            refined = guided_match((src, ref, transform, best[0], expect))
            if refined is not None:
                return refined
    return None


def match_via_candidates(job):
    """Try overlapping registered neighbours; never accept a link without star validation."""
    src, candidates, transform = job
    for anchor, ref, expect in candidates:
        result = shift_match(src, ref, transform, expect)
        if result is not None:
            return anchor, result
        for limit in (N_MATCH, min(120, N_DETECT)):
            result = match_to_ref((src, ref, transform, limit, expect))
            if result is not None:
                return anchor, result
    return None


def neighbour_candidates(i, known, tried, gid, sky, finite, depth, limit=2):
    """The next `limit` aligned lights (from `known`, ascending, not in `tried`) that pending light
    i is matched against. Own group first (another sensor's field is the slow, failing case); then,
    with both pointings finite, the nearest pointing, the lower chain depth and the nearer place in
    the session; without them, the nearer place in the session, then depth. Equal keys keep the
    lower index first (stable sort). gid, finite, depth: lists per light; sky: (n, 3) unit vectors
    or None. The distance is np.linalg.norm of sky[i] - sky[j], one candidate row at a time from
    one subtraction for all of them: the same values as a subtraction per pair, 3.6 times faster
    on a set like M 86 (1,293 lights, 607 aligned through neighbours)."""
    cand = [j for j in known if j not in tried]
    both = sky is not None and finite[i]
    diff = sky[i] - sky[cand] if both and cand else None
    gi = gid[i]
    keys = [
        (gid[j] != gi, float(np.linalg.norm(diff[r])), depth[j], abs(j - i))
        if both and finite[j]
        else (gid[j] != gi, float("inf"), abs(j - i), depth[j])
        for r, j in enumerate(cand)
    ]
    order = sorted(range(len(cand)), key=keys.__getitem__)
    return [cand[r] for r in order[:limit]]


NEIGHBOUR_ROUNDS = 6  # rounds of neighbour alignment; each tries up to two aligned lights per pending light


def register(
    stars,
    ref_idx,
    workers,
    transform,
    progress=None,
    diagnostics=None,
    pointings=None,
    pixel_scales=None,
    groups=None,
    pool=None,
):
    """dst(reference)->src 3x3 matrices, or None for frames that failed to match.
    astroalign is mostly pure Python (GIL-bound), so frames are matched in separate processes.
    pixel_scales: per light (e.g. arcsec/px from the optics) when lights come from different
    telescopes; each match must then find the scale their ratio predicts.
    groups: a group label per light (e.g. per telescope). Lights outside the reference's group
    are matched to their own group's anchor light (same sensor: reliable), the anchor is linked
    to the reference once (star_windows search), and each light is then refined directly
    against the reference through that prediction (guided_match). The anchor is the light
    nearest mid-session with a typical star count; if its link fails, the next candidates are
    tried (ANCHOR_TRIES), and if none links, the group's lights still go through neighbour
    alignment.
    pool: a ProcessPoolExecutor from start_registration_pool to use (and shut down), else one is
    started here."""
    n = len(stars)
    ps = np.ones(n) if pixel_scales is None else np.asarray(pixel_scales, dtype=float)
    if ps.shape != (n,) or not (np.isfinite(ps).all() and (ps > 0).all()):
        raise ValueError("Pixel scales must be positive, one per light")
    gid = np.zeros(n, int) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1].ravel()
    if gid.shape != (n,):
        raise ValueError("Group list must name a group for each light")
    todo = [i for i in range(n) if gid[i] == gid[ref_idx] and i != ref_idx]  # the reference's own group
    jobs = [(stars[i], stars[ref_idx], transform, N_MATCH, ps[i] / ps[ref_idx]) for i in todo]
    sky = None
    if pointings is not None:
        coordinates = np.deg2rad(np.asarray(pointings, dtype=float))
        if coordinates.shape != (len(stars), 2):
            raise ValueError("Pointing list must contain RA/Dec for each frame")
        ra, dec = coordinates.T
        sky = np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    group_of = gid.tolist()
    finite = None if sky is None else np.isfinite(sky).all(axis=1).tolist()
    with pool if pool is not None else ProcessPoolExecutor(workers) as ex:
        results = []
        for j, result in enumerate(ex.map(match_to_ref, jobs, chunksize=max(1, len(jobs) // (workers * 4))), 1):
            results.append(result)
            if progress:
                progress("registration", j, len(jobs))
        local = dict(zip(todo, results))
        out = [
            (np.eye(3), np.ones(3), {"matched_stars": len(stars[ref_idx]), "rms_px": 0.0, "scale": 1.0})
            if i == ref_idx
            else local.get(i)
            for i in range(n)
        ]
        details = [
            {"method": "reference" if i == ref_idx else "direct", "anchor": ref_idx, "depth": 0, **o[2]}
            if o
            else {"method": "failed", "reason": "No validated direct star match"}
            for i, o in enumerate(out)
        ]
        # one progress bar over the lights of every other group, so it fills once
        linking = [int(i) for i in range(n) if gid[i] != gid[ref_idx]]
        linked_so_far = 0
        if progress and linking:
            progress("linking light groups", 0, len(linking))
        for g in np.unique(gid):
            if g == gid[ref_idx]:
                continue
            members = [int(i) for i in np.flatnonzero(gid == g)]
            # anchor candidates: a typical star count (not a cloudy or empty light), mid-session first
            typical = np.median([len(stars[i]) for i in members])
            mid = members[len(members) // 2]
            candidates = sorted(
                (i for i in members if len(stars[i]) >= max(MIN_LINK_STARS, typical / 2)),
                key=lambda i: (abs(i - mid), i),
            )[:ANCHOR_TRIES]
            link, a = None, None
            for a in candidates:
                link = link_group(ex, stars[a], stars[ref_idx], transform, ps[a] / ps[ref_idx], workers)
                if link is not None:
                    break
            if link is None:  # left to neighbour alignment below
                for i in members:
                    details[i] = {
                        "method": "failed",
                        "group_anchors_tried": candidates,
                        "group_link_failed": True,
                        "reason": "No star link between this light's group and the reference",
                    }
                linked_so_far += len(members)
                if progress:
                    progress("linking light groups", linked_so_far, len(linking))
                continue
            out[a] = link
            details[a] = {
                "method": "group link",
                "anchor": ref_idx,
                "depth": 0,
                "anchors_tried": candidates.index(a) + 1,
                **link[2],
            }
            rest = [i for i in members if i != a]
            local.update(
                zip(rest, ex.map(match_to_ref, [(stars[i], stars[a], transform, N_MATCH, ps[i] / ps[a]) for i in rest]))
            )
            linked = [i for i in rest if local.get(i) is not None]
            pred = {i: local[i][0] @ link[0] for i in linked}
            refined = ex.map(
                guided_match, [(stars[i], stars[ref_idx], transform, pred[i], ps[i] / ps[ref_idx]) for i in linked]
            )
            for count, (i, result) in enumerate(zip(linked, refined), 1):
                if result is not None:
                    out[i] = result
                    details[i] = {"method": "guided", "anchor": ref_idx, "depth": 0, "group_anchor": a, **result[2]}
                else:  # keep the composed prediction (light -> group anchor -> reference)
                    out[i] = (pred[i], local[i][1] * link[1], local[i][2])
                    details[i] = {"method": "group", "anchor": a, "depth": 1, **local[i][2]}
                if progress:
                    progress("linking light groups", linked_so_far + count, len(linking))
            linked_so_far += len(members)
            if progress:
                progress("linking light groups", linked_so_far, len(linking))
        direct = sum(o is not None for o in out)
        attempted = [set([ref_idx, i]) for i in range(len(stars))]
        # A mosaic can have little or no common sky with the master reference.
        # Build an overlap graph through already registered frames, composing
        # anchor->source with master->anchor to stay in one fixed coordinate system.
        # Lights left without a direct match are aligned through overlapping lights that are
        # already aligned: one progress bar over all rounds, counting the lights aligned so far.
        left = sum(o is None for o in out)
        through_neighbours = 0
        if progress and left:
            progress("neighbour alignment", 0, left)
        rounds = 0
        for round_index in range(NEIGHBOUR_ROUNDS):
            known = [i for i, o in enumerate(out) if o is not None]
            pending = [i for i, o in enumerate(out) if o is None]
            jobs, indices = [], []
            depth = [d.get("depth", 0) for d in details]  # read for aligned lights; set after this round
            for i in pending:
                candidates = neighbour_candidates(i, known, attempted[i], group_of, sky, finite, depth)
                if not candidates:
                    continue
                attempted[i].update(candidates)
                indices.append(i)
                jobs.append((stars[i], [(j, stars[j], ps[i] / ps[j]) for j in candidates], transform))
            if not jobs:
                break
            rounds += 1
            for i, result in zip(indices, ex.map(match_via_candidates, jobs)):
                if result is not None:
                    anchor, local = result
                    matrix = local[0] @ out[anchor][0]
                    ratio = local[1] * out[anchor][1]
                    out[i] = (matrix, ratio, local[2])
                    details[i] = {
                        "method": "neighbour",
                        "anchor": anchor,
                        "depth": details[anchor]["depth"] + 1,
                        **local[2],
                    }
                    through_neighbours += 1
                    if progress:
                        progress("neighbour alignment", through_neighbours, left)
            if not any(o is None for o in out):
                break
        if progress and left:
            progress("neighbour alignment", left, left)  # finished; lights still unaligned are reported
        # Lights of another group recovered through neighbours: refine directly against the
        # reference (they overlap it, else they would not be in the group), so chained link
        # errors do not add up; a light that does not validate keeps its chained matrix.
        chained = [
            i for i in range(n) if out[i] is not None and gid[i] != gid[ref_idx] and details[i]["method"] == "neighbour"
        ]
        refined = ex.map(
            guided_match, [(stars[i], stars[ref_idx], transform, out[i][0], ps[i] / ps[ref_idx]) for i in chained]
        )
        for i, result in zip(chained, refined):
            if result is not None:
                out[i] = result
                details[i] = {
                    "method": "guided",
                    "anchor": ref_idx,
                    "depth": 0,
                    "via": details[i]["anchor"],
                    **result[2],
                }
        if diagnostics is not None:
            known = [j for j, o in enumerate(out) if o is not None]
            for i, d in enumerate(details):
                if d["method"] == "failed" and not d.get("group_link_failed"):
                    untried = sum(j not in attempted[i] for j in known)
                    d["reason"] = (
                        f"Neighbour alignment stopped after {NEIGHBOUR_ROUNDS} rounds; "
                        f"{untried} aligned lights not tried"
                        if untried
                        else "No validated star link to the reference or tested overlapping neighbours"
                    )
            diagnostics.update(
                neighbour_rounds=rounds,
                direct=direct,
                recovered=sum(o is not None for o in out) - direct,
                failed=sum(o is None for o in out),
                frames=details,
            )
    return [o[0] if o else None for o in out], [o[1] if o else None for o in out]


def output_canvas(mats, frames, w, h, framing, ref=None):
    """Return width, height and integer origin in the original reference coordinates. w/h: one
    size or per-light sizes (frame_size); "reference" framing uses the reference light's size."""
    if framing == "reference":
        rw, rh = frame_size(w, h, ref if ref is not None else frames[0])
        return rw, rh, (0, 0)
    if framing != "union":
        raise ValueError(f"Unknown framing: {framing}")
    points = []
    for i in frames:
        fw, fh = frame_size(w, h, i)
        corners = np.array([[0, 0, 1], [fw - 1, 0, 1], [0, fh - 1, 1], [fw - 1, fh - 1, 1]]).T
        p = np.linalg.inv(mats[i]) @ corners
        if not np.isfinite(p).all() or not (np.all(p[2] > 0) or np.all(p[2] < 0)):
            raise ValueError("Unbounded or invalid frame footprint")
        points.append((p[:2] / p[2]).T)
    points = np.concatenate(points)
    lower = np.floor(points.min(0)).astype(int)
    upper = np.ceil(points.max(0)).astype(int)
    return int(upper[0] - lower[0] + 1), int(upper[1] - lower[1] + 1), tuple(map(int, lower))


def frame_footprint(minv, w, h, edge=0):
    """Output-canvas polygon (4 x 2, in order) of a frame's sensor, `edge` px inside its outer
    pixel centres, mapped through the inverse of minv. With edge = sensor_edge(...), warped
    samples are valid exactly inside it."""
    lo_x, hi_x, lo_y, hi_y = edge, w - 1 - edge, edge, h - 1 - edge
    corners = np.array([[lo_x, lo_y, 1], [hi_x, lo_y, 1], [hi_x, hi_y, 1], [lo_x, hi_y, 1]], np.float64).T
    p = np.linalg.inv(minv) @ corners
    if not np.isfinite(p).all() or not (np.all(p[2] > 0) or np.all(p[2] < 0)):
        raise ValueError("Unbounded or invalid frame footprint")
    return (p[:2] / p[2]).T


def select_frames(mats, fwhm, select, groups=None):
    """Aligned frames, minus those with FWHM above median + select * sigma (0 = keep all).
    groups: a label per light (e.g. telescope): each group gets its own cutoff, since FWHM is
    measured in each sensor's own pixels through its own optics.
    Returns (retained, sharpness-dropped, cutoff or None; {group: cutoff} with groups)."""
    ok = [i for i in range(len(mats)) if mats[i] is not None]
    dropped, cut = [], None
    if select:
        label = (lambda i: None) if groups is None else (lambda i: groups[i])
        cuts = {}
        for g in dict.fromkeys(label(i) for i in ok):
            mem = [i for i in ok if label(i) == g]
            med = np.nanmedian(fwhm[mem])
            cuts[g] = med + select * 1.4826 * np.nanmedian(np.abs(fwhm[mem] - med))
        dropped = [i for i in ok if not fwhm[i] <= cuts[label(i)]]
        ok = [i for i in ok if fwhm[i] <= cuts[label(i)]]
        cut = cuts[None] if groups is None else cuts
    return ok, dropped, cut


def frame_canvas(mats, frames, w, h, framing, region=None, ref=None):
    """Output width, height and origin (reference coordinates), optionally cropped to a
    region [x, y, width, height] of that canvas."""
    width, height, origin = output_canvas(mats, frames, w, h, framing, ref)
    if region is not None:
        rx, ry, rw, rh = region
        if (
            any(int(v) != v for v in region)
            or min(rx, ry) < 0
            or min(rw, rh) < 1
            or rx + rw > width
            or ry + rh > height
        ):
            raise ValueError("Selected framing lies outside the heat-map canvas")
        origin = (origin[0] + int(rx), origin[1] + int(ry))
        width, height = int(rw), int(rh)
    return width, height, origin


def footprint_reaches(poly, width, height):
    """Whether a footprint polygon (canvas coordinates) overlaps [0, width) x [0, height) grown by
    one pixel with positive area: the test of frames_reaching."""
    for axis, bound, upper in ((0, -1, False), (0, width, True), (1, -1, False), (1, height, True)):
        inside = poly[:, axis] <= bound if upper else poly[:, axis] >= bound
        clipped = []
        for k in range(len(poly)):  # Sutherland-Hodgman against one canvas side
            a, b = poly[k - 1], poly[k]
            if inside[k - 1] != inside[k]:
                clipped.append(a + (bound - a[axis]) / (b[axis] - a[axis]) * (b - a))
            if inside[k]:
                clipped.append(b)
        poly = np.array(clipped).reshape(-1, 2)
        if len(poly) < 3:
            return False
    x, y = poly.T
    return bool(abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) > 0)


def footprints_reach(polys, width, height, margin=1e-6):
    """footprint_reaches for many footprints at once: polys (N x 4 x 2, canvas coordinates of the
    tile) -> N bools. Footprints are convex quadrilaterals, so the separating-axis test on x, y
    and their four edge normals decides overlap with positive area. Pairs within `margin` px of
    touching on some axis, and quads that are degenerate or not convex, are decided by
    footprint_reaches itself, so the result is the same as calling it for each footprint."""
    polys = np.asarray(polys, np.float64).reshape(-1, 4, 2)
    if not len(polys):
        return np.zeros(0, bool)
    rect = np.array([[-1, -1], [width, -1], [width, height], [-1, height]], np.float64)
    edges = np.roll(polys, -1, axis=1) - polys
    turns = edges[..., 0] * np.roll(edges[..., 1], -1, axis=1) - edges[..., 1] * np.roll(edges[..., 0], -1, axis=1)
    nxt = np.roll(polys, -1, axis=1)
    area = 0.5 * (polys[..., 0] * nxt[..., 1] - polys[..., 1] * nxt[..., 0]).sum(1)  # shoelace
    lengths = np.hypot(edges[..., 0], edges[..., 1])
    convex = (np.all(turns > margin, axis=1) | np.all(turns < -margin, axis=1)) & np.all(lengths > margin, axis=1)
    normals = np.stack([-edges[..., 1], edges[..., 0]], -1) / np.where(lengths > 0, lengths, 1)[..., None]
    axes = np.concatenate([np.broadcast_to(np.eye(2), (len(polys), 2, 2)), normals], axis=1)  # N x 6 x 2
    p = np.einsum("nvk,nak->nav", polys, axes)
    r = np.einsum("vk,nak->nav", rect, axes)
    overlap = np.minimum(p.max(-1) - r.min(-1), r.max(-1) - p.min(-1)).min(-1)  # > 0: no separating axis
    reach = convex & (overlap > margin) & (np.abs(area) > margin)
    separated = overlap < -margin
    for k in np.flatnonzero(~reach & ~separated):
        reach[k] = footprint_reaches(polys[k], width, height)
    return reach


def frames_reaching(mats, frames, w, h, width, height, edge=0):
    """Frames whose footprint can supply a sample to the canvas [0, width) x [0, height) (mats
    already shifted to it). A sample needs its output pixel centre inside the footprint, so a
    footprint must overlap the canvas grown by one pixel with positive area. Conservative: a
    frame passing this may still contribute nothing, never the reverse."""
    return [i for i in frames if footprint_reaches(frame_footprint(mats[i], *frame_size(w, h, i), edge), width, height)]


def retained_lights(analysis, select):
    """The lights a heat map's Overlap matching solve uses: aligned lights passing the sharpness cut
    (per group when the analysis has groups), plus the reference, the solve's anchor even when
    the cut drops it (it is not stacked then)."""
    mats = [None if m is None else np.array(m, dtype=float) for m in analysis["mats"]]
    ok, _, _ = select_frames(mats, np.array(analysis["fwhm"], dtype=float), select, analysis.get("groups"))
    return sorted(set(ok) | {int(analysis["ref_idx"])})


def frames_for_region(analysis, n, w, h, select, region, edge, framing="union", include_ref=True):
    """Lights a cached-alignment stack of `region` has to load: every retained light that can
    reach it, plus the normalization reference (spread/photometric need its statistics; cached
    overlap normalization does not). Same selection and canvas rules as stack()."""
    mats = [None if m is None else np.array(m, dtype=float) for m in analysis["mats"]]
    if len(mats) != n:
        raise ValueError("Cached registration does not match the input frame count")
    if analysis.get("sizes"):  # per-light sensor sizes recorded by the analysis
        w, h = [s[0] for s in analysis["sizes"]], [s[1] for s in analysis["sizes"]]
    ok, _, _ = select_frames(mats, np.array(analysis["fwhm"], dtype=float), select, analysis.get("groups"))
    if not ok:
        raise ValueError("No usable frames remain")
    width, height, origin = frame_canvas(mats, ok, w, h, framing, region, int(analysis["ref_idx"]))
    shift = np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1]], np.float64)
    reaching = frames_reaching([None if m is None else m @ shift for m in mats], ok, w, h, width, height, edge)
    return sorted(set(reaching) | ({int(analysis["ref_idx"])} if include_ref else set()))


def footprint_coverage(mats, frames, w, h, width, height, progress=None, edge=0, destination=None):
    """Rasterized geometric coverage; boundary pixels may differ from sampled RGB coverage.
    Each footprint is rasterized only within its bounding box (plus a guard pixel), shifted by
    whole pixels, so the counts are identical to filling a full-canvas mask per frame.
    A convex fill covers one run of pixels per row, so each footprint only marks where its runs
    start (+1) and end (-1); one running sum along each row at the end gives the counts. The
    marks wrap around in uint32, and the sums are exact because every count is small and
    non-negative. Adding each full mask instead took 17 ms per S50 Pro light (NGC 2175: 5.4 s
    for 322 lights)."""
    # An optional destination is freshly zeroed by its workspace (including new mapped files).
    coverage = np.zeros((height, width), np.uint32) if destination is None else destination
    if coverage.shape != (height, width) or coverage.dtype != np.uint32:
        raise ValueError("Invalid coverage destination")
    scratch = np.empty(0, np.uint8)
    for j, i in enumerate(frames, 1):
        polygon = np.rint(frame_footprint(mats[i], *frame_size(w, h, i), edge) * 256).astype(
            np.int64
        )  # 1/256-pixel fixed point
        lo, hi = (polygon.min(0) >> 8) - 1, (polygon.max(0) >> 8) + 2
        x0, y0 = np.maximum(lo, 0)
        x1, y1 = np.minimum(hi, (width, height))
        clipped = bool((lo < 0).any() or hi[0] > width or hi[1] > height)
        if x1 > x0 and y1 > y0:
            if scratch.size < (y1 - y0) * (x1 - x0):
                scratch = np.empty((y1 - y0) * (x1 - x0), np.uint8)
            mask = scratch[: (y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0)
            mask.fill(0)
            cv2.fillConvexPoly(
                mask, (polygon - np.array([x0, y0]) * 256).astype(np.int32), 1, lineType=cv2.LINE_8, shift=8
            )
            count = cv2.reduce(mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel()
            rows = np.flatnonzero(count)
            first = mask.argmax(1)[rows]
            if clipped:
                # a footprint clipped at the canvas edge can be filled with a gap in a row:
                # find such rows and mark each of their runs
                last = mask.shape[1] - 1 - cv2.flip(mask, 1).argmax(1)[rows]
                gap = last - first + 1 != count[rows]
            else:  # a convex fill inside the canvas: one run per row
                last = first + count[rows] - 1
                gap = np.zeros(len(rows), bool)
            for r in rows[gap]:
                steps = np.diff(mask[r].astype(np.int64), prepend=0, append=0)  # +1 at run starts, -1 after ends
                coverage[y0 + r, x0:x1] += steps[:-1].astype(np.uint32)
                if steps[-1] and x1 < width:
                    coverage[y0 + r, x1] -= np.uint32(1)
            rows, first, last = rows[~gap] + y0, first[~gap] + x0, last[~gap] + x0
            coverage[rows, first] += np.uint32(1)
            inside = last + 1 < width
            coverage[rows[inside], last[inside] + 1] -= np.uint32(1)
        if progress and (j % 8 == 0 or j == len(frames)):
            progress("frame count", j, len(frames))
    for r in range(0, height, 256):  # running sums along each row, in blocks (the destination may be on disk)
        coverage[r : r + 256] = np.cumsum(coverage[r : r + 256], axis=1, dtype=np.uint32)
    return coverage


def light_sizes(backend, frames, cached=None):
    """Per-light widths and heights {frame: px} for `frames`: from the backend for loaded lights,
    else from the analysis cache ("sizes"), else the backend's (single-sensor) size."""
    known = getattr(backend, "sizes", {})
    W, H = {}, {}
    for i in frames:
        if i in known:
            W[i], H[i] = known[i]
        elif cached is not None and cached.get("sizes"):
            W[i], H[i] = cached["sizes"][i]
        else:
            W[i], H[i] = backend.w, backend.h
    return W, H


def validate_light_statistics(backend, i, stats, *, spread=False, noise=False):
    """Reject unusable measurements explicitly; never invent a scale or drop a light."""
    try:
        med, mad, sigma = stats[:3]
        rgb = np.asarray(stats[3], np.float32) if len(stats) > 3 else None  # per-channel pixel noise
        valid = (
            np.shape(med) == (3,)
            and np.shape(mad) == (3,)
            and np.isfinite(med).all()
            and np.isfinite(mad).all()
            and (np.asarray(mad) >= 0).all()
            and np.isfinite(sigma)
            and sigma >= 0
            and (rgb is None or (rgb.shape == (3,) and np.isfinite(rgb).all() and (rgb >= 0).all()))
        )
        if spread:
            valid = valid and (np.asarray(mad) > 0).all()
        if noise:
            valid = valid and sigma > 0 and rgb is not None and (rgb > 0).all()
    except (ValueError, TypeError):
        valid = False
    if not valid:
        name = getattr(backend, "source_paths", {}).get(i, f"input light #{i + 1}")
        raise ValueError(
            f"Unusable normalization statistics for {name}: finite background levels, "
            "valid channel spreads and noise are required for the selected settings. "
            "Check this light; no light was excluded automatically."
        )


def validate_light_coefficients(backend, i, coefficients):
    a, b = coefficients[:2]
    if len(coefficients) > 2 and not (np.shape(coefficients[2]) == (3, 2) and np.isfinite(coefficients[2]).all()):
        name = getattr(backend, "source_paths", {}).get(i, f"input light #{i + 1}")
        raise ValueError(f"Unusable gradient plane for {name}. Build the heat map again; no light was excluded.")
    if (
        np.shape(a) != (3,)
        or np.shape(b) != (3,)
        or not (np.isfinite(a).all() and (a > 0).all() and np.isfinite(b).all())
    ):
        name = getattr(backend, "source_paths", {}).get(i, f"input light #{i + 1}")
        raise ValueError(
            f"Unusable normalization coefficients for {name}. Check this light and the reference; "
            "no light was excluded automatically."
        )


class SkySteps:
    """Overlap matching progress as numbered steps ("step 3 of 10"), so a bar that refills is
    explained. The number comes from the step's place in a fixed list (SKY_STEPS), never from
    counting changes: "pair fits" and "solving" alternate while the solver iterates, as often as the
    data and the 0.25 s progress throttle make them, so a step that comes back keeps the number
    reached. The number never goes back and never passes the total."""

    def __init__(self):
        self.reached = 0

    def label(self, name, stored=True):
        order = SKY_STEPS if stored else SKY_STEPS_GPU
        if name in order:
            self.reached = max(self.reached, order.index(name) + 1)
        step = min(max(self.reached, 1), len(order))
        return f"preparing overlap matching, step {step} of {len(order)}: {name}"


def overlap_normalization(
    backend,
    frames,
    ref_idx,
    mats,
    ratios,
    registration_info,
    log,
    groups=None,
    progress=None,
    check_cancel=None,
    work_dir=None,
    storage_notice=None,
    known_stats=None,
):
    """Compute overlap-based brightness normalization (overlap.py) for the retained lights, all of
    which must be loaded: a second pass builds their low-res images on the union canvas (and the
    per-light statistics for the photometric prior), then the network solve. progress: called per
    light while building and per solver iteration (stage "preparing overlap matching").
    known_stats: statistics stack() already measured, by light; only missing ones are measured."""
    import overlap

    stats = {}
    known_stats = known_stats or {}
    built_lights = [0]

    def calibrate(i):
        if check_cancel is not None:
            check_cancel()
        cfa = backend.calibrate(i)
        stats[i] = known_stats[i] if i in known_stats else backend.stats(cfa)
        validate_light_statistics(backend, i, stats[i], spread=True)
        built_lights[0] += 1
        if progress and (built_lights[0] % 8 == 0 or built_lights[0] == len(frames)):
            sky_progress("low-resolution lights", built_lights[0], len(frames))
        return cfa

    iterations = [0]

    def iteration(message):
        iterations[0] += 1
        sky_progress("solving", iterations[0], 12)

    steps = SkySteps()

    def sky_progress(name, done, total):
        if progress:
            stored = getattr(sky_context, "mode", "gpu") != "gpu"
            progress(steps.label(name, stored), done, total)

    t0 = time.perf_counter()
    device = getattr(backend, "dev", "cpu")
    W, H = light_sizes(backend, frames)
    sky_context = nullcontext(None)
    if backend.name == "gpu":
        from sky_store import SkyWorkspace

        ow, oh, _ = output_canvas(mats, frames, W, H, "union", ref_idx)
        sky_context = SkyWorkspace(
            (len(frames), 3, -(-oh // overlap.BIN), -(-ow // overlap.BIN)),
            device,
            backend.free_vram(),
            root=work_dir,
            check=check_cancel,
            progress=sky_progress,
            log=log,
            storage_notice=storage_notice,
        )
    with sky_context as sky:
        L, sky_canvas = overlap.build(calibrate, frames, mats, W, H, getattr(backend, "edge", 0), device, workspace=sky)
        built = time.perf_counter() - t0
        phot = {i: norm_coeffs(stats[i], stats[ref_idx], ratios[i]) for i in frames}
        spread = {i: norm_coeffs(stats[i], stats[ref_idx]) for i in frames}
        links = [
            (f, d["anchor"])
            for f, d in enumerate(registration_info.get("frames", []))
            if d.get("method") == "neighbour" and d.get("anchor") is not None
        ]
        result = overlap.solve(
            L,
            frames,
            ref_idx,
            {i: phot[i][0] for i in frames},
            {i: phot[i][1] for i in frames},
            {i: (float(spread[i][0][1]), float(spread[i][1][1])) for i in frames},
            getattr(backend, "edge", 0),
            links,
            log=iteration,
            groups=None if groups is None else {i: groups[i] for i in frames},
            origin=sky_canvas[2],
        )
        if sky is not None:
            result["storage"] = dict(sky.info)
        del L
    if device != "cpu":
        backend.torch.cuda.empty_cache()
    result["build_s"] = built
    log(f"[{backend.name}] Overlap matching pair fits: {result.get('pair_fit_backend', 'torch')}")
    log(
        f"[{backend.name}] overlap normalization: {len(frames)} lights, {result['pairs']} pairs, "
        f"{result['iterations']} iterations, {built:.1f}s build + {result['solve_s']:.1f}s solve; "
        f"scale from prior {len(result['scale_from_prior'])}, disconnected {len(result['disconnected'])}"
    )
    for g, term in result.get("group_terms", {}).items():
        log(
            f"[{backend.name}] light group {g}: brightness vs its star prior R {term['factor'][0]:.3f} "
            f"G {term['factor'][1]:.3f} B {term['factor'][2]:.3f} (from overlaps with the main group)"
        )
    return result


def _render_stack_tile(
    backend, bounds, here, mats, coefs, rows, weights, rej_noise, rejection, kappa, iters, kept, feather=0.0
):
    """No global output updates: an allocation failure can retry just this region."""
    x0, x1, y0, y1 = bounds
    st = backend.alloc_stack(len(here), y1 - y0, x1 - x0)
    backend.warp_tile(st, here, mats, coefs, x0, x1, y0, y1)
    fw = None
    if feather > 0:
        with backend.timer("feather"):
            fw = backend.alloc_feather(len(here), y1 - y0, x1 - x0)
            backend.feather_tile(fw, here, mats, x0, y0, feather)
    with backend.timer("coverage"):
        clip = getattr(backend, "clip", None)
        if backend.name == "gpu" and hasattr(clip, "tile_coverage") and st.is_contiguous():
            # one kernel instead of full-size isfinite/all temporaries; the same counts
            valid, contributed = clip.tile_coverage(st)
            cov, contributed = valid.cpu().numpy(), contributed.cpu().numpy()
        elif backend.name == "gpu":
            valid = backend.torch.isfinite(st).all(dim=1)
            cov = valid.sum(dim=0).cpu().numpy()
            contributed = valid.any(dim=(1, 2)).cpu().numpy()
        else:
            valid = np.isfinite(st).all(axis=3)
            cov = valid.sum(axis=0)
            contributed = valid.any(axis=(1, 2))
        del valid
    with backend.timer("integrate"):
        w_tile = None if weights is None else weights[rows]
        rej = (
            {}
            if rejection == "sigma"
            else {"rejection": rejection, "noise": None if rej_noise is None else rej_noise[rows]}
        )
        result = backend.integrate(
            st,
            kappa,
            iters,
            w_tile,
            **({"kept": True} if kept else {}),
            **({"feather": fw} if fw is not None else {}),
            **rej,
        )
    return result, cov, contributed


ALIGNED_TILE = 2048  # one light at a time needs far less memory than a stack's tile of every light


def render_light(backend, i, state, out, tile=ALIGNED_TILE):
    """Light i on the stack's canvas, as it went into the stack: warp_tile at the same absolute
    coordinates, so these are the stack's own samples before rejection. NaN where it has no data.
    state: the stack's info["light_render"]; out: a (height, width, 3) float32 array, or tensor on
    the GPU. Tiles never change pixels, so a light that runs out of GPU memory starts again with smaller ones. Returns
    the tile size that worked."""
    width, height, origin = state["width"], state["height"], state["origin"]
    mats = state["mats"]
    shift = np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1]], np.float64)
    poly = frame_footprint(mats[i] @ shift, *state["sizes"][i], backend.edge)
    lo, hi = poly.min(0), poly.max(0)
    while True:
        out[:] = np.nan
        try:
            for y0 in range(max(0, int(lo[1]) - 1) // tile * tile, min(height, int(hi[1]) + 2), tile):
                y1 = min(height, y0 + tile)
                for x0 in range(max(0, int(lo[0]) - 1) // tile * tile, min(width, int(hi[0]) + 2), tile):
                    x1 = min(width, x0 + tile)
                    if not footprint_reaches(poly - (x0, y0), x1 - x0, y1 - y0):
                        continue
                    st = backend.alloc_stack(1, y1 - y0, x1 - x0)
                    absolute = (x0 + origin[0], x1 + origin[0], y0 + origin[1], y1 + origin[1])
                    backend.warp_tile(st, [i], mats, state["coefs"], *absolute)
                    block = st[0].permute(1, 2, 0)
                    out[y0:y1, x0:x1] = block if backend.torch.is_tensor(out) else block.cpu().numpy()
                    del st, block
            return tile
        except backend.torch.OutOfMemoryError:
            st = None
        backend.torch.cuda.empty_cache()
        if tile <= 64:
            raise MemoryError("GPU memory ran out while rendering an aligned light")
        tile //= 2


def stack(
    backend,
    tile,
    reg_workers,
    transform="similarity",
    select=0.0,
    weight="none",
    norm="spread",
    kappa=3.0,
    iters=3,
    log=print,
    framing="reference",
    coverage_only=False,
    analysis_cache=None,
    on_analysis=None,
    reference_idx=None,
    region=None,
    progress=None,
    pointings=None,
    prepare_overlap=False,
    diagnostics=None,
    rejection="sigma",
    pixel_scales=None,
    reference_pool=None,
    groups=None,
    feather=0.0,
    rgb_equal=False,
    match_gradients=False,
    workspace=None,
    registration_pool=None,
):
    """pixel_scales: per light (arcsec/px) for lights from different telescopes (registration
    expects the scale ratio); reference_pool: lights the automatic reference may be chosen from
    (the main group, whose sensor grid the output uses);
    groups: a group label per light (registration matches within groups, then links them).
    prepare_overlap: compute (or reuse) overlap normalization even when not stacking with it
    (heat-map analysis caches it for later stacks); norm="overlap" stacks with it.
    diagnostics: a dict to receive extra maps (GPU: "kept", samples surviving rejection per pixel
    and channel); the stacked image is unchanged.
    registration_pool: start_registration_pool(...) started by the caller before loading the
    lights, used by register(); the caller shuts it down if stack() fails before registration.
    rejection: "sigma" (default), "winsorized" (gentler) or "noise" (each sample judged against
    its own light's noise in normalized units).
    feather: blend lights in over this many pixels from their (trimmed) sensor edges (smootherstep
    weights in the mean, feather_kernels); 0 = off. Hides steps where coverage changes.
    rgb_equal: with spread normalization, match RGB to the reference's green background
    level and spread. Stacking only; the detection/alignment cache is unaffected."""
    if type(rgb_equal) is not bool:
        raise ValueError("RGB equalization must be true or false")
    if rgb_equal and norm != "spread":
        raise ValueError("RGB equalization requires Background spread normalization")
    if type(match_gradients) is not bool:
        raise ValueError("Gradient matching must be true or false")
    if match_gradients and norm != "overlap":
        raise ValueError("Gradient matching requires Overlap matching")
    if rejection not in ("sigma", "winsorized", "noise"):
        raise ValueError(f"Unknown rejection: {rejection}")
    if not (np.isfinite(feather) and 0 <= feather <= MAX_FEATHER):
        raise ValueError(f"Feather must be 0 (off) to {MAX_FEATHER:g} pixels")
    T = backend.timer
    n = backend.n
    t_start = time.perf_counter()
    # A backend may hold only some lights (GPUBackend indices=...): those that can reach a cropped
    # output. Selection, reference and counts still describe all n lights.
    loaded = list(getattr(backend, "loaded", range(n)))
    partial = len(loaded) < n
    if partial and analysis_cache is None:
        raise ValueError("Star detection needs every light loaded")
    stars, fwhm, stats = [], [], {}
    # External storage measures statistics during analysis once; cached values avoid
    # a second full raw read when stacking. The cache is tied to the input signature.
    stat_cache = analysis_cache or {}
    cached_stats = (
        stat_cache.get("statistics", {}) if stat_cache.get("statistics_backend", "gpu") == backend.name else {}
    )
    for i in loaded:
        s = cached_stats.get(str(i))
        if s is not None and len(s) >= 4:  # a cache without per-channel noise is measured again
            try:
                stats[i] = (
                    np.asarray(s[0], np.float32),
                    np.asarray(s[1], np.float32),
                    float(s[2]),
                    np.asarray(s[3], np.float32),
                )
            except (TypeError, ValueError, IndexError) as exc:
                name = getattr(backend, "source_paths", {}).get(i, f"input light #{i + 1}")
                raise ValueError(
                    f"Unusable normalization statistics for {name}: malformed cached measurements. "
                    "Rebuild the heat map; no light was excluded automatically."
                ) from exc
            validate_light_statistics(backend, i, stats[i])
    green_ratio = None
    registration_info = {}
    need_stats = not coverage_only or getattr(backend, "storage_mode", "gpu") != "gpu"
    passes = [i for i in loaded if analysis_cache is None or (need_stats and i not in stats)]
    if coverage_only and analysis_cache is not None:
        passes = []
    reference_rule = os.environ.get("GPUCOADD_REFERENCE", "measured")  # session: the old session-order choice
    if reference_rule not in ("session", "measured"):
        raise ValueError("GPUCOADD_REFERENCE must be session or measured")
    measure_reference = analysis_cache is None and reference_idx is None and reference_rule == "measured"
    background = {}
    if progress:
        progress("frame analysis", 0, len(passes) or n)
    for k, i in enumerate(passes, 1):
        if workspace is not None:
            workspace.check_cancel()
        if backend.name == "gpu":
            from resource_guard import check_gpu

            w, h = frame_size(*light_sizes(backend, [i]), i)
            check_gpu(backend, w * h * 64 + (256 << 20), "frame analysis")
        with T("calibrate"):
            cfa = backend.calibrate(i)
        if analysis_cache is None:
            with T("detect"):
                s, f = backend.detect(cfa)
                stars.append(s)
                fwhm.append(f)
        if need_stats and i not in stats:
            with T("normalize"):
                stats[i] = backend.stats(cfa)
                validate_light_statistics(backend, i, stats[i])
        if measure_reference:
            background[i] = light_background(cfa)
        del cfa
        if progress and (k % 8 == 1 or k == len(passes)):
            progress("frame analysis", k, len(passes))
    if analysis_cache is None:
        fwhm = np.array(fwhm)
        pool = np.arange(n) if reference_pool is None else np.array(sorted(reference_pool), dtype=int)
        if not len(pool) or pool[0] < 0 or pool[-1] >= n:
            raise ValueError("Reference pool must name input lights")
        m = len(pool)
        ref_idx = int(pool[m // 2])  # mid-session: smallest field rotation to the frames at either end
        if reference_idx is not None:
            if not 0 <= reference_idx < n:
                raise ValueError("Reference frame index is outside the input list")
            ref_idx = reference_idx
        elif measure_reference:
            w0, h0 = frame_size(*light_sizes(backend, [int(pool[0])]), int(pool[0]))
            scale = None if pixel_scales is None else pixel_scales[int(pool[0])]
            field = None if not scale else min(w0, h0) * scale / 3600
            ref_idx, choice = choose_reference(pool, fwhm, stars, background, pointings, field)
            registration_info["reference_choice"] = choice
            log(f"[{backend.name}] reference chosen from measurements: {choice['summary']}")
        elif select:  # session order with the sharpness filter: sharpest of the middle half
            q = pool[m // 4 : m - m // 4]
            ref_idx = int(q[int(np.nanargmin(fwhm[q]))])
        if progress:
            progress("registration", 0, max(0, n - 1))
        with T("register (CPU)"):
            mats, ratios = register(
                stars,
                ref_idx,
                reg_workers,
                transform,
                progress=progress,
                diagnostics=registration_info,
                pointings=pointings,
                pixel_scales=pixel_scales,
                groups=groups,
                pool=registration_pool,
            )
        star_bytes = sum(s.nbytes for s in stars)
        stars.clear()
        if passes:
            del s  # detect()'s last result also holds one star array
        log(f"[{backend.name}] released {star_bytes / 2**20:.2f} MiB of star measurements after registration")
        # exactly as the cache will hold them (float64), so an Overlap matching solve now and one later
        # from the cache are identical (float32 star ratios here moved it by up to 0.16%)
        mats = [None if m is None else np.array(m, dtype=float) for m in mats]
        ratios = [None if r is None else np.array(r, dtype=float) for r in ratios]
    else:
        if len(analysis_cache["mats"]) != n:
            raise ValueError("Cached registration does not match the input frame count")
        fwhm = np.array(analysis_cache["fwhm"], dtype=float)
        ref_idx = int(analysis_cache["ref_idx"])
        mats = [None if m is None else np.array(m, dtype=float) for m in analysis_cache["mats"]]
        ratios = [None if r is None else np.array(r, dtype=float) for r in analysis_cache["ratios"]]
        registration_info = analysis_cache.get("registration", {})
        log(f"[{backend.name}] reusing cached star detection and registration")
    W, H = light_sizes(backend, range(n), analysis_cache)
    if groups is None and analysis_cache is not None:
        groups = analysis_cache.get("groups")
    if groups is not None and len(set(groups)) < 2:
        groups = None  # one group: exactly the single-instrument behaviour
    if on_analysis:
        on_analysis(
            {
                "fwhm": fwhm.tolist(),
                "ref_idx": ref_idx,
                "sizes": [[W[i], H[i]] for i in range(n)],
                **({"groups": list(groups)} if groups is not None else {}),
                "registration": registration_info,
                "mats": [None if m is None else m.tolist() for m in mats],
                "ratios": [None if r is None else r.tolist() for r in ratios],
            }
        )
        if stats:
            on_analysis(
                {
                    "statistics_backend": backend.name,
                    "statistics": {
                        str(i): [s[0].tolist(), s[1].tolist(), float(s[2]), s[3].tolist()] for i, s in stats.items()
                    },
                }
            )
    alignment_failed = [i for i in range(n) if mats[i] is None]
    ok, dropped, cut = select_frames(mats, fwhm, select, groups)
    loaded_set = set(loaded)
    active = [i for i in ok if i in loaded_set]  # retained lights actually stacked
    if not coverage_only and norm != "overlap" and ref_idx not in loaded_set:
        raise ValueError("The normalization reference light was not loaded")
    log(
        f"[{backend.name}] registered {n - len([m for m in mats if m is None])}/{n} frames ({transform}), "
        f"FWHM median {np.nanmedian(fwhm):.2f} px, retaining {len(ok)}"
        + (
            (
                f" (dropped {len(dropped)} above {cut:.2f} px)"
                if groups is None
                else f" (dropped {len(dropped)} above " + ", ".join(f"{c:.2f} px in {g}" for g, c in cut.items()) + ")"
            )
            if select
            else ""
        )
    )
    if registration_info:
        log(
            f"[{backend.name}] alignment: {registration_info.get('direct', 0)} direct, "
            f"{registration_info.get('recovered', 0)} recovered through overlap, "
            f"{len(alignment_failed)} failed; sharpness rejected {len(dropped)}"
        )
    normalization = None
    if (norm == "overlap" and not coverage_only) or prepare_overlap:
        import overlap

        cached = (analysis_cache or {}).get("normalization")
        # reused only when solved for exactly these lights: the solve is a network over them, so
        # another set (a different sharpness limit) changes the coefficients of those that remain
        if (
            cached is not None
            and cached.get("version") == overlap.VERSION
            and cached.get("reference") == ref_idx
            and sorted(cached.get("frames", [])) == sorted(set(ok) | {ref_idx})
        ):
            normalization = cached
        elif set(ok) | {ref_idx} <= loaded_set:
            # the reference anchors the solve (its statistics, scale 1) even when the sharpness cut
            # dropped it; it is still not stacked
            with T("normalize (overlap)"):
                normalization = overlap_normalization(
                    backend,
                    sorted(set(ok) | {ref_idx}),
                    ref_idx,
                    mats,
                    ratios,
                    registration_info,
                    log,
                    groups,
                    progress,
                    check_cancel=workspace.check_cancel if workspace is not None else None,
                    work_dir=workspace.work_dir if workspace is not None else None,
                    storage_notice=workspace.storage_notice if workspace is not None else None,
                    known_stats=stats,
                )
            if on_analysis:
                on_analysis({"normalization": normalization})
        else:
            raise ValueError(
                "This heat map has no usable overlap normalization (built by an older version). "
                "Build the heat map again to use Overlap matching."
            )
    if region is not None and not coverage_only and hasattr(backend, "keep_only"):
        # lights loaded only for the solve (all retained ones) that cannot reach the crop: free
        # them before stacking (they would only add NaN samples, as with crop-aware loading)
        width, height, origin = frame_canvas(mats, ok, W, H, framing, region, ref_idx)
        shift = np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1]], np.float64)
        need = set(
            frames_reaching([None if m is None else m @ shift for m in mats], ok, W, H, width, height, backend.edge)
        )
        if norm != "overlap":
            need.add(ref_idx)
        if not set(loaded) <= need:
            backend.keep_only(need)
            loaded = list(backend.loaded)
            loaded_set = set(loaded)
            active = [i for i in ok if i in loaded_set]
    if coverage_only:
        coefs = {}
    elif norm == "overlap":
        if match_gradients and "plane" not in normalization:
            raise ValueError(
                "This heat map's Overlap matching solution has no gradient planes (built by an older version). "
                "Build the heat map again to use gradient matching."
            )
        coefs = {}
        for i in active:
            scale = np.array(normalization["a"][str(i)], np.float32)
            if match_gradients:
                # the light's plane (ADU per reference pixel, per channel) and the offset consistent with
                # it, both about plane_origin; expressed about reference pixel (0, 0), the coordinates
                # the warp evaluates it at
                g = np.array(normalization["plane"][str(i)], np.float64)
                ox, oy = normalization["plane_origin"]
                b = np.array(normalization["b_plane"][str(i)], np.float64) - g[:, 0] * ox - g[:, 1] * oy
                coefs[i] = (scale, b, g)
            else:
                coefs[i] = (scale, np.array(normalization["b"][str(i)], np.float32))
        log(
            f"[{backend.name}] brightness: overlap matching ({normalization['iterations']} iterations, "
            f"{len(normalization['scale_from_prior'])} lights with scale from the star prior)"
        )
        if match_gradients:
            ext = np.array([normalization["plane_extent"][str(i)] for i in active])
            log(
                f"[{backend.name}] gradient matching: per-light planes from overlap matching "
                f"(largest excursion across a light R {ext[:, 0].max():.2f}, G {ext[:, 1].max():.2f}, "
                f"B {ext[:, 2].max():.2f} ADU; the lights' average gradient and background level are kept)"
            )
    else:
        needs_spread = lambda i: norm != "photometric" or not np.isfinite(ratios[i]).all()
        validate_light_statistics(backend, ref_idx, stats[ref_idx], spread=any(needs_spread(i) for i in active))
        for i in active:
            validate_light_statistics(
                backend, i, stats[i], spread=needs_spread(i), noise=weight == "noise" or rejection == "noise"
            )
        if rgb_equal:
            with T("normalize"):
                green_ratio = rgb_spread_ratio(backend, ref_idx)
        coefs = {
            i: norm_coeffs(
                stats[i],
                stats[ref_idx],
                ratios[i] if norm == "photometric" else None,
                rgb_equal=rgb_equal,
                green_ratio=green_ratio,
            )
            for i in active
        }
        if rgb_equal:
            log(
                f"[{backend.name}] RGB equalization: all channels target the reference green background median and spread "
                f"(reference green/red {green_ratio[0]:.3f}, green/blue {green_ratio[2]:.3f}, IKSS on demosaiced data)"
            )
    if norm == "photometric" and not coverage_only:
        tr = np.array([ratios[i][1] for i in ok])
        log(
            f"[{backend.name}] transparency (G star flux vs reference) {np.nanmin(tr):.2f}..{np.nanmax(tr):.2f}, "
            f"{int(np.isnan(tr).sum())} frames fell back to spread scaling"
        )
    weights = None
    if not coverage_only:
        for i in active:
            validate_light_coefficients(backend, i, coefs[i])
            if weight == "noise" or rejection == "noise":
                validate_light_statistics(backend, i, stats[i], noise=True)
    if weight == "noise" and not coverage_only and active:
        # 1 / noise^2 per light and channel, with each channel's pixel noise in its normalized units
        # (lights x 3); GPUCOADD_WEIGHTS=green keeps the previous single weight from the green noise
        # for every channel. Relative weights of the stacked lights; with partial loading the mean
        # (a common scale that cancels in the weighted mean) is over the loaded lights only.
        per_channel = os.environ.get("GPUCOADD_WEIGHTS", "rgb") != "green"
        if per_channel:
            nz = np.array([stats[i][3] * coefs[i][0] for i in active])
        else:
            nz = np.array([stats[i][2] * coefs[i][0][1] for i in active])
        weights = 1 / nz**2
        if not (np.isfinite(weights).all() and (weights > 0).all() and np.isfinite(weights.mean())):
            invalid = np.flatnonzero(~(np.isfinite(weights) & (weights > 0)).reshape(len(active), -1).all(1))
            bad = active[int(invalid[0])] if len(invalid) else active[0]  # finite elements may still overflow the mean
            name = getattr(backend, "source_paths", {}).get(bad, f"input light #{bad + 1}")
            raise ValueError(f"Unusable normalization noise weights for {name}. No light was excluded automatically.")
        if per_channel:  # each channel's mean over the lights, summed exactly as the single weight's mean was
            weights /= np.array([np.ascontiguousarray(weights[:, c]).mean() for c in range(3)], weights.dtype)
        else:
            weights /= weights.mean()
        effective = weights.sum(0) ** 2 / (weights**2).sum(0)
        if per_channel:
            log(
                f"[{backend.name}] noise weights per channel: R {weights[:, 0].min():.2f}..{weights[:, 0].max():.2f}, "
                f"G {weights[:, 1].min():.2f}..{weights[:, 1].max():.2f}, B {weights[:, 2].min():.2f}..{weights[:, 2].max():.2f} "
                f"(normalized noise x median: R {nz[:, 0].min() / np.median(nz[:, 0]):.2f}..{nz[:, 0].max() / np.median(nz[:, 0]):.2f}, "
                f"G {nz[:, 1].min() / np.median(nz[:, 1]):.2f}..{nz[:, 1].max() / np.median(nz[:, 1]):.2f}, "
                f"B {nz[:, 2].min() / np.median(nz[:, 2]):.2f}..{nz[:, 2].max() / np.median(nz[:, 2]):.2f}), "
                f"effective frames R {effective[0]:.0f} G {effective[1]:.0f} B {effective[2]:.0f} of {len(active)}"
            )
        else:
            log(
                f"[{backend.name}] noise weights (green, GPUCOADD_WEIGHTS) {weights.min():.2f}..{weights.max():.2f} "
                f"(normalized noise {nz.min() / np.median(nz):.2f}..{nz.max() / np.median(nz):.2f} x median), "
                f"effective frames {effective:.0f} of {len(active)}"
            )

    rej_noise = None
    if rejection == "noise" and not coverage_only:
        # each light's own pixel noise per channel, in normalized units. Green's noise scaled to
        # every channel left 1.4% more red noise than sigma clipping on SH2-142 with S50 lights
        # added (per channel: 0.5%); single-sensor sets barely change.
        rej_noise = np.stack([stats[i][3] * coefs[i][0] for i in active]).astype(np.float32)
        bad = np.flatnonzero(~(np.isfinite(rej_noise).all(axis=1) & (rej_noise > 0).all(axis=1)))
        if len(bad):
            i = active[int(bad[0])]
            name = getattr(backend, "source_paths", {}).get(i, f"input light #{i + 1}")
            raise ValueError(f"Unusable normalization rejection noise for {name}. No light was excluded automatically.")
    if not ok:
        raise ValueError("No usable frames remain")
    with T("canvas"):
        width, height, origin = frame_canvas(mats, ok, W, H, framing, region, ref_idx)
        # Geometry (which lights reach a tile, frame counts) uses matrices shifted to the canvas.
        # Warping and feathering use the original matrices at absolute reference coordinates (the
        # canvas origin is a whole pixel), so a pixel is computed identically in a crop and in the
        # full frame; gradient planes stay about reference pixel (0, 0).
        reference_mats = mats
        shift = np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1]], np.float64)
        mats = [None if m is None else m @ shift for m in mats]
        if partial and not coverage_only:
            missing = sorted(set(frames_reaching(mats, ok, W, H, width, height, backend.edge)) - loaded_set)
            if missing:
                raise ValueError(f"{len(missing)} lights that can reach the output were not loaded")
    log(
        f"[{backend.name}] {framing} canvas {width} x {height}, origin {origin}, "
        f"{width * height / (W[ref_idx] * H[ref_idx]):.2f}x reference area"
    )
    if workspace is not None:
        workspace.plan(
            width, height, coverage_only=coverage_only, kept=diagnostics is not None and backend.name == "gpu"
        )
    allocate = (
        workspace.allocate
        if workspace is not None
        else lambda name, shape, dtype, zero=False: (np.zeros if zero else np.empty)(shape, dtype)
    )
    out = None if coverage_only else allocate("stack", (height, width, 3), np.float32)
    if coverage_only:
        if progress:
            progress("frame count", 0, len(ok))
        with T("coverage geometry"):
            coverage = footprint_coverage(
                mats,
                ok,
                W,
                H,
                width,
                height,
                progress=progress,
                edge=getattr(backend, "edge", 0),
                destination=allocate("coverage", (height, width), np.uint32, zero=True)
                if workspace is not None
                else None,
            )
    else:
        coverage = allocate("coverage", (height, width), np.uint32)
    tiles = 0
    contributed = np.zeros(n, dtype=bool)
    # Each tile stacks only the lights whose (trimmed) footprint reaches it: the others would only
    # add NaN samples to allocate, warp, sort and clip (mosaics, wandering or mixed sessions).
    # GPUCOADD_TILE_LIGHTS=all stacks every light in every tile (comparisons).
    every_light = os.environ.get("GPUCOADD_TILE_LIGHTS") == "all"
    pos = {i: k for k, i in enumerate(active)}
    polys = {i: frame_footprint(mats[i], *frame_size(W, H, i), getattr(backend, "edge", 0)) for i in active}
    quads = np.array([polys[i] for i in active], np.float64).reshape(-1, 4, 2)
    active_array = np.array(active, dtype=np.int64)
    tile_requested = tile
    tile_plan = {"requested": tile_requested, "automatic": tile_requested == 0, "candidates": []}
    if not isinstance(tile, int) or tile < 0:
        raise ValueError("Tile size must be zero (automatic) or a positive integer")
    if tile == 0:
        tile = 512  # larger tiles were slower and approached paging in the item-3 trials
    tile_lights = []
    selections = {}
    if not coverage_only:
        with T("tile selection"):
            from tileplan import memory_estimate

            available = backend.free_vram() if backend.name == "gpu" else None
            while True:
                selections, tile_lights = {}, []
                for y0 in range(0, height, tile):
                    for x0 in range(0, width, tile):
                        x1, y1 = min(width, x0 + tile), min(height, y0 + tile)
                        selections[y0, x0] = (
                            active
                            if every_light
                            else active_array[footprints_reach(quads - (x0, y0), x1 - x0, y1 - y0)].tolist()
                        )
                        tile_lights.append(len(selections[y0, x0]))
                estimate = memory_estimate(
                    tile,
                    width,
                    height,
                    tile_lights,
                    debayer=getattr(backend, "debayer", "rcd"),
                    rejection=rejection,
                    feather=feather > 0,
                )
                candidate = {"tile": tile, "max_lights": max(tile_lights, default=0), **estimate}
                tile_plan["candidates"].append(candidate)
                if available is None or tile_requested != 0 or estimate["required_bytes"] <= available:
                    break
                if tile <= 64:
                    raise MemoryError(
                        "Not enough GPU memory for safe automatic tiles. Choose a smaller area or fewer lights."
                    )
                log(f"[{backend.name}] {tile}px tiles exceed the memory budget; trying {tile // 2}px")
                tile //= 2
            tile_plan.update(
                selected=tile,
                available_bytes=available,
                buffer_bytes=estimate["buffer_bytes"],
                scratch_bytes=estimate["scratch_bytes"],
                required_bytes=estimate["required_bytes"],
                retries=[],
            )
            if available is not None and tile_requested and estimate["required_bytes"] > available:
                from resource_guard import check_gpu

                check_gpu(backend, estimate["required_bytes"], "manual tile planning")
            log(
                f"[{backend.name}] {'Automatic' if tile_requested == 0 else 'Manual'} tiles: {tile}px, "
                f"up to {max(tile_lights, default=0)} lights; estimated working memory plus reserve "
                f"{estimate['required_bytes'] / 2**30:.2f} GiB after loaded lights"
            )
        if hasattr(backend, "reserve_stack") and tile_lights:
            try:
                backend.reserve_stack(max(tile_lights), min(tile, height), min(tile, width), feather=feather > 0)
            except backend.torch.OutOfMemoryError:
                # Leave allocation to the per-region retry path below.
                tile_plan["buffer_allocation_failed"] = True
                backend.reserve_stack(0, 0, 0)
                backend.torch.cuda.empty_cache()
                log(
                    "Stack buffer allocation failed; tiles will allocate directly and retry once without changing tile size"
                )
    prev_n = None
    total_tiles = ((height + tile - 1) // tile) * ((width + tile - 1) // tile)
    if progress and not coverage_only:
        progress("stacking", 0, total_tiles)
    for y0 in range(0, 0 if coverage_only else height, tile):
        y1 = min(height, y0 + tile)
        for x0 in range(0, width, tile):
            x1 = min(width, x0 + tile)
            here = selections[y0, x0]
            if backend.name == "gpu" and len(here) != prev_n:
                # integration and RCD temporaries scale with the tile's light count: release the
                # cached blocks of the previous size, or they pile up (SH2-142 8.4 -> 12.4 GiB)
                backend.torch.cuda.empty_cache()
            prev_n = len(here)
            if not here:  # no light reaches this tile: no samples (as an all-NaN stack gives)
                coverage[y0:y1, x0:x1] = 0
                out[y0:y1, x0:x1] = np.nan
                if diagnostics is not None and backend.name == "gpu":
                    if "kept" not in diagnostics:
                        diagnostics["kept"] = allocate("kept", (height, width, 3), np.uint16, zero=True)
                tiles += 1
                if progress:
                    progress("stacking", tiles, total_tiles)
                continue
            rows = [pos[i] for i in here]
            kept = diagnostics is not None and backend.name == "gpu"

            def render(bounds):
                if backend.name == "gpu":
                    from resource_guard import check_tile

                    check_tile(backend, bounds, len(here), rejection, feather > 0)
                if workspace is not None:
                    workspace.check_cancel()
                xa, xb, ya, yb = bounds
                absolute = (xa + origin[0], xb + origin[0], ya + origin[1], yb + origin[1])
                return _render_stack_tile(
                    backend,
                    absolute,
                    here,
                    reference_mats,
                    coefs,
                    rows,
                    weights,
                    rej_noise,
                    rejection,
                    kappa,
                    iters,
                    kept,
                    feather,
                )

            def release(bounds):
                backend.reserve_stack(0, 0, 0)
                backend.torch.cuda.empty_cache()
                log(
                    f"[{backend.name}] GPU allocation failed in {bounds}; retrying the same tile once; processing geometry is unchanged"
                )
                if workspace is not None:
                    workspace.check_cancel()

            from tileplan import render_same_tile

            bounds = (x0, x1, y0, y1)
            rendered = (
                render_same_tile(bounds, render, backend.torch.OutOfMemoryError, release, tile_plan["retries"])
                if backend.name == "gpu"
                else [(bounds, render(bounds))]
            )
            for (xa, xb, ya, yb), (result, cov, present) in rendered:
                coverage[ya:yb, xa:xb] = cov
                contributed[here] |= present
                if kept:
                    if "kept" not in diagnostics:
                        diagnostics["kept"] = allocate("kept", (height, width, 3), np.uint16, zero=True)
                    out[ya:yb, xa:xb], diagnostics["kept"][ya:yb, xa:xb] = result
                else:
                    out[ya:yb, xa:xb] = result
                key = f"{xb - xa}x{yb - ya}"
                used = tile_plan.setdefault("actual_tiles", {})
                used[key] = used.get(key, 0) + 1
            del rendered, result, cov, present  # output arrays already received these values
            tiles += 1
            if progress:
                progress("stacking", tiles, total_tiles)
        log(f"[{backend.name}] completed rows {y1}/{height}, {time.perf_counter() - t_start:.1f}s elapsed")
        if workspace is not None:
            workspace.flush()
    if hasattr(backend, "reserve_stack"):
        backend.reserve_stack(0, 0, 0)
    total = time.perf_counter() - t_start
    from output_store import coverage_counts

    counts = coverage_counts(coverage, len(ok))
    return (
        out,
        total,
        {
            "registered": len(ok),
            "frames": n,
            "ref_idx": ref_idx,
            "fwhm_px": fwhm.tolist(),
            "coverage": coverage,
            "coverage_kind": "geometric_preview" if coverage_only else "finite_rgb_before_rejection",
            "canvas": {
                "framing": framing,
                "width": width,
                "height": height,
                "origin_xy": list(origin),
                "area_ratio": width * height / (W[ref_idx] * H[ref_idx]),
                "tiles": tiles,
                **counts,
            },
            "dropped": dropped,
            "alignment_failed": alignment_failed,
            # what render_light needs to give a light exactly the pixels it had in this stack
            "light_render": None
            if coverage_only
            else {
                "mats": reference_mats,
                "coefs": coefs,
                "origin": list(origin),
                "width": width,
                "height": height,
                "sizes": {i: frame_size(W, H, i) for i in active},
            },
            "feather_px": float(feather),
            "rgb_equal": bool(rgb_equal and not coverage_only),
            "rgb_equal_ratio": green_ratio.tolist() if green_ratio is not None else None,
            "gradient_match": bool(match_gradients and not coverage_only),
            "registration": registration_info,
            "retained_indices": ok,
            "contributing_indices": [int(i) for i in np.flatnonzero(contributed)] if not coverage_only else None,
            "loaded_frames": len(loaded),
            "tile_plan": tile_plan,
            "tile_lights": {
                "tiles": len(tile_lights),
                "active": len(active),
                "mean": float(np.mean(tile_lights)) if tile_lights else 0.0,
                "max": int(max(tile_lights)) if tile_lights else 0,
                "empty": tile_lights.count(0),
            },
            "noise": [float(stats[i][2]) if i in stats else None for i in range(n)] if stats else [],
            "flux_ratio": [None if r is None else r.tolist() for r in ratios],
            "weights": None if weights is None else weights.tolist(),
            "weights_per_channel": weights is not None and weights.ndim == 2,
            "weighted_indices": None if weights is None else [int(i) for i in active],
            "normalization": None
            if normalization is None
            else {
                k: normalization[k]
                for k in (
                    "method",
                    "version",
                    "iterations",
                    "pairs",
                    "scale_from_prior",
                    "disconnected",
                    "solve_s",
                    "build_s",
                    "group_terms",
                    "storage",
                )
                if k in normalization
            },
        },
    )


def load_frames(paths, workers=4, ahead=8, source_guard=None):
    """Lights in order, read on `workers` threads at most `ahead` lights ahead of the consumer, so
    one that uploads each light as it arrives holds only a few in host memory (Executor.map would
    queue every file and keep each result until it is consumed: ~5.4 GiB for 351 S50 Pro lights)."""
    paths = iter(paths)
    read = read_light if source_guard is None else lambda p: source_guard.read(p, read_light)
    with ThreadPoolExecutor(workers) as ex:
        pending = deque(ex.submit(read, p) for p in itertools.islice(paths, ahead))
        while pending:
            light = pending.popleft().result()
            following = next(paths, None)
            if following is not None:
                pending.append(ex.submit(read, following))
            yield light


def autostretch(img):
    """Per-channel median/MAD midtones stretch, like a screen transfer function. Returns 0..1.
    Statistics come from covered pixels only (finite in every channel): uncovered ones, mostly
    of a sparse mosaic canvas, would otherwise collapse the median and spread. They come out
    black; an image without spread comes out black too."""
    valid = np.isfinite(img).all(-1)
    out = np.zeros(img.shape, np.float32)
    if not valid.any():
        return out
    for c in range(3):
        v = img[..., c][valid].astype(np.float64)
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        lo = med - 2.8 * 1.4826 * mad
        if not v.max() - lo > 0:
            continue
        x = np.clip((v - lo) / (v.max() - lo), 0, 1)
        m0 = max(np.median(x), 1e-6)
        t = 0.25  # target background
        mtf = m0 * (1 - t) / (m0 - 2 * t * m0 + t)
        out[..., c][valid] = ((mtf - 1) * x) / ((2 * mtf - 1) * x - mtf)
    return np.clip(out, 0, 1)


def write_png(path, image):
    """Save a PNG under any Unicode path, published only when complete. cv2.imwrite opens files
    through the Windows ANSI code page and returns False on many non-ASCII paths, so OpenCV only
    encodes here and Python writes the file."""
    done, encoded = cv2.imencode(".png", image)
    if not done:
        raise OSError(f"Could not encode PNG: {path}")
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded.tobytes())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def preview_image(img):
    """The stretched 8-bit RGB preview of a stack, a quarter of its size (at least one pixel).
    Preview pixels not fully covered (any source pixel non-finite in any channel) stay NaN for
    autostretch, so uncovered canvas never enters its statistics and comes out black."""
    size = (max(1, img.shape[1] // 4), max(1, img.shape[0] // 4))
    uncovered = (~np.isfinite(img).all(-1)).astype(np.float32)
    small = cv2.resize(np.nan_to_num(img).astype(np.float32), size, interpolation=cv2.INTER_AREA)
    # area weights times zero stay exactly zero, so this test is exact
    gaps = cv2.resize(uncovered, size, interpolation=cv2.INTER_AREA) != 0
    small[gaps] = np.nan
    return (autostretch(small) * 255).astype(np.uint8)


def preview(img, path):
    write_png(path, preview_image(img)[..., ::-1])


def coverage_preview(coverage, frames, path, predicted=False):
    """Colour-coded frame counts, with a shared 0..retained-frames scale."""
    small = cv2.resize(
        coverage.astype(np.float32),
        (max(1, coverage.shape[1] // 4), max(1, coverage.shape[0] // 4)),
        interpolation=cv2.INTER_AREA,
    )
    levels = np.uint8(np.clip(small / frames, 0, 1) * 255)
    image = cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)
    image[small == 0] = 0
    legend = np.tile(np.linspace(0, 255, image.shape[1]).astype(np.uint8), (12, 1))
    legend = cv2.applyColorMap(legend, cv2.COLORMAP_TURBO)
    label = np.zeros((28, image.shape[1], 3), np.uint8)
    title = "Predicted frames per pixel" if predicted else "Frames per pixel before rejection"
    cv2.putText(
        label,
        f"{title}: 0 (black) to {frames} (red)",
        (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    write_png(path, np.concatenate([image, legend, label], axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="folder of Light_*.fit subs")
    ap.add_argument("--frames", type=int, default=0, help="use N frames spread evenly over the session (0 = all)")
    ap.add_argument("--backends", default="gpu,cpu")
    ap.add_argument("--tile", type=int, default=0, help="output tile size in px (0: automatic)")
    # defaults are the settings that kept the most nebula detail in tests; bilinear/bilinear/0 is the fast baseline
    ap.add_argument("--interp", choices=["bilinear", "bicubic"], default="bicubic")
    ap.add_argument("--debayer", choices=["bilinear", "rcd"], default="rcd")
    ap.add_argument("--transform", choices=["similarity", "homography"], default="similarity")
    ap.add_argument(
        "--select",
        type=float,
        default=0.0,
        metavar="K",
        help="drop frames whose FWHM is over median + K sigma (0 = off)",
    )
    ap.add_argument(
        "--norm",
        choices=["spread", "photometric", "overlap"],
        default="spread",
        help="frame scaling: match background spread, star brightness, or the sky "
        "lights share (overlap matching, computed from the lights themselves)",
    )
    ap.add_argument(
        "--rejection",
        choices=["sigma", "winsorized", "noise"],
        default="sigma",
        help="sigma clipping; winsorized (gentler); noise: each sample judged against its own light's noise",
    )
    ap.add_argument(
        "--rgb-equal",
        action="store_true",
        help="match RGB backgrounds to reference green (requires --norm spread; default off)",
    )
    ap.add_argument(
        "--match-gradients",
        action="store_true",
        help="Overlap matching only: also match each light's gradient plane to the reference light (default off)",
    )
    ap.add_argument(
        "--weight",
        choices=["none", "noise"],
        default="none",
        help="noise: weight frames by 1/noise^2 in the final mean (rejection stays unweighted)",
    )
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="out")
    ap.add_argument(
        "--framing",
        choices=["reference", "union"],
        default="reference",
        help="reference: original canvas; union: include every retained frame's footprint",
    )
    ap.add_argument(
        "--coverage-only",
        action="store_true",
        help="preview retained frame footprints after registration, without demosaicing or stacking",
    )
    args = ap.parse_args()
    if args.rgb_equal and args.norm != "spread":
        ap.error("--rgb-equal requires --norm spread")
    if args.match_gradients and args.norm != "overlap":
        ap.error("--match-gradients requires --norm overlap")
    run_start = time.perf_counter()

    from lightgroups import fits_files

    paths = [str(p) for p in fits_files(args.dir)]
    if args.frames and args.frames < len(paths):
        paths = [paths[int(k)] for k in np.linspace(0, len(paths) - 1, args.frames)]
    n = len(paths)
    if not n:
        ap.error("No FITS frames (.fit, .fits, .fts) found in the input directory")
    if args.tile < 0 or args.threads <= 0 or args.frames < 0:
        ap.error("tile must be zero (automatic) or positive; threads must be positive; frames must be nonnegative")
    first = fits.getheader(paths[0])
    pedestal = float(first.get("BIAS", 0)) * Layout(first).scale
    backends = args.backends.split(",")
    if not backends or any(b not in ("cpu", "gpu") for b in backends):
        ap.error("backends must be gpu, cpu, or gpu,cpu")
    os.makedirs(args.out, exist_ok=True)
    opts = [
        args.interp if args.interp != "bilinear" else "",
        "rcd" if args.debayer == "rcd" else "",
        "homog" if args.transform == "homography" else "",
        f"sel{args.select:g}" if args.select else "",
        "phot" if args.norm == "photometric" else "ovl" if args.norm == "overlap" else "",
        "" if args.rejection == "sigma" else "rej" + args.rejection[:4],
        "wnoise" if args.weight == "noise" else "",
        "union" if args.framing == "union" else "",
        "preview" if args.coverage_only else "",
        "rgbeq" if args.rgb_equal else "",
        "grad" if args.match_gradients else "",
    ]
    tag = "_".join([f"n{n}"] + [o for o in opts if o])
    results = {
        "frames": n,
        "pedestal": pedestal,
        "tile": args.tile,
        "threads": args.threads,
        "interp": args.interp,
        "debayer": args.debayer,
        "transform": args.transform,
        "select": args.select,
        "norm": args.norm,
        "weight": args.weight,
        "rgb_equal": args.rgb_equal,
        "match_gradients": args.match_gradients,
        "framing": args.framing,
        "coverage_only": args.coverage_only,
        "runs": {},
    }
    print(f"{n} frames, pedestal {pedestal}, options: {tag}")

    raws = None
    if "cpu" in backends:
        t0 = time.perf_counter()
        raws = list(load_frames(paths))
        results["load_s"] = time.perf_counter() - t0
        print(f"loaded {n} frames into RAM in {results['load_s']:.1f}s")

    outs = {}
    canvases = {}
    for name in backends:
        backend_start = time.perf_counter()
        registration_pool = start_registration_pool(args.threads)
        try:
            if name == "gpu":
                t0 = time.perf_counter()
                shapes = [Layout(fits.getheader(p)).shape for p in paths]
                be = GPUBackend(
                    raws if raws is not None else load_frames(paths),
                    pedestal,
                    n,
                    args.interp,
                    args.debayer,
                    paths=paths,
                    sizes=shapes,
                    log=print,
                )
                load_upload = time.perf_counter() - t0
            else:
                be = CPUBackend(raws, pedestal, args.threads, args.interp, args.debayer)
                load_upload = 0.0
            img, total, info = stack(
                be,
                args.tile,
                args.threads,
                args.transform,
                args.select,
                args.weight,
                args.norm,
                framing=args.framing,
                coverage_only=args.coverage_only,
                rejection=args.rejection,
                rgb_equal=args.rgb_equal,
                match_gradients=args.match_gradients,
                registration_pool=registration_pool,
            )
        finally:
            registration_pool.shutdown(wait=False, cancel_futures=True)
        stages = dict(be.timer.t)
        stages.pop("upload", None)
        duration_key = "analysis_s" if args.coverage_only else "stack_s"
        results["runs"][name] = {
            duration_key: total,
            "load_or_upload_s": load_upload,
            "stages_s": stages,
            "registered": info["registered"],
            "reference": os.path.basename(paths[info["ref_idx"]]),
            "dropped": [os.path.basename(paths[i]) for i in info["dropped"]],
            "fwhm_px": info["fwhm_px"],
            "noise": info["noise"],
            "weights": info["weights"],
            "flux_ratio": info["flux_ratio"],
        }
        results["runs"][name]["canvas"] = info["canvas"]
        results["runs"][name]["tile_plan"] = info["tile_plan"]
        results["runs"][name]["coverage_kind"] = info["coverage_kind"]
        if name == "gpu":
            results["runs"][name]["peak_vram_reserved_gb"] = be.torch.cuda.max_memory_reserved() / 2**30
        operation = "coverage analysis" if args.coverage_only else "stack"
        print(
            f"[{name}] {operation} {total:.1f}s (+{load_upload:.1f}s {'load+upload' if raws is None else 'upload'})"
            + (f", peak VRAM {be.torch.cuda.max_memory_reserved() / 2**30:.1f} GB" if name == "gpu" else "")
        )
        for k, v in sorted(stages.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<16}{v:8.2f}s")
        save_start = time.perf_counter()
        hdr = fits.Header()
        hdr["FRAMING"] = args.framing
        hdr["ORIGINX"], hdr["ORIGINY"] = info["canvas"]["origin_xy"]
        hdr["NCOMBINE"] = info["registered"]
        hdr["RGBEQUAL"] = (info["rgb_equal"], "RGB matched to reference green background")
        hdr["GRADMTCH"] = (info["gradient_match"], "gradient planes matched (Overlap matching)")
        hdr["HISTORY"] = "ORIGINX/Y: output pixel (0,0) in reference-frame pixel coordinates"
        if img is not None:
            fits.writeto(
                os.path.join(args.out, f"stack_{name}_{tag}.fits"), np.moveaxis(img, 2, 0), hdr, overwrite=True
            )
            preview(img, os.path.join(args.out, f"stack_{name}_{tag}.png"))
        hdr["COVTYPE"] = "GEOMETRY" if args.coverage_only else "FINITE"
        hdr["HISTORY"] = (
            "Predicted geometric footprints; rasterized edges may differ from sampled counts"
            if args.coverage_only
            else "Frame counts of finite RGB inputs before rejection; not accepted sample counts"
        )
        fits.writeto(os.path.join(args.out, f"coverage_{name}_{tag}.fits"), info["coverage"], hdr, overwrite=True)
        coverage_preview(
            info["coverage"],
            info["registered"],
            os.path.join(args.out, f"coverage_{name}_{tag}.png"),
            predicted=args.coverage_only,
        )
        results["runs"][name]["save_preview_s"] = time.perf_counter() - save_start
        results["runs"][name]["wall_s"] = time.perf_counter() - backend_start
        print(f"[{name}] load through saved outputs: {results['runs'][name]['wall_s']:.1f}s")
        if img is not None:
            outs[name] = img
        canvases[name] = info["canvas"]
        if name == "gpu":
            be.torch.cuda.empty_cache()
        del be

    if "cpu" in outs and "gpu" in outs:
        a, b = outs["cpu"], outs["gpu"]
        ca, cb = canvases["cpu"], canvases["gpu"]
        ax, ay = ca["origin_xy"]
        bx, by = cb["origin_xy"]
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + a.shape[1], bx + b.shape[1]), min(ay + a.shape[0], by + b.shape[0])
        a = a[y0 - ay : y1 - ay, x0 - ax : x1 - ax]
        b = b[y0 - by : y1 - by, x0 - bx : x1 - bx]
        both = np.isfinite(a) & np.isfinite(b)
        rel = np.abs(a[both] - b[both]) / (np.abs(a[both]) + 1)
        results["cpu_vs_gpu"] = {
            "median_rel_diff": float(np.median(rel)),
            "p99_rel_diff": float(np.percentile(rel, 99)),
            "corr": float(np.corrcoef(a[both][::97], b[both][::97])[0, 1]),
        }
        print("cpu vs gpu:", results["cpu_vs_gpu"])
        c, g = results["runs"]["cpu"]["stack_s"], results["runs"]["gpu"]["stack_s"]
        print(f"speedup (stacking only): {c / g:.1f}x")

    results["total_wall_s"] = time.perf_counter() - run_start
    with open(os.path.join(args.out, f"results_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
