"""Overlap matching (overlap-based brightness normalization) against synthetic ground truth.

A known sky (broad nebula, a ramp-like nebula spanning panels, filaments, stars) is observed by
lights with known transparency, gain, sky level, light-pollution gradients, clouds and noise,
directly on a low-resolution canvas. True coefficients map each light to the reference's units:
    a_k = g_ref t_ref / (g_k t_k),   b_k = g_ref c_ref - a_k g_k c_k
Acceptance: every scale within 4x the solver's reported
uncertainty (which must therefore be honest), with the uncertainty itself small where the data
is informative (bounds per scene), and within 0.5% in the single-field transparency case; offsets
at each light's background level within 0.1 x that light's own normalized pixel noise; in
mosaics, seams (median disagreement of overlapping lights from different panels) within 0.1 x
noise, and slow drift along a chain bounded separately; nebula flux of the combined result
within 1%; cloudy lights down-weighted without moving the clean ones; disconnected and
unmeasurable lights falling back to their prior with a flag.
"""

import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))  # the reference prototype lives with the tools
sys.path.insert(0, ROOT)
import overlap  # noqa: E402  engine implementation (batched)
import overlap_norm  # noqa: E402  reference prototype

SOLVER = {"normalize": overlap_norm.normalize}  # switched per test class

NOISE = 0.5  # per low-res pixel (blurred 8 x 8 bins of real lights are ~0.3-1 ADU)


def sky(H, W, rng, ramp=60.0, blobs=4, filaments=40, stars=300):
    """Signal part (no background) of a (3, H, W) sky, and its star mask."""
    yy, xx = np.mgrid[:H, :W].astype(float)
    colour = np.array([1.0, 0.45, 0.35])[:, None, None]  # H-alpha-like: strongest in red
    s = np.zeros((H, W))
    s += ramp * xx / W  # broad ramp across all panels (looks like a gradient, but is sky)
    for _ in range(blobs):
        y, x, r = rng.uniform(0, H), rng.uniform(0, W), rng.uniform(25, 70)
        s += rng.uniform(40, 150) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * r**2))
    for _ in range(filaments):
        y, x, r = rng.uniform(0, H), rng.uniform(0, W), rng.uniform(1.5, 4)
        s += rng.uniform(10, 60) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * r**2))
    img = colour * s
    starmask = np.zeros((H, W), bool)
    for _ in range(stars):
        y, x = rng.integers(2, H - 2), rng.integers(2, W - 2)
        img[:, y, x] += rng.uniform(50, 3000)
        starmask[y - 1 : y + 2, x - 1 : x + 2] = True
    return img, starmask


class Scene:
    """Lights = rectangles on the canvas. Each: transparency t, gain g, sky level c (per channel),
    optional plane (light pollution, ADU per px), optional cloud."""

    def __init__(
        self, H, W, rects, rng, t=None, g=None, c=None, planes=None, clouds=(), ref=0, sky_kw=None, prior_error=0.03
    ):
        self.H, self.W, self.ref = H, W, ref
        n = len(rects)
        self.planes = None if planes is None else np.asarray(planes, float)
        self.signal, starmask = sky(H, W, rng, **(sky_kw or {}))
        self.t = np.asarray(t if t is not None else rng.uniform(0.5, 1.0, n), float)
        self.g = np.asarray(g if g is not None else np.ones(n), float)
        self.c = np.asarray(c if c is not None else rng.uniform(100, 300, (n, 3)), float)
        yy, xx = np.mgrid[:H, :W].astype(float)
        L = np.full((n, 3, H, W), np.nan)
        inside = np.zeros((n, H, W), bool)
        for k, (y0, y1, x0, x1) in enumerate(rects):
            m = np.zeros((H, W), bool)
            m[max(0, y0) : min(H, y1), max(0, x0) : min(W, x1)] = True
            obs = self.t[k] * self.signal + self.c[k][:, None, None]
            if planes is not None:
                gx, gy = planes[k]
                obs = obs + (gx * (xx - W / 2) + gy * (yy - H / 2))[None]
            if k in clouds:  # attenuating cloud with scattered light, over part of the light
                cy, cx = (y0 + y1) / 2, x0 + 0.3 * (x1 - x0)
                blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.25 * (x1 - x0)) ** 2))
                obs = obs * (1 - 0.5 * blob)[None] + 40 * blob[None]
            obs = self.g[k] * obs + rng.normal(0, NOISE, (3, H, W))
            L[k][:, m] = obs[:, m]
            inside[k] = m
        self.L = torch.tensor(L, dtype=torch.float32)
        self.usable = torch.tensor(inside & ~starmask[None])
        self.inside = inside
        r = self.ref
        self.a_true = (self.g[r] * self.t[r]) / (self.g * self.t)
        self.b_true = self.g[r] * self.c[r][None, :] - (self.a_true * self.g)[:, None] * self.c
        self.a_true = np.repeat(self.a_true[:, None], 3, 1)
        # star-flux-like prior with a few % error (seeing, aperture losses)
        self.prior = self.a_true * (1 + rng.normal(0, prior_error, (n, 1)))
        self.prior[r] = 1

    def run(self, **kw):
        # synthetic noise is independent per pixel: no correlation inflation, so the reported
        # uncertainty can be checked at face value
        L, U = self.L, self.usable
        if SOLVER["normalize"] is overlap.normalize and torch.cuda.is_available():
            L, U = L.cuda(), U.cuda()
        return SOLVER["normalize"](L, U, self.ref, self.prior, correlation=1.0, **kw)

    def seam_error(self, res, i, j):
        """|median(normalized j - normalized i)| on their shared usable pixels: a visible seam."""
        s = (self.usable[i] & self.usable[j]).numpy()
        Ni = res["a"][i][:, None] * self.L[i].numpy()[:, s] + res["b"][i][:, None]
        Nj = res["a"][j][:, None] * self.L[j].numpy()[:, s] + res["b"][j][:, None]
        return float(np.abs(np.median(Nj - Ni, 1)).max())

    def offset_error(self, res, k):
        """|normalized - true normalized| at light k's background (its median brightness), worst
        channel, in reference ADU: what decides whether faint sky shows a seam."""
        x = np.median(self.L[k].numpy()[:, self.inside[k]], 1)
        return float(np.abs((res["a"][k] - self.a_true[k]) * x + res["b"][k] - self.b_true[k]).max())

    def normalized(self, res, use_planes=False):
        """All lights in reference units: a L + b, or with gradient matching a L + b_plane + plane."""
        a = torch.tensor(res["a"], dtype=torch.float32)[:, :, None, None]
        if use_planes:
            yy, xx = np.mgrid[: self.H, : self.W].astype(np.float32)
            g = np.asarray(res["plane"], np.float32) / 1000  # per low-res px (the scene's pixels)
            b = torch.tensor(res["b_plane"], dtype=torch.float32)[:, :, None, None] + torch.tensor(
                g[:, :, 0, None, None] * xx + g[:, :, 1, None, None] * yy
            )
        else:
            b = torch.tensor(res["b"], dtype=torch.float32)[:, :, None, None]
        return (self.L * a + b).numpy()

    def common_plane(self):
        """The lights' mean gradient in reference units: gradient matching keeps it (mean-zero gauge)."""
        if self.planes is None:
            return 0.0
        yy, xx = np.mgrid[: self.H, : self.W].astype(float)
        scale = self.a_true[:, 0] * self.g  # each light's plane in reference units
        gx, gy = (scale[:, None] * self.planes).mean(0)  # the corrections have mean zero: the mean plane stays
        return gx * (xx - self.W / 2) + gy * (yy - self.H / 2)

    def expected_planes(self):
        """Per light and channel, the plane gradient matching should add (reference units per px)."""
        own = self.a_true[:, :, None] * self.g[:, None, None] * self.planes[:, None, :]
        return own.mean(0) - own

    def combined_flux_error(self, res, region=None, use_planes=False):
        """Relative error of the nebula flux in the mean of all normalized lights vs the truth
        (reference units), over pixels covered by at least one light (stars excluded).
        use_planes: apply the gradient-matching offsets and planes; the truth then carries the
        lights' mean plane, which the mean-zero gauge keeps."""
        N = self.normalized(res, use_planes)
        mean = np.nanmean(np.where(self.inside[:, None], N, np.nan), 0)
        r = self.ref
        plane = self.common_plane() if use_planes else 0.0
        truth = self.g[r] * (self.t[r] * self.signal + self.c[r][:, None, None]) + plane
        mask = self.inside.any(0) & self.usable.numpy().any(0)
        if region is not None:
            mask &= region
        bg = self.g[r] * self.c[r][:, None] + (plane[mask] if np.ndim(plane) else 0.0)
        f_got = (mean[:, mask] - bg).sum(1)
        f_true = (truth[:, mask] - bg).sum(1)
        return np.abs(f_got / f_true - 1)

    def pair_misfit(self, res, use_planes=False):
        """RMS disagreement of overlapping lights over all their shared usable pixels (reference
        units): what stays after normalization, including gradient mismatch across the overlap."""
        N = self.normalized(res, use_planes)
        usable = self.usable.numpy()
        total, count = 0.0, 0
        for i, j in res["pairs"]:
            s = usable[i] & usable[j]
            if s.sum() >= 100:
                d = N[j][:, s] - N[i][:, s]
                total += float((d**2).sum())
                count += d.size
        return np.sqrt(total / max(count, 1))


def dithered(y0, y1, x0, x1, rng, n, d=4):
    return [
        (
            y0 + int(rng.integers(-d, d + 1)),
            y1 + int(rng.integers(-d, d + 1)),
            x0 + int(rng.integers(-d, d + 1)),
            x1 + int(rng.integers(-d, d + 1)),
        )
        for _ in range(n)
    ]


class OverlapNormTests(unittest.TestCase):
    def check_clean(self, sc, res, lights, tol_scale=None, offset_factor=0.1, max_sigma=0.005):
        for k in lights:
            with self.subTest(light=k):
                err = np.abs(np.log(res["a"][k] / sc.a_true[k]))
                if tol_scale is not None:
                    self.assertLess(err.max(), tol_scale, f"scale error {err}")
                # 4 sigma: ~200 values are checked across the suite, 3 sigma would fail by chance
                self.assertTrue(
                    (err <= 4 * res["scale_sigma"][k] + 1e-4).all(),
                    f"scale error {err} exceeds 4 x reported sigma {res['scale_sigma'][k]}",
                )
                self.assertLess(res["scale_sigma"][k].max(), max_sigma, "uncertainty too large for this scene")
                own = NOISE * max(1.0, float(res["a"][k].max() * sc.g[k]))  # this light's normalized noise
                self.assertLess(sc.offset_error(res, k), offset_factor * own)

    def test_single_field_transparency(self):
        # 20 dithered lights of one field, transparency 0.45-1.0 (SH2-142 varied > 2x)
        rng = np.random.default_rng(1)
        rects = dithered(10, 150, 10, 230, rng, 20)
        sc = Scene(160, 240, rects, rng, t=np.r_[0.6, rng.uniform(0.45, 1.0, 19)])
        res = sc.run()
        self.check_clean(sc, res, range(20), tol_scale=0.005, max_sigma=0.002)
        self.assertLess(sc.combined_flux_error(res).max(), 0.01)

    def test_mosaic_chain_with_ramp_nebula(self):
        # 6 panels in a row (~30% overlap), 3 lights each, reference at one end; the ramp and broad
        # blobs cross panel boundaries
        rng = np.random.default_rng(2)
        rects = []
        for p in range(6):
            x0 = 10 + p * 90
            rects += dithered(10, 130, x0, x0 + 130, rng, 3)
        sc = Scene(140, 600, rects, rng, sky_kw={"ramp": 120.0, "blobs": 6})
        res = sc.run()
        self.assertEqual(res["disconnected"], [])
        self.check_clean(sc, res, range(6), max_sigma=0.004)  # panels 1-2
        # drift along the chain accumulates link by link: bounded, and reported via sigma
        self.check_clean(sc, res, range(6, 18), offset_factor=0.5, max_sigma=0.008)
        for i, j in res["pairs"]:  # seams: overlapping lights of different panels must agree
            if i // 3 != j // 3:
                with self.subTest(seam=(i, j)):
                    self.assertLess(sc.seam_error(res, i, j), 0.1 * NOISE)
        self.assertLess(sc.combined_flux_error(res).max(), 0.01)

    def test_light_pollution_gradients_are_not_scale(self):
        # per-light planes up to 0.15 ADU/px (tens of ADU across a light) on top of nebula
        rng = np.random.default_rng(3)
        rects = dithered(10, 150, 10, 230, rng, 16)
        planes = rng.uniform(-0.15, 0.15, (16, 2))
        planes[0] = 0
        sc = Scene(160, 240, rects, rng, planes=planes)
        res = sc.run()
        for k in range(16):
            with self.subTest(light=k):
                np.testing.assert_allclose(res["a"][k], sc.a_true[k], rtol=0.005)
        # gradients are not corrected (by design): offsets are judged on the footprint median,
        # the combined nebula flux must still be right
        self.assertLess(sc.combined_flux_error(res).max(), 0.01)

    def test_clouds_are_down_weighted(self):
        rng = np.random.default_rng(4)
        rects = dithered(10, 150, 10, 230, rng, 20)
        cloudy = (5, 11)
        sc = Scene(160, 240, rects, rng, clouds=cloudy)
        res = sc.run()
        self.check_clean(sc, res, [k for k in range(20) if k not in cloudy])
        for k in cloudy:
            self.assertLess(res["light_weight"][k], 0.6, f"cloudy light {k} kept full weight")
        clean = np.mean([res["light_weight"][k] for k in range(20) if k not in cloudy])
        self.assertGreater(clean, 0.8)

    def test_disconnected_light_falls_back(self):
        rng = np.random.default_rng(5)
        rects = dithered(10, 120, 10, 160, rng, 8) + [(10, 120, 300, 400)]  # no overlap
        sc = Scene(140, 420, rects, rng)
        res = sc.run(init_offset=np.zeros((9, 3)))
        self.assertEqual(res["disconnected"], [8])
        np.testing.assert_allclose(res["a"][8], sc.prior[8])  # prior kept
        self.check_clean(sc, res, range(8))

    def test_unmeasurable_scale_keeps_prior(self):
        # flat sky (no nebula, few stars): offsets measurable, scales not -> prior, flagged
        rng = np.random.default_rng(6)
        rects = dithered(10, 150, 10, 230, rng, 10)
        sc = Scene(160, 240, rects, rng, sky_kw={"ramp": 0.0, "blobs": 0, "filaments": 0, "stars": 50})
        res = sc.run()
        self.assertEqual(sorted(res["scale_from_prior"]), list(range(1, 10)))
        np.testing.assert_allclose(res["a"], sc.prior, rtol=1e-3)
        for k in range(10):  # flat sky: offset only matters, and must match
            self.assertLess(sc.offset_error(res, k), 0.1 * NOISE)

    def test_time_clustered_lights_stay_connected(self):
        # 5 groups of 6 near-identical pointings (consecutive lights), groups offset by ~20 px:
        # largest overlaps alone would only link lights within a group
        rng = np.random.default_rng(8)
        rects = []
        for gi in range(5):
            rects += dithered(10 + 4 * gi, 140 + 4 * gi, 10 + 18 * gi, 150 + 18 * gi, rng, 6, d=1)
        sc = Scene(170, 260, rects, rng)
        res = sc.run()
        self.assertEqual(res["disconnected"], [])
        self.check_clean(sc, res, range(30), tol_scale=0.005, max_sigma=0.003)

    def test_large_gain_differences(self):
        # half the lights ~10x fainter in ADU (other instrument/gain/exposure)
        rng = np.random.default_rng(7)
        rects = dithered(10, 150, 10, 230, rng, 12)
        g = np.where(np.arange(12) % 2, 0.12, 1.0)
        sc = Scene(160, 240, rects, rng, g=g, c=rng.uniform(100, 300, (12, 3)))
        res = sc.run()
        self.check_clean(sc, res, range(0, 12, 2), tol_scale=0.005, max_sigma=0.002)
        # 8x noisier lights: limited by their own noise, which the uncertainty must reflect
        self.check_clean(sc, res, range(1, 12, 2), max_sigma=0.012)


class EngineOverlapNormTests(OverlapNormTests):
    """The same scenarios and acceptance criteria against the batched engine solver (overlap.py)."""

    def setUp(self):
        SOLVER["normalize"] = overlap.normalize

    def tearDown(self):
        SOLVER["normalize"] = overlap_norm.normalize

    def test_gradient_planes_recover_light_pollution(self):
        """Gradient matching: each light's plane relative to the reference is recovered from the
        overlaps (the reference's own plane is kept, not removed); applying the planes removes the
        light-pollution mismatch that offsets alone leave, with the nebula flux still right."""
        rng = np.random.default_rng(3)
        rects = dithered(10, 150, 10, 230, rng, 16)
        planes = rng.uniform(-0.15, 0.15, (16, 2))
        planes[0] = (0.05, -0.04)  # the reference has a gradient too
        sc = Scene(160, 240, rects, rng, planes=planes)
        res = sc.run()
        np.testing.assert_allclose(res["plane"] / 1000, sc.expected_planes(), atol=0.002)  # ADU per low-res px
        np.testing.assert_allclose(res["plane"].mean(0), 0, atol=1e-9)  # the lights' mean gradient is kept
        self.assertLess(res["plane_pairs"][0]["px_residual_rms"], 3 * res["plane_pairs"][0]["px_se_median"])
        self.assertLess(sc.combined_flux_error(res, use_planes=True).max(), 0.01)
        off, on = sc.pair_misfit(res), sc.pair_misfit(res, use_planes=True)
        self.assertLess(on, 0.3 * off)
        self.assertLess(on, 3.0)  # ADU: noise, plus scale errors on the nebula

    def test_gradient_planes_leave_sky_gradients_alone(self):
        """Negative control: a steep ramp that is part of the sky, observed through differing
        transparencies, must not be taken for per-light gradients: the solved planes stay within
        the noise and the result with planes applied is as accurate as without."""
        rng = np.random.default_rng(5)
        rects = dithered(10, 150, 10, 230, rng, 16)
        sc = Scene(160, 240, rects, rng, t=np.r_[0.6, rng.uniform(0.45, 1.0, 15)], sky_kw={"ramp": 200.0})
        res = sc.run()
        self.assertLess(np.abs(res["plane"] / 1000).max(), 0.003)  # vs the sky ramp's 0.83 ADU/px
        self.assertLess(np.abs(res["plane_extent"]).max(), 2 * NOISE)
        self.assertLess(res["plane_pairs"][1]["px_rms"], 3 * res["plane_pairs"][1]["px_se_median"])  # pair planes at noise level
        self.assertLess(sc.combined_flux_error(res, use_planes=True).max(), 0.01)

    def test_gradient_planes_in_a_mosaic_chain(self):
        """Mosaic panels with a sky ramp crossing them and no per-light gradients: planes stay within
        the noise along the whole chain; with a light-pollution plane on one panel's lights, that
        panel is corrected and the others still are not."""
        rng = np.random.default_rng(2)
        rects = []
        for p in range(6):
            x0 = 10 + p * 90
            rects += dithered(10, 130, x0, x0 + 130, rng, 3)
        planes = np.zeros((18, 2))
        planes[9:12] = (0.0, 0.12)  # panel 3: a vertical light-pollution gradient
        sc = Scene(140, 600, rects, rng, planes=planes, sky_kw={"ramp": 120.0, "blobs": 6})
        res = sc.run()
        expect = sc.expected_planes()
        # The pair fit sees two lights after their scales: a scale error e on a light whose shared
        # sky is itself a plane (this ramp) cannot be told from a per-light plane, so the planes can
        # inherit at most e x (sky slope across the light). Far along the chain the scales carry
        # their accepted drift (check_clean in test_mosaic_chain_with_ramp_nebula), so the bound is
        # checked light by light; nothing beyond that error, which the scales already have, leaks.
        np.testing.assert_allclose(res["plane"] / 1000, expect, atol=0.005)
        slope = sc.g[0] * sc.t[0] * 120.0 / 600  # the sky ramp in reference ADU per px
        for k in range(18):
            cols, rows = np.flatnonzero(sc.inside[k].any(0)), np.flatnonzero(sc.inside[k].any(1))
            leak = np.abs(res["plane"][k] / 1000 - expect[k])
            leak = float((leak[:, 0] * (cols.max() - cols.min()) + leak[:, 1] * (rows.max() - rows.min())).max())
            scale_error = float(np.abs(res["a"][k] / sc.a_true[k] - 1).max())
            bound = 1.5 * scale_error * slope * (cols.max() - cols.min()) + NOISE
            with self.subTest(light=k):
                self.assertLess(leak, bound)
        self.assertLess(sc.combined_flux_error(res, use_planes=True).max(), 0.01)


class GroupTermTests(unittest.TestCase):
    """Lights of a second instrument (smaller field, ~10x fainter in ADU, ~3x noisier
    after matching) whose star-based priors share a systematic colour error (another sensor's
    stars and sky do not scale alike). Their group term must come from the overlaps."""

    def scene(self, seed=21, extra=4.0):
        rng = np.random.default_rng(seed)
        rects = dithered(10, 150, 10, 230, rng, 8) + dithered(40, 120, 70, 170, rng, 24, d=3)
        n = len(rects)
        g = np.where(np.arange(n) < 8, 1.0, 0.3)  # ~3x noisier after matching, like the S50
        sc = Scene(160, 240, rects, rng, g=g, prior_error=0.03)
        # extra noise for all: pair precision like NGC 7023 (main-main ~0.5%, cross ~5%, other ~13% in G)
        sc.L = sc.L + torch.tensor(rng.normal(0, extra, sc.L.shape), dtype=torch.float32)
        bias = np.array([0.85, 1.08, 0.93])  # common to every light of the second instrument
        sc.prior[8:] *= bias
        return sc, np.array(["main"] * 8 + ["other"] * 24)

    def run_scene(self, sc, groups):
        L, U = sc.L, sc.usable
        if torch.cuda.is_available():
            L, U = L.cuda(), U.cuda()
        return overlap.normalize(L, U, sc.ref, sc.prior, correlation=1.0, groups=groups)

    def test_group_term_corrects_shared_prior_error(self):
        sc, groups = self.scene()
        plain = self.run_scene(sc, None)
        grouped = self.run_scene(sc, list(groups))
        other = groups == "other"
        err_plain = np.abs(np.median(plain["a"][other] / sc.a_true[other], 0) - 1)
        err_grouped = np.abs(np.median(grouped["a"][other] / sc.a_true[other], 0) - 1)
        self.assertGreater(err_plain.max(), 0.1, err_plain)  # one shared scale cannot fit the colour error
        self.assertLess(err_grouped.max(), 0.035, err_grouped)
        term = grouped["group_terms"]["other"]
        truth = 1 / np.array([0.85, 1.08, 0.93])
        sigma = np.array(term["sigma"])
        self.assertTrue((sigma < 0.03).all(), sigma)
        self.assertTrue((np.abs(np.log(term["factor"] / truth)) < 3 * sigma).all(), (term, truth))  # honest
        # the main instrument's lights stay right (they were dragged before)
        np.testing.assert_allclose(np.median(grouped["a"][~other] / sc.a_true[~other], 0), 1, atol=0.01)

    def test_noise_from_pairs_uses_own_group(self):
        """Pro-like lights (variance 1) paired with 9x noisier lights: their own noise must not
        inherit the partner's variance error."""
        n = 6
        v = np.array([1, 1, 1, 9, 9, 9.0])
        pairs = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)] + [(i, j) for i in range(3) for j in range(3, 6)]
        sigma = np.array([[np.sqrt(v[i] + v[j] * (1.03 if (i < 3) != (j < 3) else 1))] * 3 for i, j in pairs])
        plain = overlap.noise_from_pairs(n, pairs, sigma, np.ones((n, 3)))
        grouped = overlap.noise_from_pairs(n, pairs, sigma, np.ones((n, 3)), groups=np.array([0, 0, 0, 1, 1, 1]))
        np.testing.assert_allclose(grouped[:, 0] ** 2, v, rtol=1e-6)
        self.assertGreater(abs(plain[0, 0] ** 2 - 1), 0.1)  # 3% of 9 is 27% of 1

    def test_single_group_is_unchanged(self):
        sc, _ = self.scene(seed=22)
        a = self.run_scene(sc, None)
        b = self.run_scene(sc, ["main"] * len(sc.prior))
        np.testing.assert_array_equal(a["a"], b["a"])
        np.testing.assert_array_equal(a["b"], b["b"])
        self.assertEqual(b["group_terms"], {})


if __name__ == "__main__":
    unittest.main()
