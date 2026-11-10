"""Overlap normalization through the real pipeline: GRBG raws of a known sky with different
transparency, sky level and pointing -> stack(prepare_overlap) caches it -> stack(norm="overlap")."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import overlap  # noqa: E402
from gpucoadd import CPUBackend, GPUBackend, frames_for_region, stack  # noqa: E402

H, W = 512, 640


def scene(seed=0, n=6, planes=None):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: H + 200, : W + 200].astype(float)
    neb = 300 * np.exp(-((yy - 330) ** 2 + (xx - 400) ** 2) / (2 * 140**2)) + 120 * xx / (W + 200)
    sky = np.stack([neb, 0.45 * neb, 0.35 * neb])  # H-alpha-like
    shifts = [(0, 0)] + [tuple(rng.integers(-60, 60, 2)) for _ in range(n - 1)]
    t = np.r_[1.0, rng.uniform(0.55, 1.0, n - 1)]  # transparency (reference 0)
    c = rng.uniform(300, 900, (n, 3))  # sky level per channel
    raws, mats = [], []
    for k, (dy, dx) in enumerate(shifts):
        crop = sky[:, 100 + dy : 100 + dy + H, 100 + dx : 100 + dx + W] * t[k] + c[k][:, None, None]
        if planes is not None:  # a per-light gradient in sensor coordinates (light pollution, vignetting)
            gx, gy = planes[k]
            crop = crop + (gx * (xx[:H, :W] - W / 2) + gy * (yy[:H, :W] - H / 2))[None]
        cfa = np.empty((H, W))
        cfa[0::2, 1::2] = crop[0, 0::2, 1::2]  # R
        cfa[0::2, 0::2] = crop[1, 0::2, 0::2]  # G
        cfa[1::2, 1::2] = crop[1, 1::2, 1::2]  # G
        cfa[1::2, 0::2] = crop[2, 1::2, 0::2]  # B
        raws.append((cfa + 100 + rng.normal(0, 3, cfa.shape)).clip(0, 65535).astype(np.uint16))
        # reference pixel (X, Y) shows sky (100 + X, 100 + Y); light pixel (x, y) shows (100 + dx + x, ...)
        mats.append(np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1.0]]))  # reference -> light pixel
    a_true = t[0] / t  # scale onto the reference
    return raws, mats, a_true, c


class OverlapEngineTests(unittest.TestCase):
    def run_case(self, backend):
        raws, mats, a_true, _ = scene()
        be = backend(raws)
        captured = {}
        kw = dict(tile=128, reg_workers=1, framing="union", select=0, log=lambda _: None)
        ratios = [np.full(3, 1 / a) * (1 + 0.04 * (-1) ** k) for k, a in enumerate(a_true)]  # star prior, 4% off
        ratios[0] = np.ones(3)
        with (
            patch.object(be, "detect", return_value=(np.ones((20, 6)), 3.0)),
            patch("gpucoadd.register", return_value=(mats, ratios)),
        ):
            _, _, info = stack(
                be, coverage_only=True, prepare_overlap=True, on_analysis=captured.update, reference_idx=0, **kw
            )
        normz = captured["normalization"]
        self.assertEqual(normz["version"], overlap.VERSION)
        self.assertEqual(normz["disconnected"], [])
        got = np.array([normz["a"][str(k)] for k in range(len(raws))])
        np.testing.assert_allclose(got, np.repeat(a_true[:, None], 3, 1), rtol=0.01)  # prior was 4% off
        self.assertEqual(info["normalization"]["method"], "overlap")
        # stack with the cached normalization: no detection/registration/statistics of the reference needed
        with (
            patch.object(be, "detect", side_effect=AssertionError("detection repeated")),
            patch("gpucoadd.register", side_effect=AssertionError("registration repeated")),
        ):
            img, _, sinfo = stack(be, norm="overlap", analysis_cache=captured, **kw)
        self.assertEqual(sinfo["normalization"]["method"], "overlap")
        self.assertTrue(np.isfinite(img).any())
        # an analysis without normalization cannot silently stack with overlap matching
        stale = {k: v for k, v in captured.items() if k != "normalization"}
        if isinstance(be, GPUBackend):
            # a corner of the union canvas that some lights reach but the reference (0) does not:
            # planned as the bridge does, without the reference, it must stack from the cache alone
            width, height = info["canvas"]["width"], info["canvas"]["height"]
            for region in (
                [0, 0, 48, 48],
                [width - 48, 0, 48, 48],
                [0, height - 48, 48, 48],
                [width - 48, height - 48, 48, 48],
            ):
                load = frames_for_region(captured, len(raws), W, H, 0, region, be.edge, include_ref=False)
                if load and 0 not in load:
                    break
            else:
                self.skipTest("no canvas corner without the reference in this scene")
            part = GPUBackend(iter([raws[i] for i in load]), 100, len(raws), indices=load)
            with self.assertRaises(ValueError):
                stack(part, norm="overlap", analysis_cache=stale, region=region, **kw)
            img2, _, _ = stack(part, norm="overlap", analysis_cache=captured, region=region, **kw)
            self.assertTrue(np.isfinite(img2).any())
        return normz

    def test_gpu(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        self.run_case(lambda raws: GPUBackend(iter(raws), 100, len(raws)))

    def test_gradient_matching_removes_a_light_gradient(self):
        """One light carries a planar gradient (light pollution). Without the option the stack steps
        where that light's footprint ends (its gradient is averaged in only where it is present); with
        gradient matching every light carries the lights' mean plane, so the stack differs from the
        stack of the same lights without the gradient (same noise draws) by one smooth plane only.
        On gradient-free lights the planes stay within the noise."""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        kw = dict(tile=128, reg_workers=1, framing="union", select=0, log=lambda _: None)
        planes = np.zeros((6, 2))
        planes[3] = (0.08, -0.05)  # about 50 ADU across light 3
        stacks, caches = {}, {}
        for name, pl in (("clean", None), ("gradient", planes)):
            raws, mats, a_true, _ = scene(seed=7, planes=pl)  # same random sequence: identical noise
            ratios = [np.full(3, 1 / a) for a in a_true]
            ratios[0] = np.ones(3)
            be = GPUBackend(iter(raws), 100, len(raws))
            captured = {}
            with (
                patch.object(be, "detect", return_value=(np.ones((20, 6)), 3.0)),
                patch("gpucoadd.register", return_value=(mats, ratios)),
            ):
                stack(be, coverage_only=True, prepare_overlap=True, on_analysis=captured.update, reference_idx=0, **kw)
            stacks[name, False], _, _ = stack(be, norm="overlap", analysis_cache=captured, **kw)
            stacks[name, True], _, info = stack(be, norm="overlap", analysis_cache=captured, match_gradients=True, **kw)
            self.assertTrue(info["gradient_match"])
            caches[name] = captured
            solved = np.array([captured["normalization"]["plane"][str(k)] for k in range(6)])  # ADU per ref px
            if pl is None:  # the sky's own ramp (0.14 ADU/px) must not leak into the planes
                self.assertLess(np.abs(solved).max() * max(H, W), 1.0)
            else:  # each light's plane in reference units, mean-zero gauge: light 3 gets -a_3 (gx, gy) less the mean
                own = a_true[:, None] * planes  # each light's gradient in reference units (6, 2)
                expect = np.repeat((own.mean(0) - own)[:, None, :], 3, 1)
                np.testing.assert_allclose(solved, expect, atol=0.004)
        ok = np.isfinite(stacks["clean", False]).all(-1)
        for key in stacks:
            ok &= np.isfinite(stacks[key]).all(-1)
        yy, xx = np.mgrid[: ok.shape[0], : ok.shape[1]]
        A = np.c_[xx[ok], yy[ok], np.ones(ok.sum())]

        def beyond_one_plane(dimg):  # |residual| after removing one plane per channel
            return np.abs(np.stack([dimg[..., c][ok] - A @ np.linalg.lstsq(A, dimg[..., c][ok], rcond=None)[0] for c in range(3)], 1))

        r_on = beyond_one_plane(stacks["gradient", True] - stacks["clean", False])
        r_off = beyond_one_plane(stacks["gradient", False] - stacks["clean", False])
        self.assertLess(np.percentile(r_on, 99), 2.0)  # one smooth plane and nothing else (raw noise 3 ADU, six lights)
        self.assertGreater(np.percentile(r_off, 99), 3 * np.percentile(r_on, 99))  # the step where light 3 ends
        # gradient-free lights: the option changes nothing beyond the plane fit's own noise
        self.assertLess(np.percentile(np.abs(stacks["clean", True] - stacks["clean", False])[ok], 99.9), 1.0)
        with self.assertRaises(ValueError):
            stack(be, norm="spread", analysis_cache=caches["gradient"], match_gradients=True, **kw)
        stale = dict(caches["gradient"], normalization={k: v for k, v in caches["gradient"]["normalization"].items() if k != "plane"})
        with self.assertRaises(ValueError):
            stack(be, norm="overlap", analysis_cache=stale, match_gradients=True, **kw)

    def test_deferred_solve_equals_heat_map_solve(self):
        """A heat map built without Overlap matching, then a crop stacked with it: the stack loads every
        retained light, solves exactly as a heat map would (same lights, reference, alignment),
        reports the solution for the cache, frees the lights the crop does not need, and gives the
        same image as the normal path (heat map with the solution, crop-aware loading)."""
        import torch
        from gpucoadd import retained_lights

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        raws, mats, a_true, _ = scene(seed=3)
        n = len(raws)
        kw = dict(tile=128, reg_workers=1, framing="union", select=0, log=lambda _: None)
        ratios = [np.full(3, 1 / a) for a in a_true]
        ratios[0] = np.ones(3)
        heat, plain = {}, {}
        be = GPUBackend(iter(raws), 100, n)
        with (
            patch.object(be, "detect", return_value=(np.ones((20, 6)), 3.0)),
            patch("gpucoadd.register", return_value=(mats, ratios)),
        ):
            _, _, info = stack(
                be, coverage_only=True, prepare_overlap=True, on_analysis=heat.update, reference_idx=0, **kw
            )
            stack(be, coverage_only=True, on_analysis=plain.update, reference_idx=0, **kw)
        self.assertNotIn("normalization", plain)  # a heat map without Overlap matching
        region = [0, 0, 60, 60]  # a corner only some lights reach
        # normal path: solution in the heat map, crop-aware loading
        load = frames_for_region(heat, n, W, H, 0, region, be.edge, include_ref=False)
        self.assertLess(len(load), n)
        ref_img, _, _ = stack(
            GPUBackend(iter([raws[i] for i in load]), 100, n, indices=load),
            norm="overlap",
            analysis_cache=heat,
            region=region,
            **kw,
        )
        # deferred path: every retained light loaded, solved during the stack
        keep = retained_lights(plain, 0)
        self.assertEqual(keep, list(range(n)))
        late = {}
        part = GPUBackend(iter([raws[i] for i in keep]), 100, n, indices=keep)
        stages = []
        img, _, sinfo = stack(
            part,
            norm="overlap",
            analysis_cache=dict(plain),
            region=region,
            on_analysis=late.update,
            progress=lambda stage, done, total: stages.append(stage),
            **kw,
        )
        for key in ("a", "b", "scale_sigma"):
            self.assertEqual(late["normalization"][key], heat["normalization"][key], key)
        self.assert_numbered_steps(stages)
        self.assertEqual(sorted(part.loaded), sorted(load))  # lights that cannot reach the crop were freed
        np.testing.assert_array_equal(np.isnan(img), np.isnan(ref_img))
        np.testing.assert_allclose(img, ref_img, rtol=1e-6, atol=1e-4, equal_nan=True)

    def assert_numbered_steps(self, stages):
        """Overlap matching reports "step k of n": one total for the run, k starting at 1, never
        going back, never past n, and reaching n."""
        import re

        steps = [re.match(r"preparing overlap matching, step (\d+) of (\d+): (.+)", s) for s in stages]
        steps = [m for m in steps if m]
        self.assertTrue(steps)
        totals = {int(m[2]) for m in steps}
        self.assertEqual(len(totals), 1, totals)
        total = totals.pop()
        numbers = [int(m[1]) for m in steps]
        self.assertEqual(numbers, sorted(numbers))
        self.assertEqual((numbers[0], numbers[-1]), (1, total))
        self.assertEqual(steps[0][3], "low-resolution lights")
        self.step_count = total

    def test_step_numbers_do_not_depend_on_timing(self):
        """However often "pair fits" and "solving" alternate (data and the progress throttle decide),
        the number follows the fixed step list: it never goes back and never passes the total."""
        from gpucoadd import SKY_STEPS, SkySteps

        rng = np.random.default_rng(4)
        head = list(SKY_STEPS[:7])
        for trial in range(200):
            loop = [str(rng.choice(["pair fits", "solving"])) for _ in range(rng.integers(1, 30))]
            refits = [str(rng.choice(["pair fits", "solving"])) for _ in range(rng.integers(0, 5))]
            tail = ["uncertainty floor"] + refits
            steps = SkySteps()
            numbers = [int(steps.label(name).split()[4]) for name in head + loop + tail]
            with self.subTest(trial=trial):
                self.assertEqual(numbers, sorted(numbers))
                self.assertLessEqual(max(numbers), len(SKY_STEPS))
                self.assertEqual(numbers[-1], len(SKY_STEPS))
                self.assertTrue(all(steps.label(n).endswith(f"of {len(SKY_STEPS)}: {n}") for n in loop))
        gpu = SkySteps()
        self.assertEqual(
            gpu.label("low-resolution lights", stored=False),
            "preparing overlap matching, step 1 of 2: low-resolution lights",
        )
        self.assertEqual(gpu.label("solving", stored=False), "preparing overlap matching, step 2 of 2: solving")

    def test_numbered_steps_with_ram_storage(self):
        """With Overlap matching data in RAM the solve has 10 numbered steps; the dialog shows them."""
        with (
            patch.dict(os.environ, {"GPUCOADD_SKY_STORAGE": "ram"}),
            patch("sky_store.available_ram", return_value=64 << 30),
            patch("resource_guard.available_ram", return_value=64 << 30),
        ):
            self.test_deferred_solve_equals_heat_map_solve()
        self.assertEqual(self.step_count, 10)

    def test_reference_rejected_for_sharpness(self):
        """A blurry reference (chosen by the user) that the sharpness cut drops: Overlap matching still
        anchors on it (statistics and scale 1) without stacking it, at heat-map time and when the
        solve is deferred to a stack. Before: KeyError on the reference's statistics."""
        import torch
        from gpucoadd import retained_lights

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        raws, mats, a_true, _ = scene(seed=5)
        n = len(raws)
        fwhm = [9.0, 3.0, 3.1, 2.9, 3.05, 2.95]  # light 0, the reference, is blurry
        ratios = [np.full(3, 1 / a) for a in a_true]
        ratios[0] = np.ones(3)
        kw = dict(tile=128, reg_workers=1, framing="union", select=2.5, log=lambda _: None)
        detect = lambda fw: iter([(np.ones((20, 6)), f) for f in fw])
        heat, plain = {}, {}
        be = GPUBackend(iter(raws), 100, n)
        with (
            patch.object(be, "detect", side_effect=detect(fwhm)),
            patch("gpucoadd.register", return_value=(mats, ratios)),
        ):
            _, _, info = stack(
                be, coverage_only=True, prepare_overlap=True, on_analysis=heat.update, reference_idx=0, **kw
            )
        self.assertEqual(info["dropped"], [0])
        self.assertEqual(heat["normalization"]["reference"], 0)
        got = np.array([heat["normalization"]["a"][str(k)] for k in range(1, n)])
        np.testing.assert_allclose(got, np.repeat(a_true[1:, None], 3, 1), rtol=0.02)
        # the heat map's lights for a deferred solve include the reference
        be2 = GPUBackend(iter(raws), 100, n)
        with (
            patch.object(be2, "detect", side_effect=detect(fwhm)),
            patch("gpucoadd.register", return_value=(mats, ratios)),
        ):
            stack(be2, coverage_only=True, on_analysis=plain.update, reference_idx=0, **kw)
        self.assertIn(0, retained_lights(plain, 2.5))
        keep = retained_lights(plain, 2.5)
        late = {}
        img, _, sinfo = stack(
            GPUBackend(iter([raws[i] for i in keep]), 100, n, indices=keep),
            norm="overlap",
            analysis_cache=dict(plain),
            region=[0, 0, 60, 60],
            on_analysis=late.update,
            **kw,
        )
        for key in ("a", "b"):
            self.assertEqual(late["normalization"][key], heat["normalization"][key], key)
        self.assertNotIn(0, sinfo["contributing_indices"])  # the reference anchors, it is not stacked

    def test_cpu(self):
        normz = self.run_case(lambda raws: CPUBackend(raws, 100, 2))
        self.assertGreater(normz["pairs"], 0)


class NoiseFloorTests(unittest.TestCase):
    def test_matches_per_channel_solve(self):
        """own_noise_floor solves the three channels together; same result as one plane fit per
        channel (to float64 rounding), inf for lights with fewer than 100 usable pixels."""
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator(device=dev).manual_seed(4)
        n, h, w = 5, 40, 60
        yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
        L = (torch.randn(n, 3, h, w, device=dev, generator=g) * 0.01 + 1 + 1e-3 * xx + 2e-3 * yy).float()
        usable = torch.rand(n, h, w, device=dev, generator=g) < 0.7
        usable[2] = False
        usable[2, :9, :11] = True  # 99 pixels
        noise = np.random.default_rng(4).uniform(0.003, 0.012, (n, 3))
        out = overlap.own_noise_floor(L, usable, noise, max_pixels=1000)
        gen = torch.Generator(device=dev).manual_seed(1)
        for k in range(n):
            cnt = int(usable[k].sum())
            if cnt < 100:
                self.assertTrue(np.isinf(out[k]).all())
                continue
            sel = torch.nonzero(usable[k].reshape(-1))[:, 0]
            if sel.numel() > 1000:
                sel = sel[torch.randperm(sel.numel(), generator=gen, device=dev)[:1000]]
            Q = torch.stack([xx.reshape(-1)[sel], yy.reshape(-1)[sel], torch.ones(sel.numel(), device=dev)], 1).double()
            for c in range(3):
                v = L[k, c].reshape(-1)[sel].double()
                resid = v - Q @ torch.linalg.lstsq(Q, v[:, None]).solution[:, 0]
                var_sig = max(float(resid.var()) - noise[k, c] ** 2, 1e-30)
                expected = overlap.CORRELATION * noise[k, c] / np.sqrt(cnt * var_sig)
                self.assertAlmostEqual(out[k, c] / expected, 1.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
