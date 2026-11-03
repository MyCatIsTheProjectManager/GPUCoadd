"""Optional RGB background equalization: reference-green target, cached stacks, CPU/CUDA."""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpucoadd as g


class RgbEqualTests(unittest.TestCase):
    def test_reference_green_target_and_default_unchanged(self):
        stats = (np.array([30, 40, 50], np.float32), np.array([2, 4, 8], np.float32))
        ref = (np.array([90, 100, 120], np.float32), np.array([3, 5, 7], np.float32))
        old_a = ref[1] / stats[1]
        for kwargs in ({}, {"rgb_equal": False}):
            a, b = g.norm_coeffs(stats, ref, **kwargs)
            np.testing.assert_array_equal(a, old_a)
            np.testing.assert_array_equal(b, ref[0] - stats[0] * old_a)
        a, b = g.norm_coeffs(stats, ref, rgb_equal=True)
        np.testing.assert_allclose(a * stats[0] + b, [100, 100, 100])
        np.testing.assert_allclose(a * stats[1], [5, 5, 5])
        # Reference is also transformed (red/blue do not remain identity).
        a, b = g.norm_coeffs(ref, ref, rgb_equal=True)
        np.testing.assert_allclose(a * ref[0] + b, [100, 100, 100])
        self.assertEqual((a[1], b[1]), (1, 0))

    def test_invalid_requests_fail_before_processing(self):
        for mode in ("photometric", "overlap"):
            with self.assertRaisesRegex(ValueError, "Background spread"):
                g.stack(None, 128, 1, norm=mode, rgb_equal=True)
        for value in (1, "true", None):
            with self.assertRaisesRegex(ValueError, "true or false"):
                g.stack(None, 128, 1, rgb_equal=value)
        med = np.array([10, 20, 30], np.float32)
        for mad in ([1, 0, 2], [1, np.nan, 2], [np.inf, 2, 3]):
            stats = (med, np.array(mad, np.float32))
            with self.assertRaisesRegex(ValueError, "positive channel spreads"):
                g.norm_coeffs(stats, stats, rgb_equal=True)

    def test_measured_ratio_replaces_cfa_ratio(self):
        stats = (np.array([30, 40, 50], np.float32), np.array([2, 4, 8], np.float32))
        ref = (np.array([90, 100, 120], np.float32), np.array([3, 5, 7], np.float32))
        ratio = np.array([0.6, 1.0, 0.8], np.float32)
        a, b = g.norm_coeffs(stats, ref, rgb_equal=True, green_ratio=ratio)
        np.testing.assert_allclose(a, ref[1] / stats[1] * ratio)
        np.testing.assert_allclose(a * stats[0] + b, [100, 100, 100])
        # green is exactly the plain spread normalization
        a0, b0 = g.norm_coeffs(stats, ref)
        self.assertEqual((a[1], b[1]), (a0[1], b0[1]))
        for bad in ([0.6, 1, 0], [np.nan, 1, 1]):
            with self.assertRaisesRegex(ValueError, "positive channel spreads"):
                g.norm_coeffs(stats, ref, rgb_equal=True, green_ratio=bad)

    def test_ikss(self):
        rng = np.random.default_rng(3)
        x = rng.normal(1000, 5, 200000)
        loc, scale = g.ikss(x)
        self.assertAlmostEqual(loc, 1000, delta=0.05)
        self.assertAlmostEqual(scale, 5, delta=0.05)
        # bright outliers (stars) are clipped away
        y = np.concatenate([x, rng.uniform(1100, 60000, 4000)])
        loc2, scale2 = g.ikss(y)
        self.assertAlmostEqual(loc2, 1000, delta=0.2)
        self.assertAlmostEqual(scale2, 5, delta=0.1)
        self.assertEqual(g.ikss(np.full(100, 7.0)), (7.0, 0.0))

    def test_strips_cover_rows_once(self):
        for h, w, budget in ((2160, 3840, 2e6), (161, 192, 9000), (100, 50, 1e9)):
            kept = []
            for a, b, ka, kb in g.strips(h, w, budget):
                self.assertEqual((a % 2, ka % 2), (0, 0))
                self.assertTrue(0 <= a <= ka < kb <= b <= h)
                self.assertEqual((a, b), (max(0, ka - 16), min(h, kb + 16)))
                kept.extend(range(ka, kb))
            self.assertEqual(kept, list(range(h)))

    def check_samples(self, be, other=None):
        full = be.demosaiced_samples(0, budget=1e9)
        split = be.demosaiced_samples(0, budget=9000)  # strips of a few dozen rows
        np.testing.assert_allclose(split, full, rtol=1e-6, atol=1e-3)
        if other is not None:
            np.testing.assert_allclose(other.demosaiced_samples(0, budget=1e9), full, rtol=1e-5, atol=0.02)
        return full

    def test_demosaiced_samples(self):
        rng = np.random.default_rng(5)
        h, w = 160, 192
        yy, xx = np.mgrid[:h, :w]
        sd = np.where(yy % 2 == xx % 2, 4.0, np.where(yy % 2 == 0, 8.0, 6.0))
        sky = 2000 + 300 * np.exp(-((xx - 90) ** 2 + (yy - 70) ** 2) / 900)
        raws = [(sky + rng.normal(0, 1, (h, w)) * sd).astype(np.uint16)]
        cpu = g.CPUBackend(raws, 100, 1, debayer="rcd")
        try:
            gpu = None
            import torch

            if torch.cuda.is_available():
                gpu = g.GPUBackend(iter(raws), 100, 1, "bicubic", "rcd")
            s = self.check_samples(cpu, gpu)
            # every third pixel of the whole demosaiced light, all Bayer phases
            direct = cpu.demosaic(cpu.calibrate(0), 100)[::3, ::3].reshape(-1, 3).T
            np.testing.assert_allclose(s, direct, rtol=1e-6, atol=1e-3)
            ratio = g.rgb_spread_ratio(cpu, 0)
        finally:
            cpu.pool.shutdown()
        self.assertEqual(ratio[1], 1)
        sc = [g.ikss(c)[1] for c in direct]
        np.testing.assert_allclose(ratio, [sc[1] / sc[0], 1, sc[1] / sc[2]], rtol=1e-4)

    def check_pipeline(self, gpu):
        rng = np.random.default_rng(12)
        h, w, n = 160, 192, 5
        yy, xx = np.mgrid[:h, :w]
        # Different channel levels/spreads, real spatial structure and per-light noise.
        signal = 50 * np.exp(-((xx - 90) ** 2 + (yy - 70) ** 2) / 900) + xx / 4
        gain = np.where(yy % 2 == xx % 2, 1.0, np.where(yy % 2 == 0, 2.0, 0.6))
        sky = np.where(yy % 2 == xx % 2, 300.0, np.where(yy % 2 == 0, 500.0, 150.0))
        raws = [(100 + sky + gain * signal + rng.normal(0, 5 + i, (h, w))).astype(np.uint16) for i in range(n)]
        cache = {
            "mats": [np.eye(3).tolist()] * n,
            "ratios": [[1.0, 1.0, 1.0]] * n,
            "fwhm": [3.0] * n,
            "ref_idx": 0,
            "sizes": [[w, h]] * n,
        }
        be = g.GPUBackend(iter(raws), 100, n, "bicubic", "rcd") if gpu else g.CPUBackend(raws, 100, 1)
        ref = be.stats(be.calibrate(0))
        # every light's channel goes onto the reference channel, then by the reference's
        # demosaiced green/channel spread ratio onto its green background
        scale = g.rgb_spread_ratio(be, 0)
        offset = ref[0][1] - ref[0] * scale
        kwargs = dict(
            tile=128,
            reg_workers=1,
            analysis_cache=cache,
            select=0,
            log=lambda _: None,
            kappa=1000.0,
            weight="noise",
            feather=20,
        )
        try:
            with (
                patch.object(be, "detect", side_effect=AssertionError("cached detection repeated")),
                patch("gpucoadd.register", side_effect=AssertionError("cached registration repeated")),
            ):
                plain, _, pinfo = g.stack(be, **kwargs)
                eq, _, einfo = g.stack(be, rgb_equal=True, **kwargs)
                again, _, _ = g.stack(be, rgb_equal=False, **kwargs)
                # The flag is ignored for geometric analysis, with the same cached alignment.
                _, _, heat = g.stack(
                    be, 128, 1, analysis_cache=cache, coverage_only=True, rgb_equal=True, log=lambda _: None
                )
            np.testing.assert_array_equal(again, plain)
            np.testing.assert_array_equal(pinfo["coverage"], einfo["coverage"])
            np.testing.assert_array_equal(plain[..., 1], eq[..., 1])
            np.testing.assert_allclose(eq, plain * scale + offset, rtol=3e-6, atol=0.002, equal_nan=True)
            self.assertFalse(pinfo["rgb_equal"])
            self.assertTrue(einfo["rgb_equal"])
            self.assertIsNone(pinfo["rgb_equal_ratio"])
            np.testing.assert_array_equal(einfo["rgb_equal_ratio"], scale)
            self.assertEqual(scale[1], 1)
            self.assertFalse(heat["rgb_equal"])
        finally:
            if not gpu:
                be.pool.shutdown()

    def test_cpu_cached_stack(self):
        self.check_pipeline(False)

    def test_gpu_cached_stack(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        self.check_pipeline(True)


if __name__ == "__main__":
    unittest.main()
