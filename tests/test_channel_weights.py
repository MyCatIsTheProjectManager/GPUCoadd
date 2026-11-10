"""Per-channel noise weights: per-channel pixel noise in the statistics, (lights x 3)
weights through both integrators, and the stack's green channel unchanged against the previous
single green weight (GPUCOADD_WEIGHTS=green)."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpucoadd import CPUBackend, GPUBackend, stack  # noqa: E402

H, W = 256, 320


def remosaic(rgb):
    """(3, H, W) -> GRBG CFA."""
    cfa = np.empty(rgb.shape[1:])
    cfa[0::2, 1::2] = rgb[0, 0::2, 1::2]
    cfa[0::2, 0::2] = rgb[1, 0::2, 0::2]
    cfa[1::2, 1::2] = rgb[1, 1::2, 1::2]
    cfa[1::2, 0::2] = rgb[2, 1::2, 0::2]
    return cfa


def lights(seed=0, n=6, noise=((2.0, 1.0, 4.0),) * 6):
    """Lights of one sky with per-light, per-channel Gaussian noise (R, G, B ADU) and small shifts."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: H + 80, : W + 80].astype(float)
    neb = 200 * np.exp(-((yy - 170) ** 2 + (xx - 200) ** 2) / (2 * 60**2)) + 300
    sky = np.stack([neb, 0.6 * neb, 0.4 * neb])
    shifts = [(0, 0)] + [tuple(rng.integers(-20, 20, 2)) for _ in range(n - 1)]
    raws, mats = [], []
    for k, (dy, dx) in enumerate(shifts):
        crop = sky[:, 40 + dy : 40 + dy + H, 40 + dx : 40 + dx + W]
        sig = np.asarray(noise[k], float)[:, None, None]
        cfa = remosaic(crop + rng.normal(0, 1, crop.shape) * sig)
        raws.append((cfa + 100).clip(0, 65535).astype(np.uint16))
        mats.append(np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1.0]]))
    return raws, mats


class ChannelNoiseTests(unittest.TestCase):
    def test_stats_measure_each_channel(self):
        raws, _ = lights(noise=((2.0, 1.0, 4.0),) * 6)
        be = CPUBackend(raws, 100, 2)
        med, mad, green, rgb = be.stats(be.calibrate(0))
        self.assertEqual(float(green), float(rgb[1]))
        np.testing.assert_allclose(rgb, [2.0, 1.0, 4.0], rtol=0.08)
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            gb = GPUBackend(iter(raws), 100, len(raws))
            _, _, green_g, rgb_g = gb.stats(gb.calibrate(0))
            np.testing.assert_allclose(rgb_g, rgb, rtol=0.02)
            self.assertEqual(float(green_g), float(rgb_g[1]))


class ChannelWeightIntegrationTests(unittest.TestCase):
    """(lights x 3) weights give, per channel, exactly what that channel's (lights,) weights give."""

    def _samples(self, rng, n, rows, cols):
        x = rng.normal(100, 5, (n, rows, cols, 3)).astype(np.float32)
        x[rng.random((n, rows, cols)) < 0.1] = np.nan
        x[1, :, :, 0] += 40  # outliers in red only
        return x

    def test_weighted_mean_against_analytic_truth(self):
        """Independent weighted-mean formula, with missing pixels and channel-specific weights.

        A very wide rejection threshold isolates averaging from clipping decisions.
        """
        import torch

        rng = np.random.default_rng(82)
        x = rng.normal(100, 2, (12, 8, 9, 3)).astype(np.float32)
        x[:3, :2, :, 0] = np.nan
        w = rng.uniform(0.1, 2.0, (12, 3)).astype(np.float32)
        valid = np.isfinite(x)
        weights = w[:, None, None, :].astype(np.float64)
        truth = np.sum(np.where(valid, x, 0) * weights, axis=0) / np.sum(valid * weights, axis=0)
        cpu = CPUBackend([np.full((8, 8), 200, np.uint16)] * 12, 0, 2)
        try:
            for mode in ("sigma", "winsorized", "noise"):
                kwargs = dict(rejection=mode, noise=np.ones((12, 3), np.float32))
                got = cpu.integrate(x, 100.0, 1, w, **kwargs)
                np.testing.assert_allclose(got, truth, rtol=2e-6)
                if torch.cuda.is_available():
                    gpu = GPUBackend.__new__(GPUBackend)
                    gpu.torch, gpu.dev = torch, torch.device("cuda")
                    import integrate_kernels, sort_kernels

                    gpu.clip, gpu.sort = integrate_kernels, sort_kernels
                    got = gpu.integrate(torch.from_numpy(np.moveaxis(x, 3, 1).copy()).cuda(), 100.0, 1, w, **kwargs)
                    np.testing.assert_allclose(got, truth, rtol=2e-6)
        finally:
            cpu.pool.shutdown()

    def test_cpu_integrator(self):
        rng = np.random.default_rng(3)
        n, rows, cols = 12, 16, 20
        be = CPUBackend([np.full((8, 8), 200, np.uint16)] * n, 0, 2)
        x = self._samples(rng, n, rows, cols)
        w = (rng.random((n, 3)) + 0.3).astype(np.float32)
        feather = (rng.random((n, rows, cols)) * 0.9 + 0.1).astype(np.float32)
        noise = (rng.random((n, 3)) + 0.5).astype(np.float32)
        for rejection in ("sigma", "winsorized", "noise"):
            for fw in (None, feather):
                kw = dict(rejection=rejection, noise=noise, feather=fw)
                got = be.integrate(x.copy(), 3.0, 3, w, **kw)
                for c in range(3):
                    want = be.integrate(x.copy(), 3.0, 3, w[:, c].copy(), **kw)
                    with self.subTest(rejection=rejection, feather=fw is not None, channel=c):
                        self.assertTrue(np.array_equal(got[..., c], want[..., c], equal_nan=True))

    def test_gpu_integrator(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(4)
        n, rows, cols = 40, 24, 28
        be = GPUBackend(iter([np.full((8, 8), 200, np.uint16)] * n), 0, n)
        x = torch.from_numpy(np.moveaxis(self._samples(rng, n, rows, cols), 3, 1).copy()).cuda()
        w = (rng.random((n, 3)) + 0.3).astype(np.float32)
        feather = torch.from_numpy((rng.random((n, rows, cols)) * 0.9 + 0.1).astype(np.float32)).cuda()
        noise = (rng.random((n, 3)) + 0.5).astype(np.float32)
        for rejection in ("sigma", "winsorized", "noise"):
            for fw in (None, feather):
                kw = dict(rejection=rejection, noise=noise, feather=fw)
                got, kept = be.integrate(x.clone(), 3.0, 3, w, kept=True, **kw)
                for c in range(3):
                    want, kept_c = be.integrate(x.clone(), 3.0, 3, w[:, c].copy(), kept=True, **kw)
                    with self.subTest(rejection=rejection, feather=fw is not None, channel=c):
                        self.assertTrue(np.array_equal(got[..., c], want[..., c], equal_nan=True))
                        self.assertTrue(np.array_equal(kept[..., c], kept_c[..., c]))


class ChannelWeightStackTests(unittest.TestCase):
    def test_green_unchanged_and_red_reweighted(self):
        """Light 2 is noisy in red only: with per-channel weights its red weight drops while green
        and blue keep their weights; the green channel of the stack is bit-identical to the
        previous single green weight (GPUCOADD_WEIGHTS=green)."""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        noise = [(2.0, 1.0, 2.0)] * 6
        noise[2] = (8.0, 1.0, 2.0)
        raws, mats = lights(seed=5, noise=noise)
        n = len(raws)
        ratios = [np.ones(3)] * n
        be = GPUBackend(iter(raws), 100, n)
        kw = dict(tile=128, reg_workers=1, framing="union", select=0, log=lambda _: None)
        captured = {}
        with (
            patch.object(be, "detect", return_value=(np.ones((20, 6)), 3.0)),
            patch("gpucoadd.register", return_value=(mats, ratios)),
        ):
            stack(be, coverage_only=True, on_analysis=captured.update, reference_idx=0, **kw)
        with patch.dict(os.environ, {"GPUCOADD_WEIGHTS": "green"}):
            img_g, _, info_g = stack(be, weight="noise", analysis_cache=captured, **kw)
        img, _, info = stack(be, weight="noise", analysis_cache=captured, **kw)
        # GPU-only heat maps need not collect statistics. Supply an explicit legacy
        # three-item cache here rather than assuming analysis happened to save one.
        legacy_stats = {
            str(i): [s[0].tolist(), s[1].tolist(), float(s[2])] for i in range(n) for s in [be.stats(be.calibrate(i))]
        }
        legacy = {**captured, "statistics_backend": "gpu", "statistics": legacy_stats}
        with patch.object(be, "stats", wraps=be.stats) as measure:
            img_legacy, _, info_legacy = stack(be, weight="noise", analysis_cache=legacy, **kw)
        self.assertGreater(measure.call_count, 0, "Old statistics must be measured again, not invent RGB noise")
        self.assertTrue(np.array_equal(img_legacy, img, equal_nan=True))
        self.assertEqual(info_legacy["weights"], info["weights"])
        self.assertFalse(info_g["weights_per_channel"])
        self.assertTrue(info["weights_per_channel"])
        wg, w = np.array(info_g["weights"]), np.array(info["weights"])
        self.assertEqual(w.shape, (n, 3))
        np.testing.assert_array_equal(w[:, 1], wg)  # green weights are the previous weights
        self.assertLess(w[2, 0], 0.2 * np.delete(w[:, 0], 2).min())  # light 2 counts little in red...
        np.testing.assert_allclose(w[2, 2], np.delete(w[:, 2], 2).mean(), rtol=0.3)  # ...but normally in blue
        self.assertTrue(np.array_equal(img[..., 1], img_g[..., 1], equal_nan=True))  # green bit-identical
        ok = np.isfinite(img).all(-1) & np.isfinite(img_g).all(-1)
        self.assertGreater(np.abs(img[..., 0] - img_g[..., 0])[ok].max(), 0.5)  # red re-weighted
        # red noise in the stack falls (light 2's noise counts less); flat sky region, high-pass
        bg = (slice(8, 60), slice(8, 60))

        def noise_of(a):
            from scipy.ndimage import gaussian_filter

            v = a[bg]
            valid = np.isfinite(v)
            filled = np.where(valid, v, np.median(v[valid]))
            residual = (filled - gaussian_filter(filled, 4))[valid]
            return 1.4826 * np.median(np.abs(residual - np.median(residual)))

        self.assertLess(noise_of(img[..., 0]), noise_of(img_g[..., 0]))
        self.assertEqual(noise_of(img[..., 1]), noise_of(img_g[..., 1]))


class ChannelRejectionNoiseTests(unittest.TestCase):
    def test_noise_rejection_uses_each_channels_noise(self):
        """Light 2 is noisy in red only: noise-aware rejection judges each channel's samples
        against that channel's own noise (stats[3]), not green's noise scaled to the channel."""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        noise = [(2.0, 1.0, 2.0)] * 6
        noise[2] = (8.0, 1.0, 2.0)
        raws, mats = lights(seed=6, noise=noise)
        n = len(raws)
        be = GPUBackend(iter(raws), 100, n)
        kw = dict(tile=128, reg_workers=1, framing="union", select=0, log=lambda _: None)
        captured, seen = {}, []
        with (
            patch.object(be, "detect", return_value=(np.ones((20, 6)), 3.0)),
            patch("gpucoadd.register", return_value=(mats, [np.ones(3)] * n)),
        ):
            stack(be, coverage_only=True, on_analysis=captured.update, reference_idx=0, **kw)
        integrate = be.integrate

        def capture(*args, **kwargs):
            seen.append(np.asarray(kwargs["noise"]))
            return integrate(*args, **kwargs)

        with patch.object(be, "integrate", side_effect=capture):
            stack(be, rejection="noise", analysis_cache=captured, **kw)
        tile = next(t for t in seen if t.shape == (n, 3))  # a tile every light reaches, in index order
        red_ratio = tile[:, 0] / tile[:, 1]
        # scaled from green, every light's red/green ratio would be its coefficients' ratio (alike
        # here); with each channel's own noise, light 2's is about four times the others'
        self.assertGreater(red_ratio[2], 2.5 * np.delete(red_ratio, 2).max())


if __name__ == "__main__":
    unittest.main()
