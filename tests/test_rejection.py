"""Rejection modes: sigma (default), winsorized, noise (per-light noise aware)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpucoadd as g  # noqa: E402

KAPPA, ITERS = 3.0, 3


def winsorized_reference(values, kappa=KAPPA, iters=ITERS):
    """Straightforward per-pixel winsorized sigma clipping, written from the algorithm description:
    each pass takes the median and a spread re-estimated after pulling values beyond median +/- 1.5
    spread in to that limit (x 1.134), then rejects beyond median +/- kappa spread (at least 3 kept)."""
    v = np.sort(values[np.isfinite(values)])
    lo, hi = 0, len(v)
    for _ in range(iters):
        s = v[lo:hi]
        med = 0.5 * (s[(len(s) - 1) // 2] + s[len(s) // 2])
        sd = np.sqrt(((s - s.mean()) ** 2).mean())
        for _ in range(g.WINSOR_ITERS):
            xw = np.clip(s, med - 1.5 * sd, med + 1.5 * sd)
            sd = 1.134 * np.sqrt(((xw - xw.mean()) ** 2).mean())
        nlo, nhi = int((v < med - kappa * sd).sum()), int((v <= med + kappa * sd).sum())
        nlo, nhi = max(lo, nlo), min(hi, nhi)
        if nhi - nlo >= 3:
            lo, hi = nlo, nhi
    return v[lo:hi].mean() if hi > lo else np.nan


def hetero(rng, n=60, m=20000, noisy=3.0, outliers=0.0):
    """True value 100; first half of the lights noise 1, second half `noisy`; optional +50
    outliers (trail-like) in a random noisy light at a share of pixels."""
    sig = np.where(np.arange(n) < n // 2, 1.0, noisy)
    x = 100 + rng.normal(0, 1, (n, m)) * sig[:, None]
    hit = np.zeros((n, m), bool)
    if outliers:
        cols = np.flatnonzero(rng.random(m) < outliers)
        rows = rng.integers(n // 2, n, cols.size)
        x[rows, cols] += 50
        hit[rows, cols] = True
    return x.astype(np.float32), sig.astype(np.float32), hit


def run_torch(x, sig, mode, kept=True):
    import torch

    t = torch.from_numpy(x)
    srt = torch.sort(t, dim=0)
    if mode == "noise":
        s = torch.gather(torch.from_numpy(np.repeat(sig[:, None], x.shape[1], 1)), 0, srt.indices)
        mean, k = g.noise_clip_torch(torch, srt.values, s, KAPPA, ITERS, kept=True)
    else:
        mean, k = g.sigma_clip_sorted_torch(torch, srt.values, KAPPA, ITERS, kept=True, winsorize=mode == "winsorized")
    return mean.numpy(), k.numpy()


class RejectionTests(unittest.TestCase):
    def test_winsorized_matches_reference(self):
        rng = np.random.default_rng(0)
        x, _, _ = hetero(rng, n=40, m=400, noisy=1.0, outliers=0.3)
        x[rng.random(x.shape) < 0.02] = np.nan  # missing samples (frame edges)
        ref = np.array([winsorized_reference(x[:, j]) for j in range(x.shape[1])])
        got_np = g.sigma_clip_sorted_np(np.sort(x.T, axis=1), KAPPA, ITERS, winsorize=True)
        got_t, _ = run_torch(x, None, "winsorized")
        np.testing.assert_allclose(got_np, ref, rtol=1e-5)
        np.testing.assert_allclose(got_t, ref, rtol=1e-5)

    def test_noise_mode_does_not_over_reject_noisy_lights(self):
        rng = np.random.default_rng(1)
        x, sig, _ = hetero(rng)
        n = x.shape[0]
        # sigma mode: pooled spread ~2.2 -> noisy samples beyond ~2.2 of their own sigma are rejected
        res = {}
        for mode in ("sigma", "winsorized", "noise"):
            mean, kept = run_torch(x, sig, mode)
            res[mode] = (mean, 1 - kept.mean() / n)
        # count which half lost samples: rerun each mode with only one half perturbed is not
        # possible; use the analytical expectation instead: sigma mode rejects ~1.3% overall (all
        # from the noisy half), noise mode ~0.27% (equally)
        self.assertGreater(res["sigma"][1], 0.008, res)
        self.assertLess(res["noise"][1], 0.005, res)
        for mode, (mean, _) in res.items():
            with self.subTest(mode=mode):
                self.assertLess(abs(float(np.mean(mean)) - 100), 0.01)
        # noise-mode result is closer to the inverse-variance ideal (lower scatter across pixels)
        self.assertLess(np.std(res["noise"][0]), np.std(res["sigma"][0]) * 1.05)

    def test_noise_mode_rejects_outliers_in_noisy_lights(self):
        rng = np.random.default_rng(2)
        x, sig, hit = hetero(rng, outliers=0.2)
        cols = np.flatnonzero(hit.any(0))
        mean, kept = run_torch(x, sig, "noise")
        # with the outlier rejected the result is unbiased; a kept +50 outlier would add ~0.8
        self.assertLess(abs(float(np.mean(mean[cols])) - 100), 0.05)
        clean_mean, _ = run_torch(x, sig, "sigma")
        self.assertLess(abs(float(np.mean(clean_mean[cols])) - 100), 0.05)

    def test_noise_mode_equals_sigma_mode_for_equal_noise(self):
        rng = np.random.default_rng(3)
        x, sig, _ = hetero(rng, noisy=1.0, outliers=0.05)
        a, ka = run_torch(x, sig, "sigma")
        b, kb = run_torch(x, sig, "noise")
        # thresholds are computed slightly differently, so a borderline sample can flip at single
        # pixels (one flip moves a 60-sample mean by ~0.05); statistically the two must agree
        self.assertLess(np.median(np.abs(a - b)), 0.005)
        self.assertLess(abs(np.std(a - 100) / np.std(b - 100) - 1), 0.02)
        self.assertLess(abs(ka.mean() - kb.mean()), 0.2)

    def test_backends_agree_and_channels_are_mapped(self):
        """GPU (n, 3, rows, cols) and CPU (n, rows, cols, 3) layouts must give the same result,
        with each channel using its own per-light noise."""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(4)
        n, R, C = 24, 40, 50
        noise = rng.uniform(0.5, 3, (n, 3)).astype(np.float32)
        data = (100 + rng.normal(0, 1, (n, 3, R, C)) * noise[:, :, None, None]).astype(np.float32)
        data[rng.random(data.shape) < 0.01] += 80
        gb = g.GPUBackend.__new__(g.GPUBackend)
        gb.torch, gb.dev = torch, torch.device("cuda")
        cb = g.CPUBackend.__new__(g.CPUBackend)
        from concurrent.futures import ThreadPoolExecutor

        cb.pool = ThreadPoolExecutor(2)
        for mode in ("sigma", "winsorized", "noise"):
            kw = {} if mode == "sigma" else {"rejection": mode, "noise": noise}
            out_g = gb.integrate(torch.from_numpy(data).cuda(), KAPPA, ITERS, **kw)
            out_c = cb.integrate(np.ascontiguousarray(np.moveaxis(data, 1, 3)), KAPPA, ITERS, **kw)
            with self.subTest(mode=mode):
                np.testing.assert_allclose(out_g, out_c, rtol=1e-5, atol=1e-3)
        # swapping the channels' noise must change the noise-mode result (channels really used)
        out_sw = cb.integrate(
            np.ascontiguousarray(np.moveaxis(data, 1, 3)), KAPPA, ITERS, rejection="noise", noise=noise[:, ::-1].copy()
        )
        self.assertFalse(np.allclose(out_sw, out_c))
        cb.pool.shutdown()


if __name__ == "__main__":
    unittest.main()
