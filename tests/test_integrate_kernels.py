"""CUDA rejection kernels (integrate_kernels.py) against the torch implementations."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ClipKernelTests(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        torch.manual_seed(3)

    def data(self, n, m):
        import torch

        x = torch.randn(n, m, device="cuda") * 10 + 1000
        hit = torch.rand(n, m, device="cuda") < 0.02
        x[hit] += 300 * torch.rand(int(hit.sum()), device="cuda")
        x[torch.rand(n, m, device="cuda") < 0.1] = float("nan")
        x[:, :50] = float("nan")  # columns with no sample at all
        x[: n - 2, 50:100] = float("nan")  # columns with only two samples
        srt = torch.sort(x, dim=0)
        sig = torch.gather((torch.rand(n, 1, device="cuda") * 2 + 0.5).expand(n, m), 0, srt.indices)
        w = (torch.rand(n, device="cuda") + 0.2)[srt.indices]
        return srt.values, sig, w

    def test_sigma_and_winsorized_decisions_identical(self):
        import torch
        import gpucoadd as g
        import integrate_kernels as ik

        for n in (3, 9, 120, 344, 587):
            s, _, w = self.data(n, 20000 if n <= 120 else 2000)
            for weights in (None, w):
                for win in (False, True):
                    with self.subTest(n=n, weighted=weights is not None, winsorized=win):
                        r, kr = g.sigma_clip_sorted_torch(torch, s, 3.0, 3, weights, True, win)
                        o, ko = ik.clip_sorted(s, 3.0, 3, weights, True, win, g.WINSOR_ITERS)
                        self.assertTrue(torch.equal(kr, ko))
                        self.assertTrue(torch.equal(torch.isnan(r), torch.isnan(o)))
                        np.testing.assert_allclose(o.cpu().numpy(), r.cpu().numpy(), rtol=1e-6, equal_nan=True)
                        plain = ik.clip_sorted(s, 3.0, 3, weights, False, win, g.WINSOR_ITERS)
                        self.assertTrue(
                            torch.equal(plain, o)
                            or np.array_equal(plain.cpu().numpy(), o.cpu().numpy(), equal_nan=True)
                        )

    def test_noise_decisions_identical_but_for_float_ties(self):
        import torch
        import gpucoadd as g
        import integrate_kernels as ik

        for n in (3, 9, 120, 344, 587):
            s, sig, w = self.data(n, 20000 if n <= 120 else 2000)
            for weights in (None, w):
                with self.subTest(n=n, weighted=weights is not None):
                    r, kr = g.noise_clip_torch(torch, s, sig, 3.0, 3, weights, True)
                    o, ko = ik.clip_noise(s, sig, 3.0, 3, weights, True)
                    self.assertLessEqual(int((kr != ko).sum()), 2)  # samples within rounding of a threshold
                    self.assertTrue(torch.equal(torch.isnan(r), torch.isnan(o)))
                    same = kr == ko
                    np.testing.assert_allclose(o[same].cpu().numpy(), r[same].cpu().numpy(), rtol=1e-6, equal_nan=True)

    def test_backend_integrate_uses_kernels(self):
        import torch
        import gpucoadd as g

        self.assertEqual(g.gpu_clip()[1], "cuda kernels")
        gb = g.GPUBackend.__new__(g.GPUBackend)
        gb.torch, gb.dev = torch, torch.device("cuda")
        rng = np.random.default_rng(4)
        data = torch.tensor(
            (100 + rng.normal(0, 1, (24, 3, 40, 50)) * rng.uniform(0.5, 3, (24, 3, 1, 1))).astype(np.float32)
        ).cuda()
        noise = rng.uniform(0.5, 3, (24, 3)).astype(np.float32)
        for mode in ("sigma", "winsorized", "noise"):
            kw = {} if mode == "sigma" else {"rejection": mode, "noise": noise}
            gb.clip = None
            ref, kref = gb.integrate(data, 3.0, 3, kept=True, **kw)
            gb.clip = g.gpu_clip()[0]
            out, kout = gb.integrate(data, 3.0, 3, kept=True, **kw)
            with self.subTest(mode=mode):
                self.assertLessEqual(int((kout != kref).sum()), 0 if mode != "noise" else 2)
                np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-3)


class TileCoverageTests(unittest.TestCase):
    """tile_coverage against isfinite(st).all(dim=1): counts and per-light contribution."""

    def test_matches_torch(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import integrate_kernels as ik

        torch.manual_seed(9)
        for n, rows, cols in ((1, 1, 1), (5, 3, 70), (40, 17, 33), (351, 64, 64), (3, 1, 1000)):
            with self.subTest(n=n, rows=rows, cols=cols):
                st = torch.randn(n, 3, rows, cols, device="cuda")
                st[torch.rand_like(st[:, :1].expand_as(st)) < 0.3] = float("nan")  # all channels
                st[torch.rand_like(st) < 0.05] = float("nan")  # single channels
                st[torch.rand_like(st) < 0.02] = float("inf")
                st[torch.rand_like(st) < 0.02] = -float("inf")
                st[0] = float("nan")  # a light with no valid sample
                if n > 2:
                    st[1, 1] = float("nan")  # valid in two channels only: never covered
                valid = torch.isfinite(st).all(dim=1)
                cov, contributed = ik.tile_coverage(st)
                self.assertEqual(cov.dtype, torch.int32)
                self.assertTrue(torch.equal(cov.long(), valid.sum(dim=0)))
                self.assertTrue(torch.equal(contributed, valid.any(dim=(1, 2))))
                self.assertFalse(bool(contributed[0]))

    def test_empty_and_layout(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import integrate_kernels as ik

        cov, contributed = ik.tile_coverage(torch.empty(0, 3, 4, 5, device="cuda"))
        self.assertTrue(torch.equal(cov, torch.zeros(4, 5, dtype=torch.int32, device="cuda")))
        self.assertEqual(contributed.shape, (0,))
        with self.assertRaises(ValueError):
            ik.tile_coverage(torch.zeros(2, 3, 4, 5, device="cuda").transpose(2, 3))


if __name__ == "__main__":
    unittest.main()
