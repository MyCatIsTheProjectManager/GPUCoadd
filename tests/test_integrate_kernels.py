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


if __name__ == "__main__":
    unittest.main()
