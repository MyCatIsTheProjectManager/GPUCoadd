"""Exact Overlap matching normal equations and fallback-policy checks."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class OverlapKernelTests(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        torch.manual_seed(61)

    def test_normal_systems_exact(self):
        import torch
        import overlap
        import overlap_kernels as kernel

        for C in (6, 7, 30, 48):
            W = (torch.rand(C, 3, 20000, device="cuda") > 0.2).float()
            W[0] = 0  # empty fit remains a regularized zero system
            target = torch.randn_like(W)
            u = torch.randn_like(W)
            X = torch.randn(C, 1, 40000, device="cuda")[:, :, ::2]  # strided columns
            Y = torch.randn(C, 1, 20000, device="cuda")
            one = torch.ones_like(Y)
            for cols in ([u, X, Y, one], [X, Y, one]):
                a, b = overlap._normal_torch(cols, W, target)
                x, y = kernel.normal(cols, W, target)
                self.assertTrue(torch.equal(a, x))
                self.assertTrue(torch.equal(b, y))

    def test_full_pair_fits_exact(self):
        import numpy as np
        import torch
        import overlap

        n, P = 8, 21000
        X = torch.arange(P, device="cuda", dtype=torch.float32)
        XY = torch.stack((X % 150, X // 150), 1)
        sky = 200 + 20 * torch.sin(X / 77) + 5 * torch.cos(X / 31)
        L = sky[None, None, :] + torch.randn(n, 3, P, device="cuda")
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)][:17]
        idx = torch.arange(20000, device="cuda").expand(len(pairs), -1).clone()
        idx[:, -137:] = -1
        L[:, :, 0] = float("nan")
        idx[:, 0] = -1  # padded pixel-zero NaN must be cleared before the fit
        a, b, noise = np.ones((n, 3)), np.zeros((n, 3)), np.ones((n, 3))
        with patch.dict(os.environ, {"GPUCOADD_OVERLAP": "torch"}):
            ref = overlap.fit_pairs(L, XY, idx, pairs, a, b, noise)
        with patch.dict(os.environ, {"GPUCOADD_OVERLAP": "reduce"}):
            actual = overlap.fit_pairs(L, XY, idx, pairs, a, b, noise)
        for key in ref:
            np.testing.assert_array_equal(actual[key], ref[key])

    def test_unsupported_and_compile_failure_fallback(self):
        import torch
        import overlap
        import overlap_kernels

        for C, P in ((1, 20000), (5, 20000), (6, 19999), (6, 8192)):
            W = torch.ones(C, 3, P, device="cuda")
            cols = [W, W, W]
            with patch.object(overlap, "pair_kernels", side_effect=AssertionError("Must not compile")):
                a, b = overlap._normal(cols, W, W)
            x, y = overlap._normal_torch(cols, W, W)
            self.assertTrue(torch.equal(a, x) and torch.equal(b, y))
            with self.assertRaises(ValueError):
                overlap_kernels.normal(cols, W, W)
        with patch.dict(os.environ, {"GPUCOADD_OVERLAP": "torch"}):
            self.assertIsNone(overlap.pair_kernels()[0])
        with patch.dict(os.environ, {"GPUCOADD_OVERLAP": "reduce"}), patch.object(torch, "__version__", "2.12.0"):
            self.assertIsNone(overlap.pair_kernels()[0])
        with (
            patch.dict(os.environ, {"GPUCOADD_OVERLAP": "reduce"}),
            patch.object(overlap, "_pair_kernels", None),
            patch.object(overlap, "_pair_failure", None),
            patch.object(overlap_kernels, "module", side_effect=RuntimeError("test compiler failure")),
        ):
            module, reason = overlap.pair_kernels()
            self.assertIsNone(module)
            self.assertIn("test compiler failure", reason)


if __name__ == "__main__":
    unittest.main()
