"""Fused CUDA RCD (rcd_fused.py) against the reference rcd_demosaic, and its kernel-cache repair."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FusedRcdTests(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

    def test_matches_reference(self):
        import torch
        import torch.nn.functional as F
        import gpucoadd as g
        from rcd_fused import rcd_demosaic_fused

        rng = np.random.default_rng(1)
        H, W = 200, 260
        yy, xx = np.mgrid[:H, :W]
        sky = 0.05 + 0.02 * xx / W + 0.01 * np.sin(yy / 17.0)
        img = sky + rng.normal(0, 0.002, (H, W))
        for _ in range(60):  # stars
            y, x = rng.integers(4, H - 4), rng.integers(4, W - 4)
            img[y - 3 : y + 4, x - 3 : x + 4] += rng.uniform(0.01, 0.3) * np.exp(
                -((np.mgrid[-3:4, -3:4] ** 2).sum(0)) / 3.0
            )
        ms = [torch.from_numpy(m).cuda() for m in g.bayer_masks(H, W)]
        for cfa in (
            torch.tensor(img, dtype=torch.float32).cuda(),  # one light
            torch.tensor(np.stack([img, img[::-1].copy(), img * 1.5]), dtype=torch.float32).cuda(),
        ):  # a batch
            ref = g.rcd_demosaic(torch, F, cfa, *ms)
            got = rcd_demosaic_fused(torch, F, cfa, *ms)
            self.assertEqual(got.shape, ref.shape)
            d = (got - ref).abs() * 65535
            self.assertLess(float(d.mean()), 0.01)  # ADU: float rounding only
            self.assertLess(float(torch.quantile(d.flatten(), 0.9999)), 0.5)

    def test_backend_uses_fused_unless_told_not_to(self):
        import gpucoadd as g

        fn, kind = g.gpu_rcd()
        self.assertEqual(kind, "staged CUDA")
        os.environ["GPUCOADD_RCD"] = "jiterator"
        try:
            fn, kind = g.gpu_rcd()
            self.assertEqual(kind, "fused")
        finally:
            del os.environ["GPUCOADD_RCD"]
        os.environ["GPUCOADD_RCD"] = "reference"
        try:
            fn, kind = g.gpu_rcd()
            self.assertIs(fn, g.rcd_demosaic)
        finally:
            del os.environ["GPUCOADD_RCD"]

    def test_cache_repair(self):
        from rcd_fused import repair_cache

        with tempfile.TemporaryDirectory() as d:
            for name in ("k_arch8.6_abc_tmp_123", "j_arch8.6_def_tmp_9", "j_arch8.6_def", "plain"):
                Path(d, name).write_bytes(b"x")
            repair_cache(d)
            self.assertEqual(sorted(os.listdir(d)), ["j_arch8.6_def", "k_arch8.6_abc", "plain"])


if __name__ == "__main__":
    unittest.main()
