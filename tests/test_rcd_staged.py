"""Five-stage RCD must preserve Jiterator pixels, including every reflected border."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class StagedRcdTests(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

    def test_exact_on_borders_stars_gradients_and_batches(self):
        import torch
        import torch.nn.functional as F
        import numpy as np
        from gpucoadd import bayer_masks
        from rcd_fused import rcd_demosaic_fused
        from rcd_kernels import rcd_demosaic_staged

        rng = np.random.default_rng(147)
        for shape in [(5, 5), (7, 11), (32, 33), (3, 47, 62), (8, 128, 160)]:
            h, w = shape[-2:]
            yy, xx = np.mgrid[:h, :w]
            sky = 0.05 + 0.02 * xx / w + 0.01 * np.sin(yy / 17)
            noise = sky + rng.normal(0, 0.002, shape)
            noise[..., h // 2, w // 2] = 1.0
            cases = [noise, np.zeros(shape), np.ones(shape) * 0.05, rng.uniform(-0.01, 1.1, shape)]
            masks = [torch.as_tensor(m, device="cuda") for m in bayer_masks(h, w)]
            for case in cases:
                x = torch.as_tensor(case, dtype=torch.float32, device="cuda")
                before = rcd_demosaic_fused(torch, F, x, *masks)
                after = rcd_demosaic_staged(torch, F, x, *masks)
                self.assertTrue(torch.equal(before, after), f"{shape}, max error {(before - after).abs().max().item()}")

    def test_strided_input_and_masks_with_alternate_phase(self):
        import torch
        import torch.nn.functional as F
        from gpucoadd import bayer_masks
        from rcd_fused import rcd_demosaic_fused
        from rcd_kernels import rcd_demosaic_staged

        torch.manual_seed(41)
        x = torch.rand((3, 40, 82), device="cuda")[..., ::2]
        masks = [torch.as_tensor(m, device="cuda")[1:38, 1:42] for m in bayer_masks(40, 90)]
        x = x[..., :37, :41]
        self.assertFalse(x.is_contiguous())
        a = rcd_demosaic_fused(torch, F, x, *masks)
        b = rcd_demosaic_staged(torch, F, x, *masks)
        self.assertTrue(torch.equal(a, b))

    def test_independent_of_batch_companions(self):
        import torch
        import torch.nn.functional as F
        from gpucoadd import bayer_masks
        from rcd_kernels import rcd_demosaic_staged

        torch.manual_seed(42)
        x = torch.rand((3, 63, 80), device="cuda") * 0.1
        masks = [torch.as_tensor(m, device="cuda") for m in bayer_masks(63, 80)]
        batch = rcd_demosaic_staged(torch, F, x, *masks)
        for i in range(3):
            self.assertTrue(torch.equal(batch[i], rcd_demosaic_staged(torch, F, x[i], *masks)))

    def test_current_stream(self):
        import torch
        import torch.nn.functional as F
        from gpucoadd import bayer_masks
        from rcd_fused import rcd_demosaic_fused
        from rcd_kernels import rcd_demosaic_staged

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.full((3, 64, 80), 0.06, device="cuda")
            masks = [torch.as_tensor(m, device="cuda") for m in bayer_masks(64, 80)]
            a = rcd_demosaic_fused(torch, F, x, *masks)
            b = rcd_demosaic_staged(torch, F, x, *masks)
        torch.cuda.current_stream().wait_stream(stream)
        self.assertTrue(torch.equal(a, b))

    def test_compile_failure_keeps_existing_fallbacks(self):
        from unittest.mock import patch
        import gpucoadd as g
        from rcd_fused import rcd_demosaic_fused

        with patch("rcd_kernels.module", side_effect=RuntimeError("test compiler unavailable")):
            fn, kind = g.gpu_rcd()
            self.assertIs(fn, rcd_demosaic_fused)
            self.assertIn("test compiler unavailable", kind)
            with patch("rcd_fused._kernels", side_effect=RuntimeError("test Jiterator unavailable")):
                fn, kind = g.gpu_rcd()
                self.assertIs(fn, g.rcd_demosaic)
                self.assertIn("test Jiterator unavailable", kind)


if __name__ == "__main__":
    unittest.main()
