"""Fused calibration and padding must preserve every float32 result bit."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpucoadd as g
import calibrate_kernels as ck


class CalibrationKernelTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

    def test_calibrate_pad_bits_unsigned_flats_strides_and_reflection(self):
        rng = np.random.default_rng(45)
        raws, flats = [], []
        boxes = [(0, 7, 0, 9), (2, 4, 3, 5), (3, 25, 4, 31), (1, 12, 1, 20)]
        peds = [0.0, 468.125, 1200.3, -100.2]
        for k in range(4):
            bits = rng.integers(0, 65536, (32, 40), dtype=np.uint16)
            bits[:2, :4] = [0, 32767, 32768, 65535]
            raw = torch.from_numpy(bits.view(np.int16)).cuda()
            flat = torch.from_numpy(rng.uniform(0, 2, (32, 40)).astype(np.float32)).cuda()
            flat[0, :4] = torch.tensor([0.0, -0.0, 1.0, 1.5], device="cuda")
            if k == 3:
                raw, flat = raw.t(), flat.t()  # exercise noncontiguous inputs
            raws.append(raw)
            flats.append(flat)
        H, W = 71, 75  # repeated reflections, including length-two crops
        ref = torch.empty((4, H, W), device="cuda")
        for q, (raw, flat, ped, (a, b, c, d)) in enumerate(zip(raws, flats, peds, boxes)):
            cal = (raw[a:b, c:d].to(torch.int32).bitwise_and(65535).float() - ped) * flat[a:b, c:d]
            iy = torch.arange(H, device="cuda") % (2 * (b - a - 1))
            ix = torch.arange(W, device="cuda") % (2 * (d - c - 1))
            iy = torch.minimum(iy, 2 * (b - a - 1) - iy)
            ix = torch.minimum(ix, 2 * (d - c - 1) - ix)
            ref[q] = cal[iy[:, None], ix[None, :]]
        out = torch.full_like(ref, float("nan"))
        ck.calibrate_pad(raws, flats, peds, boxes, out)
        self.assertTrue(torch.equal(ref.view(torch.int32), out.view(torch.int32)))
        with self.assertRaises(ValueError):
            ck.calibrate_pad(raws, flats, peds, [(0, 1, 0, 5)] * 4, out)

    def test_partial_indices_mixed_sensors_and_warp_modes(self):
        rng = np.random.default_rng(18)
        raws = [rng.integers(0, 65536, shape, dtype=np.uint16) for shape in ((96, 80), (64, 100), (80, 72))]
        indices, peds = [1, 3, 6], [468.125, 1200.3, 0.0]
        frames = [6, 1, 3]  # different order from the loaded slots
        mats = [np.array([[1.0, 0, 2.2], [0, 1, 3.1], [0, 0, 1]])] * 7
        coefs = {i: (np.array([0.9, 1.1, 1.2], np.float32), np.array([2, -3, 4], np.float32)) for i in frames}
        for interp, debayer in (("bilinear", "bilinear"), ("bicubic", "rcd")):
            be = g.GPUBackend(iter(raws), peds, 7, interp, debayer, indices=indices)
            fused, _ = g.gpu_calibrate()
            self.assertIsNotNone(fused)
            a = be.alloc_stack(3, 24, 28)
            b = torch.empty_like(a)
            be.calibration = None
            be.warp_tile(a, frames, mats, coefs, 10, 38, 12, 36)
            be.calibration = fused
            be.warp_tile(b, frames, mats, coefs, 10, 38, 12, 36)
            self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))

    def test_force_reference_and_compile_failure_fallback(self):
        with patch.dict(os.environ, {"GPUCOADD_CALIBRATE": "torch"}):
            self.assertIsNone(g.gpu_calibrate()[0])
        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(ck, "module", side_effect=RuntimeError("test unavailable")),
        ):
            os.environ.pop("GPUCOADD_CALIBRATE", None)
            fn, reason = g.gpu_calibrate()
            self.assertIsNone(fn)
            self.assertIn("test unavailable", reason)


if __name__ == "__main__":
    unittest.main()
