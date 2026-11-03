"""Feathering: smootherstep weights from each light's trimmed sensor edge in the weighted mean."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feather_kernels  # noqa: E402
from gpucoadd import CPUBackend, GPUBackend, stack  # noqa: E402

H, W = 240, 320


def cuda():
    import torch

    return torch.cuda.is_available()


def vignetted_mosaic(n=4, shift=60, vignette=0.12, seed=4):
    """Lights of a flat 1000 ADU sky, each darkened toward its corners (no flats), stepped across
    the sky: without feathering the stack steps where a light's edge enters."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W].astype(float)
    v = 1 - vignette * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    raws, mats = [], []
    for k in range(n):
        raws.append((100 + 1000 * v + rng.normal(0, 1, (H, W))).clip(0, 65535).astype(np.uint16))
        mats.append(np.array([[1, 0, -shift * k], [0, 1, 0], [0, 0, 1.0]]))  # reference -> light pixel
    return raws, mats


class FeatherWeightTests(unittest.TestCase):
    def test_ramp(self):
        t = np.linspace(0, 1, 11)
        r = feather_kernels.ramp(t)
        self.assertEqual((r[0], r[-1]), (0, 1))
        self.assertTrue((np.diff(r) > 0).all())
        np.testing.assert_allclose(feather_kernels.ramp(0.5), 0.5)

    def test_weights_edge_floor_and_outside(self):
        out = np.zeros((1, 3, 200), np.float32)
        m = np.eye(3)  # tile pixel x = light pixel x
        feather_kernels.feather_numpy(
            out, [0], [m], np.array([[150.0, 400.0]]), 2, 0, 200, 40.0
        )  # rows far from top and bottom
        row = out[0, 0]
        self.assertEqual(row[0], 0)  # x = 0: inside the 2 px edge trim -> outside the area
        self.assertAlmostEqual(float(row[2]), feather_kernels.FLOOR)  # on the trimmed edge: the floor
        self.assertAlmostEqual(float(row[42]), 1.0)  # 40 px in
        self.assertTrue((row[150:] == 0).all())  # beyond the light


class IntegrateWeightTests(unittest.TestCase):
    """With rejection effectively off, the result is the feather-weighted mean of the samples."""

    def check(self, be, gpu):
        rng = np.random.default_rng(9)
        n, rows, cols = 5, 4, 6
        vals = rng.normal(100, 5, (n, rows, cols, 3)).astype(np.float32)
        vals[1, :, :2] = np.nan  # light 1 does not cover two columns
        fw = rng.uniform(0.01, 1, (n, rows, cols)).astype(np.float32)
        lw = rng.uniform(0.5, 2, n)
        expect = np.nansum(vals * (fw * lw[:, None, None])[..., None], 0) / np.sum(
            np.where(np.isnan(vals), 0, (fw * lw[:, None, None])[..., None]), 0
        )
        if gpu:
            torch = be.torch
            st = torch.from_numpy(np.ascontiguousarray(np.moveaxis(vals, 3, 1))).to(be.dev)
            got = be.integrate(st, 100.0, 3, lw, feather=torch.from_numpy(fw).to(be.dev))
        else:
            got = be.integrate(vals, 100.0, 3, lw, feather=fw)
        np.testing.assert_allclose(got, expect, rtol=1e-5)

    def test_cpu(self):
        self.check(CPUBackend([np.zeros((8, 8), np.uint16)] * 2, 0, 1), gpu=False)

    def test_gpu(self):
        if not cuda():
            self.skipTest("CUDA unavailable")
        self.check(GPUBackend(iter([np.zeros((8, 8), np.uint16)] * 2), 0, 2), gpu=True)


class FeatherStackTests(unittest.TestCase):
    def run_stack(self, make, feather, raws, mats):
        be = make(raws)
        kw = dict(tile=128, reg_workers=1, framing="union", select=0, log=lambda _: None, feather=feather)
        with (
            patch.object(be, "detect", return_value=(np.ones((20, 6)), 3.0)),
            patch("gpucoadd.register", return_value=(mats, [np.ones(3)] * len(raws))),
        ):
            img, _, info = stack(be, reference_idx=0, **kw)
        return img, info

    def steps(self, img):
        """Largest jump between neighbouring columns of the row-averaged luminance."""
        prof = img[60:180].mean(2).mean(0)  # covered columns only (the union canvas is fully covered here)
        return np.nanmax(np.abs(np.diff(prof)))

    def check_backend(self, make):
        raws, mats = vignetted_mosaic()
        plain, pinfo = self.run_stack(make, 0, raws, mats)
        soft, sinfo = self.run_stack(make, 60, raws, mats)
        np.testing.assert_array_equal(np.isnan(plain), np.isnan(soft))
        np.testing.assert_array_equal(pinfo["coverage"], sinfo["coverage"])
        self.assertEqual((pinfo["feather_px"], sinfo["feather_px"]), (0.0, 60.0))
        hard, smooth = self.steps(plain), self.steps(soft)
        self.assertGreater(hard, 8)  # vignetting steps of tens of ADU where a light enters
        self.assertLess(smooth, hard / 3)
        return plain, soft

    def test_cpu(self):
        self.check_backend(lambda raws: CPUBackend(raws, 100, 2))

    def test_gpu_and_cpu_agree(self):
        if not cuda():
            self.skipTest("CUDA unavailable")
        _, gsoft = self.check_backend(lambda raws: GPUBackend(iter(raws), 100, len(raws)))
        _, csoft = self.check_backend(lambda raws: CPUBackend(raws, 100, 2))
        np.testing.assert_allclose(gsoft, csoft, rtol=1e-4, atol=0.05, equal_nan=True)

    def test_off_is_unchanged(self):
        """feather=0 runs the old path: no feather buffer, the same result as before the option."""
        if not cuda():
            self.skipTest("CUDA unavailable")
        raws, mats = vignetted_mosaic()
        make = lambda r: GPUBackend(iter(r), 100, len(r))
        a, _ = self.run_stack(make, 0, raws, mats)
        with patch.object(GPUBackend, "feather_tile", side_effect=AssertionError("feather used")):
            b, _ = self.run_stack(make, 0, raws, mats)
        np.testing.assert_array_equal(a, b)

    def test_range(self):
        raws, mats = vignetted_mosaic(n=2)
        for bad in (-1, float("nan"), 1e6):
            with self.subTest(feather=bad), self.assertRaises(ValueError):
                self.run_stack(lambda r: CPUBackend(r, 100, 1), bad, raws, mats)


if __name__ == "__main__":
    unittest.main()
