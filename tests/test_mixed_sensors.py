"""Lights of different sensor sizes in one stack (S50 1080x1920 + S50 Pro 4K): per-light raw
storage, pedestals and geometry (canvas, coverage, reach, sensor-edge trim)."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpucoadd as g  # noqa: E402

BASE = np.array([1000.0, 1500.0, 800.0])  # R G B sky levels
SLOPE = np.array([[0.5, 0.3], [0.4, -0.2], [-0.3, 0.6]])  # per channel d/dx, d/dy (reference px)


def forward(angle=0.0, tx=0.0, ty=0.0, scale=1.0):
    c, s = scale * np.cos(angle), scale * np.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.0]])


def render(f, w, h, pedestal):
    """GRBG raw of a linear sky seen through `f` (sensor -> reference coordinates)."""
    yy, xx = np.mgrid[:h, :w].astype(float)
    x, y = f[0, 0] * xx + f[0, 1] * yy + f[0, 2], f[1, 0] * xx + f[1, 1] * yy + f[1, 2]
    ch = np.where((yy % 2 == 0) & (xx % 2 == 1), 0, np.where((yy % 2 == 1) & (xx % 2 == 0), 2, 1)).astype(int)
    sky = BASE[ch] + SLOPE[ch, 0] * x + SLOPE[ch, 1] * y
    return np.rint(pedestal + sky).clip(0, 65535).astype(np.uint16)


def brute_coverage(mats, frames, sizes, width, height, edge):
    yy, xx = np.mgrid[:height, :width]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    cov = np.zeros(width * height, int)
    for i in frames:
        w, h = sizes[i]
        q = mats[i] @ p
        sx, sy = q[0] / q[2], q[1] / q[2]
        cov += (sx >= edge) & (sx <= w - 1 - edge) & (sy >= edge) & (sy <= h - 1 - edge)
    return cov.reshape(height, width)


class SelectionTests(unittest.TestCase):
    def test_sharpness_cutoff_per_group(self):
        """FWHM is measured in each sensor's own pixels: a global cutoff either rejects much of the
        blurrier-looking group or (groups of similar size: a wide spread) keeps everyone's soft
        lights; per group each loses only its own outliers."""
        rng = np.random.default_rng(1)
        fwhm = np.r_[rng.normal(3.0, 0.1, 20), 4.0, rng.normal(5.0, 0.15, 20), 6.5]
        mats = [np.eye(3)] * len(fwhm)
        groups = ["pro"] * 21 + ["s50"] * 21
        ok, dropped, cut = g.select_frames(mats, fwhm, 2.5, groups)
        self.assertEqual(dropped, [20, 41])
        self.assertEqual(set(cut), {"pro", "s50"})
        _, dropped_global, _ = g.select_frames(mats, fwhm, 2.5)
        self.assertEqual(dropped_global, [])  # the spread between the groups hides both soft lights
        ok1, d1, c1 = g.select_frames(mats, fwhm, 2.5, None)
        self.assertEqual((ok1, d1, c1), g.select_frames(mats, fwhm, 2.5))
        analysis = {
            "mats": [m.tolist() for m in mats],
            "fwhm": fwhm.tolist(),
            "ref_idx": 0,
            "sizes": [[64, 48]] * len(fwhm),
            "groups": groups,
        }
        self.assertNotIn(41, g.frames_for_region(analysis, len(fwhm), 64, 48, 2.5, [0, 0, 10, 10], 0))
        self.assertIn(30, g.frames_for_region(analysis, len(fwhm), 64, 48, 2.5, [0, 0, 10, 10], 0))


class MixedSensorTests(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        # "Pro" lights 96 x 64 (pedestal 100), "S50" lights 60 x 40 at 1.04 x the pixel scale
        # (pedestal 230), placed so that parts of the canvas are reached by the large sensor only
        self.sizes = [(96, 64)] * 3 + [(60, 40)] * 3
        fwd = [
            forward(0, 0, 0),
            forward(0.01, 2.3, -1.7),
            forward(-0.008, -3.1, 2.2),
            forward(0.02, 20.5, 12.25, 1.04),
            forward(-0.01, 18.2, 14.6, 1.04),
            forward(0.015, 22.7, 10.1, 1.04),
        ]
        self.peds = [100.0] * 3 + [230.0] * 3
        self.raws = [render(f, w, h, p) for f, (w, h), p in zip(fwd, self.sizes, self.peds)]
        self.mats = [np.linalg.inv(f) for f in fwd]
        self.n = len(self.raws)
        self.common = dict(tile=32, reg_workers=1, framing="union", select=0, log=lambda _: None)
        self.stats = (np.full(3, 2000.0, np.float32), np.full(3, 30.0, np.float32), 30.0, np.full(3, 30.0, np.float32))

    def run_stack(self, be, cls, cache=None, **kw):
        n = self.n
        captured = {}
        with (
            patch.object(cls, "detect", return_value=(np.ones((20, 6)), 3.0)),
            patch.object(cls, "stats", return_value=self.stats),
            patch("gpucoadd.register", return_value=(self.mats, [np.ones(3)] * n)),
        ):
            out, _, info = g.stack(be, analysis_cache=cache, on_analysis=captured.update, **{**self.common, **kw})
        return out, info["coverage"], info, captured

    def test_storage_and_pedestals(self):
        be = g.GPUBackend(iter(self.raws), self.peds, self.n, "bicubic", "rcd")
        self.assertEqual(be.sizes, dict(enumerate(self.sizes)))
        for i in (0, 4):
            w, h = self.sizes[i]
            cal = be.calibrate(i).cpu().numpy()
            self.assertEqual(cal.shape, (h, w))
            np.testing.assert_array_equal(cal, self.raws[i].astype(np.float32) - np.float32(self.peds[i]))
        cb = g.CPUBackend(self.raws, self.peds, 2, "bicubic", "rcd")
        np.testing.assert_array_equal(cb.calibrate(4), be.calibrate(4).cpu().numpy())
        with self.assertRaises(ValueError):
            g.GPUBackend(iter(self.raws), self.peds[:-1], self.n)

    def test_stack_recovers_sky_on_both_backends(self):
        for interp, debayer in (("bilinear", "bilinear"), ("bicubic", "rcd")):
            gb = g.GPUBackend(iter(self.raws), self.peds, self.n, interp, debayer)
            out, cov, info, captured = self.run_stack(gb, g.GPUBackend)
            self.assertEqual(captured["sizes"], [list(s) for s in self.sizes])
            ox, oy = info["canvas"]["origin_xy"]
            # coverage equals the brute-force per-light count (each light with its own size)
            shift = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1.0]])
            truth = brute_coverage(
                [m @ shift for m in self.mats], range(self.n), self.sizes, out.shape[1], out.shape[0], gb.edge
            )
            np.testing.assert_array_equal(cov, truth)
            self.assertTrue((truth == 3).any() and (truth == 6).any(), "both single- and mixed-sensor areas")
            # the stack is the sky wherever covered (raw rounding: <= 0.5 ADU)
            yy, xx = np.mgrid[: out.shape[0], : out.shape[1]].astype(float)
            sky = BASE + SLOPE[:, 0] * (xx + ox)[..., None] + SLOPE[:, 1] * (yy + oy)[..., None]
            m = cov > 0
            with self.subTest(interp=interp, debayer=debayer):
                self.assertTrue(np.isfinite(out[m]).all())
                self.assertTrue(np.isnan(out[~m]).all())
                self.assertLess(np.abs(out[m] - sky[m]).max(), 0.75)
            cb = g.CPUBackend(self.raws, self.peds, 2, interp, debayer)
            out_c, cov_c, _, _ = self.run_stack(cb, g.CPUBackend)
            with self.subTest(backend="cpu", interp=interp, debayer=debayer):
                np.testing.assert_array_equal(cov_c, cov)
                np.testing.assert_allclose(out_c, out, atol=0.02, equal_nan=True)

    def test_reference_from_main_group(self):
        gb = g.GPUBackend(iter(self.raws), self.peds, self.n, "bicubic", "rcd")
        _, _, _, plain = self.run_stack(gb, g.GPUBackend, coverage_only=True)
        self.assertEqual(plain["ref_idx"], 3)  # middle of all six: a small-sensor light
        for select in (0, 2.5):
            _, _, _, pooled = self.run_stack(
                gb, g.GPUBackend, coverage_only=True, reference_pool=[0, 1, 2], select=select
            )
            self.assertIn(pooled["ref_idx"], [0, 1, 2])
        with self.assertRaises(ValueError):
            self.run_stack(gb, g.GPUBackend, coverage_only=True, reference_pool=[9])

    def test_reach_and_partial_loading_use_per_light_sizes(self):
        gb = g.GPUBackend(iter(self.raws), self.peds, self.n, "bicubic", "rcd")
        _, _, _, captured = self.run_stack(gb, g.GPUBackend, coverage_only=True)
        # right of the small lights: only the large sensors reach it. Were the small lights taken to be
        # as large as the reference, they would seem to reach it too.
        region = [79, 20, 6, 20]
        load = g.frames_for_region(captured, self.n, 96, 64, 0, region, gb.edge, include_ref=False)
        self.assertEqual(load, [0, 1, 2])
        self.assertEqual(
            g.frames_for_region({**captured, "sizes": None}, self.n, 96, 64, 0, region, gb.edge, include_ref=False),
            list(range(self.n)),
        )
        # bottom-right area, reached by the small lights: partial loading equals the full stack
        region = [40, 30, 40, 30]
        load = g.frames_for_region(captured, self.n, 96, 64, 0, region, gb.edge, include_ref=False)
        self.assertIn(4, load)
        part_be = g.GPUBackend(
            iter([self.raws[i] for i in load]), [self.peds[i] for i in load], self.n, "bicubic", "rcd", indices=load
        )
        full, _, fi, _ = self.run_stack(gb, g.GPUBackend, cache=captured, region=region)
        part, _, pi, _ = self.run_stack(part_be, g.GPUBackend, cache=captured, region=region)
        self.assertEqual(fi["contributing_indices"], pi["contributing_indices"])
        np.testing.assert_allclose(part, full, rtol=1e-6, atol=1e-3, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
