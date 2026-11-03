"""Synthetic framing checks; no astronomical input files required."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpucoadd import CPUBackend, GPUBackend, output_canvas, stack


class CanvasTests(unittest.TestCase):
    def test_cached_framing_and_optional_reference(self):
        yy, xx = np.mgrid[:96, :80]
        raw = (2000 + xx * 3 + yy * 5).astype(np.uint16)
        be = CPUBackend([raw] * 3, 100, 2)
        mats = [np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]]) for dx, dy in ((24, -16), (0, 0), (-24, 16))]
        captured = {}
        with (
            patch.object(be, "detect", return_value=(np.ones((20, 6)), 3.0)),
            patch.object(be, "stats", return_value=(np.ones(3), np.ones(3), 1.0, np.ones(3))),
            patch("gpucoadd.register", return_value=(mats, [np.ones(3)] * 3)) as registration,
        ):
            union, _, info = stack(
                be, 32, 1, framing="union", select=2.5, reference_idx=0, on_analysis=captured.update, log=lambda _: None
            )
            self.assertEqual(registration.call_args.args[1], 0)
            with (
                patch.object(be, "detect", side_effect=AssertionError("detection repeated")),
                patch("gpucoadd.register", side_effect=AssertionError("registration repeated")),
            ):
                cropped, _, ci = stack(
                    be,
                    32,
                    1,
                    framing="union",
                    select=2.5,
                    analysis_cache=captured,
                    region=[13, 17, 53, 61],
                    log=lambda _: None,
                )
                with self.assertRaises(ValueError):
                    stack(
                        be,
                        32,
                        1,
                        framing="union",
                        analysis_cache=captured,
                        region=[0, 0, 10000, 10000],
                        log=lambda _: None,
                    )
        np.testing.assert_allclose(cropped, union[17:78, 13:66], atol=0.01, rtol=2e-6, equal_nan=True)
        self.assertEqual(
            ci["canvas"]["origin_xy"], [info["canvas"]["origin_xy"][0] + 13, info["canvas"]["origin_xy"][1] + 17]
        )
        self.assertEqual(cropped.shape, (61, 53, 3))
        be.pool.shutdown()

    def test_bounds(self):
        identity = np.eye(3)
        # Source frame translated +20,-10 into reference coordinates.
        inverse = np.array([[1, 0, -20], [0, 1, 10], [0, 0, 1.0]])
        self.assertEqual(output_canvas([identity, inverse], [0, 1], 80, 96, "union"), (100, 106, (0, -10)))
        self.assertEqual(output_canvas([identity, inverse], [0], 80, 96, "union"), (80, 96, (0, 0)))
        self.assertEqual(output_canvas([inverse], [0], 80, 96, "reference"), (80, 96, (0, 0)))

    def test_rotation_and_projective_horizon(self):
        inverse = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
        self.assertEqual(output_canvas([inverse], [0], 80, 96, "union"), (96, 80, (-95, 0)))
        forward = np.array([[1, 0, 0], [0, 1, 0], [0.1, 0, -2.0]])
        with self.assertRaises(ValueError):
            output_canvas([np.linalg.inv(forward)], [0], 80, 96, "union")

    def check_backend(self, gpu=False, interp="bilinear", debayer="bilinear"):
        yy, xx = np.mgrid[:96, :80]
        raw = (2000 + xx * 3 + yy * 5).astype(np.uint16)
        be = (
            GPUBackend(iter([raw] * 3), 100, 3, interp, debayer)
            if gpu
            else CPUBackend([raw] * 3, 100, 2, interp, debayer)
        )
        mats = [np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]]) for dx, dy in ((24, -16), (0, 0), (-24, 16))]
        fake_stats = (np.ones(3), np.ones(3), 1.0)
        with (
            patch.object(be, "detect", return_value=(np.ones((20, 6)), 3.0)),
            patch.object(be, "stats", return_value=fake_stats),
            patch("gpucoadd.register", return_value=(mats, [np.ones(3)] * 3)),
        ):
            reference, _, ri = stack(be, 32, 1, log=lambda _: None)
            union, _, ui = stack(be, 32, 1, framing="union", log=lambda _: None)
            with (
                patch.object(be, "alloc_stack", side_effect=AssertionError("stack allocation in preview")),
                patch.object(be, "warp_tile", side_effect=AssertionError("warping in preview")),
                patch.object(be, "integrate", side_effect=AssertionError("integration in preview")),
                patch.object(be, "stats", side_effect=AssertionError("normalization in preview")),
            ):
                image, _, pi = stack(be, 32, 1, framing="union", coverage_only=True, log=lambda _: None)
            self.assertIsNone(image)
            self.assertEqual(pi["coverage_kind"], "geometric_preview")
            self.assertEqual(pi["canvas"]["tiles"], 0)
            expected = np.zeros_like(pi["coverage"])
            px, py = pi["canvas"]["origin_xy"]
            e = be.edge  # footprints are trimmed at the sensor edge
            for matrix in mats:
                x = -int(matrix[0, 2]) - px
                y = -int(matrix[1, 2]) - py
                expected[y + e : y + 96 - e, x + e : x + 80 - e] += 1
            np.testing.assert_array_equal(pi["coverage"], expected)
            # OpenCV's NaN interpolation borders can shrink finite support;
            # the preview intentionally represents geometric footprints.
            self.assertTrue((ui["coverage"] <= pi["coverage"]).all())
        ox, oy = ui["canvas"]["origin_xy"]
        np.testing.assert_allclose(
            union[-oy : -oy + 96, -ox : -ox + 80], reference, rtol=2e-6, atol=0.01, equal_nan=True
        )
        np.testing.assert_array_equal(ui["coverage"][-oy : -oy + 96, -ox : -ox + 80], ri["coverage"])
        self.assertGreater(np.count_nonzero(ui["coverage"]), np.count_nonzero(ri["coverage"]))
        self.assertTrue(np.isnan(union[ui["coverage"] == 0]).all())
        self.assertEqual(int(ui["coverage"].max()), 3)
        if not gpu:
            be.pool.shutdown()

    def test_cpu_overlap_and_coverage(self):
        for interp, debayer in (("bilinear", "bilinear"), ("bicubic", "rcd")):
            with self.subTest(interp=interp, debayer=debayer):
                self.check_backend(interp=interp, debayer=debayer)

    def test_gpu_overlap_and_coverage(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        for interp, debayer in (("bilinear", "bilinear"), ("bicubic", "rcd")):
            with self.subTest(interp=interp, debayer=debayer):
                self.check_backend(gpu=True, interp=interp, debayer=debayer)


if __name__ == "__main__":
    unittest.main()
