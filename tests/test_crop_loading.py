"""Crop-aware loading: never skip a light that can contribute, keep full-field semantics."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpucoadd import GPUBackend, footprint_reaches, footprints_reach, frame_footprint, frames_for_region, frames_reaching, stack


def forward(angle=0.0, tx=0.0, ty=0.0, px=0.0, py=0.0):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [px, py, 1.0]])


def contributes(minv, w, h, width, height, edge=0):
    """Brute force: does any output pixel of the canvas map at least `edge` px inside the sensor? (the warp's rule)"""
    yy, xx = np.mgrid[:height, :width]
    p = minv @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    sx, sy = p[0] / p[2], p[1] / p[2]
    return bool(((sx >= edge) & (sx <= w - 1 - edge) & (sy >= edge) & (sy <= h - 1 - edge)).any())


class ReachTests(unittest.TestCase):
    def test_superset_of_brute_force(self):
        rng = np.random.default_rng(3)
        extra = 0
        for k in range(3000):
            edge = int(rng.choice([0, 2, 3, 12]))
            w, h = (int(v) for v in rng.integers(2 * edge + 2, 2 * edge + 40, 2))
            width, height = (int(v) for v in rng.integers(1, 50, 2))
            p = rng.normal(0, 4e-3, 2) if rng.random() < 0.3 else (0, 0)
            f = forward(rng.uniform(-np.pi, np.pi), rng.uniform(-w - 5, width + 5), rng.uniform(-h - 5, height + 5), *p)
            minv = np.linalg.inv(f)
            try:
                reach = frames_reaching([minv], [0], w, h, width, height, edge) == [0]
            except ValueError:  # invalid projective footprint: also rejected by stack()
                continue
            truth = contributes(minv, w, h, width, height, edge)
            with self.subTest(k=k):
                self.assertTrue(reach or not truth, "a contributing frame would be skipped")
            extra += reach and not truth
        self.assertLess(extra, 300)  # conservative, but not wildly so

    def test_barely_touching(self):
        # Footprint edges/corners at, just inside and just outside the outermost pixel centres.
        w, h, width, height = 20, 10, 30, 25
        for d, edge in [(d, e) for d in (0.0, 0.01, 0.5, 0.99, 1.0, 1.01, 1.5) for e in (0, 3)]:
            for f in (
                forward(tx=-(w - 1) - d, ty=3),  # right edge of sensor near x = 0
                forward(tx=width - 1 + d, ty=3),  # left edge near x = width - 1
                forward(tx=5, ty=-(h - 1) - d),  # bottom edge near y = 0
                forward(np.pi / 4, tx=-14.142 - d, ty=-7),  # rotated corner near the canvas corner
                forward(0.2, tx=width - 1 + d, ty=0, px=2e-3, py=-1e-3),
            ):
                minv = np.linalg.inv(f)
                f = f @ np.array([[1, 0, -edge], [0, 1, -edge], [0, 0, 1.0]])  # same distances to the trimmed edge
                minv = np.linalg.inv(f)
                with self.subTest(d=d, edge=edge, f=f.tolist()):
                    truth = contributes(minv, w, h, width, height, edge)
                    self.assertTrue(not truth or frames_reaching([minv], [0], w, h, width, height, edge) == [0])

    def test_vectorized_equals_per_footprint(self):
        """footprints_reach (tile selection) gives exactly footprint_reaches' answer, including
        footprints touching the grown tile, axis-aligned ones on integer bounds and degenerate quads."""
        rng = np.random.default_rng(7)
        quads = []
        for _ in range(4000):
            w, h = (int(v) for v in rng.integers(4, 60, 2))
            p = rng.normal(0, 3e-3, 2) if rng.random() < 0.3 else (0, 0)
            angle = 0.0 if rng.random() < 0.3 else rng.uniform(-np.pi, np.pi)
            f = forward(angle, rng.integers(-70, 70), rng.integers(-70, 70), *p)
            if rng.random() < 0.5:
                f[:2, 2] += rng.uniform(-1, 1, 2)
            try:
                quads.append(frame_footprint(np.linalg.inv(f), w, h, int(rng.choice([0, 2]))))
            except ValueError:
                continue
        quads.append(np.array([[0, 0], [10, 0], [10, 0], [0, 0]], float))  # zero area
        quads.append(np.array([[-1, 5], [5, 5], [5, 9], [-1, 9]], float))  # edge on x = -1
        quads.append(np.array([[-9, -9], [-1, -9], [-1, -1], [-9, -1]], float))  # corner at (-1, -1)
        quads = np.array(quads)
        for width, height in ((30, 25), (1, 1), (64, 7)):
            with self.subTest(width=width, height=height):
                expected = [footprint_reaches(q, width, height) for q in quads]
                self.assertEqual(footprints_reach(quads, width, height).tolist(), expected)
        self.assertEqual(footprints_reach(np.zeros((0, 4, 2)), 5, 5).tolist(), [])

    def test_plan_keeps_reference_and_full_field_selection(self):
        # Reference (index 1) lies outside the crop but must still be loaded for normalization.
        mats = [np.linalg.inv(forward(tx=x)) for x in (0, 500, 40)]
        analysis = {"mats": [m.tolist() for m in mats], "fwhm": [3.0, 3.0, 3.0], "ref_idx": 1}
        self.assertEqual(frames_for_region(analysis, 3, 64, 48, 0, [0, 0, 30, 30], 0), [0, 1])
        self.assertEqual(frames_for_region(analysis, 3, 64, 48, 0, [40, 0, 30, 30], 0), [0, 1, 2])
        # light 2's trimmed footprint starts at canvas x = 40 + 12 = 52: pixels 40..51 cannot reach it
        self.assertEqual(frames_for_region(analysis, 3, 64, 48, 0, [40, 0, 12, 30], 12), [0, 1])
        self.assertEqual(frames_for_region(analysis, 3, 64, 48, 0, [40, 0, 13, 30], 12), [0, 1, 2])


class PartialStackTests(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

    def test_partial_equals_full(self):
        rng = np.random.default_rng(11)
        H, W, n = 96, 80, 9
        raws = [
            (2000 + rng.normal(0, 30, (H, W)) + 4 * np.mgrid[:H, :W][1]).clip(0, 65535).astype(np.uint16)
            for _ in range(n)
        ]
        # small rotations plus frames spread far apart, so most cannot reach a corner crop
        shifts = [(0, 0), (3.3, -2.1), (-2.7, 1.4), (150, 0), (150, 120), (0, 140), (-1.5, -3.5), (300, 0), (1.2, 2.2)]
        mats = [np.linalg.inv(forward(0.01 * k, dx, dy)) for k, (dx, dy) in enumerate(shifts)]
        stats = (np.full(3, 2000.0), np.full(3, 30.0), 30.0)
        common = dict(tile=32, reg_workers=1, framing="union", select=0, log=lambda _: None)
        for interp, debayer in (("bilinear", "bilinear"), ("bicubic", "rcd")):
            full_be = GPUBackend(iter(raws), 100, n, interp, debayer)
            captured = {}
            with (
                patch.object(full_be, "detect", return_value=(np.ones((20, 6)), 3.0)),
                patch("gpucoadd.register", return_value=(mats, [np.ones(3)] * n)),
            ):
                stack(full_be, analysis_cache=None, coverage_only=True, on_analysis=captured.update, **common)
            region = [2, 3, 70, 60]
            load = frames_for_region(captured, n, W, H, 0, region, full_be.edge)
            self.assertLess(len(load), n)
            part_be = GPUBackend(iter([raws[i] for i in load]), 100, n, interp, debayer, indices=load)
            with patch.object(GPUBackend, "stats", return_value=stats):
                full, _, fi = stack(full_be, analysis_cache=captured, region=region, **common)
                part, _, pi = stack(part_be, analysis_cache=captured, region=region, **common)
            with self.subTest(interp=interp, debayer=debayer):
                for key in (
                    "registered",
                    "ref_idx",
                    "dropped",
                    "alignment_failed",
                    "retained_indices",
                    "contributing_indices",
                    "canvas",
                ):
                    self.assertEqual(fi[key], pi[key], key)
                self.assertEqual(pi["loaded_frames"], len(load))
                np.testing.assert_array_equal(pi["coverage"], fi["coverage"])
                np.testing.assert_array_equal(np.isnan(part), np.isnan(full))
                np.testing.assert_allclose(part, full, rtol=1e-6, atol=1e-3, equal_nan=True)
            # Guards: a reaching light or the reference missing from the loaded set is refused.
            reach = [i for i in load if i != captured["ref_idx"]]
            short = GPUBackend(iter([raws[i] for i in load[:-1]]), 100, n, interp, debayer, indices=load[:-1])
            with patch.object(GPUBackend, "stats", return_value=stats), self.assertRaises(ValueError):
                stack(short, analysis_cache=captured, region=region, **common)
            no_ref = GPUBackend(iter([raws[i] for i in reach]), 100, n, interp, debayer, indices=reach)
            with patch.object(GPUBackend, "stats", return_value=stats), self.assertRaises(ValueError):
                stack(no_ref, analysis_cache=captured, region=region, **common)
            with self.assertRaises(ValueError):  # detection without a cache needs every light
                stack(no_ref, analysis_cache=None, **common)


class TileSelectionTests(unittest.TestCase):
    """Per-tile light selection (only the lights reaching a tile are stacked there) gives the stack
    of every light in every tile: same coverage, NaN mask, contributing lights and values."""

    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

    def test_selected_equals_every_light(self):
        rng = np.random.default_rng(5)
        H, W, n = 96, 80, 9
        raws = [
            (2000 + 40 * k + rng.normal(0, 30 + 3 * k, (H, W)) + 4 * np.mgrid[:H, :W][1])
            .clip(0, 65535)
            .astype(np.uint16)
            for k in range(n)
        ]
        for k in (1, 6):  # outliers for the rejection to find
            raws[k][rng.integers(0, H, 40), rng.integers(0, W, 40)] = 60000
        # a mosaic: overlapping lights plus far ones, so the union canvas has tiles reached by
        # few lights and tiles reached by none
        shifts = [(0, 0), (3.3, -2.1), (-2.7, 1.4), (150, 0), (150, 120), (0, 140), (-1.5, -3.5), (300, 0), (1.2, 2.2)]
        mats = [np.linalg.inv(forward(0.01 * k, dx, dy)) for k, (dx, dy) in enumerate(shifts)]
        common = dict(tile=32, reg_workers=1, framing="union", select=0, log=lambda _: None)
        for interp, debayer, weight, rejection in (
            ("bilinear", "bilinear", "none", "sigma"),
            ("bicubic", "rcd", "noise", "noise"),
            ("bicubic", "rcd", "noise", "winsorized"),
        ):
            be = GPUBackend(iter(raws), 100, n, interp, debayer)
            captured = {}
            with (
                patch.object(be, "detect", return_value=(np.ones((20, 6)), 3.0)),
                patch("gpucoadd.register", return_value=(mats, [np.ones(3)] * n)),
            ):
                stack(be, analysis_cache=None, coverage_only=True, on_analysis=captured.update, **common)
            runs = {}
            for mode in ("selected", "all"):
                with patch.dict(os.environ, {"GPUCOADD_TILE_LIGHTS": mode}):
                    runs[mode] = stack(be, analysis_cache=captured, weight=weight, rejection=rejection, **common)
            (sel, _, si), (every, _, ei) = runs["selected"], runs["all"]
            with self.subTest(interp=interp, debayer=debayer, weight=weight, rejection=rejection):
                self.assertGreater(si["tile_lights"]["empty"], 0)
                self.assertLess(si["tile_lights"]["mean"], n / 2)
                self.assertEqual(ei["tile_lights"]["mean"], n)
                for key in ("contributing_indices", "canvas"):
                    self.assertEqual(si[key], ei[key], key)
                np.testing.assert_array_equal(si["coverage"], ei["coverage"])
                np.testing.assert_array_equal(np.isnan(sel), np.isnan(every))
                np.testing.assert_allclose(sel, every, rtol=1e-6, atol=1e-3, equal_nan=True)

    def test_auto_tiles_and_allocation_retry_preserve_coverage(self):
        import torch
        from tileplan import memory_estimate

        rng = np.random.default_rng(12)
        raws = [(2000 + rng.normal(0, 20, (96, 80))).astype(np.uint16) for _ in range(3)]
        be = GPUBackend(iter(raws), 100, 3, "bilinear", "bilinear")
        mats = [np.array([[1.0, 0, -dx], [0, 1, 0], [0, 0, 1]]) for dx in (0, 100, 200)]
        cache = {"mats": [m.tolist() for m in mats], "fwhm": [3.0] * 3, "ref_idx": 0, "ratios": [[1.0] * 3] * 3}
        common = dict(reg_workers=1, analysis_cache=cache, framing="union", select=0, log=lambda _: None)
        reference, _, ri = stack(be, tile=256, **common)
        # On this small canvas the 512 layout reserves more than the 256 layout.
        bound512 = memory_estimate(512, 280, 96, [3], debayer="bilinear")["required_bytes"]
        bound256 = memory_estimate(256, 280, 96, [3], debayer="bilinear")["required_bytes"]
        with patch.object(be, "free_vram", return_value=(bound512 + bound256) // 2):
            auto, _, ai = stack(be, tile=0, **common)
        self.assertEqual(ai["tile_plan"]["selected"], 256)
        np.testing.assert_array_equal(ai["coverage"], ri["coverage"])
        np.testing.assert_allclose(auto, reference, rtol=1e-6, atol=1e-3, equal_nan=True)
        with patch.object(be, "free_vram", return_value=0), self.assertRaisesRegex(MemoryError, "GPU memory"):
            stack(be, tile=0, **common)
        original = be.warp_tile
        failures = []

        def fail_once(st, frames, matrices, coefs, x0, x1, y0, y1):
            if not failures:
                failures.append(True)
                raise torch.OutOfMemoryError("injected tile failure")
            return original(st, frames, matrices, coefs, x0, x1, y0, y1)

        with patch.object(be, "warp_tile", side_effect=fail_once):
            retry, _, info = stack(be, tile=256, **common)
        self.assertEqual(len(info["tile_plan"]["retries"]), 1)
        self.assertEqual(info["tile_plan"]["actual_tiles"], ri["tile_plan"]["actual_tiles"])
        np.testing.assert_array_equal(info["coverage"], ri["coverage"])
        np.testing.assert_array_equal(np.isnan(retry), np.isnan(reference))
        self.assertEqual(info["contributing_indices"], ri["contributing_indices"])
        np.testing.assert_array_equal(retry, reference)


if __name__ == "__main__":
    unittest.main()
