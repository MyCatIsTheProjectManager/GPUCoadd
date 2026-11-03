"""Bounding-box coverage rasterization must reproduce the original full-canvas rasterizer exactly."""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpucoadd import footprint_coverage


def reference_coverage(mats, frames, w, h, width, height):
    """The rasterizer as it was before the bounding-box change, verbatim."""
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]]).T
    coverage = np.zeros((height, width), np.uint32)
    mask = np.zeros((height, width), np.uint8)
    for i in frames:
        p = np.linalg.inv(mats[i]) @ corners
        if not np.isfinite(p).all() or not (np.all(p[2] > 0) or np.all(p[2] < 0)):
            raise ValueError("Unbounded or invalid frame footprint")
        polygon = np.rint((p[:2] / p[2]).T * 256).astype(np.int32)
        mask.fill(0)
        cv2.fillConvexPoly(mask, polygon, 1, lineType=cv2.LINE_8, shift=8)
        coverage += mask
    return coverage


def forward(angle=0.0, scale=1.0, tx=0.0, ty=0.0, px=0.0, py=0.0):
    """Source -> output matrix; px/py add a projective term."""
    c, s = scale * np.cos(angle), scale * np.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [px, py, 1.0]])


class CoverageEquality(unittest.TestCase):
    def check(self, fwd, w, h, width, height):
        mats = [np.linalg.inv(f) for f in fwd]
        frames = list(range(len(mats)))
        try:
            expected = reference_coverage(mats, frames, w, h, width, height)
        except ValueError:
            with self.assertRaises(ValueError):
                footprint_coverage(mats, frames, w, h, width, height)
            return
        got = footprint_coverage(mats, frames, w, h, width, height)
        self.assertEqual(got.dtype, expected.dtype)
        np.testing.assert_array_equal(got, expected)

    def test_hand_picked_edges(self):
        W, H = 37, 23
        cases = [
            forward(),  # exactly on the canvas
            forward(tx=-0.5, ty=-0.5),  # half-pixel vertices (rounding ties)
            forward(tx=0.49, ty=0.51),
            forward(tx=-W + 0.3, ty=4),  # sub-pixel sliver at the left edge
            forward(tx=79.7, ty=-H + 0.6),  # sliver at the right/top corner
            forward(tx=-W - 5, ty=0),  # completely outside (left)
            forward(tx=200, ty=200),  # completely outside (beyond)
            forward(tx=-1000.25, ty=-1000.75, scale=40),  # canvas inside a huge footprint
            forward(np.pi / 4, 1, 40, -20),  # rotated, partly above the canvas
            forward(np.pi / 2, 1, 10.5, 3.25),
            forward(0.3, 0.9, 30, 10, 2e-4, -3e-4),  # projective
        ]
        for k, f in enumerate(cases):
            with self.subTest(case=k):
                self.check([f], W, H, 80, 50)
        self.check(cases, W, H, 80, 50)  # accumulated

    def test_degenerate_frames(self):
        # one-pixel-wide/high frames rasterize as lines; zero-area must still match
        for w, h in ((1, 30), (30, 1), (1, 1), (2, 2)):
            for f in (forward(tx=3.4, ty=5.6), forward(0.7, 1, 20, 10), forward(tx=-0.5, ty=-0.5)):
                with self.subTest(w=w, h=h):
                    self.check([f], w, h, 40, 30)

    def test_invalid_footprint_still_raises(self):
        self.check([forward(px=0.1, py=0, tx=0) @ np.diag([1, 1, -2.0])], 80, 96, 50, 50)

    def test_random(self):
        rng = np.random.default_rng(20261022)
        for k in range(1500):
            width, height = rng.integers(1, 120, 2)
            w, h = rng.integers(1, 70, 2)
            n = int(rng.integers(1, 6))
            fwd = []
            for _ in range(n):
                p = rng.normal(0, 3e-4, 2) if rng.random() < 0.3 else (0, 0)
                fwd.append(
                    forward(
                        rng.uniform(-np.pi, np.pi),
                        rng.uniform(0.3, 2.5),
                        rng.uniform(-1.5 * w, width + 0.5 * w),
                        rng.uniform(-1.5 * h, height + 0.5 * h),
                        *p,
                    )
                )
            with self.subTest(k=k):
                self.check(fwd, int(w), int(h), int(width), int(height))

    def test_trimmed_footprints(self):
        # edge = e equals the untrimmed rasterizer on a (w - 2e) x (h - 2e) sensor offset by e
        rng = np.random.default_rng(12)
        for k in range(300):
            e = int(rng.choice([1, 2, 3, 11, 12]))
            w, h = (int(v) for v in rng.integers(2 * e + 2, 2 * e + 60, 2))
            width, height = (int(v) for v in rng.integers(1, 100, 2))
            fwd = [
                forward(
                    rng.uniform(-np.pi, np.pi), rng.uniform(0.5, 2), rng.uniform(-w, width), rng.uniform(-h, height)
                )
            ]
            mats = [np.linalg.inv(f) for f in fwd]
            shifted = [np.array([[1, 0, -e], [0, 1, -e], [0, 0, 1.0]]) @ m for m in mats]
            with self.subTest(k=k):
                np.testing.assert_array_equal(
                    footprint_coverage(mats, [0], w, h, width, height, edge=e),
                    reference_coverage(shifted, [0], w - 2 * e, h - 2 * e, width, height),
                )

    def test_seestar_scale(self):
        # Real sensor sizes and a large mosaic-like canvas with negative/offset placements
        rng = np.random.default_rng(7)
        fwd = [
            forward(rng.uniform(-0.4, 0.4), 1, rng.uniform(-3000, 9000), rng.uniform(-3000, 7000)) for _ in range(12)
        ]
        self.check(fwd, 2160, 3840, 8000, 6000)
        self.check(fwd, 1080, 1920, 8000, 6000)


if __name__ == "__main__":
    unittest.main()
