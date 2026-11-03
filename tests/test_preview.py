"""Preview stretch (autostretch): statistics from covered pixels only, so sparse mosaics and
degenerate images still give a usable, finite preview."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpucoadd import autostretch  # noqa: E402


class PreviewTests(unittest.TestCase):
    def sky(self, shape, rng):
        img = 100 + rng.normal(0, 10, shape + (3,))
        img[shape[0] // 2, shape[1] // 2] += 5000  # a star
        return img.astype(np.float32)

    def test_sparse_mosaic(self):
        """95% of the canvas uncovered (NaN): the covered sky gets a normal stretch (background
        near 0.25), not white; uncovered pixels are black; nothing is NaN."""
        rng = np.random.default_rng(1)
        img = np.full((400, 400, 3), np.nan, np.float32)
        img[:90, :90] = self.sky((90, 90), rng)
        out = autostretch(img)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((out[100:, 100:] == 0).all())
        self.assertAlmostEqual(float(np.median(out[:90, :90])), 0.25, delta=0.05)

    def test_full_field_matches_dense_statistics(self):
        """No uncovered pixels: the same stretch as before (statistics over every pixel)."""
        rng = np.random.default_rng(2)
        img = self.sky((120, 160), rng)
        out = autostretch(img)
        self.assertAlmostEqual(float(np.median(out)), 0.25, delta=0.02)

    def test_degenerate_images(self):
        for img in (np.full((20, 20, 3), np.nan, np.float32), np.full((20, 20, 3), 7.0, np.float32)):
            with self.subTest(value=float(img.flat[0])):
                out = autostretch(img)
                self.assertTrue(np.isfinite(out).all())
                self.assertEqual(out.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
