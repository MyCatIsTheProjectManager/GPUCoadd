"""Preview stretch (autostretch): statistics from covered pixels only, so sparse mosaics and
degenerate images still give a usable, finite preview."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpucoadd import autostretch, preview, preview_image  # noqa: E402


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


class PreviewImageTests(unittest.TestCase):
    """preview / preview_image: downscaling must keep uncovered canvas out of the statistics."""

    def test_sparse_canvas_through_preview(self):
        """800 x 800 canvas, 14% covered by sky at 500 +- 20: the covered sky keeps a normal
        stretch (near 0.25 of 255), uncovered and partly covered preview pixels are black."""
        rng = np.random.default_rng(3)
        img = np.full((800, 800, 3), np.nan, np.float32)
        img[:300, :300] = 500 + rng.normal(0, 20, (300, 300, 3))
        out = preview_image(img)
        self.assertEqual(out.shape, (200, 200, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertAlmostEqual(float(np.median(out[:75, :75])) / 255, 0.25, delta=0.05)
        self.assertTrue((out[75:] == 0).all() and (out[:, 75:] == 0).all())

    def test_partly_covered_preview_pixels(self):
        """A NaN in one channel, or an infinity, makes its whole preview pixel uncovered."""
        rng = np.random.default_rng(4)
        img = (500 + rng.normal(0, 20, (40, 40, 3))).astype(np.float32)
        img[0, 0, 1] = np.nan
        img[8, 8, 2] = np.inf
        out = preview_image(img)
        self.assertTrue((out[0, 0] == 0).all() and (out[2, 2] == 0).all())
        self.assertAlmostEqual(float(np.median(out)) / 255, 0.25, delta=0.05)

    def test_dense_image_unchanged(self):
        """Fully covered: the same as stretching the downscaled image directly."""
        import cv2

        rng = np.random.default_rng(5)
        img = (100 + rng.normal(0, 10, (64, 96, 3))).astype(np.float32)
        small = cv2.resize(img, (24, 16), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(preview_image(img), (autostretch(small) * 255).astype(np.uint8))

    def test_dimensions_below_four(self):
        rng = np.random.default_rng(6)
        for shape in ((3, 8), (8, 3), (1, 1), (2, 2)):
            with self.subTest(shape=shape):
                img = (100 + rng.normal(0, 10, shape + (3,))).astype(np.float32)
                out = preview_image(img)
                self.assertEqual(out.shape, (max(1, shape[0] // 4), max(1, shape[1] // 4), 3))

    def test_writes_png(self):
        import tempfile

        import cv2

        img = np.full((12, 20, 3), np.nan, np.float32)
        img[:4, :4] = 7.0
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "p.png")
            preview(img, path)
            self.assertEqual(cv2.imread(path).shape, (3, 5, 3))


if __name__ == "__main__":
    unittest.main()
