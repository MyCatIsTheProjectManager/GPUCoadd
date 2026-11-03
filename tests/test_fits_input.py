"""Lights in other FITS layouts (Bayer patterns, offsets, row order, BITPIX, BZERO/BSCALE) read
into the engine's layout: GRBG from the first pixel, uint16, top row first, even dimensions."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import fits_input  # noqa: E402
import lightgroups  # noqa: E402
from fits_layouts import variant, write_raw  # noqa: E402
from raw_storage import DiskLight  # noqa: E402


def sensor(h=14, w=18, low=0, high=65536, seed=0):
    """A GRBG light as the engine reads it."""
    return np.random.default_rng(seed).integers(low, high, (h, w), dtype=np.uint16)


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, stored, cards):
        path = Path(self.dir.name) / f"{name}.fit"
        write_raw(path, stored, cards)
        return path

    def check(self, path, expected):
        """read_light, disk crops and the header layout all agree with `expected`."""
        got = fits_input.read_light(path)
        self.assertEqual(got.dtype, np.uint16)
        np.testing.assert_array_equal(got, expected)
        layout = fits_input.Layout(fits.getheader(path))
        parsed = fits_input.primary_header(path)[1]
        self.assertEqual(vars(layout), vars(fits_input.Layout(parsed)))  # own parser == astropy
        self.assertEqual(layout.shape, expected.shape)
        light = DiskLight(path, expected.shape)
        self.assertIsNotNone(light.simple)
        h, w = expected.shape
        for box in ((0, h, 0, w), (0, 2, 0, 4), (3, h - 1, 5, w - 2), (h - 3, h, 1, w)):
            a, b, c, d = box
            np.testing.assert_array_equal(light.crop(*box), expected[a:b, c:d])
            np.testing.assert_array_equal(fits_input.read_light(path, box), expected[a:b, c:d])
        return layout

    def test_patterns_by_leaving_out_rows_and_columns(self):
        image = sensor()
        for name, cut in (("rggb", (0, 2)), ("bggr", (2, 0)), ("gbrg", (2, 2))):
            with self.subTest(name):
                path = self.write(name, *variant(image, name))
                layout = self.check(path, image[cut[0] :, cut[1] :])
                self.assertFalse(layout.native)
                ref = self.write(name + "-ref", *variant(image, name + "-ref"))
                np.testing.assert_array_equal(fits_input.read_light(ref), fits_input.read_light(path))

    def test_layouts_that_keep_every_pixel(self):
        image = sensor(low=1000, high=60000)
        for name in ("bottom-up", "xoff", "yoff", "f32", "f64", "i32", "bzero"):
            with self.subTest(name):
                layout = self.check(self.write(name, *variant(image, name)), image)
                self.assertEqual(layout.native, name != "bottom-up")
        seestar = self.write("seestar", *variant(image, "xoff"))  # only to exercise the BZERO 32768 path
        self.assertEqual(fits_input.Layout(fits.getheader(seestar)).bzero, 32768)

    def test_odd_sizes_and_bottom_up_parity(self):
        image = sensor(15, 19)
        self.check(self.write("odd", *variant(image, "f32")), image[:14, :18])
        # odd height stored bottom first: the first stored row is a G R row
        stored, cards = variant(image, "bottom-up")
        self.assertEqual(cards["BAYERPAT"], "GRBG")
        self.check(self.write("odd-bottom-up", stored, cards), image[:14, :18])
        # BGGR from the top in a 13-row light, stored bottom first: the first stored row is B G
        self.check(
            self.write("bggr-bottom-up", image[1:14][::-1], {"BAYERPAT": "BGGR", "ROWORDER": "BOTTOM-UP"}),
            image[2:14, :18],
        )
        self.check(self.write("top-down", image, {"BAYERPAT": "GRBG", "ROWORDER": "TOP-DOWN"}), image[:14, :18])

    def test_values(self):
        image = sensor(high=32768)
        self.check(self.write("signed", image.astype(np.int16), {"BAYERPAT": "GRBG"}), image)
        even = image & 0xFFFE
        self.check(self.write("bscale", (even // 2).astype(np.int16), {"BSCALE": 2, "BAYERPAT": "GRBG"}), even)
        bytes8 = sensor(high=256).astype(np.uint8)
        self.check(self.write("u8", bytes8, {"BAYERPAT": "GRBG"}), bytes8.astype(np.uint16) * 256)
        full = sensor()
        for dtype in (np.float32, np.float64):  # normalized 0..1
            self.check(self.write("norm", (full / 65535).astype(dtype), {"BAYERPAT": "GRBG"}), full)
        self.check(self.write("fraction", full.astype(np.float64) + 0.25, {"BAYERPAT": "GRBG"}), full)

    def test_values_that_cannot_be_read(self):
        bad = sensor().astype(np.float32)
        bad[3, 4] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite pixels"):
            fits_input.read_light(self.write("nan", bad, {"BAYERPAT": "GRBG"}))
        with self.assertRaisesRegex(Exception, "Non-finite pixels"):
            DiskLight(self.write("nan-disk", bad, {"BAYERPAT": "GRBG"}), (14, 18))
        with self.assertRaisesRegex(ValueError, "outside 0..65535"):
            fits_input.read_light(self.write("big", sensor().astype(np.int32) + 1000, {"BAYERPAT": "GRBG"}))
        with self.assertRaisesRegex(ValueError, "outside 0..65535"):
            fits_input.read_light(self.write("neg", sensor().astype(np.float32) - 70000, {"BAYERPAT": "GRBG"}))
        with self.assertRaisesRegex(ValueError, "outside 0..65535"):
            u8 = sensor(high=256).astype(np.uint8)
            fits_input.read_light(self.write("u8big", u8, {"BAYERPAT": "GRBG", "BZERO": 1}))

    def test_header_reasons_and_light_groups(self):
        def light(**cards):
            h = {"BITPIX": 16, "NAXIS": 2, "NAXIS1": 18, "NAXIS2": 14, "BAYERPAT": "GRBG", **cards}
            return lightgroups.describe(Path("x.fit"), {k: v for k, v in h.items() if v is not None})

        self.assertIsNone(light()["unusable"])
        self.assertEqual(light()["conversion"], "")
        self.assertEqual(light(NAXIS=3)["unusable"], "not a single raw image (colour or multi-plane data)")
        self.assertEqual(light(BITPIX=12)["unusable"], "BITPIX 12 is not supported")
        self.assertEqual(light(BAYERPAT=None)["unusable"], "no Bayer pattern in the header (BAYERPAT)")
        self.assertEqual(light(ROWORDER="SIDEWAYS")["unusable"], "ROWORDER SIDEWAYS is not supported")
        rggb = light(BAYERPAT="RGGB", ROWORDER="BOTTOM-UP")
        # 14 rows stored bottom first: top-down the pattern is GBRG
        self.assertEqual((rggb["width"], rggb["height"], rggb["sensor"]), (16, 12, (18, 14)))
        self.assertEqual(
            rggb["conversion"],
            "rows turned top first (ROWORDER BOTTOM-UP); first and last row, first and last column left out "
            "to start on the GRBG layout with even dimensions",
        )
        self.assertEqual(
            light(BAYERPAT="BGGR", NAXIS2=15)["conversion"],
            "first row left out to start on the GRBG layout with even dimensions",
        )
        self.assertEqual(light(BITPIX=8, BIAS=8)["bias"], 2048)
        self.assertEqual(light(BITPIX=-32, BIAS=2000)["bias"], 2000)

    def test_quoted_values(self):
        self.assertEqual(fits_input._card_value(" 'GRBG    '           / Bayer pattern"), "GRBG")
        self.assertEqual(fits_input._card_value(" 'it''s / x' / comment"), "it's / x")
        self.assertEqual(fits_input._card_value("                32768 / offset"), "32768")


if __name__ == "__main__":
    unittest.main()
