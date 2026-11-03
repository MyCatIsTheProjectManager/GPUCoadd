"""Fast light reader: plain FITS read directly (Seestar files bit for bit), the rest through astropy."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fits_input  # noqa: E402
import gpucoadd as g  # noqa: E402


class LoaderTests(unittest.TestCase):
    def test_fast_path_matches_astropy(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 65536, (38, 54), dtype=np.uint16)
        data[0, :4] = (0, 1, 32767, 65535)  # sign-bit edge cases
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "light.fit"
            h = fits.Header()
            h["BAYERPAT"] = "GRBG"
            for k in range(40):  # a header longer than one 2880-byte block
                h[f"KEY{k}"] = ("x" * 40, "long comment " * 2)
            fits.writeto(p, data, h)  # astropy writes uint16 as int16 with BZERO 32768
            self.assertIsNotNone(fits_input._direct(str(p)))
            got = g.read_light(str(p))
            self.assertEqual(got.dtype, np.uint16)
            np.testing.assert_array_equal(got, data)
            np.testing.assert_array_equal(got, fits.getdata(p).astype(np.uint16))

    def test_other_layouts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "float.fit"
            data = np.arange(12, dtype=np.float32).reshape(3, 4) + 2
            fits.writeto(p, data)
            np.testing.assert_array_equal(g.read_light(str(p)), data[:2].astype(np.uint16))  # even height
            q = Path(d) / "signed.fit"
            fits.writeto(q, np.arange(-6, 6, dtype=np.int16).reshape(3, 4))  # BZERO absent
            with self.assertRaisesRegex(ValueError, "signed.fit has values outside 0..65535"):
                g.read_light(str(q))
            c = Path(d) / "compressed.fit"
            image = np.random.default_rng(1).integers(0, 65536, (6, 8), dtype=np.uint16)
            fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(image)]).writeto(c)
            self.assertIsNone(fits_input._direct(str(c)))
            np.testing.assert_array_equal(g.read_light(str(c)), image)

    def test_initially_truncated_or_nonfinite_light_is_identified(self):
        with tempfile.TemporaryDirectory() as d:
            short = Path(d) / "short.fit"
            fits.writeto(short, np.zeros((64, 64), np.uint16))
            with short.open("r+b") as stream:
                stream.truncate(3000)
            with self.assertRaisesRegex(ValueError, "Truncated light:.*short.fit"):
                g.read_light(short)
            bad = Path(d) / "nonfinite.fit"
            image = np.ones((16, 16), np.float32)
            image[4, 5] = np.nan
            fits.writeto(bad, image)
            with self.assertRaisesRegex(ValueError, "Non-finite pixels.*nonfinite.fit"):
                g.read_light(bad)


if __name__ == "__main__":
    unittest.main()
