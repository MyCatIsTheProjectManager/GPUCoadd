"""Exact read-back verification of written FITS files: every byte compared, failures and cancels handled."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

import fits_output
from worker.bridge import Cancelled


class ExactVerificationTests(unittest.TestCase):
    def test_short_reads_and_premature_end_are_handled(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pixels.fits"
            body = np.arange(3 * 257 * 259 * 4, dtype=np.uint8)
            fits_output.write_encoded(path, fits.Header(), (3, 257, 259), body)
            original_open = Path.open
            for stop_early in (False, True):
                class Reader:
                    def __init__(self, stream):
                        self.stream, self.reads = stream, 0
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        self.stream.close()
                    def __getattr__(self, key):
                        return getattr(self.stream, key)
                    def readinto(self, target):
                        self.reads += 1
                        if stop_early and self.reads == 2:
                            return 0
                        return self.stream.readinto(target[:32767])
                def limited_open(path, mode="r", *a, **kw):
                    stream = original_open(path, mode, *a, **kw)
                    return Reader(stream) if mode == "rb" else stream
                with patch.object(Path, "open", limited_open):
                    if stop_early:
                        with self.assertRaisesRegex(OSError, "Truncated FITS data"):
                            fits_output.verify_fits(path, (3, 257, 259), body.nbytes, None, expected_bytes=body)
                    else:
                        fits_output.verify_fits(path, (3, 257, 259), body.nbytes, None, expected_bytes=body)

    def test_cancel_during_readback_preserves_old_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pixels.fits"
            path.write_bytes(b"previous result")
            body = np.zeros(3 * 1024 * 1025 * 4, np.uint8)
            verify = fits_output.verify_fits
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise Cancelled("during verification")
            def checking(path, shape, size, digest, check, header, expected):
                return verify(path, shape, size, digest, cancel, header, expected)
            with patch("fits_output.verify_fits", side_effect=checking):
                with self.assertRaises(Cancelled):
                    fits_output.write_encoded(path, fits.Header(), (3, 1024, 1025), body)
            self.assertEqual(path.read_bytes(), b"previous result")
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_fallback_comparison_and_multiblock_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pixels.fits"
            shape = (3, 1024, 1025)
            body = np.random.default_rng(7).integers(0, 256, np.prod(shape) * 4, np.uint8)
            fits_output.write_encoded(path, fits.Header(), shape, body)
            with patch("fits_output._memcmp", None):
                fits_output.verify_fits(path, shape, body.nbytes, None, expected_bytes=body)
            for position in (0, (4 << 20) - 1, 4 << 20, body.nbytes - 1):
                with path.open("r+b") as stream:
                    stream.seek(2880 + position)
                    stream.write(bytes([int(body[position]) ^ 1]))
                with self.assertRaisesRegex(OSError, "pixel verification"):
                    fits_output.verify_fits(path, shape, body.nbytes, None, expected_bytes=body)
                with path.open("r+b") as stream:
                    stream.seek(2880 + position)
                    stream.write(bytes([int(body[position])]))
            with self.assertRaises(ValueError):
                fits_output.verify_fits(path, shape, body.nbytes, None, expected_bytes=body[:-1])



if __name__ == "__main__":
    unittest.main()
