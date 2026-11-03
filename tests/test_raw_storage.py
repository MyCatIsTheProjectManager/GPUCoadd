"""Storage selection and read-only FITS crops; no large allocations needed."""

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_storage import plan, DiskLight, GIB
from source_integrity import SourceFileError


class RawStorageTests(unittest.TestCase):
    def test_gpu_ram_disk_selection_and_reserve(self):
        small = [(1920, 1080)] * 31
        self.assertEqual(plan(small, 11 * GIB, 12 * GIB, 32 * GIB)["mode"], "gpu")
        large = [(1920, 1080)] * 5194
        ram = plan(large, 11 * GIB, 12 * GIB, 64 * GIB)
        self.assertEqual(ram["mode"], "ram")
        self.assertEqual(ram["raw_bytes"], 5194 * 1920 * 1080 * 2)
        self.assertEqual(plan(large, 11 * GIB, 12 * GIB, 16 * GIB)["mode"], "disk")
        with self.assertRaises(MemoryError):
            plan(large, 11 * GIB, 12 * GIB, 64 * GIB, mode="gpu")
        with self.assertRaises(MemoryError):
            plan(large, 11 * GIB, 12 * GIB, 16 * GIB, mode="ram")
        with self.assertRaises(MemoryError):
            plan(large, 11 * GIB, 12 * GIB, 16 * GIB, has_paths=False)

    def test_disk_crops_exact_unsigned_bits_and_files_unchanged(self):
        from gpucoadd import read_light

        values = np.random.default_rng(4).integers(0, 65536, (64, 80), dtype=np.uint16)
        values[0, :4] = [0, 32767, 32768, 65535]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "light.fit"
            fits.writeto(path, values)
            before = path.read_bytes()
            light = DiskLight(path, values.shape)
            self.assertIsNotNone(light.simple)
            np.testing.assert_array_equal(light.crop(), read_light(path))
            for box in ((0, 1, 0, 4), (1, 43, 3, 66), (0, 64, 0, 80)):
                a, b, c, e = box
                np.testing.assert_array_equal(light.crop(*box), values[a:b, c:e])
            self.assertEqual(before, path.read_bytes())
            with self.assertRaises(ValueError):
                light.crop(0, 100)
            with self.assertRaises(ValueError):
                DiskLight(path, (32, 40))

    def test_scaled_fits_and_truncation(self):
        from gpucoadd import read_light

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "float.fit"
            fits.writeto(path, np.arange(48, dtype=np.float32).reshape(6, 8) + 0.5)
            light = DiskLight(path, (6, 8))
            self.assertIsNotNone(light.simple)
            np.testing.assert_array_equal(light.crop(1, 4, 2, 7), read_light(path)[1:4, 2:7])
            path = Path(d) / "short.fit"
            fits.writeto(path, np.zeros((64, 80), np.uint16))
            light = DiskLight(path, (64, 80))
            with path.open("r+b") as f:
                f.truncate(light.simple[1] + 20)
            with self.assertRaisesRegex(SourceFileError, "changed"):
                light.crop()


if __name__ == "__main__":
    unittest.main()
