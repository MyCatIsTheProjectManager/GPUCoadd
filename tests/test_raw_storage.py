"""Storage selection and read-only FITS crops; no large allocations needed."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
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

    def test_crop_reads_bottom_up_and_full_rows(self):
        from fits_input import Layout
        from gpucoadd import read_light

        values = np.random.default_rng(5).integers(0, 65536, (41, 25), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bottom_up.fit"
            fits.writeto(path, values, fits.Header({"ROWORDER": "BOTTOM-UP", "BAYERPAT": "BGGR"}))
            shape = Layout(fits.getheader(path)).shape  # Bayer parity: a row and a column left out
            self.assertEqual(shape, (40, 24))
            light = DiskLight(path, shape)
            self.assertIsNotNone(light.simple)
            whole = read_light(path)
            np.testing.assert_array_equal(light.crop(), whole)
            for box in ((0, 40, 0, 24), (3, 17, 0, 24), (39, 40, 5, 6), (10, 30, 7, 19)):
                a, b, c, e = box
                np.testing.assert_array_equal(light.crop(*box), whole[a:b, c:e])


class ReadAheadTests(unittest.TestCase):
    """raw_storage.ReadAhead: crops identical to DiskLight.crop, bounded, errors where used."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        rng = np.random.default_rng(6)
        self.lights, self.values = [], []
        for k in range(9):
            values = rng.integers(0, 65536, (48, 32), dtype=np.uint16)
            path = Path(self.temp.name) / f"light_{k}.fit"
            fits.writeto(path, values)
            self.lights.append(DiskLight(path, values.shape))
            self.values.append(values)

    def items(self):
        return [(k * 10, light, (k, 20 + k, 2, 30 - k)) for k, light in enumerate(self.lights)]

    def test_takes_in_order_match_direct_reads_within_limits(self):
        from raw_storage import ReadAhead

        ahead = ReadAhead(self.items(), max_crops=3)
        self.assertEqual(len(ahead.ahead), 3)
        for key, light, (a, b, c, d) in self.items():
            self.assertLessEqual(len(ahead.ahead), 3)
            np.testing.assert_array_equal(ahead.take(key), self.values[key // 10][a:b, c:d])
        self.assertEqual((ahead.ahead, ahead.bytes), ({}, 0))
        self.assertIsNone(ahead.take(0))  # taken already, or never read ahead: the caller reads it

    def test_byte_limit_keeps_at_least_one_crop_in_flight(self):
        from raw_storage import ReadAhead

        ahead = ReadAhead(self.items(), max_bytes=1)
        self.assertEqual(len(ahead.ahead), 1)
        ahead.take(0)
        self.assertEqual(list(ahead.ahead), [10])
        ahead.close()
        self.assertEqual((ahead.ahead, ahead.bytes), ({}, 0))
        self.assertIsNone(ahead.take(10))

    def test_read_error_is_raised_by_the_take_of_its_crop(self):
        from raw_storage import ReadAhead

        changed = self.lights[2].path
        with open(changed, "ab") as f:
            f.write(b"\0" * 2880)  # the file changed after its stamp was recorded
        ahead = ReadAhead(self.items())
        np.testing.assert_array_equal(ahead.take(0), self.values[0][0:20, 2:30])
        np.testing.assert_array_equal(ahead.take(10), self.values[1][1:21, 2:29])
        with self.assertRaisesRegex(SourceFileError, "changed"):
            ahead.take(20)
        np.testing.assert_array_equal(ahead.take(30), self.values[3][3:23, 2:27])
        ahead.close()


class DiskTileTests(unittest.TestCase):
    """A tile from lights on disk is identical with read-ahead on and off, and to the same tile
    from lights held in GPU memory."""

    def test_warp_tile_identical_with_and_without_read_ahead(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        import gpucoadd as g
        import raw_storage

        rng = np.random.default_rng(7)
        h, w, n = 96, 128, 10
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for k in range(n):
                path = Path(d) / f"light_{k}.fit"
                fits.writeto(path, rng.integers(200, 60000, (h, w), dtype=np.uint16))
                paths.append(str(path))
            mats = []
            for k in range(n):
                angle = np.deg2rad(rng.uniform(-6, 6))
                tx, ty = rng.uniform(-8, 8, 2)
                mats.append(np.array([[np.cos(angle), -np.sin(angle), tx], [np.sin(angle), np.cos(angle), ty], [0, 0, 1]]))
            mats[4] = mats[4] @ np.array([[1, 0, 500], [0, 1, 0], [0, 0, 1]])  # misses the tile: no crop
            coefs = [(np.full(3, 1.0 + k / 50, np.float32), np.full(3, k, np.float32)) for k in range(n)]

            def tile(storage, read_ahead):
                be = g.GPUBackend(g.load_frames(paths), 100, n, "bicubic", "rcd", paths=paths,
                                  sizes=[(h, w)] * n, storage=storage)
                be.read_ahead = read_ahead
                st = be.alloc_stack(n, 50, 70)
                be.warp_tile(st, list(range(n)), mats, coefs, 20, 90, 22, 72)
                return st.cpu().numpy().view(np.uint32).copy()

            gpu = tile("gpu", True)
            off = tile("disk", False)
            taken = []
            original = raw_storage.ReadAhead.take

            def take(self, key):
                value = original(self, key)
                taken.append(value is not None)
                return value

            with patch.object(raw_storage, "READ_AHEAD_CROPS", 2), patch.object(raw_storage.ReadAhead, "take", take):
                on = tile("disk", True)
        self.assertEqual(taken, [True] * (n - 1))
        np.testing.assert_array_equal(on, off)
        np.testing.assert_array_equal(on, gpu)
        self.assertTrue(np.isnan(on.view(np.float32)[4]).all())
        self.assertGreater(np.isfinite(on.view(np.float32)[[0, 1, 2, 3, 5]]).mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
