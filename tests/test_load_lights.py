"""load_lights: plain 16-bit lights read into pinned memory and converted on the GPU (or in RAM)
must give exactly read_light's values; other lights still come from read_light."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fits_input import read_light  # noqa: E402


def cuda():
    import torch

    return torch.cuda.is_available()


class LoadLights(unittest.TestCase):
    def setUp(self):
        if not cuda():
            self.skipTest("CUDA is not available")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        rng = np.random.default_rng(9)
        self.paths = []
        for k in range(12):  # plain 16-bit, as the Seestar writes them
            data = rng.integers(0, 65536, (48, 64), dtype=np.uint16)
            data[0, :4] = [0, 32767, 32768, 65535]
            path = self.dir / f"plain_{k}.fit"
            fits.writeto(path, data, fits.Header({"BAYERPAT": "GRBG"}))
            self.paths.append(str(path))
        # bottom-up with a Bayer pattern that needs a row and a column cut off
        data = rng.integers(0, 65536, (41, 27), dtype=np.uint16)
        fits.writeto(self.dir / "bottom_up.fit", data, fits.Header({"ROWORDER": "BOTTOM-UP", "BAYERPAT": "BGGR"}))
        self.paths.append(str(self.dir / "bottom_up.fit"))
        # float data: read_light's path
        fits.writeto(self.dir / "float.fit", rng.uniform(0, 1, (48, 64)).astype(np.float32),
                     fits.Header({"BAYERPAT": "GRBG"}))
        self.paths.append(str(self.dir / "float.fit"))

    def test_gpu_and_host_values_equal_read_light(self):
        import torch
        import gpucoadd as g

        kinds = []
        for path, light in zip(self.paths, g.load_lights(self.paths, workers=3, ahead=4)):
            expected = read_light(path)
            kinds.append(type(light).__name__)
            if isinstance(light, g.PinnedLight):
                self.assertEqual(tuple(light.shape), expected.shape)
                got = light.cuda(torch, torch.device("cuda")).cpu().numpy().view(np.uint16)
            else:
                got = light
            np.testing.assert_array_equal(got, expected, path)
        self.assertEqual(kinds, ["PinnedLight"] * 13 + ["ndarray"])
        for path, light in zip(self.paths[:13], g.load_lights(self.paths[:13], ahead=2)):
            np.testing.assert_array_equal(light.host(), read_light(path), path)

    def test_buffers_are_reused(self):
        import torch
        import gpucoadd as g

        lights = g.load_lights(self.paths[:12], workers=2, ahead=3)
        pools = set()
        for light in lights:
            pools.add(id(light.pool))
            light.cuda(torch, torch.device("cuda"))
        pool = light.pool
        self.assertEqual(len(pools), 1)
        self.assertLessEqual(len(pool.buffers), 4, "at most `ahead` read ahead plus the one in use")

    def test_truncated_and_changed_files_raise(self):
        import gpucoadd as g
        from source_integrity import SourceFileError, SourceGuard

        short = self.dir / "short.fit"
        fits.writeto(short, np.zeros((48, 64), np.uint16), fits.Header({"BAYERPAT": "GRBG"}))
        data = short.read_bytes()
        short.write_bytes(data[: len(data) - 4000])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            list(g.load_lights([str(short)]))
        guard = SourceGuard(self.paths[:2])
        with open(self.paths[1], "ab") as f:
            f.write(b"\0" * 2880)
        with self.assertRaises(SourceFileError):
            list(g.load_lights(self.paths[:2], source_guard=guard))

    def test_backend_stores_the_same_raw_lights(self):
        import torch
        import gpucoadd as g

        paths = self.paths[:12]
        for storage in ("gpu", "ram"):
            with self.subTest(storage=storage):
                old = g.GPUBackend(g.load_frames(paths), 100, len(paths), paths=paths, sizes=[(48, 64)] * 12,
                                   storage=storage)
                new = g.GPUBackend(g.load_lights(paths), 100, len(paths), paths=paths, sizes=[(48, 64)] * 12,
                                   storage=storage)
                self.assertEqual(new.storage_mode, storage)
                for a, b in zip(old.raw, new.raw):
                    self.assertEqual(a.device, b.device)
                    self.assertTrue(torch.equal(a.cpu(), b.cpu()))


if __name__ == "__main__":
    unittest.main()
