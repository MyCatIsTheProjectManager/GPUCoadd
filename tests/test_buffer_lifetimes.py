"""Last-use release without altering registration, batching or arithmetic."""

import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpucoadd as g


class HostLifetimeTests(unittest.TestCase):
    def test_stars_are_freed_after_registration_before_coverage(self):
        raws = [np.random.default_rng(i).integers(1000, 3000, (64, 80), np.uint16) for i in range(4)]
        be = g.CPUBackend(raws, 0, 1)
        references = []

        def detect(cfa):
            stars = np.ones((200, 6), np.float32)
            references.append(weakref.ref(stars))
            return stars, 3.0

        def register(stars, *args, **kwargs):
            self.assertEqual(len(stars), len(raws))
            self.assertTrue(all(ref() is not None for ref in references))
            return [np.eye(3) for _ in raws], [np.ones(3) for _ in raws]

        original = g.footprint_coverage

        def coverage(*args, **kwargs):
            self.assertTrue(all(ref() is None for ref in references), "star measurements survive registration")
            return original(*args, **kwargs)

        with (
            patch.object(be, "detect", new=detect),
            patch.object(g, "register", new=register),
            patch.object(g, "footprint_coverage", new=coverage),
        ):
            _, _, info = g.stack(be, 32, 1, coverage_only=True, log=lambda _: None)
        self.assertEqual(info["registered"], len(raws))


class DeviceLifetimeTests(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        # Ample free RAM unless a test simulates pressure itself: results must not depend on what else
        # is running on the machine.
        for target in (
            "raw_storage.available_ram",
            "output_store.available_ram",
            "resource_guard.available_ram",
            "sky_store.available_ram",
        ):
            p = patch(target, return_value=64 << 30)
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        import torch

        gc.collect()
        torch.cuda.empty_cache()

    def test_warp_batches_drop_rgb_and_reuse_calibration_storage(self):
        import torch

        raws = [np.random.default_rng(i).integers(1000, 3000, (80, 96), np.uint16) for i in range(40)]
        be = g.GPUBackend(iter(raws), 0, len(raws), "bicubic", "rcd", storage="ram")
        if be.calibration is None or be.warp is None:
            self.skipTest("CUDA calibration/warp unavailable")
        original_cal = be.calibration.calibrate_pad
        original_warp = be.warp.warp
        rgb_refs = []
        pointers = []
        shapes = []

        def calibrate(*args, **kwargs):
            self.assertTrue(all(ref() is None for ref in rgb_refs), "previous batch RGB is still live")
            pointers.append(args[-1].data_ptr())
            shapes.append(tuple(args[-1].shape))
            return original_cal(*args, **kwargs)

        def warp(rgb, *args, **kwargs):
            rgb_refs.append(weakref.ref(rgb))
            return original_warp(rgb, *args, **kwargs)

        be.calibration = SimpleNamespace(calibrate_pad=calibrate)
        be.warp = SimpleNamespace(warp=warp)
        out = torch.empty((40, 3, 32, 32), device="cuda")
        be.warp_tile(
            out,
            list(range(40)),
            [np.eye(3)] * 40,
            {i: (np.ones(3, np.float32), np.zeros(3, np.float32)) for i in range(40)},
            16,
            48,
            16,
            48,
        )
        self.assertGreaterEqual(len(pointers), 2)
        self.assertEqual(len(set(pointers)), 1, "fitting calibration batches must reuse storage")
        self.assertEqual([shape[0] for shape in shapes], [32, 8], "batch membership changed")
        self.assertEqual(len(rgb_refs), len(pointers), "the accelerated warp must be exercised")
        self.assertTrue(all(ref() is None for ref in rgb_refs))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_unsynchronized_nondefault_stream_preserves_warp_pixels(self):
        import torch

        raws = [np.random.default_rng(i).integers(1000, 3000, (80, 96), np.uint16) for i in range(40)]
        be = g.GPUBackend(iter(raws), 0, len(raws), "bicubic", "rcd", storage="ram")
        frames = list(range(40))
        mats = [np.eye(3)] * 40
        coefs = {i: (np.ones(3, np.float32), np.zeros(3, np.float32)) for i in frames}
        expected = torch.empty((40, 3, 32, 32), device="cuda")
        be.warp_tile(expected, frames, mats, coefs, 16, 48, 16, 48)
        be.timer = g.Timer(None)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            actual = torch.empty_like(expected)
            be.warp_tile(actual, frames, mats, coefs, 16, 48, 16, 48)
        stream.synchronize()
        np.testing.assert_array_equal(actual.cpu().numpy(), expected.cpu().numpy())

    def test_sort_indices_and_previous_noise_are_released_at_last_use(self):
        import torch

        n = 17
        be = g.GPUBackend(iter([np.ones((64, 80), np.uint16)] * n), 0, n, storage="ram")
        if be.sort is None or be.clip is None:
            self.skipTest("CUDA sort/clip unavailable")
        original_sort = be.sort.sort
        original_clip = be.clip.clip_noise
        index_refs = []
        noise_refs = []
        calls = []

        def sort(values):
            self.assertTrue(all(ref() is None for ref in noise_refs), "previous chunk noise is still live")
            result = original_sort(values)
            index_refs.append(weakref.ref(result.indices))
            calls.append(tuple(values.shape))
            return result

        def clip(values, noise, *args):
            self.assertTrue(all(ref() is None for ref in index_refs), "sort indices are still live during rejection")
            noise_refs.append(weakref.ref(noise))
            return original_clip(values, noise, *args)

        be.sort = SimpleNamespace(sort=sort)
        be.clip = SimpleNamespace(clip_noise=clip)
        data = torch.from_numpy(np.random.default_rng(44).normal(1000, 2, (n, 3, 256, 256)).astype(np.float32)).cuda()
        image, kept = be.integrate(data, 3.0, 3, rejection="noise", noise=np.ones((n, 3), np.float32), kept=True)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(all(ref() is None for ref in noise_refs))
        self.assertEqual(image.shape, (256, 256, 3))
        self.assertTrue((kept <= n).all())


if __name__ == "__main__":
    unittest.main()
