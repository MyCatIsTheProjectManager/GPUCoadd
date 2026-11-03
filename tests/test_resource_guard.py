"""Pressure recovery preserves geometry/data; unsafe work fails before publication."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resource_guard import check_ram, check_gpu, check_tile, ResourcePressureError, RAM_RESERVE, set_handlers
from output_store import OutputWorkspace
from tileplan import render_same_tile, memory_estimate
import gpucoadd as g


class ResourceGuardTests(unittest.TestCase):
    def setUp(self):
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

    def test_low_ram_never_stops_a_job(self):
        """Short RAM returns False instead of raising; the periodic check first asks the job to
        free what it can, then warns once; an allocation check chooses its own disk form quietly."""
        warnings, relieved = [], []
        set_handlers(relieve=lambda need: relieved.append(need) or 0, warn=warnings.append)
        self.addCleanup(set_handlers)
        with patch("resource_guard.available_ram", return_value=RAM_RESERVE + 100):
            self.assertTrue(check_ram(100))
            self.assertFalse(check_ram(101))  # an allocation: no relief, no warning
            self.assertEqual((relieved, warnings), ([], []))
        with patch("resource_guard.available_ram", return_value=RAM_RESERVE - 10):
            self.assertFalse(check_ram(stage="stacking", relieve=True))
            self.assertFalse(check_ram(stage="stacking", relieve=True))
        self.assertEqual(relieved, [10, 10])
        self.assertEqual(len(warnings), 1)
        self.assertIn("GPUCoadd continues", warnings[0])
        with patch("resource_guard.available_ram", side_effect=[RAM_RESERVE - 10, RAM_RESERVE]):
            self.assertTrue(check_ram(relieve=True))  # the relief freed enough
        with patch("resource_guard.available_ram", return_value=0):
            self.assertTrue(check_ram())
        self.assertIn("could not be measured", warnings[-1])

    def test_gpu_recheck_releases_only_unused_allocator_blocks(self):
        backend = SimpleNamespace(free_vram=Mock(side_effect=[10, 30]), torch=Mock())
        self.assertEqual(check_gpu(backend, 20), 30)
        backend.torch.cuda.empty_cache.assert_called_once()
        backend.free_vram = Mock(return_value=10)
        with self.assertRaisesRegex(ResourcePressureError, "GPU memory fell"):
            check_gpu(backend, 20)

    def test_existing_tile_buffer_is_not_budgeted_twice(self):
        buf = SimpleNamespace(numel=lambda: 12 * 64 * 64, element_size=lambda: 4)
        estimate = memory_estimate(64, 64, 64, [4])
        need = estimate["required_bytes"] - buf.numel() * 4
        backend = SimpleNamespace(debayer="rcd", _stack_buf=buf, free_vram=Mock(return_value=need), torch=Mock())
        check_tile(backend, (0, 64, 0, 64), 4, "sigma")
        backend.torch.cuda.empty_cache.assert_not_called()

    def test_retry_keeps_exact_bounds_and_only_yields_completed_tile(self):
        bounds = (13, 141, 5, 133)
        render = Mock(side_effect=[MemoryError("pressure"), np.arange(9).reshape(3, 3)])
        release = Mock()
        retries = []
        result = list(render_same_tile(bounds, render, MemoryError, release, retries))
        self.assertEqual(render.call_args_list[0], render.call_args_list[1])
        np.testing.assert_array_equal(result[0][1], np.arange(9).reshape(3, 3))
        self.assertEqual([b for b, _ in result], [bounds])
        self.assertEqual(retries, [list(bounds)])
        release.assert_called_once_with(bounds)
        with self.assertRaises(MemoryError):
            list(render_same_tile(bounds, Mock(side_effect=MemoryError), MemoryError, release, []))
        with self.assertRaises(ValueError):
            list(render_same_tile(bounds, Mock(side_effect=ValueError), MemoryError, release, []))

    def test_auto_buffer_fallback_keeps_existing_array_exact(self):
        with tempfile.TemporaryDirectory() as d:
            with OutputWorkspace(d, d, {}, mode="auto") as work:
                work.plan(32, 24)
                existing = work.allocate("stack", (24, 32, 3), np.float32)
                existing[:] = np.arange(existing.size).reshape(existing.shape)
                expected = existing.copy()
                with patch("resource_guard.available_ram", side_effect=[RAM_RESERVE, RAM_RESERVE + 1]):
                    coverage = work.allocate("coverage", (24, 32), np.uint32, zero=True)
                self.assertIsInstance(coverage, np.memmap)
                self.assertEqual(work.info["mode"], "mixed")
                np.testing.assert_array_equal(existing, expected)
                np.testing.assert_array_equal(coverage, np.zeros((24, 32), np.uint32))
                directory = work.directory
            self.assertFalse(directory.exists())

    def test_forced_ram_stays_in_ram(self):
        with tempfile.TemporaryDirectory() as d:
            with OutputWorkspace(d, d, {}, mode="ram") as work:
                work.plan(32, 24)
                with patch("resource_guard.available_ram", return_value=1):
                    stack = work.allocate("stack", (24, 32, 3), np.float32)
                self.assertNotIsInstance(stack, np.memmap)
                self.assertIsNone(work.directory)

    def test_disk_loss_and_preview_ram_pressure_are_rechecked(self):
        with tempfile.TemporaryDirectory() as d:
            with OutputWorkspace(d, d, {"stack_preview": True}, mode="disk") as work:
                work.plan(32, 24)
                work.allocate("stack", (24, 32, 3), np.float32)
                with patch("output_store.shutil.disk_usage", return_value=SimpleNamespace(free=1)):
                    with self.assertRaisesRegex(OSError, "disk space"):
                        work.flush()

    def test_previews_skipped_with_reason_when_ram_is_short(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch("output_store.available_ram", return_value=RAM_RESERVE + (1 << 20)),
            OutputWorkspace(d, d, {"stack_preview": True, "coverage_png": True}) as work,
        ):
            work.plan(2000, 2000)
            self.assertEqual(work.skipped_previews, ["stack_preview", "coverage_png"])
            self.assertIn("PNG previews need", work.preview_shortage)
            self.assertEqual(work.mode, "disk")

    def test_invalid_stats_name_file_but_valid_flat_photometric_data_allowed(self):
        backend = SimpleNamespace(source_paths={1: "E:/lights/bad.fit"})
        valid = (np.ones(3), np.zeros(3), 0.0)
        g.validate_light_statistics(backend, 1, valid)
        for options in ({"spread": True}, {"noise": True}):
            with self.assertRaisesRegex(ValueError, "bad.fit"):
                g.validate_light_statistics(backend, 1, valid, **options)
        for bad in (([np.nan] * 3, [1] * 3, 1), ([1] * 2, [1] * 3, 1), ([1] * 3, [-1] * 3, 1)):
            with self.assertRaisesRegex(ValueError, "bad.fit"):
                g.validate_light_statistics(backend, 1, bad)

    def test_unusable_actual_light_stops_with_path_instead_of_nan_result(self):
        backend = g.CPUBackend([np.full((48, 64), 1000, np.uint16)], 0, 1)
        backend.source_paths = {0: "E:/lights/flat.fit"}
        cache = {"fwhm": [3.0], "ref_idx": 0, "mats": [np.eye(3).tolist()], "ratios": [[1, 1, 1]], "sizes": [[64, 48]]}
        with self.assertRaisesRegex(ValueError, "flat.fit"):
            g.stack(backend, 32, 1, analysis_cache=cache, norm="spread", log=lambda _: None)
        # Photometric normalization needs no spread estimate for this perfectly flat input.
        image, _, _ = g.stack(backend, 32, 1, analysis_cache=cache, norm="photometric", log=lambda _: None)
        self.assertTrue(np.isfinite(image).any())


if __name__ == "__main__":
    unittest.main()
