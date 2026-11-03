"""Identical raw calibration, warp/rejection and frame indices with external raws."""

import gc
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpucoadd as g


class StreamedBackendTests(unittest.TestCase):
    def test_oom_retry_keeps_image_coverage_rejection_and_tile_bounds_exact(self):
        import torch

        raws = [np.random.default_rng(i).integers(1000, 5000, (96, 112), np.uint16) for i in range(5)]
        cache = {
            "fwhm": [3] * 5,
            "ref_idx": 0,
            "mats": [np.eye(3).tolist()] * 5,
            "ratios": [[1, 1, 1]] * 5,
            "sizes": [[112, 96]] * 5,
        }
        be = g.GPUBackend(iter(raws), 0, 5, storage="ram")
        expected_diag = {}
        expected, _, before = g.stack(
            be,
            64,
            1,
            analysis_cache=cache,
            select=0,
            region=[8, 8, 80, 64],
            diagnostics=expected_diag,
            log=lambda _: None,
        )
        original = g._render_stack_tile
        calls = []

        def injected(backend, bounds, *args, **kwargs):
            calls.append(bounds)
            if len(calls) == 1:
                raise torch.OutOfMemoryError("injected pressure")
            return original(backend, bounds, *args, **kwargs)

        actual_diag = {}
        with patch.object(g, "_render_stack_tile", side_effect=injected):
            actual, _, after = g.stack(
                be,
                64,
                1,
                analysis_cache=cache,
                select=0,
                region=[8, 8, 80, 64],
                diagnostics=actual_diag,
                log=lambda _: None,
            )
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(after["tile_plan"]["actual_tiles"], before["tile_plan"]["actual_tiles"])
        # tiles are rendered in absolute reference coordinates; the report gives canvas positions
        x0, x1, y0, y1 = calls[0]
        self.assertEqual(after["tile_plan"]["retries"], [[x0 - 8, x1 - 8, y0 - 8, y1 - 8]])
        for left, right in (
            (expected, actual),
            (before["coverage"], after["coverage"]),
            (expected_diag["kept"], actual_diag["kept"]),
        ):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(before["contributing_indices"], after["contributing_indices"])
        del be
        gc.collect()
        torch.cuda.empty_cache()

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

    def test_modes_exact_with_different_sensors_rejection_and_feather(self):
        import torch

        rng = np.random.default_rng(17)
        raws = [rng.integers(30000, 45000, (128, 160), np.uint16) for _ in range(4)]
        raws += [rng.integers(30000, 45000, (96, 120), np.uint16) for _ in range(3)]
        peds = [1000, 2000, 1000, 2000, 500, 500, 500]
        mats = [np.array([[1, 0, i * 0.1], [0, 1, -i * 0.2], [0, 0, 1.0]]) for i in range(7)]
        cache = {
            "fwhm": [3] * 7,
            "ref_idx": 0,
            "mats": [m.tolist() for m in mats],
            "ratios": [[1, 1, 1]] * 7,
            "sizes": [[r.shape[1], r.shape[0]] for r in raws],
        }
        with tempfile.TemporaryDirectory() as d:
            paths = [str(Path(d) / f"{i}.fit") for i in range(7)]
            for p, r in zip(paths, raws):
                fits.writeto(p, r)
            for rejection in ("sigma", "winsorized", "noise"):
                expected = None
                # "released": RAM lights read from their files after other programs needed RAM;
                # "short": RAM ran out while loading, so the last lights stay file-backed
                for mode in ("gpu", "ram", "disk", "released", "short", "restored"):
                    free = iter([64 << 30] * 3 + [1] * 4) if mode == "short" else None
                    with patch(
                        "resource_guard.available_ram", side_effect=(lambda: next(free)) if free else lambda: 64 << 30
                    ):
                        be = g.GPUBackend(
                            iter(raws),
                            peds,
                            7,
                            "bicubic",
                            "rcd",
                            paths=paths,
                            sizes=[r.shape for r in raws],
                            storage=mode if mode in ("gpu", "ram", "disk") else "ram",
                        )
                    if mode == "restored":  # released, then read back into RAM before the next job
                        be.release_ram(1 << 40)
                        self.assertEqual(be.restore_ram(), 7)
                        self.assertEqual(be.released, 0)
                        self.assertTrue(all(isinstance(r, torch.Tensor) for r in be.raw))
                    if mode == "released":
                        self.assertEqual(be.release_ram(0), 0)
                        self.assertGreater(be.release_ram(1 << 40), 0)
                        self.assertEqual(be.release_ram(1 << 40), 0)  # one way: nothing left, nothing refilled
                    if mode in ("released", "short"):
                        self.assertEqual(be.released, 7 if mode == "released" else 4)
                        self.assertEqual(be.storage_info()["lights_read_from_files_after_ram_ran_short"], be.released)
                    be.flat_inv[(128, 160)][:] *= 1.03
                    np.testing.assert_array_equal(
                        be.calibrate(0).cpu().numpy(), (raws[0].astype(np.float32) - 1000) * np.float32(1.03)
                    )
                    diag = {}
                    img, _, info = g.stack(
                        be,
                        64,
                        1,
                        analysis_cache=cache,
                        select=0,
                        framing="union",
                        region=[8, 8, 80, 64],
                        norm="spread",
                        weight="noise",
                        rejection=rejection,
                        feather=20,
                        diagnostics=diag,
                        log=lambda _: None,
                    )
                    actual = (img, info["coverage"], diag["kept"], info["contributing_indices"])
                    if expected is None:
                        expected = actual
                    else:
                        for a, b in zip(expected[:3], actual[:3]):
                            np.testing.assert_array_equal(a, b)
                        self.assertEqual(expected[3], actual[3])
                    del be
                    gc.collect()
                    torch.cuda.empty_cache()

    def test_restore_between_jobs_stops_at_shortage_and_never_touches_gpu_lights(self):
        import torch

        raws = [np.random.default_rng(i).integers(0, 65536, (64, 80), np.uint16) for i in range(6)]
        with tempfile.TemporaryDirectory() as d:
            paths = [str(Path(d) / f"{i}.fit") for i in range(6)]
            for path, r in zip(paths, raws):
                fits.writeto(path, r)
            sizes = [r.shape for r in raws]
            be = g.GPUBackend(iter(raws), 0, 6, "bicubic", "rcd", paths=paths, sizes=sizes, storage="ram")
            be.release_ram(1 << 40)
            self.assertEqual(be.released, 6)
            # RAM for two lights only: two come back, the rest stay file-backed, nothing is released
            free = iter([64 << 30, 64 << 30, 1])
            with patch("resource_guard.available_ram", side_effect=lambda: next(free)):
                self.assertEqual(be.restore_ram(), 2)
            self.assertEqual(be.released, 4)
            for k in range(6):
                np.testing.assert_array_equal(
                    be.raw_crop(k, (0, 64, 0, 80)).cpu().numpy().view(np.uint16), raws[k]
                )
            be.keep_only([0, 5])  # the next job needs fewer lights: the count follows the set
            self.assertEqual(be.released, 1)
            gpu = g.GPUBackend(iter(raws), 0, 6, "bicubic", "rcd", paths=paths, sizes=sizes, storage="gpu")
            self.assertEqual((gpu.release_ram(1 << 40), gpu.restore_ram(), gpu.released), (0, 0, 0))
            self.assertTrue(all(r.is_cuda for r in gpu.raw))
            del be, gpu
            torch.cuda.empty_cache()

    def test_cached_statistics_avoid_full_frame_reads_and_keep_global_indices(self):
        import torch

        raws = [np.random.default_rng(i).integers(0, 65536, (128, 160), np.uint16) for i in range(5)]
        cache = {
            "fwhm": [3] * 5,
            "ref_idx": 3,
            "mats": [np.eye(3).tolist()] * 5,
            "ratios": [[1, 1, 1]] * 5,
            "sizes": [[160, 128]] * 5,
        }
        be = g.GPUBackend(iter(raws), 0, 5, "bicubic", "rcd", storage="ram")
        stats = {
            str(i): [s[0].tolist(), s[1].tolist(), s[2], s[3].tolist()]
            for i in range(5)
            for s in [be.stats(be.calibrate(i))]
        }
        cache["statistics"] = stats
        be.keep_only([1, 3, 4])
        be.transfer_bytes = 0
        # Partial input intentionally excludes contributing frames: mark them failed.
        cache["mats"][0] = cache["mats"][2] = None
        with patch.object(be, "calibrate", side_effect=AssertionError("cached stats must avoid full calibration")):
            img, _, info = g.stack(be, 64, 1, analysis_cache=cache, select=0, region=[0, 0, 64, 64], log=lambda _: None)
        self.assertEqual(info["retained_indices"], [1, 3, 4])
        self.assertEqual(info["ref_idx"], 3)
        self.assertGreater(be.transfer_bytes, 0)
        del be
        gc.collect()
        torch.cuda.empty_cache()

    def test_all_5194_samples_are_integrated_together(self):
        import torch

        n = 5194
        raw = np.full((64, 64), 1000, np.uint16)
        cache = {
            "fwhm": [3] * n,
            "ref_idx": 0,
            "mats": [np.eye(3).tolist()] * n,
            "ratios": [[1, 1, 1]] * n,
            "sizes": [[64, 64]] * n,
            "statistics": {str(i): [[1000] * 3, [0] * 3, 1, [1] * 3] for i in range(n)},
        }
        be = g.GPUBackend(iter([raw] * n), 0, n, "bicubic", "rcd", storage="ram")
        diag = {}
        image, _, info = g.stack(
            be,
            16,
            1,
            analysis_cache=cache,
            select=0,
            region=[20, 20, 16, 16],
            norm="photometric",
            log=lambda _: None,
            diagnostics=diag,
        )
        np.testing.assert_array_equal(image, np.full_like(image, 1000))
        np.testing.assert_array_equal(info["coverage"], np.full((16, 16), n))
        np.testing.assert_array_equal(diag["kept"], np.full((16, 16, 3), n))
        self.assertEqual(len(info["contributing_indices"]), n)
        del be
        gc.collect()
        torch.cuda.empty_cache()

    def test_cached_statistics_do_not_cross_cpu_gpu_median_semantics(self):
        import torch

        raws = [np.random.default_rng(i).integers(100, 1000, (64, 80), np.uint16) for i in range(3)]
        for backend in ("gpu", "cpu"):
            be = g.GPUBackend(iter(raws), 0, 3, storage="ram") if backend == "gpu" else g.CPUBackend(raws, 0, 1)
            cache = {
                "fwhm": [3] * 3,
                "ref_idx": 0,
                "mats": [np.eye(3).tolist()] * 3,
                "ratios": [[1, 1, 1]] * 3,
                "sizes": [[80, 64]] * 3,
                "statistics_backend": "cpu" if backend == "gpu" else "gpu",
                "statistics": {str(i): [[0] * 3, [0] * 3, 1, [1] * 3] for i in range(3)},
            }
            captured = {}
            with patch.object(be, "stats", wraps=be.stats) as measure:
                image, _, _ = g.stack(
                    be,
                    32,
                    1,
                    analysis_cache=cache,
                    region=[12, 12, 24, 24],
                    on_analysis=captured.update,
                    log=lambda _: None,
                )
                self.assertEqual(measure.call_count, 3)
            self.assertEqual(captured["statistics_backend"], backend)
            self.assertTrue(np.isfinite(image).all())
            del be
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
