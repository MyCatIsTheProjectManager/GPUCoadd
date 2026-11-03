"""Bounded Overlap matching keeps masks, RNG draws, pair order and scientific results."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import overlap as o
from sky_store import SkyWorkspace


def scene(device):
    rng = np.random.default_rng(15)
    n, h, w = 8, 39, 43
    yy, xx = np.mgrid[:h, :w]
    sky = 200 + 20 * np.sin(xx / 5) + 40 * np.exp(-((xx - 20) ** 2 + (yy - 16) ** 2) / 90)
    lights = np.stack(
        [
            (sky[None] * np.array([1, 0.7, 1.1])[:, None, None] * (1 + i * 0.03) + rng.normal(0, 0.2, (3, h, w)))
            for i in range(n)
        ]
    ).astype(np.float32)
    lights[1, :, :, :3] = np.nan
    lights[3, :, 20:, 35:] = np.nan
    L = torch.from_numpy(lights).to(device)
    meta = dict(
        frames=list(range(n)),
        ref=0,
        prior={i: np.full(3, 1 / (1 + i * 0.03)) for i in range(n)},
        init_offset={i: np.zeros(3) for i in range(n)},
        spread_g={i: (1 / (1 + i * 0.03), 0) for i in range(n)},
        edge=0,
    )
    return L, meta


class SkyStoreTests(unittest.TestCase):
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

    def test_automatic_plan_and_forced_budget_failures(self):
        from raw_storage import GIB

        with (
            tempfile.TemporaryDirectory() as d,
            patch("sky_store.available_ram", return_value=8 * GIB),
            patch("sky_store.check_space", return_value=[]),
        ):
            small = SkyWorkspace((10, 3, 32, 32), "cuda", 10 * GIB, d)
            self.assertEqual(small.mode, "gpu")
            large = SkyWorkspace((5194, 3, 554, 380), "cuda", 10 * GIB, d)
            self.assertEqual(large.mode, "disk")
            with self.assertRaisesRegex(MemoryError, "GPU storage"):
                SkyWorkspace((5194, 3, 554, 380), "cuda", 10 * GIB, d, mode="gpu")
            with self.assertRaisesRegex(MemoryError, "RAM storage"):
                SkyWorkspace((5194, 3, 554, 380), "cuda", 10 * GIB, d, mode="ram")

    def test_roundtrip_reshape_gather_and_cleanup(self):
        data = np.arange(4 * 3 * 17 * 19, dtype=np.float32).reshape(4, 3, 17, 19)
        data[0, 0, 0, 0] = np.nan
        for mode in ("ram", "disk"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                with SkyWorkspace(data.shape, "cpu", 0, d, mode=mode) as work:
                    L = work.create("lights", data.shape, np.float32)
                    L[:] = torch.from_numpy(data)
                    np.testing.assert_array_equal(L[:].numpy(), data)
                    view = L.reshape(4, 3, -1)
                    i = torch.tensor([0, 2])[:, None, None]
                    c = torch.arange(3)[None, :, None]
                    idx = torch.tensor([[0, 20], [3, 12]])[:, None, :]
                    np.testing.assert_array_equal(
                        view[i, c, idx].numpy(), torch.from_numpy(data).reshape(4, 3, -1)[i, c, idx].numpy()
                    )
                    directory = work.directory
                if directory:
                    self.assertFalse(directory.exists())
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    L[:]

    def test_count_matrix_exact_with_multiple_pixel_and_frame_blocks(self):
        mask = torch.from_numpy(np.random.default_rng(12).random((270, 600)) > 0.2)
        expected = o.overlap_counts(mask, block=257)
        with tempfile.TemporaryDirectory() as d, SkyWorkspace((270, 3, 20, 30), "cpu", 0, d, mode="disk") as work:
            stored = work.create("mask", mask.shape, np.bool_)
            stored[:] = mask
            np.testing.assert_array_equal(o.overlap_counts(stored, block=257), expected)

    def compare(self, device):
        L, meta = scene(device)
        valid = torch.isfinite(L).all(1)
        ag = torch.tensor([meta["spread_g"][i][0] for i in meta["frames"]], device=device)
        bg = torch.zeros(8, device=device)
        expected_mask = o.star_mask(L, valid, ag, bg)
        flat = (valid & ~expected_mask).reshape(8, -1)
        pairs = o.select_pairs(o.overlap_counts(flat))
        expected_idx = o.sample_pixels(flat, pairs)
        expected = o.solve(L, **meta)
        for mode in ("ram", "disk"):
            with self.subTest(device=device, mode=mode), tempfile.TemporaryDirectory() as d:
                with SkyWorkspace(L.shape, device, 0, d, mode=mode) as work:
                    stored = work.create("lights", L.shape, np.float32)
                    stored[:] = L
                    masks = work.create("valid", valid.shape, np.bool_)
                    masks[:] = valid
                    actual_mask = o.star_mask(stored, masks, ag, bg)
                    np.testing.assert_array_equal(actual_mask.cpu().numpy(), expected_mask.cpu().numpy())
                    with patch.object(work, "consensus_rows", return_value=1):
                        tiny = o.star_mask(stored, masks, ag, bg)
                    np.testing.assert_array_equal(tiny.cpu().numpy(), expected_mask.cpu().numpy())
                    usable = work.create("usable", valid.shape, np.bool_)
                    usable[:] = valid & ~actual_mask
                    actual_idx = o.sample_pixels(usable.reshape(8, -1), pairs)
                    np.testing.assert_array_equal(actual_idx[:].cpu().numpy(), expected_idx.cpu().numpy())
                    # solve allocates its own maps, so use a separate workspace for end-to-end.
                with SkyWorkspace(L.shape, device, 0, d, mode=mode) as work:
                    stored = work.create("lights", L.shape, np.float32)
                    stored[:] = L
                    actual = o.solve(stored, **meta)
                    for key in (
                        "a",
                        "b",
                        "noise",
                        "weight",
                        "disconnected",
                        "scale_from_prior",
                        "max_correction",
                        "iterations",
                        "pairs",
                        "group_terms",
                        "star_mask_fraction",
                    ):
                        self.assertEqual(actual.get(key), expected.get(key), key)
                    np.testing.assert_allclose(
                        np.array(list(actual["scale_sigma"].values()), float),
                        np.array(list(expected["scale_sigma"].values()), float),
                        rtol=1e-12,
                        atol=1e-14,
                        equal_nan=True,
                    )

    def test_cpu_scientific_equivalence(self):
        self.compare("cpu")

    def test_gpu_scientific_equivalence(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        self.compare("cuda")

    def test_cancel_and_disk_failure_clean_owned_files(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                with SkyWorkspace((3, 3, 20, 30), "cpu", 0, d, mode="disk") as work:
                    L = work.create("lights", work.shape, np.float32)
                    directory = work.directory
                    work.cancel = lambda: (_ for _ in ()).throw(RuntimeError("cancelled"))
                    L[:]
            self.assertFalse(directory.exists())
            with patch("sky_store.check_space", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    SkyWorkspace((3, 3, 20, 30), "cpu", 0, d, mode="disk")

    def test_disk_deferred_solve_and_rejected_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.test_overlap_engine import OverlapEngineTests

        # Exercise the real calibration/low-res builder, cache path and optional reference.
        case = OverlapEngineTests()
        with patch.dict("os.environ", {"GPUCOADD_SKY_STORAGE": "disk"}):
            case.test_deferred_solve_equals_heat_map_solve()
            case.test_reference_rejected_for_sharpness()

    def test_built_normalization_and_final_stack_pixels_match_all_modes(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import gpucoadd as g
        from output_store import OutputWorkspace
        from tests.test_overlap_engine import scene

        raws, mats, a_true, _ = scene(seed=8)
        n = len(raws)
        cache = {
            "fwhm": [3.0] * n,
            "ref_idx": 0,
            "mats": [m.tolist() for m in mats],
            "ratios": [np.full(3, 1 / a).tolist() for a in a_true],
        }
        expected = None
        with tempfile.TemporaryDirectory() as d:
            for mode in ("gpu", "ram", "disk"):
                with self.subTest(mode=mode), patch.dict("os.environ", {"GPUCOADD_SKY_STORAGE": mode}):
                    be = g.GPUBackend(iter(raws), 100, n, "bicubic", "rcd", storage="ram")
                    captured = {}
                    with OutputWorkspace(d, Path(d) / "outputs", {}) as work:
                        g.stack(
                            be,
                            128,
                            1,
                            analysis_cache=cache,
                            coverage_only=True,
                            prepare_overlap=True,
                            on_analysis=captured.update,
                            framing="union",
                            workspace=work,
                            log=lambda _: None,
                        )
                    diag = {}
                    image, _, info = g.stack(
                        be,
                        128,
                        1,
                        analysis_cache=captured,
                        norm="overlap",
                        framing="union",
                        diagnostics=diag,
                        log=lambda _: None,
                    )
                    actual = (
                        image,
                        info["coverage"],
                        diag["kept"],
                        captured["normalization"]["a"],
                        captured["normalization"]["b"],
                        info["contributing_indices"],
                    )
                    if expected is None:
                        expected = actual
                    else:
                        for a, b in zip(actual[:3], expected[:3]):
                            np.testing.assert_array_equal(a, b)
                        self.assertEqual(actual[3:], expected[3:])


if __name__ == "__main__":
    unittest.main()
