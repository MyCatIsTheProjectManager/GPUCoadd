"""Protocol validation without opening the window or processing real lights."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker.bridge import (
    Cancelled,
    Reporter,
    fingerprint,
    write_json,
    image_identity,
    quality_settings,
    output_settings,
    save_image_outputs,
    image_output_manifest,
    stack_pixels,
    nearest_light,
    observation_header,
    prune_work,
    work_root,
    weight_report,
    group_report,
    light_weight,
    header_metadata,
    light_folders,
)


class BridgeTests(unittest.TestCase):
    def test_seestar_sub_copies_are_never_read(self):
        self.assertEqual(light_folders({"lights_dirs": ["E:/Seestar S50/IC 63/lights"]}), ["E:/Seestar S50/IC 63/lights"])
        for folder in ("E:/Seestar S50/NGC 7023/NGC 7023_sub", "E:/Seestar S50/M 42/M 42_mosaic_SUB/"):
            with self.subTest(folder=folder), self.assertRaisesRegex(ValueError, "_sub"):
                light_folders({"lights_dirs": ["E:/Seestar S50/IC 63/lights", folder]})

    def test_cached_header_metadata_matches_reads_without_more_io(self):
        import numpy as np
        from astropy.io import fits
        from datetime import datetime, timezone
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            paths, headers = [], []
            for i in range(4):
                p = Path(d) / f"light{i}.fits"
                h = fits.Header(
                    dict(
                        OBJECT="test",
                        EXPTIME=str(10 + 20 * i),
                        DATE_OBS="unused",
                        FOCALLEN=250.0,
                        XPIXSZ=2.9,
                        YPIXSZ=2.9,
                        XBINNING=1,
                        YBINNING=1,
                        FILTER="LP",
                        TELESCOP="S50" if i < 2 else "S50 Pro",
                        INSTRUME="camera",
                        SITELAT=47.0,
                        SITELONG=19.0,
                        SITEELEV=100,
                    )
                )
                h["DATE-OBS"] = f"2026-10-03T20:0{i}:00"
                h["HISTORY"] = "Large history is deliberately not cached"
                fits.writeto(p, np.zeros((4, 4), np.uint16), h)
                paths.append(p)
                headers.append(header_metadata(fits.getheader(p)))
            self.assertNotIn("HISTORY", headers[0])
            now = datetime(2026, 10, 26, tzinfo=timezone.utc)
            indices, pointing = [1, 3], [[10.0, 20.0]] * 4
            identity = image_identity(paths, indices, now)
            expected = fits.Header()
            observation_header(expected, paths, indices, 0, 3, pointing)
            with patch("astropy.io.fits.getheader", side_effect=AssertionError("Header read after scan")):
                actual = fits.Header()
                self.assertEqual(image_identity(paths, indices, now, headers=headers), identity)
                observation_header(actual, paths, indices, 0, 3, pointing, headers=headers)
            self.assertEqual(actual.tostring(), expected.tostring())

    def test_observation_end_accepts_string_exposure(self):
        import numpy as np
        from astropy.io import fits

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "light.fits"
            for value, expected in (
                ("30.5", "2026-10-03T20:00:30.500000"),
                ("bad", "2026-10-03T20:00:00"),
                (-5, "2026-10-03T20:00:00"),
                ("nan", "2026-10-03T20:00:00"),
                ("inf", "2026-10-03T20:00:00"),
            ):
                fits.writeto(
                    path,
                    np.zeros((2, 2), np.uint16),
                    fits.Header({"DATE-OBS": "2026-10-03T20:00:00", "EXPTIME": value}),
                    overwrite=True,
                )
                hdr = fits.Header()
                observation_header(hdr, [path], [0], 0, None, [[0, 0]])
                self.assertEqual(hdr["DATE-END"], expected)

    def test_full_rgb_weight_report_and_group_input_mapping(self):
        # Input IDs have gaps; loaded/weighted positions need not be contiguous.
        def group(key, main, indices):
            return dict(
                key=key,
                label=key,
                telescope=key,
                filter="LP",
                exposure_s=30,
                lights=len(indices),
                total_exposure_s=30 * len(indices),
                main=main,
                used=True,
                reason=None,
                pedestal=[0],
                pedestal_source="test",
                indices=indices,
            )

        grouping = {"groups": [group("main", True, [1, 4]), group("other", False, [8, 9])]}
        info = dict(
            retained_indices=[0, 1, 2, 3],
            dropped=[],
            alignment_failed=[],
            contributing_indices=[0, 2],
            weighted_indices=[0, 2],
            weights=[[2.0, 4.0, 8.0], [0.5, 2.0, 8.0]],
            weights_per_channel=True,
        )
        report = json.loads(json.dumps(weight_report(info)))
        self.assertEqual(report["weights"], info["weights"])
        self.assertEqual(report["weighted_indices"], [0, 2])
        self.assertTrue(report["weights_per_channel"])
        rows = group_report(grouping, [1, 4, 8, 9], info, "stack")
        self.assertEqual(rows[0]["mean_weight_rgb_vs_main"], [1.0, 1.0, 1.0])
        self.assertEqual(rows[1]["mean_weight_rgb_vs_main"], [0.25, 0.5, 1.0])
        self.assertEqual(rows[1]["mean_weight_vs_main"], 0.5)
        self.assertEqual(rows[1]["main_light_equivalents"], 0.5)
        self.assertEqual(light_weight(info["weights"][0]), 4.0)
        info.update(weights=[4.0, 2.0], weights_per_channel=False)
        self.assertEqual(weight_report(info)["weights"], [4.0, 2.0])
        self.assertNotIn("mean_weight_rgb_vs_main", group_report(grouping, [1, 4, 8, 9], info, "stack")[1])
        self.assertEqual(weight_report({}), dict(weights=None, weighted_indices=None, weights_per_channel=False))

    def test_optional_image_exports_and_live_only_logging(self):
        import numpy as np
        from astropy.io import fits
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            options = output_settings({key: False for key in output_settings({})})
            img = np.arange(32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
            cov = np.ones((32, 32), np.uint32)
            files = save_image_outputs(folder, img, cov, fits.Header(), "test_stack", 1, options)
            self.assertEqual(set(files), {"stack_fits"})
            self.assertEqual([p.name for p in folder.iterdir()], ["test_stack.fits"])
            reporter = Reporter(folder, folder / "cancel", save_log=False)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                reporter.message("Visible in the live log")
            self.assertIn("Visible in the live log", output.getvalue())
            # steps that appear only for some sets say what they are when they start
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                reporter.progress("neighbour alignment", 0, 5)
                reporter.progress("preparing overlap matching, step 1 of 2: low-resolution lights", 0, 5)
                reporter.progress("preparing overlap matching, step 2 of 2: solving", 0, 12)
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            started = [e["message"] for e in events if e["type"] == "stage_started"]
            self.assertIn("does not lower quality", started[0])
            self.assertIn("numbered steps", started[1])
            self.assertEqual(started[2], "Started preparing overlap matching, step 2 of 2: solving")
            # a kept set reused by the next job reports to that job, and reads released lights back
            from worker.bridge import adopt_kept_lights

            kept = SimpleNamespace(log=None, storage_notice=None, restore_ram=lambda: 3)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(adopt_kept_lights(kept, reporter), 3)
                kept.log("released later in this job")
            self.assertEqual(kept.storage_notice, reporter.storage)
            self.assertIn("3 lights that were read from their files are held in RAM again", output.getvalue())
            self.assertIn("released later in this job", output.getvalue())
            self.assertFalse((folder / "activity.jsonl").exists())
            two = folder / "two_outputs"
            two.mkdir()
            choices = output_settings({"coverage_fits": True})
            two_files = save_image_outputs(two, img, cov, fits.Header(), "target_stack", 1, choices)
            self.assertEqual(set(two_files), {"stack_fits", "coverage_fits"})
            self.assertEqual(len(list(two.iterdir())), 2)
            self.assertEqual(len(image_output_manifest(two_files)), 2)
            self.assertTrue(all(value is False for value in output_settings({}).values()))
            files = save_image_outputs(
                folder, img, cov, fits.Header(), "test_stack", 1, output_settings({key: True for key in options})
            )
            self.assertEqual(set(files), {"stack_fits", "stack_preview", "coverage_fits", "coverage_png"})
            for path in files.values():
                self.assertTrue(Path(path).exists())
            np.testing.assert_array_equal(fits.getdata(files["coverage_fits"]), cov)
            self.assertEqual(Path(files["coverage_fits"]).name, "test_stack_coverage.fits")
            self.assertEqual(fits.getheader(files["coverage_fits"])["PRODTYPE"], "COVERAGE")
            self.assertEqual(fits.getheader(files["coverage_fits"])["BUNIT"], "frames")
            self.assertEqual(fits.getheader(files["stack_fits"])["PRODTYPE"], "STACK")
            # FITS standard row order: the first stored row is the image's bottom row, declared
            for key in ("stack_fits", "coverage_fits"):
                self.assertEqual(fits.getheader(files[key])["ROWORDER"], "BOTTOM-UP")
            np.testing.assert_array_equal(fits.getdata(files["stack_fits"])[0], stack_pixels(img[::-1, :, 0])[0])
            rows = np.arange(32, dtype=np.uint32)[:, None] * np.ones((1, 32), np.uint32)
            flipped = save_image_outputs(folder / "rows", img, rows, fits.Header(), "r", 1, choices)
            np.testing.assert_array_equal(fits.getdata(flipped["coverage_fits"]), rows[::-1])
            shared = fits.Header()
            save_image_outputs(folder / "again", img, cov, shared, "x", 1, options)
            self.assertNotIn("ROWORDER", shared, "the caller's header is not modified")
            manifest = image_output_manifest(files)
            self.assertEqual(manifest[0]["role"], "stack_fits")
            self.assertEqual(
                [i for i in manifest if i["role"] == "coverage_fits"][0]["label"], "Frame count map"
            )
            with self.assertRaises(ValueError):
                output_settings({"coverage_fits": "false"})

    def test_stack_pixels_fast_path_matches_the_general_one_bit_for_bit(self):
        import numpy as np
        from unittest.mock import patch
        from worker.bridge import ADU_SCALE

        def general(img, offset=0.0):  # stack_pixels before its fast path, verbatim
            out = (img + np.float32(offset)) / np.float32(ADU_SCALE)
            finite = np.isfinite(out)
            low, high = int(np.count_nonzero(finite & (out <= 0))), int(np.count_nonzero(finite & (out > 1)))
            np.clip(out, np.float32(1 / ADU_SCALE), 1, out=out, where=finite)
            if not finite.all():
                invalid = ~finite
                out[invalid] = np.nan_to_num(out[invalid], copy=False, nan=0.0)
            return out.astype(np.float32, copy=False), low, high

        def same(img, offset):
            got, want = stack_pixels(img, offset), general(img, offset)
            self.assertEqual(got[1:], want[1:])
            self.assertEqual(got[0].dtype, np.float32)
            np.testing.assert_array_equal(got[0].view(np.uint32), want[0].view(np.uint32))

        rng = np.random.default_rng(81)
        bits = rng.integers(0, 2**32, (64, 97), dtype=np.uint64).astype(np.uint32).view(np.float32)
        low = np.float32(1 / ADU_SCALE)
        edges = np.array([np.nan, -np.nan, np.inf, -np.inf, 0, -0.0, 1e-45, -1e-45, 65535, 131070, 131071,
                          -2063, -2062.5, 128_000, 200_000, np.float32(3.4e38)], np.float32)
        typical = rng.normal(2000, 300, (64, 97)).astype(np.float32)
        holes = typical.copy()
        holes[:, :30] = np.nan
        for offset in (0.0, 206.0, 2063.0):
            for img in (bits, edges.reshape(4, 4), typical, holes, np.full((5, 7), np.nan, np.float32),
                        np.empty((0, 3), np.float32), typical.astype(np.float64), holes[:, ::3]):
                with self.subTest(offset=offset, shape=img.shape, dtype=str(img.dtype)):
                    same(img, offset)
            # values landing exactly on the clip bounds, and just outside them
            at = np.array([low, np.nextafter(low, np.float32(0)), 1, np.nextafter(np.float32(1), np.float32(2))],
                          np.float32) * np.float32(ADU_SCALE) - np.float32(offset)
            for k in range(len(at)):
                same(np.concatenate([typical[0], at[k : k + 1]]), offset)
        # ordinary blocks never build the masks of the general path
        with patch("numpy.isfinite", side_effect=AssertionError("general path")):
            stack_pixels(holes, 206.0)
            stack_pixels(np.full((3, 3), np.nan, np.float32), 206.0)

    def test_stack_pixels_offset_headroom_and_fill(self):
        import numpy as np
        from worker.bridge import ADU_SCALE

        img = np.array([[[0, 65535, 120.5], [-40, 90000, np.nan], [-3000, 300000, 5]]], np.float32)
        out, low, high = stack_pixels(img, offset=2063)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(
            out[0, :2],
            [[2063 / ADU_SCALE, 67598 / ADU_SCALE, 2183.5 / ADU_SCALE], [2023 / ADU_SCALE, 92063 / ADU_SCALE, 0]],
            rtol=1e-6,
        )
        self.assertEqual(out[0, 1, 2], 0)  # uncovered: 0, "no data"
        self.assertGreater(out[0, 2, 0], 0)  # clipped low, but still not "no data"
        self.assertEqual(out[0, 2, 1], 1)
        self.assertEqual((low, high), (1, 1))  # only the two extreme samples clip
        self.assertEqual(stack_pixels(img[:, :2], offset=2063)[1:], (0, 0))
        with tempfile.TemporaryDirectory() as d:
            from astropy.io import fits

            options = output_settings({})
            clipping = {}
            files = save_image_outputs(
                Path(d),
                img[:, :2],
                np.ones((1, 2), np.uint32),
                fits.Header(),
                "x",
                1,
                options,
                offset=2063,
                clipping=clipping,
            )
            data, hdr = fits.getdata(files["stack_fits"], memmap=False), fits.getheader(files["stack_fits"])
            self.assertTrue(np.isfinite(data).all() and data.min() >= 0 and data.max() <= 1)
            self.assertEqual((hdr["ADUSCALE"], hdr["ADUOFFS"]), (ADU_SCALE, 2063))
            np.testing.assert_allclose(data[0, 0, 0] * hdr["ADUSCALE"] - hdr["ADUOFFS"], 0, atol=0.01)  # reversible
            self.assertEqual(clipping, {"low": 0, "high": 0})

    def test_nearest_light_and_observation_header(self):
        import numpy as np
        from astropy.io import fits

        shift = lambda dx, dy: [[1, 0, -dx], [0, 1, -dy], [0, 0, 1]]  # dst (reference) -> src (light)
        mats = [shift(0, 0), shift(900, 0), None, shift(1800, 0)]
        sizes, scales = [(1000, 600)] * 4, [2.25] * 4
        self.assertEqual(nearest_light(mats, [0, 1, 3], sizes, scales, (500, 300)), 0)
        self.assertEqual(nearest_light(mats, [0, 1, 3], sizes, scales, (1350, 300)), 1)
        self.assertEqual(nearest_light(mats, [0, 3], sizes, scales, (1350, 300)), 0)  # 0 and 3 tie: first wins
        self.assertIsNone(nearest_light(mats, [2], sizes, scales, (500, 300)))
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for k, (date, tel) in enumerate(
                [
                    ("2026-10-03T20:05:57.2", "S50 Pro"),
                    ("2026-10-03T20:01:00", "S50 Pro"),
                    ("2026-10-03T21:00:00", "S50"),
                ]
            ):
                h = fits.Header()
                h["DATE-OBS"], h["EXPTIME"], h["FILTER"], h["TELESCOP"] = date, 30.0, "LP", tel
                h["FOCALLEN"], h["XPIXSZ"], h["YPIXSZ"], h["BAYERPAT"] = 260.0 + k, 2.9, 2.9, "GRBG"
                paths.append(Path(d) / f"l{k}.fits")
                fits.writeto(paths[-1], np.zeros((2, 2), np.uint16), h)
            pointing = [[14.76, 60.91], [15.2, 61.0], [float("nan"), float("nan")]]
            hdr = fits.Header()
            observation_header(hdr, paths, [0, 1, 2], 0, 1, pointing)
            self.assertEqual((hdr["RA"], hdr["DEC"]), (15.2, 61.0))  # the centre light's pointing
            self.assertEqual(hdr["FOCALLEN"], 260.0)  # the reference light's optics
            self.assertEqual(hdr["DATE-OBS"], "2026-10-03T20:01:00")
            self.assertEqual(hdr["DATE-END"], "2026-10-03T21:00:30")
            self.assertEqual(hdr["FILTER"], "LP")
            self.assertEqual(hdr["TELESCOP"], "S50 Pro")  # lights differ: the reference light's, marked
            self.assertIn("differ", hdr.comments["TELESCOP"])
            self.assertNotIn("BAYERPAT", hdr)  # the stack is RGB
            hdr2 = fits.Header()
            observation_header(hdr2, paths, [0, 1], 0, 2, pointing)  # no usable pointing: no RA/DEC
            self.assertNotIn("RA", hdr2)
            self.assertEqual(hdr2.comments["TELESCOP"], "")

    def test_work_area_and_pruning(self):
        import os
        import time

        self.assertEqual(work_root({"work_dir": "C:/x/y"}), Path("C:/x/y").resolve())
        self.assertEqual(work_root({}).name, "GPUCoadd")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = time.time() - 40 * 86400
            for name, age in (("new_analyze", None), ("old_analyze", old), ("old_in_use", old)):
                (root / "runs" / name).mkdir(parents=True)
                (root / "runs" / name / "analysis.json").write_text("{}")
                if age:
                    os.utime(root / "runs" / name, (age, age))
            (root / "requests").mkdir()
            (root / "requests" / "request_1.json").write_text("{}")
            (root / "requests" / "request_2.json").write_text("{}")
            os.utime(root / "requests" / "request_1.json", (old, old))
            (root / "sessions" / "worker_1").mkdir(parents=True)
            os.utime(root / "sessions" / "worker_1", (old, old))
            prune_work(root, keep=[str(root / "runs" / "old_in_use" / "analysis.json"), None])
            self.assertEqual(sorted(p.name for p in (root / "runs").iterdir()), ["new_analyze", "old_in_use"])
            self.assertEqual([p.name for p in (root / "requests").iterdir()], ["request_2.json"])
            self.assertEqual(list((root / "sessions").iterdir()), [])

    def test_quality_defaults_and_allowed_choices(self):
        defaults = quality_settings({})
        self.assertEqual(
            defaults,
            {
                "debayer": "rcd",
                "interp": "bicubic",
                "norm": "spread",
                "weight": "none",
                "transform": "similarity",
                "rejection": "sigma",
                "select": 0.0,  # leaving out blurry lights is opt-in
                "feather": 0.0,
                "rgb_equal": False,
                "match_gradients": False,
            },
        )
        alternative = {
            "debayer": "bilinear",
            "interp": "bilinear",
            "norm": "photometric",
            "weight": "noise",
            "transform": "homography",
            "rejection": "winsorized",
            "select": 0,
            "feather": 200.0,
            "rgb_equal": False,
            "match_gradients": False,
        }
        self.assertEqual(quality_settings(alternative), alternative)
        self.assertEqual(quality_settings({"norm": "overlap"})["norm"], "overlap")
        self.assertTrue(quality_settings({"rgb_equal": True})["rgb_equal"])
        for bad in (1, "true", None):
            with self.subTest(rgb_equal=bad), self.assertRaises(ValueError):
                quality_settings({"rgb_equal": bad})
        for norm in ("photometric", "overlap"):
            with self.subTest(norm=norm), self.assertRaises(ValueError):
                quality_settings({"rgb_equal": True, "norm": norm})
        for bad in (-1, 1001, float("nan")):
            with self.subTest(feather=bad), self.assertRaises(ValueError):
                quality_settings({"feather": bad})
        for key in ("debayer", "interp", "norm", "weight", "transform", "rejection"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                quality_settings({key: "unknown"})
        for value in (-1, float("nan")):
            with self.assertRaises(ValueError):
                quality_settings({"select": value})

    def test_target_exposure_and_date_naming(self):
        from astropy.io import fits
        import numpy as np
        from datetime import datetime, timezone

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, seconds in enumerate((30, 30, 60)):
                p = Path(d) / f"light{i}.fit"
                h = fits.Header({"OBJECT": "HIP 106890", "EXPTIME": seconds, "DATE-OBS": "2026-10-03T20:00:00"})
                fits.writeto(p, np.zeros((2, 2), np.uint16), h)
                paths.append(p)
            now = datetime(2026, 10, 22, 12, 45, 0, tzinfo=timezone.utc)
            result = image_identity(paths, [0, 1], now)
            self.assertEqual(result["basename"], "HIP_106890_2x30sec_2026-10-03_2026-10-22_124500_gpucoadd")
            self.assertEqual(result["summed_input_exposure_s"], 60)
            mixed = image_identity(paths, [0, 2], now)
            self.assertIsNone(mixed["exposure_seconds"])
            self.assertIn("HIP_106890_1x60sec+1x30sec_", mixed["basename"])
            self.assertEqual(mixed["summed_input_exposure_s"], 90)
            three = image_identity(paths, [0, 1, 2], now)  # 60 s total each: the longer exposure first
            self.assertIn("_1x60sec+2x30sec_", three["basename"])

    def test_group_report(self):
        """Per-group counts, brightness term and main-light equivalents from the stack info."""
        from worker.bridge import group_report

        groups = [
            {
                "key": "pro",
                "label": "S50 Pro",
                "telescope": "S50 Pro",
                "filter": "IRCUT",
                "exposure_s": 30.0,
                "lights": 4,
                "total_exposure_s": 120.0,
                "main": True,
                "used": True,
                "reason": None,
                "pedestal": [2063.0],
                "pedestal_source": "BIAS header",
                "indices": [0, 1, 2, 3],
                "folder": "E:/t/s50pro/lights",
            },
            {
                "key": "s50",
                "label": "S50",
                "telescope": "S50",
                "filter": "IRCUT",
                "exposure_s": 10.0,
                "lights": 3,
                "total_exposure_s": 30.0,
                "main": False,
                "used": True,
                "reason": None,
                "pedestal": [0.0],
                "pedestal_source": "none found (0)",
                "indices": [5, 6, 7],
                "folder": "E:/t/lights",
            },
            {
                "key": "lp",
                "label": "S50 LP",
                "telescope": "S50",
                "filter": "LP",
                "exposure_s": 10.0,
                "lights": 1,
                "total_exposure_s": 10.0,
                "main": False,
                "used": False,
                "reason": "filter LP differs from the main group's IRCUT",
                "pedestal": [0.0],
                "pedestal_source": "none found (0)",
                "indices": [4],
                "folder": "E:/t/lp",
            },
        ]
        used = [0, 1, 2, 3, 5, 6, 7]  # stack indices 0..6
        info = {
            "retained_indices": [0, 1, 2, 4, 5, 6],
            "dropped": [3],
            "alignment_failed": [],
            "contributing_indices": [0, 1, 2, 4, 5],
            "weighted_indices": [0, 1, 2, 4, 5, 6],
            "weights": [1.5, 1.5, 1.5, 0.15, 0.15, 0.15],
            "normalization": {
                "group_terms": {"S50 IRCUT 10s": {"factor": [0.72, 0.89, 0.85], "sigma": [0.02, 0.01, 0.05]}}
            },
        }
        rows = {r["key"]: r for r in group_report({"groups": groups}, used, info, "stack")}
        self.assertEqual(
            (rows["pro"]["retained"], rows["pro"]["sharpness_rejected"], rows["pro"]["contributing"]), (3, 1, 3)
        )
        self.assertEqual((rows["s50"]["retained"], rows["s50"]["contributing"]), (3, 2))
        self.assertAlmostEqual(rows["s50"]["mean_weight_vs_main"], 0.1)
        self.assertAlmostEqual(rows["s50"]["main_light_equivalents"], 0.3)
        self.assertAlmostEqual(rows["pro"]["main_light_equivalents"], 3.0)
        self.assertEqual(rows["s50"]["brightness_vs_star_prior"], [0.72, 0.89, 0.85])
        self.assertFalse(rows["lp"]["used"])
        self.assertNotIn("retained", rows["lp"])
        from worker.bridge import group_names

        same = [
            {**groups[1], "key": "a", "folder": "E:/night1/lights"},
            {**groups[1], "key": "b", "folder": "E:/night2/lights"},
        ]
        self.assertEqual(
            group_names(same), {"a": "S50 IRCUT 10s (night1/lights)", "b": "S50 IRCUT 10s (night2/lights)"}
        )
        self.assertEqual(group_names(groups)["pro"], "S50 Pro IRCUT 30s")
        heat = group_report(
            {"groups": groups},
            used,
            {**info, "contributing_indices": None, "weights": None, "weighted_indices": None, "normalization": None},
            "analyze",
        )
        self.assertNotIn("contributing", heat[0])
        self.assertNotIn("main_light_equivalents", heat[1])

    def test_signature_and_strict_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "light.fit"
            p.write_bytes(b"fixture")
            a = fingerprint([p], {"select": 2.5}, "")
            self.assertEqual(a, fingerprint([p], {"select": 2.5, "tile": 256}, ""))
            self.assertEqual(a, fingerprint([p], {"select": 2.5, "rgb_equal": True}, ""))
            self.assertEqual(
                a,
                fingerprint(
                    [p],
                    {
                        "select": 2.5,
                        "debayer": "bilinear",
                        "interp": "bilinear",
                        "norm": "photometric",
                        "weight": "noise",
                    },
                    "",
                ),
            )
            self.assertNotEqual(a, fingerprint([p], {"select": 2.5, "transform": "homography"}, ""))
            self.assertNotEqual(a, fingerprint([p], {"select": 0}, ""))
            self.assertNotEqual(a, fingerprint([p], {"select": 2.5}, str(p)))
            p.write_bytes(b"changed fixture")
            self.assertNotEqual(a, fingerprint([p], {"select": 2.5}, ""))
            out = Path(d) / "report.json"
            write_json(out, {"invalid": float("nan"), "valid": 3})
            self.assertEqual(json.loads(out.read_text()), {"invalid": None, "valid": 3})

    def test_cooperative_cancellation(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            cancel = folder / "request.cancel"
            reporter = Reporter(folder, cancel)
            try:
                cancel.write_text("cancel")
                with self.assertRaises(Cancelled):
                    reporter.progress("loading", 1, 10)
            finally:
                reporter.log_file.close()


if __name__ == "__main__":
    unittest.main()
