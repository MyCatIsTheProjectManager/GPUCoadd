"""The optional aligned lights, through the worker with the real engine on small synthetic lights:
rejecting and averaging the saved files gives the stack, in every option that changes a light's
pixels, and the stack itself doesn't change when the option is on."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpucoadd import sigma_clip_sorted_np  # noqa: E402
from tests.test_combinations import write_scene  # noqa: E402
from tests.test_output_folders import folder_state  # noqa: E402
from worker.bridge import ADU_SCALE, light_copy_name, plan_light_copies, run_request  # noqa: E402

SETTINGS = {
    "debayer": "rcd",
    "interp": "bicubic",
    "norm": "spread",
    "weight": "none",
    "transform": "similarity",
    "rejection": "sigma",
    "select": 0.0,
    "feather": 0.0,
    "rgb_equal": True,
    "match_gradients": False,
    "threads": 2,
    "tile": 0,
}


def stack_adu(path):
    with fits.open(path) as hdus:
        data, header = np.array(hdus[0].data, np.float64), hdus[0].header.copy()
    return data * header["ADUSCALE"] - header["ADUOFFS"], header


class AlignedLightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        for kind in ("single", "mosaic"):
            write_scene(cls.root / kind / "lights", kind)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        for target in (
            "raw_storage.available_ram",
            "output_store.available_ram",
            "resource_guard.available_ram",
            "sky_store.available_ram",
        ):
            p = patch(target, return_value=64 << 30)
            p.start()
            self.addCleanup(p.stop)

    def run_job(self, name, request):
        path = self.root / "requests" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(request))
        with redirect_stdout(io.StringIO()):
            run_request(path)
        result = json.loads(path.with_suffix(".result.json").read_text())
        self.assertEqual(result["status"], "complete", result.get("error"))
        return result

    def stack_job(self, name, kind, aligned, settings=None, crop=False, layout="auto", debayered=False):
        base = {
            "lights_dirs": [str(self.root / kind / "lights")],
            "exclude_groups": [],
            "work_dir": str(self.root / "work"),
            "reference": "",
            "settings": {**SETTINGS, **(settings or {})},
            "output_dir": str(self.root / "out" / name),
        }
        canvas = "disk" if layout == "disk" else "auto"  # the canvas is never on the GPU
        env = {"GPUCOADD_STORAGE": layout, "GPUCOADD_SKY_STORAGE": layout, "GPUCOADD_CANVAS": canvas}
        with patch.dict(os.environ, env):
            heat = self.run_job(f"{name}_analyze", {**base, "operation": "analyze"})
            w, h = heat["canvas"]["width"], heat["canvas"]["height"]
            region = [w // 5, h // 5, w // 2, h // 2] if crop else [0, 0, w, h]
            request = {**base, "operation": "stack", "analysis_file": heat["analysis_file"], "region": region}
            outputs = {"aligned_lights": aligned, "debayered_lights": debayered}
            return self.run_job(f"{name}_stack", {**request, "outputs": outputs})

    def assert_rebuilds_stack(self, result):
        """Sigma clipping the saved lights, which hold the stack's samples, gives the stack."""
        stack, header = stack_adu(result["stack_file"])
        names = sorted(Path(result["aligned_lights"]["folder"]).iterdir())
        self.assertEqual(len(names), result["contributing"])
        self.assertEqual(result["aligned_lights"]["saved"], result["contributing"])
        samples = []
        for path in names:
            light, sub = stack_adu(path)
            self.assertEqual(light.shape, stack.shape)
            self.assertEqual((sub["ORIGINX"], sub["ORIGINY"]), (header["ORIGINX"], header["ORIGINY"]))
            self.assertEqual(sub["ADUOFFS"], header["ADUOFFS"])
            self.assertEqual(sub["NCOMBINE"], 1)
            self.assertEqual(sub["PRODTYPE"], "ALIGNED")
            self.assertEqual(sub["ROWORDER"], "BOTTOM-UP")
            light[light < -sub["ADUOFFS"] + 0.5] = np.nan  # 0 in the file: no data
            samples.append(light)
        samples = np.stack(samples)
        for channel in range(3):
            v = samples[:, channel].reshape(len(samples), -1).T
            has = np.isfinite(v).any(1)
            rebuilt = sigma_clip_sorted_np(np.sort(v[has], axis=1), 3.0, 3)
            expected = stack[channel].reshape(-1)[has]
            self.assertLess(np.abs(rebuilt - expected).max(), 0.01)
        covered = (stack > -header["ADUOFFS"] + 0.5).all(0)
        self.assertEqual(int(np.isfinite(samples).any(0).all(0).sum()), int(covered.sum()))

    def test_saved_lights_rebuild_the_stack(self):
        cases = {
            "spread": ("single", {}, False),
            "crop": ("single", {}, True),
            "mosaic": ("mosaic", {"norm": "overlap", "rgb_equal": False}, False),
            "bilinear": ("single", {"debayer": "bilinear", "interp": "bilinear"}, False),
            "gradients": ("mosaic", {"norm": "overlap", "rgb_equal": False, "match_gradients": True}, True),
        }
        for name, (kind, settings, crop) in cases.items():
            with self.subTest(name):
                self.assert_rebuilds_stack(self.stack_job(name, kind, True, settings, crop))

    def test_each_storage_layout_gives_the_same_files(self):
        hashes = []
        for layout in ("gpu", "ram", "disk"):
            result = self.stack_job(f"layout_{layout}", "single", True, layout=layout)
            folder = Path(result["aligned_lights"]["folder"])
            state = folder_state(folder)
            names = sorted(state)
            pixels = [np.asarray(fits.getdata(folder / n)) for n in names]
            hashes.append((names, pixels))
        for names, pixels in hashes[1:]:
            self.assertEqual(names, hashes[0][0])
            for a, b in zip(pixels, hashes[0][1]):
                self.assertTrue(np.array_equal(a, b))

    def test_the_stack_is_the_same_with_and_without(self):
        off = self.stack_job("without", "single", False)
        on = self.stack_job("with", "single", True)
        self.assertNotIn("aligned_lights", off)
        self.assertNotIn("aligned_lights", off["files"])
        self.assertEqual(off["output_options"]["aligned_lights"], False)
        self.assertEqual(sorted(p.name for p in Path(off["stack_file"]).parent.iterdir()), [Path(off["stack_file"]).name])
        self.assertTrue(np.array_equal(fits.getdata(off["stack_file"]), fits.getdata(on["stack_file"])))
        self.assertEqual(Path(on["files"]["aligned_lights"]).parent, Path(on["stack_file"]).parent)

    def test_lights_only_gain_nothing(self):
        before = folder_state(self.root / "single")
        self.stack_job("clean", "single", True)
        self.assertEqual(folder_state(self.root / "single"), before)

    def test_light_keeps_its_own_details(self):
        result = self.stack_job("details", "single", True)
        light = sorted(Path(result["aligned_lights"]["folder"]).iterdir())[3]
        header = fits.getheader(light)
        self.assertEqual(header["SRCLIGHT"], "light_03.fit")
        self.assertEqual(header["DATE-OBS"], "2026-10-20T21:03:00")
        self.assertEqual(header["EXPTIME"], 30.0)
        self.assertEqual(header["FILTER"], "IRCUT")
        self.assertEqual(header["OBJECT"], "Test single")

    def test_no_room_only_skips_the_extra(self):
        real = shutil.disk_usage

        class Disk:
            free = 1 << 22

        with patch("output_store.shutil.disk_usage", side_effect=lambda p: Disk if "tiny" in str(p) else real(p)):
            messages = []

            class Reporter:
                warn = staticmethod(lambda key, exc, what: messages.append((key, what)))
                message = staticmethod(messages.append)

            tiny = self.root / "tiny"
            tiny.mkdir(exist_ok=True)
            self.assertFalse(plan_light_copies(tiny, "aligned_lights", [4000 * 4000 * 12] * 300, 0, Reporter))
            self.assertEqual(messages, [("aligned_lights", "The aligned lights export")])

    def test_write_failure_keeps_the_stack_and_warns(self):
        import fits_output

        real = fits_output.write_encoded

        def fail_on_the_third_light(path, *args):
            if "_aligned" in str(path) and "light_02" in str(path):
                raise OSError("disk full")
            return real(path, *args)

        with patch("fits_output.write_encoded", side_effect=fail_on_the_third_light):
            result = self.stack_job("failure", "single", True)
        self.assertEqual(result["aligned_lights"]["saved"], 2)
        self.assertEqual(len(list(Path(result["aligned_lights"]["folder"]).iterdir())), 2)
        self.assertEqual([w["output"] for w in result["warnings"]], ["aligned_lights"])
        self.assertTrue(Path(result["stack_file"]).is_file())
        self.assertEqual(result["files"]["aligned_lights"], result["aligned_lights"]["folder"])


    def test_cancelling_keeps_the_stack_and_finished_lights(self):
        import fits_output

        real, request = fits_output.write_encoded, self.root / "requests" / "cancelled_stack.json"
        written = []

        def cancel_after_two_lights(path, *args):
            result = real(path, *args)
            if "_aligned" in str(path):
                written.append(path)
                if len(written) == 2:
                    request.with_suffix(".cancel").touch()
            return result

        with patch("fits_output.write_encoded", side_effect=cancel_after_two_lights), self.assertRaises(AssertionError):
            self.stack_job("cancelled", "single", True)
        result = json.loads(request.with_suffix(".result.json").read_text())
        self.assertEqual(result["status"], "cancelled")
        out = self.root / "out" / "cancelled"
        self.assertEqual(len(list(out.glob("*.fits"))), 1)
        folder = next(p for p in out.iterdir() if p.is_dir())
        self.assertEqual(sorted(folder.iterdir()), sorted(written))


    def debayered(self, result):
        return sorted(Path(result["debayered_lights"]["folder"]).iterdir())

    def test_debayered_lights_keep_the_sensor_values(self):
        """RCD keeps each colour at its own photosites, so there these files must be the light."""
        for debayer in ("rcd", "bilinear"):
            with self.subTest(debayer):
                result = self.stack_job(f"debayered_{debayer}", "single", False, {"debayer": debayer}, debayered=True)
                self.assertNotIn("aligned_lights", result)
                files = self.debayered(result)
                self.assertEqual(len(files), result["contributing"])
                self.assertEqual(result["debayered_lights"]["saved"], result["contributing"])
                for path in files[:3]:
                    light, sub = stack_adu(path)
                    self.assertEqual(sub["PRODTYPE"], "DEBAYERED")
                    self.assertEqual(light.shape, (3, 480, 640))
                    raw = fits.getdata(self.root / "single" / "lights" / sub["SRCLIGHT"]).astype(np.float64)
                    light = light[:, ::-1]  # stored bottom row first
                    edge = 16
                    # GRBG: green at (even, even) and (odd, odd), red at (even, odd), blue at (odd, even)
                    r = light[0, ::2, 1::2], raw[::2, 1::2]
                    g = light[1, ::2, ::2], raw[::2, ::2]
                    b = light[2, 1::2, ::2], raw[1::2, ::2]
                    if debayer == "rcd":
                        for shown, sensor in (r, g, b):
                            core = shown[edge // 2 : -edge // 2, edge // 2 : -edge // 2]
                            expected = sensor[edge // 2 : -edge // 2, edge // 2 : -edge // 2] - sub["ADUOFFS"]
                            self.assertLess(np.abs(core - expected).max(), 0.02)
                    else:
                        self.assertLess(abs(np.median(light[1]) - np.median(raw - sub["ADUOFFS"])), 200)

    def test_reference_light_is_the_same_in_both_exports(self):
        """The reference doesn't move and, without RGB equalization, keeps its brightness."""
        result = self.stack_job("both", "single", True, {"rgb_equal": False}, debayered=True)
        ref = Path(result["reference"]).name
        pick = lambda files: next(p for p in files if fits.getheader(p)["SRCLIGHT"] == ref)
        moved, sub = stack_adu(pick(sorted(Path(result["aligned_lights"]["folder"]).iterdir())))
        still, own = stack_adu(pick(self.debayered(result)))
        moved, still = moved[:, ::-1], still[:, ::-1]  # top row first
        ox, oy = sub["ORIGINX"], sub["ORIGINY"]
        area = moved[:, -oy + 20 : -oy + 460, -ox + 20 : -ox + 620]
        self.assertLess(np.abs(area - still[:, 20:460, 20:620]).max(), 0.02)
        self.assertNotIn("ORIGINX", own)

    def test_debayered_failure_leaves_the_aligned_lights(self):
        import fits_output

        real = fits_output.write_encoded

        def no_debayered(path, *args):
            if "_debayered" in str(path):
                raise OSError("disk full")
            return real(path, *args)

        with patch("fits_output.write_encoded", side_effect=no_debayered):
            result = self.stack_job("half", "single", True, debayered=True)
        self.assertEqual(result["aligned_lights"]["saved"], result["contributing"])
        self.assertEqual(result["debayered_lights"]["saved"], 0)
        self.assertEqual([w["output"] for w in result["warnings"]], ["debayered_lights"])

    def test_the_second_export_counts_the_first_when_checking_space(self):
        messages = []

        class Reporter:
            warn = staticmethod(lambda key, exc, what: messages.append(key))
            message = staticmethod(lambda text: None)

        real = shutil.disk_usage

        class Disk:
            free = 3 << 30

        with patch("output_store.shutil.disk_usage", side_effect=lambda p: Disk if "tiny" in str(p) else real(p)):
            tiny = self.root / "tiny"
            tiny.mkdir(exist_ok=True)
            self.assertTrue(plan_light_copies(tiny, "aligned_lights", [1 << 30], 0, Reporter))
            self.assertFalse(plan_light_copies(tiny, "debayered_lights", [1 << 30], 2 << 30, Reporter))
        self.assertEqual(messages, ["debayered_lights"])


class NameTests(unittest.TestCase):
    def test_lights_with_one_name_get_their_own_files(self):
        used = set()
        self.assertEqual(light_copy_name("a/light_1.fit", used, "aligned"), "light_1_aligned.fits")
        self.assertEqual(light_copy_name("b/LIGHT_1.fit", used, "aligned"), "LIGHT_1_2_aligned.fits")
        self.assertEqual(light_copy_name("c/light_1.fit", used, "aligned"), "light_1_3_aligned.fits")

    def test_scale_constant_matches_the_stack(self):
        self.assertEqual(ADU_SCALE, 2 * 65535.0)


if __name__ == "__main__":
    unittest.main()
