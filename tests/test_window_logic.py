"""The window's logic (GPUCoadd/gpucoadd_qt/logic.py), checked without Qt."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPUCoadd"))
from gpucoadd_qt import logic  # noqa: E402

CONTRACT = json.loads((ROOT / "tests" / "fixtures" / "dialog_contract.json").read_text(encoding="utf-8"))


def make_tree(root, layout):
    """layout: relative folder -> number of .fit files."""
    for folder, count in layout.items():
        path = Path(root, folder)
        path.mkdir(parents=True, exist_ok=True)
        for k in range(count):
            (path / f"Light_{k}.fit").write_bytes(b"")


class Texts(unittest.TestCase):
    def test_duration_and_exposure(self):
        self.assertEqual(logic.duration(0), "0m 0s")
        self.assertEqual(logic.duration(125.4), "2m 5s")
        self.assertEqual(logic.duration(-3), "0m 0s")
        self.assertEqual(logic.exposure_text(20), "20 s")
        self.assertEqual(logic.exposure_text(600), "10 min")
        self.assertEqual(logic.exposure_text(3 * 3600 + 25 * 60), "3 h 25 min")
        self.assertEqual(logic.seconds_text(4.25), "4.2 sec")
        self.assertEqual(logic.seconds_text(75), "1m 15s")

    def test_paths(self):
        self.assertEqual(logic.norm_path("E:\\Seestar\\IC 63\\"), "E:/Seestar/IC 63")
        self.assertEqual(logic.parent_path("E:/a/b"), "E:/a")
        self.assertEqual(logic.base_name("E:/a/b/"), "b")
        self.assertTrue(logic.is_sub_copy_folder("E:/Seestar/IC 63_sub"))
        self.assertFalse(logic.is_sub_copy_folder("E:/Seestar/IC 63"))

    def test_output_plan(self):
        self.assertEqual(logic.output_plan_text({}), "Next stack saves 1 image file(s): stacked image (FITS).")
        text = logic.output_plan_text({"coverage_fits": True, "stack_preview": True, "coverage_png": True})
        self.assertIn("4 image file(s)", text)

    def test_result_summary_and_images(self):
        result = {
            "canvas": {"width": 2778, "height": 2661},
            "contributing": 149,
            "wall_s": 33,
            "identity": {"target": "HIP_106890"},
            "image_outputs": [
                {"role": "stack_fits", "label": "Stacked image", "format": "FITS", "path": "stack.fits"},
                {"role": "coverage_fits", "label": "Frame count map", "format": "FITS", "path": "stack_coverage.fits"},
            ],
        }
        files = logic.result_images(result)
        text = logic.result_summary(result, files)
        self.assertIn("149 contributing lights", text)
        self.assertIn("Saved 2 image", text)
        self.assertTrue(text.startswith("Last completed stack: HIP_106890 · 2778 × 2661 pixels"))
        old = logic.result_images({"stack_file": "a.fits"})
        self.assertEqual(old, [{"label": "Stacked image", "format": "FITS", "path": "a.fits", "role": "stack_fits"}])

    def test_cache_text(self):
        self.assertEqual(logic.cache_text({}), "Worker ready · no GPU lights held")
        text = logic.cache_text({"resident_lights": 12, "resident_vram_gib": 0.5})
        self.assertIn("12 lights ready · 0.50 GiB", text)
        text = logic.cache_text({"resident_lights": 0, "available_lights": 400, "storage_mode": "ram"})
        self.assertTrue(text.startswith("400 lights ready from RAM"))

    def test_group_summary(self):
        self.assertEqual(logic.group_summary([{"used": True, "label": "a", "retained": 3}]), [])
        lines = logic.group_summary(
            [
                {"used": True, "label": "S50", "contributing": 10},
                {"used": True, "label": "Pro", "retained": 4, "main_light_equivalents": 6.25,
                 "brightness_vs_star_prior": [1.0, 0.98, 1.02]},
            ]
        )
        self.assertEqual(lines[0], "S50: 10 lights.")
        self.assertIn("worth 6.2 main-group lights; brightness matched", lines[1])

    def test_run_summary(self):
        self.assertTrue(
            logic.run_summary_line({"folder": "F", "output_options": {"activity_log": False}}).endswith("disabled.")
        )
        self.assertTrue(
            logic.run_summary_line(
                {"folder": "F", "output_options": {"activity_log": True}, "files": {"activity_log": "x"}}
            ).endswith("activity log.")
        )


class FolderDiscovery(unittest.TestCase):
    def test_target_lights_offered_and_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            t = logic.norm_path(d) + "/IC 63"
            make_tree(t, {"lights": 3, "s50pro/lights": 2, "IC 63_sub": 4, "empty/lights": 0, "notes": 0})
            found = logic.find_light_folders(t)
            self.assertEqual(found, {"target": t, "lights": t + "/lights", "offered": [t + "/s50pro/lights"]})
            # a picked lights folder still works: its parent is the target
            self.assertEqual(logic.find_light_folders(t + "/lights")["target"], t)
            self.assertIn("_sub", logic.find_light_folders(t + "/IC 63_sub")["error"])
            self.assertIn('No "lights" folder', logic.find_light_folders(t + "/notes")["error"])
            self.assertEqual(
                logic.find_light_folders(t + "/empty")["error"],
                f"No FITS lights (.fit, .fits, .fts) in {t}/empty/lights.",
            )
            self.assertEqual(logic.find_light_folders("")["error"], "Choose a target folder.")
            self.assertIsNone(logic.add_folder_problem(t + "/lights"))
            self.assertIn("_sub", logic.add_folder_problem(t + "/IC 63_sub"))

    def test_fits_and_fts_lights_as_the_engine_reads_them(self):
        with tempfile.TemporaryDirectory() as d:
            root = logic.norm_path(d)
            for target, name in (("A", "Light_1.fits"), ("B", "LIGHT_1.FTS"), ("C", "Light_1.Fit")):
                Path(root, target, "lights").mkdir(parents=True)
                Path(root, target, "lights", name).write_bytes(b"")
                found = logic.find_light_folders(f"{root}/{target}")
                self.assertEqual(found.get("lights"), f"{root}/{target}/lights", found)
                self.assertIsNone(logic.add_folder_problem(f"{root}/{target}/lights"))
            # a second folder of .fits lights is offered; one with other files only is not
            Path(root, "A", "s50pro", "lights").mkdir(parents=True)
            Path(root, "A", "s50pro", "lights", "Light_2.fits").write_bytes(b"")
            Path(root, "A", "jpg", "lights").mkdir(parents=True)
            Path(root, "A", "jpg", "lights", "Light_2.jpg").write_bytes(b"")
            Path(root, "A", "jpg", "lights", "Light_2.fits.txt").write_bytes(b"")
            self.assertEqual(logic.find_light_folders(f"{root}/A")["offered"], [f"{root}/A/s50pro/lights"])
            self.assertEqual(
                logic.add_folder_problem(f"{root}/A/jpg/lights"),
                f"No FITS lights (.fit, .fits, .fts) in {root}/A/jpg/lights.",
            )

    def test_light_suffixes_match_the_engine(self):
        sys.path.insert(0, str(ROOT))
        import lightgroups

        self.assertEqual(logic.LIGHT_SUFFIXES, lightgroups.LIGHT_SUFFIXES)

    def test_folder_state(self):
        f = logic.Folders()
        self.assertEqual(f.set([]), "Choose a target folder to begin.")
        hint = f.set(["E:\\T\\lights", "E:/T/lights"], [], ["E:/T/pro/lights"], "E:/T")
        self.assertEqual(f.folders, ["E:/T/lights"])
        self.assertIn("1 more light folder(s)", hint)
        f.toggle("E:/T/pro/lights", True)
        self.assertEqual((f.folders, f.offered), (["E:/T/lights", "E:/T/pro/lights"], []))
        f.exclude("E:/T/pro/lights|S50 Pro|IRCUT", False)
        f.exclude("E:/T/lights|S50|IRCUT", False)
        f.toggle("E:/T/pro/lights", False)
        # a switched-off group is forgotten once its folder is no longer used
        self.assertEqual(f.excluded, ["E:/T/lights|S50|IRCUT"])
        self.assertEqual(f.rows()[1], ("E:/T/pro/lights", False, "E:/T/pro/lights  (found; tick to include)"))
        f.remove("E:/T/pro/lights")
        self.assertEqual(f.offered, [])
        f.exclude("E:/T/lights|S50|IRCUT", True)
        self.assertEqual(f.excluded, [])
        self.assertEqual(f.target, "E:/T")


class Groups(unittest.TestCase):
    GROUPS = [
        {"key": "a", "label": "S50, IRCUT", "lights": 100, "total_exposure_s": 1000, "used": True, "main": True},
        {"key": "b", "label": "S50, LP", "lights": 20, "total_exposure_s": None, "used": False,
         "reason": "different filter than the main group"},
        {"key": "c", "label": "Pro, IRCUT", "lights": 30, "total_exposure_s": 600, "used": False,
         "reason": "left out by the user", "main_light_equivalents": 2.04},
    ]

    def test_rows_and_filter_rule(self):
        rows, hint = logic.group_rows(self.GROUPS)
        self.assertEqual(rows[0]["texts"], ["S50, IRCUT (main)", "100", "17 min",
                                            "Main group: sets the filter and the alignment reference"])
        self.assertFalse(rows[1]["checkable"], "a group with another filter can never be ticked")
        self.assertTrue(rows[2]["checkable"] and not rows[2]["checked"])
        self.assertEqual(rows[1]["texts"][2], "?")
        self.assertIn("worth 2.0 main-group lights", rows[2]["texts"][3])
        self.assertEqual(hint, "100 of 150 lights in 1 group(s) will be stacked.")
        self.assertEqual(logic.group_rows([]), ([], None))
        _, mixed = logic.group_rows([dict(self.GROUPS[0]), dict(self.GROUPS[2], used=True)])
        self.assertIn("Mixed instruments", mixed)


class Quality(unittest.TestCase):
    def test_defaults_round_trip(self):
        controls = logic.quality_controls({})
        self.assertEqual(
            logic.quality_settings(controls["choice"], controls["selection"], controls["sharpness"],
                                   controls["feather"], controls["rgb_equal"], controls["match_gradients"]),
            {"debayer": "rcd", "interp": "bicubic", "norm": "spread", "weight": "none", "transform": "similarity",
             "rejection": "sigma", "select": 0, "feather": 0, "rgb_equal": True, "match_gradients": False},
        )
        self.assertFalse(controls["selection"], "leaving out blurry lights is opt-in")
        saved = {"debayer": "bilinear", "interp": "bilinear", "norm": "overlap", "weight": "noise",
                 "transform": "homography", "rejection": "noise", "select": 0, "feather": 40,
                 "rgb_equal": False, "match_gradients": True}
        c = logic.quality_controls(saved)
        self.assertFalse(c["selection"])
        self.assertEqual(c["sharpness"], 2.5)
        back = logic.quality_settings(c["choice"], c["selection"], c["sharpness"], c["feather"], c["rgb_equal"],
                                      c["match_gradients"])
        self.assertEqual(back, saved)

    def test_option_availability(self):
        one = [{"used": True}]
        state = logic.option_availability(1, one, True, False, False)
        self.assertEqual((state["rgb_equal"], state["rgb_auto_off"], state["rgb_available"]), (False, True, False))
        back = logic.option_availability(0, one, state["rgb_equal"], state["rgb_auto_off"], False)
        self.assertEqual((back["rgb_equal"], back["messages"]), (True, ["RGB equalization turned back on."]))
        # turned off by the user: stays off
        self.assertFalse(logic.option_availability(0, one, False, False, False)["rgb_equal"])
        two = [{"used": True}, {"used": True}]
        self.assertTrue(logic.option_availability(0, two, False, False, True)["gradients_available"])
        off = logic.option_availability(0, one, True, False, True)
        self.assertFalse(off["match_gradients"])
        self.assertIn("Gradient matching turned off: it requires Overlap matching.", off["messages"])


class Contract(unittest.TestCase):
    """The cases in tests/fixtures/dialog_contract.json."""

    def test_requests(self):
        for case in CONTRACT["requests"]:
            i = case["inputs"]
            with self.subTest(case["name"]):
                settings = dict(logic.quality_settings(**logic_quality_args(i["settings"])))
                request = logic.build_request(
                    i["operation"], i["folders"], i["excluded"], i["output_dir"], "<WORK_DIR>", i["reference"],
                    settings, i["outputs"], i["threads"], 0 if i["auto_tile"] else i["tile"],
                    i.get("analysis_file"), i.get("region"),
                )
                self.assertEqual(request, case["request"])
                self.assertEqual(list(request), list(case["request"]), "same field order as the contract")

    def test_stages(self):
        for case in CONTRACT["stages"]:
            with self.subTest(case["name"]):
                stages = logic.Stages()
                stages.reset(case["operation"], True, 0.0)
                for k, (stage, done, total) in enumerate(case["events"]):
                    stages.event(stage, done, total, 1.0 + k)
                if case.get("finish"):
                    stages.finish(case["finish"], 100.0)
                got = [[s["name"], s["state"], s.get("step")] for s in stages.stages]
                self.assertEqual(got, case["expected"])

    def test_storage_notices(self):
        for case in CONTRACT["storage_notices"]:
            with self.subTest(case["name"]):
                notices = logic.StorageNotices()
                text = None
                for notice in case["notices"]:
                    text = notices.add(notice)
                self.assertEqual(text and text[0], case["title"])


def logic_quality_args(settings):
    c = logic.quality_controls(settings)
    return dict(choice=c["choice"], selection=c["selection"], sharpness=c["sharpness"], feather=c["feather"],
                rgb_equal=c["rgb_equal"], match_gradients=c["match_gradients"])


class Framing(unittest.TestCase):
    CANVAS = {"width": 4000, "height": 3000}

    def test_drawing_moving_and_typing(self):
        rect = logic.display_rect(1000, 400, 800, 600)
        self.assertAlmostEqual(rect[3], 388)
        self.assertEqual(logic.sky_point(rect, self.CANVAS, rect[0], rect[1]), (0, 0))
        self.assertEqual(logic.sky_point(rect, self.CANVAS, rect[0] + rect[2] + 9, -50), (4000, 0))
        self.assertTrue(logic.inside(rect, rect[0] + 1, rect[1] + 1))
        self.assertFalse(logic.inside(rect, rect[0] - 1, rect[1] + 1))
        # drawing covers both points whole, in either direction
        self.assertEqual(logic.drag_region((2800.5, 2100), None, self.CANVAS, (800.2, 599.9)), [800, 599, 2001, 1501])
        region = logic.drag_region((800, 600), None, self.CANVAS, (2800, 2100))
        self.assertEqual(region, [800, 600, 2000, 1500])
        self.assertFalse(logic.is_full_frame(region, self.CANVAS, True))
        # moving keeps the size and stays on the canvas at every edge
        moving = logic.drag_start(region, self.CANVAS, (1200, 900))
        self.assertEqual(moving, region)
        self.assertEqual(logic.drag_region((1200, 900), moving, self.CANVAS, (99999, 99999)), [2000, 1500, 2000, 1500])
        self.assertEqual(logic.drag_region((1200, 900), moving, self.CANVAS, (-5, -5)), [0, 0, 2000, 1500])
        self.assertEqual(logic.drag_region((1200, 900), moving, self.CANVAS, (1200.5, 900.5)), [801, 601, 2000, 1500])
        # outside the crop, or with the full frame, a new crop is drawn
        self.assertIsNone(logic.drag_start(region, self.CANVAS, (100, 100)))
        self.assertIsNone(logic.drag_start([0, 0, 4000, 3000], self.CANVAS, (100, 100)))
        self.assertEqual(logic.clamp_region([3999.0, 10, 500, 5000], self.CANVAS), [3999, 10, 1, 2990])
        self.assertTrue(logic.is_full_frame([0, 0, 4000, 3000], self.CANVAS, False))
        self.assertFalse(logic.is_full_frame([0, 0, 4000, 3000], self.CANVAS, True))
        self.assertEqual(logic.stack_caption(True, [0, 0, 4000, 3000]), "Stack Full Frame (4000 × 3000)")
        self.assertEqual(logic.stack_caption(False, None), "Stack Selected Area")

    def test_js_round(self):
        self.assertEqual([logic.js_round(v) for v in (0.5, 1.5, -0.5, 2.4)], [1, 2, 0, 2])


class WorkerMessages(unittest.TestCase):
    def test_split_events(self):
        events, rest = logic.split_events("", '{"type": "log", "message": "a"}\nplain text\r\n\n{"type": "pro')
        self.assertEqual(events, [{"type": "log", "message": "a"}, {"type": "text", "message": "plain text"}])
        events, rest = logic.split_events(rest, 'gress", "stage": "x"}\n')
        self.assertEqual((events, rest), ([{"type": "progress", "stage": "x"}], ""))
        self.assertEqual(logic.split_events("", "[1, 2]\n")[0], [{"type": "text", "message": "[1, 2]"}])

    def test_request_matching_and_files(self):
        req = "C:/Users/x/AppData/Local/GPUCoadd/requests/request_1_1.json"
        self.assertTrue(logic.same_request({"request_file": req.replace("/", "\\").upper()}, req))
        self.assertFalse(logic.same_request({"request_file": req.replace("_1.json", "_2.json")}, req))
        self.assertTrue(logic.same_request({}, req))
        self.assertEqual(logic.request_files(req)[1:], [req[:-5] + ".result.json", req[:-5] + ".cancel"])

    def test_job_problems(self):
        with tempfile.TemporaryDirectory() as d:
            python = Path(d, "python.exe")
            python.write_bytes(b"")
            args = dict(operation="analyze", folders=["E:/T/lights"], results_folder="E:/T", custom_output=False,
                        python=str(python), reference_on=False, reference="", analysis=None, region=None)
            self.assertIsNone(logic.job_problem(**args))
            self.assertIn("target folder", logic.job_problem(**dict(args, folders=[])))
            self.assertIn("results folder", logic.job_problem(**dict(args, results_folder="", custom_output=True)))
            self.assertIn("Python executable", logic.job_problem(**dict(args, python=d + "/none.exe")))
            self.assertIn("reference light", logic.job_problem(**dict(args, reference_on=True, reference=" ")))
            self.assertIn("heat map", logic.job_problem(**dict(args, operation="stack")))

    def test_hardware_result(self):
        ok = logic.hardware_result(0, '{"supported": true, "summary": "NVIDIA CUDA ready."}\n', "")
        self.assertTrue(ok["supported"])
        bad = logic.hardware_result(1, '{"supported": true, "summary": "x"}', "boom")
        self.assertEqual(bad["code"], "probe_failed")
        self.assertIn("boom", bad["error"])
        self.assertEqual(logic.hardware_result(0, "not json", "")["code"], "probe_failed")
        script = Path(logic.hardware_args("E:/T")[1])
        self.assertEqual((script.parent.name, script.name), ("worker", "hardware_check.py"))
        self.assertEqual(logic.hardware_args("")[-2:], ["--work", logic.work_dir()])
        self.assertEqual(logic.hardware_args("E:/T")[-2:], ["--results", "E:/T"])

    def test_failure_text(self):
        failure = logic.fallback_failure({"error": "boom"}, "stack", {"region": [0, 0, 1, 1]})
        self.assertEqual(failure["operation"], "stack")
        self.assertEqual(failure["summary"], "GPUCoadd could not finish this operation.")
        text = logic.failure_text(failure)
        self.assertTrue(text.startswith("GPUCoadd failure report\nGPUCoadd could not finish"))
        self.assertIn('"region"', text)


class HeatMapChecks(unittest.TestCase):
    def test_heat_map_result_checks(self):
        with tempfile.TemporaryDirectory() as d:
            heat = Path(d, "heat.png")
            heat.write_bytes(b"png")
            result = {"heatmap": str(heat), "canvas": {"width": 10, "height": 8}}
            self.assertEqual(logic.check_analysis(result), str(heat))
            for broken, message in (
                ({"canvas": {"width": 0, "height": 8}}, "invalid image dimensions"),
                ({"canvas": {"width": 1.5, "height": 8}}, "invalid image dimensions"),
                ({"heatmap": ""}, "no preview image"),
                ({"heatmap": d + "/missing.png"}, "not found"),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    logic.check_analysis(dict(result, **broken))


class HostEnvironment(unittest.TestCase):
    def test_worker_environment_drops_the_hosts_python(self):
        venv = r"C:\Users\x\AppData\Local\host\venv"
        env = logic.worker_environment({
            "PYTHONPATH": venv + r"\Lib\site-packages", "VIRTUAL_ENV": venv, "HOST_PIPE": r"\\.\pipe\x",
            "PYTHONUTF8": "1", "Path": venv + r"\Scripts;C:\Windows;" + venv + "\\Scripts\\",
            "CUDA_PATH": r"C:\CUDA", "LOCALAPPDATA": r"C:\Users\x\AppData\Local",
        }, host_variables=("host_pipe",))
        self.assertEqual(env, {"Path": r"C:\Windows", "CUDA_PATH": r"C:\CUDA",
                               "LOCALAPPDATA": r"C:\Users\x\AppData\Local"})
        self.assertEqual(logic.worker_environment({"PATH": r"C:\a;C:\b"}), {"PATH": r"C:\a;C:\b"})


class HostEntry(unittest.TestCase):
    """The entry script for the host program (GPUCoadd/GPUCoadd.py), loaded without running it."""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location("gpucoadd_entry", ROOT / "GPUCoadd" / "GPUCoadd.py")
        cls.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.entry)

    @unittest.skipUnless(sys.platform == "win32", "Windows process tree")
    def test_session_names_the_host_process(self):
        self.assertEqual(self.entry.session_id(), "", "outside the host program there is no session")
        self.assertGreater(self.entry.process_start(os.getpid()), 0)
        parents = self.entry.process_parents()
        self.assertIn(os.getpid(), parents)
        self.assertEqual(parents[os.getpid()][1].lower(), Path(sys.executable).name.lower())


class SessionState(unittest.TestCase):
    def test_restores_only_in_the_same_session(self):
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, {"T/lights": 1})
            path = Path(d, "state.json")
            f = logic.Folders()
            f.set([logic.norm_path(d) + "/T/lights"], [], [d + "/gone/lights"], logic.norm_path(d) + "/T")
            with patch.dict("os.environ", {"LOCALAPPDATA": d}):
                logic.save_state("pipe-1", f, True, "D:\\stacks\\", path)
                state = logic.load_state("pipe-1", path)
                self.assertEqual(state["folders"], [logic.norm_path(d) + "/T/lights"])
                self.assertEqual(state["offered"], [], "folders that no longer exist are dropped")
                self.assertEqual((state["custom_output"], state["output"]), (True, "D:/stacks"))
                self.assertIsNone(logic.load_state("pipe-2", path), "a restart of the host starts empty")
                self.assertIsNone(logic.load_state("", path))
                logic.save_state("", f, False, "", path)  # without a session nothing is written
                self.assertEqual(json.loads(path.read_text())["session"], "pipe-1")


if __name__ == "__main__":
    unittest.main()
