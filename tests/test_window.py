"""The window (GPUCoadd/gpucoadd_qt/dialog.py) driven offscreen against a stand-in worker
(tests/fixtures/fake_worker.py) that follows the real worker's protocol. Needs PyQt6, which
GPUCoadd's CUDA environment does not have, so it is skipped there. Run it with a Python that
has PyQt6: python -m unittest tests.test_window"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPUCoadd"))

try:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
except ImportError:  # GPUCoadd's CUDA environment has no Qt
    QApplication = None

FAKE_WORKER = ROOT / "tests" / "fixtures" / "fake_worker.py"


class Host:
    variables = ("HOST_PIPE",)

    def __init__(self, session="test-session"):
        self.session = session
        self.lines, self.opened = [], []

    def log(self, text, error=False):
        self.lines.append((text, error))

    def open(self, path):
        self.opened.append(path)


@unittest.skipIf(QApplication is None, "PyQt6 is not installed")
class Window(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from gpucoadd_qt import dialog, logic

        self.logic, self.dialog_module = logic, dialog
        self.tmp = tempfile.TemporaryDirectory()
        self.root = logic.norm_path(self.tmp.name)
        for folder, count in (("IC 63/lights", 3), ("IC 63/pro/lights", 2)):
            Path(self.root, folder).mkdir(parents=True)
            for k in range(count):
                Path(self.root, folder, f"Light_{k}.fit").write_bytes(b"")
        probe = Path(self.root, "probe.py")
        probe.write_text('print(\'{"supported": true, "summary": "GPU ready: test GPU."}\')\n', encoding="utf-8")
        self.patches = [
            patch.dict(os.environ, {"LOCALAPPDATA": self.root + "/local"}),
            patch.object(logic, "BRIDGE", FAKE_WORKER),
            patch.object(logic, "HARDWARE_CHECK", probe),
            patch.object(logic, "DEFAULT_PYTHON", Path(sys.executable)),
        ]
        for p in self.patches:
            p.start()
        self.host = Host()
        self.d = dialog.GPUCoaddDialog(self.host)
        self.d.show()
        self.wait(lambda: self.d.hardware_state == "ready")

    def tearDown(self):
        self.d.shutdown()
        self.d.close()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def wait(self, condition, timeout=20):
        end = time.time() + timeout
        while not condition():
            if time.time() > end:
                self.fail("timed out; log:\n" + "\n".join(self.d.logs[-30:]))
            QTest.qWait(20)

    def test_full_flow_with_kept_worker(self):
        d = self.d
        self.assertIn("No lights were read.", d.logs[-1])
        self.assertFalse(d.worker_running(), "opening the window never starts the worker")
        self.assertTrue(d.set_target(self.root + "/IC 63"))
        self.assertEqual(d.folders.folders, [self.root + "/IC 63/lights"])
        self.assertEqual(d.folder_list.topLevelItemCount(), 2, "a second telescope's lights are offered")
        self.assertEqual(d.output.edit.text(), self.root + "/IC 63", "results go to the target folder")
        d.folder_list.topLevelItem(1).setCheckState(0, Qt.CheckState.Checked)
        self.wait(lambda: len(d.folders.folders) == 2)
        d.folder_list.topLevelItem(1).setCheckState(0, Qt.CheckState.Unchecked)
        self.wait(lambda: len(d.folders.folders) == 1)
        self.assertEqual(d.folders.offered, [self.root + "/IC 63/pro/lights"], "an unticked folder stays offered")

        d.check_folders.click()
        self.wait(lambda: not d.busy)
        self.assertTrue(d.group_table.isVisible())
        self.assertIn("3 of 3 lights in 1 group(s)", d.group_hint.text())

        d.build.click()
        self.wait(lambda: not d.busy)
        self.assertIsNotNone(d.analysis, "\n".join(d.logs[-20:]))
        self.assertEqual(d.region, [0, 0, 400, 300])
        self.assertTrue(d.full_frame.isChecked())
        self.assertTrue(d.stack_button.isEnabled())
        self.assertEqual([s["state"] for s in d.stages.stages], ["done"] * 5)
        self.assertTrue(d.storage_panel.isVisible())
        self.assertEqual(d.storage_title.text(), "RAM-backed processing: same image quality")
        self.assertIn("3 lights ready", d.cache_state.text())
        self.assertTrue(d.worker_running(), "Keep lights ready keeps the worker")

        # drag a crop on the heat map, then move it
        rect = d.map.rect_on_screen()
        start = QPoint(int(rect[0] + rect[2] * 0.25), int(rect[1] + rect[3] * 0.25))
        end = QPoint(int(rect[0] + rect[2] * 0.75), int(rect[1] + rect[3] * 0.75))
        QTest.mousePress(d.map, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(d.map, end)
        QTest.mouseRelease(d.map, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
        crop = list(d.region)
        self.assertFalse(d.full_frame.isChecked())
        self.assertTrue(190 <= crop[2] <= 210 and 140 <= crop[3] <= 160, crop)
        self.assertTrue(d.stack_button.text().startswith("Stack Selected Area"))
        d.region_boxes[0].setValue(10)
        self.assertEqual(d.region[0], 10)
        d.full_frame.click()
        self.assertEqual(d.region, [0, 0, 400, 300])
        d.set_region(crop, True)

        d.stack_button.click()
        self.wait(lambda: not d.busy)
        self.assertTrue(d.results_panel.isVisible(), "\n".join(d.logs[-20:]))
        self.assertEqual(len(d.result_files), 1)
        self.assertIn("3 contributing lights", d.result_summary.text())
        self.assertTrue(d.logs[-1].endswith("Reused prepared lights (gpu)."), d.logs[-1])
        d.open_result.click()
        self.assertEqual([self.logic.norm_path(f) for f in self.host.opened], [self.root + "/IC 63/fake_stack.fits"])
        self.assertTrue(any("stack saved" in text for text, _ in self.host.lines))
        requests = list(Path(self.root, "local/GPUCoadd/requests").glob("*"))
        self.assertEqual(requests, [], "request, result and cancel files are removed after each job")

        # Release GPU stops the kept worker; the heat map stays usable
        d.release_button.click()
        self.wait(lambda: not d.worker_running())
        self.assertEqual(d.cache_state.text(), "GPU lights released")
        self.assertIsNotNone(d.analysis)
        self.assertTrue(d.stack_button.isEnabled())

    def test_one_job_worker_cancel_and_invalidation(self):
        d = self.d
        d.keep_ready.setChecked(False)
        d.set_target(self.root + "/IC 63")
        d.custom_output.setChecked(True)
        d.output.edit.setText(self.root + "/cancel here")
        Path(self.root, "cancel here").mkdir()
        d.build.click()
        self.wait(lambda: d.busy and any(s["state"] == "running" for s in d.stages.stages))
        self.assertFalse(d.close_button.isEnabled(), "the window cannot close while a job runs")
        d.cancel.click()
        self.wait(lambda: not d.busy)
        self.assertEqual(d.status.text(), "Operation cancelled.")
        self.assertFalse(d.force_ready, "a worker that answers the cancel is never offered Force Stop")
        self.assertEqual(d.cancel.text(), "Cancel Operation")
        self.assertIn("failed", [s["state"] for s in d.stages.stages])
        self.assertFalse(d.worker_running(), "without Keep lights ready every job has its own worker")

        d.output.edit.setText(self.root + "/IC 63")
        d.build.click()
        self.wait(lambda: not d.busy)
        self.assertIsNotNone(d.analysis)
        previous = d.analysis["analysis_file"]
        d.frame_selection.setChecked(True)  # the sharpness limit: out of date, its alignment reusable
        self.assertIsNone(d.analysis, "a new sharpness limit marks the heat map out of date")
        self.assertFalse(d.stack_button.isEnabled())
        self.assertEqual(d.build_request("analyze")["previous_analysis"], previous)
        d.sharpness.setValue(d.sharpness.value() + 0.5)  # still the same heat map to reuse
        self.assertEqual(d.build_request("analyze")["previous_analysis"], previous)
        d.choices["transform"].setCurrentIndex(1)  # changes the alignment inputs: nothing to reuse
        self.assertNotIn("previous_analysis", d.build_request("analyze"))
        d.choices["transform"].setCurrentIndex(0)
        d.choices["norm"].setCurrentIndex(2)
        self.assertFalse(d.rgb_equal.isChecked())
        self.assertFalse(d.rgb_equal.isEnabled())
        d.choices["norm"].setCurrentIndex(0)
        self.assertTrue(d.rgb_equal.isChecked(), "RGB equalization comes back when available again")

    def start_hung_job(self, out, keep_ready=False):
        """Build Heat Map with a fake worker that stops answering (results in `out`)."""
        d = self.d
        d.keep_ready.setChecked(keep_ready)
        d.set_target(self.root + "/IC 63")
        d.custom_output.setChecked(True)
        d.output.edit.setText(str(out))
        d.force_timer.setInterval(300)
        d.build.click()
        self.wait(lambda: d.job_tag and Path(d.work, "runs", d.job_tag, "tree_ready").exists())
        procs = self.logic.process_tree(d.worker.processId())
        # the worker, its child and grandchild (a venv's python.exe also starts the real interpreter)
        self.assertGreaterEqual(len(procs), 3)
        return procs

    def force_stop(self):
        d = self.d
        d.cancel.click()
        self.wait(lambda: d.cancel.text() == "Force Stop" and d.cancel.isEnabled())
        self.assertIn("has not answered the cancel request", d.status.text())
        d.cancel.click()

    def test_force_stop_ends_a_hung_worker_its_processes_and_unfinished_files(self):
        for keep_ready in (False, True):
            with self.subTest(keep_ready=keep_ready):
                d = self.d
                out = Path(self.root, f"hang {int(keep_ready)}")
                out.mkdir()
                # another job's unfinished file (another window may be writing it) and the user's own
                other = out / ".fake_stack.fits.fake_analyze_1.zz.partial"
                other.write_bytes(b"another job")
                notes = out / "notes.partial.txt"
                notes.write_text("mine")
                procs = self.start_hung_job(out, keep_ready)
                self.force_stop()
                self.wait(lambda: not d.busy)
                self.assertEqual([p for p in procs if self.logic.process_running(p)], [], "every process ended")
                self.assertEqual(d.status.text(), "Stopped. No unfinished file of this job is left. No saved "
                                 "outputs were confirmed; files may have been saved before the stop.")
                left = sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*"))
                self.assertEqual(left, sorted([other.name, notes.name, "fake_aligned"]))
                self.assertEqual(d.cancel.text(), "Cancel Operation")
                self.assertFalse(d.worker_running())
        # the window works again: a new job starts and completes
        d.output.edit.setText(self.root + "/IC 63")
        d.build.click()
        self.wait(lambda: not d.busy)
        self.assertIsNotNone(d.analysis)

    def test_force_stop_names_confirmed_saves_and_files_it_could_not_remove(self):
        d = self.d
        out = Path(self.root, "hang published")
        out.mkdir()
        self.start_hung_job(out)
        held = next(out.glob(".fake_stack.fits.*.partial"))
        with open(held, "rb"):  # another program has the unfinished file open: it can't be deleted
            self.force_stop()
            self.wait(lambda: not d.busy)
        self.assertEqual(d.status.text(), f"Stopped. These unfinished files could not be removed: "
                         f"{self.logic.norm_path(str(held))}. Other files may have been saved before the stop.")
        self.assertEqual(Path(out, "fake_stack.fits").read_bytes(), b"complete")
        d.cancel.click()  # idle again: Cancel does nothing
        held.unlink()
        out2 = Path(self.root, "hang published again")
        out2.mkdir()
        self.start_hung_job(out2)
        self.force_stop()
        self.wait(lambda: not d.busy)
        self.assertEqual(d.status.text(), "Stopped. No unfinished file of this job is left. Confirmed saved by "
                         "this job: fake_stack.fits. Other files may have been saved before the stop.")

    def test_failure_report_and_session_state(self):
        d = self.d
        d.set_target(self.root + "/IC 63")
        d.build.click()
        self.wait(lambda: not d.busy)
        report = json.loads(Path(d.analysis["files"]["report"]).read_text(encoding="utf-8"))
        self.assertEqual(report["operation"], "analyze")
        d.result_files = [{"path": self.root + "/gone.fits", "label": "Stacked image"}]
        d.result_choice.addItem("Stacked image")
        d.result_choice.setCurrentIndex(0)
        d.open_result_clicked()
        self.assertTrue(d.failure_panel.isVisible())
        self.assertEqual(d.last_failure["operation"], "open_result")
        self.assertIsNotNone(d.analysis, "a failed action keeps the heat map")
        self.assertTrue(Path(d.last_failure["error_report"]).is_file())
        self.assertTrue(any(error for _, error in self.host.lines), "failures also go to the host's console")
        d.shutdown()
        d.close()
        again = self.dialog_module.GPUCoaddDialog(self.host)
        self.assertEqual(again.folders.folders, [self.root + "/IC 63/lights"], "same session of the host: restored")
        other = self.dialog_module.GPUCoaddDialog(Host("another-session"))
        self.assertEqual(other.folders.folders, [], "a restart of the host starts empty")
        for window in (again, other):
            window.shutdown()
            window.close()
        self.d = self.dialog_module.GPUCoaddDialog(self.host)

    def test_output_warnings_stay_visible(self):
        d = self.d
        d.set_target(self.root + "/IC 63")
        d.build.click()
        self.wait(lambda: not d.busy)
        self.assertFalse(d.output_warning.isVisible())
        d.output_controls["aligned_lights"].setChecked(True)
        d.stack_button.click()
        self.wait(lambda: not d.busy)
        self.assertTrue(d.output_warning.isVisible(), "a missing export is shown outside the activity log")
        self.assertIn("Stack saved, but requested outputs are missing or incomplete", d.output_warning.text())
        self.assertIn("aligned lights export was not saved", d.output_warning.text())
        self.assertEqual(d.status.text(), "Stack saved with warnings: check missing outputs below.")
        self.assertTrue(d.results_panel.isVisible(), "the stack itself is still offered")
        d.output_controls["aligned_lights"].setChecked(False)
        d.stack_button.click()
        self.wait(lambda: not d.busy)
        self.assertFalse(d.output_warning.isVisible(), "a new job clears the previous warning")

    def test_python_problems_block_work(self):
        d = self.d
        d.python.edit.setText(self.root + "/missing/python.exe")
        d.python_changed()
        self.wait(lambda: d.hardware_state == "blocked")
        self.assertFalse(d.build.isEnabled())
        self.assertEqual(d.last_failure["operation"], "hardware_check")
        d.python.edit.setText(sys.executable)
        d.python_changed()
        self.wait(lambda: d.hardware_state == "ready")
        self.assertFalse(d.failure_panel.isVisible(), "a passing recheck clears the GPU failure")

    def test_fits_a_1080p_screen(self):
        d = self.d
        d.set_target(self.root + "/IC 63")
        d.show_groups([{"key": "k", "label": "S50", "lights": 3, "total_exposure_s": 30, "used": True, "main": True}])
        d.show_results({"canvas": {"width": 9, "height": 9}, "contributing": 3, "wall_s": 1,
                        "image_outputs": [{"label": "Stacked image", "format": "FITS", "path": "x.fits"}]})
        # 883 px: a 1080p screen less the taskbar, a host program's own menus and the title bar
        left = d.left_panel.minimumSizeHint().height()
        self.assertLessEqual(left, 883)
        for check, panel in d.sections:
            self.assertFalse(d.left_panel.isAncestorOf(panel), "option sections open in the right column")
            check.setChecked(True)
            self.assertEqual(sum(panel.isVisible() for _, panel in d.sections), 1, "one options section at a time")
            right = d.right_layout.minimumSize().height()
            self.assertEqual(d.left_panel.minimumSizeHint().height(), left, "opening a section leaves the left column")
            self.assertLessEqual(right, 883, check.text())
            self.assertLessEqual(d.height(), d.screen_size()[1] - 40, "the window never grows past the screen")
        for _, combo in d.choices.items():
            self.assertGreater(len(combo.toolTip()), 20)
        for box in d.output_controls.values():
            self.assertGreater(len(box.toolTip()), 20)


if __name__ == "__main__":
    unittest.main()
