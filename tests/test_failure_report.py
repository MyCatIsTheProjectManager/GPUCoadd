"""Failure reporting must work without activity logging or a working output drive."""

from contextlib import redirect_stdout
import errno
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker import bridge
from worker.failure_report import build_failure, explain, save_failure


class FailureReportTests(unittest.TestCase):
    def test_simple_explanations(self):
        cases = [
            (RuntimeError("CUDA out of memory"), "graphics card"),
            (MemoryError("allocation failed"), "could not provide the memory"),
            (OSError(errno.ENOSPC, "write failed"), "disk space"),
            (OSError("Not enough free disk space on E:"), "disk space"),
            (PermissionError("locked.fit"), "access"),
            (FileNotFoundError("missing.fit"), "not be found"),
            (OSError("FITS pixel verification failed"), "integrity check"),
            (MemoryError("Available GPU memory fell below the requirement"), "graphics-card memory decreased"),
            (ValueError("Unusable normalization statistics for bad.fit"), "brightness or noise"),
            (ValueError("unknown error"), "could not finish"),
        ]
        for error, expected in cases:
            with self.subTest(error=error):
                self.assertIn(expected, explain(error)[0])
                self.assertTrue(explain(error)[1])

    def test_failed_run_without_activity_logging(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as out:
            root = Path(d)
            request = root / "request.json"
            request.write_text(
                json.dumps(
                    dict(
                        operation="stack",
                        work_dir=d,
                        output_dir=str(root / "results"),
                        settings={"debayer": "rcd"},
                        outputs={"activity_log": False},
                    )
                )
            )

            def fail(request, reporter, folder, session):
                reporter.progress("stacking", 1, 4)
                reporter.message("Reading C:/lights/étoile.fit")
                raise RuntimeError("CUDA out of memory: requested 20 MiB")

            with patch.object(bridge, "execute", side_effect=fail):
                self.assertEqual(bridge.run_request(request), 1)
            result = json.loads(request.with_suffix(".result.json").read_text())
            failure = result["failure"]
            self.assertEqual(failure["stage"], "stacking")
            self.assertEqual(failure["request"]["settings"], {"debayer": "rcd"})
            self.assertIn("RuntimeError", failure["traceback"])
            self.assertIn("source_sha256", failure)
            text = Path(result["error_report"]).read_text(encoding="utf-8")
            self.assertIn("graphics card", text)
            self.assertIn("étoile.fit", text)
            self.assertIn("requested 20 MiB", text)
            self.assertFalse(list(root.rglob("activity.jsonl")))
            self.assertFalse((root / "results").exists())
            self.assertIn('"type": "finished"', out.getvalue())

    def test_bad_request_before_reporter_exists(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as out:
            request = Path(d) / "request.json"
            request.write_text("{invalid")
            with patch.object(bridge, "work_root", return_value=Path(d)):
                self.assertEqual(bridge.run_request(request), 1)
            result = json.loads(request.with_suffix(".result.json").read_text())
            self.assertEqual(result["failure"]["error_type"], "JSONDecodeError")
            self.assertTrue(Path(result["error_report"]).exists())
            self.assertIn('"type": "finished"', out.getvalue())

    def test_report_folder_unwritable_uses_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            blocked = root / "blocked"
            blocked.write_text("not a directory")
            report = build_failure(PermissionError("locked"), {"operation": "stack"})
            path, errors = save_failure(report, blocked, root)
            self.assertTrue(Path(path).is_relative_to(root / "errors"))
            self.assertTrue(errors)
            self.assertFalse(list(root.rglob("*.partial")))

    def test_final_report_failure_still_emits_details(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as out:
            request = Path(d) / "request.json"
            request.write_text(json.dumps(dict(operation="scan", work_dir=d)))
            with (
                patch.object(bridge, "execute", return_value={"status": "complete"}),
                patch.object(bridge, "write_json", side_effect=PermissionError("report.json locked")),
            ):
                self.assertEqual(bridge.run_request(request), 1)
            finished = json.loads(out.getvalue().strip().splitlines()[-1])
            self.assertEqual(finished["status"], "error")
            self.assertIn("report.json locked", Path(finished["error_report"]).read_text())
            self.assertIn("PermissionError", finished["failure"]["traceback"])

    def test_all_report_writes_fail_keep_live_details(self):
        report = build_failure(OSError(errno.ENOSPC, "disk full"), {})
        with (
            patch("worker.failure_report.tempfile.mkstemp", side_effect=OSError(errno.ENOSPC, "disk full")),
            patch("worker.failure_report.Path.mkdir"),
        ):
            path, errors = save_failure(report, Path("unused"), Path("unused"))
        self.assertIsNone(path)
        self.assertEqual(len(errors), 3)
        self.assertIn("disk full", report["error"])

    def test_recent_events_bounded_even_with_logging_off(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            reporter = bridge.Reporter(Path(d), Path(d) / "cancel", save_log=False)
            for i in range(150):
                reporter.message(str(i))
            self.assertEqual(len(reporter.recent_events), 80)
            self.assertEqual(reporter.recent_events[0]["message"], "70")


if __name__ == "__main__":
    unittest.main()
