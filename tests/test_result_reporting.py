"""One outcome per job: optional outputs (activity log, frames CSV) never change or replace it."""

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


class Session:
    """Stand-in for the persistent worker session."""

    def __init__(self):
        self.released = 0

    def validate_source(self):
        pass

    def release(self):
        self.released += 1

    def idle(self):
        return {}


class ResultReportingTests(unittest.TestCase):
    def run_job(self, root, execute, session=None, copy_error=None, activity_log=True):
        request = root / "request_1.json"
        request.write_text(
            json.dumps(
                {
                    "operation": "stack",
                    "work_dir": str(root / "work"),
                    "output_dir": str(root / "out"),
                    "outputs": {"activity_log": activity_log},
                }
            ),
            encoding="utf-8",
        )
        patches = [patch.object(bridge, "execute", side_effect=execute)]
        if copy_error is not None:
            patches.append(patch("shutil.copyfile", side_effect=copy_error))
        out = io.StringIO()
        with redirect_stdout(out):
            for p in patches:
                p.start()
            try:
                code = bridge.run_request(request, session)
            finally:
                for p in reversed(patches):
                    p.stop()
        events = [json.loads(line) for line in out.getvalue().splitlines() if line.startswith("{")]
        result = json.loads(request.with_suffix(".result.json").read_text(encoding="utf-8"))
        return code, events, result, request

    @staticmethod
    def complete(request, reporter, folder, session):
        reporter.message("stacked")
        return {"status": "complete", "operation": "stack", "files": {}, "identity": {"basename": "IC_63_stack"}}

    def test_refused_log_copy_is_one_complete_result_with_a_warning(self):
        with tempfile.TemporaryDirectory() as d:
            root, session = Path(d), Session()
            code, events, result, request = self.run_job(
                root, self.complete, session, copy_error=PermissionError(13, "Permission denied", "x_activity.jsonl")
            )
            finished = [e for e in events if e["type"] == "finished"]
            self.assertEqual([e["status"] for e in finished], ["complete"])
            self.assertEqual(code, 0, "a persistent worker keeps its prepared lights")
            self.assertEqual(session.released, 0)
            self.assertEqual(result["status"], "complete")
            self.assertNotIn("activity_log", result["files"])
            self.assertEqual([(w["output"], w["kind"]) for w in result["warnings"]], [("activity_log", "environment")])
            self.assertIn("PermissionError", result["warnings"][0]["error"])
            self.assertEqual(result["request_file"], str(request.resolve()))
            # the warning reaches the interface before the outcome
            kinds = [e["type"] for e in events]
            self.assertLess(kinds.index("warning"), kinds.index("finished"))
            self.assertEqual(finished[0]["warnings"], result["warnings"])
            self.assertFalse(list(root.rglob("*.partial")))

    def test_refused_log_copy_keeps_the_original_error(self):
        def fail(request, reporter, folder, session):
            raise OSError(errno.ENOSPC, "Not enough free disk space on E:\\ for the stack")

        with tempfile.TemporaryDirectory() as d:
            code, events, result, _ = self.run_job(
                Path(d), fail, Session(), copy_error=PermissionError(13, "Permission denied")
            )
            self.assertEqual([e["status"] for e in events if e["type"] == "finished"], ["error"])
            self.assertEqual(code, 1)
            self.assertIn("Not enough free disk space", result["error"])
            self.assertIn("disk space", result["failure"]["summary"])
            self.assertEqual([w["output"] for w in result["warnings"]], ["activity_log"])

    def test_copy_failure_from_our_own_code_is_reported_as_internal(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, result, _ = self.run_job(Path(d), self.complete, copy_error=TypeError("bad argument"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["warnings"][0]["kind"], "internal")
            self.assertIn("GPUCoadd bug", result["warnings"][0]["message"])

    def test_saved_copy_ends_with_the_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            code, events, result, _ = self.run_job(Path(d), self.complete, Session())
            self.assertEqual(code, 0)
            self.assertEqual(result["warnings"], [])
            copy = Path(result["files"]["activity_log"])
            self.assertEqual(copy.name, "IC_63_stack_activity.jsonl")
            lines = [json.loads(line) for line in copy.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(lines[-1]["type"], "finished")
            self.assertEqual(lines[-1]["status"], "complete")
            self.assertEqual(len([e for e in events if e["type"] == "finished"]), 1)

    def test_log_write_failure_during_the_run_is_a_warning(self):
        class BrokenLog:
            def write(self, _):
                raise OSError(errno.ENOSPC, "No space left on device")

            def close(self):
                pass

        def execute(request, reporter, folder, session):
            reporter.log_file.close()
            reporter.log_file = BrokenLog()
            reporter.message("still shown live")
            return self.complete(request, reporter, folder, session)

        with tempfile.TemporaryDirectory() as d:
            code, events, result, _ = self.run_job(Path(d), execute)
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "complete")
            self.assertIn("still shown live", [e.get("message") for e in events])
            self.assertEqual([w["output"] for w in result["warnings"]], ["activity_log"])
            self.assertNotIn("activity_log", result["files"])
            self.assertFalse((Path(d) / "out").exists(), "no copy of an incomplete log")

    def test_optional_file_is_published_whole_or_not_at_all(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "out"
            reporter = bridge.Reporter(Path(d), Path(d) / "cancel", save_log=False)
            files = {}

            def half_written(target):
                Path(target).write_text("file,group\nlight_1", encoding="utf-8")
                raise OSError(errno.ENOSPC, "No space left on device")

            with redirect_stdout(io.StringIO()):
                bridge.save_optional(files, "frames_csv", folder / "x_frames.csv", half_written, reporter, "The CSV")
            self.assertEqual(files, {})
            self.assertEqual(list(folder.iterdir()), [], "neither the partial nor the final file remains")
            self.assertEqual(reporter.warnings[0]["kind"], "environment")
            with redirect_stdout(io.StringIO()):
                bridge.save_optional(
                    files, "frames_csv", folder / "x_frames.csv",
                    lambda t: Path(t).write_text("file\n", encoding="utf-8"), reporter, "The CSV",
                )
            self.assertEqual([p.name for p in folder.iterdir()], ["x_frames.csv"])
            self.assertEqual(files["frames_csv"], str(folder / "x_frames.csv"))


if __name__ == "__main__":
    unittest.main()
