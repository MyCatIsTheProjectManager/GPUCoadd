"""Cache identity, crop eviction, idle cleanup and single-request file protocol."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker.worker_session import WorkerSession, serve


class WorkerSessionTests(unittest.TestCase):
    def test_geometry_reuse_dropping_first_light_matches_fresh_pixels(self):
        import numpy as np
        import torch
        import gpucoadd as g
        from worker.bridge import backend_geometry

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(731)
        shapes = [(80, 96), (128, 160), (128, 160)]
        raws = [(1000 + rng.normal(0, 12, s)).astype(np.uint16) for s in shapes]
        before, after = backend_geometry(shapes), backend_geometry(shapes[1:])
        self.assertEqual(before, after)
        self.assertNotEqual(backend_geometry(shapes[:1]), before)
        backend = g.GPUBackend(iter(raws), 100, 3, sizes=shapes)
        cache = dict(
            mats=[np.eye(3).tolist()] * 3,
            fwhm=[3.0] * 3,
            ref_idx=1,
            ratios=[[1.0, 1.0, 1.0]] * 3,
            sizes=[[w, h] for h, w in shapes],
        )
        cache["mats"][0] = [
            [1.0, 0.0, 200.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]  # first light does not reach the reference field
        session = WorkerSession()
        session.remember(backend, {"sensor_setup": before})
        reused = session.acquire({"sensor_setup": after}, [1, 2])
        self.assertIs(reused, backend)
        fresh = g.GPUBackend(iter(raws[1:]), 100, 3, indices=[1, 2], sizes=shapes[1:])
        kwargs = dict(tile=64, reg_workers=1, analysis_cache=cache, select=0, log=lambda _: None)
        try:
            a, _, info_a = g.stack(reused, **kwargs)
            b, _, info_b = g.stack(fresh, **kwargs)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(info_a["coverage"], info_b["coverage"])
            self.assertEqual(info_a["contributing_indices"], info_b["contributing_indices"])
        finally:
            session.release()

    def test_cancellation_preserves_cli_status_and_closes_session(self):
        import json
        from worker import bridge

        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            request = Path(d) / "request.json"
            request.write_text(
                json.dumps({"operation": "analyze", "output_dir": str(Path(d) / "results"), "work_dir": d})
            )
            with patch.object(bridge, "execute", side_effect=bridge.Cancelled):
                self.assertEqual(bridge.run_request(request), 0)
                self.assertEqual(bridge.run_request(request, WorkerSession()), 1)
            self.assertEqual(json.loads(request.with_suffix(".result.json").read_text())["status"], "cancelled")
            self.assertFalse((Path(d) / "results").exists())  # nothing written to the output folder
            self.assertEqual(len(list((Path(d) / "runs").iterdir())), 2)  # run reports in the work area

    def backend(self):
        be = Mock()
        be.loaded = [0, 1, 2]
        be.timer = SimpleNamespace(t={"previous": 17})
        be.torch.cuda.memory_allocated.return_value = 2**30

        def keep(indices):
            be.loaded = [i for i in be.loaded if i in indices]

        be.keep_only.side_effect = keep
        return be

    def test_cache_reuse_crop_and_reload(self):
        s = WorkerSession()
        b = self.backend()
        s.remember(b, {"input": 1})
        self.assertIs(s.acquire({"input": 1}, [0, 2]), b)
        self.assertEqual(b.loaded, [0, 2])
        self.assertEqual(b.timer.t, {})
        b.torch.cuda.reset_peak_memory_stats.assert_called_once()
        self.assertEqual(s.idle()["resident_lights"], 2)
        b.reserve_stack.assert_called_with(0, 1, 1)
        self.assertIsNone(s.acquire({"input": 1}, [0, 1, 2]))
        self.assertIsNone(s.backend)
        s.remember(b, {"input": 1})
        self.assertIsNone(s.acquire({"input": 2}, [0]))

    def test_idle_timeout_and_release_files(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as out:
            session = Path(d) / "worker_1"
            self.assertEqual(serve(session, Mock(), timeout=0.01), 0)
            self.assertIn("idle timeout", out.getvalue())
            self.assertFalse(session.exists())  # the session folder goes with the session
            session.mkdir()
            Path(session, "stop").touch()
            self.assertEqual(serve(session, Mock()), 0)
            self.assertFalse(session.exists())

    def test_changed_worker_code_requires_restart(self):
        s = WorkerSession()
        s.validate_source()
        s.source = "outdated"
        with self.assertRaisesRegex(ValueError, "Worker code changed"):
            s.validate_source()

    def test_job_is_consumed_once_and_failure_exits(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            path = Path(d)
            (path / "command.txt").write_text(str(path / "request.json"))
            (path / "ready").touch()
            run = Mock(return_value=1)
            serve(path, run)
            run.assert_called_once()
            self.assertFalse((path / "ready").exists())

    def test_command_may_be_a_json_string(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            path = Path(d)
            (path / "command.txt").write_text('"C:/\\u010c\\ud840\\udc0b/request.json"')
            (path / "ready").touch()
            run = Mock(return_value=1)
            serve(path, run)
            self.assertEqual(run.call_args[0][0], Path("C:/Č𠀋/request.json"))

    def test_two_jobs_are_serial_and_keep_one_session(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            path = Path(d)
            seen = []
            (path / "command.txt").write_text(str(path / "one.json"))
            (path / "ready").touch()

            def run(request, session):
                seen.append((request.name, id(session)))
                if len(seen) == 1:
                    (path / "command.txt").write_text(str(path / "two.json"))
                    (path / "ready").touch()
                else:
                    (path / "stop").touch()
                return 0

            serve(path, run)
            self.assertEqual([name for name, _ in seen], ["one.json", "two.json"])
            self.assertEqual(seen[0][1], seen[1][1])


if __name__ == "__main__":
    unittest.main()
