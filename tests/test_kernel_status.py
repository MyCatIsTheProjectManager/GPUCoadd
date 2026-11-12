"""Which path each GPU stage takes, and why: the selections (KernelChoice), what ran in a job
(GPUBackend.kernel_status), the report and performance warnings, the self-test and the window."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "GPUCoadd"))
import gpucoadd as g  # noqa: E402
from gpucoadd_qt import logic  # noqa: E402
from worker.bridge import kernel_warnings, run_request  # noqa: E402

BROKEN = RuntimeError("nvrtc: test build failure")


def cuda():
    import torch

    return torch.cuda.is_available()


class Selections(unittest.TestCase):
    def setUp(self):
        if not cuda():
            self.skipTest("CUDA is not available")

    def test_kernel_switch_and_build_failure(self):
        for name, select, module, variable in (
            ("warp", g.gpu_warp, "warp_kernels.module", "GPUCOADD_WARP"),
            ("clip", g.gpu_clip, "integrate_kernels.module", "GPUCOADD_CLIP"),
            ("calibrate", g.gpu_calibrate, "calibrate_kernels.module", "GPUCOADD_CALIBRATE"),
            ("sort", lambda: g.gpu_sort(40), "sort_kernels.prepare", "GPUCOADD_SORT"),
        ):
            with self.subTest(name):
                impl, choice = select()
                self.assertIsNotNone(impl)
                self.assertEqual((choice.path, choice.reason, choice.error), ("kernel", None, None))
                with patch.dict(os.environ, {variable: "torch"}):
                    impl, choice = select()
                self.assertIsNone(impl)
                self.assertEqual((choice.path, choice.reason), ("torch", "switch"))
                with patch(module, side_effect=BROKEN):
                    impl, choice = select()
                self.assertIsNone(impl)
                self.assertEqual((choice.path, choice.reason), ("torch", "build failed"))
                self.assertEqual(choice.error, "RuntimeError: nvrtc: test build failure")
                self.assertIn(choice.error, choice)  # the log line still says it

    def test_rcd_chain(self):
        self.assertEqual(g.gpu_rcd()[1].path, "kernel")
        with patch("rcd_kernels.module", side_effect=BROKEN):
            choice = g.gpu_rcd()[1]
            self.assertEqual((choice.path, choice.reason), ("jiterator", "build failed"))
            with patch("rcd_fused._kernels", side_effect=ValueError("no jiterator")):
                choice = g.gpu_rcd()[1]
        self.assertEqual((choice.path, choice.reason), ("reference", "build failed"))
        self.assertEqual(choice.error, "RuntimeError: nvrtc: test build failure; fused: ValueError: no jiterator")
        with patch.dict(os.environ, {"GPUCOADD_RCD": "jiterator"}):
            self.assertEqual(g.gpu_rcd()[1].reason, "switch")
        with patch.dict(os.environ, {"GPUCOADD_RCD": "reference"}):
            self.assertEqual((g.gpu_rcd()[1].path, g.gpu_rcd()[1].reason), ("reference", "switch"))


class BackendStatus(unittest.TestCase):
    def setUp(self):
        if not cuda():
            self.skipTest("CUDA is not available")

    def test_runtime_counts_reset_and_the_sort_above_512_lights(self):
        import torch

        rng = np.random.default_rng(9)
        raws = [(900 + rng.normal(0, 10, (64, 80))).astype(np.uint16) for _ in range(3)]
        be = g.GPUBackend(iter(raws), 100, 3, "bicubic", "rcd")
        mats = [np.eye(3)] * 3
        coefs = {i: (np.ones(3, np.float32), np.zeros(3, np.float32)) for i in range(3)}
        st = be.alloc_stack(3, 32, 32)
        be.warp_tile(st, [0, 1, 2], mats, coefs, 8, 40, 8, 40)
        be.integrate(st, 3.0, 2)
        status = be.kernel_status()
        for stage in ("calibrate", "rcd", "warp", "sort", "clip"):
            self.assertEqual(status[stage]["selected"], "kernel", stage)
            self.assertEqual(status[stage]["runtime"], {"kernel": 1}, stage)
        self.assertEqual(status["feather"]["selected"], "not built")
        self.assertEqual(status["feather"]["runtime"], {})
        # a kernel that built is not one that ran: above 512 lights the sort is torch's, by design
        be.integrate(torch.rand(600, 3, 2, 2, device="cuda"), 3.0, 2)
        self.assertEqual(be.kernel_status()["sort"]["runtime"], {"kernel": 1, "torch (more than 512 lights)": 1})
        self.assertEqual(kernel_warnings(be.kernel_status()), [], "a path taken by design is not a warning")
        be.reset_kernel_calls()
        status = be.kernel_status()
        self.assertEqual({s["runtime"] == {} for s in status.values()}, {True})
        self.assertEqual(status["warp"]["selected"], "kernel", "the choices stay")

    def test_warnings_only_for_build_failures_that_ran(self):
        failed = {"reason": "build failed", "error": "RuntimeError: x", "runtime": {"torch": 2}}
        self.assertEqual([w["stage"] for w in kernel_warnings({"warp": failed})], ["warp"])
        self.assertEqual(kernel_warnings({"warp": {**failed, "runtime": {}}}), [])
        self.assertEqual(kernel_warnings({"warp": {**failed, "reason": "switch"}}), [])
        w = kernel_warnings({"rcd": failed})[0]
        self.assertEqual((w["kind"], w["output"]), ("performance", None))
        self.assertEqual(w["message"], "The GPU kernel for RCD debayering could not be built; this stage ran on a "
                                       "slower path. RuntimeError: x")


class WorkerReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not cuda():
            raise unittest.SkipTest("CUDA is not available")
        from test_combinations import write_scene

        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        write_scene(cls.root / "single" / "lights", "single")
        cls.jobs = 0

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        for target in ("raw_storage.available_ram", "output_store.available_ram", "resource_guard.available_ram",
                       "sky_store.available_ram"):
            p = patch(target, return_value=64 << 30)
            p.start()
            self.addCleanup(p.stop)

    def job(self, operation, session=None, env=None, **extra):
        WorkerReports.jobs += 1
        path = self.root / "requests" / f"{operation}_{self.jobs}.json"
        path.parent.mkdir(exist_ok=True)
        settings = {"debayer": "rcd", "interp": "bicubic", "norm": "spread", "weight": "none",
                    "transform": "similarity", "rejection": "sigma", "select": 0, "feather": 0, "rgb_equal": True,
                    "match_gradients": False, "threads": 2, "tile": 0}
        path.write_text(json.dumps({
            "operation": operation, "lights_dirs": [str(self.root / "single" / "lights")], "exclude_groups": [],
            "output_dir": str(self.root / f"out_{self.jobs}"), "work_dir": str(self.root / "work"),
            "reference": "", "settings": settings, "outputs": {}, **extra,
        }))
        with patch.dict(os.environ, {"GPUCOADD_STORAGE": "gpu", **(env or {})}), redirect_stdout(io.StringIO()):
            run_request(path, session)
        result = json.loads(path.with_suffix(".result.json").read_text())
        self.assertEqual(result["status"], "complete", result.get("error"))
        return result

    def stack(self, heat, session=None, env=None):
        c = heat["canvas"]
        return self.job("stack", session, env, analysis_file=heat["analysis_file"],
                        region=[0, 0, c["width"], c["height"]])

    def test_build_failure_is_reported_and_warned_a_switch_only_reported(self):
        heat = self.job("analyze")
        with patch("warp_kernels.module", side_effect=BROKEN):
            broken = self.stack(heat)
        warp = broken["kernels"]["warp"]
        self.assertEqual((warp["selected"], warp["reason"]), ("torch", "build failed"))
        self.assertGreater(warp["runtime"]["torch"], 0)
        self.assertEqual([(w["kind"], w["stage"]) for w in broken["warnings"]], [("performance", "warp")])
        report = json.loads(Path(broken["files"]["report"]).read_text())
        self.assertEqual(report["kernels"]["warp"]["error"], "RuntimeError: nvrtc: test build failure")
        switched = self.stack(heat, env={"GPUCOADD_WARP": "torch"})
        self.assertEqual(switched["kernels"]["warp"]["reason"], "switch")
        self.assertEqual(switched["warnings"], [])

    def test_a_kept_backend_reports_each_job_on_its_own(self):
        from worker.worker_session import WorkerSession

        session = WorkerSession()
        try:
            heat = self.job("analyze", session)
            first = self.stack(heat, session)
            second = self.stack(heat, session)
        finally:
            session.release()
        self.assertEqual(heat["kernels"]["warp"]["runtime"], {}, "a heat map warps nothing")
        self.assertGreater(first["kernels"]["warp"]["runtime"]["kernel"], 0)
        self.assertEqual(second["kernels"], first["kernels"], "counted per job, not summed over the session")


class SelfTest(unittest.TestCase):
    def test_a_kernel_that_cannot_be_built_makes_the_result_slow(self):
        if not cuda():
            self.skipTest("CUDA is not available")
        from worker import selftest

        out = io.StringIO()
        with patch("calibrate_kernels.module", side_effect=BROKEN), redirect_stdout(out):
            code = selftest.main()
        result = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual(code, 0, "setup completes: GPUCoadd works, only slower")
        self.assertEqual(result["result"], "slow")
        self.assertIn("tile calibration", result["summary"])
        self.assertEqual(result["error"], "calibrate: RuntimeError: nvrtc: test build failure")


class WindowTexts(unittest.TestCase):
    def test_slow_stages_have_their_own_title(self):
        slow = {"kind": "performance", "message": "The GPU kernel for warping could not be built; ..."}
        missing = {"kind": "environment", "message": "The frame report (CSV) was not saved: ..."}
        status, text = logic.finish_warnings("stack", [slow])
        self.assertEqual(status, "Stack saved, but slower than it should be: see below.")
        self.assertTrue(text.startswith("Completed, but slower than it should be:"))
        self.assertNotIn("missing", text)
        status, text = logic.finish_warnings("stack", [missing, slow])
        self.assertEqual(status, "Stack saved with warnings: check missing outputs below.")
        self.assertLess(text.index("missing or incomplete"), text.index("slower than it should be"))
        self.assertEqual(logic.finish_warnings("analyze", [missing])[0], "Completed with warnings: see below.")


if __name__ == "__main__":
    unittest.main()
