"""Storage choices reach the UI/report without requiring optional activity logging."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from output_store import OutputWorkspace
from worker.bridge import run_request
from sky_store import SkyWorkspace


class StorageNoticeTests(unittest.TestCase):
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

    def test_raw_notice_precedes_loading_when_shapes_are_known(self):
        import numpy as np
        import torch
        from gpucoadd import GPUBackend

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        notices = []

        def raws():
            self.assertEqual(notices, [("raw", "ram", "forced")])
            yield np.ones((64, 80), np.uint16)

        be = GPUBackend(
            raws(),
            0,
            1,
            paths=["unused.fit"],
            sizes=[(64, 80)],
            storage="ram",
            storage_notice=lambda *args, **details: notices.append(args),
        )
        del be

    def test_events_and_final_report_with_logging_off(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as stdout:
            request = Path(d) / "request.json"
            request.write_text(
                json.dumps(
                    {
                        "operation": "stack",
                        "work_dir": d,
                        "output_dir": str(Path(d) / "out"),
                        "outputs": {"activity_log": False},
                    }
                )
            )

            def execute(req, reporter, folder, session):
                reporter.storage("raw", "ram", "capacity")
                reporter.storage("sky", "disk", "capacity")
                return {"status": "complete", "operation": "stack"}

            with patch("worker.bridge.execute", side_effect=execute):
                self.assertEqual(run_request(request), 0)
            result = json.loads(request.with_suffix(".result.json").read_text())
            self.assertEqual(len(result["storage_notices"]), 2)
            events = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual(len([e for e in events if e["type"] == "storage"]), 2)
            self.assertFalse(list(Path(d).rglob("activity.jsonl")))

    def test_planners_and_allocation_fallback_emit_structured_notices(self):
        import numpy as np
        from resource_guard import RAM_RESERVE

        notices, details = [], []

        def callback(*args, **kwargs):
            notices.append(args)
            details.append(kwargs)

        with tempfile.TemporaryDirectory() as d:
            with OutputWorkspace(d, d, {}, mode="disk", storage_notice=callback) as work:
                work.plan(16, 16)
            self.assertEqual(notices[-1], ("output", "disk", "forced"))
            # output notices say how much RAM was free and needed, so the dialog can advise
            self.assertGreater(details[-1]["free_ram_bytes"], 0)
            self.assertGreater(details[-1]["needed_ram_bytes"], 16 * 16 * 16)
            with OutputWorkspace(d, d, {}, mode="auto", storage_notice=callback) as work:
                work.plan(16, 16)
                with patch("resource_guard.available_ram", side_effect=[RAM_RESERVE, RAM_RESERVE + 1]):
                    work.allocate("coverage", (16, 16), np.uint32)
                self.assertEqual(notices[-1], ("output", "mixed", "capacity"))
                self.assertGreater(details[-1]["free_ram_bytes"], 0)
            with SkyWorkspace((4, 3, 16, 16), "cpu", 0, d, mode="disk", storage_notice=callback):
                pass
            self.assertEqual(notices[-1], ("sky", "disk", "forced"))
            with SkyWorkspace((4, 3, 16, 16), "cpu", 0, d, storage_notice=callback):
                pass
            self.assertEqual(notices[-1], ("sky", "ram", "capacity"))


if __name__ == "__main__":
    unittest.main()
