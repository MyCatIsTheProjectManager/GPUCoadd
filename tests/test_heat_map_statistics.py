"""A heat map with its lights on the GPU measures each light's statistics while the CPU registers
(gpucoadd.register_while_measuring); the stack job reads them instead of calibrating every light
again. Stacks must be identical to ones whose job measures them itself."""

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
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import gpucoadd  # noqa: E402
from test_combinations import write_scene  # noqa: E402
from worker.bridge import run_request  # noqa: E402

SETTINGS = {"debayer": "rcd", "interp": "bicubic", "norm": "spread", "weight": "noise", "transform": "similarity",
            "rejection": "noise", "select": 0, "feather": 0, "rgb_equal": True, "match_gradients": False,
            "threads": 2, "tile": 0}


class HeatMapStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        write_scene(cls.root / "mosaic" / "lights", "mosaic")
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

    def run_job(self, operation, **extra):
        HeatMapStatistics.jobs += 1
        path = self.root / "requests" / f"{operation}_{self.jobs}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({
            "operation": operation, "lights_dirs": [str(self.root / "mosaic" / "lights")], "exclude_groups": [],
            "output_dir": str(self.root / f"out_{self.jobs}"), "work_dir": str(self.root / "work"),
            "reference": "", "settings": SETTINGS, "outputs": {}, **extra,
        }))
        with patch.dict(os.environ, {"GPUCOADD_STORAGE": "gpu"}), redirect_stdout(io.StringIO()):
            run_request(path)
        return json.loads(path.with_suffix(".result.json").read_text())

    def stack(self, heat, analysis_file=None):
        c = heat["canvas"]
        got = self.run_job("stack", analysis_file=analysis_file or heat["analysis_file"],
                           region=[0, 0, c["width"], c["height"]])
        self.assertEqual(got["status"], "complete", got.get("error"))
        with fits.open(got["files"]["stack_fits"]) as f:
            return np.ascontiguousarray(f[0].data).view(np.uint32).copy()

    def test_statistics_from_the_heat_map_give_the_same_stack(self):
        heat = self.run_job("analyze")
        self.assertEqual(heat["status"], "complete", heat.get("error"))
        saved = json.loads(Path(heat["analysis_file"]).read_text())
        stats = saved["analysis"]["statistics"]
        self.assertEqual(saved["analysis"]["statistics_backend"], "gpu")
        self.assertEqual(sorted(map(int, stats)), list(range(heat["input_frames"])))
        # without them in the heat map, the stack job measures them itself, as before
        bare = Path(heat["analysis_file"]).with_name("without_statistics.json")
        for key in ("statistics", "statistics_backend"):
            del saved["analysis"][key]
        bare.write_text(json.dumps(saved))
        measured = self.stack(heat, str(bare))
        with patch.object(gpucoadd.GPUBackend, "stats", side_effect=AssertionError("measured again")):
            reused = self.stack(heat)
        np.testing.assert_array_equal(reused, measured)

    def test_registration_errors_still_end_the_heat_map(self):
        with patch("gpucoadd.register", side_effect=RuntimeError("registration broke")):
            heat = self.run_job("analyze")
        self.assertEqual(heat["status"], "error")
        self.assertIn("registration broke", heat["error"])

    def test_a_statistics_error_waits_for_registration(self):
        finished = []
        original = gpucoadd.register

        def register(*a, **k):
            result = original(*a, **k)
            finished.append(True)
            return result

        with patch("gpucoadd.register", register), \
                patch("gpucoadd.validate_light_statistics", side_effect=ValueError("unusable light")):
            heat = self.run_job("analyze")
        self.assertEqual(heat["status"], "error")
        self.assertIn("unusable light", heat["error"])
        self.assertEqual(finished, [True], "registration finished before the error was raised")


if __name__ == "__main__":
    unittest.main()
