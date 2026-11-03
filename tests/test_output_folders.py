"""What a job leaves in your folders, through the worker with the real engine on small synthetic
lights: with another results folder the target folder doesn't change at all, and without one it
only gains the stack. Files are compared by content and time, in every storage layout."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_combinations import OUTPUTS, write_scene  # noqa: E402
from worker.bridge import run_request  # noqa: E402

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
LAYOUTS = ("auto", "disk")


def folder_state(folder):
    """Every file and subfolder below folder, with its content hash and modified time."""
    state = {}
    for path in Path(folder).rglob("*"):
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "folder"
        state[str(path.relative_to(folder))] = (digest, stat.st_mtime_ns)
    return state


class OutputFolderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.target = cls.root / "target"
        write_scene(cls.target / "lights", "single")

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

    def run_job(self, name, request, layout):
        path = self.root / "requests" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(request))
        env = {"GPUCOADD_STORAGE": layout, "GPUCOADD_SKY_STORAGE": layout, "GPUCOADD_CANVAS": layout}
        with patch.dict(os.environ, env), redirect_stdout(io.StringIO()):
            run_request(path)
        result = json.loads(path.with_suffix(".result.json").read_text())
        self.assertEqual(result["status"], "complete", result.get("error"))
        return result

    def stack_job(self, name, output_dir, outputs, layout):
        base = {
            "lights_dirs": [str(self.target / "lights")],
            "exclude_groups": [],
            "work_dir": str(self.root / "work"),
            "reference": "",
            "settings": SETTINGS,
            "output_dir": str(output_dir),
        }
        heat = self.run_job(f"{name}_analyze", {**base, "operation": "analyze"}, layout)
        region = [0, 0, heat["canvas"]["width"], heat["canvas"]["height"]]
        request = {**base, "operation": "stack", "analysis_file": heat["analysis_file"], "region": region}
        return self.run_job(f"{name}_stack", {**request, "outputs": outputs}, layout)

    def test_another_folder_leaves_the_target_folder_alone(self):
        for layout in LAYOUTS:
            with self.subTest(layout=layout):
                before = folder_state(self.target)
                out = self.root / f"elsewhere_{layout}"
                result = self.stack_job(f"another_{layout}", out, dict.fromkeys(OUTPUTS, True), layout)
                self.assertEqual(folder_state(self.target), before)
                self.assertEqual(Path(result["files"]["stack_fits"]).parent, out.resolve())

    def test_the_target_folder_only_gains_the_stack(self):
        for layout in LAYOUTS:
            with self.subTest(layout=layout):
                before = folder_state(self.target)
                result = self.stack_job(f"target_{layout}", self.target, dict.fromkeys(OUTPUTS, False), layout)
                after = folder_state(self.target)
                stack = Path(result["files"]["stack_fits"])
                self.assertEqual(set(after) - set(before), {stack.name})
                self.assertEqual({k: after[k] for k in before}, before)
                stack.unlink()


if __name__ == "__main__":
    unittest.main()
