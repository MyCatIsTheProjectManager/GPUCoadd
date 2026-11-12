"""A new sharpness limit reuses the previous heat map's star detection and alignment
(request field previous_analysis). Every reused heat map and its stack must equal a fresh one at
the same limit; Overlap matching is solved again whenever the retained lights change."""

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
from worker.bridge import alignment_part, run_request, shared_sky_cached  # noqa: E402

# star widths per light: most sharp, a few soft, so the limits below retain different lights
SIGMAS = [2.3, 2.5, 2.4, 3.4, 2.6, 2.5, 4.0, 2.4, 2.7, 3.0, 2.5, 2.6]
SETTINGS = {"debayer": "rcd", "interp": "bicubic", "norm": "spread", "weight": "none", "transform": "similarity",
            "rejection": "sigma", "select": 0, "feather": 0, "rgb_equal": False, "match_gradients": False,
            "threads": 2, "tile": 0}


class SharpnessReuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        for kind in ("single", "groups"):
            write_scene(cls.root / kind / "lights", kind, sigmas=SIGMAS)
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

    def run_job(self, kind, operation, select, session=None, **extra):
        SharpnessReuse.jobs += 1
        path = self.root / "requests" / f"{kind}_{operation}_{self.jobs}.json"
        path.parent.mkdir(exist_ok=True)
        request = {
            "operation": operation,
            "lights_dirs": [str(self.root / kind / "lights")],
            "exclude_groups": [],
            "output_dir": str(self.root / kind / f"out_{self.jobs}"),
            "work_dir": str(self.root / "work"),
            "reference": "",
            "settings": {**SETTINGS, "select": select, **extra.pop("settings", {})},
            "outputs": {},
            **extra,
        }
        path.write_text(json.dumps(request))
        with patch.dict(os.environ, {"GPUCOADD_STORAGE": "auto"}), redirect_stdout(io.StringIO()):
            run_request(path, session)
        result = json.loads(path.with_suffix(".result.json").read_text())
        self.assertEqual(result["status"], "complete", result.get("error"))
        return result

    def analysis(self, result):
        return json.loads(Path(result["analysis_file"]).read_text())

    def stack(self, kind, result, select, settings=None):
        canvas = result["canvas"]
        got = self.run_job(kind, "stack", select, analysis_file=result["analysis_file"],
                           region=[0, 0, canvas["width"], canvas["height"]], settings=settings or {})
        with fits.open(got["files"]["stack_fits"]) as f:
            return np.ascontiguousarray(f[0].data).view(np.uint32).copy()

    def reused(self, kind, previous, select, settings=None, session=None):
        """A heat map at `select` from `previous`; detection, registration and its pool must not run."""
        fail = AssertionError("star detection or registration ran again")
        with patch.object(gpucoadd.GPUBackend, "detect", side_effect=fail), \
                patch("gpucoadd.register", side_effect=fail), \
                patch("gpucoadd.start_registration_pool", side_effect=fail):
            return self.run_job(kind, "analyze", select, session, previous_analysis=previous["analysis_file"],
                                settings=settings or {})

    def assert_same_heat_map(self, got, fresh):
        a, b = self.analysis(got), self.analysis(fresh)
        for key in ("mats", "ratios", "fwhm", "ref_idx", "sizes"):
            self.assertEqual(a["analysis"][key], b["analysis"][key], key)
        self.assertEqual(a["signature"], b["signature"])
        self.assertEqual(a["canvas"], b["canvas"])
        self.assertEqual(got["retained"], fresh["retained"])
        self.assertEqual(got["sharpness_rejected"], fresh["sharpness_rejected"])

    def test_limit_changes_reuse_alignment_and_match_fresh_heat_maps(self):
        heat = {s: self.run_job("single", "analyze", s) for s in (0, 1.0, 0.5)}
        retained = {s: heat[s]["retained"] for s in heat}
        self.assertEqual(len(set(retained.values())), 3, f"limits must retain different lights: {retained}")
        for start, limit in ((0, 1.0), (1.0, 0.5), (0.5, 0), (1.0, 0)):
            with self.subTest(start=start, limit=limit):
                got = self.reused("single", heat[start], limit)
                self.assertTrue(got["reused_alignment"])
                self.assert_same_heat_map(got, heat[limit])
                np.testing.assert_array_equal(self.stack("single", got, limit), self.stack("single", heat[limit], limit))

    def test_kept_lights_serve_the_heat_map_at_a_new_limit(self):
        from worker.worker_session import WorkerSession

        session = WorkerSession()
        try:
            first = self.run_job("single", "analyze", 0, session)
            fresh = self.run_job("single", "analyze", 1.0)
            with patch.object(gpucoadd.GPUBackend, "__init__", side_effect=AssertionError("lights loaded again")):
                got = self.reused("single", first, 1.0, session=session)
            self.assert_same_heat_map(got, fresh)
        finally:
            session.release()

    def test_changed_transform_or_lights_run_a_full_analysis(self):
        base = self.run_job("single", "analyze", 0)
        with patch("gpucoadd.register", wraps=gpucoadd.register) as register:
            got = self.run_job("single", "analyze", 1.0, previous_analysis=base["analysis_file"],
                               settings={"transform": "homography"})
        self.assertEqual(register.call_count, 1)
        self.assertFalse(got["reused_alignment"])
        missing = self.run_job("single", "analyze", 1.0, previous_analysis=str(self.root / "no" / "analysis.json"))
        self.assertFalse(missing["reused_alignment"])

    def test_overlap_matching_solved_again_when_the_retained_lights_change(self):
        # two light groups: Overlap matching is solved at heat-map time
        heat = {s: self.run_job("groups", "analyze", s) for s in (0, 0.5, 100.0)}
        self.assertNotEqual(heat[0]["retained"], heat[0.5]["retained"])
        self.assertEqual(heat[0]["retained"], heat[100.0]["retained"])
        for start, limit in ((0, 0.5), (0.5, 0)):
            with self.subTest(start=start, limit=limit), \
                    patch("gpucoadd.overlap_normalization", wraps=gpucoadd.overlap_normalization) as solve:
                got = self.reused("groups", heat[start], limit)
                self.assertEqual(solve.call_count, 1)
                self.assert_same_heat_map(got, heat[limit])
                a, b = (self.analysis(r)["analysis"]["normalization"] for r in (got, heat[limit]))
                self.assertEqual(a["frames"], b["frames"])
                self.assertEqual(a["a"], b["a"])
                np.testing.assert_array_equal(self.stack("groups", got, limit), self.stack("groups", heat[limit], limit))
        # the same retained lights: the solution is kept, not solved again
        with patch("gpucoadd.overlap_normalization", wraps=gpucoadd.overlap_normalization) as solve:
            got = self.reused("groups", heat[0], 100.0)
        self.assertEqual(solve.call_count, 0)
        self.assertEqual(self.analysis(got)["analysis"]["normalization"],
                         self.analysis(heat[0])["analysis"]["normalization"])


class SignatureParts(unittest.TestCase):
    def test_alignment_part_leaves_out_the_limit_unless_it_picks_the_reference(self):
        sig = {"schema": 2, "engine": "e", "files": [], "selection": 1.5, "reference": "", "transform": "similarity"}
        other = {**sig, "selection": 0}
        with patch.dict(os.environ, {"GPUCOADD_REFERENCE": "measured"}):
            self.assertEqual(alignment_part(sig), alignment_part(other))
            self.assertNotIn("selection", alignment_part(sig))
            self.assertNotEqual(alignment_part(sig), alignment_part({**sig, "transform": "homography"}))
        with patch.dict(os.environ, {"GPUCOADD_REFERENCE": "session"}):
            self.assertNotEqual(alignment_part(sig), alignment_part(other))
            # a chosen reference does not depend on the limit
            self.assertEqual(alignment_part({**sig, "reference": "r"}), alignment_part({**other, "reference": "r"}))

    def test_cached_overlap_solution_must_cover_exactly_the_retained_lights(self):
        import overlap

        analysis = {
            "mats": [np.eye(3).tolist()] * 4,
            "fwhm": [2.0, 2.1, 2.0, 9.0],
            "ref_idx": 0,
            "normalization": {"version": overlap.VERSION, "reference": 0, "frames": [0, 1, 2, 3],
                              "a": {str(i): [1.0, 1.0, 1.0] for i in range(4)}},
        }
        self.assertTrue(shared_sky_cached(analysis, 0))
        self.assertFalse(shared_sky_cached(analysis, 1.0))  # light 3 is cut: a superset is not reused
        analysis["normalization"]["frames"] = [0, 1, 2]
        self.assertTrue(shared_sky_cached(analysis, 1.0))
        self.assertFalse(shared_sky_cached(analysis, 0))


if __name__ == "__main__":
    unittest.main()
