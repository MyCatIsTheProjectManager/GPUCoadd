"""Mosaic-link validation and graph composition checks without real lights."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from skimage.transform import SimilarityTransform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpucoadd import match_to_ref, register, shift_match


class RegistrationTests(unittest.TestCase):
    def recorder(self):
        self.events = []
        return lambda stage, done, total: self.events.append((stage, done, total))

    def assert_single_pass_bars(self, stages):
        """Each alignment step fills its progress bar once, from 0 to its total, never going back."""
        self.assertEqual(list(dict.fromkeys(stage for stage, _, _ in self.events)), stages)
        for stage in stages:
            done = [(d, total) for s, d, total in self.events if s == stage]
            if stage != "registration":  # stack() reports direct registration's 0 before calling register()
                self.assertEqual(done[0][0], 0, stage)
            self.assertEqual(done[-1][0], done[-1][1], stage)
            self.assertEqual([d for d, _ in done], sorted(d for d, _ in done), stage)

    def test_validated_match_and_wrong_triangle(self):
        rng = np.random.default_rng(4821)
        xy = rng.uniform(100, 1900, (100, 2))
        forward = SimilarityTransform(rotation=0.17, translation=(600, -300))
        cols = np.c_[np.ones((100, 3)) * 1000, np.ones(100) * 5000]
        source = np.c_[xy, cols]
        target = np.c_[forward(xy) + rng.normal(0, 0.1, xy.shape), cols]
        result = match_to_ref((source, target, "similarity"))
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result[0], np.linalg.inv(forward.params), atol=0.06)
        self.assertGreater(result[2]["matched_stars"], 90)
        self.assertLess(result[2]["rms_px"], 0.3)
        # Even a triangle accepted by astroalign must pass independent pair validation.
        with patch("gpucoadd.aa.find_transform", return_value=(SimilarityTransform(), None)):
            self.assertIsNone(match_to_ref((source[:3], source[:3], "similarity")))

    def test_unmatchable_star_lists_are_alignment_failures(self):
        rng = np.random.default_rng(7)
        stars = lambda n: np.c_[rng.uniform(0, 2000, (n, 2)), np.ones((n, 4)) * 1000]
        ref = stars(40)
        duplicate = np.vstack([ref[:19], ref[:1]])
        nan = ref[:20].copy()
        nan[5, 0] = np.nan
        cases = {
            "no stars": np.empty((0, 6)),
            "two stars": ref[:2],
            "a repeated position": duplicate,
            "a non-finite position": nan,
            "an unrelated field": stars(40),
        }
        for matcher in ("fast", "reference"):
            env = {"GPUCOADD_REGISTRATION": "reference"} if matcher == "reference" else {}
            for name, source in cases.items():
                with self.subTest(matcher=matcher, case=name), patch.dict("os.environ", env):
                    self.assertIsNone(match_to_ref((source, ref, "similarity")))
                    self.assertIsNone(match_to_ref((ref, source, "similarity")))

    def test_matcher_faults_are_not_alignment_failures(self):
        # A bug in the matcher must stop the run with its error, not read as a light that failed.
        rng = np.random.default_rng(8)
        stars = np.c_[rng.uniform(0, 2000, (40, 2)), np.ones((40, 4)) * 1000]
        for error in (IndexError("index 40 is out of bounds"), ValueError("shapes not aligned"),
                      TypeError("unsupported operand"), RuntimeError("CUDA error: an illegal memory access")):
            with self.subTest(error=type(error).__name__), \
                 patch("registration_fast.find_transform", side_effect=error):
                with self.assertRaises(type(error)):
                    match_to_ref((stars, stars, "similarity"))

    def test_matcher_fault_stops_registration(self):
        rng = np.random.default_rng(9)
        stars = [np.c_[rng.uniform(0, 2000, (40, 2)), np.ones((40, 4)) * 1000] for _ in range(3)]
        with patch("gpucoadd.ProcessPoolExecutor", ThreadPoolExecutor), \
             patch("registration_fast.find_transform", side_effect=IndexError("matcher bug")):
            with self.assertRaisesRegex(IndexError, "matcher bug"):
                register(stars, 0, 2, "similarity")

    def test_mosaic_chain_and_disconnected_frame(self):
        stars = [np.full((12, 6), i, float) for i in (0, 1, 2, 5)]

        def pair(job):
            source, target = int(job[0][0, 0]), int(job[1][0, 0])
            if abs(source - target) > 1:
                return None
            m = np.eye(3)
            m[0, 2] = -(source - target) * 600
            return m, np.ones(3), {"matched_stars": 12, "rms_px": 0.1, "scale": 1.0}

        diagnostics = {}
        with (
            patch("gpucoadd.ProcessPoolExecutor", ThreadPoolExecutor),
            patch("gpucoadd.match_to_ref", side_effect=pair),
        ):
            mats, _ = register(stars, 0, 2, "similarity", diagnostics=diagnostics, progress=self.recorder())
        self.assert_single_pass_bars(["registration", "neighbour alignment"])
        self.assertEqual(mats[2][0, 2], -1200)
        self.assertIsNone(mats[3])
        self.assertEqual(diagnostics["direct"], 2)
        self.assertEqual(diagnostics["recovered"], 1)
        self.assertEqual(diagnostics["failed"], 1)
        self.assertEqual(diagnostics["frames"][2]["anchor"], 1)

    def test_shift_match(self):
        """Offset voting finds shifted lights (drift, dither, reframe) without a triangle search,
        and gives up on a rotated one (the caller then falls back to triangles)."""
        rng = np.random.default_rng(5)
        xy = rng.uniform(0, 1900, (200, 2))
        cols = np.c_[np.ones((200, 3)) * 100, np.ones(200) * 1000]
        for rotation, shift, expect, ok in (
            (0, (37.3, -120.6), 1.0, True),
            (0.002, (-250, 10), 1.0, True),
            (0, (12, 40), 1.04, True),
            (0.05, (20, 20), 1.0, False),
        ):
            f = SimilarityTransform(scale=expect, rotation=rotation, translation=shift)
            ref = np.c_[f(xy) + rng.normal(0, 0.1, xy.shape), cols]
            inside = (ref[:, :2] > 0).all(1) & (ref[:, :2] < 1900).all(1)
            result = shift_match(np.c_[xy, cols], ref[inside], "similarity", expect)
            with self.subTest(rotation=rotation, shift=shift):
                self.assertEqual(result is not None, ok)
                if ok:
                    np.testing.assert_allclose(result[0], np.linalg.inv(f.params), atol=0.05)

    def test_expected_scale_from_optics(self):
        """A light from another telescope (here 1.10x the reference pixel scale) matches only
        when the expected scale is given; the gate is +-5% around it."""
        rng = np.random.default_rng(7)
        xy = rng.uniform(100, 1900, (120, 2))
        forward = SimilarityTransform(scale=1.10, rotation=-0.4, translation=(150, 80))
        cols = np.c_[np.ones((120, 3)) * 1000, np.ones(120) * 5000]
        source, target = np.c_[xy, cols], np.c_[forward(xy) + rng.normal(0, 0.1, xy.shape), cols]
        self.assertIsNone(match_to_ref((source, target, "similarity")))
        result = match_to_ref((source, target, "similarity", 60, 1.10))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[2]["scale"], 1.10, places=2)
        np.testing.assert_allclose(result[0], np.linalg.inv(forward.params), atol=0.06)
        self.assertIsNone(match_to_ref((source, target, "similarity", 60, 1.20)))

    def test_register_passes_scale_ratios(self):
        """Expected scales are the ratio of the two lights' pixel scales, for direct matches and
        for neighbour links (anchor = another telescope's light)."""
        stars = [np.full((12, 6), i, float) for i in range(4)]
        seen = []

        def pair(job):
            source, target = int(job[0][0, 0]), int(job[1][0, 0])
            seen.append((source, target, round(job[4], 6)))
            if (source == 3) != (target == 1):  # 3 links only through 1 (other telescope)
                return None
            return np.eye(3), np.ones(3), {"matched_stars": 12, "rms_px": 0.1, "scale": job[4]}

        with (
            patch("gpucoadd.ProcessPoolExecutor", ThreadPoolExecutor),
            patch("gpucoadd.match_to_ref", side_effect=pair),
        ):
            mats, _ = register(stars, 0, 2, "similarity", pixel_scales=[2.30, 2.39, 2.30, 2.30])
        self.assertIn((1, 0, round(2.39 / 2.30, 6)), seen)
        self.assertIn((2, 0, 1.0), seen)
        self.assertIn((3, 1, round(2.30 / 2.39, 6)), seen)
        self.assertIsNotNone(mats[3])
        with self.assertRaises(ValueError):
            register(stars, 0, 2, "similarity", pixel_scales=[1, 1, 0, 1])

    def mixed_scene(self, seed=12):
        """Two S50 Pro-like lights (2160 x 3840, reference first) and four S50-like lights
        (1080 x 1920 at 1.04x the pixel scale, rotated, ~27% of the sky area): star lists and
        the true light -> reference transforms."""
        rng = np.random.default_rng(seed)
        sky = rng.uniform(-400, 4400, (6000, 2))  # star positions in reference pixels
        flux = rng.pareto(1.2, len(sky)) + 1

        def light(forward, w, h, noise=0.05):
            inv = np.linalg.inv(forward)
            p = (inv @ np.c_[sky, np.ones(len(sky))].T)[:2].T
            inside = (p[:, 0] > 10) & (p[:, 0] < w - 10) & (p[:, 1] > 10) & (p[:, 1] < h - 10)
            order = np.flatnonzero(inside)[np.argsort(-flux[inside])][:200]
            xy = p[order] + rng.normal(0, noise, (len(order), 2))
            return np.c_[xy, np.repeat(flux[order, None], 3, 1) * 100, np.full(len(order), 1000.0)]

        pro = [np.eye(3), SimilarityTransform(rotation=0.004, translation=(3, -2)).params]
        s50 = [
            SimilarityTransform(scale=260 / 250, rotation=0.9 + d, translation=(1900 + 5 * k, 600 - 3 * k)).params
            for k, d in enumerate((0, 0.003, -0.002, 0.005))
        ]
        stars = [light(f, 2160, 3840) for f in pro] + [light(f, 1080, 1920) for f in s50]
        scales = [2.30] * 2 + [2.30 * 260 / 250] * 4
        return stars, pro + s50, scales, ["pro"] * 2 + ["s50"] * 4

    def assert_aligned(self, mats, truths, lights):
        corners = np.array([[0, 0, 1], [1000, 1800, 1], [1000, 0, 1.0]]).T
        for i in lights:
            with self.subTest(light=i):
                self.assertIsNotNone(mats[i])
                np.testing.assert_allclose(mats[i] @ truths[i] @ corners, corners, atol=0.1)

    def test_small_sensor_group_links_to_large_reference(self):
        """S50 lights registered to an S50 Pro reference: through their own group's anchor, one
        windowed link, and guided refinement, every light lands within 0.1 px."""
        stars, truths, scales, groups = self.mixed_scene()
        with patch("gpucoadd.ProcessPoolExecutor", ThreadPoolExecutor):
            with patch("gpucoadd.match_via_candidates", return_value=None):  # (slow, and fails too)
                plain, _ = register(stars, 0, 4, "similarity", pixel_scales=scales)
            diagnostics = {}
            mats, ratios = register(
                stars, 0, 4, "similarity", pixel_scales=scales, groups=groups, diagnostics=diagnostics
            )
        self.assertTrue(all(m is None for m in plain[2:]), "whole-list matching is expected to fail here")
        self.assert_aligned(mats, truths, range(6))
        methods = [d["method"] for d in diagnostics["frames"]]
        self.assertEqual(methods[:2], ["reference", "direct"])
        self.assertEqual(sorted(methods[2:]), ["group link", "guided", "guided", "guided"])
        np.testing.assert_allclose(ratios[3], 1, atol=0.01)

    def test_group_anchor_without_stars(self):
        """The mid-session light of a group has no stars (cloud, lens cap): another light anchors
        the group instead of a crash, and the rest of the group still aligns."""
        stars, truths, scales, groups = self.mixed_scene()
        stars[4] = np.zeros((0, 6))  # the old anchor choice: members[len // 2]
        diagnostics = {}
        with (
            patch("gpucoadd.ProcessPoolExecutor", ThreadPoolExecutor),
            patch("gpucoadd.match_via_candidates", return_value=None),
        ):
            mats, _ = register(stars, 0, 4, "similarity", pixel_scales=scales, groups=groups, diagnostics=diagnostics)
        self.assert_aligned(mats, truths, [0, 1, 2, 3, 5])
        self.assertIsNone(mats[4])
        self.assertEqual(sum(d["method"] == "group link" for d in diagnostics["frames"]), 1)

    def test_next_group_anchor_when_the_first_link_fails(self):
        """The first anchor candidate's link to the reference fails; the next candidate links and
        the whole group aligns through it."""
        stars, truths, scales, groups = self.mixed_scene()
        import gpucoadd

        real = gpucoadd.link_group
        calls = []

        def link(ex, src, ref, *a, **k):
            calls.append(len(calls))
            return None if len(calls) == 1 else real(ex, src, ref, *a, **k)

        diagnostics = {}
        with (
            patch("gpucoadd.ProcessPoolExecutor", ThreadPoolExecutor),
            patch("gpucoadd.link_group", side_effect=link),
            patch("gpucoadd.match_via_candidates", return_value=None),
        ):
            mats, _ = register(
                stars,
                0,
                4,
                "similarity",
                pixel_scales=scales,
                groups=groups,
                diagnostics=diagnostics,
                progress=self.recorder(),
            )
        self.assert_single_pass_bars(["registration", "linking light groups"])
        self.assertEqual(len(calls), 2)
        self.assert_aligned(mats, truths, range(6))
        anchor = [d for d in diagnostics["frames"] if d["method"] == "group link"][0]
        self.assertEqual(anchor["anchors_tried"], 2)

    def test_failed_group_link_falls_back_to_neighbour_recovery(self):
        """No group link at all: the group's lights are not given up; neighbour recovery can still
        align them (here: same-size lights of the same sky labelled as another group)."""
        rng = np.random.default_rng(3)
        sky = np.c_[rng.uniform(0, 2000, (300, 2)), np.ones((300, 3)) * 100, np.full(300, 1000.0)]
        shifted = sky.copy()
        shifted[:, :2] += (7.5, -4.25)
        stars = [sky, sky.copy(), shifted, shifted.copy()]
        diagnostics = {}
        with patch("gpucoadd.ProcessPoolExecutor", ThreadPoolExecutor), patch("gpucoadd.link_group", return_value=None):
            mats, _ = register(stars, 0, 2, "similarity", groups=[0, 0, 1, 1], diagnostics=diagnostics)
        for i in (2, 3):
            with self.subTest(light=i):
                self.assertIsNotNone(mats[i])
                np.testing.assert_allclose(mats[i] @ [100, 200, 1], [107.5, 195.75, 1], atol=0.05)

    def test_unrelated_group_is_reported(self):
        """A group of a different sky: no link and no recovery; reported as a group-link failure."""
        rng = np.random.default_rng(3)
        ref = np.c_[rng.uniform(0, 2000, (150, 2)), np.ones((150, 3)) * 100, np.full(150, 1000.0)]
        other = np.c_[rng.uniform(0, 1000, (150, 2)), np.ones((150, 3)) * 100, np.full(150, 1000.0)]  # unrelated sky
        stars = [ref, ref.copy(), other, other.copy()]
        diagnostics = {}
        import gpucoadd

        few = lambda *a, **k: gpucoadd_windows(*a, **k)[:6]  # a failing search tries every window: keep it short
        gpucoadd_windows = gpucoadd.star_windows
        with patch("gpucoadd.ProcessPoolExecutor", ThreadPoolExecutor), patch("gpucoadd.star_windows", side_effect=few):
            mats, _ = register(stars, 0, 2, "similarity", groups=[0, 0, 1, 1], diagnostics=diagnostics)
        self.assertIsNotNone(mats[1])
        self.assertIsNone(mats[2])
        self.assertIsNone(mats[3])
        self.assertIn("group", diagnostics["frames"][2]["reason"])


if __name__ == "__main__":
    unittest.main()
