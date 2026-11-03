"""Batched triangle preparation (registration_fast) gives astroalign's exact invariants and order."""

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import astroalign as aa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import registration_fast as fast  # noqa: E402


class TrianglePreparationTests(unittest.TestCase):
    def check_points(self, points):
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = aa._generate_invariants(points)
            actual = fast.generate_invariants(points)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)

    def test_exact_values_ordering_and_ties(self):
        rng = np.random.default_rng(1729)
        for dtype in (np.float32, np.float64, np.int64):
            for n in (3, 4, 5, 15, 60, 120):
                with self.subTest(dtype=dtype, n=n):
                    self.check_points(rng.uniform(0, 2000, (n, 2)).astype(dtype))
        yy, xx = np.mgrid[:6, :6]
        self.check_points(np.c_[xx.ravel(), yy.ravel()])  # a grid: many equal distances
        self.check_points(np.array([[0, 0], [0, 0], [1, 0], [0, 1], [1, 1]], float))  # duplicate point
        self.check_points(np.c_[np.arange(30), np.arange(30) * 1e-10])  # nearly collinear

    def test_matching_unchanged_on_star_lists(self):
        rng = np.random.default_rng(7)
        ref = rng.uniform(0, 1900, (60, 2))
        c, s = np.cos(0.3), np.sin(0.3)
        src = (ref - 950) @ np.array([[c, -s], [s, c]]).T * 1.04 + 950 + (12.5, -7.25)
        src = src[rng.permutation(60)]
        src[:5] = rng.uniform(0, 1900, (5, 2))  # stars not in the reference
        results = []
        for find in (aa.find_transform, fast.find_transform):
            with patch.object(np.random, "default_rng", lambda seed=None, f=np.random.default_rng: f(816)):
                t, pairs = find(src, ref, max_control_points=60)
            results.append((t.params, pairs))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        for a, b in zip(results[0][1], results[1][1]):
            np.testing.assert_array_equal(a, b)

    def test_ransac_same_result_as_astroalign(self):
        """Matching and non-matching star lists, several random draws: the same transform and
        inliers, or the same failure, as astroalign's own _ransac; and hopeless seeds skipped."""
        rng = np.random.default_rng(11)
        ref = rng.uniform(0, 1900, (60, 2))
        c, s = np.cos(-0.7), np.sin(-0.7)
        good = (ref - 950) @ np.array([[c, -s], [s, c]]).T * 0.97 + 950 + (40, 3) + rng.normal(0, 0.3, (60, 2))
        good = good[rng.permutation(60)]
        good[:20] = rng.uniform(0, 1900, (20, 2))
        other = rng.uniform(0, 1900, (60, 2))  # a different field: nothing matches
        captured = []

        def capture(data, model, thresh, min_matches):
            captured.append((data, model, thresh, min_matches))
            raise aa.MaxIterError("captured")

        for src in (good, other):
            with patch.object(fast, "_ransac", capture):
                fast._adapter.cache_clear()
                with self.assertRaises(aa.MaxIterError):
                    fast.find_transform(src, ref, max_control_points=60)
        fast._adapter.cache_clear()
        for (data, model, thresh, min_matches), matches in zip(captured, (True, False)):
            possible = fast.SeedCheck(data, model, thresh, min_matches).possible(np.arange(len(data)))
            self.assertLess(possible.mean(), 0.5)  # most seeds ruled out without a Python fit
            for seed in range(5):
                outcomes = []
                for ransac in (aa._ransac, fast._ransac):
                    with patch.object(np.random, "default_rng", lambda _=None, f=np.random.default_rng, k=seed: f(k)):
                        try:
                            t, inliers = ransac(data, model, thresh, min_matches)
                            outcomes.append((t.params.tolist(), inliers.tolist()))
                        except aa.MaxIterError:
                            outcomes.append("failed")
                with self.subTest(matches=matches, seed=seed):
                    self.assertEqual(outcomes[0], outcomes[1])
                    self.assertEqual(outcomes[0] == "failed", not matches)

    def test_matching_is_repeatable(self):
        """With noisy stars, astroalign's result depends on its unseeded shuffle (M 42: 4 of 131
        lights moved by up to 0.06 px between two heat maps); the seeded order gives one answer."""
        rng = np.random.default_rng(1000)
        ref = rng.uniform(0, 1900, (60, 2))
        c, s = np.cos(0.2), np.sin(0.2)
        src = (ref - 950) @ np.array([[c, -s], [s, c]]).T + 950 + (20, -5) + rng.normal(0, 1.2, (60, 2))
        src = src[rng.permutation(60)]
        src[:15] = rng.uniform(0, 1900, (15, 2))
        varies = set()
        for seed in range(6):
            with patch.object(np.random, "default_rng", lambda _=None, f=np.random.default_rng, k=seed: f(k)):
                t, _ = aa.find_transform(src, ref, max_control_points=60)
            varies.add(t.params.tobytes())
        self.assertGreater(len(varies), 1, "the case must depend on the shuffle to test anything")
        runs = [fast.find_transform(src, ref, max_control_points=60) for _ in range(4)]
        for t, pairs in runs[1:]:
            np.testing.assert_array_equal(t.params, runs[0][0].params)
            for a, b in zip(pairs, runs[0][1]):
                np.testing.assert_array_equal(a, b)

    def test_bounded_cache_immutable_and_input_identity(self):
        fast._cached.cache_clear()
        p = np.random.default_rng(3).uniform(0, 2000, (60, 2))
        a = fast.generate_invariants(p)
        self.assertIs(a, fast.generate_invariants(p.copy()))
        self.assertFalse(a[0].flags.writeable)
        p[0, 0] += 1
        self.assertIsNot(a, fast.generate_invariants(p))
        self.assertEqual(fast._cached.cache_info().maxsize, 128)

    def test_adapter_is_private_and_preserves_mock_or_unknown_api(self):
        original = aa._generate_invariants
        adapted = fast._adapter(aa.find_transform)
        self.assertIs(aa._generate_invariants, original)
        self.assertIs(adapted.__globals__["_generate_invariants"], fast.generate_invariants)
        with patch.object(aa, "find_transform", return_value="mock"):
            self.assertEqual(fast.find_transform([], []), "mock")
        self.assertIs(fast._adapter(len), len)
        with patch.object(aa, "__version__", "unknown"):
            fast._adapter.cache_clear()
            self.assertIs(fast._adapter(aa.find_transform), aa.find_transform)
        fast._adapter.cache_clear()

    def test_threaded_generation_has_no_global_patching(self):
        points = np.random.default_rng(27).uniform(0, 1000, (60, 2))
        original = aa._generate_invariants
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(fast.generate_invariants, [points] * 8))
        self.assertIs(aa._generate_invariants, original)
        for result in results:
            np.testing.assert_array_equal(results[0][0], result[0])


if __name__ == "__main__":
    unittest.main()
