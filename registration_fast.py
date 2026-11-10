"""Exact triangle preparation for astroalign's matcher, batched, with a per-worker cache.

astroalign 2.6.2 builds its triangle invariants point by point in Python (KD-tree query, one
Counter per triangle, a norm per side, an O(n^2) duplicate scan): about 85% of matching time on
SH2-142. This builds the same arrays, value for value and in the same order, with batched numpy;
the reference's invariants are computed once per worker instead of once per light. Matching,
thresholds, RANSAC and star validation stay in astroalign and gpucoadd. Other astroalign
versions keep their own implementation (find_transform falls back to it).

Adapted from astroalign, Copyright (c) 2016 Martin Beroiz, MIT License. The licence text is in
THIRD-PARTY-NOTICES.txt.
"""

from functools import lru_cache
from itertools import combinations
from types import FunctionType

import astroalign as aa
import numpy as np
from scipy.spatial import KDTree


def generate_invariants(sources):
    """Drop-in for astroalign._generate_invariants (returns read-only arrays, cached by value)."""
    sources = np.asarray(sources)
    return _cached(sources.shape, sources.dtype.str, sources.tobytes(), aa.NUM_NEAREST_NEIGHBORS)


@lru_cache(maxsize=128)
def _cached(shape, dtype, data, neighbours_count):
    sources = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
    n = len(sources)
    knn = min(n, neighbours_count)
    if knn < 3:
        return aa._generate_invariants(sources)
    # one batched tree query gives each point's neighbours in the same order (ties included)
    neighbours = KDTree(sources).query(sources, knn)[1]
    triangles = neighbours[:, np.array(list(combinations(range(knn), 3)))].reshape(-1, 3)
    # astroalign measures each side with a scalar np.linalg.norm; measure every distinct edge
    # once the same way (same rounding), then look the sides up
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    distances = np.zeros((n, n), dtype=np.asarray(np.linalg.norm(sources[0] - sources[0])).dtype)
    for a, b in edges:
        distances[a, b] = distances[b, a] = np.linalg.norm(sources[a] - sources[b])
    sides = np.stack(
        [
            distances[triangles[:, 0], triangles[:, 1]],
            distances[triangles[:, 1], triangles[:, 2]],
            distances[triangles[:, 2], triangles[:, 0]],
        ],
        axis=1,
    )
    order = np.argsort(sides, axis=1)
    # _arrangetriplet: vertices (a, b, c) where L1 & L2, L2 & L3, L3 & L1 meet, i.e. the
    # vertices opposite L3, L1 and L2; side k joins vertices k and k + 1, opposite vertex k + 2
    opposite = np.array([2, 0, 1])
    ordered = np.take_along_axis(triangles, opposite[order[:, [2, 0, 1]]], axis=1)
    sorted_sides = np.take_along_axis(sides, order, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        features = np.stack([sorted_sides[:, 2] / sorted_sides[:, 1], sorted_sides[:, 1] / sorted_sides[:, 0]], axis=1)
    # astroalign keeps the last occurrence of each feature pair, in the original order (NaN
    # pairs never compare equal, so they are all kept, as there)
    seen, keep = set(), []
    for i in range(len(features) - 1, -1, -1):
        key = tuple(features[i])
        if key not in seen:
            seen.add(key)
            keep.append(i)
    keep.reverse()
    features, ordered = features[keep], ordered[keep]
    features.flags.writeable = ordered.flags.writeable = False
    return features, ordered


RANSAC_MARGIN_PX = 1e-6  # batched residuals vs astroalign's: rounding differs by ~1e-9 px
LOOSE_PX = 1e-3  # SeedCheck's distance form (|x|^2 + |y|^2 - 2xy) vs the direct one


class SeedCheck:
    """Whether astroalign's RANSAC step seeded by a candidate triangle pair (a row of data) could
    accept: at least min_matches other pairs whose three vertices all lie within thresh of their
    match under the seed's 3-point similarity. Batched and counted against thresh +
    RANSAC_MARGIN_PX (+ LOOSE_PX for the distance form), so a seed found impossible certainly
    fails the exact test; ill-conditioned fits count as possible (left to the exact test). Only
    pairs whose first vertex is close are examined (an index by first vertex): a failed search
    over 120 stars took 10-14 s when every seed was checked against every pair, about 3 s now."""

    def __init__(self, data, model, thresh, min_matches):
        self.idx_s, self.idx_t = data[:, :, 0], data[:, :, 1]
        self.points = np.asarray(model.source, np.float64)
        self.targets = np.asarray(model.target, np.float64)
        self.limit, self.min_matches = thresh + RANSAC_MARGIN_PX, min_matches
        # pairs indexed by their first vertex's (source, target) stars: only pairs whose first
        # vertex lands near its target under a seed's fit can support that seed
        q = len(self.targets)
        key = self.idx_s[:, 0].astype(np.int64) * q + self.idx_t[:, 0]
        self.by_first = np.argsort(key, kind="stable")
        self.first_keys = key[self.by_first]
        self.target_norms = (self.targets**2).sum(1)
        # |x|^2 + |y|^2 - 2xy rounds differently from the direct distance (about 1e-8 px^2 at
        # canvas coordinates); the looser bound keeps every seed the direct test would pass
        self.limit2 = (self.limit + LOOSE_PX) ** 2

    def fit(self, seeds):
        """Umeyama similarity of each seed's 3 point pairs (as skimage's estimate_transform
        "similarity"): (a, shift, ill-conditioned)."""
        src, dst = self.points[self.idx_s[seeds]], self.targets[self.idx_t[seeds]]
        sm, dm = src.mean(1), dst.mean(1)
        sc, dc = src - sm[:, None], dst - dm[:, None]
        cov = np.einsum("nki,nkj->nij", dc, sc) / 3
        u, s, vt = np.linalg.svd(cov)
        d = np.ones((len(seeds), 2))
        d[np.linalg.det(cov) < 0, 1] = -1
        rot = u @ (d[:, :, None] * vt)
        var = (sc**2).sum((1, 2)) / 3
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = (s * d).sum(1) / var
        a = scale[:, None, None] * rot
        shift = dm - np.einsum("nij,nj->ni", a, sm)
        ill = ~(np.isfinite(a).all((1, 2)) & np.isfinite(shift).all(1)) | (var < 1e-6) | (s[:, 1] < 1e-9 * s[:, 0])
        return a, shift, ill

    def possible(self, seeds):
        seeds = np.asarray(seeds)
        a, shift, ill = self.fit(seeds)
        # every source star mapped by each seed's fit, then which targets lie within the limit
        mapped = np.einsum("bij,mj->bmi", a, self.points) + shift[:, None]
        d2 = (mapped**2).sum(-1)[:, :, None] + self.target_norms - 2 * mapped @ self.targets.T
        close = d2 < self.limit2  # (b, m, q)
        # candidate supporting pairs: those whose first vertex is close, found through the index
        b, s, t = np.nonzero(close)
        key = s.astype(np.int64) * close.shape[2] + t
        lo = np.searchsorted(self.first_keys, key, "left")
        counts = np.searchsorted(self.first_keys, key, "right") - lo
        seed_of = np.repeat(b, counts)
        starts = np.repeat(lo - (np.cumsum(counts) - counts), counts)
        pair = self.by_first[starts + np.arange(len(seed_of))]
        fits = close[seed_of, self.idx_s[pair, 1], self.idx_t[pair, 1]]
        fits &= close[seed_of, self.idx_s[pair, 2], self.idx_t[pair, 2]]
        count = np.bincount(seed_of[fits], minlength=len(seeds))
        rows = np.arange(len(seeds))
        itself = close[rows, self.idx_s[seeds, 0], self.idx_t[seeds, 0]]  # the seed is not a test pair
        for v in (1, 2):
            itself &= close[rows, self.idx_s[seeds, v], self.idx_t[seeds, v]]
        return ill | (count - itself >= self.min_matches)


RANSAC_SEED = 0  # a fixed candidate order, so the same lights always give the same alignment


def _ransac(data, model, thresh, min_matches):
    """astroalign 2.6.2's _ransac, same draws and order for the same generator, skipping the seeds
    SeedCheck rules out (each one is a full Python fit and test in the original). Seeds are
    checked in the shuffled order in growing blocks (8, 16, ... 256): a light that matches early
    checks few, one that never matches is checked in large batches.

    astroalign shuffles the candidate triangle pairs with an unseeded generator and keeps the
    first acceptable one, so two runs could settle on slightly different star pairs for a few
    lights (M 42: 4 of 131 lights, up to 0.06 px). The shuffle is seeded here instead: any fixed
    order is as valid as a random one, and heat maps become repeatable."""
    good_fit = None
    n_data = data.shape[0]
    all_idxs = np.arange(n_data)
    np.random.default_rng(RANSAC_SEED).shuffle(all_idxs)
    check = SeedCheck(data, model, thresh, min_matches)

    block, size = 0, 8
    while block < n_data and good_fit is None:
        possible = check.possible(all_idxs[block : block + size])
        for iter_i in range(block, min(block + size, n_data)):
            if not possible[iter_i - block]:
                continue
            maybe_idxs = all_idxs[iter_i : iter_i + 1]
            test_idxs = list(all_idxs[:iter_i])
            test_idxs.extend(list(all_idxs[iter_i + 1 :]))
            test_idxs = np.array(test_idxs, dtype="i8")
            maybeinliers = data[maybe_idxs, :]
            test_points = data[test_idxs, :]
            maybemodel = model.fit(maybeinliers)
            test_err = model.get_error(test_points, maybemodel)
            also_idxs = test_idxs[test_err < thresh]
            alsoinliers = data[also_idxs, :]
            if len(alsoinliers) >= min_matches:
                good_data = np.concatenate((maybeinliers, alsoinliers))
                good_fit = model.fit(good_data)
                break
        block, size = block + size, min(2 * size, 256)

    if good_fit is None:
        raise aa.MaxIterError("List of matching triangles exhausted before an acceptable transformation was found")

    better_fit = good_fit
    for i in range(3):
        test_err = model.get_error(data, better_fit)
        better_inlier_idxs = np.arange(n_data)[test_err < thresh]
        better_data = data[better_inlier_idxs]
        better_fit = model.fit(better_data)
    return better_fit, better_inlier_idxs


@lru_cache(maxsize=1)
def _adapter(original):
    """find_transform with generate_invariants and _ransac bound in a private copy of its
    globals: astroalign itself is not patched (matching may run in threads)."""
    if (
        not compatible()
        or not isinstance(original, FunctionType)
        or not {"_generate_invariants", "_ransac"} <= set(original.__code__.co_names)
    ):
        return original
    namespace = dict(original.__globals__, _generate_invariants=generate_invariants, _ransac=_ransac)
    adapted = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    adapted.__kwdefaults__ = original.__kwdefaults__
    return adapted


def find_transform(*args, **kwargs):
    return _adapter(aa.find_transform)(*args, **kwargs)


def compatible():
    """The replacement mirrors a private astroalign function: only for the version it was checked on."""
    return getattr(aa, "__version__", None) == "2.6.2"
