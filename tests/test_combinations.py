"""Pairwise combinations of the quality options and storage layouts, through the worker, on small
synthetic star fields with real star detection and registration: a single field, a three-panel
mosaic whose far panel shares no sky with the reference, and two light groups (30 s and 10 s).

Every pair of option values appears in at least one run. Each run must finish (or be refused with
a clear message when the combination is invalid), write valid FITS with the expected headers and
the chosen optional files, and give exactly the pixels of the same quality settings with the
default storage layout, automatic tiles and the full frame (a crop is compared with that region)."""

from contextlib import redirect_stdout
from itertools import combinations, product
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.bridge import run_request  # noqa: E402

H, W, N = 480, 640, 12
PANEL = 330  # mosaic panel offset: the third panel shares no sky with the first

OPTIONS = {
    "brightness": ["spread", "spread+rgb", "photometric", "overlap", "overlap+gradients"],
    "debayer": ["rcd", "bilinear"],
    "interp": ["bicubic", "bilinear"],
    "weight": ["none", "noise"],
    "transform": ["similarity", "homography"],
    "rejection": ["sigma", "winsorized", "noise"],
    "select": [0.0, 2.5],
    "feather": [0.0, 20.0],
    "lights": ["gpu", "ram", "disk"],
    "sky": ["gpu", "ram", "disk"],
    "canvas": ["ram", "disk"],
    "tile": [0, 128],
    "crop": [False, True],
    "outputs": [False, True],
}
OUTPUTS = ("stack_preview", "coverage_fits", "coverage_png", "frames_csv", "activity_log")


def pairwise(options):
    """Rows covering every pair of values of every two options (greedy, deterministic)."""
    names = list(options)
    todo = {(a, va, b, vb) for a, b in combinations(names, 2) for va in options[a] for vb in options[b]}
    rows = []
    while todo:
        # start from the first uncovered pair, then fill each option with the value covering most
        a, va, b, vb = min(todo, key=lambda p: (names.index(p[0]), names.index(p[2]), str(p[1]), str(p[3])))
        row = {a: va, b: vb}
        for name in names:
            if name in row:
                continue
            row[name] = max(
                options[name],
                key=lambda v: sum(
                    (n, row[n], name, v) in todo if names.index(n) < names.index(name) else (name, v, n, row[n]) in todo
                    for n in row
                ),
            )
        todo -= {(x, row[x], y, row[y]) for x, y in combinations(names, 2)}
        rows.append(row)
    return rows


def write_scene(folder, kind, seed=5, sigmas=None):
    """GRBG lights with stars, a faint nebula, sky gradients and noise, as Seestar FITS files.
    sigmas: each light's star width (Gaussian sigma, px); 2.6 for every light by default."""
    rng = np.random.default_rng(seed)
    width = W + (2 * PANEL if kind == "mosaic" else 0) + 40
    stars = []
    while len(stars) < 60 * width // W:
        p = rng.uniform([30, 30], [width - 30, H + 10])
        if all(np.hypot(*(p - q)) > 30 for q in stars):
            stars.append(p)
    stars, peaks = np.array(stars), rng.uniform(1500, 6000, len(stars))
    yy, xx = np.mgrid[:H, :W].astype(float)
    # mosaic: panels 1, 0, 2 in time order, so the mid-session reference sits in panel 0
    panels = [1] * 4 + [0] * 4 + [2] * 4 if kind == "mosaic" else [0] * N
    folder.mkdir(parents=True)
    for i, panel in enumerate(panels):
        angle = rng.uniform(-0.01, 0.01)
        c, s = np.cos(angle), np.sin(angle)
        x = c * xx - s * yy + 20 + panel * PANEL + rng.uniform(-8, 8)
        y = s * xx + c * yy + 20 + rng.uniform(-8, 8)
        sky = 30 * np.exp(-((x - width / 2) ** 2 / (2 * 200**2) + (y - H / 2) ** 2 / (2 * 120**2)))
        for (sx, sy), peak in zip(stars, peaks):
            near = (np.abs(x - sx) < 15) & (np.abs(y - sy) < 15)
            sigma = 2.6 if sigmas is None else sigmas[i]
            sky[near] += peak * np.exp(-((x[near] - sx) ** 2 + (y[near] - sy) ** 2) / (2 * sigma**2))
        rgb = rng.uniform(0.7, 1.0) * np.array([1.0, 0.6, 0.45])[:, None, None] * sky
        rgb += rng.uniform(600, 900, 3)[:, None, None] + 0.05 * (xx - W / 2)[None]
        cfa = np.empty((H, W))
        cfa[::2, 1::2], cfa[::2, ::2] = rgb[0, ::2, 1::2], rgb[1, ::2, ::2]
        cfa[1::2, 1::2], cfa[1::2, ::2] = rgb[1, 1::2, 1::2], rgb[2, 1::2, ::2]
        raw = np.clip(np.rint(cfa + 100 + rng.normal(0, 6, cfa.shape)), 0, 65535).astype(np.uint16)
        short = kind == "groups" and i % 2  # every other light: a second group (10 s, gain 80)
        header = fits.Header(
            {
                "TELESCOP": "S50 Pro_test",
                "INSTRUME": "Seestar S50 Pro",
                "EXPTIME": 10.0 if short else 30.0,
                "GAIN": 80 if short else 200,
                "BAYERPAT": "GRBG",
                "BIAS": 100,
                "FILTER": "IRCUT",
                "XPIXSZ": 2.9,
                "FOCALLEN": 260.0,
                "RA": 341.0,
                "DEC": 58.1,
                "OBJECT": f"Test {kind}",
                "DATE-OBS": f"2026-10-20T21:{i:02d}:00",
            }
        )
        fits.writeto(folder / f"light_{i:02d}.fit", raw, header)


def settings(row):
    norm = row["brightness"].split("+")[0]
    return {
        "debayer": row["debayer"],
        "interp": row["interp"],
        "norm": norm,
        "weight": row["weight"],
        "transform": row["transform"],
        "rejection": row["rejection"],
        "select": row["select"],
        "feather": row["feather"],
        "rgb_equal": row["brightness"] == "spread+rgb",
        "match_gradients": row["brightness"] == "overlap+gradients",
        "threads": 2,
        "tile": row["tile"],
    }


class CombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        cls.rows = pairwise(OPTIONS)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        for kind in ("single", "mosaic", "groups"):
            write_scene(cls.root / kind / "lights", kind)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        # Results must not depend on what else is running on the machine.
        for target in (
            "raw_storage.available_ram",
            "output_store.available_ram",
            "resource_guard.available_ram",
            "sky_store.available_ram",
        ):
            p = patch(target, return_value=64 << 30)
            p.start()
            self.addCleanup(p.stop)

    def run_job(self, name, request, env=None):
        path = self.root / "requests" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(request))
        layout = {"GPUCOADD_STORAGE": "auto", "GPUCOADD_SKY_STORAGE": "auto", "GPUCOADD_CANVAS": "auto", **(env or {})}
        with patch.dict(os.environ, layout), redirect_stdout(io.StringIO()):
            run_request(path)
        return json.loads(path.with_suffix(".result.json").read_text())

    def read_stack(self, result):
        with fits.open(result["files"]["stack_fits"]) as f:
            header = f[0].header
            data = f[0].data.astype(np.float32)
        self.assertEqual(header["ROWORDER"], "BOTTOM-UP")
        self.assertEqual(data.shape[0], 3)
        return np.ascontiguousarray(data[:, ::-1]), header  # rows top-down, as regions are given

    def check_scene(self, kind):
        lights = str(self.root / kind / "lights")
        timings = []
        for r, row in enumerate(self.rows):
            with self.subTest(scene=kind, row=r, **{k: str(v) for k, v in row.items()}):
                started = time.perf_counter()
                out = self.root / kind / f"out_{r}"
                base = {
                    "lights_dirs": [lights],
                    "exclude_groups": [],
                    "work_dir": str(self.root / "work"),
                    "reference": "",
                    "settings": settings(row),
                }
                heat = self.run_job(f"{kind}_{r}_analyze", {**base, "operation": "analyze", "output_dir": str(out)})
                if kind == "groups" and row["brightness"] == "spread+rgb":
                    # an invalid request (the dialog never sends it): refused with a clear message
                    self.assertEqual(heat["status"], "error")
                    self.assertIn("Mixed light groups use Overlap matching; turn RGB equalization off", heat["error"])
                    continue
                self.assertEqual(heat["status"], "complete", heat.get("error"))
                self.assertEqual(heat["alignment_failed"], 0)
                width, height = heat["canvas"]["width"], heat["canvas"]["height"]
                region = [width // 4, height // 3, width // 2, height // 3] if row["crop"] else [0, 0, width, height]
                stack = {**base, "operation": "stack", "analysis_file": heat["analysis_file"]}
                env = {
                    "GPUCOADD_STORAGE": row["lights"],
                    "GPUCOADD_SKY_STORAGE": row["sky"],
                    "GPUCOADD_CANVAS": row["canvas"],
                }
                outputs = {k: row["outputs"] for k in OUTPUTS}
                got = self.run_job(
                    f"{kind}_{r}_stack",
                    {**stack, "region": region, "outputs": outputs, "output_dir": str(out / "row")},
                    env,
                )
                self.assertEqual(got["status"], "complete", got.get("error"))
                self.assertEqual(got["warnings"], [])
                reference = self.run_job(
                    f"{kind}_{r}_reference",
                    {
                        **stack,
                        "settings": {**base["settings"], "tile": 0},
                        "region": [0, 0, width, height],
                        "outputs": {},
                        "output_dir": str(out / "reference"),
                    },
                )
                self.assertEqual(reference["status"], "complete", reference.get("error"))
                image, header = self.read_stack(got)
                full, _ = self.read_stack(reference)
                x, y, w, h = region
                np.testing.assert_array_equal(image, full[:, y : y + h, x : x + w])
                self.assertEqual(header["RGBEQUAL"], row["brightness"] == "spread+rgb")
                self.assertEqual(header["GRADMTCH"], row["brightness"] == "overlap+gradients")
                self.assertGreater(header["NCOMBINE"], 0)
                covered = image[image > 0]
                self.assertGreater(covered.size, 0)
                self.assertTrue(np.isfinite(image).all() and covered.max() <= 1)
                saved = {i["role"] for i in got["image_outputs"]}
                expected = {"stack_fits"} | ({"stack_preview", "coverage_fits", "coverage_png"} if row["outputs"] else set())
                self.assertEqual(saved, expected)
                if row["outputs"]:
                    with fits.open(got["files"]["coverage_fits"]) as f:
                        self.assertEqual(f[0].header["PRODTYPE"], "COVERAGE")
                        self.assertEqual(f[0].data.dtype.kind, "u")
                    for key in ("frames_csv", "activity_log"):
                        self.assertTrue(Path(got["files"][key]).exists(), key)
                self.assertFalse(list((self.root / "work").rglob("canvas_*")), "temporary canvas files left")
                timings.append(time.perf_counter() - started)
        return timings

    def test_pairwise_rows_cover_every_pair(self):
        names = list(OPTIONS)
        for a, b in combinations(names, 2):
            seen = {(row[a], row[b]) for row in self.rows}
            self.assertEqual(seen, set(product(OPTIONS[a], OPTIONS[b])), (a, b))
        self.assertLess(len(self.rows), 30)

    def test_single_field(self):
        self.check_scene("single")

    def test_mosaic(self):
        self.check_scene("mosaic")

    def test_two_light_groups(self):
        self.check_scene("groups")


if __name__ == "__main__":
    unittest.main()
