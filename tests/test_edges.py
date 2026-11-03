"""Sensor-edge handling: measured demosaic reach, and warped samples independent of batching."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpucoadd as g


def scene(rng, h, w):
    """Smooth sky with sharp stars, plus noise: exercises RCD's direction decisions."""
    yy, xx = np.mgrid[:h, :w]
    img = rng.uniform(0.02, 0.05, (h, w))
    for _ in range(int(rng.integers(5, 40))):
        y, x = rng.integers(0, h), rng.integers(0, w)
        img += rng.uniform(0.05, 0.9) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * rng.uniform(0.6, 2.5) ** 2))
    return img + rng.normal(0, 0.005, (h, w))


class EdgeTests(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        self.torch = torch

    def test_demosaic_reach_matches_constants(self):
        """Columns next to a sensor edge whose demosaiced value changes when data beyond the edge
        changes; must equal DEMOSAIC_REACH (sensor_edge relies on it)."""
        torch = self.torch
        import torch.nn.functional as F

        H, W = 64, 72
        masks = torch.from_numpy(np.stack(g.bayer_masks(H, W))).cuda()
        k_db = torch.from_numpy(np.stack([g.K_RB, g.K_G, g.K_RB])[:, None]).cuda()
        run = {
            "rcd": lambda x, ms: g.rcd_demosaic(torch, F, x, *ms),
            "bilinear": lambda x, ms: F.conv2d(
                F.pad(x[:, None] * ms[None], (1, 1, 1, 1), mode="reflect"), k_db, groups=3
            ),
        }
        rng = np.random.default_rng(1)
        reach = {k: 0 for k in run}
        for _ in range(60):
            full = torch.tensor(scene(rng, H, W), dtype=torch.float32).cuda()
            for wc in (40, 47):  # both Bayer parities of the edge column
                for kind, f in run.items():
                    ref = f(full[None, :, :wc], masks[:, :, :wc])[0]
                    other = full.clone()
                    other[:, wc:] = torch.rand(H, W - wc).cuda()
                    alt = f(other[None], masks)[0, :, :, :wc]
                    cols = torch.nonzero((ref - alt).abs().amax((0, 1)) > 0).flatten()
                    if cols.numel():
                        reach[kind] = max(reach[kind], int(wc - cols.min()))
        self.assertEqual(reach, g.DEMOSAIC_REACH)

    def test_samples_independent_of_batch(self):
        """A light's warped samples must not depend on which other crops share its GPU batch."""
        torch = self.torch
        h, w = 120, 100
        for debayer, interp in (
            ("rcd", "bicubic"),
            ("rcd", "bilinear"),
            ("bilinear", "bicubic"),
            ("bilinear", "bilinear"),
        ):
            worst = 0.0
            for seed in range(12):
                rng = np.random.default_rng(seed)
                img = scene(rng, h, w)
                yy, xx = np.mgrid[:h, :w]
                for y in rng.integers(0, h, 25):  # dense sharp stars right at the right sensor edge
                    x = rng.uniform(w - 14, w)
                    img += rng.uniform(0.2, 1.0) * np.exp(
                        -((yy - y) ** 2 + (xx - x) ** 2) / (2 * rng.uniform(0.5, 1.2) ** 2)
                    )
                raws = [(2000 + 30000 * img).clip(0, 65535).astype(np.uint16), np.full((h, w), 2000, np.uint16)]
                be = g.GPUBackend(iter(raws), 100, 2, interp, debayer)
                # light 0 is cut by the tile at its right sensor edge (small crop); light 1 is the identity (larger crop)
                ang, tx, ty = rng.uniform(-0.08, 0.08), rng.uniform(-62, -58), rng.uniform(8, 12)
                fwd = np.array([[np.cos(ang), -np.sin(ang), tx], [np.sin(ang), np.cos(ang), ty], [0, 0, 1]])
                mats = [np.linalg.inv(fwd), np.eye(3)]
                coefs = {i: (np.ones(3, np.float32), np.zeros(3, np.float32)) for i in (0, 1)}
                solo = be.alloc_stack(1, 64, 64)
                be.warp_tile(solo, [0], mats, coefs, 0, 64, 20, 84)
                pair = be.alloc_stack(2, 64, 64)
                be.warp_tile(pair, [1, 0], mats, coefs, 0, 64, 20, 84)
                a, b = solo[0], pair[1]
                with self.subTest(debayer=debayer, interp=interp, seed=seed):
                    self.assertTrue(torch.equal(torch.isnan(a), torch.isnan(b)))
                    self.assertGreater(int(torch.isfinite(a).all(0).sum()), 500)  # the sensor edge is inside the tile
                m = torch.isfinite(a)
                worst = max(worst, float(((a - b).abs() / (a.abs() + 1))[m].max()))
            with self.subTest(debayer=debayer, interp=interp):
                # only sampling-coordinate rounding remains (coordinates are normalized by the batch's widest crop)
                self.assertLess(worst, 1e-4)


if __name__ == "__main__":
    unittest.main()
