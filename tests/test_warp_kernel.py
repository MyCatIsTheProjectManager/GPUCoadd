"""Fused warp kernel (warp_kernels.py): against its torch twin, against torch's grid_sample, and the
property it exists for: an output pixel gets the same value whatever tile and crop it is warped in."""

from functools import partial
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warp_kernels as wk

if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA unavailable")
rng = np.random.default_rng(0)
B, Hm, Wm = 6, 560, 560
rows, cols, x0, y0 = 512, 512, 1024.0, 2048.0
rgb = torch.tensor(rng.random((B, 3, Hm, Wm)).astype(np.float32) * 4000 + 500, device="cuda")
mats, boxes, sizes = [], [], []
for k in range(B):
    ang, s = rng.uniform(-0.02, 0.02), 1 + rng.uniform(-0.002, 0.002)
    f = np.array(
        [
            [s * np.cos(ang), -s * np.sin(ang), rng.uniform(-30, 30)],
            [s * np.sin(ang), s * np.cos(ang), rng.uniform(-30, 30)],
            [0, 0, 1.0],
        ]
    )
    mats.append(np.linalg.inv(f))
    a, c = 2048 - 20 + rng.integers(-5, 5), 1024 - 20 + rng.integers(-5, 5)
    boxes.append([a, a + Hm - rng.integers(0, 12), c, c + Wm - rng.integers(0, 12)])
    sizes.append([2160, 3840])
mats, boxes, sizes = np.array(mats), np.array(boxes, float), np.array(sizes, float)
sc = (1 + rng.random((B, 3)) * 0.1).astype(np.float32)
of = (rng.random((B, 3)) * 20 - 10).astype(np.float32)
edge = 12
SLOTS = np.arange(B) + 1


def run(fn, interp, planes=None, source=rgb, bx=boxes, origin=(x0, y0), shape=(rows, cols)):
    st = torch.full((B + 2, 3, *shape), -7.0, device="cuda")
    fn(source, mats, bx, sizes, edge, *origin, *shape, sc, of, SLOTS, st, planes=planes, interp=interp)
    return st


class WarpKernelTests(unittest.TestCase):
    def test_kernel_matches_torch_twin(self):
        """Without fused multiply-adds the kernel is the torch twin bit for bit; with them (the
        default, faster) it differs only by that rounding (here at most a few millionths of the
        data range)."""
        planes = (rng.random((B, 3, 2)) * 0.02 - 0.01).astype(np.float32)  # up to 0.01 ADU/px
        scale = float(rgb.abs().max())
        for interp in ("bicubic", "bilinear"):
            for p in (None, planes):
                with self.subTest(interp=interp, planes=p is not None):
                    ref = run(wk.warp_torch, interp, p)
                    exact, st = run(partial(wk.warp, fmad=False), interp, p), run(wk.warp, interp, p)
                    r, o = ref[1 : B + 1], st[1 : B + 1]
                    self.assertTrue(torch.equal(r.nan_to_num(-1.0), exact[1 : B + 1].nan_to_num(-1.0)))
                    self.assertTrue(torch.equal(torch.isnan(r), torch.isnan(o)))
                    fin = ~torch.isnan(r)
                    self.assertGreater(int(fin.sum()), B * 3 * rows * cols // 2)
                    self.assertLess(float((r[fin] - o[fin]).abs().max()), 5e-6 * scale)
                    self.assertTrue(bool((st[0] == -7.0).all()) and bool((st[B + 1] == -7.0).all()))  # rows untouched
        plain = run(wk.warp, "bicubic")[1 : B + 1]
        tilted = run(wk.warp, "bicubic", planes)[1 : B + 1]
        fin = ~torch.isnan(plain)
        self.assertGreater(float((tilted[fin] - plain[fin]).abs().max()), 1.0)  # the planes were applied

    def test_same_pixel_in_any_tile_and_crop(self):
        """A smaller tile at another position, read from a crop that starts elsewhere (with a narrower
        batch), gives bit-identical values for the pixels both cover."""
        sub_origin, sub_shape = (x0 + 131, y0 + 77), (128, 160)
        dy, dx = 50, 90  # the second crop starts 50 rows and 90 columns further in
        sub_boxes = boxes + np.array([dy, 0, dx, 0])
        sub_rgb = rgb[:, :, dy:, dx:].contiguous()
        planes = (rng.random((B, 3, 2)) * 0.02 - 0.01).astype(np.float32)
        for fn in (wk.warp, wk.warp_torch):
            for interp in ("bicubic", "bilinear"):
                with self.subTest(fn=fn.__name__, interp=interp):
                    whole = run(fn, interp, planes)
                    part = run(fn, interp, planes, sub_rgb, sub_boxes, sub_origin, sub_shape)
                    overlap = whole[1 : B + 1, :, 77 : 77 + 128, 131 : 131 + 160]
                    self.assertTrue(torch.equal(torch.isnan(overlap), torch.isnan(part[1 : B + 1])))
                    fin = ~torch.isnan(overlap)
                    self.assertGreater(int(fin.sum()), 0)
                    self.assertTrue(torch.equal(overlap[fin], part[1 : B + 1][fin]))

    def test_same_interpolation_as_grid_sample(self):
        """Values agree with torch's grid_sample up to its float32 coordinate rounding: it holds the
        position in float32 (a step of 6e-5 px near pixel 512), and on this test image, whose
        neighbouring pixels differ by up to 4000 ADU, that moves values by up to ~0.4 ADU; a
        different interpolation would be off by hundreds."""
        gy, gx = torch.meshgrid(
            torch.arange(rows, device="cuda", dtype=torch.float64) + y0,
            torch.arange(cols, device="cuda", dtype=torch.float64) + x0,
            indexing="ij",
        )
        mv, t = torch.from_numpy(mats).cuda(), torch.from_numpy(boxes).cuda()
        den = mv[:, 2, 0, None, None] * gx + mv[:, 2, 1, None, None] * gy + mv[:, 2, 2, None, None]
        sx = (mv[:, 0, 0, None, None] * gx + mv[:, 0, 1, None, None] * gy + mv[:, 0, 2, None, None]) / den
        sy = (mv[:, 1, 0, None, None] * gx + mv[:, 1, 1, None, None] * gy + mv[:, 1, 2, None, None]) / den
        sx, sy = sx - t[:, 2, None, None], sy - t[:, 0, None, None]
        grid = torch.stack([2 * sx / (Wm - 1) - 1, 2 * sy / (Hm - 1) - 1], -1).float()
        for interp in ("bicubic", "bilinear"):
            with self.subTest(interp=interp):
                ref = F.grid_sample(rgb, grid, mode=interp, padding_mode="zeros", align_corners=True)
                ref = ref * torch.from_numpy(sc).cuda()[:, :, None, None] + torch.from_numpy(of).cuda()[:, :, None, None]
                got = run(wk.warp, interp)[1 : B + 1]
                fin = ~torch.isnan(got)
                self.assertLess(float((got[fin] - ref[fin]).abs().max()), 0.5)


if __name__ == "__main__":
    unittest.main()
