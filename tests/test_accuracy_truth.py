"""Accuracy against known inputs, rather than another engine implementation.

Catalog tests bypass detection; signal tests bypass calibration, debayer and warp.
These deliberately bounded tests are not an end-to-end certification.
"""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from skimage.transform import SimilarityTransform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpucoadd as g

METRICS = []


class AccuracyTruthTests(unittest.TestCase):
    def test_held_out_alignment_and_star_flux(self):
        # Withheld grid includes corners, where a small rotation/scale error grows.
        xx, yy = np.meshgrid(np.linspace(0, 2160, 9), np.linspace(0, 3840, 13))
        held = np.c_[xx.ravel(), yy.ravel()]
        for seed, scale, angle in ((81, 1.0, 0.17), (82, 1.04, -0.43), (83, 0.98, 0.72)):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                xy = rng.uniform([40, 40], [2120, 3800], (160, 2))
                truth = SimilarityTransform(scale=scale, rotation=angle, translation=(300, -240))
                flux = rng.uniform(500, 5000, (160, 3))
                ratio = np.array([0.73, 0.81, 0.66])
                source = np.c_[xy + rng.normal(0, 0.08, xy.shape), flux * ratio, np.full(160, 8000.0)]
                reference = np.c_[truth(xy) + rng.normal(0, 0.08, xy.shape), flux, np.full(160, 8000.0)]
                result = g.match_to_ref((source, reference, "similarity", 60, scale))
                self.assertIsNotNone(result)
                forward = SimilarityTransform(matrix=np.linalg.inv(result[0]))
                error = np.linalg.norm(forward(held) - truth(held), axis=1)
                flux_error = float(np.max(np.abs(result[1] / ratio - 1)))
                METRICS.append(
                    dict(
                        check="held_out_alignment",
                        seed=seed,
                        rms_px=float(np.sqrt(np.mean(error**2))),
                        max_px=float(error.max()),
                        flux_ratio_error=flux_error,
                        max_px_limit=0.15,
                        flux_ratio_error_limit=1e-5,
                    )
                )
                self.assertLess(float(error.max()), 0.15)
                self.assertLess(flux_error, 1e-5)

    def signal_recovery(self, device):
        # A broad, coloured faint structure, present in EVERY exposure. Independent
        # noise and sparse bright contamination must not erase or invent that signal.
        rng = np.random.default_rng(180)
        yy, xx = np.mgrid[:64, :80]
        template = np.exp(-((xx - 40) ** 2 / 450 + (yy - 30) ** 2 / 220))
        template += 0.25 * np.sin(xx / 9) * np.cos(yy / 11)
        amplitudes = np.array([2.0, 0.9, 0.7])
        truth = 100 + amplitudes[:, None, None] * template
        data = truth[None] + rng.normal(0, 3, (96, 3, 64, 80))
        data[rng.random(data.shape) < 0.002] += 60
        # Missing samples in one edge strip, but enough remain to measure it.
        data[:20, :, :, :5] = np.nan
        values = torch.sort(torch.tensor(data, dtype=torch.float32, device=device).reshape(96, -1), dim=0).values
        design = np.c_[np.ones(template.size), template.ravel()]
        if device == "cuda":
            import integrate_kernels as ik
        for mode in ("sigma", "winsorized", "noise"):
            with self.subTest(device=device, mode=mode):
                if mode == "noise":
                    noise = torch.full_like(values, 3.0)
                    out, kept = (
                        ik.clip_noise(values, noise, 3.0, 3, kept=True)
                        if device == "cuda"
                        else g.noise_clip_torch(torch, values, noise, 3.0, 3, kept=True)
                    )
                else:
                    out, kept = (
                        ik.clip_sorted(
                            values, 3.0, 3, kept=True, winsorize=mode == "winsorized", winsor_iters=g.WINSOR_ITERS
                        )
                        if device == "cuda"
                        else g.sigma_clip_sorted_torch(torch, values, 3.0, 3, kept=True, winsorize=mode == "winsorized")
                    )
                image = out.cpu().numpy().reshape(3, 64, 80)
                coeff = np.linalg.lstsq(design, image.reshape(3, -1).T, rcond=None)[0]
                amplitude_error = float(np.max(np.abs(coeff[1] / amplitudes - 1)))
                bias = float(np.max(np.abs(coeff[0] - 100)))
                rmse = float(np.sqrt(np.mean((image - truth) ** 2)))
                METRICS.append(
                    dict(
                        check="faint_signal",
                        device=device,
                        rejection=mode,
                        amplitude_relative_error=amplitude_error,
                        background_bias_adu=bias,
                        rmse_adu=rmse,
                        amplitude_error_limit=0.05,
                        background_bias_limit_adu=0.03,
                        rmse_limit_adu=0.38,
                    )
                )
                self.assertTrue(np.isfinite(image).all())
                self.assertTrue(bool((kept > 0).all()))
                self.assertLess(amplitude_error, 0.05)
                self.assertLess(bias, 0.03)
                self.assertLess(rmse, 0.38)

    def test_faint_signal_cpu(self):
        self.signal_recovery("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: GPU accuracy not checked")
    def test_faint_signal_cuda(self):
        self.signal_recovery("cuda")


if __name__ == "__main__":
    unittest.main()
