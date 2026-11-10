"""CUDA sample sorting, weight/noise association and hybrid-selection checks."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SortExperimentTests(unittest.TestCase):
    def test_values_and_frame_identity(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from sort_kernels import sort_samples

        def sort_transposed(x):  # control: torch sorts adjacent samples, including layout copies
            result = torch.sort(x.T.contiguous(), dim=1)
            return result.values.T.contiguous(), result.indices.T.contiguous()

        torch.manual_seed(51)
        for n in (1, 2, 31, 32, 33, 64, 120, 256, 344, 587, 1024):
            with self.subTest(frames=n):
                backing = torch.randn(n, 74, device="cuda")
                x = backing[:, :37]
                x[torch.rand_like(x) < 0.2] = float("nan")
                x[:, 0], x[:, 1], x[:, 2] = float("nan"), 1.0, -0.0
                x[0, 3] = float("inf")
                if n > 1:
                    x[1, 3] = -float("inf")
                ref = torch.sort(x, dim=0)
                for fn in (sort_samples, sort_transposed):
                    values, indices = fn(x)
                    torch.testing.assert_close(values, ref.values, rtol=0, atol=0, equal_nan=True)
                    torch.testing.assert_close(values, x.gather(0, indices), rtol=0, atol=0, equal_nan=True)
                    self.assertTrue(
                        torch.equal(
                            indices.sort(dim=0).values, torch.arange(n, device="cuda")[:, None].expand_as(indices)
                        )
                    )

    def test_values_only_sort(self):
        """sort_values gives exactly torch.sort's values (NaN last, infinities, ties, -0.0)."""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from sort_kernels import sort_values

        torch.manual_seed(54)
        for n in (1, 2, 31, 32, 33, 64, 120, 256, 344, 512, 513, 587, 1024):
            with self.subTest(frames=n):
                backing = torch.randn(n, 74, device="cuda")
                x = backing[:, :37]
                x[torch.rand_like(x) < 0.2] = float("nan")
                x[:, 0], x[:, 1], x[:, 2] = float("nan"), 1.0, -0.0
                x[::2, 4] = 0.0  # +0.0 and -0.0 together
                x[1::2, 4] = -0.0
                x[::3, 5:9] = 2.5  # many ties
                x[0, 3] = float("inf")
                if n > 1:
                    x[1, 3] = -float("inf")
                values = sort_values(x)
                ref = torch.sort(x, dim=0).values
                torch.testing.assert_close(values, ref, rtol=0, atol=0, equal_nan=True)

    def test_backend_values_only_path(self):
        """Equal weights, no feathering, sigma or winsorized: the values-only sort is used and the
        stack is bit-identical to the torch.sort path; with weights the indexed sort is used."""
        from unittest.mock import patch

        import numpy as np
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import gpucoadd as g
        import sort_kernels as sk

        torch.manual_seed(55)
        backend = g.GPUBackend.__new__(g.GPUBackend)
        backend.torch, backend.dev = torch, torch.device("cuda")
        backend.clip = g.gpu_clip()[0]
        for n in (31, 344, 587):
            data = 100 + torch.randn(n, 3, 6, 9, device="cuda")
            data[:, :, 0, 0] = float("nan")
            data[: n // 2, 1, 2, :] = float("nan")
            data[::3, :, 1, :] = 100.0
            for mode in ("sigma", "winsorized"):
                with self.subTest(frames=n, mode=mode):
                    backend.sort = None
                    expected, ec = backend.integrate(data, 3.0, 3, None, True, mode)
                    backend.sort = sk
                    with patch.object(sk, "sort", wraps=sk.sort) as indexed:
                        actual, ac = backend.integrate(data, 3.0, 3, None, True, mode)
                        self.assertEqual(indexed.call_count, 0)
                        np.testing.assert_array_equal(ac, ec)
                        np.testing.assert_array_equal(actual, expected)
                        backend.integrate(data, 3.0, 3, np.ones(n, np.float32), True, mode)
                        self.assertGreater(indexed.call_count, 0)

    def test_weight_noise_associations_and_rejection(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from sort_kernels import sort
        import integrate_kernels as ik

        torch.manual_seed(52)
        for n in (344, 587):
            x = torch.randn(n, 53, device="cuda")
            x[:, :3] = float("nan")
            x[::3, 3:10] = 1.0
            weights = torch.rand(n, device="cuda") + 0.2
            noise = torch.rand(n, 53, device="cuda") + 0.5
            ref = torch.sort(x, dim=0)
            values, indices = sort(x)
            for mode in ("sigma", "winsorized", "noise"):

                def reject(v, ix):
                    w = weights[ix]
                    if mode == "noise":
                        return ik.clip_noise(v, noise.gather(0, ix), 3.0, 3, w, True)
                    return ik.clip_sorted(v, 3.0, 3, w, True, mode == "winsorized")

                expected, ec = reject(ref.values, ref.indices)
                actual, ac = reject(values, indices)
                self.assertTrue(torch.equal(ec, ac))
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True)

    def test_cutoff_and_build_failure_fallback(self):
        from unittest.mock import patch
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import gpucoadd as g
        import sort_kernels as sk

        for n in (512, 513, 587):
            x = torch.randn(n, 5, device="cuda")
            with patch.object(sk, "sort_samples", wraps=sk.sort_samples) as custom:
                result = sk.sort(x)
                self.assertEqual(custom.call_count, int(n <= 512))
                torch.testing.assert_close(result.values, torch.sort(x, dim=0).values, rtol=0, atol=0)
        with patch.dict("os.environ", {"GPUCOADD_SORT": "torch"}):
            self.assertIsNone(g.gpu_sort(344)[0])
        with (
            patch.dict("os.environ", {"GPUCOADD_SORT": "warp"}),
            patch.object(sk, "prepare", side_effect=RuntimeError("test compile failure")),
        ):
            fn, reason = g.gpu_sort(344)
            self.assertIsNone(fn)
            self.assertIn("test compile failure", reason)

    def test_backend_all_rejection_modes(self):
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import gpucoadd as g
        import sort_kernels as sk

        torch.manual_seed(53)
        backend = g.GPUBackend.__new__(g.GPUBackend)
        backend.torch, backend.dev = torch, torch.device("cuda")
        backend.clip = g.gpu_clip()[0]
        rng = np.random.default_rng(53)
        for n in (344, 587):
            data = 100 + torch.randn(n, 3, 5, 7, device="cuda")
            data[:, :, 0, 0] = float("nan")
            data[::3, :, 1, :] = 100.0
            weights = rng.uniform(0.2, 2, n).astype(np.float32)
            noise = rng.uniform(0.5, 3, (n, 3)).astype(np.float32)
            for mode in ("sigma", "winsorized", "noise"):
                backend.sort = None
                expected, ec = backend.integrate(data, 3.0, 3, weights, True, mode, noise)
                backend.sort = sk
                actual, ac = backend.integrate(data, 3.0, 3, weights, True, mode, noise)
                np.testing.assert_array_equal(ac, ec)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
