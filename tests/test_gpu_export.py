"""GPU formatting of light exports: byte identity with the CPU path, and publication checks."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from export_kernels import SOURCE, LightEncoder, open_encoder
from fits_output import stack_chunks, write_encoded, write_fits
from worker.bridge import ADU_SCALE, Cancelled, stack_pixels

LOW = float(np.float32(1 / ADU_SCALE))


def formatted(image, offset):
    """The FITS bytes of an image, from the CPU path."""
    pixels = stack_pixels(image, offset, count_clipping=False)[0]
    planes = np.ascontiguousarray(np.moveaxis(pixels[::-1], 2, 0), dtype=">f4")
    return np.frombuffer(planes.tobytes(), np.uint8)


class EncodedWriterTests(unittest.TestCase):
    def test_encoded_writer_matches_existing_file_bytes_and_headers(self):
        image = np.random.default_rng(1).normal(1000, 4000, (19, 23, 3)).astype(np.float32)
        image[:2] = np.nan
        header = fits.Header({"OBJECT": "test", "CHECKSUM": "obsolete", "DATASUM": "obsolete"})
        with tempfile.TemporaryDirectory() as folder:
            baseline, prototype = Path(folder) / "current.fits", Path(folder) / "gpu.fits"
            shape = (3, *image.shape[:2])
            write_fits(baseline, header, shape, np.float32, stack_chunks(image[::-1], 2063, stack_pixels))
            write_encoded(prototype, header, shape, formatted(image, 2063), lambda: None)
            self.assertEqual(baseline.read_bytes(), prototype.read_bytes())

    def test_encoded_writer_rejects_corruption_and_keeps_old_file(self):
        import os

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "light.fits"
            path.write_bytes(b"old result")
            original = os.fsync

            def corrupt(fd):
                os.lseek(fd, 2880, os.SEEK_SET)
                os.write(fd, b"bad!")
                original(fd)

            with patch("fits_output.os.fsync", side_effect=corrupt):
                with self.assertRaisesRegex(OSError, "pixel verification"):
                    write_encoded(path, fits.Header(), (3, 5, 7), formatted(np.ones((5, 7, 3), np.float32), 0), lambda: None)
            self.assertEqual(path.read_bytes(), b"old result")
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_encoded_writer_cancel_and_shape_error_preserve_old_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "light.fits"
            path.write_bytes(b"old result")
            body = formatted(np.ones((5, 7, 3), np.float32), 0)

            def cancel():
                raise Cancelled("Cancelled")

            with self.assertRaises(Cancelled):
                write_encoded(path, fits.Header(), (3, 5, 7), body, cancel)
            with self.assertRaises(ValueError):
                write_encoded(path, fits.Header(), (3, 5, 8), body, lambda: None)
            self.assertEqual(path.read_bytes(), b"old result")
            self.assertEqual(list(Path(folder).iterdir()), [path])


class GPUFormattingTests(unittest.TestCase):
    def test_fused_formatting_matches_cpu_bits_on_extremes_and_random_float_patterns(self):
        import torch
        from cuda_kernels import Module

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        module = Module(SOURCE)
        rng = np.random.default_rng(5)
        for shape in ((7, 11, 3), (32, 45, 3)):
            image = rng.integers(0, 2**32, shape, dtype=np.uint32).view(np.float32)
            image.reshape(-1)[:9] = [np.nan, np.inf, -np.inf, 0, -0.0, -2063, 65535, 131070, -1]
            source = torch.from_numpy(image).cuda()
            output = torch.empty(image.size, dtype=torch.int32, device="cuda")
            for offset in (0.0, 2063.0, 2063.5, -1048576.0):
                with self.subTest(shape=shape, offset=offset), np.errstate(all="ignore"):
                    module.launch("encode", (image.size + 255) // 256, 256,
                                  [source, output, shape[0], shape[1], offset, ADU_SCALE, LOW])
                    expected = formatted(image, offset).tobytes()
                    self.assertEqual(output.cpu().numpy().tobytes(), expected)


class EncoderTests(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        self.backend = SimpleNamespace(
            torch=torch, dev=torch.device("cuda"), VRAM_RESERVE=1 << 30, free_vram=lambda: 8 << 30
        )

    def test_smaller_light_after_a_larger_one_and_alternating_hosts(self):
        import torch

        encoder = LightEncoder(self.backend, 40 * 30, 2, ADU_SCALE, LOW)
        rng = np.random.default_rng(9)
        for slot, (height, width) in enumerate(((30, 40), (17, 23))):
            image = rng.normal(900, 400, (height, width, 3)).astype(np.float32)
            image[:3] = np.nan
            encoder.canvas(height, width).copy_(torch.from_numpy(image).cuda())
            body = encoder.encode(height, width, 2063, slot)
            self.assertEqual(body.tobytes(), formatted(image, 2063).tobytes())

    def test_integer_offset_reaches_the_kernel_as_a_float(self):
        import torch

        encoder = LightEncoder(self.backend, 12, 1, ADU_SCALE, LOW)
        image = np.full((3, 4, 3), 100.0, np.float32)
        encoder.canvas(3, 4).copy_(torch.from_numpy(image).cuda())
        self.assertEqual(encoder.encode(3, 4, 2063, 0).tobytes(), formatted(image, 2063).tobytes())

    def test_each_shortage_falls_back_to_the_cpu_path_and_says_why(self):
        messages = []
        short = SimpleNamespace(**{**vars(self.backend), "free_vram": lambda: 1 << 30})
        self.assertIsNone(open_encoder(short, 8294400, 2, ADU_SCALE, LOW, messages.append))
        self.assertIsNone(open_encoder(self.backend, 800_000_000, 2, ADU_SCALE, LOW, messages.append))
        with patch("resource_guard.check_ram", return_value=False):
            self.assertIsNone(open_encoder(self.backend, 8294400, 2, ADU_SCALE, LOW, messages.append))
        with patch("export_kernels.module", side_effect=RuntimeError("no compiler")):
            self.assertIsNone(open_encoder(self.backend, 8294400, 2, ADU_SCALE, LOW, messages.append))
        for text, reason in zip(messages, ("GPU memory", "too large", "RAM", "no compiler")):
            self.assertIn(reason, text)
            self.assertIn("on the CPU", text)


class GPUExportSceneTests(unittest.TestCase):
    """The whole export through the worker, on each side of the switch."""

    @classmethod
    def setUpClass(cls):
        from tests import test_aligned_lights

        cls.fixture_class = test_aligned_lights.AlignedLightTests
        cls.fixture_class.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.fixture_class.tearDownClass()

    def test_files_match_the_cpu_path_across_geometry_and_quality_options(self):
        import hashlib
        from worker import bridge

        fixture = self.fixture_class()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        save = bridge.save_light_copies
        cases = [
            ("single", {}, False),
            ("mosaic", {"debayer": "bilinear", "interp": "bilinear", "rgb_equal": False}, True),
            ("single", {"norm": "overlap", "match_gradients": True, "rgb_equal": False}, True),
            ("mosaic", {"norm": "overlap", "rgb_equal": False, "feather": 20.0, "rejection": "winsorized"}, False),
        ]
        for index, (scene, settings, crop) in enumerate(cases):
            with self.subTest(scene=scene, settings=settings, crop=crop):
                cpu = lambda *a, **kw: save(*a, **kw, gpu_format=False)
                with patch.object(bridge, "save_light_copies", side_effect=cpu):
                    baseline = fixture.stack_job(f"cpu_format_{index}", scene, True, settings, crop, debayered=True)
                result = fixture.stack_job(f"gpu_format_{index}", scene, True, settings, crop, debayered=True)
                self.assertFalse(result.get("warnings"))
                for key in ("aligned_lights", "debayered_lights"):
                    self.assertEqual(baseline[key]["formatting"], "cpu")
                    self.assertEqual(result[key]["formatting"], "gpu")
                    self.assertEqual(result[key]["verification"], "exact")
                    self.assertEqual(result[key]["saved"], result["contributing"])
                    expected, got = {}, {}
                    for report, digests in ((baseline, expected), (result, got)):
                        for path in Path(report[key]["folder"]).glob("*.fits"):
                            with path.open("rb") as stream:
                                digests[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
                    self.assertEqual(len(expected), result["contributing"])
                    self.assertEqual(expected, got)
