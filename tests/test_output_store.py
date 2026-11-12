"""Output quality, bounded allocation, disk checks, corruption and interrupted saves."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_plan import ram_room
from output_store import OutputWorkspace, check_space, coverage_counts, GIB
from fits_output import write_fits, coverage_chunks
from worker.bridge import save_image_outputs, output_settings, stack_pixels, Cancelled


class SafeOutputTests(unittest.TestCase):
    def test_exact_readback_matches_hashed_float_and_unsigned_fits_bytes(self):
        patterns = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 1, 0x3F800000, 0xFFFFFFFF], np.uint32)
        with tempfile.TemporaryDirectory() as folder:
            for values in (patterns.view(np.float32).reshape(2, 4), patterns.reshape(2, 4)):
                baseline, exact = Path(folder) / "hash.fits", Path(folder) / "exact.fits"
                shape = values.shape
                chunks = lambda: [values[::-1, ::-1]]
                write_fits(baseline, fits.Header(), shape, values.dtype, chunks())
                with patch("fits_output.hashlib.sha256", side_effect=AssertionError("exact path hashed pixels")):
                    write_fits(exact, fits.Header(), shape, values.dtype, chunks(),
                               verify_buffer=np.empty(values.nbytes + 7, np.uint8))
                self.assertEqual(baseline.read_bytes(), exact.read_bytes())

    def test_exact_readback_rejects_corruption_and_keeps_previous_result(self):
        import os

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "light.fits"
            path.write_bytes(b"previous result")
            original = os.fsync
            for offset in (2880, 20):
                def corrupt(fd):
                    os.lseek(fd, offset, os.SEEK_SET)
                    os.write(fd, b"bad!")
                    original(fd)

                with patch("fits_output.os.fsync", side_effect=corrupt), self.assertRaises(OSError):
                    write_fits(path, fits.Header(), (4, 4), np.float32, [np.ones((4, 4), np.float32)],
                               verify_buffer=np.empty(64, np.uint8))
                self.assertEqual(path.read_bytes(), b"previous result")
                self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_exact_buffer_validation_and_cancel_preserve_previous_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "light.fits"
            path.write_bytes(b"previous result")
            readonly = np.empty(64, np.uint8)
            readonly.flags.writeable = False
            for buffer in (np.empty(63, np.uint8), np.empty(64, np.float32),
                           np.empty(128, np.uint8)[::2], readonly):
                with self.assertRaises(ValueError):
                    write_fits(path, fits.Header(), (4, 4), np.float32, [np.ones((4, 4), np.float32)],
                               verify_buffer=buffer)
            def cancel():
                yield np.ones((2, 4), np.float32)
                raise Cancelled("Cancelled")

            with self.assertRaises(Cancelled):
                write_fits(path, fits.Header(), (4, 4), np.float32, cancel(),
                           verify_buffer=np.empty(64, np.uint8))
            self.assertEqual(path.read_bytes(), b"previous result")
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_stack_checked_against_ram_copy_when_it_fits_else_hashed(self):
        img = np.random.default_rng(72).normal(2000, 500, (37, 29, 3)).astype(np.float32)
        img[0, :2] = np.nan  # uncovered; no clipped sample, so one write
        options = output_settings({})
        saved = {}
        with tempfile.TemporaryDirectory() as folder:
            for room in (True, False):
                buffers = []
                original = write_fits

                def recording(*a, **k):
                    buffers.append(k.get("verify_buffer"))
                    return original(*a, **k)

                with patch("resource_guard.check_ram", return_value=room), patch("fits_output.write_fits", recording):
                    files = save_image_outputs(Path(folder) / str(room), img, None, fits.Header(), "s", 1, options)
                self.assertEqual(len(buffers), 1)
                self.assertEqual(buffers[0] is not None, room)
                if room:
                    self.assertEqual(buffers[0].nbytes, img.nbytes)
                saved[room] = Path(files["stack_fits"]).read_bytes()
        self.assertEqual(saved[True], saved[False])

    def two_pass_stack(self, path, img, offset):
        """The stack file as written before clipping was counted in the writing pass."""
        from fits_output import clipping_counts, stack_chunks
        from worker.bridge import ADU_SCALE

        low, high = clipping_counts(img, offset, stack_pixels)
        header = fits.Header()
        header["ROWORDER"] = ("BOTTOM-UP", "FITS standard: first stored row is the bottom")
        header["PRODTYPE"] = "STACK"
        header["ADUSCALE"] = (ADU_SCALE, "ADU = value x ADUSCALE - ADUOFFS")
        header["ADUOFFS"] = (float(offset), "camera offset of the reference light, ADU")
        header["HISTORY"] = "Pixels: (ADU + ADUOFFS) / ADUSCALE in 0..1; uncovered pixels are 0"
        if low or high:
            header["HISTORY"] = f"Clipped samples: {low} low, {high} high"
        write_fits(path, header, (3, *img.shape[:2]), np.float32, stack_chunks(img[::-1], offset, stack_pixels))
        return path.read_bytes(), (low, high)

    def test_stack_converted_once_and_clipping_named_as_before(self):
        from fits_output import row_blocks

        rng = np.random.default_rng(73)
        clean = rng.normal(2000, 300, (61, 47, 3)).astype(np.float32)
        clean[0, :2] = np.nan
        clipped = clean.copy()
        clipped[3, 4] = [-5000, 2000, 400000]  # one sample low, one high
        clipped[9, 9, 1] = 300000
        options = output_settings({})
        blocks = 3 * len(list(row_blocks(*clean.shape[:2])))
        with tempfile.TemporaryDirectory() as folder:
            for name, img, expect in (("clean", clean, (0, 0)), ("clipped", img := clipped, (1, 2))):
                expected, counts = self.two_pass_stack(Path(folder) / f"{name}_expected.fits", img, 206.0)
                self.assertEqual(counts, expect)
                out, clipping, published = Path(folder) / name, {}, []
                with patch("worker.bridge.stack_pixels", wraps=stack_pixels) as convert:
                    files = save_image_outputs(out, img, None, fits.Header(), "s", 1, options, offset=206.0,
                                               clipping=clipping, before_publish=lambda: published.append(1))
                self.assertEqual(Path(files["stack_fits"]).read_bytes(), expected, name)
                self.assertEqual((clipping["low"], clipping["high"]), expect)
                self.assertEqual(convert.call_count, blocks if name == "clean" else 2 * blocks)
                self.assertEqual(published, [1], "the file is published once")
                self.assertEqual([p.name for p in out.iterdir()], ["s.fits"], "no unfinished file is left")

    def test_temporary_files_carry_the_job_tag_and_saved_files_are_reported(self):
        import tempfile as tf
        import output_store
        from gpucoadd import write_png
        from worker.bridge import cancel_message, write_optional

        made, published = [], []
        real = tf.mkstemp

        def mkstemp(*a, **k):
            fd, name = real(*a, **k)
            made.append(Path(name).name)
            return fd, name

        output_store.set_job("20261112_101500_000001_stack", published.append)
        try:
            with tempfile.TemporaryDirectory() as folder, patch("tempfile.mkstemp", mkstemp):
                img = np.full((20, 30, 3), 2000, np.float32)
                files = save_image_outputs(Path(folder), img, np.ones((20, 30), np.uint32), fits.Header(), "s", 1,
                                           output_settings({"coverage_fits": True}))
                write_png(Path(folder) / "p.png", np.zeros((4, 4, 3), np.uint8))
                seen = []
                write_optional(Path(folder) / "f.csv", lambda temp: (seen.append(temp.name), temp.write_text("x")))
                self.assertEqual(sorted(p.name for p in Path(folder).iterdir()),
                                 ["f.csv", "p.png", "s.fits", "s_coverage.fits"])
        finally:
            output_store.set_job()
        for name in made + seen:
            self.assertTrue(name.startswith(".") and ".20261112_101500_000001_stack." in name and
                            name.endswith(".partial"), name)
        self.assertEqual([Path(p).name for p in published], ["s_coverage.fits", "s.fits", "p.png", "f.csv"])
        self.assertEqual(Path(files["stack_fits"]).name, "s.fits")
        self.assertEqual(output_store.temp_prefix("a.fits"), ".a.fits.", "no job: the names as before")
        self.assertEqual(cancel_message([]), "Cancelled. No output file was saved; earlier results are as they were.")
        self.assertEqual(cancel_message(["C:/r/s.fits"]),
                         "Cancelled. Saved before the cancel: s.fits. Other earlier results are as they were.")

    def test_export_conversion_keeps_original_bits_without_changing_input(self):
        from tools.benchmark_light_export import original_pixels

        rng = np.random.default_rng(71)
        image = rng.normal(2000, 90000, (31, 47, 3)).astype(np.float32)
        image[0, :3] = [[np.nan, np.inf, -np.inf], [0, -0.0, 131070], [-2063, 65535, 300000]]
        original = image.copy()
        for offset in (0, 2063, -100):
            for view in (image, image[::-1, :, 1], image[1:, 2:, 0]):
                expected, lo, hi = original_pixels(view, offset)
                actual, actual_lo, actual_hi = stack_pixels(view, offset)
                self.assertEqual(actual.tobytes(), expected.tobytes())
                self.assertEqual((actual_lo, actual_hi), (lo, hi))
                fast, _, _ = stack_pixels(view, offset, count_clipping=False)
                self.assertEqual(fast.tobytes(), expected.tobytes())
        self.assertEqual(image.tobytes(), original.tobytes())

    def setUp(self):
        # Ample free RAM unless a test simulates pressure itself: results must not depend on what else
        # is running on the machine.
        for target in (
            "raw_storage.available_ram",
            "output_store.available_ram",
            "resource_guard.available_ram",
            "sky_store.available_ram",
        ):
            p = patch(target, return_value=64 << 30)
            p.start()
            self.addCleanup(p.stop)

    def test_streamed_stack_matches_original_scaling_bits_and_metadata(self):
        rng = np.random.default_rng(73)
        img = rng.normal(2000, 3000, (73, 91, 3)).astype(np.float32)
        img[0, :3] = [[np.nan] * 3, [0, -0.0, 300000], [65535, 90000, -3000]]
        header = fits.Header({"OBJECT": "test", "NCOMBINE": 5194, "RA": 15.1234})
        expected, lo, hi = stack_pixels(img, 2063)
        with tempfile.TemporaryDirectory() as d:
            counts = {}
            files = save_image_outputs(
                Path(d),
                img,
                np.ones(img.shape[:2], np.uint32),
                header,
                "image",
                5194,
                output_settings({}),
                offset=2063,
                clipping=counts,
            )
            got = fits.getdata(files["stack_fits"], memmap=False)
            np.testing.assert_array_equal(got, np.moveaxis(expected[::-1], 2, 0))  # FITS rows bottom-up
            self.assertEqual(got.tobytes(), np.moveaxis(expected[::-1], 2, 0).astype(">f4").tobytes())
            h = fits.getheader(files["stack_fits"])
            for k in header:
                self.assertEqual(header[k], h[k])
            self.assertEqual(counts, {"low": lo, "high": hi})
            self.assertEqual([p.name for p in Path(d).iterdir()], ["image.fits"])

    def test_uint32_coverage_keeps_unsigned_extremes(self):
        x = np.array([[0, 5194, 2**31, 2**32 - 1]], np.uint32)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "coverage.fits"
            write_fits(p, fits.Header(), x.shape, np.uint32, coverage_chunks(x))
            np.testing.assert_array_equal(fits.getdata(p), x)

    def test_interrupt_failure_or_corruption_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "image.fits"
            p.write_bytes(b"previous valid result")

            def broken():
                yield np.zeros((1, 4), np.float32)
                raise OSError("disk full")

            with self.assertRaisesRegex(OSError, "disk full"):
                write_fits(p, fits.Header(), (4, 4), np.float32, broken())
            self.assertEqual(p.read_bytes(), b"previous valid result")
            with patch("fits_output.verify_fits", side_effect=OSError("pixel verification failed")):
                with self.assertRaisesRegex(OSError, "verification"):
                    write_fits(p, fits.Header(), (4, 4), np.float32, [np.zeros((4, 4), np.float32)])
            with self.assertRaises(Cancelled):
                write_fits(
                    p,
                    fits.Header(),
                    (4, 4),
                    np.float32,
                    [np.zeros((4, 4), np.float32)],
                    lambda: (_ for _ in ()).throw(Cancelled()),
                )
            self.assertEqual(p.read_bytes(), b"previous valid result")
            self.assertEqual([q.name for q in Path(d).iterdir()], ["image.fits"])

    def test_short_stream_and_low_space_fail_without_publication(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "image.fits"
            with self.assertRaisesRegex(OSError, "Incomplete"):
                write_fits(p, fits.Header(), (8, 8), np.float32, [np.zeros((1, 8), np.float32)])
            with patch("output_store.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError, "free disk space"):
                    write_fits(p, fits.Header(), (8, 8), np.float32, [np.zeros((8, 8), np.float32)])
            self.assertFalse(p.exists())

    def test_ram_and_mapped_canvas_zeroing_counts_and_cleanup(self):
        cov = np.arange(80, dtype=np.uint32).reshape(8, 10)
        expected = {"covered_pixels": 79, "half_coverage_pixels": 60, "full_coverage_pixels": 1}
        with tempfile.TemporaryDirectory() as d:
            for mode in ("ram", "disk"):
                with OutputWorkspace(d, Path(d) / "results", {}, mode=mode) as work:
                    work.plan(10, 8, True)
                    a = work.allocate("coverage", cov.shape, np.uint32, zero=True)
                    self.assertEqual(np.count_nonzero(a), 0)
                    a[:] = cov
                    self.assertEqual(coverage_counts(a, 40), expected)
                    np.testing.assert_array_equal(work.float_coverage(a), cov.astype(np.float32))
                    directory = work.directory
                if directory:
                    self.assertFalse(directory.exists())

    def test_output_budget_accounts_for_same_disk_and_limits_previews(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("output_store.shutil.disk_usage", return_value=SimpleNamespace(free=100 << 20)):
                with self.assertRaises(OSError):
                    check_space([(d, 30 << 20), (Path(d) / "other", 30 << 20)])
            with (
                patch("output_store.available_ram", return_value=4 * GIB),
                patch("output_store.shutil.disk_usage", return_value=SimpleNamespace(free=1 << 40)),
            ):
                with OutputWorkspace(d, d, {}, mode="auto") as work:
                    work.plan(20000, 20000, False)
                    self.assertEqual(work.mode, "disk")
                with OutputWorkspace(d, d, {"stack_preview": True}) as work:
                    work.plan(20000, 20000, False)  # previews that cannot fit are skipped, not fatal
                    self.assertEqual(work.skipped_previews, ["stack_preview"])

    def test_low_free_ram_without_previews_uses_disk(self):
        # 0.5 GiB free leaves 0.375 GiB for the canvas. Without PNG previews it goes to disk (a 6248 x 9693
        # heat map was once refused with a PNG-preview error although none was selected).
        with tempfile.TemporaryDirectory() as d:
            logs = []
            with (
                patch("output_store.available_ram", return_value=int(0.5 * GIB)),
                patch("output_store.shutil.disk_usage", return_value=SimpleNamespace(free=1 << 40)),
            ):
                for coverage_only in (True, False):
                    with self.subTest(coverage_only=coverage_only), OutputWorkspace(d, d, {}, log=logs.append) as work:
                        work.plan(6248, 9693, coverage_only)
                        self.assertEqual(work.mode, "disk")
                        self.assertEqual(work.info["ram_budget_bytes"], int(0.5 * GIB) - (128 << 20))
                self.assertIn("0.50 GiB of RAM free", logs[-1])
                with OutputWorkspace(d, d, {"coverage_png": True}) as work:
                    work.plan(6248, 9693, True)
                    self.assertRegex(work.preview_shortage, r"PNG previews need .* but 0\.50 GiB of RAM is free")
                    self.assertEqual(work.mode, "disk")
                with OutputWorkspace(d, d, {}, mode="ram") as work:
                    with self.assertRaisesRegex(MemoryError, r"canvas needs .* but 0\.50 GiB of RAM is free"):
                        work.plan(6248, 9693, True)

    def test_canvas_ram_rule(self):
        # all available RAM except 128 MiB for the system (the shared rule in memory_plan)
        self.assertEqual(ram_room(100 << 20), 0)
        self.assertEqual(ram_room(int(2.96 * GIB)), int(2.96 * GIB) - (128 << 20))
        self.assertEqual(ram_room(8 * GIB), 8 * GIB - (128 << 20))
        # an M42 mosaic stack (6237 x 9718, 0.90 GiB) with 2.96 GiB free now stays in RAM
        with tempfile.TemporaryDirectory() as d, patch("output_store.available_ram", return_value=int(2.96 * GIB)):
            with OutputWorkspace(d, d, {}) as work:
                work.plan(6237, 9718, False)
                self.assertEqual(work.mode, "ram")

    def test_real_readback_detects_altered_bytes_before_publication(self):
        import os

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "image.fits"
            p.write_bytes(b"previous result")
            original = os.fsync

            def corrupt(fd):
                os.lseek(fd, 2880, os.SEEK_SET)
                os.write(fd, b"bad!")
                original(fd)

            with patch("fits_output.os.fsync", side_effect=corrupt):
                with self.assertRaisesRegex(OSError, "pixel verification"):
                    write_fits(p, fits.Header(), (4, 4), np.float32, [np.ones((4, 4), np.float32)])
            self.assertEqual(p.read_bytes(), b"previous result")
            self.assertEqual(len(list(Path(d).iterdir())), 1)

            def corrupt_header(fd):
                os.lseek(fd, 0, os.SEEK_SET)
                position = os.read(fd, 2880).find(b"nebula")
                self.assertGreater(position, 0)
                os.lseek(fd, position, os.SEEK_SET)
                os.write(fd, b"x")
                original(fd)

            with patch("fits_output.os.fsync", side_effect=corrupt_header):
                with self.assertRaisesRegex(OSError, "header bytes"):
                    write_fits(p, fits.Header({"OBJECT": "nebula"}), (4, 4), np.float32, [np.ones((4, 4), np.float32)])
            self.assertEqual(p.read_bytes(), b"previous result")

    def test_engine_ram_mapped_and_reference_images_and_heatmaps_match(self):
        import gpucoadd as g

        raws = [np.random.default_rng(i).integers(2000, 5000, (48, 64), np.uint16) for i in range(3)]
        cache = {
            "fwhm": [3] * 3,
            "ref_idx": 0,
            "mats": [np.eye(3).tolist()] * 3,
            "ratios": [[1, 1, 1]] * 3,
            "sizes": [[64, 48]] * 3,
        }
        with tempfile.TemporaryDirectory() as d:
            for geometry in (False, True):
                be = g.CPUBackend(raws, 0, 1)
                expected, _, info = g.stack(
                    be, 32, 1, analysis_cache=cache, coverage_only=geometry, framing="union", log=lambda _: None
                )
                for mode in ("ram", "disk"):
                    with OutputWorkspace(d, Path(d) / "outputs", {}, mode=mode) as w:
                        got, _, other = g.stack(
                            be,
                            32,
                            1,
                            analysis_cache=cache,
                            coverage_only=geometry,
                            framing="union",
                            workspace=w,
                            log=lambda _: None,
                        )
                        if not geometry:
                            np.testing.assert_array_equal(got, expected)
                        np.testing.assert_array_equal(other["coverage"], info["coverage"])
                        self.assertEqual(other["canvas"], info["canvas"])


if __name__ == "__main__":
    unittest.main()
