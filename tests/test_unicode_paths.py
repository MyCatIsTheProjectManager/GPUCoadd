"""Paths and names in any script, and optional images that can't be saved: neither stops a job."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fits_output
import gpucoadd
from resource_guard import ResourcePressureError
from source_integrity import SourceFileError
from worker.bridge import Cancelled, filename_text, fits_text, image_identity, output_settings, save_image_outputs

# Accented Latin, Cyrillic, Greek, Chinese, Japanese, Arabic and a rare Chinese character (above U+FFFF)
NAMES = ("Čačak Đurđevo", "Größe élève", "Кассиопея", "Ωμέγα", "仙后座", "カシオペヤ", "ذات الكرسي", "sky 𠀋")


class WritePngTests(unittest.TestCase):
    def test_round_trip_in_every_script(self):
        image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        for name in NAMES:
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                folder = Path(d) / name
                folder.mkdir()
                gpucoadd.write_png(folder / f"{name}.png", image)
                encoded = np.frombuffer((folder / f"{name}.png").read_bytes(), np.uint8)
                np.testing.assert_array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), image)
                self.assertEqual(len(list(folder.iterdir())), 1, "no temporary file left")

    def test_failed_write_keeps_the_old_file_and_leaves_nothing_else(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.png"
            path.write_bytes(b"old")
            with patch("os.replace", side_effect=PermissionError("locked")), self.assertRaises(PermissionError):
                gpucoadd.write_png(path, np.zeros((4, 4, 3), np.uint8))
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual([p.name for p in Path(d).iterdir()], ["a.png"])

    def test_missing_folder_is_an_error(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(OSError):
            gpucoadd.write_png(Path(d) / "missing" / "a.png", np.zeros((4, 4, 3), np.uint8))


class NameTests(unittest.TestCase):
    def test_file_names_keep_letters_of_any_script(self):
        self.assertEqual(filename_text("Čačak Đurđevo"), "Čačak_Đurđevo")
        self.assertEqual(filename_text("Кассиопея A"), "Кассиопея_A")
        self.assertEqual(filename_text("仙后座 / NGC 7023"), "仙后座_NGC_7023")

    def test_combined_and_precomposed_accents_give_one_name(self):
        self.assertEqual(filename_text("Čačak"), filename_text("Čačak"))

    def test_plain_names_are_unchanged(self):
        for name, expected in (("NGC 7023", "NGC_7023"), ("IC_63", "IC_63"), ("M 42 (2)", "M_42_2"), ("..", "")):
            self.assertEqual(filename_text(name), expected)

    def test_fits_text_is_ascii(self):
        self.assertEqual(fits_text("Čačak_Đurđevo"), "Cacak_Durdevo")
        self.assertEqual(fits_text("Größe_Ærø"), "Grosse_AEro")
        self.assertEqual(fits_text("仙后座_NGC_7023"), "NGC_7023")
        self.assertEqual(fits_text("Кассиопея"), "Unknown_Target")
        self.assertEqual(fits_text("NGC_7023"), "NGC_7023")

    def test_target_from_the_folder_name_keeps_its_letters(self):
        paths = [Path("E:/Seestar S50/Čačak Кассиопея/lights/a.fit")]
        identity = image_identity(paths, [0], headers=[{}])
        self.assertEqual(identity["target"], "Čačak_Кассиопея")
        self.assertTrue(identity["basename"].startswith("Čačak_Кассиопея_"))

    def test_every_name_makes_a_valid_header(self):
        for name in NAMES:
            with self.subTest(name):
                hdr = fits.Header()
                hdr["OBJECT"] = fits_text(filename_text(name))
                self.assertTrue(hdr.tostring().isascii())


class OutputFailureTests(unittest.TestCase):
    def setUp(self):
        self.img = np.ones((32, 32, 3), np.float32)
        self.cov = np.ones((32, 32), np.uint32)
        self.options = output_settings({"stack_preview": True, "coverage_fits": True, "coverage_png": True})
        self.warnings = []

    def save(self, folder, warn=True):
        warn = (lambda key, exc, what: self.warnings.append((key, what))) if warn else None
        return save_image_outputs(folder, self.img, self.cov, fits.Header(), "s", 1, self.options, warn=warn)

    def test_every_output_saves_in_every_script(self):
        for name in NAMES:
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                files = self.save(Path(d) / name)
                self.assertEqual(set(files), {"stack_fits", "stack_preview", "coverage_fits", "coverage_png"})
                self.assertTrue(all(Path(p).is_file() for p in files.values()))
                self.assertEqual(self.warnings, [])

    def test_stack_preview_failure_is_a_warning_and_keeps_the_stack(self):
        with tempfile.TemporaryDirectory() as d, patch("gpucoadd.preview", side_effect=OSError("refused")):
            files = self.save(Path(d))
            self.assertIn("stack_fits", files)
            self.assertNotIn("stack_preview", files)
            self.assertTrue(Path(files["stack_fits"]).is_file())
            self.assertEqual(self.warnings, [("stack_preview", "The stack preview (PNG)")])

    def test_failures_before_the_stack_do_not_stop_it(self):
        with tempfile.TemporaryDirectory() as d, patch("gpucoadd.coverage_preview", side_effect=cv2.error("bad")):
            files = self.save(Path(d))
            self.assertIn("stack_fits", files)
            self.assertEqual(self.warnings, [("coverage_png", "The frame count heat map (PNG)")])
        stack_write = fits_output.write_fits

        def refuse_coverage(path, *args, **kwargs):
            if "_coverage" in Path(path).name:
                raise OSError("refused")
            return stack_write(path, *args, **kwargs)

        self.warnings.clear()
        with tempfile.TemporaryDirectory() as d, patch("fits_output.write_fits", refuse_coverage):
            files = self.save(Path(d))
            self.assertIn("stack_fits", files)
            self.assertEqual(self.warnings, [("coverage_fits", "The frame count map (FITS)")])

    def test_memory_shortage_in_an_extra_is_a_warning(self):
        with tempfile.TemporaryDirectory() as d, patch("gpucoadd.preview", side_effect=MemoryError("short")):
            self.assertIn("stack_fits", self.save(Path(d)))
            self.assertEqual([w[0] for w in self.warnings], ["stack_preview"])

    def test_stopping_the_job_is_not_a_warning(self):
        for error in (Cancelled("stop"), SourceFileError("changed"), ResourcePressureError("short")):
            with self.subTest(type(error).__name__):
                with tempfile.TemporaryDirectory() as d, patch("gpucoadd.coverage_preview", side_effect=error):
                    with self.assertRaises(type(error)):
                        self.save(Path(d))
                self.assertEqual(self.warnings, [])

    def test_without_a_warning_channel_the_failure_is_raised(self):
        with tempfile.TemporaryDirectory() as d, patch("gpucoadd.preview", side_effect=OSError("refused")):
            with self.assertRaises(OSError):
                self.save(Path(d), warn=False)


if __name__ == "__main__":
    unittest.main()
