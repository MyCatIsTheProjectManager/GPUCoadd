"""Source changes abort safely; unmodified lights and FITS outputs stay exact."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_integrity import SourceGuard, SourceFileError
from raw_storage import DiskLight
import gpucoadd
from fits_output import write_fits
from worker.bridge import Reporter, scan_lights, execute, fingerprint


class SourceIntegrityTests(unittest.TestCase):
    def make_light(self, root, name="light.fit"):
        p = Path(root) / name
        values = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40)
        fits.writeto(p, values, fits.Header({"BAYERPAT": "GRBG"}))
        return p, values

    def touch(self, p):
        s = p.stat()
        os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns + 2_000_000_000))

    def test_unchanged_values_exact(self):
        with tempfile.TemporaryDirectory() as d:
            p, values = self.make_light(d)
            guard = SourceGuard([p])
            before = p.read_bytes()
            np.testing.assert_array_equal(list(gpucoadd.load_frames([p], source_guard=guard))[0], values)
            np.testing.assert_array_equal(DiskLight(p, values.shape).crop(), values)
            guard.validate()
            self.assertEqual(p.read_bytes(), before)

    def test_edit_truncation_deletion_and_same_size_replacement(self):
        for mode in ("edit", "truncate", "delete", "replace"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                p, _ = self.make_light(d)
                guard = SourceGuard([p])
                if mode == "edit":
                    self.touch(p)
                elif mode == "truncate":
                    with p.open("r+b") as stream:
                        stream.truncate(2880)
                elif mode == "delete":
                    p.unlink()
                else:
                    q = Path(d) / "replacement.fit"
                    q.write_bytes(p.read_bytes())
                    stamp = p.stat()
                    os.utime(q, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    os.replace(q, p)
                with self.assertRaises(SourceFileError) as error:
                    guard.validate()
                self.assertIn(str(p), str(error.exception))

    def test_change_inside_reader_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p, _ = self.make_light(d)
            guard = SourceGuard([p])

            def read(path):
                result = gpucoadd.read_light(path)
                self.touch(p)
                return result

            with self.assertRaisesRegex(SourceFileError, "changed"):
                guard.read(p, read)

    def test_read_failure_names_file(self):
        with tempfile.TemporaryDirectory() as d:
            p, _ = self.make_light(d)
            with self.assertRaisesRegex(SourceFileError, "light.fit.*corrupt"):
                SourceGuard([p]).read(p, lambda _: (_ for _ in ()).throw(ValueError("corrupt data")))

    def test_disk_crop_changed_before_and_during_read(self):
        for during in (False, True):
            with tempfile.TemporaryDirectory() as d:
                p, values = self.make_light(d)
                light = DiskLight(p, values.shape)
                original = light._crop

                def changed(*args):
                    result = original(*args)
                    self.touch(p)
                    return result

                if not during:
                    self.touch(p)
                with patch.object(light, "_crop", side_effect=changed if during else original):
                    with self.assertRaisesRegex(SourceFileError, "changed"):
                        light.crop()

    def test_selection_keeps_original_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            p, _ = self.make_light(d)
            guard = SourceGuard([p])
            self.touch(p)
            with self.assertRaises(SourceFileError):
                guard.select([p])

    def test_header_change_identifies_file(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            p, _ = self.make_light(d)
            reporter = Reporter(Path(d), Path(d) / "cancel", save_log=False)
            original = fits.getheader

            def changed(path):
                result = original(path)
                self.touch(p)
                return result

            with patch.object(fits, "getheader", side_effect=changed):
                with self.assertRaisesRegex(SourceFileError, "light.fit"):
                    scan_lights({"lights_dirs": [d]}, reporter)

    def test_periodic_check_also_covers_unloaded_lights(self):
        with tempfile.TemporaryDirectory() as d:
            p, _ = self.make_light(d)
            reporter = Reporter(Path(d), Path(d) / "cancel", save_log=False)
            reporter.source_guard = SourceGuard([p])
            self.touch(p)
            with self.assertRaisesRegex(SourceFileError, "changed"):
                reporter.check_cancel()

    def test_changed_light_since_heat_map_names_file_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            p, _ = self.make_light(d)
            settings = {"select": 0}
            cache = Path(d) / "analysis.json"
            cache.write_text(json.dumps({"signature": fingerprint([p], settings, "")}))
            self.touch(p)
            reporter = Reporter(Path(d), Path(d) / "cancel", save_log=False)
            with self.assertRaisesRegex(SourceFileError, "changed since the heat map.*light.fit"):
                execute(
                    {
                        "operation": "stack",
                        "lights_dirs": [d],
                        "output_dir": str(Path(d) / "outputs"),
                        "settings": settings,
                        "analysis_file": str(cache),
                        "region": [0, 0, 16, 16],
                    },
                    reporter,
                    Path(d),
                    None,
                )

    def test_change_during_export_prevents_final_publication(self):
        with tempfile.TemporaryDirectory() as d:
            p, _ = self.make_light(d)
            guard = SourceGuard([p])
            out = Path(d) / "stack.fits"
            out.write_bytes(b"existing verified result")
            values = np.arange(20, dtype=np.float32).reshape(4, 5)

            def chunks():
                yield values
                self.touch(p)

            with self.assertRaisesRegex(SourceFileError, "changed"):
                write_fits(out, fits.Header(), values.shape, values.dtype, chunks(), before_publish=guard.validate)
            self.assertEqual(out.read_bytes(), b"existing verified result")
            self.assertFalse(list(Path(d).glob("*.partial")))


if __name__ == "__main__":
    unittest.main()
