"""Thread ownership, buffer lifetime and shutdown checks without a GPU."""

from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from worker.bridge import save_light_copies, Cancelled
from source_integrity import SourceFileError
from resource_guard import ResourcePressureError


class FakeEncoder:
    """Records which thread touches it and which host buffer each light takes."""

    def __init__(self):
        self.thread = threading.get_ident()
        self.slots = []

    def canvas(self, height, width):
        assert threading.get_ident() == self.thread
        return np.empty((height, width, 3), np.float32)

    def encode(self, height, width, offset, slot):
        assert threading.get_ident() == self.thread
        self.slots.append(slot)
        return np.full(height * width * 12, slot, np.uint8)


class PipelineTests(unittest.TestCase):
    def test_gpu_formatting_stays_on_main_thread_and_alternates_hosts(self):
        encoder = FakeEncoder()
        writes, warnings = [], []

        def write(path, header, shape, body, check):
            self.assertNotEqual(threading.get_ident(), encoder.thread)
            writes.append((path.name, body[0]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calls = []
            workspace = SimpleNamespace(mode="ram", allocate=lambda *a: calls.append(a))
            reporter = SimpleNamespace(
                cancel_path=root / "cancel", check_cancel=lambda: None, progress=lambda *a: None,
                message=lambda text: None, warn=lambda *a: warnings.append(a),
            )
            be = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
            info = {"ref_idx": 0, "contributing_indices": [0, 1, 2, 3], "light_render": {"width": 5, "height": 4}}
            with patch("gpucoadd.render_light", side_effect=lambda be, i, state, canvas, tile: tile), \
                 patch("fits_output.write_encoded", side_effect=write), \
                 patch("export_kernels.open_encoder", return_value=encoder), \
                 patch("worker.bridge.light_copy_header", return_value=fits.Header()):
                result = save_light_copies(
                    "aligned_lights", root, "test", be, info, fits.Header(),
                    [Path(f"light{i}.fits") for i in range(4)], [], [{}] * 4, [0] * 4, workspace, reporter,
                )
        self.assertEqual(warnings, [])
        self.assertEqual(calls, [], "GPU formatting needs no workspace buffers")
        self.assertEqual(encoder.slots, [0, 1, 0, 1])
        self.assertEqual([slot for _, slot in writes], [0, 1, 0, 1])
        self.assertEqual((result["saved"], result["formatting"], result["verification"]), (4, "gpu", "exact"))

    def run_export(self, mode="ram", failure=None):
        main_thread = threading.get_ident()
        writing = threading.Event()
        second_render = threading.Event()
        allocations, warnings, written, threads = [], [], [], []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def check():
                self.assertEqual(threading.get_ident(), main_thread)

            def allocate(name, shape, dtype):
                check()
                if mode == "verify_failure" and dtype == np.uint8:
                    raise MemoryError("verification buffer refused")
                array = np.empty(shape, dtype)
                allocations.append(array)
                if mode == "fallback" and len(allocations) == 2:
                    workspace.mode = "disk"
                return array

            def render(be, index, state, canvas, tile):
                check()
                if mode == "ram" and index == 1:
                    self.assertTrue(writing.wait(5), "writer never started")
                    second_render.set()
                    if failure == "render":
                        raise OSError("render failed")
                    if failure == "cancel":
                        (root / "cancel").touch()
                        raise Cancelled("Cancellation requested")
                    if failure == "source":
                        raise SourceFileError("light changed")
                    if failure == "resource":
                        raise ResourcePressureError("resource pressure")
                canvas[:] = index + 100
                return tile

            def write(path, header, shape, dtype, chunks, check_write, before_publish=None, verify_buffer=None):
                threads.append(threading.get_ident())
                if not written and mode == "ram":
                    writing.set()
                    self.assertTrue(second_render.wait(5), "rendering did not overlap writing")
                    check_write()
                if failure == "write":
                    raise OSError("disk full")
                data = b"".join(block.tobytes() for block in chunks)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                written.append(data)

            workspace = SimpleNamespace(
                mode="ram" if mode in ("fallback", "lowram", "verify_failure") else mode, allocate=allocate
            )
            reporter = SimpleNamespace(
                cancel_path=root / "cancel", check_cancel=check,
                progress=lambda *args: check(), warn=lambda *args: warnings.append(args),
            )
            be = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=check)))
            info = {"ref_idx": 0, "contributing_indices": [0, 1, 2, 3],
                    "light_render": {"width": 5, "height": 4}}
            with patch("gpucoadd.render_light", side_effect=render), \
                 patch("fits_output.write_fits", side_effect=write), \
                 patch("resource_guard.check_ram", return_value=mode != "lowram"), \
                 patch("worker.bridge.light_copy_header", return_value=fits.Header()):
                args = ("aligned_lights", root, "test", be, info, fits.Header(),
                        [Path(f"light{i}.fits") for i in range(4)], [], [{}] * 4, [0] * 4, workspace, reporter)
                fatal = {"cancel": Cancelled, "source": SourceFileError, "resource": ResourcePressureError}
                if failure in fatal:
                    with self.assertRaises(fatal[failure]):
                        save_light_copies(*args, gpu_format=False)
                    self.assertFalse(any(t.name.startswith("light-export") for t in threading.enumerate()))
                    return
                result = save_light_copies(*args, gpu_format=False)
            self.assertFalse(any(t.name.startswith("light-export") for t in threading.enumerate()))
            if failure:
                self.assertEqual(result["saved"], len(written))
                self.assertLessEqual(result["saved"], 1 if failure == "render" else 0)
                self.assertEqual(len(warnings), 1)
            else:
                self.assertEqual(result["saved"], 4, warnings)
                self.assertEqual(len(allocations), {"ram": 3, "fallback": 2, "disk": 1,
                                                    "lowram": 2, "verify_failure": 2}[mode])
                threaded = mode in ("ram", "lowram", "verify_failure")
                self.assertEqual(result["overlap"], threaded)
                self.assertEqual(result["verification"], "exact" if mode in ("ram", "fallback") else "sha256")
                for index, data in enumerate(written):
                    pixels = np.frombuffer(data, np.float32)
                    np.testing.assert_array_equal(pixels, np.full(60, (index + 100) / 131070, np.float32))
                self.assertEqual(all(t != main_thread for t in threads), threaded)

    def test_overlap_owns_two_buffers_and_keeps_reporter_on_main_thread(self):
        self.run_export()

    def test_disk_canvas_uses_one_buffer_on_main_thread(self):
        self.run_export(mode="disk")

    def test_allocation_fallback_drains_writer_and_continues_serially(self):
        self.run_export(mode="fallback")

    def test_low_ram_keeps_sha256_without_extra_buffer(self):
        self.run_export(mode="lowram")

    def test_verification_buffer_memory_error_keeps_sha256(self):
        self.run_export(mode="verify_failure")

    def test_writer_failure_stops_and_joins(self):
        self.run_export(failure="write")

    def test_render_failure_aborts_and_joins_writer(self):
        self.run_export(failure="render")

    def test_cancellation_aborts_writer(self):
        self.run_export(failure="cancel")

    def test_source_change_aborts_writer(self):
        self.run_export(failure="source")

    def test_resource_pressure_aborts_writer(self):
        self.run_export(failure="resource")
