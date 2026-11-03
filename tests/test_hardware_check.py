"""Startup hardware checks without requiring AMD or broken local installations."""

from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from worker.hardware_check import check

GIB = 2**30


def runtime(cuda="12.8", hip=None, available=True, name="NVIDIA test GPU", fail=False, free_vram=11 * 2**30):
    class Value:
        def __add__(self, other):
            return self

        def item(self):
            return 2

    def ones(*args, **kwargs):
        if fail:
            raise RuntimeError("simulated driver failure")
        return Value()

    return NS(
        __version__="test",
        version=NS(cuda=cuda, hip=hip),
        ones=ones,
        cuda=NS(
            is_available=lambda: available,
            get_device_name=lambda i: name,
            synchronize=lambda: None,
            get_device_properties=lambda i: NS(total_memory=12 * 2**30),
            mem_get_info=lambda: (free_vram, 12 * 2**30),
        ),
    )


class HardwareCheckTests(unittest.TestCase):
    def test_supported_execution(self):
        result = check(runtime(), inventory=lambda: self.fail("inventory unnecessary"))
        self.assertTrue(result["supported"])
        self.assertEqual(result["total_vram_bytes"], 12 * 2**30)

    def test_missing_torch(self):
        with patch("worker.hardware_check.importlib.import_module", side_effect=ImportError("missing")):
            self.assertEqual(check()["code"], "torch_unavailable")

    def test_hip_and_translation_rejected(self):
        self.assertEqual(check(runtime(hip="6.0"))["code"], "amd_unsupported")
        self.assertEqual(check(runtime(name="AMD Radeon"))["code"], "unsupported_runtime")

    def test_unavailable_cuda_inventory(self):
        for name in ("AMD Radeon", "Intel Graphics"):
            self.assertEqual(check(runtime(available=False), lambda: [{"Name": name}])["code"], "unsupported_gpu")
        for devices in ([], [{"Name": "NVIDIA GPU"}], [{"Name": "AMD Radeon"}, {"Name": "NVIDIA GPU"}]):
            self.assertEqual(check(runtime(cuda=None), lambda: devices)["code"], "cuda_unavailable")

    def resources(self, free_ram_gb, free_vram_gb=11, free_disk_gb=(200, 200)):
        disks = {
            "work": (free_disk_gb[0] * GIB, 500 * GIB, "C:\\"),
            "results": (free_disk_gb[1] * GIB, 2000 * GIB, "E:\\"),
        }
        return check(
            runtime(free_vram=int(free_vram_gb * GIB)),
            folders=[("work area", "work"), ("results folder", "results")],
            memory=lambda: (16 * GIB, int(free_ram_gb * GIB)),
            disk=lambda folder: disks[folder],
        )

    def test_ready_summary_reports_gpu_ram_and_disks(self):
        result = self.resources(free_ram_gb=9)
        self.assertTrue(result["supported"])
        self.assertEqual(result["warnings"], [])
        self.assertIn("GPU ready: NVIDIA test GPU, 11.0 GB free of 12.0 GB.", result["summary"])
        self.assertIn("RAM 9.0 GB free of 16.0 GB", result["summary"])
        self.assertIn("200.0 GB free on C: (work area)", result["summary"])
        self.assertIn("E: (results folder)", result["summary"])
        self.assertEqual(result["resources"]["free_ram_bytes"], 9 * GIB)
        # what to expect, from the rule jobs use: 11 GB free - 0.96 GB desktop - 3 GB working space
        self.assertIn(
            "Sets up to 7.0 GB of lights (about 455 S50 Pro or 1,822 S50 lights) stay on the GPU", result["summary"]
        )
        self.assertIn("larger sets use RAM up to 8.9 GB, then disk storage", result["summary"])

    def test_warnings_only_when_action_helps(self):
        cases = [
            (dict(free_ram_gb=3), []),  # usual on a busy 16 GB PC: lights stay on the GPU, nothing to do
            (dict(free_ram_gb=1.5), ["Only 1.5 GB of RAM is free. GPUCoadd still works"]),
            (dict(free_ram_gb=9, free_vram_gb=6), ["Other programs are using 6.0 GB of the GPU's 12.0 GB"]),
            (dict(free_ram_gb=9, free_disk_gb=(5, 200)), ["Only 5.0 GB is free on C:, which holds the work area"]),
            (dict(free_ram_gb=9, free_disk_gb=(200, 3)), ["Only 3.0 GB is free on E:, which holds the results folder"]),
            (dict(free_ram_gb=4.5, free_vram_gb=10), []),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                warnings = self.resources(**kwargs)["warnings"]
                self.assertEqual(len(warnings), len(expected), warnings)
                for w, e in zip(warnings, expected):
                    self.assertTrue(w.startswith(e), w)

    def test_one_drive_reported_once(self):
        result = check(
            runtime(),
            folders=[("work area", "a"), ("results folder", "b")],
            memory=lambda: (16 * GIB, 8 * GIB),
            disk=lambda folder: (4 * GIB, 500 * GIB, "C:\\"),
        )
        self.assertEqual(result["summary"].count("free on C:"), 1)
        self.assertEqual(len(result["warnings"]), 1)

    def test_execution_failure(self):
        result = check(runtime(fail=True))
        self.assertFalse(result["supported"])
        self.assertIn("simulated driver failure", result["error"])


if __name__ == "__main__":
    unittest.main()
