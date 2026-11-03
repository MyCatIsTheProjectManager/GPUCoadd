"""Resource-pressure matrix: every storage decision (lights, Overlap matching data, output canvas)
on simulated machines and real dataset sizes. Automatic storage never refuses while slower
storage still fits, never fills the desktop's or the system's reserve, and places data in order of
speed benefit. No real GPU memory or RAM is used, so results do not depend on this machine."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_plan import SYSTEM_RAM, desktop_vram, gpu_room, plan_summary, ram_room, start_summary, tile_workspace
from output_store import MARGIN, OutputWorkspace
from raw_storage import GIB, plan as plan_lights
from sky_store import SkyWorkspace

S50 = (1080, 1920)
S50_PRO = (2160, 3840)

# name: (total GPU, free GPU, free RAM) in GiB
MACHINES = {
    "busy 16 GB PC, 12 GB GPU": (12, 11, 3.5),
    "6 GB GPU, 64 GB RAM": (6, 5.5, 50),
    "8 GB GPU, 8 GB RAM": (8, 7, 3),
    "24 GB GPU workstation": (24, 23, 60),
    "almost no free RAM": (12, 11, 0.8),
    "GPU used by other programs": (12, 3, 10),
}
# name: (light sizes, stack canvas width x height)
DATASETS = {
    "IC 63, 200 S50 lights": ([S50] * 200, (2428, 3986)),
    "M 42 mosaic, 131 S50 Pro lights": ([S50_PRO] * 131, (6237, 9718)),
    "SH2-142, 351 S50 Pro lights": ([S50_PRO] * 351, (2600, 4400)),
    "SH2-114, 5194 S50 lights": ([S50] * 5194, (3000, 4000)),
    "large mosaic, 300 S50 Pro lights": ([S50_PRO] * 300, (20000, 20000)),
}


def simulate(machine, dataset):
    """The three decisions of one job, in the order a job makes them, with this job's own data
    taken out of the simulated free memory as a real measurement would show it."""
    total_gpu, free_gpu, free_ram = (int(v * GIB) for v in MACHINES[machine])
    sizes, (width, height) = DATASETS[dataset]
    lights = plan_lights(sizes, free_gpu, total_gpu, free_ram)
    if lights["mode"] == "gpu":
        free_gpu -= lights["raw_bytes"]
    elif lights["mode"] == "ram":
        free_ram -= lights["raw_bytes"]
    with tempfile.TemporaryDirectory() as d:
        with (
            patch("sky_store.available_ram", return_value=free_ram),
            patch("sky_store.check_space", return_value=[]),
            patch("sky_store.torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=total_gpu)),
        ):
            sky = SkyWorkspace((len(sizes), 3, -(-height // 8), -(-width // 8)), "cuda", free_gpu, d)
            sky_mode, resident, estimated = sky.mode, sky.info["resident_gpu_requirement_bytes"], sky.estimated_bytes
        # the Overlap matching data is freed before stacking: the canvas sees the memory after the lights
        with (
            patch("output_store.available_ram", return_value=free_ram),
            patch("output_store.shutil.disk_usage", return_value=SimpleNamespace(free=1 << 42)),
        ):
            with OutputWorkspace(d, d, {}) as work:
                work.plan(width, height)
                canvas_mode, canvas_bytes = work.mode, work.canvas_bytes
    return dict(
        lights=lights,
        sky=(sky_mode, resident, estimated),
        canvas=(canvas_mode, canvas_bytes),
        after_lights=(free_gpu, free_ram),
        machine=(total_gpu, *(int(v * GIB) for v in MACHINES[machine][1:])),
    )


class MemoryPlanTests(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(ram_room(100 << 20), 0)
        self.assertEqual(ram_room(5 * GIB), 5 * GIB - (128 << 20))
        self.assertEqual(desktop_vram(12 * GIB), int(12 * GIB * 0.08))
        self.assertEqual(desktop_vram(4 * GIB), 512 << 20)
        self.assertEqual(tile_workspace(12 * GIB), 3 * GIB)
        self.assertEqual(tile_workspace(24 * GIB), 6 * GIB)
        self.assertEqual(gpu_room(2 * GIB, 12 * GIB, 3 * GIB), 0)

    def test_matrix_invariants(self):
        for machine in MACHINES:
            for dataset in DATASETS:
                with self.subTest(machine=machine, dataset=dataset):
                    r = simulate(machine, dataset)
                    total_gpu, free_gpu, free_ram = r["machine"]
                    lights = r["lights"]
                    raw = lights["raw_bytes"]
                    sizes = DATASETS[dataset][0]
                    workspace = tile_workspace(total_gpu, max(h * w for h, w in sizes))
                    # lights: the fastest place that fits, never into the reserves
                    expected = (
                        "gpu"
                        if raw <= free_gpu - desktop_vram(total_gpu) - workspace
                        else "ram"
                        if raw <= free_ram - SYSTEM_RAM
                        else "disk"
                    )
                    self.assertEqual(lights["mode"], expected)
                    if lights["mode"] == "gpu":
                        self.assertGreaterEqual(free_gpu - raw, desktop_vram(total_gpu) + workspace)
                    if lights["mode"] == "ram":
                        self.assertGreaterEqual(free_ram - raw, SYSTEM_RAM)
                    # Overlap matching data: after the lights, never into the desktop reserve or system RAM
                    gpu_after, ram_after = r["after_lights"]
                    sky_mode, resident, estimated = r["sky"]
                    if sky_mode == "gpu":
                        self.assertGreaterEqual(gpu_after - resident, desktop_vram(total_gpu))
                    elif sky_mode == "ram":
                        self.assertGreaterEqual(ram_after - estimated, SYSTEM_RAM)
                        self.assertGreater(resident, gpu_room(gpu_after, total_gpu))
                    else:
                        self.assertGreater(estimated, ram_room(ram_after))
                    # canvas: RAM when it fits next to the lights, else disk; never refused
                    canvas_mode, canvas_bytes = r["canvas"]
                    fits = canvas_bytes + MARGIN <= ram_room(ram_after)
                    self.assertEqual(canvas_mode, "ram" if fits else "disk")

    def test_examples(self):
        """The decisions explained to the user, on named machines."""
        cases = [
            ("busy 16 GB PC, 12 GB GPU", "M 42 mosaic, 131 S50 Pro lights", "gpu", "ram"),
            ("6 GB GPU, 64 GB RAM", "M 42 mosaic, 131 S50 Pro lights", "ram", "ram"),  # GPU not taken over
            ("6 GB GPU, 64 GB RAM", "SH2-114, 5194 S50 lights", "ram", "ram"),
            ("busy 16 GB PC, 12 GB GPU", "SH2-114, 5194 S50 lights", "disk", "ram"),
            ("8 GB GPU, 8 GB RAM", "large mosaic, 300 S50 Pro lights", "disk", "disk"),
            ("almost no free RAM", "M 42 mosaic, 131 S50 Pro lights", "gpu", "disk"),
            ("24 GB GPU workstation", "SH2-142, 351 S50 Pro lights", "gpu", "ram"),
        ]
        for machine, dataset, lights, canvas in cases:
            with self.subTest(machine=machine, dataset=dataset):
                r = simulate(machine, dataset)
                self.assertEqual((r["lights"]["mode"], r["canvas"][0]), (lights, canvas))

    def test_low_ram_without_previews_never_refuses(self):
        """The M 42 failure: little RAM free and no PNG preview must use disk, not stop."""
        with (
            tempfile.TemporaryDirectory() as d,
            patch("output_store.shutil.disk_usage", return_value=SimpleNamespace(free=1 << 42)),
        ):
            for free in (0, int(0.3 * GIB), int(0.8 * GIB)):
                with patch("output_store.available_ram", return_value=free), OutputWorkspace(d, d, {}) as work:
                    work.plan(6248, 9693, True)
                    work.plan(6237, 9718, False)
                    self.assertEqual(work.mode, "disk")

    def test_summaries(self):
        self.assertEqual(
            start_summary(int(11 * GIB), 12 * GIB, int(3.5 * GIB)),
            "Memory at start: 11.0 GB of 12.0 GB GPU memory and 3.5 GB of RAM free; "
            "1.0 GB of GPU memory and 128 MB of RAM stay free for other programs.",
        )
        notices = {
            "raw": {"mode": "gpu", "bytes": int(2.2 * GIB)},
            "sky": {"mode": "gpu"},
            "output": {"mode": "disk", "bytes": int(0.9 * GIB)},
        }
        self.assertEqual(
            plan_summary(notices),
            "Memory plan: lights in GPU memory (2.2 GB); Overlap matching data in GPU memory; "
            "output canvas on disk (0.9 GB).",
        )
        self.assertEqual(plan_summary({}), "")


if __name__ == "__main__":
    unittest.main()
