"""Raw-light storage planning and bounded reads from original FITS files.

The GPU path keeps its existing tensors. RAM stores unsigned raw bits without
conversion; disk mode opens only the current crop and never modifies a light.
"""

import ctypes
import os
from pathlib import Path


GIB = 1 << 30


def available_ram():
    if os.name == "nt":

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        status = Status()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available)
        return 0  # unknown: do not assume enough RAM for an entire dataset
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


def plan(sizes, free_gpu, total_gpu, free_ram=None, mode="auto", has_paths=True):
    if mode not in ("auto", "gpu", "ram", "disk"):
        raise ValueError("GPUCOADD_STORAGE must be auto, gpu, ram or disk")
    from memory_plan import desktop_vram, gpu_room, ram_room, tile_workspace

    pixels = [int(h) * int(w) for h, w in sizes]
    raw_bytes = sum(pixels) * 2
    # The lights come first, after the desktop reserve and the working space for detection and tiles.
    workspace = tile_workspace(total_gpu, max(pixels, default=0))
    reserve = workspace + desktop_vram(total_gpu)
    gpu_budget = gpu_room(free_gpu, total_gpu, workspace)
    ram = available_ram() if free_ram is None else int(free_ram)
    ram_budget = ram_room(ram)
    selected = mode
    if mode == "auto":
        selected = "gpu" if raw_bytes <= gpu_budget else "ram" if raw_bytes <= ram_budget else "disk"
    if selected == "gpu" and raw_bytes > gpu_budget:
        raise MemoryError(
            f"Raw lights need {raw_bytes / GIB:.2f} GiB, exceeding the safe GPU raw budget "
            f"of {gpu_budget / GIB:.2f} GiB. Use automatic storage, RAM or disk mode."
        )
    if selected == "ram" and raw_bytes > ram_budget:
        raise MemoryError(
            f"Raw lights need {raw_bytes / GIB:.2f} GiB, exceeding the safe RAM budget "
            f"of {ram_budget / GIB:.2f} GiB. Use automatic or disk storage."
        )
    if selected == "disk" and not has_paths:
        raise MemoryError("This dataset needs disk-backed storage; supply the original light paths.")
    return {
        "mode": selected,
        "raw_bytes": raw_bytes,
        "gpu_raw_budget_bytes": gpu_budget,
        "gpu_workspace_reserve_bytes": reserve,
        "ram_raw_budget_bytes": ram_budget,
    }


class DiskLight:
    """A light's shape and read-only source. No persistent mapping or file handle."""

    def __init__(self, path, shape):
        from fits_input import direct_source
        from source_integrity import SourceGuard

        self.path = str(Path(path).resolve())
        self.shape = tuple(shape)
        self.source_guard = SourceGuard([self.path])
        self.simple = self.source_guard.read(self.path, direct_source)
        if self.simple is not None and self.simple[0].shape != self.shape:
            raise ValueError("Light dimensions changed: " + self.path)

    def crop(self, a=0, b=None, c=0, d=None):
        stop_b = self.shape[0] if b is None else b
        stop_d = self.shape[1] if d is None else d
        if not (0 <= a < stop_b <= self.shape[0] and 0 <= c < stop_d <= self.shape[1]):
            raise ValueError("Invalid raw-light crop")
        return self.source_guard.read(self.path, lambda _: self._crop(a, b, c, d))

    def _crop(self, a=0, b=None, c=0, d=None):
        from fits_input import read_crop, read_light

        if self.simple is None:
            return read_light(self.path, (a, b, c, d))
        return read_crop(self.path, self.simple, (a, b, c, d))
