"""Lossless RAM/disk backing for Overlap matching arrays; computations stay on their device."""

import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from memory_plan import gpu_room, ram_room
from output_store import check_space
from raw_storage import available_ram, GIB
from resource_guard import check_ram, check_gpu, ResourcePressureError


class StoredTensor:
    def __init__(self, data, owner):
        self.data = data
        self.owner = owner
        self.device = owner.device
        self.shape = data.shape
        self.is_cuda = self.device.type == "cuda"
        self.dirty_bytes = 0

    def reshape(self, *shape):
        return StoredTensor(self.data.reshape(*shape), self.owner)

    def __getitem__(self, key):
        self.owner.check()

        def index(value):
            return value.detach().cpu().numpy() if torch.is_tensor(value) else value

        key = tuple(index(k) for k in key) if isinstance(key, tuple) else index(key)
        data = self.data[key]
        check_ram(int(np.asarray(data).nbytes), "Overlap matching transfer")
        return torch.from_numpy(np.ascontiguousarray(data)).to(self.device)

    def __setitem__(self, key, value):
        self.owner.check()
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        self.data[key] = value
        if isinstance(self.data, np.memmap):
            self.dirty_bytes += np.asarray(value).nbytes
            if self.dirty_bytes >= 64 << 20:
                self.data.flush()
                self.dirty_bytes = 0


class SkyWorkspace:
    def __init__(
        self, shape, device, free_gpu, root=None, mode=None, check=None, progress=None, log=None, storage_notice=None
    ):
        self.shape = tuple(shape)
        self.device = torch.device(device)
        self.torch = torch
        self.root = Path(root) if root else Path(tempfile.gettempdir()) / "GPUCoadd"
        self.cancel = check or (lambda: None)
        self.storage_notice = storage_notice
        self.progress = progress
        self.log = log or (lambda _: None)
        self.maps = []
        self.directory = None
        self.closed = False
        self.last_check = 0
        self.allocated = 0
        self.last_phase = None
        self.phase_at = 0
        requested = mode or os.environ.get("GPUCOADD_SKY_STORAGE", "auto")
        if requested not in ("auto", "gpu", "ram", "disk"):
            raise ValueError("GPUCOADD_SKY_STORAGE must be auto, gpu, ram or disk")
        n, _, h, w = self.shape
        pixels = h * w
        pairs = min(n * (n - 1) // 2, n * 10)
        self.estimated_bytes = n * pixels * 14 + pairs * min(20000, pixels) * 8 + n * n * 4
        resident = n * pixels * 12 + max(4 * GIB, self.estimated_bytes - n * pixels * 12 + 2 * GIB)
        # The shared rule (memory_plan): the lights already hold their memory; the data is freed
        # before stacking, so the tile working space is not kept back here, only the desktop reserve.
        total_gpu = torch.cuda.get_device_properties(self.device).total_memory if self.device.type == "cuda" else 0
        gpu = gpu_room(free_gpu, total_gpu) if self.device.type == "cuda" else 0
        ram = ram_room(available_ram())
        self.mode = requested
        if requested == "auto":
            self.mode = "gpu" if resident <= gpu else "ram" if self.estimated_bytes <= ram else "disk"
        if self.mode == "gpu" and (self.device.type != "cuda" or resident > gpu):
            raise MemoryError(
                f"Overlap matching GPU storage needs {resident / GIB:.2f} GiB, exceeding the safe budget "
                f"of {gpu / GIB:.2f} GiB. Use automatic, RAM or disk storage."
            )
        if self.mode == "ram" and self.estimated_bytes > ram:
            raise MemoryError(
                f"Overlap matching RAM storage needs {self.estimated_bytes / GIB:.2f} GiB, exceeding "
                f"the safe budget of {ram / GIB:.2f} GiB. Use automatic or disk storage."
            )
        if self.mode == "disk":
            check_space([(self.root, self.estimated_bytes)])
        self.info = {
            "mode": self.mode,
            "estimated_storage_bytes": self.estimated_bytes,
            "resident_gpu_requirement_bytes": resident,
            "sample_precision": "float32 lights / bool masks / int64 pixel indices",
        }
        if storage_notice:
            storage_notice(
                "sky",
                self.mode,
                "normal" if self.mode == "gpu" else "capacity" if requested == "auto" else "forced",
                bytes=resident if self.mode == "gpu" else self.estimated_bytes,
            )
        self.log(
            f"Overlap matching storage: {self.mode}; estimated {self.estimated_bytes / GIB:.2f} GiB. "
            "Samples and fitting rules are unchanged."
        )

    def __enter__(self):
        return self

    def check(self):
        if self.closed:
            raise RuntimeError("Overlap matching temporary storage is closed")
        self.cancel()
        if time.perf_counter() - self.last_check >= 2:
            check_ram(stage="Overlap matching preparation/solve", relieve=True)
            if self.mode in ("disk", "mixed"):
                check_space([(self.root, max(1, self.estimated_bytes - self.allocated))])
            self.last_check = time.perf_counter()

    def phase(self, name, done, total):
        self.check()
        now = time.perf_counter()
        if self.progress and (name != self.last_phase or done == total or now - self.phase_at >= 0.25):
            self.progress(name, done, total)
            self.last_phase = name
            self.phase_at = now

    def free_vram(self):
        return (
            torch.cuda.mem_get_info(self.device)[0]
            + torch.cuda.memory_reserved(self.device)
            - torch.cuda.memory_allocated(self.device)
        )

    def require(self, working_bytes, stage):
        self.check()
        if self.device.type == "cuda":
            check_gpu(self, int(working_bytes) + GIB, "Overlap matching " + stage)

    def consensus_rows(self, n, height, width):
        cells = max(0, (available_ram() - (128 << 20)) // 8)
        if self.device.type == "cuda":
            cells = min(cells, max(0, self.free_vram() - GIB) // 24)
        if cells < n * width:
            raise ResourcePressureError(
                "Overlap matching cannot fit even one exact consensus row. Free more RAM/GPU memory; "
                "samples were not reduced."
            )
        return min(32, height, cells // (n * width))

    def create(self, name, shape, dtype):
        self.check()
        size = int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize
        if self.mode == "gpu":
            return torch.empty(shape, device=self.device, dtype=torch.from_numpy(np.empty((), dtype)).dtype)
        if size == 0 or self.mode == "ram" and check_ram(size, "Overlap matching array allocation"):
            data = np.empty(shape, dtype)
        else:  # disk mode, or RAM ran short: this array is disk-backed (same values, slower)
            if self.mode == "ram":
                self.mode = "mixed"
                self.info["mode"] = "mixed"
                self.log("RAM is short: further Overlap matching arrays use temporary disk storage. Same results.")
                if self.storage_notice:
                    self.storage_notice("sky", "mixed", "capacity", bytes=self.estimated_bytes)
            check_space([(self.root, max(size, self.estimated_bytes - self.allocated))])
            if self.directory is None:
                self.root.mkdir(parents=True, exist_ok=True)
                self.directory = Path(tempfile.mkdtemp(prefix="shared_sky_", dir=self.root))
            data = np.memmap(self.directory / (name + ".bin"), mode="w+", dtype=dtype, shape=shape)
            self.maps.append(data)
        self.allocated += size
        self.info["backed_array_bytes"] = self.allocated
        return StoredTensor(data, self)

    def flush(self):
        for data in self.maps:
            data.flush()

    def __exit__(self, *exc):
        self.closed = True
        for data in self.maps:
            data._mmap.close()
        self.maps.clear()
        if self.directory is not None:
            for path in self.directory.iterdir():
                path.unlink()
            self.directory.rmdir()


def stored(value):
    return isinstance(value, StoredTensor)


def overlap_counts(flat, block):
    """Identical counts: each block sums <=2^18 exact float32 zero/one products.

    Changing row/column tiling cannot round those integers; pixel blocks and their
    float32 accumulation order are kept exactly as in the resident path.
    """
    if block > 1 << 24:
        raise ValueError("Overlap matching count blocks must keep integer sums exactly representable")
    n, pixels = flat.shape
    work = flat.owner
    out = work.create("overlaps", (n, n), np.float32).data
    for i in range(0, n, 256):
        for j in range(0, n, 256):
            count = torch.zeros((min(256, n - i), min(256, n - j)), device=flat.device, dtype=torch.float32)
            for s in range(0, pixels, block):
                work.require(2 * 256 * min(block, pixels) * 5, "overlap counts")
                work.phase("overlap counts", i * n + j, n * n)
                a = flat[i : i + 256, s : s + block].float()
                b = flat[j : j + 256, s : s + block].float()
                count += a @ b.T
                del a, b
            out[i : i + 256, j : j + 256] = count.cpu().numpy()
    return out
