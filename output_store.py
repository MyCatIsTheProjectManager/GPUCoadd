"""Canvas budgeting and job-owned disk-backed arrays. No pixel calculations change."""

import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from memory_plan import SYSTEM_RAM, ram_room
from raw_storage import available_ram, GIB

BLOCK_BYTES = 4 << 20
MARGIN = 64 << 20

# The running job (worker/bridge.py sets it): its tag, the name of its run folder, unique per
# job, goes into every temporary output file's name, so a window that had to end a stuck worker
# removes that job's unfinished files and nothing another window may be writing. `published`
# is told each output file the job has put in place.
JOB = {"tag": "", "published": None}


def set_job(tag="", published=None):
    JOB.update(tag=tag, published=published)


def temp_prefix(path):
    """The mkstemp prefix of a temporary file for output `path`: hidden, named after it and the job."""
    return "." + Path(path).name + "." + (JOB["tag"] + "." if JOB["tag"] else "")


def published(path):
    if JOB["published"] is not None:
        JOB["published"](str(path))


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        if path.parent == path:
            raise OSError(f"Storage location is unavailable: {path}")
        path = path.parent
    return path if path.is_dir() else path.parent


def check_space(requirements):
    """Sum simultaneous needs on the same filesystem; this is not an exclusive reservation."""
    volumes = {}
    for path, size in requirements:
        if not size:
            continue
        parent = existing_parent(path)
        key = parent.stat().st_dev
        row = volumes.setdefault(key, [parent, 0])
        row[1] += int(size)
    result = []
    for parent, size in volumes.values():
        free = shutil.disk_usage(parent).free
        need = size + MARGIN
        if free < need:
            raise OSError(
                f"Not enough free disk space on {parent}: need {need / GIB:.2f} GiB "
                f"including safety space, have {free / GIB:.2f} GiB. Choose another "
                "working/output drive or a smaller output area."
            )
        result.append({"location": str(parent), "required_bytes": need, "free_bytes": free})
    return result


def row_blocks(height, width, itemsize=4):
    rows = max(1, BLOCK_BYTES // max(1, int(width) * itemsize))
    for y in range(0, height, rows):
        yield y, min(height, y + rows)


def coverage_counts(coverage, frames):
    covered = half = full = 0
    for a, b in row_blocks(*coverage.shape):
        c = coverage[a:b]
        covered += int(np.count_nonzero(c))
        half += int(np.count_nonzero(c >= frames / 2))
        full += int(np.count_nonzero(c == frames))
    return {"covered_pixels": covered, "half_coverage_pixels": half, "full_coverage_pixels": full}


class OutputWorkspace:
    def __init__(self, work_dir, output_dir, options, log=None, check_cancel=None, mode=None, storage_notice=None):
        self.work_dir = Path(work_dir)
        self.output_dir = Path(output_dir)
        self.options = options
        self.log = log or (lambda _: None)
        self.check_cancel = check_cancel or (lambda: None)
        self.storage_notice = storage_notice
        self.requested = mode or os.environ.get("GPUCOADD_CANVAS", "auto")
        if self.requested not in ("auto", "ram", "disk"):
            raise ValueError("GPUCOADD_CANVAS must be auto, ram or disk")
        self.mode = "ram"
        self.maps = []
        self.directory = None
        self.info = {}
        self.canvas_bytes = 0
        self.ram_bytes = 0
        self.skipped_previews, self.preview_shortage = [], None

    def __enter__(self):
        return self

    def plan(self, width, height, coverage_only=False, kept=False):
        pixels = int(width) * int(height)
        canvas_bytes = pixels * ((8 if coverage_only else 16) + (6 if kept else 0))
        preview_bytes = pixels * (
            (64 if self.options.get("stack_preview") and not coverage_only else 0)
            + (16 if self.options.get("coverage_png") else 0)
        )
        free_ram = available_ram()
        ram_budget = ram_room(free_ram)  # the shared rule (memory_plan); the lights already hold theirs
        free = (
            f"{free_ram / GIB:.2f} GiB of RAM is free, of which {ram_budget / GIB:.2f} GiB can be used "
            f"({SYSTEM_RAM >> 20} MiB stay free for the system)"
        )
        # PNG previews are rendered in RAM. When they do not fit they are skipped with a warning
        # (the FITS output does not need them) instead of stopping the job.
        self.skipped_previews, self.preview_shortage = [], None
        if preview_bytes and preview_bytes + MARGIN > ram_budget:
            self.skipped_previews = [k for k in ("stack_preview", "coverage_png") if self.options.get(k)]
            self.preview_shortage = (
                f"the selected PNG previews need {(preview_bytes + MARGIN) / GIB:.2f} GiB of RAM, but {free}"
            )
            self.options = {**self.options, "stack_preview": False, "coverage_png": False}
            preview_bytes = 0
        mode = self.requested
        if mode == "auto":
            mode = "ram" if canvas_bytes + preview_bytes + MARGIN <= ram_budget else "disk"
        if mode == "ram" and canvas_bytes + preview_bytes + MARGIN > ram_budget:
            raise MemoryError(
                f"The output canvas needs {(canvas_bytes + preview_bytes + MARGIN) / GIB:.2f} GiB of RAM, but {free}. "
                "Use automatic or disk-backed canvas storage."
            )
        exports = pixels * (
            (0 if coverage_only else 12)
            + (4 if self.options.get("coverage_fits") else 0)
            + (1 if self.options.get("coverage_png") else 0)
            + (1 if self.options.get("stack_preview") and not coverage_only else 0)
        )
        disk = check_space([(self.work_dir, canvas_bytes if mode == "disk" else 0), (self.output_dir, exports)])
        if exports:
            # Discover inaccessible output locations before rendering, not at the final save.
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fd, probe = tempfile.mkstemp(prefix=".gpucoadd_write_check.", dir=self.output_dir)
            os.close(fd)
            Path(probe).unlink()
        self.mode = mode
        if self.storage_notice:
            # free and needed RAM let the dialog say whether closing other programs would help
            self.storage_notice(
                "output",
                mode,
                "normal" if mode == "ram" else "capacity" if self.requested == "auto" else "forced",
                free_ram_bytes=free_ram,
                needed_ram_bytes=canvas_bytes + preview_bytes + MARGIN,
                bytes=canvas_bytes,
            )
        self.canvas_bytes = canvas_bytes
        self.info = {
            "mode": mode,
            "canvas_bytes": canvas_bytes,
            "ram_budget_bytes": ram_budget,
            "preview_ram_bytes": preview_bytes,
            "export_bytes": exports,
            "disk_checks": disk,
        }
        self.log(
            f"Output canvas: {width} x {height}; {canvas_bytes / GIB:.2f} GiB of buffers, "
            f"using {'RAM' if mode == 'ram' else 'temporary disk-backed arrays'} "
            f"({free_ram / GIB:.2f} GiB of RAM free). "
            "FITS output is written and verified in blocks before publication."
        )

    def allocate(self, name, shape, dtype, zero=False):
        self.check_cancel()
        from resource_guard import check_ram

        size = int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize
        if self.mode == "ram":
            # forced RAM (GPUCOADD_CANVAS=ram) allocates regardless; only a real failure stops it
            if self.requested == "ram" or check_ram(size, "output buffer allocation"):
                try:
                    array = (np.zeros if zero else np.empty)(shape, dtype)
                    self.ram_bytes += size
                    return array
                except MemoryError:  # Windows refused the memory after all
                    if self.requested != "auto":
                        raise
            check_space([(self.work_dir, max(size, self.canvas_bytes - self.ram_bytes))])
            self.mode = "disk"
            self.info["mode"] = "mixed" if self.ram_bytes else "disk"
            self.info["allocation_fallback"] = "RAM pressure: subsequent buffers use disk"
            if self.storage_notice:
                self.storage_notice(
                    "output", "mixed", "capacity", free_ram_bytes=available_ram(), needed_ram_bytes=size + MARGIN
                )
            self.log(
                "Available RAM decreased; subsequent output buffers use temporary disk storage. "
                "Pixel values are unchanged."
            )
        self.check_resources()
        if self.directory is None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix="canvas_", dir=self.work_dir))
        a = np.memmap(self.directory / (name + ".bin"), dtype=dtype, mode="w+", shape=shape)
        # New files read as zero; don't fault the entire map into RAM to initialize it.
        self.maps.append(a)
        return a

    def check_resources(self, include_exports=False):
        from resource_guard import check_ram

        check_ram(stage="output processing", relieve=True)
        allocated = sum(a.nbytes for a in self.maps)
        pending = max(0, self.canvas_bytes - self.ram_bytes - allocated) if self.mode == "disk" else 0
        exports = self.info.get("export_bytes", 0) if include_exports else 0
        check_space([(self.work_dir, max(1, pending) if self.mode == "disk" else 0), (self.output_dir, exports)])

    def float_coverage(self, coverage):
        target = self.allocate("coverage_float", coverage.shape, np.float32)
        for a, b in row_blocks(*coverage.shape):
            self.check_cancel()
            target[a:b] = coverage[a:b].astype(np.float32)
        return target

    def flush(self):
        self.check_resources()
        for a in self.maps:
            a.flush()

    def __exit__(self, *exc):
        for a in self.maps:
            a._mmap.close()
        self.maps.clear()
        if self.directory is not None:
            for path in self.directory.iterdir():
                path.unlink()
            self.directory.rmdir()
