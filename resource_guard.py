"""Live resource checks; never change samples, precision or tile layout.

Low RAM never stops a job. Data that can be held elsewhere moves: the running job registers
`relieve` (lights held in RAM are read again from their files instead), and allocations that
cannot fit go to disk where they have a disk form. When nothing more can move, one warning is
logged and processing continues, while Windows moves idle memory of other programs to its page
file. Only a real allocation failure (MemoryError) stops a job. GPU memory stays strictly checked:
the driver can silently fall back to much slower shared system memory instead of failing."""

from raw_storage import available_ram, GIB

RAM_RESERVE = 128 << 20


class ResourcePressureError(MemoryError):
    pass


_handlers = {"relieve": None, "warn": None}
_warned = set()


def set_handlers(relieve=None, warn=None):
    """Register the running job's ways to free RAM (relieve(bytes) -> bytes freed) and to log a
    warning (warn(text)); call with no arguments when the job ends."""
    _handlers.update(relieve=relieve, warn=warn)
    _warned.clear()


def warn_once(key, text):
    if key not in _warned:
        _warned.add(key)
        if _handlers["warn"] is not None:
            _handlers["warn"](text)


def check_ram(required=0, stage="processing", relieve=False):
    """Whether `required` bytes plus RAM_RESERVE are free. Never raises.

    relieve=True (the periodic checks): when short, the job's data that has another form is
    moved out of RAM first. Allocations pass relieve=False and choose their own disk form, so a
    lower-priority buffer never pushes out the lights."""
    free = available_ram()
    if free <= 0:
        warn_once("unmeasured", "Free RAM could not be measured; processing continues without RAM checks.")
        return True
    need = int(required) + RAM_RESERVE
    if free >= need:
        return True
    if relieve and _handlers["relieve"] is not None:
        _handlers["relieve"](need - free)
        free = available_ram()
        if free >= need:
            return True
    if relieve or not required:
        warn_once(
            "low",
            f"RAM is very low during {stage} ({free / GIB:.2f} GiB free). GPUCoadd continues; other "
            "programs may slow down while Windows moves their idle memory to disk. Closing programs helps.",
        )
    return False


def check_gpu(backend, required, stage="stacking"):
    free = backend.free_vram()
    if free < required:
        # Only unused allocator blocks: live raw data, tile buffers and batch grouping stay put.
        backend.torch.cuda.empty_cache()
        free = backend.free_vram()
    if free < required:
        raise ResourcePressureError(
            f"Available GPU memory fell below the safe working requirement during {stage}: "
            f"need {required / GIB:.2f} GiB, available {free / GIB:.2f} GiB. "
            "Close other GPU-heavy applications and retry. Tile size and image quality were not changed."
        )
    return free


def check_tile(backend, bounds, lights, rejection, feather=False):
    from tileplan import memory_estimate

    x0, x1, y0, y1 = bounds
    estimate = memory_estimate(
        max(x1 - x0, y1 - y0), x1 - x0, y1 - y0, [lights], debayer=backend.debayer, rejection=rejection, feather=feather
    )
    held = 0
    for name in ("_stack_buf", "_feather_buf"):
        buffer = getattr(backend, name, None)
        if buffer is not None:
            held += buffer.numel() * buffer.element_size()
    return check_gpu(backend, estimate["required_bytes"] - min(held, estimate["buffer_bytes"]), f"tile {bounds}")
