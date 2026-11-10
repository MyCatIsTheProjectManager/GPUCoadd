"""One memory rule for every job, applied to the free GPU memory and RAM measured when each part
decides (so data this job already holds is counted once, and other programs growing is noticed).

Reserves stay free for the desktop and the system. Within what is left, data is placed by speed
benefit: the lights first (read again for every tile), then the Overlap matching data (read again
for every solver pass, and freed before stacking), and the output canvas last (written once per
tile: RAM or disk cost about the same, measured on M42: 26 s against 28 s). The GPU also keeps
working space for stacking tiles. Where data is held never changes a pixel.
"""

from raw_storage import GIB, available_ram
from resource_guard import RAM_RESERVE as SYSTEM_RAM  # RAM left free for Windows, as in the live checks


def desktop_vram(total_gpu):
    """GPU memory left free for the desktop, the browser and other programs."""
    return max(512 << 20, int(total_gpu * 0.08))


def tile_workspace(total_gpu, largest_light_pixels=0):
    """GPU memory kept for full-frame star detection and stacking tiles."""
    return max(3 * GIB, int(total_gpu * 0.25), int(largest_light_pixels) * 64)


def ram_room(free_ram):
    """RAM this job may fill: everything available (Windows counts its file cache as available,
    since it gives that back at once) except SYSTEM_RAM."""
    return max(0, int(free_ram) - SYSTEM_RAM)


def gpu_room(free_gpu, total_gpu, workspace=0):
    """GPU memory this job may fill with data, after the desktop reserve and `workspace`."""
    return max(0, int(free_gpu) - desktop_vram(total_gpu) - int(workspace))


def gb(value):
    return f"{value / GIB:.1f} GB" if value >= GIB // 5 else f"{value / (1 << 20):.0f} MB"


def start_summary(free_gpu, total_gpu, free_ram=None):
    """The memory a job starts with, and what stays free for other programs."""
    free_ram = available_ram() if free_ram is None else free_ram
    return (
        f"Memory at start: {gb(free_gpu)} of {gb(total_gpu)} GPU memory and {gb(free_ram)} of RAM free; "
        f"{gb(desktop_vram(total_gpu))} of GPU memory and {gb(SYSTEM_RAM)} of RAM stay free for other programs."
    )


PLACES = {"gpu": "in GPU memory", "ram": "in RAM", "disk": "on disk", "mixed": "in RAM and on disk"}
NAMES = {"raw": "lights", "sky": "Overlap matching data", "output": "output canvas"}


def plan_summary(notices):
    """One line from the storage notices of a job: where each kind of data is held."""
    parts = []
    for component in ("raw", "sky", "output"):
        notice = notices.get(component)
        if notice:
            size = notice.get("bytes")
            parts.append(
                f"{NAMES[component]} {PLACES.get(notice['mode'], notice['mode'])}"
                + (f" ({gb(size)})" if size else "")
            )
    return "Memory plan: " + "; ".join(parts) + "." if parts else ""
