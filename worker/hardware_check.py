"""Fast, standalone startup check: NVIDIA/CUDA, free GPU memory, RAM and disk space. No lights,
engine or kernels are loaded. Resource warnings only advise; they never block processing."""

import argparse
import ctypes
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # memory_plan: the rule jobs use

GIB = 2**30
# Below these, a warning asks for action (close programs, free disk space). Processing still runs:
# GPUCoadd then uses slower RAM or disk storage, or stops with a clear message if even that cannot fit.
LOW_RAM = 2 * GIB  # free RAM below which large sets need disk storage
LOW_DISK = 10 * GIB  # free space for heat maps, temporary data and results
GPU_IN_USE = 0.25  # share of GPU memory held by other programs
S50_PRO_LIGHT = 2160 * 3840 * 2  # bytes of one raw light, for "about N lights"
S50_LIGHT = 1080 * 1920 * 2


def memory_status():
    """(total, available) physical RAM in bytes; (0, 0) when unknown."""
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
            return int(status.total), int(status.available)
        return 0, 0
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        return os.sysconf("SC_PHYS_PAGES") * page, os.sysconf("SC_AVPHYS_PAGES") * page
    except (ValueError, OSError, AttributeError):
        return 0, 0


def disk_space(path):
    """(free, total, drive) for the drive that holds `path` (its nearest existing parent)."""
    p = Path(path).resolve()
    while not p.exists() and p.parent != p:
        p = p.parent
    usage = shutil.disk_usage(p)
    return int(usage.free), int(usage.total), p.anchor or str(p)


def gb(value):
    return f"{value / GIB:.1f} GB"


def resources(torch, folders, memory=memory_status, disk=disk_space):
    """Free GPU memory, RAM and disk space, a one-line summary and the warnings worth acting on."""
    info, parts, warnings = {}, [], []
    try:
        free_vram, total_vram = (int(v) for v in torch.cuda.mem_get_info())
    except Exception:
        free_vram = total_vram = None
    if total_vram:
        info.update(free_vram_bytes=free_vram, total_vram_bytes=total_vram)
        used = total_vram - free_vram
        if used > GPU_IN_USE * total_vram:
            warnings.append(
                f"Other programs are using {gb(used)} of the GPU's {gb(total_vram)}. Close them (games, video "
                "editors, browsers using the GPU) for the fastest stacking; otherwise large sets keep their "
                "lights in RAM or read them from disk."
            )
    total_ram, free_ram = memory()
    if total_ram:
        info.update(free_ram_bytes=free_ram, total_ram_bytes=total_ram)
        parts.append(f"RAM {gb(free_ram)} free of {gb(total_ram)}")
        if free_ram < LOW_RAM:
            warnings.append(
                f"Only {gb(free_ram)} of RAM is free. GPUCoadd still works, but large sets will use slower "
                "disk storage and may stop with a request to free memory; closing other programs avoids that."
            )
    if total_vram and total_ram:
        from memory_plan import gpu_room, ram_room, tile_workspace

        # the same rule each job applies when it starts, so the user knows what to expect
        gpu = gpu_room(free_vram, total_vram, tile_workspace(total_vram))
        ram = ram_room(free_ram)
        info.update(gpu_room_bytes=gpu, ram_room_bytes=ram)
        info["expectation"] = (
            f"Sets up to {gb(gpu)} of lights (about {gpu // S50_PRO_LIGHT:,} S50 Pro or {gpu // S50_LIGHT:,} S50 "
            f"lights) stay on the GPU; larger sets use RAM up to {gb(ram)}, then disk storage automatically."
        )
    seen = set()
    for label, folder in folders:
        try:
            free, total, drive = disk(folder)
        except OSError:
            continue
        info.setdefault("disks", []).append({"use": label, "path": str(folder), "drive": drive, "free_bytes": free})
        if drive in seen:
            continue
        seen.add(drive)
        parts.append(f"{gb(free)} free on {drive.rstrip(chr(92))} ({label})")
        if free < LOW_DISK:
            warnings.append(
                f"Only {gb(free)} is free on {drive.rstrip(chr(92))}, which holds the {label}. Large mosaics can "
                "need several GB of temporary data and results; free some space there."
            )
    return info, parts, warnings


def display_devices():
    """Best-effort Windows display inventory, only needed for unsupported setups."""
    if os.name != "nt":
        return []
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=True,
        )
        data = json.loads(result.stdout)
        return data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def check(torch_module=None, inventory=display_devices, folders=(), memory=memory_status, disk=disk_space):
    def failed(code, summary, next_step, error="", **extra):
        return dict(
            supported=False,
            code=code,
            summary=summary,
            next_step=next_step,
            error=error,
            python=sys.executable,
            **extra,
        )

    try:
        torch = torch_module if torch_module is not None else importlib.import_module("torch")
    except Exception as exc:
        return failed(
            "torch_unavailable",
            "The selected Python environment cannot load PyTorch.",
            "Choose the GPUCoadd CUDA Python environment under Engine settings, then click Check again.",
            str(exc),
        )
    version = dict(
        torch=str(torch.__version__), cuda=getattr(torch.version, "cuda", None), hip=getattr(torch.version, "hip", None)
    )
    if version["hip"]:
        return failed(
            "amd_unsupported",
            "AMD/ROCm is not supported by this GPUCoadd release.",
            "This GPUCoadd release requires an NVIDIA GPU and CUDA-enabled PyTorch.",
            version=version,
        )
    try:
        available = bool(torch.cuda.is_available())
        if not version["cuda"] or not available:
            devices = inventory()
            names = [str(d.get("Name", "")) for d in devices]
            nvidia = any(
                "VEN_10DE" in str(d.get("PNPDeviceID", "")).upper() or "nvidia" in str(d.get("Name", "")).lower()
                for d in devices
            )
            known_other = any(any(v in n.lower() for v in ("amd", "radeon", "intel")) for n in names)
            if known_other and not nvidia:
                return failed(
                    "unsupported_gpu",
                    "No supported NVIDIA GPU was detected. AMD and Intel GPUs are not supported.",
                    "This release requires NVIDIA CUDA; there is no CPU fallback.",
                    devices=devices,
                    version=version,
                )
            return failed(
                "cuda_unavailable",
                "No working NVIDIA/CUDA device is available in the selected Python environment.",
                "If you have NVIDIA, check its driver and select the CUDA-enabled GPUCoadd Python environment. AMD/Intel GPUs are not supported.",
                devices=devices,
                version=version,
            )
        name = str(torch.cuda.get_device_name(0))
        if any(v in name.lower() for v in ("radeon", "amd", "intel")):
            return failed(
                "unsupported_runtime",
                "This GPU/runtime is not a supported NVIDIA CUDA setup.",
                "Use native NVIDIA CUDA. AMD/Intel and CUDA translation layers are not supported in this release.",
                gpu=name,
                version=version,
            )
        # Check actual execution on device 0, which is also what the worker uses.
        value = torch.ones(1, device="cuda") + 1
        torch.cuda.synchronize()
        if float(value.item()) != 2.0:
            raise RuntimeError("CUDA execution check returned an unexpected value")
        del value
        total = int(torch.cuda.get_device_properties(0).total_memory)
        info, parts, warnings = resources(torch, folders, memory, disk)
        free_vram = info.get("free_vram_bytes")
        gpu = f"{name}, {gb(free_vram)} free of {gb(total)}" if free_vram is not None else f"{name} ({gb(total)})"
        return dict(
            supported=True,
            code="ready",
            summary=f"GPU ready: {gpu}."
            + (" " + "; ".join(parts) + "." if parts else "")
            + (" " + info["expectation"] if "expectation" in info else ""),
            warnings=warnings,
            gpu=name,
            total_vram_bytes=total,
            resources=info,
            python=sys.executable,
            version=version,
        )
    except Exception as exc:
        return failed(
            "cuda_failed",
            "The NVIDIA/CUDA execution check failed.",
            "Check the NVIDIA driver and CUDA-enabled Python installation. Copy the error details if the problem persists.",
            str(exc),
            version=version,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", help="GPUCoadd's work area (heat maps, temporary data)")
    ap.add_argument("--results", help="the results folder")
    args = ap.parse_args()
    folders = [(label, path) for label, path in (("work area", args.work), ("results folder", args.results)) if path]
    print(json.dumps(check(folders=folders), allow_nan=False), flush=True)
