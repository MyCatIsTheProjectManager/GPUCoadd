"""Small failure-only reports; independent of optional activity logging and CUDA."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback


def explain(exc):
    text = str(exc).lower()
    if "source light" in text:
        return (
            "A source light changed, disappeared, or could not be read safely.",
            "Check the file named in the details. Keep the lights unchanged, then rebuild the heat map. No light was excluded automatically.",
        )
    if "unusable normalization" in text:
        return (
            "A light has unusable brightness or noise measurements.",
            "Check the light named in the details and the reference. No light was excluded; decide how to handle it before retrying.",
        )
    if "available gpu memory fell" in text:
        return (
            "Available graphics-card memory decreased during processing.",
            "Close other GPU-heavy applications and retry. Processing stopped without changing image quality settings.",
        )
    if "out of memory" in text and ("cuda" in text or "gpu" in text):
        return (
            "The graphics card ran out of memory.",
            "Close other GPU-heavy applications and retry. Include this report if it happens again.",
        )
    if isinstance(exc, MemoryError) or "unable to allocate" in text:
        return (
            "Windows could not provide the memory this step needed, even after GPUCoadd moved what it "
            "could to disk.",
            "Close other memory-heavy programs and run it again; a smaller framed area also needs less.",
        )
    if (
        getattr(exc, "errno", None) == 28
        or "not enough free disk space" in text
        or "not enough disk space" in text
        or "insufficient disk" in text
        or "no space left" in text
    ):
        return ("There is not enough free disk space.", "Free space on the results and working drives, then retry.")
    if isinstance(exc, PermissionError):
        return (
            "GPUCoadd could not access a required file or folder.",
            "Check the path, write permission and whether another application has locked the file.",
        )
    if "verification failed" in text:
        return (
            "A saved FITS file failed its integrity check.",
            "The unverified file was not published. Check the drive and share this report before retrying.",
        )
    if isinstance(exc, FileNotFoundError):
        return (
            "A required file or folder could not be found.",
            "Check that the lights, working files and drives are still available.",
        )
    if "build the heat map again" in text:
        return (
            "The saved heat map no longer matches the lights or processing setup.",
            "Build the heat map again before stacking.",
        )
    if "no lights reach" in text or "no contributing frames" in text:
        return ("No lights cover the selected area.", "Choose a covered area on the heat map, then try again.")
    if "truncated light" in text or "light dimensions changed" in text:
        return (
            "A light file is incomplete or has changed.",
            "Check the file named in the technical details. Restore it before retrying; no light is excluded automatically.",
        )
    if "no .fit lights found" in text or "light folder not found" in text:
        return (
            "The selected folder does not contain readable light files.",
            "Check the lights folder and make sure its drive is connected.",
        )
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return (
            "The worker could not load a required software component.",
            "Check the Python installation under Engine settings. Share the report if you need help repairing it.",
        )
    return (
        "GPUCoadd could not finish this operation.",
        "Read the technical details below. Copy or export this report when asking for help.",
    )


def build_failure(exc, request, *, stage=None, started=None, elapsed=None, events=()):
    summary, action = explain(exc)
    modules = {}
    for name in ("numpy", "torch", "astropy", "cv2"):
        module = sys.modules.get(name)
        if module is not None:
            modules[name] = str(getattr(module, "__version__", "unknown"))
    sources = {}
    root = Path(__file__).resolve().parents[1]
    for name in (
        "gpucoadd.py",
        "overlap.py",
        "sky_store.py",
        "resource_guard.py",
        "output_store.py",
        "worker/bridge.py",
        "worker/failure_report.py",
        "start_gpucoadd.pyw",
        "GPUCoadd/GPUCoadd.py",
        "GPUCoadd/gpucoadd_qt/dialog.py",
        "GPUCoadd/gpucoadd_qt/logic.py",
    ):
        try:
            sources[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        except OSError:
            pass
    return dict(
        schema=1,
        summary=summary,
        next_step=action,
        error_type=type(exc).__name__,
        error=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        operation=request.get("operation"),
        stage=stage,
        started_utc=started,
        failed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=elapsed,
        request=request,
        recent_events=list(events),
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            versions=modules,
            switches={k: v for k, v in os.environ.items() if k.startswith("GPUCOADD_")},
        ),
        source_sha256=sources,
        privacy_note="Contains local file paths, settings and recent messages. Review before sharing.",
    )


def report_text(report):
    return (
        "GPUCoadd failure report\n"
        + report["summary"]
        + "\n"
        + report["next_step"]
        + "\n\nTechnical details\n"
        + json.dumps(report, indent=2, ensure_ascii=False, default=str)
        + "\n"
    )


def save_failure(report, folder, fallback_root):
    """Best effort, even after disk/permission failures; never hide the original error."""
    errors = []
    for directory in (Path(folder), Path(fallback_root) / "errors", Path(tempfile.gettempdir()) / "GPUCoadd-errors"):
        temporary = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            # Unique files allow fallback reports from several workers to coexist.
            fd, temporary = tempfile.mkstemp(prefix="GPUCoadd-error-", suffix=".partial", dir=directory)
            final = Path(temporary).with_suffix(".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(report_text(report))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, final)
            return str(final), errors
        except OSError as exc:
            errors.append(str(exc))
        finally:
            if temporary:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass
    return None, errors
