"""A stand-in for worker/bridge.py in the window test: the same command line, session files and
events, without a GPU. It writes a small heat-map PNG and an
empty "stack" so the dialog has files to show."""

import argparse
import json
from pathlib import Path
import struct
import sys
import time
import zlib


def png(path, width=40, height=30):
    rows = b"".join(b"\x00" + bytes([200, 40, 40]) * width for _ in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def emit(kind, **fields):
    print(json.dumps({"type": kind, "time": "", "elapsed_s": 0, **fields}), flush=True)


def hang(out, tag, run_folder):
    """A worker that stops answering: unfinished files of this job (tagged, as the real writers
    name them), optionally a saved stack it announced, a child process that started a grandchild
    (then writes tree_ready in the run folder), and no reaction to the cancel file."""
    import subprocess

    out.mkdir(parents=True, exist_ok=True)
    (out / "fake_aligned").mkdir(exist_ok=True)
    (out / f".fake_stack.fits.{tag}.k3j9.partial").write_bytes(b"unfinished")
    (out / "fake_aligned" / f".Light_0_aligned.fits.{tag}.q81z.partial").write_bytes(b"unfinished")
    if "published" in out.name:
        (out / "fake_stack.fits").write_bytes(b"complete")
        emit("published", path=str(out / "fake_stack.fits"))
    grandchild = "import time; time.sleep(600)"
    ready = str(run_folder / "tree_ready")
    child = (f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
             f"time.sleep(0.5); open({ready!r}, 'w').close(); time.sleep(600)")
    subprocess.Popen([sys.executable, "-c", child])
    emit("stage_started", stage="stacking", message="Started stacking")
    time.sleep(600)


def run(request_path, session):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    op = request["operation"]
    folder = Path(request["work_dir"]) / "runs" / f"fake_{op}_{time.time_ns()}"
    folder.mkdir(parents=True)
    emit("started", folder=str(folder), tag=folder.name, message="Operation started")
    if "hang" in request["output_dir"]:
        hang(Path(request["output_dir"]), folder.name, folder)
    groups = [{"key": request["lights_dirs"][0] + "|S50|IRCUT|10", "label": "S50, IRCUT, 3 x 10 s", "lights": 3,
               "total_exposure_s": 30, "used": True, "main": True, "reason": None}]
    result = {"status": "complete", "operation": op, "folder": str(folder), "groups": groups,
              "request_file": str(request_path), "files": {"report": str(folder / "report.json")},
              "output_options": request["outputs"], "warnings": [], "storage_notices": [], "wall_s": 1.0}
    if op == "scan":
        emit("stage_started", stage="checking folders", message="Started checking folders")
        result.update(input_frames=3, used_frames=3)
    else:
        stages = (["loading lights", "frame analysis", "registration", "frame count", "saving outputs"]
                  if op == "analyze" else ["loading lights", "stacking", "saving outputs"])
        for stage in stages:
            emit("stage_started", stage=stage, message=f"Started {stage}")
            for done in (1, 2):
                if Path(str(request_path)[:-5] + ".cancel").exists():
                    result = {"status": "cancelled", "operation": op, "folder": str(folder),
                              "request_file": str(request_path)}
                    break
                emit("progress", stage=stage, done=done, total=2, stage_eta_s=None)
                time.sleep(0.05 if "cancel" not in request["output_dir"] else 0.5)
            if result["status"] != "complete":
                break
            emit("stage_finished", stage=stage, duration_s=0.1)
        if result["status"] == "complete":
            notice = {"component": "raw", "mode": "ram", "reason": "capacity"}
            emit("storage", **notice)
            result["storage_notices"] = [notice]
            canvas = {"width": 400, "height": 300}
            if op == "analyze":
                png(folder / "heatmap.png")
                (folder / "analysis.json").write_text("{}", encoding="utf-8")
                result.update(canvas=canvas, heatmap=str(folder / "heatmap.png"),
                              analysis_file=str(folder / "analysis.json"), input_frames=3, retained=3, aligned=3,
                              alignment_failed=0, sharpness_rejected=0, reference="Light_0.fit",
                              reference_override=False, settings=request["settings"])
            else:
                stack = Path(request["output_dir"]) / "fake_stack.fits"
                stack.write_bytes(b"")
                result.update(canvas={"width": request["region"][2], "height": request["region"][3]},
                              stack_file=str(stack), contributing=3, retained=3, identity={"target": "Fake"},
                              image_outputs=[{"label": "Stacked image", "format": "FITS", "path": str(stack),
                                              "role": "stack_fits"}])
                if request["outputs"].get("aligned_lights"):
                    # As if the drive were too full: the stack is kept and the export becomes a warning
                    warning = {"output": "aligned_lights", "kind": "environment",
                               "message": "The aligned lights export was not saved: Not enough free disk space. "
                                          "The rest of the result is unaffected."}
                    emit("warning", output=warning["output"], message=warning["message"])
                    result["warnings"] = [warning]
    if session:
        result["reused_prepared_lights"] = op == "stack"
        result["worker_cache"] = {"resident_lights": 3, "available_lights": 3, "storage_mode": "gpu",
                                  "resident_vram_gib": 0.01, "idle_timeout_s": 300}
    (folder / "report.json").write_text(json.dumps(result), encoding="utf-8")
    Path(str(request_path)[:-5] + ".result.json").write_text(json.dumps(result), encoding="utf-8")
    emit("finished", **result)
    return 0 if result["status"] == "complete" else 1


def serve(directory):
    directory = Path(directory)
    reason = "released"
    while not (directory / "stop").exists():
        if (directory / "ready").exists():
            path = (directory / "command.txt").read_text(encoding="utf-8").strip()
            (directory / "ready").unlink()
            if run(path, True):
                reason = "operation failed or cancelled"
                break
        time.sleep(0.02)
    for name in ("command.txt", "ready", "stop"):
        (directory / name).unlink(missing_ok=True)
    emit("session_closed", reason=reason)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request")
    mode.add_argument("--session")
    args = ap.parse_args()
    sys.exit(serve(args.session) if args.session else run(args.request, False))
