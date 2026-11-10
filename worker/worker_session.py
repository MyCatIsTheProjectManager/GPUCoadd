"""Single-client local worker. Ready/stop files are written only after request data is closed."""

import gc
import hashlib
import json
from pathlib import Path
import time


class WorkerSession:
    def __init__(self, timeout=300):
        self.backend = None
        self.key = None
        self.timeout = timeout
        self.source = self.source_stamp()

    @staticmethod
    def source_stamp():
        root = Path(__file__).resolve().parents[1]
        paths = sorted(root.glob("*.py")) + sorted((root / "worker").glob("*.py"))
        return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in paths)).hexdigest()

    def validate_source(self):
        if self.source != self.source_stamp():
            raise ValueError("Worker code changed while it was open. GPU cache released; run the operation again.")

    def release(self):
        backend, self.backend = self.backend, None
        self.key = None
        if backend is not None:
            torch = backend.torch
            del backend
            gc.collect()
            torch.cuda.empty_cache()

    def acquire(self, key, needed):
        be = self.backend
        if be is None or key != self.key or not set(needed) <= set(be.loaded):
            del be
            self.release()
            return None
        # Match a fresh crop-aware worker, including its normalization pass and VRAM budget.
        be.keep_only(needed)
        be.timer.t.clear()
        if isinstance(getattr(be, "transfer_bytes", None), int):
            be.transfer_bytes = 0
        be.torch.cuda.reset_peak_memory_stats()
        return be

    @staticmethod
    def warm_up():
        """Import torch and create the CUDA context once the first job (usually the folder scan,
        which needs neither) is done, while the window waits for the user, instead of in the
        first heat map (about 2.6 s measured). A failure here is left for a GPU job to report."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.zeros(1, device="cuda")
        except Exception:
            pass

    def remember(self, backend, key):
        self.backend, self.key = backend, key

    def idle(self):
        be = self.backend
        if be is not None:
            be.reserve_stack(0, 1, 1)
            gc.collect()
            be.torch.cuda.empty_cache()
        info = be.storage_info() if be is not None and isinstance(getattr(be, "storage_mode", None), str) else {}
        return {
            "resident_lights": info.get("resident_lights", len(be.loaded) if be is not None else 0),
            "available_lights": info.get("available_lights", 0),
            "storage_mode": info.get("mode", "gpu"),
            "host_raw_gib": info.get("host_raw_bytes", 0) / 2**30,
            "resident_vram_gib": be.torch.cuda.memory_allocated() / 2**30 if be is not None else 0,
            "idle_timeout_s": self.timeout,
        }


def serve(directory, run_request, timeout=300):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    session = WorkerSession(timeout)
    last = time.monotonic()
    reason = "released"
    warmed = False
    try:
        while not (directory / "stop").exists():
            if (directory / "ready").exists():
                # The client closes request.json and command.txt before creating ready.
                command = (directory / "command.txt").read_text(encoding="utf-8-sig").strip()
                path = Path(json.loads(command) if command.startswith('"') else command)  # a JSON-quoted path
                (directory / "ready").unlink()
                code = run_request(path, session)
                if code:  # discard CUDA/context state after an error or cancellation
                    reason = "operation failed or cancelled"
                    break
                if not warmed:
                    warmed = True
                    session.warm_up()
                last = time.monotonic()
            elif time.monotonic() - last >= timeout:
                reason = "idle timeout"
                break
            else:
                time.sleep(0.1)
    finally:
        session.release()
        print(json.dumps({"type": "session_closed", "reason": reason}), flush=True)
        # the session folder only held command/ready/stop files: remove it with the session
        for name in ("command.txt", "ready", "stop"):
            (directory / name).unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass
    return 0
