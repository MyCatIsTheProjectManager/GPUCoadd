# GPUCoadd window for Siril's Scripts menu. Add this folder to Siril's script paths (Preferences >
# Scripts) and run GPUCoadd from the Scripts menu. Siril runs this file with its own Python; the
# stacking runs in GPUCoadd's CUDA Python environment (worker/bridge.py), as for the window on its own
# dialog. Everything specific to Siril is in this file; the window is in gpucoadd_qt/.

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

MIN_VERSION = "1.4.0"
# Siril's own variables for a script's connection; the worker must not inherit them
SIRIL_VARIABLES = ("MY_PIPE", "MY_SOCKET", "SIRIL_PYTHON_CLI", "SIRIL_PYTHON_DEBUG")


class SirilHost:
    """What the window needs from Siril: its console, showing results, and which run of Siril this is."""

    variables = SIRIL_VARIABLES

    def __init__(self, siril):
        self.siril = siril
        self.session = session_id()

    def log(self, text, error=False):
        from sirilpy import LogColor

        self.siril.log(text, LogColor.RED if error else LogColor.DEFAULT)

    def open(self, path):
        """Opens a saved image. The stack is linear, so the display (not the data) is set to
        autostretch. The path goes to the command line with forward slashes, which Windows accepts,
        so no backslash can be read as an escape."""
        import sirilpy

        self.siril.cmd("load", '"' + str(path).replace("\\", "/") + '"')
        if self.siril.is_cli():
            return  # no display to set: the command-line program would wait 60 s for an answer
        try:
            self.siril.set_siril_stf(sirilpy.STFType.AUTOSTRETCH_DISPLAY)
        except Exception:  # an older sirilpy without display control: the image is still open
            pass


def session_id(pid=None):
    """Names the running Siril by its process id and start time, or "" when it cannot be found.
    Scripts run in a child of Siril's Python launcher, so this walks up the process tree. (The pipe a
    script talks through changes with every script.)"""
    if os.name != "nt":
        return ""
    try:
        parents = process_parents()
        pid = os.getpid() if pid is None else pid
        for _ in range(8):
            pid = parents.get(pid, (0, ""))[0]
            if not pid:
                return ""
            if parents.get(pid, (0, ""))[1].lower() in ("siril.exe", "siril-cli.exe"):
                started = process_start(pid)
                return f"{pid}@{started}" if started else ""
    except OSError:
        pass
    return ""


def process_parents():
    """{pid: (parent pid, exe name)} of all processes (Windows)."""
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if snapshot in (None, wintypes.HANDLE(-1).value):
        raise OSError("no process snapshot")
    entries = {}
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(Entry)
        ok = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            entries[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile)
            ok = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return entries


def process_start(pid):
    """A process's creation time (Windows FILETIME), or 0: with the pid it names one process run."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return 0
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return 0
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def main():
    import sirilpy
    from sirilpy import LogColor

    sirilpy.ensure_installed("PyQt6")
    from gpucoadd_qt import dialog

    siril = sirilpy.SirilInterface()
    try:
        siril.connect()
    except Exception as exc:
        print(f"GPUCoadd must be started from the Scripts menu ({exc}).", file=sys.stderr)
        return 1
    try:
        try:
            siril.cmd("requires", MIN_VERSION)
        except Exception:
            siril.log(f"GPUCoadd needs version {MIN_VERSION} or newer.", LogColor.RED)
            return 1
        siril.log("GPUCoadd opened. Stacking runs in GPUCoadd's own CUDA Python environment.", LogColor.GREEN)
        dialog.run(SirilHost(siril))
        siril.log("GPUCoadd closed.", LogColor.GREEN)
        return 0
    finally:
        try:
            siril.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
