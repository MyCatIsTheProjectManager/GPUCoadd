"""GPUCoadd's window on its own, without Siril. The Start menu entry setup creates runs this file with
GPUCoadd's Python (.venv\\Scripts\\pythonw.exe start_gpucoadd.pyw); the window is the same one Siril
opens from its Scripts menu (GPUCoadd/gpucoadd_qt)."""

import os
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent / "GPUCoadd"))


class StandaloneHost:
    """What the window needs from a host program, without one: saved images open in the program
    Windows uses for FITS files, or are shown in Explorer when there is none."""

    variables = ()

    def __init__(self):
        # Each start is its own session, as each start of Siril is: folders are restored only while
        # this window's process runs, and never mixed with Siril's.
        self.session = f"standalone:{uuid.uuid4().hex}"

    def log(self, text, error=False):
        pass  # the window's own activity log already shows everything

    def open(self, path):
        try:
            os.startfile(path)
        except OSError:  # no program is set up for .fits files
            subprocess.run(["explorer.exe", "/select,", str(Path(path))], check=False)


def main():
    from gpucoadd_qt import dialog

    if os.name == "nt":  # the taskbar shows GPUCoadd's icon instead of grouping it with Python
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GPUCoadd.Window")
    dialog.run(StandaloneHost())
    return 0


if __name__ == "__main__":
    sys.exit(main())
