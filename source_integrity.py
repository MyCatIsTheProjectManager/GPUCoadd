"""Detect ordinary source changes without modifying or re-encoding any light."""

from pathlib import Path


class SourceFileError(OSError):
    """An input cannot safely be used; never implies permission to exclude it."""


def file_stamp(path):
    try:
        s = Path(path).stat()
    except OSError as exc:
        raise SourceFileError(f"Source light unavailable: {path}: {exc}") from exc
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, device=s.st_dev, inode=s.st_ino)


class SourceGuard:
    def __init__(self, paths):
        self.stamps = {str(Path(p).absolute()): file_stamp(p) for p in paths}

    def validate(self, path=None):
        paths = self.stamps if path is None else (str(Path(path).absolute()),)
        for p in paths:
            if file_stamp(p) != self.stamps[p]:
                raise SourceFileError(
                    f"Source light changed during this operation: {p}. "
                    "Stop changing the lights and rebuild the heat map. No light was excluded."
                )

    def read(self, path, reader):
        self.validate(path)
        try:
            value = reader(path)
        except Exception as exc:
            self.validate(path)
            raise SourceFileError(f"Could not read source light: {path}: {exc}") from exc
        self.validate(path)
        return value

    def select(self, paths):
        # Preserve the pre-header snapshot, rather than accepting any changes since then.
        selected = object.__new__(SourceGuard)
        selected.stamps = {str(Path(p).absolute()): self.stamps[str(Path(p).absolute())] for p in paths}
        selected.validate()
        return selected
