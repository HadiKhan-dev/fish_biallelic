"""A command's stdout mirrored to its log file (stderr stays on the terminal)."""
from __future__ import annotations

from contextlib import contextmanager
import sys


class TeeOutput:
    """Mirror writes to the original stream and a line-buffered log, proxying other stream attributes."""

    def __init__(self, log_path, original):
        object.__setattr__(self, "_log_file", open(log_path, "a", buffering=1))
        object.__setattr__(self, "_original", original)

    def write(self, message):
        self._original.write(message)
        try:
            self._log_file.write(message)
        except (ValueError, OSError):
            pass
        return None

    def flush(self):
        self._original.flush()
        try:
            self._log_file.flush()
        except (ValueError, OSError):
            pass

    def close(self):
        self._log_file.close()

    def __getattr__(self, name):
        return getattr(self._original, name)


@contextmanager
def run_log(log_path):
    """Mirror stdout into ``log_path`` (appending) and restore it on exit, also on failure."""
    original = sys.stdout
    stream = TeeOutput(log_path, original)
    sys.stdout = stream
    try:
        yield stream
    finally:
        sys.stdout = original
        stream.flush()
        stream.close()
