"""Atomic file publication with distinct, exception-safe staging for each writer."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path


def temporary_path(path: Path) -> Path:
    """A hidden temporary beside ``path``, unique per writer and call.

    Duplicate workflow steps may write the same file at once; a shared temporary
    name would let one writer rename the other's file away. Final names, contents
    and the atomic rename are unchanged.
    """
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


@contextmanager
def staged_path(path: Path):
    """Yield a unique neighbour and remove it on exit, including interrupted writes.

    Used directly when publication also needs an index; ordinary writers use
    ``atomic_path``. Handles must close inside this context before placement.
    """
    temporary = temporary_path(Path(path))
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def atomic_path(path: Path):
    """Yield a private neighbour, rename on success, and clean up on any exception."""
    path = Path(path)
    with staged_path(path) as temporary:
        yield temporary
        temporary.replace(path)
