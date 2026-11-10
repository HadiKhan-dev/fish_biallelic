"""Atomic file publication with distinct, exception-safe staging for each writer."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path


def temporary_path(path: Path) -> Path:
    """A hidden temporary beside ``path``, unique per writer and call.

    Duplicate workflow steps may write the same file at once; a shared temporary
    name would let one writer rename the other's file away. Each writer publishes
    its own contents through an atomic rename.
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
def placement_lock(path: Path):
    """Serialise the placement of ``path`` and its companion files among writers sharing its filesystem.

    Placing a file with an index is several renames, and a writer that fails part-way restores the previous pair.
    Without serialisation, a failed duplicate writer could restore older data beside another writer's newly placed
    index. Holds an exclusive lock on a hidden file beside ``path`` while files are moved. The lock file
    is coordination metadata, not a product: it holds no state and is left in place (removing it could let a waiting
    writer and a newcomer lock different files).
    """
    import fcntl

    with placement_lock_path(path).open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def placement_lock_path(path: Path) -> Path:
    """The lock file placement_lock uses for ``path``."""
    path = Path(path)
    return path.with_name(f".{path.name}.lock")


def is_placement_lock(entry: Path, suffixes: tuple[str, ...]) -> bool:
    """Whether ``entry`` is an (empty) placement_lock file of a product whose name ends in one of ``suffixes``."""
    name = entry.name
    return (
        name.startswith(".")
        and any(name.endswith(f"{suffix}.lock") for suffix in suffixes)
        and entry.is_file()
        and entry.stat().st_size == 0
    )


@contextmanager
def atomic_path(path: Path):
    """Yield a private neighbour, rename on success, and clean up on any exception."""
    path = Path(path)
    with staged_path(path) as temporary:
        yield temporary
        temporary.replace(path)
