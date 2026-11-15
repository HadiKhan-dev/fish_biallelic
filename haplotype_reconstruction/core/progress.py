"""The partial-progress capability a stage hands to a long science loop (implemented by store.progress.UnitProgress).

A science layer resumes from the frame it saved (``take``), registers a reader of its live state while inner
layers run (``frame``), and the innermost loop offers its complete state at each of its boundaries
(``boundary``); the capability decides when a snapshot is written. Science never sees files or clocks.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol

__all__ = ["Progress", "frame"]


class Progress(Protocol):
    def take(self, name: str) -> Any | None: ...
    def frame(self, name: str, state: Callable[[], Any]) -> AbstractContextManager[None]: ...
    def boundary(self, name: str, state: Callable[[], Any]) -> bool: ...


def frame(progress: Progress | None, name: str, state: Callable[[], Any]) -> AbstractContextManager[None]:
    """``progress.frame(name, state)``, or nothing without a progress capability."""
    return nullcontext() if progress is None else progress.frame(name, state)
