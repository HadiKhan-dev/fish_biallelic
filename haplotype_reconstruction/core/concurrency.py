"""Ordered maps for independent items, supplied by the caller to science and readers.

A scientific ``map_work(function, items)`` has Python map's input/output ordering and
propagates item exceptions. Its caller owns the concurrency bound and Numba teams;
science consumes its results before reducing values or releasing shared scratch.
Python's ``map`` is the serial default. Stages may pass ``threaded_each`` to use the
engine's budgeted map, whose concurrent items each have one CPU and one Numba thread.
Nested maps stay within that one-CPU budget. Single-item calls use the calling thread;
without the engine, ``threaded_each`` is serial.

A ``ThreadSource`` is a count or a caller-owned callable, read with
``thread_count`` at numerical work boundaries. A live source may refresh the
caller's ambient kernel team as well as report its count; the caller owns that
policy and refresh cadence. An integer only describes the team/layout and never
configures it. Forward sources unchanged through whole searches.

``live_threads()`` returns the caller's engine CPU allowance, or 1 without the engine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

__all__ = ["ThreadSource", "thread_count", "live_threads", "threaded_each"]

ThreadSource = int | Callable[[], int]


def thread_count(source: ThreadSource) -> int:
    """Read a fixed or live caller-owned thread count at this work boundary."""
    return max(1, int(source() if callable(source) else source))


# Installed at the engine runtime boundary: (function, items) -> [function(item) ...] in item order on the caller's budget.
ordered_map: Callable[[Callable[[Any], Any], list[Any]], list[Any]] | None = None
# Installed at the engine runtime boundary: () -> the calling thread's live CPU allowance.
allowance: Callable[[], int] | None = None


def live_threads() -> int:
    """The CPUs the caller may use now (its engine share), at least 1; 1 without the engine."""
    return 1 if allowance is None else max(1, int(allowance()))


def threaded_each(function: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
    """function(item) for every item, results in item order: on the engine's budgeted map when installed."""
    items = list(items)
    if len(items) <= 1 or ordered_map is None:
        return [function(item) for item in items]
    return ordered_map(function, items)
