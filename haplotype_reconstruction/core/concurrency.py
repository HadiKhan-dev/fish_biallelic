"""Independent record reads of several containers, results in item order (release 4.10).

Checks that derive and verify publications contig after contig (identity chains, completion checks) spend their time
opening, reading and decoding record headers on a shared filesystem. They are independent per item. With the engine
loaded (a run's coordinator and its workers), ``ordered_map`` is the engine's budgeted one-thread map on the caller's
CPU budget (installed by engine.scheduler): a task's reads stay within its claim, a failure propagates as from any task
map, and a map inside one of its items gets a one-CPU budget and runs serially, so nested checks never multiply
threads. Without the engine (a library reader), the items run one after another.

``live_threads()`` is the CPUs the caller may use now: its engine share (installed by engine.scheduler; e.g. an output
writer's compression threads), 1 without the engine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

__all__ = ["live_threads", "threaded_each"]

# Installed by engine.scheduler: (function, items) -> [function(item) ...] in item order on the caller's budget.
ordered_map: Callable[[Callable[[Any], Any], list[Any]], list[Any]] | None = None
# Installed by engine.scheduler: () -> the calling thread's live CPU allowance.
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
