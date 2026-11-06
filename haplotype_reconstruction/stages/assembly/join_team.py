"""Lend a join's CPUs to independent refits, then restore logical result order.

The engine owns every child share. A sample kernel uses its useful sample width;
that width does not limit the number of independent refits the task can execute.
"""

from collections.abc import Callable, Iterable
from typing import TypeVar, cast

from haplotype_reconstruction.engine.scheduler import claim_threads, current_budget, kernel_team

_T = TypeVar("_T")
_R = TypeVar("_R")


def task_cpu_ceiling(default: int) -> int:
    """Preserve the enclosing task's declared maximum across stage expansion.

    The ``cores`` argument injected into a running graph action is its initial
    share. It must not become the permanent ceiling of every later assembly
    child. Standalone callers keep their explicitly supplied maximum.
    """
    return max(int(default), int(getattr(current_budget(), "maximum", default)))


class JoinTeam:
    """The current engine child authority, capped only at a kernel boundary."""

    def __init__(self, sample_width: int) -> None:
        self.sample_width = max(1, int(sample_width))

    def threads(self) -> int:
        # Refit children consult their own authority. A captured parent budget
        # could claim their siblings' CPUs at a nested kernel boundary.
        return claim_threads(max_threads=self.sample_width)

    def thread_map(self, function: Callable[[_T], _R], items: Iterable[_T]) -> list[_R]:
        items = list(items)
        results: list[_R | None] = [None] * len(items)

        def run(item: _T) -> _R:
            with kernel_team(max_threads=self.sample_width):
                return function(item)

        for index, result in current_budget().concurrent(
            [lambda child, item=item: run(item) for item in items], label="assembly join refits"
        ):
            results[index] = result
        return cast(list[_R], results)
