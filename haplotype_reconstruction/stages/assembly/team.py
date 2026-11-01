"""Supply an explicit CPU team to refinement without exposing engine policy to science.

Owns scoped Numba masks and parallel template mapping. Must not change result order,
read conventions, set process environment, or make a scientific admission decision.
The surrounding task owns its declared memory and CPU reservation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TypeVar

import numba

from haplotype_reconstruction.core.kernels import enable_caches
from haplotype_reconstruction.science.assembly.refinement.capabilities import RefinementExecution

__all__ = ["RefinementTeam"]

_T = TypeVar("_T")
_R = TypeVar("_R")


class RefinementTeam:
    """Fixed task CPU share; parallel templates use one kernel thread each.

    A team changes only its calling thread's Numba mask inside scope, restoring it
    on success or failure. map preserves input order and joins workers on failure.
    """

    def __init__(self, cores: int) -> None:
        enable_caches()
        self.cores = int(cores)

    def threads(self) -> int:
        """Return the declared CPU count; no live system resources are inspected."""
        return self.cores

    @contextmanager
    def scope(self, cores: int | None = None) -> Iterator[None]:
        """Temporarily set this thread's kernel team, restoring it even on failure."""
        previous = numba.get_num_threads()
        numba.set_num_threads(self.cores if cores is None else cores)
        try:
            yield
        finally:
            numba.set_num_threads(previous)

    def map(self, function: Callable[[_T], _R], items: list[_T]) -> list[_R]:
        """Run independent templates in supplied order and return results in that order."""

        def run(item: _T) -> _R:
            with self.scope(1):
                return function(item)

        if self.cores == 1:
            # Dispatch is cost-ordered, so a failing item can precede lower logical
            # ordinals. Drain the bound batch; each solver prunes only later work.
            results = []
            failure = None
            for item in items:
                try:
                    results.append(run(item))
                except Exception as error:
                    if failure is None:
                        failure = error
            if failure is not None:
                raise failure
            return results
        # Executor.map's result iterator cancels queued futures after its first
        # exception. Submit explicitly and drain every future: a cost-ordered
        # failure may precede an earlier logical ordinal that must still run.
        results = []
        failure = None
        with ThreadPoolExecutor(max_workers=self.cores) as pool:
            futures = [pool.submit(run, item) for item in items]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as error:
                    if failure is None:
                        failure = error
        if failure is not None:
            raise failure
        return results

    def thread_map(
        self,
        function: Callable[[_T], _R],
        items: Iterable[_T],
        *,
        item_bytes: Sequence[int] | None = None,
        label: str = "",
    ) -> list[_R]:
        """Compatibility during relocation: ordered one-thread template tasks.

        Byte estimates accompany the enclosing task's reservation; the numerical
        solver cannot make live-memory admission decisions.
        """
        return self.map(function, list(items))

    @contextmanager
    def reserve(self, size: int, *, label: str = "") -> Iterator[None]:
        """Use scratch already covered by the enclosing component's reservation."""
        yield

    def concurrent(
        self,
        functions: Sequence[Callable[[RefinementExecution], _R]],
        sizes: int | Sequence[int] = 0,
        *,
        label: str = "",
    ) -> Iterator[tuple[int, _R]]:
        """Run ordered child tasks on live fair shares, joining every child on failure.

        Each running child holds its fair share of this team among the running children (3.5's slot rule,
        execution/budget.py `_Slots.allocation`): survivors take the CPUs of finished peers at their next
        boundary, and a child only ever takes free CPUs, so the claims never sum above the team. Kernel thread
        counts never decide a result (refinement is exact across 1/4/16 threads), so shares are a schedule, not
        science. Results are yielded in index order.
        """
        functions = list(functions)
        if not functions:
            return
        width = min(self.cores, len(functions))
        # The children split this team's LIVE count: a nested team grows when its own siblings finish (3.5's
        # `_Slots.allocation` reads `budget.threads()` of the enclosing budget at every boundary).
        shares = _Shares(lambda: self.cores, range(width))

        def run(item: tuple[int, Callable[[RefinementExecution], _R]]) -> tuple[int, _R]:
            index, function = item
            shares.start(index)
            try:
                child = _ChildTeam(shares, index)
                with child.scope():
                    return index, function(child)
            finally:
                shares.finish(index)

        if width == 1:
            for item in enumerate(functions):
                yield run(item)
        else:
            # Shutdown joins all submitted fits even if the consumer raises on
            # an earlier result. No result iterator may cancel queued fits.
            with ThreadPoolExecutor(max_workers=width) as pool:
                futures = [pool.submit(run, item) for item in enumerate(functions)]
                for future in futures:
                    yield future.result()


class _Shares:
    """Live fair shares of one team among its running concurrent children (3.5 `_Slots.allocation`).

    A child's share is the team's live CPU count divided among the running children, the remainder to the lowest
    indices; for a nested team that count is its own live share, so nested children grow as their parent's peers
    finish.
    A child above its share returns the excess at its own boundary; below it, it takes only CPUs no peer holds.
    The first `width` children are registered before any starts, so an early starter never takes their CPUs.
    """

    def __init__(self, total: Callable[[], int], admitted: Iterable[int]) -> None:
        self.total = total
        self.claims = {int(index): 1 for index in admitted}
        self.lock = threading.Lock()

    def start(self, index: int) -> None:
        with self.lock:
            self.claims.setdefault(index, 1)

    def finish(self, index: int) -> None:
        with self.lock:
            self.claims.pop(index, None)

    def allocation(self, index: int) -> int:
        """The child's CPUs from now to its next boundary."""
        cores = max(1, int(self.total()))  # the enclosing team's live count, read outside this lock
        with self.lock:
            count = max(1, len(self.claims))
            rank = sum(other < index for other in self.claims)
            want = max(1, cores // count + int(rank < cores % count))
            mine = self.claims.get(index, 1)
            free = cores - sum(value for other, value in self.claims.items() if other != index)
            claim = max(1, min(want, free)) if want < mine else max(mine, min(want, free))
            self.claims[index] = claim
            return claim


class _ChildTeam(RefinementTeam):
    """A concurrent child's team: its CPU count is its live share, re-read at every boundary."""

    def __init__(self, shares: _Shares, index: int) -> None:
        enable_caches()
        self._shares = shares
        self._index = index

    @property
    def cores(self) -> int:  # type: ignore[override]  # a live share, read-only by design
        return self._shares.allocation(self._index)
