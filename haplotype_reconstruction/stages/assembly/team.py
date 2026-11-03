"""Supply an explicit CPU team to refinement without exposing engine policy to science.

Owns scoped Numba masks and parallel template mapping. Must not change result order,
read conventions, set process environment, or make a scientific admission decision.
The surrounding task owns its declared memory and CPU reservation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from typing import Any, TypeVar, cast

import numba

from haplotype_reconstruction.core.kernels import enable_caches
from haplotype_reconstruction.science.assembly.refinement.capabilities import RefinementExecution

__all__ = ["RefinementTeam"]

_T = TypeVar("_T")
_R = TypeVar("_R")

# While a team's coordinating thread waits for its concurrent children or mapped templates, its boundary recurs this
# often: it re-reads the team's live share (growing into CPUs peers have freed, returning them when a peer is short),
# as 3.5's concurrent and thread_map re-read their budget while tasks ran. Without it a component that started beside
# busy peers kept its starting share for its whole refinement while the node emptied (chr8's last level: 424 s on
# about 8 of 112 CPUs).
SHARE_POLL_SECONDS = 0.2


class RefinementTeam:
    """The enclosing task's CPU share; parallel templates use one kernel thread each.

    A team changes only its calling thread's Numba mask inside scope, restoring it
    on success or failure. map preserves input order and joins workers on failure.
    """

    budget: Any = None  # the task's live allowance (None: the declared count; concurrent child teams have none)
    live = False  # whether cores is a live share (a budget's, or a concurrent child's); else the declared count
    _held: Callable[[], int] | None = None  # CPUs the team's running templates or children use (while one is open)

    def __init__(self, cores: int, budget: Any | None = None) -> None:
        enable_caches()
        self.declared = int(cores)
        self.budget = budget
        self.live = budget is not None

    @property
    def cores(self) -> int:
        """This team's CPUs now. Given the enclosing task's live allowance (``budget``, the engine's current_budget()),
        its share re-read at this boundary: it grows as peer tasks finish, as 3.5 re-read its lease at every boundary,
        instead of keeping the share the task started with (a component started beside many peers otherwise ran its
        whole refinement on one or two threads while their CPUs idled). Without one, the declared count. While
        templates or children run, the claim keeps the CPUs they use (CpuBudget.allowance: a running kernel cannot be
        stopped) and the count returned, which bounds what starts next, may already be smaller."""
        if self.budget is None:
            return self.declared
        return max(1, int(self.budget.allowance(self.holding())))

    def holding(self) -> int:
        """CPUs this team's running templates and concurrent children use now."""
        return 0 if self._held is None else self._held()

    @contextmanager
    def _holding(self, held: Callable[[], int]) -> Iterator[None]:
        """Count ``held`` in holding() while a map or concurrent of this team is open (added to any open one)."""
        previous = self._held
        self._held = held if previous is None else (lambda: previous() + held())
        try:
            yield
        finally:
            self._held = previous

    def threads(self) -> int:
        """This team's CPUs now (see cores)."""
        return self.cores

    @contextmanager
    def scope(self, cores: int | None = None) -> Iterator[None]:
        """Temporarily set this thread's kernel team, restoring it even on failure."""
        previous = numba.get_num_threads()
        numba.set_num_threads(self.cores if cores is None else cores)
        try:
            yield
        finally:
            # Never back above a live share now: it may have shrunk inside the scope.
            numba.set_num_threads(min(previous, self.cores) if self.live else previous)

    def map(self, function: Callable[[_T], _R], items: list[_T]) -> list[_R]:
        """Run independent templates in supplied order and return results in that order."""

        def run(item: _T) -> _R:
            # A fixed one-thread mask that restores without reading the live share: the map's coordinating thread alone
            # resizes the team (a template's exit read the share from a helper thread and could return CPUs while the
            # coordinator was still admitting templates on the old allowance).
            previous = numba.get_num_threads()
            numba.set_num_threads(1)
            try:
                return function(item)
            finally:
                numba.set_num_threads(previous)

        # Dispatch is cost-ordered, so a failing item can precede lower logical ordinals: every item runs (each
        # solver prunes only later work) and the failure of the lowest ordinal is raised after all have finished.
        outcomes: list[Any] = [None] * len(items)
        failed: dict[int, BaseException] = {}
        if len(items) <= 1:
            for index, item in enumerate(items):
                try:
                    outcomes[index] = run(item)
                except Exception as error:
                    failed[index] = error
        else:
            # As many templates run at once as the team's live share, re-read at every completion and every
            # SHARE_POLL_SECONDS: templates start as the share grows and none starts while it is exceeded.
            pending = list(enumerate(items))
            pending.reverse()
            running: dict[Any, int] = {}
            with (
                self._holding(lambda: len(running)),
                ThreadPoolExecutor(max_workers=min(len(items), cast(Any, numba.config).NUMBA_NUM_THREADS)) as pool,
            ):
                while pending or running:
                    allowed = max(1, self.cores)
                    while pending and len(running) < allowed:
                        index, item = pending.pop()
                        running[pool.submit(run, item)] = index
                    done, _ = wait(tuple(running), timeout=SHARE_POLL_SECONDS, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = running.pop(future)
                        try:
                            outcomes[index] = future.result()
                        except Exception as error:
                            failed[index] = error
        if failed:
            raise failed[min(failed)]
        return outcomes

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
        if len(functions) == 1:  # the only child holds the whole team
            with self.scope():
                value = functions[0](self)
            yield 0, value
            return
        # The children split this team's LIVE count, which this coordinating thread re-reads at every completion and
        # every SHARE_POLL_SECONDS while it waits (its boundary) and publishes to them (shares.limit): the team grows
        # into freed CPUs and returns them when a peer is short, and a nested team grows when its own siblings finish
        # (3.5's `_Slots.allocation` re-read the enclosing budget at every boundary). The team's claim keeps the CPUs
        # its running children hold until they shrink at their own boundaries or finish.
        shares = _Shares(self.cores)

        def run(item: tuple[int, Callable[[RefinementExecution], _R]]) -> tuple[int, _R]:
            index, function = item
            shares.start(index)
            try:
                child = _ChildTeam(shares, index)
                with child.scope():
                    return index, function(child)
            finally:
                shares.finish(index)

        # Children start in index order as the share allows: each admitted child's first CPU is reserved under the
        # shares' lock (one no running child holds) before any of them is submitted, so the claims never sum above the
        # published share.
        pending = list(enumerate(functions))
        pending.reverse()
        futures: dict[int, Any] = {}

        def admit() -> None:
            # The hold the share is read with, the share and its publication form one step: no child grows on the old
            # limit in between (the shares' lock is reentrant: holding() re-reads the claims under it).
            with shares.lock:
                shares.limit = self.cores
            starting = []
            while pending and shares.reserve(pending[-1][0]):
                starting.append(pending.pop())
            for position, item in enumerate(starting):
                try:
                    futures[item[0]] = pool.submit(run, item)
                except BaseException:
                    # The executor cannot start work (no thread): children never attempted cannot run, so their
                    # reservations go back and admission stops. The failed one may already be queued and run on an
                    # existing thread: it keeps its reservation until the pool is joined. Submitted children are joined.
                    for index, _ in starting[position + 1 :]:
                        shares.finish(index)
                    pending.clear()
                    raise

        def wait_running() -> None:
            running = [future for future in futures.values() if not future.done()]
            if running:
                wait(running, timeout=SHARE_POLL_SECONDS, return_when=FIRST_COMPLETED)

        with (
            self._holding(shares.held),
            ThreadPoolExecutor(max_workers=min(len(functions), cast(Any, numba.config).NUMBA_NUM_THREADS)) as pool,
        ):
            try:
                for index in range(len(functions)):
                    admit()
                    while not (index in futures and futures[index].done()):
                        wait_running()
                        admit()
                    yield futures.pop(index).result()
            finally:
                # Every child runs, also when an earlier one failed or the consumer stopped early: the rest still start
                # as the share allows, and the pool joins them all; their results are discarded. (Only an executor
                # that cannot start a thread abandons the children it never submitted.)
                while pending:
                    admit()
                    wait_running()


class _Shares:
    """Live fair shares of one team among its running concurrent children (3.5 `_Slots.allocation`).

    A child's share is the team's live CPU count (``limit``, published by the team's coordinating thread at its
    boundaries) divided among the running children, the remainder to the lowest indices; for a nested team that count
    is its own live share, so nested children grow as their parent's peers finish.
    A child above its share returns the excess at its own boundary, never below the CPUs its own running children use
    (``hold``); below it, it takes only CPUs no peer holds.
    A child is admitted by reserving its first CPU (reserve) before it is submitted, so an early starter never takes
    the CPUs of children admitted with it; while no CPU is unclaimed, the next child waits and counts as a contender,
    so running children above their share return CPUs for it (the engine's `_Slots.admit` and `waiting`).
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self.claims: dict[int, int] = {}
        self.waiting = 0
        self.lock = threading.RLock()

    def held(self) -> int:
        """CPUs the running children hold."""
        with self.lock:
            return sum(self.claims.values())

    def reserve(self, index: int) -> bool:
        """Claim one CPU for child ``index`` if one is unclaimed (always when no child holds any) and return True;
        else count it as the waiting contender and return False."""
        with self.lock:
            cores = max(1, int(self.limit))
            if self.claims and sum(self.claims.values()) >= cores:
                self.waiting = 1
                return False
            self.claims[index] = 1
            self.waiting = 0
            return True

    def start(self, index: int) -> None:
        with self.lock:
            self.claims.setdefault(index, 1)

    def finish(self, index: int) -> None:
        with self.lock:
            self.claims.pop(index, None)

    def allocation(self, index: int, hold: int = 0) -> int:
        """The child's CPUs from now to its next boundary; its claim keeps at least ``hold`` of its current CPUs (what
        its own running children use)."""
        with self.lock:
            cores = max(1, int(self.limit))
            count = max(1, len(self.claims)) + self.waiting
            rank = sum(other < index for other in self.claims)
            want = max(1, cores // count + int(rank < cores % count))
            mine = self.claims.get(index, 1)
            free = cores - sum(value for other, value in self.claims.items() if other != index)
            claim = max(1, min(want, free)) if want < mine else max(mine, min(want, free))
            self.claims[index] = max(claim, min(int(hold), mine))
            return claim


class _ChildTeam(RefinementTeam):
    """A concurrent child's team: its CPU count is its live share, re-read at every boundary."""

    live = True

    def __init__(self, shares: _Shares, index: int) -> None:
        enable_caches()
        self._shares = shares
        self._index = index

    @property
    def cores(self) -> int:  # a live share, read-only by design
        return self._shares.allocation(self._index, self.holding())
