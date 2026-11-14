"""Own CPU budgets, seats, leases and thread-local team claims."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError
from contextlib import contextmanager

from .graph import TaskKey
from .memory import executing_work

_LOG = logging.getLogger(__name__)


CPU_POLL_SECONDS = 0.2


def _numba():
    import numba

    return numba


def _set_team(n: int) -> int:
    """Set the calling thread's Numba team to ``n``."""
    numba = _numba()
    n = max(1, int(n))
    if numba.get_num_threads() != n:
        numba.set_num_threads(n)
    return n


@contextmanager
def thread_team(n: int | None = None, *, cap: int | None = None, default: int | None = None):
    """Hold an explicit team or refresh the live claim for one native section.

    With default, a task uses its bound budget and a standalone caller uses that
    default. Without a count/default, refresh the current claim, capped by useful
    width. Restoration cannot reclaim CPUs returned at a nested boundary.
    """
    from ..core.kernels import enable_caches

    if default is not None:
        enable_caches()
    old = _numba().get_num_threads()
    budget = thread_budget() if default is not None else None
    selected = n if n is not None else (budget.threads() if budget is not None else default)
    team = claim_threads(max_threads=cap) if selected is None else _set_team(selected)
    try:
        yield team
    finally:
        restore = old if budget is None else max(1, min(old, int(budget.threads())))
        ceiling = _team_ceiling()
        _set_team(restore if ceiling is None else max(1, min(restore, ceiling)))


class CpuBudget:
    """A CPU allowance (see the module docstring). Picklable: a live allowance pickles as its current value."""

    live = False

    @staticmethod
    def fixed(n: int) -> FixedBudget:
        return FixedBudget(n)

    def threads(self) -> int:
        """The current allowance (>= 1), re-read now; a lease may grow on each call."""
        raise NotImplementedError

    def allowance(self, held: int) -> int:
        """threads() at the boundary of a coordinator whose running children (map items, concurrent tasks) use
        ``held`` CPUs: a live claim keeps at least those reserved until the children finish or shrink at their own
        boundaries (a running kernel cannot be stopped), while the returned allowance, which bounds what the
        coordinator admits from now, may already be smaller. Budgets that return CPUs only at their own boundaries
        (or never) need no hold."""
        return self.threads()

    def phase(self, name: str) -> CpuBudget:
        """Mark the start of phase ``name`` (a boundary: a lane rebalances its lease); return the budget."""
        return self

    def current(self) -> int:
        """The allowance as it stands, without claiming anything (what a pickled copy carries)."""
        return self.threads()

    def __reduce__(self):
        return (FixedBudget, (max(1, int(self.current())),))

    def concurrent(self, functions, *, label: str = "concurrent"):
        """Run nested concurrent children with CPU-seat-before-admission ordering."""
        from .threadmaps import completed_tasks

        return completed_tasks(self, list(functions), label)

    def thread_map(self, function, items, *, label: str = "thread map") -> list:
        """Run one-thread nested calls, placing values in input order."""
        from .threadmaps import one_thread_map

        return one_thread_map(self, function, list(items), label)


class FixedBudget(CpuBudget):
    """A fixed count of threads."""

    def __init__(self, n: int):
        n = int(n)
        if n < 1:
            raise ValueError("a CPU budget must be at least one thread")
        self.n = n

    def threads(self) -> int:
        return self.n

    def __reduce__(self):
        return (FixedBudget, (self.n,))

    def __repr__(self):
        return f"CpuBudget.fixed({self.n})"


def read_counter(value):
    """Lock-free read of a shared counter (an aligned int read is atomic; a slightly stale value is fine)."""
    return value.get_obj().value if hasattr(value, "get_obj") else value.value


class PoolCounters:
    """One lane's shared CPU counters, created by the coordinator and passed to its workers."""

    lease_state: LeaseState
    lease_index: int

    def __init__(self, context, total: int, processes: int):
        value = context.Value
        self.total = value("i", int(total))
        self.active = value("i", 0)
        self.claimed = value("i", 0)
        self.admission = context.Condition(self.claimed.get_lock())
        self.waiting = value("i", 0, lock=False)
        self.short = value("i", 0, lock=False)
        self.extra = value("i", 0)
        self.started = value("i", 0)
        self.participants = value("i", 0)
        self.generation = value("i", 0)
        self.task_count = value("i", 0)
        self.startup_target = value("i", min(1, processes))
        self.startup_ready = value("i", 0)
        self.closed = value("i", 0)
        # Graph tasks sent to this pool's workers (written only by the coordinator) and admitted by them (under
        # admission): the difference is tasks on their way, contenders before their workers begin them.
        self.sent = value("i", 0)
        self.begun = value("i", 0, lock=False)
        # Begun tasks that have not yet taken their first owning share (under admission): until they have, their
        # fair entitlements are unclaimed, not free, so no peer borrows them (PoolSeat.share).
        self.unshared = value("i", 0, lock=False)

    @staticmethod
    def _store(counter, number):
        # Aligned assignments are atomic. The coordinator is the sole resetting
        # writer; workers only open startup_ready (write 1) within a batch.
        # Never acquire a semaphore a dead worker could have left locked.
        counter.get_obj().value = int(number)

    def start_batch(self, tasks: int, processes: int) -> None:
        """Open a new batch (generation): reset the start-up gate for ``tasks`` tasks. The gate waits for as many
        task starts as the pool can admit at once, bounded by tasks, processes and CPUs.
        Counting only concurrent starts lets borrowing begin without waiting for queued tasks to start."""
        if read_counter(self.active):
            raise RuntimeError("pool workers of the preceding batch are still active")
        target = min(int(tasks), int(processes), max(1, read_counter(self.total)))
        self._store(self.started, 0)
        self._store(self.participants, 0)
        self._store(self.task_count, tasks)
        self._store(self.startup_target, target)
        self._store(self.startup_ready, target <= 1)
        self.generation.get_obj().value += 1

    def set_total(self, threads: int) -> None:
        """Follow the live parent budget both ways. A raise wakes workers waiting for a CPU; a cut applies at the
        running tasks' own boundaries (share() then returns only CPUs within the new total; claims are never
        revoked inside a kernel) and holds back new task starts until the claims fit."""
        threads = max(1, int(threads))
        if not self.admission.acquire(timeout=CPU_POLL_SECONDS):
            return
        try:
            old = read_counter(self.total)
            if threads != old:
                self._store(self.total, threads)
                if threads > old:
                    self.admission.notify_all()
        finally:
            self.admission.release()

    def dispatch_contender(self) -> None:
        """Count a task the coordinator is sending to a worker as a contender until its worker begins it (begin_task
        with dispatched=True): siblings sent together each count from the start, so the first to begin takes its fair
        share instead of borrowing CPUs its siblings need before the borrower's next boundary.
        The coordinator is the counter's only writer: no lock is taken."""
        self._store(self.sent, read_counter(self.sent) + 1)

    def withdraw_contender(self) -> None:
        """Undo one dispatch_contender for a task the coordinator no longer expects to send soon (coordinator only)."""
        self._store(self.sent, read_counter(self.sent) - 1)

    def arriving(self) -> int:
        """Tasks counted as on their way to this pool's workers (sent, or expected by a free worker) that have not
        begun yet."""
        return max(0, read_counter(self.sent) - self.begun.value)

    def open_gate(self) -> None:
        """Open the current batch's start-up gate now (memory holds the remaining tasks back)."""
        self._store(self.startup_ready, 1)

    def close(self) -> None:
        """Publish cancellation without waiting on a possibly dead CPU-lock holder."""
        self.closed.get_obj().value = 1
        if self.admission.acquire(False):
            try:
                self.admission.notify_all()
            finally:
                self.admission.release()


class PoolSeat:
    """A pool worker's view of its pool's shares (one per worker process)."""

    def __init__(self, counters: PoolCounters):
        self.counters = counters
        self.has_extra = False
        self.claimed = 0
        self.peak = 0  # the most CPUs the running task has held (run-record diagnostics)
        self.first_share = False  # the running task has not yet taken its first owning share (counted in unshared)
        self.short = 0
        self.task_thread: int | None = None
        self.last_generation: int | None = None
        self.last_applied: int | None = None
        self.last_logged: int | None = None

    def begin_task(self, minimum: int = 1, *, dispatched: bool = False) -> None:
        """Reserve one CPU before admission (waiting meanwhile counts as a contender, so peers above their fair
        share give back at their boundaries), then register the task with the start-up gate. ``dispatched``: the
        graph coordinator counted this task as arriving when it sent it (dispatch_contender); it now waits instead."""
        c = self.counters
        with c.admission:
            if dispatched:
                c.begun.value += 1  # from now on it counts as waiting, not as arriving
            c.waiting.value += 1
            try:
                while read_counter(c.claimed) + minimum > read_counter(c.total):
                    c.admission.wait()
            finally:
                c.waiting.value -= 1
            c.claimed.get_obj().value += minimum
            self.claimed = self.peak = minimum
            self.first_share = True
            c.unshared.value += 1
            self.task_thread = threading.get_ident()
            with c.active.get_lock():
                c.active.get_obj().value += 1
        with c.started.get_lock():
            c.started.get_obj().value += 1
            started = c.started.get_obj().value
        generation = read_counter(c.generation)
        if self.last_generation != generation:
            with c.participants.get_lock():
                c.participants.get_obj().value += 1
                participants = c.participants.get_obj().value
            self.last_generation = generation
        else:
            participants = read_counter(c.participants)
        if participants >= read_counter(c.startup_target) or started >= read_counter(c.task_count):
            with c.startup_ready.get_lock():
                c.startup_ready.get_obj().value = 1

    def end_task(self) -> None:
        """Release this finished task's claims and wake workers waiting for admission."""
        c = self.counters
        with c.admission:
            self._release_extra()
            c.claimed.get_obj().value -= self.claimed
            c.short.value -= self.short
            if self.first_share:
                c.unshared.value -= 1
                self.first_share = False
            self.claimed = self.short = 0
            self.task_thread = None
            with c.active.get_lock():
                c.active.get_obj().value -= 1
            c.admission.notify_all()

    def _total(self) -> int:
        """The CPUs this seat's tasks share: the pool's total."""
        c = self.counters
        return max(1, read_counter(c.total))

    def owns_team(self) -> bool:
        """Whether the calling thread owns this task's claim; nested work publishes explicit holds."""
        return self.task_thread == threading.get_ident()

    def _claim_extra(self, remainder: int) -> None:
        if self.has_extra:
            return
        extra = self.counters.extra
        with extra.get_lock():
            if extra.get_obj().value < remainder:
                extra.get_obj().value += 1
                self.has_extra = True

    def _release_extra(self) -> None:
        if not self.has_extra:
            return
        extra = self.counters.extra
        with extra.get_lock():
            extra.get_obj().value -= 1
        self.has_extra = False

    def share(self, max_threads: int | None = None, *, give_back: bool = True, hold: int = 0) -> int:
        """This task's CPUs from now to its next boundary (claims change only at boundaries and at task start and
        end, never inside a running kernel; the pool's claims never exceed its total).

        Fair share: floor(total / contenders), plus one of the remainder CPUs while this task holds a remainder
        claim; contenders are the running tasks and the workers waiting for their first CPU. ``max_threads``
        (what the task can use) caps it. While no peer is short and no worker waits, a task also takes the free
        CPUs (work-conserving); once a peer is short, a task above its fair share returns the excess here
        (``give_back``; False for queries from threads that do not own the task's team). Borrowing waits for the
        batch's start-up gate and for every other begun task's first owning share. ``hold``: CPUs the task's running
        children still use (CpuBudget.allowance); a give-back keeps them claimed, and the smaller allowance is
        returned.
        """
        if max_threads is not None:
            max_threads = int(max_threads)
            if max_threads < 1:
                raise ValueError("max_threads must be a positive integer or None")
        c = self.counters
        with c.admission:
            if hasattr(c, "lease_state") and give_back:
                lease = LaneLease(c.lease_state, c.lease_index)
                _, total = lease.grow(floor=read_counter(c.claimed), starting=self.first_share)
                c.set_total(total)
            if not self.claimed:
                return 1
            total = self._total()
            # Fair shares count every contender: running tasks, tasks waiting for a first CPU and tasks the coordinator
            # has counted as arriving (a pass's sends and its free workers' next tasks). So a batch that is still
            # starting can claim fair shares immediately; its start-up gate only withholds borrowing.
            contenders = max(read_counter(c.active), 1) + c.waiting.value + c.arriving()
            floor, remainder = divmod(total, contenders)
            if max_threads is not None and max_threads <= floor:
                self._release_extra()
            else:
                extras = read_counter(c.extra)
                if not self.has_extra and extras < remainder:
                    self._claim_extra(remainder)
                elif self.has_extra and extras > remainder:
                    self._release_extra()
            want = fair_share(total, contenders, 0 if self.has_extra else contenders)
            if max_threads is not None:
                want = min(want, max_threads)
            target = want
            if not give_back:
                target = max(target, self.claimed)
            others = read_counter(c.claimed) - self.claimed
            available = read_counter(c.total) - others
            if (
                give_back
                and read_counter(c.startup_ready)
                and not c.waiting.value
                and not c.arriving()
                and c.unshared.value == int(self.first_share)
                and read_counter(c.short) - self.short <= 0
            ):
                # Work-conserving borrowing: while no peer is short of its fair share, no
                # worker waits for a first CPU and every other begun task has taken its first share (until then its
                # entitlement is unclaimed, not free: a borrower could not return it before its next boundary), this
                # task also takes the pool's free CPUs (up to what it can use), at every boundary of a step, not only
                # at its edges. A peer that comes up short gets them back at this task's next boundary. Thread counts
                # never decide a result.
                target = max(target, available if max_threads is None else min(available, max_threads))
            allocated = max(1, min(target, available)) if give_back else max(self.claimed, min(target, available))
            held = max(allocated, min(int(hold), self.claimed))
            change = held - self.claimed
            c.claimed.get_obj().value += change
            self.claimed = held
            self.peak = max(self.peak, held)
            short = max(0, want - held)
            c.short.value += short - self.short
            self.short = short
            if change < 0:
                c.admission.notify_all()
            if give_back and self.first_share:
                c.unshared.value -= 1
                self.first_share = False
            return allocated

    def apply(self, max_threads: int | None = None) -> int:
        """Recompute the share at this boundary and apply it to the calling thread's Numba team; return it."""
        numba = _numba()
        n = min(self.share(max_threads, give_back=self.owns_team()), numba.config.NUMBA_NUM_THREADS)
        current = int(numba.get_num_threads())
        if self.last_applied != n or current != n:
            numba.set_num_threads(n)
            current = int(numba.get_num_threads())
        self.last_applied = current
        self._trace(current)
        return current

    def _trace(self, n: int) -> None:
        from .runtime import profiling

        if not profiling() or n == self.last_logged:
            return
        self.last_logged = n
        c = self.counters
        _LOG.info(
            f"[threads pid={os.getpid()} t={time.monotonic():.1f}] active={read_counter(c.active)} "
            f"total={read_counter(c.total)} extras={read_counter(c.extra)} mine={self.has_extra} -> {n}"
        )


class PoolShareBudget(CpuBudget):
    """The calling pool worker's live share (see PoolSeat.share)."""

    live = True

    def threads(self) -> int:
        return self.allowance(0)

    def allowance(self, held: int) -> int:
        seat = _POOL_SEAT
        if seat is None:
            return 1
        if not seat.owns_team():
            return seat.share(give_back=False)
        n = seat.share(hold=held)
        numba = _numba()
        if numba.get_num_threads() > n:
            numba.set_num_threads(n)
        return n

    def current(self) -> int:
        seat = _POOL_SEAT
        return 1 if seat is None or seat.last_applied is None else seat.last_applied

    def __repr__(self):
        return "PoolShareBudget()"


_POOL_SEAT: PoolSeat | None = None


def set_pool_seat(seat: PoolSeat) -> None:
    """Select this worker's seat when it takes a task from a lane."""
    global _POOL_SEAT
    _POOL_SEAT = seat


def pool_seat() -> PoolSeat | None:
    return _POOL_SEAT


def _water_fill(shares: list[int], demands: list[int]) -> list[int]:
    """Lane shares by demand: each lane gets its base share (equal, or the straggler floor) but never more than its
    demand (the summed widths of its ready and running tasks; 0: unknown, no limit); what a lane cannot use goes to
    the lanes that can, in proportion to their base shares, until every CPU is placed or every demand is met."""
    shares = list(shares)
    fixed = [False] * len(shares)
    while True:
        capped = [k for k in range(len(shares)) if not fixed[k] and 0 < demands[k] < shares[k]]
        if not capped:
            return shares
        leftover = sum(shares[k] - demands[k] for k in capped)
        for k in capped:
            shares[k], fixed[k] = demands[k], True
        open_ = [k for k in range(len(shares)) if not fixed[k]]
        if not open_:
            return shares
        for k, extra in zip(open_, _spread(leftover, [shares[k] for k in open_]), strict=True):
            shares[k] += extra


def _spread(extra: int, weights) -> list[int]:
    """Split ``extra`` CPUs (zero or more) in proportion to ``weights`` (largest remainders, ties to the lower
    index)."""
    weights = [max(0.0, float(w)) for w in weights]
    total = sum(weights) or float(len(weights))
    exact = [extra * (w if sum(weights) else 1.0) / total for w in weights]
    result = [int(x) for x in exact]
    for i in sorted(range(len(exact)), key=lambda i: (int(exact[i]) - exact[i], i))[: extra - sum(result)]:
        result[i] += 1
    return result


def floor_shares(budget: int, weights, total: float) -> list[int]:
    """Equal shares of ``budget`` CPUs among workers, with a proportional floor for stragglers.

    A worker's entitlement is one CPU plus its weight's fraction of the other CPUs, the fraction taken of
    ``total``: the work not yet finished (the running workers' tasks and the tasks not yet started). A worker whose
    entitlement exceeds its equal share keeps the entitlement; the others split the rest equally (water-filling,
    remainder to the lower indices). With no weights, or weights small against the work still to start, this is
    the equal split. Each share is at least one CPU and the shares sum to at most ``budget``.
    """
    weights = [max(0.0, float(value)) for value in weights]
    n = len(weights)
    total = max(float(total), float(sum(weights)))
    if n == 0 or total <= 0.0:
        return [fair_share(budget, n, rank) for rank in range(n)]
    spare = budget - n
    entitled = [1 + int(spare * value / total) for value in weights]
    fixed = [False] * n
    while True:
        rest = [i for i in range(n) if not fixed[i]]
        left = budget - sum(entitled[i] for i in range(n) if fixed[i])
        equal = [fair_share(left, len(rest), rank) for rank in range(len(rest))]
        grow = [i for i, share in zip(rest, equal, strict=False) if entitled[i] > share]
        if not grow:
            break
        for i in grow:
            fixed[i] = True
    shares = list(entitled)
    for i, share in zip(rest, equal, strict=False):
        shares[i] = share
    return shares


class LeaseState:
    """Shared lane lease state: claims, lending, one lock and the CPU budget.

    ``task_weights`` (one nonnegative work estimate per task, or None) turn on the straggler floor of the shares
    (floor_shares): a worker's floor weight is its running task's weight when that is at least ``straggler_ratio``
    times the median task weight (else zero: it takes the equal split), and ``remaining`` is the weight of the tasks
    not yet finished."""

    def __init__(self, context, workers: int, budget: int, task_weights=None, straggler_ratio: float = 0.0):
        self.claims = context.Array("i", workers, lock=False)
        self.lent = context.Array("i", workers, lock=False)
        # Per worker, the most CPUs it can use while it has nothing ready to start (its running tasks' maxima;
        # 0: no limit): it wants no more than that, so peers borrow the rest of its share (LaneAuthority.limit).
        self.wants = context.Array("i", workers, lock=False)
        # Per worker, the CPUs its running tasks hold now (as of its last rebalance). A claim (the target) may fall
        # below it when peers are short: the worker's tasks return the excess at their boundaries, and until then
        # those CPUs are not free for anyone else.
        self.held = context.Array("i", workers, lock=False)
        self.lock = context.Lock()
        self.budget = int(budget)
        self.proportional = task_weights is not None
        if self.proportional:
            values = [max(0.0, float(value)) for value in task_weights]
            positive = sorted(value for value in values if value > 0.0)
            median = positive[len(positive) // 2] if positive else 0.0
            self.threshold = float(straggler_ratio) * median
            self.task_weights = context.Array("d", values, lock=False)
            self.weights = context.Array("d", workers, lock=False)
            self.remaining = context.Value("d", float(sum(values)), lock=False)

    def free(self) -> int:
        """CPUs no worker claims or holds, under the lock: a claim recalled below what a worker's tasks hold still
        holds those CPUs until the tasks return them at their boundaries."""
        return self.budget - sum(max(claim, held) for claim, held in zip(self.claims, self.held, strict=True))

    def shares(self, active) -> list[int]:
        """The shares of the ``active`` workers (equal, or with the straggler floor), under the lock."""
        if not self.proportional:
            return [fair_share(self.budget, len(active), rank) for rank in range(len(active))]
        return floor_shares(self.budget, [self.weights[i] for i in active], self.remaining.value)


class LaneLease:
    """Rebalance a lane's lease at task boundaries.

    Its share is an equal split of the CPUs among the lanes still running, or,
    with task weights, the straggler floor (floor_shares: a task's whole weight against the work not yet finished,
    never an estimate of a running task's progress). At
    its boundaries a worker takes free CPUs up to that share; when no peer is below its own want it also borrows
    every idle CPU (work-conserving); when a peer is short, a worker above its share returns the excess, but only
    down to what it is using (``floor``), so no running team ever loses CPUs. An idle lane keeps one CPU so peers
    may borrow the rest until it has work again.
    """

    def __init__(self, state: LeaseState, index: int):
        self.state = state
        self.index = index

    def _balance(self, starting: bool = False):
        """(this worker's share, CPUs other workers lack, free CPUs), under the state lock. Shares are equal
        among the running workers (with the straggler floor when the executor has task weights); a worker that
        lends wants one CPU. ``starting``: this worker's task is taking its first share, so its want (what its
        running tasks held when the coordinator last looked: one CPU between a task's dispatch and its first share)
        does not limit it."""
        state, index = (self.state, self.index)
        active = [i for i, value in enumerate(state.claims) if value > 0 or state.lent[i] == -1 or i == index]
        wants = [0 if starting and i == index else int(state.wants[i]) for i in active]
        shares = _water_fill(state.shares(active), wants)
        others_short = sum(
            (
                max(0, (1 if state.lent[i] == 1 else share) - state.claims[i])
                for i, share in zip(active, shares, strict=False)
                if i != index
            )
        )
        return (shares[active.index(index)], others_short, state.free())

    def grow(self, *, floor: int | None = None, starting: bool = False) -> tuple[int, int]:
        """Rebalance this worker's claim at a boundary; return (before, after). ``floor``: the CPUs this worker
        is using (it may return CPUs above it); None never shrinks. ``starting``: see _balance."""
        with self.state.lock:
            return self._grow_locked(floor, starting)

    def _grow_locked(self, floor=None, starting=False):
        """Rebalance this worker's claim. ``floor``: the CPUs its tasks hold now (recorded); None never shrinks. While a
        peer is short the claim may fall below ``floor`` to the worker's share: its tasks give the excess back at their
        boundaries (and those CPUs stay counted as taken until then); otherwise it keeps at least what it holds."""
        state, index = (self.state, self.index)
        before = int(state.claims[index])
        if floor is not None:
            state.held[index] = max(0, int(floor))
        state.lent[index] = 0
        share, others_short, free = self._balance(starting)
        if others_short == 0:
            target = max(before + free, before if floor is None else max(1, floor))
        else:
            target = max(share, before) if floor is None else share
        state.claims[index] = max(1, min(target, before + free))
        return (before, int(state.claims[index]))


def _team_ceiling() -> int | None:
    """The calling thread's held claim, bounded by its task declaration."""
    local = thread_budget()
    ceiling = None
    while isinstance(local, TaskBudget):
        ceiling = local.maximum if ceiling is None else min(ceiling, local.maximum)
        local = local.budget
    if isinstance(local, SlotBudget):
        with local.slots.lock:
            if local.slots.threads.get(local.index) == threading.get_ident():
                claim = max(1, int(local.slots.claims.get(local.index, 1)))
                return claim if ceiling is None else min(claim, ceiling)
    if isinstance(local, FixedBudget):
        return local.n if ceiling is None else min(local.n, ceiling)
    seat = _POOL_SEAT
    if seat is not None and seat.task_thread == threading.get_ident():
        claim = max(1, int(seat.claimed))
        return claim if ceiling is None else min(claim, ceiling)
    return ceiling


_THREAD_BUDGET = threading.local()


def task_owner() -> TaskKey | None:
    """The graph task owning the calling thread's nested work."""
    return getattr(_THREAD_BUDGET, "task_owner", None)


def thread_budget() -> CpuBudget | None:
    """The explicitly bound task allowance, or None outside task work."""
    return getattr(_THREAD_BUDGET, "value", None)


@contextmanager
def executing_task(owner: TaskKey | None):
    """Carry task ownership separately from its changing nested CPU allowance."""
    previous = task_owner()
    _THREAD_BUDGET.task_owner = owner
    try:
        if owner is None:
            yield
        else:
            with executing_work():
                yield
    finally:
        _THREAD_BUDGET.task_owner = previous


@contextmanager
def budget_context(budget):
    """Bind deep numerical claims to this task, including nested thread work."""
    previous = getattr(_THREAD_BUDGET, "value", None)
    _THREAD_BUDGET.value = budget
    try:
        yield
    finally:
        _THREAD_BUDGET.value = previous


class _TaskThreads:
    """The calling task's live CPU share for codecs, read at each encode or decode: give it to a store (``nthreads``)
    so a reader inside a graph task decompresses with the CPUs that task holds to parallelise large input reloads.
    Pickles by reference."""

    def threads(self) -> int:
        budget = current_budget()
        return max(1, int(budget.threads()))

    def __reduce__(self) -> str:
        return "TASK_THREADS"


TASK_THREADS = _TaskThreads()


def current_budget():
    """Capture the caller's live allowance before creating worker threads."""
    own = getattr(_THREAD_BUDGET, "value", None)
    if own is not None:
        return own
    if _POOL_SEAT is not None:
        return PoolShareBudget()
    return FixedBudget(int(_numba().get_num_threads()))


def capped_threads(budget, cap: int, *, hold: int = 0) -> int:
    """Threads for a phase that cannot use more than ``cap`` well: at most ``cap`` from the budget.
    A concurrent slot or pool task owning its claim returns the excess at
    this boundary, so peers take it (the next uncapped boundary grows it back to its share). Results never
    depend on it. ``hold``: CPUs the caller's running children still use (CpuBudget.allowance)."""
    cap = max(1, int(cap))
    if isinstance(budget, SlotBudget):
        return max(1, min(budget.claim(cap, hold=hold), cap))
    if isinstance(budget, PoolShareBudget) and _POOL_SEAT is not None:
        # The cap bounds the claim itself on every thread: a helper thread's read (no give-back) still grows the
        # task toward its fair share, but never past what the task can use.
        if _POOL_SEAT.owns_team():
            return max(1, min(_POOL_SEAT.share(cap, hold=hold), cap))
        return max(1, min(_POOL_SEAT.share(cap, give_back=False), cap))
    return max(1, min(int(budget.allowance(hold)), cap))


def authority_threads(cap: int) -> int:
    """At most ``cap`` threads under the caller's CPU authority: its task's claim or pool share (as
    capped_threads); ``cap`` itself for a caller outside any authority (a standalone process owns its CPUs)."""
    if getattr(_THREAD_BUDGET, "value", None) is None and _POOL_SEAT is None:
        return max(1, int(cap))
    return capped_threads(current_budget(), cap)


def idle_lane_cpus() -> int:
    """CPUs of this pool worker's lane that no task holds and none waits for (0 outside a pool, or while a task
    waits for CPUs). A serial step of a single-CPU task that the rest of its contig waits for (a chain
    step, a reply copy) borrows them briefly; a task dispatched meanwhile oversubscribes them only for that step."""
    seat = _POOL_SEAT
    if seat is None or getattr(_THREAD_BUDGET, "no_idle_loans", False):
        return 0
    counters = seat.counters
    if counters.waiting.value:
        return 0
    return max(0, read_counter(counters.total) - read_counter(counters.claimed))


@contextmanager
def _no_idle_loans():
    """Refuse idle-lane loans to the calling thread: it is one item of a one-thread map, whose siblings
    run beside it, so every item borrowing the lane's idle CPUs for one decode would multiply the lane's threads."""
    previous = getattr(_THREAD_BUDGET, "no_idle_loans", False)
    _THREAD_BUDGET.no_idle_loans = True
    try:
        yield
    finally:
        _THREAD_BUDGET.no_idle_loans = previous


def _install_codec_idle_cpus() -> None:
    from ..store import codec

    codec.idle_cpus = idle_lane_cpus


def _ordered_reads(function, items):
    """core.concurrency's map: independent record reads on the caller's budget (a task's claim; in
    the coordinator its configured threads), one thread per item; inside an item a nested map runs serially."""
    return current_budget().thread_map(function, items, label="independent record reads")


def _live_allowance() -> int:
    """core.concurrency.live_threads: the caller's live CPU allowance (a task's claim or its share)."""
    return current_budget().threads()


def _install_ordered_reads() -> None:
    from ..core import concurrency

    concurrency.ordered_map = _ordered_reads
    concurrency.allowance = _live_allowance


def install_runtime_hooks() -> None:
    """Bind optional core/store execution hooks at application and worker startup."""
    _install_codec_idle_cpus()
    _install_ordered_reads()


def spare_threads(cap: int) -> int:
    """At most ``cap`` threads: this task's CPU authority plus its lane's idle CPUs (idle_lane_cpus)."""
    return max(1, min(int(cap), authority_threads(cap) + idle_lane_cpus()))


def share_reader(interval: float = 0.1) -> Callable[[], int]:
    """The calling task's live CPU share as a callable: claim_threads (a boundary, which also sets the calling thread's
    kernel team) re-read when ``interval`` seconds have passed since the last read. A long task that started on one
    CPU beside many siblings grows into the CPUs they leave (for example, discovery blocks)."""
    import math
    import time

    read_at, share = [-math.inf], [1]

    def read() -> int:
        now = time.monotonic()
        if now - read_at[0] >= interval:
            read_at[0], share[0] = now, claim_threads()
        return share[0]

    return read


def claim_threads(*, max_threads: int | None = None) -> int:
    """Boundary call: recompute this process's share, apply it to the calling thread's Numba team, return it.

    Pool worker: its pool share; elsewhere the current team. ``max_threads`` caps the result for this phase.
    """
    local = getattr(_THREAD_BUDGET, "value", None)
    if local is not None:
        n = (
            local.claim(max_threads)
            if isinstance(local, (SlotBudget, TaskBudget))
            else capped_threads(local, max_threads)
            if max_threads is not None
            else local.threads()
        )
        if max_threads is not None:
            n = min(n, max_threads)
        numba = _numba()
        n = max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(n)
        return n
    if _POOL_SEAT is not None:
        return _POOL_SEAT.apply(max_threads)
    numba = _numba()
    threads = int(numba.get_num_threads())
    if max_threads is not None:
        threads = max(1, min(threads, int(max_threads)))
    return _set_team(threads)


def fair_share(total: int, contenders: int, rank: int) -> int:
    """Equal entitlement, with remainder CPUs assigned in rank order."""
    return max(1, total // contenders + int(rank < total % contenders))


def fair_claim(
    total: int,
    contenders: int,
    rank: int,
    mine: int,
    free: int,
    *,
    cap: int | None = None,
    give_back: bool = True,
    hold: int = 0,
) -> tuple[int, int]:
    """Return (allowance, reservation) at a concurrent child's own boundary.

    free includes this owner's reservation: CPUs no peer holds. Only the owning
    thread may shrink a team; running descendants retain hold CPUs even when the
    allowance falls. Growth takes only unclaimed CPUs.
    """
    want = fair_share(total, contenders, rank)
    if cap is not None:
        want = min(want, cap)
    target = want if give_back else max(want, mine)
    allocated = max(1, min(target, free)) if target < mine else max(mine, min(target, free))
    return allocated, max(allocated, min(int(hold), mine))


class ThreadSlots:
    """Reusable thread slots of concurrent in-process tasks under one live budget."""

    def __init__(self, workers: int, budget: CpuBudget):
        self.workers = workers
        self.budget = budget
        self.limit = 1
        self.owners: dict[int, int] = {}
        self.admitted = 0
        self.claims: dict[int, int] = {}
        self.wants: dict[int, int] = {}
        self.threads: dict[int, int] = {}
        self.waiting = 0
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def publish(self) -> int:
        # Serialize child growth/admission with the hold snapshot, upstream claim
        # change and publication. No child calls the parent budget itself.
        with self.lock:
            self.limit = max(1, int(self.budget.allowance(sum(self.claims.values()))))
            return self.limit

    def admit(self, index):
        with self.lock:
            if sum(self.claims.values()) >= self.limit:
                return False
            slot = next(i for i in range(self.workers) if i not in self.owners.values())
            self.owners[index] = slot
            self.claims[index] = 1
            self.admitted += 1
            return True

    def start(self, index):
        assert index in self.claims
        with self.lock:
            self.threads[index] = threading.get_ident()

    def finish(self, index):
        with self.lock:
            del self.owners[index]
            del self.claims[index]
            self.wants.pop(index, None)
            self.threads.pop(index, None)
            self.admitted -= 1

    def allocation(self, index, max_threads=None, hold=0) -> int:
        """The task's CPUs from now to its next boundary (see PoolSeat.share for the rule): its fair share among
        the admitted tasks and those held back for a CPU (the remainder to the lowest slots), capped by
        ``max_threads``; a task above it returns the excess, but only at its own thread's boundary (other threads'
        queries and tasks with nested work never shrink a running team; ``hold``, CpuBudget.allowance, keeps a
        coordinating task's running children covered). Admitted tasks that have not started yet are counted, so a
        just-started task never takes their threads."""
        with self.lock:
            if self.stopped.is_set():
                raise CancelledError("another concurrent task failed")
            count = max(1, self.admitted) + self.waiting
            rank = sum(slot < self.owners[index] for slot in self.owners.values())
            cores = self.limit
            want = fair_share(cores, count, rank)
            self.wants[index] = want if max_threads is None else min(want, max_threads)
            mine = self.claims[index]
            others = sum(value for owner, value in self.claims.items() if owner != index)
            allocated, self.claims[index] = fair_claim(
                cores,
                count,
                rank,
                mine,
                cores - others,
                cap=max_threads,
                hold=hold,
                give_back=self.threads.get(index) == threading.get_ident(),
            )
            return allocated


class SlotBudget(CpuBudget):
    """A concurrent task's live share (see ThreadSlots.allocation)."""

    live = True

    def __init__(self, slots: ThreadSlots, index: int):
        self.slots = slots
        self.index = index

    def threads(self) -> int:
        n = self.slots.allocation(self.index)
        if self.slots.threads.get(self.index) == threading.get_ident() and _numba().get_num_threads() > n:
            _numba().set_num_threads(n)
        return n

    def allowance(self, held: int) -> int:
        n = self.slots.allocation(self.index, hold=held)
        if self.slots.threads.get(self.index) == threading.get_ident() and _numba().get_num_threads() > n:
            _numba().set_num_threads(n)
        return n

    def claim(self, maximum=None, *, hold: int = 0) -> int:
        return self.slots.allocation(self.index, maximum, hold)


def _threading_layer_is_workqueue() -> bool:
    numba = _numba()
    try:
        return numba.threading_layer() == "workqueue"
    except ValueError:
        return numba.config.THREADING_LAYER == "workqueue"


class TaskBudget(CpuBudget):
    """A task's declared range around the native elastic share policy."""

    live = True

    def __init__(self, budget: CpuBudget, minimum: int, maximum: int, *, observe: Callable[[int], None] | None = None):
        self.budget, self.minimum, self.maximum = (budget, minimum, maximum)
        self.observe = observe

    def threads(self) -> int:
        return self.claim()

    def allowance(self, held: int) -> int:
        return self.claim(hold=held)

    def claim(self, maximum: int | None = None, *, hold: int = 0) -> int:
        ceiling = self.maximum if maximum is None else min(self.maximum, max(1, maximum))
        floor = min(self.minimum, ceiling)
        # The declared minimum stays claimed through give-backs, as running children's CPUs do (hold): begin_task
        # reserved it, and a minimum returned at one boundary would have to be won back from peers at the next.
        # A phase capped below the minimum runs fewer threads but keeps the reservation.
        team = capped_threads(self.budget, ceiling, hold=max(int(hold), min(self.minimum, self.maximum)))
        if isinstance(self.budget, PoolShareBudget) and team < floor:
            seat = pool_seat()
            assert seat is not None
            c = seat.counters
            with c.admission:
                while seat.claimed < floor and read_counter(c.total) - (read_counter(c.claimed) - seat.claimed) < floor:
                    if read_counter(c.closed):
                        raise CancelledError("CPU group closed during minimum-team admission")
                    c.admission.wait(CPU_POLL_SECONDS)
                reserved = max(floor, seat.claimed)
                c.claimed.get_obj().value += reserved - seat.claimed
                seat.claimed = reserved
            team = floor
        team = min(ceiling, max(floor, team))
        if getattr(_THREAD_BUDGET, "value", None) is self:
            # On the task's own thread the claim is also its kernel team: a team left at a larger earlier claim
            # oversubscribes, one left smaller cannot grow (other threads keep their own one-thread teams).
            numba = _numba()
            numba.set_num_threads(max(1, min(team, numba.config.NUMBA_NUM_THREADS)))
        if self.observe is not None:
            # A helper's capped query cannot release an owning thread's wider claim.
            seat = pool_seat() if isinstance(self.budget, PoolShareBudget) else None
            self.observe(seat.claimed if seat is not None else max(team, hold))
        return team

    def current(self) -> int:
        return min(self.maximum, self.budget.current())
