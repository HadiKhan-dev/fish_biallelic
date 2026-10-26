"""CPU budgets: how many threads each piece of work may run, and how that changes as work finishes.

Every allowance is a `CpuBudget`; `threads()` re-reads it at a safe boundary (never inside a running kernel).

Rebalancing (2.10): a piece of work runs its fair share of the CPUs among its contenders, capped by what it can
use (``max_threads``); a holder above its share (it grew while it had fewer contenders) returns the excess at its own
next boundary, and as contenders finish the shares of the rest grow. Contig lanes, whose shares are equal among the
running lanes, also borrow idle CPUs (of finished lanes, or of a lane waiting for memory) while no lane is short and
return them once a lane wants its share again. Claims are only ever taken from free CPUs and returned by their
holder, so the CPUs in use never exceed the budget, and no task keeps more than its share while others wait for
CPUs.

* Fixed: a count (the run's cores in serial mode, a pool's or executor's total).
* Lease: a chromosome-executor worker's (a contig lane's) claim on the executor's CPUs. Its share is an equal split
  of the executor's CPUs among the running workers (the remainder to the lower indices); at its boundaries (task
  starts, `phase()` starts, `threads()` and `claim_threads()` calls in its coordinating thread, and its pools'
  re-reads) it takes free CPUs up to that share, borrows idle ones beyond it when no peer is short, and returns
  CPUs above its share that it is not using when a peer is short. A lane waiting for memory lends all but one CPU
  until it resumes. Claims sum to at most the budget.
* Pool share: a pool worker's share of its pool's total: floor(total / (active + waiting workers)), plus one of the
  remainder CPUs while it holds a remainder claim. Until a batch's start-up gate opens (min(tasks, workers)
  distinct workers have started, or every task has), each worker runs one thread, so an early worker cannot take
  the CPUs its peers are about to need. A pool's total follows a live budget both ways; a pool adds workers when
  its total exceeds its workers while tasks wait (execution.pools).
* Slot share: one of several concurrent in-process tasks (`concurrent`). Slots reserve one CPU before admission;
  their shares follow the same rule among the admitted tasks and a task waiting for admission.

Process x thread totals stay within the budget: a late pool seat or slot waits for a free claim; running
claims are never taken from peers (they return CPUs at their own boundaries). Thread counts never change results:
every parallel kernel reduces per-sample (or per-site) rows in a fixed order.

A process holds at most one seat: a pool worker's (`_POOL_SEAT`) or an executor worker's (`_LEASE_SEAT`).
`claim_threads` acts on the calling process's seat, so deep numerical code needs no budget argument.

Memory: `concurrent` and `thread_map` reserve each task's bytes with the run's memory governor
(execution.memory) before starting it, so memory paces them as well as the CPU allowance.
"""
from __future__ import annotations

from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import os
import sys
import threading
import time

from .memory import adopted, depth_below, held_context, memory_governor

# How often a call with queued in-process tasks re-reads a live budget between completions.
POLL_SECONDS = 0.2


def _numba():
    import numba
    return numba


def _set_team(n: int) -> int:
    """Set the calling thread's Numba team to ``n``; a lane coordinator's team is recorded on its lease (CPUs it
    uses, which its pools' rebalancing counts)."""
    numba = _numba()
    n = max(1, int(n))
    lease = _LEASE_SEAT
    owner = lease is not None and lease.owner == threading.get_ident()
    if owner and lease.users:
        n = 1           # while the lane's pools or maps hold its CPUs its coordinator runs one thread
    if numba.get_num_threads() != n:
        numba.set_num_threads(n)
    if owner:
        lease.team = n
    return n


def numba_threads_set(n: int) -> int:
    """Set the calling thread's Numba team to ``n`` at a boundary inside a `numba_threads` block (whose exit
    restores within the current claim); returns the team."""
    return _set_team(n)


@contextmanager
def numba_threads(n: int):
    """Run the block with a Numba team of ``n`` threads. On exit the calling thread's setting is restored, within
    its current claim when it owns one (a share given back meanwhile is not taken again by the restore)."""
    old = _numba().get_num_threads()
    _set_team(n)
    try:
        yield
    finally:
        ceiling = _team_ceiling()
        _set_team(old if ceiling is None else max(1, min(old, ceiling)))


# ---------------------------------------------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------------------------------------------

class CpuBudget:
    """A CPU allowance (see the module docstring). Picklable: a live allowance pickles as its current value."""

    live = False

    @staticmethod
    def fixed(n: int) -> "FixedBudget":
        return FixedBudget(n)

    def threads(self) -> int:
        """The current allowance (>= 1), re-read now; a lease may grow on each call."""
        raise NotImplementedError

    def phase(self, name: str) -> "CpuBudget":
        """Mark the start of phase ``name`` (a boundary: a lane rebalances its lease); return the budget."""
        return self

    def current(self) -> int:
        """The allowance as it stands, without claiming anything (what a pickled copy carries)."""
        return self.threads()

    def __reduce__(self):
        return (FixedBudget, (max(1, int(self.current())),))

    def concurrent(self, functions, bytes_per_task, *, label: str = "concurrent", envelope: bool = False):
        """Yield ``(index, functions[index](child_budget))`` as independent in-process tasks complete.

        At most one running task per thread of the live allowance; each task holds ``bytes_per_task`` (an int for
        every task, or one per task) of the run's memory while it runs and starts, in order, only when they fit
        (execution.memory). Every running task gets a child budget: its share among the admitted tasks, re-read
        at its boundaries, so survivors take the threads of finished peers. Queued tasks start as the allowance
        grows or memory is released (re-read every POLL_SECONDS and at every completion). One task, or an
        allowance of one thread, runs inline with this budget. An exception stops the others. Results must not
        depend on this schedule. ``envelope``: a task's bytes also cover what its nested constructs reserve.
        """
        return _completed_tasks(self, list(functions), bytes_per_task, label, envelope)

    def thread_map(self, function, items, *, item_bytes=None, label: str = "thread map",
                   envelope: bool = False) -> list:
        """``[function(item) for item in items]`` in input order; each call on one Numba thread, with as many
        calls at once as the live allowance (workers start when it grows and stop while they exceed it). With
        ``item_bytes`` (an int for every item, or one per item) each call holds that much of the run's memory."""
        items = list(items)
        return _one_thread_map(self, function, items, _sizes(item_bytes, len(items)), label, envelope)


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


def as_budget(budget) -> CpuBudget:
    """A CpuBudget for a budget or a plain thread count."""
    return budget if isinstance(budget, CpuBudget) else FixedBudget(budget)


# ---------------------------------------------------------------------------------------------------------------
# Pool shares
# ---------------------------------------------------------------------------------------------------------------

def _read(value):
    """Lock-free read of a shared counter (an aligned int read is atomic; a slightly stale value is fine)."""
    return value.get_obj().value if hasattr(value, "get_obj") else value.value


class PoolCounters:
    """The shared state of one pool's thread shares (created by the pool's parent, passed to its workers)."""

    def __init__(self, context, total: int, processes: int):
        value = context.Value
        self.total = value("i", int(total))        # follows a live parent budget (both ways)
        self.active = value("i", 0)                # workers inside a task
        self.claimed = value("i", 0)               # retained numerical-team claims
        self.admission = context.Condition(self.claimed.get_lock())
        # Under the admission lock: workers waiting for their first CPU, and the summed shortfall of running tasks
        # below their fair share (what peers above their share give back at their boundaries).
        self.waiting = value("i", 0, lock=False)
        self.short = value("i", 0, lock=False)
        self.extra = value("i", 0)                 # remainder claims held
        self.started = value("i", 0)               # tasks started in the current batch
        self.participants = value("i", 0)          # distinct workers started in the current batch
        self.generation = value("i", 0)            # batch number
        self.task_count = value("i", 0)            # tasks in the current batch
        self.startup_target = value("i", min(1, processes))
        self.startup_ready = value("i", 0)
        self.closed = value("i", 0)                # the pool is closing: waiting consumers and helpers stop
        # Helper seats (BudgetedPool.helper_seat), under the admission lock: the CPUs set aside for them while they
        # run and the CPUs they hold. Tasks share total - reserved among themselves; helpers are not task contenders.
        self.reserved = value("i", 0, lock=False)
        self.helper_claimed = value("i", 0, lock=False)

    @staticmethod
    def _store(counter, number):
        with counter.get_lock():
            counter.get_obj().value = int(number)

    def start_batch(self, tasks: int, processes: int) -> None:
        """Open a new batch (generation): reset the start-up gate for ``tasks`` tasks."""
        if _read(self.active):
            raise RuntimeError("pool workers of the preceding batch are still active")
        target = min(int(tasks), int(processes))
        self._store(self.started, 0)
        self._store(self.participants, 0)
        self._store(self.task_count, tasks)
        self._store(self.startup_target, target)
        self._store(self.startup_ready, target <= 1)
        with self.generation.get_lock():
            self.generation.get_obj().value += 1

    def set_total(self, threads: int) -> None:
        """Follow the live parent budget both ways. A raise wakes workers waiting for a CPU; a cut applies at the
        running tasks' own boundaries (share() then returns only CPUs within the new total; claims are never
        revoked inside a kernel) and holds back new task starts until the claims fit."""
        threads = max(1, int(threads))
        with self.admission:
            old = _read(self.total)
            if threads != old:
                self._store(self.total, threads)
                if threads > old:
                    self.admission.notify_all()

    def open_gate(self) -> None:
        """Open the current batch's start-up gate now (memory holds the remaining tasks back)."""
        self._store(self.startup_ready, 1)

    def close(self) -> None:
        """The pool is closing: wake waiting consumers and helper seats (they stop instead of waiting for CPUs
        that terminated workers will never return)."""
        with self.admission:
            self._store(self.closed, 1)
            self.admission.notify_all()

    def take_one(self, cancel=None) -> bool:
        """The pool's consumer (the thread iterating its results) takes one of the pool's CPUs before it works on
        a result: as a waiting contender it makes tasks above their fair share give back at their boundaries,
        and a finishing task frees one. The caller waits as suspended in the memory governor, so tasks that need
        memory it holds can still be admitted and finish. Returns False only when the pool closes or ``cancel``
        (a threading.Event) is set."""
        with self.admission:
            self.waiting.value += 1
            try:
                while ((_read(self.claimed) >= _read(self.total)
                        or _read(self.claimed) - self.helper_claimed.value >= _read(self.total) - self.reserved.value)
                       and not _read(self.closed) and not (cancel is not None and cancel.is_set())):
                    self.admission.wait(0.5)
            finally:
                self.waiting.value -= 1
            if _read(self.closed) or (cancel is not None and cancel.is_set()):
                return False
            self.claimed.get_obj().value += 1
            return True

    def give_one(self) -> None:
        """The consumer returns the CPU it took (it waits for the next result)."""
        with self.admission:
            self.claimed.get_obj().value -= 1
            self.admission.notify_all()


class _PoolSeat:
    """A pool worker's view of its pool's shares (one per worker process)."""

    def __init__(self, counters: PoolCounters):
        self.counters = counters
        self.has_extra = False
        self.claimed = 0
        self.short = 0                 # this task's contribution to counters.short
        self.task_thread = None        # the thread running the current task (its boundaries may give back)
        self.users = 0                 # in-process maps and nested pools of the task (they hold its CPUs)
        self.lock = threading.Lock()
        self.last_generation = None
        self.last_applied = None
        self.last_logged = None

    def begin_task(self) -> None:
        """Reserve one CPU before admission (waiting meanwhile counts as a contender, so peers above their fair
        share give back at their boundaries), then register the task with the start-up gate."""
        c = self.counters
        with c.admission:
            c.waiting.value += 1
            try:
                # A free CPU of the whole total, and room in the tasks' unreserved capacity (helpers' own claims are
                # inside their reservation, so they are not counted against it).
                while (_read(c.claimed) >= _read(c.total)
                       or _read(c.claimed) - c.helper_claimed.value >= _read(c.total) - c.reserved.value):
                    c.admission.wait()
            finally:
                c.waiting.value -= 1
            c.claimed.get_obj().value += 1
            self.claimed = 1
            self.task_thread = threading.get_ident()
            with c.active.get_lock():
                c.active.get_obj().value += 1
        with c.started.get_lock():
            c.started.get_obj().value += 1
            started = c.started.get_obj().value
        generation = _read(c.generation)
        if self.last_generation != generation:
            with c.participants.get_lock():
                c.participants.get_obj().value += 1
                participants = c.participants.get_obj().value
            self.last_generation = generation
        else:
            participants = _read(c.participants)
        # Distinct participants, not task starts, open the gate, so one fast worker cannot open it alone; if
        # fewer distinct workers serve a short batch, starting every task opens it too.
        if participants >= _read(c.startup_target) or started >= _read(c.task_count):
            with c.startup_ready.get_lock():
                c.startup_ready.get_obj().value = 1

    def end_task(self) -> None:
        """Release this finished task's claims and wake workers waiting for admission."""
        c = self.counters
        with c.admission:
            self._release_extra()
            c.claimed.get_obj().value -= self.claimed
            c.short.value -= self.short
            self.claimed = self.short = 0
            self.task_thread = None
            self.users = 0                 # a task's maps and nested pools end with it
            with c.active.get_lock():
                c.active.get_obj().value -= 1
            c.admission.notify_all()

    def enter_user(self) -> None:
        with self.lock:
            self.users += 1

    def exit_user(self) -> None:
        with self.lock:
            self.users = max(0, self.users - 1)

    def _total(self) -> int:
        """The CPUs this seat's tasks share: the pool's total minus what running helper seats hold back."""
        c = self.counters
        return max(1, _read(c.total) - c.reserved.value)

    def owns_team(self) -> bool:
        """Whether the calling thread may return CPUs: the task's own thread, with no map or pool of it open."""
        return self.task_thread == threading.get_ident() and self.users == 0

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

    def share(self, max_threads: int | None = None, *, give_back: bool = True) -> int:
        """This task's CPUs from now to its next boundary (claims change only at boundaries and at task start and
        end, never inside a running kernel; the pool's claims never exceed its total).

        Fair share: floor(total / contenders), plus one of the remainder CPUs while this task holds a remainder
        claim; contenders are the running tasks and the workers waiting for their first CPU. ``max_threads``
        (what the task can use) caps it. A task grows only up to its share (as peers finish, the share grows);
        a task above it (it grew while it had fewer peers) returns the excess here (``give_back``; False for
        queries from threads that do not own the task's team). Before the batch's start-up gate opens every task
        runs one thread.
        """
        if max_threads is not None:
            max_threads = int(max_threads)
            if max_threads < 1:
                raise ValueError("max_threads must be a positive integer or None")
        c = self.counters
        with c.admission:
            if not self.claimed:  # outside a task, including worker initialization
                return 1
            total = self._total()
            if not _read(c.startup_ready):
                self._release_extra()
                want = target = 1
            else:
                contenders = max(_read(c.active), 1) + c.waiting.value
                floor, remainder = divmod(total, contenders)
                if max_threads is not None and max_threads <= floor:
                    self._release_extra()
                else:
                    extras = _read(c.extra)
                    if not self.has_extra and extras < remainder:
                        self._claim_extra(remainder)
                    elif self.has_extra and extras > remainder:
                        self._release_extra()
                want = max(1, floor + int(self.has_extra))
                if max_threads is not None:
                    want = min(want, max_threads)
                # Never beyond the fair share: a peer that has not reached a boundary yet is a contender too
                # (borrowing ahead of it would leave it one thread until this task's next boundary).
                target = want
            if not give_back:
                target = max(target, self.claimed)
            # Free CPUs of the whole total, and of the tasks' unreserved capacity (helpers hold their slices).
            others = _read(c.claimed) - self.claimed
            available = min(_read(c.total) - others, self._total() - (others - c.helper_claimed.value))
            allocated = max(1, min(target, available)) if give_back else max(self.claimed, min(target, available))
            change = allocated - self.claimed
            c.claimed.get_obj().value += change
            self.claimed = allocated
            short = max(0, want - allocated)
            c.short.value += short - self.short
            self.short = short
            if change < 0:
                c.admission.notify_all()
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
        from .bootstrap import profiling
        if not profiling() or n == self.last_logged:
            return
        self.last_logged = n
        c = self.counters
        print(f"[threads pid={os.getpid()} t={time.monotonic():.1f}] active={_read(c.active)} "
              f"total={_read(c.total)} extras={_read(c.extra)} mine={self.has_extra} -> {n}",
              file=sys.stderr, flush=True)


class _HelperSeat(_PoolSeat):
    """A thread of the pool's parent process working beside the pool's tasks on their critical path (discovery's
    evidence helper, which prepares the next contig's blocks): while it runs, ``slice`` of the pool's CPUs are set
    aside for it (the tasks share the rest and give the excess back at their boundaries), and it grows into the
    slice as CPUs free; outside the batch's start-up gate."""

    def __init__(self, counters, slice_cpus: int):
        super().__init__(counters)
        self.slice = max(1, int(slice_cpus))

    def share(self, max_threads=None, *, give_back: bool = True) -> int:
        """Up to the helper's slice (capped by ``max_threads``), from free CPUs of the whole total; it returns
        CPUs only at its own boundary (``give_back``), never below one."""
        c = self.counters
        with c.admission:
            target = self.slice if max_threads is None else max(1, min(self.slice, int(max_threads)))
            available = _read(c.total) - (_read(c.claimed) - self.claimed)
            wanted = max(1, min(target, available))
            allocated = wanted if give_back else max(self.claimed, wanted)
            change = allocated - self.claimed
            c.claimed.get_obj().value += change
            c.helper_claimed.value += change
            if change < 0:
                c.admission.notify_all()
            self.claimed = allocated
            return allocated

    def end_task(self) -> None:
        """Return the helper's CPUs and its reservation (it was never a task contender)."""
        c = self.counters
        with c.admission:
            c.claimed.get_obj().value -= self.claimed
            c.helper_claimed.value -= self.claimed
            c.reserved.value -= self.slice
            self.claimed = 0
            self.task_thread = None
            self.users = 0
            c.admission.notify_all()

    def begin(self) -> None:
        """Set the slice aside and wait for a first CPU of the whole total; CancelledError if the pool closes
        first. Helpers are not task contenders (tasks share total - reserved among themselves)."""
        c = self.counters
        with c.admission:
            c.reserved.value += self.slice
            c.admission.notify_all()            # tasks above their (now smaller) share give back at boundaries
            while _read(c.claimed) >= _read(c.total) and not _read(c.closed):
                c.admission.wait(0.5)
            if _read(c.closed):
                c.reserved.value -= self.slice
                raise CancelledError("the pool closed before its helper got a CPU")
            c.claimed.get_obj().value += 1
            c.helper_claimed.value += 1
            self.claimed = 1
            self.task_thread = threading.get_ident()


class HelperBudget(CpuBudget):
    """The live share of a helper seat (see BudgetedPool.helper_seat)."""

    live = True

    def __init__(self, seat: _HelperSeat):
        self.seat = seat

    def threads(self) -> int:
        seat = self.seat
        if not seat.owns_team():
            return seat.share(give_back=False)
        n = seat.share()
        numba = _numba()
        if numba.get_num_threads() > n:
            numba.set_num_threads(n)
        return n

    def current(self) -> int:
        return max(1, int(self.seat.claimed))

    def __repr__(self):
        return "HelperBudget()"


class PoolShareBudget(CpuBudget):
    """The calling pool worker's live share (see _PoolSeat.share)."""

    live = True

    def threads(self) -> int:
        seat = _POOL_SEAT
        if seat is None:
            return 1
        if not seat.owns_team():
            # Another thread's query, or the task's maps hold its CPUs: never shrinks a running team.
            return seat.share(give_back=False)
        n = seat.share()
        numba = _numba()
        if numba.get_num_threads() > n:             # the task's team stays within its claim
            numba.set_num_threads(n)
        return n

    def current(self) -> int:
        seat = _POOL_SEAT
        return 1 if seat is None or seat.last_applied is None else seat.last_applied

    def __repr__(self):
        return "PoolShareBudget()"


_POOL_SEAT: _PoolSeat | None = None


def enter_pool_seat(counters: PoolCounters) -> None:
    """In a pool worker (initializer): take a seat in the pool's shares."""
    global _POOL_SEAT
    _POOL_SEAT = _PoolSeat(counters)


def pool_seat() -> _PoolSeat | None:
    return _POOL_SEAT


# ---------------------------------------------------------------------------------------------------------------
# Leases of chromosome-executor workers
# ---------------------------------------------------------------------------------------------------------------

def apportion(budget: int, work) -> list[int]:
    """Split ``budget`` CPUs over workers in proportion to their work.

    Each worker keeps one CPU; the others follow the largest remainders of the proportional shares (ties to the
    lower index). Equal shares if no work is known. Requires at least one CPU per worker.
    """
    work = list(work)
    total = float(sum(work))
    if total <= 0.0:
        work, total = [1.0] * len(work), float(len(work))
    spare = budget - len(work)
    shares = [spare * value / total for value in work]
    result = [1 + int(share) for share in shares]
    order = sorted(range(len(work)), key=lambda i: (int(shares[i]) - shares[i], i))
    for i in order[:budget - sum(result)]:
        result[i] += 1
    return result


class LeaseState:
    """Shared lease state of one executor: per worker its claim and whether it lends (while it waits for memory
    it wants one CPU, not its share), one lock, the budget."""

    def __init__(self, context, workers: int, budget: int):
        self.claims = context.Array("i", workers, lock=False)
        self.lent = context.Array("i", workers, lock=False)
        self.lock = context.Lock()
        self.budget = int(budget)

    def start_phase(self, loads, initial=None) -> list[int]:
        """Initial claims follow initial work (or whole load); after that every running worker's share is equal."""
        with self.lock:
            shares = apportion(self.budget, loads if initial is None else initial)
            for index, share in enumerate(shares):
                self.claims[index] = share
                self.lent[index] = 0
        return shares


class _LeaseSeat:
    """An executor worker's lease during one phase.

    Its share is an equal split of the CPUs among the workers still running (initial claims follow the workers'
    loads; remaining-work estimates can be far off inside long tasks, so they never take CPUs from a lane). At
    its boundaries a worker takes free CPUs up to that share; when no peer is below its own want it also borrows
    every idle CPU (work-conserving); when a peer is short, a worker above its share returns the excess, but only
    down to what it is using (``floor``), so no running team ever loses CPUs. A worker that lends (while it waits
    for memory) keeps one CPU and wants no more, so peers may borrow the rest until it resumes.
    """

    def __init__(self, state: LeaseState, index: int):
        self.state = state
        self.index = index
        self.owner = threading.get_ident()      # the worker's coordinating thread
        self.users = 0                # open pools and in-process maps on this lease (they hold its CPUs)
        self.team = 1                 # the coordinating thread's Numba team (CPUs it holds, see _set_team)
        self.waiting = 0              # >0 while the coordinating thread only waits for its pool's results
        self.lock = threading.Lock()

    def _balance(self):
        """(this worker's share, CPUs other workers lack, free CPUs), under the state lock. Shares are equal
        among the running workers; a worker that lends wants one CPU."""
        state, index = self.state, self.index
        active = [i for i, value in enumerate(state.claims) if value > 0 or i == index]
        shares = apportion(state.budget, [1.0] * len(active))
        others_short = sum(max(0, (1 if state.lent[i] else share) - state.claims[i])
                           for i, share in zip(active, shares) if i != index)
        return shares[active.index(index)], others_short, state.budget - sum(state.claims)

    def grow(self, *, floor: int | None = None) -> tuple[int, int]:
        """Rebalance this worker's claim at a boundary; return (before, after). ``floor``: the CPUs this worker
        is using (it may return CPUs above it); None never shrinks."""
        state, index = self.state, self.index
        with state.lock:
            before = int(state.claims[index])
            state.lent[index] = 0
            share, others_short, free = self._balance()
            target = before + free if others_short == 0 else share
            target = max(target, before if floor is None else max(1, floor))
            state.claims[index] = max(1, min(target, before + free))
            return before, int(state.claims[index])

    def settle(self, in_use: int) -> int:
        """A pool's re-read of this lease (pool total; called under the pool's admission lock, so its tasks'
        claims cannot change meanwhile): rebalance with ``in_use`` CPUs held by the pool's running tasks plus the
        coordinating thread's team; the claim stays at least that and the returned total may be lower (tasks
        shrink to it at their boundaries, and the claim follows at later re-reads)."""
        state, index = self.state, self.index
        coordinator = 0 if self.waiting else int(self.team)
        in_use = int(in_use) + coordinator
        with state.lock:
            before = int(state.claims[index])
            state.lent[index] = 0
            share, others_short, free = self._balance()
            target = max(1, min(before + free, before + free if others_short == 0 else share))
            # Several pools or maps on this lease: each sees only its own use, so none may shrink the claim.
            state.claims[index] = max(target, min(in_use, max(before, target)), 1,
                                      before if self.users > 1 else 1)
            return max(1, target - coordinator)

    def lend(self) -> None:
        """Keep one CPU and want no more, so peers may borrow the rest."""
        with self.state.lock:
            self.state.claims[self.index] = 1
            self.state.lent[self.index] = 1

    def pause(self) -> None:
        """While this worker waits for memory: keep one CPU and want no more, so that peers claim the rest at
        their boundaries."""
        self.lend()

    def resume(self) -> int:
        """After a memory wait: grow toward this worker's share; return threads."""
        return self.grow(floor=1)[1]

    def release(self) -> None:
        with self.state.lock:
            self.state.claims[self.index] = 0
            self.state.lent[self.index] = 0

    def owner_boundary(self) -> bool:
        """Whether the calling thread may return CPUs now: the coordinating thread with no open pool or map."""
        return threading.get_ident() == self.owner and self.users == 0

    def enter_user(self) -> None:
        """A pool or map on this lease starts: in the coordinating thread, its own team drops to one (it may not
        hold CPUs the construct's workers are given)."""
        with self.lock:
            self.users += 1
        if threading.get_ident() == self.owner:
            _set_team(1)

    def exit_user(self) -> None:
        """A pool or map on this lease ended. The claim stays until the lane's next boundary, where CPUs above its
        share go back if a peer is short: a lane does not lend its CPUs over the short serial steps between its
        parallel constructs, because peers return borrowed CPUs only at their own boundaries, and a lane waiting
        for them (2.10 candidates: the critical contig's next phase) lost more than the peers gained."""
        with self.lock:
            self.users -= 1


_LEASE_SEAT: _LeaseSeat | None = None


@contextmanager
def lease_seat(seat: _LeaseSeat):
    """Hold ``seat`` as this process's lease for the duration of a phase."""
    global _LEASE_SEAT
    previous = _LEASE_SEAT
    _LEASE_SEAT = seat
    try:
        yield seat
    finally:
        _LEASE_SEAT = previous


class _SlotUser:
    """A pool or map nested in one concurrent in-process task (its slot gives nothing back while it runs)."""

    def __init__(self, slots, index):
        self.slots, self.index = slots, index

    def enter_user(self) -> None:
        with self.slots.lock:
            self.slots.users[self.index] = self.slots.users.get(self.index, 0) + 1

    def exit_user(self) -> None:
        with self.slots.lock:
            self.slots.users[self.index] = max(0, self.slots.users.get(self.index, 0) - 1)


def lease_user(budget):
    """The seat a pool or map on ``budget`` holds while it runs: this process's lease (a contig lane's), pool
    share (a pool task's) or concurrent slot; None for fixed budgets. While held, the seat's own boundaries return
    no CPUs (its children hold them)."""
    if isinstance(budget, LeaseBudget) and _LEASE_SEAT is not None:
        return _LEASE_SEAT
    if isinstance(budget, PoolShareBudget) and _POOL_SEAT is not None:
        return _POOL_SEAT
    if isinstance(budget, SlotBudget):
        return _SlotUser(budget.slots, budget.index)
    if isinstance(budget, HelperBudget):
        return budget.seat
    return None


def lease_exit(seat) -> None:
    """After a pool or map on the seat ended (all its workers and threads finished): the seat's user count drops;
    in a lane's coordinating thread its own team drops to one until its next boundary rebalances the lease. Every
    seat keeps its claim and gives back at its next boundary."""
    seat.exit_user()
    if isinstance(seat, _LeaseSeat) and seat.users == 0 and threading.get_ident() == seat.owner:
        _set_team(1)


class LeaseBudget(CpuBudget):
    """The calling executor worker's lease (live: every threads() call rebalances it, see _LeaseSeat)."""

    live = True

    def __init__(self, phase_name: str | None = None):
        self.phase_name = phase_name

    def _rebalance(self, phase: str | None):
        seat = _LEASE_SEAT
        if seat is None:
            return None
        if seat.owner_boundary():
            # The coordinating thread at its own boundary: its team is about to follow the result.
            before, after = seat.grow(floor=1)
            if _numba().get_num_threads() > after:
                _set_team(after)
        else:
            before, after = seat.grow()
        return before, after

    def threads(self) -> int:
        result = self._rebalance(self.phase_name)
        if result is None:
            return int(_numba().get_num_threads())
        before, after = result
        if after != before and self.phase_name is not None:
            print(f"[lease] {self.phase_name}: {before} -> {after} threads", flush=True)
        return after

    def codec_threads(self) -> int:
        """Compression threads now: the lease at the coordinating thread's own boundary; one while the lane's
        pools or maps hold its CPUs (or from another thread)."""
        seat = _LEASE_SEAT
        if seat is not None and not seat.owner_boundary():
            return 1
        return self.threads()

    def settle(self, in_use: int) -> int:
        """A pool's live total on this lease (see _LeaseSeat.settle)."""
        seat = _LEASE_SEAT
        return int(_numba().get_num_threads()) if seat is None else seat.settle(in_use)

    def current(self) -> int:
        seat = _LEASE_SEAT
        if seat is None:
            return int(_numba().get_num_threads())
        return int(seat.state.claims[seat.index])

    def phase(self, name: str) -> "LeaseBudget":
        result = self._rebalance(name)
        if result is not None and result[0] != result[1]:
            print(f"[lease] {name}: {result[0]} -> {result[1]} threads", flush=True)
        return LeaseBudget(name)

    def __repr__(self):
        return f"LeaseBudget({self.phase_name!r})"


def _team_ceiling() -> int | None:
    """The calling thread's current claim when it owns one: a pool task's thread, or a lane's coordinating
    thread (its lease)."""
    local = getattr(_THREAD_BUDGET, "value", None)
    if isinstance(local, SlotBudget):
        with local.slots.lock:
            if local.slots.threads.get(local.index) == threading.get_ident():
                return max(1, int(local.slots.claims.get(local.index, 1)))
    seat = _POOL_SEAT
    if seat is not None and seat.task_thread == threading.get_ident():
        return max(1, int(seat.claimed))
    if isinstance(local, HelperBudget) and local.seat.task_thread == threading.get_ident():
        return max(1, int(local.seat.claimed))
    lease = _LEASE_SEAT
    if lease is not None and lease.owner == threading.get_ident():
        return max(1, int(lease.state.claims[lease.index]))
    return None


# ---------------------------------------------------------------------------------------------------------------
# The calling process's seat
# ---------------------------------------------------------------------------------------------------------------

_THREAD_BUDGET = threading.local()


@contextmanager
def budget_context(budget):
    """Bind deep numerical claims to this task, including nested thread work."""
    previous = getattr(_THREAD_BUDGET, "value", None)
    _THREAD_BUDGET.value = budget
    try:
        yield
    finally:
        _THREAD_BUDGET.value = previous


def current_budget():
    """Capture the caller's live allowance before creating worker threads."""
    own = getattr(_THREAD_BUDGET, "value", None)
    if own is not None:
        return own
    if _POOL_SEAT is not None:
        return PoolShareBudget()
    if _LEASE_SEAT is not None:
        return LeaseBudget()
    return FixedBudget(int(_numba().get_num_threads()))


def capped_threads(budget, cap: int) -> int:
    """Threads for a phase that cannot use more than ``cap`` well (owner, 25 Oct: no task keeps cores it cannot
    use): at most ``cap`` from the budget; a concurrent slot or pool task owning its claim returns the excess at
    this boundary, so peers take it (the next uncapped boundary grows it back to its share). Results never
    depend on it."""
    cap = max(1, int(cap))
    if isinstance(budget, SlotBudget):
        return max(1, min(budget.claim(cap), cap))
    if isinstance(budget, PoolShareBudget) and _POOL_SEAT is not None and _POOL_SEAT.owns_team():
        return max(1, min(_POOL_SEAT.share(cap), cap))
    return max(1, min(int(budget.threads()), cap))


def claim_threads(*, max_threads: int | None = None) -> int:
    """Boundary call: recompute this process's share, apply it to the calling thread's Numba team, return it.

    Pool worker: its pool share; executor worker: its lease, grown toward its work share; elsewhere the current
    team. ``max_threads`` caps the result for this phase.
    """
    local = getattr(_THREAD_BUDGET, "value", None)
    if local is not None:
        n = local.claim(max_threads) if isinstance(local, SlotBudget) else local.threads()
        if max_threads is not None:
            n = min(n, max_threads)
        numba = _numba()
        n = max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(n)
        return n
    if _POOL_SEAT is not None:
        return _POOL_SEAT.apply(max_threads)
    numba = _numba()
    if _LEASE_SEAT is not None:
        threads = LeaseBudget().threads()
    else:
        threads = int(numba.get_num_threads())
    if max_threads is not None:
        threads = max(1, min(threads, int(max_threads)))
    return _set_team(threads)


# ---------------------------------------------------------------------------------------------------------------
# In-process concurrency on a live budget
# ---------------------------------------------------------------------------------------------------------------

class _Slots:
    """Reusable thread slots of concurrent in-process tasks under one live budget."""

    def __init__(self, workers: int, budget: CpuBudget):
        self.workers = workers
        self.budget = budget
        self.owners = {}
        self.admitted = 0
        self.claims = {}
        self.wants = {}             # each running task's fair (capped) share at its last boundary
        self.threads = {}           # each running task's thread (its boundaries may give back)
        self.waiting = 0            # queued tasks held back for a CPU (peers above their share give back)
        self.users = {}             # pools and maps nested in a task (it gives nothing back while they run)
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def admit(self, index):
        with self.lock:
            if sum(self.claims.values()) >= self.budget.threads():
                return False
            slot = next(i for i in range(self.workers) if i not in self.owners.values())
            self.owners[index] = slot
            self.claims[index] = 1
            self.admitted += 1
            return True

    def start(self, index):
        # Admission already reserved one thread, including workers not yet started.
        assert index in self.claims
        with self.lock:
            self.threads[index] = threading.get_ident()

    def finish(self, index):
        with self.lock:
            del self.owners[index]
            del self.claims[index]
            self.wants.pop(index, None)
            self.threads.pop(index, None)
            self.users.pop(index, None)
            self.admitted -= 1

    def allocation(self, index, max_threads=None) -> int:
        """The task's CPUs from now to its next boundary (see _PoolSeat.share for the rule): its fair share among
        the admitted tasks and those held back for a CPU (the remainder to the lowest slots), capped by
        ``max_threads``; a task above it returns the excess, but only at its own thread's boundary (other threads'
        queries and tasks with nested work never shrink a running team). Admitted tasks that have not started yet
        are counted, so a just-started task never takes their threads."""
        with self.lock:
            if self.stopped.is_set():
                raise CancelledError("another concurrent task failed")
            count = max(1, self.admitted) + self.waiting
            rank = sum(slot < self.owners[index] for slot in self.owners.values())
            cores = int(self.budget.threads())
            want = max(1, cores // count + int(rank < cores % count))
            if max_threads is not None:
                want = min(want, max_threads)
            self.wants[index] = want
            mine = self.claims[index]
            others = sum(value for owner, value in self.claims.items() if owner != index)
            # Never beyond the fair share (a peer that has not reached a boundary yet still counts).
            target = want
            if self.threads.get(index) != threading.get_ident() or self.users.get(index, 0):
                target = max(target, mine)          # another thread's query, or nested work holds the CPUs
            self.claims[index] = max(1, min(target, cores - others)) if target < mine else \
                max(mine, min(target, cores - others))
            return self.claims[index]


class SlotBudget(CpuBudget):
    """A concurrent task's live share (see _Slots.allocation)."""

    live = True

    def __init__(self, slots: _Slots, index: int):
        self.slots = slots
        self.index = index

    def threads(self) -> int:
        n = self.slots.allocation(self.index)
        if self.slots.threads.get(self.index) == threading.get_ident() and _numba().get_num_threads() > n:
            _numba().set_num_threads(n)             # the task's own team stays within its claim
        return n

    def claim(self, maximum=None) -> int:
        return self.slots.allocation(self.index, maximum)


def _threading_layer_is_workqueue() -> bool:
    numba = _numba()
    try:
        return numba.threading_layer() == "workqueue"
    except ValueError:      # not started yet: it will start as the configured layer
        return numba.config.THREADING_LAYER == "workqueue"


def _sizes(task_bytes, count: int) -> list[int]:
    """Per-task byte estimates from None, an int (each task) or a sequence (one per task)."""
    if task_bytes is None:
        return [0] * count
    if isinstance(task_bytes, (list, tuple)) or hasattr(task_bytes, "__len__"):
        sizes = [max(0, int(value)) for value in task_bytes]
        if len(sizes) != count:
            raise ValueError(f"{len(sizes)} byte estimates for {count} tasks")
        return sizes
    return [max(0, int(task_bytes))] * count


def _completed_tasks(budget: CpuBudget, functions, bytes_per_task, label, envelope=False):
    governor = memory_governor()
    sizes = _sizes(bytes_per_task, len(functions))
    depth = depth_below(held_context())

    def reserve(index):
        """The task's memory request; a task without an estimate runs under a zero-byte presence token. Both are
        QUEUED until the task starts: a granted request whose task still waits for a CPU slot is not running work,
        so the governor's progress rule must not count it as such."""
        if not sizes[index]:
            return governor.presence(label, depth, queued=True)
        return governor.request(sizes[index], label=label, depth=depth, envelope=envelope, queued=True)

    # The workqueue layer is not safe for concurrent parallel regions.
    if (_threading_layer_is_workqueue() or len(functions) <= 1
            or (not budget.live and budget.threads() <= 1)):
        for index, function in enumerate(functions):
            request = reserve(index)
            request.wait()
            governor.start(request.id)
            try:
                with adopted(request.held), budget_context(budget), numba_threads(budget.threads()):
                    value = function(budget)
            finally:
                request.release()
            yield index, value
        return
    slots = _Slots(len(functions), budget)
    lease = lease_user(budget)
    if lease is not None:
        lease.enter_user()

    def execute(index, request):
        slots.start(index)
        governor.start(request.id)
        try:
            own = SlotBudget(slots, index)
            with adopted(request.held), budget_context(own), numba_threads(own.threads()):
                return functions[index](own)
        finally:
            slots.finish(index)
            request.release()

    queued = list(range(len(functions)))
    running = {}
    pending = None          # the next task's memory request, keeping its place in the governor's queue
    try:
        yield from _drive_slots(budget, functions, slots, queued, running, reserve, execute, governor)
    finally:
        # The executor has joined every running task (also after a failure): the lease's user ends now.
        if lease is not None:
            lease_exit(lease)


def _drive_slots(budget, functions, slots, queued, running, reserve, execute, governor):
    """Admit and run the queued in-process tasks (see _completed_tasks); yields (index, result) as they finish."""
    pending = None
    with ThreadPoolExecutor(max_workers=len(functions)) as executor:
        try:
            while queued or running:
                limit = max(1, int(budget.threads()))
                # Admit every task that starts now before starting any, so none reads its share while its peers
                # are still being admitted (the shares then sum to the budget).
                starting = []
                slots.waiting = 0
                while queued and len(running) + len(starting) < limit:
                    # A pressure progress grant must have a CPU seat: otherwise an unstarted task could
                    # own the only progress permit while its running sibling waits for nested memory.
                    if not slots.admit(queued[0]):
                        slots.waiting = 1           # peers above their fair share give back for it
                        break
                    if pending is None:
                        pending = reserve(queued[0])
                    if not pending.poll():
                        if running or starting:
                            slots.finish(queued[0])
                            break
                        pending.wait()      # nothing of ours runs: wait here (the governor guarantees progress)
                    starting.append((queued.pop(0), pending))
                    pending = None
                for index, request in starting:
                    running[executor.submit(execute, index, request)] = index
                timeout = POLL_SECONDS if queued else None
                # The caller waits for its tasks, which hold their reservations or presence tokens.
                with governor.suspended():
                    done, _ = wait(running, return_when=FIRST_COMPLETED, timeout=timeout)
                for future in done:
                    yield running.pop(future), future.result()
        except BaseException:
            slots.stopped.set()
            if pending is not None:
                pending.release()
            for future in running:
                future.cancel()
            raise


def _one_thread_map(budget: CpuBudget, function, items, sizes, label, envelope=False):
    results = [None] * len(items)
    state = dict(taken=0, finished=0, workers=0, error=None)
    changed = threading.Condition()
    governor = memory_governor()
    caller = held_context()
    depth, parent = depth_below(caller), caller[0] if caller else 0

    def share():
        return max(1, int(budget.threads()))

    def call(index):
        if not sizes[index]:
            return function(items[index])
        request = governor.request(sizes[index], label=label, depth=depth, envelope=envelope)
        request.wait()
        try:
            with adopted(request.held):
                return function(items[index])
        finally:
            request.release()

    def work():
        # Each worker thread holds a zero-byte presence token while it runs items (running work stays visible;
        # the thread marks it while it waits for an item's memory).
        presence = governor.presence(label, depth, parent=parent)
        try:
            with numba_threads(1), budget_context(FixedBudget(1)), adopted(presence.held):
                while True:
                    with changed:
                        if (state["error"] is not None or state["taken"] == len(items)
                                or state["workers"] > share()):
                            state["workers"] -= 1
                            changed.notify()
                            return
                        index = state["taken"]
                        state["taken"] += 1
                    try:
                        results[index] = call(index)
                    except BaseException as error:
                        with changed:
                            if state["error"] is None:
                                state["error"] = error
                    with changed:
                        state["finished"] += 1
                        changed.notify()
        finally:
            presence.release(record=False)

    started = []
    lease = lease_user(budget)
    if lease is not None:
        lease.enter_user()
    # The caller only waits for its worker threads (which hold presence tokens or their items' reservations).
    with governor.suspended(), changed:
        try:
            while state["finished"] < len(items) and state["error"] is None:
                while state["workers"] < min(share(), len(items) - state["taken"]):
                    state["workers"] += 1
                    started.append(threading.Thread(target=work, daemon=True))
                    started[-1].start()
                changed.wait(POLL_SECONDS)
        except BaseException as error:
            if state["error"] is None:
                state["error"] = error
    for worker in started:
        worker.join()
    if lease is not None:
        lease_exit(lease)
    if state["error"] is not None:
        raise state["error"]
    return results
