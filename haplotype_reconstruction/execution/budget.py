"""CPU budgets: how many threads each piece of work may run, and how that changes as work finishes.

Every allowance is a `CpuBudget`; `threads()` re-reads it at a safe boundary (never inside a running kernel).

* Fixed: a count (the run's cores in serial mode, a pool's or executor's total).
* Lease: a chromosome-executor worker's (a contig lane's) claim on the executor's CPUs. Claims start in
  proportion to each worker's work; at its boundaries (task starts, `phase()` starts, every `threads()` call) a
  worker claims free CPUs toward its share of the work still remaining. Claims are never taken from a peer that
  may be inside a kernel: finished workers release theirs, survivors grow. They sum to at most the budget.
* Pool share: a pool worker's share of its pool's total: floor(total / active workers), plus one of the
  remainder CPUs while it holds a remainder claim. Until a batch's start-up gate opens (min(tasks, workers)
  distinct workers have started, or every task has), each worker runs one thread, so an early worker cannot take
  the CPUs its peers are about to need. A pool whose total is a live lease raises its total as the lease grows.
* Slot share: one of several concurrent in-process tasks (`concurrent`). Like pool seats, slots reserve
  one CPU before admission and retain their claims until an owner boundary releases them. Fair shares guide
  growth, but a task may claim only CPUs that are currently free.

Process x thread totals stay within the budget: a late pool seat or slot waits for a free claim; running
claims are never taken from peers. Thread counts never change results: every parallel kernel reduces per-sample (or per-site)
rows in a fixed order.

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


@contextmanager
def numba_threads(n: int):
    """Run the block with a Numba team of ``n`` threads (the calling thread's setting is restored on exit)."""
    numba = _numba()
    old = numba.get_num_threads()
    numba.set_num_threads(max(1, int(n)))
    try:
        yield
    finally:
        numba.set_num_threads(old)


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
        """Mark the start of phase ``name`` (lanes weight their leases by work progress); return the budget."""
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

    def __init__(self, context, total: int, processes: int, *, yield_to_waiters: bool = False):
        value = context.Value
        self.total = value("i", int(total))        # raised while a live parent budget grows
        self.active = value("i", 0)                # workers inside a task
        self.claimed = value("i", 0)               # retained numerical-team claims
        self.admission = context.Condition(self.claimed.get_lock())
        # Optional discovery policy; only memory-admitted workers enter begin_task.
        # The admission lock protects this count; default pools retain the old policy.
        self.waiting = value("i", 0, lock=False) if yield_to_waiters else None
        self.extra = value("i", 0)                 # remainder claims held
        self.started = value("i", 0)               # tasks started in the current batch
        self.participants = value("i", 0)          # distinct workers started in the current batch
        self.generation = value("i", 0)            # batch number
        self.task_count = value("i", 0)            # tasks in the current batch
        self.startup_target = value("i", min(1, processes))
        self.startup_ready = value("i", 0)

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

    def raise_total(self, threads: int) -> None:
        with self.admission:
            if threads > _read(self.total):
                self._store(self.total, threads)
                self.admission.notify_all()

    def open_gate(self) -> None:
        """Open the current batch's start-up gate now (memory holds the remaining tasks back)."""
        self._store(self.startup_ready, 1)


class _PoolSeat:
    """A pool worker's view of its pool's shares (one per worker process)."""

    def __init__(self, counters: PoolCounters):
        self.counters = counters
        self.has_extra = False
        self.claimed = 0
        self.last_generation = None
        self.last_applied = None
        self.last_logged = None

    def begin_task(self) -> None:
        """Reserve one CPU before admission, then register the task with the start-up gate."""
        c = self.counters
        with c.admission:
            if c.waiting is not None:
                c.waiting.value += 1
            try:
                while _read(c.claimed) >= _read(c.total):
                    c.admission.wait()
            finally:
                if c.waiting is not None:
                    c.waiting.value -= 1
            c.claimed.get_obj().value += 1
            self.claimed = 1
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
        """Release this finished kernel's claim and wake workers waiting for admission."""
        c = self.counters
        with c.admission:
            self._release_extra()
            c.claimed.get_obj().value -= self.claimed
            self.claimed = 0
            with c.active.get_lock():
                c.active.get_obj().value -= 1
            c.admission.notify_all()

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

    def share(self, max_threads: int | None = None) -> int:
        """Claim only free CPUs; peers retain their teams until their own boundary or task end.

        As with in-process slots, an existing claim is not revoked because another
        task starts. A cap releases this caller's excess at its kernel boundary.
        """
        if max_threads is not None:
            max_threads = int(max_threads)
            if max_threads < 1:
                raise ValueError("max_threads must be a positive integer or None")
        c = self.counters
        with c.admission:
            if not self.claimed:  # outside a task, including worker initialization
                return 1
            if not _read(c.startup_ready):
                self._release_extra()
                desired = 1
            else:
                active = max(_read(c.active), 1)
                total = _read(c.total)
                floor, remainder = divmod(total, active)
                if max_threads is not None and max_threads <= floor:
                    self._release_extra()
                else:
                    extras = _read(c.extra)
                    if not self.has_extra and extras < remainder:
                        self._claim_extra(remainder)
                    elif self.has_extra and extras > remainder:
                        self._release_extra()
                desired = max(self.claimed, floor + int(self.has_extra))
            if max_threads is not None:
                desired = min(desired, max_threads)
            available = _read(c.total) - (_read(c.claimed) - self.claimed)
            allocated = max(1, min(desired, available))
            change = allocated - self.claimed
            c.claimed.get_obj().value += change
            self.claimed = allocated
            if change < 0:
                c.admission.notify_all()
            return allocated

    def apply(self, max_threads: int | None = None) -> int:
        """Recompute the share and apply it to the calling thread's Numba team; return it."""
        c = self.counters
        if c.waiting is not None:
            with c.admission:
                waiting = _read(c.waiting)
                if waiting:
                    # This owner boundary immediately resizes the Numba team below.
                    # Pure share()/threads() queries must retain running-team claims.
                    fair = max(1, _read(c.total) // max(1, _read(c.active) + waiting))
                    max_threads = fair if max_threads is None else min(int(max_threads), fair)
        numba = _numba()
        n = min(self.share(max_threads), numba.config.NUMBA_NUM_THREADS)
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


class PoolShareBudget(CpuBudget):
    """The calling pool worker's live share (see _PoolSeat.share)."""

    live = True

    def threads(self) -> int:
        seat = _POOL_SEAT
        return 1 if seat is None else seat.share()

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
    """Shared lease state of one executor: per worker its claim and remaining work, one lock, the budget."""

    def __init__(self, context, workers: int, budget: int):
        self.claims = context.Array("i", workers, lock=False)
        self.remaining = context.Array("d", workers, lock=False)
        self.lock = context.Lock()
        self.budget = int(budget)

    def start_phase(self, loads, initial=None) -> list[int]:
        """Initial claims follow initial work (or whole load); growth tracks all remaining work."""
        with self.lock:
            shares = apportion(self.budget, loads if initial is None else initial)
            for index, share in enumerate(shares):
                self.claims[index] = share
                self.remaining[index] = loads[index]
        return shares


class _LeaseSeat:
    """An executor worker's lease during one phase."""

    def __init__(self, state: LeaseState, index: int, progress: dict):
        self.state = state
        self.index = index
        self.progress = progress      # phase name -> fraction of a task done when that phase starts
        self.task = None              # (weight, work left after the task) of the running task

    def grow(self, phase: str | None = None) -> tuple[int, int]:
        """Claim free CPUs toward this worker's share of the remaining work; return (before, after).

        A ``phase`` listed in the progress table first lowers this worker's remaining work to its later tasks
        plus the unfinished part of the current one, so that shares follow the work still to do inside long
        tasks too.
        """
        state, index = self.state, self.index
        done = self.progress.get(phase)
        with state.lock:
            if done is not None and self.task is not None:
                weight, left = self.task
                state.remaining[index] = left + weight * (1.0 - done)
            before = int(state.claims[index])
            active = [i for i, value in enumerate(state.claims) if value > 0]
            desired = apportion(state.budget, [state.remaining[i] for i in active])[active.index(index)]
            free = state.budget - sum(state.claims)
            state.claims[index] += min(free, max(0, desired - state.claims[index]))
            return before, int(state.claims[index])

    def finish_task(self, left: float) -> None:
        with self.state.lock:
            self.state.remaining[self.index] = left

    def pause(self) -> None:
        """While this worker waits for memory: keep one CPU and no share of the remaining work, so that peers
        claim the rest at their boundaries."""
        state, index = self.state, self.index
        with state.lock:
            self._paused = state.remaining[index]
            state.remaining[index] = 0.0
            state.claims[index] = 1

    def resume(self) -> int:
        """After a memory wait: restore this worker's remaining work and grow toward its share; return threads."""
        with self.state.lock:
            self.state.remaining[self.index] = self._paused
        return self.grow()[1]

    def release(self) -> None:
        with self.state.lock:
            self.state.claims[self.index] = 0


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


class LeaseBudget(CpuBudget):
    """The calling executor worker's lease (live: every threads() call may grow it)."""

    live = True

    def __init__(self, phase_name: str | None = None):
        self.phase_name = phase_name

    def threads(self) -> int:
        seat = _LEASE_SEAT
        if seat is None:
            return int(_numba().get_num_threads())
        before, after = seat.grow(self.phase_name)
        if after != before and self.phase_name is not None:
            print(f"[lease] {self.phase_name}: {before} -> {after} threads", flush=True)
        return after

    def current(self) -> int:
        seat = _LEASE_SEAT
        if seat is None:
            return int(_numba().get_num_threads())
        return int(seat.state.claims[seat.index])

    def phase(self, name: str) -> "LeaseBudget":
        seat = _LEASE_SEAT
        if seat is not None:
            before, after = seat.grow(name)
            if after != before:
                print(f"[lease] {name}: {before} -> {after} threads", flush=True)
        return LeaseBudget(name)

    def __repr__(self):
        return f"LeaseBudget({self.phase_name!r})"


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
        _, threads = _LEASE_SEAT.grow()
    else:
        threads = int(numba.get_num_threads())
    if max_threads is not None:
        threads = max(1, min(threads, int(max_threads)))
    if numba.get_num_threads() != threads:
        numba.set_num_threads(threads)
    return threads


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

    def finish(self, index):
        with self.lock:
            del self.owners[index]
            del self.claims[index]
            self.admitted -= 1

    def allocation(self, index, max_threads=None) -> int:
        """The task's share of the live budget among the admitted (running or starting) tasks; the remainder
        goes to the lowest slots. Admitted tasks that have not started yet are counted, so a just-started task
        never takes their threads."""
        with self.lock:
            if self.stopped.is_set():
                raise CancelledError("another concurrent task failed")
            count = max(1, self.admitted)
            rank = sum(slot < self.owners[index] for slot in self.owners.values())
            cores = int(self.budget.threads())
            desired = max(self.claims[index], cores // count + int(rank < cores % count))
            # Existing kernels keep their claims. A new task cannot assume peers
            # have already applied a smaller fair share. Only an explicit cap at
            # the owner's kernel boundary releases some of its own claim.
            if max_threads is not None:
                desired = min(desired, max_threads)
            available = cores - sum(value for owner, value in self.claims.items() if owner != index)
            self.claims[index] = max(1, min(desired, available))
            return self.claims[index]


class SlotBudget(CpuBudget):
    """A concurrent task's live share (see _Slots.allocation)."""

    live = True

    def __init__(self, slots: _Slots, index: int):
        self.slots = slots
        self.index = index

    def threads(self) -> int:
        return self.slots.allocation(self.index)

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
        """The task's memory request; a task without an estimate runs under a zero-byte presence token."""
        if not sizes[index]:
            return governor.presence(label, depth)
        return governor.request(sizes[index], label=label, depth=depth, envelope=envelope)

    # The workqueue layer is not safe for concurrent parallel regions.
    if (_threading_layer_is_workqueue() or len(functions) <= 1
            or (not budget.live and budget.threads() <= 1)):
        for index, function in enumerate(functions):
            request = reserve(index)
            request.wait()
            try:
                with adopted(request.held), budget_context(budget), numba_threads(budget.threads()):
                    value = function(budget)
            finally:
                request.release()
            yield index, value
        return
    slots = _Slots(len(functions), budget)

    def execute(index, request):
        slots.start(index)
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
    with ThreadPoolExecutor(max_workers=len(functions)) as executor:
        try:
            while queued or running:
                limit = max(1, int(budget.threads()))
                # Admit every task that starts now before starting any, so none reads its share while its peers
                # are still being admitted (the shares then sum to the budget).
                starting = []
                while queued and len(running) + len(starting) < limit:
                    if pending is None:
                        pending = reserve(queued[0])
                    if not pending.poll():
                        if running or starting:
                            break
                        pending.wait()      # nothing of ours runs: wait here (the governor guarantees progress)
                    if not slots.admit(queued[0]):
                        break
                    starting.append((queued.pop(0), pending))
                    pending = None
                for index, request in starting:
                    running[executor.submit(execute, index, request)] = index
                timeout = POLL_SECONDS if queued and (budget.live or pending is not None) else None
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
    if state["error"] is not None:
        raise state["error"]
    return results
