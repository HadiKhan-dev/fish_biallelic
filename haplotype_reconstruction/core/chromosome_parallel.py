"""Persistent chromosome workers with a bounded, phase-local CPU budget.

Tasks and results should be small descriptors. Large chromosome tensors belong
in the handler's retained state and never cross the result pipe. A handler is a
module-scope callable ``(task, state, phase, common) -> (state, result)``.
"""
from __future__ import annotations

import functools
import math
import operator
import os
import traceback
from multiprocessing.connection import wait

import numba

from . import parallel


_LEASE = None
# In an executor worker: the executor's phase_progress, and (weight, work left
# after it) of the task the worker is running, for lease_threads(phase=...).
_PROGRESS = {}
_TASK = None


def memory_worker_limit(estimates, workers, available, *, label):
    """Limit concurrent chromosome workspaces without lowering the CPU ceiling.

    Stages supply decoded peak-byte estimates, including their own numerical
    copies and per-process overhead. Budget the largest possible concurrent
    group, retaining 10% headroom (at least 4 GiB, at most half of available
    memory). Missing memory information or an oversized chromosome retains the
    one-worker route; this is an estimate, not a guarantee that it fits.
    """
    if not estimates or available is None:
        return 1
    reserve = min(available // 2, max(4 << 30, available // 10))
    usable = max(0, available - reserve)
    total = count = 0
    for estimate in sorted(estimates, reverse=True)[:workers]:
        if total + estimate > usable:
            break
        total += estimate
        count += 1
    if not count:
        print(f"{label} memory estimate exceeds available memory; "
              "retaining one chromosome worker", flush=True)
        return 1
    print(f"{label} memory budget: {count} chromosome workers; "
          f"estimated concurrent peak {total / 2**30:.1f} GiB, "
          f"usable {usable / 2**30:.1f} GiB", flush=True)
    return count


def _apportion(budget, work):
    """Split ``budget`` CPUs over workers in proportion to their work.

    Each worker keeps one CPU; the others follow the largest remainders of the
    proportional shares (ties to the lower index). Equal shares if no work is
    known. Requires at least one CPU per worker.
    """
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


def current_threads():
    """Grow this worker's lease at a numeric boundary; otherwise change nothing.

    The target is this worker's share of the CPU budget in proportion to the
    work it still has (its partition's unfinished tasks, the current one
    included) among the workers still running. Claims are never stolen from a
    peer that may still be inside a kernel: finished partitions release their
    whole lease, and surviving workers claim free capacity toward their targets
    at their next boundary (a task start here, a phase start through
    lease_threads, or while a phase runs through lease_budget). No mid-kernel
    resizing is attempted. The shared claim sum cannot exceed the executor's
    CPU budget.
    """
    if _LEASE is None:
        return numba.get_num_threads()
    _, threads = _grow()
    if numba.get_num_threads() != threads:
        numba.set_num_threads(threads)
    return threads


def _grow(phase=None):
    """Claim free capacity toward this worker's share; return (before, after).

    A ``phase`` of the executor's phase_progress first lowers this worker's
    remaining work to its later tasks plus the unfinished part of the current
    one (all but the fraction done when that phase starts), so that shares
    follow the work still to do inside long tasks too.
    """
    claims, remaining, lock, index, budget = _LEASE
    done = _PROGRESS.get(phase)
    with lock:
        if done is not None and _TASK is not None:
            weight, left = _TASK
            remaining[index] = left + weight * (1.0 - done)
        before = int(claims[index])
        active = [i for i, value in enumerate(claims) if value > 0]
        desired = _apportion(budget, [remaining[i] for i in active])[active.index(index)]
        free = budget - sum(claims)
        claims[index] += min(free, max(0, desired - claims[index]))
        return before, int(claims[index])


def lease_threads(default=None, phase=None):
    """A running task's CPU lease at the start of one of its phases.

    Inside an executor worker this grows the lease as current_threads() does
    at task starts, so capacity that finished partitions released reaches the
    running tasks at their next phase instead of their next task. ``phase``
    names the phase (phase_progress, log). The caller sizes its own pools and
    Numba scopes from the returned count; the calling thread's Numba team is
    unchanged. Outside a worker it returns ``default`` (None: the current
    Numba team size). Leases are execution choices: every phase's result is
    invariant to its thread and process counts. A module-level function, so
    functools.partial(lease_threads, n, phase) is a picklable dynamic budget
    that a long phase can re-read as it runs (it returns n in pool processes).
    """
    if _LEASE is None:
        return numba.get_num_threads() if default is None else int(default)
    before, threads = _grow(phase)
    if threads != before:
        print(f"[lease] {phase}: {before} -> {threads} threads", flush=True)
    return threads


def lease_budget(default, phase):
    """A phase's CPU budget that, in a contig lane, keeps growing while it runs.

    Outside an executor worker this is lease_threads(default, phase), an int,
    so the phase runs as before. Inside one it is the picklable callable
    functools.partial(lease_threads, threads, phase), which the phase re-reads
    as it runs: pool phases through core/parallel.live_total (a parent thread
    raises the workers' shared total every LIVE_TOTAL_SECONDS), the founder
    refinement at every sweep allocation. CPUs that finished partitions release
    then reach a running phase without waiting for its next phase start. The
    claim rule is lease_threads': never stolen, the claim sum within the budget.
    """
    threads = lease_threads(default, phase)
    if _LEASE is None:
        return threads
    return functools.partial(lease_threads, threads, phase)


def _worker(connection, partition, weights, handler, claims, remaining, lock, index, budget,
            progress):
    global _LEASE, _PROGRESS, _TASK
    states = {task_index: None for task_index, _ in partition}
    # Work left after each task of the partition, in task order.
    after = [sum(weights[position + 1:]) for position in range(len(weights))]
    _PROGRESS = progress
    try:
        capacity = min(len(os.sched_getaffinity(0)), numba.config.NUMBA_NUM_THREADS)
        if capacity < budget:
            raise RuntimeError(f"chromosome worker capacity {capacity} < budget {budget}")
        numba.set_num_threads(1)
        connection.send(("ready", None))
        while True:
            command = connection.recv()
            if command is None:
                break
            phase, common = command
            _LEASE = (claims, remaining, lock, index, budget)
            # Apply the exact initial allocation before allowing growth.
            numba.set_num_threads(int(claims[index]))
            results = []
            for weight, (task_index, task), left in zip(weights, partition, after):
                current_threads()
                _TASK = (weight, left)
                states[task_index], result = handler(task, states[task_index], phase, common)
                _TASK = None
                results.append((task_index, result))
                with lock:
                    remaining[index] = left
            numba.set_num_threads(1)
            with lock:
                claims[index] = 0
            _LEASE = None
            connection.send(("result", results))
    except BaseException:
        # A string traceback also handles exceptions that cannot be pickled.
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        _LEASE = _TASK = None
        connection.close()


class ChromosomeExecutor:
    """Retain chromosome-local state across serially requested phase barriers.

    ``n_workers`` is the total CPU budget, not a process-times-thread multiplier.
    It is bounded by affinity and configured Numba capacity. There are at most
    min(number of tasks, CPU budget) persistent processes. ``weights`` are
    nonnegative workload estimates in task order; largest tasks are assigned
    first to the least-loaded partition. Each phase starts every worker with a
    lease proportional to its partition's work, and leases grow toward shares of
    the work still remaining (current_threads at task starts, lease_threads at
    the handler's own phase starts, lease_budget while its phases run).
    ``phase_progress`` optionally maps the phase names a handler passes to
    lease_threads to the fraction of a task done when that phase starts.
    Results always retain input order.
    ``max_workers`` optionally limits concurrent processes for memory-heavy
    tasks without reducing the total CPU budget shared between those processes.
    """

    def __init__(self, tasks, handler, *, n_workers=None, weights=None,
                 label="CHROMOSOME", max_workers=None, phase_progress=None):
        self.tasks = tuple(tasks)
        self.handler = handler
        self.label = label
        self.phase_progress = dict(phase_progress or {})
        capacity = min(len(os.sched_getaffinity(0)), numba.config.NUMBA_NUM_THREADS)
        requested = numba.get_num_threads() if n_workers is None else operator.index(n_workers)
        if requested < 1:
            raise ValueError("n_workers must be positive")
        self.cpu_budget = min(requested, capacity)
        process_limit = self.cpu_budget if max_workers is None else operator.index(max_workers)
        if process_limit < 1:
            raise ValueError("max_workers must be positive")
        self.worker_count = min(len(self.tasks), self.cpu_budget, process_limit)
        values = [1.0] * len(self.tasks) if weights is None else list(weights)
        if len(values) != len(self.tasks) or any(
                not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("weights must be finite nonnegative values in task order")
        self.partitions = [[] for _ in range(self.worker_count)]
        self._loads = [0.0] * self.worker_count
        for task_index in sorted(range(len(self.tasks)), key=lambda i: (-values[i], i)):
            owner = min(range(self.worker_count),
                        key=lambda i: (self._loads[i], len(self.partitions[i]), i))
            self.partitions[owner].append((task_index, self.tasks[task_index]))
            self._loads[owner] += values[task_index]
        self._weights = values
        context = parallel.forkserver_context
        self._claims = context.Array("i", self.worker_count, lock=False)
        self._remaining = context.Array("d", self.worker_count, lock=False)
        self._lock = context.Lock()
        self._connections = []
        self._processes = []
        self._states = [None] * len(self.tasks)
        self._entered = False
        self._closed = False

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("chromosome executor cannot be entered twice")
        self._entered = True
        if self.worker_count <= 1:
            return self
        try:
            # Match the established pool guard: workflow __main__ must not be
            # re-executed in forkserver children. Handlers must be importable.
            with parallel.main_module_guard():
                for index, partition in enumerate(self.partitions):
                    parent, child = parallel.forkserver_context.Pipe()
                    process = parallel.forkserver_context.Process(
                        target=_worker,
                        args=(child, partition,
                              [self._weights[task_index] for task_index, _ in partition],
                              self.handler, self._claims, self._remaining,
                              self._lock, index, self.cpu_budget, self.phase_progress),
                    )
                    try:
                        process.start()
                    except BaseException:
                        parent.close()
                        child.close()
                        raise
                    child.close()
                    self._connections.append(parent)
                    self._processes.append(process)
            self._receive("ready")
        except BaseException:
            self._shutdown(abort=True)
            raise
        return self

    def _receive(self, expected):
        pending = set(range(self.worker_count))
        results = []
        while pending:
            connections = {self._connections[i]: i for i in pending}
            sentinels = {process.sentinel: i for i, process in enumerate(self._processes)}
            ready = wait([*connections, *sentinels])
            for connection in ready:
                if connection not in connections:
                    continue
                index = connections[connection]
                try:
                    kind, value = connection.recv()
                except EOFError as error:
                    raise RuntimeError(f"chromosome worker {index} closed its pipe") from error
                if kind == "error":
                    raise RuntimeError(f"chromosome worker {index} failed:\n{value}")
                if kind != expected:
                    raise RuntimeError(f"chromosome worker {index}: expected {expected}, got {kind}")
                results.append(value)
                pending.remove(index)
            for sentinel in ready:
                if sentinel in sentinels:
                    index = sentinels[sentinel]
                    raise RuntimeError(f"chromosome worker {index} exited unexpectedly")
        return results

    def run(self, phase, common=None):
        """Run a barrier phase and return small results in original task order."""
        global _LEASE
        if not self._entered or self._closed:
            raise RuntimeError("run requires an open chromosome executor context")
        if not self.tasks:
            return []
        with self._lock:
            for index, share in enumerate(_apportion(self.cpu_budget, self._loads)):
                self._claims[index] = share
                self._remaining[index] = self._loads[index]
        print(f"{self.label} chromosome phase={phase}: {self.worker_count} workers, "
              f"{self.cpu_budget} CPUs; initial threads={list(self._claims)}", flush=True)
        results = [None] * len(self.tasks)
        try:
            if self.worker_count == 1:
                old_lease = _LEASE
                try:
                    _LEASE = (self._claims, self._remaining, self._lock, 0, self.cpu_budget)
                    with parallel.numba_thread_scope(self.cpu_budget):
                        for index, task in self.partitions[0]:
                            self._states[index], results[index] = self.handler(
                                task, self._states[index], phase, common)
                finally:
                    _LEASE = old_lease
                    self._claims[0] = 0
                return results
            for connection in self._connections:
                connection.send((phase, common))
            for partition_results in self._receive("result"):
                for index, result in partition_results:
                    results[index] = result
            return results
        except BaseException:
            self._shutdown(abort=True)
            raise

    def _shutdown(self, *, abort):
        if self._closed:
            return
        self._closed = True
        if not abort:
            for connection in self._connections:
                try:
                    connection.send(None)
                except (OSError, EOFError):
                    pass
        for process in self._processes:
            if abort and process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=1)
        for process in self._processes:
            if process.is_alive():
                process.kill()
            process.join()
            process.close()
        for connection in self._connections:
            connection.close()

    def __exit__(self, exc_type, exc_value, tb):
        self._shutdown(abort=exc_type is not None)
