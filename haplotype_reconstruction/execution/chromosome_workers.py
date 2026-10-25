"""Persistent chromosome workers that keep each chromosome's state resident across phase barriers.

Tasks and results are small descriptors; large chromosome tensors live in the handler's retained ``state`` and
never cross the result pipe. A handler is a module-level callable
``(task, state, phase, common, budget) -> (state, result)``; ``budget`` is the worker's lease
(execution.budget): it starts in proportion to the worker's work and grows, at the handler's boundaries, toward
the worker's share of the work still remaining.

Memory (execution.memory): each handler call holds its task's ``task_bytes`` of the run's memory and starts when
they fit; ``state_bytes`` of it stay reserved between calls while the task's retained state lives (until the
executor closes). A worker waiting for memory hands its lease's CPUs to its peers meanwhile. Each call's peak RSS
is measured for the run's statistics. A worker process that holds no task state is recycled after a task when
its private memory has grown by more than the executor's threshold since its first task (execution.pools): it
reports the phase's results so far and exits, and a fresh worker continues its partition (results unchanged).
"""
from __future__ import annotations

import math
import operator
import traceback
from multiprocessing.connection import wait

from . import bootstrap
from .budget import LeaseBudget, LeaseState, _LeaseSeat, as_budget, lease_seat, numba_threads
from .memory import (adopted, depth_below, held_context, memory_governor, release_memory, task_peak_bytes,
                     task_peak_start)
from .pools import FORKSERVER, _private_bytes, main_module_guard, record_recycle, recycle_threshold, task_sizes
from .resources import affinity_cpus


def memory_worker_limit(estimates, workers: int, *, label: str, available: int | None = None) -> int:
    """Limit concurrent chromosome workspaces without lowering the CPU budget.

    ``estimates`` are peak bytes per task (including the stage's own copies and per-process overhead). The
    largest possible concurrent group is budgeted against the run's memory limit (execution.memory), or against
    ``available`` bytes minus the headroom when given. An oversized chromosome keeps one worker. The tasks still
    reserve their bytes as they run (``task_bytes``), so this only avoids starting workers that would wait.
    """
    return memory_governor().worker_limit(estimates, workers, label=label, available=available)


class _TaskMemory:
    """One process's reservations for an executor's tasks: a call reservation around each handler call, and for
    tasks with retained state a state reservation kept between calls until the executor closes."""

    def __init__(self, label: str, depth: int, parent: int = 0, envelope: bool = False):
        self.governor = memory_governor()
        self.label = label
        self.depth = depth
        self.parent = parent
        self.envelope = envelope
        self.states = {}

    def acquire(self, task_index, task_bytes: int, *, seat=None):
        """Reserve the call's bytes (task_bytes minus the task's retained-state reservation); wait while they do
        not fit, handing the lease's CPUs to peers meanwhile. Returns (request, threads after a wait or None)."""
        state = self.states.get(task_index)
        held = state.nbytes if state is not None else 0
        # Zero-byte calls use the same admission/CPU-yield path. Byte metadata never determines whether
        # a waiting handler gives its CPU lease to a peer.
        request = self.governor.request(max(0, task_bytes - held), label=self.label, depth=self.depth,
                                       parent=self.parent, envelope=self.envelope, outer=True)
        if request.poll():
            return request, None
        if seat is None:
            request.wait()
            return request, None
        seat.pause()
        try:
            request.wait()
        finally:
            threads = seat.resume()
        return request, threads

    def held(self, task_index, request):
        """The reservation a handler call runs under (for the depth and state of nested requests)."""
        return request.held

    def finish(self, task_index, request, task_bytes: int, state_bytes: int, measured) -> None:
        """After the call: keep ``state_bytes`` reserved for the retained state, release the rest. A state
        reservation is resident memory, not running work: it never keeps the progress rule from seeing that
        this worker waits."""
        if state_bytes > 0 and task_index not in self.states and request.nbytes:
            request.shrink(min(state_bytes, request.nbytes), measured=measured, estimate=task_bytes)
            self.governor.rest(request.id)
            self.states[task_index] = request
        else:
            request.release(measured, estimate=task_bytes)

    def close(self) -> None:
        for state in self.states.values():
            state.release(record=False)
        self.states.clear()


def _numba_capacity() -> int:
    import numba
    return min(affinity_cpus(), int(numba.config.NUMBA_NUM_THREADS))


def _worker(connection, partition, weights, handler, state, index, progress, settings, label, recycle_bytes=0):
    """An executor worker: run every phase's tasks of its partition in order (a phase command names the first
    one), keeping their states. Holding no state, it retires after a task once its private memory exceeds its
    baseline (after its first task) by ``recycle_bytes``: it sends ("recycle", (results so far, next position))
    and exits, keeping its lease claim for its replacement."""
    import numba
    bootstrap.apply_worker_settings(settings)
    states = {task_index: None for task_index, _ in partition}
    # Work left after each task of the partition, in task order.
    after = [sum(weights[position + 1:]) for position in range(len(weights))]
    memory = None
    baseline = None
    tasks_run = 0
    try:
        if _numba_capacity() < state.budget:
            raise RuntimeError(f"chromosome worker capacity {_numba_capacity()} < budget {state.budget}")
        numba.set_num_threads(1)
        connection.send(("ready", None))
        while True:
            command = connection.recv()
            if command is None:
                break
            phase, common, sizes, state_sizes, depth, parent, envelope, first_position = command
            retire_at = private = None
            if memory is None:
                memory = _TaskMemory(f"{label} {phase}", depth, parent, envelope)
            memory.label = f"{label} {phase}"
            seat = _LeaseSeat(state, index, progress)
            with lease_seat(seat):
                # The exact initial allocation applies before any growth.
                numba.set_num_threads(int(state.claims[index]))
                results = []
                for position, (weight, (task_index, task), left) in enumerate(zip(weights, partition, after)):
                    if position < first_position:     # done by a retired predecessor in this phase
                        continue
                    _, threads = seat.grow()
                    request, resumed = memory.acquire(task_index, sizes[task_index], seat=seat)
                    threads = resumed or threads
                    if numba.get_num_threads() != threads:
                        numba.set_num_threads(threads)
                    seat.task = (weight, left)
                    start = task_peak_start()
                    try:
                        with adopted(memory.held(task_index, request)):
                            states[task_index], result = handler(task, states[task_index], phase, common,
                                                                 LeaseBudget())
                    except BaseException:
                        request.release()
                        raise
                    memory.finish(task_index, request, sizes[task_index], state_sizes[task_index],
                                  task_peak_bytes(start))
                    seat.task = None
                    results.append((task_index, result))
                    seat.finish_task(left)
                    release_memory()
                    tasks_run += 1
                    if recycle_bytes and all(value is None for value in states.values()):
                        private = _private_bytes()
                        if baseline is None:
                            baseline = private
                        elif private is not None and private - baseline > recycle_bytes:
                            retire_at = position + 1
                            break
                numba.set_num_threads(1)
                if retire_at is None:
                    seat.release()          # a retiring worker's claim passes to its replacement
            if retire_at is not None:
                record_recycle(label, private, baseline, recycle_bytes, tasks_run)
                connection.send(("recycle", (results, retire_at)))
                break
            connection.send(("result", results))
    except BaseException:
        # A string traceback also carries exceptions that cannot be pickled.
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if memory is not None:
            memory.close()
        connection.close()


class ChromosomeExecutor:
    """Retain chromosome-local state across serially requested phase barriers.

    ``budget`` (a CpuBudget or thread count) is the total CPU budget shared by the workers, bounded by affinity
    and Numba's pool size; there are at most min(tasks, budget, max_workers) persistent processes.
    ``frontload`` apportions initial leases by each partition's first task; later growth still follows its
    remaining total work. This gives long first tasks their cores before short queued tasks begin.
    ``weights`` are nonnegative workload estimates in task order: the largest tasks go first to the
    least-loaded partition, and every phase starts each worker with a lease proportional to its partition's
    work. ``phase_progress`` maps phase names that handlers pass to ``budget.phase(name)`` to the fraction of a
    task done when that phase starts. ``max_workers`` limits concurrent processes for memory-heavy tasks without
    reducing the CPU budget. Results always keep task order. One worker runs in the calling process.
    ``task_bytes`` (None, an int for every task, or one per task in task order) are each handler call's peak bytes
    and ``state_bytes`` the part the task's retained state keeps between calls (execution.memory); `run` may give
    a phase its own ``task_bytes``; ``envelope``: a call's bytes also cover what its nested constructs reserve.
    """

    def __init__(self, tasks, handler, *, budget, weights=None, max_workers=None, label="CHROMOSOME",
                 phase_progress=None, task_bytes=None, state_bytes=None, envelope=False, recycle_bytes=None,
                 frontload=False):
        self.tasks = tuple(tasks)
        self.handler = handler
        self.label = label
        self.phase_progress = dict(phase_progress or {})
        self.task_bytes = task_sizes(task_bytes, self.tasks)
        self.state_bytes = task_sizes(state_bytes, self.tasks)
        self.envelope = bool(envelope)
        self._memory = None
        requested = as_budget(budget).threads()
        self.cpu_budget = min(requested, _numba_capacity())
        process_limit = self.cpu_budget if max_workers is None else operator.index(max_workers)
        if process_limit < 1:
            raise ValueError("max_workers must be positive")
        self.worker_count = max(1, min(len(self.tasks), self.cpu_budget, process_limit))
        values = [1.0] * len(self.tasks) if weights is None else [float(value) for value in weights]
        if len(values) != len(self.tasks) or any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("weights must be finite nonnegative values in task order")
        self.partitions = [[] for _ in range(self.worker_count)]
        self._loads = [0.0] * self.worker_count
        for task_index in sorted(range(len(self.tasks)), key=lambda i: (-values[i], i)):
            owner = min(range(self.worker_count), key=lambda i: (self._loads[i], len(self.partitions[i]), i))
            self.partitions[owner].append((task_index, self.tasks[task_index]))
            self._loads[owner] += values[task_index]
        self._weights = values
        self._initial_loads = ([values[partition[0][0]] for partition in self.partitions]
                               if frontload and self.tasks else None)
        self._state = LeaseState(FORKSERVER, self.worker_count, self.cpu_budget)
        # Worker processes are recycled (execution.pools.recycle_threshold); the in-process worker never is.
        self.recycle_bytes = recycle_threshold(self.worker_count, recycle_bytes) if self.worker_count > 1 else 0
        self._command = None
        self._connections = []
        self._processes = []
        self._states = [None] * len(self.tasks)
        self._entered = False
        self._closed = False

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("a chromosome executor cannot be entered twice")
        self._entered = True
        if self.worker_count <= 1:
            return self
        try:
            for index in range(len(self.partitions)):
                connection, process = self._start_worker(index)
                self._connections.append(connection)
                self._processes.append(process)
            self._receive("ready")
        except BaseException:
            self._shutdown(abort=True)
            raise
        return self

    def _start_worker(self, index):
        """Start the worker process of partition ``index``: (parent connection, process)."""
        partition = self.partitions[index]
        parent, child = FORKSERVER.Pipe()
        process = FORKSERVER.Process(
            target=_worker,
            args=(child, partition, [self._weights[task_index] for task_index, _ in partition], self.handler,
                  self._state, index, self.phase_progress, bootstrap.worker_settings(), self.label,
                  self.recycle_bytes))
        try:
            with main_module_guard():
                process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        return parent, process

    def _replace(self, index):
        """A fresh worker for partition ``index`` in place of one that retired (it exits by itself)."""
        connection, process = self._connections[index], self._processes[index]
        process.join(timeout=60)
        if process.is_alive():              # our own child that did not exit after retiring
            process.kill()
            process.join()
        process.close()
        connection.close()
        connection, process = self._start_worker(index)
        self._connections[index], self._processes[index] = connection, process
        if connection not in wait([connection, process.sentinel]):
            raise RuntimeError(f"chromosome worker {index}: its replacement exited at start")
        try:
            kind, value = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"chromosome worker {index}: its replacement closed its pipe") from error
        if kind == "error":
            raise RuntimeError(f"chromosome worker {index} (replacement) failed:\n{value}")
        if kind != "ready":
            raise RuntimeError(f"chromosome worker {index} (replacement): expected ready, got {kind}")

    def _receive(self, expected):
        pending = set(range(self.worker_count))
        partial = {index: [] for index in pending}          # results of retired workers in this phase
        results = []
        while pending:
            connections = {self._connections[i]: i for i in pending}
            sentinels = {process.sentinel: i for i, process in enumerate(self._processes)}
            ready = wait([*connections, *sentinels])
            replaced = set()
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
                if kind == "recycle" and expected == "result":
                    done, position = value
                    partial[index].extend(done)
                    self._replace(index)
                    replaced.add(index)
                    if position < len(self.partitions[index]):
                        self._connections[index].send(self._command + (position,))
                        continue
                    value = []
                elif kind != expected:
                    raise RuntimeError(f"chromosome worker {index}: expected {expected}, got {kind}")
                results.append(partial[index] + value if expected == "result" else value)
                pending.remove(index)
            for sentinel in ready:
                if sentinel in sentinels and sentinels[sentinel] not in replaced:
                    raise RuntimeError(f"chromosome worker {sentinels[sentinel]} exited unexpectedly")
        return results

    def run(self, phase: str, common=None, *, task_bytes=None) -> list:
        """Run a barrier phase and return its small results in task order (``task_bytes``: this phase's)."""
        if not self._entered or self._closed:
            raise RuntimeError("run requires an open chromosome executor context")
        if not self.tasks:
            return []
        sizes = self.task_bytes if task_bytes is None else task_sizes(task_bytes, self.tasks)
        caller = held_context()
        depth, parent = depth_below(caller), caller[0] if caller else 0
        shares = self._state.start_phase(self._loads, initial=self._initial_loads)
        print(f"{self.label} chromosome phase={phase}: {self.worker_count} workers, {self.cpu_budget} CPUs; "
              f"initial threads={shares}", flush=True)
        results = [None] * len(self.tasks)
        try:
            if self.worker_count == 1:
                if self._memory is None:
                    self._memory = _TaskMemory(f"{self.label} {phase}", depth, parent, self.envelope)
                memory = self._memory
                memory.label = f"{self.label} {phase}"
                seat = _LeaseSeat(self._state, 0, self.phase_progress)
                try:
                    with lease_seat(seat), numba_threads(self.cpu_budget):
                        for index, task in self.partitions[0]:
                            request, _ = memory.acquire(index, sizes[index])
                            try:
                                with adopted(memory.held(index, request)):
                                    self._states[index], results[index] = self.handler(
                                        task, self._states[index], phase, common, LeaseBudget())
                            except BaseException:
                                request.release()
                                raise
                            memory.finish(index, request, sizes[index], self.state_bytes[index], None)
                            release_memory()
                finally:
                    seat.release()
                return results
            self._command = (phase, common, sizes, self.state_bytes, depth, parent, self.envelope)
            for connection in self._connections:
                connection.send(self._command + (0,))
            with memory_governor().suspended():
                received = self._receive("result")
            for partition_results in received:
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
        if self._memory is not None:
            self._memory.close()
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
