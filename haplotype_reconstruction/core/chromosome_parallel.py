"""Persistent chromosome workers with a bounded, phase-local CPU budget.

Tasks and results should be small descriptors. Large chromosome tensors belong
in the handler's retained state and never cross the result pipe. A handler is a
module-scope callable ``(task, state, phase, common) -> (state, result)``.
"""
from __future__ import annotations

import math
import operator
import os
import traceback
from multiprocessing.connection import wait

import numba

from . import parallel


_LEASE = None


def current_threads():
    """Grow this worker's lease at a numeric boundary; otherwise change nothing.

    Claims are never stolen from a peer that may still be inside a kernel.
    Finished partitions release their whole lease, and surviving workers can
    claim the free capacity at their next boundary. No mid-kernel resizing is
    attempted. The shared claim sum cannot exceed the executor's CPU budget.
    """
    if _LEASE is None:
        return numba.get_num_threads()
    claims, lock, index, budget = _LEASE
    with lock:
        active = [i for i, value in enumerate(claims) if value > 0]
        floor, remainder = divmod(budget, len(active))
        desired = floor + (active.index(index) < remainder)
        free = budget - sum(claims)
        claims[index] += min(free, max(0, desired - claims[index]))
        threads = int(claims[index])
    if numba.get_num_threads() != threads:
        numba.set_num_threads(threads)
    return threads


def _worker(connection, partition, handler, claims, lock, index, budget):
    global _LEASE
    states = {task_index: None for task_index, _ in partition}
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
            _LEASE = (claims, lock, index, budget)
            # Apply the exact initial allocation before allowing tail growth.
            numba.set_num_threads(int(claims[index]))
            results = []
            for task_index, task in partition:
                current_threads()
                states[task_index], result = handler(task, states[task_index], phase, common)
                results.append((task_index, result))
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
        _LEASE = None
        connection.close()


class ChromosomeExecutor:
    """Retain chromosome-local state across serially requested phase barriers.

    ``n_workers`` is the total CPU budget, not a process-times-thread multiplier.
    It is bounded by affinity and configured Numba capacity. There are at most
    min(number of tasks, CPU budget) persistent processes. ``weights`` are
    nonnegative workload estimates in task order; largest tasks are assigned
    first to the least-loaded partition. Results always retain input order.
    ``max_workers`` optionally limits concurrent processes for memory-heavy
    tasks without reducing the total CPU budget shared between those processes.
    """

    def __init__(self, tasks, handler, *, n_workers=None, weights=None,
                 label="CHROMOSOME", max_workers=None):
        self.tasks = tuple(tasks)
        self.handler = handler
        self.label = label
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
        loads = [0.0] * self.worker_count
        for task_index in sorted(range(len(self.tasks)), key=lambda i: (-values[i], i)):
            owner = min(range(self.worker_count), key=lambda i: (loads[i], len(self.partitions[i]), i))
            self.partitions[owner].append((task_index, self.tasks[task_index]))
            loads[owner] += values[task_index]
        context = parallel.forkserver_context
        self._claims = context.Array("i", self.worker_count, lock=False)
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
                        args=(child, partition, self.handler, self._claims,
                              self._lock, index, self.cpu_budget),
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
        floor, remainder = divmod(self.cpu_budget, self.worker_count)
        with self._lock:
            for index in range(self.worker_count):
                self._claims[index] = floor + (index < remainder)
        print(f"{self.label} chromosome phase={phase}: {self.worker_count} workers, "
              f"{self.cpu_budget} CPUs; initial threads={list(self._claims)}", flush=True)
        results = [None] * len(self.tasks)
        try:
            if self.worker_count == 1:
                old_lease = _LEASE
                try:
                    _LEASE = (self._claims, self._lock, 0, self.cpu_budget)
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
