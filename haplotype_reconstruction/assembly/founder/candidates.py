"""Work sharing among independent conditional founder-path candidates.

Keep queued/active candidates in the budget until they finish. With enough
candidates each receives one thread; in the tail the survivors grow at a scan
boundary. Native kernels release the GIL; all checkpoint I/O and acceptance
decisions stay on the caller thread.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from threading import Lock, Event
import os

import numba

from ...core import parallel
from.workspace import resolve_threads


def completed_candidates(functions, budget, bytes_per_candidate):
    """Yield (input index, result), without prescribing scientific selection."""
    total = int(resolve_threads(budget))
    workers = min(len(functions), total)
    from ...painting.model import available_process_memory_bytes
    available = available_process_memory_bytes()
    if available is not None:
        # Count-refit processes share a node. Restrict this thread team's
        # estimate to its CPU-budget fraction of the current RAM allowance.
        allowance = available * min(1., total / len(os.sched_getaffinity(0)))
        workers = min(workers, max(1, int(allowance // max(1, bytes_per_candidate))))
    # Numba's workqueue fallback does not support concurrent Python callers.
    if numba.threading_layer() == "workqueue":
        workers = 1
    if workers <= 1:
        for index, function in enumerate(functions):
            with parallel.numba_thread_scope(resolve_threads(budget)):
                yield index, function(budget if callable(budget) else None)
        return

    pending = set(range(len(functions)))
    lock = Lock()
    stopped = Event()

    def allocation(index):
        with lock:
            if stopped.is_set():
                raise CancelledError("another founder candidate failed")
            cores = int(resolve_threads(budget))
            # Counting queued work prevents startup and task-replacement
            # over-allocation. During the tail there is no queued work.
            count = len(pending)
            rank = sum(other < index for other in pending)
            return max(1, cores // count + int(rank < cores % count))

    def execute(index):
        try:
            with parallel.numba_thread_scope(allocation(index)):
                return functions[index](lambda: allocation(index))
        finally:
            with lock:
                pending.remove(index)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, i): i for i in range(len(functions))}
        try:
            for future in as_completed(futures):
                yield futures[future], future.result()
        except BaseException:
            stopped.set()
            for future in futures:
                future.cancel()
            raise
