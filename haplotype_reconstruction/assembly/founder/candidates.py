"""Independent founder candidates with memory-limited reusable thread slots.

Queued tasks reserve slots, not individual CPUs. Surviving workers acquire
freed threads at existing safe numerical boundaries. Results retain input IDs.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from threading import Lock, Event
import os
import numba
from ...core import parallel
from .workspace import resolve_threads


class SlotBudget:
    def __init__(self, count, workers, budget):
        self.remaining = count
        self.workers = workers
        self.budget = budget
        self.owners = {}
        self.lock = Lock()
        self.stopped = Event()

    def start(self, index):
        with self.lock:
            slot = next(i for i in range(self.workers) if i not in self.owners.values())
            self.owners[index] = slot

    def finish(self, index):
        with self.lock:
            del self.owners[index]
            self.remaining -= 1

    def allocation(self, index):
        with self.lock:
            if self.stopped.is_set():
                raise CancelledError('another founder candidate failed')
            count = min(self.workers, self.remaining)
            reserved = set(self.owners.values())
            # Reserve the next reusable slots during startup/replacement, so
            # a just-started worker never gives itself the whole allocation.
            for slot in range(self.workers):
                if len(reserved) == count:
                    break
                reserved.add(slot)
            rank = sum(slot < self.owners[index] for slot in reserved)
            cores = int(resolve_threads(self.budget))
            return max(1, cores // count + int(rank < cores % count))


def completed_candidates(functions, budget, bytes_per_candidate):
    total = int(resolve_threads(budget))
    workers = min(len(functions), total)
    from ...painting.model import available_process_memory_bytes
    available = available_process_memory_bytes()
    if available is not None:
        allowance = available * min(1., total / len(os.sched_getaffinity(0)))
        workers = min(workers,max(1,int(allowance // max(1,bytes_per_candidate))))
    if numba.threading_layer() == 'workqueue':
        workers = 1
    if workers <= 1:
        for index, function in enumerate(functions):
            with parallel.numba_thread_scope(resolve_threads(budget)):
                yield index, function(budget if callable(budget) else None)
        return
    slots = SlotBudget(len(functions),workers,budget)
    def execute(index):
        slots.start(index)
        try:
            with parallel.numba_thread_scope(slots.allocation(index)):
                return functions[index](lambda: slots.allocation(index))
        finally:
            slots.finish(index)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute,i):i for i in range(len(functions))}
        try:
            for future in as_completed(futures):
                yield futures[future],future.result()
        except BaseException:
            slots.stopped.set()
            for future in futures:
                future.cancel()
            raise
