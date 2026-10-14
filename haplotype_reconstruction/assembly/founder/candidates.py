"""Independent founder candidates with memory-limited reusable thread slots.

Queued tasks reserve slots, not individual CPUs. Surviving workers acquire
freed threads at existing safe numerical boundaries. Results retain input IDs.
The number of running candidates follows the live budget: a queued candidate
starts as soon as the budget (re-read every POLL_SECONDS and at every
completion) gives one more running candidate a thread and memory admits it, so
candidates queued under a small budget (a round or a fit that started while its
lane's other work held the threads) run concurrently once threads are freed.
Each candidate is a pure function of its inputs: results never depend on this
schedule.
"""
from concurrent.futures import ThreadPoolExecutor, CancelledError, wait, FIRST_COMPLETED
from threading import Lock, Event
import os
import numba
from ...core import parallel
from .workspace import resolve_threads

# How often a call with queued candidates re-reads a callable budget between completions.
POLL_SECONDS = 0.2


class SlotBudget:
    def __init__(self, workers, budget):
        self.workers = workers
        self.budget = budget
        self.owners = {}
        self.admitted = 0
        self.lock = Lock()
        self.stopped = Event()

    def admit(self):
        with self.lock:
            self.admitted += 1

    def start(self, index):
        with self.lock:
            slot = next(i for i in range(self.workers) if i not in self.owners.values())
            self.owners[index] = slot

    def finish(self, index):
        with self.lock:
            del self.owners[index]
            self.admitted -= 1

    def allocation(self, index):
        """The candidate's share of the live budget among the admitted (running or
        starting) candidates; the remainder goes to the lowest slots. Admitted candidates
        that have not started yet are counted, so a just-started candidate never gives
        itself their threads."""
        with self.lock:
            if self.stopped.is_set():
                raise CancelledError('another founder candidate failed')
            count = max(1, self.admitted)
            rank = sum(slot < self.owners[index] for slot in self.owners.values())
            cores = int(resolve_threads(self.budget))
            return max(1, cores // count + int(rank < cores % count))


def completed_candidates(functions, budget, bytes_per_candidate):
    """Yield (index, functions[index](share)) as the candidates complete.

    budget: threads, an int or a callable live budget. At most one running candidate per
    thread of the budget, and as many as the memory available at the call admits for the
    budget's share of this process's CPUs (bytes_per_candidate each); every running
    candidate gets a callable share (SlotBudget.allocation) that it re-reads at its safe
    numerical boundaries. Run one at a time, a candidate gets the whole budget.
    """
    from ...painting.model import available_process_memory_bytes
    available = available_process_memory_bytes()
    cpus = len(os.sched_getaffinity(0))

    def admitted(cores):
        workers = min(len(functions), max(1, cores))
        if available is not None:
            allowance = available * min(1., cores / cpus)
            workers = min(workers, max(1, int(allowance // max(1, bytes_per_candidate))))
        return workers

    if (numba.threading_layer() == 'workqueue' or len(functions) <= 1
            or (not callable(budget) and admitted(int(budget)) <= 1)):
        for index, function in enumerate(functions):
            with parallel.numba_thread_scope(resolve_threads(budget)):
                yield index, function(budget if callable(budget) else None)
        return
    slots = SlotBudget(len(functions), budget)

    def execute(index):
        slots.start(index)
        try:
            with parallel.numba_thread_scope(slots.allocation(index)):
                return functions[index](lambda: slots.allocation(index))
        finally:
            slots.finish(index)

    queued = list(range(len(functions)))
    running = {}
    with ThreadPoolExecutor(max_workers=len(functions)) as executor:
        try:
            while queued or running:
                limit = admitted(int(resolve_threads(budget)))
                # Admit every candidate that starts now before starting any, so none reads its
                # share while its peers are still being admitted (the shares then sum to the budget).
                starting = []
                while queued and len(running) + len(starting) < limit:
                    starting.append(queued.pop(0))
                    slots.admit()
                for index in starting:
                    running[executor.submit(execute, index)] = index
                done, _ = wait(running, return_when=FIRST_COMPLETED,
                               timeout=POLL_SECONDS if queued and callable(budget) else None)
                for future in done:
                    yield running.pop(future), future.result()
        except BaseException:
            slots.stopped.set()
            for future in running:
                future.cancel()
            raise
