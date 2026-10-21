"""The one process-pool type: forkserver workers with shared arrays and dynamic thread shares.

Workers are fresh interpreters forked from the forkserver, which preloads NumPy, Numba, SciPy and the stage
modules (bootstrap.start_run), so they start in well under a second and share those pages copy-on-write.
Workers are non-daemonic: a task may itself open a pool or an executor.

`BudgetedPool` replaces the hand-written pool set-ups of the stages: shared-memory arrays copied once and
attached once per worker, the pool-share counters of execution.budget, the start-up gate per batch, a total
that follows a growing parent lease, and per-task accounting (active count and remainder claims) done around
every task. Tasks apply their share at their own boundaries with `claim_threads()` (or read `budget.threads()`).

Dispatch: a batch's tasks are sent to the workers one by one by a dispatcher thread, at most about one per
worker ahead of the results (so a task is handed out only when a worker is about to be free), in task order.
Memory (execution.memory): the shared arrays are reserved for the pool's lifetime, and with ``task_bytes`` the
dispatcher reserves each task's bytes before it sends the task (tasks start in task order as memory allows) and
releases them when the task's result arrives; workers measure each task's peak RSS for the run's statistics.
Pacing changes when tasks start, never their results.

Worker recycling: a worker whose private memory (USS) has grown by more than the pool's recycle threshold since
its first task (start-up and compiled kernels loaded) is retired after its task: it sends the result, leaves the
task queue and exits, and the pool starts a fresh worker in its place. Heap growth that malloc_trim cannot return
(fragmentation, caches, leaks) is then not carried from task to task. Tasks are independent, so results never
change; recycle events (count, private bytes at retirement) are in the run's memory statistics.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import islice
import multiprocessing as mp
import multiprocessing.pool as mp_pool
import queue
import sys
import threading
import time

import numpy as np

from . import bootstrap
from .budget import CpuBudget, PoolCounters, PoolShareBudget, as_budget, enter_pool_seat, pool_seat
from .memory import (adopted, depth_below, held_context, malloc_trim, memory_governor, task_peak_bytes,
                     task_peak_start, _CONTEXT)
from .shared_arrays import SharedArrayRef, attach_shared_array, close_shared_memory, create_shared_array

# Seconds between a pool's re-reads of a growing parent budget (a contig lane's lease).
LIVE_TOTAL_SECONDS = 1.0
# A worker returns freed heap pages to the system after a task whose peak exceeded this (so the memory it freed
# is visible to the governor's live check).
TRIM_AFTER_BYTES = 64 << 20
# Worker recycling (COORD 15 Oct 14:22; base died on AulStu with workers grown to 3.0-3.7 GB private after ~1 h
# for 0.35-0.6 GB task peaks): the threshold is a quarter of a worker's share of the run's memory limit, within
# these bounds; a worker reads its private memory (smaps_rollup, ~0.5 ms) at most every RECYCLE_CHECK_SECONDS.
RECYCLE_MIN_BYTES = 256 << 20
RECYCLE_MAX_BYTES = 1 << 30
RECYCLE_CHECK_SECONDS = 0.2

try:
    FORKSERVER = mp.get_context("forkserver")
except ValueError:      # platforms without forkserver
    FORKSERVER = mp.get_context("fork")


_GUARD_LOCK = threading.RLock()
_GUARD_DEPTH = 0
_GUARD_SAVED = None


@contextmanager
def main_module_guard():
    """Prevent new workers from re-executing the caller's entry script while processes start.

    A started worker imports the parent's __main__ by path unless __main__ has neither __file__ nor __spec__; a
    script without an ``if __name__ == "__main__"`` guard would then run again in every worker. Thread-safe and
    reentrant (pool handler threads start replacement workers concurrently); the values are restored when the
    last holder leaves.
    """
    global _GUARD_DEPTH, _GUARD_SAVED
    with _GUARD_LOCK:
        if _GUARD_DEPTH == 0:
            main_module = sys.modules.get("__main__")
            _GUARD_SAVED = (main_module, getattr(main_module, "__file__", None),
                            getattr(main_module, "__spec__", None))
            if main_module is not None:
                if hasattr(main_module, "__file__"):
                    del main_module.__file__
                main_module.__spec__ = None
        _GUARD_DEPTH += 1
    try:
        yield
    finally:
        with _GUARD_LOCK:
            _GUARD_DEPTH -= 1
            if _GUARD_DEPTH == 0:
                main_module, saved_file, saved_spec = _GUARD_SAVED
                if main_module is not None:
                    if saved_file is not None:
                        main_module.__file__ = saved_file
                    main_module.__spec__ = saved_spec
                _GUARD_SAVED = None


class _NonDaemonicProcess(FORKSERVER.Process):
    """A forkserver process allowed to create child processes; every start (also a pool's replacement of a
    recycled worker) is guarded against re-executing the entry script."""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

    def start(self):
        with main_module_guard():
            super().start()


class _NonDaemonicContext(type(FORKSERVER)):
    Process = _NonDaemonicProcess


# Worker-side state: the attached shared arrays and their handles, a failed initialisation, and whether the
# worker has run a task yet (its first task also loads the process's compiled kernels: +0.3-0.4 GB RSS).
_SHARED: dict = {}
_HANDLES: list = []
_INIT_ERROR: str | None = None
_FIRST_TASK = True
# Recycling: the Pool worker loop's task queue (closed to retire the worker after its task), the threshold, the
# private bytes after the first task, the time of the last check and the tasks run.
_INQUEUE = None
_RECYCLE_BYTES = 0
_BASELINE: int | None = None
_CHECKED_AT = 0.0
_TASKS_RUN = 0
_RETIRING = False


def _initialize(counters, references, settings, recycle_bytes=0):
    """Pool initializer: run settings, a seat in the pool's shares, attached shared arrays, one Numba thread.

    Starting the threading layer here (set_num_threads) matters: Numba 0.64 cannot link a cached kernel that
    calls a parallel kernel loaded from the cache before the layer runs. A failure is kept and reported by the
    worker's first task: multiprocessing would otherwise replace a worker whose initializer raises forever, and
    the batch would hang without an error.
    """
    global _INIT_ERROR, _INQUEUE, _RECYCLE_BYTES
    import atexit
    try:
        import numba
        # Called by multiprocessing.pool.worker: its task queue, which recycling closes after a task (the loop then
        # exits cleanly once the result is sent). Without it (another Python's pool) workers are not recycled.
        caller = sys._getframe(1)
        if caller.f_code.co_name == "worker" and hasattr(caller.f_locals.get("inqueue"), "_reader"):
            _INQUEUE = caller.f_locals["inqueue"]
            _RECYCLE_BYTES = max(0, int(recycle_bytes or 0))
        bootstrap.apply_worker_settings(settings)
        enter_pool_seat(counters)
        for name, reference in references.items():
            handle, array = attach_shared_array(reference)
            _HANDLES.append(handle)
            _SHARED[name] = array
        atexit.register(close_shared_memory, _HANDLES)
        numba.set_num_threads(1)
    except BaseException:
        import traceback
        _INIT_ERROR = traceback.format_exc()


def _run_task(packed):
    """One task in a worker: accounted as active (start-up gate, remainder claims) while it runs, under the
    memory reservation its parent made for it (marked running now), or under a zero-byte presence token when it
    has no estimate (so the governor sees running work). Returns (task index, measured peak bytes, first, result):
    ``first`` marks the worker's first task, whose peak includes loading the process's compiled kernels (its
    statistics are kept apart from the other tasks')."""
    global _FIRST_TASK
    function, task, index, held, label, parent = packed
    if _INIT_ERROR is not None:
        raise RuntimeError(f"pool worker initialisation failed:\n{_INIT_ERROR}")
    seat = pool_seat()
    seat.begin_task()
    governor = memory_governor()
    presence = peak = None
    first, _FIRST_TASK = _FIRST_TASK, False
    if held[0]:
        governor.start(held[0])
    else:
        presence = governor.presence(label, held[1], parent=parent)
        held = presence.held
    start = task_peak_start()
    try:
        with adopted(held):
            result = function(task, _SHARED, PoolShareBudget())
        peak = task_peak_bytes(start)
        if peak is not None and peak > TRIM_AFTER_BYTES:
            malloc_trim()
        return index, peak, first, result
    finally:
        if presence is not None:
            presence.release(None if first else peak)
            if first and peak is not None:
                governor.note(f"{label} (first task of a worker)", 0, peak)
        import numba
        numba.set_num_threads(1)
        seat.end_task()


def _run_chunk(chunk):
    """Consecutive tasks of one dispatch, run in order in one worker; then the worker may retire."""
    global _TASKS_RUN
    rows = [_run_task(packed) for packed in chunk]
    _TASKS_RUN += len(chunk)
    if _RECYCLE_BYTES and not _RETIRING:
        _consider_retiring(chunk[-1][4])
    return rows


def _private_bytes() -> int | None:
    """This process's private resident memory (USS: Private_Clean + Private_Dirty of /proc/self/smaps_rollup)."""
    try:
        total = 0
        with open("/proc/self/smaps_rollup", "rb") as handle:
            for line in handle:
                if line.startswith((b"Private_Clean:", b"Private_Dirty:")):
                    total += int(line.split()[1]) << 10
        return total
    except (OSError, ValueError, IndexError):
        return None


def _consider_retiring(label: str) -> None:
    """After a dispatch: the first sets the worker's baseline (start-up and kernels loaded, heap trimmed); later ones
    (at most every RECYCLE_CHECK_SECONDS) retire the worker when its private memory, after trimming the heap,
    exceeds the baseline by more than the pool's threshold."""
    global _BASELINE, _CHECKED_AT, _RETIRING
    now = time.monotonic()
    if _BASELINE is None:
        malloc_trim()
        _BASELINE, _CHECKED_AT = _private_bytes(), now
        if _BASELINE is None:
            _RETIRING = True                # no smaps_rollup: no recycling (and no further checks)
        return
    if now - _CHECKED_AT < RECYCLE_CHECK_SECONDS:
        return
    _CHECKED_AT = now
    private = _private_bytes()
    if private is None or private - _BASELINE <= _RECYCLE_BYTES:
        return
    malloc_trim()                           # only memory the worker cannot return counts
    private = _private_bytes()
    if private is None or private - _BASELINE <= _RECYCLE_BYTES:
        return
    _retire(label, private)


def _retire(label: str, private: int) -> None:
    """Leave the pool after this result: the worker loop sends it, its next read of the closed task queue fails and
    it exits; the pool starts a fresh worker in its place. Recorded in the run's memory statistics."""
    global _RETIRING
    try:
        _INQUEUE._reader.close()
    except Exception:
        return
    _RETIRING = True
    record_recycle(label, private, _BASELINE, _RECYCLE_BYTES, _TASKS_RUN)


def record_recycle(label: str, private: int, baseline: int, threshold: int, tasks: int) -> None:
    """A worker's retirement in the run's memory statistics (label "<label> worker recycled": count, private bytes
    at retirement); the first one of a label is also logged."""
    name = f"{label} worker recycled"
    try:
        governor = memory_governor()
        if not governor.statistics()["labels"].get(name, {}).get("tasks", 0):
            gib = 1 << 30
            print(f"[memory] {label}: a worker is recycled after {tasks} tasks (private memory "
                  f"{private / gib:.2f} GiB, {(private - baseline) / gib:.2f} GiB above its baseline after its first "
                  f"task; threshold {threshold / gib:.2f} GiB); further recycles are counted in the memory summary",
                  flush=True)
        governor.note(name, 0, private)
    except Exception:
        pass


def recycle_threshold(processes: int, recycle_bytes: int | None = None) -> int:
    """The private-memory growth after which a worker is replaced: ``recycle_bytes`` if given (0: never), else a
    quarter of a worker's share of the run's memory limit within RECYCLE_MIN_BYTES..RECYCLE_MAX_BYTES."""
    if recycle_bytes is None:
        recycle_bytes = min(RECYCLE_MAX_BYTES,
                            max(RECYCLE_MIN_BYTES, memory_governor().limit // (4 * max(1, int(processes)))))
    return max(0, int(recycle_bytes))


def task_sizes(task_bytes, tasks) -> list[int]:
    """Per-task byte estimates from None (none), an int (each task), a sequence (one per task) or a callable."""
    if task_bytes is None:
        return [0] * len(tasks)
    if callable(task_bytes):
        return [max(0, int(task_bytes(task))) for task in tasks]
    if isinstance(task_bytes, (int, np.integer)):
        return [max(0, int(task_bytes))] * len(tasks)
    sizes = [max(0, int(value)) for value in task_bytes]
    if len(sizes) != len(tasks):
        raise ValueError(f"task_bytes gives {len(sizes)} estimates for {len(tasks)} tasks")
    return sizes


class BudgetedPool:
    """A forkserver pool whose workers share ``budget``'s CPUs dynamically (see the module docstring).

    ``processes`` workers (the caller bounds it by tasks and memory); ``shared`` arrays are copied into shared
    memory once (their bytes reserved for the pool's lifetime) and attached by every worker (no copy), and
    unlinked when the pool closes; a ``SharedArrayRef`` value (a segment the caller created, e.g. with
    `empty_shared_array`) is attached as it is, neither copied nor unlinked nor reserved by the pool. Use as a
    context manager: a normal exit closes and joins the pool, an exception terminates it. ``recycle_bytes``: a
    worker whose private memory grows by more than this since its first task is replaced after its task (None:
    a quarter of a worker's share of the run's memory limit, 256 MiB to 1 GiB; 0: never). ``yield_to_waiters``:
    at numerical-team boundaries, yield borrowed CPUs to memory-admitted workers waiting for their first CPU.
    The default retains the existing policy; memory reservations and pure budget queries are unchanged.
    """

    def __init__(self, budget, *, processes: int, shared=None, max_tasks_per_child: int | None = None,
                 label: str = "pool", recycle_bytes: int | None = None,
                 yield_to_waiters: bool = False):
        self.budget: CpuBudget = as_budget(budget)
        self.processes = max(1, int(processes))
        self.label = label
        total = max(1, int(self.budget.threads()))
        self._counters = PoolCounters(FORKSERVER, total, self.processes,
                                      yield_to_waiters=yield_to_waiters)
        self._handles = []
        self._closed = False
        self._batch_in_progress = False
        self._stop = threading.Event()
        self._refresher = None
        self._batch_stop = None
        self._outstanding = set()           # reservations of dispatched tasks whose results have not arrived
        self._outstanding_lock = threading.Lock()
        self._governor = memory_governor()
        self.recycle_bytes = recycle_threshold(self.processes, recycle_bytes)
        self._shared_hold = self._context_entry = None
        shared = dict(shared or {})
        shared_bytes = sum(int(np.asarray(array).nbytes) for array in shared.values()
                           if not isinstance(array, SharedArrayRef))
        if shared_bytes:
            # The shared copies live as long as the pool: held as the creating thread's innermost reservation.
            self._shared_hold = self._governor.request(shared_bytes, label=f"{label} shared")
            self._shared_hold.wait()
            self._context_list = _CONTEXT.held
            self._context_entry = [self._shared_hold.id, self._shared_hold.depth, True]
            self._context_list.append(self._context_entry)
        references = {}
        try:
            for name, array in shared.items():
                if isinstance(array, SharedArrayRef):
                    references[name] = array
                    continue
                handle, reference = create_shared_array(array, copy_threads=total)
                self._handles.append(handle)
                references[name] = reference
            with main_module_guard():
                self._pool = mp_pool.Pool(
                    processes=self.processes, initializer=_initialize,
                    initargs=(self._counters, references, bootstrap.worker_settings(), self.recycle_bytes),
                    maxtasksperchild=max_tasks_per_child, context=_NonDaemonicContext())
        except BaseException:
            close_shared_memory(self._handles, unlink=True)
            self._release_shared()
            raise
        if self.budget.live:
            self._refresher = threading.Thread(target=self._follow_budget, name=f"{label}-total",
                                               daemon=True)
            self._refresher.start()

    def _follow_budget(self):
        """Raise the pool total as the live parent budget grows (never lowers it)."""
        while not self._stop.wait(LIVE_TOTAL_SECONDS):
            self._counters.raise_total(int(self.budget.threads()))

    @property
    def total(self) -> int:
        """The pool's current CPU total."""
        return int(self._counters.total.value)

    def imap_unordered(self, function, tasks, *, chunksize: int = 1, task_bytes=None, envelope: bool = False):
        """Run ``function(task, shared, budget)`` for every task; yield results in completion order.

        Each call is one batch with its own start-up gate. ``function`` must be importable (module level).
        ``task_bytes`` are the tasks' peak bytes (execution.memory): None, an int for every task, one int per task
        in task order, or a callable of the task; each task starts when its bytes fit the run's memory. ``tasks``
        may be a lazy iterable: it is drawn from only as tasks are sent, and with an int or a sequence of sizes
        only after the task's bytes are reserved (so a task's payload is built only once it can run).
        ``envelope``: a task's bytes also cover what its nested constructs reserve (they borrow them first).
        ``chunksize`` consecutive tasks are sent and run together.
        """
        return (result for _, result in self._batch(function, tasks, chunksize, task_bytes, envelope))

    def map(self, function, tasks, *, chunksize: int = 1, task_bytes=None, envelope: bool = False) -> list:
        """Like imap_unordered, but return the results as a list in task order."""
        results = {}
        for index, result in self._batch(function, tasks, chunksize, task_bytes, envelope):
            results[index] = result
        return [results[index] for index in range(len(results))]

    def _batch(self, function, tasks, chunksize, task_bytes, envelope=False):
        """Start one batch; return a generator of (task index, result) in completion order."""
        if self._closed:
            raise RuntimeError(f"{self.label}: the pool is closed")
        if self._batch_in_progress:
            raise RuntimeError(f"{self.label}: the pool already has an unfinished batch")
        chunksize = max(1, int(chunksize))
        count = len(tasks) if hasattr(tasks, "__len__") else None
        if task_bytes is None or isinstance(task_bytes, (int, np.integer)):
            each = max(0, int(task_bytes or 0))
            size_at = lambda position: each        # noqa: E731
            paced = each > 0
        elif callable(task_bytes):
            size_at, paced = None, True
        else:
            sizes = [max(0, int(value)) for value in task_bytes]
            if count is not None and len(sizes) != count:
                raise ValueError(f"task_bytes gives {len(sizes)} estimates for {count} tasks")
            count = len(sizes) if count is None else count
            size_at = lambda position: sizes[position] if position < len(sizes) else 0    # noqa: E731
            paced = any(sizes)
        governor, counters, label = self._governor, self._counters, self.label
        caller = held_context()
        depth, parent = depth_below(caller), caller[0] if caller else 0
        # Dispatched chunks without a result yet: one per worker; a batch without estimates also keeps a few more
        # queued (look-ahead). A memory-paced batch has none, so every reservation is a task that runs or is about
        # to (a granted task waiting behind busy workers would hold memory that no one uses).
        free_slots = threading.Semaphore(self.processes + (0 if paced else 1 + self.processes // 4))
        stop = self._batch_stop = threading.Event()
        if self._stop.is_set():
            stop.set()
        arrivals = queue.SimpleQueue()          # ("rows", rows) | ("error", exception) | ("end", tasks sent)
        outstanding, outstanding_lock = self._outstanding, self._outstanding_lock

        def settle(request, peak):
            with outstanding_lock:
                owned = request in outstanding
                outstanding.discard(request)
            if owned:
                request.release(peak)

        def arrived(request, rows):
            # In the pool's result thread: release the chunk's memory and its dispatch slot at once (a task
            # without an estimate recorded its measured peak through its presence token in the worker). A worker's
            # first task also loaded the process's kernels: its peak is recorded under a separate label. Nothing
            # may raise here (the pool's result thread would stop): a failure is handed to the consumer.
            try:
                if request is not None:
                    peaks = [row[1] for row in rows if row[1] is not None and not row[2]]
                    firsts = [row[1] for row in rows if row[1] is not None and row[2]]
                    settle(request, max(peaks, default=None))
                    if firsts:
                        governor.note(f"{label} (first task of a worker)", request.nbytes, max(firsts))
            except BaseException as error:
                arrivals.put(("error", error))
            finally:
                free_slots.release()
                arrivals.put(("rows", rows))

        def failed(request, error):
            try:
                if request is not None:
                    settle(request, None)
            finally:
                free_slots.release()
                arrivals.put(("error", error))

        def reserve(nbytes):
            """The chunk's reservation (None without bytes), or False if the batch stopped while waiting."""
            if not nbytes:
                return None
            request = governor.request(nbytes, label=label, depth=depth, queued=True, parent=parent,
                                       envelope=envelope)
            if not request.poll():
                # Memory paces this batch: no peer is about to start, so running tasks take the threads.
                counters.open_gate()
                if not request.wait(cancel=stop):
                    return False
            return request

        def dispatch():
            iterator, position, exhausted = iter(tasks), 0, False
            try:
                while not stop.is_set():
                    if count is not None and position >= count:     # every task sent
                        exhausted = True
                        return
                    while not free_slots.acquire(timeout=0.5):
                        if stop.is_set():
                            return
                    request = None
                    if size_at is not None:                  # sizes known before the tasks are drawn
                        width = chunksize if count is None else min(chunksize, count - position)
                        if width <= 0:
                            exhausted = True
                            free_slots.release()
                            return
                        request = reserve(sum(size_at(position + offset) for offset in range(width)))
                        if request is False:
                            return
                        chunk = list(islice(iterator, width))
                    else:
                        chunk = list(islice(iterator, chunksize))
                        if chunk:
                            request = reserve(sum(max(0, int(task_bytes(task))) for task in chunk))
                            if request is False:
                                return
                    if not chunk:
                        exhausted = True
                        if request:
                            request.release()
                        free_slots.release()
                        return
                    held = (0, depth)
                    if request:
                        with outstanding_lock:
                            outstanding.add(request)
                        held = (request.id, depth)
                    packed = [(function, task, position + offset, held, label, parent)
                              for offset, task in enumerate(chunk)]
                    self._pool.apply_async(_run_chunk, (packed,), callback=lambda rows, r=request: arrived(r, rows),
                                           error_callback=lambda error, r=request: failed(r, error))
                    position += len(chunk)
            except BaseException as error:          # e.g. a failing task iterable or a closed pool
                arrivals.put(("error", error))
                exhausted = None
            finally:
                if exhausted:
                    counters.open_gate()            # every task has started or is about to
                    arrivals.put(("end", position))
                elif exhausted is False:
                    arrivals.put(("error", RuntimeError(f"{label}: the batch stopped before every task was sent")))

        counters.start_batch(self.processes if count is None else count, self.processes)
        self._batch_in_progress = True
        dispatcher = threading.Thread(target=dispatch, name=f"{label}-dispatch", daemon=True)
        dispatcher.start()
        # While it waits for results the caller is blocked on its tasks (the progress rule): they hold their
        # reservations or presence tokens, so running work stays visible.
        blocked = governor.suspended

        def consume():
            received, total, completed = 0, None, False
            try:
                while total is None or received < total:
                    with blocked():
                        kind, payload = arrivals.get()
                    if kind == "error":
                        raise payload
                    if kind == "end":
                        total = payload
                        continue
                    for index, _, _, result in payload:
                        received += 1
                        yield index, result
                completed = True
            finally:
                # An abandoned or failed batch: no further dispatch; the reservations of tasks still in flight are
                # released (the pool's exit stops those tasks). No new batch may start on this pool then.
                stop.set()
                dispatcher.join()
                if completed:
                    self._batch_in_progress = False
                else:
                    self._release_outstanding()

        return consume()

    def _release_outstanding(self):
        with self._outstanding_lock:
            requests = list(self._outstanding)
            self._outstanding.clear()
        for request in requests:
            request.release()

    def _finish(self, terminate: bool):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._batch_stop is not None:
            self._batch_stop.set()
        try:
            if terminate:
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()
        finally:
            if self._refresher is not None:
                self._refresher.join()
            close_shared_memory(self._handles, unlink=True)
            self._release_outstanding()
            self._release_shared()

    def _release_shared(self):
        if self._context_entry is not None:
            self._context_list.remove(self._context_entry)
            self._context_entry = None
        if self._shared_hold is not None:
            self._shared_hold.release()
            self._shared_hold = None

    def close(self):
        """Wait for the workers to finish their tasks and exit; release the shared arrays."""
        self._finish(terminate=False)

    def terminate(self):
        """Stop the workers at once; release the shared arrays."""
        self._finish(terminate=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finish(terminate=exc_type is not None)
