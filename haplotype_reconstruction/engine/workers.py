"""Keep shared arrays and resident state in bounded, fail-fast forkserver workers.

Owns transport, shared-array lifetimes and the two native recycling cadences.
Scheduling/admission belongs to scheduler; science never opens a pool.
BudgetedPool is retained for engine-owned C2 compilation. P0 ENG-21/22/24;
methods: WP_E1 sections 4.4/4.7. Constants carry execution/pools.py and
shared_arrays.py values from engine.parameters.csv; bytes and seconds as named.
"""

from __future__ import annotations

import atexit
import io
import logging
import math
import multiprocessing as mp
import multiprocessing.pool as mp_pool
import os
import pickle
import queue
import sys
import threading
import time
import traceback
import weakref
from concurrent.futures import CancelledError
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from multiprocessing.connection import wait
from multiprocessing.context import ForkServerContext, ForkServerProcess
from multiprocessing.reduction import ForkingPickler
from multiprocessing.shared_memory import SharedMemory
from typing import Any, cast

import numba
import numpy as np

from ..core.errors import InvariantError
from ..core.immutable import declared_immutable
from ..core.kernels import kernel
from . import runtime as bootstrap
from .graph import Task, TaskKey
from .scheduler import (
    _CONTEXT,
    AdmissionToken,
    CpuBudget,
    HelperBudget,
    PoolCounters,
    PoolShareBudget,
    _HelperSeat,
    _read,
    adopted,
    as_budget,
    budget_context,
    depth_below,
    enter_pool_seat,
    held_context,
    lease_exit,
    lease_user,
    malloc_trim,
    memory_governor,
    numba_threads,
    pool_seat,
    release_memory,
    task_peak_bytes,
    task_peak_start,
)

_LOG = logging.getLogger(__name__)
__all__ = [
    "SharedArrayRef",
    "new_segment",
    "create_shared_array",
    "empty_shared_array",
    "attach_shared_array",
    "close_shared_memory",
    "main_module_guard",
    "private_memory_measured",
    "record_recycle",
    "recycle_threshold",
    "BudgetedPool",
    "ResidentRef",
    "WorkerReply",
    "retain",
    "release_resident",
    "resident_value",
    "SharedValue",
    "share_value",
    "call_shared",
    "release_shared_value",
    "RecyclingPolicy",
    "WorkerGroup",
]

COPY_CHUNK_BYTES = 1 << 20  # ENG-24 shared_arrays.py: 1 MiB copy partitions.
COPY_THREADS = 16
_PARALLEL_COPY_BYTES = 4 << 20


@dataclass(frozen=True)
class SharedArrayRef:
    """What a worker needs to attach a shared array: segment name, shape and dtype (string)."""

    name: str
    shape: tuple[int, ...]
    dtype: str


def new_segment(size: int, *, name: str | None = None) -> SharedMemory:
    """A new segment of ``size`` bytes (at least one) named after the run (see the module docstring), or ``name``
    (FileExistsError if it exists)."""
    from .scheduler import mark_in_use, segment_name

    for _ in range(1 if name is not None else 64):
        try:
            handle = SharedMemory(name=name or segment_name(), create=True, size=max(int(size), 1))
        except FileExistsError:
            if name is not None:
                raise
            continue
        mark_in_use(getattr(handle, "_fd", -1))
        return handle
    raise FileExistsError("no free shared-memory segment name")


@kernel(overloads=((numba.types.uint8[::1], numba.types.uint8[::1]),), cache=True, parallel=True)
def _copy_shared_bytes(source, destination):
    chunk = COPY_CHUNK_BYTES
    for part in numba.prange((source.size + chunk - 1) // chunk):
        start = part * chunk
        stop = min(source.size, start + chunk)
        destination[start:stop] = source[start:stop]


def create_shared_array(array, *, copy_threads: int = 1) -> tuple[SharedMemory, SharedArrayRef]:
    """Copy ``array`` (C-contiguous) into a new segment; return (handle, reference).

    POSIX shared memory has no zero-byte segments, so an empty array gets a one-byte segment and keeps its shape
    and dtype. ``copy_threads`` parallelises the first-touch copy of large arrays, capped at COPY_THREADS and at
    the caller's CPU authority (in a task, its claim). If anything fails after allocation, the segment is closed
    and unlinked at once.
    """
    contiguous = _shareable(array)
    handle = new_segment(contiguous.nbytes)
    try:
        return (handle, _fill_shared(contiguous, handle, copy_threads))
    except BaseException:
        close_shared_memory([handle], unlink=True)
        raise


def _shareable(array) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        raise TypeError("object arrays cannot be transported by shared memory")
    return contiguous


def _fill_shared(contiguous: np.ndarray, handle: SharedMemory, copy_threads: int) -> SharedArrayRef:
    """Copy into an allocated segment (create_shared_array's second half); the caller owns failure cleanup."""
    threads = min(int(copy_threads), COPY_THREADS)
    if threads > 1 and contiguous.nbytes >= _PARALLEL_COPY_BYTES:
        from .scheduler import authority_threads, numba_threads

        with numba_threads(authority_threads(threads)):
            _copy_shared_bytes(
                np.frombuffer(memoryview(contiguous), dtype=np.uint8),
                np.frombuffer(handle.buf, dtype=np.uint8, count=contiguous.nbytes),
            )
    else:
        np.copyto(np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=handle.buf), contiguous)
    return SharedArrayRef(handle.name, tuple(contiguous.shape), contiguous.dtype.str)


def empty_shared_array(shape, dtype) -> tuple[SharedMemory, SharedArrayRef, np.ndarray]:
    """A new uninitialised segment for an array of ``shape`` and ``dtype``: (handle, reference, writable view).

    The caller fills it (e.g. in one parallel scan), passes ``reference`` to pools (``shared={"name": reference}``:
    attached, not copied), reserves its bytes itself (execution.reserve_memory), and at the end drops the view and
    closes and unlinks the segment (close_shared_memory([handle], unlink=True)).
    """
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise TypeError("object arrays cannot be transported by shared memory")
    shape = tuple(int(extent) for extent in shape)
    handle = new_segment(math.prod(shape) * dtype.itemsize)
    try:
        return (
            handle,
            SharedArrayRef(handle.name, shape, dtype.str),
            np.ndarray(shape, dtype=dtype, buffer=handle.buf),
        )
    except BaseException:
        close_shared_memory([handle], unlink=True)
        raise


def attach_shared_array(ref: SharedArrayRef) -> tuple[SharedMemory, np.ndarray]:
    """Attach to a segment created by create_shared_array; return (handle, array view)."""
    from .scheduler import mark_in_use

    dtype = np.dtype(ref.dtype)
    handle = SharedMemory(name=ref.name, create=False)
    mark_in_use(getattr(handle, "_fd", -1))
    try:
        buffer = handle.buf
        assert buffer is not None
        if math.prod(ref.shape) * dtype.itemsize > len(buffer):
            raise ValueError("shared-array reference describes more bytes than the segment")
        return (handle, np.ndarray(ref.shape, dtype=dtype, buffer=handle.buf))
    except BaseException:
        handle.close()
        raise


def close_shared_memory(handles, *, unlink: bool = False) -> None:
    """Close handles, optionally unlinking their segments (best effort)."""
    for handle in handles:
        try:
            handle.close()
        except Exception:
            pass
        if unlink:
            try:
                handle.unlink()
            except Exception:
                pass


LIVE_TOTAL_SECONDS = 0.25
TRIM_AFTER_BYTES = 64 << 20
RECYCLE_MIN_BYTES = 256 << 20
RECYCLE_MAX_BYTES = 1 << 30
RECYCLE_CHECK_SECONDS = 0.2
FORKSERVER = mp.get_context("forkserver")
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
            _GUARD_SAVED = (main_module, getattr(main_module, "__file__", None), getattr(main_module, "__spec__", None))
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


ARRIVAL_CHECK_SECONDS = 1.0
_STARTED: dict[int, list] = {}
_STARTED_LOCK = threading.Lock()


class _BatchEnd:
    """The consumer's CPU-wait cancellation: the batch's stop event is set, or a worker of the pool has died."""

    def __init__(self, stop, pool):
        self.stop, self.pool = (stop, pool)

    def is_set(self) -> bool:
        return self.stop.is_set() or (not self.pool._closed and bool(self.pool._lost_workers()))


def _exited(process):
    """The exit code the stdlib pool's worker handler recorded when it reaped ``process`` (None while it runs)."""
    return None if process._popen is None else process._popen.returncode


class _NonDaemonicProcess(ForkServerProcess):
    """A forkserver process allowed to create child processes; every start (also a pool's replacement of a
    recycled worker) is guarded against re-executing the entry script and recorded for its pool (_STARTED)."""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

    def start(self):
        args = getattr(self, "_args", ())
        initargs = args[3] if len(args) > 3 and isinstance(args[3], tuple) else ()
        with main_module_guard():
            super().start()
        from .scheduler import _start_ticks

        self._eng40_started = _start_ticks(self.pid)
        if initargs:
            with _STARTED_LOCK:
                started = _STARTED.get(id(initargs[0]))
                if started is not None:
                    started.append(self)


class _NonDaemonicContext(ForkServerContext):
    Process = _NonDaemonicProcess


_SHARED: dict = {}
_HANDLES: list = []
_INIT_ERROR: str | None = None
_FIRST_TASK = True
_INQUEUE: Any = None
_RECYCLE_BYTES = 0
_BASELINE: int | None = None
_CHECKED_AT = 0.0
_TASKS_RUN = 0
_CHUNKS_RUN = 0
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
    memory admission token its parent made for it (marked running now), or under an
    explicit presence token (so the governor sees running work). Returns task index, measured peak bytes, first, result.
    ``first`` marks the worker's first task, whose peak includes loading the process's compiled kernels (its
    statistics are kept apart from the other tasks')."""
    global _FIRST_TASK
    function, task, index, held, label, parent = packed
    if _INIT_ERROR is not None:
        raise RuntimeError(f"pool worker initialisation failed:\n{_INIT_ERROR}")
    bootstrap.task_boundary()
    seat = pool_seat()
    seat.begin_task()
    governor = memory_governor()
    presence = peak = None
    first, _FIRST_TASK = (_FIRST_TASK, False)
    if held[0]:
        governor.start(held[0])
    else:
        presence = governor.presence(label, held[1], parent=parent, outer=True)
        held = presence.held
    start = task_peak_start()
    try:
        with adopted(held):
            result = function(task, _SHARED, PoolShareBudget())
        peak = task_peak_bytes(start)
        if peak is not None and peak > TRIM_AFTER_BYTES:
            malloc_trim()
        return (index, peak, first, result)
    finally:
        if presence is not None:
            presence.release(None if first else peak)
            if first and peak is not None:
                governor.note(f"{label} (first task of a worker)", peak)
        import numba

        numba.set_num_threads(1)
        seat.end_task()


def _run_chunk(chunk):
    """Consecutive tasks of one dispatch, run in order in one worker; then the worker may retire."""
    global _TASKS_RUN, _CHUNKS_RUN
    rows = [_run_task(packed) for packed in chunk]
    _TASKS_RUN += len(chunk)
    _CHUNKS_RUN += 1
    seat = pool_seat()
    assert seat is not None
    counters = seat.counters
    if counters.retire_chunks and _CHUNKS_RUN >= counters.retire_chunks:
        _announce_pool_retirement()
    if _RECYCLE_BYTES and (not _RETIRING):
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


def private_memory_measured() -> bool:
    """Whether this system reports a process's private memory (Linux smaps_rollup), which recycling needs."""
    return _private_bytes() is not None


def _consider_retiring(label: str) -> None:
    """After a dispatch: the first sets the worker's baseline (start-up and kernels loaded, heap trimmed); later ones
    (at most every RECYCLE_CHECK_SECONDS) retire the worker when its private memory, after trimming the heap,
    exceeds the baseline by more than the pool's threshold."""
    global _BASELINE, _CHECKED_AT, _RETIRING
    now = time.monotonic()
    if _BASELINE is None:
        malloc_trim()
        _BASELINE, _CHECKED_AT = (_private_bytes(), now)
        if _BASELINE is None:
            _RETIRING = True
        return
    if now - _CHECKED_AT < RECYCLE_CHECK_SECONDS:
        return
    _CHECKED_AT = now
    private = _private_bytes()
    if private is None or private - _BASELINE <= _RECYCLE_BYTES:
        return
    malloc_trim()
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
    _announce_pool_retirement()
    assert _BASELINE is not None
    record_recycle(label, private, _BASELINE, _RECYCLE_BYTES, _TASKS_RUN)


def record_recycle(label: str, private: int, baseline: int, threshold: int, tasks: int) -> None:
    """A worker's retirement in the run's memory statistics (label "<label> worker recycled": count, private bytes
    at retirement); the first one of a label is also logged."""
    name = f"{label} worker recycled"
    try:
        governor = memory_governor()
        if not governor.statistics()["labels"].get(name, {}).get("tasks", 0):
            gib = 1 << 30
            _LOG.info(
                f"[memory] {label}: recycled after {tasks} tasks (private {private / gib:.2f} GiB, "
                f"{(private - baseline) / gib:.2f} GiB above baseline; threshold {threshold / gib:.2f} GiB); "
                "further recycles are counted in the memory summary"
            )
        governor.note(name, private)
    except Exception:
        pass


def recycle_threshold(processes: int, recycle_bytes: int | None = None) -> int:
    """The private-memory growth after which a worker is replaced: ``recycle_bytes`` if given (0: never), else a
    quarter of a worker's share of the run's memory limit within RECYCLE_MIN_BYTES..RECYCLE_MAX_BYTES."""
    if recycle_bytes is None:
        recycle_bytes = min(
            RECYCLE_MAX_BYTES, max(RECYCLE_MIN_BYTES, memory_governor().limit // (4 * max(1, int(processes))))
        )
    return max(0, int(recycle_bytes))


class BudgetedPool:
    """A forkserver pool whose workers share ``budget``'s CPUs dynamically (see the module docstring).

    ``processes`` workers (the caller bounds it by tasks and memory); ``shared`` arrays are copied into shared
    memory once (their bytes reserved for the pool's lifetime) and attached by every worker (no copy), and
    unlinked when the pool closes; a ``SharedArrayRef`` value (a segment the caller created, e.g. with
    `empty_shared_array`) is attached as it is, neither copied nor unlinked nor reserved by the pool. Use as a
    context manager: a normal exit closes and joins the pool, an exception terminates it. ``recycle_bytes``: a
    worker whose private memory grows by more than this since its first task is replaced after its task (None:
    a quarter of a worker's share of the run's memory limit, 256 MiB to 1 GiB; 0: never). Tasks share the
    total fairly and give CPUs above their share back at their boundaries to peers that are short or waiting
    (execution.budget._PoolSeat.share). ``max_processes`` bounds growth (None: the total and waiting tasks).
    ``yield_to_waiters`` is accepted for compatibility (yielding to waiting workers is always on).
    """

    def __init__(
        self,
        budget,
        *,
        processes: int,
        shared=None,
        max_tasks_per_child: int | None = None,
        label: str = "pool",
        recycle_bytes: int | None = None,
        max_processes: int | None = None,
    ):
        self.budget: CpuBudget = as_budget(budget)
        self.processes = max(1, int(processes))
        self.max_processes = None if max_processes is None else max(self.processes, int(max_processes))
        self.label = label
        total = max(1, int(self.budget.threads()))
        self._counters = PoolCounters(FORKSERVER, total, self.processes)
        self._counters.retirements = FORKSERVER.Array("q", 4096, lock=False)
        self._counters.retire_chunks = int(max_tasks_per_child or 0)
        with _STARTED_LOCK:
            self._workers = _STARTED[id(self._counters)] = []
        self._lease = None
        self._handles = []
        self._closed = False
        self._batch_in_progress = False
        self._stop = threading.Event()
        self._refresher = None
        self._batch_stop = None
        self._outstanding: set[AdmissionToken] = set()
        self._outstanding_lock = threading.Lock()
        self._pools, self._sizes, self._busy = ([], [], [])
        self._group_lock = threading.Lock()
        self._governor = memory_governor()
        self.recycle_bytes = recycle_threshold(self.processes, recycle_bytes)
        self._shared_hold = self._context_entry = None
        shared = dict(shared or {})
        shared_bytes = sum(
            int(np.asarray(array).nbytes) for array in shared.values() if not isinstance(array, SharedArrayRef)
        )
        if shared_bytes:
            self._shared_hold = self._governor.request(label=f"{label} shared")
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
            if self._shared_hold is not None:
                self._governor.rest(self._shared_hold.id)
                assert self._context_entry is not None
                self._context_entry[2] = False
            self._pool_arguments: dict[str, Any] = dict(
                initializer=_initialize,
                initargs=(self._counters, references, bootstrap.worker_settings(), self.recycle_bytes),
                maxtasksperchild=max_tasks_per_child,
            )
            with main_module_guard():
                self._pool = mp_pool.Pool(
                    processes=self.processes, context=_NonDaemonicContext(), **self._pool_arguments
                )
            self._pools.append(self._pool)
            self._sizes.append(self.processes)
            self._busy.append(0)
        except BaseException:
            close_shared_memory(self._handles, unlink=True)
            self._release_shared()
            with _STARTED_LOCK:
                _STARTED.pop(id(self._counters), None)
            raise
        if self.budget.live:
            self._lease = lease_user(self.budget)
            if self._lease is not None:
                self._lease.enter_user()
            self._refresher = threading.Thread(target=self._follow_budget, name=f"{label}-total", daemon=True)
            self._refresher.start()

    def _follow_budget(self):
        """Follow the live parent budget (a lane's lease grows as other lanes finish and returns CPUs above its
        share that it does not use): the pool total is re-read every LIVE_TOTAL_SECONDS."""
        while not self._stop.wait(LIVE_TOTAL_SECONDS):
            if self._lease is not None and hasattr(self.budget, "settle"):
                if self._counters.admission.acquire(timeout=LIVE_TOTAL_SECONDS):
                    try:
                        self._counters.set_total(self.budget.settle(_read(self._counters.claimed)))
                    finally:
                        self._counters.admission.release()
            else:
                self._counters.set_total(int(self.budget.threads()))

    def _grow(self, waiting: int | None) -> int:
        """Start more workers when the total exceeds the workers: up to the total, the ``waiting`` dispatches
        (None: unknown) and max_processes. Returns the number added (0 if none or if starting them failed)."""
        room = self.total - self.processes
        if waiting is not None:
            room = min(room, waiting)
        if self.max_processes is not None:
            room = min(room, self.max_processes - self.processes)
        if room <= 0 or self._closed:
            return 0
        with _STARTED_LOCK:
            registered = {id(process) for process in self._workers}
        try:
            with main_module_guard():
                pool = mp_pool.Pool(processes=room, context=_NonDaemonicContext(), **self._pool_arguments)
        except Exception as error:
            with self._group_lock:
                current = {id(process) for group in self._pools for process in list(cast(Any, group)._pool)}
            with _STARTED_LOCK:
                self._workers[:] = [
                    process for process in self._workers if id(process) in registered or id(process) in current
                ]
            _LOG.info(
                f"[pool] {self.label}: could not add {room} workers ({type(error).__name__}: {error}); "
                f"continuing with {self.processes}"
            )
            self.max_processes = self.processes
            return 0
        with self._group_lock:
            if self._closed:
                pool.terminate()
                return 0
            self._pools.append(pool)
            self._sizes.append(room)
            self._busy.append(0)
            self.processes += room
        return room

    @property
    def total(self) -> int:
        """The pool's current CPU total."""
        return int(self._counters.total.value)

    def imap_unordered(self, function, tasks, *, chunksize: int = 1):
        """Run ``function(task, shared, budget)`` for every task; yield results in completion order.

        Each call is one batch with its own start-up gate. ``function`` must be importable (module level).
        A lazy task iterable is drawn from only when a dispatch slot is free;
        measured pressure controls sending the task. No byte reservation exists.
        ``chunksize`` consecutive tasks are sent and run together.
        """
        return (result for _, result in self._batch(function, tasks, chunksize))

    def map(self, function, tasks, *, chunksize: int = 1) -> list:
        """Like imap_unordered, but return the results as a list in task order."""
        results = {}
        for index, result in self._batch(function, tasks, chunksize):
            results[index] = result
        return [results[index] for index in range(len(results))]

    def _batch(self, function, tasks, chunksize):
        """Start one batch; return a generator of (task index, result) in completion order."""
        if self._closed:
            raise RuntimeError(f"{self.label}: the pool is closed")
        if self._batch_in_progress:
            raise RuntimeError(f"{self.label}: the pool already has an unfinished batch")
        chunksize = max(1, int(chunksize))
        count = len(tasks) if hasattr(tasks, "__len__") else None
        governor, counters, label = (self._governor, self._counters, self.label)
        caller = held_context()
        depth, parent = (depth_below(caller), caller[0] if caller else 0)
        free_slots = threading.Semaphore(self.processes)
        stop = self._batch_stop = threading.Event()
        if self._stop.is_set():
            stop.set()
        arrivals = queue.SimpleQueue()
        outstanding, outstanding_lock = (self._outstanding, self._outstanding_lock)

        def settle(request, peak):
            with outstanding_lock:
                owned = request in outstanding
                outstanding.discard(request)
            if owned:
                request.release(peak)

        def free_slot(group):
            with self._group_lock:
                self._busy[group] -= 1
            free_slots.release()

        def arrived(request, rows, group):
            try:
                if request is not None:
                    peaks = [row[1] for row in rows if row[1] is not None and (not row[2])]
                    firsts = [row[1] for row in rows if row[1] is not None and row[2]]
                    settle(request, max(peaks, default=None))
                    if firsts:
                        governor.note(f"{label} (first task of a worker)", max(firsts))
            except BaseException as error:
                arrivals.put(("error", error))
            finally:
                free_slot(group)
                arrivals.put(("rows", rows))

        def failed(request, error, group):
            try:
                if request is not None:
                    settle(request, None)
            finally:
                free_slot(group)
                arrivals.put(("error", error))

        def reserve():
            """The chunk's admission token (including zero-byte work), or False on cancellation."""
            request = governor.request(label=label, depth=depth, queued=True, parent=parent, outer=True)
            if not request.poll():
                counters.open_gate()
                if not request.wait(cancel=stop):
                    return False
            return request

        def dispatch():
            iterator, position, exhausted = (iter(tasks), 0, False)
            try:
                while not stop.is_set():
                    if count is not None and position >= count:
                        exhausted = True
                        return
                    request = None
                    width = chunksize if count is None else min(chunksize, count - position)
                    chunk = list(islice(iterator, width))
                    if not chunk:
                        exhausted = True
                        return
                    if not free_slots.acquire(blocking=False):
                        waiting = None if count is None else -(-(count - position) // chunksize)
                        added = self._grow(waiting)
                        if added:
                            free_slots.release(added)
                        while not free_slots.acquire(timeout=0.5):
                            if stop.is_set():
                                return
                            added = self._grow(waiting)
                            if added:
                                free_slots.release(added)
                    request = reserve()
                    if request is False:
                        return
                    held = (0, depth)
                    if request:
                        with outstanding_lock:
                            outstanding.add(request)
                        held = (request.id, depth)
                    packed = [
                        (function, task, position + offset, held, label, parent) for offset, task in enumerate(chunk)
                    ]
                    with self._group_lock:
                        group = next(
                            (
                                index
                                for index, (size, busy) in enumerate(zip(self._sizes, self._busy, strict=False))
                                if busy < size
                            )
                        )
                        self._busy[group] += 1
                        pool = self._pools[group]
                    pool.apply_async(
                        _run_chunk,
                        (packed,),
                        callback=lambda rows, r=request, g=group: arrived(r, rows, g),
                        error_callback=lambda error, r=request, g=group: failed(r, error, g),
                    )
                    position += len(chunk)
            except BaseException as error:
                arrivals.put(("error", error))
                exhausted = None
            finally:
                if exhausted:
                    counters.open_gate()
                    arrivals.put(("end", position))
                elif exhausted is False:
                    arrivals.put(("error", RuntimeError(f"{label}: the batch stopped before every task was sent")))

        counters.start_batch(self.processes if count is None else count, self.processes)
        self._batch_in_progress = True
        dispatcher = threading.Thread(target=dispatch, name=f"{label}-dispatch", daemon=True)
        dispatcher.start()
        blocked = governor.suspended

        def consume():
            received, total, completed = (0, None, False)
            lease = (
                self._lease
                if self._lease is not None
                and hasattr(self._lease, "owner")
                and (self._lease.owner == threading.get_ident())
                else None
            )
            if lease is not None:
                lease.waiting += 1
            try:
                while total is None or received < total:
                    with blocked():
                        kind, payload = self._next_arrival(arrivals)
                    if kind == "error":
                        raise payload
                    if kind == "end":
                        total = payload
                        continue
                    for index, _, _, result in payload:
                        received += 1
                        with blocked():
                            held = counters.take_one(cancel=_BatchEnd(stop, self))
                        if not held:
                            self._raise_lost()
                            raise CancelledError(f"{label}: the pool closed while a result waited for a CPU")
                        if lease is not None:
                            lease.waiting -= 1
                        try:
                            yield (index, result)
                        finally:
                            if held and not counters.give_one(cancel=_BatchEnd(stop, self)):
                                self._raise_lost()
                                raise CancelledError(f"{label}: pool closed while returning a CPU")
                            if lease is not None:
                                lease.waiting += 1
                completed = True
            finally:
                stop.set()
                if not completed:
                    counters.closed.get_obj().value = 1
                dispatcher.join(timeout=1.0 if not completed else None)
                if lease is not None:
                    lease.waiting -= 1
                if completed:
                    self._batch_in_progress = False
                    self._close_added()
                else:
                    self._release_outstanding()

        return consume()

    def _lost_workers(self) -> list:
        """Every unannounced exit fails, including status zero (A40-7 R4)."""
        with _STARTED_LOCK:
            lost, retained = [], []
            for process in self._workers:
                status = _exited(process)
                if status is None:
                    retained.append(process)
                    continue
                slots = [i for i, pid in enumerate(self._counters.retirements) if pid == process.pid]
                if status == 0 and slots:
                    for slot in slots:
                        self._counters.retirements[slot] = 0
                else:
                    retained.append(process)
                    lost.append((process.pid, status))
            self._workers[:] = retained
            return lost

    def _raise_lost(self) -> None:
        """RuntimeError if a worker died during the batch; nothing once the pool is closing (closing or terminating
        it stops its workers deliberately; the waiters then end with CancelledError)."""
        if self._closed:
            return
        lost = self._lost_workers()
        if lost:
            raise RuntimeError(
                f"{self.label}: pool workers exited unexpectedly (pid, exit code) {lost} "
                "during the batch; their tasks were lost"
            )

    def _next_arrival(self, arrivals):
        """The batch's next arrival, after checking that no worker has died (also while results keep arriving);
        every ARRIVAL_CHECK_SECONDS without one: CancelledError if the pool was closed or terminated meanwhile,
        RuntimeError if a worker it started has exited abnormally."""
        while True:
            try:
                arrival = arrivals.get(timeout=ARRIVAL_CHECK_SECONDS)
            except queue.Empty:
                arrival = None
            if arrival is None and self._closed:
                raise CancelledError(f"{self.label}: the pool closed while results were awaited")
            self._raise_lost()
            if arrival is not None:
                return arrival

    @contextmanager
    def helper_seat(self, share: float = 0.25):
        """A seat in this pool's CPUs for a thread of its parent process on the tasks' critical path (discovery's
        evidence helper): ``share`` of the pool's total is set aside for it while it runs (the tasks share the rest
        and give back above it at their boundaries); yields its live budget (also bound for the thread's nested
        claims), so the helper's work and its tasks' together stay within the pool's total."""
        seat = _HelperSeat(self._counters, max(1, int(self.total * share)), self._raise_lost)
        seat.begin()
        budget = HelperBudget(seat)
        try:
            with budget_context(budget):
                yield budget
        finally:
            seat.end_task()

    def _close_added(self):
        """After a completed batch: close the worker groups added by growth (their workers are idle), so a pool
        does not keep its peak worker count; the next batch grows again if its total allows."""
        with self._group_lock:
            added = self._pools[1:]
            del self._pools[1:], self._sizes[1:], self._busy[1:]
            self.processes = self._sizes[0]
        for pool in added:
            pool.close()
        for pool in added:
            pool.join()

    def _release_outstanding(self):
        with self._outstanding_lock:
            requests = list(self._outstanding)
            self._outstanding.clear()
        for request in requests:
            request.release()

    def _abort_pools(self, pools) -> None:
        """Stop owned processes without stdlib finalizers acquiring dead-worker locks.

        multiprocessing.Pool._terminate_pool first acquires its queue read lock;
        SIGKILL can leave that lock held forever. Stop its daemon management
        threads, cancel that finalizer, and reap only our recorded PID/start pairs.
        """
        from .scheduler import _alive

        for pool in pools:
            pool._state = mp_pool.TERMINATE
            pool._terminate.cancel()
            for thread in (pool._worker_handler, pool._task_handler, pool._result_handler):
                thread._state = mp_pool.TERMINATE
            pool._change_notifier.put(None)
            pool._taskqueue.put(None)
        for pool in pools:
            pool._worker_handler.join(timeout=0.5)
        with _STARTED_LOCK:
            processes = tuple(self._workers)

        def alive(process):
            started = getattr(process, "_eng40_started", None)
            return started is not None and _alive(process.pid, started)

        for process in processes:
            if alive(process):
                process.terminate()
        deadline = time.monotonic() + 1.0
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in processes:
            if alive(process):
                process.kill()
            process.join(timeout=0.5)
        for pool in pools:
            pool._inqueue.close()
            pool._outqueue.close()
            for thread in (pool._worker_handler, pool._task_handler, pool._result_handler):
                if thread is not threading.current_thread():
                    thread.join(timeout=0.2)

    def _finish(self, terminate: bool):
        with self._group_lock:
            if self._closed:
                return
            self._closed = True
            pools = list(self._pools)
        self._stop.set()
        if self._batch_stop is not None:
            self._batch_stop.set()
        self._counters.closed.get_obj().value = 1
        try:
            if terminate:
                self._abort_pools(pools)
            else:
                self._counters.close()
                for pool in pools:
                    pool.close()
                for pool in pools:
                    pool.join()
        finally:
            with _STARTED_LOCK:
                _STARTED.pop(id(self._counters), None)
            if self._refresher is not None:
                self._refresher.join(timeout=1.0)
            close_shared_memory(self._handles, unlink=True)
            self._release_outstanding()
            self._release_shared()
            if self._lease is not None:
                lease_exit(self._lease)

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


@dataclass(frozen=True)
class ResidentRef:
    """A borrowed reference resolved only by the worker holding its affinity."""

    affinity: str
    key: str


@dataclass(frozen=True)
class WorkerReply:
    """One completed action plus its resource accounting and retirement notice."""

    key: TaskKey
    value: Any
    wall_seconds: float
    cpu_seconds: float
    peak_bytes: int | None
    resident_affinities: tuple[str, ...]
    retire: bool = False
    storage_routes: tuple[dict[str, Any], ...] = ()


_RESIDENT: dict[tuple[str, str], Any] = {}
_ACTIVE_AFFINITY: str | None = None


def retain(value: Any, key: str) -> ResidentRef:
    """Retain a large value without serializing it back to the scheduler.

    Called by an affinity action. Consecutive actions receive the same object;
    explicit release is required before its worker may recycle.
    """
    if _ACTIVE_AFFINITY is None:
        raise InvariantError("retained state requires task affinity")
    identity = (_ACTIVE_AFFINITY, key)
    if identity in _RESIDENT:
        raise InvariantError(f"resident key already exists: {identity}")
    _RESIDENT[identity] = value
    return ResidentRef(*identity)


def release_resident(reference: ResidentRef) -> None:
    """Release one retained value at its declared last consumer."""
    if reference.affinity != _ACTIVE_AFFINITY:
        raise InvariantError("resident state may only be released by its affinity")
    del _RESIDENT[reference.affinity, reference.key]


def resident_value(reference: ResidentRef) -> Any:
    """Borrow state within its lifetime; returning this value copies across workers."""
    try:
        return _RESIDENT[reference.affinity, reference.key]
    except KeyError as error:
        raise InvariantError(f"resident reference is absent or released: {reference}") from error


@dataclass(frozen=True)
class SharedValue:
    """A read-only value whose arrays one affinity action placed in shared memory once (ENG-24 transport).

    The independent subtasks of that action's Expansion receive this small reference instead of the
    arrays and attach them without copying (call_shared). The owner releases the segments at its
    declared last consumer, normally the Expansion's gather on the same affinity (release_shared_value).
    ``plan`` mirrors the value: dataclasses, C- or Fortran-ordered arrays, tuples of 1-D arrays of one
    dtype, and plain picklable leaves.
    """

    affinity: str
    key: str
    plan: Any
    arrays: tuple[tuple[str, SharedArrayRef], ...]


_SHARED_OWNED: dict[tuple[str, str], list[SharedMemory]] = {}
_SHARED_UNCLOSED: list[tuple[SharedMemory, Any]] = []
# A graph worker's connection to its coordinator (None elsewhere). Each owned segment's name is announced on it as
# soon as the segment is allocated (before it is filled), and its release afterwards, so the coordinator unlinks
# whatever remains when it aborts or closes the group, even if this worker is terminated or killed mid-copy.
_OWNER_CHANNEL: Any = None


def _announce_shared(kind: str, names: tuple[str, ...]) -> None:
    """Tell the coordinator that this worker now owns ("shared+") or has released ("shared-") ``names``."""
    if _OWNER_CHANNEL is not None and names:
        _OWNER_CHANNEL.send((kind, names))


def _share_plan(value: Any, name: str, arrays: dict[str, np.ndarray]) -> Any:
    from dataclasses import fields, is_dataclass

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{name}: object arrays cannot be shared")
        if value.flags.c_contiguous:
            arrays[name] = value
            return ("array", name, "C")
        if value.flags.f_contiguous:
            arrays[name] = value.T  # C-contiguous view; the attached array is transposed back (same strides)
            return ("array", name, "F")
        raise InvariantError(f"{name}: a shared array must be C- or Fortran-contiguous")
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value),
            tuple((f.name, _share_plan(getattr(value, f.name), f"{name}.{f.name}", arrays)) for f in fields(value)),
        )
    if (
        isinstance(value, tuple)
        and value
        and all(isinstance(part, np.ndarray) and part.ndim == 1 for part in value)
        and len({part.dtype for part in value}) == 1
    ):
        lengths = np.asarray([len(part) for part in value], dtype=np.int64)
        arrays[name + "/offsets"] = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        arrays[name + "/values"] = np.concatenate(value)
        return ("ragged", name)
    return ("value", value)


def _shared_build(plan: Any, attached: dict[str, np.ndarray]) -> Any:
    kind = plan[0]
    if kind == "array":
        array = attached[plan[1]]
        return array.T if plan[2] == "F" else array
    if kind == "dataclass":
        return plan[1](**{name: _shared_build(entry, attached) for name, entry in plan[2]})
    if kind == "ragged":
        offsets, values = attached[plan[1] + "/offsets"], attached[plan[1] + "/values"]
        return tuple(values[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1))
    return plan[1]


def share_value(value: Any, key: str, *, copy_threads: int = 1) -> SharedValue:
    """Copy ``value``'s arrays into shared memory once; called by an affinity action, which owns the segments.

    Exact transport: every array keeps its values, shape, dtype and memory order. Any failure after
    allocation closes and unlinks the segments created so far.
    """
    if _ACTIVE_AFFINITY is None:
        raise InvariantError("a shared value requires task affinity")
    identity = (_ACTIVE_AFFINITY, key)
    if identity in _SHARED_OWNED:
        raise InvariantError(f"shared value key already exists: {identity}")
    arrays: dict[str, np.ndarray] = {}
    plan = _share_plan(value, key, arrays)
    # Ownership is registered (and announced to a coordinator) at each allocation, before the copy into it:
    # a worker stopped mid-construction leaves no segment its coordinator does not know about.
    handles: list[SharedMemory] = []
    _SHARED_OWNED[identity] = handles
    references = []
    try:
        for name, array in arrays.items():
            contiguous = _shareable(array)
            handle = new_segment(contiguous.nbytes)
            handles.append(handle)
            _announce_shared("shared+", (handle.name,))
            references.append((name, _fill_shared(contiguous, handle, copy_threads)))
    except BaseException:
        _SHARED_OWNED.pop(identity, None)
        close_shared_memory(handles, unlink=True)
        _announce_shared("shared-", tuple(handle.name for handle in handles))
        raise
    return SharedValue(identity[0], identity[1], plan, tuple(references))


def _close_unreferenced(entries: list[tuple[SharedMemory, Any]]) -> list[tuple[SharedMemory, Any]]:
    """Unmap attachments whose attached array no longer exists; return the others, to be retried later.

    NumPy keeps no buffer export on a shared segment (closing one under a live array would leave that
    array pointing at unmapped memory), so liveness is read from a weak reference to the attached array:
    every view of it, however derived, keeps it alive.
    """
    kept = []
    for handle, alive in entries:
        if alive() is not None:
            kept.append((handle, alive))
        else:
            handle.close()
    return kept


def call_shared(reference: SharedValue, function: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``function(value, *args, **kwargs)`` on the attached read-only value of a SharedValue.

    The segments are attached without copying for this call only and unmapped afterwards once no
    array refers to them (otherwise at a later call or at process exit). The result must not refer to
    the shared arrays.
    """
    global _SHARED_UNCLOSED
    _SHARED_UNCLOSED = _close_unreferenced(_SHARED_UNCLOSED)
    entries: list[tuple[SharedMemory, Any]] = []
    try:
        attached = {}
        for name, array_reference in reference.arrays:
            handle, array = attach_shared_array(array_reference)
            array.flags.writeable = False
            entries.append((handle, weakref.ref(array)))
            attached[name] = array
            del array
        value = _shared_build(reference.plan, attached)
        del attached
        try:
            return function(value, *args, **kwargs)
        finally:
            del value
    finally:
        _SHARED_UNCLOSED.extend(_close_unreferenced(entries))


def release_shared_value(reference: SharedValue) -> None:
    """Close and unlink a SharedValue's segments; only its owning affinity may release it."""
    if reference.affinity != _ACTIVE_AFFINITY:
        raise InvariantError("a shared value may only be released by its affinity")
    handles = _SHARED_OWNED.pop((reference.affinity, reference.key))
    close_shared_memory(handles, unlink=True)
    _announce_shared("shared-", tuple(handle.name for handle in handles))


def _release_all_shared() -> None:
    """Unlink every segment this process still owns (process exit or an aborted in-process graph)."""
    while _SHARED_OWNED:
        _, handles = _SHARED_OWNED.popitem()
        close_shared_memory(handles, unlink=True)


atexit.register(_release_all_shared)


@dataclass
class RecyclingPolicy:
    """ENG-21/22: pool chunk and affinity task have distinct check cadences."""

    threshold: int
    baseline: int | None = None
    checked_at: float = 0.0

    def eligible(self, *, affinity: bool, retained: bool, now: float | None = None) -> bool:
        if retained or self.threshold <= 0:
            return False
        now = time.monotonic() if now is None else now
        if affinity:
            release_memory()
        elif self.baseline is not None and now - self.checked_at < RECYCLE_CHECK_SECONDS:
            return False
        elif self.baseline is None:
            malloc_trim()
        self.checked_at = now
        private = _private_bytes()
        if private is None:
            return False
        if self.baseline is None:
            self.baseline = private
            return False
        if private - self.baseline <= self.threshold:
            return False
        if not affinity:
            malloc_trim()
            private = _private_bytes()
            if private is None or private - self.baseline <= self.threshold:
                return False
        return True


def _perform(task: Task, token: tuple[int, int], budget: CpuBudget) -> WorkerReply:
    global _ACTIVE_AFFINITY
    from .graph import _readonly
    from .runtime import task_boundary

    task_boundary()
    governor = memory_governor()
    governor.start(token[0])
    _ACTIVE_AFFINITY = task.affinity
    start, cpu, peak = (time.perf_counter(), time.process_time(), task_peak_start())
    try:
        args, kwargs = (task.args, dict(task.kwargs))
        from .scheduler import TaskBudget, executing_task

        budget = TaskBudget(budget, task.resources.min_cores, task.resources.max_cores)
        team = budget.threads()
        if task.thread_argument is not None:
            kwargs[task.thread_argument] = team
        _readonly((args, kwargs))
        from .records import capture_storage_routes

        try:
            with (
                capture_storage_routes() as routes,
                executing_task(task.key),
                adopted(token),
                budget_context(budget),
                numba_threads(team),
            ):
                value = _readonly(task.function(*args, **kwargs))
        except BaseException as error:
            raise bootstrap.task_failure(error) from error
        return WorkerReply(
            task.key,
            value,
            time.perf_counter() - start,
            time.process_time() - cpu,
            task_peak_bytes(peak),
            tuple(sorted({key[0] for key in _RESIDENT})),
            storage_routes=tuple(routes),
        )
    finally:
        _ACTIVE_AFFINITY = None


def _graph_worker(connection: Any, settings: Any, groups: tuple[PoolCounters, ...], threshold: int) -> None:
    """Persistent forkserver child; failure is sent once, never respawned in a loop.

    Owned shared values are unlinked whenever this loop ends (multiprocessing children skip atexit); a worker
    that is terminated instead leaves them to its coordinator, which knows their names (_announce_shared).
    """
    global _OWNER_CHANNEL
    from dataclasses import replace

    _OWNER_CHANNEL = connection
    try:
        bootstrap.apply_worker_settings(settings)
        from .scheduler import _PoolSeat

        seats = [_PoolSeat(counters) for counters in groups]
        policy = RecyclingPolicy(threshold)
        connection.send(("ready", os.getpid()))
        while True:
            message = connection.recv()
            if message is None:
                return
            task, token, lane = message
            if isinstance(task, _PackedTask):
                task = task.load()
            from . import scheduler

            seat = scheduler._POOL_SEAT = seats[lane]
            seat.begin_task(task.resources.min_cores)
            try:
                reply = _perform(task, token, PoolShareBudget())
            finally:
                seat.end_task()
            retire = policy.eligible(affinity=task.affinity is not None, retained=bool(_RESIDENT or _SHARED_OWNED))
            connection.send(("result", replace(reply, retire=retire)))
            # Idle workers retain only explicitly registered affinity state.
            del reply, task, token, message
            if retire:
                return
    except BaseException as error:
        # A failed task fails its graph: release this worker's shared values before reporting.
        _release_all_shared()
        try:
            connection.send(("error", (error, traceback.format_exc())))
        except BaseException:
            pass
    finally:
        _release_all_shared()
        _OWNER_CHANNEL = None
        connection.close()


# Task transport, coordinator to graph worker. A task's large immutable arrays travel through shared memory: each
# distinct array is copied once into a segment the coordinator owns, and every task that carries it sends a small
# reference that the worker attaches without copying. Only arrays whose data provably can no longer change are shared
# (_fixed: read-only views of immutable bytes, as every array a worker sends back arrives, or of an owner its producer
# declared immutable at birth, core.immutable), with an exact dtype; every other array travels by value as before,
# and the transport never changes a caller's array. Tasks receive their arguments read-only (_perform freezes them),
# so a read-only shared view is exactly what each task saw before. Pickling such arrays per task cost the single
# coordinator thread O(tasks x bytes): one contig's 925 MB evidence rode on every selection block task.
TRANSPORT_SHARE_BYTES = 8 << 20


def _fixed(array: np.ndarray) -> bool:
    """Whether ``array`` may be shared by identity: its data provably can no longer change (it and every array it
    views are read-only, down to immutable ``bytes`` or an owner declared immutable at birth) and ``dtype.str``, which
    a SharedArrayRef carries, names its dtype exactly (no fields, subarrays or metadata). Read-only flags alone prove
    nothing: a view made before a freeze stays writable, and a read-only memoryview of a bytearray changes with it."""
    dtype = array.dtype
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None or dtype.metadata is not None:
        return False
    if np.dtype(dtype.str) != dtype:
        return False
    item: Any = array
    while isinstance(item, np.ndarray):
        if item.flags.writeable:
            return False
        if item.base is None:
            return declared_immutable(item)
        item = item.base
    if isinstance(item, memoryview):
        item = item.obj
    return type(item) is bytes


@dataclass(frozen=True)
class _PackedTask:
    """A task pickled by the coordinator with its large arrays replaced by shared-memory references."""

    payload: bytes

    def load(self) -> Task:
        return cast(Task, _TransportUnpickler(io.BytesIO(self.payload)).load())


class _TransportPickler(ForkingPickler):
    def __init__(self, file: Any, transport: _ArrayTransport):
        super().__init__(file, pickle.HIGHEST_PROTOCOL)
        self.transport = transport
        self.used: list[np.ndarray] = []

    def persistent_id(self, obj: Any) -> Any:
        if type(obj) is np.ndarray and obj.nbytes >= TRANSPORT_SHARE_BYTES:
            reference = self.transport.reference(obj) if _fixed(obj) else None
            if reference is not None:
                self.used.append(obj)
                return reference
            self.transport.by_value(obj)
        return None


class _TransportUnpickler(pickle.Unpickler):
    def __init__(self, file: Any):
        super().__init__(file)
        self.attached: dict[tuple, np.ndarray] = {}  # one array per reference: a repeated array stays one object

    def persistent_load(self, pid: Any) -> Any:
        global _SHARED_UNCLOSED
        kind, reference, transposed = pid
        if kind != "shared-array":
            raise pickle.UnpicklingError(f"unknown task transport reference {kind!r}")
        key = (reference, transposed)
        if key in self.attached:
            return self.attached[key]
        _SHARED_UNCLOSED = _close_unreferenced(_SHARED_UNCLOSED)
        handle, array = attach_shared_array(reference)
        array.flags.writeable = False
        _SHARED_UNCLOSED.append((handle, weakref.ref(array)))  # unmapped once no array refers to it
        result = array.T if transposed else array
        self.attached[key] = result
        return result


class _ArrayTransport:
    """The coordinator's shared copies of large immutable task arrays (TRANSPORT_SHARE_BYTES, _fixed).

    An array is shared once per descriptor (data address, shape, strides, dtype: a reshaped array is copied anew),
    and its segments live while it does; arrays of a task in flight are kept alive until its worker replies, so a
    worker always finds the segment it was sent. A segment is registered before it is filled, so close() unlinks it
    whatever interrupts the copy.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.entries: dict[tuple, tuple[weakref.ref, tuple, SharedMemory]] = {}
        self.filling: dict[str, SharedMemory | None] = {}  # by name, from before its creation until its entry
        self.in_flight: dict[int, list[np.ndarray]] = {}
        self.copied: dict[tuple, list[int]] = {}  # large arrays sent by value: (shape, dtype) -> [tasks, bytes]

    def pack(self, index: int, task: Task) -> _PackedTask:
        buffer = io.BytesIO()
        pickler = _TransportPickler(buffer, self)
        pickler.dump(task)
        with self.lock:
            self.in_flight[index] = pickler.used
        return _PackedTask(buffer.getvalue())

    def done(self, index: int) -> None:
        with self.lock:
            self.in_flight.pop(index, None)

    def by_value(self, array: np.ndarray) -> None:
        """Count a large array that travels by value (writable, viewing writable data, strided or with an inexact
        dtype): close() logs them, so producers that should publish immutable arrays can be found."""
        with self.lock:
            entry = self.copied.setdefault((array.shape, array.dtype.str), [0, 0])
            entry[0] += 1
            entry[1] += array.nbytes

    def reference(self, array: np.ndarray) -> tuple | None:
        if not (array.flags.c_contiguous or array.flags.f_contiguous):
            return None
        key = (id(array), array.__array_interface__["data"][0], array.shape, array.strides, array.dtype.str)
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None and entry[0]() is array:
                return entry[1]
            transposed = not array.flags.c_contiguous
            handle = self._new_segment(array.nbytes)
            try:
                shared = _fill_shared(array.T if transposed else array, handle, 1)
                pid = ("shared-array", shared, transposed)
                self.entries[key] = (weakref.ref(array, self._finalizer(key)), pid, handle)
            except BaseException:
                self.entries.pop(key, None)
                close_shared_memory([handle], unlink=True)
                raise
            finally:
                self.filling.pop(handle.name, None)
            return pid

    def _new_segment(self, size: int) -> SharedMemory:
        """A new segment, its name recorded before it exists: whatever interrupts its creation, close() unlinks it."""
        from .scheduler import segment_name

        for _ in range(64):
            name = segment_name()
            self.filling[name] = None
            try:
                handle = new_segment(size, name=name)
            except FileExistsError:
                del self.filling[name]
                continue
            self.filling[name] = handle
            return handle
        raise FileExistsError("no free shared-memory segment name")

    def _finalizer(self, key: tuple) -> Any:
        def dropped(_: Any) -> None:
            self._drop(key)

        return dropped

    def _drop(self, key: tuple) -> None:
        with self.lock:
            entry = self.entries.pop(key, None)
        if entry is not None:
            close_shared_memory([entry[2]], unlink=True)

    def close(self) -> None:
        with self.lock:
            handles = [entry[2] for entry in self.entries.values()]
            handles += [handle for handle in self.filling.values() if handle is not None]
            unopened = [name for name, handle in self.filling.items() if handle is None]
            copied = sorted(self.copied.items(), key=lambda item: -item[1][1])
            self.entries.clear()
            self.filling.clear()
            self.in_flight.clear()
            self.copied.clear()
        close_shared_memory(handles, unlink=True)
        for name in unopened:  # interrupted between its name and its handle: unlink it if it was created
            try:
                close_shared_memory([SharedMemory(name=name, create=False)], unlink=True)
            except FileNotFoundError:
                pass
        if copied:
            gib = 1 << 30
            largest = "; ".join(f"{shape} {dtype} x{tasks}" for (shape, dtype), (tasks, _) in copied[:3])
            _LOG.info(
                f"[transport] {sum(tasks for _, (tasks, _) in copied)} large task arrays "
                f"({sum(size for _, (_, size) in copied) / gib:.2f} GiB) travelled by value: {largest}"
            )


@dataclass
class _WorkerHandle:
    process: Any
    connection: Any
    started: int | None
    busy: TaskKey | None = None
    ready: bool = False
    announced: bool = False


class WorkerGroup:
    """A bounded graph's transport; scheduler owns all admission and dependencies.

    Every wait observes process liveness, including clean exits. Only an explicit
    completed-result retirement announcement permits replacement. Retained state
    pins affinity to a worker and is never transferred or recycled.
    """

    def __init__(
        self,
        cores: int,
        processes: int,
        *,
        lanes: int = 1,
        chains=(),
        recycle_bytes: int | None = None,
        static_affinities=(),
    ):
        self.context = mp.get_context("forkserver")
        from .scheduler import LaneAuthority

        self.authority = LaneAuthority(self.context, cores, lanes, processes, chains)
        self.settings = bootstrap.worker_settings()
        self.threshold = recycle_threshold(processes, recycle_bytes)
        self.handles: list[_WorkerHandle] = []
        # Each worker's announced shared-value segments, unlinked by close() whatever became of the worker.
        self.shared: dict[int, set[str]] = {}
        self.transport = _ArrayTransport()
        self.affinities: dict[str, int] = {}
        if static_affinities:
            from .scheduler import partition_chains

            partitions = partition_chains([weight for _, weight in static_affinities], min(lanes, processes))
            self.affinities.update(
                (static_affinities[task][0], worker) for worker, tasks in enumerate(partitions) for task in tasks
            )
        self.closed = False
        try:
            for index in range(processes):
                self._spawn(index)
        except BaseException:
            self.close(abort=True)
            raise

    def _spawn(self, index: int) -> None:
        from .scheduler import _start_ticks

        parent, child = self.context.Pipe()
        process = self.context.Process(
            target=_graph_worker, args=(child, self.settings, self.authority.groups, self.threshold)
        )
        process.daemon = False
        try:
            with main_module_guard():
                process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        assert process.pid is not None
        handle = _WorkerHandle(process, parent, _start_ticks(process.pid))
        if index == len(self.handles):
            self.handles.append(handle)
        else:
            self.handles[index] = handle

    def free_worker(self, affinity: str | None) -> int | None:
        """An idle worker for a task of ``affinity``: its pinned worker, else the first idle one. A new affinity is
        pinned to it only when a task is sent (pin), not when admission may still refuse the task."""
        if affinity in self.affinities:
            index = self.affinities[affinity]
            return index if self.handles[index].ready and self.handles[index].busy is None else None
        for index, handle in enumerate(self.handles):
            if handle.ready and handle.busy is None:
                return index
        return None

    def pin(self, affinity: str | None, index: int) -> None:
        """Pin ``affinity`` to worker ``index`` as its task is sent (its later tasks run there)."""
        if affinity is not None:
            self.affinities.setdefault(affinity, index)

    def submit(self, index: int, task: Task, token: tuple[int, int], *, lane: int = 0) -> None:
        handle = self.handles[index]
        if handle.busy is not None or not handle.ready:
            raise InvariantError("worker already has a task or has not initialized")
        handle.busy = task.key
        handle.connection.send((self.transport.pack(index, task), token, lane))

    def receive(self, timeout: float = 0.2) -> list[tuple[int, WorkerReply]]:
        ready = wait([h.connection for h in self.handles], timeout)
        replies = []
        for index, handle in enumerate(self.handles):
            if handle.connection in ready:
                try:
                    kind, value = handle.connection.recv()
                except (EOFError, OSError) as error:
                    handle.process.join(timeout=0)
                    raise BrokenProcessPool(
                        f"worker {handle.process.pid} exited unexpectedly with status {handle.process.exitcode}"
                    ) from error
                if kind == "ready":
                    handle.ready = True
                elif kind in ("shared+", "shared-"):
                    self._record_shared(index, kind, value)
                elif kind == "error":
                    task_error, trace = value
                    if hasattr(task_error, "add_note"):
                        task_error.add_note(trace)
                    raise bootstrap.task_failure(task_error)
                elif kind == "result":
                    if value.key != handle.busy:
                        raise InvariantError("worker replied for a different task")
                    handle.busy = None
                    self.transport.done(index)
                    replies.append((index, value))
                    if value.retire:
                        if value.resident_affinities:
                            raise InvariantError("worker attempted to recycle live resident state")
                        handle.announced = True
                        handle.process.join(timeout=1.0)
                        if handle.process.is_alive() or handle.process.exitcode != 0:
                            raise RuntimeError("announced worker retirement did not finish cleanly")
                        handle.connection.close()
                        self.shared.pop(index, None)
                        self._spawn(index)
                else:
                    raise InvariantError(f"unknown worker message: {kind}")
            current = self.handles[index]
            if current.process.exitcode is not None and (not current.announced):
                if current.connection.poll():
                    continue
                raise BrokenProcessPool(
                    f"worker {current.process.pid} exited unexpectedly with status {current.process.exitcode}"
                )
        return replies

    def close(self, *, abort: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        for group in self.authority.groups:
            group.closed.get_obj().value = 1
        from .scheduler import _alive

        if not abort:
            for handle in self.handles:
                try:
                    handle.connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
        for handle in self.handles:
            process = handle.process
            if abort and handle.started is not None and _alive(process.pid, handle.started):
                process.terminate()
        deadline = time.monotonic() + 1.0
        for handle in self.handles:
            handle.process.join(timeout=max(0.0, deadline - time.monotonic()))
        for handle in self.handles:
            if handle.started is not None and _alive(handle.process.pid, handle.started):
                handle.process.kill()
            handle.process.join(timeout=1.0)
        for index, handle in enumerate(self.handles):
            self._drain_shared(index, handle)
            handle.connection.close()
        self._unlink_shared()
        self.transport.close()

    def _drain_shared(self, index: int, handle: _WorkerHandle) -> None:
        """Read announcements still queued from a stopped worker (other messages no longer matter)."""
        try:
            while handle.connection.poll(0):
                kind, value = handle.connection.recv()
                if kind in ("shared+", "shared-"):
                    self._record_shared(index, kind, value)
        except (EOFError, OSError):
            pass
        except Exception:
            pass

    def _record_shared(self, index: int, kind: str, names: tuple[str, ...]) -> None:
        owned = self.shared.setdefault(index, set())
        if kind == "shared+":
            owned.update(names)
        else:
            owned.difference_update(names)

    def _unlink_shared(self) -> None:
        """Unlink every announced segment that still exists (the workers have stopped; best effort)."""
        names = sorted({name for announced in self.shared.values() for name in announced})
        self.shared.clear()
        for name in names:
            try:
                handle = SharedMemory(name=name, create=False)
            except FileNotFoundError:
                continue
            except OSError:
                continue
            close_shared_memory([handle], unlink=True)

    def __enter__(self) -> WorkerGroup:
        return self

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        self.close(abort=kind is not None)


def _announce_pool_retirement() -> None:
    """Publish the exact retiring PID before stdlib sends its final chunk result."""
    seat = pool_seat()
    assert seat is not None
    counters = seat.counters
    with counters.admission:
        for i, pid in enumerate(counters.retirements):
            if pid == os.getpid():
                return
            if not pid:
                counters.retirements[i] = os.getpid()
                return
        raise InvariantError("pool retirement announcement table is full")
