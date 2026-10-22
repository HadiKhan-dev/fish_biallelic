"""Run memory: one governor that paces work so that planned peaks fit, and heap trimming at phase boundaries.

The memory governor (blueprint 11.14). No run may die of memory because too much work was admitted at once:
when memory gets tight, work waits (fewer concurrent tasks) instead of failing. There is one governor per run,
shared by every process of the run (the driver, contig-lane coordinators, pool and executor workers, and their
threads) through a small ledger in node-local shared memory (/dev/shm):

* Limit: the bytes the run may plan to use at once = the memory available when the governor starts
  (`resources.available_memory_bytes`: MemAvailable, every enclosing cgroup's limit minus its working set,
  SLURM_MEM_PER_NODE) minus a headroom of 10% (at least 4 GiB, at most half), and at most the configured
  ``[execution] memory_limit``.
* Ledger: the planned peak bytes of every running task (a *reservation*). A request is admitted when its bytes
  fit next to the reservations already held (ledger <= limit) and into the memory the node has right now (live
  available memory minus the headroom, minus the part of the held reservations not yet visible as used memory);
  otherwise it waits until enough is released (a waiter notices a release within WAKE_SECONDS).
* Flexible requests (``minimum``): "up to nbytes, at least minimum" (e.g. a batch size that adapts) are granted
  as much as fits when they are admitted, never less than the minimum; their claim in the queue is the minimum.
* Envelopes: estimates are additive by default (a nested request adds to its parent's). A reservation made with
  ``envelope=True`` covers the work nested inside it: nested requests first borrow its unused bytes (granted at
  once, no ledger traffic) and take only the excess from the ledger.
* Graded headroom: a request at depth d may fill the ledger only to limit x (1 - NESTED_HEADROOM_FRACTION /
  2^(d-1)), so the top of the limit stays for deeper (nested) requests: outer work that waits for its own sub-tasks
  cannot leave them without room (they would otherwise only progress by forced admissions). When the only holders
  are retained states (`rest`: no work that could make nested requests), the first waiting request may use that
  room at once, within the limit.
* Order: waiting requests are served deepest first (a request made inside running work, e.g. a pool task inside
  a contig lane, finishes what is started before new work starts), then first come, first served. A later
  request may overtake the first waiting one only if both fit, so small tasks never starve a large one.
* Progress: a request too large to fit even alone runs alone (when no reservation is held); if every holder of a
  reservation is itself waiting (for memory, or for its own sub-tasks; tasks granted but not started do not count
  as running) for PROGRESS_SECONDS, the first waiting
  request is admitted over the limit so that work continues, one such task at a time (when it ends and every
  holder still waits, the next one follows at once). Both are logged; nothing else is ever admitted over the
  limit, and the governor never deadlocks. Reservations of processes that exited without releasing them (a
  terminated pool) are released by the next waiter. The governor never signals a process.
* Killed runs: a new governor removes this user's ledgers and shared-memory segments (SEGMENT_PREFIX, named after
  the run's driver) that no process holds open (liveness marks: open-file-description locks, valid across PID
  namespaces) and whose named process is dead, before it reads the available memory.

Pacing changes when work runs, never what it computes: every construct returns results in task order, or in an
order its callers already accept, and thread counts never change results.

The parallel constructs reserve through the governor: BudgetedPool tasks (``task_bytes``) and shared arrays,
ChromosomeExecutor tasks (``task_bytes``; contig lanes reserve each contig's resident bytes), CpuBudget.concurrent
and thread_map. Stage code that allocates large memory outside them wraps it in `reserve_memory`. Statistics
(waits, wait seconds, peak reserved and peak live use, estimates against measured peak RSS per task label) go to
the log and the run record.
"""
from __future__ import annotations

from contextlib import contextmanager
import atexit
import ctypes
import fcntl
import gc
import mmap
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import uuid

import numpy as np

from .resources import available_memory_bytes

GIB = 1 << 30
HEADROOM_FRACTION = 0.10
HEADROOM_MIN_BYTES = 4 * GIB
# How long every holder must be waiting before the first waiting request is admitted over the limit.
PROGRESS_SECONDS = 2.0
# Graded headroom for nested work (15 Oct 36b stall): a request at depth d may fill the ledger only up to
# limit x (1 - NESTED_HEADROOM_FRACTION / 2^(d-1)) - 80% at depth 1, 90% at 2, 95% at 3, ... - so suspended outer
# work (lane residents, pool tasks waiting for their searches) never takes the room their nested work needs.
NESTED_HEADROOM_FRACTION = 0.20
# Longest pause between full re-checks of a waiting request (the first re-checks come sooner); between them a
# waiter looks every WAKE_SECONDS, without the lock, whether anything was released and then re-checks at once.
POLL_SECONDS = 0.5
WAKE_SECONDS = 0.02
# A process re-reads the live available memory at most this often outside waits.
LIVE_CACHE_SECONDS = 0.2
# Waiters look for reservations of exited processes at most this often.
PURGE_SECONDS = 2.0
# A process logs a waiting task label at most this often.
LOG_SECONDS = 30.0

TOKENS = 4096
WAITERS = 4096
LABELS = 256
_MAGIC = 0x48_41_50_4D_45_4D_33        # "HAPMEM3"

# Token states: its holder runs, waits for its own sub-tasks, waits for memory, or the task it was reserved for
# has not started yet (a pool task granted by the parent until its worker picks it up). Only RUNNING counts as
# work in progress for the progress rule.
RUNNING, SUSPENDED, WAITING, QUEUED, RESIDENT = 1, 2, 3, 4, 5      # RESIDENT: retained state only (rest)

# Header: one int64 per field, by index (fast scalar access from every process).
_FIELDS = ("magic", "limit", "configured", "headroom", "live_headroom", "start_available", "next_id", "reserved",
           "waiting", "peak_reserved", "peak_used", "largest", "requests", "waits", "wait_ns", "alone", "forced",
           "purged", "blocked_since", "created_ns", "released", "purge_ns")
(_MAGIC_AT, _LIMIT, _CONFIGURED, _HEADROOM, _LIVE_HEADROOM, _START, _NEXT_ID, _RESERVED, _WAITING, _PEAK_RESERVED,
 _PEAK_USED, _LARGEST, _REQUESTS, _WAITS, _WAIT_NS, _ALONE, _FORCED, _PURGED, _BLOCKED_SINCE, _CREATED,
 _RELEASED, _PURGE_NS) = range(len(_FIELDS))
_HEADER_SLOTS = 32
# A token's ledger bytes plus what it borrowed from an envelope ancestor (lender) are what it holds; an envelope
# lends its unused bytes (bytes + borrowed - lent) to the requests nested under it (parent links).
_TOKEN = np.dtype([("id", "<i8"), ("pid", "<i8"), ("started", "<i8"), ("bytes", "<i8"), ("granted", "<i8"),
                   ("depth", "<i8"), ("state", "<i8"), ("label", "<i8"), ("forced", "<i8"), ("parent", "<i8"),
                   ("envelope", "<i8"), ("lent", "<i8"), ("borrowed", "<i8"), ("lender", "<i8")])
_WAITER = np.dtype([("id", "<i8"), ("pid", "<i8"), ("started", "<i8"), ("bytes", "<i8"), ("since", "<i8"),
                    ("depth", "<i8"), ("label", "<i8")])
_LABEL = np.dtype([("name", "S56"), ("tasks", "<i8"), ("estimate_sum", "<f8"), ("estimate_max", "<i8"),
                   ("measured", "<i8"), ("measured_sum", "<f8"), ("measured_max", "<i8"), ("under", "<i8"),
                   ("ratio_max", "<f8"), ("waits", "<i8"), ("wait_ns", "<i8"), ("alone", "<i8"),
                   ("forced", "<i8")])
_TOKENS_AT = 8 * _HEADER_SLOTS
_WAITERS_AT = _TOKENS_AT + TOKENS * _TOKEN.itemsize
_LABELS_AT = _WAITERS_AT + WAITERS * _WAITER.itemsize
_SIZE = _LABELS_AT + LABELS * _LABEL.itemsize
_PREFIX = "haplotypes-memory-"
# Shared-memory segments of a run (execution.shared_arrays) are named f"{SEGMENT_PREFIX}{uid}-{pid}-{started}-
# {random}" after the process that created the run's ledger (the driver), so that the next governor created on
# the node removes the segments of a run that was killed before it unlinked them (they would hold RAM).
SEGMENT_PREFIX = "haplotypes-shm-"
_SEGMENT_DIRECTORY = Path("/dev/shm")
_PROGRESS_NS = int(PROGRESS_SECONDS * 1e9)


# ---------------------------------------------------------------------------------------------------------------
# Heap trimming
# ---------------------------------------------------------------------------------------------------------------

try:
    _LIBC = ctypes.CDLL("libc.so.6")
    # flock called WITHOUT releasing the GIL (PyDLL): an uncontended ledger lock then costs one quick system call
    # instead of a GIL hand-over to the process's other busy threads (a pool's task and result handlers).
    _FLOCK = ctypes.PyDLL("libc.so.6", use_errno=True).flock
    _FLOCK.argtypes = (ctypes.c_int, ctypes.c_int)
    _FLOCK.restype = ctypes.c_int
except (OSError, AttributeError):
    _LIBC = _FLOCK = None


def malloc_trim() -> None:
    """Ask glibc to return free heap pages to the operating system (no-op without glibc).

    glibc keeps freed heap pages in its arenas; after a phase that freed large temporaries, trimming hands them
    back so the next phase's peak (and the node's view of this process) does not include them.
    """
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def release_memory() -> None:
    """Collect unreachable Python cycles, then trim the heap (executors call it after every task)."""
    gc.collect()
    malloc_trim()


# ---------------------------------------------------------------------------------------------------------------
# Measured peak of one task in its own process
# ---------------------------------------------------------------------------------------------------------------

def _status_bytes(field: str) -> int | None:
    try:
        with open("/proc/self/status", "rt", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def task_peak_start() -> int | None:
    """Start measuring a task's peak: reset this process's peak RSS to its current RSS and return that RSS
    (None where Linux's clear_refs is unavailable). Meaningful only while the process runs one task at a time."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5")
    except OSError:
        return None
    return _status_bytes("VmRSS:")


def task_peak_bytes(start: int | None) -> int | None:
    """Peak RSS above the task's starting RSS since `task_peak_start` (shared pages it touched included)."""
    if start is None:
        return None
    peak = _status_bytes("VmHWM:")
    return None if peak is None else max(0, peak - start)


# ---------------------------------------------------------------------------------------------------------------
# Process identity (a pid plus its start time, so that a reused pid is not mistaken for a live holder)
# ---------------------------------------------------------------------------------------------------------------

def _start_ticks(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return int(text.rpartition(")")[2].split()[19])
    except (OSError, ValueError, IndexError):
        return None


_IDENTITY: tuple[int, int] | None = None


def _identity() -> tuple[int, int]:
    global _IDENTITY
    pid = os.getpid()
    if _IDENTITY is None or _IDENTITY[0] != pid:
        _IDENTITY = (pid, _start_ticks(pid) or 0)
    return _IDENTITY


def _alive(pid: int, started: int) -> bool:
    ticks = _start_ticks(pid)
    return ticks is not None and (started == 0 or ticks == started)


# ---------------------------------------------------------------------------------------------------------------
# What the calling thread holds (the depth of its requests; marking holders that wait)
# ---------------------------------------------------------------------------------------------------------------

class _Context(threading.local):
    def __init__(self):
        self.held = []          # [token id (0: none), depth, executing]; the last entry is the innermost
        self.suspended = 0      # nesting of suspended() in this thread


_CONTEXT = _Context()


def held_context() -> tuple[int, int] | None:
    """The calling thread's innermost reservation as (token id, depth), or None: what its sub-work inherits."""
    held = _CONTEXT.held
    return (held[-1][0], held[-1][1]) if held else None


def depth_below(held: tuple[int, int] | None) -> int:
    """The depth of requests made under ``held`` (1 at the top level)."""
    return 1 if held is None else int(held[1]) + 1


def _executing_ids() -> list[int]:
    return [entry[0] for entry in _CONTEXT.held if entry[2] and entry[0]]


@contextmanager
def adopted(held: tuple[int, int] | None, *, executing: bool = True):
    """Run the block under a reservation made elsewhere: a pool task under the token its parent reserved for
    it (``executing``: this thread does that work, so it marks the token while it waits), or a helper thread
    under its parent's innermost reservation (``executing=False``: depth only)."""
    if held is None:
        yield
        return
    entry = [int(held[0]), int(held[1]), bool(executing)]
    _CONTEXT.held.append(entry)
    try:
        yield
    finally:
        _CONTEXT.held.remove(entry)


# ---------------------------------------------------------------------------------------------------------------
# The shared ledger
# ---------------------------------------------------------------------------------------------------------------

def _directory() -> Path:
    shm = Path("/dev/shm")
    return shm if shm.is_dir() and os.access(shm, os.W_OK) else Path(tempfile.gettempdir())


# Liveness marks, valid across PID namespaces: every process that opens a ledger or a run's shared segment holds an
# open-file-description read lock on its first byte while it keeps the file open (independent of the ledger's
# flock; the kernel drops it when the process ends). A file is stale only when no process marks it AND the process
# its name records is dead (files of earlier versions carry no marks).
_LINUX = sys.platform.startswith("linux")
_F_OFD_GETLK = getattr(fcntl, "F_OFD_GETLK", 36 if _LINUX else None)
_F_OFD_SETLK = getattr(fcntl, "F_OFD_SETLK", 37 if _LINUX else None)
_FLOCK_STRUCT = "hhqqi4x"          # struct flock (LP64): l_type, l_whence, l_start, l_len, l_pid


def mark_in_use(fd: int) -> None:
    """Mark the file open on ``fd`` as in use by this process for as long as ``fd`` stays open."""
    if _F_OFD_SETLK is not None and fd >= 0:
        try:
            fcntl.fcntl(fd, _F_OFD_SETLK, struct.pack(_FLOCK_STRUCT, fcntl.F_RDLCK, os.SEEK_SET, 0, 1, 0))
        except OSError:
            pass


def _in_use(path: str) -> bool:
    """Whether some process marks the file as in use (True when that cannot be told)."""
    if _F_OFD_GETLK is None:
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return True
    try:
        probe = fcntl.fcntl(fd, _F_OFD_GETLK, struct.pack(_FLOCK_STRUCT, fcntl.F_WRLCK, os.SEEK_SET, 0, 1, 0))
        return struct.unpack(_FLOCK_STRUCT, probe)[0] != fcntl.F_UNLCK
    except OSError:
        return True
    finally:
        os.close(fd)


def _named_process(name: str, prefix: str) -> tuple[int, int] | None:
    """(pid, start time) named by a ledger or segment name ``prefix + "<pid>-<started>-<random>"``."""
    try:
        pid, started = (int(part) for part in name[len(prefix):].split("-")[:2])
    except ValueError:
        return None
    return pid, started


def _remove_stale(directory: Path, prefix: str) -> tuple[int, int]:
    """Delete this user's files ``<prefix><uid>-<pid>-<started>-...`` in ``directory`` that no process marks as in use
    and whose named process no longer exists (the ledger or shared segments of a run killed before it removed them):
    (files, bytes freed)."""
    prefix = f"{prefix}{os.getuid()}-"
    removed = freed = 0
    try:
        entries = [entry for entry in os.scandir(directory) if entry.name.startswith(prefix)]
    except OSError:
        return 0, 0
    for entry in entries:
        owner = _named_process(entry.name, prefix)
        if owner is None or _alive(*owner) or _in_use(entry.path):
            continue
        try:
            size = entry.stat(follow_symlinks=False).st_blocks * 512
            os.unlink(entry.path)
        except OSError:
            continue
        removed, freed = removed + 1, freed + size
    return removed, freed


# (segments, bytes) of dead runs' shared segments this process removed when it created its governor.
_STALE_SEGMENTS = (0, 0)


class _Ledger:
    """The mapped ledger file: header, reservations (tokens), queued requests (waiters), per-label statistics.
    Fields are exposed as views (``t_*`` tokens, ``w_*`` waiters, ``l_*`` labels) so access stays cheap."""

    def __init__(self, path: Path, *, create: bool):
        self.path = Path(path)
        self.pid = os.getpid()
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        self.fd = os.open(self.path, flags, 0o600)
        mark_in_use(self.fd)
        try:
            if create:
                os.ftruncate(self.fd, _SIZE)
            elif os.fstat(self.fd).st_size != _SIZE:
                raise RuntimeError(f"{self.path} is not a memory ledger of this package version")
            self.map = mmap.mmap(self.fd, _SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException:
            os.close(self.fd)
            raise
        self.h = np.ndarray((_HEADER_SLOTS,), dtype="<i8", buffer=self.map, offset=0)
        tokens = np.ndarray((TOKENS,), dtype=_TOKEN, buffer=self.map, offset=_TOKENS_AT)
        waiters = np.ndarray((WAITERS,), dtype=_WAITER, buffer=self.map, offset=_WAITERS_AT)
        self.labels = np.ndarray((LABELS,), dtype=_LABEL, buffer=self.map, offset=_LABELS_AT)
        (self.t_id, self.t_pid, self.t_started, self.t_bytes, self.t_granted, self.t_depth, self.t_state,
         self.t_label, self.t_forced, self.t_parent, self.t_envelope, self.t_lent, self.t_borrowed,
         self.t_lender) = (tokens[name] for name in _TOKEN.names)
        (self.w_id, self.w_pid, self.w_started, self.w_bytes, self.w_since, self.w_depth,
         self.w_label) = (waiters[name] for name in _WAITER.names)
        self.l = {name: self.labels[name] for name in _LABEL.names}
        self.thread_lock = threading.Lock()

    @contextmanager
    def locked(self):
        """Exclusive access across processes (flock, released by the kernel if a holder dies) and threads.

        The lock is first tried without blocking and without releasing the GIL; only a contended lock waits
        (releasing the GIL)."""
        with self.thread_lock:
            if self.pid != os.getpid():          # a forked copy needs its own descriptor (flock is per descriptor)
                self.fd = os.open(self.path, os.O_RDWR)
                self.pid = os.getpid()
            if _FLOCK is None or _FLOCK(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB) != 0:
                fcntl.flock(self.fd, fcntl.LOCK_EX)
            try:
                yield self.h
            finally:
                if _FLOCK is None or _FLOCK(self.fd, fcntl.LOCK_UN) != 0:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------------------------------------------
# Reservations
# ---------------------------------------------------------------------------------------------------------------

class Reservation:
    """A request for ``nbytes`` of the run's memory: queued until granted (`poll` / `wait`), then held until
    `release`. Requests of zero bytes are granted at once and take no ledger entry."""

    __slots__ = ("governor", "nbytes", "minimum", "label", "depth", "id", "granted", "waited", "closed",
                 "queued", "presence", "parent", "envelope", "_waiter", "_slot", "_label", "_logged")

    def __init__(self, governor: "MemoryGovernor", nbytes: int, label: str, depth: int, minimum: int | None = None,
                 *, queued: bool = False, presence: bool = False, parent: int = 0, envelope: bool = False):
        self.governor = governor
        self.nbytes = max(0, int(nbytes))
        # parent: the token this request is nested under (0: none); envelope: it covers its nested requests.
        self.parent = int(parent or 0)
        self.envelope = bool(envelope)
        # queued: granted for a task that has not started (state QUEUED until its executor marks it running);
        # presence: a zero-byte token that only marks running work (tasks without an estimate).
        self.queued = bool(queued)
        self.presence = bool(presence) and self.nbytes == 0
        # A flexible request may be granted fewer bytes (``nbytes`` then holds the granted amount), never fewer
        # than ``minimum``.
        self.minimum = self.nbytes if minimum is None else min(self.nbytes, max(0, int(minimum)))
        self.label = label
        self.depth = int(depth)
        self.id = 0
        self.granted = self.nbytes == 0 and not self.presence
        self.waited = False
        self.closed = False
        self._waiter = -1
        self._slot = -1
        self._label = -1
        self._logged = False

    @property
    def held(self) -> tuple[int, int]:
        """(token id, depth): what work done under this reservation inherits (`adopted`)."""
        return self.id, self.depth

    def poll(self, *, fresh: bool = False) -> bool:
        """Try to be admitted now (never blocks); True once granted. The first failed call queues the request."""
        if self.granted:
            return True
        if self.closed:
            raise RuntimeError("a released reservation cannot be granted")
        return self.governor._admit(self, fresh)

    def wait(self, cancel: threading.Event | None = None) -> bool:
        """Block until granted (True), or until ``cancel`` is set (False: the request is withdrawn)."""
        if self.poll():
            return True
        governor = self.governor
        waiting_ids = _executing_ids()
        governor._set_states(waiting_ids, WAITING)
        delay = 0.005
        released = governor._released()
        due = time.monotonic() + delay
        try:
            while True:
                pause = min(WAKE_SECONDS, max(0.0, due - time.monotonic()))
                if cancel is not None:
                    if cancel.wait(pause):
                        self.release()
                        return False
                else:
                    time.sleep(pause)
                current = governor._released()
                if current == released and time.monotonic() < due:
                    continue
                released = current
                if self.poll(fresh=True):
                    return True
                delay = min(2.0 * delay, POLL_SECONDS)
                due = time.monotonic() + delay
        except BaseException:
            self.release()
            raise
        finally:
            governor._set_states(waiting_ids, RUNNING)

    def release(self, measured: int | None = None, *, estimate: int | None = None, record: bool = True) -> None:
        """Release the bytes (or withdraw the queued request). The task's statistics are recorded under its label:
        ``estimate`` (default: the reserved bytes) and ``measured`` (its measured peak bytes, if known)."""
        if self.closed:
            return
        self.closed = True
        estimate = self.nbytes if estimate is None else estimate
        if self._slot >= 0 or self._waiter >= 0:
            self.governor._release(self, measured, estimate, record)
        elif record and measured is not None and self.granted:
            self.governor.note(self.label, estimate, measured)      # a task without a ledger entry

    def shrink(self, nbytes: int, *, measured: int | None = None, estimate: int | None = None) -> None:
        """Keep only ``nbytes`` (<= the held bytes) reserved from now on, releasing the rest at once (the state a
        finished call retains); the finished call's statistics are recorded as for `release`."""
        nbytes = max(0, int(nbytes))
        if not self.granted or self.closed or nbytes > self.nbytes:
            raise ValueError("only a granted reservation can shrink, and only to fewer bytes")
        if self.nbytes and self._slot >= 0:
            self.governor._shrink(self, nbytes, measured, self.nbytes if estimate is None else estimate)
        self.nbytes = nbytes

    def __repr__(self):
        state = "granted" if self.granted else "released" if self.closed else "queued"
        return f"Reservation({self.label!r}, {self.nbytes / GIB:.3f} GiB, depth {self.depth}, {state})"


def _gib(value) -> str:
    return f"{value / GIB:.2f} GiB"


def graded_limit(limit: int, depth: int) -> int:
    """The ledger total that a request at ``depth`` may fill (see NESTED_HEADROOM_FRACTION)."""
    return limit - (int(limit * NESTED_HEADROOM_FRACTION) >> max(0, int(depth) - 1))


def headroom_bytes(available: int) -> int:
    """The headroom kept free of reservations: 10% of ``available``, at least 4 GiB, at most half."""
    return min(available // 2, max(HEADROOM_MIN_BYTES, int(available * HEADROOM_FRACTION)))


class MemoryGovernor:
    """The run's memory admission control (see the module docstring). Use `memory_governor()` to get it."""

    def __init__(self, ledger: _Ledger, *, owner: bool):
        self._ledger = ledger
        self._owner = owner
        self._labels: dict[str, int] = {}
        self._slots: dict[int, int] = {}          # token id -> slot, for tokens this process touched
        self._live_value: int | None = None
        self._live_time = -1.0
        self._live_lock = threading.Lock()
        self._purged_at = 0.0
        self._logged: dict[str, float] = {}

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, memory_limit: int | None = None) -> "MemoryGovernor":
        """A new ledger owned by this process: limit = available memory now minus headroom, at most
        ``memory_limit`` bytes (None: no configured limit)."""
        global _STALE_SEGMENTS
        directory = _directory()
        _remove_stale(directory, _PREFIX)
        if _SEGMENT_DIRECTORY.is_dir():         # before reading the available memory, which they reduce
            _STALE_SEGMENTS = _remove_stale(_SEGMENT_DIRECTORY, SEGMENT_PREFIX)
        pid, started = _identity()
        path = directory / f"{_PREFIX}{os.getuid()}-{pid}-{started}-{uuid.uuid4().hex[:8]}"
        ledger = _Ledger(path, create=True)
        available = available_memory_bytes()
        if available is None:                  # nothing known about the node: only a configured limit binds
            headroom, node_limit, available = 0, 1 << 62, -1
        else:
            headroom = headroom_bytes(available)
            node_limit = max(0, available - headroom)
        limit = node_limit if memory_limit is None else min(node_limit, max(0, int(memory_limit)))
        with ledger.locked() as h:
            h[_MAGIC_AT] = _MAGIC
            h[_LIMIT] = limit
            h[_CONFIGURED] = -1 if memory_limit is None else int(memory_limit)
            h[_HEADROOM] = h[_LIVE_HEADROOM] = headroom
            h[_START] = available
            h[_CREATED] = time.time_ns()
        governor = cls(ledger, owner=True)
        atexit.register(governor.close)
        return governor

    @classmethod
    def attach(cls, handle: str) -> "MemoryGovernor":
        """The ledger created by an ancestor process (`handle`)."""
        ledger = _Ledger(Path(handle), create=False)
        if int(ledger.h[_MAGIC_AT]) != _MAGIC:
            raise RuntimeError(f"{handle} is not a memory ledger")
        return cls(ledger, owner=False)

    @property
    def handle(self) -> str:
        """What a child process needs to attach (the ledger's path)."""
        return str(self._ledger.path)

    @property
    def owner(self) -> tuple[int, int]:
        """(pid, start time) of the process that created the ledger (the run's driver)."""
        return _named_process(self._ledger.path.name, f"{_PREFIX}{os.getuid()}-") or _identity()

    def close(self) -> None:
        """The owner removes the ledger file (attached processes keep their mappings until they exit)."""
        if self._owner:
            self._owner = False
            try:
                os.unlink(self._ledger.path)
            except OSError:
                pass

    # ------------------------------------------------------------------ properties
    @property
    def limit(self) -> int:
        """Bytes the run may hold in reservations at once."""
        return int(self._ledger.h[_LIMIT])

    @property
    def headroom(self) -> int:
        return int(self._ledger.h[_HEADROOM])

    @property
    def reserved(self) -> int:
        """Bytes currently held in reservations (all processes of the run)."""
        return int(self._ledger.h[_RESERVED])

    def describe(self) -> str:
        h = self._ledger.h
        configured, start = int(h[_CONFIGURED]), int(h[_START])
        text = (f"memory limit {_gib(self.limit)} for planned peaks (available at start "
                f"{'unknown' if start < 0 else _gib(start)}, headroom {_gib(self.headroom)}")
        text += f", configured {_gib(configured)})" if configured >= 0 else ")"
        if self._owner and _STALE_SEGMENTS[0]:
            text += (f"; removed {_STALE_SEGMENTS[0]} shared-memory segments of killed runs "
                     f"({_gib(_STALE_SEGMENTS[1])})")
        return text

    # ------------------------------------------------------------------ requests
    def request(self, nbytes: int, *, label: str = "task", depth: int | None = None, minimum: int | None = None,
                queued: bool = False, parent: int | None = None, envelope: bool = False) -> Reservation:
        """A queued request (not yet granted): `poll` / `wait`, then `release`. ``depth`` and ``parent`` default to
        one below / the calling thread's innermost reservation; ``minimum`` makes it flexible and ``envelope`` makes
        it cover its nested requests (see the module docstring); ``queued``: it is reserved for a task that starts
        later (`start` marks it running)."""
        held = held_context()
        return Reservation(self, nbytes, label, depth_below(held) if depth is None else depth, minimum,
                           queued=queued, parent=(held[0] if held else 0) if parent is None else parent,
                           envelope=envelope)

    def presence(self, label: str, depth: int, parent: int | None = None, *, queued: bool = False) -> Reservation:
        """A granted zero-byte token marking running work without an estimate (so that the progress rule sees it);
        release it when the work ends (its measured peak is recorded under ``label``). ``queued``: the work starts
        later (`start` marks it running)."""
        held = held_context()
        token = Reservation(self, 0, label, depth, presence=True, queued=queued,
                            parent=(held[0] if held else 0) if parent is None else parent)
        token.poll()
        return token

    def start(self, token: int) -> None:
        """The task a QUEUED token was reserved for has started: the token now marks running work."""
        if token:
            self._set_states([token], RUNNING)

    def rest(self, token: int) -> None:
        """A token now holds resident memory only (retained state between calls), not running work."""
        if token:
            self._set_states([token], RESIDENT)

    @contextmanager
    def reserve(self, nbytes: int, *, label: str = "task", minimum: int | None = None, envelope: bool = False):
        """Wait until ``nbytes`` fit (or, with ``minimum``, as much as fits now but at least ``minimum``), then hold
        them for the block as the calling thread's innermost reservation; the granted bytes are ``.nbytes``.
        ``envelope``: the bytes cover what the block's nested requests reserve (they borrow them first)."""
        reservation = self.request(nbytes, label=label, minimum=minimum, envelope=envelope)
        reservation.wait()
        entry = [reservation.id, reservation.depth, True]
        if reservation.id:
            _CONTEXT.held.append(entry)
        try:
            yield reservation
        finally:
            if reservation.id:
                _CONTEXT.held.remove(entry)
            reservation.release()

    @contextmanager
    def suspended(self):
        """The calling thread waits for its own sub-tasks during the block: its reservations count as held by a
        waiting holder (the progress rule), and are marked running again afterwards."""
        context = _CONTEXT
        ids = _executing_ids() if context.suspended == 0 else []
        context.suspended += 1
        if ids:
            self._set_states(ids, SUSPENDED)
        try:
            yield
        finally:
            context.suspended -= 1
            if ids:
                self._set_states(ids, RUNNING)

    def worker_limit(self, estimates, workers: int, *, label: str, available: int | None = None) -> int:
        """How many of the largest ``workers`` concurrent tasks (peak bytes ``estimates``) fit in the limit
        (or in ``available`` bytes minus headroom); at least one. Tasks then reserve their bytes as they run."""
        estimates = [max(0, int(value)) for value in estimates]
        if not estimates:
            return 1
        usable = (graded_limit(self.limit, depth_below(held_context())) if available is None
                  else max(0, int(available) - headroom_bytes(int(available))))
        total = count = 0
        for estimate in sorted(estimates, reverse=True)[:max(1, int(workers))]:
            if total + estimate > usable:
                break
            total += estimate
            count += 1
        if not count:
            print(f"[memory] {label}: the largest task ({_gib(max(estimates))}) exceeds the memory limit "
                  f"({_gib(usable)}); one worker", flush=True)
            return 1
        print(f"[memory] {label}: {count} concurrent workers; largest concurrent peak {_gib(total)} of "
              f"{_gib(usable)}", flush=True)
        return count

    def note(self, label: str, estimate: int, measured: int | None) -> None:
        """Record the statistics of a finished task that held no reservation (its estimate was zero)."""
        index = self._label_index(label)
        with self._ledger.locked():
            self._record(index, estimate, measured)

    # ------------------------------------------------------------------ admission
    def _released(self) -> int:
        """How many releases the ledger has seen (read without the lock: an aligned 8-byte load)."""
        return int(self._ledger.h[_RELEASED])

    def _live(self, fresh: bool) -> int | None:
        now = time.monotonic()
        with self._live_lock:
            if fresh or now - self._live_time > LIVE_CACHE_SECONDS:
                self._live_value = available_memory_bytes()
                self._live_time = now
            return self._live_value

    def _label_index(self, label: str) -> int:
        index = self._labels.get(label)
        if index is not None:
            return index
        name = label.encode("utf-8", "replace")[:_LABEL["name"].itemsize]
        ledger = self._ledger
        with ledger.locked():
            names = ledger.l["name"]
            found = np.flatnonzero(names == name)
            if found.size:
                index = int(found[0])
            else:
                free = np.flatnonzero(names == b"")
                index = int(free[0]) if free.size else LABELS - 1
                if free.size:
                    names[index] = name
        self._labels[label] = index
        return index

    def _slot(self, token: int) -> int:
        """The slot of a live token (-1 if it is gone); call with the ledger locked."""
        t_id = self._ledger.t_id
        slot = self._slots.get(token, -1)
        if slot < 0 or t_id[slot] != token:
            found = np.flatnonzero(t_id == token)
            slot = int(found[0]) if found.size else -1
            if slot >= 0:
                self._slots[token] = slot
            else:
                self._slots.pop(token, None)
        return slot

    def _lender(self, token: int) -> tuple[int, int]:
        """(slot, spare bytes) of the nearest envelope with spare bytes on the parent chain of ``token`` (-1, 0 if
        none); call with the ledger locked."""
        ledger = self._ledger
        for _ in range(64):
            slot = self._slot(token)
            if slot < 0:
                return -1, 0
            if ledger.t_envelope[slot]:
                spare = int(ledger.t_bytes[slot]) + int(ledger.t_borrowed[slot]) - int(ledger.t_lent[slot])
                if spare > 0:
                    return slot, spare
            token = int(ledger.t_parent[slot])
            if not token:
                return -1, 0
        return -1, 0

    def _give_back(self, slot: int) -> None:
        """Return what the token in ``slot`` borrowed to its lender (call with the ledger locked)."""
        ledger = self._ledger
        borrowed = int(ledger.t_borrowed[slot])
        if borrowed:
            lender = self._slot(int(ledger.t_lender[slot]))
            if lender >= 0:
                ledger.t_lent[lender] = max(0, int(ledger.t_lent[lender]) - borrowed)
            ledger.t_borrowed[slot] = 0

    def _admit(self, request: Reservation, fresh: bool) -> bool:
        """One admission attempt (see the module docstring for the rules)."""
        ledger = self._ledger
        live = self._live(fresh)
        if fresh:
            self._purge()
        if request._label < 0:
            request._label = self._label_index(request.label)
        nbytes, minimum = request.nbytes, request.minimum
        now = time.monotonic_ns()
        alone = forced = False
        with ledger.locked() as h:
            # What the nearest envelope above the request can lend: taken first, the rest comes from the ledger.
            lender, borrow = self._lender(request.parent) if request.parent and nbytes else (-1, 0)
            borrow = min(borrow, nbytes)
            need, need_min = nbytes - borrow, max(0, minimum - borrow)
            committed, limit, start = int(h[_RESERVED]), int(h[_LIMIT]), int(h[_START])
            usable = graded_limit(limit, request.depth)
            # Reserved bytes not yet visible as used memory (their tasks may not have allocated yet).
            pending = committed
            if live is not None and start >= 0:
                used = max(0, start - live)
                if used > h[_PEAK_USED]:
                    h[_PEAK_USED] = used
                pending = max(0, committed - used)
            room = None if live is None else live - int(h[_LIVE_HEADROOM]) - pending

            def take(extra, ceiling=usable):
                """Ledger bytes this request takes next to ``extra`` claimed bytes (None: it does not fit)."""
                if need <= 0:
                    return 0
                free = ceiling - committed - extra
                if room is not None:
                    free = min(free, room - extra)
                if free >= need:
                    return need
                return free if free >= max(need_min, 1) else None

            grant = take(0) if request._waiter < 0 and (not h[_WAITING] or need <= 0) else None
            if request.presence:
                admit, since, grant = True, now, 0               # marks running work; holds no memory
            elif grant is not None:
                admit = True                                     # it fits and nothing queued (or its envelope
                since = now                                      # covers it): no queue
            else:
                if request._waiter < 0:                          # join the queue
                    slot = int(ledger.w_id.argmin())
                    if ledger.w_id[slot]:
                        raise RuntimeError("the memory ledger's request table is full")
                    pid, started = _identity()
                    request.id = int(h[_NEXT_ID]) + 1
                    h[_NEXT_ID] = request.id
                    (ledger.w_id[slot], ledger.w_pid[slot], ledger.w_started[slot], ledger.w_bytes[slot],
                     ledger.w_since[slot], ledger.w_depth[slot], ledger.w_label[slot]) = (
                        request.id, pid, started, need_min, now, request.depth, request._label)
                    h[_WAITING] += 1
                    request._waiter = slot
                ledger.w_bytes[request._waiter] = need_min      # its claim (borrowing can change it)
                queued = np.flatnonzero(ledger.w_id)
                head = int(queued[np.lexsort((ledger.w_id[queued], ledger.w_since[queued],
                                              -ledger.w_depth[queued]))[0]])
                is_head = head == request._waiter
                grant = take(0 if is_head else int(ledger.w_bytes[head]))
                admit = grant is not None
                if not admit and is_head:
                    active = ledger.t_id != 0
                    if not active.any():
                        admit = alone = True             # a flexible request: what fits now, >= its minimum
                        grant = max(need_min, min(need, limit - committed, need if room is None else room))
                    elif (ledger.t_state[active] == RUNNING).any():
                        h[_BLOCKED_SINCE] = 0
                    elif (ledger.t_state[active] == RESIDENT).all() and (grant := take(0, limit)) is not None:
                        admit = True                     # only retained states are held (no work that could need
                                                         # nested room): the head may use the whole limit at once
                    elif not h[_BLOCKED_SINCE]:
                        h[_BLOCKED_SINCE] = now
                    elif now - int(h[_BLOCKED_SINCE]) >= _PROGRESS_NS:
                        admit = forced = True
                        grant = need_min
                since = int(ledger.w_since[request._waiter])
                if admit:
                    ledger.w_id[request._waiter] = 0
                    h[_WAITING] -= 1
                    request._waiter = -1
                else:
                    request.waited = True
                waiting = int(queued.size)
            if admit:
                ledger_bytes = int(grant)
                if not nbytes:
                    borrow = 0
                request.nbytes = borrow + ledger_bytes
                slot = int(ledger.t_id.argmin())
                if ledger.t_id[slot]:
                    raise RuntimeError("the memory ledger's reservation table is full")
                if not request.id:
                    request.id = int(h[_NEXT_ID]) + 1
                    h[_NEXT_ID] = request.id
                pid, started = _identity()
                (ledger.t_id[slot], ledger.t_pid[slot], ledger.t_started[slot], ledger.t_bytes[slot],
                 ledger.t_granted[slot], ledger.t_depth[slot], ledger.t_state[slot], ledger.t_label[slot],
                 ledger.t_forced[slot], ledger.t_parent[slot], ledger.t_envelope[slot], ledger.t_lent[slot],
                 ledger.t_borrowed[slot], ledger.t_lender[slot]) = (
                    request.id, pid, started, ledger_bytes, now, request.depth,
                    QUEUED if request.queued else RUNNING, request._label, int(forced), request.parent,
                    int(request.envelope), 0, borrow, int(ledger.t_id[lender]) if borrow else 0)
                if borrow:
                    ledger.t_lent[lender] += borrow
                self._slots[request.id] = slot
                request._slot = slot
                request.granted = True
                nbytes = request.nbytes
                h[_RESERVED] = committed + ledger_bytes
                if committed + ledger_bytes > h[_PEAK_RESERVED]:
                    h[_PEAK_RESERVED] = committed + ledger_bytes
                if nbytes > h[_LARGEST]:
                    h[_LARGEST] = nbytes
                h[_REQUESTS] += 1
                if not request.queued:
                    h[_BLOCKED_SINCE] = 0
                labels = ledger.l
                if request.waited:
                    h[_WAITS] += 1
                    h[_WAIT_NS] += now - since
                    labels["waits"][request._label] += 1
                    labels["wait_ns"][request._label] += now - since
                if alone:
                    h[_ALONE] += 1
                    labels["alone"][request._label] += 1
                if forced:
                    h[_FORCED] += 1
                    labels["forced"][request._label] += 1
        if admit:
            if (alone or forced) and self._first_in_a_while(("alone" if alone else "forced", request.label)):
                reason = ("does not fit even alone; it runs alone" if alone else
                          "is admitted over the limit because every holder of memory is waiting")
                print(f"[memory] WARNING {request.label}: {_gib(nbytes)} {reason} (reserved {_gib(committed)} of "
                      f"{_gib(limit)}; available now {'unknown' if live is None else _gib(live)}; further cases "
                      f"of this label are counted in the summary)", flush=True)
            return True
        if not request._logged:
            request._logged = True
            self._log_wait(request, committed, limit, live, waiting)
        return False

    def _first_in_a_while(self, key) -> bool:
        """Whether this process last logged ``key`` more than LOG_SECONDS ago (then it logs it now)."""
        now = time.monotonic()
        if now - self._logged.get(key, -LOG_SECONDS) < LOG_SECONDS:
            return False
        self._logged[key] = now
        return True

    def _log_wait(self, request, committed, limit, live, waiting):
        if not self._first_in_a_while(("wait", request.label)):
            return
        print(f"[memory] pacing {request.label}: waits for {_gib(request.nbytes)} (reserved {_gib(committed)} of "
              f"{_gib(limit)}; available now {'unknown' if live is None else _gib(live)}; {waiting} waiting)",
              flush=True)

    def _release(self, request: Reservation, measured, estimate: int, record: bool) -> None:
        ledger = self._ledger
        with ledger.locked() as h:
            if request._waiter >= 0:                 # withdrawn before it was granted (the head may change)
                if ledger.w_id[request._waiter] == request.id:
                    ledger.w_id[request._waiter] = 0
                    h[_WAITING] -= 1
                    h[_RELEASED] += 1
                request._waiter = -1
                return
            slot = request._slot
            request._slot = -1
            self._slots.pop(request.id, None)
            if ledger.t_id[slot] != request.id:
                return                               # already released by a purge
            h[_RESERVED] = max(0, int(h[_RESERVED]) - int(ledger.t_bytes[slot]))
            forced = bool(ledger.t_forced[slot])
            self._give_back(slot)
            ledger.t_id[slot] = 0
            h[_RELEASED] += 1
            if forced:
                active = ledger.t_id != 0
                if active.any() and not (ledger.t_state[active] == RUNNING).any():
                    # Every holder still waits: the next waiting request may follow over the limit at once.
                    h[_BLOCKED_SINCE] = time.monotonic_ns() - _PROGRESS_NS
            if record:
                self._record(request._label, estimate, measured)

    def _shrink(self, request: Reservation, nbytes: int, measured, estimate: int) -> None:
        ledger = self._ledger
        with ledger.locked() as h:
            slot = request._slot
            if ledger.t_id[slot] != request.id:
                return
            # Keep the borrowed part first (up to nbytes), then ledger bytes.
            borrowed = int(ledger.t_borrowed[slot])
            keep_borrowed = min(borrowed, nbytes)
            keep_ledger = min(int(ledger.t_bytes[slot]), nbytes - keep_borrowed)
            if keep_borrowed < borrowed:
                lender = self._slot(int(ledger.t_lender[slot]))
                if lender >= 0:
                    ledger.t_lent[lender] = max(0, int(ledger.t_lent[lender]) - (borrowed - keep_borrowed))
                ledger.t_borrowed[slot] = keep_borrowed
            h[_RESERVED] = max(0, int(h[_RESERVED]) - (int(ledger.t_bytes[slot]) - keep_ledger))
            ledger.t_bytes[slot] = keep_ledger
            h[_RELEASED] += 1
            self._record(request._label, estimate, measured)

    def _record(self, index: int, estimate: int, measured) -> None:
        """Statistics of one finished task under its label (call with the ledger locked)."""
        labels = self._ledger.l
        labels["tasks"][index] += 1
        labels["estimate_sum"][index] += estimate
        if estimate > labels["estimate_max"][index]:
            labels["estimate_max"][index] = estimate
        if measured is not None:
            labels["measured"][index] += 1
            labels["measured_sum"][index] += measured
            if measured > labels["measured_max"][index]:
                labels["measured_max"][index] = measured
            if estimate > 0 and measured / estimate > labels["ratio_max"][index]:
                labels["ratio_max"][index] = measured / estimate
            if 0 < estimate < measured:                 # tasks without an estimate are not "under" it
                labels["under"][index] += 1

    def _set_states(self, ids, state: int) -> None:
        """Mark tokens (the progress clock is reset only by an admission attempt that sees a running token, so a
        holder that resumes for a moment between two waits does not keep restarting it)."""
        if not ids:
            return
        ledger = self._ledger
        with ledger.locked():
            for token in ids:
                slot = self._slot(token)
                if slot >= 0:
                    ledger.t_state[slot] = state

    def _purge(self) -> None:
        """Release reservations and withdraw requests of processes that no longer exist."""
        now = time.monotonic()
        if now - self._purged_at < PURGE_SECONDS:
            return
        self._purged_at = now
        ledger = self._ledger
        holders = set()
        with ledger.locked() as h:
            # One waiter of the run checks per PURGE_SECONDS (not every waiting process).
            if time.monotonic_ns() - int(h[_PURGE_NS]) < int(PURGE_SECONDS * 1e9):
                return
            h[_PURGE_NS] = time.monotonic_ns()
            for ids, pids, starts in ((ledger.t_id, ledger.t_pid, ledger.t_started),
                                      (ledger.w_id, ledger.w_pid, ledger.w_started)):
                used = np.flatnonzero(ids)
                holders.update(zip(pids[used].tolist(), starts[used].tolist()))
        dead = {holder for holder in holders if not _alive(*holder)}
        if not dead:
            return
        released = []
        with ledger.locked() as h:
            for slot in np.flatnonzero(ledger.t_id):
                if (int(ledger.t_pid[slot]), int(ledger.t_started[slot])) in dead:
                    released.append((int(ledger.t_pid[slot]), int(ledger.t_bytes[slot]),
                                     self._label_name(int(ledger.t_label[slot]))))
                    h[_RESERVED] = max(0, int(h[_RESERVED]) - int(ledger.t_bytes[slot]))
                    self._give_back(int(slot))
                    ledger.t_id[slot] = 0
            for slot in np.flatnonzero(ledger.w_id):
                if (int(ledger.w_pid[slot]), int(ledger.w_started[slot])) in dead:
                    ledger.w_id[slot] = 0
                    h[_WAITING] -= 1
            h[_PURGED] += len(released)
            h[_RELEASED] += 1
        for pid, nbytes, label in released:
            print(f"[memory] released {_gib(nbytes)} ({label}) held by exited process {pid}", flush=True)

    def _label_name(self, index: int) -> str:
        return bytes(self._ledger.l["name"][index]).decode("utf-8", "replace")

    # ------------------------------------------------------------------ statistics
    def statistics(self) -> dict:
        """Pacing statistics of the run so far (JSON-compatible; bytes and seconds)."""
        ledger = self._ledger
        with ledger.locked() as h:
            header = {name: int(h[index]) for index, name in enumerate(_FIELDS)}
            rows = ledger.labels[ledger.l["name"] != b""].copy()
        labels = {}
        for row in rows:
            tasks, measured = int(row["tasks"]), int(row["measured"])
            labels[bytes(row["name"]).decode("utf-8", "replace")] = dict(
                tasks=tasks, estimate_mean_bytes=float(row["estimate_sum"]) / tasks if tasks else 0.0,
                estimate_max_bytes=int(row["estimate_max"]), measured_tasks=measured,
                measured_mean_bytes=float(row["measured_sum"]) / measured if measured else None,
                measured_max_bytes=int(row["measured_max"]) if measured else None,
                measured_over_estimate_max=float(row["ratio_max"]) if measured else None,
                underestimated_tasks=int(row["under"]), waits=int(row["waits"]),
                wait_seconds=int(row["wait_ns"]) / 1e9, alone=int(row["alone"]), forced=int(row["forced"]))
        return dict(
            limit_bytes=header["limit"], configured_bytes=None if header["configured"] < 0 else header["configured"],
            headroom_bytes=header["headroom"],
            available_at_start_bytes=None if header["start_available"] < 0 else header["start_available"],
            reserved_bytes=header["reserved"], peak_reserved_bytes=header["peak_reserved"],
            peak_live_use_bytes=header["peak_used"], largest_request_bytes=header["largest"],
            reservations=header["requests"], waits=header["waits"], wait_seconds=header["wait_ns"] / 1e9,
            alone=header["alone"], forced=header["forced"], purged=header["purged"], labels=labels)

    def summary_lines(self) -> list[str]:
        """The statistics as log lines: totals, then one line per task label."""
        data = self.statistics()
        lines = [f"[memory] {self.describe()}; peak reserved {_gib(data['peak_reserved_bytes'])}, peak live use "
                 f"{_gib(data['peak_live_use_bytes'])}; {data['reservations']} reservations, {data['waits']} paced "
                 f"({data['wait_seconds']:.1f} s waiting), {data['alone']} ran alone, {data['forced']} admitted "
                 f"over the limit for progress"]
        for label, row in sorted(data["labels"].items()):
            if label.endswith(" worker recycled"):         # pools' recycle events (execution.pools)
                lines.append(f"[memory]   {label}: {row['tasks']} workers replaced; private memory at retirement "
                             f"max {_gib(row['measured_max_bytes'] or 0)}")
                continue
            measured = ("" if not row["measured_tasks"] else
                        f"; measured peak max {_gib(row['measured_max_bytes'])} (max measured/estimate "
                        f"{row['measured_over_estimate_max']:.2f}, {row['underestimated_tasks']} above estimate)")
            lines.append(f"[memory]   {label}: {row['tasks']} tasks, estimate max {_gib(row['estimate_max_bytes'])}"
                         f"{measured}; {row['waits']} paced ({row['wait_seconds']:.1f} s)")
        return lines


# ---------------------------------------------------------------------------------------------------------------
# The process's governor
# ---------------------------------------------------------------------------------------------------------------

_GOVERNOR: MemoryGovernor | None = None
_GOVERNOR_LOCK = threading.Lock()


def memory_governor() -> MemoryGovernor:
    """The run's governor: created by `execution.start_run` (or attached from the parent process); outside a
    run, a governor for this process tree is created on first use (limit from the memory available then)."""
    global _GOVERNOR
    governor = _GOVERNOR
    if governor is None:
        with _GOVERNOR_LOCK:
            if _GOVERNOR is None:
                _GOVERNOR = MemoryGovernor.create()
            governor = _GOVERNOR
    return governor


def start_governor(memory_limit: int | None = None) -> MemoryGovernor:
    """A new governor for this process tree (run start), replacing any earlier one of this process."""
    global _GOVERNOR
    governor = MemoryGovernor.create(memory_limit)
    with _GOVERNOR_LOCK:
        previous, _GOVERNOR = _GOVERNOR, governor
    if previous is not None:
        previous.close()
    return governor


def governor_handle() -> str:
    """What pools and executors pass to their workers (creating this process tree's governor if needed)."""
    return memory_governor().handle


def attach_governor(handle: str | None) -> None:
    """In a freshly started worker: use the parent's governor (one of its own if that fails)."""
    global _GOVERNOR
    if handle is None:
        return
    try:
        governor = MemoryGovernor.attach(handle)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[memory] WARNING cannot attach the run's memory ledger ({error}); this process paces alone",
              flush=True)
        return
    with _GOVERNOR_LOCK:
        _GOVERNOR = governor


def segment_name() -> str:
    """A new name for a shared-memory segment of this run (SEGMENT_PREFIX, named after the run's driver; outside a
    run, after this process)."""
    governor = _GOVERNOR
    pid, started = governor.owner if governor is not None else _identity()
    return f"{SEGMENT_PREFIX}{os.getuid()}-{pid}-{started}-{uuid.uuid4().hex[:12]}"


def reserve_memory(nbytes: int, *, label: str = "task", minimum: int | None = None, envelope: bool = False):
    """``with reserve_memory(n, label=...) as r:`` hold ``n`` bytes of the run's memory for the block (waits while
    they do not fit). For large allocations of stage code outside the parallel constructs. With ``minimum`` the
    block gets as much as fits now (``r.nbytes``), at least ``minimum`` (e.g. to size batches); with ``envelope``
    the bytes also cover the requests nested in the block."""
    return memory_governor().reserve(nbytes, label=label, minimum=minimum, envelope=envelope)


def memory_statistics() -> dict | None:
    """The run's pacing statistics, or None if this process has no governor."""
    return None if _GOVERNOR is None else _GOVERNOR.statistics()
