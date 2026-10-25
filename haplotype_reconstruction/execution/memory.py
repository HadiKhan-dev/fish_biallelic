"""Admission from measured job memory, shared by all processes of a run.

The job cgroup working set (usage minus inactive file cache; MemAvailable fallback) is sampled once per run
per MEMORY_SAMPLE_SECONDS. Below SOFT, outer starts are unrestricted by memory. From SOFT to HARD, an outer
start must wait for a measurement taken at least one full sampling interval after the previous outer admission.
At HARD, no new outer work starts except one progress task when no work is running or already queued to start.
Nested continuations of admitted work never wait for memory; independent pool/contig tasks are outer work.
QUEUED never counts as RUNNING, and concurrent tasks acquire CPU seats before their admission. Waiting parents
do not count as running, so the progress rule can always admit the next task needed to finish and free memory.
Admission never signals processes; the owner watchdog aborts a stalled run with a diagnostic.

No estimate enters admission and no bytes are reserved. Compatibility byte fields stay zero. Requested sizes
remain only for legacy estimate diagnostics and flexible buffer/batch sizing; they never delay a start.
The request/presence/start/release/reserve_memory APIs, heap trimming and stale-ledger cleanup are retained.
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

from .resources import memory_pressure
from .config import MEMORY_SOFT_FRACTION, MEMORY_HARD_FRACTION, MEMORY_SAMPLE_SECONDS

GIB = 1 << 30
HEADROOM_FRACTION = 0.10
HEADROOM_MIN_BYTES = 4 * GIB
# Longest pause between full re-checks of a waiting request (the first re-checks come sooner); between them a
# waiter looks every WAKE_SECONDS, without the lock, whether anything was released and then re-checks at once.
POLL_SECONDS = 0.5
WAKE_SECONDS = 0.02
# A process re-reads the live available memory at most this often outside waits.
LIVE_CACHE_SECONDS = MEMORY_SAMPLE_SECONDS
# Waiters look for reservations of exited processes at most this often.
PURGE_SECONDS = 2.0
# A process logs a waiting task label at most this often.
LOG_SECONDS = 30.0

TOKENS = 4096
WAITERS = 4096
LABELS = 256
_MAGIC = 0x48_41_50_4D_45_4D_34        # "HAPMEM4": pressure admission

# Token states: its holder runs, waits for its own sub-tasks, waits for memory, or the task it was reserved for
# has not started yet (a pool task granted by the parent until its worker picks it up). Only RUNNING counts as
# work in progress for the progress rule.
RUNNING, SUSPENDED, WAITING, QUEUED, RESIDENT = 1, 2, 3, 4, 5      # RESIDENT: retained state only (rest)

# Header: one int64 per field, by index (fast scalar access from every process).
_FIELDS = ("magic", "limit", "configured", "headroom", "live_headroom", "start_available", "next_id", "reserved",
           "waiting", "peak_reserved", "peak_used", "largest", "requests", "waits", "wait_ns", "alone", "forced",
           "purged", "blocked_since", "created_ns", "released", "purge_ns",
           "live_bytes", "sample_ns", "starts", "available_bytes", "outer_start_ns")
(_MAGIC_AT, _LIMIT, _CONFIGURED, _HEADROOM, _LIVE_HEADROOM, _START, _NEXT_ID, _RESERVED, _WAITING, _PEAK_RESERVED,
 _PEAK_USED, _LARGEST, _REQUESTS, _WAITS, _WAIT_NS, _ALONE, _FORCED, _PURGED, _BLOCKED_SINCE, _CREATED,
 _RELEASED, _PURGE_NS, _LIVE_BYTES, _SAMPLE_NS, _STARTS, _AVAILABLE_BYTES, _OUTER_START_NS) = range(len(_FIELDS))
_HEADER_SLOTS = 32
# Retain the token layout for diagnostics/watchdog compatibility; byte/lending fields are always zero.
_TOKEN = np.dtype([("id", "<i8"), ("pid", "<i8"), ("started", "<i8"), ("bytes", "<i8"), ("granted", "<i8"),
                   ("depth", "<i8"), ("state", "<i8"), ("label", "<i8"), ("forced", "<i8"), ("parent", "<i8"),
                   ("envelope", "<i8"), ("lent", "<i8"), ("borrowed", "<i8"), ("lender", "<i8")])
_WAITER = np.dtype([("id", "<i8"), ("pid", "<i8"), ("started", "<i8"), ("bytes", "<i8"), ("since", "<i8"),
                    ("depth", "<i8"), ("label", "<i8")])
_LABEL = np.dtype([("name", "S56"), ("tasks", "<i8"), ("estimate_sum", "<f8"), ("estimate_max", "<i8"),
                   ("measured", "<i8"), ("measured_sum", "<f8"), ("measured_max", "<i8"), ("under", "<i8"),
                   ("ratio_max", "<f8"), ("waits", "<i8"), ("wait_ns", "<i8"), ("alone", "<i8"),
                   ("forced", "<i8"), ("live_peak", "<i8")])
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
_SAMPLE_NS_INTERVAL = int(MEMORY_SAMPLE_SECONDS * 1e9)


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
    """An admission/progress token. ``nbytes`` is diagnostic or flexible-buffer metadata, never reserved."""

    __slots__ = ("governor", "nbytes", "minimum", "label", "depth", "id", "granted", "waited", "closed",
                 "queued", "presence", "parent", "envelope", "outer", "_waiter", "_slot", "_label", "_logged")

    def __init__(self, governor: "MemoryGovernor", nbytes: int, label: str, depth: int, minimum: int | None = None,
                 *, queued: bool = False, presence: bool = False, parent: int = 0, envelope: bool = False,
                 outer: bool = False):
        self.governor = governor
        self.nbytes = max(0, int(nbytes))
        # parent: the token this request is nested under (0: none); envelope: it covers its nested requests.
        self.parent = int(parent or 0)
        self.envelope = bool(envelope)
        self.outer = bool(outer) or not self.parent
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
        self.granted = False
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
        self._purged_at = 0.0
        self._logged: dict[str, float] = {}
        self._sample_stop = threading.Event()
        self._sampler = None
        self._watchdog = None

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, memory_limit: int | None = None, *, diagnostic_directory=None) -> "MemoryGovernor":
        """New run ledger: job memory capacity, capped by ``memory_limit`` when supplied."""
        global _STALE_SEGMENTS
        directory = _directory()
        _remove_stale(directory, _PREFIX)
        if _SEGMENT_DIRECTORY.is_dir():         # before reading the available memory, which they reduce
            _STALE_SEGMENTS = _remove_stale(_SEGMENT_DIRECTORY, SEGMENT_PREFIX)
        pid, started = _identity()
        path = directory / f"{_PREFIX}{os.getuid()}-{pid}-{started}-{uuid.uuid4().hex[:8]}"
        ledger = _Ledger(path, create=True)
        sample = memory_pressure()
        used, node_limit, available = (0, 1 << 62, -1) if sample is None else sample
        limit = node_limit if memory_limit is None else min(node_limit, max(1, int(memory_limit)))
        with ledger.locked() as h:
            h[_MAGIC_AT] = _MAGIC
            h[_LIMIT] = limit
            h[_CONFIGURED] = -1 if memory_limit is None else int(memory_limit)
            h[_HEADROOM] = h[_LIVE_HEADROOM] = int(limit * (1 - MEMORY_HARD_FRACTION))
            h[_START] = available
            h[_LIVE_BYTES] = h[_PEAK_USED] = used
            h[_AVAILABLE_BYTES] = available
            h[_SAMPLE_NS] = time.monotonic_ns()
            h[_CREATED] = time.time_ns()
        governor = cls(ledger, owner=True)
        governor._sampler = threading.Thread(target=governor._sample_loop, name="memory-pressure", daemon=True)
        governor._sampler.start()
        from .memory_watchdog import LedgerWatchdog
        governor._watchdog = LedgerWatchdog(governor, diagnostic_directory=diagnostic_directory)
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
        if self._owner and self.owner[0] == os.getpid():
            if self._watchdog is not None:
                self._watchdog.close()
            self._owner = False
            self._sample_stop.set()
            if self._sampler is not None:
                self._sampler.join(timeout=2 * MEMORY_SAMPLE_SECONDS)
            try:
                os.unlink(self._ledger.path)
            except OSError:
                pass

    # ------------------------------------------------------------------ properties
    @property
    def limit(self) -> int:
        """Capacity against which measured working-set pressure is compared."""
        return int(self._ledger.h[_LIMIT])

    @property
    def headroom(self) -> int:
        return int(self._ledger.h[_HEADROOM])

    @property
    def reserved(self) -> int:
        """Compatibility counter: zero, since admission never reserves bytes."""
        return int(self._ledger.h[_RESERVED])

    def describe(self) -> str:
        h = self._ledger.h
        configured, start = int(h[_CONFIGURED]), int(h[_START])
        text = (f"memory limit {_gib(self.limit)} for measured pressure (available at start "
                f"{'unknown' if start < 0 else _gib(start)}, headroom {_gib(self.headroom)}")
        text += f", configured {_gib(configured)})" if configured >= 0 else ")"
        if self._owner and _STALE_SEGMENTS[0]:
            text += (f"; removed {_STALE_SEGMENTS[0]} shared-memory segments of killed runs "
                     f"({_gib(_STALE_SEGMENTS[1])})")
        return text

    # ------------------------------------------------------------------ requests
    def request(self, nbytes: int, *, label: str = "task", depth: int | None = None, minimum: int | None = None,
                queued: bool = False, parent: int | None = None, envelope: bool = False,
                outer: bool = False) -> Reservation:
        """Request a start; `poll` / `wait`, then `release` when done.

        ``outer`` marks an independent contig or pool task; nested continuations never wait for memory.
        ``queued`` remains QUEUED until `start`. Requested bytes do not affect admission; ``minimum``
        allows a smaller buffer/batch size after admission. ``envelope`` is accepted for API compatibility.
        """
        held = held_context()
        return Reservation(self, nbytes, label, depth_below(held) if depth is None else depth, minimum,
                           queued=queued, parent=(held[0] if held else 0) if parent is None else parent,
                           envelope=envelope, outer=outer)

    def presence(self, label: str, depth: int, parent: int | None = None, *, queued: bool = False,
                 outer: bool = False) -> Reservation:
        """A granted zero-byte token marking running work without an estimate (so that the progress rule sees it);
        release it when the work ends (its measured peak is recorded under ``label``). ``queued``: the work starts
        later (`start` marks it running)."""
        held = held_context()
        token = Reservation(self, 0, label, depth, presence=True, queued=queued,
                            parent=(held[0] if held else 0) if parent is None else parent, outer=outer)
        if not queued:
            token.wait()
        return token

    def start(self, token: int) -> None:
        """Mark an admitted task RUNNING only when it actually begins (never while it waits for a CPU slot)."""
        if token:
            with self._ledger.locked() as h:
                slot = self._slot(token)
                if slot >= 0 and self._ledger.t_state[slot] == QUEUED:
                    self._ledger.t_state[slot] = RUNNING
                    h[_STARTS] += 1

    def rest(self, token: int) -> None:
        """A token now holds resident memory only (retained state between calls), not running work."""
        if token:
            self._set_states([token], RESIDENT)

    @contextmanager
    def reserve(self, nbytes: int, *, label: str = "task", minimum: int | None = None, envelope: bool = False):
        """Admit a block from measured pressure, keeping only a progress token for its lifetime.

        A flexible request gets ``.nbytes`` between ``minimum`` and the requested estimate. Envelopes are ignored.
        """
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
        """Keep the CPU/task bound; measured pressure controls dispatch, not estimated lifetime peaks."""
        return max(1, min(int(workers), len(estimates))) if hasattr(estimates, "__len__") else max(1, int(workers))

    def note(self, label: str, estimate: int, measured: int | None) -> None:
        """Record the statistics of a finished task that held no reservation (its estimate was zero)."""
        index = self._label_index(label)
        with self._ledger.locked():
            self._record(index, estimate, measured)

    # ------------------------------------------------------------------ admission
    def _released(self) -> int:
        """How many releases the ledger has seen (read without the lock: an aligned 8-byte load)."""
        return int(self._ledger.h[_RELEASED])

    def _sample_loop(self):
        while not self._sample_stop.wait(MEMORY_SAMPLE_SECONDS):
            self._sample_pressure()

    def _sample_pressure(self) -> None:
        """One shared working-set sample at most every 0.5 s; observations are not task progress."""
        now = time.monotonic_ns()
        ledger = self._ledger
        if now - int(ledger.h[_SAMPLE_NS]) < _SAMPLE_NS_INTERVAL:
            return
        with ledger.locked() as h:
            if now - int(h[_SAMPLE_NS]) < _SAMPLE_NS_INTERVAL:
                return
            sample = memory_pressure()
            if sample is None:
                return
            used, _, available = sample
            # A lower configured limit measures this same job working set, not growth since run start.
            used = max(used, int(h[_LIMIT]) - available)
            h[_LIVE_BYTES] = used
            h[_AVAILABLE_BYTES] = available
            h[_PEAK_USED] = max(int(h[_PEAK_USED]), used)
            for label in np.unique(ledger.t_label[ledger.t_id != 0]):
                ledger.l["live_peak"][label] = max(int(ledger.l["live_peak"][label]), used)
            # The timestamp follows the read, so a slow-start grant has a full interval of observation.
            h[_SAMPLE_NS] = time.monotonic_ns()

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

    def _admit(self, request: Reservation, fresh: bool) -> bool:
        """Only measured pressure and a fresh observation gate outer starts; nested work continues."""
        self._sample_pressure()
        if fresh:
            self._purge()
        ledger = self._ledger
        if request._label < 0:
            request._label = self._label_index(request.label)
        now = time.monotonic_ns()
        forced = False
        with ledger.locked() as h:
            limit, used = int(h[_LIMIT]), int(h[_LIVE_BYTES])
            soft, hard = int(MEMORY_SOFT_FRACTION * limit), int(MEMORY_HARD_FRACTION * limit)
            active = ledger.t_id != 0
            running = bool((ledger.t_state[active] == RUNNING).any())
            queued_to_start = bool((ledger.t_state[active] == QUEUED).any())
            observed = int(h[_SAMPLE_NS]) - int(h[_OUTER_START_NS]) >= _SAMPLE_NS_INTERVAL
            # Existing work must be able to finish and free memory, including above HARD. An independent
            # pool/contig task is outer work even when its scheduler belongs to an existing contig.
            fits = not request.outer or (used < hard and (used < soft or observed))
            if fits and (not request.outer or used < soft or not h[_WAITING]):
                admit = True
                since = int(ledger.w_since[request._waiter]) if request._waiter >= 0 else now
            else:
                if request._waiter < 0:
                    slot = int(ledger.w_id.argmin())
                    if ledger.w_id[slot]:
                        raise RuntimeError("the memory ledger's request table is full")
                    request.id = int(h[_NEXT_ID]) + 1
                    h[_NEXT_ID] = request.id
                    pid, started = _identity()
                    (ledger.w_id[slot], ledger.w_pid[slot], ledger.w_started[slot], ledger.w_bytes[slot],
                     ledger.w_since[slot], ledger.w_depth[slot], ledger.w_label[slot]) = (
                        request.id, pid, started, 0, now, request.depth, request._label)
                    request._waiter = slot
                    h[_WAITING] += 1
                queued = np.flatnonzero(ledger.w_id)
                head = int(queued[np.lexsort((ledger.w_id[queued], -ledger.w_depth[queued]))[0]])
                is_head = head == request._waiter
                admit = fits and (not request.outer or is_head)
                if not admit and is_head and not running and not queued_to_start:
                    admit = forced = True
                since = int(ledger.w_since[request._waiter])
                if not admit:
                    request.waited = True
            if admit:
                if request._waiter >= 0:
                    ledger.w_id[request._waiter] = 0
                    h[_WAITING] -= 1
                    request._waiter = -1
                if not request.id:
                    request.id = int(h[_NEXT_ID]) + 1
                    h[_NEXT_ID] = request.id
                slot = int(ledger.t_id.argmin())
                if ledger.t_id[slot]:
                    raise RuntimeError("the memory ledger's reservation table is full")
                pid, started = _identity()
                (ledger.t_id[slot], ledger.t_pid[slot], ledger.t_started[slot], ledger.t_bytes[slot],
                 ledger.t_granted[slot], ledger.t_depth[slot], ledger.t_state[slot], ledger.t_label[slot],
                 ledger.t_forced[slot], ledger.t_parent[slot], ledger.t_envelope[slot], ledger.t_lent[slot],
                 ledger.t_borrowed[slot], ledger.t_lender[slot]) = (
                    request.id, pid, started, 0, now, request.depth,
                    QUEUED if request.queued else RUNNING, request._label, int(forced), request.parent,
                    int(request.outer), 0, 0, 0)
                self._slots[request.id] = request._slot = slot
                request.granted = True
                if request.outer:
                    h[_OUTER_START_NS] = time.monotonic_ns()
                # These sizes remain for buffer batching and diagnostics only, after admission is decided.
                if request.minimum < request.nbytes:
                    request.nbytes = min(request.nbytes, max(request.minimum, soft - used))
                h[_LARGEST] = max(int(h[_LARGEST]), request.nbytes)
                h[_REQUESTS] += 1
                if not request.queued:
                    h[_STARTS] += 1
                labels = ledger.l
                labels["live_peak"][request._label] = max(int(labels["live_peak"][request._label]), used)
                if request.waited:
                    h[_WAITS] += 1
                    h[_WAIT_NS] += now - since
                    labels["waits"][request._label] += 1
                    labels["wait_ns"][request._label] += now - since
                if forced:
                    h[_FORCED] += 1
                    labels["forced"][request._label] += 1
        if admit:
            if forced and self._first_in_a_while(("progress", request.label)):
                print(f"[memory] progress {request.label}: no running task; live {_gib(used)} of {_gib(limit)}",
                      flush=True)
            return True
        if not request._logged:
            request._logged = True
            self._log_wait(request, limit, used, int(h[_WAITING]))
        return False

    def _first_in_a_while(self, key) -> bool:
        """Whether this process last logged ``key`` more than LOG_SECONDS ago (then it logs it now)."""
        now = time.monotonic()
        if now - self._logged.get(key, -LOG_SECONDS) < LOG_SECONDS:
            return False
        self._logged[key] = now
        return True

    def _log_wait(self, request, limit, live, waiting):
        if not self._first_in_a_while(("wait", request.label)):
            return
        print(f"[memory] pacing {request.label}: live {_gib(live)} of {_gib(limit)}, "
              f"{'hard pressure' if live >= MEMORY_HARD_FRACTION * limit else 'awaiting next measured sample'}; "
              f"{waiting} waiting",
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
            ledger.t_id[slot] = 0
            h[_RELEASED] += 1
            if record:
                self._record(request._label, estimate, measured)

    def _shrink(self, request: Reservation, nbytes: int, measured, estimate: int) -> None:
        ledger = self._ledger
        with ledger.locked() as h:
            slot = request._slot
            if ledger.t_id[slot] != request.id:
                return
            # Retained state is already part of the measured working set.
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
        """Mark actual executing/waiting state; samples and burst expiry never count as task progress."""
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
                wait_seconds=int(row["wait_ns"]) / 1e9, alone=int(row["alone"]), forced=int(row["forced"]), peak_live_use_bytes=int(row["live_peak"]))
        return dict(
            limit_bytes=header["limit"], configured_bytes=None if header["configured"] < 0 else header["configured"],
            headroom_bytes=header["headroom"],
            available_at_start_bytes=None if header["start_available"] < 0 else header["start_available"],
            reserved_bytes=header["reserved"], peak_reserved_bytes=header["peak_reserved"],
            peak_live_use_bytes=header["peak_used"], largest_request_bytes=header["largest"],
            live_bytes=header["live_bytes"], sample_ns=header["sample_ns"], outer_start_ns=header["outer_start_ns"],
            starts=header["starts"], waiting=header["waiting"],
            burst_bytes=header["reserved"], soft_fraction=MEMORY_SOFT_FRACTION, hard_fraction=MEMORY_HARD_FRACTION,
            reservations=header["requests"], waits=header["waits"], wait_seconds=header["wait_ns"] / 1e9,
            alone=header["alone"], forced=header["forced"], purged=header["purged"], labels=labels)

    def summary_lines(self) -> list[str]:
        """The statistics as log lines: totals, then one line per task label."""
        data = self.statistics()
        lines = [f"[memory] {self.describe()}; peak live use "
                 f"{_gib(data['peak_live_use_bytes'])}; {data['reservations']} admissions, {data['waits']} paced "
                 f"({data['wait_seconds']:.1f} s waiting), {data['alone']} ran alone, {data['forced']} admitted "
                 f"for progress"]
        for label, row in sorted(data["labels"].items()):
            if label.endswith(" worker recycled"):         # pools' recycle events (execution.pools)
                lines.append(f"[memory]   {label}: {row['tasks']} workers replaced; private memory at retirement "
                             f"max {_gib(row['measured_max_bytes'] or 0)}")
                continue
            measured = ("" if not row["measured_tasks"] else
                        f"; measured peak max {_gib(row['measured_max_bytes'])} (max measured/estimate "
                        f"{row['measured_over_estimate_max']:.2f}, {row['underestimated_tasks']} above estimate)")
            lines.append(f"[memory]   {label}: {row['tasks']} tasks, estimate max {_gib(row['estimate_max_bytes'])}"
                         f"{measured}; observed job peak {_gib(row['peak_live_use_bytes'])}; {row['waits']} paced ({row['wait_seconds']:.1f} s)")
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


def start_governor(memory_limit: int | None = None, *, diagnostic_directory=None) -> MemoryGovernor:
    """A new governor for this process tree (run start), replacing any earlier one of this process."""
    global _GOVERNOR
    governor = MemoryGovernor.create(memory_limit, diagnostic_directory=diagnostic_directory)
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
    """Admit a block under measured pressure; ``nbytes`` never affects admission.

    ``r.nbytes`` is the admitted allocation size, at least ``minimum`` when supplied; no peak is held until exit.
    """
    return memory_governor().reserve(nbytes, label=label, minimum=minimum, envelope=envelope)


def memory_statistics() -> dict | None:
    """The run's pacing statistics, or None if this process has no governor."""
    return None if _GOVERNOR is None else _GOVERNOR.statistics()
