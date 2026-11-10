"""Own measured-pressure admission, ledger lifetimes and the stall watchdog."""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import gc
import json
import logging
import mmap
import os
import signal
import struct
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import ResourceError
from .runtime import memory_pressure

_LOG = logging.getLogger(__name__)


MEMORY_SOFT_FRACTION = 0.9
MEMORY_HARD_FRACTION = 1.0
MEMORY_SAMPLE_SECONDS = 0.5
MEMORY_STALL_MINUTES = 30.0
MEMORY_STALL_POLL_SECONDS = 60.0
MEMORY_STALL_CPU_CORES = 0.05
GIB = 1 << 30
POLL_SECONDS = 0.5
WAKE_SECONDS = 0.02
PURGE_SECONDS = 2.0
LOG_SECONDS = 30.0
TOKENS = 4096
WAITERS = 4096
LABELS = 256
_MAGIC = 20338011472743740
RUNNING, SUSPENDED, WAITING, QUEUED, RESIDENT = (1, 2, 3, 4, 5)
_FIELDS = (
    "magic",
    "limit",
    "configured",
    "start_available",
    "next_id",
    "waiting",
    "peak_used",
    "requests",
    "waits",
    "wait_ns",
    "forced",
    "purged",
    "created_ns",
    "released",
    "purge_ns",
    "live_bytes",
    "sample_ns",
    "starts",
    "available_bytes",
    "outer_start_ns",
    "token_capacity",
    "waiter_capacity",
    "label_capacity",
    "tokens_at",
    "waiters_at",
    "labels_at",
    "recovery_request_ns",
)
(
    _MAGIC_AT,
    _LIMIT,
    _CONFIGURED,
    _START,
    _NEXT_ID,
    _WAITING,
    _PEAK_USED,
    _REQUESTS,
    _WAITS,
    _WAIT_NS,
    _FORCED,
    _PURGED,
    _CREATED,
    _RELEASED,
    _PURGE_NS,
    _LIVE_BYTES,
    _SAMPLE_NS,
    _STARTS,
    _AVAILABLE_BYTES,
    _OUTER_START_NS,
    _TOKEN_CAPACITY,
    _WAITER_CAPACITY,
    _LABEL_CAPACITY,
    _TOKEN_OFFSET,
    _WAITER_OFFSET,
    _LABEL_OFFSET,
    _RECOVERY_REQUEST_NS,
) = range(len(_FIELDS))
_HEADER_SLOTS = 32
_TOKEN = np.dtype(
    [
        ("id", "<i8"),
        ("pid", "<i8"),
        ("started", "<i8"),
        ("granted", "<i8"),
        ("depth", "<i8"),
        ("state", "<i8"),
        ("label", "<i8"),
        ("forced", "<i8"),
        ("parent", "<i8"),
        ("outer", "<i8"),
    ]
)
_WAITER = np.dtype(
    [
        ("id", "<i8"),
        ("pid", "<i8"),
        ("started", "<i8"),
        ("since", "<i8"),
        ("depth", "<i8"),
        ("label", "<i8"),
        ("polled", "<i8"),
    ]
)
_LABEL = np.dtype(
    [
        ("name", "S56"),
        ("tasks", "<i8"),
        ("measured", "<i8"),
        ("measured_sum", "<f8"),
        ("measured_max", "<i8"),
        ("waits", "<i8"),
        ("wait_ns", "<i8"),
        ("forced", "<i8"),
        ("live_peak", "<i8"),
    ]
)
_TOKENS_AT = 8 * _HEADER_SLOTS
_WAITERS_AT = _TOKENS_AT + TOKENS * _TOKEN.itemsize
_LABELS_AT = _WAITERS_AT + WAITERS * _WAITER.itemsize
_SIZE = _LABELS_AT + LABELS * _LABEL.itemsize
_PREFIX = "haplotypes-memory-"
SEGMENT_PREFIX = "haplotypes-shm-"
_SEGMENT_DIRECTORY = Path("/dev/shm")
_SAMPLE_NS_INTERVAL = int(MEMORY_SAMPLE_SECONDS * 1000000000.0)
_LIBC: ctypes.CDLL | None
_FLOCK: Any  # ctypes function pointer exposes dynamic argtypes/restype descriptors.
try:
    _LIBC = ctypes.CDLL("libc.so.6")
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


def _status_bytes(field: str) -> int | None:
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
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


def process_start_ticks(pid: int) -> int | None:
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
        _IDENTITY = (pid, process_start_ticks(pid) or 0)
    return _IDENTITY


def process_alive(pid: int, started: int) -> bool:
    ticks = process_start_ticks(pid)
    return ticks is not None and (started == 0 or ticks == started)


class _Context(threading.local):
    def __init__(self):
        self.held = []
        self.suspended = 0


_CONTEXT = _Context()


def held_context() -> tuple[int, int] | None:
    """The calling thread's innermost admission token as (token id, depth), or None: what its sub-work inherits."""
    held = _CONTEXT.held
    return (held[-1][0], held[-1][1]) if held else None


def depth_below(held: tuple[int, int] | None) -> int:
    """The depth of requests made under ``held`` (1 at the top level)."""
    return 1 if held is None else int(held[1]) + 1


def _executing_ids() -> list[int]:
    return [entry[0] for entry in _CONTEXT.held if entry[2] and entry[0]]


@contextmanager
def adopted(held: tuple[int, int] | None, *, executing: bool = True):
    """Run the block under a admission token made elsewhere: a pool task under the token its parent reserved for
    it (``executing``: this thread does that work, so it marks the token while it waits), or a helper thread
    under its parent's innermost admission token (``executing=False``: depth only)."""
    if held is None:
        yield
        return
    entry = [int(held[0]), int(held[1]), bool(executing)]
    _CONTEXT.held.append(entry)
    try:
        yield
    finally:
        _CONTEXT.held.remove(entry)


def _directory() -> Path:
    shm = Path("/dev/shm")
    return shm if shm.is_dir() and os.access(shm, os.W_OK) else Path(tempfile.gettempdir())


_LINUX = sys.platform.startswith("linux")
_F_OFD_GETLK = getattr(fcntl, "F_OFD_GETLK", 36 if _LINUX else None)
_F_OFD_SETLK = getattr(fcntl, "F_OFD_SETLK", 37 if _LINUX else None)
_FLOCK_STRUCT = "hhqqi4x"


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
        pid, started = (int(part) for part in name[len(prefix) :].split("-")[:2])
    except ValueError:
        return None
    return (pid, started)


def _remove_stale(directory: Path, prefix: str) -> tuple[int, int]:
    """Delete this user's files ``<prefix><uid>-<pid>-<started>-...`` in ``directory`` that no process marks as in use
    and whose named process no longer exists (the ledger or shared segments of a run killed before it removed them):
    (files, bytes freed)."""
    prefix = f"{prefix}{os.getuid()}-"
    removed = freed = 0
    try:
        entries = [entry for entry in os.scandir(directory) if entry.name.startswith(prefix)]
    except OSError:
        return (0, 0)
    for entry in entries:
        owner = _named_process(entry.name, prefix)
        if owner is None or process_alive(*owner) or _in_use(entry.path):
            continue
        try:
            size = entry.stat(follow_symlinks=False).st_blocks * 512
            os.unlink(entry.path)
        except OSError:
            continue
        removed, freed = (removed + 1, freed + size)
    return (removed, freed)


_STALE_SEGMENTS = (0, 0)


class _Ledger:
    """The mapped ledger file: header, admission tokens (tokens), queued requests (waiters), per-label statistics.
    Fields are exposed as views (``t_*`` tokens, ``w_*`` waiters, ``l_*`` labels) so access stays cheap."""

    # Views rebound by _refresh through the structured table field names.
    t_id: np.ndarray
    t_pid: np.ndarray
    t_started: np.ndarray
    t_granted: np.ndarray
    t_depth: np.ndarray
    t_state: np.ndarray
    t_label: np.ndarray
    t_forced: np.ndarray
    t_parent: np.ndarray
    t_outer: np.ndarray
    w_id: np.ndarray
    w_pid: np.ndarray
    w_started: np.ndarray
    w_since: np.ndarray
    w_depth: np.ndarray
    w_label: np.ndarray
    w_polled: np.ndarray

    def __init__(self, path: Path, *, create: bool):
        self.path = Path(path)
        self.pid = os.getpid()
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        self.fd = os.open(self.path, flags, 384)
        mark_in_use(self.fd)
        self.thread_lock = threading.Lock()
        try:
            if create:
                os.ftruncate(self.fd, _SIZE)
            elif os.fstat(self.fd).st_size < 8 * _HEADER_SLOTS:
                raise RuntimeError(f"{self.path} is not a memory ledger of this package version")
            # The header never moves. Tables grow by appending a doubled copy under
            # flock; slot numbers stay stable in every process. Old ndarray views
            # keep their old mapping alive until the reader lets them go.
            self.header_map = mmap.mmap(self.fd, 8 * _HEADER_SLOTS, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            self.h = np.ndarray((_HEADER_SLOTS,), dtype="<i8", buffer=self.header_map)
            if create:
                self.h[
                    [_TOKEN_CAPACITY, _WAITER_CAPACITY, _LABEL_CAPACITY, _TOKEN_OFFSET, _WAITER_OFFSET, _LABEL_OFFSET]
                ] = (TOKENS, WAITERS, LABELS, _TOKENS_AT, _WAITERS_AT, _LABELS_AT)
            self._layout = ()
            with self.locked():
                pass
        except BaseException:
            os.close(self.fd)
            raise

    def _refresh(self):
        """Reattach tables after another process grew one; caller holds flock."""
        layout = tuple(
            int(self.h[i])
            for i in (_TOKEN_CAPACITY, _WAITER_CAPACITY, _LABEL_CAPACITY, _TOKEN_OFFSET, _WAITER_OFFSET, _LABEL_OFFSET)
        )
        if layout == self._layout:
            return
        nt, nw, nl, at, aw, al = layout
        self.map = mmap.mmap(self.fd, 0, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.tokens = np.ndarray((nt,), dtype=_TOKEN, buffer=self.map, offset=at)
        self.waiters = np.ndarray((nw,), dtype=_WAITER, buffer=self.map, offset=aw)
        self.labels = np.ndarray((nl,), dtype=_LABEL, buffer=self.map, offset=al)
        for prefix, table in (("t", self.tokens), ("w", self.waiters)):
            for name in table.dtype.names:
                setattr(self, f"{prefix}_{name}", table[name])
        self.l = {name: self.labels[name] for name in _LABEL.names}
        self._layout = layout

    def grow(self, table: str) -> None:
        """Append a doubled table, preserving every index. Call with the ledger locked.

        Capacities are initial sizes, never execution limits. Growth metadata is
        published only after the copy; attached processes refresh on their next lock.
        """
        cap, offset, values = {
            "tokens": (_TOKEN_CAPACITY, _TOKEN_OFFSET, self.tokens),
            "waiters": (_WAITER_CAPACITY, _WAITER_OFFSET, self.waiters),
            "labels": (_LABEL_CAPACITY, _LABEL_OFFSET, self.labels),
        }[table]
        start = os.fstat(self.fd).st_size
        count = 2 * len(values)
        os.ftruncate(self.fd, start + count * values.dtype.itemsize)
        mapped = mmap.mmap(self.fd, 0, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        grown = np.ndarray((count,), dtype=values.dtype, buffer=mapped, offset=start)
        grown[: len(values)] = values
        self.h[cap], self.h[offset] = count, start
        self._refresh()

    @contextmanager
    def locked(self):
        """Exclusive access across processes (flock, released by the kernel if a holder dies) and threads.

        The lock is first tried without blocking and without releasing the GIL; only a contended lock waits
        (releasing the GIL)."""
        with self.thread_lock:
            if self.pid != os.getpid():
                self.fd = os.open(self.path, os.O_RDWR)
                self.pid = os.getpid()
            if _FLOCK is None or _FLOCK(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB) != 0:
                fcntl.flock(self.fd, fcntl.LOCK_EX)
            try:
                self._refresh()
                yield self.h
            finally:
                if _FLOCK is None or _FLOCK(self.fd, fcntl.LOCK_UN) != 0:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)


class AdmissionToken:
    """An admission/progress token. ``nbytes`` is diagnostic or flexible-buffer metadata, never reserved."""

    __slots__ = (
        "governor",
        "label",
        "depth",
        "id",
        "granted",
        "waited",
        "closed",
        "queued",
        "parent",
        "outer",
        "_waiter",
        "_slot",
        "_label",
        "_logged",
    )

    def __init__(
        self,
        governor: MemoryGovernor,
        label: str,
        depth: int,
        *,
        queued: bool = False,
        parent: int = 0,
        outer: bool = False,
    ):
        self.governor = governor
        self.parent = int(parent or 0)
        self.outer = bool(outer) or not self.parent
        self.queued = bool(queued)
        self.label, self.depth = (label, int(depth))
        self.id = 0
        self.granted = self.waited = self.closed = self._logged = False
        self._waiter = self._slot = self._label = -1

    @property
    def held(self) -> tuple[int, int]:
        """(token id, depth): what work done under this admission token inherits (`adopted`)."""
        return (self.id, self.depth)

    def poll(self, *, fresh: bool = False) -> bool:
        """Try to be admitted now (never blocks); True once granted. The first failed call queues the request."""
        if self.granted:
            return True
        if self.closed:
            raise RuntimeError("a released admission token cannot be granted")
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

    def release(self, measured: int | None = None, *, record: bool = True) -> None:
        """Release a progress token or withdraw its queued request, waking waiters."""
        if self.closed:
            return
        self.closed = True
        if self._slot >= 0 or self._waiter >= 0:
            self.governor._release(self, measured, record)

    def __repr__(self) -> str:
        state = "released" if self.closed else "granted" if self.granted else "queued"
        return f"AdmissionToken({self.label!r}, depth={self.depth}, {state})"


def _gib(value) -> str:
    return f"{value / GIB:.2f} GiB"


class MemoryGovernor:
    """The run's memory admission control (see the module docstring). Use `memory_governor()` to get it."""

    def __init__(self, ledger: _Ledger, *, owner: bool):
        self._ledger = ledger
        self._owner = owner
        self._labels: dict[str, int] = {}
        self._slots: dict[int, int] = {}
        self._purged_at = 0.0
        self._logged: dict[str, float] = {}
        self._sample_stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self._watchdog: LedgerWatchdog | None = None
        self._recovery_seen = 0

    @classmethod
    def create(cls, memory_limit: int | None = None, *, diagnostic_directory=None) -> MemoryGovernor:
        """New run ledger: job memory capacity, capped by ``memory_limit`` when supplied."""
        global _STALE_SEGMENTS
        directory = _directory()
        _remove_stale(directory, _PREFIX)
        if _SEGMENT_DIRECTORY.is_dir():
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
            h[_START] = available
            h[_LIVE_BYTES] = h[_PEAK_USED] = used
            h[_AVAILABLE_BYTES] = available
            h[_SAMPLE_NS] = time.monotonic_ns()
            h[_CREATED] = time.time_ns()
        governor = cls(ledger, owner=True)
        governor._sampler = threading.Thread(target=governor._sample_loop, name="memory-pressure", daemon=True)
        governor._sampler.start()
        governor._watchdog = LedgerWatchdog(governor, diagnostic_directory=diagnostic_directory)
        atexit.register(governor.close)
        return governor

    @classmethod
    def attach(cls, handle: str) -> MemoryGovernor:
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

    @property
    def limit(self) -> int:
        """Capacity against which measured working-set pressure is compared."""
        return int(self._ledger.h[_LIMIT])

    def describe(self) -> str:
        return f"memory limit {_gib(self.limit)}; measured pressure {_gib(int(self._ledger.h[_LIVE_BYTES]))}"

    def request(
        self,
        *,
        label: str = "task",
        depth: int | None = None,
        queued: bool = False,
        parent: int | None = None,
        outer: bool = False,
    ) -> AdmissionToken:
        """Create a progress request; only measured pressure gates outer work."""
        held = held_context()
        return AdmissionToken(
            self,
            label,
            depth_below(held) if depth is None else depth,
            queued=queued,
            parent=(held[0] if held else 0) if parent is None else parent,
            outer=outer,
        )

    def presence(
        self, label: str, depth: int, parent: int | None = None, *, queued: bool = False, outer: bool = False
    ) -> AdmissionToken:
        token = self.request(label=label, depth=depth, parent=parent, queued=queued, outer=outer)
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

    def suspend(self, token: int) -> None:
        """Keep an expanding parent's admission while its children execute."""
        self._set_states([token], SUSPENDED)

    def queue(self, token: int) -> None:
        """Make a suspended parent's admission eligible for its gather."""
        self._set_states([token], QUEUED)

    def rest(self, token: int) -> None:
        """A token now holds resident memory only (retained state between calls), not running work."""
        if token:
            self._set_states([token], RESIDENT)

    @contextmanager
    def suspended(self):
        """The calling thread waits for its own sub-tasks during the block: its admission tokens count as held by a
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

    def note(self, label: str, measured: int | None) -> None:
        index = self._label_index(label)
        with self._ledger.locked():
            self._record(index, measured)

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
            used = max(used, int(h[_LIMIT]) - available)
            h[_LIVE_BYTES] = used
            h[_AVAILABLE_BYTES] = available
            h[_PEAK_USED] = max(int(h[_PEAK_USED]), used)
            for label in np.unique(ledger.t_label[ledger.t_id != 0]):
                ledger.l["live_peak"][label] = max(int(ledger.l["live_peak"][label]), used)
            h[_SAMPLE_NS] = time.monotonic_ns()

    def _label_index(self, label: str) -> int:
        index = self._labels.get(label)
        if index is not None:
            return index
        name = label.encode("utf-8", "replace")[: _LABEL["name"].itemsize]
        ledger = self._ledger
        with ledger.locked():
            names = ledger.l["name"]
            found = np.flatnonzero(names == name)
            if found.size:
                index = int(found[0])
            else:
                free = np.flatnonzero(names == b"")
                if not free.size:
                    index = len(names)
                    ledger.grow("labels")
                else:
                    index = int(free[0])
                ledger.l["name"][index] = name
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

    def _admit(self, request: AdmissionToken, fresh: bool) -> bool:
        """Only sampled working-set pressure paces outer starts; nested work continues."""
        self._sample_pressure()
        if fresh:
            self._purge()
        ledger = self._ledger
        if request._label < 0:
            request._label = self._label_index(request.label)
        now = time.monotonic_ns()
        forced = False
        with ledger.locked() as h:
            limit, used = (int(h[_LIMIT]), int(h[_LIVE_BYTES]))
            soft = int(MEMORY_SOFT_FRACTION * limit)
            # Unrestricted below real pressure, then continuously increase spacing
            # between outer starts as measured headroom vanishes. No planned peaks.
            pressure = max(0.0, (used - soft) / max(1, limit - soft))
            interval = int(_SAMPLE_NS_INTERVAL * pressure / max(0.001, 1.0 - pressure))
            fits = not request.outer or (used < limit and (used <= soft or now - int(h[_OUTER_START_NS]) >= interval))
            if fits and (not request.outer or used < soft or (not h[_WAITING])):
                admit = True
                since = int(ledger.w_since[request._waiter]) if request._waiter >= 0 else now
            else:
                if request._waiter < 0:
                    slot = int(ledger.w_id.argmin(axis=0))
                    if ledger.w_id[slot]:
                        slot = len(ledger.w_id)
                        ledger.grow("waiters")
                    request.id = int(h[_NEXT_ID]) + 1
                    h[_NEXT_ID] = request.id
                    pid, started = _identity()
                    (
                        ledger.w_id[slot],
                        ledger.w_pid[slot],
                        ledger.w_started[slot],
                        ledger.w_since[slot],
                        ledger.w_depth[slot],
                        ledger.w_label[slot],
                    ) = (request.id, pid, started, now, request.depth, request._label)
                    request._waiter = slot
                    h[_WAITING] += 1
                ledger.w_polled[request._waiter] = now
                # Only recent contenders can hold the head: a graph request whose
                # affinity/CPU seat is no longer available must not deadlock peers.
                queued = np.flatnonzero((ledger.w_id != 0) & (ledger.w_polled >= now - int(2 * POLL_SECONDS * 1e9)))
                head = int(queued[np.lexsort((ledger.w_id[queued], -ledger.w_depth[queued]))[0]])
                is_head = head == request._waiter
                admit = fits and (not request.outer or is_head)
                if not admit and is_head:
                    active = ledger.t_id != 0
                    if not (ledger.t_state[active] == RUNNING).any() and (not (ledger.t_state[active] == QUEUED).any()):
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
                slot = int(ledger.t_id.argmin(axis=0))
                if ledger.t_id[slot]:
                    slot = len(ledger.t_id)
                    ledger.grow("tokens")
                pid, started = _identity()
                (
                    ledger.t_id[slot],
                    ledger.t_pid[slot],
                    ledger.t_started[slot],
                    ledger.t_granted[slot],
                    ledger.t_depth[slot],
                    ledger.t_state[slot],
                    ledger.t_label[slot],
                    ledger.t_forced[slot],
                    ledger.t_parent[slot],
                    ledger.t_outer[slot],
                ) = (
                    request.id,
                    pid,
                    started,
                    now,
                    request.depth,
                    QUEUED if request.queued else RUNNING,
                    request._label,
                    int(forced),
                    request.parent,
                    int(request.outer),
                )
                self._slots[request.id] = request._slot = slot
                request.granted = True
                if request.outer:
                    h[_OUTER_START_NS] = time.monotonic_ns()
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
                _LOG.info(f"[memory] progress {request.label}: no running task; live {_gib(used)} of {_gib(limit)}")
            return True
        if not request._logged:
            request._logged = True
            self._log_wait(request, limit, used, int(h[_WAITING]))
        return False

    def recovery_requested(self) -> bool:
        """Consume a watchdog recovery request on the executor's own thread."""
        requested = int(self._ledger.h[_RECOVERY_REQUEST_NS])
        changed = requested > self._recovery_seen
        self._recovery_seen = requested
        return changed

    def under_pressure(self) -> bool:
        """Idle caches may be released near the measured limit, never from estimates."""
        return int(self._ledger.h[_LIVE_BYTES]) > MEMORY_SOFT_FRACTION * self.limit

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
        _LOG.info(
            f"[memory] pacing {request.label}: live {_gib(live)} of {_gib(limit)}, "
            f"{('hard pressure' if live >= MEMORY_HARD_FRACTION * limit else 'spacing outer starts')}; "
            f"{waiting} waiting"
        )

    def _release(self, request: AdmissionToken, measured, record: bool) -> None:
        ledger = self._ledger
        with ledger.locked() as h:
            if request._waiter >= 0:
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
                return
            ledger.t_id[slot] = 0
            h[_RELEASED] += 1
            if record:
                self._record(request._label, measured)

    def _record(self, index: int, measured: int | None) -> None:
        labels = self._ledger.l
        labels["tasks"][index] += 1
        if measured is not None:
            labels["measured"][index] += 1
            labels["measured_sum"][index] += measured
            labels["measured_max"][index] = max(int(labels["measured_max"][index]), measured)

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
        """Release admission tokens and withdraw requests of processes that no longer exist."""
        now = time.monotonic()
        if now - self._purged_at < PURGE_SECONDS:
            return
        self._purged_at = now
        ledger = self._ledger
        holders: set[tuple[int, int]] = set()
        with ledger.locked() as h:
            if time.monotonic_ns() - int(h[_PURGE_NS]) < int(PURGE_SECONDS * 1000000000.0):
                return
            h[_PURGE_NS] = time.monotonic_ns()
            for ids, pids, starts in (
                (ledger.t_id, ledger.t_pid, ledger.t_started),
                (ledger.w_id, ledger.w_pid, ledger.w_started),
            ):
                used = np.flatnonzero(ids)
                holders.update(zip(pids[used].tolist(), starts[used].tolist(), strict=False))
        dead = {holder for holder in holders if not process_alive(*holder)}
        if not dead:
            return
        released = []
        with ledger.locked() as h:
            for slot in np.flatnonzero(ledger.t_id):
                if (int(ledger.t_pid[slot]), int(ledger.t_started[slot])) in dead:
                    released.append((int(ledger.t_pid[slot]), self._label_name(int(ledger.t_label[slot]))))
                    ledger.t_id[slot] = 0
            for slot in np.flatnonzero(ledger.w_id):
                if (int(ledger.w_pid[slot]), int(ledger.w_started[slot])) in dead:
                    ledger.w_id[slot] = 0
                    h[_WAITING] -= 1
            h[_PURGED] += len(released)
            h[_RELEASED] += 1
        for pid, label in released:
            _LOG.info(f"[memory] released {label} held by exited process {pid}")

    def _label_name(self, index: int) -> str:
        return bytes(self._ledger.l["name"][index]).decode("utf-8", "replace")

    def statistics(self) -> dict:
        ledger = self._ledger
        with ledger.locked() as h:
            header = {name: int(h[i]) for i, name in enumerate(_FIELDS)}
            rows = ledger.labels[ledger.l["name"] != b""].copy()
        labels = {}
        for row in rows:
            measured = int(row["measured"])
            labels[bytes(row["name"]).decode("utf-8", "replace")] = dict(
                tasks=int(row["tasks"]),
                measured_tasks=measured,
                measured_mean_bytes=float(row["measured_sum"]) / measured if measured else None,
                measured_max_bytes=int(row["measured_max"]) if measured else None,
                waits=int(row["waits"]),
                wait_seconds=int(row["wait_ns"]) / 1000000000.0,
                forced=int(row["forced"]),
                peak_live_use_bytes=int(row["live_peak"]),
            )
        return dict(
            limit_bytes=header["limit"],
            configured_bytes=None if header["configured"] < 0 else header["configured"],
            available_at_start_bytes=None if header["start_available"] < 0 else header["start_available"],
            peak_live_use_bytes=header["peak_used"],
            live_bytes=header["live_bytes"],
            sample_ns=header["sample_ns"],
            outer_start_ns=header["outer_start_ns"],
            starts=header["starts"],
            waiting=header["waiting"],
            admissions=header["requests"],
            waits=header["waits"],
            wait_seconds=header["wait_ns"] / 1000000000.0,
            forced=header["forced"],
            purged=header["purged"],
            labels=labels,
            soft_fraction=MEMORY_SOFT_FRACTION,
            hard_fraction=MEMORY_HARD_FRACTION,
        )

    def pressure_sample(self) -> tuple[int, int]:
        """Observe live and limit bytes without sampling or acquiring the ledger lock."""
        return int(self._ledger.h[_LIVE_BYTES]), int(self._ledger.h[_LIMIT])

    def summary_lines(self) -> list[str]:
        data = self.statistics()
        return [
            f"[memory] {self.describe()}; {data['admissions']} admissions, "
            f"{data['waits']} waits, {data['forced']} forced grants"
        ]


_GOVERNOR: MemoryGovernor | None = None
_GOVERNOR_LOCK = threading.Lock()


_TASK_OWNERS: dict[int, int] = {}
_TASK_OWNERS_LOCK = threading.RLock()


def tasks_executing() -> bool:
    """Whether any thread still executes a graph task or its nested work."""
    with _TASK_OWNERS_LOCK:
        return bool(_TASK_OWNERS)


@contextmanager
def executing_work():
    """Keep native descendants visible until their executing thread unwinds."""
    ident = threading.get_ident()
    with _TASK_OWNERS_LOCK:
        _TASK_OWNERS[ident] = _TASK_OWNERS.get(ident, 0) + 1
    try:
        yield
    finally:
        with _TASK_OWNERS_LOCK:
            _TASK_OWNERS[ident] -= 1
            if not _TASK_OWNERS[ident]:
                del _TASK_OWNERS[ident]


def current_governor() -> MemoryGovernor | None:
    """Observe the attached governor without creating an authority."""
    return _GOVERNOR


def set_governor(governor: MemoryGovernor | None) -> None:
    """Bind the authority used by this process's task and record boundaries."""
    global _GOVERNOR
    _GOVERNOR = governor


def memory_governor() -> MemoryGovernor:
    """The run's governor: created by `engine.runtime.start` (or attached from the parent process); outside a
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
        previous, _GOVERNOR = (_GOVERNOR, governor)
    if previous is not None:
        previous.close()
    return governor


def governor_handle() -> str:
    """What pools and executors pass to their workers (creating this process tree's governor if needed)."""
    return memory_governor().handle


def attach_governor(handle: str | None) -> None:
    """Attach the one run authority; an invalid handle fails initialization."""
    global _GOVERNOR
    if handle is None:
        return
    try:
        governor = MemoryGovernor.attach(handle)
    except (OSError, RuntimeError, ValueError) as error:
        raise ResourceError(f"cannot attach the run memory authority: {error}") from error
    with _GOVERNOR_LOCK:
        _GOVERNOR = governor


def segment_name() -> str:
    """A new name for a shared-memory segment of this run (SEGMENT_PREFIX, named after the run's driver; outside a
    run, after this process)."""
    governor = _GOVERNOR
    pid, started = governor.owner if governor is not None else _identity()
    return f"{SEGMENT_PREFIX}{os.getuid()}-{pid}-{started}-{uuid.uuid4().hex[:12]}"


def memory_statistics() -> dict | None:
    """The run's pacing statistics, or None if this process has no governor."""
    return None if _GOVERNOR is None else _GOVERNOR.statistics()


def process_tree(root: int) -> dict:
    """Read this run's process tree, including children created by non-main threads.

    Keys include Linux start ticks, so PID reuse cannot confuse CPU differences or shutdown.
    An unreadable live process invalidates the sample instead of implying zero CPU.
    """
    pending, found, seen = ([root], {}, set())
    ticks = os.sysconf("SC_CLK_TCK")
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
            fields = text.rpartition(")")[2].split()
            row = dict(
                pid=pid,
                started=int(fields[19]),
                parent=int(fields[1]),
                name=text.partition("(")[2].rpartition(")")[0],
                state=fields[0],
                cpu_seconds=(int(fields[11]) + int(fields[12])) / ticks,
            )
            try:
                io = dict(line.split(": ", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())
                row["io_bytes"] = int(io["read_bytes"]) + int(io["write_bytes"])
            except (OSError, ValueError, KeyError):
                row["io_bytes"] = 0
            found[pid, row["started"]] = row
            for task in Path(f"/proc/{pid}/task").iterdir():
                try:
                    pending.extend(map(int, (task / "children").read_text().split()))
                except FileNotFoundError:
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


class LedgerWatchdog:
    """One daemon in the ledger owner; attachers never create a monitor.

    The optional shorter intervals are for bounded runtime checks. No scientific option,
    environment variable, worker, or additional CPU budget is introduced.
    """

    def __init__(self, governor, *, stall_seconds=None, poll_seconds=None, diagnostic_directory=None):
        self.governor = governor
        self.pid = os.getpid()
        self.stall_seconds = 60 * MEMORY_STALL_MINUTES if stall_seconds is None else stall_seconds
        self.poll_seconds = MEMORY_STALL_POLL_SECONDS if poll_seconds is None else poll_seconds
        self.diagnostic_directory = (
            Path(diagnostic_directory) if diagnostic_directory is not None else Path(governor.handle).parent
        )
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="memory-ledger-watchdog", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if os.getpid() == self.pid and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1.0)

    def _run(self):
        progress_indices = tuple(_FIELDS.index(name) for name in ("requests", "released", "starts") if name in _FIELDS)
        baseline = None
        while not self.stop.wait(self.poll_seconds):
            h = self.governor._ledger.h
            counters = tuple(int(h[i]) for i in progress_indices)
            waiting = int(h[_WAITING])
            if waiting <= 0:
                baseline = None
                continue
            try:
                processes = process_tree(self.pid)
            except (OSError, ValueError, IndexError):
                baseline = None
                continue
            now = time.monotonic()
            if not processes or baseline is None or counters != baseline[0] or (processes.keys() != baseline[2].keys()):
                baseline = (counters, now, processes, False)
                continue
            elapsed = now - baseline[1]
            if elapsed < self.stall_seconds:
                continue
            cpu = []
            for key, row in processes.items():
                seconds = max(0.0, row["cpu_seconds"] - baseline[2][key]["cpu_seconds"])
                cpu.append(dict(row, window_cpu_seconds=seconds, window_cores=seconds / elapsed))
            # The owner polls admissions even when no worker can make progress.
            # Its CPU is useful only while it is itself executing an in-process
            # task. Worker CPU still protects slow serial/fractional-core work.
            cores = sum(
                row["window_cores"]
                for key, row in zip(processes, cpu, strict=True)
                if key[0] != self.pid or tasks_executing()
            )
            io_progress = any(
                row.get("io_bytes", 0) > baseline[2][key].get("io_bytes", 0) for key, row in processes.items()
            )
            if cores >= MEMORY_STALL_CPU_CORES or io_progress:
                baseline = (counters, now, processes, False)
                continue
            if tuple(int(h[i]) for i in progress_indices) != counters or int(h[_WAITING]) <= 0 or self.stop.is_set():
                baseline = None
                continue
            if not baseline[3]:
                # Ask the executor to drain idle caches before diagnosing a hang.
                # This aligned write cannot wait on a possibly stuck ledger lock.
                h[_RECOVERY_REQUEST_NS] = time.monotonic_ns()
                baseline = (counters, now, processes, True)
                continue
            try:
                self._fail(elapsed, cores, cpu, processes)
            finally:
                os._exit(1)

    def _fail(self, elapsed, cores, cpu, processes):
        """Write the snapshot, then stop only verified descendants and the run owner.

        Do not unwind pool/ledger finalizers: they may need the very lock that stopped the run.
        Existing checkpoints are untouched. In-flight units remain incomplete and can be resumed.
        """
        import numpy as np

        ledger = self.governor._ledger
        states = {1: "RUNNING", 2: "SUSPENDED", 3: "WAITING", 4: "QUEUED", 5: "RESIDENT"}
        labels = [bytes(value).decode("utf-8", "replace") for value in ledger.l["name"]]

        def records(dtype, count, offset):
            rows = np.ndarray((count,), dtype=dtype, buffer=ledger.map, offset=offset).copy()
            result = []
            for row in rows[rows["id"] != 0]:
                entry = {name: int(row[name]) for name in dtype.names}
                entry["label_index"] = entry["label"]
                entry["label"] = labels[entry["label"]]
                if "state" in entry:
                    entry["state_code"] = entry["state"]
                    entry["state"] = states.get(entry["state"], "unknown")
                result.append(entry)
            return result

        error = (
            f"Memory ledger stalled: no grants, starts or releases for {elapsed / 60:.2f} minutes; "
            f"process tree used {cores:.3f} CPU cores. Failing the run; checkpoints are kept for resume."
        )
        report = dict(
            error=error,
            owner_pid=self.pid,
            ledger=str(ledger.path),
            snapshot="lock-free diagnostic; concurrent fields may be sampled at different instants",
            elapsed_seconds=elapsed,
            cpu_cores=cores,
            processes=cpu,
            header={name: int(ledger.h[i]) for i, name in enumerate(_FIELDS)},
            tokens=records(_TOKEN, len(ledger.t_id), ledger._layout[3]),
            waiters=records(_WAITER, len(ledger.w_id), ledger._layout[4]),
        )
        path = self.diagnostic_directory / f"memory-stall-{self.pid}-{time.time_ns()}.json"
        data = json.dumps(report, indent=2) + "\n"
        try:
            path.write_text(data)
        except OSError as exc:
            error += f" Diagnostic file unavailable: {exc}."
        try:
            os.write(2, (error + f"\nMemory-stall diagnostic: {path}\n" + data).encode())
        except OSError:
            pass
        targets = [row for key, row in processes.items() if key[0] != self.pid]
        try:
            os.write(
                2,
                (
                    "Memory-stall shutdown of verified run descendants: "
                    + json.dumps([row["pid"] for row in targets])
                    + "\n"
                ).encode(),
            )
        except OSError:
            pass
        for signum in (signal.SIGTERM, signal.SIGKILL):
            for row in reversed(targets):
                if process_alive(row["pid"], row["started"]):
                    try:
                        os.kill(row["pid"], signum)
                    except ProcessLookupError:
                        pass
            if signum == signal.SIGTERM:
                time.sleep(1.0)
        os._exit(1)


@contextmanager
def admit(label: str, *, outer: bool = False):
    """Admit measured-pressure work; no bytes are reserved or used as a gate."""
    governor = memory_governor()
    token = governor.request(label=label, outer=outer)
    token.wait()
    try:
        with adopted(token.held):
            yield token
    finally:
        token.release()
