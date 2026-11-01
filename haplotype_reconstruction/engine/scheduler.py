"""Execute dependency graphs with one pressure ledger and one bounded CPU authority.

Owns admission states, native fair shares/leases, expansion and the stall watchdog.
No science, stage names or predicted-memory admission belong here. P0 ENG-20/23/25/26;
methods: WP_E1 sections 4.1-4.6. Scheduling constants below carry unchanged values
from execution/config.py and memory.py (engine.parameters.csv), in seconds/bytes
or dimensionless fractions as named. These are operational, never science identity.
"""

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
from collections.abc import Callable, Container, Mapping, MutableSequence, Sequence
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, wait
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from ..core.errors import InvariantError, ResourceError, SettingsError
from .graph import Expansion, ResultRef, Task, TaskKey, _readonly, _references, cyclic_keys, dependency_order
from .runtime import memory_pressure

if TYPE_CHECKING:
    from .workers import WorkerGroup

_LOG = logging.getLogger(__name__)
__all__ = [
    "malloc_trim",
    "release_memory",
    "task_peak_start",
    "task_peak_bytes",
    "held_context",
    "depth_below",
    "adopted",
    "mark_in_use",
    "AdmissionToken",
    "TaskSession",
    "MemoryGovernor",
    "memory_governor",
    "start_governor",
    "governor_handle",
    "attach_governor",
    "segment_name",
    "memory_statistics",
    "numba_threads_set",
    "numba_threads",
    "CpuBudget",
    "FixedBudget",
    "as_budget",
    "PoolCounters",
    "HelperBudget",
    "PoolShareBudget",
    "enter_pool_seat",
    "pool_seat",
    "apportion",
    "floor_shares",
    "LeaseState",
    "lease_seat",
    "lease_user",
    "lease_exit",
    "LeaseBudget",
    "budget_context",
    "current_budget",
    "capped_threads",
    "authority_threads",
    "claim_threads",
    "SlotBudget",
    "process_tree",
    "LedgerWatchdog",
    "plan_lanes",
    "AdmissionState",
    "admit",
    "admission_state",
    "batch_hint",
    "planning_cost",
    "run_graph",
    "TaskBudget",
    "ChainPlan",
    "LaneAuthority",
    "partition_chains",
]

MEMORY_SOFT_FRACTION = 0.8
MEMORY_HARD_FRACTION = 0.9
MEMORY_SAMPLE_SECONDS = 0.5
MEMORY_STALL_MINUTES = 30.0
MEMORY_STALL_POLL_SECONDS = 60.0
MEMORY_STALL_CPU_CORES = 1.0
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
    [("id", "<i8"), ("pid", "<i8"), ("started", "<i8"), ("since", "<i8"), ("depth", "<i8"), ("label", "<i8")]
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
        if owner is None or _alive(*owner) or _in_use(entry.path):
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

    def __init__(self, path: Path, *, create: bool):
        self.path = Path(path)
        self.pid = os.getpid()
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        self.fd = os.open(self.path, flags, 384)
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
        (
            self.t_id,
            self.t_pid,
            self.t_started,
            self.t_granted,
            self.t_depth,
            self.t_state,
            self.t_label,
            self.t_forced,
            self.t_parent,
            self.t_outer,
        ) = (tokens[name] for name in _TOKEN.names or ())
        self.w_id, self.w_pid, self.w_started, self.w_since, self.w_depth, self.w_label = (
            waiters[name] for name in _WAITER.names or ()
        )
        self.l = {name: self.labels[name] for name in _LABEL.names or ()}
        self.thread_lock = threading.Lock()

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

    def _admit(self, request: AdmissionToken, fresh: bool) -> bool:
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
            limit, used = (int(h[_LIMIT]), int(h[_LIVE_BYTES]))
            soft, hard = (int(MEMORY_SOFT_FRACTION * limit), int(MEMORY_HARD_FRACTION * limit))
            observed = int(h[_SAMPLE_NS]) - int(h[_OUTER_START_NS]) >= _SAMPLE_NS_INTERVAL
            fits = not request.outer or (used < hard and (used < soft or observed))
            if fits and (not request.outer or used < soft or (not h[_WAITING])):
                admit = True
                since = int(ledger.w_since[request._waiter]) if request._waiter >= 0 else now
            else:
                if request._waiter < 0:
                    slot = int(ledger.w_id.argmin(axis=0))
                    if ledger.w_id[slot]:
                        raise RuntimeError("the memory ledger's request table is full")
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
                queued = np.flatnonzero(ledger.w_id)
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
                    raise RuntimeError("the memory ledger's admission token table is full")
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
            f"{('hard pressure' if live >= MEMORY_HARD_FRACTION * limit else 'awaiting next measured sample')}; "
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
        dead = {holder for holder in holders if not _alive(*holder)}
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

    def summary_lines(self) -> list[str]:
        data = self.statistics()
        return [
            f"[memory] {self.describe()}; {data['admissions']} admissions, "
            f"{data['waits']} waits, {data['forced']} forced grants"
        ]


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


CPU_POLL_SECONDS = 0.2


def _numba():
    import numba

    return numba


def _set_team(n: int) -> int:
    """Set the calling thread's Numba team to ``n``; a lane coordinator's team is recorded on its lease (CPUs it
    uses, which its pools' rebalancing counts)."""
    numba = _numba()
    n = max(1, int(n))
    lease = _LEASE_SEAT
    owner = lease is not None and lease.owner == threading.get_ident()
    if owner and lease is not None and lease.users:
        n = 1
    if numba.get_num_threads() != n:
        numba.set_num_threads(n)
    if owner and lease is not None:
        lease.team = n
    return n


def numba_threads_set(n: int) -> int:
    """Set the calling thread's Numba team to ``n`` at a boundary inside a `numba_threads` block (whose exit
    restores within the current claim); returns the team."""
    return _set_team(n)


@contextmanager
def numba_threads(n: int):
    """Run the block with a Numba team of ``n`` threads. On exit the calling thread's setting is restored, within
    its current claim when it owns one (a share given back meanwhile is not taken again by the restore)."""
    old = _numba().get_num_threads()
    _set_team(n)
    try:
        yield
    finally:
        ceiling = _team_ceiling()
        _set_team(old if ceiling is None else max(1, min(old, ceiling)))


class CpuBudget:
    """A CPU allowance (see the module docstring). Picklable: a live allowance pickles as its current value."""

    live = False

    @staticmethod
    def fixed(n: int) -> FixedBudget:
        return FixedBudget(n)

    def threads(self) -> int:
        """The current allowance (>= 1), re-read now; a lease may grow on each call."""
        raise NotImplementedError

    def phase(self, name: str) -> CpuBudget:
        """Mark the start of phase ``name`` (a boundary: a lane rebalances its lease); return the budget."""
        return self

    def current(self) -> int:
        """The allowance as it stands, without claiming anything (what a pickled copy carries)."""
        return self.threads()

    def __reduce__(self):
        return (FixedBudget, (max(1, int(self.current())),))

    def concurrent(self, functions, *, label: str = "concurrent"):
        """Run nested concurrent children with CPU-seat-before-admission ordering."""
        return _completed_tasks(self, list(functions), label)

    def thread_map(self, function, items, *, label: str = "thread map") -> list:
        """Run one-thread nested calls, placing values in input order."""
        return _one_thread_map(self, function, list(items), label)


class FixedBudget(CpuBudget):
    """A fixed count of threads."""

    def __init__(self, n: int):
        n = int(n)
        if n < 1:
            raise ValueError("a CPU budget must be at least one thread")
        self.n = n

    def threads(self) -> int:
        return self.n

    def __reduce__(self):
        return (FixedBudget, (self.n,))

    def __repr__(self):
        return f"CpuBudget.fixed({self.n})"


def as_budget(budget) -> CpuBudget:
    """A CpuBudget for a budget or a plain thread count."""
    return budget if isinstance(budget, CpuBudget) else FixedBudget(budget)


def _read(value):
    """Lock-free read of a shared counter (an aligned int read is atomic; a slightly stale value is fine)."""
    return value.get_obj().value if hasattr(value, "get_obj") else value.value


class PoolCounters:
    """The shared state of one pool's thread shares (created by the pool's parent, passed to its workers)."""

    lease_state: LeaseState
    lease_index: int
    retirements: MutableSequence[int]
    retire_chunks: int

    def __init__(self, context, total: int, processes: int):
        value = context.Value
        self.total = value("i", int(total))
        self.active = value("i", 0)
        self.claimed = value("i", 0)
        self.admission = context.Condition(self.claimed.get_lock())
        self.waiting = value("i", 0, lock=False)
        self.short = value("i", 0, lock=False)
        self.extra = value("i", 0)
        self.started = value("i", 0)
        self.participants = value("i", 0)
        self.generation = value("i", 0)
        self.task_count = value("i", 0)
        self.startup_target = value("i", min(1, processes))
        self.startup_ready = value("i", 0)
        self.closed = value("i", 0)
        self.reserved = value("i", 0, lock=False)
        self.helper_claimed = value("i", 0, lock=False)

    @staticmethod
    def _store(counter, number):
        # Aligned assignments are atomic. The coordinator is the sole resetting
        # writer; workers only open startup_ready (write 1) within a batch.
        # Never acquire a semaphore a dead worker could have left locked.
        counter.get_obj().value = int(number)

    def start_batch(self, tasks: int, processes: int) -> None:
        """Open a new batch (generation): reset the start-up gate for ``tasks`` tasks."""
        if _read(self.active):
            raise RuntimeError("pool workers of the preceding batch are still active")
        target = min(int(tasks), int(processes))
        self._store(self.started, 0)
        self._store(self.participants, 0)
        self._store(self.task_count, tasks)
        self._store(self.startup_target, target)
        self._store(self.startup_ready, target <= 1)
        self.generation.get_obj().value += 1

    def set_total(self, threads: int) -> None:
        """Follow the live parent budget both ways. A raise wakes workers waiting for a CPU; a cut applies at the
        running tasks' own boundaries (share() then returns only CPUs within the new total; claims are never
        revoked inside a kernel) and holds back new task starts until the claims fit."""
        threads = max(1, int(threads))
        if not self.admission.acquire(timeout=CPU_POLL_SECONDS):
            return
        try:
            old = _read(self.total)
            if threads != old:
                self._store(self.total, threads)
                if threads > old:
                    self.admission.notify_all()
        finally:
            self.admission.release()

    def open_gate(self) -> None:
        """Open the current batch's start-up gate now (memory holds the remaining tasks back)."""
        self._store(self.startup_ready, 1)

    def close(self) -> None:
        """Publish cancellation without waiting on a possibly dead CPU-lock holder."""
        self.closed.get_obj().value = 1
        if self.admission.acquire(False):
            try:
                self.admission.notify_all()
            finally:
                self.admission.release()

    def take_one(self, cancel=None) -> bool:
        """Take a consumer CPU while polling death, even if a worker owns the lock.

        Condition.wait reacquires its semaphore without a timeout. Coordinator
        waits therefore release and poll explicitly; no worker can strand them.
        """
        registered = False
        while True:
            if _read(self.closed) or (cancel is not None and cancel.is_set()):
                return False
            if not self.admission.acquire(timeout=CPU_POLL_SECONDS):
                continue
            try:
                if not registered:
                    self.waiting.value += 1
                    registered = True
                if (
                    _read(self.claimed) < _read(self.total)
                    and _read(self.claimed) - self.helper_claimed.value < _read(self.total) - self.reserved.value
                ):
                    self.waiting.value -= 1
                    self.claimed.get_obj().value += 1
                    return True
            finally:
                self.admission.release()
            time.sleep(CPU_POLL_SECONDS)

    def give_one(self, cancel=None) -> bool:
        """Return a consumer CPU without waiting on a dead worker's semaphore."""
        while not self.admission.acquire(timeout=CPU_POLL_SECONDS):
            if _read(self.closed) or (cancel is not None and cancel.is_set()):
                return False
        try:
            self.claimed.get_obj().value -= 1
            self.admission.notify_all()
            return True
        finally:
            self.admission.release()


class _PoolSeat:
    """A pool worker's view of its pool's shares (one per worker process)."""

    def __init__(self, counters: PoolCounters):
        self.counters = counters
        self.has_extra = False
        self.claimed = 0
        self.short = 0
        self.task_thread: int | None = None
        self.users = 0
        self.lock = threading.Lock()
        self.last_generation: int | None = None
        self.last_applied: int | None = None
        self.last_logged: int | None = None

    def begin_task(self, minimum: int = 1) -> None:
        """Reserve one CPU before admission (waiting meanwhile counts as a contender, so peers above their fair
        share give back at their boundaries), then register the task with the start-up gate."""
        c = self.counters
        with c.admission:
            c.waiting.value += 1
            try:
                while (
                    _read(c.claimed) + minimum > _read(c.total)
                    or _read(c.claimed) - c.helper_claimed.value + minimum > _read(c.total) - c.reserved.value
                ):
                    c.admission.wait()
            finally:
                c.waiting.value -= 1
            c.claimed.get_obj().value += minimum
            self.claimed = minimum
            self.task_thread = threading.get_ident()
            with c.active.get_lock():
                c.active.get_obj().value += 1
        with c.started.get_lock():
            c.started.get_obj().value += 1
            started = c.started.get_obj().value
        generation = _read(c.generation)
        if self.last_generation != generation:
            with c.participants.get_lock():
                c.participants.get_obj().value += 1
                participants = c.participants.get_obj().value
            self.last_generation = generation
        else:
            participants = _read(c.participants)
        if participants >= _read(c.startup_target) or started >= _read(c.task_count):
            with c.startup_ready.get_lock():
                c.startup_ready.get_obj().value = 1

    def end_task(self) -> None:
        """Release this finished task's claims and wake workers waiting for admission."""
        c = self.counters
        with c.admission:
            self._release_extra()
            c.claimed.get_obj().value -= self.claimed
            c.short.value -= self.short
            self.claimed = self.short = 0
            self.task_thread = None
            self.users = 0
            with c.active.get_lock():
                c.active.get_obj().value -= 1
            c.admission.notify_all()

    def enter_user(self) -> None:
        with self.lock:
            self.users += 1

    def exit_user(self) -> None:
        with self.lock:
            self.users = max(0, self.users - 1)

    def _total(self) -> int:
        """The CPUs this seat's tasks share: the pool's total minus what running helper seats hold back."""
        c = self.counters
        return max(1, _read(c.total) - c.reserved.value)

    def owns_team(self) -> bool:
        """Whether the calling thread may return CPUs: the task's own thread, with no map or pool of it open."""
        return self.task_thread == threading.get_ident() and self.users == 0

    def _claim_extra(self, remainder: int) -> None:
        if self.has_extra:
            return
        extra = self.counters.extra
        with extra.get_lock():
            if extra.get_obj().value < remainder:
                extra.get_obj().value += 1
                self.has_extra = True

    def _release_extra(self) -> None:
        if not self.has_extra:
            return
        extra = self.counters.extra
        with extra.get_lock():
            extra.get_obj().value -= 1
        self.has_extra = False

    def share(self, max_threads: int | None = None, *, give_back: bool = True) -> int:
        """This task's CPUs from now to its next boundary (claims change only at boundaries and at task start and
        end, never inside a running kernel; the pool's claims never exceed its total).

        Fair share: floor(total / contenders), plus one of the remainder CPUs while this task holds a remainder
        claim; contenders are the running tasks and the workers waiting for their first CPU. ``max_threads``
        (what the task can use) caps it. A task grows only up to its share (as peers finish, the share grows);
        a task above it (it grew while it had fewer peers) returns the excess here (``give_back``; False for
        queries from threads that do not own the task's team). Before the batch's start-up gate opens every task
        runs one thread.
        """
        if max_threads is not None:
            max_threads = int(max_threads)
            if max_threads < 1:
                raise ValueError("max_threads must be a positive integer or None")
        c = self.counters
        with c.admission:
            if hasattr(c, "lease_state") and give_back:
                lease = _LeaseSeat(c.lease_state, c.lease_index)
                _, total = lease.grow(floor=_read(c.claimed))
                c.set_total(total)
            if not self.claimed:
                return 1
            total = self._total()
            if not _read(c.startup_ready):
                self._release_extra()
                want = target = 1
            else:
                contenders = max(_read(c.active), 1) + c.waiting.value
                floor, remainder = divmod(total, contenders)
                if max_threads is not None and max_threads <= floor:
                    self._release_extra()
                else:
                    extras = _read(c.extra)
                    if not self.has_extra and extras < remainder:
                        self._claim_extra(remainder)
                    elif self.has_extra and extras > remainder:
                        self._release_extra()
                want = max(1, floor + int(self.has_extra))
                if max_threads is not None:
                    want = min(want, max_threads)
                target = want
            if not give_back:
                target = max(target, self.claimed)
            others = _read(c.claimed) - self.claimed
            available = min(_read(c.total) - others, self._total() - (others - c.helper_claimed.value))
            allocated = max(1, min(target, available)) if give_back else max(self.claimed, min(target, available))
            change = allocated - self.claimed
            c.claimed.get_obj().value += change
            self.claimed = allocated
            short = max(0, want - allocated)
            c.short.value += short - self.short
            self.short = short
            if change < 0:
                c.admission.notify_all()
            return allocated

    def apply(self, max_threads: int | None = None) -> int:
        """Recompute the share at this boundary and apply it to the calling thread's Numba team; return it."""
        numba = _numba()
        n = min(self.share(max_threads, give_back=self.owns_team()), numba.config.NUMBA_NUM_THREADS)
        current = int(numba.get_num_threads())
        if self.last_applied != n or current != n:
            numba.set_num_threads(n)
            current = int(numba.get_num_threads())
        self.last_applied = current
        self._trace(current)
        return current

    def _trace(self, n: int) -> None:
        from .runtime import profiling

        if not profiling() or n == self.last_logged:
            return
        self.last_logged = n
        c = self.counters
        _LOG.info(
            f"[threads pid={os.getpid()} t={time.monotonic():.1f}] active={_read(c.active)} "
            f"total={_read(c.total)} extras={_read(c.extra)} mine={self.has_extra} -> {n}"
        )


class _HelperSeat(_PoolSeat):
    """A thread of the pool's parent process working beside the pool's tasks on their critical path (discovery's
    evidence helper, which prepares the next contig's blocks): while it runs, ``slice`` of the pool's CPUs are set
    aside for it (the tasks share the rest and give the excess back at their boundaries), and it grows into the
    slice as CPUs free; outside the batch's start-up gate."""

    def __init__(self, counters, slice_cpus: int, check_alive=None):
        super().__init__(counters)
        self.slice = max(1, int(slice_cpus))
        self.check_alive = check_alive

    @contextmanager
    def _locked(self):
        c = self.counters
        while not c.admission.acquire(timeout=CPU_POLL_SECONDS):
            if self.check_alive is not None:
                self.check_alive()
            if _read(c.closed):
                raise CancelledError("pool closed while its helper awaited a CPU lock")
        try:
            if self.check_alive is not None:
                self.check_alive()
            yield
        finally:
            c.admission.release()

    def share(self, max_threads=None, *, give_back: bool = True) -> int:
        """Up to the helper's slice (capped by ``max_threads``), from free CPUs of the whole total; it returns
        CPUs only at its own boundary (``give_back``), never below one."""
        c = self.counters
        with self._locked():
            target = self.slice if max_threads is None else max(1, min(self.slice, int(max_threads)))
            available = _read(c.total) - (_read(c.claimed) - self.claimed)
            wanted = max(1, min(target, available))
            allocated = wanted if give_back else max(self.claimed, wanted)
            change = allocated - self.claimed
            c.claimed.get_obj().value += change
            c.helper_claimed.value += change
            if change < 0:
                c.admission.notify_all()
            self.claimed = allocated
            return allocated

    def end_task(self) -> None:
        """Return the helper's CPUs and its admission token (it was never a task contender)."""
        c = self.counters
        with self._locked():
            c.claimed.get_obj().value -= self.claimed
            c.helper_claimed.value -= self.claimed
            c.reserved.value -= self.slice
            self.claimed = 0
            self.task_thread = None
            self.users = 0
            c.admission.notify_all()

    def begin(self) -> None:
        """Set the slice aside and wait for a first CPU of the whole total; CancelledError if the pool closes
        first. Helpers are not task contenders (tasks share total - reserved among themselves)."""
        c = self.counters
        with self._locked():
            c.reserved.value += self.slice
            c.admission.notify_all()
        while True:
            with self._locked():
                if _read(c.closed):
                    c.reserved.value -= self.slice
                    raise CancelledError("the pool closed before its helper got a CPU")
                if _read(c.claimed) < _read(c.total):
                    c.claimed.get_obj().value += 1
                    c.helper_claimed.value += 1
                    self.claimed = 1
                    self.task_thread = threading.get_ident()
                    return
            time.sleep(CPU_POLL_SECONDS)


class HelperBudget(CpuBudget):
    """The live share of a helper seat (see BudgetedPool.helper_seat)."""

    live = True

    def __init__(self, seat: _HelperSeat):
        self.seat = seat

    def threads(self) -> int:
        seat = self.seat
        if not seat.owns_team():
            return seat.share(give_back=False)
        n = seat.share()
        numba = _numba()
        if numba.get_num_threads() > n:
            numba.set_num_threads(n)
        return n

    def current(self) -> int:
        return max(1, int(self.seat.claimed))

    def __repr__(self):
        return "HelperBudget()"


class PoolShareBudget(CpuBudget):
    """The calling pool worker's live share (see _PoolSeat.share)."""

    live = True

    def threads(self) -> int:
        seat = _POOL_SEAT
        if seat is None:
            return 1
        if not seat.owns_team():
            return seat.share(give_back=False)
        n = seat.share()
        numba = _numba()
        if numba.get_num_threads() > n:
            numba.set_num_threads(n)
        return n

    def current(self) -> int:
        seat = _POOL_SEAT
        return 1 if seat is None or seat.last_applied is None else seat.last_applied

    def __repr__(self):
        return "PoolShareBudget()"


_POOL_SEAT: _PoolSeat | None = None


def enter_pool_seat(counters: PoolCounters) -> None:
    """In a pool worker (initializer): take a seat in the pool's shares."""
    global _POOL_SEAT
    _POOL_SEAT = _PoolSeat(counters)


def pool_seat() -> _PoolSeat | None:
    return _POOL_SEAT


def apportion(budget: int, work) -> list[int]:
    """Split ``budget`` CPUs over workers in proportion to their work.

    Each worker keeps one CPU; the others follow the largest remainders of the proportional shares (ties to the
    lower index). Equal shares if no work is known. Requires at least one CPU per worker.
    """
    work = list(work)
    total = float(sum(work))
    if total <= 0.0:
        work, total = ([1.0] * len(work), float(len(work)))
    spare = budget - len(work)
    shares = [spare * value / total for value in work]
    result = [1 + int(share) for share in shares]
    order = sorted(range(len(work)), key=lambda i: (int(shares[i]) - shares[i], i))
    for i in order[: budget - sum(result)]:
        result[i] += 1
    return result


def floor_shares(budget: int, weights, total: float) -> list[int]:
    """Equal shares of ``budget`` CPUs among workers, with a proportional floor for stragglers.

    A worker's entitlement is one CPU plus its weight's fraction of the other CPUs, the fraction taken of
    ``total``: the work not yet finished (the running workers' tasks and the tasks not yet started). A worker whose
    entitlement exceeds its equal share keeps the entitlement; the others split the rest equally (water-filling,
    remainder to the lower indices). With no weights, or weights small against the work still to start, this is
    the equal split. Each share is at least one CPU and the shares sum to at most ``budget``.
    """
    weights = [max(0.0, float(value)) for value in weights]
    n = len(weights)
    total = max(float(total), float(sum(weights)))
    if n == 0 or total <= 0.0:
        return apportion(budget, [1.0] * n)
    spare = budget - n
    entitled = [1 + int(spare * value / total) for value in weights]
    fixed = [False] * n
    while True:
        rest = [i for i in range(n) if not fixed[i]]
        left = budget - sum(entitled[i] for i in range(n) if fixed[i])
        equal = apportion(left, [1.0] * len(rest)) if rest else []
        grow = [i for i, share in zip(rest, equal, strict=False) if entitled[i] > share]
        if not grow:
            break
        for i in grow:
            fixed[i] = True
    shares = list(entitled)
    for i, share in zip(rest, equal, strict=False):
        shares[i] = share
    return shares


class LeaseState:
    """Shared lease state of one executor: per worker its claim and whether it lends (while it waits for memory
    it wants one CPU, not its share), one lock, the budget.

    ``task_weights`` (one nonnegative work estimate per task, or None) turn on the straggler floor of the shares
    (floor_shares): a worker's floor weight is its running task's weight when that is at least ``straggler_ratio``
    times the median task weight (else zero: it takes the equal split), and ``remaining`` is the weight of the tasks
    not yet finished."""

    def __init__(self, context, workers: int, budget: int, task_weights=None, straggler_ratio: float = 0.0):
        self.claims = context.Array("i", workers, lock=False)
        self.lent = context.Array("i", workers, lock=False)
        self.lock = context.Lock()
        self.budget = int(budget)
        self.proportional = task_weights is not None
        if self.proportional:
            values = [max(0.0, float(value)) for value in task_weights]
            positive = sorted(value for value in values if value > 0.0)
            median = positive[len(positive) // 2] if positive else 0.0
            self.threshold = float(straggler_ratio) * median
            self.task_weights = context.Array("d", values, lock=False)
            self.weights = context.Array("d", workers, lock=False)
            self.remaining = context.Value("d", float(sum(values)), lock=False)

    def begin_task(self, worker: int, task_index: int) -> None:
        """Worker ``worker`` starts task ``task_index``: its weight is the task's (straggler floor only)."""
        if self.proportional:
            weight = self.task_weights[task_index]
            with self.lock:
                self.weights[worker] = weight if weight >= self.threshold else 0.0

    def end_task(self, worker: int, task_index: int) -> None:
        """Worker ``worker`` finished task ``task_index``: its weight leaves the work not yet finished."""
        if self.proportional:
            with self.lock:
                self.remaining.value = max(0.0, self.remaining.value - self.task_weights[task_index])
                self.weights[worker] = 0.0

    def shares(self, active) -> list[int]:
        """The shares of the ``active`` workers (equal, or with the straggler floor), under the lock."""
        if not self.proportional:
            return apportion(self.budget, [1.0] * len(active))
        return floor_shares(self.budget, [self.weights[i] for i in active], self.remaining.value)

    def start_phase(self, loads, initial=None) -> list[int]:
        """Initial claims follow initial work (or whole load); after that every running worker's share is equal."""
        with self.lock:
            shares = apportion(self.budget, loads if initial is None else initial)
            for index, share in enumerate(shares):
                self.claims[index] = share
                self.lent[index] = 0
        return shares


class _LeaseSeat:
    """An executor worker's lease during one phase.

    Its share is an equal split of the CPUs among the workers still running (initial claims follow the workers'
    loads; remaining-work estimates can be far off inside long tasks, so they never take CPUs from a lane), or,
    with task weights, the straggler floor (floor_shares: a task's whole weight against the work not yet finished,
    never an estimate of a running task's progress). At
    its boundaries a worker takes free CPUs up to that share; when no peer is below its own want it also borrows
    every idle CPU (work-conserving); when a peer is short, a worker above its share returns the excess, but only
    down to what it is using (``floor``), so no running team ever loses CPUs. A worker that lends (while it waits
    for memory) keeps one CPU and wants no more, so peers may borrow the rest until it resumes.
    """

    def __init__(self, state: LeaseState, index: int):
        self.state = state
        self.index = index
        self.owner = threading.get_ident()
        self.users = 0
        self.team = 1
        self.waiting = 0
        self.lock = threading.Lock()

    def _balance(self):
        """(this worker's share, CPUs other workers lack, free CPUs), under the state lock. Shares are equal
        among the running workers (with the straggler floor when the executor has task weights); a worker that
        lends wants one CPU."""
        state, index = (self.state, self.index)
        active = [i for i, value in enumerate(state.claims) if value > 0 or state.lent[i] == -1 or i == index]
        shares = state.shares(active)
        others_short = sum(
            (
                max(0, (1 if state.lent[i] == 1 else share) - state.claims[i])
                for i, share in zip(active, shares, strict=False)
                if i != index
            )
        )
        return (shares[active.index(index)], others_short, state.budget - sum(state.claims))

    def grow(self, *, floor: int | None = None) -> tuple[int, int]:
        """Rebalance this worker's claim at a boundary; return (before, after). ``floor``: the CPUs this worker
        is using (it may return CPUs above it); None never shrinks."""
        with self.state.lock:
            return self._grow_locked(floor)

    def settle(self, in_use: int) -> int:
        """A pool's re-read of this lease (pool total; called under the pool's admission lock, so its tasks'
        claims cannot change meanwhile): rebalance with ``in_use`` CPUs held by the pool's running tasks plus the
        coordinating thread's team; the claim stays at least that and the returned total may be lower (tasks
        shrink to it at their boundaries, and the claim follows at later re-reads)."""
        state, index = (self.state, self.index)
        coordinator = 0 if self.waiting else int(self.team)
        in_use = int(in_use) + coordinator
        with state.lock:
            before = int(state.claims[index])
            state.lent[index] = 0
            share, others_short, free = self._balance()
            target = max(1, min(before + free, before + free if others_short == 0 else share))
            state.claims[index] = max(target, min(in_use, max(before, target)), 1, before if self.users > 1 else 1)
            return max(1, target - coordinator)

    def lend(self) -> None:
        """Keep one CPU and want no more, so peers may borrow the rest."""
        with self.state.lock:
            self.state.claims[self.index] = 1
            self.state.lent[self.index] = 1

    def pause(self) -> None:
        """While this worker waits for memory: keep one CPU and want no more, so that peers claim the rest at
        their boundaries."""
        self.lend()

    def resume(self) -> int:
        """After a memory wait: grow toward this worker's share; return threads."""
        return self.grow(floor=1)[1]

    def release(self) -> None:
        with self.state.lock:
            self.state.claims[self.index] = 0
            self.state.lent[self.index] = 0
            if self.state.proportional:
                self.state.weights[self.index] = 0.0

    def owner_boundary(self) -> bool:
        """Whether the calling thread may return CPUs now: the coordinating thread with no open pool or map."""
        return threading.get_ident() == self.owner and self.users == 0

    def enter_user(self) -> None:
        """A pool or map on this lease starts: in the coordinating thread, its own team drops to one (it may not
        hold CPUs the construct's workers are given)."""
        with self.lock:
            self.users += 1
        if threading.get_ident() == self.owner:
            _set_team(1)

    def exit_user(self) -> None:
        """A pool or map on this lease ended. The claim stays until the lane's next boundary, where CPUs above its
        share go back if a peer is short: a lane does not lend its CPUs over the short serial steps between its
        parallel constructs, because peers return borrowed CPUs only at their own boundaries, and a lane waiting
        for them (2.10 candidates: the critical contig's next phase) lost more than the peers gained."""
        with self.lock:
            self.users -= 1

    def _grow_locked(self, floor=None):
        state, index = (self.state, self.index)
        before = int(state.claims[index])
        state.lent[index] = 0
        share, others_short, free = self._balance()
        target = before + free if others_short == 0 else share
        target = max(target, before if floor is None else max(1, floor))
        state.claims[index] = max(1, min(target, before + free))
        return (before, int(state.claims[index]))


_LEASE_SEAT: _LeaseSeat | None = None


@contextmanager
def lease_seat(seat: _LeaseSeat):
    """Hold ``seat`` as this process's lease for the duration of a phase."""
    global _LEASE_SEAT
    previous = _LEASE_SEAT
    _LEASE_SEAT = seat
    try:
        yield seat
    finally:
        _LEASE_SEAT = previous


class _SlotUser:
    """A pool or map nested in one concurrent in-process task (its slot gives nothing back while it runs)."""

    def __init__(self, slots, index):
        self.slots, self.index = (slots, index)

    def enter_user(self) -> None:
        with self.slots.lock:
            self.slots.users[self.index] = self.slots.users.get(self.index, 0) + 1

    def exit_user(self) -> None:
        with self.slots.lock:
            self.slots.users[self.index] = max(0, self.slots.users.get(self.index, 0) - 1)


def lease_user(budget):
    """The seat a pool or map on ``budget`` holds while it runs: this process's lease (a contig lane's), pool
    share (a pool task's) or concurrent slot; None for fixed budgets. While held, the seat's own boundaries return
    no CPUs (its children hold them)."""
    if isinstance(budget, TaskBudget):
        return lease_user(budget.budget)
    if isinstance(budget, LeaseBudget) and _LEASE_SEAT is not None:
        return _LEASE_SEAT
    if isinstance(budget, PoolShareBudget) and _POOL_SEAT is not None:
        return _POOL_SEAT
    if isinstance(budget, SlotBudget):
        return _SlotUser(budget.slots, budget.index)
    if isinstance(budget, HelperBudget):
        return budget.seat
    return None


def lease_exit(seat) -> None:
    """After a pool or map on the seat ended (all its workers and threads finished): the seat's user count drops;
    in a lane's coordinating thread its own team drops to one until its next boundary rebalances the lease. Every
    seat keeps its claim and gives back at its next boundary."""
    seat.exit_user()
    if isinstance(seat, _LeaseSeat) and seat.users == 0 and (threading.get_ident() == seat.owner):
        _set_team(1)


class LeaseBudget(CpuBudget):
    """The calling executor worker's lease (live: every threads() call rebalances it, see _LeaseSeat)."""

    live = True

    def __init__(self, phase_name: str | None = None):
        self.phase_name = phase_name

    def _rebalance(self, phase: str | None):
        seat = _LEASE_SEAT
        if seat is None:
            return None
        if seat.owner_boundary():
            before, after = seat.grow(floor=1)
            if _numba().get_num_threads() > after:
                _set_team(after)
        else:
            before, after = seat.grow()
        return (before, after)

    def threads(self) -> int:
        result = self._rebalance(self.phase_name)
        if result is None:
            return int(_numba().get_num_threads())
        before, after = result
        if after != before and self.phase_name is not None:
            _LOG.info(f"[lease] {self.phase_name}: {before} -> {after} threads")
        return after

    def codec_threads(self) -> int:
        """Compression threads now: the lease at the coordinating thread's own boundary; one while the lane's
        pools or maps hold its CPUs (or from another thread)."""
        seat = _LEASE_SEAT
        if seat is not None and (not seat.owner_boundary()):
            return 1
        return self.threads()

    def settle(self, in_use: int) -> int:
        """A pool's live total on this lease (see _LeaseSeat.settle)."""
        seat = _LEASE_SEAT
        return int(_numba().get_num_threads()) if seat is None else seat.settle(in_use)

    def current(self) -> int:
        seat = _LEASE_SEAT
        if seat is None:
            return int(_numba().get_num_threads())
        return int(seat.state.claims[seat.index])

    def phase(self, name: str) -> LeaseBudget:
        result = self._rebalance(name)
        if result is not None and result[0] != result[1]:
            _LOG.info(f"[lease] {name}: {result[0]} -> {result[1]} threads")
        return LeaseBudget(name)

    def __repr__(self):
        return f"LeaseBudget({self.phase_name!r})"


def _team_ceiling() -> int | None:
    """The calling thread's current claim when it owns one: a pool task's thread, or a lane's coordinating
    thread (its lease)."""
    local = getattr(_THREAD_BUDGET, "value", None)
    if isinstance(local, SlotBudget):
        with local.slots.lock:
            if local.slots.threads.get(local.index) == threading.get_ident():
                return max(1, int(local.slots.claims.get(local.index, 1)))
    seat = _POOL_SEAT
    if seat is not None and seat.task_thread == threading.get_ident():
        return max(1, int(seat.claimed))
    if isinstance(local, HelperBudget) and local.seat.task_thread == threading.get_ident():
        return max(1, int(local.seat.claimed))
    lease = _LEASE_SEAT
    if lease is not None and lease.owner == threading.get_ident():
        return max(1, int(lease.state.claims[lease.index]))
    return None


_THREAD_BUDGET = threading.local()
_TASK_OWNERS: dict[int, int] = {}
_TASK_OWNERS_LOCK = threading.RLock()


@contextmanager
def executing_task(owner: TaskKey | None):
    """Carry task ownership separately from its changing nested CPU allowance."""
    previous = getattr(_THREAD_BUDGET, "task_owner", None)
    _THREAD_BUDGET.task_owner = owner
    ident = threading.get_ident()
    if owner is not None:
        with _TASK_OWNERS_LOCK:
            _TASK_OWNERS[ident] = _TASK_OWNERS.get(ident, 0) + 1
    try:
        yield
    finally:
        _THREAD_BUDGET.task_owner = previous
        if owner is not None:
            with _TASK_OWNERS_LOCK:
                _TASK_OWNERS[ident] -= 1
                if not _TASK_OWNERS[ident]:
                    del _TASK_OWNERS[ident]


@contextmanager
def budget_context(budget):
    """Bind deep numerical claims to this task, including nested thread work."""
    previous = getattr(_THREAD_BUDGET, "value", None)
    _THREAD_BUDGET.value = budget
    try:
        yield
    finally:
        _THREAD_BUDGET.value = previous


def current_budget():
    """Capture the caller's live allowance before creating worker threads."""
    own = getattr(_THREAD_BUDGET, "value", None)
    if own is not None:
        return own
    if _POOL_SEAT is not None:
        return PoolShareBudget()
    if _LEASE_SEAT is not None:
        return LeaseBudget()
    return FixedBudget(int(_numba().get_num_threads()))


def capped_threads(budget, cap: int) -> int:
    """Threads for a phase that cannot use more than ``cap`` well (owner, 25 Oct: no task keeps cores it cannot
    use): at most ``cap`` from the budget; a concurrent slot or pool task owning its claim returns the excess at
    this boundary, so peers take it (the next uncapped boundary grows it back to its share). Results never
    depend on it."""
    cap = max(1, int(cap))
    if isinstance(budget, SlotBudget):
        return max(1, min(budget.claim(cap), cap))
    if isinstance(budget, PoolShareBudget) and _POOL_SEAT is not None and _POOL_SEAT.owns_team():
        return max(1, min(_POOL_SEAT.share(cap), cap))
    return max(1, min(int(budget.threads()), cap))


def authority_threads(cap: int) -> int:
    """At most ``cap`` threads under the caller's CPU authority: its task's claim, pool share or lease (as
    capped_threads); ``cap`` itself for a caller outside any authority (a standalone process owns its CPUs)."""
    if getattr(_THREAD_BUDGET, "value", None) is None and _POOL_SEAT is None and _LEASE_SEAT is None:
        return max(1, int(cap))
    return capped_threads(current_budget(), cap)


def claim_threads(*, max_threads: int | None = None) -> int:
    """Boundary call: recompute this process's share, apply it to the calling thread's Numba team, return it.

    Pool worker: its pool share; executor worker: its lease, grown toward its work share; elsewhere the current
    team. ``max_threads`` caps the result for this phase.
    """
    local = getattr(_THREAD_BUDGET, "value", None)
    if local is not None:
        n = (
            local.claim(max_threads)
            if isinstance(local, (SlotBudget, TaskBudget))
            else capped_threads(local, max_threads)
            if max_threads is not None
            else local.threads()
        )
        if max_threads is not None:
            n = min(n, max_threads)
        numba = _numba()
        n = max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(n)
        return n
    if _POOL_SEAT is not None:
        return _POOL_SEAT.apply(max_threads)
    numba = _numba()
    if _LEASE_SEAT is not None:
        threads = LeaseBudget().threads()
    else:
        threads = int(numba.get_num_threads())
    if max_threads is not None:
        threads = max(1, min(threads, int(max_threads)))
    return _set_team(threads)


class _Slots:
    """Reusable thread slots of concurrent in-process tasks under one live budget."""

    def __init__(self, workers: int, budget: CpuBudget):
        self.workers = workers
        self.budget = budget
        self.owners: dict[int, int] = {}
        self.admitted = 0
        self.claims: dict[int, int] = {}
        self.wants: dict[int, int] = {}
        self.threads: dict[int, int] = {}
        self.waiting = 0
        self.users: dict[int, int] = {}
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def admit(self, index):
        with self.lock:
            if sum(self.claims.values()) >= self.budget.threads():
                return False
            slot = next(i for i in range(self.workers) if i not in self.owners.values())
            self.owners[index] = slot
            self.claims[index] = 1
            self.admitted += 1
            return True

    def start(self, index):
        assert index in self.claims
        with self.lock:
            self.threads[index] = threading.get_ident()

    def finish(self, index):
        with self.lock:
            del self.owners[index]
            del self.claims[index]
            self.wants.pop(index, None)
            self.threads.pop(index, None)
            self.users.pop(index, None)
            self.admitted -= 1

    def allocation(self, index, max_threads=None) -> int:
        """The task's CPUs from now to its next boundary (see _PoolSeat.share for the rule): its fair share among
        the admitted tasks and those held back for a CPU (the remainder to the lowest slots), capped by
        ``max_threads``; a task above it returns the excess, but only at its own thread's boundary (other threads'
        queries and tasks with nested work never shrink a running team). Admitted tasks that have not started yet
        are counted, so a just-started task never takes their threads."""
        with self.lock:
            if self.stopped.is_set():
                raise CancelledError("another concurrent task failed")
            count = max(1, self.admitted) + self.waiting
            rank = sum(slot < self.owners[index] for slot in self.owners.values())
            cores = int(self.budget.threads())
            want = max(1, cores // count + int(rank < cores % count))
            if max_threads is not None:
                want = min(want, max_threads)
            self.wants[index] = want
            mine = self.claims[index]
            others = sum((value for owner, value in self.claims.items() if owner != index))
            target = want
            if self.threads.get(index) != threading.get_ident() or self.users.get(index, 0):
                target = max(target, mine)
            self.claims[index] = (
                max(1, min(target, cores - others)) if target < mine else max(mine, min(target, cores - others))
            )
            return self.claims[index]


class SlotBudget(CpuBudget):
    """A concurrent task's live share (see _Slots.allocation)."""

    live = True

    def __init__(self, slots: _Slots, index: int):
        self.slots = slots
        self.index = index

    def threads(self) -> int:
        n = self.slots.allocation(self.index)
        if self.slots.threads.get(self.index) == threading.get_ident() and _numba().get_num_threads() > n:
            _numba().set_num_threads(n)
        return n

    def claim(self, maximum=None) -> int:
        return self.slots.allocation(self.index, maximum)


def _threading_layer_is_workqueue() -> bool:
    numba = _numba()
    try:
        return numba.threading_layer() == "workqueue"
    except ValueError:
        return numba.config.THREADING_LAYER == "workqueue"


def _completed_tasks(budget: CpuBudget, functions, label):
    owner = getattr(_THREAD_BUDGET, "task_owner", None)
    governor = memory_governor()
    depth = depth_below(held_context())

    def reserve(index):
        return governor.request(label=label, depth=depth, queued=True)

    if _threading_layer_is_workqueue() or len(functions) <= 1 or (not budget.live and budget.threads() <= 1):
        for index, function in enumerate(functions):
            request = reserve(index)
            request.wait()
            governor.start(request.id)
            try:
                with adopted(request.held), budget_context(budget), numba_threads(budget.threads()):
                    value = function(budget)
            finally:
                request.release()
            yield (index, value)
        return
    slots = _Slots(len(functions), budget)
    lease = lease_user(budget)
    if lease is not None:
        lease.enter_user()

    def execute(index, request):
        slots.start(index)
        governor.start(request.id)
        try:
            own = SlotBudget(slots, index)
            with executing_task(owner), adopted(request.held), budget_context(own), numba_threads(own.threads()):
                return functions[index](own)
        finally:
            slots.finish(index)
            request.release()

    queued = list(range(len(functions)))
    running: dict[Future[Any], int] = {}
    try:
        yield from _drive_slots(budget, functions, slots, queued, running, reserve, execute, governor)
    finally:
        if lease is not None:
            lease_exit(lease)


def _drive_slots(budget, functions, slots, queued, running, reserve, execute, governor):
    """Start daemon threads only for admitted slots; propagate failure before cleanup.

    Python cannot interrupt another thread's native call. On failure the worker
    sends the exception immediately and its coordinator terminates the process.
    Daemon threads avoid an executor's unconditional interpreter-exit join.
    """
    pending = None

    def perform(future, index, request):
        if not future.set_running_or_notify_cancel():
            slots.finish(index)
            request.release()
            return
        try:
            future.set_result(execute(index, request))
        except BaseException as error:
            future.set_exception(error)

    try:
        while queued or running:
            limit = max(1, int(budget.threads()))
            starting = []
            slots.waiting = 0
            while queued and len(running) + len(starting) < limit:
                if not slots.admit(queued[0]):
                    slots.waiting = 1
                    break
                if pending is None:
                    pending = reserve(queued[0])
                if not pending.poll():
                    if running or starting:
                        slots.finish(queued[0])
                        break
                    pending.wait()
                starting.append((queued.pop(0), pending))
                pending = None
            for index, request in starting:
                future: Future[Any] = Future()
                running[future] = index
                threading.Thread(target=perform, args=(future, index, request), daemon=True).start()
            timeout = CPU_POLL_SECONDS if queued else None
            with governor.suspended():
                done, _ = wait(running, return_when=FIRST_COMPLETED, timeout=timeout)
            for future in done:
                yield (running.pop(future), future.result())
    except BaseException:
        slots.stopped.set()
        if pending is not None:
            pending.release()
        for future in running:
            future.cancel()
        raise


class _ThreadMapState(TypedDict):
    taken: int
    finished: int
    workers: int
    error: BaseException | None


def _one_thread_map(budget: CpuBudget, function, items, label):
    owner = getattr(_THREAD_BUDGET, "task_owner", None)
    results = [None] * len(items)
    state: _ThreadMapState = dict(taken=0, finished=0, workers=0, error=None)
    changed = threading.Condition()
    governor = memory_governor()
    caller = held_context()
    depth, parent = (depth_below(caller), caller[0] if caller else 0)

    def share():
        return max(1, int(budget.threads()))

    def call(index):
        return function(items[index])

    def work():
        presence = governor.presence(label, depth, parent=parent)
        try:
            with executing_task(owner), numba_threads(1), budget_context(FixedBudget(1)), adopted(presence.held):
                while True:
                    with changed:
                        if state["error"] is not None or state["taken"] == len(items) or state["workers"] > share():
                            state["workers"] -= 1
                            changed.notify()
                            return
                        index = state["taken"]
                        state["taken"] += 1
                    try:
                        results[index] = call(index)
                    except BaseException as error:
                        with changed:
                            if state["error"] is None:
                                state["error"] = error
                    with changed:
                        state["finished"] += 1
                        changed.notify()
        except BaseException as error:
            with changed:
                if state["error"] is None:
                    state["error"] = error
                changed.notify_all()
        finally:
            presence.release(record=False)

    started = []
    lease = lease_user(budget)
    if lease is not None:
        lease.enter_user()
    with governor.suspended(), changed:
        try:
            while state["finished"] < len(items) and state["error"] is None:
                while state["workers"] < min(share(), len(items) - state["taken"]):
                    state["workers"] += 1
                    started.append(threading.Thread(target=work, daemon=True))
                    started[-1].start()
                changed.wait(CPU_POLL_SECONDS)
        except BaseException as error:
            if state["error"] is None:
                state["error"] = error
    if state["error"] is None:
        for worker in started:
            worker.join()
    if lease is not None:
        lease_exit(lease)
    if state["error"] is not None:
        raise state["error"]
    return results


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
                baseline = (counters, now, processes)
                continue
            elapsed = now - baseline[1]
            if elapsed < self.stall_seconds:
                continue
            cpu = []
            for key, row in processes.items():
                seconds = max(0.0, row["cpu_seconds"] - baseline[2][key]["cpu_seconds"])
                cpu.append(dict(row, window_cpu_seconds=seconds, window_cores=seconds / elapsed))
            cores = sum(row["window_cores"] for row in cpu)
            if cores >= MEMORY_STALL_CPU_CORES:
                baseline = (counters, now, processes)
                continue
            if tuple(int(h[i]) for i in progress_indices) != counters or int(h[_WAITING]) <= 0 or self.stop.is_set():
                baseline = None
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
            tokens=records(_TOKEN, TOKENS, _TOKENS_AT),
            waiters=records(_WAITER, WAITERS, _WAITERS_AT),
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
                if _alive(row["pid"], row["started"]):
                    try:
                        os.kill(row["pid"], signum)
                    except ProcessLookupError:
                        pass
            if signum == signal.SIGTERM:
                time.sleep(1.0)
        os._exit(1)


CONTIG_LANE_CORES = 8
STRAGGLER_RATIO = 2.0


def plan_lanes(sizes, pending, *, cores: int, requested: int | None, available: int | None = None):
    """Concurrent lanes and longest-first work weights; live pressure paces dispatch, not the lane count."""
    count = sum(bool(flag) for flag in pending)
    limit = max(1, cores // CONTIG_LANE_CORES) if requested is None else requested
    lanes = min(count, limit, cores)
    if lanes < 2:
        return (1, None)
    weights = [float(size) if flag else 0.0 for size, flag in zip(sizes, pending, strict=False)]
    return (lanes, weights)


@dataclass(frozen=True)
class AdmissionState:
    """An immutable observation; reading state never changes an admission decision."""

    id: int
    depth: int
    parent: int
    outer: bool
    state: int


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


def admission_state(token: AdmissionToken) -> AdmissionState:
    with token.governor._ledger.locked():
        slot = token.governor._slot(token.id)
        state = 0 if slot < 0 else int(token.governor._ledger.t_state[slot])
    return AdmissionState(token.id, token.depth, token.parent, token.outer, state)


def batch_hint(label: str, *, minimum: int, maximum: int, bytes_per_item: int) -> int:
    """Cost-only hint for consumers with forced-batch equality receipts.

    This does not allocate, admit, reserve, or change a scientific work allowance.
    The 3.5 flexible-buffer arithmetic is kept separate from admission.
    """
    if minimum < 1 or maximum < minimum or bytes_per_item < 1:
        raise SettingsError(f"{label}: invalid batch dimensions")
    governor = memory_governor()
    governor._sample_pressure()
    free = int(MEMORY_SOFT_FRACTION * governor.limit) - int(governor._ledger.h[_LIVE_BYTES])
    return min(maximum, max(minimum, free // bytes_per_item))


def planning_cost(decoded_discovery_bytes: int, samples: int, sites: int) -> int:
    """ENG-23 dimension model, used for ordering only (never for admission)."""
    d, n, length = (decoded_discovery_bytes, samples, sites)
    if min(d, n, length) < 0:
        raise SettingsError("planning dimensions must be nonnegative")
    return d + 63 * n * length + 200 * (-(-d // 13) + 8 * length) + 8 * n * length + 9 * length + 32 * n


def _resolve_graph(value: Any, results: Mapping[TaskKey, Any]) -> Any:
    if isinstance(value, ResultRef):
        return results[value.task]
    if isinstance(value, tuple):
        return tuple(_resolve_graph(item, results) for item in value)
    if isinstance(value, list):
        return [_resolve_graph(item, results) for item in value]
    if is_dataclass(value) and (not isinstance(value, type)):
        if not any(_references(value)):
            return value
        return replace(
            value, **{f.name: _resolve_graph(getattr(value, f.name), results) for f in fields(value) if f.init}
        )
    if isinstance(value, Mapping):
        return {key: _resolve_graph(item, results) for key, item in value.items()}
    return value


def _resolved_task(task: Task, results: Mapping[TaskKey, Any]) -> Task:
    return replace(task, args=_resolve_graph(task.args, results), kwargs=_resolve_graph(task.kwargs, results))


def _precompleted() -> None:
    """Validation-only node representing an identity-checked published prerequisite."""


class TaskSession:
    """Keep affinity state across bounded graph batches under one engine authority.

    The caller owns each returned result mapping and may discard it before the next
    batch. The session retains only workers, admission tokens and completed chain
    keys; it never accumulates result arrays. Dependencies across batches must be
    passed explicitly as initial_results or as resident handles in task arguments.
    Closing the context releases every retained state, including after failure.
    Sessions belong at the coordinator boundary; executing tasks return Expansion.
    """

    def __init__(
        self,
        cores: int,
        *,
        max_workers: int | None = None,
        in_process: bool = False,
        contig_lanes: int | None = None,
        recycle_bytes: int | None = None,
        governor: MemoryGovernor | None = None,
        preload: Sequence[str] = (),
        chains: Sequence[ChainPlan] = (),
        dynamic: bool = True,
    ):
        self.cores, self.max_workers, self.in_process = cores, max_workers, in_process
        self.contig_lanes, self.recycle_bytes = contig_lanes, recycle_bytes
        self.preload, self.chains, self.dynamic = tuple(preload), tuple(chains), dynamic
        self.governor = governor
        self.group: WorkerGroup | None = None
        self.residents: dict[tuple[int, str], AdmissionToken] = {}
        self.chain_keys = {key for chain in chains for key in chain.tasks}
        self.completed: set[TaskKey] = set()
        self.active = self.running = self.closed = False

    def __enter__(self) -> TaskSession:
        if self.active or self.closed:
            raise SettingsError("task session cannot be reopened")
        self.active = True
        return self

    def run(
        self,
        tasks: Sequence[Task],
        *,
        initial_results: Mapping[TaskKey, Any] | None = None,
        priorities: Mapping[TaskKey, float] | None = None,
        serial_order: Sequence[str] = (),
        on_task: Callable[[Task, Any], None] | None = None,
    ) -> dict[TaskKey, Any]:
        """Run one bounded batch; no implicit results survive from earlier batches."""
        if not self.active or self.closed or self.running:
            raise SettingsError("task session must be open and idle")
        self.running = True
        try:
            return run_graph(
                tasks,
                cores=self.cores,
                max_workers=self.max_workers,
                in_process=self.in_process,
                contig_lanes=self.contig_lanes,
                recycle_bytes=self.recycle_bytes,
                preload=self.preload,
                chains=self.chains,
                dynamic=self.dynamic,
                governor=self.governor,
                initial_results=initial_results,
                priorities=priorities,
                serial_order=serial_order,
                on_task=on_task,
                _session=self,
            )
        except BaseException:
            self.close(abort=True)
            raise
        finally:
            self.running = False

    def close(self, *, abort: bool = False) -> None:
        """Reap this session's workers and release its resident tokens exactly once."""
        if self.closed:
            return
        self.closed, self.active = True, False
        if self.group is not None:
            self.group.close(abort=abort)
        for token in self.residents.values():
            token.release()
        self.residents.clear()
        if self.in_process:
            from .workers import _RESIDENT, _release_all_shared

            _RESIDENT.clear()
            _release_all_shared()

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        self.close(abort=error is not None)


def run_graph(
    tasks: Sequence[Task],
    *,
    cores: int = 1,
    in_process: bool = True,
    max_workers: int | None = None,
    initial_results: Mapping[TaskKey, Any] | None = None,
    priorities: Mapping[TaskKey, float] | None = None,
    serial_order: Sequence[str] = (),
    contig_lanes: int | None = None,
    recycle_bytes: int | None = None,
    governor: MemoryGovernor | None = None,
    on_task: Callable[[Task, Any], None] | None = None,
    preload: Sequence[str] = (),
    chains: Sequence[ChainPlan] = (),
    dynamic: bool = True,
    _session: TaskSession | None = None,
) -> dict[TaskKey, Any]:
    """Execute static and expanded tasks under one pressure/CPU authority.

    Independent subunits are outer admissions; the expanding parent is suspended
    until its nested gather takes a CPU. Affinity state remains in its worker.
    Supplied initial_results must have been checked against published identities;
    G1's test-only oracle hook performs that same check for fixed-token fixtures.
    """
    from .runtime import affinity_cpus, configure_worker, signal_boundary

    # A failed in-process task can leave a native child thread unwinding. Python
    # cannot interrupt that call; refuse a second authority until it has exited.
    with _TASK_OWNERS_LOCK:
        if _TASK_OWNERS:
            raise SettingsError(
                "an executing task must return Expansion; nested or overlapping graph executors are not allowed"
            )
    tasks = tuple(tasks)
    results = dict(initial_results or {})
    if set(results) & {task.key for task in tasks}:
        raise SettingsError("initial results and scheduled task keys overlap")
    original_order = dependency_order((*tasks, *(Task(key, _precompleted) for key in results)))
    if cores < 1 or cores > affinity_cpus():
        raise ResourceError(f"requested {cores} CPUs; available affinity is {affinity_cpus()}")
    if max_workers is not None and max_workers < 1:
        raise SettingsError("max_workers must be positive")
    for task in tasks:
        if task.resources.min_cores > cores:
            raise ResourceError(f"{task.key}: CPU minimum exceeds allocation")
    if not tasks:
        return results
    configure_worker(cores)
    from .runtime import _RUN, start

    if _RUN is None:
        start(cores, preload=preload)
    elif _RUN.cores < cores:
        raise ResourceError("runtime CPU ceiling is smaller than this graph")
    from .workers import _RESIDENT, WorkerGroup, _perform, _release_all_shared

    global _GOVERNOR
    previous_governor = _GOVERNOR
    injected_governor = governor is not None
    if injected_governor:
        _GOVERNOR = governor
    governor = memory_governor()
    if _session is not None:
        _session.governor = governor
    count = min(cores, max_workers or cores)
    lanes = max(1, min(count, max(1, cores // CONTIG_LANE_CORES) if contig_lanes is None else contig_lanes))
    if contig_lanes is not None and contig_lanes < 1:
        raise SettingsError("contig_lanes must be positive")
    indexed = {task.key: task for task in tasks}
    writers = {output: task.key for task in tasks for output in task.outputs}
    pending = set(indexed)
    requests: dict[TaskKey, AdmissionToken] = {}
    gathers: dict[TaskKey, tuple[TaskKey, int]] = {}
    expansions: dict[TaskKey, tuple[tuple[TaskKey, ...], Task, int]] = {}
    parents: dict[TaskKey, TaskKey] = {}
    depths = {key: 0 for key in indexed}
    running: dict[TaskKey, int] = {}
    # A task's lane owner is its affinity, else the contig chain it belongs to (ENG-23 chains name every discovery,
    # selection, assembly, painting and refit task of a contig): affinity-free block tasks share their contig's lane,
    # weight and straggler floor without being pinned to one worker. Owning them all by None put every contig's block
    # tasks into one lane of about cores / lanes CPUs.
    chain_owners = {key: chain.affinity for chain in chains for key in chain.tasks}
    owners: dict[TaskKey, str | None] = {
        key: task.affinity if task.affinity is not None else chain_owners.get(key) for key, task in indexed.items()
    }
    assigned_lanes: dict[TaskKey, int] = {}
    residents: dict[tuple[int, str], AdmissionToken] = {} if _session is None else _session.residents
    rank: dict[str | None, int] = {scope: index for index, scope in enumerate(serial_order)}
    costs: dict[str | None, float] = {chain.affinity: chain.weight for chain in chains}
    priorities = (
        {key: costs.get(task.affinity, 0.0) for key, task in indexed.items()} if priorities is None else priorities
    )
    group: WorkerGroup | None = None if _session is None else _session.group
    stopped = False

    def cancel() -> None:
        nonlocal stopped
        stopped = True

    def task_order(key: TaskKey) -> tuple:
        task = indexed[key]
        return (-depths.get(key, 0), rank.get(task.affinity, len(rank)), -priorities.get(key, 0.0), key)

    def completed(worker: int, reply: Any) -> None:
        from .records import record_storage_routes

        record_storage_routes(reply.storage_routes)
        task = indexed[reply.key]
        token = requests.pop(reply.key)
        running.pop(reply.key, None)
        for identity, resident in tuple(residents.items()):
            if identity[0] == worker and identity[1] not in reply.resident_affinities:
                resident.release()
                del residents[identity]
        if isinstance(reply.value, Expansion):
            expansion = reply.value
            child_keys = tuple(child.key for child in expansion.subtasks)
            if len(set(child_keys)) != len(child_keys) or set(child_keys) & (indexed.keys() | results.keys()):
                token.release()
                raise SettingsError("expansion task keys must be new and unique")
            gather = expansion.gather
            if gather.key == task.key:
                gather = replace(gather, key=replace(task.key, unit=task.key.unit + ".gather"))
            if gather.key in indexed or gather.key in results or gather.key in child_keys:
                token.release()
                raise SettingsError("expansion gather key must be unique (or the parent key)")
            if gather.affinity is None:
                gather = replace(gather, affinity=task.affinity)
            elif gather.affinity != task.affinity:
                token.release()
                raise SettingsError("expansion gather must use its parent's affinity")
            gather = replace(gather, depends_on=tuple(dict.fromkeys((*gather.depends_on, *child_keys))))
            available = {*results, *indexed, *child_keys, gather.key} - {task.key}
            for child in (*expansion.subtasks, gather):
                if set(child.depends_on) - available:
                    token.release()
                    raise SettingsError("expanded child depends on an unfinished parent or unknown task")
                if child.resources.min_cores > cores:
                    token.release()
                    raise ResourceError("expanded task CPU minimum exceeds allocation")
            new = (*expansion.subtasks, gather)
            # The new tasks' own checks (CPU ranges, references, writers, cycles among them), with finished stand-ins
            # for the existing tasks they depend on; then cycles through the unfinished graph. A suspended parent
            # is completed by its gather, not a resolved leaf: earlier expansions keep that dependency, so cycles
            # through them cannot disappear behind placeholders. Finished tasks depend on nothing.
            new_keys = {child.key for child in new}
            standing = {
                dependency: Task(dependency, _precompleted)
                for child in new
                for dependency in child.depends_on
                if dependency not in new_keys
            }
            depends = {key: item.depends_on for key, item in indexed.items() if key not in results}
            for parent, (_, continuation, _) in expansions.items():
                depends[parent] = (continuation.key,)
                depends[continuation.key] = continuation.depends_on
            depends[task.key] = (gather.key,)
            depends.update((child.key, child.depends_on) for child in new)
            try:
                dependency_order((*new, *standing.values()))
                cycle = cyclic_keys(depends)
                if cycle:
                    raise SettingsError(f"task graph contains a dependency cycle: {cycle}")
            except BaseException:
                token.release()
                raise
            for child in new:
                for output in child.outputs:
                    previous = writers.get(output)
                    if previous is not None and not (child.key == gather.key and previous == task.key):
                        token.release()
                        raise SettingsError(f"{output}: expanded task conflicts with writer {previous}")
                    writers[output] = child.key
            governor._set_states([token.id], SUSPENDED)
            requests[task.key] = token
            expansions[task.key] = (child_keys, gather, worker)
            for child in expansion.subtasks:
                indexed[child.key] = child
                owners[child.key] = child.affinity if child.affinity is not None else owners[task.key]
                pending.add(child.key)
                parents[child.key] = task.key
                depths[child.key] = depths[task.key] + 1
            return
        results[reply.key] = _readonly(reply.value)
        finished = reply.key
        while finished in gathers:
            parent, _ = gathers.pop(finished)
            results[parent] = reply.value
            expansions.pop(parent)
            finished = parent
        if (
            task.affinity is not None
            and task.affinity in reply.resident_affinities
            and (worker, task.affinity) not in residents
        ):
            governor.rest(token.id)
            residents[worker, task.affinity] = token
        else:
            token.release(reply.peak_bytes)
        if on_task is not None:
            on_task(task, reply)
        # A task that has run never needs its arguments again (its value is in results; an expansion's parent is
        # completed by its gather). Keeping them held every round's inputs for the whole graph: each expansion
        # brought a copy of its contig's evidence back from the worker, kept alive with its shared-memory twin.
        if task.args or task.kwargs:
            indexed[reply.key] = replace(task, args=(), kwargs={})

    try:
        with signal_boundary(cancel):
            group = (
                None
                if in_process
                else group
                or WorkerGroup(
                    cores,
                    count,
                    lanes=lanes,
                    chains=chains,
                    recycle_bytes=recycle_bytes,
                    static_affinities=()
                    if dynamic
                    else tuple(
                        (owner, costs.get(owner, 1.0))
                        for owner in dict.fromkeys(task.affinity for task in tasks)
                        if owner is not None
                    ),
                )
            )
            if _session is not None:
                _session.group = group
            while pending or running or expansions:
                if stopped:
                    raise SystemExit(130)
                for parent, (children, gather, worker) in tuple(expansions.items()):
                    if gather.key not in indexed and all(key in results for key in children):
                        indexed[gather.key] = gather
                        owners[gather.key] = owners[parent]
                        pending.add(gather.key)
                        depths[gather.key] = depths[parent]
                        gathers[gather.key] = (parent, worker)
                ready = [
                    key
                    for key in sorted(pending, key=task_order)
                    if all(dependency in results for dependency in indexed[key].depends_on)
                ]
                if not ready and not running:
                    raise InvariantError("graph has unfinished work but no executable dependency")
                if group is not None:
                    needed = {owners[k] for k in (*ready, *running)}
                    done = results.keys() if _session is None else results.keys() | _session.completed
                    group.authority.reconcile(needed, done)
                    group.authority.refresh()  # every lane's total, once per pass
                active_affinities = {indexed[k].affinity for k in running if indexed[k].affinity is not None}
                dispatched = False
                running_cores = sum(indexed[k].resources.min_cores for k in running)
                idle = None if group is None else sum(1 for h in group.handles if h.ready and h.busy is None)
                lane_load: dict[int, int] = {}
                for k in running:
                    if k in assigned_lanes:
                        held = assigned_lanes[k]
                        lane_load[held] = lane_load.get(held, 0) + indexed[k].resources.min_cores
                # Per owner, the smallest CPU minimum its lane could not start this pass: its later ready tasks
                # needing as many are skipped at once (a full lane stays full until the next pass's replies).
                full: dict[str | None, int] = {}
                # Gathers and pinned tasks (which can run only on one worker) come first in every pass: otherwise the
                # first idle workers, where the earliest plans and expansions ran, went to the next block every pass
                # and a contig's gather waited for thousands of other tasks. Nothing is reserved, so a pinned task its
                # lane cannot admit never holds a worker that the lane's own work needs.
                if group is not None:
                    pinned = [k for k in ready if k in gathers or indexed[k].affinity in group.affinities]
                    if pinned:
                        chosen = set(pinned)
                        ready = pinned + [k for k in ready if k not in chosen]
                for key in ready:
                    if idle is not None and idle <= 0:
                        break  # every worker is busy: nothing more can start this pass
                    task = indexed[key]
                    if key not in gathers and task.resources.min_cores >= full.get(owners[key], cores + 1):
                        continue
                    affinity = task.affinity
                    if affinity is not None and affinity not in active_affinities:
                        if len(active_affinities) >= (1 if serial_order else lanes):
                            continue
                    if running_cores + task.resources.min_cores > cores:
                        continue
                    if group is not None:
                        if key in gathers:
                            worker = gathers[key][1]
                            handle = group.handles[worker]
                            if not handle.ready or handle.busy is not None:
                                continue
                        else:
                            available_worker = group.free_worker(affinity)
                            if available_worker is None:
                                continue
                            worker = available_worker
                        if task.function.__module__ == "__main__" or "<locals>" in task.function.__qualname__:
                            raise SettingsError(f"{key}: process tasks must be importable module-level functions")
                    else:
                        worker = 0
                    if group is not None:
                        lane = group.authority.acquire(owners[key], minimum=task.resources.min_cores)
                        if lane is None:
                            if key not in gathers:
                                full[owners[key]] = min(full.get(owners[key], cores + 1), task.resources.min_cores)
                            continue
                        counters = group.authority.groups[lane]
                        # Hand a lane no more tasks than its CPUs admit: the rest would wait in begin_task on their
                        # workers, leaving none for other lanes' ready tasks (head-of-line blocking across contigs).
                        load = lane_load.get(lane, 0) + task.resources.min_cores
                        if key not in gathers and load > max(1, _read(counters.total)):
                            full[owners[key]] = min(full.get(owners[key], cores + 1), task.resources.min_cores)
                            continue
                        assigned_lanes[key] = lane
                        if not any(owners[k] == owners[key] for k in running) and (not _read(counters.active)):
                            batch = [k for k in ready if owners[k] == owners[key]]
                            participants = count if any(indexed[k].affinity is None for k in batch) else 1
                            counters.start_batch(len(batch), participants)
                    token = requests.get(key)
                    if key in gathers and token is None:
                        parent = gathers[key][0]
                        token = requests.pop(parent)
                        requests[key] = token
                        governor._set_states([token.id], QUEUED)
                    elif token is None:
                        parent_key = parents.get(key)
                        parent_token = None if parent_key is None else requests.get(parent_key)
                        token = governor.request(
                            label=f"{key.stage}.{key.unit}",
                            queued=True,
                            outer=True,
                            parent=0 if parent_token is None else parent_token.id,
                            depth=depths[key],
                        )
                        requests[key] = token
                    if not token.poll(fresh=True):
                        if group is not None:
                            group.authority.groups[assigned_lanes[key]].open_gate()
                            group.authority.lend_idle(owners[key])
                        continue
                    action = _resolved_task(task, results)
                    pending.remove(key)
                    running[key] = worker
                    running_cores += task.resources.min_cores
                    if idle is not None:
                        idle -= 1
                    if key in assigned_lanes:
                        held = assigned_lanes[key]
                        lane_load[held] = lane_load.get(held, 0) + task.resources.min_cores
                    if affinity is not None:
                        active_affinities.add(affinity)
                    dispatched = True
                    if group is None:
                        completed(worker, _perform(action, token.held, FixedBudget(cores)))
                        break
                    group.pin(affinity, worker)
                    group.submit(worker, action, token.held, lane=assigned_lanes[key])
                if group is not None:
                    if ready and all(k not in pending for k in ready):
                        for counters in group.authority.groups:
                            counters.open_gate()
                    for worker, reply in group.receive(timeout=0.0 if dispatched else CPU_POLL_SECONDS):
                        completed(worker, reply)
                elif not dispatched:
                    if not ready and (not running):
                        raise InvariantError("graph has pending work but no executable dependency")
                    time.sleep(WAKE_SECONDS)
            if _session is not None:
                _session.completed.update(results.keys() & _session.chain_keys)
                if group is not None:
                    group.authority.reconcile(set(), _session.completed)
    except BaseException:
        if _session is not None:
            _session.close(abort=True)
        elif group is not None:
            group.close(abort=True)
        raise
    finally:
        retained = residents.values() if _session is None else ()
        for token in (*requests.values(), *retained):
            token.release()
        if group is not None and _session is None:
            group.close()
        if in_process and _session is None:
            _RESIDENT.clear()
            _release_all_shared()
        if injected_governor:
            _GOVERNOR = previous_governor
    order = (*original_order, *sorted(results.keys() - set(original_order)))
    return {key: results[key] for key in order}


class TaskBudget(CpuBudget):
    """A task's declared range around the native elastic share policy."""

    live = True

    def __init__(self, budget: CpuBudget, minimum: int, maximum: int):
        self.budget, self.minimum, self.maximum = (budget, minimum, maximum)

    def threads(self) -> int:
        return self.claim()

    def claim(self, maximum: int | None = None) -> int:
        ceiling = self.maximum if maximum is None else min(self.maximum, max(1, maximum))
        floor = min(self.minimum, ceiling)
        team = capped_threads(self.budget, ceiling)
        if isinstance(self.budget, PoolShareBudget) and team < floor:
            seat = pool_seat()
            assert seat is not None
            c = seat.counters
            with c.admission:
                while _read(c.total) - (_read(c.claimed) - seat.claimed) < floor:
                    if _read(c.closed):
                        raise CancelledError("CPU group closed during minimum-team admission")
                    c.admission.wait(CPU_POLL_SECONDS)
                c.claimed.get_obj().value += floor - seat.claimed
                seat.claimed = floor
            team = floor
        return min(ceiling, max(floor, team))

    def current(self) -> int:
        return min(self.maximum, self.budget.current())


@dataclass(frozen=True)
class ChainPlan:
    """One scheduling chain's cost and last boundary, independent of model edges."""

    affinity: str
    weight: float
    tasks: tuple[TaskKey, ...]


class LaneAuthority:
    """Native leases and group shares, with nonblocking coordinator observations.

    Workers may die while holding CPU locks. The coordinator never waits for
    those locks: it returns to transport liveness checks and aborts the group.
    Only the worker's own boundary changes its running team; every lease floor
    covers all currently claimed group CPUs.
    """

    def __init__(self, context: Any, cores: int, lanes: int, processes: int, chains: Sequence[ChainPlan]):
        self.chains = tuple(chains)
        self.indices = {chain.affinity: i for i, chain in enumerate(chains)}
        self.state = LeaseState(
            context,
            lanes,
            cores,
            task_weights=[chain.weight for chain in chains] if chains else None,
            straggler_ratio=STRAGGLER_RATIO,
        )
        self.seats = [_LeaseSeat(self.state, i) for i in range(lanes)]
        self.groups = tuple(PoolCounters(context, 1, processes) for _ in range(lanes))
        for i, group in enumerate(self.groups):
            group.lease_state, group.lease_index = self.state, i
            group.start_batch(processes, processes)
        self.owners: dict[str | None, int] = {}
        self.finished: set[str] = set()

    def acquire(self, owner: str | None, *, minimum: int = 1) -> int | None:
        """``owner``'s lane once it holds ``minimum`` CPUs (None meanwhile). A lane that already holds them returns
        at once: the dispatcher refreshes every lane's total once per pass (a pass walks thousands of ready
        blocks), not once per task."""
        if owner in self.owners:
            index = self.owners[owner]
            if not self.state.lock.acquire(False):
                return None
            try:
                if self.state.lent[index] == 0 and self.state.claims[index] >= minimum:
                    return index
            finally:
                self.state.lock.release()
        if owner not in self.owners:
            free = [i for i in range(len(self.groups)) if i not in self.owners.values()]
            if not free or not self.state.lock.acquire(False):
                return None
            try:
                index = free[0]
                self.owners[owner] = index
                self.state.lent[index] = -1
                if owner in self.indices and owner not in self.finished:
                    weight = self.state.task_weights[self.indices[owner]]
                    self.state.weights[index] = weight if weight >= self.state.threshold else 0.0
            finally:
                self.state.lock.release()
        index = self.owners[owner]
        self.refresh()
        if not self.state.lock.acquire(False):
            return None
        try:
            claim = int(self.state.claims[index])
            free = self.state.budget - sum(self.state.claims)
            if claim < minimum:
                if free < minimum - claim:
                    return None
                self.state.claims[index] = minimum
            self.state.lent[index] = 0
        finally:
            self.state.lock.release()
        self.refresh()
        return index

    def refresh(self) -> None:
        for index in sorted(self.owners.values()):
            group = self.groups[index]
            if not group.admission.acquire(False):
                continue
            try:
                if not self.state.lock.acquire(False):
                    continue
                try:
                    if self.state.claims[index] <= 0 or self.state.lent[index] == 1:
                        continue
                    _, total = self.seats[index]._grow_locked(_read(group.claimed))
                    group.total.get_obj().value = total
                finally:
                    self.state.lock.release()
            finally:
                group.admission.release()

    def lend_idle(self, owner: str | None) -> None:
        if owner not in self.owners:
            return
        index = self.owners[owner]
        group = self.groups[index]
        if not group.admission.acquire(False):
            return
        try:
            if not self.state.lock.acquire(False):
                return
            try:
                if not _read(group.claimed) and self.state.claims[index] > 0:
                    self.state.claims[index] = 1
                    self.state.lent[index] = 1
                    group.total.get_obj().value = 1
            finally:
                self.state.lock.release()
        finally:
            group.admission.release()
        self.refresh()

    def reconcile(self, needed: set[str | None], results: Container[TaskKey]) -> None:
        if not self.state.lock.acquire(False):
            return
        try:
            for chain in self.chains:
                if chain.affinity not in self.finished and all(key in results for key in chain.tasks):
                    index = self.owners.get(chain.affinity)
                    self.state.remaining.value = max(0.0, self.state.remaining.value - chain.weight)
                    if index is not None:
                        self.state.weights[index] = 0.0
                    self.finished.add(chain.affinity)
            for owner, index in tuple(self.owners.items()):
                if owner not in needed and not _read(self.groups[index].claimed):
                    self.state.claims[index] = self.state.lent[index] = 0
                    if self.state.proportional:
                        self.state.weights[index] = 0.0
                    del self.owners[owner]
        finally:
            self.state.lock.release()
        self.refresh()


def partition_chains(weights: Sequence[float], workers: int) -> tuple[tuple[int, ...], ...]:
    """ENG-22 static placement: cost, then load/task-count/lower-index ties."""
    if workers < 1:
        raise SettingsError("chain partitions require a positive worker count")
    partitions: list[list[int]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for index in sorted(range(len(weights)), key=lambda i: (-weights[i], i)):
        worker = min(range(workers), key=lambda i: (loads[i], len(partitions[i]), i))
        partitions[worker].append(index)
        loads[worker] += max(0.0, weights[index])
    return tuple(tuple(partition) for partition in partitions)
