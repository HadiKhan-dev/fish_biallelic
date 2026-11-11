"""Keep shared arrays and resident state in bounded, fail-fast forkserver workers.

Owns transport, shared-array lifetimes and the two native recycling cadences.
Scheduling/admission belongs to scheduler; science never opens a pool.
Graph tasks also own eager kernel compilation. Methods: ENG-21, ENG-22 and ENG-24 in
docs/methods/execution.md. Constants use bytes and seconds as named.
"""

from __future__ import annotations

import atexit
import functools
import gc
import io
import logging
import math
import mmap
import multiprocessing as mp
import os
import pickle
import sys
import threading
import time
import traceback
import weakref
from collections.abc import Callable, Container, Mapping
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import count
from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, TypeVar, cast

import numba
import numpy as np

from ..core.errors import InvariantError, SettingsError
from ..core.immutable import declared_immutable
from ..core.kernels import kernel
from . import runtime as bootstrap
from .cpu import (
    CpuBudget,
    PoolCounters,
    PoolShareBudget,
    budget_context,
    thread_team,
)
from .graph import Expansion, PackedCall, ResultRef, Task, TaskKey, _field_names, _readonly
from .memory import (
    adopted,
    malloc_trim,
    memory_governor,
    release_memory,
    task_peak_bytes,
    task_peak_start,
)

_LOG = logging.getLogger(__name__)
__all__ = [
    "SharedArrayRef",
    "MappedArrayRef",
    "new_segment",
    "empty_shared_array",
    "attach_shared_array",
    "close_shared_memory",
    "main_module_guard",
    "recycle_threshold",
    "ResidentRef",
    "WorkerReply",
    "retain",
    "release_resident",
    "resident_value",
    "resident_or_load",
    "release_resident_namespace",
    "SharedValue",
    "share_value",
    "call_shared",
    "release_shared_value",
    "share_output",
    "release_output",
    "RecyclingPolicy",
    "WorkerGroup",
]

COPY_CHUNK_BYTES = 1 << 20  # ENG-24: copy partitions of 1 MiB.
_PARALLEL_COPY_BYTES = 4 << 20


@dataclass(frozen=True)
class SharedArrayRef:
    """What a worker needs to attach a shared array: segment name, shape and dtype (string)."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class MappedArrayRef:
    """Read-only file bytes pinned by the publishing process until the last consumer.

    The proc-fd path remains open even after the source mapping closes or its file
    is unlinked. Shape/order are view metadata; no scientific array is copied.
    """

    name: str
    offset: int
    shape: tuple[int, ...]
    dtype: str


class _FilePin:
    """One independent read-only descriptor for an existing mapped backing."""

    def __init__(self, fd: int):
        self.fd = fd
        self.name = f"/proc/{os.getpid()}/fd/{fd}"

    def close(self) -> None:
        os.close(self.fd)


def _read_only_mapping(array: np.ndarray) -> bool:
    """Cheap candidate check; actual mapping permissions are checked before reuse."""
    if not array.nbytes or not _plain_dtype(array.dtype):
        return False
    item: Any = array
    while isinstance(item, np.ndarray):
        if item.flags.writeable:
            return False
        item = item.base
    if isinstance(item, memoryview):
        item = item.obj
    return isinstance(item, mmap.mmap)


def _mapped_source(array: np.ndarray) -> tuple[int, int, int, str] | None:
    """Device, inode, byte offset and path of a contiguous immutable file region.

    A private copy-on-write mapping may differ from the file and must be copied.
    Writable file maps are also copied. Read-only views of engine/shared tmpfs
    inputs follow SharedValue's immutable-input lifetime contract.
    """
    if not _read_only_mapping(array):
        return None
    address = int(array.__array_interface__["data"][0])
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split(None, 5)
        if len(parts) != 6:
            continue
        span, permissions, offset, device, inode, path = parts
        first, last = (int(value, 16) for value in span.split("-"))
        if first <= address and address + array.nbytes <= last:
            shared_tmpfs = path.startswith("/dev/shm/") and permissions.endswith("s")
            if not int(inode) or ("w" in permissions and not shared_tmpfs):
                return None
            major, minor = (int(value, 16) for value in device.split(":"))
            return os.makedev(major, minor), int(inode), int(offset, 16) + address - first, path
    return None


def _pin_file(source: tuple[int, int, int, str]) -> _FilePin | None:
    """Pin the exact inode when reopenable; otherwise leave the snapshot-copy route."""
    device, inode, _, path = source
    paths = [path] if not path.endswith(" (deleted)") else []
    try:
        paths.extend(str(entry) for entry in Path("/proc/self/fd").iterdir())
    except OSError:
        return None
    for candidate in paths:
        try:
            stat = os.stat(candidate)
            if (stat.st_dev, stat.st_ino) != (device, inode):
                continue  # do not open unrelated pipes/sockets while finding the mapping
            fd = os.open(candidate, os.O_RDONLY)
        except OSError:
            continue
        stat = os.fstat(fd)
        if (stat.st_dev, stat.st_ino) == (device, inode):
            return _FilePin(fd)
        os.close(fd)
    return None  # anonymous mappings and deleted trackfd=False maps have no reopenable fd


def _attach_array(reference: SharedArrayRef | MappedArrayRef) -> tuple[Any, np.ndarray]:
    if isinstance(reference, SharedArrayRef):
        return attach_shared_array(reference, track=False)
    fd = os.open(reference.name, os.O_RDONLY)
    try:
        start = reference.offset - reference.offset % mmap.ALLOCATIONGRANULARITY
        delta = reference.offset - start
        size = math.prod(reference.shape) * np.dtype(reference.dtype).itemsize
        handle = mmap.mmap(fd, size + delta, access=mmap.ACCESS_READ, offset=start)
    finally:
        os.close(fd)
    try:
        return handle, np.ndarray(reference.shape, dtype=reference.dtype, buffer=handle, offset=delta)
    except BaseException:
        handle.close()
        raise


def _array_reference(reference: SharedArrayRef | MappedArrayRef, array: np.ndarray):
    """Another shape/dtype over the same contiguous bytes (Fortran already transposed)."""
    if isinstance(reference, MappedArrayRef):
        return MappedArrayRef(reference.name, reference.offset, tuple(array.shape), array.dtype.str)
    return SharedArrayRef(reference.name, tuple(array.shape), array.dtype.str)


def new_segment(size: int, *, name: str | None = None, track: bool = True) -> SharedMemory:
    """A new segment of ``size`` bytes (at least one) named after the run (see the module docstring), or ``name``
    (FileExistsError if it exists). ``track=False`` leaves it out of this process's resource tracker: a segment a
    worker creates for its coordinator to adopt."""
    from .memory import mark_in_use, segment_name

    for _ in range(1 if name is not None else 64):
        try:
            # Python 3.13+ lets the creator control resource-tracker ownership.
            handle = SharedMemory(name=name or segment_name(), create=True, size=max(int(size), 1), track=track)
        except FileExistsError:
            if name is not None:
                raise
            continue
        mark_in_use(getattr(handle, "_fd", -1))
        return handle
    raise FileExistsError("no free shared-memory segment name")


@kernel(
    overloads=(
        (numba.types.uint8[::1], numba.types.uint8[::1]),
        # Declare read-only sources as well as writable sources for frozen record-array bytes.
        (numba.types.Array(numba.types.uint8, 1, "C", readonly=True), numba.types.uint8[::1]),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _copy_shared_bytes(source, destination):
    chunk = COPY_CHUNK_BYTES
    for part in numba.prange((source.size + chunk - 1) // chunk):
        start = part * chunk
        stop = min(source.size, start + chunk)
        destination[start:stop] = source[start:stop]


def _shareable(array) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        raise TypeError("object arrays cannot be transported by shared memory")
    return contiguous


def _fill_shared(
    contiguous: np.ndarray, handle: SharedMemory, copy_threads: int, *, capped: bool = True
) -> SharedArrayRef:
    """Copy into an allocated segment (create_shared_array's second half); the caller owns failure cleanup.
    ``capped=False``: ``copy_threads`` already accounts for the CPUs the caller may use (a reply copy borrowing its
    lane's idle CPUs), so it is not capped at the task's claim again."""
    threads = int(copy_threads)
    if threads > 1 and contiguous.nbytes >= _PARALLEL_COPY_BYTES:
        from .cpu import authority_threads, thread_team

        with thread_team(authority_threads(threads) if capped else threads):
            _copy_shared_bytes(
                np.frombuffer(memoryview(contiguous), dtype=np.uint8),
                np.frombuffer(cast(memoryview, handle.buf), dtype=np.uint8, count=contiguous.nbytes),
            )
    else:
        np.copyto(np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=handle.buf), contiguous)
    return SharedArrayRef(handle.name, tuple(contiguous.shape), contiguous.dtype.str)


def empty_shared_array(shape, dtype) -> tuple[SharedMemory, SharedArrayRef, np.ndarray]:
    """A new uninitialised segment for an array of ``shape`` and ``dtype``: (handle, reference, writable view).

    The caller fills it (e.g. in one parallel scan), passes ``reference`` to pools (``shared={"name": reference}``:
    attached, not copied). The allocation contributes to measured memory pressure.
    At the end the caller drops the view, closes and unlinks the segment
    (close_shared_memory([handle], unlink=True)).
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


def attach_shared_array(ref: SharedArrayRef, *, track: bool = True) -> tuple[SharedMemory, np.ndarray]:
    """Attach to a segment created by create_shared_array; return (handle, array view)."""
    from .memory import mark_in_use

    dtype = np.dtype(ref.dtype)
    handle = SharedMemory(name=ref.name, create=False, track=track)
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


RECYCLE_MIN_BYTES = 256 << 20
RECYCLE_MAX_BYTES = 1 << 30
# Give a worker time to release native resources before its coordinator escalates.
# runtime.COORDINATOR_GRACE_SECONDS also covers the coordinator's bounded reaping.
GROUP_TERMINATE_GRACE_SECONDS = 10.0
RECYCLE_CHECK_SECONDS = 0.2
FORKSERVER = mp.get_context("forkserver")
_GUARD_LOCK = threading.RLock()
_GUARD_DEPTH = 0
_GUARD_SAVED = None
# A forkserver listens on <multiprocessing's temporary directory>/sock-<12 hex digits>, and a socket path holds at
# most 107 bytes on Linux (103 on macOS and the BSDs). CPython 3.14 moves that directory off a long TMPDIR to /tmp,
# but sizes the socket name as 8 characters. Account for the full socket path to prevent
# an AF_UNIX path-length failure when creating a pool.
_SOCKET_NAME_BYTES = len("/sock-") + 12
_SOCKET_PATH_LIMIT = 107 if sys.platform.startswith("linux") else 103
_SYSTEM_TEMPORARY_DIRECTORIES = ("/tmp", "/var/tmp", "/usr/tmp")


def _short_socket_directory() -> None:
    """Keep multiprocessing's per-process temporary directory, which holds only its sockets, short enough for a
    forkserver's listener: when TMPDIR is too long, that one directory moves to the system temporary directory. TMPDIR
    and every other temporary file stay where the user put them."""
    if os.name == "nt":
        return
    import shutil
    import tempfile
    from multiprocessing import process, util

    config = cast(Any, process.current_process())._config  # where multiprocessing.util.get_temp_dir() keeps it
    directory = config.get("tempdir") or os.path.join(tempfile.gettempdir(), "pymp-XXXXXXXX")
    if len(os.fsencode(directory)) + _SOCKET_NAME_BYTES <= _SOCKET_PATH_LIMIT:
        return
    base = next((d for d in _SYSTEM_TEMPORARY_DIRECTORIES if os.access(d, os.W_OK | os.X_OK)), None)
    if base is None:
        raise SettingsError(
            f"TMPDIR {tempfile.gettempdir()} is too long for worker sockets and no system temporary directory "
            "(/tmp) is writable: set TMPDIR to a shorter path"
        )
    directory = tempfile.mkdtemp(prefix="pymp-", dir=base)
    util.Finalize(None, shutil.rmtree, args=(directory,), kwargs={"ignore_errors": True}, exitpriority=-100)
    config["tempdir"] = directory


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
            # Every engine process start passes here: its forkserver's socket path must fit (_short_socket_directory).
            _short_socket_directory()
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


def recycle_threshold(processes: int, recycle_bytes: int | None = None) -> int:
    """The private-memory growth after which a worker is replaced: ``recycle_bytes`` if given (0: never), else a
    quarter of a worker's share of the run's memory limit within RECYCLE_MIN_BYTES..RECYCLE_MAX_BYTES."""
    if recycle_bytes is None:
        recycle_bytes = min(
            RECYCLE_MAX_BYTES, max(RECYCLE_MIN_BYTES, memory_governor().limit // (4 * max(1, int(processes))))
        )
    return max(0, int(recycle_bytes))


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
    threads: tuple[int, int] = (1, 1)  # CPUs the task held at its start and at most while it ran
    diagnostics: tuple[str, ...] = ()
    publish_lock_wait_seconds: float = 0.0
    publish_fsync_seconds: float = 0.0
    thread_seconds: tuple[tuple[int, float], ...] = ()  # seconds at each claimed CPU count


_RESIDENT: dict[tuple[str, str], Any] = {}
_RELOADABLE: set[tuple[str, str]] = set()  # Cache entries only; explicit ResidentRefs stay pinned.
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
    _RELOADABLE.discard((reference.affinity, reference.key))


def resident_value(reference: ResidentRef) -> Any:
    """Borrow state within its lifetime; returning this value copies across workers."""
    try:
        return _RESIDENT[reference.affinity, reference.key]
    except KeyError as error:
        raise InvariantError(f"resident reference is absent or released: {reference}") from error


_Loaded = TypeVar("_Loaded")


def resident_or_load(key: str, load: Callable[[], _Loaded]) -> _Loaded:
    """The value an earlier action of the running affinity retained under ``key``, else ``load()``, retained for the
    affinity's later actions until released (release_resident_namespace) or until the graph ends.

    ``key`` names what determines the value, such as the store record identities it is decoded from, so a retained
    value equals a fresh load and a worker without it (another worker, a resumed run) simply loads it again. Every
    later action receives the same object: values are immutable (frozen records, tuples, read-only arrays). Outside
    an affinity action the value is only loaded."""
    if _ACTIVE_AFFINITY is None:
        return load()
    identity = (_ACTIVE_AFFINITY, key)
    if identity not in _RESIDENT:
        _RESIDENT[identity] = load()
        _RELOADABLE.add(identity)
    return _RESIDENT[identity]


def release_resident_namespace(namespace: str) -> None:
    """Release the running affinity's values retained under ``namespace/...`` keys, at their declared last consumer
    (values never loaded on this worker have nothing to release)."""
    if _ACTIVE_AFFINITY is None:
        return
    prefix = namespace + "/"
    for identity in [item for item in _RESIDENT if item[0] == _ACTIVE_AFFINITY and item[1].startswith(prefix)]:
        del _RESIDENT[identity]
        _RELOADABLE.discard(identity)


def _release_idle_cache() -> int:
    """Drop only reloadable state, at an idle task boundary; live continuations stay.

    A cache loader is identity-keyed and recreates the same immutable value. A
    continuation may still refer to that value; Python then keeps it alive until
    the continuation finishes. Attached segments close only after their last view.
    """
    global _SHARED_UNCLOSED
    count = sum(identity in _RESIDENT for identity in _RELOADABLE)
    for identity in _RELOADABLE:
        _RESIDENT.pop(identity, None)
    _RELOADABLE.clear()
    release_memory()
    _SHARED_UNCLOSED = _close_unreferenced(_SHARED_UNCLOSED)
    _retry_reply_closes()
    return count


@dataclass(frozen=True)
class SharedValue:
    """A read-only value over copied or pinned immutable backing (ENG-24 transport).

    The independent subtasks of that action's Expansion receive this small reference instead of the
    arrays and attach them without copying (call_shared). The owner releases the segments at its
    declared last consumer, normally the Expansion's gather on the same affinity (release_shared_value).
    ``plan`` mirrors the value: dataclasses, C- or Fortran-ordered arrays, tuples of 1-D arrays of one
    dtype, and plain picklable leaves.
    """

    affinity: str
    key: str
    plan: Any
    arrays: tuple[tuple[str, SharedArrayRef | MappedArrayRef], ...]


class _SharedBacking:
    """A copied segment or pinned file shared by independent SharedValue lifetimes."""

    def __init__(self, key: tuple | None, handle: SharedMemory | _FilePin):
        self.key, self.handle, self.users = key, handle, 0
        # Weak source views validate address reuse without retaining the private twin.
        self.sources: dict[int, weakref.ref] = {}

    def release(self) -> None:
        self.users -= 1
        if self.users:
            return
        if self.key is not None and _SHARED_BACKINGS.get(self.key) is self:
            del _SHARED_BACKINGS[self.key]
        if isinstance(self.handle, SharedMemory):
            close_shared_memory([self.handle], unlink=True)
            _announce_shared("shared-", (self.handle.name,))
        else:
            self.handle.close()


_SHARED_BACKINGS: dict[tuple, _SharedBacking] = {}
_SHARED_OWNED: dict[tuple[str, str], list[_SharedBacking]] = {}
_SHARED_UNCLOSED: list[tuple[SharedMemory | mmap.mmap, Any]] = []
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
    """Publish arrays once per immutable backing; the affinity owns independent lifetime holds.

    Exact transport: every array keeps its values, shape, dtype and memory order. Any failure after
    allocation releases every acquired hold. Private mutable inputs are copied;
    read-only mapped/shared inputs borrow their pinned backing and must remain immutable
    until their last SharedValue releases it. Private copy-on-write data is copied.
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
    handles: list[_SharedBacking] = []
    _SHARED_OWNED[identity] = handles
    references = []
    copied: dict[tuple[int, int], SharedArrayRef | MappedArrayRef] = {}
    reference: SharedArrayRef | MappedArrayRef
    pinned: set[_SharedBacking] = set()
    try:
        for name, array in arrays.items():
            descriptor = (array.__array_interface__["data"][0], array.nbytes)
            if descriptor in copied:
                references.append((name, _array_reference(copied[descriptor], array)))
                continue
            source = _mapped_source(array)
            cache_key = (
                ("file", *source[:2]) if source is not None else (("copy", *descriptor) if _fixed(array) else None)
            )
            backing = _SHARED_BACKINGS.get(cache_key) if cache_key is not None else None
            if (
                backing is not None
                and source is None
                and not any(ref() is not None for ref in backing.sources.values())
            ):
                backing = None  # the old address can now belong to different bytes
            if backing is None:
                file = _pin_file(source) if source is not None else None
                if source is not None and file is None:
                    source, cache_key = None, None
                if source is not None:
                    backing = _SharedBacking(cache_key, cast(_FilePin, file))
                else:
                    contiguous = _shareable(array)
                    handle = new_segment(contiguous.nbytes)
                    _announce_shared("shared+", (handle.name,))
                    try:
                        _fill_shared(contiguous, handle, copy_threads)
                    except BaseException:
                        close_shared_memory([handle], unlink=True)
                        _announce_shared("shared-", (handle.name,))
                        raise
                    backing = _SharedBacking(cache_key, handle)
                if cache_key is not None:
                    _SHARED_BACKINGS[cache_key] = backing
            if backing not in pinned:
                backing.users += 1
                pinned.add(backing)
                handles.append(backing)
            if source is not None:
                reference = MappedArrayRef(backing.handle.name, source[2], tuple(array.shape), array.dtype.str)
            else:
                backing.sources[id(array)] = weakref.ref(array)
                reference = SharedArrayRef(backing.handle.name, tuple(array.shape), array.dtype.str)
            copied[descriptor] = reference
            references.append((name, reference))
    except BaseException:
        _SHARED_OWNED.pop(identity, None)
        for backing in handles:
            backing.release()
        raise
    return SharedValue(identity[0], identity[1], plan, tuple(references))


def _close_unreferenced(entries: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
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
    entries: list[tuple[Any, Any]] = []
    try:
        attached, unique = {}, {}
        for name, array_reference in reference.arrays:
            if array_reference not in unique:
                handle, array = _attach_array(array_reference)
                array.flags.writeable = False
                entries.append((handle, weakref.ref(array)))
                unique[array_reference] = array
                del array
            attached[name] = unique[array_reference]
        value = _shared_build(reference.plan, attached)
        del attached, unique
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
    for backing in handles:
        backing.release()


def share_output(shape, dtype, key: str) -> tuple[SharedArrayRef, np.ndarray]:
    """A writable shared array owned by the calling task's affinity like a SharedValue: subtasks attach
    its reference (attach_shared_array) and each fills its own part; ``release_output(key)`` unlinks it after its last
    use. Until then it is one of the affinity's owned segments, so the engine's failure paths unlink it as they unlink
    shared values (a failed task's holds, an aborted in-process graph, the coordinator's announced names, exit).
    Returns (reference, the owner's writable view)."""
    if _ACTIVE_AFFINITY is None:
        raise InvariantError("a shared output requires task affinity")
    identity = (_ACTIVE_AFFINITY, key)
    if identity in _SHARED_OWNED:
        raise InvariantError(f"shared output {key!r} already exists for {_ACTIVE_AFFINITY!r}")
    handle, reference, view = empty_shared_array(shape, dtype)
    backing = _SharedBacking(None, handle)
    backing.users = 1
    _SHARED_OWNED[identity] = [backing]
    _announce_shared("shared+", (handle.name,))
    return reference, view


def release_output(key: str) -> None:
    """Unlink the calling affinity's shared output ``key`` (drop the views of it first)."""
    release_shared_value(SharedValue(_ACTIVE_AFFINITY, key, None, ()))


def _release_all_shared() -> None:
    """Unlink every segment this process still owns (process exit or an aborted in-process graph)."""
    while _SHARED_OWNED:
        _, handles = _SHARED_OWNED.popitem()
        for backing in handles:
            backing.release()


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


def _task_peak_threads() -> int:
    """The most CPUs this process's running task has held (its pool seat), 1 outside a pool."""
    from . import cpu

    seat = cpu.pool_seat()
    return 1 if seat is None else max(1, int(seat.peak))


def perform_task(task: Task, token: tuple[int, int], budget: CpuBudget, *, keep_gathers: bool = False) -> WorkerReply:
    global _ACTIVE_AFFINITY
    from ..store.protocol import ClaimBusy
    from .cooperation import Deferred
    from .graph import _readonly
    from .runtime import task_boundary

    task_boundary()
    governor = memory_governor()
    governor.start(token[0])
    _ACTIVE_AFFINITY = task.affinity
    start, cpu, peak = (time.perf_counter(), time.process_time(), task_peak_start())
    try:
        args, kwargs = (task.args, dict(task.kwargs))
        from .cpu import TaskBudget, executing_task
        from .records import ThreadUsage

        usage = ThreadUsage(task.resources.min_cores, start=start)
        budget = TaskBudget(budget, task.resources.min_cores, task.resources.max_cores, observe=usage.observe)
        team = budget.threads()
        if task.thread_argument is not None:
            kwargs[task.thread_argument] = team
        _readonly((args, kwargs))
        from ..store.container import capture_publish_timing
        from .records import capture_diagnostics

        try:
            with (
                capture_diagnostics() as diagnostics,
                capture_publish_timing() as publish_timing,
                executing_task(task.key),
                adopted(token),
                # Restore the outer mask after the finished task's budget unbinds.
                thread_team(team),
                budget_context(budget),
            ):
                value = _readonly(task.function(*args, **kwargs))
                if keep_gathers and task.affinity is not None and isinstance(value, Expansion):
                    value = replace(value, gather=_kept_gather(value.gather, task.affinity))
        except ClaimBusy as error:
            value = Deferred(str(error))
        except bootstrap.RunInterrupted:
            raise
        except BaseException as error:
            # The existing exception transport retains observed I/O times even when no reply is produced.
            failure = bootstrap.task_failure(error)
            failure.publish_lock_wait_seconds = publish_timing.lock_wait_seconds
            failure.publish_fsync_seconds = publish_timing.fsync_seconds
            raise failure from error
        stop = time.perf_counter()
        return WorkerReply(
            task.key,
            value,
            stop - start,
            time.process_time() - cpu,
            task_peak_bytes(peak),
            tuple(sorted({key[0] for key in _RESIDENT})),
            diagnostics=tuple(diagnostics),
            publish_lock_wait_seconds=publish_timing.lock_wait_seconds,
            publish_fsync_seconds=publish_timing.fsync_seconds,
            threads=(int(team), max(int(team), _task_peak_threads())),
            thread_seconds=usage.finish(stop),
        )
    finally:
        _ACTIVE_AFFINITY = None


def _graph_worker(connection: Any, settings: Any, groups: tuple[PoolCounters, ...], threshold: int) -> None:
    """Persistent forkserver child; failure is sent once, never respawned in a loop.

    Owned shared values are unlinked whenever this loop ends (multiprocessing children skip atexit); a worker
    that is terminated instead leaves them to its coordinator, which knows their names (_announce_shared).
    """
    global _OWNER_CHANNEL, _SHARED_UNCLOSED
    from dataclasses import replace

    from .cooperation import watch_coordinator

    watch_coordinator()  # Before bootstrap, unpacking, CPU admission or a scientific call can block.
    _OWNER_CHANNEL = connection
    try:
        with bootstrap.signal_boundary(lambda: None):
            bootstrap.apply_worker_settings(settings)
            from .cpu import PoolSeat

            seats = [PoolSeat(counters) for counters in groups]
            policy = RecyclingPolicy(threshold)
            connection.send(("ready", os.getpid()))
            while True:
                message = connection.recv()
                if message is None:
                    return
                if message == "release-idle":
                    _release_idle_cache()
                    affinities = tuple(sorted({key[0] for key in _RESIDENT}))
                    retire = not (_RESIDENT or _SHARED_OWNED)
                    connection.send(("idle-released", (affinities, retire)))
                    if retire:
                        return
                    continue
                task, token, lane = message
                if isinstance(task, _PackedTask):
                    task = task.load()
                from . import cpu

                seat = seats[lane]
                cpu.set_pool_seat(seat)
                seat.begin_task(task.resources.min_cores, dispatched=True)
                # Failed native descendants cannot be interrupted by Python unwind.
                # The graph aborts and reaps this worker; keep its seat until then.
                reply = perform_task(task, token, PoolShareBudget(), keep_gathers=True)
                retire = policy.eligible(affinity=task.affinity is not None, retained=bool(_RESIDENT or _SHARED_OWNED))
                # Pack the reply while the task holds its CPUs, using its team plus idle lane CPUs.
                # Expanding chain steps return large child arrays; parallel copying keeps their
                # shared-memory transfer from holding up the contig.
                team = cpu.authority_threads(_task_peak_threads())
                packed = _pack_reply(replace(reply, retire=retire), copy_threads=team + cpu.idle_lane_cpus())
                seat.end_task()
                connection.send(("result", packed))
                # Idle workers retain only explicitly registered affinity state.
                del reply, packed, task, token, message
                _SHARED_UNCLOSED = _close_unreferenced(_SHARED_UNCLOSED)
                _retry_reply_closes()
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
# declared immutable at birth, core.immutable), with an exact dtype; other arrays travel by value.
# The transport never changes a caller's array. Tasks receive read-only arguments (_perform freezes them),
# so a shared view preserves their access semantics while avoiding O(tasks x bytes) coordinator pickling.
TRANSPORT_SHARE_BYTES = 8 << 20


def _plain_dtype(dtype: np.dtype) -> bool:
    """Whether ``dtype.str``, which a SharedArrayRef carries, names ``dtype`` exactly (no objects, fields, subarrays or
    metadata)."""
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None or dtype.metadata is not None:
        return False
    return np.dtype(dtype.str) == dtype


def _fixed(array: np.ndarray) -> bool:
    """Whether ``array`` may be shared by identity: its data provably can no longer change (it and every array it
    views are read-only, down to immutable ``bytes`` or an owner declared immutable at birth) and ``dtype.str``, which
    a SharedArrayRef carries, names its dtype exactly (no fields, subarrays or metadata). Read-only flags alone prove
    nothing: a view made before a freeze stays writable, and a read-only memoryview of a bytearray changes with it."""
    if not _plain_dtype(array.dtype):
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


# Arrays attached from a transport segment in this process: id -> (weak reference, reference, owner, byte address at
# attachment). Sending one
# again passes the same reference on without a copy, in either direction, while its segment is known to live: in the
# coordinator, ``owner`` is the transport that holds the segment (and must still be open); in a worker it is the task
# number during which the array was attached (its coordinator keeps that task's segments until the reply is in).
_ATTACHED: dict[int, tuple[weakref.ref, tuple, Any, int]] = {}
_TASK_NUMBER = 0  # in a graph worker, the number of the task it is running
# In a graph worker, the current task's attachments by reference (weak: a finished task's arrays are not kept).
_TASK_ATTACHED: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _attach(reference: SharedArrayRef | MappedArrayRef, transposed: bool, owner: Any) -> np.ndarray:
    """Attach a transport segment read-only, register the array for pass-through and return it."""
    global _SHARED_UNCLOSED
    _SHARED_UNCLOSED = _close_unreferenced(_SHARED_UNCLOSED)
    handle, array = _attach_array(reference)
    array.flags.writeable = False
    _SHARED_UNCLOSED.append((handle, weakref.ref(array)))  # unmapped once no array refers to it
    result = array.T if transposed else array
    _register_attached(array, _pid(reference, False), owner)
    if result is not array:
        _register_attached(result, _pid(reference, transposed), owner)
    if isinstance(owner, _ArrayTransport):
        # Every derived view keeps this actual base alive, including Fortran
        # views whose .base skips the originally returned transposed object.
        owner.hold(reference.name, array)
    return result


def _register_attached(array: np.ndarray, reference: tuple, owner: Any) -> None:
    key = id(array)

    def forget(_: weakref.ref, key: int = key) -> None:
        _ATTACHED.pop(key, None)

    _ATTACHED[key] = (weakref.ref(array, forget), reference, owner, int(array.__array_interface__["data"][0]))


def _pid(reference: SharedArrayRef | MappedArrayRef, transposed: bool) -> tuple:
    return ("shared-array", reference, transposed)


def _passed_through(array: np.ndarray, owner: Any) -> tuple | None:
    """The reference ``array`` was attached from, if it may be passed on: attached in this coordinator's transport
    (``owner``, still open) or, in a worker (``owner`` None), during the current task."""
    if not _plain_dtype(array.dtype) or not (array.flags.c_contiguous or array.flags.f_contiguous):
        return None
    address = int(array.__array_interface__["data"][0])
    item: Any = array
    while isinstance(item, np.ndarray):
        entry = _ATTACHED.get(id(item))
        if entry is not None and entry[0]() is item:
            reference = entry[1][1]
            extent = math.prod(reference.shape) * np.dtype(reference.dtype).itemsize
            live = entry[2] == _TASK_NUMBER if owner is None else entry[2] is owner and not owner.closed
            if live and address == entry[3] and array.nbytes == extent:
                transposed = not array.flags.c_contiguous
                contiguous = array.T if transposed else array
                return _pid(_array_reference(reference, contiguous), transposed)
        item = item.base
    return None


@dataclass(frozen=True)
class _PackedTask:
    """A task pickled by the coordinator with its large arrays replaced by shared-memory references."""

    payload: bytes

    def load(self) -> Task:
        global _TASK_NUMBER
        _TASK_NUMBER += 1
        _TASK_ATTACHED.clear()
        return cast(Task, _TransportUnpickler(io.BytesIO(self.payload), _TASK_NUMBER).load())


@dataclass(frozen=True)
class _PackedReply:
    """A worker's reply pickled with its large arrays as shared-memory references (_ReplyPickler)."""

    payload: bytes

    def load(self, transport: _ArrayTransport) -> WorkerReply:
        return cast(WorkerReply, _TransportUnpickler(io.BytesIO(self.payload), transport).load())


class _TransportPickler(ForkingPickler):
    def __init__(self, file: Any, transport: _ArrayTransport):
        super().__init__(file, pickle.HIGHEST_PROTOCOL)
        self.transport = transport
        self.names: set[str] = set()
        # The arrays referenced, alive until pack() has recorded their flight: a reducer may build temporary arrays,
        # whose segment would otherwise be released (the transport holds sources weakly) before then.
        self.kept: list[np.ndarray] = []

    def reducer_override(self, obj: Any) -> Any:
        # reducer_override, unlike persistent_id, is not called for atoms, strings or builtin containers: pickling
        # an object-heavy task costs about a third less (persistent_id ran Python code for every int and float).
        if isinstance(obj, np.ndarray) and obj.nbytes >= TRANSPORT_SHARE_BYTES:
            reference = _passed_through(obj, self.transport)
            if reference is None and (_fixed(obj) or _read_only_mapping(obj)):
                reference = self.transport.reference(obj)
            if reference is not None:
                self.names.add(reference[1].name)
                self.kept.append(obj)
                return _shared_array, reference
            self.transport.by_value(obj)
        return NotImplemented


class _ByValue(Exception):
    """A packed child would carry a large array by value (_packed_child then leaves the child unpacked)."""


class _ReplyPickler(ForkingPickler):
    """A graph worker's reply: arrays attached during this task pass their references back; any other large,
    contiguous, read-only array of a plain dtype is copied once into a new segment, which the coordinator adopts.
    The copy is taken while the reply is pickled, exactly as pickling by value would take it, and it arrives
    read-only as a read-only array sent by value would (results and arguments are read-only throughout, _readonly),
    so unlike the coordinator's identity cache (_fixed) it needs no proof that the array cannot change later. Each new
    segment's name is announced to the coordinator before the segment exists, so an abort unlinks it whatever
    happens. Every other array travels by value; in a packed child (``child``) a large one raises _ByValue instead,
    since each child is pickled on its own and would carry its own copy."""

    def __init__(
        self,
        file: Any,
        made: dict[tuple, tuple] | None = None,
        kept: list[np.ndarray] | None = None,
        *,
        child: bool = False,
        shared: frozenset[int] = frozenset(),
        copy_threads: int = 1,
    ):
        super().__init__(file, pickle.HIGHEST_PROTOCOL)
        self.copy_threads = copy_threads  # threads that copy a large array into its reply segment
        # Shared by every pickler of one reply, so an array its children all carry is copied into one segment.
        self.made: dict[tuple, tuple] = {} if made is None else made
        self.kept: list[np.ndarray] = [] if kept is None else kept  # the arrays keyed in ``made`` stay alive
        self.references: list[tuple] = []
        self.child = child
        self.shared = shared  # ids of smaller arrays that several packed calls of this reply carry (_shared_arrays)
        self.refs: dict[TaskKey, int] = {}  # a packed call's result references, by first appearance

    def reducer_override(self, obj: Any) -> Any:
        if self.child and type(obj) is ResultRef:
            self.refs.setdefault(obj.task, len(self.refs))
            return NotImplemented
        if isinstance(obj, np.ndarray) and (obj.nbytes >= TRANSPORT_SHARE_BYTES or id(obj) in self.shared):
            reference = _passed_through(obj, None)
            if (
                reference is None
                and not obj.flags.writeable
                and _plain_dtype(obj.dtype)
                and (obj.flags.c_contiguous or obj.flags.f_contiguous)
            ):
                span = (int(obj.__array_interface__["data"][0]), obj.nbytes)
                reference = self.made.get(span)
                if reference is None:
                    reference = self.made[span] = _reply_segment(obj, self.copy_threads)
                    # Retain the source for this whole reply, including all packed
                    # children: its address cannot be reused for unrelated bytes.
                    self.kept.append(obj)
                transposed = not obj.flags.c_contiguous
                reference = _pid(_array_reference(reference[1], obj.T if transposed else obj), transposed)
            if reference is None:
                if self.child:
                    raise _ByValue
                return NotImplemented
            self.references.append(reference)
            return _shared_array, reference
        return NotImplemented


# Reply segments whose mapping could not be closed yet (a view of it was still alive), closed at a later reply.
_REPLY_UNCLOSED: list[SharedMemory] = []


def _close_reply_handle(handle: SharedMemory) -> bool:
    try:
        handle.close()
        return True
    except BufferError:
        return False


def _retry_reply_closes() -> None:
    """Close unreferenced reply mappings before a new reply, after each send and at idle boundaries,
    so a worker needs no further large reply to release its mappings."""
    _REPLY_UNCLOSED[:] = [handle for handle in _REPLY_UNCLOSED if not _close_reply_handle(handle)]


def _reply_segment(array: np.ndarray, copy_threads: int = 1) -> tuple:
    """Copy ``array`` into a new segment for the coordinator to adopt, with ``copy_threads`` threads when it is large;
    If live views prevent closing the mapping, collect them and retry; retain any still-open mapping
    for a later retry. The coordinator adopts the segment by name independently of this handle's lifetime."""
    from .memory import segment_name

    _retry_reply_closes()
    transposed = not array.flags.c_contiguous
    for _ in range(64):
        name = segment_name()
        _announce_shared("shared+", (name,))
        try:
            handle = new_segment(array.nbytes, name=name, track=False)
        except FileExistsError:
            _announce_shared("shared-", (name,))
            continue
        try:
            return _pid(_fill_shared(array.T if transposed else array, handle, copy_threads, capped=False), transposed)
        finally:
            if not _close_reply_handle(handle):
                gc.collect()
                if not _close_reply_handle(handle):
                    _REPLY_UNCLOSED.append(handle)
    raise FileExistsError("no free shared-memory segment name")


def _pack_reply(reply: WorkerReply, *, copy_threads: int = 1) -> _PackedReply:
    made: dict[tuple, tuple] = {}
    kept: list[np.ndarray] = []
    if isinstance(reply.value, Expansion):
        expansion = reply.value
        shared = _shared_arrays((*expansion.subtasks, expansion.gather))
        children = tuple(_packed_child(task, made, kept, shared, copy_threads) for task in expansion.subtasks)
        gather = _packed_child(expansion.gather, made, kept, shared, copy_threads)
        reply = replace(reply, value=replace(expansion, subtasks=children, gather=gather))
    buffer = io.BytesIO()
    _ReplyPickler(buffer, made, kept, copy_threads=copy_threads).dump(reply)
    return _PackedReply(buffer.getvalue())


_KEPT_NUMBERS = count()


def _kept_gather(gather: Task, affinity: str) -> Task:
    """An affinity task's expansion gather, its call kept in this worker (``_RESIDENT``) rather than pickled.

    A gather shares its parent's affinity, so it runs on this worker, after the children; its arguments are the
    chain's continuation state (a selection round's blocks, previous selections and evidence; an assembly session),
    tens of megabytes that the coordinator never reads. Pickled into the reply, they crossed to the coordinator and
    back to this same worker at every hand-off. Kept here, the coordinator carries only the results the call reads
    (``refs``), sends them beside the call as for any packed call, and the worker cannot retire while it holds one."""
    from .graph import _references

    kwargs = dict(gather.kwargs)
    name = f"continuation/{next(_KEPT_NUMBERS)}"
    _RESIDENT[affinity, name] = (gather.function, gather.args, kwargs)
    refs = tuple(dict.fromkeys(_references((gather.args, kwargs))))
    return replace(gather, function=_call_packed, args=(PackedCall(b"", (), refs, kept=name),), kwargs={})


SHARED_CHILD_ARRAY_BYTES = 64 << 10  # smaller arrays several packed calls carry are copied into each


def _shared_arrays(tasks: tuple[Task, ...]) -> frozenset[int]:
    """Ids of the arrays of at least SHARED_CHILD_ARRAY_BYTES that two or more of ``tasks`` carry (through tuples,
    lists, mappings, dataclasses and object arrays): each packed call is pickled on its own, so such an array would
    otherwise be copied into every call that carries it (16 calls sharing 1 MiB made a 16 MiB reply)."""
    seen: dict[int, int] = {}
    for number, task in enumerate(tasks):
        visited: set[int] = set()
        pending: list[Any] = [task.args, task.kwargs]
        while pending:
            item = pending.pop()
            if isinstance(item, (str, bytes, int, float, complex, bool, type(None), np.generic)):
                continue
            if id(item) in visited:
                continue
            visited.add(id(item))
            if isinstance(item, np.ndarray):
                if item.dtype.hasobject:
                    pending.extend(item.flat)
                elif item.nbytes >= SHARED_CHILD_ARRAY_BYTES and seen.setdefault(id(item), number) != number:
                    seen[id(item)] = -1
                continue
            names = _field_names(item)
            if names is not None:
                pending.extend(getattr(item, name) for name in names)
            elif isinstance(item, (list, tuple)):
                pending.extend(item)
            elif isinstance(item, Mapping):
                pending.extend(item.values())
    return frozenset(key for key, number in seen.items() if number == -1)


def _packed_child(
    task: Task,
    made: dict[tuple, tuple],
    kept: list[np.ndarray],
    shared: frozenset[int] = frozenset(),
    copy_threads: int = 1,
) -> Task:
    """An expansion child or gather with its call pickled here once (_call_packed unpacks it in the worker that runs
    it and resolves its result references there, with the coordinator's own resolver). A task that would carry a
    large array by value stays as it is (the main reply carries such an array once for every child)."""
    if task.function is _call_packed:
        return task
    buffer = io.BytesIO()
    pickler = _ReplyPickler(buffer, made, kept, child=True, shared=shared, copy_threads=copy_threads)
    try:
        pickler.dump((task.function, task.args, dict(task.kwargs)))
    except _ByValue:
        return task
    names = tuple(sorted({reference[1].name for reference in pickler.references}))
    packed = PackedCall(buffer.getvalue(), names, tuple(pickler.refs))  # dict order: by index
    return replace(task, function=_call_packed, args=(packed,), kwargs={})


def _call_packed(packed: PackedCall, values: tuple = (), /, **extra: Any) -> Any:
    """Run a child's call packed by its expanding worker: ``values`` are the results its references name (in
    ``packed.refs`` order, sent by the coordinator), substituted with the coordinator's own resolver
    (graph.resolve_results: same traversal, rebuilt containers and dataclasses, literal mapping keys), and
    ``extra`` the engine's per-run arguments (its thread team)."""
    from .graph import resolve_results

    kept_key = None
    if packed.kept is None:
        function, args, kwargs = _TransportUnpickler(io.BytesIO(packed.payload), _TASK_NUMBER).load()
    else:
        try:
            if _ACTIVE_AFFINITY is None:
                raise KeyError(packed.kept)
            kept_key = (_ACTIVE_AFFINITY, packed.kept)
            function, args, kwargs = _RESIDENT[kept_key]
        except KeyError as error:
            raise InvariantError(f"kept continuation {packed.kept} is not on this worker") from error
    # Always, as the coordinator resolved every task (references or not): it rebuilds tuples, lists and mappings.
    results = dict(zip(packed.refs, values, strict=True))
    args, kwargs = resolve_results(args, results), resolve_results(kwargs, results)
    args, kwargs = _readonly((args, dict(kwargs, **extra)))
    from ..store.protocol import ClaimBusy
    from .cooperation import Deferred

    try:
        value = function(*args, **kwargs)
    except ClaimBusy:
        raise  # Retry the same kept continuation with the same original references.
    else:
        if kept_key is not None and not isinstance(value, Deferred):
            _RESIDENT.pop(kept_key)
        return value


def _shared_array(kind: str, reference: SharedArrayRef | MappedArrayRef, transposed: bool) -> np.ndarray:
    """A transport reference in a pickle: only _TransportUnpickler loads one (its find_class binds this name to the
    payload's own loader); any other unpickler reaching it is outside the transport."""
    raise pickle.UnpicklingError("a task-transport array reference can only be loaded by the task transport")


class _TransportUnpickler(pickle.Unpickler):
    """Attaches each shared-memory reference once per payload (a repeated array stays one object) and registers it
    for pass-through; in the coordinator (``owner`` a transport) the attached array also holds its segment."""

    def __init__(self, file: Any, owner: Any):
        super().__init__(file)
        self.owner = owner
        # In a worker, every payload of the current task shares one memo (its message and its packed call: an
        # array both carried and referenced stays one object, as when the coordinator pickled the whole task).
        self.attached: Any = _TASK_ATTACHED if type(owner) is int and owner == _TASK_NUMBER else {}

    def load(self) -> Any:
        """Load one complete payload and release its temporary object graph now.

        The pickle memo contains our bound shared-array loader, which points back
        to this unpickler. Without clearing it, that cycle also keeps every decoded
        array alive until cyclic GC, despite the transport's weak lifetime holds.
        """
        try:
            return super().load()
        finally:
            self.memo.clear()

    def find_class(self, module: str, name: str) -> Any:
        if module == __name__ and name == "_shared_array":
            return self._shared_array
        return super().find_class(module, name)

    def _shared_array(self, kind: str, reference: SharedArrayRef | MappedArrayRef, transposed: bool) -> np.ndarray:
        if kind != "shared-array":
            raise pickle.UnpicklingError(f"unknown task transport reference {kind!r}")
        key = (reference, transposed)
        if key in self.attached:
            return self.attached[key]
        result = _attach(reference, transposed, self.owner)
        self.attached[key] = result
        return result


class _Segment:
    """One transport segment in the coordinator: its handle (None when adopted from a worker), the arrays here that
    refer to it (sources or attached views) and the tasks in flight that carry it."""

    __slots__ = ("flight", "handle", "holders")

    def __init__(self, handle: SharedMemory | _FilePin | None):
        self.handle = handle
        self.holders: dict[int, weakref.ref] = {}  # by the reference's id (arrays are not hashable)
        self.flight = 0


class _ArrayTransport:
    """The coordinator's shared copies of large immutable task arrays (TRANSPORT_SHARE_BYTES, _fixed).

    A segment is either created here (compatible contiguous source views reuse
    that backing) or adopted from a worker's reply. It lives while an array here refers to it (a
    source, or an array attached from it) or a task in flight carries it, and is unlinked when neither remains, so
    each round's inputs leave memory with the round. A segment created here is registered before it is filled, so
    close() unlinks it whatever interrupts the copy.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.closed = False
        self.sources: dict[tuple, tuple[dict[int, weakref.ref], tuple]] = {}
        self.segments: dict[str, _Segment] = {}
        self.filling: dict[str, SharedMemory | None] = {}  # by name, from before its creation until its entry
        self.in_flight: dict[int, set[str]] = {}
        self.copied: dict[tuple, list[int]] = {}  # large arrays sent by value: (shape, dtype) -> [tasks, bytes]

    def pack(self, index: int, task: Task) -> _PackedTask:
        buffer = io.BytesIO()
        pickler = _TransportPickler(buffer, self)
        pickler.dump(task)
        with self.lock:
            for name in pickler.names:
                self.segments[name].flight += 1
            self.in_flight[index] = pickler.names
        return _PackedTask(buffer.getvalue())

    def receive(self, reply: Any) -> Any:
        """A worker's reply, its shared-memory references attached and held here; an expansion's packed children
        hold the segments their calls refer to until the children finish (their arguments are then dropped)."""
        if not isinstance(reply, _PackedReply):
            return reply
        reply = reply.load(self)
        if isinstance(reply.value, Expansion):
            for child in (*reply.value.subtasks, reply.value.gather):
                for packed in child.args:
                    if isinstance(packed, PackedCall):
                        for name in packed.names:
                            self.hold(name, packed)
        return reply

    def done(self, index: int) -> None:
        with self.lock:
            for name in self.in_flight.pop(index, ()):
                segment = self.segments.get(name)
                if segment is not None:
                    segment.flight -= 1
                    self._release_if_unused(name)

    def by_value(self, array: np.ndarray) -> None:
        """Count a large array that travels by value (writable, viewing writable data, strided or with an inexact
        dtype): close() logs them, so producers that should publish immutable arrays can be found."""
        with self.lock:
            entry = self.copied.setdefault((array.shape, array.dtype.str), [0, 0])
            entry[0] += 1
            entry[1] += array.nbytes

    def hold(self, name: str, array: Any) -> None:
        """``array`` (here: an attached array, or a packed call that refers to the segment) refers to segment
        ``name``: adopt it if a worker created it, and keep it while ``array`` lives."""
        with self.lock:
            segment = self.segments.get(name)
            if segment is None:
                segment = self.segments[name] = _Segment(None)
            holder = weakref.ref(array, functools.partial(self._released, name))
            segment.holders[id(holder)] = holder

    def reference(self, array: np.ndarray) -> tuple | None:
        if not (array.flags.c_contiguous or array.flags.f_contiguous):
            return None
        key = (int(array.__array_interface__["data"][0]), array.nbytes)
        with self.lock:
            entry = self.sources.get(key)
            transposed = not array.flags.c_contiguous
            contiguous = array.T if transposed else array
            if entry is not None:
                reference = _array_reference(entry[1][1], contiguous)
                self._source_hold(key, entry[0], array, reference.name)
                return _pid(reference, transposed)
            source = _mapped_source(array)
            if source is None and not _fixed(array):
                return None
            handle = self._new_segment(array.nbytes) if source is None else _pin_file(source)
            if handle is None:
                return None  # the ordinary by-value route still supports non-reopenable mappings
            try:
                if source is None:
                    reference = _fill_shared(contiguous, cast(SharedMemory, handle), 1)
                else:
                    reference = MappedArrayRef(handle.name, source[2], tuple(contiguous.shape), array.dtype.str)
                pid = _pid(reference, transposed)
                self.segments[handle.name] = _Segment(handle)
                sources: dict[int, weakref.ref] = {}
                self.sources[key] = (sources, pid)
                self._source_hold(key, sources, array, handle.name)
            except BaseException:
                self.sources.pop(key, None)
                self.segments.pop(handle.name, None)
                _unlink_segment(handle.name, handle)
                raise
            finally:
                self.filling.pop(handle.name, None)
            return pid

    def _source_hold(self, key: tuple, sources: dict, array: np.ndarray, name: str) -> None:
        if id(array) in sources:
            return
        sources[id(array)] = weakref.ref(array, functools.partial(self._source_dropped, key, id(array)))
        self.hold(name, array)

    def _new_segment(self, size: int) -> SharedMemory:
        """A new segment, its name recorded before it exists: whatever interrupts its creation, close() unlinks it."""
        from .memory import segment_name

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

    def _source_dropped(self, key: tuple, identity: int, _: Any) -> None:
        with self.lock:
            entry = self.sources.get(key)
            if entry is not None:
                entry[0].pop(identity, None)
                if not entry[0]:
                    del self.sources[key]

    def _released(self, name: str, holder: weakref.ref) -> None:
        with self.lock:
            segment = self.segments.get(name)
            if segment is not None:
                segment.holders.pop(id(holder), None)
                self._release_if_unused(name)

    def _release_if_unused(self, name: str) -> None:
        segment = self.segments[name]
        if segment.holders or segment.flight > 0:
            return
        del self.segments[name]
        _unlink_segment(name, segment.handle)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            segments = list(self.segments.items())
            handles = [handle for handle in self.filling.values() if handle is not None]
            unopened = [name for name, handle in self.filling.items() if handle is None]
            copied = sorted(self.copied.items(), key=lambda item: -item[1][1])
            self.sources.clear()
            self.segments.clear()
            self.filling.clear()
            self.in_flight.clear()
            self.copied.clear()
        for name, segment in segments:
            _unlink_segment(name, segment.handle)
        close_shared_memory(handles, unlink=True)
        for name in unopened:  # interrupted between its name and its handle: unlink it if it was created
            _unlink_segment(name, None)
        if copied:
            gib = 1 << 30
            largest = "; ".join(f"{shape} {dtype} x{tasks}" for (shape, dtype), (tasks, _) in copied[:3])
            _LOG.info(
                f"[transport] {sum(tasks for _, (tasks, _) in copied)} large task arrays "
                f"({sum(size for _, (_, size) in copied) / gib:.2f} GiB) travelled by value: {largest}"
            )


def _unlink_segment(name: str, handle: SharedMemory | _FilePin | None) -> None:
    """Close and unlink a transport segment by handle, or by name when it was adopted (best effort)."""
    if handle is None:
        _unlink_name(name)
        return
    if isinstance(handle, _FilePin):
        handle.close()
    else:
        close_shared_memory([handle], unlink=True)


def _unlink_name(name: str) -> None:
    """Unlink a shared-memory name without mapping it (best effort): a segment interrupted between its creation and
    its sizing is empty, and mapping it fails."""
    try:
        import _posixshmem
    except ImportError:  # no POSIX shared memory: only a mapped handle unlinks
        try:
            close_shared_memory([SharedMemory(name=name, create=False, track=False)], unlink=True)
        except (OSError, ValueError):
            pass
        return
    try:
        _posixshmem.shm_unlink(name if name.startswith("/") else "/" + name)
    except OSError:
        pass


@dataclass
class _WorkerHandle:
    process: Any
    connection: Any
    started: int | None
    busy: TaskKey | None = None
    ready: bool = False
    announced: bool = False
    idle_released: bool = False


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
        from .lanes import LaneAuthority

        self.authority = LaneAuthority(self.context, cores, lanes, processes, chains)
        self.settings = bootstrap.worker_settings()
        self.threshold = recycle_threshold(processes, recycle_bytes)
        self.handles: list[_WorkerHandle] = []
        self.failed_task: TaskKey | None = None  # attributed only while receive processes this worker
        # Each worker's announced shared-value segments, unlinked by close() whatever became of the worker.
        self.shared: dict[int, set[str]] = {}
        self.released: list[tuple[int, tuple[str, ...]]] = []
        self.transport = _ArrayTransport()
        # Coordinator traffic by (direction, stage): [messages, payload bytes, pickling or unpickling seconds, pipe
        # receive seconds], logged when the group closes (where the coordinator's time goes).
        self.traffic: dict[tuple[str, str], list[float]] = {}
        self.affinities: dict[str, int] = {}
        if static_affinities:
            from .lanes import partition_chains

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
        from .memory import process_start_ticks

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
        handle = _WorkerHandle(process, parent, process_start_ticks(process.pid))
        if index == len(self.handles):
            self.handles.append(handle)
        else:
            self.handles[index] = handle

    def free_worker(self, affinity: str | None, reserved: Container[int] = (), own: Container[int] = ()) -> int | None:
        """An idle worker for a task of ``affinity``: its pinned worker, else the first idle one outside
        ``reserved`` (the workers a pinned chain or a pending gather will need), then one reserved for the task's own
        chain or gather (``own``: what waits there waits for this task anyway), and another chain's only when no
        other is idle. Giving the chain's worker other work can delay its next step. A
        new affinity is pinned to it only when a task is sent (pin), not when admission may still refuse the task."""
        if affinity in self.affinities:
            index = self.affinities[affinity]
            return index if self.handles[index].ready and self.handles[index].busy is None else None
        mine = other = None
        for index, handle in enumerate(self.handles):
            if handle.ready and handle.busy is None:
                if index not in reserved:
                    return index
                if index in own:
                    mine = index if mine is None else mine
                elif other is None:
                    other = index
        return mine if mine is not None else other

    def pin(self, affinity: str | None, index: int) -> None:
        """Pin ``affinity`` to worker ``index`` as its task is sent (its later tasks run there)."""
        if affinity is not None:
            self.affinities.setdefault(affinity, index)

    def claim(self, index: int, key: TaskKey) -> None:
        """Take idle worker ``index`` for ``key``, which send() then transmits: a pass decides all its tasks before
        sending any (free_worker no longer offers a claimed worker)."""
        handle = self.handles[index]
        if handle.busy is not None or not handle.ready:
            raise InvariantError("worker already has a task or has not initialized")
        handle.busy = key

    def send(self, index: int, task: Task, token: tuple[int, int], *, lane: int = 0) -> None:
        handle = self.handles[index]
        if handle.busy != task.key:
            raise InvariantError("a task is sent only to the worker claimed for it")
        began = time.perf_counter()
        packed = self.transport.pack(index, task)
        self._count("task", task.key.stage, len(packed.payload), time.perf_counter() - began, 0.0)
        handle.connection.send((packed, token, lane))

    def reclaim_attachments(self) -> None:
        """Unmap dead coordinator views after consuming a reply, not at a future attach.

        Unlinking a segment frees its name; the pages stay charged until the final
        mapping closes. Live arrays are protected by the weak-view lifetime check.
        """
        global _SHARED_UNCLOSED
        _SHARED_UNCLOSED = _close_unreferenced(_SHARED_UNCLOSED)

    def release_idle(self) -> None:
        """Pressure recovery: release reloadable caches and native heaps on idle workers.

        One request per task boundary, so permanent pressure cannot cause a respawn
        loop or fake watchdog progress. Workers holding explicit state trim but do
        not retire. A maintenance worker is unavailable until its acknowledgement.
        """
        for handle in self.handles:
            if handle.ready and handle.busy is None and not handle.idle_released:
                handle.ready = False
                handle.idle_released = True
                handle.connection.send("release-idle")

    def _retired(self, index: int, *, idle: bool = False) -> None:
        handle = self.handles[index]
        handle.announced = True
        handle.process.join(timeout=1.0)
        if handle.process.is_alive() or handle.process.exitcode != 0:
            raise RuntimeError("announced worker retirement did not finish cleanly")
        handle.connection.close()
        self.shared.pop(index, None)
        self._spawn(index)
        self.handles[index].idle_released = idle

    def _count(self, direction: str, stage: str, size: int, seconds: float, receiving: float) -> None:
        row = self.traffic.get((direction, stage))
        if row is None:
            row = self.traffic[direction, stage] = [0, 0, 0.0, 0.0]
        row[0] += 1
        row[1] += size
        row[2] += seconds
        row[3] += receiving

    def receive(self, timeout: float = 0.2) -> list[tuple[int, WorkerReply]]:
        ready = wait([h.connection for h in self.handles], timeout)
        replies = []
        for index, handle in enumerate(self.handles):
            self.failed_task = handle.busy
            if handle.connection in ready:
                began = time.perf_counter()
                try:
                    kind, value = handle.connection.recv()
                except (EOFError, OSError) as error:
                    # Pipe EOF can precede the forkserver's exit-status notification.
                    # Bound this diagnostic wait even if the worker only closed its pipe.
                    handle.process.join(timeout=0.2)
                    status = handle.process.exitcode
                    raise BrokenProcessPool(
                        f"worker {handle.process.pid} exited unexpectedly with status "
                        f"{status if status is not None else 'unknown'}"
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
                elif kind == "idle-released":
                    affinities, retire = value
                    handle.ready = True
                    self.released.append((index, affinities))
                    if retire:
                        self._retired(index, idle=True)
                elif kind == "result":
                    received = time.perf_counter()
                    size = len(value.payload) if isinstance(value, _PackedReply) else 0
                    value = self.transport.receive(value)
                    self._count("reply", value.key.stage, size, time.perf_counter() - received, received - began)
                    if value.key != handle.busy:
                        raise InvariantError("worker replied for a different task")
                    handle.busy = None
                    handle.idle_released = False
                    self.transport.done(index)
                    replies.append((index, value))
                    if value.retire:
                        if value.resident_affinities:
                            raise InvariantError("worker attempted to recycle live resident state")
                        self._retired(index)
                else:
                    raise InvariantError(f"unknown worker message: {kind}")
            current = self.handles[index]
            if current.process.exitcode is not None and (not current.announced):
                if current.connection.poll():
                    continue
                raise BrokenProcessPool(
                    f"worker {current.process.pid} exited unexpectedly with status {current.process.exitcode}"
                )
        self.failed_task = None
        return replies

    def close(self, *, abort: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        for group in self.authority.groups:
            group.closed.get_obj().value = 1
        from .memory import process_alive

        if not abort:
            for handle in self.handles:
                try:
                    handle.connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
        for handle in self.handles:
            process = handle.process
            if abort and handle.started is not None and process_alive(process.pid, handle.started):
                process.terminate()
        deadline = time.monotonic() + GROUP_TERMINATE_GRACE_SECONDS
        for handle in self.handles:
            handle.process.join(timeout=max(0.0, deadline - time.monotonic()))
        forced = 0
        for handle in self.handles:
            if handle.started is not None and process_alive(handle.process.pid, handle.started):
                handle.process.kill()
                forced += 1
            handle.process.join(timeout=1.0)
        if forced and abort:
            _LOG.warning(
                "%d worker(s) did not stop within %.0f s and were killed; the resource tracker reclaims "
                "anything they still owned",
                forced,
                GROUP_TERMINATE_GRACE_SECONDS,
            )
        try:
            for index, handle in enumerate(self.handles):
                self._drain_shared(index, handle)
                handle.connection.close()
            self._unlink_shared()
        finally:
            self.transport.close()
        if self.traffic:
            rows = sorted(self.traffic.items(), key=lambda item: -(item[1][2] + item[1][3]))
            _LOG.info(
                "[transport] coordinator traffic (messages, MiB, pickle s, receive s): "
                + "; ".join(
                    f"{direction} {stage} {int(n)} {size / (1 << 20):.0f} {pickled:.1f} {receiving:.1f}"
                    for (direction, stage), (n, size, pickled, receiving) in rows[:12]
                )
            )

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
            _unlink_name(name)

    def __enter__(self) -> WorkerGroup:
        return self

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        self.close(abort=kind is not None)


def release_local_state() -> None:
    """Release resident values and shared backing when an in-process graph closes."""
    _RESIDENT.clear()
    _release_all_shared()
