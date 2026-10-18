"""NumPy arrays in POSIX shared memory: created once by a parent, attached (no copy) by its workers.

Segments are named after the run (execution.memory.segment_name: uid, the driver's pid and start time), so a run
killed before it unlinked them (e.g. at the end of an allocation) does not leave their RAM in /dev/shm for later
runs on the node: the next run's memory governor removes them. Every process holding a segment open marks it
(memory.mark_in_use), so a segment still in use is never removed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from multiprocessing.shared_memory import SharedMemory

import numba
import numpy as np

# First-touch copies into a new segment saturate memory bandwidth at about 16 threads
# (512 MiB: 16 threads 0.058 s, 76 threads 0.087 s).
COPY_THREADS = 16

_PARALLEL_COPY_BYTES = 4 << 20


@dataclass(frozen=True)
class SharedArrayRef:
    """What a worker needs to attach a shared array: segment name, shape and dtype (string)."""

    name: str
    shape: tuple[int, ...]
    dtype: str


def new_segment(size: int) -> SharedMemory:
    """A new segment of ``size`` bytes (at least one) named after the run (see the module docstring)."""
    from .memory import mark_in_use, segment_name
    for _ in range(64):
        try:
            handle = SharedMemory(name=segment_name(), create=True, size=max(int(size), 1))
        except FileExistsError:        # a random suffix collided: draw another
            continue
        mark_in_use(getattr(handle, "_fd", -1))      # held while the handle is open (see memory._remove_stale)
        return handle
    raise FileExistsError("no free shared-memory segment name")


@numba.njit(cache=True, parallel=True)
def _copy_shared_bytes(source, destination):
    chunk = 1 << 20
    for part in numba.prange((source.size + chunk - 1) // chunk):
        start = part * chunk
        stop = min(source.size, start + chunk)
        destination[start:stop] = source[start:stop]


def create_shared_array(array, *, copy_threads: int = 1) -> tuple[SharedMemory, SharedArrayRef]:
    """Copy ``array`` (C-contiguous) into a new segment; return (handle, reference).

    POSIX shared memory has no zero-byte segments, so an empty array gets a one-byte segment and keeps its shape
    and dtype. ``copy_threads`` (capped at COPY_THREADS) parallelises the first-touch copy of large arrays. If
    anything fails after allocation, the segment is closed and unlinked at once.
    """
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        raise TypeError("object arrays cannot be transported by shared memory")
    handle = new_segment(contiguous.nbytes)
    try:
        threads = min(int(copy_threads), COPY_THREADS)
        if threads > 1 and contiguous.nbytes >= _PARALLEL_COPY_BYTES:
            from .budget import numba_threads
            with numba_threads(threads):
                _copy_shared_bytes(np.frombuffer(memoryview(contiguous), dtype=np.uint8),
                                   np.frombuffer(handle.buf, dtype=np.uint8, count=contiguous.nbytes))
        else:
            np.copyto(np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=handle.buf), contiguous)
        return handle, SharedArrayRef(handle.name, tuple(contiguous.shape), contiguous.dtype.str)
    except BaseException:
        close_shared_memory([handle], unlink=True)
        raise


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
        return handle, SharedArrayRef(handle.name, shape, dtype.str), np.ndarray(shape, dtype=dtype,
                                                                                  buffer=handle.buf)
    except BaseException:
        close_shared_memory([handle], unlink=True)
        raise


def attach_shared_array(ref: SharedArrayRef) -> tuple[SharedMemory, np.ndarray]:
    """Attach to a segment created by create_shared_array; return (handle, array view)."""
    from .memory import mark_in_use
    dtype = np.dtype(ref.dtype)
    handle = SharedMemory(name=ref.name, create=False)
    mark_in_use(getattr(handle, "_fd", -1))
    try:
        if math.prod(ref.shape) * dtype.itemsize > len(handle.buf):
            raise ValueError("shared-array reference describes more bytes than the segment")
        return handle, np.ndarray(ref.shape, dtype=dtype, buffer=handle.buf)
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
