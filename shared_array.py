"""Small, process-safe primitives for numpy arrays in shared memory.

The creating process owns each segment and is responsible for unlinking it
after every worker has detached.  Attaching processes only close their local
handles.  Metadata using either of the repository's historical name keys
(``shm_name`` or ``name``) is accepted so existing worker initializers remain
compatible.
"""

from contextlib import contextmanager
import math
import operator
from multiprocessing.shared_memory import SharedMemory

import numpy as np


def _array_description(metadata):
    """Return validated ``(name, shape, dtype)`` from shared-array metadata."""
    shm_name = metadata.get("shm_name")
    name = metadata.get("name")
    if shm_name is not None and name is not None and shm_name != name:
        raise ValueError("shared-array metadata contains conflicting names")
    segment_name = shm_name if shm_name is not None else name
    if not isinstance(segment_name, str) or not segment_name:
        raise ValueError("shared-array metadata needs 'shm_name' or 'name'")

    try:
        shape = tuple(operator.index(size) for size in metadata["shape"])
    except (KeyError, TypeError) as exc:
        raise ValueError("shared-array metadata has an invalid shape") from exc
    if any(size < 0 for size in shape):
        raise ValueError("shared-array shape dimensions must be non-negative")

    try:
        dtype = np.dtype(metadata["dtype"])
    except (KeyError, TypeError) as exc:
        raise ValueError("shared-array metadata has an invalid dtype") from exc
    if dtype.hasobject:
        raise TypeError("object arrays cannot be transported by shared memory")
    return segment_name, shape, dtype


def create_shared_array(array, *, name_key="shm_name", dtype_as_string=True):
    """Copy an array into a new segment and return ``(handle, metadata)``.

    Non-contiguous inputs are copied to C-contiguous storage.  POSIX shared
    memory does not allow zero-byte segments, so empty arrays receive a
    one-byte backing segment while retaining their original shape and dtype.
    If setup or copying fails after allocation, the new segment is immediately
    closed and unlinked.
    """
    if name_key not in ("shm_name", "name"):
        raise ValueError("name_key must be 'shm_name' or 'name'")
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        raise TypeError("object arrays cannot be transported by shared memory")

    handle = SharedMemory(create=True, size=max(int(contiguous.nbytes), 1))
    try:
        shared_view = np.ndarray(
            contiguous.shape, dtype=contiguous.dtype, buffer=handle.buf
        )
        np.copyto(shared_view, contiguous)
        metadata = {
            name_key: handle.name,
            "shape": tuple(contiguous.shape),
            "dtype": (
                str(contiguous.dtype) if dtype_as_string else contiguous.dtype
            ),
        }
        return handle, metadata
    except BaseException:
        close_shared_memory([handle], unlink=True)
        raise


def attach_shared_array(metadata):
    """Attach to a segment described by either supported metadata format."""
    segment_name, shape, dtype = _array_description(metadata)
    handle = SharedMemory(name=segment_name, create=False)
    try:
        required_bytes = math.prod(shape) * dtype.itemsize
        if required_bytes > len(handle.buf):
            raise ValueError(
                "shared-array metadata describes more bytes than the segment"
            )
        array = np.ndarray(shape, dtype=dtype, buffer=handle.buf)
        return handle, array
    except BaseException:
        handle.close()
        raise


def close_shared_memory(handles, *, unlink=False):
    """Close handles, optionally unlinking their segments, best-effort."""
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


@contextmanager
def shared_memory_cleanup(handles):
    """Unlink parent-owned segments after the protected lifetime ends."""
    try:
        yield
    finally:
        close_shared_memory(handles, unlink=True)
