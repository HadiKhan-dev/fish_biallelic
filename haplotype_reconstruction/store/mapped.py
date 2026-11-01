"""Temporary file-backed arrays with explicit eviction of completed contiguous windows.

The ndarray owns its mmap; the temporary file lives on the checkpoint root's filesystem and is already unlinked
while open. No checkpoint or scratch pathname survives a crash. Eviction flushes dirty pages first, so values
remain available to ordinary readers.
"""

from __future__ import annotations

import mmap
import os
import tempfile
import weakref
from typing import BinaryIO

import numpy as np

WINDOW_BYTES = 16 << 20
_FILES: weakref.WeakKeyDictionary[mmap.mmap, BinaryIO] = weakref.WeakKeyDictionary()


def mapped_array(shape, dtype, fill=0, *, root):
    """A C ndarray backed by an anonymous file under the checkpoint root, initialized in bounded byte windows."""
    dtype = np.dtype(dtype)
    size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if not size:
        return np.empty(shape, dtype=dtype)
    os.makedirs(root, exist_ok=True)
    file = tempfile.TemporaryFile(dir=root)
    try:
        file.truncate(size)
        backing = mmap.mmap(file.fileno(), size)
    except BaseException:
        file.close()
        raise
    _FILES[backing] = file
    weakref.finalize(backing, file.close)
    array = np.ndarray(shape, dtype=dtype, buffer=backing)
    if fill != 0:
        flat = array.reshape(-1)
        step = max(1, WINDOW_BYTES // dtype.itemsize)
        for start in range(0, len(flat), step):
            part = flat[start : start + step]
            part.fill(fill)
            discard_pages(part)
    return array


def discard_pages(value):
    """Flush and evict a contiguous view of one of our temporary mapped arrays; ordinary arrays are untouched."""
    owner = value
    while isinstance(owner, (np.ndarray, memoryview)):
        owner = owner.base if isinstance(owner, np.ndarray) else owner.obj
    if not isinstance(owner, mmap.mmap) or owner not in _FILES:
        return
    view = np.asarray(value) if isinstance(value, np.ndarray) else np.frombuffer(value, dtype=np.uint8)
    if not view.nbytes or not (view.flags.c_contiguous or view.flags.f_contiguous):
        return
    base = np.frombuffer(owner, dtype=np.uint8)
    offset = view.ctypes.data - base.ctypes.data
    start = offset // mmap.PAGESIZE * mmap.PAGESIZE
    stop = min(len(owner), ((offset + view.nbytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE)
    owner.flush(start, stop - start)
    owner.madvise(mmap.MADV_DONTNEED, start, stop - start)
    # Drop the clean file cache too: removing PTEs alone reduces RSS but can leave a chromosome charged to the job.
    os.posix_fadvise(_FILES[owner].fileno(), start, stop - start, os.POSIX_FADV_DONTNEED)


__all__ = ["mapped_array", "discard_pages", "WINDOW_BYTES"]
