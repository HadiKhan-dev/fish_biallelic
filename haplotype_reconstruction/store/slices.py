"""Checked windows of C-contiguous numeric arrays in an existing codec payload.

Only metadata is decoded at open. A gather reads contiguous sample rows at selected columns (and all trailing
axes). Each needed chunk is decoded in full and sliced with NumPy: no library partial-range decoder is used.
The decoded-chunk cache is bounded, shared by the selected arrays, and discarded when the reader closes.
"""

from __future__ import annotations

import bisect
import tempfile
import zlib
from collections import OrderedDict

import numpy as np

from . import codec

# At most one codec-sized chunk's worth of decoded data; one oversized chunk is allowed alone.
# A miss additionally holds its compressed input while decoding, as the full codec reader does.

CACHE_BYTES = codec._CHUNK_BYTES


class WindowArray:
    """One C-contiguous array's shape/dtype and bounded gathers over its first two axes.

    Returned arrays own their storage. The source reader must remain open throughout a gather.
    """

    def __init__(self, reader, shape, dtype, group, offset):
        self.reader, self.shape, self.dtype = reader, tuple(shape), np.dtype(dtype)
        self.group, self.offset = group, offset

    def __len__(self):
        return self.shape[0]

    def sample_tiles(self, samples, maximum):
        """Contiguous sample tiles whose full rows share one physical chunk.

        A row crossing a chunk boundary stands alone, so its site windows walk
        forward through chunks. This is an I/O traversal choice, not a sample cap.
        """
        first, stop, step = samples.indices(self.shape[0])
        if step != 1:
            raise ValueError("window reads require a contiguous sample slice")
        row_bytes = self.shape[1] * self.dtype.itemsize * int(np.prod(self.shape[2:], dtype=np.int64))
        if not row_bytes:
            if first < stop:
                yield slice(first, stop)
            return
        chunks, starts = self.reader.groups[self.group]
        while first < stop:
            begin = self.offset + first * row_bytes
            index = bisect.bisect_right(starts, begin) - 1
            end = starts[index] + chunks[index][0]
            rows = max(1, (end - begin) // row_bytes)
            last = min(stop, first + min(max(1, maximum), rows))
            yield slice(first, last)
            first = last

    def gather(self, samples, columns):
        """Contiguous sample slice x increasing column indices x all trailing axes, in C order."""
        if self.reader.closed:
            raise ValueError("window reader is closed")
        start, stop, step = samples.indices(self.shape[0])
        if step != 1:
            raise ValueError("window reads require a contiguous sample slice")
        columns = np.asarray(columns, dtype=np.int64)
        result = np.empty((max(0, stop - start), len(columns), *self.shape[2:]), dtype=self.dtype)
        if not result.nbytes:
            return result
        left, right = int(columns[0]), int(columns[-1]) + 1
        contiguous = right - left == len(columns)
        cell_bytes = self.dtype.itemsize * int(np.prod(self.shape[2:], dtype=np.int64))
        row_bytes = self.shape[1] * cell_bytes
        scratch = None if contiguous else np.empty((right - left, *self.shape[2:]), dtype=self.dtype)
        indices = columns - left
        for row, sample in enumerate(range(start, stop)):
            target = result[row] if contiguous else scratch
            begin = self.offset + sample * row_bytes + left * cell_bytes
            self.reader.read_range(self.group, begin, begin + target.nbytes, memoryview(target).cast("B"))
            if not contiguous:
                np.take(scratch, indices, axis=0, out=result[row])
        return result


class ArrayReader:
    """Arrays of a dict payload, read through one open handle (owned by CheckpointFile.open_arrays).

    The format is codec's existing checked chunks. A miss reads/checks and fully decodes one compressed chunk.
    The LRU is charged by decoded bytes, evicting before allocation; one oversized chunk may stand alone.
    The compressed input is transient. Decoding uses one Blosc thread; callers parallelise numerical work after
    gathering, without nested I/O teams.
    """

    def __init__(self, handle, start, end, metadata_offset, expected_buffers, keys, nthreads, context, *, path=()):
        self.handle, self.context = handle, context
        self.closed = False
        self.groups, self.cache, self.cache_bytes = [], OrderedDict(), 0
        handle.seek(start)
        buffers, locations = [], []
        for descriptors, offsets, total, n_chunks in codec._buffer_group_headers(
            handle, metadata_offset, expected_buffers
        ):
            chunks = codec._chunk_layout(handle, total, n_chunks, metadata_offset, "buffer group")
            raw_starts, position = [], 0
            for chunk in chunks:
                raw_starts.append(position)
                position += chunk[0]
            group = len(self.groups)
            self.groups.append((chunks, raw_starts))
            for (size, _readonly), offset in zip(descriptors, offsets, strict=False):
                buffers.append(np.empty(size, dtype=np.uint8))
                locations.append((group, offset))
        with tempfile.SpooledTemporaryFile(max_size=codec._METADATA_SPOOL_BYTES, mode="w+b") as metadata:
            codec._spool_metadata(handle, end, nthreads, metadata)
            payload = codec._select_path(codec._unpickle(metadata, buffers), path, context)
        spans = sorted(
            (buffer.ctypes.data, buffer.nbytes, index) for index, buffer in enumerate(buffers) if buffer.nbytes
        )
        starts = [address for address, _, _ in spans]
        self.arrays = {}
        for key in keys:
            array = payload[key]
            if (
                not isinstance(array, np.ndarray)
                or array.dtype.hasobject
                or array.ndim < 2
                or not array.flags.c_contiguous
            ):
                raise TypeError(f"{context}: window item {key!r} must be a C-contiguous numeric array with >=2 axes")
            group, offset = None, 0
            if array.nbytes:
                span = bisect.bisect_right(starts, array.ctypes.data) - 1
                if span < 0 or array.ctypes.data + array.nbytes > spans[span][0] + spans[span][1]:
                    raise ValueError(f"{context}: window item {key!r} has no array buffer")
                address, _, index = spans[span]
                group, offset = locations[index]
                offset += array.ctypes.data - address
            self.arrays[key] = WindowArray(self, array.shape, array.dtype, group, offset)

    def __getitem__(self, key):
        return self.arrays[key]

    def close(self):
        self.closed = True
        self.cache.clear()
        self.cache_bytes = 0

    def _chunk(self, group, index):
        """Return owned decoded bytes, using only the codec's full-chunk primitive."""
        key = (group, index)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        raw_size, compressed_size, checksum, payload_offset = self.groups[group][0][index]
        while self.cache and self.cache_bytes + raw_size > CACHE_BYTES:
            _, discarded = self.cache.popitem(last=False)
            self.cache_bytes -= discarded.nbytes
            del discarded
        self.handle.seek(payload_offset)
        blob = codec._read_exact(self.handle, compressed_size, "window chunk")
        if zlib.crc32(blob) != checksum:
            raise ValueError(f"{self.context}: corrupt checkpoint: window chunk checksum mismatch")
        raw = np.empty(raw_size, dtype=np.uint8)
        codec._decompress_into(blob, raw, 1, "window chunk")
        self.cache[key] = raw
        self.cache_bytes += raw.nbytes
        return raw

    def read_range(self, group, start, stop, destination):
        chunks, starts = self.groups[group]
        offset = 0
        while start < stop:
            index = bisect.bisect_right(starts, start) - 1
            end = min(stop, starts[index] + chunks[index][0])
            left, right = start - starts[index], end - starts[index]
            raw = self._chunk(group, index)
            destination[offset : offset + end - start] = raw[left:right]
            del raw
            offset += end - start
            start = end


__all__ = ["WindowArray", "ArrayReader", "CACHE_BYTES"]
