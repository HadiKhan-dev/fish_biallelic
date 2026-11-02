"""Checked windows of C-contiguous numeric arrays in an existing codec payload.

Only metadata is decoded at open. A gather reads contiguous sample rows at selected columns (and all trailing
axes). Blosc2's partial decompression writes the requested bytes; it never materialises a whole raw buffer group.
The compressed-chunk cache is bounded, shared by the selected arrays, and discarded when the reader closes.
"""

from __future__ import annotations

import bisect
import tempfile
import zlib
from collections import OrderedDict

import numpy as np

from . import codec

# At most one codec-sized chunk's worth of compressed data; one oversized chunk is allowed alone.
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

    The format is codec's existing checked chunks. A miss reads/checks one compressed chunk before retaining an
    SChunk copy; that input copy adds at most one chunk to CACHE_BYTES. Partial decoding uses one Blosc thread;
    callers parallelise numerical work after gathering, without nested I/O teams.
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
        # Compression libraries initialize process settings; load them only at the I/O boundary.
        import blosc2

        key = (group, index)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        raw_size, compressed_size, checksum, payload_offset = self.groups[group][0][index]
        while self.cache and self.cache_bytes + compressed_size > CACHE_BYTES:
            _, (discarded, size) = self.cache.popitem(last=False)
            self.cache_bytes -= size
            del discarded
        self.handle.seek(payload_offset)
        blob = codec._read_exact(self.handle, compressed_size, "window chunk")
        if zlib.crc32(blob) != checksum:
            raise ValueError(f"{self.context}: corrupt checkpoint: window chunk checksum mismatch")
        actual_raw, actual_compressed, _ = blosc2.get_cbuffer_sizes(blob)
        if (actual_raw, actual_compressed) != (raw_size, compressed_size):
            raise ValueError(f"{self.context}: corrupt checkpoint: window chunk size mismatch")
        # codec._compress writes eight-byte items, except odd-sized groups which use bytes.
        typesize = 8 if raw_size % 8 == 0 else 1
        chunk = blosc2.SChunk(
            chunksize=raw_size,
            cparams=blosc2.CParams(typesize=typesize, nthreads=1, codec=blosc2.Codec.ZSTD),
            dparams=blosc2.DParams(nthreads=1),
        )
        chunk.append_chunk(blob)
        self.cache[key] = (chunk, compressed_size)
        self.cache_bytes += compressed_size
        return chunk

    def read_range(self, group, start, stop, destination):
        chunks, starts = self.groups[group]
        offset = 0
        while start < stop:
            index = bisect.bisect_right(starts, start) - 1
            end = min(stop, starts[index] + chunks[index][0])
            left, right = start - starts[index], end - starts[index]
            chunk = self._chunk(group, index)
            size = chunk.typesize
            target = destination[offset : offset + end - start]
            if left % size == 0 and right % size == 0:
                chunk.get_slice(left // size, right // size, out=target)
            else:
                # Byte masks can start/end inside the codec's eight-byte items.
                raw = chunk.get_slice(left // size, (right + size - 1) // size)
                target[:] = memoryview(raw)[left % size : left % size + len(target)]
                del raw
            del chunk
            offset += end - start
            start = end


__all__ = ["WindowArray", "ArrayReader", "CACHE_BYTES"]
