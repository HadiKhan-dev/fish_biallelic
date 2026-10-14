"""Append-only checkpoint containers and the protocol-5/Blosc2 payload encoding.

A container is one file holding a sequence of immutable records, each a key
and one encoded payload. Committed bytes are never modified: a newer record
supersedes an older one with the same key (the latest complete record wins).
Layout: a file header (magic, format version, kind, name, file identity, CRC),
then records of

    header   magic, sequence, written time, stored and decoded payload bytes,
             payload layout, record identity, key, header CRC
    payload  pickle protocol-5 metadata plus array buffers, as Blosc2-ZSTD
             chunks that each carry a CRC32
    trailer  magic, sequence, record length, header CRC, trailer CRC

A record is valid only once its trailer is complete, so an interrupted writer
can only leave an incomplete last record: readers ignore it, and the next
writer truncates it (with one warning) before appending.

Appends are serialized by a per-process thread lock plus ``flock`` on the
file (cluster-coherent on the /rds Lustre mount), so threads, processes and
nodes may append to one file. Inside the lock a writer refreshes its index,
truncates a torn tail, encodes and appends its record, and fsyncs it. Readers
take no lock: they index complete records only, by scanning record headers
and trailers (never payloads), and refresh incrementally from the last
indexed end. Indexes are cached per process.

Standalone payload files (``write``/``read``) use the same encoding with a
small header; they hold transient state (work states, batch transport) and
are replaced atomically.
"""
from __future__ import annotations


import bisect
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import os
import pickle
import struct
import sys
import tempfile
import threading
import time
import zlib
from collections import deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import RLock

import blosc2
import numpy as np


CLEVEL = 5


# Assembly records and work states are rewritten or superseded often and are
# not evidence arrays: ZSTD level 1 writes them ~2.4x faster than level 5 for
# ~5% more bytes (assembly preprocess, 452 MB raw). Decoding is unchanged.
# Not for floating-point evidence arrays (genotype likelihoods: 25% more bytes).
FAST_CLEVEL = 1


CODEC = blosc2.Codec.ZSTD


_MAGIC = b"BHDB2P52"


_HEADER = struct.Struct("<8sQQ")


_U32 = struct.Struct("<I")


_U64 = struct.Struct("<Q")


_BUFFER_DESC = struct.Struct("<QB")


_CHUNK_DESC = struct.Struct("<QQI")


_BUFFER_ALIGNMENT = 64


_DIRECT_BUFFER_BYTES = 16 << 20


_GROUP_BYTES = 128 << 20


_CHUNK_BYTES = 1 << 30


_METADATA_SPOOL_BYTES = 64 << 20


# Blosc2's GIL-release flag is process-global. Production parallel I/O uses
# separate processes; serialize overlapping pipeline scopes within a process.
_PARALLEL_READ_LOCK = RLock()


def _cpu_threads(nthreads):
    nthreads = max(1, int(nthreads))
    if hasattr(os, "sched_getaffinity"):
        nthreads = min(nthreads, len(os.sched_getaffinity(0)))
    return nthreads


def _buffer_layout(sizes):
    offsets = []
    end = 0
    for size in sizes:
        start = (end + _BUFFER_ALIGNMENT - 1) & -_BUFFER_ALIGNMENT
        if start > sys.maxsize - size:
            raise OverflowError("checkpoint buffer group is too large")
        offsets.append(start)
        end = start + size
    return offsets, end


def _aligned_buffer(nbytes):
    storage = np.empty(nbytes + _BUFFER_ALIGNMENT - 1, dtype=np.uint8)
    offset = (-storage.ctypes.data) % _BUFFER_ALIGNMENT
    return storage[offset:offset + nbytes]


def _compress(raw, nthreads, clevel):
    # A packed group may end with int8 calls and have an odd byte length.
    # Blosc2 4.5.1 emits unreadable all-zero special chunks when that length
    # is not divisible by its default eight-byte element size. Keep the
    # established shuffle for aligned chunks; otherwise compress as bytes.
    typesize = 8 if memoryview(raw).nbytes % 8 == 0 else 1
    return blosc2.compress2(
        raw,
        cparams=blosc2.CParams(
            nthreads=max(1, int(nthreads)), clevel=clevel, codec=CODEC,
            typesize=typesize,
        ),
    )


def _decompress_into(blob, destination, nthreads, context):
    try:
        raw_size, compressed_size, _block_size = blosc2.get_cbuffer_sizes(blob)
    except Exception as error:
        raise ValueError(f"corrupt checkpoint: invalid {context} header") from error
    destination_size = memoryview(destination).nbytes
    if raw_size != destination_size or compressed_size != len(blob):
        raise ValueError(
            f"corrupt checkpoint: {context} size mismatch "
            f"(frame raw/compressed={destination_size}/{len(blob)}, "
            f"Blosc raw/compressed={raw_size}/{compressed_size})"
        )
    try:
        blosc2.decompress2(
            blob,
            dst=destination,
            dparams=blosc2.DParams(nthreads=max(1, int(nthreads))),
        )
    except Exception as error:
        raise ValueError(f"corrupt checkpoint: cannot decompress {context}") from error


class _ArrayPickler(pickle.Pickler):
    """Use NumPy's protocol-5 buffer reduction for exact numeric ndarrays."""

    def reducer_override(self, obj):
        if (
            type(obj) is np.ndarray
            and not obj.dtype.hasobject
            and not (obj.flags.c_contiguous or obj.flags.f_contiguous)
        ):
            # A previous large materialized copy may still back async chunks.
            # Drain it before allocating another; small grouped arrays retain
            # their normal packing behavior and do not force serialization.
            if obj.nbytes >= _DIRECT_BUFFER_BYTES:
                writer = getattr(self, "_buffer_writer", None)
                if writer is not None and writer.pipeline is not None:
                    writer.pipeline.finish()
            contiguous = np.ascontiguousarray(obj)
            if not obj.flags.writeable:
                contiguous.flags.writeable = False
            return contiguous.__reduce_ex__(5)
        return NotImplemented



class _ChunkPipeline:
    """Rolling CPU/byte credits for checkpoint chunks, not an executor API.

    One caller CPU handles parsing or pickle/packing/ordered writes. Remaining
    CPUs belong to Blosc tasks. Reservations cover compressed input for reads
    and raw plus worst-case compressed output for writes. One oversized chunk
    is allowed only alone; a producer may additionally hold one packing group.
    """
    def __init__(self, nthreads, *, ordered):
        self.cpus = max(1, nthreads - 1)
        self.free_cpus = self.cpus
        self.ordered = ordered
        self.entries = deque()
        self.queued = deque()
        self.active = {}
        self.bytes = 0
        self.started = False
        self.pool = None
        _PARALLEL_READ_LOCK.acquire()
        try:
            self.previous_releasegil = blosc2.set_releasegil(True)
        except BaseException:
            _PARALLEL_READ_LOCK.release()
            raise

    def add(self, function, args, weight, consume):
        while self.entries and self.bytes + weight > _CHUNK_BYTES:
            self._progress()
        entry = dict(function=function, args=args, weight=weight, consume=consume,
                     future=None, done=False)
        self.entries.append(entry)
        self.queued.append(entry)
        self.bytes += weight
        if self.started:
            self._retire_ready()
            self._launch()
        elif len(self.queued) >= self.cpus or self.bytes >= _CHUNK_BYTES:
            self.started = True
            self._launch()

    def _launch(self):
        count = min(self.free_cpus, len(self.queued))
        if not count:
            return
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=self.cpus)
        floor, remainder = divmod(self.free_cpus, count)
        for index in range(count):
            entry = self.queued.popleft()
            threads = floor + (index < remainder)
            future = self.pool.submit(entry['function'], *entry['args'], nthreads=threads)
            entry['future'] = future
            self.active[future] = (entry, threads)
            self.free_cpus -= threads

    def _consume(self, entry):
        entry['consume'](entry['future'].result())
        self.bytes -= entry['weight']

    def _retire_ready(self):
        for future in tuple(self.active):
            if not future.done():
                continue
            entry, threads = self.active.pop(future)
            self.free_cpus += threads
            # Surface a failed chunk promptly even if ordered writes are still
            # waiting on an earlier chunk. The record is never completed.
            future.result()
            entry['done'] = True
            if not self.ordered:
                self.entries.remove(entry)
                self._consume(entry)
        if self.ordered:
            while self.entries and self.entries[0]['done']:
                self._consume(self.entries.popleft())

    def _progress(self):
        self.started = True
        self._retire_ready()
        self._launch()
        if self.active:
            wait(tuple(self.active), return_when=FIRST_COMPLETED)
            self._retire_ready()
            self._launch()

    def finish(self):
        while self.entries:
            self._progress()

    def close(self):
        try:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
            self.active.clear()
            self.entries.clear()
            self.queued.clear()
        finally:
            try:
                blosc2.set_releasegil(self.previous_releasegil)
            finally:
                _PARALLEL_READ_LOCK.release()


def _compress_checked(raw, clevel, nthreads):
    compressed = _compress(raw, nthreads, clevel)
    return raw.nbytes, compressed, zlib.crc32(compressed)


class _BufferWriter:
    def __init__(self, handle, nthreads, clevel):
        self.handle = handle
        self.nthreads = nthreads
        self.clevel = clevel
        self.pending = []
        self.pending_bytes = 0
        self.n_buffers = 0
        self.decoded_bytes = 0
        self.prefix = bytearray()
        self.pipeline = None
        self.owners = {}

    def __call__(self, pickle_buffer):
        raw = pickle_buffer.raw()
        nbytes = raw.nbytes
        self.n_buffers += 1
        if nbytes >= _DIRECT_BUFFER_BYTES:
            self.flush()
            self._write_group([(raw, raw.readonly, pickle_buffer)])
        else:
            aligned_start = (self.pending_bytes + _BUFFER_ALIGNMENT - 1) & -_BUFFER_ALIGNMENT
            next_bytes = aligned_start + nbytes
            if self.pending and next_bytes > _GROUP_BYTES:
                self.flush()
                next_bytes = nbytes
            self.pending.append((raw, raw.readonly, pickle_buffer))
            self.pending_bytes = next_bytes
        return None

    def _release(self, key):
        for raw, _readonly, pickle_buffer in self.owners.pop(key, ()):
            raw.release()
            pickle_buffer.release()

    def flush(self):
        if self.pending:
            pending, self.pending = self.pending, []
            self.pending_bytes = 0
            self._write_group(pending)

    def _chunk(self, raw, release=None):
        prefix, self.prefix = self.prefix, bytearray()

        def consume(result):
            raw_size, compressed, checksum = result
            self.handle.write(prefix)
            self.handle.write(_CHUNK_DESC.pack(raw_size, len(compressed), checksum))
            self.handle.write(compressed)
            raw.release()
            if release is not None:
                self._release(release)

        if self.nthreads > 1 and (self.pipeline is not None or raw.nbytes >= _DIRECT_BUFFER_BYTES):
            if self.pipeline is None:
                self.pipeline = _ChunkPipeline(self.nthreads, ordered=True)
            # Blosc output is bounded by raw bytes plus its small frame overhead.
            self.pipeline.add(_compress_checked, (raw, self.clevel),
                              2 * raw.nbytes + blosc2.MAX_OVERHEAD, consume)
        else:
            consume(_compress_checked(raw, self.clevel, self.nthreads))

    def _write_group(self, buffers):
        key = id(buffers)
        self.owners[key] = buffers
        offsets, total = _buffer_layout(raw.nbytes for raw, _, _ in buffers)
        self.decoded_bytes += total
        self.prefix += _U32.pack(len(buffers))
        for raw, readonly, _ in buffers:
            self.prefix += _BUFFER_DESC.pack(raw.nbytes, int(readonly))
        if total == 0:
            self.prefix += _U32.pack(0)
            self._release(key)
            return
        if len(buffers) == 1:
            raw = buffers[0][0]
            self.prefix += _U32.pack((total + _CHUNK_BYTES - 1) // _CHUNK_BYTES)
            for start in range(0, total, _CHUNK_BYTES):
                stop = min(total, start + _CHUNK_BYTES)
                self._chunk(raw[start:stop], key if stop == total else None)
            return
        packed = bytearray(total)
        destination = memoryview(packed)
        for (raw, _readonly, _), offset in zip(buffers, offsets):
            destination[offset:offset + raw.nbytes] = raw
        # The packed copy now owns these bytes; source views are no longer used
        # by asynchronous compression. Direct buffers retain owners until done.
        self._release(key)
        self.prefix += _U32.pack(1)
        self._chunk(destination)

    def finish(self):
        self.flush()
        if self.pipeline is not None:
            self.pipeline.finish()
        self.handle.write(self.prefix)
        self.prefix.clear()

    def close(self):
        try:
            if self.pipeline is not None:
                self.pipeline.close()
        finally:
            for key in tuple(self.owners):
                self._release(key)
            for raw, _, buffer in self.pending:
                raw.release()
                buffer.release()
            self.pending.clear()


def _write_metadata(handle, metadata, metadata_size, nthreads, clevel):
    n_chunks = (
        (metadata_size + _GROUP_BYTES - 1) // _GROUP_BYTES
        if metadata_size else 0
    )
    handle.write(_U64.pack(metadata_size))
    handle.write(_U32.pack(n_chunks))
    metadata.seek(0)
    remaining = metadata_size
    while remaining:
        raw = metadata.read(min(_GROUP_BYTES, remaining))
        if not raw:
            raise OSError("failed to read spooled checkpoint metadata")
        compressed = _compress(raw, nthreads, clevel)
        handle.write(
            _CHUNK_DESC.pack(len(raw), len(compressed), zlib.crc32(compressed))
        )
        handle.write(compressed)
        remaining -= len(raw)


def _encode(handle, obj, nthreads, clevel):
    """Write obj's payload at the handle's position.

    Returns the metadata's absolute offset, the buffer count and the decoded
    bytes (padded array-buffer groups plus raw pickle metadata).
    """
    with tempfile.SpooledTemporaryFile(
        max_size=_METADATA_SPOOL_BYTES, mode="w+b"
    ) as metadata:
        buffer_writer = _BufferWriter(handle, nthreads, clevel)
        try:
            pickler = _ArrayPickler(
                metadata, protocol=5, buffer_callback=buffer_writer
            )
            pickler._buffer_writer = buffer_writer
            pickler.dump(obj)
            metadata_size = metadata.tell()
            buffer_writer.finish()
            metadata_offset = handle.tell()
            _write_metadata(handle, metadata, metadata_size, nthreads, clevel)
        finally:
            buffer_writer.close()
    return metadata_offset, buffer_writer.n_buffers, buffer_writer.decoded_bytes + metadata_size


def _create(path):
    """Open a new file, creating its directory only when it is missing.

    Retried: a concurrent WorkState.clear can remove an empty parent between
    its creation and the open.
    """
    for attempt in range(3):
        try:
            return open(path, "w+b")
        except FileNotFoundError:
            parent = os.path.dirname(path)
            if not parent or attempt == 2:
                raise
            try:
                os.makedirs(parent, exist_ok=True)
            except FileNotFoundError:
                pass


PAYLOAD_SUFFIX = ".p5.b2"


def write(path, obj, nthreads=1, *, clevel=CLEVEL):
    """Write a standalone payload file atomically and durably; return its size."""
    nthreads = _cpu_threads(nthreads)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with _create(tmp) as handle:
            handle.write(_HEADER.pack(_MAGIC, 0, 0))
            metadata_offset, n_buffers, _decoded = _encode(handle, obj, nthreads, clevel)
            end = handle.tell()
            handle.seek(0)
            handle.write(_HEADER.pack(_MAGIC, metadata_offset, n_buffers))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return end


def _read_exact(handle, size, context):
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"corrupt checkpoint: truncated {context}")
    return data


def _read_chunk_at(fd, payload_offset, compressed_size, checksum, destination,
                   context, nthreads):
    """Read/check one independent chunk without changing the parser's offset."""
    compressed = bytearray(compressed_size)
    view = memoryview(compressed)
    offset = 0
    while offset < compressed_size:
        count = os.preadv(fd, [view[offset:]], payload_offset + offset)
        if count == 0:
            raise ValueError(f"corrupt checkpoint: truncated {context} payload")
        offset += count
    if zlib.crc32(compressed) != checksum:
        raise ValueError(f"corrupt checkpoint: {context} checksum mismatch")
    _decompress_into(compressed, destination, nthreads, context)


class _ChunkReader:
    """Rolling checked reads into disjoint final buffers, capped at 1 GiB input."""
    def __init__(self, handle, nthreads):
        self.fd = handle.fileno()
        self.pipeline = _ChunkPipeline(nthreads, ordered=False)

    def add(self, payload_offset, compressed_size, checksum, destination, context):
        self.pipeline.add(
            _read_chunk_at,
            (self.fd, payload_offset, compressed_size, checksum, destination, context),
            compressed_size, lambda _result: None)

    def flush(self):
        self.pipeline.finish()

    def close(self):
        self.pipeline.close()


def _chunk_layout(handle, raw_total, n_chunks, limit, context):
    if raw_total == 0:
        if n_chunks != 0:
            raise ValueError(f"corrupt checkpoint: nonempty {context} chunk list")
        return []
    if n_chunks == 0:
        raise ValueError(f"corrupt checkpoint: missing {context} chunks")

    chunks = []
    raw_seen = 0
    for index in range(n_chunks):
        raw_size, compressed_size, checksum = _CHUNK_DESC.unpack(
            _read_exact(handle, _CHUNK_DESC.size, f"{context} chunk header")
        )
        if raw_size == 0 or raw_size > _CHUNK_BYTES:
            raise ValueError(f"corrupt checkpoint: invalid {context} chunk size")
        if raw_seen + raw_size > raw_total:
            raise ValueError(f"corrupt checkpoint: oversized {context} chunks")
        payload_offset = handle.tell()
        if compressed_size > limit - payload_offset:
            raise ValueError(f"corrupt checkpoint: truncated {context} chunk")
        chunks.append((raw_size, compressed_size, checksum, payload_offset))
        handle.seek(compressed_size, os.SEEK_CUR)
        raw_seen += raw_size
    if raw_seen != raw_total:
        raise ValueError(f"corrupt checkpoint: incomplete {context} chunks")

    return chunks


def _read_chunks(handle, raw_total, n_chunks, limit, nthreads, context, reader=None):
    chunks = _chunk_layout(handle, raw_total, n_chunks, limit, context)
    if not chunks:
        return bytearray()

    backing = _aligned_buffer(raw_total)
    destination = memoryview(backing)
    raw_offset = 0
    for index, (raw_size, compressed_size, checksum, payload_offset) in enumerate(chunks):
        target = destination[raw_offset:raw_offset + raw_size]
        if reader is not None:
            reader.add(payload_offset, compressed_size, checksum, target,
                       f"{context} chunk {index}")
        else:
            handle.seek(payload_offset)
            compressed = _read_exact(
                handle, compressed_size, f"{context} chunk payload"
            )
            if zlib.crc32(compressed) != checksum:
                raise ValueError(
                    f"corrupt checkpoint: {context} chunk {index} checksum mismatch"
                )
            _decompress_into(
                compressed, target, nthreads, f"{context} chunk {index}",
            )
        raw_offset += raw_size
    handle.seek(chunks[-1][3] + chunks[-1][1])
    return backing


def _read_chunks_to_file(
    handle, raw_total, n_chunks, limit, nthreads, context, destination
):
    if raw_total == 0:
        if n_chunks != 0:
            raise ValueError(f"corrupt checkpoint: nonempty {context} chunk list")
        return
    if n_chunks == 0:
        raise ValueError(f"corrupt checkpoint: missing {context} chunks")

    raw_seen = 0
    for index in range(n_chunks):
        raw_size, compressed_size, checksum = _CHUNK_DESC.unpack(
            _read_exact(handle, _CHUNK_DESC.size, f"{context} chunk header")
        )
        if raw_size == 0 or raw_size > _CHUNK_BYTES:
            raise ValueError(f"corrupt checkpoint: invalid {context} chunk size")
        if raw_seen + raw_size > raw_total:
            raise ValueError(f"corrupt checkpoint: oversized {context} chunks")
        payload_offset = handle.tell()
        if compressed_size > limit - payload_offset:
            raise ValueError(f"corrupt checkpoint: truncated {context} chunk")
        compressed = _read_exact(
            handle, compressed_size, f"{context} chunk payload"
        )
        if zlib.crc32(compressed) != checksum:
            raise ValueError(
                f"corrupt checkpoint: {context} chunk {index} checksum mismatch"
            )
        raw = bytearray(raw_size)
        _decompress_into(
            compressed,
            raw,
            nthreads,
            f"{context} chunk {index}",
        )
        destination.write(raw)
        raw_seen += raw_size
    if raw_seen != raw_total:
        raise ValueError(f"corrupt checkpoint: incomplete {context} chunks")


def _spool_metadata(handle, end, nthreads, metadata):
    """Decode the payload's pickle metadata (at the handle) into ``metadata``."""
    metadata_size, = _U64.unpack(_read_exact(handle, _U64.size, "metadata size"))
    n_chunks, = _U32.unpack(_read_exact(handle, _U32.size, "metadata chunk count"))
    if metadata_size > sys.maxsize:
        raise ValueError("corrupt checkpoint: metadata is too large")
    _read_chunks_to_file(handle, metadata_size, n_chunks, end, nthreads, "metadata", metadata)
    if handle.tell() != end:
        raise ValueError("corrupt checkpoint: trailing data")
    metadata.seek(0)


def _unpickle(metadata, buffers):
    metadata.seek(0)
    try:
        return pickle.Unpickler(metadata, buffers=buffers).load()
    except (EOFError, pickle.UnpicklingError) as error:
        raise ValueError("corrupt checkpoint: invalid pickle metadata") from error


def _decode(handle, start, end, metadata_offset, expected_buffers, nthreads):
    """Decode the payload in [start, end) with checked chunks.

    nthreads is the total loading CPU budget: large payloads overlap
    independent chunk reads, CRCs and decompression across buffer groups.
    Metadata keeps its bounded spool and is unpickled only after verification.
    """
    nthreads = _cpu_threads(nthreads)
    handle.seek(start)
    reader = (_ChunkReader(handle, nthreads)
              if nthreads > 1 and end - start >= _DIRECT_BUFFER_BYTES else None)
    buffers = []
    try:
        _read_buffer_groups(handle, metadata_offset, expected_buffers,
                            nthreads, buffers, reader)
        if reader is not None:
            reader.flush()
    finally:
        if reader is not None:
            reader.close()
    with tempfile.SpooledTemporaryFile(
        max_size=_METADATA_SPOOL_BYTES, mode="w+b"
    ) as metadata:
        _spool_metadata(handle, end, nthreads, metadata)
        return _unpickle(metadata, buffers)


def _plain_arrays(value):
    """The arrays inside plain data (arrays, dicts, lists, tuples, scalars, strings)."""
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        return [value]
    if isinstance(value, dict):
        return [array for item in value.values() for array in _plain_arrays(item)]
    if isinstance(value, (list, tuple)):
        return [array for item in value for array in _plain_arrays(item)]
    if value is None or isinstance(value, (str, bytes, int, float, complex, np.generic)):
        return []
    raise TypeError(f"selective reads select plain data, not {type(value).__name__}")


def _decode_items(handle, start, end, metadata_offset, expected_buffers, keys,
                  nthreads, context):
    """Selected top-level entries of a dict payload, decoding only their arrays.

    Entries must be plain data. The metadata is first unpickled against
    untouched placeholder buffers (an unwritten allocation holds no memory) to
    find which buffers the selected arrays use; only the buffer groups holding
    those are read, checked and decoded, and the metadata is unpickled again
    with them. Missing keys are omitted.
    """
    nthreads = _cpu_threads(nthreads)
    handle.seek(start)
    groups = []
    for descriptors, offsets, total, n_chunks in _buffer_group_headers(
            handle, metadata_offset, expected_buffers):
        groups.append((descriptors, offsets, total, n_chunks, handle.tell()))
        _chunk_layout(handle, total, n_chunks, metadata_offset, "buffer group")
    with tempfile.SpooledTemporaryFile(
        max_size=_METADATA_SPOOL_BYTES, mode="w+b"
    ) as metadata:
        _spool_metadata(handle, end, nthreads, metadata)

        def selected(buffers):
            payload = _unpickle(metadata, buffers)
            if not isinstance(payload, dict):
                raise TypeError(f"{context}: selective reads need a dict payload")
            return {key: payload[key] for key in keys if key in payload}

        buffers = [np.empty(size, dtype=np.uint8)
                   for descriptors, *_ in groups for size, _readonly in descriptors]
        items = selected(buffers)
        spans = sorted((buffer.ctypes.data, buffer.nbytes, index)
                       for index, buffer in enumerate(buffers) if buffer.nbytes)
        starts = [begin for begin, _, _ in spans]
        needed = set()
        for array in _plain_arrays(items):
            if array.nbytes:
                address = array.ctypes.data
                position = bisect.bisect_right(starts, address) - 1
                if position >= 0 and address < spans[position][0] + spans[position][1]:
                    needed.add(spans[position][2])
        if not needed:
            return items
        first = 0
        for descriptors, offsets, total, n_chunks, chunk_list in groups:
            indices = range(first, first + len(descriptors))
            first += len(descriptors)
            if needed.isdisjoint(indices):
                continue
            handle.seek(chunk_list)
            backing = memoryview(_read_chunks(
                handle, total, n_chunks, metadata_offset, nthreads, "buffer group"))
            for index, (size, readonly), offset in zip(indices, descriptors, offsets):
                view = backing[offset:offset + size]
                buffers[index] = view.toreadonly() if readonly else view
        return selected(buffers)


def read(path, nthreads=1):
    """Read a standalone payload file written by :func:`write`."""
    with open(path, "rb") as handle:
        file_size, metadata_offset, expected_buffers = _read_header(handle, path)
        return _decode(handle, _HEADER.size, file_size, metadata_offset,
                       expected_buffers, nthreads)


def _buffer_group_headers(handle, metadata_offset, expected_buffers):
    """Yield descriptors; the consumer must advance over each chunk list."""
    buffer_count = 0
    while handle.tell() < metadata_offset:
        if metadata_offset - handle.tell() < _U32.size:
            raise ValueError("corrupt checkpoint: truncated buffer group")
        (n_group_buffers,) = _U32.unpack(
            _read_exact(handle, _U32.size, "buffer group count")
        )
        if n_group_buffers == 0 or n_group_buffers > expected_buffers - buffer_count:
            raise ValueError("corrupt checkpoint: invalid buffer group count")
        descriptors = []
        for _ in range(n_group_buffers):
            size, readonly = _BUFFER_DESC.unpack(
                _read_exact(handle, _BUFFER_DESC.size, "buffer descriptor")
            )
            if readonly not in (0, 1):
                raise ValueError("corrupt checkpoint: invalid buffer descriptor")
            descriptors.append((size, bool(readonly)))
        try:
            offsets, total = _buffer_layout(
                size for size, _readonly in descriptors
            )
        except OverflowError as error:
            raise ValueError(
                "corrupt checkpoint: invalid buffer descriptor"
            ) from error
        (n_chunks,) = _U32.unpack(
            _read_exact(handle, _U32.size, "buffer chunk count")
        )
        buffer_count += n_group_buffers
        yield descriptors, offsets, total, n_chunks
    if handle.tell() != metadata_offset or buffer_count != expected_buffers:
        raise ValueError("corrupt checkpoint: buffer count or boundary mismatch")


def _read_buffer_groups(handle, metadata_offset, expected_buffers, nthreads,
                        buffers, reader):
    for descriptors, offsets, total, n_chunks in _buffer_group_headers(
            handle, metadata_offset, expected_buffers):
        backing = _read_chunks(
            handle, total, n_chunks, metadata_offset, nthreads, "buffer group", reader)
        group_view = memoryview(backing)
        for (size, readonly), offset in zip(descriptors, offsets):
            view = group_view[offset:offset + size]
            buffers.append(view.toreadonly() if readonly else view)


def _read_header(handle, path):
    handle.seek(0, os.SEEK_END)
    file_size = handle.tell()
    handle.seek(0)
    if file_size < _HEADER.size + _U64.size + _U32.size:
        raise ValueError(f"{path}: corrupt checkpoint (truncated header)")
    magic, metadata_offset, expected_buffers = _HEADER.unpack(
        _read_exact(handle, _HEADER.size, "header")
    )
    if magic != _MAGIC:
        raise ValueError(f"{path}: not a payload file (bad magic)")
    if not (_HEADER.size <= metadata_offset <= file_size):
        raise ValueError(f"{path}: corrupt checkpoint (bad metadata offset)")

    return file_size, metadata_offset, expected_buffers


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

CONTAINER_VERSION = 1


CONTAINER_KINDS = ("run", "contig")


# File header: magic, format version, kind, file identity, name length; then
# the UTF-8 name and a CRC32 of everything before it.
_FILE_MAGIC = b"HRCKFILE"


_FILE_HEADER = struct.Struct("<8sHB16sH")


# Record header: magic, sequence, written (UNIX seconds), stored payload
# bytes, decoded bytes, metadata offset (from the payload start), buffer
# count, record identity, key length; then the UTF-8 key and a CRC32.
_RECORD_MAGIC = b"HRCKREC1"


_RECORD = struct.Struct("<8sQdQQQQ16sH")


# Trailer: magic, sequence, record length, header CRC; then its own CRC32.
_TRAILER_MAGIC = b"HRCKEND1"


_TRAILER = struct.Struct("<8sQQI")


# Positional reads while indexing start small and double (up to the maximum)
# while records are small enough to be read in passing; a large record's
# trailer read also covers the next header.
_SCAN_WINDOW = 64 << 10


_SCAN_WINDOW_MAX = 1 << 20


_MAX_KEY_BYTES = 0xFFFF


class CheckpointMismatch(RuntimeError):
    """A bound record (put_once) differs from the value this run would bind."""


@dataclass(frozen=True)
class RecordInfo:
    """Index entry of one committed record; reading it needs no payload access.

    ``identity`` is unique to this committed record (a rewrite, even with the
    same content, gets a new one): downstream caches record it to identify
    their sources. ``decoded_bytes`` counts decoded array buffers plus pickle
    metadata, not peak memory of a consumer.
    """

    key: str
    seq: int
    offset: int
    stored_bytes: int
    decoded_bytes: int
    written_utc: str
    identity: str
    payload_offset: int = field(repr=False)
    metadata_offset: int = field(repr=False)
    n_buffers: int = field(repr=False)

    @property
    def payload_end(self):
        return self.payload_offset + self.stored_bytes


class _Index:
    """Per-process index of one container: latest complete record per key."""

    def __init__(self):
        self.lock = threading.Lock()
        self.append_lock = threading.Lock()
        self.reset(None)

    def reset(self, inode):
        self.inode = inode
        self.end = 0
        self.next_seq = 1
        self.records = {}
        self.header = None


_INDEXES = {}


_INDEXES_GUARD = threading.Lock()


def _forget_indexes():
    # A forked child must not inherit locks another thread may hold.
    global _INDEXES_GUARD
    _INDEXES.clear()
    _INDEXES_GUARD = threading.Lock()


os.register_at_fork(after_in_child=_forget_indexes)


def _index(path):
    with _INDEXES_GUARD:
        index = _INDEXES.get(path)
        if index is None:
            index = _INDEXES[path] = _Index()
        return index


class _Window:
    """Bounded positional reads of a file prefix through a sliding window."""

    def __init__(self, fd, size):
        self.fd = fd
        self.size = size
        self.start = 0
        self.data = b""
        self.span = _SCAN_WINDOW

    def read(self, offset, count):
        if offset + count > self.size:
            return None
        relative = offset - self.start
        if relative < 0 or relative + count > len(self.data):
            ahead = offset - (self.start + len(self.data))
            self.span = (min(2 * self.span, _SCAN_WINDOW_MAX) if 0 <= ahead < self.span
                         else _SCAN_WINDOW)
            self.start = offset
            self.data = os.pread(self.fd, max(count, self.span), offset)
            relative = 0
            if len(self.data) < count:
                return None
        return self.data[relative:relative + count]


def _file_header(kind, name, file_identity):
    name_bytes = name.encode()
    header = _FILE_HEADER.pack(
        _FILE_MAGIC, CONTAINER_VERSION, CONTAINER_KINDS.index(kind), file_identity,
        len(name_bytes)) + name_bytes
    return header + _U32.pack(zlib.crc32(header))


def _parse_file_header(window, path):
    fixed = window.read(0, _FILE_HEADER.size)
    if fixed is None:
        raise ValueError(f"{path}: not a checkpoint container (truncated header)")
    magic, version, kind, file_identity, name_length = _FILE_HEADER.unpack(fixed)
    if magic != _FILE_MAGIC:
        raise ValueError(f"{path}: not a checkpoint container (bad magic)")
    if version != CONTAINER_VERSION:
        raise ValueError(
            f"{path}: checkpoint container format {version} is not supported by "
            f"this release (format {CONTAINER_VERSION}); use a new output root")
    rest = window.read(_FILE_HEADER.size, name_length + _U32.size)
    if (rest is None or kind >= len(CONTAINER_KINDS)
            or zlib.crc32(fixed + rest[:name_length]) != _U32.unpack(rest[name_length:])[0]):
        raise ValueError(f"{path}: corrupt checkpoint container header")
    header = dict(kind=CONTAINER_KINDS[kind], name=rest[:name_length].decode(),
                  identity=file_identity.hex())
    return header, _FILE_HEADER.size + name_length + _U32.size


def _parse_record(window, offset, expected_seq, path):
    """The complete record at ``offset`` and its end, or None if incomplete."""
    fixed = window.read(offset, _RECORD.size)
    if fixed is None:
        return None
    (magic, seq, written, stored, decoded, metadata_offset, n_buffers, identity,
     key_length) = _RECORD.unpack(fixed)
    if magic != _RECORD_MAGIC:
        return None
    rest = window.read(offset + _RECORD.size, key_length + _U32.size)
    if rest is None:
        return None
    header_crc = zlib.crc32(fixed + rest[:key_length])
    if header_crc != _U32.unpack(rest[key_length:])[0]:
        return None
    payload_offset = offset + _RECORD.size + key_length + _U32.size
    trailer_offset = payload_offset + stored
    trailer = window.read(trailer_offset, _TRAILER.size + _U32.size)
    if trailer is None:
        return None
    end = trailer_offset + _TRAILER.size + _U32.size
    if (_TRAILER.unpack(trailer[:_TRAILER.size])
            != (_TRAILER_MAGIC, seq, end - offset, header_crc)
            or zlib.crc32(trailer[:_TRAILER.size]) != _U32.unpack(trailer[_TRAILER.size:])[0]):
        return None
    if seq != expected_seq:
        raise ValueError(
            f"{path}: corrupt checkpoint container (record {seq} at byte {offset}, "
            f"expected record {expected_seq})")
    written_utc = datetime.fromtimestamp(written, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return RecordInfo(
        key=rest[:key_length].decode(), seq=seq, offset=offset, stored_bytes=stored,
        decoded_bytes=decoded, written_utc=written_utc, identity=identity.hex(),
        payload_offset=payload_offset, metadata_offset=metadata_offset,
        n_buffers=n_buffers), end


def _scan(index, fd, size, path):
    """Index the complete records in [index.end, size); stop at an incomplete one."""
    window = _Window(fd, size)
    if index.header is None:
        index.header, index.end = _parse_file_header(window, path)
    while True:
        parsed = _parse_record(window, index.end, index.next_seq, path)
        if parsed is None:
            return
        info, index.end = parsed
        index.records[info.key] = info
        index.next_seq += 1


def _torn(fd, offset, size):
    """Whether the bytes from ``offset`` to EOF are an interrupted append.

    An append writes a zero header, the payload and trailer, makes them
    durable and only then the header, so an interruption leaves a zero header
    (possibly cut short) or a valid header whose trailer is missing or still
    zero (pages lost in a node crash). Anything else at the end of the
    complete records is damage, which must not be truncated away.
    """
    window = _Window(fd, size)
    fixed = window.read(offset, min(_RECORD.size, size - offset))
    if not any(fixed):
        return True
    if len(fixed) < _RECORD.size:
        return False
    *_, stored, _decoded, _metadata, _buffers, _identity, key_length = _RECORD.unpack(fixed)
    rest = window.read(offset + _RECORD.size, key_length + _U32.size)
    if (fixed[:8] != _RECORD_MAGIC or rest is None
            or zlib.crc32(fixed + rest[:key_length]) != _U32.unpack(rest[key_length:])[0]):
        return False
    trailer_offset = offset + _RECORD.size + key_length + _U32.size + stored
    trailer = window.read(trailer_offset, _TRAILER.size + _U32.size)
    return trailer is None or not any(trailer)


def _pwrite_all(fd, data, offset):
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        view = view[written:]
        offset += written


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CheckpointFile:
    """One append-only container: a run's ``run.ckpt`` or a contig's file.

    Cheap and picklable (a path plus settings); the record index is cached
    per process. ``nthreads`` is the default CPU budget for encoding and
    decoding; ``clevel`` the default ZSTD level of puts.
    """

    def __init__(self, path, *, kind, name="", nthreads=1, clevel=CLEVEL):
        if kind not in CONTAINER_KINDS:
            raise ValueError(f"unknown checkpoint container kind: {kind!r}")
        self.path = os.path.abspath(os.fspath(path))
        self.kind = kind
        self.name = str(name)
        self.nthreads = max(1, int(nthreads))
        self.clevel = int(clevel)

    def __repr__(self):
        return f"CheckpointFile({self.path!r})"

    @property
    def label(self):
        return self.name or os.path.basename(self.path)

    def exists(self):
        return os.path.exists(self.path)

    def size(self):
        """Bytes on disk (0 when absent)."""
        try:
            return os.stat(self.path).st_size
        except FileNotFoundError:
            return 0

    def _check_header(self, index):
        header = index.header
        if header is not None and (header["kind"], header["name"]) != (self.kind, self.name):
            raise ValueError(
                f"{self.path}: holds checkpoint {header['kind']} {header['name']!r}, "
                f"not {self.kind} {self.name!r}")

    def _refresh(self, index, fd=None):
        """Bring the index up to the file's current complete records (index lock held)."""
        try:
            status = os.stat(self.path) if fd is None else os.fstat(fd)
        except FileNotFoundError:
            index.reset(None)
            return 0
        if status.st_ino != index.inode or status.st_size < index.end:
            index.reset(status.st_ino)
        if status.st_size > index.end:
            if fd is None:
                with open(self.path, "rb") as handle:
                    _scan(index, handle.fileno(), status.st_size, self.path)
            else:
                _scan(index, fd, status.st_size, self.path)
        self._check_header(index)
        return status.st_size

    def _records(self):
        index = _index(self.path)
        with index.lock:
            self._refresh(index)
            return dict(index.records)

    def keys(self, prefix=""):
        """Live keys (latest record per key) starting with ``prefix``, sorted."""
        return tuple(sorted(key for key in self._records() if key.startswith(prefix)))

    def infos(self, prefix=""):
        """RecordInfo of the live keys starting with ``prefix``, sorted by key."""
        records = self._records()
        return tuple(records[key] for key in sorted(records) if key.startswith(prefix))

    def has(self, key):
        return key in self._records()

    def info(self, key):
        return self._records().get(key)

    def get(self, key, *, nthreads=None, items=None):
        """Decode the latest record of ``key`` (KeyError if absent).

        ``items`` selects top-level entries of a dict payload (missing ones are
        omitted), decoding only the arrays they use.
        """
        info = self.info(key)
        if info is None:
            raise KeyError(f"{self.label}: no checkpoint record {key!r}")
        return self._read(info, self.nthreads if nthreads is None else nthreads, items)

    def _read(self, info, nthreads, items=None):
        with open(self.path, "rb") as handle:
            metadata_offset = info.payload_offset + info.metadata_offset
            if items is None:
                return _decode(handle, info.payload_offset, info.payload_end,
                               metadata_offset, info.n_buffers, nthreads)
            return _decode_items(handle, info.payload_offset, info.payload_end,
                                 metadata_offset, info.n_buffers, tuple(items), nthreads,
                                 f"{self.label}:{info.key}")

    def put(self, key, payload, *, nthreads=None, clevel=None):
        """Append a record superseding any earlier ``key``; return its RecordInfo."""
        with self._append_lock() as (fd, index):
            return self._append(fd, index, key, payload, nthreads, clevel)

    def put_once(self, key, payload, *, nthreads=None):
        """Bind ``key`` to a JSON-compatible payload, exactly once.

        The check and the append are atomic across threads, processes and
        nodes. An existing record must equal ``payload`` as canonical JSON
        (else CheckpointMismatch); the bound payload is returned.
        """
        expected = _canonical(payload)
        with self._append_lock() as (fd, index):
            existing = index.records.get(key)
            if existing is None:
                self._append(fd, index, key, payload, nthreads, None)
                return payload
            bound = self._read(existing, 1)
        if _canonical(bound) != expected:
            raise CheckpointMismatch(
                f"{self.label}: checkpoint record {key!r} does not match this run; "
                "use a new output root")
        return bound

    def _create(self):
        """Publish a new container with its complete header (atomic: link)."""
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        temporary = os.path.join(
            directory, f".{os.path.basename(self.path)}.{os.getpid()}."
            f"{threading.get_ident()}.{os.urandom(4).hex()}.tmp")
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        try:
            try:
                _pwrite_all(fd, _file_header(self.kind, self.name, os.urandom(16)), 0)
                os.fsync(fd)
            finally:
                os.close(fd)
            try:
                os.link(temporary, self.path)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)

    @contextmanager
    def _append_lock(self):
        """Hold this process's append lock and the file's flock; yield (fd, index).

        Inside, the index is current and any incomplete tail is truncated.
        """
        index = _index(self.path)
        with index.append_lock:
            if not os.path.exists(self.path):
                self._create()
            fd = os.open(self.path, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                with index.lock:
                    size = self._refresh(index, fd)
                    if size > index.end:
                        if not _torn(fd, index.end, size):
                            raise ValueError(
                                f"{self.path}: corrupt checkpoint container at byte "
                                f"{index.end} (not an interrupted append); refusing to "
                                "truncate")
                        os.ftruncate(fd, index.end)
                        print(f"[Checkpoint] WARNING: {self.path}: discarded an incomplete "
                              f"record ({size - index.end} bytes) left by an interrupted "
                              "writer", flush=True)
                yield fd, index
            finally:
                os.close(fd)  # also releases the flock

    def _append(self, fd, index, key, payload, nthreads, clevel):
        key_bytes = str(key).encode()
        if not key_bytes or len(key_bytes) > _MAX_KEY_BYTES:
            raise ValueError(f"invalid checkpoint record key: {key!r}")
        nthreads = _cpu_threads(self.nthreads if nthreads is None else nthreads)
        clevel = self.clevel if clevel is None else int(clevel)
        # Fixed before anything is written: once the header is, a reader
        # thread of this process may index the record and advance the index.
        offset, seq = index.end, index.next_seq
        payload_offset = offset + _RECORD.size + len(key_bytes) + _U32.size
        with open(fd, "r+b", closefd=False) as handle:
            # A zero header until the payload and trailer are durable: until
            # then the record is invalid (an interrupted append, _torn).
            handle.seek(offset)
            handle.write(bytes(payload_offset - offset))
            metadata_offset, n_buffers, decoded = _encode(handle, payload, nthreads, clevel)
            stored = handle.tell() - payload_offset
            written = time.time()
            identity = os.urandom(16)
            header = _RECORD.pack(
                _RECORD_MAGIC, seq, written, stored, decoded,
                metadata_offset - payload_offset, n_buffers, identity,
                len(key_bytes)) + key_bytes
            header_crc = zlib.crc32(header)
            end = payload_offset + stored + _TRAILER.size + _U32.size
            trailer = _TRAILER.pack(_TRAILER_MAGIC, seq, end - offset, header_crc)
            handle.write(trailer + _U32.pack(zlib.crc32(trailer)))
            # Payload and trailer are durable before the header can validate them.
            handle.flush()
            os.fsync(fd)
            handle.seek(offset)
            handle.write(header + _U32.pack(header_crc))
            handle.flush()
        os.fsync(fd)
        info = RecordInfo(
            key=key_bytes.decode(), seq=seq, offset=offset,
            stored_bytes=stored, decoded_bytes=decoded,
            written_utc=datetime.fromtimestamp(written, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            identity=identity.hex(), payload_offset=payload_offset,
            metadata_offset=metadata_offset - payload_offset, n_buffers=n_buffers)
        # Absolute, not incremental: a reader thread of this process may have
        # indexed the new record already (its scan holds only index.lock).
        with index.lock:
            index.records[info.key] = info
            index.next_seq = seq + 1
            index.end = end
        return info
