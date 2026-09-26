"""Atomic protocol-5/Blosc checkpoint serialization for scientific arrays."""
from __future__ import annotations


import os
import pickle
import struct
import sys
import tempfile
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import RLock

import blosc2
import numpy as np


SUFFIX = ".p5.b2"


DONE_MARKER = "_done.p5v2"


CLEVEL = 5


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


def _compress(raw, nthreads):
    # A packed group may end with int8 calls and have an odd byte length.
    # Blosc2 4.5.1 emits unreadable all-zero special chunks when that length
    # is not divisible by its default eight-byte element size. Keep the
    # established shuffle for aligned chunks; otherwise compress as bytes.
    typesize = 8 if memoryview(raw).nbytes % 8 == 0 else 1
    return blosc2.compress2(
        raw,
        cparams=blosc2.CParams(
            nthreads=max(1, int(nthreads)), clevel=CLEVEL, codec=CODEC,
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
            # waiting on an earlier chunk. The temporary file is not published.
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


def _compress_checked(raw, nthreads):
    compressed = _compress(raw, nthreads)
    return raw.nbytes, compressed, zlib.crc32(compressed)


class _BufferWriter:
    def __init__(self, handle, nthreads):
        self.handle = handle
        self.nthreads = nthreads
        self.pending = []
        self.pending_bytes = 0
        self.n_buffers = 0
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
            self.pipeline.add(_compress_checked, (raw,), 2 * raw.nbytes + blosc2.MAX_OVERHEAD, consume)
        else:
            consume(_compress_checked(raw, self.nthreads))

    def _write_group(self, buffers):
        key = id(buffers)
        self.owners[key] = buffers
        offsets, total = _buffer_layout(raw.nbytes for raw, _, _ in buffers)
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


def contig_path(ckpt_dir, stage, r_name):
    return os.path.join(ckpt_dir, stage, r_name + SUFFIX)


def global_path(ckpt_dir, stage):
    return os.path.join(ckpt_dir, stage, "_global" + SUFFIX)


def _write_metadata(handle, metadata, metadata_size, nthreads):
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
        compressed = _compress(raw, nthreads)
        handle.write(
            _CHUNK_DESC.pack(len(raw), len(compressed), zlib.crc32(compressed))
        )
        handle.write(compressed)
        remaining -= len(raw)


def write(path, obj, nthreads=1):
    """Write a v2 protocol-5 checkpoint atomically and return its byte size."""
    nthreads = max(1, int(nthreads))
    if hasattr(os, 'sched_getaffinity'):
        nthreads = min(nthreads, len(os.sched_getaffinity(0)))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with tempfile.SpooledTemporaryFile(
            max_size=_METADATA_SPOOL_BYTES, mode="w+b"
        ) as metadata, open(tmp, "w+b") as handle:
            handle.write(_HEADER.pack(_MAGIC, 0, 0))
            buffer_writer = _BufferWriter(handle, nthreads)
            try:
                pickler = _ArrayPickler(
                    metadata, protocol=5, buffer_callback=buffer_writer
                )
                pickler._buffer_writer = buffer_writer
                pickler.dump(obj)
                metadata_size = metadata.tell()
                buffer_writer.finish()

                metadata_offset = handle.tell()
                _write_metadata(
                    handle, metadata, metadata_size, max(1, int(nthreads))
                )
                end = handle.tell()
                handle.seek(0)
                handle.write(
                    _HEADER.pack(
                        _MAGIC, metadata_offset, buffer_writer.n_buffers
                    )
                )
                handle.seek(end)
            finally:
                buffer_writer.close()
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return os.path.getsize(path)


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


def read(path, nthreads=1):
    """Read v2 with checked chunks; nthreads is the total loading CPU budget.

    Large files overlap independent chunk reads, CRCs and decompression across
    buffer groups. Small files and nthreads=1 retain the direct serial path.
    Metadata keeps its bounded spool and is unpickled only after verification.
    """
    with open(path, "rb") as handle:
        file_size, metadata_offset, expected_buffers = _read_header(handle, path)

        nthreads = max(1, int(nthreads))
        if hasattr(os, "sched_getaffinity"):
            nthreads = min(nthreads, len(os.sched_getaffinity(0)))
        reader = (_ChunkReader(handle, nthreads)
                  if nthreads > 1 and file_size >= _DIRECT_BUFFER_BYTES else None)
        buffers = []
        try:
            _read_buffer_groups(handle, metadata_offset, expected_buffers,
                                nthreads, buffers, reader)
            if reader is not None:
                reader.flush()
        finally:
            if reader is not None:
                reader.close()

        metadata_size, = _U64.unpack(
            _read_exact(handle, _U64.size, "metadata size")
        )
        n_metadata_chunks, = _U32.unpack(
            _read_exact(handle, _U32.size, "metadata chunk count")
        )
        if metadata_size > sys.maxsize:
            raise ValueError("corrupt checkpoint: metadata is too large")
        with tempfile.SpooledTemporaryFile(
            max_size=_METADATA_SPOOL_BYTES, mode="w+b"
        ) as metadata:
            _read_chunks_to_file(
                handle,
                metadata_size,
                n_metadata_chunks,
                file_size,
                nthreads,
                "metadata",
                metadata,
            )
            if handle.tell() != file_size:
                raise ValueError("corrupt checkpoint: trailing data")
            metadata.seek(0)
            try:
                return pickle.Unpickler(metadata, buffers=buffers).load()
            except (EOFError, pickle.UnpicklingError) as error:
                raise ValueError("corrupt checkpoint: invalid pickle metadata") from error


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


def read_size_bytes(path):
    """Estimate decoded v2 storage bytes without payload reads or checksums.

    Sum padded array-buffer groups and raw pickle metadata. This is NOT peak
    process memory: compressed working buffers, Python objects and consumers'
    numerical workspaces are additional. The scan validates frame layout but
    intentionally does not validate CRCs, Blosc headers or pickle contents.
    """
    with open(path, "rb") as handle:
        file_size, metadata_offset, expected_buffers = _read_header(handle, path)
        total_bytes = 0
        for _descriptors, _offsets, total, n_chunks in _buffer_group_headers(
                handle, metadata_offset, expected_buffers):
            _chunk_layout(handle, total, n_chunks, metadata_offset, "buffer group")
            total_bytes += total
        metadata_size, = _U64.unpack(_read_exact(handle, _U64.size, "metadata size"))
        n_chunks, = _U32.unpack(_read_exact(handle, _U32.size, "metadata chunk count"))
        if metadata_size > sys.maxsize:
            raise ValueError("corrupt checkpoint: metadata is too large")
        _chunk_layout(handle, metadata_size, n_chunks, file_size, "metadata")
        if handle.tell() != file_size:
            raise ValueError("corrupt checkpoint: trailing data")
        return total_bytes + metadata_size


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
        raise ValueError(f"{path}: not a v2 {SUFFIX} checkpoint (bad magic)")
    if not (_HEADER.size <= metadata_offset <= file_size):
        raise ValueError(f"{path}: corrupt checkpoint (bad metadata offset)")

    return file_size, metadata_offset, expected_buffers
