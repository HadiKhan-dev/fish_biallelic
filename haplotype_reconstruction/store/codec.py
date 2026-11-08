"""Payload encoding: pickle protocol 5 with array buffers in checked Blosc2-ZSTD chunks.

A payload is the protocol-5 pickle of any object, its numeric ndarray buffers written out of band:

    buffer groups   per group: buffer count, (size, read-only flag) per buffer, chunk count, then chunks;
                    small buffers are packed into groups of <= 128 MiB at 64-byte alignment, buffers of
                    >= 16 MiB form a group of their own; chunks hold <= 1 GiB of raw bytes
    metadata        the pickle stream: raw size, chunk count, chunks
    chunk           raw size, compressed size, CRC32 of the compressed bytes, Blosc2 frame

Decoding reads every buffer group into one aligned allocation and unpickles the metadata against views of it
(zero copy); `decode_items` decodes only the groups that selected entries of a dict payload use. Large payloads
overlap chunk reads, CRCs and (de)compression on a rolling pipeline of at most 1 GiB in flight, with ordered
writes; the CPU budget is the caller's ``nthreads`` (bounded by affinity).

Standalone payload files (`write_payload` / `read_payload`) hold transient state (work states, batch transport):
a small header, then one payload; they are replaced atomically and durably.
"""

from __future__ import annotations

import bisect
import os
import pickle
import struct
import sys
import tempfile
import threading
import zlib
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import RLock

import numpy as np

CLEVEL = 5

# Assembly records and work states are rewritten or superseded often and are not evidence arrays: ZSTD level 1
# writes them ~2.4x faster than level 5 for ~5% more bytes (assembly preprocess, 452 MB raw). Decoding is
# unchanged. Not for floating-point evidence arrays (genotype likelihoods: 25% more bytes).
FAST_CLEVEL = 1


PAYLOAD_SUFFIX = ".p5.b2"

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

# Blosc2's GIL-release flag is process-global. Production parallel I/O uses separate processes; overlapping
# pipeline scopes within one process are serialised.
_PARALLEL_READ_LOCK = RLock()


def cpu_threads(nthreads) -> int:
    """A positive thread count no larger than this process's CPU affinity; a live CPU budget (a contig lane's
    lease) is re-read now, so compression takes the CPUs the lane holds at that moment."""
    if hasattr(nthreads, "codec_threads"):
        nthreads = nthreads.codec_threads()  # a lane's lease: one thread while its pools hold its CPUs
    elif hasattr(nthreads, "threads"):
        nthreads = nthreads.threads()
    return min(max(1, int(nthreads)), _affinity_cpus())


def thread_setting(nthreads):
    """A store's thread setting: a live CPU budget as it is (re-read at each encode/decode), else a count."""
    return nthreads if hasattr(nthreads, "threads") else max(1, int(nthreads))


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
    return storage[offset : offset + nbytes]


# Raw bytes per Blosc thread (release 4.8, execution only: Blosc output and decoded values do not depend on threads).
# Blosc starts its threads per call: a store opened with a whole node's 112 threads spent ~8 ms per call on a small
# record (a selection round's 1,647 panel reads: 14.2 s, against 2.0 s on one thread).
_CODEC_BYTES_PER_THREAD = 1 << 20


def _codec_threads(nthreads, raw_bytes):
    """Threads for one Blosc call on ``raw_bytes``: the caller's count, at most one per _CODEC_BYTES_PER_THREAD."""
    return max(1, min(int(nthreads), int(raw_bytes) // _CODEC_BYTES_PER_THREAD))


def _compress(raw, nthreads, clevel):
    # A packed group may end with int8 calls and have an odd byte length. Blosc2 4.5.1 emits unreadable all-zero
    # special chunks when that length is not divisible by its default eight-byte element size. Keep the
    # established shuffle for aligned chunks; otherwise compress as bytes.
    # Compression libraries initialize process settings; load them only at the I/O boundary.
    import blosc2

    size = memoryview(raw).nbytes
    typesize = 8 if size % 8 == 0 else 1
    return blosc2.compress2(
        raw,
        cparams=blosc2.CParams(
            nthreads=_codec_threads(nthreads, size),
            clevel=clevel,
            codec=blosc2.Codec.ZSTD,
            typesize=typesize,
        ),
    )


def _decompress_into(blob, destination, nthreads, context):
    # Compression libraries initialize process settings; load them only at the I/O boundary.
    import blosc2

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
            blob, dst=destination, dparams=blosc2.DParams(nthreads=_codec_threads(nthreads, destination_size))
        )
    except Exception as error:
        raise ValueError(f"corrupt checkpoint: cannot decompress {context}") from error


class _ArrayPickler(pickle.Pickler):
    """Use NumPy's protocol-5 buffer reduction for exact numeric ndarrays (made contiguous if needed)."""

    def reducer_override(self, obj):
        if (
            type(obj) is np.ndarray
            and not obj.dtype.hasobject
            and not (obj.flags.c_contiguous or obj.flags.f_contiguous)
        ):
            # A previous large materialised copy may still back asynchronous chunks. Drain it before allocating
            # another; small grouped arrays keep their normal packing and do not force serialisation.
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
    """Rolling CPU/byte credits for checkpoint chunks.

    One caller CPU handles parsing or pickle/packing/ordered writes; the remaining CPUs belong to Blosc tasks.
    Reservations cover compressed input for reads and raw plus worst-case compressed output for writes. One
    oversized chunk is allowed only alone; a producer may additionally hold one packing group.
    """

    def __init__(self, nthreads, *, ordered):
        # Compression libraries initialize process settings; load them only at the I/O boundary.
        import blosc2

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
        entry = dict(function=function, args=args, weight=weight, consume=consume, future=None, done=False)
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
            future = self.pool.submit(entry["function"], *entry["args"], nthreads=threads)
            entry["future"] = future
            self.active[future] = (entry, threads)
            self.free_cpus -= threads

    def _consume(self, entry):
        entry["consume"](entry["future"].result())
        self.bytes -= entry["weight"]

    def _retire_ready(self):
        for future in tuple(self.active):
            if not future.done():
                continue
            entry, threads = self.active.pop(future)
            self.free_cpus += threads
            # Surface a failed chunk promptly even if ordered writes still wait on an earlier chunk; the record
            # is never completed.
            future.result()
            entry["done"] = True
            if not self.ordered:
                self.entries.remove(entry)
                self._consume(entry)
        if self.ordered:
            while self.entries and self.entries[0]["done"]:
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
        # Compression libraries initialize process settings; load them only at the I/O boundary.
        import blosc2

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
    """protocol-5 buffer callback: groups buffers and writes their checked chunks in order."""

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
        # Compression libraries initialize process settings; load them only at the I/O boundary.
        import blosc2

        prefix, self.prefix = self.prefix, bytearray()

        def consume(result):
            raw_size, compressed, checksum = result
            self.handle.write(prefix)
            self.handle.write(_CHUNK_DESC.pack(raw_size, len(compressed), checksum))
            self.handle.write(compressed)
            from .mapped import discard_pages

            discard_pages(raw)
            raw.release()
            if release is not None:
                self._release(release)

        if self.nthreads > 1 and (self.pipeline is not None or raw.nbytes >= _DIRECT_BUFFER_BYTES):
            if self.pipeline is None:
                self.pipeline = _ChunkPipeline(self.nthreads, ordered=True)
            # Blosc output is bounded by raw bytes plus its small frame overhead.
            self.pipeline.add(_compress_checked, (raw, self.clevel), 2 * raw.nbytes + blosc2.MAX_OVERHEAD, consume)
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
        for (raw, _readonly, _), offset in zip(buffers, offsets, strict=False):
            destination[offset : offset + raw.nbytes] = raw
        # Asynchronous compression reads the packed copy, so source views can be released.
        # Direct buffers retain their owners until done.
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
    n_chunks = (metadata_size + _GROUP_BYTES - 1) // _GROUP_BYTES if metadata_size else 0
    handle.write(_U64.pack(metadata_size))
    handle.write(_U32.pack(n_chunks))
    metadata.seek(0)
    remaining = metadata_size
    while remaining:
        raw = metadata.read(min(_GROUP_BYTES, remaining))
        if not raw:
            raise OSError("failed to read spooled checkpoint metadata")
        compressed = _compress(raw, nthreads, clevel)
        handle.write(_CHUNK_DESC.pack(len(raw), len(compressed), zlib.crc32(compressed)))
        handle.write(compressed)
        remaining -= len(raw)


def encode(handle, obj, nthreads, clevel):
    """Write obj's payload at the handle's position.

    Returns the metadata's absolute offset, the buffer count and the decoded bytes (padded array-buffer groups
    plus raw pickle metadata).
    """
    with tempfile.SpooledTemporaryFile(max_size=_METADATA_SPOOL_BYTES, mode="w+b") as metadata:
        buffer_writer = _BufferWriter(handle, nthreads, clevel)
        try:
            pickler = _ArrayPickler(metadata, protocol=5, buffer_callback=buffer_writer)
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

    Retried: concurrent scratch cleanup can remove an empty parent between its creation and the open.
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


def write_payload(path, obj, *, nthreads=1, clevel=CLEVEL) -> int:
    """Write a standalone payload file atomically and durably; return its size in bytes."""
    nthreads = cpu_threads(nthreads)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with _create(tmp) as handle:
            handle.write(_HEADER.pack(_MAGIC, 0, 0))
            metadata_offset, n_buffers, _decoded = encode(handle, obj, nthreads, clevel)
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


def _read_chunk_at(fd, payload_offset, compressed_size, checksum, destination, context, nthreads):
    """Read and check one independent chunk without changing the parser's offset."""
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
    """Rolling checked reads into disjoint final buffers, capped at 1 GiB of input in flight."""

    def __init__(self, handle, nthreads):
        self.fd = handle.fileno()
        self.pipeline = _ChunkPipeline(nthreads, ordered=False)

    def add(self, payload_offset, compressed_size, checksum, destination, context):
        self.pipeline.add(
            _read_chunk_at,
            (self.fd, payload_offset, compressed_size, checksum, destination, context),
            compressed_size,
            lambda _result: None,
        )

    def flush(self):
        self.pipeline.finish()

    def close(self):
        self.pipeline.close()


def _chunk_layout(handle, raw_total, n_chunks, limit, context):
    """Parse a chunk list at the handle; return [(raw, compressed, crc, payload offset)], handle after it."""
    if raw_total == 0:
        if n_chunks != 0:
            raise ValueError(f"corrupt checkpoint: nonempty {context} chunk list")
        return []
    if n_chunks == 0:
        raise ValueError(f"corrupt checkpoint: missing {context} chunks")
    chunks = []
    raw_seen = 0
    for _index in range(n_chunks):
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


def _read_chunk(handle, chunk, destination, nthreads, context):
    """Read, check and decompress one chunk into ``destination`` (sequential path)."""
    _raw_size, compressed_size, checksum, payload_offset = chunk
    handle.seek(payload_offset)
    compressed = _read_exact(handle, compressed_size, f"{context} payload")
    if zlib.crc32(compressed) != checksum:
        raise ValueError(f"corrupt checkpoint: {context} checksum mismatch")
    _decompress_into(compressed, destination, nthreads, context)


def _read_chunks(handle, raw_total, n_chunks, limit, nthreads, context, reader=None, out=None):
    """Decode a chunk list into one aligned buffer (through ``reader``'s pipeline when given)."""
    chunks = _chunk_layout(handle, raw_total, n_chunks, limit, context)
    if not chunks:
        return bytearray()
    backing = _aligned_buffer(raw_total) if out is None else out
    destination = memoryview(backing)
    raw_offset = 0
    for index, chunk in enumerate(chunks):
        raw_size, compressed_size, checksum, payload_offset = chunk
        target = destination[raw_offset : raw_offset + raw_size]
        if reader is not None:
            reader.add(payload_offset, compressed_size, checksum, target, f"{context} chunk {index}")
        else:
            _read_chunk(handle, chunk, target, nthreads, f"{context} chunk {index}")
        raw_offset += raw_size
    handle.seek(chunks[-1][3] + chunks[-1][1])
    return backing


def _read_chunks_to_file(handle, raw_total, n_chunks, limit, nthreads, context, destination):
    """Decode a chunk list sequentially into a file object (the metadata spool)."""
    chunks = _chunk_layout(handle, raw_total, n_chunks, limit, context)
    end = handle.tell()
    for index, chunk in enumerate(chunks):
        raw = bytearray(chunk[0])
        _read_chunk(handle, chunk, raw, nthreads, f"{context} chunk {index}")
        destination.write(raw)
    handle.seek(end)


def _spool_metadata(handle, end, nthreads, metadata):
    """Decode the payload's pickle metadata (at the handle) into ``metadata``."""
    (metadata_size,) = _U64.unpack(_read_exact(handle, _U64.size, "metadata size"))
    (n_chunks,) = _U32.unpack(_read_exact(handle, _U32.size, "metadata chunk count"))
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


def _buffer_group_headers(handle, metadata_offset, expected_buffers):
    """Yield (descriptors, offsets, total, chunk count) per group; the consumer advances over each chunk list."""
    buffer_count = 0
    while handle.tell() < metadata_offset:
        if metadata_offset - handle.tell() < _U32.size:
            raise ValueError("corrupt checkpoint: truncated buffer group")
        (n_group_buffers,) = _U32.unpack(_read_exact(handle, _U32.size, "buffer group count"))
        if n_group_buffers == 0 or n_group_buffers > expected_buffers - buffer_count:
            raise ValueError("corrupt checkpoint: invalid buffer group count")
        # A group's descriptors are contiguous: one read for all of them (a read per descriptor dominated loading
        # records made of many small arrays), within the payload's buffer section.
        if n_group_buffers * _BUFFER_DESC.size > metadata_offset - handle.tell():
            raise ValueError("corrupt checkpoint: truncated buffer descriptor")
        packed = _read_exact(handle, n_group_buffers * _BUFFER_DESC.size, "buffer descriptors")
        descriptors = []
        for size, readonly in _BUFFER_DESC.iter_unpack(packed):
            if readonly not in (0, 1):
                raise ValueError("corrupt checkpoint: invalid buffer descriptor")
            descriptors.append((size, bool(readonly)))
        try:
            offsets, total = _buffer_layout(size for size, _readonly in descriptors)
        except OverflowError as error:
            raise ValueError("corrupt checkpoint: invalid buffer descriptor") from error
        (n_chunks,) = _U32.unpack(_read_exact(handle, _U32.size, "buffer chunk count"))
        buffer_count += n_group_buffers
        yield descriptors, offsets, total, n_chunks
    if handle.tell() != metadata_offset or buffer_count != expected_buffers:
        raise ValueError("corrupt checkpoint: buffer count or boundary mismatch")


def _read_buffer_groups(handle, metadata_offset, expected_buffers, nthreads, buffers, reader):
    for descriptors, offsets, total, n_chunks in _buffer_group_headers(handle, metadata_offset, expected_buffers):
        backing = _read_chunks(handle, total, n_chunks, metadata_offset, nthreads, "buffer group", reader)
        group_view = memoryview(backing)
        for (size, readonly), offset in zip(descriptors, offsets, strict=False):
            view = group_view[offset : offset + size]
            buffers.append(view.toreadonly() if readonly else view)


def decode(handle, start, end, metadata_offset, expected_buffers, nthreads):
    """Decode the payload in [start, end) with checked chunks.

    ``nthreads`` is the total loading CPU budget: large payloads overlap independent chunk reads, CRCs and
    decompression across buffer groups. Metadata is spooled (bounded memory) and unpickled only after its
    chunks are verified.
    """
    nthreads = cpu_threads(nthreads)
    handle.seek(start)
    reader = _ChunkReader(handle, nthreads) if nthreads > 1 and end - start >= _DIRECT_BUFFER_BYTES else None
    buffers = []
    try:
        _read_buffer_groups(handle, metadata_offset, expected_buffers, nthreads, buffers, reader)
        if reader is not None:
            reader.flush()
    finally:
        if reader is not None:
            reader.close()
    with tempfile.SpooledTemporaryFile(max_size=_METADATA_SPOOL_BYTES, mode="w+b") as metadata:
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


def _select_path(payload, path, context):
    """Descend plain dictionary metadata without touching any array buffer."""
    for key in path:
        if type(payload) is not dict:
            raise TypeError(f"{context}: selection path requires plain dictionaries")
        payload = payload[key]
    return payload


def decode_items(
    handle,
    start,
    end,
    metadata_offset,
    expected_buffers,
    keys,
    nthreads,
    context,
    *,
    out=None,
    path=(),
    metadata_only=False,
    check=None,
):
    """Selected entries of a dict payload, decoding only their arrays.

    ``check`` (release 4.8), when given, receives the whole top-level record (arrays as untouched placeholders)
    from the first metadata pass, before any buffer group is decompressed: a publication envelope is checked on the
    same decode that selects its payload fields instead of a decode of its own.

    ``path`` descends plain-dict keys before selection; an empty path preserves the
    3.5 top-level behavior. A missing path key raises KeyError; non-dicts raise TypeError.

    Entries must be plain data. The metadata is first unpickled against untouched placeholder buffers (an
    unwritten allocation holds no memory) to find which buffers the selected arrays use; only the buffer groups
    holding those are read, checked and decoded, and the metadata is unpickled again with them. Missing keys
    are omitted. ``out`` maps selected array keys to writable contiguous ndarrays of the same shape and dtype.
    A full array in its own buffer group is decoded straight into the destination when its storage order matches;
    packed groups and other layouts use the ordinary checked decode followed by a copy. Returned entries are the
    caller's arrays. No on-disk format changes are needed. ``metadata_only`` is the
    status-reader capability: selected arrays or output buffers raise TypeError
    before any array buffer group is decompressed; ordinary reads are unchanged.
    """
    out = {} if out is None else out
    nthreads = cpu_threads(nthreads)
    handle.seek(start)
    groups = []
    for descriptors, offsets, total, n_chunks in _buffer_group_headers(handle, metadata_offset, expected_buffers):
        groups.append((descriptors, offsets, total, n_chunks, handle.tell()))
        _chunk_layout(handle, total, n_chunks, metadata_offset, "buffer group")
    with tempfile.SpooledTemporaryFile(max_size=_METADATA_SPOOL_BYTES, mode="w+b") as metadata:
        _spool_metadata(handle, end, nthreads, metadata)

        def selected(buffers, first=False):
            record = _unpickle(metadata, buffers)
            if first and check is not None:
                check(record)
            payload = _select_path(record, path, context)
            if not isinstance(payload, dict):
                raise TypeError(f"{context}: selective reads need a dict payload")
            return {key: payload[key] for key in keys if key in payload}

        buffers = [np.empty(size, dtype=np.uint8) for descriptors, *_ in groups for size, _readonly in descriptors]
        items = selected(buffers, first=True)
        if metadata_only:
            if out or _plain_arrays(items):
                raise TypeError(f"{context}: status metadata cannot contain arrays or output buffers")
            return items
        spans = sorted(
            (buffer.ctypes.data, buffer.nbytes, index) for index, buffer in enumerate(buffers) if buffer.nbytes
        )
        starts = [begin for begin, _, _ in spans]

        def buffer_index(array):
            if array.nbytes:
                address = array.ctypes.data
                position = bisect.bisect_right(starts, address) - 1
                if position >= 0 and address < spans[position][0] + spans[position][1]:
                    return spans[position][2]
            return None

        destinations = {}
        for key, target in out.items():
            source = items[key]
            if (
                not isinstance(source, np.ndarray)
                or not isinstance(target, np.ndarray)
                or target.shape != source.shape
                or target.dtype != source.dtype
                or not target.flags.writeable
                or not (target.flags.c_contiguous or target.flags.f_contiguous)
            ):
                raise ValueError(f"{context}: out[{key!r}] must be a writable contiguous array of matching shape/dtype")
            index = buffer_index(source)
            if (
                index is not None
                and source.ctypes.data == buffers[index].ctypes.data
                and source.nbytes == buffers[index].nbytes
                and source.strides == target.strides
            ):
                destinations[index] = memoryview(target.ravel(order="K")).cast("B")
        needed = {index for array in _plain_arrays(items) if (index := buffer_index(array)) is not None}
        first = 0
        for descriptors, offsets, total, n_chunks, chunk_list in groups:
            indices = range(first, first + len(descriptors))
            first += len(descriptors)
            if needed.isdisjoint(indices):
                continue
            handle.seek(chunk_list)
            destination = destinations.get(indices.start) if len(descriptors) == 1 else None
            backing = memoryview(
                _read_chunks(handle, total, n_chunks, metadata_offset, nthreads, "buffer group", out=destination)
            )
            for index, (size, readonly), offset in zip(indices, descriptors, offsets, strict=False):
                view = backing[offset : offset + size]
                buffers[index] = view.toreadonly() if readonly else view
        result = selected(buffers)
        for key, target in out.items():
            if not np.shares_memory(result[key], target):
                np.copyto(target, result[key], casting="no")
            result[key] = target
        return result


def _read_header(handle, path):
    handle.seek(0, os.SEEK_END)
    file_size = handle.tell()
    handle.seek(0)
    if file_size < _HEADER.size + _U64.size + _U32.size:
        raise ValueError(f"{path}: corrupt checkpoint (truncated header)")
    magic, metadata_offset, expected_buffers = _HEADER.unpack(_read_exact(handle, _HEADER.size, "header"))
    if magic != _MAGIC:
        raise ValueError(f"{path}: not a payload file (bad magic)")
    if not (_HEADER.size <= metadata_offset <= file_size):
        raise ValueError(f"{path}: corrupt checkpoint (bad metadata offset)")
    return file_size, metadata_offset, expected_buffers


def read_payload(path, *, nthreads=1):
    """Read a standalone payload file written by :func:`write_payload`."""
    with open(path, "rb") as handle:
        file_size, metadata_offset, expected_buffers = _read_header(handle, path)
        return decode(handle, _HEADER.size, file_size, metadata_offset, expected_buffers, nthreads)


def _affinity_cpus() -> int:
    """CPUs this process may run on (its affinity mask; the node's count only where no mask exists)."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, int(count))


__all__ = [
    "cpu_threads",
    "thread_setting",
    "encode",
    "write_payload",
    "decode",
    "decode_items",
    "read_payload",
    "CLEVEL",
    "FAST_CLEVEL",
    "PAYLOAD_SUFFIX",
]
