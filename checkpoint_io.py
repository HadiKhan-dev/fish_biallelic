"""Streaming protocol-5 checkpoint I/O for large NumPy pipeline payloads.

Pickle metadata is spooled separately while numeric arrays are emitted through
protocol 5 out-of-band buffers and compressed directly with Blosc. This avoids
the serial, memory-heavy ``pickle.dumps`` copy of an entire multi-gigabyte
checkpoint before compression can begin. Non-contiguous numeric arrays are
copied once into contiguous storage; ordinary C/F-contiguous arrays are not.

On-disk layout beneath a caller-supplied checkpoint root::

    {stage}/{contig}.p5.b2
    {stage}/_global.p5.b2
    {stage}/_done.p5v2

The format is intentionally v2-only. Earlier ``.pkl.b2`` payloads and
``_done`` markers are not read, so a format change requires checkpoint
regeneration rather than compatibility machinery in this hot I/O path.
"""

import os
import pickle
import struct
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import blosc2
import numpy as np


SUFFIX = ".p5.b2"
DONE_MARKER = "_done.p5v2"
CLEVEL = 5
CODEC = blosc2.Codec.ZSTD

_MAGIC = b"BHDB2P52"
_HEADER = struct.Struct("<8sQQ")  # magic, metadata offset, OOB buffer count
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_BUFFER_DESC = struct.Struct("<QB")  # byte length, read-only flag
_CHUNK_DESC = struct.Struct("<QQI")  # raw length, compressed length, CRC32

_BUFFER_ALIGNMENT = 64
_DIRECT_BUFFER_BYTES = 16 << 20
_GROUP_BYTES = 128 << 20
_CHUNK_BYTES = 1 << 30
_METADATA_SPOOL_BYTES = 64 << 20


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
    return blosc2.compress2(
        raw,
        cparams=blosc2.CParams(
            nthreads=max(1, int(nthreads)), clevel=CLEVEL, codec=CODEC
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
            contiguous = np.ascontiguousarray(obj)
            if not obj.flags.writeable:
                contiguous.flags.writeable = False
            return contiguous.__reduce_ex__(5)
        return NotImplemented


class _BufferWriter:
    def __init__(self, handle, nthreads):
        self.handle = handle
        self.nthreads = nthreads
        self.pending = []
        self.pending_bytes = 0
        self.n_buffers = 0

    def __call__(self, pickle_buffer):
        raw = pickle_buffer.raw()
        nbytes = raw.nbytes
        self.n_buffers += 1

        if nbytes >= _DIRECT_BUFFER_BYTES:
            self.flush()
            try:
                self._write_group([(raw, raw.readonly)])
            finally:
                raw.release()
                pickle_buffer.release()
        else:
            aligned_start = (
                self.pending_bytes + _BUFFER_ALIGNMENT - 1
            ) & -_BUFFER_ALIGNMENT
            next_bytes = aligned_start + nbytes
            if self.pending and next_bytes > _GROUP_BYTES:
                self.flush()
                next_bytes = nbytes
            self.pending.append((raw, raw.readonly, pickle_buffer))
            self.pending_bytes = next_bytes
        return None

    def flush(self):
        if self.pending:
            pending = self.pending
            try:
                self._write_group(
                    [(raw, readonly) for raw, readonly, _buffer in pending]
                )
            finally:
                self.pending = []
                self.pending_bytes = 0
                for raw, _readonly, pickle_buffer in pending:
                    raw.release()
                    pickle_buffer.release()

    def _write_group(self, buffers):
        offsets, total = _buffer_layout(
            raw.nbytes for raw, _readonly in buffers
        )
        self.handle.write(_U32.pack(len(buffers)))
        for raw, readonly in buffers:
            self.handle.write(_BUFFER_DESC.pack(raw.nbytes, int(readonly)))

        if total == 0:
            self.handle.write(_U32.pack(0))
            return

        if len(buffers) == 1:
            raw = buffers[0][0]
            n_chunks = (total + _CHUNK_BYTES - 1) // _CHUNK_BYTES
            self.handle.write(_U32.pack(n_chunks))
            for start in range(0, total, _CHUNK_BYTES):
                chunk = raw[start:start + _CHUNK_BYTES]
                compressed = _compress(chunk, self.nthreads)
                self.handle.write(
                    _CHUNK_DESC.pack(
                        chunk.nbytes, len(compressed), zlib.crc32(compressed)
                    )
                )
                self.handle.write(compressed)
            return

        packed = bytearray(total)
        destination = memoryview(packed)
        for (raw, _readonly), offset in zip(buffers, offsets):
            destination[offset:offset + raw.nbytes] = raw
        compressed = _compress(destination, self.nthreads)
        self.handle.write(_U32.pack(1))
        self.handle.write(
            _CHUNK_DESC.pack(total, len(compressed), zlib.crc32(compressed))
        )
        self.handle.write(compressed)


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
            pickler = _ArrayPickler(
                metadata, protocol=5, buffer_callback=buffer_writer
            )
            pickler.dump(obj)
            metadata_size = metadata.tell()
            buffer_writer.flush()

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


def _read_chunks(handle, raw_total, n_chunks, limit, nthreads, context):
    if raw_total == 0:
        if n_chunks != 0:
            raise ValueError(f"corrupt checkpoint: nonempty {context} chunk list")
        return bytearray()
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

    backing = _aligned_buffer(raw_total)
    destination = memoryview(backing)
    raw_offset = 0
    for index, (raw_size, compressed_size, checksum, payload_offset) in enumerate(chunks):
        handle.seek(payload_offset)
        compressed = _read_exact(
            handle, compressed_size, f"{context} chunk payload"
        )
        if zlib.crc32(compressed) != checksum:
            raise ValueError(
                f"corrupt checkpoint: {context} chunk {index} checksum mismatch"
            )
        _decompress_into(
            compressed,
            destination[raw_offset:raw_offset + raw_size],
            nthreads,
            f"{context} chunk {index}",
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
    """Read a v2 checkpoint, rejecting old formats and malformed frames."""
    with open(path, "rb") as handle:
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

        buffers = []
        while handle.tell() < metadata_offset:
            if metadata_offset - handle.tell() < _U32.size:
                raise ValueError("corrupt checkpoint: truncated buffer group")
            (n_group_buffers,) = _U32.unpack(
                _read_exact(handle, _U32.size, "buffer group count")
            )
            if n_group_buffers == 0 or n_group_buffers > expected_buffers - len(buffers):
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
            backing = _read_chunks(
                handle, total, n_chunks, metadata_offset, nthreads, "buffer group"
            )
            group_view = memoryview(backing)
            for (size, readonly), offset in zip(descriptors, offsets):
                view = group_view[offset:offset + size]
                buffers.append(view.toreadonly() if readonly else view)
        if handle.tell() != metadata_offset or len(buffers) != expected_buffers:
            raise ValueError("corrupt checkpoint: buffer count or boundary mismatch")

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


def save_contigs_parallel(ckpt_dir, stage, items, total_cores):
    items = list(items)
    n = len(items)
    if n == 0:
        return []
    os.makedirs(os.path.join(ckpt_dir, stage), exist_ok=True)
    total = max(1, int(total_cores))
    workers = min(n, total)
    threads = max(1, total // workers)

    def _one(item):
        r_name, data = item
        try:
            nbytes = write(
                contig_path(ckpt_dir, stage, r_name), data, nthreads=threads
            )
            return (r_name, nbytes, None)
        except OSError as error:
            return (r_name, 0, error)

    if workers == 1:
        results = [_one(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_one, items))

    ok = [(name, size) for name, size, error in results if error is None]
    mb = sum(size for _name, size in ok) / (1024 * 1024)
    print(
        f"  [Checkpoint] {stage}: {len(ok)}/{n} contigs written "
        f"({workers} parallel x {threads} blosc2 threads, {mb:.1f} MB total)"
    )
    for name, _size, error in results:
        if error is not None:
            print(f"  [Checkpoint] WARNING: {stage}/{name}: {error}")
    return results
