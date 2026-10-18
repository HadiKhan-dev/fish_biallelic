"""Append-only checkpoint containers (format of 92b3daf, unchanged).

A container is one file holding a sequence of immutable records, each a key and one encoded payload
(store.codec). Committed bytes are never modified: a newer record supersedes an older one with the same key
(the latest complete record wins). Layout: a file header (magic, format version, kind, name, file identity, CRC),
then records of

    header   magic, sequence, written time, stored and decoded payload bytes, payload layout,
             record identity, key, header CRC
    payload  store.codec payload
    trailer  magic, sequence, record length, header CRC, trailer CRC

A record is valid only once its trailer is complete, so an interrupted writer can only leave an incomplete last
record: readers ignore it, and the next writer truncates it (with one warning) before appending.

Appends are serialised by a per-process thread lock plus ``flock`` on the file (cluster-coherent on the /rds
Lustre mount), so threads, processes and nodes may append to one file. Inside the lock a writer refreshes its
index, truncates a torn tail, encodes and appends its record, and fsyncs it. Readers take no lock: they index
complete records only, by scanning record headers and trailers (never payloads), and refresh incrementally from
the last indexed end. Indexes are cached per process.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import os
import struct
import threading
import time
import zlib

from . import codec
from .identity import canonical_json

CONTAINER_VERSION = 1

CONTAINER_KINDS = ("run", "contig")

# File header: magic, format version, kind, file identity, name length; then the UTF-8 name and a CRC32 of
# everything before it.
_FILE_MAGIC = b"HRCKFILE"
_FILE_HEADER = struct.Struct("<8sHB16sH")

# Record header: magic, sequence, written (UNIX seconds), stored payload bytes, decoded bytes, metadata offset
# (from the payload start), buffer count, record identity, key length; then the UTF-8 key and a CRC32.
_RECORD_MAGIC = b"HRCKREC1"
_RECORD = struct.Struct("<8sQdQQQQ16sH")

# Trailer: magic, sequence, record length, header CRC; then its own CRC32.
_TRAILER_MAGIC = b"HRCKEND1"
_TRAILER = struct.Struct("<8sQQI")

_U32 = struct.Struct("<I")

# Positional reads while indexing start small and double (up to the maximum) while records are small enough to
# be read in passing; a large record's trailer read also covers the next header.
_SCAN_WINDOW = 64 << 10
_SCAN_WINDOW_MAX = 1 << 20

_MAX_KEY_BYTES = 0xFFFF


class CheckpointMismatch(RuntimeError):
    """A bound record (put_once / bind_identity) differs from the value this run would bind."""


@dataclass(frozen=True)
class RecordInfo:
    """Index entry of one committed record; reading it needs no payload access.

    ``identity`` (32 hex digits) is unique to this committed record (a rewrite, even with the same content, gets
    a new one): downstream stages bind it to identify their sources. ``decoded_bytes`` counts decoded array
    buffers plus pickle metadata, not the peak memory of a consumer.
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
    def payload_end(self) -> int:
        return self.payload_offset + self.stored_bytes


class _Index:
    """Per-process index of one container: the latest complete record per key."""

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


_INDEXES: dict[str, _Index] = {}
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
            self.span = (min(2 * self.span, _SCAN_WINDOW_MAX) if 0 <= ahead < self.span else _SCAN_WINDOW)
            self.start = offset
            self.data = os.pread(self.fd, max(count, self.span), offset)
            relative = 0
            if len(self.data) < count:
                return None
        return self.data[relative:relative + count]


def _file_header(kind, name, file_identity):
    name_bytes = name.encode()
    header = _FILE_HEADER.pack(
        _FILE_MAGIC, CONTAINER_VERSION, CONTAINER_KINDS.index(kind), file_identity, len(name_bytes)) + name_bytes
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
            f"{path}: checkpoint container format {version} is not supported by this release "
            f"(format {CONTAINER_VERSION}); use a new output root")
    rest = window.read(_FILE_HEADER.size, name_length + _U32.size)
    if (rest is None or kind >= len(CONTAINER_KINDS)
            or zlib.crc32(fixed + rest[:name_length]) != _U32.unpack(rest[name_length:])[0]):
        raise ValueError(f"{path}: corrupt checkpoint container header")
    header = dict(kind=CONTAINER_KINDS[kind], name=rest[:name_length].decode(), identity=file_identity.hex())
    return header, _FILE_HEADER.size + name_length + _U32.size


def _utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    if (_TRAILER.unpack(trailer[:_TRAILER.size]) != (_TRAILER_MAGIC, seq, end - offset, header_crc)
            or zlib.crc32(trailer[:_TRAILER.size]) != _U32.unpack(trailer[_TRAILER.size:])[0]):
        return None
    if seq != expected_seq:
        raise ValueError(
            f"{path}: corrupt checkpoint container (record {seq} at byte {offset}, "
            f"expected record {expected_seq})")
    return RecordInfo(
        key=rest[:key_length].decode(), seq=seq, offset=offset, stored_bytes=stored, decoded_bytes=decoded,
        written_utc=_utc(written), identity=identity.hex(), payload_offset=payload_offset,
        metadata_offset=metadata_offset, n_buffers=n_buffers), end


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

    An append writes a zero header, the payload and trailer, makes them durable and only then the header, so an
    interruption leaves a zero header (possibly cut short) or a valid header whose trailer is missing or still
    zero (pages lost in a node crash). Anything else at the end of the complete records is damage, which must
    not be truncated away.
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


class CheckpointFile:
    """One append-only container: a run's ``run.ckpt`` or a contig's file.

    Cheap and picklable (a path plus settings); the record index is cached per process. ``nthreads`` is the
    default CPU budget for encoding and decoding, ``clevel`` the default ZSTD level of puts.
    """

    def __init__(self, path, *, kind, name="", nthreads=1, clevel=codec.CLEVEL):
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
    def label(self) -> str:
        return self.name or os.path.basename(self.path)

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def size(self) -> int:
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

    def keys(self, prefix="") -> tuple[str, ...]:
        """Live keys (latest record per key) starting with ``prefix``, sorted."""
        return tuple(sorted(key for key in self._records() if key.startswith(prefix)))

    def infos(self, prefix="") -> tuple[RecordInfo, ...]:
        """RecordInfo of the live keys starting with ``prefix``, sorted by key."""
        records = self._records()
        return tuple(records[key] for key in sorted(records) if key.startswith(prefix))

    def has(self, key) -> bool:
        return key in self._records()

    def info(self, key) -> RecordInfo | None:
        return self._records().get(key)

    def get(self, key, *, nthreads=None, items=None):
        """Decode the latest record of ``key`` (KeyError if absent).

        ``items`` selects top-level entries of a dict payload (missing ones are omitted), decoding only the
        arrays they use.
        """
        info = self.info(key)
        if info is None:
            raise KeyError(f"{self.label}: no checkpoint record {key!r}")
        return self._read(info, self.nthreads if nthreads is None else nthreads, items)

    def _read(self, info, nthreads, items=None):
        with open(self.path, "rb") as handle:
            metadata_offset = info.payload_offset + info.metadata_offset
            if items is None:
                return codec.decode(handle, info.payload_offset, info.payload_end, metadata_offset,
                                    info.n_buffers, nthreads)
            return codec.decode_items(handle, info.payload_offset, info.payload_end, metadata_offset,
                                      info.n_buffers, tuple(items), nthreads, f"{self.label}:{info.key}")

    def put(self, key, payload, *, nthreads=None, clevel=None) -> RecordInfo:
        """Append a record superseding any earlier ``key``; return its RecordInfo (durable on return)."""
        with self._append_lock() as (fd, index):
            return self._append(fd, index, key, payload, nthreads, clevel)

    def put_once(self, key, payload, *, nthreads=None):
        """Bind ``key`` to a JSON-compatible payload, exactly once; return the bound payload.

        The check and the append are atomic across threads, processes and nodes. An existing record must equal
        ``payload`` as canonical JSON, else CheckpointMismatch.
        """
        expected = canonical_json(payload)
        with self._append_lock() as (fd, index):
            existing = index.records.get(key)
            if existing is None:
                self._append(fd, index, key, payload, nthreads, None)
                return payload
            bound = self._read(existing, 1)
        if canonical_json(bound) != expected:
            raise CheckpointMismatch(
                f"{self.label}: checkpoint record {key!r} does not match this run; use a new output root")
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
                                f"{self.path}: corrupt checkpoint container at byte {index.end} "
                                "(not an interrupted append); refusing to truncate")
                        os.ftruncate(fd, index.end)
                        print(f"[Checkpoint] WARNING: {self.path}: discarded an incomplete record "
                              f"({size - index.end} bytes) left by an interrupted writer", flush=True)
                yield fd, index
            finally:
                os.close(fd)  # also releases the flock

    def _append(self, fd, index, key, payload, nthreads, clevel):
        key_bytes = str(key).encode()
        if not key_bytes or len(key_bytes) > _MAX_KEY_BYTES:
            raise ValueError(f"invalid checkpoint record key: {key!r}")
        nthreads = codec.cpu_threads(self.nthreads if nthreads is None else nthreads)
        clevel = self.clevel if clevel is None else int(clevel)
        # Fixed before anything is written: once the header is, a reader thread of this process may index the
        # record and advance the index.
        offset, seq = index.end, index.next_seq
        payload_offset = offset + _RECORD.size + len(key_bytes) + _U32.size
        with open(fd, "r+b", closefd=False) as handle:
            # A zero header until the payload and trailer are durable: until then the record is invalid (an
            # interrupted append, _torn).
            handle.seek(offset)
            handle.write(bytes(payload_offset - offset))
            metadata_offset, n_buffers, decoded = codec.encode(handle, payload, nthreads, clevel)
            stored = handle.tell() - payload_offset
            written = time.time()
            identity = os.urandom(16)
            header = _RECORD.pack(
                _RECORD_MAGIC, seq, written, stored, decoded, metadata_offset - payload_offset, n_buffers,
                identity, len(key_bytes)) + key_bytes
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
            key=key_bytes.decode(), seq=seq, offset=offset, stored_bytes=stored, decoded_bytes=decoded,
            written_utc=_utc(written), identity=identity.hex(), payload_offset=payload_offset,
            metadata_offset=metadata_offset - payload_offset, n_buffers=n_buffers)
        # Absolute, not incremental: a reader thread of this process may have indexed the new record already
        # (its scan holds only index.lock).
        with index.lock:
            index.records[info.key] = info
            index.next_seq = seq + 1
            index.end = end
        return info
