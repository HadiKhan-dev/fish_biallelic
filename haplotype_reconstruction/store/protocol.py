"""Publish complete results once and resume only by deterministic derivation identity.

The container remains the checked, append-only 3.5 format (P0 ENG-14/15); its random
physical record token is never a scientific upstream identity. A single append binds
payload and derivation. Per-result flock claims disappear on process death, and
completion markers are consequences of record checks, never evidence of completion.
ARCHITECTURE_40 §4.3 additionally binds contig payloads, code closure and library policy.
"""

from __future__ import annotations

import fcntl
import os
import platform
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.digest import array_digest, json_digest
from ..core.errors import InputError, InvariantError
from .container import CheckpointFile, CheckpointMismatch, RecordInfo
from .identity import CodeIdentity

__all__ = [
    "NumericalPolicy",
    "Derivation",
    "ContigPayload",
    "PublishedRecord",
    "PublicationMetadata",
    "publication_metadata",
    "numerical_policy_identity",
    "derivation_identity",
    "payload_identity",
    "bind_payload_identity",
    "cached_payload_identity",
    "verify_payload_identity",
    "publish",
    "resume",
    "read_items",
    "open_windows",
    "publish_output",
    "read_output",
    "is_complete",
    "claim",
    "mark_complete",
]

_PUBLICATION_SCHEMA = 1
# The corrected payload boundary binds GT fallback/header/drop counts, not input FILTER.
_PAYLOAD_SCHEMA = 2


@dataclass(frozen=True)
class NumericalPolicy:
    """Result-affecting runtime settings and the five required toolchain versions."""

    python: str
    numpy: str
    scipy: str
    numba: str
    llvmlite: str
    disable_intel_svml: bool = True
    threading_layer: str = "omp"
    blas_threads: int = 1
    fastmath: bool = False

    @classmethod
    def current(cls) -> NumericalPolicy:
        """Describe production policy using the installed numerical libraries at the explicit boundary.

        This describes intended settings. The engine verifies/applies them at its
        execution boundary; this method never mutates the process environment.
        """
        import llvmlite
        import numba
        import scipy

        return cls(
            platform.python_version(), np.__version__, scipy.__version__, numba.__version__, llvmlite.__version__
        )


@dataclass(frozen=True)
class Derivation:
    """Everything that determines a result, excluding timings and physical storage IDs.

    ``code`` is a static dependency closure rooted at science, stage adapters, input
    construction and codecs. ``inputs`` uses named upstream derivation/payload hashes,
    not RecordInfo.identity. Bound sample and contig orders belong in parameters.
    ``tie_keep`` names discovery reversals explicitly held at their default.
    """

    code: CodeIdentity
    parameters: Any
    numerical_policy: NumericalPolicy
    inputs: Mapping[str, str]
    tie_reverse: bool = False
    tie_keep: tuple[str, ...] = ()
    schema: int = 1

    @property
    def identity(self) -> str:
        """The stable scientific identity, unchanged by duplicate work or retries."""
        return derivation_identity(self)


def numerical_policy_identity(policy: NumericalPolicy) -> str:
    """Hash both numerical settings and Python/NumPy/SciPy/Numba/llvmlite versions."""
    return json_digest(policy)


def derivation_identity(derivation: Derivation) -> str:
    """Hash a resolved derivation using the single canonical digest encoding."""
    return json_digest(derivation)


@dataclass(frozen=True)
class ContigPayload:
    """Bind the source values 3.5 inference and export consume for one contig.

    positions: int64[L] bp before site-quality filtering; counts: int32[N,L,2]
    loader AD (missing/negative values contribute zero); ref/alt: length-L alleles.
    sample_ids: length-N canonical identifiers. ad_usable: bool[L], a record-level
    flag from the original AD field, not a positive-depth test. False means AD is
    absent or is HTSlib's single all-negative column (B/export/sites.py:148).
    gt_types: int8[N,L] cyvcf2 genotype classes (0 HOM_REF, 1 HET, 2 UNKNOWN,
    3 HOM_ALT); only columns with ad_usable=False enter the identity. It may be
    None when all AD records are usable. GT phase/order is not consumed by 3.5.
    contig_length: header length in bp, or None when unspecified. dropped: the
    loader's per-reason record counts, before site-quality rules.

    Input FILTER is not read by 3.5 and has no field here. Export derives FILTER
    from site-quality rules. Identity is independent of filename/compression/strides.
    The future source adapter supplies this record; it must not infer ad_usable
    from counts, since zero AD is usable and must not trigger GT fallback.
    """

    contig: str
    sample_ids: tuple[str, ...]
    positions: np.ndarray
    ref: tuple[str, ...]
    alt: tuple[str, ...]
    counts: np.ndarray
    ad_usable: np.ndarray
    gt_types: np.ndarray | None
    contig_length: int | None
    dropped: Mapping[str, int]


def payload_identity(payload: ContigPayload) -> str:
    """Digest consumed AD, conditional GT, alleles, positions, header length and drops.

    Inputs are not mutated. Raises InputError if the AD availability axis differs
    from positions or fallback GT is missing. Other array shape errors propagate.
    GT at usable AD records is ignored, including records whose AD totals are zero.
    """
    usable = np.asarray(payload.ad_usable, dtype=bool)
    if usable.shape != payload.positions.shape:
        raise InputError(f"AD availability shape {usable.shape} differs from positions {payload.positions.shape}")
    fallback = ~usable
    gt_identity = None
    if np.any(fallback):
        if payload.gt_types is None:
            raise InputError("GT types are required where source AD is unusable")
        gt_identity = array_digest(payload.gt_types[:, fallback])
    return json_digest(
        dict(
            payload_schema=_PAYLOAD_SCHEMA,
            contig=payload.contig,
            sample_ids=payload.sample_ids,
            positions=array_digest(payload.positions),
            ref=payload.ref,
            alt=payload.alt,
            counts=array_digest(payload.counts),
            ad_usable=array_digest(usable),
            fallback_gt=gt_identity,
            contig_length=payload.contig_length,
            dropped=payload.dropped,
        )
    )


def bind_payload_identity(file: CheckpointFile, identity: str) -> str:
    """Immutably bind this contig's source payload; a changed payload refuses resume."""
    file.put_once("input/payload", dict(payload_schema=_PAYLOAD_SCHEMA, contig=file.name, identity=identity))
    return identity


def cached_payload_identity(file: CheckpointFile, source: Path, compute: Callable[[], str]) -> str:
    """Reuse a payload digest only while source path, size and mtime_ns match its cache key.

    After a touch or move, ``compute`` re-reads the consumed fields. Equal content
    keeps the immutable binding; different content raises CheckpointMismatch. A
    concurrent source change during the read is refused rather than cached. An older
    payload-schema cache is recomputed; its old immutable binding refuses resume.
    """
    source = Path(source)
    status = source.stat()
    key = dict(path=str(source.resolve()), size=status.st_size, mtime_ns=status.st_mtime_ns)
    cache_key = "input/payload-cache"
    if file.has(cache_key):
        cached = file.get(cache_key)
        if cached.get("payload_schema") == _PAYLOAD_SCHEMA and cached["source"] == key:
            return bind_payload_identity(file, cached["identity"])
    identity = compute()
    after = source.stat()
    if (status.st_size, status.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise CheckpointMismatch(f"{source}: source changed while computing payload identity")
    bind_payload_identity(file, identity)
    file.put(cache_key, dict(payload_schema=_PAYLOAD_SCHEMA, source=key, identity=identity))
    return identity


def bound_payload_identity(file: CheckpointFile) -> str:
    """Read the immutable source binding after checking its schema and contig.

    This control record is created by init, before derived publications exist.
    Missing or incompatible bindings raise CheckpointMismatch; this is not a
    shortcut for reading a scientific product's own derivation envelope.
    """
    info = file.info("input/payload")
    if info is None:
        raise CheckpointMismatch(f"{file.label}: no bound contig payload identity")
    bound = file._read(info, 1)
    if bound.get("payload_schema") != _PAYLOAD_SCHEMA:
        raise CheckpointMismatch(f"{file.label}: unsupported source payload schema {bound.get('payload_schema')!r}")
    if bound.get("contig") != file.name:
        raise CheckpointMismatch(f"{file.label}: source payload contig differs from its container")
    identity = bound.get("identity")
    if not isinstance(identity, str) or not identity:
        raise CheckpointMismatch(f"{file.label}: malformed source payload identity")
    return identity


def verify_payload_identity(file: CheckpointFile, actual: str) -> None:
    """Verify export's re-read payload against the immutable input binding."""
    expected = bound_payload_identity(file)
    if actual != expected:
        raise CheckpointMismatch(f"{file.label}: input payload {actual} differs from bound {expected}")


@dataclass(frozen=True)
class PublishedRecord:
    """A complete decoded product and its scientific identity; physical info is diagnostic."""

    identity: str
    payload: Any
    info: RecordInfo


@dataclass(frozen=True)
class PublicationMetadata:
    """Observed status fields, explicitly not a scientific compatibility certificate."""

    recorded_derivation: str
    fields: Mapping[str, Any]
    info: RecordInfo


def publication_metadata(file: CheckpointFile, key: str, items: tuple[str, ...]) -> PublicationMetadata | None:
    """Read array-free status fields from one committed publication snapshot.

    Missing returns None. Unsupported headers raise CheckpointMismatch; requested
    arrays (including nested arrays) raise TypeError before decoding buffer groups.
    No input mutation. The recorded derivation is diagnostic only: readers and
    resume MUST still provide an independently derived identity to read_items.
    This status boundary validates container/publication structure, not scientific
    compatibility or payload schema; the owning status adapter checks its schema.
    """
    info = file.info(key)
    if info is None:
        return None
    envelope = file._read(info, 1, items=("publication_schema", "derivation"), metadata_only=True)
    if envelope.get("publication_schema") != _PUBLICATION_SCHEMA or not isinstance(envelope.get("derivation"), str):
        raise CheckpointMismatch(f"{file.label}:{key}: not a supported 4.0 publication")
    fields = file._read(info, 1, items=items, path=("payload",), metadata_only=True)
    return PublicationMetadata(envelope["derivation"], fields, info)


def _check(envelope: dict, expected: str, label: str) -> None:
    if envelope.get("publication_schema") != _PUBLICATION_SCHEMA:
        raise CheckpointMismatch(f"{label}: not a supported 4.0 publication")
    actual = envelope["derivation"]
    if actual != expected:
        raise CheckpointMismatch(f"{label}: derivation {actual} differs from expected {expected}")


def publish(file: CheckpointFile, key: str, identity: str, payload: Any) -> RecordInfo:
    """Atomically publish payload and derivation; an identical duplicate is a no-op.

    A conflicting derivation raises CheckpointMismatch without changing the file.
    The append lock is also the container's writer lock; crashes retain the original
    torn-tail recovery behavior. Identical identities promise identical computation,
    so this protocol does not redundantly hash all scientific outputs.
    """
    envelope = dict(publication_schema=_PUBLICATION_SCHEMA, derivation=identity, payload=payload)
    with file.encode(envelope) as encoded, file._append_lock() as (fd, index):
        existing = index.records.get(key)
        if existing is not None:
            envelope = file._read(existing, 1, items=("publication_schema", "derivation"))
            _check(envelope, identity, f"{file.label}:{key}")
            return existing
        return file._append(fd, index, key, encoded, None, None)


def resume(file: CheckpointFile, key: str, identity: str, *, reader: Any = None) -> PublishedRecord | None:
    """Load a matching complete product; missing returns None, mismatch raises CheckpointMismatch. ``reader``: a
    CheckpointFile.batch() reader of ``file``, for many reads."""
    source = file if reader is None else reader
    info = source.info(key)
    if info is None:
        return None
    envelope = source._read(info, file.nthreads)
    _check(envelope, identity, f"{file.label}:{key}")
    return PublishedRecord(identity, envelope["payload"], info)


def read_items(
    file: CheckpointFile,
    key: str,
    identity: str,
    items: tuple[str, ...],
    *,
    out: Mapping[str, np.ndarray] | None = None,
    reader: Any = None,
) -> PublishedRecord | None:
    """Read selected stored fields from one identity-checked immutable snapshot.

    Missing records return None; absent fields are omitted. An identity mismatch
    raises CheckpointMismatch before any payload buffer group is decompressed (checked
    on the same metadata decode, release 4.8). ``out`` receives matching selected arrays
    in place; inputs are not mutated. ``reader``: a CheckpointFile.batch() reader of ``file``.
    """
    source = file if reader is None else reader
    info = source.info(key)
    if info is None:
        return None
    label = f"{file.label}:{key}"
    payload = source._read(
        info,
        file.nthreads,
        items=items,
        out=out,
        path=("payload",),
        check=lambda record: _check(record, identity, label),
    )
    return PublishedRecord(identity, payload, info)


@contextmanager
def open_windows(file: CheckpointFile, key: str, identity: str, items: tuple[str, ...]) -> Iterator[dict | None]:
    """Yield checked payload windows for this context, or None when the record is absent.

    Mismatch raises before array decoding. Windows retain this record's snapshot
    and raise ValueError on use after exit, even for empty gathers.
    """
    info = file.info(key)
    if info is None:
        yield None
        return
    envelope = file._read(info, 1, items=("publication_schema", "derivation"))
    _check(envelope, identity, f"{file.label}:{key}")
    with file._open_arrays(info, items, path=("payload",)) as reader:
        yield reader.arrays


def publish_output(
    file: CheckpointFile, key: str, identity: str, payload: Any, *, supersede: bool = False
) -> RecordInfo:
    """Publish a derived output, optionally replacing an earlier output's identity.

    Only records first published by this function carry the output capability.
    A scientific record cannot be superseded through this API. Equal identities
    are no-ops; changed identities require explicit supersede=True. The append
    lock keeps comparison and publication atomic; failures preserve prior data.
    """
    envelope = dict(
        publication_schema=_PUBLICATION_SCHEMA, derivation=identity, payload=payload, publication_kind="output"
    )
    with file.encode(envelope) as encoded, file._append_lock() as (fd, index):
        existing = index.records.get(key)
        if existing is not None:
            envelope = file._read(existing, 1, items=("publication_schema", "derivation", "publication_kind"))
            if envelope.get("publication_kind") != "output":
                raise InvariantError(f"{file.label}:{key}: cannot supersede a scientific record")
            _check(envelope, str(envelope.get("derivation")), f"{file.label}:{key}")
            if not supersede or envelope.get("derivation") == identity:
                _check(envelope, identity, f"{file.label}:{key}")
                return existing
        return file._append(fd, index, key, encoded, None, None)


def read_output(file: CheckpointFile, key: str, *, identity: Callable[[Any], str]) -> PublishedRecord | None:
    """Read the latest output snapshot and verify its identity from the decoded payload.

    The output adapter supplies its pure identity function (for example the
    selection digest). This API is for mutable output choices only: scientific
    publications are refused. It never treats the envelope's own identity as
    independent evidence. Schema/kind errors raise InvariantError; a recomputed
    identity mismatch raises CheckpointMismatch. The callback's errors propagate.
    """
    info = file.info(key)
    if info is None:
        return None
    envelope = file._read(info, 1, items=("publication_schema", "publication_kind", "derivation"))
    if envelope.get("publication_kind") != "output":
        raise InvariantError(f"{file.label}:{key}: current-output reader requires an output publication")
    token = envelope.get("derivation")
    if not isinstance(token, str) or not token:
        raise InvariantError(f"{file.label}:{key}: output derivation must be a nonempty string")
    _check(envelope, token, f"{file.label}:{key}")
    payload = file._read(info, 1, items=("payload",))["payload"]
    expected = identity(payload)
    _check(envelope, expected, f"{file.label}:{key}")
    return PublishedRecord(expected, payload, info)


def is_complete(file: CheckpointFile, records: Mapping[str, str]) -> bool:
    """Check every declared record's identity; never read a completion marker as truth (many records: one index
    snapshot and one handle, CheckpointFile.batch)."""
    if len(records) == 1:
        ((key, expected),) = records.items()
        return _envelope_matches(file, file, key, expected)
    complete = True
    with file.batch() as reader:
        for key, expected in records.items():
            complete = _envelope_matches(file, reader, key, expected) and complete
    return complete


def _envelope_matches(file: CheckpointFile, source: Any, key: str, expected: str) -> bool:
    info = source.info(key)
    if info is None:
        return False
    envelope = source._read(info, 1, items=("publication_schema", "derivation"))
    _check(envelope, expected, f"{file.label}:{key}")
    return True


_DIRECTORY_CHECKS = 5  # status checks of an existing directory before a failure is believed


def _ensure_directory(path: Path) -> None:
    """mkdir -p for a directory many processes create at once.

    pathlib's exist_ok re-raises EEXIST when its follow-up status check fails, and on a shared filesystem under a
    burst of metadata requests that check can fail for an existing directory (release 4.10: a resumed N = 320 run's
    concurrent block claims stopped the run with FileExistsError on checkpoints/contigs). The entry's existence is then
    confirmed by its own status, retried briefly, before the error is raised.
    """
    import stat
    import time

    try:
        path.mkdir(parents=True, exist_ok=True)
        return
    except FileExistsError as error:
        failure = error
    for attempt in range(_DIRECTORY_CHECKS):
        try:
            if stat.S_ISDIR(os.stat(path).st_mode):
                return
            break  # a non-directory entry: a real error
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    raise failure


@contextmanager
def claim(file: CheckpointFile, key: str, identity: str) -> Iterator[bool]:
    """Claim one result; yield whether it needs computation, waiting for a competing writer.

    The kernel releases flock on process death. A lock file stays in place while its record is
    incomplete, so waiting writers never split across different inodes. Once the record is complete
    the lock file is removed, still under the lock: any later claimer sees the completed record and
    computes nothing, whichever inode it locks, so a contig keeps one container file rather than one
    lock file per record. Lock files hold no completion state.
    """
    path = Path(file.path)
    _ensure_directory(path.parent)
    lock = path.with_name(f".{path.name}.{json_digest(key)}.claim")
    with lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        completed = False
        try:
            yield not is_complete(file, {key: identity})
            completed = True
        finally:
            try:
                if completed and is_complete(file, {key: identity}):
                    lock.unlink(missing_ok=True)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def mark_complete(
    file: CheckpointFile, step: str, records: Mapping[str, str], *, marker_path: Path | None = None
) -> None:
    """Derive completion from matching records, then publish optional workflow marker last.

    Rerunning after a crash between record and marker recreates the marker without
    recomputing. Missing records raise InvariantError. Existing conflicting records
    raise CheckpointMismatch before any marker is written.
    """
    if not records or not is_complete(file, records):
        raise InvariantError(f"{file.label}: cannot mark {step} complete before its records exist")
    file.put_once(f"complete/{step}", dict(records=records))
    from .status_index import save_index

    save_index(file)
    if marker_path is not None:
        marker_path = Path(marker_path)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{marker_path.name}.", dir=marker_path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(json_digest(records) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, marker_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
