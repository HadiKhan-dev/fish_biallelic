"""Publish and selectively read one contig's immutable evidence product (EC25).

Owns the flat stored array layout and schema adapter. Science does no storage I/O;
identity binds parameters, numerical policy, canonical samples and consumed payload.
Method: docs/methods/evidence.md#records; B evidence/records.py.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from ...contracts.evidence import ContigEvidence, EvidenceDecisions, EvidenceDiagnostics
from ...contracts.likelihoods import CodedLikelihoods, gather_coded
from ...contracts.observations import ObservationKind, PairArray, gather_pairs, observation_counts
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.errors import InputError
from ...core.ties import TieCount, TieCounts
from ...science.evidence.parameters import EvidenceParameters
from ...science.evidence.run import PreparedEvidence
from ...store.container import CheckpointFile, RecordInfo
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, bound_payload_identity, publish, read_items
from ...store.protocol import open_windows as record_windows
from ...store.store import CheckpointStore

__all__ = [
    "EVIDENCE",
    "SCHEMA",
    "ARRAYS",
    "identity",
    "write",
    "load",
    "load_arrays",
    "load_likelihoods",
    "load_observed",
    "load_observed_counts",
    "load_dense",
    "open_arrays",
    "load_report",
    "load_decisions",
    "load_diagnostics",
    "complete",
    "progress",
    "result_reader",
]
EVIDENCE = "evidence"
SCHEMA = 2
SUPPORTED_SCHEMAS = (1, 2)
GL_FIELDS = tuple(CodedLikelihoods.__dataclass_fields__)
GL_ARRAYS = tuple(name for name in GL_FIELDS if name != "sites")
GL_WINDOWS = ("pairs", "values", "codes8", "codes16", "codes32", "dense")
ARRAYS = ("positions", "counts", "likelihoods", "observed", "discovery_sites", "founder_excluded", "block_spans")


def identity(
    parameters: EvidenceParameters,
    payload_identity: str,
    sample_ids: Sequence[str],
    *,
    reverse: bool = False,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Derivation of evidence from content, with no path, time or scheduling dependence."""
    return Derivation(
        code=code_dependency_identity(
            (
                "stages.evidence.tasks",
                "stages.evidence.records",
                "stages.inputs.vcf",
                "stages.inputs.sources",
                "science.evidence.run",
            )
        ),
        parameters=dict(evidence=parameters, sample_ids=tuple(sample_ids)),
        numerical_policy=numerical_policy or NumericalPolicy.current(),
        inputs={"payload": payload_identity},
        tie_reverse=reverse,
        schema=SCHEMA,
    ).identity


def _identity(file: CheckpointFile, expected: str | None) -> str:
    if expected is not None:
        return expected
    raise InputError("evidence reads require an expected identity; use result_reader with the bound manifest")


def _tie_counts(decisions: EvidenceDecisions) -> TieCounts:
    """Collect the same three recorded counter groups as the native evidence report."""
    groups = [decisions.site_quality.parameters.tie_events, decisions.read_model.tie_events]
    if decisions.sample_quality is not None:
        groups.append(decisions.sample_quality.tie_events)
    counts: dict[str, list[int]] = {}
    for group in groups:
        for row in group.entries:
            values = counts.setdefault(row.kind, [0, 0, 0, 0])
            for index, value in enumerate((row.decisions, row.inside_tie, row.wide, row.boundary)):
                values[index] += value
    return TieCounts(tuple(TieCount(kind, *values) for kind, values in sorted(counts.items())))


def write(
    file: CheckpointFile,
    prepared: PreparedEvidence,
    derivation: str,
    *,
    payload_identity: str,
    diagnostics: EvidenceDiagnostics | None = None,
) -> RecordInfo:
    """Publish flat selectively readable evidence-v2 arrays and unchanged decisions."""
    evidence = replace(prepared.evidence, payload_identity=payload_identity)
    if evidence.counts is None:
        raise InputError("the evidence record keeps the counts")
    paired = (
        isinstance(evidence.likelihoods, CodedLikelihoods)
        and isinstance(evidence.counts, PairArray)
        and isinstance(evidence.observed, PairArray)
        and evidence.observed.excluded is None
    )
    if paired:
        assert isinstance(evidence.counts, PairArray) and isinstance(evidence.observed, PairArray)
        payload = {
            name: getattr(evidence, name)
            for name in (
                "payload_identity",
                "contig",
                "positions",
                "discovery_sites",
                "founder_excluded",
                "block_spans",
            )
        }
        payload.update(
            schema=SCHEMA,
            sample_ids=list(evidence.sample_ids),
            counts_dense=evidence.counts.dense,
            observed_dense=evidence.observed.dense,
        )
    else:
        # Explicit dense publication and derived founder masks keep their exact
        # requested values. A masked flag is not inferable from raw depth alone.
        evidence = replace(
            evidence,
            counts=evidence.counts.materialise() if isinstance(evidence.counts, PairArray) else evidence.counts,
            observed=evidence.observed.materialise() if isinstance(evidence.observed, PairArray) else evidence.observed,
        )
        payload = to_payload(evidence)
        payload["schema"] = SCHEMA
    if isinstance(evidence.likelihoods, CodedLikelihoods):
        # Flat top-level arrays keep partial decompression and selective reads.
        payload.pop("likelihoods", None)
        payload["gl_encoding"] = "paired" if paired else "coded"
        payload.update({"gl_" + name: getattr(evidence.likelihoods, name) for name in GL_FIELDS})
    else:
        payload["gl_encoding"] = "dense"
    payload["decisions"] = to_payload(prepared.decisions)
    payload["diagnostics"] = to_payload(
        replace(diagnostics or EvidenceDiagnostics(), tie_events=_tie_counts(prepared.decisions))
    )
    return publish(file, EVIDENCE, derivation, payload)


def complete(file: CheckpointFile, expected_identity: str | None = None) -> bool:
    """Check a completed publication and its identity without decompressing scientific arrays."""
    if not file.has(EVIDENCE):
        return False
    record = read_items(file, EVIDENCE, _identity(file, expected_identity), ("schema",))
    if record is None:
        return False
    require_schema(record.payload, SUPPORTED_SCHEMAS)
    return True


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Report the committed record's checked container metadata, as native EC25.

    The container index includes only complete, checksummed records. No payload,
    scientific array or workflow marker is read. Schema and compatibility with
    current settings remain the identity-checked reader's responsibility.
    """
    info = file.info(EVIDENCE)
    return dict(evidence=info is not None, **({} if info is None else dict(decoded_bytes=info.decoded_bytes)))


def load(
    file: CheckpointFile, *, counts: bool = False, expected_identity: str | None = None, nthreads: int | None = None
) -> ContigEvidence:
    """Read frozen evidence; omit count buffers unless explicitly requested.

    Identity/schema mismatches fail before constructing the record. The container's
    configured decoder team is used; nthreads is retained for consumer compatibility.
    """
    version, encoding = _layout(file, expected_identity)
    request = Reader(_identity(file, expected_identity), "evidence", counts, schema=version, encoding=encoding)
    record = read_items(file, EVIDENCE, request.identity, request.items)
    if record is None:
        raise KeyError(EVIDENCE)
    return cast(ContigEvidence, request.decode(record.payload))


def _layout(file: CheckpointFile, expected_identity: str | None) -> tuple[int, str]:
    record = read_items(file, EVIDENCE, _identity(file, expected_identity), ("schema",))
    if record is None:
        raise KeyError(EVIDENCE)
    version = require_schema(record.payload, SUPPORTED_SCHEMAS)
    if version == 1:
        return version, "dense"
    record = read_items(file, EVIDENCE, _identity(file, expected_identity), ("gl_encoding",))
    if record is None:
        raise KeyError(EVIDENCE)
    encoding = record.payload["gl_encoding"]
    if encoding not in ("dense", "coded", "paired"):
        raise InputError(f"unsupported GL encoding {encoding!r}")
    return version, encoding


def _decode_gl(payload: Mapping[str, Any], encoding: str) -> np.ndarray | CodedLikelihoods:
    if encoding == "dense":
        return payload["likelihoods"]
    return CodedLikelihoods(**{name: payload["gl_" + name] for name in GL_FIELDS})


def load_likelihoods(file: CheckpointFile, *, expected_identity: str) -> np.ndarray | CodedLikelihoods:
    """Read the resident dense/coded source without projecting an entire contig."""
    version, encoding = _layout(file, expected_identity)
    names = ("likelihoods",) if encoding == "dense" else tuple("gl_" + name for name in GL_FIELDS)
    record = read_items(file, EVIDENCE, expected_identity, ("schema", *names))
    if record is None:
        raise KeyError(EVIDENCE)
    require_schema(record.payload, version)
    return _decode_gl(record.payload, encoding)


def load_arrays(
    file: CheckpointFile,
    names: Sequence[str],
    *,
    expected_identity: str | None = None,
    nthreads: int | None = None,
    out: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Decode selected evidence arrays, optionally directly into caller-owned destinations."""
    names = tuple(names)
    unknown = sorted(set(names) - set(ARRAYS))
    if unknown:
        raise KeyError(f"not evidence arrays: {unknown}")
    version, encoding = _layout(file, expected_identity)
    projections = tuple(
        name
        for name in names
        if (encoding == "paired" and name in ("likelihoods", "counts", "observed"))
        or (encoding == "coded" and name == "likelihoods")
    )
    plain = tuple(name for name in names if name not in projections)
    direct = None if out is None else {name: value for name, value in out.items() if name in plain}
    record = read_items(file, EVIDENCE, _identity(file, expected_identity), ("schema", *plain), out=direct)
    if record is None:
        raise KeyError(EVIDENCE)
    require_schema(record.payload, version)
    arrays = {name: record.payload[name] for name in plain}
    if projections:
        with open_arrays(file, projections, expected_identity=_identity(file, expected_identity)) as windows:
            for name in projections:
                source = windows[name]
                value = None if out is None else out.get(name)
                if value is None:
                    value = np.empty(source.shape, dtype=source.dtype)
                if value.shape != source.shape or value.dtype != source.dtype:
                    raise ValueError(f"{name} projection requires a matching destination")
                # At most 1 MiB of projection per sample window.
                size = max(1, 1048576 // (source.dtype.itemsize * int(np.prod(source.shape[2:]))))
                for sample in range(source.shape[0]):
                    for left in range(0, source.shape[1], size):
                        right = min(source.shape[1], left + size)
                        value[sample, left:right] = source.gather(slice(sample, sample + 1), slice(left, right))[0]
                arrays[name] = value
    return arrays


def load_dense(file: CheckpointFile, *, counts: bool = False, expected_identity: str) -> ContigEvidence:
    """One dense logical projection, read without a resident coded/dense twin.

    Whole-contig kernels (phase, founder refit, exclusion) keep their original
    memory path even for a mixed record. Other consumers retain compact sources.
    """
    version, _ = _layout(file, expected_identity)
    header = read_items(file, EVIDENCE, expected_identity, ("schema", "contig", "sample_ids", "payload_identity"))
    if header is None:
        raise KeyError(EVIDENCE)
    names = tuple(name for name in ARRAYS if counts or name != "counts")
    arrays = load_arrays(file, names, expected_identity=expected_identity)
    return ContigEvidence(
        schema=version,
        contig=header.payload["contig"],
        sample_ids=tuple(header.payload["sample_ids"]),
        payload_identity=header.payload["payload_identity"],
        counts=arrays.pop("counts", None),
        **arrays,
    )


class _TableWindows:
    """Gather selected dictionary rows, with O(window) storage and decoded bytes.

    Sort/deduplicate the local codes, read consecutive row runs, then restore the
    requested order. Interleaved sample batches never decode whole dictionaries.
    The store's bounded compressed-chunk cache remains shared by all windows.
    """

    def __init__(self, array: Any) -> None:
        self.array = array

    def gather_rows(self, first: int, last: int, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices)
        size = int(last) - int(first)
        row_bytes = self.array.dtype.itemsize * int(np.prod(self.array.shape[1:]))
        low = int(indices.min()) if len(indices) else 0
        span = int(indices.max()) + 1 - low if len(indices) else 0
        if 0 < span <= len(indices):
            # A selection whose rows span no more rows than it requests reads that span once and indexes it: the same
            # rows as reading each run of distinct codes, with one read instead of one per run, and never more rows
            # than requested (release 4.7.1).
            block = np.empty((span, *self.array.shape[1:]), dtype=self.array.dtype)
            begin = self.array.offset + (int(first) + low) * row_bytes
            self.array.reader.read_range(self.array.group, begin, begin + block.nbytes, memoryview(block).cast("B"))
            return block[indices - low]
        if 0 < len(indices) and size <= 4 * len(indices) + 4096:
            # The sorted distinct codes and their inverse by marking, O(size + n), instead of np.unique's sort:
            # the same arrays (release 4.7; painting gathers every sample's rows once per batch and pass).
            mark = np.zeros(size, dtype=np.bool_)
            mark[indices] = True
            selected = np.flatnonzero(mark)
            inverse = (np.cumsum(mark) - 1)[indices]
        else:
            selected, inverse = np.unique(indices, return_inverse=True)
        rows = np.empty((len(selected), *self.array.shape[1:]), dtype=self.array.dtype)
        # Each run is at most the selected window size, even when D grows with L.
        boundaries = np.r_[0, np.flatnonzero(np.diff(selected) != 1) + 1, len(selected)]
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            if left == right:
                continue
            begin = self.array.offset + (first + int(selected[left])) * row_bytes
            target = rows[left:right]
            self.array.reader.read_range(self.array.group, begin, begin + target.nbytes, memoryview(target).cast("B"))
        return rows[inverse]


class LikelihoodWindows:
    """Evidence-owned coded GL window adapter; never escape its open context."""

    sites: int
    widths: np.ndarray
    rows: np.ndarray
    offsets: np.ndarray
    values: Any
    codes8: Any
    codes16: Any
    codes32: Any
    dense: Any

    def __init__(self, metadata: Mapping[str, Any], arrays: Mapping[str, Any]) -> None:
        for name in GL_FIELDS:
            setattr(self, name, arrays["gl_" + name] if name in GL_WINDOWS else metadata["gl_" + name])
        self.values = _TableWindows(arrays["gl_values"])
        self.pairs = _TableWindows(arrays["gl_pairs"])
        self.reader = arrays["gl_dense"].reader
        self.shape = (len(self.widths), self.sites, 3)
        self.dtype = np.dtype(np.float64)

    def gather(self, samples: slice, columns: Any) -> np.ndarray:
        if self.reader.closed:
            raise ValueError("window reader is closed")
        return gather_coded(self, samples, columns)

    def resident_bytes(self, n_columns: int, samples: slice = slice(None)) -> int:
        """Bytes of resident_columns over ``n_columns`` sites for ``samples``: their dictionaries (pairs and values,
        32 bytes a row), each coded sample's codes (1, 2 or 4 bytes a site) and each dense sample's rows (24 bytes)."""
        first, stop, _ = samples.indices(len(self.widths))
        widths = np.asarray(self.widths[first:stop], dtype=np.int64)
        rows = int(self.offsets[stop]) - int(self.offsets[first])
        return 32 * rows + int(n_columns) * (int(np.sum(widths)) + 24 * int(np.sum(widths == 0)))

    def resident_columns(self, columns: np.ndarray, samples: slice = slice(None)) -> CodedLikelihoods:
        """The evidence of ``samples`` (a contiguous range) at ``columns`` (strictly increasing site indices) as a
        resident CodedLikelihoods: each sample's own dictionary rows, and its codes or dense rows at those sites only
        (release 4.7.1: a component's selected evidence, read once, so that repeated emission builds gather in
        memory, in parallel). Sample k of the copy is sample ``samples.start + k`` here."""
        if self.reader.closed:
            raise ValueError("window reader is closed")
        columns = np.asarray(columns, dtype=np.int64)
        first, stop, _ = samples.indices(len(self.widths))
        widths = np.array(self.widths[first:stop], dtype=np.uint8)
        source_rows = np.asarray(self.rows[first:stop], dtype=np.int64)
        rows = np.zeros(len(widths), dtype=np.int64)
        arrays = {}
        for width, name in ((1, "codes8"), (2, "codes16"), (4, "codes32"), (0, "dense")):
            members = np.flatnonzero(widths == width)
            rows[members] = np.arange(len(members), dtype=np.int64)
            arrays[name] = _window_rows(getattr(self, name), source_rows[members], columns)
        offsets = np.array(self.offsets[first : stop + 1], dtype=np.int64)
        low, high = int(offsets[0]), int(offsets[-1])
        local = np.arange(high - low, dtype=np.int64)
        return CodedLikelihoods(
            sites=len(columns),
            widths=widths,
            rows=rows,
            offsets=offsets - low,
            pairs=self.pairs.gather_rows(low, high, local),
            values=self.values.gather_rows(low, high, local),
            **arrays,
        )


def _window_rows(window: Any, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Rows ``rows`` of a window array at ``columns``, in the given order, reading each run of consecutive rows once."""
    result = np.empty((len(rows), len(columns), *window.shape[2:]), dtype=window.dtype)
    if not result.size:
        return result
    breaks = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1, len(rows)]
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        first = int(rows[left])
        result[left:right] = window.gather(slice(first, first + right - left), columns)
    return result


class PairWindows:
    """Logical count/mask windows sharing the GL codes and pair table handles."""

    def __init__(self, source: LikelihoodWindows, dense: Any, kind: ObservationKind) -> None:
        self.source, self.dense, self.kind = source, dense, kind
        self.shape = source.shape[:2] + ((2,) if kind == "counts" else ())
        self.dtype = np.dtype(np.int32 if kind == "counts" else np.bool_)

    def gather(self, samples: slice, columns: Any) -> np.ndarray:
        if self.source.reader.closed:
            raise ValueError("window reader is closed")
        return gather_pairs(self.source, self.dense, self.kind, samples, columns)

    def resident_bytes(self, n_columns: int, samples: slice = slice(None)) -> int:
        """Bytes of resident_columns' own rows (the dense fallback rows of ``samples``) over ``n_columns`` sites; the
        codes and dictionaries belong to the likelihoods' copy."""
        first, stop, _ = samples.indices(len(self.source.widths))
        dense = int(np.sum(np.asarray(self.source.widths[first:stop]) == 0))
        return int(n_columns) * dense * self.dtype.itemsize * (2 if self.kind == "counts" else 1)

    def resident_columns(
        self, columns: np.ndarray, source: CodedLikelihoods, samples: slice = slice(None)
    ) -> PairArray:
        """These observations of ``samples`` at ``columns``, borrowing ``source`` = the likelihoods' resident_columns
        of the same samples and sites; the dense fallback rows are numbered as in that copy."""
        if self.source.reader.closed:
            raise ValueError("window reader is closed")
        columns = np.asarray(columns, dtype=np.int64)
        first, stop, _ = samples.indices(len(self.source.widths))
        widths = np.asarray(self.source.widths[first:stop])
        rows = np.asarray(self.source.rows[first:stop], dtype=np.int64)
        return PairArray(source, _window_rows(self.dense, rows[widths == 0], columns), self.kind)


@contextmanager
def open_arrays(file: CheckpointFile, names: Sequence[str], *, expected_identity: str) -> Iterator[dict[str, Any]]:
    """Checked v1/v2 GL/count/mask windows; paired fields share physical codes."""
    version, encoding = _layout(file, expected_identity)
    projections = tuple(
        name
        for name in names
        if (encoding == "paired" and name in ("likelihoods", "counts", "observed"))
        or (encoding == "coded" and name == "likelihoods")
    )
    plain = tuple(name for name in names if name not in projections)
    if not projections:
        with record_windows(file, EVIDENCE, expected_identity, plain) as arrays:
            if arrays is None:
                raise KeyError(EVIDENCE)
            yield arrays
        return
    small = tuple("gl_" + name for name in GL_FIELDS if name not in GL_WINDOWS)
    record = read_items(file, EVIDENCE, expected_identity, ("schema", *small))
    if record is None:
        raise KeyError(EVIDENCE)
    require_schema(record.payload, version)
    fallback = tuple(name + "_dense" for name in projections if name != "likelihoods")
    windows = plain + tuple("gl_" + name for name in GL_WINDOWS) + fallback
    with record_windows(file, EVIDENCE, expected_identity, windows) as arrays:
        if arrays is None:
            raise KeyError(EVIDENCE)
        logical = {name: arrays[name] for name in plain}
        source = LikelihoodWindows(record.payload, arrays)
        for name in projections:
            logical[name] = (
                source
                if name == "likelihoods"
                else PairWindows(source, arrays[name + "_dense"], cast(ObservationKind, name))
            )
        yield logical


def load_observed(file: CheckpointFile, *, expected_identity: str, likelihoods: Any = None) -> np.ndarray | PairArray:
    """Borrow supplied GL codes; mask-only callers stream one dense mask.

    Never read an unused GL table/dense fallback merely to retain mask flags.
    Bounded mask-only consumers should use open_arrays instead.
    """
    _, encoding = _layout(file, expected_identity)
    if encoding != "paired" or not isinstance(likelihoods, CodedLikelihoods):
        return load_arrays(file, ("observed",), expected_identity=expected_identity)["observed"]
    source = likelihoods
    record = read_items(file, EVIDENCE, expected_identity, ("observed_dense",))
    if record is None:
        raise KeyError(EVIDENCE)
    return PairArray(source, record.payload["observed_dense"], "observed")


def load_observed_counts(
    file: CheckpointFile, *, expected_identity: str, rows: tuple[int, int] | None = None
) -> np.ndarray:
    """Exact per-sample observed totals from checked, bounded sample-major windows (``rows``: samples first..stop-1
    only, each range on its own reader)."""
    with open_arrays(file, ("observed",), expected_identity=expected_identity) as arrays:
        return observation_counts(arrays["observed"], axis=1, rows=rows)


def load_decisions(file: CheckpointFile, *, expected_identity: str | None = None) -> EvidenceDecisions:
    """Read site, sample and read-model decisions without decoding the evidence arrays."""
    record = read_items(file, EVIDENCE, _identity(file, expected_identity), ("schema", "decisions"))
    if record is None:
        raise KeyError(EVIDENCE)
    require_schema(record.payload, SUPPORTED_SCHEMAS)
    return from_payload(EvidenceDecisions, record.payload["decisions"])


def load_report(
    file: CheckpointFile, *, expected_identity: str | None = None, nthreads: int | None = None
) -> EvidenceDecisions:
    """Consumer entry for the typed decision record (decoder team belongs to the store)."""
    return load_decisions(file, expected_identity=expected_identity)


def load_diagnostics(file: CheckpointFile, *, expected_identity: str | None = None) -> EvidenceDiagnostics:
    """Read runtime diagnostics independently of scientific decisions."""
    record = read_items(file, EVIDENCE, _identity(file, expected_identity), ("schema", "diagnostics"))
    if record is None:
        raise KeyError(EVIDENCE)
    require_schema(record.payload, SUPPORTED_SCHEMAS)
    return from_payload(EvidenceDiagnostics, record.payload["diagnostics"])


@dataclass(frozen=True)
class Reader:
    """Results' structural reader; its derivation is rebuilt from independently bound inputs."""

    identity: str
    product: str
    counts: bool = False
    key: str = EVIDENCE
    schema: int = SCHEMA
    encoding: str = "dense"

    @property
    def items(self) -> tuple[str, ...]:
        if self.product == "evidence":
            names = tuple(name for name in ARRAYS if self.counts or name != "counts")
            excluded = ("likelihoods", "counts", "observed") if self.encoding == "paired" else ("likelihoods",)
            names = tuple(name for name in names if self.encoding == "dense" or name not in excluded)
            gl = tuple("gl_" + name for name in GL_FIELDS) if self.encoding != "dense" else ()
            fallback = (
                (("counts_dense",) if self.counts else ()) + ("observed_dense",) if self.encoding == "paired" else ()
            )
            return ("schema", "contig", "sample_ids", "payload_identity", *names, *gl, *fallback)
        if self.product == "diagnostics":
            return ("schema", "diagnostics")
        return ("schema", "decisions")

    def decode(self, payload: Mapping[str, Any]) -> ContigEvidence | EvidenceDecisions | EvidenceDiagnostics:
        require_schema(payload, self.schema)
        if self.product == "evidence":
            names = tuple(name for name in ARRAYS if name not in ("counts", "likelihoods", "observed"))
            gl = _decode_gl(payload, self.encoding)
            counts = payload.get("counts")
            if self.encoding == "paired":
                assert isinstance(gl, CodedLikelihoods)
                counts = PairArray(gl, payload["counts_dense"], "counts") if self.counts else None
                observed = PairArray(gl, payload["observed_dense"], "observed")
            else:
                observed = payload["observed"]
            return ContigEvidence(
                schema=self.schema,
                contig=payload["contig"],
                sample_ids=tuple(payload["sample_ids"]),
                payload_identity=payload["payload_identity"],
                counts=counts,
                likelihoods=gl,
                observed=observed,
                **{name: payload[name] for name in names},
            )
        if self.product == "diagnostics":
            return from_payload(EvidenceDiagnostics, payload["diagnostics"])
        return from_payload(EvidenceDecisions, payload["decisions"])


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    counts: bool = False,
) -> Reader | None:
    """Derive a checked evidence request from the init manifest and bound source payload.

    The input/payload record is the immutable ST1 input binding. No evidence envelope
    identity, source file path, clock or execution setting determines this request.
    """
    if product not in ("evidence", "summary", "diagnostics") or contig is None:
        raise InputError("invalid evidence reader request")
    if contig not in manifest["contigs"]:
        raise InputError(f"{contig}: contig is absent from the bound run manifest")
    file = store.contig(contig)
    if not file.has(EVIDENCE):
        return None
    consumed = bound_payload_identity(file)
    parameters = from_payload(EvidenceParameters, manifest["parameters"]["evidence"])
    policy = NumericalPolicy(**manifest["numerical_policy"])
    derivation = identity(
        parameters,
        consumed,
        tuple(manifest["sample_ids"]),
        reverse=bool(manifest["ties"]["reverse"]),
        numerical_policy=policy,
    )
    version, encoding = _layout(file, derivation)
    return Reader(derivation, product, counts, schema=version, encoding=encoding)
