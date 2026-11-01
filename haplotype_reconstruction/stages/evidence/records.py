"""Publish and selectively read one contig's immutable evidence product (EC25).

Owns the flat stored array layout and schema adapter. Science does no storage I/O;
identity binds parameters, numerical policy, canonical samples and consumed payload.
Method: docs/methods/evidence.md#records; B evidence/records.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...contracts.evidence import ContigEvidence, EvidenceDecisions, EvidenceDiagnostics
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.errors import InputError
from ...core.ties import TieCount, TieCounts
from ...science.evidence.parameters import EvidenceParameters
from ...science.evidence.run import PreparedEvidence
from ...store.container import CheckpointFile, RecordInfo
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, bound_payload_identity, publish, read_items
from ...store.store import CheckpointStore

__all__ = [
    "EVIDENCE",
    "SCHEMA",
    "ARRAYS",
    "identity",
    "write",
    "load",
    "load_arrays",
    "load_report",
    "load_decisions",
    "load_diagnostics",
    "complete",
    "progress",
    "result_reader",
]
EVIDENCE = "evidence"
SCHEMA = 1
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
    """Publish flat selectively readable arrays and typed schema-1 decisions."""
    evidence = replace(prepared.evidence, payload_identity=payload_identity)
    if evidence.counts is None:
        raise InputError("the evidence record keeps the counts")
    payload = to_payload(evidence)
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
    require_schema(record.payload, SCHEMA)
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
    names = tuple(name for name in ARRAYS if counts or name != "counts")
    record = read_items(
        file,
        EVIDENCE,
        _identity(file, expected_identity),
        ("schema", "contig", "sample_ids", "payload_identity", *names),
    )
    if record is None:
        raise KeyError(EVIDENCE)
    payload = record.payload
    require_schema(payload, SCHEMA)
    return ContigEvidence(
        contig=payload["contig"],
        sample_ids=tuple(payload["sample_ids"]),
        counts=payload.get("counts"),
        payload_identity=payload["payload_identity"],
        **{name: payload[name] for name in names if name != "counts"},
    )


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
    record = read_items(file, EVIDENCE, _identity(file, expected_identity), ("schema", *names), out=out)
    if record is None:
        raise KeyError(EVIDENCE)
    require_schema(record.payload, SCHEMA)
    return {name: record.payload[name] for name in names}


def load_decisions(file: CheckpointFile, *, expected_identity: str | None = None) -> EvidenceDecisions:
    """Read site, sample and read-model decisions without decoding the evidence arrays."""
    record = read_items(file, EVIDENCE, _identity(file, expected_identity), ("schema", "decisions"))
    if record is None:
        raise KeyError(EVIDENCE)
    require_schema(record.payload, SCHEMA)
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
    require_schema(record.payload, SCHEMA)
    return from_payload(EvidenceDiagnostics, record.payload["diagnostics"])


@dataclass(frozen=True)
class Reader:
    """Results' structural reader; its derivation is rebuilt from independently bound inputs."""

    identity: str
    product: str
    counts: bool = False
    key: str = EVIDENCE
    schema: int = SCHEMA

    @property
    def items(self) -> tuple[str, ...]:
        if self.product == "evidence":
            names = tuple(name for name in ARRAYS if self.counts or name != "counts")
            return ("schema", "contig", "sample_ids", "payload_identity", *names)
        if self.product == "diagnostics":
            return ("schema", "diagnostics")
        return ("schema", "decisions")

    def decode(self, payload: Mapping[str, Any]) -> ContigEvidence | EvidenceDecisions | EvidenceDiagnostics:
        require_schema(payload, self.schema)
        if self.product == "evidence":
            selected = {name: payload[name] for name in self.items}
            selected.setdefault("counts", None)
            return from_payload(ContigEvidence, selected)
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
    return Reader(derivation, product, counts)
