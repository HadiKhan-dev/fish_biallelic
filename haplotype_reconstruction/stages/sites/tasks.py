"""Publish the option-free released site record from evidence and its source.

Owns OU01 kept masks and OU02 r5-only common alleles; never reads truth or export
options. Method: docs/methods/export.md#sites. Rules and exact integer comparisons
use strict majority among the non-excluded samples; there is no random stream or tie band.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

import numpy as np

from ...core.errors import InvariantError
from .records import SitesAdapter

if TYPE_CHECKING:
    from ...contracts.evidence import ContigEvidence, EvidenceDecisions
    from ...contracts.release import ReleasedSites
    from ...store.container import CheckpointFile

__all__ = ["SiteEvidence", "SiteSource", "SiteDecisions", "build", "evidence_sites"]
# Bit k denotes applied site-quality rule rk.
R5_ONLY = np.uint8(1 << 5)


class SiteSource(Protocol):
    """The ObservationSource export methods used by the sites task."""

    def site_alleles(self, contig: str, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, int | None]: ...
    def allele_totals(
        self, contig: str, positions: np.ndarray, samples: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class VerifiedSource(Protocol):
    """A source that verifies fresh consumed content against the evidence binding."""

    def verified_view(self, contig: str, expected_identity: str, *, team: int = 1) -> SiteSource: ...


@dataclass(frozen=True)
class SiteEvidence:
    """Evidence projection: input/kept bp axes and the original rule/QC report."""

    contig: str
    sample_ids: tuple[str, ...]
    positions: np.ndarray
    input_positions: np.ndarray
    founder_excluded: np.ndarray
    rule_bits: np.ndarray | None
    applied_rules: tuple[str, ...]
    enabled: bool
    applied: bool
    reason: str | None  # why the site filter was off or not applied; None when it was applied
    dropped: Mapping[str, int]
    synthetic: bool = False


@dataclass(frozen=True)
class SiteDecisions:
    """Exact operands of outputs kept and common-allele decisions, on all input rows."""

    removed_bits: np.ndarray
    ref_total: np.ndarray
    alt_total: np.ndarray
    source: np.ndarray


def evidence_sites(evidence: SiteEvidence) -> tuple[np.ndarray, np.ndarray, str, tuple[str, ...]]:
    """Validate the evidence's kept membership and applied-rule mask without mutation.

    Returns bool[M] kept, uint8[M] removed bits, filter-state text and counted names.
    Raises InvariantError if the kept and original input axes or rules disagree.
    """
    kept = np.isin(evidence.input_positions, evidence.positions, assume_unique=True)
    if int(kept.sum()) != len(evidence.positions):
        raise InvariantError(f"{evidence.contig}: evidence sites are not among the input sites")
    mask = np.uint8(sum(1 << int(rule[1:]) for rule in evidence.applied_rules))
    bits = (
        np.zeros(len(evidence.input_positions), np.uint8)
        if evidence.rule_bits is None
        else np.asarray(evidence.rule_bits, dtype=np.uint8) & mask
    )
    if not np.array_equal(bits == 0, kept):
        raise InvariantError(f"{evidence.contig}: site-filter rule bits disagree with the kept sites")
    state = "off" if not evidence.enabled else f"not applied ({evidence.reason})" if not evidence.applied else "applied"
    counted = tuple(
        name for name, excluded in zip(evidence.sample_ids, evidence.founder_excluded, strict=False) if not excluded
    )
    return kept, bits, state, counted


@overload
def build(source: SiteSource, evidence: SiteEvidence, *, trace: Literal[False] = False) -> ReleasedSites: ...


@overload
def build(
    source: SiteSource, evidence: SiteEvidence, *, trace: Literal[True]
) -> tuple[ReleasedSites, SiteDecisions]: ...


def build(
    source: SiteSource, evidence: SiteEvidence, *, trace: bool = False
) -> ReleasedSites | tuple[ReleasedSites, SiteDecisions]:
    """Build ReleasedSites with exact r5-only fills through ObservationSource methods.

    Source calls may raise InputError for a changed payload; inconsistent evidence
    raises InvariantError. Returns a new immutable record; trace adds exact operands.
    No export setting affects this record and no source input is mutated.
    """
    from ...contracts.release import ReleasedSites

    kept, bits, state, counted = evidence_sites(evidence)
    positions = np.asarray(evidence.input_positions, dtype=np.int64)
    ref, alt, length = source.site_alleles(evidence.contig, positions)
    r5_only = bits == R5_ONLY
    ref_total: np.ndarray = np.zeros(len(positions), dtype=np.int64)
    alt_total: np.ndarray = np.zeros(len(positions), dtype=np.int64)
    provenance: np.ndarray = np.zeros(len(positions), dtype=np.int8)
    if np.any(r5_only):
        ref_total[r5_only], alt_total[r5_only], provenance[r5_only] = source.allele_totals(
            evidence.contig, positions[r5_only], counted
        )
    common: np.ndarray = np.full(len(positions), -1, dtype=np.int8)
    common[r5_only & (ref_total > alt_total)] = 0
    common[r5_only & (alt_total > ref_total)] = 1
    decided = r5_only & (common >= 0)
    counts = dict(
        ref=int(np.sum(common == 0)),
        alt=int(np.sum(common == 1)),
        tie=int(np.sum(r5_only & (common < 0))),
        from_reads=int(np.sum(decided & (provenance == 1))),
        from_gt=int(np.sum(decided & (provenance == 2))),
    )
    result = ReleasedSites(
        contig=evidence.contig,
        positions=positions,
        kept=kept,
        removed_bits=bits,
        ref=ref,
        alt=alt,
        common=common,
        common_counts=counts,
        contig_length=length,
        site_filter=state,
        applied_rules=evidence.applied_rules,
        dropped=evidence.dropped,
        synthetic=evidence.synthetic,
    )
    return (result, SiteDecisions(bits, ref_total, alt_total, provenance)) if trace else result


def from_evidence(evidence: ContigEvidence, report: EvidenceDecisions, *, synthetic: bool = False) -> SiteEvidence:
    """Project evidence's typed evidence/report onto the source-site boundary; no dictionary walk."""
    quality = report.site_quality
    return SiteEvidence(
        contig=evidence.contig,
        sample_ids=evidence.sample_ids,
        positions=evidence.positions,
        input_positions=report.input_positions,
        founder_excluded=evidence.founder_excluded,
        rule_bits=quality.rule_bits,
        applied_rules=tuple(quality.applied_rules),
        enabled=quality.enabled,
        applied=quality.applied,
        reason=quality.reason,
        dropped=report.dropped,
        synthetic=synthetic,
    )


def publish_sites(
    file: CheckpointFile,
    source: VerifiedSource,
    evidence: SiteEvidence,
    *,
    expected: str,
    payload_identity: str,
    team: int = 1,
) -> ReleasedSites:
    """Verify the source once, build/resume and publish sites before output tasks read them.

    The source's verified_view binds a fresh VCF payload to the evidence digest and
    shares it between both protocol methods. Simulation verifies its design binding.
    Publication uses the common stage protocol and never stores output settings.
    """
    from ..framework import load_product, publish_product

    view = source.verified_view(evidence.contig, payload_identity, team=team)
    adapter = SitesAdapter()
    previous = load_product(file, "sites", expected, adapter)
    if previous is not None:
        return previous
    result = build(view, evidence)
    publish_product(file, "sites", expected, result, adapter)
    return result


from ..inputs.products import task_inputs, required_product
from ..context import StageRun
from ..framework import Task, TaskKey, RecordKey, ResourceRequest


def run(root: str, contig: str, source: Any, *, team: int = 1) -> str:
    """Publish the option-independent site product after checked evidence."""
    from ..evidence import records
    from . import records as sites_records

    reader, store = task_inputs(root)
    evidence = required_product(reader, records, "evidence", contig)
    report = required_product(reader, records, "summary", contig)
    view = from_evidence(evidence, report, synthetic=reader.synthetic)
    expected = sites_records.expected(store, reader.manifest, product="sites", contig=contig)
    publish_sites(
        store.contig(contig),
        source,
        view,
        expected=expected.identity,
        payload_identity=evidence.payload_identity,
        team=team,
    )
    return expected.identity


def task(key, context: StageRun) -> Task:
    """Declare the production option-independent site publisher."""
    return Task(key, run, (context.root, key.scope, context.source),
                depends_on=(TaskKey("evidence", "contig", key.scope),),
                inputs=(RecordKey(key.scope, "evidence"),), outputs=(RecordKey(key.scope, "sites"),),
                resources=ResourceRequest(1, context.cores), thread_argument="team")
