"""Bind source content and execute the per-contig evidence transition (EC11/25).

Owns payload caching, task resources and publication; all calculations live in
science/evidence. Simulation identity is design plus realised seed, with no VCF
payload. Method: docs/methods/evidence.md#stage.4.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ...contracts.source import ContigReads
from ...core.concurrency import threaded_each
from ...engine.cpu import thread_team
from ...science.evidence.parameters import EvidenceParameters
from ...science.evidence.run import prepare_contig
from ...store.protocol import NumericalPolicy, bind_payload_identity, cached_payload_identity, payload_identity
from ...store.store import CheckpointStore
from ..framework import RecordKey, ResourceRequest, Task
from ..inputs.sources import CanonicalOrderSource
from . import records

__all__ = ["run_contig", "task"]


def _search_alleles_reversed(ref: Iterable[str], alt: Iterable[str]) -> bool:
    # The first distinct nucleotide pair chooses one frame for the contig. A
    # global REF/ALT exchange toggles this bit, even for self-complementary panels.
    for ref_base, alt_base in zip(ref, alt, strict=True):
        first, second = ref_base.upper(), alt_base.upper()
        if first != second:
            return first > second
    return False


def run_contig(
    root: str | Path, source: CanonicalOrderSource, contig: str, parameters: EvidenceParameters, *, team: int = 1
) -> str:
    """Prepare/publish one contig, or resume an identical derivation without loading its arrays.

    The source is already canonical and init has bound simulation seeds. A changed
    consumed payload refuses resume. Only this contig's capability is writable.
    """
    store = CheckpointStore(root, nthreads=team)
    file = store.scatter_view(contig).contig
    decoded: list[tuple[ContigReads, bool]] = []

    def read_vcf() -> str:
        reads, payload = source.read_with_payload(contig, team)
        decoded.append((reads, _search_alleles_reversed(payload.ref, payload.alt)))
        return payload_identity(payload)
    if source.kind == "vcf":
        consumed = cached_payload_identity(file, source.path, read_vcf)
    else:
        consumed = bind_payload_identity(file, source.payload_identity(contig))
    from ...contracts.payload import to_payload

    manifest = dict(
        contigs=(contig,), sample_ids=source.sample_ids(), parameters={"evidence": to_payload(parameters)},
        ties={"reverse": source.reverse}, numerical_policy=to_payload(NumericalPolicy.current()),
    )
    derivation = records.expected(store, manifest, product="evidence", contig=contig).identity
    if records.complete(file, derivation):
        return derivation
    if source.kind == "vcf":
        if not decoded:
            bind_payload_identity(file, read_vcf())
        reads, reversed_alleles = decoded.pop()
    else:
        reads = source.reads(contig, team)
        ref, alt, _ = source.site_alleles(contig, reads.positions[:1])
        reversed_alleles = _search_alleles_reversed(ref, alt)
    with thread_team(default=team):
        prepared = prepare_contig(
            reads, parameters, team, map_work=threaded_each, reverse=source.reverse,
            search_alleles_reversed=reversed_alleles,
        )
    # Pair-backed output no longer owns coded raw count rows. Release the input
    # before publication; dense fallback rows remain alive through prepared.
    del reads
    records.write_read_sharing(file, prepared.read_sharing, derivation)
    records.write(file, prepared, derivation, payload_identity=consumed)
    return derivation


def task(key, context) -> Task:
    """Declare one production evidence contig with its unchanged runtime team."""
    from ...core.errors import SettingsError

    if context.source is None:
        raise SettingsError("evidence requires a checked observation source")
    return Task(key, run_contig,
                args=(context.root, context.source, key.scope, context.resolved.parameters.evidence),
                outputs=(RecordKey(key.scope, records.EVIDENCE),),
                resources=ResourceRequest(min_cores=1, max_cores=context.cores),
                affinity=key.scope, thread_argument="team")
