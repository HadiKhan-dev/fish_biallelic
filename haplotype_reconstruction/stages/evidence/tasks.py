"""Bind source content and execute the per-contig evidence transition (EC11/25).

Owns payload caching, task resources and publication; all calculations live in
science/evidence. Simulation identity is design plus realised seed, with no VCF
payload. Method: docs/methods/evidence.md#stage; WP_EV3 section 4.4.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ...contracts.source import ContigReads
from ...engine.runtime import task_team
from ...science.evidence.parameters import EvidenceParameters
from ...science.evidence.run import prepare_contig
from ...store.protocol import bind_payload_identity, cached_payload_identity, payload_identity
from ...store.store import CheckpointStore
from ..framework import RecordKey, ResourceRequest, Task, TaskKey
from ..inputs.sources import CanonicalOrderSource
from . import records

__all__ = ["run_contig", "tasks"]


def run_contig(
    root: str | Path, source: CanonicalOrderSource, contig: str, parameters: EvidenceParameters, *, team: int = 1
) -> str:
    """Prepare/publish one contig, or resume an identical derivation without loading its arrays.

    The source is already canonical and init has bound simulation seeds. A changed
    consumed payload refuses resume. Only this contig's capability is writable.
    """
    file = CheckpointStore(root, nthreads=team).scatter_view(contig).contig
    decoded: list[ContigReads] = []
    if source.kind == "vcf":

        def compute_identity() -> str:
            reads, payload = source.read_with_payload(contig, team)
            decoded.append(reads)
            return payload_identity(payload)

        consumed = cached_payload_identity(file, source.path, compute_identity)
    else:
        consumed = bind_payload_identity(file, source.payload_identity(contig))
    derivation = records.identity(parameters, consumed, source.sample_ids(), reverse=source.reverse)
    if records.complete(file, derivation):
        return derivation
    if decoded:
        reads = decoded.pop()
    elif source.kind == "vcf":
        reads, payload = source.read_with_payload(contig, team)
        bind_payload_identity(file, payload_identity(payload))
    else:
        reads = source.reads(contig, team)
    with task_team(team):
        prepared = prepare_contig(reads, parameters, team, reverse=source.reverse)
    records.write(file, prepared, derivation, payload_identity=consumed)
    return derivation


def tasks(
    root: str | Path,
    source: CanonicalOrderSource,
    parameters: EvidenceParameters,
    *,
    contigs: Sequence[str] | None = None,
    maximum_cores: int = 16,
    depends_on: tuple[TaskKey, ...] = (),
) -> tuple[Task, ...]:
    """Declare one evidence task per requested contig, with a bounded runtime team."""
    selected = source.contigs() if contigs is None else tuple(contigs)
    return tuple(
        Task(
            key=TaskKey("evidence", "contig", contig),
            function=run_contig,
            args=(str(root), source, contig, parameters),
            depends_on=depends_on,
            outputs=(RecordKey(contig, records.EVIDENCE),),
            resources=ResourceRequest(min_cores=1, max_cores=maximum_cores),
            affinity=contig,
            thread_argument="team",
        )
        for contig in selected
    )
