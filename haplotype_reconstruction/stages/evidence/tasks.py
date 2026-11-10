"""Bind source content and execute the per-contig evidence transition (EC11/25).

Owns payload caching, task resources and publication; all calculations live in
science/evidence. Simulation identity is design plus realised seed, with no VCF
payload. Method: docs/methods/evidence.md#stage.4.
"""

from __future__ import annotations

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


def run_contig(
    root: str | Path, source: CanonicalOrderSource, contig: str, parameters: EvidenceParameters, *, team: int = 1
) -> str:
    """Prepare/publish one contig, or resume an identical derivation without loading its arrays.

    The source is already canonical and init has bound simulation seeds. A changed
    consumed payload refuses resume. Only this contig's capability is writable.
    """
    store = CheckpointStore(root, nthreads=team)
    file = store.scatter_view(contig).contig
    decoded: list[ContigReads] = []
    if source.kind == "vcf":

        def compute_identity() -> str:
            reads, payload = source.read_with_payload(contig, team)
            decoded.append(reads)
            return payload_identity(payload)

        consumed = cached_payload_identity(file, source.path, compute_identity)
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
    if decoded:
        reads = decoded.pop()
    elif source.kind == "vcf":
        reads, payload = source.read_with_payload(contig, team)
        bind_payload_identity(file, payload_identity(payload))
        del payload
    else:
        reads = source.reads(contig, team)
    with thread_team(default=team):
        prepared = prepare_contig(reads, parameters, team, map_work=threaded_each, reverse=source.reverse)
    # Pair-backed output no longer owns coded raw count rows. Release the input
    # before publication; dense fallback rows remain alive through prepared.
    del reads
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
