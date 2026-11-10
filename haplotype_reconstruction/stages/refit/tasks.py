"""Execute independent contig refits and publish complete identity-bound products.

The engine owns the team; science sees immutable painting and founder-view evidence.
The per-site driver preserves sample order and its three scientific sweeps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configure import ResolvedConfiguration

import time

from ...contracts.evidence import ContigEvidence
from ...contracts.painting import Painting
from ...contracts.refit import FounderRefit
from ...core.errors import InvariantError
from ...core.ties import TieSettings
from ...science.refit.parameters import RefitParameters
from ...science.refit.run import refit_chromosome, summary
from ...store.container import CheckpointFile
from ...store.protocol import claim, mark_complete, publish
from ..framework import RecordKey, ResourceRequest, Task
from . import records

_DEFAULT_TIES = TieSettings()

__all__ = ["run_contig", "task"]


def run_contig(
    file: CheckpointFile,
    expected: str,
    painting: Painting,
    evidence: ContigEvidence,
    parameters: RefitParameters,
    *,
    payload_identity: str,
    painting_identity: str | None = None,
    evidence_identity: str | None = None,
    ties: TieSettings = _DEFAULT_TIES,
    threads: int = 1,
) -> FounderRefit:
    """Resume or refit; reject mismatched payload/sample axes before any science."""
    if painting_identity is not None or evidence_identity is not None:
        if painting_identity is None or evidence_identity is None:
            raise InvariantError("refit needs both painting and evidence source identities")
        from ..painting.records import require_evidence_binding

        require_evidence_binding(file, painting_identity, evidence_identity)
    if evidence.payload_identity != payload_identity:
        raise InvariantError("refit evidence payload binding differs")
    if painting.contig != evidence.contig or painting.sample_ids != tuple(evidence.sample_ids):
        raise InvariantError("refit painting/evidence contig or sample axes differ")
    with claim(file, records.FOUNDER_REFIT, expected) as needed:
        if not needed:
            result = records.load(file, expected, painting=painting)
            if result is None:
                raise InvariantError("completed refit claim has no product")
        else:
            wall, cpu = time.perf_counter(), time.process_time()
            result = refit_chromosome(painting, evidence, parameters, reverse=ties.reverse)
            publish(
                file,
                "founder_refit/diagnostics",
                expected,
                dict(
                    schema=1,
                    wall_seconds=time.perf_counter() - wall,
                    cpu_seconds=time.process_time() - cpu,
                    threads=threads,
                ),
            )
            records.write(file, expected, result, summary(result))
        mark_complete(
            file,
            "founder_refit",
            {records.FOUNDER_REFIT: expected, records.SUMMARY: expected, records.IDENTITY: expected},
        )
        return result


from ..inputs.products import task_inputs, required_product, required_dense_evidence
from ..context import StageRun


def run(
    root: str, contig: str, resolved: ResolvedConfiguration, ties: TieSettings, *, team: int = 1
) -> str:
    """Pass the checked immutable painting and evidence to founder refit's one contig action."""
    from ..painting import records as painting_records
    from ..refit.tasks import run_contig
    from ..refit import records

    reader, store = task_inputs(root)
    painting = required_product(reader, painting_records, "painting", contig)
    evidence = required_dense_evidence(reader, contig)
    expected = records.expected(store, reader.manifest, product="refit", contig=contig).identity
    run_contig(
        store.contig(contig),
        expected,
        painting,
        evidence,
        resolved.parameters.founder_refit,
        payload_identity=evidence.payload_identity,
        ties=ties,
        threads=team,
    )
    return expected


def task(key, context: StageRun) -> Task:
    """Declare the production founder-refit action and its published record spellings."""
    return Task(
        key, run, (context.root, key.scope, context.resolved, context.ties),
        outputs=tuple(RecordKey(key.scope, name) for name in ("founder_refit", "founder_refit/summary", "identity/founder_refit")),
        resources=ResourceRequest(1, context.cores), thread_argument="team",
    )
