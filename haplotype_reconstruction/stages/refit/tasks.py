"""Execute independent contig refits and publish complete identity-bound products.

The engine owns the team; science sees immutable painting and founder-view evidence.
The per-site driver preserves sample order and its three scientific sweeps.
"""

from __future__ import annotations

import time

from ...contracts.evidence import ContigEvidence
from ...contracts.painting import Painting
from ...contracts.refit import FounderRefit
from ...core.errors import InvariantError
from ...core.ties import TieSettings
from ...science.refit.parameters import RefitParameters
from ...science.refit.run import refit_chromosome, summary
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, claim, mark_complete, publish
from ..framework import ALL_CORES, RecordKey, ResourceRequest, Task, TaskKey
from . import records

_DEFAULT_TIES = TieSettings()

__all__ = ["identity", "run_contig", "refit_task"]


def identity(
    parameters: RefitParameters,
    *,
    contig: str,
    sample_ids: tuple[str, ...],
    painting_identity: str,
    evidence_identity: str,
    payload_identity: str,
    ties: TieSettings = _DEFAULT_TIES,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    return Derivation(
        code_dependency_identity(("science.refit", "stages.refit.tasks", "stages.refit.records")),
        dict(parameters=parameters, contig=contig, sample_ids=sample_ids),
        numerical_policy or NumericalPolicy.current(),
        dict(painting=painting_identity, evidence=evidence_identity, payload=payload_identity),
        tie_reverse=ties.reverse,
        tie_keep=ties.keep,
        schema=records.SCHEMA,
    ).identity


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


def refit_task(
    contig: str,
    *,
    file: CheckpointFile,
    painting: Painting,
    evidence: ContigEvidence,
    parameters: RefitParameters,
    painting_identity: str,
    evidence_identity: str,
    payload_identity: str,
    ties: TieSettings = _DEFAULT_TIES,
    numerical_policy: NumericalPolicy | None = None,
    dependencies: tuple[TaskKey, ...] = (),
    max_cores: int = ALL_CORES,
) -> Task:
    """Declare one affinity-scoped contig task; no live resources enter identity."""
    if max_cores < 1:
        raise InvariantError("refit max_cores must be positive")
    expected = identity(
        parameters,
        contig=contig,
        sample_ids=tuple(evidence.sample_ids),
        painting_identity=painting_identity,
        evidence_identity=evidence_identity,
        payload_identity=payload_identity,
        ties=ties,
        numerical_policy=numerical_policy,
    )
    if any(c.panel is None for c in painting.components):
        raise InvariantError("refit resources require painted panels")
    working = max(
        (
            4 * len(evidence.sample_ids) * len(c.panel.positions) + 48 * c.panel.founders**2 * max_cores
            for c in painting.components
            if c.panel is not None
        ),
        default=0,
    )
    return Task(
        TaskKey("refit", "contig", contig),
        run_contig,
        (file, expected, painting, evidence, parameters),
        kwargs=dict(
            payload_identity=payload_identity,
            painting_identity=painting_identity,
            evidence_identity=evidence_identity,
            ties=ties,
        ),
        depends_on=dependencies,
        inputs=(RecordKey(contig, "painting"), RecordKey(contig, "evidence")),
        outputs=(RecordKey(contig, records.FOUNDER_REFIT), RecordKey(contig, records.SUMMARY)),
        resources=ResourceRequest(1, max_cores, (512 << 20) + working),
        affinity=contig,
        thread_argument="threads",
    )
