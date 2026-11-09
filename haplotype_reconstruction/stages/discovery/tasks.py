"""Declare independent discovery blocks and publish complete results by derivation.

A block is done exactly when its scientific record exists under the expected
identity, including an explicit uninformative result. Plan and completion records
are checked consequences, never shortcuts around missing blocks. Timings and
resource models live here; scientific functions receive explicit settings only.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, replace

import numpy as np

from ...contracts.discovery import DiscoveryBlockResult, DiscoveryPlan
from ...core.errors import InvariantError
from ...core.ties import TieSettings
from ...science.discovery.diagnostics import BlockDiagnostics, DiagnosticOptions
from ...science.discovery.fixed_k.workspace import search_bytes
from ...science.discovery.parameters import PROBABILITY_BAND, DiscoveryParameters
from ...science.discovery.release import DiscoveredBlock
from ...science.discovery.run import BlockEvidence, discover_block
from ...store.container import CheckpointFile
from ...store.protocol import NumericalPolicy, claim, mark_complete, publish, resume
from ..framework import ALL_CORES, RecordKey, ResourceRequest, Task, TaskKey
from .records import PLAN, BlockAdapter, PlanAdapter, block_key, diagnostic_key
from .records import identities as record_identities

__all__ = [
    "discovery_tasks",
    "execute_block",
    "publish_plan",
    "complete_contig",
    "complete_contig_receipt",
    "block_task_bytes",
]

# Resource-only provenance: B/discovery/stage.py TASK_BASE_BYTES.
_TASK_BASE_BYTES = 1280 << 20
_DEFAULT_TIES = TieSettings()
_DEFAULT_DIAGNOSTICS = DiagnosticOptions()


def block_task_bytes(evidence: BlockEvidence, parameters: DiscoveryParameters, *, founders: int = 32) -> int:
    """Report the baseline working-set model as a function of K; never limit science."""
    n, sites = evidence.likelihoods.shape[:2]
    kept = int(np.count_nonzero(evidence.kept))
    threshold = float(parameters.residual_births.min_soft_responsibility) - PROBABILITY_BAND
    rows_per_sample = math.ceil(2.0 / threshold) if threshold > 0.0 else founders
    records = n * rows_per_sample
    a = parameters.residual_births.landmark_budget
    b = parameters.search.seed_landmark_budget
    residual = 5 * records * a + 4 * a * a + 5 * n * min(n, a) if a and records > a else records * records + 2 * n * n
    seeds = 5 * n * b + 5 * b * b if b and n > b else 3 * n * n
    return (
        _TASK_BASE_BYTES
        + 48 * n * sites
        + 25 * n * kept
        + search_bytes(n, kept, founders, parameters.model.snps_per_bin)
        + 8 * (residual + seeds)
    )


def publish_plan(file: CheckpointFile, identity: str, blocks: int) -> int:
    """Publish or check the block count before any block task executes."""
    adapter = PlanAdapter()
    existing = resume(file, PLAN, identity)
    if existing is not None:
        if adapter.decode(existing.payload).blocks != blocks:
            raise InvariantError("discovery plan block count differs from evidence")
        return blocks
    publish(file, PLAN, identity, adapter.encode(DiscoveryPlan(blocks=blocks)))
    return blocks


def execute_block(
    file: CheckpointFile,
    identity: str,
    evidence: BlockEvidence,
    parameters: DiscoveryParameters,
    *,
    ties: TieSettings = _DEFAULT_TIES,
    diagnostics: DiagnosticOptions = _DEFAULT_DIAGNOSTICS,
    threads: int = 1,
) -> DiscoveredBlock | None:
    """Resume or compute one block; publish diagnostics before the complete scientific record."""
    key = block_key(evidence.block)
    adapter = BlockAdapter()
    with claim(file, key, identity) as needed:
        if not needed:
            record = resume(file, key, identity)
            if record is None:
                raise InvariantError("completed discovery claim has no record")
            return adapter.decode(record.payload).result
        collector = BlockDiagnostics(diagnostics)
        wall, cpu = time.perf_counter(), time.process_time()
        # Release 4.9: the block reads its task's live share at every kernel choice (a heavy block that started on
        # one CPU beside its siblings ran serially to the end, 50 s for chr1 0.5x block 287; 11 s on 32 threads).
        from ...engine.scheduler import live_threads

        result = discover_block(
            evidence,
            parameters,
            reverse=ties.reverse,
            keep=ties.keep,
            diagnostics=collector,
            threads=live_threads(),
        )
        measured = replace(
            collector.snapshot(), wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu
        )
        publish(file, diagnostic_key(evidence.block), identity, {"schema": 1, **asdict(measured)})
        publish(
            file,
            key,
            identity,
            adapter.encode(
                DiscoveryBlockResult(
                    block=evidence.block, kind="uninformative" if result is None else "discovered", result=result
                )
            ),
        )
        return result


def complete_contig(file: CheckpointFile, identities: dict[str, str]) -> None:
    """Derive contig completion from the plan and every per-block scientific record."""
    mark_complete(file, "discovery", identities)


def complete_contig_receipt(file: CheckpointFile, contig: str, identities: dict[str, str]) -> dict[RecordKey, str]:
    """complete_contig, replying with the records it checked (release 4.10): mark_complete has verified every
    record's envelope against ``identities``, so the run's completion step takes the receipt instead of deriving and
    checking the whole block domain again in the coordinator (one contig at a time, after the graph)."""
    complete_contig(file, identities)
    return {RecordKey(contig, key): identity for key, identity in identities.items()}


def discovery_tasks(
    contig: str,
    blocks: Sequence[BlockEvidence],
    parameters: DiscoveryParameters,
    *,
    file: CheckpointFile,
    evidence_identity: str,
    numerical_policy: NumericalPolicy,
    ties: TieSettings = _DEFAULT_TIES,
    diagnostics: DiagnosticOptions = _DEFAULT_DIAGNOSTICS,
    dependencies: tuple[TaskKey, ...] = (),
    max_cores: int = ALL_CORES,
    estimated_founders: int = 32,
) -> tuple[Task, ...]:
    """Declare plan, independent blocks and final completion for one evidence contig.

    Blocks must cover its complete contiguous index domain in order, including
    uninformative blocks. Upstream evidence identity and policy are supplied by
    orchestration; task execution and completion order cannot change derivation.
    """
    if tuple(block.block for block in blocks) != tuple(range(len(blocks))):
        raise InvariantError("discovery tasks require the complete ordered block domain")
    if max_cores < 1:
        raise InvariantError("discovery max_cores must be positive")
    identities = record_identities(
        parameters,
        contig=contig,
        blocks=len(blocks),
        evidence_identity=evidence_identity,
        numerical_policy=numerical_policy,
        ties=ties,
    )
    plan_identity = identities[PLAN]
    plan_key = TaskKey("discovery", "plan", contig)
    tasks = [
        Task(
            plan_key,
            publish_plan,
            (file, plan_identity, len(blocks)),
            depends_on=dependencies,
            inputs=(RecordKey(contig, "evidence"),),
            outputs=(RecordKey(contig, PLAN),),
            affinity=contig,
        )
    ]
    block_tasks = []
    for block in blocks:
        key = TaskKey("discovery", "block", contig, str(block.block))
        derivation = identities[block_key(block.block)]
        block_tasks.append(key)
        tasks.append(
            Task(
                key,
                execute_block,
                (file, derivation, block, parameters),
                kwargs=dict(ties=ties, diagnostics=diagnostics),
                depends_on=(plan_key,),
                inputs=(RecordKey(contig, "evidence"), RecordKey(contig, PLAN)),
                outputs=(RecordKey(contig, block_key(block.block)), RecordKey(contig, diagnostic_key(block.block))),
                resources=ResourceRequest(
                    1, max_cores, block_task_bytes(block, parameters, founders=estimated_founders)
                ),
                affinity=None,
                thread_argument="threads",
            )
        )
    tasks.append(
        Task(
            TaskKey("discovery", "complete", contig),
            complete_contig_receipt,
            (file, contig, identities),
            depends_on=tuple(block_tasks) or (plan_key,),
            inputs=tuple(RecordKey(contig, key) for key in identities),
            affinity=contig,
        )
    )
    return tuple(tasks)
