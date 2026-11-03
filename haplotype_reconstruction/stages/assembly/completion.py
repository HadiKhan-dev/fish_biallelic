"""Stage founder completion and construct the shared assembly evidence view.

Owns record decoding, bounded execution and completion publication; AS1 owns
all completion decisions. No scientific result depends on worker allocation.
Method: WP_AS6 §4.1/4.6; WP_AS1 §6.2.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from haplotype_reconstruction.contracts.assembly import CompletionProduct
from haplotype_reconstruction.contracts.evidence import ContigEvidence
from haplotype_reconstruction.contracts.panels import LocalPanel
from haplotype_reconstruction.contracts.payload import to_payload
from haplotype_reconstruction.core.ties import TieSettings
from haplotype_reconstruction.engine.graph import ResourceRequest, ResultRef, Task, TaskKey
from haplotype_reconstruction.engine.runtime import task_team, thread_team
from haplotype_reconstruction.engine.workers import (
    SharedArrayRef,
    attach_shared_array,
    close_shared_memory,
)
from haplotype_reconstruction.science.assembly.completion.parameters import CompletionParameters
from haplotype_reconstruction.science.assembly.completion.run import (
    CompletionBlockResult,
    CompletionPlan,
    CompletionWorkspace,
    block_memory_bytes,
    complete,
    complete_block,
    completion_blocks,
    finish_completion,
    plan_completion,
)
from haplotype_reconstruction.science.assembly.evidence import AssemblyEvidence
from haplotype_reconstruction.science.assembly.run import AssemblyInputs
from haplotype_reconstruction.stages.framework import Expansion

from .records import COMPLETION, RecordCapability, decode_completion, encode_completion


def build_evidence(evidence: ContigEvidence, identity: str, *, cores: int = 1) -> AssemblyEvidence:
    """Construct one exact contig view after evidence has been identity-checked."""
    with task_team(cores):
        return AssemblyEvidence.from_contig(evidence, identity)


def complete_stage(
    inputs: AssemblyInputs,
    parameters: CompletionParameters,
    records: RecordCapability | None,
    *,
    ties: TieSettings | None = None,
    cores: int = 1,
    map_tasks: Callable | None = None,
) -> CompletionProduct:
    """Resume completed preparation, or execute AS1 and publish once."""
    ties = TieSettings() if ties is None else ties
    stored = None if records is None else records.load(COMPLETION)
    if stored is not None:
        return decode_completion(stored)
    with task_team(cores):
        prepared, summary = complete(
            inputs.panels, inputs.evidence, parameters, CompletionWorkspace(ties=ties, map_tasks=map_tasks)
        )
    product = CompletionProduct(tuple(prepared), to_payload(summary))
    if records is not None:
        records.save(COMPLETION, encode_completion(product))
    return product


# Blocks per completion task. A block of 200 sites at 116 samples is ~50 ms of one thread's work, so 16 blocks keep a
# task under a second against tens of milliseconds of transport; a contig's few hundred blocks make tens of tasks.
COMPLETION_BLOCKS_PER_TASK = 16


def complete_range(
    panels: Sequence[LocalPanel],
    inference: Sequence[np.ndarray],
    neutral: np.ndarray | SharedArrayRef,
    observed: np.ndarray | SharedArrayRef,
    parameters: CompletionParameters,
    folds: np.ndarray,
    settings: np.ndarray,
) -> tuple[CompletionBlockResult, ...]:
    """Complete a consecutive run of blocks (one numeric thread); the contig's evidence
    arrives as shared segments, attached without a copy."""
    with ExitStack() as stack, thread_team(1):
        neutral = stack.enter_context(attached_array(neutral))
        observed = stack.enter_context(attached_array(observed))
        blocks = completion_blocks(panels, inference, neutral, observed, parameters, folds, settings)
        return tuple(complete_block(block) for block in blocks)


def finish_range_completion(
    inputs: AssemblyInputs,
    parameters: CompletionParameters,
    records: RecordCapability | None,
    ties: TieSettings,
    plan: CompletionPlan,
    parts: Sequence[tuple[CompletionBlockResult, ...]],
    then: Callable[..., Any],
    *,
    cores: int = 1,
) -> Any:
    """The completion's gather: link, gate and release from the ranges' results in block order, publish once, then
    continue the assembly (``then(product, cores=cores)``)."""
    results = tuple(result for part in parts for result in part)
    with task_team(cores):
        prepared, summary = finish_completion(
            inputs.panels, inputs.evidence, parameters, CompletionWorkspace(ties=ties), plan, results
        )
    product = CompletionProduct(tuple(prepared), to_payload(summary))
    if records is not None:
        records.save(COMPLETION, encode_completion(product))
    return then(product, cores=cores)


def expand_completion(
    inputs: AssemblyInputs,
    parameters: CompletionParameters,
    records: RecordCapability | None,
    then: Callable[..., Any],
    *,
    ties: TieSettings,
    parent: TaskKey,
    max_cores: int,
    storage: SharedEvidence | None = None,
    cores: int = 1,
) -> Any:
    """Resume a published completion, or run its blocks as independent engine tasks of up to
    COMPLETION_BLOCKS_PER_TASK consecutive blocks and continue in their gather. Results equal complete()'s: the same
    blocks, plan and order (WP_AS1 §6.2)."""
    stored = None if records is None else records.load(COMPLETION)
    if stored is not None:
        return then(decode_completion(stored), cores=cores)
    panels = tuple(inputs.panels)
    if len(panels) <= COMPLETION_BLOCKS_PER_TASK or inputs.evidence.neutral is None:
        return then(complete_stage(inputs, parameters, records, ties=ties, cores=cores), cores=cores)
    with task_team(cores):
        plan = plan_completion(panels, inputs.evidence, parameters, CompletionWorkspace(ties=ties))
    neutral = inputs.evidence.neutral if storage is None else storage.references["neutral"]
    observed = inputs.evidence.observed if storage is None else storage.references["observed"]
    samples = inputs.evidence.samples
    tasks = []
    for first in range(0, len(panels), COMPLETION_BLOCKS_PER_TASK):
        stop = min(len(panels), first + COMPLETION_BLOCKS_PER_TASK)
        working = max(block_memory_bytes(samples, p.founders, p.sites, parameters) for p in panels[first:stop])
        tasks.append(
            Task(
                replace(parent, unit=f"{parent.unit}.completion{first // COMPLETION_BLOCKS_PER_TASK}"),
                complete_range,
                args=(
                    panels[first:stop],
                    plan.inference[first:stop],
                    neutral,
                    observed,
                    parameters,
                    plan.fold_plan.folds,
                    plan.settings,
                ),
                resources=ResourceRequest(1, 1, working),
            )
        )
    gather = Task(
        replace(parent, unit=f"{parent.unit}.completion.gather"),
        finish_range_completion,
        args=(inputs, parameters, records, ties, plan, tuple(ResultRef(t.key) for t in tasks), then),
        depends_on=tuple(t.key for t in tasks),
        affinity=parent.scope,
        resources=ResourceRequest(1, max_cores),
        thread_argument="cores",
    )
    return Expansion(tuple(tasks), gather)


@contextmanager
def attached_array(array: np.ndarray | SharedArrayRef) -> Iterator[np.ndarray]:
    """Borrow a shared segment's read-only view in a child worker (an array passes through)."""
    if isinstance(array, np.ndarray):
        yield array
        return
    handle, view = attach_shared_array(array)
    view.flags.writeable = False
    try:
        yield view
    finally:
        close_shared_memory((handle,))


@dataclass
class SharedEvidence:
    """Stage-owned lifetime of engine shared arrays and the plain science view.

    Child batches receive only scoring_ref and attach without copying. The owner
    keeps the segments alive until their last consumer and can release neutral
    independently before final refinement. No handle enters AssemblyEvidence.
    """

    view: AssemblyEvidence
    references: dict[str, SharedArrayRef]
    handles: dict[str, SharedMemory]

    @property
    def scoring_ref(self) -> SharedArrayRef:
        return self.references["scoring"]

    def release_neutral(self) -> None:
        self.view.release_neutral()
        handle = self.handles.pop("neutral", None)
        self.references.pop("neutral", None)
        if handle is not None:
            close_shared_memory((handle,), unlink=True)

    def close(self) -> None:
        """Close only after all science consumers have completed."""
        close_shared_memory(tuple(self.handles.values()), unlink=True)
        self.handles.clear()
        self.references.clear()

    def __enter__(self) -> SharedEvidence:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@contextmanager
def attached_scoring(scoring: np.ndarray | SharedArrayRef) -> Iterator[np.ndarray]:
    """Borrow a child worker's scoring view without a transport copy."""
    if isinstance(scoring, np.ndarray):
        yield scoring
        return
    handle, array = attach_shared_array(scoring)
    array.flags.writeable = False
    try:
        yield array
    finally:
        close_shared_memory((handle,))
