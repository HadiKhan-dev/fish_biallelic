"""Compose PE4/PE5 with PE2 exclusion and PE3 relative tasks, then publish once.

Owns PD27's engine/storage boundary. Science run.py owns the ordered transitions;
this adapter never renders tables or repairs a partial cached product.
Method: docs/methods/pedigree.md#products-and-published-tables (WP_PE5 §4.10).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.payload import from_payload
from ...contracts.pedigree import (
    CalibrationRecord,
    ChromosomeScores,
    ChromosomeSummaryView,
    Decision,
    ExclusionEvidence,
    ExclusionRequest,
    GenomeScores,
    GuardRelatedness,
    Pedigree,
    PedigreeReportInputs,
    PreparedChromosome,
    ReleasedDecision,
    SearchDiagnosticView,
)
from ...core.errors import InputError, InvariantError
from ...engine.workers import ResidentRef, release_resident, release_shared_value, resident_value, retain
from ...science.pedigree import panel, resampling, stacking
from ...science.pedigree import run as science
from ...science.pedigree.parameters import ExclusionConfig, PedigreeConfig
from ...science.pedigree.resampling import FitResult
from ...store.container import CheckpointFile
from ...store.protocol import NumericalPolicy, claim
from ...store.store import CheckpointStore
from ..evidence import records as evidence_records
from ..framework import (
    Expansion,
    RecordKey,
    ResourceRequest,
    ResultRef,
    Task,
    TaskKey,
    load_product,
    publish_product,
)
from ..relatedness import families_records
from . import exclusion, execution, prepare, relatives, scores
from .codec import PedigreeAdapter
from .records import (
    FLOORED_SCORES,
    PREPARED,
    PRODUCT,
    REQUEST,
    PedigreeDependencies,
    RequestAdapter,
    bound_inputs,
    derivation,
    request_dependencies,
    result_reader,
)

__all__ = [
    "DecisionStageInputs",
    "StoredPrepared",
    "StoredEvidence",
    "task",
    "tasks",
    "task_stored",
    "run",
    "run_stored",
    "resume",
    "product_receipt",
    "precheck",
]
precheck = science.precheck
_AFFINITY = "pedigree/decision"


def product_receipt(identity: str) -> dict[RecordKey, str]:
    """The stored route's reply: the published product's record key and identity.

    The product itself (with its retained decision, 5.8 GB pickled at N = 2,503)
    stays in the store; consumers read the published record, never a reply copy.
    """
    return {RecordKey("run", PRODUCT): identity}


@dataclass(frozen=True)
class StoredPrepared:
    """Small descriptor loaded only by its contig report task."""

    file: CheckpointFile
    identity: str
    summary: ChromosomeSummaryView

    @property
    def contig(self) -> str:
        return self.summary.contig


@dataclass(frozen=True)
class StoredEvidence:
    """Checked evidence descriptor; no genotype buffer crosses the decision task."""

    file: CheckpointFile
    identity: str


@dataclass(frozen=True)
class DecisionStageInputs:
    """Already prepared/scored PE1–PE3 inputs in their declared axes.

    Evidence is consumed only by the independent PE2 exclusion tasks. Observed
    counts are streamed from those masks by the caller, one configured contig at
    a time. No raw genotype or painting computation occurs in this stage.
    """

    genome: GenomeScores
    contigs: tuple[str, ...]
    prepared: tuple[PreparedChromosome | StoredPrepared, ...]
    relatedness: GuardRelatedness
    observed_counts: np.ndarray
    evidence: Mapping[str, ContigEvidence | StoredEvidence]
    identities: PedigreeDependencies
    omitted: dict[str, str]


def _dependencies(
    inputs: DecisionStageInputs, request: ExclusionRequest, parameters: PedigreeConfig, policy: NumericalPolicy | None
) -> PedigreeDependencies:
    if request.contigs != inputs.contigs:
        raise InputError("exclusion request differs from configured contigs")
    return request_dependencies(inputs.identities, request, parameters, policy)


def _identity(
    inputs: DecisionStageInputs, request: ExclusionRequest, parameters: PedigreeConfig, policy: NumericalPolicy | None
) -> str:
    return derivation(
        parameters,
        inputs.genome.sample_ids,
        inputs.contigs,
        inputs.genome.contigs,
        _dependencies(inputs, request, parameters, policy),
        numerical_policy=policy,
    )


def _request_identity(
    inputs: DecisionStageInputs,
    parameters: PedigreeConfig,
    policy: NumericalPolicy | None,
) -> str:
    return derivation(
        parameters,
        inputs.genome.sample_ids,
        inputs.contigs,
        inputs.genome.contigs,
        replace(inputs.identities, exclusion=()),
        numerical_policy=policy,
    )


def resume(
    file: CheckpointFile,
    inputs: DecisionStageInputs,
    parameters: PedigreeConfig,
    *,
    numerical_policy: NumericalPolicy | None = None,
) -> Pedigree | None:
    """Validate a whole cached product without refitting or recomputing report values.

    The cached ordered dyad request supplies the derived exclusion dependency;
    every cause of that request (code, parameters, scores, families and axes) is
    independently bound in the expected derivation. A changed cause refuses reuse.
    """
    found = _resumed(file, inputs, parameters, numerical_policy)
    return None if found is None else found[1]


def _resumed(
    file: CheckpointFile,
    inputs: DecisionStageInputs,
    parameters: PedigreeConfig,
    numerical_policy: NumericalPolicy | None,
) -> tuple[str, Pedigree] | None:
    if not file.has(PRODUCT):
        return None
    request_id = _request_identity(inputs, parameters, numerical_policy)
    request = load_product(file, REQUEST, request_id, RequestAdapter())
    if request is None:
        raise InvariantError("pedigree publication lacks its independently bound exclusion request")
    identity = _identity(inputs, request, parameters, numerical_policy)
    cached = load_product(file, PRODUCT, identity, PedigreeAdapter())
    return None if cached is None else (identity, cached)


def run(
    inputs: DecisionStageInputs,
    parameters: PedigreeConfig,
    *,
    workers: int = 1,
    file: CheckpointFile | None = None,
    contig_files: Mapping[str, CheckpointFile] | None = None,
    numerical_policy: NumericalPolicy | None = None,
    receipt: bool = False,
) -> Pedigree | dict[RecordKey, str] | Expansion:
    """Resume or fit once, then expose independent per-contig exclusion work.

    With ``receipt`` (the stored route) the final reply is the published record's
    key and identity rather than the product; it requires the run ``file``.
    """
    precheck(len(inputs.genome.sample_ids), len(inputs.contigs), parameters, basis="configured")
    if tuple(p.contig for p in inputs.prepared) != inputs.contigs:
        raise InputError("prepared inputs do not follow configured contig order")
    if receipt and file is None:
        raise InvariantError("a pedigree receipt names a published record; the run file is required")
    if file is not None:
        found = _resumed(file, inputs, parameters, numerical_policy)
        if found is not None:
            return _reply(found[0], found[1], receipt)
    started = science.start_full_fit(inputs.genome, parameters)
    options: dict[str, Any] = dict(
        file=file, contig_files=contig_files, numerical_policy=numerical_policy, receipt=receipt
    )
    return execution.calibrate(
        started.calibration,
        workers,
        stage="pedigree",
        affinity=_AFFINITY,
        then=_calibrated,
        args=(inputs, parameters, started),
        kwargs=options,
    )


def _calibrated(
    fitted: tuple[float, CalibrationRecord],
    inputs: DecisionStageInputs,
    parameters: PedigreeConfig,
    started: science.decision.FitStart,
    *,
    workers: int,
    **options: Any,
) -> Pedigree | dict[RecordKey, str] | Expansion:
    """Finish the full fit with its evidence scale, then resample (serially, or as fit subtasks)."""
    state = science.finish_full_fit(started, fitted, parameters)
    planned = resampling.plan(state.inputs, parameters, workers)
    if planned.worker_count == 1:
        results = (resampling.evaluate_task(state.inputs, task) for task in planned.tasks)
        decided = science.complete_decision(state, resampling.combine(state.inputs, planned, results), parameters)
        return _requested(inputs, parameters, decided, workers=workers, **options)
    shared, subtasks = execution.fit_subtasks(state.inputs, planned, "pedigree", copy_threads=workers)
    retained = retain((inputs, parameters, state, planned, shared), "fit")
    gather = Task(
        TaskKey("pedigree", "resampled"),
        _resampled_retained,
        args=(retained, tuple(ResultRef(t.key) for t in subtasks)),
        kwargs=options,
        depends_on=tuple(t.key for t in subtasks),
        affinity=_AFFINITY,
        resources=ResourceRequest(max_cores=workers),
        thread_argument="workers",
    )
    return Expansion(subtasks, gather)


def _resampled_retained(
    reference: ResidentRef,
    fitted: tuple[FitResult, ...],
    *,
    workers: int = 1,
    file: CheckpointFile | None = None,
    contig_files: Mapping[str, CheckpointFile] | None = None,
    numerical_policy: NumericalPolicy | None = None,
    receipt: bool = False,
) -> Expansion:
    """Reduce the fit subtasks' results in the plan's order on the affinity worker, then request exclusion."""
    inputs, parameters, state, planned, shared = resident_value(reference)
    try:
        release_shared_value(shared)
        decided = science.complete_decision(state, resampling.combine(state.inputs, planned, fitted), parameters)
    finally:
        release_resident(reference)
    return _requested(
        inputs,
        parameters,
        decided,
        workers=workers,
        file=file,
        contig_files=contig_files,
        numerical_policy=numerical_policy,
        receipt=receipt,
    )


def _requested(
    inputs: DecisionStageInputs,
    parameters: PedigreeConfig,
    decided: Decision,
    *,
    workers: int = 1,
    file: CheckpointFile | None = None,
    contig_files: Mapping[str, CheckpointFile] | None = None,
    numerical_policy: NumericalPolicy | None = None,
    receipt: bool = False,
) -> Expansion:
    """Freeze the exclusion request, then expose independent per-contig exclusion work."""
    request = science.exclusion_request(decided, inputs.contigs, inputs.relatedness, parameters)
    identities = _dependencies(inputs, request, parameters, numerical_policy)
    if file is not None:
        publish_product(
            file, REQUEST, _request_identity(inputs, parameters, numerical_policy), request, RequestAdapter()
        )
    parts = (
        tuple(
            _exclusion_task(
                c,
                inputs.evidence[c],
                request.dyads,
                parameters.exclusion,
                len(inputs.contigs),
                len(inputs.genome.sample_ids),
                cores=workers,
                file=None if contig_files is None else contig_files[c],
                identity=None if contig_files is None else dict(identities.exclusion)[c],
            )
            for c in inputs.contigs
        )
        if len(request.dyads)
        else ()
    )
    retained = retain((inputs, parameters, decided, request), "decision")
    gather = Task(
        TaskKey("pedigree", "release"),
        _release_retained,
        args=(retained, tuple(ResultRef(t.key) for t in parts)),
        kwargs=dict(file=file, numerical_policy=numerical_policy, receipt=receipt),
        depends_on=tuple(t.key for t in parts),
        affinity=_AFFINITY,
        resources=ResourceRequest(max_cores=workers),
        thread_argument="workers",
    )
    return Expansion(parts, gather)


def _release_retained(
    reference: ResidentRef,
    evidence: tuple[ExclusionEvidence, ...],
    *,
    workers: int = 1,
    file: CheckpointFile | None = None,
    numerical_policy: NumericalPolicy | None = None,
    receipt: bool = False,
) -> Expansion:
    """Borrow the original large input/decision bundle on its affinity worker."""
    inputs, parameters, decided, request = resident_value(reference)
    try:
        return _release(
            inputs,
            parameters,
            decided,
            request,
            evidence,
            workers=workers,
            file=file,
            numerical_policy=numerical_policy,
            receipt=receipt,
        )
    finally:
        release_resident(reference)


def _release(
    inputs: DecisionStageInputs,
    parameters: PedigreeConfig,
    decided: Decision,
    request: ExclusionRequest,
    evidence: tuple[ExclusionEvidence, ...],
    *,
    workers: int = 1,
    file: CheckpointFile | None = None,
    numerical_policy: NumericalPolicy | None = None,
    receipt: bool = False,
) -> Expansion:
    released = science.release_decision(
        decided, request, evidence, inputs.relatedness, inputs.observed_counts, parameters
    )
    search = panel.search_diagnostics(
        decided.scores.panel,
        decided.eligibility,
        released.pre_guard_calls.parents,
        reverse=parameters.reverse_conventions,
    )
    if not isinstance(search, tuple) or not all(isinstance(row, SearchDiagnosticView) for row in search):
        raise InvariantError("PE2 search_diagnostics must return typed SearchDiagnosticView rows")
    wanted = science.relative_request(released)
    parts = (
        tuple(
            _relative_task(p, parameters, wanted.pairs, cores=workers)
            for p in inputs.prepared
            if _summary(p).scored_component_count
        )
        if len(wanted.pairs)
        else ()
    )
    retained = retain((inputs, parameters, released, search), "release")
    gather = Task(
        TaskKey("pedigree", "publish"),
        _publish_retained,
        args=(retained, tuple(ResultRef(t.key) for t in parts)),
        kwargs=dict(file=file, numerical_policy=numerical_policy, receipt=receipt),
        depends_on=tuple(t.key for t in parts),
        affinity=_AFFINITY,
    )
    return Expansion(parts, gather)


def _publish_retained(
    reference: ResidentRef,
    contrasts: tuple[relatives.ContigContrasts, ...],
    *,
    file: CheckpointFile | None = None,
    numerical_policy: NumericalPolicy | None = None,
    receipt: bool = False,
) -> Pedigree | dict[RecordKey, str]:
    """Release the affinity's retained decision state after its last consumer."""
    inputs, parameters, released, search = resident_value(reference)
    try:
        return _publish(
            inputs,
            parameters,
            released,
            search,
            contrasts,
            file=file,
            numerical_policy=numerical_policy,
            receipt=receipt,
        )
    finally:
        release_resident(reference)


def _publish(
    inputs: DecisionStageInputs,
    parameters: PedigreeConfig,
    released: ReleasedDecision,
    search: tuple[SearchDiagnosticView, ...],
    contrasts: tuple[relatives.ContigContrasts, ...],
    *,
    file: CheckpointFile | None = None,
    numerical_policy: NumericalPolicy | None = None,
    receipt: bool = False,
) -> Pedigree | dict[RecordKey, str]:
    wanted = science.relative_request(released)
    reports = PedigreeReportInputs(
        search, tuple(_summary(p) for p in inputs.prepared), relatives.aggregate(wanted.pairs, wanted.views, contrasts)
    )
    product = science.assemble(released, inputs.omitted, parameters, reports).pedigree
    adapter = PedigreeAdapter()
    # Freeze/validate through the identical publication boundary even without a store.
    product = adapter.decode(adapter.encode(product))
    if file is not None:
        identity = _identity(inputs, released.request, parameters, numerical_policy)
        with claim(file, PRODUCT, identity) as needed:
            if needed:
                publish_product(file, PRODUCT, identity, product, adapter)
            else:
                cached = load_product(file, PRODUCT, identity, adapter)
                if cached is None:
                    raise InvariantError("claimed complete pedigree disappeared")
                return _reply(identity, cached, receipt)
        return _reply(identity, product, receipt)
    return product


def _reply(identity: str, product: Pedigree, receipt: bool) -> Pedigree | dict[RecordKey, str]:
    return product_receipt(identity) if receipt else product


def task(
    inputs: DecisionStageInputs,
    parameters: PedigreeConfig,
    *,
    cores: int = 1,
    file: CheckpointFile | None = None,
    contig_files: Mapping[str, CheckpointFile] | None = None,
    numerical_policy: NumericalPolicy | None = None,
    depends_on: tuple[TaskKey, ...] = (),
) -> Task:
    """Declare the scored-input stage cut; expansion preserves the pure transition order."""
    return Task(
        TaskKey("pedigree", "decision"),
        run,
        args=(inputs, parameters),
        kwargs=dict(file=file, contig_files=contig_files, numerical_policy=numerical_policy),
        depends_on=depends_on,
        affinity=_AFFINITY,
        outputs=(RecordKey("run", PRODUCT),),
        resources=ResourceRequest(max_cores=cores),
        thread_argument="workers",
    )


def _summary(value: PreparedChromosome | StoredPrepared) -> ChromosomeSummaryView:
    return value.summary if isinstance(value, StoredPrepared) else scores.chromosome_summary((value,))[0]


def _exclusion_stored(
    contig: str,
    reference: StoredEvidence,
    dyads: np.ndarray,
    parameters: ExclusionConfig,
    configured_contigs: int,
    sample_count: int,
    *,
    threads: int = 1,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> ExclusionEvidence:
    evidence = evidence_records.load_dense(reference.file, expected_identity=reference.identity)
    return exclusion.run(
        contig,
        evidence,
        dyads,
        parameters,
        configured_contigs,
        sample_count,
        threads=threads,
        file=file,
        identity=identity,
    )


def _exclusion_task(
    contig: str,
    evidence: ContigEvidence | StoredEvidence,
    dyads: np.ndarray,
    parameters: ExclusionConfig,
    configured_contigs: int,
    sample_count: int,
    *,
    cores: int,
    file: CheckpointFile | None,
    identity: str | None,
) -> Task:
    if not isinstance(evidence, StoredEvidence):
        return exclusion.task(
            contig,
            evidence,
            dyads,
            parameters,
            configured_contigs,
            sample_count,
            cores=cores,
            file=file,
            identity=identity,
        )
    return Task(
        TaskKey("pedigree", "exclusion", contig),
        _exclusion_stored,
        args=(contig, evidence, dyads, parameters, configured_contigs, sample_count),
        kwargs=dict(file=file, identity=identity),
        inputs=(RecordKey(contig, "evidence"),),
        outputs=(RecordKey(contig, exclusion.KEY),),
        resources=ResourceRequest(max_cores=cores),
        affinity=contig,
        thread_argument="threads",
    )


def _relative_stored(
    reference: StoredPrepared,
    parameters: PedigreeConfig,
    pairs: np.ndarray,
    *,
    threads: int = 1,
) -> relatives.ContigContrasts:
    prepared = load_product(reference.file, PREPARED, reference.identity, prepare.PreparedAdapter())
    if prepared is None:
        raise InvariantError("relative report requires complete prepared inputs")
    return relatives.run(prepared, parameters, pairs, threads=threads, reverse=parameters.reverse_conventions)


def _relative_task(
    prepared: PreparedChromosome | StoredPrepared,
    parameters: PedigreeConfig,
    pairs: np.ndarray,
    *,
    cores: int,
) -> Task:
    if not isinstance(prepared, StoredPrepared):
        return relatives.task(prepared, parameters, pairs, cores=cores, reverse=parameters.reverse_conventions)
    return Task(
        TaskKey("pedigree", "relatives", prepared.contig),
        _relative_stored,
        args=(prepared, parameters, pairs),
        inputs=(RecordKey(prepared.contig, PREPARED),),
        resources=ResourceRequest(max_cores=cores),
        affinity=prepared.contig,
        thread_argument="threads",
    )


def run_stored(
    root: str, manifest: Mapping[str, Any] | None = None, *, workers: int = 1
) -> dict[RecordKey, str] | Expansion:
    """Read one chromosome at a time; retain only scores and immutable descriptors.

    Fit work shares the one assembled genome bundle. Exclusion and relatives load
    their own per-contig records on their affinity workers. Resume validates the
    whole published product before loading scoring arrays or observed masks. The
    reply is the published product's receipt (``product_receipt``), never the product.
    """
    store = CheckpointStore(root, nthreads=workers)
    store.check_layout()
    manifest = store.manifest("run") if manifest is None else manifest
    if manifest is None:
        raise InputError("pedigree requires the bound run manifest")
    configured_parameters = from_payload(PedigreeConfig, manifest["parameters"]["pedigree"])
    precheck(len(manifest["sample_ids"]), len(manifest["contigs"]), configured_parameters, basis="configured")
    reader = result_reader(store, manifest, product="pedigree")
    if reader is not None:
        cached = load_product(store.run, PRODUCT, reader.identity, PedigreeAdapter())
        if cached is not None:
            return product_receipt(reader.identity)
    bound = bound_inputs(store, manifest)
    parameters = bound.parameters
    precheck(len(bound.sample_ids), len(bound.contigs), parameters, basis="configured")
    precheck(len(bound.sample_ids), len(bound.informative_contigs), parameters, basis="informative")

    def chromosome_scores() -> Iterator[ChromosomeScores]:
        for c, identity in bound.dependencies.floored_scores:
            value = load_product(store.contig(c), FLOORED_SCORES, identity, scores.ScoresAdapter())
            if value is None:
                raise InvariantError("pedigree requires complete floored chromosome scores")
            yield value
            del value

    genome = stacking.stack(bound.sample_ids, bound.panel, chromosome_scores(), bound.eligibility)
    family = families_records.load(store.run, bound.dependencies.relatedness_families)
    if family is None or family.sample_ids != bound.sample_ids:
        raise InputError("relatedness/families sample axis differs from the pedigree")
    relatedness = GuardRelatedness(bound.sample_ids, family.family, family.member_support)
    observed_counts = np.empty((len(bound.contigs), len(bound.sample_ids)), dtype=np.int64)
    for k, (c, identity) in enumerate(bound.dependencies.evidence):
        counts = evidence_records.load_observed_counts(store.contig(c), expected_identity=identity)
        if counts.shape != (len(bound.sample_ids),):
            raise InputError("observed-count sample axis differs from the pedigree")
        observed_counts[k] = counts
    prepared = tuple(
        StoredPrepared(store.contig(c), identity, summary)
        for (c, identity), summary in zip(bound.dependencies.prepared, bound.summary, strict=True)
    )
    evidence = {c: StoredEvidence(store.contig(c), identity) for c, identity in bound.dependencies.evidence}
    inputs = DecisionStageInputs(
        genome,
        bound.contigs,
        prepared,
        relatedness,
        observed_counts,
        evidence,
        bound.dependencies,
        {row.contig: row.omitted_reason for row in bound.summary if row.omitted_reason is not None},
    )
    reply = run(
        inputs,
        parameters,
        workers=workers,
        file=store.run,
        contig_files={c: store.contig(c) for c in bound.contigs},
        numerical_policy=bound.policy,
        receipt=True,
    )
    # The receipt route replies with a receipt or an expansion, never the product.
    assert not isinstance(reply, Pedigree)
    return reply


def _stored_result(published: dict[RecordKey, str]) -> dict[RecordKey, str]:
    """Return the completed affinity branch's receipt to an outer barrier without retaining it."""
    return published


def _stored_dispatch(root: str, manifest: Mapping[str, Any] | None, cores: int) -> Expansion:
    """Start a resident decision branch from an arbitrary upstream barrier.

    An expansion gather inherits its parent's affinity. The score barrier has no
    resident state, so the large decision must be a child task with its own
    affinity, followed by a small result gather on the outer barrier.
    """
    fit = Task(
        TaskKey("pedigree", "decision.fit"),
        run_stored,
        args=(root, manifest),
        affinity=_AFFINITY,
        resources=ResourceRequest(max_cores=cores),
        thread_argument="workers",
    )
    gather = Task(
        TaskKey("pedigree", "decision.result"),
        _stored_result,
        args=(ResultRef(fit.key),),
        depends_on=(fit.key,),
    )
    return Expansion((fit,), gather)


def task_stored(
    root: str,
    manifest: Mapping[str, Any] | None = None,
    *,
    cores: int = 1,
    depends_on: tuple[TaskKey, ...] = (),
) -> Task:
    """Join any PE1–PE3 score barrier; construction performs no I/O.

    The dispatch itself retains no arrays and inherits any outer affinity. Its
    decision child establishes the affinity required for the retained fit and
    release bundles. This same route serves standalone stored-input replay.
    """
    return Task(
        TaskKey("pedigree", "decision"),
        _stored_dispatch,
        args=(root, manifest, cores),
        depends_on=depends_on,
        outputs=(RecordKey("run", PRODUCT),),
    )


def tasks(
    root: str,
    *,
    maximum_cores: int = 1,
    depends_on: tuple[TaskKey, ...] = (),
) -> tuple[Task, ...]:
    """Expose the stored-input action to integration and the kernel declaration audit."""
    return (task_stored(root, cores=maximum_cores, depends_on=depends_on),)
