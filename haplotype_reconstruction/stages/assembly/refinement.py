"""Build explicit refinement capabilities and run one AS8 component task.

The caller supplies immutable component data, a declared team and record scope.
The stage chooses evidence/message storage and measures elapsed component time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.contracts.assembly import ComponentRefinement, JoinedLevel, RefinedLevel
from haplotype_reconstruction.contracts.panels import FounderComponent, PreparedPanel
from haplotype_reconstruction.engine.graph import Expansion, ResultRef
from haplotype_reconstruction.science.assembly.parameters import AssemblyParameters
from haplotype_reconstruction.science.assembly.refinement.capabilities import RefinementExecution, RefinementRecords

if TYPE_CHECKING:
    from .final import AssemblySession, HierarchyState
    from .records import RecordCapability
from time import perf_counter

from haplotype_reconstruction.engine.workers import SharedArrayRef
from haplotype_reconstruction.science.assembly.refinement.component_workspace import ComponentWorkspace
from haplotype_reconstruction.science.assembly.refinement.run import RefinementSettings
from haplotype_reconstruction.science.assembly.refinement.run import refine_component as science_refine_component
from haplotype_reconstruction.science.assembly.refinement.window import RESIDENT_BYTES, resident_bytes, window_sites

from .completion import attached_scoring
from .storage import StoragePlan, Windows


def refine_component(
    number: int,
    component: FounderComponent,
    panels: Sequence[PreparedPanel],
    evidence: NDArray[np.float32],
    context: object,
    settings: RefinementSettings,
    team: RefinementExecution,
    records: RefinementRecords | None = None,
    *,
    incoming: tuple = (),
    storage: StoragePlan | None = None,
) -> tuple[FounderComponent, ComponentRefinement]:
    """Return (FounderComponent, ComponentRefinement) on the original block span.

    records is an optional load/save/scoped capability; the stage owner chooses
    whether to record small fits. Explicit storage overrides are replay controls.
    """
    storage = StoragePlan() if storage is None else storage
    started = perf_counter()

    def windows_factory(
        evidence: NDArray[np.float32], kept: NDArray[np.bool_], partial: NDArray[np.bool_], hazards: NDArray[np.float64]
    ) -> Windows:
        samples, sites = evidence.shape[:2]
        resident = (
            resident_bytes(samples, sites, int(partial.sum())) <= RESIDENT_BYTES
            if storage.resident is None
            else storage.resident
        )
        capacity = window_sites(samples) if storage.capacity is None else storage.capacity
        return Windows(evidence, kept, partial, hazards, resident=resident, capacity=capacity)

    if component.stop - component.start < 2 or component.founders < 2:
        return science_refine_component(
            number, component, panels, evidence, context, settings, team, records, incoming=incoming, workspace=None
        )
    with team.scope():
        workspace = ComponentWorkspace(
            panels,
            evidence,
            min_sites_per_bin=settings.min_sites_per_bin,
            max_bins=settings.config.proposal_max_bins,
            genetic_map=settings.genetic_map,
            genotype_error=settings.config.genotype_error,
            reset_rate=settings.config.reset_rate,
            map_probability_floor=settings.config.map_probability_floor,
            windows_factory=windows_factory,
            message_step=storage.message_step,
        )
        component, result = science_refine_component(
            number,
            component,
            panels,
            evidence,
            context,
            settings,
            team,
            records,
            incoming=incoming,
            workspace=workspace,
        )
    from .refinement_diagnostics import report_refinement

    report_refinement(result)
    return component, replace(result, elapsed_seconds=perf_counter() - started)


def refine(
    *, session: AssemblySession, state: HierarchyState, joined: JoinedLevel, cores: int = 1
) -> RefinedLevel | Expansion:
    """AS6 callback: resume components or expand independent AS8 component tasks.

    Results and continuation families return in genomic component order. Each
    component's identity binds its exact ordered coarse context and final flag.
    The native fit-record threshold is an I/O policy; direct replay may force it.
    """
    from haplotype_reconstruction.engine.graph import ResourceRequest, Task
    from haplotype_reconstruction.science.assembly.refinement.costs import component_task_bytes
    from haplotype_reconstruction.science.assembly.refinement.run import bin_floor, macro_context
    from haplotype_reconstruction.stages.framework import Expansion

    from .records import RecordCapability, component_derivation, refined_key

    parameters = cast(AssemblyParameters, session.parameters)
    config = parameters.refinement
    settings = RefinementSettings(
        config,
        bin_floor(state.level, state.prepared, config),
        float(parameters.cost_scale),
        joined.record.stop_reason is not None,
        parameters.count_rule,
        state.level,
        session.inputs.genetic_map,
        session.ties.reverse,
        session.ties.census,
    )
    records = None if session.records is None else session.records.scoped(refined_key(state.level))
    tasks = []
    results: list[Any] = []
    started = perf_counter()
    for number, component in enumerate(joined.components):
        name = f"{number:05d}"
        panels = tuple(state.prepared[component.start : component.stop])
        context = macro_context(component, state.l1_context)
        incoming = tuple(
            dict(f)
            for f in joined.release_alternatives
            if int(f["start"]) < component.stop and int(f["stop"]) > component.start
        )
        bound = None
        if records is not None:
            identity = component_derivation(
                records.identity,
                component,
                level=state.level,
                context=context,
                parameters=settings,
                additions=settings.additions,
            )
            bound = RecordCapability(records.file, records.prefix, identity)
            saved = bound.load(name)
            if saved is not None:
                results.append(_decode_component(saved))
                continue
        key = replace(session.parent, unit=f"{session.parent.unit}.refine{state.level}.{name}")
        working = component_task_bytes(
            panels,
            session.inputs.evidence.samples,
            component.founders,
            config,
            min_sites_per_bin=settings.min_sites_per_bin,
            context=context,
            cores=session.max_cores,
            incoming=incoming,
        )
        task = Task(
            key,
            _component_stage,
            args=(
                number,
                component,
                panels,
                session.inputs.evidence.scoring if session.storage is None else session.storage.scoring_ref,
                context,
                settings,
                bound,
                incoming,
            ),
            resources=ResourceRequest(1, session.max_cores, working_set_bytes=working),
            # Components are independent (each records its own scope): not pinned, so they run side by side across
            # the contig's lane rather than one at a time on the parent's worker.
            thread_argument="cores",
        )
        tasks.append(task)
        results.append(ResultRef(key))
    if not tasks:
        return _gather_level(settings, results, records, started)
    gather = Task(
        replace(session.parent, unit=f"{session.parent.unit}.refine{state.level}.components"),
        _gather_level,
        args=(settings, tuple(results), records, started),
        depends_on=tuple(task.key for task in tasks),
        affinity=session.parent.scope,
    )
    return Expansion(tuple(tasks), gather)


def _decode_component(payload: Mapping[str, Any]) -> tuple[FounderComponent, ComponentRefinement]:
    """Restore the typed component and refinement record from one validated scope."""
    from .records import decode_components, decode_refinement

    (component,) = decode_components(payload["component"])
    return component, decode_refinement(payload["refinement"])


def _component_stage(
    number: int,
    component: FounderComponent,
    panels: Sequence[PreparedPanel],
    evidence: NDArray[np.float32] | SharedArrayRef,
    context: object,
    settings: RefinementSettings,
    records: RecordCapability | None,
    incoming: tuple,
    *,
    cores: int = 1,
    storage: StoragePlan | None = None,
    force_fit_records: bool = False,
) -> tuple[FounderComponent, ComponentRefinement]:
    """Run and atomically record one independently resumable component."""
    from haplotype_reconstruction.science.assembly.refinement.component_workspace import component_evidence

    from .records import encode_components, encode_refinement
    from .team import RefinementTeam

    name = f"{number:05d}"
    fit_records = (
        None
        if records is None or (not force_fit_records and component.stop - component.start <= 100)
        else records.scoped(name + "/fit", nthreads=1)
    )
    with attached_scoring(evidence) as scoring:
        result = refine_component(
            number,
            component,
            panels,
            component_evidence(scoring, panels),
            context,
            settings,
            RefinementTeam(cores),
            fit_records,
            incoming=incoming,
            storage=storage,
        )
    if records is not None:
        records.save(name, dict(component=encode_components([result[0]]), refinement=encode_refinement(result[1])))
    return result


def _gather_level(
    settings: RefinementSettings,
    results: Sequence[tuple[FounderComponent, ComponentRefinement]],
    records: RecordCapability | None,
    started: float,
) -> RefinedLevel:
    """Publish the level census only after every component is complete."""
    from dataclasses import asdict

    from haplotype_reconstruction.contracts.assembly import RefinedLevel, ReleaseFamily
    from haplotype_reconstruction.science.assembly.refinement.run import summarise

    components = tuple(item[0] for item in results)
    refinements = tuple(item[1] for item in results)
    summary = summarise(
        settings.level, settings.additions, settings.min_sites_per_bin, refinements, perf_counter() - started
    )
    if records is not None:
        records.save("level", asdict(summary))
    families = tuple(ReleaseFamily.from_mapping(family) for item in refinements for family in item.release_alternatives)
    return RefinedLevel(components, asdict(summary), families)
