"""Join one batch and compose terminal alternatives on its common evidence domain.

Owns materialisation, canonical composition and the bounded parent comparison.
AS3 fits transitions and AS4 searches each input. Execution resources are supplied
by the caller; storage, clocks and worker attachment belong to stages (P0 U12).
Method: docs/methods/assembly.md#hierarchy.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TYPE_CHECKING, cast

import numpy as np
from numba import prange, types

from haplotype_reconstruction.contracts.assembly import JoinInputs, JoinRecord, JoinSummaryView, JoinTask, ReleaseFamily
from haplotype_reconstruction.contracts.panels import materialise
from haplotype_reconstruction.core.kernels import kernel
from haplotype_reconstruction.core.ties import SharedCounter, TieSettings

if TYPE_CHECKING:
    from ..join.diagnostics import JoinSummary


from ..alternatives import (
    AMBIGUOUS_K,
    COMMON_COHORT_PARENT_JOIN,
    PARENT_JOIN_ALTERNATIVES,
    SAME_K_EQUAL_LL,
    SMALLER_K_THEN_EXACT_MINIMUM_SWITCHES,
)


@kernel(
    overloads=tuple(
        (
            types.Array(types.float32, 3, layout, readonly=r),
            types.Array(types.int64, 1, "C", readonly=samples_frozen),
            types.Array(types.int64, 1, "C", readonly=sites_frozen),
        )
        for layout in ("C", "A")
        for r in (False, True)
        for samples_frozen in (False, True)
        for sites_frozen in (False, True)
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def gather_evidence(scoring, samples, sites):
    """float32[S, L, 3] = scoring[samples][:, sites] (a pure copy spread over the samples)."""
    result = np.empty((len(samples), len(sites), 3), dtype=scoring.dtype)
    for row in prange(len(samples)):
        sample = samples[row]
        for column in range(len(sites)):
            site = sites[column]
            result[row, column, 0] = scoring[sample, site, 0]
            result[row, column, 1] = scoring[sample, site, 1]
            result[row, column, 2] = scoring[sample, site, 2]
    return result


def join_inputs(task: JoinTask, scoring: np.ndarray, budget) -> JoinInputs:
    """The batch's materialised input components and their evidence."""
    blocks = tuple(panel for panel in materialise(task.components, task.prepared))
    if task.common_usable:
        masked = []
        for block, usable in zip(blocks, task.common_usable, strict=False):
            kept = np.asarray(block.kept, dtype=bool) & usable
            # materialise owns these small founder arrays. Evidence, positions,
            # bin widths and count-cost axes remain untouched across branches.
            inference = block.inference.copy()
            inference[:, ~kept] = -1
            masked.append(replace(block, kept=kept, inference=inference))
        blocks = tuple(masked)
    sites = np.concatenate([block.site_indices() for block in blocks])
    offsets: np.ndarray = np.zeros(len(blocks) + 1, dtype=np.int64)
    np.cumsum([len(block.positions) for block in blocks], out=offsets[1:])
    # A slice preserves the shared evidence for contiguous cohorts and genomic spans.
    # Nonadjacent rows or sites still require one gathered array in local input order.
    samples = np.asarray(task.samples)
    if len(samples) and len(sites) and np.all(samples[1:] == samples[:-1] + 1) and np.all(sites[1:] == sites[:-1] + 1):
        gl = scoring[int(samples[0]) : int(samples[-1]) + 1, int(sites[0]) : int(sites[-1]) + 1]
    else:
        gl = gather_evidence(scoring, samples, sites)
    return JoinInputs(blocks=blocks, gl=gl, offsets=offsets, samples=task.samples)


def _run_single(task: JoinTask, scoring: np.ndarray, budget, *, report_switches=False, max_branches=4) -> JoinRecord:
    """Join one batch (see the module docstring); results never depend on the budget.

    The joined component's founders are returned in canonical order (lexicographic order of their prepared-row
    index paths, design 5.1), so every later level's slot and row-index tie-breaks are content rules."""
    from ..count_rule.parameters import CountRuleParameters
    from ..join import beam as join_beam
    from ..join import search as join_search
    from ..linker import em as linker_em
    from ..orders import canonical_rows
    from ..parameters import JoinParameters, LinkerParameters

    settings = task.settings
    ties = TieSettings(reverse=settings.reverse, keep=settings.tie_keep, census=settings.census)
    linker_parameters = LinkerParameters(**asdict(settings.linker))
    join_parameters = JoinParameters(**asdict(settings.join))
    count_parameters = CountRuleParameters(**asdict(settings.count_rule))

    def ordered_map(function, items, task_bytes=0):
        return budget.thread_map(function, items) if budget is not None else tuple(map(function, items))

    workspace = linker_em.LinkerWorkspace(
        map_tasks=ordered_map, lane_threads=None if budget is None else budget.threads
    )
    inputs = join_inputs(task, scoring, budget)
    phase_seconds = {}
    # The batch's content label (valley-census keys): its inputs, without settings or evidence identity.
    label = None
    gap = (
        None
        if settings.level_one
        else join_beam.max_gap(
            inputs.blocks, settings.genetic_map, settings.gap_horizon_scale, settings.recombination_tolerance
        )
    )
    transitions = linker_em.fit_transitions(
        inputs, settings.genetic_map, max_gap=gap, config=linker_parameters, workspace=workspace, label=label
    )
    linker_event = dict(stage="linker_em", gaps=[asdict(event) for event in transitions.diagnostics])
    # Tie-class decisions of the beam and the search (the join summary's tie_events).
    counter = SharedCounter()

    def bounded_beam(quota, overrides=None):
        return join_beam.endpoint_beam(
            transitions, quota=quota, overrides=overrides, max_gap=gap, counter=counter, label=label, tie_settings=ties
        )

    beam = bounded_beam(settings.join.paths_per_endpoint)

    def beam_runner(quota, overrides):
        # Pool extension and saturation re-runs of this batch's beam.
        return bounded_beam(quota, overrides)

    outcome = join_search.search_panel(
        inputs,
        beam,
        beam_runner,
        config=join_parameters,
        count_rule=count_parameters,
        cost_scale=settings.cost_scale,
        claim_threads=None if budget is None else budget.threads,
        map_tasks=ordered_map,
        tie_settings=ties,
        label=label,
        counter=counter,
        linkage=join_beam.path_linkage(transitions, gap),
        report_switches=report_switches,
        max_branches=max_branches,
    )
    # Count work is an inclusive subset of join_search, not an additional phase.
    phase_seconds["count_step"] = float(outcome.summary.phase_seconds.get("count_step", outcome.summary.count.seconds))
    paths = np.asarray(outcome.paths, dtype=np.int64)
    rows = np.concatenate([block.rows[paths[:, j]] for j, block in enumerate(inputs.blocks)], axis=1).astype(np.int32)
    rows, _ = canonical_rows(rows)
    release_alternatives = []
    for family in getattr(outcome, "release_alternatives", ()):
        panels = [
            np.concatenate(
                [block.rows[np.asarray(panel, np.int64)[:, j]] for j, block in enumerate(inputs.blocks)], axis=1
            ).astype(np.int32)
            for panel in family["panels"]
        ]
        release_family = dict(family, start=0, stop=len(task.prepared), panels=panels)
        release_alternatives.append(release_family)
    work_counters = dict(
        samples=inputs.sample_count,
        sites=int(inputs.offsets[-1]),
        blocks=len(inputs.blocks),
        linker_gaps=transitions.gaps,
        beam_paths=(len(beam.covering()) if isinstance(beam, join_beam.CartesianBeamFamily) else len(beam)),
        beam_materialization_bounds=outcome.summary.work_counters.get("beam_materialization_bounds", 0),
        full_scores=outcome.summary.total_full_scores,
        count_scores=outcome.summary.count_exact_scores,
        count_sample_scores=outcome.summary.count_exact_sample_scores,
    )
    summary = replace(
        outcome.summary,
        phase_seconds=phase_seconds,
        work_counters=work_counters,
        events=(linker_event, *outcome.summary.events),
    )
    return JoinRecord(
        index=task.index,
        rows=rows,
        log_likelihood=float(outcome.log_likelihood),
        # The frozen JoinSummary is the view's implementation (as in the branch combination below).
        summary=cast(JoinSummaryView, summary),
        seconds=0.0,
        release_alternatives=tuple(ReleaseFamily.from_mapping(f) for f in release_alternatives),
        minimum_switches=outcome.minimum_switches,
    )


def run_batch(task: JoinTask, scoring: np.ndarray, budget) -> JoinRecord:
    """Compare a complete bounded Cartesian input family on one common cohort.

    No child and parent scores are added, and linker scores from different
    fitted models are not used to rank the parent outputs.
    """
    if not task.branches:
        return _run_single(task, scoring, budget)
    from ..alternatives import terminal_winners
    from ..orders import panel_key

    results: list[JoinRecord] = []
    for components in task.branches:
        if results:
            # CountStep callbacks hold a search cycle. Retire the previous
            # mutable caches before constructing the next sequential branch.
            import gc

            gc.collect()
        branch = replace(task, components=components, branches=(), branch_sources=())
        # The complete incoming family shares at most 16 executions. Do not
        # multiply that allowance by an independent per-input replay budget.
        # Unperformed inner histories remain explicit source uncertainty.
        results.append(_run_single(branch, scoring, budget, report_switches=True, max_branches=1))
    survivors = terminal_winners(results, reverse=task.settings.reverse)
    chosen = min(survivors, key=lambda result: (len(result.rows), result.minimum_switches, panel_key(result.rows)))
    survivor_ids = {id(result) for result in survivors}
    families = [
        dict(family)
        for result in results
        for family in result.release_alternatives
        if family.get("unresolved_search", False) or id(result) in survivor_ids
    ]
    if len(survivors) > 1:
        families.append(
            dict(
                start=0,
                stop=len(task.prepared),
                reason=PARENT_JOIN_ALTERNATIVES,
                source_decision=COMMON_COHORT_PARENT_JOIN,
                panels=[result.rows for result in survivors],
                objective="join_ll",
                values=[result.log_likelihood for result in survivors],
                minimum_switches=[result.minimum_switches for result in survivors],
                status=(AMBIGUOUS_K if len({len(result.rows) for result in survivors}) > 1 else SAME_K_EQUAL_LL),
                unresolved_search=False,
                representative_prior=SMALLER_K_THEN_EXACT_MINIMUM_SWITCHES,
                input_families=task.branch_sources,
            )
        )
    diagnostic = dict(
        combinations=len(results),
        complete=True,
        common_samples=len(task.samples),
        lost_samples=task.common_mask_lost,
        common_usable_sites=sum(int(mask.sum()) for mask in task.common_usable),
        lost_usable_sites=task.common_usable_lost,
        usable_site_reference="union_of_input_alternatives",
        branch_seconds=[result.seconds for result in results],
        branch_K=[len(result.rows) for result in results],
        branch_LL=[result.log_likelihood for result in results],
        survivors=len(survivors),
        resolved=len(survivors) == 1 and not families,
        additional_seconds=0.0,
    )
    # Every completed branch comes from _run_single, which always supplies its summary.
    summaries = tuple(cast(JoinSummaryView, result.summary) for result in results)
    chosen_summary = cast("JoinSummary", chosen.summary)
    phase_seconds: dict[str, float] = {}
    work_counters: dict[str, int] = {}
    for branch_summary in summaries:
        for name, value in branch_summary.phase_seconds.items():
            phase_seconds[name] = phase_seconds.get(name, 0.0) + value
        for name, value in branch_summary.work_counters.items():
            work_counters[name] = work_counters.get(name, 0) + value
    summary = replace(
        chosen_summary,
        phase_seconds=phase_seconds,
        work_counters=work_counters,
        total_full_scores=sum(item.total_full_scores for item in summaries),
        count_exact_scores=sum(item.count_exact_scores for item in summaries),
        count_exact_sample_scores=sum(item.count_exact_sample_scores for item in summaries),
        beam_seconds=sum(item.beam_seconds for item in summaries),
        events=(
            *(
                dict(event, input_branch=index)
                for index, item in enumerate(summaries)
                for event in item.events
                if event.get("stage") == "linker_em"
            ),
            *(event for event in chosen_summary.events if event.get("stage") != "linker_em"),
            dict(stage="parent_shadow_join", **diagnostic),
        ),
    )
    return replace(
        chosen,
        summary=summary,
        seconds=0.0,
        release_alternatives=tuple(ReleaseFamily.from_mapping(f) for f in families),
        shadow=diagnostic,
    )
