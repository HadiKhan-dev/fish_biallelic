"""Join one batch and compose terminal alternatives on its common evidence domain.

Owns materialisation, canonical composition and the bounded parent comparison.
The linker fits transitions and the join searches each distinct join view. Execution resources are supplied
by the caller; storage, clocks and worker attachment belong to stages (U12).
Method: docs/methods/assembly.md#hierarchy.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numba import prange, types

from haplotype_reconstruction.contracts.assembly import JoinInputs, JoinRecord, JoinSummaryView, JoinTask, ReleaseFamily
from haplotype_reconstruction.contracts.panels import ComponentPanel, materialise
from haplotype_reconstruction.core.digest import array_digest, json_digest
from haplotype_reconstruction.core.kernels import kernel
from haplotype_reconstruction.core.ties import SharedCounter, TieSettings

if TYPE_CHECKING:
    from ..join.diagnostics import JoinOutcome, JoinSummary


from ..alternatives import (
    AMBIGUOUS_K,
    COMMON_COHORT_PARENT_JOIN,
    PARENT_JOIN_ALTERNATIVES,
    SAME_K_EQUAL_LL,
    SMALLER_K_THEN_EXACT_MINIMUM_SWITCHES,
)
from ..parameters import MAX_BRANCHES


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


def _input_blocks(task: JoinTask) -> tuple[ComponentPanel, ...]:
    """The batch's materialised input components on its common usable sites."""
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
    return blocks


def source_partition(rows: np.ndarray) -> np.ndarray:
    """int32[K, B]: in each atomic span, the first local founder copying the same prepared row as founder f.

    Founders copying one prepared row share its unknown alleles (linker pair codes, join emissions); the search
    reads only this equality pattern, never which prepared row it is.
    """
    rows = np.asarray(rows)
    return np.argmax(rows[:, None, :] == rows[None, :, :], axis=1).astype(np.int32)


def join_view(task: JoinTask) -> str:
    """Digest of everything a join search reads from its input components (u13 shared branch searches).

    Per input component, in local founder order: positions, the kept mask and the inference calls after the
    batch's common usable-site masks, the atomic span offsets and site starts (which evidence columns are read),
    and the source-row partition. The founder count is the calls' shape. The search, its tie conventions and its
    outcome (local row paths, likelihoods, families in local terms) are a function of these arrays and of what all
    branches of one batch share: the prepared panels, cohort, evidence and settings. Views are therefore compared
    only between branches of one batch. Prepared-row identities, released calls, ALT probabilities, supporters,
    breaks and count calls are not read by the search (the census label, which does read the rows, is handled by
    run_batch).
    """
    return json_digest(
        [
            [
                array_digest(array)
                for array in (
                    block.positions,
                    block.kept,
                    block.inference,
                    block.atomic_offsets,
                    block.atomic_site_starts,
                    source_partition(block.rows),
                )
            ]
            for block in _input_blocks(task)
        ]
    )


def join_inputs(task: JoinTask, scoring: np.ndarray) -> JoinInputs:
    """The batch's materialised input components and their evidence."""
    blocks = _input_blocks(task)
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


def _run_single(
    task: JoinTask, scoring: np.ndarray, budget, *, report_switches=False, max_branches=MAX_BRANCHES
) -> JoinRecord:
    """Join one batch (see the module docstring); results never depend on the budget.

    The joined component's founders are returned in canonical order (lexicographic order of their prepared-row
    index paths), so every later level's slot and row-index tie-breaks are content rules."""
    return _record(task, _search(task, scoring, budget, report_switches=report_switches, max_branches=max_branches))


@dataclass(frozen=True)
class _LocalJoin:
    """One join search in local row terms: its outcome (local row paths, likelihood, summary, families with local
    provenance), the linker's event and the search's input-side work counters."""

    outcome: JoinOutcome
    linker_event: dict[str, Any]
    work_counters: dict[str, int]


def _search(task: JoinTask, scoring: np.ndarray, budget, *, report_switches, max_branches) -> _LocalJoin:
    """The join search of one batch in local row terms; its outcome reads only the batch's join view (and, with a
    census attached, the census label)."""
    from ..join import beam as join_beam
    from ..join import search as join_search
    from ..linker import em as linker_em

    settings = task.settings
    ties = TieSettings(reverse=settings.reverse, keep=settings.tie_keep, census=settings.census)
    linker_parameters = settings.linker
    join_parameters = settings.join
    count_parameters = settings.count_rule

    def ordered_map(function, items, task_bytes=0):
        return budget.thread_map(function, items) if budget is not None else tuple(map(function, items))

    workspace = linker_em.LinkerWorkspace(
        map_tasks=ordered_map, lane_threads=None if budget is None else budget.threads, census=settings.census
    )
    inputs = join_inputs(task, scoring)
    # The batch's content label (valley-census keys): its inputs, without settings or evidence identity.
    label = (
        f"join/{task.prepared[0].positions[0]}-{task.prepared[-1].positions[-1]}/"
        + json_digest([component.rows.tolist() for component in task.components])
        if settings.census is not None
        else None
    )
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
    return _LocalJoin(
        outcome=outcome,
        linker_event=linker_event,
        work_counters=dict(
            samples=inputs.sample_count,
            sites=int(inputs.offsets[-1]),
            blocks=len(inputs.blocks),
            linker_gaps=transitions.gaps,
            beam_paths=(len(beam.covering()) if isinstance(beam, join_beam.CartesianBeamFamily) else len(beam)),
        ),
    )


def _record(task: JoinTask, searched: _LocalJoin) -> JoinRecord:
    """The batch's record: the local outcome mapped through this task's own input rows.

    Local paths become prepared-row paths through ``task.components`` (the founders' rows of the inputs this task
    joins), then canonical order. Families keep their local provenance (values, replay panels, choices); only
    their panels are mapped. The summary is the search's, with its work counters.
    """
    from ..orders import canonical_rows

    outcome = searched.outcome
    sources = [np.asarray(component.rows) for component in task.components]
    paths = np.asarray(outcome.paths, dtype=np.int64)
    rows = np.concatenate([source[paths[:, j]] for j, source in enumerate(sources)], axis=1).astype(np.int32)
    rows, _ = canonical_rows(rows)
    release_alternatives = []
    for family in getattr(outcome, "release_alternatives", ()):
        panels = [
            np.concatenate(
                [source[np.asarray(panel, np.int64)[:, j]] for j, source in enumerate(sources)], axis=1
            ).astype(np.int32)
            for panel in family["panels"]
        ]
        release_family = dict(family, start=0, stop=len(task.prepared), panels=panels)
        release_alternatives.append(release_family)
    work_counters = dict(
        searched.work_counters,
        beam_materialization_bounds=outcome.summary.work_counters.get("beam_materialization_bounds", 0),
        full_scores=outcome.summary.total_full_scores,
        count_scores=outcome.summary.count_exact_scores,
        count_sample_scores=outcome.summary.count_exact_sample_scores,
        # Replays merged into an earlier execution and the sweeps they adopted (present only when one merged).
        **{name: value for name, value in outcome.summary.work_counters.items() if name.startswith("replay_merge")},
    )
    summary = replace(
        outcome.summary,
        work_counters=work_counters,
        events=(searched.linker_event, *outcome.summary.events),
    )
    return JoinRecord(
        index=task.index,
        rows=rows,
        log_likelihood=float(outcome.log_likelihood),
        summary=summary,
        release_alternatives=tuple(ReleaseFamily.from_mapping(f) for f in release_alternatives),
        minimum_switches=outcome.minimum_switches,
    )


def run_batch(task: JoinTask, scoring: np.ndarray, budget) -> JoinRecord:
    """Compare a complete bounded Cartesian input family on one common cohort.

    No child and parent scores are added, and linker scores from different
    fitted models are not used to rank the parent outputs. Branches with equal
    join views share one search (see join_view); the batch's work counters count
    each search once, and its shadow diagnostic names the search of each branch.
    """
    if not task.branches:
        return _run_single(task, scoring, budget)
    from ..alternatives import terminal_winners
    from ..orders import panel_key

    results: list[JoinRecord] = []
    # One search per distinct join view (u13). Branches whose views are equal (tied inputs that differ only in
    # which of several inference-identical prepared rows a founder copies) run the same computation, so one
    # search serves them all and each branch maps its local outcome through its own input rows: every branch
    # keeps its own record and families, exactly as if it had been searched alone. With a tie census attached,
    # every branch is searched, so each records its decisions under its own row label.
    searches: list[_LocalJoin] = []
    search_of_view: dict[object, int] = {}
    branch_search: list[int] = []
    for index, components in enumerate(task.branches):
        branch = replace(task, components=components, branches=(), branch_sources=())
        view = join_view(branch) if task.settings.census is None else index
        if view not in search_of_view:
            if searches:
                # CountStep callbacks hold a search cycle. Retire the previous
                # mutable caches before constructing the next sequential branch.
                gc.collect()
            # The complete incoming family shares at most 16 executions. Do not
            # multiply that allowance by an independent per-input replay budget.
            # Unperformed inner histories remain explicit source uncertainty.
            search_of_view[view] = len(searches)
            searches.append(_search(branch, scoring, budget, report_switches=True, max_branches=1))
        branch_search.append(search_of_view[view])
        results.append(_record(branch, searches[branch_search[-1]]))
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
        branch_K=[len(result.rows) for result in results],
        branch_LL=[result.log_likelihood for result in results],
        survivors=len(survivors),
        resolved=len(survivors) == 1 and not families,
        # The searches run and, per branch, the search whose local outcome it maps (branches mapping one search
        # report that search's linker event).
        searches=len(searches),
        branch_search=branch_search,
    )
    # Every completed branch comes from _record, which always supplies its summary.
    summaries = tuple(cast(JoinSummaryView, result.summary) for result in results)
    chosen_summary = cast("JoinSummary", chosen.summary)
    # Work is counted once per search run, not once per branch mapping it.
    executed = tuple(summaries[branch_search.index(search)] for search in range(len(searches)))
    work_counters: dict[str, int] = {}
    for branch_summary in executed:
        for name, value in branch_summary.work_counters.items():
            work_counters[name] = work_counters.get(name, 0) + value
    summary = replace(
        chosen_summary,
        work_counters=work_counters,
        total_full_scores=sum(item.total_full_scores for item in executed),
        count_exact_scores=sum(item.count_exact_scores for item in executed),
        count_exact_sample_scores=sum(item.count_exact_sample_scores for item in executed),
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
        release_alternatives=tuple(ReleaseFamily.from_mapping(f) for f in families),
        shadow=diagnostic,
    )
