"""Refine one component through count choice and optional fixed-K polish.

Inputs and capabilities are supplied by the stage;
no files, runtime configuration, clocks or live resource admission occur here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.contracts.assembly import ComponentRefinement, LevelSummary
from haplotype_reconstruction.contracts.genetic_map import ChromosomeGeneticMap
from haplotype_reconstruction.contracts.panels import CountCall, CountStepDecision, FounderComponent, PreparedPanel
from haplotype_reconstruction.core import ties
from haplotype_reconstruction.core.ties import TieCensus

from .. import orders as tie_rules
from ..count_rule import criterion as count_criterion
from ..count_rule import geometry as count_geometry
from ..count_rule.parameters import CountRuleParameters
from .capabilities import ComponentWorkspaceView, RefinementExecution, RefinementRecords
from .count import _terminal_family, count_pass
from .fit import SweepEngine
from .parameters import RefinementParameters
from .reductions import panel_objective
from .work_limit import WindowWorkLimit

if TYPE_CHECKING:
    from haplotype_reconstruction.contracts.assembly import WindowDomain

MacroContext = tuple[list[tuple[int, int]], list[NDArray[np.int64]]]


@dataclass(frozen=True)
class RefinementSettings:
    config: RefinementParameters
    min_sites_per_bin: int
    cost_scale: float
    count_rule: CountRuleParameters
    level: int = 0
    genetic_map: ChromosomeGeneticMap | None = None
    reverse: bool = False
    census: Callable[[TieCensus], None] | None = field(
        default=None, repr=False, compare=False, metadata={"stored": False}
    )


def _tie_record(counter):
    return {
        entry.kind: {name: getattr(entry, name) for name in ("decisions", "inside_tie", "wide", "boundary")}
        for entry in counter.record().entries
    }


def bin_floor(level: int, prepared, config: RefinementParameters) -> int:
    """The level's minimum surrogate bin: levels 1-2 share the chromosome's resolution
    max(proposal_min_sites_per_bin, ceil(chromosome sites / proposal_max_bins)), so the many small
    early components do not each get a chromosome's bin budget; later levels use the minimum."""
    if level <= 2:
        sites = sum(panel.sites for panel in prepared)
        return max(config.proposal_min_sites_per_bin, math.ceil(sites / config.proposal_max_bins))
    return config.proposal_min_sites_per_bin


def macro_context(component, l1_context):
    """(groups, paths) of the component's L1 spans, or None: groups[u] = (first, end) component-local
    blocks of L1 span u (only fully contained spans; uncovered edges are allowed); paths[u] = the distinct L1 row
    paths on that
    span (stable order: refined L1 paths first). None at level 1 or when every span is one block."""
    if l1_context is None:
        return None
    by_span = {}
    for context in l1_context:
        if context.start < component.start or context.stop > component.stop:
            continue
        rows = by_span.setdefault((context.start - component.start, context.stop - component.start), {})
        for row in np.asarray(context.rows, np.int64):
            rows.setdefault(tuple(row.tolist()), row)
    groups = sorted(by_span)
    if not groups or all(end - start == 1 for start, end in groups):
        return None
    return groups, [np.asarray(list(by_span[group].values()), np.int64) for group in groups]


def _final_alternatives(result) -> tuple:
    """Current final-F alternatives only, never the union of historical decisions."""
    return tuple(tuple(family["panels"]) for family in result.live_alternatives)


def _release_alternatives(result, component, level):
    """Biological families and nonwithholding count-verification metadata for release."""
    return tuple(
        dict(family, start=int(component.start), stop=int(component.stop), source_level=int(level))
        for family in (*result.live_alternatives, *result.unresolved_alternatives, *result.count_unverified)
    )


def _refinement_inputs(component, incoming):
    """Whole-span terminal families can be compared on this workspace's common mask.

    A partial-span source or an uncontinued source has no justified full panel
    correspondence. Keep it explicitly unresolved while refining the representative.
    """
    nominal = tie_rules.panel_key(component.rows)
    panels = {nominal}
    retained = []
    for family in incoming:
        exact = int(family["start"]) == component.start and int(family["stop"]) == component.stop
        family_panels = {tie_rules.panel_key(np.asarray(panel)) for panel in family["panels"]}
        if exact and not family.get("unresolved_search", False) and nominal in family_panels:
            panels.update(family_panels)
        elif family.get("unresolved_search", False):
            retained.append(family)
        else:
            reason = "unresolved_component_correspondence" if exact else "unresolved_refinement_continuation"
            retained.append(dict(family, source_reason=family.get("reason"), reason=reason, unresolved_search=True))
    return tuple(sorted(panels)), tuple(retained)


def _polish(engine, result, budget):
    """The polish refit (RefinementParameters.polish) of the count pass's decided
    panel and of its same-K equal-F alternatives (converge(polish=True): beam escapes and bridges at
    every ordinary fixed point). The best polished fit (tie classes of F, content order) and its
    equal-F family follow the count pass's own terminal rule (count._terminal_family). K is never
    changed. Returns (the CountPass with the polished panel, the polish fits appended to its fits; the polish
    record: the count pass's final fit and F, the polished F and gain, the polish fits' labels). The count pass's
    decisions, bands and best-F restoration were taken on the pre-polish panel and are not re-evaluated
    (count_reevaluated False)."""
    panels = [np.asarray(result.rows, np.int64)]
    seen = {tie_rules.panel_key(panels[0])}
    for family in result.live_alternatives:
        for panel in family["panels"]:
            key = tie_rules.panel_key(np.asarray(panel))
            if key not in seen:
                seen.add(key)
                panels.append(np.asarray(panel, np.int64))
    labels = ["polish" if i == 0 else f"polish.alt{i}" for i in range(len(panels))]
    if len(panels) < 2:
        fits = [engine.converge(panels[0], labels[0], budget, polish=True)]
    else:
        # The alternatives' polish fits are independent: concurrent, in panel order (each search reserves its own).
        fits = [None] * len(panels)
        ordinals = engine.bind_fits(
            len(panels), [(label, panel, "polish") for label, panel in zip(labels, panels, strict=False)]
        )
        for index, fit in budget.concurrent(
            [
                lambda child, panel=panel, label=label, ordinal=ordinal: engine.converge(
                    panel, label, child, polish=True, ordinal=ordinal
                )
                for panel, label, ordinal in zip(panels, labels, ordinals, strict=False)
            ],
            0,
            label="founder_refinement polish",
        ):
            fits[index] = fit
    values = [float(fit.summary.F) for fit in fits]
    contents = [tie_rules.panel_key(fit.rows) for fit in fits]
    classes = ties.fixed_policy(
        "assembly.refine.polish", tie_rules.quantum("refine_F", max(abs(v) for v in values)), reverse=engine.reverse
    ).classes(values, content=contents, descending=True)
    final = fits[classes.winner]
    chosen, scores, live = _terminal_family(engine, fits, final, engine.per_sample(final.rows, budget), budget)
    recorded = (*result.fits, *(fit.summary for fit in fits))
    unchanged = (
        tie_rules.panel_key(chosen.rows) == tie_rules.panel_key(result.rows) and live == result.live_alternatives
    )
    record = dict(
        from_fit=result.final_fit,
        F_before=float(result.F),
        F_after=float(result.F if unchanged else panel_objective(scores)),
        gain=0.0 if unchanged else panel_objective(scores) - float(result.F),
        changed=not unchanged,
        fits=tuple(fit.summary.label for fit in fits),
        starts=len(panels),
        count_reevaluated=False,
    )
    if unchanged:
        return replace(result, fits=recorded), record
    return replace(
        result,
        rows=chosen.rows,
        scores=scores,
        F=panel_objective(scores),
        fits=recorded,
        final_fit=chosen.summary.label,
        weak_junctions=chosen.weak_junctions,
        live_alternatives=live,
    ), record


def _count_call(refinement: ComponentRefinement) -> CountCall:
    """The component's count call (founder_counts.tsv) from its refinement."""
    if refinement.reason is not None:
        return CountCall(
            founders=None,
            founders_before=None,
            ambiguous=False,
            count_cycle=False,
            cycle_undecided=False,
            decisions=(),
            reason=refinement.reason,
        )
    return CountCall(
        founders=refinement.founders,
        founders_before=refinement.founders_before,
        ambiguous=refinement.ambiguous,
        count_cycle=False,
        cycle_undecided=False,
        decisions=tuple(
            CountStepDecision(
                kind=d.kind,
                founders=d.founders,
                accept_larger=d.accept_larger,
                skipped_by_bound=d.skipped_by_bound,
                margin=d.margin,
                band=d.band,
                ambiguous=d.ambiguous,
            )
            for d in refinement.decisions
        ),
        reason=None,
    )


def refine_component(
    number: int,
    component: FounderComponent,
    panels: Sequence[PreparedPanel],
    context: MacroContext | None,
    settings: RefinementSettings,
    budget: RefinementExecution,
    fit_records: RefinementRecords | None,
    incoming: Sequence[Mapping[str, Any]] = (),
    *,
    workspace: ComponentWorkspaceView | None,
) -> tuple[FounderComponent, ComponentRefinement]:
    """Refine one component (the converged fit of its incoming panel and its count pass).

    panels: its prepared panels; workspace supplies its float32 (S, L_c, 3) evidence; context: its L1
    context (macro_context); fit_records: its fit-state scope or None. Returns (refined
    FounderComponent with its count call, ComponentRefinement). The refined founders are in
    canonical order (rows sorted lexicographically)."""
    start_rows = np.asarray(component.rows, np.int64)
    founders, blocks = start_rows.shape
    if blocks < 2 or founders < 2:
        refinement = ComponentRefinement(
            component=int(number),
            start=int(component.start),
            stop=int(component.stop),
            changed=False,
            reason="single_block_or_founder",
            founders_before=int(founders),
            founders=int(founders),
            release_alternatives=tuple(incoming),
            input_alternatives=tuple(incoming),
        )
        return replace(component, count_call=_count_call(refinement)), refinement
    # Tie-class decisions of the component (fits and count pass) and its valley-census prefix.
    counter = tie_rules.SharedCounter()
    census_prefix = f"L{settings.level}/{int(component.start)}-{int(component.stop)}/K{int(founders)}"
    # The component's set-up kernels run on its whole share (also in a pool worker).
    with budget.scope(max(1, int(budget.threads()))):
        engine = SweepEngine(
            cast(ComponentWorkspaceView, workspace),
            context,
            settings.config,
            fit_records,
            f"component{number}",
            counter=counter,
            census_prefix=census_prefix,
            census=settings.census,
            source_start=component.start,
            reverse=settings.reverse,
        )
    geometry = count_geometry.RowGeometry.from_panels(panels, reverse=settings.reverse)
    # S = the component's informative samples: founder-excluded (uniform) samples and samples
    # without evidence at a complete site score exactly 0 for every panel.
    cost = float(
        count_criterion.founder_cost(
            [panel.sites for panel in panels],
            cast(ComponentWorkspaceView, workspace).count_sample_count,
            settings.cost_scale,
        )
    )
    # Deterministic bootstrap streams per component and decision (positions, size).
    seed_key = [int(panels[0].positions[0]), int(panels[-1].positions[-1]), int(blocks), int(founders)]
    alternative_starts, retained = _refinement_inputs(component, incoming)
    try:
        with engine.memoized_windows(enabled=len(alternative_starts) > 1):
            result = count_pass(
                engine,
                geometry,
                start_rows,
                cost,
                settings.count_rule,
                budget,
                seed_key,
                counter=counter,
                alternative_starts=alternative_starts,
            )
    except WindowWorkLimit:
        # No interrupted direction, count fit or replay may enter a
        # scientific comparison. Preserve the complete original component/families.
        stop = engine.windows.work.stop_record()
        domains: tuple[WindowDomain, ...] | tuple[dict[str, Any], ...]
        domains, work = (stop.domain,), stop.work
        domains = tuple(asdict(cast("WindowDomain", domain)) for domain in domains)
        work = asdict(work)
        family = dict(
            start=int(component.start),
            stop=int(component.stop),
            source_level=int(settings.level),
            source_decision="component_refinement",
            reason="window_work_bound",
            status="unperformed_window_search",
            unresolved_search=True,
            panels=(),
            window_domains=domains,
            window_domains_scope="observed work-limit witnesses; remaining component search unperformed",
            domain_basis="unevaluated single-focal search domains; not biological alternatives",
        )
        refinement = ComponentRefinement(
            component=int(number),
            start=int(component.start),
            stop=int(component.stop),
            changed=False,
            reason="window_work_bound",
            window_stop=stop,
            founders_before=int(founders),
            founders=int(founders),
            bin_size=int(cast(ComponentWorkspaceView, workspace).bin_size),
            coarse_units=engine.windows.coarse_units,
            cost=cost,
            branch_evaluations=0,
            branch_status="window_work_bound",
            input_alternatives=tuple(incoming),
            release_alternatives=(*incoming, family),
            work_counters=dict(engine._work_counters, selected_count_decisions=0, selected_fits=0, **work),
        )
        return replace(component, count_call=_count_call(refinement)), refinement
    polish: dict[str, Any] = dict(status="off")
    stop = None
    if engine.beam is not None:
        try:
            result, record = _polish(engine, result, budget)
            polish = dict(status="done", **record)
        except WindowWorkLimit:
            # The component's window-work account ran out during the polish: the count pass's complete,
            # decided panel stands unpolished (declared policy: the polish is a separate stage after the count
            # pass, whose decisions never depend on it; RefinementParameters.window_work_coefficient).
            stop = engine.windows.work.stop_record()
            polish = dict(
                status="window_work_bound", from_fit=result.final_fit, F_before=float(result.F), count_reevaluated=False
            )
    rows, _ = tie_rules.canonical_rows(np.asarray(result.rows, np.int64))
    same_shape = rows.shape == start_rows.shape
    refinement = ComponentRefinement(
        component=int(number),
        start=int(component.start),
        stop=int(component.stop),
        changed=not (same_shape and np.array_equal(rows, start_rows)),
        reason=None,
        founders_before=int(founders),
        founders=int(len(rows)),
        F0=float(cast(float, result.input_F0)),
        F=float(result.F),
        changed_local_rows=int(np.count_nonzero(rows != start_rows)) if same_shape else None,
        bin_size=int(cast(ComponentWorkspaceView, workspace).bin_size),
        coarse_units=engine.windows.coarse_units,
        cost=cost,
        decisions=result.decisions,
        fits=result.fits,
        ambiguous=result.ambiguous,
        final_fit=result.final_fit,
        best_f_replacement=result.best_f_replacement,
        weak_label_junctions=result.weak_junctions,
        tie_events=_tie_record(counter),
        alternatives=_final_alternatives(result),
        release_alternatives=(*retained, *_release_alternatives(result, component, settings.level)),
        input_alternatives=tuple(incoming),
        branch_evaluations=result.branch_evaluations,
        branch_reused_fits=result.branch_reused_fits,
        branch_status=result.branch_status,
        selected_branch=result.selected_branch,
        count_continuations=result.count_continuations,
        count_branch_evaluations=result.count_branch_evaluations,
        polish=polish,
        window_stop=stop,
        work_counters=dict(
            selected_count_decisions=len(result.decisions), selected_fits=len(result.fits), **engine._work_counters
        ),
    )
    refined = replace(component, rows=np.ascontiguousarray(rows, np.int32), count_call=_count_call(refinement))
    return refined, refinement


def summarise(level, min_sites_per_bin, refinements, elapsed_seconds) -> LevelSummary:
    """The level census from its components' refinements."""
    fits = nonconverged = replacements = 0
    without_sequence = sequence_rejected = hidden = weak = fallbacks = 0
    sweeps, frontiers, ambiguous = [], [], []
    for item in refinements:
        for fit in item.fits:
            fits += 1
            nonconverged += int(not fit.converged)
            sweeps.append(int(fit.sweeps))
            frontiers.append(int(fit.max_frontier))
            fallbacks += int(fit.frontier_fallbacks)
        weak += len(item.weak_label_junctions)
        for decision in item.decisions:
            without_sequence += int(bool(decision.without_sequence) and decision.kind == "deletion")
            criterion = decision.criterion or {}
            sequence_rejected += int(
                not criterion.get("accept_larger", True)
                and criterion.get("total", {}).get("passes", False)
                and not criterion.get("sequence", {}).get("passes", True)
            )
            hidden += len(decision.hidden_sequence_failures or ())
            if decision.ambiguous:
                ambiguous.append(
                    dict(
                        component=item.component,
                        kind=decision.kind,
                        round=decision.round,
                        margin=decision.margin,
                        band=decision.band,
                    )
                )
        replaced = item.best_f_replacement
        if replaced is not None:
            replacements += 1
            if replaced.ambiguous:
                test = replaced.own_deletion_test or {}
                ambiguous.append(
                    dict(
                        component=item.component,
                        kind="best_f_restored_deletion",
                        round=test.get("round"),
                        margin=test.get("margin"),
                        band=test.get("band"),
                    )
                )
    return LevelSummary(
        level=int(level),
        min_sites_per_bin=int(min_sites_per_bin),
        elapsed_seconds=float(elapsed_seconds),
        components=len(refinements),
        fits=fits,
        nonconverged_fits=nonconverged,
        max_sweeps=max(sweeps or [0]),
        mean_sweeps=float(np.mean(sweeps)) if sweeps else 0.0,
        max_frontier=max(frontiers or [0]),
        frontier_fallbacks=fallbacks,
        best_f_replacements=replacements,
        weak_label_junctions=weak,
        deletions_without_sequence=without_sequence,
        decided_by_sequence_part=sequence_rejected,
        hidden_sequence_failures=hidden,
        ambiguous_count_calls=tuple(ambiguous),
        work_counters={
            name: sum(item.work_counters.get(name, 0) for item in refinements)
            for name in (
                "selected_count_decisions",
                "selected_fits",
                "searches",
                "windows",
                "templates_solved",
                "templates_reused",
                "dp",
                "fixing_dps",
                "guard_shared_tests",
                "guard_traced_blocks",
                "beam_truncations",
                "beam_fallbacks",
                "content_rounds",
                "round_cap",
                "stale_evaluated",
                "exchange_cap",
                "escape_windows",
                "escape_beam_cut",
                "escape_beam_binding",
                "long_windows",
                "long_beam_cut",
                "long_beam_binding",
                "long_candidates",
                "lex_items",
                "lex_abort",
            )
        },
    )
