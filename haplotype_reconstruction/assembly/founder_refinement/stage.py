"""Founder refinement of one final-assembly level: the stage entry refine_level.

After every level k of the final L1-L4 assembly (never in a context assembly) each phase
component's founders are re-chosen as paths of original prepared local rows (one row per prepared
block, alleles and evidence masks fixed) to maximise the full-site objective F (objective.py), and
its founder count K is decided between converged fits (count.py): deletion rounds at every level,
addition rounds at the last executed level (the final count step). No truth, pedigree or target K
enters. Components are independent: one runs in this process with the whole budget, several run
in a forkserver pool reading the chromosome's float32 evidence from shared memory (each worker's
threads follow its share of the budget), largest first; each component is recorded as it completes
and a completed level is decoded instead of computed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from .. import count_rule
from ...model.components import CountCall, CountStepDecision, FounderComponents, PreparedPanels
from ... import execution
from .config import FounderRefinementConfig
from .count import count_pass
from .records import ComponentRefinement, LevelSummary, RefinementRecords
from .search import SweepEngine
from .workspace import ComponentWorkspace, component_bytes, component_evidence



@dataclass(frozen=True)
class LevelInput:
    """Inputs of one level's refinement (built by the assembly level loop).

    level: the final assembly level k (1-based).
    prepared: the contig's prepared panels (model.components.PreparedPanel), genomic order: per
        block site_start, positions int64[L_b], inference int8[R_b, L_b] (pre-fill, -1 uncalled;
        every likelihood and the count-rule geometry read it), kept bool[L_b]. The published
        released calls (completion fills) are never read.
    components: the level-k join output (model.components.FounderComponent), contiguous, covering
        every prepared panel; rows int32[K, B] over the prepared panels [start, stop).
    evidence: float32 (S, L, 3), the chromosome's neutral evidence over the contig's sites
        (normalised genotype likelihoods of observed kept cells, exactly 1/3 elsewhere; excluded
        samples uniform): the joins' copy; a component reads the columns of its panels' sites.
    l1_context: None at level 1; from level 2 on, the refined L1 components followed by the
        unrefined L1 join components: inside each L1 span of a component their distinct row paths
        (refined first) form the coarse alphabets.
    cost_scale: AssemblyConfig.cost_scale; a component's per-founder cost is
        c = cost_scale x (mean prepared-panel sites / 200) x S x B (count_rule.founder_cost), S the
        component's informative samples (workspace.count_samples).
    additions: True at the last executed level (deletion and addition rounds); False: deletion
        rounds only.
    count_rule: the shared count rule's configuration.
    evidence_ref: optional execution.SharedArrayRef of a shared-memory segment holding ``evidence``
        (the caller keeps it alive during the call): pooled levels attach it instead of copying the
        evidence into a new segment (12 S L bytes saved per pooled level).
    """

    level: int
    prepared: PreparedPanels
    components: FounderComponents
    evidence: np.ndarray
    l1_context: FounderComponents | None
    cost_scale: float
    additions: bool
    count_rule: count_rule.CountRuleConfig
    evidence_ref: object | None = None


@dataclass(frozen=True)
class LevelResult:
    """components: same boundaries and order as the input, rows replaced by the refined
    prepared-row matrices (int32; K may change) and count_call set for every component;
    refinements: per component summaries (same order); summary: the level census."""

    components: FounderComponents
    refinements: tuple[ComponentRefinement, ...]
    summary: LevelSummary


@dataclass(frozen=True)
class _Settings:
    """What every component task of a level shares."""

    config: FounderRefinementConfig
    min_sites_per_bin: int
    cost_scale: float
    additions: bool
    count_rule: count_rule.CountRuleConfig


def bin_floor(level: int, prepared, config: FounderRefinementConfig) -> int:
    """The level's minimum surrogate bin: levels 1-2 share the chromosome's resolution
    max(proposal_min_sites_per_bin, ceil(chromosome sites / proposal_max_bins)), so the many small
    early components do not each get a chromosome's bin budget; later levels use the minimum."""
    if level <= 2:
        sites = sum(panel.sites for panel in prepared)
        return max(config.proposal_min_sites_per_bin, math.ceil(sites / config.proposal_max_bins))
    return config.proposal_min_sites_per_bin


def macro_context(component, l1_context):
    """(groups, paths) of the component's L1 spans, or None: groups[u] = (first, end) component-local
    blocks of L1 span u (a partition of the component); paths[u] = the distinct L1 row paths on that
    span (stable order: refined L1 paths first). None at level 1 or when every span is one block."""
    if l1_context is None:
        return None
    by_span = {}
    for context in l1_context:
        if context.start < component.start or context.stop > component.stop:
            continue
        rows = by_span.setdefault((context.start - component.start, context.stop - component.start),
                                  {})
        for row in np.asarray(context.rows, np.int64):
            rows.setdefault(tuple(row.tolist()), row)
    groups = sorted(by_span)
    if not groups or all(end - start == 1 for start, end in groups):
        return None
    return groups, [np.asarray(list(by_span[group].values()), np.int64) for group in groups]


def _count_call(refinement: ComponentRefinement) -> CountCall:
    """The component's count call (founder_counts.tsv) from its refinement."""
    if refinement.reason is not None:
        return CountCall(founders=None, founders_before=None, ambiguous=False, count_cycle=False,
                         cycle_undecided=False, decisions=(), reason=refinement.reason)
    return CountCall(
        founders=refinement.founders, founders_before=refinement.founders_before,
        ambiguous=refinement.ambiguous, count_cycle=refinement.count_cycle,
        cycle_undecided=refinement.cycle_undecided,
        decisions=tuple(CountStepDecision(
            kind=d.kind, founders=d.founders, accept_larger=d.accept_larger,
            skipped_by_bound=d.skipped_by_bound, margin=d.margin, band=d.band,
            ambiguous=d.ambiguous) for d in refinement.decisions),
        reason=None)


def refine_component(number, component, panels, evidence, context, settings, budget, fit_records):
    """Refine one component (the converged fit of its incoming panel and its count pass).

    panels: its prepared panels; evidence: its float32 (S, L_c, 3) evidence; context: its L1
    context (macro_context); fit_records: its fit-state scope or None. Returns (refined
    FounderComponent with its count call, ComponentRefinement)."""
    started = time.perf_counter()
    start_rows = np.asarray(component.rows, np.int64)
    founders, blocks = start_rows.shape
    if blocks < 2 or founders < 2:
        refinement = ComponentRefinement(
            component=int(number), start=int(component.start), stop=int(component.stop),
            changed=False, reason="single_block_or_founder", founders_before=int(founders),
            founders=int(founders), elapsed_seconds=time.perf_counter() - started)
        return replace(component, count_call=_count_call(refinement)), refinement
    # The component's set-up kernels run on its whole share (also in a pool worker).
    with execution.numba_threads(max(1, int(budget.threads()))):
        workspace = ComponentWorkspace(panels, evidence,
                                       min_sites_per_bin=settings.min_sites_per_bin,
                                       max_bins=settings.config.proposal_max_bins)
        engine = SweepEngine(workspace, context, settings.config, fit_records, f"component{number}")
    geometry = count_rule.RowGeometry.from_panels(panels)
    # S = the component's informative samples: founder-excluded (uniform) samples and samples
    # without evidence at a complete site score exactly 0 for every panel.
    cost = float(count_rule.founder_cost([panel.sites for panel in panels],
                                         workspace.count_sample_count, settings.cost_scale))
    # Deterministic bootstrap streams per component and decision (positions, size).
    seed_key = [int(panels[0].positions[0]), int(panels[-1].positions[-1]), int(blocks),
                int(founders)]
    result = count_pass(engine, geometry, start_rows, cost, settings.count_rule, budget, seed_key,
                        settings.additions)
    rows = result.rows
    same_shape = rows.shape == start_rows.shape
    refinement = ComponentRefinement(
        component=int(number), start=int(component.start), stop=int(component.stop),
        changed=not (same_shape and np.array_equal(rows, start_rows)), reason=None,
        founders_before=int(founders), founders=int(len(rows)),
        F0=float(result.fits[0].F0), F=float(result.F),
        changed_local_rows=int(np.count_nonzero(rows != start_rows)) if same_shape else None,
        bin_size=int(workspace.bin_size), coarse_units=engine.windows.coarse_units, cost=cost,
        decisions=result.decisions, fits=result.fits, count_cycle=result.count_cycle,
        cycle_undecided=result.cycle_undecided, count_limit=result.count_limit,
        count_limit_reached=result.count_limit_reached, ambiguous=result.ambiguous,
        final_fit=result.final_fit, best_f_replacement=result.best_f_replacement,
        weak_label_junctions=result.weak_junctions, elapsed_seconds=time.perf_counter() - started)
    refined = replace(component, rows=np.ascontiguousarray(rows, np.int32),
                      count_call=_count_call(refinement))
    return refined, refinement


def _component_task(task, shared, budget):
    """One component's refinement in a pool worker (the chromosome evidence attached, not copied)."""
    number, component, panels, context, settings, fit_records = task
    execution.release_memory()
    evidence = component_evidence(shared["evidence"], panels)
    return number, refine_component(number, component, panels, evidence, context, settings, budget,
                                    fit_records)


def summarise(level, additions, min_sites_per_bin, refinements, elapsed_seconds) -> LevelSummary:
    """The level census from its components' refinements."""
    fits = nonconverged = limits = cycles = undecided = replacements = 0
    without_sequence = sequence_rejected = hidden = weak = fallbacks = 0
    sweeps, frontiers, ambiguous = [], [], []
    for item in refinements:
        for fit in item.fits:
            fits += 1
            nonconverged += int(not fit.converged)
            sweeps.append(int(fit.sweeps))
            frontiers.append(int(fit.max_frontier))
            fallbacks += int(fit.frontier_fallbacks)
        limits += int(item.count_limit_reached)
        cycles += int(item.count_cycle)
        undecided += int(item.cycle_undecided)
        weak += len(item.weak_label_junctions)
        for decision in item.decisions:
            without_sequence += int(bool(decision.without_sequence) and decision.kind == "deletion")
            criterion = decision.criterion or {}
            sequence_rejected += int(not criterion.get("accept_larger", True)
                                     and criterion.get("total", {}).get("passes", False)
                                     and not criterion.get("sequence", {}).get("passes", True))
            hidden += len(decision.hidden_sequence_failures or ())
            if decision.ambiguous:
                ambiguous.append(dict(component=item.component, kind=decision.kind,
                                      round=decision.round, margin=decision.margin,
                                      band=decision.band))
        replaced = item.best_f_replacement
        if replaced is not None:
            replacements += 1
            if replaced.ambiguous:
                test = replaced.own_deletion_test or {}
                ambiguous.append(dict(component=item.component, kind="best_f_restored_deletion",
                                      round=test.get("round"), margin=test.get("margin"),
                                      band=test.get("band")))
    return LevelSummary(
        level=int(level), additions=bool(additions), min_sites_per_bin=int(min_sites_per_bin),
        elapsed_seconds=float(elapsed_seconds), components=len(refinements), fits=fits,
        nonconverged_fits=nonconverged, max_sweeps=max(sweeps or [0]),
        mean_sweeps=float(np.mean(sweeps)) if sweeps else 0.0, max_frontier=max(frontiers or [0]),
        frontier_fallbacks=fallbacks, count_limit_hits=limits, count_cycles=cycles,
        count_cycles_undecided=undecided, best_f_replacements=replacements,
        weak_label_junctions=weak, deletions_without_sequence=without_sequence,
        decided_by_sequence_part=sequence_rejected, hidden_sequence_failures=hidden,
        ambiguous_count_calls=tuple(ambiguous))


def refine_level(level: LevelInput, config: FounderRefinementConfig, budget: execution.CpuBudget,
                 records) -> LevelResult:
    """Refine every component of one final-assembly level (module docstring).

    budget: the phase's execution.CpuBudget (live share re-read at numerical boundaries); records:
    the level's store.RecordScope (``<assembly>/l<k>/refine``) or None. Results never depend on the
    budget, worker count or completion order."""
    started = time.perf_counter()
    store = None if records is None else RefinementRecords(records)
    count = len(level.components)
    results = [None] * count
    if store is not None:
        for number in range(count):
            results[number] = store.load_component(number)
        summary = store.load_level()
        if summary is not None and all(item is not None for item in results):
            return LevelResult(tuple(item[0] for item in results),
                               tuple(item[1] for item in results), summary)
    settings = _Settings(config=config, min_sites_per_bin=bin_floor(level.level, level.prepared,
                                                                       config),
                         cost_scale=float(level.cost_scale), additions=bool(level.additions),
                         count_rule=level.count_rule)
    tasks = []
    for number, component in enumerate(level.components):
        if results[number] is not None:
            continue
        panels = tuple(level.prepared[component.start:component.stop])
        fit_records = None if store is None else store.fits(number, component.stop - component.start)
        tasks.append((number, component, panels, macro_context(component, level.l1_context),
                      settings, fit_records))

    def record(number, result):
        results[number] = result
        if store is not None:
            store.save_component(number, *result)

    # Each component task reserves its workspace (workspace.component_bytes); its fits, searches, window solves and
    # score tables reserve their own bytes inside it (additive estimates).
    samples = int(level.evidence.shape[0])
    sizes = [component_bytes(task[2], samples, min_sites_per_bin=settings.min_sites_per_bin,
                             max_bins=config.proposal_max_bins) for task in tasks]
    per_component = max(sizes or [1])
    workers = min(len(tasks), max(1, int(budget.threads())),
                  max(1, int(execution.memory_governor().limit // per_component)))
    label = f"founder_refinement_l{level.level}"
    if workers <= 1:
        functions = [lambda child, task=task: refine_component(
            task[0], task[1], task[2], component_evidence(level.evidence, task[2]), task[3],
            task[4], child, task[5]) for task in tasks]
        for index, result in budget.concurrent(functions, sizes, label=label):
            record(tasks[index][0], result)
    else:
        # Largest components first: the level's tail is the smallest work.
        order = sorted(range(len(tasks)), key=lambda i: -sum(panel.sites for panel in tasks[i][2]))
        shared = level.evidence if level.evidence_ref is None else level.evidence_ref
        with execution.BudgetedPool(budget, processes=workers, shared={"evidence": shared},
                                    label=label) as pool:
            for number, result in pool.imap_unordered(_component_task, [tasks[i] for i in order],
                                                      task_bytes=[sizes[i] for i in order]):
                record(number, result)
    refinements = tuple(item[1] for item in results)
    summary = summarise(level.level, level.additions, settings.min_sites_per_bin, refinements,
                        time.perf_counter() - started)
    print(summary.line(), flush=True)
    if store is not None:
        store.save_level(summary)
    return LevelResult(tuple(item[0] for item in results), refinements, summary)
