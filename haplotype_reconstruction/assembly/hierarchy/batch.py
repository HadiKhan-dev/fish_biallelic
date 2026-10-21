"""One joined batch: its inputs gathered, transitions fitted, the endpoint beam, the certified panel
search, and the chosen panel composed into row paths over the prepared panels.

A batch task is self-contained (small arrays; the evidence comes from the contig's float32 view, which a
pool worker attaches from shared memory), so it runs in a hierarchy pool worker, in the coordinating
process, or on another node (batch queue).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import time

import numba
import numpy as np
from numba import njit, prange

from ...execution.shared_arrays import attach_shared_array, close_shared_memory
from ...model.components import FounderComponent, PreparedPanel, materialise
from ..config import JoinConfig, LinkerConfig
from ..count_rule.config import CountRuleConfig
from ..join.inputs import JoinInputs

JOIN_SCHEMA = 2


@dataclass(frozen=True)
class JoinSettings:
    """Everything a join reads besides its inputs (same for every batch of an assembly level).

    level_one: level 1 joins use every transition gap; later levels limit the beam's gap by
    ``gap_horizon_scale`` and ``recombination_tolerance``.
    """

    linker: LinkerConfig
    join: JoinConfig
    count_rule: CountRuleConfig
    cost_scale: float
    gap_horizon_scale: int
    recombination_tolerance: float
    level_one: bool
    genetic_map: object


@dataclass(frozen=True)
class BatchTask:
    """One batch to join.

    index: batch index in its level. components: the batch's input components with their start/stop
    relative to ``prepared`` (the prepared panels the batch covers, in order). samples: int64[S] the
    batch's informative samples (contig sample indices, ascending). common_usable:
    per-component site masks shared by every input branch; empty for an ordinary join.
    common_usable_lost: sites usable in any input alternative but not in all of them.
    """

    index: int
    components: tuple[FounderComponent, ...]
    prepared: tuple[PreparedPanel, ...]
    samples: np.ndarray
    settings: JoinSettings
    branches: tuple = ()
    branch_sources: tuple = ()
    common_mask_lost: int = 0
    common_usable: tuple = ()
    common_usable_lost: int = 0


@dataclass(frozen=True)
class JoinResult:
    """index: the batch; rows: int32[K, B] the joined component's row paths over the task's prepared
    panels; log_likelihood: the chosen panel's join objective (nats); summary: the search summary
    (``join.search.JoinSummary``); seconds: wall time of the join."""

    index: int
    rows: np.ndarray
    log_likelihood: float
    summary: object
    seconds: float
    release_alternatives: tuple = ()
    minimum_switches: int | None = None
    shadow: dict | None = None


@njit(cache=True, parallel=True, nogil=True)
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


def team_size(budget) -> int:
    """The budget's current thread count, within Numba's pool size."""
    return max(1, min(int(budget.threads()), numba.config.NUMBA_NUM_THREADS))


def join_inputs(task: BatchTask, scoring: np.ndarray, budget) -> JoinInputs:
    """The batch's materialised input components and their evidence."""
    blocks = tuple(panel for panel in materialise(task.components, task.prepared))
    if task.common_usable:
        masked = []
        for block, usable in zip(blocks, task.common_usable):
            kept = np.asarray(block.kept, dtype=bool) & usable
            # materialise owns these small founder arrays. Evidence, positions,
            # bin widths and count-cost axes remain untouched across branches.
            block.inference[:, ~kept] = -1
            masked.append(replace(block, kept=kept))
        blocks = tuple(masked)
    sites = np.concatenate([block.site_indices() for block in blocks])
    offsets = np.zeros(len(blocks) + 1, dtype=np.int64)
    np.cumsum([len(block.positions) for block in blocks], out=offsets[1:])
    numba.set_num_threads(team_size(budget))
    gl = gather_evidence(scoring, task.samples, sites)
    return JoinInputs(blocks=blocks, gl=gl, offsets=offsets, samples=task.samples)


def _run_single(task: BatchTask, scoring: np.ndarray, budget, *, report_switches=False,
                max_branches=4) -> JoinResult:
    """Join one batch (see the module docstring); results never depend on the budget.

    The joined component's founders are returned in canonical order (lexicographic order of their prepared-row
    index paths, design 5.1), so every later level's slot and row-index tie-breaks are content rules."""
    from ..join import beam as join_beam
    from ..join import search as join_search
    from ..linker import em as linker_em
    from ..tie_rules import SharedCounter, canonical_rows
    started = time.perf_counter()
    settings = task.settings
    inputs = join_inputs(task, scoring, budget)
    # The batch's content label (valley-census keys): its inputs, without settings or evidence identity.
    label = batch_digest(task, context="", evidence_identity="")[:16]
    gap = None if settings.level_one else join_beam.max_gap(
        inputs.blocks, settings.genetic_map, settings.gap_horizon_scale,
        settings.recombination_tolerance)
    transitions = linker_em.fit_transitions(inputs, settings.genetic_map, max_gap=gap,
                                            config=settings.linker, budget=budget, label=label)
    numba.set_num_threads(team_size(budget))
    # Tie-class decisions of the beam and the search (the join summary's tie_events).
    counter = SharedCounter()
    beam = join_beam.endpoint_beam(transitions, quota=settings.join.paths_per_endpoint,
                                   max_gap=gap, counter=counter, label=label)

    def beam_runner(quota, overrides):
        # Pool extension and saturation re-runs of this batch's beam.
        return join_beam.endpoint_beam(transitions, quota=quota, overrides=overrides, max_gap=gap,
                                       counter=counter, label=label)

    outcome = join_search.search_panel(
        inputs, beam, beam_runner, config=settings.join, count_rule=settings.count_rule,
        cost_scale=settings.cost_scale, budget=budget, label=label, counter=counter,
        linkage=join_beam.path_linkage(transitions, gap), report_switches=report_switches,
        max_branches=max_branches)
    paths = np.asarray(outcome.paths, dtype=np.int64)
    rows = np.concatenate([block.rows[paths[:, j]] for j, block in enumerate(inputs.blocks)],
                          axis=1).astype(np.int32)
    rows, _ = canonical_rows(rows)
    release_alternatives = []
    for family in getattr(outcome, "release_alternatives", ()):
        panels = [np.concatenate([block.rows[np.asarray(panel, np.int64)[:, j]]
                                  for j, block in enumerate(inputs.blocks)], axis=1).astype(np.int32)
                  for panel in family["panels"]]
        release_alternatives.append(dict(family, start=0, stop=len(task.prepared), panels=panels))
    return JoinResult(index=task.index, rows=rows, log_likelihood=float(outcome.log_likelihood),
                      summary=outcome.summary, seconds=time.perf_counter() - started,
                      release_alternatives=tuple(release_alternatives),
                      minimum_switches=outcome.minimum_switches)


def run_batch(task: BatchTask, scoring: np.ndarray, budget) -> JoinResult:
    """Compare a complete bounded Cartesian input family on one common cohort.

    No child and parent scores are added, and linker scores from different
    fitted models are not used to rank the parent outputs.
    """
    if not task.branches:
        return _run_single(task, scoring, budget)
    from ..alternatives import terminal_winners
    from ..tie_rules import panel_key
    started = time.perf_counter()
    results = []
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
        results.append(_run_single(branch, scoring, budget, report_switches=True,
                                   max_branches=1))
    survivors = terminal_winners(results)
    chosen = min(survivors, key=lambda result: (len(result.rows), result.minimum_switches,
                                               panel_key(result.rows)))
    survivor_ids = {id(result) for result in survivors}
    families = [family for result in results for family in result.release_alternatives
                if family.get("unresolved_search", False) or id(result) in survivor_ids]
    if len(survivors) > 1:
        families.append(dict(start=0, stop=len(task.prepared), reason="parent_join_alternatives",
                             source_decision="common_cohort_parent_join",
                             panels=[result.rows for result in survivors],
                             objective="join_ll", values=[result.log_likelihood for result in survivors],
                             minimum_switches=[result.minimum_switches for result in survivors],
                             status=("ambiguous_k" if len({len(result.rows) for result in survivors}) > 1
                                     else "same_k_equal_LL"), unresolved_search=False,
                             representative_prior="smaller_K_then_exact_minimum_switches",
                             input_families=task.branch_sources))
    diagnostic = dict(combinations=len(results), complete=True,
                      common_samples=len(task.samples), lost_samples=task.common_mask_lost,
                      common_usable_sites=sum(int(mask.sum()) for mask in task.common_usable),
                      lost_usable_sites=task.common_usable_lost,
                      usable_site_reference="union_of_input_alternatives",
                      branch_seconds=[result.seconds for result in results],
                      branch_K=[len(result.rows) for result in results],
                      branch_LL=[result.log_likelihood for result in results],
                      survivors=len(survivors), resolved=len(survivors) == 1 and not families,
                      additional_seconds=max(0.0, time.perf_counter() - started - results[0].seconds))
    summary = replace(chosen.summary,
                      total_full_scores=sum(result.summary.total_full_scores for result in results),
                      count_exact_scores=sum(result.summary.count_exact_scores for result in results),
                      count_exact_sample_scores=sum(result.summary.count_exact_sample_scores for result in results),
                      beam_seconds=sum(result.summary.beam_seconds for result in results),
                      events=(*chosen.summary.events, dict(stage="parent_shadow_join", **diagnostic)))
    return replace(chosen, summary=summary, seconds=time.perf_counter() - started,
                   release_alternatives=tuple(families), shadow=diagnostic)


def join_batch(packed, shared, budget) -> JoinResult:
    """Hierarchy pool task ``(scoring reference, BatchTask)``: join one batch on the float32 evidence,
    attached from shared memory for the task (the join copies its samples' columns at the start)."""
    reference, task = packed
    handle, scoring = attach_shared_array(reference)
    try:
        return run_batch(task, scoring, budget)
    finally:
        del scoring                 # drop the view: a segment with live views cannot close
        close_shared_memory([handle])


def batch_digest(task: BatchTask, *, context: str, evidence_identity: str) -> str:
    """SHA-256 of everything a join reads (B8: explicit typed inputs).

    ``context``: canonical JSON of the join settings and the assembly code identity. The digest
    covers the evidence record identity, the informative samples, the prepared panels' content
    digests (sites, positions, kept, released and inference calls), the shared usable-site
    masks, and every input configuration's relative ranges and rows. Joins are deterministic, so equal digests give equal joins; the
    digest does not depend on where the batch lies in the assembly's panel list, so identical batches
    of different assemblies of the contig share one join record.
    """
    digest = hashlib.sha256(f"assembly-join-v{JOIN_SCHEMA}\0{context}\0{evidence_identity}\0".encode())
    samples = np.ascontiguousarray(task.samples, dtype=np.int64)
    digest.update(f"samples{samples.shape}".encode())
    digest.update(samples.tobytes())
    for panel in task.prepared:
        digest.update(panel.digest.encode())
    for mask in task.common_usable:
        usable = np.ascontiguousarray(mask, dtype=np.bool_)
        digest.update(f"common_usable{usable.shape}".encode())
        digest.update(usable.tobytes())
    for component in task.components:
        rows = np.ascontiguousarray(component.rows, dtype=np.int32)
        digest.update(f"component{component.start}:{component.stop}:{rows.shape}".encode())
        digest.update(rows.tobytes())
    for branch in task.branches:
        digest.update(b"branch")
        for component in branch:
            rows = np.ascontiguousarray(component.rows, dtype=np.int32)
            digest.update(f"component{component.start}:{component.stop}:{rows.shape}".encode())
            digest.update(rows.tobytes())
    if task.branch_sources:
        from ...store.identity import canonical_json
        from ..join.search import _plain
        digest.update(canonical_json(_plain(task.branch_sources)).encode())
    return digest.hexdigest()
