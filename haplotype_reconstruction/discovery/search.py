"""Adaptive reversible search over complete panels of one block.

The archive ranks every fitted complete panel (one per canonical key, the lowest-NLL fit) and
keeps, per K, the panels the search reads: the ``beam_width`` lowest; the minimum-NLL panel of
each represented K is its *representative* and is cavity-scored (score
cached by key; at most ``exact_score_backstop(K_rep)`` scores, K_rep the largest represented K).
The selected panel is the representative with the best score (ties: smaller K, then key).

Each expansion takes the selected K's beam (its ``beam_width`` lowest-NLL archived panels): the
first panel not yet expanded ordinarily gets the ordinary moves (deaths, gauge rewires, data
births; then, if the score did not improve, birth-delete replacements); once every beam panel
was expanded ordinarily, the first not yet expanded in the residual phase gets soft-residual
births (then replacements, then, if still no improvement, a K + 2 bridge: data and soft births
of the K + 1 representative). The search stops when the beam is exhausted in both phases, or at
the reported expansion backstop. There is no K grid and no K cap besides the identifiability
ceiling min(2 N_active, 2^L).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cavity import CavityWorkspace, score_solutions
from .fixed_k.fit import prepare_workspace
from .moves import (MaximumCuts, SoftBirths, data_births, growth_inputs, initial_solutions,
                    ordinary_moves, replacements, soft_births)
from .panel import FixedKSolution, deduplicate
from .residual_births import ResidualInputs

# Relative K of the moves a search step generates (death, same-K, birth, bridge).
_MOVE_OFFSETS = (-1, 0, 1, 2)


@dataclass(frozen=True)
class SearchSummary:
    """What one block's search did (diagnostics; nothing downstream decides on it).

    k_values: int16[R] represented K values, ascending; k_log_scores: float64[R] the cavity
        log score (leave-one-out predictive + telescoping K prior + unordered set code) of each
        K's representative. The selected K maximises it.
    natural_k_ceiling: identifiability ceiling min(2 N_active, 2^L).
    expansions, data_starts, exact_scores, exact_score_cache_hits: work counts.
    residual_calls, residual_rows: soft-residual birth calls and the novel rows they emitted.
    limits: reported search limitations in first-occurrence order, from
        "expansion_backstop", "exact_score_backstop", "proposal_budget",
        "replacement_proposal_budget", "soft_residual_proposal_budget",
        "gauge_max_cut_ties_truncated", "mean_field_nonconvergence".
    heuristic_gauge_cuts: some expanded panel had K > exact_cut_max_k, so its gauge rewires came
        from the degree-ordered local-search maximum cut (reported here, not as a limit).
    stop_reason: "generated_neighbourhood_local_optimum", "incumbent_neighbourhood_search_limited"
        or "expansion_backstop".
    certified: the selected panel's generated neighbourhood was completely generated and scored
        without improvement (a local certificate, not a global one).
    """

    k_values: np.ndarray
    k_log_scores: np.ndarray
    natural_k_ceiling: int
    expansions: int
    data_starts: int
    exact_scores: int
    exact_score_cache_hits: int
    residual_calls: int
    residual_rows: int
    limits: tuple[str, ...]
    heuristic_gauge_cuts: bool
    stop_reason: str
    certified: bool


@dataclass(frozen=True)
class BlockSearch:
    """The selected solution of a block search and its summary."""

    selected: FixedKSolution
    summary: SearchSummary


def natural_k_ceiling(n_samples: int, n_sites: int) -> int:
    """The largest identifiable K: 2 copies per sample with depth, at most 2^L rows."""
    sample_ceiling = 2 * int(n_samples)
    if n_sites < sample_ceiling.bit_length():
        return min(sample_ceiling, 1 << int(n_sites))
    return sample_ceiling


def _score_key(score):
    return (-score.log_score, score.k, score.solution.key)


def search_block(likelihoods, observed, config) -> BlockSearch:
    """Search the panel of one block's discovery sites.

    likelihoods: float64[N, L, 3] normalised GLs; observed: bool[N, L] (at least one True).
    Unobserved cells are set to exactly 1/3 before any seed, fit or score.
    """
    observed = np.ascontiguousarray(observed, dtype=np.bool_)
    likelihood = np.array(likelihoods, dtype=np.float64, order="C", copy=True)
    likelihood[~observed] = 1.0 / 3.0
    active = np.ascontiguousarray(observed.any(axis=1))
    n_sites = likelihood.shape[1]
    ceiling = natural_k_ceiling(int(active.sum()), n_sites)
    search = config.search
    tolerance = search.score_tolerance
    max_iterations = config.model.max_iterations

    workspace = prepare_workspace(likelihood, observed, config.model)
    growth = growth_inputs(likelihood, observed)
    residual_inputs = ResidualInputs(likelihood, observed)
    cuts = MaximumCuts(search.exact_cut_max_k, search.max_cut_ties)
    data_solutions = initial_solutions(workspace, likelihood, active, config)
    cavity_workspace = CavityWorkspace(
        likelihood if bool(active.all()) else likelihood[active], config.cavity)

    # Per canonical key the order of its lowest-NLL fit; per K only the beam_width best fits (the
    # first is the representative): the search reads nothing else of the thousands of panels a
    # large-K block fits, so the archive's memory stays bounded.
    archive_order: dict[bytes, tuple] = {}
    archive_beams: dict[int, list] = {}
    represented = dict(k=0)
    representatives: dict[int, FixedKSolution] = {}
    score_cache: dict = {}
    scores: dict = {}
    soft_cache: dict = {}
    limits: list[str] = []
    state = dict(exact_scores=0, cache_hits=0, mean_field_failed=False)

    def add(solutions):
        width = search.beam_width
        for solution in solutions:
            order = solution.order()
            previous = archive_order.get(solution.key)
            if previous is not None and not order < previous:
                continue
            archive_order[solution.key] = order
            represented["k"] = max(represented["k"], solution.k)
            panels = archive_beams.setdefault(solution.k, [])
            if previous is not None:
                panels[:] = [entry for entry in panels if entry.key != solution.key]
            if len(panels) < width or order < panels[-1].order():
                panels.append(solution)
                panels.sort(key=FixedKSolution.order)
                del panels[width:]

    def represented_k() -> int:
        return represented["k"]

    def refresh(preferred_k: int) -> bool:
        """Score changed representatives (nearest K first); True if the backstop omitted any."""
        changed = []
        for k in sorted(archive_beams):
            representative = archive_beams[k][0]
            previous = representatives.get(k)
            if previous is None or previous.key != representative.key:
                changed.append(representative)
        changed.sort(key=lambda solution: (abs(solution.k - preferred_k), solution.k, solution.nll,
                                           solution.key))
        novel = []
        for solution in changed:
            cached = score_cache.get(solution.key)
            if cached is None:
                novel.append(solution)
            else:
                state["cache_hits"] += 1
                representatives[solution.k] = solution
                scores[solution.k] = cached
        room = max(0, search.exact_score_backstop(represented_k()) - state["exact_scores"])
        retained = novel[:room]
        for score in score_solutions(cavity_workspace, retained, active):
            score_cache[score.solution.key] = score
            representatives[score.k] = score.solution
            scores[score.k] = score
            if score.mean_field_failures:
                state["mean_field_failed"] = True
        state["exact_scores"] += len(retained)
        return len(novel) > room

    def selected():
        return min(scores.values(), key=_score_key)

    def beam(k: int):
        return list(archive_beams.get(k, ()))[:search.beam_width]

    def soft(solution) -> SoftBirths:
        if solution.k >= ceiling:
            return SoftBirths((), 0, 0)
        return soft_births(workspace, solution, likelihood, residual_inputs, config, soft_cache)

    add(data_solutions)
    if refresh(1):
        limits.append("exact_score_backstop")
    if state["mean_field_failed"]:
        limits.append("mean_field_nonconvergence")
    ordinary_expanded: set[bytes] = set()
    residual_expanded: set[bytes] = set()
    ordinary_complete: dict[bytes, bool] = {}
    residual_complete: dict[bytes, bool] = {}
    generated_offsets: dict[bytes, set] = {}
    no_improvement: set[bytes] = set()
    residual_calls = residual_rows = 0
    heuristic_cuts = False
    stop_reason = ""
    expansion = 0
    while True:
        if expansion >= search.expansion_backstop(represented_k()):
            limits.append("expansion_backstop")
            stop_reason = "expansion_backstop"
            break
        before = selected()
        candidates = beam(before.k)
        parent = next((solution for solution in candidates
                       if solution.key not in ordinary_expanded), None)
        phase = "ordinary"
        if parent is None:
            parent = next((solution for solution in candidates
                           if solution.key not in residual_expanded), None)
            phase = "residual"
        if parent is None:
            stop_reason = ("generated_neighbourhood_local_optimum" if not limits
                           else "incumbent_neighbourhood_search_limited")
            break
        local_limits: list[str] = []
        generated: list = []
        if phase == "ordinary":
            ordinary_expanded.add(parent.key)
            moves = ordinary_moves(workspace, parent, likelihood, growth, ceiling, cuts, config)
            if parent.k > 1 and cuts.heuristic(parent.k):
                heuristic_cuts = True
            generated.extend(moves.children)
            local_limits.extend(moves.limits)
            add(moves.children)
            if refresh(parent.k):
                local_limits.append("exact_score_backstop")
            if not selected().log_score > before.log_score + tolerance:
                children, omitted = replacements(workspace, parent, moves.births, config)
                generated.extend(children)
                if omitted:
                    local_limits.append("replacement_proposal_budget")
                add(children)
                if refresh(parent.k):
                    local_limits.append("exact_score_backstop")
            ordinary_complete[parent.key] = not local_limits
        else:
            residual_expanded.add(parent.key)
            births = soft(parent)
            residual_calls += 1
            residual_rows += births.emitted
            if births.omitted:
                local_limits.append("soft_residual_proposal_budget")
            generated.extend(births.children)
            add(births.children)
            # Refresh unconditionally: a retained soft proposal counts even if others were capped.
            if refresh(parent.k):
                local_limits.append("exact_score_backstop")
            if not selected().log_score > before.log_score + tolerance:
                children, omitted = replacements(workspace, parent, births.children, config)
                generated.extend(children)
                if omitted:
                    local_limits.append("replacement_proposal_budget")
                add(children)
                if refresh(parent.k):
                    local_limits.append("exact_score_backstop")
            if (selected().log_score <= before.log_score + tolerance
                    and parent.k + 2 <= ceiling and parent.k + 1 in representatives):
                bridge = representatives[parent.k + 1]
                paired = [child for child in data_births(workspace, bridge, growth, max_iterations)
                          if child.k == parent.k + 2]
                bridge_births = soft(bridge)
                residual_calls += 1
                residual_rows += bridge_births.emitted
                if bridge_births.omitted:
                    local_limits.append("soft_residual_proposal_budget")
                paired.extend(child for child in bridge_births.children
                              if child.k == parent.k + 2)
                paired = deduplicate(paired)
                generated.extend(paired)
                add(paired)
                if refresh(parent.k):
                    local_limits.append("exact_score_backstop")
            residual_complete[parent.key] = not local_limits
            after = selected()
            if after.solution.key == parent.key and after.log_score <= before.log_score + tolerance:
                no_improvement.add(parent.key)
        limits.extend(local_limits)
        if state["mean_field_failed"]:
            limits.append("mean_field_nonconvergence")
        generated_offsets.setdefault(parent.key, set()).update(
            child.k - parent.k for child in generated if child.k - parent.k in _MOVE_OFFSETS)
        expansion += 1

    if state["mean_field_failed"]:
        limits.append("mean_field_nonconvergence")
    unique_limits = tuple(dict.fromkeys(limits))
    best = selected()
    key = best.solution.key
    offsets = generated_offsets.get(key, set())
    certified = bool(
        key in ordinary_expanded and key in residual_expanded
        and ordinary_complete.get(key, False) and residual_complete.get(key, False)
        and all(best.k + offset in scores for offset in offsets)
        and key in no_improvement and not unique_limits and not state["mean_field_failed"])
    k_values = np.array(sorted(scores), dtype=np.int16)
    summary = SearchSummary(
        k_values=k_values,
        k_log_scores=np.array([scores[int(k)].log_score for k in k_values], dtype=np.float64),
        natural_k_ceiling=int(ceiling), expansions=int(expansion),
        data_starts=len(data_solutions), exact_scores=int(state["exact_scores"]),
        exact_score_cache_hits=int(state["cache_hits"]), residual_calls=int(residual_calls),
        residual_rows=int(residual_rows), limits=unique_limits,
        heuristic_gauge_cuts=bool(heuristic_cuts), stop_reason=stop_reason, certified=certified)
    return BlockSearch(selected=best.solution, summary=summary)
