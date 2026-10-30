"""Release rules: Tier A / Tier B configuration calls, marginal parent edges and candidate sets.

A tier releases a child's marginal parent state when its informative chromosomes hold enough map units, the state
survives the bootstrap and leave-one-chromosome-out (LOCO) cut-offs and, for M0, every alternative
prior; it releases the exact configuration when the local configuration is also the graph's choice,
is resampling-stable as local and graph configuration, each named parent is stable, and every
inferred (non-chronology) edge has a decisive full-data orientation. Marginal ("partial") edges are
released one by one from the same acyclic solution when the exact configuration is not. Candidate
sets report the identities carrying most of the evidence when identity stays unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..maths import ties as tie_rules
from .alternatives import Alternatives, softmax_finite
from .config import PedigreeConfig, TierThresholds
from .eligibility import ParentEligibility
from .records import M0, M1, M2, NO_PARENT, STATE_NAMES, UNRESOLVED
from .resampling import Resampling
from .selection import Selection


@dataclass(frozen=True)
class ChildEvidence:
    """Per-child quantities the release rules read (N samples; NaN where undefined).

    local_parents / graph_parents: int64 [N, 2] parents of the local / graph row (NO_PARENT).
    graph_state: int8 [N]; graph_conflict: bool [N] (tie conflict or graph row != local row).
    informative_units: float64 [N] map units of the chromosomes whose own evidence picks a unique state.
    state_bootstrap / state_loco: support of the local state.
    configuration_*: support of the local row as local / graph configuration.
    parent_bootstrap / parent_loco: float64 [N, 2] local-parent support per slot.
    graph_parent_bootstrap / graph_parent_loco: float64 [N, 2] graph-parent support per slot.
    graph_configuration_bootstrap / _loco: support of the graph row as graph configuration.
    orientation_resolved: bool [N, 2] per local parent slot (True for empty slots / chronology).
    m0_state_agreement / m0_graph_agreement / m0_veto: bool [N] prior-sensitivity checks of M0.
    """

    local_parents: np.ndarray
    graph_parents: np.ndarray
    graph_state: np.ndarray
    graph_conflict: np.ndarray
    informative_units: np.ndarray
    state_bootstrap: np.ndarray
    state_loco: np.ndarray
    configuration_bootstrap: np.ndarray
    configuration_loco: np.ndarray
    configuration_graph_bootstrap: np.ndarray
    configuration_graph_loco: np.ndarray
    parent_bootstrap: np.ndarray
    parent_loco: np.ndarray
    graph_parent_bootstrap: np.ndarray
    graph_parent_loco: np.ndarray
    graph_configuration_bootstrap: np.ndarray
    graph_configuration_loco: np.ndarray
    orientation_resolved: np.ndarray
    m0_state_agreement: np.ndarray
    m0_graph_agreement: np.ndarray
    m0_veto: np.ndarray


@dataclass(frozen=True)
class TierCalls:
    """Exact configuration view of one tier (or the complete DAG view).

    state int8 [N] (UNRESOLVED), row int64 [N] (-1 = identity not released), status tuple[str];
    for tiers also state_pass / identity_pass / orientation_withheld bool [N] and parent_flags
    bool [N, 2] (per local parent slot: resampling-stable and oriented).
    """

    state: np.ndarray
    row: np.ndarray
    status: tuple[str, ...]
    state_pass: np.ndarray | None = None
    identity_pass: np.ndarray | None = None
    orientation_withheld: np.ndarray | None = None
    parent_flags: np.ndarray | None = None


@dataclass(frozen=True)
class PartialEdges:
    """Edge-level release of one tier.

    parents int64 [N, 2] released or exact-configuration parents per slot (NO_PARENT otherwise);
    supported bool [N, 2]; exact bool [N] (exact configuration released); count_minimum /
    count_maximum int64 [N] observed-parent count bounds; state int8 [N] (from the exact view);
    status tuple[str]; release_evidence tuple[str] ("chromosome_resampling" or, after the
    full-marker release, "mendelian_exclusion_and_supported_direction").
    """

    parents: np.ndarray
    supported: np.ndarray
    exact: np.ndarray
    count_minimum: np.ndarray
    count_maximum: np.ndarray
    state: np.ndarray
    status: tuple[str, ...]
    release_evidence: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSets:
    """Tier-B candidate sets: parents [N, 2] (supported singletons), side candidates and the
    configuration candidates (tuples of sample indices), exact flags and statuses."""

    parents: np.ndarray
    side_candidates: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    configurations: tuple[tuple[tuple[int, ...], ...], ...]
    exact: np.ndarray
    status: tuple[str, ...]


def _row_parents(rows: np.ndarray, row: int) -> np.ndarray:
    result = np.full(2, NO_PARENT, dtype=np.int64)
    if row >= 0:
        parents = [int(parent) for parent in rows[row, 1:] if parent >= 0]
        result[:len(parents)] = parents
    return result


def orientation_cutoff(config: PedigreeConfig) -> float:
    """Minimum full-data orientation log odds of a released inferred edge."""
    probability = config.release.orientation_reversal_probability
    return float(np.log1p(-probability) - np.log(probability))


def child_evidence(full: Selection, sensitivity: list[tuple[tuple, Selection]],
                   resampling: Resampling, alternatives: Alternatives, informative: np.ndarray,
                   orientation_log_odds: np.ndarray, chronology: np.ndarray,
                   config: PedigreeConfig) -> ChildEvidence:
    """Collect the per-child support quantities of the full fit and the resampled fits."""
    rows, states = alternatives.rows, alternatives.states
    n = alternatives.n_samples
    boot, loco = resampling.bootstrap, resampling.loco
    cutoff = orientation_cutoff(config)
    local_parents = np.full((n, 2), NO_PARENT, dtype=np.int64)
    graph_parents = np.full((n, 2), NO_PARENT, dtype=np.int64)
    graph_state = np.full(n, UNRESOLVED, dtype=np.int8)
    graph_conflict = np.zeros(n, dtype=np.bool_)
    values = {name: np.full(n, np.nan) for name in (
        "state_bootstrap", "state_loco", "configuration_bootstrap", "configuration_loco",
        "configuration_graph_bootstrap", "configuration_graph_loco",
        "graph_configuration_bootstrap", "graph_configuration_loco")}
    parent_bootstrap, parent_loco, graph_parent_bootstrap, graph_parent_loco = (
        np.full((n, 2), np.nan) for _ in range(4))
    oriented = np.ones((n, 2), dtype=np.bool_)
    m0_state = np.zeros(n, dtype=np.bool_)
    m0_graph = np.zeros(n, dtype=np.bool_)
    for child in range(n):
        state, row, graph_row = int(full.local_state[child]), int(full.local_row[child]), \
            int(full.graph_row[child])
        local_parents[child] = _row_parents(rows, row)
        graph_parents[child] = _row_parents(rows, graph_row)
        if graph_row >= 0:
            graph_state[child] = states[graph_row]
        graph_conflict[child] = bool(full.tie_conflict[child] or (row >= 0 and graph_row != row))
        if state != UNRESOLVED:
            values["state_bootstrap"][child] = boot.fraction(boot.local_state[child, state])
            values["state_loco"][child] = loco.fraction(loco.local_state[child, state])
        if row >= 0:
            values["configuration_bootstrap"][child] = boot.fraction(boot.local_configuration[row])
            values["configuration_graph_bootstrap"][child] = boot.fraction(
                boot.graph_configuration[row])
            values["configuration_loco"][child] = loco.fraction(loco.local_configuration[row])
            values["configuration_graph_loco"][child] = loco.fraction(loco.graph_configuration[row])
        for slot, parent in enumerate(local_parents[child]):
            if parent < 0:
                continue
            parent_bootstrap[child, slot] = boot.fraction(boot.local_parent[child, parent])
            parent_loco[child, slot] = loco.fraction(loco.local_parent[child, parent])
            oriented[child, slot] = bool(chronology[child, parent]
                                         or orientation_log_odds[child, parent] >= cutoff)
        if graph_row >= 0:
            values["graph_configuration_bootstrap"][child] = boot.fraction(
                boot.graph_configuration[graph_row])
            values["graph_configuration_loco"][child] = loco.fraction(
                loco.graph_configuration[graph_row])
        for slot, parent in enumerate(graph_parents[child]):
            if parent >= 0:
                graph_parent_bootstrap[child, slot] = boot.fraction(boot.graph_parent[child, parent])
                graph_parent_loco[child, slot] = loco.fraction(loco.graph_parent[child, parent])
        m0_state[child] = state != M0 or all(
            selection.local_state[child] == M0 for _, selection in sensitivity)
        m0_graph[child] = state != M0 or all(
            selection.graph_row[child] >= 0 and states[selection.graph_row[child]] == M0
            for _, selection in sensitivity)
    m0_veto = (full.local_state == M0) & ~m0_state
    return ChildEvidence(local_parents, graph_parents, graph_state, graph_conflict, informative,
                         parent_bootstrap=parent_bootstrap, parent_loco=parent_loco,
                         graph_parent_bootstrap=graph_parent_bootstrap,
                         graph_parent_loco=graph_parent_loco, orientation_resolved=oriented,
                         m0_state_agreement=m0_state, m0_graph_agreement=m0_graph,
                         m0_veto=m0_veto, **values)


def tier_calls(label: str, thresholds: TierThresholds, full: Selection, evidence: ChildEvidence,
               eligibility: ParentEligibility, config: PedigreeConfig) -> TierCalls:
    """Exact configuration calls of one tier ("tier_a" or "tier_b")."""
    n = len(full.local_state)
    state_out = np.full(n, UNRESOLVED, dtype=np.int8)
    row_out = np.full(n, -1, dtype=np.int64)
    status: list[str] = []
    state_pass = np.zeros(n, dtype=np.bool_)
    identity_pass = np.zeros(n, dtype=np.bool_)
    withheld = np.zeros(n, dtype=np.bool_)
    flags = np.zeros((n, 2), dtype=np.bool_)
    minimum_units = config.release.minimum_informative_units
    for child in range(n):
        state, row = int(full.local_state[child]), int(full.local_row[child])
        passed = bool(state != UNRESOLVED
                      and evidence.informative_units[child] >= minimum_units
                      and evidence.state_bootstrap[child] >= thresholds.configuration_bootstrap
                      and evidence.state_loco[child] >= thresholds.loco_fraction
                      and evidence.m0_state_agreement[child])
        named = [slot for slot in range(2) if evidence.local_parents[child, slot] >= 0]
        stable = [bool(passed and evidence.parent_bootstrap[child, slot] >= thresholds.parent_bootstrap
                       and evidence.parent_loco[child, slot] >= thresholds.loco_fraction)
                  for slot in named]
        oriented = [stable[index] and bool(evidence.orientation_resolved[child, slot])
                    for index, slot in enumerate(named)]
        configuration = bool(
            passed and row >= 0 and int(full.graph_row[child]) == row
            and not full.tie_conflict[child]
            and evidence.configuration_bootstrap[child] >= thresholds.configuration_bootstrap
            and evidence.configuration_graph_bootstrap[child] >= thresholds.configuration_bootstrap
            and evidence.configuration_loco[child] >= thresholds.loco_fraction
            and evidence.configuration_graph_loco[child] >= thresholds.loco_fraction
            and evidence.m0_graph_agreement[child])
        exact = bool(configuration and all(oriented))
        state_pass[child], identity_pass[child] = passed, exact
        withheld[child] = bool(configuration and all(stable) and not exact)
        for index, slot in enumerate(named):
            flags[child, slot] = oriented[index]
        if not eligibility.children[child]:
            status.append("excluded_by_parent_eligibility")
        elif not passed:
            status.append(f"unresolved_{label}_m0_prior_sensitivity" if evidence.m0_veto[child]
                          else f"unresolved_below_{label}_state_support")
        else:
            state_out[child] = state
            if exact:
                row_out[child] = row
                status.append(f"{label}_supported_{STATE_NAMES[state]}")
            else:
                suffix = ("graph_tie_conflict" if full.tie_conflict[child] else
                          "graph_conflict" if evidence.graph_conflict[child] else
                          "orientation_unresolved" if withheld[child] else "identity_unresolved")
                status.append(f"{label}_parent_state_supported_{suffix}")
    return TierCalls(state_out, row_out, tuple(status), state_pass, identity_pass, withheld, flags)


def complete_calls(full: Selection, eligibility: ParentEligibility,
                   alternatives: Alternatives) -> TierCalls:
    """The full-data acyclic view (not a release)."""
    n = len(full.local_state)
    state_out = np.full(n, UNRESOLVED, dtype=np.int8)
    row_out = np.full(n, -1, dtype=np.int64)
    status: list[str] = []
    for child in range(n):
        local_state, local_row = int(full.local_state[child]), int(full.local_row[child])
        graph_row = int(full.graph_row[child])
        if not eligibility.children[child]:
            status.append("excluded_by_parent_eligibility")
        elif full.tie_conflict[child]:
            state_out[child] = local_state
            status.append("graph_tie_conflict_parent_identity_unresolved")
        elif graph_row >= 0:
            row_out[child] = graph_row
            graph_state = int(alternatives.states[graph_row])
            state_out[child] = graph_state
            status.append(f"selected_{STATE_NAMES[graph_state]}" if graph_row == local_row else
                          "graph_displaced_direction_resolved_same_state_hypothesis_not_tier_eligible"
                          if full.direction_resolved[child] else
                          "graph_displaced_hypothesis_not_tier_eligible")
        elif local_state != UNRESOLVED:
            state_out[child] = local_state
            status.append(full.unresolved_reason[child] or "parent_state_resolved_identity_unresolved")
        else:
            status.append(full.unresolved_reason[child] or "unresolved_parent_state")
    return TierCalls(state_out, row_out, tuple(status))


def partial_edges(label: str, thresholds: TierThresholds, calls: TierCalls,
                  evidence: ChildEvidence, alternatives: Alternatives,
                  config: PedigreeConfig) -> PartialEdges:
    """Release stable parent edges of the full-data DAG one by one where the configuration is not.

    Edges come only from children whose local configuration is the graph's (no conflict) and
    with enough informative chromosomes; a stable edge with an unresolved orientation is withheld.
    """
    n = len(calls.state)
    parents = np.full((n, 2), NO_PARENT, dtype=np.int64)
    for child in range(n):
        parents[child] = _row_parents(alternatives.rows, int(calls.row[child]))
    supported = parents >= 0
    exact = np.asarray([status.startswith(f"{label}_supported_") for status in calls.status])
    known = calls.state != UNRESOLVED
    count_minimum = np.where(known, calls.state, 0).astype(np.int64)
    count_maximum = np.where(known, calls.state, 2).astype(np.int64)
    status = list(calls.status)
    for child in range(n):
        if exact[child] or evidence.graph_conflict[child] or (
                evidence.informative_units[child] < config.release.minimum_informative_units):
            continue
        retained = withheld = 0
        for slot in range(2):
            parent = int(evidence.local_parents[child, slot])
            if parent < 0 or parent != int(evidence.graph_parents[child, slot]):
                continue
            support = np.asarray([evidence.parent_bootstrap[child, slot],
                                  evidence.parent_loco[child, slot],
                                  evidence.graph_parent_bootstrap[child, slot],
                                  evidence.graph_parent_loco[child, slot]])
            cutoffs = (thresholds.parent_bootstrap, thresholds.loco_fraction,
                       thresholds.parent_bootstrap, thresholds.loco_fraction)
            if np.all(np.isfinite(support)) and np.all(support >= cutoffs):
                if not evidence.orientation_resolved[child, slot]:
                    withheld += 1
                    continue
                parents[child, slot] = parent
                supported[child, slot] = True
                retained += 1
        if retained:
            count_minimum[child] = max(int(count_minimum[child]), retained)
            status[child] = f"{label}_marginal_parent_support"
        elif withheld:
            status[child] = f"{label}_marginal_parent_orientation_unresolved"
    return PartialEdges(parents, supported, exact, count_minimum, count_maximum,
                        calls.state.copy(), tuple(status), ("chromosome_resampling",) * n)


def coverage_rows(rows: np.ndarray, scores: np.ndarray, coverage: float) -> tuple[int, ...]:
    """Fewest rows, by descending softmax probability (ties by row), carrying ``coverage`` mass.

    Probabilities in one tie class (probability quantum, ``maths.ties``) are tied: they are ordered by row and a
    cut inside a class keeps the whole class (release 2.1; GL noise reorders near-equal probabilities)."""
    candidate_rows = np.asarray(rows, dtype=np.int64)
    probabilities = softmax_finite(scores[candidate_rows])
    positive = probabilities > 0.0
    candidate_rows = candidate_rows[positive]
    probabilities = probabilities[positive]
    if not len(candidate_rows):
        return ()
    ranked = tie_rules.tie_classes(probabilities, tie_rules.tie_quantum(0.0, "probability"),
                                   content=[int(row) for row in candidate_rows], descending=True)
    selected = []
    cumulative = 0.0
    cut_class = None
    for position in ranked.order:
        if cut_class is not None and ranked.classes[position] != cut_class:
            break
        selected.append(int(candidate_rows[position]))
        cumulative += float(probabilities[position])
        if cut_class is None and cumulative >= coverage:
            cut_class = ranked.classes[position]
    return tuple(selected)


def support_sides(child_rows: np.ndarray, state: int, row: int, alternatives: Alternatives,
                  decision_scores: np.ndarray, coverage: float,
                  ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """(Parent1 side, Parent2 side, configuration rows) of the evidence set of an M1/M2 state.

    M1: the parents of the covering rows. M2 (parents unordered): one row gives its two sides; if
    every covering pair shares one parent, that parent is one side (Parent1 when it is the local
    row's first parent) and the other parents form the other side; otherwise no factorisation
    (CLASS-B note: base documented conditional sides here but returned none).
    """
    if state not in (M1, M2):
        return (), (), ()
    states = alternatives.states
    support = coverage_rows(child_rows[states[child_rows] == state], decision_scores, coverage)
    if not support:
        return (), (), ()
    rows = alternatives.rows
    if state == M1:
        return tuple(dict.fromkeys(int(rows[r, 1]) for r in support)), (), support
    pairs = [frozenset((int(rows[r, 1]), int(rows[r, 2]))) for r in support]
    if len(pairs) == 1:
        return (int(rows[support[0], 1]),), (int(rows[support[0], 2]),), support
    common = set(pairs[0])
    for pair in pairs[1:]:
        common.intersection_update(pair)
    if len(common) == 1:
        fixed = next(iter(common))
        variable = tuple(sorted({parent for pair in pairs for parent in pair if parent != fixed}))
        if row >= 0 and int(rows[row, 1]) == fixed:
            return (fixed,), variable, support
        return variable, (fixed,), support
    return (), (), support


def candidate_sets(calls: TierCalls, partial: PartialEdges, full: Selection,
                   alternatives: Alternatives, resampling: Resampling,
                   orientation_log_odds: np.ndarray, chronology: np.ndarray,
                   config: PedigreeConfig) -> CandidateSets:
    """Tier-B candidate sets: supported singleton sides, side candidates, configuration sets."""
    thresholds = config.release.tier_b
    cutoff = orientation_cutoff(config)
    boot, loco = resampling.bootstrap, resampling.loco
    n = len(calls.state)
    parents = np.full((n, 2), NO_PARENT, dtype=np.int64)
    sides, configurations, statuses = [], [], []
    exact = calls.row >= 0
    for child in range(n):
        state = int(calls.state[child])
        if state == UNRESOLVED:
            first, second, configuration_rows = (), (), ()
        else:
            first, second, configuration_rows = support_sides(
                alternatives.child_rows[child], int(full.local_state[child]),
                int(full.local_row[child]), alternatives, full.decision_scores,
                config.release.support_set_coverage)
        for slot, side in enumerate((first, second)):
            if len(side) != 1:
                continue
            parent = int(side[0])
            if (boot.fraction(boot.local_parent[child, parent]) >= thresholds.parent_bootstrap
                    and loco.fraction(loco.local_parent[child, parent]) >= thresholds.loco_fraction
                    and (chronology[child, parent]
                         or orientation_log_odds[child, parent] >= cutoff)):
                parents[child, slot] = parent
        if state == UNRESOLVED:
            status = "unresolved_below_tier_b_state_support"
        elif exact[child]:
            status = "tier_b_exact_configuration"
        elif np.any(parents[child] >= 0):
            status = "tier_b_partial_parent_with_candidate_set"
        elif first or second or configuration_rows:
            status = "tier_b_candidate_set_identity_unresolved"
        else:
            status = "tier_b_parent_state_only"
        configurations.append(tuple(
            tuple(int(parent) for parent in alternatives.rows[r, 1:] if parent >= 0)
            for r in configuration_rows))
        sides.append((tuple(first), tuple(second)))
        # Retained marginal edges replace the sides (no complete configuration is invented).
        if not partial.exact[child] and np.any(partial.supported[child]):
            for slot in range(2):
                if partial.supported[child, slot]:
                    parents[child, slot] = partial.parents[child, slot]
                    sides[-1] = tuple((int(partial.parents[child, s]),)
                                      if s == slot else sides[-1][s] for s in range(2))
            status = partial.status[child]
        statuses.append(status)
    return CandidateSets(parents, tuple(sides), tuple(configurations), exact, tuple(statuses))
