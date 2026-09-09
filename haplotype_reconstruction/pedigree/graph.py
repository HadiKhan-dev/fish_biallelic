"""pedigree / graph for the canonical reconstruction pipeline."""
from __future__ import annotations


from typing import Mapping, Optional, Sequence


import numpy as np


def _path_exists(adjacency: list[set[int]], start: int, target: int) -> bool:
    if start == target:
        return True
    stack = [start]
    seen = {start}
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour == target:
                return True
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return False


def _observed_parents(
    alternatives: np.ndarray, row: int
) -> tuple[int, ...]:
    return tuple(
        int(parent) for parent in alternatives[row, 1:] if int(parent) >= 0
    )


def _acyclic_local_parent_state_selection(
    alternatives: np.ndarray,
    local_rows: Mapping[int, int],
    state_margins: np.ndarray,
    identity_margins: np.ndarray,
    depth_posterior: Optional[np.ndarray],
) -> pedigree_direction._GraphParentStateSelection:
    """Return the exact graph result when all local rows already form a DAG."""
    confidence = {}
    local_role_probability = {}
    for child, row in local_rows.items():
        margins = (state_margins[child], identity_margins[child])
        confidence[child] = min(
            value for value in margins if not np.isnan(value)
        )
        local_role_probability[child] = pedigree_direction._parent_role_probability(
            row, alternatives, depth_posterior
        )
    child_order = sorted(
        local_rows,
        key=lambda child: (
            -local_role_probability[child],
            -confidence[child],
            child,
        ),
    )
    selected = {}
    role_probabilities = {}
    for child in child_order:
        row = int(local_rows[child])
        selected[child] = row
        role_probabilities[child] = pedigree_direction._parent_role_probability(
            row, alternatives, depth_posterior
        )
    return pedigree_direction._GraphParentStateSelection(
        selected,
        frozenset(),
        role_probabilities,
    )


def _acyclic_parent_state_selection(
    alternatives: np.ndarray,
    states: np.ndarray,
    decision_scores: np.ndarray,
    by_child: Sequence[np.ndarray],
    local_rows: Mapping[int, int],
    state_margins: np.ndarray,
    identity_margins: np.ndarray,
    n_samples: int,
    local_search_passes: int,
    depth_posterior: Optional[np.ndarray] = None,
    downward_fallback: bool = False,
) -> pedigree_direction._GraphParentStateSelection:
    """Choose a DAG without converting graph feasibility into parent evidence.

    A unique locally preferred row is retained whenever it is acyclic. Legacy
    mode may substitute another same-state identity with depth support, then M0.
    Combined-v1 may instead search deterministic unique finite winners from the
    local state downward (M2 to M1 to M0, or M1 to M0); it never promotes a
    child to a higher parent-count state. Every non-M0 combined candidate has
    already passed the effective source-mode selection policy, including
    exposure, and the configured-or-explicit direction identity gate.
    """
    confidence = {}
    local_role_probability = {}
    for child, row in local_rows.items():
        margins = (state_margins[child], identity_margins[child])
        confidence[child] = min(
            value for value in margins if not np.isnan(value)
        )
        local_role_probability[child] = pedigree_direction._parent_role_probability(
            row, alternatives, depth_posterior
        )
    child_order = sorted(
        local_rows,
        key=lambda child: (
            -local_role_probability[child],
            -confidence[child],
            child,
        ),
    )
    adjacency = [set() for _ in range(n_samples)]
    selected: dict[int, int] = {}
    direction_resolved: set[int] = set()
    role_probabilities: dict[int, float] = {}

    def can_add(row: int) -> bool:
        child = int(alternatives[row, 0])
        return not any(
            _path_exists(adjacency, child, parent)
            for parent in _observed_parents(alternatives, row)
        )

    def add(row: int, displaced_local: bool = False) -> None:
        child = int(alternatives[row, 0])
        for parent in _observed_parents(alternatives, row):
            adjacency[parent].add(child)
        selected[child] = row
        role_probabilities[child] = pedigree_direction._parent_role_probability(
            row, alternatives, depth_posterior
        )
        if displaced_local and int(states[row]) != pedigree_models._ZERO_OBSERVED:
            direction_resolved.add(child)
        else:
            direction_resolved.discard(child)

    def remove(child: int) -> Optional[int]:
        row = selected.pop(child, None)
        role_probabilities.pop(child, None)
        direction_resolved.discard(child)
        if row is not None:
            for parent in _observed_parents(alternatives, row):
                adjacency[parent].discard(child)
        return row

    def best_feasible(child: int) -> tuple[Optional[int], bool]:
        local = local_rows.get(child)
        if local is not None and can_add(local):
            return local, False
        if local is not None and downward_fallback:
            local_state = int(states[local])
            for target_state in range(local_state, -1, -1):
                feasible = np.asarray([
                    int(row) for row in by_child[child]
                    if (
                        int(states[row]) == target_state
                        and np.isfinite(decision_scores[row])
                        and can_add(int(row))
                    )
                ], dtype=np.int64)
                winner, _ = pedigree_states._unique_finite_winner(
                    feasible, decision_scores
                )
                if winner is not None:
                    return winner, winner != local
            return None, False
        if local is not None and depth_posterior is not None:
            local_state = int(states[local])
            eligible = []
            for row in by_child[child]:
                row = int(row)
                if (
                    row != local
                    and int(states[row]) == local_state
                    and np.isfinite(decision_scores[row])
                    and can_add(row)
                ):
                    role_probability = pedigree_direction._parent_role_probability(
                        row, alternatives, depth_posterior
                    )
                    tolerance = pedigree_states._contrast_tolerance(np.asarray(
                        (role_probability, 0.5), dtype=np.float64
                    ))
                    if role_probability > 0.5 + tolerance:
                        eligible.append(row)
            if eligible:
                winner, _ = pedigree_states._unique_finite_winner(
                    np.asarray(eligible, dtype=np.int64), decision_scores
                )
                if winner is not None:
                    return winner, True
        zero_rows = [
            int(row)
            for row in by_child[child]
            if int(states[row]) == pedigree_models._ZERO_OBSERVED
        ]
        if len(zero_rows) != 1:
            raise pedigree_models.PedigreeEvidenceError(
                "each child must have exactly one zero-observed-parent row"
            )
        zero = zero_rows[0]
        return (
            (zero, local is not None and zero != local)
            if np.isfinite(decision_scores[zero])
            else (None, False)
        )

    for child in child_order:
        row, displaced = best_feasible(child)
        if row is not None:
            add(row, displaced)
    for _ in range(local_search_passes):
        changed = False
        for child in child_order:
            previous = remove(child)
            replacement, displaced = best_feasible(child)
            if replacement is not None:
                add(replacement, displaced)
            changed |= replacement != previous
        if not changed:
            break
    return pedigree_direction._GraphParentStateSelection(
        selected,
        frozenset(direction_resolved),
        role_probabilities,
    )


def _graph_tie_conflict_children(
    alternatives: np.ndarray,
    local_rows: Mapping[int, int],
    state_margins: np.ndarray,
    identity_margins: np.ndarray,
    n_samples: int,
) -> frozenset[int]:
    """Find tied local rows that a DAG cannot resolve without arbitrariness.

    Cycles are peeled at their least-supported child configuration.  A unique
    weakest row can be left for the normal DAG optimizer to displace.  If two
    or more weakest rows are numerically tied, all are marked ambiguous and
    excluded from graph selection; selecting one by sample-array order would
    manufacture exact confidence from a graph constraint.
    """
    active = {int(child): int(row) for child, row in local_rows.items()}
    ambiguous: set[int] = set()

    def confidence(child: int) -> float:
        values = (
            float(state_margins[child]),
            float(identity_margins[child]),
        )
        return min(value for value in values if not np.isnan(value))

    def cyclic_components() -> list[frozenset[int]]:
        adjacency = [set() for _ in range(n_samples)]
        reverse = [set() for _ in range(n_samples)]
        for child, row in active.items():
            for parent in _observed_parents(alternatives, row):
                adjacency[parent].add(child)
                reverse[child].add(parent)

        def reachable(graph: Sequence[set[int]], start: int) -> set[int]:
            seen = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                for neighbour in graph[node]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            return seen

        remaining = set(range(n_samples))
        components = []
        while remaining:
            start = min(remaining)
            component = reachable(adjacency, start) & reachable(reverse, start)
            remaining.difference_update(component)
            if len(component) > 1:
                components.append(frozenset(component))
        return components

    while True:
        components = cyclic_components()
        if not components:
            break
        removed = set()
        for component in components:
            implicated = [
                child
                for child, row in active.items()
                if child in component
                and any(
                    parent in component
                    for parent in _observed_parents(alternatives, row)
                )
            ]
            if not implicated:
                continue
            values = np.asarray(
                [confidence(child) for child in implicated], dtype=np.float64
            )
            minimum = float(np.min(values))
            if np.isfinite(minimum):
                finite = values[np.isfinite(values)]
                tolerance = pedigree_states._contrast_tolerance(finite)
                tied = [
                    child for child, value in zip(implicated, values)
                    if np.isfinite(value) and abs(float(value) - minimum) <= tolerance
                ]
            else:
                tied = [
                    child for child, value in zip(implicated, values)
                    if float(value) == minimum
                ]
            if len(tied) > 1:
                ambiguous.update(tied)
                removed.update(tied)
            else:
                removed.add(tied[0])
        if not removed:
            raise pedigree_models.PedigreeEvidenceError("failed to peel a cyclic local pedigree")
        for child in removed:
            active.pop(child, None)
    return frozenset(ambiguous)

import haplotype_reconstruction.pedigree.direction as pedigree_direction
import haplotype_reconstruction.pedigree.models as pedigree_models
import haplotype_reconstruction.pedigree.states as pedigree_states
