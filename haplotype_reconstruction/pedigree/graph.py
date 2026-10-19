"""Acyclic pedigree selection from per-child local winners.

A graph constraint must not create parent evidence: a locally preferred configuration is kept
whenever it fits the DAG; otherwise the child takes the unique finite winner of its local state or a
lower one (M2 -> M1 -> M0), never a higher parent count. Children are visited by descending
orientation support of their local configuration, then confidence, then index; a coordinate local
search repeats the pass. Cycles among numerically tied local rows are left unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numba import njit

from ..maths import ties as tie_rules
from .orientation import Orientation, parent_role_probability
from .records import M0
from .ties import contrast_tolerance, unique_finite_winner, winner_tolerance


@dataclass(frozen=True)
class GraphSelection:
    """rows: child -> selected configuration row; direction_resolved: children whose local row was
    displaced by a non-M0 alternative; role_probability: child -> orientation support of its row."""

    rows: dict[int, int]
    direction_resolved: frozenset[int]
    role_probability: dict[int, float]


@njit(cache=True, nogil=True)
def _is_acyclic(selected_rows, rows):
    """Kahn's algorithm over the parent -> child edges of the selected rows (-1 = none)."""
    n_samples = selected_rows.shape[0]
    indegree = np.zeros(n_samples, dtype=np.int64)
    outdegree = np.zeros(n_samples, dtype=np.int64)
    n_edges = 0
    for child in range(n_samples):
        row = selected_rows[child]
        if row < 0:
            continue
        first = rows[row, 1]
        second = rows[row, 2]
        if first >= 0:
            indegree[child] += 1
            outdegree[first] += 1
            n_edges += 1
        if second >= 0 and second != first:
            indegree[child] += 1
            outdegree[second] += 1
            n_edges += 1
    offsets = np.empty(n_samples + 1, dtype=np.int64)
    offsets[0] = 0
    for node in range(n_samples):
        offsets[node + 1] = offsets[node] + outdegree[node]
    cursor = offsets[:-1].copy()
    outgoing = np.empty(n_edges, dtype=np.int64)
    for child in range(n_samples):
        row = selected_rows[child]
        if row < 0:
            continue
        first = rows[row, 1]
        second = rows[row, 2]
        if first >= 0:
            outgoing[cursor[first]] = child
            cursor[first] += 1
        if second >= 0 and second != first:
            outgoing[cursor[second]] = child
            cursor[second] += 1
    queue = np.empty(n_samples, dtype=np.int64)
    tail = 0
    for node in range(n_samples):
        if indegree[node] == 0:
            queue[tail] = node
            tail += 1
    head = 0
    visited = 0
    while head < tail:
        node = queue[head]
        head += 1
        visited += 1
        for edge in range(offsets[node], offsets[node + 1]):
            child = outgoing[edge]
            indegree[child] -= 1
            if indegree[child] == 0:
                queue[tail] = child
                tail += 1
    return visited == n_samples


def is_acyclic(local_rows: Mapping[int, int], rows: np.ndarray, n_samples: int) -> bool:
    """Whether the parent -> child edges of the selected rows form a DAG."""
    selected = np.full(n_samples, -1, dtype=np.int64)
    for child, row in local_rows.items():
        selected[child] = row
    return bool(_is_acyclic(selected, rows))


def _parents(rows: np.ndarray, row: int) -> tuple[int, ...]:
    return tuple(int(parent) for parent in rows[row, 1:] if parent >= 0)


def _confidence(child: int, state_margins: np.ndarray, identity_margins: np.ndarray) -> float:
    return min(value for value in (state_margins[child], identity_margins[child])
               if not np.isnan(value))


def _visit_order(local_rows, rows, state_margins, identity_margins, orientation):
    """Children by descending orientation support (exact: junction counts), then confidence in tie classes
    (release 2.1: confidences closer than the winner tolerance are tied; GL noise moves them), then index (the run's
    canonical sample order)."""
    role = {child: parent_role_probability(row, rows, orientation)
            for child, row in local_rows.items()}
    children = list(local_rows)
    if not children:
        return []
    confidence = np.asarray([_confidence(child, state_margins, identity_margins) for child in children])
    finite = confidence[np.isfinite(confidence)]
    quantum = winner_tolerance(float(np.max(np.abs(finite))) if finite.size else 1.0)
    classes = tie_rules.tie_classes(np.where(np.isfinite(confidence), confidence, -np.inf), max(quantum, 1e-300),
                                    descending=True).classes
    rank = {child: int(classes[index]) for index, child in enumerate(children)}
    # The index is the arbitrary last key (reversed under the convention reversal, release 2.2).
    return sorted(children, key=lambda child: (-role[child], rank[child], tie_rules.conv_key(child)))


def select_local(rows: np.ndarray, local_rows: Mapping[int, int], state_margins: np.ndarray,
                 identity_margins: np.ndarray, orientation: Orientation) -> GraphSelection:
    """The exact result when the local rows already form a DAG."""
    selected = {}
    role_probability = {}
    for child in _visit_order(local_rows, rows, state_margins, identity_margins, orientation):
        row = int(local_rows[child])
        selected[child] = row
        role_probability[child] = parent_role_probability(row, rows, orientation)
    return GraphSelection(selected, frozenset(), role_probability)


def select_acyclic(rows: np.ndarray, states: np.ndarray, decision_scores: np.ndarray,
                   child_rows: Sequence[np.ndarray], local_rows: Mapping[int, int],
                   state_margins: np.ndarray, identity_margins: np.ndarray, n_samples: int,
                   local_search_passes: int, orientation: Orientation) -> GraphSelection:
    """Greedy acyclic selection with downward fallback and coordinate local search."""
    order = _visit_order(local_rows, rows, state_margins, identity_margins, orientation)
    adjacency: list[set[int]] = [set() for _ in range(n_samples)]
    selected: dict[int, int] = {}
    direction_resolved: set[int] = set()
    role_probability: dict[int, float] = {}

    def path_exists(start: int, target: int) -> bool:
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

    def can_add(row: int) -> bool:
        child = int(rows[row, 0])
        return not any(path_exists(child, parent) for parent in _parents(rows, row))

    def add(row: int, displaced: bool) -> None:
        child = int(rows[row, 0])
        for parent in _parents(rows, row):
            adjacency[parent].add(child)
        selected[child] = row
        role_probability[child] = parent_role_probability(row, rows, orientation)
        if displaced and int(states[row]) != M0:
            direction_resolved.add(child)
        else:
            direction_resolved.discard(child)

    def remove(child: int) -> int | None:
        row = selected.pop(child, None)
        role_probability.pop(child, None)
        direction_resolved.discard(child)
        if row is not None:
            for parent in _parents(rows, row):
                adjacency[parent].discard(child)
        return row

    def best_feasible(child: int) -> tuple[int | None, bool]:
        """The local row if it fits, else the unique finite winner of the highest feasible state
        at or below the local state; (row, displaced)."""
        local = int(local_rows[child])
        if can_add(local):
            return local, False
        for target_state in range(int(states[local]), -1, -1):
            feasible = np.asarray([
                int(row) for row in child_rows[child]
                if int(states[row]) == target_state and np.isfinite(decision_scores[row])
                and can_add(int(row))], dtype=np.int64)
            winner, _ = unique_finite_winner(feasible, decision_scores)
            if winner is not None:
                return winner, winner != local
        return None, False

    for child in order:
        row, displaced = best_feasible(child)
        if row is not None:
            add(row, displaced)
    for _ in range(local_search_passes):
        changed = False
        for child in order:
            previous = remove(child)
            replacement, displaced = best_feasible(child)
            if replacement is not None:
                add(replacement, displaced)
            changed |= replacement != previous
        if not changed:
            break
    return GraphSelection(selected, frozenset(direction_resolved), role_probability)


def tie_conflict_children(rows: np.ndarray, local_rows: Mapping[int, int],
                          state_margins: np.ndarray, identity_margins: np.ndarray,
                          score_scales: np.ndarray, n_samples: int) -> frozenset[int]:
    """Children whose cyclic local rows are numerically tied at the weakest link.

    Cycles are peeled at their least-supported child (confidence: the smaller of its state and
    identity margins); a unique weakest row is left for the optimizer to displace, while two or
    more tied weakest rows are all marked unresolved (choosing one by sample order would
    manufacture confidence from a graph constraint).

    Confidences tie when they differ by at most 512 ULP of the largest magnitude among the
    implicated children's ``score_scales`` (float64 [N]: per child, the largest |score| its
    margins were computed from) and their confidences. A margin is a difference of composite log
    likelihoods of 1e4-1e5 nats, so its rounding is set by those scores, not by the margin's own
    size (measured: up to ~136 ULP of the scores when the chromosome sums are reassociated, so ~272
    for a difference of two margins): evaluation order must not decide a mathematically exact tie.
    (Base took 512 ULP of the margins, below their rounding resolution; T7 question 6.)
    """
    active = {int(child): int(row) for child, row in local_rows.items()}
    ambiguous: set[int] = set()

    def cyclic_components() -> list[frozenset[int]]:
        adjacency = [set() for _ in range(n_samples)]
        reverse = [set() for _ in range(n_samples)]
        for child, row in active.items():
            for parent in _parents(rows, row):
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
            implicated = [child for child, row in active.items()
                          if child in component
                          and any(parent in component for parent in _parents(rows, row))]
            if not implicated:
                continue
            values = np.asarray([_confidence(child, state_margins, identity_margins)
                                 for child in implicated], dtype=np.float64)
            minimum = float(np.min(values))
            if np.isfinite(minimum):
                finite = np.isfinite(values)
                scales = np.asarray(score_scales, dtype=np.float64)[implicated]
                tolerance = max(contrast_tolerance(np.r_[values[finite], scales[finite]]),
                                winner_tolerance(1.0))
                tied = [child for child, value in zip(implicated, values)
                        if np.isfinite(value) and abs(float(value) - minimum) <= tolerance]
            else:
                tied = [child for child, value in zip(implicated, values) if float(value) == minimum]
            if len(tied) > 1:
                ambiguous.update(tied)
                removed.update(tied)
            else:
                removed.add(tied[0])
        if not removed:
            raise RuntimeError("failed to peel a cyclic local pedigree")
        for child in removed:
            active.pop(child, None)
    return frozenset(ambiguous)
