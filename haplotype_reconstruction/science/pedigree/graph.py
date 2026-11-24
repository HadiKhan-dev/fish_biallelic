"""The pedigree graph of one fit's choices: the acyclicity check, the cycle members the joint read restricts
(selection.select) and the reported selection of an acyclic set of rows.

Method: docs/methods/pedigree.md#joint-read.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ...contracts.pedigree import GraphSelection
from ...core import ties as tie_rules
from ...core.kernels import kernel
from .kernels.decision import OVERLOADS
from .numerical_parameters import WINNER_NUMERICS
from .orientation import Orientation, parent_role_probability
from .ties import winner_tolerance

__all__ = ["is_acyclic", "select_local", "cycle_members"]


@kernel(overloads=OVERLOADS["graph._is_acyclic"], cache=True, nogil=True)
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
    return min(value for value in (state_margins[child], identity_margins[child]) if not np.isnan(value))


def _visit_order(local_rows, rows, state_margins, identity_margins, orientation, *, reverse=False):
    """Children by descending orientation support (exact: junction counts), then confidence in tie classes
    (confidences closer than the winner tolerance are tied because GL noise moves them), then index (the run's
    canonical sample order)."""
    role = {child: parent_role_probability(row, rows, orientation) for child, row in local_rows.items()}
    children = list(local_rows)
    if not children:
        return []
    confidence = np.asarray([_confidence(child, state_margins, identity_margins) for child in children])
    finite = confidence[np.isfinite(confidence)]
    quantum = winner_tolerance(float(np.max(np.abs(finite))) if finite.size else 1.0)
    classes = tie_rules.tie_classes(
        np.where(np.isfinite(confidence), confidence, -np.inf),
        max(quantum, WINNER_NUMERICS.minimum_graph_quantum),
        descending=True,
    ).classes
    rank = {child: int(classes[index]) for index, child in enumerate(children)}
    # The index is the arbitrary last key (reversed under the convention reversal).
    return sorted(children, key=lambda child: (-role[child], rank[child], tie_rules.conv_key(child, reverse=reverse)))


def select_local(
    rows: np.ndarray,
    local_rows: Mapping[int, int],
    state_margins: np.ndarray,
    identity_margins: np.ndarray,
    orientation: Orientation,
    *,
    reverse=False,
) -> GraphSelection:
    """The exact result when the local rows already form a DAG."""
    selected = {}
    role_probability = {}
    for child in _visit_order(local_rows, rows, state_margins, identity_margins, orientation, reverse=reverse):
        row = int(local_rows[child])
        selected[child] = row
        role_probability[child] = parent_role_probability(row, rows, orientation)
    return GraphSelection(selected, frozenset(), role_probability)


def _cyclic_components(adjacency, reverse) -> list[frozenset[int]]:
    """Iterative Kosaraju, O(vertices + edges); components in minimum-vertex order.

    Traversal order cannot affect membership. The final vertex scan preserves the
    minimum-vertex peeling order without sorting or a Python recursion-depth limit.
    """
    n = len(adjacency)
    seen = [False] * n
    finished = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack = [(start, iter(adjacency[start]))]
        while stack:
            node, neighbours = stack[-1]
            neighbour = next(neighbours, None)
            if neighbour is None:
                finished.append(node)
                stack.pop()
            elif not seen[neighbour]:
                seen[neighbour] = True
                stack.append((neighbour, iter(adjacency[neighbour])))
    component_of = [-1] * n
    components: list[frozenset[int]] = []
    for start in reversed(finished):
        if component_of[start] >= 0:
            continue
        index = len(components)
        members = set()
        frontier = [start]
        component_of[start] = index
        while frontier:
            node = frontier.pop()
            members.add(node)
            for neighbour in reverse[node]:
                if component_of[neighbour] < 0:
                    component_of[neighbour] = index
                    frontier.append(neighbour)
        components.append(frozenset(members))
    emitted = set()
    result = []
    for node in range(n):
        index = component_of[node]
        if index not in emitted:
            emitted.add(index)
            if len(components[index]) > 1:
                result.append(components[index])
    return result


def cycle_members(selected_rows: Mapping[int, int], rows: np.ndarray, n_samples: int) -> frozenset[int]:
    """The children on a directed cycle of the selected rows' parent -> child edges (the strongly connected components
    with more than one sample; eligibility names no child as its own parent)."""
    adjacency: list[list[int]] = [[] for _ in range(n_samples)]
    reverse: list[list[int]] = [[] for _ in range(n_samples)]
    for child, row in selected_rows.items():
        for parent in _parents(rows, int(row)):
            adjacency[parent].append(int(child))
            reverse[int(child)].append(parent)
    return frozenset(member for component in _cyclic_components(adjacency, reverse) for member in component)
