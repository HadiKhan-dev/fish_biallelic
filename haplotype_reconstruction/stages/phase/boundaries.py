"""Check canonical upstream axes and released pedigree before phase allocates work.

These adapter assertions never reorder samples, repair a pedigree or infer
biological identities from sample names.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...contracts.pedigree import ParentEdges
from ...core.errors import InvariantError

__all__ = ["validate_sample_order", "validate_parent_edges", "chromosome_prior_counts"]


def validate_sample_order(reference: Sequence[str], **sources: Sequence[str]) -> None:
    """Require exactly the evidence sample order in every independently read record."""
    expected = tuple(reference)
    if len(set(expected)) != len(expected):
        raise InvariantError("phase evidence has duplicate sample identities")
    for name, samples in sources.items():
        if tuple(samples) != expected:
            raise InvariantError(f"phase {name} sample order differs from evidence")


def validate_parent_edges(edges: ParentEdges) -> None:
    """Verify canonical Tier-B indices, slot uniqueness/order and acyclicity.

    pedigree releases a subset of its selected DAG. This consumer assertion rejects a
    corrupt or mismatched record before LabelModel traverses descendants.
    """
    n = len(edges.sample_ids)
    parents, children, slots = (np.asarray(a) for a in (edges.parents, edges.children, edges.slots))
    if parents.ndim != 1 or children.shape != parents.shape or slots.shape != parents.shape:
        raise InvariantError("phase ParentEdges arrays must be matching vectors")
    if any(a.dtype.kind not in "iu" for a in (parents, children, slots)):
        raise InvariantError("phase ParentEdges indices must be integers")
    if np.any((parents < 0) | (parents >= n) | (children < 0) | (children >= n)):
        raise InvariantError("phase ParentEdges sample index is outside the sample axis")
    if np.any((slots < 0) | (slots > 1)) or np.any(parents == children):
        raise InvariantError("phase ParentEdges has an invalid slot or self edge")
    keys = 2 * children + slots
    if np.any(np.diff(keys) <= 0):
        raise InvariantError("phase ParentEdges must have unique slots in child/slot order")
    indegree = np.bincount(children, minlength=n)
    outgoing: list[list[int]] = [[] for _ in range(n)]
    for parent, child in zip(parents, children, strict=True):
        outgoing[int(parent)].append(int(child))
    ready = [int(x) for x in np.flatnonzero(indegree == 0)]
    visited = 0
    while ready:
        parent = ready.pop()
        visited += 1
        for child in outgoing[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != n:
        raise InvariantError("phase ParentEdges is not a directed acyclic graph")


def chromosome_prior_counts(inference: Sequence[np.ndarray]) -> tuple[int, int]:
    """CAL2 prior projection over all components' called refit row-by-site cells.

    Returns exact integer (ones, called), never a component-local probability.
    Science applies p=.5 when called=0, clips to [.01,.99], then takes logit.
    """
    return (
        sum(int(np.count_nonzero(a == 1)) for a in inference),
        sum(int(np.count_nonzero(a >= 0)) for a in inference),
    )
