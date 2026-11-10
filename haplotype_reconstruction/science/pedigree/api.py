"""Materialize pedigree calls, edges and complete numeric report views.

Never scores genotypes, reads storage or renders reports. Report views apply
the released slot order and sibling guard.
Method: docs/methods/pedigree.md#products-and-published-tables.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ...contracts.pedigree import (
    NO_PARENT,
    CountRelease,
    Decision,
    GuardRecord,
    ParentEdges,
    PartialEdges,
    PedigreeCalls,
    PedigreeReportInputs,
    PedigreeViews,
    TierCalls,
)
from ...core.errors import InvariantError
from . import slots, views
from .parameters import PedigreeParameters

__all__ = [
    "calls",
    "edges",
    "released_order",
    "report_views",
]


def calls(tier: TierCalls, rows: np.ndarray, orientation: slots.ParentOrientation | None = None) -> PedigreeCalls:
    """Materialize int8[N] states and int64[N,2] parents, optionally ordering slots."""
    parents = np.full((len(tier.state), 2), NO_PARENT, dtype=np.int64)
    for child, row in enumerate(tier.row):
        if row >= 0:
            named = [int(parent) for parent in rows[row, 1:] if parent >= 0]
            parents[child, : len(named)] = named
    if orientation is not None:
        parents = orientation.orient(parents)[0]
    return PedigreeCalls(tier.state.copy(), parents, tier.status)


def edges(
    partial: PartialEdges, sample_ids: Sequence[str], orientation: slots.ParentOrientation | None = None
) -> ParentEdges:
    """Materialize supported edges ordered by child and then released slot."""
    children, slot_index = np.nonzero(partial.supported)
    parents = partial.parents[children, slot_index].astype(np.int64)
    slot_index = slot_index.astype(np.int64)
    if orientation is not None and len(children):
        slot_index = slots.orient_edges(parents, children, slot_index, orientation)
        order = np.lexsort((slot_index, children))
        children, parents, slot_index = children[order], parents[order], slot_index[order]
    return ParentEdges(
        tuple(sample_ids),
        parents,
        children.astype(np.int64),
        slot_index,
        partial.count_minimum.copy(),
        partial.count_maximum.copy(),
    )


def released_order(
    observed_counts: np.ndarray,
    sample_ids: Sequence[str],
    parent_arrays: Iterable[np.ndarray],
    *,
    reverse: bool = False,
) -> slots.ParentOrientation:
    """Order released parent pairs by observed counts, with name as the final convention."""
    pairs = set()
    for parents in parent_arrays:
        for first, second in np.asarray(parents, dtype=np.int64).tolist():
            if first >= 0 and second >= 0:
                pairs.add((first, second))
    ranks = (
        slots.canonical_ranks(observed_counts, sample_ids, pairs)
        if pairs
        else np.arange(len(sample_ids), dtype=np.int64)
    )
    return slots.ParentOrientation(ranks, reverse=reverse)


def report_views(
    decision: Decision,
    count_release: CountRelease,
    orientation_ranks: np.ndarray,
    guard: GuardRecord,
    sample_ids: Sequence[str],
    contigs: Sequence[str],
    reports: PedigreeReportInputs,
    parameters: PedigreeParameters,
) -> PedigreeViews:
    """The report-complete numeric views: supplied report inputs, released slot order, then the guard.

    Requires one search row per sample and one summary per configured chromosome.
    """
    if tuple(r.sample for r in reports.search) != tuple(range(len(sample_ids))):
        raise InvariantError("search diagnostics must have every sample in canonical order")
    if tuple(r.contig for r in reports.chromosomes) != tuple(contigs):
        raise InvariantError("chromosome summary must have every configured contig in order")
    numeric = views.build(
        decision,
        count_release,
        reports.search,
        reports.chromosomes,
        parameters,
        relative_contrasts=reports.relatives,
    )
    numeric = views.orient(numeric, slots.ParentOrientation(orientation_ranks, reverse=parameters.reverse_conventions))
    return views.apply_guard(numeric, guard)
