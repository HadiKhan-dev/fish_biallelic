"""Carry already-floored pedigree evidence through the decision, guard and final release.

Owns composition of P0 PD01-PD24. Never scores genotypes, reads storage or renders
reports. The guard contracts its edge set before the count uses any direction.
Method: docs/methods/pedigree.md#products-and-published-tables.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

import numpy as np

from ...contracts.pedigree import (
    NO_PARENT,
    CountRelease,
    Decision,
    ExclusionEvidence,
    GuardRecord,
    ParentEdges,
    PartialEdges,
    Pedigree,
    PedigreeCalls,
    PedigreeInput,
    PedigreeReportInputs,
    PedigreeResult,
    PedigreeViews,
    TierCalls,
)
from ...core.errors import InvariantError
from . import slots, views
from .calibration import CalibrationExecutor
from .parameters import PedigreeConfig

__all__ = [
    "run",
    "release_supplied",
    "calls",
    "edges",
    "combine_exclusion",
    "released_order",
    "report_views",
    "complete_reports",
    "restore_result",
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


def combine_exclusion(evidence: Sequence[ExclusionEvidence]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average log exclusion endpoints in configured contig order; sum observed sites."""
    # NumPy 2.5's reduce stub omits the ordinary allocating out=None overload.
    lower = np.logaddexp.reduce(np.stack([value.log_e for value in evidence]), axis=0) - np.log(len(evidence))  # type: ignore[call-overload]
    upper = np.logaddexp.reduce(  # type: ignore[call-overload]
        np.stack([value.log_e if value.log_e_upper is None else value.log_e_upper for value in evidence]), axis=0
    )
    upper -= np.log(len(evidence))
    observed = np.sum([value.observed_sites for value in evidence], axis=0).astype(np.int64)
    return lower, upper, observed


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


def run(
    inputs: PedigreeInput, parameters: PedigreeConfig, executor: CalibrationExecutor | None = None
) -> PedigreeResult:
    """Execute the supplied-evidence pilot through the same pure request/release composition (serial fits)."""
    from .run import precheck, prepare_decision

    precheck(len(inputs.scores.sample_ids), len(inputs.contigs), parameters, basis="configured")
    return release_supplied(prepare_decision(inputs.scores, parameters, executor), inputs, parameters)


def release_supplied(decided: Decision, inputs: PedigreeInput, parameters: PedigreeConfig) -> PedigreeResult:
    """The supplied-evidence pilot after its decision: exclusion request, release and assembly."""
    from .run import assemble, exclusion_request, release_decision

    request = exclusion_request(decided, inputs.contigs, inputs.relatedness, parameters)
    released = release_decision(
        decided, request, inputs.exclusion, inputs.relatedness, inputs.observed_counts, parameters
    )
    return assemble(released, inputs.omitted, parameters)


def report_views(
    decision: Decision,
    count_release: CountRelease,
    orientation_ranks: np.ndarray,
    guard: GuardRecord,
    sample_ids: Sequence[str],
    contigs: Sequence[str],
    reports: PedigreeReportInputs,
    parameters: PedigreeConfig,
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


def complete_reports(
    result: PedigreeResult, reports: PedigreeReportInputs, parameters: PedigreeConfig
) -> PedigreeResult:
    """Attach every supplied report input and native view before complete publication.

    Requires one search row per sample and one summary per configured chromosome.
    Pure: returns a new product; no scoring, storage or guard refit occurs.
    """
    product = result.pedigree
    numeric = report_views(
        result.decision,
        result.count_release,
        result.orientation_ranks,
        product.sibling_guard,
        product.sample_ids,
        product.contigs,
        reports,
        parameters,
    )
    return replace(result, pedigree=replace(product, views=numeric, reports_complete=True))


def restore_result(product: Pedigree) -> PedigreeResult:
    """Reconstruct the whole internal result from publication, with no scientific computation."""
    retained = product.retained
    return PedigreeResult(
        product, retained.decision, retained.count_release, retained.orientation_ranks, product.exclusion
    )
