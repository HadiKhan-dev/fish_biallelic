"""Orientation evidence from the joint transmission likelihood of painted homologue schedules.

For each chromosome compare the ordered dyads' joint log likelihoods
(``transmission.log_likelihood``), allowing painting failure through
``chromosome_error_probability``. Two impossible directions are uninformative.
Log odds add over chromosomes, weighted by resampling multiplicity.
A paired junction-density contrast only orders the acyclic graph search and is reported.

Method: docs/methods/pedigree.md#finite-continuous-direction
"""

from __future__ import annotations

import math

import numpy as np

from ...contracts.pedigree import JunctionEvidence, Orientation
from ...core.kernels import kernel
from .kernels.signatures import OVERLOADS
from .parameters import ORDER_MINIMUM_WEIGHT, ORDER_VARIANCE_OFFSET

__all__ = ["junction_evidence", "fit", "parent_role_probability"]


@kernel(overloads=OVERLOADS["orientation._order_probability"], cache=True, nogil=True)
def _order_probability(junctions, callable_bins, total_bins, multiplicity):
    """Paired chromosome contrast softened by missingness and count variance.

    A summary-level MAR approximation, not a common-interval recount. Inclusion-exclusion bounds
    the common exposure; absent overlap is neutral.
    """
    chromosomes, samples = junctions.shape
    result = np.full((samples, samples), 0.5)
    for child in range(samples):
        for parent in range(child):
            count = mean_sum = square_sum = missing_variance = 0.0
            for chrom in range(chromosomes):
                if multiplicity[chrom] <= 0 or total_bins[chrom] <= 0:
                    continue
                fc = min(1.0, callable_bins[chrom, child] / (2.0 * total_bins[chrom]))
                fp = min(1.0, callable_bins[chrom, parent] / (2.0 * total_bins[chrom]))
                common = max(0.0, fc + fp - 1.0)
                if common <= 0:
                    continue
                weight = multiplicity[chrom] * common
                dc = junctions[chrom, child] / fc
                dp = junctions[chrom, parent] / fp
                delta = dc - dp
                count += weight
                mean_sum += weight * delta
                square_sum += weight * delta * delta
                missing_variance += weight * ((1.0 - fc) * dc / fc + (1.0 - fp) * dp / fp)
            if count <= ORDER_MINIMUM_WEIGHT:
                continue
            variance = max(0.0, square_sum - mean_sum * mean_sum / count)
            variance *= count / (count - 1.0)
            variance += missing_variance + ORDER_VARIANCE_OFFSET
            probability = 0.5 * (1.0 + math.erf(mean_sum / math.sqrt(2.0 * variance)))
            result[child, parent] = probability
            result[parent, child] = 1.0 - probability
    return result


@kernel(overloads=OVERLOADS["orientation._pair_log_odds"], cache=True, nogil=True)
def _pair_log_odds(
    log_likelihood, forward_index, reverse_index, pair_child, pair_parent, callable_bins, total_bins, error_probability
):
    """values[chrom, u] for the unordered dyads u (pair_parent -> pair_child against the reverse) where
    ``compared``; log_likelihood[chrom, d] over the ordered dyads (pedigree.transmission.log_likelihood)."""
    chromosomes = log_likelihood.shape[0]
    pairs = len(pair_child)
    values = np.zeros((chromosomes, pairs))
    compared = np.zeros((chromosomes, pairs), dtype=np.bool_)
    keep = 1.0 - error_probability
    floor = 0.5 * error_probability
    for chrom in range(chromosomes):
        if total_bins[chrom] <= 0:
            continue
        for u in range(pairs):
            child = pair_child[u]
            parent = pair_parent[u]
            fc = min(1.0, callable_bins[chrom, child] / (2.0 * total_bins[chrom]))
            fp = min(1.0, callable_bins[chrom, parent] / (2.0 * total_bins[chrom]))
            if fc <= 0.0 or fp <= 0.0:
                continue
            forward = log_likelihood[chrom, forward_index[u]]
            reverse = log_likelihood[chrom, reverse_index[u]]
            if forward == -np.inf and reverse == -np.inf:
                continue  # impossible both ways: uninformative
            difference = forward - reverse
            if difference == 0.0:
                continue
            scaled = math.exp(-abs(difference))
            larger = keep / (1.0 + scaled) + floor
            smaller = keep * scaled / (1.0 + scaled) + floor
            value = math.log(larger) - math.log(smaller)
            if difference < 0.0:
                value = -value
            values[chrom, u] = value
            compared[chrom, u] = True
    return values, compared


@kernel(overloads=OVERLOADS["orientation._weighted_log_odds"], cache=True, nogil=True)
def _weighted_log_odds(values, compared, pair_child, pair_parent, total_bins, multiplicity, samples):
    """Chromosome-weighted sum of the pair log odds, ``result[child, parent]``, in chromosome order."""
    chromosomes = values.shape[0]
    result = np.zeros((samples, samples))
    for chrom in range(chromosomes):
        if multiplicity[chrom] <= 0 or total_bins[chrom] <= 0:
            continue
        for u in range(len(pair_child)):
            if compared[chrom, u]:
                value = values[chrom, u]
                result[pair_child[u], pair_parent[u]] += multiplicity[chrom] * value
                result[pair_parent[u], pair_child[u]] -= multiplicity[chrom] * value
    return result


def _unordered_dyads(dyads: np.ndarray, samples: int):
    """(child, parent, forward index, reverse index) of the unordered dyads whose two directions are both in
    the ordered ``dyads`` [D, 2], child > parent, in row-major (child, parent) order."""
    dyads = np.asarray(dyads, dtype=np.int64).reshape(-1, 2)
    keys = dyads[:, 0] * samples + dyads[:, 1]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    forward = np.flatnonzero(dyads[:, 0] > dyads[:, 1])
    reverse_keys = dyads[forward, 1] * samples + dyads[forward, 0]
    where = np.searchsorted(sorted_keys, reverse_keys)
    found = (where < len(sorted_keys)) & (sorted_keys[np.minimum(where, len(sorted_keys) - 1)] == reverse_keys)
    forward = forward[found]
    reverse = order[where[found]]
    rank = np.lexsort((dyads[forward, 1], dyads[forward, 0]))
    forward, reverse = forward[rank], reverse[rank]
    return (
        np.ascontiguousarray(dyads[forward, 0]),
        np.ascontiguousarray(dyads[forward, 1]),
        np.ascontiguousarray(forward),
        np.ascontiguousarray(reverse),
    )


def junction_evidence(
    junctions: np.ndarray,
    callable_bins: np.ndarray,
    total_bins: np.ndarray,
    error_probability: float,
    dyads: np.ndarray,
    log_likelihood: np.ndarray,
) -> JunctionEvidence:
    """Weight-independent junction evidence of the informative chromosomes (``dyads`` [D, 2] and ``log_likelihood``
    [C, D]: the joint transmission log likelihoods of the scores)."""
    junctions = np.ascontiguousarray(junctions, dtype=np.float64)
    callable_bins = np.ascontiguousarray(callable_bins, dtype=np.float64)
    total_bins = np.ascontiguousarray(total_bins, dtype=np.float64)
    child, parent, forward, reverse = _unordered_dyads(dyads, junctions.shape[1])
    values, compared = _pair_log_odds(
        np.ascontiguousarray(log_likelihood, dtype=np.float64).reshape(len(total_bins), -1),
        forward,
        reverse,
        child,
        parent,
        callable_bins,
        total_bins,
        float(error_probability),
    )
    return JunctionEvidence(junctions, callable_bins, total_bins, child, parent, values, compared)


def fit(evidence: JunctionEvidence, weights: np.ndarray) -> Orientation:
    """Orientation of a fit whose chromosomes carry multiplicities ``weights`` [C]."""
    # Burden and callability are report-only; direction decisions use the kernels below.
    counts = np.sum(weights[:, None] * evidence.junctions, axis=0)
    exposure = np.sum(weights[:, None] * evidence.callable_bins, axis=0)
    full = float(np.sum(weights * (2.0 * evidence.total_bins)))
    fractions = np.divide(exposure, full, out=np.zeros_like(exposure), where=full > 0)
    adjusted = np.divide(counts, fractions, out=np.full_like(counts, np.nan), where=fractions > 0)
    return Orientation(
        order_probability=_order_probability(evidence.junctions, evidence.callable_bins, evidence.total_bins, weights),
        log_odds=_weighted_log_odds(
            evidence.pair_log_odds,
            evidence.compared,
            evidence.pair_child,
            evidence.pair_parent,
            evidence.total_bins,
            np.ascontiguousarray(weights, dtype=np.float64),
            evidence.junctions.shape[1],
        ),
        adjusted_burden=adjusted,
        callability=fractions,
    )


def parent_role_probability(row: int, rows: np.ndarray, orientation: Orientation) -> float:
    """Graph-ordering support of a configuration: product of its edges' order probabilities
    (1 for M0); orientation-ordering support, not a parenthood probability."""
    parents = [int(parent) for parent in rows[row, 1:] if parent >= 0]
    if not parents:
        return 1.0
    return float(np.prod(orientation.order_probability[int(rows[row, 0]), parents]))


@kernel(overloads=OVERLOADS["orientation._order_statistics"], cache=True, nogil=True)
def _order_statistics(junctions, callable_bins, total_bins):
    """Additive four-moment statistics of the block-level graph-order model."""
    h, n = junctions.shape
    result = np.zeros((n, n, 4))
    for child in range(n):
        for parent in range(child):
            for block in range(h):
                if total_bins[block] <= 0:
                    continue
                fc = min(1.0, callable_bins[block, child] / (2 * total_bins[block]))
                fp = min(1.0, callable_bins[block, parent] / (2 * total_bins[block]))
                common = max(0.0, fc + fp - 1.0)
                if common <= 0:
                    continue
                dc, dp = junctions[block, child] / fc, junctions[block, parent] / fp
                delta = dc - dp
                result[child, parent, 0] += common
                result[child, parent, 1] += common * delta
                result[child, parent, 2] += common * delta * delta
                result[child, parent, 3] += common * ((1-fc)*dc/fc + (1-fp)*dp/fp)
    return result


@kernel(overloads=OVERLOADS["orientation._omit_order_probability"], cache=True, nogil=True)
def _omit_order_probability(totals, junctions, callable_bins, total_bins):
    """O(N²) order probabilities after removing one block's moment contributions."""
    n = len(junctions)
    result = np.full((n, n), .5)
    for child in range(n):
        for parent in range(child):
            count, mean, square, missing = totals[child, parent]
            if total_bins > 0:
                fc = min(1.0, callable_bins[child] / (2*total_bins))
                fp = min(1.0, callable_bins[parent] / (2*total_bins))
                common = max(0.0, fc+fp-1.0)
                if common > 0:
                    dc, dp = junctions[child]/fc, junctions[parent]/fp
                    delta = dc-dp
                    count -= common
                    mean -= common*delta
                    square -= common*delta*delta
                    missing -= common*((1-fc)*dc/fc+(1-fp)*dp/fp)
            if count <= ORDER_MINIMUM_WEIGHT:
                continue
            variance = max(0.0, square-mean*mean/count) * count/(count-1.0)
            variance += missing + ORDER_VARIANCE_OFFSET
            probability = .5 * (1+math.erf(mean/math.sqrt(2*variance)))
            result[child,parent] = probability
            result[parent,child] = 1-probability
    return result


def fit_omitting(evidence, totals, omitted):
    """The same block-level model, using totals minus the omitted block."""
    counts = totals.junctions - evidence.junctions[omitted]
    exposure = totals.callable_bins - evidence.callable_bins[omitted]
    full = 2.0 * (totals.total_bins - evidence.total_bins[omitted])
    fractions = np.divide(exposure, full, out=np.zeros_like(exposure), where=full > 0)
    adjusted = np.divide(counts, fractions, out=np.full_like(counts, np.nan), where=fractions > 0)
    values = totals.pair_log_odds - evidence.pair_log_odds[omitted]
    log_odds = np.zeros((len(counts), len(counts)))
    log_odds[evidence.pair_child, evidence.pair_parent] = values
    log_odds[evidence.pair_parent, evidence.pair_child] = -values
    return Orientation(_omit_order_probability(totals.order_statistics, evidence.junctions[omitted],
        evidence.callable_bins[omitted], float(evidence.total_bins[omitted])), log_odds, adjusted, fractions)
