"""Orientation evidence from painted junction counts (which of two linked samples is the parent).

A transmitted homologue copies one parental homologue at each position and switches between the two at
meiotic crossovers, so the parent's painted junctions are inherited together along a homologue (release 2.16;
pedigree.junctions.linked_log_cdf): the junctions sit on the parent's two homologues of its minimum-switch
trajectory pair, the selector switches with the meiotic probability between bins, and an inherited junction is
seen where the child is callable (its chromosome-wide callable fraction f_c, missing at random); new crossovers,
co-parent and false junctions form a non-negative remainder. Per chromosome, L(p -> c) is proportional to
P(inherited detected junctions <= J_c); with probability ``chromosome_error_probability`` the comparison is
painting failure carrying no orientation evidence (two impossible directions are uninformative). The
resulting log odds add over chromosomes (weighted by resampling multiplicity).
A paired junction-density contrast only orders the acyclic graph search and is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit



@dataclass(frozen=True)
class JunctionEvidence:
    """Per-chromosome painted junction data of the informative chromosomes (weight independent).

    junctions:     float64 [C, N] minimum founder switches of each painting.
    callable_bins: float64 [C, N] observed haplotype-bin labels behind them.
    total_bins:    float64 [C] painting bins (two haplotypes each).
    pair_child, pair_parent: int64 [U] the unordered dyads with both directions oriented, as (larger index,
                   smaller index) like the old [child, parent] (parent < child) triangle.
    pair_log_odds: float64 [C, U] per-chromosome log odds of pair_parent -> pair_child against the reverse;
                   ``compared`` bool [C, U] marks entries with evidence. Computed once; resamples only reweight
                   it. Stored per dyad (release 2.16), not as [C, N, N] triangles.
    """

    junctions: np.ndarray
    callable_bins: np.ndarray
    total_bins: np.ndarray
    pair_child: np.ndarray
    pair_parent: np.ndarray
    pair_log_odds: np.ndarray
    compared: np.ndarray


@dataclass(frozen=True)
class Orientation:
    """Direction evidence of one (re)weighted fit.

    order_probability: float64 [N, N] [child, parent] paired junction-density contrast in [0, 1]
                       (0.5 = no information); orders the graph search, reported, never scored.
    log_odds:          float64 [N, N] [child, parent] weighted sum of the chromosome log odds of
                       parent -> child against child -> parent (antisymmetric); score evidence.
    adjusted_burden:   float64 [N] weighted junctions / callability fraction (report).
    callability:       float64 [N] weighted callable bins / (2 x weighted total bins) (report).
    """

    order_probability: np.ndarray
    log_odds: np.ndarray
    adjusted_burden: np.ndarray
    callability: np.ndarray


@njit(cache=True, nogil=True)
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
            if count <= 2.0:
                continue
            variance = max(0.0, square_sum - mean_sum * mean_sum / count)
            variance *= count / (count - 1.0)
            variance += missing_variance + 1.0
            probability = 0.5 * (1.0 + math.erf(mean_sum / math.sqrt(2.0 * variance)))
            result[child, parent] = probability
            result[parent, child] = 1.0 - probability
    return result


@njit(cache=True, nogil=True)
def _pair_log_odds(log_cdf, forward_index, reverse_index, pair_child, pair_parent, callable_bins, total_bins,
                   error_probability):
    """values[chrom, u] for the unordered dyads u (pair_parent -> pair_child against the reverse) where
    ``compared``; log_cdf[chrom, d] over the ordered dyads (pedigree.junctions.linked_log_cdf)."""
    chromosomes = log_cdf.shape[0]
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
            forward = log_cdf[chrom, forward_index[u]]
            reverse = log_cdf[chrom, reverse_index[u]]
            if forward == -np.inf and reverse == -np.inf:
                continue                       # impossible both ways: uninformative
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


@njit(cache=True, nogil=True)
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
    return (np.ascontiguousarray(dyads[forward, 0]), np.ascontiguousarray(dyads[forward, 1]),
            np.ascontiguousarray(forward), np.ascontiguousarray(reverse))


def junction_evidence(junctions: np.ndarray, callable_bins: np.ndarray, total_bins: np.ndarray,
                      error_probability: float, dyads: np.ndarray, log_cdf: np.ndarray) -> JunctionEvidence:
    """Weight-independent junction evidence of the informative chromosomes (``dyads`` [D, 2] and ``log_cdf``
    [C, D]: the linked transmission compatibilities of the scores)."""
    junctions = np.ascontiguousarray(junctions, dtype=np.float64)
    callable_bins = np.ascontiguousarray(callable_bins, dtype=np.float64)
    total_bins = np.ascontiguousarray(total_bins, dtype=np.float64)
    child, parent, forward, reverse = _unordered_dyads(dyads, junctions.shape[1])
    values, compared = _pair_log_odds(np.ascontiguousarray(log_cdf, dtype=np.float64).reshape(len(total_bins), -1),
                                      forward, reverse, child, parent, callable_bins, total_bins,
                                      float(error_probability))
    return JunctionEvidence(junctions, callable_bins, total_bins, child, parent, values, compared)


def fit(evidence: JunctionEvidence, weights: np.ndarray) -> Orientation:
    """Orientation of a fit whose chromosomes carry multiplicities ``weights`` [C]."""
    counts = weights @ evidence.junctions
    exposure = weights @ evidence.callable_bins
    full = float(weights @ (2.0 * evidence.total_bins))
    fractions = np.divide(exposure, full, out=np.zeros_like(exposure), where=full > 0)
    adjusted = np.divide(counts, fractions, out=np.full_like(counts, np.nan), where=fractions > 0)
    return Orientation(
        order_probability=_order_probability(evidence.junctions, evidence.callable_bins,
                                             evidence.total_bins, weights),
        log_odds=_weighted_log_odds(evidence.pair_log_odds, evidence.compared, evidence.pair_child,
                                    evidence.pair_parent, evidence.total_bins,
                                    np.ascontiguousarray(weights, dtype=np.float64), evidence.junctions.shape[1]),
        adjusted_burden=adjusted, callability=fractions)


def parent_role_probability(row: int, rows: np.ndarray, orientation: Orientation) -> float:
    """Graph-ordering support of a configuration: product of its edges' order probabilities
    (1 for M0); orientation-ordering support, not a parenthood probability."""
    parents = [int(parent) for parent in rows[row, 1:] if parent >= 0]
    if not parents:
        return 1.0
    return float(np.prod(orientation.order_probability[int(rows[row, 0]), parents]))
