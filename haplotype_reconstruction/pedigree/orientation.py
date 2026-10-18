"""Orientation evidence from painted junction counts (which of two linked samples is the parent).

A transmitted homologue copies one parental homologue at each position, so on chromosome k it
carries each painted parental junction with probability 1/2, seen where the child is callable
(missing at random); new crossovers, co-parent and false junctions form a non-negative remainder.
Per chromosome, L(p -> c) is proportional to P(Binomial(J_p, f_c / 2) <= J_c); with probability
``chromosome_error_probability`` the comparison is painting failure carrying no orientation
evidence. The resulting log odds add over chromosomes (weighted by resampling multiplicity).
A paired junction-density contrast only orders the acyclic graph search and is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..maths.logspace import logaddexp


@dataclass(frozen=True)
class JunctionEvidence:
    """Per-chromosome painted junction data of the informative chromosomes (weight independent).

    junctions:     float64 [C, N] minimum founder switches of each painting.
    callable_bins: float64 [C, N] observed haplotype-bin labels behind them.
    total_bins:    float64 [C] painting bins (two haplotypes each).
    pair_log_odds: float64 [C, N, N] [chrom, child, parent] (parent < child) per-chromosome log odds of
                   parent -> child against child -> parent; ``compared`` bool [C, N, N] marks entries
                   with evidence. Computed once; resamples only reweight it.
    """

    junctions: np.ndarray
    callable_bins: np.ndarray
    total_bins: np.ndarray
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
def _binomial_log_cdf(trials, success, observed):
    """log P(Binomial(trials, success) <= observed) for small integer counts."""
    if observed < 0:
        return -np.inf
    if observed >= trials or success <= 0.0:
        return 0.0
    log_ratio = math.log(success) - math.log1p(-success)
    term = trials * math.log1p(-success)
    total = term
    for count in range(observed):
        term += math.log(trials - count) - math.log(count + 1.0) + log_ratio
        total = logaddexp(total, term)
    return min(0.0, total)


@njit(cache=True, nogil=True)
def _pair_log_odds(junctions, callable_bins, total_bins, error_probability):
    """values[chrom, child, parent] (parent < child) where ``compared``; see the module docstring."""
    chromosomes, samples = junctions.shape
    values = np.zeros((chromosomes, samples, samples))
    compared = np.zeros((chromosomes, samples, samples), dtype=np.bool_)
    keep = 1.0 - error_probability
    floor = 0.5 * error_probability
    for chrom in range(chromosomes):
        if total_bins[chrom] <= 0:
            continue
        for child in range(samples):
            fc = min(1.0, callable_bins[chrom, child] / (2.0 * total_bins[chrom]))
            if fc <= 0.0:
                continue
            jc = int(junctions[chrom, child])
            for parent in range(child):
                fp = min(1.0, callable_bins[chrom, parent] / (2.0 * total_bins[chrom]))
                if fp <= 0.0:
                    continue
                jp = int(junctions[chrom, parent])
                difference = (_binomial_log_cdf(jp, 0.5 * fc, jc)
                              - _binomial_log_cdf(jc, 0.5 * fp, jp))
                if difference == 0.0:
                    continue
                scaled = math.exp(-abs(difference))
                larger = keep / (1.0 + scaled) + floor
                smaller = keep * scaled / (1.0 + scaled) + floor
                value = math.log(larger) - math.log(smaller)
                if difference < 0.0:
                    value = -value
                values[chrom, child, parent] = value
                compared[chrom, child, parent] = True
    return values, compared


@njit(cache=True, nogil=True)
def _weighted_log_odds(values, compared, total_bins, multiplicity):
    """Chromosome-weighted sum of the pair log odds, ``result[child, parent]``, in chromosome order."""
    chromosomes, samples = values.shape[:2]
    result = np.zeros((samples, samples))
    for chrom in range(chromosomes):
        if multiplicity[chrom] <= 0 or total_bins[chrom] <= 0:
            continue
        for child in range(samples):
            for parent in range(child):
                if compared[chrom, child, parent]:
                    value = values[chrom, child, parent]
                    result[child, parent] += multiplicity[chrom] * value
                    result[parent, child] -= multiplicity[chrom] * value
    return result


def junction_evidence(junctions: np.ndarray, callable_bins: np.ndarray, total_bins: np.ndarray,
                      error_probability: float) -> JunctionEvidence:
    """Weight-independent junction evidence of the informative chromosomes."""
    junctions = np.ascontiguousarray(junctions, dtype=np.float64)
    callable_bins = np.ascontiguousarray(callable_bins, dtype=np.float64)
    total_bins = np.ascontiguousarray(total_bins, dtype=np.float64)
    values, compared = _pair_log_odds(junctions, callable_bins, total_bins, float(error_probability))
    return JunctionEvidence(junctions, callable_bins, total_bins, values, compared)


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
        log_odds=_weighted_log_odds(evidence.pair_log_odds, evidence.compared,
                                    evidence.total_bins, weights),
        adjusted_burden=adjusted, callability=fractions)


def parent_role_probability(row: int, rows: np.ndarray, orientation: Orientation) -> float:
    """Graph-ordering support of a configuration: product of its edges' order probabilities
    (1 for M0); orientation-ordering support, not a parenthood probability."""
    parents = [int(parent) for parent in rows[row, 1:] if parent >= 0]
    if not parents:
        return 1.0
    return float(np.prod(orientation.order_probability[int(rows[row, 0]), parents]))
