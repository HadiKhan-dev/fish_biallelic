"""Map bins, interval-censored event mass and the aggregated recombination map of one chromosome.

Rates are posterior expected crossovers per observable meiosis-bp; bins without exposure have NaN rates, not
zero recombination. No event or exposure crosses an assembly break (map bins are split at component edges).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit


@dataclass(frozen=True)
class RecombinationMap:
    """Per map bin (edges_bp float64 [B+1]): crossover_count (called events, spread over their intervals),
    exposure_meiosis_bp (observable meiosis-bp), expected_crossovers (posterior mean), called_exposure_meiosis_bp,
    called_rate_cM_per_Mb, covered_bp (union of observable sequence), effective_meioses, rate_cM_per_Mb,
    cM_on_covered_bp; cumulative_observed_cM float64 [B+1] (cumulative over covered sequence; gaps omitted)."""

    edges_bp: np.ndarray
    crossover_count: np.ndarray
    exposure_meiosis_bp: np.ndarray
    expected_crossovers: np.ndarray
    called_exposure_meiosis_bp: np.ndarray
    called_rate_cM_per_Mb: np.ndarray
    covered_bp: np.ndarray
    effective_meioses: np.ndarray
    rate_cM_per_Mb: np.ndarray
    cM_on_covered_bp: np.ndarray
    cumulative_observed_cM: np.ndarray


def map_bins(positions, components, bin_bp):
    """Map bin edges (bp, float64): every bin_bp from the first site, split at component (segment) edges."""
    lo, hi = float(positions[0]), float(positions[-1])
    bins = np.r_[np.arange(lo, hi, float(bin_bp)), hi]
    changes = np.flatnonzero(components[1:] != components[:-1])
    return np.unique(np.r_[bins, positions[changes], positions[changes + 1]])


def genetic_coordinates(positions, chromosome_map):
    """(site Morgans, interpolation knots bp, knot Morgans) with a supplied map; (None, None, None) without.

    The knots keep every map knot between the first and last site (hotspots between SNPs included).
    """
    if not chromosome_map.has_map:
        return None, None, None
    genetic = chromosome_map.cumulative_morgans(positions)
    knots = np.unique(np.r_[positions[0], chromosome_map.positions_bp, positions[-1]])
    return genetic, knots, chromosome_map.cumulative_morgans(knots)


@njit(cache=True)
def add_interval(bins, totals, left, right, density):
    """Add density x overlap (bp) of [left, right] to every bin it overlaps."""
    index = max(0, np.searchsorted(bins, left, side="right") - 1)
    while index < len(totals) and bins[index] < right:
        width = max(0., min(right, bins[index + 1]) - max(left, bins[index]))
        totals[index] += density * width
        index += 1


@njit(cache=True)
def add_genetic_interval(bins, totals, left, right, density, map_positions_bp, map_genetic_morgans):
    """Add density x integrated genetic length (Morgans) of [left, right] inside every bin it overlaps."""
    index = max(0, np.searchsorted(bins, left, side="right") - 1)
    while index < len(totals) and bins[index] < right:
        a, b = max(left, bins[index]), min(right, bins[index + 1])
        if b > a:
            mass = (np.interp(b, map_positions_bp, map_genetic_morgans)
                    - np.interp(a, map_positions_bp, map_genetic_morgans))
            totals[index] += density * mass
        index += 1


@njit(cache=True)
def posterior_intervals(positions, components, resets, switch, bins, rate,
                        genetic=None, map_positions_bp=None, map_genetic_morgans=None):
    """Per bin expected crossovers and exposure of one meiosis, and its informative spans (left, right, component).

    Between consecutive informative markers with genetic length lambda, Poisson/Haldane endpoint parity gives
    E[N | even] = lambda tanh(lambda) and E[N | odd] = lambda / tanh(lambda) (-> 1 as lambda -> 0); the posterior
    switch probability mixes them. Event mass is uniform in genetic distance (physical without a map).
    """
    counts = np.zeros(len(bins) - 1)
    exposure = np.zeros_like(counts)
    spans = [(0., 0., 0.) for _ in range(0)]
    for i in range(1, len(positions)):
        if resets[i]:
            continue
        left, right = float(positions[i - 1]), float(positions[i])
        distance = right - left
        lam = rate * distance if genetic is None else genetic[i] - genetic[i - 1]
        tangent = math.tanh(lam)
        odd_mean = lam / tangent if lam > 0 else 1.
        expected = lam * tangent + switch[i] * (odd_mean - lam * tangent)
        if genetic is None:
            add_interval(bins, counts, left, right, expected / distance)
        elif lam > 0:
            add_genetic_interval(bins, counts, left, right, expected / lam, map_positions_bp, map_genetic_morgans)
        add_interval(bins, exposure, left, right, 1.)
        component = float(components[i])
        if spans and spans[-1][1] == left and spans[-1][2] == component:
            spans[-1] = (spans[-1][0], right, component)
        else:
            spans.append((left, right, component))
    array = np.empty((len(spans), 3), dtype=np.float64)
    for i in range(len(spans)):
        for k in range(3):
            array[i, k] = spans[i][k]
    return counts, exposure, array


def aggregate_map(bins, crossovers, callable_spans, expected, exposure, informative_spans, knots, knot_genetic):
    """RecombinationMap from the decoded meioses.

    ``crossovers`` (n, 5) and ``callable_spans`` (m, 3) of all edges (called events are spread over their flanking
    interval by genetic mass for binning only); ``expected``/``exposure`` float64 [E, B] posterior per-edge
    bins; ``informative_spans`` (k, 3) of all edges (their union is the covered sequence).
    """
    counts = np.zeros(len(bins) - 1)
    called_exposure = np.zeros_like(counts)
    for left, right, *_ in crossovers:
        if knots is None:
            add_interval(bins, counts, left, right, 1 / (right - left))
        else:
            mass = np.interp(right, knots, knot_genetic) - np.interp(left, knots, knot_genetic)
            if mass > 0:
                add_genetic_interval(bins, counts, left, right, 1 / mass, knots, knot_genetic)
    for left, right, _ in callable_spans:
        add_interval(bins, called_exposure, left, right, 1.)
    expected_total = expected.sum(axis=0)
    exposure_total = exposure.sum(axis=0)
    union = []
    for left, right in sorted((left, right) for left, right, _ in informative_spans):
        if union and left <= union[-1][1]:
            union[-1] = (union[-1][0], max(right, union[-1][1]))
        else:
            union.append((left, right))
    covered = np.zeros_like(exposure_total)
    for left, right in union:
        add_interval(bins, covered, left, right, 1.)
    rate = np.divide(expected_total * 1e8, exposure_total, out=np.full_like(counts, np.nan),
                     where=exposure_total > 0)
    distance = rate * covered / 1e6
    return RecombinationMap(
        bins, counts, exposure_total, expected_total, called_exposure,
        np.divide(counts * 1e8, called_exposure, out=np.full_like(counts, np.nan), where=called_exposure > 0),
        covered, exposure_total / np.diff(bins), rate, distance,
        np.r_[0., np.cumsum(np.nan_to_num(distance, nan=0.))])

