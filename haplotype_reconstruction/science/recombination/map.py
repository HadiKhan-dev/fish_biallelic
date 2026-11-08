"""Map bins, interval-censored event mass and the aggregated recombination map of one chromosome.

Rates are expected crossovers per observable meiosis-bp, given the switch probabilities of the meioses' marker
intervals (in release 3.0 the parity of consecutive released chain bins); bins without exposure have NaN rates, not
zero recombination. No event or exposure crosses an assembly break (map bins are split at component edges).
"""

from __future__ import annotations

import math

import numpy as np
from numba import types

from ...contracts.recombination import RecombinationMap
from ...core.kernels import kernel
from .diagnostics import RecombinationTrace, Traced

_OVERLOADS = {
    "add_genetic_interval": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.float64,
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "add_interval": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.float64,
        ),
    ),
    "posterior_intervals": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.bool, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.bool, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.Omitted(None),
            types.Omitted(None),
            types.Omitted(None),
        ),
        (
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.bool, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.Omitted(None),
            types.Omitted(None),
            types.Omitted(None),
        ),
    ),
}


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


@kernel(overloads=_OVERLOADS["add_interval"], cache=True, nogil=True)
def add_interval(bins, totals, left, right, density):
    """Add density x overlap (bp) of [left, right] to every bin it overlaps."""
    index = max(0, np.searchsorted(bins, left, side="right") - 1)
    while index < len(totals) and bins[index] < right:
        width = max(0.0, min(right, bins[index + 1]) - max(left, bins[index]))
        totals[index] += density * width
        index += 1


@kernel(overloads=_OVERLOADS["add_genetic_interval"], cache=True, nogil=True)
def add_genetic_interval(bins, totals, left, right, density, map_positions_bp, map_genetic_morgans):
    """Add density x integrated genetic length (Morgans) of [left, right] inside every bin it overlaps."""
    index = max(0, np.searchsorted(bins, left, side="right") - 1)
    while index < len(totals) and bins[index] < right:
        a, b = max(left, bins[index]), min(right, bins[index + 1])
        if b > a:
            mass = np.interp(b, map_positions_bp, map_genetic_morgans) - np.interp(
                a, map_positions_bp, map_genetic_morgans
            )
            totals[index] += density * mass
        index += 1


@kernel(
    overloads=tuple(
        tuple(types.none if isinstance(t, types.Omitted) else t for t in sig) + (receipt,)
        for sig in _OVERLOADS["posterior_intervals"]
        for receipt in (types.none, types.Array(types.float64, 2, "C"))
    ),
    cache=True,
    nogil=True,
)
def _posterior_intervals(
    positions, resets, switch, bins, rate, genetic, map_positions_bp, map_genetic_morgans, receipt
):
    """Per bin expected crossovers and exposure between consecutive released markers of one meiosis.

    Between consecutive informative markers with genetic length lambda, Poisson/Haldane endpoint parity gives
    E[N | even] = lambda tanh(lambda) and E[N | odd] = lambda / tanh(lambda) (-> 1 as lambda -> 0); the interval's
    switch probability mixes them. Event mass is uniform in genetic distance; it is uniform in bp without a map and
    where the map gives the interval no genetic length (there an odd parity still counts its lambda -> 0 limit, one
    crossover: the data override the map's zero prior).
    """
    counts = np.zeros(len(bins) - 1)
    exposure = np.zeros_like(counts)
    for i in range(1, len(positions)):
        if resets[i]:
            continue
        left, right = float(positions[i - 1]), float(positions[i])
        distance = right - left
        lam = rate * distance if genetic is None else genetic[i] - genetic[i - 1]
        tangent = math.tanh(lam)
        positive = lam > 0
        odd_mean = lam / tangent if positive else 1.0
        expected = lam * tangent + switch[i] * (odd_mean - lam * tangent)
        uniform = genetic is None or lam <= 0
        if receipt is not None:
            receipt[i, 0] = 1.0
            receipt[i, 1] = left
            receipt[i, 2] = right
            receipt[i, 3] = lam
            receipt[i, 4] = positive
            receipt[i, 5] = not uniform
        if uniform:
            add_interval(bins, counts, left, right, expected / distance)
        else:
            add_genetic_interval(bins, counts, left, right, expected / lam, map_positions_bp, map_genetic_morgans)
        add_interval(bins, exposure, left, right, 1.0)
    return counts, exposure


def posterior_intervals(
    positions,
    resets,
    switch,
    bins,
    rate,
    genetic=None,
    map_positions_bp=None,
    map_genetic_morgans=None,
    *,
    trace=False,
    audit=None,
):
    """Conditional parity map, optionally retaining actual compiled branch operands.

    One compact row per visited marker interval records lambda and the two
    branches. Expanding those rows preserves the kernel's odd-mean/spread order.
    No numerical expression is repeated in the observer.
    """
    audit = RecombinationTrace() if trace and audit is None else audit
    receipt = np.zeros((len(positions), 6)) if audit is not None else None
    value = _posterior_intervals(
        positions, resets, switch, bins, rate, genetic, map_positions_bp, map_genetic_morgans, receipt
    )
    if audit is not None:
        for row in receipt:
            if row[0]:
                args = dict(lam=float(row[3]), has_map=genetic is not None, left=float(row[1]), right=float(row[2]))
                audit.emit("lambda.odd_mean", bool(row[4]), **args)
                audit.emit("lambda.spread", bool(row[5]), **args)
    return Traced(value, audit) if trace else value


def aggregate_map(bins, crossovers, callable_spans, expected, exposure, knots, knot_genetic):
    """RecombinationMap from the decoded meioses.

    ``crossovers`` (n, 5) and ``callable_spans`` (m, 3) of all edges (called events are spread over their flanking
    interval by genetic mass for binning only, uniformly in bp where it has none);
    ``expected``/``exposure`` float64 [E, B] per edge and map
    bin; the union of ``callable_spans`` is the covered sequence.
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
            else:
                add_interval(bins, counts, left, right, 1 / (right - left))
    for left, right, _ in callable_spans:
        add_interval(bins, called_exposure, left, right, 1.0)
    expected_total = expected.sum(axis=0)
    exposure_total = exposure.sum(axis=0)
    union = []
    for left, right in sorted((left, right) for left, right, _ in callable_spans):
        if union and left <= union[-1][1]:
            union[-1] = (union[-1][0], max(right, union[-1][1]))
        else:
            union.append((left, right))
    covered = np.zeros_like(exposure_total)
    for left, right in union:
        add_interval(bins, covered, left, right, 1.0)
    rate = np.divide(expected_total * 1e8, exposure_total, out=np.full_like(counts, np.nan), where=exposure_total > 0)
    distance = rate * covered / 1e6
    return RecombinationMap(
        bins,
        counts,
        exposure_total,
        expected_total,
        called_exposure,
        np.divide(counts * 1e8, called_exposure, out=np.full_like(counts, np.nan), where=called_exposure > 0),
        covered,
        exposure_total / np.diff(bins),
        rate,
        distance,
        np.r_[0.0, np.cumsum(np.nan_to_num(distance, nan=0.0))],
    )
