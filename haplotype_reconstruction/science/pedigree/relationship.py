"""Relationship evidence on bin-local painting classes (PD28).

Sequences are unordered in samples and homologues. The process is defined on the
known-bin grid: it conditions on the previous known observation and accumulates
skipped Haldane distances. Components restart independently.
Method: docs/methods/pedigree.md#pd28.
"""

from __future__ import annotations

import math

import numpy as np
from numba import prange

from ...core.kernels import kernel
from .kernels.signatures import OVERLOADS
from .parameters import RelationshipParameters


@kernel(overloads=OVERLOADS["relationship._pair_chance"], cache=True, parallel=True, nogil=True)
def _pair_chance(classes, known):
    """Independent bins; sorted observed classes make renumbering immaterial."""
    n, bins, _ = classes.shape
    out = np.empty(bins)
    for t in prange(bins):
        labels = np.empty(2 * n, np.int64)
        count = 0
        for i in range(n):
            if known[i, t]:
                labels[count] = classes[i, t, 0]
                labels[count + 1] = classes[i, t, 1]
                count += 2
        if count == 0:
            out[t] = 1.0 - 1e-9
            continue
        labels[:count].sort()
        frequencies = np.zeros(count)
        used = 0
        for i in range(count):
            if i == 0 or labels[i] != labels[i - 1]:
                used += 1
            frequencies[used - 1] += 1.0 / count
        miss = 0.0
        for a in range(used):
            for b in range(used):
                rest = 1.0 - frequencies[a] - (frequencies[b] if a != b else 0.0)
                miss += frequencies[a] * frequencies[b] * rest * rest
        out[t] = min(1.0 - 1e-9, max(1e-9, 1.0 - miss))
    return out


@kernel(overloads=OVERLOADS["relationship._pair_value"], cache=True, nogil=True)
def _pair_value(classes, known, a, b, t):
    if not (known[a, t] and known[b, t]):
        return -1
    return int(
        not (
            classes[a, t, 0] == classes[b, t, 0]
            or classes[a, t, 0] == classes[b, t, 1]
            or classes[a, t, 1] == classes[b, t, 0]
            or classes[a, t, 1] == classes[b, t, 1]
        )
    )


@kernel(overloads=OVERLOADS["relationship._feasible3"], cache=True, nogil=True)
def _feasible3(g):
    # Orient each genotype between the latent parents. Each parent must use at
    # most two distinct classes. These eight assignments exhaust every solution.
    for mask in range(8):
        m0, f0 = g[0, mask & 1], g[0, 1 - (mask & 1)]
        m1, f1 = g[1, (mask >> 1) & 1], g[1, 1 - ((mask >> 1) & 1)]
        m2, f2 = g[2, (mask >> 2) & 1], g[2, 1 - ((mask >> 2) & 1)]
        if (m0 == m1 or m0 == m2 or m1 == m2) and (f0 == f1 or f0 == f2 or f1 == f2):
            return True
    return False


@kernel(overloads=OVERLOADS["relationship._pair_sequence"], cache=True, parallel=True, nogil=True)
def _pair_sequence(classes, known, dyads):
    """Parallel over sequences; no reductions shared between sequences."""
    out = np.empty((len(dyads), classes.shape[1]), np.int8)
    for i in prange(len(dyads)):
        for t in range(classes.shape[1]):
            out[i, t] = _pair_value(classes, known, dyads[i, 0], dyads[i, 1], t)
    return out


@kernel(overloads=OVERLOADS["relationship._trio_sequence"], cache=True, parallel=True, nogil=True)
def _trio_sequence(classes, known, trios):
    """Parallel over sequences; constant private feasibility workspace."""
    out = np.full((len(trios), classes.shape[1]), -1, np.int8)
    for i in prange(len(trios)):
        g = np.empty((3, 2), np.int64)
        for t in range(classes.shape[1]):
            if known[trios[i, 0], t] and known[trios[i, 1], t] and known[trios[i, 2], t]:
                for j in range(3):
                    g[j, 0] = classes[trios[i, j], t, 0]
                    g[j, 1] = classes[trios[i, j], t, 1]
                out[i, t] = int(not _feasible3(g))
    return out


@kernel(overloads=OVERLOADS["relationship._pair_persistence"], cache=True, parallel=True, nogil=True)
def _pair_persistence(classes, known):
    """All unordered pairs; integer event counts reduced in sample order."""
    numerator = np.zeros(len(classes), np.int64)
    denominator = np.zeros(len(classes), np.int64)
    for a in prange(len(classes)):
        for b in range(a):
            previous = -1
            for t in range(classes.shape[1]):
                value = _pair_value(classes, known, np.int64(a), np.int64(b), t)
                if previous == 1 and value >= 0:
                    denominator[a] += 1
                    numerator[a] += int(value == 1)
                previous = value
    den = denominator.sum()
    return numerator.sum() / den if den else 0.5


@kernel(overloads=OVERLOADS["relationship._process_log_likelihoods"], cache=True, parallel=True, nogil=True)
def _process_log_likelihoods(x, distances, start, stop, q, persistence, rates, isolated_error):
    """Parallel over sequences, four forward sums, independent component starts."""
    out = np.zeros((len(x), len(rates)))
    for i in prange(len(x)):
        for h in range(len(rates)):
            rate = rates[h, 0] + rates[h, 1]
            pn = rates[h, 0] / rate if rate > 0.0 else 0.0
            total = 0.0
            for k in range(len(start)):
                fs, fn = 1.0 - pn, pn
                previous = -1
                gap = 0.0
                for t in range(start[k], stop[k]):
                    gap += distances[t]
                    v = x[i, t]
                    if v < 0:
                        continue
                    if previous < 0:
                        en = q[t] if v == 0 else 1.0 - q[t]
                    else:
                        mix = 1.0 - math.exp(-rate * gap)
                        to_n, to_s = pn * mix, (1.0 - pn) * mix
                        ns = fs * (1.0 - to_n) + fn * to_s
                        fn = fs * to_n + fn * (1.0 - to_s)
                        fs = ns
                        s = persistence[k]
                        if previous == 1:
                            en = s if v == 1 else 1.0 - s
                        else:
                            p1 = min(1.0 - 1e-6, (1.0 - q[t]) * (1.0 - s) / q[t])
                            en = p1 if v == 1 else 1.0 - p1
                    error = isolated_error * (1.0 - q[t])
                    fs *= error if v == 1 else 1.0 - error
                    fn *= en
                    norm = fs + fn
                    if norm <= 0.0:
                        total = -np.inf
                        break
                    fs /= norm
                    fn /= norm
                    total += math.log(norm)
                    previous = v
                    gap = 0.0
                if not np.isfinite(total):
                    break
            out[i, h] = total
    return out


def _inputs(classes, known):
    return np.array(classes, dtype=np.int64, order="C", copy=True), np.array(
        known, dtype=np.bool_, order="C", copy=True
    )


def pair_chance(classes, known):
    """Per-bin chance of a shared class in two independent cohort genotypes."""
    return _pair_chance(*_inputs(classes, known))


def pair_sequence(classes, known, dyads):
    """int8 [D,B], zero shared class, one no share, minus one unknown."""
    return _pair_sequence(
        *_inputs(classes, known), np.array(dyads, dtype=np.int64, order="C", copy=True).reshape(-1, 2)
    )


def trio_sequence(classes, known, trios):
    """int8 [T,B], zero compatible with two latent parents, one incompatible."""
    return _trio_sequence(
        *_inputs(classes, known), np.array(trios, dtype=np.int64, order="C", copy=True).reshape(-1, 3)
    )


def pair_persistence(classes, known):
    """P(1|previous 1), over all pairs and adjacent bins of ONE component."""
    return float(np.clip(_pair_persistence(*_inputs(classes, known)), 0.01, 0.99))


def trio_reference(classes, known, triples):
    """Chance q and persistence s of the supplied reference triples, one component."""
    x = trio_sequence(classes, known, triples)
    phi = (x == 1).sum(axis=0) / np.maximum((x >= 0).sum(axis=0), 1)
    events = (x[:, :-1] == 1) & (x[:, 1:] >= 0)
    denominator = int(events.sum())
    s = float(((x[:, 1:] == 1) & events).sum() / denominator) if denominator else 0.5
    return np.clip(1.0 - phi, 1e-3, 1.0 - 1e-3), float(np.clip(s, 0.01, 0.99))


def process_rates(parameters: RelationshipParameters):
    """FEASIBLE, FS-like, D2-like, NEVER; rates per Morgan."""
    return np.array(
        [
            [parameters.failure_region_rate, 1.0 / parameters.failure_region_length],
            [4.0 / 3.0, 4.0],
            [2.0, 2.0],
            [1.0, 0.0],
        ]
    )


def haldane_distances(theta):
    """Distance entering each bin, Morgans; prepared theta in [0, 0.5]."""
    theta = np.asarray(theta, dtype=np.float64)
    if np.any(~np.isfinite(theta)) or np.any(theta < 0) or np.any(theta > 0.5):
        raise ValueError("relationship selector probabilities must lie in [0, 0.5]")
    with np.errstate(divide="ignore"):
        return np.concatenate(([0.0], -0.5 * np.log1p(-2.0 * theta)))


def process_log_likelihoods(x, distances, q, s, rates=None, *, starts=None, stops=None, isolated_error=1e-3):
    """float64 [sequences,4], sums of independent component likelihoods.

    distances[B] holds the entering-boundary Haldane distance (zero at each
    component start). q[B] is bin-specific; s is scalar or [components].
    """
    if starts is None:
        starts = [0]
        stops = [np.shape(x)[1]]
    start = np.array(starts, dtype=np.int64, order="C", copy=True)
    stop = np.array(stops, dtype=np.int64, order="C", copy=True)
    persistence = np.clip(np.full(len(start), s) if np.ndim(s) == 0 else s, 0.01, 0.99)
    return _process_log_likelihoods(
        np.array(x, dtype=np.int8, order="C", copy=True),
        np.array(distances, dtype=np.float64, order="C", copy=True),
        start,
        stop,
        np.array(q, dtype=np.float64, order="C", copy=True),
        np.array(persistence, dtype=np.float64, order="C", copy=True),
        process_rates(RelationshipParameters())
        if rates is None
        else np.array(rates, dtype=np.float64, order="C", copy=True),
        float(isolated_error),
    )


def evidence(log_likelihoods):
    """Feasible minus the best of all four processes, after summing chromosomes."""
    values = np.asarray(log_likelihoods)
    return values[..., 0] - np.max(values, axis=-1)
