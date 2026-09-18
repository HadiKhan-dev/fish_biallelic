"""Fused, missing-aware evidence kernels for painting and pedigree preparation."""
import math
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=False)
def _gather_painting_sites(evidence, observed, indices):
    """Gather sparse raw evidence once, preserving dtype and observation flags."""
    samples = evidence.shape[0]
    sites = len(indices)
    values = np.empty((samples, sites, 3), dtype=evidence.dtype)
    flags = np.empty((samples, sites), dtype=np.bool_)
    for sample in prange(samples):
        for site in range(sites):
            source = indices[site]
            values[sample, site, 0] = evidence[sample, source, 0]
            values[sample, site, 1] = evidence[sample, source, 1]
            values[sample, site, 2] = evidence[sample, source, 2]
            flags[sample, site] = (
                True if observed is None else observed[sample, source]
            )
    return values, flags


def select_site_evidence(evidence, observed, indices):
    """Select increasing sites for read-only painting consumers.

    Both callers supply strictly increasing indices from aligned coordinates
    or flatnonzero. Consecutive selections can remain views; sparse float32/64
    selections use one native gather instead of advanced-indexing temporaries.
    A missing observed mask retains the all-observed API default.
    """
    if not len(indices):
        start = stop = 0
    elif int(indices[-1]) - int(indices[0]) + 1 == len(indices):
        start, stop = int(indices[0]), int(indices[-1]) + 1
    else:
        if evidence.dtype in (np.dtype("float32"), np.dtype("float64")):
            return _gather_painting_sites(evidence, observed, indices)
        values = np.ascontiguousarray(evidence[:, indices,:])
        flags = (
            np.ones(values.shape[:2], dtype=np.bool_) if observed is None
            else np.ascontiguousarray(observed[:, indices])
        )
        return values, flags
    values = evidence[:, start:stop,:]
    flags = (
        np.ones(values.shape[:2], dtype=np.bool_) if observed is None
        else observed[:, start:stop]
    )
    return values, flags


@njit(cache=True, parallel=True, fastmath=False)
def gather_component_evidence(evidence, observed, indices, epsilon):
    """Copy selected raw rows and count observed rows with spread > epsilon."""
    samples = evidence.shape[0]
    sites = len(indices)
    values = np.empty((samples, sites, 3), dtype=evidence.dtype)
    called = np.empty((samples, sites), dtype=np.bool_)
    counts = np.zeros(samples, dtype=np.int64)
    for sample in prange(samples):
        count = 0
        for site in range(sites):
            source = indices[site]
            a, b, c = evidence[sample, source, 0], evidence[sample, source, 1], evidence[sample, source, 2]
            values[sample, site, 0] = a
            values[sample, site, 1] = b
            values[sample, site, 2] = c
            flag = observed[sample, source]
            called[sample, site] = flag
            if flag and max(a, b, c) - min(a, b, c) > epsilon:
                count += 1
        counts[sample] = count
    return values, called, counts


@njit(cache=True, inline="always")
def _normalized_row(x0, x1, x2):
    # Normalize in float64 exactly as the public NumPy implementation does.
    a, b, c = np.float64(x0), np.float64(x1), np.float64(x2)
    valid = (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)
             and a >= 0.0 and b >= 0.0 and c >= 0.0)
    total = a + b + c
    if total > 0.0:
        return a / total, b / total, c / total, valid
    return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, valid

@njit(cache=True, parallel=True)
def normalize_rows(evidence):
    """One read/normalize/write pass, with a scalar invalid-row reduction."""
    samples, sites, _ = evidence.shape
    out = np.empty((samples, sites, 3), dtype=np.float64)
    invalid = 0
    for row in prange(samples * sites):
        sample, site = row // sites, row % sites
        a, b, c, valid = _normalized_row(
            evidence[sample, site, 0], evidence[sample, site, 1],
            evidence[sample, site, 2])
        out[sample, site, 0] = a
        out[sample, site, 1] = b
        out[sample, site, 2] = c
        invalid += int(not valid)
    return out, invalid

@njit(cache=True, parallel=True)
def eligible_samples(evidence, observed):
    """Same normalized-row spread predicate, without a full normalized copy."""
    result = np.zeros(evidence.shape[0], dtype=np.bool_)
    invalid = 0
    epsilon = 16.0 * np.finfo(np.float64).eps
    for sample in prange(evidence.shape[0]):
        found = False
        bad = 0
        for site in range(evidence.shape[1]):
            a, b, c, valid = _normalized_row(
                evidence[sample, site, 0], evidence[sample, site, 1],
                evidence[sample, site, 2])
            bad += int(not valid)
            if observed[sample, site] and max(a, b, c) - min(a, b, c) > epsilon:
                found = True
        result[sample] = found
        invalid += bad
    return result, invalid

@njit(cache=True, parallel=True)
def raw_nonuniform(evidence, epsilon):
    """Raw-row spread predicate, preserving the input floating-point dtype."""
    samples, sites, _ = evidence.shape
    result = np.empty((samples, sites), dtype=np.bool_)
    for row in prange(samples * sites):
        sample, site = row // sites, row % sites
        a, b, c = evidence[sample, site, 0], evidence[sample, site, 1], evidence[sample, site, 2]
        total = (a + b) + c
        result[sample, site] = total > 0 and max(a, b, c) - min(a, b, c) > epsilon * total
    return result


@njit(cache=True, parallel=True)
def direct_callability(internal_grid, released_grid, called, observed,
                       nonuniform, starts, stops, background_index):
    """Exact per-bin observed-founder support, stopping at its first witness."""
    samples, _, bins = internal_grid.shape
    result = np.zeros(internal_grid.shape, dtype=np.bool_)
    for task in prange(samples * 2):
        sample, track = task // 2, task % 2
        for bin_index in range(bins):
            state = internal_grid[sample, track, bin_index]
            if released_grid[sample, track, bin_index] < 0 or state == background_index:
                continue
            for site in range(starts[bin_index], stops[bin_index]):
                if (called[state, site] and observed[sample, site]
                        and nonuniform[sample, site]):
                    result[sample, track, bin_index] = True
                    break
    return result


@njit(cache=True, parallel=True)
def count_component_information(evidence, observed, indices, epsilon):
    """The T10 normalized-row criterion without gathering an unused GL copy."""
    counts = np.zeros(evidence.shape[0], dtype=np.int64)
    for sample in prange(evidence.shape[0]):
        count = 0
        for source in indices:
            a, b, c = evidence[sample, source, 0], evidence[sample, source, 1], evidence[sample, source, 2]
            if observed[sample, source] and max(a, b, c) - min(a, b, c) > epsilon:
                count += 1
        counts[sample] = count
    return counts


@njit(cache=True, parallel=True)
def store_symmetric_emissions(emissions, packed, sample_start, states):
    """Store only the upper triangle of symmetric diploid emission scores."""
    for task in prange(emissions.shape[0] * states):
        sample, first = task // states, task % states
        row_start = first * states - first * (first - 1) // 2
        for second in range(first, states):
            for block in range(emissions.shape[2]):
                packed[sample_start + sample, row_start + second - first, block] = (
                    emissions[sample, first * states + second, block])


@njit(cache=True, parallel=True)
def expand_symmetric_emissions(packed, states):
    """Restore ordered source emissions; never symmetrize source transitions."""
    output = np.empty((packed.shape[0], states * states, packed.shape[2]))
    for task in prange(packed.shape[0] * states):
        sample, first = task // states, task % states
        row_start = first * states - first * (first - 1) // 2
        for second in range(first, states):
            for block in range(packed.shape[2]):
                value = packed[sample, row_start + second - first, block]
                output[sample, first * states + second, block] = value
                output[sample, second * states + first, block] = value
    return output
