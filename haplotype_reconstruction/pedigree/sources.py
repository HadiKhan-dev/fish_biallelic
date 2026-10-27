"""Exact source posterior of every sample as a candidate parent, within pedigree's reduced source model.

A sample's source chain is an HMM over ordered diplotypes (S^2 states, S = classes + BACKGROUND) with the
painting's label-level Hamming transition, which is symmetric and depends only on how many founders change, and its
tempered emissions; it leaves out painting's panel-error state (release 2.14), so it is not the marginal of the
painting's own chain.
Its backward pass is stored compactly as initial probabilities and max-normalised right weights
r_b (O(N B S^2)); the posterior chain kernel K_b(u, v) = A_b(u, v) r_b(v) / sum_w A_b(u, w) r_b(w)
and the per-bin marginals follow in O(S^2) per bin through row, column and total sums (never an
S^4 transition).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .records import HammingTransition, SourceFactors


@njit(cache=True, inline="always")
def _hamming_contract(matrix, same, one, two, output, rows, columns):
    """output = A matrix for the symmetric ordered-diplotype Hamming kernel A, in O(S^2)."""
    states = matrix.shape[0]
    rows[:] = 0.0
    columns[:] = 0.0
    total = 0.0
    for first in range(states):
        for second in range(states):
            value = matrix[first, second]
            rows[first] += value
            columns[second] += value
            total += value
    for first in range(states):
        for second in range(states):
            cell = matrix[first, second]
            output[first, second] = (same * cell + one * (rows[first] + columns[second] - 2.0 * cell)
                                     + two * (total - rows[first] - columns[second] + cell))


@njit(cache=True, parallel=True)
def _factor_kernel(emission, same, one, two, available):
    """Backward factors per candidate (prange): initial [N, S, S], right [N, B-1, S, S], valid [N]."""
    candidates, bins, states, _ = emission.shape
    initial = np.zeros((candidates, states, states), dtype=np.float64)
    right = np.ones((candidates, max(0, bins - 1), states, states), dtype=np.float64)
    valid_out = np.ones(candidates, dtype=np.bool_)
    for candidate in prange(candidates):
        if not available[candidate]:
            initial[candidate] = 1.0 / float(states * states)
            continue
        beta = np.ones((states, states), dtype=np.float64)
        contraction = np.empty((states, states), dtype=np.float64)
        rows = np.empty(states, dtype=np.float64)
        columns = np.empty(states, dtype=np.float64)
        valid = True
        for boundary in range(bins - 2, -1, -1):
            maximum = -np.inf
            for first in range(states):
                for second in range(states):
                    maximum = max(maximum, emission[candidate, boundary + 1, first, second])
            if not np.isfinite(maximum):
                valid = False
                break
            maximum_right = 0.0
            for first in range(states):
                for second in range(states):
                    value = (math.exp(emission[candidate, boundary + 1, first, second] - maximum)
                             * beta[first, second])
                    right[candidate, boundary, first, second] = value
                    maximum_right = max(maximum_right, value)
            if maximum_right <= 0.0:
                valid = False
                break
            right[candidate, boundary] /= maximum_right
            _hamming_contract(right[candidate, boundary], same[boundary], one[boundary],
                              two[boundary], contraction, rows, columns)
            maximum_beta = np.max(contraction)
            if maximum_beta <= 0.0 or not np.isfinite(maximum_beta):
                valid = False
                break
            beta = contraction / maximum_beta
        if valid:
            maximum = np.max(emission[candidate, 0])
            total = 0.0
            for first in range(states):
                for second in range(states):
                    value = math.exp(emission[candidate, 0, first, second] - maximum) * beta[first, second]
                    initial[candidate, first, second] = value
                    total += value
            if total <= 0.0 or not np.isfinite(total):
                valid = False
            else:
                initial[candidate] /= total
                # Unphased source GLs have an arbitrary homologue gauge: remove only floating-point
                # asymmetry, never unordered mass.
                for first in range(states):
                    for second in range(first + 1, states):
                        value = 0.5 * (initial[candidate, first, second]
                                       + initial[candidate, second, first])
                        initial[candidate, first, second] = value
                        initial[candidate, second, first] = value
        if not valid:
            valid_out[candidate] = False
            initial[candidate] = 1.0 / float(states * states)
            right[candidate] = 1.0
    return initial, right, valid_out


@njit(cache=True, parallel=True)
def _marginal_kernel(initial, right, same, one, two):
    """Per-bin ordered-diplotype posterior marginals [N, B, S, S] of the stored chains."""
    candidates, states, _ = initial.shape
    boundaries = right.shape[1]
    posterior = np.empty((candidates, boundaries + 1, states, states), dtype=np.float64)
    valid = np.ones(candidates, dtype=np.bool_)
    for candidate in prange(candidates):
        rows = np.empty(states, dtype=np.float64)
        columns = np.empty(states, dtype=np.float64)
        workspace = np.empty((states, states), dtype=np.float64)
        total = 0.0
        for first in range(states):
            for second in range(states):
                value = initial[candidate, first, second]
                posterior[candidate, 0, first, second] = value
                total += value
        for first in range(states):
            for second in range(states):
                posterior[candidate, 0, first, second] /= total
        for boundary in range(boundaries):
            scale = 0.0
            for first in range(states):
                for second in range(states):
                    scale = max(scale, right[candidate, boundary, first, second])
            if scale <= 0.0 or not np.isfinite(scale):
                valid[candidate] = False
                break
            # K(u, v) is unchanged by a common rescaling of r(v): keep the scaled r in the next
            # output slice, then reuse the one S x S workspace for the denominator and the ratio.
            for first in range(states):
                for second in range(states):
                    posterior[candidate, boundary + 1, first, second] = (
                        right[candidate, boundary, first, second] / scale)
            _hamming_contract(posterior[candidate, boundary + 1], same[boundary], one[boundary],
                              two[boundary], workspace, rows, columns)
            for first in range(states):
                for second in range(states):
                    denominator = workspace[first, second]
                    if denominator <= 0.0 or not np.isfinite(denominator):
                        valid[candidate] = False
                    else:
                        workspace[first, second] = (posterior[candidate, boundary, first, second]
                                                    / denominator)
            if not valid[candidate]:
                break
            _hamming_contract(workspace, same[boundary], one[boundary], two[boundary],
                              posterior[candidate, boundary + 1], rows, columns)
            total = 0.0
            for first in range(states):
                for second in range(states):
                    value = (posterior[candidate, boundary + 1, first, second]
                             * (right[candidate, boundary, first, second] / scale))
                    posterior[candidate, boundary + 1, first, second] = value
                    total += value
            if total <= 0.0 or not np.isfinite(total):
                valid[candidate] = False
                break
            for first in range(states):
                for second in range(states):
                    posterior[candidate, boundary + 1, first, second] /= total
    return posterior, valid


def infer_factors(log_emissions: np.ndarray, transition: HammingTransition,
                  available: np.ndarray) -> SourceFactors:
    """Source factors from the painting's binned log emissions float64 [N, S*S, B].

    ``available`` bool [N]: samples with an informative painting site; the others keep a uniform
    initial law. Every available sample must keep supported mass (else FloatingPointError).
    """
    candidates, diplotypes, bins = log_emissions.shape
    states = int(round(math.sqrt(diplotypes)))
    emission = np.ascontiguousarray(
        np.transpose(log_emissions, (0, 2, 1)).reshape(candidates, bins, states, states))
    initial, right, valid = _factor_kernel(
        emission, np.ascontiguousarray(transition.same), np.ascontiguousarray(transition.one_change),
        np.ascontiguousarray(transition.two_changes), np.ascontiguousarray(available))
    if np.any(available & ~valid):
        raise FloatingPointError("candidate source factor inference lost all supported mass")
    return SourceFactors(initial, right, np.ascontiguousarray(available))


def marginals(factors: SourceFactors, transition: HammingTransition) -> np.ndarray:
    """float64 [N, B, S, S] exact per-bin posterior marginals of every sample's source chain.

    Evidence-free samples also get normalised marginals; exposure gates them by availability.
    """
    posterior, valid = _marginal_kernel(
        factors.initial, factors.right_weight, np.ascontiguousarray(transition.same),
        np.ascontiguousarray(transition.one_change), np.ascontiguousarray(transition.two_changes))
    if np.any(~valid):
        raise FloatingPointError("source transition has a state with zero supported mass")
    return posterior
