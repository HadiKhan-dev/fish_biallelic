"""Pieces of the completion observation model shared by the cross-fitted EM, the cavity fill and the
boundary links (one implementation each).

A sample's two homologues copy an unordered pair of founder rows (a, b), a <= b (a "pair state"); with
founder alleles x its genotype dosage is x_a + x_b, and a homozygous pair (a, a) copies one shared allele.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# Normalised genotype evidence whose range (max - min over the three genotypes) is at most this is
# uniform: the sample is not informative at the site.
UNIFORM_TOLERANCE = 1e-12


def unordered_pairs(founders: int) -> np.ndarray:
    """int64[K(K+1)/2, 2]: the unordered founder pairs (a, b), a <= b, in lexicographic order."""
    return np.asarray(
        [(first, second) for first in range(founders) for second in range(first, founders)],
        dtype=np.int64,
    )


def normalised_evidence(evidence) -> np.ndarray:
    """float64[N, L, 3] genotype likelihoods scaled to sum to one over the three genotypes; exactly
    1/3 where they sum to zero (no evidence)."""
    values = np.asarray(evidence, dtype=np.float64)
    totals = np.sum(values, axis=2, keepdims=True)
    result = np.full(values.shape, 1.0 / 3.0, dtype=np.float64)
    np.divide(values, totals, out=result, where=totals > 0)
    return result


@njit(cache=True)
def carrier_probability(pairs, weights, founders):
    """float64[N, K]: each sample's probability of carrying each founder row, from its pair-state
    probabilities weights float64[N, S] (a homozygous pair counts once). States are added in order."""
    result = np.zeros((len(weights), founders), dtype=np.float64)
    for sample in range(len(weights)):
        for state in range(len(pairs)):
            first, second = pairs[state]
            mass = weights[sample, state]
            result[sample, first] += mass
            if first != second:
                result[sample, second] += mass
    return result


def row_classes(alleles: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Founder rows grouped by identical allele vectors (int8[K, L], -1 unresolved): classes in order of
    their first row, each class's rows ascending. Rows of one class are observationally exchangeable."""
    classes: list[list[int]] = []
    for row in range(alleles.shape[0]):
        for members in classes:
            if np.array_equal(alleles[row], alleles[members[0]]):
                members.append(row)
                break
        else:
            classes.append([row])
    return tuple(tuple(members) for members in classes)
