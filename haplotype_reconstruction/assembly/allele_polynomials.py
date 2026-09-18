"""Exact binary dosage polynomials for shared-allele completion."""
import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def add_dosage_term(constant, unary, coupling, first, second, fixed, bit_index,
                    f0, f1, f2):
    """Absorb known bits; a repeated founder is one shared binary variable."""
    a, b = bit_index[first], bit_index[second]
    if first == second:
        if a >= 0:
            constant += f0
            unary[a] += f2 - f0
        else:
            constant += f2 if fixed[first] else f0
        return constant
    linear = f1 - f0
    interaction = f2 - 2.0 * f1 + f0
    constant += f0
    if a >= 0:
        unary[a] += linear
    else:
        constant += linear * fixed[first]
    if b >= 0:
        unary[b] += linear
    else:
        constant += linear * fixed[second]
    if a >= 0 and b >= 0:
        coupling[a, b] += interaction
        coupling[b, a] += interaction
    elif a >= 0:
        unary[a] += interaction * fixed[second]
    elif b >= 0:
        unary[b] += interaction * fixed[first]
    else:
        constant += interaction * fixed[first] * fixed[second]
    return constant


@njit(cache=True)
def quadratic_values(constant, unary, coupling):
    """Evaluate all binary configurations in O(2**U), in little-endian order.

    On inserting bit j, the difference between its two halves is a linear
    function of the preceding bits. Build that linear table by doubling,
    then add it to the previous quadratic table. Summing the geometric
    table sizes costs O(2**U), without a long Gray-path rounding recurrence.
    """
    count = 1 << len(unary)
    result = np.empty(count)
    result[0] = constant
    for bit in range(len(unary)):
        width = 1 << bit
        difference = np.empty(width)
        difference[0] = unary[bit]
        for previous in range(bit):
            half = 1 << previous
            for code in range(half):
                difference[half + code] = difference[code] + coupling[bit, previous]
        for code in range(width):
            result[width + code] = result[code] + difference[code]
    return result


@njit(cache=True)
def shared_sample_likelihoods(evidence, weights, pairs, fixed, unknown):
    """O(N K² + N 2**U), except positive-sum fallbacks near cancellation.

    The public result needs sample/configuration likelihoods for its carrier
    sensitivity check. Near-zero masses must not become positive through
    cancellation, particularly when robust mixture is explicitly zero.
    """
    bit_index = np.full(len(fixed), -1, dtype=np.int64)
    for bit in range(len(unknown)):
        bit_index[unknown[bit]] = bit
    result = np.empty((1 << len(unknown), len(evidence)))
    for sample in range(len(evidence)):
        unary = np.zeros(len(unknown))
        coupling = np.zeros((len(unknown), len(unknown)))
        constant = 0.0
        for state in range(len(pairs)):
            first, second = pairs[state]
            mass = weights[sample, state]
            constant = add_dosage_term(
                constant, unary, coupling, first, second, fixed, bit_index,
                mass * evidence[sample, 0], mass * evidence[sample, 1],
                mass * evidence[sample, 2])
        values = quadratic_values(constant, unary, coupling)
        scale = np.max(evidence[sample])
        for code in range(len(values)):
            value = values[code]
            if value < 1e-10 * scale:
                value = 0.0
                for state in range(len(pairs)):
                    first, second = pairs[state]
                    a, b = bit_index[first], bit_index[second]
                    x = ((code >> a) & 1) if a >= 0 else fixed[first]
                    y = ((code >> b) & 1) if b >= 0 else fixed[second]
                    value += weights[sample, state] * evidence[sample, x + y]
            result[code, sample] = value
    return result


@njit(cache=True)
def binary_dosage_marginals(probability, bits):
    """All unary/pair dosage masses in O(2**U + U²), by positive folding.

    Integrate out the highest remaining bit. Its two conditional tables give
    its pairs with every lower bit by successive half-table folds. Work is
    geometric in table size, without subtracting nearly equal probabilities.
    """
    unary = np.zeros((bits, 2))
    pair = np.zeros((bits, bits, 3))
    marginal = probability.copy()
    for high in range(bits - 1, -1, -1):
        half = 1 << high
        zero = marginal[:half].copy()
        one = marginal[half:2 * half].copy()
        unary[high, 0] = np.sum(zero)
        unary[high, 1] = np.sum(one)
        for low in range(high - 1, -1, -1):
            width = 1 << low
            pair[low, high, 0] = np.sum(zero[:width])
            pair[low, high, 1] = (np.sum(zero[width:2 * width])
                                  + np.sum(one[:width]))
            pair[low, high, 2] = np.sum(one[width:2 * width])
            for index in range(width):
                zero[index] += zero[index + width]
                one[index] += one[index + width]
        for index in range(half):
            marginal[index] += marginal[index + half]
    return unary, pair
