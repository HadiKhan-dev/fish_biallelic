"""Exact binary-dosage polynomials in unresolved founder alleles.

At one site a sample's genotype likelihood under an unordered founder pair (a, b) depends only
on the dosage x_a + x_b in {0, 1, 2}. Summed over pair states with weights, the likelihood of an
assignment b in {0, 1}^U of the U unresolved founder alleles is a quadratic polynomial

    Q(b) = c + sum_i u_i b_i + sum_{i<j} v_ij b_i b_j,

where called founders enter as constants and a founder that appears twice in a pair is one
shared binary variable (x_a = x_b). :func:`pair_quadratic` builds (c, u, v) from per-state
dosage terms, :func:`quadratic_values` evaluates every assignment in O(2^U) by doubling, and
:func:`binary_dosage_marginals` folds a probability table over assignments into its unary and
pair dosage masses without subtracting nearly equal numbers.

Conventions: unresolved founder ``unknown[t]`` is bit t; assignment codes are little-endian
(bit t of the code is founder ``unknown[t]``'s allele); ``fixed`` holds the called alleles
(int8 0/1, ignored at unresolved founders); ``pairs`` is int64[S, 2] unordered founder pairs.
"""
import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def add_dosage_term(constant, unary, coupling, first, second, fixed, bit_index, f0, f1, f2):
    """Add one pair state's dosage term f_d (d = dosage 0, 1, 2) to the quadratic (c, u, v).

    Known bits are absorbed into the constant and the unary terms; a repeated founder is one
    shared binary variable. ``unary`` and ``coupling`` are updated in place; returns c.
    """
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
def unknown_bit_index(founders, unknown):
    """int64[founders]: the bit of each unresolved founder (``unknown[t]`` -> t), -1 if called."""
    bit_index = np.full(founders, -1, dtype=np.int64)
    for bit in range(len(unknown)):
        bit_index[unknown[bit]] = bit
    return bit_index


@njit(cache=True)
def pair_quadratic(terms, pairs, fixed, bit_index, unary, coupling):
    """The quadratic sum_s Q_s(b) of unordered pair states s (the one builder of this shape).

    terms: float64[S, 3] the term of state s at dosage d (e.g. its weight times the sample's
    genotype likelihood of d). ``unary`` (float64[U]) and ``coupling`` (float64[U, U]) must be
    zero on entry and receive u and v (symmetric); returns c. States are added in order.
    """
    constant = 0.0
    for state in range(len(pairs)):
        first, second = pairs[state]
        constant = add_dosage_term(
            constant, unary, coupling, first, second, fixed, bit_index,
            terms[state, 0], terms[state, 1], terms[state, 2])
    return constant


@njit(cache=True)
def weighted_terms(weights, evidence, out):
    """out[s, d] = weights[s] * evidence[d]: the dosage terms of one sample (in place)."""
    for state in range(len(weights)):
        mass = weights[state]
        out[state, 0] = mass * evidence[0]
        out[state, 1] = mass * evidence[1]
        out[state, 2] = mass * evidence[2]
    return out


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
    """float64[2^U, N]: each sample's likelihood sum_s w_ns P(e_n | dosage_s(b)) of every
    assignment b of the unresolved founders, in O(N K^2 + N 2^U).

    evidence: float64[N, 3] genotype likelihoods; weights: float64[N, S] pair-state weights.
    Values below 1e-10 x max(evidence of the sample) are recomputed as positive sums, so
    near-zero masses never become positive (or negative) through cancellation, in particular
    when the robust mixture is zero.
    """
    bit_index = unknown_bit_index(len(fixed), unknown)
    result = np.empty((1 << len(unknown), len(evidence)))
    terms = np.empty((len(pairs), 3))
    for sample in range(len(evidence)):
        unary = np.zeros(len(unknown))
        coupling = np.zeros((len(unknown), len(unknown)))
        constant = pair_quadratic(weighted_terms(weights[sample], evidence[sample], terms),
                                  pairs, fixed, bit_index, unary, coupling)
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

    probability: float64[2^bits] over little-endian assignments. Returns unary float64[bits, 2]
    (mass of allele 0 / 1 of each bit) and pair float64[bits, bits, 3] (pair[low, high, d]:
    mass of dosage d = b_low + b_high, filled for low < high).

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
