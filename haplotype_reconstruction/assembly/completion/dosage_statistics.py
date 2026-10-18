"""Three-dosage sufficient statistics of the cross-fitted completion EM.

For diplotype state s and founder configuration c, the observation depends only
on d(c,s) in {0,1,2}. Collapse the sample sum before visiting configurations:
W[s,l,d] = sum_n z[n,s] log_evidence[n,l,d]. Conversely, form each state's
dosage probabilities before multiplying by sample evidence. Per iteration this
costs O(N*L*D + L*D + sum_l C_l), for D=K*(K+1)/2 and C_l configurations at
site l: 2**U_l for an exhaustive (cube) site, U_l <= 10 unresolved alleles or
an exact core; a Hamming-ball support (founder_alleles.site_posterior) costs
O(D + C_l (U_l + r**2)) with radius r, from local fields around its first
(centre) row. Quadratic allele polynomials and positive-only marginal folding
remove the D*C_l configuration/state cross product.

Packed layout (``prepare``): the configurations of all sites are stacked, int8[C, K] founder
alleles, site l owning rows offsets[l]:offsets[l + 1]; a cube site lists every assignment of its
varying founders in little-endian order (first row all zero, last row all one in those founders).
"""
import math

import numpy as np
from numba import njit

from ..founder_alleles.polynomials import binary_dosage_marginals, pair_quadratic, quadratic_values


def prepare(configurations, log_priors, log_emission, n_founders, support_radius):
    """Pack the site configurations without a configurations-by-D tensor.

    Returns (evidence float64[N, 3L] = log_emission reshaped, alleles int8[C, K], offsets int64[L + 1],
    log_prior float64[C], cube bool[L]); ``support_radius[l]`` is 0 for a cube site (every
    assignment of the bits that vary, little-endian) and the Hamming radius of a ball support.
    """
    counts = [len(values) for values in configurations]
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    alleles = (np.concatenate(configurations) if configurations
               else np.empty((0, n_founders), dtype=np.int8))
    log_prior = np.concatenate(log_priors) if log_priors else np.empty(0)
    evidence = np.ascontiguousarray(log_emission.reshape(len(log_emission), -1))
    cube = np.ascontiguousarray(np.asarray(support_radius) == 0)
    return evidence, np.ascontiguousarray(alleles), offsets, log_prior, cube


@njit(cache=True)
def _varying_bits(alleles, start, stop):
    """Founders whose allele varies over a site's configurations, numbered in founder order."""
    bit_index = np.full(alleles.shape[1], -1, dtype=np.int64)
    bits = 0
    for founder in range(alleles.shape[1]):
        for config in range(start + 1, stop):
            if alleles[config, founder] != alleles[start, founder]:
                bit_index[founder] = bits
                bits += 1
                break
    return bit_index, bits


@njit(cache=True)
def _cube_bits(alleles, start, stop):
    """Varying founders of a cube site: those whose allele differs between its first (all zero)
    and last (all one) assignment, numbered in founder order."""
    bit_index = np.full(alleles.shape[1], -1, dtype=np.int64)
    bits = 0
    for founder in range(alleles.shape[1]):
        if alleles[start, founder] != alleles[stop - 1, founder]:
            bit_index[founder] = bits
            bits += 1
    return bit_index, bits


@njit(cache=True)
def _ball_site_dosages(out, site, alleles, pairs, start, stop, probability):
    """Dosage probabilities of every state over an explicit (ball) configuration list.

    Two-bit tables are accumulated relative to the first (centre) row: a row
    moves its mass only for the bits it flips, so the cost is O(C v r) with v
    varying bits and r flips per row, not O(C v**2).
    """
    bit_index, bits = _varying_bits(alleles, start, stop)
    founders = np.empty(bits, dtype=np.int64)
    for founder in range(alleles.shape[1]):
        if bit_index[founder] >= 0:
            founders[bit_index[founder]] = founder
    centre = np.empty(bits, dtype=np.int64)
    for bit in range(bits):
        centre[bit] = alleles[start, founders[bit]]
    total = 0.0
    for config in range(start, stop):
        total += probability[config]
    flipped_mass = np.zeros(bits)
    pair_cells = np.zeros((bits, bits, 2, 2))
    flipped = np.zeros(bits, dtype=np.bool_)
    flips = np.empty(bits, dtype=np.int64)
    for config in range(start + 1, stop):
        mass = probability[config]
        count = 0
        for bit in range(bits):
            flipped[bit] = alleles[config, founders[bit]] != centre[bit]
            if flipped[bit]:
                flips[count] = bit
                count += 1
        for a in range(count):
            first = flips[a]
            flipped_mass[first] += mass
            for other in range(bits):
                if other == first or (flipped[other] and other < first):
                    continue
                low = min(first, other)
                high = max(first, other)
                x = alleles[config, founders[low]]
                y = alleles[config, founders[high]]
                pair_cells[low, high, x, y] += mass
                pair_cells[low, high, centre[low], centre[high]] -= mass
    for low in range(bits):
        for high in range(low + 1, bits):
            pair_cells[low, high, centre[low], centre[high]] += total
    for state in range(len(pairs)):
        first, second = pairs[state]
        a, b = bit_index[first], bit_index[second]
        if a < 0 and b < 0:
            dosage = alleles[start, first] + alleles[start, second]
            out[3 * site + dosage, state] = total
        elif a >= 0 and b >= 0 and a != b:
            low = min(a, b)
            high = max(a, b)
            out[3 * site, state] = max(pair_cells[low, high, 0, 0], 0.0)
            out[3 * site + 1, state] = max(pair_cells[low, high, 0, 1], 0.0) + max(pair_cells[low, high, 1, 0], 0.0)
            out[3 * site + 2, state] = max(pair_cells[low, high, 1, 1], 0.0)
        else:
            bit = a if a >= 0 else b
            stays = max(total - flipped_mass[bit], 0.0)
            one = flipped_mass[bit] if centre[bit] == 0 else stays
            zero = stays if centre[bit] == 0 else flipped_mass[bit]
            if first == second:
                out[3 * site, state] = zero
                out[3 * site + 2, state] = one
            else:
                fixed = alleles[start, first] if a < 0 else alleles[start, second]
                out[3 * site + fixed, state] = zero
                out[3 * site + fixed + 1, state] = one


@njit(cache=True)
def _ball_site_probabilities(out, statistics, site, alleles, pairs, start, stop, log_prior):
    """Configuration update over an explicit (ball) list from the site's quadratic."""
    bit_index, bits = _varying_bits(alleles, start, stop)
    unary = np.zeros(bits)
    coupling = np.zeros((bits, bits))
    constant = pair_quadratic(statistics[:, 3 * site:3 * site + 3], pairs, alleles[start],
                              bit_index, unary, coupling)
    founders = np.empty(bits, dtype=np.int64)
    for founder in range(alleles.shape[1]):
        if bit_index[founder] >= 0:
            founders[bit_index[founder]] = founder
    centre = np.empty(bits)
    for bit in range(bits):
        centre[bit] = alleles[start, founders[bit]]
    # Value and local fields of the quadratic at the centre row.
    value = constant
    field = np.empty(bits)
    for bit in range(bits):
        acc = unary[bit]
        for other in range(bits):
            if other != bit:
                acc += coupling[bit, other] * centre[other]
        field[bit] = acc
        value += unary[bit] * centre[bit]
        for other in range(bit + 1, bits):
            value += coupling[bit, other] * centre[bit] * centre[other]
    flips = np.empty(bits, dtype=np.int64)
    maximum = -np.inf
    for config in range(start, stop):
        count = 0
        for bit in range(bits):
            if alleles[config, founders[bit]] != centre[bit]:
                flips[count] = bit
                count += 1
        score = value
        for a in range(count):
            bit = flips[a]
            sign = 1.0 - 2.0 * centre[bit]
            score += sign * field[bit]
            for d in range(a):
                other = flips[d]
                score += sign * (1.0 - 2.0 * centre[other]) * coupling[bit, other]
        out[config] = log_prior[config] + score
        maximum = max(maximum, out[config])
    total = 0.0
    for config in range(start, stop):
        out[config] = math.exp(out[config] - maximum)
        total += out[config]
    for config in range(start, stop):
        out[config] /= total


@njit(cache=True)
def dosage_marginals(alleles, pairs, offsets, probability, cube):
    """float64[3L, S]: rows (site, dosage), columns unordered diplotype states: the probability of
    each dosage under each state given the configuration probabilities of every site."""
    out = np.zeros((3 * (len(offsets) - 1), len(pairs)))
    for site in range(len(offsets) - 1):
        start, stop = offsets[site], offsets[site + 1]
        if start == stop:
            continue
        if stop == start + 1:
            for state in range(len(pairs)):
                dosage = alleles[start, pairs[state, 0]] + alleles[start, pairs[state, 1]]
                out[3 * site + dosage, state] = probability[start]
            continue
        if not cube[site]:
            _ball_site_dosages(out, site, alleles, pairs, start, stop, probability)
            continue
        bit_index, bits = _cube_bits(alleles, start, stop)
        unary, pair = binary_dosage_marginals(probability[start:stop], bits)
        total = np.sum(probability[start:stop])
        for state in range(len(pairs)):
            first, second = pairs[state]
            a, b = bit_index[first], bit_index[second]
            if a < 0 and b < 0:
                dosage = alleles[start, first] + alleles[start, second]
                out[3 * site + dosage, state] = total
            elif first == second:
                out[3 * site, state] = unary[a, 0]
                out[3 * site + 2, state] = unary[a, 1]
            elif a < 0 or b < 0:
                bit = b if a < 0 else a
                fixed = alleles[start, first] if a < 0 else alleles[start, second]
                out[3 * site + fixed, state] = unary[bit, 0]
                out[3 * site + fixed + 1, state] = unary[bit, 1]
            else:
                for dosage in range(3):
                    out[3 * site + dosage, state] = pair[min(a, b), max(a, b), dosage]
    return out


@njit(cache=True)
def configuration_probabilities(statistics, alleles, pairs, offsets, log_prior, cube):
    """Exact conditional founder-configuration update from dosage statistics float64[S, 3L]."""
    out = np.empty(len(log_prior))
    for site in range(len(offsets) - 1):
        start, stop = offsets[site], offsets[site + 1]
        if start == stop:
            continue
        if stop == start + 1:
            out[start] = 1.0
            continue
        if not cube[site]:
            _ball_site_probabilities(out, statistics, site, alleles, pairs, start, stop, log_prior)
            continue
        bit_index, bits = _cube_bits(alleles, start, stop)
        unary = np.zeros(bits)
        coupling = np.zeros((bits, bits))
        constant = pair_quadratic(statistics[:, 3 * site:3 * site + 3], pairs, alleles[start],
                                  bit_index, unary, coupling)
        values = quadratic_values(constant, unary, coupling)
        maximum = -np.inf
        for config in range(start, stop):
            value = log_prior[config] + values[config - start]
            out[config] = value
            maximum = max(maximum, value)
        total = 0.0
        for config in range(start, stop):
            out[config] = math.exp(out[config] - maximum)
            total += out[config]
        for config in range(start, stop):
            out[config] /= total
    return out
