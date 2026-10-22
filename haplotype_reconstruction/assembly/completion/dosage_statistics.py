"""Three-dosage sufficient statistics of the cross-fitted completion EM.

Collapse samples before visiting configurations, W[s,l,d]=sum_n z[n,s] log E[n,l,d],
and fold configuration mass before the reverse contraction. Each iteration costs
O(N L K^2 + sum_l(K^2+C_l*r_l^2)). Cubes have <=1,024 configurations; larger
supports use radius-three sparse flip sets. Neither work nor storage contains
C_l*K for a growing ball. The scientific support and update schedule are unchanged.
"""
import math
import numpy as np
from numba import njit
from ..founder_alleles.polynomials import binary_dosage_marginals, pair_quadratic, quadratic_values
from .support import ball_sums


def prepare(configurations, log_priors, log_emission, n_founders, support_radius):
    """Pack centres/bit maps once per site and <=r flips per configuration."""
    counts = [len(value) for value in configurations]
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    centres = np.empty((len(counts), n_founders), np.int8)
    bit_index = np.full((len(counts), n_founders), -1, np.int64)
    width = max((value.flips.shape[1] for value in configurations), default=0)
    flips = np.full((int(offsets[-1]), width), -1, np.int64)
    sizes = np.zeros(int(offsets[-1]), np.int64)
    for site, value in enumerate(configurations):
        centres[site] = value.centre
        bit_index[site, value.varying] = np.arange(len(value.varying))
        start, stop = offsets[site:site + 2]
        if value.radius:
            flips[start:stop, :value.flips.shape[1]] = value.flips
            sizes[start:stop] = value.sizes
    prior = np.concatenate(log_priors) if log_priors else np.empty(0)
    evidence = np.ascontiguousarray(log_emission.reshape(len(log_emission), -1))
    cube = np.ascontiguousarray(np.asarray(support_radius) == 0)
    return evidence, centres, bit_index, flips, sizes, offsets, prior, cube


@njit(cache=True)
def _ball_site_dosages(out, site, centres, indices, flips, sizes, pairs, start, stop, probability):
    """Single/joint flip masses in O(C*r^2+K^2), not C*K."""
    bits = int(np.max(indices[site])) + 1
    mass = probability[start:stop]
    changed, stayed, _ = ball_sums(mass, flips[start:stop], sizes[start:stop], bits, -1, False)
    joint = np.zeros((bits, bits))
    for row in range(start, stop):
        for a in range(sizes[row]):
            first = flips[row, a]
            for b in range(a + 1, sizes[row]):
                joint[first, flips[row, b]] += probability[row]
    total = np.sum(mass)
    for state in range(len(pairs)):
        first, second = pairs[state]
        a, b = indices[site, first], indices[site, second]
        x, y = centres[site, first], centres[site, second]
        if a < 0 and b < 0:
            out[3 * site + x + y, state] = total
        elif first == second:
            out[3 * site + 2 * x, state] = stayed[a]
            out[3 * site + 2 * (1 - x), state] = changed[a]
        elif a < 0 or b < 0:
            bit = b if a < 0 else a
            current = y if a < 0 else x
            fixed = x if a < 0 else y
            out[3 * site + fixed + current, state] = stayed[bit]
            out[3 * site + fixed + 1 - current, state] = changed[bit]
        else:
            both = joint[min(a, b), max(a, b)]
            # The old table also formed its centre cell by subtracting moved
            # masses. Clamp roundoff at zero; mathematical pair masses agree.
            out[3 * site + x + y, state] += max(total - changed[a] - changed[b] + both, 0.0)
            out[3 * site + 1 - x + y, state] += max(changed[a] - both, 0.0)
            out[3 * site + x + 1 - y, state] += max(changed[b] - both, 0.0)
            out[3 * site + 2 - x - y, state] += both


@njit(cache=True)
def dosage_marginals(centres, indices, flips, sizes, pairs, offsets, probability, cube):
    """float64[3L,S] dosage probabilities, preserving every support member."""
    out = np.zeros((3 * len(cube), len(pairs)))
    for site in range(len(cube)):
        start, stop = offsets[site], offsets[site + 1]
        if not cube[site]:
            _ball_site_dosages(out, site, centres, indices, flips, sizes, pairs, start, stop, probability)
            continue
        bits = int(np.max(indices[site])) + 1
        unary, pair = binary_dosage_marginals(probability[start:stop], bits)
        total = np.sum(probability[start:stop])
        for state in range(len(pairs)):
            first, second = pairs[state]
            a, b = indices[site, first], indices[site, second]
            if a < 0 and b < 0:
                dosage = centres[site, first] + centres[site, second]
                out[3 * site + dosage, state] = total
            elif first == second:
                out[3 * site, state] = unary[a, 0]
                out[3 * site + 2, state] = unary[a, 1]
            elif a < 0 or b < 0:
                bit = b if a < 0 else a
                fixed = centres[site, first] if a < 0 else centres[site, second]
                out[3 * site + fixed, state] = unary[bit, 0]
                out[3 * site + fixed + 1, state] = unary[bit, 1]
            else:
                for dosage in range(3):
                    out[3 * site + dosage, state] = pair[min(a, b), max(a, b), dosage]
    return out


@njit(cache=True)
def _ball_values(constant, unary, coupling, centre, flips, sizes):
    bits = len(unary)
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
    out = np.empty(len(sizes))
    for row in range(len(sizes)):
        score = value
        for a in range(sizes[row]):
            bit = flips[row, a]
            sign = 1.0 - 2.0 * centre[bit]
            score += sign * field[bit]
            for d in range(a):
                other = flips[row, d]
                score += sign * (1.0 - 2.0 * centre[other]) * coupling[bit, other]
        out[row] = score
    return out


@njit(cache=True)
def configuration_probabilities(statistics, centres, indices, flips, sizes, pairs, offsets, log_prior, cube):
    """Exact conditional update from dosage statistics float64[S,3L]."""
    out = np.empty(len(log_prior))
    for site in range(len(cube)):
        start, stop = offsets[site], offsets[site + 1]
        if stop == start + 1:
            out[start] = 1.0
            continue
        bits = int(np.max(indices[site])) + 1
        unary = np.zeros(bits)
        coupling = np.zeros((bits, bits))
        constant = pair_quadratic(statistics[:, 3 * site:3 * site + 3], pairs, centres[site],
                                  indices[site], unary, coupling)
        if cube[site]:
            values = quadratic_values(constant, unary, coupling)
        else:
            centre = np.empty(bits)
            for founder in range(centres.shape[1]):
                bit = indices[site, founder]
                if bit >= 0:
                    centre[bit] = centres[site, founder]
            values = _ball_values(constant, unary, coupling, centre, flips[start:stop], sizes[start:stop])
        maximum = -np.inf
        for row in range(start, stop):
            value = log_prior[row] + values[row - start]
            out[row] = value
            maximum = max(maximum, value)
        total = 0.0
        for row in range(start, stop):
            out[row] = math.exp(out[row] - maximum)
            total += out[row]
        for row in range(start, stop):
            out[row] /= total
    return out
