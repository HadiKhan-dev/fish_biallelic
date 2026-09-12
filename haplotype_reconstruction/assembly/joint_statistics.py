"""Three-dosage sufficient statistics for joint founder-allele completion.

For diplotype state s and founder configuration c, the observation depends only
on d(c,s) in {0,1,2}. Collapse the sample sum before visiting configurations:
W[s,l,d] = sum_n z[n,s] log_evidence[n,l,d]. Conversely, form each state's
dosage probabilities before multiplying by sample evidence. Per iteration this
costs O(N*L*D + L*D + sum_l C_l), for D=K*(K+1)/2 and
C_l=2**U_l. Quadratic allele polynomials and positive-only marginal folding
remove the former D*C_l configuration/state cross product.
No independence, allele, likelihood, prior or convergence assumption changes.
"""
import math
import numpy as np
from numba import njit
from .allele_polynomials import add_dosage_term, quadratic_values, binary_dosage_marginals


def prepare(configurations, log_priors, log_emission, n_founders):
    """Pack existing site configurations without a configurations-by-D tensor."""
    counts = [0 if values is None else len(values) for values in configurations]
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    present = [values for values in configurations if values is not None]
    alleles = (np.concatenate(present) if present
               else np.empty((0, n_founders), dtype=np.int8))
    priors = [values for values in log_priors if values is not None]
    log_prior = np.concatenate(priors) if priors else np.empty(0)
    evidence = np.ascontiguousarray(log_emission.reshape(len(log_emission), -1))
    return evidence, np.ascontiguousarray(alleles), offsets, log_prior


@njit(cache=True)
def dosage_marginals(alleles, pairs, offsets, probability):
    """Rows are (site, dosage), columns are unordered diplotypes."""
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
        bit_index = np.full(alleles.shape[1], -1, dtype=np.int64)
        bits = 0
        for founder in range(alleles.shape[1]):
            if alleles[start, founder] != alleles[stop - 1, founder]:
                bit_index[founder] = bits
                bits += 1
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
def configuration_probabilities(statistics, alleles, pairs, offsets, log_prior):
    """Exact conditional founder-configuration update from dosage statistics."""
    out = np.empty(len(log_prior))
    for site in range(len(offsets) - 1):
        start, stop = offsets[site], offsets[site + 1]
        if start == stop:
            continue
        if stop == start + 1:
            out[start] = 1.0
            continue
        bit_index = np.full(alleles.shape[1], -1, dtype=np.int64)
        bits = 0
        for founder in range(alleles.shape[1]):
            if alleles[start, founder] != alleles[stop - 1, founder]:
                bit_index[founder] = bits
                bits += 1
        unary = np.zeros(bits)
        coupling = np.zeros((bits, bits))
        constant = 0.0
        for state in range(len(pairs)):
            first, second = pairs[state]
            constant = add_dosage_term(
                constant, unary, coupling, first, second, alleles[start], bit_index,
                statistics[state, 3 * site], statistics[state, 3 * site + 1],
                statistics[state, 3 * site + 2])
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
