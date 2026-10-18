"""The linker's observation model: the pair-class table of a block and its log emissions.

Founder pairs are classified by maths.emissions.pair_code (seven predictive classes; unknown alleles are
Bernoulli(1/2), two copies of one latent allele share it) and scored by maths.emissions.predictive_likelihood mixed
with the linker's robustness (no log floor). Storage is O(N L + L K^2) (a code per site and ordered pair, a log
likelihood per sample, site and code), never an N L K^2 floating-point tensor.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ...maths.emissions import pair_code, predictive_likelihood


@njit(cache=True, parallel=True)
def pair_codes(alleles, source_rows):
    """uint8[L, K * K] code of every ordered founder pair (a, b) at column a * K + b of each site.

    alleles: int8[K, L] pre-fill founder alleles (-1 = unknown). source_rows: int[K, L] the latent row each
    founder copies at each site (equal rows share an unknown allele).
    """
    founders, sites = alleles.shape
    result = np.empty((sites, founders * founders), np.uint8)
    for site in prange(sites):
        for first in range(founders):
            for second in range(founders):
                result[site, first * founders + second] = pair_code(
                    alleles[first, site], alleles[second, site],
                    source_rows[first, site] == source_rows[second, site])
    return result


@njit(cache=True, parallel=True, nogil=True)
def log_emissions(likelihoods, robustness, codes):
    """float32[S, L, codes] log emission of every sample, site and pair class (the linker's observation model).

    likelihoods: float32[S, L, 3] genotype likelihoods of the batch's informative samples (the neutral view:
        exactly 1/3 where a sample is unobserved; not renormalised). robustness r: each class's predictive
        likelihood is mixed with a uniform genotype, (1 - r) P + r / 3 (no further log floor: the mixture keeps
        incompatible observations finite while preserving their evidence). codes: 3 for a fully called block,
        7 otherwise (the first three classes are identical in both).
    """
    samples, sites, _ = likelihoods.shape
    result = np.empty((samples, sites, codes), np.float32)
    keep = 1.0 - robustness
    uniform = robustness / 3.0
    for sample in prange(samples):
        for site in range(sites):
            p0, p1, p2 = likelihoods[sample, site]
            for code in range(codes):
                result[sample, site, code] = math.log(max(
                    keep * predictive_likelihood(p0, p1, p2, code) + uniform, 1e-300))
    return result
