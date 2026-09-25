"""Compact predictive emissions for hard or unobserved founder alleles.

Unknown alleles have a fixed Bernoulli(1/2) predictive distribution, never a
fitted or released value. Distinct founders are marginalized independently;
two copies of the same founder share its allele (an unknown homozygote cannot
produce a heterozygous genotype). This is a per-observation predictive
approximation, not exact joint integration of unknown alleles across samples.

Seven genotype distributions suffice: hard dosages 0/1/2, 0+?, 1+?, two
distinct unknown founders, and two copies of one unknown founder. Storage is
O(N L + L K²), without an N L K² floating-point tensor.
"""
import math
import numpy as np
from numba import njit, prange


@njit(inline="always")
def pair_code(first, second, same_founder):
    if first >= 0 and second >= 0:
        return first + second
    if first >= 0:
        return 3 + first
    if second >= 0:
        return 3 + second
    return 6 if same_founder else 5


@njit(cache=True, parallel=True)
def pair_codes(alleles, source_rows):
    founders, sites = alleles.shape
    result = np.empty((sites, founders * founders), np.uint8)
    for site in prange(sites):
        for first in range(founders):
            for second in range(founders):
                result[site, first * founders + second] = pair_code(
                    alleles[first, site], alleles[second, site],
                    source_rows[first, site] == source_rows[second, site])
    return result


@njit(inline="always")
def predictive_likelihood(p0, p1, p2, code):
    if code == 0:
        return p0
    if code == 1:
        return p1
    if code == 2:
        return p2
    if code == 3:
        return .5 * (p0 + p1)
    if code == 4:
        return .5 * (p1 + p2)
    if code == 5:
        return .25 * p0 + .5 * p1 + .25 * p2
    return .5 * (p0 + p2)


@njit(cache=True, parallel=True, nogil=True)
def compact_log_emissions(samples, epsilon):
    result = np.empty((samples.shape[0], samples.shape[1], 7), np.float32)
    for sample in prange(samples.shape[0]):
        for site in range(samples.shape[1]):
            p0, p1, p2 = samples[sample, site]
            for code in range(7):
                value = predictive_likelihood(p0, p1, p2, code)
                result[sample, site, code] = math.log(max(
                    (1.0 - epsilon) * value + epsilon / 3.0, 1e-300))
    return result


@njit(cache=True, parallel=True)
def binned_log_emissions(samples, alleles, source_rows, kept, snps_per_bin):
    """Partial counterpart of the existing panel scorer, with its -2 floor.

    All states use the same retained sites and bin geometry. Uniform sample
    evidence and sites at which every founder is unknown remain neutral.
    The floor is applied after marginalization, not to individual genotypes.
    """
    founders, sites = alleles.shape
    bins = (sites + snps_per_bin - 1) // snps_per_bin
    result = np.zeros((len(samples), founders, founders, bins), np.float64)
    center = math.log(1.0 / 3.0)
    for sample in prange(len(samples)):
        logs = np.empty(7, np.float64)
        for site in range(sites):
            if not kept[site]:
                continue
            p0, p1, p2 = samples[sample, site]
            total = p0 + p1 + p2
            if total <= 0.0 or (p0 == p1 and p1 == p2):
                continue
            p0, p1, p2 = p0 / total, p1 / total, p2 / total
            for code in range(7):
                value = .99 * predictive_likelihood(p0, p1, p2, code) + .01 / 3.0
                logs[code] = max(math.log(value), -2.0) - center
            for first in range(founders):
                for second in range(first, founders):
                    code = pair_code(alleles[first, site], alleles[second, site],
                                     source_rows[first, site] == source_rows[second, site])
                    value = logs[code]
                    result[sample, first, second, site // snps_per_bin] += value
                    if first != second:
                        result[sample, second, first, site // snps_per_bin] += value
    return result


def source_row_ids(block):
    """Preserve shared unknown alleles when super-paths reuse a local row.

    Two different chromosome paths may traverse the same atomic founder row.
    Its missing allele is shared, just as for a diagonal local diplotype.
    Position columns are compared only within one atomic block.
    """
    from .paths import missing_aware_atomic_source_provenance
    rows, _, counts, _ = missing_aware_atomic_source_provenance(block)
    return np.ascontiguousarray(np.repeat(rows, counts, axis=1), dtype=np.int32)
