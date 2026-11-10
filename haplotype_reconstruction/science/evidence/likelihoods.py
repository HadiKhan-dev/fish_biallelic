"""Apply the observed-allele-depth model and validate normalized genotype likelihoods."""

import numpy as np
from numba import prange, types

from haplotype_reconstruction.core.kernels import kernel

from .parameters import ROW_SUM_TOLERANCE as _ROW_SUM_TOLERANCE
from .read_model.emission import cell_probabilities, prefix


@kernel(
    overloads=(
        (
            types.Array(types.int32, 3, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
            types.int64,
            types.Array(types.float64, 3, "C", readonly=False),
        ),
        (
            types.Array(types.int32, 3, "C", readonly=True),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
            types.int64,
            types.Array(types.float64, 3, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def likelihoods_into(counts, error, heterozygote, het_inverse, lookup_depth, out):
    """Apply the same lookup/direct kernels with one lookup table per active sample.

    Prefix sums and cell normalisation have the same evaluation order as
    the evidence read model. Only table lifetime changes: O(team * cap**2)
    scratch instead of O(samples * cap**2), with no second contig-sized output.
    """
    n, length = counts.shape[:2]
    for sample in prange(n):
        deepest = 0
        for site in range(length):
            depth = int(counts[sample, site, 0]) + int(counts[sample, site, 1])
            deepest = max(deepest, depth)
        cap = min(lookup_depth, deepest)
        hom = prefix(error[sample], 0.0, deepest)
        het = prefix(heterozygote[sample], het_inverse, deepest)
        table = np.empty((cap + 1, cap + 1, 3), np.float64)
        table[0, 0, :] = 1.0 / 3.0
        for depth in range(1, cap + 1):
            for alt in range(depth + 1):
                table[depth, alt, 0], table[depth, alt, 1], table[depth, alt, 2] = cell_probabilities(
                    hom, het, alt, depth - alt, depth
                )
        for site in range(length):
            ref = int(counts[sample, site, 0])
            alt = int(counts[sample, site, 1])
            depth = ref + alt
            if depth <= cap:
                for genotype in range(3):
                    out[sample, site, genotype] = table[depth, alt, genotype]
            else:
                out[sample, site, 0], out[sample, site, 1], out[sample, site, 2] = cell_probabilities(
                    hom, het, alt, ref, depth
                )


@kernel(
    overloads=(
        (
            types.Array(types.int32, 3, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
            types.int64,
        ),
        (
            types.Array(types.int32, 3, "C", readonly=True),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
            types.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
def likelihoods_by_sample(counts, error, heterozygote, het_inverse, lookup_depth):
    """Allocate the dense result, then fill it with likelihoods_into."""
    n, length = counts.shape[:2]
    out = np.empty((n, length, 3), np.float64)
    likelihoods_into(counts, error, heterozygote, het_inverse, lookup_depth, out)
    return out


@kernel(
    overloads=(
        (
            types.Array(types.float64, 3, "C", readonly=False),
            types.Array(types.int32, 3, "C", readonly=True),
            types.float64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False),
            types.Array(types.int32, 3, "C", readonly=False),
            types.float64,
        ),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _invalid_likelihood_cells(likelihoods, counts, tolerance):
    """Cells whose likelihoods are not finite, negative, not normalised, or not 1/3 at zero depth."""
    n, length = likelihoods.shape[:2]
    bad = np.zeros(n, np.int64)
    third = 1.0 / 3.0
    for sample in prange(n):
        for site in range(length):
            p0, p1, p2 = likelihoods[sample, site, 0], likelihoods[sample, site, 1], likelihoods[sample, site, 2]
            if counts[sample, site, 0] + counts[sample, site, 1] == 0:
                if p0 != third or p1 != third or p2 != third:
                    bad[sample] += 1
            elif (
                not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(p2))
                or p0 < 0.0
                or p1 < 0.0
                or p2 < 0.0
                or abs(p0 + p1 + p2 - 1.0) > tolerance
            ):
                bad[sample] += 1
    return bad.sum()


@kernel(overloads=((types.Array(types.int32, 3, "C", readonly=True),),), cache=True, nogil=True, parallel=True)
def _observation_mask(counts):
    """One output allocation, without two full-contig comparison temporaries."""
    n, length = counts.shape[:2]
    observed = np.empty((n, length), np.bool_)
    for sample in prange(n):
        for site in range(length):
            observed[sample, site] = counts[sample, site, 0] > 0 or counts[sample, site, 1] > 0
    return observed


__all__ = ["likelihoods_by_sample", "_ROW_SUM_TOLERANCE"]
