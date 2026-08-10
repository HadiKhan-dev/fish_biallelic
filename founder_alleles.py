"""Shared deterministic founder-allele conversion primitives.

The pipeline stores founder haplotypes either as hard allele vectors or as
per-site allele probabilities.  These helpers provide the single conversion
rule used by painting, phase correction, pedigree inference, and map building:
two-dimensional inputs are concretized with ``argmax(axis=1)``.
"""

import numpy as np


def hard_alleles(haplotype, dtype=np.int8):
    """Return one deterministic allele per site with the requested dtype."""
    values = np.asarray(haplotype)
    if values.ndim == 2:
        values = np.argmax(values, axis=1)
    return values.astype(dtype, copy=False)


def founder_allele_matrix(haplotypes, n_sites, dtype=np.int8,
                          empty_shape=None):
    """Build a founder-ID-indexed dense allele matrix.

    ``empty_shape`` exists for compatibility with the two historical empty
    conventions: dense-block callers return ``(0, 0)``, while lookup callers
    retain a sentinel row with shape ``(1, n_sites)``.
    """
    if not haplotypes:
        shape = (1, n_sites) if empty_shape is None else empty_shape
        return np.full(shape, -1, dtype=dtype)

    matrix = np.full((max(haplotypes) + 1, n_sites), -1, dtype=dtype)
    for founder_id, haplotype in haplotypes.items():
        matrix[founder_id, :] = hard_alleles(haplotype, dtype=dtype)
    return matrix


def founder_block_to_dense(founder_block):
    """Return the historical dense ``(alleles, positions)`` representation."""
    positions = np.asarray(founder_block.positions)
    alleles = founder_allele_matrix(
        founder_block.haplotypes,
        len(positions),
        dtype=np.int8,
        empty_shape=(0, 0),
    )
    return alleles, positions
