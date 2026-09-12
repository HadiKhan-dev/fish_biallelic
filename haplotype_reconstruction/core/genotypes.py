"""core / genotypes for the canonical reconstruction pipeline."""
from __future__ import annotations


from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import haplotype_reconstruction.core.config as core_config

# Small discovery blocks avoid thread startup. Chromosome tensors use bounded
# NumPy tiles with identical arithmetic; 64K cells measured better than 16K.
_GL_TILE_CELLS = 65536
_GL_PARALLEL_MIN_CELLS = 1_000_000


def _fill_likelihood_tile(counts, likelihood, log_ref, log_alt, bounds):
    sample, sample_stop, first, last = bounds
    tile = counts[sample:sample_stop, first:last]
    ref = tile[..., 0].astype(np.float64, copy=False)
    alt = tile[..., 1].astype(np.float64, copy=False)
    block = likelihood[sample:sample_stop, first:last]
    block[:] = ref[..., None] * log_ref + alt[..., None] * log_alt
    block -= np.max(block, axis=2, keepdims=True)
    np.exp(block, out=block)
    block /= np.sum(block, axis=2, keepdims=True)
    block[(ref + alt) == 0.0] = 1.0 / 3.0


def allele_depths_to_raw_genotype_likelihoods(
    allele_depths,
    read_error_probability=core_config.DEFAULT_READ_ERROR_PROBABILITY,
    *,
    require_nonempty=False,
    require_integer=False,
):
    """Return normalized ``P(reads | genotype)`` for dosages 0, 1, and 2.

    No population-frequency or HWE prior is applied.  A max-shift followed by
    ``exp`` and a left-to-right NumPy sum fixes one normalization order for all
    callers.  Zero-depth cells are exactly uniform.
    """

    counts = np.asarray(allele_depths)
    if counts.ndim != 3 or counts.shape[2] != 2:
        raise ValueError(
            "allele_depths must have shape (samples, sites, 2)"
        )
    if require_nonempty and (counts.shape[0] < 1 or counts.shape[1] < 1):
        raise ValueError("allele depths must contain samples and sites")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("allele depths must be finite and non-negative")
    if require_integer:
        integer_dtype = np.issubdtype(counts.dtype, np.integer)
        integer_values = np.all(counts == np.floor(counts))
        if not integer_dtype and not integer_values:
            raise ValueError("allele depths must be integer-valued")
    if not 0.0 < read_error_probability < 0.5:
        raise ValueError(
            "read_error_probability must lie strictly between 0 and 0.5"
        )

    alt_probability = np.asarray(
        [read_error_probability, 0.5, 1.0 - read_error_probability],
        dtype=np.float64,
    )
    log_ref = np.log1p(-alt_probability)
    log_alt = np.log(alt_probability)
    likelihood = np.empty(counts.shape[:2] + (3,), dtype=np.float64)
    # Disjoint output slices; each worker uses a few MiB of scratch rather
    # than several whole GL tensors. NumPy retains the same per-cell order.
    tile_sites = max(1, min(counts.shape[1], _GL_TILE_CELLS))
    tile_samples = max(1, _GL_TILE_CELLS // tile_sites)
    tiles = ((sample, min(sample+tile_samples, counts.shape[0]),
              first, min(first+tile_sites, counts.shape[1]))
             for sample in range(0, counts.shape[0], tile_samples)
             for first in range(0, counts.shape[1], tile_sites))
    threads = 1
    if counts.shape[0]*counts.shape[1] >= _GL_PARALLEL_MIN_CELLS:
        from numba import get_num_threads
        from .runtime import available_cpu_count
        # Reuse the caller's active budget, including discovery worker masks.
        # No Numba/BLAS numerical work runs concurrently with this tile pool.
        threads = min(get_num_threads(), available_cpu_count())
    worker = partial(_fill_likelihood_tile, counts, likelihood, log_ref, log_alt)
    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads,
                initializer=partial(np.seterr, **np.geterr())) as pool:
            for _ in pool.map(worker, tiles):
                pass
    else:
        for tile in tiles:
            worker(tile)
    return likelihood



def validate_normalized_genotype_evidence(
    evidence,
    *,
    n_sites=None,
    n_samples=None,
):
    """Validate and return contiguous normalized three-genotype evidence."""

    result = np.ascontiguousarray(evidence, dtype=np.float64)
    if result.ndim != 3 or result.shape[2] != 3:
        raise ValueError("evidence must have shape (samples, sites, 3)")
    if result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("evidence must contain samples and sites")
    if n_sites is not None and result.shape[1] != n_sites:
        raise ValueError("evidence site count does not match")
    if n_samples is not None and result.shape[0] != n_samples:
        raise ValueError("evidence sample count does not match")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("evidence must be finite and non-negative")
    evidence_mass = np.sum(result, axis=2)
    if np.any(evidence_mass <= 0.0):
        raise ValueError("every sample/site must have positive evidence mass")
    if not np.allclose(evidence_mass, 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("evidence must contain normalized genotype likelihoods")
    return result
