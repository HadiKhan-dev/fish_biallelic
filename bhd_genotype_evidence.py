"""Raw allele-depth genotype evidence shared by stage-1 discovery paths."""

import numpy as np
from bhd_config import DEFAULT_READ_ERROR_PROBABILITY


def allele_depths_to_raw_genotype_likelihoods(
    allele_depths,
    read_error_probability=DEFAULT_READ_ERROR_PROBABILITY,
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

    ref = counts[..., 0].astype(np.float64, copy=False)
    alt = counts[..., 1].astype(np.float64, copy=False)
    alt_probability = np.asarray(
        [read_error_probability, 0.5, 1.0 - read_error_probability],
        dtype=np.float64,
    )
    log_likelihood = (
        ref[..., None] * np.log1p(-alt_probability)[None, None, :]
        + alt[..., None] * np.log(alt_probability)[None, None, :]
    )
    log_likelihood -= np.max(log_likelihood, axis=2, keepdims=True)
    likelihood = np.exp(log_likelihood)
    likelihood /= np.sum(likelihood, axis=2, keepdims=True)
    likelihood[(ref + alt) == 0.0] = 1.0 / 3.0
    return np.ascontiguousarray(likelihood)


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


__all__ = [
    "allele_depths_to_raw_genotype_likelihoods",
    "validate_normalized_genotype_evidence",
]
