"""Independent-homologue boundary propagation without a diploid H^4 matrix."""
import numpy as np
from numba import njit, prange
from.edge_counts import _log_matmul, _MAX_PRODUCT_LOG_RANGE


@njit(cache=True, parallel=True)
def _log_propagation(scores, hap_log_transition):
    """Compute log(T.T @ exp(scores[s]) @ T) in two stable contractions.

    T is the possibly rectangular haploid transition. Each sample is an
    independent parallel unit. Cost is O(N*(H_left^2*H_right +
    H_left*H_right^2)); temporary state is quadratic, not quartic. Neither
    same-copy nor diagonal diplotype states receive a special transition.
    """
    n_samples = scores.shape[0]
    n_left, n_right = hap_log_transition.shape
    result = np.empty((n_samples, n_right * n_right), dtype=np.float64)
    for sample in prange(n_samples):
        intermediate = np.empty((n_right, n_left), dtype=np.float64)
        for c in range(n_right):
            for b in range(n_left):
                largest = -np.inf
                for a in range(n_left):
                    value = scores[sample, a * n_left + b] + hap_log_transition[a, c]
                    largest = max(largest, value)
                if largest == -np.inf:
                    intermediate[c, b] = -np.inf
                else:
                    total = 0.0
                    for a in range(n_left):
                        total += np.exp(scores[sample, a * n_left + b]
                                        + hap_log_transition[a, c] - largest)
                    intermediate[c, b] = largest + np.log(total)
        for c in range(n_right):
            for d in range(n_right):
                largest = -np.inf
                for b in range(n_left):
                    largest = max(largest, intermediate[c, b] + hap_log_transition[b, d])
                if largest == -np.inf:
                    result[sample, c * n_right + d] = -np.inf
                else:
                    total = 0.0
                    for b in range(n_left):
                        total += np.exp(intermediate[c, b] + hap_log_transition[b, d] - largest)
                    result[sample, c * n_right + d] = largest + np.log(total)
    return result


@njit(cache=True, parallel=True)
def _scaled_propagation(scores, hap_log_transition):
    """Same dense contraction with a range-guarded positive BLAS path."""
    n_samples = scores.shape[0]
    left, right = hap_log_transition.shape
    result = np.empty((n_samples, right * right))
    top_t = np.max(hap_log_transition)
    t_range = top_t - np.min(hap_log_transition)
    t = np.exp(hap_log_transition - top_t)
    for sample in prange(n_samples):
        a = scores[sample].reshape((left, left))
        top_a = np.max(a)
        spread = top_a - np.min(a) + 2.0 * t_range
        if np.isfinite(spread) and spread <= _MAX_PRODUCT_LOG_RANGE:
            # BLAS stays single-threaded; samples own Numba parallelism.
            product = t.T @ (np.exp(a - top_a) @ t)
            result[sample] = (
                np.log(product) + top_a + 2.0 * top_t
            ).reshape(right * right)
        else:
            result[sample] = _log_matmul(
                _log_matmul(hap_log_transition.T, a), hap_log_transition
            ).reshape(right * right)
    return result


@njit(cache=True)
def propagate_homologue_priors(scores, hap_log_transition):
    """Exact dense O(N*(L²R + LR²)) propagation with quadratic workspace.

    Small panels retain the low-overhead log kernel. Larger panels use max-
    scaled matrix products only when the complete exponent spread is safe;
    zero transitions and extreme likelihoods keep stable cubic log arithmetic.
    This changes neither the dense transition model nor any likelihood floor.
    """
    if max(hap_log_transition.shape) < 16:
        return _log_propagation(scores, hap_log_transition)
    return _scaled_propagation(scores, hap_log_transition)
