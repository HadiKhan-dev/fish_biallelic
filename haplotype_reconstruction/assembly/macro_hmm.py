"""Independent-homologue boundary propagation without a diploid H^4 matrix."""
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def propagate_homologue_priors(scores, hap_log_transition):
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
