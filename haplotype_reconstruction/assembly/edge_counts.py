"""Cubic posterior homologue edge counts for the shared linker.

For each sample let A[u1,u2], B[v1,v2] be forward and backward boundary
likelihoods and T[u,v] the same forward haploid transition matrix used to
compute them. W1=A T B^t, W2=A^t T B, Z=sum(T*W1).
Expected counts are T*(W1+W2)/Z and sum to two per informative sample.
Both homologues are counted, without assuming swap symmetry or square T.
"""
import math
import numpy as np
from numba import njit, prange

# Bound the total exponent spread of a diploid product, not each factor alone.
_MAX_PRODUCT_LOG_RANGE = 500.0


@njit(cache=True)
def _log_matmul(a, b):
    out = np.full((a.shape[0], b.shape[1]), -np.inf)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            peak = -np.inf
            for k in range(a.shape[1]):
                peak = max(peak, a[i, k] + b[k, j])
            if not np.isfinite(peak):
                continue
            total = 0.0
            for k in range(a.shape[1]):
                total += math.exp(a[i, k] + b[k, j] - peak)
            out[i, j] = peak + math.log(total)
    return out


@njit(cache=True)
def _log_sample_counts(a, b, log_t):
    """Stable cubic path, including zero transitions/impossible sample states."""
    w1 = _log_matmul(_log_matmul(a, log_t), b.T)
    w2 = _log_matmul(_log_matmul(a.T, log_t), b)
    peak = np.max(log_t + w1)
    out = np.full(log_t.shape, -np.inf)
    if not np.isfinite(peak):
        return out
    total = 0.0
    for i in range(log_t.shape[0]):
        for j in range(log_t.shape[1]):
            total += math.exp(log_t[i, j] + w1[i, j] - peak)
    log_z = peak + math.log(total)
    for i in range(log_t.shape[0]):
        for j in range(log_t.shape[1]):
            # Absent transitions had zero diploid mass in the dense reference.
            if np.isfinite(log_t[i, j]):
                out[i, j] = np.logaddexp(w1[i, j], w2[i, j]) - log_z
                out[i, j] += log_t[i, j]
    return out


@njit(cache=True)
def _probability_sample_counts(a, b, log_t):
    """Max-scaled cubic contraction; all reduction ordering is deterministic."""
    left, right = log_t.shape
    top_t = np.max(log_t)
    t = np.exp(log_t - top_t)
    aa = np.exp(a - np.max(a))
    bb = np.exp(b - np.max(b))
    at = np.zeros((left, right))
    att = np.zeros((left, right))
    for i in range(left):
        for k in range(left):
            for j in range(right):
                at[i, j] += aa[i, k] * t[k, j]
                att[i, j] += aa[k, i] * t[k, j]
    w1 = np.zeros((left, right))
    w2 = np.zeros((left, right))
    for i in range(left):
        for j in range(right):
            for k in range(right):
                w1[i, j] += at[i, k] * bb[j, k]
                w2[i, j] += att[i, k] * bb[k, j]
    z = 0.0
    for i in range(left):
        for j in range(right):
            z += t[i, j] * w1[i, j]
    out = np.empty((left, right))
    log_z = math.log(z)
    for i in range(left):
        for j in range(right):
            out[i, j] = math.log(w1[i, j] + w2[i, j]) - log_z
            # Z contains two factors exp(-top_t), W contains one.
            out[i, j] += log_t[i, j] - top_t
    return out


@njit(cache=True, parallel=True)
def homologue_edge_log_counts(s, r, log_t):
    """O(N*(L^2 R + L R^2)) work; O(N L R) output workspace.

    prange partitions samples and then output edges. Neither reduction crosses
    workers, so changing the Numba thread allocation does not reorder sums.
    Extreme inputs use a log-domain cubic contraction, never a quartic fallback.
    """
    samples = s.shape[0]
    left, right = log_t.shape
    per_sample = np.empty((samples, left, right))
    t_range = np.max(log_t) - np.min(log_t)
    for sample in prange(samples):
        a = s[sample].reshape((left, left))
        b = r[sample].reshape((right, right))
        spread = (np.max(a) - np.min(a) + np.max(b) - np.min(b)
                  + 2.0 * t_range)
        if np.isfinite(spread) and spread <= _MAX_PRODUCT_LOG_RANGE:
            per_sample[sample] = _probability_sample_counts(a, b, log_t)
        else:
            per_sample[sample] = _log_sample_counts(a, b, log_t)
    result = np.full((left, right), -np.inf)
    for edge in prange(left * right):
        i, j = edge // right, edge % right
        peak = -np.inf
        for sample in range(samples):
            peak = max(peak, per_sample[sample, i, j])
        if not np.isfinite(peak):
            continue
        total = 0.0
        for sample in range(samples):
            total += math.exp(per_sample[sample, i, j] - peak)
        result[i, j] = peak + math.log(total)
    return result
