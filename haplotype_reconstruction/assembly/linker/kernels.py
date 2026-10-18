"""Cubic kernels of the linker's EM between two blocks (base macro_hmm + edge_counts + M-step smoothing).

A sample's diploid state at a block boundary is an ordered founder pair; the two homologues move
independently through the same haploid transition matrix T (K_left x K_right), so a boundary message is
propagated as T^T A T and the posterior edge counts as T * (A T B^T + A^T T B) / Z, both O(N (K_l^2 K_r +
K_l K_r^2)) without the K^4 diploid matrix. Scaled probability paths with (single-threaded) matrix products
are used only when the complete exponent spread of the product is safe; zero transitions and extreme
likelihoods take the stable log-domain cubic path. Every reduction runs per sample and then per edge in a fixed
order, so Numba's thread count never changes a result.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# Bound on the total exponent spread of a diploid product (not of each factor alone).
_MAX_PRODUCT_LOG_RANGE = 500.0
# Haplotype counts from which propagation (16) and edge counts (32) use matrix products.
_PROPAGATION_PRODUCT_FOUNDERS = 16
_EDGE_PRODUCT_FOUNDERS = 32


@njit(cache=True, nogil=True)
def entrywise_log(probabilities):
    """float64 log of every entry of a matrix (the scalar libm log, identical to Python's math.log; NumPy's
    vectorised log can differ in the last bit)."""
    rows, columns = probabilities.shape
    result = np.empty((rows, columns))
    for i in range(rows):
        for j in range(columns):
            result[i, j] = math.log(probabilities[i, j])
    return result


@njit(cache=True)
def log_matmul(a, b):
    """log(exp(a) @ exp(b)) by a max-shifted sum per output cell (cells without a finite term stay -inf)."""
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


@njit(cache=True, parallel=True, nogil=True)
def _log_propagation(scores, log_transition):
    """log(T^T exp(scores[s]) T) per sample in two stable contractions (quadratic temporary state)."""
    n_samples = scores.shape[0]
    n_left, n_right = log_transition.shape
    result = np.empty((n_samples, n_right * n_right), dtype=np.float64)
    for sample in prange(n_samples):
        intermediate = np.empty((n_right, n_left), dtype=np.float64)
        for c in range(n_right):
            for b in range(n_left):
                largest = -np.inf
                for a in range(n_left):
                    value = scores[sample, a * n_left + b] + log_transition[a, c]
                    largest = max(largest, value)
                if largest == -np.inf:
                    intermediate[c, b] = -np.inf
                else:
                    total = 0.0
                    for a in range(n_left):
                        total += np.exp(scores[sample, a * n_left + b]
                                        + log_transition[a, c] - largest)
                    intermediate[c, b] = largest + np.log(total)
        for c in range(n_right):
            for d in range(n_right):
                largest = -np.inf
                for b in range(n_left):
                    largest = max(largest, intermediate[c, b] + log_transition[b, d])
                if largest == -np.inf:
                    result[sample, c * n_right + d] = -np.inf
                else:
                    total = 0.0
                    for b in range(n_left):
                        total += np.exp(intermediate[c, b] + log_transition[b, d] - largest)
                    result[sample, c * n_right + d] = largest + np.log(total)
    return result


@njit(cache=True, parallel=True, nogil=True)
def _scaled_propagation(scores, log_transition):
    """The same contraction with max-scaled matrix products where the exponent spread is safe."""
    n_samples = scores.shape[0]
    left, right = log_transition.shape
    result = np.empty((n_samples, right * right))
    top_t = np.max(log_transition)
    t_range = top_t - np.min(log_transition)
    t = np.exp(log_transition - top_t)
    for sample in prange(n_samples):
        a = scores[sample].reshape((left, left))
        top_a = np.max(a)
        spread = top_a - np.min(a) + 2.0 * t_range
        if np.isfinite(spread) and spread <= _MAX_PRODUCT_LOG_RANGE:
            # BLAS stays single-threaded; samples own Numba parallelism.
            product = t.T @ (np.exp(a - top_a) @ t)
            result[sample] = (np.log(product) + top_a + 2.0 * top_t).reshape(right * right)
        else:
            result[sample] = log_matmul(log_matmul(log_transition.T, a), log_transition).reshape(right * right)
    return result


@njit(cache=True, nogil=True)
def propagate_pair_priors(scores, log_transition):
    """float64[S, K_r^2] log priors of the ordered pair at the right block from the left block's scores.

    scores: float64[S, K_l^2] (C-contiguous); log_transition: float64[K_l, K_r] log haploid transitions.
    Exact dense O(S (K_l^2 K_r + K_l K_r^2)) propagation with quadratic workspace; neither same-copy nor
    diagonal pair states get a special transition.
    """
    if max(log_transition.shape) < _PROPAGATION_PRODUCT_FOUNDERS:
        return _log_propagation(scores, log_transition)
    return _scaled_propagation(scores, log_transition)


@njit(cache=True)
def _log_sample_counts(a, b, log_t):
    """One sample's log edge counts by the stable cubic log path (zero transitions, impossible states)."""
    w1 = log_matmul(log_matmul(a, log_t), b.T)
    w2 = log_matmul(log_matmul(a.T, log_t), b)
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
            # Absent transitions have zero diploid mass.
            if np.isfinite(log_t[i, j]):
                out[i, j] = np.logaddexp(w1[i, j], w2[i, j]) - log_z
                out[i, j] += log_t[i, j]
    return out


@njit(cache=True)
def _probability_sample_counts(a, b, log_t):
    """One sample's log edge counts by max-scaled cubic contractions (deterministic reduction order)."""
    left, right = log_t.shape
    top_t = np.max(log_t)
    t = np.exp(log_t - top_t)
    aa = np.exp(a - np.max(a))
    bb = np.exp(b - np.max(b))
    if max(left, right) >= _EDGE_PRODUCT_FOUNDERS:
        at = aa @ t
        att = aa.T @ t
        w1 = at @ bb.T
        w2 = att @ bb
    else:
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


@njit(cache=True, parallel=True, nogil=True)
def edge_log_counts(s, r, log_t):
    """float64[K_l, K_r] log posterior homologue edge counts of one boundary, summed over samples.

    s: float64[S, K_l^2] forward scores of the left block, r: float64[S, K_r^2] backward scores of the right
    block, log_t: float64[K_l, K_r] the log transitions both were computed with. For each sample, with A, B the
    boundary likelihood matrices, W1 = A T B^T, W2 = A^T T B, Z = sum(T * W1), the expected counts T * (W1 + W2)
    / Z count both homologues (two per informative sample), without assuming swap symmetry or square T.
    prange partitions samples, then output edges: no reduction crosses workers.
    """
    samples = s.shape[0]
    left, right = log_t.shape
    per_sample = np.empty((samples, left, right))
    t_range = np.max(log_t) - np.min(log_t)
    for sample in prange(samples):
        a = s[sample].reshape((left, left))
        b = r[sample].reshape((right, right))
        spread = (np.max(a) - np.min(a) + np.max(b) - np.min(b) + 2.0 * t_range)
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


@njit(cache=True, nogil=True)
def regularised_rows(log_counts, log_pseudo, min_log_probability, uniform_mix):
    """float64[K_l, K_r] transition probabilities from log edge counts (the M-step target, row by row).

    smoothed = logaddexp(count, log pseudocount); log p = max(smoothed - logsumexp(row), floor);
    p = exp(log p) / sum; final = (1 - mix) p + mix / K_r (robust uniform mixture).
    """
    n_c, n_n = log_counts.shape
    final_p = np.zeros((n_c, n_n), dtype=np.float64)
    if n_n == 0:
        return final_p
    uniform_val = 1.0 / n_n
    for u_i in range(n_c):
        smoothed_row = np.empty(n_n, dtype=np.float64)
        for v_i in range(n_n):
            d = log_counts[u_i, v_i]
            if d == -np.inf:
                smoothed_row[v_i] = log_pseudo
            else:
                m = d if d > log_pseudo else log_pseudo
                smoothed_row[v_i] = m + np.log(np.exp(d - m) + np.exp(log_pseudo - m))
        m = -np.inf
        for v_i in range(n_n):
            if smoothed_row[v_i] > m:
                m = smoothed_row[v_i]
        if m == -np.inf:
            row_total = -np.inf
        else:
            s = 0.0
            for v_i in range(n_n):
                s += np.exp(smoothed_row[v_i] - m)
            row_total = m + np.log(s)
        # The floor keeps every p > 0, so the renormalising sum is positive.
        renorm_sum = 0.0
        temp_p = np.empty(n_n, dtype=np.float64)
        for v_i in range(n_n):
            if row_total == -np.inf:
                log_p = -np.inf
            else:
                log_p = smoothed_row[v_i] - row_total
            if log_p < min_log_probability:
                log_p = min_log_probability
            p = np.exp(log_p)
            temp_p[v_i] = p
            renorm_sum += p
        for v_i in range(n_n):
            norm_p = temp_p[v_i] / renorm_sum
            final_p[u_i, v_i] = norm_p * (1.0 - uniform_mix) + uniform_val * uniform_mix
    return final_p
