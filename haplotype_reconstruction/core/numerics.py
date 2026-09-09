"""core / numerics for the canonical reconstruction pipeline."""
from __future__ import annotations


import numpy as np
import math
from scipy.special import softmax, gammaln, logsumexp
from numba import njit, prange

import haplotype_reconstruction.core.config as core_config


np.seterr(divide='ignore', invalid="ignore")


def log_fac(x):
    """
    Returns log(x!) using the gamma function.
    Works for scalars and numpy arrays.
    """
    return gammaln(np.asarray(x) + 1)


def log_binomial(n, k):
    """
    Returns log(nCk).
    Works for scalars and numpy arrays.
    """
    return log_fac(n) - log_fac(k) - log_fac(n - k)


@njit(cache=True)
def _logsumexp_scalar_kernel(x):
    """1D input -> scalar.  Numerically stable log-sum-exp.

    Matches scipy.special.logsumexp(x) on a 1D array:
      m = max(x)
      if not isfinite(m): m_safe = 0.0
      else:                m_safe = m
      return m_safe + log(sum(exp(x - m_safe)))

    All-(-inf) input returns -inf, matching scipy.
    """
    K = x.shape[0]
    if K == 0:
        return -np.inf
    m = x[0]
    for k in range(1, K):
        if x[k] > m:
            m = x[k]
    if not np.isfinite(m):
        m_safe = 0.0
    else:
        m_safe = m
    s = 0.0
    for k in range(K):
        s += np.exp(x[k] - m_safe)
    return m_safe + np.log(s)


@njit(cache=True)
def _logsumexp_axis_last_2d_kernel(x):
    """2D input -> 1D, reducing the last axis.

    Matches scipy.special.logsumexp(x, axis=-1) or axis=1 on a 2D array.
    Output shape (N,) for input shape (N, K).
    """
    N, K = x.shape
    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        if K == 0:
            out[i] = -np.inf
            continue
        m = x[i, 0]
        for k in range(1, K):
            if x[i, k] > m:
                m = x[i, k]
        if not np.isfinite(m):
            m_safe = 0.0
        else:
            m_safe = m
        s = 0.0
        for k in range(K):
            s += np.exp(x[i, k] - m_safe)
        out[i] = m_safe + np.log(s)
    return out


@njit(cache=True)
def _logsumexp_axis0_2d_kernel(x):
    """2D input -> 1D, reducing axis=0.

    Matches scipy.special.logsumexp(x, axis=0) on a 2D array.
    Output shape (M,) for input shape (N, M).
    """
    N, M = x.shape
    out = np.empty(M, dtype=np.float64)
    for j in range(M):
        if N == 0:
            out[j] = -np.inf
            continue
        m = x[0, j]
        for i in range(1, N):
            if x[i, j] > m:
                m = x[i, j]
        if not np.isfinite(m):
            m_safe = 0.0
        else:
            m_safe = m
        s = 0.0
        for i in range(N):
            s += np.exp(x[i, j] - m_safe)
        out[j] = m_safe + np.log(s)
    return out


@njit(cache=True)
def _log_matmul_2d_kernel(A, B):
    """Log-space matrix multiplication: C[i,j] = logsumexp_k(A[i,k] + B[k,j]).

    For 2D inputs A=(M, K) and B=(K, N), returns (M, N).
    Eliminates the (M, K, N) broadcast intermediate that the scipy-based
    implementation allocates.
    """
    M, K = A.shape
    K2, N = B.shape
    # Numba doesn't allow `assert K == K2` to raise informatively in nopython
    # mode without object-mode fallback; the caller (log_matmul wrapper)
    # validates shape before invoking this kernel.  We keep the assertion-
    # like check via array-bounds anyway (out of bounds would surface as a
    # numba error).
    C = np.empty((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            # Find max of A[i, k] + B[k, j] over k for numerical stability
            m = A[i, 0] + B[0, j]
            for k in range(1, K):
                v = A[i, k] + B[k, j]
                if v > m:
                    m = v
            if not np.isfinite(m):
                m_safe = 0.0
            else:
                m_safe = m
            s = 0.0
            for k in range(K):
                s += np.exp(A[i, k] + B[k, j] - m_safe)
            C[i, j] = m_safe + np.log(s)
    return C


@njit(cache=True, parallel=True)
def _log_matmul_3d_2d_kernel(A_batch, B):
    """Batched log-space matrix multiplication with shared B.

    A_batch: (S, M, K) float64.
    B:       (K, N)    float64 — shared across all S slices.
    Returns: (S, M, N) float64, where
        C[s, i, j] = logsumexp_k(A_batch[s, i, k] + B[k, j])

    Per-slice (any fixed s) the scalar math and iteration order are
    IDENTICAL to _log_matmul_2d_kernel(A_batch[s], B), so the output is
    bit-equivalent to looping
        for s in range(S): C[s] = _log_matmul_2d_kernel(A_batch[s], B)
    The win is amortising the numba dispatch / array-prep overhead across
    S calls (typically S=320 in block_linking's F/B passes, where the
    inner matmul shape is tiny — 6×6 × 6×6).  prange over s lets numba
    parallelise across samples when the worker has more than one thread.
    """
    S, M, K = A_batch.shape
    K2, N = B.shape
    C = np.empty((S, M, N), dtype=np.float64)
    for s in prange(S):
        for i in range(M):
            for j in range(N):
                # Find max of A_batch[s, i, k] + B[k, j] over k for stability
                m = A_batch[s, i, 0] + B[0, j]
                for k in range(1, K):
                    v = A_batch[s, i, k] + B[k, j]
                    if v > m:
                        m = v
                if not np.isfinite(m):
                    m_safe = 0.0
                else:
                    m_safe = m
                ssum = 0.0
                for k in range(K):
                    ssum += np.exp(A_batch[s, i, k] + B[k, j] - m_safe)
                C[s, i, j] = m_safe + np.log(ssum)
    return C


@njit(cache=True, parallel=True)
def _log_matmul_2d_3d_kernel(A, B_batch):
    """Batched log-space matrix multiplication with shared A.

    A:       (M, K)    float64 — shared across all S slices.
    B_batch: (S, K, N) float64.
    Returns: (S, M, N) float64, where
        C[s, i, j] = logsumexp_k(A[i, k] + B_batch[s, k, j])

    Per-slice (any fixed s) the scalar math and iteration order are
    IDENTICAL to _log_matmul_2d_kernel(A, B_batch[s]) so the output is
    bit-equivalent to looping
        for s in range(S): C[s] = _log_matmul_2d_kernel(A, B_batch[s])
    Used for the T.T @ Z step of block_linking's F/B recurrence, where
    Z is the per-sample batched (S, K_prev, K_curr) tensor.
    """
    M, K = A.shape
    S, K2, N = B_batch.shape
    C = np.empty((S, M, N), dtype=np.float64)
    for s in prange(S):
        for i in range(M):
            for j in range(N):
                # Find max of A[i, k] + B_batch[s, k, j] over k for stability
                m = A[i, 0] + B_batch[s, 0, j]
                for k in range(1, K):
                    v = A[i, k] + B_batch[s, k, j]
                    if v > m:
                        m = v
                if not np.isfinite(m):
                    m_safe = 0.0
                else:
                    m_safe = m
                ssum = 0.0
                for k in range(K):
                    ssum += np.exp(A[i, k] + B_batch[s, k, j] - m_safe)
                C[s, i, j] = m_safe + np.log(ssum)
    return C


def lse_scalar(x):
    """Numba-accelerated logsumexp for 1D inputs returning a scalar.

    Drop-in faster replacement for scipy.special.logsumexp(x) on 1D
    arrays or Python lists.  At small input sizes (typical 4-50 in the
    project's HMM code) this is 10-200x faster than scipy due to
    eliminating scipy's per-call Python dispatch overhead.

    Args:
        x: 1D array-like (np.ndarray or Python list) of float64-castable
           values.

    Returns:
        float — log(sum(exp(x))) computed numerically stably.

    Edge cases:
        - Empty input returns -inf.
        - All-(-inf) input returns -inf (no NaN).
    """
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    return _logsumexp_scalar_kernel(x_arr)


def lse_axis_last(x, keepdims=False):
    """Numba-accelerated logsumexp along the last axis of a 2D array.

    Drop-in faster replacement for scipy.special.logsumexp(x, axis=-1)
    or scipy.special.logsumexp(x, axis=1) on 2D inputs.

    Args:
        x: 2D array (N, K).
        keepdims: if True, returned shape is (N, 1) instead of (N,);
            matches scipy's keepdims semantic.

    Returns:
        np.ndarray — (N,) or (N, 1) of logsumexp values along axis=1.
    """
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    if x_arr.ndim != 2:
        # Defensive: fall back to scipy for non-2D inputs to preserve
        # the public API semantics.  Should not be hit by current
        # callers — they all use 2D.
        return logsumexp(x_arr, axis=-1, keepdims=keepdims)
    out = _logsumexp_axis_last_2d_kernel(x_arr)
    if keepdims:
        return out.reshape(-1, 1)
    return out


def lse_axis0(x):
    """Numba-accelerated logsumexp along axis=0.

    Drop-in faster replacement for scipy.special.logsumexp(x, axis=0).
    Supports any ndim >= 2 by flattening the trailing dims, calling the
    2D axis=0 kernel, and reshaping back.  For 1D input falls back to
    lse_scalar (which is what scipy does — axis=0 on 1D returns scalar).

    Args:
        x: array of ndim >= 1, or a Python list of equal-shape arrays
           which gets stacked along a new leading axis (matching scipy's
           behavior for list input).

    Returns:
        np.ndarray (or scalar for 1D input) — logsumexp reduction along
        axis=0; output shape is x.shape[1:].
    """
    # Handle Python list input (e.g. logsumexp(batch_results, axis=0)
    # where batch_results is a list of equal-shape arrays).  scipy's
    # logsumexp implicitly converts via np.asarray which stacks along
    # the first new axis — same as np.stack(..., axis=0).
    if isinstance(x, list):
        if len(x) == 0:
            return logsumexp(np.asarray(x, dtype=np.float64), axis=0)
        x = np.stack(x, axis=0)
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        # axis=0 on 1D is full reduction to scalar — matches scipy
        return _logsumexp_scalar_kernel(x_arr)
    # General ndim >= 2 path: flatten the trailing dims to (B, M), call
    # the 2D kernel, reshape result to x.shape[1:].
    leading = x_arr.shape[0]
    trailing_shape = x_arr.shape[1:]
    M = 1
    for d in trailing_shape:
        M *= d
    # Reshape preserves underlying data layout for C-contiguous input.
    x_flat = x_arr.reshape(leading, M)
    out_flat = _logsumexp_axis0_2d_kernel(x_flat)
    return out_flat.reshape(trailing_shape)


def _build_haploid_log_T_from_dict(trans_dict, prev_keys, curr_keys, prev_idx, curr_idx,
                                     missing_default=None):
    """Helper: build a dense (n_prev, n_curr) log-transition matrix from
    a sparse Python dict.

    The dict is expected to be keyed by ((prev_idx, prev_hap_key),
    (curr_idx, curr_hap_key)) tuples — the project's standard
    transition-probability dict format.  Missing entries yield -inf by
    default, or a caller-supplied `missing_default` value.

    Cannot be numba-accelerated (numba doesn't support arbitrary
    Python dict key types).  Extracted to a single location to ensure
    block_linking.get_full_probs_forward/backward,
    hmm_matching.build_dense_transition_matrix, and the hap_log_prior
    construction in hmm_matching.update_transitions_layered_hmm all
    use the SAME construction logic.

    Args:
        trans_dict: dict {((prev_idx, prev_hap), (curr_idx, curr_hap)): prob}
        prev_keys: list of hap IDs at prev_idx
        curr_keys: list of hap IDs at curr_idx
        prev_idx, curr_idx: int block indices used to build the
            two-level tuple key
        missing_default: optional float for missing entries.  Default
            None means use -inf (the original behavior, matching
            block_linking.get_full_probs_forward's
            `T = np.full(..., -np.inf)`).  Set to math.log(1e-9) to
            match hmm_matching.update_transitions_layered_hmm's
            original `sparse_trans.get(..., 1e-9)` then `math.log()`
            pattern, where missing edges were treated as having a
            very small (but non-zero) probability rather than being
            forbidden.

    Returns:
        np.ndarray (n_prev, n_curr) float64 — log of probability where
            the key exists, missing_default (or -inf if None) elsewhere.
    """
    n_prev = len(prev_keys)
    n_curr = len(curr_keys)
    if missing_default is None:
        # Original behavior — preserved exactly for existing callers.
        hap_log_T = np.full((n_prev, n_curr), -np.inf)
    else:
        hap_log_T = np.full((n_prev, n_curr), float(missing_default))
    for u_i, u_key in enumerate(prev_keys):
        for x_i, x_key in enumerate(curr_keys):
            key = ((prev_idx, u_key), (curr_idx, x_key))
            if key in trans_dict:
                hap_log_T[u_i, x_i] = math.log(trans_dict[key])
    return hap_log_T


def log_matmul(A, B):
    """
    Performs Matrix Multiplication in Log-Space.
    Mathematically equivalent to C = A @ B but entirely in the log domain.

    Formula: C_ij = logsumexp_k(A_ik + B_kj)

    Args:
        A: Tensor of shape (..., M, K) or (M, K)
        B: Tensor of shape (..., K, N) or (K, N)

    Returns:
        Tensor of shape (..., M, N) resulting from log-space multiplication.
        Supports broadcasting for batches.

    Implementation: dispatches to a numba kernel for the three common
    shape combinations used by the project:
      - 2D × 2D     -> _log_matmul_2d_kernel    (eliminates scipy's
                                                  (M, K, N) intermediate).
      - 3D × 2D     -> _log_matmul_3d_2d_kernel (batched-A, shared-B;
                                                  bit-equivalent to looping
                                                  the 2D kernel per slice).
      - 2D × 3D     -> _log_matmul_2d_3d_kernel (shared-A, batched-B;
                                                  same equivalence).
    For higher-dim broadcast inputs falls back to the scipy implementation
    (no current project caller passes higher-dim inputs, but the public
    API supports it).  The numba paths are bit-equivalent to within ~2e-15
    of the scipy path on typical inputs.
    """
    A_arr = np.asarray(A)
    B_arr = np.asarray(B)

    # 2D × 2D fast path — what every original caller uses.
    if A_arr.ndim == 2 and B_arr.ndim == 2 and A_arr.shape[1] == B_arr.shape[0]:
        A_c = np.ascontiguousarray(A_arr, dtype=np.float64)
        B_c = np.ascontiguousarray(B_arr, dtype=np.float64)
        return _log_matmul_2d_kernel(A_c, B_c)

    # 3D × 2D: batched A with shared B — block_linking's per-sample
    # forward/backward step `log_matmul(prev_matrix_batched, T)`.
    if A_arr.ndim == 3 and B_arr.ndim == 2 and A_arr.shape[2] == B_arr.shape[0]:
        A_c = np.ascontiguousarray(A_arr, dtype=np.float64)
        B_c = np.ascontiguousarray(B_arr, dtype=np.float64)
        return _log_matmul_3d_2d_kernel(A_c, B_c)

    # 2D × 3D: shared A with batched B — block_linking's
    # `log_matmul(T.T, Z_batched)` step.
    if A_arr.ndim == 2 and B_arr.ndim == 3 and A_arr.shape[1] == B_arr.shape[1]:
        A_c = np.ascontiguousarray(A_arr, dtype=np.float64)
        B_c = np.ascontiguousarray(B_arr, dtype=np.float64)
        return _log_matmul_2d_3d_kernel(A_c, B_c)

    # Fallback for broadcasting / non-2D-or-3D cases: original scipy-based
    # implementation, preserved for API compatibility.
    return logsumexp(A[..., np.newaxis] + B[..., np.newaxis, :, :], axis=-2)


def reads_to_probabilities(reads_array,
                           read_error_prob=core_config.DEFAULT_READ_ERROR_PROBABILITY,
                           min_total_reads=5, use_hwe_prior=True):
    """
    Convert a reads array to a probability of the underlying
    genotype being 0, 1 or 2.

    ``use_hwe_prior=True`` preserves the historical empirical-Bayes
    posterior. ``False`` returns normalized raw genotype likelihoods while
    still returning the estimated site priors for diagnostics. The latter is
    useful when population structure is modelled separately.

    Returns:
        (site_priors, genotype_probs)
    """
    num_samples, num_sites, _ = reads_array.shape

    # 0. Handle empty case
    if num_sites == 0:
        return (np.empty((num_sites, 3)), np.empty((num_samples, num_sites, 3)))

    # --- PART 1: Calculate Site Priors ---

    # Sum reads across samples
    reads_sum = np.sum(reads_array, axis=0)
    total_reads_per_site = np.sum(reads_sum, axis=1)

    # Create mask for valid sites
    threshold = max(min_total_reads, read_error_prob * num_samples)
    valid_mask = total_reads_per_site >= threshold

    # Calculate ratios
    numerator = 1 + reads_sum[:, 1]
    denominator = 2 + total_reads_per_site
    calculated_ratios = numerator / denominator

    # Apply condition: if invalid, use error prob
    singleton = np.where(valid_mask, calculated_ratios, read_error_prob)

    # Calculate priors (00, 01, 11) assuming HWE approximation
    priors_00 = (1 - singleton) ** 2
    priors_01 = 2 * singleton * (1 - singleton)
    priors_11 = singleton ** 2

    site_priors = np.stack([priors_00, priors_01, priors_11], axis=1)

    if not use_hwe_prior:
        return (
            site_priors,
            core_genotypes.allele_depths_to_raw_genotype_likelihoods(
                reads_array, read_error_probability=read_error_prob
            ),
        )

    # --- PART 2: Calculate Likelihoods ---

    log_half = math.log(0.5)
    log_read_error = math.log(read_error_prob)
    log_read_nonerror = math.log(1 - read_error_prob)

    zeros = reads_array[..., 0]
    ones = reads_array[..., 1]
    total = zeros + ones

    # Compute log binomials
    lb_total_ones = log_binomial(total, ones)
    lb_total_zeros = log_binomial(total, zeros)

    # Calculate Genotype Likelihoods
    # 00: Homozygous Ref
    ll_00 = lb_total_ones + zeros * log_read_nonerror + ones * log_read_error
    # 11: Homozygous Alt
    ll_11 = lb_total_zeros + zeros * log_read_error + ones * log_read_nonerror
    # 01: Heterozygous
    ll_01 = lb_total_ones + total * log_half

    log_likli_matrix = np.stack([ll_00, ll_01, ll_11], axis=-1)

    # --- PART 3: Combine and Normalize ---

    # Broadcast priors (1, Sites, 3) against likelihoods (Samples, Sites, 3).
    log_site_priors = np.log(site_priors)
    log_evidence = log_likli_matrix + log_site_priors[np.newaxis, :, :]

    # Softmax normalizes in log space
    genotype_probs = softmax(log_evidence, axis=-1)

    return (site_priors, genotype_probs)


def calc_distance(first_row, second_row, calc_type="diploid"):
    """
    Calculate the probabilistic distance between two rows (einsum based).
    Used for single comparisons.
    """
    if calc_type == "diploid":
        distances = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype=float)
    else:
        distances = np.array([[0,1],[1,0]], dtype=float)

    ens = np.einsum("ij,ik->ijk", first_row, second_row)
    ensd = ens * distances

    return np.sum(ensd, axis=None)


@njit(fastmath=True)
def probability_to_information(probs_list):
    """
    Signed information metric for a [1-p, p] pair.
    """
    p = probs_list[1]

    if p >= 1.0: return 1.0
    if p <= 0.0: return -1.0

    sgn = -1.0 if p < 0.5 else 1.0
    entropy = -(1-p)*math.log2(1-p) - p*math.log2(p)

    return sgn * (1.0 - entropy)


@njit(fastmath=True)
def add_informations(first_information, second_information):
    """
    Relativistic addition of information.
    """
    if abs(first_information * second_information + 1.0) < 1e-9:
        return 0.0

    return (first_information + second_information) / (1.0 + first_information * second_information)


@njit(fastmath=True)
def combine_probabilities(first_prob, second_prob, prior_prob, required_accuracy=1e-13):
    """
    Combine observed probabilities with a prior using binary search inversion.
    JIT-compiled for speed.
    """
    first_information = probability_to_information(first_prob)
    second_information = probability_to_information(second_prob)
    prior_information = probability_to_information(prior_prob)

    first_relative = add_informations(first_information, -prior_information)
    second_relative = add_informations(second_information, -prior_information)

    combined_relative = add_informations(first_relative, second_relative)
    full_combined = add_informations(combined_relative, prior_information)

    # Binary search inversion
    low = 0.0
    high = 1.0

    for _ in range(60):
        midpoint = (low + high) * 0.5
        test_prob = np.array([1.0 - midpoint, midpoint])
        test_val = probability_to_information(test_prob)

        if test_val < full_combined:
            low = midpoint
        else:
            high = midpoint

    final_prob = (low + high) * 0.5
    return np.array([1.0 - final_prob, final_prob])


import haplotype_reconstruction.core.genotypes as core_genotypes
