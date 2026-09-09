"""discovery / assignments for the canonical reconstruction pipeline."""
from __future__ import annotations


import numpy as np
import math
import numba

from functools import lru_cache

from numba import njit, prange


@lru_cache(maxsize=32)
def _pair_indices_for_K(K):
    """Return immutable row-major ``i <= j`` pair indices for one K.

    These arrays depend only on K but are consumed by every assignment update
    in every fixed-K fit.  Keeping one process-local copy avoids rebuilding
    the same triangular geometry thousands of times per block.  The cached
    arrays have exactly the dtype, values, and order returned by
    ``np.triu_indices``; making them read-only prevents accidental mutation.
    """
    rr_i, rr_j = np.triu_indices(int(K))
    rr_i = np.ascontiguousarray(rr_i, dtype=np.int64)
    rr_j = np.ascontiguousarray(rr_j, dtype=np.int64)
    rr_i.setflags(write=False)
    rr_j.setflags(write=False)
    return rr_i, rr_j


def _update_A(probs_k, H_k, lam, cost_WW=None, WW_bin_emis=None, log_probs=None,
              blas_lp_cache=None, binary_pattern_cache=None,
              WW_total_cost=None):
    """For each sample, pick the pair assignment that minimises its
    capped cost.

    Arguments:
        probs_k: (N, L_kept, 3)
        H_k:     (K, L_kept) discrete in {0, 1}
        lam:     wildcard penalty
        cost_WW: (N, L_kept) optional — precomputed per-(sample, site)
                  WW cost from `_per_site_cost_W_W(probs_k, lam)`.
                  When None, computed internally on every call.  When
                  provided, the WW state of the fused kernel skips its
                  inline pmax/log work; this is what `_fit_at_fixed_K`
                  uses to amortise WW computation across CD iterations.
        WW_bin_emis: (N, n_bins) optional — precomputed Viterbi binned
                  emissions for the WW state.  Derived from cost_WW via
                  `_ww_bin_emis_from_cost_ww`; if cost_WW is supplied
                  but WW_bin_emis is None, the latter is derived here.
        WW_total_cost: (N,) optional — exact left-to-right per-sample sum of
                  ``cost_WW``.  The table-fed kernel reuses this evidence-only
                  value rather than resumming all sites during every
                  coordinate-descent assignment update.
        log_probs: (N, L_kept, 3) optional — precomputed
                  log(max(probs_k[s, l, g], LOG_EPS_LOCAL)).  When
                  provided, the fused kernel reads it instead of
                  computing log(probs[s, l, d]) inline at every
                  (s, state, l) visit.  Stable across CD iterations
                  (probs_k doesn't change), so `_fit_at_fixed_K`
                  precomputes once and threads through.  Tier 0 of
                  the optimisation programme; expected ~25% CPU
                  reduction at K=6 N=320 L=200 because the kernel
                  was log()-bound.
        blas_lp_cache: optional 5-tuple from
                  `_update_A_blas_lp_precompute(log_probs, lam,
                  snps_per_bin, n_bins)` — the log_probs-derived inputs
                  (C0b, diff1_bt, w_bt, kW_Cb, kWdiff_bt) for the
                  BLAS-hybrid kernel.  Like cost_WW / log_probs these
                  are invariant across CD iterations, so
                  `_fit_at_fixed_K` precomputes them once and threads
                  them through; standalone callers leave this None and
                  it is computed internally.  Consulted only on the
                  K > 0 production path; ignored at K = 0.
        binary_pattern_cache: optional immutable evidence-local contraction
                  tables prepared by `_prepare_fixed_k_fit_workspace`.  For
                  exactly binary H with K >= 1, all possible within-bin
                  binary patterns were contracted by the same BLAS operation
                  once, so per-fit H-dependent GEMMs become exact table
                  lookups.  K=0, non-integer/nonbinary H, incompatible
                  tables, and standalone calls fall back to the existing
                  GEMM path.

    Returns:
        A: (N, 2) int array — A[s, *] in {0..K-1, K} where K = wildcard
            sentinel (one past the last real founder index).  Entries are
            sorted ascending so each unordered pair has a canonical
            representation; W is always placed last.
        per_sample_cost: (N,) — total CAPPED cost under chosen pair (used
            internally as the M-step's view of per-sample fit; bounded
            above by N_kept_sites × cost_WW_per_site)
        per_sample_cost_unc: (N,) — total UNCAPPED cost under the same
            assignment (used as the K-growth NLL improvement signal,
            since capped NLL plateaus when adding founders only converts
            samples from "way over cost_WW" to "still over cost_WW")
        wildcard_slots: (N,) int — number of wildcard strands used by sample
            from the pair assignment alone (0, 1, or 2).  Note: with the
            cap, even (real, real)-assigned samples may effectively use
            wildcards at some sites; this slot count reflects only the
            global pair structure.
    """
    K, L = H_k.shape
    N = probs_k.shape[0]

    probs_c = discovery_objectives._maybe_c_contig(probs_k, np.float64)
    H_c = discovery_objectives._maybe_c_contig(H_k, np.int64)

    # === FUSED BLAS PATH (production default) ===
    # When _VITERBI_BIC_ENABLED is True (the production default) AND
    # K > 0, dispatch to _update_A_fused_blas_kernel, which produces the
    # baseline outputs (A, wildcard_slots) AND the Viterbi-BIC signal
    # (per_sample_cost = -viterbi_ll) from precomputed GEMM contractions
    # of the rr/kW state emissions.  See that kernel and
    # _update_A_blas_lp_precompute for the factorisation and its
    # equivalence to a direct scalar per-(state, site) summation.
    #
    # K = 0 falls through to the baseline-only path below for bit-
    # identity with the legacy K=0 behaviour, which skipped the Viterbi
    # override (the "if _VITERBI_BIC_ENABLED and K > 0" guard).  At K = 0
    # the only state is (W, W), there are no transitions, and
    # per_sample_cost equals the baseline WW cost directly via
    # _per_site_cost_W_W's left-to-right sum.
    if core_config._VITERBI_BIC_ENABLED and K > 0:
        # Bin sizing — same logic as _viterbi_ll_per_sample so the
        # fused kernel's bin_emis matches _viterbi_binned_emissions_-
        # kernel's output bit-for-bit.
        if core_config.VITERBI_SNPS_PER_BIN > 1 and core_config.VITERBI_SNPS_PER_BIN < L:
            n_bins = int(math.ceil(L / core_config.VITERBI_SNPS_PER_BIN))
            snps_per_bin = core_config.VITERBI_SNPS_PER_BIN
        else:
            n_bins = L
            snps_per_bin = 1

        # Precompute the WW state's arrays if not provided by the caller.
        # cost_WW depends only on (probs_k, lam) and so doesn't change
        # across CD iterations; `_fit_at_fixed_K` passes a once-per-fit
        # cached version through every call.  WW_bin_emis is derived
        # from cost_WW + (snps_per_bin, n_bins) and is also stable
        # across CD iterations once cached.
        if cost_WW is None:
            cost_WW = discovery_objectives._per_site_cost_W_W(probs_c, float(lam))
        cost_WW_c = discovery_objectives._maybe_c_contig(cost_WW, np.float64)
        if WW_bin_emis is None:
            WW_bin_emis = discovery_objectives._ww_bin_emis_from_cost_ww(
                cost_WW_c, int(snps_per_bin), int(n_bins))
        WW_bin_emis_c = discovery_objectives._maybe_c_contig(WW_bin_emis, np.float64)
        if WW_total_cost is None:
            WW_total_cost = discovery_fitting._ww_total_cost_kernel(cost_WW_c)
        WW_total_cost_c = discovery_objectives._maybe_c_contig(WW_total_cost, np.float64)

        # Precompute log_probs once if not provided (Tier 0).  Like
        # cost_WW, log_probs is invariant across CD iterations because
        # probs_k doesn't change, so `_fit_at_fixed_K` caches it once
        # per invocation.  Standalone callers (e.g. unit tests, the
        # recovery loop's _update_A calls outside the CD loop) get
        # internal computation here.
        if log_probs is None:
            log_probs = discovery_objectives._log_probs_kernel(probs_c)
        log_probs_c = discovery_objectives._maybe_c_contig(log_probs, np.float64)

        # Factor the rr and kW state emissions into GEMM contractions
        # (see _update_A_fused_blas_kernel and _update_A_blas_lp_-
        # precompute).  The log_probs-derived inputs are invariant across
        # CD iterations, so _fit_at_fixed_K threads a once-per-fit
        # blas_lp_cache; standalone callers compute it here.
        if blas_lp_cache is None:
            blas_lp_cache = _update_A_blas_lp_precompute(
                log_probs_c, float(lam), int(snps_per_bin), int(n_bins))
        C0b, diff1_bt, w_bt, kW_Cb, kWdiff_bt = blas_lp_cache

        # rr pair indices (i, j) with i <= j in row-major order — matches
        # the rr state order of _update_A_fused_blas_kernel.
        n_rr = K * (K + 1) // 2
        rr_i, rr_j = _pair_indices_for_K(K)

        # Exact binary-pattern path.  The table workspace is tied by object
        # identity to this evidence's BLAS precompute and is never mutated.
        # Restrict the shortcut to integer/bool binary H with K>=1. The
        # canonical recurrence is deliberately independent of BLAS row shape;
        # general/nonbinary callers retain the established GEMM behaviour.
        H_input = np.asarray(H_k)
        use_binary_patterns = (
            K >= 1
            and isinstance(
                binary_pattern_cache,
                _BinaryPatternContractionWorkspace,
            )
            and binary_pattern_cache.blas_lp_reference is blas_lp_cache
            and binary_pattern_cache.n_bins == int(n_bins)
            and binary_pattern_cache.snps_per_bin == int(snps_per_bin)
            and binary_pattern_cache.n_samples == int(N)
            and binary_pattern_cache.supports_k(K)
            and H_input.dtype.kind in "biu"
        )
        if use_binary_patterns:
            is_binary, h_patterns, pair_patterns = (
                _binary_haplotype_patterns(
                    H_c, rr_i, rr_j,
                    int(snps_per_bin), int(n_bins), int(L),
                )
            )
            if is_binary:
                pattern_kernel = _update_A_fused_pattern_kernel
                if numba.get_num_threads() == 1:
                    pattern_kernel = _update_A_fused_pattern_kernel_serial
                A, per_sample_cost, wildcard_slots = (
                    pattern_kernel(
                        C0b,
                        binary_pattern_cache.diff1_table,
                        binary_pattern_cache.w_table,
                        kW_Cb,
                        binary_pattern_cache.kWdiff_table,
                        h_patterns,
                        pair_patterns,
                        WW_bin_emis_c,
                        WW_total_cost_c,
                        rr_i,
                        rr_j,
                        float(core_config.VITERBI_SWITCH_PENALTY),
                        K,
                        int(n_bins),
                    )
                )
                per_sample_cost_unc = per_sample_cost
                return (
                    A,
                    per_sample_cost,
                    per_sample_cost_unc,
                    wildcard_slots,
                )

        # H-dependent GEMM inputs.  Pad H to n_bins*snps_per_bin with
        # zeros when L is not a multiple of snps_per_bin; padded sites
        # contribute 0 to every binned contraction (see precompute).
        Lpad = int(n_bins) * int(snps_per_bin)
        if Lpad == L:
            Hf = H_c.astype(np.float64)
        else:
            Hf = np.zeros((K, Lpad), dtype=np.float64)
            Hf[:, :L] = H_c
        Hrb = np.ascontiguousarray(
            Hf.reshape(K, int(n_bins), int(snps_per_bin)).transpose(1, 0, 2))
        # Ub[k,s,b]   = sum_{l in bin b} (a1-a0)[s,l] * H[k,l]
        Ub = np.ascontiguousarray(
            np.matmul(Hrb, diff1_bt).transpose(1, 2, 0))
        # kW_Ub[k,s,b] = sum_{l in bin b} (m12-m01)[s,l] * H[k,l]
        kW_Ub = np.ascontiguousarray(
            np.matmul(Hrb, kWdiff_bt).transpose(1, 2, 0))
        # BB[p,l] = H[i,l]*H[j,l] for rr pair p=(i,j); then
        # Mb[p,s,b] = sum_{l in bin b} (a0-2a1+a2)[s,l] * BB[p,l]
        BB = Hf[rr_i] * Hf[rr_j]
        BBrb = np.ascontiguousarray(
            BB.reshape(n_rr, int(n_bins), int(snps_per_bin)).transpose(1, 0, 2))
        Mb = np.ascontiguousarray(
            np.matmul(BBrb, w_bt).transpose(1, 2, 0))

        A, baseline_cost, wildcard_slots, viterbi_ll = (
            _update_A_fused_blas_kernel(
                C0b, Ub, Mb, kW_Cb, kW_Ub,
                cost_WW_c, WW_bin_emis_c,
                rr_i, rr_j, float(core_config.VITERBI_SWITCH_PENALTY),
                K, int(n_bins), int(L)))

        # Per-sample cost = -log-likelihood (NLL convention used elsewhere
        # in this module).  Alias _unc to match the previous code's
        # invariant (downstream callers treat per_sample_cost ==
        # per_sample_cost_unc when Viterbi BIC is active).
        per_sample_cost = (-viterbi_ll).astype(np.float64)
        per_sample_cost_unc = per_sample_cost
        return A, per_sample_cost, per_sample_cost_unc, wildcard_slots

    # === BASELINE-ONLY PATH (K = 0 or Viterbi BIC disabled) ===
    # Fused baseline pass: build cost-per-candidate-pair AND track
    # argmin in-flight, in a single njit kernel that never materialises
    # the (N, n_pairs_rr, L), (N, K, L), or (N, L) per-site cost tensors.
    # At N=320, K=6, L=200 these three tensors total ~14 MB of allocator
    # churn per _update_A call; the fused kernel keeps the running-best
    # candidate in scalar registers and gains cache locality by walking
    # probs_k[s, l, *] for all candidates of sample s in immediate
    # succession (~28 candidates per sample at K=6).
    #
    # Pair assignment uses UNCAPPED costs.  The strict-diploid constraint
    # says each sample has exactly two strands, and the per-pair cost
    # reflects the true model's prediction error under that pair.  We
    # apply the per-(strand, site) wildcard-escape cap (Fix H) only in
    # the M-step (_update_H), where it prevents non-carrier samples from
    # contaminating the founder's update at incompatible sites.  Using
    # the cap in pair assignment would make non-carriers prefer (real, W)
    # ties with (W, W), routing them away from (W, W) and inflating
    # their effective uncapped NLL — which would break the K-growth
    # improvement signal.
    A, per_sample_cost, wildcard_slots = _update_A_baseline_kernel(
        probs_c, H_c, float(lam))
    # Uncapped is the same as the assignment cost since we used uncapped
    # to assign in the first place.  Returned for API symmetry with the
    # Fix-H-cap-in-pair-assignment design that was rejected; downstream
    # callers can treat per_sample_cost == per_sample_cost_unc.
    per_sample_cost_unc = per_sample_cost
    return A, per_sample_cost, per_sample_cost_unc, wildcard_slots


@njit(cache=True, parallel=True, fastmath=False)
def _update_A_baseline_kernel(probs_k, H_k, lam):
    """Fused baseline cost + argmin kernel for _update_A.

    For each sample s, evaluates every candidate pair (real-real with
    i <= j, real-W in k order, then W-W) in immediate succession,
    tracking the running-best (lowest-cost) candidate via scalar
    registers.  Returns the winning (a, b, per_sample_cost,
    wildcard_slots) tuple per sample, with NO intermediate (N, n_pairs,
    L), (N, K, L), or (N, L) cost tensors allocated.

    Iteration order matches the original's flat all_costs concatenation:
        [real-real pairs (i, j) with i <= j, row-major]
      + [real-W pairs (k, W) in k order]
      + [(W, W)]
    so the running-best with STRICT-< update produces the same first-
    occurrence-tiebreak result as np.argmin on the concatenated array.

    Per-site summation order also matches the numpy version: for each
    candidate, the L-sites are accumulated left-to-right (l = 0, 1,
    ..., L-1).  Combined with float64 arithmetic this gives bit-
    identical results to the original three-tensor implementation
    (verified at machine precision).

    Inputs:
        probs_k: (N, L, 3) float64, C-contig
        H_k:     (K, L)    int64,   C-contig
        lam:     wildcard penalty (per strand-site)

    Returns:
        A:               (N, 2)   int64, canonical (real-first, W-second)
        per_sample_cost: (N,)     float64, BASELINE best-pair NLL
        wildcard_slots:  (N,)     int64,  count of W strands in A[s]
                                          (0 for real-real, 1 for kW, 2 for WW)

    Floors -log(p) at LOG_EPS_LOCAL = 1e-12 to match _safe_neg_log.
    LOG_EPS_LOCAL is inlined as a literal because module-level
    constants are not importable inside @njit functions; if you change
    LOG_EPS in the module body, change LOG_EPS_LOCAL in every kernel.
    """
    LOG_EPS_LOCAL = 1e-12

    N = probs_k.shape[0]
    L = probs_k.shape[1]
    K = H_k.shape[0]
    W = K   # wildcard sentinel = one past last real founder index

    A = np.empty((N, 2), dtype=np.int64)
    per_sample_cost = np.empty(N, dtype=np.float64)
    wildcard_slots = np.empty(N, dtype=np.int64)

    # prange over samples — each sample's argmin is independent.  Inner
    # candidate loops are sequential per sample, with the L-site loop
    # innermost for cache locality on the C-contig probs_k.
    for s in prange(N):
        # Sentinels: best_cost = +inf forces the first candidate (the
        # (0, 0) real-real pair if K >= 1, or the (W, W) state if K == 0)
        # to set the initial best.  Strict-< on subsequent updates
        # preserves first-occurrence tiebreak semantics matching the
        # original's np.argmin.
        best_cost = np.inf
        best_a = 0
        best_b = 0
        best_wcs = 0

        # ----- Real-real pairs (i, j) with i <= j, in row-major -----
        # Inner L loop accumulates -log(probs[s, l, H[i,l]+H[j,l]]).
        # When K == 0 this nested loop has zero iterations and falls
        # through to the (W, W) branch below.
        for i in range(K):
            for j in range(i, K):
                cost = 0.0
                for l in range(L):
                    d = H_k[i, l] + H_k[j, l]
                    pv = probs_k[s, l, d]
                    if pv < LOG_EPS_LOCAL:
                        pv = LOG_EPS_LOCAL
                    cost -= math.log(pv)
                if cost < best_cost:
                    best_cost = cost
                    best_a = i
                    best_b = j
                    best_wcs = 0

        # ----- Real-W pairs (k, W) in k order -----
        # Wildcard strand picks its allele w in {0, 1} to maximise
        # probs[s, l, H[k, l] + w] per site; cost = -log of that max
        # plus lam per site.  Summed left-to-right over L.
        for k in range(K):
            cost = 0.0
            for l in range(L):
                d0 = H_k[k, l]
                d1 = d0 + 1
                p0 = probs_k[s, l, d0]
                p1 = probs_k[s, l, d1]
                pmax = p0 if p0 > p1 else p1
                if pmax < LOG_EPS_LOCAL:
                    pmax = LOG_EPS_LOCAL
                cost += -math.log(pmax) + lam
            if cost < best_cost:
                best_cost = cost
                best_a = k
                best_b = W
                best_wcs = 1

        # ----- (W, W) -----
        # Both strands wildcard; each picks its allele optimally,
        # giving max over (p0, p1, p2) per site, plus 2*lam.
        cost = 0.0
        for l in range(L):
            p0 = probs_k[s, l, 0]
            p1 = probs_k[s, l, 1]
            p2 = probs_k[s, l, 2]
            pmax = p0
            if p1 > pmax:
                pmax = p1
            if p2 > pmax:
                pmax = p2
            if pmax < LOG_EPS_LOCAL:
                pmax = LOG_EPS_LOCAL
            cost += -math.log(pmax) + 2.0 * lam
        if cost < best_cost:
            best_cost = cost
            best_a = W
            best_b = W
            best_wcs = 2

        A[s, 0] = best_a
        A[s, 1] = best_b
        per_sample_cost[s] = best_cost
        wildcard_slots[s] = best_wcs

    return A, per_sample_cost, wildcard_slots


@njit(cache=True, parallel=True, fastmath=False)
def _update_A_blas_lp_precompute_kernel(log_probs, lam, snps_per_bin, n_bins, L):
    """Parallel njit core of _update_A_blas_lp_precompute.

    Computes the five H-independent BLAS inputs in one prange over bins.  The
    previous implementation built them with a chain of serial numpy ops
    (reshape + sum + three transpose-and-copy passes over the (N, L) log-prob
    planes), which ran single-threaded; this fills the same arrays directly,
    parallel over the n_bins axis.  Bins are the OUTERMOST axis of the three
    (n_bins, snps_per_bin, N) outputs, so each thread writes its own contiguous
    block of those (bulk) arrays with no false sharing.

    Bit-identical to the numpy version: each output element is the same scalar
    expression; the per-bin sums (C0b, kW_Cb) accumulate the bin's real sites in
    the same site order numpy's sum(axis=2) uses; padded sites (l >= L)
    contribute exact 0.0 (left at the np.zeros init), matching the numpy
    zero-pad path; and the m01/m12 ternaries equal np.maximum on these finite
    log-probabilities.  L is the true (unpadded) site count.
    """
    N = log_probs.shape[0]
    C0b = np.zeros((N, n_bins), dtype=np.float64)
    kW_Cb = np.zeros((N, n_bins), dtype=np.float64)
    diff1_bt = np.zeros((n_bins, snps_per_bin, N), dtype=np.float64)
    w_bt = np.zeros((n_bins, snps_per_bin, N), dtype=np.float64)
    kWdiff_bt = np.zeros((n_bins, snps_per_bin, N), dtype=np.float64)
    for b in prange(n_bins):
        for s in range(N):
            c0 = 0.0
            cm01 = 0.0
            ncnt = 0
            for t in range(snps_per_bin):
                l = b * snps_per_bin + t
                if l < L:
                    a0 = log_probs[s, l, 0]
                    a1 = log_probs[s, l, 1]
                    a2 = log_probs[s, l, 2]
                    m01 = a0 if a0 > a1 else a1
                    m12 = a1 if a1 > a2 else a2
                    diff1_bt[b, t, s] = a1 - a0
                    w_bt[b, t, s] = a0 - 2.0 * a1 + a2
                    kWdiff_bt[b, t, s] = m12 - m01
                    c0 += a0
                    cm01 += m01
                    ncnt += 1
            C0b[s, b] = c0
            kW_Cb[s, b] = cm01 - ncnt * lam
    return C0b, diff1_bt, w_bt, kW_Cb, kWdiff_bt


def _update_A_blas_lp_precompute(log_probs, lam, snps_per_bin, n_bins):
    """Precompute the log_probs-derived BLAS inputs for
    `_update_A_fused_blas_kernel`.

    These depend only on log_probs (hence on probs_k) and lam — NOT on
    H — so they are INVARIANT across a coordinate-descent run and can be
    computed once per `_fit_at_fixed_K` invocation, which threads the
    result through every `_update_A` call (exactly as it already does for
    cost_WW / WW_bin_emis / log_probs).  Standalone `_update_A` callers
    pass blas_lp_cache=None and get this computed internally per call.

    Factorisation (the algebraic identity the kernel relies on).  Write
    a_g = log P(g) for g in {0, 1, 2} (i.e. log_probs[s, l, g], already
    LOG_EPS-clamped).  The rr state (i, j) per-site emission is
    a_{H[i,l] + H[j,l]}, which expands EXACTLY (verified to ~1e-13) as
        a_d = a0
              + (a1 - a0) * (H[i,l] + H[j,l])
              + (a0 - 2 a1 + a2) * H[i,l] * H[j,l]
    (check d = 0, 1, 2 against H[i,l], H[j,l] in {0,1}).  Summing over
    the sites of a bin gives, per (sample, bin),
        C0b + Ub[i] + Ub[j] + Mb[(i,j)]
    with C0b = sum a0, Ub[k] = sum (a1-a0) H[k,l] = (H @ (a1-a0)),
    and Mb[(i,j)] = sum (a0-2a1+a2) H[i,l] H[j,l] = ((H_i*H_j) @ w).
    The kW state (k, W) per-site emission max(a_{H[k,l]}, a_{H[k,l]+1})
    - lam is likewise linear in H[k,l]: with m01 = max(a0, a1), m12 =
    max(a1, a2), it equals m01 + (m12 - m01) H[k,l] - lam, summing to
        kW_Cb + kW_Ub[k]
    with kW_Cb = sum m01 - (real sites in bin) * lam and kW_Ub[k] =
    sum (m12 - m01) H[k,l] = (H @ (m12-m01)).

    Returns the H-INDEPENDENT pieces as a 5-tuple
    (C0b, diff1_bt, w_bt, kW_Cb, kWdiff_bt):
        C0b:       (N, n_bins)                    sum_bin a0
        diff1_bt:  (n_bins, snps_per_bin, N)      (a1 - a0), binned + T
        w_bt:      (n_bins, snps_per_bin, N)      (a0 - 2a1 + a2), binned + T
        kW_Cb:     (N, n_bins)                    sum_bin m01 - n_real*lam
        kWdiff_bt: (n_bins, snps_per_bin, N)      (m12 - m01), binned + T
    The "_bt" layout (n_bins, snps_per_bin, N) lets np.matmul against a
    (n_bins, K, snps_per_bin) H-tensor produce the binned (n_bins, K, N)
    contraction in one batched GEMM.  The H-dependent GEMMs (Ub, Mb,
    kW_Ub) are formed by the caller (`_update_A`).

    Ragged last bin: when L is not a multiple of snps_per_bin the arrays
    are zero-padded to n_bins*snps_per_bin.  A padded site contributes 0
    to every binned sum (0 is the additive identity) and 0 to every
    matmul (its a-values are 0, so a1-a0, a0-2a1+a2, m12-m01 are all 0,
    and the H / H_i*H_j factors are 0 there too), so the result matches
    the scalar kernel's `if end > L: end = L` clamp exactly.  The kW
    per-bin lam offset uses the REAL (unpadded) site count per bin.
    """
    # The five H-independent inputs are now built in one parallel njit kernel
    # (prange over bins) instead of the serial numpy reshape/sum/transpose
    # chain below; output is bit-identical (see the kernel docstring).
    lp_c = discovery_objectives._maybe_c_contig(log_probs, np.float64)
    L = lp_c.shape[1]
    return _update_A_blas_lp_precompute_kernel(
        lp_c, float(lam), int(snps_per_bin), int(n_bins), int(L))


@njit(cache=True, parallel=True, fastmath=False)
def _update_A_blas_lp_precompute_depth_mask_kernel(
        log_probs, observed_mask, lam, snps_per_bin, n_bins, L):
    """Depth-mask counterpart of the established assignment precompute.

    Missing sample/site cells have all three log emissions fixed to zero by
    the caller.  This kernel additionally charges the real-wildcard penalty
    only at observed cells.  Real-real contractions are otherwise identical
    to the established implementation, so the existing BLAS/pattern kernels
    remain valid.
    """
    N = log_probs.shape[0]
    C0b = np.zeros((N, n_bins), dtype=np.float64)
    kW_Cb = np.zeros((N, n_bins), dtype=np.float64)
    diff1_bt = np.zeros((n_bins, snps_per_bin, N), dtype=np.float64)
    w_bt = np.zeros((n_bins, snps_per_bin, N), dtype=np.float64)
    kWdiff_bt = np.zeros((n_bins, snps_per_bin, N), dtype=np.float64)
    for b in prange(n_bins):
        for s in range(N):
            c0 = 0.0
            cm01 = 0.0
            observed_count = 0
            for t in range(snps_per_bin):
                l = b * snps_per_bin + t
                if l < L:
                    a0 = log_probs[s, l, 0]
                    a1 = log_probs[s, l, 1]
                    a2 = log_probs[s, l, 2]
                    m01 = a0 if a0 > a1 else a1
                    m12 = a1 if a1 > a2 else a2
                    diff1_bt[b, t, s] = a1 - a0
                    w_bt[b, t, s] = a0 - 2.0 * a1 + a2
                    kWdiff_bt[b, t, s] = m12 - m01
                    c0 += a0
                    cm01 += m01
                    if observed_mask[s, l]:
                        observed_count += 1
            C0b[s, b] = c0
            kW_Cb[s, b] = cm01 - observed_count * lam
    return C0b, diff1_bt, w_bt, kW_Cb, kWdiff_bt


def _update_A_blas_lp_precompute_depth_mask(
        log_probs, observed_mask, lam, snps_per_bin, n_bins):
    """Prepare assignment contractions with lambda charged only when observed."""
    lp_c = discovery_objectives._maybe_c_contig(log_probs, np.float64)
    mask_c = discovery_objectives._maybe_c_contig(observed_mask, np.bool_)
    if mask_c.shape != lp_c.shape[:2]:
        raise ValueError("observed_mask must match evidence samples and sites")
    return _update_A_blas_lp_precompute_depth_mask_kernel(
        lp_c, mask_c, float(lam), int(snps_per_bin), int(n_bins),
        int(lp_c.shape[1]),
    )


_BINARY_PATTERN_MAX_SNPS_PER_BIN = 10


class _BinaryPatternContractionWorkspace:
    """Read-only evidence-local contractions for every binary bin pattern.

    For production N=116, L=200, and ten sites per bin, each of the three
    arrays has shape ``(20, 1024, 116)`` and occupies 18.125 MiB; retained
    table storage is therefore exactly 54.375 MiB (57,016,320 bytes).  Storage
    is fixed at construction and never grows with the number of fits.
    """

    __slots__ = (
        "blas_lp_reference",
        "snps_per_bin",
        "n_bins",
        "n_samples",
        "n_patterns",
        "diff1_table",
        "w_table",
        "kWdiff_table",
        "nbytes",
    )

    def __init__(self, blas_lp_reference, snps_per_bin, n_bins,
                 diff1_table, w_table, kWdiff_table):
        self.blas_lp_reference = blas_lp_reference
        self.snps_per_bin = int(snps_per_bin)
        self.n_bins = int(n_bins)
        self.n_samples = int(diff1_table.shape[2])
        self.n_patterns = int(diff1_table.shape[1])
        self.diff1_table = diff1_table
        self.w_table = w_table
        self.kWdiff_table = kWdiff_table
        self.nbytes = int(
            diff1_table.nbytes + w_table.nbytes + kWdiff_table.nbytes
        )

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("binary-pattern workspace is immutable")
        object.__setattr__(self, name, value)

    def supports_k(self, k):
        """Return whether the canonical contraction supports this K.

        Every within-bin binary pattern is contracted once with one fixed
        numerical definition, independent of the K values subsequently
        visited by adaptive search. Reconstructing three complete tables at
        every new K to test a BLAS row-shape implementation detail added no
        statistical information and made results route-dependent.

        K=0 has no founder state. Every positive K uses the canonical table
        with unchanged state ordering, accumulation order, and tie rules.
        """
        return int(k) >= 1


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _canonical_binary_pattern_contractions(
        diff1_bt, w_bt, kWdiff_bt, previous_pattern, added_bit):
    """Build all binary contractions with one K-independent reduction.

    For each pattern, the highest set bit extends the already-computed lower
    prefix. This is the ascending-set-bit sum for that pattern, costs one
    addition per output rather than a dense ten-term dot product, and is
    independent of BLAS microkernels, thread count, and the K search route.
    """
    n_bins, snps_per_bin, n_samples = diff1_bt.shape
    n_patterns = 1 << snps_per_bin
    diff1_table = np.empty(
        (n_bins, n_patterns, n_samples), dtype=np.float64
    )
    w_table = np.empty_like(diff1_table)
    kWdiff_table = np.empty_like(diff1_table)
    for flat_index in prange(n_bins * n_samples):
        b = flat_index // n_samples
        s = flat_index - b * n_samples
        diff1_table[b, 0, s] = 0.0
        w_table[b, 0, s] = 0.0
        kWdiff_table[b, 0, s] = 0.0
        for pattern in range(1, n_patterns):
            highest_bit = added_bit[pattern]
            previous = previous_pattern[pattern]
            diff1_table[b, pattern, s] = (
                diff1_table[b, previous, s]
                + diff1_bt[b, highest_bit, s]
            )
            w_table[b, pattern, s] = (
                w_table[b, previous, s]
                + w_bt[b, highest_bit, s]
            )
            kWdiff_table[b, pattern, s] = (
                kWdiff_table[b, previous, s]
                + kWdiff_bt[b, highest_bit, s]
            )
    return diff1_table, w_table, kWdiff_table


def _prepare_binary_pattern_contractions(blas_lp_cache, snps_per_bin, n_bins):
    """Contract every supported binary within-bin pattern exactly once.

    A compiled subset recurrence gives every pattern one fixed ascending-bit
    float64 reduction, independent of K, BLAS implementation, and thread
    allocation. Wider bins would grow exponentially and deliberately fall
    back to the existing direct path.
    """
    spb = int(snps_per_bin)
    bins = int(n_bins)
    if spb < 1 or spb > _BINARY_PATTERN_MAX_SNPS_PER_BIN or bins < 1:
        return None
    if blas_lp_cache is None or len(blas_lp_cache) != 5:
        return None
    _C0b, diff1_bt, w_bt, _kW_Cb, kWdiff_bt = blas_lp_cache
    if (
        diff1_bt.ndim != 3
        or w_bt.shape != diff1_bt.shape
        or kWdiff_bt.shape != diff1_bt.shape
        or diff1_bt.shape[0] != bins
        or diff1_bt.shape[1] != spb
        or diff1_bt.dtype != np.float64
        or w_bt.dtype != np.float64
        or kWdiff_bt.dtype != np.float64
    ):
        return None

    n_patterns = 1 << spb
    previous_pattern = np.zeros(n_patterns, dtype=np.int64)
    added_bit = np.zeros(n_patterns, dtype=np.int64)
    for pattern in range(1, n_patterns):
        highest_bit = int(pattern).bit_length() - 1
        added_bit[pattern] = highest_bit
        previous_pattern[pattern] = pattern ^ (1 << highest_bit)
    diff1_table, w_table, kWdiff_table = (
        _canonical_binary_pattern_contractions(
            np.ascontiguousarray(diff1_bt),
            np.ascontiguousarray(w_bt),
            np.ascontiguousarray(kWdiff_bt),
            previous_pattern,
            added_bit,
        )
    )
    for table in (diff1_table, w_table, kWdiff_table):
        table.setflags(write=False)


    expected_nbytes = (
        3 * bins * n_patterns * int(diff1_bt.shape[2])
        * np.dtype(np.float64).itemsize
    )
    actual_nbytes = int(
        diff1_table.nbytes + w_table.nbytes + kWdiff_table.nbytes
    )
    if actual_nbytes != expected_nbytes:
        raise AssertionError("binary-pattern table byte accounting mismatch")
    return _BinaryPatternContractionWorkspace(
        blas_lp_cache,
        spb,
        bins,
        diff1_table,
        w_table,
        kWdiff_table,
    )


@njit(cache=True, nogil=True)
def _binary_haplotype_patterns(H, rr_i, rr_j,
                               snps_per_bin, n_bins, L):
    """Encode binary H rows and pairwise products, or request fallback."""
    K = H.shape[0]
    n_rr = rr_i.shape[0]
    h_patterns = np.zeros((K, n_bins), dtype=np.int64)
    pair_patterns = np.empty((n_rr, n_bins), dtype=np.int64)
    for k in range(K):
        for b in range(n_bins):
            pattern = 0
            for t in range(snps_per_bin):
                l = b * snps_per_bin + t
                if l < L:
                    value = H[k, l]
                    if value == 1:
                        pattern |= 1 << t
                    elif value != 0:
                        return False, h_patterns, pair_patterns
            h_patterns[k, b] = pattern
    for p in range(n_rr):
        i = rr_i[p]
        j = rr_j[p]
        for b in range(n_bins):
            pair_patterns[p, b] = h_patterns[i, b] & h_patterns[j, b]
    return True, h_patterns, pair_patterns


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _update_A_fused_blas_kernel(C0b, Ub, Mb, kW_Cb, kW_Ub,
                                 cost_WW, WW_bin_emis,
                                 rr_i, rr_j, penalty, K, n_bins, L):
    """BLAS-fed fused argmin + Viterbi-BIC kernel — numerically
    equivalent to a direct scalar per-(state, site) summation of the
    state emissions, but with that walk replaced by precomputed GEMM
    contractions.

    Equivalence to the direct scalar form:
      - rr state (i, j): per-bin emission assembled as
            C0b[s,b] + Ub[i,s,b] + Ub[j,s,b] + Mb[p,s,b]
        which equals a direct sum_{l in bin} log_probs[s,l,d]
        (d = H[i,l] + H[j,l]) up to floating-point reassociation only
        (~1e-13): the BLAS contractions in Ub/Mb sum within each bin in
        blocked order rather than strictly left-to-right.  See
        `_update_A_blas_lp_precompute` for the exact algebraic identity.
      - kW state (k, W): per-bin emission assembled as
            kW_Cb[s,b] + kW_Ub[k,s,b]
        equal to a direct sum_{l in bin} (max(log_probs[s,l,d0],
        log_probs[s,l,d1]) - lam) up to the same reassociation; the max
        is taken per site in the precompute (m01 = max(a0,a1), m12 =
        max(a1,a2)), so monotonicity of log is preserved exactly.
      - WW state: state cost is summed from cost_WW LEFT-TO-RIGHT and the
        bin emissions are copied from WW_bin_emis — BIT-IDENTICAL to the
        direct scalar sum (zero reassociation on this state).

    The state order (rr (i, j) row-major in [0, n_rr), then (k, W) in
    [n_rr, n_rr+K), then (W, W) at n_rr+K), the argmin (strict-< first-
    occurrence tiebreak), the rr-index decode, and the in-place Viterbi
    forward all match the direct scalar form exactly, so the only
    departure from bit-identity is the rr/kW reassociation above.  Each
    state's per-bin emission is assembled ON THE FLY (a few adds per
    (state, bin)); no (K_states, n_bins, N) tensor is materialised.

    Inputs:
        C0b:         (N, n_bins)        float64 — sum_bin a0 (precompute)
        Ub:          (K, N, n_bins)     float64 — H @ (a1-a0), binned
        Mb:          (n_rr, N, n_bins)  float64 — (H_i*H_j) @ (a0-2a1+a2)
        kW_Cb:       (N, n_bins)        float64 — sum_bin m01 - n_real*lam
        kW_Ub:       (K, N, n_bins)     float64 — H @ (m12-m01), binned
        cost_WW:     (N, L)             float64 — per-(sample,site) WW cost
        WW_bin_emis: (N, n_bins)        float64 — WW per-bin LL emission
        rr_i, rr_j:  (n_rr,)            int64   — first/second hap index of
                                                  each rr pair (i <= j)
        penalty:     Viterbi switch penalty between adjacent bins
        K:           number of founders
        n_bins:      bin count
        L:           number of kept sites (for the WW flat sum)

    Returns:
        A:               (N, 2)   int64
        baseline_cost:   (N,)     float64 — UNCAPPED best-pair NLL/sample
        wildcard_slots:  (N,)     int64
        viterbi_ll:      (N,)     float64 — best Viterbi-path LL/sample
    """
    N = C0b.shape[0]
    n_rr = Mb.shape[0]
    K_states = n_rr + K + 1
    W = K   # wildcard sentinel

    A = np.empty((N, 2), dtype=np.int64)
    baseline_cost = np.empty(N, dtype=np.float64)
    wildcard_slots = np.empty(N, dtype=np.int64)
    viterbi_ll = np.empty(N, dtype=np.float64)

    for s in prange(N):
        bin_emis = np.empty((K_states, n_bins), dtype=np.float64)
        state_cost = np.empty(K_states, dtype=np.float64)
        st = 0

        # ----- Real-real pairs (i, j) with i <= j (row-major) -----
        # Per-bin emission C0b + Ub[i] + Ub[j] + Mb[pair]; state cost is
        # the negated sum over bins (= -total LL under this pair).
        for p in range(n_rr):
            i = rr_i[p]
            j = rr_j[p]
            cost_total = 0.0
            for b in range(n_bins):
                e = C0b[s, b] + Ub[i, s, b] + Ub[j, s, b] + Mb[p, s, b]
                bin_emis[st, b] = e
                cost_total += e
            state_cost[st] = -cost_total
            st += 1

        # ----- Real-W pairs (k, W) -----
        for k in range(K):
            cost_total = 0.0
            for b in range(n_bins):
                e = kW_Cb[s, b] + kW_Ub[k, s, b]
                bin_emis[st, b] = e
                cost_total += e
            state_cost[st] = -cost_total
            st += 1

        # ----- (W, W) — bit-identical to the scalar fused kernel -----
        # State cost: sum_l cost_WW[s, l] left-to-right; bin emissions are
        # the precomputed WW_bin_emis row.
        cost_total = 0.0
        for l in range(L):
            cost_total += cost_WW[s, l]
        for b in range(n_bins):
            bin_emis[st, b] = WW_bin_emis[s, b]
        state_cost[st] = cost_total

        # ----- Baseline argmin (strict-< first-occurrence tiebreak) -----
        best_cost = np.inf
        best_state_idx = 0
        for st_iter in range(K_states):
            if state_cost[st_iter] < best_cost:
                best_cost = state_cost[st_iter]
                best_state_idx = st_iter

        if best_state_idx < n_rr:
            remaining = best_state_idx
            for i in range(K):
                row_len = K - i           # pairs (i, j) with j in [i, K)
                if remaining < row_len:
                    A[s, 0] = i
                    A[s, 1] = i + remaining
                    wildcard_slots[s] = 0
                    break
                remaining -= row_len
        elif best_state_idx < n_rr + K:
            A[s, 0] = best_state_idx - n_rr
            A[s, 1] = W
            wildcard_slots[s] = 1
        else:
            A[s, 0] = W
            A[s, 1] = W
            wildcard_slots[s] = 2

        baseline_cost[s] = best_cost

        # ----- Viterbi forward on bin_emis (in-place alpha) -----
        alpha = np.empty(K_states, dtype=np.float64)
        for st_iter in range(K_states):
            alpha[st_iter] = bin_emis[st_iter, 0]
        for b in range(1, n_bins):
            best_prev = -np.inf
            for st_iter in range(K_states):
                if alpha[st_iter] > best_prev:
                    best_prev = alpha[st_iter]
            switch_base = best_prev - penalty
            for st_iter in range(K_states):
                em = bin_emis[st_iter, b]
                stay = alpha[st_iter]
                if stay > switch_base:
                    alpha[st_iter] = stay + em
                else:
                    alpha[st_iter] = switch_base + em
        best_final = -np.inf
        for st_iter in range(K_states):
            if alpha[st_iter] > best_final:
                best_final = alpha[st_iter]
        viterbi_ll[s] = best_final

    return A, baseline_cost, wildcard_slots, viterbi_ll


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _update_A_fused_pattern_kernel(C0b, diff1_table, w_table,
                                    kW_Cb, kWdiff_table,
                                    h_patterns, pair_patterns,
                                    WW_bin_emis, WW_total_cost,
                                    rr_i, rr_j, penalty, K, n_bins):
    """Exact table-fed counterpart of `_update_A_fused_blas_kernel`.

    Table entries are the same float64 BLAS contractions formerly supplied as
    Ub, Mb, and kW_Ub.  State order, emission-expression order, baseline
    argmin, Viterbi forward pass, strict tie rules, WW summation, and returned
    arrays are otherwise identical to the established kernel.
    """
    N = C0b.shape[0]
    n_rr = pair_patterns.shape[0]
    K_states = n_rr + K + 1
    W = K   # wildcard sentinel

    A = np.empty((N, 2), dtype=np.int64)
    wildcard_slots = np.empty(N, dtype=np.int64)
    viterbi_ll = np.empty(N, dtype=np.float64)

    for s in prange(N):
        # Each founder contraction is reused in every rr state containing
        # that founder.  Above very small K, gather those scattered pattern-
        # table reads once into a compact sample-local array: this changes no
        # arithmetic (the exact same table scalar enters the exact same
        # expression) while reducing diff1 lookups from 2*n_rr*n_bins to
        # K*n_bins.  K<4 retains the direct route because the gather setup is
        # not amortised there.
        gather_founder_terms = K >= 4
        if gather_founder_terms:
            founder_diff1 = np.empty((K, n_bins), dtype=np.float64)
            for k in range(K):
                for b in range(n_bins):
                    founder_diff1[k, b] = diff1_table[
                        b, h_patterns[k, b], s
                    ]

        bin_emis = np.empty((K_states, n_bins), dtype=np.float64)
        best_cost = np.inf
        best_state_idx = 0
        st = 0

        # ----- Real-real pairs (i, j) with i <= j (row-major) -----
        # Per-bin emission C0b + Ub[i] + Ub[j] + Mb[pair]; state cost is
        # the negated sum over bins (= -total LL under this pair).
        for p in range(n_rr):
            i = rr_i[p]
            j = rr_j[p]
            cost_total = 0.0
            for b in range(n_bins):
                if gather_founder_terms:
                    e = (
                        C0b[s, b]
                        + founder_diff1[i, b]
                        + founder_diff1[j, b]
                        + w_table[b, pair_patterns[p, b], s]
                    )
                else:
                    e = (
                        C0b[s, b]
                        + diff1_table[b, h_patterns[i, b], s]
                        + diff1_table[b, h_patterns[j, b], s]
                        + w_table[b, pair_patterns[p, b], s]
                    )
                bin_emis[st, b] = e
                cost_total += e
            candidate_cost = -cost_total
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_state_idx = st
            st += 1

        # ----- Real-W pairs (k, W) -----
        for k in range(K):
            cost_total = 0.0
            for b in range(n_bins):
                e = (
                    kW_Cb[s, b]
                    + kWdiff_table[b, h_patterns[k, b], s]
                )
                bin_emis[st, b] = e
                cost_total += e
            candidate_cost = -cost_total
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_state_idx = st
            st += 1

        # ----- (W, W) — bit-identical to the scalar fused kernel -----
        # State cost: sum_l cost_WW[s, l] left-to-right; bin emissions are
        # the precomputed WW_bin_emis row.
        for b in range(n_bins):
            bin_emis[st, b] = WW_bin_emis[s, b]
        candidate_cost = WW_total_cost[s]
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_state_idx = st

        if best_state_idx < n_rr:
            A[s, 0] = rr_i[best_state_idx]
            A[s, 1] = rr_j[best_state_idx]
            wildcard_slots[s] = 0
        elif best_state_idx < n_rr + K:
            A[s, 0] = best_state_idx - n_rr
            A[s, 1] = W
            wildcard_slots[s] = 1
        else:
            A[s, 0] = W
            A[s, 1] = W
            wildcard_slots[s] = 2

        # ----- Viterbi forward on bin_emis (in-place alpha) -----
        # Reuse the dead first emission column as the Viterbi alpha row.
        for b in range(1, n_bins):
            best_prev = -np.inf
            for st_iter in range(K_states):
                if bin_emis[st_iter, 0] > best_prev:
                    best_prev = bin_emis[st_iter, 0]
            switch_base = best_prev - penalty
            for st_iter in range(K_states):
                em = bin_emis[st_iter, b]
                stay = bin_emis[st_iter, 0]
                if stay > switch_base:
                    bin_emis[st_iter, 0] = stay + em
                else:
                    bin_emis[st_iter, 0] = switch_base + em
        best_final = -np.inf
        for st_iter in range(K_states):
            if bin_emis[st_iter, 0] > best_final:
                best_final = bin_emis[st_iter, 0]
        viterbi_ll[s] = best_final

    return A, -viterbi_ll, wildcard_slots


@njit(cache=True, parallel=False, fastmath=False, nogil=True)
def _update_A_fused_pattern_kernel_serial(C0b, diff1_table, w_table,
                                           kW_Cb, kWdiff_table,
                                           h_patterns, pair_patterns,
                                           WW_bin_emis, WW_total_cost,
                                           rr_i, rr_j, penalty, K, n_bins):
    """Serial one-thread clone of `_update_A_fused_pattern_kernel`.

    Table entries are the same float64 BLAS contractions formerly supplied as
    Ub, Mb, and kW_Ub.  State order, emission-expression order, baseline
    argmin, Viterbi forward pass, strict tie rules, WW summation, and returned
    arrays are otherwise identical to the established kernel.
    """
    N = C0b.shape[0]
    n_rr = pair_patterns.shape[0]
    K_states = n_rr + K + 1
    W = K   # wildcard sentinel

    A = np.empty((N, 2), dtype=np.int64)
    wildcard_slots = np.empty(N, dtype=np.int64)
    viterbi_ll = np.empty(N, dtype=np.float64)

    # The serial driver reuses one scratch slab across samples. Every entry
    # is overwritten before it is read, so this removes N heap allocations
    # per transition without changing any state or bin arithmetic.
    gather_founder_terms = K >= 4
    founder_diff1 = np.empty((K, n_bins), dtype=np.float64)
    bin_emis = np.empty((K_states, n_bins), dtype=np.float64)

    for s in range(N):
        # Each founder contraction is reused in every rr state containing
        # that founder.  Above very small K, gather those scattered pattern-
        # table reads once into a compact sample-local array: this changes no
        # arithmetic (the exact same table scalar enters the exact same
        # expression) while reducing diff1 lookups from 2*n_rr*n_bins to
        # K*n_bins.  K<4 retains the direct route because the gather setup is
        # not amortised there.
        if gather_founder_terms:
            for k in range(K):
                for b in range(n_bins):
                    founder_diff1[k, b] = diff1_table[
                        b, h_patterns[k, b], s
                    ]

        best_cost = np.inf
        best_state_idx = 0
        st = 0

        # ----- Real-real pairs (i, j) with i <= j (row-major) -----
        # Per-bin emission C0b + Ub[i] + Ub[j] + Mb[pair]; state cost is
        # the negated sum over bins (= -total LL under this pair).
        for p in range(n_rr):
            i = rr_i[p]
            j = rr_j[p]
            cost_total = 0.0
            for b in range(n_bins):
                if gather_founder_terms:
                    e = (
                        C0b[s, b]
                        + founder_diff1[i, b]
                        + founder_diff1[j, b]
                        + w_table[b, pair_patterns[p, b], s]
                    )
                else:
                    e = (
                        C0b[s, b]
                        + diff1_table[b, h_patterns[i, b], s]
                        + diff1_table[b, h_patterns[j, b], s]
                        + w_table[b, pair_patterns[p, b], s]
                    )
                bin_emis[st, b] = e
                cost_total += e
            candidate_cost = -cost_total
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_state_idx = st
            st += 1

        # ----- Real-W pairs (k, W) -----
        for k in range(K):
            cost_total = 0.0
            for b in range(n_bins):
                e = (
                    kW_Cb[s, b]
                    + kWdiff_table[b, h_patterns[k, b], s]
                )
                bin_emis[st, b] = e
                cost_total += e
            candidate_cost = -cost_total
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_state_idx = st
            st += 1

        # ----- (W, W) — bit-identical to the scalar fused kernel -----
        # State cost: sum_l cost_WW[s, l] left-to-right; bin emissions are
        # the precomputed WW_bin_emis row.
        for b in range(n_bins):
            bin_emis[st, b] = WW_bin_emis[s, b]
        candidate_cost = WW_total_cost[s]
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_state_idx = st

        if best_state_idx < n_rr:
            A[s, 0] = rr_i[best_state_idx]
            A[s, 1] = rr_j[best_state_idx]
            wildcard_slots[s] = 0
        elif best_state_idx < n_rr + K:
            A[s, 0] = best_state_idx - n_rr
            A[s, 1] = W
            wildcard_slots[s] = 1
        else:
            A[s, 0] = W
            A[s, 1] = W
            wildcard_slots[s] = 2

        # ----- Viterbi forward on bin_emis (in-place alpha) -----
        # Reuse the dead first emission column as the Viterbi alpha row.
        for b in range(1, n_bins):
            best_prev = -np.inf
            for st_iter in range(K_states):
                if bin_emis[st_iter, 0] > best_prev:
                    best_prev = bin_emis[st_iter, 0]
            switch_base = best_prev - penalty
            for st_iter in range(K_states):
                em = bin_emis[st_iter, b]
                stay = bin_emis[st_iter, 0]
                if stay > switch_base:
                    bin_emis[st_iter, 0] = stay + em
                else:
                    bin_emis[st_iter, 0] = switch_base + em
        best_final = -np.inf
        for st_iter in range(K_states):
            if bin_emis[st_iter, 0] > best_final:
                best_final = bin_emis[st_iter, 0]
        viterbi_ll[s] = best_final

    return A, -viterbi_ll, wildcard_slots

import haplotype_reconstruction.core.config as core_config
import haplotype_reconstruction.discovery.fitting as discovery_fitting
import haplotype_reconstruction.discovery.objectives as discovery_objectives
