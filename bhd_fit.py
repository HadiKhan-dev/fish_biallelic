"""EM coordinate-descent fitter (update A / H / one-founder, fixed-K fit) plus BIC
model selection, split out of bhd_kernels.  Imports the Viterbi / log / cost
primitives it needs from the bhd_kernels foundation."""

import numpy as np
import math
from numba import njit, prange

from bhd_kernels import (
    _log_probs_kernel,
    _maybe_c_contig,
    _per_site_cost_W_W,
    _viterbi_nll,
    _ww_bin_emis_from_cost_ww,
)
from bhd_config import (
    VITERBI_SNPS_PER_BIN,
    VITERBI_SWITCH_PENALTY,
    _VITERBI_BIC_ENABLED,
)


# =============================================================================
# UPDATE STEP A: pair assignments per sample
# =============================================================================

def _update_A(probs_k, H_k, lam, cost_WW=None, WW_bin_emis=None, log_probs=None,
              blas_lp_cache=None):
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

    probs_c = _maybe_c_contig(probs_k, np.float64)
    H_c = _maybe_c_contig(H_k, np.int64)

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
    if _VITERBI_BIC_ENABLED and K > 0:
        # Bin sizing — same logic as _viterbi_ll_per_sample so the
        # fused kernel's bin_emis matches _viterbi_binned_emissions_-
        # kernel's output bit-for-bit.
        if VITERBI_SNPS_PER_BIN > 1 and VITERBI_SNPS_PER_BIN < L:
            n_bins = int(math.ceil(L / VITERBI_SNPS_PER_BIN))
            snps_per_bin = VITERBI_SNPS_PER_BIN
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
            cost_WW = _per_site_cost_W_W(probs_c, float(lam))
        cost_WW_c = _maybe_c_contig(cost_WW, np.float64)
        if WW_bin_emis is None:
            WW_bin_emis = _ww_bin_emis_from_cost_ww(
                cost_WW_c, int(snps_per_bin), int(n_bins))
        WW_bin_emis_c = _maybe_c_contig(WW_bin_emis, np.float64)

        # Precompute log_probs once if not provided (Tier 0).  Like
        # cost_WW, log_probs is invariant across CD iterations because
        # probs_k doesn't change, so `_fit_at_fixed_K` caches it once
        # per invocation.  Standalone callers (e.g. unit tests, the
        # recovery loop's _update_A calls outside the CD loop) get
        # internal computation here.
        if log_probs is None:
            log_probs = _log_probs_kernel(probs_c)
        log_probs_c = _maybe_c_contig(log_probs, np.float64)

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
        rr_i, rr_j = np.triu_indices(K)
        rr_i = rr_i.astype(np.int64)
        rr_j = rr_j.astype(np.int64)

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
                rr_i, rr_j, float(VITERBI_SWITCH_PENALTY),
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
    N = log_probs.shape[0]
    L = log_probs.shape[1]
    Lpad = n_bins * snps_per_bin
    if Lpad == L:
        a0 = np.ascontiguousarray(log_probs[:, :, 0])
        a1 = np.ascontiguousarray(log_probs[:, :, 1])
        a2 = np.ascontiguousarray(log_probs[:, :, 2])
    else:
        a0 = np.zeros((N, Lpad), dtype=np.float64)
        a1 = np.zeros((N, Lpad), dtype=np.float64)
        a2 = np.zeros((N, Lpad), dtype=np.float64)
        a0[:, :L] = log_probs[:, :, 0]
        a1[:, :L] = log_probs[:, :, 1]
        a2[:, :L] = log_probs[:, :, 2]

    C0b = np.ascontiguousarray(
        a0.reshape(N, n_bins, snps_per_bin).sum(axis=2))
    diff1_bt = np.ascontiguousarray(
        (a1 - a0).reshape(N, n_bins, snps_per_bin).transpose(1, 2, 0))
    w_bt = np.ascontiguousarray(
        (a0 - 2.0 * a1 + a2).reshape(N, n_bins, snps_per_bin).transpose(1, 2, 0))

    m01 = np.maximum(a0, a1)
    m12 = np.maximum(a1, a2)
    real_per_bin = np.empty(n_bins, dtype=np.float64)
    for b in range(n_bins):
        start = b * snps_per_bin
        end = start + snps_per_bin
        if end > L:
            end = L
        cnt = end - start
        if cnt < 0:
            cnt = 0
        real_per_bin[b] = cnt
    kW_Cb = np.ascontiguousarray(
        m01.reshape(N, n_bins, snps_per_bin).sum(axis=2)
        - real_per_bin[None, :] * lam)
    kWdiff_bt = np.ascontiguousarray(
        (m12 - m01).reshape(N, n_bins, snps_per_bin).transpose(1, 2, 0))

    return C0b, diff1_bt, w_bt, kW_Cb, kWdiff_bt


@njit(cache=True, parallel=True, fastmath=False)
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


# =============================================================================
# UPDATE STEP H: founder allele updates
# =============================================================================

def _update_H(probs_k, H_k, A, lam, cost_WW=None, log_probs=None):
    """For each (founder, kept site), pick the binary value that minimises
    NLL contribution from samples carrying that founder.

    Updates H_k in-place and returns the number of bits flipped (so the
    coordinate descent loop can detect convergence).

    Arguments:
        probs_k: (N, L_kept, 3)
        H_k:     (K, L_kept) — modified in place
        A:       (N, 2)   pair assignments, with K used as the wildcard sentinel
        lam:     wildcard penalty
        cost_WW: (N, L_kept) optional — precomputed per-(sample, site) WW
                  cost from `_per_site_cost_W_W(probs_k, lam)`.  When
                  provided, threads through to `_update_one_founder` and
                  ultimately to the kernel, which reads cost_WW[s, l]
                  instead of recomputing it inline for every (k, l).
                  When None, the kernel falls back to inline computation
                  (so the wrapper computes it once here for consistency).
        log_probs: (N, L_kept, 3) optional — precomputed
                  log(max(probs_k[s, l, g], LOG_EPS_LOCAL)) (Tier 0).
                  When provided, the kernel uses it in place of every
                  inline log call.  Same caching pattern as cost_WW.

    Returns:
        n_changes: int — number of (founder, site) bits that flipped
    """
    K, L = H_k.shape
    N = probs_k.shape[0]
    W = K

    # Compute cost_WW once here if not supplied, so the K founder
    # updates below all share the same precomputed array (no per-
    # founder re-derivation).
    if cost_WW is None:
        cost_WW = _per_site_cost_W_W(probs_k, lam)

    # Same pattern for log_probs (Tier 0).  When called from
    # `_fit_at_fixed_K`, log_probs arrives precomputed and is reused
    # across all K founder updates AND across all CD iterations.
    if log_probs is None:
        log_probs = _log_probs_kernel(_maybe_c_contig(probs_k, np.float64))

    # We update founders in decreasing order of usage.  Compute usage from A
    # via an njit kernel — the inner Python loop over K with two boolean-
    # mask sums per K was a small hot spot (4K mask scans, each scanning
    # N entries).  The kernel does it in a single pass over A.
    A_c = _maybe_c_contig(A, np.int64)
    usage = _update_H_usage_kernel(A_c, K)
    update_order = np.argsort(-usage, kind='stable')

    n_changes = 0
    for k in update_order:
        n_changes += _update_one_founder(probs_k, H_k, A, int(k), lam,
                                          cost_WW=cost_WW, log_probs=log_probs)
    return n_changes


@njit(cache=True)
def _update_H_usage_kernel(A, K):
    """Count per-founder usage from pair-assignment array A.

    A entries are in [0, K]; entries equal to K are the wildcard sentinel
    and don't count.  For each k in [0, K), usage[k] = number of (s, slot)
    entries with A[s, slot] == k.  A pair (k, k) contributes 2.
    """
    N = A.shape[0]
    usage = np.zeros(K, dtype=np.int64)
    for s in range(N):
        for slot in range(2):
            f = A[s, slot]
            if f != K:
                usage[f] += 1
    return usage


def _update_one_founder(probs_k, H_k, A, k, lam, cost_WW=None, log_probs=None):
    """For founder k, at each kept site, evaluate cost contribution from
    samples carrying k under H_k[k, l] = 0 vs = 1, and pick the lower.

    Per-(sample, site) costs are capped at cost_WW(s, l) (Fix H).  This
    implements per-strand-per-site wildcard escape: a sample whose
    pair-fit at site l exceeds cost_WW (i.e., the founder doesn't
    represent the sample at this site) contributes cost_WW to BOTH H=0
    and H=1, contributing zero preference.  This prevents non-carrier
    samples from contaminating the founder's update at incompatible
    sites while still allowing them to vote at agreeing sites.

    H_k is modified in place at row k.  Returns the number of sites flipped.

    cost_WW: optional precomputed (N, L) WW cost array.  When None,
    computed here so the kernel can read from it.  When provided
    (typical inside `_fit_at_fixed_K`'s CD loop), reused across
    K founders within one M-step iteration and across all CD iterations.

    log_probs: optional precomputed (N, L, 3) log array (Tier 0).  When
    provided, the kernel reads log_probs[s, l, g] in place of every
    inline math.log(probs[s, l, g]).  Stable across CD iterations and
    across M-step founders, so `_fit_at_fixed_K` precomputes once.
    """
    K, L = H_k.shape
    N = probs_k.shape[0]
    W = K

    # Identify samples whose pair contains k as a REAL strand.  We split
    # into three buckets by partner type:
    #   Bucket H: A[s] = (k, k)              — homozygous for k
    #   Bucket J: A[s] = (k, j) with j != k, j real   — k paired with another real founder
    #   Bucket P: A[s] = (k, W)              — k paired with wildcard
    # Note A is sorted ascending with W always last, so (k, k) means both
    # entries are k; (k, j) with j != k real means one entry is k and the
    # other is some j with j != k and j != W; (k, W) means one entry is k
    # and the other is W.
    is_kk = (A[:, 0] == k) & (A[:, 1] == k)
    is_kW = ((A[:, 0] == k) & (A[:, 1] == W))
    # A[:, 0] != A[:, 1] and one of them is k and the other is real (not W)
    has_k = (A[:, 0] == k) | (A[:, 1] == k)
    is_kj = has_k & ~is_kk & ~is_kW

    # If founder k has no support, leave H_k[k] untouched
    n_supp = int(is_kk.sum() + is_kj.sum() + is_kW.sum())
    if n_supp == 0:
        return 0

    # For each sample in bucket J, identify the partner founder index j_s
    if is_kj.any():
        # j_s = the entry in A[s] that's NOT k
        a0 = A[is_kj, 0]
        a1 = A[is_kj, 1]
        partner_J = np.where(a0 == k, a1, a0)               # (n_J,)
        kj_sample_idx = np.where(is_kj)[0]                  # (n_J,)
    else:
        partner_J = np.empty(0, dtype=np.int64)
        kj_sample_idx = np.empty(0, dtype=np.int64)

    kk_sample_idx = np.where(is_kk)[0]
    kW_sample_idx = np.where(is_kW)[0]

    # Compute cost_WW if not supplied.  This is the wrapper's last chance
    # to reuse a precomputed array; the kernel REQUIRES a passed-in
    # cost_WW (no internal fallback) because keeping the kernel signature
    # uniform avoids a code-duplicated cold path.
    if cost_WW is None:
        cost_WW = _per_site_cost_W_W(probs_k, lam)

    # Compute log_probs if not supplied (Tier 0).  Same kernel-requires-
    # uniform-signature rationale as cost_WW above.
    if log_probs is None:
        log_probs = _log_probs_kernel(_maybe_c_contig(probs_k, np.float64))

    # Hand off to the njit per-site kernel.  This replaces the original
    # Python `for l in range(L):` loop with O(L) per-site numpy slicing
    # (which had high overhead at L≈200 across many coord-descent
    # iterations).
    probs_c = _maybe_c_contig(probs_k, np.float64)
    H_row_c = _maybe_c_contig(H_k[k], np.int64)
    H_c = _maybe_c_contig(H_k, np.int64)
    kk_idx_c = _maybe_c_contig(kk_sample_idx, np.int64)
    kj_idx_c = _maybe_c_contig(kj_sample_idx, np.int64)
    kW_idx_c = _maybe_c_contig(kW_sample_idx, np.int64)
    partner_J_c = _maybe_c_contig(partner_J, np.int64)
    cost_WW_c = _maybe_c_contig(cost_WW, np.float64)
    log_probs_c = _maybe_c_contig(log_probs, np.float64)
    new_row, n_changes = _update_one_founder_kernel(
        probs_c, H_row_c, H_c,
        kk_idx_c, kj_idx_c, kW_idx_c, partner_J_c,
        float(lam), cost_WW_c, log_probs_c)
    # Write back into H_k in-place at row k.
    H_k[k] = new_row
    return int(n_changes)


@njit(cache=True, parallel=True, fastmath=False)
def _update_one_founder_kernel(probs_k, H_row, H_full,
                                kk_idx, kj_idx, kW_idx, partner_J,
                                lam, cost_WW, log_probs):
    """njit version of the per-site loop in _update_one_founder.

    Replaces the original Python `for l in range(L):` body with explicit
    per-bucket per-sample accumulation.  Each bucket's contribution to
    nll0 and nll1 is capped at cost_WW(s, l) — the cap mirrors Fix H
    exactly.

    Three changes from the original parallel=False version:
      1. cost_WW is passed in (precomputed once per `_fit_at_fixed_K`
         invocation) instead of recomputed inline per (s, l, bucket).
         Eliminates the redundant -log(p_max) + 2*lam computation that
         was previously done THREE TIMES per (s, l) within each kernel
         call (once per bucket type), times the K founders updated in
         one M-step iteration.
      2. log_probs is passed in (Tier 0 precompute).  All inline
         `-math.log(max(probs[s, l, d], EPS))` patterns become
         `-log_probs[s, l, d]`, eliminating the log() call (the kernel
         was log()-bound on bucket J under K=6).  Bucket P's max-pick
         on (p0, p1) and (p1, p2) uses monotonicity of log: select via
         comparison on log_probs values directly.
      3. The outer site loop is `prange(L)` (parallel=True) so that
         large numba thread budgets — which workers get dynamically via
         `dynamic_threads.apply_dynamic_threads` as the contig's slow-tail
         blocks finish — actually accelerate the M-step.  Each l writes an
         independent `new_row[l]` and the `n_changes` accumulator is a
         simple `+= 1` reduction that numba recognises automatically.

    Bit-equivalence: identical scalars in identical accumulation order
    within each l (sequential bucket loops); each l is independent so
    parallel iteration is safe.  Verified against the legacy parallel=
    False version at machine precision.

    Arguments:
        probs_k:    (N, L, 3) float64, C-contig — kept in the signature
                    for symmetry, but no longer read inside the kernel
                    body (all probability lookups go through log_probs).
                    Removing the parameter would force changes in every
                    caller; leaving it as a noop preserves wrapper code.
        H_row:      (L,) int64 — the CURRENT row k of H_k that will be
                    updated.  The function returns the NEW row; the
                    caller writes it back into H_k[k].  We pass the row
                    in (not just an empty buffer) so the "keep current
                    value when nll0 == nll1" branch sees the right
                    reference value.
        H_full:     (K, L) int64 — full H matrix; only column 'partner_J'
                    rows are read inside the loop (Bucket J uses partner
                    indices into H_full).
        kk_idx:     (n_kk,) int64 — sample indices in bucket H (kk)
        kj_idx:     (n_J,)  int64 — sample indices in bucket J
        kW_idx:     (n_P,)  int64 — sample indices in bucket P (kW)
        partner_J:  (n_J,)  int64 — partner founder index for each
                    bucket-J sample, in the same order as kj_idx
        lam:        wildcard penalty
        cost_WW:    (N, L) float64 — precomputed per-(sample, site) WW
                    cost from `_per_site_cost_W_W_kernel`.  Read inline
                    inside each bucket as the cap value, replacing the
                    previous inline pmax+log computation.
        log_probs:  (N, L, 3) float64 — precomputed log(max(probs[s, l, g],
                    LOG_EPS_LOCAL)).  Read in place of every inline log
                    call in the three buckets.  The LOG_EPS clamp is
                    applied at precompute time in `_log_probs_kernel`,
                    so the kernel body itself doesn't clamp.

    Returns:
        new_row:    (L,) int64 — new H_k[k] row
        n_changes:  int — number of (k, l) bits that flipped

    The "no-signal" branch (|nll0 - nll1| < 1e-9) keeps H_row[l]
    unchanged, matching the Python version's behaviour.
    """
    LOG_EPS_LOCAL = 1e-12

    L = H_row.shape[0]
    n_kk = kk_idx.shape[0]
    n_J = kj_idx.shape[0]
    n_P = kW_idx.shape[0]

    new_row = np.empty(L, dtype=np.int64)
    n_changes = 0

    for l in prange(L):
        cur_val = H_row[l]
        nll0 = 0.0
        nll1 = 0.0

        # Bucket H (k, k): dosage = 0 under hk=0, dosage = 2 under hk=1.
        for ii in range(n_kk):
            s = kk_idx[ii]
            cost_WW_s = cost_WW[s, l]

            # Raw cost under hk=0: -log p[s, l, 0]; under hk=1: -log p[s, l, 2].
            # Read precomputed log values (Tier 0); EPS clamp was applied
            # in `_log_probs_kernel`.
            c_h0 = -log_probs[s, l, 0]
            c_h1 = -log_probs[s, l, 2]

            # Cap at cost_WW.
            if c_h0 > cost_WW_s:
                c_h0 = cost_WW_s
            if c_h1 > cost_WW_s:
                c_h1 = cost_WW_s
            nll0 += c_h0
            nll1 += c_h1

        # Bucket J (k, j): dosage = partner_h_at_l + hk.
        for ii in range(n_J):
            s = kj_idx[ii]
            j = partner_J[ii]
            cost_WW_s = cost_WW[s, l]
            partner_h = H_full[j, l]

            d_h0 = partner_h          # dosage if hk=0
            d_h1 = partner_h + 1      # dosage if hk=1
            c_h0 = -log_probs[s, l, d_h0]
            c_h1 = -log_probs[s, l, d_h1]
            if c_h0 > cost_WW_s:
                c_h0 = cost_WW_s
            if c_h1 > cost_WW_s:
                c_h1 = cost_WW_s
            nll0 += c_h0
            nll1 += c_h1

        # Bucket P (k, W): wildcard strand picks its bit optimally.
        # Under hk=0: candidate dosages {0, 1}; under hk=1: {1, 2}.
        # Cost = -log max-prob + lam.
        # log(max(p_a, p_b)) = max(log p_a, log p_b) by monotonicity, so
        # we select the larger of the two precomputed log values directly
        # (no need to read probs and apply EPS clamp again — both were
        # done at precompute time).  The comparison on log_probs values
        # gives the same selection as the legacy comparison on raw probs
        # after EPS clamping, since log is strictly increasing.
        for ii in range(n_P):
            s = kW_idx[ii]
            cost_WW_s = cost_WW[s, l]
            lp0 = log_probs[s, l, 0]
            lp1 = log_probs[s, l, 1]
            lp2 = log_probs[s, l, 2]

            # hk=0: best of (p0, p1) -> max(lp0, lp1)
            best_lp0 = lp0 if lp0 > lp1 else lp1
            # hk=1: best of (p1, p2) -> max(lp1, lp2)
            best_lp1 = lp1 if lp1 > lp2 else lp2
            c_h0 = -best_lp0 + lam
            c_h1 = -best_lp1 + lam
            if c_h0 > cost_WW_s:
                c_h0 = cost_WW_s
            if c_h1 > cost_WW_s:
                c_h1 = cost_WW_s
            nll0 += c_h0
            nll1 += c_h1

        # Pick lower-NLL value.  No-signal handling: if nll0 == nll1
        # (within numerical precision), no sample expressed a meaningful
        # preference at this site (e.g., all attributing samples were
        # capped out under both H values, giving zero discriminating
        # signal).  In that case keep cur_val to avoid arbitrary flips.
        diff = nll0 - nll1
        if diff < 0.0:
            diff = -diff
        if diff < 1e-9:
            new_val = cur_val
        else:
            if nll0 < nll1:
                new_val = 0
            else:
                new_val = 1
        new_row[l] = new_val
        if new_val != cur_val:
            n_changes += 1

    return new_row, n_changes


# =============================================================================
# COORDINATE DESCENT AT FIXED K
# =============================================================================

def _fit_at_fixed_K(probs_k, H_init, lam, max_iter=50):
    """Run discrete coordinate descent at the K determined by H_init.shape[0].

    Alternates updating A (pair assignments) and H (founder bits) until
    no changes are made in a full pass, or max_iter is reached.

    Arguments:
        probs_k: (N, L_kept, 3) — kept-site posteriors
        H_init:  (K, L_kept) discrete {0, 1} — initial founder values
        lam:     wildcard penalty
        max_iter: cap on coordinate descent iterations

    Returns:
        H:               final (K, L_kept)
        A:               final (N, 2)
        per_sample_cost: (N,) total CAPPED cost per sample under final state
                          (used in worst-fit-sample selection for K-growth seed)
        wildcard_slots:  (N,) wildcard strand count per sample
        n_iter:          how many iterations were used
        total_NLL:       scalar — UNCAPPED NLL summed across samples (used
                          by K-growth as the improvement signal; the cap
                          would mask improvements where adding a founder
                          converts samples from "way over cost_WW" to
                          "still over cost_WW but less so")
    """
    H = H_init.copy()
    A_prev = None
    n_iter = 0

    # Initialise result variables so the post-loop block can refer to
    # them whether or not the loop body executed (max_iter=0 edge case)
    # or completed any iterations.
    A = None
    per_sample_cost = None
    per_sample_cost_unc = None
    wildcard_slots = None

    # ---------------------------------------------------------------------
    # Pre-bake invariant quantities ONCE per CD invocation.
    #
    # cost_WW depends only on (probs_k, lam) — neither changes inside the
    # CD loop — so it's wasteful to recompute it in every _update_A and
    # _update_H call.  Same for WW_bin_emis, which is a fixed binning of
    # cost_WW.  Same for log_probs (Tier 0), which depends only on
    # probs_k.  We compute all three here and thread them through.
    #
    # The fused _update_A_fused_blas_kernel takes the WW state's bin
    # emissions from WW_bin_emis (no inner WW site-loop), while the rr/kW
    # state emissions come from GEMM contractions of log_probs-derived
    # arrays (built once by _update_A_blas_lp_precompute) rather than
    # per-visit log() calls.  The M-step kernel _update_one_founder_kernel
    # reads cost_WW[s, l] for the cap value and log_probs[s, l, g] for the
    # per-genotype cost.
    # ---------------------------------------------------------------------
    L = probs_k.shape[1]
    cost_WW = _per_site_cost_W_W(probs_k, lam)
    probs_c = _maybe_c_contig(probs_k, np.float64)
    log_probs = _log_probs_kernel(probs_c)
    if _VITERBI_BIC_ENABLED:
        if VITERBI_SNPS_PER_BIN > 1 and VITERBI_SNPS_PER_BIN < L:
            _snps_per_bin = VITERBI_SNPS_PER_BIN
            _n_bins = int(math.ceil(L / VITERBI_SNPS_PER_BIN))
        else:
            _snps_per_bin = 1
            _n_bins = L
        cost_WW_c = _maybe_c_contig(cost_WW, np.float64)
        WW_bin_emis = _ww_bin_emis_from_cost_ww(
            cost_WW_c, int(_snps_per_bin), int(_n_bins))
        # Pre-bake the BLAS-hybrid kernel's log_probs-derived inputs ONCE
        # (invariant across CD iterations, exactly like cost_WW /
        # WW_bin_emis / log_probs above).  _update_A threads these through
        # so only the H-dependent GEMMs run per iteration — the bulk of
        # the K-growth speedup.  Skipped at K=0, where the baseline
        # _update_A path doesn't invoke the kernel.
        if H.shape[0] > 0:
            _blas_lp_cache = _update_A_blas_lp_precompute(
                _maybe_c_contig(log_probs, np.float64), float(lam),
                int(_snps_per_bin), int(_n_bins))
        else:
            _blas_lp_cache = None
    else:
        WW_bin_emis = None
        _blas_lp_cache = None

    # Tracks whether we need to recompute A and per-sample costs after
    # the loop exits.  When the loop exits via CD CONVERGENCE
    # (not a_changed and h_changes == 0 at the break point), the last
    # _update_H call did NOT change H, so the A computed earlier in the
    # same loop iteration is still consistent with the (unchanged) H.
    # In that case we can skip the post-loop _update_A — a clean win of
    # one _update_A call per converged _fit_at_fixed_K invocation, which
    # is the common case (most blocks converge in 3-10 iterations, well
    # under max_iter=50).
    #
    # When the loop exits via max_iter (no convergence), the LAST
    # _update_H call MAY have changed H, in which case A is now stale
    # and must be recomputed.  We can't tell whether h_changes was 0 at
    # the final iteration of the max_iter path, so we conservatively
    # recompute in that case.  (Cheap insurance — max_iter is rarely
    # reached.)
    need_recompute = True

    for it in range(max_iter):
        # Update A given H — pass precomputed WW arrays and log_probs
        # for the fused kernel to consume; identical results to
        # recomputing inside.
        A, per_sample_cost, per_sample_cost_unc, wildcard_slots = _update_A(
            probs_k, H, lam, cost_WW=cost_WW, WW_bin_emis=WW_bin_emis,
            log_probs=log_probs, blas_lp_cache=_blas_lp_cache)

        # Convergence check via A
        a_changed = (A_prev is None) or (not np.array_equal(A, A_prev))
        A_prev = A.copy()

        # Update H given A — pass precomputed cost_WW and log_probs for
        # the kernel to use directly.
        h_changes = _update_H(probs_k, H, A, lam, cost_WW=cost_WW,
                              log_probs=log_probs)

        n_iter = it + 1
        if not a_changed and h_changes == 0:
            # Converged: H didn't change in this iteration's _update_H
            # call, so the A computed above is still consistent.  Skip
            # the post-loop recompute.
            need_recompute = False
            break

    if need_recompute:
        # Either we never entered the loop (max_iter=0) or we exited via
        # max_iter without converging (in which case the final _update_H
        # call may have changed H, making A stale).  Recompute.
        A, per_sample_cost, per_sample_cost_unc, wildcard_slots = _update_A(
            probs_k, H, lam, cost_WW=cost_WW, WW_bin_emis=WW_bin_emis,
            log_probs=log_probs, blas_lp_cache=_blas_lp_cache)
    # Use UNCAPPED NLL as the K-growth signal (see docstring).
    total_NLL = float(per_sample_cost_unc.sum())

    return H, A, per_sample_cost, wildcard_slots, n_iter, total_NLL


# =============================================================================
# BIC HELPERS (the complexity-cost and BIC formula, shared by every BIC site)
# =============================================================================
#
# BIC convention used throughout this module (linear-BIC, project standard
# matching beam_search_core / chimera_resolution):
#
#     BIC(K) = K * cc + 2 * NLL_K          (lower is better)
#
# where cc is the per-founder complexity cost.  At fixed K, BIC and NLL
# differ only by an additive constant K*cc, so any fixed-K decision (e.g.
# per-(founder, site) bit voting in _update_one_founder, picking the best
# K_cur+1 candidate among K_cur+1 same-K candidates) is identical under
# either score.  Decisions across different K values (K-growth acceptance,
# multi-medoid trajectory selection, history reporting) MUST use BIC so
# the complexity penalty is properly accounted for.
#
# The acceptance criterion BIC(K+1) < BIC(K) reduces algebraically to
# NLL_improvement > cc/2; see _grow_K for the derivation.

def _compute_cc(cc_scale, N, L_kept, use_log_bic=False):
    """Per-founder complexity cost cc used in BIC = K * cc + 2 * NLL.

    Linear BIC (default, project standard):
        cc = cc_scale * (L_kept / 200) * N
    Standard log-BIC (use_log_bic=True):
        cc = cc_scale * log(N * L_kept) * (L_kept / 200)

    Linear scaling is preferred over log(N) at the per-block EM stage
    because log(N) is too weak when N is large (~320 here), allowing
    spurious founder additions to slip past the BIC threshold.  See
    chimera_resolution.compute_cc and _grow_K's docstring for the full
    rationale.
    """
    snp_growth = L_kept / 200.0
    if use_log_bic:
        log_n = math.log(max(N * L_kept, 2))
        return cc_scale * log_n * snp_growth
    return cc_scale * snp_growth * N


def _compute_bic(K, nll, cc):
    """BIC = K * cc + 2 * NLL.  Lower is better.

    Centralises the formula so every place that compares solutions
    across different K values uses the same convention.  At fixed K
    this is a constant offset from NLL (so ordering is preserved) but
    the absolute number is informative when comparing across K.
    """
    return K * cc + 2.0 * nll


def _compute_nll_for_subset(haps_list, probs_k, lam):
    """Score a subset of haps by computing UNCAPPED NLL on sample data.
    Haps are FIXED (not refined) during scoring — used for outer BIC
    subset-selection where we want to compare different SUBSETS of a
    fixed candidate pool.

    For empty subset (K=0): NLL = sum of cost_WW per sample.

    When _VITERBI_BIC_ENABLED is True (production default), NLL is the
    Viterbi best-path NLL over founder-pair states with switching
    allowed at VITERBI_SWITCH_PENALTY cost per switch.  When False, the
    baseline best-pair-per-sample NLL is returned (preserved for A/B
    revert; see the VITERBI BIC SCORING constants block in bhd_config for rationale).
    """
    # K=0 path is identical under both scoring schemes: only state is
    # (W, W), no transitions to consider.  Handled inline so this branch
    # bypasses both _update_A and _viterbi_ll_per_sample.
    if len(haps_list) == 0:
        cost_WW_per_site = _per_site_cost_W_W(probs_k, lam)
        return float(cost_WW_per_site.sum())

    # Viterbi scoring (production default): call _viterbi_nll directly,
    # bypassing _update_A entirely.  This is correct AND faster than
    # going through _update_A, which would compute the baseline best-
    # pair-per-sample cost and then overwrite it with Viterbi NLL —
    # wasted work for a scoring-only call site that doesn't need A or
    # wildcard_slots.
    if _VITERBI_BIC_ENABLED:
        return _viterbi_nll(
            haps_list, probs_k,
            penalty=VITERBI_SWITCH_PENALTY,
            snps_per_bin=VITERBI_SNPS_PER_BIN,
            lam=lam)

    # Baseline best-pair-per-sample scoring (legacy path, retained for
    # A/B comparison when _VITERBI_BIC_ENABLED = False).  Stacks the
    # haps, runs _update_A which enumerates all candidate pairs, and
    # returns the sum of per-sample uncapped costs under the best pair.
    H = np.stack(haps_list, axis=0)
    A, per_sample_cost, per_sample_cost_unc, wildcard_slots = _update_A(probs_k, H, lam)
    return float(per_sample_cost_unc.sum())