# bhd_recovery_candidates.py - residual-candidate generation for subtraction recovery.
#
# Produces candidate haplotypes for downstream BIC subset-selection: iterative
# subtraction rounds (hard + soft) that peel each current founder off every
# sample's argmax dosage, and per-carrier / per-sample residual generation.
#
# Leaf module of the bhd_recovery 4-file split: imports only numpy and the
# residual cleanliness thresholds from bhd_config.  See bhd_recovery.py for
# subsystem context.

import numpy as np

import warnings
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Numba not found; bhd_recovery_candidates kernels fall back to pure Python "
        "(slower but numerically identical).",
        ImportWarning,
    )
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator
    prange = range

from bhd_config import (
    RECOVERY_CLEANNESS_THRESHOLD,
    RESIDUAL_TRIO_CLEANNESS_THRESHOLD,
)


# =============================================================================
# SUBTRACTION-BASED RECOVERY: CANDIDATE GENERATION & OUTER BIC SELECTION
# =============================================================================

@njit(cache=True, parallel=True, fastmath=False)
def _run_subtraction_round_kernel(pool_arr, argmax_dosage_kept,
                                    cleanness_threshold):
    """Numba kernel for _run_subtraction_round.

    Replaces the per-pool-member numpy temporaries (residual, in_01,
    cleanness; each O(N*L) per iteration) with a scalar loop over
    (hap, sample) pairs.  For each pair:
      (1) count admissible sites by computing the residual on the fly,
          break-decision via `cleanness < threshold`;
      (2) if accepted, write the clipped residual into the output
          buffer in pool-outer / sample-inner order.
    This recomputes each residual twice (once for counting, once for
    writing) but avoids any O(N*L) heap allocation per iteration; on
    production data the second pass is bound by the ~50% acceptance
    rate so the recomputation cost is dwarfed by the allocation
    savings.

    Output buffer is preallocated worst-case (pool_size * N rows).  The
    wrapper truncates to the active count.

    Args:
        pool_arr: (P, L) int64 — stacked pool haps (each row is one
            (L,) hap from the original Python list)
        argmax_dosage_kept: (N, L) int64 — argmax dosages in {0, 1, 2}
        cleanness_threshold: float — min fraction of admissible sites

    Returns:
        out_buf: (out_count, L) int64 — tight slice of admissible
            clipped residuals, in (hap, sample) traversal order
        out_count: int — number of valid rows in out_buf
    """
    P, L = pool_arr.shape
    N = argmax_dosage_kept.shape[0]

    # Parallelised over pool members (prange).  A running append counter
    # would race across threads, so we use a deterministic two-pass scheme:
    #   Pass 1 (parallel over p): decide cleanness per (pool member, sample)
    #     and count the accepted samples for each pool member.
    #   cumsum the counts into per-pool output offsets (serial, O(P), cheap).
    #   Pass 2 (parallel over p): each pool member writes its accepted clipped
    #     residuals into its OWN disjoint output block, in sample order.
    # The block offsets reproduce pool-outer order and the inner sample loop
    # reproduces sample-inner order, so the output is byte-identical to the
    # serial single-counter version (same values, same sequence).
    #
    # Cleanness decision matches the original `cleanness = in_01.mean(axis=1);
    # clean_mask = cleanness >= cleanness_threshold` exactly: we compute
    # n_in_01 as an int, divide by L, and compare to the threshold in float64
    # (so the same float-mean rounding governs the accept/skip decision, with
    # no integer-count off-by-one drift at non-exact threshold*L).
    keep = np.zeros((P, N), dtype=np.bool_)
    counts = np.zeros(P, dtype=np.int64)
    for p in prange(P):
        cnt = 0
        for s in range(N):
            # Count admissible sites for this (pool member, sample) pair
            n_in_01 = 0
            for l in range(L):
                r = argmax_dosage_kept[s, l] - pool_arr[p, l]
                if 0 <= r <= 1:
                    n_in_01 += 1
            cleanness = n_in_01 / L
            if cleanness >= cleanness_threshold:
                keep[p, s] = True
                cnt += 1
        counts[p] = cnt

    offsets = np.zeros(P, dtype=np.int64)
    total = 0
    for p in range(P):
        offsets[p] = total
        total += counts[p]

    out_buf = np.empty((total, L), dtype=np.int64)
    for p in prange(P):
        pos = offsets[p]
        for s in range(N):
            if not keep[p, s]:
                continue
            # Accepted — write clipped residual into this pool member's block
            for l in range(L):
                r = argmax_dosage_kept[s, l] - pool_arr[p, l]
                # np.clip(r, 0, 1)
                if r < 0:
                    out_buf[pos, l] = 0
                elif r > 1:
                    out_buf[pos, l] = 1
                else:
                    out_buf[pos, l] = r
            pos += 1
    return out_buf, total


def _run_subtraction_round(pool, argmax_dosage_kept,
                            cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD):
    """Generate clean residual candidates by subtracting each pool member
    from each sample's argmax dosage.

    For sample s with argmax dosage d_s and pool member h, the residual
    r_s = d_s - h is interpreted as an estimate of the OTHER strand
    given that h was one strand.  At sites where d_s in {h, h+1}, r_s in
    {0, 1} — admissible.  At sites where d_s != h and d_s != h+1, r_s is
    out of range — the (h, ?) hypothesis is inconsistent at that site.

    A residual is "clean" if at least cleanness_threshold of its sites
    have admissible values.  Clean residuals are clipped to {0, 1} and
    returned as binary candidate haps.

    Args:
      pool: list of (L_kept,) binary arrays (the founders to subtract)
      argmax_dosage_kept: (N, L_kept) int array of argmax genotype dosages
        in {0, 1, 2} per (sample, kept site)
      cleanness_threshold: min fraction of admissible sites to accept

    Returns:
      list of (L_kept,) binary candidate arrays

    Implementation: delegates to a numba kernel that uses a scalar
    two-pass loop over (pool, sample) pairs and writes directly into a
    preallocated worst-case output buffer.  Output ordering preserved
    (pool-outer, sample-inner) so downstream callers see byte-identical
    candidate sequences.
    """
    # Edge case: empty pool returns an empty list (kernel would handle
    # P=0 correctly but we short-circuit to skip the buffer allocation).
    if len(pool) == 0:
        return []

    # Stack pool haps into a contiguous (P, L) int64 array.  Each
    # element of pool is a (L,) int array (from the round loop, dtype
    # is np.int64; defensive cast ensures kernel signature match).
    pool_arr = np.empty((len(pool), pool[0].shape[0]), dtype=np.int64)
    for p in range(len(pool)):
        pool_arr[p] = pool[p]

    argmax_arr = np.ascontiguousarray(argmax_dosage_kept, dtype=np.int64)
    out_buf, out_count = _run_subtraction_round_kernel(
        pool_arr, argmax_arr, float(cleanness_threshold))

    # Repackage into list-of-arrays (matching the legacy return shape).
    # Copy each row to detach from the kernel's slice — defensive in
    # case downstream callers mutate.
    return [out_buf[i].copy() for i in range(out_count)]


@njit(cache=True, parallel=True, fastmath=False)
def _run_subtraction_round_soft_kernel(pool_arr, P0, P1, P2,
                                         Lk0, Lk1, Lk2, cleanness_threshold):
    """Numba kernel for _run_subtraction_round_soft, parallel over pool.

    Replaces the per-pool-member numpy temporaries (adm, keep, l0, l1; each
    O(N*L) per pool member) with a prange over pool members.  A running
    append counter would race across threads, so the writes use a
    deterministic two-pass scheme (count -> cumsum offsets -> write to
    disjoint per-pool blocks), the same pattern as the hard-call kernel:
      Pass 1 (parallel over p): cleanness screen per (pool member, sample) —
        admissible-mass mean over sites >= threshold — and count accepted
        samples per pool member.
      cumsum the counts into per-pool output offsets (serial, O(P), cheap).
      Pass 2 (parallel over p): each pool member writes its accepted
        (L0, L1) rows into its own disjoint block, in sample order.

    Numerical behaviour vs the numpy version:
      - The (L0, L1) VALUES are element-wise selects of the same Lk0/Lk1/Lk2
        the caller computed (h[l]=0 -> (Lk0, Lk1); h[l]=1 -> (Lk1, Lk2)), so
        they are bit-identical to `np.where(...)` and the vstack output.
      - The cleanness screen sums admissible mass per row sequentially rather
        than via numpy's pairwise `adm.mean(axis=1)`, a sanctioned ULP-level
        difference in summation order.  It can only change which samples are
        admitted for a sample whose mass sits within a few ULP of the
        threshold, which essentially never happens; the admitted-row VALUES
        are unaffected either way.

    Output ordering (pool-outer, sample-inner) matches the old
    `for h in pool` + per-h `l0[keep]` + vstack exactly.

    Args:
        pool_arr: (P, L) int64 — stacked founder bits (0/1); used as a
            truthiness per site, matching `h.astype(bool)`.
        P0, P1, P2: (N, L) float64 — genotype POSTERIORS (for the screen).
        Lk0, Lk1, Lk2: (N, L) float64 — genotype LIKELIHOODS (for the
            output; equal to P0/P1/P2 divided by the site prior, or the
            posteriors themselves when no prior is supplied).
        cleanness_threshold: float — min mean admissible mass to admit.

    Returns:
        L0_out, L1_out: (M, L) float64 — admitted other-strand likelihoods
            for the M admitted (pool member, sample) pairs, contiguous, in
            (pool-outer, sample-inner) order.  Both have M==0 rows when the
            pool admits nothing.
    """
    P, L = pool_arr.shape
    N = P0.shape[0]
    # Pass 1: cleanness screen per (pool member, sample); count per pool.
    keep = np.zeros((P, N), dtype=np.bool_)
    counts = np.zeros(P, dtype=np.int64)
    for p in prange(P):
        cnt = 0
        for s in range(N):
            adm_sum = 0.0
            for l in range(L):
                if pool_arr[p, l]:
                    adm_sum += P1[s, l] + P2[s, l]
                else:
                    adm_sum += P0[s, l] + P1[s, l]
            if adm_sum / L >= cleanness_threshold:
                keep[p, s] = True
                cnt += 1
        counts[p] = cnt

    offsets = np.zeros(P, dtype=np.int64)
    total = 0
    for p in range(P):
        offsets[p] = total
        total += counts[p]

    L0_out = np.empty((total, L), dtype=np.float64)
    L1_out = np.empty((total, L), dtype=np.float64)
    # Pass 2: write the admitted (L0, L1) rows into each pool member's block.
    for p in prange(P):
        pos = offsets[p]
        for s in range(N):
            if not keep[p, s]:
                continue
            for l in range(L):
                if pool_arr[p, l]:
                    L0_out[pos, l] = Lk1[s, l]
                    L1_out[pos, l] = Lk2[s, l]
                else:
                    L0_out[pos, l] = Lk0[s, l]
                    L1_out[pos, l] = Lk1[s, l]
            pos += 1

    return L0_out, L1_out


def _run_subtraction_round_soft(pool, probs_k, site_priors=None,
                                  cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD):
    """Soft analogue of _run_subtraction_round (RECOVERY_RESIDUAL_MODE ==
    "soft"): generate other-strand LIKELIHOODS for the marginal-likelihood
    mixture, rather than hard-called residual bits.

    For founder strand h and a sample with genotype likelihoods L(g) over
    dosages g in {0, 1, 2}, the other strand o satisfies g = h + o, so the
    per-site other-strand likelihoods are:
      h[l]=0: L0 = L(g=0)  (o=0),  L1 = L(g=1)  (o=1)
      h[l]=1: L0 = L(g=1)  (o=0),  L1 = L(g=2)  (o=1)
    (the third, "inadmissible", genotype — g=2 when h=0, g=0 when h=1 — is
    dropped; a sample whose reads favour it gets low L0 and L1, hence low
    likelihood under every mixture component.)  These (L0, L1) feed
    _fit_bernoulli_mixture_ml_select_K, which marginalises the latent
    other-strand allele instead of plugging in its posterior mean — so
    uninformative sites defer to the cluster consensus rather than diluting
    it (see that function and RECOVERY_RESIDUAL_MODE).

    probs_k holds the genotype POSTERIOR.  When site_priors is given the HWE
    site prior is divided out to recover the genotype LIKELIHOOD (the correct
    quantity for the marginalisation): L(g) prop probs_k[g] / site_priors[g].
    The per-(sample, site) normaliser is irrelevant — it cancels in both EM
    steps, so only the per-genotype prior ratio matters — hence no
    renormalisation is needed.  When site_priors is None probs_k is used
    directly, which is exact when probs_k already is the likelihood (flat
    prior, e.g. synthetic data).

    A candidate (founder h, sample s) is admitted when its mean admissible
    POSTERIOR mass (P0+P1 for h=0, P1+P2 for h=1) over sites meets
    cleanness_threshold — the same screen as the argmax path.

    Args:
      pool: list of (L_kept,) binary arrays (the founders to subtract)
      probs_k: (N, L_kept, 3) genotype posteriors (kept sites)
      site_priors: (L_kept, 3) genotype priors to divide out, or None
      cleanness_threshold: min mean admissible posterior mass to accept

    Returns:
      (L0, L1): two (M, L_kept) float64 arrays of other-strand likelihoods
        for the M admitted candidates, in (pool-outer, sample-inner) order.
        Both are empty (M=0) when the pool is empty or nothing is clean.
    """
    L = probs_k.shape[1]
    empty = np.empty((0, L), dtype=np.float64)
    if len(pool) == 0:
        return empty, empty

    P0 = probs_k[:, :, 0]
    P1 = probs_k[:, :, 1]
    P2 = probs_k[:, :, 2]                                          # (N, L) views

    # Genotype likelihoods for the model: divide out the site prior if given.
    if site_priors is not None:
        sp = np.asarray(site_priors, dtype=np.float64)            # (L, 3)
        Lk0 = P0 / sp[None, :, 0]
        Lk1 = P1 / sp[None, :, 1]
        Lk2 = P2 / sp[None, :, 2]
    else:
        Lk0, Lk1, Lk2 = P0, P1, P2

    L0_list = []
    L1_list = []
    # Parallelise the per-pool work: stack the pool and make the per-genotype
    # arrays contiguous, then hand off to the kernel, which prange's over pool
    # and returns the admitted (L0, L1) rows in (pool-outer, sample-inner)
    # order -- exactly the order the old `for h in pool` + vstack produced.
    # The likelihood VALUES are element-wise selects of the same Lk0/Lk1/Lk2
    # numpy computed above, so they are bit-identical; only the cleanness
    # screen's per-row mean is summed in a different order (sequential vs
    # numpy's pairwise `adm.mean`), a sanctioned ULP-level difference that
    # could only matter for a sample whose admissible mass sits within a few
    # ULP of the threshold.  (L0_list / L1_list are retained as the empty
    # sentinels the no-candidate paths below still reference.)
    pool_arr = np.empty((len(pool), L), dtype=np.int64)
    for p in range(len(pool)):
        pool_arr[p] = np.asarray(pool[p])
    L0_out, L1_out = _run_subtraction_round_soft_kernel(
        pool_arr,
        np.ascontiguousarray(P0, dtype=np.float64),
        np.ascontiguousarray(P1, dtype=np.float64),
        np.ascontiguousarray(P2, dtype=np.float64),
        np.ascontiguousarray(Lk0, dtype=np.float64),
        np.ascontiguousarray(Lk1, dtype=np.float64),
        np.ascontiguousarray(Lk2, dtype=np.float64),
        float(cleanness_threshold))

    if L0_out.shape[0] == 0:
        return empty, empty
    return L0_out, L1_out


@njit(cache=True, parallel=True, fastmath=False)
def _generate_carrier_residuals_kernel(argmax_dosage, H, A,
                                          low_idx_mask, cleanness_threshold):
    """Numba kernel for _generate_carrier_residuals (verbose=False only).

    Replaces the triple Python loop + numpy temporaries with a tight
    scalar loop over (s, slot, partner_idx) triples that:
      1. checks `A[s, slot] in low_idx_set` via a bool mask of length K
         (numba doesn't support Python sets);
      2. skips partner_idx == low_idx or partner_idx in low_idx_set;
      3. computes the residual on the fly for accepted partner pairs;
      4. writes accepted clipped residuals into a preallocated buffer.

    Iteration order: s outer, slot middle, partner_idx inner —
    matching the original.  Output buffer is preallocated worst-case
    (N * 2 * K rows).  The wrapper truncates to the active count.

    Args:
        argmax_dosage: (N, L) int64 — precomputed argmax dosages
        H: (K, L) int64 — current founder bits
        A: (N, 2) int64 — current pair assignments (entry K is the
            wildcard sentinel)
        low_idx_mask: (K,) bool — True at indices belonging to low_idx_set
        cleanness_threshold: float — min admissible-site fraction

    Returns:
        out_buf: (out_count, L) int64 — accepted clipped residuals
        out_count: int — number of valid rows in out_buf
    """
    N, L = argmax_dosage.shape
    K = H.shape[0]
    # Parallelised over samples (prange), two-pass so the append positions
    # are deterministic (a shared counter would race).  Pass 1 records the
    # accept decision per (sample, slot, partner_idx) and counts accepted
    # triples per sample; cumsum gives per-sample output offsets; pass 2 has
    # each sample write its accepted clipped residuals into its own disjoint
    # block, in (slot, partner_idx) order.  The offsets reproduce the
    # s-outer order and the inner loops the slot-middle / partner-inner
    # order, so the output is byte-identical to the serial single-counter
    # version.
    keep = np.zeros((N, 2, K), dtype=np.bool_)
    counts = np.zeros(N, dtype=np.int64)
    for s in prange(N):
        cnt = 0
        for slot in range(2):
            a_idx = A[s, slot]
            # A entries: integers in [0, K] where K is the wildcard
            # sentinel.  Index K is out of range for low_idx_mask, so
            # we must guard before lookup.  Wildcard slots are skipped
            # (the original `int(A[s, slot]) not in low_idx_set` returns
            # False for the wildcard sentinel value since low_idx_set
            # contains only real founder indices < K).
            if a_idx < 0 or a_idx >= K:
                continue
            if not low_idx_mask[a_idx]:
                continue
            low_idx = a_idx
            for partner_idx in range(K):
                if partner_idx == low_idx:
                    continue
                if low_idx_mask[partner_idx]:
                    # Subtractor in low_idx_set — skip
                    continue
                # Compute residual on the fly and count admissible sites
                n_in_01 = 0
                for l in range(L):
                    r = argmax_dosage[s, l] - H[partner_idx, l]
                    if 0 <= r <= 1:
                        n_in_01 += 1
                cleanness = n_in_01 / L
                if cleanness >= cleanness_threshold:
                    keep[s, slot, partner_idx] = True
                    cnt += 1
        counts[s] = cnt

    offsets = np.zeros(N, dtype=np.int64)
    total = 0
    for s in range(N):
        offsets[s] = total
        total += counts[s]

    out_buf = np.empty((total, L), dtype=np.int64)
    for s in prange(N):
        pos = offsets[s]
        for slot in range(2):
            for partner_idx in range(K):
                if not keep[s, slot, partner_idx]:
                    continue
                # Accepted — write clipped residual
                for l in range(L):
                    r = argmax_dosage[s, l] - H[partner_idx, l]
                    if r < 0:
                        out_buf[pos, l] = 0
                    elif r > 1:
                        out_buf[pos, l] = 1
                    else:
                        out_buf[pos, l] = r
                pos += 1

    return out_buf, total


@njit(cache=True, parallel=True, fastmath=False)
def _generate_all_sample_residuals_kernel(argmax_dosage, H, A,
                                          cleanness_threshold):
    """Numba kernel for _generate_all_sample_residuals (verbose=False path).

    Variant of _generate_carrier_residuals_kernel that loops over EVERY
    (sample, slot) pair — not just samples whose A[s, slot] is in a
    low_idx_set.  Used by _residual_trio_rescue to mine residuals across
    the whole population, surfacing near-clone founders that K-growth's
    residual-mass seeding missed.

    For each (sample, slot), the OTHER slot's H-row is the partner being
    subtracted: we want residual = strand_at_slot, so we subtract the
    OTHER strand's H entry.  This is the algebra of _late_low_carrier_-
    rescue but applied to all samples regardless of usage statistics.

    Iteration: s outer, slot middle.  For each (s, slot), the partner is
    fixed as A[s, 1 - slot] (the OTHER slot's H row).  Wildcard partners
    (A entry == K, the wildcard sentinel) are skipped because subtracting
    a wildcard means we don't know the other strand and any residual
    derived from it is uninterpretable.

    Output buffer is preallocated worst-case (N * 2 rows) — every sample
    can contribute at most 2 residuals.  The wrapper truncates to the
    active count.

    Args:
        argmax_dosage: (N, L) int64 — precomputed argmax dosages
        H: (K, L) int64 — current founder bits
        A: (N, 2) int64 — current pair assignments (entry K = wildcard
            sentinel)
        cleanness_threshold: float — min admissible-site fraction

    Returns:
        out_buf: (out_count, L) int64 — accepted clipped residuals
        out_count: int — number of valid rows in out_buf
    """
    N, L = argmax_dosage.shape
    K = H.shape[0]
    # Parallelised over samples (prange), two-pass so the append positions
    # are deterministic.  Pass 1 records the accept decision per (sample,
    # slot) and counts accepted slots per sample; cumsum gives per-sample
    # output offsets; pass 2 has each sample write its accepted clipped
    # residuals into its own disjoint block in slot order.  Byte-identical
    # to the serial single-counter version (same values, same s-outer /
    # slot-inner sequence).
    keep = np.zeros((N, 2), dtype=np.bool_)
    counts = np.zeros(N, dtype=np.int64)
    for s in prange(N):
        cnt = 0
        for slot in range(2):
            # The partner is the OTHER slot's H row.  We want to expose
            # the founder at `slot` so we subtract A[s, 1 - slot].
            partner_idx = A[s, 1 - slot]
            # Wildcard partners (sentinel index K) are not valid
            # subtractors — we don't have an H row for the wildcard,
            # and a residual derived against it is meaningless.
            if partner_idx < 0 or partner_idx >= K:
                continue
            # Compute residual on the fly and count admissible sites
            n_in_01 = 0
            for l in range(L):
                r = argmax_dosage[s, l] - H[partner_idx, l]
                if 0 <= r <= 1:
                    n_in_01 += 1
            cleanness = n_in_01 / L
            if cleanness >= cleanness_threshold:
                keep[s, slot] = True
                cnt += 1
        counts[s] = cnt

    offsets = np.zeros(N, dtype=np.int64)
    total = 0
    for s in range(N):
        offsets[s] = total
        total += counts[s]

    out_buf = np.empty((total, L), dtype=np.int64)
    for s in prange(N):
        pos = offsets[s]
        for slot in range(2):
            if not keep[s, slot]:
                continue
            partner_idx = A[s, 1 - slot]
            # Accepted — write clipped residual.  Clipping is defensive:
            # if cleanness_threshold < 1.0 some sites can be out-of-
            # range; we clip to {0, 1} so downstream consumers see a
            # well-formed binary vector.  At cleanness_threshold == 1.0
            # (the default) no clipping is needed in principle, but we
            # keep the clip for symmetry with the low-carrier kernel.
            for l in range(L):
                r = argmax_dosage[s, l] - H[partner_idx, l]
                if r < 0:
                    out_buf[pos, l] = 0
                elif r > 1:
                    out_buf[pos, l] = 1
                else:
                    out_buf[pos, l] = r
            pos += 1

    return out_buf, total


def _generate_carrier_residuals(probs_k, H, A, low_idx_list,
                                 cleanness_threshold=RECOVERY_CLEANNESS_THRESHOLD,
                                 verbose=False):
    """Generate per-(carrier_sample, partner_candidate) residuals for
    low-carrier haps.

    This is the targeted analogue of _run_subtraction_round, used by
    the late low-carrier rescue (see RECOVERY_LOW_CARRIER_TRIGGER_FRAC).

    For each carrier strand of a low-carrier hap h_low, we want to
    recover h_low's "right" version (the missing truth founder).  The
    algebra: if carrier sample s has true strands (truth_X, truth_Y)
    where truth_Y is the missing one we want to recover, then
      argmax_dosage[s] = truth_X + truth_Y       (noiseless data)
      residual = argmax_dosage[s] - H[partner]
              = (truth_X + truth_Y) - H[partner]
    The residual is "clean" (every site in {0, 1}) if and only if
    H[partner] = truth_X (the actual other strand), in which case
    residual = truth_Y exactly.

    The challenge: when h_low is a chimera, _update_A's choice of A[s,
    other_slot] is whichever founder makes (h_low, partner) optimally
    fit the dosage given h_low is in the pair — NOT necessarily the
    actual other strand truth_X.  At chr6:23624234, A pairs h_low
    (= chim_5) with H[1] = truth_5 for some carriers, but the actual
    other strands are different truth founders; the residual ends up
    = chim_5 by construction.  Verified empirically: all 4 carrier
    residuals at H_low = 0.00%.

    Solution: for each carrier strand, try every H row (excluding
    low_idx_set) as the candidate subtractor.  Most produce noisy
    out-of-range residuals (cleanness < threshold) and are rejected.
    Exactly one row matches truth_X for that carrier and produces a
    clean residual = truth_Y at 100% cleanness.

    Strands paired with the wildcard slot — when the wildcard is the
    "other_slot" — are still mined: we just iterate over all H rows
    as subtractors regardless of A's wildcard assignment.  Subtracting
    a low-carrier hap (a known suspect) from itself or another low-
    carrier hap gives at best noisy residuals, so subtractors in
    low_idx_set are skipped.

    Args:
      probs_k: (N, L_kept, 3) — kept-site posteriors
      H: (K, L_kept) discrete {0, 1} — current founder bits
      A: (N, 2) — current pair assignments (entry K = wildcard sentinel)
      low_idx_list: list of founder indices whose carriers we mine
      cleanness_threshold: min fraction of admissible sites per residual
      verbose: if True, additionally return per-(sample, slot, subtractor)
        provenance

    Returns:
      If verbose=False: list of (L_kept,) binary candidate arrays — one
        per (sample, strand, subtractor) triple that survived cleanness
        filtering.
      If verbose=True: tuple (residuals, provenance) where provenance is
        a list of dicts, one per (sample, slot, subtractor) triple
        examined, with keys:
          sample_idx, slot, low_idx, partner_idx, partner_kind, cleanness,
          accepted, residual (the clipped binary array if accepted else None)
        partner_kind ∈ {'self_low', 'low_carrier', 'normal'}.

    Implementation: the production (verbose=False) path delegates to a
    numba kernel that uses a bool mask (not a Python set) for low_idx
    lookups and writes accepted residuals into a preallocated worst-
    case buffer.  The verbose=True path stays in pure Python because
    numba cannot build Python dicts (and verbose=True is diagnostics-
    only — not called in production).
    """
    N, L_kept = probs_k.shape[0], probs_k.shape[1]
    K = H.shape[0]
    if K == 0 or not low_idx_list:
        return ([], []) if verbose else []
    argmax_dosage = probs_k.argmax(axis=2)                        # (N, L_kept)

    # Build a bool mask over [0, K) for low_idx_set membership.  Any
    # index in low_idx_list that's >= K is ignored (defensive — the
    # production path passes founder indices, which are always < K).
    low_idx_mask = np.zeros(K, dtype=np.bool_)
    for k in low_idx_list:
        ki = int(k)
        if 0 <= ki < K:
            low_idx_mask[ki] = True

    # VERBOSE PATH — stays in pure Python (numba can't return dicts;
    # this path is diagnostics-only and not on any production code path).
    if verbose:
        low_idx_set = set(int(k) for k in low_idx_list)
        residuals = []
        provenance = []
        for s in range(N):
            for slot in range(2):
                if int(A[s, slot]) not in low_idx_set:
                    continue
                low_idx = int(A[s, slot])
                for partner_idx in range(K):
                    if partner_idx == low_idx:
                        provenance.append({
                            'sample_idx': s, 'slot': slot,
                            'low_idx': low_idx, 'partner_idx': partner_idx,
                            'partner_kind': 'self_low',
                            'cleanness': float('nan'), 'accepted': False,
                            'residual': None})
                        continue
                    if partner_idx in low_idx_set:
                        provenance.append({
                            'sample_idx': s, 'slot': slot,
                            'low_idx': low_idx, 'partner_idx': partner_idx,
                            'partner_kind': 'low_carrier',
                            'cleanness': float('nan'), 'accepted': False,
                            'residual': None})
                        continue
                    residual = argmax_dosage[s] - H[partner_idx]
                    in_01 = (residual >= 0) & (residual <= 1)
                    cleanness = float(in_01.mean())
                    accepted = cleanness >= cleanness_threshold
                    clipped = (np.clip(residual, 0, 1).astype(np.int64)
                               if accepted else None)
                    if accepted:
                        residuals.append(clipped)
                    provenance.append({
                        'sample_idx': s, 'slot': slot,
                        'low_idx': low_idx, 'partner_idx': partner_idx,
                        'partner_kind': 'normal',
                        'cleanness': cleanness, 'accepted': accepted,
                        'residual': clipped})
        return residuals, provenance

    # PRODUCTION PATH — delegate to numba kernel.  Ensure dtypes match
    # the kernel signature.  argmax_dosage is already int64 (numpy's
    # default for argmax on integer dtype).  H may be int8 or int64;
    # the kernel signature expects int64 to avoid mixed-type subtraction
    # overhead, so cast defensively.
    argmax_arr = np.ascontiguousarray(argmax_dosage, dtype=np.int64)
    H_arr = np.ascontiguousarray(H, dtype=np.int64)
    A_arr = np.ascontiguousarray(A, dtype=np.int64)
    out_buf, out_count = _generate_carrier_residuals_kernel(
        argmax_arr, H_arr, A_arr, low_idx_mask, float(cleanness_threshold))
    # Repackage into list-of-arrays matching the legacy return shape.
    return [out_buf[i].copy() for i in range(out_count)]


# =============================================================================
# RESIDUAL-TRIO RESCUE (added 2026-05): post-K-growth pass that mines
# per-sample residuals across ALL samples to surface near-clone founders
# K-growth's residual-mass seeding missed.
# =============================================================================
#
# Algorithm:
#   1. For every (sample, slot), compute residual = argmax_dosage[s] -
#      H[A[s, other_slot]].  When the other-slot partner is a clean,
#      truth-near founder, residual = the actual strand at `slot`.  When
#      the partner is itself a chimera, residual will fail the cleanness
#      filter (out-of-range bits at heterozygous sites where the chimera
#      differs from the true partner).
#   2. Filter to clean residuals (every site in {0, 1}).
#   3. Cluster the clean residuals using bhd_trio's
#      _cluster_haps_consensus_kernel (same clustering machinery the
#      hom-recovery candidate emitter uses, for compatibility with the
#      trio pipeline's pool composition).
#   4. For each cluster centroid:
#      - if within RESIDUAL_TRIO_DEDUP_VS_H_PCT of any existing H row →
#        skip (already in dictionary)
#      - else admit as a candidate
#   5. Pool = current H + admitted residual-trio candidates + (optionally)
#      fresh trio/hom/pairwise candidates.  Run greedy BIC forward + swap
#      + prune + refit, same as _late_low_carrier_rescue.
#   6. Accept iff BIC strictly improves.
#
# Differs from _late_low_carrier_rescue:
#   - Mines EVERY sample's residuals (not just low-carrier-hap carriers).
#   - Trigger is per-block "any near-clone-derived candidate exists",
#     evaluated after residual clustering.  No usage-based gate.
#
# Motivating case: chr10:503 F0, where 9 clean F0/F2 samples are
# misfitted to (F4, F2) and produce 9 identical residuals = pure F0 at
# 5 bits from H[F4].  See conversation notes 2026-05-25 for the full
# diagnostic trace.

def _generate_all_sample_residuals(probs_k, H, A,
                                   cleanness_threshold=RESIDUAL_TRIO_CLEANNESS_THRESHOLD):
    """Generate per-(sample, slot) residuals across ALL samples.

    Variant of _generate_carrier_residuals that mines every sample's
    residual against its currently-assigned OTHER slot, not just samples
    whose A[s, slot] is in a low-carrier set.  Used by _residual_trio_-
    rescue to surface near-clone founders K-growth missed.

    For each (sample, slot) with a non-wildcard partner at A[s, 1-slot]:
        residual = argmax_dosage[s] - H[A[s, 1-slot]]
    Residuals where < cleanness_threshold of sites land in {0, 1} are
    rejected (the partner is impure, or the data is noisy).

    Args:
      probs_k: (N, L_kept, 3) — kept-site posteriors
      H: (K, L_kept) — current founder bits (int64)
      A: (N, 2) — current pair assignments (entry K is wildcard sentinel)
      cleanness_threshold: float — min admissible-site fraction

    Returns:
      list of (L_kept,) np.int64 binary arrays — one per (sample, slot)
      pair that survived cleanness filtering.
    """
    N, L_kept = probs_k.shape[0], probs_k.shape[1]
    K = H.shape[0]
    if K == 0:
        return []
    argmax_dosage = probs_k.argmax(axis=2)                        # (N, L_kept)
    argmax_arr = np.ascontiguousarray(argmax_dosage.astype(np.int64))
    H_arr = np.ascontiguousarray(H.astype(np.int64))
    A_arr = np.ascontiguousarray(A.astype(np.int64))

    out_buf, out_count = _generate_all_sample_residuals_kernel(
        argmax_arr, H_arr, A_arr, float(cleanness_threshold))
    # Repackage into list-of-arrays for consistency with the rest of
    # the candidate-source APIs (trio, hom, pairwise all return lists).
    return [out_buf[i].copy() for i in range(out_count)]