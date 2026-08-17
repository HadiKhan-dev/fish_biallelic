"""Neutral result containers and output materialization for block discovery.

This leaf module is shared by the discrete and reversible discovery paths.  It
must not import either orchestrator, so those paths remain independently
importable and the block-discovery dependency graph stays acyclic.
"""

import math

import numpy as np
from numba import njit, prange


class BlockResult:
    """
    Container for the reconstructed haplotypes of a single genomic block.
    """
    def __init__(self, positions, haplotypes, reads_count_matrix=None,
                 keep_flags=None, probs_array=None,
                 genotype_evidence_mode=None):
        self.positions = positions
        self.haplotypes = haplotypes # Dictionary {id: numpy_array}
        self.reads_count_matrix = reads_count_matrix # Optional: source reads (Samples x Sites x 2)
        self.keep_flags = keep_flags
        self.probs_array = probs_array # New Optional: genotype probabilities (Samples x Sites x 3)
        # Provenance for probs_array; None preserves compatibility with
        # historical BlockResult objects whose evidence mode was not recorded.
        self.genotype_evidence_mode = genotype_evidence_mode

    def __len__(self):
        return len(self.haplotypes)

    def __repr__(self):
        active_sites = np.sum(self.keep_flags) if self.keep_flags is not None else len(self.positions)
        has_probs = "with probs" if self.probs_array is not None else "no probs"
        return f"<BlockResult: {len(self.haplotypes)} haplotypes at {len(self.positions)} sites ({active_sites} active), {has_probs}>"


class BlockResults:
    """
    A container class holding a list of BlockResult objects, representing
    the reconstruction results for an entire genomic region.
    """
    def __init__(self, block_result_list):
        self.blocks = block_result_list

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        """Allows accessing blocks by index: block_results[i]"""
        return self.blocks[index]

    def __iter__(self):
        """Allows iterating: for block in block_results: ..."""
        return iter(self.blocks)

    def __repr__(self):
        return f"<BlockResults: containing {len(self.blocks)} processed blocks>"


def _compute_per_site_confidence(probs_k, H_k, A, lam, min_supporters=2):
    """For each (founder, kept site), compute confidence as the fraction
    of attributing samples whose data is consistent with the founder's
    inferred allele under their pair assignment.

    "Consistent" = the per-(sample, site) cost under the real-pair beats
    the wildcard cost — i.e., the founder's allele genuinely fits this
    sample at this site rather than the data being indifferent.

    For sites with fewer than min_supporters attributing samples, the
    confidence is 0 (and the site will be MASKed at output).

    Arguments:
        probs_k: (N, L_kept, 3)
        H_k:     (K, L_kept)
        A:       (N, 2) with K used as wildcard sentinel
        lam:     wildcard penalty
        min_supporters: minimum supporting samples to compute confidence

    Returns:
        confidence: (K, L_kept) float in [0, 1]
        n_supporters: (K, L_kept) int
    """
    K, L = H_k.shape
    if K == 0:
        return (np.zeros((0, L), dtype=np.float64),
                np.zeros((0, L), dtype=np.int64))
    # Hand off to the njit kernel.  The kernel replaces the original
    # Python `for k in range(K): for l in range(L):` double loop with
    # per-site numpy slicing.  Same pattern as bhd_kernels'
    # _update_one_founder rewrite (which gave 22x on the same shape of
    # work); expect comparable speedup here.
    probs_c = np.ascontiguousarray(probs_k, dtype=np.float64)
    H_c = np.ascontiguousarray(H_k, dtype=np.int64)
    A_c = np.ascontiguousarray(A, dtype=np.int64)
    return _compute_per_site_confidence_kernel(
        probs_c, H_c, A_c, float(lam), int(min_supporters))


@njit(cache=True, parallel=True, fastmath=False)
def _compute_per_site_confidence_kernel(probs_k, H_k, A, lam, min_supporters):
    """njit version of _compute_per_site_confidence.

    For each (k, l), determine which samples are "supporting" k at l
    (via their pair-assignment bucket) and how many of those samples'
    data is "consistent" with the founder's allele.  Three buckets:

      Bucket H (k, k): consistent iff argmax of P(g | s, l) == 2*H_k[k, l]
      Bucket J (k, j) with j != k, j real: consistent iff argmax of
                                            P(g | s, l) == H_k[k, l] + H_k[j, l]
      Bucket P (k, W): consistent iff per-site real-pair cost <
                       per-site wildcard cost (so the real founder
                       actually contributed information, not just letting
                       the wildcard absorb)

    Same arithmetic as the Python version.  `prange` over founders
    because samples-in-pair-assignment are unevenly distributed across
    founders (some founders dominate carriers; parallelising over k
    balances out via the global sample-mask scan inside each k).

    Floors -log(p) at LOG_EPS_LOCAL = 1e-12 to match _safe_neg_log's
    behaviour.

    Returns:
        confidence:   (K, L) float64
        n_supporters: (K, L) int64
    """
    LOG_EPS_LOCAL = 1e-12

    N = probs_k.shape[0]
    K = H_k.shape[0]
    L = H_k.shape[1]
    W = K

    confidence = np.zeros((K, L), dtype=np.float64)
    n_supporters = np.zeros((K, L), dtype=np.int64)

    for k in prange(K):
        # Walk samples once to classify each into bucket H, J, P, or
        # not-supporting.  We don't pre-materialise the bucket masks
        # (the Python version did) because numba prefers explicit loops
        # over fancy mask indexing in the inner kernel — and at typical
        # N=320 a single sample-walk per (k, l) inner site is cheap.
        for l in range(L):
            cur_val = H_k[k, l]
            n_supp = 0
            n_consistent = 0

            for s in range(N):
                a0 = A[s, 0]
                a1 = A[s, 1]
                # Check whether sample s supports founder k under any
                # bucket.  Bucket-H test first since it's the cheapest.
                if a0 == k and a1 == k:
                    # Bucket H (k, k): consistent iff argmax P(g) == 2*cur_val
                    p0 = probs_k[s, l, 0]
                    p1 = probs_k[s, l, 1]
                    p2 = probs_k[s, l, 2]
                    # argmax of (p0, p1, p2).  Tie-break: first-max
                    # matches np.argmax's behaviour (which the Python
                    # version used).
                    if p0 >= p1 and p0 >= p2:
                        amax = 0
                    elif p1 >= p2:
                        amax = 1
                    else:
                        amax = 2
                    n_supp += 1
                    if amax == 2 * cur_val:
                        n_consistent += 1
                elif a0 == k and a1 == W:
                    # Bucket P (k, W): consistent iff real-(k,W) cost <
                    # (W, W) cost at site l.
                    p0 = probs_k[s, l, 0]
                    p1 = probs_k[s, l, 1]
                    p2 = probs_k[s, l, 2]
                    # best_real = max(p[cur_val], p[cur_val+1])
                    if cur_val == 0:
                        best_real = p0 if p0 > p1 else p1
                    else:
                        best_real = p1 if p1 > p2 else p2
                    pmax = p0
                    if p1 > pmax:
                        pmax = p1
                    if p2 > pmax:
                        pmax = p2
                    if best_real < LOG_EPS_LOCAL:
                        best_real = LOG_EPS_LOCAL
                    if pmax < LOG_EPS_LOCAL:
                        pmax = LOG_EPS_LOCAL
                    cost_real = -math.log(best_real) + lam
                    cost_WW = -math.log(pmax) + 2.0 * lam
                    n_supp += 1
                    if cost_real < cost_WW:
                        n_consistent += 1
                elif (a0 == k or a1 == k) and a0 != a1 and a0 != W and a1 != W:
                    # Bucket J (k, j) with j != k, j real.  Find
                    # partner founder index j.
                    j = a1 if a0 == k else a0
                    partner_h = H_k[j, l]
                    expected_dosage = cur_val + partner_h
                    p0 = probs_k[s, l, 0]
                    p1 = probs_k[s, l, 1]
                    p2 = probs_k[s, l, 2]
                    if p0 >= p1 and p0 >= p2:
                        amax = 0
                    elif p1 >= p2:
                        amax = 1
                    else:
                        amax = 2
                    n_supp += 1
                    if amax == expected_dosage:
                        n_consistent += 1
                # else: sample s does not support founder k at this
                # site under any bucket; skip.

            n_supporters[k, l] = n_supp
            if n_supp >= min_supporters:
                confidence[k, l] = n_consistent / n_supp
            # else: confidence stays 0 (low-support site).

    return confidence, n_supporters


def _discrete_haps_to_prob_arrays(H_k_full, n_sites_full, kept_mask, confidence_full,
                                    n_supporters_full, min_supporters):
    """Convert the (K, L_full) discrete H to a dict of (n_sites_full, 2)
    [P(allele=0), P(allele=1)] arrays.

    Sites that fall below min_supporters or are not in kept_mask are
    represented as (0.5, 0.5) — the legacy format's encoding for "no
    information."  Confident sites are crisp (1.0, 0.0) or (0.0, 1.0).

    Arguments:
        H_k_full: (K, L_full) — discrete haps padded to full block length
                  (sites outside kept_mask are 0 by default)
        n_sites_full: int
        kept_mask: (L_full,) bool — only kept sites are scored
        confidence_full: (K, L_full) float
        n_supporters_full: (K, L_full) int
        min_supporters: int — sites with fewer supporters become (0.5, 0.5)

    Returns:
        haps_dict: {k: (n_sites_full, 2)} float arrays
    """
    K = H_k_full.shape[0]
    haps_dict = {}
    for k in range(K):
        h_arr = np.full((n_sites_full, 2), 0.5, dtype=np.float64)
        # For each site, if it's kept AND has enough supporters, set crisp
        for l in range(n_sites_full):
            if kept_mask is not None and not kept_mask[l]:
                continue
            if n_supporters_full[k, l] < min_supporters:
                continue
            if H_k_full[k, l] == 0:
                h_arr[l, 0] = 1.0
                h_arr[l, 1] = 0.0
            else:
                h_arr[l, 0] = 0.0
                h_arr[l, 1] = 1.0
        haps_dict[k] = h_arr
    return haps_dict


__all__ = [
    "BlockResult",
    "BlockResults",
]
