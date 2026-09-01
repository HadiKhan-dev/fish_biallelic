"""Neutral result containers and output materialization for block discovery.

This leaf module defines the public block containers and the canonical
founder-site pseudo-evidence materializer. It does not import the discovery
orchestrator, keeping the block-discovery dependency graph acyclic.
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
        # Optional provenance describing the genotype-evidence representation.
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


def _materialize_founder_site_pseudo_evidence(
        probs_k, H_k, A, observed_mask, lam,
        min_directional_supporters, min_hard_call_pseudo_probability):
    """Materialize fixed-A founder/site evidence while preserving missingness.

    Costs exactly follow the fitted H update's H/J/P buckets and per-cell WW
    caps. Only observed carriers with different capped costs for H=0 and H=1
    count as informative unique supporters. The returned conditional capped
    log pseudo-odds are NLL(H=0) - NLL(H=1), so positive values favour
    allele 1.

    Returns (q, supporters, log_pseudo_odds, hard_mask, hard_values),
    each with shape (K, L). Here q is the capped fixed-assignment
    pseudo-probability for H=1; unknown hard values are -1.
    """
    return _materialize_founder_site_pseudo_evidence_kernel(
        np.ascontiguousarray(probs_k, dtype=np.float64),
        np.ascontiguousarray(H_k, dtype=np.int64),
        np.ascontiguousarray(A, dtype=np.int64),
        np.ascontiguousarray(observed_mask, dtype=np.bool_),
        float(lam), int(min_directional_supporters),
        float(min_hard_call_pseudo_probability))


@njit(cache=True, parallel=True, fastmath=False)
def _materialize_founder_site_pseudo_evidence_kernel(
        probs_k, H_k, A, observed_mask, lam,
        min_directional_supporters, min_hard_call_pseudo_probability):
    """Compiled fixed-assignment counterpart of the founder-site materializer."""
    LOG_EPS_LOCAL = 1e-12
    N = probs_k.shape[0]
    K, L = H_k.shape
    W = K
    q = np.full((K, L), 0.5, dtype=np.float64)
    supporters = np.zeros((K, L), dtype=np.int64)
    log_pseudo_odds = np.zeros((K, L), dtype=np.float64)
    hard_mask = np.zeros((K, L), dtype=np.bool_)
    hard_values = np.full((K, L), -1, dtype=np.int8)

    for k in prange(K):
        for l in range(L):
            support = 0
            evidence = 0.0
            for s in range(N):
                if not observed_mask[s, l]:
                    continue
                a0 = A[s, 0]
                a1 = A[s, 1]
                bucket = 0
                partner_h = 0
                if a0 == k and a1 == k:
                    bucket = 1
                elif a0 == k and a1 == W:
                    bucket = 3
                elif ((a0 == k or a1 == k) and a0 != a1
                      and a0 != W and a1 != W):
                    bucket = 2
                    j = a1 if a0 == k else a0
                    partner_h = H_k[j, l]
                else:
                    continue

                p0 = probs_k[s, l, 0]
                p1 = probs_k[s, l, 1]
                p2 = probs_k[s, l, 2]
                pmax = max(p0, p1, p2, LOG_EPS_LOCAL)
                cost_WW = -math.log(pmax) + 2.0 * lam
                if bucket == 1:
                    cost_h0 = -math.log(max(p0, LOG_EPS_LOCAL))
                    cost_h1 = -math.log(max(p2, LOG_EPS_LOCAL))
                elif bucket == 2:
                    cost_h0 = -math.log(max(
                        probs_k[s, l, partner_h], LOG_EPS_LOCAL))
                    cost_h1 = -math.log(max(
                        probs_k[s, l, partner_h + 1], LOG_EPS_LOCAL))
                else:
                    cost_h0 = -math.log(max(
                        p0, p1, LOG_EPS_LOCAL)) + lam
                    cost_h1 = -math.log(max(
                        p1, p2, LOG_EPS_LOCAL)) + lam

                cost_h0 = min(cost_h0, cost_WW)
                cost_h1 = min(cost_h1, cost_WW)
                contribution = cost_h0 - cost_h1
                if contribution != 0.0:
                    evidence += contribution
                    support += 1

            log_pseudo_odds[k, l] = evidence
            supporters[k, l] = support
            if evidence >= 0.0:
                probability = 1.0 / (1.0 + math.exp(-evidence))
            else:
                exp_evidence = math.exp(evidence)
                probability = exp_evidence / (1.0 + exp_evidence)
            q[k, l] = probability
            if support >= min_directional_supporters:
                if probability >= min_hard_call_pseudo_probability:
                    hard_mask[k, l] = True
                    hard_values[k, l] = 1
                elif 1.0 - probability >= min_hard_call_pseudo_probability:
                    hard_mask[k, l] = True
                    hard_values[k, l] = 0

    return q, supporters, log_pseudo_odds, hard_mask, hard_values


__all__ = [
    "BlockResult",
    "BlockResults",
]
