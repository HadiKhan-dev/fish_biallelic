"""Hard-painted candidate-screen kernels shared by pedigree engines.

The numerical code is preserved verbatim from the historical pedigree engine
because candidate-screen ranking fixes the downstream Smart M2 panel.
"""

import thread_config  # must precede NumPy/Numba imports

import numpy as np
from numba import prange

njit = thread_config.original_njit


DEFAULT_MISMATCH_PENALTY = -4.605170


@njit(fastmath=True, cache=True)
def run_phase_agnostic_hmm(child_dip_alleles, child_potential_hom_mask, parent_dip_alleles, 
                           switch_costs, stay_costs, error_penalty, phase_penalty,
                           mismatch_penalty=-4.6):
    n_sites = len(child_dip_alleles)
    scores = np.zeros(8)
    BURST_EMISSION = -0.693147 
    for k in range(4, 8):
        scores[k] = -error_penalty
    for i in range(n_sites):
        c0_a, c1_a = child_dip_alleles[i, 0], child_dip_alleles[i, 1]
        p0_a, p1_a = parent_dip_alleles[i, 0], parent_dip_alleles[i, 1]
        def soft_match(child_allele, parent_allele):
            if child_allele == -1 or parent_allele == -1:
                return 0.0
            elif child_allele == parent_allele:
                return 0.0
            else:
                return mismatch_penalty
        e0 = soft_match(c0_a, p0_a)
        e1 = soft_match(c1_a, p0_a)
        e2 = soft_match(c0_a, p1_a)
        e3 = soft_match(c1_a, p1_a)
        emissions = np.array([e0, e1, e2, e3])
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        prev = scores.copy()
        new_scores = np.zeros(8)
        for k in range(4):
            burst_idx = k + 4
            from_burst = prev[burst_idx] 
            from_normal = prev[k] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + BURST_EMISSION
        prev_b0 = prev[4]; prev_b1 = prev[5]; prev_b2 = prev[6]; prev_b3 = prev[7]
        src0 = prev[0] + c_stay
        src1 = prev[1] + c_stay + c_phase
        src2 = prev[2] + c_recomb
        new_scores[0] = max(src0, src1, src2, prev_b0) + emissions[0]
        src1 = prev[1] + c_stay
        src0 = prev[0] + c_stay + c_phase
        src3 = prev[3] + c_recomb
        new_scores[1] = max(src1, src0, src3, prev_b1) + emissions[1]
        src2 = prev[2] + c_stay
        src3 = prev[3] + c_stay + c_phase
        src0 = prev[0] + c_recomb
        new_scores[2] = max(src2, src3, src0, prev_b2) + emissions[2]
        src3 = prev[3] + c_stay
        src2 = prev[2] + c_stay + c_phase
        src1 = prev[1] + c_recomb
        new_scores[3] = max(src3, src2, src1, prev_b3) + emissions[3]
        scores = new_scores
    best_final = -np.inf
    for k in range(8):
        if scores[k] > best_final:
            best_final = scores[k]
    return best_final


@njit(fastmath=True, cache=True)
def run_trio_phase_aware_hmm(child_dip_alleles, child_potential_hom_mask, 
                             p1_dip_alleles, p2_dip_alleles, 
                             switch_costs, stay_costs, error_penalty, phase_penalty,
                             mismatch_penalty=-4.6):
    n_sites = len(child_dip_alleles)
    BURST_EMISSION = -1.386
    scores = np.zeros(16)
    for k in range(8, 16): scores[k] = -error_penalty
    for i in range(n_sites):
        c0, c1 = child_dip_alleles[i, 0], child_dip_alleles[i, 1]
        p1_h0, p1_h1 = p1_dip_alleles[i, 0], p1_dip_alleles[i, 1]
        p2_h0, p2_h1 = p2_dip_alleles[i, 0], p2_dip_alleles[i, 1]
        def soft_match(parent_allele, child_allele):
            if parent_allele == -1 or child_allele == -1:
                return 0.0
            elif parent_allele == child_allele:
                return 0.0
            else:
                return mismatch_penalty
        m_p1h0_c0 = soft_match(p1_h0, c0); m_p1h1_c0 = soft_match(p1_h1, c0)
        m_p1h0_c1 = soft_match(p1_h0, c1); m_p1h1_c1 = soft_match(p1_h1, c1)
        m_p2h0_c0 = soft_match(p2_h0, c0); m_p2h1_c0 = soft_match(p2_h1, c0)
        m_p2h0_c1 = soft_match(p2_h0, c1); m_p2h1_c1 = soft_match(p2_h1, c1)
        e = np.zeros(8)
        e[0] = m_p1h0_c0 + m_p2h0_c1
        e[1] = m_p1h0_c0 + m_p2h1_c1
        e[2] = m_p1h1_c0 + m_p2h0_c1
        e[3] = m_p1h1_c0 + m_p2h1_c1
        e[4] = m_p1h0_c1 + m_p2h0_c0
        e[5] = m_p1h0_c1 + m_p2h1_c0
        e[6] = m_p1h1_c1 + m_p2h0_c0
        e[7] = m_p1h1_c1 + m_p2h1_c0
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        prev = scores.copy()
        new_scores = np.zeros(16)
        for k in range(8):
            burst_idx = k + 8
            from_burst = prev[burst_idx]
            from_normal = prev[k] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + BURST_EMISSION
        cc_0 = 2 * c_stay
        cc_1 = c_recomb + c_stay
        cc_2 = 2 * c_recomb
        p0, p1, p2, p3 = prev[0], prev[1], prev[2], prev[3]
        a0 = max(p0+cc_0, p1+cc_1, p2+cc_1, p3+cc_2)
        a1 = max(p0+cc_1, p1+cc_0, p2+cc_2, p3+cc_1)
        a2 = max(p0+cc_1, p1+cc_2, p2+cc_0, p3+cc_1)
        a3 = max(p0+cc_2, p1+cc_1, p2+cc_1, p3+cc_0)
        p4, p5, p6, p7 = prev[4], prev[5], prev[6], prev[7]
        b4 = max(p4+cc_0, p5+cc_1, p6+cc_1, p7+cc_2)
        b5 = max(p4+cc_1, p5+cc_0, p6+cc_2, p7+cc_1)
        b6 = max(p4+cc_1, p5+cc_2, p6+cc_0, p7+cc_1)
        b7 = max(p4+cc_2, p5+cc_1, p6+cc_1, p7+cc_0)
        pb = prev[8:16]
        new_scores[0] = max(a0 + c_stay, b4 + c_stay + c_phase, pb[0]) + e[0]
        new_scores[1] = max(a1 + c_stay, b5 + c_stay + c_phase, pb[1]) + e[1]
        new_scores[2] = max(a2 + c_stay, b6 + c_stay + c_phase, pb[2]) + e[2]
        new_scores[3] = max(a3 + c_stay, b7 + c_stay + c_phase, pb[3]) + e[3]
        new_scores[4] = max(b4 + c_stay, a0 + c_stay + c_phase, pb[4]) + e[4]
        new_scores[5] = max(b5 + c_stay, a1 + c_stay + c_phase, pb[5]) + e[5]
        new_scores[6] = max(b6 + c_stay, a2 + c_stay + c_phase, pb[6]) + e[6]
        new_scores[7] = max(b7 + c_stay, a3 + c_stay + c_phase, pb[7]) + e[7]
        scores = new_scores
    best_final = -np.inf
    for k in range(16):
        if scores[k] > best_final: best_final = scores[k]
    return best_final


@njit(fastmath=True, cache=True)
def run_phase_agnostic_hmm_multisnp(child_dip_alleles, child_potential_hom_mask, parent_dip_alleles, 
                                     switch_costs, stay_costs, error_penalty, phase_penalty,
                                     mismatch_penalty=-4.6):
    n_bins = child_dip_alleles.shape[0]
    k_snps = child_dip_alleles.shape[2]
    scores = np.zeros(8)
    BURST_EMISSION_PER_SNP = -0.693147
    for state in range(4, 8):
        scores[state] = -error_penalty
    for i in range(n_bins):
        e0, e1, e2, e3 = 0.0, 0.0, 0.0, 0.0
        valid_snps = 0
        for s in range(k_snps):
            c0_a = child_dip_alleles[i, 0, s]
            c1_a = child_dip_alleles[i, 1, s]
            p0_a = parent_dip_alleles[i, 0, s]
            p1_a = parent_dip_alleles[i, 1, s]
            if c0_a < 0 or c1_a < 0 or p0_a < 0 or p1_a < 0:
                continue
            valid_snps += 1
            if c0_a != p0_a: e0 += mismatch_penalty
            if c1_a != p0_a: e1 += mismatch_penalty
            if c0_a != p1_a: e2 += mismatch_penalty
            if c1_a != p1_a: e3 += mismatch_penalty
        emissions = np.array([e0, e1, e2, e3])
        burst_emission = BURST_EMISSION_PER_SNP * max(valid_snps, 1)
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        prev = scores.copy()
        new_scores = np.zeros(8)
        for state in range(4):
            burst_idx = state + 4
            from_burst = prev[burst_idx]
            from_normal = prev[state] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + burst_emission
        prev_b0 = prev[4]; prev_b1 = prev[5]; prev_b2 = prev[6]; prev_b3 = prev[7]
        src0 = prev[0] + c_stay
        src1 = prev[1] + c_stay + c_phase
        src2 = prev[2] + c_recomb
        new_scores[0] = max(src0, src1, src2, prev_b0) + emissions[0]
        src1 = prev[1] + c_stay
        src0 = prev[0] + c_stay + c_phase
        src3 = prev[3] + c_recomb
        new_scores[1] = max(src1, src0, src3, prev_b1) + emissions[1]
        src2 = prev[2] + c_stay
        src3 = prev[3] + c_stay + c_phase
        src0 = prev[0] + c_recomb
        new_scores[2] = max(src2, src3, src0, prev_b2) + emissions[2]
        src3 = prev[3] + c_stay
        src2 = prev[2] + c_stay + c_phase
        src1 = prev[1] + c_recomb
        new_scores[3] = max(src3, src2, src1, prev_b3) + emissions[3]
        scores = new_scores
    best_final = -np.inf
    for state in range(8):
        if scores[state] > best_final:
            best_final = scores[state]
    return best_final


@njit(fastmath=True, cache=True)
def run_trio_phase_aware_hmm_multisnp(child_dip_alleles, child_potential_hom_mask, 
                                       p1_dip_alleles, p2_dip_alleles, 
                                       switch_costs, stay_costs, error_penalty, phase_penalty,
                                       mismatch_penalty=-4.6):
    n_bins = child_dip_alleles.shape[0]
    k_snps = child_dip_alleles.shape[2]
    BURST_EMISSION_PER_SNP = -1.386
    scores = np.zeros(16)
    for state in range(8, 16):
        scores[state] = -error_penalty
    for i in range(n_bins):
        e = np.zeros(8)
        valid_snps = 0
        for s in range(k_snps):
            c0 = child_dip_alleles[i, 0, s]
            c1 = child_dip_alleles[i, 1, s]
            p1_h0 = p1_dip_alleles[i, 0, s]
            p1_h1 = p1_dip_alleles[i, 1, s]
            p2_h0 = p2_dip_alleles[i, 0, s]
            p2_h1 = p2_dip_alleles[i, 1, s]
            if c0 < 0 or c1 < 0 or p1_h0 < 0 or p1_h1 < 0 or p2_h0 < 0 or p2_h1 < 0:
                continue
            valid_snps += 1
            if c0 != p1_h0: e[0] += mismatch_penalty
            if c1 != p2_h0: e[0] += mismatch_penalty
            if c0 != p1_h0: e[1] += mismatch_penalty
            if c1 != p2_h1: e[1] += mismatch_penalty
            if c0 != p1_h1: e[2] += mismatch_penalty
            if c1 != p2_h0: e[2] += mismatch_penalty
            if c0 != p1_h1: e[3] += mismatch_penalty
            if c1 != p2_h1: e[3] += mismatch_penalty
            if c1 != p1_h0: e[4] += mismatch_penalty
            if c0 != p2_h0: e[4] += mismatch_penalty
            if c1 != p1_h0: e[5] += mismatch_penalty
            if c0 != p2_h1: e[5] += mismatch_penalty
            if c1 != p1_h1: e[6] += mismatch_penalty
            if c0 != p2_h0: e[6] += mismatch_penalty
            if c1 != p1_h1: e[7] += mismatch_penalty
            if c0 != p2_h1: e[7] += mismatch_penalty
        burst_emission = BURST_EMISSION_PER_SNP * max(valid_snps, 1)
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        prev = scores.copy()
        new_scores = np.zeros(16)
        for state in range(8):
            burst_idx = state + 8
            from_burst = prev[burst_idx]
            from_normal = prev[state] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + burst_emission
        cc_0 = 2 * c_stay
        cc_1 = c_recomb + c_stay
        cc_2 = 2 * c_recomb
        p0, p1, p2, p3 = prev[0], prev[1], prev[2], prev[3]
        a0 = max(p0+cc_0, p1+cc_1, p2+cc_1, p3+cc_2)
        a1 = max(p0+cc_1, p1+cc_0, p2+cc_2, p3+cc_1)
        a2 = max(p0+cc_1, p1+cc_2, p2+cc_0, p3+cc_1)
        a3 = max(p0+cc_2, p1+cc_1, p2+cc_1, p3+cc_0)
        p4, p5, p6, p7 = prev[4], prev[5], prev[6], prev[7]
        b4 = max(p4+cc_0, p5+cc_1, p6+cc_1, p7+cc_2)
        b5 = max(p4+cc_1, p5+cc_0, p6+cc_2, p7+cc_1)
        b6 = max(p4+cc_1, p5+cc_2, p6+cc_0, p7+cc_1)
        b7 = max(p4+cc_2, p5+cc_1, p6+cc_1, p7+cc_0)
        pb = prev[8:16]
        new_scores[0] = max(a0 + c_stay, b4 + c_stay + c_phase, pb[0]) + e[0]
        new_scores[1] = max(a1 + c_stay, b5 + c_stay + c_phase, pb[1]) + e[1]
        new_scores[2] = max(a2 + c_stay, b6 + c_stay + c_phase, pb[2]) + e[2]
        new_scores[3] = max(a3 + c_stay, b7 + c_stay + c_phase, pb[3]) + e[3]
        new_scores[4] = max(b4 + c_stay, a0 + c_stay + c_phase, pb[4]) + e[4]
        new_scores[5] = max(b5 + c_stay, a1 + c_stay + c_phase, pb[5]) + e[5]
        new_scores[6] = max(b6 + c_stay, a2 + c_stay + c_phase, pb[6]) + e[6]
        new_scores[7] = max(b7 + c_stay, a3 + c_stay + c_phase, pb[7]) + e[7]
        scores = new_scores
    best_final = -np.inf
    for state in range(16):
        if scores[state] > best_final:
            best_final = scores[state]
    return best_final


@njit(fastmath=True, cache=True, parallel=True)
def score_pair_batch_kernel_multisnp(
    child_alleles, child_hom_mask, stacked_alleles, parent_indices,
    switch_costs, stay_costs, error_penalty, phase_penalty, mismatch_penalty,
):
    """Score (child, parent) pairs for one contig in a single numba call.

    Args:
        child_alleles: (n_bins, 2, k_snps) int8 -- the fixed child's grid
        child_hom_mask: (n_bins,) -- the fixed child's hom mask
        stacked_alleles: (N, n_bins, 2, k_snps) int8 -- all samples stacked
        parent_indices: (n_parents,) int64 -- indices into stacked_alleles
        switch_costs, stay_costs: (n_bins,) per-bin transition costs
        error_penalty, phase_penalty, mismatch_penalty: scalars
    Returns:
        out: (n_parents,) float64 -- one score per parent index

    parallel=True + prange: each iteration is independent (writes to
    out[k], reads from disjoint slices of stacked_alleles), so this is
    safely parallelisable across the worker's allocated numba threads.
    Math is identical to the per-iteration loop: each k computes the
    same run_phase_agnostic_hmm_multisnp(...) call with the same args.
    """
    n_parents = parent_indices.shape[0]
    out = np.empty(n_parents, dtype=np.float64)
    for k in prange(n_parents):
        out[k] = run_phase_agnostic_hmm_multisnp(
            child_alleles, child_hom_mask,
            stacked_alleles[parent_indices[k]],
            switch_costs, stay_costs,
            error_penalty, phase_penalty, mismatch_penalty,
        )
    return out


@njit(fastmath=True, cache=True, parallel=True)
def score_pair_batch_kernel(
    child_alleles, child_hom_mask, stacked_alleles, parent_indices,
    switch_costs, stay_costs, error_penalty, phase_penalty, mismatch_penalty,
):
    """Non-multisnp variant of score_pair_batch_kernel_multisnp.

    Args:
        child_alleles: (n_bins, 2) int8 -- the fixed child's grid
        stacked_alleles: (N, n_bins, 2) int8 -- all samples stacked
        (other args as in score_pair_batch_kernel_multisnp)

    parallel=True + prange: same safety argument as the multisnp variant.
    """
    n_parents = parent_indices.shape[0]
    out = np.empty(n_parents, dtype=np.float64)
    for k in prange(n_parents):
        out[k] = run_phase_agnostic_hmm(
            child_alleles, child_hom_mask,
            stacked_alleles[parent_indices[k]],
            switch_costs, stay_costs,
            error_penalty, phase_penalty, mismatch_penalty,
        )
    return out


@njit(fastmath=True, cache=True, parallel=True)
def score_trio_batch_kernel_multisnp(
    child_alleles, child_hom_mask, stacked_alleles, p1_indices, p2_indices,
    switch_costs, stay_costs, error_penalty, phase_penalty, mismatch_penalty,
):
    """Score (child, p1, p2) trios for one contig in a single numba call.

    Args:
        child_alleles: (n_bins, 2, k_snps) int8 -- the fixed child's grid
        child_hom_mask: (n_bins,) -- the fixed child's hom mask
        stacked_alleles: (N, n_bins, 2, k_snps) int8 -- all samples stacked
        p1_indices, p2_indices: (n_pairs,) int64 -- pair indices into stacked
        switch_costs, stay_costs: (n_bins,) per-bin transition costs
        error_penalty, phase_penalty, mismatch_penalty: scalars
    Returns:
        out: (n_pairs,) float64 -- one trio score per (p1, p2) pair

    parallel=True + prange: each iteration is independent (writes out[k],
    reads disjoint slices of stacked_alleles).  Math identical to the
    per-iteration loop -- each k computes the same trio HMM with the
    same arguments.
    """
    n_pairs = p1_indices.shape[0]
    out = np.empty(n_pairs, dtype=np.float64)
    for k in prange(n_pairs):
        out[k] = run_trio_phase_aware_hmm_multisnp(
            child_alleles, child_hom_mask,
            stacked_alleles[p1_indices[k]],
            stacked_alleles[p2_indices[k]],
            switch_costs, stay_costs,
            error_penalty, phase_penalty, mismatch_penalty,
        )
    return out


@njit(fastmath=True, cache=True, parallel=True)
def score_trio_batch_kernel(
    child_alleles, child_hom_mask, stacked_alleles, p1_indices, p2_indices,
    switch_costs, stay_costs, error_penalty, phase_penalty, mismatch_penalty,
):
    """Non-multisnp variant of score_trio_batch_kernel_multisnp.

    Args:
        child_alleles: (n_bins, 2) int8 -- the fixed child's grid
        stacked_alleles: (N, n_bins, 2) int8 -- all samples stacked
        (other args as in score_trio_batch_kernel_multisnp)

    parallel=True + prange: same safety argument as the multisnp variant.
    """
    n_pairs = p1_indices.shape[0]
    out = np.empty(n_pairs, dtype=np.float64)
    for k in prange(n_pairs):
        out[k] = run_trio_phase_aware_hmm(
            child_alleles, child_hom_mask,
            stacked_alleles[p1_indices[k]],
            stacked_alleles[p2_indices[k]],
            switch_costs, stay_costs,
            error_penalty, phase_penalty, mismatch_penalty,
        )
    return out



__all__ = [
    "DEFAULT_MISMATCH_PENALTY",
    "run_phase_agnostic_hmm",
    "run_phase_agnostic_hmm_multisnp",
    "run_trio_phase_aware_hmm",
    "run_trio_phase_aware_hmm_multisnp",
    "score_pair_batch_kernel",
    "score_pair_batch_kernel_multisnp",
    "score_trio_batch_kernel",
    "score_trio_batch_kernel_multisnp",
]
