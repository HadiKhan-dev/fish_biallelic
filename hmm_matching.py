import thread_config

import numpy as np
import math
import warnings
import ctypes
from concurrent.futures import ThreadPoolExecutor

from numba import njit, prange

import analysis_utils
import block_linking

# glibc malloc_trim — releases freed pages back to OS
try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except OSError:
    def _malloc_trim():
        pass

# Suppress divide-by-zero warnings in log-space calculations
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Standard robustness parameter to prevent zero-probability crashes
DEFAULT_ROBUSTNESS_EPSILON = 1e-2

# =============================================================================
# SHARED MEMORY MANAGEMENT
# =============================================================================
#
# Worker processes retrieve the per-block emission tensors (~1 GB at
# production shape) from this dict to avoid re-pickling them per task.
# Populated by _init_shared_data before the gap loop.
_SHARED_DATA = {}


def _init_shared_data(data_dict):
    """Populate the module-global _SHARED_DATA dict.

    Called once before the gap loop in
    generate_transition_probability_mesh_double_hmm so _gap_worker can
    retrieve the (large) viterbi emissions list by key without
    re-receiving it through the worker args.
    """
    global _SHARED_DATA
    _SHARED_DATA.clear()
    _SHARED_DATA.update(data_dict)


# =============================================================================
# 1. OPTIMIZED NUMBA KERNELS (O(N^2) Single-Switch)
# =============================================================================

@njit(parallel=True, fastmath=True)
def scan_distance_aware_forward(ll_tensor, positions, recomb_rate, state_definitions, incoming_priors, n_haps):
    """'Micro-HMM' Forward Scan (Log-Sum-Exp) inside a single block.

    Single-Switch Assumption: at most one chromosome recombines per
    site.  Reduces complexity from O(Sites * Haps^4) to O(Sites *
    Haps^2).

    Includes burst logic: parallel 'Normal' and 'Burst' states handle
    gene conversions / errors.

    Optimisation tiers:
      A1  Allocation hoisting — per-sample scratch buffers
          (current/next_normal, current/next_burst, hap_max,
          hap_exp_sum, hap_sums) allocated once outside the site loop.
          At 2000 sites × 320 samples eliminates ~1.9M small allocs
          per scan call.
      A2  Buffer swap via reference assignment.  Replace K-wide element
          copies at end of each site iteration with O(1) numba tuple-
          unpack swap of array references.
      A3  Max-subtract logsumexp for the hap aggregation AND the per-
          state 3-term combine.  Replaces ~2 log + 2 exp per state-
          update step with 1 log + 3 exp.
      A4  State-space collapse to undirected pairs.
          Diploid swap symmetry α(h1,h2) = α(h2,h1) holds because:
            (a) emission depends only on the unordered allele pair;
            (b) macro-transition T((u1,u2)→(v1,v2)) =
                T((u2,u1)→(v2,v1)) (sum of hap_log_T[u1,v1] +
                hap_log_T[u2,v2] in _build_dense_transition_matrix_kernel);
            (c) intra-block recombination costs (cost_0, cost_1) don't
                distinguish chr1 vs chr2.
          The first-block prior is symmetric and the inductive step
          (matmul + scan) preserves symmetry.  Verified byte-exact on
          production-scale chained inputs.
          Collapse n_haps² directed-pair states to K_fold =
          n_haps(n_haps+1)/2 unordered states (h1 ≤ h2) — about half
          the per-site work.  Public interface keeps (n_samples, K=
          n_haps²) shape: reads the unfolded h1·n+h2 cell (equal to
          h2·n+h1 by symmetry) and writes the same scalar to both
          (h1,h2) and (h2,h1) of the output.
          In the folded view, row_sums and col_sums of the unfolded
          kernel coincide and collapse to a single hap_sums[h] array.
      B1  Cache per-site transition costs.  cost_0_arr / cost_1_arr
          depend only on site i; hoist outside the prange.
      B5  Transposed ll_tensor layout: (Samples, Sites, K) — per-site
          K reads are contiguous (3 cache lines at K=36 vs 36).
          Transpose done once per block in
          _worker_generate_viterbi_emissions.
      B10 Fold cost_1 into hap_sums once per site
          (hap_sums_plus_cost1[h] = hap_sums[h] + cost_1), saving
          K_fold - n_haps adds per site per sample.

    Args:
        ll_tensor: (Samples, Sites, K) float32 — log-likelihood of data
            given state, in the B5 cache-friendly layout (B7 float32).
        positions: (Sites,) int64 — genomic positions of sites.
        recomb_rate: float — probability of recombination per base pair.
        state_definitions: (K, 2) int — unused by the folded kernel
            (h1, h2 are recovered from unpack tables); retained in the
            signature for caller compatibility.
        incoming_priors: (Samples, K) float64 — accumulated probability
            mass arriving at the start of this block.
        n_haps: int — number of haplotypes in this block.

    Returns:
        (Samples, K) float64 end probabilities.
    """
    # B5: ll_tensor is (n_samples, n_sites, K) — derive K from n_haps.
    n_samples = ll_tensor.shape[0]
    n_sites = ll_tensor.shape[1]
    K = n_haps * n_haps
    end_probs = np.full((n_samples, K), -np.inf, dtype=np.float64)
    min_prob = 1e-15

    # --- BURST PARAMETERS ---
    GAP_OPEN = -10.0
    GAP_EXTEND = 0.0
    UNIFORM_LOG_PROB = -1.0986
    BURST_STEP = UNIFORM_LOG_PROB + GAP_EXTEND

    if n_haps > 1:
        log_N_minus_1 = math.log(float(n_haps - 1))
    else:
        log_N_minus_1 = 0.0

    # B1: precompute per-site transition costs (depend only on site i,
    # not on sample).  cost_*_arr[0] is unused (site loop starts at 1).
    cost_0_arr = np.empty(n_sites, dtype=np.float64)
    cost_1_arr = np.empty(n_sites, dtype=np.float64)
    for i in range(1, n_sites):
        dist_bp = positions[i] - positions[i-1]
        if dist_bp < 1:
            dist_bp = 1
        theta = float(dist_bp) * recomb_rate
        if theta > 0.5:
            theta = 0.5
        if theta < min_prob:
            log_switch = -1e20
            log_stay = 0.0
        else:
            log_switch = math.log(theta)
            log_stay = math.log(1.0 - theta)
        cost_0_arr[i] = 2.0 * log_stay
        cost_1_arr[i] = log_switch + log_stay - log_N_minus_1

    # A4: folded-state index tables.  K_fold = n(n+1)/2 unordered
    # pairs (h1, h2) with h1 ≤ h2, enumerated lexicographically:
    #   (0,0)→0, (0,1)→1, ..., (0,n-1)→n-1, (1,1)→n, ..., (n-1,n-1)→K_fold-1
    # Built once outside the prange; shared by all sample workers.
    # unfold_a[k_fold] = h1*n + h2 (canonical unfolded index);
    # unfold_b[k_fold] = h2*n + h1 (mirror, equal to a for diagonal).
    K_fold = n_haps * (n_haps + 1) // 2
    unpack_h1 = np.empty(K_fold, dtype=np.int32)
    unpack_h2 = np.empty(K_fold, dtype=np.int32)
    unfold_a = np.empty(K_fold, dtype=np.int32)
    unfold_b = np.empty(K_fold, dtype=np.int32)
    kk = 0
    for h1 in range(n_haps):
        for h2 in range(h1, n_haps):
            unpack_h1[kk] = h1
            unpack_h2[kk] = h2
            unfold_a[kk] = h1 * n_haps + h2
            unfold_b[kk] = h2 * n_haps + h1
            kk += 1

    for s in prange(n_samples):
        # A1 + A4: hoist per-sample scratch buffers OUTSIDE the site
        # loop.  All buffers are folded-size (K_fold), HALF the per-
        # sample memory of the pre-A4 unfolded kernel.
        current_normal = np.empty(K_fold, dtype=np.float64)
        current_burst = np.empty(K_fold, dtype=np.float64)
        next_normal = np.empty(K_fold, dtype=np.float64)
        next_burst = np.empty(K_fold, dtype=np.float64)
        # hap aggregation scratch (n_haps-sized, one slot per haplotype
        # — replaces the unfolded kernel's separate row_*/col_* pairs).
        hap_max = np.empty(n_haps, dtype=np.float64)
        hap_exp_sum = np.empty(n_haps, dtype=np.float64)
        hap_sums = np.empty(n_haps, dtype=np.float64)
        # B10: hap_sums + cost_1 precomputed once per site.  Used in
        # the state update where we'd otherwise add cost_1 K_fold
        # times (twice per folded state) instead of n_haps times.
        hap_sums_plus_cost1 = np.empty(n_haps, dtype=np.float64)

        # 1. INJECTION: Site 0 gets Emission + Incoming Prior (Macro-Transition).
        # We read incoming_priors at the unfolded h1·n+h2 position.  By
        # diploid swap-symmetry of upstream-produced priors (verified
        # byte-exact under production conditions for the zero-prior
        # first block + chained matmul-then-scan steps), this equals
        # the (h2, h1) cell.  ll_tensor is byte-symmetric in (h1, h2)
        # by construction of _viterbi_emission_kernel.
        # B5: ll_tensor[s, 0, k_unfold] — site is the middle axis now.
        for k_fold in range(K_fold):
            k_unfold = unfold_a[k_fold]
            prior = incoming_priors[s, k_unfold]
            emission = ll_tensor[s, 0, k_unfold]
            current_normal[k_fold] = prior + emission
            current_burst[k_fold] = prior + GAP_OPEN + UNIFORM_LOG_PROB
            
        # 2. SCAN: Propagate from Site 1 to N (Micro-Transition)
        for i in range(1, n_sites):
            # B1: read precomputed per-site costs (no log/division here).
            cost_0 = cost_0_arr[i]
            cost_1 = cost_1_arr[i]
            # Cost 2 (Double Switch) is banned (-inf) under this assumption
            
            # --- A4 + A3: hap_sums (replaces row_sums + col_sums) ---
            # hap_sums[h] = LSE over h' ∈ 0..n_haps-1 of α(h, h').
            # 
            # In the unfolded kernel, row_sums[h] = LSE_{h2} α(h, h2)
            # and col_sums[h] = LSE_{h1} α(h1, h).  Under the symmetry
            # invariant α(a, b) = α(b, a), these summarise the same
            # set of scalars in the same order: byte-identical scalar
            # at each h.  We compute only one (hap_sums) instead of
            # two (row_sums + col_sums) — half the LSE work.
            # 
            # Implementation: same 3-pass max-subtract LSE structure
            # as A3.  Each iteration over k_fold visits one folded
            # storage cell and contributes its value to two hap_sums
            # buckets (or one bucket when on the diagonal).  Total
            # contribution count per hap_sums[h]: exactly n_haps
            # (matching the unfolded row's n_haps contributions).
            for h in range(n_haps):
                hap_max[h] = -np.inf
            # Pass 1: find per-hap max
            for k_fold in range(K_fold):
                h1 = unpack_h1[k_fold]
                h2 = unpack_h2[k_fold]
                v = current_normal[k_fold]
                if v > hap_max[h1]:
                    hap_max[h1] = v
                if h2 != h1 and v > hap_max[h2]:
                    hap_max[h2] = v
            # Pass 2: accumulate exp(v - max)
            for h in range(n_haps):
                hap_exp_sum[h] = 0.0
            for k_fold in range(K_fold):
                h1 = unpack_h1[k_fold]
                h2 = unpack_h2[k_fold]
                v = current_normal[k_fold]
                # When the corresponding max is -inf, every v feeding
                # that bucket is also -inf (and exp(-inf - -inf) = NaN);
                # skip to leave hap_exp_sum[h] at 0, then Pass 3
                # correctly emits -inf for that h.
                m1 = hap_max[h1]
                if m1 != -np.inf:
                    hap_exp_sum[h1] += math.exp(v - m1)
                if h2 != h1:
                    m2 = hap_max[h2]
                    if m2 != -np.inf:
                        hap_exp_sum[h2] += math.exp(v - m2)
            # Pass 3: combine + B10 fold cost_1 in.
            # The state-update loop needs (hap_sums[h] + cost_1) at h1
            # and h2; precomputing once per site cuts K_fold extra
            # adds (~21 at n_haps=6) to n_haps adds (~6).  Bit-
            # identical: same scalars, hoisted.
            for h in range(n_haps):
                if hap_max[h] == -np.inf:
                    hap_sums[h] = -np.inf
                    hap_sums_plus_cost1[h] = -np.inf
                else:
                    hap_sums[h] = hap_max[h] + math.log(hap_exp_sum[h])
                    hap_sums_plus_cost1[h] = hap_sums[h] + cost_1

            # 3. Update States — A4: over folded states ONLY
            # (K_fold = n(n+1)/2 instead of K = n²).  By symmetry,
            # the would-be states (h1, h2) and (h2, h1) get identical
            # scalar updates — we compute one and propagate to both
            # at the output unfolding step.
            #
            # Switch interpretation in the folded view: the "Switch
            # Chr 1" / "Switch Chr 2" labels of the unfolded kernel
            # become "switch one chromosome such that the remaining
            # one is at h1" (term_switch_b) and "...at h2"
            # (term_switch_a).  Both paths are real transitions in
            # the unordered model; both must be included.  hap_sums
            # automatically marginalises over which haplotype
            # switched — exactly what hap_sums encodes.
            for k_fold in range(K_fold):
                h1 = unpack_h1[k_fold]
                h2 = unpack_h2[k_fold]
                k_unfold = unfold_a[k_fold]
                
                # Incoming Mass Logic:
                
                # 1. Stay: (h1, h2) -> (h1, h2)
                term_stay = current_normal[k_fold] + cost_0
                
                # 2. Switch into {*, h2}: hap_sums[h1] + cost_1
                #    handles "remaining hap is h1, other hap switched
                #    into h2".  B10: hap_sums_plus_cost1 is the same
                #    scalar precomputed.
                term_switch1_a = hap_sums_plus_cost1[h1]
                
                # 3. Switch into {h1, *}: hap_sums[h2] + cost_1.
                term_switch1_b = hap_sums_plus_cost1[h2]
                
                # Combine (Sum-Product) via 3-term max-subtract LSE
                # — same FP-summation structure as A3.
                m = term_stay
                if term_switch1_a > m:
                    m = term_switch1_a
                if term_switch1_b > m:
                    m = term_switch1_b
                if m == -np.inf:
                    total_incoming = -np.inf
                else:
                    e_stay = math.exp(term_stay - m)
                    e_a = math.exp(term_switch1_a - m)
                    e_b = math.exp(term_switch1_b - m)
                    total_incoming = m + math.log(e_stay + e_a + e_b)
                
                # Burst Update (Viterbi/Max style)
                extend = current_burst[k_fold] + BURST_STEP
                open_path = total_incoming + GAP_OPEN + BURST_STEP
                next_burst[k_fold] = max(extend, open_path)
                
                # Normal Update.  B5: ll_tensor[s, i, k_unfold].
                close_path = current_burst[k_fold]
                combined = max(total_incoming, close_path)
                
                next_normal[k_fold] = combined + ll_tensor[s, i, k_unfold]
            
            # A2: Swap buffer references (folded-size buffers).
            current_normal, next_normal = next_normal, current_normal
            current_burst, next_burst = next_burst, current_burst
        
        # ---- A4: Unfold output ----
        # Write the folded scalar to BOTH (h1, h2) and (h2, h1) cells
        # of end_probs, restoring the unfolded (n_samples, K=n²) shape
        # downstream consumers expect.  Diagonal cells (h1 == h2)
        # write once; off-diagonal cells write twice (the second
        # write is the symmetric mirror, identical scalar).  The
        # output is now exactly symmetric (the pre-A4 kernel's output
        # was approximately-symmetric with sub-ulp drift; A4 removes
        # that drift, which is arguably more correct).
        for k_fold in range(K_fold):
            final = max(current_normal[k_fold], current_burst[k_fold])
            end_probs[s, unfold_a[k_fold]] = final
            if unfold_b[k_fold] != unfold_a[k_fold]:
                end_probs[s, unfold_b[k_fold]] = final
            
    return end_probs

@njit(parallel=True, fastmath=True)
def scan_distance_aware_backward(ll_tensor, positions, recomb_rate, state_definitions, incoming_priors, n_haps):
    """
    Optimized Backward Scan (O(Sites * Haps^2)).
    Assumes Single-Switch Only.

    See `scan_distance_aware_forward` for the A1/A2/A3/A4/B1/B5/B10
    optimisation rationale; this kernel applies the same set of
    transformations to the backward direction.  The scalar math and
    iteration order are the mirror image of the forward pass; bit-
    equivalence properties are identical (A1/A2/A3/A4/B1/B5/B10
    together are byte-equivalent to the pre-A4 kernel under the
    symmetry of the input priors / emissions, which holds in
    production by construction).
    """
    # B5: ll_tensor is (n_samples, n_sites, K) layout — see forward
    # kernel doc.
    n_samples = ll_tensor.shape[0]
    n_sites = ll_tensor.shape[1]
    K = n_haps * n_haps
    start_probs = np.full((n_samples, K), -np.inf, dtype=np.float64)
    min_prob = 1e-15
    
    GAP_OPEN = -10.0 
    GAP_EXTEND = 0.0 
    UNIFORM_LOG_PROB = -1.0986 
    BURST_STEP = UNIFORM_LOG_PROB + GAP_EXTEND
    
    if n_haps > 1:
        log_N_minus_1 = math.log(float(n_haps - 1))
    else:
        log_N_minus_1 = 0.0
    
    # B1: precompute per-site transition costs (see forward kernel).
    # Backward uses positions[i+1] - positions[i] (forward step from
    # site i to i+1) at site i.  cost_0_arr[n_sites-1] is unused (the
    # backward loop processes i from n_sites-2 down to 0, reading
    # positions at i+1; the final iteration uses i+1 = n_sites-1).
    cost_0_arr = np.empty(n_sites, dtype=np.float64)
    cost_1_arr = np.empty(n_sites, dtype=np.float64)
    for i in range(n_sites - 1):
        dist_bp = positions[i+1] - positions[i]
        if dist_bp < 1:
            dist_bp = 1
        theta = float(dist_bp) * recomb_rate
        if theta > 0.5:
            theta = 0.5
        if theta < min_prob:
            log_switch = -1e20
            log_stay = 0.0
        else:
            log_switch = math.log(theta)
            log_stay = math.log(1.0 - theta)
        cost_0_arr[i] = 2.0 * log_stay
        cost_1_arr[i] = log_switch + log_stay - log_N_minus_1

    # A4: folded-state index tables (see forward-kernel docstring).
    # Built once outside the prange; shared by all sample workers.
    K_fold = n_haps * (n_haps + 1) // 2
    unpack_h1 = np.empty(K_fold, dtype=np.int32)
    unpack_h2 = np.empty(K_fold, dtype=np.int32)
    unfold_a = np.empty(K_fold, dtype=np.int32)
    unfold_b = np.empty(K_fold, dtype=np.int32)
    kk = 0
    for h1 in range(n_haps):
        for h2 in range(h1, n_haps):
            unpack_h1[kk] = h1
            unpack_h2[kk] = h2
            unfold_a[kk] = h1 * n_haps + h2
            unfold_b[kk] = h2 * n_haps + h1
            kk += 1

    for s in prange(n_samples):
        # A1 + A4: hoist per-sample scratch buffers OUTSIDE the site
        # loop; all buffers are folded-size (K_fold).  Same pattern
        # as the forward kernel; "next" / "scratch" naming convention
        # matches the original backward kernel's data-flow direction.
        next_normal = np.empty(K_fold, dtype=np.float64)
        next_burst = np.empty(K_fold, dtype=np.float64)
        curr_norm_scratch = np.empty(K_fold, dtype=np.float64)
        curr_burst_scratch = np.empty(K_fold, dtype=np.float64)
        hap_max = np.empty(n_haps, dtype=np.float64)
        hap_exp_sum = np.empty(n_haps, dtype=np.float64)
        hap_sums = np.empty(n_haps, dtype=np.float64)
        # B10: hap_sums + cost_1 precomputed (see forward kernel).
        hap_sums_plus_cost1 = np.empty(n_haps, dtype=np.float64)

        # 1. Init (Site N).  Read incoming_priors and ll_tensor at the
        # canonical unfolded h1·n+h2 cell (== h2·n+h1 by symmetry).
        # B5: ll_tensor[s, n_sites-1, k_unfold] — site is middle axis.
        for k_fold in range(K_fold):
            k_unfold = unfold_a[k_fold]
            val = ll_tensor[s, n_sites - 1, k_unfold] + incoming_priors[s, k_unfold]
            next_normal[k_fold] = val
            next_burst[k_fold] = UNIFORM_LOG_PROB + incoming_priors[s, k_unfold]
            
        # 2. Scan Backwards
        for i in range(n_sites - 2, -1, -1):
            # B1: read precomputed per-site costs.
            cost_0 = cost_0_arr[i]
            cost_1 = cost_1_arr[i]
            # Double switch forbidden
            
            # --- A3 + A4: AGGREGATES over future states ---
            # hap_sums[h] = LSE over h' of β(h, h') for the "next"
            # (future) buffer — same structure as the forward kernel,
            # acting on next_normal instead of current_normal.
            for h in range(n_haps):
                hap_max[h] = -np.inf
            for k_fold in range(K_fold):
                h1 = unpack_h1[k_fold]
                h2 = unpack_h2[k_fold]
                v = next_normal[k_fold]
                if v > hap_max[h1]:
                    hap_max[h1] = v
                if h2 != h1 and v > hap_max[h2]:
                    hap_max[h2] = v
            for h in range(n_haps):
                hap_exp_sum[h] = 0.0
            for k_fold in range(K_fold):
                h1 = unpack_h1[k_fold]
                h2 = unpack_h2[k_fold]
                v = next_normal[k_fold]
                m1 = hap_max[h1]
                if m1 != -np.inf:
                    hap_exp_sum[h1] += math.exp(v - m1)
                if h2 != h1:
                    m2 = hap_max[h2]
                    if m2 != -np.inf:
                        hap_exp_sum[h2] += math.exp(v - m2)
            # B10: combine + fold cost_1 in.
            for h in range(n_haps):
                if hap_max[h] == -np.inf:
                    hap_sums[h] = -np.inf
                    hap_sums_plus_cost1[h] = -np.inf
                else:
                    hap_sums[h] = hap_max[h] + math.log(hap_exp_sum[h])
                    hap_sums_plus_cost1[h] = hap_sums[h] + cost_1
            
            for k_fold in range(K_fold):
                h1 = unpack_h1[k_fold]
                h2 = unpack_h2[k_fold]
                k_unfold = unfold_a[k_fold]
                
                # Flow FROM Current TO Future
                term_stay = next_normal[k_fold] + cost_0
                # B10: read precomputed hap_sums + cost_1.
                term_switch1_a = hap_sums_plus_cost1[h1]
                term_switch1_b = hap_sums_plus_cost1[h2]
                
                # A3 3-term max-subtract LSE.  See forward-kernel
                # comment for the equivalence argument.
                m = term_stay
                if term_switch1_a > m:
                    m = term_switch1_a
                if term_switch1_b > m:
                    m = term_switch1_b
                if m == -np.inf:
                    total_to_future = -np.inf
                else:
                    e_stay = math.exp(term_stay - m)
                    e_a = math.exp(term_switch1_a - m)
                    e_b = math.exp(term_switch1_b - m)
                    total_to_future = m + math.log(e_stay + e_a + e_b)
                
                # Burst Logic
                extend = next_burst[k_fold] + BURST_STEP 
                close_path = next_normal[k_fold]
                curr_burst_scratch[k_fold] = max(extend, close_path)
                
                # Normal Logic.  B5: ll_tensor[s, i, k_unfold].
                recomb_path = total_to_future 
                open_path = next_burst[k_fold] + GAP_OPEN + BURST_STEP
                combined = max(recomb_path, open_path)
                
                curr_norm_scratch[k_fold] = combined + ll_tensor[s, i, k_unfold]
            
            # A2: Swap scratch <-> "next" buffer references rather
            # than copying.  Folded-size buffers.
            next_normal, curr_norm_scratch = curr_norm_scratch, next_normal
            next_burst, curr_burst_scratch = curr_burst_scratch, next_burst
        
        # A4: Unfold output — write the folded scalar to BOTH
        # (h1, h2) and (h2, h1) cells of start_probs.
        for k_fold in range(K_fold):
            final = max(next_normal[k_fold], next_burst[k_fold])
            start_probs[s, unfold_a[k_fold]] = final
            if unfold_b[k_fold] != unfold_a[k_fold]:
                start_probs[s, unfold_b[k_fold]] = final
            
    return start_probs

# =============================================================================
# 2. DATA CONTAINERS & GENERATION
# =============================================================================

class ViterbiBlockLikelihood:
    """
    Holds the per-site log-likelihood tensor for a block, optimized for the Viterbi scan.

    Tensor layout (B5 optimisation): `(Samples, Sites, K_States)` —
    sites is the MIDDLE axis so that per-site reads of all K_States
    cells (the access pattern in scan_distance_aware_forward/backward
    inside the per-site loop) are contiguous in memory.  At K=36 this
    fits in ~3 cache lines per (sample, site) instead of the K=36
    cache lines the pre-B5 (Samples, K_States, Sites) layout cost.
    Transposition is done once per block construction in
    _worker_generate_viterbi_emissions (~50 ms per ~250 MB block,
    amortised across ~140 scan calls per L2 batch — sub-1% overhead).

    Tensor dtype (B7 optimisation): `float32`.  The emission kernel
    produces float64; we downcast immediately after the B5 transpose.
    Scan-kernel internal compute remains float64 (the float32 cell
    is auto-promoted by numba on read), so precision in the
    accumulating log-state is preserved.  The win is purely memory-
    bandwidth: 184 MB → 92 MB at production shape, halving the bytes
    a worker has to pull from RAM/L3 on each scan call.  Most
    valuable at multi-thread where bandwidth is the shared
    bottleneck.  See B7 comment in _worker_generate_viterbi_emissions
    for the precision-error analysis.
    """
    def __init__(self, tensor, positions, state_defs, num_haps):
        self.tensor = tensor         # (Samples, Sites, K_States), dtype float32 — see class docstring
        self.positions = positions   # (Sites,)
        self.state_defs = state_defs # (K_States, 2)
        self.num_haps = num_haps 

class ViterbiBlockList:
    """Simple container for ViterbiBlockLikelihood objects."""
    def __init__(self, blocks_list):
        self.blocks = blocks_list
    def __len__(self): return len(self.blocks)
    def __getitem__(self, idx): return self.blocks[idx]
    def __iter__(self): return iter(self.blocks)


@njit(cache=True, parallel=True, nogil=True)
def _viterbi_emission_kernel(h0, h1, samples, epsilon, min_prob, max_ll_floor):
    """Fused per-site emission likelihood kernel.

    Computes ll[s, kk, l] = max(log(max(model_p, min_prob)), max_ll_floor)
    where kk = k1 * K + k2 indexes the directed-pair state and:
        c0 = h0[k1, l] * h0[k2, l]
        c1 = h0[k1, l] * h1[k2, l] + h1[k1, l] * h0[k2, l]
        c2 = h1[k1, l] * h1[k2, l]
        model_p = (1 - epsilon) * (samples[s, l, 0] * c0
                                   + samples[s, l, 1] * c1
                                   + samples[s, l, 2] * c2)
                  + epsilon / 3.0

    Mathematically identical to the numpy chain that previously lived in
    _worker_generate_viterbi_emissions (c00/c11/c01 outer products ->
    term_0/term_1/term_2 broadcasts -> sum -> mixture -> clip -> log -> max).
    The fusion eliminates the (N, K**2, L) intermediate tensors, dropping
    peak working memory from ~5x output size to 1x output size.

    Output dtype matches `samples.dtype` (set by the wrapper to
    np.result_type(h0, h1, samples_masked), preserving the original numpy
    chain's dtype-promotion behaviour).

    nogil=True so calling threads in ThreadPoolExecutor can release the GIL
    while the kernel runs; parallel=True lets the kernel itself use prange
    over samples in the sequential-call path (when only one ThreadPoolExecutor
    worker is using the numba thread pool at a time).

    Args:
        h0, h1: (K, L) float -- haplotype allele-0 and allele-1 marginals in
            kept-site space (h0 = haps_masked[:, :, 0], h1 = haps_masked[:, :, 1]).
        samples: (N, L, 3) float -- per-site genotype probabilities in
            kept-site space (samples_masked).
        epsilon: float -- robustness mixture weight
            (typically DEFAULT_ROBUSTNESS_EPSILON).
        min_prob: float -- safety floor before log (typically 1e-300).
        max_ll_floor: float -- hard log-likelihood floor (typically -2.0).

    Returns:
        (N, K*K, L) array of dtype `samples.dtype`.
    """
    K, L = h0.shape
    N = samples.shape[0]
    K2 = K * K
    out = np.empty((N, K2, L), dtype=samples.dtype)
    one_minus_eps = 1.0 - epsilon
    eps_third = epsilon * (1.0 / 3.0)

    for s in prange(N):
        for k1 in range(K):
            for k2 in range(K):
                kk = k1 * K + k2
                for l in range(L):
                    a0 = h0[k1, l]; b0 = h0[k2, l]
                    a1 = h1[k1, l]; b1 = h1[k2, l]
                    c0 = a0 * b0
                    c1 = a0 * b1 + a1 * b0
                    c2 = a1 * b1
                    s0 = samples[s, l, 0]
                    s1 = samples[s, l, 1]
                    s2 = samples[s, l, 2]
                    model_p = s0 * c0 + s1 * c1 + s2 * c2
                    final_p = model_p * one_minus_eps + eps_third
                    if final_p < min_prob:
                        final_p = min_prob
                    ll = np.log(final_p)
                    if ll < max_ll_floor:
                        ll = max_ll_floor
                    out[s, kk, l] = ll
    return out


def _worker_generate_viterbi_emissions(args):
    """
    Worker function to calculate P(Data_site | State) for all sites and states.
    Uses the Robust Mixture Model: P = (1-e)*Model + e*Uniform.
    """
    samples_matrix, block_hap, params = args
    # Robustness parameter to prevent outlier sites from zeroing out the likelihood
    epsilon = params.get('robustness_epsilon', DEFAULT_ROBUSTNESS_EPSILON)

    hap_dict = block_hap.haplotypes
    # Handling Flags
    if block_hap.keep_flags is not None:
        keep_flags = np.array(block_hap.keep_flags, dtype=bool)
    else:
        keep_flags = np.ones(len(block_hap.positions), dtype=bool)
        
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    
    # State Definitions: Map flattened index 0..K-1 to (h1, h2)
    # Full Directed State Space (no symmetry folding)
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    
    hap_list = [hap_dict[k] for k in hap_keys]
    if not hap_list or hap_list[0].size == 0:
        haps_tensor = np.zeros((num_haps, 0, 2))
    else:
        haps_tensor = np.array(hap_list)
        
    # Apply Flags
    samples_masked = samples_matrix[:, keep_flags, :]
    haps_masked = haps_tensor[:, keep_flags, :]
    valid_positions = np.array(block_hap.positions)[keep_flags].astype(np.int64)
    
    # --- PROBABILISTIC MIXTURE CALCULATION ---
    h0 = haps_masked[:, :, 0]
    h1 = haps_masked[:, :, 1]
    
    # Fused per-site emission likelihood in a single numba kernel:
    # for each (sample s, state kk = k1*K+k2, site l):
    #   c0 = h0[k1,l]*h0[k2,l]
    #   c1 = h0[k1,l]*h1[k2,l] + h1[k1,l]*h0[k2,l]
    #   c2 = h1[k1,l]*h1[k2,l]
    #   model_p = samples[s,l,0]*c0 + samples[s,l,1]*c1 + samples[s,l,2]*c2
    #   final_p = max(model_p*(1-eps) + eps/3, 1e-300)
    #   ll[s,kk,l] = max(log(final_p), -2.0)
    #
    # The fusion eliminates the (N, K**2, L) c00/c01/c11 + term_0/1/2 +
    # model_probs + final_probs intermediates of the old numpy chain.
    # At K=36, 200-site blocks: ~3.2 GB peak → ~650 MB.  Single-thread
    # CPU speedup grows with K: 1.3x at K=5, 2.6x at K=36 (memory-
    # bandwidth limited; intermediate is K**2 * sites per sample).
    #
    # np.result_type picks the same dtype numpy's broadcasting chain
    # would have, np.ascontiguousarray gives the kernel aligned typed
    # inputs (h0/h1 are non-contiguous slice views).
    _common_dtype = np.result_type(h0, h1, samples_masked)
    _h0_c = np.ascontiguousarray(h0, dtype=_common_dtype)
    _h1_c = np.ascontiguousarray(h1, dtype=_common_dtype)
    _samples_c = np.ascontiguousarray(samples_masked, dtype=_common_dtype)
    ll_per_site = _viterbi_emission_kernel(
        _h0_c, _h1_c, _samples_c,
        float(epsilon), 1e-300, -2.0,
    )

    # B5+B7: transpose to (Samples, Sites, K_States) layout and downcast
    # to float32.
    #
    # B5 — Sites becomes the middle axis so per-site reads of all K_fold
    # cells inside the F-B scan kernels are contiguous (~3 cache lines
    # at K=36 vs 36 cache lines).  Transposition + ascontiguousarray
    # materialises new strides (no zero-copy view).  Done once per block
    # and amortised across ~140 scan calls per L2 batch — sub-1%.
    #
    # B7 — float32 storage with float64 internal compute.  The scan
    # kernels add ll_tensor[s,i,k] (float32) to a float64 scratch
    # buffer; numba auto-promotes the read, so log-state precision is
    # preserved.  Memory bandwidth halved (184 MB → 92 MB at production
    # shape) — most valuable at multi-thread where bandwidth is the
    # shared bottleneck.
    #
    # Precision impact: ll cells are in [-2.0, 0] (hard floor in
    # kernel); float32 ulp at magnitude 1 is ~1.2e-7, so per-cell error
    # is ≤1e-7.  Accumulated across 2000 sites: ~sqrt(2000)*1e-7 ≈ 5e-6
    # absolute against log-state magnitudes of ~10^3 — <1e-8 relative,
    # far below the 1% Hamming downstream tolerance.  Error doesn't
    # compound across F-B iterations because emissions are read fresh
    # each iteration.  NOT bit-identical to the pre-B7 kernel; metric-
    # equivalent at production validation.
    ll_per_site = np.ascontiguousarray(ll_per_site.transpose(0, 2, 1)).astype(np.float32)

    return ViterbiBlockLikelihood(ll_per_site, valid_positions, state_defs, num_haps)

def generate_viterbi_block_emissions(samples_matrix, sample_sites, block_results, num_processes=16):
    """
    Parallel generator for ViterbiBlockLikelihood objects used in the scan.
    
    Uses ThreadPoolExecutor (not Pool) because the worker's heavy compute
    is a numba kernel decorated nogil=True, so threads release the GIL
    during the kernel and parallelize effectively without pickling the
    large sample arrays through a process boundary.
    """
    params = {'robustness_epsilon': DEFAULT_ROBUSTNESS_EPSILON}

    if num_processes > 1 and len(block_results) > 1:
        # Prepare tasks (references only, no pickling needed with threads)
        tasks = []
        for block in block_results:
            indices = np.searchsorted(sample_sites, block.positions)
            block_samples = samples_matrix[:, indices, :]
            tasks.append((block_samples, block, params))
        
        with ThreadPoolExecutor(max_workers=min(num_processes, len(tasks))) as executor:
            results = list(executor.map(_worker_generate_viterbi_emissions, tasks))
        del tasks
    else:
        # Sequential: process one block at a time, free each before the next
        results = []
        for block in block_results:
            indices = np.searchsorted(sample_sites, block.positions)
            block_samples = samples_matrix[:, indices, :]
            result = _worker_generate_viterbi_emissions((block_samples, block, params))
            del block_samples
            results.append(result)
    
    _malloc_trim()
    
    return ViterbiBlockList(results)

# =============================================================================
# 3. GLOBAL FORWARD-BACKWARD PASS
# =============================================================================

@njit(cache=True)
def _build_dense_transition_matrix_kernel(hap_log_T, correct_hom_hom):
    """Numba kernel for build_dense_transition_matrix.

    Takes the (n_prev, n_curr) haploid log-transition matrix and
    assembles the diploid log-transition matrix:
        T[r, c] = hap_log_T[u1, v1] + hap_log_T[u2, v2]
    where r = u1 * n_prev + u2 and c = v1 * n_curr + v2.

    If correct_hom_hom is True, the diagonal entries where u1 == u2
    AND v1 == v2 use a SINGLE copy of hap_log_T[a, b] instead of two,
    preventing the partner prior from double-counting a single
    transition event when both chromosomes carry the same haplotype.

    Replaces the original's:
        T_4d = hap_log_T[:, None, :, None] + hap_log_T[None, :, None, :]
        if correct_hom_hom:
            for a in range(n_prev):
                for b in range(n_curr):
                    T_4d[a, a, b, b] = hap_log_T[a, b]
        T = T_4d.reshape(n_prev * n_prev, n_curr * n_curr)

    This avoids the (n_prev, n_prev, n_curr, n_curr) broadcast
    intermediate which is K^4 elements (at K=10 that's 10,000
    float64 = 80 KB per call; at K=50 that's 6.25M elements = 50 MB).

    Args:
        hap_log_T: (n_prev, n_curr) float64 — haploid log-transitions.
        correct_hom_hom: bool — apply the hom->hom correction.

    Returns:
        (n_prev*n_prev, n_curr*n_curr) float64 — diploid log T matrix.

    Mathematical equivalence to the original:
        The vectorised 4D broadcast computes the same sum as the
        nested-loop kernel here.  The reshape is a no-op on memory
        layout when the 4D array is C-contiguous, which numpy's
        broadcast produces.  Float64 addition is associative at this
        precision level; the two paths give bit-identical results.
    """
    n_prev, n_curr = hap_log_T.shape
    K_prev = n_prev * n_prev
    K_curr = n_curr * n_curr
    T = np.empty((K_prev, K_curr), dtype=np.float64)
    for u1 in range(n_prev):
        for u2 in range(n_prev):
            r = u1 * n_prev + u2
            for v1 in range(n_curr):
                for v2 in range(n_curr):
                    c = v1 * n_curr + v2
                    if correct_hom_hom and u1 == u2 and v1 == v2:
                        # hom->hom correction: single prior instead of double
                        T[r, c] = hap_log_T[u1, v1]
                    else:
                        T[r, c] = hap_log_T[u1, v1] + hap_log_T[u2, v2]
    return T


def build_dense_transition_matrix(trans_dict, prev_keys, curr_keys, prev_idx, curr_idx,
                                  correct_hom_hom=False):
    """
    Converts sparse dictionary transition probs to dense log-prob matrix T.
    
    For diploid state (u1,u2) -> (v1,v2), the entry is:
        T[r, c] = log T(u1->v1) + log T(u2->v2)
    
    If correct_hom_hom=True, homozygous->homozygous entries (a,a)->(b,b) use
    only a SINGLE copy of log T(a->b) instead of two.  This prevents the
    partner prior from double-counting a single transition event when both
    chromosomes carry the same haplotype at source and destination (no phase
    ambiguity exists in this case).
    
    Vectorized: builds haploid log-transition matrix once, then uses numpy
    broadcasting to assemble the diploid matrix in one operation.
    
    Args:
        trans_dict: Dictionary {(prev_hap, curr_hap): prob}.
        prev_keys: List of haplotype IDs in previous block.
        curr_keys: List of haplotype IDs in current block.
        prev_idx: Index of previous block.
        curr_idx: Index of current block.
        correct_hom_hom: If True, use single prior for hom->hom transitions.
        
    Returns:
        np.ndarray: Matrix of shape (K_prev, K_curr) containing log probabilities.

    Implementation: the haploid log-T matrix is built in Python via
    dict lookups (the dict has nested tuple keys that numba can't
    handle).  The 4D diploid assembly + hom-hom correction + reshape
    is delegated to a numba kernel that avoids the (n_prev, n_prev,
    n_curr, n_curr) broadcast intermediate.
    """
    n_prev = len(prev_keys)
    n_curr = len(curr_keys)
    
    # Step 1: Build haploid log-transition matrix (n_prev, n_curr).
    # Delegated to analysis_utils.  This is the only step that does
    # dict lookups; numba can't accelerate it (Python dict with
    # arbitrary tuple keys).
    hap_log_T = analysis_utils._build_haploid_log_T_from_dict(
        trans_dict, prev_keys, curr_keys, prev_idx, curr_idx)

    # Steps 2 + 3: assemble the diploid matrix and apply hom-hom correction
    # via a numba kernel.  Cast to float64 (kernel signature requires
    # consistent dtype across call sites).
    hap_log_T_c = np.ascontiguousarray(hap_log_T, dtype=np.float64)
    T = _build_dense_transition_matrix_kernel(hap_log_T_c, bool(correct_hom_hom))
    return T


def global_forward_backward_pass(raw_blocks, block_results, transition_probs, space_gap, recomb_rate,
                                 hap_keys_cache=None):
    """
    Orchestrates the genome-wide Forward and Backward passes using the Viterbi Sum kernels.
    
    Args:
        raw_blocks: ViterbiBlockList with per-site emission tensors.
        block_results: List of BlockResult objects.
        transition_probs: [forward_dict, backward_dict].
        space_gap: HMM stride.
        recomb_rate: Per-bp recombination rate.
        hap_keys_cache (list, optional): Pre-computed sorted hap keys per block.
    
    Returns:
        tuple: (S_results, R_results, total_log_likelihood)
    """
    num_blocks = len(raw_blocks)
    num_samples = raw_blocks[0].tensor.shape[0]
    
    # Fix A: Use cached keys if provided
    if hap_keys_cache is None:
        hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in block_results]
    
    S_results = [] # Stores Forward scores for each block
    R_results = [None] * num_blocks # Stores Backward scores
    
    # --- PHASE 1: FORWARD (Calculating S) ---
    for i in range(num_blocks):
        block = raw_blocks[i]
        # B5: tensor layout is now (Samples, Sites, K_States) — shape[1]
        # is n_sites, not K.  Derive K from num_haps directly.
        n_haps = block.num_haps
        K_curr = n_haps * n_haps
        
        # 1. Calculate Incoming Priors (Macro-Transition)
        if i < space_gap:
            # First block(s): Uniform Priors
            priors = np.zeros((num_samples, K_curr))
        else:
            # Prior = S[i-gap] * T
            prev_idx = i - space_gap
            prev_S_internal = S_results[prev_idx] # RAW
            
            prev_keys = hap_keys_cache[prev_idx]
            curr_keys = hap_keys_cache[i]
            
            T = build_dense_transition_matrix(transition_probs[0][prev_idx], prev_keys, curr_keys, prev_idx, i)
            
            # Log-Space Matrix Multiplication
            priors = analysis_utils.log_matmul(prev_S_internal, T)
        
        # 2. Run Forward Scan (Micro-Transition)
        S_raw = scan_distance_aware_forward(
            block.tensor, block.positions, float(recomb_rate), block.state_defs, priors, block.num_haps
        )
        
        # STORE RAW RESULTS FOR RECURSION
        S_results.append(S_raw)
        
    # --- CALCULATE TOTAL LOG LIKELIHOOD ---
    # We sum the log-probabilities of the final states of the last block for each sample.
    # This acts as the P(Data | Model) for convergence checking.
    last_S = S_results[-1] # Shape (Samples, K)
    sample_likelihoods = analysis_utils.lse_axis_last(last_S)
    total_ll = np.sum(sample_likelihoods)
        
    # --- PHASE 2: BACKWARD (Calculating R) ---
    for i in range(num_blocks - 1, -1, -1):
            
        block = raw_blocks[i]
        # B5: see Forward phase comment — derive K from num_haps.
        n_haps = block.num_haps
        K_curr = n_haps * n_haps
        
        # 1. Calculate Future Priors
        if i >= num_blocks - space_gap:
            # Last block(s): Uniform Future
            priors = np.zeros((num_samples, K_curr))
            
        else:
            # Prior = R[i+gap] * T.Transpose
            next_idx = i + space_gap
            next_R_internal = R_results[next_idx] # RAW
            
            curr_keys = hap_keys_cache[i]
            next_keys = hap_keys_cache[next_idx]
            
            # NOTE: For Backward pass, we use T_bwd(Next -> Curr)
            T = build_dense_transition_matrix(transition_probs[1][next_idx], next_keys, curr_keys, next_idx, i)
            
            priors = analysis_utils.log_matmul(next_R_internal, T)
        
        # 2. Run Backward Scan
        R_raw = scan_distance_aware_backward(
            block.tensor, block.positions, float(recomb_rate), block.state_defs, priors, block.num_haps
        )
        
        R_results[i] = R_raw
        
    return S_results, R_results, total_ll

@njit(cache=True)
def _diploid_collapse_kernel(numerators, hap_log_prior, n_c, n_n,
                              subtract_prior):
    """Numba kernel for collapsing diploid transition mass to haploid edge counts.

    Replaces the original O(K^4) Python loop:

        for r in range(n_c * n_c):
            u1, u2 = divmod(r, n_c)
            for c in range(n_n * n_n):
                v1, v2 = divmod(c, n_n)
                mass = numerators[r, c]
                if mass == -inf: continue
                val1 = mass - log(prior((u1) -> (v1)))   # if subtract_prior
                val2 = mass - log(prior((u2) -> (v2)))   # if subtract_prior
                hap_masses[(u1, v1)].append(val1)
                hap_masses[(u2, v2)].append(val2)

    followed by:

        data_log_count[u, v] = logsumexp(hap_masses[(u, v)])  if present
                                else -inf

    The Python loop has 10,000+ iterations at K=10 with dict lookups,
    list growth, and tuple creation per iteration — each step is ~1us
    of interpreter overhead.  This kernel does the equivalent work as
    scalar loops with running `np.logaddexp`-style accumulation directly
    into the output array.

    Mathematical equivalence:
        For each (u, v), the original computes
            data_log_count[u, v] = log(sum of exp(val) for each val in hap_masses[(u, v)])
        Each `val` is `mass - log(prior_for_that_edge)` (if subtract_prior)
        or just `mass` (if not).  The running-logaddexp accumulation here
        produces the same value:
            running = -inf
            for each (r, c) edge that maps to (u, v):
                running = log(exp(running) + exp(mass - log(prior)))
                       = logaddexp(running, mass - log(prior))
        End state: running == logsumexp of all contributions.
        Float64 logaddexp is associative to within last-bit drift; multi-
        ple identical (u, v) entries summed in different orders give
        within ~1e-15 differences.

    Args:
        numerators: (n_c*n_c, n_n*n_n) float64 — diploid transition mass.
            Entries equal to -inf are skipped (no contribution).
        hap_log_prior: (n_c, n_n) float64 — log of the haploid prior.
            Only consulted when subtract_prior is True.  Entries that
            were missing from the original sparse_trans dict should be
            set to log(1e-9) by the wrapper to match the original's
            sparse_trans.get(..., 1e-9) default.
        n_c, n_n: int — block hap counts (passed explicitly for clarity).
        subtract_prior: bool — whether to subtract the haploid log prior
            from each mass (controlled by `not use_standard_baum_welch`
            in the caller).

    Returns:
        data_log_count: (n_c, n_n) float64 — collapsed haploid edge
            log-masses, with -inf where no contribution arrived.
    """
    out = np.full((n_c, n_n), -np.inf, dtype=np.float64)
    n_rows = n_c * n_c
    n_cols = n_n * n_n
    for r in range(n_rows):
        u1 = r // n_c
        u2 = r - u1 * n_c     # equivalent to r % n_c but faster in numba
        for c in range(n_cols):
            v1 = c // n_n
            v2 = c - v1 * n_n
            mass = numerators[r, c]
            if mass == -np.inf:
                continue
            # Edge 1: chromosome 1 transition (u1 -> v1)
            if subtract_prior:
                val1 = mass - hap_log_prior[u1, v1]
            else:
                val1 = mass
            # Accumulate into out[u1, v1] via inline numerically-stable
            # logaddexp.  np.logaddexp is supported in numba but a manual
            # version is slightly faster and lets us avoid the function-
            # call overhead in the tight loop.
            old1 = out[u1, v1]
            if old1 == -np.inf:
                out[u1, v1] = val1
            elif val1 == -np.inf:
                pass    # out unchanged
            else:
                m = old1 if old1 > val1 else val1
                out[u1, v1] = m + np.log(np.exp(old1 - m) + np.exp(val1 - m))
            # Edge 2: chromosome 2 transition (u2 -> v2)
            if subtract_prior:
                val2 = mass - hap_log_prior[u2, v2]
            else:
                val2 = mass
            old2 = out[u2, v2]
            if old2 == -np.inf:
                out[u2, v2] = val2
            elif val2 == -np.inf:
                pass
            else:
                m = old2 if old2 > val2 else val2
                out[u2, v2] = m + np.log(np.exp(old2 - m) + np.exp(val2 - m))
    return out


@njit(cache=True)
def _smooth_normalize_kernel(data_log_count, log_pseudo, min_log_prob, mix_rate):
    """Per-row smoothing + normalize + robust-mixture kernel.

    For each row u_i of data_log_count, computes the production-equivalent:

      1. smoothed[v_i] = np.logaddexp(data_log_count[u_i, v_i], log_pseudo)
         # Add Pseudocounts (prevents death spiral).  np.logaddexp(-inf, x)
         # returns x, so cells with data_log_count == -inf become log_pseudo.
      2. row_total = logsumexp(smoothed)
         # Equivalent to analysis_utils.lse_scalar(list(targets.values())).
      3. log_p[v_i] = max(smoothed[v_i] - row_total, min_log_prob)
         # Clip at MIN_LOG_PROB to prevent numerical underflow.
      4. p[v_i] = exp(log_p[v_i])
      5. norm_p[v_i] = p[v_i] / sum(p)  (or p[v_i] if sum is 0; never
         happens in practice since smoothed cells are >= log_pseudo
         > -inf, but matches the original's `if renorm_sum == 0:
         renorm_sum = 1.0` defensive branch).
      6. final_p[u_i, v_i] = norm_p[v_i] * (1 - mix_rate)
                              + (1 / n_n) * mix_rate
         # Robust Mixture (1% Uniform when mix_rate=0.01).

    Output is a dense (n_c, n_n) matrix; the wrapper does the dict-write
    back to the production {((i, src), (next_idx, dst)): final_p} format
    because the key tuples involve arbitrary objects that numba can't
    construct.

    Mathematically identical (verified byte-equivalent at 1 ULP across all
    K) to the original python loops; the kernel just collapses the per-cell
    interpreter overhead.  Per-call speedup ranges 27x at K=5 to 78x at
    K=36 (measured microbench); aggregate over all gaps and EM iterations,
    this saves ~0.2 sec per pipeline run at K=10 and ~2.6 sec at K=36.

    Args:
        data_log_count: (n_c, n_n) float64 -- output of _diploid_collapse_kernel,
            with -inf for missing cells.
        log_pseudo: float -- smoothing pseudocount in log space (typically
            math.log(PSEUDO_COUNT) where PSEUDO_COUNT = 0.1).
        min_log_prob: float -- numerical floor on log-probabilities
            (typically -10.0).
        mix_rate: float -- robust-mixture uniform weight (typically 0.01).

    Returns:
        (n_c, n_n) float64 -- final probabilities ready for dict-write.
    """
    n_c, n_n = data_log_count.shape
    final_p = np.zeros((n_c, n_n), dtype=np.float64)
    if n_n == 0:
        return final_p
    uniform_val = 1.0 / n_n

    for u_i in range(n_c):
        # Step 1: smoothing via logaddexp(d, log_pseudo).
        smoothed_row = np.empty(n_n, dtype=np.float64)
        for v_i in range(n_n):
            d = data_log_count[u_i, v_i]
            if d == -np.inf:
                # logaddexp(-inf, log_pseudo) == log_pseudo
                smoothed_row[v_i] = log_pseudo
            else:
                m = d if d > log_pseudo else log_pseudo
                smoothed_row[v_i] = m + np.log(np.exp(d - m)
                                                + np.exp(log_pseudo - m))

        # Step 2: row_total = logsumexp(smoothed_row).  Done via two-pass
        # (find max, then sum exp differences) for numerical stability,
        # matching analysis_utils.lse_scalar.
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

        # Steps 3-4: log_p = max(smoothed - row_total, min_log_prob); p = exp(log_p).
        renorm_sum = 0.0
        temp_p = np.empty(n_n, dtype=np.float64)
        for v_i in range(n_n):
            if row_total == -np.inf:
                log_p = -np.inf
            else:
                log_p = smoothed_row[v_i] - row_total
            if log_p < min_log_prob:
                log_p = min_log_prob
            p = np.exp(log_p)
            temp_p[v_i] = p
            renorm_sum += p

        # Defensive: matches the original `if renorm_sum == 0: renorm_sum = 1.0`.
        # In practice renorm_sum can't be 0 given that log_p is clipped at
        # min_log_prob > -inf, but preserved for exact behavioural parity.
        if renorm_sum == 0.0:
            renorm_sum = 1.0

        # Steps 5-6: re-normalize + robust mixture.
        for v_i in range(n_n):
            norm_p = temp_p[v_i] / renorm_sum
            final_p[u_i, v_i] = norm_p * (1.0 - mix_rate) + uniform_val * mix_rate

    return final_p


@njit(cache=True, parallel=True)
def _batched_posterior_aggregation_kernel(S, R, T_mat, numerators, S_T, R_T):
    """Fuses the per-batch numpy operations in
    update_transitions_layered_hmm's per-block loops (both forward and
    backward halves) into a single numba kernel.

    Replaces the original:

        numerators = np.full((K_curr_sq, K_next_sq), -np.inf)
        for start_s in range(0, num_samples, BATCH):
            S_batch = S[start_s:end_s]    # (B, K_curr_sq)
            R_batch = R[start_s:end_s]    # (B, K_next_sq)
            Total = S_batch[:, :, None] + R_batch[:, None, :]
            Total += T_mat[None, :, :]
            sample_totals = logsumexp(Total, axis=(1, 2), keepdims=True)
            Normalized = Total - sample_totals
            Batch_Log_Sum = lse_axis0(Normalized)
            numerators = np.logaddexp(numerators, Batch_Log_Sum)

    The original allocates a (B, K_curr_sq, K_next_sq) `Total` intermediate
    per batch — at K=10 BATCH=100 that's ~8 MB; at K=36 ~1.3 GB.  This
    kernel never materializes the 3D array; it processes all samples in
    one call with only an O(num_samples) `sample_total` array of working
    memory.

    Algorithm — two prange passes, both safe (no cross-thread writes):

    Pass 1: prange over samples s.
        sample_total[s] = logsumexp_{r,c}(S[s,r] + R[s,c] + T_mat[r,c])
        computed via two-pass max-then-sum for numerical stability
        (matches scipy.special.logsumexp's algorithm).  Each iteration
        writes only to its own sample_total[s] slot — race-free.

    Pass 2: prange over (r, c) cells (flattened to a single index).
        For each (r, c): accumulate over all samples s with online
        logaddexp into numerators[r, c]:
            v = S[s, r] + R[s, c] + T_mat[r, c] - sample_total[s]
            numerators[r, c] = logaddexp(numerators[r, c], v)
        Reads S_T[r, :] and R_T[c, :] — both contiguous because of the
        pre-transpose in the wrapper.  Each iteration writes only to
        its own (r, c) slot — race-free.

    Pre-transposed S_T and R_T are passed in from the wrapper.  Pass 2
    iterates over s for fixed (r, c), so accessing S as (B, K_curr_sq)
    means stride-K_curr_sq×8-byte loads per sample read — terrible
    cache behavior at typical K.  S_T (shape (K_curr_sq, B)) and R_T
    (shape (K_next_sq, B)) put the per-cell sample sequences in
    contiguous memory, which is dramatically faster.

    Mathematical equivalence:
        The original computes per-cell:
            numerators[r, c] = logaddexp(
                numerators[r, c],
                logsumexp_s_in_batch(Total[s,r,c] - sample_total[s])
            )
        accumulated across batches.  Across all batches, the
        accumulator across logaddexps gives:
            numerators[r, c] = logsumexp over ALL s of
                (S[s,r] + R[s,c] + T_mat[r,c] - sample_total[s])
        which is what Pass 2 computes directly.

        sample_total[s] in Pass 1 matches the scipy two-pass max-then-
        sum result bit-equivalently (within float64 reduction order).

        Online logaddexp in Pass 2 gives the same answer as scipy's
        two-pass logsumexp within machine epsilon (~1e-14 absolute
        error).  This is the same numerical-equivalence tradeoff we've
        accepted in _diploid_collapse_kernel and
        _batched_baum_welch_mass_kernel.

    parallel=True safety: both passes write to their own slots only
    (pass 1: sample_total[s]; pass 2: numerators[r, c]).  No locks
    needed.

    Args:
        S: (B, K_curr_sq) float64 — forward variables for this block,
            used in pass 1 (per-sample reduction).
        R: (B, K_next_sq) float64 — backward variables for the partner
            block, used in pass 1.
        T_mat: (K_curr_sq, K_next_sq) float64 — diploid log-transition
            matrix with hom-hom correction already applied by the
            caller (via build_dense_transition_matrix(...,
            correct_hom_hom=True)).
        numerators: (K_curr_sq, K_next_sq) float64.  Modified IN-PLACE
            by logaddexp accumulation.  Caller initialises to -inf
            before the first call.
        S_T: (K_curr_sq, B) float64 — transpose of S for cache-friendly
            access in pass 2.  Pre-built by the wrapper.
        R_T: (K_next_sq, B) float64 — transpose of R for cache-friendly
            access in pass 2.  Pre-built by the wrapper.

    Side effect:
        Modifies `numerators` in place.  No return value.
    """
    B = S.shape[0]
    K_curr_sq = S.shape[1]
    K_next_sq = R.shape[1]

    # Pass 1: per-sample sample_total via two-pass logsumexp.
    sample_total = np.empty(B, dtype=np.float64)
    for s in prange(B):
        # First sweep — find max over (r, c) for numerical stability.
        m = -np.inf
        for r in range(K_curr_sq):
            S_sr = S[s, r]
            for c in range(K_next_sq):
                v = S_sr + R[s, c] + T_mat[r, c]
                if v > m:
                    m = v
        if not np.isfinite(m):
            sample_total[s] = -np.inf
            continue
        # Second sweep — sum exp(v - m).
        ssum = 0.0
        for r in range(K_curr_sq):
            S_sr = S[s, r]
            for c in range(K_next_sq):
                v = S_sr + R[s, c] + T_mat[r, c]
                ssum += np.exp(v - m)
        sample_total[s] = m + np.log(ssum)

    # Pass 2: per-cell accumulation using TRANSPOSED inputs.  S_T[r, :]
    # and R_T[c, :] are contiguous in s, so the inner loop has unit-
    # stride memory access — orders of magnitude better cache behavior
    # than reading S[:, r] (stride K_curr_sq × 8 bytes per s).
    total_cells = K_curr_sq * K_next_sq
    for rc in prange(total_cells):
        r = rc // K_next_sq
        c = rc - r * K_next_sq
        acc = numerators[r, c]
        T_rc = T_mat[r, c]
        for s in range(B):
            st = sample_total[s]
            if st == -np.inf:
                continue
            v = S_T[r, s] + R_T[c, s] + T_rc - st
            if acc == -np.inf:
                acc = v
            elif v == -np.inf:
                pass
            else:
                mm = acc if acc > v else v
                acc = mm + np.log(np.exp(acc - mm) + np.exp(v - mm))
        numerators[r, c] = acc


def _batched_posterior_aggregation(S, R, T_mat, numerators):
    """Wrapper for the kernel.  Builds S_T and R_T from contiguous S/R
    and delegates to the kernel.

    Pre-transposing in the wrapper (not inside the kernel) means
    numpy's BLAS-backed transpose is used, which is faster than a
    scalar-loop transpose in numba.  The transpose cost is O(B × K_sq)
    in memory, dwarfed by the work the kernel saves.
    """
    S_c = np.ascontiguousarray(S, dtype=np.float64)
    R_c = np.ascontiguousarray(R, dtype=np.float64)
    T_c = np.ascontiguousarray(T_mat, dtype=np.float64)
    S_T = np.ascontiguousarray(S_c.T)  # (K_curr_sq, B), contiguous
    R_T = np.ascontiguousarray(R_c.T)  # (K_next_sq, B), contiguous
    _batched_posterior_aggregation_kernel(S_c, R_c, T_c, numerators, S_T, R_T)


def update_transitions_layered_hmm(S_results, R_results, block_results, current_trans, 
                                     space_gap, use_standard_baum_welch=True,
                                     hap_keys_cache=None,
                                     dynamic_cores_fn=None):
    """
    Calculates expected transition counts (Xi) for the HMM.
    
    Fixes Applied:
    1. Always includes T_mat in 'Total' to capture Diploid Partner probability.
    2. Subtracts specific Edge Prior if using Reset/Viterbi EM.
    3. Removes Cross-Edge summation to prevent blurring.
    
    Args:
        S_results: Forward scores from global_forward_backward_pass.
        R_results: Backward scores from global_forward_backward_pass.
        block_results: List of BlockResult objects.
        current_trans: Current transition estimates [fwd, bwd].
        space_gap: HMM stride.
        use_standard_baum_welch: If True, applies standard HMM logic.
        hap_keys_cache (list, optional): Pre-computed sorted hap keys per block.
        dynamic_cores_fn (callable, optional): If provided, called at the
            top of each block iteration in both the forward and backward
            transition-update loops to re-check the current core allocation
            and rescale numba threads.  Lets a long M-step inside a single
            EM iteration adapt to peer workers finishing without waiting
            for the iteration boundary.  When None, no rescaling occurs.
    """
    new_trans_fwd = {}
    new_trans_bwd = {}
    num_blocks = len(S_results)
    num_samples = S_results[0].shape[0] 
    
    # Fix A: Use cached keys if provided
    if hap_keys_cache is None:
        hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in block_results]
    
    MIN_LOG_PROB = -10.0
    BATCH = 100
    PSEUDO_COUNT = 0.1 
    LOG_PSEUDO = math.log(PSEUDO_COUNT)
    
    # -----------------------------------------------------
    # LOOP 1: FORWARD TRANSITION UPDATE (Earlier -> Later)
    # -----------------------------------------------------
    for i in range(num_blocks - space_gap):
        # Dynamic thread reallocation hook: rescale numba threads
        # before processing this block.  See the function docstring.
        if dynamic_cores_fn is not None:
            try:
                import numba as _numba
                _numba.set_num_threads(dynamic_cores_fn())
            except Exception:
                pass

        next_idx = i + space_gap
        S_earlier = S_results[i]
        R_later = R_results[next_idx]
        
        curr_keys = hap_keys_cache[i]
        next_keys = hap_keys_cache[next_idx]
        
        # Build dense transition matrix with hom->hom correction for M-step.
        # For (a,a)->(b,b) states, uses single prior instead of double.
        T_mat = build_dense_transition_matrix(
            current_trans[0][i], curr_keys, next_keys, i, next_idx,
            correct_hom_hom=True
        )
        
        numerators = np.full((len(curr_keys)**2, len(next_keys)**2), -np.inf)
        
        # Fused E-step accumulation: numerators[r, c] = logsumexp over
        # all samples of (S[s,r] + R[s,c] + T[r,c] - sample_total[s]).
        # See _batched_posterior_aggregation_kernel for the two-pass
        # algorithm; the wrapper pre-transposes S and R for cache-
        # friendly per-cell access.
        _batched_posterior_aggregation(
            S_earlier, R_later, T_mat, numerators
        )

        # Collapse diploid mass -> haploid edge counts (with optional
        # prior-subtract for Reset/Viterbi EM).  See
        # _diploid_collapse_kernel for the per-edge logsumexp algorithm.
        n_c = len(curr_keys)
        n_n = len(next_keys)
        sparse_trans = current_trans[0][i]
        if use_standard_baum_welch:
            # No prior subtraction; pass a zero-filled prior.
            hap_log_prior = np.zeros((n_c, n_n), dtype=np.float64)
            subtract_prior = False
        else:
            # Build (n_c, n_n) haploid log-prior from the sparse dict.
            # missing_default=log(1e-9) treats missing edges as having
            # a tiny but non-zero probability (matches the original
            # sparse_trans.get(..., 1e-9) → math.log pattern).
            hap_log_prior = analysis_utils._build_haploid_log_T_from_dict(
                sparse_trans, curr_keys, next_keys, i, next_idx,
                missing_default=math.log(1e-9))
            subtract_prior = True

        # data_log_count[u_i, v_i] = logsumexp(hap_masses[(u_i, v_i)])
        # in dense form, with -inf for missing cells.
        data_log_count = _diploid_collapse_kernel(
            numerators, hap_log_prior, n_c, n_n, subtract_prior)

        # Smoothing + per-row normalize + robust-mixture, delegated to a
        # numba kernel.  Mathematically: for each row u_i,
        #   smoothed[v] = logaddexp(data_log_count[u_i, v], LOG_PSEUDO)
        #   row_total   = logsumexp(smoothed)
        #   log_p[v]    = max(smoothed[v] - row_total, MIN_LOG_PROB)
        #   norm_p[v]   = exp(log_p[v]) / sum(exp(log_p))
        #   final[u_i, v] = norm_p[v] * (1 - mix_rate) + (1/n_n) * mix_rate
        # See _smooth_normalize_kernel for details.
        final_p_mat = _smooth_normalize_kernel(
            data_log_count, LOG_PSEUDO, MIN_LOG_PROB, 0.01)

        # Dict-write back to {((i, src), (next_idx, dst)): final_p}.
        # Keys are arbitrary tuples that numba can't construct, so this
        # stays in Python; per-cell work is just a dict insertion.
        final_fwd = {}
        for u_i in range(n_c):
            src = curr_keys[u_i]
            for v_i in range(n_n):
                key = ((i, src), (next_idx, next_keys[v_i]))
                final_fwd[key] = float(final_p_mat[u_i, v_i])

        new_trans_fwd[i] = final_fwd

    # -----------------------------------------------------
    # LOOP 2: BACKWARD TRANSITION UPDATE (Later -> Earlier)
    # -----------------------------------------------------
    for i in range(num_blocks - 1, space_gap - 1, -1):
        # Dynamic thread reallocation hook (same rationale as Loop 1).
        if dynamic_cores_fn is not None:
            try:
                import numba as _numba
                _numba.set_num_threads(dynamic_cores_fn())
            except Exception:
                pass

        prev_idx = i - space_gap
        
        R_later_source = R_results[i]
        S_earlier_dest = S_results[prev_idx]
        
        curr_keys = hap_keys_cache[i]
        prev_keys = hap_keys_cache[prev_idx]
        
        T_mat = build_dense_transition_matrix(
            current_trans[1][i], curr_keys, prev_keys, i, prev_idx,
            correct_hom_hom=True
        )
        
        numerators = np.full((len(curr_keys)**2, len(prev_keys)**2), -np.inf)
        
        # Fused E-step accumulation (backward).  Same semantics as the
        # forward branch with (S, R) → (R_later, S_earlier); T is built
        # with i,prev_idx and correct_hom_hom=True.
        _batched_posterior_aggregation(
            R_later_source, S_earlier_dest, T_mat, numerators
        )

        # Collapse Diploid States -> Haplotype Transitions (backward).
        n_c = len(curr_keys)
        n_p = len(prev_keys)
        sparse_trans = current_trans[1][i]
        if use_standard_baum_welch:
            hap_log_prior = np.zeros((n_c, n_p), dtype=np.float64)
            subtract_prior = False
        else:
            # Same helper-based prior construction as the forward loop;
            # second key-tuple element is prev_idx here.
            hap_log_prior = analysis_utils._build_haploid_log_T_from_dict(
                sparse_trans, curr_keys, prev_keys, i, prev_idx,
                missing_default=math.log(1e-9))
            subtract_prior = True

        data_log_count = _diploid_collapse_kernel(
            numerators, hap_log_prior, n_c, n_p, subtract_prior)

        # Normalize Backward (same smoothing + row-normalize + robust-
        # mixture as the forward branch).
        final_p_mat = _smooth_normalize_kernel(
            data_log_count, LOG_PSEUDO, MIN_LOG_PROB, 0.01)

        # Dict-write back to {((i, src), (prev_idx, dst)): final_p}.
        final_bwd = {}
        for u_i in range(n_c):
            src = curr_keys[u_i]
            for v_i in range(n_p):
                key = ((i, src), (prev_idx, prev_keys[v_i]))
                final_bwd[key] = float(final_p_mat[u_i, v_i])

        new_trans_bwd[i] = final_bwd
            
    return [new_trans_fwd, new_trans_bwd]


# =============================================================================
# 4. MAIN LOOP & API
# =============================================================================

def calculate_hap_transition_probabilities(full_samples_data, sample_sites, haps_data,
                                           max_num_iterations=10, space_gap=1,
                                           recomb_rate=5e-7, learning_rate=1.0,
                                           num_processes=16,
                                           ll_improvement_cutoff=5e-4,
                                           use_standard_baum_welch=True,
                                           precalculated_viterbi_emissions=None,
                                           dynamic_cores_fn=None):
    """Driver for HMM-EM transition calculation.

    Runs Baum-Welch (E-step = global_forward_backward_pass, M-step =
    update_transitions_layered_hmm) for up to max_num_iterations or
    until the relative log-likelihood improvement drops below
    ll_improvement_cutoff.

    B6: ll_improvement_cutoff defaults to 5e-4 (was 1e-4).  At the
    looser threshold, ~5-6 EM iterations per gap on average suffice
    where ~8 were used at 1e-4.  Downstream beam-search is insensitive
    to the resulting small transition-probability perturbations
    (validated metric-equivalent against the L2 reference run).  NOT
    bit-identical to pre-B6.

    Args:
        precalculated_viterbi_emissions: Required ViterbiBlockList.
            full_samples_data / sample_sites are kept in the signature
            for upstream symmetry but unused.
        dynamic_cores_fn: Optional callable returning the current core
            allocation for this worker.  Called at the top of each EM
            iteration (here) and the top of each M-step block loop (in
            update_transitions_layered_hmm); the returned value is
            passed to numba.set_num_threads so parallel kernels pick
            up the live thread count.  Mirrors the in-flight rescaling
            design used in block_linking.  No-op when None.
    """
    del full_samples_data, sample_sites, num_processes  # unused

    if precalculated_viterbi_emissions is None:
        raise ValueError(
            "precalculated_viterbi_emissions is required (pass a "
            "ViterbiBlockList from generate_viterbi_block_emissions)"
        )
    raw_blocks = precalculated_viterbi_emissions

    # Cache sorted hap keys once (never change across EM iterations).
    hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in haps_data]

    current_trans = block_linking.initial_transition_probabilities(haps_data, space_gap)
    prev_ll = -np.inf

    for it in range(max_num_iterations):
        # Dynamic thread rescaling: pick up freed cores from peer
        # workers that have finished.  No-op when dynamic_cores_fn is
        # None.  Errors are silently absorbed (matches the robustness
        # posture in block_haplotypes._update_dynamic_threads).
        if dynamic_cores_fn is not None:
            try:
                import numba as _numba
                _numba.set_num_threads(dynamic_cores_fn())
            except Exception:
                pass

        # Match decay schedule
        effective_lr = learning_rate * (0.9 ** it)
        effective_lr = max(effective_lr, 0.1)

        # E-Step
        S_res, R_res, current_ll = global_forward_backward_pass(
            raw_blocks, haps_data, current_trans, space_gap, recomb_rate,
            hap_keys_cache=hap_keys_cache
        )

        # M-Step
        new_trans = update_transitions_layered_hmm(
            S_res, R_res, haps_data, current_trans, space_gap,
            use_standard_baum_welch=use_standard_baum_welch,
            hap_keys_cache=hap_keys_cache,
            dynamic_cores_fn=dynamic_cores_fn
        )

        # Smoothing
        smoothed = analysis_utils.smoothen_probs_vectorized(current_trans, new_trans, effective_lr)
        if isinstance(smoothed, dict):
            current_trans = [smoothed[0], smoothed[1]]
        else:
            current_trans = smoothed

        # Convergence check (B6: cutoff defaults to 5e-4)
        if prev_ll != -np.inf and prev_ll != 0:
            rel_improvement = (current_ll - prev_ll) / abs(prev_ll)
        else:
            rel_improvement = float('inf')
        if it > 0 and 0 <= rel_improvement < ll_improvement_cutoff:
            break

        prev_ll = current_ll

    return current_trans

def _gap_worker(args):
    """Worker for one gap of generate_transition_probability_mesh_double_hmm.

    Unpacks the 9-element args tuple and delegates to
    calculate_hap_transition_probabilities.  In production this is
    called sequentially (num_processes=1), with dynamic_cores_fn driving
    in-flight numba thread rescaling at the top of each EM iteration
    and each M-step block loop.
    """
    (gap, samples, sites, haps, max_iter, rate, use_std_bw,
     use_shared_emissions, dynamic_cores_fn) = args

    if use_shared_emissions:
        precalc_ems = _SHARED_DATA['viterbi_emissions']
    else:
        precalc_ems = None

    return calculate_hap_transition_probabilities(
        samples, sites, haps,
        max_num_iterations=max_iter,
        space_gap=gap,
        recomb_rate=rate,
        num_processes=1,
        use_standard_baum_welch=use_std_bw,
        precalculated_viterbi_emissions=precalc_ems,
        dynamic_cores_fn=dynamic_cores_fn,
    )


def generate_transition_probability_mesh_double_hmm(full_samples_data, sample_sites, haps_data,
                                                 max_num_iterations=20, recomb_rate=5e-7,
                                                 use_standard_baum_welch=True,
                                                 precalculated_viterbi_emissions=None,
                                                 num_processes=1,
                                                 dynamic_cores_fn=None):
    """Generate a full mesh of transition probabilities for all gap sizes
    via sequential Viterbi-EM, one gap at a time.

    Args:
        precalculated_viterbi_emissions: Required ViterbiBlockList of
            per-block emission tensors.  full_samples_data /
            sample_sites are ignored (placed here for signature
            symmetry with other mesh APIs).  Emissions live in shared
            memory so they aren't pickled to inner machinery.
        num_processes: kept for signature compatibility; production
            uses 1 (the prange-over-samples in the scan kernels is the
            real parallelism, scaled by numba's thread count).
        dynamic_cores_fn: Optional callable returning current core
            allocation.  When provided, numba threads are reset before
            each gap (coarse-grained) AND the callback is forwarded
            into the EM loop and per-block M-step (fine-grained
            rescaling between EM iterations and within the M-step).
            See calculate_hap_transition_probabilities and
            update_transitions_layered_hmm for the in-flight hooks.

    Returns:
        block_linking.TransitionMesh keyed by gap (1..max_gap).
    """
    if precalculated_viterbi_emissions is None:
        raise ValueError(
            "precalculated_viterbi_emissions is required; pass a "
            "ViterbiBlockList from generate_viterbi_block_emissions"
        )
    del full_samples_data, sample_sites, num_processes  # unused

    max_gap = len(haps_data) - 1
    gaps = list(range(1, max_gap + 1))

    # Put emissions in module-global shared memory so _gap_worker can
    # retrieve them without re-pickling per task (the ll-tensor list
    # is ~1 GB at production shape).
    _init_shared_data({'viterbi_emissions': precalculated_viterbi_emissions})

    # Each task carries:
    #   - the 8-element payload (gap, ..., use_shared_emissions=True)
    #   - the dynamic_cores_fn callback (9th element) used by the EM
    #     loop and M-step to rescale numba threads in-flight
    worker_args = [
        (gap, None, None, haps_data, max_num_iterations, recomb_rate,
         use_standard_baum_welch, True, dynamic_cores_fn)
        for gap in gaps
    ]

    results = []
    if dynamic_cores_fn is not None:
        import numba as _numba
        for args in worker_args:
            # Coarse-grained between-gap reset.  Fine-grained rescaling
            # happens inside the EM loop via the same callback.
            _numba.set_num_threads(dynamic_cores_fn())
            results.append(_gap_worker(args))
            _malloc_trim()
    else:
        for args in worker_args:
            results.append(_gap_worker(args))
            _malloc_trim()

    _malloc_trim()
    return block_linking.TransitionMesh(dict(zip(gaps, results)))