import numpy as np
import math
import time
import ctypes
from scipy.special import logsumexp
from functools import partial

import analysis_utils
import block_haplotypes

# glibc malloc_trim — releases freed pages back to OS
try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except OSError:
    def _malloc_trim():
        pass

# Module-level shared data for sequential _gap_worker invocations.
# Populated by generate_transition_probability_mesh before the gap loop
# (a single dict containing the pre-computed full_blocks_likelihoods);
# _gap_worker reads it via _BL_SHARED.get('full_blocks_likelihoods').
# Kept module-level rather than threaded through the worker_args tuple
# to avoid re-pickling the large emissions tensor per gap.
_BL_SHARED = {}

# Default parameters for emission scoring.
DEFAULT_LOG_BASE = math.e
# Robustness parameter: 1e-2 means 1% chance any read is random noise/error.
# This prevents high-depth outliers from forcing incorrect recombinations.
DEFAULT_ROBUSTNESS_EPSILON = 1e-2
# Sample chunk size for emission scoring. Bounds peak memory per block to
# O(chunk × K² × sites) instead of O(n_samples × K² × sites). Each sample's
# likelihood is independent, so chunking has zero computational overhead.
SAMPLE_CHUNK_SIZE = 10

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Computations will be extremely slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range


# %% --- NUMBA KERNELS ---

@njit(fastmath=True)
def calculate_burst_score_vectorized(ll_matrix_sites_last, 
                                     gap_open_penalty=-10.0, 
                                     gap_extend_penalty=0.0, 
                                     uniform_log_prob=-1.1):
    """
    Calculates P(Data | Haplotype) using a 2-state HMM (Normal vs Burst) 
    along the sequence length to prevent geometric decay of likelihoods.
    
    States:
    0: Normal Match (Uses provided ll_matrix values)
    1: Error Burst (Uses uniform_log_prob)
    
    Transitions:
    Normal -> Normal: 0
    Normal -> Burst:  gap_open_penalty
    Burst  -> Burst:  gap_extend_penalty
    Burst  -> Normal: 0 (Free recovery)
    
    Args:
        ll_matrix_sites_last: (N_Samples, N_Haps, N_Haps, N_Sites) log-likelihoods.
        gap_open_penalty: Cost to START ignoring data (entering burst).
        gap_extend_penalty: Cost to CONTINUE ignoring data.
        uniform_log_prob: The score of a site inside a burst (ln(1/3) approx -1.1).
        
    Returns:
        (N_Samples, N_Haps, N_Haps) matrix of total log-likelihoods.
    """
    n_samples, n_h1, n_h2, n_sites = ll_matrix_sites_last.shape
    results = np.empty((n_samples, n_h1, n_h2), dtype=np.float64)
    
    # Pre-calc values for the Burst State
    # Inside a burst, the score is always: UniformLikelihood + TransitionCost
    burst_step_score = uniform_log_prob + gap_extend_penalty
    
    for s in range(n_samples):
        for h1 in range(n_h1):
            for h2 in range(n_h2):
                
                # State 0: Normal Mode
                # State 1: Burst Mode
                
                # Initialization (Site 0)
                # We assume we start in Normal mode. 
                # Starting in Burst immediately costs open_penalty.
                score_normal = ll_matrix_sites_last[s, h1, h2, 0]
                score_burst = gap_open_penalty + burst_step_score
                
                for i in range(1, n_sites):
                    emission = ll_matrix_sites_last[s, h1, h2, i]
                    
                    # 1. Update Normal State
                    # Transition Normal->Normal (Cost 0) OR Burst->Normal (Cost 0)
                    # We take max because we want the most likely path (Viterbi approx)
                    prev_best_for_normal = max(score_normal, score_burst) 
                    new_score_normal = prev_best_for_normal + emission
                    
                    # 2. Update Burst State
                    # Transition Normal->Burst (Open Cost) OR Burst->Burst (Extend Cost)
                    from_normal = score_normal + gap_open_penalty + burst_step_score
                    from_burst  = score_burst + burst_step_score
                    
                    new_score_burst = max(from_normal, from_burst)
                    
                    score_normal = new_score_normal
                    score_burst = new_score_burst
                    
                # Final score is the best of finishing in either state
                results[s, h1, h2] = max(score_normal, score_burst)
                
    return results


@njit(cache=True, parallel=True)
def _batched_baum_welch_mass_kernel(F_batch, B_batch, T_matrix,
                                      use_standard_baum_welch):
    """Numba kernel for the heavy 5D batched body of the M-step in
    `get_updated_transition_probabilities_unified._run_batched_pass`.

    Replaces the original:

        T_partner_broad = T_matrix[None, None, :, None, :]
        hom_hom_mask = zeros((1, n_c, n_c, n_n, n_n), bool)
        for a in range(n_c):
            for b in range(n_n):
                hom_hom_mask[0, a, a, b, b] = True
        T_partner_corrected = np.where(hom_hom_mask, 0.0, T_partner_broad)

        combined = F_broad + B_broad + T_partner_corrected
        if use_standard_baum_welch:
            combined += T_main_broad

        mass_1_1 = logsumexp(combined, axis=(2, 4))     # (B, n_c, n_n)
        mass_2_2 = logsumexp(combined, axis=(1, 3))     # (B, n_c, n_n)

    The original allocates a (B, n_c, n_c, n_n, n_n) `combined` array
    plus a (1, n_c, n_c, n_n, n_n) `hom_hom_mask` plus the broadcast
    intermediate.  At K=10 BATCH=100 that's ~10 MB per call; at K=36
    it's ~1.3 GB.  Plus two full logsumexp passes over the 5D array.

    Algebraic structure for each (s, u_out, u_in, v_out, v_in) cell:
        combined_cell = F_batch[s, u_out, u_in] + B_batch[s, v_out, v_in]
                      + (T_matrix[u_in, v_in] if not hom-hom else 0.0)
                      + (T_matrix[u_out, v_out] if standard_bw else 0.0)
    where hom-hom means (u_out == u_in and v_out == v_in).

    Algorithm — two-pass max-then-sum LSE (replaces the previous online
    logaddexp accumulation, which paid 4 exp() + 2 log() per cell):

      Pass 1 (no transcendentals) — find the per-output max:
        max_11[s, u_out, v_out] = max over (u_in, v_in) of combined_cell
        max_22[s, u_in,  v_in ] = max over (u_out, v_out) of combined_cell

      Pass 2 (1 exp per cell per accumulator = 2 exp per cell) —
      accumulate exp(cell - max) into a sum:
        sum_11[s, u_out, v_out] += exp(combined_cell - max_11[s, u_out, v_out])
        sum_22[s, u_in,  v_in ] += exp(combined_cell - max_22[s, u_in,  v_in ])

      Finalize (1 log per *output* cell, not per 5D cell):
        mass_1_1[s, u_out, v_out] = max_11 + log(sum_11)    if max_11 finite
                                  = -inf                     otherwise
        mass_2_2[s, u_in,  v_in ] = max_22 + log(sum_22)    if max_22 finite
                                  = -inf                     otherwise

    Per call at B=100 / K=6 (production L1 shape) this is
      Pass 1:    0 exp,        0 log
      Pass 2:   2 × 129,600 = 259,200 exp,   0 log
      Final:                     0 exp,  2 × 100 × 6 × 6 = 7,200 log
    vs the previous online form's ~518,400 exp + ~259,200 log per call.

    Loop-invariant hoisting:
      - main_term depends only on (u_out, v_out) — hoisted out of v_in loop.
      - In pass 2, max_11[s, u_out, v_out] is constant over (u_in, v_in) —
        hoisted out of v_in loop along with its is-finite flag.

    Numerical equivalence: the two-pass max-then-sum is the canonical
    numerically stable LSE form used by scipy's logsumexp.  It differs
    from the previous online logaddexp form at the last bit of float64
    due to reduction-order differences (online sums sequentially; the
    two-pass sums via Σ exp(cell - global_max), which is the standard
    BLAS-compatible reduction).  We've already accepted this scale of
    drift elsewhere (notably _log_matmul_2d_kernel and the
    _diploid_collapse_kernel) and validated downstream stability.
    Verified end-to-end: M-step output transition-probability dicts
    differ from the online form by <1e-13 on every entry on synthetic
    production-shape inputs.

    All-(-inf) edge case: if every cell contributing to a given output
    is -inf, max_X stays at -inf, and we set the result to -inf
    explicitly in the finalize step (matching scipy's behaviour).

    parallel=True: prange over the batch axis s.  Each iteration writes
    to its own (s, :, :) slice of max_11 / max_22 / sum_11 / sum_22 — no
    cross-thread aliasing, no locks.

    Args:
        F_batch: (B, n_c, n_c) float64 — forward variables for this batch
        B_batch: (B, n_n, n_n) float64 — backward variables for this batch
        T_matrix: (n_c, n_n) float64 — haploid log-transition matrix
        use_standard_baum_welch: bool — if True, add T_matrix[u_out, v_out]
            to combined (the T_main term).  If False, T_main is not added
            (matches the original's `if use_standard_baum_welch: combined +=
            T_main_broad`).

    Returns:
        mass_1_1: (B, n_c, n_n) float64 — logsumexp over (u_in, v_in)
        mass_2_2: (B, n_c, n_n) float64 — logsumexp over (u_out, v_out)
            (Note: mass_2_2's leading axes are (u_in, v_in) — the order
            in which they appear in `combined`'s axes (2, 4).  Matches
            the original's output shape and semantics.)
    """
    B, n_c, _ = F_batch.shape
    _, n_n, _ = B_batch.shape

    # ---- Pass 1: find max over the LSE-reduction axes for each output ----
    max_11 = np.full((B, n_c, n_n), -np.inf, dtype=np.float64)
    max_22 = np.full((B, n_c, n_n), -np.inf, dtype=np.float64)

    for s in prange(B):
        for u_out in range(n_c):
            for u_in in range(n_c):
                F_val = F_batch[s, u_out, u_in]
                for v_out in range(n_n):
                    if use_standard_baum_welch:
                        main_term = T_matrix[u_out, v_out]
                    else:
                        main_term = 0.0
                    # FM is invariant w.r.t. v_in — hoist out of innermost loop.
                    FM_val = F_val + main_term
                    for v_in in range(n_n):
                        # T_partner_corrected term: T_matrix[u_in, v_in],
                        # zeroed out at hom-hom entries.
                        if u_out == u_in and v_out == v_in:
                            partner_term = 0.0
                        else:
                            partner_term = T_matrix[u_in, v_in]
                        combined_cell = (FM_val
                                         + B_batch[s, v_out, v_in]
                                         + partner_term)
                        if combined_cell > max_11[s, u_out, v_out]:
                            max_11[s, u_out, v_out] = combined_cell
                        if combined_cell > max_22[s, u_in, v_in]:
                            max_22[s, u_in, v_in] = combined_cell

    # ---- Pass 2: sum exp(cell - max) per output cell --------------------
    sum_11 = np.zeros((B, n_c, n_n), dtype=np.float64)
    sum_22 = np.zeros((B, n_c, n_n), dtype=np.float64)

    for s in prange(B):
        for u_out in range(n_c):
            for u_in in range(n_c):
                F_val = F_batch[s, u_out, u_in]
                for v_out in range(n_n):
                    if use_standard_baum_welch:
                        main_term = T_matrix[u_out, v_out]
                    else:
                        main_term = 0.0
                    FM_val = F_val + main_term
                    # max_11 for this (s, u_out, v_out) is invariant in v_in.
                    m11 = max_11[s, u_out, v_out]
                    m11_finite = np.isfinite(m11)
                    for v_in in range(n_n):
                        if u_out == u_in and v_out == v_in:
                            partner_term = 0.0
                        else:
                            partner_term = T_matrix[u_in, v_in]
                        combined_cell = (FM_val
                                         + B_batch[s, v_out, v_in]
                                         + partner_term)
                        # Accumulate into sum_11[s, u_out, v_out].  If
                        # max_11 was -inf, every contributor here is -inf
                        # too (max would be at least as large), so sum
                        # stays 0 and the finalize sets mass to -inf.
                        if m11_finite:
                            sum_11[s, u_out, v_out] += np.exp(combined_cell - m11)
                        # Accumulate into sum_22[s, u_in, v_in].  The
                        # (u_in, v_in) cell varies per iteration, so we
                        # can't hoist this lookup further than the s loop.
                        m22 = max_22[s, u_in, v_in]
                        if np.isfinite(m22):
                            sum_22[s, u_in, v_in] += np.exp(combined_cell - m22)

    # ---- Finalize: mass = max + log(sum), with -inf fallback -----------
    mass_1_1 = np.empty((B, n_c, n_n), dtype=np.float64)
    mass_2_2 = np.empty((B, n_c, n_n), dtype=np.float64)
    for s in prange(B):
        for a in range(n_c):
            for b in range(n_n):
                m11 = max_11[s, a, b]
                if np.isfinite(m11):
                    mass_1_1[s, a, b] = m11 + np.log(sum_11[s, a, b])
                else:
                    mass_1_1[s, a, b] = -np.inf
                m22 = max_22[s, a, b]
                if np.isfinite(m22):
                    mass_2_2[s, a, b] = m22 + np.log(sum_22[s, a, b])
                else:
                    mass_2_2[s, a, b] = -np.inf

    return mass_1_1, mass_2_2


# %% --- CLASSES ---

class TransitionMesh:
    """
    A specialized container for transition probability meshes across different gap sizes.
    This structure allows efficient lookups of Forward and Backward transition 
    probabilities between genomic blocks separated by variable distances.

    Attributes:
        forward (dict): Maps gap_size (int) -> Forward Transition Dictionary.
                        Structure: { gap_size: { block_index: { ((curr_idx, curr_hap), (next_idx, next_hap)): prob } } }
        backward (dict): Maps gap_size (int) -> Backward Transition Dictionary.
                         Structure: { gap_size: { block_index: { ((curr_idx, curr_hap), (prev_idx, prev_hap)): prob } } }
    """
    def __init__(self, raw_gap_results=None):
        """
        Initializes the TransitionMesh.

        Args:
            raw_gap_results (dict, optional): A dictionary where keys are gap sizes and 
                                            values are [forward_dict, backward_dict] lists.
        """
        self.forward = {}
        self.backward = {}
        
        if raw_gap_results:
            for gap, probs_pair in raw_gap_results.items():
                self.forward[gap] = probs_pair[0]
                self.backward[gap] = probs_pair[1]

    def __getitem__(self, gap):
        """
        Retrieve the [Forward, Backward] transition dictionaries for a specific gap size.
        
        Args:
            gap (int): The distance (in number of blocks) between connected nodes.
            
        Returns:
            list: [forward_transition_dict, backward_transition_dict]
        """
        return [self.forward.get(gap), self.backward.get(gap)]
    
    def __contains__(self, gap):
        """Checks if a specific gap size has been computed in this mesh."""
        return gap in self.forward
    
    def keys(self):
        """Returns an iterator over the gap sizes available in the mesh."""
        return self.forward.keys()
    
    def items(self):
        """Yields (gap, [forward, backward]) tuples."""
        for gap in self.forward:
            yield gap, [self.forward[gap], self.backward[gap]]


class StandardBlockLikelihood:
    """
    Container for the likelihoods of ONE genomic block across ALL samples.
    Represents the emission probabilities P(Data | Genotype) for the HMM.
    
    Attributes:
        likelihood_tensor (np.ndarray): A tensor of shape (Num_Samples, Num_Haps, Num_Haps)
                                        containing log-likelihoods.
                                        Entry [s, i, j] is log(P(Sample_s | Hap_i, Hap_j)).
    """
    def __init__(self, likelihood_tensor):
        """
        Args:
            likelihood_tensor (np.ndarray): Tensor of log-likelihoods.
        """
        self.likelihood_tensor = likelihood_tensor
        
    def __len__(self):
        """Returns the number of samples."""
        return self.likelihood_tensor.shape[0]
    
    def __getitem__(self, sample_index):
        """
        Returns the (Num_Haps, Num_Haps) symmetric likelihood matrix for a specific sample.
        
        Args:
            sample_index (int): Index of the sample.
        """
        return self.likelihood_tensor[sample_index]
    
    def __repr__(self):
        return f"<StandardBlockLikelihood: {self.likelihood_tensor.shape[0]} samples, {self.likelihood_tensor.shape[1]} haps>"


class StandardBlockLikelihoods:
    """
    Container for the likelihoods of ALL genomic blocks in the dataset.
    This acts as the global 'Emission Probability' matrix for the downstream HMM.
    
    Attributes:
        blocks (list): A list of StandardBlockLikelihood objects, one per genomic block.
    """
    def __init__(self, blocks_list):
        """
        Args:
            blocks_list (list): A list of StandardBlockLikelihood objects.
        """
        # Validate input
        if blocks_list and not isinstance(blocks_list[0], StandardBlockLikelihood):
            self.blocks = [StandardBlockLikelihood(b) for b in blocks_list]
        else:
            self.blocks = blocks_list
            
    def __len__(self):
        """Returns the number of genomic blocks processed."""
        return len(self.blocks)
    
    def __getitem__(self, block_index):
        """Returns the StandardBlockLikelihood object for a specific block index."""
        return self.blocks[block_index]
    
    def __iter__(self):
        return iter(self.blocks)
    
    def __repr__(self):
        return f"<StandardBlockLikelihoods: covering {len(self.blocks)} blocks>"

# %% --- LIKELIHOOD GENERATION ---

def _worker_calculate_single_block_likelihood(args):
    """
    Internal worker function to calculate genotype likelihoods for a single block.
    
    It converts the per-site probabilistic genotypes (00, 01, 11) of the samples
    into diploid likelihoods for every pair of candidate haplotypes in the block.
    
    Uses Robust Categorical Likelihood (Mixture Model) to prevent over-penalization 
    of outliers at high read depth.
    
    Args:
        args (tuple): Contains:
            - samples_matrix (np.ndarray): (Samples x Sites x 3) probability matrix.
            - block_hap (BlockResult): The candidate haplotypes for this block.
            - params (dict): Configuration dictionary including robustness_epsilon.
    
    Returns:
        StandardBlockLikelihood: Object containing the unnormalized log-likelihood tensor.
    """
    samples_matrix, block_hap, params = args
    
    # Unpack params
    log_likelihood_base = params.get('log_likelihood_base', DEFAULT_LOG_BASE)
    epsilon = params.get('robustness_epsilon', DEFAULT_ROBUSTNESS_EPSILON)

    if len(samples_matrix) == 0:
        return StandardBlockLikelihood(np.array([]))

    num_samples, num_sites, _ = samples_matrix.shape

    hap_dict = block_hap.haplotypes
    # Ensure flags are boolean
    if block_hap.keep_flags is None:
        keep_flags = np.ones(num_sites, dtype=bool)
    else:
        keep_flags = block_hap.keep_flags.astype(bool)
        
    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    
    if num_haps == 0:
        return StandardBlockLikelihood(np.zeros((num_samples, 0, 0)))

    # --- 1. ROBUST TENSOR CREATION ---
    hap_list = [hap_dict[k] for k in hap_keys]
    
    if len(hap_list) > 0 and (hap_list[0].size == 0):
        haps_tensor = np.zeros((num_haps, 0, 2))
    else:
        haps_tensor = np.array(hap_list)
    
    # --- 2. MASKING ---
    samples_masked = samples_matrix[:, keep_flags, :] 
    haps_masked = haps_tensor[:, keep_flags, :]       
    
    num_active_sites = samples_masked.shape[1]
    
    if num_active_sites > 0:
        # --- 3. GENERATE DIPLOID COMBINATIONS ---
        # These are sample-independent: only depend on haplotype pairs.
        # Shape: (N_Haps, N_Haps, Sites) — small, computed once.
        h0 = haps_masked[:, :, 0]
        h1 = haps_masked[:, :, 1]
        
        c00 = h0[:, None, :] * h0[None, :, :]
        c11 = h1[:, None, :] * h1[None, :, :]
        c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
        del h0, h1, haps_masked
        
        # Flatten haplotypes for broadcasting
        c00_flat = c00.reshape(-1, num_active_sites); del c00
        c01_flat = c01.reshape(-1, num_active_sites); del c01
        c11_flat = c11.reshape(-1, num_active_sites); del c11
        
        # --- 4-5. CHUNKED EMISSION SCORING ---
        # Process samples in chunks to bound peak memory at
        # O(chunk × K² × sites) instead of O(n_samples × K² × sites).
        # Each sample's likelihood is independent — zero overhead from chunking.
        uniform_prob = 1.0 / 3.0
        min_prob = 1e-300
        _log_uniform = math.log(1.0 / 3.0)
        
        final_tensor = np.empty((num_samples, num_haps, num_haps), dtype=np.float64)
        
        for s_start in range(0, num_samples, SAMPLE_CHUNK_SIZE):
            s_end = min(s_start + SAMPLE_CHUNK_SIZE, num_samples)
            chunk_samples = samples_masked[s_start:s_end]
            chunk_n = s_end - s_start
            
            # A. Calculate "Pure" Model Likelihood for this sample chunk
            term_0 = chunk_samples[:, np.newaxis, :, 0] * c00_flat[np.newaxis, :, :]
            term_1 = chunk_samples[:, np.newaxis, :, 1] * c01_flat[np.newaxis, :, :]
            term_2 = chunk_samples[:, np.newaxis, :, 2] * c11_flat[np.newaxis, :, :]
            
            model_probs = term_0 + term_1 + term_2
            del term_0, term_1, term_2
            
            # B. Apply Robust Mixture
            final_probs = (model_probs * (1.0 - epsilon)) + (epsilon * uniform_prob)
            del model_probs
            
            # --- 5. LOG LIKELIHOOD ---
            final_probs[final_probs < min_prob] = min_prob
            ll_per_site = np.log(final_probs)
            del final_probs
            
            # --- APPLY BURST/AFFINE LOGIC UPGRADE ---
            # 1. Apply Hard Floor of -2.0 per site (prevents single-site overkill)
            ll_per_site = np.maximum(ll_per_site, -2.0)
            
            # 2. Reshape for Kernel: (ChunkSamples, Haps, Haps, Sites)
            ll_4d = ll_per_site.reshape(chunk_n, num_haps, num_haps, num_active_sites)
            del ll_per_site
            
            # 3. Burst Aware Summation
            final_tensor[s_start:s_end] = calculate_burst_score_vectorized(
                ll_4d, 
                gap_open_penalty=-10.0,
                gap_extend_penalty=0.0, 
                uniform_log_prob=_log_uniform
            )
            del ll_4d
        
        del c00_flat, c01_flat, c11_flat, samples_masked
        
    else:
        final_tensor = np.zeros((num_samples, num_haps, num_haps))

    # --- 6. FORMAT OUTPUT ---
    # Reshape to (N_Samples, N_Haps, N_Haps)
    # The burst kernel returns exactly this shape, so just wrapping it.
    
    return StandardBlockLikelihood(final_tensor)

def generate_all_block_likelihoods(
    sample_probs_matrix,
    global_site_locations,
    haplotype_data,
    num_processes=16,
    log_likelihood_base=math.e,
    robustness_epsilon=DEFAULT_ROBUSTNESS_EPSILON):
    """
    Calculates diploid genotype log-likelihoods for all blocks against all samples.
    This generates the "Emission Matrix" for the HMM.
    
    Updated to support both contiguous blocks and sparse (proxy) blocks by using 
    exact index mapping rather than slicing.
    
    Args:
        sample_probs_matrix (np.ndarray): (N_Samples x Total_Sites x 3) probability matrix.
        global_site_locations (np.ndarray): Array of genomic positions.
        haplotype_data (list or BlockResults): List of BlockResult objects or a single object.
        num_processes (int): Number of parallel processes to use.
        log_likelihood_base (float): Base for the log calculation.
        robustness_epsilon (float): The mixture weight for the uniform error model.

    Returns:
        StandardBlockLikelihoods: A container with symmetric likelihood matrices for all blocks.
        Or a single StandardBlockLikelihood if a single block was passed.
    """
    
    is_single_block = False
    
    if hasattr(haplotype_data, 'positions') and hasattr(haplotype_data, 'haplotypes'):
        blocks_to_process = [haplotype_data]
        is_single_block = True
    else:
        blocks_to_process = haplotype_data
        
    params = {
        'log_likelihood_base': log_likelihood_base,
        'robustness_epsilon': robustness_epsilon
    }

    # Sequential: process one block at a time, free each before the next.
    # num_processes is kept for signature compatibility; production never
    # exercises the parallel pool (every caller passes num_processes=1 or
    # leaves the default and provides a pre-computed likelihoods tensor
    # via the shared-data path).
    del num_processes
    results = []
    for block in blocks_to_process:
        if not hasattr(block, 'positions'):
            raise ValueError(f"Encountered invalid block object in list. Type: {type(block)}")
        indices = np.searchsorted(global_site_locations, block.positions)
        block_samples = sample_probs_matrix[:, indices, :]
        result = _worker_calculate_single_block_likelihood((block_samples, block, params))
        del block_samples
        _malloc_trim()
        results.append(result)

    if is_single_block:
        return results[0]
        
    return StandardBlockLikelihoods(results)
# %% --- EM HELPERS ---

def initial_transition_probabilities(haps_data, space_gap=1):
    """
    Creates a dictionary of initial transition probabilities assuming a Uniform Prior.
    Connects every haplotype in Block N to every haplotype in Block N + space_gap.
    
    Args:
        haps_data (list): List of BlockResult objects.
        space_gap (int): The distance (stride) between blocks to link.

    Returns:
        list: [forward_dict, backward_dict] containing uniform probabilities.
    """
    transition_dict_forward = {}
    transition_dict_reverse = {}
    
    # Forward Pass initialization
    for i in range(0,len(haps_data)-space_gap):
        transition_dict_forward[i] = {}
        
        these_haps = haps_data[i].haplotypes
        next_haps = haps_data[i+space_gap].haplotypes
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            for second_idx in next_haps.keys():
                second_hap_name = (i+space_gap,second_idx)
                transition_dict_forward[i][(first_hap_name,second_hap_name)] = 1
                
    # Backward Pass initialization
    for i in range(len(haps_data)-1,space_gap-1,-1):
        transition_dict_reverse[i] = {}
        
        these_haps = haps_data[i].haplotypes
        next_haps = haps_data[i-space_gap].haplotypes
        
        for first_idx in these_haps.keys():
            first_hap_name = (i,first_idx)
            for second_idx in next_haps.keys():
                second_hap_name = (i-space_gap,second_idx)
                transition_dict_reverse[i][(first_hap_name,second_hap_name)] = 1
    
    # Normalize Forward
    scaled_dict_forward = {}
    for idx in transition_dict_forward.keys():
        scaled_dict_forward[idx] = {}
        start_dict = {}
        for s in transition_dict_forward[idx].keys():
            start_dict[s[0]] = start_dict.get(s[0], 0) + transition_dict_forward[idx][s]
        
        for s in transition_dict_forward[idx].keys():
            scaled_dict_forward[idx][s] = transition_dict_forward[idx][s]/start_dict[s[0]]
        
    # Normalize Backward
    scaled_dict_reverse = {}
    for idx in transition_dict_reverse.keys():
        scaled_dict_reverse[idx] = {}
        start_dict = {}
        for s in transition_dict_reverse[idx].keys():
            start_dict[s[0]] = start_dict.get(s[0], 0) + transition_dict_reverse[idx][s]
        
        for s in transition_dict_reverse[idx].keys():
            scaled_dict_reverse[idx][s] = transition_dict_reverse[idx][s]/start_dict[s[0]]
        
    return [scaled_dict_forward, scaled_dict_reverse]

# %% --- EM FORWARD/BACKWARD ---

def get_full_probs_forward(sample_data, sample_sites, haps_data,
                           bidirectional_transition_probs,
                           sample_block_likelihoods=None,
                           space_gap=1,
                           hap_keys_cache=None,
                           T_per_block_cache=None,
                           dense_matrices_out=None):
    """
    Calculates the Forward Variables (Alpha) for the HMM for a SINGLE sample.
    Computes P(State_t = i, Data_1:t) recursively using log-space matrix multiplication.
    Uses FULL directed state space (no symmetry collapsing).
    
    Args:
        sample_data (np.ndarray): (Sites x 3) probability array for one sample.
        sample_sites (np.ndarray): Site coordinates.
        haps_data (list): List of BlockResult objects.
        bidirectional_transition_probs (list): [forward_dict, backward_dict].
        sample_block_likelihoods (list, optional): Pre-computed emission probabilities for this sample.
        space_gap (int): The stride of the HMM chain.
        hap_keys_cache (list, optional): Pre-computed sorted hap keys per block.
        T_per_block_cache (list, optional): Pre-computed dense log-transition
            matrices, one entry per block.  When provided, each entry at
            index `earlier_block` is the (n_prev, n_haps) log-T matrix used
            in the recursion step for block `earlier_block -> earlier_block
            + space_gap`.  When None, the matrices are built inline from
            the transition_probs_dict (legacy path).  Pre-building these
            once at the caller and passing them in saves
            (num_samples - 1) * num_blocks repeated dict-to-dense
            conversions; in EM workflows where forward+backward are
            invoked once per sample per iteration, this is substantial.
        dense_matrices_out (dict, optional): If provided, this dict is
            populated in-place with `dense_matrices_out[block_idx] =
            current_matrix` for every block, sharing memory with the
            internal computation.  Allows downstream consumers (e.g.
            the M-step in `get_updated_transition_probabilities_unified`)
            to bypass the dict-encoded `likelihood_numbers` output and
            read the dense matrices directly, avoiding O(K^2) dict
            lookups per sample per block in the F/B tensor construction.
            The matrices are aliased into the dict — caller must not
            mutate them.  Default None disables this.
        
    Returns:
        dict: likelihood_numbers mapping block_index -> { (HapA, HapB): log_prob }
    """
    
    if sample_block_likelihoods is None:
        full_res = generate_all_block_likelihoods(
            sample_data[np.newaxis, :, :], sample_sites, haps_data, num_processes=1
        )
        sample_block_likelihoods = [b[0] for b in full_res]

    # Use the pre-computed sorted hap-key cache if provided; otherwise
    # build it on the fly.  Sorting once and reusing avoids O(num_blocks
    # × n_haps log n_haps) re-sorting per EM iteration.
    if hap_keys_cache is None:
        hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in haps_data]

    transition_probs_dict = bidirectional_transition_probs[0]
    
    likelihood_numbers = {} 
    shadow_cache = {} 
    
    for i in range(len(haps_data)):
        
        # 1. Load Emission Probabilities (Matrix)
        E = sample_block_likelihoods[i] # (N_Haps, N_Haps)

        hap_keys = hap_keys_cache[i]
        n_haps = len(hap_keys)
        
        if i < space_gap:
            # Initialization Step: Just Emissions
            # In directed space, we don't need correction factors for hets.
            current_matrix = E
            
        else:
            # Recursion Step: Alpha_t = (Alpha_t-1 @ T) * E
            earlier_block = i - space_gap
            prev_matrix = shadow_cache[earlier_block]['matrix']
            prev_keys = shadow_cache[earlier_block]['keys']
            n_prev = len(prev_keys)
            
            # Construct Transition Matrix T (Sparse to Dense).  Use the
            # cached T matrix if available (pre-built once at the caller
            # level), otherwise rebuild from the dict.  The dict-lookup
            # path can't be numba-accelerated (Python dict keys), but
            # pre-building avoids redoing it per-sample.
            if T_per_block_cache is not None:
                T = T_per_block_cache[earlier_block]
            else:
                T = analysis_utils._build_haploid_log_T_from_dict(
                    transition_probs_dict[earlier_block],
                    prev_keys, hap_keys, earlier_block, i)
            
            # Z = Alpha_prev @ T
            Z = analysis_utils.log_matmul(prev_matrix, T)
            # Pred = T.T @ Z
            pred_matrix = analysis_utils.log_matmul(T.T, Z)
            
            # Combine with Emissions
            current_matrix = pred_matrix + E

        shadow_cache[i] = {'matrix': current_matrix, 'keys': hap_keys}

        # If a dense-matrix output dict was provided, alias the matrix
        # into it for downstream zero-copy consumption.
        if dense_matrices_out is not None:
            dense_matrices_out[i] = current_matrix

        # Output Results (Full Grid)
        result_dict = {}
        for r in range(n_haps):
            for c in range(n_haps): # Iterate full grid
                key = ((i, hap_keys[r]), (i, hap_keys[c]))
                result_dict[key] = current_matrix[r, c]

        likelihood_numbers[i] = result_dict
        
    return likelihood_numbers

def get_full_probs_backward(sample_data, sample_sites, haps_data,
                           bidirectional_transition_probs,
                           sample_block_likelihoods=None,
                           space_gap=1,
                           hap_keys_cache=None,
                           T_per_block_cache=None,
                           dense_matrices_out=None):
    """
    Calculates the Backward Variables (Beta) for the HMM for a SINGLE sample.
    Computes P(Data_t+1:T | State_t = i) recursively.
    Uses FULL directed state space.
    
    Args:
        sample_data (np.ndarray): (Sites x 3) probability array for one sample.
        sample_sites (np.ndarray): Site coordinates.
        haps_data (list): List of BlockResult objects.
        bidirectional_transition_probs (list): [forward_dict, backward_dict].
        sample_block_likelihoods (list, optional): Pre-computed emission probabilities.
        space_gap (int): The stride of the HMM chain.
        hap_keys_cache (list, optional): Pre-computed sorted hap keys per block.
        T_per_block_cache (list, optional): Pre-computed dense log-transition
            matrices indexed by `future_block`, each of shape
            (n_haps_curr, n_haps_future).  When None, the matrices are
            built inline from the transition_probs_dict (legacy path).
        dense_matrices_out (dict, optional): If provided, populated in-
            place with the per-block dense matrices.  Same semantics as
            in get_full_probs_forward.

    Returns:
        dict: likelihood_numbers mapping block_index -> { (HapA, HapB): log_prob }
    """
    
    if sample_block_likelihoods is None:
        full_res = generate_all_block_likelihoods(
            sample_data[np.newaxis, :, :], sample_sites, haps_data, num_processes=1
        )
        sample_block_likelihoods = [b[0] for b in full_res]

    # Use the pre-computed sorted hap-key cache if provided; otherwise
    # build it on the fly.  See get_full_probs_forward for rationale.
    if hap_keys_cache is None:
        hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in haps_data]

    transition_probs_dict = bidirectional_transition_probs[1]
    
    likelihood_numbers = {} 
    shadow_cache = {} 
    
    for i in range(len(haps_data)-1, -1, -1):
        
        # 1. Load Emission Probabilities (Matrix)
        E = sample_block_likelihoods[i]

        hap_keys = hap_keys_cache[i]
        n_haps = len(hap_keys)
        
        if i >= len(haps_data) - space_gap:
            # Initialization: Beta_T = 1 (log 0)
            current_matrix = E
            
        else:
            # Recursion: Beta_t = T @ (Beta_t+1 * E_t+1)
            future_block = i + space_gap
            future_matrix = shadow_cache[future_block]['matrix']
            future_keys = shadow_cache[future_block]['keys']
            n_fut = len(future_keys)
            
            # Construct Transition Matrix T (Sparse to Dense).  Use the
            # cached T matrix if available (pre-built once at the caller
            # level), otherwise rebuild from the dict.  Matrix has shape
            # (n_haps, n_fut): T[r, c] is the log-prob of going from
            # current-block hap_keys[r] to future-block future_keys[c].
            if T_per_block_cache is not None:
                T = T_per_block_cache[future_block]
            else:
                # Helper signature: (trans_dict, prev_keys, curr_keys, prev_idx, curr_idx)
                # builds matrix M[u_i, x_i] keyed by
                # ((prev_idx, prev_keys[u_i]), (curr_idx, curr_keys[x_i])).
                # Here the original lookup is
                # ((future_block, future_keys[c]), (i, hap_keys[r])).
                # So prev=future_block, curr=i.  Helper returns
                # shape (n_fut, n_haps); we want (n_haps, n_fut), so transpose.
                T_fwd_form = analysis_utils._build_haploid_log_T_from_dict(
                    transition_probs_dict[future_block],
                    future_keys, hap_keys, future_block, i)
                T = T_fwd_form.T
            
            Z = analysis_utils.log_matmul(future_matrix, T.T)
            pred_matrix = analysis_utils.log_matmul(T, Z)
            current_matrix = pred_matrix + E

        shadow_cache[i] = {'matrix': current_matrix, 'keys': hap_keys}

        if dense_matrices_out is not None:
            dense_matrices_out[i] = current_matrix
        
        # Output Results (Full Grid)
        result_dict = {}
        for r in range(n_haps):
            for c in range(n_haps): # Iterate full grid
                key = ((i, hap_keys[r]), (i, hap_keys[c]))
                result_dict[key] = current_matrix[r, c]

        likelihood_numbers[i] = result_dict

    return likelihood_numbers


# %% --- BATCHED (ALL-SAMPLES) FORWARD/BACKWARD PASSES ---
#
# These functions are the all-samples-at-once analogues of
# get_full_probs_forward and get_full_probs_backward.  They produce the
# same per-sample dense matrices as those functions but run a SINGLE
# numba kernel per block-step instead of one kernel call per sample, by
# operating on a leading (num_samples, K, K) tensor throughout.
#
# Why this exists:
#   The per-sample functions call analysis_utils.log_matmul twice per
#   (sample, block) — at K=6 the inner matmul is a (6,6)×(6,6) operation
#   whose Python/numba dispatch overhead per call (~3-4 µs) dominates the
#   actual arithmetic (~216 mul + 36 log).  At chr1 L1 production scale
#   (320 samples × ~10 blocks/batch × ~80 gaps/batch × 2 directions ×
#   2 matmuls/block-step) that's ~2.5M dispatch overheads per L1 batch,
#   totalling ~10 s of pure overhead per batch.
#
#   The batched form does the same scalar math on the same scalars in
#   the same order — per-slice bit-equivalent to the per-sample form —
#   but pays the dispatch overhead once per block-step instead of once
#   per (sample, block-step).  At S=320 that's a 320× reduction.
#
# Bit-equivalence: see analysis_utils._log_matmul_3d_2d_kernel and
# _log_matmul_2d_3d_kernel docstrings.  Each per-sample slice of the
# batched output is byte-identical to the corresponding per-sample
# scalar call.  Verified against the per-sample path on production-
# scale shapes.

def _forward_pass_batched(full_blocks_likelihoods, haps_data, T_fwd_cache,
                          hap_keys_cache, space_gap):
    """All-samples forward pass.

    Returns a list `F_dense` of length num_blocks where F_dense[b] is a
    (num_samples, n_haps_b, n_haps_b) float64 tensor giving the forward
    variable α at block b for every sample.  Per-sample slice
    F_dense[b][s] is bit-equivalent to the (n_haps_b, n_haps_b) matrix
    that get_full_probs_forward would have produced for sample s at
    block b under the same inputs.

    Args:
        full_blocks_likelihoods (StandardBlockLikelihoods): per-block
            emission likelihood tensors; full_blocks_likelihoods[b]
            .likelihood_tensor already has shape (num_samples, K_b, K_b),
            so we use the batched form zero-copy.
        haps_data (list): per-block BlockResult objects.  Only used for
            len() in the outer iteration; haplotype keys come from
            hap_keys_cache.
        T_fwd_cache (list): pre-built (n_prev, n_curr) log-T matrices,
            same structure as in the per-sample path.  Entry at
            `earlier_block` is consumed at recursion step earlier_block
            -> earlier_block + space_gap.
        hap_keys_cache (list): sorted hap keys per block (used to derive
            K_b sizes for assertion; the values themselves are not
            needed in the dense path).
        space_gap (int): HMM stride.
    """
    num_blocks = len(haps_data)
    F_dense = [None] * num_blocks

    for i in range(num_blocks):
        # Emission tensor for block i: (num_samples, K_i, K_i).
        E_batch = full_blocks_likelihoods[i].likelihood_tensor

        if i < space_gap:
            # Initialisation: α_1 = emissions.  Match the per-sample
            # function which returns E as-is for the boundary blocks
            # (no diploid correction factor in directed state space).
            F_dense[i] = np.ascontiguousarray(E_batch, dtype=np.float64)
        else:
            # Recursion: α_t = (α_{t-gap} @ T) then T.T @ that, then + E.
            earlier_block = i - space_gap
            prev_batch = F_dense[earlier_block]   # (S, n_prev, n_prev)
            T = T_fwd_cache[earlier_block]        # (n_prev, n_curr) — shared across samples
            T_c = np.ascontiguousarray(T, dtype=np.float64)

            # Z[s] = prev[s] @ T  ->  (S, n_prev, n_curr)
            Z_batch = analysis_utils._log_matmul_3d_2d_kernel(
                np.ascontiguousarray(prev_batch, dtype=np.float64), T_c)

            # pred[s] = T.T @ Z[s]  ->  (S, n_curr, n_curr).  The per-
            # sample function does log_matmul(T.T, Z); we use the
            # 2D × 3D kernel with A = T.T (must be contiguous) and
            # B_batch = Z.
            T_T_c = np.ascontiguousarray(T_c.T)
            pred_batch = analysis_utils._log_matmul_2d_3d_kernel(T_T_c, Z_batch)

            # Combine with emissions: shapes match (S, n_curr, n_curr).
            F_dense[i] = pred_batch + np.ascontiguousarray(E_batch, dtype=np.float64)

    return F_dense


def _backward_pass_batched(full_blocks_likelihoods, haps_data, T_bwd_cache,
                           hap_keys_cache, space_gap):
    """All-samples backward pass.

    Returns a list `B_dense` of length num_blocks where B_dense[b] is a
    (num_samples, n_haps_b, n_haps_b) float64 tensor giving the backward
    variable β at block b for every sample.  Per-sample slice
    B_dense[b][s] is bit-equivalent to the (n_haps_b, n_haps_b) matrix
    that get_full_probs_backward would have produced for sample s at
    block b.

    Args:
        T_bwd_cache (list): pre-built (n_curr, n_fut) log-T matrices —
            same `T_bwd_cache[future_block]` form built in
            get_updated_transition_probabilities_unified, where the
            entry is the .T of the forward-form helper output.
    """
    num_blocks = len(haps_data)
    B_dense = [None] * num_blocks

    for i in range(num_blocks - 1, -1, -1):
        E_batch = full_blocks_likelihoods[i].likelihood_tensor

        if i >= num_blocks - space_gap:
            B_dense[i] = np.ascontiguousarray(E_batch, dtype=np.float64)
        else:
            future_block = i + space_gap
            future_batch = B_dense[future_block]   # (S, n_fut, n_fut)
            T = T_bwd_cache[future_block]          # (n_curr, n_fut) — shared

            # Per-sample form: Z = log_matmul(future_matrix, T.T); pred = log_matmul(T, Z)
            # Equivalent batched form:
            #   Z[s] = future_matrix[s] @ T.T          ->  (S, n_fut, n_curr)
            #   pred[s] = T @ Z[s]                      ->  (S, n_curr, n_curr)
            T_c = np.ascontiguousarray(T, dtype=np.float64)
            T_T_c = np.ascontiguousarray(T_c.T)

            Z_batch = analysis_utils._log_matmul_3d_2d_kernel(
                np.ascontiguousarray(future_batch, dtype=np.float64), T_T_c)
            pred_batch = analysis_utils._log_matmul_2d_3d_kernel(T_c, Z_batch)

            B_dense[i] = pred_batch + np.ascontiguousarray(E_batch, dtype=np.float64)

    return B_dense


# %% --- UNIFIED UPDATE FUNCTION ---

def get_updated_transition_probabilities_unified(
        full_samples_data,
        sample_sites,
        haps_data,
        current_transition_probs,
        full_blocks_likelihoods,
        space_gap=1,
        minimum_transition_log_likelihood=-10,
        BATCH_SIZE=100,
        use_standard_baum_welch=True,
        uniform_prior=None,
        hap_keys_cache=None,
        all_block_likelihoods_by_sample=None): 
    """
    Performs the Expectation-Maximization (EM) update step (Baum-Welch).

    1. E-Step: Runs Forward and Backward algorithms for all samples to compute 
       the probability of being in state (u,v) at time t given the data.
    2. M-Step: Updates the transition probabilities T_ij to maximize the likelihood.
       Utilizes vectorized batch processing to handle the summation over samples efficiently.
       Includes Robust M-Step logic to handle diploid phase ambiguity.
    
    Args:
        full_samples_data (list): List of sample data arrays.
        sample_sites (np.ndarray): Genomic locations.
        haps_data (list): List of BlockResult objects.
        current_transition_probs (list): Current estimates [fwd, bwd].
        full_blocks_likelihoods (StandardBlockLikelihoods): Pre-computed emissions.
        space_gap (int): HMM stride.
        minimum_transition_log_likelihood (float): Floor for probabilities.
        BATCH_SIZE (int): Number of samples to process in a vectorized chunk.
        use_standard_baum_welch (bool): If True, applies standard HMM logic.
        uniform_prior (list, optional): Pre-computed uniform prior [fwd, bwd].
        hap_keys_cache (list, optional): Pre-computed sorted hap keys per block.
        all_block_likelihoods_by_sample (list, optional): Pre-restructured emissions.

    Returns:
        tuple: ([new_fwd, new_bwd], total_data_log_likelihood)
    """

    # Use the pre-computed uniform prior if provided; otherwise build
    # it on the fly.  Computing this once per EM run instead of once per
    # iteration is a free win since the prior never changes.
    if uniform_prior is None:
        prior_a_posteriori = initial_transition_probabilities(haps_data, space_gap=space_gap)
    else:
        prior_a_posteriori = uniform_prior

    # Use the pre-computed sorted hap-key cache if provided; otherwise
    # build it on the fly.
    if hap_keys_cache is None:
        hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in haps_data]

    full_samples_likelihoods = full_samples_data
    num_samples = len(full_samples_likelihoods)
    num_blocks = len(full_blocks_likelihoods)
    
    # Per-sample restructure of emissions
    # (all_block_likelihoods_by_sample) is no longer needed: the batched
    # F/B passes consume full_blocks_likelihoods[b].likelihood_tensor
    # directly (already shape (num_samples, K_b, K_b)).  We keep the
    # variable as None for any downstream code that still inspects the
    # parameter — every project caller passes None already.
    if all_block_likelihoods_by_sample is None:
        pass  # batched path doesn't need it

    # PRE-BUILD dense haploid log-T matrices once per EM iteration,
    # shared across all num_samples forward+backward invocations.
    # Without this, each get_full_probs_{forward,backward} call rebuilds
    # the same matrices from the dict — that's
    # num_samples * num_blocks * 2 redundant dict-to-dense conversions.
    # The dict-to-dense step itself can't be numba-accelerated (Python
    # dict keys), but hoisting it out of the per-sample loop saves the
    # repeated work entirely.
    #
    # Forward T cache: T_fwd_cache[earlier_block] is the (n_prev, n_haps)
    # matrix used by get_full_probs_forward's recursion step.  Entries
    # where earlier_block + space_gap is out of range are unused; we
    # still build a fixed-length list keyed by block index for easy
    # indexing.
    T_fwd_cache = [None] * num_blocks
    fwd_trans = current_transition_probs[0]
    for earlier_block in range(num_blocks - space_gap):
        if earlier_block in fwd_trans:
            curr_block = earlier_block + space_gap
            T_fwd_cache[earlier_block] = (
                analysis_utils._build_haploid_log_T_from_dict(
                    fwd_trans[earlier_block],
                    hap_keys_cache[earlier_block],
                    hap_keys_cache[curr_block],
                    earlier_block, curr_block))

    # Backward T cache: T_bwd_cache[future_block] is the (n_haps_curr,
    # n_haps_future) matrix used by get_full_probs_backward at
    # current_block = future_block - space_gap.  Built so that
    # T_bwd_cache[future_block][r, c] is the log-prob of going from
    # current-block hap_keys_cache[future_block - space_gap][r] to
    # future-block hap_keys_cache[future_block][c] — equivalent to
    # transposing the helper's output (which has the future-block as
    # leading axis).
    T_bwd_cache = [None] * num_blocks
    bwd_trans = current_transition_probs[1]
    for future_block in range(space_gap, num_blocks):
        curr_block = future_block - space_gap
        if future_block in bwd_trans:
            T_fwd_form = analysis_utils._build_haploid_log_T_from_dict(
                bwd_trans[future_block],
                hap_keys_cache[future_block],
                hap_keys_cache[curr_block],
                future_block, curr_block)
            T_bwd_cache[future_block] = np.ascontiguousarray(T_fwd_form.T)

    # 1. E-Step: all-samples Forward and Backward passes via batched
    # log-matmul kernels.  Per-sample slice forward_dense[b][s] is
    # bit-equivalent to the (K_b, K_b) dense matrix that the per-sample
    # get_full_probs_forward would have produced for sample s at block
    # b.  See _forward_pass_batched and _backward_pass_batched.
    #
    # The dict-form outputs that the per-sample functions produced
    # (likelihood_numbers, samples_probs) are not built here: they were
    # only consumed by (a) the total_data_log_likelihood computation
    # (which we do directly from the dense forward tensor below) and
    # (b) the dead `samples_probs = list(zip(...))` assignment that no
    # downstream code reads.  _run_batched_pass uses the dense tensors
    # only, accessed via forward_dense / backward_dense per block.
    forward_dense = _forward_pass_batched(
        full_blocks_likelihoods, haps_data, T_fwd_cache,
        hap_keys_cache, space_gap,
    )
    backward_dense = _backward_pass_batched(
        full_blocks_likelihoods, haps_data, T_bwd_cache,
        hap_keys_cache, space_gap,
    )

    # Total Data Log-Likelihood: sum over samples of LSE over all
    # (K, K) cells in the final-block forward matrix.  Equivalent to
    # the per-sample form
    #   sum_s lse_scalar(forward_nums[s][last_block_idx].values())
    # since forward_nums[s][last_block_idx][(.,.)] = forward_dense
    # [last_block_idx][s][r, c] by construction (dict insertion order
    # is row-major over (r, c), matching reshape(-1)).
    #
    # We use the same scalar lse_scalar kernel and sequential float
    # accumulation as the old code to preserve bit-exact reduction
    # order; the cost is negligible (S=320 calls per EM iteration).
    last_block_idx = num_blocks - 1
    last_F = forward_dense[last_block_idx]              # (S, K_last, K_last)
    total_data_log_likelihood = 0.0
    for s in range(num_samples):
        final_states_log_probs = last_F[s].reshape(-1)
        if final_states_log_probs.size:
            total_data_log_likelihood += analysis_utils.lse_scalar(final_states_log_probs)
    
    
    # 2. M-Step: Vectorized Update (Baum-Welch ξ calculation)
    def _run_batched_pass(indices, is_forward):
        new_transition_probs = {}
        dir_idx = 0 if is_forward else 1
        
        for i in indices:
            next_bundle = i + space_gap if is_forward else i - space_gap
            
            hap_keys_current = hap_keys_cache[i]
            hap_keys_next    = hap_keys_cache[next_bundle]
            n_curr = len(hap_keys_current)
            n_next = len(hap_keys_next)
            
            # Load priors via the shared dict-to-dense helper.  The dict
            # lookups themselves can't be numba-accelerated (Python dict
            # keys are tuple-of-tuples), but using the centralised helper
            # eliminates a code-duplicated double loop here.
            T_matrix = analysis_utils._build_haploid_log_T_from_dict(
                current_transition_probs[dir_idx][i],
                hap_keys_current, hap_keys_next, i, next_bundle)
            P_matrix = analysis_utils._build_haploid_log_T_from_dict(
                prior_a_posteriori[dir_idx][i],
                hap_keys_current, hap_keys_next, i, next_bundle)

            # Accumulate Forward/Backward probabilities across samples.
            #
            # forward_dense / backward_dense are the per-block batched
            # tensors produced by _forward_pass_batched / _backward_pass_batched.
            # forward_dense[b] has shape (num_samples, K_b, K_b) and is
            # bit-equivalent (per-sample slice) to what the per-sample
            # get_full_probs_forward would have produced for block b
            # under the same inputs.
            #
            # Since these tensors already encode (sample, out, in) in the
            # same axis order F_tensor / B_tensor expect, we can alias
            # them directly — no per-sample copy loop, no extra
            # allocation.  The downstream batched kernel does
            # np.ascontiguousarray on slices before consuming them, so
            # aliasing is safe (kernel is read-only on these inputs).
            if is_forward:
                F_tensor = forward_dense[i]              # (num_samples, n_curr, n_curr)
                B_tensor = backward_dense[next_bundle]   # (num_samples, n_next, n_next)
            else:
                F_tensor = backward_dense[i]             # (num_samples, n_curr, n_curr)
                B_tensor = forward_dense[next_bundle]    # (num_samples, n_next, n_next)
                
            batch_results = []
            
            # NOTE: The original code allocated a (1, n_curr, n_curr,
            # n_next, n_next) hom-hom mask and a (1, n_curr, n_curr,
            # n_next, n_next) T_partner_corrected broadcast intermediate
            # OUTSIDE the per-batch loop.  The fused kernel below handles
            # the hom-hom correction inline, so neither array is needed.
            
            # Process batches.  The fused kernel computes mass_1_1 and
            # mass_2_2 directly from F_batch, B_batch, T_matrix without
            # materializing the (B, n_c, n_c, n_n, n_n) combined array.
            # See _batched_baum_welch_mass_kernel's docstring for the
            # algebraic decomposition.
            #
            # T_matrix and use_standard_baum_welch are invariant across
            # the BATCH_SIZE loop iterations; hoist their normalisation
            # out so we don't redo them per batch-chunk.
            T_matrix_c = np.ascontiguousarray(T_matrix, dtype=np.float64)
            use_standard_bw_b = bool(use_standard_baum_welch)
            for start_idx in range(0, num_samples, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                
                F_batch = np.ascontiguousarray(F_tensor[start_idx:end_idx])
                B_batch = np.ascontiguousarray(B_tensor[start_idx:end_idx])

                mass_1_1, mass_2_2 = _batched_baum_welch_mass_kernel(
                    F_batch, B_batch, T_matrix_c, use_standard_bw_b)

                # sample_lik_batch[s, a, b] = logaddexp(mass_1_1[s, a, b],
                #                                       mass_2_2[s, a, b])
                # Equivalent to the original's
                #   stacked_evidence = np.stack([mass_1_1, mass_2_2])
                #   sample_lik_batch = lse_axis0(stacked_evidence)
                # but skipping the stack.  np.logaddexp is elementwise and
                # vectorized over numpy arrays — no per-element Python cost.
                sample_lik_batch = np.logaddexp(mass_1_1, mass_2_2)

                # Normalize per sample.  Original used
                #   total_per_sample = logsumexp(sample_lik_batch,
                #                                  axis=(1, 2), keepdims=True)
                # which is logsumexp over the flattened (a, b) axes per
                # sample.  We flatten axes (1, 2) to compute it via the
                # 2D axis-last helper, then reshape back to (B, 1, 1) for
                # broadcasting compatibility with the subtraction.
                B_size = sample_lik_batch.shape[0]
                flat_lik = sample_lik_batch.reshape(B_size, -1)
                total_per_sample_flat = analysis_utils.lse_axis_last(flat_lik)
                total_per_sample = total_per_sample_flat.reshape(B_size, 1, 1)

                batch_aggregated = analysis_utils.lse_axis0(
                    sample_lik_batch - total_per_sample)
                batch_results.append(batch_aggregated)
            
            if len(batch_results) > 0:
                final_aggregated = analysis_utils.lse_axis0(batch_results)
            else:
                final_aggregated = np.full((n_curr, n_next), -np.inf)
            
            posterior_with_prior = final_aggregated + P_matrix
            
            row_sums = analysis_utils.lse_axis_last(posterior_with_prior, keepdims=True)
            log_probs = posterior_with_prior - row_sums
            log_probs_clipped = np.maximum(log_probs, minimum_transition_log_likelihood)
            
            probs_nonnorm = np.exp(log_probs_clipped)
            row_sums_final = np.sum(probs_nonnorm, axis=1, keepdims=True)
            row_sums_final[row_sums_final == 0] = 1.0 
            final_probs_matrix = probs_nonnorm / row_sums_final
            
            block_dict = {}
            for u_idx, u in enumerate(hap_keys_current):
                for v_idx, v in enumerate(hap_keys_next):
                    key = ((i, u), (next_bundle, v))
                    block_dict[key] = final_probs_matrix[u_idx, v_idx]
            
            new_transition_probs[i] = block_dict
            
        return new_transition_probs
    
    forward_indices = range(len(haps_data) - space_gap)
    new_transition_probs_forward = _run_batched_pass(forward_indices, is_forward=True)
    
    backward_indices = range(len(haps_data) - 1, space_gap - 1, -1)
    new_transition_probs_backwards = _run_batched_pass(backward_indices, is_forward=False)
    
    return ([new_transition_probs_forward, new_transition_probs_backwards], total_data_log_likelihood)

def calculate_hap_transition_probabilities(full_samples_data,
            sample_sites,
            haps_data,
            full_blocks_likelihoods=None,
            max_num_iterations=10,
            space_gap=1,
            min_cutoff_change=0.001,
            ll_improvement_cutoff=1e-4,
            learning_rate=1.0, 
            minimum_transition_log_likelihood=-10,
            use_standard_baum_welch=True):
    """
    Main loop for calculating transition probabilities between blocks using EM.
    Iteratively refines the transition matrix until the likelihood converges.
    
    Args:
        full_samples_data (list): Sample data arrays.
        sample_sites (np.ndarray): Genomic locations.
        haps_data (list): BlockResult objects.
        full_blocks_likelihoods (StandardBlockLikelihoods, optional): Pre-computed emissions.
        max_num_iterations (int): Maximum EM steps.
        space_gap (int): HMM stride.
        min_cutoff_change (float): (Unused) Threshold param.
        ll_improvement_cutoff (float): Convergence threshold for Log Likelihood.
        learning_rate (float): Smoothing factor for updates.

    Returns:
        list: [final_forward_transitions, final_backward_transitions]
    """
    
    start_probs = initial_transition_probabilities(haps_data, space_gap=space_gap)
    
    if full_blocks_likelihoods is None:
        print("Warning: full_blocks_likelihoods not provided. Calculating.")
        full_blocks_likelihoods = generate_all_block_likelihoods(
            full_samples_data, sample_sites, haps_data
        )

    # Compute the uniform prior ONCE before the EM loop (it never
    # changes across iterations).
    uniform_prior = initial_transition_probabilities(haps_data, space_gap=space_gap)

    # Cache sorted hap keys ONCE (they never change across iterations).
    hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in haps_data]

    # Restructure emissions ONCE into per-sample chains.  Required by
    # the legacy per-sample E-step code paths
    # (get_full_probs_forward/backward); the new batched E-step inside
    # get_updated_transition_probabilities_unified does not consume this
    # structure but we still construct it here so direct callers of the
    # per-sample functions continue to work without rewiring.
    num_samples = len(full_samples_data)
    num_blocks = len(full_blocks_likelihoods)
    all_block_likelihoods_by_sample = []
    for s in range(num_samples):
        sample_chain = []
        for b in range(num_blocks):
            sample_chain.append(full_blocks_likelihoods[b][s])
        all_block_likelihoods_by_sample.append(sample_chain)

    current_probs = start_probs
    prev_ll = -np.inf
    
    for i in range(max_num_iterations):
        effective_lr = learning_rate * (0.9 ** i)
        effective_lr = max(effective_lr, 0.1)

        new_probs_raw, current_ll = get_updated_transition_probabilities_unified(
            full_samples_data,
            sample_sites,
            haps_data,
            current_probs, 
            full_blocks_likelihoods,
            space_gap=space_gap,
            minimum_transition_log_likelihood=minimum_transition_log_likelihood,
            BATCH_SIZE=100,
            use_standard_baum_welch=use_standard_baum_welch,
            uniform_prior=uniform_prior,
            hap_keys_cache=hap_keys_cache,
            all_block_likelihoods_by_sample=all_block_likelihoods_by_sample
        )
        
        current_probs_smoothed = analysis_utils.smoothen_probs_vectorized(current_probs, new_probs_raw, effective_lr)
        
        if isinstance(current_probs_smoothed, dict):
            current_probs_new = [current_probs_smoothed[0], current_probs_smoothed[1]]
        else:
            current_probs_new = current_probs_smoothed

        current_probs = current_probs_new
        
        # Relative improvement check
        rel_improvement = 0.0
        if prev_ll != -np.inf and prev_ll != 0:
            rel_improvement = (current_ll - prev_ll) / abs(prev_ll)
        elif prev_ll == -np.inf:
            rel_improvement = float('inf') 
            
        if i > 0 and 0 <= rel_improvement < ll_improvement_cutoff:
            break
            
        prev_ll = current_ll
            
    return current_probs

# %% --- WORKER WRAPPER FOR POOL ---
def _gap_worker(args):
    """
    Unpacks arguments and calls the calculation function.
    Reads emissions from _BL_SHARED (populated by the mesh entry point
    before the sequential gap loop) to avoid threading the large
    likelihoods tensor through the per-task argument tuple.
    """
    (gap, full_samples, sites, haps, max_iter, min_ll, lr, use_std_bw) = args
    likes = _BL_SHARED.get('full_blocks_likelihoods', None)
    return calculate_hap_transition_probabilities(
        full_samples,
        sites,
        haps,
        full_blocks_likelihoods=likes,
        max_num_iterations=max_iter,
        space_gap=gap,
        minimum_transition_log_likelihood=min_ll,
        learning_rate=lr,
        use_standard_baum_welch=use_std_bw
    )


def generate_transition_probability_mesh(full_samples_data,
                                         sample_sites,
                                         haps_data,
                                         max_num_iterations=10,
                                         minimum_transition_log_likelihood=-10,
                                         learning_rate=1,
                                         use_standard_baum_welch=True,
                                         num_processes=16):
    """
    Generates a TransitionMesh by calculating transition probabilities
    for ALL possible gap sizes (1 to N).

    This creates a multi-scale view of the haplotype graph, allowing
    downstream algorithms (like Beam Search) to skip over noisy blocks.

    Emissions are stored once in the module-level _BL_SHARED dict and
    read from there by _gap_worker, avoiding the cost of threading the
    full likelihoods tensor through the per-task argument tuple.

    Args:
        full_samples_data (list): Sample data.
        sample_sites (np.ndarray): Genomic locations.
        haps_data (list): List of BlockResult objects.
        max_num_iterations (int): EM iterations per gap size.
        use_standard_baum_welch (bool):
            If True: Uses standard update (sensitive to initialization/priors).
            If False: Uses Reset update (recommended for Viterbi/Hard EM).
        num_processes (int): kept for signature compatibility with callers
                            (notably hierarchical_assembly._process_single_batch);
                            the function always runs the gap loop in-process.
                            See the body comment for rationale.

    Returns:
        TransitionMesh: The fully populated mesh of transition probabilities.
    """

    full_blocks_likelihoods = generate_all_block_likelihoods(
        full_samples_data, sample_sites, haps_data, num_processes=1
    )
    _malloc_trim()

    max_gap = len(haps_data) - 1
    gaps = list(range(1, max_gap + 1))

    # Tasks carry only lightweight args — emissions shared via _BL_SHARED
    worker_args = []
    for gap in gaps:
        worker_args.append((
            gap,
            full_samples_data,
            sample_sites,
            haps_data,
            max_num_iterations,
            minimum_transition_log_likelihood,
            learning_rate,
            use_standard_baum_welch
        ))

    shared_data = {'full_blocks_likelihoods': full_blocks_likelihoods}

    # Sequential execution.  num_processes is kept in the signature for
    # backward compatibility with callers (notably
    # hierarchical_assembly._process_single_batch, which passes whatever
    # pool_budget the outer scheduler computed), but the function always
    # runs the gap loop in-process — production calls this function from
    # an outer Pool worker already, where dispatching another inner Pool
    # would compete for the same physical cores.  Parallelism inside the
    # EM step is provided by numba prange in _batched_baum_welch_mass_kernel
    # (and the new _log_matmul_*_kernel batched forms in analysis_utils),
    # which scale automatically with whatever numba.set_num_threads value
    # the outer scheduler has applied to this worker.
    del num_processes
    global _BL_SHARED
    _BL_SHARED = shared_data
    results = []
    for args in worker_args:
        results.append(_gap_worker(args))
        _malloc_trim()

    del full_blocks_likelihoods, worker_args, shared_data
    _malloc_trim()

    mesh_dict = dict(zip(gaps, results))
    return TransitionMesh(mesh_dict)