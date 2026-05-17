import thread_config

import numpy as np
import math
import os
import warnings
import ctypes
from multiprocessing import Pool
import multiprocessing as _mp
import multiprocessing.pool as _mpp
from concurrent.futures import ThreadPoolExecutor
from scipy.special import logsumexp
from functools import partial

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

# Suppress divide by zero warnings in log-space calculations
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Standard robustness parameter to prevent zero-probability crashes
DEFAULT_ROBUSTNESS_EPSILON = 1e-2

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

# ---------------------------------------------------------------------------
# Forkserver pool for generate_transition_probability_mesh_double_hmm's
# parallel-pool path.  Required when the parent process has JIT-compiled
# parallel=True numba kernels before forking workers — GNU OpenMP (used
# by numba's prange on the OMP threading layer) attaches per-process
# state that is unsafe to inherit across fork().  Forkserver workers
# spawn from a clean intermediate process that never touched OpenMP, so
# they're safe even after the parent has used parallel kernels.
#
# Mirrors block_linking.py's _ForkserverPool exactly.  Preloads are
# configured in thread_config.py (hmm_matching is already in the list).
# Falls back to fork context on platforms without forkserver.
# ---------------------------------------------------------------------------
try:
    _forkserver_ctx = _mp.get_context('forkserver')
except ValueError:
    _forkserver_ctx = _mp.get_context('fork')


class _ForkserverPool(_mpp.Pool):
    """A Pool that uses the forkserver context.  Mirrors
    block_haplotypes._ForkserverPool and block_linking._ForkserverPool.
    """
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _forkserver_ctx
        super().__init__(*args, **kwargs)


# =============================================================================
# SHARED MEMORY MANAGEMENT
# =============================================================================

_SHARED_DATA = {}

# ---------------------------------------------------------------------------
# Dynamic thread reallocation for the pool path in
# generate_transition_probability_mesh_double_hmm.
#
# These mirror block_linking.py's _BL_ACTIVE_COUNTER / _BL_TOTAL_CORES
# globals, with one difference: the rest of hmm_matching uses a
# `dynamic_cores_fn` *callback* (passed in via the public API) for
# dynamic rescaling.  When the parallel-pool path is taken (else branch
# of generate_transition_probability_mesh_double_hmm), no external
# caller-supplied callback is available, so workers build one INTERNALLY
# from these globals via _get_pool_dynamic_cores_fn().  The callback is
# then plumbed down through _gap_worker ->
# calculate_hap_transition_probabilities -> update_transitions_layered_hmm
# the same way an external dynamic_cores_fn is plumbed in the
# sequential-with-callback path.
#
# Both globals are None outside a properly-initialised pool worker.  All
# code that reads them checks for None first.
# ---------------------------------------------------------------------------
_HM_ACTIVE_COUNTER = None
_HM_TOTAL_CORES = None

# ---------------------------------------------------------------------------
# Remainder distribution (mirrors block_linking._BL_EXTRA_COUNTER).
# When total_cores is not evenly divisible by active workers,
# floor(total/active) leaves `remainder = total % active` idle cores.
# E.g. total=112, active=76: floor=1, remainder=36 — 36 cores sit idle
# until floor jumps to 2 (at active=56).  The extra-counter mechanism
# tracks how many workers hold +1 threads so that exactly `remainder`
# workers get ceil and the rest get floor — keeping total threads in
# use equal to total_cores with zero idle.
#
# _HM_EXTRA_COUNTER: pool-wide atomic int.  Reflects the current
#     number of workers holding +1 threads.
# _HM_I_HAVE_EXTRA: per-worker-process bool.  True iff this worker
#     currently holds a claim from _HM_EXTRA_COUNTER's pool.
#
# Both are None / False outside a properly-initialised pool.  See
# _try_claim_extra_hm and _try_release_extra_hm for atomicity details.
# ---------------------------------------------------------------------------
_HM_EXTRA_COUNTER = None
_HM_I_HAVE_EXTRA = False


def _try_claim_extra_hm(remainder):
    """Atomically attempt to claim an extra thread from the remainder pool.

    Returns True if successfully claimed (and sets _HM_I_HAVE_EXTRA).
    Returns False if the pool is exhausted or the counter isn't set up.
    Idempotent: re-calling while already holding does not double-claim.

    Mirrors block_linking._try_claim_extra — see that function's
    docstring for the full atomicity/race-analysis discussion.
    """
    global _HM_I_HAVE_EXTRA
    if _HM_I_HAVE_EXTRA:
        return True
    if _HM_EXTRA_COUNTER is None:
        return False
    try:
        with _HM_EXTRA_COUNTER.get_lock():
            if _HM_EXTRA_COUNTER.value < remainder:
                _HM_EXTRA_COUNTER.value += 1
                _HM_I_HAVE_EXTRA = True
                return True
    except Exception:
        pass
    return False


def _try_release_extra_hm():
    """Atomically release this worker's extra claim, if held.

    Returns True if released, False if nothing to release.  Defensive:
    clears the local flag even if the shared counter mutation failed.

    Mirrors block_linking._try_release_extra.
    """
    global _HM_I_HAVE_EXTRA
    if not _HM_I_HAVE_EXTRA:
        return False
    if _HM_EXTRA_COUNTER is None:
        _HM_I_HAVE_EXTRA = False
        return False
    try:
        with _HM_EXTRA_COUNTER.get_lock():
            _HM_EXTRA_COUNTER.value -= 1
            _HM_I_HAVE_EXTRA = False
            return True
    except Exception:
        _HM_I_HAVE_EXTRA = False
        return False


def _init_shared_data(data_dict, numba_threads=None,
                       active_counter=None, total_cores=None,
                       extra_counter=None):
    """
    Initializer for the worker pool.
    Updates the global _SHARED_DATA dict in the worker process.

    Args:
        data_dict: shared payload (e.g. viterbi_emissions).  Stored in
            module-global _SHARED_DATA.
        numba_threads: optional initial numba thread count for this
            worker.  Legacy parameter, preserved for backwards
            compatibility.  If provided, calls numba.set_num_threads.
        active_counter: optional multiprocessing.Value('i', 0) shared
            across workers, used for dynamic thread reallocation in the
            parallel-pool path.  When None (default — preserving the
            original signature), no counter wiring is set up.
        total_cores: optional int — the original num_processes budget.
            When provided alongside active_counter, sets up the numba
            pool ceiling and stores both in module globals so
            _get_pool_dynamic_cores_fn() works.  When None, no pool
            ceiling change.
        extra_counter: optional multiprocessing.Value('i', 0) for
            remainder distribution.  When provided, workers atomically
            claim/release from this pool so that exactly `remainder =
            total % active` workers hold ceil(total/active) threads
            and the rest hold floor — keeping total threads in use
            equal to total_cores with zero idle.  When None (default),
            falls back to floor-only allocation.  Mirrors
            block_linking's extras-counter mechanism.

    Backwards-compatible: callers passing only `data_dict` (or
    `data_dict, numba_threads`) get the original behavior — counter
    globals stay None, no dynamic reallocation.
    """
    global _SHARED_DATA, _HM_ACTIVE_COUNTER, _HM_TOTAL_CORES
    global _HM_EXTRA_COUNTER, _HM_I_HAVE_EXTRA
    _SHARED_DATA.clear()
    _SHARED_DATA.update(data_dict)
    if numba_threads is not None:
        try:
            import numba
            numba.set_num_threads(numba_threads)
        except Exception:
            pass

    _HM_ACTIVE_COUNTER = active_counter
    _HM_TOTAL_CORES = total_cores
    _HM_EXTRA_COUNTER = extra_counter
    # Defensive: ensure no stale claim from a previous worker recycle.
    _HM_I_HAVE_EXTRA = False

    # Mirror block_linking._init_bl_shared: when dynamic-thread wiring
    # is set up, configure the numba pool ceiling for this worker so
    # set_num_threads() can scale freely up to total_cores later.
    # Starting at 1 thread is the safe default — _gap_worker will
    # rescale immediately based on the live active count when it picks
    # up its first task.
    if total_cores is not None:
        try:
            import os as _os, numba as _numba
            _os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
            _numba.config.NUMBA_NUM_THREADS = total_cores
            _numba.set_num_threads(1)
        except Exception:
            pass


def _get_pool_dynamic_cores_fn():
    """Build a dynamic_cores_fn callback from the pool worker's globals.

    Returns a callable that reads _HM_ACTIVE_COUNTER (live count of
    active peers) and computes the worker's fair share including
    remainder distribution: floor + (1 if this worker holds an extra
    else 0).  The callback may attempt to claim or release an extra
    on each call based on the current `remainder`, just like
    block_linking._update_dynamic_threads.

    Returns None when the globals aren't set up (i.e. this worker
    wasn't initialised by a counter-wired pool), so callers can fall
    back to no-op behavior.

    The closure captures the globals by reference via module attribute
    lookup, so subsequent updates to _HM_ACTIVE_COUNTER.value and
    _HM_EXTRA_COUNTER.value are visible to every call of the returned
    callable.
    """
    if _HM_ACTIVE_COUNTER is None or _HM_TOTAL_CORES is None:
        return None

    def _callback():
        try:
            active = max(_HM_ACTIVE_COUNTER.value, 1)
            floor = _HM_TOTAL_CORES // active
            remainder = _HM_TOTAL_CORES - floor * active

            # Adjust extra-claim based on current remainder (same logic
            # as block_linking._update_dynamic_threads):
            #   - If I don't hold extra and remainder has room, try to claim.
            #   - If I hold extra but extras-in-circulation exceeds
            #     current remainder, release.
            if _HM_EXTRA_COUNTER is not None:
                try:
                    current_extras = _HM_EXTRA_COUNTER.value
                except Exception:
                    current_extras = 0
                if not _HM_I_HAVE_EXTRA:
                    if current_extras < remainder:
                        _try_claim_extra_hm(remainder)
                else:
                    if current_extras > remainder:
                        _try_release_extra_hm()

            return max(1, floor + (1 if _HM_I_HAVE_EXTRA else 0))
        except Exception:
            return 1
    return _callback

# =============================================================================
# 1. OPTIMIZED NUMBA KERNELS (O(N^2) Single-Switch)
# =============================================================================

@njit(fastmath=True)
def log_add_exp(a, b):
    """
    Numerically stable log-add-exp helper for scalars.
    Calculates log(exp(a) + exp(b)).
    """
    if a == -np.inf: return b
    if b == -np.inf: return a
    
    if a > b:
        return a + math.log(1.0 + math.exp(b - a))
    else:
        return b + math.log(1.0 + math.exp(a - b))

@njit(parallel=True, fastmath=True)
def scan_distance_aware_forward(ll_tensor, positions, recomb_rate, state_definitions, incoming_priors, n_haps):
    """
    Performs the 'Micro-HMM' Forward Scan (Log-Sum-Exp) inside a single block.
    
    OPTIMIZATION:
    Uses the Single-Switch Assumption (at most one chromosome recombines per site).
    This reduces complexity from O(Sites * Haps^4) to O(Sites * Haps^2).
    
    Includes BURST LOGIC:
    Maintains parallel 'Normal' and 'Burst' states to handle gene conversions/errors.
    
    Args:
        ll_tensor (np.ndarray): Shape (Samples, K, Sites). Log-likelihood of data given state.
        positions (np.ndarray): Genomic positions of sites in this block.
        recomb_rate (float): Probability of recombination per base pair.
        state_definitions (np.ndarray): Shape (K, 2). Maps state index to (Hap1, Hap2).
        incoming_priors (np.ndarray): Shape (Samples, K). The accumulated probability 
                                      mass arriving at the *start* of this block.
        n_haps (int): Number of haplotypes in this block.
        
    Returns:
        np.ndarray: End probabilities (Samples, K).
    """
    n_samples, K, n_sites = ll_tensor.shape
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

    for s in prange(n_samples):
        # 1. INJECTION: Site 0 gets Emission + Incoming Prior (Macro-Transition)
        current_normal = np.empty(K, dtype=np.float64)
        current_burst = np.empty(K, dtype=np.float64)
        
        for k in range(K):
            prior = incoming_priors[s, k]
            emission = ll_tensor[s, k, 0]
            current_normal[k] = prior + emission
            current_burst[k] = prior + GAP_OPEN + UNIFORM_LOG_PROB
            
        # 2. SCAN: Propagate from Site 1 to N (Micro-Transition)
        for i in range(1, n_sites):
            next_normal = np.empty(K, dtype=np.float64)
            next_burst = np.empty(K, dtype=np.float64)
            
            # Costs
            dist_bp = positions[i] - positions[i-1]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5 
            
            if theta < min_prob:
                log_switch = -1e20
                log_stay = 0.0
            else:
                log_switch = math.log(theta)
                log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay
            cost_1 = log_switch + log_stay - log_N_minus_1
            # Cost 2 (Double Switch) is banned (-inf) under this assumption
            
            # --- OPTIMIZATION: PRE-CALCULATE ROW/COL AGGREGATES ---
            # row_sums[h1] = Sum over h2 of P(h1, h2) -> Mass where Chr1 is h1
            # col_sums[h2] = Sum over h1 of P(h1, h2) -> Mass where Chr2 is h2
            
            row_sums = np.full(n_haps, -np.inf, dtype=np.float64)
            col_sums = np.full(n_haps, -np.inf, dtype=np.float64)
            
            for h1 in range(n_haps):
                for h2 in range(n_haps):
                    k = h1 * n_haps + h2
                    val = current_normal[k]
                    row_sums[h1] = log_add_exp(row_sums[h1], val)
                    col_sums[h2] = log_add_exp(col_sums[h2], val)
            
            # 3. Update States
            for k_curr in range(K):
                h1_curr = k_curr // n_haps
                h2_curr = k_curr % n_haps
                
                # Incoming Mass Logic:
                
                # 1. Stay: (h1, h2) -> (h1, h2)
                term_stay = current_normal[k_curr] + cost_0
                
                # 2. Switch Chr 2: (h1, *) -> (h1, h2)
                term_switch1_a = row_sums[h1_curr] + cost_1
                
                # 3. Switch Chr 1: (*, h2) -> (h1, h2)
                term_switch1_b = col_sums[h2_curr] + cost_1
                
                # Combine (Sum-Product)
                total_incoming = log_add_exp(term_stay, term_switch1_a)
                total_incoming = log_add_exp(total_incoming, term_switch1_b)
                
                # Burst Update (Viterbi/Max style)
                extend = current_burst[k_curr] + BURST_STEP
                open_path = total_incoming + GAP_OPEN + BURST_STEP
                next_burst[k_curr] = max(extend, open_path)
                
                # Normal Update
                close_path = current_burst[k_curr]
                combined = max(total_incoming, close_path)
                
                next_normal[k_curr] = combined + ll_tensor[s, k_curr, i]
            
            # Swap buffers
            for k in range(K):
                current_normal[k] = next_normal[k]
                current_burst[k] = next_burst[k]
        
        # Save final state at last site
        for k in range(K):
            end_probs[s, k] = max(current_normal[k], current_burst[k])
            
    return end_probs

@njit(parallel=True, fastmath=True)
def scan_distance_aware_backward(ll_tensor, positions, recomb_rate, state_definitions, incoming_priors, n_haps):
    """
    Optimized Backward Scan (O(Sites * Haps^2)).
    Assumes Single-Switch Only.
    """
    n_samples, K, n_sites = ll_tensor.shape
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
    
    for s in prange(n_samples):
        # 1. Init (Site N)
        next_normal = np.empty(K, dtype=np.float64)
        next_burst = np.empty(K, dtype=np.float64)
        
        for k in range(K):
            val = ll_tensor[s, k, n_sites - 1] + incoming_priors[s, k]
            next_normal[k] = val
            next_burst[k] = UNIFORM_LOG_PROB + incoming_priors[s, k]
            
        # 2. Scan Backwards
        for i in range(n_sites - 2, -1, -1):
            curr_norm_scratch = np.empty(K, dtype=np.float64)
            curr_burst_scratch = np.empty(K, dtype=np.float64)
            
            dist_bp = positions[i+1] - positions[i]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            
            if theta < min_prob:
                log_switch = -1e20
                log_stay = 0.0
            else:
                log_switch = math.log(theta)
                log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay
            cost_1 = log_switch + log_stay - log_N_minus_1
            # Double switch forbidden
            
            # --- AGGREGATES (Future States) ---
            row_sums = np.full(n_haps, -np.inf, dtype=np.float64)
            col_sums = np.full(n_haps, -np.inf, dtype=np.float64)
            
            for h1 in range(n_haps):
                for h2 in range(n_haps):
                    k = h1 * n_haps + h2
                    val = next_normal[k]
                    row_sums[h1] = log_add_exp(row_sums[h1], val)
                    col_sums[h2] = log_add_exp(col_sums[h2], val)
            
            for k_curr in range(K):
                h1_curr = k_curr // n_haps
                h2_curr = k_curr % n_haps
                
                # Flow FROM Current TO Future
                term_stay = next_normal[k_curr] + cost_0
                term_switch1_a = row_sums[h1_curr] + cost_1
                term_switch1_b = col_sums[h2_curr] + cost_1
                
                total_to_future = log_add_exp(term_stay, term_switch1_a)
                total_to_future = log_add_exp(total_to_future, term_switch1_b)
                
                # Burst Logic
                extend = next_burst[k_curr] + BURST_STEP 
                close_path = next_normal[k_curr]
                curr_burst_scratch[k_curr] = max(extend, close_path)
                
                # Normal Logic
                recomb_path = total_to_future 
                open_path = next_burst[k_curr] + GAP_OPEN + BURST_STEP
                combined = max(recomb_path, open_path)
                
                curr_norm_scratch[k_curr] = combined + ll_tensor[s, k_curr, i]
            
            for k in range(K):
                next_normal[k] = curr_norm_scratch[k]
                next_burst[k] = curr_burst_scratch[k]
        
        for k in range(K):
            start_probs[s, k] = max(next_normal[k], next_burst[k])
            
    return start_probs

# =============================================================================
# 2. DATA CONTAINERS & GENERATION
# =============================================================================

class ViterbiBlockLikelihood:
    """
    Holds the per-site log-likelihood tensor for a block, optimized for the Viterbi scan.
    """
    def __init__(self, tensor, positions, state_defs, num_haps):
        self.tensor = tensor         # (Samples, K_States, Sites)
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
    
    # Calculate implied genotype probabilities for each pair, mix with
    # uniform for robustness, and convert to log-likelihood -- all in a
    # single fused numba kernel.  Mathematically identical to the original
    # numpy chain reproduced here for reference:
    #
    #   # Calculate implied genotype probabilities for each pair
    #   # Shape: (K, Sites, 3)
    #   c00 = h0[:, None, :] * h0[None, :, :]
    #   c11 = h1[:, None, :] * h1[None, :, :]
    #   c01 = (h0[:, None, :] * h1[None, :, :]) + (h1[:, None, :] * h0[None, :, :])
    #
    #   # Flatten to (K, Sites) per genotype channel
    #   combos_flat_0 = c00.reshape(num_haps**2, -1)
    #   combos_flat_1 = c01.reshape(num_haps**2, -1)
    #   combos_flat_2 = c11.reshape(num_haps**2, -1)
    #
    #   # Calculate Model Probability: Sum_g P(Data|g) * P(g|Model)
    #   # Samples: (N, Sites, 3) -> Extract columns (N, Sites)
    #   # Broadcasting: (N, 1, Sites) * (1, K, Sites)
    #
    #   # Term 0: Read=0 * Genotype=0
    #   term_0 = samples_masked[:, np.newaxis, :, 0] * combos_flat_0[np.newaxis, :, :]
    #   term_1 = samples_masked[:, np.newaxis, :, 1] * combos_flat_1[np.newaxis, :, :]
    #   term_2 = samples_masked[:, np.newaxis, :, 2] * combos_flat_2[np.newaxis, :, :]
    #
    #   model_probs = term_0 + term_1 + term_2 # (N, K, Sites)
    #
    #   # Robustness Mixture: (1-eps)*Model + eps*Uniform
    #   # Uniform for 3 genotype states is 1/3
    #   uniform_prob = 1.0 / 3.0
    #
    #   final_probs = (model_probs * (1.0 - epsilon)) + (epsilon * uniform_prob)
    #
    #   # Safety floor for log (avoids -inf)
    #   min_prob = 1e-300
    #   final_probs[final_probs < min_prob] = min_prob
    #
    #   ll_per_site = np.log(final_probs)
    #
    #   # FIX: Apply Hard Floor of -2.0 per site
    #   ll_per_site = np.maximum(ll_per_site, -2.0)
    #
    # The kernel eliminates the (N, K**2, Sites) intermediate tensors
    # (term_0, term_1, term_2, model_probs, final_probs), dropping peak
    # working memory from ~5x output size to 1x output size.  At K=36 with
    # 200-site blocks this is the difference between ~3.2 GB peak (which
    # OOMs the pool worker) and ~650 MB.  CPU benchmarks (single-thread,
    # the relevant case in ThreadPoolExecutor workers) show 1.3x speedup
    # at K=5, 1.7x at K=10, 2.1x at K=20, 2.6x at K=36 -- growing with K
    # because the win is memory-bandwidth (eliminated intermediate
    # traffic), and the intermediate is K**2 * sites per sample.
    # See _viterbi_emission_kernel for the math expressed as inline loops.
    #
    # np.result_type(...) picks the same dtype numpy's broadcasting chain
    # would have produced (the highest-precision input among h0/h1/
    # samples_masked).  np.ascontiguousarray then guarantees both contiguity
    # (h0/h1 are non-contiguous views from haps_masked[:, :, 0/1]) and
    # the chosen dtype, so the kernel sees aligned, typed inputs.  The
    # kernel's output dtype matches samples.dtype (== the common dtype),
    # preserving the original chain's dtype-promotion semantics exactly.
    _common_dtype = np.result_type(h0, h1, samples_masked)
    _h0_c = np.ascontiguousarray(h0, dtype=_common_dtype)
    _h1_c = np.ascontiguousarray(h1, dtype=_common_dtype)
    _samples_c = np.ascontiguousarray(samples_masked, dtype=_common_dtype)
    # Constants: 1e-300 is the safety floor used by the original
    # final_probs[final_probs < min_prob] = min_prob clip; -2.0 is the
    # hard per-site LL floor introduced by the "FIX: Apply Hard Floor"
    # change above.
    ll_per_site = _viterbi_emission_kernel(
        _h0_c, _h1_c, _samples_c,
        float(epsilon), 1e-300, -2.0,
    )
    
    return ViterbiBlockLikelihood(np.ascontiguousarray(ll_per_site), valid_positions, state_defs, num_haps)

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
        K_curr = block.tensor.shape[1] 
        n_haps = block.num_haps
        
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
        K_curr = block.tensor.shape[1]
        n_haps = block.num_haps
        
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
        
        # Numba-fused replacement of the original per-batch numpy
        # operations.  The original code allocated a 3D Total array of
        # shape (B, K_curr_sq, K_next_sq) per batch (8 MB at K=10,
        # 1.3 GB at K=36) and ran scipy.logsumexp / numpy elementwise
        # ops on it.  The kernel processes ALL samples in one call
        # without materialising the 3D intermediate — only an
        # O(num_samples) sample_total array of working memory.
        # See _batched_posterior_aggregation_kernel for the algorithm
        # and numerical-equivalence argument.
        _batched_posterior_aggregation(
            S_earlier, R_later, T_mat, numerators
        )

        # Collapse Diploid States -> Haplotype Transitions.
        # The kernel does the equivalent of:
        #   for each (r, c) with r = u1*n_c + u2 and c = v1*n_n + v2:
        #     hap_masses[(u1, v1)] += [mass - log(prior_u1_v1)]   (if subtract)
        #     hap_masses[(u2, v2)] += [mass - log(prior_u2_v2)]   (if subtract)
        #   data_log_count[u, v] = logsumexp(hap_masses[(u, v)])
        # ...as a single tight numba loop with inlined logaddexp,
        # avoiding the per-cell Python dict + list allocations that
        # dominated the original loop's cost.
        n_c = len(curr_keys)
        n_n = len(next_keys)
        sparse_trans = current_trans[0][i]
        if use_standard_baum_welch:
            # No prior subtraction needed; pass a zero-filled hap_log_prior.
            hap_log_prior = np.zeros((n_c, n_n), dtype=np.float64)
            subtract_prior = False
        else:
            # Build the (n_c, n_n) haploid prior matrix from the sparse
            # dict via the shared analysis_utils helper.  missing_default
            # = log(1e-9) matches the original's sparse_trans.get(..., 1e-9)
            # then math.log() pattern, where missing edges are treated as
            # having a very small (but non-zero) probability rather than
            # being forbidden.  The helper is the same one used by
            # block_linking.get_full_probs_forward/backward (with its
            # default missing_default=-inf) and by hmm_matching's
            # build_dense_transition_matrix — keeps the dict-to-dense
            # logic in one place.
            hap_log_prior = analysis_utils._build_haploid_log_T_from_dict(
                sparse_trans, curr_keys, next_keys, i, next_idx,
                missing_default=math.log(1e-9))
            subtract_prior = True

        # data_log_count[u_i, v_i] is what the original loop produces in
        # one go.  Cells with no contribution remain -inf.
        data_log_count = _diploid_collapse_kernel(
            numerators, hap_log_prior, n_c, n_n, subtract_prior)

        # Apply Smoothing and Normalize.  data_log_count is now an
        # (n_c, n_n) dense matrix with -inf for missing cells, equivalent
        # to logsumexp(hap_masses[(u_i, v_i)]) in the original.
        #
        # The smoothing + per-row normalize + robust-mixture computation
        # is delegated to _smooth_normalize_kernel, which produces a
        # dense (n_c, n_n) final_p matrix.  Mathematically identical
        # (byte-equivalent at 1 ULP) to the original numpy/python chain
        # reproduced below for reference:
        #
        #   fwd_raw_edges = {u: {} for u in curr_keys}
        #
        #   for u_i in range(n_c):
        #       for v_i in range(n_n):
        #           data_lc = data_log_count[u_i, v_i]
        #           # Add Pseudocounts (prevents death spiral).  Note:
        #           # np.logaddexp(-inf, x) returns x, so missing cells get
        #           # smoothed_val = LOG_PSEUDO, matching the original.
        #           smoothed_val = np.logaddexp(data_lc, LOG_PSEUDO)
        #
        #           src = curr_keys[u_i]
        #           dst = next_keys[v_i]
        #           fwd_raw_edges[src][dst] = smoothed_val
        #
        #   # Row Normalization
        #   final_fwd = {}
        #   for src, targets in fwd_raw_edges.items():
        #       if not targets: continue
        #       row_vals = list(targets.values())
        #       row_total = analysis_utils.lse_scalar(row_vals)
        #
        #       renorm_sum = 0.0
        #       temp_probs = {}
        #       for dst, log_val in targets.items():
        #           log_p = log_val - row_total if row_total != -np.inf else -np.inf
        #           if log_p < MIN_LOG_PROB: log_p = MIN_LOG_PROB
        #           p = math.exp(log_p)
        #           temp_probs[dst] = p
        #           renorm_sum += p
        #
        #       if renorm_sum == 0: renorm_sum = 1.0
        #
        #       # Robust Mixture (1% Uniform)
        #       uniform_val = 1.0 / len(temp_probs)
        #       mix_rate = 0.01
        #
        #       for dst, p in temp_probs.items():
        #           norm_p = p / renorm_sum
        #           final_p = (norm_p * (1.0 - mix_rate)) + (uniform_val * mix_rate)
        #           key = ((i, src), (next_idx, dst))
        #           final_fwd[key] = final_p
        #
        # The kernel collapses the per-cell python overhead -- ~27x speedup
        # at K=5, ~78x at K=36 (measured) -- on what becomes hot at high K.
        final_p_mat = _smooth_normalize_kernel(
            data_log_count, LOG_PSEUDO, MIN_LOG_PROB, 0.01)

        # Dict-write back to the production {((i, src), (next_idx, dst)):
        # final_p} format.  Keys are arbitrary tuples that numba can't
        # construct, so this stays in Python; the per-cell work here is
        # just a dict insertion (negligible vs the math the kernel just
        # finished).
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
        
        # Numba-fused replacement of the per-batch numpy operations.
        # Same kernel as LOOP 1's forward case — the semantics are
        # symmetric:
        #   LOOP 1 (forward):  Total = S_earlier + R_later + T_fwd
        #   LOOP 2 (backward): Total = R_later  + S_earlier + T_bwd
        # The kernel takes (S, R, T_mat) positionally; for the backward
        # pass we pass (R_later_source, S_earlier_dest, T_mat) so that
        # the kernel's "S" axis indexes states at block i (curr_keys²)
        # and the "R" axis indexes states at block prev_idx (prev_keys²),
        # matching the original (R_batch on first newaxis, S_batch on
        # second newaxis) order.  T_mat is built with i,prev_idx and
        # correct_hom_hom=True, exactly as the original.
        _batched_posterior_aggregation(
            R_later_source, S_earlier_dest, T_mat, numerators
        )

        # Collapse Diploid States -> Haplotype Transitions (backward).
        # Same algorithm as the forward case, with prev_keys playing the
        # role of next_keys.  See the forward block's comment block for
        # the algorithmic details and numerical-equivalence argument.
        n_c = len(curr_keys)
        n_p = len(prev_keys)
        sparse_trans = current_trans[1][i]
        if use_standard_baum_welch:
            hap_log_prior = np.zeros((n_c, n_p), dtype=np.float64)
            subtract_prior = False
        else:
            # Same helper-based prior construction as LOOP 1, but for
            # the backward direction (key tuple uses prev_idx for the
            # second element).
            hap_log_prior = analysis_utils._build_haploid_log_T_from_dict(
                sparse_trans, curr_keys, prev_keys, i, prev_idx,
                missing_default=math.log(1e-9))
            subtract_prior = True

        data_log_count = _diploid_collapse_kernel(
            numerators, hap_log_prior, n_c, n_p, subtract_prior)

        # Normalize Backward.  Same data_log_count -> smoothed_val
        # transformation as in the forward pass.  See LOOP 1's comment
        # block above for the full numpy/python reference; this is the
        # backward-direction equivalent, differing only in that the
        # second key tuple element is prev_idx (not next_idx) and the
        # destination keys come from prev_keys (not next_keys).
        final_p_mat = _smooth_normalize_kernel(
            data_log_count, LOG_PSEUDO, MIN_LOG_PROB, 0.01)

        # Dict-write back to the production {((i, src), (prev_idx, dst)):
        # final_p} format.  See LOOP 1's matching comment for rationale.
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
                                           ll_improvement_cutoff=1e-4,
                                           use_standard_baum_welch=True,
                                           precalculated_viterbi_emissions=None, # NEW ARGUMENT
                                           dynamic_cores_fn=None):
    """
    Main driver for HMM-EM transition calculation.
    Supports pre-calculated emissions to avoid passing massive raw data to workers.

    Args:
        dynamic_cores_fn (callable, optional): If provided, called at the top
            of each EM iteration to obtain the current core allocation for
            this worker.  The returned value is passed to
            numba.set_num_threads(), so parallel=True kernels (notably
            scan_distance_aware_forward/backward inside
            global_forward_backward_pass) use the live thread count.  This
            extends the existing "between-gaps" rescaling pattern in
            generate_transition_probability_mesh_double_hmm to ALSO rescale
            between EM iterations within a single gap — long stragglers
            mid-EM now pick up cores freed by peer workers finishing,
            instead of waiting until the next gap boundary.

            Mirrors the design block_linking.py uses for stage-4 dynamic
            scaling (Portion 4) — in-flight hooks at top of EM iteration
            and top of M-step block loop.

            When None, no dynamic rescaling is performed within this
            function — preserves the original behavior exactly.
    """
    current_trans = block_linking.initial_transition_probabilities(haps_data, space_gap)
    
    # Use pre-calculated emissions if provided, otherwise calculate them here
    if precalculated_viterbi_emissions is not None:
        raw_blocks = precalculated_viterbi_emissions
    else:
        # Fallback to internal calculation (High Memory usage if data is large)
        raw_blocks = generate_viterbi_block_emissions(
            full_samples_data, sample_sites, haps_data, num_processes=num_processes
        )
    
    # Fix A: Cache sorted hap keys ONCE (never change across EM iterations)
    hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in haps_data]
    
    prev_ll = -np.inf
    
    for it in range(max_num_iterations):
        # Dynamic thread reallocation: re-check the current core
        # allocation and rescale numba threads.  When this worker is a
        # straggler (peer workers have finished their gaps and freed
        # cores), the callback's return value drops and this call
        # scales us up accordingly.  parallel=True kernels (the prange
        # in scan_distance_aware_forward/backward) will then use the
        # new thread count on their next invocation.
        #
        # No-op when dynamic_cores_fn is None (no caller-supplied
        # callback) — preserves the original behavior exactly.
        if dynamic_cores_fn is not None:
            try:
                import numba as _numba
                n_cores = dynamic_cores_fn()
                _numba.set_num_threads(n_cores)
            except Exception:
                # Numba unavailable or callback errored — silently
                # ignore, same robustness posture as
                # block_haplotypes._update_dynamic_threads.
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
            
        # Convergence Check
        rel_improvement = 0.0
        if prev_ll != -np.inf and prev_ll != 0:
            rel_improvement = (current_ll - prev_ll) / abs(prev_ll)
        elif prev_ll == -np.inf:
            rel_improvement = float('inf') 
            
        if it > 0 and 0 <= rel_improvement < ll_improvement_cutoff:
            break
            
        prev_ll = current_ll
            
    return current_trans

def _gap_worker(args):
    """Worker for multiprocessing gap calculations.

    The args tuple is one of two lengths:
      - 8 elements: legacy signature (no dynamic_cores_fn).  Falls
        through to calculate_hap_transition_probabilities with no
        in-flight rescaling.
      - 9 elements: extended signature with dynamic_cores_fn as the
        9th element.  The callback is forwarded down to
        calculate_hap_transition_probabilities so the EM iteration
        loop and the M-step's per-block loop can rescale numba
        threads mid-gap.  When the callback is None, behavior is
        identical to the 8-element path.

    Supporting both lengths keeps backwards compatibility with any
    pre-existing pickled task tuples or external test harnesses that
    might call this with the original 8-element form.

    Dynamic thread reallocation in the parallel-pool path: when
    _HM_ACTIVE_COUNTER and _HM_TOTAL_CORES are set (by an
    appropriately-configured _init_shared_data call), this worker
      - atomically increments the counter on entry and sets initial
        numba threads to max(1, total // active)
      - builds an internal dynamic_cores_fn (via
        _get_pool_dynamic_cores_fn) and passes it to
        calculate_hap_transition_probabilities so the in-flight hooks
        also use the live counter
      - atomically decrements the counter on exit (via try/finally so
        crashes still decrement)
    When the globals aren't set (sequential path or 8-element args),
    none of this fires and behavior is the original.
    """
    # Unpack the new flag from the arguments tuple (now includes emissions)
    if len(args) == 9:
        (gap, samples, sites, haps, max_iter, rate, use_std_bw,
         use_shared_emissions, dynamic_cores_fn) = args
    else:
        gap, samples, sites, haps, max_iter, rate, use_std_bw, use_shared_emissions = args
        dynamic_cores_fn = None

    # Pool-path dynamic-thread setup: when the worker globals are
    # wired (i.e. _init_shared_data was called with active_counter +
    # total_cores), increment the counter and build an internal
    # callback that the downstream EM loop will use for in-flight
    # rescaling.
    counter_inc = False
    if _HM_ACTIVE_COUNTER is not None and _HM_TOTAL_CORES is not None:
        try:
            import numba as _numba
            # Increment under the counter's lock — required because
            # two workers picking up tasks near-simultaneously could
            # otherwise race on the read-modify-write.  Mirrors
            # block_linking._gap_worker exactly.
            with _HM_ACTIVE_COUNTER.get_lock():
                _HM_ACTIVE_COUNTER.value += 1
            counter_inc = True
            active = max(_HM_ACTIVE_COUNTER.value, 1)
            floor = _HM_TOTAL_CORES // active
            remainder = _HM_TOTAL_CORES - floor * active
            # Try to claim an extra thread from the remainder pool.
            # No-op when _HM_EXTRA_COUNTER is None (legacy callers
            # without remainder distribution).  Mirrors
            # block_linking._gap_worker.
            _try_claim_extra_hm(remainder)
            n_threads = max(1, floor + (1 if _HM_I_HAVE_EXTRA else 0))
            _numba.set_num_threads(n_threads)
        except Exception:
            # Transient issue — fall through and run the task without
            # dynamic scaling.  Counter still decremented (if
            # incremented) so peers see correct activity.
            pass

        # Build/replace the dynamic_cores_fn with the pool-internal
        # callback so in-flight EM hooks read the counter too.  This
        # overrides any externally-supplied callback in the
        # parallel-pool path because the pool's counter is the
        # authoritative source of "active workers" within this pool.
        # External callbacks (e.g. from hierarchical_assembly) are
        # only meaningful in the sequential path where there's no
        # internal counter.
        pool_callback = _get_pool_dynamic_cores_fn()
        if pool_callback is not None:
            dynamic_cores_fn = pool_callback

    try:
        if use_shared_emissions:
            # Retrieve the massive emissions object from Shared Memory
            precalc_ems = _SHARED_DATA['viterbi_emissions']
        else:
            # If not using shared memory, this argument would have been passed directly (or None)
            # But in our new architecture, we aim to rely on shared memory for the large object.
            precalc_ems = None

        return calculate_hap_transition_probabilities(
            samples, sites, haps, 
            max_num_iterations=max_iter, 
            space_gap=gap, 
            recomb_rate=rate, 
            num_processes=1, # No nested pool
            use_standard_baum_welch=use_std_bw,
            precalculated_viterbi_emissions=precalc_ems,
            dynamic_cores_fn=dynamic_cores_fn
        )
    finally:
        # Always release any held extra and decrement the active
        # counter, regardless of whether the body succeeded.  Order:
        # release the extra FIRST so peers see the freed extra-slot
        # before the active-count decrement (cleanest invariant; see
        # block_linking._gap_worker for the longer rationale).
        _try_release_extra_hm()
        if counter_inc and _HM_ACTIVE_COUNTER is not None:
            try:
                with _HM_ACTIVE_COUNTER.get_lock():
                    _HM_ACTIVE_COUNTER.value -= 1
            except Exception:
                pass


def _gap_worker_tagged(args):
    """Wraps _gap_worker for use with pool.imap_unordered.

    imap_unordered returns results in completion order, not input
    order, so we tag each result with its gap-size.  The parent
    re-assembles results by gap-size to restore the {gap: result}
    mapping that downstream code expects.

    args[0] is the gap-size (the same convention used in
    generate_transition_probability_mesh_double_hmm's worker_args
    construction).  Used ONLY by the pool path — the sequential paths
    call _gap_worker directly and rely on positional ordering.
    """
    gap = args[0]
    return (gap, _gap_worker(args))


def generate_transition_probability_mesh_double_hmm(full_samples_data, sample_sites, haps_data, 
                                                 max_num_iterations=20, recomb_rate=5e-7,
                                                 use_standard_baum_welch=True,
                                                 precalculated_viterbi_emissions=None,
                                                 num_processes=16,
                                                 dynamic_cores_fn=None):
    """
    Generates a full mesh of transition probabilities for all gap sizes using Viterbi-EM.
    
    Args:
        precalculated_viterbi_emissions: Optional ViterbiBlockList. If provided, 
        full_samples_data and sample_sites are IGNORED to prevent memory pickling overhead.
        num_processes: Number of parallel processes. Use 1 for sequential execution
                      (required when called from within a worker process).
        dynamic_cores_fn: Optional callable returning current core allocation.
            When provided, gaps are processed sequentially with numba threads
            updated between each gap. This enables dynamic scaling as peer
            workers finish — remaining workers automatically use freed cores.
            Throughput is equivalent to pool-based processing (prange over
            samples uses the same total cores) but can adapt mid-computation.
    """
    max_gap = len(haps_data) - 1
    gaps = list(range(1, max_gap + 1))
    
    use_shared_emissions = False
    
    # CRITICAL MEMORY FIX:
    # If using pre-calculated emissions, we MUST put them in Shared Memory
    # rather than passing them as arguments to the worker.
    # Passing as args duplicates the object (pickles) for every worker task.
    shared_context = {}
    
    if precalculated_viterbi_emissions is not None:
        data_arg = None
        sites_arg = None
        use_shared_emissions = True
        shared_context['viterbi_emissions'] = precalculated_viterbi_emissions
    else:
        data_arg = full_samples_data
        sites_arg = sample_sites
    
    # Build the per-task args.  Each task carries:
    #   - the standard 8-element payload (gap, data, sites, haps, ...,
    #     use_shared_emissions)
    #   - the dynamic_cores_fn callback as a 9th element, used by
    #     _gap_worker to enable in-flight (per-EM-iteration and
    #     per-block) numba thread rescaling.
    # In the sequential-with-callback path, the callback is the caller-
    # supplied function (typically hierarchical_assembly's
    # _get_dynamic_threads, which reads its OWN active_counter).  In
    # the sequential-no-callback path, it's None — no rescaling.  In
    # the parallel-pool path, the args carry None at construction time
    # (callbacks can't be pickled to workers); each worker BUILDS its
    # own callback locally from the module-global counter that's set
    # up by _init_shared_data — see _gap_worker for the wiring.
    worker_args = []
    for gap in gaps:
        worker_args.append((
            gap, 
            data_arg, 
            sites_arg, 
            haps_data, 
            max_num_iterations, 
            recomb_rate, 
            use_standard_baum_welch,
            use_shared_emissions, # Flag to tell worker to check shared memory
            dynamic_cores_fn      # In-flight rescaling callback (or None)
        ))
    
    # Handle sequential vs parallel execution
    n_gaps = len(gaps)
    if dynamic_cores_fn is not None:
        # Dynamic scaling: process gaps sequentially, updating numba threads
        # between each gap.  The prange loops in forward/backward scans
        # parallelize across samples, so N numba threads ≈ N pool workers
        # with 1 thread each.  Sequential can re-check the allocation
        # between gaps and scale up as peer workers finish.
        #
        # Improvement A: the callback is now ALSO passed inside the
        # worker args (worker_args[i][8] above), so the in-flight hooks
        # inside calculate_hap_transition_probabilities (top of EM
        # iteration loop) and update_transitions_layered_hmm (top of
        # per-block loops) ALSO rescale during a single gap.  The
        # between-gap rescale below remains as a coarse-grained reset.
        import numba as _numba
        _init_shared_data(shared_context)
        results = []
        for args in worker_args:
            n_cores = dynamic_cores_fn()
            _numba.set_num_threads(n_cores)
            results.append(_gap_worker(args))
            _malloc_trim()
    elif num_processes == 1 or n_gaps <= 1:
        # Sequential: either required (within worker) or only 1 gap (L4).
        # For 1 gap, the caller's numba_thread_scope gives full thread count
        # to the prange scans — no need for Pool overhead.
        _init_shared_data(shared_context)
        results = []
        for args in worker_args:
            results.append(_gap_worker(args))
            _malloc_trim()
    else:
        # Parallel pool path with full dynamic-thread reallocation.
        # Mirrors the architecture block_linking.py uses for its
        # generate_transition_probability_mesh:
        #
        #   - active_counter: _forkserver_ctx.Value('i', 0) shared across
        #     workers.  Workers atomically increment on task entry and
        #     decrement on exit (in _gap_worker).  The counter's live
        #     value drives an internal dynamic_cores_fn callback built
        #     by each worker via _get_pool_dynamic_cores_fn, which the
        #     EM iteration and per-block hooks call to rescale threads.
        #
        #   - _ForkserverPool: required because the parent process may
        #     have JIT-compiled parallel=True numba kernels before
        #     reaching here (e.g.  scan_distance_aware_forward), which
        #     initialises GNU OpenMP.  Forking workers from such a
        #     parent crashes them with "fork() called from a process
        #     already using GNU OpenMP, this is unsafe."  Forkserver
        #     spawns workers from a clean intermediate process.
        #     set_forkserver_preload (in thread_config.py) ensures
        #     hmm_matching and deps are pre-imported in the forkserver,
        #     so worker startup is sub-second.
        #
        #   - imap_unordered(chunksize=1): dispatches tasks one at a
        #     time from the master queue.  Fast gaps finish, decrement
        #     the counter, and stragglers' next in-flight hook sees the
        #     lower count and scales up immediately.  pool.map's chunk
        #     pre-division would defeat this by leaving idle workers.
        #
        #   - _gap_worker_tagged: wraps results with their gap-size so
        #     completion-order output from imap_unordered can be
        #     re-keyed by gap.
        #
        #   - __main__.__file__ / __main__.__spec__ clearing: belt-and-
        #     suspenders to prevent the forkserver process from re-
        #     running the user's entry script during its bootstrap.
        #     Mirrors block_haplotypes and block_linking.

        # Use the original num_processes as the total_cores budget.
        # We do NOT subdivide num_processes into n_pool ×
        # threads_per_worker the way the original parallel branch did,
        # because the dynamic-scaling mechanism handles that
        # automatically: workers start at 1 thread each and scale up
        # to num_processes // active as peers finish.  Caller's
        # original intent of "use up to num_processes total cores" is
        # preserved.
        n_pool = min(num_processes, n_gaps)
        total_cores = num_processes

        active_counter = _forkserver_ctx.Value('i', 0)
        # Extra-thread counter for remainder distribution.  See
        # _try_claim_extra_hm / _try_release_extra_hm and
        # _get_pool_dynamic_cores_fn's docstrings.  Mirrors
        # block_linking's extra_counter — same forkserver context as
        # active_counter for shared-memory consistency.
        extra_counter = _forkserver_ctx.Value('i', 0)

        # Belt-and-suspenders __main__ clearing — see block_linking and
        # block_haplotypes for rationale.
        import sys as _sys
        _main_mod = _sys.modules.get('__main__')
        _saved_main_file = getattr(_main_mod, '__file__', None)
        _saved_main_spec = getattr(_main_mod, '__spec__', None)
        if _main_mod is not None:
            if hasattr(_main_mod, '__file__'):
                del _main_mod.__file__
            _main_mod.__spec__ = None

        try:
            results_by_gap = {}
            with _ForkserverPool(n_pool, initializer=_init_shared_data,
                                 initargs=(shared_context, None,
                                           active_counter, total_cores,
                                           extra_counter)) as pool:
                for gap_key, gap_result in pool.imap_unordered(
                        _gap_worker_tagged, worker_args, chunksize=1):
                    results_by_gap[gap_key] = gap_result
        finally:
            # Restore __main__ attributes (paired with the deletion above).
            if _main_mod is not None:
                if _saved_main_file is not None:
                    _main_mod.__file__ = _saved_main_file
                _main_mod.__spec__ = _saved_main_spec

        # Re-order results to match gaps list (positional order
        # expected by dict(zip(gaps, results)) below).
        results = [results_by_gap[g] for g in gaps]

    del worker_args, shared_context
    _malloc_trim()
    
    mesh_dict = dict(zip(gaps, results))
    return block_linking.TransitionMesh(mesh_dict)