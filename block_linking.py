import numpy as np
import math
import time
import ctypes
from multiprocessing import Pool as _StdPool
import multiprocessing as _mp
import multiprocessing.pool as _mpp
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

# ---------------------------------------------------------------------------
# Forkserver pool — required when generate_transition_probability_mesh runs
# in a session where the parent process may have JIT-compiled
# parallel=True numba kernels.  GNU OpenMP (used by numba for prange
# parallelism on the OMP threading layer) attaches a per-process state
# that is unsafe to inherit across fork().  Forking after OpenMP init
# triggers "Terminating: fork() called from a process already using GNU
# OpenMP, this is unsafe." and aborts workers.
#
# Forkserver starts workers from a lightweight intermediate process
# that never touched OpenMP.  set_forkserver_preload (configured in
# thread_config.py) imports numpy, numba, block_linking, hmm_matching,
# analysis_utils etc. so worker startup stays fast — the forkserver
# process imports them once at startup, and all forked workers inherit
# the imports via COW (~500 MB shared instead of ~2 GB per worker on
# fresh import).
#
# Falls back to the fork context on platforms where forkserver is
# unavailable (e.g. Windows).  Same fallback pattern block_haplotypes
# uses.
# ---------------------------------------------------------------------------
try:
    _forkserver_ctx = _mp.get_context('forkserver')
except ValueError:
    _forkserver_ctx = _mp.get_context('fork')


class _ForkserverPool(_mpp.Pool):
    """A Pool that uses the forkserver context.

    Workers spawn from a clean intermediate process that never
    initialized OpenMP, so it's safe to fork them even after the
    parent has JIT-compiled a parallel=True numba kernel.

    Mirrors block_haplotypes._ForkserverPool exactly.
    """
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _forkserver_ctx
        super().__init__(*args, **kwargs)


# Fix #3: Shared data for Pool workers (avoids pickling large objects per task)
_BL_SHARED = {}

def _init_bl_shared(shared_dict, active_counter=None, total_cores=None,
                     extra_counter=None):
    """Pool initializer: store shared data in worker's global scope.

    Args:
        shared_dict: dict of large objects to share across workers (e.g.
            full_blocks_likelihoods).  Stored in module-global _BL_SHARED.
        active_counter: optional multiprocessing.Value('i', 0) shared
            across workers, used by _update_dynamic_threads to scale
            numba threads up as peer workers finish.  When None
            (default, preserving the original signature), no dynamic
            reallocation occurs and this worker behaves exactly as
            before.
        total_cores: optional int — the original num_processes budget
            for the entire pool.  When provided alongside
            active_counter, the worker configures numba's pool ceiling
            to total_cores and starts at 1 thread; subsequent
            _update_dynamic_threads() calls in the worker body can then
            scale up to total_cores when peers free their share.  When
            None, no numba pool reconfiguration is done.
        extra_counter: optional multiprocessing.Value('i', 0) shared
            across workers, used for remainder distribution.  When
            total_cores is not evenly divisible by active workers
            (e.g. total=112, active=76: remainder=36), this counter
            tracks how many workers currently hold an "extra" thread
            (ceil = floor+1).  Workers atomically claim/release from
            this pool via _try_claim_extra / _try_release_extra so that
            exactly `remainder` workers hold ceil and the rest hold
            floor — keeping total threads in use equal to total_cores
            with zero idle.  When None (default), workers fall back to
            floor-only allocation (the pre-remainder-distribution
            behavior, which can leave up to active-1 cores idle).

    Backwards-compatible: callers passing only shared_dict (as in any
    pre-existing caller before the dynamic-thread wiring) get the
    original behavior — globals stay None, no thread reallocation, no
    numba pool ceiling change.
    """
    global _BL_SHARED, _BL_ACTIVE_COUNTER, _BL_TOTAL_CORES
    global _BL_EXTRA_COUNTER, _BL_I_HAVE_EXTRA
    _BL_SHARED.clear()
    _BL_SHARED.update(shared_dict)
    _BL_ACTIVE_COUNTER = active_counter
    _BL_TOTAL_CORES = total_cores
    _BL_EXTRA_COUNTER = extra_counter
    # Defensive: ensure no stale claim is carried into the new worker
    # context.  Forkserver spawns a fresh process from a clean parent
    # so this is already zero, but if a pool is somehow recycled (e.g.
    # max_tasks_per_child > 1 with a re-init) we want a clean slate.
    _BL_I_HAVE_EXTRA = False

    # When the caller has wired up dynamic threads, configure the
    # numba pool ceiling for this worker so set_num_threads() can
    # scale freely up to total_cores later.  Mirrors
    # block_haplotypes._init_block_worker.  Starting at 1 thread is
    # the safe default — _gap_worker will rescale immediately based
    # on the live active count when it picks up its first task.
    if total_cores is not None:
        try:
            import os as _os, numba as _numba
            _os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
            _numba.config.NUMBA_NUM_THREADS = total_cores
            _numba.set_num_threads(1)
        except Exception:
            # Numba unavailable or thread-set failure — silently ignore,
            # same robustness posture as _update_dynamic_threads.
            pass

# ---------------------------------------------------------------------------
# Dynamic thread reallocation for straggler gap-workers.
# (Portion 1 of 4: dormant infrastructure only — globals + helper.  No
# caller wires these up yet; everything is no-op until later portions hook
# in the pool initializer, the worker entry/exit, and the in-loop re-check
# points.  This portion is safe to ship on its own because none of the
# existing call paths reference the new globals or call the helper.)
#
# Mirrors the proven pattern in block_haplotypes.py:
#   _BH_ACTIVE_COUNTER + _BH_TOTAL_CORES + _update_dynamic_threads().
#
# Mechanism (once fully wired up in later portions):
#   - _BL_ACTIVE_COUNTER: multiprocessing.Value('i', 0) shared across all
#     pool workers.  Incremented atomically when a worker takes a task,
#     decremented when it finishes.  At any moment .value equals the
#     number of workers currently doing work.
#   - _BL_TOTAL_CORES: the original num_processes budget for the whole
#     pool (e.g. 16 in the default generate_transition_probability_mesh
#     call).  Constant within a pool's lifetime; set by the initializer.
#   - _update_dynamic_threads(): read the counter, compute
#     max(1, total_cores // active), and call numba.set_num_threads(...).
#     This is what lets a lone straggler gap-worker run on all cores
#     once its peers have finished.
#
# The numba thread setting controls how many threads parallel=True
# kernels use via prange.  In stage 4 the parallel kernel that
# benefits is _batched_baum_welch_mass_kernel (prange over the batch
# axis of the M-step's 5D body).  Sequential @njit kernels
# (_logsumexp_*, _diploid_collapse_kernel,
# _build_dense_transition_matrix_kernel) are unaffected by the thread
# count — they have no prange — so the setting has no effect on them
# but no harm either.
#
# Safety: both globals are None outside a worker process.  When
# generate_transition_probability_mesh runs with num_processes=1, or
# when calculate_hap_transition_probabilities is called directly (no
# pool at all), the counter stays None and _update_dynamic_threads()
# returns immediately — exactly preserving the existing behavior.
# ---------------------------------------------------------------------------
_BL_ACTIVE_COUNTER = None
_BL_TOTAL_CORES = None

# ---------------------------------------------------------------------------
# Remainder distribution: when total_cores is not evenly divisible by
# active workers, floor(total/active) leaves a remainder of idle cores.
# E.g. total=112, active=76: floor=1, remainder=36 — so 36 of 112 cores
# sit idle until enough peers finish to push floor up to 2 (active=56).
#
# To eliminate that waste we maintain a second atomic counter,
# _BL_EXTRA_COUNTER, that tracks how many workers currently hold an
# "extra" thread (ceil = floor + 1).  At any time at most `remainder =
# total_cores % active` workers may hold an extra; the rest hold floor.
# Total threads in use = (active - n_extra) * floor + n_extra * (floor+1)
#                     = active * floor + n_extra
#                     ≤ active * floor + remainder
#                     = total_cores   (when n_extra == remainder)
#
# Each worker owns at most one "extra claim" at a time, tracked by the
# per-worker-process _BL_I_HAVE_EXTRA bool (set on successful claim,
# cleared on release).  The bool is module-global in the worker process
# but is logically per-worker because each worker process is a
# different OS process with its own module-globals.
#
# Both globals are None outside a pool worker; the wiring code handles
# absence cleanly (no-op fallback to the original floor-only behavior).
# ---------------------------------------------------------------------------
_BL_EXTRA_COUNTER = None
_BL_I_HAVE_EXTRA = False


def _try_claim_extra(remainder):
    """Atomically attempt to claim an extra thread.

    Returns True if successfully claimed (and sets _BL_I_HAVE_EXTRA to
    True as a side effect).  Returns False if the remainder pool is
    already exhausted (i.e. _BL_EXTRA_COUNTER.value >= remainder), in
    which case this worker doesn't get an extra and stays at floor.

    Atomicity is provided by _BL_EXTRA_COUNTER.get_lock().  Without
    the lock, two workers concurrently doing "read; check < remainder;
    increment" could both succeed and over-claim — leading to
    oversubscription beyond total_cores.

    Idempotent: if _BL_I_HAVE_EXTRA is already True, returns True
    without re-claiming (each worker holds at most one extra at any
    time, so a duplicate claim would over-count).
    """
    global _BL_I_HAVE_EXTRA
    if _BL_I_HAVE_EXTRA:
        return True
    if _BL_EXTRA_COUNTER is None:
        return False
    try:
        with _BL_EXTRA_COUNTER.get_lock():
            if _BL_EXTRA_COUNTER.value < remainder:
                _BL_EXTRA_COUNTER.value += 1
                _BL_I_HAVE_EXTRA = True
                return True
    except Exception:
        # Lock acquisition or counter mutation failed — fall back to
        # floor-only.  Same robustness posture as _update_dynamic_threads.
        pass
    return False


def _try_release_extra():
    """Atomically release this worker's extra claim, if held.

    Returns True if a claim was released (and clears _BL_I_HAVE_EXTRA
    as a side effect).  Returns False if this worker didn't hold an
    extra.

    Used in two situations:
      1. Worker exit (in _gap_worker's finally) — release on the way
         out so the pool's total claim count stays accurate.
      2. In-flight rebalance (in _update_dynamic_threads) — when active
         grows (new workers entered), the remainder shrinks and there
         may be too many extras in circulation; some holders need to
         drop theirs.
    """
    global _BL_I_HAVE_EXTRA
    if not _BL_I_HAVE_EXTRA:
        return False
    if _BL_EXTRA_COUNTER is None:
        _BL_I_HAVE_EXTRA = False  # defensive
        return False
    try:
        with _BL_EXTRA_COUNTER.get_lock():
            _BL_EXTRA_COUNTER.value -= 1
            _BL_I_HAVE_EXTRA = False
            return True
    except Exception:
        # Defensive: still clear our local flag even if counter mutation
        # failed.  Otherwise the worker thinks it holds an extra
        # forever, leading to under-claim by peers.
        _BL_I_HAVE_EXTRA = False
        return False


def _update_dynamic_threads():
    """Recheck active worker count and rescale numba threads accordingly.

    Modeled directly after block_haplotypes._update_dynamic_threads.
    Called from inside long-running gap-worker bodies (top of EM
    iteration and top of M-step batched pass — wired up in Portion 4)
    so that stragglers re-evaluate their thread budget as peer workers
    finish and free cores.

    Computes the EXACT fair share with remainder distribution:
        floor     = total_cores // active_workers
        remainder = total_cores % active_workers
        my_share  = floor + (1 if I hold an extra-claim else 0)

    The extra-claim is tracked via _BL_I_HAVE_EXTRA (per-worker) and
    _BL_EXTRA_COUNTER (pool-wide atomic).  At any time at most
    `remainder` workers hold extras, and total threads in use equals
    exactly total_cores (no idle cores).

    On each recheck:
      - If I don't hold an extra and the pool has room (current
        extra-count < remainder), try to claim one.  Lets stragglers
        pick up newly-freed cores immediately.
      - If I hold an extra but active grew (so remainder shrank below
        the current extra-count), try to release — keeps the total
        threads from exceeding total_cores when new workers enter.
    Both operations are atomic via the counter's lock; brief
    contention but never races.

    Behavior:
        - Outside a pool worker (or in a pool that wasn't initialized
          with active_counter/total_cores): silent no-op.  All existing
          call paths are unaffected.
        - Inside a properly-initialized worker: dynamically scales up
          and down to keep total CPU utilization at 100%.

    The counter read for `active` is intentionally lock-free.  A torn
    read can only return a stale value, which means we either over-
    allocate by 1 (harmless — extra threads sleep on OMP PASSIVE / TBB)
    or under-allocate by 1 (harmless — the next call corrects it).
    Acquiring the lock on every read would add unnecessary contention.
    The extra-claim lock IS acquired (briefly) because incrementing the
    extra-counter must be atomic to avoid over-claim.
    """
    if _BL_ACTIVE_COUNTER is None or _BL_TOTAL_CORES is None:
        return
    active = max(_BL_ACTIVE_COUNTER.value, 1)
    floor = _BL_TOTAL_CORES // active
    remainder = _BL_TOTAL_CORES - floor * active

    # Adjust extra-claim based on current remainder:
    #   - If I don't hold extra and remainder has room, try to claim.
    #   - If I hold extra but extras-in-circulation already exceeds
    #     current remainder, release.
    # Read extra-counter lock-free first to avoid taking the lock
    # unnecessarily; only acquire when an action is plausibly needed.
    if _BL_EXTRA_COUNTER is not None:
        try:
            current_extras = _BL_EXTRA_COUNTER.value
        except Exception:
            current_extras = 0
        if not _BL_I_HAVE_EXTRA:
            # Try to claim if there's room.  _try_claim_extra is
            # idempotent and re-checks under the lock, so a torn read
            # here is harmless.
            if current_extras < remainder:
                _try_claim_extra(remainder)
        else:
            # I hold an extra.  Release if the pool is over its
            # remainder budget (active grew while I was working).
            if current_extras > remainder:
                _try_release_extra()

    n = max(1, floor + (1 if _BL_I_HAVE_EXTRA else 0))
    try:
        import numba
        numba.set_num_threads(n)
    except Exception:
        # Numba import failure or thread-set failure — silently ignore.
        # This matches block_haplotypes._update_dynamic_threads's
        # exception handling, which prioritises robustness over
        # surfacing transient thread-pool errors mid-pipeline.
        pass

# Define defaults as constants
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

    This kernel folds everything into one loop nest that updates both
    mass_1_1 and mass_2_2 via online numerically-stable logaddexp,
    never materializing the 5D `combined` array.  The hom-hom case is
    handled inline by checking `u_out == u_in and v_out == v_in` per
    cell; this branch predicts well (true once per (n_c, n_n) pair).

    Algebraic structure for each (s, u_out, u_in, v_out, v_in) cell:
        combined_cell = F_batch[s, u_out, u_in] + B_batch[s, v_out, v_in]
                      + (T_matrix[u_in, v_in] if not hom-hom else 0.0)
                      + (T_matrix[u_out, v_out] if standard_bw else 0.0)
        mass_1_1[s, u_out, v_out] = logaddexp(running_1_1, combined_cell)
        mass_2_2[s, u_in,  v_in ] = logaddexp(running_2_2, combined_cell)

    Numerical equivalence: scipy's logsumexp is a two-pass (max-then-
    sum) reduction.  This kernel uses online numerically-stable
    logaddexp:
        m = max(a, b)
        result = m + log1p(exp(-|a - b|))
    which is also numerically stable.  Differences from scipy are at
    the last bit of float64 due to reduction-order; we've already
    accepted this scale of drift elsewhere (notably the
    _diploid_collapse_kernel) and validated downstream stability.

    parallel=True: the outer batch axis s is independent across
    samples, so prange over s gives ~linear speedup on a multi-core
    machine.  Per-thread work is purely local to one sample slice of
    mass_1_1 and mass_2_2.

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

    NOTE on parallel=True: prange over the batch axis is safe because
    each iteration writes to its own (s, :, :) slice of mass_1_1 and
    mass_2_2 — no cross-thread aliasing.  No locks needed.
    """
    B, n_c, _ = F_batch.shape
    _, n_n, _ = B_batch.shape

    mass_1_1 = np.full((B, n_c, n_n), -np.inf, dtype=np.float64)
    mass_2_2 = np.full((B, n_c, n_n), -np.inf, dtype=np.float64)

    for s in prange(B):
        for u_out in range(n_c):
            for u_in in range(n_c):
                F_val = F_batch[s, u_out, u_in]
                for v_out in range(n_n):
                    for v_in in range(n_n):
                        # T_partner_corrected term: T_matrix[u_in, v_in],
                        # zeroed out at hom-hom entries.
                        if u_out == u_in and v_out == v_in:
                            partner_term = 0.0
                        else:
                            partner_term = T_matrix[u_in, v_in]
                        # T_main term: T_matrix[u_out, v_out] if standard BW,
                        # else 0.
                        if use_standard_baum_welch:
                            main_term = T_matrix[u_out, v_out]
                        else:
                            main_term = 0.0
                        combined_cell = (F_val
                                         + B_batch[s, v_out, v_in]
                                         + partner_term
                                         + main_term)

                        # Online logaddexp into mass_1_1[s, u_out, v_out].
                        # Standard formula: m + log1p(exp(-|a - b|)).
                        # We inline it for tightness.
                        old11 = mass_1_1[s, u_out, v_out]
                        if old11 == -np.inf:
                            mass_1_1[s, u_out, v_out] = combined_cell
                        elif combined_cell == -np.inf:
                            pass    # unchanged
                        else:
                            m = old11 if old11 > combined_cell else combined_cell
                            mass_1_1[s, u_out, v_out] = m + np.log(
                                np.exp(old11 - m) + np.exp(combined_cell - m))

                        # Online logaddexp into mass_2_2[s, u_in, v_in].
                        # Same formula.
                        old22 = mass_2_2[s, u_in, v_in]
                        if old22 == -np.inf:
                            mass_2_2[s, u_in, v_in] = combined_cell
                        elif combined_cell == -np.inf:
                            pass
                        else:
                            m = old22 if old22 > combined_cell else combined_cell
                            mass_2_2[s, u_in, v_in] = m + np.log(
                                np.exp(old22 - m) + np.exp(combined_cell - m))

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

    if num_processes > 1 and len(blocks_to_process) > 1:
        # Parallel: build all tasks upfront (pool needs them)
        tasks = []
        for block in blocks_to_process:
            if not hasattr(block, 'positions'):
                raise ValueError(f"Encountered invalid block object in list. Type: {type(block)}")
            indices = np.searchsorted(global_site_locations, block.positions)
            block_samples = sample_probs_matrix[:, indices, :]
            tasks.append((block_samples, block, params))
        with _StdPool(num_processes) as pool:
            results = pool.map(_worker_calculate_single_block_likelihood, tasks)
        del tasks
    else:
        # Sequential: process one block at a time, free each before the next
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

    # Fix #4: Use cached keys if provided
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

    # Fix #4: Use cached keys if provided
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

    # Fix #2: Use pre-computed uniform prior if provided
    if uniform_prior is None:
        prior_a_posteriori = initial_transition_probabilities(haps_data, space_gap=space_gap)
    else:
        prior_a_posteriori = uniform_prior

    # Fix #4: Use pre-computed keys cache if provided
    if hap_keys_cache is None:
        hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in haps_data]

    full_samples_likelihoods = full_samples_data
    num_samples = len(full_samples_likelihoods)
    num_blocks = len(full_blocks_likelihoods)
    
    # Fix #6: Use pre-restructured emissions if provided
    if all_block_likelihoods_by_sample is None:
        all_block_likelihoods_by_sample = []
        for s in range(num_samples):
            sample_chain = []
            for b in range(num_blocks):
                sample_chain.append(full_blocks_likelihoods[b][s])
            all_block_likelihoods_by_sample.append(sample_chain)

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

    # Collect dense forward/backward matrices alongside the dict outputs.
    # Each entry forward_dense[s] is a dict {block_idx: dense_matrix} that
    # _run_batched_pass reads directly to populate F_tensor / B_tensor,
    # avoiding O(K^2) dict lookups per (sample, block).
    forward_dense = [{} for _ in range(num_samples)]
    backward_dense = [{} for _ in range(num_samples)]

    # 1. E-Step: Forward Pass
    forward_nums = [get_full_probs_forward(
                        full_samples_data[i],
                        sample_sites, haps_data, current_transition_probs, 
                        all_block_likelihoods_by_sample[i], space_gap=space_gap,
                        hap_keys_cache=hap_keys_cache,
                        T_per_block_cache=T_fwd_cache,
                        dense_matrices_out=forward_dense[i]
                    ) for i in range(num_samples)]
    

    # Calculate Total Data Log Likelihood (summing full grid)
    total_data_log_likelihood = 0.0
    last_block_idx = len(haps_data) - 1
    
    for s in range(num_samples):
        final_states_log_probs = list(forward_nums[s][last_block_idx].values())
        if final_states_log_probs:
            total_data_log_likelihood += analysis_utils.lse_scalar(final_states_log_probs)

    # 1. E-Step: Backward Pass
    backward_nums = [get_full_probs_backward(
                        full_samples_data[i],
                        sample_sites, haps_data, current_transition_probs, 
                        all_block_likelihoods_by_sample[i], space_gap=space_gap,
                        hap_keys_cache=hap_keys_cache,
                        T_per_block_cache=T_bwd_cache,
                        dense_matrices_out=backward_dense[i]
                    ) for i in range(num_samples)]
    
    
    samples_probs = list(zip(forward_nums, backward_nums))
    # Parallel structure of dense matrices: samples_dense[s] = (forward_dense[s], backward_dense[s])
    # Each is a dict {block_idx -> (n_haps, n_haps) dense log-probability matrix}.
    # _run_batched_pass uses these for direct dense-matrix access in
    # F_tensor / B_tensor population, bypassing the O(K^2) dict lookups
    # that the dict-encoded `samples_probs` requires.
    samples_dense = list(zip(forward_dense, backward_dense))
    
    
    # 2. M-Step: Vectorized Update (Baum-Welch ξ calculation)
    def _run_batched_pass(indices, is_forward):
        new_transition_probs = {}
        dir_idx = 0 if is_forward else 1
        
        for i in indices:
            # Dynamic thread reallocation: fine-grained re-check per
            # block, so a straggler in the middle of an EM iteration's
            # forward or backward pass picks up newly-freed cores
            # without waiting until the next EM iteration's top-of-loop
            # check.  At K=10 with ~100 blocks/gap × 10 EM iterations,
            # this fires ~2000x per gap; each call is a couple of µs
            # (atomic read + numba.set_num_threads), so total overhead
            # is well under 10ms per gap — negligible against the
            # tens-of-seconds-per-gap EM cost.
            _update_dynamic_threads()

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
            # We use the pre-computed dense matrices (samples_dense) instead
            # of the dict-encoded samples_probs.  The dense path is identical
            # in math: forward_nums[s][i][((i, hap_keys[r]), (i, hap_keys[c]))]
            # equals forward_dense[s][i][r, c] by construction (see the
            # final result-dict-building loop in get_full_probs_forward).
            # Going dense saves O(K^2) per-cell dict lookups per (sample,
            # block) pair, which adds up to (num_samples * num_blocks * K^2)
            # extra dict ops per EM iteration in the legacy path.
            F_tensor = np.full((num_samples, n_curr, n_curr), -np.inf)
            B_tensor = np.full((num_samples, n_next, n_next), -np.inf)

            for s in range(num_samples):
                if is_forward:
                    fwd_dense = samples_dense[s][0][i]            # (n_curr, n_curr)
                    bwd_dense = samples_dense[s][1][next_bundle]  # (n_next, n_next)
                else:
                    fwd_dense = samples_dense[s][1][i]            # (n_curr, n_curr) — backward output for block i
                    bwd_dense = samples_dense[s][0][next_bundle]  # (n_next, n_next) — forward output for block next_bundle

                # The dense matrices already encode the same (out, in)
                # axes that F_tensor / B_tensor expect, so the assignment
                # is a single bulk copy with no per-cell dict work.
                F_tensor[s] = fwd_dense
                B_tensor[s] = bwd_dense
                
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
            for start_idx in range(0, num_samples, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                
                F_batch = np.ascontiguousarray(F_tensor[start_idx:end_idx])
                B_batch = np.ascontiguousarray(B_tensor[start_idx:end_idx])

                mass_1_1, mass_2_2 = _batched_baum_welch_mass_kernel(
                    F_batch, B_batch,
                    np.ascontiguousarray(T_matrix, dtype=np.float64),
                    bool(use_standard_baum_welch))

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

    # Fix #2: Compute uniform prior ONCE before EM loop (never changes)
    uniform_prior = initial_transition_probabilities(haps_data, space_gap=space_gap)
    
    # Fix #4: Cache sorted hap keys ONCE (never change)
    hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in haps_data]
    
    # Fix #6: Restructure emissions ONCE (never change)
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
        # Dynamic thread reallocation: re-check the live count of active
        # peer workers and rescale numba threads.  When this gap-worker
        # is a straggler (peer workers have finished their gaps and
        # exited from _gap_worker), the counter drops and this call
        # scales us up accordingly.  Inside parallel=True numba kernels
        # called downstream (notably _batched_baum_welch_mass_kernel),
        # prange will then use the new thread count.
        #
        # No-op when running outside a pool (sequential path or direct
        # call) — _update_dynamic_threads short-circuits on the None
        # globals.  See Portion 1's helper docstring for details.
        _update_dynamic_threads()

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
    Unpacks arguments and calls the calculation function for multiprocessing.
    Reads emissions from _BL_SHARED (set by Pool initializer) to avoid pickle overhead.

    Dynamic thread reallocation (when wired up by the caller — i.e. when
    _BL_ACTIVE_COUNTER and _BL_TOTAL_CORES are set by _init_bl_shared):
      - On entry: atomically increment the active-worker counter, then
        compute floor = total_cores // active and remainder = total_cores
        % active.  Try to claim an "extra" thread from the remainder
        pool via _try_claim_extra.  Call set_num_threads(floor + (1 if
        extra claimed else 0)).  This gives the new worker its share of
        the remainder distribution — exactly `remainder` workers will
        end up with floor+1, the rest with floor, summing to exactly
        total_cores.
      - On exit (via try/finally so crashes still decrement): release
        the extra (if held) and decrement the active counter, freeing
        the worker's share for any remaining stragglers.

    When dynamic-thread wiring is absent (counter is None — the default
    state before Portion 2's caller changes take effect, or when
    generate_transition_probability_mesh runs sequentially), all the
    counter machinery is bypassed entirely and the function behaves
    exactly as the pre-Portion-2 version.
    """
    # Fix #3: Emissions read from shared data, not passed per task
    (gap, full_samples, sites, haps, max_iter, min_ll, lr, use_std_bw) = args

    # Track whether we successfully incremented so the finally clause
    # only decrements if the increment actually happened.  This avoids
    # a spurious decrement if the counter happens to be None.
    counter_inc = False
    if _BL_ACTIVE_COUNTER is not None and _BL_TOTAL_CORES is not None:
        try:
            import numba as _numba
            # Increment under the counter's lock — this is the one place
            # we MUST hold the lock, since two workers picking up tasks
            # near-simultaneously could otherwise race on the read-
            # modify-write.  Reads in _update_dynamic_threads stay lock-
            # free (torn-read tolerant; see its docstring).
            with _BL_ACTIVE_COUNTER.get_lock():
                _BL_ACTIVE_COUNTER.value += 1
            counter_inc = True
            active = max(_BL_ACTIVE_COUNTER.value, 1)
            floor = _BL_TOTAL_CORES // active
            remainder = _BL_TOTAL_CORES - floor * active
            # Try to claim an extra thread from the remainder pool.
            # _try_claim_extra is a no-op when _BL_EXTRA_COUNTER is
            # None (legacy callers without remainder distribution) —
            # in that case behavior falls back to floor-only allocation
            # exactly as before.
            _try_claim_extra(remainder)
            n_threads = max(1, floor + (1 if _BL_I_HAVE_EXTRA else 0))
            _numba.set_num_threads(n_threads)
        except Exception:
            # Numba unavailable or any other transient issue — fall
            # through and run the task without dynamic scaling.  The
            # counter is still decremented (if incremented) so peers
            # see correct activity, but this worker stays at whatever
            # thread count it had.
            pass

    try:
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
    finally:
        # Always release any held extra and decrement active counter,
        # regardless of whether the body succeeded.  Order matters:
        # release the extra FIRST so peers see the freed extra-slot
        # before they see the decremented active count (otherwise a
        # peer rechecking between our two decrements might try to claim
        # an extra that's not yet released).  In practice the
        # interleaving is harmless (claims are bounded by remainder),
        # but releasing-first is the cleanest invariant.
        _try_release_extra()
        if counter_inc and _BL_ACTIVE_COUNTER is not None:
            try:
                with _BL_ACTIVE_COUNTER.get_lock():
                    _BL_ACTIVE_COUNTER.value -= 1
            except Exception:
                # Counter lock acquisition failed — very unlikely (only
                # happens on broken multiprocessing state).  Silently
                # ignore; the lost decrement only causes peers to slightly
                # under-scale on their next _update_dynamic_threads call,
                # which is a benign performance issue rather than a
                # correctness issue.
                pass


def _gap_worker_tagged(args):
    """Wraps _gap_worker for use with pool.imap_unordered.

    imap_unordered returns results in completion order, not input
    order, so we tag each result with the gap-size it corresponds to.
    The parent re-assembles results by gap-size to restore the
    {gap: result} mapping that downstream code expects.

    `args[0]` is the gap-size — see the worker_args construction in
    generate_transition_probability_mesh.  We pass args through to
    _gap_worker unchanged so all of Portion 2's counter machinery
    operates exactly as before.

    Used ONLY by the pool path.  The sequential path calls _gap_worker
    directly and relies on positional ordering, so no tagging needed
    there.
    """
    gap = args[0]
    return (gap, _gap_worker(args))


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
    
    Uses Pool initializer to share emissions across workers (Fix #3),
    avoiding pickling the full_blocks_likelihoods object once per gap task.
    
    Args:
        full_samples_data (list): Sample data.
        sample_sites (np.ndarray): Genomic locations.
        haps_data (list): List of BlockResult objects.
        max_num_iterations (int): EM iterations per gap size.
        use_standard_baum_welch (bool): 
            If True: Uses standard update (sensitive to initialization/priors).
            If False: Uses Reset update (recommended for Viterbi/Hard EM).
        num_processes (int): Number of parallel processes. Use 1 for sequential
                            execution (required when called from worker processes).
        
    Returns:
        TransitionMesh: The fully populated mesh of transition probabilities.
    """
    
    full_blocks_likelihoods = generate_all_block_likelihoods(
        full_samples_data, sample_sites, haps_data, num_processes=1
    )
    _malloc_trim()
    
    max_gap = len(haps_data) - 1
    gaps = list(range(1, max_gap + 1))

    # Fix #3: Tasks carry only lightweight args — emissions shared via initializer
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

    if num_processes > 1:
        # Dynamic thread reallocation: create an atomic counter shared
        # across all workers.  Each worker increments it on task entry
        # and decrements on task exit (in _gap_worker); the counter's
        # live value drives numba.set_num_threads(total // active) so
        # that as faster gaps finish, the stragglers automatically scale
        # up to use the freed cores.
        #
        # Counter creation uses the SAME context as the pool
        # (_forkserver_ctx).  Mixing a fork-context Value with a
        # forkserver-context Pool produces undefined behavior because
        # the synchronisation primitives behind .get_lock() use
        # context-specific shared-memory mechanisms.  We must use
        # _forkserver_ctx.Value here to match _ForkserverPool below.
        active_counter = _forkserver_ctx.Value('i', 0)

        # Extra-thread counter for remainder distribution.  When
        # num_processes is not evenly divisible by active workers, the
        # floor allocation leaves `remainder = total % active` cores
        # idle (e.g. total=112, active=76: 36 idle).  The extra
        # counter tracks how many workers currently hold a +1 thread
        # so that exactly `remainder` workers get floor+1 and the rest
        # get floor, summing to total.  See _try_claim_extra /
        # _try_release_extra and _update_dynamic_threads' docstrings.
        # Same forkserver context as active_counter for the same
        # shared-memory consistency reason.
        extra_counter = _forkserver_ctx.Value('i', 0)

        # Belt-and-suspenders: clear __main__.__file__ to prevent the
        # forkserver from re-executing the entry script.  Mirrors the
        # safeguard in block_haplotypes._build_block_haplotypes_parallel.
        # If the forkserver process has already been started earlier in
        # this Python session (e.g. by a prior block_haplotypes run),
        # this is a no-op — preload state is locked in.  If we're the
        # first user of the forkserver, this prevents the forkserver
        # process from re-running the user's entry script during its
        # bootstrap, which would otherwise cause infinite recursion or
        # unexpected side effects.
        import sys as _sys
        _main_mod = _sys.modules.get('__main__')
        _saved_main_file = getattr(_main_mod, '__file__', None)
        _saved_main_spec = getattr(_main_mod, '__spec__', None)
        if _main_mod is not None:
            if hasattr(_main_mod, '__file__'):
                del _main_mod.__file__
            _main_mod.__spec__ = None

        try:
            # Use imap_unordered(chunksize=1) instead of pool.map.  Reasons:
            #
            #   1. pool.map pre-divides worker_args into chunks of size
            #      max(1, len(args) // (4 * num_processes)) and pre-assigns
            #      one chunk per worker.  A worker that finishes its chunk
            #      goes idle while peers still have queued tasks.  This
            #      defeats the dynamic-thread reallocation: the counter
            #      shows N-1 active workers but the freshly-idle worker
            #      can't pick up another straggler-prone task.
            #
            #   2. imap_unordered dispatches tasks ONE AT A TIME from the
            #      master queue.  A worker that finishes a fast gap
            #      immediately pulls the next gap, so all num_processes
            #      workers stay active until only the truly last few
            #      stragglers remain.  At that point the counter drops
            #      naturally and the surviving workers' next
            #      _update_dynamic_threads() (added in Portion 4) sees a
            #      lower active count and scales up their numba threads.
            #
            # imap_unordered returns results in completion order, so we use
            # _gap_worker_tagged which wraps each result with its gap-size.
            # We collect into a dict keyed by gap-size and re-pair to the
            # original gaps list afterwards — preserving the
            # {gap: result} mapping the downstream code expects.
            #
            # Pool class: _ForkserverPool, NOT _StdPool.  The parent
            # process in production typically reaches this point AFTER
            # having JIT-compiled parallel=True numba kernels (notably
            # _batched_baum_welch_mass_kernel inside any earlier
            # calculate_hap_transition_probabilities call), which
            # initialises GNU OpenMP.  Forking the worker pool from such
            # a parent crashes workers with "fork() called from a process
            # already using GNU OpenMP, this is unsafe."  Forkserver
            # workers spawn from a clean intermediate process that never
            # touched OpenMP.  set_forkserver_preload (set in
            # thread_config.py) ensures block_linking and its deps are
            # pre-imported in the forkserver process, so worker startup
            # stays sub-second.
            results_by_gap = {}
            with _ForkserverPool(num_processes, initializer=_init_bl_shared,
                                 initargs=(shared_data, active_counter, num_processes,
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

        # Re-order results to match gaps list (positional order expected
        # by the dict(zip(gaps, results)) below).
        results = [results_by_gap[g] for g in gaps]
    else:
        # Sequential execution — set shared data directly.  No pool, so
        # no active_counter wiring is meaningful: a single sequential
        # caller is always "active=1" and dynamic reallocation would
        # have nothing to react to.  Passing None for both leaves the
        # globals at their defaults and _gap_worker's counter machinery
        # short-circuits to a no-op (exactly as in Portion 1).
        _init_bl_shared(shared_data)
        results = []
        for args in worker_args:
            results.append(_gap_worker(args))
            _malloc_trim()
    
    del full_blocks_likelihoods, worker_args, shared_data
    _malloc_trim()
    
    mesh_dict = dict(zip(gaps, results))
    return TransitionMesh(mesh_dict)