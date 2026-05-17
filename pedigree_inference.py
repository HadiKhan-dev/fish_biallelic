"""
pedigree_inference.py

Pedigree inference using tolerance paintings from paint_samples.py.

- Uses SampleTolerancePainting/SampleConsensusPainting from paint_samples.py
- When scoring parent-child relationships, considers ALL consensus paintings
- Score = max over all (parent_consensus, child_consensus) combinations
- This handles uncertainty in founder assignment properly

The 16-state HMM and free switches in homozygous regions are preserved.
"""
import thread_config

import numpy as np
import pandas as pd
import warnings
import math
import os
from itertools import combinations, product
from tqdm import tqdm
from sklearn.cluster import KMeans

# Visualization Imports
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_VIS = True
except ImportError:
    HAS_VIS = False

warnings.filterwarnings("ignore")

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Pedigree inference will be slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# =============================================================================
# SHARED DATA FOR POOL WORKERS (avoids pickling large objects per task)
# =============================================================================

_PEDIGREE_SHARED = {}

# ---------------------------------------------------------------------------
# Two-step dynamic thread reallocation for straggler tasks.
#
# Adapted from the same pattern in block_haplotypes.py and block_linking.py
# (May 2026).  The setup:
#   - When a worker enters a task, it atomically increments
#     _PEDIGREE_ACTIVE_COUNTER; on exit it decrements.  At any moment the
#     counter holds the number of workers currently processing a task.
#   - Each worker computes its fair-share thread budget as
#         floor     = total_cores // active
#         remainder = total_cores % active
#         my_share  = floor + (1 if I hold an "extra" else 0)
#     and calls numba.set_num_threads(my_share).
#   - At most `remainder` workers can hold an extra at any time; they
#     claim/release atomically via _PEDIGREE_EXTRA_COUNTER.  This makes
#     the per-worker allocation EXACTLY total_cores in total -- no idle
#     cores -- even when total_cores % active != 0.
#   - As peer workers finish (active drops), in-flight stragglers can
#     re-check at periodic hooks (`_update_dynamic_threads()`) inserted
#     inside long-running task bodies, claim newly-available extras, and
#     scale up.  The reverse also works: if active grows, holders that
#     are now over-budget release their extras.
#
# All four globals are None outside a pool worker (or in a pool that
# wasn't initialized with counters) -- in that case the helpers and the
# worker entry/exit wrapping are no-ops, preserving the pre-existing
# behavior exactly for any legacy caller.  The batched HMM kernels work
# regardless: they use prange but with numba.set_num_threads(1) they
# behave as a single-threaded for-loop (numba's prange tolerates a
# single-thread schedule with no measurable overhead vs. range).
# ---------------------------------------------------------------------------
_PEDIGREE_ACTIVE_COUNTER = None
_PEDIGREE_EXTRA_COUNTER = None
_PEDIGREE_TOTAL_CORES = None
_PEDIGREE_I_HAVE_EXTRA = False


def _try_claim_extra(remainder):
    """Atomically attempt to claim an extra thread from the remainder pool.

    Returns True if successfully claimed (and sets _PEDIGREE_I_HAVE_EXTRA
    True as a side effect).  Returns False if the remainder pool is
    already exhausted (current extra-count >= remainder).

    Atomicity is provided by _PEDIGREE_EXTRA_COUNTER.get_lock(); without
    the lock, two workers could both pass the "< remainder" check and
    over-claim, oversubscribing CPU.

    Idempotent: returns True without re-claiming if _PEDIGREE_I_HAVE_EXTRA
    is already True (each worker holds at most one extra at any time).
    """
    global _PEDIGREE_I_HAVE_EXTRA
    if _PEDIGREE_I_HAVE_EXTRA:
        return True
    if _PEDIGREE_EXTRA_COUNTER is None:
        return False
    try:
        with _PEDIGREE_EXTRA_COUNTER.get_lock():
            if _PEDIGREE_EXTRA_COUNTER.value < remainder:
                _PEDIGREE_EXTRA_COUNTER.value += 1
                _PEDIGREE_I_HAVE_EXTRA = True
                return True
    except Exception:
        # Lock acquisition or counter mutation failed -- fall back to
        # floor-only.  Same robustness posture as _update_dynamic_threads.
        pass
    return False


def _try_release_extra():
    """Atomically release this worker's extra claim, if held.

    Returns True if a claim was released.  Used (a) at worker exit via
    try/finally (release on the way out so the pool stays accurate)
    and (b) in _update_dynamic_threads when active grew and the
    remainder shrank below the current extras-in-circulation count.
    """
    global _PEDIGREE_I_HAVE_EXTRA
    if not _PEDIGREE_I_HAVE_EXTRA:
        return False
    if _PEDIGREE_EXTRA_COUNTER is None:
        _PEDIGREE_I_HAVE_EXTRA = False  # defensive
        return False
    try:
        with _PEDIGREE_EXTRA_COUNTER.get_lock():
            _PEDIGREE_EXTRA_COUNTER.value -= 1
            _PEDIGREE_I_HAVE_EXTRA = False
            return True
    except Exception:
        # Defensive: still clear the local flag even if counter mutation
        # failed.  Otherwise the worker thinks it holds an extra forever,
        # leading to under-claim by peers on subsequent rechecks.
        _PEDIGREE_I_HAVE_EXTRA = False
        return False


def _update_dynamic_threads():
    """Recheck active worker count and rescale this worker's numba threads.

    Called from inside long-running task bodies (top of the contig
    loop in `_score_trios_batch`, `_score_pairs_by_children`, and
    `_check_trio_consistency_worker`) so that stragglers re-evaluate
    their thread budget as peers finish.

    Computes:
        floor     = total_cores // active
        remainder = total_cores % active
        my_share  = floor + (1 if I currently hold an extra else 0)
    and calls numba.set_num_threads(my_share).

    Also adjusts the extra-claim status:
      - If I don't hold an extra and the pool has room
        (current_extras < remainder), try to claim one.
      - If I hold an extra but extras-in-circulation already exceeds
        the current remainder budget, release it.

    Behavior:
      - Outside a pool worker, or in a pool not initialized with
        counters: silent no-op.
      - Inside a properly-initialized worker: rebalances live as peer
        activity changes.

    The counter read for `active` is intentionally lock-free.  A torn
    read can only return a stale value; either over- or under-allocate
    by 1, both harmless (extra threads sleep on OMP PASSIVE / next call
    corrects).  The extra-claim lock IS acquired briefly because the
    extras counter must be mutated atomically.
    """
    if _PEDIGREE_ACTIVE_COUNTER is None or _PEDIGREE_TOTAL_CORES is None:
        return
    active = max(_PEDIGREE_ACTIVE_COUNTER.value, 1)
    floor = _PEDIGREE_TOTAL_CORES // active
    remainder = _PEDIGREE_TOTAL_CORES - floor * active

    # Adjust extra-claim if needed.  Lock-free read first to avoid the
    # lock when no action is plausibly required.
    if _PEDIGREE_EXTRA_COUNTER is not None:
        try:
            current_extras = _PEDIGREE_EXTRA_COUNTER.value
        except Exception:
            current_extras = 0
        if not _PEDIGREE_I_HAVE_EXTRA:
            if current_extras < remainder:
                _try_claim_extra(remainder)
        else:
            if current_extras > remainder:
                _try_release_extra()

    n = max(1, floor + (1 if _PEDIGREE_I_HAVE_EXTRA else 0))
    try:
        import numba
        numba.set_num_threads(n)
    except Exception:
        # Numba import or set_num_threads failure -- silently ignore.
        # Same robustness posture as the helpers above; the next call
        # will retry.
        pass


def _init_pedigree_shared(shared_dict, active_counter=None, total_cores=None,
                          extra_counter=None):
    """Pool initializer: store shared data in worker's global scope.

    Args:
        shared_dict: dict of large objects to share across workers via
            fork-COW (e.g. contig_caches, contig_data_list, total_switches).
            Stored in module-global _PEDIGREE_SHARED.
        active_counter: optional multiprocessing.Value('i', 0) shared
            across workers.  When provided, workers increment on task
            entry and decrement on task exit, and _update_dynamic_threads
            uses it to compute fair-share allocation.  Default None
            (preserves the pre-existing behavior bit-identically).
        total_cores: optional int -- the original n_workers budget for
            the entire pool.  When provided alongside active_counter,
            configures the per-worker numba pool ceiling to total_cores
            and starts the worker at 1 thread; subsequent
            _update_dynamic_threads() calls scale up to total_cores
            when peers free their share.  Default None.
        extra_counter: optional multiprocessing.Value('i', 0) shared
            across workers, used for remainder distribution (see the
            module-level globals comment).  Default None, in which
            case workers fall back to floor-only allocation.

    Backwards-compatible: callers passing only shared_dict get the
    original pre-counter behavior exactly.
    """
    global _PEDIGREE_SHARED
    global _PEDIGREE_ACTIVE_COUNTER, _PEDIGREE_EXTRA_COUNTER
    global _PEDIGREE_TOTAL_CORES, _PEDIGREE_I_HAVE_EXTRA
    _PEDIGREE_SHARED.clear()
    _PEDIGREE_SHARED.update(shared_dict)
    _PEDIGREE_ACTIVE_COUNTER = active_counter
    _PEDIGREE_TOTAL_CORES = total_cores
    _PEDIGREE_EXTRA_COUNTER = extra_counter
    # Defensive: ensure no stale claim carries into the new worker context.
    # fork() copies the parent's globals; if the parent had set
    # _PEDIGREE_I_HAVE_EXTRA=True before forking (shouldn't happen, but
    # defensive), the child must start with a clean slate.
    _PEDIGREE_I_HAVE_EXTRA = False

    # When dynamic threads are wired up, configure the per-worker numba
    # pool ceiling to total_cores so set_num_threads() can later scale
    # freely.  Starting at 1 thread is the safe default -- the first
    # worker entry into a task will immediately rescale to floor+extra
    # based on the live active count.  Mirrors block_linking._init_bl_shared.
    if total_cores is not None:
        try:
            import os as _os, numba as _numba
            _os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
            _numba.config.NUMBA_NUM_THREADS = total_cores
            _numba.set_num_threads(1)
        except Exception:
            # Numba unavailable or thread-set failure -- silently ignore,
            # same robustness posture as _update_dynamic_threads.
            pass


def _check_trio_consistency_worker(args):
    """
    Worker: check one trio's consistency across all chromosomes.
    Reads tolerance paintings from _PEDIGREE_SHARED (fork COW).
    Returns (df_idx, child_name, mean_explained).

    Dynamic thread reallocation (when the pool was initialized with
    counters): on entry, atomically increment _PEDIGREE_ACTIVE_COUNTER
    and claim an extra thread if remainder allows; rescale numba
    threads to floor+extra.  On exit, release the extra and decrement.
    Inside the per-contig loop, _update_dynamic_threads() lets the
    worker rebalance as peers finish.

    Math is unchanged: the per-contig _check_trio_on_chromosome call
    and the final np.mean(fracs) are identical to the pre-wrapping
    implementation.  The counter-management adds wall-clock work but
    no numerical effect.
    """
    # Track whether we successfully incremented so the finally clause
    # only decrements if the increment actually happened.  This avoids
    # a spurious decrement if counters are None (legacy path).
    counter_inc = False
    if _PEDIGREE_ACTIVE_COUNTER is not None and _PEDIGREE_TOTAL_CORES is not None:
        try:
            import numba as _numba
            with _PEDIGREE_ACTIVE_COUNTER.get_lock():
                _PEDIGREE_ACTIVE_COUNTER.value += 1
            counter_inc = True
            active = max(_PEDIGREE_ACTIVE_COUNTER.value, 1)
            floor = _PEDIGREE_TOTAL_CORES // active
            remainder = _PEDIGREE_TOTAL_CORES - floor * active
            _try_claim_extra(remainder)
            n_threads = max(1, floor + (1 if _PEDIGREE_I_HAVE_EXTRA else 0))
            _numba.set_num_threads(n_threads)
        except Exception:
            # Numba unavailable or any other transient issue -- fall
            # through and run the task without dynamic scaling.
            pass

    try:
        df_idx, child_name, child_i, p1_i, p2_i = args
        contig_data_list = _PEDIGREE_SHARED['contig_data_list']
        step = _PEDIGREE_SHARED['step']

        fracs = []
        for contig_data in contig_data_list:
            # In-loop rebalance hook: as peer workers finish, this
            # straggler picks up their freed-share by reclaiming
            # extras.  Cost is one atomic counter read (~ns) per
            # contig iteration.
            _update_dynamic_threads()

            tol = contig_data['tolerance_painting']
            child_tol = tol[child_i]
            p1_tol = tol[p1_i]
            p2_tol = tol[p2_i]

            child_path = child_tol if child_tol.chunks else None
            p1_path = p1_tol if p1_tol.chunks else None
            p2_path = p2_tol if p2_tol.chunks else None

            if child_path and p1_path and p2_path:
                frac = _check_trio_on_chromosome(child_path, p1_path, p2_path, step=step)
            else:
                frac = 0.0
            fracs.append(frac)

        return df_idx, child_name, float(np.mean(fracs))
    finally:
        # Always release any held extra and decrement active count,
        # regardless of whether the body succeeded.  Release-first so
        # peers see the freed extra-slot before the decremented active.
        _try_release_extra()
        if counter_inc and _PEDIGREE_ACTIVE_COUNTER is not None:
            try:
                with _PEDIGREE_ACTIVE_COUNTER.get_lock():
                    _PEDIGREE_ACTIVE_COUNTER.value -= 1
            except Exception:
                # Counter lock acquisition failed -- very unlikely.
                # Silently ignore; the lost decrement only causes peers
                # to slightly under-scale on their next rebalance call,
                # benign perf issue rather than a correctness issue.
                pass


def _check_trio_on_chromosome(child_painting, p1_painting, p2_painting, step=1000):
    """Check fraction of child's genome explained by the two parents on one chromosome.

    Public signature preserved for backward compatibility.  The hot inner loop
    is delegated to the numba kernel `_check_trio_kernel` after converting each
    painting's chunk list (Python objects with .start/.end/.hap1/.hap2 attrs)
    into four flat numpy arrays.

    The math is bit-identical to the previous pure-Python implementation:
      - same step-based position iteration `range(start, end, step)`
      - same chunk lookup semantics (first chunk containing pos, -1 if none)
      - same skip-on-missing rule (continue if any of 6 alleles is -1)
      - same "from_a/from_b set membership" inheritance check
      - same `explained / total` (with total > 0 guard) return
    """
    if not child_painting.chunks:
        return 0.0
    c_starts, c_ends, c_h1, c_h2 = _painting_to_arrays(child_painting)
    a_starts, a_ends, a_h1, a_h2 = _painting_to_arrays(p1_painting)
    b_starts, b_ends, b_h1, b_h2 = _painting_to_arrays(p2_painting)
    start = int(c_starts[0])
    end = int(c_ends[-1])
    return float(_check_trio_kernel(
        c_starts, c_ends, c_h1, c_h2,
        a_starts, a_ends, a_h1, a_h2,
        b_starts, b_ends, b_h1, b_h2,
        start, end, int(step),
    ))


def _painting_to_arrays(painting):
    """Convert a painting's chunk list to four parallel numpy arrays
    (starts, ends, hap1, hap2) suitable for numba kernels.

    A zero-chunk painting yields four empty arrays; the kernel handles that
    case (the inner chunk-search loops simply find no match and the position
    is skipped via the -1 sentinel).
    """
    n = len(painting.chunks)
    starts = np.empty(n, dtype=np.int64)
    ends = np.empty(n, dtype=np.int64)
    h1 = np.empty(n, dtype=np.int64)
    h2 = np.empty(n, dtype=np.int64)
    for i in range(n):
        c = painting.chunks[i]
        starts[i] = c.start
        ends[i] = c.end
        h1[i] = c.hap1
        h2[i] = c.hap2
    return starts, ends, h1, h2


@njit(fastmath=True, cache=True)
def _check_trio_kernel(
    c_starts, c_ends, c_h1, c_h2,
    a_starts, a_ends, a_h1, a_h2,
    b_starts, b_ends, b_h1, b_h2,
    start, end, step,
):
    """Numba inner loop for _check_trio_on_chromosome.

    Iterates positions `pos in range(start, end, step)`.  At each position,
    performs a linear scan over each parent's chunk list to find the chunk
    containing `pos`; if found, records (hap1, hap2) into local scalars,
    else sentinel -1.  Positions with any -1 are skipped.  Otherwise:
      `from_a = {a1, a2}`, `from_b = {b1, b2}`
      explained iff (ch1 in from_a and ch2 in from_b) or (ch1 in from_b and ch2 in from_a)
    Returns explained / total (0.0 if total == 0).
    """
    explained = 0
    total = 0
    n_c = c_starts.shape[0]
    n_a = a_starts.shape[0]
    n_b = b_starts.shape[0]
    for pos in range(start, end, step):
        ch1 = -1
        ch2 = -1
        for k in range(n_c):
            if c_starts[k] <= pos < c_ends[k]:
                ch1 = c_h1[k]
                ch2 = c_h2[k]
                break
        a1 = -1
        a2 = -1
        for k in range(n_a):
            if a_starts[k] <= pos < a_ends[k]:
                a1 = a_h1[k]
                a2 = a_h2[k]
                break
        b1 = -1
        b2 = -1
        for k in range(n_b):
            if b_starts[k] <= pos < b_ends[k]:
                b1 = b_h1[k]
                b2 = b_h2[k]
                break
        if ch1 < 0 or ch2 < 0 or a1 < 0 or a2 < 0 or b1 < 0 or b2 < 0:
            continue
        total += 1
        # from_a = {a1, a2}, from_b = {b1, b2}; explained iff
        # (ch1 in from_a and ch2 in from_b) or (ch1 in from_b and ch2 in from_a)
        ch1_in_a = (ch1 == a1) or (ch1 == a2)
        ch1_in_b = (ch1 == b1) or (ch1 == b2)
        ch2_in_a = (ch2 == a1) or (ch2 == a2)
        ch2_in_b = (ch2 == b1) or (ch2 == b2)
        if (ch1_in_a and ch2_in_b) or (ch1_in_b and ch2_in_a):
            explained += 1
    if total > 0:
        return explained / total
    return 0.0

# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

class PedigreeResult:
    def __init__(self, samples, relationships, parent_candidates, 
                 recombination_map, systematic_errors, kinship_matrix, ibd0_matrix,
                 trio_scores=None, total_bins=0):
        self.samples = samples
        self.relationships = relationships 
        self.parent_candidates = parent_candidates 
        self.recombination_map = recombination_map 
        self.systematic_errors = systematic_errors 
        self.kinship_matrix = kinship_matrix
        self.ibd0_matrix = ibd0_matrix
        self.trio_scores = trio_scores if trio_scores is not None else {}
        self.total_bins = total_bins

    def _recalculate_generations(self):
        self.relationships['Generation'] = 'Unknown'
        is_root = self.relationships['Parent1'].isna()
        self.relationships.loc[is_root, 'Generation'] = 'F1'
        name_to_gen = dict(zip(self.relationships['Sample'], self.relationships['Generation']))
        for _ in range(10):
            updates = 0
            for idx, row in self.relationships.iterrows():
                if row['Generation'] != 'Unknown': continue
                p1 = row['Parent1']
                p2 = row['Parent2']
                if p1 in name_to_gen and p2 in name_to_gen:
                    g1_str = name_to_gen[p1]
                    g2_str = name_to_gen[p2]
                    if g1_str != 'Unknown' and g2_str != 'Unknown':
                        try:
                            g1 = int(g1_str[1:])
                            g2 = int(g2_str[1:])
                            new_gen_num = max(g1, g2) + 1
                            new_gen_str = f"F{new_gen_num}"
                            self.relationships.at[idx, 'Generation'] = new_gen_str
                            name_to_gen[row['Sample']] = new_gen_str
                            updates += 1
                        except:
                            pass
            if updates == 0:
                break

    def perform_automatic_cutoff(self, force_clusters=2, sigma_threshold=5.0):
        if self.total_bins == 0: return
        scores = []
        indices = []
        for i, s in enumerate(self.samples):
            raw = self.trio_scores.get(s, -1e9)
            norm_score = raw / self.total_bins
            scores.append(norm_score)
            indices.append(i)
        scores_arr = np.array(scores).reshape(-1, 1)
        try:
            kmeans = KMeans(n_clusters=force_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scores_arr)
            centers = kmeans.cluster_centers_.flatten()
            good_cluster_idx = np.argmax(centers)
            good_scores = scores_arr[labels == good_cluster_idx]
            if len(good_scores) < 2:
                cutoff = np.mean(centers)
                print(f"[Auto-Cutoff] Not enough good samples for stats. Using midpoint: {cutoff:.4f}")
            else:
                mean_good = np.mean(good_scores)
                std_good = np.std(good_scores)
                cutoff = mean_good - (sigma_threshold * std_good)
                bad_center = centers[1-good_cluster_idx]
                if cutoff < bad_center:
                    cutoff = (mean_good + bad_center) / 2
                    print("[Auto-Cutoff] Sigma cutoff too wide, reverting to midpoint.")
            print(f"\n[Auto-Cutoff] Centers: {centers}")
            print(f"[Auto-Cutoff] Good Cluster Stats: Mean={np.mean(good_scores):.4f}, Std={np.std(good_scores):.4f}")
            print(f"[Auto-Cutoff] Calculated Threshold (Mean - {sigma_threshold}*Std): {cutoff:.4f}")
            updates = 0
            for i, score in zip(indices, scores):
                if score < cutoff:
                    self.relationships.at[i, 'Parent1'] = None
                    self.relationships.at[i, 'Parent2'] = None
                    updates += 1
            print(f"[Auto-Cutoff] Reclassified {updates} samples as Founders/F1 (Score < {cutoff:.4f}).")
            self._recalculate_generations()
        except Exception as e:
            print(f"[Auto-Cutoff] Failed: {e}")

    def perform_consistency_cutoff(self, contig_data_list, threshold=0.90, step=1000,
                                     n_workers=None, verbose=True):
        """
        Strip parents from individuals whose trio is not consistently explained
        across chromosomes.

        For each individual with assigned parents, checks on every chromosome
        what fraction of the child's genome can be explained as inheriting one
        haplotype from each parent. Real parent-child trios score ~95-100%.
        Spurious trios (e.g. siblings misidentified as parents) score ~30-55%.

        Parallelized: each trio is checked by a separate worker across all
        chromosomes simultaneously.

        Args:
            contig_data_list: List of dicts with 'tolerance_painting' key
                              (BlockTolerancePainting objects, one per chromosome)
            threshold: Minimum mean_explained fraction to keep parents
                       (default 0.90).  Raised from the historical 0.80 in
                       May 2026 based on threshold_diagnostic.py findings:
                       on the 320-sample dataset under union mode, CORRECT
                       trios scored 99.62-99.99% (min/max), while WRONG_ROOT
                       trios (F1s with Founder parents getting decoy F2/F2
                       assignments) scored 67.82-80.65% (min/max).  The 19%
                       gap between the two distributions makes 0.90 (midpoint)
                       a safe choice with zero false negatives and zero false
                       positives on this dataset; the previous 0.80 default
                       kept F1_17 at 80.65% as a wrong-direction assignment
                       that had to be cleaned up post-hoc by cycle resolution.
            step: Base-pair sampling interval for the consistency check
            n_workers: Number of parallel workers (default: all available cores)
            verbose: Print details
        """
        import multiprocessing as mp

        if n_workers is None:
            n_workers = os.cpu_count() or 4

        name_to_idx = {s: i for i, s in enumerate(self.samples)}

        # Find all individuals with assigned parents
        trios = []
        for idx, row in self.relationships.iterrows():
            p1, p2 = row['Parent1'], row['Parent2']
            if (pd.notna(p1) and pd.notna(p2) and
                    p1 in name_to_idx and p2 in name_to_idx):
                trios.append({
                    'df_idx': idx,
                    'child': row['Sample'],
                    'child_i': name_to_idx[row['Sample']],
                    'p1_i': name_to_idx[p1],
                    'p2_i': name_to_idx[p2],
                })

        if not trios:
            if verbose:
                print("[Consistency-Cutoff] No trios to check.")
            return

        if verbose:
            print(f"[Consistency-Cutoff] Checking {len(trios)} trios across "
                  f"{len(contig_data_list)} chromosomes ({n_workers} workers)...")

        # Store tolerance paintings in shared dict for fork COW
        shared = {
            'contig_data_list': contig_data_list,
            'step': step,
        }

        # Build tasks: one per trio
        tasks = []
        for t in trios:
            tasks.append((t['df_idx'], t['child'], t['child_i'], t['p1_i'], t['p2_i']))

        ctx = mp.get_context('fork')
        actual_workers = min(n_workers, len(tasks))
        # Counters for the two-step dynamic-thread reallocation pattern
        # (see module-level _PEDIGREE_ACTIVE_COUNTER docstring).  Fresh
        # mp.Value('i', 0) per pool so stale state from previous pools
        # doesn't leak in.  total_cores=actual_workers tells worker
        # processes the maximum thread budget they can ever request.
        active_counter = ctx.Value('i', 0)
        extra_counter = ctx.Value('i', 0)

        with ctx.Pool(processes=actual_workers,
                      initializer=_init_pedigree_shared,
                      initargs=(shared, active_counter, actual_workers, extra_counter)) as pool:
            results = list(tqdm(
                pool.imap_unordered(_check_trio_consistency_worker, tasks),
                total=len(tasks),
                desc="Checking trio consistency"
            ))

        # Apply threshold
        stripped = 0
        for df_idx, child_name, mean_expl in results:
            if mean_expl < threshold:
                self.relationships.at[df_idx, 'Parent1'] = None
                self.relationships.at[df_idx, 'Parent2'] = None
                stripped += 1
                if verbose:
                    print(f"[Consistency-Cutoff] Stripped {child_name}: "
                          f"mean_explained={mean_expl*100:.1f}%")

        if verbose:
            print(f"[Consistency-Cutoff] Stripped {stripped}/{len(trios)} trios "
                  f"(threshold={threshold*100:.0f}%)")
        self._recalculate_generations()

    def resolve_cycles(self, verbose=True):
        G = nx.DiGraph()
        for _, row in self.relationships.iterrows():
            child = row['Sample']
            G.add_node(child)
            if pd.notna(row['Parent1']):
                G.add_edge(row['Parent1'], child)
            if pd.notna(row['Parent2']):
                G.add_edge(row['Parent2'], child)
        if nx.is_directed_acyclic_graph(G):
            if verbose:
                print("[Cycle Resolution] Pedigree is already acyclic.")
            return
        trio_scores_norm = {}
        if self.total_bins > 0:
            for s in self.samples:
                raw = self.trio_scores.get(s, 0)
                trio_scores_norm[s] = raw / self.total_bins
        iteration = 0
        max_iterations = 20
        while iteration < max_iterations:
            iteration += 1
            G = nx.DiGraph()
            for _, row in self.relationships.iterrows():
                child = row['Sample']
                G.add_node(child)
                if pd.notna(row['Parent1']):
                    G.add_edge(row['Parent1'], child)
                if pd.notna(row['Parent2']):
                    G.add_edge(row['Parent2'], child)
            if nx.is_directed_acyclic_graph(G):
                if verbose:
                    print(f"[Cycle Resolution] Pedigree is acyclic after {iteration-1} fix(es).")
                break
            cycles = list(nx.simple_cycles(G))
            if verbose:
                print(f"[Cycle Resolution] Iteration {iteration}: Found {len(cycles)} cycle(s)")
            samples_in_cycles = set()
            for cycle in cycles:
                samples_in_cycles.update(cycle)
            candidates_to_fix = []
            for sample in samples_in_cycles:
                row = self.relationships[self.relationships['Sample'] == sample]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                has_parents = pd.notna(row['Parent1']) or pd.notna(row['Parent2'])
                n_children = G.out_degree(sample)
                if has_parents and n_children > 0:
                    score = trio_scores_norm.get(sample, 0.0)
                    candidates_to_fix.append({
                        'sample': sample,
                        'score': score,
                        'n_children': n_children,
                        'parents': (row['Parent1'], row['Parent2'])
                    })
            if not candidates_to_fix:
                if verbose:
                    print(f"[Cycle Resolution] No candidates found to fix")
                break
            candidates_to_fix.sort(key=lambda x: (-x['n_children'], x['score']))
            to_fix = candidates_to_fix[0]
            sample_to_fix = to_fix['sample']
            if verbose:
                print(f"[Cycle Resolution] Fixing {sample_to_fix}: "
                      f"n_children={to_fix['n_children']}, score={to_fix['score']:.4f}, "
                      f"removing parents {to_fix['parents']}")
            idx = self.relationships[self.relationships['Sample'] == sample_to_fix].index[0]
            self.relationships.at[idx, 'Parent1'] = None
            self.relationships.at[idx, 'Parent2'] = None
            self.relationships.at[idx, 'Generation'] = 'F1'
        self._recalculate_generations()

# =============================================================================
# 2. DISCRETIZATION & ALLELE CONVERSION FOR TOLERANCE PAINTINGS
# =============================================================================

def discretize_consensus_with_uncertainty(consensus_painting, start_pos, end_pos, snps_per_bin=100):
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100 
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100: num_bins = 100 
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    id_grid = np.zeros((num_bins, 2), dtype=np.int32) - 1
    potential_hom_mask = np.ones(num_bins, dtype=np.bool_)
    cons_chunks = consensus_painting.chunks
    if not cons_chunks:
        return id_grid, potential_hom_mask, bin_centers
    if consensus_painting.representative_path is not None:
        rep_chunks = consensus_painting.representative_path.chunks
    else:
        rep_chunks = None
    c_ends = np.array([c.end for c in cons_chunks])
    c_starts = np.array([c.start for c in cons_chunks])
    chunk_indices = np.searchsorted(c_ends, bin_centers)
    chunk_indices = np.clip(chunk_indices, 0, len(cons_chunks) - 1)
    for b in range(num_bins):
        cidx = chunk_indices[b]
        chunk = cons_chunks[cidx]
        if bin_centers[b] < chunk.start or bin_centers[b] >= chunk.end:
            potential_hom_mask[b] = True
            continue
        intersection = chunk.possible_hap1 & chunk.possible_hap2
        potential_hom_mask[b] = len(intersection) > 0
    if rep_chunks:
        rep_ends = np.array([c.end for c in rep_chunks])
        rep_h1 = np.array([c.hap1 for c in rep_chunks])
        rep_h2 = np.array([c.hap2 for c in rep_chunks])
        rep_starts = np.array([c.start for c in rep_chunks])
        rep_indices = np.searchsorted(rep_ends, bin_centers)
        rep_indices = np.clip(rep_indices, 0, len(rep_chunks) - 1)
        valid_mask = bin_centers >= rep_starts[rep_indices]
        id_grid[:, 0] = np.where(valid_mask, rep_h1[rep_indices], -1)
        id_grid[:, 1] = np.where(valid_mask, rep_h2[rep_indices], -1)
    return id_grid, potential_hom_mask, bin_centers


def discretize_sample_painting_with_hom_mask(sample_painting, start_pos, end_pos, snps_per_bin=100):
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100 
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100: num_bins = 100 
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    id_grid = np.zeros((num_bins, 2), dtype=np.int32) - 1
    chunks = sample_painting.chunks
    if not chunks:
        potential_hom_mask = np.ones(num_bins, dtype=np.bool_)
        return id_grid, potential_hom_mask, bin_centers
    c_ends = np.array([c.end for c in chunks])
    c_h1 = np.array([c.hap1 for c in chunks])
    c_h2 = np.array([c.hap2 for c in chunks])
    c_starts = np.array([c.start for c in chunks])
    indices = np.searchsorted(c_ends, bin_centers)
    indices = np.clip(indices, 0, len(chunks) - 1)
    valid_mask = bin_centers >= c_starts[indices]
    id_grid[:, 0] = np.where(valid_mask, c_h1[indices], -1)
    id_grid[:, 1] = np.where(valid_mask, c_h2[indices], -1)
    potential_hom_mask = (id_grid[:, 0] == id_grid[:, 1]) | (id_grid[:, 0] == -1) | (id_grid[:, 1] == -1)
    return id_grid, potential_hom_mask, bin_centers


def discretize_tolerance_paintings(block_tolerance_painting, snps_per_bin=100):
    start_pos = block_tolerance_painting.start_pos
    end_pos = block_tolerance_painting.end_pos
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100 
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100: num_bins = 100 
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    num_samples = len(block_tolerance_painting)
    sample_grids = []
    for i in range(num_samples):
        sample_obj = block_tolerance_painting[i]
        grids_for_sample = []
        consensus_list = sample_obj.consensus_list
        if consensus_list:
            for cons in consensus_list:
                id_grid, potential_hom_mask, _ = discretize_consensus_with_uncertainty(
                    cons, start_pos, end_pos, snps_per_bin
                )
                grids_for_sample.append((id_grid, potential_hom_mask, cons.weight))
        if not grids_for_sample and sample_obj.paths:
            id_grid, potential_hom_mask, _ = discretize_sample_painting_with_hom_mask(
                sample_obj.paths[0], start_pos, end_pos, snps_per_bin
            )
            grids_for_sample.append((id_grid, potential_hom_mask, 1.0))
        if not grids_for_sample:
            id_grid = np.zeros((num_bins, 2), dtype=np.int32) - 1
            potential_hom_mask = np.ones(num_bins, dtype=np.bool_)
            grids_for_sample.append((id_grid, potential_hom_mask, 1.0))
        sample_grids.append(grids_for_sample)
    return sample_grids, bin_centers


def convert_id_grid_to_allele_grid(id_grid, bin_centers, founder_block, bin_width_bp=None):
    num_bins = id_grid.shape[0]
    bin_indices = np.searchsorted(founder_block.positions, bin_centers)
    bin_indices = np.clip(bin_indices, 0, len(founder_block.positions) - 1)
    if bin_width_bp is None:
        if len(bin_centers) > 1:
            bin_width_bp = bin_centers[1] - bin_centers[0]
        else:
            bin_width_bp = 10000 
    found_snps_pos = founder_block.positions[bin_indices]
    dist_to_center = np.abs(found_snps_pos - bin_centers)
    valid_snp_mask = dist_to_center <= (bin_width_bp / 2.0)
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    allele_lookup = np.full((max_id + 1, num_bins), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2: raw_alleles = np.argmax(h_arr, axis=1)
        else: raw_alleles = h_arr
        extracted = raw_alleles[bin_indices]
        extracted[~valid_snp_mask] = -1
        allele_lookup[fid, :] = extracted
    allele_grid = np.full_like(id_grid, -1, dtype=np.int8)
    b_indices = np.arange(num_bins)
    for chrom in [0, 1]:
        ids = id_grid[:, chrom]
        valid_mask = (ids != -1)
        safe_ids = ids.copy()
        safe_ids[~valid_mask] = 0
        alleles = allele_lookup[safe_ids, b_indices]
        alleles[~valid_mask] = -1
        allele_grid[:, chrom] = alleles
    return allele_grid


def convert_id_grid_to_allele_grid_multisnp(id_grid, bin_centers, founder_block, 
                                             bin_width_bp=None, max_snps_per_bin=10):
    num_bins = id_grid.shape[0]
    snp_positions = founder_block.positions
    n_snps = len(snp_positions)
    allele_grid = np.full((num_bins, 2, max_snps_per_bin), -1, dtype=np.int8)
    if n_snps == 0:
        return allele_grid
    if bin_width_bp is None:
        if len(bin_centers) > 1:
            bin_width_bp = bin_centers[1] - bin_centers[0]
        else:
            bin_width_bp = 10000
    half_width = bin_width_bp / 2.0
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    founder_alleles = np.full((max_id + 1, n_snps), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            founder_alleles[fid, :] = np.argmax(h_arr, axis=1).astype(np.int8)
        else:
            founder_alleles[fid, :] = h_arr.astype(np.int8)
    bin_starts = bin_centers - half_width
    bin_ends = bin_centers + half_width
    start_indices = np.searchsorted(snp_positions, bin_starts, side='left')
    end_indices = np.searchsorted(snp_positions, bin_ends, side='right')
    for b in range(num_bins):
        s_start = start_indices[b]
        s_end = end_indices[b]
        bin_n_snps = s_end - s_start
        if bin_n_snps == 0:
            continue
        if bin_n_snps <= max_snps_per_bin:
            sampled_indices = list(range(s_start, s_end))
        else:
            step = bin_n_snps / max_snps_per_bin
            sampled_indices = [s_start + int(i * step) for i in range(max_snps_per_bin)]
        f0 = id_grid[b, 0]
        f1 = id_grid[b, 1]
        for k_idx, snp_idx in enumerate(sampled_indices):
            if k_idx >= max_snps_per_bin:
                break
            if f0 >= 0:
                allele_grid[b, 0, k_idx] = founder_alleles[f0, snp_idx]
            if f1 >= 0:
                allele_grid[b, 1, k_idx] = founder_alleles[f1, snp_idx]
    return allele_grid


# =============================================================================
# 2b. SNP-LEVEL DATA STRUCTURES FOR BINNED AGGREGATION
# =============================================================================

def get_snp_level_founder_ids(painting_chunks, snp_positions):
    n_snps = len(snp_positions)
    id_array = np.full((n_snps, 2), -1, dtype=np.int32)
    if not painting_chunks:
        return id_array
    c_ends = np.array([c.end for c in painting_chunks])
    c_h1 = np.array([c.hap1 for c in painting_chunks])
    c_h2 = np.array([c.hap2 for c in painting_chunks])
    c_starts = np.array([c.start for c in painting_chunks])
    indices = np.searchsorted(c_ends, snp_positions)
    indices = np.clip(indices, 0, len(painting_chunks) - 1)
    valid_mask = snp_positions >= c_starts[indices]
    id_array[:, 0] = np.where(valid_mask, c_h1[indices], -1)
    id_array[:, 1] = np.where(valid_mask, c_h2[indices], -1)
    return id_array


def get_snp_level_hom_mask_from_consensus(consensus_painting, snp_positions):
    n_snps = len(snp_positions)
    hom_mask = np.ones(n_snps, dtype=np.bool_)
    cons_chunks = consensus_painting.chunks
    if not cons_chunks:
        return hom_mask
    c_ends = np.array([c.end for c in cons_chunks])
    c_starts = np.array([c.start for c in cons_chunks])
    chunk_indices = np.searchsorted(c_ends, snp_positions)
    chunk_indices = np.clip(chunk_indices, 0, len(cons_chunks) - 1)
    for s in range(n_snps):
        cidx = chunk_indices[s]
        chunk = cons_chunks[cidx]
        if snp_positions[s] < chunk.start or snp_positions[s] >= chunk.end:
            hom_mask[s] = True
            continue
        intersection = chunk.possible_hap1 & chunk.possible_hap2
        hom_mask[s] = len(intersection) > 0
    return hom_mask


def build_founder_allele_lookup(founder_block):
    snp_positions = founder_block.positions
    n_snps = len(snp_positions)
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    allele_lookup = np.full((max_id + 1, n_snps), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            raw_alleles = np.argmax(h_arr, axis=1)
        else:
            raw_alleles = h_arr
        allele_lookup[fid, :] = raw_alleles.astype(np.int8)
    return allele_lookup, snp_positions


def create_bin_structure(snp_positions, snps_per_bin=100):
    n_snps = len(snp_positions)
    if n_snps == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([])
    num_bins = max(1, n_snps // snps_per_bin)
    if num_bins < 100 and n_snps >= 100:
        num_bins = 100
    bin_starts = np.zeros(num_bins, dtype=np.int32)
    bin_ends = np.zeros(num_bins, dtype=np.int32)
    bin_centers = np.zeros(num_bins, dtype=np.float64)
    snps_per = n_snps // num_bins
    remainder = n_snps % num_bins
    start_idx = 0
    for b in range(num_bins):
        extra = 1 if b < remainder else 0
        end_idx = start_idx + snps_per + extra
        bin_starts[b] = start_idx
        bin_ends[b] = end_idx
        if end_idx > start_idx:
            bin_centers[b] = np.mean(snp_positions[start_idx:end_idx])
        start_idx = end_idx
    return bin_starts, bin_ends, bin_centers


def aggregate_hom_mask_to_bins(snp_hom_mask, bin_starts, bin_ends):
    n_bins = len(bin_starts)
    bin_hom_mask = np.zeros(n_bins, dtype=np.bool_)
    for b in range(n_bins):
        bin_hom_mask[b] = np.any(snp_hom_mask[bin_starts[b]:bin_ends[b]])
    return bin_hom_mask


# =============================================================================
# 3. STATISTICAL CALCULATIONS
# =============================================================================

def calculate_ibd_matrices(grid):
    num_samples, num_bins, _ = grid.shape
    kinship_sum = np.zeros((num_samples, num_samples))
    ibd0_sum = np.zeros((num_samples, num_samples))
    valid_counts = np.zeros((num_samples, num_samples))
    batch_size = 500
    for b_start in range(0, num_bins, batch_size):
        b_end = min(b_start + batch_size, num_bins)
        g_sub = grid[:, b_start:b_end, :]
        A = g_sub[:, np.newaxis, :, :]
        B = g_sub[np.newaxis, :, :, :]
        valid_mask = (A[..., 0] != -1) & (B[..., 0] != -1)
        valid_counts += np.sum(valid_mask, axis=2)
        match_direct = (A[...,0] == B[...,0]) & (A[...,1] == B[...,1])
        match_cross  = (A[...,0] == B[...,1]) & (A[...,1] == B[...,0])
        match_both   = match_direct | match_cross
        any_match = (A[...,0] == B[...,0]) | (A[...,0] == B[...,1]) | (A[...,1] == B[...,0]) | (A[...,1] == B[...,1])
        k_scores = np.zeros_like(match_both, dtype=float)
        k_scores[any_match] = 0.5; k_scores[match_both] = 1.0; k_scores[~valid_mask] = 0.0
        kinship_sum += np.sum(k_scores, axis=2)
        ibd0_scores = (~any_match).astype(float); ibd0_scores[~valid_mask] = 0.0
        ibd0_sum += np.sum(ibd0_scores, axis=2)
    valid_counts[valid_counts == 0] = 1.0
    return kinship_sum / valid_counts, ibd0_sum / valid_counts

# =============================================================================
# 4. HMM KERNELS (8-STATE FILTER & 16-STATE VERIFIER)
# =============================================================================

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


# =============================================================================
# 4b. MULTI-SNP HMM KERNELS (k samples per bin)
# =============================================================================

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

# =============================================================================
# 5. TOLERANCE-AWARE SCORING FUNCTIONS
# =============================================================================

DEFAULT_MISMATCH_PENALTY = -4.605170  # math.log(0.01)

def score_parent_child_all_consensus_precomputed(child_grids, parent_grids, 
                                                  switch_costs, stay_costs, 
                                                  error_penalty, phase_penalty,
                                                  mismatch_penalty=DEFAULT_MISMATCH_PENALTY):
    best_score = -np.inf
    for child_allele_grid, child_hom_mask, child_weight in child_grids:
        for parent_allele_grid, parent_hom_mask, parent_weight in parent_grids:
            if child_allele_grid.ndim == 3:
                score = run_phase_agnostic_hmm_multisnp(
                    child_allele_grid, child_hom_mask,
                    parent_allele_grid,
                    switch_costs, stay_costs,
                    error_penalty, phase_penalty,
                    mismatch_penalty
                )
            else:
                score = run_phase_agnostic_hmm(
                    child_allele_grid, child_hom_mask,
                    parent_allele_grid,
                    switch_costs, stay_costs,
                    error_penalty, phase_penalty,
                    mismatch_penalty
                )
            if score > best_score:
                best_score = score
    return best_score


def score_trio_all_consensus_precomputed(child_grids, p1_grids, p2_grids,
                                          switch_costs, stay_costs,
                                          error_penalty, phase_penalty,
                                          mismatch_penalty=DEFAULT_MISMATCH_PENALTY):
    best_score = -np.inf
    for child_allele_grid, child_hom_mask, child_weight in child_grids:
        for p1_allele_grid, p1_hom_mask, p1_weight in p1_grids:
            for p2_allele_grid, p2_hom_mask, p2_weight in p2_grids:
                if child_allele_grid.ndim == 3:
                    score = run_trio_phase_aware_hmm_multisnp(
                        child_allele_grid, child_hom_mask,
                        p1_allele_grid, p2_allele_grid,
                        switch_costs, stay_costs,
                        error_penalty, phase_penalty,
                        mismatch_penalty
                    )
                else:
                    score = run_trio_phase_aware_hmm(
                        child_allele_grid, child_hom_mask,
                        p1_allele_grid, p2_allele_grid,
                        switch_costs, stay_costs,
                        error_penalty, phase_penalty,
                        mismatch_penalty
                    )
                if score > best_score:
                    best_score = score
    return best_score


# -----------------------------------------------------------------------------
# Batched HMM-call kernels (optimization #4 from May 2026 numba pass).
#
# These wrap the per-(child, parent[, parent2]) HMM kernels in a numba outer
# loop so that, given an array of parent indices, an entire contig's worth of
# pair- or trio- scores can be computed in a single Python-to-numba call.  The
# math is bit-identical to looping in Python and calling
# run_phase_agnostic_hmm[_multisnp] or run_trio_phase_aware_hmm[_multisnp] per
# pair -- the per-bin Viterbi state, transition costs, emission costs and
# final max-over-states are unchanged.
#
# Inputs are arranged as a stacked array of shape (N, n_bins, 2, k_snps) for
# the multisnp variants or (N, n_bins, 2) for the non-multisnp variants, built
# once at cache-construction time in infer_pedigree_multi_contig_tolerance.
# Only the CHILD slot needs a hom_mask -- parents do not (the trio HMM uses
# the child's hom_mask to decide whether the phase_penalty applies on each
# bin, and the pair HMM is phase-agnostic).
#
# A separate kernel exists for each (multisnp y/n) × (pair / trio) combination
# because numba cannot dispatch on array ndim at runtime within a single
# jitted function -- but at the Python level we pick the right kernel based on
# stacked_alleles.ndim (4 = multisnp, 3 = non-multisnp).
# -----------------------------------------------------------------------------

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


# =============================================================================
# 6. MULTI-CONTIG INFERENCE LOGIC (PARALLELIZED WITH SHARED DATA)
# =============================================================================

def _process_contig_batch(args):
    """
    Worker: process a BATCH of samples for one contig in Phase 1.
    Uses IBS-aware single Viterbi painting — accesses SamplePainting.chunks
    directly and derives hom mask from allele identity.
    """
    from paint_samples import process_contig_for_pedigree
    
    contig_idx, sample_start, sample_end = args
    contig_data = _PEDIGREE_SHARED['contig_data_list'][contig_idx]
    tol_painting = contig_data['tolerance_painting']
    founder_block = contig_data['founder_block']
    snps_per_bin = _PEDIGREE_SHARED['snps_per_bin']
    recomb_rate = _PEDIGREE_SHARED['recomb_rate']
    max_snps_per_bin = _PEDIGREE_SHARED['max_snps_per_bin']
    
    return process_contig_for_pedigree(
        contig_idx, sample_start, sample_end,
        tol_painting, founder_block,
        snps_per_bin, recomb_rate, max_snps_per_bin
    )


def _score_contig_pairs(contig_idx):
    """Worker: score all parent-child pairs for one contig. Reads from shared data."""
    cache = _PEDIGREE_SHARED['contig_caches'][contig_idx]
    sample_allele_grids = cache['sample_allele_grids']
    sw_costs = cache['sw_costs']
    st_costs = cache['st_costs']
    error_pen = _PEDIGREE_SHARED['error_pen']
    phase_pen = _PEDIGREE_SHARED['phase_pen']
    mismatch_penalty = _PEDIGREE_SHARED['mismatch_penalty']
    num_samples = _PEDIGREE_SHARED['num_samples']
    
    scores = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            if i == j:
                continue
            score = score_parent_child_all_consensus_precomputed(
                sample_allele_grids[i],
                sample_allele_grids[j],
                sw_costs, st_costs,
                error_pen, phase_pen,
                mismatch_penalty
            )
            scores[i, j] = score
    return scores


def _score_pairs_by_children(child_indices):
    """
    Worker: score a batch of children against ALL parents across ALL contigs.
    
    Parallelizes by child rows instead of by contig, giving many more tasks
    (e.g. 32-64 batches of 5-10 children) to fill all available cores.

    Uses the batched kernels score_pair_batch_kernel[_multisnp] (optimization
    #4 from the May 2026 numba pass) so each (child, contig) combination
    incurs one Python-to-numba call instead of num_samples calls.  Math is
    bit-identical to the previous per-pair Python loop:
      - For each parent index in [0, num_samples), the kernel calls the same
        underlying HMM (run_phase_agnostic_hmm[_multisnp]) with the same args.
      - The self-pair score (ci == j) is computed by the kernel then
        overwritten to 0.0 to match the legacy skip-on-self semantics.
      - Contig sums are accumulated identically (total += s, but now in numpy
        rather than scalar Python).

    Dynamic thread reallocation (when the pool was initialized with
    counters): on entry, atomically increment _PEDIGREE_ACTIVE_COUNTER
    and claim an extra thread; rescale numba threads to floor+extra.
    On exit, release the extra and decrement.  Inside the per-contig
    loop, _update_dynamic_threads() lets the worker rebalance as peers
    finish.  When parallel=True is enabled on the batched kernels, this
    gives stragglers a dynamic share of the freed cores.
    """
    # Counter inc/threads-up on entry.  See _check_trio_consistency_worker
    # for the full rationale; this is the same pattern.
    counter_inc = False
    if _PEDIGREE_ACTIVE_COUNTER is not None and _PEDIGREE_TOTAL_CORES is not None:
        try:
            import numba as _numba
            with _PEDIGREE_ACTIVE_COUNTER.get_lock():
                _PEDIGREE_ACTIVE_COUNTER.value += 1
            counter_inc = True
            active = max(_PEDIGREE_ACTIVE_COUNTER.value, 1)
            floor = _PEDIGREE_TOTAL_CORES // active
            remainder = _PEDIGREE_TOTAL_CORES - floor * active
            _try_claim_extra(remainder)
            n_threads = max(1, floor + (1 if _PEDIGREE_I_HAVE_EXTRA else 0))
            _numba.set_num_threads(n_threads)
        except Exception:
            pass

    try:
        contig_caches = _PEDIGREE_SHARED['contig_caches']
        error_pen = _PEDIGREE_SHARED['error_pen']
        phase_pen = _PEDIGREE_SHARED['phase_pen']
        mismatch_penalty = _PEDIGREE_SHARED['mismatch_penalty']
        num_samples = _PEDIGREE_SHARED['num_samples']

        n_children = len(child_indices)
        scores = np.zeros((n_children, num_samples))

        # All-parents index vector reused across (child, contig) calls.
        all_parent_indices = np.arange(num_samples, dtype=np.int64)

        for ci_local, ci in enumerate(child_indices):
            for cache in contig_caches:
                # In-loop rebalance hook: as peer workers finish their
                # tasks, this worker re-checks the active count and may
                # claim newly-available extras (or release if active
                # grew).  Cost is one atomic counter read (~ns).
                _update_dynamic_threads()

                stacked = cache['stacked_alleles']
                hom = cache['stacked_hom_mask']
                if stacked.ndim == 4:
                    # multisnp path: (N, n_bins, 2, k_snps)
                    contig_scores = score_pair_batch_kernel_multisnp(
                        stacked[ci], hom[ci], stacked, all_parent_indices,
                        cache['sw_costs'], cache['st_costs'],
                        error_pen, phase_pen, mismatch_penalty,
                    )
                else:
                    # non-multisnp path: (N, n_bins, 2)
                    contig_scores = score_pair_batch_kernel(
                        stacked[ci], hom[ci], stacked, all_parent_indices,
                        cache['sw_costs'], cache['st_costs'],
                        error_pen, phase_pen, mismatch_penalty,
                    )
                scores[ci_local] += contig_scores
            # Restore the legacy skip-on-self semantics: scores[ci_local, ci] = 0.
            # In the previous code path the inner `if ci == j: continue` skipped
            # the self-pair entirely, leaving the np.zeros-initialised value of 0.
            # The batched kernel computes a (typically very high) self-similarity
            # score that we must overwrite to preserve identical output.
            scores[ci_local, ci] = 0.0

        return child_indices, scores
    finally:
        _try_release_extra()
        if counter_inc and _PEDIGREE_ACTIVE_COUNTER is not None:
            try:
                with _PEDIGREE_ACTIVE_COUNTER.get_lock():
                    _PEDIGREE_ACTIVE_COUNTER.value -= 1
            except Exception:
                pass


def _score_trios_batch(batch_sample_args):
    """Worker: score trios for a batch of samples. Reads from shared data.

    Dynamic thread reallocation (when the pool was initialized with
    counters): on entry, atomically increment _PEDIGREE_ACTIVE_COUNTER
    and claim an extra thread; rescale numba threads to floor+extra.
    On exit, release the extra and decrement.  Inside the per-contig
    loop, _update_dynamic_threads() lets the worker rebalance as peers
    finish -- the major Phase 2 win because that's where the bulk of
    pipeline time is spent and where the work-distribution tail is.
    """
    # Counter inc/threads-up on entry.  See _check_trio_consistency_worker
    # for the full rationale; this is the same pattern.
    counter_inc = False
    if _PEDIGREE_ACTIVE_COUNTER is not None and _PEDIGREE_TOTAL_CORES is not None:
        try:
            import numba as _numba
            with _PEDIGREE_ACTIVE_COUNTER.get_lock():
                _PEDIGREE_ACTIVE_COUNTER.value += 1
            counter_inc = True
            active = max(_PEDIGREE_ACTIVE_COUNTER.value, 1)
            floor = _PEDIGREE_TOTAL_CORES // active
            remainder = _PEDIGREE_TOTAL_CORES - floor * active
            _try_claim_extra(remainder)
            n_threads = max(1, floor + (1 if _PEDIGREE_I_HAVE_EXTRA else 0))
            _numba.set_num_threads(n_threads)
        except Exception:
            pass

    try:
        contig_caches = _PEDIGREE_SHARED['contig_caches']
        total_switches = _PEDIGREE_SHARED['total_switches']
        error_pen = _PEDIGREE_SHARED['error_pen']
        phase_pen = _PEDIGREE_SHARED['phase_pen']
        mismatch_penalty = _PEDIGREE_SHARED['mismatch_penalty']
        complexity_penalty = _PEDIGREE_SHARED['complexity_penalty']

        results = []
        for args in batch_sample_args:
            # Tuple format depends on caller in infer_pedigree_multi_contig_tolerance:
            #   (sample_idx, top_indices)              -- legacy enumeration
            #                                             (use_anchor_union=False);
            #                                             worker builds pairs as
            #                                             unordered top_indices
            #                                             pairs (dedup #1 from
            #                                             May 2026 numba pass --
            #                                             run_trio_phase_aware_hmm
            #                                             is symmetric in (p1, p2)
            #                                             so (p2, p1) was redundant).
            #   (sample_idx, top_indices, pairs_list)  -- new union enumeration
            #                                             (use_anchor_union=True);
            #                                             pairs_list is the
            #                                             deduplicated union of
            #                                             (top_indices × top_indices)
            #                                             and (anchor × all-N),
            #                                             stored as canonical
            #                                             (min, max) ordering.
            if len(args) == 3:
                sample_idx, top_indices, pairs = args
            else:
                sample_idx, top_indices = args
                pairs = None
            if len(top_indices) < 1:
                results.append((sample_idx, None, -1e9, []))
                continue
            if pairs is None:
                # Legacy enumeration with symmetric-pair dedup: enumerate only
                # i < j unordered pairs.  The trio HMM is bit-identical under
                # (p1, p2) swap (verified empirically May 2026 -- 100/100 random
                # trios match exactly), so the previous
                #     [(p1, p2) for p1 in top for p2 in top if p1 != p2]
                # double-counted each unordered pair: 380 → 190 for top_k=20.
                top_list = list(top_indices)
                pairs = [(top_list[i], top_list[j])
                         for i in range(len(top_list))
                         for j in range(i + 1, len(top_list))]
            if not pairs:
                pairs = [(top_indices[0], top_indices[0])]
            # Build int64 index arrays for the batched kernel.
            n_pairs = len(pairs)
            p1_arr = np.empty(n_pairs, dtype=np.int64)
            p2_arr = np.empty(n_pairs, dtype=np.int64)
            for k_idx in range(n_pairs):
                p1_arr[k_idx] = pairs[k_idx][0]
                p2_arr[k_idx] = pairs[k_idx][1]
            # Sum trio_ll across contigs using the batched kernel.  Math is
            # bit-identical to the old per-pair, per-contig Python loop that
            # called score_trio_all_consensus_precomputed (which calls
            # run_trio_phase_aware_hmm[_multisnp]) inside.  The kernel just
            # wraps the same per-pair HMM call in a numba outer loop, so each
            # entry of `contig_scores` is exactly what `score` was before.
            trio_lls = np.zeros(n_pairs, dtype=np.float64)
            for cache in contig_caches:
                # In-loop rebalance hook: as peer workers finish their
                # tasks, this worker re-checks the active count and may
                # claim newly-available extras (or release if active
                # grew).  This is the most impactful hook in the file --
                # Phase 2 tail dynamics depend on it.
                _update_dynamic_threads()

                stacked = cache['stacked_alleles']
                hom = cache['stacked_hom_mask']
                sw_costs = cache['sw_costs']
                st_costs = cache['st_costs']
                if stacked.ndim == 4:
                    contig_scores = score_trio_batch_kernel_multisnp(
                        stacked[sample_idx], hom[sample_idx], stacked,
                        p1_arr, p2_arr,
                        sw_costs, st_costs,
                        error_pen, phase_pen, mismatch_penalty,
                    )
                else:
                    contig_scores = score_trio_batch_kernel(
                        stacked[sample_idx], hom[sample_idx], stacked,
                        p1_arr, p2_arr,
                        sw_costs, st_costs,
                        error_pen, phase_pen, mismatch_penalty,
                    )
                trio_lls += contig_scores
            # Apply complexity penalty per pair and find argmax.  This matches
            # the legacy
            #     final = trio_ll - (total_switches[p1] + total_switches[p2]) * complexity_penalty
            # exactly (vectorised here).
            cp = (total_switches[p1_arr] + total_switches[p2_arr]) * complexity_penalty
            final_scores = trio_lls - cp
            best_k = int(np.argmax(final_scores))
            best_trio = (int(p1_arr[best_k]), int(p2_arr[best_k]))
            best_trio_score = float(final_scores[best_k])
            results.append((sample_idx, best_trio, best_trio_score, top_indices))
        return results
    finally:
        _try_release_extra()
        if counter_inc and _PEDIGREE_ACTIVE_COUNTER is not None:
            try:
                with _PEDIGREE_ACTIVE_COUNTER.get_lock():
                    _PEDIGREE_ACTIVE_COUNTER.value -= 1
            except Exception:
                pass


def infer_pedigree_multi_contig_tolerance(contig_data_list, sample_ids, top_k=20,
                                          snps_per_bin=100, recomb_rate=5e-8,
                                          mismatch_penalty=DEFAULT_MISMATCH_PENALTY,
                                          max_snps_per_bin=10,
                                          n_workers=None,
                                          anchor_k=5,
                                          use_anchor_union=True):
    """
    Multi-contig pedigree inference using tolerance paintings with multi-SNP voting.

    Phase 2 candidate-pair enumeration:
      use_anchor_union=False  (legacy):
          For each child i, enumerate trios from top_k × top_k candidates
          where the candidate set is filtered by the +5 margin generation
          filter (cand_mask: parent_switches <= child_switches + 5).
      use_anchor_union=True   (default, new):
          For each child i, enumerate trios from the UNION of:
            (a) top_k × top_k (same as legacy but with cand_mask filter
                NOT applied -- the filter was found to spuriously exclude
                high-switch true parents in May 2026 diagnostic work),
            (b) top_anchor_k × all-N -- each of the top anchor_k candidates
                is paired with every other valid sample (both orderings).
          This catches asymmetric cases where one true parent dominates
          Phase 1b scoring so much that the other true parent falls outside
          top-K.  Cost: ~17-18x current Phase 2; mathematically a strict
          superset of the legacy candidate pair set.
    """
    import multiprocessing as mp
    
    num_samples = len(sample_ids)
    num_contigs = len(contig_data_list)
    
    if n_workers is None:
        # Use all available cores by default.  The historical
        #   min(os.cpu_count() or 4, 16)
        # cap was a leftover from earlier development on smaller hardware
        # and silently bottlenecked any caller passing n_workers=None to
        # 16 workers regardless of cpu_count -- inconsistent with
        # perform_consistency_cutoff (line 234) which defaults to all
        # cores.  Mathematical results are deterministic across worker
        # counts; only batching granularity and wall-clock time change.
        n_workers = os.cpu_count() or 4
    
    # Phase 1: Discretize and convert allele grids — parallelize across
    # (contig, sample_batch) pairs to use all cores, not just 7.
    # With 7 contigs × ~10 batches each = ~70 tasks across 112 cores.
    samples_per_phase1_batch = max(1, num_samples // max(1, n_workers // num_contigs))
    
    phase1_tasks = []
    for c_idx in range(num_contigs):
        for s_start in range(0, num_samples, samples_per_phase1_batch):
            s_end = min(s_start + samples_per_phase1_batch, num_samples)
            phase1_tasks.append((c_idx, s_start, s_end))
    
    print(f"\n--- Phase 1: Processing {num_contigs} Contigs × {len(phase1_tasks)} tasks "
          f"({n_workers} workers, {max_snps_per_bin} SNPs/bin) ---")
    
    error_pen = -math.log(1e-2)
    phase_pen = 50.0
    
    # Share contig data via initializer — tolerance paintings + founder blocks
    shared_phase1 = {
        'contig_data_list': contig_data_list,
        'snps_per_bin': snps_per_bin,
        'recomb_rate': recomb_rate,
        'max_snps_per_bin': max_snps_per_bin,
    }
    
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=min(n_workers, len(phase1_tasks)),
                  initializer=_init_pedigree_shared,
                  initargs=(shared_phase1,)) as pool:
        phase1_results = list(tqdm(
            pool.imap_unordered(_process_contig_batch, phase1_tasks),
            total=len(phase1_tasks),
            desc="Processing contigs in parallel"
        ))
    
    # Reassemble results into per-contig caches
    # First pass: collect transition costs and num_bins per contig
    contig_meta = {}
    for result in phase1_results:
        c_idx = result['contig_idx']
        if c_idx not in contig_meta:
            contig_meta[c_idx] = {
                'sw_costs': result['sw_costs'],
                'st_costs': result['st_costs'],
                'num_bins': result['num_bins'],
            }
    
    # Second pass: assemble per-sample allele grids in order
    contig_sample_grids = {c_idx: [None] * num_samples for c_idx in range(num_contigs)}
    total_switches = np.zeros(num_samples)
    global_total_bins = 0
    
    for result in phase1_results:
        c_idx = result['contig_idx']
        s_start = result['sample_start']
        s_end = result['sample_end']
        for local_i, global_i in enumerate(range(s_start, s_end)):
            contig_sample_grids[c_idx][global_i] = result['sample_allele_grids'][local_i]
        total_switches[s_start:s_end] += result['switch_counts']
    
    contig_caches = []
    for c_idx in range(num_contigs):
        meta = contig_meta[c_idx]
        # Build stacked arrays for the batched HMM-call kernels.
        # paint_samples.process_contig_for_pedigree (line ~1170) always emits
        # a single-element list [(allele_grid, hom_mask, 1.0)] per sample, so
        # we can stack across samples into a single contiguous array.  This is
        # the input format expected by score_*_batch_kernel[_multisnp] above.
        # The legacy 'sample_allele_grids' list is kept so the unused
        # score_*_precomputed wrappers (and any external callers) still work.
        grids_list = contig_sample_grids[c_idx]
        assert all(len(g) == 1 for g in grids_list), (
            "Expected exactly one allele grid per sample per contig "
            "(paint_samples emits length-1 lists with weight 1.0); "
            "found a multi-grid sample.  Batched kernels assume single-grid.")
        first_alleles = grids_list[0][0][0]
        first_hom = grids_list[0][0][1]
        stacked_alleles = np.stack(
            [grids_list[i][0][0] for i in range(len(grids_list))], axis=0,
        )
        stacked_hom_mask = np.stack(
            [grids_list[i][0][1] for i in range(len(grids_list))], axis=0,
        )
        # Sanity-check the resulting shapes match the per-sample grids
        assert stacked_alleles.shape[0] == len(grids_list)
        assert stacked_alleles.shape[1:] == first_alleles.shape
        assert stacked_hom_mask.shape[0] == len(grids_list)
        assert stacked_hom_mask.shape[1:] == first_hom.shape
        contig_caches.append({
            'sample_allele_grids': contig_sample_grids[c_idx],
            'sw_costs': meta['sw_costs'],
            'st_costs': meta['st_costs'],
            # New stacked arrays for batched kernels (optimization #4):
            'stacked_alleles': stacked_alleles,
            'stacked_hom_mask': stacked_hom_mask,
        })
        global_total_bins += meta['num_bins']
    
    # Phase 1b + Phase 2: Use ONE Pool for both phases to avoid double fork overhead.
    # Set all shared data before Pool creation — workers see it via fork COW.
    COMPLEXITY_PENALTY = 0.0
    
    shared_all = {
        'contig_caches': contig_caches,
        'error_pen': error_pen,
        'phase_pen': phase_pen,
        'mismatch_penalty': mismatch_penalty,
        'num_samples': num_samples,
        # Phase 2 fields (set now so workers have them at fork time)
        'total_switches': total_switches,
        'complexity_penalty': COMPLEXITY_PENALTY,
    }
    
    # Phase 1b: Score all pairs — parallelize by CHILD ROWS (not by contig)
    print(f"\n--- Phase 1b: Scoring Parent-Child Pairs ({n_workers} workers) ---")
    
    children_per_batch = max(1, num_samples // (n_workers * 2))
    child_batches = []
    for start in range(0, num_samples, children_per_batch):
        end = min(start + children_per_batch, num_samples)
        child_batches.append(list(range(start, end)))
    
    total_scores = np.zeros((num_samples, num_samples))

    # Counters for the two-step dynamic-thread reallocation pattern
    # (see module-level _PEDIGREE_ACTIVE_COUNTER docstring).  Fresh
    # mp.Value('i', 0) per pool.  This pool services BOTH Phase 1b
    # and Phase 2, so the same counters tracking active workers carry
    # across the two phases -- by design, since at the boundary between
    # the phases all workers have decremented (Phase 1b finishes) before
    # the Phase 2 dispatch starts, so the counter is naturally back at 0.
    pool_active_counter = ctx.Value('i', 0)
    pool_extra_counter = ctx.Value('i', 0)
    pool_workers = min(n_workers, len(child_batches))

    with ctx.Pool(processes=pool_workers,
                  initializer=_init_pedigree_shared,
                  initargs=(shared_all, pool_active_counter, n_workers, pool_extra_counter)) as pool:
        
        # --- Phase 1b ---
        results = list(tqdm(
            pool.imap_unordered(_score_pairs_by_children, child_batches),
            total=len(child_batches),
            desc="Scoring pairs in parallel"
        ))
        
        for child_indices, scores_block in results:
            for ci_local, ci in enumerate(child_indices):
                total_scores[ci, :] = scores_block[ci_local, :]
        
        total_scores[total_scores == -np.inf] = -1e9
        
        cand_mask = np.zeros((num_samples, num_samples), dtype=bool)
        margin = 5
        for i in range(num_samples):
            valid_gen = total_switches <= (total_switches[i] + margin)
            cand_mask[i, :] = valid_gen
            cand_mask[i, i] = False

        # --- Phase 2: Trio Verification (same Pool, no re-fork) ---
        # cand_mask above is preserved in both paths: the legacy path applies
        # it to gate top_indices, while the anchor-union path no longer
        # applies it for gating (the +5 margin spuriously excluded high-
        # switch true parents in May 2026 diagnostic work; the 80%
        # consistency cutoff in perform_consistency_cutoff is the real
        # correctness gate).  cand_mask is still computed because
        # investigate_phase1b_coverage.py captures it via stack-frame spy
        # for diagnostic purposes.
        if use_anchor_union:
            print(f"\n--- Phase 2: Trio Verification "
                  f"(UNION: top-{top_k} × top-{top_k} ∪ top-{anchor_k} anchors × all-N, "
                  f"reusing pool) ---")
        else:
            print(f"\n--- Phase 2: Trio Verification (Top {top_k} Pairs, reusing pool) ---")

        all_sample_args = []
        for i in range(num_samples):
            valid_scores = total_scores[i].copy()
            if use_anchor_union:
                # ANCHOR-UNION PATH: cand_mask NOT applied; only exclude
                # self-pair.  See doc-comment on use_anchor_union above.
                valid_scores[i] = -np.inf
            else:
                # LEGACY PATH: apply +5 margin generation filter.
                valid_scores[~cand_mask[i, :]] = -np.inf
            for j in range(num_samples):
                if valid_scores[j] > -1e9:
                    valid_scores[j] -= (total_switches[j] * COMPLEXITY_PENALTY)
            top_indices = np.argsort(valid_scores)[-top_k:][::-1]
            top_indices = [x for x in top_indices if valid_scores[x] > -1e10]

            if use_anchor_union:
                # Build the union pair set per child:
                #   (a) top-K × top-K (no self) -- legacy semantics
                #   (b) top-anchor_k × all-N (no self/anchor)
                # Pairs are stored as canonical (min, max) tuples to dedupe
                # symmetric duplicates.  The trio HMM
                # (run_trio_phase_aware_hmm[_multisnp]) is bit-identical under
                # (p1, p2) swap -- verified empirically May 2026 (100/100
                # random trios match exactly).  The earlier version stored
                # both (a, j) and (j, a), doubling Phase 2 cost.
                pair_set = set()
                # (a) top-K × top-K (unordered i<j)
                top_list_int = [int(p) for p in top_indices]
                for ii in range(len(top_list_int)):
                    for jj in range(ii + 1, len(top_list_int)):
                        pair_set.add((top_list_int[ii], top_list_int[jj]))
                # (b) anchors × all-N (single canonical ordering)
                anchor_list_int = top_list_int[:anchor_k] if anchor_k > 0 else []
                for a_int in anchor_list_int:
                    for j in range(num_samples):
                        if j == i or j == a_int:
                            continue
                        if valid_scores[j] <= -1e10:
                            continue
                        j_int = int(j)
                        if a_int < j_int:
                            pair_set.add((a_int, j_int))
                        else:
                            pair_set.add((j_int, a_int))
                pairs_list = list(pair_set)
                all_sample_args.append((i, top_indices, pairs_list))
            else:
                # LEGACY PATH: worker enumerates pairs from top_indices.
                all_sample_args.append((i, top_indices))
        
        # Per-sample tasks for finer progress granularity.  The previous
        # heuristic `batch_size = max(1, num_samples // (n_workers * 4))`
        # was a generic "4 batches per worker" load-balancing trick, but
        # imap_unordered already does dynamic task scheduling, and each
        # per-sample trio enumeration is heavy enough that IPC overhead
        # is negligible compared to compute (sub-1% even at 112 workers).
        # At 112 workers the old heuristic clamped batch_size to 1 anyway
        # (so this is functionally identical there); at lower worker
        # counts the IPC overhead is still negligible.  Wrapping each
        # arg in a single-element list preserves the worker function's
        # signature (it accepts a list of sample args).
        single_arg_batches = [[arg] for arg in all_sample_args]

        batch_results = list(tqdm(
            pool.imap_unordered(_score_trios_batch, single_arg_batches),
            total=num_samples,
            desc="Inferring Trios in parallel"
        ))
    
    trio_results = []
    for batch in batch_results:
        trio_results.extend(batch)
    
    # CRITICAL: Sort by sample_idx so DataFrame rows match self.samples order.
    # imap_unordered returns results in arbitrary order; perform_automatic_cutoff
    # assumes self.relationships.at[i, ...] corresponds to self.samples[i].
    trio_results.sort(key=lambda x: x[0])
    
    # Collect results
    relationships = []
    parent_candidates = {}
    trio_scores_map = {}
    
    for sample_idx, best_trio, best_trio_score, top_indices in trio_results:
        valid_scores = total_scores[sample_idx].copy()
        # Match the filter behaviour used in Phase 2 args construction so
        # the scores reported in parent_candidates correspond to what Phase 2
        # actually saw at trio-scoring time.
        if use_anchor_union:
            valid_scores[sample_idx] = -np.inf  # self-pair only
        else:
            valid_scores[~cand_mask[sample_idx, :]] = -np.inf
        parent_candidates[sample_ids[sample_idx]] = [
            (sample_ids[x], valid_scores[x]) for x in top_indices
        ]
        trio_scores_map[sample_ids[sample_idx]] = best_trio_score
        if best_trio:
            p1n, p2n = sample_ids[best_trio[0]], sample_ids[best_trio[1]]
            relationships.append({
                'Sample': sample_ids[sample_idx], 
                'Generation': 'Unknown', 
                'Parent1': p1n, 
                'Parent2': p2n
            })
        else:
            relationships.append({
                'Sample': sample_ids[sample_idx], 
                'Generation': 'F1', 
                'Parent1': None, 
                'Parent2': None
            })

    rel_df = pd.DataFrame(relationships)
    
    name_to_gen = {row['Sample']: 'F1' for _, row in rel_df.iterrows() if pd.isna(row['Parent1'])}
    for _ in range(10): 
        for idx, row in rel_df.iterrows():
            if row['Generation'] != 'Unknown': continue
            p1, p2 = row['Parent1'], row['Parent2']
            if p1 in name_to_gen and p2 in name_to_gen:
                try:
                    g1 = int(name_to_gen[p1][1:])
                    g2 = int(name_to_gen[p2][1:])
                    gen = f"F{max(g1, g2) + 1}"
                    rel_df.at[idx, 'Generation'] = gen
                    name_to_gen[row['Sample']] = gen
                except: pass

    res = PedigreeResult(sample_ids, rel_df, parent_candidates, None, [], None, None, 
                         trio_scores_map, global_total_bins)
    res.perform_consistency_cutoff(contig_data_list, n_workers=n_workers)
    res.resolve_cycles()
    return res


# =============================================================================
# 7. CYCLE DETECTION AND RESOLUTION
# =============================================================================

def build_pedigree_graph(relationships_df):
    G = nx.DiGraph()
    for _, row in relationships_df.iterrows():
        child = row['Sample']
        G.add_node(child)
        p1 = row.get('Parent1')
        p2 = row.get('Parent2')
        if pd.notna(p1):
            G.add_edge(p1, child)
        if pd.notna(p2):
            G.add_edge(p2, child)
    return G


def detect_cycles(relationships_df):
    G = build_pedigree_graph(relationships_df)
    if nx.is_directed_acyclic_graph(G):
        return []
    return list(nx.simple_cycles(G))


def propagate_generations(relationships_df):
    fixed_df = relationships_df.copy()
    name_to_gen = {}
    for idx, row in fixed_df.iterrows():
        if pd.isna(row['Parent1']) and pd.isna(row['Parent2']):
            fixed_df.at[idx, 'Generation'] = 'F1'
            name_to_gen[row['Sample']] = 'F1'
        else:
            fixed_df.at[idx, 'Generation'] = 'Unknown'
    for _ in range(10):
        changed = False
        for idx, row in fixed_df.iterrows():
            if row['Generation'] != 'Unknown':
                continue
            p1, p2 = row['Parent1'], row['Parent2']
            p1_gen = name_to_gen.get(p1)
            p2_gen = name_to_gen.get(p2)
            if p1_gen and p2_gen:
                try:
                    g1 = int(p1_gen[1:])
                    g2 = int(p2_gen[1:])
                    gen = f"F{max(g1, g2) + 1}"
                    fixed_df.at[idx, 'Generation'] = gen
                    name_to_gen[row['Sample']] = gen
                    changed = True
                except:
                    pass
        if not changed:
            break
    return fixed_df


# =============================================================================
# 8. VISUALIZATION & WRAPPER
# =============================================================================

def draw_pedigree_tree(relationships_df, output_file="pedigree_tree.png"):
    if not HAS_VIS: return
    G = nx.DiGraph()
    gen_nodes = {'F1':[], 'F2':[], 'F3':[], 'Unknown':[]}
    parents_of = {}
    for _, row in relationships_df.iterrows():
        sample = row['Sample']; gen = row['Generation']
        if gen not in gen_nodes: gen_nodes[gen] = []
        gen_nodes[gen].append(sample)
        color = "#999999"
        if gen == 'F1': color = "#1f77b4"
        elif gen == 'F2': color = "#ff7f0e"
        elif gen == 'F3': color = "#2ca02c"
        G.add_node(sample, color=color, gen=gen)
        if pd.notna(row['Parent1']): 
            G.add_edge(row['Parent1'], sample); parents_of.setdefault(sample, []).append(row['Parent1'])
        if pd.notna(row['Parent2']): 
            G.add_edge(row['Parent2'], sample); parents_of.setdefault(sample, []).append(row['Parent2'])
    pos = {}
    node_y = {}
    layers = sorted([k for k in gen_nodes.keys() if k.startswith('F')], key=lambda x: int(x[1:]))
    if 'Unknown' in gen_nodes: layers.append('Unknown')
    for x_idx, gen in enumerate(layers):
        nodes = gen_nodes[gen]
        if not nodes: continue
        if x_idx == 0: nodes.sort()
        else: nodes.sort(key=lambda n: sum([node_y.get(p, 0.5) for p in parents_of.get(n, [])])/len(parents_of.get(n, [])) if parents_of.get(n) else 0.5, reverse=True)
        for i, n in enumerate(nodes):
            y = 1.0 - (i + 0.5) / len(nodes); pos[n] = (x_idx, y); node_y[n] = y
    plt.figure(figsize=(20, max(10, len(gen_nodes.get('F3', []))*0.2)))
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color=node_colors, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, width=0.5, arrows=False)
    to_label = gen_nodes.get('F1', []) + gen_nodes.get('F2', [])
    labels = {n:n for n in to_label}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.axis('off'); plt.tight_layout(); plt.savefig(output_file, dpi=150); plt.close()


def run_pedigree_inference_tolerance(tolerance_painting, sample_ids=None, snps_per_bin=100, 
                                     founder_block=None, recomb_rate=5e-8,
                                     output_prefix="pedigree",
                                     mismatch_penalty=DEFAULT_MISMATCH_PENALTY,
                                     n_workers=None):
    if founder_block is None: 
        raise ValueError("Founder Block is required.")
    if sample_ids is None: 
        sample_ids = [f"S_{i}" for i in range(len(tolerance_painting))]
    contig_input = [{'tolerance_painting': tolerance_painting, 'founder_block': founder_block}]
    result = infer_pedigree_multi_contig_tolerance(contig_input, sample_ids, top_k=20,
                                                    snps_per_bin=snps_per_bin, recomb_rate=recomb_rate,
                                                    mismatch_penalty=mismatch_penalty,
                                                    n_workers=n_workers)
    result.relationships.to_csv(f"{output_prefix}.ped", index=False)
    draw_pedigree_tree(result.relationships, output_file=f"{output_prefix}_tree.png")
    return result