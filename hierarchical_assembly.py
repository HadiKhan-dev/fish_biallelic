import thread_config
from thread_config import numba_thread_scope

import numpy as np
import math
import gc
import time
from tqdm import tqdm
import multiprocessing as mp
import multiprocessing.pool
from multiprocessing.shared_memory import SharedMemory

# Import your specific modules
import block_haplotypes
import block_linking
import hmm_matching
import beam_search_core
import analysis_utils
import chimera_resolution

import os

# env-gated per-phase wall-clock profiling of _process_single_batch (no effect
# on results when off): set BHD_HIER_PROFILE=1 to print a per-phase breakdown
# for batches whose super-block spans >= BHD_HIER_PROFILE_MIN_L sites.
_HIER_PROFILE = os.environ.get('BHD_HIER_PROFILE', '') not in ('', '0', 'false', 'False')
_HIER_PROFILE_MIN_L = int(os.environ.get('BHD_HIER_PROFILE_MIN_L', '500000'))


# =============================================================================
# NON-DAEMONIC FORKSERVER POOL
# =============================================================================
# Workers spawn via forkserver from a lightweight intermediate process,
# NOT forked from the parent's heap — no COW page dirtying from a
# ~200 GB parent.  Workers import modules fresh, receive _SHARED_META
# via the pool initializer (tiny dict), and attach to POSIX
# SharedMemory for global_probs/global_sites.
#
# Non-daemonic so workers can spawn their own child pools (HMM mesh
# generation, viterbi emissions).
#
# Uses stdlib multiprocessing (not dill-based multiprocess) because
# multiprocess doesn't properly support forkserver.
#
# IMPORTANT: The parent's entry script must NOT be named main.py,
# otherwise forkserver workers will re-execute it when importing __main__.

try:
    # Preloads configured in thread_config.py (imported above).
    _forkserver_ctx = mp.get_context('forkserver')
except (ValueError, AttributeError):
    _forkserver_ctx = mp.get_context('fork')

class _NoDaemonProcess(_forkserver_ctx.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class _NoDaemonContext(type(_forkserver_ctx)):
    Process = _NoDaemonProcess

class NoDaemonPool(multiprocessing.pool.Pool):
    """A Pool using forkserver context with non-daemonic workers."""
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _NoDaemonContext()
        super().__init__(*args, **kwargs)


# =============================================================================
# SHARED MEMORY MANAGEMENT
# =============================================================================
# Large numpy arrays (global_probs, global_sites) live in POSIX shared
# memory (/dev/shm), completely outside any process's Python heap.
# Parent creates segments and passes metadata via the pool initializer;
# workers attach by name for zero-copy views.

_SHARED_META = {}

# =============================================================================
# DYNAMIC THREAD REALLOCATION
# =============================================================================
# Workers track how many peers are active via a shared atomic counter.
# Between major phases (mesh → beam → chimera → reconstruction), each
# worker recalculates: threads = total_cores // active_workers.
#   - All workers running:    each gets its normal share
#   - 3 workers remain:       each gets total_cores // 3
#   - Last worker standing:   gets all total_cores
#
# Numba threads and inner pools are ALTERNATIVE ways to use cores, not
# additive.  Inner-pool phases (mesh generation) set numba threads to
# 1.  Numba-only phases (beam search, chimera resolution) give all
# dynamic threads to numba.  Prevents oversubscription.
#
# The numba thread pool ceiling is set to total_cores at worker init,
# so set_num_threads() can scale freely.  With OMP PASSIVE or TBB,
# idle threads sleep at zero CPU.
#
# Remainder distribution: when total_cores is not evenly divisible by
# active workers, floor(total/active) leaves `remainder = total %
# active` cores otherwise idle.  The extras-counter mechanism tracks
# how many workers hold +1 threads so exactly `remainder` workers get
# ceil and the rest get floor — zero idle cores.  This outer-pool
# counter is the most production-impactful of the project's extras
# counters: _get_dynamic_threads is passed as `dynamic_cores_fn` into
# inner stage-7 sequential paths
# (hmm_matching.generate_transition_probability_mesh_double_hmm with
# num_processes=1), so the outer remainder distribution propagates
# down to those inner workers' in-flight thread rescaling.
#
# _ACTIVE_COUNTER / _TOTAL_CORES: pool-wide active count + total budget.
# _EXTRA_COUNTER: pool-wide atomic int = workers currently holding +1.
# _I_HAVE_EXTRA: per-worker-process bool, True iff this worker has +1.
# All are None/False outside a properly-initialised pool worker.

_ACTIVE_COUNTER = None
_TOTAL_CORES = None
_EXTRA_COUNTER = None
_I_HAVE_EXTRA = False


def _try_claim_extra(remainder):
    """Atomically attempt to claim an extra thread from the remainder pool.

    Returns True if successfully claimed (and sets _I_HAVE_EXTRA),
    False if the pool is exhausted or the counter isn't set up.
    Idempotent: re-calling while already holding does not double-claim.

    Race analysis: the counter increment is guarded by its own lock.
    `current_extras < remainder` is evaluated INSIDE the lock so two
    workers can't both observe `current_extras = remainder - 1` and
    both push the counter to `remainder + 1`.  The local
    `_I_HAVE_EXTRA = True` happens-after the counter increment
    (within the same thread of execution).
    """
    global _I_HAVE_EXTRA
    if _I_HAVE_EXTRA:
        return True
    if _EXTRA_COUNTER is None:
        return False
    try:
        with _EXTRA_COUNTER.get_lock():
            if _EXTRA_COUNTER.value < remainder:
                _EXTRA_COUNTER.value += 1
                _I_HAVE_EXTRA = True
                return True
    except Exception:
        pass
    return False


def _try_release_extra():
    """Atomically release this worker's extra claim, if held.

    Defensive: clears the local flag even if the shared counter
    mutation fails.
    """
    global _I_HAVE_EXTRA
    if not _I_HAVE_EXTRA:
        return False
    if _EXTRA_COUNTER is None:
        _I_HAVE_EXTRA = False
        return False
    try:
        with _EXTRA_COUNTER.get_lock():
            _EXTRA_COUNTER.value -= 1
            _I_HAVE_EXTRA = False
            return True
    except Exception:
        _I_HAVE_EXTRA = False
        return False


def _init_worker_meta(meta_dict, total_cores, active_counter, extra_counter):
    """Pool initializer — called once per worker at creation time.

    Stores SharedMemory metadata so workers can attach to the global
    arrays, configures the numba thread pool ceiling to total_cores
    so set_num_threads can scale freely later (starts at 1 — the real
    value is set per phase in _process_single_batch), and wires the
    active/extra counters used by _get_dynamic_threads.

    With OMP PASSIVE or TBB threading layers, idle threads in an
    oversized pool sleep and consume zero CPU.  Avoid workqueue —
    those threads spin.

    Args:
        meta_dict: SharedMemory metadata dict.
        total_cores: int — total cores available to the pool.
        active_counter: mp.Value('i', 0) shared across workers.
        extra_counter: mp.Value('i', 0) for remainder distribution.
            Workers atomically claim/release from this pool so that
            exactly `remainder = total % active` workers hold
            ceil(total/active) threads and the rest hold floor —
            zero idle cores.
    """
    import os
    os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
    try:
        import numba
        numba.config.NUMBA_NUM_THREADS = total_cores
        numba.set_num_threads(1)
    except Exception:
        pass

    global _SHARED_META, _ACTIVE_COUNTER, _TOTAL_CORES
    global _EXTRA_COUNTER, _I_HAVE_EXTRA
    _SHARED_META = meta_dict
    _ACTIVE_COUNTER = active_counter
    _TOTAL_CORES = total_cores
    _EXTRA_COUNTER = extra_counter
    # Defensive: ensure no stale claim from worker recycling.
    _I_HAVE_EXTRA = False


def _get_dynamic_threads():
    """Compute optimal thread count for this worker based on active peers.

    Returns floor(total_cores / active_workers) + (1 if this worker
    holds an extra-claim else 0), clamped to [1, total_cores].

    Remainder distribution: exactly `remainder = total % active`
    workers hold a +1 thread; the rest hold floor.  Net effect: total
    threads in use = total_cores at all times, with no idle cores.
    This call may claim or release an extra based on the current
    `remainder`.

    The read of active_counter.value is intentionally lock-free — a
    slightly stale count is fine since we recheck between every major
    phase.  The extra-counter lock IS acquired briefly during
    claim/release because those mutations must be atomic.

    Returns 1 in the sequential path (when _ACTIVE_COUNTER is None).
    """
    if _ACTIVE_COUNTER is None or _TOTAL_CORES is None:
        return 1
    active = max(_ACTIVE_COUNTER.value, 1)
    floor = _TOTAL_CORES // active
    remainder = _TOTAL_CORES - floor * active

    # Adjust extra-claim based on current remainder.
    if _EXTRA_COUNTER is not None:
        try:
            current_extras = _EXTRA_COUNTER.value
        except Exception:
            current_extras = 0
        if not _I_HAVE_EXTRA:
            if current_extras < remainder:
                _try_claim_extra(remainder)
        else:
            if current_extras > remainder:
                _try_release_extra()

    return max(1, floor + (1 if _I_HAVE_EXTRA else 0))


def _apply_dynamic_threads():
    """
    Recompute and apply numba thread allocation.
    Returns the thread count.
    
    Use this before phases that use numba directly (beam search,
    chimera resolution, reconstruction). Do NOT use before phases
    that spawn inner pools — those should set numba to 1 instead.
    """
    import numba
    n = _get_dynamic_threads()
    numba.set_num_threads(n)
    return n


def _create_shared_array(array, label):
    """
    Copy a numpy array into a POSIX shared memory segment.
    
    Returns:
        (SharedMemory handle, metadata_dict)
    """
    shm = SharedMemory(create=True, size=array.nbytes)
    shared_view = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    np.copyto(shared_view, array)
    metadata = {
        'name': shm.name,
        'shape': array.shape,
        'dtype': array.dtype,
    }
    return shm, metadata


def _attach_shared_array(metadata):
    """
    Attach to an existing shared memory segment and return a numpy view.
    
    Returns:
        (SharedMemory handle, numpy array view)
    """
    shm = SharedMemory(name=metadata['name'], create=False)
    array = np.ndarray(metadata['shape'], dtype=metadata['dtype'], buffer=shm.buf)
    return shm, array


# =============================================================================
# DATA HELPERS
# =============================================================================

def create_downsampled_proxy(block, max_sites=2000):
    """
    Creates a lightweight 'Proxy' of a BlockResult.
    Returns the original block if it's small enough.
    """
    total_sites = len(block.positions)
    
    if total_sites <= max_sites:
        return block
        
    stride = math.ceil(total_sites / max_sites)
    
    new_pos = np.ascontiguousarray(block.positions[::stride])
    
    new_haps = {}
    for k, v in block.haplotypes.items():
        if v.ndim > 1:
            new_haps[k] = np.ascontiguousarray(v[::stride, :])
        else:
            new_haps[k] = np.ascontiguousarray(v[::stride])
            
    if block.keep_flags is not None:
        new_flags = np.ascontiguousarray(block.keep_flags[::stride])
    else:
        new_flags = None

    new_reads = None
    if block.reads_count_matrix is not None:
        new_reads = np.ascontiguousarray(block.reads_count_matrix[:, ::stride, :])
        
    new_probs = None
    if block.probs_array is not None:
        new_probs = np.ascontiguousarray(block.probs_array[:, ::stride, :])

    proxy = block_haplotypes.BlockResult(
        positions=new_pos,
        haplotypes=new_haps,
        keep_flags=new_flags,
        reads_count_matrix=new_reads,
        probs_array=new_probs
    )
    
    return proxy

def convert_reconstruction_to_superblock(reconstructed_data, original_blocks, global_probs=None, global_sites=None):
    """
    Packages reconstruction results into a BlockResult (Super-Block).
    """
    if not reconstructed_data:
        return None

    super_haplotypes = {}
    for i, data in enumerate(reconstructed_data):
        super_haplotypes[i] = data['haplotype'] 

    super_positions = reconstructed_data[0]['positions']

    super_flags = []
    for b in original_blocks:
        if b.keep_flags is not None:
            super_flags.extend(b.keep_flags)
        else:
            super_flags.extend(np.ones(len(b.positions), dtype=int))
    super_flags = np.array(super_flags)
    
    super_probs = None
    if global_probs is not None and global_sites is not None:
        indices = np.searchsorted(global_sites, super_positions)
        # parallel gather (bit-identical to global_probs[:, indices, :]); the
        # numpy advanced-index version is single-threaded and copies the whole
        # (N, n_super, 3) probs -- ~5.4 GB / ~10-20s at L4 -- inside the batch
        # worker.  See chimera_resolution._gather_samples_numba.
        super_probs = chimera_resolution._gather_samples_numba(global_probs, indices)
    else:
        probs_list = []
        for b in original_blocks:
            if b.probs_array is not None:
                probs_list.append(b.probs_array)
            elif b.reads_count_matrix is not None:
                _, probs = analysis_utils.reads_to_probabilities(b.reads_count_matrix)
                probs_list.append(probs)
        
        if probs_list:
            super_probs = np.concatenate(probs_list, axis=1)
    
    reads_list = []
    for b in original_blocks:
        if b.reads_count_matrix is not None:
            reads_list.append(b.reads_count_matrix)
    
    super_reads = None
    if reads_list and len(reads_list) == len(original_blocks):
        super_reads = np.concatenate(reads_list, axis=1)

    super_block = block_haplotypes.BlockResult(
        positions=super_positions,
        haplotypes=super_haplotypes,
        keep_flags=super_flags,
        reads_count_matrix=super_reads,
        probs_array=super_probs
    )
    
    return super_block


def compute_max_gap(blocks, recomb_rate, n_generations, recomb_tolerance):
    """
    Compute the maximum gap to use for beam search transition lookups.
    """
    block_spans = [b.positions[-1] - b.positions[0] for b in blocks if len(b.positions) > 1]
    if not block_spans:
        return 1
    
    avg_block_span = np.mean(block_spans)
    recombs_per_step = avg_block_span * recomb_rate * n_generations
    
    if recombs_per_step <= 0:
        return len(blocks)
    
    max_gap = max(1, 1 + int(math.floor(recomb_tolerance / recombs_per_step)))
    
    return max_gap


# =============================================================================
# BATCH WORKER FUNCTION (For Parallel Processing)
# =============================================================================

def _process_single_batch(args):
    """Worker function to process a single batch.

    Attaches to POSIX shared memory segments for global_probs and
    global_sites using metadata from _SHARED_META (set by pool
    initializer).  Between major phases, dynamically adjusts numba
    thread count based on how many peer workers are still active.

    Phase-by-phase parallelism:
      Mesh generation (sequential with dynamic numba): gaps processed
        one at a time; between each gap numba threads = total_cores
        // active_workers.  prange over samples provides equivalent
        throughput to pool-based processing but can adapt mid-
        computation.
      Numba-only phases (beam search, chimera resolution,
        reconstruction): no inner pool; all dynamic threads go to
        numba in this process.

    Explicitly deletes large intermediates and calls malloc_trim
    between steps to release freed pages back to the OS.

    Returns dict with 'batch_idx', 'super_block', and 'status'.
    """
    import ctypes
    import numba
    _libc = ctypes.CDLL("libc.so.6")

    (b_idx, start_i, end_i, original_blocks_list,
     use_hmm_linking, recomb_rate, beam_width, max_founders,
     max_sites_for_linking, n_generations, recomb_tolerance,
     top_n_swap, max_cr_iterations, paint_penalty, min_hotspot_samples,
     cc_scale, inner_num_processes, verbose) = args

    # Attach to shared memory (zero-copy).
    shm_probs, global_probs = _attach_shared_array(_SHARED_META['probs'])
    shm_sites, global_sites = _attach_shared_array(_SHARED_META['sites'])

    try:
        # Register this worker as active.
        if _ACTIVE_COUNTER is not None:
            with _ACTIVE_COUNTER.get_lock():
                _ACTIVE_COUNTER.value += 1

        # Initial allocation — first phase uses inner pools, so start
        # numba at 1.
        _get_dynamic_threads()
        numba.set_num_threads(1)

        original_portion = block_haplotypes.BlockResults(original_blocks_list)

        if len(original_portion) < 2:
            return {
                'batch_idx': b_idx,
                'super_block': original_portion[0],
                'status': 'passthrough'
            }

        # --- env-gated per-phase wall timing (no effect on results when off:
        # _acc only mutates _pt inside `if _prof`, and the report is gated) ---
        _prof = _HIER_PROFILE
        _pt = {}
        _t_batch = time.perf_counter()
        def _acc(_key, _t0):
            _e = _pt.get(_key)
            _dt = time.perf_counter() - _t0
            if _e is None:
                _pt[_key] = [_dt, 1]
            else:
                _e[0] += _dt
                _e[1] += 1

        _t = time.perf_counter()
        # 1. Create proxies (downsample large blocks for linking).
        proxy_list = []
        for b in original_portion:
            proxy_list.append(create_downsampled_proxy(b, max_sites_for_linking))
        portion_proxy = block_haplotypes.BlockResults(proxy_list)

        # 2. Slice to batch-relevant sites only.
        all_positions = np.concatenate([b.positions for b in original_portion])
        batch_indices = np.searchsorted(global_sites, all_positions)
        idx_min, idx_max = batch_indices.min(), batch_indices.max()
        batch_probs = np.ascontiguousarray(global_probs[:, idx_min:idx_max+1, :])
        batch_sites = np.ascontiguousarray(global_sites[idx_min:idx_max+1])

        # 3. Compute max gap.
        if n_generations is not None and recomb_tolerance is not None:
            beam_max_gap = compute_max_gap(original_blocks_list, recomb_rate,
                                           n_generations, recomb_tolerance)
        else:
            beam_max_gap = None
        if _prof: _acc('setup(proxy+slice+gap)', _t)

        # =================================================================
        # 4. Generate Mesh — DYNAMIC SEQUENTIAL phase.
        # Emissions: ThreadPoolExecutor (pure numpy, fast, no numba).
        # Mesh EM: sequential over gaps with dynamic numba threads;
        # _get_dynamic_threads called between each gap (and inside the
        # EM loop via dynamic_cores_fn) so this worker scales up as
        # peers finish.
        # =================================================================
        dyn_threads = _get_dynamic_threads()
        pool_budget = max(inner_num_processes, dyn_threads)

        if use_hmm_linking:
            # Emissions: ThreadPoolExecutor (threads release GIL inside
            # the numba kernel; no oversubscription risk).
            _t = time.perf_counter()
            viterbi_emissions = hmm_matching.generate_viterbi_block_emissions(
                batch_probs, batch_sites, portion_proxy, num_processes=pool_budget
            )
            if _prof: _acc('mesh_emissions', _t)
            _t = time.perf_counter()
            mesh = hmm_matching.generate_transition_probability_mesh_double_hmm(
                None, None, portion_proxy,
                recomb_rate=recomb_rate,
                use_standard_baum_welch=False,
                precalculated_viterbi_emissions=viterbi_emissions,
                num_processes=1,
                dynamic_cores_fn=_get_dynamic_threads,
            )
            if _prof: _acc('mesh_transition', _t)
            del viterbi_emissions
        else:
            _t = time.perf_counter()
            mesh = block_linking.generate_transition_probability_mesh(
                batch_probs, batch_sites, portion_proxy,
                use_standard_baum_welch=True,
                num_processes=pool_budget
            )
            if _prof: _acc('mesh_transition', _t)

        # =================================================================
        # 5. Beam Search — NUMBA-ONLY phase.  No inner pool; give all
        # dynamic threads to numba.
        # =================================================================
        _apply_dynamic_threads()
        _t = time.perf_counter()
        beam_results = beam_search_core.run_full_mesh_beam_search(
            portion_proxy, mesh, beam_width=beam_width,
            max_gap=beam_max_gap, verbose=verbose
        )
        if _prof: _acc('beam_search', _t)

        if not beam_results:
            return {
                'batch_idx': b_idx,
                'super_block': None,
                'status': 'beam_search_failed'
            }

        fast_mesh = beam_search_core.FastMesh(portion_proxy, mesh)

        # Free mesh — fast_mesh has what it needs.
        del mesh
        _libc.malloc_trim(0)

        # =================================================================
        # 6. Selection + Swap + CR — NUMBA-ONLY phase.
        # =================================================================
        _apply_dynamic_threads()
        _t = time.perf_counter()
        resolved_beam = chimera_resolution.select_and_resolve(
            beam_results=beam_results,
            fast_mesh=fast_mesh,
            batch_blocks=list(original_portion),
            global_probs=batch_probs,
            global_sites=batch_sites,
            max_founders=max_founders,
            top_n_swap=top_n_swap,
            max_cr_iterations=max_cr_iterations,
            paint_penalty=paint_penalty,
            min_hotspot_samples=min_hotspot_samples,
            cc_scale=cc_scale,
            num_threads=_get_dynamic_threads,
        )
        if _prof: _acc('select_and_resolve(CR)', _t)

        del beam_results
        _libc.malloc_trim(0)

        # =================================================================
        # 7. Reconstruction — NUMBA-ONLY phase.
        # =================================================================
        _apply_dynamic_threads()
        _t = time.perf_counter()
        reconstructed_data = beam_search_core.reconstruct_haplotypes_from_beam(
            resolved_beam, fast_mesh, original_portion
        )
        if _prof: _acc('reconstruction', _t)

        del resolved_beam, fast_mesh
        _libc.malloc_trim(0)

        # 8. Package.
        _t = time.perf_counter()
        super_block = convert_reconstruction_to_superblock(
            reconstructed_data, original_portion, batch_probs, batch_sites
        )
        if _prof: _acc('package', _t)

        del reconstructed_data, batch_probs, batch_sites, portion_proxy, proxy_list
        _libc.malloc_trim(0)

        # 9. Structural chimera pruning.
        _t = time.perf_counter()
        super_block = beam_search_core.prune_superblock_chimeras(super_block)
        if _prof: _acc('prune', _t)

        if _prof:
            try:
                _nL = len(super_block.positions) if super_block is not None else 0
                if _nL >= _HIER_PROFILE_MIN_L:
                    _wall = time.perf_counter() - _t_batch
                    _timed = sum(v[0] for v in _pt.values())
                    _aw = _ACTIVE_COUNTER.value if _ACTIVE_COUNTER is not None else 1
                    _hdr = (f"  [hier profile] batch={b_idx} N={global_probs.shape[0]} "
                            f"L={_nL} | numba_threads={numba.get_num_threads()} "
                            f"active_workers={_aw} | batch_wall={_wall:.1f}s")
                    _body = "\n".join(
                        f"      {_k:24s} {_v[0]:7.2f}s  ({_v[1]:5d} calls)"
                        for _k, _v in sorted(_pt.items(), key=lambda kv: -kv[1][0]))
                    _oth = f"      {'other(numpy+setup)':24s} {_wall - _timed:7.2f}s"
                    print(_hdr + "\n" + _body + "\n" + _oth, flush=True)
            except Exception:
                pass

        return {
            'batch_idx': b_idx,
            'super_block': super_block,
            'status': 'success' if super_block else 'reconstruction_failed'
        }
    finally:
        # Release any held extra FIRST, then decrement the active
        # counter, so peers see the freed extra-slot before the
        # decremented active count.  _try_release_extra is a no-op
        # when this worker holds no extra or _EXTRA_COUNTER is None.
        _try_release_extra()
        if _ACTIVE_COUNTER is not None:
            with _ACTIVE_COUNTER.get_lock():
                _ACTIVE_COUNTER.value -= 1
        # Detach from shared memory (parent unlinks).
        shm_probs.close()
        shm_sites.close()


# =============================================================================
# MAIN DRIVER
# =============================================================================

def run_hierarchical_step(input_blocks, global_probs, global_sites,
                          batch_size=10,
                          # Linking Parameters
                          use_hmm_linking=True,
                          recomb_rate=5e-8,
                          # Search Parameters
                          beam_width=200,
                          # Selection Parameters
                          max_founders=12,
                          # Memory Safety
                          max_sites_for_linking=2000,
                          # Max Gap Parameters
                          n_generations=None,
                          recomb_tolerance=0.5,
                          # Chimera Resolution Parameters
                          top_n_swap=20,
                          max_cr_iterations=10,
                          paint_penalty=10.0,
                          min_hotspot_samples=5,
                          cc_scale=0.5,
                          # Parallelization
                          num_processes=16,
                          maxtasksperchild=None,
                          min_gb_per_worker=4.0,
                          # Output control
                          verbose=False,
                          # Per-level post-stitch refinement (see level_refine.py)
                          refine_after_stitch=True,
                          refine_max_iter=None):
    """Performs one level of Hierarchical Assembly.

    Memory strategy:
      - global_probs/global_sites placed in POSIX shared memory (/dev/shm).
      - Workers spawned via forkserver — start from a lightweight
        intermediate process, NOT from the parent's large heap (no COW).
      - Workers attach to shared segments by name (zero-copy).
      - batch_probs slicing keeps inner pools pickling small arrays.
      - Non-daemonic workers so they can spawn inner child pools at L2+.
      - maxtasksperchild recycles workers after N batches, releasing
        accumulated memory (Python doesn't return freed pages to OS).
      - global_probs downcast to float32 (halves shared memory + all
        downstream tensors).
      - Worker count auto-capped based on available RAM /
        min_gb_per_worker.

    Dynamic thread reallocation: shared counter tracks active workers;
    between major phases each worker recalculates
    threads = total_cores // active_workers (+1 for `remainder %
    active` workers).  See the DYNAMIC THREAD REALLOCATION header
    block.

    Args:
      num_processes: Maximum total cores.  Strict ceiling on both
          concurrent workers AND total thread allocation.  Function
          may use fewer workers if RAM is tight, but thread allocation
          across workers will sum to at most num_processes.
      maxtasksperchild: Recycle workers after this many batches.  Set
          to 1 to prevent memory accumulation from glibc malloc
          fragmentation.
      min_gb_per_worker: GB of RAM to budget per concurrent worker.
          Used to auto-cap worker count: max_workers = available_ram
          / min_gb_per_worker.  Increase if blocks have many
          haplotypes (>15); decrease if RAM is tight but blocks are
          small.

    IMPORTANT: The entry script must NOT be named main.py — otherwise
    forkserver workers will re-execute it.
    """
    total_blocks = len(input_blocks)
    num_batches = math.ceil(total_blocks / batch_size)
    
    # num_processes is the user's ceiling — never exceed it for either
    # concurrent workers or total thread allocation.
    total_cores = num_processes
    
    print(f"\n--- Starting Hierarchical Step ---")
    print(f"Input: {total_blocks} blocks -> Target: ~{num_batches} Super-Blocks")
    if n_generations is not None:
        preview_max_gap = compute_max_gap(list(input_blocks), recomb_rate, 
                                           n_generations, recomb_tolerance)
        print(f"Max gap: {preview_max_gap} (n_gen={n_generations}, tol={recomb_tolerance}, rate={recomb_rate})")
    else:
        print(f"Max gap: unlimited (n_generations not specified)")
    
    # Strip redundant probs_array from input blocks.  Each BlockResult
    # carries probs_array of shape (n_samples, ~200, 3) per block;
    # across a full chromosome that totals the same size as
    # global_probs (~5 GB).  Workers access sample data via
    # global_probs in shared memory, so probs_array in blocks is
    # redundant.  Stripping reduces parent process memory AND pickle
    # size when sending blocks to workers as task arguments.
    import ctypes as _ctypes
    _stripped_bytes = 0
    for block in input_blocks:
        if hasattr(block, 'probs_array') and block.probs_array is not None:
            _stripped_bytes += block.probs_array.nbytes
            block.probs_array = None
    if _stripped_bytes > 0:
        gc.collect()
        try:
            _ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        print(f"  Stripped probs_array from blocks ({_stripped_bytes / (1024**3):.1f} GB freed)")

    # Downcast to float32: global_probs is float64 from R01 (HDBSCAN
    # needs float64) but assembly only uses it for emission scoring
    # where float32 precision is sufficient.  Halves shared memory,
    # per-worker batch_probs slices, and all downstream emission/
    # chimera tensors (they inherit dtype).
    if global_probs.dtype == np.float64:
        global_probs = global_probs.astype(np.float32)
        print(f"  Downcast global_probs to float32 ({global_probs.nbytes / (1024**3):.1f} GB)")

    # Also downcast block haplotypes (soft probabilities) if float64.
    for block in input_blocks:
        for k, h in block.haplotypes.items():
            if h.dtype == np.float64:
                block.haplotypes[k] = h.astype(np.float32)

    # Create POSIX shared memory for the global arrays.
    t0 = time.time()
    shm_probs, probs_meta = _create_shared_array(global_probs, 'global_probs')
    shm_sites, sites_meta = _create_shared_array(global_sites, 'global_sites')

    shared_meta = {
        'probs': probs_meta,
        'sites': sites_meta,
    }

    probs_gb = global_probs.nbytes / (1024**3)
    print(f"  Shared memory created: {probs_gb:.1f} GB probs + sites ({time.time()-t0:.1f}s)")

    # Auto-size worker count based on available RAM, read AFTER shared
    # memory creation so it already accounts for parent + loaded data
    # + shared segments.
    max_by_ram = num_processes  # fallback: no capping
    try:
        with open('/proc/meminfo') as _f:
            for _line in _f:
                if _line.startswith('MemAvailable:'):
                    mem_available_gb = int(_line.split()[1]) / (1024 * 1024)
                    max_by_ram = max(1, int(mem_available_gb / min_gb_per_worker))
                    break
    except Exception:
        pass  # non-Linux or /proc unavailable — use num_processes as-is
    
    outer_workers = min(num_batches, num_processes, max_by_ram)
    inner_num_processes = max(1, total_cores // outer_workers)
    
    # Preview
    print(f"Parallelism: {outer_workers} outer workers x {inner_num_processes} inner cores "
          f"= {outer_workers * inner_num_processes} total")
    if outer_workers < num_processes and outer_workers < num_batches:
        print(f"  Workers capped by RAM: {mem_available_gb:.0f} GB available / "
              f"{min_gb_per_worker} GB per worker = {max_by_ram} max")
    print(f"  Dynamic threading: enabled (ceiling={total_cores} cores)")
    
    if maxtasksperchild is not None:
        tasks_per_worker = math.ceil(num_batches / outer_workers)
        n_recycles = max(0, tasks_per_worker // maxtasksperchild - 1)
        print(f"  Worker recycling: every {maxtasksperchild} batches "
              f"(~{tasks_per_worker} tasks/worker, ~{n_recycles} recycles each)")
    
    # Shared counters for dynamic thread reallocation.  active_counter
    # tracks live worker count; extra_counter distributes the
    # remainder = total_cores % active so no cores stay idle.  Same
    # forkserver context for shared-memory consistency.  See the
    # DYNAMIC THREAD REALLOCATION header block for the full mechanism.
    active_counter = _forkserver_ctx.Value('i', 0)
    extra_counter = _forkserver_ctx.Value('i', 0)
    
    # =====================================================================
    # Prepare worker arguments
    # =====================================================================
    worker_args = []
    
    for b_idx in range(num_batches):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, total_blocks)
        
        original_blocks_list = list(input_blocks[start_i:end_i])
        
        worker_args.append((
            b_idx, start_i, end_i, original_blocks_list,
            use_hmm_linking, recomb_rate, beam_width, max_founders,
            max_sites_for_linking, n_generations, recomb_tolerance,
            top_n_swap, max_cr_iterations, paint_penalty, min_hotspot_samples,
            cc_scale, inner_num_processes, verbose
        ))
    
    # =====================================================================
    # Process batches
    # =====================================================================
    # Belt-and-suspenders: temporarily clear __main__.__file__ so
    # forkserver workers don't re-execute the entry script, even if
    # the caller forgot to add main guards to their pipeline file.
    import sys as _sys
    _main_mod = _sys.modules.get('__main__')
    _saved_main_file = getattr(_main_mod, '__file__', None)
    _saved_main_spec = getattr(_main_mod, '__spec__', None)
    if _main_mod is not None:
        if hasattr(_main_mod, '__file__'):
            del _main_mod.__file__
        _main_mod.__spec__ = None
    
    try:
        if num_processes > 1:
            t0 = time.time()
            pool = NoDaemonPool(
                processes=outer_workers,
                initializer=_init_worker_meta,
                initargs=(shared_meta, total_cores, active_counter, extra_counter),
                maxtasksperchild=maxtasksperchild
            )
            print(f"  Pool creation ({outer_workers} workers): {time.time()-t0:.1f}s")
            
            t0 = time.time()
            results = []
            try:
                for result in tqdm(
                    pool.imap_unordered(_process_single_batch, worker_args),
                    total=num_batches,
                    desc="Processing Batches"
                ):
                    results.append(result)
                pool.close()
            except (KeyboardInterrupt, SystemExit, Exception) as e:
                pool.terminate()
                raise
            finally:
                pool.join()
            print(f"  Pool work + result collection: {time.time()-t0:.1f}s")
        else:
            # Sequential execution — for testing/debugging.  Set
            # module globals directly so _process_single_batch can
            # use them without going through the pool initialiser.
            # _ACTIVE_COUNTER stays None (no peers to coordinate
            # with); the caller is expected to call
            # numba.set_num_threads(total_cores) directly.
            global _SHARED_META, _ACTIVE_COUNTER, _TOTAL_CORES
            global _EXTRA_COUNTER, _I_HAVE_EXTRA
            _SHARED_META = shared_meta
            _ACTIVE_COUNTER = None
            _TOTAL_CORES = total_cores
            _EXTRA_COUNTER = None
            _I_HAVE_EXTRA = False
            results = []
            for args in tqdm(worker_args, desc="Processing Batches"):
                results.append(_process_single_batch(args))
    finally:
        # Clean up shared memory (always, even on error)
        shm_probs.close()
        shm_probs.unlink()
        shm_sites.close()
        shm_sites.unlink()
        # Restore __main__ attributes
        if _main_mod is not None:
            if _saved_main_file is not None:
                _main_mod.__file__ = _saved_main_file
            _main_mod.__spec__ = _saved_main_spec
    
    # Sort by batch index and collect super blocks
    results = sorted(results, key=lambda x: x['batch_idx'])
    
    output_super_blocks = []
    success_count = 0
    passthrough_count = 0
    failed_count = 0
    
    for result in results:
        if result['super_block'] is not None:
            output_super_blocks.append(result['super_block'])
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'passthrough':
                passthrough_count += 1
        else:
            failed_count += 1
            
    print(f"Hierarchical Step Complete. Produced {len(output_super_blocks)} Super-Blocks.")
    print(f"  Success: {success_count}, Passthrough: {passthrough_count}, Failed: {failed_count}")
    
    # =====================================================================
    # Per-level post-stitch REFINEMENT (default on).
    # After the level is stitched, refine each super-block AGAINST ITSELF
    # (self-block painting LL, no access to lower levels) up to
    # refine_max_iter fixed-point passes per block.  Replaces a haplotype
    # only when a carrier-re-derived version raises the self-block painting
    # likelihood (dLL > 0); never adds.  refine_level sizes its own pool to
    # the block count and uses the same dynamic inner-thread machinery as
    # this function, so it stays core-saturated even at L2/L3/L4 where there
    # are few blocks.  Controlled by refine_after_stitch (default True).
    # See level_refine.py for the validation (converges; net-positive on
    # truth; 0 founders lost).
    # =====================================================================
    if refine_after_stitch and len(output_super_blocks) > 0:
        import level_refine
        print(f"  Refining {len(output_super_blocks)} super-blocks "
              f"(self-scoring, up to {refine_max_iter or '3x#haps'} passes/block)...")
        refined = level_refine.refine_level(
            output_super_blocks, global_probs, global_sites,
            n_workers=num_processes, paint_penalty=paint_penalty,
            max_block_iter=refine_max_iter, verbose=verbose)
        output_super_blocks = list(refined)
    
    return block_haplotypes.BlockResults(output_super_blocks)