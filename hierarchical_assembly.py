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


# =============================================================================
# NON-DAEMONIC FORKSERVER POOL
# =============================================================================
# Workers are spawned via forkserver, NOT forked from the parent process.
# This means workers start from a lightweight forkserver process (~50 MB),
# not the parent's ~200 GB Python heap. No COW page dirtying.
#
# Workers import modules fresh, receive _SHARED_META via the pool
# initializer (tiny dict, cheap to pickle), and attach to POSIX
# SharedMemory for global_probs/global_sites.
#
# Non-daemonic so workers can spawn their own child pools for internal
# parallelism at L2+ (HMM mesh generation, viterbi emissions).
#
# Uses stdlib multiprocessing (not dill-based multiprocess) because
# multiprocess doesn't properly support forkserver.
#
# IMPORTANT: The parent's entry script must NOT be named main.py,
# otherwise forkserver workers will re-execute it when importing __main__.

try:
    # Preloading is configured in thread_config.py (imported above).
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
# Uses POSIX shared memory to share large numpy arrays. Data lives in
# /dev/shm, completely outside any process's Python heap.
#
# Parent creates segments and passes metadata (name, shape, dtype) to
# workers via pool initializer. Workers attach by name and get zero-copy
# numpy views.

_SHARED_META = {}

# =============================================================================
# DEBUG INSTRUMENTATION (opt-in via HIERARCHICAL_DEBUG_DIR env var)
# =============================================================================
# When the env var HIERARCHICAL_DEBUG_DIR is set to a writable path, this
# module will:
#   - Save the worker_args tuple for each batch to <dir>/batch_NNNNN_args.pkl
#     at the start of _process_single_batch (before any numba code runs), so
#     a segfaulted batch's exact input can be replayed offline.
#   - Append a phase marker to <dir>/batch_NNNNN_phases.log at every major
#     phase transition (mesh/beam/chimera/reconstruction/etc.), line-buffered.
#     After a hang, the last marker in a batch's log tells us which phase
#     segfaulted.
#   - Write an empty <dir>/batch_NNNNN_done file when the batch returns
#     normally, so the crashed batch is the one with an _args.pkl and a
#     _phases.log but NO _done file.
#   - Save contig-level context (global_probs, global_sites, all kwargs) to
#     <dir>/context.pkl once at the start of run_hierarchical_step, so
#     replay can reconstruct the shared-memory state.
# When the env var is empty/unset, all of these are no-ops.
# This instrumentation is ALWAYS safe to leave in — it never affects the
# pipeline output, only emits diagnostic files.
import os as _os_dbg
import pickle as _pickle_dbg

def _debug_dir():
    """Return HIERARCHICAL_DEBUG_DIR if set, else empty string (disabled)."""
    return _os_dbg.environ.get('HIERARCHICAL_DEBUG_DIR', '')

def _debug_save_args(args):
    """Pickle the worker_args tuple for this batch.  Blocks inside args have
    already been stripped of probs_array by run_hierarchical_step, so the
    pickle is light (tens of MB typical)."""
    d = _debug_dir()
    if not d:
        return
    try:
        _os_dbg.makedirs(d, exist_ok=True)
        b_idx = args[0]
        path = _os_dbg.path.join(d, f'batch_{b_idx:05d}_args.pkl')
        with open(path, 'wb') as f:
            _pickle_dbg.dump(args, f, protocol=_pickle_dbg.HIGHEST_PROTOCOL)
    except Exception:
        pass  # never let instrumentation crash the worker

def _debug_mark_phase(b_idx, phase, extra=""):
    """Append a phase marker to this batch's log file.  Line-buffered so the
    file is readable in real time from another terminal while the worker
    runs (`tail -f batch_NNNNN_phases.log`)."""
    d = _debug_dir()
    if not d:
        return
    try:
        path = _os_dbg.path.join(d, f'batch_{b_idx:05d}_phases.log')
        with open(path, 'a', buffering=1) as f:
            f.write(f"{time.time():.6f} {phase} pid={_os_dbg.getpid()} {extra}\n")
    except Exception:
        pass

def _debug_mark_done(b_idx):
    """Write the batch's completion marker.  A batch with _args.pkl and
    _phases.log but NO _done file is the one that died."""
    d = _debug_dir()
    if not d:
        return
    try:
        path = _os_dbg.path.join(d, f'batch_{b_idx:05d}_done')
        with open(path, 'w') as f:
            f.write(f"{time.time():.6f} pid={_os_dbg.getpid()}\n")
    except Exception:
        pass

def _debug_save_context(context_dict):
    """Save contig-level context so replay can reconstruct shared memory.
    Called once at the start of run_hierarchical_step.  Saves ~1-5 GB per
    contig (global_probs is the bulk)."""
    d = _debug_dir()
    if not d:
        return
    try:
        _os_dbg.makedirs(d, exist_ok=True)
        # Clean out any previous run's batch files so a fresh hang is
        # unambiguous (only the current run's batches present)
        import glob as _glob
        for old in _glob.glob(_os_dbg.path.join(d, 'batch_*')):
            try: _os_dbg.unlink(old)
            except OSError: pass
        path = _os_dbg.path.join(d, 'context.pkl')
        with open(path, 'wb') as f:
            _pickle_dbg.dump(context_dict, f, protocol=_pickle_dbg.HIGHEST_PROTOCOL)
        print(f"  [DEBUG] context saved: {path}")
    except Exception as e:
        print(f"  [DEBUG] WARNING: failed to save context: {e}")

# =============================================================================
# DYNAMIC THREAD REALLOCATION
# =============================================================================
# Workers track how many peers are active via a shared atomic counter.
# Between major phases (mesh → beam → chimera → reconstruction), each
# worker recalculates its thread allocation as:
#
#   threads = total_cores // active_workers
#
# This means:
#   - When all workers run: each gets its normal share (e.g. 1 at L1)
#   - When 3 workers remain: each gets total_cores // 3 (e.g. 37)
#   - Last worker standing: gets all total_cores
#
# IMPORTANT: Numba threads and inner pools are ALTERNATIVE ways to use
# cores, not additive. When a phase uses inner pools (mesh generation),
# numba threads are set to 1 so each inner worker uses 1 thread.
# When a phase uses numba directly (beam search, chimera resolution),
# all dynamic threads go to numba. This prevents oversubscription.
#
# The numba thread pool ceiling is set to total_cores at worker init,
# so set_num_threads() can freely scale up. With OMP PASSIVE or TBB,
# idle threads sleep and cost zero CPU.

_ACTIVE_COUNTER = None   # mp.Value('i', 0) — shared across all workers
_TOTAL_CORES = None       # Total cores available (e.g. 112)


def _init_worker_meta(meta_dict, total_cores, active_counter):
    """
    Pool initializer — called once per worker at creation time.
    Stores SharedMemory metadata so workers can attach to the segments.
    
    Sets the numba thread pool ceiling to total_cores (not inner_num_processes)
    so that workers can dynamically scale up their thread count as peers
    finish. The pool is created lazily on first parallel function call.
    
    With OMP PASSIVE or TBB threading layers, idle threads in an oversized
    pool sleep and consume zero CPU. With workqueue, idle threads spin —
    avoid using workqueue with dynamic threading.
    """
    import os
    os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
    try:
        import numba
        # Set ceiling to total_cores so set_num_threads can scale up later
        numba.config.NUMBA_NUM_THREADS = total_cores
        # Start conservative — _process_single_batch will set the real value
        numba.set_num_threads(1)
    except Exception:
        pass
    
    global _SHARED_META, _ACTIVE_COUNTER, _TOTAL_CORES
    _SHARED_META = meta_dict
    _ACTIVE_COUNTER = active_counter
    _TOTAL_CORES = total_cores


def _get_dynamic_threads():
    """
    Compute optimal thread count for this worker based on active peers.
    
    Uses total_cores // active_workers, clamped to [1, total_cores].
    The read of active_counter.value is intentionally lock-free — a
    slightly stale count (off by 1-2) is fine since we recheck between
    every major phase. The cost of being briefly wrong is a few seconds
    of mild over/under-subscription, not correctness.
    """
    if _ACTIVE_COUNTER is None or _TOTAL_CORES is None:
        return 1
    active = max(_ACTIVE_COUNTER.value, 1)
    return max(1, _TOTAL_CORES // active)


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
        super_probs = global_probs[:, indices, :]
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

_MEMORY_DEBUG = False  # Set to True to enable per-step memory logging

def _get_rss_mb():
    """Get current process RSS in MB."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except:
        pass
    return 0

def _process_single_batch(args):
    """
    Worker function to process a single batch.
    
    Attaches to POSIX shared memory segments for global_probs and
    global_sites using metadata from _SHARED_META (set by pool initializer).
    
    Dynamically adjusts parallelism between major phases based on how
    many peer workers are still active. Two modes:
    
      Mesh generation (sequential with dynamic numba):
        Gaps processed one at a time. Between each gap, numba threads
        are updated to total_cores // active_workers. prange over samples
        provides equivalent throughput to pool-based processing but can
        scale up mid-computation as peers finish.
      
      Numba-only phases (beam search, chimera resolution, reconstruction):
        No inner pool, dyn_threads numba threads in this process.
        Total cores used = 1 × dyn_threads = dyn_threads. ✓
    
    Explicitly deletes large intermediates and calls malloc_trim between
    steps to release freed pages back to the OS, reducing peak memory.
    
    Returns dict with 'batch_idx', 'super_block', and 'status'.
    """
    import ctypes
    import numba
    _libc = ctypes.CDLL("libc.so.6")
    
    # DIAGNOSTIC: capture input for replay BEFORE any numba code runs
    _debug_save_args(args)
    _debug_mark_phase(args[0], "start")
    
    (b_idx, start_i, end_i, original_blocks_list,
     use_hmm_linking, recomb_rate, beam_width, max_founders,
     max_sites_for_linking, n_generations, recomb_tolerance,
     top_n_swap, max_cr_iterations, paint_penalty, min_hotspot_samples,
     cc_scale, inner_num_processes, verbose) = args
    
    # Memory debug logging
    _log = []
    if _MEMORY_DEBUG and b_idx == 0:
        _log.append(f"b{b_idx}_00_start: {_get_rss_mb():.0f} MB")
    
    # Attach to shared memory segments (zero-copy)
    _debug_mark_phase(b_idx, "shm_attach_start")
    shm_probs, global_probs = _attach_shared_array(_SHARED_META['probs'])
    shm_sites, global_sites = _attach_shared_array(_SHARED_META['sites'])
    _debug_mark_phase(b_idx, "shm_attach_done")
    
    try:
        # Register this worker as active
        if _ACTIVE_COUNTER is not None:
            with _ACTIVE_COUNTER.get_lock():
                _ACTIVE_COUNTER.value += 1
        
        # Initial dynamic allocation
        dyn_threads = _get_dynamic_threads()
        # Start with 1 numba thread — first phase uses inner pools
        numba.set_num_threads(1)
        
        if _MEMORY_DEBUG and b_idx == 0:
            _log.append(f"b{b_idx}_01_attached: {_get_rss_mb():.0f} MB (dyn_threads={dyn_threads})")
        
        original_portion = block_haplotypes.BlockResults(original_blocks_list)
        
        if len(original_portion) < 2:
            _debug_mark_phase(b_idx, "passthrough_single_block")
            _debug_mark_done(b_idx)
            return {
                'batch_idx': b_idx,
                'super_block': original_portion[0],
                'status': 'passthrough'
            }

        # 1. Create Proxies
        _debug_mark_phase(b_idx, "create_proxies_start",
                         f"n_blocks={len(original_portion)} max_sites={max_sites_for_linking}")
        proxy_list = []
        for b in original_portion:
            proxy_list.append(create_downsampled_proxy(b, max_sites_for_linking))
        portion_proxy = block_haplotypes.BlockResults(proxy_list)
        _debug_mark_phase(b_idx, "create_proxies_done",
                         f"hap_counts={[len(b.haplotypes) for b in portion_proxy]}")
        
        # 2. Slice to batch-relevant sites only
        _debug_mark_phase(b_idx, "slice_sites_start")
        all_positions = np.concatenate([b.positions for b in original_portion])
        batch_indices = np.searchsorted(global_sites, all_positions)
        idx_min, idx_max = batch_indices.min(), batch_indices.max()
        batch_probs = np.ascontiguousarray(global_probs[:, idx_min:idx_max+1, :])
        batch_sites = np.ascontiguousarray(global_sites[idx_min:idx_max+1])
        _debug_mark_phase(b_idx, "slice_sites_done",
                         f"n_sites={len(batch_sites)} batch_probs_shape={batch_probs.shape}")

        if _MEMORY_DEBUG and b_idx == 0:
            _log.append(f"b{b_idx}_02_sliced: {_get_rss_mb():.0f} MB")

        # 3. Compute Max Gap
        _debug_mark_phase(b_idx, "compute_max_gap_start")
        if n_generations is not None and recomb_tolerance is not None:
            beam_max_gap = compute_max_gap(original_blocks_list, recomb_rate, 
                                            n_generations, recomb_tolerance)
        else:
            beam_max_gap = None
        _debug_mark_phase(b_idx, "compute_max_gap_done", f"beam_max_gap={beam_max_gap}")

        # =================================================================
        # 4. Generate Mesh — DYNAMIC SEQUENTIAL phase
        #    Emissions: ThreadPoolExecutor (pure numpy, fast, no numba).
        #    Mesh EM: Sequential over gaps with dynamic numba threads.
        #    Between each gap, _get_dynamic_threads is called to re-check
        #    the active worker count and scale numba threads up as peers
        #    finish. prange over 320 samples gives equivalent throughput
        #    to pool-based processing, but can adapt mid-computation.
        # =================================================================
        dyn_threads = _get_dynamic_threads()
        pool_budget = max(inner_num_processes, dyn_threads)
        
        if use_hmm_linking:
            # Emissions: ThreadPoolExecutor (threads, not processes — no numba,
            # no oversubscription risk, cheap to create/destroy)
            _debug_mark_phase(b_idx, "viterbi_emissions_start", f"pool_budget={pool_budget}")
            viterbi_emissions = hmm_matching.generate_viterbi_block_emissions(
                batch_probs, batch_sites, portion_proxy, num_processes=pool_budget
            )
            _debug_mark_phase(b_idx, "viterbi_emissions_done")
            # Mesh EM: sequential with dynamic numba scaling between gaps
            _debug_mark_phase(b_idx, "transition_mesh_hmm_start")
            mesh = hmm_matching.generate_transition_probability_mesh_double_hmm(
                None, None, portion_proxy, 
                recomb_rate=recomb_rate, 
                use_standard_baum_welch=False,
                precalculated_viterbi_emissions=viterbi_emissions,
                num_processes=1,
                dynamic_cores_fn=_get_dynamic_threads
            )
            _debug_mark_phase(b_idx, "transition_mesh_hmm_done")
            del viterbi_emissions
        else:
            _debug_mark_phase(b_idx, "transition_mesh_nohmm_start", f"pool_budget={pool_budget}")
            mesh = block_linking.generate_transition_probability_mesh(
                batch_probs, batch_sites, portion_proxy,
                use_standard_baum_welch=True,
                num_processes=pool_budget
            )
            _debug_mark_phase(b_idx, "transition_mesh_nohmm_done")
        
        if _MEMORY_DEBUG and b_idx == 0:
            _log.append(f"b{b_idx}_03_mesh: {_get_rss_mb():.0f} MB")

        # =================================================================
        # 5. Beam Search — NUMBA-ONLY phase
        #    No inner pool. Give all dynamic threads to numba.
        # =================================================================
        dyn_threads = _apply_dynamic_threads()
            
        _debug_mark_phase(b_idx, "beam_search_start",
                         f"beam_width={beam_width} max_gap={beam_max_gap}")
        beam_results = beam_search_core.run_full_mesh_beam_search(
            portion_proxy, mesh, beam_width=beam_width, 
            max_gap=beam_max_gap, verbose=verbose
        )
        _debug_mark_phase(b_idx, "beam_search_done",
                         f"n_results={len(beam_results) if beam_results else 0}")
        
        if _MEMORY_DEBUG and b_idx == 0:
            _log.append(f"b{b_idx}_04_beam: {_get_rss_mb():.0f} MB")
        
        if not beam_results:
            _debug_mark_phase(b_idx, "early_return_no_beam")
            _debug_mark_done(b_idx)
            return {
                'batch_idx': b_idx,
                'super_block': None,
                'status': 'beam_search_failed'
            }
        
        _debug_mark_phase(b_idx, "fast_mesh_build_start")
        fast_mesh = beam_search_core.FastMesh(portion_proxy, mesh)
        _debug_mark_phase(b_idx, "fast_mesh_build_done")
        
        # Free mesh — fast_mesh has what it needs
        del mesh
        _libc.malloc_trim(0)
        
        if _MEMORY_DEBUG and b_idx == 0:
            _log.append(f"b{b_idx}_04b_mesh_freed: {_get_rss_mb():.0f} MB")
        
        # =================================================================
        # 6. Selection + Swap + CR — NUMBA-ONLY phase
        # =================================================================
        dyn_threads = _apply_dynamic_threads()
        
        _debug_mark_phase(b_idx, "chimera_resolution_start",
                         f"max_founders={max_founders} top_n_swap={top_n_swap}")
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
        _debug_mark_phase(b_idx, "chimera_resolution_done")
        
        # Free beam_results and batch data — chimera resolution is done with them
        del beam_results
        _libc.malloc_trim(0)
        
        if _MEMORY_DEBUG and b_idx == 0:
            _log.append(f"b{b_idx}_05_chimera: {_get_rss_mb():.0f} MB")
        
        # =================================================================
        # 7. Reconstruction — NUMBA-ONLY phase
        # =================================================================
        dyn_threads = _apply_dynamic_threads()
        
        _debug_mark_phase(b_idx, "reconstruction_start")
        reconstructed_data = beam_search_core.reconstruct_haplotypes_from_beam(
            resolved_beam, fast_mesh, original_portion
        )
        _debug_mark_phase(b_idx, "reconstruction_done")
        
        # Free resolved_beam and fast_mesh — reconstruction is done
        del resolved_beam, fast_mesh
        _libc.malloc_trim(0)
        
        # 8. Package
        _debug_mark_phase(b_idx, "superblock_convert_start")
        super_block = convert_reconstruction_to_superblock(
            reconstructed_data, original_portion, batch_probs, batch_sites
        )
        _debug_mark_phase(b_idx, "superblock_convert_done")
        
        # Free intermediates — super_block has everything needed
        del reconstructed_data, batch_probs, batch_sites, portion_proxy, proxy_list
        _libc.malloc_trim(0)
        
        # 9. Structural Chimera Pruning
        _debug_mark_phase(b_idx, "prune_chimeras_start")
        super_block = beam_search_core.prune_superblock_chimeras(super_block)
        _debug_mark_phase(b_idx, "prune_chimeras_done")
        
        if _MEMORY_DEBUG and b_idx == 0:
            _log.append(f"b{b_idx}_06_done: {_get_rss_mb():.0f} MB")
            # Write to temp file since we can't print from forkserver workers
            import os, tempfile
            log_path = os.path.join(tempfile.gettempdir(), f"ha_mem_b{b_idx}_{os.getpid()}.log")
            with open(log_path, 'w') as f:
                f.write('\n'.join(_log) + '\n')
            print(f"  [Memory debug] Worker {b_idx} log: {log_path}")
        
        _debug_mark_done(b_idx)
        return {
            'batch_idx': b_idx,
            'super_block': super_block,
            'status': 'success' if super_block else 'reconstruction_failed'
        }
    finally:
        # Unregister this worker
        if _ACTIVE_COUNTER is not None:
            with _ACTIVE_COUNTER.get_lock():
                _ACTIVE_COUNTER.value -= 1
        # Detach from shared memory (do NOT unlink — parent handles that)
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
                          verbose=False): 
    """
    Performs one level of Hierarchical Assembly.
    
    Memory strategy:
      - global_probs/global_sites placed in POSIX shared memory (/dev/shm)
      - Workers spawned via forkserver — start from lightweight process
        (~50 MB), NOT from the parent's large heap. Zero COW overhead.
      - Workers receive SharedMemory metadata via pool initializer (tiny)
      - Workers attach to shared segments by name for zero-copy access
      - batch_probs slicing ensures inner pools only pickle small arrays
      - Workers are non-daemonic, allowing inner child pools at L2+
      - maxtasksperchild recycles workers after N batches, releasing
        accumulated memory (Python doesn't return freed pages to OS)
      - global_probs downcast to float32 (halves shared memory + all tensors)
      - Worker count auto-capped based on available RAM / min_gb_per_worker
    
    Dynamic thread reallocation:
      - A shared counter tracks active workers
      - Between major phases, each worker recalculates its allocation
        as total_cores // active_workers
      - Inner-pool phases: dyn_threads pool processes × 1 numba thread
      - Numba-only phases: 1 process × dyn_threads numba threads
      - When peers finish, remaining workers scale up to use freed cores
    
    Args:
      num_processes: Maximum total cores to use. Acts as a strict ceiling for both
          concurrent workers AND total thread allocation. The function may use fewer
          workers if available RAM is insufficient, but thread allocation across
          workers will sum to at most num_processes. Typically set to os.cpu_count()
          or the number of cores allocated by a job scheduler.
      maxtasksperchild: Recycle workers after this many batches (default None).
          Set to 1 to prevent memory accumulation from glibc malloc fragmentation.
      min_gb_per_worker: Minimum GB of RAM to budget per concurrent worker.
          Used to auto-cap worker count: max_workers = available_ram / min_gb_per_worker.
          Increase if blocks have many haplotypes (>15); decrease if RAM is tight
          but blocks are small. Default 4.0 (appropriate for float32 with typical data).
    
    IMPORTANT: The entry script must NOT be named main.py, otherwise
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
    
    # =====================================================================
    # Strip redundant probs_array from input blocks
    # =====================================================================
    # Each BlockResult carries probs_array — shape (n_samples, ~200, 3)
    # per block. Across a full chromosome, this totals the same size as
    # global_probs (~5 GB for 290 samples × 1.5M sites). Workers access
    # sample data via global_probs in shared memory, so probs_array in
    # blocks is redundant. Stripping it reduces:
    #   - Parent process memory (fewer live arrays)
    #   - Pickle size when sending blocks to workers as task arguments
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
    
    # =====================================================================
    # Downcast to float32 for memory efficiency
    # =====================================================================
    # global_probs is float64 from R01 (HDBSCAN needs float64) but
    # assembly only uses it for emission scoring where float32 precision
    # is more than sufficient. Casting halves:
    #   - Shared memory (10 GB → 5 GB for chr3)
    #   - Per-worker batch_probs slices
    #   - All downstream emission/chimera tensors (they inherit dtype)
    # Numba auto-specializes @njit functions for float32.
    if global_probs.dtype == np.float64:
        global_probs = global_probs.astype(np.float32)
        print(f"  Downcast global_probs to float32 ({global_probs.nbytes / (1024**3):.1f} GB)")
    
    # Also downcast block haplotypes (soft probabilities) if float64
    for block in input_blocks:
        for k, h in block.haplotypes.items():
            if h.dtype == np.float64:
                block.haplotypes[k] = h.astype(np.float32)
    
    # =====================================================================
    # Create POSIX shared memory for large arrays
    # =====================================================================
    t0 = time.time()
    shm_probs, probs_meta = _create_shared_array(global_probs, 'global_probs')
    shm_sites, sites_meta = _create_shared_array(global_sites, 'global_sites')
    
    shared_meta = {
        'probs': probs_meta,
        'sites': sites_meta,
    }
    
    probs_gb = global_probs.nbytes / (1024**3)
    print(f"  Shared memory created: {probs_gb:.1f} GB probs + sites ({time.time()-t0:.1f}s)")
    
    # DIAGNOSTIC: save contig-level context for offline replay (opt-in via
    # HIERARCHICAL_DEBUG_DIR env var).  Also clears any previous run's batch
    # files from the debug dir so the next hang is unambiguous.
    _debug_save_context({
        'global_probs': global_probs,  # final form (downcast to float32 above)
        'global_sites': global_sites,
        # All params that could affect _process_single_batch behaviour:
        'batch_size': batch_size,
        'use_hmm_linking': use_hmm_linking,
        'recomb_rate': recomb_rate,
        'beam_width': beam_width,
        'max_founders': max_founders,
        'max_sites_for_linking': max_sites_for_linking,
        'n_generations': n_generations,
        'recomb_tolerance': recomb_tolerance,
        'top_n_swap': top_n_swap,
        'max_cr_iterations': max_cr_iterations,
        'paint_penalty': paint_penalty,
        'min_hotspot_samples': min_hotspot_samples,
        'cc_scale': cc_scale,
        'num_processes': num_processes,
    })
    
    # =====================================================================
    # Auto-size worker count based on available RAM
    # =====================================================================
    # Read MemAvailable AFTER shared memory creation — it already accounts
    # for the parent process, loaded data, and the shared memory segment.
    # This gives the most accurate picture of what's left for workers.
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
    
    # =====================================================================
    # Create shared active-worker counter for dynamic thread reallocation
    # =====================================================================
    active_counter = _forkserver_ctx.Value('i', 0)
    
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
                initargs=(shared_meta, total_cores, active_counter),
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
            # Sequential execution — set metadata for current process
            global _SHARED_META, _ACTIVE_COUNTER, _TOTAL_CORES
            _SHARED_META = shared_meta
            _ACTIVE_COUNTER = None  # No counter needed for single process
            _TOTAL_CORES = total_cores
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
    
    return block_haplotypes.BlockResults(output_super_blocks)