import thread_config
from thread_config import numba_thread_scope

import numpy as np
import math
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


def _init_worker_meta(meta_dict, numba_threads=1):
    """
    Pool initializer — called once per worker at creation time.
    Stores SharedMemory metadata so workers can attach to the segments.
    
    Also overrides numba's thread pool size BEFORE it gets created.
    Numba reads NUMBA_NUM_THREADS at import time (in the forkserver),
    but the thread pool is created lazily on first parallel function call.
    Overriding numba.config.NUMBA_NUM_THREADS here ensures the pool
    is created at the correct size for this worker, not 112.
    """
    import os
    os.environ['NUMBA_NUM_THREADS'] = str(numba_threads)
    try:
        import numba
        # Override the config that numba read at import time
        numba.config.NUMBA_NUM_THREADS = numba_threads
        # If thread pool already exists (shouldn't in forkserver workers),
        # limit active threads
        numba.set_num_threads(min(numba_threads, numba.get_num_threads()))
    except Exception:
        pass
    
    global _SHARED_META
    _SHARED_META = meta_dict


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
    
    Forkserver workers start clean (~50 MB), import modules, and only
    allocate memory for actual work (~150 MB per batch).
    
    Returns dict with 'batch_idx', 'super_block', and 'status'.
    """
    (b_idx, start_i, end_i, original_blocks_list,
     use_hmm_linking, recomb_rate, beam_width, max_founders,
     max_sites_for_linking, n_generations, recomb_tolerance,
     top_n_swap, max_cr_iterations, paint_penalty, min_hotspot_samples,
     cc_scale, inner_num_processes, verbose) = args
    
    inner_num_threads = max(inner_num_processes, 8)
    
    # Memory debug logging
    _log = []
    if _MEMORY_DEBUG and b_idx == 0:
        _log.append(f"b{b_idx}_00_start: {_get_rss_mb():.0f} MB")
    
    # Attach to shared memory segments (zero-copy)
    shm_probs, global_probs = _attach_shared_array(_SHARED_META['probs'])
    shm_sites, global_sites = _attach_shared_array(_SHARED_META['sites'])
    
    try:
        with numba_thread_scope(inner_num_processes):
            
            if _MEMORY_DEBUG and b_idx == 0:
                _log.append(f"b{b_idx}_01_attached: {_get_rss_mb():.0f} MB")
            
            original_portion = block_haplotypes.BlockResults(original_blocks_list)
            
            if len(original_portion) < 2:
                return {
                    'batch_idx': b_idx,
                    'super_block': original_portion[0],
                    'status': 'passthrough'
                }

            # 1. Create Proxies
            proxy_list = []
            for b in original_portion:
                proxy_list.append(create_downsampled_proxy(b, max_sites_for_linking))
            portion_proxy = block_haplotypes.BlockResults(proxy_list)
            
            # 2. Slice to batch-relevant sites only
            all_positions = np.concatenate([b.positions for b in original_portion])
            batch_indices = np.searchsorted(global_sites, all_positions)
            idx_min, idx_max = batch_indices.min(), batch_indices.max()
            batch_probs = np.ascontiguousarray(global_probs[:, idx_min:idx_max+1, :])
            batch_sites = np.ascontiguousarray(global_sites[idx_min:idx_max+1])

            if _MEMORY_DEBUG and b_idx == 0:
                _log.append(f"b{b_idx}_02_sliced: {_get_rss_mb():.0f} MB")

            # 3. Compute Max Gap
            if n_generations is not None and recomb_tolerance is not None:
                beam_max_gap = compute_max_gap(original_blocks_list, recomb_rate, 
                                                n_generations, recomb_tolerance)
            else:
                beam_max_gap = None

            # 4. Generate Mesh — uses batch-sliced data
            if use_hmm_linking:
                viterbi_emissions = hmm_matching.generate_viterbi_block_emissions(
                    batch_probs, batch_sites, portion_proxy, num_processes=inner_num_processes
                )
                mesh = hmm_matching.generate_transition_probability_mesh_double_hmm(
                    None, None, portion_proxy, 
                    recomb_rate=recomb_rate, 
                    use_standard_baum_welch=False,
                    precalculated_viterbi_emissions=viterbi_emissions,
                    num_processes=inner_num_processes
                )
            else:
                mesh = block_linking.generate_transition_probability_mesh(
                    batch_probs, batch_sites, portion_proxy,
                    use_standard_baum_welch=True,
                    num_processes=inner_num_processes
                )
            
            if _MEMORY_DEBUG and b_idx == 0:
                _log.append(f"b{b_idx}_03_mesh: {_get_rss_mb():.0f} MB")
                
            # 5. Beam Search on Proxy
            beam_results = beam_search_core.run_full_mesh_beam_search(
                portion_proxy, mesh, beam_width=beam_width, 
                max_gap=beam_max_gap, verbose=verbose
            )
            
            if _MEMORY_DEBUG and b_idx == 0:
                _log.append(f"b{b_idx}_04_beam: {_get_rss_mb():.0f} MB")
            
            if not beam_results:
                return {
                    'batch_idx': b_idx,
                    'super_block': None,
                    'status': 'beam_search_failed'
                }
                
            fast_mesh = beam_search_core.FastMesh(portion_proxy, mesh)
            
            # 6. Selection + Swap + CR
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
                num_threads=inner_num_threads,
            )
            
            if _MEMORY_DEBUG and b_idx == 0:
                _log.append(f"b{b_idx}_05_chimera: {_get_rss_mb():.0f} MB")
            
            # 7. Reconstruction on Originals
            reconstructed_data = beam_search_core.reconstruct_haplotypes_from_beam(
                resolved_beam, fast_mesh, original_portion
            )
            
            # 8. Package
            super_block = convert_reconstruction_to_superblock(
                reconstructed_data, original_portion, batch_probs, batch_sites
            )
            
            # 9. Structural Chimera Pruning
            super_block = beam_search_core.prune_superblock_chimeras(super_block)
            
            if _MEMORY_DEBUG and b_idx == 0:
                _log.append(f"b{b_idx}_06_done: {_get_rss_mb():.0f} MB")
                # Write to temp file since we can't print from forkserver workers
                import os, tempfile
                log_path = os.path.join(tempfile.gettempdir(), f"ha_mem_b{b_idx}_{os.getpid()}.log")
                with open(log_path, 'w') as f:
                    f.write('\n'.join(_log) + '\n')
                print(f"  [Memory debug] Worker {b_idx} log: {log_path}")
            
            return {
                'batch_idx': b_idx,
                'super_block': super_block,
                'status': 'success' if super_block else 'reconstruction_failed'
            }
    finally:
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
                          cc_scale=0.2,
                          # Parallelization
                          num_processes=16,
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
    
    IMPORTANT: The entry script must NOT be named main.py, otherwise
    forkserver workers will re-execute it.
    """
    total_blocks = len(input_blocks)
    num_batches = math.ceil(total_blocks / batch_size)
    
    outer_workers = min(num_batches, num_processes)
    inner_num_processes = max(1, num_processes // outer_workers)
    
    # Preview
    if n_generations is not None:
        preview_max_gap = compute_max_gap(list(input_blocks), recomb_rate, 
                                           n_generations, recomb_tolerance)
        print(f"\n--- Starting Hierarchical Step ---")
        print(f"Input: {total_blocks} blocks -> Target: ~{num_batches} Super-Blocks")
        print(f"Max gap: {preview_max_gap} (n_gen={n_generations}, tol={recomb_tolerance}, rate={recomb_rate})")
        print(f"Parallelism: {outer_workers} outer workers x {inner_num_processes} inner cores = {outer_workers * inner_num_processes} total")
    else:
        print(f"\n--- Starting Hierarchical Step ---")
        print(f"Input: {total_blocks} blocks -> Target: ~{num_batches} Super-Blocks")
        print(f"Max gap: unlimited (n_generations not specified)")
        print(f"Parallelism: {outer_workers} outer workers x {inner_num_processes} inner cores = {outer_workers * inner_num_processes} total")
    
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
                initargs=(shared_meta, inner_num_processes)
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
            global _SHARED_META
            _SHARED_META = shared_meta
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