"""Shared-memory hierarchical assembly with dynamic process/thread budgeting."""
from __future__ import annotations


from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import pickle
import numpy as np
import numba
from numba.typed import List
import math
import gc
import time
import copy
from tqdm import tqdm
import os


_HIER_PROFILE = os.environ.get('BHD_HIER_PROFILE', '') not in ('', '0', 'false', 'False')


_HIER_PROFILE_MIN_L = int(os.environ.get('BHD_HIER_PROFILE_MIN_L', '500000'))


_SHARED_META = {}


def _init_worker_meta(meta_dict, total_cores, active_counter, extra_counter,
                      startup_counters=None):
    """Pool initializer — called once per worker at creation time.

    Stores SharedMemory metadata so workers can attach to the global
    arrays, configures the numba thread pool ceiling to total_cores
    so set_num_threads can scale freely later (starts at 1 — the real
    value is set per phase in _process_single_batch), and wires the
    active/extra counters used by core.parallel.get_dynamic_threads.

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
        numba.config.NUMBA_NUM_THREADS = total_cores
        numba.set_num_threads(1)
    except Exception:
        pass

    global _SHARED_META
    _SHARED_META = meta_dict
    # Wire this worker to the shared dynamic-thread counters; the state and
    # helpers live in core.parallel.  set_dynamic_thread_state resets the
    # per-worker extra-claim flag, so a recycled worker can't inherit a stale
    # claim.
    core_parallel.set_dynamic_thread_state(
        total_cores, active_counter, extra_counter,
        **({} if startup_counters is None else startup_counters))


def _create_shared_array(array, label, copy_threads=1):
    """
    Copy a numpy array into a POSIX shared memory segment.

    Returns:
        (SharedMemory handle, metadata_dict)
    """
    return core_parallel.create_shared_array(
        array, name_key='name', dtype_as_string=False, copy_threads=copy_threads
    )


def _attach_shared_array(metadata):
    """
    Attach to an existing shared memory segment and return a numpy view.

    Returns:
        (SharedMemory handle, numpy array view)
    """
    return core_parallel.attach_shared_array(metadata)


def _missing_aware_block_indices(block, global_sites, max_sites=None):
    """Marker indices with any observed founder allele in the frozen panel."""

    panel = assembly_observations.founder_inference_panel_from_block_result(block)
    keep_flags = getattr(block, 'keep_flags', None)
    if keep_flags is None:
        retained = np.ones(len(block.positions), dtype=np.bool_)
    else:
        retained = np.asarray(keep_flags, dtype=np.bool_)
        if retained.shape != (len(block.positions),):
            raise ValueError("block keep_flags are not aligned with positions")
    globally_complete = retained & np.any(panel.called, axis=0)
    if max_sites is not None:
        sampled = _linking_proxy_indices(block, max_sites, panel=panel)
        proxy_mask = np.zeros(len(block.positions), dtype=np.bool_)
        proxy_mask[sampled] = True
        globally_complete &= proxy_mask
    if not np.any(globally_complete):
        return np.empty(0, dtype=np.int64)
    indices = np.searchsorted(global_sites, block.positions)
    if (np.any(indices >= len(global_sites))
            or not np.array_equal(global_sites[indices], block.positions)):
        raise ValueError("block positions are absent from global sites")
    return np.ascontiguousarray(indices[globally_complete], dtype=np.int64)


@numba.njit(cache=True, parallel=True, fastmath=False)
def _informative_samples_at_indices(probabilities, index_sets):
    """Scan independent block/sample rows without gathering a GL tensor.

    One task per block: its site indices are read from the typed List once (a
    List element read inside a parallel loop updates a reference count shared
    by all threads, and a sample's scan usually stops at its first site).
    """
    samples = probabilities.shape[0]
    result = np.zeros((len(index_sets), samples), np.bool_)
    for block_index in numba.prange(len(index_sets)):
        block = np.int64(block_index)
        indices = index_sets[block]
        for sample in range(samples):
            for site in indices:
                a, b, c = probabilities[sample, site]
                if (a + b) + c > 0.0 and (a != b or b != c):
                    result[block, sample] = True
                    break
    return result


def _missing_aware_batch_ranges(
        input_blocks, global_probs, global_sites, batch_size,
        min_boundary_informative_samples=1, max_sites=None):
    """Partition without crossing persistent or data-unidentifiable breaks."""

    if (
            isinstance(min_boundary_informative_samples, bool)
            or int(min_boundary_informative_samples)
            != min_boundary_informative_samples
            or min_boundary_informative_samples < 1):
        raise ValueError(
            "min_boundary_informative_samples must be a positive integer"
        )
    blocks = list(input_blocks)
    index_sets = List.empty_list(numba.types.int64[::1])
    for block in blocks:
        index_sets.append(_missing_aware_block_indices(
            block, global_sites, max_sites=max_sites))
    masks = list(_informative_samples_at_indices(
        np.asarray(global_probs), index_sets))
    boundary_joint_counts = []
    for boundary in range(max(0, len(blocks) - 1)):
        left = blocks[boundary]
        right = blocks[boundary + 1]
        joint_count = int(np.sum(masks[boundary] & masks[boundary + 1]))
        boundary_joint_counts.append(joint_count)
        existing_break = bool(
            getattr(left, 'missing_aware_break_after', False)
            or getattr(right, 'missing_aware_break_before', False)
        )
        if existing_break or joint_count < min_boundary_informative_samples:
            left.missing_aware_break_after = True
            right.missing_aware_break_before = True
            left.missing_aware_joint_informative_samples_after = joint_count
            right.missing_aware_joint_informative_samples_before = joint_count
            if not existing_break:
                reason = 'insufficient_joint_informative_samples'
                left.missing_aware_break_reason_after = reason
                right.missing_aware_break_reason_before = reason

    ranges = []
    start = 0
    while start < len(blocks):
        stop = min(start + batch_size, len(blocks))
        for candidate in range(start + 1, stop):
            if (
                    getattr(blocks[candidate - 1],
                            'missing_aware_break_after', False)
                    or getattr(blocks[candidate],
                               'missing_aware_break_before', False)):
                stop = candidate
                break
        ranges.append((start, stop))
        start = stop
    return ranges, masks, tuple(boundary_joint_counts)


def _missing_aware_batch_informative_sample_mask(block_masks):
    """Return samples informative in every block of one actual batch.

    Inference spans the complete batch, so union would retain samples whose
    state is arbitrary at one or more constituent blocks.
    """

    masks = tuple(np.asarray(mask, dtype=np.bool_) for mask in block_masks)
    if not masks:
        raise ValueError("a missing-aware batch must contain at least one block")
    shape = masks[0].shape
    if len(shape) != 1 or any(mask.shape != shape for mask in masks):
        raise ValueError("batch informative masks must be aligned vectors")
    return np.logical_and.reduce(masks)


# Block attributes no join reads: path selection's records (discovery/path_blocks.py) and the
# panel-search records of earlier levels. They differ between a contig's assemblies without
# changing any join, so the batch input digest leaves them out.
_UNREAD_BLOCK_ATTRIBUTES = frozenset((
    'path_fit', 'path_unknown_copy_fraction', 'path_wildcard_interpretation',
    'panel_search_diagnostics',
))
_DIGEST_SAMPLE_CHUNK = 32


def _update_value_digest(digest, value):
    """Feed one attribute value to a digest in a type-tagged, length-prefixed encoding."""
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        digest.update(f'A{value.dtype.str}{value.shape}'.encode())
        digest.update(np.ascontiguousarray(value))
    elif isinstance(value, dict):
        digest.update(f'D{len(value)}'.encode())
        for key in sorted(value, key=repr):
            _update_value_digest(digest, key)
            _update_value_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f'{type(value).__name__}{len(value)}'.encode())
        for item in value:
            _update_value_digest(digest, item)
    elif value is None or isinstance(value, (bool, int, float, complex, str, bytes, np.generic)):
        text = repr(value).encode()
        digest.update(f'S{type(value).__name__}{len(text)}:'.encode())
        digest.update(text)
    else:
        text = pickle.dumps(value, protocol=5)
        digest.update(f'P{len(text)}:'.encode())
        digest.update(text)


def _evidence_digest(blocks, global_probs, global_sites):
    """SHA-256 of a batch's evidence slice as _process_single_batch takes it: the scoring evidence
    between its first and last site, and their positions. Contiguous copies of a few samples at
    a time keep the copy and the hash outside the GIL, so batches are digested concurrently."""
    indices = np.searchsorted(global_sites, np.concatenate([b.positions for b in blocks]))
    first, last = int(indices.min()), int(indices.max())
    digest = hashlib.sha256()
    _update_value_digest(digest, global_sites[first:last + 1])
    digest.update(f'E{global_probs.dtype.str}{global_probs.shape[0]}x{last + 1 - first}'.encode())
    for start in range(0, global_probs.shape[0], _DIGEST_SAMPLE_CHUNK):
        digest.update(np.ascontiguousarray(
            global_probs[start:start + _DIGEST_SAMPLE_CHUNK, first:last + 1]))
    return digest.digest()


def _batch_input_digest(context, blocks, informative_mask, evidence_digest):
    """SHA-256 of everything one batch's join reads: context (the join settings and the assembly
    code identity, canonical JSON bytes), the batch's informative samples, its evidence slice
    (_evidence_digest) and every block attribute except _UNREAD_BLOCK_ATTRIBUTES. Joins are
    deterministic, so equal digests give equal super-blocks."""
    digest = hashlib.sha256(context)
    _update_value_digest(digest, np.asarray(informative_mask, dtype=np.bool_))
    digest.update(evidence_digest)
    for block in blocks:
        names = sorted(set(vars(block)) - _UNREAD_BLOCK_ATTRIBUTES)
        digest.update(f'B{len(names)}'.encode())
        for name in names:
            digest.update(f'{len(name)}:{name}'.encode())
            _update_value_digest(digest, getattr(block, name))
    return digest.hexdigest()


def _linking_proxy_indices(block, max_sites, panel=None):
    """Thin usable panel markers, not positions that later become empty.

    Fully called/kept panels retain the original stride selection. Small blocks
    stay untouched; emissions marginalize missing alleles at partial sites.
    Wholly unsupported blocks stay intact for unresolved passthrough.
    """
    total = len(block.positions)
    if total <= max_sites:
        return np.arange(total, dtype=np.int64)
    if panel is None:
        panel = assembly_observations.founder_inference_panel_from_block_result(block)
    usable = np.any(panel.called, axis=0)
    if block.keep_flags is not None:
        usable &= np.asarray(block.keep_flags, dtype=np.bool_)
    candidates = np.flatnonzero(usable)
    if not len(candidates):
        return np.arange(total, dtype=np.int64)
    stride = max(1, math.ceil(len(candidates) / max_sites))
    return candidates[::stride]


def create_downsampled_proxy(block, max_sites=2000):
    """
    Creates a lightweight 'Proxy' of a BlockResult.
    Returns the original block if it's small enough.
    """
    total_sites = len(block.positions)

    if total_sites <= max_sites:
        return block

    sampled_indices = _linking_proxy_indices(block, max_sites)
    if len(sampled_indices) == total_sites:
        return block

    new_pos = np.ascontiguousarray(block.positions[sampled_indices])

    new_haps = {}
    for k, v in block.haplotypes.items():
        if v.ndim > 1:
            new_haps[k] = np.ascontiguousarray(v[sampled_indices,:])
        else:
            new_haps[k] = np.ascontiguousarray(v[sampled_indices])

    if block.keep_flags is not None:
        new_flags = np.ascontiguousarray(block.keep_flags[sampled_indices])
    else:
        new_flags = None

    new_reads = None
    if block.reads_count_matrix is not None:
        new_reads = np.ascontiguousarray(block.reads_count_matrix[:, sampled_indices,:])

    new_probs = None
    if block.probs_array is not None:
        new_probs = np.ascontiguousarray(block.probs_array[:, sampled_indices,:])

    proxy = core_haplotypes.BlockResult(
        positions=new_pos,
        haplotypes=new_haps,
        keep_flags=new_flags,
        reads_count_matrix=new_reads,
        probs_array=new_probs,
        genotype_evidence_mode=getattr(
            block, 'genotype_evidence_mode', None
        ),
    )

    source_paths, source_offsets, source_counts, source_row_counts = (
        assembly_paths.missing_aware_atomic_source_provenance(block)
    )
    sampled_source_counts = (
        np.searchsorted(
            sampled_indices, source_offsets + source_counts, side='left'
        )
        - np.searchsorted(sampled_indices, source_offsets, side='left')
    ).astype(np.int64, copy=False)
    sampled_source_offsets = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.cumsum(sampled_source_counts[:-1]),
    ))
    proxy.missing_aware_atomic_source_row_paths = source_paths.copy()
    proxy.missing_aware_atomic_position_offsets = sampled_source_offsets
    proxy.missing_aware_atomic_position_counts = sampled_source_counts
    proxy.missing_aware_atomic_source_row_counts = source_row_counts.copy()

    # These founder-by-site arrays remain aligned with the downsampled
    # haplotypes. In particular, discrete_haps is the authoritative call mask;
    # never reconstruct it from the downsampled probability values.
    for attribute in (
            'discrete_haps',
            'missing_aware_inference_discrete_haps',
            'founder_alt_pseudo_probability',
            'n_directional_site_supporters',
            'founder_allele_pseudo_confidence',
            'founder_log_pseudo_odds',
            'precleanup_candidate_discrete_haps'):
        values = getattr(block, attribute, None)
        if values is not None:
            values = np.asarray(values)
            if values.ndim != 2 or values.shape[1] != total_sites:
                raise ValueError(
                    f"block.{attribute} is not founder-by-site aligned"
                )
            setattr(
                proxy, attribute,
                np.ascontiguousarray(values[:, sampled_indices]).copy(),
            )

    # Sample-aligned cavity arrays are copied without site slicing.
    for attribute in (
            "sample_has_observed_kept_depth",
            "pair_assignments",
            "wildcard_slots"):
        values = getattr(block, attribute, None)
        if values is not None:
            setattr(proxy, attribute, np.asarray(values).copy())

    # Small mode/provenance fields are safe to retain on the linking proxy.
    for attribute in (
            'uncertainty_flag',
            'K_final',
            'precleanup_candidate_k',
            'wildcard_mass',
            'cavity_selected_mode_digest',
            'cavity_materialization_iterations',
            'cavity_selected_mode_iterations',
            'cavity_selected_mode_nll',
            'cavity_score_calibration',
            'cavity_weight_calibration',
            'cavity_materialization_uncertainty_reasons',
            'cavity_discovery_diagnostics',
            'missing_aware_unresolved_phase_component',
            'hierarchy_informative_sample_count',
            'hierarchy_total_sample_count',
            'hierarchy_informative_sample_rule',
            'missing_aware_break_before',
            'missing_aware_break_after',
            'missing_aware_break_reason_before',
            'missing_aware_break_reason_after',
            'missing_aware_joint_informative_samples_before',
            'missing_aware_joint_informative_samples_after'):
        if hasattr(block, attribute):
            setattr(proxy, attribute, copy.deepcopy(getattr(block, attribute)))

    return proxy


def convert_reconstruction_to_superblock(
        reconstructed_data, original_blocks, global_probs=None,
        global_sites=None):
    """
    Packages reconstruction results into a BlockResult (Super-Block).
    """
    if not reconstructed_data:
        return None

    super_haplotypes = {}
    for i, data in enumerate(reconstructed_data):
        super_haplotypes[i] = data['haplotype']

    super_positions = reconstructed_data[0]['positions']

    expected_positions = np.concatenate([
        np.asarray(block.positions) for block in original_blocks
    ])
    if not np.array_equal(super_positions, expected_positions):
        raise ValueError("reconstructed positions do not preserve input block order")

    atomic_position_count_parts = []
    atomic_source_row_count_parts = []
    for block in original_blocks:
        _, _, position_counts, source_row_counts = (
            assembly_paths.missing_aware_atomic_source_provenance(block)
        )
        atomic_position_count_parts.append(position_counts)
        atomic_source_row_count_parts.append(source_row_counts)
    atomic_position_counts = np.concatenate(atomic_position_count_parts)
    atomic_position_offsets = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.cumsum(atomic_position_counts[:-1]),
    ))
    atomic_source_row_counts = np.concatenate(atomic_source_row_count_parts)
    atomic_source_row_paths = np.stack([
        np.asarray(
            data['missing_aware_atomic_source_row_path'], dtype=np.int64
        )
        for data in reconstructed_data
    ])
    expected_path_shape = (
        len(reconstructed_data), len(atomic_position_counts)
    )
    if atomic_source_row_paths.shape != expected_path_shape:
        raise ValueError("reconstructed atomic source paths are not span-aligned")
    if (
            np.any(atomic_source_row_paths < 0)
            or np.any(
                atomic_source_row_paths
                >= atomic_source_row_counts[None,:]
            )):
        raise ValueError("reconstructed atomic source path is out of range")

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
        # worker.  See chimera_kernels._gather_samples_numba.
        super_probs = assembly_chimera_kernels._gather_samples_numba(global_probs, indices)
    else:
        probs_list = []
        for b in original_blocks:
            if b.probs_array is not None:
                probs_list.append(b.probs_array)
            elif b.reads_count_matrix is not None:
                _, probs = core_numerics.reads_to_probabilities(
                    b.reads_count_matrix,
                    use_hwe_prior=False,
                )
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

    super_block = core_haplotypes.BlockResult(
        positions=super_positions,
        haplotypes=super_haplotypes,
        keep_flags=super_flags,
        reads_count_matrix=super_reads,
        probs_array=super_probs
    )
    super_block.missing_aware_atomic_source_row_paths = (
        atomic_source_row_paths
    )
    super_block.missing_aware_atomic_position_offsets = (
        atomic_position_offsets
    )
    super_block.missing_aware_atomic_position_counts = atomic_position_counts
    super_block.missing_aware_atomic_source_row_counts = (
        atomic_source_row_counts
    )
    # Rows are in reconstructed founder order; each row was copied from the
    # exact local founder selected along that path. ``discrete_haps`` remains
    # the authoritative called mask and is never inferred from q == 0.5.
    super_block.discrete_haps = np.stack([
        data['discrete_haplotype'] for data in reconstructed_data
    ])
    super_block.missing_aware_inference_discrete_haps = np.stack([
        data['missing_aware_inference_discrete_haplotype']
        for data in reconstructed_data
    ])
    super_block.founder_alt_pseudo_probability = np.stack([
        data['founder_alt_pseudo_probability']
        for data in reconstructed_data
    ])
    if all(
            'n_directional_site_supporters' in data
            for data in reconstructed_data):
        super_block.n_directional_site_supporters = np.stack([
            data['n_directional_site_supporters']
            for data in reconstructed_data
        ])
    first_block = original_blocks[0]
    last_block = original_blocks[-1]
    for suffix, source in (('before', first_block), ('after', last_block)):
        marker = f'missing_aware_break_{suffix}'
        setattr(super_block, marker, bool(getattr(source, marker, False)))
        for stem in (
                'missing_aware_break_reason',
                'missing_aware_joint_informative_samples'):
            attribute = f'{stem}_{suffix}'
            if hasattr(source, attribute):
                setattr(super_block, attribute, getattr(source, attribute))

    return super_block


def compute_max_gap(blocks, recomb_rate, n_generations, recomb_tolerance,
                    chromosome_map=None):
    """
    Compute the maximum gap to use for beam search transition lookups.
    """
    block_spans = [b.positions[-1] - b.positions[0] for b in blocks if len(b.positions) > 1]
    if not block_spans:
        return 1

    avg_block_span = np.mean(block_spans)
    recombs_per_step = avg_block_span * recomb_rate * n_generations
    if chromosome_map is not None:
        if chromosome_map.has_map:
            spans = [float(chromosome_map.interval_morgans(
                b.positions[0], b.positions[-1])) for b in blocks if len(b.positions) > 1]
            recombs_per_step = np.mean(spans) * n_generations
        else:
            recombs_per_step = (avg_block_span * chromosome_map.fallback_rate_per_bp
                                * n_generations)

    if recombs_per_step <= 0:
        return len(blocks)

    max_gap = max(1, 1 + int(math.floor(recomb_tolerance / recombs_per_step)))

    return max_gap


# Position of inner_num_processes in a _process_single_batch task (unpacked
# below). The cross-node transport (core/batch_tasks.py) replaces it with the
# executing allocation's budget.
_INNER_PROCESSES_FIELD = 9


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
    (b_idx, start_i, end_i, original_blocks_list,
     recomb_rate,
     max_sites_for_linking, n_generations, recomb_tolerance,
     cc_scale, inner_num_processes, verbose,
     chromosome_map, panel_search_config, precomputed_informative_sample_mask) = args

    # A reused worker has finished serializing its previous result. Release
    # that batch's dead Python cycles and allocator arenas before attaching.
    gc.collect()
    core_parallel.malloc_trim()
    # Attach to shared memory (zero-copy).
    shm_probs, global_probs = _attach_shared_array(_SHARED_META['probs'])
    shm_sites, global_sites = _attach_shared_array(_SHARED_META['sites'])

    try:
        # Register this worker as active.
        core_parallel.increment_active()

        # Initial allocation — first phase uses inner pools, so start
        # numba at 1.
        core_parallel.get_dynamic_threads()
        numba.set_num_threads(1)

        original_portion = core_haplotypes.BlockResults(original_blocks_list)

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
        portion_proxy = core_haplotypes.BlockResults(proxy_list)

        # 2. Slice to batch-relevant sites only.
        all_positions = np.concatenate([b.positions for b in original_portion])
        batch_indices = np.searchsorted(global_sites, all_positions)
        idx_min, idx_max = batch_indices.min(), batch_indices.max()
        batch_sites = np.ascontiguousarray(global_sites[idx_min:idx_max + 1])
        total_sample_count = global_probs.shape[0]
        informative_sample_mask = np.asarray(
            precomputed_informative_sample_mask, dtype=np.bool_
        )
        if informative_sample_mask.shape != (total_sample_count,):
            raise ValueError("precomputed informative sample mask is not aligned")
        informative_sample_count = int(np.sum(informative_sample_mask))
        if informative_sample_count == 0:
            unresolved_components = list(original_portion)
            for block in unresolved_components:
                block.missing_aware_unresolved_phase_component = True
                block.hierarchy_informative_sample_count = 0
                block.hierarchy_total_sample_count = total_sample_count
                block.hierarchy_informative_sample_rule = (
                    'intersection_across_linking_proxy_batch_v2'
                )
            return {
                'batch_idx': b_idx,
                'unresolved_components': unresolved_components,
                'status': 'unresolved_passthrough',
                'informative_sample_count': 0,
                'total_sample_count': total_sample_count,
            }
        if len(original_portion) < 2:
            block = original_portion[0]
            block.hierarchy_informative_sample_count = informative_sample_count
            block.hierarchy_total_sample_count = total_sample_count
            block.hierarchy_informative_sample_rule = (
                'intersection_across_linking_proxy_batch_v2'
            )
            return {
                'batch_idx': b_idx,
                'super_block': block,
                'status': 'passthrough',
            }

        # The informative samples' evidence over the batch span, copied in parallel from the
        # shared evidence.
        core_parallel.apply_dynamic_threads()
        inference_batch_probs = assembly_chimera_kernels._gather_rows_numba(
            global_probs, np.flatnonzero(informative_sample_mask), int(idx_min), int(idx_max) + 1)

        # 3. Compute max gap.
        if n_generations is not None and recomb_tolerance is not None:
            beam_max_gap = compute_max_gap(original_blocks_list, recomb_rate,
                                           n_generations, recomb_tolerance,
                                           chromosome_map=chromosome_map)
        else:
            beam_max_gap = None
        if _prof:
            _acc('setup(proxy+slice+gap)', _t)

        # =================================================================
        # 4. Generate Mesh — DYNAMIC SEQUENTIAL phase.
        # Emissions: ThreadPoolExecutor (pure numpy, fast, no numba).
        # Mesh EM: sequential over gaps with dynamic numba threads;
        # core.parallel.get_dynamic_threads called between each gap (and inside the
        # EM loop via dynamic_cores_fn) so this worker scales up as
        # peers finish.
        # =================================================================
        dyn_threads = core_parallel.get_dynamic_threads()
        pool_budget = max(inner_num_processes, dyn_threads)

        # Emissions: ThreadPoolExecutor (threads release GIL inside
        # the numba kernel; no oversubscription risk).
        _t = time.perf_counter()
        viterbi_emissions = assembly_linking.generate_viterbi_block_emissions(
            inference_batch_probs, batch_sites, portion_proxy,
            num_processes=pool_budget,
        )
        if _prof:
            _acc('mesh_emissions', _t)
        _t = time.perf_counter()
        mesh = assembly_linking.generate_transition_probability_mesh(
            portion_proxy,
            recomb_rate=recomb_rate,
            precalculated_viterbi_emissions=viterbi_emissions,
            dynamic_cores_fn=core_parallel.get_dynamic_threads,
            chromosome_map=chromosome_map,
            max_gap=beam_max_gap,
        )
        if _prof:
            _acc('mesh_transition', _t)
        del viterbi_emissions

        # =================================================================
        # 5. Beam Search — NUMBA-ONLY phase.  No inner pool; give all
        # dynamic threads to numba.
        # =================================================================
        core_parallel.apply_dynamic_threads()
        _t = time.perf_counter()
        # One dense mesh per batch, shared by the beam and the panel search.
        fast_mesh = assembly_paths.FastMesh(portion_proxy, mesh)

        # Free mesh — fast_mesh has what it needs.
        del mesh
        core_parallel.malloc_trim()

        beam_results = assembly_paths.run_endpoint_beam(
            fast_mesh, endpoint_quota=panel_search_config.paths_per_endpoint,
            max_gap=beam_max_gap, verbose=verbose,
        )
        if _prof:
            _acc('beam_search', _t)

        if not beam_results:
            return {
                'batch_idx': b_idx,
                'super_block': None,
                'status': 'beam_search_failed'
            }

        # =================================================================
        # 6. Selection + Swap + CR — NUMBA-ONLY phase.
        # =================================================================
        core_parallel.apply_dynamic_threads()
        _t = time.perf_counter()
        search_diagnostics = []
        from .panel_search import select_and_resolve

        def beam_runner(quota, overrides):
            # Pool extension and saturation re-runs of this batch's beam.
            return assembly_paths.run_endpoint_beam(
                fast_mesh, endpoint_quota=quota, endpoint_quota_overrides=overrides,
                max_gap=beam_max_gap, verbose=False)
        resolved_beam = select_and_resolve(
            beam_results, fast_mesh, list(original_portion),
            inference_batch_probs, batch_sites, beam_runner=beam_runner,
            config=panel_search_config,
            cc_scale=cc_scale, num_threads=core_parallel.get_dynamic_threads,
            diagnostics=search_diagnostics,
        )
        if _prof:
            _acc('select_and_resolve(CR)', _t)

        del beam_results
        core_parallel.malloc_trim()

        # =================================================================
        # 7. Reconstruction — NUMBA-ONLY phase.
        # =================================================================
        core_parallel.apply_dynamic_threads()
        _t = time.perf_counter()
        reconstructed_data = assembly_paths.reconstruct_haplotypes_from_beam(
            resolved_beam, fast_mesh, original_portion,
        )
        if _prof:
            _acc('reconstruction', _t)

        del resolved_beam, fast_mesh
        core_parallel.malloc_trim()

        # 8. Package. The super-block carries no evidence array: every hierarchy level's output
        # drops block evidence at once (pipeline: core_runtime.strip_block_evidence), so
        # gathering it (N x sites x 3, a whole chromosome at L4) and pickling it back to the
        # parent only cost time and memory.
        _t = time.perf_counter()
        super_block = convert_reconstruction_to_superblock(
            reconstructed_data, original_portion,
        )
        if super_block is not None:
            super_block.hierarchy_informative_sample_count = (
                informative_sample_count
            )
            super_block.hierarchy_total_sample_count = total_sample_count
            if search_diagnostics:
                super_block.panel_search_diagnostics = search_diagnostics
            super_block.hierarchy_informative_sample_rule = (
                'intersection_across_linking_proxy_batch_v2'
            )
        if _prof:
            _acc('package', _t)

        del reconstructed_data, inference_batch_probs
        del batch_sites, portion_proxy, proxy_list
        core_parallel.malloc_trim()


        if _prof:
            try:
                _nL = len(super_block.positions) if super_block is not None else 0
                if _nL >= _HIER_PROFILE_MIN_L:
                    _wall = time.perf_counter() - _t_batch
                    _timed = sum(v[0] for v in _pt.values())
                    _aw = core_parallel.active_value()
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
        core_parallel.release_dynamic_extra()
        core_parallel.decrement_active()
        # Detach from shared memory (parent unlinks).
        core_parallel.close_shared_memory([shm_probs, shm_sites])


def _hmm_batch_memory_gb(input_blocks, batch_ranges, n_samples, max_sites,
                         global_sites, panel_search_config=None):
    """Budget the arrays one batch holds, in GB, for the largest batch.

    The batch keeps a float32 genotype-probability slice of its informative
    samples over its site span; as much again covers transient copies (the
    panel search's flat emission layout while it is built, the working arrays
    of concurrent duplicate-seed refits). Linker scans keep seven float32
    log emissions and seven float64 weights per sample and proxy site, plus
    uint8 pair lookups per proxy site. Panel search keeps float64 binned
    sub-emissions (S, K_b, K_b, m_b), with m_b = ceil(sites_b / spb) binned as
    select_and_resolve bins them, their static per-block totals (S, K_b, K_b),
    an int32 painting (S, sum m_b) and painting back pointers bounded by
    tensor_budget_mb. The certified search adds its exclusion forwards
    (S, sum m_b) per excluded row (at most K + 1), one chunk of per-sample
    bounds and, at the pool backstop (quota 16 * 2**6 at every endpoint, which
    a join whose 256 rung accepted a move can reach), the move arrays and beam
    lists of O(B q K^2) entries. Counting all samples and holding these together
    is conservative; the configured per-worker minimum covers the rest of the
    batch machinery.
    """
    from .panel_certify import _MOVES_PER_CHUNK
    from .panel_search import POOL_EXTENDED_DOUBLINGS
    from .panel_search import PanelSearchConfig
    config = PanelSearchConfig() if panel_search_config is None else panel_search_config
    maximum_bytes = 0
    for start, stop in batch_ranges:
        if stop - start < 2:
            continue
        batch = input_blocks[start:stop]
        site_counts = [len(block.positions) for block in batch]
        spb = max(assembly_chimera_scoring.compute_spb(batch),
                  math.ceil(sum(site_counts) / config.max_bins))
        first = min(int(block.positions[0]) for block in batch)
        last = max(int(block.positions[-1]) for block in batch)
        span = int(np.searchsorted(global_sites, last)
                   - np.searchsorted(global_sites, first)) + 1
        batch_bytes = config.tensor_budget_mb * 1024 ** 2 + 2 * 12 * n_samples * span
        total_bins = 0
        for block, sites in zip(batch, site_counts):
            proxy_sites = min(sites, max_sites)
            haps = len(block.haplotypes)
            ordered = haps * haps
            folded = haps * (haps + 1) // 2
            bins = math.ceil(sites / spb)
            total_bins += bins
            batch_bytes += (n_samples * proxy_sites * 7 * 12
                            + proxy_sites * (ordered + folded)
                            + 8 * n_samples * ordered * (bins + 1)
                            + 4 * n_samples * bins)
        rows = max(len(block.haplotypes) for block in batch)
        quota = config.paths_per_endpoint * 2 ** POOL_EXTENDED_DOUBLINGS
        batch_bytes += (8 * n_samples * total_bins * (rows + 2)
                        + 8 * n_samples * _MOVES_PER_CHUNK
                        + 200 * len(batch) * quota * rows * rows)
        maximum_bytes = max(maximum_bytes, batch_bytes)
    return maximum_bytes / (1024 ** 3)


def run_hierarchical_step(input_blocks, global_probs, global_sites,
                          batch_size=10,
                          # Linking Parameters
                          recomb_rate=5e-8,
                          # Memory Safety
                          max_sites_for_linking=2000,
                          # Max Gap Parameters
                          n_generations=None,
                          recomb_tolerance=0.5,
                          # Panel selection cost scale
                          cc_scale=0.5,
                          # Parallelization
                          num_processes=16,
                          maxtasksperchild=None,
                          min_gb_per_worker=4.0,
                          # Output control
                          verbose=False,
                          min_boundary_informative_samples=1,
                          chromosome_map=None,
                          panel_search_config=None, scoring_probs=None,
                          batch_queue_key=None, code_identity=None, checkpoints=None,
                          batches=None):
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
      - Worker count auto-capped based on available RAM / the per-worker
        budget (batch estimate, at least min_gb_per_worker).

    Dynamic thread reallocation: shared counter tracks active workers;
    between major phases each worker recalculates
    threads = total_cores // active_workers (+1 for `remainder %
    active` workers).  See the DYNAMIC THREAD REALLOCATION header
    block.

    Args:
      num_processes: Maximum total cores.  Strict ceiling on both
          concurrent workers AND total thread allocation.  Function
          may use fewer workers if RAM is tight, but thread allocation
          across workers will sum to at most num_processes.  An int, or a
          callable live budget (a contig lane's growing lease): the worker
          count follows its value at the start, the workers' shared thread
          total its growth (core.parallel.live_total).
      maxtasksperchild: Recycle workers after this many batches.  Set
          to 1 to prevent memory accumulation from glibc malloc
          fragmentation.
      min_gb_per_worker: Minimum GB of RAM to budget per concurrent
          worker.  The budget is the larger of this and the largest
          batch's estimated arrays (_hmm_batch_memory_gb), and
          max_workers = available_ram / budget.
      min_boundary_informative_samples: Require this many samples informative
          in both adjacent frozen inference panels.
      code_identity: the assembly code identity (pipeline.assembly_release_code_identity).
          With it every joined batch (two or more blocks, some informative sample) gets
          the digest of its exact inputs (_batch_input_digest).
      checkpoints: this level's records (assembly.checkpoints.AssemblyCheckpoints scoped
          to the level): its plan, [first input, end input, digest or None] per batch, is
          recorded before any batch runs; a batch whose digest some assembly of the contig
          already joined reuses that join record (joins are deterministic, so its
          super-block is identical); every other joined batch is recorded by this
          (coordinating) process as it completes, also from the batch queue.
          Pass-through batches have no digest and are not recorded here.
      batches: a list that receives, per batch in order, (digest or None, its output
          blocks), for the caller's level manifest.

    IMPORTANT: The entry script must NOT be named main.py — otherwise
    forkserver workers will re-execute it.
    """
    budget, num_processes = num_processes, core_parallel.budget_value(num_processes)
    total_blocks = len(input_blocks)
    with core_parallel.numba_thread_scope(num_processes):
        batch_ranges, block_informative_masks, boundary_joint_counts = (
            _missing_aware_batch_ranges(
                input_blocks,
                global_probs,
                global_sites,
                batch_size,
                min_boundary_informative_samples,
                max_sites=max_sites_for_linking,
            )
        )
    num_batches = len(batch_ranges)

    # num_processes is the user's ceiling — never exceed it for either
    # concurrent workers or total thread allocation.
    total_cores = num_processes

    print(f"\n--- Starting Hierarchical Step ---")
    print(f"Input: {total_blocks} blocks -> Target: ~{num_batches} Super-Blocks")
    if boundary_joint_counts:
        hard_break_count = sum(
            bool(getattr(input_blocks[index], 'missing_aware_break_after', False))
            for index in range(total_blocks - 1)
        )
        print(
            f"  Missing-aware hard phase breaks: {hard_break_count}; "
            f"minimum joint informative samples="
            f"{min_boundary_informative_samples}"
        )
    if n_generations is not None:
        preview_max_gap = compute_max_gap(list(input_blocks), recomb_rate,
                                           n_generations, recomb_tolerance,
                                           chromosome_map=chromosome_map)
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
    _stripped_bytes = 0
    for block in input_blocks:
        if hasattr(block, 'probs_array') and block.probs_array is not None:
            _stripped_bytes += block.probs_array.nbytes
            block.probs_array = None
    if _stripped_bytes > 0:
        gc.collect()
        core_parallel.malloc_trim()
        print(f"  Stripped probs_array from blocks ({_stripped_bytes / (1024**3):.1f} GB freed)")

    # Downcast to float32: global_probs is float64 from block discovery (HDBSCAN
    # needs float64) but assembly only uses it for emission scoring
    # where float32 precision is sufficient.  Halves shared memory,
    # per-worker batch_probs slices, and all downstream emission/
    # chimera tensors (they inherit dtype).
    if scoring_probs is not None:
        # Eligibility above deliberately uses the original precision; only
        # numerical assembly scoring has always used the float32 tensor.
        if scoring_probs.shape != global_probs.shape or scoring_probs.dtype != np.float32:
            raise ValueError("shared hierarchy scoring evidence must be aligned float32")
        global_probs = scoring_probs
    elif global_probs.dtype == np.float64:
        global_probs = global_probs.astype(np.float32)
        print(f"  Downcast global_probs to float32 ({global_probs.nbytes / (1024**3):.1f} GB)")

    # Also downcast block haplotypes (soft probabilities) if float64.
    for block in input_blocks:
        for k, h in block.haplotypes.items():
            if h.dtype == np.float64:
                block.haplotypes[k] = h.astype(np.float32)

    batch_masks = [_missing_aware_batch_informative_sample_mask(
        block_informative_masks[start_i:end_i]) for start_i, end_i in batch_ranges]
    digests = {}
    if code_identity is not None:
        context = json.dumps(dict(
            schema='hierarchy-batch-inputs-v1', code=code_identity, recomb_rate=recomb_rate,
            max_sites_for_linking=max_sites_for_linking, n_generations=n_generations,
            recomb_tolerance=recomb_tolerance, cc_scale=cc_scale,
            chromosome_map=None if chromosome_map is None else chromosome_map.identity(),
            panel_search=None if panel_search_config is None else asdict(panel_search_config)),
            sort_keys=True, separators=(',', ':')).encode()
        joined = [b_idx for b_idx, (start_i, end_i) in enumerate(batch_ranges)
                  if end_i - start_i >= 2 and np.any(batch_masks[b_idx])]
        # Evidence slices are digested concurrently; the block attributes, mostly small Python
        # objects, in this thread (concurrent small updates only contend for the GIL).
        with ThreadPoolExecutor(max_workers=max(1, min(total_cores, len(joined)))) as executor:
            evidence = list(executor.map(lambda b_idx: _evidence_digest(
                input_blocks[batch_ranges[b_idx][0]:batch_ranges[b_idx][1]], global_probs,
                global_sites), joined))
        digests = {b_idx: _batch_input_digest(
            context, input_blocks[batch_ranges[b_idx][0]:batch_ranges[b_idx][1]],
            batch_masks[b_idx], value) for b_idx, value in zip(joined, evidence)}
    reused = {}
    if checkpoints is not None:
        if not checkpoints.has("plan"):
            checkpoints.save("plan", {"batches": [
                [start_i, end_i, digests.get(b_idx)]
                for b_idx, (start_i, end_i) in enumerate(batch_ranges)]})
        recorded = checkpoints.joins(digests.values())
        reused = {b_idx: recorded[digest]["super_block"] for b_idx, digest in digests.items()
                  if digest in recorded}
    if reused:
        print(f"  Reused {len(reused)} of {num_batches} batches already joined with "
              f"identical inputs")

    def record(result):
        """Record a completed join (the coordinating process, as results arrive)."""
        digest = digests.get(result['batch_idx'])
        if checkpoints is not None and digest is not None and result['status'] == 'success':
            checkpoints.save_join(digest, {'status': result['status'],
                                           'super_block': result['super_block']})
        return result

    def make_worker_args(inner_num_processes):
        worker_args = []

        for b_idx, (start_i, end_i) in enumerate(batch_ranges):
            if b_idx in reused:
                continue
            original_blocks_list = list(input_blocks[start_i:end_i])

            worker_args.append((
                b_idx, start_i, end_i, original_blocks_list,
                recomb_rate,
                max_sites_for_linking, n_generations, recomb_tolerance,
                cc_scale, inner_num_processes, verbose,
                chromosome_map, panel_search_config, batch_masks[b_idx],
            ))
        return worker_args

    def finish(results):
        results = list(results) + [
            {'batch_idx': b_idx, 'super_block': block, 'status': 'reused'}
            for b_idx, block in reused.items()]
        if batches is not None:
            batches.extend(_batch_outputs(results, digests, num_batches))
        return _collect_batch_results(results)

    pending_ranges = [batch_range for b_idx, batch_range in enumerate(batch_ranges)
                      if b_idx not in reused]
    if not pending_ranges:
        return finish([])

    from ..core import batch_queue
    if batch_queue.configured() and num_batches > 1:
        worker_gb = max(min_gb_per_worker, _hmm_batch_memory_gb(
            input_blocks, pending_ranges, global_probs.shape[0], max_sites_for_linking,
            global_sites, panel_search_config))
        if reused and batch_queue_key is not None:
            # Queue results are keyed by bundle number: a different task subset is another stage.
            batch_queue_key = hashlib.sha256(json.dumps(
                [batch_queue_key, sorted(reused)]).encode()).hexdigest()
        results = [record(result) for result in batch_queue.map_batches(
            'hierarchy', make_worker_args(1), (global_probs, global_sites), total_cores,
            key=batch_queue_key, worker_memory_gb=worker_gb)]
        return finish(results)

    # Create POSIX shared memory for the global arrays.
    t0 = time.time()
    shm_probs, probs_meta = _create_shared_array(global_probs, 'global_probs', total_cores)
    shm_sites, sites_meta = _create_shared_array(global_sites, 'global_sites', total_cores)

    shared_meta = {
        'probs': probs_meta,
        'sites': sites_meta,
    }

    probs_gb = global_probs.nbytes / (1024 ** 3)
    print(f"  Shared memory created: {probs_gb:.1f} GB probs + sites ({time.time()-t0:.1f}s)")

    # Auto-size worker count based on available RAM, read AFTER shared
    # memory creation so it already accounts for parent + loaded data
    # + shared segments.
    worker_gb = max(min_gb_per_worker, _hmm_batch_memory_gb(
        input_blocks, pending_ranges, global_probs.shape[0], max_sites_for_linking,
        global_sites, panel_search_config))
    max_by_ram = num_processes  # fallback: no capping
    try:
        with open('/proc/meminfo') as _f:
            for _line in _f:
                if _line.startswith('MemAvailable:'):
                    mem_available_gb = int(_line.split()[1]) / (1024 * 1024)
                    max_by_ram = max(1, int(mem_available_gb / worker_gb))
                    break
    except Exception:
        pass  # non-Linux or /proc unavailable — use num_processes as-is

    num_tasks = len(pending_ranges)
    outer_workers = min(num_tasks, num_processes, max_by_ram)
    inner_num_processes = max(1, total_cores // outer_workers)

    # Preview
    print(f"Parallelism: {outer_workers} outer workers x {inner_num_processes} inner cores "
          f"= {outer_workers * inner_num_processes} total")
    if outer_workers < num_processes and outer_workers < num_tasks:
        print(f"  Workers capped by RAM: {mem_available_gb:.0f} GB available / "
              f"{worker_gb:.2f} GB per worker = {max_by_ram} max")
    print(f"  Dynamic threading: enabled (ceiling={total_cores} cores)")

    if maxtasksperchild is not None:
        tasks_per_worker = math.ceil(num_tasks / outer_workers)
        n_recycles = max(0, tasks_per_worker // maxtasksperchild - 1)
        print(f"  Worker recycling: every {maxtasksperchild} batches "
              f"(~{tasks_per_worker} tasks/worker, ~{n_recycles} recycles each)")

    # Shared counters for dynamic thread reallocation.  active_counter
    # tracks live worker count; extra_counter distributes the
    # remainder = total_cores % active so no cores stay idle.  Same
    # forkserver context for shared-memory consistency.  See the
    # DYNAMIC THREAD REALLOCATION header block for the full mechanism.
    active_counter = core_parallel.forkserver_context.Value('i', 0)
    extra_counter = core_parallel.forkserver_context.Value('i', 0)
    # The same distinct-worker startup gate used by block discovery prevents
    # an early worker from claiming the whole node while peers are starting.
    startup_counters = {
        "started_counter": core_parallel.forkserver_context.Value('i', 0),
        "participant_counter": core_parallel.forkserver_context.Value('i', 0),
        "batch_generation": core_parallel.forkserver_context.Value('i', 0),
        "batch_task_count": core_parallel.forkserver_context.Value('i', num_tasks),
        "startup_target": core_parallel.forkserver_context.Value('i', outer_workers),
        "startup_ready": core_parallel.forkserver_context.Value('i', 0),
    }

    # =====================================================================
    # Prepare worker arguments
    # =====================================================================
    worker_args = make_worker_args(inner_num_processes)

    informative_counts = [int(np.sum(args[-1])) for args in worker_args]
    print(
        "  Batch sample rule=intersection_across_linking_proxy_batch_v2; "
        f"informative samples min/max={min(informative_counts)}/"
        f"{max(informative_counts)} across {len(informative_counts)} "
        "actual batches"
    )

    # =====================================================================
    # Process batches
    # =====================================================================
    # Belt-and-suspenders: temporarily clear __main__.__file__ so
    # forkserver workers don't re-execute the entry script, even if
    # the caller forgot to add main guards to their pipeline file.
    _main_guard = core_parallel.main_module_guard()
    _main_guard.__enter__()

    try:
        if num_processes > 1:
            t0 = time.time()
            # A growing lane lease raises the workers' shared total mid-level.
            with core_parallel.live_total(budget) as live_cores:
                pool = core_parallel.NonDaemonicForkserverPool(
                    processes=outer_workers,
                    initializer=_init_worker_meta,
                    initargs=(shared_meta, total_cores, active_counter, extra_counter,
                              dict(startup_counters, live_total=live_cores)),
                    maxtasksperchild=maxtasksperchild
                )
                print(f"  Pool creation ({outer_workers} workers): {time.time()-t0:.1f}s")

                t0 = time.time()
                results = []
                try:
                    for result in tqdm(
                        pool.imap_unordered(_process_single_batch, worker_args),
                        total=num_tasks,
                        desc="Processing Batches"
                    ):
                        results.append(record(result))
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
            global _SHARED_META
            _SHARED_META = shared_meta
            # _ACTIVE_COUNTER stays None -> dynamic_threads helpers return 1.
            core_parallel.set_dynamic_thread_state(total_cores, None, None)
            results = []
            for args in tqdm(worker_args, desc="Processing Batches"):
                results.append(record(_process_single_batch(args)))
    finally:
        try:
            # Clean up shared memory (always, even on error)
            core_parallel.close_shared_memory([shm_probs, shm_sites], unlink=True)
        finally:
            _main_guard.__exit__(None, None, None)

    return finish(results)


def _batch_outputs(results, digests, num_batches):
    """Per batch in order: (the digest of its join record, or None when its output is
    kept inline: pass-through, unresolved or failed batches; its output blocks)."""
    by_batch = {result['batch_idx']: result for result in results}
    outputs = []
    for b_idx in range(num_batches):
        result = by_batch.get(b_idx, {})
        if result.get('unresolved_components') is not None:
            outputs.append((None, tuple(result['unresolved_components'])))
        elif result.get('super_block') is None:
            outputs.append((None, ()))
        elif result['status'] in ('success', 'reused'):
            outputs.append((digests.get(b_idx), (result['super_block'],)))
        else:
            outputs.append((None, (result['super_block'],)))
    return outputs


def _collect_batch_results(results):
    # Sort by batch index and collect super blocks
    results = sorted(results, key=lambda x: x['batch_idx'])

    output_super_blocks = []
    success_count = 0
    reused_count = 0
    passthrough_count = 0
    unresolved_count = 0
    failed_count = 0

    for result in results:
        unresolved = result.get('unresolved_components')
        if unresolved is not None:
            # Preserve each input block as its own unresolved phase component.
            # Never concatenate or invent a founder mapping without evidence.
            output_super_blocks.extend(unresolved)
            unresolved_count += len(unresolved)
            passthrough_count += len(unresolved)
            continue
        if result['super_block'] is not None:
            output_super_blocks.append(result['super_block'])
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'reused':
                reused_count += 1
            elif result['status'] == 'passthrough':
                passthrough_count += 1
        else:
            failed_count += 1

    print(f"Hierarchical Step Complete. Produced {len(output_super_blocks)} Super-Blocks.")
    print(f"  Success: {success_count}, Reused: {reused_count}, Passthrough: {passthrough_count}, "
          f"Unresolved: {unresolved_count}, Failed: {failed_count}")
    from .panel_search import search_summary
    summary = search_summary(
        getattr(block, 'panel_search_diagnostics', ()) for block in output_super_blocks)
    if summary['searched']:
        print("  Panel search: {searched} searched, {certified} certified, {changed} changed "
              "by tier 2 or pool saturation ({tier2_accepted} tier-2 and {saturation_accepted} "
              "saturation moves), {tier2_passes} tier-2 passes, sweep backstop hits "
              "{backstop_reached}, saturation ladders extended beyond 256 {extended_ladders}, "
              "pool backstop hits {pool_backstop}, largest pool "
              "{largest_pool}".format(**summary))
        print("  Join count rule: rows {start_rows} at the start -> {final_rows}; {births} births, "
              "{deaths} deaths ({pure_mosaic_deaths} without a private cell), {ambiguous} "
              "ambiguous calls, births frozen by a count cycle in {frozen} super-blocks ({cycles_undecided} "
              "undecided cycles), at most "
              "{most_decisions} decisions in one super-block; duplicate seeds: {seed_refits} refits, "
              "{seed_births} births, {seed_swaps} swaps; {ladder_births} saturation births; "
              "{trap_blocks} super-blocks with chimera-trap indicators; {failing_final} final paths "
              "fail their own death test".format(**summary))

    return core_haplotypes.BlockResults(output_super_blocks)

import haplotype_reconstruction.assembly.chimera_kernels as assembly_chimera_kernels
import haplotype_reconstruction.assembly.chimera_scoring as assembly_chimera_scoring
import haplotype_reconstruction.assembly.linking as assembly_linking
import haplotype_reconstruction.assembly.observations as assembly_observations
import haplotype_reconstruction.assembly.paths as assembly_paths
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.numerics as core_numerics
import haplotype_reconstruction.core.parallel as core_parallel
