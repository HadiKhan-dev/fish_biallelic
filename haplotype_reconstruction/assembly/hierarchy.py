"""assembly / hierarchy for the canonical reconstruction pipeline."""
from __future__ import annotations


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
    active/extra counters used by dynamic_threads.get_dynamic_threads.

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
    # helpers live in dynamic_threads.  set_dynamic_thread_state resets the
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


def _missing_aware_informative_sample_mask(batch_probs):
    """Samples with any positive-mass, nonuniform GL cell in this batch."""
    probabilities = np.asarray(batch_probs)
    positive_mass = np.sum(probabilities, axis=2) > 0.0
    nonuniform = (
        (probabilities[:, :, 0] != probabilities[:, :, 1])
        | (probabilities[:, :, 1] != probabilities[:, :, 2])
    )
    return np.any(positive_mass & nonuniform, axis=1)


def _missing_aware_block_indices(block, global_sites, max_sites=None):
    """Marker indices used by the unchanged frozen-panel eligibility rule."""

    panel = assembly_observations.founder_inference_panel_from_block_result(block)
    keep_flags = getattr(block, 'keep_flags', None)
    if keep_flags is None:
        retained = np.ones(len(block.positions), dtype=np.bool_)
    else:
        retained = np.asarray(keep_flags, dtype=np.bool_)
        if retained.shape != (len(block.positions),):
            raise ValueError("block keep_flags are not aligned with positions")
    globally_complete = retained & np.all(panel.called, axis=0)
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
    """Scan independent block/sample rows without gathering a GL tensor."""
    samples = probabilities.shape[0]
    result = np.zeros((len(index_sets), samples), np.bool_)
    for task in numba.prange(len(index_sets) * samples):
        block, sample = task // samples, task % samples
        indices = index_sets[block]
        for site in indices:
            a, b, c = probabilities[sample, site]
            if (a + b) + c > 0.0 and (a != b or b != c):
                result[block, sample] = True
                break
    return result


def _missing_aware_block_informative_sample_mask(
        block, global_probs, global_sites, max_sites=None):
    indices = List.empty_list(numba.types.int64[::1])
    indices.append(_missing_aware_block_indices(block, global_sites, max_sites))
    return _informative_samples_at_indices(np.asarray(global_probs), indices)[0]


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


def _linking_proxy_indices(block, max_sites, panel=None):
    """Thin usable panel markers, not positions that later become empty.

    Fully called/kept panels retain the original stride selection. Small blocks
    stay untouched; emission filtering still excludes their missing sites.
    Wholly unsupported blocks stay intact for unresolved passthrough.
    """
    total = len(block.positions)
    if total <= max_sites:
        return np.arange(total, dtype=np.int64)
    if panel is None:
        panel = assembly_observations.founder_inference_panel_from_block_result(block)
    usable = np.all(panel.called, axis=0)
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
            new_haps[k] = np.ascontiguousarray(v[sampled_indices, :])
        else:
            new_haps[k] = np.ascontiguousarray(v[sampled_indices])

    if block.keep_flags is not None:
        new_flags = np.ascontiguousarray(block.keep_flags[sampled_indices])
    else:
        new_flags = None

    new_reads = None
    if block.reads_count_matrix is not None:
        new_reads = np.ascontiguousarray(block.reads_count_matrix[:, sampled_indices, :])

    new_probs = None
    if block.probs_array is not None:
        new_probs = np.ascontiguousarray(block.probs_array[:, sampled_indices, :])

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
                >= atomic_source_row_counts[None, :]
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
        # worker.  See chimera_resolution._gather_samples_numba.
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
     recomb_rate, beam_width, max_founders,
     max_sites_for_linking, n_generations, recomb_tolerance,
     top_n_swap, max_cr_iterations, paint_penalty, min_hotspot_samples,
     cc_scale, inner_num_processes, verbose,
     chromosome_map, structured_transition_config, panel_search_config, precomputed_informative_sample_mask) = args

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
        batch_probs = np.ascontiguousarray(global_probs[:, idx_min:idx_max+1, :])
        batch_sites = np.ascontiguousarray(global_sites[idx_min:idx_max+1])
        total_sample_count = batch_probs.shape[0]
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
        inference_batch_probs = np.ascontiguousarray(
            batch_probs[informative_sample_mask]
        )

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

        # 3. Compute max gap.
        if n_generations is not None and recomb_tolerance is not None:
            beam_max_gap = compute_max_gap(original_blocks_list, recomb_rate,
                                           n_generations, recomb_tolerance,
                                           chromosome_map=chromosome_map)
        else:
            beam_max_gap = None
        if _prof: _acc('setup(proxy+slice+gap)', _t)

        # =================================================================
        # 4. Generate Mesh — DYNAMIC SEQUENTIAL phase.
        # Emissions: ThreadPoolExecutor (pure numpy, fast, no numba).
        # Mesh EM: sequential over gaps with dynamic numba threads;
        # dynamic_threads.get_dynamic_threads called between each gap (and inside the
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
        if _prof: _acc('mesh_emissions', _t)
        _t = time.perf_counter()
        mesh = assembly_linking.generate_transition_probability_mesh(
            None, None, portion_proxy,
            recomb_rate=recomb_rate,
            precalculated_viterbi_emissions=viterbi_emissions,
            num_processes=1,
            dynamic_cores_fn=core_parallel.get_dynamic_threads,
            chromosome_map=chromosome_map,
            max_gap=beam_max_gap,
            structured_config=structured_transition_config,
        )
        if _prof: _acc('mesh_transition', _t)
        del viterbi_emissions

        # =================================================================
        # 5. Beam Search — NUMBA-ONLY phase.  No inner pool; give all
        # dynamic threads to numba.
        # =================================================================
        core_parallel.apply_dynamic_threads()
        _t = time.perf_counter()
        beam_results = assembly_paths.run_full_mesh_beam_search(
            portion_proxy, mesh, beam_width=beam_width,
            max_gap=beam_max_gap, verbose=verbose,
            endpoint_quota=None if panel_search_config is None else panel_search_config.paths_per_endpoint,
        )
        if _prof: _acc('beam_search', _t)

        if not beam_results:
            return {
                'batch_idx': b_idx,
                'super_block': None,
                'status': 'beam_search_failed'
            }

        fast_mesh = assembly_paths.FastMesh(portion_proxy, mesh)

        # Free mesh — fast_mesh has what it needs.
        del mesh
        core_parallel.malloc_trim()

        # =================================================================
        # 6. Selection + Swap + CR — NUMBA-ONLY phase.
        # =================================================================
        core_parallel.apply_dynamic_threads()
        _t = time.perf_counter()
        search_diagnostics = []
        if panel_search_config is not None:
            from .panel_search import select_and_resolve
            resolved_beam = select_and_resolve(
                beam_results, fast_mesh, list(original_portion),
                inference_batch_probs, batch_sites, config=panel_search_config,
                cc_scale=cc_scale, num_threads=core_parallel.get_dynamic_threads,
                diagnostics=search_diagnostics,
            )
        else:
            resolved_beam = assembly_chimera_resolution.select_and_resolve(
                beam_results=beam_results,
                fast_mesh=fast_mesh,
                batch_blocks=list(original_portion),
                global_probs=inference_batch_probs,
                global_sites=batch_sites,
                max_founders=max_founders,
                top_n_swap=top_n_swap,
                max_cr_iterations=max_cr_iterations,
                paint_penalty=paint_penalty,
                min_hotspot_samples=min_hotspot_samples,
                cc_scale=cc_scale,
                num_threads=core_parallel.get_dynamic_threads,
            )
        if _prof: _acc('select_and_resolve(CR)', _t)

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
        if _prof: _acc('reconstruction', _t)

        del resolved_beam, fast_mesh
        core_parallel.malloc_trim()

        # 8. Package.
        _t = time.perf_counter()
        super_block = convert_reconstruction_to_superblock(
            reconstructed_data, original_portion, batch_probs, batch_sites,
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
        if _prof: _acc('package', _t)

        del reconstructed_data, inference_batch_probs
        del batch_probs, batch_sites, portion_proxy, proxy_list
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


def _hmm_batch_memory_gb(input_blocks, batch_ranges, n_samples, max_sites):
    """Budget retained scan arrays plus one full log-tensor workspace.

    Production stores three float32 log emissions and three float64 weights
    per sample/site, plus uint8 dosage indices. Reserve one dense fallback
    workspace as well. Proxy size is bounded by the cap, including after
    missing-aware marker selection; counting all samples is conservative.
    This supplements, rather than replaces, the configured minimum covering
    the rest of the batch machinery.
    """
    maximum_bytes = 0
    for start, stop in batch_ranges:
        if stop - start < 2:
            continue
        batch_bytes = 0
        for block in input_blocks[start:stop]:
            sites = len(block.positions)
            proxy_sites = min(sites, max_sites)
            haps = len(block.haplotypes)
            ordered = haps * haps
            folded = haps * (haps + 1) // 2
            batch_bytes += (n_samples * proxy_sites * (3 * 12 + 4 * ordered)
                            + proxy_sites * (ordered + folded))
        maximum_bytes = max(maximum_bytes, batch_bytes)
    return maximum_bytes / (1024 ** 3)


def run_hierarchical_step(input_blocks, global_probs, global_sites,
                          batch_size=10,
                          # Linking Parameters
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
                          min_boundary_informative_samples=1,
                          chromosome_map=None, structured_transition_config=None,
                          panel_search_config=None, scoring_probs=None):
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
      min_boundary_informative_samples: Require this many samples informative
          in both adjacent frozen inference panels.

    IMPORTANT: The entry script must NOT be named main.py — otherwise
    forkserver workers will re-execute it.
    """
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

    # Downcast to float32: global_probs is float64 from R01 (HDBSCAN
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

    # Create POSIX shared memory for the global arrays.
    t0 = time.time()
    shm_probs, probs_meta = _create_shared_array(global_probs, 'global_probs', total_cores)
    shm_sites, sites_meta = _create_shared_array(global_sites, 'global_sites', total_cores)

    shared_meta = {
        'probs': probs_meta,
        'sites': sites_meta,
    }

    probs_gb = global_probs.nbytes / (1024**3)
    print(f"  Shared memory created: {probs_gb:.1f} GB probs + sites ({time.time()-t0:.1f}s)")

    # Auto-size worker count based on available RAM, read AFTER shared
    # memory creation so it already accounts for parent + loaded data
    # + shared segments.
    worker_gb = max(min_gb_per_worker, _hmm_batch_memory_gb(
        input_blocks, batch_ranges, global_probs.shape[0], max_sites_for_linking))
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

    outer_workers = min(num_batches, num_processes, max_by_ram)
    inner_num_processes = max(1, total_cores // outer_workers)

    # Preview
    print(f"Parallelism: {outer_workers} outer workers x {inner_num_processes} inner cores "
          f"= {outer_workers * inner_num_processes} total")
    if outer_workers < num_processes and outer_workers < num_batches:
        print(f"  Workers capped by RAM: {mem_available_gb:.0f} GB available / "
              f"{worker_gb:.2f} GB per worker = {max_by_ram} max")
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
    active_counter = core_parallel.forkserver_context.Value('i', 0)
    extra_counter = core_parallel.forkserver_context.Value('i', 0)
    # The same distinct-worker startup gate used by block discovery prevents
    # an early worker from claiming the whole node while peers are starting.
    startup_counters = {
        "started_counter": core_parallel.forkserver_context.Value('i', 0),
        "participant_counter": core_parallel.forkserver_context.Value('i', 0),
        "batch_generation": core_parallel.forkserver_context.Value('i', 0),
        "batch_task_count": core_parallel.forkserver_context.Value('i', num_batches),
        "startup_target": core_parallel.forkserver_context.Value('i', outer_workers),
        "startup_ready": core_parallel.forkserver_context.Value('i', 0),
    }

    # =====================================================================
    # Prepare worker arguments
    # =====================================================================
    worker_args = []

    for b_idx, (start_i, end_i) in enumerate(batch_ranges):
        original_blocks_list = list(input_blocks[start_i:end_i])
        batch_informative_sample_mask = (
            _missing_aware_batch_informative_sample_mask(
                block_informative_masks[start_i:end_i]
            )
        )

        worker_args.append((
            b_idx, start_i, end_i, original_blocks_list,
            recomb_rate, beam_width, max_founders,
            max_sites_for_linking, n_generations, recomb_tolerance,
            top_n_swap, max_cr_iterations, paint_penalty, min_hotspot_samples,
            cc_scale, inner_num_processes, verbose,
            chromosome_map, structured_transition_config, panel_search_config, batch_informative_sample_mask
        ))

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
            pool = core_parallel.NonDaemonicForkserverPool(
                processes=outer_workers,
                initializer=_init_worker_meta,
                initargs=(shared_meta, total_cores, active_counter, extra_counter,
                          startup_counters),
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
            global _SHARED_META
            _SHARED_META = shared_meta
            # _ACTIVE_COUNTER stays None -> dynamic_threads helpers return 1.
            core_parallel.set_dynamic_thread_state(total_cores, None, None)
            results = []
            for args in tqdm(worker_args, desc="Processing Batches"):
                results.append(_process_single_batch(args))
    finally:
        try:
            # Clean up shared memory (always, even on error)
            core_parallel.close_shared_memory([shm_probs, shm_sites], unlink=True)
        finally:
            _main_guard.__exit__(None, None, None)

    # Sort by batch index and collect super blocks
    results = sorted(results, key=lambda x: x['batch_idx'])

    output_super_blocks = []
    success_count = 0
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
            elif result['status'] == 'passthrough':
                passthrough_count += 1
        else:
            failed_count += 1

    print(f"Hierarchical Step Complete. Produced {len(output_super_blocks)} Super-Blocks.")
    print(f"  Success: {success_count}, Passthrough: {passthrough_count}, "
          f"Unresolved: {unresolved_count}, Failed: {failed_count}")

    return core_haplotypes.BlockResults(output_super_blocks)

import haplotype_reconstruction.assembly.chimera_kernels as assembly_chimera_kernels
import haplotype_reconstruction.assembly.chimera_resolution as assembly_chimera_resolution
import haplotype_reconstruction.assembly.linking as assembly_linking
import haplotype_reconstruction.assembly.observations as assembly_observations
import haplotype_reconstruction.assembly.paths as assembly_paths
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.numerics as core_numerics
import haplotype_reconstruction.core.parallel as core_parallel
