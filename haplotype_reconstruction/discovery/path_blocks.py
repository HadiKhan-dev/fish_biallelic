"""Parallel, checkpointed normalized-path selection of local founder panels.

Raw discovery supplies latent starts; assembled context supplies candidates,
never additional observations. Every round fits the same calibrated GLs.
"""
import copy
from dataclasses import asdict
import time

import numpy as np
from numba import set_num_threads

from ..core import parallel
from ..core.haplotypes import BlockResult, BlockResults
from . import feedback, path_fitting, path_scoring, path_selection


def kept(block):
    return (np.ones(len(block.positions), dtype=bool) if block.keep_flags is None
            else np.asarray(block.keep_flags) > 0)


def materialize(original, result):
    """Release supported alleles without inventing constant sample pairs.

    Directional support is fractional expected independent carriers. Spatial
    MAP-U counts, used by completion, are distinct from posterior mean U mass.
    """
    release, fit = result['release'], result['fit']
    keep = kept(original)
    calls = np.full((len(fit['panel']), len(keep)), -1, dtype=np.int8)
    probability = np.full(calls.shape, .5)
    support = np.zeros(calls.shape)
    calls[:, keep] = release['calls']
    probability[:, keep] = fit['allele_probability']
    support[:, keep] = release['directional_support']
    public = np.where(calls >= 0, probability, .5)
    block = BlockResult(original.positions.copy(),
        {i: np.stack((1-public[i], public[i]), axis=-1) for i in range(len(calls))},
        keep_flags=copy.deepcopy(original.keep_flags), genotype_evidence_mode='raw_likelihood')
    block.discrete_haps = calls
    block.founder_alt_pseudo_probability = probability
    block.founder_allele_pseudo_confidence = np.maximum(probability, 1-probability)
    block.n_directional_site_supporters = support
    block.path_fit = fit
    block.path_unknown_copy_fraction = release['unknown_copy_mass'] / release['total_copy_mass']
    block.sample_has_observed_kept_depth = result['wildcard']['has_depth']
    block.wildcard_slots = result['wildcard']['slots']
    block.wildcard_mass = result['wildcard']['mass']
    block.path_wildcard_interpretation = (
        'Maximum over observed kept markers of marginal MAP unknown-copy count; '
        'not a constant pair or probability of any unknown state.')
    for name in ('missing_aware_break_before', 'missing_aware_break_after'):
        if hasattr(original, name):
            setattr(block, name, getattr(original, name))
    return block


def worker(task):
    """Picklable callback; shared inputs contain no simulation truth."""
    operation, index, columns, positions, latent, proposals, incumbent, config, search, model = task
    parallel.increment_active()
    started = time.monotonic()
    try:
        shared_gl, shared_observed = feedback._ARRAYS
        gl = np.ascontiguousarray(shared_gl[:, columns], dtype=np.float64)
        observed = np.asarray(shared_observed[:, columns], dtype=bool, order='C')
        parallel.apply_dynamic_threads()
        prepared = path_fitting.prepare_fit_observations(
            path_scoring.prepare_observations(gl, observed, positions, **model))
        if operation == 'select':
            start, bank, bank_diagnostic = path_selection.prepare_candidate_bank(
                gl, observed, latent, proposals, config, incumbent=incumbent)
            attempted = path_selection.search_panel(start, bank, gl, observed, positions,
                learn_frequencies=True, prepared=prepared, previous_search=incumbent,
                **search, **model)
            retained = incumbent is not None and incumbent['objective'] > attempted['objective'] + 1e-6
            fit = incumbent if retained else attempted
            diagnostic = dict(incumbent_retained=bool(retained),
                candidate_bank=bank_diagnostic, attempted_objective=attempted['objective'],
                attempted_trace=attempted['search_trace'], trace=fit['search_trace'],
                computational_work=attempted.get('computational_work'))
        else:
            from .path_exchange import exchange_panel
            fit, diagnostic = exchange_panel(incumbent, gl, observed, positions,
                prepared=prepared, max_updates=search['max_updates'], **model)
        # Reusing a deterministic search also permits reuse of its release
        # calculation when the calling rule is unchanged. These cached
        # posteriors are outputs only, never additional read observations.
        release_rule = (config.discovery.score_tolerance,
                        config.discovery.min_hard_call_pseudo_probability,
                        config.discovery.min_directional_supporters)
        saved_release = fit.get('release_result')
        reused_search = (operation == 'select' and
            diagnostic.get('computational_work', {}).get('reused_previous_search', False))
        reuse_release = (reused_search and saved_release is not None
                         and saved_release['rule'] == release_rule)
        if reuse_release:
            release, wildcard = saved_release['release'], saved_release['wildcard']
        else:
            release, wildcard = path_selection.release_and_wildcard(
                fit, gl, observed, positions, config.discovery, **model)
        fit = dict(fit, release_result=dict(
            rule=release_rule, release=release, wildcard=wildcard))
        diagnostic['release_reused'] = bool(reuse_release)
        diagnostic.update(k=len(fit['panel']), objective=fit['objective'],
            fixed_k_converged=fit['converged'], update_budget_reached=fit['update_budget_reached'],
            unknown_copy_fraction=release['unknown_copy_mass']/release['total_copy_mass'],
            spatial_map_unknown_mass=wildcard['mass'],
            expected_state_changes=release['expected_state_changes'],
            seconds=time.monotonic()-started)
        return index, dict(fit=fit, release=release, wildcard=wildcard, diagnostic=diagnostic)
    finally:
        set_num_threads(1)
        parallel.release_dynamic_extra()
        parallel.decrement_active()


def select_blocks(originals, proposal_sets, gl, sites, observed, cpus, config,
                  search_config, checkpoint_io, *, generations, recombination_rate_per_bp,
                  wildcard_mass=.01, chromosome_map=None, segment_exchange=False):
    """One worker per block initially; freed cores join remaining numerical work.

    Each completed block is saved by the parent. The caller supplies a
    single-compression-thread checkpoint store so tiny writes do not spawn a
    node-wide compression team while numerical workers are running.
    """
    output = BlockResults(list(originals))
    diagnostics, tasks = [None]*len(originals), []
    for index, original in enumerate(originals):
        phase = f'block{index:06d}'
        saved = checkpoint_io.load(phase)
        if saved is not None:
            output.blocks[index], diagnostics[index] = saved['block'], saved['diagnostic']
            continue
        keep = kept(original)
        columns = np.searchsorted(sites, original.positions)
        if np.any(columns >= len(sites)) or not np.array_equal(sites[columns], original.positions):
            raise ValueError('path selection block positions do not match genotype evidence')
        mode = getattr(original, 'cavity_selected_mode', None)
        incumbent = (getattr(original, 'path_fit', None) if segment_exchange else
                     getattr(proposal_sets[-2][index], 'path_fit', None)
                     if len(proposal_sets) >= 2 else None)
        unavailable = (incumbent is None if segment_exchange else mode is None)
        if unavailable or not keep.any() or not observed[:, columns[keep]].any():
            diagnostic = dict(skipped='no_latent_fit_or_observed_kept_sites')
            fallback = copy.copy(original)
            fallback.reads_count_matrix = fallback.probs_array = None
            output.blocks[index], diagnostics[index] = fallback, diagnostic
            checkpoint_io.save(phase, dict(block=fallback, diagnostic=diagnostic))
            continue
        proposals = []
        for blocks in proposal_sets:
            if not np.array_equal(blocks[index].positions, original.positions):
                raise ValueError('feedback proposal positions differ from original block')
            proposals.append(np.ascontiguousarray(blocks[index].discrete_haps[:, keep]))
        latent = None if segment_exchange else mode.haplotypes
        if latent is not None and latent.shape[1] != int(keep.sum()):
            raise ValueError('latent discovery panel does not match kept markers')
        positions = np.ascontiguousarray(original.positions[keep])
        intervals = (None if chromosome_map is None else
                     chromosome_map.interval_morgans(positions[:-1], positions[1:]))
        model = dict(generations=generations, recombination_rate_per_bp=recombination_rate_per_bp,
                     wildcard_mass=wildcard_mass, interval_morgans=intervals)
        tasks.append(('exchange' if segment_exchange else 'select', index, columns[keep],
                      positions, latent, proposals, incumbent, config, asdict(search_config), model))
    if not tasks:
        return output, diagnostics
    handles, metadata = [], []
    try:
        for array in (gl, observed):
            handle, meta = parallel.create_shared_array(array)
            handles.append(handle)
            metadata.append(meta)
        ctx = parallel.forkserver_context
        active, extra = ctx.Value('i', 0), ctx.Value('i', 0)
        workers = min(cpus, len(tasks))
        startup = {name: ctx.Value('i', value) for name, value in dict(
            started_counter=0, participant_counter=0, batch_generation=0,
            batch_task_count=len(tasks), startup_target=workers, startup_ready=0).items()}
        print(f'[Path selection] {len(tasks)} blocks; {workers} workers, '
              f'one initial thread each, dynamic ceiling {cpus}', flush=True)
        with parallel.ForkserverPool(workers, initializer=feedback.initialize,
                initargs=(metadata, cpus, active, extra, startup)) as pool:
            for done, (index, result) in enumerate(pool.imap_unordered(worker, tasks, chunksize=1), 1):
                block = materialize(originals[index], result)
                diagnostic = result['diagnostic']
                output.blocks[index], diagnostics[index] = block, diagnostic
                checkpoint_io.save(f'block{index:06d}', dict(block=block, diagnostic=diagnostic))
                if done % 20 == 0 or done == len(tasks):
                    print(f'[Path selection] completed {done}/{len(tasks)}', flush=True)
    finally:
        parallel.close_shared_memory(handles, unlink=True)
    return output, diagnostics
