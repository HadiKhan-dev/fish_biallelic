"""Parallel, checkpointed normalized-path selection of local founder panels.

Raw discovery supplies latent starts; assembled context supplies candidates,
never additional observations. Every round fits the same calibrated GLs.

A round's records (key prefix ``feedback/<round>`` of the contig file):
``blocks/<k:06d>``, its selected blocks CHUNK_BLOCKS at a time in completion
order, each with its diagnostic and likelihood memo (the round's product), and
``selected``, its completion: {chunks: count, skipped: indices of the blocks
kept as discovered (fallback), blocks: count}, so the round's panels are the
chunks' blocks plus fallback copies of the skipped discovery blocks.
"""
import copy
from dataclasses import asdict
import time

import numpy as np
from numba import set_num_threads

from ..core import parallel
from ..core.haplotypes import BlockResult, BlockResults
from . import feedback, path_fitting, path_scoring, path_selection

# Completed blocks per path-selection record (_BlockChunks).
CHUNK_BLOCKS = 64


# Diagnostic of a block kept as discovered (fallback).
SKIPPED = 'no_latent_fit_or_observed_kept_sites'


def kept(block):
    return (np.ones(len(block.positions), dtype=bool) if block.keep_flags is None
            else np.asarray(block.keep_flags) > 0)


def materialize(original, result):
    """Release supported alleles without inventing constant sample pairs.

    Directional supporters are the favourable samples that more likely than
    not carry the founder at the site (one of the release rule's two measures;
    the other, their expected carrier mass, is the release's
    directional_support). Spatial
    MAP-U counts, used by completion, are distinct from posterior mean U mass.
    """
    release, fit = result['release'], result['fit']
    keep = kept(original)
    calls = np.full((len(fit['panel']), len(keep)), -1, dtype=np.int8)
    probability = np.full(calls.shape, .5)
    support = np.zeros(calls.shape)
    calls[:, keep] = release['calls']
    probability[:, keep] = fit['allele_probability']
    support[:, keep] = release['directional_supporters']
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


def fallback(original):
    """A block without a latent fit or observed kept site: its discovery panel, no evidence."""
    block = copy.copy(original)
    block.reads_count_matrix = block.probs_array = None
    return block


def worker(task):
    """Picklable callback; shared inputs contain no simulation truth."""
    (index, columns, positions, latent, proposals, incumbent, config, search, model, founders,
     memo_state, keep_memo) = task
    parallel.increment_active()
    started = time.monotonic()
    try:
        shared_gl, shared_observed = feedback._ARRAYS
        gl = np.ascontiguousarray(shared_gl[:, columns], dtype=np.float64)
        observed = np.asarray(shared_observed[:, columns], dtype=bool, order='C')
        parallel.apply_dynamic_threads()
        prepared = path_fitting.prepare_fit_observations(
            path_scoring.prepare_observations(gl, observed, positions, **model))
        start, bank, bank_diagnostic = path_selection.prepare_candidate_bank(
            gl, observed, latent, proposals, config, incumbent=incumbent)
        memo = path_selection.LikelihoodMemo(memo_state)
        attempted = path_selection.search_panel(start, bank, gl, observed, positions,
            learn_frequencies=True, prepared=prepared, previous_search=incumbent,
            incumbent=incumbent, founders=founders, memo=memo, **search, **model)
        context = None if founders is None else path_selection.ContextFounders(founders)
        retained = incumbent is not None and (
            path_selection.selection_objective(incumbent, True, context)
            > attempted['selection_objective'] + 1e-6)
        fit = incumbent if retained else attempted
        diagnostic = dict(incumbent_retained=bool(retained),
            candidate_bank=bank_diagnostic, attempted_objective=attempted['objective'],
            attempted_trace=attempted['search_trace'], trace=fit['search_trace'],
            computational_work=attempted.get('computational_work'),
            search_start=attempted.get('search_start'),
            round_backstop=bool(attempted.get('round_backstop', False)))
        # The release is a function of the final fit's (panel, frequencies),
        # the block data and the calling rule. A final fit equal to the
        # incumbent's (retained, reused, or found again) reuses its release.
        # These cached posteriors are outputs only, never additional read
        # observations.
        release_rule = ('carrier_majority_supporters', config.discovery.score_tolerance,
                        config.discovery.min_hard_call_pseudo_probability,
                        config.discovery.min_directional_supporters)
        saved_release = None if incumbent is None else incumbent.get('release_result')
        reuse_release = (saved_release is not None and saved_release['rule'] == release_rule
                         and np.array_equal(fit['panel'], incumbent['panel'])
                         and np.array_equal(fit['frequencies'], incumbent['frequencies']))
        if reuse_release:
            release, wildcard = saved_release['release'], saved_release['wildcard']
        else:
            release, wildcard = path_selection.release_and_wildcard(
                fit, gl, observed, positions, config.discovery, **model)
        fit = dict(fit, release_result=dict(
            rule=release_rule, release=release, wildcard=wildcard))
        diagnostic['release_reused'] = bool(reuse_release)
        diagnostic.update(k=len(fit['panel']), objective=fit['objective'],
            selection_objective=path_selection.selection_objective(fit, True, context),
            context_founders=None if context is None else len(context),
            free_rows=len(fit['panel']) if context is None else context.free(fit['panel']),
            fixed_k_converged=fit['converged'], update_backstop=fit['update_backstop_reached'],
            unknown_copy_fraction=release['unknown_copy_mass']/release['total_copy_mass'],
            spatial_map_unknown_mass=wildcard['mass'],
            expected_state_changes=release['expected_state_changes'],
            memo_hits=memo.hits, seconds=time.monotonic()-started)
        # The memo goes to the next round beside, never inside, the block.
        return index, dict(fit=fit, release=release, wildcard=wildcard, diagnostic=diagnostic,
                           memo=memo.state() if keep_memo else None)
    finally:
        set_num_threads(1)
        parallel.release_dynamic_extra()
        parallel.decrement_active()


def backstop_summary(diagnostics):
    """Count reported backstop hits (target 0) over completed block selections."""
    counts = dict(blocks=0, round_backstop=0, update_backstop=0, candidate_bank_backstop=0)
    for diagnostic in diagnostics:
        if not diagnostic or 'skipped' in diagnostic:
            continue
        counts['blocks'] += 1
        counts['round_backstop'] += bool(diagnostic.get('round_backstop'))
        counts['update_backstop'] += bool(diagnostic.get('update_backstop'))
        counts['candidate_bank_backstop'] += bool(
            diagnostic.get('candidate_bank', {}).get('search_budget_bound'))
    return counts


class _BlockChunks:
    """Completed blocks recorded CHUNK_BLOCKS at a time, numbered from zero.

    Chunks are written in completion order and their numbers stay contiguous
    across resumes; a kill loses at most the unwritten blocks of one chunk.
    """

    def __init__(self, checkpoints):
        self.checkpoints, self.pending, self.written = checkpoints, [], 0

    def load(self, count=None):
        """Yield the recorded (index, block, diagnostic, memo) of every chunk (of ``count``)."""
        while count is None or self.written < count:
            saved = self.checkpoints.load(f'blocks/{self.written:06d}')
            if saved is None:
                if count is not None:
                    raise RuntimeError(f'{self.checkpoints!r}: path-selection chunk '
                                       f'{self.written} is missing')
                return
            self.written += 1
            yield from saved['blocks']

    def add(self, index, block, diagnostic, memo):
        self.pending.append((index, block, diagnostic, memo))
        if len(self.pending) == CHUNK_BLOCKS:
            self.flush()

    def flush(self):
        if self.pending:
            self.checkpoints.save(f'blocks/{self.written:06d}', dict(blocks=self.pending))
            self.pending, self.written = [], self.written + 1


def load_selection(checkpoints, originals):
    """A completed round's (blocks, diagnostics, memos) as select_blocks returned them.

    Rebuilt from the round's chunk records and fallback copies of its skipped
    discovery blocks (``originals``); None while the round is incomplete.
    """
    selected = checkpoints.load('selected')
    if selected is None:
        return None
    if selected['blocks'] != len(originals):
        raise ValueError(f'{checkpoints!r}: the round selected {selected["blocks"]} blocks, '
                         f'not the {len(originals)} discovery blocks given')
    output = BlockResults(list(originals))
    diagnostics, memos = [None]*len(originals), [None]*len(originals)
    for index, block, diagnostic, memo in _BlockChunks(checkpoints).load(selected['chunks']):
        output.blocks[index], diagnostics[index], memos[index] = block, diagnostic, memo
    for index in selected['skipped']:
        output.blocks[index], diagnostics[index] = fallback(originals[index]), dict(skipped=SKIPPED)
    return output, diagnostics, memos


def select_blocks(originals, proposal_sets, gl, sites, observed, cpus, config,
                  search_config, checkpoints, *, generations, recombination_rate_per_bp,
                  wildcard_mass=.01, chromosome_map=None, memo_states=None, keep_memo=False):
    """One worker per block initially; freed cores join remaining numerical work.

    The parent records completed blocks in chunks (_BlockChunks) and, at the
    end, the round's completion (``selected``; see the module docstring);
    blocks without a latent fit or observed kept site are cheap fallbacks
    rebuilt on resume. ``checkpoints`` (the round's records,
    assembly.checkpoints.AssemblyCheckpoints) write with one compression
    thread, so small writes do not spawn a node-wide compression team while
    numerical workers run. ``memo_states`` are the previous round's likelihood
    memos by block (LikelihoodMemo states); with ``keep_memo`` the updated ones are returned
    as the third value for the next round, else Nones. ``cpus``: an int, or a
    callable live budget (a contig lane's growing lease): the pool's process
    count follows its value at the start, the workers' shared thread total its
    growth (core.parallel.live_total).
    """
    budget, cpus = cpus, parallel.budget_value(cpus)
    output = BlockResults(list(originals))
    diagnostics, memos, tasks, skipped = [None]*len(originals), [None]*len(originals), [], []
    chunks = _BlockChunks(checkpoints)
    for index, block, diagnostic, memo in chunks.load():
        output.blocks[index], diagnostics[index], memos[index] = block, diagnostic, memo
    for index, original in enumerate(originals):
        if diagnostics[index] is not None:
            continue
        keep = kept(original)
        columns = np.searchsorted(sites, original.positions)
        if np.any(columns >= len(sites)) or not np.array_equal(sites[columns], original.positions):
            raise ValueError('path selection block positions do not match genotype evidence')
        mode = getattr(original, 'cavity_selected_mode', None)
        incumbent = (getattr(proposal_sets[-2][index], 'path_fit', None)
                     if len(proposal_sets) >= 2 else None)
        if mode is None or not keep.any() or not observed[:, columns[keep]].any():
            output.blocks[index] = fallback(original)
            diagnostics[index] = dict(skipped=SKIPPED)
            skipped.append(index)
            continue
        proposals = []
        for blocks in proposal_sets:
            if not np.array_equal(blocks[index].positions, original.positions):
                raise ValueError('feedback proposal positions differ from original block')
            proposals.append(np.ascontiguousarray(blocks[index].discrete_haps[:, keep]))
        latent = mode.haplotypes
        # This round's context refit (the newest proposal set) names the
        # founders the multi-block assembly found in this block; blocks no
        # context component covered, and the initial fit, carry none.
        founders = (getattr(proposal_sets[-1][index], 'context_founder_rows', None)
                    if proposal_sets else None)
        if founders is not None:
            founders = np.ascontiguousarray(np.asarray(founders)[:, keep])
        if latent.shape[1] != int(keep.sum()):
            raise ValueError('latent discovery panel does not match kept markers')
        positions = np.ascontiguousarray(original.positions[keep])
        intervals = (None if chromosome_map is None else
                     chromosome_map.interval_morgans(positions[:-1], positions[1:]))
        model = dict(generations=generations, recombination_rate_per_bp=recombination_rate_per_bp,
                     wildcard_mass=wildcard_mass, interval_morgans=intervals)
        tasks.append((index, columns[keep], positions, latent, proposals, incumbent, config,
                      asdict(search_config), model, founders,
                      None if memo_states is None else memo_states[index], keep_memo))

    def complete():
        checkpoints.save('selected', dict(chunks=chunks.written, skipped=tuple(skipped),
                                          blocks=len(originals)))
        return output, diagnostics, memos

    if not tasks:
        return complete()

    def completed(index, result):
        block, diagnostic = materialize(originals[index], result), result['diagnostic']
        output.blocks[index], diagnostics[index], memos[index] = block, diagnostic, result['memo']
        chunks.add(index, block, diagnostic, result['memo'])

    def finish():
        chunks.flush()
        hits = {}
        for diagnostic in diagnostics:
            for name, value in diagnostic.get('memo_hits', {}).items():
                hits[name] = hits.get(name, 0) + value
        print(f'[Path selection] backstop hits {backstop_summary(diagnostics)}; '
              f'likelihood memo hits {hits}', flush=True)
        return complete()

    from ..core import batch_queue
    if batch_queue.configured():
        for index, result in batch_queue.map_batches(
                'local_fit', tasks, (gl, observed), cpus):
            completed(index, result)
        return finish()
    handles, metadata = [], []
    try:
        for array in (gl, observed):
            handle, meta = parallel.create_shared_array(array, copy_threads=cpus)
            handles.append(handle)
            metadata.append(meta)
        ctx = parallel.forkserver_context
        active, extra = ctx.Value('i', 0), ctx.Value('i', 0)
        workers = min(cpus, len(tasks))
        startup = {name: ctx.Value('i', value) for name, value in dict(
            started_counter=0, participant_counter=0, batch_generation=0,
            batch_task_count=len(tasks), startup_target=workers, startup_ready=0).items()}
        print(f'[Path selection] {len(tasks)} blocks; {workers} workers, '
              f'one initial thread each, dynamic ceiling {cpus}'
              + (' (grows with the lease)' if callable(budget) else ''), flush=True)
        with parallel.live_total(budget) as live, parallel.ForkserverPool(
                workers, initializer=feedback.initialize,
                initargs=(metadata, cpus, active, extra, dict(startup, live_total=live))) as pool:
            for done, (index, result) in enumerate(pool.imap_unordered(worker, tasks, chunksize=1), 1):
                completed(index, result)
                if done % 20 == 0 or done == len(tasks):
                    print(f'[Path selection] completed {done}/{len(tasks)}', flush=True)
    finally:
        parallel.close_shared_memory(handles, unlink=True)
    return finish()
