"""Exact binned scores for contiguous, bounded two-founder exchanges.

The emission alphabet is unchanged by a pair permutation. Relabel the prefix
at the first boundary, propagate the original emissions inside the interval,
then relabel against the unchanged suffix. This scores both boundaries jointly
without introducing any extra diploid states or inventing local alleles.
The bounded interval scan costs O(P N W M K²), where P is the pair quota,
W the window budget and M the total number of prepared emission bins. Sparse
L1 starts and their reversed counterpart retain every bin and fine endpoint.
Pair shortlisting, when needed, additionally uses the cubic-in-K suffix queries
in founder_exchanges; the complete refinement is not claimed to be quadratic.
"""
import numpy as np
import time
from numba import njit, prange, get_num_threads
from..import founder_refinement as fr
from.import scoring as fs
from.import exchanges as ex
from..import paths, hierarchy
from.workspace import component_workspace
from.packing import emission_arrays, pack_selected, score_rows
from ...core import haplotypes


@njit(cache=True, parallel=True, nogil=True)
def flanks(emissions, offsets, penalty):
    blocks = len(offsets) - 1
    samples, _, states = emissions.shape
    prefix = np.zeros((blocks + 1, samples, states))
    suffix = np.zeros_like(prefix)
    for sample in prange(samples):
        row = np.zeros(states)
        value = np.empty(states)
        for block in range(blocks):
            for site in range(offsets[block], offsets[block + 1]):
                switched = np.max(row) - penalty
                for state in range(states):
                    row[state] = max(row[state], switched) + emissions[sample, site, state]
            prefix[block + 1, sample] = row
        row[:] = 0.0
        for block in range(blocks - 1, -1, -1):
            for site in range(offsets[block + 1] - 1, offsets[block] - 1, -1):
                for state in range(states):
                    value[state] = row[state] + emissions[sample, site, state]
                switched = np.max(value) - penalty
                for state in range(states):
                    row[state] = max(value[state], switched)
            suffix[block, sample] = row
    return prefix, suffix


def permutations(founders, pairs):
    first, second = np.triu_indices(founders)
    lookup = np.empty((founders, founders), np.int64)
    lookup[first, second] = lookup[second, first] = np.arange(len(first))
    result = []
    for a, b in pairs:
        swap = np.arange(founders)
        swap[a], swap[b] = (b, a)
        result.append(lookup[swap[first], swap[second]])
    return (np.asarray(result, np.int64), first, second)


@njit(cache=True, parallel=True)
def interval_scores(emissions, offsets, prefix, suffix, permutation, penalty, starts, stops):
    samples, states = prefix.shape[1:]
    width = int(np.max(stops - starts))
    result = np.full((len(permutation), len(starts), width), -np.inf, np.float64)
    for task in prange(len(permutation) * len(starts)):
        pair = task // len(starts)
        index = task % len(starts)
        start, stop = (starts[index], stops[index])
        total = np.zeros(stop - start, np.float64)
        row = np.empty(states, np.float64)
        for sample in range(samples):
            for state in range(states):
                row[state] = prefix[start, sample, permutation[pair, state]]
            for block in range(start, stop):
                for site in range(offsets[block], offsets[block + 1]):
                    switched = np.max(row) - penalty
                    for state in range(states):
                        row[state] = max(row[state], switched) + emissions[sample, site, state]
                best = -np.inf
                for state in range(states):
                    best = max(best, row[state] + suffix[block + 1, sample, permutation[pair, state]])
                total[block - start] += best
        result[pair, index,:stop - start] = total
    return result


def _interval_ranges(blocks, window, groups=None):
    if groups is None:
        starts = np.arange(blocks, dtype=np.int64)
        stops = np.minimum(starts + window, blocks)
    else:
        assert groups[0][0] == 0 and groups[-1][1] == blocks
        assert all((a[1] == b[0] for a, b in zip(groups, groups[1:])))
        starts = np.asarray([a for a, b in groups], np.int64)
        stops = np.asarray(
            [groups[min(len(groups), i + window) - 1][1] for i in range(len(groups))],
            np.int64
        )
    return (starts, stops)


def prepare_intervals(submodels, selected, penalty):
    first, second = np.triu_indices(len(selected))
    arrays = emission_arrays(submodels)
    offsets = np.asarray([0, *np.cumsum([m["bin_emissions"].shape[3] for m in submodels])], np.int64)
    emissions = pack_selected(arrays, selected, offsets, first, second)
    prefix, suffix = flanks(emissions, offsets, float(penalty))
    return emissions, offsets, prefix, suffix


def score_intervals(submodels, selected, pairs, penalty, window, groups=None, prepared=None):
    permutation, _, _ = permutations(len(selected), pairs)
    emissions, offsets, prefix, suffix = (
        prepare_intervals(submodels, selected, penalty) if prepared is None else prepared)
    starts, stops = _interval_ranges(len(offsets) - 1, window, groups)
    values = interval_scores(
        emissions, offsets, prefix, suffix, permutation, float(penalty), starts, stops)
    return values, float(np.max(prefix[-1], axis=1).sum()), starts


def refine_components(prepared, components, neutral, sites, *, config,
                      checkpoints=None, l1_blocks=None, workspaces=None):
    """Refine bounded paired intervals inside unchanged phase components.

    The caller owns the Numba scope. This internal final-release pass takes
    the caller's actual search budgets and never reads truth or pedigree.
    """
    workspaces = {} if workspaces is None else workspaces
    iterations = config.max_iterations
    output = []
    diagnostics = []
    for number, component in enumerate(components):
        token = f'founder_interval_refinement.component{number}'
        cached = fr._load(checkpoints, token)
        if cached is not None:
            output.append(cached['block'])
            diagnostics.append(cached['diagnostic'])
            continue
        batch = [b for b in prepared
                 if b.positions[0] >= component.positions[0]
                 and b.positions[-1] <= component.positions[-1]]
        assert np.array_equal(np.concatenate([b.positions for b in batch]), component.positions)
        selected = fr._local_selection(component, batch)
        original = selected.copy()
        context = fr._macro_context(batch, l1_blocks)
        scales = [('local', None, False)]
        if context is not None:
            groups = context[0]
            backwards = [(len(batch) - b, len(batch) - a) for a, b in groups[::-1]]
            scales.extend([('l1_start', groups, False), ('l1_stop', backwards, True)])
        if len(batch) < 2 or len(selected) < 2:
            output.append(component)
            diagnostics.append(dict(component=number, changed=False))
            continue
        workspace = component_workspace(workspaces, batch, neutral, sites,
            config.proposal_max_bins, get_num_threads(),
            minimum_bin_size=config.proposal_min_sites_per_bin)
        evidence, leaves, offsets = workspace.evidence, workspace.leaves, workspace.offsets
        complete, penalty, logs = workspace.complete, workspace.penalty, workspace.logs
        submodels = workspace.models()

        def evaluate(rows):
            calls = fs.selected_alleles(leaves, offsets, rows)
            values, switches = fs.score_and_switch_count(calls, evidence, complete, penalty, logs)
            return float(values.sum()), int(switches.sum())
        current, switches = evaluate(selected)
        initial = current
        history = []
        for iteration in range(iterations):
            phase = f'{token}.iteration{iteration}'
            cached = fr._load(checkpoints, phase)
            if cached is not None:
                selected, current, switches, history = (
                    cached[k] for k in ('selected', 'score', 'switches', 'history')
                )
                if cached['converged']:
                    break
                continue
            started = time.monotonic()
            pairs = list(zip(*np.triu_indices(len(selected), 1)))
            if len(pairs) > config.interval_partners * len(selected):
                calls = fs.selected_alleles(leaves, offsets, selected)
                value, reference, first, second = ex.score_exchanges(
                    calls,
                    evidence,
                    complete,
                    offsets[1:-1],
                    penalty,
                    logs
                )
                assert np.max(abs(reference - current)) <= max(1e-06, abs(current) * 1e-10)
                ranking = value.max(axis=0)
                retained = set()
                for founder in range(len(selected)):
                    incident = np.flatnonzero((first == founder) | (second == founder))
                    order = np.argsort(-ranking[incident], kind="stable")[:config.interval_partners]
                    retained.update(int(incident[i]) for i in order)
                pairs = [pairs[i] for i in sorted(retained)]
            records = []
            best = None
            seen = set()
            packed = {}
            for scale, groups, reverse in scales:
                models = submodels if not reverse else [
                    dict(m, bin_emissions=np.ascontiguousarray(m['bin_emissions'][:,:,:,::-1])) for m in submodels[::-1]
                ]
                rows = np.ascontiguousarray(selected[:,::-1]) if reverse else selected
                if reverse not in packed:
                    packed[reverse] = prepare_intervals(models, rows, penalty)
                values, binned, starts = score_intervals(models, rows, pairs, penalty,
                    config.window_blocks, groups, packed[reverse])
                for pair, (a, b) in enumerate(pairs):
                    index, length = np.unravel_index(np.argmax(values[pair]), values[pair].shape)
                    start = int(starts[index])
                    stop = start + length + 1
                    predicted = float(values[pair, index, length])
                    if reverse:
                        start, stop = (len(batch) - stop, len(batch) - start)
                    key = (int(a), int(b), int(start), int(stop))
                    if key in seen:
                        continue
                    seen.add(key)
                    trial = selected.copy()
                    trial[[a, b], start:stop] = selected[[b, a], start:stop]
                    if np.array_equal(trial, selected):
                        continue
                    checked = score_rows(submodels, trial, penalty)
                    assert abs(checked - predicted) <= max(1e-05, abs(checked) * 1e-10), (checked, predicted)
                    score, next_switches = evaluate(trial)
                    fit_gain = score - current + penalty * (next_switches - switches)
                    eligible = bool(score > current + 1e-06 and fit_gain >= -1e-06)
                    records.append(
                        dict(pair=[int(a), int(b)], scale=scale, start=int(start), stop=int(stop), binned_gain=predicted - binned, gain=score - current, genotype_fit_gain=fit_gain, switch_delta=next_switches - switches, eligible=eligible)
                    )
                    if eligible and (best is None or score > best[0]):
                        best = (score, next_switches, trial, records[-1])
            record = dict(
                iteration=iteration,
                proposals=records,
                accepted=None,
                seconds=time.monotonic() - started
            )
            if best is not None:
                current, switches, selected, record['accepted'] = best
            history.append(record)
            fr._save(
                checkpoints,
                phase,
                dict(selected=selected, score=current, switches=switches, history=history, converged=best is None)
            )
            if best is None:
                break
        result = component
        if not np.array_equal(selected, original):
            reconstructed = paths.reconstruct_haplotypes_from_beam(
                [(list(row), current) for row in selected],
                fr._LeafKeyMap(batch),
                batch
            )
            result = hierarchy.convert_reconstruction_to_superblock(reconstructed, batch)
            for side in ('before', 'after'):
                for stem in (
                    'missing_aware_break',
                    'missing_aware_break_reason',
                    'missing_aware_joint_informative_samples'
                ):
                    name = f'{stem}_{side}'
                    setattr(
                        result,
                        name,
                        getattr(component, name, False if stem == 'missing_aware_break' else None)
                    )
            assert np.array_equal(
                np.sort(result.discrete_haps, axis=0),
                np.sort(component.discrete_haps, axis=0)
            )
        diagnostic = dict(
            component=number,
            initial_score=initial,
            final_score=current,
            changed=not np.array_equal(selected, original),
            history=history
        )
        fr._save(checkpoints, token, dict(block=result, diagnostic=diagnostic))
        output.append(result)
        diagnostics.append(diagnostic)
    return (
        haplotypes.BlockResults(output),
        dict(model='bounded_staggered_paired_intervals', components=diagnostics)
    )
