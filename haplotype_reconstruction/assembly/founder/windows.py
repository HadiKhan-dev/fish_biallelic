"""Bounded founder windows with exact flanks and complementary branch rankings.

Stable and tie-breaking trajectories use the incumbent suffix as their
primary branch score. A further optimistic trajectory instead prioritizes a
sample-relaxed future: each sample may independently select the focal local
row in each remaining block. This enlarges the feasible set and supplies an
upper bound, not an accepted founder. Returned paths remain cohort-shared and
are rescored against the exact model. Optimistic pruning can cross barriers
that reject every partial prefix against the unchanged incumbent suffix.

For t bins per block, the shared-background suffix costs O(N*(K²*t² +
branch_cap*(K*t+t²))); its messages use O(window_blocks*N*K²) storage.
No truth or pedigree enters the search.
"""
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
from numba.typed import List
from.import path_search as search, sparse as founder_sparse, dual_short as founder_dual_short
from.import beam_kernels as kernels
from.beam import workspace, macro_proposals
from.candidates import completed_candidates
from.dual_search import canonical_score
from.packing import emission_arrays, packed_emissions, candidate_alphabet, short_models, reversed_models

@njit(cache=True, parallel=True, nogil=True)
def relaxed_suffix(emissions, known, choices, penalty, start, stop, terminal,
                   first, second, prepared=None, mapping=None, focal_index=0):
    samples, states = terminal.shape
    result = np.empty((stop - start + 1, samples, states))
    for sample in prange(samples):
        result[-1, sample] = terminal[sample]
        for block in range(stop - 1, start - 1, -1):
            position = block - start
            if prepared is None:
                result[position, sample] = founder_sparse.backward_sample(
                    result[position + 1, sample], emissions[block][sample], known[:, block],
                    choices[block], np.zeros(len(choices[block])), penalty, True, first, second)
            else:
                result[position, sample] = founder_sparse.backward_sample(
                    result[position + 1, sample], emissions[block][sample], known[:, block],
                    choices[block], np.zeros(len(choices[block])), penalty, True, first, second,
                    prepared[block], mapping, focal_index, sample)
    return result


def local_choices(emissions, incumbent, branch_cap):
    """Use the existing capped local alphabet, always retaining the incumbent."""
    result = List()
    for block, emission in enumerate(emissions):
        choices = np.arange(emission.shape[1], dtype=np.int64)
        if len(choices) > branch_cap:
            rank = emission.max(axis=2).sum(axis=(0, 2))
            choices = np.argsort(-rank, kind="stable")[:branch_cap].astype(np.int64)
            if incumbent[block] not in choices:
                choices[-1] = incumbent[block]
        result.append(choices)
    return result


def excluded(upper, best):
    return upper + 1e-10 * max(1.0, abs(upper), abs(best)) <= best + 1e-06

def proposals(
    models,
    known,
    incumbent,
    penalty,
    *,
    width=64,
    branch_cap=16,
    window=100,
    ranking='tie',
    thread_budget=None,
    statistics=None,
    background=None
):
    emissions = emission_arrays(models)
    if not short_models(models):
        yield from macro_proposals(
            models,
            known,
            incumbent,
            penalty,
            width=width,
            branch_cap=branch_cap,
            window=window,
            ranking=ranking,
            thread_budget=thread_budget,
            statistics=statistics,
            background=background
        )
        return
    known = np.ascontiguousarray(known, np.int64)
    blocks = len(models)
    choices, offsets = candidate_alphabet(models, incumbent, branch_cap)
    first, second = (np.ascontiguousarray(x, np.int64) for x in np.triu_indices(len(known) + 1))
    packed = packed_emissions(models)
    suffix = search._incumbent_suffix(models, known, incumbent, float(penalty), False, first, second)
    prefix = search._incumbent_suffix(models, known, incumbent, float(penalty), True, first, second)
    starts = list(range(0, max(1, blocks - window + 1), max(1, window // 2)))
    if starts[-1] + window < blocks:
        starts.append(max(0, blocks - window))
    initial = float(np.max(prefix[-1], axis=1).sum())
    rank = {'incumbent': 0, 'tie': 1, 'upper': 2}[ranking]
    max_choices = int(np.max(np.diff(offsets)))

    def task(start, budget):
        stop = min(blocks, start + window)
        bound = founder_dual_short.window_suffix(
            *packed,
            known,
            choices,
            offsets,
            float(penalty),
            start,
            stop,
            np.ascontiguousarray(suffix[stop]),
            first,
            second
        )
        optimistic = float(np.max(prefix[blocks - start] + bound[0], axis=1).sum())
        if excluded(optimistic, initial):
            return dict(start=start, stop=stop, optimistic=optimistic, pruned=True)
        dp, alternate, values, uppers, ancestry, rows = workspace(
            blocks,
            len(packed[0]),
            len(first),
            width,
            max_choices
        )
        dp[0] = prefix[blocks - start]
        beams = 1
        aborted = False
        trace_upper = np.full(blocks, np.inf)
        trace_equivalent = np.ones(blocks, bool)
        for begin in range(start, stop, 32):
            if budget is not None:
                set_num_threads(budget())
            dp, alternate, beams, scores, order, aborted, equivalent = kernels.chunk(
                *packed,
                known,
                choices,
                offsets,
                float(penalty),
                False,
                first,
                second,
                suffix,
                bound,
                True,
                start,
                initial,
                rank,
                begin,
                min(stop, begin + 32),
                dp,
                alternate,
                beams,
                width,
                values,
                uppers,
                ancestry,
                rows,
                trace_upper,
                trace_equivalent
            )
            if aborted:
                break
        record = dict(
            start=start,
            stop=stop,
            optimistic=optimistic,
            pruned=False,
            upper=trace_upper[start:stop],
            equivalent=trace_equivalent[start:stop],
            aborted=aborted
        )
        if not aborted:
            best = int(np.argmax(scores[order]))
            node = best
            path = incumbent.copy()
            for block in range(stop - 1, start - 1, -1):
                path[block] = rows[block, node]
                node = ancestry[block, node]
            record.update(path=path, score=float(scores[order[best]]))
        return record
    functions = [lambda budget, start=start: task(start, budget) for start in starts]
    size = 8 * len(packed[0]) * (2 * width * (len(first) + max_choices) + (window + 1) * len(first)) + 16 * blocks * width
    records = {i: r for i, r in completed_candidates(functions, thread_budget or get_num_threads(), size)}
    best_seen = initial
    for index in range(len(starts)):
        record = records[index]
        start, stop = (record['start'], record['stop'])
        if excluded(record['optimistic'], best_seen):
            if statistics is not None:
                statistics['pruned_windows'] += 1
            continue
        assert not record['pruned']
        aborted = False
        for value, equivalent in zip(record['upper'], record['equivalent']):
            if excluded(value, best_seen):
                aborted = True
                if statistics is not None:
                    statistics['aborted_windows'] += 1
                break
            if statistics is not None and (not equivalent):
                statistics['equivalent_incumbent_tie'] = False
        if aborted:
            continue
        assert not record['aborted']
        best_seen = max(best_seen, record['score'])
        yield (record['path'], record['score'], (start, stop))


def solve(submodels, known, incumbent, penalty, *, branch_cap=16, reverse=False,
          width=64, window_blocks=100, ranking="tie", thread_budget=None, background=None):
    if ranking not in {"incumbent", "tie", "upper"}:
        raise ValueError("unknown founder-window ranking")
    started = time.monotonic()
    known = np.ascontiguousarray(known, np.int64)
    incumbent = np.asarray(incumbent, np.int64)
    initial = canonical_score(submodels, known, incumbent, penalty)
    best, best_score = incumbent.copy(), initial
    models = submodels
    if reverse:
        models = reversed_models(submodels)
        known = np.ascontiguousarray(known[:,::-1])
        incumbent = incumbent[::-1].copy()
        if background is not None:
            background = background.reversed()
    history = []
    statistics = dict(pruned_windows=0, aborted_windows=0, equivalent_incumbent_tie=True)
    for path, score, span in proposals(
            models, known, incumbent, penalty, branch_cap=branch_cap,
            width=width, window=window_blocks, ranking=ranking, thread_budget=thread_budget,
            statistics=statistics, background=background):
        if score > best_score + 1e-6:
            best = path[::-1].copy() if reverse else path.copy()
            best_score = score
            history.append(dict(start=span[0], stop=span[1], score=score))
    if history:
        # Exact flank messages already score each complete candidate panel.
        # Validate the winner once, not the entire chromosome after every
        # improving window: that extra scan would be quadratic in block count.
        # Selection now uses one consistent evaluation order for flank scores;
        # the caller still accepts only a canonical full-site improvement.
        checked = canonical_score(models, known,
                                  best[::-1].copy() if reverse else best, penalty)
        if abs(checked - best_score) > 1e-5:
            raise RuntimeError("window flank score disagrees with complete panel")
        best_score = checked
    return best, best_score, dict(
        algorithm="bounded_exact_flank_windows", ranking=ranking, **statistics,
        window_blocks=window_blocks, width=width,
        initial_score=initial, best_score=best_score, history=history,
        seconds=time.monotonic() - started)
