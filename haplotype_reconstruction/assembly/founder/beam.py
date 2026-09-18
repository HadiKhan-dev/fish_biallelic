"""Bounded founder beams with packed short-block and compiled macro scans.

Branch alphabets, beam width, ordinary state values, stable tie ordering and
full-site acceptance remain unchanged. Independent windows are orchestrated by founder_windows.
"""
import numpy as np
from numba import set_num_threads
from.import path_search as search, sparse as founder_sparse, beam_kernels as kernels
from.packing import emission_arrays, packed_emissions, candidate_alphabet, short_models

def workspace(blocks, samples, states, width, max_choices):
    dp = np.empty((width, samples, states))
    dp[0] = 0.0
    alternate = np.empty_like(dp)
    values = np.empty((samples, width * max_choices))
    uppers = np.empty_like(values)
    ancestry = np.empty((blocks, width), np.int64)
    rows = np.empty_like(ancestry)
    return (dp, alternate, values, uppers, ancestry, rows)

def conditional_path(
    submodels,
    known,
    incumbent,
    penalty,
    *,
    width=64,
    branch_cap=16,
    reverse=False,
    thread_budget=None
):
    emissions = emission_arrays(submodels)
    if not short_models(submodels):
        return _macro_conditional(
            submodels,
            known,
            incumbent,
            penalty,
            width=width,
            branch_cap=branch_cap,
            reverse=reverse,
            thread_budget=thread_budget
        )
    known = np.ascontiguousarray(known, np.int64)
    choices, offsets = candidate_alphabet(submodels, incumbent, branch_cap)
    first, second = (np.ascontiguousarray(x, np.int64) for x in np.triu_indices(len(known) + 1))
    suffix = search._incumbent_suffix(submodels, known, incumbent, float(penalty), reverse, first, second)
    packed = packed_emissions(submodels)
    blocks = len(emissions)
    dp, alternate, values, uppers, ancestry, rows = workspace(
        blocks,
        len(packed[0]),
        len(first),
        width,
        int(np.max(np.diff(offsets)))
    )
    beams = 1
    trace_upper = np.empty(blocks)
    trace_equivalent = np.ones(blocks, bool)
    for start in range(0, blocks, 32):
        if thread_budget is not None:
            set_num_threads(thread_budget())
        dp, alternate, beams, scores, order, aborted, _ = kernels.chunk(
            *packed,
            known,
            choices,
            offsets,
            float(penalty),
            reverse,
            first,
            second,
            suffix,
            suffix,
            False,
            0,
            -np.inf,
            0,
            start,
            min(blocks, start + 32),
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
        assert not aborted
    path = np.empty(blocks, np.int64)
    node = 0
    for step in range(blocks - 1, -1, -1):
        block = blocks - 1 - step if reverse else step
        path[block] = rows[step, node]
        node = ancestry[step, node]
    return (path, float(scores[order[0]]))

def setup(models, known, incumbent, cap, width):
    emissions = emission_arrays(models)
    choices, offsets = candidate_alphabet(models, incumbent, cap)
    first, second = (np.ascontiguousarray(x, np.int64) for x in np.triu_indices(len(known) + 1))
    packed = packed_emissions(models)
    dp, alternate, _, _, ancestry, rows = workspace(
        len(models),
        len(packed[0]),
        len(first),
        width,
        int(np.max(np.diff(offsets)))
    )
    return (emissions, choices, offsets, first, second, packed, dp, alternate, ancestry, rows)

def _macro_conditional(
    models,
    known,
    incumbent,
    penalty,
    *,
    width=64,
    branch_cap=16,
    reverse=False,
    thread_budget=None
):
    known = np.ascontiguousarray(known, np.int64)
    emissions, choices, offsets, first, second, packed, dp, alternate, ancestry, rows = setup(
        models,
        known,
        incumbent,
        branch_cap,
        width
    )
    suffix = search._incumbent_suffix(models, known, incumbent, float(penalty), reverse, first, second)
    beams = 1
    blocks = len(models)
    for start in range(0, blocks, 32):
        if thread_budget is not None:
            set_num_threads(thread_budget())
        dp, alternate, beams, scores, order, aborted, _ = kernels.macro_chunk(
            emissions,
            *packed,
            known,
            choices,
            offsets,
            float(penalty),
            reverse,
            first,
            second,
            suffix,
            suffix,
            False,
            0,
            -np.inf,
            0,
            start,
            min(blocks, start + 32),
            dp,
            alternate,
            beams,
            width,
            ancestry,
            rows
        )
        assert not aborted
    node = 0
    path = np.empty(blocks, np.int64)
    for step in range(blocks - 1, -1, -1):
        block = blocks - 1 - step if reverse else step
        path[block] = rows[step, node]
        node = ancestry[step, node]
    return (path, float(scores[order[0]]))

def macro_proposals(
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
    known = np.ascontiguousarray(known, np.int64)
    blocks = len(models)
    emissions, choices, offsets, first, second, packed, dp, alternate, ancestry, rows = setup(
        models,
        known,
        incumbent,
        branch_cap,
        width
    )
    from.import windows as windows
    local = windows.local_choices(emissions, incumbent, branch_cap)
    backward = None if background is None else background.get(True, True)
    forward = None if background is None else background.get(False, False)
    bgmap = None if background is None else background.mapping
    focal = 0 if background is None else background.focal
    suffix = search._incumbent_suffix(models, known, incumbent, float(penalty), False, first, second)
    prefix = search._incumbent_suffix(models, known, incumbent, float(penalty), True, first, second)
    starts = list(range(0, max(1, blocks - window + 1), max(1, window // 2)))
    if starts[-1] + window < blocks:
        starts.append(max(0, blocks - window))
    best_seen = float(np.max(prefix[-1], axis=1).sum())
    rank = {'incumbent': 0, 'tie': 1, 'upper': 2}[ranking]
    for start in starts:
        stop = min(blocks, start + window)
        bound = windows.relaxed_suffix(
            emissions,
            known,
            local,
            float(penalty),
            start,
            stop,
            np.ascontiguousarray(suffix[stop]),
            first,
            second,
            backward,
            bgmap,
            focal
        )
        dp[0] = prefix[blocks - start]
        optimistic = float(np.max(dp[0] + bound[0], axis=1).sum())
        margin = 1e-10 * max(1.0, abs(optimistic), abs(best_seen))
        if optimistic + margin <= best_seen + 1e-06:
            if statistics is not None:
                statistics['pruned_windows'] += 1
            continue
        beams = 1
        aborted = False
        for begin in range(start, stop, 32):
            if thread_budget is not None:
                set_num_threads(thread_budget())
            dp, alternate, beams, scores, order, aborted, equivalent = kernels.macro_chunk(
                emissions,
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
                best_seen,
                rank,
                begin,
                min(stop, begin + 32),
                dp,
                alternate,
                beams,
                width,
                ancestry,
                rows,
                forward,
                bgmap,
                focal
            )
            if statistics is not None and (not equivalent):
                statistics['equivalent_incumbent_tie'] = False
            if aborted:
                if statistics is not None:
                    statistics['aborted_windows'] += 1
                break
        if aborted:
            continue
        best = int(np.argmax(scores[order]))
        node = best
        path = incumbent.copy()
        for block in range(stop - 1, start - 1, -1):
            path[block] = rows[block, node]
            node = ancestry[block, node]
        value = float(scores[order[best]])
        best_seen = max(best_seen, value)
        yield (path, value, (start, stop))
