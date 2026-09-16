"""Bounded founder windows with exact flanks and complementary branch rankings.

Stable and tie-breaking trajectories use the incumbent suffix as their
primary branch score. A further optimistic trajectory instead prioritizes a
sample-relaxed future: each sample may independently select the focal local
row in each remaining block. This enlarges the feasible set and supplies an
upper bound, not an accepted founder. Returned paths remain cohort-shared and
are rescored against the exact model. Optimistic pruning can cross barriers
that reject every partial prefix against the unchanged incumbent suffix.

With capped local choices, the extra suffix scan is O(bins*N*branch_cap*K²);
its storage is O(window_blocks*N*K²). No truth or pedigree enters the search.
"""
import time

import numpy as np
from numba import njit, prange
from numba.typed import List

from . import founder_path_search as search
from .founder_dual_search import canonical_score


@njit(cache=True, parallel=True)
def relaxed_suffix(emissions, known, choices, penalty, start, stop, terminal,
                   first, second):
    """Backward upper bounds with fixed flanks and sample-specific future rows."""
    samples, states = terminal.shape
    result = np.empty((stop - start + 1, samples, states), np.float64)
    focal = len(known)
    for sample in prange(samples):
        result[-1, sample] = terminal[sample]
        row = np.empty(states, np.float64)
        value = np.empty(states, np.float64)
        for block in range(stop - 1, start - 1, -1):
            position = block - start
            result[position, sample] = -np.inf
            emission = emissions[block]
            for candidate in choices[block]:
                row[:] = result[position + 1, sample]
                # A single local row spans all bins of this prepared block.
                for site in range(emission.shape[3] - 1, -1, -1):
                    for state in range(states):
                        a, b = first[state], second[state]
                        aa = known[a, block] if a < focal else candidate
                        bb = known[b, block] if b < focal else candidate
                        value[state] = emission[sample, aa, bb, site] + row[state]
                    switched = np.max(value) - penalty
                    for state in range(states):
                        row[state] = max(value[state], switched)
                for state in range(states):
                    result[position, sample, state] = max(
                        result[position, sample, state], row[state])
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


def proposals(submodels, known, incumbent, penalty, *, width=64, branch_cap=16,
              window=100, ranking="tie"):
    emissions = List([model["bin_emissions"] for model in submodels])
    blocks = len(emissions)
    first, second = (np.ascontiguousarray(a, np.int64)
                     for a in np.triu_indices(len(known) + 1))
    choices_by_block = local_choices(emissions, incumbent, branch_cap)
    suffix = search._incumbent_suffix(
        emissions, known, incumbent, float(penalty), False, first, second)
    prefix = search._incumbent_suffix(
        emissions, known, incumbent, float(penalty), True, first, second)
    starts = list(range(0, max(1, blocks - window + 1), max(1, window // 2)))
    if starts[-1] + window < blocks:
        starts.append(max(0, blocks - window))
    for start in starts:
        stop = min(blocks, start + window)
        bound = (relaxed_suffix(
            emissions, known, choices_by_block, float(penalty), start, stop,
            np.ascontiguousarray(suffix[stop]), first, second)
            if ranking != "incumbent" else None)
        dp = np.ascontiguousarray(prefix[blocks - start][None, :, :])
        ancestors, local_rows = [], []
        for block in range(start, stop):
            emission = emissions[block]
            choices = choices_by_block[block]
            local_known = np.ascontiguousarray(known[:, block])
            scores = search._score_branches(
                dp, emission, local_known, choices, float(penalty), False,
                suffix[block + 1], first, second)
            if ranking != "incumbent":
                upper = search._score_branches(
                    dp, emission, local_known, choices, float(penalty), False,
                    bound[block - start + 1], first, second)
                index = np.arange(len(scores))
                if ranking == "upper":
                    order = np.lexsort((index, -scores, -upper))[:width]
                else:
                    # Exact score ties only; no numerical tie tolerance.
                    order = np.lexsort((index, -upper, -scores))[:width]
            else:
                order = np.argsort(-scores, kind="stable")[:width]
            ancestors.append(order // len(choices))
            local_rows.append(choices[order % len(choices)])
            dp = search._retain_branches(
                dp, order, emission, local_known, choices, float(penalty), False,
                first, second)
        best = int(np.argmax(scores[order]))
        node, path = best, incumbent.copy()
        for step in range(stop - start - 1, -1, -1):
            path[start + step] = local_rows[step][node]
            node = ancestors[step][node]
        yield path, float(scores[order[best]]), (start, stop)


def solve(submodels, known, incumbent, penalty, *, branch_cap=16, reverse=False,
          width=64, window_blocks=100, ranking="tie"):
    if ranking not in {"incumbent", "tie", "upper"}:
        raise ValueError("unknown founder-window ranking")
    started = time.monotonic()
    known = np.ascontiguousarray(known, np.int64)
    incumbent = np.asarray(incumbent, np.int64)
    initial = canonical_score(submodels, known, incumbent, penalty)
    best, best_score = incumbent.copy(), initial
    models = submodels
    if reverse:
        models = [dict(model, bin_emissions=np.ascontiguousarray(
            model["bin_emissions"][:, :, :, ::-1])) for model in submodels[::-1]]
        known = np.ascontiguousarray(known[:, ::-1])
        incumbent = incumbent[::-1].copy()
    history = []
    for path, score, span in proposals(
            models, known, incumbent, penalty, branch_cap=branch_cap,
            width=width, window=window_blocks, ranking=ranking):
        if score > best_score + 1e-6:
            checked = canonical_score(models, known, path, penalty)
            if abs(checked - score) > 1e-5:
                raise RuntimeError("window flank score disagrees with complete panel")
            best = path[::-1].copy() if reverse else path.copy()
            best_score = checked
            history.append(dict(start=span[0], stop=span[1], score=score))
    return best, best_score, dict(
        algorithm="bounded_exact_flank_windows", ranking=ranking,
        window_blocks=window_blocks, width=width,
        initial_score=initial, best_score=best_score, history=history,
        seconds=time.monotonic() - started)
