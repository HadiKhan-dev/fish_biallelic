"""Bounded conditional founder paths retaining chromosome-wide sample states.

All branches are scored against the incumbent suffix. Only retained branches
materialize their sample-state arrays; discarded branches use thread-local
workspace. Search breadth, cohort scoring and stable tie ordering are unchanged.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numba.typed import List


@njit(inline="always")
def _advance_row(previous, row, emission, sample, known, local_row,
                 penalty, reverse, first, second):
    founders = len(known) + 1
    row[:] = previous
    bins = emission.shape[3]
    for step in range(bins):
        site = bins - 1 - step if reverse else step
        switched = np.max(row) - penalty
        for state in range(len(first)):
            a, b = first[state], second[state]
            aa = known[a] if a < founders - 1 else local_row
            bb = known[b] if b < founders - 1 else local_row
            row[state] = max(row[state], switched) + emission[sample, aa, bb, site]


@njit(parallel=True, cache=True)
def _score_branches(dp, emission, known, choices, penalty, reverse,
                    suffix, first, second):
    beams, samples, states = dp.shape
    branches = len(choices)
    individual = np.empty((beams * branches, samples), np.float64)
    for candidate in prange(beams * branches):
        prior = candidate // branches
        local_row = choices[candidate % branches]
        row = np.empty(states, np.float64)
        for sample in range(samples):
            _advance_row(dp[prior, sample], row, emission, sample, known,
                         local_row, penalty, reverse, first, second)
            value = -np.inf
            for state in range(states):
                value = max(value, row[state] + suffix[sample, state])
            individual[candidate, sample] = value
    # Same ascending per-sample reduction as the original branch scorer.
    return individual.sum(axis=1)


@njit(parallel=True, cache=True)
def _retain_branches(dp, order, emission, known, choices, penalty, reverse,
                     first, second):
    samples, states = dp.shape[1:]
    branches = len(choices)
    updated = np.empty((len(order), samples, states), np.float64)
    for task in prange(len(order) * samples):
        retained, sample = task // samples, task % samples
        candidate = order[retained]
        _advance_row(dp[candidate // branches, sample], updated[retained, sample],
                     emission, sample, known, choices[candidate % branches],
                     penalty, reverse, first, second)
    return updated


@njit(parallel=True, cache=True)
def _incumbent_suffix(emissions, known, incumbent, penalty, reverse, first, second):
    blocks = len(emissions)
    samples = emissions[0].shape[0]
    states = len(first)
    founders = len(known) + 1
    result = np.empty((blocks + 1, samples, states), np.float64)
    # Samples are independent across the entire backward scan. Avoid launching
    # a new parallel region at every original 200-SNP block.
    for sample in prange(samples):
        row = np.zeros(states)
        value = np.empty(states)
        result[blocks, sample] = row
        for step in range(blocks - 1, -1, -1):
            block = blocks - 1 - step if reverse else step
            emission = emissions[block]
            for offset in range(emission.shape[3]):
                site = offset if reverse else emission.shape[3] - 1 - offset
                for state in range(states):
                    a, b = first[state], second[state]
                    aa = known[a, block] if a < founders - 1 else incumbent[block]
                    bb = known[b, block] if b < founders - 1 else incumbent[block]
                    value[state] = emission[sample, aa, bb, site] + row[state]
                switched = np.max(value) - penalty
                for state in range(states):
                    row[state] = max(value[state], switched)
            result[step, sample] = row
    return result


def conditional_path(submodels, known, incumbent, penalty, *, width=64,
                     branch_cap=16, reverse=False):
    """Return the same bounded-beam proposal in O(bins*N*width*branch_cap*K²).

    Branch scores use O(width*branch_cap*N) storage; retained forward states
    use O(width*N*K²). Recomputing only selected branches avoids the larger
    O(width*branch_cap*N*K²) candidate-state tensor and its serial gather.
    """
    emissions = List([entry["bin_emissions"] for entry in submodels])
    samples = emissions[0].shape[0]
    blocks = len(submodels)
    founders = len(known) + 1
    first, second = (np.ascontiguousarray(x, np.int64)
                     for x in np.triu_indices(founders))
    dp = np.zeros((1, samples, len(first)), np.float64)
    ancestry, local_rows = [], []
    order_blocks = list(range(blocks - 1, -1, -1) if reverse else range(blocks))
    suffix = _incumbent_suffix(
        emissions, known, incumbent, float(penalty), reverse, first, second)
    for step, block in enumerate(order_blocks):
        emission = emissions[block]
        choices = np.arange(emission.shape[1], dtype=np.int64)
        if len(choices) > branch_cap:
            rank = emission.max(axis=2).sum(axis=(0, 2))
            choices = np.argsort(-rank, kind="stable")[:branch_cap].astype(np.int64)
            if incumbent[block] not in choices:
                choices[-1] = incumbent[block]
        local_known = np.ascontiguousarray(known[:, block])
        scores = _score_branches(
            dp, emission, local_known, choices, float(penalty), reverse,
            suffix[step + 1], first, second)
        order = np.argsort(-scores, kind="stable")[:width]
        ancestry.append(order // len(choices))
        local_rows.append(choices[order % len(choices)])
        dp = _retain_branches(
            dp, order, emission, local_known, choices, float(penalty),
            reverse, first, second)
    path = np.empty(blocks, np.int64)
    node = 0
    for step in range(blocks - 1, -1, -1):
        path[order_blocks[step]] = local_rows[step][node]
        node = ancestry[step][node]
    return path, float(scores[order[0]])

@njit(cache=True, parallel=True)
def count_diplotype_switches(painting):
    """Count the transitions penalized by the internal unordered-state HMM."""
    changes = 0
    for sample in prange(painting.shape[0]):
        for site in range(1, painting.shape[1]):
            changes += painting[sample, site] != painting[sample, site - 1]
    return changes


@njit(cache=True, parallel=True)
def _gather_macro_emissions(emissions, rows):
    """Keep every original bin and its candidate-independent evidence mask."""
    samples = emissions[0].shape[0]
    count = len(rows)
    bins = 0
    for emission in emissions:
        bins += emission.shape[3]
    result = np.empty((samples, count, count, bins), np.float64)
    for task in prange(samples * count * count):
        sample = task // (count * count)
        first = (task // count) % count
        second = task % count
        offset = 0
        for block in range(len(emissions)):
            value = emissions[block]
            for site in range(value.shape[3]):
                result[sample, first, second, offset + site] = value[
                    sample, rows[first, block], rows[second, block], site]
            offset += value.shape[3]
    return result


def coarsen_submodels(submodels, selected, groups, context_paths):
    """Offer short assembled paths as single moves, retaining all incumbents.

    Grouping changes the proposal alphabet only: emissions, bin boundaries,
    sample-state transitions and the full-site acceptance model are unchanged.
    The local alphabet is the exact union of context rows and current paths;
    the conditional beam retains its existing bounded branch quota.
    """
    models, alphabets = [], []
    macro_selected = np.empty((len(selected), len(groups)), np.int64)
    for number, ((start, end), context) in enumerate(zip(groups, context_paths)):
        alphabet, lookup = [], {}
        for row in [*selected[:, start:end], *context]:
            key = tuple(map(int, row))
            if key not in lookup:
                lookup[key] = len(alphabet)
                alphabet.append(row.copy())
        rows = np.asarray(alphabet, np.int64)
        macro_selected[:, number] = [
            lookup[tuple(map(int, row))] for row in selected[:, start:end]]
        values = _gather_macro_emissions(
            List([entry["bin_emissions"] for entry in submodels[start:end]]),
            rows)
        models.append({"bin_emissions": values,
                       "hap_keys": list(range(len(rows))),
                       "n_bins": values.shape[3]})
        alphabets.append(rows)
    return models, alphabets, macro_selected


def expand_macro_path(path, alphabets, groups):
    """Decode a proposed macro path back to original prepared-row indices."""
    result = np.empty(groups[-1][1], np.int64)
    for choice, rows, (start, end) in zip(path, alphabets, groups):
        result[start:end] = rows[choice]
    return result
