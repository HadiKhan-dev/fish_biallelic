"""Bounded conditional founder paths retaining chromosome-wide sample states.

All branches are scored against the incumbent suffix. Only retained branches
materialize their sample-state arrays; discarded branches use thread-local
workspace. Search breadth, cohort scoring and stable tie ordering are unchanged.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange, set_num_threads
from numba.typed import List
from.packing import PreparedModels, emission_arrays, gather_macro, packed_emissions, selected_emission_addresses


@njit(cache=True, parallel=True, nogil=True)
def _packed_incumbent_suffix(data, offsets, counts, bins, known, incumbent, penalty, reverse, first, second):
    selected = np.empty((len(known) + 1, len(incumbent)), np.int64)
    selected[:-1] = known
    selected[-1] = incumbent
    index = selected_emission_addresses(offsets, counts, bins, selected, first, second)
    blocks, samples, states = (len(counts), len(data), len(first))
    result = np.empty((blocks + 1, samples, states))
    for sample in prange(samples):
        row = np.zeros(states)
        value = np.empty(states)
        result[blocks, sample] = row
        for step in range(blocks - 1, -1, -1):
            block = blocks - 1 - step if reverse else step
            size = bins[block]
            for offset in range(size):
                site = offset if reverse else size - 1 - offset
                for state in range(states):
                    value[state] = data[sample, index[block, state] + site] + row[state]
                switched = np.max(value) - penalty
                for state in range(states):
                    row[state] = max(value[state], switched)
            result[step, sample] = row
    return result


def _incumbent_suffix(models, known, incumbent, penalty, reverse, first, second):
    return _packed_incumbent_suffix(*packed_emissions(models),
        known, incumbent, penalty, reverse, first, second)


def conditional_path(submodels, known, incumbent, penalty, *, width=64,
                     branch_cap=16, reverse=False, thread_budget=None):
    """Bounded beam; packed short blocks and ordinary compiled macro states."""
    from.beam import conditional_path as solve
    return solve(submodels, known, incumbent, penalty, width=width,
                 branch_cap=branch_cap, reverse=reverse, thread_budget=thread_budget)


@njit(cache=True, parallel=True)
def count_diplotype_switches(painting):
    """Count the transitions penalized by the internal unordered-state HMM."""
    changes = 0
    for sample in prange(painting.shape[0]):
        for site in range(1, painting.shape[1]):
            changes += painting[sample, site] != painting[sample, site - 1]
    return changes


def coarsen_submodels(submodels, selected, groups, context_paths):
    """Offer short assembled paths as single moves, retaining all incumbents.

    Grouping changes the proposal alphabet only: emissions, bin boundaries,
    sample-state transitions and the full-site acceptance model are unchanged.
    The local alphabet is the exact union of context rows and current paths;
    the conditional beam retains its existing bounded branch quota.
    """
    alphabets = []
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
        alphabets.append(rows)
    outputs = gather_macro(emission_arrays(submodels),
                           np.asarray(groups, np.int64), List(alphabets))
    models = PreparedModels([{
        "bin_emissions": values, "hap_keys": list(range(len(rows))),
        "n_bins": values.shape[3],
    } for values, rows in zip(outputs, alphabets)])
    return models, alphabets, macro_selected


def expand_macro_path(path, alphabets, groups):
    """Decode a proposed macro path back to original prepared-row indices."""
    result = np.empty(groups[-1][1], np.int64)
    for choice, rows, (start, end) in zip(path, alphabets, groups):
        result[start:end] = rows[choice]
    return result
