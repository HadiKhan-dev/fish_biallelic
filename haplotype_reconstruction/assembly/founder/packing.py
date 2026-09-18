"""Packed numerical inputs for final founder refinement, not a new model.

PreparedModels owns immutable per-block arrays once. Local row ordinals stay in
the original panel order. Packed chosen-panel emissions keep every bin and use
unordered diploid states only for the symmetric uniform-switch Potts model.
"""
from threading import Lock
from collections import OrderedDict

import numpy as np
from numba import njit, prange
from numba.typed import List


class PreparedModels(list):
    def __init__(self, models):
        super().__init__(models)
        self.arrays = List([m["bin_emissions"] for m in models])
        self._packed = None
        self._pack_lock = Lock()
        self._scores = OrderedDict()
        self._score_bytes = 0
        self._alphabets = {}
        self._reversed = None
        self.is_short = all(m["bin_emissions"].shape[3] <= 2 for m in models)
        self.bin_offsets = np.asarray(
            [0, *np.cumsum([m["bin_emissions"].shape[3] for m in models])], np.int64)


def short_models(models):
    return (models.is_short if isinstance(models, PreparedModels) else
            all(m["bin_emissions"].shape[3] <= 2 for m in models))


def _alphabet_template(models, branch_cap):
    choices, offsets, capped = [], [0], []
    for block, model in enumerate(models):
        emission = model["bin_emissions"]
        local = np.arange(emission.shape[1], dtype=np.int64)
        if len(local) > branch_cap:
            rank = emission.max(axis=2).sum(axis=(0, 2))
            local = np.argsort(-rank, kind="stable")[:branch_cap].astype(np.int64)
            capped.append(block)
        choices.extend(local)
        offsets.append(len(choices))
    return np.asarray(choices, np.int64), np.asarray(offsets, np.int64), capped


def candidate_alphabet(models, incumbent, branch_cap):
    """Reuse invariant rankings; retain each query's incumbent in capped blocks.

    Uncapped arrays are shared read-only. Capped arrays are copied before their
    last candidate is replaced, preserving the original stable choice order.
    """
    if isinstance(models, PreparedModels):
        with models._pack_lock:
            if branch_cap not in models._alphabets:
                models._alphabets[branch_cap] = _alphabet_template(models, branch_cap)
            choices, offsets, capped = models._alphabets[branch_cap]
    else:
        choices, offsets, capped = _alphabet_template(models, branch_cap)
    if capped:
        choices = choices.copy()
        for block in capped:
            start, stop = offsets[block:block + 2]
            if incumbent[block] not in choices[start:stop]:
                choices[stop - 1] = incumbent[block]
    return choices, offsets


def reversed_models(models):
    """Share the same immutable reverse-bin view across window queries."""
    def build():
        return PreparedModels([
            dict(model, bin_emissions=np.ascontiguousarray(
                model["bin_emissions"][:,:,:,::-1]))
            for model in models[::-1]
        ])

    if not isinstance(models, PreparedModels):
        return build()
    with models._pack_lock:
        if models._reversed is None:
            models._reversed = build()
        return models._reversed


def emission_arrays(models):
    return models.arrays if isinstance(models, PreparedModels) else List(
        [m["bin_emissions"] for m in models])


def _pack_emissions(emissions):
    arrays = list(emissions)
    counts = np.asarray([a.shape[1] for a in arrays], np.int64)
    bins = np.asarray([a.shape[3] for a in arrays], np.int64)
    offsets = np.asarray([0, *np.cumsum(counts * counts * bins)], np.int64)
    data = np.concatenate([a.reshape(a.shape[0], -1) for a in arrays], axis=1)
    return data, offsets, counts, bins


def packed_emissions(models):
    """Pack immutable emissions once per workspace, shared by its queries.

    Ownership ends with PreparedModels; there is no process-global array cache.
    Plain model lists remain valid for direct numerical callers.
    """
    if not isinstance(models, PreparedModels):
        return _pack_emissions(emission_arrays(models))
    with models._pack_lock:
        if models._packed is None:
            models._packed = _pack_emissions(models.arrays)
        return models._packed


@njit(cache=True, nogil=True)
def selected_emission_addresses(offsets, counts, bins, selected, first, second):
    out = np.empty((len(counts), len(first)), np.int64)
    for block in range(len(counts)):
        for state in range(len(first)):
            out[block, state] = offsets[block] + (selected[first[state], block] * counts[block] + selected[second[state], block]) * bins[block]
    return out


@njit(cache=True, parallel=True, nogil=True)
def score_selected(data, offsets, counts, bins, selected, penalty):
    founders = len(selected)
    states = founders * (founders + 1) // 2
    first = np.empty(states, np.int64)
    second = np.empty(states, np.int64)
    state = 0
    for a in range(founders):
        for b in range(a, founders):
            first[state] = a
            second[state] = b
            state += 1
    index = selected_emission_addresses(offsets, counts, bins, selected, first, second)
    answer = np.empty(len(data))
    for sample in prange(len(data)):
        row = np.empty(states)
        initialized = False
        for block in range(len(counts)):
            for marker in range(bins[block]):
                switched = np.max(row) - penalty if initialized else 0.0
                for state in range(states):
                    value = data[sample, index[block, state] + marker]
                    row[state] = max(row[state], switched) + value if initialized else value
                initialized = True
        answer[sample] = np.max(row) if initialized else 0.0
    return answer


def score_rows(models, selected, penalty):
    """Cache exact ordered panel scores within the immutable model workspace."""
    selected = np.ascontiguousarray(selected, np.int64)
    penalty = float(penalty)
    if not isinstance(models, PreparedModels):
        return float(score_selected(*packed_emissions(models), selected, penalty).sum())
    key = (selected.shape, penalty, selected.tobytes())
    with models._pack_lock:
        if key in models._scores:
            models._scores.move_to_end(key)
            return models._scores[key]
    value = float(score_selected(*packed_emissions(models), selected, penalty).sum())
    # Small memory-only caps, not proposal/search budgets. Duplicate concurrent
    # evaluations are harmless; never serialize their numerical kernels.
    limit = 8 * 1024 ** 2
    if len(key[-1]) <= limit:
        with models._pack_lock:
            if key not in models._scores:
                models._score_bytes += len(key[-1])
            models._scores[key] = value
            while len(models._scores) > 64 or models._score_bytes > limit:
                expired, _ = models._scores.popitem(last=False)
                models._score_bytes -= len(expired[-1])
    return value


@njit(cache=True, parallel=True, nogil=True)
def pack_selected(emissions, selected, offsets, first, second):
    samples = emissions[0].shape[0]
    result = np.empty((samples, offsets[-1], len(first)))
    for task in prange(samples * len(emissions)):
        sample, block = task // len(emissions), task % len(emissions)
        values = emissions[block]
        for site in range(values.shape[3]):
            for state in range(len(first)):
                result[sample, offsets[block] + site, state] = values[
                    sample, selected[first[state], block], selected[second[state], block], site]
    return result


@njit(cache=True, parallel=True, nogil=True)
def gather_macro(emissions, groups, rows_by_group):
    """One parallel launch for all groups; fetch each source block once/task."""
    outputs = List()
    for group in range(len(groups)):
        start, stop = groups[group]
        bins = 0
        for block in range(start, stop):
            bins += emissions[block].shape[3]
        count = len(rows_by_group[group])
        outputs.append(np.empty((emissions[0].shape[0], count, count, bins)))
    samples = emissions[0].shape[0]
    for task in prange(len(groups) * samples):
        group, sample = task // samples, task % samples
        rows, output = rows_by_group[group], outputs[group]
        start, stop = groups[group]
        offset = 0
        for block in range(start, stop):
            value = emissions[block]
            for a in range(len(rows)):
                aa = rows[a, block - start]
                for b in range(len(rows)):
                    bb = rows[b, block - start]
                    for site in range(value.shape[3]):
                        output[sample, a, b, offset + site] = value[sample, aa, bb, site]
            offset += value.shape[3]
    return outputs
