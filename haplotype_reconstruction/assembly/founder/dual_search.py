"""Dual-decomposition kernels of the window programmes' screen (window_dp).

Each sample has a relaxed copy of the focal founder's block choices. Zero-sum
messages enforce consensus by equalizing max-marginals across sample chains:
suffix_bound gives the relaxed bound, coordinate_sweep decodes a single common
path (keeping the incumbent's choice on ties) and residual_correction removes the
messages' residual from the bound. dual_short holds the one/two-bin variants.

The dual bound applies only to this restricted, binned conditional problem.
It is neither a global assembly certificate nor a biological confidence score.
"""
import numpy as np
from numba import njit, prange
from . import sparse as founder_sparse


@njit(cache=True, parallel=True, nogil=True)
def suffix_bound(emissions, known, choices, offsets, messages, penalty,
                 reverse, first, second, prepared=None, mapping=None, focal_index=0):
    blocks, samples, states = len(emissions), messages.shape[0], len(first)
    result = np.zeros((blocks + 1, samples, states))
    scales = np.zeros(samples)
    for sample in prange(samples):
        for step in range(blocks - 1, -1, -1):
            block = blocks - 1 - step if reverse else step
            begin, end = offsets[block], offsets[block + 1]
            if prepared is None:
                row = founder_sparse.backward_sample(result[step + 1, sample],
                    emissions[block][sample], known[:, block], choices[begin:end],
                    messages[sample, begin:end], penalty, not reverse, first, second)
            else:
                row = founder_sparse.backward_sample(result[step + 1, sample],
                    emissions[block][sample], known[:, block], choices[begin:end],
                    messages[sample, begin:end], penalty, not reverse, first, second,
                    prepared[block], mapping, focal_index, sample)
            normalizer = np.max(row)
            result[step, sample] = row - normalizer
            scales[sample] += normalizer
    # The total in sample order by one thread: scales.sum() here would be a parallel
    # reduction whose association depends on the Numba team size.
    total = 0.0
    for sample in range(samples):
        total += scales[sample]
    return result, total


@njit(cache=True, nogil=True)
def coordinate_sweep(emissions, known, incumbent, choices, offsets, messages,
                     suffix, penalty, reverse, first, second, prepared=None, mapping=None, focal_index=0):
    blocks, samples, states = len(emissions), messages.shape[0], len(first)
    prefix = np.zeros((samples, states))
    scales = np.zeros(samples)
    decoded = np.empty(blocks, np.int64)
    predicted_change = 0.0
    normalizers = np.empty(samples)
    for step in range(blocks):
        block = blocks - 1 - step if reverse else step
        start, stop = offsets[block], offsets[block + 1]
        count = stop - start
        local_known = np.ascontiguousarray(known[:, block])
        if prepared is None:
            rows, entries, marginals = founder_sparse.forward_details(prefix,
                emissions[block], local_known, choices[start:stop], penalty,
                reverse, first, second, suffix[step + 1], weights=messages[:, start:stop])
        else:
            rows, entries, marginals = founder_sparse.forward_details(prefix,
                emissions[block], local_known, choices[start:stop], penalty,
                reverse, first, second, suffix[step + 1], prepared[block], mapping, focal_index,
                messages[:, start:stop])
        common = np.zeros(count)
        for local in range(count):
            for sample in range(samples):
                common[local] += marginals[sample, local]
            common[local] /= samples
        best = np.argmax(common)
        for local in range(count):
            if choices[start + local] == incumbent[block] and common[local] == common[best]:
                best = local
        decoded[block] = choices[start + best]
        predicted_change += samples * common[best]
        if prepared is None:
            prefix = founder_sparse.combine_forward(prefix, emissions[block], local_known,
                reverse, first, second, rows, entries, messages[:, start:stop],
                marginals=marginals, common=common, normalizers=normalizers)
        else:
            prefix = founder_sparse.combine_forward(prefix, emissions[block], local_known,
                reverse, first, second, rows, entries, messages[:, start:stop],
                prepared[block], mapping, focal_index, marginals, common, normalizers)
        for sample in range(samples):
            scales[sample] += normalizers[sample]
    return decoded, scales.sum(), predicted_change


@njit(cache=True)
def residual_correction(residual, offsets):
    result = 0.0
    for block in range(len(offsets) - 1):
        result += np.min(residual[offsets[block]:offsets[block + 1]])
    return result

