"""Binned primary evidence for proposal ranking, never final acceptance.

Every usable site contributes its existing log likelihood. Restricting sample
switches to bin boundaries gives a lower bound, not a safe rejection bound.
"""
import math
import numpy as np
from numba import njit, prange
from numba.typed import List
from .packing import (
    PreparedModels, packed_emissions, score_rows, selected_emission_addresses)
from .scoring import _unordered_pairs


@njit(cache=True, parallel=True, nogil=True)
def fill_primary(logs, leaves, offsets, bin_size, output):
    samples = logs.shape[0]
    center = math.log(1./3.)
    for task in prange(samples * len(leaves)):
        block, sample = task // samples, task % samples
        local = leaves[block]
        values = output[block]
        for position in range(local.shape[1]):
            site = offsets[block]+position
            a,b,c = logs[sample,site]
            if a == 0. and b == 0. and c == 0.:
                continue
            marker = position // bin_size
            for first in range(len(local)):
                for second in range(first,len(local)):
                    dosage = local[first,position] + local[second,position]
                    value = logs[sample,site,dosage] - center
                    values[sample,first,second,marker] += value
                    if first != second:
                        values[sample,second,first,marker] += value


class CoarsePrimary:
    def __init__(self, workspace, bin_size=None):
        self.bin_size = workspace.bin_size if bin_size is None else int(bin_size)
        if self.bin_size < 1:
            raise ValueError('bin size must be positive')
        output = List()
        models = []
        for leaf in workspace.leaves:
            bins = (leaf.shape[1]+self.bin_size-1)//self.bin_size
            output.append(np.zeros((len(workspace.logs),len(leaf),len(leaf),bins),np.float64))
        fill_primary(workspace.logs,workspace.leaves,workspace.offsets,self.bin_size,output)
        for values in output:
            keys = list(range(values.shape[1]))
            models.append(dict(hap_keys=keys,key_to_local_idx={i:i for i in keys},
                bin_emissions=values,n_bins=values.shape[3]))
        self.models = PreparedModels(models)
        self.offsets = workspace.offsets
        self.penalty = workspace.penalty

    def score(self, selected):
        return score_rows(self.models,selected,self.penalty)



@njit(cache=True, parallel=True, nogil=True)
def paint_compact(data, packed_offsets, counts, bins, selected, penalty, bin_offsets):
    founders = len(selected)
    first, second = _unordered_pairs(founders)
    states = len(first)
    addresses = selected_emission_addresses(packed_offsets, counts, bins,
                                            selected, first, second)
    samples = len(data)
    answer = np.empty((samples, bin_offsets[-1]), np.int32)
    scores = np.empty(samples, np.float64)
    for sample in prange(samples):
        row = np.zeros(states, np.float64)
        previous = np.empty(bin_offsets[-1], np.int32)
        flags = np.empty((bin_offsets[-1], states), np.bool_)
        for block in range(len(bins)):
            for marker in range(bins[block]):
                index = bin_offsets[block] + marker
                best = int(np.argmax(row))
                previous[index] = best
                switched = row[best] - penalty
                for state in range(states):
                    flags[index, state] = row[state] < switched
                    row[state] = max(row[state], switched) + data[sample, addresses[block, state] + marker]
        state = int(np.argmax(row))
        scores[sample] = row[state]
        for index in range(bin_offsets[-1] - 1, -1, -1):
            answer[sample, index] = first[state] * founders + second[state]
            if flags[index, state]:
                state = previous[index]
    return answer, scores


@njit(cache=True, parallel=True, nogil=True)
def informative_weights(evidence, complete, offsets, bin_offsets, bin_size, tolerance):
    """EXACT native occupancy predicate, not log-evidence or physical bin size."""
    weights = np.zeros((len(evidence), bin_offsets[-1]), np.int64)
    for sample in prange(len(evidence)):
        for block in range(len(offsets) - 1):
            for local in range(offsets[block + 1] - offsets[block]):
                site = offsets[block] + local
                if complete[site]:
                    p = evidence[sample, site]
                    if max(p[0], p[1], p[2]) - min(p[0], p[1], p[2]) > tolerance:
                        weights[sample, bin_offsets[block] + local // bin_size] += 1
    return weights


@njit(cache=True, parallel=True, nogil=True)
def compact_occupancy(painting, weights, founders):
    counts = np.zeros((len(painting), founders), np.int64)
    for sample in prange(len(painting)):
        for marker in range(painting.shape[1]):
            state = painting[sample, marker]
            counts[sample, state // founders] += weights[sample, marker]
            counts[sample, state % founders] += weights[sample, marker]
    return counts.sum(axis=0)


@njit(cache=True, parallel=True, nogil=True)
def fixed_compact(selected, painting, tables, bin_offsets):
    """Same constant-cell branch/order as fixed_proxy.fixed_cached, no SNP scan."""
    founders, blocks = selected.shape
    proposed = selected.copy()
    gains = np.zeros((founders, blocks))
    for block in prange(blocks):
        table = tables[block]
        unary = np.zeros((founders, table.shape[1]))
        for sample in range(len(painting)):
            for marker in range(table.shape[3]):
                state = painting[sample, bin_offsets[block] + marker]
                first, second = state // founders, state % founders
                row_a, row_b = selected[first, block], selected[second, block]
                current = table[sample, row_a, row_b, marker]
                for side in range(1 if first == second else 2):
                    focal = first if side == 0 else second
                    for candidate in range(table.shape[1]):
                        a = candidate if first == focal else row_a
                        b = candidate if second == focal else row_b
                        unary[focal, candidate] += table[sample, a, b, marker] - current
        for focal in range(founders):
            best = int(np.argmax(unary[focal]))
            if unary[focal, best] > 1e-8:
                proposed[focal, block] = best
                gains[focal, block] = unary[focal, best]
    return proposed, gains.sum(axis=1)


class CompactPrimary(CoarsePrimary):
    def __init__(self, workspace, bin_size=200):
        super().__init__(workspace, bin_size)
        self.bin_offsets = np.asarray([0, *np.cumsum([
            model['n_bins'] for model in self.models])], np.int64)
        self.weights = informative_weights(workspace.evidence, workspace.complete,
            workspace.offsets, self.bin_offsets, self.bin_size,
            workspace.evidence.dtype.type(1e-10))

    def paint_compact(self, selected):
        return paint_compact(*packed_emissions(self.models), selected, self.penalty,
                             self.bin_offsets)

    def fixed(self, selected, painting):
        return fixed_compact(selected, painting, self.models.arrays, self.bin_offsets)

    def occupancy(self, painting, founders):
        return compact_occupancy(painting, self.weights, founders)
