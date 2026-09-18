"""Exact shared-prefix candidate scores and bounded replacement ranking."""
import numpy as np
from numba import njit, prange
from numba.typed import List

@njit(cache=True, parallel=True)
def prefix_scores(emissions, local, candidates, order, common_prefix, penalty, samples):
    m, blocks = candidates.shape
    k = len(local)
    result = np.empty((samples, m))
    starts = np.zeros(blocks + 1, np.bool_)
    for b in range(blocks):
        starts[b + 1] = starts[b] or emissions[b].shape[3] > 0
    for sample in prange(samples):
        # The row after block b remains valid while that prefix is shared.
        prefix_dp = np.zeros((blocks + 1, k + 1))
        for index in range(m):
            candidate = order[index]
            for b in range(common_prefix[index], blocks):
                block = emissions[b]
                c = candidates[candidate, b]
                dp = prefix_dp[b + 1]
                dp[:] = prefix_dp[b]
                started = starts[b]
                for site in range(block.shape[3]):
                    switch_base = np.max(dp) - penalty
                    for mate in range(k + 1):
                        partner = local[mate, b] if mate < k else c
                        e = block[sample, c, partner, site]
                        dp[mate] = (max(dp[mate], switch_base) if started else 0.) + e
                    started = True
            result[sample, candidate] = np.max(prefix_dp[blocks])
    return result

def prepare_candidate_scores(sub, candidates):
    """Immutable emission/prefix metadata reused throughout one search."""
    emissions = List()
    for item in sub:
        emissions.append(np.ascontiguousarray(item["bin_emissions"]))
    # lexsort's final key is primary: first block first, without changing
    # candidate output order or the forward floating-point operation order.
    order = np.lexsort(candidates[:,::-1].T)
    common = np.zeros(len(order), np.int64)
    for index in range(1, len(order)):
        prior, current = candidates[order[index - 1]], candidates[order[index]]
        length = 0
        while length < candidates.shape[1] and prior[length] == current[length]:
            length += 1
        common[index] = length
    return emissions, order, common


def continuous_candidate_scores(sub, local, candidates, penalty, *, workspace=None):
    if workspace is None:
        workspace = prepare_candidate_scores(sub, candidates)
    emissions, order, common = workspace
    return prefix_scores(emissions, local, candidates, order, common, float(penalty),
                         emissions[0].shape[0])

@njit(cache=True)
def top_replacement_indices(gains, eligible, quota):
    """Same finite-gain order as (-gain, ('replace', founder, candidate))."""
    capacity = min(quota, gains.shape[0] * len(eligible))
    scores = np.empty(capacity)
    founders = np.empty(capacity, np.int64)
    candidates = np.empty(capacity, np.int64)
    count = 0
    for founder in range(gains.shape[0]):
        for candidate in eligible:
            gain = gains[founder, candidate]
            at = 0
            # Equal gains retain founder/candidate order from the traversal.
            while at < count and gain <= scores[at]:
                at += 1
            if at < capacity:
                end = min(count, capacity - 1)
                for slot in range(end, at, -1):
                    scores[slot] = scores[slot - 1]
                    founders[slot] = founders[slot - 1]
                    candidates[slot] = candidates[slot - 1]
                scores[at] = gain
                founders[at] = founder
                candidates[at] = candidate
                count = min(count + 1, capacity)
    return scores[:count], founders[:count], candidates[:count]

def replacement_proposals(gains, eligible, quota):
    scores, founders, candidates = top_replacement_indices(gains, eligible, quota)
    return [(float(score), ("replace", int(founder), int(candidate)))
            for score, founder, candidate in zip(scores, founders, candidates)]


@njit(cache=True)
def _insert_edit(scores, edits, count, gain, first, second, third):
    """Bounded insertion in (-gain, integer-description) order."""
    at = 0
    while at < count:
        if gain > scores[at]:
            break
        if gain == scores[at]:
            a, b, c = edits[at]
            if (first < a or (first == a and
                    (second < b or (second == b and third < c)))):
                break
        at += 1
    if at < len(scores):
        for slot in range(min(count, len(scores) - 1), at, -1):
            scores[slot] = scores[slot - 1]
            edits[slot] = edits[slot - 1]
        scores[at] = gain
        edits[at, 0] = first
        edits[at, 1] = second
        edits[at, 2] = third
        count = min(count + 1, len(scores))
    return count


@njit(cache=True)
def _top_local_edits(fields, local, quota):
    scores = np.empty(quota)
    edits = np.empty((quota, 3), np.int64)
    count = 0
    screened = 0
    for founder in range(len(local)):
        for block in range(len(fields)):
            for hap in range(fields[block].shape[1]):
                if hap != local[founder, block]:
                    screened += 1
                    count = _insert_edit(scores, edits, count,
                        fields[block][founder, hap], founder, block, hap)
    return scores[:count], edits[:count], screened


def local_proposals(fields, local, quota):
    arrays = List()
    for values in fields:
        arrays.append(values)
    scores, edits, screened = _top_local_edits(arrays, local, quota)
    return [(float(score), ("local", int(a), int(b), int(c)))
            for score, (a, b, c) in zip(scores, edits)], screened


@njit(cache=True)
def _top_boundary_edits(gains, boundary, quota, upper_triangle):
    scores = np.empty(quota)
    edits = np.empty((quota, 3), np.int64)
    count = 0
    for first in range(len(gains)):
        for second in range(first + 1 if upper_triangle else 0, len(gains)):
            if first != second:
                count = _insert_edit(scores, edits, count,
                    gains[first, second], first, second, boundary)
    return scores[:count], edits[:count]


def boundary_proposals(gains, boundary, quota, kind):
    # Keeping q from every boundary is sufficient for the global top-q.
    # Its final ranking still precedes cross-category panel de-duplication.
    scores, edits = _top_boundary_edits(gains, boundary, quota, kind == "splice")
    return [(float(score), (kind, int(a), int(b), int(c)))
            for score, (a, b, c) in zip(scores, edits)]
