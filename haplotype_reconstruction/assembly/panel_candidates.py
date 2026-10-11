"""Exact shared-prefix candidate scores and bounded replacement ranking."""
import numpy as np
from numba import njit, prange
from numba.typed import List

@njit(cache=True, inline='always')
def _max_range(values, lo, hi, initial):
    """max(initial, values[lo:hi]) through eight independent running maxima. A maximum is exact,
    so the value equals one sequential scan's; the independent chains let comparisons overlap."""
    m0 = initial
    m1 = m2 = m3 = m4 = m5 = m6 = m7 = -np.inf
    z = lo
    while z + 8 <= hi:
        v = values[z]
        m0 = v if v > m0 else m0
        v = values[z + 1]
        m1 = v if v > m1 else m1
        v = values[z + 2]
        m2 = v if v > m2 else m2
        v = values[z + 3]
        m3 = v if v > m3 else m3
        v = values[z + 4]
        m4 = v if v > m4 else m4
        v = values[z + 5]
        m5 = v if v > m5 else m5
        v = values[z + 6]
        m6 = v if v > m6 else m6
        v = values[z + 7]
        m7 = v if v > m7 else m7
        z += 8
    while z < hi:
        v = values[z]
        m0 = v if v > m0 else m0
        z += 1
    m0 = m1 if m1 > m0 else m0
    m2 = m3 if m3 > m2 else m2
    m4 = m5 if m5 > m4 else m4
    m6 = m7 if m7 > m6 else m6
    m0 = m2 if m2 > m0 else m0
    m4 = m6 if m6 > m4 else m4
    return m4 if m4 > m0 else m0


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
                    switch_base = _max_range(dp, 0, k + 1, -np.inf) - penalty
                    for mate in range(k + 1):
                        partner = local[mate, b] if mate < k else c
                        e = block[sample, c, partner, site]
                        dp[mate] = (max(dp[mate], switch_base) if started else 0.) + e
                    started = True
            result[sample, candidate] = _max_range(prefix_dp[blocks], 0, k + 1, -np.inf)
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


@njit(cache=True, parallel=True)
def block_restart_scores(emissions, local, penalty, samples):
    """W[s, b, r]: prefix_scores' recursion over block b alone for candidate row r, started
    afresh at the block's first bin (-inf for rows a block does not have).

    Any state sequence restricted to one block is a sequence of that block alone, and the
    switches at block starts are free here, so sum_b W[s, b, c_b] >= the candidate's score."""
    k = local.shape[0]
    blocks = len(emissions)
    widest = 1
    for b in range(blocks):
        widest = max(widest, emissions[b].shape[1])
    result = np.full((samples, blocks, widest), -np.inf)
    for task in prange(samples * blocks):
        sample = task // blocks
        b = task % blocks
        block = emissions[b]
        dp = np.empty(k + 1)
        for c in range(block.shape[1]):
            if block.shape[3] == 0:
                result[sample, b, c] = 0.
                continue
            for site in range(block.shape[3]):
                switch_base = _max_range(dp, 0, k + 1, -np.inf) - penalty if site else 0.
                for mate in range(k + 1):
                    partner = local[mate, b] if mate < k else c
                    e = block[sample, c, partner, site]
                    dp[mate] = (max(dp[mate], switch_base) if site else 0.) + e
            result[sample, b, c] = _max_range(dp, 0, k + 1, -np.inf)
    return result


@njit(cache=True, parallel=True)
def _possibly_novel(restart, candidates, margin):
    """True where some sample's block-restart bound exceeds margin[s]."""
    samples, blocks, _ = restart.shape
    result = np.zeros(candidates.shape[0], np.bool_)
    for c in prange(candidates.shape[0]):
        for s in range(samples):
            total = 0.
            for b in range(blocks):
                total += restart[s, b, candidates[c, b]]
            if total > margin[s]:
                result[c] = True
                break
    return result


def candidate_novelty(local, candidates, penalty, panel_scores, eligible, *, workspace):
    """novelty[c] = sum_s max(score_s(c) - V_s(P), 0) for the eligible candidates, where
    score_s(c) is prefix_scores' value (one strand on c, the other on a panel path or c).

    A candidate is scored exactly unless its block-restart bound stays below V_s(P) in every
    sample by more than the float resolution (the bound and the score sum the same terms in
    another order); then score_s(c) < V_s(P) in every sample and its novelty is exactly 0 (any
    candidate with score_s(c) > V_s(P) in some sample, even by rounding, is scored). Exact scores
    reuse DP prefixes shared by consecutive scored candidates in sorted order, so every scored
    value equals the all-candidate computation. Returns (novelty, candidates scored exactly)."""
    emissions, order, _ = workspace
    samples = emissions[0].shape[0]
    restart = block_restart_scores(emissions, local, float(penalty), samples)
    margin = panel_scores - 1e-9 * (1.0 + np.abs(panel_scores))
    possible = _possibly_novel(restart, candidates, margin) & eligible
    novelty = np.zeros(len(candidates))
    chosen = order[possible[order]]
    if not len(chosen):
        return novelty, 0
    paths = np.ascontiguousarray(candidates[chosen])
    common = np.zeros(len(chosen), np.int64)
    if len(chosen) > 1:
        equal = paths[1:] == paths[:-1]
        common[1:] = np.where(equal.all(axis=1), paths.shape[1], np.argmin(equal, axis=1))
    scores = prefix_scores(emissions, local, paths, np.arange(len(chosen)), common, float(penalty),
                           samples)
    novelty[chosen] = np.maximum(scores - panel_scores[:, None], 0.).sum(axis=0)
    return novelty, len(chosen)

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
