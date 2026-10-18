"""Tier-1 proposal generators of the join's panel search.

A tier-1 sweep ranks same-count edits of the current panel by cheap scores and scores only the
top ones of each kind exactly (search.PanelSearch). The generators here:

* conditional fields (fixed painting): per path f and block row r, the change of the painted
  emissions when f takes row r (both copies of a homozygous state replaced once), plus the pair
  correction of a suffix exchange; they rank replace, local, onesided, splice and cycle edits;
* candidate novelty: per pool path c, sum_s max(score_s(c) - V_s(P), 0) with score_s(c) the
  mate-HMM score (one strand on c, the other on a panel path or c), scored exactly only where a
  block-restart bound leaves it open; candidates sharing a prefix reuse its DP rows;
* static replacement gains: exact all-replacement gains in the no-switch objective (pair
  reassignment allowed), streamed over the sorted pool with shared prefix sums;
* boundary votes of an exploratory (lower-penalty) painting and greedy / cycle permutations.

Ranking ties follow the canonical edit descriptions (-gain, then the integer description), and
every kernel reduces per sample in a fixed order, so proposals do not depend on thread counts.
"""
from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange
from numba.typed import List

from .objective import max_range


@njit(cache=True, parallel=True, nogil=True)
def prefix_scores(flat, block_start, block_haps, block_bins, local, candidates, order,
                  common_prefix, penalty, samples):
    """Per-sample scores of the candidates in ``order`` (one strand on the candidate, the other
    on a panel path or the candidate), the DP rows of a shared prefix reused; emissions: an
    EmissionLayout's arrays."""
    m, blocks = candidates.shape
    k = len(local)
    result = np.empty((samples, m))
    starts = np.zeros(blocks + 1, np.bool_)
    for b in range(blocks):
        starts[b + 1] = starts[b] or block_bins[b] > 0
    for sample in prange(samples):
        # The row after block b remains valid while that prefix is shared.
        prefix_dp = np.zeros((blocks + 1, k + 1))
        for index in range(m):
            candidate = order[index]
            for b in range(common_prefix[index], blocks):
                haps, bins = block_haps[b], block_bins[b]
                c = candidates[candidate, b]
                row = block_start[b] + (sample * haps + c) * haps * bins
                dp = prefix_dp[b + 1]
                dp[:] = prefix_dp[b]
                started = starts[b]
                for site in range(bins):
                    switch_base = max_range(dp, 0, k + 1, -np.inf) - penalty
                    for mate in range(k + 1):
                        partner = local[mate, b] if mate < k else c
                        e = flat[row + partner * bins + site]
                        dp[mate] = (max(dp[mate], switch_base) if started else 0.) + e
                    started = True
            result[sample, candidate] = max_range(prefix_dp[blocks], 0, k + 1, -np.inf)
    return result


def prepare_candidate_scores(layout, candidates):
    """The pool's sorted order and shared prefixes, reused throughout one search (layout: the
    search's EmissionLayout): (layout, order, common)."""
    # lexsort's final key is primary: first block first, without changing
    # candidate output order or the forward floating-point operation order.
    order = np.lexsort(candidates[:, ::-1].T)
    common = np.zeros(len(order), np.int64)
    for index in range(1, len(order)):
        prior, current = candidates[order[index - 1]], candidates[order[index]]
        length = 0
        while length < candidates.shape[1] and prior[length] == current[length]:
            length += 1
        common[index] = length
    return layout, order, common


@njit(cache=True, parallel=True, nogil=True)
def block_restart_scores(flat, block_start, block_haps, block_bins, local, penalty, samples):
    """W[s, b, r]: prefix_scores' recursion over block b alone for candidate row r, started
    afresh at the block's first bin (-inf for rows a block does not have).

    Any state sequence restricted to one block is a sequence of that block alone, and the
    switches at block starts are free here, so sum_b W[s, b, c_b] >= the candidate's score."""
    k = local.shape[0]
    blocks = len(block_start)
    widest = 1
    for b in range(blocks):
        widest = max(widest, block_haps[b])
    result = np.full((samples, blocks, widest), -np.inf)
    for task in prange(samples * blocks):
        sample = task // blocks
        b = task % blocks
        haps, bins = block_haps[b], block_bins[b]
        dp = np.empty(k + 1)
        for c in range(haps):
            if bins == 0:
                result[sample, b, c] = 0.
                continue
            row = block_start[b] + (sample * haps + c) * haps * bins
            for site in range(bins):
                switch_base = max_range(dp, 0, k + 1, -np.inf) - penalty if site else 0.
                for mate in range(k + 1):
                    partner = local[mate, b] if mate < k else c
                    e = flat[row + partner * bins + site]
                    dp[mate] = (max(dp[mate], switch_base) if site else 0.) + e
            result[sample, b, c] = max_range(dp, 0, k + 1, -np.inf)
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
    layout, order, _ = workspace
    samples = layout.samples
    restart = block_restart_scores(*layout.arrays(), local, float(penalty), samples)
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
    scores = prefix_scores(*layout.arrays(), local, paths, np.arange(len(chosen)), common,
                           float(penalty), samples)
    novelty[chosen] = np.maximum(scores - panel_scores[:, None], 0.).sum(axis=0)
    return novelty, len(chosen)


@njit(cache=True)
def _top_replacement_indices(gains, eligible, quota):
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
    """The top ``quota`` replace(f, c) edits by gains[f, c] over the eligible pool paths."""
    scores, founders, candidates = _top_replacement_indices(gains, eligible, quota)
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
    """The top ``quota`` local(f, b, r) edits by their conditional field, and the number screened."""
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
    """The top ``quota`` (kind, f, g, boundary) edits by gains[f, g] (splices: f < g only)."""
    # Keeping q from every boundary is sufficient for the global top-q.
    # Its final ranking still precedes cross-category panel de-duplication.
    scores, edits = _top_boundary_edits(gains, boundary, quota, kind == "splice")
    return [(float(score), (kind, int(a), int(b), int(c)))
            for score, (a, b, c) in zip(scores, edits)]


@njit(cache=True, parallel=True)
def conditional_fields(emission, painted, local_paths, panel_size):
    """Per-path replacement deltas and pair correction for a suffix exchange.

    Use the same emission values as the float64 full-score tensor. Homozygous
    painting states replace BOTH copies once, rather than treating them as
    two independent replacements. Uniform/masked input has zero deltas.
    """
    samples, local_k, _, bins = emission.shape
    unary = np.zeros((panel_size, local_k))
    pair = np.zeros((panel_size, panel_size))
    for founder in prange(panel_size):
        for sample in range(samples):
            for site in range(bins):
                state = painted[sample, site]
                first = state // panel_size
                second = state % panel_size
                if first != founder and second != founder:
                    continue
                a, b = local_paths[first], local_paths[second]
                current = np.float64(emission[sample, a, b, site])
                for candidate in range(local_k):
                    aa = candidate if first == founder else a
                    bb = candidate if second == founder else b
                    unary[founder, candidate] += np.float64(emission[sample, aa, bb, site]) - current
                if first == founder and first != second:
                    # Only this ordered painting contributes here. Each
                    # unordered pair is symmetrized after the parallel pass.
                    pair[first, second] += (2 * current - np.float64(emission[sample, a, a, site])
                                            - np.float64(emission[sample, b, b, site]))
    return unary, pair + pair.T


@njit(cache=True)
def fixed_paint_delta(emission, painted, old_local, new_local, panel_size):
    """Change of the painted emissions when the panel's local rows change old -> new."""
    result = 0.
    for sample in range(len(painted)):
        for site in range(painted.shape[1]):
            state = painted[sample, site]
            a, b = state // panel_size, state % panel_size
            result += np.float64(emission[sample, new_local[a], new_local[b], site])
            result -= np.float64(emission[sample, old_local[a], old_local[b], site])
    return result


@njit(cache=True)
def _exclusion_scores(current):
    """All one-founder exclusions need at most three K^2 scans per sample."""
    samples, k, _ = current.shape
    best = np.empty(samples)
    excluded = np.empty((samples, k))
    for sample in range(samples):
        flat = np.argmax(current[sample])
        first, second = flat // k, flat % k
        best[sample] = current[sample, first, second]
        excluded[sample, :] = best[sample]
        for i in (first, second):
            maximum = -np.inf
            for a in range(k):
                for b in range(k):
                    if a != i and b != i:
                        maximum = max(maximum, current[sample, a, b])
            excluded[sample, i] = maximum
    return best, excluded


@njit(cache=True, parallel=True)
def _stream_static_gains(flat, start, haps, local, candidates, order, common, best, excluded,
                         chunks):
    """Accumulate blocks before mate maxima; retain only a K-vector per task.

    Candidates are visited in sorted order (order, common = leading blocks shared with the
    previous one) in contiguous chunks, one task each: the block sums of a shared prefix are
    reused, accumulated in block order as before; per candidate the samples are summed in order.
    Each task sums into rows of its own (the same chain per cell, from zero) and copies them
    out at the end: rows of tasks interleaved in candidate order would share cache lines.
    Block b's totals are flat[start[b] + (sample * K_b + a) * K_b + c] (K_b = haps[b]).
    """
    samples, k = excluded.shape
    blocks = len(start)
    gains = np.zeros((len(candidates), k))
    count = len(order)
    for chunk in prange(chunks):
        lo = (count * chunk) // chunks
        hi = (count * (chunk + 1)) // chunks
        if lo >= hi:
            continue
        cross = np.empty((blocks + 1, k))
        diagonal = np.empty(blocks + 1)
        own = np.zeros((hi - lo, k))
        for sample in range(samples):
            for position in range(lo, hi):
                candidate = order[position]
                shared = 0 if position == lo else min(common[position], blocks)
                if shared == 0:
                    cross[0, :] = 0.
                    diagonal[0] = 0.
                for block in range(shared, blocks):
                    kb = haps[block]
                    base = start[block] + sample * kb * kb
                    c = candidates[candidate, block]
                    diagonal[block + 1] = diagonal[block] + np.float64(flat[base + c * kb + c])
                    for partner in range(k):
                        cross[block + 1, partner] = (
                            cross[block, partner]
                            + np.float64(flat[base + c * kb + local[partner, block]]))
                row = cross[blocks]
                winner = np.argmax(row)
                first = row[winner]
                second = -np.inf
                for partner in range(k):
                    if partner != winner:
                        second = max(second, row[partner])
                for removed in range(k):
                    retained = second if removed == winner else first
                    score = max(excluded[sample, removed], retained, diagonal[blocks])
                    own[position - lo, removed] += score - best[sample]
        for position in range(lo, hi):
            for removed in range(k):
                gains[order[position], removed] = own[position - lo, removed]
    return gains.T


def static_replacement_gains(local, candidates, *, totals, workspace):
    """Exact all-replacement gains in the *no-switch* proposal objective.

    This allows pair reassignment, unlike fixed-painting deltas. Only two
    endpoints of the best old pair need an exclusion rescan. The best and
    second-best candidate partners handle all removed founders in O(K*M).
    No-switch scores screen proposals, not the final recombining model.
    The N*M*K candidate/mate tensor is streamed in native code, preserving
    float64 block accumulation and the sample-reduction order; candidates
    sharing a prefix (workspace: prepare_candidate_scores' sorted order) share
    its block sums; totals: the blocks' bin-summed emissions (S, K_b, K_b).
    """
    samples = totals[0].shape[0]
    k = len(local)
    current = np.zeros((samples, k, k))
    for b, total in enumerate(totals):
        a = local[:, b]
        current += total[:, a[:, None], a[None, :]]
    best, excluded = _exclusion_scores(current)
    _, order, common = workspace
    chunks = max(1, min(len(order), 4 * max(1, numba.get_num_threads())))
    # The kernel reads the block totals as one flat array: taking a block's array out of a
    # typed List in its inner loop would update that array's reference count, an atomic shared
    # by all threads, which made the kernel slower with more threads than with one.
    haps = np.asarray([total.shape[1] for total in totals], dtype=np.int64)
    start = np.zeros(len(totals), dtype=np.int64)
    start[1:] = np.cumsum([total.size for total in totals])[:-1]
    flat = np.concatenate([np.ascontiguousarray(total).ravel() for total in totals])
    return _stream_static_gains(flat, start, haps, local, candidates, order, common, best,
                                excluded, chunks)


def cycle_permutation(preference):
    """Take cycles from a best-target graph; no cubic assignment solver."""
    size = len(preference)
    target = np.argmax(preference, axis=1)
    permutation = np.arange(size)
    finished = set()
    for first in range(size):
        if first in finished:
            continue
        path = []
        offset = {}
        current = first
        while current not in finished and current not in offset:
            offset[current] = len(path)
            path.append(current)
            current = int(target[current])
        if current in offset:
            cycle = path[offset[current]:]
            if len(cycle) > 1:
                for item in cycle:
                    permutation[item] = target[item]
        finished.update(path)
    return permutation


def greedy_permutation(weights):
    """Quadratic-log greedy assignment plus a fixed quadratic swap search."""
    k = len(weights)
    permutation = np.full(k, -1, dtype=np.int64)
    used = np.zeros(k, bool)
    for flat in np.argsort(-weights.ravel(), kind='stable'):
        first, second = divmod(int(flat), k)
        if permutation[first] < 0 and not used[second]:
            permutation[first] = second
            used[second] = True
    return _improve_permutation(weights, permutation)


@njit(cache=True)
def _improve_permutation(weights, permutation):
    for _ in range(20):
        changed = False
        for i in range(len(permutation)):
            for j in range(i + 1, len(permutation)):
                a, b = permutation[i], permutation[j]
                if weights[i, b] + weights[j, a] > weights[i, a] + weights[j, b]:
                    permutation[i], permutation[j] = b, a
                    changed = True
        if not changed:
            break
    return permutation


def boundary_vote_matrix(sample_paths, boundary_bin, k):
    """W[i, j]: strand units on path i at bin boundary_bin - 1 and on path j at bin boundary_bin
    (each sample contributes its two strands; painted states a * k + c)."""
    votes = np.zeros((k, k), dtype=np.float64)
    for s in range(sample_paths.shape[0]):
        pre = int(sample_paths[s, boundary_bin - 1])
        post = int(sample_paths[s, boundary_bin])
        pre_a, pre_b = pre // k, pre % k
        post_a, post_b = post // k, post % k
        votes[pre_a, post_a] += 1
        votes[pre_b, post_b] += 1
    return votes
