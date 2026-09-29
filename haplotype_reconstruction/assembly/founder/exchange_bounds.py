"""Exact top-quota suffix exchanges via inexpensive founder-incidence bounds.

For a swap a<->b, an altered term is left[a,c]+right[b,c] (or the reverse).
It cannot exceed max_c(left[a,c])+max_c(right[b,c]). Unaltered and switched
terms cannot exceed the original reference. Sum these sample-wise bounds,
then evaluate exchanges in bound order until none can enter the top quota.
Tied bounds are not pruned. All actually returned shortlist scores are exact.

Preparation O(N B K^2 log K), plus O(B K^2 log(B K)) bound ordering.
Q exact queries cost O(N Q K). Q<=B*K*(K-1)/2,
so the worst case remains cubic in K; memory is O(N B K^2). Unqueried entries
are -inf, so this is a top-quota interface, NOT a full score-matrix replacement.
"""
import numpy as np
from numba import njit, prange, get_num_threads
from . import scoring


@njit(cache=True, parallel=True, nogil=True)
def prepare_bounds(forward, backward, founders, penalty):
    first, second = scoring._unordered_pairs(founders)
    a, b = np.triu_indices(founders, 1)
    samples, boundaries, states = forward.shape
    orders = np.empty((samples, boundaries, states), np.int32)
    upper = np.zeros((boundaries, len(a)))
    reference = np.zeros(boundaries)
    for boundary in prange(boundaries):
        terms = np.empty(states)
        left_max = np.empty(founders)
        right_max = np.empty(founders)
        for sample in range(samples):
            left, right = forward[sample, boundary], backward[sample, boundary]
            left_max[:] = -np.inf
            right_max[:] = -np.inf
            for state in range(states):
                i, j = first[state], second[state]
                left_max[i] = max(left_max[i], left[state])
                left_max[j] = max(left_max[j], left[state])
                right_max[i] = max(right_max[i], right[state])
                right_max[j] = max(right_max[j], right[state])
                terms[state] = left[state] + right[state]
            order = np.argsort(-terms)
            orders[sample, boundary] = order
            baseline = max(np.max(left) + np.max(right) - penalty, terms[order[0]])
            reference[boundary] += baseline
            for pair in range(len(a)):
                bound = max(baseline, left_max[a[pair]] + right_max[b[pair]],
                            left_max[b[pair]] + right_max[a[pair]])
                upper[boundary, pair] += bound
    return upper, reference, orders, first, second, a, b


@njit(cache=True, parallel=True, nogil=True)
def query(forward, backward, orders, first, second, pair_a, pair_b, founders, penalty, flats):
    index = np.empty((founders, founders), np.int64)
    for state in range(len(first)):
        index[first[state], second[state]] = state
        index[second[state], first[state]] = state
    result = np.empty(len(flats))
    for task in prange(len(flats)):
        boundary, pair = flats[task] // len(pair_a), flats[task] % len(pair_a)
        a, b = pair_a[pair], pair_b[pair]
        total = 0.
        for sample in range(len(forward)):
            left, right = forward[sample, boundary], backward[sample, boundary]
            best = np.max(left) + np.max(right) - penalty
            for state in orders[sample, boundary]:
                i, j = first[state], second[state]
                if ((i != a and i != b and j != a and j != b) or (i == a and j == b)):
                    best = max(best, left[state] + right[state])
                    break
            best = max(best, left[index[a, a]] + right[index[b, b]])
            best = max(best, left[index[b, b]] + right[index[a, a]])
            for c in range(founders):
                if c != a and c != b:
                    best = max(best, left[index[a, c]] + right[index[b, c]])
                    best = max(best, left[index[b, c]] + right[index[a, c]])
            total += best
        result[task] = total
    return result


def top_scores(forward, backward, founders, penalty, quota):
    upper, reference, orders, first, second, a, b = prepare_bounds(forward, backward, founders, penalty)
    values = np.full(upper.size, -np.inf)
    priority = np.argsort(-upper.ravel(), kind='stable')
    quota = min(max(0, int(quota)), len(priority))
    if not quota:
        return values.reshape(upper.shape), reference, a, b, dict(queries=0, total=len(priority))
    size = max(quota, get_num_threads())
    consumed = 0
    best_values = np.empty(0)
    while consumed < len(priority):
        if consumed >= quota:
            best = best_values[0]
            margin = max(1e-7, 1e-12 * abs(best))
            if upper.ravel()[priority[consumed]] < best - margin:
                break
        picked = np.ascontiguousarray(priority[consumed:consumed+size])
        exact = query(forward, backward, orders, first, second, a, b, founders, penalty, picked)
        if np.any(exact > upper.ravel()[picked] + np.maximum(1e-7, 1e-12*np.abs(exact))):
            raise RuntimeError('Suffix-exchange incidence bound fell below its exact score')
        values[picked] = exact
        best_values = np.sort(np.concatenate((best_values, exact)))[-quota:]
        consumed += len(picked)
    return values.reshape(upper.shape), reference, a, b, dict(queries=consumed, total=len(priority))
