"""Lazy-flip segment trees for the shared-orientation coordinate ascent (O(log J) per trial and accepted flip).

Candidate tract endpoints are known before the ascent; they (plus fixed 4096-marker cuts that bound build
tasks) partition each affected edge's or sample's markers into leaves. A trial flips a range of leaves; nodes
keep every flip variant, so a query swaps variants instead of rebuilding, and an accepted flip is a lazy tag.

EdgeProductTree: per affected edge, scaled 4x4 interval products of the edge HMM (decoder.transition). Four
variants: parent flip x child flip (a child flip changes which observations exist, so its variants are built
from the data; a parent flip is a selector-state permutation of the product).
OrientationPriorTree: per affected sample, the log prior of its orientation path under the orientation-error
CTMC; two variants (path and its complement). A summary excludes its first valid marker's initial
probability; joining two adds one boundary transition (or a stationary restart). Missing markers reset the prior.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .decoder import transition


LEAF_MARKERS = 4096


def _leaf_cuts(cuts, sites):
    return np.asarray(sorted(set(cuts) | set(range(0, sites, LEAF_MARKERS)) | {sites}), dtype=np.int64)


@njit(cache=True)
def _tree_depth(size):
    depth = 0
    while size > 1:
        depth += 1
        size //= 2
    return depth


# Edge interval products ------------------------------------------------------------------------------------


@njit(cache=True)
def _join_products(a, b, positions, genetic, parent_runs, child_runs, maximum_gap, artifact_rate,
                   artifact_mean, stationary):
    ma, sa, fa, la = a
    mb, sb, fb, lb = b
    if fa < 0:
        return b
    if fb < 0:
        return a
    reset = (parent_runs[la] != parent_runs[fb]
             or child_runs[la] != child_runs[fb]
             or positions[fb] - positions[la] > maximum_gap)
    matrix = transition(positions[fb] - positions[la], genetic[fb] - genetic[la], artifact_rate, artifact_mean)
    if reset:
        for i in range(4):
            for j in range(4):
                matrix[i, j] = stationary[j]
    work = np.zeros((4, 4))
    result = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                work[i, j] += ma[i, k] * matrix[k, j]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i, j] += work[i, k] * mb[k, j]
    scale = np.max(result)
    result /= scale
    return result, sa + sb + math.log(scale), fa, lb


@njit(cache=True)
def _leaf_product(calls, positions, runs, parent, child, slot, genetic, left, right,
                  child_flip, maximum_gap, error, artifact_rate, artifact_mean, stationary):
    matrix = np.eye(4)
    scale = 0.
    first = np.int64(-1)
    last = np.int64(-1)
    for site in range(left, right):
        if runs[parent, site] < 0 or runs[child, site] < 0:
            continue
        a, b = calls[parent, site, 0], calls[parent, site, 1]
        c = calls[child, site, slot ^ child_flip]
        if a < 0 or b < 0 or a == b or c < 0:
            continue
        observation = int(c != a)
        emission = np.zeros((4, 4))
        for state in range(4):
            emission[state, state] = 2 * (1 - error if ((state >> 1) ^ (state & 1)) == observation else error)
        matrix, scale, first, last = _join_products(
            (matrix, scale, first, last), (emission, 0., site, site), positions, genetic, runs[parent],
            runs[child], maximum_gap, artifact_rate, artifact_mean, stationary)
    return matrix, scale, first, last


@njit(cache=True)
def _product_value(tree, node, variant):
    matrix, scale, first, last, lazy, roots, capacity = tree
    return matrix[node, variant], scale[node, variant], first[node, variant], last[node, variant]


@njit(cache=True)
def _store_product(tree, node, variant, value):
    matrix, scale, first, last, lazy, roots, capacity = tree
    matrix[node, variant], scale[node, variant], first[node, variant], last[node, variant] = value


@njit(cache=True, parallel=True)
def _build_product_leaves(tree, data, leaf_jobs, maximum_gap, error, rate, mean, stationary):
    calls, positions, runs, parents, children, slots, genetic = data
    for task in prange(len(leaf_jobs)):
        edge, node, left, right = leaf_jobs[task]
        for child_flip in range(2):
            value = _leaf_product(calls, positions, runs, parents[edge], children[edge], slots[edge],
                                  genetic, left, right, child_flip, maximum_gap, error, rate, mean, stationary)
            matrix, scale, first, last = value
            variant = child_flip * 2
            _store_product(tree, node, variant, value)
            flipped = np.empty((4, 4))
            for i in range(4):
                for j in range(4):
                    flipped[i, j] = matrix[i ^ 2, j ^ 2]
            _store_product(tree, node, variant + 1, (flipped, scale, first, last))


@njit(cache=True, parallel=True)
def _build_product_parents(tree, data, edges, maximum_gap, rate, mean, stationary):
    calls, positions, runs, parents, children, slots, genetic = data
    matrix, scale, first, last, lazy, roots, capacity = tree
    for task in prange(len(edges)):
        edge = edges[task]
        base = roots[edge]
        for local in range(capacity[edge] - 1, 0, -1):
            for variant in range(4):
                value = _join_products(_product_value(tree, base + 2 * local, variant),
                                       _product_value(tree, base + 2 * local + 1, variant),
                                       positions, genetic, runs[parents[edge]], runs[children[edge]],
                                       maximum_gap, rate, mean, stationary)
                _store_product(tree, base + local, variant, value)


@njit(cache=True)
def _query_product(tree, base, node, lo, hi, left, right, toggle, carry, positions, genetic,
                   parent_runs, child_runs, maximum_gap, rate, mean, stationary):
    # Iterative depth-first traversal (cached Numba kernels cannot recurse).
    stack = np.empty((2 * _tree_depth(hi) + 3, 4), dtype=np.int64)
    stack[0] = (node, lo, hi, carry)
    top = 1
    # Match stored index types; literal sentinels can poison cached callee signatures.
    result = (np.eye(4), 0., np.int64(-1), np.int64(-1))
    while top:
        top -= 1
        local, a, b, inherited = stack[top]
        absolute = base + local
        if right <= a or b <= left:
            value = _product_value(tree, absolute, inherited)
        elif left <= a and b <= right:
            value = _product_value(tree, absolute, inherited ^ toggle)
        else:
            middle = (a + b) // 2
            tag = inherited ^ int(tree[4][absolute])
            stack[top] = (2 * local + 1, middle, b, tag)
            stack[top + 1] = (2 * local, a, middle, tag)
            top += 2
            continue
        result = _join_products(result, value, positions, genetic, parent_runs, child_runs,
                                maximum_gap, rate, mean, stationary)
    return result


@njit(cache=True)
def _flip_product_node(tree, node, toggle):
    matrix, scale, first, last, lazy, roots, capacity = tree
    for variant in range(4):
        other = variant ^ toggle
        if other > variant:
            for i in range(4):
                for j in range(4):
                    matrix[node, variant, i, j], matrix[node, other, i, j] = (
                        matrix[node, other, i, j], matrix[node, variant, i, j])
            scale[node, variant], scale[node, other] = scale[node, other], scale[node, variant]
            first[node, variant], first[node, other] = first[node, other], first[node, variant]
            last[node, variant], last[node, other] = last[node, other], last[node, variant]
    lazy[node] ^= toggle


@njit(cache=True)
def _update_product(tree, base, node, lo, hi, left, right, toggle, positions, genetic,
                    parent_runs, child_runs, maximum_gap, rate, mean, stationary):
    stack = np.empty((3 * _tree_depth(hi) + 4, 4), dtype=np.int64)
    stack[0] = (node, lo, hi, 0)
    top = 1
    while top:
        top -= 1
        local, a, b, closing = stack[top]
        absolute = base + local
        if right <= a or b <= left:
            continue
        if closing:
            for variant in range(4):
                _store_product(tree, absolute, variant, _join_products(
                    _product_value(tree, base + 2 * local, variant),
                    _product_value(tree, base + 2 * local + 1, variant), positions, genetic,
                    parent_runs, child_runs, maximum_gap, rate, mean, stationary))
            continue
        if left <= a and b <= right:
            _flip_product_node(tree, absolute, toggle)
            continue
        if tree[4][absolute]:
            tag = int(tree[4][absolute])
            _flip_product_node(tree, base + 2 * local, tag)
            _flip_product_node(tree, base + 2 * local + 1, tag)
            tree[4][absolute] = 0
        middle = (a + b) // 2
        stack[top] = (local, a, b, 1)
        stack[top + 1] = (2 * local + 1, middle, b, 0)
        stack[top + 2] = (2 * local, a, middle, 0)
        top += 3


@njit(cache=True)
def _product_score(value, stationary):
    matrix, scale, first, last = value
    if first < 0:
        return 0.
    total = 0.
    for i in range(4):
        for j in range(4):
            total += stationary[i] * matrix[i, j]
    return scale + math.log(total)


@njit(cache=True, parallel=True)
def _trial_product_scores(tree, data, jobs, maximum_gap, rate, mean, stationary):
    calls, positions, runs, parents, children, slots, genetic = data
    scores = np.empty(len(jobs))
    for task in prange(len(jobs)):
        edge, left, right, toggle = jobs[task]
        value = _query_product(tree, tree[5][edge], 1, 0, tree[6][edge], left, right, toggle, 0,
                               positions, genetic, runs[parents[edge]], runs[children[edge]],
                               maximum_gap, rate, mean, stationary)
        scores[task] = _product_score(value, stationary)
    return scores


@njit(cache=True)
def _apply_product_jobs(tree, data, jobs, maximum_gap, rate, mean, stationary):
    calls, positions, runs, parents, children, slots, genetic = data
    for task in range(len(jobs)):
        edge, left, right, toggle = jobs[task]
        _update_product(tree, tree[5][edge], 1, 0, tree[6][edge], left, right, toggle,
                        positions, genetic, runs[parents[edge]], runs[children[edge]],
                        maximum_gap, rate, mean, stationary)


class EdgeProductTree:
    """One flat product forest for the edges incident to candidate tracts (scores = edge_log_evidence)."""

    def __init__(self, data, candidates, adjacent, config):
        self.data = data
        calls, positions, runs, parents, children, slots, genetic = data
        edge_cuts = {}
        for sample, left, right in candidates:
            for edge in adjacent[sample]:
                edge_cuts.setdefault(edge, set()).update((left, right))
        self.edges = np.array(sorted(edge_cuts), dtype=np.int64)
        roots = np.full(len(parents), -1, dtype=np.int64)
        capacity = np.zeros(len(parents), dtype=np.int64)
        self.cuts = {}
        leaf_jobs = []
        total = 0
        for edge in self.edges:
            cuts = _leaf_cuts(edge_cuts[edge], len(positions))
            self.cuts[int(edge)] = cuts
            count = len(cuts) - 1
            cap = 1 << (count - 1).bit_length()
            roots[edge] = total
            capacity[edge] = cap
            leaf_jobs.extend((edge, total + cap + i, int(cuts[i]), int(cuts[i + 1])) for i in range(count))
            total += 2 * cap
        self.tree = (np.zeros((total, 4, 4, 4)), np.zeros((total, 4)),
                     np.full((total, 4), -1, dtype=np.int64), np.full((total, 4), -1, dtype=np.int64),
                     np.zeros(total, dtype=np.uint8), roots, capacity)
        stationary = config.artifact_rate / (config.artifact_rate + 1 / config.artifact_mean_bp)
        self.stationary = np.array([.5 * (1 - stationary), .5 * stationary] * 2)
        self.args = (config.maximum_gap_bp, config.artifact_rate, config.artifact_mean_bp, self.stationary)
        jobs = np.asarray(leaf_jobs, dtype=np.int64).reshape(-1, 4)
        _build_product_leaves(self.tree, data, jobs, config.maximum_gap_bp, config.copy_error,
                              config.artifact_rate, config.artifact_mean_bp, self.stationary)
        _build_product_parents(self.tree, data, self.edges, *self.args)
        self.job_cache = {}

    def jobs(self, sample, left, right, edges):
        """(edge, first leaf, stop leaf, toggle) rows of a trial flip of ``sample`` on [left, right)."""
        key = (int(sample), int(left), int(right))
        if key not in self.job_cache:
            parents, children = self.data[3:5]
            rows = []
            for edge in edges:
                cuts = self.cuts[int(edge)]
                a, b = np.searchsorted(cuts, [left, right])
                toggle = (1 if parents[edge] == sample else 0) | (2 if children[edge] == sample else 0)
                rows.append((edge, a, b, toggle))
            self.job_cache[key] = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
        return self.job_cache[key]

    def score(self, jobs):
        return _trial_product_scores(self.tree, self.data, jobs, *self.args)

    def accept(self, jobs):
        _apply_product_jobs(self.tree, self.data, jobs, *self.args)

    def current_scores(self):
        return np.array([_product_score(_product_value(self.tree, int(self.tree[5][edge]) + 1, 0),
                                        self.stationary) for edge in self.edges])


# Orientation prior sums ------------------------------------------------------------------------------------


@njit(cache=True)
def _prior_boundary(previous, current, left, right, positions, runs, rate, mean):
    stationary = rate / (rate + 1 / mean)
    if left < 0 or right != left + 1 or runs[left] != runs[right]:
        probability = stationary if current else 1 - stationary
    else:
        changed = -math.expm1(-(rate + 1 / mean) * (positions[right] - positions[left]))
        enter, leave = stationary * changed, (1 - stationary) * changed
        probability = ((leave if not current else 1 - leave) if previous
                       else (enter if current else 1 - enter))
    return math.log(probability)


@njit(cache=True)
def _prior_value(tree, node, variant):
    score, first, last, states, lazy, roots, capacity = tree
    return score[node, variant], first[node], last[node], states[node, variant, 0], states[node, variant, 1]


@njit(cache=True)
def _store_prior(tree, node, variant, value):
    score, first, last, states, lazy, roots, capacity = tree
    score[node, variant], first[node], last[node], states[node, variant, 0], states[node, variant, 1] = value


@njit(cache=True)
def _join_priors(a, b, positions, runs, rate, mean):
    sa, fa, la, pa, qa = a
    sb, fb, lb, pb, qb = b
    if fa < 0:
        return b
    if fb < 0:
        return a
    return sa + sb + _prior_boundary(qa, pb, la, fb, positions, runs, rate, mean), fa, lb, pa, qb


@njit(cache=True, parallel=True)
def _build_prior_leaves(tree, positions, runs, jobs, rate, mean):
    for task in prange(len(jobs)):
        sample, node, left, right = jobs[task]
        for variant in range(2):
            first = -1
            previous = -1
            score = 0.
            for site in range(left, right):
                if runs[sample, site] < 0:
                    continue
                if first < 0:
                    first = site
                else:
                    score += _prior_boundary(variant, variant, previous, site, positions, runs[sample], rate, mean)
                previous = site
            _store_prior(tree, node, variant, (score, first, previous, variant, variant))


@njit(cache=True, parallel=True)
def _build_prior_parents(tree, positions, runs, samples, rate, mean):
    for task in prange(len(samples)):
        sample = samples[task]
        base = tree[5][sample]
        for local in range(tree[6][sample] - 1, 0, -1):
            for variant in range(2):
                _store_prior(tree, base + local, variant, _join_priors(
                    _prior_value(tree, base + 2 * local, variant), _prior_value(tree, base + 2 * local + 1, variant),
                    positions, runs[sample], rate, mean))


@njit(cache=True)
def _query_prior(tree, base, node, lo, hi, left, right, carry, positions, runs, rate, mean):
    stack = np.empty((2 * _tree_depth(hi) + 3, 4), dtype=np.int64)
    stack[0] = (node, lo, hi, carry)
    top = 1
    # Match _prior_value: int64 indices and int8 states, including the empty summary.
    result = (0., np.int64(-1), np.int64(-1), np.int8(0), np.int8(0))
    while top:
        top -= 1
        local, a, b, inherited = stack[top]
        absolute = base + local
        if right <= a or b <= left:
            value = _prior_value(tree, absolute, inherited)
        elif left <= a and b <= right:
            value = _prior_value(tree, absolute, inherited ^ 1)
        else:
            middle = (a + b) // 2
            tag = inherited ^ int(tree[4][absolute])
            stack[top] = (2 * local + 1, middle, b, tag)
            stack[top + 1] = (2 * local, a, middle, tag)
            top += 2
            continue
        result = _join_priors(result, value, positions, runs, rate, mean)
    return result


@njit(cache=True)
def _flip_prior_node(tree, node):
    score, first, last, states, lazy, roots, capacity = tree
    score[node, 0], score[node, 1] = score[node, 1], score[node, 0]
    for end in range(2):
        states[node, 0, end], states[node, 1, end] = states[node, 1, end], states[node, 0, end]
    lazy[node] ^= 1


@njit(cache=True)
def _update_prior(tree, base, node, lo, hi, left, right, positions, runs, rate, mean):
    stack = np.empty((3 * _tree_depth(hi) + 4, 4), dtype=np.int64)
    stack[0] = (node, lo, hi, 0)
    top = 1
    while top:
        top -= 1
        local, a, b, closing = stack[top]
        absolute = base + local
        if right <= a or b <= left:
            continue
        if closing:
            for variant in range(2):
                _store_prior(tree, absolute, variant, _join_priors(
                    _prior_value(tree, base + 2 * local, variant), _prior_value(tree, base + 2 * local + 1, variant),
                    positions, runs, rate, mean))
            continue
        if left <= a and b <= right:
            _flip_prior_node(tree, absolute)
            continue
        if tree[4][absolute]:
            _flip_prior_node(tree, base + 2 * local)
            _flip_prior_node(tree, base + 2 * local + 1)
            tree[4][absolute] = 0
        middle = (a + b) // 2
        stack[top] = (local, a, b, 1)
        stack[top + 1] = (2 * local + 1, middle, b, 0)
        stack[top + 2] = (2 * local, a, middle, 0)
        top += 3


@njit(cache=True)
def _prior_score(value, positions, runs, rate, mean):
    body, first, last, phase, end = value
    if first < 0:
        return 0.
    return body + _prior_boundary(0, phase, -1, first, positions, runs, rate, mean)


@njit(cache=True, parallel=True)
def _prior_scores(tree, positions, runs, jobs, rate, mean):
    result = np.empty(len(jobs))
    for task in prange(len(jobs)):
        sample, left, right = jobs[task]
        value = _query_prior(tree, tree[5][sample], 1, 0, tree[6][sample], left, right, 0,
                             positions, runs[sample], rate, mean)
        result[task] = _prior_score(value, positions, runs[sample], rate, mean)
    return result


class OrientationPriorTree:
    """Range-flip sums of the orientation prior for the samples with candidate tracts."""

    def __init__(self, positions, runs, candidates, config):
        self.positions, self.runs = positions, runs
        self.args = (config.error_rate, config.error_mean_bp)
        sample_cuts = {}
        for sample, left, right in candidates:
            sample_cuts.setdefault(sample, set()).update((left, right))
        self.samples = np.asarray(sorted(sample_cuts), dtype=np.int64)
        self.cuts = {}
        self.job_cache = {}
        roots = np.full(len(runs), -1, dtype=np.int64)
        capacity = np.zeros(len(runs), dtype=np.int64)
        jobs = []
        total = 0
        for sample in self.samples:
            cuts = _leaf_cuts(sample_cuts[sample], len(positions))
            self.cuts[int(sample)] = cuts
            count = len(cuts) - 1
            cap = 1 << (count - 1).bit_length()
            roots[sample] = total
            capacity[sample] = cap
            jobs.extend((sample, total + cap + i, int(cuts[i]), int(cuts[i + 1])) for i in range(count))
            total += 2 * cap
        self.tree = (np.zeros((total, 2)), np.full(total, -1, dtype=np.int64),
                     np.full(total, -1, dtype=np.int64), np.zeros((total, 2, 2), dtype=np.int8),
                     np.zeros(total, dtype=np.uint8), roots, capacity)
        _build_prior_leaves(self.tree, positions, runs, np.asarray(jobs, dtype=np.int64).reshape(-1, 4), *self.args)
        _build_prior_parents(self.tree, positions, runs, self.samples, *self.args)

    def jobs(self, trials):
        rows = []
        for sample, left, right in trials:
            key = (int(sample), int(left), int(right))
            if key not in self.job_cache:
                cuts = self.cuts[int(sample)]
                a, b = np.searchsorted(cuts, [left, right])
                self.job_cache[key] = (sample, a, b)
            rows.append(self.job_cache[key])
        return np.asarray(rows, dtype=np.int64).reshape(-1, 3)

    def score(self, jobs):
        return _prior_scores(self.tree, self.positions, self.runs, jobs, *self.args)

    def accept(self, job):
        sample, left, right = job
        _update_prior(self.tree, self.tree[5][sample], 1, 0, self.tree[6][sample], left, right,
                      self.positions, self.runs[sample], *self.args)

    def current_scores(self):
        return np.asarray([_prior_score(_prior_value(self.tree, int(self.tree[5][s]) + 1, 0),
                                        self.positions, self.runs[s], *self.args) for s in self.samples])
