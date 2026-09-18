"""Exact interval products for repeated shared-orientation HMM trials.

Candidate endpoints are known before coordinate ascent. They partition each
affected edge into immutable-observation segments. Optional fixed-size cuts
bound build tasks. Four XOR variants preserve insertion/removal of child
observations; parent flips are a selector-state permutation, not a new model.
"""
import math
import numpy as np
from numba import njit, prange
from.model import _transition


@njit(cache=True)
def _join(a, b, positions, genetic, parent_runs, child_runs, maximum_gap,
          artifact_rate, artifact_mean, stationary):
    ma, sa, fa, la = a
    mb, sb, fb, lb = b
    if fa < 0:
        return b
    if fb < 0:
        return a
    reset = (parent_runs[la] != parent_runs[fb]
             or child_runs[la] != child_runs[fb]
             or positions[fb] - positions[la] > maximum_gap)
    transition = _transition(positions[fb] - positions[la],
                             genetic[fb] - genetic[la],
                             artifact_rate, artifact_mean)
    if reset:
        for i in range(4):
            for j in range(4):
                transition[i, j] = stationary[j]
    work = np.zeros((4, 4))
    result = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                work[i, j] += ma[i, k] * transition[k, j]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i, j] += work[i, k] * mb[k, j]
    scale = np.max(result)
    result /= scale
    return result, sa + sb + math.log(scale), fa, lb


@njit(cache=True)
def _leaf(calls, positions, runs, parent, child, slot, genetic, left, right,
          child_flip, maximum_gap, error, artifact_rate, artifact_mean, stationary):
    matrix = np.eye(4)
    scale = 0.
    first = -1
    last = -1
    for site in range(left, right):
        if runs[parent, site] < 0 or runs[child, site] < 0:
            continue
        a, b = calls[parent, site, 0], calls[parent, site, 1]
        c = calls[child, site, slot ^ child_flip]
        if a < 0 or b < 0 or a == b or c < 0:
            continue
        obs = int(c != a)
        emission = np.zeros((4, 4))
        for state in range(4):
            emission[state, state] = 2 * (1 - error if ((state >> 1) ^ (state & 1)) == obs else error)
        value = _join((matrix, scale, first, last), (emission, 0., site, site),
                      positions, genetic, runs[parent], runs[child], maximum_gap,
                      artifact_rate, artifact_mean, stationary)
        matrix, scale, first, last = value
    return matrix, scale, first, last


@njit(cache=True)
def _node_value(tree, node, variant):
    matrix, scale, first, last, lazy, roots, capacity = tree
    return matrix[node, variant], scale[node, variant], first[node, variant], last[node, variant]


@njit(cache=True)
def _store(tree, node, variant, value):
    matrix, scale, first, last, lazy, roots, capacity = tree
    matrix[node, variant], scale[node, variant], first[node, variant], last[node, variant] = value


@njit(cache=True, parallel=True)
def _build_leaves(tree, data, leaf_jobs, maximum_gap, error, rate, mean, stationary):
    calls, positions, runs, parents, children, slots, genetic = data
    for task in prange(len(leaf_jobs)):
        edge, node, left, right = leaf_jobs[task]
        for child_flip in range(2):
            value = _leaf(calls, positions, runs, parents[edge], children[edge], slots[edge],
                          genetic, left, right, child_flip, maximum_gap, error, rate, mean, stationary)
            matrix, scale, first, last = value
            variant = child_flip * 2
            _store(tree, node, variant, value)
            flipped = np.empty((4, 4))
            for i in range(4):
                for j in range(4):
                    flipped[i, j] = matrix[i ^ 2, j ^ 2]
            _store(tree, node, variant + 1, (flipped, scale, first, last))


@njit(cache=True, parallel=True)
def _build_parents(tree, data, edges, maximum_gap, rate, mean, stationary):
    calls, positions, runs, parents, children, slots, genetic = data
    matrix, scale, first, last, lazy, roots, capacity = tree
    for task in prange(len(edges)):
        edge = edges[task]
        base = roots[edge]
        for local in range(capacity[edge] - 1, 0, -1):
            for variant in range(4):
                value = _join(_node_value(tree, base + 2 * local, variant),
                              _node_value(tree, base + 2 * local + 1, variant),
                              positions, genetic, runs[parents[edge]], runs[children[edge]],
                              maximum_gap, rate, mean, stationary)
                _store(tree, base + local, variant, value)


@njit(cache=True)
def _query(tree, base, node, lo, hi, left, right, toggle, carry, positions, genetic,
           parent_runs, child_runs, maximum_gap, rate, mean, stationary):
    # Iterative DFS also permits reliable Numba native-cache reloads.
    depth = 0
    size = hi
    while size > 1:
        depth += 1
        size //= 2
    stack = np.empty((2 * depth + 3, 4), dtype=np.int64)
    stack[0] = (node, lo, hi, carry)
    top = 1
    result = (np.eye(4), 0., -1, -1)
    while top:
        top -= 1
        local, a, b, inherited = stack[top]
        absolute = base + local
        if right <= a or b <= left:
            value = _node_value(tree, absolute, inherited)
        elif left <= a and b <= right:
            value = _node_value(tree, absolute, inherited ^ toggle)
        else:
            middle = (a + b) // 2
            tag = inherited ^ int(tree[4][absolute])
            stack[top] = (2 * local + 1, middle, b, tag)
            stack[top + 1] = (2 * local, a, middle, tag)
            top += 2
            continue
        result = _join(result, value, positions, genetic, parent_runs, child_runs,
                     maximum_gap, rate, mean, stationary)
    return result


@njit(cache=True)
def _flip_node(tree, node, toggle):
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
def _update(tree, base, node, lo, hi, left, right, toggle, positions, genetic,
            parent_runs, child_runs, maximum_gap, rate, mean, stationary):
    depth = 0
    size = hi
    while size > 1:
        depth += 1
        size //= 2
    stack = np.empty((3 * depth + 4, 4), dtype=np.int64)
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
                _store(tree, absolute, variant, _join(
                    _node_value(tree, base + 2 * local, variant),
                    _node_value(tree, base + 2 * local + 1, variant), positions, genetic,
                    parent_runs, child_runs, maximum_gap, rate, mean, stationary))
            continue
        if left <= a and b <= right:
            _flip_node(tree, absolute, toggle)
            continue
        if tree[4][absolute]:
            tag = int(tree[4][absolute])
            _flip_node(tree, base + 2 * local, tag)
            _flip_node(tree, base + 2 * local + 1, tag)
            tree[4][absolute] = 0
        middle = (a + b) // 2
        stack[top] = (local, a, b, 1)
        stack[top + 1] = (2 * local + 1, middle, b, 0)
        stack[top + 2] = (2 * local, a, middle, 0)
        top += 3


@njit(cache=True)
def _score_summary(value, stationary):
    matrix, scale, first, last = value
    if first < 0:
        return 0.
    total = 0.
    for i in range(4):
        for j in range(4):
            total += stationary[i] * matrix[i, j]
    return scale + math.log(total)


@njit(cache=True, parallel=True)
def _trial_scores(tree, data, jobs, maximum_gap, rate, mean, stationary):
    calls, positions, runs, parents, children, slots, genetic = data
    scores = np.empty(len(jobs))
    for task in prange(len(jobs)):
        edge, left, right, toggle = jobs[task]
        value = _query(tree, tree[5][edge], 1, 0, tree[6][edge], left, right, toggle, 0,
                       positions, genetic, runs[parents[edge]], runs[children[edge]],
                       maximum_gap, rate, mean, stationary)
        scores[task] = _score_summary(value, stationary)
    return scores


@njit(cache=True)
def _apply_jobs(tree, data, jobs, maximum_gap, rate, mean, stationary):
    calls, positions, runs, parents, children, slots, genetic = data
    for task in range(len(jobs)):
        edge, left, right, toggle = jobs[task]
        _update(tree, tree[5][edge], 1, 0, tree[6][edge], left, right, toggle,
                positions, genetic, runs[parents[edge]], runs[children[edge]],
                maximum_gap, rate, mean, stationary)


class EdgeIntervalProducts:
    """One bounded flat product forest for the edges affected by candidates."""
    def __init__(self, data, candidates, adjacent, config, maximum_leaf_markers=4096):
        self.data = data
        self.config = config
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
            cuts = np.array(sorted(edge_cuts[edge] | set(range(0, len(positions), maximum_leaf_markers))
                                   | {len(positions)}), dtype=np.int64)
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
        stationary = config.phase_artifact_rate / (
            config.phase_artifact_rate + 1 / config.phase_artifact_mean_bp
        )
        self.stationary = np.array([.5 * (1 - stationary), .5 * stationary] * 2)
        self.args = (config.maximum_gap_bp, config.phase_artifact_rate,
                     config.phase_artifact_mean_bp, self.stationary)
        jobs = np.asarray(leaf_jobs, dtype=np.int64).reshape(-1, 4)
        _build_leaves(self.tree, data, jobs, config.maximum_gap_bp, config.copy_error,
                      config.phase_artifact_rate, config.phase_artifact_mean_bp, self.stationary)
        _build_parents(self.tree, data, self.edges, *self.args)
        self.job_cache = {}

    def jobs(self, sample, left, right, edges):
        key = (int(sample), int(left), int(right))
        if key not in self.job_cache:
            parents, children = self.data[3:5]
            rows = []
            for edge in edges:
                cuts = self.cuts[int(edge)]
                a, b = np.searchsorted(cuts, [left, right])
                assert cuts[a] == left and cuts[b] == right
                toggle = (1 if parents[edge] == sample else 0) | (2 if children[edge] == sample else 0)
                rows.append((edge, a, b, toggle))
            self.job_cache[key] = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
        return self.job_cache[key]

    def score(self, jobs):
        return _trial_scores(self.tree, self.data, jobs, *self.args)

    def accept(self, jobs):
        _apply_jobs(self.tree, self.data, jobs, *self.args)

    def current_scores(self):
        return np.array([_score_summary(_node_value(self.tree, int(self.tree[5][edge]) + 1, 0),
                                         self.stationary) for edge in self.edges])
