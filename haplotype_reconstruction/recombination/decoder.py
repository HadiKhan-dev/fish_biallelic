"""Per-meiosis decoding: transmitted parental homologue x correlated orientation-artifact HMM.

For edge (parent, child, slot) the observation at an informative site (parent heterozygous, child's slot track
called, both covered) is which parental homologue (decoder frame) the child's allele matches. Hidden state
2 * S + B: S the transmitted homologue (switches with the Haldane probability of the genetic map), B an
asymmetric correlated orientation artifact (entering at ``artifact_rate`` per bp, leaving after
``artifact_mean_bp`` on average); the observed homologue is S XOR B, read with ``copy_error``. The chain
restarts at every coverage run, component (segment) change and gap > maximum_gap_bp. Posterior expected
crossovers, called interval-censored events (supported homologue changes) and observable exposure follow.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..model.genetic_map import haldane_fraction
from .map import posterior_intervals


@njit(cache=True, inline="always")
def fill_transition(matrix, theta, enter, leave):
    """4x4 transition of (homologue, artifact): homologue flips with theta, artifact enters/leaves."""
    for a in range(4):
        for b in range(4):
            p = theta if (a >> 1) != (b >> 1) else 1 - theta
            p *= ((leave if not b & 1 else 1 - leave) if a & 1 else (enter if b & 1 else 1 - enter))
            matrix[a, b] = p


@njit(cache=True)
def transition(distance, genetic_distance, artifact_rate, artifact_mean):
    """Transition matrix over ``distance`` bp and ``genetic_distance`` Morgans."""
    theta = haldane_fraction(genetic_distance)
    stationary = artifact_rate / (artifact_rate + 1 / artifact_mean)
    changed = -math.expm1(-(artifact_rate + 1 / artifact_mean) * distance)
    matrix = np.empty((4, 4))
    fill_transition(matrix, theta, stationary * changed, (1 - stationary) * changed)
    return matrix


@njit(cache=True)
def origin_with_artifacts(observations, positions, resets, rate, error, artifact_rate, artifact_mean,
                          genetic=None):
    """(P(homologue 0), P(no artifact), posterior switch probability into each marker), scaled forward-backward.

    Haldane theta from ``genetic`` Morgans when given, else from ``rate`` x bp. No confidence from the family
    model is multiplied into these emissions.
    """
    n = len(observations)
    exit_rate = 1 / artifact_mean
    stationary = artifact_rate / (artifact_rate + exit_rate)
    forward = np.empty((n, 4), dtype=np.float64)
    matrix = np.empty((4, 4), dtype=np.float64)
    for i in range(n):
        if i and not resets[i]:
            distance = positions[i] - positions[i - 1]
            if genetic is None:
                theta = haldane_fraction(rate * distance)
            else:
                theta = haldane_fraction(genetic[i] - genetic[i - 1])
            changed = -math.expm1(-(artifact_rate + exit_rate) * distance)
            fill_transition(matrix, theta, stationary * changed, (1 - stationary) * changed)
        total = 0.
        for b in range(4):
            value = .5 * (stationary if b & 1 else 1 - stationary)
            if i and not resets[i]:
                value = 0.
                for a in range(4):
                    value += forward[i - 1, a] * matrix[a, b]
            value *= 1 - error if ((b >> 1) ^ (b & 1)) == observations[i] else error
            forward[i, b] = value
            total += value
        forward[i] /= total
    origin = np.empty(n)
    clean = np.empty(n)
    back = np.ones(4)
    following = np.empty(4)
    switch = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        w0 = forward[i, 0] * back[0]
        w1 = forward[i, 1] * back[1]
        w2 = forward[i, 2] * back[2]
        w3 = forward[i, 3] * back[3]
        weight = w0 + w1 + w2 + w3
        origin[i] = (w0 + w1) / weight
        clean[i] = (w0 + w2) / weight
        if i and not resets[i]:
            distance = positions[i] - positions[i - 1]
            if genetic is None:
                theta = haldane_fraction(rate * distance)
            else:
                theta = haldane_fraction(genetic[i] - genetic[i - 1])
            changed = -math.expm1(-(artifact_rate + exit_rate) * distance)
            fill_transition(matrix, theta, stationary * changed, (1 - stationary) * changed)
            jump_weight = 0.
            normalizer = 0.
            for a in range(4):
                total = 0.
                for b in range(4):
                    emission = 1 - error if ((b >> 1) ^ (b & 1)) == observations[i] else error
                    value = matrix[a, b] * emission * back[b]
                    total += value
                    if (a >> 1) != (b >> 1):
                        jump_weight += forward[i - 1, a] * value
                following[a] = total
                normalizer += forward[i - 1, a] * total
            switch[i] = jump_weight / normalizer
            weight = following.sum()
            for a in range(4):
                back[a] = following[a] / weight
        else:
            back[:] = 1.
    return origin, clean, switch


@njit(cache=True)
def edge_log_evidence(observations, positions, resets, genetic, copy_error, artifact_rate, artifact_mean):
    """Marginal log likelihood of one edge's observations relative to neutral Bernoulli(1/2) observations.

    The ratio gives uninformative observations likelihood one, so exchanging known and missing alleles (a
    child's strand flip) earns no normalisation advantage.
    """
    stationary = artifact_rate / (artifact_rate + 1 / artifact_mean)
    previous = np.empty(4)
    current = np.empty(4)
    logz = 0.
    for i in range(len(observations)):
        if i and not resets[i]:
            matrix = transition(positions[i] - positions[i - 1], genetic[i] - genetic[i - 1],
                                artifact_rate, artifact_mean)
        for b in range(4):
            value = .5 * (stationary if b & 1 else 1 - stationary)
            if i and not resets[i]:
                value = 0.
                for a in range(4):
                    value += previous[a] * matrix[a, b]
            value *= 2 * (1 - copy_error if ((b >> 1) ^ (b & 1)) == observations[i] else copy_error)
            current[b] = value
        total = current.sum()
        logz += math.log(total)
        for b in range(4):
            previous[b] = current[b] / total
    return logz


@njit(cache=True)
def one_meiosis(calls, positions, components, coverage, offsets, parent, child, slot, rate, error, threshold,
                max_gap, artifact_rate, artifact_mean, bins, genetic=None, knots=None, knot_genetic=None):
    """Decode one edge from decoder-frame calls int8 [N, L, 2].

    Returns (called events (n, 5): left bp, right bp, left/right homologue support, component; callable spans
    (m, 3); informative, supported and artifact-uncertain marker counts; expected crossovers and exposure per
    map bin; informative spans (k, 3)).
    """
    n = len(positions)
    indices = np.empty(n, dtype=np.int64)
    observations = np.empty(n, dtype=np.int8)
    resets = np.empty(n, dtype=np.bool_)
    p_run, c_run = offsets[parent], offsets[child]
    last_p_run = last_c_run = -1
    count = 0
    for j in range(n):
        while p_run < offsets[parent + 1] and coverage[p_run, 1] <= j:
            p_run += 1
        while c_run < offsets[child + 1] and coverage[c_run, 1] <= j:
            c_run += 1
        if (components[j] < 0 or p_run == offsets[parent + 1] or c_run == offsets[child + 1]
                or coverage[p_run, 0] > j or coverage[c_run, 0] > j):
            continue
        p0, p1 = calls[parent, j, 0], calls[parent, j, 1]
        c = calls[child, j, slot]
        if c < 0 or p0 < 0 or p1 < 0 or p0 == p1:
            continue
        reset = count == 0
        if count:
            previous = indices[count - 1]
            reset = (components[j] != components[previous] or p_run != last_p_run
                     or c_run != last_c_run or positions[j] - positions[previous] > max_gap)
        indices[count] = j
        observations[count] = 0 if c == p0 else 1
        resets[count] = reset
        count += 1
        last_p_run, last_c_run = p_run, c_run
    indices = indices[:count]
    observation_positions = positions[indices]
    if genetic is None:
        posterior, clean, switch = origin_with_artifacts(observations[:count], observation_positions,
                                                         resets[:count], rate, error, artifact_rate, artifact_mean)
        soft_counts, soft_exposure, soft_spans = posterior_intervals(
            observation_positions, components[indices], resets[:count], switch, bins, rate)
    else:
        observation_genetic = genetic[indices]
        posterior, clean, switch = origin_with_artifacts(observations[:count], observation_positions,
                                                         resets[:count], rate, error, artifact_rate, artifact_mean,
                                                         observation_genetic)
        soft_counts, soft_exposure, soft_spans = posterior_intervals(
            observation_positions, components[indices], resets[:count], switch, bins, rate,
            observation_genetic, knots, knot_genetic)
    # Typically only a few spans/crossovers survive: outputs grow per edge, never as sample x site x state.
    events = [(0., 0., 0., 0., 0.) for _ in range(0)]
    spans = [(0., 0., 0.) for _ in range(0)]
    last = -1
    supported = 0
    for i in range(count):
        if resets[i]:
            last = -1
        support = max(posterior[i], 1 - posterior[i])
        if support < threshold or clean[i] < threshold:
            continue
        supported += 1
        if last >= 0 and observation_positions[i] - observation_positions[last] <= max_gap:
            left, right = float(observation_positions[last]), float(observation_positions[i])
            component = float(components[indices[i]])
            if spans and spans[-1][1] == left and spans[-1][2] == component:
                spans[-1] = (spans[-1][0], right, component)
            else:
                spans.append((left, right, component))
            if (posterior[i] >= .5) != (posterior[last] >= .5):
                events.append((left, right, max(posterior[last], 1 - posterior[last]), support, component))
        last = i
    event_array = np.empty((len(events), 5), dtype=np.float64)
    span_array = np.empty((len(spans), 3), dtype=np.float64)
    for i in range(len(events)):
        for k in range(5):
            event_array[i, k] = events[i][k]
    for i in range(len(spans)):
        for k in range(3):
            span_array[i, k] = spans[i][k]
    return (event_array, span_array, count, supported, int(np.sum(clean < threshold)),
            soft_counts, soft_exposure, soft_spans)


@njit(cache=True, parallel=True)
def decode_meioses(calls, positions, components, coverage, offsets, parents, children, slots, rate, error,
                   threshold, max_gap, artifact_rate, artifact_mean, bins, genetic=None, knots=None,
                   knot_genetic=None):
    """Every edge independently (parallel over edges); see ``one_meiosis``."""
    n = len(parents)
    events = [np.empty((0, 5), dtype=np.float64) for _ in range(n)]
    spans = [np.empty((0, 3), dtype=np.float64) for _ in range(n)]
    soft_spans = [np.empty((0, 3), dtype=np.float64) for _ in range(n)]
    soft_counts = np.zeros((n, len(bins) - 1))
    soft_exposure = np.zeros_like(soft_counts)
    marker_counts = np.zeros((n, 3), dtype=np.int64)
    for edge in prange(n):
        event, span, informative, supported, uncertain, sc, se, ss = one_meiosis(
            calls, positions, components, coverage, offsets, parents[edge], children[edge], slots[edge],
            rate, error, threshold, max_gap, artifact_rate, artifact_mean, bins, genetic, knots, knot_genetic)
        events[edge], spans[edge] = event, span
        soft_counts[edge] = sc
        soft_exposure[edge] = se
        soft_spans[edge] = ss
        marker_counts[edge, 0], marker_counts[edge, 1] = informative, supported
        marker_counts[edge, 2] = uncertain
    return events, spans, marker_counts, soft_counts, soft_exposure, soft_spans
