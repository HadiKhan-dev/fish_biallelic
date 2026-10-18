"""Binary-hypercube chain kernels: hidden states are independent bits, each flipping with probability theta.

Two-state chains (scaled forward-backward with an optional anchor, Viterbi), four-state chains of two independent
bits (one transition ``mix4``, scaled forward-backward cavity, full-enumeration Viterbi, in-place max-product
butterfly step ``max_step4``) and the run contraction of chains with zero-probability transitions. Family phase uses
them for its phase and selector chains and gauges, recombination for its two-state screening posterior. All
recurrences are scaled forward-backward or max-product scans in a fixed order, so results do not depend on thread
counts.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def mix4(a, b, c, d, t):
    """One transition of two independent bits (state = 2*bit1 + bit0), each flipping with probability t."""
    u = 1 - t
    x0, x1, x2, x3 = u * a + t * c, u * b + t * d, u * c + t * a, u * d + t * b
    return u * x0 + t * x1, u * x1 + t * x0, u * x2 + t * x3, u * x3 + t * x2


@njit(cache=True)
def selector_cavity(emission, theta):
    """Scaled four-state forward-backward; the cavity omits each marker's own emission. Shape (n, 4)."""
    n = len(emission)
    alpha = np.empty((n, 4), dtype=np.float64)
    predicted = np.empty((n, 4), dtype=np.float64)
    beta = np.ones((n, 4), dtype=np.float64)
    for site in range(n):
        if site == 0:
            p = (.25, .25, .25, .25)
        else:
            p = mix4(alpha[site - 1, 0], alpha[site - 1, 1], alpha[site - 1, 2], alpha[site - 1, 3],
                     theta[site - 1])
        total = 0.
        for state in range(4):
            predicted[site, state] = p[state]
            alpha[site, state] = p[state] * emission[site, state]
            total += alpha[site, state]
        for state in range(4):
            alpha[site, state] /= total
    for site in range(n - 2, -1, -1):
        values = mix4(emission[site + 1, 0] * beta[site + 1, 0], emission[site + 1, 1] * beta[site + 1, 1],
                      emission[site + 1, 2] * beta[site + 1, 2], emission[site + 1, 3] * beta[site + 1, 3],
                      theta[site])
        total = sum(values)
        for state in range(4):
            beta[site, state] = values[state] / total
    cavity = np.empty_like(alpha)
    for site in range(n):
        total = 0.
        for state in range(4):
            cavity[site, state] = predicted[site, state] * beta[site, state]
            total += cavity[site, state]
        for state in range(4):
            cavity[site, state] /= total
    return cavity


@njit(cache=True)
def selector_viterbi(emission, theta):
    """Max-product path of a four-state selector chain (uniform start); ties keep the lower previous state."""
    n = len(emission)
    score = np.zeros(4, dtype=np.float64)
    new = np.empty(4, dtype=np.float64)
    back = np.zeros((n, 4), dtype=np.int8)
    for site in range(n):
        if site == 0:
            for state in range(4):
                new[state] = math.log(max(1e-300, emission[site, state])) - math.log(4.)
        else:
            t = theta[site - 1]
            stay = math.log(1 - t)
            jump = math.log(t) if t != 0 else 0.
            for state in range(4):
                best = -np.inf
                arg = 0
                for previous in range(4):
                    bits = previous ^ state
                    changes = (bits >> 1) + (bits & 1)
                    if changes and t == 0:
                        continue
                    value = score[previous] + (2 - changes) * stay
                    if changes:
                        value += changes * jump
                    if value > best:
                        best = value
                        arg = previous
                new[state] = best + math.log(max(1e-300, emission[site, state]))
                back[site, state] = arg
        score, new = new, score
    result = np.empty(n, dtype=np.int8)
    state = np.argmax(score)
    for site in range(n - 1, -1, -1):
        result[site] = state
        if site:
            state = back[site, state]
    return result


@njit(cache=True, inline="always")
def max_step4(score, t):
    """In-place max-product transition of two independent bits (four states), each flipping with probability t."""
    stay = math.log1p(-t)
    jump = math.log(t) if t > 0 else -np.inf
    for bit in (1, 2):
        for state in range(4):
            if state & bit:
                continue
            a = score[state]
            c = score[state | bit]
            score[state] = max(a + stay, c + jump)
            score[state | bit] = max(c + stay, a + jump)


@njit(cache=True)
def binary_posterior(emission, theta, anchor):
    """Scaled two-state forward-backward marginals, shape (n, 2).

    theta[i] is the switch probability between markers i and i + 1 (1/2 severs the chain). At ``anchor``
    (>= 0) state 1 is forbidden: the gauge of a root's phase chain.
    """
    length = emission.shape[0]
    alpha = np.empty((length, 2), dtype=np.float64)
    predicted = np.empty((length, 2), dtype=np.float64)
    beta = np.ones((length, 2), dtype=np.float64)
    for site in range(length):
        if site == 0:
            p0, p1 = 0.5, 0.5
        else:
            t = theta[site - 1]
            p0 = alpha[site - 1, 0] * (1 - t) + alpha[site - 1, 1] * t
            p1 = alpha[site - 1, 1] * (1 - t) + alpha[site - 1, 0] * t
        if site == anchor:
            p1 = 0.0
        predicted[site, 0], predicted[site, 1] = p0, p1
        a0, a1 = p0 * emission[site, 0], p1 * emission[site, 1]
        total = a0 + a1
        alpha[site, 0], alpha[site, 1] = a0 / total, a1 / total
    for site in range(length - 2, -1, -1):
        e0 = emission[site + 1, 0] * beta[site + 1, 0]
        e1 = emission[site + 1, 1] * beta[site + 1, 1]
        if site + 1 == anchor:
            e1 = 0.0
        t = theta[site]
        b0, b1 = (1 - t) * e0 + t * e1, t * e0 + (1 - t) * e1
        total = b0 + b1
        beta[site, 0], beta[site, 1] = b0 / total, b1 / total
    posterior = np.empty_like(alpha)
    for site in range(length):
        c0 = predicted[site, 0] * beta[site, 0]
        c1 = predicted[site, 1] * beta[site, 1]
        total = c0 + c1
        c0, c1 = c0 / total, c1 / total
        p0, p1 = c0 * emission[site, 0], c1 * emission[site, 1]
        posterior[site, 0], posterior[site, 1] = p0 / (p0 + p1), p1 / (p0 + p1)
    return posterior


@njit(cache=True)
def binary_viterbi(emission, theta, anchor):
    """Max-product path of a two-state chain (uniform start, state 1 forbidden at ``anchor``); ties stay."""
    n = len(emission)
    scores = np.zeros(2, dtype=np.float64)
    back = np.zeros((n, 2), dtype=np.int8)
    for i in range(n):
        new = np.empty(2, dtype=np.float64)
        for state in range(2):
            if i == anchor and state == 1:
                new[state] = -np.inf
                continue
            if i == 0:
                new[state] = math.log(max(emission[i, state], 1e-300)) - math.log(2.)
                continue
            stay = math.log(max(1 - theta[i - 1], 1e-300))
            jump = math.log(theta[i - 1]) if theta[i - 1] > 0 else -np.inf
            a = scores[state] + stay
            b = scores[1 - state] + jump
            if a >= b:
                new[state] = a
                back[i, state] = state
            else:
                new[state] = b
                back[i, state] = 1 - state
            new[state] += math.log(max(emission[i, state], 1e-300))
        scores = new
    path = np.empty(n, dtype=np.int8)
    state = 0 if scores[0] >= scores[1] else 1
    for i in range(n - 1, -1, -1):
        path[i] = state
        if i:
            state = back[i, state]
    return path


@njit(cache=True)
def chain_segments(theta, sites):
    """Runs of markers joined by zero-probability transitions: (starts int64[S+1], switch probabilities [S-1]).

    A phase chain cannot switch inside a run, so its HMM runs over runs (painting bins), not markers.
    """
    count = 1
    for t in theta:
        if t > 0:
            count += 1
    starts = np.empty(count + 1, dtype=np.int64)
    trans = np.empty(count - 1)
    starts[0] = 0
    starts[count] = sites
    cursor = 1
    for site in range(1, sites):
        if theta[site - 1] > 0:
            starts[cursor] = site
            trans[cursor - 1] = theta[site - 1]
            cursor += 1
    return starts, trans


@njit(cache=True)
def contracted_emissions(emissions, starts, anchor):
    """Per-run products of marker emissions (scaled per run) and the run holding ``anchor`` (-1 without).

    The anchor's forbidden state is removed before scaling, so the allowed state cannot underflow against it
    in a long run.
    """
    runs = len(starts) - 1
    compact = np.empty((runs, 2))
    anchor_run = -1
    for run in range(runs):
        a = 0.
        b = 0.
        for site in range(starts[run], starts[run + 1]):
            a += math.log(emissions[site, 0])
            b += math.log(emissions[site, 1])
        if starts[run] <= anchor < starts[run + 1]:
            anchor_run = run
            b = -np.inf
        maximum = max(a, b)
        compact[run, 0] = math.exp(a - maximum)
        compact[run, 1] = math.exp(b - maximum)
    return compact, anchor_run
