"""Genotype-preserving phase polish of two-parent children.

Starting from the scaffold (every frame 0), the polish maximises a conditional copy-consistency score over
sample frames f[n, j] (which scaffold track is the first) and per-edge transmission selectors s[e, j] (which
parental homologue the child's track copies):

    score = sum_e sum_j copy(parent allele s ^ f_parent, child track slot ^ f_child)   (called alleles only,
            log(2(1 - copy_error)) on a match, log(2 copy_error) on a mismatch, 0 when either is unknown)
          + sum_e sum_j selector switch terms (Haldane, 1/2 at component edges)
          + sum_n sum_j frame switch terms (phase_transitions)

by graph-coloured coordinate ascent: within a colour (no two vertices adjacent in the pedigree) every two-parent
child's (frame, two incoming selectors) path is re-decoded exactly by an eight-state Viterbi with its
neighbours fixed, and contiguous intervals of the new path are accepted when they raise the score (and, where
they change an observable phase, the copy-consistency part by at least ``minimum_biological_gain``). Called
alleles never change: only the track order does. Samples with fewer than two observed parents keep frame 0.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit, prange

from ..execution import claim_threads


@dataclass(frozen=True)
class PolishResult:
    """frames int8 [N, L] (relative to the polished scaffold), selectors int8 [E, L], accepted vertex updates
    per sweep; ``converged`` = the last sweep accepted nothing."""

    frames: np.ndarray
    selectors: np.ndarray
    sweep_updates: tuple[int, ...]
    converged: bool


@njit(cache=True, inline='always')
def copy_log(a, b, match, mismatch):
    """Copy-consistency log factor of two alleles (0 when either is unknown)."""
    if a < 0 or b < 0:
        return 0.
    return match if a == b else mismatch


def pedigree_layout(samples, parents, children, slots):
    """(incoming int64 [N, 2] edge per parental slot or -1, outgoing offsets int64 [N+1], outgoing edges
    int64 [E], colour classes): a greedy colouring by descending degree (ties by index) of the pedigree graph."""
    incoming = np.full((samples, 2), -1, dtype=np.int64)
    outgoing = [[] for _ in range(samples)]
    adjacent = [set() for _ in range(samples)]
    for e, (p, c, s) in enumerate(zip(parents, children, slots)):
        incoming[c, s] = e
        outgoing[p].append(e)
        adjacent[p].add(int(c))
        adjacent[c].add(int(p))
    colours = np.full(samples, -1, dtype=np.int64)
    for i in sorted(range(samples), key=lambda j: (-len(adjacent[j]), j)):
        used = {colours[j] for j in adjacent[i] if colours[j] >= 0}
        colour = 0
        while colour in used:
            colour += 1
        colours[i] = colour
    groups = tuple(np.flatnonzero(colours == c) for c in range(int(colours.max()) + 1))
    offsets = np.r_[0, np.cumsum([len(x) for x in outgoing])].astype(np.int64)
    edges = np.asarray([e for group in outgoing for e in group], dtype=np.int64)
    return incoming, offsets, edges, groups


def phase_transitions(bins, components, probability, frames, selectors, window):
    """Frame switch probabilities per marker interval: coarse painting-bin edges, refined near changes.

    A coarse boundary is an interval where the phase bin changes inside a component; it switches with
    probability p in the coarse model (the family chain's). Its group is the boundary plus the following
    intervals up to the next coarse boundary or component edge (the interior of the bin to its right). Intervals
    of a group within ``window`` sites of a current frame or selector change are allowed as well, and for n
    allowed intervals theta = (1 - (1 - 2p)^(1/n)) / 2, so their composed binary transition keeps the group's
    probability p. Component edges switch with probability 1/2. Every other interval has theta 0: a group led by
    a component edge and a chromosome's leading group without a coarse boundary hold no switch probability, so
    the phase stays rigid inside the first painting bin of a chromosome or component (family-phase B2: base gave
    those groups p as well). Windows come from current paths.
    """
    bins = np.asarray(bins)
    components = np.asarray(components)
    theta = np.zeros(len(bins) - 1, dtype=np.float64)
    if not len(theta):
        return theta
    coarse = bins[1:] != bins[:-1]
    edge = components[1:] != components[:-1]
    selected = coarse.copy()
    if window > 0:
        changes = np.any(frames[:, 1:] != frames[:, :-1], axis=0)
        if len(selectors):
            changes |= np.any(selectors[:, 1:] != selectors[:, :-1], axis=0)
        delta = np.zeros(len(bins), dtype=np.int64)
        for site in np.flatnonzero(changes):
            delta[max(0, site - window)] += 1
            delta[min(len(bins) - 1, site + window + 1)] -= 1
        selected |= np.cumsum(delta[:-1]) > 0
    # Groups start at coarse boundaries and component edges (contiguous runs, not global bin IDs: ragged gaps
    # may be -1); the first group starts at interval 0 whatever leads it.
    leaders = np.flatnonzero(coarse | edge)
    cuts = np.r_[0, leaders[leaders > 0], len(theta)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if coarse[a] and not edge[a]:
            sites = np.flatnonzero(selected[a:b]) + a
            theta[sites] = -.5 * np.expm1(np.log1p(-2 * probability) / len(sites))
    theta[edge] = .5
    return theta


@njit(cache=True, parallel=True)
def initialize_selectors(scaffold, frames, parents, children, slots, logstay, logjump, error):
    """Per edge, the MAP selector path given the current frames (two-state Viterbi on copy consistency)."""
    edges, sites = len(parents), scaffold.shape[1]
    result = np.zeros((edges, sites), dtype=np.int8)
    match, mismatch = math.log(2.) + math.log1p(-error), math.log(2.) + math.log(error)
    for e in prange(edges):
        p, c, slot = parents[e], children[e], slots[e]
        back = np.zeros((sites, 2), dtype=np.int8)
        score = np.full(2, -math.log(2.))
        current = np.empty(2)
        for j in range(sites):
            child = scaffold[c, j, slot ^ frames[c, j]]
            for s in range(2):
                value = score[s]
                if j:
                    stay = score[s] + logstay[j - 1]
                    jump = score[1 - s] + logjump[j - 1]
                    if jump > stay:
                        value = jump
                        back[j, s] = 1 - s
                    else:
                        value = stay
                        back[j, s] = s
                current[s] = value + copy_log(scaffold[p, j, s ^ frames[p, j]], child, match, mismatch)
            score, current = current, score
        state = 0 if score[0] >= score[1] else 1
        for j in range(sites - 1, -1, -1):
            result[e, j] = state
            if j:
                state = back[j, state]
    return result


@njit(cache=True, parallel=True)
def conditional_score(scaffold, frames, selectors, parents, children, slots,
                      logstay, logjump, phase_stay, phase_jump, error):
    """The polish objective (module docstring) of a configuration; the reference for tests of the local gains."""
    samples, sites = scaffold.shape[:2]
    values = np.zeros(samples + len(parents), dtype=np.float64)
    match, mismatch = math.log(2.) + math.log1p(-error), math.log(2.) + math.log(error)
    for task in prange(len(values)):
        value = -math.log(2.)
        if task < samples:
            for j in range(1, sites):
                value += phase_jump[j - 1] if frames[task, j] != frames[task, j - 1] else phase_stay[j - 1]
        else:
            e = task - samples
            p, c, slot = parents[e], children[e], slots[e]
            for j in range(sites):
                value += copy_log(scaffold[p, j, selectors[e, j] ^ frames[p, j]],
                                  scaffold[c, j, slot ^ frames[c, j]], match, mismatch)
                if j:
                    value += logjump[j - 1] if selectors[e, j] != selectors[e, j - 1] else logstay[j - 1]
        values[task] = value
    # Fixed-order sequential total: a parallel reduction's association order would depend on the team size.
    total = 0.
    for task in range(len(values)):
        total += values[task]
    return total


@njit(cache=True)
def interval_gain(i, a, b, path, scaffold, frames, selectors, incoming, offsets, outgoing,
                  parents, logstay, logjump, phase_stay, phase_jump, error):
    """(copy-consistency gain, frame-prior gain) of replacing vertex i's configuration on [a, b) by ``path``."""
    e0, e1 = incoming[i, 0], incoming[i, 1]
    match, mismatch = math.log(2 * (1 - error)), math.log(2 * error)
    biological, regularisation = 0., 0.
    for j in range(a, b):
        new_f, old_f = path[j] & 1, frames[i, j]
        for slot, e in ((0, e0), (1, e1)):
            if e < 0:
                continue
            parent = parents[e]
            new_s = (path[j] >> (slot + 1)) & 1
            old_s = selectors[e, j]
            biological += copy_log(scaffold[parent, j, new_s ^ frames[parent, j]],
                                   scaffold[i, j, slot ^ new_f], match, mismatch)
            biological -= copy_log(scaffold[parent, j, old_s ^ frames[parent, j]],
                                   scaffold[i, j, slot ^ old_f], match, mismatch)
    for j in range(max(1, a), min(scaffold.shape[1], b + 1)):
        new_left = (path[j - 1] & 1) if a <= j - 1 < b else frames[i, j - 1]
        new_right = (path[j] & 1) if a <= j < b else frames[i, j]
        old_jump = frames[i, j - 1] != frames[i, j]
        new_jump = new_left != new_right
        regularisation += (phase_jump[j - 1] if new_jump else phase_stay[j - 1])
        regularisation -= (phase_jump[j - 1] if old_jump else phase_stay[j - 1])
        for slot, e in ((0, e0), (1, e1)):
            if e < 0:
                continue
            left = ((path[j - 1] >> (slot + 1)) & 1) if a <= j - 1 < b else selectors[e, j - 1]
            right = ((path[j] >> (slot + 1)) & 1) if a <= j < b else selectors[e, j]
            biological += logjump[j - 1] if left != right else logstay[j - 1]
            biological -= logjump[j - 1] if selectors[e, j - 1] != selectors[e, j] else logstay[j - 1]
        for k in range(offsets[i], offsets[i + 1]):
            e = outgoing[k]
            left = selectors[e, j - 1] ^ frames[i, j - 1] ^ new_left
            right = selectors[e, j] ^ frames[i, j] ^ new_right
            biological += logjump[j - 1] if left != right else logstay[j - 1]
            biological -= logjump[j - 1] if selectors[e, j - 1] != selectors[e, j] else logstay[j - 1]
    return biological, regularisation


@njit(cache=True)
def accept_supported_intervals(i, path, scaffold, frames, selectors, incoming, offsets, outgoing,
                               parents, logstay, logjump, phase_stay, phase_jump, error,
                               minimum_gain, minimum_biological_gain):
    """Apply the maximal intervals where ``path`` differs from vertex i's configuration that gain enough."""
    e0, e1 = incoming[i, 0], incoming[i, 1]
    old = frames[i].astype(np.int8).copy()
    if e0 >= 0:
        old += 2 * selectors[e0]
    if e1 >= 0:
        old += 4 * selectors[e1]
    total_gain, changed = 0., 0
    site = 0
    while site < len(path):
        if path[site] == old[site]:
            site += 1
            continue
        a = site
        meaningful_phase = False
        while site < len(path) and path[site] != old[site]:
            if (path[site] & 1) != (old[site] & 1) and scaffold[i, site, 0] != scaffold[i, site, 1]:
                meaningful_phase = True
            site += 1
        b = site
        biological, prior = interval_gain(i, a, b, path, scaffold, frames, selectors, incoming, offsets,
                                          outgoing, parents, logstay, logjump, phase_stay, phase_jump, error)
        gain = biological + prior
        if gain <= minimum_gain or (meaningful_phase and biological < minimum_biological_gain):
            continue
        total_gain += gain
        for j in range(a, b):
            f = path[j] & 1
            different = f ^ frames[i, j]
            changed += different
            for k in range(offsets[i], offsets[i + 1]):
                selectors[outgoing[k], j] ^= different
            if e0 >= 0:
                selectors[e0, j] = (path[j] >> 1) & 1
            if e1 >= 0:
                selectors[e1, j] = (path[j] >> 2) & 1
            frames[i, j] = f
    return total_gain, changed


@njit(cache=True)
def update_vertex(i, scaffold, frames, selectors, incoming, offsets, outgoing,
                  parents, logstay, logjump, phase_stay, phase_jump, error, minimum_gain,
                  minimum_biological_gain):
    """Re-decode vertex i (frame bit 0, incoming selector bits 1 and 2) by a factorised 8-state Viterbi."""
    sites = scaffold.shape[1]
    e0, e1 = incoming[i, 0], incoming[i, 1]
    match, mismatch = math.log(2.) + math.log1p(-error), math.log(2.) + math.log(error)
    scores = np.zeros(8, dtype=np.float64)
    values = np.empty(8, dtype=np.float64)
    origin = np.empty(8, dtype=np.int8)
    origins = np.empty(8, dtype=np.int8)
    back = np.zeros((sites, 8), dtype=np.int8)
    old_score = 0.
    for j in range(sites):
        trans_stay, trans_jump = 0., 0.
        if j:
            trans_stay, trans_jump = phase_stay[j - 1], phase_jump[j - 1]
            old_flip = frames[i, j] != frames[i, j - 1]
            for k in range(offsets[i], offsets[i + 1]):
                edge = outgoing[k]
                source_changed = (selectors[edge, j] != selectors[edge, j - 1]) ^ old_flip
                trans_stay += logjump[j - 1] if source_changed else logstay[j - 1]
                trans_jump += logstay[j - 1] if source_changed else logjump[j - 1]
            old_score += trans_jump if old_flip else trans_stay
            if e0 >= 0:
                old_score += logjump[j - 1] if selectors[e0, j] != selectors[e0, j - 1] else logstay[j - 1]
            if e1 >= 0:
                old_score += logjump[j - 1] if selectors[e1, j] != selectors[e1, j - 1] else logstay[j - 1]
        if j:
            for state in range(8):
                origin[state] = state
            # Factorised max-product transition: 3 binary axes, O(8 log 8).
            for bit in range(3):
                stay = trans_stay if bit == 0 else logstay[j - 1]
                jump = trans_jump if bit == 0 else logjump[j - 1]
                absent = (bit == 1 and e0 < 0) or (bit == 2 and e1 < 0)
                if absent:
                    stay, jump = 0., -np.inf
                for state in range(8):
                    other = state ^ (1 << bit)
                    a, b = scores[state] + stay, scores[other] + jump
                    if b > a:
                        values[state], origins[state] = b, origin[other]
                    else:
                        values[state], origins[state] = a, origin[state]
                scores, values = values, scores
                origin, origins = origins, origin
            back[j] = origin
        for state in range(8):
            f, s0, s1 = state & 1, (state >> 1) & 1, (state >> 2) & 1
            if (e0 < 0 and s0) or (e1 < 0 and s1):
                scores[state] = -np.inf
                continue
            emission = 0.
            if e0 >= 0:
                p = parents[e0]
                emission += copy_log(scaffold[p, j, s0 ^ frames[p, j]], scaffold[i, j, f], match, mismatch)
            if e1 >= 0:
                p = parents[e1]
                emission += copy_log(scaffold[p, j, s1 ^ frames[p, j]], scaffold[i, j, 1 ^ f], match, mismatch)
            scores[state] += emission
        if e0 >= 0:
            p = parents[e0]
            old_score += copy_log(scaffold[p, j, selectors[e0, j] ^ frames[p, j]],
                                  scaffold[i, j, frames[i, j]], match, mismatch)
        if e1 >= 0:
            p = parents[e1]
            old_score += copy_log(scaffold[p, j, selectors[e1, j] ^ frames[p, j]],
                                  scaffold[i, j, 1 ^ frames[i, j]], match, mismatch)
    state = int(np.argmax(scores))
    gain = scores[state] - old_score
    if gain <= minimum_gain:
        return 0., 0
    path = np.empty(sites, dtype=np.int8)
    for j in range(sites - 1, -1, -1):
        path[j] = state
        if j:
            state = back[j, state]
    # A DP sum and the same fixed path can differ slightly in evaluation order over long chromosomes. An
    # unchanged configuration is not a move, irrespective of the apparent floating-point score gain.
    different_configuration = False
    for j in range(sites):
        if ((path[j] & 1) != frames[i, j]
                or (e0 >= 0 and ((path[j] >> 1) & 1) != selectors[e0, j])
                or (e1 >= 0 and ((path[j] >> 2) & 1) != selectors[e1, j])):
            different_configuration = True
            break
    if not different_configuration:
        return 0., 0
    return accept_supported_intervals(i, path, scaffold, frames, selectors, incoming, offsets, outgoing,
                                      parents, logstay, logjump, phase_stay, phase_jump, error, minimum_gain,
                                      minimum_biological_gain)


@njit(cache=True, parallel=True)
def update_colour(vertices, scaffold, frames, selectors, incoming, offsets, outgoing,
                  parents, logstay, logjump, phase_stay, phase_jump, error, minimum_gain,
                  minimum_biological_gain):
    """Independent vertex updates of one colour class; returns the accepted gain of each vertex."""
    gains = np.zeros(len(vertices))
    for v in prange(len(vertices)):
        gains[v], _ = update_vertex(vertices[v], scaffold, frames, selectors, incoming, offsets, outgoing,
                                    parents, logstay, logjump, phase_stay, phase_jump, error, minimum_gain,
                                    minimum_biological_gain)
    return gains


def selector_log_terms(theta):
    """(log stay, log switch) of the polish's selector transitions; -inf where switching is impossible."""
    logstay, logjump = np.log1p(-theta), np.full_like(theta, -np.inf)
    np.log(theta, out=logjump, where=theta > 0)
    return logstay, logjump


def polish(scaffold, bins, components, parents, children, slots, meiosis_theta, config):
    """Polish the frames of two-parent children starting from the scaffold (module docstring)."""
    scaffold = np.ascontiguousarray(scaffold, dtype=np.int8)
    frames = np.zeros(scaffold.shape[:2], dtype=np.int8)
    bins = np.asarray(bins, dtype=np.int64)
    components = np.asarray(components, dtype=np.int64)
    incoming, offsets, outgoing, colours = pedigree_layout(len(scaffold), parents, children, slots)
    eligible = np.sum(incoming >= 0, axis=1) == 2
    theta = np.array(meiosis_theta, dtype=np.float64)
    theta[components[1:] != components[:-1]] = .5
    logstay, logjump = selector_log_terms(theta)
    selectors = initialize_selectors(scaffold, frames, parents, children, slots, logstay, logjump,
                                     config.polish_copy_error)
    phase_theta = phase_transitions(bins, components, config.phase_switch_probability, frames, selectors,
                                    config.polish_window_sites)
    phase_stay, phase_jump = selector_log_terms(phase_theta)
    sweep_updates = []
    converged = False
    for sweep in range(config.polish_max_sweeps):
        updates = 0
        for vertices in (colours if sweep % 2 == 0 else colours[::-1]):
            claim_threads()
            vertices = vertices[eligible[vertices]]
            if not len(vertices):
                continue
            gains = update_colour(vertices, scaffold, frames, selectors, incoming, offsets, outgoing, parents,
                                  logstay, logjump, phase_stay, phase_jump, config.polish_copy_error,
                                  config.polish_minimum_gain, config.polish_minimum_biological_gain)
            updates += int((gains > 0).sum())
        sweep_updates.append(updates)
        if updates == 0:
            converged = True
            break
    return PolishResult(frames, selectors, tuple(sweep_updates), converged)
