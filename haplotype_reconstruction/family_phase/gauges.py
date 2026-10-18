"""Joint root/children gauge moves of the family solver.

A root's phase chain and its children's selector chains share a gauge: relabelling the root's homologues from a
boundary on flips the root's frame and every child's selector bit for that parent. A move is made at a single
boundary when it raises the joint MAP score of the root's frame path and its children's selector paths and at
least ``minimum_support`` children change selector there (one child cannot tell a root frame change from its own
crossover).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..maths import chains


@njit(cache=True, parallel=True)
def _selector_paths(emissions, theta, groups, result):
    for job in prange(len(groups)):
        group = groups[job]
        result[group] = chains.selector_viterbi(emissions[group], theta)


def selector_paths(emissions, theta, groups):
    """MAP selector paths int8 [G, L] of ``groups`` (other rows are not written and must not be read)."""
    paths = np.empty(emissions.shape[:2], dtype=np.int8)
    if len(groups):
        _selector_paths(emissions, theta, np.asarray(groups, dtype=np.int64), paths)
    return paths


@njit(cache=True)
def best_gauge_flips(phase_path, child_paths, phase_theta, theta, anchor, minimum_support):
    """Cumulative single-boundary moves of one root (int8 relabel mask over sites, gauge-fixed at the anchor)."""
    n = len(phase_path)
    flip = np.zeros(n, dtype=np.int8)
    for i in range(n - 1):
        # A zero-probability phase boundary cannot be crossed by this move.
        if phase_theta[i] <= 0:
            flip[i + 1] = flip[i]
            continue
        phase_changed = phase_path[i] != phase_path[i + 1]
        phase_odds = math.log(phase_theta[i] / (1 - phase_theta[i]))
        improvement = (-1. if phase_changed else 1.) * phase_odds
        meiosis_odds = math.log(theta[i] / (1 - theta[i]))
        supporting = 0
        for child in range(len(child_paths)):
            changed = child_paths[child, i] != child_paths[child, i + 1]
            if changed:
                supporting += 1
            improvement += (-1. if changed else 1.) * meiosis_odds
        move = improvement > 1e-6 and supporting >= minimum_support
        flip[i + 1] = flip[i] ^ int(move)
    gauge = flip[anchor] if anchor >= 0 else flip[0]
    for i in range(n):
        flip[i] ^= gauge
    return flip


@njit(cache=True, parallel=True)
def root_mode_flips(belief, orientation, phase, parents, edge_group, edge_slot,
                    roots, anchors, phase_theta, theta, selector_paths, phase_error,
                    frozen, frozen_theta, minimum_support):
    """Relabel masks int8 [N, L] of every root (zero for other samples)."""
    samples, sites, _ = belief.shape
    flips = np.zeros((samples, sites), dtype=np.int8)
    # Schedule the root chains themselves: contiguous root rows would otherwise occupy only a few static
    # worker chunks when most samples are offspring.
    active_roots = np.flatnonzero(roots)
    for index in prange(len(active_roots)):
        root = active_roots[index]
        outgoing = np.where(parents == root)[0]
        if len(outgoing) == 0:
            continue
        emissions = np.empty((sites, 2), dtype=np.float64)
        for site in range(sites):
            q = np.empty(4, dtype=np.float64)
            orient = orientation[root, site]
            current = phase_error + (1 - 2 * phase_error) * phase[root, site]
            for state in range(4):
                if orient < 0 or state == 0 or state == 3:
                    factor = .5
                else:
                    factor = current if state == (1 if orient == 0 else 2) else 1 - current
                q[state] = math.exp(float(belief[root, site, state])) / factor
            q /= np.sum(q)
            if orient < 0:
                emissions[site, 0] = .5
                emissions[site, 1] = .5
            else:
                matching = 1 if orient == 0 else 2
                opposite = 3 - matching
                emissions[site, 0] = .5 * (q[0] + q[3]) + (1 - phase_error) * q[matching] + phase_error * q[opposite]
                emissions[site, 1] = .5 * (q[0] + q[3]) + phase_error * q[matching] + (1 - phase_error) * q[opposite]
        root_theta = frozen_theta if frozen[root] else phase_theta
        path = chains.binary_viterbi(emissions, root_theta, anchors[root])
        child_paths = np.empty((len(outgoing), sites), dtype=np.int8)
        for row in range(len(outgoing)):
            edge = outgoing[row]
            group = edge_group[edge]
            shift = 1 - edge_slot[edge]
            for site in range(sites):
                child_paths[row, site] = (selector_paths[group, site] >> shift) & 1
        flips[root] = best_gauge_flips(path, child_paths, root_theta, theta, anchors[root], minimum_support)
    return flips


@njit(cache=True, parallel=True)
def apply_root_flips(flips, phase, parent_log, segregation, parents, group_parent):
    """Relabel roots' homologues where ``flips`` is 1: phase messages, parent messages, selector cavities."""
    samples, sites = phase.shape
    for sample in prange(samples):
        for site in range(sites):
            if flips[sample, site]:
                phase[sample, site] = 1 - phase[sample, site]
    for edge in prange(len(parents)):
        parent = parents[edge]
        for site in range(sites):
            if flips[parent, site]:
                value = parent_log[edge, site, 1]
                parent_log[edge, site, 1] = parent_log[edge, site, 2]
                parent_log[edge, site, 2] = value
    # Both roots may parent the same child: apply their commuting selector-bit permutations once per child.
    for group in prange(len(group_parent)):
        p0, p1 = group_parent[group, 0], group_parent[group, 1]
        for site in range(sites):
            permutation = (2 * flips[p0, site] if p0 >= 0 else 0) + (flips[p1, site] if p1 >= 0 else 0)
            if permutation:
                previous = segregation[group, site].copy()
                for state in range(4):
                    segregation[group, site, state] = previous[state ^ permutation]
