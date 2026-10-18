"""Genotype beliefs and Mendelian copy factors of the family model.

Ordered genotypes have four states per sample and site (2 * first-track allele + second-track allele). A copy
factor joins a child to its observed parents through two transmission selectors (one bit per parent: which
parental homologue the child's track copies), each copy erring with probability ``transmission_error``.
Factor work runs in 32-site tiles listing only (child, site) entries where some member is not a hard
homozygote; hard homozygote point masses keep fixed emissions. Messages are float32 log values updated with
damping in probability space.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


FACTOR_TILE_SITES = 32
BELIEF_TILE_SITES = 256


@njit(cache=True, inline="always")
def normalise_log(row):
    """Normalise a four-state log row in place (fixed four-term summation order)."""
    maximum = max(row[0], row[1], row[2], row[3])
    total = ((np.exp(row[0] - maximum) + np.exp(row[1] - maximum))
             + np.exp(row[2] - maximum)) + np.exp(row[3] - maximum)
    denominator = maximum + math.log(total)
    for state in range(4):
        row[state] -= denominator


@njit(cache=True, inline="always")
def cavity4(belief, message):
    """Normalised probabilities of a belief with one incoming log message removed.

    Impossible scaffold genotypes stay exactly zero.
    """
    a = np.float64(belief[0]) - np.float64(message[0])
    b = np.float64(belief[1]) - np.float64(message[1])
    c = np.float64(belief[2]) - np.float64(message[2])
    d = np.float64(belief[3]) - np.float64(message[3])
    maximum = max(a, b, c, d)
    a = math.exp(a - maximum) if a > -np.inf else 0.
    b = math.exp(b - maximum) if b > -np.inf else 0.
    c = math.exp(c - maximum) if c > -np.inf else 0.
    d = math.exp(d - maximum) if d > -np.inf else 0.
    total = ((a + b) + c) + d
    return a / total, b / total, c / total, d / total


@njit(cache=True, inline="always")
def fixed_homozygote(row):
    """A hard homozygote: exactly one of the two homozygous states is possible."""
    return (row[1] == -np.inf and row[2] == -np.inf and
            ((row[0] == -np.inf and row[3] > -np.inf) or (row[3] == -np.inf and row[0] > -np.inf)))


@njit(cache=True, parallel=True)
def fixed_nodes(base):
    """bool [N, L]: hard homozygotes of the base potentials."""
    n, l, _ = base.shape
    fixed = np.zeros((n, l), dtype=np.bool_)
    for i in prange(n):
        for j in range(l):
            fixed[i, j] = (base[i, j, 1] == -np.inf and base[i, j, 2] == -np.inf and
                           ((base[i, j, 0] > -np.inf and base[i, j, 3] == -np.inf) or
                            (base[i, j, 3] > -np.inf and base[i, j, 0] == -np.inf)))
    return fixed


@njit(cache=True, parallel=True)
def joint_beliefs(base, orientation, phase, parent_log, child_log, node_group,
                  offsets, adjacency, edge_child, phase_error):
    """Normalised log beliefs float32 [N, L, 4]: base potential x phase message x factor messages.

    The phase message ``phase`` (probability that the sample's frame matches its scaffold orientation) acts
    on the two heterozygous states of samples with a scaffold orientation, with error ``phase_error``.
    """
    samples, sites, _ = base.shape
    result = np.empty_like(base)
    tiles = (sites + BELIEF_TILE_SITES - 1) // BELIEF_TILE_SITES
    for task in prange(samples * tiles):
        sample = task % samples
        start = (task // samples) * BELIEF_TILE_SITES
        group = node_group[sample]
        for site in range(start, min(sites, start + BELIEF_TILE_SITES)):
            # Hard-conditioned homozygotes are exact point masses, whatever the incoming messages.
            if (np.isfinite(base[sample, site, 0]) and not np.isfinite(base[sample, site, 1])
                    and not np.isfinite(base[sample, site, 2]) and not np.isfinite(base[sample, site, 3])):
                result[sample, site, 0] = 0.
                for state in range(1, 4):
                    result[sample, site, state] = -np.inf
                continue
            if (np.isfinite(base[sample, site, 3]) and not np.isfinite(base[sample, site, 0])
                    and not np.isfinite(base[sample, site, 1]) and not np.isfinite(base[sample, site, 2])):
                result[sample, site, 3] = 0.
                for state in range(3):
                    result[sample, site, state] = -np.inf
                continue
            a = float(base[sample, site, 0])
            b = float(base[sample, site, 1])
            c = float(base[sample, site, 2])
            d = float(base[sample, site, 3])
            orient = orientation[sample, site]
            p = phase_error + (1 - 2 * phase_error) * phase[sample, site]
            a += math.log(.5)
            d += math.log(.5)
            if orient < 0:
                b += math.log(.5)
                c += math.log(.5)
            elif orient == 0:
                b += math.log(p)
                c += math.log(1 - p)
            else:
                b += math.log(1 - p)
                c += math.log(p)
            if group >= 0:
                a += child_log[group, site, 0]
                b += child_log[group, site, 1]
                c += child_log[group, site, 2]
                d += child_log[group, site, 3]
            for index in range(offsets[sample], offsets[sample + 1]):
                edge = adjacency[index]
                if edge_child[edge] != sample:
                    a += parent_log[edge, site, 0]
                    b += parent_log[edge, site, 1]
                    c += parent_log[edge, site, 2]
                    d += parent_log[edge, site, 3]
            maximum = max(a, b, c, d)
            normalizer = maximum + math.log(
                ((math.exp(a - maximum) + math.exp(b - maximum)) + math.exp(c - maximum)) + math.exp(d - maximum)
            )
            result[sample, site, 0] = a - normalizer
            result[sample, site, 1] = b - normalizer
            result[sample, site, 2] = c - normalizer
            result[sample, site, 3] = d - normalizer
    return result


@njit(cache=True, parallel=True)
def _count_factor_entries(fixed, gp, gc):
    sites = fixed.shape[1]
    tiles = (sites + FACTOR_TILE_SITES - 1) // FACTOR_TILE_SITES
    counts = np.zeros(tiles, dtype=np.int64)
    for tile in prange(tiles):
        for group in range(len(gc)):
            p0, p1 = gp[group]
            child = gc[group]
            for site in range(tile * FACTOR_TILE_SITES, min(sites, (tile + 1) * FACTOR_TILE_SITES)):
                if (not fixed[child, site] or (p0 >= 0 and not fixed[p0, site]) or
                        (p1 >= 0 and not fixed[p1, site])):
                    counts[tile] += 1
    return counts


@njit(cache=True, parallel=True)
def _fill_factor_entries(base, fixed, gp, gc, error, offsets):
    sites = fixed.shape[1]
    tiles = len(offsets) - 1
    group_index = np.empty(offsets[-1], dtype=np.int32)
    site_index = np.empty(offsets[-1], dtype=np.int32)
    emissions = np.empty((len(gc), sites, 4), dtype=np.float32)
    for tile in prange(tiles):
        cursor = offsets[tile]
        for group in range(len(gc)):
            p0, p1 = gp[group]
            child = gc[group]
            for site in range(tile * FACTOR_TILE_SITES, min(sites, (tile + 1) * FACTOR_TILE_SITES)):
                if (not fixed[child, site] or (p0 >= 0 and not fixed[p0, site]) or
                        (p1 >= 0 and not fixed[p1, site])):
                    group_index[cursor] = group
                    site_index[cursor] = site
                    cursor += 1
                else:
                    a = .5 if p0 < 0 else (1. if base[p0, site, 3] > -np.inf else 0.)
                    b = .5 if p1 < 0 else (1. if base[p1, site, 3] > -np.inf else 0.)
                    a = error + (1 - 2 * error) * a
                    b = error + (1 - 2 * error) * b
                    value = (1 - a) * (1 - b) if base[child, site, 0] > -np.inf else a * b
                    for state in range(4):
                        emissions[group, site, state] = value
    return offsets, group_index, site_index, emissions


def factor_support(base, gp, gc, error):
    """(tile offsets, group index, site index, selector emissions float32 [G, L, 4]) of the copy factors.

    Entries list (group, site) pairs with a member that is not a hard homozygote, by 32-site tile; the
    emissions of all-fixed pairs are constant and filled here. Sweeps overwrite the listed entries in place.
    """
    fixed = fixed_nodes(base)
    counts = _count_factor_entries(fixed, gp, gc)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    return _fill_factor_entries(base, fixed, gp, gc, error, offsets)


@njit(cache=True, parallel=True)
def copy_factor_sweep(belief, parent_log, child_log, segregation, group_parent,
                      group_edge, group_child, error, damping, reverse, support, dirty_tiles=None):
    """One sweep of every copy factor: selector emissions (in support's array) and damped messages.

    Within a tile the factor order alternates with ``reverse``; tiles run in parallel (distinct sites never
    share messages). ``damping == 0`` only recomputes the selector emissions. ``dirty_tiles`` (cleared by the
    caller) marks group tiles whose emissions changed. Returns (emissions, largest change per tile).
    """
    offsets, group_index, site_index, emissions = support
    tiles = len(offsets) - 1
    delta = np.zeros(tiles, dtype=np.float64)
    for tile in prange(tiles):
        parent_alt = np.empty((2, 2), dtype=np.float64)
        pm = np.empty((2, 4), dtype=np.float64)
        parent_fixed = np.empty(2, dtype=np.bool_)
        cm = np.empty(4, dtype=np.float64)
        for step in range(offsets[tile], offsets[tile + 1]):
            entry = offsets[tile + 1] - 1 - (step - offsets[tile]) if reverse else step
            group = group_index[entry]
            site = site_index[entry]
            old0, old1, old2, old3 = emissions[group, site]
            child = group_child[group]
            qchild = cavity4(belief[child, site], child_log[group, site])
            child_fixed = fixed_homozygote(belief[child, site])
            parent_alt[:, :] = .5
            parent_fixed[:] = True
            for slot in range(2):
                parent = group_parent[group, slot]
                edge = group_edge[group, slot]
                if parent < 0:
                    continue
                parent_fixed[slot] = fixed_homozygote(belief[parent, site])
                q = cavity4(belief[parent, site], parent_log[edge, site])
                parent_alt[slot, 0] = q[2] + q[3]
                parent_alt[slot, 1] = q[1] + q[3]
            pm[:, :] = 0.
            cm[:] = 0.
            if parent_fixed[0] and parent_fixed[1]:
                a = error + (1 - 2 * error) * parent_alt[0, 0]
                b = error + (1 - 2 * error) * parent_alt[1, 0]
                chance = ((1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b)
                total = 0.
                for genotype in range(4):
                    total += qchild[genotype] * chance[genotype]
                for state in range(4):
                    emissions[group, site, state] = total
                if damping > 0 and not child_fixed:
                    for genotype in range(4):
                        cm[genotype] = chance[genotype]
            else:
                for state in range(4):
                    s0, s1 = state >> 1, state & 1
                    a = error + (1 - 2 * error) * parent_alt[0, s0]
                    b = error + (1 - 2 * error) * parent_alt[1, s1]
                    chance = ((1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b)
                    total = 0.
                    for genotype in range(4):
                        total += qchild[genotype] * chance[genotype]
                    emissions[group, site, state] = total
                    if damping == 0:
                        continue
                    weight = segregation[group, site, state]
                    if not child_fixed:
                        for genotype in range(4):
                            cm[genotype] += weight * chance[genotype]
                    u0 = qchild[0] * (1 - b) + qchild[1] * b
                    u1 = qchild[2] * (1 - b) + qchild[3] * b
                    v0 = qchild[0] * (1 - a) + qchild[2] * a
                    v1 = qchild[1] * (1 - a) + qchild[3] * a
                    for genotype in range(4):
                        allele0 = (genotype >> (1 - s0)) & 1
                        allele1 = (genotype >> (1 - s1)) & 1
                        x = error + (1 - 2 * error) * allele0
                        y = error + (1 - 2 * error) * allele1
                        if not parent_fixed[0]:
                            pm[0, genotype] += weight * (u0 * (1 - x) + u1 * x)
                        if not parent_fixed[1]:
                            pm[1, genotype] += weight * (v0 * (1 - y) + v1 * y)
            if dirty_tiles is not None:
                if (emissions[group, site, 0] != old0 or emissions[group, site, 1] != old1
                        or emissions[group, site, 2] != old2 or emissions[group, site, 3] != old3):
                    dirty_tiles[group, tile] = True
            if damping == 0:
                continue
            if not child_fixed:
                cm /= np.sum(cm)
                for genotype in range(4):
                    old_log = float(child_log[group, site, genotype])
                    old = math.exp(old_log)
                    updated = (1 - damping) * old + damping * cm[genotype]
                    value = math.log(max(1e-30, updated))
                    child_log[group, site, genotype] = value
                    belief[child, site, genotype] += value - old_log
                    delta[tile] = max(delta[tile], abs(updated - old))
                normalise_log(belief[child, site])
            for slot in range(2):
                parent = group_parent[group, slot]
                edge = group_edge[group, slot]
                if parent < 0 or parent_fixed[slot]:
                    continue
                denominator = np.sum(pm[slot])
                for genotype in range(4):
                    old_log = float(parent_log[edge, site, genotype])
                    old = math.exp(old_log)
                    updated = (1 - damping) * old + damping * pm[slot, genotype] / denominator
                    value = math.log(max(1e-30, updated))
                    parent_log[edge, site, genotype] = value
                    belief[parent, site, genotype] += value - old_log
                    delta[tile] = max(delta[tile], abs(updated - old))
                normalise_log(belief[parent, site])
    return emissions, delta


@njit(cache=True, parallel=True)
def _count_selector_markers(fixed, gp):
    counts = np.zeros(len(gp), dtype=np.int64)
    for group in prange(len(gp)):
        p0, p1 = gp[group]
        for site in range(fixed.shape[1]):
            if (p0 >= 0 and not fixed[p0, site]) or (p1 >= 0 and not fixed[p1, site]):
                counts[group] += 1
    return counts


@njit(cache=True, parallel=True)
def _fill_selector_markers(fixed, gp, offsets, log_correlation, resets):
    indices = np.empty(offsets[-1], dtype=np.int32)
    theta = np.zeros(offsets[-1])
    for group in prange(len(gp)):
        p0, p1 = gp[group]
        cursor = offsets[group]
        previous = -1
        for site in range(fixed.shape[1]):
            if not ((p0 >= 0 and not fixed[p0, site]) or (p1 >= 0 and not fixed[p1, site])):
                continue
            indices[cursor] = site
            if previous >= 0:
                theta[cursor - 1] = (.5 if resets[site] != resets[previous] else
                                     -.5 * math.expm1(log_correlation[site] - log_correlation[previous]))
            previous = site
            cursor += 1
    return offsets, indices, theta


def selector_support(base, gp, theta):
    """(offsets, marker indices, transition probabilities) of every group's selector chain.

    A selector carries evidence only where some parent is not a hard homozygote; the other markers are
    integrated out exactly: two supported markers are joined by the composed switch probability
    (1 - prod(1 - 2 theta)) / 2 of the intervals between them (1/2 across a theta = 1/2 reset).
    """
    fixed = fixed_nodes(base)
    counts = _count_selector_markers(fixed, gp)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    resets = np.r_[0, np.cumsum(theta == .5)].astype(np.int64)
    logs = np.zeros(len(theta))
    np.log1p(-2 * theta, out=logs, where=theta < .5)
    log_correlation = np.r_[0., np.cumsum(logs)]
    return _fill_selector_markers(fixed, gp, offsets, log_correlation, resets)
