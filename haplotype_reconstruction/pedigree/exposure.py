"""Posterior-expected exposure of candidate parent edges and parent pairs (exposure gate input).

From the exact per-bin source marginals (unordered founder pairs of each sample), an edge child-parent
is "matched" in a bin with the expected probability that the child's pair shares an anchored founder
with the parent's, and "exposed" with the probability that the bin can test it (a match, or both
pairs fully anchored and disjoint). A panel pair (child, parent, parent) is likewise explained when
one parent can supply each child founder. Evidence-free samples contribute nothing; unanchored
named trajectories and BACKGROUND are never identity evidence.

Every quantity is a function of the unordered pair masses (S (S + 1) / 2 per sample and bin), which
replace the ordered marginals once the projection is built (half their size). The joint presence
features are formed inside the edge and pair loops with the arithmetic of the stored features of
92b3daf, so no further N x B x S (S + 1) / 2 arrays exist (base kept three, ~7.6 GB at N = 320,
K = 30). Cost O((E + T) B S^2 / 2).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _unit(value):
    return min(1.0, max(0.0, value))


@njit(cache=True, parallel=True)
def _pair_masses(posterior, pair_first, pair_second, pair_diagonal):
    """float64 [N, B, F]: unordered pair mass p[a, b] + p[b, a] (p[a, a] on the diagonal)."""
    n_samples, n_bins = posterior.shape[:2]
    n_features = len(pair_first)
    masses = np.empty((n_samples, n_bins, n_features), dtype=np.float64)
    for sample in prange(n_samples):
        for block in range(n_bins):
            for feature in range(n_features):
                first = pair_first[feature]
                second = pair_second[feature]
                mass = posterior[sample, block, first, second]
                mass += (1.0 - pair_diagonal[feature]) * posterior[sample, block, second, first]
                masses[sample, block, feature] = mass
    return masses


def pair_masses(posterior: np.ndarray) -> np.ndarray:
    """Unordered pair masses [N, B, S (S + 1) / 2] (upper-triangle order) of the source marginals."""
    pair_first, pair_second = np.triu_indices(posterior.shape[2])
    return _pair_masses(posterior, pair_first.astype(np.int64), pair_second.astype(np.int64),
                        (pair_first == pair_second).astype(np.float64))


@njit(cache=True, parallel=True)
def _presence(masses, feature_of, state_anchored, pair_anchored, available):
    """both [N, B]: mass of fully anchored distinct pairs; any / full presence [N, B, S]: mass of
    pairs containing an anchored founder (with any partner / with an anchored partner), minus the
    homozygous pair. ``feature_of`` [S, S] maps an unordered pair to its feature."""
    n_samples, n_bins, n_features = masses.shape
    n_states = state_anchored.shape[0]
    both = np.zeros((n_samples, n_bins), dtype=np.float64)
    any_presence = np.zeros((n_samples, n_bins, n_states), dtype=np.float64)
    full_presence = np.zeros((n_samples, n_bins, n_states), dtype=np.float64)
    for sample in prange(n_samples):
        if not available[sample]:
            continue
        for block in range(n_bins):
            for first in range(n_states):
                any_mass = 0.0
                full_mass = 0.0
                for other in range(n_states):
                    # p[first, other] + p[other, first], exactly (the diagonal pair twice).
                    pair = masses[sample, block, feature_of[first, other]]
                    ordered_mass = pair + pair if other == first else pair
                    any_mass += ordered_mass
                    full_mass += ordered_mass * state_anchored[other]
                diagonal_mass = masses[sample, block, feature_of[first, first]]
                anchored_first = state_anchored[first]
                any_presence[sample, block, first] = anchored_first * (any_mass - diagonal_mass)
                full_presence[sample, block, first] = anchored_first * (full_mass - diagonal_mass)
            for feature in range(n_features):
                both[sample, block] += masses[sample, block, feature] * pair_anchored[feature]
    return both, any_presence, full_presence


@njit(cache=True, parallel=True)
def _edges(masses, pair_first, pair_second, pair_has_anchor, pair_anchored,
           available, diagonal_required, edge_first, edge_second, both, any_presence, full_presence,
           interval_morgans):
    """Symmetric matched and exposed bins [N, N] of the requested edges (diagonal: self coverage), and their exposed
    map length [N, N] (off the diagonal): each interval between adjacent bins (``interval_morgans`` [B-1]) counts
    as far as the edge is exposed at both its ends."""
    n_samples, n_bins = both.shape
    n_features = len(pair_first)
    matched = np.zeros((n_samples, n_samples), dtype=np.float64)
    exposed = np.zeros((n_samples, n_samples), dtype=np.float64)
    exposed_morgans = np.zeros((n_samples, n_samples), dtype=np.float64)
    for child in prange(n_samples):
        if available[child] and diagonal_required[child]:
            diagonal = 0.0
            for block in range(n_bins):
                present = 0.0
                for feature in range(n_features):
                    present += masses[child, block, feature] * pair_has_anchor[feature]
                diagonal += _unit(present)
            matched[child, child] = diagonal
            exposed[child, child] = diagonal
    # One equal-sized parallel task per requested upper-triangle edge (no triangular tail).
    for edge in prange(len(edge_first)):
        child = edge_first[edge]
        parent = edge_second[edge]
        edge_match = 0.0
        edge_exposure = 0.0
        edge_morgans = 0.0
        previous = 0.0
        for block in range(n_bins):
            block_match = 0.0
            full_nonmatch = 0.0
            for feature in range(n_features):
                first = pair_first[feature]
                second = pair_second[feature]
                child_mass = masses[child, block, feature]
                # Joint presence of the parent: its presence of a homozygous pair, else the anchored
                # pair mass (equal to base's 0/1-weighted sum of the two).
                if first == second:
                    joint_any = any_presence[parent, block, first]
                    joint_full = full_presence[parent, block, first]
                else:
                    joint_any = masses[parent, block, feature] * pair_anchored[feature]
                    joint_full = joint_any
                match_probability = (any_presence[parent, block, first]
                                     + any_presence[parent, block, second] - joint_any)
                full_nonmatch_probability = pair_anchored[feature] * (
                    both[parent, block] - full_presence[parent, block, first]
                    - full_presence[parent, block, second] + joint_full)
                block_match += child_mass * match_probability
                full_nonmatch += child_mass * full_nonmatch_probability
            block_match = _unit(block_match)
            edge_match += block_match
            # A disjoint probability mass: inclusion-exclusion can cancel to a tiny negative value,
            # so keep its nonnegative bound and exposure never rounds below the match.
            block_exposure = _unit(block_match + max(0.0, full_nonmatch))
            edge_exposure += block_exposure
            if block > 0:
                # min(): the larger joint probability of exposure at both ends the two marginals allow.
                edge_morgans += min(previous, block_exposure) * interval_morgans[block - 1]
            previous = block_exposure
        matched[child, parent] = edge_match
        matched[parent, child] = edge_match
        exposed[child, parent] = edge_exposure
        exposed[parent, child] = edge_exposure
        exposed_morgans[child, parent] = edge_morgans
        exposed_morgans[parent, child] = edge_morgans
    return matched, exposed, exposed_morgans


@njit(cache=True, parallel=True)
def _pairs(masses, pair_first, pair_second, pair_anchored, available, trios, both,
           any_presence, full_presence, interval_morgans):
    """Explained and exposed bins [T] of the panel rows, and their exposed map length [T] (as ``_edges``)."""
    n_bins = both.shape[1]
    n_features = len(pair_first)
    explained = np.zeros(len(trios), dtype=np.float64)
    exposed = np.zeros(len(trios), dtype=np.float64)
    exposed_morgans = np.zeros(len(trios), dtype=np.float64)
    for row in prange(len(trios)):
        child = int(trios[row, 0])
        parent1 = int(trios[row, 1])
        parent2 = int(trios[row, 2])
        if not available[child] or not available[parent1] or not available[parent2]:
            continue
        previous = 0.0
        for block in range(n_bins):
            block_explained = 0.0
            full_nonexplained = 0.0
            for feature in range(n_features):
                first = pair_first[feature]
                second = pair_second[feature]
                child_mass = masses[child, block, feature]
                if first == second:
                    joint_any1 = any_presence[parent1, block, first]
                    joint_any2 = any_presence[parent2, block, first]
                    joint_full1 = full_presence[parent1, block, first]
                    joint_full2 = full_presence[parent2, block, first]
                else:
                    joint_any1 = masses[parent1, block, feature] * pair_anchored[feature]
                    joint_any2 = masses[parent2, block, feature] * pair_anchored[feature]
                    joint_full1 = joint_any1
                    joint_full2 = joint_any2
                assignment = (any_presence[parent1, block, first] * any_presence[parent2, block, second]
                              + any_presence[parent1, block, second] * any_presence[parent2, block, first]
                              - joint_any1 * joint_any2)
                block_explained += child_mass * assignment
                full_assignment = (
                    full_presence[parent1, block, first] * full_presence[parent2, block, second]
                    + full_presence[parent1, block, second] * full_presence[parent2, block, first]
                    - joint_full1 * joint_full2)
                full_nonexplained += (child_mass * pair_anchored[feature]
                                      * (both[parent1, block] * both[parent2, block] - full_assignment))
            block_explained = _unit(block_explained)
            explained[row] += block_explained
            block_exposure = _unit(block_explained + max(0.0, full_nonexplained))
            exposed[row] += block_exposure
            if block > 0:
                exposed_morgans[row] += min(previous, block_exposure) * interval_morgans[block - 1]
            previous = block_exposure
    return explained, exposed, exposed_morgans


def expected_structure(masses: np.ndarray, anchored_states: np.ndarray, required_edges: np.ndarray,
                       trios: np.ndarray, available: np.ndarray, claim_threads=lambda: None,
                       interval_morgans: np.ndarray | None = None):
    """(edge matched [N, N], edge exposed [N, N], pair explained [T], pair exposed [T]) in bins, then (edge exposed
    [N, N], pair exposed [T]) in measured map length.

    ``masses`` float64 [N, B, S (S + 1) / 2] unordered pair masses (``pair_masses``);
    ``required_edges`` bool [N, N] symmetric (edges to evaluate; the diagonal requests self
    coverage); ``available`` bool [N] gates evidence-free samples to zero; ``interval_morgans``
    float64 [B-1] the measured map length between adjacent bins (default none): an interval counts
    as far as the edge (or pair) is exposed at both its ends, so an exposed island counts its own
    length, however many bins it holds.
    """
    n_states = anchored_states.shape[0]
    pair_first, pair_second = np.triu_indices(n_states)
    feature_of = np.empty((n_states, n_states), dtype=np.int64)
    feature_of[pair_first, pair_second] = np.arange(len(pair_first))
    feature_of[pair_second, pair_first] = np.arange(len(pair_first))
    pair_first = np.ascontiguousarray(pair_first, dtype=np.int64)
    pair_second = np.ascontiguousarray(pair_second, dtype=np.int64)
    state_anchored = np.ascontiguousarray(anchored_states, dtype=np.float64)
    pair_anchored = np.ascontiguousarray(state_anchored[pair_first] * state_anchored[pair_second])
    pair_has_anchor = np.ascontiguousarray(
        np.maximum(state_anchored[pair_first], state_anchored[pair_second]))
    edge_first, edge_second = np.nonzero(np.triu(required_edges & available[:, None]
                                                 & available[None, :], k=1))
    n_bins = masses.shape[1]
    intervals = (np.zeros(max(n_bins - 1, 0)) if interval_morgans is None
                 else np.ascontiguousarray(interval_morgans, dtype=np.float64))
    if len(intervals) != max(n_bins - 1, 0):
        raise ValueError("interval_morgans needs one length per pair of adjacent bins")
    claim_threads()
    both, any_presence, full_presence = _presence(masses, feature_of, state_anchored, pair_anchored,
                                                  available)
    claim_threads()
    matched, exposed, exposed_morgans = _edges(masses, pair_first, pair_second, pair_has_anchor,
                                               pair_anchored, available,
                                               np.ascontiguousarray(np.diag(required_edges)),
                                               np.ascontiguousarray(edge_first, dtype=np.int64),
                                               np.ascontiguousarray(edge_second, dtype=np.int64),
                                               both, any_presence, full_presence, intervals)
    claim_threads()
    explained, pair_exposed, pair_morgans = _pairs(masses, pair_first, pair_second, pair_anchored,
                                                   available, np.ascontiguousarray(trios), both,
                                                   any_presence, full_presence, intervals)
    return matched, exposed, explained, pair_exposed, exposed_morgans, pair_morgans
