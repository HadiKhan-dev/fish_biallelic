"""Proposal generators of the reversible search (each move proposes start panels; the
fixed-K fitter refits them and the cavity score decides).

* K = 1 data seeds: HDBSCAN clusters of samples by expected genotype agreement, each pooled
  into one row (fallback: the most decisive sample's dosage row).
* Deaths: every one-row deletion.
* Gauge rewires: for a maximum cut of the assignment graph (rows linked by samples assigned to
  both), flip the sites where both sides are internally constant and differ: per site to the
  better allele for the samples within one side ("sitewise"), and all such sites ("opposite").
* Data births: K + 1 starts from the samples with the largest excess NLL over their sitewise
  genotype oracle (subtraction and dosage seeds).
* Soft births: K + 1 starts from soft-residual consensus rows (residual_births.py).
* Birth-delete replacements: a birth followed by deleting any row (same K).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..execution import claim_threads
from .fixed_k.fit import FIT_MAX_THREADS, fit_one, fit_panels, fit_starts
from .panel import canonical_solution, deduplicate, has_distinct_rows, panel_key
from .residual_births import soft_candidates

_TINY = np.finfo(np.float64).tiny
_NLL_TOLERANCE = 1e-10
# Maximum-cut results memoised per block search (an exact memo of a pure function).
MAX_CUT_CACHE_ENTRIES = 256


# ---------------------------------------------------------------------------------------------
# K = 1 data seeds
# ---------------------------------------------------------------------------------------------
def _soft_cluster_seeds(likelihoods, n_seeds, min_cluster_size):
    """Up to n_seeds pooled binary rows from HDBSCAN clusters of the samples (distance =
    max agreement - expected per-site genotype agreement), largest clusters first."""
    import hdbscan
    if likelihoods.shape[0] < int(min_cluster_size):
        return []
    likelihoods = np.ascontiguousarray(likelihoods, dtype=np.float64)
    n_sites = likelihoods.shape[1]
    # Expected agreement S[i, j] = mean over sites of sum_g P_i(g) P_j(g), as three BLAS Gram
    # matrices (their L-axis reduction order is BLAS-internal; only HDBSCAN reads them).
    similarity = np.zeros((likelihoods.shape[0], likelihoods.shape[0]), dtype=np.float64)
    for genotype in range(3):
        channel = np.ascontiguousarray(likelihoods[:, :, genotype])
        similarity += channel @ channel.T
    similarity /= float(n_sites)
    distance = similarity.max() - similarity
    np.fill_diagonal(distance, 0.0)
    labels = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=int(min_cluster_size)).fit(
        np.ascontiguousarray(distance, dtype=np.float64)).labels_
    clusters = [np.flatnonzero(labels == label) for label in np.unique(labels) if label != -1]
    clusters.sort(key=lambda members: -len(members))
    alt_fraction = 0.5 * likelihoods[:, :, 1] + likelihoods[:, :, 2]
    return [(alt_fraction[members].mean(axis=0) > 0.5).astype(np.int64)
            for members in clusters[:n_seeds]]


def _dosage_seed(likelihoods, sample):
    """The sample's argmax dosage as a row: dosage 2 -> 1, dosage 1 -> 1 where the population
    ALT fraction exceeds 1/2, else 0."""
    dosage = likelihoods[sample].argmax(axis=1)
    population_alt = likelihoods[..., 1].mean(axis=0) * 0.5 + likelihoods[..., 2].mean(axis=0)
    row = np.zeros(likelihoods.shape[1], dtype=np.int64)
    row[dosage == 2] = 1
    row[(dosage == 1) & (population_alt > 0.5)] = 1
    return row


def initial_solutions(workspace, likelihoods, active, config):
    """The K = 1 search starts: fitted soft-cluster seeds (of the samples with depth), at most
    ``data_start_beam_width`` distinct ones by NLL."""
    seed_likelihoods = likelihoods if np.all(active) else np.ascontiguousarray(likelihoods[active])
    search = config.search
    seeds = _soft_cluster_seeds(seed_likelihoods, search.data_seed_count,
                                search.seed_min_cluster_size)
    if not seeds:
        decisiveness = seed_likelihoods.max(axis=2).sum(axis=1)
        seeds = [_dosage_seed(seed_likelihoods, int(decisiveness.argmax()))]
    fits = fit_starts(workspace, [seed[None, :] for seed in seeds], config.model.max_iterations)
    solutions = [canonical_solution(fit, workspace) for fit in fits]
    return deduplicate(solutions, search.data_start_beam_width)


# ---------------------------------------------------------------------------------------------
# Deaths and gauge rewires
# ---------------------------------------------------------------------------------------------
def death_starts(solution):
    """Every one-row deletion of a fitted panel (K >= 2)."""
    if solution.k <= 1:
        return ()
    return tuple(np.ascontiguousarray(np.delete(solution.haplotypes, index, axis=0))
                 for index in range(solution.k))


@njit(cache=True, nogil=True)
def assignment_graph(assignments, k):
    """Symmetric weights: the number of samples assigned to rows (i, j), i != j."""
    weights = np.zeros((k, k), dtype=np.int64)
    for sample in range(assignments.shape[0]):
        first = assignments[sample, 0]
        second = assignments[sample, 1]
        if first < k and second < k and first != second:
            weights[first, second] += 1
            weights[second, first] += 1
    return weights


@njit(cache=True, nogil=True)
def _exact_cut_scores(weights):
    """Cut weight of every bipartition with vertex 0 on the first side, by Gray-code updates
    (O(K) per candidate); entry (bits - 1) holds the cut whose other side is ``bits``."""
    k = len(weights)
    n_candidates = (1 << (k - 1)) - 1
    cross_scores = np.empty(n_candidates, dtype=np.int64)
    side = np.zeros(k, dtype=np.bool_)
    cross = 0
    previous_gray = 0
    for traversal in range(1, 1 << (k - 1)):
        gray = traversal ^ (traversal >> 1)
        changed = gray ^ previous_gray
        bit = 0
        while (changed >> bit) != 1:
            bit += 1
        vertex = bit + 1
        delta = 0
        old_side = side[vertex]
        for other in range(k):
            if other == vertex:
                continue
            if old_side == side[other]:
                delta += weights[vertex, other]
            else:
                delta -= weights[vertex, other]
        side[vertex] = not old_side
        cross += delta
        cross_scores[gray - 1] = cross
        previous_gray = gray
    return cross_scores


@njit(cache=True, nogil=True)
def _component_exact_cuts(weights, max_ties):
    """Maximum cuts by independent components: bipartite components by 2-colouring,
    the others by exhaustive enumeration. Cuts are integer codes with vertex order as most to
    least significant bit; the smallest codes are kept (byte-lexicographic tie rule), at most
    ``max_ties`` non-empty cuts."""
    k = len(weights)
    component = np.full(k, -1, dtype=np.int64)
    color = np.zeros(k, dtype=np.bool_)
    vertices = np.empty((k, k), dtype=np.int64)
    sizes = np.zeros(k, dtype=np.int64)
    bipartite = np.ones(k, dtype=np.bool_)
    count = 0
    for root in range(k):
        if component[root] >= 0:
            continue
        component[root] = count
        vertices[count, 0] = root
        sizes[count] = 1
        head = 0
        while head < sizes[count]:
            vertex = vertices[count, head]
            head += 1
            for other in range(k):
                if weights[vertex, other] == 0:
                    continue
                if component[other] < 0:
                    component[other] = count
                    color[other] = not color[vertex]
                    vertices[count, sizes[count]] = other
                    sizes[count] += 1
                elif color[other] == color[vertex]:
                    bipartite[count] = False
        count += 1
    # One extra permits omission of the all-False empty cut at the end.
    limit = max_ties + 1
    combined = np.zeros(1, dtype=np.int64)
    for c in range(count):
        members = np.sort(vertices[c, :sizes[c]])
        mask = np.int64(0)
        for vertex in members:
            mask |= np.int64(1) << (k - 1 - vertex)
        if bipartite[c]:
            code = np.int64(0)
            for vertex in members:
                if color[vertex]:
                    code |= np.int64(1) << (k - 1 - vertex)
            choices = np.empty(1 if c == 0 else 2, dtype=np.int64)
            choices[0] = code
            if c > 0:
                choices[1] = code ^ mask
        else:
            m = len(members)
            local = np.empty((m, m), dtype=np.int64)
            for i in range(m):
                for j in range(m):
                    local[i, j] = weights[members[i], members[j]]
            cross = _exact_cut_scores(local)
            best = np.max(cross)
            winners = np.flatnonzero(cross == best)
            choices = np.empty(len(winners) * (1 if c == 0 else 2), dtype=np.int64)
            cursor = 0
            for index in winners:
                bits = index + 1
                code = np.int64(0)
                for j in range(1, m):
                    if bits & (1 << (j - 1)):
                        code |= np.int64(1) << (k - 1 - members[j])
                choices[cursor] = code
                cursor += 1
                if c > 0:
                    choices[cursor] = code ^ mask
                    cursor += 1
        choices = np.sort(choices)[:limit]
        joined = np.empty(len(combined) * len(choices), dtype=np.int64)
        cursor = 0
        for previous in combined:
            for choice in choices:
                joined[cursor] = previous | choice
                cursor += 1
        combined = np.sort(joined)[:limit]
    combined = combined[combined != 0][:max_ties]
    result = np.zeros((len(combined), k), dtype=np.bool_)
    for row in range(len(combined)):
        for vertex in range(k):
            result[row, vertex] = bool(combined[row] & (np.int64(1) << (k - 1 - vertex)))
    return result


@njit(cache=True, nogil=True)
def _cut_score(weights, side):
    cross = 0
    within = 0
    for first in range(len(weights) - 1):
        for second in range(first + 1, len(weights)):
            if side[first] != side[second]:
                cross += weights[first, second]
            else:
                within += weights[first, second]
    return cross, within


@njit(cache=True, nogil=True)
def _local_search_cut(weights, initial, order):
    """Single-vertex moves in ``order`` while one improves the cut."""
    side = initial.copy()
    while True:
        changed = False
        for position in range(len(order)):
            index = order[position]
            old_side = side[index]
            delta = 0
            for other in range(len(weights)):
                if other == index:
                    continue
                if old_side == side[other]:
                    delta += weights[index, other]
                else:
                    delta -= weights[index, other]
            if delta > 0:
                side[index] = not old_side
                changed = True
        if not changed:
            return side


class MaximumCuts:
    """Deterministic maximum-weight bipartitions of assignment graphs (memoised per block).

    Vertex 0 stays on side False (complement cuts are one cut). Exact by components up to
    ``exact_max_k`` vertices; above, degree-ordered local searches from every singleton and an
    alternating start (``heuristic`` reports that). At most ``max_ties`` equal optima, in
    lexicographic bit order.
    """

    def __init__(self, exact_max_k: int, max_ties: int):
        self.exact_max_k = int(exact_max_k)
        self.max_ties = int(max_ties)
        self._cache: OrderedDict = OrderedDict()

    def heuristic(self, k: int) -> bool:
        return k > self.exact_max_k

    def partitions(self, weights, max_ties=None):
        """Up to ``max_ties`` (default the configured number) maximum cuts of ``weights``."""
        max_ties = self.max_ties if max_ties is None else int(max_ties)
        matrix = np.ascontiguousarray(weights, dtype=np.int64)
        if len(matrix) < 2:
            return ()
        key = (len(matrix), max_ties, matrix.tobytes())
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        k = len(matrix)
        if k <= self.exact_max_k:
            result = tuple(side.copy() for side in _component_exact_cuts(matrix, max_ties))
        else:
            order = np.asarray(sorted(range(1, k), key=lambda index: (-int(np.sum(matrix[index])),
                                                                      index)), dtype=np.int64)
            candidates = []
            for index in range(1, k):
                side = np.zeros(k, dtype=bool)
                side[index] = True
                candidates.append(_local_search_cut(matrix, side, order))
            alternating = np.zeros(k, dtype=bool)
            alternating[order[::2]] = True
            candidates.append(_local_search_cut(matrix, alternating, order))
            unique: dict[bytes, np.ndarray] = {}
            for side in candidates:
                if np.any(side):
                    unique.setdefault(side.tobytes(), side)
            scored = [(_cut_score(matrix, side), side.tobytes(), side) for side in unique.values()]
            if scored:
                best_cross = max(item[0][0] for item in scored)
                best_within = min(item[0][1] for item in scored if item[0][0] == best_cross)
                winners = sorted((item for item in scored if item[0] == (best_cross, best_within)),
                                 key=lambda item: item[1])
                result = tuple(item[2].copy() for item in winners[:max_ties])
            else:
                result = ()
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > MAX_CUT_CACHE_ENTRIES:
            self._cache.popitem(last=False)
        return result


def _gauge_sites(haplotypes, side):
    """Sites where both sides of the cut are internally constant and differ."""
    first_side = np.flatnonzero(~side)
    second_side = np.flatnonzero(side)
    if len(first_side) == 0 or len(second_side) == 0:
        return np.zeros(haplotypes.shape[1], dtype=bool)
    first_anchor = haplotypes[first_side[0]]
    second_anchor = haplotypes[second_side[0]]
    first_constant = np.all(haplotypes[first_side] == first_anchor[None, :], axis=0)
    second_constant = np.all(haplotypes[second_side] == second_anchor[None, :], axis=0)
    return first_constant & second_constant & (first_anchor != second_anchor)


def gauge_starts(solution, likelihoods, partitions):
    """Gauge-rewire starts of a fitted panel for the given cuts (sitewise, then opposite per
    cut), skipping panels already proposed or equal to the parent."""
    haplotypes = solution.haplotypes
    assignments = solution.assignments
    k = solution.k
    first = assignments[:, 0]
    second = assignments[:, 1]
    real = (first < k) & (second < k)
    real_indices = np.flatnonzero(real)
    starts = []
    seen = {solution.key}
    for raw_side in partitions:
        side = np.asarray(raw_side, dtype=bool)
        sites = _gauge_sites(haplotypes, side)
        if not np.any(sites):
            continue
        same_side = real.copy()
        same_side[real_indices] = side[first[real_indices]] == side[second[real_indices]]
        resolving = np.flatnonzero(same_side)
        sitewise = haplotypes.copy()
        for site in np.flatnonzero(sites):
            if len(resolving) == 0:
                continue
            dosage = haplotypes[first[resolving], site] + haplotypes[second[resolving], site]
            current = -float(np.sum(np.log(np.maximum(likelihoods[resolving, site, dosage], _TINY))))
            opposite = -float(np.sum(np.log(np.maximum(
                likelihoods[resolving, site, 2 - dosage], _TINY))))
            if opposite < current - _NLL_TOLERANCE:
                sitewise[:, site] = 1 - sitewise[:, site]
        key = panel_key(sitewise)
        if key not in seen:
            seen.add(key)
            starts.append(sitewise)
        opposite_panel = haplotypes.copy()
        opposite_panel[:, sites] = 1 - opposite_panel[:, sites]
        key = panel_key(opposite_panel)
        if key not in seen:
            seen.add(key)
            starts.append(opposite_panel)
    return starts


# ---------------------------------------------------------------------------------------------
# Data births
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GrowthInputs:
    """Evidence-only inputs of data births (one block): the sitewise genotype-oracle NLL and the
    decisiveness of every sample over its observed sites, argmax dosages, a dosage seed row per
    sample (dosage 2 -> 1; dosage 1 or unobserved -> 1 where the observed population ALT
    fraction exceeds 1/2), and the samples with depth."""

    oracle_nll: np.ndarray
    decisiveness: np.ndarray
    dosage: np.ndarray
    seed_rows: np.ndarray
    active: np.ndarray


def growth_inputs(likelihoods, observed) -> GrowthInputs:
    maximum = np.max(likelihoods, axis=2)
    dosage = np.argmax(likelihoods, axis=2)
    active = np.any(observed, axis=1)
    oracle_nll = -np.sum(np.where(observed, np.log(np.maximum(maximum, _TINY)), 0.0), axis=1)
    decisiveness = np.sum(np.where(observed, maximum, 0.0), axis=1)
    alt_fraction = 0.5 * likelihoods[..., 1] + likelihoods[..., 2]
    observed_count = np.sum(observed, axis=0)
    population_alt = np.divide(np.sum(np.where(observed, alt_fraction, 0.0), axis=0),
                               observed_count, out=np.full(likelihoods.shape[1], 0.5),
                               where=observed_count > 0)
    seed_rows = np.zeros_like(dosage)
    seed_rows[observed & (dosage == 2)] = 1
    ambiguous = (~observed) | (dosage == 1)
    seed_rows[ambiguous & (population_alt[None, :] > 0.5)] = 1
    return GrowthInputs(oracle_nll, decisiveness, dosage, seed_rows, active)


def data_births(workspace, solution, growth: GrowthInputs, max_iterations):
    """K + 1 children of a panel from its most misfit samples.

    The parent is first refitted (unless the workspace certified it a fixed point). Samples
    with depth are ranked by excess NLL over their genotype oracle (then decisiveness, then
    index); each contributes K subtraction seeds clip(dosage - row, 0, 1) and its dosage seed
    row. Novel seeds (at most max(16, 2(K + 1)) starts) are fitted in waves of K + 1 and
    consumed in order until K + 1 distinct complete children exist. If none is complete, the
    assignment-only fits of the attempted starts with distinct rows are returned instead
    (deduplicated, at most K + 1): the requested K without moving an unsupported row.
    """
    claim_threads(max_threads=FIT_MAX_THREADS)
    if solution.fixed_point and workspace.certifies_fixed_point(solution.haplotypes):
        parent = solution
    else:
        parent = canonical_solution(fit_one(workspace, solution.haplotypes, max_iterations),
                                    workspace)
    excess = np.maximum(parent.per_sample_cost - growth.oracle_nll, 0.0)
    order = sorted(np.flatnonzero(growth.active).tolist(), key=lambda sample: (
        -float(excess[sample]), -float(growth.decisiveness[sample]), int(sample)))
    target = parent.k + 1
    maximum_attempts = max(16, 2 * target)
    existing = {np.ascontiguousarray(row, dtype=np.int8).tobytes() for row in parent.haplotypes}
    attempted: set[bytes] = set()
    starts = []
    for sample in order:
        seeds = [np.clip(growth.dosage[sample] - parent.haplotypes[index], 0, 1).astype(np.int64)
                 for index in range(parent.k)]
        seeds.append(growth.seed_rows[sample])
        for seed in seeds:
            row = np.ascontiguousarray(seed, dtype=np.int64)
            key = row.astype(np.int8).tobytes()
            if key in existing or key in attempted:
                continue
            attempted.add(key)
            starts.append(np.vstack([parent.haplotypes, row[None, :]]))
            if len(starts) >= maximum_attempts:
                break
        if len(starts) >= maximum_attempts:
            break
    children = []
    child_keys = set()
    attempts = 0
    for wave_start in range(0, len(starts), target):
        stop = False
        for fit in fit_starts(workspace, starts[wave_start:wave_start + target], max_iterations):
            child = canonical_solution(fit, workspace)
            attempts += 1
            if has_distinct_rows(child):
                children.append(child)
                child_keys.add(child.key)
            if len(child_keys) >= target or attempts >= maximum_attempts:
                stop = True
                break
        if stop:
            break
    if children:
        return tuple(children)
    fallbacks = [canonical_solution(fit, workspace)
                 for fit in fit_starts(workspace, starts[:attempts], 0)]
    return deduplicate([child for child in fallbacks if has_distinct_rows(child)], target)


# ---------------------------------------------------------------------------------------------
# Soft births and birth-delete replacements
# ---------------------------------------------------------------------------------------------
def _hardened_rows(soft, likelihoods):
    """Binary rows of soft candidates (> 1/2 is ALT; within 1e-12 of 1/2 the population ALT
    fraction decides), distinct, in candidate order."""
    marginal = np.mean(0.5 * likelihoods[:, :, 1] + likelihoods[:, :, 2], axis=0)
    binary = soft > 0.5
    ties = np.isclose(soft, 0.5, rtol=0.0, atol=1e-12)
    binary[ties] = np.broadcast_to(marginal >= 0.5, binary.shape)[ties]
    unique: dict[bytes, np.ndarray] = {}
    for row in np.asarray(binary, dtype=np.int64):
        row = np.ascontiguousarray(row, dtype=np.int64)
        unique.setdefault(row.astype(np.int8).tobytes(), row)
    return tuple(unique.values())


@dataclass(frozen=True)
class SoftBirths:
    """K + 1 refits of soft-residual birth starts: children, novel rows emitted, starts
    omitted by the generator size."""

    children: tuple
    emitted: int
    omitted: int


def soft_births(workspace, solution, likelihoods, residual_inputs, config, cache) -> SoftBirths:
    """Soft-residual births of a panel (memoised per (panel, assignments) in ``cache``)."""
    key = (solution.key, solution.assignments.tobytes())
    cached = cache.get(key)
    if cached is not None:
        return cached
    soft = soft_candidates(solution.haplotypes, residual_inputs, config.residual_births)
    if len(soft) == 0:
        result = SoftBirths((), 0, 0)
    else:
        existing = {np.asarray(row, dtype=np.int8).tobytes() for row in solution.haplotypes}
        starts = [np.vstack((solution.haplotypes, row[None, :]))
                  for row in _hardened_rows(soft, likelihoods)
                  if np.asarray(row, dtype=np.int8).tobytes() not in existing]
        limit = config.search.max_proposals_per_expansion
        synchronized, refitted = fit_panels(workspace, starts[:limit], config.model.max_iterations)
        children = deduplicate((*synchronized, *refitted))
        result = SoftBirths(tuple(child for child in children if child.k == solution.k + 1),
                            len(starts), max(0, len(starts) - limit))
    cache[key] = result
    return result


def replacements(workspace, parent, births, config):
    """Same-K starts: each K + 1 birth (in search order) with any one row deleted, distinct by
    the remaining rows, at most ``max_replacement_children``; returns (children, omitted)."""
    width = parent.sites
    descriptors = []
    seen: set[bytes] = set()
    for birth in sorted(deduplicate(births), key=lambda solution: solution.order()):
        if birth.k != parent.k + 1:
            continue
        for deleted in range(birth.k):
            offset = deleted * width
            key = birth.key[:offset] + birth.key[offset + width:]
            if key in seen:
                continue
            seen.add(key)
            descriptors.append((birth, deleted))
    limit = config.search.max_replacement_children
    starts = [np.ascontiguousarray(np.delete(birth.haplotypes, deleted, axis=0))
              for birth, deleted in descriptors[:limit]]
    synchronized, refitted = fit_panels(workspace, starts, config.model.max_iterations)
    children = deduplicate((*synchronized, *refitted))
    return tuple(child for child in children if child.k == parent.k), max(0, len(descriptors) - limit)


@dataclass(frozen=True)
class OrdinaryMoves:
    """One ordinary expansion: every fitted neighbour (deaths, gauge rewires, data births),
    the K + 1 births among them, starts omitted by the generator size, and limitations."""

    children: tuple
    births: tuple
    omitted: int
    limits: tuple


def ordinary_moves(workspace, solution, likelihoods, growth, ceiling, cuts: MaximumCuts,
                   config) -> OrdinaryMoves:
    """Deaths and gauge rewires (interleaved one each, each kind in key order, at most
    ``max_proposals_per_expansion`` starts), fitted, plus data births below the ceiling."""
    search = config.search
    buckets = {-1: [], 0: []}
    limits = []
    if solution.k > 1:
        buckets[-1].extend(death_starts(solution))
        weights = assignment_graph(solution.assignments, solution.k)
        partitions = cuts.partitions(weights, search.max_cut_ties + 1)
        if len(partitions) > search.max_cut_ties:
            limits.append("gauge_max_cut_ties_truncated")
        buckets[0].extend(gauge_starts(solution, likelihoods, partitions[:search.max_cut_ties]))
    unique = {}
    for offset, starts in buckets.items():
        by_key = {}
        for start in starts:
            value = np.ascontiguousarray(start, dtype=np.int64)
            by_key.setdefault((value.shape, value.tobytes()), value)
        unique[offset] = [by_key[key] for key in sorted(by_key)]
    total = sum(len(starts) for starts in unique.values())
    retained = []
    index = 0
    while len(retained) < min(total, search.max_proposals_per_expansion):
        added = False
        for offset in (-1, 0):
            if index < len(unique[offset]):
                retained.append(unique[offset][index])
                added = True
                if len(retained) == search.max_proposals_per_expansion:
                    break
        if not added:
            break
        index += 1
    omitted = total - len(retained)
    if omitted:
        limits.append("proposal_budget")
    synchronized, refitted = fit_panels(workspace, retained, config.model.max_iterations)
    children = list(deduplicate((*synchronized, *refitted)))
    if solution.k < ceiling:
        children.extend(child for child in data_births(
            workspace, solution, growth, config.model.max_iterations)
            if child.k == solution.k + 1)
    children = deduplicate(children)
    births = tuple(child for child in children if child.k == solution.k + 1)
    return OrdinaryMoves(children, births, omitted, tuple(dict.fromkeys(limits)))
