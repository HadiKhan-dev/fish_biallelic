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

Release 2.1 (noise stability, design/DESIGN.md 3.3): no proposal order is decided by rounding,
perturbation-level differences or array layout. K = 1 similarities are computed in one fixed
summation order (no BLAS layout), exactly symmetric; consensus and dosage rows treat values within
the ``probability`` quantum of 1/2 as exactly 1/2 (population ALT fraction, then REF); gauge
rewires need a gain above the ``nll`` quantum; data births rank samples by tie classes of their
excess NLL (kind ``sample``; an all-unknown sample's excess is its exact value 2 lambda n_obs), then
observed sites, then canonical sample order, and consume tied samples as whole groups; replacements
follow births in tie-class order.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from ..execution import claim_threads
from ..maths.ties import NOISE_MOTION, WIDE_CLASS_QUANTA, tie_classes
from .fixed_k.fit import FIT_MAX_THREADS, fit_one, fit_panels, fit_starts
from .panel import canonical_solution, deduplicate, has_distinct_rows, panel_key, sort_solutions
from .residual_births import (HALF_BAND, alt_fraction_rule, exact_hdbscan_labels, half_band_rows,
                              hardening_fallback, note_band, scalar_log_floored,
                              soft_candidates)

_TINY = np.finfo(np.float64).tiny
# Maximum-cut results memoised per block search (an exact memo of a pure function).
MAX_CUT_CACHE_ENTRIES = 256


# ---------------------------------------------------------------------------------------------
# K = 1 data seeds
# ---------------------------------------------------------------------------------------------
@njit(cache=True, parallel=True, nogil=True)
def expected_agreement(likelihoods):
    """S[i, j] = mean over sites of sum_g P_i(g) P_j(g), each entry summed over sites (then
    genotypes 0, 1, 2) in one fixed order and stored in both triangles: the values do not depend
    on the sample order, threads, BLAS layout or CPU type (2.0 used three BLAS Gram matrices)."""
    n, n_sites, _ = likelihoods.shape
    similarity = np.empty((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(i, n):
            total = 0.0
            for site in range(n_sites):
                total += (likelihoods[i, site, 0] * likelihoods[j, site, 0]
                          + likelihoods[i, site, 1] * likelihoods[j, site, 1]
                          + likelihoods[i, site, 2] * likelihoods[j, site, 2])
            value = total / n_sites
            similarity[i, j] = value
            similarity[j, i] = value
    return similarity


def _soft_cluster_seeds(likelihoods, n_seeds, min_cluster_size, counter=None):
    """Up to n_seeds pooled binary rows from HDBSCAN clusters of the samples (distance =
    max agreement - expected per-site genotype agreement), largest clusters first (ties: HDBSCAN's
    label order). A consensus within HALF_BAND of 1/2 takes the population ALT fraction (REF at
    1/2 itself). HDBSCAN: residual_births.exact_hdbscan_labels (the package's algorithm, equal-weight tree edges in
    a CPU-independent order)."""
    if likelihoods.shape[0] < int(min_cluster_size):
        return []
    likelihoods = np.ascontiguousarray(likelihoods, dtype=np.float64)
    similarity = expected_agreement(likelihoods)
    distance = similarity.max() - similarity
    np.fill_diagonal(distance, 0.0)
    labels = exact_hdbscan_labels(distance, min_cluster_size=int(min_cluster_size),
                                  min_samples=int(min_cluster_size), allow_single_cluster=False)
    clusters = [np.flatnonzero(labels == label) for label in np.unique(labels) if label != -1]
    clusters.sort(key=lambda members: -len(members))
    alt_fraction = 0.5 * likelihoods[:, :, 1] + likelihoods[:, :, 2]
    population = alt_fraction.mean(axis=0)
    marginal = alt_fraction_rule(population, False)
    note_band(counter, "discovery.k1.half_band", population, 0.5, HALF_BAND)
    rows = []
    for members in clusters[:n_seeds]:
        mean = alt_fraction[members].mean(axis=0)
        note_band(counter, "discovery.k1.half_band", mean, 0.5, HALF_BAND)
        rows.append(half_band_rows(mean, marginal).astype(np.int64))
    return rows


def _dosage_seed(likelihoods, sample):
    """The sample's argmax dosage as a row: dosage 2 -> 1, dosage 1 -> 1 where the population
    ALT fraction exceeds 1/2 (outside HALF_BAND), else 0."""
    dosage = likelihoods[sample].argmax(axis=1)
    population_alt = likelihoods[..., 1].mean(axis=0) * 0.5 + likelihoods[..., 2].mean(axis=0)
    row = np.zeros(likelihoods.shape[1], dtype=np.int64)
    row[dosage == 2] = 1
    row[(dosage == 1) & alt_fraction_rule(population_alt, False)] = 1
    return row


def _most_decisive(likelihoods, observed, sample_quantum):
    """The fallback seed sample: most observed sites (exact), then the best class of
    decisiveness (sum of the largest GL over observed sites; tie classes of kind ``sample``,
    descending), then canonical sample order."""
    observed_sites = observed.sum(axis=1)
    candidates = np.flatnonzero(observed_sites == observed_sites.max())
    decisiveness = np.where(observed, likelihoods.max(axis=2), 0.0).sum(axis=1)
    classes = tie_classes(decisiveness[candidates], sample_quantum, descending=True)
    return int(candidates[classes.winner])


def initial_solutions(workspace, likelihoods, active, config, counter=None):
    """The K = 1 search starts: fitted soft-cluster seeds (of the samples with depth), at most
    ``data_start_beam_width`` distinct ones in tie-class order of NLL (kind ``nll``)."""
    seed_likelihoods = likelihoods if np.all(active) else np.ascontiguousarray(likelihoods[active])
    search = config.search
    seeds = _soft_cluster_seeds(seed_likelihoods, search.data_seed_count,
                                search.seed_min_cluster_size, counter)
    if not seeds:
        seed_observed = workspace.observed if np.all(active) else workspace.observed[active]
        sample = _most_decisive(seed_likelihoods, seed_observed, workspace.quanta.sample)
        seeds = [_dosage_seed(seed_likelihoods, sample)]
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


def gauge_starts(solution, likelihoods, partitions, quantum):
    """Gauge-rewire starts of a fitted panel for the given cuts (sitewise, then opposite per
    cut), skipping panels already proposed or equal to the parent. A sitewise flip needs its
    resolving samples' NLL to fall by more than ``quantum`` (the block's ``nll`` quantum)."""
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
            current = -float(np.sum(scalar_log_floored(
                np.ascontiguousarray(likelihoods[resolving, site, dosage], dtype=np.float64), _TINY)))
            opposite = -float(np.sum(scalar_log_floored(
                np.ascontiguousarray(likelihoods[resolving, site, 2 - dosage], dtype=np.float64), _TINY)))
            if opposite < current - quantum:
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
    fraction exceeds 1/2 outside HALF_BAND), the samples with depth, and each sample's number of
    observed sites."""

    oracle_nll: np.ndarray
    decisiveness: np.ndarray
    dosage: np.ndarray
    seed_rows: np.ndarray
    active: np.ndarray
    observed_sites: np.ndarray


def growth_inputs(likelihoods, observed) -> GrowthInputs:
    maximum = np.max(likelihoods, axis=2)
    dosage = np.argmax(likelihoods, axis=2)
    active = np.any(observed, axis=1)
    oracle_nll = -np.sum(np.where(observed, scalar_log_floored(
        np.ascontiguousarray(maximum, dtype=np.float64), _TINY), 0.0), axis=1)
    decisiveness = np.sum(np.where(observed, maximum, 0.0), axis=1)
    alt_fraction = 0.5 * likelihoods[..., 1] + likelihoods[..., 2]
    observed_count = np.sum(observed, axis=0)
    population_alt = np.divide(np.sum(np.where(observed, alt_fraction, 0.0), axis=0),
                               observed_count, out=np.full(likelihoods.shape[1], 0.5),
                               where=observed_count > 0)
    seed_rows = np.zeros_like(dosage)
    seed_rows[observed & (dosage == 2)] = 1
    ambiguous = (~observed) | (dosage == 1)
    seed_rows[ambiguous & alt_fraction_rule(population_alt, False)[None, :]] = 1
    observed_sites = np.sum(observed, axis=1).astype(np.int64)
    return GrowthInputs(oracle_nll, decisiveness, dosage, seed_rows, active, observed_sites)


@dataclass(frozen=True)
class DataBirths:
    """K + 1 children of one data-birth call; ``tie_cut``: a tie group was cut at the hard cap
    (reported as the search limit "data_birth_tie_budget")."""

    children: tuple
    tie_cut: bool = False


def data_birth_order(excess, observed_sites, samples, quantum):
    """(samples in data-birth order, their tie-group ids, the TieClasses) of the samples with depth.

    Order: tie classes of excess NLL (kind ``sample``, descending: the most misfit first), then more
    observed sites, then canonical sample order (inside a class wider than 4 quanta: excess first,
    the width guard). A tie group is the samples of one class with one observed-site count (a wide
    class is one group); groups are consumed whole.
    """
    samples = [int(sample) for sample in samples]
    if not samples:
        return [], [], None
    classes = tie_classes([float(excess[sample]) for sample in samples], quantum,
                          content=[(-int(observed_sites[sample]), sample) for sample in samples],
                          descending=True, motion=NOISE_MOTION["sample"])
    wide = classes.spans > WIDE_CLASS_QUANTA * quantum
    order, groups = [], []
    for index in classes.order:
        sample = samples[int(index)]
        cls = int(classes.classes[int(index)])
        order.append(sample)
        groups.append((cls,) if wide[cls] else (cls, int(observed_sites[sample])))
    return order, groups, classes


def collect_birth_starts(order, groups, seeds_of, budget: int, hard_cap: int):
    """(groups of start rows, cut) of data births: the starts of the samples in ``order`` collected
    group by group (``groups[i]`` is the tie-group id of ``order[i]``) until at least ``budget``
    starts are collected; the last group is completed unless the total would exceed ``hard_cap``
    (then it is cut in its order: ``cut``). ``seeds_of(sample)`` yields the sample's novel starts
    (it skips rows already proposed)."""
    collected: list[list] = []
    total = 0
    position = 0
    cut = False
    while position < len(order) and total < budget and not cut:
        group_id = groups[position]
        group = []
        while position < len(order) and groups[position] == group_id:
            sample = order[position]
            position += 1
            if cut:
                continue
            for row in seeds_of(sample):
                if total + len(group) >= hard_cap:
                    cut = True
                    break
                group.append(row)
        if group:
            collected.append(group)
            total += len(group)
    return collected, cut


def data_births(workspace, solution, growth: GrowthInputs, max_iterations, search,
                counter=None) -> DataBirths:
    """K + 1 children of a panel from its most misfit samples (``search``: SearchConfig).

    The parent is first refitted (unless the workspace certified it a fixed point). Samples
    with depth are ranked by excess NLL over their genotype oracle in tie classes (kind
    ``sample``; an excess within the quantum of an all-unknown sample's exact value 2 lambda n_obs
    is that value, so mathematically equal excesses are equal), then observed sites, then
    canonical sample order (data_birth_order). Each sample contributes K subtraction seeds
    clip(dosage - row, 0, 1) and its dosage seed row. Tie groups are consumed whole: novel seeds
    are collected group by group until the attempt budget (search.data_birth_attempts(K)) is
    reached, then fitted group by group until search.data_birth_children(K) distinct complete
    children exist or the budget is spent, both tested only at group boundaries; a group is cut
    (canonical order) only at search.data_birth_tie_budget_factor x the budget (``tie_cut``). If no
    child is complete, the assignment-only fits of the fitted starts with distinct rows are
    returned instead (deduplicated, at most K + 1): the requested K without moving an unsupported
    row.
    """
    claim_threads(max_threads=FIT_MAX_THREADS)
    if solution.fixed_point and workspace.certifies_fixed_point(solution.haplotypes):
        parent = solution
    else:
        parent = canonical_solution(fit_one(workspace, solution.haplotypes, max_iterations),
                                    workspace)
    quantum = workspace.quanta.sample
    excess = np.maximum(parent.per_sample_cost - growth.oracle_nll, 0.0)
    # An all-unknown sample's excess is exactly 2 lambda n_obs (a grid point by construction): snap
    # values within the quantum to it, so equal mathematics gives equal keys (2.0: 1-2 ulps apart).
    unknown_excess = 2.0 * workspace.penalty * growth.observed_sites.astype(np.float64)
    snapped = np.abs(excess - unknown_excess) <= quantum
    excess = np.where(snapped, unknown_excess, excess)
    order, groups, classes = data_birth_order(excess, growth.observed_sites,
                                              np.flatnonzero(growth.active), quantum)
    target = search.data_birth_children(parent.k)
    budget = search.data_birth_attempts(parent.k)
    hard_cap = budget * max(1, int(search.data_birth_tie_budget_factor))
    seen = {np.ascontiguousarray(row, dtype=np.int8).tobytes() for row in parent.haplotypes}

    def seeds_of(sample):
        """The sample's novel K + 1 starts: K subtraction seeds, then its dosage seed row."""
        seeds = [np.clip(growth.dosage[sample] - parent.haplotypes[index], 0, 1)
                 for index in range(parent.k)]
        seeds.append(growth.seed_rows[sample])
        for seed in seeds:
            row = np.ascontiguousarray(seed, dtype=np.int64)
            key = row.astype(np.int8).tobytes()
            if key in seen:
                continue
            seen.add(key)
            yield np.vstack([parent.haplotypes, row[None, :]])

    collected, cut = collect_birth_starts(order, groups, seeds_of, budget, hard_cap)
    children = []
    child_keys: set[bytes] = set()
    fitted = []
    tie_cut = False
    for number, group in enumerate(collected):
        for fit in fit_starts(workspace, group, max_iterations):
            child = canonical_solution(fit, workspace)
            if has_distinct_rows(child):
                children.append(child)
                child_keys.add(child.key)
        fitted.extend(group)
        if cut and number == len(collected) - 1:
            tie_cut = True
        if len(child_keys) >= target or len(fitted) >= budget:
            break
    if counter is not None and classes is not None:
        counter.note("discovery.data_birth", inside_tie=tie_cut, wide=classes.wide_classes,
                     boundary=len(classes.boundary_gaps))
    if children:
        return DataBirths(tuple(children), tie_cut)
    fallbacks = [canonical_solution(fit, workspace) for fit in fit_starts(workspace, fitted, 0)]
    return DataBirths(deduplicate([child for child in fallbacks if has_distinct_rows(child)],
                                  parent.k + 1), tie_cut)


# ---------------------------------------------------------------------------------------------
# Soft births and birth-delete replacements
# ---------------------------------------------------------------------------------------------
def _hardened_rows(soft, likelihoods, counter=None):
    """Binary rows of soft candidates (> 1/2 is ALT; within HALF_BAND of 1/2 the population ALT
    fraction decides; release 2.1, was 1e-12), distinct, in candidate order."""
    note_band(counter, "discovery.soft.half_band", soft, 0.5, HALF_BAND)
    binary = half_band_rows(soft, hardening_fallback(likelihoods))
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


def soft_births(workspace, solution, likelihoods, residual_inputs, config, cache, counter=None) -> SoftBirths:
    """Soft-residual births of a panel (memoised per (panel, assignments) in ``cache``; ``counter``: optional
    TieCounter of the band decisions)."""
    key = (solution.key, solution.assignments.tobytes())
    cached = cache.get(key)
    if cached is not None:
        return cached
    soft = soft_candidates(solution.haplotypes, residual_inputs, config.residual_births, counter)
    if len(soft) == 0:
        result = SoftBirths((), 0, 0)
    else:
        existing = {np.asarray(row, dtype=np.int8).tobytes() for row in solution.haplotypes}
        starts = [np.vstack((solution.haplotypes, row[None, :]))
                  for row in _hardened_rows(soft, likelihoods, counter)
                  if np.asarray(row, dtype=np.int8).tobytes() not in existing]
        limit = config.search.max_proposals_per_expansion
        synchronized, refitted = fit_panels(workspace, starts[:limit], config.model.max_iterations)
        children = deduplicate((*synchronized, *refitted))
        result = SoftBirths(tuple(child for child in children if child.k == solution.k + 1),
                            len(starts), max(0, len(starts) - limit))
    cache[key] = result
    return result


def replacements(workspace, parent, births, config, limit=None):
    """Same-K starts: each K + 1 birth (in tie-class order of NLL, then key) with any one row
    deleted, distinct by the remaining rows, at most ``limit`` (default
    ``max_replacement_children``); returns (children, omitted)."""
    width = parent.sites
    descriptors = []
    seen: set[bytes] = set()
    for birth in sort_solutions(deduplicate(births)):
        if birth.k != parent.k + 1:
            continue
        for deleted in range(birth.k):
            offset = deleted * width
            key = birth.key[:offset] + birth.key[offset + width:]
            if key in seen:
                continue
            seen.add(key)
            descriptors.append((birth, deleted))
    limit = config.search.max_replacement_children if limit is None else int(limit)
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
                   config, counter=None) -> OrdinaryMoves:
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
        buckets[0].extend(gauge_starts(solution, likelihoods, partitions[:search.max_cut_ties],
                                       workspace.quanta.nll))
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
        births = data_births(workspace, solution, growth, config.model.max_iterations, search,
                             counter=counter)
        if births.tie_cut:
            limits.append("data_birth_tie_budget")
        children.extend(child for child in births.children if child.k == solution.k + 1)
    children = deduplicate(children)
    births = tuple(child for child in children if child.k == solution.k + 1)
    return OrdinaryMoves(children, births, omitted, tuple(dict.fromkeys(limits)))
