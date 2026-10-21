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
from copy import copy
from dataclasses import dataclass
import math
import sys

import numpy as np
from numba import njit, prange

from ..execution import claim_threads
from ..maths.ties import BOUNDARY_GUARD, NOISE_MOTION, WIDE_CLASS_QUANTA, tie_classes
from .conventions import rev
from .fixed_k.fit import FIT_MAX_THREADS, fit_families, fit_panels, fit_starts
from .panel import _nll_quantum, canonical_solution, deduplicate, has_distinct_rows, panel_key, sort_solutions, whole_class_cut
from .residual_births import (HALF_BAND, alt_fraction_families, capped_by_size, exact_hdbscan_labels, half_band_rows,
                              hardening_fallbacks, note_band, sample_sum, scalar_log_floored,
                              soft_candidates)

_TINY = np.finfo(np.float64).tiny
# Maximum-cut results memoised per block search (an exact memo of a pure function).
MAX_CUT_CACHE_ENTRIES = 256
# Assignment-only replacement screening retains at most this many starts and this estimated
# worst-case compact-membership payload per batch. These are memory bounds, not search budgets.
_REPLACEMENT_SCORE_BATCH = 32
_REPLACEMENT_SCORE_BYTES = 32 * 1024 * 1024
_REPLACEMENT_MEMO_BYTES = 32 * 1024 * 1024


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


def _note_omission(omissions, cause, count, scope, count_kind="exact", kind="tie_class_at_budget"):
    if omissions is not None and count:
        omissions.append(dict(kind=kind, cause=cause, count=int(count), scope=scope, count_kind=count_kind))


def _unique_seeds(rows):
    """The union of binary seeds; byte order labels the set and never truncates it."""
    unique = {np.asarray(row, dtype=np.int8).tobytes(): np.asarray(row, dtype=np.int64) for row in rows}
    return [unique[key] for key in sorted(unique)]


def _soft_cluster_seeds(likelihoods, n_seeds, min_cluster_size, counter=None, omissions=None, landmark_budget=0):
    """Distinct cluster seeds admitted by whole cluster-size classes, largest first.

    Duplicate consensus rows consume one slot. Complete straddling classes extend to twice the
    nominal budget; omitted straddling classes and lower-ranked truncation are counted separately.
    """
    if likelihoods.shape[0] < int(min_cluster_size):
        return []
    likelihoods = np.ascontiguousarray(likelihoods, dtype=np.float64)
    if landmark_budget and len(likelihoods) > landmark_budget:
        from .landmarks import seed_groups
        clusters, _stats = seed_groups(likelihoods, landmark_budget, int(min_cluster_size), omissions)
    else:
        similarity = expected_agreement(likelihoods)
        distance = similarity.max() - similarity
        del similarity  # No longer needed; do not retain an extra N-by-N array through clustering.
        np.fill_diagonal(distance, 0.0)
        labels = exact_hdbscan_labels(distance, min_cluster_size=int(min_cluster_size),
                                      min_samples=int(min_cluster_size), allow_single_cluster=False)
        clusters = [np.flatnonzero(labels == label) for label in np.unique(labels) if label != -1]
    alt_fraction = 0.5 * likelihoods[:, :, 1] + likelihoods[:, :, 2]
    population = sample_sum(alt_fraction) / len(alt_fraction)
    marginals = alt_fraction_families(population, False)
    note_band(counter, "discovery.k1.half_band", population, 0.5, HALF_BAND)
    rows, classes, seen = [], [], set()
    for size in sorted({len(members) for members in clusters}, reverse=True):
        group = []
        for members in clusters:
            if len(members) != size:
                continue
            mean = sample_sum(alt_fraction[members]) / len(members)
            note_band(counter, "discovery.k1.half_band", mean, 0.5, HALF_BAND)
            group.extend(half_band_rows(mean, marginal).astype(np.int64) for marginal in marginals)
        group = [row for row in _unique_seeds(group) if row.astype(np.int8).tobytes() not in seen]
        seen.update(row.astype(np.int8).tobytes() for row in group)
        rows.extend(group)
        classes.extend([size] * len(group))
    rows, cut = whole_class_cut(rows, classes, n_seeds, omissions,
                               cause="k1_cluster_seed", scope="initial cluster-size classes")
    if counter is not None:
        counter.note("discovery.k1.seed_overflow", inside_tie=cut)
    return rows


def _dosage_extremes(values, counter=None, kind="discovery.dosage.genotype"):
    """Lowest/highest genotype indices in the full probability-q winning class.

    For three sorted probabilities a <= b <= c, the winner includes b iff c-b <= q,
    and includes a iff both adjacent gaps are <= q. This is exactly the central
    descending chained-class definition, independent of genotype enumeration.
    The likelihood values themselves are never changed.
    """
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values, axis=-1)
    gaps = np.diff(ordered, axis=-1)
    upper_tied = gaps[..., 1] <= HALF_BAND
    all_tied = upper_tied & (gaps[..., 0] <= HALF_BAND)
    floor = np.where(all_tied, ordered[..., 0],
                     np.where(upper_tied, ordered[..., 1], ordered[..., 2]))
    members = values >= floor[..., None]
    low = np.argmax(members, axis=-1)
    high = 2 - np.argmax(members[..., ::-1], axis=-1)
    if counter is not None:
        entry = counter.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
        entry["decisions"] += int(low.size)
        entry["inside_tie"] += int(np.count_nonzero(low != high))
        entry["boundary"] += int(np.count_nonzero(
            (gaps != 0.0) & (np.abs(gaps - HALF_BAND) <= BOUNDARY_GUARD * NOISE_MOTION["probability"])))
        # Three members can span at most 2q, so no winning class reaches the 4q width guard.
    return low, high


def _dosage_families(likelihoods, counter=None, kind="discovery.dosage.genotype"):
    low, high = _dosage_extremes(likelihoods, counter, kind)
    return (low,) if np.array_equal(low, high) else (low, high)


def _dosage_seeds(likelihoods, samples, counter=None):
    """Complete dosage and population-half families of each decisive sample, deduplicated."""
    samples = np.asarray(samples, dtype=np.int64)
    dosages = _dosage_families(likelihoods[samples], counter, "discovery.k1.dosage")
    population_alt = sample_sum(0.5 * likelihoods[..., 1] + likelihoods[..., 2]) / len(likelihoods)
    population_rows = alt_fraction_families(population_alt, False)
    return _unique_seeds(row for dosage in dosages for population_row in population_rows for row in
                         ((dosage == 2) | ((dosage == 1) & population_row[None, :])))


def _most_decisive(likelihoods, observed, sample_quantum):
    """The fallback seed sample: most observed sites (exact), then the best class of
    decisiveness (sum of the largest GL over observed sites; tie classes of kind ``sample``,
    descending), then canonical sample order."""
    observed_sites = observed.sum(axis=1)
    candidates = np.flatnonzero(observed_sites == observed_sites.max())
    decisiveness = np.where(observed, likelihoods.max(axis=2), 0.0).sum(axis=1)
    classes = tie_classes(decisiveness[candidates], sample_quantum, descending=True)
    return int(candidates[classes.winner])


def _most_decisive_class(likelihoods, observed, sample_quantum):
    """The samples tied for the fallback seed: most observed sites (exact), then the best class of decisiveness (sum
    of the largest GL over observed sites; tie classes of kind ``sample``, descending), in sample order."""
    observed_sites = observed.sum(axis=1)
    candidates = np.flatnonzero(observed_sites == observed_sites.max())
    decisiveness = np.where(observed, likelihoods.max(axis=2), 0.0).sum(axis=1)
    classes = tie_classes(decisiveness[candidates], sample_quantum, descending=True)
    first = classes.classes[classes.winner]
    return [int(candidates[i]) for i in range(len(candidates)) if classes.classes[i] == first]


def initial_solutions(workspace, likelihoods, active, config, counter=None, omissions=None):
    """K=1 starts from whole cluster-size or fallback decisive-sample classes.

    The fallback is used only if clustering supplies no seed class. If all available cluster seeds
    are omitted, no unscored population row substitutes for them: the block has no incumbent.
    """
    seed_likelihoods = likelihoods if np.all(active) else np.ascontiguousarray(likelihoods[active])
    search = config.search
    omitted = []
    seeds = _soft_cluster_seeds(seed_likelihoods, search.data_seed_count,
                                search.seed_min_cluster_size, counter, omitted, search.seed_landmark_budget)
    if not seeds and not any("landmark" not in event["cause"] for event in omitted):
        seed_observed = workspace.observed if np.all(active) else workspace.observed[active]
        samples = _most_decisive_class(seed_likelihoods, seed_observed, workspace.quanta.sample)
        seeds = _dosage_seeds(seed_likelihoods, samples, counter)
        seeds, _ = whole_class_cut(seeds, [0] * len(seeds), search.data_seed_count, omitted,
                                   cause="k1_sample_seed", scope="initial decisive-sample class")
    if omissions is not None:
        omissions.extend(omitted)
    families = fit_families(workspace, [seed[None, :] for seed in seeds], config.model.max_iterations)
    solutions = deduplicate(canonical_solution(fit, workspace) for family in families for fit in family)
    ordered = sort_solutions(solutions, workspace.quanta)
    classes = tie_classes([solution.nll for solution in ordered], workspace.quanta.nll).classes
    kept, _ = whole_class_cut(ordered, classes.tolist(), search.data_start_beam_width, omissions,
                              cause="k1_fit_beam", scope="initial fitted NLL classes")
    return tuple(kept)


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
def _component_exact_cuts(weights, max_ties, reverse):
    """Maximum cuts by independent components: bipartite components by 2-colouring,
    the others by exhaustive enumeration. Cuts are integer codes with vertex order as most to
    least significant bit; the smallest codes are kept (byte-lexicographic tie rule; the largest under reversed
    conventions), at most ``max_ties`` non-empty cuts."""
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
        choices = np.sort(choices)
        if reverse:
            choices = choices[::-1].copy()
        choices = choices[:limit]
        joined = np.empty(len(combined) * len(choices), dtype=np.int64)
        cursor = 0
        for previous in combined:
            for choice in choices:
                joined[cursor] = previous | choice
                cursor += 1
        combined = np.sort(joined)
        if reverse:
            combined = combined[::-1].copy()
        combined = combined[:limit]
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
            result = tuple(side.copy() for side in _component_exact_cuts(matrix, max_ties, rev("d9")))
        else:
            reverse = rev("d9")
            order = np.asarray(sorted(range(1, k), key=lambda index: (-int(np.sum(matrix[index])),
                                                                      -index if reverse else index)), dtype=np.int64)
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
                                 key=lambda item: item[1], reverse=reverse)
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
            current = -math.fsum(scalar_log_floored(
                np.ascontiguousarray(likelihoods[resolving, site, dosage], dtype=np.float64), _TINY).tolist())
            opposite = -math.fsum(scalar_log_floored(
                np.ascontiguousarray(likelihoods[resolving, site, 2 - dosage], dtype=np.float64), _TINY).tolist())
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
    decisiveness of every sample over its observed sites, low/high winning-class dosage
    vectors and their seed rows (dosage 2 -> 1; dosage 1 or unobserved -> 1 where the observed population ALT
    fraction exceeds 1/2 outside HALF_BAND; both coherent hardenings inside the band), the samples
    with depth, and each sample's number of observed sites."""

    oracle_nll: np.ndarray
    decisiveness: np.ndarray
    dosages: tuple[np.ndarray, ...]
    seed_rows: tuple[np.ndarray, ...]
    active: np.ndarray
    observed_sites: np.ndarray


def growth_inputs(likelihoods, observed, counter=None) -> GrowthInputs:
    maximum = np.max(likelihoods, axis=2)
    dosages = _dosage_families(likelihoods, counter)
    active = np.any(observed, axis=1)
    oracle_nll = -np.sum(np.where(observed, scalar_log_floored(
        np.ascontiguousarray(maximum, dtype=np.float64), _TINY), 0.0), axis=1)
    decisiveness = np.sum(np.where(observed, maximum, 0.0), axis=1)
    alt_fraction = 0.5 * likelihoods[..., 1] + likelihoods[..., 2]
    observed_count = np.sum(observed, axis=0)
    population_alt = np.divide(sample_sum(np.where(observed, alt_fraction, 0.0)),
                               observed_count, out=np.full(likelihoods.shape[1], 0.5),
                               where=observed_count > 0)
    seed_rows = []
    population_rows = alt_fraction_families(population_alt, False)
    for dosage in dosages:
        for population_row in population_rows:
            rows = np.zeros_like(dosage)
            rows[observed & (dosage == 2)] = 1
            ambiguous = (~observed) | (dosage == 1)
            rows[ambiguous & population_row[None, :]] = 1
            seed_rows.append(rows)
    observed_sites = np.sum(observed, axis=1).astype(np.int64)
    return GrowthInputs(oracle_nll, decisiveness, dosages, tuple(seed_rows), active, observed_sites)


def _data_seed_rows(growth, sample, parent_rows):
    """Union dosage, population-half and subtraction seeds before the whole-class budget.

    Missing-cell genotype ties matter to subtraction starts even when the direct
    seed at that cell comes from the population hardening family.
    """
    # The same integer subtraction/clip for all parent rows in one array operation;
    # retain dosage-then-parent order before the existing exact union.
    seeds = [row for dosage in growth.dosages
             for row in np.clip(dosage[sample][None, :] - parent_rows, 0, 1)]
    seeds.extend(rows[sample] for rows in growth.seed_rows)
    return _unique_seeds(seeds)


@dataclass(frozen=True)
class DataBirths:
    """K+1 data-birth children; tie_cut marks a wholly omitted class at the hard cap.

    omitted counts that class's distinct starts (search limit data_birth_tie_budget).
    """

    children: tuple
    tie_cut: bool = False
    omitted: int = 0
    omissions: tuple = ()


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
                          secondary=[(-int(observed_sites[sample]),) for sample in samples],
                          content=[sample for sample in samples],
                          descending=True, motion=NOISE_MOTION["sample"])
    wide = classes.spans > WIDE_CLASS_QUANTA * quantum
    order, groups = [], []
    for index in classes.order:
        sample = samples[int(index)]
        cls = int(classes.classes[int(index)])
        order.append(sample)
        groups.append((cls,) if wide[cls] else (cls, int(observed_sites[sample])))
    return order, groups, classes


def collect_birth_starts(order, groups, seeds_of, budget: int, hard_cap: int, omissions=None):
    """Union each excess/coverage class's distinct starts before its hard-cap decision.

    The attempt budget is checked between groups. A class crossing the hard cap is omitted whole;
    no sample, founder-row or seed enumeration order can select a privileged prefix.
    """
    collected, seen = [], set()
    total = position = ranked = 0
    cut = stopped = False
    while position < len(order):
        group_id = groups[position]
        unique = {}
        while position < len(order) and groups[position] == group_id:
            for row in seeds_of(order[position]):
                key = panel_key(row)
                if key not in seen:
                    unique.setdefault(key, row if not stopped else None)
            position += 1
        seen.update(unique)
        if stopped or (total >= budget and total > 0):
            stopped = True
            ranked += len(unique)
        elif total + len(unique) > hard_cap:
            cut = stopped = True
            _note_omission(omissions, "data_birth_tie_class_at_budget", len(unique),
                           "excess/coverage seed class")
        elif unique:
            collected.append([unique[key] for key in sorted(unique)])
            total += len(unique)
    _note_omission(omissions, "data_birth_ranked_truncation", ranked,
                   "lower excess/coverage seed classes", kind="ranked_truncation")
    return collected, cut


def data_births(workspace, solution, growth: GrowthInputs, max_iterations, search,
                counter=None) -> DataBirths:
    """K + 1 children of a panel from its most misfit samples (``search``: SearchConfig).

    The parent is first refitted to its complete schedule family (unless the workspace
    certified both schedules fixed). Parent/sample pairs with depth are ranked together by excess NLL over their genotype oracle in tie classes (kind
    ``sample``; an excess within the quantum of an all-unknown sample's exact value 2 lambda n_obs
    is that value, so mathematically equal excesses are equal), then observed sites, then
    canonical sample order (data_birth_order). Each sample contributes both winning-class extremal dosage families: K subtraction seeds
    clip(dosage - row, 0, 1) and a direct dosage seed row for each, deduplicated before admission. Tie groups are consumed whole: novel seeds
    are collected group by group until the attempt budget (search.data_birth_attempts(K)) is
    reached, then fitted group by group until search.data_birth_children(K) distinct complete
    children exist or the budget is spent, both tested only at group boundaries. A group crossing
    search.data_birth_tie_budget_factor x the budget is omitted whole (``tie_cut``). If no child is
    complete, distinct assignment-only fits are returned by whole NLL classes at width K+1:
    the requested K without moving an unsupported row.
    """
    claim_threads(max_threads=FIT_MAX_THREADS)
    if solution.fixed_point and workspace.certifies_fixed_point(solution.haplotypes):
        parents = (solution,)
    else:
        family, = fit_families(workspace, [solution.haplotypes], max_iterations)
        distinct = {}
        for endpoint in family:
            parent = canonical_solution(endpoint, workspace)
            distinct.setdefault(parent.key, parent)
        parents = tuple(distinct[key] for key in sorted(distinct))
    parent_k = solution.k
    quantum = workspace.quanta.sample
    # Rank parent/sample pairs together and spend one budget for the complete parent family.
    excess = np.concatenate([np.maximum(parent.per_sample_cost - growth.oracle_nll, 0.0)
                             for parent in parents])
    observed_sites = np.tile(growth.observed_sites, len(parents))
    active = np.tile(growth.active, len(parents))
    unknown_excess = 2.0 * workspace.penalty * observed_sites.astype(np.float64)
    excess = np.where(np.abs(excess - unknown_excess) <= quantum, unknown_excess, excess)
    order, groups, classes = data_birth_order(excess, observed_sites, np.flatnonzero(active), quantum)
    target = search.data_birth_children(parent_k)
    budget = search.data_birth_attempts(parent_k)
    hard_cap = budget * max(1, int(search.data_birth_tie_budget_factor))
    seen = [{np.ascontiguousarray(row, dtype=np.int8).tobytes() for row in parent.haplotypes}
            for parent in parents]

    def seeds_of(pair):
        parent_index, sample = divmod(pair, len(growth.active))
        parent = parents[parent_index]
        for seed in _data_seed_rows(growth, sample, parent.haplotypes):
            row = np.ascontiguousarray(seed, dtype=np.int64)
            key = row.astype(np.int8).tobytes()
            if key in seen[parent_index]:
                continue
            seen[parent_index].add(key)
            yield np.vstack([parent.haplotypes, row[None, :]])

    omissions = []
    collected, cut = collect_birth_starts(order, groups, seeds_of, budget, hard_cap, omissions)
    children = []
    child_keys: set[bytes] = set()
    fitted = []
    for number, group in enumerate(collected):
        for family in fit_families(workspace, group, max_iterations):
            for fit in family:
                child = canonical_solution(fit, workspace)
                if has_distinct_rows(child):
                    children.append(child)
                    child_keys.add(child.key)
        fitted.extend(group)
        if len(child_keys) >= target or len(fitted) >= budget:
            break
    tie_cut = cut and len(child_keys) < target and len(fitted) < budget
    if not tie_cut:
        # The complete-child target can stop before the planned attempt boundary. Those later
        # classes are ordinary rank truncations, including a previously planned straddling class.
        for item in omissions:
            if item["kind"] == "tie_class_at_budget":
                item.update(kind="ranked_truncation", cause="data_birth_ranked_truncation")
    _note_omission(omissions, "data_birth_ranked_truncation",
                   sum(len(group) for group in collected) - len(fitted),
                   "seed classes below complete-child target", kind="ranked_truncation")
    if counter is not None and classes is not None:
        counter.note("discovery.data_birth", inside_tie=tie_cut, wide=classes.wide_classes,
                     boundary=len(classes.boundary_gaps))
    if children:
        return DataBirths(tuple(children), tie_cut, sum(item["count"] for item in omissions), tuple(omissions))
    fallbacks = [canonical_solution(fit, workspace) for fit in fit_starts(workspace, fitted, 0)]
    ordered = list(deduplicate(child for child in fallbacks if has_distinct_rows(child)))
    classes = tie_classes([child.nll for child in ordered], workspace.quanta.nll).classes
    kept, cut = whole_class_cut(ordered, classes.tolist(), parent_k + 1, omissions,
                                cause="data_birth_fallback", scope="assignment-only fallback NLL classes")
    return DataBirths(tuple(kept), tie_cut, sum(item["count"] for item in omissions), tuple(omissions))


# ---------------------------------------------------------------------------------------------
# Soft births and birth-delete replacements
# ---------------------------------------------------------------------------------------------
def _hardened_rows(soft, likelihoods, counter=None):
    """Binary rows of soft candidates (> 1/2 is ALT; within HALF_BAND of 1/2 the population ALT
    fraction decides). Both population-half hardenings enter as distinct proposal starts."""
    note_band(counter, "discovery.soft.half_band", soft, 0.5, HALF_BAND)
    fallbacks = hardening_fallbacks(likelihoods)
    unique: dict[bytes, np.ndarray] = {}
    for fallback in fallbacks:
        binary = half_band_rows(soft, fallback)
        for row in np.asarray(binary, dtype=np.int64):
            row = np.ascontiguousarray(row, dtype=np.int64)
            unique.setdefault(row.astype(np.int8).tobytes(), row)
    if len(fallbacks) > 1:
        # Canonical order labels the complete union; no within-family prefix is taken.
        return tuple(unique[key] for key in sorted(unique))
    return tuple(unique.values())


@dataclass(frozen=True)
class SoftBirths:
    """K + 1 refits of soft-residual birth starts: children, novel rows emitted, starts
    omitted by the generator size."""

    children: tuple
    emitted: int
    omitted: int
    omissions: tuple = ()


def soft_births(workspace, solution, likelihoods, residual_inputs, config, cache, counter=None) -> SoftBirths:
    """Soft-residual births of a panel (memoised per (panel, assignments) in ``cache``; ``counter``: optional
    TieCounter of the band decisions)."""
    key = (solution.key, solution.assignments.tobytes())
    cached = cache.get(key)
    if cached is not None:
        return cached
    omissions = []
    soft = soft_candidates(solution.haplotypes, residual_inputs, config.residual_births, counter,
                           limit=config.search.max_proposals_per_expansion, omissions=omissions)
    if len(soft) == 0:
        result = SoftBirths((), 0, 0, tuple(omissions))
    else:
        existing = {np.asarray(row, dtype=np.int8).tobytes() for row in solution.haplotypes}
        starts = [np.vstack((solution.haplotypes, row[None, :]))
                  for row in _hardened_rows(soft, likelihoods, counter)
                  if np.asarray(row, dtype=np.int8).tobytes() not in existing]
        limit = config.search.max_proposals_per_expansion
        synchronized, refitted = fit_panels(workspace, starts, config.model.max_iterations)
        children = deduplicate((*synchronized, *refitted))
        result = SoftBirths(tuple(child for child in children if child.k == solution.k + 1),
                            len(starts), 0, tuple(omissions))
    cache[key] = result
    return result


def _replacement_score_entry_bytes(key, value):
    # Count the packed-state key and float, conservatively including the two integers even
    # when Python interns them. OrderedDict's own table/node storage is counted separately.
    return sys.getsizeof(key) + sum(sys.getsizeof(part) for part in key) + sys.getsizeof(value)


def _remember_replacement_score(workspace, key, value):
    cache = workspace.replacement_score_cache
    if key in cache:
        cache.move_to_end(key)
        return
    value = float(value)
    cache[key] = value
    workspace.replacement_score_payload_bytes += _replacement_score_entry_bytes(key, value)
    stats = workspace.replacement_ranking_counts
    while cache and workspace.replacement_score_payload_bytes + sys.getsizeof(cache) > _REPLACEMENT_MEMO_BYTES:
        old_key, old_value = cache.popitem(last=False)
        workspace.replacement_score_payload_bytes -= _replacement_score_entry_bytes(old_key, old_value)
        stats["score_cache_evictions"] += 1
    if not cache:
        cache.clear()
    stats["max_score_cache_bytes"] = max(stats["max_score_cache_bytes"],
        workspace.replacement_score_payload_bytes + sys.getsizeof(cache))


def _replacement_assignment_scores(workspace, descriptors):
    """Exact C(H) for deletion starts, using the assignment-only fitter in bounded batches.

    Evidence and pattern tables are shared with the search workspace. Temporary fit/assignment
    caches are isolated and cleared after each batch, so rejected starts retain no membership in
    the search workspace. Existing search assignment records supply exact cache hits. Only the
    scalar scores are memoized across replacement rounds under exact packed panel keys.
    The memo retains at most 32 MiB of estimated Python object storage per block; eviction
    changes only work, never admission. Rejected starts retain no tied-state membership.
    """
    scores = np.empty(len(descriptors), dtype=np.float64)
    if not descriptors:
        return scores
    stats = getattr(workspace, "replacement_ranking_counts", None)
    if stats is None:
        stats = workspace.replacement_ranking_counts = dict(
            starts=0, assignment_evaluations=0, assignment_cache_hits=0, batches=0,
            max_batch_size=0, max_batch_payload_bytes=0, retained_starts=0,
            score_cache_hits=0, score_cache_evictions=0, max_score_cache_bytes=0)
    if not hasattr(workspace, "replacement_score_cache"):
        workspace.replacement_score_cache = OrderedDict()
        workspace.replacement_score_payload_bytes = 0
    score_cache = workspace.replacement_score_cache
    stats["starts"] += len(descriptors)
    k = descriptors[0][0].k - 1
    n, length = workspace.observed.shape
    states = k * (k + 1) // 2 + k + 1
    # Worst case: every state retained for every sample, plus offsets, A/cost/unknown arrays,
    # and the int64 panel. Dense A-step masks are separately bounded by the fitter
    # and shared evidence/pattern arrays are outside this compact-membership estimate.
    per_start = 4 * n * states + 48 * n + 8 * k * length
    batch_size = max(1, min(_REPLACEMENT_SCORE_BATCH, _REPLACEMENT_SCORE_BYTES // max(1, per_start)))
    scoring = copy(workspace)
    scoring.assignments = OrderedDict()
    scoring.fits = OrderedDict()
    scoring.transitions = OrderedDict()
    scoring.fixed_point_keys = set()
    pending, indices, pending_keys = [], [], []

    def finish_batch():
        if not pending:
            return
        records = fit_starts(scoring, pending, 0)
        for index, key, record in zip(indices, pending_keys, records):
            scores[index] = record[5]
            _remember_replacement_score(workspace, key, record[5])
        stats["assignment_evaluations"] += len(pending)
        stats["batches"] += 1
        stats["max_batch_size"] = max(stats["max_batch_size"], len(pending))
        stats["max_batch_payload_bytes"] = max(stats["max_batch_payload_bytes"], len(pending) * per_start)
        scoring.assignments.clear()
        scoring.fits.clear()
        pending.clear()
        indices.clear()
        pending_keys.clear()

    for index, (birth, deleted) in enumerate(descriptors):
        start = np.ascontiguousarray(np.delete(birth.haplotypes, deleted, axis=0))
        key = workspace.state_key(start)
        record = workspace.cached_assignment(key)
        if record is not None:
            scores[index] = record[3]
            stats["assignment_cache_hits"] += 1
            _remember_replacement_score(workspace, key, record[3])
        elif key in score_cache:
            scores[index] = score_cache[key]
            score_cache.move_to_end(key)
            stats["score_cache_hits"] += 1
        else:
            pending.append(start)
            indices.append(index)
            pending_keys.append(key)
            if len(pending) == batch_size:
                finish_batch()
    finish_batch()
    return scores


def replacements(workspace, parent, births, config, limit=None, omissions=None):
    """Same-K starts: every K + 1 birth with any one row deleted, distinct by the remaining rows, taken by whole
    NLL tie classes of the births (best first) while at most ``limit`` (default ``max_replacement_children``)
    starts are taken. Any straddling class is admitted through 2 x limit starts. A larger class is omitted
    whole, and lower-ranked classes are counted separately. Returns (children, all omitted starts)."""
    width = parent.sites
    limit = config.search.max_replacement_children if limit is None else int(limit)
    pool = [birth for birth in deduplicate(births) if birth.k == parent.k + 1]
    if not pool:
        return (), 0
    classes = tie_classes([birth.nll for birth in pool], _nll_quantum(pool, None),
                          content=[birth.key for birth in pool])
    by_class: dict[int, list] = {}
    for index in classes.order:
        by_class.setdefault(int(classes.classes[int(index)]), []).append(pool[int(index)])
    seen: set[bytes] = set()
    descriptors, descriptor_classes = [], []
    for cls in sorted(by_class):
        unit, unit_keys = [], set()
        for birth in by_class[cls]:
            for deleted in range(birth.k):
                offset = deleted * width
                key = birth.key[:offset] + birth.key[offset + width:]
                if key in seen or key in unit_keys:
                    continue
                unit_keys.add(key)
                unit.append((birth, deleted))
        descriptors.extend(unit)
        descriptor_classes.extend([cls] * len(unit))
        seen |= unit_keys
    retained, _ = whole_class_cut(descriptors, descriptor_classes, limit, omissions,
                                  cause="replacement", scope=f"K={parent.k} birth NLL classes")
    omitted = len(descriptors) - len(retained)
    descriptors = retained
    starts = [np.ascontiguousarray(np.delete(birth.haplotypes, deleted, axis=0)) for birth, deleted in descriptors]
    synchronized, refitted = fit_panels(workspace, starts, config.model.max_iterations)
    children = deduplicate((*synchronized, *refitted))
    return tuple(child for child in children if child.k == parent.k), omitted


@dataclass(frozen=True)
class OrdinaryMoves:
    """One ordinary expansion: every fitted neighbour (deaths, gauge rewires, data births),
    the K + 1 births among them, starts omitted by the generator size, and limitations."""

    children: tuple
    births: tuple
    omitted: int
    limits: tuple
    omissions: tuple = ()


def ordinary_moves(workspace, solution, likelihoods, growth, ceiling, cuts: MaximumCuts,
                   config, counter=None) -> OrdinaryMoves:
    """Whole death and gauge proposal families, in that search-priority order, then births.

    Admit whole families through twice max_proposals_per_expansion; never choose a byte prefix.
    """
    search = config.search
    buckets = {-1: [], 0: []}
    limits = []
    omissions = []
    if solution.k > 1:
        buckets[-1].extend(death_starts(solution))
        weights = assignment_graph(solution.assignments, solution.k)
        # The historical effective budget is 4 x max_cut_ties. Its single tied class may extend
        # through twice that effective budget; probe one further cut to detect whole-class omission.
        cap = 4 * search.max_cut_ties
        partitions = cuts.partitions(weights, 2 * cap + 1)
        if len(partitions) > 2 * cap:
            _note_omission(omissions, "gauge_tie_class_at_budget", len(partitions),
                           "gauge maximum-cut class", "lower_bound")
            partitions = ()
        buckets[0].extend(gauge_starts(solution, likelihoods, partitions, workspace.quanta.nll))
    unique = {}
    for offset, starts in buckets.items():
        by_key = {}
        for start in starts:
            value = np.ascontiguousarray(start, dtype=np.int64)
            by_key.setdefault((value.shape, value.tobytes()), value)
        unique[offset] = [by_key[key] for key in sorted(by_key, reverse=rev("d9p"))]
    total = sum(len(starts) for starts in unique.values())
    ordered = [start for offset in (-1, 0) for start in unique[offset]]
    classes = [offset for offset in (-1, 0) for start in unique[offset]]
    retained, _ = whole_class_cut(ordered, classes, search.max_proposals_per_expansion, omissions,
                                  cause="proposal", scope="death/gauge proposal families")
    omitted = total - len(retained)
    synchronized, refitted = fit_panels(workspace, retained, config.model.max_iterations)
    children = list(deduplicate((*synchronized, *refitted)))
    if solution.k < ceiling:
        births = data_births(workspace, solution, growth, config.model.max_iterations, search,
                             counter=counter)
        omissions.extend(births.omissions)
        children.extend(child for child in births.children if child.k == solution.k + 1)
    children = deduplicate(children)
    births = tuple(child for child in children if child.k == solution.k + 1)
    limits.extend(item["cause"] for item in omissions if item.get("kind") != "ranked_truncation")
    return OrdinaryMoves(children, births, omitted, tuple(dict.fromkeys(limits)), tuple(omissions))
