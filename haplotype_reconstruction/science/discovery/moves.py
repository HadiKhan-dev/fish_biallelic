"""Proposal generators of the reversible search (each move proposes start panels; the
fixed-K fitter refits them and the cavity score decides).

Owns: the K = 1 initial starts, death, gauge-rewire, data-birth, soft-birth and birth-delete replacement
generators, their growth tables (GrowthInputs) and the maximum-cut cache.
Must not: score or select panels (the cavity score and search.py decide), read storage, the environment or a clock,
or log.
Method: docs/methods/discovery.md, units di13-di16 and di19-di20.
Identity: death/gauge start families are admitted whole through twice ``max_proposals_per_expansion``; data births
consume whole excess/coverage tie groups (budgets ``data_birth_attempts``,
``data_birth_children``, ``data_birth_tie_budget_factor``); gauge rewires use the best-found cuts of one
heuristic search at every K, with canonical cut-family truncation; replacements follow births in tie-class order
(``max_replacement_children``).

* K = 1 data seeds: HDBSCAN clusters of samples by expected genotype agreement, each pooled
  into one row (fallback: the most decisive sample's dosage row).
* Deaths: every one-row deletion.
* Gauge rewires: for a best-found cut of the assignment graph (rows linked by samples assigned to
  both), flip the sites where both sides are internally constant and differ: per site to the
  better allele for the samples within one side ("sitewise"), and all such sites ("opposite").
* Data births: K + 1 starts from the samples with the largest excess NLL over their sitewise
  genotype oracle (subtraction and dosage seeds).
* Soft births: K + 1 starts from soft-residual consensus rows (residual_births.py).
* Birth-delete replacements: a birth followed by deleting any row (same K).

Proposal order does not use rounding-level differences, perturbation-level differences or array layout. K = 1
similarities use one fixed summation order and are exactly symmetric. Consensus and dosage rows treat values
within the ``probability`` quantum of 1/2 as exactly 1/2 (population ALT fraction, then REF); gauge rewires
need a gain above the ``nll`` quantum. Data births rank samples by tie classes of excess NLL (kind ``sample``;
an all-unknown sample's excess is exactly 2 lambda n_obs), then observed sites, then canonical sample order,
consuming tied samples as whole groups. Replacements follow births in tie-class order.
"""

from __future__ import annotations

import math
import zlib
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numba import prange, types

from haplotype_reconstruction.contracts.roles import record_field
from haplotype_reconstruction.core.kernels import kernel
from haplotype_reconstruction.core.ties import (
    BOUNDARY_GUARD,
    NOISE_MOTION,
    WIDE_CLASS_QUANTA,
    fixed_policy,
    tie_classes,
)

from .conventions import rev
from .fixed_k.fit import fit_families, fit_panels, fit_starts
from .fixed_k.workspace import FixedKWorkspace
from .panel import (
    FitResult,
    FixedKSolution,
    _nll_quantum,
    canonical_solution,
    deduplicate,
    has_distinct_rows,
    panel_key,
    sort_solutions,
    whole_class_cut,
)
from .parameters import COUNT_CHUNK as _COUNT_CHUNK
from .parameters import (
    GAUGE_CUT_GW_PLANES,
    GAUGE_CUT_GW_SEED,
    GAUGE_CUT_SDP_SWEEPS,
    GAUGE_CUT_SDP_TOLERANCE,
    MAX_CUT_CACHE_ENTRIES,
    DiscoveryParameters,
    SearchParameters,
)
from .residual_births import (
    HALF_BAND,
    alt_fraction_families,
    exact_hdbscan_labels,
    half_band_rows,
    hardening_fallbacks,
    note_band,
    sample_sum,
    scalar_log_floored,
    soft_candidates,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

    from haplotype_reconstruction.core.ties import TieClasses, TieCounter

    from .residual_births import ResidualInputs

__all__ = [
    "MaximumCuts",
    "GrowthInputs",
    "DataBirths",
    "SoftBirths",
    "OrdinaryMoves",
    "growth_inputs",
    "initial_solutions",
    "ordinary_moves",
    "data_births",
    "soft_births",
    "replacements",
    "data_birth_order",
    "collect_birth_starts",
    "gauge_cut_family",
]

# Supported lazy overloads for the declared kernel routes.
_INT64_2C = types.Array(types.int64, 2, "C", readonly=False)
_BOOLEAN_1C = types.Array(types.boolean, 1, "C", readonly=False)
_INT64_1C = types.Array(types.int64, 1, "C", readonly=False)
_INT64_2C_RO = types.Array(types.int64, 2, "C", readonly=True)
_FLOAT64_3C = types.Array(types.float64, 3, "C", readonly=False)
_FLOAT64_2C = types.Array(types.float64, 2, "C", readonly=False)


_TINY = np.finfo(np.float64).tiny
# Maximum-cut results memoised per block search (an exact memo of a pure function).
# Assignment-only replacement screening retains at most this many starts and this estimated
# worst-case compact-membership payload per batch. These are memory bounds, not search budgets.


# ---------------------------------------------------------------------------------------------
# K = 1 data seeds
# ---------------------------------------------------------------------------------------------
@kernel(overloads=((_FLOAT64_3C,),), cache=True, parallel=True, nogil=True)
def expected_agreement(likelihoods):
    """S[i, j] = mean over sites of sum_g P_i(g) P_j(g), each entry summed over sites (then genotypes 0, 1, 2) in
    one fixed order and stored in both triangles, independently of sample order, threads and BLAS layout."""
    n, n_sites, _ = likelihoods.shape
    similarity = np.empty((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(i, n):
            total = 0.0
            for site in range(n_sites):
                total += (
                    likelihoods[i, site, 0] * likelihoods[j, site, 0]
                    + likelihoods[i, site, 1] * likelihoods[j, site, 1]
                    + likelihoods[i, site, 2] * likelihoods[j, site, 2]
                )
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


def _soft_cluster_seeds(
    likelihoods,
    n_seeds,
    min_cluster_size,
    counter=None,
    omissions=None,
    landmark_budget=0,
    *,
    reverse: bool = False,
    keep: tuple[str, ...] = (),
):
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
        labels = exact_hdbscan_labels(
            distance,
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_cluster_size),
            allow_single_cluster=False,
        )
        clusters = [np.flatnonzero(labels == label) for label in np.unique(labels) if label != -1]
    alt_fraction = 0.5 * likelihoods[:, :, 1] + likelihoods[:, :, 2]
    population = sample_sum(alt_fraction) / len(alt_fraction)
    marginals = alt_fraction_families(population, False, reverse=reverse, keep=keep)
    note_band(counter, "discovery.k1.half_band", population, 0.5, HALF_BAND)
    rows, classes = [], []
    seen: set[bytes] = set()
    for size in sorted({len(members) for members in clusters}, reverse=True):
        group: list[np.ndarray] = []
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
    rows, cut = whole_class_cut(
        rows, classes, n_seeds, omissions, cause="k1_cluster_seed", scope="initial cluster-size classes"
    )
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
    floor = np.where(all_tied, ordered[..., 0], np.where(upper_tied, ordered[..., 1], ordered[..., 2]))
    members = values >= floor[..., None]
    low = np.argmax(members, axis=-1)
    high = 2 - np.argmax(members[..., ::-1], axis=-1)
    if counter is not None:
        counter.add_counts(
            kind,
            decisions=int(low.size),
            inside_tie=int(np.count_nonzero(low != high)),
            boundary=int(
                np.count_nonzero(
                    (gaps != 0.0) & (np.abs(gaps - HALF_BAND) <= BOUNDARY_GUARD * NOISE_MOTION["probability"])
                )
            ),
        )
        # Three members can span at most 2q, so no winning class reaches the 4q width guard.
    return low, high


def _dosage_families(likelihoods, counter=None, kind="discovery.dosage.genotype"):
    low, high = _dosage_extremes(likelihoods, counter, kind)
    return (low,) if np.array_equal(low, high) else (low, high)


def _dosage_seeds(likelihoods, samples, counter=None, *, reverse: bool, keep: tuple[str, ...] = ()):
    """Complete dosage and population-half families of each decisive sample, deduplicated."""
    samples = np.asarray(samples, dtype=np.int64)
    dosages = _dosage_families(likelihoods[samples], counter, "discovery.k1.dosage")
    population_alt = sample_sum(0.5 * likelihoods[..., 1] + likelihoods[..., 2]) / len(likelihoods)
    population_rows = alt_fraction_families(population_alt, False, reverse=reverse, keep=keep)
    return _unique_seeds(
        row
        for dosage in dosages
        for population_row in population_rows
        for row in ((dosage == 2) | ((dosage == 1) & population_row[None, :]))
    )


def _most_decisive_class(likelihoods, observed, sample_quantum, *, reverse: bool):
    """The samples tied for the fallback seed: most observed sites (exact), then the best class of decisiveness (sum
    of the largest GL over observed sites; tie classes of kind ``sample``, descending), in sample order."""
    observed_sites = observed.sum(axis=1)
    candidates = np.flatnonzero(observed_sites == observed_sites.max())
    decisiveness = np.where(observed, likelihoods.max(axis=2), 0.0).sum(axis=1)
    classes = tie_classes(
        cast(Sequence[float], decisiveness[candidates]),
        fixed_policy("discovery.k1", sample_quantum, reverse=reverse).quantum(),
        descending=True,
    )
    first = classes.classes[classes.winner]
    return [int(candidates[i]) for i in range(len(candidates)) if classes.classes[i] == first]


def initial_solutions(
    workspace: FixedKWorkspace,
    likelihoods: np.ndarray,
    active: np.ndarray,
    config: DiscoveryParameters,
    counter: TieCounter | None = None,
    omissions: list[dict] | None = None,
    *,
    keep: tuple[str, ...] = (),
) -> tuple[FixedKSolution, ...]:
    """K=1 starts from whole cluster-size or fallback decisive-sample classes.

    The fallback is used only if clustering supplies no seed class. If all available cluster seeds
    are omitted, no unscored population row substitutes for them: the block has no incumbent.
    """
    seed_likelihoods = likelihoods if np.all(active) else np.ascontiguousarray(likelihoods[active])
    search = config.search
    omitted: list[dict] = []
    seeds = _soft_cluster_seeds(
        seed_likelihoods,
        search.data_seed_count,
        search.seed_min_cluster_size,
        counter,
        omitted,
        search.seed_landmark_budget,
        reverse=workspace.reverse,
        keep=keep,
    )
    if not seeds and not any("landmark" not in event["cause"] for event in omitted):
        seed_observed = workspace.observed if np.all(active) else workspace.observed[active]
        samples = _most_decisive_class(
            seed_likelihoods, seed_observed, workspace.quanta.sample, reverse=workspace.reverse
        )
        seeds = _dosage_seeds(seed_likelihoods, samples, counter, reverse=workspace.reverse, keep=keep)
        seeds, _ = whole_class_cut(
            seeds,
            [0] * len(seeds),
            search.data_seed_count,
            omitted,
            cause="k1_sample_seed",
            scope="initial decisive-sample class",
        )
    if omissions is not None:
        omissions.extend(omitted)
    families = cast(
        list[tuple[FitResult, ...]],
        fit_families(workspace, [seed[None, :] for seed in seeds], config.model.max_iterations),
    )
    solutions = deduplicate(
        (canonical_solution(fit, workspace) for family in families for fit in family), reverse=workspace.reverse
    )
    ordered = sort_solutions(solutions, workspace.quanta, reverse=workspace.reverse)
    classes = tie_classes(
        [solution.nll for solution in ordered],
        fixed_policy("discovery.k1", workspace.quanta.nll, reverse=workspace.reverse).quantum(),
        reverse=workspace.reverse,
    ).classes
    kept, _ = whole_class_cut(
        ordered,
        classes.tolist(),
        search.data_start_beam_width,
        omissions,
        cause="k1_fit_beam",
        scope="initial fitted NLL classes",
    )
    return tuple(kept)


# ---------------------------------------------------------------------------------------------
# Deaths and gauge rewires
# ---------------------------------------------------------------------------------------------
def death_starts(solution):
    """Every one-row deletion of a fitted panel (K >= 2)."""
    if solution.k <= 1:
        return ()
    return tuple(np.ascontiguousarray(np.delete(solution.haplotypes, index, axis=0)) for index in range(solution.k))


@kernel(
    overloads=(
        (
            _INT64_2C_RO,
            types.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
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


@kernel(
    overloads=(
        (
            _INT64_2C,
            _BOOLEAN_1C,
        ),
    ),
    cache=True,
    nogil=True,
)
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


@kernel(
    overloads=(
        (
            _INT64_2C,
            _BOOLEAN_1C,
            _INT64_1C,
        ),
    ),
    cache=True,
    nogil=True,
)
def _local_search_cut(weights, initial, order):
    """Strictly improving single-vertex moves in ``order`` with exact integer gains.

    Gain is same-side weight minus cross weight. A move negates its own gain and changes every
    other gain by +2w on the moved side or -2w opposite it. Updating gains costs O(K) per move;
    initialization costs O(K^2). Each move raises the integer cut weight by at least one, so a
    start costs O(K^2 + K W), where W is the graph's total upper-triangle weight. The endpoint is
    locally optimal with respect to the visited, unpinned vertices.
    """
    side = initial.copy()
    k = len(weights)
    gain = np.zeros(k, dtype=np.int64)
    for index in range(k):
        total = 0
        for other in range(k):
            if other == index:
                continue
            if side[index] == side[other]:
                total += weights[index, other]
            else:
                total -= weights[index, other]
        gain[index] = total
    while True:
        changed = False
        for position in range(len(order)):
            index = order[position]
            if gain[index] > 0:
                moved = not side[index]
                side[index] = moved
                gain[index] = -gain[index]
                for other in range(k):
                    if other == index:
                        continue
                    if side[other] == moved:
                        gain[other] += 2 * weights[other, index]
                    else:
                        gain[other] -= 2 * weights[other, index]
                changed = True
        if not changed:
            return side


@kernel(
    overloads=(
        (
            _FLOAT64_2C,
            _FLOAT64_2C,
            types.int64,
            types.float64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _mixing_sweeps(weights, vectors, sweeps, tolerance):
    """Burer-Monteiro mixing method for the max-cut SDP (maximise 1/4 sum_ij w_ij (1 - <v_i, v_j>) over unit vectors):
    exact block-coordinate updates v_i <- -normalise(sum_j w_ij v_j) in place, at most ``sweeps`` sweeps, stopping
    when a sweep's squared change is below ``tolerance``. A vertex with no weighted neighbour keeps its vector.
    Returns the sweeps used. O(sweeps K^2 r) for rank r."""
    k, rank = vectors.shape
    direction = np.empty(rank)
    for sweep in range(sweeps):
        change = 0.0
        for i in range(k):
            for c in range(rank):
                direction[c] = 0.0
            for j in range(k):
                if j != i and weights[i, j] != 0.0:
                    for c in range(rank):
                        direction[c] += weights[i, j] * vectors[j, c]
            norm = 0.0
            for c in range(rank):
                norm += direction[c] * direction[c]
            norm = math.sqrt(norm)
            if norm == 0.0:
                continue
            for c in range(rank):
                updated = -direction[c] / norm
                change += (updated - vectors[i, c]) ** 2
                vectors[i, c] = updated
        if change < tolerance:
            return sweep + 1
    return sweeps


def _gauge_cut_starts(weights, planes=None, labels=None):
    """Goemans-Williamson starts for the single-vertex cut search.

    The max-cut SDP is solved in rank ceil(sqrt(2K)) + 1 by the mixing method (Burer-Monteiro) from a random start;
    each of ``planes`` random hyperplanes g gives the cut sign(<v_i, g>) >= 0, complemented so that vertex 0 stays on
    side False (the searches' pinned vertex). Returns the distinct non-empty cuts in hyperplane order.

    Canonical order: with ``labels`` (one content row per vertex, the founder haplotypes) the SDP, its random start
    and the hyperplanes are computed on the weights in founder-content order (rows sorted by their bytes, equal rows
    by input index) and mapped back, and the generator is seeded with (GAUGE_CUT_GW_SEED, CRC-32 of those reordered
    int64 weights). The weights count samples per row pair, so they do not depend on the sample order; the starts,
    as bipartitions of founders, are then identical under any permutation of distinct founder rows. Without labels
    the input order is used (the production caller passes the canonical, content-sorted panel, so both agree).

    Approximation: the solve is a finite, rank-factorised approximation of the SDP (at most GAUGE_CUT_SDP_SWEEPS
    sweeps; GAUGE_CUT_SDP_TOLERANCE bounds the vectors' movement in one sweep, not the objective error), and no
    dual certificate is computed in production. Its accuracy only decides which extra starts are tried: every start
    is a valid pinned cut, and every end point is found and scored exactly by the integer search, so a poor or
    unconverged solve cannot return an invalid or mis-scored cut. The singleton and alternating starts are always
    searched, so adding GW starts cannot decrease the best cut weight found. The returned family (endpoints tied
    at that weight, truncated to max_ties) can change, and a larger found family can exceed the gauge budget and
    be truncated
    (gauge_cut_family). Cost O(GAUGE_CUT_SDP_SWEEPS K^2 sqrt(K) + planes K sqrt(K))."""
    planes = GAUGE_CUT_GW_PLANES if planes is None else int(planes)
    matrix = np.ascontiguousarray(weights, dtype=np.int64)
    k = len(matrix)
    if planes <= 0 or k < 3 or not np.any(matrix):
        return []
    if labels is None:
        permutation = np.arange(k, dtype=np.int64)
    else:
        keys = [np.ascontiguousarray(row).tobytes() for row in labels]
        permutation = np.asarray(sorted(range(k), key=lambda index: (keys[index], index)), dtype=np.int64)
    canonical = np.ascontiguousarray(matrix[np.ix_(permutation, permutation)])
    rng = np.random.default_rng([GAUGE_CUT_GW_SEED, zlib.crc32(canonical.tobytes())])
    rank = int(math.ceil(math.sqrt(2 * k))) + 1
    vectors = rng.standard_normal((k, rank))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    _mixing_sweeps(canonical.astype(np.float64), vectors, int(GAUGE_CUT_SDP_SWEEPS), float(GAUGE_CUT_SDP_TOLERANCE))
    hyperplanes = rng.standard_normal((planes, rank))
    starts = []
    seen = set()
    for normal in hyperplanes:
        side = np.empty(k, dtype=bool)
        side[permutation] = (vectors @ normal) >= 0.0
        if side[0]:
            side = ~side
        key = side.tobytes()
        if key not in seen and np.any(side):
            seen.add(key)
            starts.append(np.ascontiguousarray(side))
    return starts


class MaximumCuts:
    """Deterministic best-found bipartitions of assignment graphs (memoised per block).

    Vertex 0 stays on side False (complement cuts are one cut). At every K, degree-ordered single-vertex
    improving searches run from every singleton, an alternating start and the Goemans-Williamson starts. Return
    distinct endpoints at the best (cross, within) weight, at most ``max_ties``, in lexicographic bit order
    (reversed under reversed conventions). The cuts are locally optimal with respect to unpinned vertices;
    global maxima are not certified.
    """

    def __init__(self, max_ties: int, *, reverse: bool = False, keep: tuple[str, ...] = ()):
        self.reverse = reverse
        self.keep = tuple(keep)
        self.max_ties = int(max_ties)
        self._cache: OrderedDict = OrderedDict()

    def heuristic(self, k: int) -> bool:
        """Whether gauge cuts of a K-row panel come from the heuristic search: every K > 1."""
        return k > 1

    def partitions(self, weights, max_ties=None, labels=None):
        """Up to ``max_ties`` best-found cuts (default the configured number).

        Return distinct endpoints at the best found (cross, within) weight, locally optimal with respect to
        unpinned vertices. Global maxima are not certified. ``labels`` supplies one founder-content row per
        vertex to put the Goemans-Williamson starts in founder-content order.
        """
        max_ties = self.max_ties if max_ties is None else int(max_ties)
        matrix = np.ascontiguousarray(weights, dtype=np.int64)
        if len(matrix) < 2:
            return ()
        label_key = None if labels is None else np.ascontiguousarray(labels).tobytes()
        key = (len(matrix), max_ties, matrix.tobytes(), label_key)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        k = len(matrix)
        reverse = rev("d9", reverse=self.reverse, keep=self.keep)
        order = np.asarray(
            sorted(range(1, k), key=lambda index: (-int(np.sum(matrix[index])), -index if reverse else index)),
            dtype=np.int64,
        )
        candidates = []
        for index in range(1, k):
            side = np.zeros(k, dtype=bool)
            side[index] = True
            candidates.append(_local_search_cut(matrix, side, order))
        alternating = np.zeros(k, dtype=bool)
        alternating[order[::2]] = True
        candidates.append(_local_search_cut(matrix, alternating, order))
        for side in _gauge_cut_starts(matrix, labels=labels):
            candidates.append(_local_search_cut(matrix, side, order))
        unique: dict[bytes, np.ndarray] = {}
        for side in candidates:
            if np.any(side):
                unique.setdefault(side.tobytes(), side)
        scored = [(_cut_score(matrix, side), side.tobytes(), side) for side in unique.values()]
        if scored:
            best_cross = max(item[0][0] for item in scored)
            best_within = min(item[0][1] for item in scored if item[0][0] == best_cross)
            winners = sorted(
                (item for item in scored if item[0] == (best_cross, best_within)),
                key=lambda item: item[1],
                reverse=reverse,
            )
            result = tuple(item[2].copy() for item in winners[:max_ties])
        else:
            result = ()
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > MAX_CUT_CACHE_ENTRIES:
            self._cache.popitem(last=False)
        return result


def gauge_cut_family(cuts, weights, haplotypes, budget, omissions=None):
    """The gauge cuts of a panel: the tied best cuts MaximumCuts finds, at most ``budget`` of them.

    A found family larger than ``budget`` is truncated canonically. The cuts come in lexicographic order of
    their sides (vertex 0, the
    content-first founder of the canonical panel, on side False; reversed under reversed conventions), with the
    vertices in founder-content order (the caller's canonical panel; the Goemans-Williamson starts are computed in
    that order too), so the first ``budget`` cuts are a function of the founders' content, not of their or the
    samples' input order (for distinct founder rows; equal rows keep their input order). The truncation is recorded
    as a tie_class_truncated omission (cause gauge_tie_class_truncated) whose count is the number of distinct cuts
    found at the best found weight: exact for the found family and a lower bound on all cuts of that weight (other
    cuts of that weight may not have been reached; it is not a count of global maximum cuts, which the heuristic
    search does not certify). The scope names the weight and how many were kept. Distinct founder rows are a
    precondition (discovery panels are deduplicated). This cut-family truncation is separate from the later
    whole-class budget on death/gauge start families (max_proposals_per_expansion)."""
    k = len(weights)
    # K singleton/alternating starts plus at most P GW starts bound the complete found family by K + P.
    family = cuts.partitions(weights, k + 2 + max(0, GAUGE_CUT_GW_PLANES), labels=haplotypes)
    if len(family) > budget:
        best_weight = int(_cut_score(np.ascontiguousarray(weights, dtype=np.int64), np.ascontiguousarray(family[0]))[0])
        _note_omission(
            omissions,
            "gauge_tie_class_truncated",
            len(family),
            f"gauge cuts at the best found weight {best_weight} (kept {budget} of {len(family)} found)",
            "lower_bound",
            kind="tie_class_truncated",
        )
    return tuple(family[:budget])


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
            current = -math.fsum(
                scalar_log_floored(
                    np.ascontiguousarray(likelihoods[resolving, site, dosage], dtype=np.float64), _TINY
                ).tolist()
            )
            opposite = -math.fsum(
                scalar_log_floored(
                    np.ascontiguousarray(likelihoods[resolving, site, 2 - dosage], dtype=np.float64), _TINY
                ).tolist()
            )
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

    oracle_nll: np.ndarray = record_field("state", dtype="float64", axes=("N",), units="nats")
    decisiveness: np.ndarray = record_field("state", dtype="float64", axes=("N",))
    dosages: tuple[np.ndarray, ...] = record_field("state", dtype="tuple[np.ndarray, ...]")
    seed_rows: tuple[np.ndarray, ...] = record_field("state", dtype="tuple[np.ndarray, ...]")
    active: np.ndarray = record_field("state", dtype="bool", axes=("N",))
    observed_sites: np.ndarray = record_field("state", dtype="int64", axes=("N",))


def growth_inputs(
    likelihoods: np.ndarray,
    observed: np.ndarray,
    counter: TieCounter | None = None,
    *,
    reverse: bool = False,
    keep: tuple[str, ...] = (),
) -> GrowthInputs:
    """The block's per-sample growth tables, computed once per search.

    ``likelihoods``: float64[N, L, 3] genotype likelihoods; ``observed``: bool[N, L]. Returns GrowthInputs: each
    sample's sitewise genotype-oracle NLL (float64[N]) and decisiveness, the extremal dosage families, one dosage
    seed row family per member of the population-half hardening family (alt_fraction_families), the samples with
    depth and their observed-site counts (int64[N]). ``counter`` (optional) receives the dosage band decisions.
    Mutates only ``counter``; raises nothing of its own.
    """
    maximum = np.max(likelihoods, axis=2)
    dosages = _dosage_families(likelihoods, counter)
    active = np.any(observed, axis=1)
    oracle_nll = -np.sum(
        np.where(observed, scalar_log_floored(np.ascontiguousarray(maximum, dtype=np.float64), _TINY), 0.0), axis=1
    )
    decisiveness = np.sum(np.where(observed, maximum, 0.0), axis=1)
    alt_fraction = 0.5 * likelihoods[..., 1] + likelihoods[..., 2]
    observed_count = np.sum(observed, axis=0)
    population_alt = np.divide(
        sample_sum(np.where(observed, alt_fraction, 0.0)),
        observed_count,
        out=np.full(likelihoods.shape[1], 0.5),
        where=observed_count > 0,
    )
    seed_rows = []
    population_rows = alt_fraction_families(population_alt, False, reverse=reverse, keep=keep)
    for dosage in dosages:
        for population_row in population_rows:
            rows = np.zeros_like(dosage)
            rows[observed & (dosage == 2)] = 1
            ambiguous = (~observed) | (dosage == 1)
            rows[ambiguous & population_row[None, :]] = 1
            seed_rows.append(rows)
    observed_sites = np.sum(observed, axis=1).astype(np.int64)
    return GrowthInputs(oracle_nll, decisiveness, dosages, tuple(seed_rows), active, observed_sites)


def _data_seed_keys(growth, sample, parent_rows):
    """Union dosage, population-half and subtraction seeds before the whole-class budget.

    Missing-cell genotype ties matter to subtraction starts even when the direct
    seed at that cell comes from the population hardening family.
    """
    # The same integer subtraction/clip for all parent rows in one array operation;
    # retain dosage-then-parent order before the exact union.
    seeds = [row for dosage in growth.dosages for row in np.clip(dosage[sample][None, :] - parent_rows, 0, 1)]
    seeds.extend(rows[sample] for rows in growth.seed_rows)
    return sorted({np.asarray(row, dtype=np.int8).tobytes() for row in seeds})


@dataclass(frozen=True)
class DataBirths:
    """K+1 data-birth children; tie_cut marks a wholly omitted class at the hard cap.

    omitted counts that class's distinct starts (search limit data_birth_tie_budget).
    """

    children: tuple[FixedKSolution, ...] = record_field("state", dtype="tuple[FixedKSolution, ...]")
    tie_cut: bool = record_field("state", dtype="bool", default=False)
    omitted: int = record_field("state", dtype="int64", default=0)
    omissions: tuple = record_field("state", dtype="tuple", default=())


def data_birth_order(
    excess: np.ndarray, observed_sites: np.ndarray, samples: Iterable[int], quantum: float, *, reverse: bool
) -> tuple[list[int], list[tuple[int, ...]], TieClasses | None]:
    """(samples in data-birth order, their tie-group ids, the TieClasses) of the samples with depth.

    Order: tie classes of excess NLL (kind ``sample``, descending: the most misfit first), then more
    observed sites, then canonical sample order (inside a class wider than 4 quanta: excess first,
    the width guard). A tie group is the samples of one class with one observed-site count (a wide
    class is one group); groups are consumed whole.
    """
    samples = [int(sample) for sample in samples]
    if not samples:
        return [], [], None
    classes = tie_classes(
        [float(excess[sample]) for sample in samples],
        fixed_policy("discovery.data_birth", quantum, reverse=reverse).quantum(),
        secondary=[(-int(observed_sites[sample]),) for sample in samples],
        content=[sample for sample in samples],
        descending=True,
        motion=NOISE_MOTION["sample"],
        reverse=reverse,
    )
    wide = classes.spans > WIDE_CLASS_QUANTA * quantum
    order: list[int]
    groups: list[tuple[int, ...]]
    order, groups = [], []
    for index in classes.order:
        sample = samples[int(index)]
        cls = int(classes.classes[int(index)])
        order.append(sample)
        groups.append((cls,) if wide[cls] else (cls, int(observed_sites[sample])))
    return order, groups, classes


# Samples per bulk seed array when the omitted data-birth starts are counted (a memory bound only: at most
# (dosage families x K + seed families) x chunk x L int8 cells at a time).


def collect_birth_starts(
    order: Sequence[Any],
    groups: Sequence[Hashable],
    seeds_of: Callable[[Any], Iterable[Any]],
    budget: int,
    hard_cap: int,
    omissions: list[dict] | None = None,
    *,
    key_of: Callable[[Any], bytes] = panel_key,
    count_rest: Callable[[Sequence[Any], set[bytes]], int] | None = None,
) -> tuple[list[list[Any]], bool]:
    """Union each excess/coverage class's distinct starts before its hard-cap decision.

    The attempt budget is checked between groups. A class crossing the hard cap is omitted whole;
    no sample, founder-row or seed enumeration order can select a privileged prefix.
    ``key_of`` also accepts deferred starts: omitted classes need their identities only.
    ``count_rest(items, seen) -> int`` (optional): once collection has stopped, the number of
    distinct start keys of ``items`` (the rest of ``order``) not in ``seen``, computed in bulk. The omitted
    classes are only counted (the ranked-truncation diagnostic), and the count is the size of the same union,
    so it equals the group-by-group enumeration it replaces.
    """
    collected = []
    seen: set[bytes] = set()
    total = position = ranked = 0
    cut = stopped = False
    while position < len(order):
        if stopped and count_rest is not None:
            ranked += count_rest(order[position:], seen)
            break
        group_id = groups[position]
        unique: dict[bytes, object] = {}
        while position < len(order) and groups[position] == group_id:
            for row in seeds_of(order[position]):
                key = key_of(row)
                if key not in seen:
                    unique.setdefault(key, row if not stopped else None)
            position += 1
        seen.update(unique)
        if stopped or (total >= budget and total > 0):
            stopped = True
            ranked += len(unique)
        elif total + len(unique) > hard_cap:
            cut = stopped = True
            _note_omission(omissions, "data_birth_tie_class_at_budget", len(unique), "excess/coverage seed class")
        elif unique:
            collected.append([unique[key] for key in sorted(unique)])
            total += len(unique)
    _note_omission(
        omissions,
        "data_birth_ranked_truncation",
        ranked,
        "lower excess/coverage seed classes",
        kind="ranked_truncation",
    )
    return collected, cut


def data_births(
    workspace: FixedKWorkspace,
    solution: FixedKSolution,
    growth: GrowthInputs,
    max_iterations: int,
    search: SearchParameters,
    counter: TieCounter | None = None,
) -> DataBirths:
    """K + 1 children of a panel from its most misfit samples (``search``: SearchParameters).

    The parent is first refitted to its complete schedule family (unless the workspace
    certified both schedules fixed). Parent/sample pairs with depth are ranked together by excess NLL over their
    genotype oracle in tie classes (kind
    ``sample``; an excess within the quantum of an all-unknown sample's exact value 2 lambda n_obs
    is that value, so mathematically equal excesses are equal), then observed sites, then
    canonical sample order (data_birth_order). Each sample contributes both winning-class extremal dosage families: K
    subtraction seeds
    clip(dosage - row, 0, 1) and a direct dosage seed row for each, deduplicated before admission. Tie groups are
    consumed whole: novel seeds
    are collected group by group until the attempt budget (search.data_birth_attempts(K)) is
    reached, then fitted group by group until search.data_birth_children(K) distinct complete
    children exist or the budget is spent, both tested only at group boundaries. A group crossing
    search.data_birth_tie_budget_factor x the budget is omitted whole (``tie_cut``). If no child is
    complete, distinct assignment-only fits are returned by whole NLL classes at width K+1:
    the requested K without moving an unsupported row.
    """
    parents: tuple[FixedKSolution, ...]
    if solution.fixed_point and workspace.certifies_fixed_point(solution.haplotypes):
        parents = (solution,)
    else:
        (family,) = cast(list[tuple[FitResult, ...]], fit_families(workspace, [solution.haplotypes], max_iterations))
        distinct: dict[bytes, FixedKSolution] = {}
        for endpoint in family:
            parent = canonical_solution(endpoint, workspace)
            distinct.setdefault(parent.key, parent)
        parents = tuple(distinct[key] for key in sorted(distinct))
    parent_k = solution.k
    quantum = workspace.quanta.sample
    # Rank parent/sample pairs together and spend one budget for the complete parent family.
    excess = np.concatenate([np.maximum(parent.per_sample_cost - growth.oracle_nll, 0.0) for parent in parents])
    observed_sites = np.tile(growth.observed_sites, len(parents))
    active = np.tile(growth.active, len(parents))
    unknown_excess = 2.0 * workspace.penalty * observed_sites.astype(np.float64)
    excess = np.where(np.abs(excess - unknown_excess) <= quantum, unknown_excess, excess)
    order, groups, classes = data_birth_order(
        excess, observed_sites, np.flatnonzero(active), quantum, reverse=workspace.reverse
    )
    target = search.data_birth_children(parent_k)
    budget = search.data_birth_attempts(parent_k)
    hard_cap = budget * max(1, int(search.data_birth_tie_budget_factor))
    parent_rows = [
        tuple(sorted(np.asarray(row, dtype=np.int8).tobytes() for row in parent.haplotypes)) for parent in parents
    ]
    parent_keys = [b"".join(rows) for rows in parent_rows]
    seen = [set(rows) for rows in parent_rows]
    width = solution.sites

    def seeds_of(pair):
        parent_index, sample = divmod(pair, len(growth.active))
        parent = parents[parent_index]
        for key in _data_seed_keys(growth, sample, parent.haplotypes):
            if key in seen[parent_index]:
                continue
            seen[parent_index].add(key)
            # Inserting one binary row into sorted parent rows gives exactly panel_key
            # without building/sorting a full int64 panel for every omitted proposal.
            offset = bisect_left(parent_rows[parent_index], key) * width
            base = parent_keys[parent_index]
            yield base[:offset] + key + base[offset:], parent_index, key

    def count_rest(pairs, seen_keys) -> int:
        """Distinct start keys of the stopped pairs not in ``seen_keys``: the union of seeds_of over ``pairs``
        (a parent's own rows excluded; its earlier yields are in ``seen_keys``), in sample chunks of whole
        int8 row arrays instead of one Python generator per sample."""
        if width == 0:
            return 0
        pairs = np.asarray(pairs, dtype=np.int64)
        novel = set()
        for parent_index, parent in enumerate(parents):
            samples = pairs[pairs // len(growth.active) == parent_index] % len(growth.active)
            if not len(samples):
                continue
            own = set(parent_rows[parent_index])
            base = parent_keys[parent_index]
            haplotypes = np.asarray(parent.haplotypes, dtype=np.int8)
            rows: set[bytes] = set()
            for start in range(0, len(samples), _COUNT_CHUNK):
                chunk = samples[start : start + _COUNT_CHUNK]
                blocks = [
                    np.clip(
                        np.asarray(dosage[chunk], dtype=np.int8)[:, None, :] - haplotypes[None, :, :], 0, 1
                    ).reshape(-1, width)
                    for dosage in growth.dosages
                ]
                blocks.extend(np.asarray(seed[chunk], dtype=np.int8) for seed in growth.seed_rows)
                # Row keys straight from the contiguous int8 buffer (no structured dtype per call: see
                # local_selection.fit.unique_rows on numpy's per-dtype retention).
                packed = np.ascontiguousarray(np.concatenate(blocks), dtype=np.int8).tobytes()
                rows.update(packed[offset : offset + width] for offset in range(0, len(packed), width))
            for key in rows - own:
                offset = bisect_left(parent_rows[parent_index], key) * width
                full = base[:offset] + key + base[offset:]
                if full not in seen_keys:
                    novel.add(full)
        return len(novel)

    omissions: list[dict] = []
    collected, cut = collect_birth_starts(
        order, groups, seeds_of, budget, hard_cap, omissions, key_of=itemgetter(0), count_rest=count_rest
    )
    children = []
    child_keys: set[bytes] = set()
    fitted = []
    for descriptors in collected:
        group = [
            np.vstack((parents[parent_index].haplotypes, np.frombuffer(key, dtype=np.int8).astype(np.int64)[None, :]))
            for _, parent_index, key in descriptors
        ]
        for family in cast(list[tuple[FitResult, ...]], fit_families(workspace, group, max_iterations)):
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
    _note_omission(
        omissions,
        "data_birth_ranked_truncation",
        sum(len(group) for group in collected) - len(fitted),
        "seed classes below complete-child target",
        kind="ranked_truncation",
    )
    if counter is not None and classes is not None:
        counter.note(
            "discovery.data_birth", inside_tie=tie_cut, wide=classes.wide_classes, boundary=len(classes.boundary_gaps)
        )
    if children:
        return DataBirths(tuple(children), tie_cut, sum(item["count"] for item in omissions), tuple(omissions))
    fallbacks = [canonical_solution(fit, workspace) for fit in cast(list[FitResult], fit_starts(workspace, fitted, 0))]
    ordered = list(deduplicate((child for child in fallbacks if has_distinct_rows(child)), reverse=workspace.reverse))
    # The fallback reuses ``classes`` for its int64 class indices.
    classes = tie_classes(  # type: ignore[assignment]
        [child.nll for child in ordered],
        fixed_policy("discovery.data_birth", workspace.quanta.nll, reverse=workspace.reverse).quantum(),
        reverse=workspace.reverse,
    ).classes
    kept, cut = whole_class_cut(
        ordered,
        classes.tolist(),  # type: ignore[union-attr]
        parent_k + 1,
        omissions,
        cause="data_birth_fallback",
        scope="assignment-only fallback NLL classes",
    )
    return DataBirths(tuple(kept), tie_cut, sum(item["count"] for item in omissions), tuple(omissions))


# ---------------------------------------------------------------------------------------------
# Soft births and birth-delete replacements
# ---------------------------------------------------------------------------------------------
def _hardened_rows(soft, likelihoods, counter=None, *, reverse: bool = False, keep: tuple[str, ...] = ()):
    """Binary rows of soft candidates (> 1/2 is ALT; within HALF_BAND of 1/2 the population ALT
    fraction decides). Both population-half hardenings enter as distinct proposal starts."""
    note_band(counter, "discovery.soft.half_band", soft, 0.5, HALF_BAND)
    fallbacks = hardening_fallbacks(likelihoods, reverse=reverse, keep=keep)
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

    children: tuple[FixedKSolution, ...] = record_field("state", dtype="tuple[FixedKSolution, ...]")
    emitted: int = record_field("state", dtype="int64")
    omitted: int = record_field("state", dtype="int64")
    omissions: tuple = record_field("state", dtype="tuple", default=())


def soft_births(
    workspace: FixedKWorkspace,
    solution: FixedKSolution,
    likelihoods: np.ndarray,
    residual_inputs: ResidualInputs,
    config: DiscoveryParameters,
    cache: dict[tuple[bytes, bytes], SoftBirths],
    counter: TieCounter | None = None,
) -> SoftBirths:
    """Soft-residual births of a panel (memoised per (panel, assignments) in ``cache``; ``counter``: optional
    TieCounter of the band decisions)."""
    key = (solution.key, solution.assignments.tobytes())
    cached = cache.get(key)
    if cached is not None:
        return cached
    omissions: list[dict] = []
    soft = soft_candidates(
        solution.haplotypes,
        residual_inputs,
        config.residual_births,
        counter,
        limit=config.search.max_proposals_per_expansion,
        omissions=omissions,
    )
    if len(soft) == 0:
        result = SoftBirths((), 0, 0, tuple(omissions))
    else:
        existing = {np.asarray(row, dtype=np.int8).tobytes() for row in solution.haplotypes}
        starts = [
            np.vstack((solution.haplotypes, row[None, :]))
            for row in _hardened_rows(soft, likelihoods, counter, reverse=workspace.reverse, keep=residual_inputs.keep)
            if np.asarray(row, dtype=np.int8).tobytes() not in existing
        ]
        synchronized, refitted = cast(
            tuple[tuple[FixedKSolution, ...], tuple[FixedKSolution, ...]],
            fit_panels(workspace, starts, config.model.max_iterations),
        )
        children = deduplicate((*synchronized, *refitted), reverse=workspace.reverse)
        result = SoftBirths(
            tuple(child for child in children if child.k == solution.k + 1), len(starts), 0, tuple(omissions)
        )
    cache[key] = result
    return result


def replacements(
    workspace: FixedKWorkspace,
    parent: FixedKSolution,
    births: Iterable[FixedKSolution],
    config: DiscoveryParameters,
    limit: int | None = None,
    omissions: list[dict] | None = None,
) -> tuple[tuple[FixedKSolution, ...], int]:
    """Same-K starts: every K + 1 birth with any one row deleted, distinct by the remaining rows, taken by whole
    NLL tie classes of the births (best first) while at most ``limit`` (default ``max_replacement_children``)
    starts are taken. Any straddling class is admitted through 2 x limit starts. A larger class is omitted
    whole, and lower-ranked classes are counted separately.

    ``parent``: the K-founder panel; ``births``: its fitted K + 1 children (others are ignored); ``config``:
    DiscoveryParameters. The starts are refitted with ``workspace`` (fit_panels), which updates the workspace's fit
    cache and its fit-schedule, evaluation and sweep counters. Returns (the distinct fitted K-founder children, the
    number of omitted starts). Appends whole-class omissions to ``omissions``; mutates nothing else.
    """
    width = parent.sites
    limit = config.search.max_replacement_children if limit is None else int(limit)
    pool = [birth for birth in deduplicate(births, reverse=workspace.reverse) if birth.k == parent.k + 1]
    if not pool:
        return (), 0
    classes = tie_classes(
        [birth.nll for birth in pool],
        fixed_policy("discovery.replacement", _nll_quantum(pool, None), reverse=workspace.reverse).quantum(),
        content=[birth.key for birth in pool],
        reverse=workspace.reverse,
    )
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
                key = birth.key[:offset] + birth.key[offset + width :]
                if key in seen or key in unit_keys:
                    continue
                unit_keys.add(key)
                unit.append((birth, deleted))
        descriptors.extend(unit)
        descriptor_classes.extend([cls] * len(unit))
        seen |= unit_keys
    retained, _ = whole_class_cut(
        descriptors, descriptor_classes, limit, omissions, cause="replacement", scope=f"K={parent.k} birth NLL classes"
    )
    omitted = len(descriptors) - len(retained)
    descriptors = retained
    starts = [np.ascontiguousarray(np.delete(birth.haplotypes, deleted, axis=0)) for birth, deleted in descriptors]
    synchronized, refitted = cast(
        tuple[tuple[FixedKSolution, ...], tuple[FixedKSolution, ...]],
        fit_panels(workspace, starts, config.model.max_iterations),
    )
    children = deduplicate((*synchronized, *refitted), reverse=workspace.reverse)
    return tuple(child for child in children if child.k == parent.k), omitted


@dataclass(frozen=True)
class OrdinaryMoves:
    """One ordinary expansion: every fitted neighbour (deaths, gauge rewires, data births),
    the K + 1 births among them, starts omitted by the generator size, and limitations."""

    children: tuple[FixedKSolution, ...] = record_field("state", dtype="tuple[FixedKSolution, ...]")
    births: tuple[FixedKSolution, ...] = record_field("state", dtype="tuple[FixedKSolution, ...]")
    omitted: int = record_field("state", dtype="int64")
    limits: tuple[str, ...] = record_field("state", dtype="tuple[str, ...]")
    omissions: tuple = record_field("state", dtype="tuple", default=())


def ordinary_moves(
    workspace: FixedKWorkspace,
    solution: FixedKSolution,
    likelihoods: np.ndarray,
    growth: GrowthInputs,
    ceiling: int,
    cuts: MaximumCuts,
    config: DiscoveryParameters,
    counter: TieCounter | None = None,
) -> OrdinaryMoves:
    """Whole death and gauge proposal families, in that search-priority order, then births.

    Admit whole families through twice max_proposals_per_expansion; never choose a byte prefix.
    """
    search = config.search
    buckets: dict[int, list[np.ndarray]] = {-1: [], 0: []}
    limits: list[str] = []
    omissions: list[dict] = []
    if solution.k > 1:
        buckets[-1].extend(death_starts(solution))
        weights = assignment_graph(solution.assignments, solution.k)
        # Retain the canonical prefix of the complete found cut family and report any truncation.
        cap = 4 * search.max_cut_ties
        partitions = gauge_cut_family(cuts, weights, solution.haplotypes, 2 * cap, omissions)
        buckets[0].extend(gauge_starts(solution, likelihoods, partitions, workspace.quanta.nll))
    unique = {}
    for offset, starts in buckets.items():
        by_key: dict[tuple[tuple[int, ...], bytes], np.ndarray] = {}
        for start in starts:
            value = np.ascontiguousarray(start, dtype=np.int64)
            by_key.setdefault((value.shape, value.tobytes()), value)
        unique[offset] = [
            by_key[key] for key in sorted(by_key, reverse=rev("d9p", reverse=workspace.reverse, keep=cuts.keep))
        ]
    total = sum(len(starts) for starts in unique.values())
    ordered = [start for offset in (-1, 0) for start in unique[offset]]
    classes = [offset for offset in (-1, 0) for start in unique[offset]]
    retained, _ = whole_class_cut(
        ordered,
        classes,
        search.max_proposals_per_expansion,
        omissions,
        cause="proposal",
        scope="death/gauge proposal families",
    )
    omitted = total - len(retained)
    synchronized, refitted = cast(
        tuple[tuple[FixedKSolution, ...], tuple[FixedKSolution, ...]],
        fit_panels(workspace, retained, config.model.max_iterations),
    )
    children: list[FixedKSolution] | tuple[FixedKSolution, ...] = list(
        deduplicate((*synchronized, *refitted), reverse=workspace.reverse)
    )
    births: DataBirths | tuple[FixedKSolution, ...]
    if solution.k < ceiling:
        births = data_births(workspace, solution, growth, config.model.max_iterations, search, counter=counter)
        omissions.extend(births.omissions)
        cast(list[FixedKSolution], children).extend(child for child in births.children if child.k == solution.k + 1)
    children = deduplicate(children, reverse=workspace.reverse)
    births = tuple(child for child in children if child.k == solution.k + 1)
    limits.extend(item["cause"] for item in omissions if item.get("kind") != "ranked_truncation")
    return OrdinaryMoves(children, births, omitted, tuple(dict.fromkeys(limits)), tuple(omissions))
