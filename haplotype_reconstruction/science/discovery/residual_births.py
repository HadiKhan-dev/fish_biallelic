"""Soft-residual birth candidates of a complete panel (the search's residual phase).

Owns: the soft-residual records and their dominant-partner routes, the level-wise HDBSCAN* used for residual and
K = 1 grouping, residual grouping with its edge-clique cover, the soft consensus with its emission order, and the
population-half hardening rules shared with moves.py.
Must not: fit or score panels, read storage, the environment (the clique statistics are an explicit
BlockDiagnostics option) or a clock, or log.
Method: docs/methods/discovery.md, units di11 and di17-di19.
Identity: global groups first, then the sorted partner routes, merged in source order; emission order as below;
``max_cluster_hamming`` bounds group diameter; ``max_alternative_groups`` and size-class budgets bound admission.

For a fitted complete panel, each sample's diplotype posterior over unordered row pairs (a softmax of summed
log genotype likelihoods, uniform pair prior) gives each row an expected copy count. Every row with at least
``min_soft_responsibility`` expected copies in a sample is subtracted from that sample: the residual strand's
ALT probability is P(GL | subtracted allele + residual allele), normalized over the residual allele. Residual
strands are called at ``residual_call_probability`` and clustered by missing-aware Hamming distance, globally
and within each (subtracted row, dominant partner) route. Groups include level-wise HDBSCAN clusters and an
edge clique cover of noise, wider clusters and tied stragglers of multi-way merges; every group's diameter is
at most ``max_cluster_hamming``. Each consensus pools responsibility-weighted log odds with one capped vote
per sample. Emission requires distinct and effective sample support, sufficient call coverage and no duplicate
of the panel or an earlier emission. Emission order is decreasing distinct-sample support, then the hardened
0/1 row (> 1/2 is ALT, with the population ALT fraction deciding within the ``probability`` band), then source
order. This prefers the better-supported row when near-duplicates compete. Every record enters all
dominant-partner routes within the relative ``probability`` band of its largest partner mass. It remains one
global observation with its weight counted once within each route. Candidates are soft ALT probabilities; the
search hardens them into birth starts and its cavity score decides.
"""

from __future__ import annotations

import math
from hashlib import blake2b
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numba import types

from haplotype_reconstruction.core.kernels import kernel
from haplotype_reconstruction.core.numerics import expansion_sum as _accurate_sum
from haplotype_reconstruction.core.ties import conv_key, rev, tie_quantum

from ._cliques import lift_groups, screened_components
from .diagnostics import BlockDiagnostics
from .parameters import LOG_ODDS_MAX, LOG_ODDS_MIN, RESIDUAL_GEOMETRY_SLACK, ResidualBirthParameters
from .parameters import PREDICTIVE_CHUNK_BYTES as _PREDICTIVE_CHUNK_BYTES

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from haplotype_reconstruction.core.ties import TieCounter

__all__ = [
    "ResidualInputs",
    "soft_candidates",
    "sample_sum",
    "scalar_log_floored",
    "alt_fraction_rule",
    "alt_fraction_families",
    "half_band_rows",
    "hardening_fallback",
    "hardening_fallbacks",
    "note_band",
    "exact_hdbscan_labels",
    "level_hdbscan",
    "level_hierarchy",
    "level_hdbscan_labels",
]

# Supported lazy overloads for the declared kernel routes.
_FLOAT64_1C = types.Array(types.float64, 1, "C", readonly=False)
_FLOAT64_2C = types.Array(types.float64, 2, "C", readonly=False)
_INT64_1C = types.Array(types.int64, 1, "C", readonly=False)
_FLOAT64_2F = types.Array(types.float64, 2, "F", readonly=False)
_FLOAT64_3C = types.Array(types.float64, 3, "C", readonly=False)
_INT64_2C_RO = types.Array(types.int64, 2, "C", readonly=True)
_FLOAT64_3F = types.Array(types.float64, 3, "F", readonly=False)
_INT8_2C = types.Array(types.int8, 2, "C", readonly=False)
_BOOLEAN_1C = types.Array(types.boolean, 1, "C", readonly=False)
_BOOLEAN_2C = types.Array(types.boolean, 2, "C", readonly=False)


# Probability-kind bands govern the dominant-partner choice (relative to max(1, mass)), record weights and
# effective support. Values within the band of a threshold count as exactly at that threshold, which passes,
# including symmetric masses at 1/2 and effective support at 1.5.
PARTNER_BAND = tie_quantum(0.0, "probability")
MASS_BAND = tie_quantum(0.0, "probability")
# Values within this band of 1/2 count as 1/2. A consensus in the band uses the population ALT fraction; a
# population fraction also in the band takes the exact-equality decision: REF for "> 1/2", ALT for ">= 1/2".
HALF_BAND = tie_quantum(0.0, "probability")


def note_band(counter: TieCounter | None, kind: str, values: ArrayLike, threshold: float, band: float) -> None:
    """Count one batch of threshold decisions for the stage records (counter: core.ties.TieCounter or None):
    decisions = the values compared, inside_tie = those strictly inside the band (0 < |value - threshold| <= band:
    where the band, not the exact comparison, decided)."""
    if counter is None:
        return
    values = np.asarray(values, dtype=np.float64)
    distance = np.abs(values - threshold)
    counter.add_counts(
        kind, decisions=int(values.size), inside_tie=int(np.count_nonzero((distance > 0.0) & (distance <= band)))
    )


def half_band_rows(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Binary rows of probabilities: > 1/2 is 1, < 1/2 is 0; within HALF_BAND of 1/2 the
    ``fallback`` row (broadcast) decides."""
    values = np.asarray(values, dtype=np.float64)
    rows = values > 0.5
    tied = np.abs(values - 0.5) <= HALF_BAND
    rows[tied] = np.broadcast_to(fallback, rows.shape)[tied]
    return rows


def alt_fraction_rule(
    population_alt: ArrayLike, at_half: bool, *, reverse: bool = False, keep: tuple[str, ...] = ()
) -> np.ndarray:
    """population ALT fraction > 1/2 (outside the band); within HALF_BAND of 1/2: ``at_half`` (a binary default
    state: the opposite under reversed conventions)."""
    population_alt = np.asarray(population_alt, dtype=np.float64)
    rows = population_alt > 0.5
    rows[np.abs(population_alt - 0.5) <= HALF_BAND] = (
        (not at_half) if rev("d8s", reverse=reverse, keep=keep) else bool(at_half)
    )
    return rows


def alt_fraction_families(
    population_alt: ArrayLike, at_half: bool, *, reverse: bool = False, keep: tuple[str, ...] = ()
) -> tuple[np.ndarray, ...]:
    """Both coherent hardenings of population-half ties, or one untied row.

    The convention only orders this complete two-member proposal family; tied
    sites are not independently enumerated. Callers deduplicate resulting starts.
    """
    first = alt_fraction_rule(population_alt, at_half, reverse=reverse, keep=keep)
    tied = np.abs(np.asarray(population_alt, dtype=np.float64) - 0.5) <= HALF_BAND
    if not np.any(tied):
        return (first,)
    second = first.copy()
    second[tied] = ~second[tied]
    return first, second


@kernel(overloads=((_FLOAT64_2C,),), cache=True, nogil=True)
def _sample_sum_2d(values):
    result = np.empty(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        result[column] = _accurate_sum(values[:, column])
    return result


def sample_sum(values: np.ndarray) -> np.ndarray:
    """Accurately sum the sample axis, matching math.fsum of each column."""
    values = np.asarray(values, dtype=np.float64)
    return _sample_sum_2d(values.reshape((values.shape[0], -1))).reshape(values.shape[1:])


def hardening_fallback(likelihoods: np.ndarray, *, reverse: bool = False, keep: tuple[str, ...] = ()) -> np.ndarray:
    """ALT/REF row hardening uses the population ALT fraction when within HALF_BAND of 1/2; a population fraction
    within HALF_BAND of 1/2 chooses ALT under the >= 1/2 rule."""
    marginal = sample_sum(0.5 * likelihoods[:, :, 1] + likelihoods[:, :, 2]) / len(likelihoods)
    return alt_fraction_rule(marginal, True, reverse=reverse, keep=keep)


def hardening_fallbacks(
    likelihoods: np.ndarray, *, reverse: bool = False, keep: tuple[str, ...] = ()
) -> tuple[np.ndarray, ...]:
    """Complete population-half fallback family for soft proposal starts."""
    marginal = sample_sum(0.5 * likelihoods[:, :, 1] + likelihoods[:, :, 2]) / len(likelihoods)
    return alt_fraction_families(marginal, True, reverse=reverse, keep=keep)


# Residual kernels use scalar math.log / math.exp, element-wise arithmetic and fixed-order sums. NumPy can
# select CPU-dependent SIMD ufunc paths with different rounding; scalar evaluation keeps residual candidates
# independent of those paths. Discovery makes no BLAS calls.
@kernel(
    overloads=(
        (
            _FLOAT64_1C,
            types.float64,
        ),
        (
            _FLOAT64_2C,
            types.float64,
        ),
        (
            _FLOAT64_3C,
            types.float64,
        ),
    ),
    cache=True,
    nogil=True,
)
def scalar_log_floored(values, floor):
    """float64 array of log(max(value, floor)) for a C-contiguous float64 array (C library scalar log)."""
    flat = values.reshape(-1)
    out = np.empty(flat.size, dtype=np.float64)
    for index in range(flat.size):
        value = flat[index]
        out[index] = math.log(value if value > floor else floor)
    return out.reshape(values.shape)


@kernel(overloads=((_FLOAT64_2F,),), cache=True, nogil=True)
def _exp_inplace(values):
    """values[i, j] = exp(values[i, j]) for a 2-D float64 array of any layout (C library scalar exp)."""
    rows, columns = values.shape
    for row in range(rows):
        for column in range(columns):
            values[row, column] = math.exp(values[row, column])


@kernel(
    overloads=(
        (
            _FLOAT64_1C,
            _BOOLEAN_1C,
        ),
    ),
    cache=True,
    nogil=True,
)
def _logistic_where(total, observed):
    """ALT probability 1 / (1 + exp(-total)) where observed (0.5 elsewhere), evaluated on the stable side of 0 with
    the C library scalar exp."""
    out = np.full(total.shape[0], 0.5, dtype=np.float64)
    for site in range(total.shape[0]):
        if observed[site]:
            value = total[site]
            if value >= 0.0:
                out[site] = 1.0 / (1.0 + math.exp(-value))
            else:
                exponential = math.exp(value)
                out[site] = exponential / (1.0 + exponential)
    return out


def exact_hdbscan_labels(
    distance: np.ndarray, min_cluster_size: int, min_samples: int, allow_single_cluster: bool
) -> np.ndarray:
    """int64 HDBSCAN labels (-1 = noise) of a precomputed float64 distance matrix that depend neither on the order of
    the records nor on the order of equal-weight links: :func:`level_hdbscan`."""
    return level_hdbscan(distance, min_cluster_size, min_samples, allow_single_cluster)[0]


def level_hdbscan(
    distance: np.ndarray, min_cluster_size: int, min_samples: int, allow_single_cluster: bool
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """(labels int64[n], stragglers bool[n], tree_groups) of HDBSCAN* on a precomputed float64 distance matrix.

    Mutual reachability uses the distance to the min_samples-th nearest record and a minimum spanning tree
    (precomputed metric, alpha 1). The hierarchy merges all links of one weight simultaneously, as in HDBSCAN*
    (Campello, Moulavi and Sander 2013; Campello et al. 2015). Thus each level is the connected components of
    links up to that weight, independent of binary merge order (:func:`level_hdbscan_labels`). Stragglers are
    points whose grouping a binary merge order would decide; :func:`_cluster` offers them as alternative
    groups.
    """
    from hdbscan._hdbscan_linkage import mst_linkage_core

    distance = np.ascontiguousarray(distance, dtype=np.float64)
    n = distance.shape[0]
    if n < 2:
        return np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.bool_), []
    neighbours = max(1, min(n - 1, int(min_samples)))
    if neighbours == 1:
        # Symmetric nonnegative distances, zero diagonal: each endpoint's nearest
        # other distance is <= their mutual distance. Mutual reachability therefore
        # leaves every off-diagonal entry unchanged. Prim never reads the diagonal
        # or writes its input (hdbscan._hdbscan_linkage.mst_linkage_core).
        reachability = distance
    else:
        # Copy just the core vector so the partition's dense backing array dies
        # before allocating reachability. These are the same alpha=1 operations
        # as HDBSCAN, with memory bounded independently of its implementation.
        core = np.partition(distance, neighbours, axis=0)[neighbours].copy()
        reachability = np.array(distance, copy=True)
        np.maximum(reachability, core, out=reachability)
        np.maximum(reachability, core[:, None], out=reachability)
    tree = mst_linkage_core(reachability)
    return level_hdbscan_labels(
        n,
        tree[:, 0].astype(np.int64),
        tree[:, 1].astype(np.int64),
        np.asarray(tree[:, 2], dtype=np.float64),
        int(min_cluster_size),
        bool(allow_single_cluster),
    )


def level_hierarchy(
    n: int, first: np.ndarray, second: np.ndarray, weight: np.ndarray
) -> tuple[list[list[int]], list[float], list[list[int]]]:
    """The non-binary single-linkage hierarchy of a spanning tree on n points (edge i joins first[i] and second[i] at
    weight[i]): all edges of one weight are merged at once, so each node is a connected component of the edges of at
    most its weight, independent of the edge order. Returns (children, level, members) of the merge nodes; node n + i
    is the i-th (children: its component ids, a point index or a node id; level: its weight; members: its points);
    the last node is the root."""
    parent = list(range(n))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    component = list(range(n))
    children: list[list[int]]
    level: list[float]
    members: list[list[int]]
    children, level, members = [], [], []
    weight = np.asarray(weight, dtype=np.float64)
    order = np.argsort(weight, kind="stable")
    start = 0
    while start < len(order):
        value = weight[order[start]]
        stop = start
        while stop < len(order) and weight[order[stop]] == value:
            stop += 1
        edges = [(int(first[index]), int(second[index])) for index in order[start:stop]]
        previous = {}
        for a, b in edges:
            previous[a] = component[find(a)]
            previous[b] = component[find(b)]
        for a, b in edges:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a
        groups: dict[int, set[int]] = {}
        for point, old in previous.items():
            groups.setdefault(find(point), set()).add(old)
        # ``olds`` is reused for the merged component set and then its sorted list.
        olds: Any
        for root, olds in groups.items():
            node = n + len(children)
            olds = sorted(olds)
            children.append(olds)
            level.append(float(value))
            members.append([p for old in olds for p in (members[old - n] if old >= n else (old,))])
            component[root] = node
        start = stop
    return children, level, members


def level_hdbscan_labels(
    n: int,
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
    min_cluster_size: int,
    allow_single_cluster: bool,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """HDBSCAN labels on the non-binary hierarchy of a spanning tree (:func:`level_hierarchy`).

    Condensed tree, top down: a cluster that splits at lambda = 1/weight (inf at weight 0) into components makes
    those with at least min_cluster_size points child clusters (born at lambda) when at least two qualify; otherwise
    it persists as its qualifying component, if any, and the points of the other components fall out at lambda. With
    two components this is the hdbscan package's binary rule. Stability of a cluster: sum over its rows of
    (lambda - lambda_birth) x size (points and child clusters; the root is born at 0), summed exactly (math.fsum), so
    no order enters. Excess of mass bottom up as in the package: a cluster whose children's (updated) stabilities sum
    to more than its own is replaced by them; otherwise its descendants are dropped; the root takes part only with
    allow_single_cluster, and then labels only its points that fall out at its largest lambda. A point takes the
    selected cluster nearest above where it falls out (-1 if none). Clusters are numbered by their smallest member.
    Returns (labels int64[n], stragglers bool[n], tree_groups): tree_groups are the member sets (sorted tuples, at
    least two points) of every cluster of the condensed tree, selected or not; stragglers are the points of the small
    components of a node where
    at least three components merge at one weight, at least one of them big and at least two small, i.e. the points
    whose grouping a binary merge order would decide (pair-ups of tied small components).
    """
    children, level, members = level_hierarchy(n, first, second, weight)
    stragglers = np.zeros(n, dtype=np.bool_)
    if not children:
        return np.full(n, -1, dtype=np.int64), stragglers, []

    def size(component):
        return 1 if component < n else len(members[component - n])

    def points(component):
        return (component,) if component < n else members[component - n]

    terms: list[list[float]]
    child_clusters: list[list[int]]
    cluster_parent, birth, terms, child_clusters = [-1], [0.0], [[]], [[]]
    # (cluster, lambda) where each point falls out; every point falls out once before it is read.
    fall: list[Any] = [None] * n
    root_lambdas = []
    stack = [(n + len(children) - 1, 0)]
    while stack:
        node, cluster = stack.pop()
        value = level[node - n]
        lam = math.inf if value <= 0.0 else 1.0 / value
        components = children[node - n]
        big = [c for c in components if size(c) >= min_cluster_size]
        small = [c for c in components if size(c) < min_cluster_size]
        if len(big) >= 2:
            for component in big:
                child = len(birth)
                cluster_parent.append(cluster)
                birth.append(lam)
                terms.append([])
                child_clusters.append([])
                child_clusters[cluster].append(child)
                terms[cluster].append(0.0 if lam == birth[cluster] else (lam - birth[cluster]) * size(component))
                if cluster == 0:
                    root_lambdas.append(lam)
                stack.append((component, child))
        elif len(big) == 1:
            stack.append((big[0], cluster))
        else:
            small = components
        if big and len(components) >= 3 and len(small) >= 2:
            # The cluster continues through a big component while several small ones fall out at one lambda: a
            # binary merge order could pair those up into clusters of their own (zero-lifetime leaves that the
            # excess of mass can select) -- the tie-dependent resolutions. Where no component is big, such pairs
            # never beat their parent, so nothing depends on the order.
            for component in small:
                for point in points(component):
                    stragglers[point] = True
        for component in small:
            for point in points(component):
                fall[point] = (cluster, lam)
                terms[cluster].append(0.0 if lam == birth[cluster] else lam - birth[cluster])
                if cluster == 0:
                    root_lambdas.append(lam)
    n_clusters = len(birth)
    stability = [math.fsum(values) for values in terms]
    candidates = list(range(n_clusters - 1, -1 if allow_single_cluster else 0, -1))
    selected = [False] * n_clusters
    for cluster in candidates:
        selected[cluster] = True
    # ``subtree`` holds a stability sum here and cluster-member lists below.
    subtree: Any
    for cluster in candidates:
        subtree = math.fsum(stability[child] for child in child_clusters[cluster])
        if subtree > stability[cluster]:
            selected[cluster] = False
            stability[cluster] = subtree
        else:
            below = list(child_clusters[cluster])
            while below:
                descendant = below.pop()
                selected[descendant] = False
                below.extend(child_clusters[descendant])
    root_cutoff = max(root_lambdas) if root_lambdas else math.inf
    assigned = [-1] * n
    for point in range(n):
        cluster, lam = fall[point]
        while cluster != -1 and not selected[cluster]:
            cluster = cluster_parent[cluster]
        if cluster > 0 or (cluster == 0 and lam >= root_cutoff):
            assigned[point] = cluster
    first_member = {}
    for point in range(n):
        if assigned[point] != -1 and assigned[point] not in first_member:
            first_member[assigned[point]] = point
    # dict.get is total here (keys of the same dict); its stub admits None.
    rank = {cluster: index for index, cluster in enumerate(sorted(first_member, key=first_member.get))}  # type: ignore[arg-type]
    subtree = [[] for _ in range(n_clusters)]
    for point in range(n):
        cluster = fall[point][0]
        while cluster != -1:
            subtree[cluster].append(point)
            cluster = cluster_parent[cluster]
    tree_groups = sorted(tuple(members) for members in subtree if len(members) >= 2)
    return np.array([rank[c] if c != -1 else -1 for c in assigned], dtype=np.int64), stragglers, tree_groups


# Residual strand hard-call sentinel.
_UNCALLED = -1
# Residual log odds use log(ALT mass) - log(REF mass), clipped to [log tiny, log((1 - eps) / eps)]. Computing
# from the two masses avoids the cancellation in 1 - p when p is near one, which can move near-cancelling
# consensus sites.
_LOG_ODDS_MAX = LOG_ODDS_MAX
_LOG_ODDS_MIN = LOG_ODDS_MIN
# Bound of the largest temporary of the diplotype responsibility (bytes).


class ResidualInputs:
    """Evidence-only inputs of one block: likelihood (unobserved cells exactly 1/3), its floored
    logarithm, the observation mask, and an exact cache of residual clusterings (identical
    distance matrices recur across proposals of one block search)."""

    def __init__(
        self,
        likelihood: np.ndarray,
        observed: np.ndarray,
        *,
        reverse: bool = False,
        keep: tuple[str, ...] = (),
        diagnostics: BlockDiagnostics | None = None,
    ) -> None:
        self.reverse = reverse
        self.keep = tuple(keep)
        self.diagnostics = diagnostics
        self.likelihood = np.ascontiguousarray(likelihood, dtype=np.float64)
        self.log_likelihood = scalar_log_floored(self.likelihood, np.finfo(np.float64).tiny)
        self.observed = np.ascontiguousarray(observed, dtype=np.bool_)
        self.cluster_cache: dict = {}
        self.landmark_cache: dict = {}


@kernel(
    overloads=(
        (
            _FLOAT64_3C,
            _INT64_2C_RO,
            _INT64_1C,
            _INT64_1C,
            _FLOAT64_3F,
        ),
    ),
    cache=True,
    nogil=True,
)
def _gather_binary_predictive(likelihood, haplotypes, pair_i, pair_j, predictive):
    """One exact log likelihood per hard diplotype and site, into an F-ordered output."""
    n_samples = likelihood.shape[0]
    n_pairs = pair_i.shape[0]
    n_sites = haplotypes.shape[1]
    for site_index in range(n_sites):
        for pair_index in range(n_pairs):
            genotype = haplotypes[pair_i[pair_index], site_index] + haplotypes[pair_j[pair_index], site_index]
            for sample_index in range(n_samples):
                predictive[sample_index, pair_index, site_index] = likelihood[sample_index, site_index, genotype]


def _diplotype_responsibility(log_likelihood, haplotypes, pair_i, pair_j):
    """float64[N, pairs] posterior of each sample over the unordered row pairs (uniform prior).

    The per-pair log emission is a NumPy sum over sites of an F-ordered (samples, pairs, sites)
    array (fixed summation order); samples are chunked to bound that array, pairs never are
    (chunking pairs would change the array's site stride and hence the summation order).
    """
    n_samples = log_likelihood.shape[0]
    n_pairs = len(pair_i)
    if n_pairs == 1:
        return np.ones((n_samples, 1), dtype=np.float64)
    n_sites = haplotypes.shape[1]
    bytes_per_sample = max(1, n_pairs * n_sites * 8)
    samples_per_chunk = max(1, min(n_samples, _PREDICTIVE_CHUNK_BYTES // bytes_per_sample))
    log_emission = np.empty((n_samples, n_pairs), dtype=np.float64, order="F" if n_samples > 1 else "C")
    for start in range(0, n_samples, samples_per_chunk):
        stop = min(start + samples_per_chunk, n_samples)
        log_predictive = np.empty((stop - start, n_pairs, n_sites), dtype=np.float64, order="F")
        _gather_binary_predictive(log_likelihood[start:stop], haplotypes, pair_i, pair_j, log_predictive)
        log_emission[start:stop] = np.sum(log_predictive, axis=2)
    row_maximum = np.max(log_emission, axis=1, keepdims=True)
    responsibility = log_emission - row_maximum
    _exp_inplace(responsibility)
    responsibility /= np.sum(responsibility, axis=1, keepdims=True)
    return np.ascontiguousarray(responsibility)


@kernel(
    overloads=(
        (
            _FLOAT64_2C,
            _INT64_1C,
            _INT64_1C,
            types.int64,
            _FLOAT64_3C,
            _BOOLEAN_2C,
            _INT64_2C_RO,
            types.float64,
            types.float64,
            types.float64,
            types.float64,
            types.boolean,
        ),
    ),
    cache=True,
    nogil=True,
)
def _residual_records(
    responsibility,
    pair_i,
    pair_j,
    n_rows,
    likelihood,
    observed,
    haplotypes,
    call_probability,
    minimum_responsibility,
    partner_band,
    mass_band,
    reverse,
):
    """Residual strands of every (sample, subtracted row) with enough expected copies.

    Returns sample, subtracted row, reporting-only partner label, weight min(1, expected copies), residual ALT
    probabilities (0.5 where unobserved), hard calls (-1 uncalled), the compatible mask (observed sites with
    positive residual mass), residual log odds log(r1) - log(r0) within [_LOG_ODDS_MIN, _LOG_ODDS_MAX] (zero
    elsewhere), and the full dominant-partner membership mask. A record needs weight + mass_band >=
    minimum_responsibility. Every partner within partner_band * max(1, global maximum mass) receives the same
    record. The reverse argument cannot change this class; its lower member is a reporting label, never a
    routing decision.
    """
    n_samples, n_states = responsibility.shape
    n_sites = haplotypes.shape[1]
    maximum_records = n_samples * n_rows
    sample_indices = np.empty(maximum_records, dtype=np.int64)
    subtracted = np.empty(maximum_records, dtype=np.int64)
    partners = np.empty(maximum_records, dtype=np.int64)
    partner_membership = np.zeros((maximum_records, n_rows), dtype=np.uint8)
    weights = np.empty(maximum_records, dtype=np.float64)
    copy_mass = np.zeros(n_rows, dtype=np.float64)
    partner_mass = np.zeros((n_rows, n_rows), dtype=np.float64)
    count = 0
    for sample in range(n_samples):
        copy_mass.fill(0.0)
        partner_mass.fill(0.0)
        for state in range(n_states):
            first = pair_i[state]
            second = pair_j[state]
            mass = responsibility[sample, state]
            if first == second:
                copy_mass[first] += 2.0 * mass
                partner_mass[first, first] += 2.0 * mass
            else:
                copy_mass[first] += mass
                copy_mass[second] += mass
                partner_mass[first, second] += mass
                partner_mass[second, first] += mass
        for row in range(n_rows):
            expected = copy_mass[row]
            weight = min(1.0, expected)
            if weight + mass_band < minimum_responsibility:
                continue
            best = np.max(partner_mass[row])
            band = partner_band * max(1.0, best)
            partner = -1
            for other in range(n_rows):
                if best - partner_mass[row, other] <= band:
                    partner_membership[count, other] = 1
                    if partner < 0:
                        partner = other
            sample_indices[count] = sample
            subtracted[count] = row
            partners[count] = partner
            weights[count] = weight
            count += 1
    soft_alt = np.full((count, n_sites), 0.5, dtype=np.float64)
    hard_calls = np.full((count, n_sites), _UNCALLED, dtype=np.int8)
    compatible = np.zeros((count, n_sites), dtype=np.bool_)
    log_odds = np.zeros((count, n_sites), dtype=np.float64)
    for record in range(count):
        sample = sample_indices[record]
        row = subtracted[record]
        for site in range(n_sites):
            q = float(haplotypes[row, site])
            residual_zero = (1.0 - q) * likelihood[sample, site, 0] + q * likelihood[sample, site, 1]
            residual_one = (1.0 - q) * likelihood[sample, site, 1] + q * likelihood[sample, site, 2]
            denominator = residual_zero + residual_one
            if observed[sample, site] and denominator > 0.0:
                value = residual_one / denominator
                soft_alt[record, site] = value
                compatible[record, site] = True
                if residual_zero <= 0.0:
                    odds = _LOG_ODDS_MAX
                elif residual_one <= 0.0:
                    odds = _LOG_ODDS_MIN
                else:
                    odds = math.log(residual_one) - math.log(residual_zero)
                    if odds > _LOG_ODDS_MAX:
                        odds = _LOG_ODDS_MAX
                    elif odds < _LOG_ODDS_MIN:
                        odds = _LOG_ODDS_MIN
                log_odds[record, site] = odds
                if value >= call_probability:
                    hard_calls[record, site] = 1
                if value <= 1.0 - call_probability:
                    hard_calls[record, site] = 0
    return (
        sample_indices[:count],
        subtracted[:count],
        partners[:count],
        weights[:count],
        soft_alt,
        hard_calls,
        compatible,
        log_odds,
        partner_membership[:count],
    )


@kernel(overloads=((types.uint64,),), cache=True, inline="always", nogil=True)
def _popcount(value):
    """64-bit population count (SWAR reduction)."""
    value = value - ((value >> np.uint64(1)) & np.uint64(0x5555555555555555))
    value = (value & np.uint64(0x3333333333333333)) + ((value >> np.uint64(2)) & np.uint64(0x3333333333333333))
    value = (value + (value >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((value * np.uint64(0x0101010101010101)) >> np.uint64(56))


@kernel(
    overloads=(
        (
            _INT8_2C,
            types.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _hamming_distances(hard_calls, minimum_joint):
    """Missing-aware Hamming distance of every record pair: mismatches over jointly called
    sites, 1 when fewer than ``minimum_joint`` sites are called in both (bit-packed)."""
    n_records, n_sites = hard_calls.shape
    n_words = (n_sites + 63) // 64
    known_bits = np.zeros((n_records, n_words), dtype=np.uint64)
    allele_bits = np.zeros((n_records, n_words), dtype=np.uint64)
    for record in range(n_records):
        for site in range(n_sites):
            value = hard_calls[record, site]
            if value != _UNCALLED:
                word = site >> 6
                bit = np.uint64(1) << np.uint64(site & 63)
                known_bits[record, word] |= bit
                if value == 1:
                    allele_bits[record, word] |= bit
    distance = np.zeros((n_records, n_records), dtype=np.float64)
    for first in range(n_records - 1):
        for second in range(first + 1, n_records):
            n_joint = 0
            n_mismatch = 0
            for word in range(n_words):
                joint = known_bits[first, word] & known_bits[second, word]
                mismatch = (allele_bits[first, word] ^ allele_bits[second, word]) & joint
                n_joint += _popcount(joint)
                n_mismatch += _popcount(mismatch)
            value = 1.0 if n_joint < minimum_joint else n_mismatch / n_joint
            distance[first, second] = value
            distance[second, first] = value
    return distance


@kernel(
    overloads=(
        (
            _FLOAT64_2C,
            _INT64_1C,
            types.float64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _diameter_within(distance, members, bound):
    """The dense submatrix maximum test, without allocating that submatrix."""
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if distance[members[i], members[j]] > bound:
                return False
    return True


def _cluster(
    distance,
    maximum_hamming,
    max_groups,
    content_keys,
    omissions=None,
    lifted=None,
    *,
    reverse: bool = False,
    keep: tuple[str, ...] = (),
    diagnostics: BlockDiagnostics | None = None,
):
    """Density groups plus a content-ordered maximal edge-clique cover.

    Every compatible alternative pair shares a group before size screening. A
    larger complete-linkage subgroup need not be contained in any cover member.
    Density groups are lifted with their sample weights. The size-class rule
    applies separately to each alternative graph component.
    With lifted=(membership_bits, record_count), sizes/identities use original
    records. Content keys are the hard-call descriptors on the distance axis.
    """
    n = distance.shape[0]
    if n < 2:
        return (), False
    bound = maximum_hamming + RESIDUAL_GEOMETRY_SLACK
    groups: list[tuple[int, ...]] | tuple[tuple[int, ...], ...]
    if n == 2:
        groups = ((0, 1),) if float(distance[0, 1]) <= bound else ()
        if lifted is not None:
            groups = tuple(lift_groups(groups, lifted[0]))
        return groups, False
    labels, stragglers, tree_groups = level_hdbscan(distance, 2, 1, True)
    # Every cluster of the condensed tree (all density levels, selected or not) within the bound is a group: the
    # excess of mass picks one level per branch, a binary merge order can shift that pick, and the block score can
    # decide instead.
    groups = [
        members for members in tree_groups if _diameter_within(distance, np.asarray(members, dtype=np.int64), bound)
    ]
    pool = set(np.flatnonzero(labels == -1).tolist())
    for label in sorted(set(labels.tolist()) - {-1}):
        members = tuple(np.flatnonzero(labels == label).tolist())
        if len(members) < 2:
            pool.update(members)
        elif _diameter_within(distance, np.asarray(members, dtype=np.int64), bound):
            groups.append(members)
        else:
            pool.update(members)
    groups = sorted(set(groups))
    pool.update(np.flatnonzero(stragglers).tolist())
    if lifted is not None:
        # Core groups do not consume the alternative budget, even when a clique
        # later lifts to one of these same groups.
        groups = lift_groups(groups, lifted[0])
    overflow = False
    if len(pool) >= 2:
        vertices = sorted(pool)
        # The same upper-triangle comparisons, batched before graph traversal.
        # Python scalar indexing dominated construction of large residual pools.
        vertex_array = np.asarray(vertices, dtype=np.int64)
        compatible = np.triu(distance[np.ix_(vertex_array, vertex_array)] <= bound, 1)
        compatible |= compatible.T
        adjacent = {
            vertex: vertex_array[np.flatnonzero(compatible[position])].tolist()
            for position, vertex in enumerate(vertices)
        }
        components, seen = [], set()
        for vertex in vertices:
            if vertex in seen:
                continue
            component, frontier = {vertex}, [vertex]
            while frontier:
                current = frontier.pop()
                for other in adjacent[current]:
                    if other not in component:
                        component.add(other)
                        frontier.append(other)
            seen |= component
            if len(component) >= 2:
                components.append(tuple(sorted(component)))
        kept, overflow, work, clique_count = screened_components(
            components,
            adjacent,
            content_keys,
            int(max_groups),
            memberships=None if lifted is None else lifted[0],
            records=None if lifted is None else lifted[1],
            omissions=omissions,
            reverse=reverse,
            keep=keep,
        )
        if diagnostics is not None:
            diagnostics.record(len(pool), work, overflow, cliques=clique_count)
        known = set(groups)
        cast(list[tuple[int, ...]], groups).extend(clique for clique in kept if clique not in known)
    return tuple(sorted(set(groups))), overflow


def _cluster_cached(
    distance,
    maximum_hamming,
    cache,
    content_keys,
    max_groups=64,
    omissions=None,
    *,
    reverse: bool = False,
    keep: tuple[str, ...] = (),
    diagnostics: BlockDiagnostics | None = None,
):
    # The greedy cover depends on content order as well as geometry.
    value = np.ascontiguousarray(distance)
    content = blake2b(b"".join(content_keys), digest_size=32).digest()
    key = (
        float(maximum_hamming),
        int(max_groups),
        rev("d13", reverse=reverse, keep=keep),
        value.shape,
        content,
        blake2b(memoryview(value).cast("B"), digest_size=32).digest(),
    )
    result = cache.get(key)
    if result is None:
        details: list[dict] = []
        groups, overflow = _cluster(
            distance,
            maximum_hamming,
            max_groups,
            content_keys,
            details,
            reverse=reverse,
            keep=keep,
            diagnostics=diagnostics,
        )
        result = cache[key] = (groups, overflow, tuple(details))
    if omissions is not None:
        omissions.extend(result[2])
    return result[:2]


@kernel(
    overloads=(
        (
            _INT64_1C,
            _INT64_1C,
            _FLOAT64_1C,
            _BOOLEAN_2C,
            _FLOAT64_2C,
        ),
    ),
    cache=True,
    nogil=True,
)
def _pooled_log_odds(members, sample_indices, weights, compatible, log_odds):
    """Per site, the sum over samples of min(1, w_s) x (the sample's responsibility-weighted
    mean log odds over its records), with the number of contributing samples. ``members`` are
    ordered by sample.

    Each site keeps its contributing samples' terms contiguously, and only those enter the
    site's exact sum. A sample without weight at a site adds an exact +0, which leaves the correctly rounded sum
    unchanged (up to the sign of a zero total, which the logistic maps to 1/2 either way). One partials buffer
    serves every site, and no members x sites matrix is allocated."""
    n_sites = log_odds.shape[1]
    total = np.zeros(n_sites, dtype=np.float64)
    contributing = np.zeros(n_sites, dtype=np.int32)
    if members.shape[0] == 0:
        return total, contributing
    n_terms = 1
    for position in range(1, members.shape[0]):
        if sample_indices[members[position]] != sample_indices[members[position - 1]]:
            n_terms += 1
    site_terms = np.empty((n_sites, n_terms), dtype=np.float64)
    weighted = np.zeros(n_sites, dtype=np.float64)
    weight_at_site = np.zeros(n_sites, dtype=np.float64)
    current = sample_indices[members[0]]
    for position in range(members.shape[0]):
        record = members[position]
        sample = sample_indices[record]
        if sample != current:
            for site in range(n_sites):
                weight = weight_at_site[site]
                if weight > 0.0:
                    sample_weight = 1.0 if weight > 1.0 else weight
                    site_terms[site, contributing[site]] = sample_weight * (weighted[site] / weight)
                    contributing[site] += 1
                    weighted[site] = 0.0
                    weight_at_site[site] = 0.0
            current = sample
        record_weight = weights[record]
        for site in range(n_sites):
            if compatible[record, site]:
                weighted[site] += record_weight * log_odds[record, site]
                weight_at_site[site] += record_weight
    partials = np.empty(n_terms, dtype=np.float64)
    for site in range(n_sites):
        weight = weight_at_site[site]
        if weight > 0.0:
            sample_weight = 1.0 if weight > 1.0 else weight
            site_terms[site, contributing[site]] = sample_weight * (weighted[site] / weight)
            contributing[site] += 1
        total[site] = _accurate_sum(site_terms[site, : contributing[site]], partials)
    return total, contributing


@kernel(
    overloads=(
        (
            _INT64_1C,
            _INT64_1C,
            _FLOAT64_1C,
        ),
    ),
    cache=True,
    nogil=True,
)
def _sample_support(ordered, sample_indices, weights):
    """(distinct samples, each sample's capped support min(1, running sum of its records' weights) in member
    order). ``ordered`` is sorted by sample, so each sample is one run (the per-record loop of
    soft_candidates, same arithmetic in the same order)."""
    support = np.empty(ordered.shape[0], dtype=np.float64)
    count = 0
    for position in range(ordered.shape[0]):
        record = ordered[position]
        if position == 0 or sample_indices[record] != sample_indices[ordered[position - 1]]:
            support[count] = min(1.0, 0.0 + weights[record])
            count += 1
        else:
            support[count - 1] = min(1.0, support[count - 1] + weights[record])
    return count, support


def _consensus(members, sample_indices, weights, compatible, log_odds, call_probability):
    """(ALT probability per site, fraction of sites called at ``call_probability``)."""
    total, contributing = _pooled_log_odds(
        np.ascontiguousarray(members, dtype=np.int64), sample_indices, weights, compatible, log_odds
    )
    observed = contributing > 0
    posterior = _logistic_where(total, observed)
    confident = observed & ((posterior >= call_probability) | (posterior <= 1.0 - call_probability))
    return np.ascontiguousarray(posterior), float(np.mean(confident))


@kernel(
    overloads=(
        (
            _FLOAT64_1C,
            _FLOAT64_2C,
            types.int64,
            types.int64,
            types.float64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _closest(candidate, existing, minimum_joint, n_sites, known_probability):
    """Nearest existing row by missing-aware Hamming distance over sites called in both
    (rounded ALT probabilities beyond ``known_probability``); ties within 1e-12 prefer the
    larger joint-called fraction (a stated preference: more shared evidence). Returns (found, distance, joint
    fraction, coverage, coverage), the coverage being the smaller of the candidate's and the row's call coverage,
    maximised over the rows tied at the nearest (distance, joint fraction): no row order decides."""
    lower = 1.0 - known_probability + RESIDUAL_GEOMETRY_SLACK
    upper = known_probability - RESIDUAL_GEOMETRY_SLACK
    candidate_known = np.zeros(n_sites, dtype=np.bool_)
    candidate_calls = np.empty(n_sites, dtype=np.float64)
    n_candidate_known = 0
    for site in range(n_sites):
        value = candidate[site]
        known = value <= lower or value >= upper
        candidate_known[site] = known
        if known:
            candidate_calls[site] = np.rint(value)
            n_candidate_known += 1
    found = False
    closest_distance = 0.0
    closest_joint = 0.0
    closest_candidate_coverage = 0.0
    for other in range(existing.shape[0]):
        n_other_known = 0
        n_joint = 0
        n_mismatch = 0
        for site in range(n_sites):
            other_value = existing[other, site]
            if not (other_value <= lower or other_value >= upper):
                continue
            n_other_known += 1
            if candidate_known[site]:
                n_joint += 1
                if candidate_calls[site] != np.rint(other_value):
                    n_mismatch += 1
        if n_joint < minimum_joint:
            continue
        distance = n_mismatch / n_joint
        joint_fraction = n_joint / n_sites
        coverage = min(n_joint / max(1, n_candidate_known), n_joint / max(1, n_other_known))
        if (
            not found
            or distance < closest_distance - RESIDUAL_GEOMETRY_SLACK
            or (abs(distance - closest_distance) <= RESIDUAL_GEOMETRY_SLACK and joint_fraction > closest_joint)
        ):
            found = True
            closest_distance = distance
            closest_joint = joint_fraction
            closest_candidate_coverage = coverage
        elif abs(distance - closest_distance) <= RESIDUAL_GEOMETRY_SLACK and joint_fraction == closest_joint:
            closest_candidate_coverage = max(closest_candidate_coverage, coverage)
    return (found, closest_distance, closest_joint, closest_candidate_coverage, closest_candidate_coverage)


def _hardened_key(candidate, fallback) -> bytes:
    """The emission-order key of a soft candidate: its hardened 0/1 row as bytes (> 1/2 is ALT;
    within the probability quantum of 1/2 the ``fallback`` row decides, as moves.half_band_rows)."""
    return np.ascontiguousarray(half_band_rows(candidate, fallback), dtype=np.uint8).tobytes()


def _partner_routes(subtracted, membership):
    """Unique original observation indices in each complete winning partner route.

    Multiple routes are alternative proposal pools, not additional observations or
    copies: the global pool and every within-route consensus keep each weight once.
    """
    routes: dict[tuple[int, int], list[int]] = {}
    for index in range(len(subtracted)):
        for partner in np.flatnonzero(membership[index]):
            routes.setdefault((int(subtracted[index]), int(partner)), []).append(index)
    return {route: tuple(indices) for route, indices in routes.items()}


def _residual_pool_groups(
    pool,
    hard_calls,
    distance,
    minimum_joint,
    config,
    cluster_cache,
    landmark_cache,
    *,
    reverse: bool = False,
    keep: tuple[str, ...] = (),
    diagnostics: BlockDiagnostics | None = None,
):
    """Evaluate one pool using caches shared in source order.

    Entries computed for earlier pools are available to later pools. Scheduling
    independent blocks belongs to the stage.
    """
    size = len(hard_calls) if pool is None else len(pool)
    details: list[dict] = []
    if config.landmark_budget and size > config.landmark_budget:
        from .landmarks import residual_groups

        local_calls = hard_calls if pool is None else np.ascontiguousarray(hard_calls[list(pool)])
        groups, overflow, _ = residual_groups(
            local_calls,
            config.landmark_budget,
            minimum_joint,
            config,
            details,
            "global residuals" if pool is None else "partner residual route",
            landmark_cache,
            reverse=reverse,
            keep=keep,
            diagnostics=diagnostics,
        )
    else:
        local_distance = (distance if pool is None else distance[np.ix_(pool, pool)]) if distance is not None else None
        if local_distance is None:
            local_calls = hard_calls if pool is None else np.ascontiguousarray(hard_calls[list(pool)])
            local_distance = _hamming_distances(np.ascontiguousarray(local_calls), minimum_joint)
        local_calls = hard_calls if pool is None else hard_calls[list(pool)]
        content_keys = tuple(np.ascontiguousarray(row, dtype=np.int8).tobytes() for row in local_calls)
        groups, overflow = _cluster_cached(
            local_distance,
            config.max_cluster_hamming,
            cluster_cache,
            content_keys,
            config.max_alternative_groups,
            details,
            reverse=reverse,
            keep=keep,
            diagnostics=diagnostics,
        )
    return groups, overflow, details


def _residual_pool_results(hard_calls, routes, minimum_joint, config, inputs, omissions):
    records, sites = hard_calls.shape
    tasks = [(None, None)] if records >= 2 else []
    tasks.extend((route, routes[route]) for route in sorted(routes) if len(routes[route]) >= 2)
    if not tasks:
        return {}
    distance = None
    if records >= 2 and (not config.landmark_budget or records <= config.landmark_budget):
        distance = _hamming_distances(np.ascontiguousarray(hard_calls), minimum_joint)
    # Pure source-order pool evaluation. The stage schedules independent blocks.
    if tasks:
        results = {}
        for route, pool in tasks:
            groups, overflow, details = _residual_pool_groups(
                pool,
                hard_calls,
                distance,
                minimum_joint,
                config,
                inputs.cluster_cache,
                inputs.landmark_cache,
                reverse=inputs.reverse,
                keep=inputs.keep,
                diagnostics=inputs.diagnostics,
            )
            if omissions is not None:
                omissions.extend(details)
            results[route] = groups, overflow
        return results


def soft_candidates(
    haplotypes: np.ndarray,
    inputs: ResidualInputs,
    config: ResidualBirthParameters,
    counter: TieCounter | None = None,
    limit: int | None = None,
    omissions: list[dict] | None = None,
) -> np.ndarray:
    """float64[C, L] soft birth candidates of a complete binary panel, in emission order
    (``config``: ResidualBirthParameters; the block's sites are all kept; ``counter``: optional TieCounter of the
    band decisions, kinds discovery.residual.half_band and discovery.residual.mass_band, and of the order-invariant
    bounds, discovery.residual.group_overflow and discovery.residual.emission_overflow; ``limit``: at most this many
    candidates, by whole support classes)."""
    rows = np.ascontiguousarray(haplotypes, dtype=np.int64)
    k, n_sites = rows.shape
    pair_i, pair_j = np.triu_indices(k)
    pair_i = np.ascontiguousarray(pair_i, dtype=np.int64)
    pair_j = np.ascontiguousarray(pair_j, dtype=np.int64)
    responsibility = _diplotype_responsibility(inputs.log_likelihood, rows, pair_i, pair_j)
    (sample_indices, subtracted, partners, weights, soft_alt, hard_calls, compatible, log_odds, partner_membership) = (
        _residual_records(
            responsibility,
            pair_i,
            pair_j,
            k,
            inputs.likelihood,
            inputs.observed,
            rows,
            float(config.residual_call_probability),
            float(config.min_soft_responsibility),
            float(PARTNER_BAND),
            float(MASS_BAND),
            False,
        )
    )
    n_records = len(sample_indices)
    # Sources: global clusters, then clusters within each (subtracted row, partner) route.
    sources = []
    routes = _partner_routes(subtracted, partner_membership)
    minimum_joint = max(1, int(math.ceil(config.min_residual_joint_known_fraction * n_sites)))
    grouped = _residual_pool_results(hard_calls, routes, minimum_joint, config, inputs, omissions)
    if n_records >= 2:
        groups, overflow = grouped[None]
        sources.extend(groups)
        if counter is not None:
            counter.note("discovery.residual.group_overflow", inside_tie=overflow)
    if counter is not None:
        sizes = np.count_nonzero(partner_membership, axis=1)
        for kind, decisions, tied in (
            ("discovery.residual.partner_routes", n_records, int(np.count_nonzero(sizes > 1))),
            ("discovery.residual.partner_route_memberships", int(sizes.sum()), int(np.maximum(sizes - 1, 0).sum())),
            ("discovery.residual.partner_route_pools", len(routes), 0),
        ):
            counter.add_counts(kind, decisions=decisions, inside_tie=tied)
    for route in sorted(routes):
        pool = routes[route]
        if len(pool) < 2:
            continue
        local_groups, overflow = grouped[route]
        if counter is not None:
            counter.note("discovery.residual.group_overflow", inside_tie=overflow)
        for local in local_groups:
            sources.append(tuple(pool[index] for index in local))
    if not sources:
        return np.empty((0, n_sites), dtype=np.float64)
    fallback = hardening_fallback(inputs.likelihood, reverse=inputs.reverse, keep=inputs.keep)
    log_odds = np.ascontiguousarray(log_odds)
    # One consensus per distinct member set, in emission order (more distinct supporting samples,
    # hardened row, source order).
    prepared = []
    consensus_by_members: dict[tuple, tuple] = {}
    # Members ordered by (sample, subtracted row, record index): the rank of every record in that total order,
    # computed once (the same order as sorting each source by that key).
    record_rank = np.empty(n_records, dtype=np.int64)
    record_rank[np.lexsort((np.arange(n_records), subtracted, sample_indices))] = np.arange(n_records)
    for source_index, members in enumerate(sources):
        member_array = np.asarray(members, dtype=np.int64)
        ordered_array = member_array[np.argsort(record_rank[member_array], kind="stable")]
        ordered = tuple(ordered_array.tolist())
        cached = consensus_by_members.get(ordered)
        if cached is None:
            n_samples, support = _sample_support(ordered_array, sample_indices, weights)
            candidate, known_fraction = _consensus(
                ordered_array, sample_indices, weights, compatible, log_odds, float(config.candidate_call_probability)
            )
            cached = consensus_by_members[ordered] = (
                _hardened_key(candidate, fallback),
                n_samples,
                math.fsum(support[:n_samples].tolist()),
                candidate,
                known_fraction,
            )
            note_band(counter, "discovery.residual.half_band", candidate, 0.5, HALF_BAND)
            note_band(
                counter,
                "discovery.residual.mass_band",
                [cached[2]],
                float(config.min_effective_sample_support),
                MASS_BAND,
            )
        prepared.append((cached, source_index))
    prepared.sort(key=lambda item: (-item[0][1], conv_key((item[0][0], item[1]), reverse=inputs.reverse)))
    # Support classes are processed most supported first. A candidate is a near-duplicate only of a panel row
    # or a MORE supported candidate class; equally supported near-duplicates all remain, and exact duplicates
    # are one candidate. A straddling class fits through 2*limit; larger classes are omitted whole. Lower
    # support classes are counted as ranked truncation, separately from tie-class omissions.
    panel_rows = [np.asarray(row, dtype=np.float64) for row in rows]
    emitted: list[np.ndarray] = []
    emitted_bytes: set[bytes] = set()
    minimum_dedup_joint = max(1, int(math.ceil(config.min_dedup_joint_known_fraction * n_sites)))
    overflow = False
    stopped = False
    index = 0
    while index < len(prepared):
        support = prepared[index][0][1]
        stop = index
        while stop < len(prepared) and prepared[stop][0][1] == support:
            stop += 1
        if support < config.min_unique_sample_support:
            break
        reference = np.ascontiguousarray(np.stack(panel_rows + emitted))
        accepted, accepted_bytes = [], set()
        for (_digest, _unique_support, effective_support, candidate, known_fraction), _ in prepared[index:stop]:
            if effective_support + MASS_BAND < config.min_effective_sample_support:
                continue
            if known_fraction < config.min_candidate_known_fraction:
                continue
            exact = np.ascontiguousarray(candidate).tobytes()
            if exact in emitted_bytes or exact in accepted_bytes:
                continue
            found, closest_distance, _joint, candidate_coverage, other_coverage = _closest(
                candidate, reference, minimum_dedup_joint, n_sites, float(config.candidate_call_probability)
            )
            if (
                found
                and closest_distance <= config.dedup_hamming_fraction + RESIDUAL_GEOMETRY_SLACK
                and candidate_coverage >= config.min_dedup_bidirectional_coverage
                and other_coverage >= config.min_dedup_bidirectional_coverage
            ):
                continue
            accepted.append(candidate)
            accepted_bytes.add(exact)
        if limit is not None and (stopped or (len(emitted) > 0 and len(emitted) >= int(limit))):
            if accepted and omissions is not None:
                omissions.append(
                    dict(
                        cause="soft_residual_proposal_ranked_truncation",
                        kind="ranked_truncation",
                        count=len(accepted),
                        scope=f"residual support={support} class",
                        count_kind="exact",
                    )
                )
            stopped = True
            index = stop
            continue
        if limit is not None and len(emitted) + len(accepted) > 2 * int(limit):
            overflow = True
            stopped = True
            if omissions is not None:
                omissions.append(
                    dict(
                        cause="soft_residual_proposal_tie_class_at_budget",
                        kind="tie_class_at_budget",
                        count=len(accepted),
                        scope=f"residual support={support} class",
                        count_kind="exact",
                    )
                )
            index = stop
            continue
        emitted.extend(accepted)
        emitted_bytes |= accepted_bytes
        index = stop
    if counter is not None and limit is not None:
        counter.note("discovery.residual.emission_overflow", inside_tie=overflow)
    if not emitted:
        return np.empty((0, n_sites), dtype=np.float64)
    return np.ascontiguousarray(np.stack(emitted))
