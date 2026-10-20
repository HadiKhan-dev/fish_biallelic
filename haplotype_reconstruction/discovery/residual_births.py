"""Soft-residual birth candidates of a complete panel (the search's residual phase).

For a fitted complete panel, every sample's diplotype posterior over unordered row pairs (a
softmax of the summed log genotype likelihood of each hard pair, uniform pair prior) gives each
row an expected copy count. Every row with at least ``min_soft_responsibility`` expected copies
in a sample is *subtracted* from that sample: the residual strand's ALT probability at an
observed site is P(GL | subtracted allele + residual allele), normalised over the residual
allele. Residual strands are called at ``residual_call_probability``, clustered by missing-aware
Hamming distance (release 2.2: the level-wise HDBSCAN's clusters plus, as alternative groups, the maximal cliques of
its noise, of wider clusters and of the tied stragglers of multi-way merges; every group's diameter at
most ``max_cluster_hamming``), globally and within each (subtracted row, dominant partner)
route. Each cluster's consensus pools its samples' responsibility-weighted log odds with one
capped vote per sample; a consensus is emitted when enough distinct and effective samples
support it, enough of it is called, and it is no duplicate of the panel or an earlier emission.
Emission order (release 2.1): more distinct supporting samples first, then the consensus's hardened
0/1 row (> 1/2 is ALT; within the ``probability`` quantum of 1/2 the population ALT fraction decides,
as the search hardens it), then source order -- exact discrete keys (2.0 sorted by a digest of the
probabilities rounded to 12 decimals, which 1e-12 GL noise reshuffled). The order decides which of
two near-duplicate consensus rows is kept, so the better supported one is (measured on stress_n80:
fewer blocks losing a founder than with the hardened row first, design A-6). Every record enters all dominant-partner routes within the existing relative ``probability``
band of its largest partner mass. It remains one observation in global clustering and has its
original weight once within each route; convention order cannot select a route. The candidates are soft (ALT probabilities); the search hardens them into birth
starts and its cavity score decides.
"""
from __future__ import annotations

import math
from hashlib import blake2b
from itertools import product

import numpy as np
from numba import njit

from ..maths.ties import conv_key, tie_quantum
from .conventions import rev

# Bands of posterior-mass decisions (probability-kind quantum): the dominant-partner choice (relative to
# max(1, mass)), and the record-weight and effective-support thresholds, where symmetric evidence puts masses
# exactly at the threshold (e.g. a posterior split exactly in half, effective support exactly 1.5): values within
# the band count as at the threshold, which passes (2.0's decision at exact equality; 2.0 used a 1e-12 fuzz, the
# 1e-12-noise scale).
PARTNER_BAND = tie_quantum(0.0, "probability")
MASS_BAND = tie_quantum(0.0, "probability")
# Values within this band of 1/2 are 1/2 (probability-kind quantum; 2.0 compared exactly or with 1e-12, the 1e-12-noise
# scale). Inside the band a consensus row takes the population ALT fraction, and a population ALT fraction itself
# inside the band takes the rule's decision at exactly 1/2 (2.0's: REF for "> 1/2" rules, ALT for ">= 1/2").
HALF_BAND = tie_quantum(0.0, "probability")


def note_band(counter, kind, values, threshold, band):
    """Count one batch of threshold decisions for the stage records (counter: maths.ties.TieCounter or None):
    decisions = the values compared, inside_tie = those strictly inside the band (0 < |value - threshold| <= band:
    where the band, not the exact comparison, decided)."""
    if counter is None:
        return
    values = np.asarray(values, dtype=np.float64)
    distance = np.abs(values - threshold)
    entry = counter.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
    entry["decisions"] += int(values.size)
    entry["inside_tie"] += int(np.count_nonzero((distance > 0.0) & (distance <= band)))


def half_band_rows(values, fallback):
    """Binary rows of probabilities: > 1/2 is 1, < 1/2 is 0; within HALF_BAND of 1/2 the
    ``fallback`` row (broadcast) decides."""
    values = np.asarray(values, dtype=np.float64)
    rows = values > 0.5
    tied = np.abs(values - 0.5) <= HALF_BAND
    rows[tied] = np.broadcast_to(fallback, rows.shape)[tied]
    return rows


def alt_fraction_rule(population_alt, at_half: bool):
    """population ALT fraction > 1/2 (outside the band); within HALF_BAND of 1/2: ``at_half`` (a binary default
    state: the opposite under reversed conventions)."""
    population_alt = np.asarray(population_alt, dtype=np.float64)
    rows = population_alt > 0.5
    rows[np.abs(population_alt - 0.5) <= HALF_BAND] = (not at_half) if rev("d8s") else bool(at_half)
    return rows


@njit(cache=True)
def _accurate_sum(values, partials=None):
    """Finite-value expansion sum (the fsum reference, including halfway correction).

    The partials retain low bits discarded by ordinary addition. This makes population and
    cluster reductions independent of sample order, including cancellation of log odds.
    """
    if partials is None:
        partials = np.empty(len(values), dtype=np.float64)
    n = 0
    for value in values:
        x = value
        i = 0
        for j in range(n):
            y = partials[j]
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo != 0.0:
                partials[i] = lo
                i += 1
            x = hi
        partials[i] = x
        n = i + 1
    hi = 0.0
    lo = 0.0
    if n:
        n -= 1
        hi = partials[n]
        while n:
            x = hi
            n -= 1
            y = partials[n]
            hi = x + y
            lo = y - (hi - x)
            if lo != 0.0:
                break
        if n and ((lo < 0.0 and partials[n - 1] < 0.0) or (lo > 0.0 and partials[n - 1] > 0.0)):
            y = lo * 2.0
            x = hi + y
            if y == x - hi:
                hi = x
    return hi


@njit(cache=True)
def _sample_sum_2d(values):
    result = np.empty(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        result[column] = _accurate_sum(values[:, column])
    return result


def sample_sum(values):
    """Accurately sum the sample axis, matching math.fsum of each column."""
    values = np.asarray(values, dtype=np.float64)
    return _sample_sum_2d(values.reshape((values.shape[0], -1))).reshape(values.shape[1:])


def hardening_fallback(likelihoods):
    """The ALT/REF row hardening uses within HALF_BAND of 1/2: the population ALT fraction of all
    samples >= 1/2 (ALT within HALF_BAND of 1/2, 2.0's decision at exactly 1/2)."""
    marginal = sample_sum(0.5 * likelihoods[:, :, 1] + likelihoods[:, :, 2]) / len(likelihoods)
    return alt_fraction_rule(marginal, True)


# Elementary functions (release 2.1): discovery takes log and exp from the C library's scalar functions (math.log /
# math.exp in Numba kernels), never from NumPy's float64 ufuncs. NumPy picks those ufuncs' SIMD code paths at run time
# from the CPU (AVX-512 vs AVX2 kernels), and they differ from the scalar function by an ulp in ~0.3% (log) and ~5%
# (exp) of values, so the same evidence gave different soft-residual candidates on different CPUs (actm3 chr8:204 under
# env_haswell). The scalar functions, element-wise arithmetic and fixed-order sums give the same bits on every x86-64
# CPU with FMA (all CSD3 node types); discovery makes no BLAS calls.
@njit(cache=True, nogil=True)
def scalar_log_floored(values, floor):
    """float64 array of log(max(value, floor)) for a C-contiguous float64 array (C library scalar log)."""
    flat = values.reshape(-1)
    out = np.empty(flat.size, dtype=np.float64)
    for index in range(flat.size):
        value = flat[index]
        out[index] = math.log(value if value > floor else floor)
    return out.reshape(values.shape)


@njit(cache=True, nogil=True)
def _exp_inplace(values):
    """values[i, j] = exp(values[i, j]) for a 2-D float64 array of any layout (C library scalar exp)."""
    rows, columns = values.shape
    for row in range(rows):
        for column in range(columns):
            values[row, column] = math.exp(values[row, column])


@njit(cache=True, nogil=True)
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


def exact_hdbscan_labels(distance, min_cluster_size, min_samples, allow_single_cluster):
    """int64 HDBSCAN labels (-1 = noise) of a precomputed float64 distance matrix that depend neither on the order of
    the records nor on the order of equal-weight links (release 2.2): :func:`level_hdbscan`."""
    return level_hdbscan(distance, min_cluster_size, min_samples, allow_single_cluster)[0]


def level_hdbscan(distance, min_cluster_size, min_samples, allow_single_cluster):
    """(labels int64[n], stragglers bool[n], tree_groups) of HDBSCAN* on a precomputed float64 distance matrix.

    Mutual reachability (core distance: the distance to the min_samples-th nearest record) and a minimum spanning tree
    as in the hdbscan package (metric "precomputed", alpha 1); then the hierarchy merges all links of one weight at
    once, as HDBSCAN* is defined (Campello, Moulavi and Sander 2013; Campello et al. 2015: edges of equal weight are
    removed simultaneously), so it is the unique set of connected components of the links of at most each weight and
    no merge order enters (:func:`level_hdbscan_labels`). The package's binary dendrogram orders equal-weight links by
    NumPy's unstable argsort (CPU-dependent; release 2.1 by a fixed content key). Stragglers are the points whose
    grouping a binary merge order would decide; :func:`_cluster` offers them as alternative groups.
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
    return level_hdbscan_labels(n, tree[:, 0].astype(np.int64), tree[:, 1].astype(np.int64),
                                np.asarray(tree[:, 2], dtype=np.float64), int(min_cluster_size),
                                bool(allow_single_cluster))


def level_hierarchy(n, first, second, weight):
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
        groups = {}
        for point, old in previous.items():
            groups.setdefault(find(point), set()).add(old)
        for root, olds in groups.items():
            node = n + len(children)
            olds = sorted(olds)
            children.append(olds)
            level.append(float(value))
            members.append([p for old in olds for p in (members[old - n] if old >= n else (old,))])
            component[root] = node
        start = stop
    return children, level, members


def level_hdbscan_labels(n, first, second, weight, min_cluster_size, allow_single_cluster):
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

    cluster_parent, birth, terms, child_clusters = [-1], [0.0], [[]], [[]]
    fall = [None] * n
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
    rank = {cluster: index for index, cluster in enumerate(sorted(first_member, key=first_member.get))}
    subtree = [[] for _ in range(n_clusters)]
    for point in range(n):
        cluster = fall[point][0]
        while cluster != -1:
            subtree[cluster].append(point)
            cluster = cluster_parent[cluster]
    tree_groups = sorted(tuple(members) for members in subtree if len(members) >= 2)
    return np.array([rank[c] if c != -1 else -1 for c in assigned], dtype=np.int64), stragglers, tree_groups


class _CliqueStats:
    """Diagnostics only (no effect on results): with HAPLOTYPES_DISCOVERY_CLIQUE_STATS set to a directory, every clustering appends
    (alternative pool size, enumeration work units, overflow) to a per-process file there."""

    def __init__(self):
        self.path = None

    def record(self, pool, work, overflow, cliques=0, enumerated=True):
        import os
        directory = os.environ.get("HAPLOTYPES_DISCOVERY_CLIQUE_STATS")
        if not directory:
            return
        if self.path is None:
            self.path = os.path.join(directory, f"cliques_{os.getpid()}.tsv")
        with open(self.path, "a") as handle:
            handle.write(f"{pool}\t{work}\t{int(bool(overflow))}\t{cliques}\t{int(bool(enumerated))}\n")


_CLIQUE_STATS = _CliqueStats()


def _degeneracy_order(vertices, adjacent):
    """(degeneracy, smallest-last vertex order) of a graph. The degeneracy is a graph invariant; the order (ties by
    vertex label) only steers the enumeration, never its result."""
    remaining = set(vertices)
    degree = {vertex: len(adjacent[vertex] & remaining) for vertex in vertices}
    order, degeneracy = [], 0
    while remaining:
        vertex = min(remaining, key=lambda u: (degree[u], u))
        degeneracy = max(degeneracy, degree[vertex])
        order.append(vertex)
        remaining.discard(vertex)
        for other in adjacent[vertex]:
            if other in remaining:
                degree[other] -= 1
    return degeneracy, order


def _moon_moser(size: int) -> float:
    """The largest number of maximal independent sets of a graph on ``size`` vertices (Moon and Moser 1965)."""
    if size <= 2:
        return float(max(size, 1))
    if size % 3 == 0:
        return 3.0 ** (size // 3)
    if size % 3 == 1:
        return 4.0 * 3.0 ** ((size - 4) // 3)
    return 2.0 * 3.0 ** ((size - 2) // 3)


def clique_count_bound(vertices, adjacent) -> float:
    """An order-invariant upper bound on the number of maximal cliques of a connected graph: the smaller of
    Eppstein, Loeffler and Strash's (m - d) x 3^(d/3) (degeneracy d; tight for sparse graphs) and the product over the
    complement's connected components of their Moon-Moser maxima (maximal cliques are the complement's maximal
    independent sets; tight for dense graphs, e.g. 1 for a complete graph)."""
    m = len(vertices)
    if m == 0:
        return 0.0
    degeneracy, _ = _degeneracy_order(vertices, adjacent)
    sparse = (m - degeneracy) * 3.0 ** (degeneracy / 3.0)
    members = set(vertices)
    unseen = set(vertices)
    dense = 1.0
    while unseen:
        start = unseen.pop()
        component, frontier = 1, [start]
        while frontier:
            current = frontier.pop()
            others = (members - adjacent[current]) - {current}
            fresh = others & unseen
            unseen -= fresh
            component += len(fresh)
            frontier.extend(fresh)
        dense *= _moon_moser(component)
        if dense > sparse:
            break
    return min(sparse, dense)


def _graph_components(vertices, adjacent, complement=False):
    """Connected components, including isolated vertices, independent of vertex names."""
    members = set(vertices)
    unseen = set(vertices)
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, frontier = {start}, [start]
        while frontier:
            vertex = frontier.pop()
            neighbours = members - adjacent[vertex] - {vertex} if complement else adjacent[vertex]
            fresh = neighbours & unseen
            unseen -= fresh
            component |= fresh
            frontier.extend(fresh)
        components.append(tuple(sorted(component)))
    return components


def _clique_plan(vertices, adjacent):
    """Exact union/join decomposition and invariant (clique count, work) upper bounds.

    Disconnected graphs union their components' maximal cliques. Disconnected complements
    give a complete join, whose maximal cliques are Cartesian products of its parts' cliques.
    The bounds are diagnostic only; neither may reject a component. Count one work unit
    per recursive Bron-Kerbosch call or emitted join combination; this does not bound the
    optional reverse-search fallback. Prime pieces use sum_v 2^min(d, degree(v)).
    """
    members = set(vertices)
    degrees = [len(adjacent[vertex] & members) for vertex in vertices]
    if all(degree == len(vertices) - 1 for degree in degrees):
        return ("complete", tuple(vertices), ()), 1, 1
    components = _graph_components(vertices, adjacent)
    kind = "union"
    if len(components) == 1:
        components = _graph_components(vertices, adjacent, complement=True)
        kind = "join"
    if len(components) > 1:
        children = [_clique_plan(component, adjacent) for component in components]
        counts = [count for _, count, _ in children]
        count = sum(counts) if kind == "union" else math.prod(counts)
        work = sum(work for _, _, work in children) + (count if kind == "join" else 0)
        return (kind, tuple(vertices), tuple(plan for plan, _, _ in children)), count, work
    degeneracy, _ = _degeneracy_order(vertices, adjacent)
    work = sum(1 << min(degeneracy, degree) for degree in degrees)
    try:
        count = int(math.ceil(clique_count_bound(vertices, adjacent)))
    except OverflowError:
        # This upper bound is diagnostic only; enumeration never consults it.
        # All cliques are vertex subsets, so arbitrary-precision 2**n is valid
        # when the tighter floating-point bound cannot be represented.
        count = 1 << len(vertices)
    return ("prime", tuple(vertices), ()), count, work


def _reverse_search_cliques(vertices, adjacent, limit):
    """Polynomial-delay maximal-clique enumeration using bitset lexicographic completion.

    The parent/child reconstruction is Proposition 7 of Comin & Rizzi (2015),
    https://arxiv.org/abs/1506.01082, Section 3. Each emitted clique has at most n
    prospective children, each checked with two O(n) bitset completions. Child indices
    strictly increase, so the stack depth is at most n. Labels steer traversal only;
    the whole component is admitted iff its actual maximal-clique count <= limit.
    """
    vertices = tuple(sorted(vertices))
    n = len(vertices)
    slots = {vertex: i for i, vertex in enumerate(vertices)}
    neighbours = [sum(1 << slots[other] for other in adjacent[vertex] if other in slots)
                  for vertex in vertices]
    universe = (1 << n) - 1

    def complete(seed):
        available = universe
        remaining = seed
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            available &= neighbours[bit.bit_length() - 1]
        clique = seed
        while available:
            bit = available & -available
            clique |= bit
            available &= neighbours[bit.bit_length() - 1]
        return clique

    # Every stack entry is an actual, distinct maximal clique. There are no dead
    # recursive search branches and no traversal-dependent scientific cutoff.
    stack = [(complete(0), 0)]
    cliques = []
    work = 0
    while stack:
        parent, parent_index = stack.pop()
        work += 1
        if len(cliques) >= limit:
            return None, work
        cliques.append(tuple(vertices[i] for i in range(n) if parent & (1 << i)))
        remaining = universe & ~parent & ~((parent_index << 1) - 1 if parent_index else 0)
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            before = bit - 1
            prefix = parent & before
            seed = prefix & neighbours[bit.bit_length() - 1]
            work += 1
            if complete(seed) & before != prefix:
                continue
            child = complete(seed | bit)
            if child & before == seed:
                stack.append((child, bit))
    return cliques, work


def _bron_kerbosch_cliques(vertices, adjacent, limit, max_calls):
    """Fast exact traversal, or a request to restart with the bounded-delay method.

    ``max_calls`` only switches algorithm: it never rejects a graph. ``None`` for
    cliques means the actual output limit was exceeded; ``restart`` means use RS.
    """
    _, order = _degeneracy_order(vertices, adjacent)
    cliques = []
    work = 0
    restart = False
    overflow = False

    def expand(chosen, candidates, excluded):
        nonlocal work, restart, overflow
        work += 1
        if work > max_calls:
            restart = True
            return
        if not candidates and not excluded:
            if len(cliques) >= limit:
                overflow = True
            else:
                cliques.append(tuple(sorted(chosen)))
            return
        pivot = max(candidates | excluded, key=lambda u: (len(adjacent[u] & candidates), u))
        for vertex in sorted(candidates - adjacent[pivot]):
            expand(chosen | {vertex}, candidates & adjacent[vertex], excluded & adjacent[vertex])
            if restart or overflow:
                return
            candidates = candidates - {vertex}
            excluded = excluded | {vertex}

    position = {vertex: index for index, vertex in enumerate(order)}
    members = set(vertices)
    for vertex in order:
        neighbours = adjacent[vertex] & members
        later = {u for u in neighbours if position[u] > position[vertex]}
        expand({vertex}, later, neighbours - later)
        if restart or overflow:
            break
    return None if overflow else cliques, work, restart


def _enumerate_clique_plan(plan, adjacent, limit, max_calls):
    kind, vertices, children = plan
    if kind == "complete":
        return ([tuple(sorted(vertices))] if limit >= 1 else None), 1
    if children:
        parts, work, count = [], 0, (0 if kind == "union" else 1)
        for child in children:
            child_limit = limit - count if kind == "union" else limit // count
            groups, calls = _enumerate_clique_plan(child, adjacent, child_limit, max_calls)
            work += calls
            if groups is None:
                return None, work
            parts.append(groups)
            count = count + len(groups) if kind == "union" else count * len(groups)
        if kind == "union":
            return [group for groups in parts for group in groups], work
        cliques = [tuple(sorted(vertex for group in choice for vertex in group))
                   for choice in product(*parts)]
        return cliques, work + len(cliques)
    # Same complete output/count policy; the packed compiled traversal replaces
    # the Python fast-traversal/reverse-search performance switch only.
    from ._cliques import maximal_cliques as packed_cliques
    return packed_cliques(vertices, adjacent, limit)


def maximal_cliques(vertices, adjacent, limit, max_calls=200_000, omissions=None):
    """All maximal cliques, or omit the component if its actual count exceeds limit.

    Exact union/join decomposition avoids materializing an oversized Cartesian product.
    Prime pieces use exact compiled, pivoted bitset Bron--Kerbosch traversal.
    Count first, materialize only when admitted. The legacy max_calls argument
    remains compatible with reference callers; it cannot truncate scientific output.
    No a-priori worst-case bound rejects a component.
    """
    if not vertices:
        return [], 0
    plan, _, _ = _clique_plan(vertices, adjacent)
    cliques, work = _enumerate_clique_plan(plan, adjacent, int(limit), int(max_calls))
    if cliques is None:
        if omissions is not None:
            omissions.append(dict(cause="residual_clique_enumeration_tie_class_at_budget",
                                  kind="tie_class_at_budget", count=1,
                                  scope=f"component with {len(vertices)} records; more than {limit} maximal cliques",
                                  count_kind="exact"))
        return None, work
    return sorted(cliques), work


def capped_by_size(groups, cap, omissions=None, *, cause="residual_group", scope="size classes", group_size=len):
    """Whole size classes through the cap, admitting a straddler through 2*cap.

    A straddler beyond twice the cap is an omitted tie class. Strictly smaller
    classes beyond the stopping boundary are ordinary ranked truncation.
    """
    kept = []
    by_size = {}
    for group in groups:
        by_size.setdefault(group_size(group), []).append(group)
    classes = [by_size[size] for size in sorted(by_size, reverse=True)]
    for index, members in enumerate(classes):
        before, after = len(kept), len(kept) + len(members)
        if after <= cap:
            kept.extend(members)
            continue
        tie_omitted = (before < cap or before == 0) and after > 2 * cap
        if before < cap and after <= 2 * cap:
            kept.extend(members)
        if omissions is not None:
            if tie_omitted:
                omissions.append(dict(cause=cause + "_tie_class_at_budget", kind="tie_class_at_budget",
                                      count=len(members), scope=scope, count_kind="exact"))
            tail = sum(map(len, classes[index + 1:])) + (len(members) if before >= cap and not tie_omitted else 0)
            if tail:
                omissions.append(dict(cause=cause + "_ranked_truncation", kind="ranked_truncation",
                                      count=tail, scope=scope, count_kind="exact"))
        return kept, tie_omitted
    return kept, False


# Residual strand hard-call sentinel.
_UNCALLED = -1
# Bounds of a residual strand's log odds: release 2.0 clipped the ALT probability to [tiny, 1 - eps] before taking
# log(p) - log1p(-p), so its log odds lay in [log tiny, log((1 - eps) / eps)]. 2.1 computes the log odds exactly as
# log(residual ALT mass) - log(residual REF mass) within the same bounds: from p itself they lost all precision as
# p -> 1 (1 - p carries ~1e-16 absolute error), and 1e-12 GL noise moved a record's log odds by 0.15 nats there
# (TroMau chr7:665, chr17:425), flipping near-cancelling consensus sites.
_LOG_ODDS_MAX = math.log((1.0 - np.finfo(np.float64).eps) / np.finfo(np.float64).eps)
_LOG_ODDS_MIN = math.log(np.finfo(np.float64).tiny) - math.log1p(-np.finfo(np.float64).tiny)
# Bound of the largest temporary of the diplotype responsibility (bytes).
_PREDICTIVE_CHUNK_BYTES = 32 * 1024 * 1024


class ResidualInputs:
    """Evidence-only inputs of one block: likelihood (unobserved cells exactly 1/3), its floored
    logarithm, the observation mask, and an exact cache of residual clusterings (identical
    distance matrices recur across proposals of one block search)."""

    def __init__(self, likelihood, observed):
        self.likelihood = np.ascontiguousarray(likelihood, dtype=np.float64)
        self.log_likelihood = scalar_log_floored(self.likelihood, np.finfo(np.float64).tiny)
        self.observed = np.ascontiguousarray(observed, dtype=np.bool_)
        self.cluster_cache: dict = {}
        self.landmark_cache: dict = {}


@njit(cache=True, nogil=True)
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
    log_emission = np.empty((n_samples, n_pairs), dtype=np.float64,
                            order="F" if n_samples > 1 else "C")
    for start in range(0, n_samples, samples_per_chunk):
        stop = min(start + samples_per_chunk, n_samples)
        log_predictive = np.empty((stop - start, n_pairs, n_sites), dtype=np.float64, order="F")
        _gather_binary_predictive(log_likelihood[start:stop], haplotypes, pair_i, pair_j,
                                  log_predictive)
        log_emission[start:stop] = np.sum(log_predictive, axis=2)
    row_maximum = np.max(log_emission, axis=1, keepdims=True)
    responsibility = log_emission - row_maximum
    _exp_inplace(responsibility)
    responsibility /= np.sum(responsibility, axis=1, keepdims=True)
    return np.ascontiguousarray(responsibility)


@njit(cache=True, nogil=True)
def _residual_records(responsibility, pair_i, pair_j, n_rows, likelihood, observed, haplotypes,
                      call_probability, minimum_responsibility, partner_band, mass_band, reverse):
    """Residual strands of every (sample, subtracted row) with enough expected copies.

    Returns sample, subtracted row, a reporting-only partner label, weight min(1, expected
    copies), the residual ALT probability per site (0.5 where unobserved), hard calls (-1
    uncalled), the compatible mask (observed sites with positive residual mass) and the residual
    ALT log odds log(r1) - log(r0) at compatible sites within [_LOG_ODDS_MIN, _LOG_ODDS_MAX] (0
    elsewhere), and the complete dominant-partner membership mask. A record needs
    weight + mass_band >= minimum_responsibility. Partners within
    partner_band * max(1, global maximum mass) of that maximum all receive the same
    original record. The legacy reverse argument cannot change this class; the lower
    member is retained only as a reporting label, never as a routing decision.
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
    return (sample_indices[:count], subtracted[:count], partners[:count], weights[:count],
            soft_alt, hard_calls, compatible, log_odds, partner_membership[:count])


@njit(cache=True, inline="always")
def _popcount(value):
    """64-bit population count (SWAR reduction)."""
    value = value - ((value >> np.uint64(1)) & np.uint64(0x5555555555555555))
    value = ((value & np.uint64(0x3333333333333333))
             + ((value >> np.uint64(2)) & np.uint64(0x3333333333333333)))
    value = (value + (value >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((value * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
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


@njit(cache=True)
def _diameter_within(distance, members, bound):
    """The dense submatrix maximum test, without allocating that submatrix."""
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if distance[members[i], members[j]] > bound:
                return False
    return True


def _cluster(distance, maximum_hamming, max_groups, max_enumeration, max_calls=200_000, omissions=None, group_size=len, group_identity=None):
    """(groups, overflow): cohesive groups of residual records, independent of the record and link order
    (release 2.2; replaces one tie order's clusters plus complete linkage).

    Core: every cluster of the level-wise HDBSCAN's condensed tree (:func:`level_hdbscan`; min cluster size 2, min
    samples 1, single cluster allowed), selected or not, whose diameter is at most maximum_hamming. Alternatives:
    every maximal clique (at least
    two records) of the graph "distance <= maximum_hamming" on the records HDBSCAN leaves as noise, the records of
    clusters wider than the bound and the stragglers of multi-way merges (which stay in their cluster too). Every
    complete-linkage group at the bound, and every pair-up of tied stragglers within it, under any order, lies inside
    one of these cliques: the tied resolutions are carried forward as candidates and the block score decides. Bound:
    at most ``max_groups`` cliques, by whole size classes (capped_by_size), and no cliques at all when there are more
    than ``max_enumeration``; either is reported as ``overflow``. Groups are sorted tuples of record indices, in
    sorted order."""
    n = distance.shape[0]
    if n < 2:
        return (), False
    bound = maximum_hamming + 1e-12
    if n == 2:
        return (((0, 1),) if float(distance[0, 1]) <= bound else ()), False
    labels, stragglers, tree_groups = level_hdbscan(distance, 2, 1, True)
    # Every cluster of the condensed tree (all density levels, selected or not) within the bound is a group: the
    # excess of mass picks one level per branch, a binary merge order can shift that pick, and the block score can
    # decide instead.
    groups = [members for members in tree_groups
              if _diameter_within(distance, np.asarray(members, dtype=np.int64), bound)]
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
    overflow = False
    if len(pool) >= 2:
        vertices = sorted(pool)
        adjacent = {vertex: set() for vertex in vertices}
        for position, first in enumerate(vertices):
            for second in vertices[position + 1:]:
                if float(distance[first, second]) <= bound:
                    adjacent[first].add(second)
                    adjacent[second].add(first)
        cliques, work, enumerated = [], 0, True
        components = []
        compact = group_identity is None and group_size is len
        seen = set()
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
            if len(component) < 2:
                continue
            if compact:
                components.append(tuple(sorted(component)))
                continue
            found, calls = maximal_cliques(sorted(component), adjacent, int(max_enumeration), int(max_calls), omissions)
            work += calls
            if found is None:
                enumerated = False
                continue
            cliques.extend(clique for clique in found if len(clique) >= 2)
        if group_identity is not None:
            unique = {}
            for clique in cliques:
                unique.setdefault(group_identity(clique), clique)
            cliques = list(unique.values())
        if compact:
            from ._cliques import screened_components
            kept, capped, enumerated, work, clique_count = screened_components(
                components, adjacent, int(max_enumeration), int(max_groups), omissions)
        else:
            kept, capped = capped_by_size(cliques, int(max_groups), omissions,
                                          scope="residual alternative size classes", group_size=group_size)
            clique_count = len(cliques)
        overflow = capped or not enumerated
        known = set(groups)
        groups.extend(clique for clique in kept if clique not in known)
        _CLIQUE_STATS.record(len(vertices), work, overflow, clique_count, enumerated)
    return tuple(sorted(set(groups))), overflow


def _cluster_cached(distance, maximum_hamming, cache, max_groups=64, max_enumeration=10_000_000, max_calls=200_000,
                    omissions=None):
    # A fixed-size content digest avoids retaining another dense matrix per memo.
    # Hash the contiguous buffer directly, without an R-by-R bytes temporary.
    value = np.ascontiguousarray(distance)
    key = (float(maximum_hamming), int(max_groups), int(max_enumeration), int(max_calls), value.shape,
           blake2b(memoryview(value).cast("B"), digest_size=32).digest())
    result = cache.get(key)
    if result is None:
        details = []
        groups, overflow = _cluster(distance, maximum_hamming, max_groups, max_enumeration, max_calls, details)
        result = cache[key] = (groups, overflow, tuple(details))
    if omissions is not None:
        omissions.extend(result[2])
    return result[:2]


@njit(cache=True)
def _pooled_log_odds(members, sample_indices, weights, compatible, log_odds):
    """Per site, the sum over samples of min(1, w_s) x (the sample's responsibility-weighted
    mean log odds over its records), with the number of contributing samples. ``members`` are
    ordered by sample."""
    n_sites = log_odds.shape[1]
    total = np.zeros(n_sites, dtype=np.float64)
    sample_terms = np.zeros((members.shape[0], n_sites), dtype=np.float64)
    term_index = 0
    contributing = np.zeros(n_sites, dtype=np.int32)
    weighted = np.zeros(n_sites, dtype=np.float64)
    weight_at_site = np.zeros(n_sites, dtype=np.float64)
    if members.shape[0] == 0:
        return total, contributing
    current = sample_indices[members[0]]
    for position in range(members.shape[0]):
        record = members[position]
        sample = sample_indices[record]
        if sample != current:
            for site in range(n_sites):
                weight = weight_at_site[site]
                if weight > 0.0:
                    sample_weight = 1.0 if weight > 1.0 else weight
                    sample_terms[term_index, site] = sample_weight * (weighted[site] / weight)
                    contributing[site] += 1
                    weighted[site] = 0.0
                    weight_at_site[site] = 0.0
            term_index += 1
            current = sample
        record_weight = weights[record]
        for site in range(n_sites):
            if compatible[record, site]:
                weighted[site] += record_weight * log_odds[record, site]
                weight_at_site[site] += record_weight
    for site in range(n_sites):
        weight = weight_at_site[site]
        if weight > 0.0:
            sample_weight = 1.0 if weight > 1.0 else weight
            sample_terms[term_index, site] = sample_weight * (weighted[site] / weight)
            contributing[site] += 1
        total[site] = _accurate_sum(sample_terms[:term_index + 1, site])
    return total, contributing


def _consensus(members, sample_indices, weights, compatible, log_odds, call_probability):
    """(ALT probability per site, fraction of sites called at ``call_probability``)."""
    total, contributing = _pooled_log_odds(
        np.ascontiguousarray(members, dtype=np.int64), sample_indices, weights, compatible,
        log_odds)
    observed = contributing > 0
    posterior = _logistic_where(total, observed)
    confident = observed & ((posterior >= call_probability) | (posterior <= 1.0 - call_probability))
    return np.ascontiguousarray(posterior), float(np.mean(confident))


@njit(cache=True)
def _closest(candidate, existing, minimum_joint, n_sites, known_probability):
    """Nearest existing row by missing-aware Hamming distance over sites called in both
    (rounded ALT probabilities beyond ``known_probability``); ties within 1e-12 prefer the
    larger joint-called fraction (a stated preference: more shared evidence). Returns (found, distance, joint
    fraction, coverage, coverage), the coverage being the smaller of the candidate's and the row's call coverage,
    maximised over the rows tied at the nearest (distance, joint fraction): no row order decides (release 2.2)."""
    lower = 1.0 - known_probability + 1e-12
    upper = known_probability - 1e-12
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
    closest_other_coverage = 0.0
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
        if (not found or distance < closest_distance - 1e-12
                or (abs(distance - closest_distance) <= 1e-12 and joint_fraction > closest_joint)):
            found = True
            closest_distance = distance
            closest_joint = joint_fraction
            closest_candidate_coverage = coverage
        elif abs(distance - closest_distance) <= 1e-12 and joint_fraction == closest_joint:
            closest_candidate_coverage = max(closest_candidate_coverage, coverage)
    return (found, closest_distance, closest_joint, closest_candidate_coverage,
            closest_candidate_coverage)


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


def soft_candidates(haplotypes, inputs: ResidualInputs, config, counter=None, limit=None, omissions=None) -> np.ndarray:
    """float64[C, L] soft birth candidates of a complete binary panel, in emission order
    (``config``: ResidualBirthConfig; the block's sites are all kept; ``counter``: optional TieCounter of the
    band decisions, kinds discovery.residual.half_band and discovery.residual.mass_band, and of the order-invariant
    bounds, discovery.residual.group_overflow and discovery.residual.emission_overflow; ``limit``: at most this many
    candidates, by whole support classes)."""
    rows = np.ascontiguousarray(haplotypes, dtype=np.int64)
    k, n_sites = rows.shape
    pair_i, pair_j = np.triu_indices(k)
    pair_i = np.ascontiguousarray(pair_i, dtype=np.int64)
    pair_j = np.ascontiguousarray(pair_j, dtype=np.int64)
    responsibility = _diplotype_responsibility(inputs.log_likelihood, rows, pair_i, pair_j)
    (sample_indices, subtracted, partners, weights, soft_alt, hard_calls,
     compatible, log_odds, partner_membership) = _residual_records(
        responsibility, pair_i, pair_j, k, inputs.likelihood, inputs.observed, rows,
        float(config.residual_call_probability), float(config.min_soft_responsibility),
        float(PARTNER_BAND), float(MASS_BAND), False)
    n_records = len(sample_indices)
    # Sources: global clusters, then clusters within each (subtracted row, partner) route.
    sources = []
    distance = None
    if n_records >= 2:
        minimum_joint = max(1, int(math.ceil(config.min_residual_joint_known_fraction * n_sites)))
        if config.landmark_budget and n_records > config.landmark_budget:
            from .landmarks import residual_groups
            groups, overflow, _stats = residual_groups(hard_calls, config.landmark_budget, minimum_joint,
                                                       config, omissions, "global residuals", inputs.landmark_cache)
        else:
            distance = _hamming_distances(np.ascontiguousarray(hard_calls), minimum_joint)
            groups, overflow = _cluster_cached(distance, config.max_cluster_hamming, inputs.cluster_cache,
                                               config.max_alternative_groups, config.max_clique_enumeration,
                                               config.max_clique_calls, omissions)
        sources.extend(groups)
        if counter is not None:
            counter.note("discovery.residual.group_overflow", inside_tie=overflow)
    routes = _partner_routes(subtracted, partner_membership)
    if counter is not None:
        sizes = np.count_nonzero(partner_membership, axis=1)
        for kind, decisions, tied in (
                ("discovery.residual.partner_routes", n_records, int(np.count_nonzero(sizes > 1))),
                ("discovery.residual.partner_route_memberships", int(sizes.sum()),
                 int(np.maximum(sizes - 1, 0).sum())),
                ("discovery.residual.partner_route_pools", len(routes), 0)):
            entry = counter.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
            entry["decisions"] += decisions
            entry["inside_tie"] += tied
    for route in sorted(routes):
        pool = routes[route]
        if len(pool) < 2:
            continue
        if config.landmark_budget and len(pool) > config.landmark_budget:
            from .landmarks import residual_groups
            local_groups, overflow, _stats = residual_groups(hard_calls[list(pool)], config.landmark_budget,
                minimum_joint, config, omissions, "partner residual route", inputs.landmark_cache)
        else:
            sub_distance = (distance[np.ix_(pool, pool)] if distance is not None else
                            _hamming_distances(np.ascontiguousarray(hard_calls[list(pool)]), minimum_joint))
            local_groups, overflow = _cluster_cached(sub_distance, config.max_cluster_hamming, inputs.cluster_cache,
                                                     config.max_alternative_groups, config.max_clique_enumeration,
                                                     config.max_clique_calls, omissions)
        if counter is not None:
            counter.note("discovery.residual.group_overflow", inside_tie=overflow)
        for local in local_groups:
            sources.append(tuple(pool[index] for index in local))
    if not sources:
        return np.empty((0, n_sites), dtype=np.float64)
    fallback = hardening_fallback(inputs.likelihood)
    log_odds = np.ascontiguousarray(log_odds)
    # One consensus per distinct member set, in emission order (more distinct supporting samples,
    # hardened row, source order).
    prepared = []
    consensus_by_members: dict[tuple, tuple] = {}
    for source_index, members in enumerate(sources):
        ordered = tuple(sorted(members, key=lambda index: (
            sample_indices[index], subtracted[index], index)))
        cached = consensus_by_members.get(ordered)
        if cached is None:
            by_sample: dict[int, float] = {}
            for index in ordered:
                sample = int(sample_indices[index])
                by_sample[sample] = min(1.0, by_sample.get(sample, 0.0) + float(weights[index]))
            candidate, known_fraction = _consensus(
                ordered, sample_indices, weights, compatible, log_odds,
                float(config.candidate_call_probability))
            cached = consensus_by_members[ordered] = (
                _hardened_key(candidate, fallback), len(by_sample), math.fsum(by_sample.values()),
                candidate, known_fraction)
            note_band(counter, "discovery.residual.half_band", candidate, 0.5, HALF_BAND)
            note_band(counter, "discovery.residual.mass_band", [cached[2]],
                      float(config.min_effective_sample_support), MASS_BAND)
        prepared.append((cached, source_index))
    prepared.sort(key=lambda item: (-item[0][1], conv_key((item[0][0], item[1]))))
    # Support classes, most supported first (a stated preference). A candidate is a near-duplicate only of a panel
    # row or of a candidate of a MORE supported class: equally supported near-duplicates are all kept, since no order
    # may decide between them, and exact duplicates are one candidate (release 2.2; 2.1 kept the first by content
    # key). A straddling class fits through 2*limit; beyond that it is omitted whole.
    # Lower support classes are counted as ranked truncation, separately from tie-class omissions.
    panel_rows = [np.asarray(row, dtype=np.float64) for row in rows]
    emitted, emitted_bytes = [], set()
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
        for (digest, unique_support, effective_support, candidate, known_fraction), _ in prepared[index:stop]:
            if effective_support + MASS_BAND < config.min_effective_sample_support:
                continue
            if known_fraction < config.min_candidate_known_fraction:
                continue
            exact = np.ascontiguousarray(candidate).tobytes()
            if exact in emitted_bytes or exact in accepted_bytes:
                continue
            found, closest_distance, _joint, candidate_coverage, other_coverage = _closest(
                candidate, reference, minimum_dedup_joint, n_sites, float(config.candidate_call_probability))
            if (found and closest_distance <= config.dedup_hamming_fraction + 1e-12
                    and candidate_coverage >= config.min_dedup_bidirectional_coverage
                    and other_coverage >= config.min_dedup_bidirectional_coverage):
                continue
            accepted.append(candidate)
            accepted_bytes.add(exact)
        if limit is not None and (stopped or (len(emitted) > 0 and len(emitted) >= int(limit))):
            if accepted and omissions is not None:
                omissions.append(dict(cause="soft_residual_proposal_ranked_truncation",
                                      kind="ranked_truncation", count=len(accepted),
                                      scope=f"residual support={support} class", count_kind="exact"))
            stopped = True
            index = stop
            continue
        if limit is not None and len(emitted) + len(accepted) > 2 * int(limit):
            overflow = True
            stopped = True
            if omissions is not None:
                omissions.append(dict(cause="soft_residual_proposal_tie_class_at_budget",
                                      kind="tie_class_at_budget", count=len(accepted),
                                      scope=f"residual support={support} class", count_kind="exact"))
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
