"""The local selection objective: fitted log likelihood plus a coding regularizer.

Owns LS03/LS04; no environment, storage, clocks or diagnostics side effects.
Method: docs/methods/selection.md#ls03.
Selection.objective.description prefers the largest matching within its relative band;
matching relaxations are strict and canonical matching order is a stated rule, never reversed.

regularizer(panel, f) = -log C(2^L, free) - claimed_code - log K - log(K + 1)
                        + (1/K) sum_a log(K f_a)
The dictionary term codes the ``free`` rows as new arbitrary binary haplotypes; rows claimed by
the assembled context's founders are coded by reference to their founder plus their edits
(``ContextFounders``). The frequency term is a shrinkage penalty centred at uniform frequencies
(zero there). This is a regularised best-panel objective, not a Bayesian posterior over K.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from numba import types
from numpy.typing import NDArray

from ...contracts.selection import PathFit
from ...core.kernels import kernel
from ..discovery.api import log_haplotype_set_count
from .parameters import DESCRIPTION_MEMO_ENTRIES, DESCRIPTION_TIE, MATCHING_SLACK

# Exact Numba argument types observed at the immutable public boundaries.
_F64_C2 = types.Array(types.float64, 2, "C", readonly=False)

__all__ = [
    "ContextFounders",
    "ContextScreen",
    "coded_regularizer",
    "edit_ball_code",
    "matching_costs",
    "regularizer",
    "selection_objective",
]


def regularizer(
    panel: NDArray[np.int64], frequencies: NDArray[np.float64], free: int | None = None, claimed_code: float = 0.0
) -> float:
    """Return the fixed-K founder prior and description penalty in nats.

    ``panel`` is a distinct binary int64[K, L] dictionary with K > 0;
    ``frequencies`` is a positive normalized float64[K] vector in the same row
    order. ``free`` counts rows charged the dictionary code (default K), while
    ``claimed_code`` is the context description cost in nats. The returned
    log-prior includes the K prior and uniform-centred frequency shrinkage.
    Inputs are neither normalized nor mutated here.
    """
    k, length = panel.shape
    free = k if free is None else int(free)
    dictionary = float(log_haplotype_set_count(length, free)) if free else 0.0
    value = -dictionary - float(claimed_code) - math.log(k) - math.log(k + 1)
    # Shrinkage centred at uniform; its MAP extra counts total one.
    value += float(np.log(k * frequencies).sum() / k)
    return value


class ContextFounders:
    """The assembled context's founders in one block, for the dictionary code.

    ``founders`` holds one local row per assembled founder of the covering component (this round's
    context proposals repeated by their multiplicities; -1 = no call): founders whose local rows are
    equal are separate claimers. Founders without a called site claim no row.

    Any founder may claim any panel row; a claimed row is coded by reference to its founder plus its
    edits, log V(n_f, d) with V(n, d) = sum_{j<=d} C(n, j) for d differing sites among the founder's
    n_f called sites (0 for an exact copy, increasing in d, at most n_f log 2). Rows no founder claims
    are coded as new rows (the dictionary term). The panel's code is its shortest description: the
    number of claimed rows m and the matching minimise the summed edit codes plus the free rows'
    dictionary term (``_best_assignment``), so the code is a function of the row set, and for fixed
    claimers and called masks one changed call moves it by at most one edit increment, never by a
    whole free row. The code covers only called sites: a claimed row's alleles at its founder's
    uncalled sites carry no code (a context-reuse penalty, not a complete code). Distances are cached
    per distinct row.

    """

    def __init__(self, founders: NDArray[np.int64], *, reverse: bool = False) -> None:
        founders = np.asarray(founders, dtype=np.int64)
        called = founders >= 0
        informative = called.any(axis=1)
        self.founders = np.ascontiguousarray(founders[informative])
        self.called = np.ascontiguousarray(called[informative])
        # Index of each kept founder in the ``founders`` argument (uninformative founders are dropped).
        self.source = np.flatnonzero(informative)
        self.n_called = self.called.sum(axis=1)
        for array in (self.founders, self.called, self.source, self.n_called):
            array.flags.writeable = False
        self._distances: dict[bytes, np.ndarray] = {}
        # A panel's description is independent of frequencies and is reused by fits and snaps.
        self._descriptions: dict[bytes, tuple[np.ndarray, int, float]] = {}

    def __len__(self) -> int:
        return len(self.founders)

    def _columns(self, panel):
        columns = []
        for row in np.asarray(panel, dtype=np.int64):
            key = row.tobytes()
            value = self._distances.get(key)
            if value is None:
                value = self._distances[key] = ((self.founders != row[None, :]) & self.called).sum(axis=1)
            columns.append(value)
        return np.stack(columns, axis=1)

    def claims(self, panel: NDArray[np.int64]) -> tuple[NDArray[np.bool_], int, float]:
        """(claimed mask, free-row count, claimed rows' edit code in nats)."""
        k, length = np.shape(panel)
        claimed = np.zeros(k, dtype=bool)
        if not len(self.founders):
            return claimed, k, 0.0
        panel = np.ascontiguousarray(panel, dtype=np.int64)
        key = repr(panel.shape).encode() + panel.tobytes()
        known = self._descriptions.get(key)
        if known is None:
            if len(self._descriptions) >= DESCRIPTION_MEMO_ENTRIES:
                self._descriptions.clear()
            known = self._descriptions[key] = self._best_assignment(self._columns(panel), k, length)
        return known[0].copy(), known[1], known[2]

    def matched(self, panel: NDArray[np.int64]) -> list[tuple[int, int]]:
        """(founder, row) pairs of one shortest description of the panel, chosen in canonical order (founders by
        content and called mask, rows by content) and mapped back, so that equal-cost choices do not depend on the
        order of the founders or of the rows (founders with equal rows and masks are interchangeable)."""
        panel = np.asarray(panel, dtype=np.int64)
        k, length = panel.shape
        if not len(self.founders):
            return []
        founder_order = sorted(
            range(len(self.founders)), key=lambda f: (self.founders[f].tobytes(), self.called[f].tobytes())
        )
        row_order = sorted(range(k), key=lambda r: panel[r].tobytes())
        distance = self._columns(panel)[np.ix_(founder_order, row_order)]
        assignment = self._description(distance, k, length, self.n_called[founder_order])[3]
        return sorted((founder_order[f], row_order[int(r)]) for f, r in enumerate(assignment) if r >= 0)

    def _best_assignment(self, distance, k, length):
        return self._description(distance, k, length)[:3]

    def _description(self, distance, k, length, n_called=None):
        """(claimed mask, free-row count, edit code) of the panel's shortest description:
        any founder may claim any row at the monotone edit-ball code of their distance, and the number
        of claimed rows m is chosen to minimise the summed edit codes plus the free rows' dictionary
        term. Minimum edit costs of every matching size come from one successive-shortest-augmenting-path pass
        (O(min(F, K) (F + K)^2)). For a fixed set of claimers and called masks, one changed call moves the code by
        at most its edit-ball increment, never by a whole free row; founders with equal local rows stay separate
        claimers, so a call making two founders' rows equal keeps that bound."""
        f_count = len(self.founders)
        n_called = self.n_called if n_called is None else n_called
        cost = np.empty((f_count, k))
        for f in range(f_count):
            cost[f] = _edit_codes(int(n_called[f]))[np.asarray(distance[f], dtype=np.int64)]
        sizes = matching_costs(cost)
        free, _ = _shortest_description([code for code, _ in sizes], k, length)
        m = k - free
        code, assignment = sizes[m]
        claimed = np.zeros(k, dtype=bool)
        claimed[[r for r in assignment if r >= 0]] = True
        return claimed, k - m, float(code), assignment

    def free(self, panel: NDArray[np.int64]) -> int:
        return self.claims(panel)[1]


@functools.cache
def edit_ball_code(n: int, d: int) -> float:
    """log sum_{j<=d} C(n, j) (nats): the code length of a row within Hamming distance d of a founder's n called sites.
    Increasing in d (unlike log C(n, d), which falls again beyond n/2); 0 at d = 0."""
    if d <= 0:
        return 0.0
    if d >= n:
        return n * math.log(2.0)  # the whole space: the exact endpoint
    terms = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) for j in range(min(d, n) + 1)]
    top = max(terms)
    # Never above the whole space's exact code (rounding of the sum near d = n - 1).
    return min(top + math.log(math.fsum(math.exp(x - top) for x in terms)), n * math.log(2.0))


@functools.cache
def _edit_codes(n: int) -> NDArray[np.float64]:
    """edit_ball_code(n, d) for d = 0..n (a founder's distance to a row never exceeds its called sites), read-only."""
    codes = np.array([edit_ball_code(n, d) for d in range(n + 1)], dtype=np.float64)
    codes.flags.writeable = False
    return codes


def matching_costs(cost: NDArray[np.float64]) -> list[tuple[float, NDArray[np.int64]]]:
    """[(minimum total cost, row of each founder or -1)] of bipartite matchings of every size m = 0..min(F, K)
    between founders (rows of ``cost``, non-negative) and panel rows (columns): successive shortest augmenting paths
    from a source through founders and rows to a sink, Dijkstra on reduced costs with Johnson potentials (dense,
    O((F + K)^2) per augmentation, O(K^3) in all). Lowest node index wins equal distances (deterministic)."""
    cost = np.ascontiguousarray(cost, dtype=np.float64)
    totals, assignments = _matching_costs(cost)
    return [(float(totals[m]), assignments[m].copy()) for m in range(len(totals))]


@kernel(overloads=((_F64_C2,),), cache=True, nogil=True)
def _matching_costs(cost):
    f_count, k = cost.shape
    n = f_count + k + 2
    source, sink = 0, n - 1
    steps = min(f_count, k)
    totals = np.zeros(steps + 1)
    assignments = np.full((steps + 1, f_count), -1, dtype=np.int64)
    match_f = np.full(f_count, -1, dtype=np.int64)
    match_r = np.full(k, -1, dtype=np.int64)
    height = np.zeros(n)
    filled = 0
    for step in range(steps):
        dist = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=np.int64)
        done = np.zeros(n, dtype=np.bool_)
        dist[source] = 0.0
        while True:
            u = -1
            best = np.inf
            for v in range(n):
                if not done[v] and dist[v] < best:
                    best = dist[v]
                    u = v
            if u < 0:
                break
            done[u] = True
            if u == source:
                for f in range(f_count):
                    if match_f[f] < 0:
                        value = dist[u] + height[u] - height[1 + f]
                        if value < dist[1 + f] - MATCHING_SLACK:
                            dist[1 + f] = value
                            prev[1 + f] = u
            elif u <= f_count:
                f = u - 1
                for r in range(k):
                    if match_f[f] != r:
                        v = 1 + f_count + r
                        value = dist[u] + cost[f, r] + height[u] - height[v]
                        if value < dist[v] - MATCHING_SLACK:
                            dist[v] = value
                            prev[v] = u
            elif u < sink:
                r = u - 1 - f_count
                if match_r[r] >= 0:
                    g = match_r[r]
                    v = 1 + g
                    value = dist[u] - cost[g, r] + height[u] - height[v]
                else:
                    v = sink
                    value = dist[u] + height[u] - height[v]
                if value < dist[v] - MATCHING_SLACK:
                    dist[v] = value
                    prev[v] = u
        if not dist[sink] < np.inf:
            break
        # The path back from the sink alternates row, founder, row, ... ; each (founder, next row) pair matches.
        v = prev[sink]
        while v != source:
            r = v - 1 - f_count
            f = prev[v] - 1
            match_f[f] = r
            match_r[r] = f
            v = prev[prev[v]]
        for v in range(n):
            if dist[v] < np.inf:
                height[v] += dist[v]
        total = 0.0
        for f in range(f_count):
            if match_f[f] >= 0:
                total += cost[f, match_f[f]]
        filled = step + 1
        totals[filled] = total
        assignments[filled] = match_f
    return totals[: filled + 1], assignments[: filled + 1]


@kernel(overloads=((_F64_C2,),), cache=True, nogil=True)
def _matching_sensitivities(cost):
    """_matching_costs plus, after every augmentation m, the exact one-row sensitivities of the size-m optimum.
    The size-m optimal row sets of a bipartite assignment form a valuated
    matroid layer, so the optimum without row j is the size-m optimum with j exchanged for one row, and the optimum
    with an extra row b is the size-m optimum with one row exchanged for b. In the residual network of the size-m
    optimum (valid Johnson potentials ``height``) these exchanges are cycles through the sink:
      drop_delta[m, j] = dist(j -> sink) (0 when j is unmatched): g_{R-j}(m) = g(m) + drop_delta[m, j];
      add_rho[m, f] = min over matched rows x of dist(x -> f): g_{R+b}(m) = g(m) + min(0, min_f add_rho[m, f] + c_fb);
      add_sigma[f] = dist(source -> f) after the last augmentation (the size min(F, K) + 1 optimum of R + b).
    Three dense Dijkstras per augmentation cost O(min(F, K) (F + K)^2) in all, the cost of one matching.
    Augmentations follow _matching_costs (same relaxations, same order), so the cardinality costs agree."""
    f_count, k = cost.shape
    n = f_count + k + 2
    source, sink = 0, n - 1
    steps = min(f_count, k)
    totals = np.zeros(steps + 1)
    drop_delta = np.zeros((steps + 1, k))
    add_rho = np.full((steps + 1, f_count), np.inf)
    add_sigma = np.full(f_count, np.inf)
    match_f = np.full(f_count, -1, dtype=np.int64)
    match_r = np.full(k, -1, dtype=np.int64)
    height = np.zeros(n)
    filled = 0
    key = np.empty(n)
    done = np.zeros(n, dtype=np.bool_)
    for step in range(steps):
        dist = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=np.int64)
        settled = np.zeros(n, dtype=np.bool_)
        dist[source] = 0.0
        while True:
            u = -1
            best = np.inf
            for v in range(n):
                if not settled[v] and dist[v] < best:
                    best = dist[v]
                    u = v
            if u < 0:
                break
            settled[u] = True
            if u == source:
                for f in range(f_count):
                    if match_f[f] < 0:
                        value = dist[u] + height[u] - height[1 + f]
                        if value < dist[1 + f] - MATCHING_SLACK:
                            dist[1 + f] = value
                            prev[1 + f] = u
            elif u <= f_count:
                f = u - 1
                for r in range(k):
                    if match_f[f] != r:
                        v = 1 + f_count + r
                        value = dist[u] + cost[f, r] + height[u] - height[v]
                        if value < dist[v] - MATCHING_SLACK:
                            dist[v] = value
                            prev[v] = u
            elif u < sink:
                r = u - 1 - f_count
                if match_r[r] >= 0:
                    g = match_r[r]
                    v = 1 + g
                    value = dist[u] - cost[g, r] + height[u] - height[v]
                else:
                    v = sink
                    value = dist[u] + height[u] - height[v]
                if value < dist[v] - MATCHING_SLACK:
                    dist[v] = value
                    prev[v] = u
        if not dist[sink] < np.inf:
            break
        v = prev[sink]
        while v != source:
            r = v - 1 - f_count
            f = prev[v] - 1
            match_f[f] = r
            match_r[r] = f
            v = prev[prev[v]]
        for v in range(n):
            if dist[v] < np.inf:
                height[v] += dist[v]
        total = 0.0
        for f in range(f_count):
            if match_f[f] >= 0:
                total += cost[f, match_f[f]]
        filled = step + 1
        totals[filled] = total
        # Drop sensitivities: reverse Dijkstra toward the sink over the residual arcs (reduced costs >= 0):
        # r -> sink (r unmatched), f -> r ((f, r) unmatched), r -> f ((f, r) matched), source -> f (f unmatched),
        # f -> source (f matched). Arcs out of the sink are not used.
        for v in range(n):
            key[v] = np.inf
            done[v] = False
        key[sink] = 0.0
        while True:
            u = -1
            best = np.inf
            for v in range(n):
                if not done[v] and key[v] < best:
                    best = key[v]
                    u = v
            if u < 0:
                break
            done[u] = True
            if u == sink:
                for r in range(k):
                    if match_r[r] < 0:
                        w = 1 + f_count + r
                        value = key[u] + height[w] - height[u]
                        if not done[w] and value < key[w]:
                            key[w] = value
            elif u > f_count:
                r = u - 1 - f_count
                for f in range(f_count):
                    if match_f[f] != r:
                        w = 1 + f
                        value = key[u] + cost[f, r] + height[w] - height[u]
                        if not done[w] and value < key[w]:
                            key[w] = value
            elif u >= 1:
                f = u - 1
                if match_f[f] >= 0:
                    w = 1 + f_count + match_f[f]
                    value = key[u] - cost[f, match_f[f]] + height[w] - height[u]
                else:
                    w = source
                    value = key[u] + height[w] - height[u]
                if not done[w] and value < key[w]:
                    key[w] = value
            else:
                for f in range(f_count):
                    if match_f[f] >= 0:
                        w = 1 + f
                        value = key[u] + height[w] - height[u]
                        if not done[w] and value < key[w]:
                            key[w] = value
        for r in range(k):
            w = 1 + f_count + r
            if match_r[r] < 0:
                drop_delta[filled, r] = 0.0
            elif key[w] < np.inf:
                drop_delta[filled, r] = key[w] - height[w] + height[sink]
            else:
                drop_delta[filled, r] = np.inf
        # Add sensitivities: forward Dijkstra from every matched row x (initial key -height[x]) over the residual
        # arcs other than those of the sink.
        for v in range(n):
            key[v] = np.inf
            done[v] = False
        for r in range(k):
            if match_r[r] >= 0:
                key[1 + f_count + r] = -height[1 + f_count + r]
        while True:
            u = -1
            best = np.inf
            for v in range(n - 1):
                if not done[v] and key[v] < best:
                    best = key[v]
                    u = v
            if u < 0:
                break
            done[u] = True
            if u == source:
                for f in range(f_count):
                    if match_f[f] < 0:
                        w = 1 + f
                        value = key[u] + height[u] - height[w]
                        if not done[w] and value < key[w]:
                            key[w] = value
            elif u <= f_count:
                f = u - 1
                for r in range(k):
                    if match_f[f] != r:
                        w = 1 + f_count + r
                        value = key[u] + cost[f, r] + height[u] - height[w]
                        if not done[w] and value < key[w]:
                            key[w] = value
                if match_f[f] >= 0:
                    value = key[u] + height[u] - height[source]
                    if not done[source] and value < key[source]:
                        key[source] = value
            else:
                r = u - 1 - f_count
                if match_r[r] >= 0:
                    g = match_r[r]
                    w = 1 + g
                    value = key[u] - cost[g, r] + height[u] - height[w]
                    if not done[w] and value < key[w]:
                        key[w] = value
        for f in range(f_count):
            if key[1 + f] < np.inf:
                add_rho[filled, f] = key[1 + f] + height[1 + f]
    if filled == steps:
        # One more augmenting step's distances from the source to every founder (no sink arcs).
        for v in range(n):
            key[v] = np.inf
            done[v] = False
        key[source] = 0.0
        while True:
            u = -1
            best = np.inf
            for v in range(n - 1):
                if not done[v] and key[v] < best:
                    best = key[v]
                    u = v
            if u < 0:
                break
            done[u] = True
            if u == source:
                for f in range(f_count):
                    if match_f[f] < 0:
                        w = 1 + f
                        value = key[u] + height[u] - height[w]
                        if not done[w] and value < key[w]:
                            key[w] = value
            elif u <= f_count:
                f = u - 1
                for r in range(k):
                    if match_f[f] != r:
                        w = 1 + f_count + r
                        value = key[u] + cost[f, r] + height[u] - height[w]
                        if not done[w] and value < key[w]:
                            key[w] = value
            else:
                r = u - 1 - f_count
                if match_r[r] >= 0:
                    g = match_r[r]
                    w = 1 + g
                    value = key[u] - cost[g, r] + height[u] - height[w]
                    if not done[w] and value < key[w]:
                        key[w] = value
        for f in range(f_count):
            if key[1 + f] < np.inf:
                add_sigma[f] = key[1 + f] + height[1 + f] - height[source]
    return (
        totals[: filled + 1],
        drop_delta[: filled + 1],
        add_rho[: filled + 1],
        add_sigma,
    )


_B1 = types.Array(types.boolean, 1, "C", readonly=False)
_B2 = types.Array(types.boolean, 2, "C", readonly=False)
_I1 = types.Array(types.int64, 1, "C", readonly=False)


@kernel(overloads=((_B2, _I1, _I1, types.int64, types.int64, _B1),), cache=True, nogil=True)
def _minimum_edges_saturate(edges, match_row, match_f, i, j, extra):
    """Whether the bipartite graph of every row to its minimum-cost founders (``edges`` [K, F], the incumbent's rows)
    without rows i and j and with one extra row (``extra`` [F], its minimum-cost founders) has a matching covering all
    those rows. Starts from the incumbent's maximum matching (``match_row``/``match_f``, not modified) and augments
    from each uncovered row (Kuhn), O(K F) per augmentation."""
    k, f_count = edges.shape
    rows = np.full(f_count, -1, dtype=np.int64)  # founder -> row (k = the extra row)
    for f in range(f_count):
        r = match_f[f]
        if r >= 0 and r != i and r != j:
            rows[f] = r
    covered = np.zeros(k + 1, dtype=np.bool_)
    for r in range(k):
        if r != i and r != j and match_row[r] >= 0:
            covered[r] = True
    stack = np.empty(k + 1, dtype=np.int64)
    via = np.empty(k + 1, dtype=np.int64)
    for start in range(k + 1):
        if start < k and (start == i or start == j or covered[start]):
            continue
        # Depth-first augmenting search over alternating paths (row -> minimum founder -> its row ...).
        seen = np.zeros(f_count, dtype=np.bool_)
        depth = 0
        stack[0] = start
        via[0] = 0
        found = -1
        while depth >= 0:
            r = stack[depth]
            f = via[depth]
            advanced = False
            while f < f_count:
                is_edge = extra[f] if r == k else edges[r, f]
                if is_edge and not seen[f]:
                    seen[f] = True
                    via[depth] = f + 1
                    owner = rows[f]
                    if owner < 0:
                        found = f
                        break
                    depth += 1
                    stack[depth] = owner
                    via[depth] = 0
                    advanced = True
                    break
                f += 1
            if found >= 0:
                break
            if not advanced:
                depth -= 1
        if found < 0:
            return False
        # Flip the path: stack[t] takes founder via[t] - 1 (whose owner was stack[t + 1]); the last row takes `found`.
        rows[found] = stack[depth]
        for t in range(depth - 1, -1, -1):
            rows[via[t] - 1] = stack[t]
        covered[start] = True
    return True


_F1 = types.Array(types.float64, 1, "C", readonly=False)


@kernel(overloads=((_F64_C2, _F64_C2, _I1, _I1, _I1, _F1),), cache=True, nogil=True)
def _merge_codes_exact(cost, row_cost, first, second, which, dictionary):
    """Exact (free rows, edit code) of the merge panels ``which`` (rows first[m], second[m] of the incumbent's cost
    matrix ``cost`` [F, K] replaced by one row with founder codes row_cost[m]), each by one all-cardinality matching
    (_matching_costs) and the description rule of ContextFounders._description (``dictionary[n]`` = the dictionary
    term of n free rows, 0 for n = 0). Compiled batch of exact merge descriptions."""
    f_count, k = cost.shape
    rows = k - 1
    free = np.zeros(len(which), dtype=np.int64)
    code = np.zeros(len(which))
    panel = np.empty((f_count, rows))
    for t in range(len(which)):
        m = which[t]
        column = 0
        for r in range(k):
            if r != first[m] and r != second[m]:
                for f in range(f_count):
                    panel[f, column] = cost[f, r]
                column += 1
        for f in range(f_count):
            panel[f, column] = row_cost[m, f]
        totals, _ = _matching_costs(panel)
        best = np.inf
        for size in range(len(totals)):
            value = totals[size] + dictionary[rows - size]
            if value < best:
                best = value
        band = best + DESCRIPTION_TIE * max(1.0, abs(best))
        chosen = 0
        for size in range(len(totals)):
            if totals[size] + dictionary[rows - size] <= band:
                chosen = size
        free[t] = rows - chosen
        code[t] = totals[chosen]
    return free, code


@kernel(overloads=((_B2, _I1, _I1, _F1, _F64_C2, _I1, _I1, _F1, types.int64),), cache=True, nogil=True)
def _merge_certificates(edges, match_row, match_f, minimum, row_cost, first, second, dictionary, top):
    """For every merge m: the independent-minima certificate (``certified[m]``) and then its exact (free, code), else
    the sorted-minima lower bound of its description length (``lower[m]``). Compiled batch of
    the independent-minima certificate and the sorted row-minima lower bound."""
    k, f_count = edges.shape
    count = len(first)
    certified = np.zeros(count, dtype=np.bool_)
    free = np.zeros(count, dtype=np.int64)
    code = np.zeros(count)
    lower = np.zeros(count)
    rows = k - 1
    values = np.empty(k - 1)
    extra = np.empty(f_count, dtype=np.bool_)
    for m in range(count):
        i, j = first[m], second[m]
        smallest = np.inf
        for f in range(f_count):
            smallest = min(smallest, row_cost[m, f])
        for f in range(f_count):
            extra[f] = row_cost[m, f] == smallest
        n = 0
        for r in range(k):
            if r != i and r != j:
                values[n] = minimum[r]
                n += 1
        values[n] = smallest
        values.sort()
        best = np.inf
        running = 0.0
        sums = np.empty(top + 1)
        sums[0] = 0.0
        for size in range(1, top + 1):
            running += values[size - 1]
            sums[size] = running
        for size in range(top + 1):
            best = min(best, sums[size] + dictionary[rows - size])
        lower[m] = best
        if rows <= f_count and _minimum_edges_saturate(edges, match_row, match_f, i, j, extra):
            certified[m] = True
            band = best + DESCRIPTION_TIE * max(1.0, abs(best))
            chosen = 0
            for size in range(top + 1):
                if sums[size] + dictionary[rows - size] <= band:
                    chosen = size
            free[m] = rows - chosen
            code[m] = sums[chosen]
    return certified, free, code, lower


def _shortest_description(codes, rows: int, length: int) -> tuple[int, float]:
    """(free rows, edit code) of the shortest description from the minimum edit codes ``codes[m]`` of every
    claimed-row count m = 0..len(codes) - 1 of a ``rows``-row panel: the rule of ContextFounders._description
    (descriptions within DESCRIPTION_TIE relative of the shortest prefer more claims)."""
    totals = [
        code + (float(log_haplotype_set_count(length, rows - m)) if rows - m else 0.0) for m, code in enumerate(codes)
    ]
    best = min(totals)
    m = max(size for size, total in enumerate(totals) if total <= best + DESCRIPTION_TIE * max(1.0, abs(best)))
    return rows - m, float(codes[m])


class ContextScreen:
    """One search round's incumbent panel and context: exact shortest descriptions of every one-row deletion and
    one-row addition of the panel from a single matching pass.

    ``drop(j)`` and ``add(row)`` return (free rows, edit code), the values ContextFounders.claims gives for the panel
    without row j or with ``row`` appended, in O(K) and O(F K) after the O(min(F, K)(F + K)^2) construction.
    Values agree with a fresh matching up to rounding of the summation order (exchange costs use potentials).
    ``merge_batch`` gives the merge screen's certificates, optimistic lower bounds and exact codes, and
    ``pair_drop_minima_lower`` the lower bound of a merge whose consensus equals a third row."""

    def __init__(self, panel: NDArray[np.int64], context: ContextFounders) -> None:
        self.context = context
        self.panel = np.asarray(panel, dtype=np.int64)
        self.rows, self.length = self.panel.shape
        distance = context._columns(self.panel)
        f_count = len(context)
        cost = np.empty((f_count, self.rows))
        for f in range(f_count):
            cost[f] = _edit_codes(int(context.n_called[f]))[np.asarray(distance[f], dtype=np.int64)]
        self.cost = cost
        (
            self.totals,
            self.drop_delta,
            self.add_rho,
            self.add_sigma,
        ) = _matching_sensitivities(np.ascontiguousarray(cost))
        self.steps = len(self.totals) - 1

    def row_costs(self, row: NDArray[np.int64]) -> NDArray[np.float64]:
        """Edit codes of every context founder claiming ``row`` (float64[F]): the same table entries
        (_edit_codes(n_f)[d]) as the matching's cost matrix, gathered from one padded table."""
        distance = self.context._columns(np.asarray(row, dtype=np.int64)[None, :])[:, 0]
        if not hasattr(self, "_code_table"):
            n_called = np.asarray(self.context.n_called, dtype=np.int64)
            table = np.zeros((len(n_called), int(n_called.max(initial=0)) + 1))
            for f, n in enumerate(n_called):
                table[f, : n + 1] = _edit_codes(int(n))
            self._code_table = table
        return self._code_table[np.arange(len(distance)), np.asarray(distance, dtype=np.int64)]

    def drop(self, j: int) -> tuple[int, float]:
        rows = self.rows - 1
        top = min(len(self.context), rows)
        codes = [float(self.totals[m] + self.drop_delta[m, j]) for m in range(top + 1)]
        return _shortest_description(codes, rows, self.length)

    def _minimum_structure(self):
        """Each incumbent row's minimum edit code and minimum-cost founders, and a maximum matching of that graph."""
        if not hasattr(self, "_row_minimum"):
            cost = self.cost
            self._row_minimum = cost.min(axis=0) if len(cost) else np.zeros(self.rows)
            self._minimum_edges = np.ascontiguousarray((cost == self._row_minimum[None, :]).T)
            match_row = np.full(self.rows, -1, dtype=np.int64)
            match_f = np.full(len(cost), -1, dtype=np.int64)
            for r in range(self.rows):  # Kuhn on the incumbent once per round, O(K^2 F)
                self._augment(r, match_row, match_f)
            self._match_row, self._match_f = match_row, match_f
        return self._row_minimum, self._minimum_edges

    def _augment(self, start, match_row, match_f):
        seen = np.zeros(len(match_f), dtype=bool)

        def visit(r):
            for f in np.flatnonzero(self._minimum_edges[r]):
                if not seen[f]:
                    seen[f] = True
                    if match_f[f] < 0 or visit(int(match_f[f])):
                        match_f[f] = r
                        match_row[r] = f
                        return True
            return False

        visit(start)

    def pair_drop_minima_lower(self, i: int, j: int) -> float:
        """A lower bound of the shortest description length (edit code plus dictionary term) of a merge whose
        consensus equals a third row (the panel loses rows i and j and gains none: K - 2 rows): any m claimed rows
        cost at least the sum of the m smallest row minima. O(K log K)."""
        minimum, _ = self._minimum_structure()
        keep = np.ones(self.rows, dtype=bool)
        keep[[i, j]] = False
        values = np.sort(minimum[keep])
        rows = self.rows - 2
        top = min(len(self.context), rows)
        sums = np.concatenate([[0.0], np.cumsum(values[:top])])
        return min(
            float(sums[m]) + (float(log_haplotype_set_count(self.length, rows - m)) if rows - m else 0.0)
            for m in range(top + 1)
        )

    def merge_batch(self, rows_new: NDArray[np.int64], first: NDArray[np.int64], second: NDArray[np.int64]):
        """Compiled batch over merges (rows first[m], second[m] replaced by rows_new[m]): founder codes of the new rows
        (one vectorised distance pass), the certificate results and lower bounds, and a function solving the exact
        codes of chosen merges. Returns (row_cost [M, F], certified, free, code, lower, solve)."""
        context = self.context
        new = np.asarray(rows_new, dtype=np.int64)
        distance = ((context.founders[None, :, :] != new[:, None, :]) & context.called[None, :, :]).sum(axis=2)
        if not hasattr(self, "_code_table"):
            self.row_costs(self.panel[0])
        row_cost = np.ascontiguousarray(self._code_table[np.arange(len(context))[None, :], distance])
        rows = self.rows - 1
        top = min(len(context), rows)
        dictionary = np.asarray(
            [0.0] + [float(log_haplotype_set_count(self.length, n)) for n in range(1, rows + 1)], dtype=np.float64
        )
        minimum, edges = self._minimum_structure()
        first = np.ascontiguousarray(first, dtype=np.int64)
        second = np.ascontiguousarray(second, dtype=np.int64)
        certified, free, code, lower = _merge_certificates(
            edges,
            self._match_row,
            self._match_f,
            np.ascontiguousarray(minimum, dtype=np.float64),
            row_cost,
            first,
            second,
            dictionary,
            top,
        )

        def solve(which):
            return _merge_codes_exact(
                np.ascontiguousarray(self.cost),
                row_cost,
                first,
                second,
                np.ascontiguousarray(which, dtype=np.int64),
                dictionary,
            )

        return row_cost, certified, free, code, lower, solve

    def add(self, row: NDArray[np.int64]) -> tuple[int, float]:
        rows = self.rows + 1
        top = min(len(self.context), rows)
        cost = self.row_costs(row)
        codes = []
        for m in range(top + 1):
            if m <= self.steps:
                exchange = float(np.min(self.add_rho[m] + cost)) if len(cost) else np.inf
                codes.append(float(self.totals[m] + min(0.0, exchange)))
            else:
                codes.append(float(self.totals[self.steps] + np.min(self.add_sigma + cost)))
        return _shortest_description(codes, rows, self.length)


def coded_regularizer(
    panel: NDArray[np.int64], frequencies: NDArray[np.float64], context: ContextFounders | None
) -> float:
    """The regularizer with context coding (every row free without a context)."""
    if context is None:
        return regularizer(panel, frequencies)
    _, free, code = context.claims(panel)
    return regularizer(panel, frequencies, free, code)


def selection_objective(fit: PathFit, context: ContextFounders | None) -> float:
    """A fit's selection objective: its own objective without a context, else its log
    likelihood plus the context-coded regularizer."""
    if context is None:
        return fit.objective
    return fit.log_likelihood + coded_regularizer(fit.panel, fit.frequencies, context)
