"""The local selection objective: fitted log likelihood plus a coding regularizer.

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
from numba import njit

from ..discovery.panel import log_haplotype_set_count



def regularizer(panel, frequencies, free=None, claimed_code=0.0) -> float:
    k, length = panel.shape
    free = k if free is None else int(free)
    dictionary = float(log_haplotype_set_count(length, free)) if free else 0.
    value = -dictionary - float(claimed_code) - math.log(k) - math.log(k + 1)
    # Shrinkage centred at uniform; its MAP extra counts total one.
    value += float(np.log(k * frequencies).sum() / k)
    return value


class ContextFounders:
    """The assembled context's founders in one block, for the dictionary code.

    ``founders`` holds one local row per assembled founder of the covering component (this round's
    context proposals repeated by their multiplicities; -1 = no call): founders whose local rows are
    equal are separate claimers (release 2.12). Founders without a called site claim no row.

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

    History: releases 2.1-2.11 let each founder claim only its nearest rows, maximised the number of
    claims before their cost, coded a claim by the shell code log C(n_f, d), and merged founders with
    equal local rows into one claimer; 2.0 let a founder equidistant to several rows claim the lowest
    of them in the fit's row order.
    """

    def __init__(self, founders):
        founders = np.asarray(founders, dtype=np.int64)
        called = founders >= 0
        informative = called.any(axis=1)
        self.founders = np.ascontiguousarray(founders[informative])
        self.called = np.ascontiguousarray(called[informative])
        # Index of each kept founder in the ``founders`` argument (uninformative founders are dropped).
        self.source = np.flatnonzero(informative)
        self.n_called = self.called.sum(axis=1)
        self.distances: dict[bytes, np.ndarray] = {}

    def __len__(self):
        return len(self.founders)

    def _columns(self, panel):
        columns = []
        for row in np.asarray(panel, dtype=np.int64):
            key = row.tobytes()
            value = self.distances.get(key)
            if value is None:
                value = self.distances[key] = ((self.founders != row[None, :]) & self.called).sum(axis=1)
            columns.append(value)
        return np.stack(columns, axis=1)

    def claims(self, panel):
        """(claimed mask, free-row count, claimed rows' edit code in nats)."""
        k, length = np.shape(panel)
        claimed = np.zeros(k, dtype=bool)
        if not len(self.founders):
            return claimed, k, 0.
        return self._best_assignment(self._columns(panel), k, length)

    def matched(self, panel):
        """(founder, row) pairs of one shortest description of the panel, chosen in canonical order (founders by
        content and called mask, rows by content) and mapped back, so that equal-cost choices do not depend on the
        order of the founders or of the rows (founders with equal rows and masks are interchangeable)."""
        panel = np.asarray(panel, dtype=np.int64)
        k, length = panel.shape
        if not len(self.founders):
            return []
        founder_order = sorted(range(len(self.founders)),
                               key=lambda f: (self.founders[f].tobytes(), self.called[f].tobytes()))
        row_order = sorted(range(k), key=lambda r: panel[r].tobytes())
        distance = self._columns(panel)[np.ix_(founder_order, row_order)]
        assignment = self._description(distance, k, length, self.n_called[founder_order])[3]
        return sorted((founder_order[f], row_order[int(r)]) for f, r in enumerate(assignment) if r >= 0)

    def _best_assignment(self, distance, k, length):
        return self._description(distance, k, length)[:3]

    def _description(self, distance, k, length, n_called=None):
        """(claimed mask, free-row count, edit code) of the panel's shortest description (release 2.12):
        any founder may claim any row at the monotone edit-ball code of their distance, and the number
        of claimed rows m is chosen to minimise the summed edit codes plus the free rows' dictionary
        term. Minimum edit costs of every matching size come from one successive-shortest-augmenting-path pass
        (O(min(F, K) (F + K)^2)). For a fixed set of claimers and called masks, one changed call moves the code by
        at most its edit-ball increment, never by a whole free row; founders with equal local rows stay separate
        claimers (release 2.12), so a call making two founders' rows equal keeps that bound."""
        f_count = len(self.founders)
        n_called = self.n_called if n_called is None else n_called
        cost = np.array([[edit_ball_code(int(n_called[f]), int(distance[f, r])) for r in range(k)]
                         for f in range(f_count)], dtype=np.float64)
        sizes = matching_costs(cost)
        totals = [code + (float(log_haplotype_set_count(length, k - m)) if k - m else 0.)
                  for m, (code, _) in enumerate(sizes)]
        # The shortest description; descriptions equal within rounding (1e-9 relative) prefer more claims.
        best = min(totals)
        m = max(size for size, total in enumerate(totals) if total <= best + 1e-9 * max(1., abs(best)))
        code, assignment = sizes[m]
        claimed = np.zeros(k, dtype=bool)
        claimed[[r for r in assignment if r >= 0]] = True
        return claimed, k - m, float(code), assignment


    def free(self, panel) -> int:
        return self.claims(panel)[1]


@functools.lru_cache(maxsize=None)
def edit_ball_code(n: int, d: int) -> float:
    """log sum_{j<=d} C(n, j) (nats): the code length of a row within Hamming distance d of a founder's n called sites.
    Increasing in d (unlike log C(n, d), which falls again beyond n/2); 0 at d = 0."""
    if d <= 0:
        return 0.
    if d >= n:
        return n * math.log(2.)          # the whole space: the exact endpoint
    terms = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) for j in range(min(d, n) + 1)]
    top = max(terms)
    # Never above the whole space's exact code (rounding of the sum near d = n - 1).
    return min(top + math.log(math.fsum(math.exp(x - top) for x in terms)), n * math.log(2.))


def matching_costs(cost):
    """[(minimum total cost, row of each founder or -1)] of bipartite matchings of every size m = 0..min(F, K)
    between founders (rows of ``cost``, non-negative) and panel rows (columns): successive shortest augmenting paths
    from a source through founders and rows to a sink, Dijkstra on reduced costs with Johnson potentials (dense,
    O((F + K)^2) per augmentation, O(K^3) in all). Lowest node index wins equal distances (deterministic)."""
    cost = np.ascontiguousarray(cost, dtype=np.float64)
    totals, assignments = _matching_costs(cost)
    return [(float(totals[m]), assignments[m].copy()) for m in range(len(totals))]


@njit(cache=True)
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
        dist[source] = 0.
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
                        if value < dist[1 + f] - 1e-12:
                            dist[1 + f] = value
                            prev[1 + f] = u
            elif u <= f_count:
                f = u - 1
                for r in range(k):
                    if match_f[f] != r:
                        v = 1 + f_count + r
                        value = dist[u] + cost[f, r] + height[u] - height[v]
                        if value < dist[v] - 1e-12:
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
                if value < dist[v] - 1e-12:
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
        total = 0.
        for f in range(f_count):
            if match_f[f] >= 0:
                total += cost[f, match_f[f]]
        filled = step + 1
        totals[filled] = total
        assignments[filled] = match_f
    return totals[:filled + 1], assignments[:filled + 1]


def coded_regularizer(panel, frequencies, context: ContextFounders | None) -> float:
    """The regularizer with context coding (every row free without a context)."""
    if context is None:
        return regularizer(panel, frequencies)
    _, free, code = context.claims(panel)
    return regularizer(panel, frequencies, free, code)


def selection_objective(fit, context: ContextFounders | None) -> float:
    """A fit's selection objective: its own objective without a context, else its log
    likelihood plus the context-coded regularizer."""
    if context is None:
        return fit.objective
    return fit.log_likelihood + coded_regularizer(fit.panel, fit.frequencies, context)
