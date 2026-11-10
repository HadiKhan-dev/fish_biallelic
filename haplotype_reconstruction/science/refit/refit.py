"""Conditional founder-allele refit kernel.

Given the painting, the allele of founder row k at site j maximises the robust
genotype log likelihood of the samples painted as carrying k at j, with each
sample's other painted homologue held at its current allele. Each site is
updated by coordinate ascent over single-row flips and two-row moves (a fixed
number of sweeps). The minimum gain is charged per changed cell (a hysteresis
rule, not a fixed prior on the assembled allele): a move is accepted only when
its gain reaches the minimum gain for every cell it changes (one for
a flip, two for a two-row move). A swap exchanges the different alleles of two rows
painted together at the site: it keeps the genotype of every sample painted
with that pair, so a single flip of either row loses likelihood on those
samples and only the swap corrects an allele carried by the wrong one of the
two rows. A pair flip changes the equal alleles of two rows painted
together: where both carry the wrong allele, their shared carriers' reads favour
neither single flip (under the robust error floor a heterozygote fits a homozygote's
reads little better than the wrong homozygote), and only the joint change reaches the
genotype the reads show. Unknown alleles are filled only by the fill rule below.

Move order: each sweep has a single-row flip phase and a two-row phase (swaps and pair flips). Moves are
scored against the state at the start of a round. Qualifying moves are applied in decision order (gain tie
classes using REFIT_TIE_QUANTUM and core.ties; content order inside a class, the canonical row indices) only
while they do not interact with an applied move. Interactions share a row or a co-painted partner whose score
changes; those moves are re-scored next round. A row changes at most once per phase and sweep. A sweep that
changes nothing ends the site. Scoring against a common start state prevents evaluation order from deciding
between interacting moves.

Ambiguous cells: two moves of one tie class that interact and exclude each other (after either move the other
no longer reaches its minimum gain) are alternatives the data do not separate. Typically two rows painted only
together could each carry the observed allele. Their cells are released unknown (-1, exported as missing)
instead of being decided by row order; the round is re-scored without them.

Filled cells: after the sweeps, an allele the painted panel left unknown is called when one
allele's robust log likelihood over the row's scoring carriers (partners with a called allele, or the row
itself) beats the other by the minimum gain and at least ``fill_min_supporters`` of those carriers favour
it on their own reads (by more than the tie quantum): the minimum of two favourable samples that local
selection and discovery also require, not their whole release rule.
Every fill of a site is decided against the post-sweep state and the fills are applied together, so no
fill depends on another; carriers whose partner allele is unknown give no evidence, as everywhere here.
Cells the tie rule released unknown in this pass are not filled.
"""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import kernel
from .parameters import (
    REFIT_TIE_QUANTUM,
    REFIT_TIE_QUANTUM_PER_CARRIER,
    SITE_TILE,
    STABLE_SORT_COMPARISON_LIMIT,
    WIDE_QUANTA,
)

# A move gain sums its carriers' log terms. Relative GL noise of 1e-6 moves one carrier's two terms by at most
# about 2e-6, so two gains with n carriers between them tie within max(REFIT_TIE_QUANTUM, n
# REFIT_TIE_QUANTUM_PER_CARRIER). Both quanta lie far below the 5-nat minimum gain. Gains within the band use
# content order. REFIT_TIE_QUANTUM also sets the per-carrier support margin for a fill; classes wider than
# WIDE_QUANTA keep value order.


# Supported layouts for typed stage routes.
_OVERLOADS = {
    "_stable_argsort": (
        (types.Array(types.float64, 1, "C", readonly=False, aligned=True), types.literal(0)),
        (types.Array(types.float64, 1, "C", readonly=False, aligned=True), types.int64),
    ),
    "_tie_components": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.literal(0),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_decision_order": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.literal(0),
            types.boolean,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.boolean,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_flip_gain": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "A", readonly=False, aligned=True),
            types.int64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
    "_painted_together": (
        (
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
        ),
    ),
    "_flip_conflicts": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.literal(0),
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_apply_two_row": (
        (
            types.Array(types.int8, 1, "A", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
        ),
        (
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
        ),
    ),
    "_swap_pair_conflict": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "A", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_swap_conflicts": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.literal(0),
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_carriers": (
        (
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "A", readonly=False, aligned=True),
            types.int64,
        ),
        (
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
    "_touch": (
        (
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
    "_two_row_gain": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "A", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
        ),
    ),
    "_swap_gain": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "A", readonly=False, aligned=True),
            types.int64,
            types.int64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
        ),
    ),
    "_pair_curvature": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "A", readonly=False, aligned=True),
            types.int64,
            types.int64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
        ),
    ),
    "_flip_conflicts_fast": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.literal(0),
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_swap_conflicts_fast": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.literal(0),
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int8, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_refit_kernel": (
        (
            types.Array(types.int8, 2, "C", readonly=False, aligned=True),
            types.Array(types.int16, 2, "C", readonly=False, aligned=True),
            types.Array(types.int16, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.int64,
            types.boolean,
            types.int64,
        ),
        (
            types.Array(types.int8, 2, "C", readonly=True, aligned=True),
            types.Array(types.int16, 2, "C", readonly=False, aligned=True),
            types.Array(types.int16, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.int64,
            types.boolean,
            types.int64,
        ),
    ),
}


@kernel(overloads=_OVERLOADS["_stable_argsort"], cache=True, nogil=True)
def _stable_argsort(keys, n):
    """Indices 0..n-1 ordered by keys[:n] ascending, equal keys in index order: the order of
    np.argsort(keys[:n], kind="mergesort") for non-NaN keys (-0.0 is ordered as +0.0), by an 8-pass LSD radix sort
    of the order-preserving unsigned images of the IEEE bit patterns, O(n). Ordering adds no log factor to
    the sweep. At most STABLE_SORT_COMPARISON_LIMIT keys use comparison sorting (every round at K <= 30,
    with at most K + K(K-1)/2 candidates); its log factor is then at most 12, a constant;
    the bounded comparison sort is faster than eight radix passes at this size."""
    if n <= STABLE_SORT_COMPARISON_LIMIT:
        # Bounded key count: the comparison sort's log factor is a constant here and it beats eight radix passes.
        small = np.empty(n, dtype=np.float64)
        for i in range(n):
            small[i] = keys[i] + 0.0
        return np.argsort(small, kind="mergesort")
    bits = np.empty(n, dtype=np.uint64)
    scratch = np.empty(1, dtype=np.float64)
    image = scratch.view(np.uint64)
    top = np.uint64(1) << np.uint64(63)
    for i in range(n):
        scratch[0] = keys[i] + 0.0
        u = image[0]
        bits[i] = ~u if u & top else u | top
    order = np.arange(n)
    other = np.empty(n, dtype=np.int64)
    count = np.empty(257, dtype=np.int64)
    mask = np.uint64(255)
    for shift in range(0, 64, 8):
        step = np.uint64(shift)
        count[:] = 0
        for t in range(n):
            count[np.int64((bits[order[t]] >> step) & mask) + 1] += 1
        for b in range(256):
            count[b + 1] += count[b]
        for t in range(n):
            index = order[t]
            byte = np.int64((bits[index] >> step) & mask)
            other[count[byte]] = index
            count[byte] += 1
        order, other = other, order
    return order


@kernel(overloads=_OVERLOADS["_tie_components"], cache=True, nogil=True)
def _tie_components(values, carriers, n, component):
    """Tie classes of n candidate gains: the connected components of the relation |g_i - g_j| <=
    max(REFIT_TIE_QUANTUM, (n_i + n_j) REFIT_TIE_QUANTUM_PER_CARRIER), which is the union of two interval
    relations: consecutive gains (in value order) within REFIT_TIE_QUANTUM, and intervals of radius
    n_i REFIT_TIE_QUANTUM_PER_CARRIER around each gain that overlap. Their edges (consecutive value-order pairs
    within the quantum; each candidate joined to the sweep frontier of the lower-end order it overlaps) are at most
    2 (n - 1); a depth-first labelling gives the components in O(n) after the two O(n) sorts. Neither the convention nor the
    candidates' listing order enters. component[i] receives the smallest index of i's component."""
    by_value = _stable_argsort(values, n)
    lower = np.empty(n, dtype=np.float64)
    for i in range(n):
        lower[i] = values[i] - carriers[i] * REFIT_TIE_QUANTUM_PER_CARRIER
    by_lower = _stable_argsort(lower, n)
    head = np.full(n, -1, dtype=np.int64)
    link = np.empty(4 * n, dtype=np.int64)
    target = np.empty(4 * n, dtype=np.int64)
    edges = 0
    for t in range(n - 1):
        i, j = by_value[t], by_value[t + 1]
        if values[j] - values[i] <= REFIT_TIE_QUANTUM:
            target[edges] = j
            link[edges] = head[i]
            head[i] = edges
            target[edges + 1] = i
            link[edges + 1] = head[j]
            head[j] = edges + 1
            edges += 2
    if n:
        frontier = by_lower[0]
        for t in range(1, n):
            i = by_lower[t]
            if lower[i] <= values[frontier] + carriers[frontier] * REFIT_TIE_QUANTUM_PER_CARRIER:
                target[edges] = i
                link[edges] = head[frontier]
                head[frontier] = edges
                target[edges + 1] = frontier
                link[edges + 1] = head[i]
                head[i] = edges + 1
                edges += 2
                if (
                    values[i] + carriers[i] * REFIT_TIE_QUANTUM_PER_CARRIER
                    > values[frontier] + carriers[frontier] * REFIT_TIE_QUANTUM_PER_CARRIER
                ):
                    frontier = i
            else:
                frontier = i
    for i in range(n):
        component[i] = -1
    stack = np.empty(n, dtype=np.int64)
    for root in range(n):
        if component[root] >= 0:
            continue
        component[root] = root
        stack[0] = root
        top = 1
        while top > 0:
            top -= 1
            x = stack[top]
            e = head[x]
            while e >= 0:
                y = target[e]
                if component[y] < 0:
                    component[y] = root
                    stack[top] = y
                    top += 1
                e = link[e]


@kernel(overloads=_OVERLOADS["_decision_order"], cache=True, nogil=True)
def _decision_order(values, carriers, n, reverse, classes):
    """Decision order of n candidate moves with gains values[:n] summing carriers[:n] carriers each (candidates
    listed in content order): exact gain descending; tie classes are the components of _tie_components (two gains
    within max(REFIT_TIE_QUANTUM, (n_i + n_j) REFIT_TIE_QUANTUM_PER_CARRIER), chained), consecutive in value
    order; content (list) order inside a class; a class wider than WIDE_QUANTA x its largest pair quantum keeps
    value order (list order among equal gains). Under the convention reversal (reverse) the list order is
    reversed; class membership does not depend on it. classes[i] receives the class of candidate i (equal values
    = same tie class). O(n) using radix order and counting placement."""
    keys = np.empty(n, dtype=np.float64)
    for i in range(n):
        keys[i] = -values[n - 1 - i] if reverse else -values[i]
    order = _stable_argsort(keys, n)
    if reverse:
        for t in range(n):
            order[t] = n - 1 - order[t]
    component = np.empty(n, dtype=np.int64)
    _tie_components(values, carriers, n, component)
    segment_start = np.empty(n, dtype=np.int64)
    segment_end = np.empty(n, dtype=np.int64)
    in_content_order = np.zeros(n, dtype=np.bool_)
    start = 0
    while start < n:
        # A component covers a contiguous range of gains, so its members are consecutive in the order.
        end = start + 1
        largest = carriers[order[start]]
        while end < n and component[order[end]] == component[order[start]]:
            largest = max(largest, carriers[order[end]])
            end += 1
        quantum = max(REFIT_TIE_QUANTUM, 2 * largest * REFIT_TIE_QUANTUM_PER_CARRIER)
        wide = values[order[start]] - values[order[end - 1]] > WIDE_QUANTA * quantum
        for t in range(start, end):
            # a wide class keeps value order: its members are not tied alternatives (own class ids)
            classes[order[t]] = n + t if wide else start
        for t in range(start, end):
            segment_start[t] = start
            segment_end[t] = end
            in_content_order[t] = end - start > 1 and not wide
        start = end
    # Content (list) order inside each tied class, in one pass over the candidates in list order (a counting
    # placement instead of a sort per class, O(n)).
    position = np.empty(n, dtype=np.int64)
    for t in range(n):
        position[order[t]] = t
    filled = np.zeros(n, dtype=np.int64)
    arranged = order.copy()
    for i in range(n):
        t = position[i]
        if in_content_order[t]:
            first = segment_start[t]
            k = filled[first]
            filled[first] += 1
            if reverse:
                arranged[segment_end[t] - 1 - k] = i
            else:
                arranged[first + k] = i
    return arranged


@kernel(overloads=_OVERLOADS["_flip_gain"], cache=True, nogil=True)
def _flip_gain(table, alleles, row):
    """Log-likelihood change from flipping called row `row` (partners' alleles held)."""
    ll0 = table[row, row, 0]
    ll1 = table[row, row, 2]
    for partner in range(len(alleles)):
        other = alleles[partner]
        if partner == row or other < 0:
            continue
        lo, hi = min(row, partner), max(row, partner)
        ll0 += table[lo, hi, other]
        ll1 += table[lo, hi, other + 1]
    return ll1 - ll0 if alleles[row] == 0 else ll0 - ll1


@kernel(overloads=_OVERLOADS["_painted_together"], cache=True, nogil=True)
def _painted_together(together, a, b):
    """Rows a and b share a painted sample at the site (or are the same row)."""
    return a == b or together[min(a, b), max(a, b)]


@kernel(overloads=_OVERLOADS["_flip_conflicts"], cache=True, nogil=True)
def _flip_conflicts(table, refit, site, together, cand_a, cand_class, order, n, minimum_gain, marked):
    """Marks (in ``marked``) the rows of flip candidates of one tie class that interact and exclude each
    other: after either flip the other no longer reaches the minimum gain, so the row order would decide.
    Only pairs inside a tie class are examined (consecutive in ``order``). Returns whether any row was
    marked."""
    marked[:] = False
    found = False
    column = refit[:, site]
    start = 0
    while start < n:
        end = start + 1
        while end < n and cand_class[order[end]] == cand_class[order[start]]:
            end += 1
        for p in range(start, end):
            for q in range(p + 1, end):
                i, j = order[p], order[q]
                ri, rj = cand_a[i], cand_a[j]
                if not _painted_together(together, ri, rj):
                    continue
                column[ri] = 1 - column[ri]
                gj = _flip_gain(table, column, rj)
                column[ri] = 1 - column[ri]
                column[rj] = 1 - column[rj]
                gi = _flip_gain(table, column, ri)
                column[rj] = 1 - column[rj]
                if not (gj > 0.0 and gj >= minimum_gain and gi > 0.0 and gi >= minimum_gain):
                    marked[ri] = True
                    marked[rj] = True
                    found = True
        start = end
    return found


@kernel(overloads=_OVERLOADS["_apply_two_row"], cache=True, nogil=True)
def _apply_two_row(alleles, a, b, kind):
    """Applies a two-row move to ``alleles`` (swap: exchange; pair flip: both flipped)."""
    if kind == 0:
        alleles[a], alleles[b] = alleles[b], alleles[a]
    else:
        alleles[a] = 1 - alleles[a]
        alleles[b] = 1 - alleles[b]


@kernel(overloads=_OVERLOADS["_swap_pair_conflict"], cache=True, nogil=True)
def _swap_pair_conflict(table, column, together, a1, b1, k1, a2, b2, k2, minimum_gain, marked):
    """One pair of two-row candidates of a tie class (rows a1, b1 with kind k1 and a2, b2 with kind k2; kind 0
    swap, 1 pair flip): when they share a row or a row painted together and either no longer passes after
    the other is applied, marks their four rows. Returns whether rows were marked."""
    if not (
        _painted_together(together, a1, a2)
        or _painted_together(together, a1, b2)
        or _painted_together(together, b1, a2)
        or _painted_together(together, b1, b2)
    ):
        return False
    x1, y1, x2, y2 = column[a1], column[b1], column[a2], column[b2]
    _apply_two_row(column, a1, b1, k1)
    g2 = _two_row_gain(table, column, a2, b2, k2)
    pass_2 = g2 > 0.0 and g2 >= minimum_gain
    column[a1], column[b1], column[a2], column[b2] = x1, y1, x2, y2
    _apply_two_row(column, a2, b2, k2)
    g1 = _two_row_gain(table, column, a1, b1, k1)
    pass_1 = g1 > 0.0 and g1 >= minimum_gain
    column[a1], column[b1], column[a2], column[b2] = x1, y1, x2, y2
    if pass_1 and pass_2:
        return False
    marked[a1] = True
    marked[b1] = True
    marked[a2] = True
    marked[b2] = True
    return True


@kernel(overloads=_OVERLOADS["_swap_conflicts"], cache=True, nogil=True)
def _swap_conflicts(
    table, refit, site, together, cand_a, cand_b, cand_kind, cand_class, order, n, minimum_gain, marked
):
    """As _flip_conflicts for the two-row candidates (swaps and pair flips): two of one tie class that share a
    row or a row painted together, each excluding the other (only pairs inside a tie class are examined)."""
    marked[:] = False
    found = False
    column = refit[:, site]
    start = 0
    while start < n:
        end = start + 1
        while end < n and cand_class[order[end]] == cand_class[order[start]]:
            end += 1
        for p in range(start, end):
            for q in range(p + 1, end):
                i, j = order[p], order[q]
                if _swap_pair_conflict(
                    table,
                    column,
                    together,
                    cand_a[i],
                    cand_b[i],
                    cand_kind[i],
                    cand_a[j],
                    cand_b[j],
                    cand_kind[j],
                    minimum_gain,
                    marked,
                ):
                    found = True
        start = end
    return found


@kernel(overloads=_OVERLOADS["_carriers"], cache=True, nogil=True)
def _carriers(pair_count, alleles, row):
    """Scoring carriers a flip of ``row`` sums: its own pair's samples and those of partners with a called
    allele."""
    total = pair_count[row, row]
    for partner in range(len(alleles)):
        if partner == row or alleles[partner] < 0:
            continue
        total += pair_count[min(row, partner), max(row, partner)]
    return total


@kernel(overloads=_OVERLOADS["_touch"], cache=True, nogil=True)
def _touch(touched, together, row):
    """Marks `row` and every row painted together with it at the site."""
    touched[row] = True
    for partner in range(len(touched)):
        lo, hi = min(row, partner), max(row, partner)
        if together[lo, hi]:
            touched[partner] = True


@kernel(overloads=_OVERLOADS["_two_row_gain"], cache=True, nogil=True)
def _two_row_gain(table, alleles, a, b, kind):
    """Exact gain of a two-row move on ``alleles`` (swap: kind 0, different alleles exchanged; pair flip:
    kind 1, equal alleles both flipped), or -inf when the move does not apply to these alleles."""
    xa, xb = alleles[a], alleles[b]
    if xa < 0 or xb < 0 or (kind == 0) != (xa != xb):
        return -np.inf
    if kind == 0:
        return _swap_gain(table, alleles, min(a, b), max(a, b))
    first = _flip_gain(table, alleles, a)
    alleles[a] = 1 - alleles[a]
    second = _flip_gain(table, alleles, b)
    alleles[a] = 1 - alleles[a]
    return first + second


@kernel(overloads=_OVERLOADS["_swap_gain"], cache=True, nogil=True)
def _swap_gain(table, alleles, a, b):
    """Log-likelihood change from exchanging the different alleles of called rows a < b.

    The samples painted with the pair {a, b} keep their genotype (their table
    entry cancels), so the change is the sum of each row's change on its other
    carriers: row a takes allele x_b and row b takes x_a.
    """
    xa, xb = alleles[a], alleles[b]
    change = (table[a, a, 2 * xb] - table[a, a, 2 * xa]) + (table[b, b, 2 * xa] - table[b, b, 2 * xb])
    for partner in range(len(alleles)):
        other = alleles[partner]
        if partner == a or partner == b or other < 0:
            continue
        lo, hi = min(a, partner), max(a, partner)
        change += table[lo, hi, xb + other] - table[lo, hi, xa + other]
        lo, hi = min(b, partner), max(b, partner)
        change += table[lo, hi, xa + other] - table[lo, hi, xb + other]
    return change


@kernel(overloads=_OVERLOADS["_pair_curvature"], cache=True, nogil=True)
def _pair_curvature(table, alleles, u, v):
    """Delta_uv: the change of row u's flip gain when row v flips (u != v, both called), which equals the change of
    v's flip gain when u flips. With t = table[min(u, v), max(u, v)] and s = x_u + x_v it is the second difference
    t[2 - s] - t[1 + x_u - x_v] - t[1 - x_u + x_v] + t[s] (zero unless u and v are painted together). Hence a
    two-row move (both rows flip) gains flip_gain(u) + flip_gain(v) + Delta_uv, and after a move flipping rows V a
    row's flip gain is its old one plus its Delta with every row of V (both rows must be called)."""
    lo, hi = min(u, v), max(u, v)
    xu = np.int64(alleles[u])
    xv = np.int64(alleles[v])
    total = xu + xv
    return table[lo, hi, 2 - total] - table[lo, hi, 1 + xu - xv] - table[lo, hi, 1 - xu + xv] + table[lo, hi, total]


@kernel(overloads=_OVERLOADS["_flip_conflicts_fast"], cache=True, nogil=True)
def _flip_conflicts_fast(table, refit, site, together, cand_a, cand_gain, cand_class, order, n, minimum_gain, marked):
    """_flip_conflicts with O(1) pair tests: after row ri flips, rj's flip gain is cand_gain[j] + Delta_{ri rj}
    (and symmetrically), so a pair costs O(1) instead of two O(K) gains: the same predicate in mathematically equivalent
    arithmetic for finite tables."""
    marked[:] = False
    found = False
    column = refit[:, site]
    start = 0
    while start < n:
        end = start + 1
        while end < n and cand_class[order[end]] == cand_class[order[start]]:
            end += 1
        for p in range(start, end):
            for q in range(p + 1, end):
                i, j = order[p], order[q]
                ri, rj = cand_a[i], cand_a[j]
                if not _painted_together(together, ri, rj):
                    continue
                d = _pair_curvature(table, column, ri, rj)
                gj = cand_gain[j] + d
                gi = cand_gain[i] + d
                fails = not (gj > 0.0 and gj >= minimum_gain and gi > 0.0 and gi >= minimum_gain)
                if fails:
                    marked[ri] = True
                    marked[rj] = True
                    found = True
        start = end
    return found


@kernel(overloads=_OVERLOADS["_swap_conflicts_fast"], cache=True, nogil=True)
def _swap_conflicts_fast(
    table, refit, site, together, cand_a, cand_b, cand_kind, cand_gain, cand_class, order, n, minimum_gain, marked,
    row_uses
):
    """The rows _swap_conflicts marks, in O(K) per examined candidate instead of O(K^2) for finite tables.

    Inside one tie class: (i) two members sharing exactly one row always exclude each other (a move flips both its
    rows, so after one the other's kind no longer matches its alleles and its gain is -inf): every member with a
    row used by another member is marked, O(c); (ii) a member using no row of another member (at most K/2 per
    class, pairwise row-disjoint) is tested against every other member with the O(1) form: after a move flipping
    rows V the other's gain is its own plus W = sum of Delta_uv over its rows u and v in V (a symmetric quantity),
    so the pair fails when either gain + W misses the minimum. Members whose rows are all marked need no test (a
    pair's marks are the union of both members' rows, and each unmarked member runs its own tests). Tests within
    ``row_uses``: zeroed int64[K] scratch, left zeroed.

    Certified members: let T be the rows of the
    members of tie classes with at least two members (the only rows a test can involve) and m(v) = min(0,
    min_{u in T, u != v} Delta_vu). Every cross term of W(i, j) for two row-disjoint members of one class pairs a
    row of i with a row of j, both in T, so W(i, j) >= max(L_i, L_j) with L_i = 2 (m(a_i) + m(b_i)). A member
    whose gain + L passes is certified (safe); two certified members cannot fail against each other (each gain + W
    >= gain + own L passes), so a certified member is tested only against the uncertified members of its class.
    Exact (a mathematical bound; which classes are examined and the ambiguity rule are unchanged); members sharing a
    row are still marked by (i). Restricting the minimum to T gives a bound at least as tight as minimising
    over all called rows. m costs O(K |T|) <= O(K^2) per call.

    Cost per round: O(K^2 + P) plus the explicit pair scans. A scan between two unshared members involves an
    uncertified unshared member, so these total at most K Uu_r for Uu_r uncertified unshared members in round r
    (u_c <= K/2 unshared members per class). Scans against members sharing a row are at most K/2 per such member;
    the round releases its rows and released rows never return, so these total O(K^3) over the site. A sweep has at
    most K/2 two-row rounds that apply a move and a site at most K/2 that release, so with S sweeps the two-row
    conflict work is O((S + 1) K^3 + K sum_r Uu_r), the sum over every two-row round of the site, applying and
    releasing alike (a chain of releasing rounds re-examines its uncertified members each round). With every member
    certified (for example sustained matching classes, whose interactions are positive) the sum is
    zero and the work is cubic. For fixed S, supercubic work requires the sum to exceed order K^2;
    order K uncertified members per round
    over order K rounds is still cubic. Many classes with strongly negative curvature can leave
    order K^2 unshared members uncertified in each round. At worst O(K^3) per round and O(K^4) per sweep
    (the documented exception). Algebraic decisions equal the direct rule except where a threshold or
    tie comparison sits within rounding of an exact equality (the O(1) gains round differently from the direct
    sums)."""
    marked[:] = False
    found = False
    column = refit[:, site]
    k = column.shape[0]
    tied = np.zeros(k, dtype=np.bool_)
    start = 0
    while start < n:
        end = start + 1
        while end < n and cand_class[order[end]] == cand_class[order[start]]:
            end += 1
        if end - start > 1:
            for p in range(start, end):
                tied[cand_a[order[p]]] = True
                tied[cand_b[order[p]]] = True
        start = end
    lowest = np.zeros(k)
    for v in range(k):
        if not tied[v]:
            continue
        for u in range(k):
            if u != v and tied[u] and _painted_together(together, u, v):
                d = _pair_curvature(table, column, v, u)
                if d < lowest[v]:
                    lowest[v] = d
    safe = np.zeros(n, dtype=np.bool_)
    unsafe = np.empty(n, dtype=np.int64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and cand_class[order[end]] == cand_class[order[start]]:
            end += 1
        if end - start > 1:
            for p in range(start, end):
                row_uses[cand_a[order[p]]] += 1
                row_uses[cand_b[order[p]]] += 1
            for p in range(start, end):
                i = order[p]
                if row_uses[cand_a[i]] > 1 or row_uses[cand_b[i]] > 1:
                    marked[cand_a[i]] = True
                    marked[cand_b[i]] = True
                    found = True
            n_unsafe = 0
            for p in range(start, end):
                i = order[p]
                bound = cand_gain[i] + 2.0 * (lowest[cand_a[i]] + lowest[cand_b[i]])
                safe[p] = bound > 0.0 and bound >= minimum_gain
                if not safe[p]:
                    unsafe[n_unsafe] = p
                    n_unsafe += 1
            for p in range(start, end):
                i = order[p]
                a1, b1 = cand_a[i], cand_b[i]
                if row_uses[a1] > 1 or row_uses[b1] > 1 or (marked[a1] and marked[b1]):
                    continue
                count = n_unsafe if safe[p] else end - start
                for t in range(count):
                    q = unsafe[t] if safe[p] else start + t
                    if q == p:
                        continue
                    j = order[q]
                    a2, b2 = cand_a[j], cand_b[j]
                    if not (
                        _painted_together(together, a1, a2)
                        or _painted_together(together, a1, b2)
                        or _painted_together(together, b1, a2)
                        or _painted_together(together, b1, b2)
                    ):
                        continue
                    w = (
                        _pair_curvature(table, column, a2, a1)
                        + _pair_curvature(table, column, a2, b1)
                        + _pair_curvature(table, column, b2, a1)
                        + _pair_curvature(table, column, b2, b1)
                    )
                    g2 = cand_gain[j] + w
                    g1 = cand_gain[i] + w
                    fails = not (g2 > 0.0 and g2 >= minimum_gain and g1 > 0.0 and g1 >= minimum_gain)
                    if fails:
                        marked[a1] = True
                        marked[b1] = True
                        marked[a2] = True
                        marked[b2] = True
                        found = True
                        break
            for p in range(start, end):
                row_uses[cand_a[order[p]]] = 0
                row_uses[cand_b[order[p]]] = 0
        start = end
    return found



@kernel(overloads=_OVERLOADS["_refit_kernel"], cache=True, parallel=True, nogil=True)
def _refit_kernel(
    calls,
    first,
    second,
    likelihoods,
    observed,
    evidence_sites,
    robustness_epsilon,
    minimum_gain,
    sweeps,
    reverse,
    fill_min_supporters,
):
    """Coordinate ascent per site on a founder-pair log-likelihood table.

    ``table[a, b, g]`` (a <= b) sums log((1 - eps) P(reads | dosage g) + eps / 3)
    over observed samples painted with the unordered pair {a, b}; a row's
    objective then needs only its partners' current alleles. ``favour[a, b]``
    counts those samples whose own reads favour one dosage over the next (a < b:
    1 over 0, 0 over 1, 2 over 1, 1 over 2; a == b: 2 over 0, 0 over 2) by more
    than the tie quantum, the supporters of a fill (module docstring). Pairs whose
    partner allele is unknown contribute nothing, as do unpainted tracks. Each
    sweep tries the single-row flips (accepted at a gain >= minimum_gain),
    then the swaps of two rows with different alleles that are painted
    together at the site (``together[a, b]``: some sample painted {a, b} is
    observed there), accepted at a gain >= 2 x minimum_gain because it changes
    two cells; moves are judged against the round's start state in decision
    order (module docstring). Per site (finite tables): O(N) to fill the table, then per round
    O(K^2) for the K flip gains and the O(1) two-row gains of the P co-painted pairs (P <= min(N, K (K - 1) / 2))
    and O(n) to order n qualifying moves; a round applies a move or releases alleles, so a phase has at most K
    rounds per sweep (and K/2 releasing rounds per site). Conflict tests: flip phase O(K^2) per round; two-row
    phase O(K^2 + P) per round for certified classes plus the tests of uncertified members (_swap_conflicts_fast).
    Total O(N + (S + 1) K^3 + K sum_r Uu_r) per site over S sweeps (Uu_r: uncertified unshared tied members of
    two-row round r; _swap_conflicts_fast), cubic when every tied member is certified, O(N + K^4) per sweep at worst;
    tables with an infinite entry use direct O(K) gains and tests (up to O(K^5) in one round). Fills add O(K^2)
    per site. SITE_TILE-site tiles in parallel.
    """
    founders, sites = calls.shape
    samples = first.shape[0]
    refit = calls.copy()
    gain = np.zeros((founders, sites), dtype=np.float64)
    swap_minimum_gain = 2.0 * minimum_gain
    tiles = (sites + SITE_TILE - 1) // SITE_TILE
    for tile in prange(tiles):
        table = np.zeros((founders, founders, 3), dtype=np.float64)
        together = np.zeros((founders, founders), dtype=np.bool_)
        flipped = np.zeros(founders, dtype=np.bool_)
        swapped = np.zeros(founders, dtype=np.bool_)
        touched = np.zeros(founders, dtype=np.bool_)
        cand_a = np.empty(founders * founders, dtype=np.int64)
        cand_b = np.empty(founders * founders, dtype=np.int64)
        cand_gain = np.empty(founders * founders, dtype=np.float64)
        cand_class = np.empty(founders * founders, dtype=np.int64)
        cand_kind = np.empty(founders * founders, dtype=np.int64)
        cand_n = np.empty(founders * founders, dtype=np.int64)
        row_carriers = np.zeros(founders, dtype=np.int64)
        favour = np.zeros((founders, founders, 4), dtype=np.int32)
        pair_count = np.zeros((founders, founders), dtype=np.int64)
        flip_gain = np.zeros(founders, dtype=np.float64)
        row_uses = np.zeros(founders, dtype=np.int64)
        for site in range(tile * SITE_TILE, min(sites, (tile + 1) * SITE_TILE)):
            table[:] = 0.0
            together[:] = False
            favour[:] = 0
            pair_count[:] = 0
            column = evidence_sites[site]
            carried = False
            for sample in range(samples):
                a, b = first[sample, site], second[sample, site]
                if a < 0 or b < 0 or not observed[sample, column]:
                    continue
                if a > b:
                    a, b = b, a
                l0 = math.log((1.0 - robustness_epsilon) * likelihoods[sample, column, 0] + robustness_epsilon / 3.0)
                l1 = math.log((1.0 - robustness_epsilon) * likelihoods[sample, column, 1] + robustness_epsilon / 3.0)
                l2 = math.log((1.0 - robustness_epsilon) * likelihoods[sample, column, 2] + robustness_epsilon / 3.0)
                table[a, b, 0] += l0
                table[a, b, 1] += l1
                table[a, b, 2] += l2
                if a == b:
                    if l2 - l0 > REFIT_TIE_QUANTUM:
                        favour[a, b, 0] += 1
                    elif l0 - l2 > REFIT_TIE_QUANTUM:
                        favour[a, b, 1] += 1
                else:
                    if l1 - l0 > REFIT_TIE_QUANTUM:
                        favour[a, b, 0] += 1
                    elif l0 - l1 > REFIT_TIE_QUANTUM:
                        favour[a, b, 1] += 1
                    if l2 - l1 > REFIT_TIE_QUANTUM:
                        favour[a, b, 2] += 1
                    elif l1 - l2 > REFIT_TIE_QUANTUM:
                        favour[a, b, 3] += 1
                together[a, b] = True
                pair_count[a, b] += 1
                carried = True
            if not carried:
                continue
            # With zero likelihoods at epsilon=0, the flip-gain identity can evaluate inf - inf
            # even when a swap has finite direct gain because its shared dosage is unchanged.
            finite = True
            for a in range(founders):
                for b in range(a, founders):
                    for g in range(3):
                        if not np.isfinite(table[a, b, g]):
                            finite = False
            for _ in range(sweeps):
                moved = False
                flipped[:] = False
                swapped[:] = False
                # Flip phase: rounds of non-interacting flips judged against the round's start state.
                while True:
                    n = 0
                    for row in range(founders):
                        if flipped[row] or refit[row, site] < 0:
                            continue
                        delta = _flip_gain(table, refit[:, site], row)
                        if delta > 0.0 and delta >= minimum_gain:
                            cand_a[n] = row
                            cand_gain[n] = delta
                            cand_n[n] = _carriers(pair_count, refit[:, site], row)
                            n += 1
                    if n == 0:
                        break
                    order = _decision_order(cand_gain, cand_n, n, reverse, cand_class)
                    if (
                        _flip_conflicts_fast(
                            table, refit, site, together, cand_a, cand_gain, cand_class, order, n, minimum_gain, touched
                        )
                        if finite
                        else _flip_conflicts(
                            table, refit, site, together, cand_a, cand_class, order, n, minimum_gain, touched
                        )
                    ):
                        for row in range(founders):
                            if touched[row]:
                                refit[row, site] = -1
                                gain[row, site] = 0.0
                                moved = True
                        continue
                    touched[:] = False
                    for t in order:
                        row = cand_a[t]
                        if touched[row]:
                            continue
                        refit[row, site] = 1 - refit[row, site]
                        gain[row, site] = cand_gain[t]
                        flipped[row] = True
                        moved = True
                        _touch(touched, together, row)
                # Two-row moves act on co-painted rows: unequal alleles swap, equal alleles flip jointly. Both
                # change two cells and require twice the per-cell minimum gain. Rounds follow the flip-phase
                # rule, with pairs in (a, b) order.
                while True:
                    n = 0
                    for row in range(founders):
                        if refit[row, site] >= 0:
                            row_carriers[row] = _carriers(pair_count, refit[:, site], row)
                            flip_gain[row] = _flip_gain(table, refit[:, site], row)
                    for a in range(founders):
                        if swapped[a]:
                            continue
                        for b in range(a + 1, founders):
                            if swapped[b] or not together[a, b]:
                                continue
                            xa, xb = refit[a, site], refit[b, site]
                            if xa < 0 or xb < 0:
                                continue
                            # Both rows flip: the joint gain is their flip gains plus the pair curvature.
                            if finite:
                                delta = flip_gain[a] + flip_gain[b] + _pair_curvature(table, refit[:, site], a, b)
                            else:
                                delta = _two_row_gain(table, refit[:, site], a, b, 0 if xa != xb else 1)
                            if delta > 0.0 and delta >= swap_minimum_gain:
                                cand_a[n] = a
                                cand_b[n] = b
                                cand_kind[n] = 0 if xa != xb else 1
                                cand_gain[n] = delta
                                cand_n[n] = row_carriers[a] + row_carriers[b]
                                n += 1
                    if n == 0:
                        break
                    order = _decision_order(cand_gain, cand_n, n, reverse, cand_class)
                    if not finite:
                        conflict = _swap_conflicts(
                            table, refit, site, together, cand_a, cand_b, cand_kind, cand_class, order, n,
                            swap_minimum_gain, touched,
                        )
                    else:
                        conflict = _swap_conflicts_fast(
                            table, refit, site, together, cand_a, cand_b, cand_kind, cand_gain, cand_class, order, n,
                            swap_minimum_gain, touched, row_uses,
                        )
                    if conflict:
                        for row in range(founders):
                            if touched[row]:
                                refit[row, site] = -1
                                gain[row, site] = 0.0
                                moved = True
                        continue
                    touched[:] = False
                    for t in order:
                        a, b = cand_a[t], cand_b[t]
                        if touched[a] or touched[b]:
                            continue
                        _apply_two_row(refit[:, site], a, b, cand_kind[t])
                        gain[a, site] = cand_gain[t]
                        gain[b, site] = cand_gain[t]
                        swapped[a] = True
                        swapped[b] = True
                        moved = True
                        _touch(touched, together, a)
                        _touch(touched, together, b)
                if not moved:
                    break
            # Fills: cells unknown in the painted panel, decided against the post-sweep state
            # and applied together (module docstring).
            n = 0
            for row in range(founders):
                if calls[row, site] >= 0 or refit[row, site] >= 0:
                    continue
                ll0 = 0.0
                ll1 = 0.0
                up = 0
                down = 0
                if together[row, row]:
                    ll0 += table[row, row, 0]
                    ll1 += table[row, row, 2]
                    up += favour[row, row, 0]
                    down += favour[row, row, 1]
                for partner in range(founders):
                    other = refit[partner, site]
                    if partner == row or other < 0:
                        continue
                    lo, hi = min(row, partner), max(row, partner)
                    if not together[lo, hi]:
                        continue
                    ll0 += table[lo, hi, other]
                    ll1 += table[lo, hi, other + 1]
                    up += favour[lo, hi, 2 * other]
                    down += favour[lo, hi, 2 * other + 1]
                if ll1 - ll0 >= minimum_gain and up >= fill_min_supporters:
                    cand_a[n] = row
                    cand_b[n] = 1
                    cand_gain[n] = ll1 - ll0
                    n += 1
                elif ll0 - ll1 >= minimum_gain and down >= fill_min_supporters:
                    cand_a[n] = row
                    cand_b[n] = 0
                    cand_gain[n] = ll0 - ll1
                    n += 1
            for t in range(n):
                refit[cand_a[t], site] = cand_b[t]
                gain[cand_a[t], site] = cand_gain[t]
    return refit, gain


def refit_alleles(
    calls: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    likelihoods: np.ndarray,
    observed: np.ndarray,
    evidence_sites: np.ndarray,
    *,
    robustness_epsilon: float,
    minimum_gain_nats: float,
    coordinate_sweeps: int,
    fill_min_supporters: int,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Refit a K-by-L allele matrix (-1 unknown; filled only by the fill rule) given painted rows.

    ``first``/``second`` int16[N, L]: founder rows of the two painted tracks at
    each panel site (-1 unknown); ``likelihoods`` float64[N, L_e, 3]
    normalised and ``observed`` bool[N, L_e] on the evidence site axis;
    ``evidence_sites`` int64[L] maps panel sites to it. Returns the refit
    matrix and the accepted gain per cell (zero where unchanged; a cell flipped
    twice keeps its last gain).
    """
    return _refit_kernel(
        np.ascontiguousarray(calls, dtype=np.int8),
        np.ascontiguousarray(first, dtype=np.int16),
        np.ascontiguousarray(second, dtype=np.int16),
        likelihoods,
        np.ascontiguousarray(observed, dtype=np.bool_),
        np.ascontiguousarray(evidence_sites, dtype=np.int64),
        float(robustness_epsilon),
        float(minimum_gain_nats),
        int(coordinate_sweeps),
        bool(reverse),
        int(fill_min_supporters),
    )
