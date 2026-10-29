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
two rows. A pair flip (release 2.18) changes the equal alleles of two rows painted
together: where both carry the wrong allele, their shared carriers' reads favour
neither single flip (under the robust error floor a heterozygote fits a homozygote's
reads little better than the wrong homozygote), and only the joint change reaches the
genotype the reads show. Unknown alleles are filled only by the fill rule below.

Move order (release 2.1, design A-4 / review R4): a sweep's moves are judged against the state at the
start of a round, not in row-index order. Each sweep has a flip phase and a two-row phase (swaps and pair flips); in a round
every eligible move of the phase is scored against the round's start state, and the moves that reach
their minimum gain are applied in decision order (tie classes of the gains, quantum REFIT_TIE_QUANTUM,
maths.ties rules; inside a class the content order of the rows, which is their canonical index order)
while they do not interact with a move already applied in that round (share a row, or a row painted
together with one of its rows at the site, whose score it would change); the others are re-scored in
the next round. A row changes at most once per phase and sweep; a sweep that changes nothing ends the
site. 2.0 applied the first move reaching its minimum gain in row-index order (Gauss-Seidel), so its
result depended on the row order whenever two candidate moves interacted.

Ambiguous cells (release 2.2, tickets/IC/DESIGN_22.md B3): two moves of one tie class that interact
and exclude each other (after either is applied the other no longer reaches its minimum gain) are
alternatives the data do not separate -- typically two rows painted only together, either of which
could carry the allele the carriers show. Their rows' alleles at the site are released unknown (-1,
'.' in the exports) instead of being decided by row order, and the round is re-scored without them.

Filled cells (release 2.18): after the sweeps, an allele the painted panel left unknown is called when one
allele's robust log likelihood over the row's scoring carriers (partners with a called allele, or the row
itself) beats the other by the minimum gain and at least ``fill_min_supporters`` of those carriers favour
it on their own reads (by more than the tie quantum): the minimum of two favourable samples that local
selection and discovery also require, not their whole release rule.
Every fill of a site is decided against the post-sweep state and the fills are applied together, so no
fill depends on another; carriers whose partner allele is unknown give no evidence, as everywhere here.
Cells the tie rule released unknown in this pass are not filled.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from numba import njit, prange

from ..maths import ties
from ..model.components import ComponentPanel
from ..model.evidence import ContigEvidence
from ..painting.release import ChromosomePainting
from .config import FounderRefitConfig

# Sites per parallel task; each task owns one founder-pair table.
SITE_TILE = 1024
# Tie quantum of the move gains (nats). A gain sums its carriers' log terms; 1e-6 relative genotype-likelihood
# noise (the 2.13 noise floor) moves one carrier's two terms by at most about 2e-6, so two gains summing n
# carriers between them are tied within max(REFIT_TIE_QUANTUM, n REFIT_TIE_QUANTUM_PER_CARRIER) (release 2.18;
# until 2.17 the fixed 2^-16, placed against 1e-9 noise). Both lie far below the 5-nat minimum gain. Gains
# closer than their quantum are ordered by content. REFIT_TIE_QUANTUM is also the margin by which a carrier's
# own reads must favour an allele to support a fill (one sample's terms).
REFIT_TIE_QUANTUM = 2.0 ** -16
REFIT_TIE_QUANTUM_PER_CARRIER = 2.0 ** -18
# A tie class wider than this many quanta keeps value order (maths.ties.WIDE_CLASS_QUANTA).
WIDE_QUANTA = 4.0


@njit(cache=True)
def _root(parent, i):
    """Union-find root with path halving."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True)
def _tie_components(values, carriers, n, component):
    """Tie classes of n candidate gains: the connected components of the relation |g_i - g_j| <=
    max(REFIT_TIE_QUANTUM, (n_i + n_j) REFIT_TIE_QUANTUM_PER_CARRIER), which is the union of two interval
    relations: consecutive gains (in value order) within REFIT_TIE_QUANTUM, and intervals of radius
    n_i REFIT_TIE_QUANTUM_PER_CARRIER around each gain that overlap. Two sorts and union-find, O(n log n);
    neither the convention nor the candidates' listing order enters. component[i] receives the root of i."""
    parent = np.arange(n)
    by_value = np.argsort(values[:n], kind="mergesort")
    for t in range(n - 1):
        i, j = by_value[t], by_value[t + 1]
        if values[j] - values[i] <= REFIT_TIE_QUANTUM:
            a, b = _root(parent, i), _root(parent, j)
            if a != b:
                parent[b] = a
    lower = np.empty(n, dtype=np.float64)
    for i in range(n):
        lower[i] = values[i] - carriers[i] * REFIT_TIE_QUANTUM_PER_CARRIER
    by_lower = np.argsort(lower, kind="mergesort")
    if n:
        frontier = by_lower[0]
        for t in range(1, n):
            i = by_lower[t]
            if lower[i] <= values[frontier] + carriers[frontier] * REFIT_TIE_QUANTUM_PER_CARRIER:
                a, b = _root(parent, frontier), _root(parent, i)
                if a != b:
                    parent[b] = a
                if values[i] + carriers[i] * REFIT_TIE_QUANTUM_PER_CARRIER > \
                        values[frontier] + carriers[frontier] * REFIT_TIE_QUANTUM_PER_CARRIER:
                    frontier = i
            else:
                frontier = i
    for i in range(n):
        component[i] = _root(parent, i)


@njit(cache=True)
def _decision_order(values, carriers, n, reverse, classes):
    """Decision order of n candidate moves with gains values[:n] summing carriers[:n] carriers each (candidates
    listed in content order): exact gain descending; tie classes are the components of _tie_components (two gains
    within max(REFIT_TIE_QUANTUM, (n_i + n_j) REFIT_TIE_QUANTUM_PER_CARRIER), chained), consecutive in value
    order; content (list) order inside a class; a class wider than WIDE_QUANTA x its largest pair quantum keeps
    value order (list order among equal gains). Under the convention reversal (reverse) the list order is
    reversed; class membership does not depend on it. classes[i] receives the class of candidate i (equal values
    = same tie class). O(n log n)."""
    keys = np.empty(n, dtype=np.float64)
    for i in range(n):
        keys[i] = -values[n - 1 - i] if reverse else -values[i]
    order = np.argsort(keys, kind="mergesort")
    if reverse:
        for t in range(n):
            order[t] = n - 1 - order[t]
    component = np.empty(n, dtype=np.int64)
    _tie_components(values, carriers, n, component)
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
        if end - start > 1 and not wide:
            segment = np.sort(order[start:end])
            for t in range(end - start):
                order[start + t] = segment[end - start - 1 - t] if reverse else segment[t]
        start = end
    return order


@njit(cache=True)
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


@njit(cache=True)
def _painted_together(together, a, b):
    """Rows a and b share a painted sample at the site (or are the same row)."""
    return a == b or together[min(a, b), max(a, b)]


@njit(cache=True)
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


@njit(cache=True)
def _apply_two_row(alleles, a, b, kind):
    """Applies a two-row move to ``alleles`` (swap: exchange; pair flip: both flipped)."""
    if kind == 0:
        alleles[a], alleles[b] = alleles[b], alleles[a]
    else:
        alleles[a] = 1 - alleles[a]
        alleles[b] = 1 - alleles[b]


@njit(cache=True)
def _swap_pair_conflict(table, column, together, a1, b1, k1, a2, b2, k2, minimum_gain, marked):
    """One pair of two-row candidates of a tie class (rows a1, b1 with kind k1 and a2, b2 with kind k2; kind 0
    swap, 1 pair flip): when they share a row or a row painted together and either no longer passes after
    the other is applied, marks their four rows. Returns whether rows were marked."""
    if not (_painted_together(together, a1, a2) or _painted_together(together, a1, b2)
            or _painted_together(together, b1, a2) or _painted_together(together, b1, b2)):
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


@njit(cache=True)
def _swap_conflicts(table, refit, site, together, cand_a, cand_b, cand_kind, cand_class, order, n, minimum_gain,
                    marked):
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
                if _swap_pair_conflict(table, column, together, cand_a[i], cand_b[i], cand_kind[i], cand_a[j],
                                       cand_b[j], cand_kind[j], minimum_gain, marked):
                    found = True
        start = end
    return found


@njit(cache=True)
def _carriers(pair_count, alleles, row):
    """Scoring carriers a flip of ``row`` sums: its own pair's samples and those of partners with a called
    allele (release 2.18, the tie quantum)."""
    total = pair_count[row, row]
    for partner in range(len(alleles)):
        if partner == row or alleles[partner] < 0:
            continue
        total += pair_count[min(row, partner), max(row, partner)]
    return total


@njit(cache=True)
def _touch(touched, together, row):
    """Marks `row` and every row painted together with it at the site."""
    touched[row] = True
    for partner in range(len(touched)):
        lo, hi = min(row, partner), max(row, partner)
        if together[lo, hi]:
            touched[partner] = True


@njit(cache=True)
def _swap_gain_from_flips(table, flip_gains, a, b):
    """The swap gain of called rows a < b with different alleles from their single-flip gains, O(1):
    exchanging x_a != x_b is flipping both rows, except that the pair's own samples keep their genotype
    (x_a + x_b = 1), so their two flip terms T[a, b, 2 x_b] - T[a, b, 1] and T[a, b, 2 x_a] - T[a, b, 1]
    are removed: swap(a, b) = flip(a) + flip(b) - T[a, b, 0] - T[a, b, 2] + 2 T[a, b, 1]."""
    return flip_gains[a] + flip_gains[b] - table[a, b, 0] - table[a, b, 2] + 2.0 * table[a, b, 1]


@njit(cache=True)
def _pair_flip_gain_from_flips(table, flip_gains, a, b):
    """Release 2.18: the gain of flipping called rows a < b with equal alleles x together, O(1) from their
    single-flip gains: the pair's own samples go from dosage 2x to 2 - 2x, so each single flip's pair term
    T[a, b, 1] - T[a, b, 2x] is replaced by their joint change:
    pair(a, b) = flip(a) + flip(b) + T[a, b, 0] + T[a, b, 2] - 2 T[a, b, 1]."""
    return flip_gains[a] + flip_gains[b] + table[a, b, 0] + table[a, b, 2] - 2.0 * table[a, b, 1]


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True, parallel=True)
def _refit_kernel(calls, first, second, likelihoods, observed, evidence_sites,
                  robustness_epsilon, minimum_gain, sweeps, reverse, fill_min_supporters):
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
    order (module docstring). Per site: O(N) to fill the table, then per round
    O(K^2) for the K flip gains, O(P) for the swap gains of the P co-painted
    pairs (P <= min(N, K (K - 1) / 2); each from the two flip gains in O(1)) and
    O(n log n) to order the n qualifying moves; a phase has at most K rounds (a
    row changes at most once per phase), so O(N + K^3 log K) per site and sweep.
    Tie conflicts are examined only inside tie classes (gains within a few
    2^-16 nats), at O(K) per tied interacting pair; only many simultaneous
    exact ties would push them past the cubic bound. Fills add O(K^2) per site.
    1024-site tiles in parallel.
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
        flip_gains = np.zeros(founders, dtype=np.float64)
        row_carriers = np.zeros(founders, dtype=np.int64)
        favour = np.zeros((founders, founders, 4), dtype=np.int32)
        pair_count = np.zeros((founders, founders), dtype=np.int64)
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
                    if _flip_conflicts(table, refit, site, together, cand_a, cand_class, order, n, minimum_gain,
                                       touched):
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
                # Two-row phase: swaps (class-B fix B1; 92b3daf flipped any two called rows together on a
                # summed gain >= minimum_gain, so rows never painted together, or with equal alleles, changed
                # on less than the prior of each cell and identical rows could end with different calls) and,
                # release 2.18, pair flips of two rows painted together with equal alleles; both change two
                # cells and need twice the minimum gain. Rounds as above, pairs in (a, b) order.
                while True:
                    n = 0
                    for row in range(founders):
                        if refit[row, site] >= 0:
                            flip_gains[row] = _flip_gain(table, refit[:, site], row)
                            row_carriers[row] = _carriers(pair_count, refit[:, site], row)
                    for a in range(founders):
                        if swapped[a]:
                            continue
                        for b in range(a + 1, founders):
                            if swapped[b] or not together[a, b]:
                                continue
                            xa, xb = refit[a, site], refit[b, site]
                            if xa < 0 or xb < 0:
                                continue
                            if xa != xb:
                                delta = _swap_gain_from_flips(table, flip_gains, a, b)
                            else:
                                delta = _pair_flip_gain_from_flips(table, flip_gains, a, b)
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
                    if _swap_conflicts(table, refit, site, together, cand_a, cand_b, cand_kind, cand_class, order,
                                       n, swap_minimum_gain, touched):
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
            # Fills (release 2.18): cells unknown in the painted panel, decided against the post-sweep state
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


def refit_alleles(calls: np.ndarray, first: np.ndarray, second: np.ndarray,
                  likelihoods: np.ndarray, observed: np.ndarray, evidence_sites: np.ndarray, *,
                  robustness_epsilon: float, minimum_gain_nats: float,
                  coordinate_sweeps: int, fill_min_supporters: int) -> tuple[np.ndarray, np.ndarray]:
    """Refit a K-by-L allele matrix (-1 unknown; filled only by the fill rule) given painted rows.

    ``first``/``second`` int16[N, L]: founder rows of the two painted tracks at
    each panel site (-1 unknown); ``likelihoods`` float64[N, L_e, 3]
    normalised and ``observed`` bool[N, L_e] on the evidence site axis;
    ``evidence_sites`` int64[L] maps panel sites to it. Returns the refit
    matrix and the accepted gain per cell (zero where unchanged; a cell flipped
    twice keeps its last gain).
    """
    return _refit_kernel(
        np.ascontiguousarray(calls, dtype=np.int8), np.ascontiguousarray(first, dtype=np.int16),
        np.ascontiguousarray(second, dtype=np.int16), likelihoods,
        np.ascontiguousarray(observed, dtype=np.bool_),
        np.ascontiguousarray(evidence_sites, dtype=np.int64),
        float(robustness_epsilon), float(minimum_gain_nats), int(coordinate_sweeps),
        bool(ties.reverse_conventions()), int(fill_min_supporters))


@dataclass(frozen=True)
class ComponentRefit:
    """One painted component after the refit.

    panel: the painted panel with ``released`` replaced by the refit alleles and
        ``inference`` = the refit alleles on the frozen snapshot's called cells
        (-1 elsewhere: the snapshot's unknown cells stay unknown).
    changed_rows int32[M], changed_sites int64[M] (panel site index),
    gain_nats float64[M]: released cells that changed, with their accepted gain.
    filled bool[M] (release 2.18): the changed cell was unknown in the painted panel (a fill).
    """

    panel: ComponentPanel
    changed_rows: np.ndarray
    changed_sites: np.ndarray
    gain_nats: np.ndarray
    filled: np.ndarray


@dataclass(frozen=True)
class FounderRefit:
    """Refit product of one chromosome (record ``founder_refit``).

    excluded_samples: sample ids whose reads the sample QC keeps out of this
    contig's founder evidence (and so out of the refit).
    """

    contig: str
    sample_ids: tuple[str, ...]
    components: tuple[ComponentRefit, ...]
    excluded_samples: tuple[str, ...]

    @property
    def panels(self) -> tuple[ComponentPanel, ...]:
        """The refit ComponentPanels in chromosome order."""
        return tuple(component.panel for component in self.components)


def refit_chromosome(painting: ChromosomePainting, evidence: ContigEvidence,
                     config: FounderRefitConfig) -> FounderRefit:
    """Refit every painted component of one chromosome; the painting is unchanged.

    ``painting`` must carry its panels (painting.records.load(..., panels=True));
    ``evidence`` is the contig's evidence (its founder view is used here).
    """
    founder = evidence.founder_view()
    components = []
    for component in painting.components:
        panel = component.panel
        sites = panel.site_indices()
        labels = component.site_labels(panel.positions)
        refit_calls, gain = refit_alleles(
            panel.released, labels[:, 0], labels[:, 1], founder.likelihoods, founder.observed,
            sites, robustness_epsilon=painting.config.robustness_epsilon,
            minimum_gain_nats=config.minimum_gain_nats,
            coordinate_sweeps=config.coordinate_sweeps, fill_min_supporters=config.fill_min_supporters)
        del labels
        rows, columns = np.nonzero(refit_calls != panel.released)
        components.append(ComponentRefit(
            panel=replace(panel, released=refit_calls,
                          inference=np.where(panel.inference >= 0, refit_calls, -1).astype(np.int8)),
            changed_rows=rows.astype(np.int32), changed_sites=columns.astype(np.int64),
            gain_nats=gain[rows, columns], filled=panel.released[rows, columns] < 0))
    excluded = tuple(sample for sample, flag in zip(evidence.sample_ids, evidence.founder_excluded)
                     if flag)
    return FounderRefit(contig=painting.contig, sample_ids=tuple(painting.sample_ids),
                        components=tuple(components), excluded_samples=excluded)


def summary(refit: FounderRefit) -> dict:
    """Small per-chromosome summary (log line, stage summary table)."""
    gains = (np.concatenate([component.gain_nats for component in refit.components])
             if refit.components else np.zeros(0))
    return {
        "contig": refit.contig, "components": len(refit.components),
        "founder_rows": sum(component.panel.founders for component in refit.components),
        "sites": sum(len(component.panel.positions) for component in refit.components),
        "called_alleles": sum(int(np.sum(component.panel.released >= 0))
                              for component in refit.components),
        "changed_alleles": int(len(gains)),
        # release 2.2: cells released unknown because two tied moves excluded each other (module docstring)
        "ambiguous_alleles": sum(
            int(np.sum(component.panel.released[component.changed_rows, component.changed_sites] < 0))
            for component in refit.components),
        # release 2.18: cells the painted panel left unknown, called by the fill rule
        "filled_alleles": sum(int(np.sum(component.filled)) for component in refit.components),
        "changed_inference_alleles": sum(
            int(np.sum(component.panel.inference[component.changed_rows,
                                                 component.changed_sites] >= 0))
            for component in refit.components),
        "changed_sites": sum(len(np.unique(component.changed_sites))
                             for component in refit.components),
        "median_changed_gain_nats": float(np.median(gains)) if len(gains) else None,
        "excluded_samples": list(refit.excluded_samples),
    }
