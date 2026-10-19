"""Conditional founder-allele refit kernel.

Given the painting, the allele of founder row k at site j maximises the robust
genotype log likelihood of the samples painted as carrying k at j, with each
sample's other painted homologue held at its current allele. Each site is
updated by coordinate ascent over single-row flips and two-row swaps (a fixed
number of sweeps). The minimum gain is a prior per allele: a move is accepted
only when its gain reaches the minimum gain for every cell it changes (one for
a flip, two for a swap). A swap exchanges the different alleles of two rows
painted together at the site: it keeps the genotype of every sample painted
with that pair, so a single flip of either row loses likelihood on those
samples and only the swap corrects an allele carried by the wrong one of the
two rows. Unknown alleles stay unknown.

Move order (release 2.1, design A-4 / review R4): a sweep's moves are judged against the state at the
start of a round, not in row-index order. Each sweep has a flip phase and a swap phase; in a round
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
# Tie quantum of the move gains (nats): gains are sums of per-sample log terms, which 1e-9 relative GL
# noise moves by about 1e-9 per carrier (<= ~1e-6 for a few hundred carriers); 2^-16 = 1.5e-5 lies above
# that and far below the 5-nat minimum gain. Gains closer than this are ordered by content.
REFIT_TIE_QUANTUM = 2.0 ** -16
# A tie class wider than this many quanta keeps value order (maths.ties.WIDE_CLASS_QUANTA).
WIDE_QUANTA = 4.0


@njit(cache=True)
def _decision_order(values, n, quantum, reverse, classes):
    """Decision order of n candidate moves with gains values[:n] (candidates listed in content order):
    exact gain descending, then tie classes chaining gaps <= quantum, content (list) order inside a
    class; a class wider than WIDE_QUANTA x quantum keeps value order (list order among equal gains).
    Under the convention reversal (reverse) the list order is reversed. classes[i] receives the class of
    candidate i (equal values = same tie class)."""
    order = np.arange(n)
    for i in range(1, n):
        j = i
        while j > 0 and (values[order[j]] > values[order[j - 1]] or (
                values[order[j]] == values[order[j - 1]] and ((order[j] > order[j - 1]) if reverse
                                                              else (order[j] < order[j - 1])))):
            order[j - 1], order[j] = order[j], order[j - 1]
            j -= 1
    start = 0
    while start < n:
        end = start + 1
        while end < n and values[order[end - 1]] - values[order[end]] <= quantum:
            end += 1
        wide = values[order[start]] - values[order[end - 1]] > WIDE_QUANTA * quantum
        for t in range(start, end):
            # a wide class keeps value order: its members are not tied alternatives (own class ids)
            classes[order[t]] = n + t if wide else start
        if end - start > 1 and values[order[start]] - values[order[end - 1]] <= WIDE_QUANTA * quantum:
            for i in range(start + 1, end):
                j = i
                while j > start and ((order[j] > order[j - 1]) if reverse else (order[j] < order[j - 1])):
                    order[j - 1], order[j] = order[j], order[j - 1]
                    j -= 1
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
def _flip_conflicts(table, refit, site, together, cand_a, cand_class, n, minimum_gain, marked):
    """Marks (in ``marked``) the rows of flip candidates of one tie class that interact and exclude each
    other: after either flip the other no longer reaches the minimum gain, so the row order would decide.
    Returns whether any row was marked."""
    marked[:] = False
    found = False
    column = refit[:, site]
    for i in range(n):
        for j in range(i + 1, n):
            if cand_class[i] != cand_class[j]:
                continue
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
    return found


@njit(cache=True)
def _swap_conflicts(table, refit, site, together, cand_a, cand_b, cand_class, n, minimum_gain, marked):
    """As _flip_conflicts for swap candidates: two swaps of one tie class that share a row or a row painted
    together, each excluding the other."""
    marked[:] = False
    found = False
    column = refit[:, site]
    for i in range(n):
        for j in range(i + 1, n):
            if cand_class[i] != cand_class[j]:
                continue
            a1, b1, a2, b2 = cand_a[i], cand_b[i], cand_a[j], cand_b[j]
            if not (_painted_together(together, a1, a2) or _painted_together(together, a1, b2)
                    or _painted_together(together, b1, a2) or _painted_together(together, b1, b2)):
                continue
            # swap i applied: does swap j still pass (with different alleles)?
            x1, y1 = column[a1], column[b1]
            column[a1], column[b1] = y1, x1
            pass_j = column[a2] >= 0 and column[b2] >= 0 and column[a2] != column[b2]
            if pass_j:
                gj = _swap_gain(table, column, a2, b2)
                pass_j = gj > 0.0 and gj >= minimum_gain
            column[a1], column[b1] = x1, y1
            x2, y2 = column[a2], column[b2]
            column[a2], column[b2] = y2, x2
            pass_i = column[a1] >= 0 and column[b1] >= 0 and column[a1] != column[b1]
            if pass_i:
                gi = _swap_gain(table, column, a1, b1)
                pass_i = gi > 0.0 and gi >= minimum_gain
            column[a2], column[b2] = x2, y2
            if not (pass_i and pass_j):
                marked[a1] = True
                marked[b1] = True
                marked[a2] = True
                marked[b2] = True
                found = True
    return found


@njit(cache=True)
def _touch(touched, together, row):
    """Marks `row` and every row painted together with it at the site."""
    touched[row] = True
    for partner in range(len(touched)):
        lo, hi = min(row, partner), max(row, partner)
        if together[lo, hi]:
            touched[partner] = True


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
                  robustness_epsilon, minimum_gain, sweeps, reverse):
    """Coordinate ascent per site on a founder-pair log-likelihood table.

    ``table[a, b, g]`` (a <= b) sums log((1 - eps) P(reads | dosage g) + eps / 3)
    over observed samples painted with the unordered pair {a, b}; a row's
    objective then needs only its partners' current alleles. Pairs whose
    partner allele is unknown contribute nothing, as do unpainted tracks. Each
    sweep tries the single-row flips (accepted at a gain >= minimum_gain),
    then the swaps of two rows with different alleles that are painted
    together at the site (``together[a, b]``: some sample painted {a, b} is
    observed there), accepted at a gain >= 2 x minimum_gain because it changes
    two cells; moves are judged against the round's start state in decision
    order (module docstring). O(L (N + K^2 + P K)) per round for P co-painted
    pairs per site (P <= min(N, K (K - 1) / 2)), with 1024-site tiles in parallel.
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
        for site in range(tile * SITE_TILE, min(sites, (tile + 1) * SITE_TILE)):
            table[:] = 0.0
            together[:] = False
            column = evidence_sites[site]
            carried = False
            for sample in range(samples):
                a, b = first[sample, site], second[sample, site]
                if a < 0 or b < 0 or not observed[sample, column]:
                    continue
                if a > b:
                    a, b = b, a
                for dosage in range(3):
                    table[a, b, dosage] += math.log(
                        (1.0 - robustness_epsilon) * likelihoods[sample, column, dosage]
                        + robustness_epsilon / 3.0)
                together[a, b] = True
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
                            n += 1
                    if n == 0:
                        break
                    order = _decision_order(cand_gain, n, REFIT_TIE_QUANTUM, reverse, cand_class)
                    if _flip_conflicts(table, refit, site, together, cand_a, cand_class, n, minimum_gain,
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
                # Swap phase (class-B fix B1; 92b3daf flipped any two called rows together on a
                # summed gain >= minimum_gain, so rows never painted together, or with equal
                # alleles, changed on less than the prior of each cell and identical rows could
                # end with different calls): rounds as above, pairs in (a, b) order.
                while True:
                    n = 0
                    for a in range(founders):
                        if swapped[a]:
                            continue
                        for b in range(a + 1, founders):
                            if swapped[b] or not together[a, b]:
                                continue
                            xa, xb = refit[a, site], refit[b, site]
                            if xa < 0 or xb < 0 or xa == xb:
                                continue
                            delta = _swap_gain(table, refit[:, site], a, b)
                            if delta > 0.0 and delta >= swap_minimum_gain:
                                cand_a[n] = a
                                cand_b[n] = b
                                cand_gain[n] = delta
                                n += 1
                    if n == 0:
                        break
                    order = _decision_order(cand_gain, n, REFIT_TIE_QUANTUM, reverse, cand_class)
                    if _swap_conflicts(table, refit, site, together, cand_a, cand_b, cand_class, n,
                                       swap_minimum_gain, touched):
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
                        xa, xb = refit[a, site], refit[b, site]
                        refit[a, site] = xb
                        refit[b, site] = xa
                        gain[a, site] = cand_gain[t]
                        gain[b, site] = cand_gain[t]
                        swapped[a] = True
                        swapped[b] = True
                        moved = True
                        _touch(touched, together, a)
                        _touch(touched, together, b)
                if not moved:
                    break
    return refit, gain


def refit_alleles(calls: np.ndarray, first: np.ndarray, second: np.ndarray,
                  likelihoods: np.ndarray, observed: np.ndarray, evidence_sites: np.ndarray, *,
                  robustness_epsilon: float, minimum_gain_nats: float,
                  coordinate_sweeps: int) -> tuple[np.ndarray, np.ndarray]:
    """Refit a K-by-L allele matrix (-1 unknown, never filled) given painted rows.

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
        bool(ties.reverse_conventions()))


@dataclass(frozen=True)
class ComponentRefit:
    """One painted component after the refit.

    panel: the painted panel with ``released`` replaced by the refit alleles and
        ``inference`` = the refit alleles on the frozen snapshot's called cells
        (-1 elsewhere: the snapshot's unknown cells stay unknown).
    changed_rows int32[M], changed_sites int64[M] (panel site index),
    gain_nats float64[M]: released cells that changed, with their accepted gain.
    """

    panel: ComponentPanel
    changed_rows: np.ndarray
    changed_sites: np.ndarray
    gain_nats: np.ndarray


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
            coordinate_sweeps=config.coordinate_sweeps)
        del labels
        rows, columns = np.nonzero(refit_calls != panel.released)
        components.append(ComponentRefit(
            panel=replace(panel, released=refit_calls,
                          inference=np.where(panel.inference >= 0, refit_calls, -1).astype(np.int8)),
            changed_rows=rows.astype(np.int32), changed_sites=columns.astype(np.int64),
            gain_nats=gain[rows, columns]))
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
        "changed_inference_alleles": sum(
            int(np.sum(component.panel.inference[component.changed_rows,
                                                 component.changed_sites] >= 0))
            for component in refit.components),
        "changed_sites": sum(len(np.unique(component.changed_sites))
                             for component in refit.components),
        "median_changed_gain_nats": float(np.median(gains)) if len(gains) else None,
        "excluded_samples": list(refit.excluded_samples),
    }
