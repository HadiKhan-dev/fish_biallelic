"""The join's founder-count step: the pipeline's count rule (T15) with the novelty condition.

The join objective is unchanged (``panel_scoring.score_emissions``: per sample, the
maximum over sequences of unordered panel-path pairs of binned emissions minus a
uniform penalty per state change). Same-count moves are judged by it alone. A count
move compares a larger panel X+ with X- = X+ minus one path f (a death of f or a birth
of f) and keeps X+ iff the effective-carrier test of ``count_criterion`` passes on

* the total per-sample gain d_s = V_s(X+) - V_s(X-), and
* its sequence part q_s = V_s(X+) - V_s(X+^sigma): X+^sigma replaces every cell
  (block, row) of f that no path of X- uses by the nearest row X- uses at that block
  (Hamming distance over sites called in both rows; ties to the lowest row).

A path whose every cell another path uses (a pure mosaic, e.g. any recombinant of
panel paths), or which lies within NEAR_IDENTITY (1% of called sites) of another panel
path over the span (near-identical = identical), has q = 0: it is never added and
always deleted. Switch savings
never justify a path; a near-identical row's q comes only from its few differing
sites; q_s <= d_s for every sample (X+^sigma contains X-).

Count moves are single deaths of panel paths and single births of pool paths. A merge
or rehoming move would introduce a recombinant of the larger panel, so it is not a
count move: linkage changes are same-count moves (judged by the likelihood) and a path
left without private cells is then deleted here.

Order at a panel certified for same-count moves: (1) a path without private cells is
deleted (largest V(P minus f) first); (2) otherwise the best-F death candidate
(largest V(P minus f)) is decided; (3) otherwise the birth with the largest total gain
among the pool births the rule accepts is added. Births are excluded by per-sample
upper bounds (two-region bounds of ``panel_certify``; u_s >= d_s >= q_s) through the
necessary conditions of the effective-carrier test, and scored exactly best-first
until the bound excludes every remaining birth. Work per step: O(D k |pool|) bounds
plus exact scores restricted to the samples a bound cannot rule out.
"""
from __future__ import annotations

import threading
import numpy as np
from numba import njit, prange

from . import panel_certify
from .founder import count_criterion

# Near-identical = identical over the span (the spec's 2 sites per 200 SNPs): a path within this
# called-site mismatch rate of another panel path owns no sequence of its own (q = 0).
NEAR_IDENTITY = 0.01
_COUNT_CONFIG = []


def count_config():
    """The effective-carrier test's bootstrap constants: the founder refinement's defaults.

    One source for both count decisions of the pipeline; the join has no setting of its
    own. The refinement configuration is imported on first use.
    """
    if not _COUNT_CONFIG:
        from .founder_refinement import FounderRefinementConfig
        _COUNT_CONFIG.append(FounderRefinementConfig())
    return _COUNT_CONFIG[0]


def block_calls(block):
    """Called alleles of a block's rows in dense (sorted-key) order; -1 = unknown."""
    discrete = getattr(block, 'discrete_haps', None)
    if discrete is not None:
        return np.asarray(discrete, dtype=np.int8)
    rows = []
    for key in sorted(block.haplotypes):
        values = np.asarray(block.haplotypes[key])
        if values.ndim == 1:
            rows.append(np.where(np.isin(values, (0, 1)), values, -1).astype(np.int8))
        else:
            ref = (values[:, 0] == 1.0) & (values[:, 1] == 0.0)
            alt = (values[:, 0] == 0.0) & (values[:, 1] == 1.0)
            row = np.full(len(values), -1, dtype=np.int8)
            row[ref], row[alt] = 0, 1
            rows.append(row)
    return np.asarray(rows, dtype=np.int8)


class RowGeometry:
    """Called-site distances between a batch's block rows (computed lazily, cached).

    nearest(b, r, covered): sigma_b(r), the covered row nearest to r (Hamming over sites
    called in both; ties to the lowest row). span_mismatch(p, g): the called-site
    mismatch rate of two paths over the batch's span.
    """

    def __init__(self, blocks):
        self.calls = [block_calls(block) for block in blocks]
        self._nearest = {}
        self._pairs = {}

    def nearest(self, block, row, covered):
        key = (block, row, covered)
        found = self._nearest.get(key)
        if found is None:
            calls = self.calls[block]
            target = calls[row]
            candidates = calls[list(covered)]
            both = (target >= 0)[None, :] & (candidates >= 0)
            distance = ((candidates != target[None, :]) & both).sum(axis=1)
            found = covered[int(np.argmin(distance))]   # covered is sorted: ties -> lowest row
            self._nearest[key] = found
        return found

    def _pair(self, block, first, second):
        key = (block, first, second) if first <= second else (block, second, first)
        found = self._pairs.get(key)
        if found is None:
            a, c = self.calls[block][first], self.calls[block][second]
            both = (a >= 0) & (c >= 0)
            found = (int(((a != c) & both).sum()), int(both.sum()))
            self._pairs[key] = found
        return found

    def span_mismatch(self, path, other):
        mismatch = called = 0
        for block, (first, second) in enumerate(zip(path, other)):
            m, n = self._pair(block, int(first), int(second))
            mismatch += m
            called += n
        return mismatch / called if called else 0.0

    def near_identical(self, path, others):
        """True iff some other path is within NEAR_IDENTITY over the span."""
        return any(self.span_mismatch(path, other) <= NEAR_IDENTITY for other in others)


def covered_rows(rows):
    """Per block, the sorted tuple of rows used by the panel."""
    return [tuple(sorted(set(int(r) for r in rows[:, b]))) for b in range(rows.shape[1])]


def novel_paths(panel, geometry):
    """novel[i]: path i owns sequence the other paths lack: a private cell, and no other path
    within NEAR_IDENTITY of it over the span. A path without it has q = 0 (a pure mosaic or a
    near-duplicate): never added, deleted first, never the result of a same-count move."""
    rows = np.asarray(panel, dtype=np.int64)
    private = private_mask(rows).any(axis=1)
    novel = private.copy()
    for i in np.flatnonzero(private).tolist():
        others = [path for j, path in enumerate(panel) if j != i]
        if geometry.near_identical(panel[i], others):
            novel[i] = False
    return novel


def has_novelty(path, others, cover, geometry):
    """A candidate path owns sequence a panel (cover = its covered rows) lacks."""
    if not any(row not in cover[b] for b, row in enumerate(path)):
        return False
    return not geometry.near_identical(path, others)


def private_mask(rows):
    """private[i, b]: path i is the only path using its row at block b."""
    rows = np.asarray(rows, dtype=np.int64)
    private = np.zeros(rows.shape, dtype=bool)
    for b in range(rows.shape[1]):
        _, inverse, counts = np.unique(rows[:, b], return_inverse=True, return_counts=True)
        private[:, b] = counts[inverse] == 1
    return private


def shadow_path(path, smaller_cover, nearest):
    """f^sigma: f's cells not used by the smaller panel replaced by its nearest rows there."""
    result = list(path)
    lost = 0
    for b, row in enumerate(path):
        if row not in smaller_cover[b]:
            result[b] = nearest.nearest(b, int(row), smaller_cover[b])
            lost += 1
    return tuple(result), lost


@njit(cache=True, inline='always')
def _max_range(values, lo, hi, initial):
    """max(initial, values[lo:hi]) through eight independent running maxima. A maximum is exact,
    so the value equals one sequential scan's; the independent chains let comparisons overlap."""
    m0 = initial
    m1 = m2 = m3 = m4 = m5 = m6 = m7 = -np.inf
    z = lo
    while z + 8 <= hi:
        v = values[z]
        m0 = v if v > m0 else m0
        v = values[z + 1]
        m1 = v if v > m1 else m1
        v = values[z + 2]
        m2 = v if v > m2 else m2
        v = values[z + 3]
        m3 = v if v > m3 else m3
        v = values[z + 4]
        m4 = v if v > m4 else m4
        v = values[z + 5]
        m5 = v if v > m5 else m5
        v = values[z + 6]
        m6 = v if v > m6 else m6
        v = values[z + 7]
        m7 = v if v > m7 else m7
        z += 8
    while z < hi:
        v = values[z]
        m0 = v if v > m0 else m0
        z += 1
    m0 = m1 if m1 > m0 else m0
    m2 = m3 if m3 > m2 else m2
    m4 = m5 if m5 > m4 else m4
    m6 = m7 if m7 > m6 else m6
    m0 = m2 if m2 > m0 else m0
    m4 = m6 if m6 > m4 else m4
    return m4 if m4 > m0 else m0


@njit(cache=True, parallel=True, nogil=True)
def score_samples(flat, block_start, block_haps, block_bins, local_indices, penalty, sample_ids):
    """``panel_scoring.score_emissions`` for a subset of samples (same recursion and order;
    emissions: a chimera_scoring.EmissionLayout's arrays).

    Each block's state emissions are first copied into one array, contiguous per bin."""
    k = local_indices.shape[1]
    pairs = k * (k + 1) // 2
    first = np.empty(pairs, np.int64)
    second = np.empty(pairs, np.int64)
    index = 0
    for a in range(k):
        for b in range(a, k):
            first[index], second[index] = a, b
            index += 1
    widest = 1
    for block in range(len(block_start)):
        widest = max(widest, block_bins[block])
    result = np.empty(len(sample_ids), dtype=np.float64)
    for position in prange(len(sample_ids)):
        sample = sample_ids[position]
        scores = np.empty(pairs, dtype=np.float64)
        gathered = np.empty((widest, pairs))
        initialized = False
        for block in range(len(block_start)):
            haps, bins = block_haps[block], block_bins[block]
            base = block_start[block] + sample * haps * haps * bins
            for pair in range(pairs):
                at = base + (local_indices[block, first[pair]] * haps
                             + local_indices[block, second[pair]]) * bins
                for marker in range(bins):
                    gathered[marker, pair] = flat[at + marker]
            for marker in range(bins):
                if not initialized:
                    for pair in range(pairs):
                        scores[pair] = gathered[marker, pair]
                    initialized = True
                    continue
                switch_base = _max_range(scores, 0, pairs, -np.inf) - penalty
                for pair in range(pairs):
                    stay = scores[pair]
                    scores[pair] = (stay if stay > switch_base else switch_base) + gathered[marker, pair]
        result[position] = _max_range(scores, 0, pairs, -np.inf)
    return result


def _decision(larger, smaller, shadow, cost, kind, seed_panel):
    return count_criterion.decide_with_sequence(
        larger, smaller, shadow, cost, kind=kind, config=count_config(),
        seed_key=('join_count', kind, sorted(map(tuple, seed_panel))))


def main_coverer(panel, path):
    """The panel path sharing most cells with ``path`` (ties: lowest index)."""
    shared = [sum(int(a == b) for a, b in zip(row, path)) for row in panel]
    return int(np.argmax(shared))


class CountStep:
    """One count decision at a panel certified for same-count moves (deaths first, then births)."""

    def __init__(self, *, layout, total_bins, offsets, penalty, cost, samples, geometry, to_local,
                 score):
        self.layout = layout                # local-index emissions (chimera_scoring.EmissionLayout)
        self.total_bins = int(total_bins)
        self.offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        self.penalty = float(penalty)
        self.cost = float(cost)
        self.samples = int(samples)
        self.threshold = self.cost / (2.0 * self.samples)
        self.geometry = geometry
        self.to_local = to_local            # dense paths -> local-index rows
        self.score = score                  # dense panel -> exact per-sample scores (evaluate_panel)
        self.exact_scores = 0
        self.exact_sample_scores = 0
        self._counts = threading.Lock()     # concurrent seed refits settle on one CountStep

    def _counted(self, scores=0, samples=0):
        with self._counts:
            self.exact_scores += scores
            self.exact_sample_scores += samples

    def _sparse(self, panel, base, bound):
        """Exact V_s(panel) where the bound exceeds base (elsewhere V_s = base exactly)."""
        need = np.flatnonzero(bound > base + 1e-9 * (1.0 + np.abs(base)))
        values = base.copy()
        if len(need):
            rows = self.to_local(panel)
            values[need] = score_samples(*self.layout.arrays(), np.ascontiguousarray(rows.T),
                                         self.penalty, need.astype(np.int64))
            self._counted(samples=len(need))
        return values

    def deletion(self, panel, forwards, base):
        """Delete a path without private cells, else decide the best-F death candidate."""
        k = len(panel)
        if k < 2:
            return None
        removal = forwards.removal_scores()
        totals = panel_certify.column_sums(np.ascontiguousarray(removal))
        pure = np.flatnonzero(~novel_paths(panel, self.geometry))
        pool = pure if len(pure) else np.arange(k)
        founder = int(pool[np.lexsort((pool, -totals[pool]))[0]])
        rest = [path for i, path in enumerate(panel) if i != founder]
        smaller = removal[:, founder]
        if len(pure):
            shadow = base
            lost = 0
        else:
            cover = covered_rows(np.asarray(rest, dtype=np.int64))
            shadow_founder, lost = shadow_path(panel[founder], cover, self.geometry)
            shadow_panel = list(panel)
            shadow_panel[founder] = shadow_founder
            shadow = self.score(shadow_panel)
            self._counted(scores=1)
        decision = _decision(base, smaller, shadow, self.cost, 'deletion', panel)
        record = dict(kind='death', founder=founder, pure_mosaic=bool(len(pure)), lost_cells=int(lost),
                      panel_size=k, deleted=not decision.accept_larger, margin=decision.margin,
                      band=list(decision.band), ambiguous=decision.ambiguous,
                      total=decision.diagnostics['total'], sequence=decision.diagnostics['sequence'])
        if decision.accept_larger:
            return record, None, None
        return record, rest, smaller

    def _birth_bounds(self, forwards, new_paths):
        """Per-sample upper bounds on V_s(P + h) for each new path h (local rows; samples x paths):
        ``panel_certify.birth_bounds`` over the sorted paths (shared prefixes). Bounds of one panel
        are kept by path on its forwards (shared with the panel's pool bounds), so a pool bounded
        at a stall is not bounded again at the same panel."""
        return forwards.birth_values(new_paths, self.offsets)

    def birth(self, panel, pool, forwards, base):
        """The pool birth of largest total gain among those the rule accepts, or None.

        Births without a private cell are pure mosaics (never accepted) and are not scored.
        Upper bounds u_s >= d_s >= q_s exclude a birth unless two samples exceed c/(2S) and
        sum u - max u > c/(2S) (necessary for m > 0 and m' > 0 on d and on q). The rest are
        scored exactly in decreasing sum u until no remaining bound can beat the best
        accepted total gain; exact scores are restricted to samples whose bound exceeds V_s(P).
        """
        k = len(panel)
        rows = np.asarray(panel, dtype=np.int64)
        cover = covered_rows(rows)
        present = set(map(tuple, panel))
        outside = [path for path in dict.fromkeys(pool) if path not in present]
        is_novel = np.array([has_novelty(path, panel, cover, self.geometry) for path in outside],
                            dtype=bool)
        record = dict(kind='birth', panel_size=k, pool=len(pool), novel=int(is_novel.sum()), survivors=0,
                      exact=0, accepted=None, trap_indicators=0)
        self.indicators = []
        if not outside:
            return record, None, None
        bounds_all = self._birth_bounds(forwards, self.to_local(outside))
        gain_all = np.maximum(bounds_all - base[:, None], 0.0)
        totals_all = panel_certify.column_sums(gain_all)
        # Float margin: a bound may undershoot the exact value by the summation-order error.
        margin = self.threshold - 1e-9 * (1.0 + float(np.abs(base).max()))
        if self.samples >= 2:
            top = np.partition(gain_all, self.samples - 2, axis=0)[self.samples - 2:]
            passing = (top.min(axis=0) > margin) & (totals_all - top.max(axis=0) > margin)
        else:
            passing = np.zeros(len(outside), dtype=bool)
        # Pure-mosaic pool births that could pass on their total gain: a panel path covers
        # their cells (the chimera-trap indicator); their coverers seed duplicate births.
        trap = np.flatnonzero(passing & ~is_novel)
        record['trap_indicators'] = int(len(trap))
        self.indicators = [(float(totals_all[i]), outside[i]) for i in
                           trap[np.lexsort((trap, -totals_all[trap]))].tolist()]
        novel_index = np.flatnonzero(is_novel)
        novel = [outside[i] for i in novel_index.tolist()]
        if not novel:
            return record, None, None
        bounds = bounds_all[:, novel_index]
        totals = totals_all[novel_index]
        survivors = np.flatnonzero(passing[novel_index])
        record['survivors'] = int(len(survivors))
        order = survivors[np.lexsort((survivors, -totals[survivors]))]
        best = None
        for index in order.tolist():
            if best is not None and totals[index] < best[0]:
                break
            path = novel[index]
            larger_panel = list(panel) + [path]
            larger = self._sparse(larger_panel, base, bounds[:, index])
            shadow, lost = shadow_path(path, cover, self.geometry)
            shadow_bound = self._birth_bounds(forwards, self.to_local([shadow]))[:, 0]
            shadow_values = self._sparse(list(panel) + [shadow], base, shadow_bound)
            self._counted(scores=1)
            record['exact'] += 1
            decision = _decision(larger, base, shadow_values, self.cost, 'addition', larger_panel)
            if decision.accept_larger:
                total = float((larger - base).sum())
                if best is None or total > best[0]:
                    best = (total, index, decision, larger, lost)
        if best is None:
            return record, None, None
        total, index, decision, larger, lost = best
        record.update(accepted=list(novel[index]), lost_cells=int(lost), total_gain=total,
                      margin=decision.margin, band=list(decision.band), ambiguous=decision.ambiguous,
                      total=decision.diagnostics['total'], sequence=decision.diagnostics['sequence'])
        return record, list(panel) + [novel[index]], larger


    def total_gain(self, panel, path, forwards, base):
        """Exact total gain of adding ``path`` (sparse), or None unless the effective-carrier
        test passes on it (m > 0 and m' > 0): the strength of a chimera-trap indicator."""
        bound = self._birth_bounds(forwards, self.to_local([path]))[:, 0]
        gain = self._sparse(list(panel) + [path], base, bound) - base
        top = int(np.argmax(gain))
        if (count_criterion._ecb_margin(gain, self.cost, self.samples) <= 0.
                or count_criterion._ecb_margin(np.delete(gain, top), self.cost, self.samples) <= 0.):
            return None
        return float(gain.sum())

    def settle(self, panel, scores):
        """A refitted seed panel: delete paths without private cells (best-F first), then the
        best-F candidate must pass the death test. Returns (panel, scores, record, survives)."""
        removed = 0
        panel = list(panel)
        while len(panel) > 1:
            pure = np.flatnonzero(~novel_paths(panel, self.geometry))
            if not len(pure):
                break
            forwards = panel_certify.PanelForwards(self.layout, self.to_local(panel), self.penalty,
                                                   self.total_bins)
            removal = forwards.removal_scores()
            totals = panel_certify.column_sums(np.ascontiguousarray(removal))
            founder = int(pure[np.lexsort((pure, -totals[pure]))[0]])
            scores = removal[:, founder]
            panel = [path for i, path in enumerate(panel) if i != founder]
            removed += 1
        if len(panel) < 2:
            return panel, scores, dict(removed_pure=removed, decided=None), True
        forwards = panel_certify.PanelForwards(self.layout, self.to_local(panel), self.penalty,
                                               self.total_bins)
        outcome = self.deletion(panel, forwards, scores)
        record = dict(removed_pure=removed, decided=None if outcome is None else outcome[0])
        survives = outcome is None or outcome[1] is None
        return panel, scores, record, survives
