"""The join's founder-count step: the pipeline's count rule with the novelty condition.

The join objective is :mod:`.objective`'s (per sample, the maximum over sequences of unordered
panel-path pairs of binned emissions minus a uniform penalty per state change). Same-count moves
are judged by it alone. A count move compares a larger panel X+ with X- = X+ minus one path f (a
death of f or a birth of f) and keeps X+ iff the effective-carrier test of
:func:`assembly.count_rule.decide` passes on

* the total per-sample gain d_s = V_s(X+) - V_s(X-), and
* its sequence part q_s = V_s(X+) - V_s(X+^sigma): X+^sigma replaces every cell (block, row) of
  f that no path of X- uses by the nearest row X- uses at that block (Hamming distance over sites
  called in both rows; ties to the lowest row).

Every distance of this step (nearest rows, near identity) is measured on the blocks' pre-fill
inference calls, the calls the join's likelihood reads (:mod:`assembly.count_rule.geometry`).

A path whose every cell another path uses (a pure mosaic, e.g. any recombinant of panel paths),
or which lies within the near-identity rate (CountRuleConfig.near_identity, 1% of called sites) of
another panel path over the span, has q = 0: it is never added and always deleted. Switch savings
never justify a path; a near-identical row's q comes only from its few differing sites;
q_s <= d_s for every sample (X+^sigma contains X-).

Count moves are single deaths of panel paths and single births of pool paths. A merge or rehoming
move would introduce a recombinant of the larger panel, so it is not a count move: linkage
changes are same-count moves (judged by the likelihood) and a path left without private cells is
then deleted here.

Order at a panel certified for same-count moves: (1) a path without private cells is deleted
(largest V(P minus f) first); (2) otherwise the best-F death candidate (largest V(P minus f)) is
decided; (3) otherwise the birth with the largest total gain among the pool births the rule
accepts is added. Births are excluded by per-sample upper bounds (two-region bounds of
:mod:`.certify`; u_s >= d_s >= q_s) through the necessary conditions of the effective-carrier test,
and scored exactly best-first until the bound excludes every remaining birth. Work per step:
O(D k |pool|) bounds plus exact scores restricted to the samples a bound cannot rule out.
"""
from __future__ import annotations

from dataclasses import asdict
import threading

import numpy as np

from .. import count_rule
from .certify import PanelForwards
from .objective import column_sums, score_samples


def _test_record(test) -> dict:
    """A count_rule.MarginTest as a plain JSON record (lists, Python numbers)."""
    record = asdict(test)
    for name in ("top_samples", "top_differences", "top_shares"):
        record[name] = list(record[name])
    return record


def main_coverer(panel, path) -> int:
    """The panel path sharing most cells with ``path`` (ties: lowest index)."""
    shared = [sum(int(a == b) for a, b in zip(row, path)) for row in panel]
    return int(np.argmax(shared))


class CountStep:
    """One count decision at a panel certified for same-count moves (deaths first, then births).

    layout: the join's EmissionLayout; penalty: the join's switch penalty; cost: the per-founder
    cost c (count_rule.founder_cost); samples: S; geometry: count_rule.RowGeometry of the batch's
    blocks; config: CountRuleConfig; score(panel) -> exact per-sample scores of a panel (the
    search's full scorer, which counts its evaluations). Panels are lists of path tuples (local
    rows per block). exact_scores / exact_sample_scores count the exact scores this step made.
    """

    def __init__(self, *, layout, penalty, cost, samples, geometry, config, score):
        self.layout = layout
        self.total_bins = layout.total_bins
        self.offsets = np.ascontiguousarray(layout.offsets[:-1], dtype=np.int64)
        self.penalty = float(penalty)
        self.cost = float(cost)
        self.samples = int(samples)
        self.threshold = self.cost / (2.0 * self.samples)
        self.geometry = geometry
        self.config = config
        self.near_identity = float(config.near_identity)
        self.score = score
        self.exact_scores = 0
        self.exact_sample_scores = 0
        self._counts = threading.Lock()     # concurrent seed refits settle on one CountStep

    @staticmethod
    def rows(paths) -> np.ndarray:
        """int64[len(paths), n] local rows of a list of paths."""
        return np.asarray(paths, dtype=np.int64).reshape(len(paths), -1)

    def _counted(self, scores=0, samples=0):
        with self._counts:
            self.exact_scores += scores
            self.exact_sample_scores += samples

    def _decision(self, larger, smaller, shadow, kind, seed_panel):
        return count_rule.decide(larger, smaller, shadow, self.cost, kind=kind, config=self.config,
                                 seed_key=('join_count', kind, sorted(map(tuple, seed_panel))))

    def _sparse(self, panel, base, bound):
        """Exact V_s(panel) where the bound exceeds base (elsewhere V_s = base exactly)."""
        need = np.flatnonzero(bound > base + 1e-9 * (1.0 + np.abs(base)))
        values = base.copy()
        if len(need):
            values[need] = score_samples(*self.layout.arrays(), self.rows(panel), self.penalty,
                                         need.astype(np.int64))
            self._counted(samples=len(need))
        return values

    def deletion(self, panel, forwards, base):
        """Delete a path without private cells, else decide the best-F death candidate.

        Returns None (fewer than two paths) or (record, smaller panel or None when the founder
        stays, the smaller panel's per-sample scores or None)."""
        k = len(panel)
        if k < 2:
            return None
        removal = forwards.removal_scores()
        totals = column_sums(np.ascontiguousarray(removal))
        pure = np.flatnonzero(~count_rule.novel_paths(panel, self.geometry, self.near_identity))
        pool = pure if len(pure) else np.arange(k)
        founder = int(pool[np.lexsort((pool, -totals[pool]))[0]])
        rest = [path for i, path in enumerate(panel) if i != founder]
        smaller = removal[:, founder]
        if len(pure):
            shadow = base
            lost = 0
        else:
            cover = count_rule.covered_rows(np.asarray(rest, dtype=np.int64))
            shadow_founder, lost = count_rule.shadow_path(panel[founder], cover, self.geometry)
            shadow_panel = list(panel)
            shadow_panel[founder] = shadow_founder
            shadow = self.score(shadow_panel)
            self._counted(scores=1)
        decision = self._decision(base, smaller, shadow, 'deletion', panel)
        record = dict(kind='death', founder=founder, pure_mosaic=bool(len(pure)), lost_cells=int(lost),
                      panel_size=k, deleted=not decision.accept_larger, margin=decision.margin,
                      band=list(decision.band), ambiguous=decision.ambiguous,
                      total=_test_record(decision.total), sequence=_test_record(decision.sequence))
        if decision.accept_larger:
            return record, None, None
        return record, rest, smaller

    def birth(self, panel, pool, forwards, base):
        """The pool birth of largest total gain among those the rule accepts.

        Births without a private cell are pure mosaics (never accepted) and are not scored.
        Upper bounds u_s >= d_s >= q_s exclude a birth unless two samples exceed c/(2S) and
        sum u - max u > c/(2S) (necessary for m > 0 and m' > 0 on d and on q). The rest are
        scored exactly in decreasing sum u until no remaining bound can beat the best
        accepted total gain; exact scores are restricted to samples whose bound exceeds V_s(P).

        Returns (record, larger panel or None, its per-sample scores or None, indicators):
        indicators are the chimera-trap indicators, (total bound gain, path) of the pure-mosaic
        pool births that could pass on their total gain, strongest first.
        """
        k = len(panel)
        rows = np.asarray(panel, dtype=np.int64)
        cover = count_rule.covered_rows(rows)
        present = set(map(tuple, panel))
        outside = [path for path in dict.fromkeys(pool) if path not in present]
        is_novel = np.array([count_rule.has_novelty(path, panel, cover, self.geometry,
                                                    self.near_identity) for path in outside],
                            dtype=bool)
        record = dict(kind='birth', panel_size=k, pool=len(pool), novel=int(is_novel.sum()),
                      survivors=0, exact=0, accepted=None, trap_indicators=0)
        if not outside:
            return record, None, None, []
        bounds_all = forwards.birth_values(self.rows(outside), self.offsets)
        gain_all = np.maximum(bounds_all - base[:, None], 0.0)
        totals_all = column_sums(gain_all)
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
        indicators = [(float(totals_all[i]), outside[i]) for i in
                      trap[np.lexsort((trap, -totals_all[trap]))].tolist()]
        novel_index = np.flatnonzero(is_novel)
        novel = [outside[i] for i in novel_index.tolist()]
        if not novel:
            return record, None, None, indicators
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
            shadow, lost = count_rule.shadow_path(path, cover, self.geometry)
            shadow_bound = forwards.birth_values(self.rows([shadow]), self.offsets)[:, 0]
            shadow_values = self._sparse(list(panel) + [shadow], base, shadow_bound)
            self._counted(scores=1)
            record['exact'] += 1
            decision = self._decision(larger, base, shadow_values, 'addition', larger_panel)
            if decision.accept_larger:
                total = float((larger - base).sum())
                if best is None or total > best[0]:
                    best = (total, index, decision, larger, lost)
        if best is None:
            return record, None, None, indicators
        total, index, decision, larger, lost = best
        record.update(accepted=list(novel[index]), lost_cells=int(lost), total_gain=total,
                      margin=decision.margin, band=list(decision.band), ambiguous=decision.ambiguous,
                      total=_test_record(decision.total), sequence=_test_record(decision.sequence))
        return record, list(panel) + [novel[index]], larger, indicators

    def total_gain(self, panel, path, forwards, base):
        """Exact total gain of adding ``path`` (sparse), or None unless the effective-carrier
        test passes on it (m > 0 and m' > 0): the strength of a chimera-trap indicator."""
        bound = forwards.birth_values(self.rows([path]), self.offsets)[:, 0]
        gain = self._sparse(list(panel) + [path], base, bound) - base
        top = int(np.argmax(gain))
        if (count_rule.margin(gain, self.cost, self.samples) <= 0.
                or count_rule.margin(np.delete(gain, top), self.cost, self.samples) <= 0.):
            return None
        return float(gain.sum())

    def settle(self, panel, scores):
        """A refitted seed panel: delete paths without private cells (best-F first), then the
        best-F candidate must pass the death test. Returns (panel, scores, record, survives)."""
        removed = 0
        panel = list(panel)
        while len(panel) > 1:
            pure = np.flatnonzero(~count_rule.novel_paths(panel, self.geometry, self.near_identity))
            if not len(pure):
                break
            forwards = PanelForwards(self.layout, self.rows(panel), self.penalty, self.total_bins)
            removal = forwards.removal_scores()
            totals = column_sums(np.ascontiguousarray(removal))
            founder = int(pure[np.lexsort((pure, -totals[pure]))[0]])
            scores = removal[:, founder]
            panel = [path for i, path in enumerate(panel) if i != founder]
            removed += 1
        if len(panel) < 2:
            return panel, scores, dict(removed_pure=removed, decided=None), True
        forwards = PanelForwards(self.layout, self.rows(panel), self.penalty, self.total_bins)
        outcome = self.deletion(panel, forwards, scores)
        record = dict(removed_pure=removed, decided=None if outcome is None else outcome[0])
        survives = outcome is None or outcome[1] is None
        return panel, scores, record, survives
