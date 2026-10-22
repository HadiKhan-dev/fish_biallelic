"""The join's founder-count step: the pipeline's count rule with the novelty condition.

The join objective is :mod:`.objective`'s (per sample, the maximum over sequences of unordered
panel-path pairs of binned emissions minus a uniform penalty per state change). Same-count moves
are judged by it alone. A count move compares a larger panel X+ with X- = X+ minus one path f (a
death of f or a birth of f) and keeps X+ iff the effective-carrier test of
:func:`assembly.count_rule.decide` passes on

* the total per-sample gain d_s = V_s(X+) - V_s(X-), and
* its sequence part q_s = V_s(X+) - V_s(X+^sigma): X+^sigma replaces every cell (block, row) of
  f that no path of X- uses by the nearest row X- uses at that block (Hamming distance over sites
  called in both rows; all equally nearest choices are evaluated when their full Cartesian
  family fits the shadow budget; otherwise the incumbent shadow is evaluated and flagged).

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
accepts is added. "Largest" is decided by tie classes of the join log likelihood (release 2.1,
tie_rules.quantum("join_ll", |V(P)|)): within a class the path's content decides (base: the
lowest slot or pool index). Births are excluded by per-sample upper bounds (two-region bounds of
:mod:`.certify`; u_s >= d_s >= q_s) through the necessary conditions of the effective-carrier test,
and scored exactly best-first until the bound excludes every remaining birth. Work per step:
O(D k |pool|) bounds plus exact scores restricted to the samples a bound cannot rule out.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import itertools
import math
import threading

import numpy as np

from .. import count_rule
from ..count_rule.geometry import shadow_choices
from ...maths import ties
from .. import tie_rules
from .certify import PanelForwards
from .objective import column_sums, score_samples


def _test_record(test) -> dict:
    """A count_rule.MarginTest as a plain JSON record (lists, Python numbers)."""
    record = asdict(test)
    for name in ("top_samples", "top_differences", "top_shares"):
        record[name] = list(record[name])
    return record


def main_coverer(panel, path) -> int:
    """The panel path sharing most cells with ``path`` (ties: lowest index; the panel is in canonical
    order, so the lexicographically smallest path; the highest under ties.reverse_conventions())."""
    shared = [sum(int(a == b) for a, b in zip(row, path)) for row in panel]
    return ties.conv_argmax(shared)


class CountStep:
    """One count decision at a panel certified for same-count moves (deaths first, then births).

    layout: the join's EmissionLayout; penalty: the join's switch penalty; cost: the per-founder
    cost c (count_rule.founder_cost); samples: S; geometry: count_rule.RowGeometry of the batch's
    blocks; config: CountRuleConfig; score(panel) -> exact per-sample scores of a panel (the
    search's full scorer, which counts its evaluations). Panels are lists of path tuples (local
    rows per block). exact_scores / exact_sample_scores count the exact scores this step made.
    """

    def __init__(self, *, layout, penalty, cost, samples, geometry, config, score, tie_counter=None,
                 label=None, linkage=None, choose_panel=None, record_count_family=None):
        self.layout = layout
        # The batch linker's path score (release 2.2): the data criterion inside tie classes (None: content).
        self.linkage = linkage
        self.choose_panel = choose_panel
        self.record_count_family = record_count_family
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
        self.tie_counter = tie_counter
        self.label = label

    def _choose(self, kind, items, value, content, scale, key=None, quantum_kind='join_ll', link=None, panel=None):
        """The tie-class winner of ``items`` by value(item) (descending; nats of the join log
        likelihood, quantum_kind's quantum at |scale|); inside a class link(item) (the batch linker's
        score, descending; release 2.2), then content(item)."""
        items = list(items)
        values = [float(value(item)) for item in items]
        quantum = tie_rules.quantum(quantum_kind, scale)
        contents = [content(item) for item in items]
        result = ties.tie_classes(values, quantum, content=contents, descending=True)
        with self._counts:
            tie_rules.add_decision(self.tie_counter, kind, result, values, quantum,
                                   tie_rules.motion(quantum_kind, scale), descending=True)
            if key is not None and self.label is not None and ties.census_enabled():
                ties.census(kind, f"{self.label}/{key}", [-v for v in values])
        links = None
        if link is not None and self.linkage is not None and result.tied_winner:
            links = [link(item) for item in items]
        if result.tied_winner and self.tie_counter is not None:
            with self._counts:
                self.tie_counter.note(kind + ".symmetric", inside_tie=bool(tie_rules.symmetric_members(result, links)))
        symmetric = tie_rules.symmetric_members(result, links)
        if symmetric and panel is not None and self.choose_panel is not None:
            chosen = self.choose_panel(kind, [panel(items[i]) for i in symmetric],
                                       [values[i] for i in symmetric], key)
            return items[symmetric[chosen]]
        return items[tie_rules.linkage_order(result, links, contents)[0]]

    @staticmethod
    def rows(paths) -> np.ndarray:
        """int64[len(paths), n] local rows of a list of paths."""
        return np.asarray(paths, dtype=np.int64).reshape(len(paths), -1)

    def _counted(self, scores=0, samples=0):
        with self._counts:
            self.exact_scores += scores
            self.exact_sample_scores += samples

    def _decision(self, larger, smaller, shadow, kind, seed_panel, key=None, family=None):
        decision = (count_rule.decide(larger, smaller, shadow, self.cost, kind=kind, config=self.config,
                                     seed_key=('join_count', kind, sorted(map(tuple, seed_panel))))
                    if family is None else family.decide(shadow))
        if key is not None and self.label is not None and ties.census_enabled():
            # Valley census of the count rule's continuous thresholds (margins at 0, units of c).
            with self._counts:
                ties.census('assembly.count_margin', f"{self.label}/{key}",
                            [decision.total.margin, decision.total.loso_margin, decision.sequence.margin,
                             decision.sequence.loso_margin])
        return decision

    def _shadow_decision(self, larger_panel, smaller_panel, focal, larger, smaller, kind, key,
                         forwards=None, cover=None):
        """Evaluate a whole shadow family or retain its incumbent evaluated verdict.

        ``cover`` may reuse the unchanged smaller panel's covered rows across births.
        Computational overflow is not a competing count outcome. It records unperformed
        verification without changing the nominal test, biological ambiguity, or release.
        The smaller-K prior applies only to actually opposed evaluated verdicts.
        """
        if cover is None:
            cover = count_rule.covered_rows(np.asarray(smaller_panel, np.int64))
        choices, lost = shadow_choices(larger_panel[focal], cover, self.geometry)
        combinations = math.prod(map(len, choices))
        total_passes = count_rule.passes(larger - smaller, self.cost, self.samples)
        complete = combinations <= int(self.config.shadow_combinations)
        paths = (list(itertools.product(*choices)) if complete and total_passes
                 else [tuple(rows[0] for rows in choices)])
        family = (count_rule.ShadowDecisionFamily(
            larger, smaller, self.cost, kind=kind, config=self.config,
            seed_key=('join_count', kind, sorted(map(tuple, larger_panel))))
                  if len(paths) > 1 else None)
        decisions = []
        for path in paths:
            panel = list(larger_panel)
            panel[focal] = path
            if forwards is None:
                values = self.score(tie_rules.sort_panel(panel))
            else:
                bound = forwards.birth_values(self.rows([path]), self.offsets)[:, 0]
                values = self._sparse(tie_rules.sort_panel(panel), smaller, bound)
            self._counted(scores=1)
            decisions.append(self._decision(larger, smaller, values, kind, larger_panel, key, family))
        verdicts = {decision.accept_larger for decision in decisions}
        unresolved = total_passes and len(verdicts) > 1
        count_unverified = total_passes and not complete
        # Pick diagnostics consistently. The prior, when used, is explicit alongside them.
        decision = min(decisions, key=lambda d: (d.accept_larger, d.margin, d.sequence.loso_margin))
        if unresolved:
            decision = replace(decision, accept_larger=False)
        details = dict(nearest_choices=choices, combinations=combinations,
                       evaluated=len(paths), complete=len(paths) == combinations,
                       resolved_by_total=not total_passes, unresolved=unresolved,
                       count_unverified=count_unverified,
                       cause='nearest_shadow_budget' if count_unverified else None,
                       budget=int(self.config.shadow_combinations),
                       incumbent_shadow=tuple(rows[0] for rows in choices),
                       verdicts=[d.accept_larger for d in decisions],
                       sequence_margins=[d.sequence.margin for d in decisions],
                       sequence_loso_margins=[d.sequence.loso_margin for d in decisions])
        return decision, lost, details

    def _record_shadow_family(self, larger_panel, smaller_panel, details, key):
        if self.record_count_family is None:
            return
        if details.get('count_unverified', False):
            self.record_count_family(dict(
                panels=[], reason='nearest_shadow_count', source_decision=key,
                status='count_unverified', count_unverified=True, unresolved_search=True,
                cause='nearest_shadow_budget', domain_size=details['combinations'],
                evaluated=details['evaluated'], scope='join_count_candidate',
                representative_prior=None, shadow=details))
            return
        if not details['unresolved']:
            return
        self.record_count_family(dict(
            panels=sorted({tie_rules.panel_key(larger_panel), tie_rules.panel_key(smaller_panel)}),
            reason='nearest_shadow_count', source_decision=key, objective='count_rule',
            representative_prior='smaller_K_then_exact_minimum_switches',
            status='ambiguous_k',
            unresolved_search=True, shadow=details))

    def _sparse(self, panel, base, bound):
        """Exact V_s(panel) where the bound exceeds base (elsewhere V_s = base exactly)."""
        need = np.flatnonzero(bound > base + 1e-9 * (1.0 + np.abs(base)))
        values = base.copy()
        if len(need):
            values[need] = score_samples(*self.layout.arrays(), self.rows(panel), self.penalty,
                                         need.astype(np.int64))
            self._counted(samples=len(need))
        return values

    def deletion(self, panel, forwards, base, key=None):
        """Delete a path without private cells, else decide the best-F death candidate.

        Returns None (fewer than two paths) or (record, smaller panel or None when the founder
        stays, the smaller panel's per-sample scores or None). key: the decision's census key."""
        k = len(panel)
        if k < 2:
            return None
        removal = forwards.removal_scores()
        totals = column_sums(np.ascontiguousarray(removal))
        pure = np.flatnonzero(~count_rule.novel_paths(panel, self.geometry, self.near_identity))
        pool = pure if len(pure) else np.arange(k)
        founder = int(self._choose('assembly.join_death', pool.tolist(), lambda f: totals[f],
                                   lambda f: tie_rules.path_key(panel[f]), float(np.sum(base)), key,
                                   quantum_kind='join_ll_wide',
                                   link=lambda f: -self.linkage(panel[f]),
                                   panel=lambda f: [path for i, path in enumerate(panel) if i != f]))
        rest = [path for i, path in enumerate(panel) if i != founder]
        smaller = removal[:, founder]
        shadow_details = None
        if len(pure):
            lost = 0
            decision = self._decision(base, smaller, base, 'deletion', panel,
                                      None if key is None else f"{key}/margin")
        else:
            decision, lost, shadow_details = self._shadow_decision(
                panel, rest, founder, base, smaller, 'deletion',
                None if key is None else f"{key}/margin")
            self._record_shadow_family(panel, rest, shadow_details, key)
        record = dict(kind='death', founder=founder, pure_mosaic=bool(len(pure)), lost_cells=int(lost),
                      panel_size=k, deleted=not decision.accept_larger, margin=decision.margin,
                      band=list(decision.band), ambiguous=decision.ambiguous,
                      total=_test_record(decision.total), sequence=_test_record(decision.sequence))
        if shadow_details is not None and shadow_details['combinations'] > 1:
            record['shadow'] = shadow_details
            record['shadow_ambiguous'] = shadow_details['unresolved']
            record['count_unverified'] = shadow_details.get('count_unverified', False)
            record['ambiguous'] |= shadow_details['unresolved']
        if decision.accept_larger:
            return record, None, None
        return record, rest, smaller

    def birth(self, panel, pool, forwards, base, key=None):
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
        # Strongest first: tie classes of the bound gain, the path's content inside a class.
        indicators = []
        if len(trap):
            scale = float(np.max(np.abs(totals_all[trap])))
            result = ties.tie_classes([float(totals_all[i]) for i in trap.tolist()],
                                      tie_rules.quantum('join_proxy', scale),
                                      content=[outside[i] for i in trap.tolist()], descending=True,
                                      motion=tie_rules.motion('join_proxy', scale))
            indicators = [(float(totals_all[trap[j]]), outside[trap[j]]) for j in result.order.tolist()]
        novel_index = np.flatnonzero(is_novel)
        novel = [outside[i] for i in novel_index.tolist()]
        if not novel:
            return record, None, None, indicators
        bounds = bounds_all[:, novel_index]
        totals = totals_all[novel_index]
        survivors = np.flatnonzero(passing[novel_index])
        record['survivors'] = int(len(survivors))
        # Decreasing exact total, equal totals in pool order (reversed under the convention reversal).
        tie_order = -survivors if ties.reverse_conventions() else survivors
        order = survivors[np.lexsort((tie_order, -totals[survivors]))]
        # Births whose bound reaches the tie classes of the best accepted total gain are scored too.
        scale = float(np.sum(base))
        reach = ties.WIDE_CLASS_QUANTA * tie_rules.quantum('join_ll', scale)
        best = None
        accepted = []
        unresolved_births = []
        for index in order.tolist():
            if best is not None and totals[index] < best[0] - reach:
                break
            path = novel[index]
            larger_panel = list(panel) + [path]
            larger = self._sparse(larger_panel, base, bounds[:, index])
            decision, lost, shadow_details = self._shadow_decision(
                larger_panel, panel, len(panel), larger, base, 'addition',
                None if key is None else f"{key}/add{index}/margin", forwards=forwards, cover=cover)
            record['exact'] += 1
            if shadow_details['unresolved'] or shadow_details.get('count_unverified', False):
                unresolved_births.append((float((larger - base).sum()), larger_panel, shadow_details))
            if decision.accept_larger:
                total = float((larger - base).sum())
                accepted.append((total, index, decision, larger, lost))
                if best is None or total > best[0]:
                    best = (total, index, decision, larger, lost)
        for total_gain, alternative, details in unresolved_births:
            if best is None or total_gain >= best[0] - reach:
                self._record_shadow_family(alternative, panel, details, key)
                record.setdefault('shadow_alternatives', []).append(details)
        if best is None:
            return record, None, None, indicators
        total, index, decision, larger, lost = self._choose(
            'assembly.join_birth', accepted, lambda item: item[0], lambda item: tie_rules.path_key(novel[item[1]]),
            scale, None if key is None else f"{key}/birth", link=lambda item: self.linkage(novel[item[1]]),
            panel=lambda item: list(panel) + [novel[item[1]]])
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

    def settle(self, panel, scores, key=None):
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
            founder = int(self._choose('assembly.join_death', pure.tolist(), lambda f: totals[f],
                                       lambda f: tie_rules.path_key(panel[f]), float(np.sum(scores)),
                                       None if key is None else f"{key}/pure{removed}",
                                       quantum_kind='join_ll_wide', link=lambda f: -self.linkage(panel[f]),
                                       panel=lambda f: [path for i, path in enumerate(panel) if i != f]))
            scores = removal[:, founder]
            panel = [path for i, path in enumerate(panel) if i != founder]
            removed += 1
        if len(panel) < 2:
            return panel, scores, dict(removed_pure=removed, decided=None), True
        forwards = PanelForwards(self.layout, self.rows(panel), self.penalty, self.total_bins)
        outcome = self.deletion(panel, forwards, scores, None if key is None else f"{key}/death")
        record = dict(removed_pure=removed, decided=None if outcome is None else outcome[0])
        survives = outcome is None or outcome[1] is None
        return panel, scores, record, survives
