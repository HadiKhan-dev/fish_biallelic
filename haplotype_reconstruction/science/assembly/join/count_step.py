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
ties.quantum("join_ll", |V(P)|)): within a class the path's content decides (base: the
lowest slot or pool index). Births are excluded by per-sample upper bounds (two-region bounds of
:mod:`.certify`; u_s >= d_s >= q_s) through the necessary conditions of the effective-carrier test,
and scored exactly best-first until the bound excludes every remaining birth. Work per step:
O(D k |pool|) bounds plus exact scores restricted to the samples a bound cannot rule out.

Owns: count candidates, shadow decisions and search-side count drivers.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: WP_AS5 §4; AJ21–AJ25, AJ32–AJ33.
Identity: Death precedes birth; shared bootstrap keys, frozen revisits and born tags retain native semantics.
"""

from __future__ import annotations

import copy
import itertools
import math
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from haplotype_reconstruction.core import ties

from .. import count_rule
from .. import orders as tie_rules
from ..count_rule.criterion import _test as margin_test
from ..count_rule.geometry import shadow_choices
from ..count_rule.parameters import BIRTH_SETS
from ..parameters import SPARSE_RELATIVE
from .certify import PanelForwards
from .diagnostics import append_event, ordered_refit_events
from .moves import improves as _improves
from .objective import column_sums, score_samples
from .state import DEFAULT_TIE_SETTINGS, Panel, PanelKey, Path, SearchState, search_tie_policy

if TYPE_CHECKING:
    from ..count_rule.geometry import RowGeometry
    from ..count_rule.parameters import CountRuleParameters
    from .emissions import EmissionLayout

# Bytes of persistent storage the birth memo of one panel may hold, as charged (context, scored births and
# novelty flags; beyond it: recompute). Transient work (bootstrap draws while a decision resolves, copies returned
# to the caller) and the search's own arrays are outside it.
BIRTH_MEMO_BYTES = 256 << 20
# Upper charges (bytes) of the memo's Python objects (CPython 3.14 sizes with margin; the memory probe checks
# them against deep sizes): an object header (array, bytes, tuple, small record), one dict slot (key and value
# references, index and growth slack), one sequence item, a float, a count-rule margin test (measured 1.5-1.6 KB),
# a recorded count decision (measured 3.5 KB).
_OBJECT_BYTES = 128
_SLOT_BYTES = 128
_ITEM_BYTES = 16
_FLOAT_BYTES = 32
_MARGIN_TEST_BYTES = 2048
_DECISION_BYTES = 5120


def _test_record(test) -> dict:
    """A count_rule.MarginTest as a plain JSON record (lists, Python numbers)."""
    record = asdict(test)
    for name in ("top_samples", "top_differences", "top_shares"):
        record[name] = list(record[name])
    return record


class PendingDecision:
    """A count decision whose bootstrap band is computed only when a caller records it.

    The point part (both effective-carrier tests, the verdict and min(m_d, m_q)) equals
    count_rule.decide's: the same differences and the same test. The band of min(m_d, m_q)
    (count_rule.decide or the shadow family's shared draws) depends only on the decision's own
    scores, cost, configuration and seed key, and only the decision a record reports reads it:
    the shadow and birth choices compare verdicts, margins and gains. :meth:`resolve` returns the
    CountDecision count_rule.decide would have returned (with accept_larger False where the
    shadow family's opposed verdicts withheld the larger panel), so every recorded field is
    unchanged; the other candidates' bands are never drawn (computed only for the winner).
    """

    __slots__ = ("kind", "accept_larger", "margin", "total", "sequence", "_inputs", "_resolved")

    def __init__(self, kind, accept_larger, margin, total, sequence, inputs):
        self.kind = kind
        self.accept_larger = accept_larger
        self.margin = margin
        self.total = total
        self.sequence = sequence
        self._inputs = inputs
        self._resolved = None

    def withheld(self) -> PendingDecision:
        """The same decision with the larger panel withheld (opposed shadow verdicts)."""
        withheld = PendingDecision(self.kind, False, self.margin, self.total, self.sequence, self._inputs)
        if self._resolved is not None:
            withheld._resolved = (
                replace(self._resolved, accept_larger=False) if self._resolved.accept_larger else self._resolved
            )
        return withheld

    def retained_bytes(self) -> int:
        """Upper charge of what this decision keeps: its point tests, and either its recorded decision or the
        inputs it needs to compute one (scores, shadow, seed panel and shadow family; the smaller panel's scores
        are the memo context's)."""
        charge = _OBJECT_BYTES + 2 * _MARGIN_TEST_BYTES
        if self._resolved is not None:
            return charge + _DECISION_BYTES
        larger, smaller, shadow, seed_panel, family = self._inputs
        charge += _OBJECT_BYTES + _array_bytes(shadow)
        # The decision's and the family's seed-panel tuples; their paths are the memo context's panel paths and
        # the entry's key path, charged there.
        charge += 2 * (_OBJECT_BYTES + _ITEM_BYTES * len(seed_panel))
        if family is not None:
            charge += 2 * _OBJECT_BYTES + _MARGIN_TEST_BYTES
        return charge + _DECISION_BYTES  # room for the decision a later resolve records

    def resolve(self, cost, config) -> count_rule.CountDecision:
        if self._resolved is not None:
            return self._resolved
        larger, smaller, shadow, seed_panel, family = self._inputs
        if family is None:
            decision = count_rule.decide(
                larger,
                smaller,
                shadow,
                cost,
                kind=self.kind,
                config=config,
                seed_key=("join_count", self.kind, sorted(seed_panel)),
            )
        else:
            decision = family.family().decide(shadow)
        if decision.accept_larger and not self.accept_larger:
            decision = replace(decision, accept_larger=False)
        self._resolved = decision
        # The recorded decision is all a later caller reads: release the scores, the shadow and a shadow
        # family's bootstrap draws (2,000 x S indices) that only this computation needed.
        self._inputs = None
        return decision


class _ShadowFamily:
    """One shadow family's shared total test, and its count_rule.ShadowDecisionFamily (bootstrap draws)
    built when the first of its decisions is resolved."""

    __slots__ = ("total", "_arguments", "_family")

    def __init__(self, larger, smaller, cost, kind, config, seed_panel):
        self.total = margin_test(larger - smaller, cost, len(larger))
        self._arguments = (larger, smaller, cost, kind, config, seed_panel)
        self._family = None

    def family(self) -> count_rule.ShadowDecisionFamily:
        if self._family is None:
            larger, smaller, cost, kind, config, seed_panel = self._arguments
            self._family = count_rule.ShadowDecisionFamily(
                larger, smaller, cost, kind=kind, config=config, seed_key=("join_count", kind, sorted(seed_panel))
            )
        return self._family


def _array_bytes(array) -> int:
    """Upper charge of an array: the bytes of the buffer that owns its data (a view keeps its whole owner alive)
    and a header."""
    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    return _OBJECT_BYTES + max(int(owner.nbytes), int(array.nbytes))


def _panel_bytes(paths) -> int:
    """Upper charge of a list or tuple of path tuples (small int entries are shared objects)."""
    return _OBJECT_BYTES + sum(_OBJECT_BYTES + _ITEM_BYTES * len(path) for path in paths)


def _details_bytes(details) -> int:
    """Upper charge of a shadow-decision details record: its dict, the nearest-row choices, the incumbent shadow
    and the per-shadow verdict and margin lists."""
    charge = _OBJECT_BYTES + _SLOT_BYTES * len(details)
    charge += _panel_bytes(details["nearest_choices"]) + _panel_bytes([details["incumbent_shadow"]])
    for name in ("verdicts", "sequence_margins", "sequence_loso_margins"):
        charge += _OBJECT_BYTES + (_ITEM_BYTES + _FLOAT_BYTES) * len(details[name])
    return charge


class CountStep:
    """One count decision at a panel certified for same-count moves (deaths first, then births).

    layout: the join's EmissionLayout; penalty: the join's switch penalty; cost: the per-founder
    cost c (count_rule.founder_cost); samples: S; geometry: count_rule.RowGeometry of the batch's
    blocks; config: CountRuleConfig; score(panel) -> exact per-sample scores of a panel (the
    search's full scorer, which counts its evaluations). Panels are lists of path tuples (local
    rows per block). exact_scores / exact_sample_scores count the exact scores this step made.
    """

    def __init__(
        self,
        *,
        layout: EmissionLayout,
        penalty: float,
        cost: float,
        samples: int,
        geometry: RowGeometry,
        config: CountRuleParameters,
        score: Callable[[Panel], np.ndarray],
        tie_counter: ties.TieCounter | None = None,
        label: str | None = None,
        linkage: Callable[[Path], float] | None = None,
        choose_panel=None,
        record_count_family=None,
        tie_settings: ties.TieSettings = DEFAULT_TIE_SETTINGS,
    ) -> None:
        self.tie_settings = tie_settings
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
        self._counts = threading.Lock()  # concurrent seed refits settle on one CountStep
        self.tie_counter = tie_counter
        self.label = label
        # Birth memo: the pure numerics of the births at one panel (its novelty tests, sparse
        # exact scores and point decisions per pool path), reused when the search decides births again at the
        # same panel with the same scores and forwards (after a pool extension and again after tier 2).
        self._memo_context = None
        # 'novelty' -> {path: novelty flag}; ('birth', path, bound bytes) -> one birth's kept entry.
        self._memo: dict[str | tuple[str, Path, bytes], Any] = {}
        self._memo_bytes = 0

    def _choose(self, kind, items, value, content, scale, key=None, quantum_kind="join_ll", link=None, panel=None):
        """The tie-class winner of ``items`` by value(item) (descending; nats of the join log
        likelihood, quantum_kind's quantum at |scale|); inside a class link(item) (the batch linker's
        score, descending; release 2.2), then content(item)."""
        items = list(items)
        values = [float(value(item)) for item in items]
        policy = search_tie_policy(kind, quantum_kind, self.tie_settings)
        quantum = policy.quantum(scale)
        contents = [content(item) for item in items]
        result = ties.tie_classes(values, quantum, content=contents, descending=True, reverse=self.tie_settings.reverse)
        with self._counts:
            ties.add_decision(
                self.tie_counter, kind, result, values, quantum, ties.motion(quantum_kind, scale), descending=True
            )
            if key is not None and self.label is not None and ties.census_enabled(sink=self.tie_settings.census):
                ties.census(kind, f"{self.label}/{key}", [-v for v in values], sink=self.tie_settings.census)
        links = None
        if link is not None and self.linkage is not None and result.tied_winner:
            links = [link(item) for item in items]
        if result.tied_winner and self.tie_counter is not None:
            with self._counts:
                self.tie_counter.note(
                    kind + ".symmetric",
                    inside_tie=bool(tie_rules.symmetric_members(result, links, reverse=self.tie_settings.reverse)),
                )
        symmetric = tie_rules.symmetric_members(result, links, reverse=self.tie_settings.reverse)
        if symmetric and panel is not None and self.choose_panel is not None:
            chosen = self.choose_panel(kind, [panel(items[i]) for i in symmetric], [values[i] for i in symmetric], key)
            return items[symmetric[chosen]]
        return items[tie_rules.linkage_order(result, links, contents, reverse=self.tie_settings.reverse)[0]]

    @staticmethod
    def rows(paths) -> np.ndarray:
        """int64[len(paths), n] local rows of a list of paths."""
        return np.asarray(paths, dtype=np.int64).reshape(len(paths), -1)

    def _counted(self, scores=0, samples=0):
        with self._counts:
            self.exact_scores += scores
            self.exact_sample_scores += samples

    def _decision(self, larger, smaller, shadow, kind, seed_panel, key=None, family=None):
        """The point part of count_rule.decide (the family's total test when given); the band waits for
        PendingDecision.resolve."""
        larger = np.asarray(larger, np.float64)
        smaller = np.asarray(smaller, np.float64)
        shadow = np.asarray(shadow, np.float64)
        samples = len(larger)
        total = margin_test(larger - smaller, self.cost, samples) if family is None else family.total
        sequence = margin_test(larger - shadow, self.cost, samples)
        decision = PendingDecision(
            kind,
            bool(total.passes and sequence.passes),
            float(min(total.margin, sequence.margin)),
            total,
            sequence,
            (larger, smaller, shadow, tuple(map(tuple, seed_panel)), family),
        )
        if key is not None and self.label is not None and ties.census_enabled(sink=self.tie_settings.census):
            # Valley census of the count rule's continuous thresholds (margins at 0, units of c).
            with self._counts:
                ties.census(
                    "assembly.count_margin",
                    f"{self.label}/{key}",
                    [
                        decision.total.margin,
                        decision.total.loso_margin,
                        decision.sequence.margin,
                        decision.sequence.loso_margin,
                    ],
                    sink=self.tie_settings.census,
                )
        return decision

    def shadow_decision(
        self, larger_panel, smaller_panel, focal, larger, smaller, kind, key, forwards=None, cover=None
    ):
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
        paths = list(itertools.product(*choices)) if complete and total_passes else [tuple(rows[0] for rows in choices)]
        family = (
            _ShadowFamily(
                np.asarray(larger, np.float64),
                np.asarray(smaller, np.float64),
                self.cost,
                kind,
                self.config,
                tuple(map(tuple, larger_panel)),
            )
            if len(paths) > 1
            else None
        )
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
            decision = decision.withheld()
        details = dict(
            nearest_choices=choices,
            combinations=combinations,
            evaluated=len(paths),
            complete=len(paths) == combinations,
            resolved_by_total=not total_passes,
            unresolved=unresolved,
            count_unverified=count_unverified,
            cause="nearest_shadow_budget" if count_unverified else None,
            budget=int(self.config.shadow_combinations),
            incumbent_shadow=tuple(rows[0] for rows in choices),
            verdicts=[d.accept_larger for d in decisions],
            sequence_margins=[d.sequence.margin for d in decisions],
            sequence_loso_margins=[d.sequence.loso_margin for d in decisions],
        )
        return decision, lost, details

    def _record_shadow_family(self, larger_panel, smaller_panel, details, key):
        if self.record_count_family is None:
            return
        if details.get("count_unverified", False):
            self.record_count_family(
                dict(
                    panels=[],
                    reason="nearest_shadow_count",
                    source_decision=key,
                    status="count_unverified",
                    count_unverified=True,
                    unresolved_search=True,
                    cause="nearest_shadow_budget",
                    domain_size=details["combinations"],
                    evaluated=details["evaluated"],
                    scope="join_count_candidate",
                    representative_prior=None,
                    shadow=details,
                )
            )
            return
        if not details["unresolved"]:
            return
        self.record_count_family(
            dict(
                panels=sorted({tie_rules.panel_key(larger_panel), tie_rules.panel_key(smaller_panel)}),
                reason="nearest_shadow_count",
                source_decision=key,
                objective="count_rule",
                representative_prior="smaller_K_then_exact_minimum_switches",
                status="ambiguous_k",
                unresolved_search=True,
                shadow=details,
            )
        )

    def _sparse(self, panel, base, bound):
        """Exact V_s(panel) where the bound exceeds base (elsewhere V_s = base exactly)."""
        need = np.flatnonzero(bound > base + SPARSE_RELATIVE * (1.0 + np.abs(base)))
        values = base.copy()
        if len(need):
            values[need] = score_samples(*self.layout.arrays(), self.rows(panel), self.penalty, need.astype(np.int64))
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
        founder = int(
            self._choose(
                "assembly.join_death",
                pool.tolist(),
                lambda f, totals=totals: totals[f],
                lambda f, panel=panel: tie_rules.path_key(panel[f]),
                float(np.sum(base)),
                key,
                quantum_kind="join_ll_wide",
                link=lambda f, panel=panel: -self.linkage(panel[f]),
                panel=lambda f, panel=panel: [path for i, path in enumerate(panel) if i != f],
            )
        )
        rest = [path for i, path in enumerate(panel) if i != founder]
        # An owned copy: the column becomes the incumbent's scores (and may be held by the birth memo); a view
        # would keep the whole samples x founders removal matrix alive.
        smaller = removal[:, founder].copy()
        shadow_details = None
        if len(pure):
            lost = 0
            decision = self._decision(base, smaller, base, "deletion", panel, None if key is None else f"{key}/margin")
        else:
            decision, lost, shadow_details = self.shadow_decision(
                panel, rest, founder, base, smaller, "deletion", None if key is None else f"{key}/margin"
            )
            self._record_shadow_family(panel, rest, shadow_details, key)
        decision = decision.resolve(self.cost, self.config)
        record = dict(
            kind="death",
            founder=founder,
            pure_mosaic=bool(len(pure)),
            lost_cells=int(lost),
            panel_size=k,
            deleted=not decision.accept_larger,
            margin=decision.margin,
            band=list(decision.band),
            ambiguous=decision.ambiguous,
            total=_test_record(decision.total),
            sequence=_test_record(decision.sequence),
        )
        if shadow_details is not None and shadow_details["combinations"] > 1:
            record["shadow"] = shadow_details
            record["shadow_ambiguous"] = shadow_details["unresolved"]
            record["count_unverified"] = shadow_details.get("count_unverified", False)
            record["ambiguous"] |= shadow_details["unresolved"]
        if decision.accept_larger:
            return record, None, None
        return record, rest, smaller

    def _birth_memo(self, panel, base, forwards):
        """The birth memo of this panel: a dict reused only by births at an identical panel (path order),
        per-sample scores and forwards; any other panel starts an empty one. None (recompute) while the valley
        census records per-call margins, or when the panel's context alone would exceed BIRTH_MEMO_BYTES.
        birth() runs only in the search's own thread (never in seed refits)."""
        if self.label is not None and ties.census_enabled(sink=self.tie_settings.census):
            return None
        context = (
            tuple(map(tuple, panel)),
            np.asarray(base, np.float64).tobytes(),
            forwards.rows.tobytes(),
            forwards.rows.shape,
        )
        if context != self._memo_context:
            # The context key itself (panel, score bytes and forwards' rows), the empty novelty table, and the
            # panel's score array (its owning buffer), which the kept decisions reference while the memo lives.
            charge = (
                _panel_bytes(context[0])
                + 3 * _OBJECT_BYTES
                + len(context[1])
                + len(context[2])
                + 2 * _OBJECT_BYTES
                + _array_bytes(np.asarray(base))
            )
            if charge > BIRTH_MEMO_BYTES:
                # A context larger than the cap is not kept: these births are recomputed (no memo).
                self._memo_context, self._memo, self._memo_bytes = None, {"novelty": {}}, 0
                return None
            self._memo_context = context
            self._memo = {"novelty": {}}
            self._memo_bytes = charge
        return self._memo

    def _remember(self, memo, key, value, nbytes) -> bool:
        """Keep a memo entry when everything the memo then holds is charged within BIRTH_MEMO_BYTES (beyond it,
        the caller recomputes)."""
        nbytes += _SLOT_BYTES
        if self._memo_bytes + nbytes > BIRTH_MEMO_BYTES:
            return False
        memo[key] = value
        self._memo_bytes += nbytes
        return True

    def _scored_birth(self, memo, path, larger_panel, panel, base, bound, key, forwards, cover):
        """(larger, decision, lost, shadow details) of one birth: _sparse then _shadow_decision, or their
        memo entry (the counters advance as the computation would advance them; values are copies)."""
        entry = None if memo is None else memo.get(("birth", path, bound.tobytes()))
        if entry is not None:
            larger, decision, lost, details, scores, samples = entry
            self._counted(scores=scores, samples=samples)
            details = copy.deepcopy(details) if details is not None else dict(unresolved=False, count_unverified=False)
            return larger.copy(), decision, lost, details
        before = (self.exact_scores, self.exact_sample_scores)
        larger = self._sparse(larger_panel, base, bound)
        decision, lost, details = self.shadow_decision(
            larger_panel, panel, len(panel), larger, base, "addition", key, forwards=forwards, cover=cover
        )
        if memo is not None:
            counted = (self.exact_scores - before[0], self.exact_sample_scores - before[1])
            # The entry keeps the key (path and bound bytes), the scores the decision also references (the
            # caller gets a copy) and the decision: an unresolved decision's inputs stay until it is resolved,
            # then only its record (PendingDecision.resolve). birth() reads a birth's shadow details only when
            # they are unresolved or count-unverified (it then records them), otherwise only those two flags, so
            # the entry keeps a copy of the details only then.
            entry_key = ("birth", path, bound.tobytes())
            kept = copy.deepcopy(details) if details["unresolved"] or details.get("count_unverified", False) else None
            charge = (
                3 * _OBJECT_BYTES
                + _OBJECT_BYTES
                + _ITEM_BYTES * len(path)
                + len(entry_key[2])
                + _array_bytes(larger)
                + decision.retained_bytes()
                + (_details_bytes(kept) if kept is not None else 0)
            )
            if self._remember(memo, entry_key, (larger, decision, lost, kept, *counted), charge):
                return larger.copy(), decision, lost, details
        return larger, decision, lost, details

    def birth(self, panel, pool, forwards, base, key=None):
        """The pool birth of largest total gain among those the rule accepts.

        Births without a private cell are pure mosaics (never accepted) and are not scored.
        Upper bounds u_s >= d_s >= q_s exclude a birth unless two samples exceed c/(2S) and
        sum u - max u > c/(2S) (necessary for m > 0 and m' > 0 on d and on q). The rest are
        scored exactly in decreasing sum u until no remaining bound can beat the best
        accepted total gain; exact scores are restricted to samples whose bound exceeds V_s(P).

        Returns (record, larger panel or None, its per-sample scores or None, trap_count):
        trap_count counts pure-mosaic pool births that could pass on their total gain.
        Their presence triggers duplicate seeds from every selected founder.
        """
        k = len(panel)
        rows = np.asarray(panel, dtype=np.int64)
        cover = count_rule.covered_rows(rows)
        present = set(map(tuple, panel))
        outside = [path for path in dict.fromkeys(pool) if path not in present]
        memo = self._birth_memo(panel, base, forwards)
        novelty = {} if memo is None else memo["novelty"]
        flags = []
        for path in outside:
            found = novelty.get(path)
            if found is None:
                found = count_rule.has_novelty(path, panel, cover, self.geometry, self.near_identity)
                # A novelty entry: one table slot and the path tuple it keeps alive (the flag is a shared bool).
                charge = _SLOT_BYTES + _OBJECT_BYTES + _ITEM_BYTES * len(path)
                if memo is not None and self._memo_bytes + charge <= BIRTH_MEMO_BYTES:
                    novelty[path] = found
                    self._memo_bytes += charge
            flags.append(found)
        is_novel = np.array(flags, dtype=bool)
        record = dict(
            kind="birth",
            panel_size=k,
            pool=len(pool),
            novel=int(is_novel.sum()),
            survivors=0,
            exact=0,
            accepted=None,
            trap_indicators=0,
        )
        if not outside:
            return record, None, None, 0
        bounds_all = forwards.birth_values(self.rows(outside), self.offsets)
        gain_all = np.maximum(bounds_all - base[:, None], 0.0)
        totals_all = column_sums(gain_all)
        # Float margin: a bound may undershoot the exact value by the summation-order error.
        margin = self.threshold - SPARSE_RELATIVE * (1.0 + float(np.abs(base).max()))
        if self.samples >= 2:
            top = np.partition(gain_all, self.samples - 2, axis=0)[self.samples - 2 :]
            passing = (top.min(axis=0) > margin) & (totals_all - top.max(axis=0) > margin)
        else:
            passing = np.zeros(len(outside), dtype=bool)
        # Pure-mosaic pool births that could pass on their total gain: a panel path covers
        # their cells (the chimera-trap indicator); their presence triggers all selected seeds.
        trap = np.flatnonzero(passing & ~is_novel)
        trap_count = int(len(trap))
        record["trap_indicators"] = trap_count
        novel_index = np.flatnonzero(is_novel)
        novel = [outside[i] for i in novel_index.tolist()]
        if not novel:
            return record, None, None, trap_count
        bounds = bounds_all[:, novel_index]
        totals = totals_all[novel_index]
        survivors = np.flatnonzero(passing[novel_index])
        record["survivors"] = int(len(survivors))
        # Decreasing exact total, equal totals in pool order (reversed under the convention reversal).
        tie_order = -survivors if self.tie_settings.reverse else survivors
        order = survivors[np.lexsort((tie_order, -totals[survivors]))]
        # Births whose bound reaches the tie classes of the best accepted total gain are scored too.
        scale = float(np.sum(base))
        reach = ties.WIDE_CLASS_QUANTA * ties.quantum("join_ll", scale)
        best = None
        accepted = []
        unresolved_births = []
        for index in order.tolist():
            if best is not None and totals[index] < best[0] - reach:
                break
            path = novel[index]
            larger_panel = list(panel) + [path]
            larger, decision, lost, shadow_details = self._scored_birth(
                memo,
                path,
                larger_panel,
                panel,
                base,
                np.ascontiguousarray(bounds[:, index]),
                None if key is None else f"{key}/add{index}/margin",
                forwards,
                cover,
            )
            record["exact"] += 1
            if shadow_details["unresolved"] or shadow_details.get("count_unverified", False):
                unresolved_births.append((float((larger - base).sum()), larger_panel, shadow_details))
            if decision.accept_larger:
                total = float((larger - base).sum())
                accepted.append((total, index, decision, larger, lost))
                if best is None or total > best[0]:
                    best = (total, index, decision, larger, lost)
        for total_gain, alternative, details in unresolved_births:
            if best is None or total_gain >= best[0] - reach:
                self._record_shadow_family(alternative, panel, details, key)
                record.setdefault("shadow_alternatives", []).append(details)
        if best is None:
            return record, None, None, trap_count
        total, index, decision, larger, lost = self._choose(
            "assembly.join_birth",
            accepted,
            lambda item: item[0],
            lambda item: tie_rules.path_key(novel[item[1]]),
            scale,
            None if key is None else f"{key}/birth",
            link=lambda item: self.linkage(novel[item[1]]),
            panel=lambda item: list(panel) + [novel[item[1]]],
        )
        decision = decision.resolve(self.cost, self.config)
        record.update(
            accepted=list(novel[index]),
            lost_cells=int(lost),
            total_gain=total,
            margin=decision.margin,
            band=list(decision.band),
            ambiguous=decision.ambiguous,
            total=_test_record(decision.total),
            sequence=_test_record(decision.sequence),
        )
        return record, list(panel) + [novel[index]], larger, trap_count

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
            founder = int(
                self._choose(
                    "assembly.join_death",
                    pure.tolist(),
                    lambda f, totals=totals: totals[f],
                    lambda f, panel=panel: tie_rules.path_key(panel[f]),
                    float(np.sum(scores)),
                    None if key is None else f"{key}/pure{removed}",
                    quantum_kind="join_ll_wide",
                    link=lambda f, panel=panel: -self.linkage(panel[f]),
                    panel=lambda f, panel=panel: [path for i, path in enumerate(panel) if i != f],
                )
            )
            scores = removal[:, founder].copy()  # owned: a settled panel's scores outlive the removal matrix
            panel = [path for i, path in enumerate(panel) if i != founder]
            removed += 1
        if len(panel) < 2:
            return panel, scores, dict(removed_pure=removed, decided=None), True
        forwards = PanelForwards(self.layout, self.rows(panel), self.penalty, self.total_bins)
        outcome = self.deletion(panel, forwards, scores, None if key is None else f"{key}/death")
        record = dict(removed_pure=removed, decided=None if outcome is None else outcome[0])
        survives = outcome is None or outcome[1] is None
        return panel, scores, record, survives


# Relocated driver cache capacity (cost only); unified with parameters next.


def record_count_family(search: SearchState, family):
    """A count outcome left unresolved by equally near shadow rows."""
    source = family.get("source_decision")
    with search.tie_lock:
        append_event(search, dict(stage="count_shadow_ambiguity", **family), source)
        if source is not None and re.search(r"/seed[0-9]+(?:/|$)", source) is not None:
            key = source.split("/p")[0].split("/pure")[0].split("/death")[0]
            search.seed_families.setdefault(key, []).append(family)
        else:
            search.release_families.append(family)


def apply_same_count(search: SearchState, new_panel, new_scores):
    # This is a cycle guard, not a claim about founder ancestry. Retain tags on
    # unchanged paths; if a tagged path is replaced in a multi-path edit, every new
    # path is a possible descendant. Slot correspondence is not identifiable from
    # the target panel and must not make the guard depend on its edit witness.
    old, new = set(search.selected), set(new_panel)
    inherited = bool((old - new) & search.born)
    search.born.intersection_update(new)
    if inherited:
        search.born.update(new - old)
    search.set_panel(new_panel, new_scores, float(new_scores.sum()))


def refit_seeds(search: SearchState, seeds, refit_scores):
    """The duplicate-seed refits: for each seed, certified best-improvement from P plus a copy
    of the seed over the same-count moves that change the seed or its copy (both exempt from
    the novelty rule meanwhile). Returns (panel, scores, likelihood, pass records) per seed.

    The refits are independent searches, advanced together one certify pass at a time: the
    passes of one round run concurrently (each refit with a share of the threads); before
    them, the pool's birth bounds of every path set the round meets are computed once with
    all threads (they depend on the path set alone, and the refits mostly reach the same
    sets), and the caches keep all of them through the round."""
    selected = search.selected
    states = [
        [selected + [selected[seed]], search.baseline_scores, search.likelihood, frozenset((seed, len(selected))), []]
        for seed in seeds
    ]
    active = list(range(len(states)))
    candidates = search.candidates

    step = search.count_steps

    def certify_pass(index):
        panel, _, panel_ll, slots, passes = states[index]
        return search.certify(
            panel,
            candidates,
            0,
            True,
            len(candidates) + 1,
            panel_ll=panel_ll,
            slots=slots,
            refit_scores=refit_scores,
            key=f"c{step}/seed{seeds[index]}/p{len(passes)}",
        )[:2]

    for _ in range(search.sweep_backstop):
        if not active:
            break
        panels: dict[PanelKey, Panel] = {}
        for index in active:
            panel = cast("Panel", states[index][0])  # state rows: [panel, scores, ll, slots, passes]
            panels.setdefault(tuple(sorted(set(map(tuple, panel)))), panel)
        search.birth_sets = max(BIRTH_SETS, len(panels))
        for path_set, panel in panels.items():
            members = set(path_set)
            outside = [path for path in candidates if path not in members]
            forwards = search.panel_forwards(panel)
            if outside:
                search.claim()
                forwards.birth_values(search.rows(outside), search.block_offsets)
        moved = []
        outcomes = search.concurrent(certify_pass, active)
        for index, (best_move, record) in zip(active, outcomes, strict=False):
            state = states[index]
            cast("list[dict[str, Any]]", state[4]).append(
                dict(
                    accepted=record["accepted"],
                    survivors=record["survivors"],
                    exact=record["exact_scores"],
                    change=record["accepted_likelihood_change"],
                )
            )
            if best_move is not None:
                state[0], state[2], state[1] = (list(best_move[3]), best_move[4][0], best_move[4][1])
                moved.append(index)
        active = moved
    search.birth_sets = BIRTH_SETS
    return [(state[0], state[1], state[2], state[4]) for state in states]


def deaths(search: SearchState, forwards) -> bool:
    """Deaths by the count rule; True if any path died.

    Paths without sequence of their own are deleted unconditionally, one after another
    without re-certifying in between (largest V(P minus f) first); otherwise the best-F
    candidate (largest V(P minus f)) is decided by the rule, and a deletion returns the
    search to tier 1 (the next death is decided at the re-optimized panel). Deleting a
    path that a birth here added closes a count cycle and freezes births; the deletion
    stands. It is recorded as undecided when it is ambiguous (its bootstrap band contains
    zero) and the cycle began with a decided deletion (count_rule.pre_cycle_count;
    deleting a path without sequence of its own is not a decision)."""
    count = search.count
    changed = False
    deaths = 0
    while True:
        outcome = search.counter.deletion(
            search.selected, forwards, search.baseline_scores, key=f"c{search.count_steps}/death{deaths}"
        )
        deaths += 1
        if outcome is None:
            break
        record, rest, values = outcome
        removed = search.selected[record["founder"]]
        record["tagged"] = bool(removed in search.born)
        if rest is not None and record["tagged"]:
            count["cycles"] += 1
            if not record["pure_mosaic"]:
                origin = count_rule.pre_cycle_count(search.sizes, len(search.selected), search.cleanup_steps)
                record["cycle_start_size"] = origin
                if origin >= len(search.selected) and record["ambiguous"]:
                    record["cycle_undecided"] = True
                    count["cycles_undecided"] += 1
        search.events.append(dict(stage="count_death", **record))
        if rest is None:
            break
        if removed in search.born:
            search.births_frozen = True  # count cycle: no further births here
            search.born.discard(removed)
        count["decisions"] += 1
        count["deaths"] += 1
        count["pure_mosaic_deaths"] += record["pure_mosaic"]
        count["ambiguous"] += record["ambiguous"]
        search.set_panel(rest, values, float(values.sum()))
        note_count_state(search, "death")
        search.sizes.append(len(search.selected))
        if record["pure_mosaic"]:
            search.cleanup_steps.add(len(search.sizes) - 1)
        changed = True
        if not record["pure_mosaic"]:
            break
        forwards = search.panel_forwards(search.selected)
    return changed


def count_step(search: SearchState, forwards, pool=None, seeds=False) -> bool:

    search.count_steps += 1
    return count_decision(search, forwards, pool, seeds)


def count_decision(search: SearchState, forwards, pool, seeds) -> bool:
    """One count decision; True if the panel changed. pool None: deaths, then births from the
    current pool, then (seeds=True, at a certified panel) duplicate-seed births; otherwise
    births by the given (saturation) paths only."""
    count = search.count
    if pool is None:
        if deaths(search, forwards):
            return True
        pool_paths = search.candidates
    else:
        pool_paths = pool
    if not search.births_frozen and count["decisions"] + count["seed_swaps"] >= search.count_backstop:
        # Reported backstop: births and seed swaps stop (deaths still decided, panel moves continue).
        search.births_frozen = True
        count["count_backstop_reached"] = True
        search.events.append(
            dict(
                stage="count_backstop",
                changes=count["decisions"] + count["seed_swaps"],
                panel_size=len(search.selected),
            )
        )
    if search.births_frozen:
        return False
    record, larger_panel, values, trap_count = search.counter.birth(
        search.selected, pool_paths, forwards, search.baseline_scores, key=f"c{search.count_steps}"
    )
    count["trap_indicators"] = max(count["trap_indicators"], record["trap_indicators"])
    best = None if larger_panel is None else (float(values.sum()), larger_panel, values, record, "pool")
    if seeds and best is None and trap_count:
        # The chimera indicator triggers the same AJ23 route; every selected founder is
        # a duplicate seed. No indicator ranking can exclude a converged endpoint.
        # O(K) quadratic fits cost O(N L K^3) per round; repeated count rounds can
        # make a pass quartic. This is not a whole-search cubic guarantee.
        chosen = list(range(len(search.selected)))

        # Every refit starts from P plus a copy, so the refits of one count step reach many of
        # the same one-path moves (above all replace(seed, c) = P + c): each is scored once
        # (its value does not depend on the refit that scored it).
        with ordered_refit_events(search, chosen):
            refits = refit_seeds(search, chosen, {})
            step = search.count_steps
            settles = search.concurrent(
                lambda item: search.counter.settle(item[1][0], item[1][1], key=f"c{step}/seed{item[0]}"),
                list(zip(chosen, refits, strict=False)),
            )
        candidates = []
        for seed, (_, _, _, used), (settled, settled_scores, settle_record, survives) in zip(
            chosen, refits, settles, strict=False
        ):
            count["seed_refits"] += 1
            settled_ll = float(settled_scores.sum())
            seed_record = dict(
                kind="seed_birth",
                seed=seed,
                refit_passes=len(used),
                refit=used,
                settled_size=len(settled),
                survives=bool(survives),
                likelihood_change=settled_ll - search.likelihood,
                settle=settle_record,
            )
            search.events.append(dict(stage="count_seed", **seed_record))
            if survives and len(settled) >= len(search.selected) and _improves(settled_ll, search.likelihood):
                candidates.append((settled_ll, settled, settled_scores, seed_record, "seed"))
        if candidates:
            # The best surviving refit: tie classes of its log likelihood, the settled panel's
            # content inside a class (base: the first seed of the maximum).
            best = search.class_winner(
                "assembly.join_seed",
                candidates,
                lambda item: item[0],
                lambda item: tie_rules.panel_key(item[1]),
                census_key=f"c{step}/seed",
                link=lambda item: search.panel_link(item[1]),
                panel=lambda item: item[1],
            )
        count["seed_seconds"] += 0.0
    search.events.append(dict(stage="count_birth", **record))
    if best is None:
        return False
    new_ll, new_panel, new_scores, new_record, source = best
    if tie_rules.panel_key(new_panel) in search.count_states:
        # The step would return to a panel this search already held after a count change: a count
        # cycle (births, deaths and seed swaps chasing each other; seed swaps never tag born paths, so
        # the born-path freeze misses it). Births and seed swaps stop; the panel stays.
        search.births_frozen = True
        count["state_revisits"] += 1
        search.events.append(dict(stage="count_revisit", source=source, panel_size=len(new_panel)))
        return False
    if source == "seed":
        # Equivalent settled endpoints can have different unresolved source histories.
        # Keep every accepted witness of this exact panel: selecting one seed's
        # diagnostic record must not select which ambiguity reaches release.
        selected_key = tie_rules.panel_key(new_panel)
        accepted_seeds = sorted(
            {candidate[3]["seed"] for candidate in candidates if tie_rules.panel_key(candidate[1]) == selected_key},
            key=lambda seed: (search.selected[seed], seed),
        )
        # Seed histories arrive concurrently; release them in founder-path content order.
        search.release_families.extend(
            family
            for seed in accepted_seeds
            for family in search.seed_families.get(f"c{search.count_steps}/seed{seed}", ())
        )
    old = set(search.selected)
    if len(new_panel) > len(search.selected):
        count["decisions"] += 1
        count["births"] += 1
        count["seed_births" if source == "seed" else "pool_births"] += 1
        count["ambiguous"] += bool(new_record.get("ambiguous"))
        search.born.update(path for path in new_panel if path not in old)
    else:
        count["seed_swaps"] += 1
    search.set_panel(new_panel, new_scores, new_ll)
    search.count_states.add(tie_rules.panel_key(search.selected))
    if len(search.selected) != search.sizes[-1]:
        search.sizes.append(len(search.selected))
    return True


def note_count_state(search: SearchState, source):
    """Record the panel held after a death; a death returning to a panel this search already held after
    a count change stands but stops births and seed swaps (a count cycle)."""
    key = tie_rules.panel_key(search.selected)
    if key in search.count_states and not search.births_frozen:
        search.births_frozen = True
        search.count["state_revisits"] += 1
        search.events.append(dict(stage="count_revisit", source=source, panel_size=len(search.selected)))
    search.count_states.add(key)


def final_census(search: SearchState, forwards) -> int:
    """Final paths that would fail their own death test (reported; nothing changes)."""
    selected = search.selected
    k = len(selected)
    if k < 2:
        return 0
    failing = 0
    removal = forwards.removal_scores()
    for founder in range(k):
        rest = [path for i, path in enumerate(selected) if i != founder]
        cover = count_rule.covered_rows(np.asarray(rest, dtype=np.int64))
        if not count_rule.has_novelty(selected[founder], rest, cover, search.geometry, search.near_identity):
            failing += 1
            continue
        decision, _, _ = search.counter.shadow_decision(
            selected, rest, founder, search.baseline_scores, removal[:, founder], "deletion", None
        )
        failing += not decision.accept_larger
    return failing


__all__ = [
    "BIRTH_MEMO_BYTES",
    "PendingDecision",
    "CountStep",
    "record_count_family",
    "apply_same_count",
    "refit_seeds",
    "deaths",
    "count_step",
    "count_decision",
    "note_count_state",
    "final_census",
]
