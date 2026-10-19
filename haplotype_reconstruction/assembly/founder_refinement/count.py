"""Count pass of one component: converged count decisions between converged fixed-K fits.

Count rule (assembly/count_rule, shared with the join): a decision between a larger panel X+ and a
smaller X- keeps X+ iff the effective-carrier test passes on the total per-sample gain
d_s = F_s(X+) - F_s(X-) AND on its sequence part q_s = F_s(X+) - F_s(X+^sigma). The shadow X+^sigma
replaces every cell (block, row) of X+ that a reference panel does not use by the reference's row
nearest to it at that block; the reference is X+ minus the decided founder f for a deletion (only
f's private cells change) and the incumbent X- for an addition (every cell the refit added). A
founder without a private cell (every cell another founder uses at that block) owns no sequence:
q = 0 exactly when X+ holds one. Otherwise q is the exact shadow difference; unlike the join,
refinement takes no near-identity shortcut, because its last level is final and no later level can
re-split a merged pair. Switch savings alone never justify a founder.

Pass: X_K = C(start); deletion rounds (founders without a private cell are the round's candidates
when there are any, otherwise every founder; with more than r = ranked_refits candidates the r
best-ranked by the point margin of their exact unrefitted deletion are refitted, else every one;
the best-F refit is decided; a rejection ends the deletions; K >= 2).

Ties and cleanup (release 2.1, design 5.3; fits are in canonical founder order, so founder indices are
content ranks): the refit decided is the content-smallest member (deleted founder's or copied seed's
path) of the best tie class of the refits' F (quantum tie_rules.quantum("refine_F", F)); the decision is
also evaluated for every other member of that class (one shadow score each) and recorded
(tied_candidates, tie_decisions, tie_decisions_agree; the point decision uses the canonical pick).
Shortlists keep the tie class their cut falls in (at most 2r candidates). Before every deletion round of
the main pass, founders that no sample uses are deleted as cleanup (declared model change, design A-9a):
a founder whose unrefitted deletion leaves every informative sample's score unchanged within the
per-sample quantum (tie_rules.quantum("refine_F")) is removed (content order, re-checked one at a time),
the reduced panel is refitted, and a "cleanup" record is kept; it is not a count decision (2.0 kept such
a founder whenever the best-F tie-break examined another founder first). The best-F keep, the count cycle
and the restore compare F with max(TOL, quantum).
At the final count step only (the last level), duplicate-seed addition rounds (seeds X plus a copy
of founder f; with more than r founders the r seeds of largest exact copy best-response gain are
refitted), each accepted addition followed by one deletion round that must reject, otherwise the
call is a reported count cycle, resolved to the smaller K with the better-F of its two K panels and
flagged undecided when the deletion round's call is ambiguous and the pass reached K by a decided
deletion from K + 1 (count_rule.pre_cycle_count). An optimistic bound (upper_bound) may skip a round
only when it proves the decision and its non-ambiguity for every candidate (a deletion round's
bootstrap band through count_rule.deletion_bound_excludes' replicate-wise bound). The count rule
reads the informative samples only (workspace.count_samples, count_rule.founder_cost's S): every
other sample's F_s is exactly 0 for every panel. The decisions choose K; the panel kept at that
K is the best-F converged fit computed at K during the pass (with the restored panel's adjacent-K
comparisons reported, and the call flagged ambiguous when it fails its own deletion test). Earlier
levels make deletion decisions only: their K is the working set of the next join. K stays within
[2, 2 x incoming K] (the upper bound is a reported limit).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import warnings

import numpy as np
from numba import njit, prange

from .. import count_rule
from .. import tie_rules
from ...maths import ties
from .acceptance import TOL
from .objective import forward_sites, row_max, unordered_pairs
from ... import execution

# Additions stop at the incoming count plus COUNT_LIMIT_MULTIPLE times it (reported when reached).
COUNT_LIMIT_MULTIPLE = 1


@dataclass(frozen=True)
class CountDecisionRecord:
    """One count round of a component's count pass.

    kind: "deletion" | "addition"; round: "r<i>" (pass order); founders: K of the incumbent.
    skipped_by_bound: an optimistic bound proved the decision and its non-ambiguity without
    refits; bound: which one ("deletion_per_sample", "addition_total", "addition_per_sample") or
    None; optimistic_score: the bound's total (None when no bound was computed);
    per_sample_gain_bound(_sample): the per-sample addition bound's largest optimistic gain and its
    sample. band_bound: at a deletion round whose per-sample bound proved the point decision, the
    replicate-wise lower bound of band[0] (count_rule.deletion_bound_excludes): the round was
    skipped iff it is positive beyond the tolerance, otherwise the round ran (None when the point
    bound failed or was not computed).
    accept_larger: the larger panel is kept (deletion rejected) / accepted (addition).
    margin: min of the count rule's total and sequence margins (units of the per-founder cost);
    band: bootstrap percentile interval of that margin; ambiguous: band contains zero (or the
    round closed a count cycle). None for rounds skipped by a bound.
    ranking: "all_candidates" | "unrefitted_point_margin" | "copy_best_response_gain";
    candidate_scores: the ranking keys per founder/seed (None when every candidate is refitted);
    refitted: the candidates converged; candidate_F: their converged F (per founder index, None if
    not refitted); selected, selected_F: the refit decided (release 2.1: the content-smallest member of the
    best tie class of F; founder indices are positions in the incumbent's canonical founder order, so they are
    content ranks);
    runner_up_margin_over_cost: (selected F - next best F) / cost.
    criterion: the count rule's decision record (total and sequence tests). without_sequence:
    founders of the incumbent without a private cell; shadow_cells: cells the shadow replaced;
    hidden_sequence_failures: at a kept deletion round, the other founders whose sequence part
    fails ((founder, margin, loso_margin)). copy_responses: the seeds' copy best responses.
    after_addition: the deletion round that checks an accepted addition. count_cycle,
    cycle_start_count, cycle_undecided: the cycle flags of the two rounds of a count cycle.
    larger_fit, smaller_fit: labels of the two fits compared.
    tied_candidates: the candidates of the selected refit's tie class (release 2.1), in decision order
    (selected first); tie_decisions: per member (candidate, F, accept_larger, margin) of the decision it
    would have given; tie_decisions_agree: all members give the same point decision (None without a
    tie). Kind "cleanup" (release 2.1): founders no sample uses deleted before a deletion round
    (cleanup_removed: their row paths; selected: None; accept_larger False; not a count decision).
    Release 2.2 (tied refits decided by data, else a stated prior): ambiguous_k: the members of the
    selected refit's tie class disagree on the count decision, which then takes the smaller K (prior);
    alternatives: the distinct resulting panels (panel keys, sorted) equally supported by the data when
    the decision changes the panel, released flagged (empty when the choice is unique); prior: the rule
    that decided ("smaller_k", "margin": the member whose count margin most strongly supports the decision
    taken, "symmetric": a remaining exact tie, decided by the canonical panel key and flagged).
    Sample indices (per_sample_gain_bound_sample, the criterion's loso_sample and top_samples) are
    evidence rows; the count rule's S (the criterion's "samples") counts the informative samples.
    """

    kind: str
    round: str
    founders: int
    skipped_by_bound: bool
    accept_larger: bool
    ambiguous: bool
    bound: str | None = None
    optimistic_score: float | None = None
    per_sample_gain_bound: float | None = None
    per_sample_gain_bound_sample: int | None = None
    band_bound: float | None = None
    margin: float | None = None
    band: tuple[float, float] | None = None
    ranking: str | None = None
    candidate_scores: tuple | None = None
    refitted: tuple[int, ...] = ()
    candidate_F: tuple | None = None
    selected: int | None = None
    selected_F: float | None = None
    runner_up_margin_over_cost: float | None = None
    criterion: dict | None = None
    without_sequence: tuple[int, ...] = ()
    shadow_cells: int = 0
    hidden_sequence_failures: tuple | None = None
    copy_responses: tuple | None = None
    after_addition: bool = False
    count_cycle: bool = False
    cycle_start_count: int | None = None
    cycle_undecided: bool = False
    larger_fit: str | None = None
    smaller_fit: str | None = None
    tied_candidates: tuple = ()
    tie_decisions: tuple | None = None
    tie_decisions_agree: bool | None = None
    cleanup_removed: tuple = ()
    ambiguous_k: bool = False
    alternatives: tuple = ()
    prior: str | None = None


@dataclass(frozen=True)
class BestFReplacement:
    """The count pass returned to a K it had held with a better converged panel.

    from_fit, to_fit: labels of the last panel and of the restored best-F fit at that K; gain: F
    difference. recorded: the recorded decisions with the restored panel on one side;
    deletion_vs_best_smaller / addition_vs_best_larger: the count rule between the restored panel
    and the best-F fit at K - 1 / K + 1 (diagnostics); own_deletion_test: the deletion test of the
    restored panel (its recorded round, else the K - 1 comparison); fails_own_deletion_test: then
    the call is flagged ambiguous.
    """

    from_fit: str
    to_fit: str
    gain: float
    recorded: tuple
    deletion_vs_best_smaller: dict | None
    addition_vs_best_larger: dict | None
    own_deletion_test: dict | None
    fails_own_deletion_test: bool | None
    ambiguous: bool


@dataclass(frozen=True)
class CountPass:
    """Result of count_pass: the decided panel and the pass's record."""

    rows: np.ndarray
    scores: np.ndarray
    F: float
    founders_before: int
    fits: tuple
    decisions: tuple[CountDecisionRecord, ...]
    final_fit: str
    weak_junctions: tuple
    count_cycle: bool
    cycle_undecided: bool
    count_limit: int
    count_limit_reached: bool
    additions: bool
    ambiguous: bool
    best_f_replacement: BestFReplacement | None


# ------------------------------------------------------------------ optimistic count bound
@njit(cache=True)
def _block_sample_scores(haps, logs, start, stop, penalty, removed):
    """Per-sample optimistic scores of one block: its rows (minus row `removed` if >= 0) as a free
    panel, every sample's chain started fresh at the block."""
    rows = np.arange(len(haps))
    if removed >= 0:
        rows = rows[rows != removed]
    first, second = unordered_pairs(len(rows))
    dos = np.empty((stop - start, len(first)), np.int8)
    for j in range(stop - start):
        for q in range(len(first)):
            dos[j, q] = haps[rows[first[q]], j] + haps[rows[second[q]], j]
    result = np.empty(logs.shape[0], np.float64)
    for s in range(logs.shape[0]):
        row = np.zeros(len(first), np.float64)
        forward_sites(row, dos, start, logs, s, start, stop, penalty)
        result[s] = row_max(row)
    return result


@njit(cache=True, parallel=True)
def upper_bound(leaves, offsets, logs, penalty, target):
    """Per-block optimistic scores bounding every panel with at most `target` distinct rows per
    block, and their per-sample form.

    Each block chooses its own panel and every sample's diplotype resets freely between blocks;
    a block with target + 1 rows takes its best one-row deletion (exact for that block), any other
    block every local row (possibly more than target). Both relaxations enlarge the feasible set,
    so the sum over blocks bounds every chromosome panel with <= target founder paths drawn from
    these prepared rows (same complete-site mask, emissions and penalty as fitting).
    bounds[b] is the block's total over samples; per_sample[b] lets every sample drop a different
    row of a target+1-row block (a maximum of sums becomes a sum of maxima): summed over blocks it
    bounds each sample. constrained[b] = 1 for a target+1-row block."""
    samples = logs.shape[0]
    bounds = np.empty(len(leaves), np.float64)
    constrained = np.zeros(len(leaves), np.int64)
    per_sample = np.empty((len(leaves), samples), np.float64)
    for block_index in prange(len(leaves)):
        block = np.int64(block_index)
        haps = leaves[block]
        deletion = len(haps) == target + 1
        best = -np.inf
        best_sample = np.full(samples, -np.inf)
        for removed in range(len(haps) if deletion else 1):
            scores = _block_sample_scores(haps, logs, offsets[block], offsets[block + 1], penalty,
                                          removed if deletion else -1)
            total = 0.
            for sample in range(samples):
                total += scores[sample]
            best = max(best, total)
            best_sample = np.maximum(best_sample, scores)
        bounds[block] = best
        per_sample[block] = best_sample
        constrained[block] = int(deletion)
    return bounds, constrained, per_sample


# ------------------------------------------------------------------ helpers
class _CountSamples:
    """The count rule's samples: the component's informative samples (workspace.count_samples,
    evidence rows ascending; None = every row). Per-sample score vectors are restricted to them
    before every count-rule call (vectors pass through unchanged when every sample is informative),
    and the rule's sample indices are mapped back to evidence rows for the records."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, values):
        values = np.asarray(values, np.float64)
        return values if self.rows is None else values[self.rows]

    def row(self, index) -> int:
        return int(index) if self.rows is None else int(self.rows[index])

    def record(self, decision) -> dict:
        """decision.record() with its sample indices (loso and top samples) as evidence rows."""
        record = decision.record()
        if self.rows is not None:
            for part in ("total", "sequence"):
                test = record[part]
                test["loso_sample"] = self.row(test["loso_sample"])
                test["top_samples"] = tuple(self.row(index) for index in test["top_samples"])
        return record


def _point_margins(larger, smaller, cost):
    """The effective-carrier point margins of larger - smaller: m and the leave-one-sample-out m'."""
    differences = np.asarray(larger, np.float64) - np.asarray(smaller, np.float64)
    samples = len(differences)
    rest = np.delete(differences, int(np.argmax(differences)))
    return (count_rule.margin(differences, cost, samples), count_rule.margin(rest, cost, samples))


def _shadow(geometry, larger, reference):
    """X+^sigma: every cell of `larger` that `reference` does not use replaced by the row
    `reference` uses there nearest to it. Returns the shadow rows and the replaced cells."""
    cover = count_rule.covered_rows(np.asarray(reference, np.int64))
    rows, replaced = [], 0
    for path in larger:
        shadow, lost = count_rule.shadow_path(path, cover, geometry)
        rows.append(shadow)
        replaced += lost
    return np.asarray(rows, np.int64), replaced


def _own_sequence(rows):
    """owns[f]: founder f has a private cell (a row at some block that no other founder uses)."""
    return count_rule.private_mask(np.asarray(rows, np.int64)).any(axis=1)


def _sequence_scores(engine, geometry, larger, larger_scores, reference, budget):
    """Per-sample scores of the shadow of `larger` against `reference` (q = larger - shadow).

    q = 0 when a founder of the larger panel has no private cell or when every cell of the larger
    panel is used by the reference; otherwise the shadow is scored exactly. Returns (shadow scores,
    founders without sequence, replaced cells)."""
    owns = _own_sequence(larger)
    without = tuple(int(f) for f in np.flatnonzero(~owns))
    if not owns.all():
        return larger_scores, without, 0
    shadow, replaced = _shadow(geometry, larger, reference)
    if not replaced:
        return larger_scores, without, 0
    return engine.per_sample(shadow, budget), without, int(replaced)


def _loso_margin(criterion):
    """The joint decision's leave-one-sample-out margin: the smaller of the two parts'."""
    return float(min(criterion["total"]["loso_margin"], criterion["sequence"]["loso_margin"]))


def _death_candidate(engine, rows, budget):
    """The join's death candidate of a panel without refits: founders without a private cell first,
    then the largest unrefitted F(rows minus f) (tie classes of F; the founder's path inside a class)."""
    owns = _own_sequence(rows)
    pool = np.flatnonzero(~owns) if not owns.all() else np.arange(len(rows))
    totals = [engine.score(np.delete(rows, f, axis=0), budget) for f in pool.tolist()]
    scale = max(abs(t) for t in totals)
    result = ties.tie_classes(totals, tie_rules.quantum("refine_F", scale),
                              content=[tie_rules.path_key(rows[f]) for f in pool.tolist()], descending=True,
                              motion=tie_rules.motion("refine_F", scale))
    return int(pool[result.winner])


def _fit_many(engine, starts, labels, budget):
    """Converge several independent starts concurrently; results (Fits) in input order. Each fit's
    share of the budget is re-read within its phases, so the last fits take over the finished
    ones' cores; a fit splits its share between its two searches. Its two searches reserve their
    message arrays (engine.converge: fit_bytes each, base's two per fit), so a fit reserves nothing more."""
    results = [None] * len(starts)
    for index, fit in budget.concurrent(
            [lambda child, start=start, label=label: engine.converge(start, label, child)
             for start, label in zip(starts, labels)], 0, label="founder_refinement fits"):
        results[index] = fit
    return results


def _best_refit(chosen, values):
    """Position (in chosen) of the best-F refit; exact ties go to the lower candidate (base rule, kept
    for reference; release 2.1 decides with _refit_choice)."""
    return max(range(len(chosen)), key=lambda i: (values[i], -chosen[i]))


def _refit_choice(values, keys, scale, counter=None, census_key=None):
    """The refit decided at a count step: the content-smallest member (keys: the deleted founder's or
    copied seed's row path) of the best tie class of the refits' F (quantum at |scale|, the incumbent's
    F). Returns (position, positions of its class in decision order)."""
    quantum = tie_rules.quantum("refine_F", scale)
    result = ties.tie_classes([float(v) for v in values], quantum, content=keys, descending=True)
    tie_rules.add_decision(counter, "assembly.refine_refit", result, [float(v) for v in values], quantum,
                           tie_rules.motion("refine_F", scale), descending=True)
    tie_rules.census("assembly.refine_refit", census_key, [-float(v) for v in values])
    pick = result.winner
    members = [int(i) for i in result.members(int(result.classes[pick]))]
    return pick, members


def _tied_resolution(members, pick, outcomes, refits, X):
    """Release 2.2: the count decision among the members of the selected refit's tie class (refits whose F
    the data do not separate), decided by the data where they can and by stated priors otherwise -- never by
    index or content order (design tickets/IC/DESIGN_22.md B1).

    outcomes: {position: CountDecision} of every member. When the members disagree on the decision (one
    would keep the larger panel, another accept the smaller), the smaller K is taken (parsimony: the side the
    count rule takes at an exact zero margin) and flagged ambiguous_k. Among the members giving the decision
    taken, the one whose count margin most strongly supports it is chosen (the most negative margin for the
    smaller panel, the most positive for the larger), by tie classes of the margins (count_margin quantum);
    members still tied are alternatives: when their resulting panels differ, the set is released flagged and
    the canonical panel key decides ("symmetric"). A decision that keeps the incumbent panel X has no
    alternatives. Returns (position, record flags)."""
    if len(members) < 2:
        return pick, {}
    taken_larger = {position: bool(outcomes[position].accept_larger) for position in members}
    disagree = len(set(taken_larger.values())) > 1
    # accept_larger False = the smaller panel (a deletion accepted, an addition rejected)
    candidates = [p for p in members if not taken_larger[p]] if disagree else list(members)
    larger_side = taken_larger[candidates[0]]

    def panel_of(position):
        # The panel the decision leaves: the member's refit (K + 1 rows for an addition, K - 1 for a deletion)
        # when the decision takes it, else the incumbent X.
        refit_rows = np.asarray(refits[position].rows)
        takes_refit = larger_side if len(refit_rows) > len(X) else not larger_side
        return tie_rules.panel_key(refit_rows if takes_refit else X)

    margins = [float(outcomes[p].margin) for p in candidates]
    contents = [panel_of(p) for p in candidates]
    quantum = tie_rules.quantum("count_margin")
    result = ties.tie_classes(margins, quantum, content=contents, descending=larger_side)
    winner = candidates[result.winner]
    tied = [candidates[i] for i in result.members(int(result.classes[result.winner]))]
    distinct = sorted({panel_of(p) for p in tied})
    flags = dict(ambiguous_k=bool(disagree), prior="smaller_k" if disagree else "margin")
    if len(distinct) > 1:
        flags["alternatives"] = tuple(distinct)
        flags["prior"] = flags["prior"] + "+symmetric"
    return winner, flags


def _shortlist(candidates, keys, limit, quantum, contents, *, descending, counter=None, kind="",
               census_key=None):
    """The first ``limit`` candidates in tie-class order of their ranking keys (descending or not),
    extended by the tie class the cut falls in (at most 2 x limit)."""
    values = [float(keys[c]) for c in candidates]
    result = ties.tie_classes(values, quantum, content=contents, descending=descending)
    tie_rules.add_decision(counter, kind, result, values, quantum, None, descending=descending, cut=limit)
    tie_rules.census(kind, census_key, [(-v if descending else v) for v in values])
    return tie_rules.truncate(list(candidates), result, limit)


def _decision(kind, label, founders, decision, ranking, scores, chosen, fits, pick, cost, view, *,
              larger_fit, smaller_fit, **extra):
    values = [fit.summary.F for fit in fits]
    candidate_F = [None] * int(founders)
    for candidate, value in zip(chosen, values):
        candidate_F[candidate] = float(value)
    others = sorted((v for i, v in enumerate(values) if i != pick), reverse=True)
    return CountDecisionRecord(
        kind=kind, round=label, founders=int(founders), skipped_by_bound=False,
        accept_larger=bool(decision.accept_larger), ambiguous=bool(decision.ambiguous),
        margin=float(decision.margin), band=tuple(float(x) for x in decision.band),
        ranking=ranking,
        candidate_scores=None if scores is None else tuple(None if v is None else float(v)
                                                           for v in scores),
        refitted=tuple(int(c) for c in chosen), candidate_F=tuple(candidate_F),
        selected=int(chosen[pick]), selected_F=float(values[pick]),
        runner_up_margin_over_cost=None if not others else float((values[pick] - others[0]) / cost),
        criterion=view.record(decision), larger_fit=larger_fit, smaller_fit=smaller_fit, **extra)


def _restored_comparisons(engine, geometry, budget, rows, scores, best_at, decisions, panels, cost,
                          rule, seed_key, view):
    """Adjacent-K comparisons of a best-F restored panel from fits already computed (diagnostics,
    except the ambiguity flag): the recorded decisions with the restored panel on one side, and the
    count rule between the restored panel and the best-F fit at K - 1 (deletion; the sequence part
    of the restored panel's death candidate) and at K + 1 (addition; the shadow taken against the
    restored panel). The restored panel fails its own deletion test when its recorded deletion
    round accepted a deletion or, without a recorded round, when the best-F K - 1 fit is preferred
    to it; the call is then flagged ambiguous. panels[i]: (larger rows, smaller rows) of
    decisions[i] (None for a round skipped by a bound); view: the count rule's _CountSamples."""
    K = len(rows)
    recorded = []
    for entry, pair in zip(decisions, panels):
        if pair is None:
            continue
        role = ("larger" if np.array_equal(pair[0], rows) else
                "smaller" if np.array_equal(pair[1], rows) else None)
        if role is not None:
            recorded.append(dict(kind=entry.kind, round=entry.round, role=role,
                                 accept_larger=entry.accept_larger, margin=entry.margin,
                                 loso_margin=_loso_margin(entry.criterion),
                                 band=list(entry.band), ambiguous=entry.ambiguous))

    def replay(kind, other):
        held = best_at.get(other)
        if held is None:
            return None
        if kind == "deletion":
            larger, larger_scores, smaller = rows, scores, held[3]
            founder = _death_candidate(engine, rows, budget)
            reference = np.delete(rows, founder, axis=0)
        else:
            larger, larger_scores, smaller, reference = held[1], held[3], scores, rows
        shadow, _, _ = _sequence_scores(engine, geometry, larger, larger_scores, reference, budget)
        decision = count_rule.decide(view(larger_scores), view(smaller), view(shadow), cost,
                                     kind=kind, config=rule, seed_key=[*seed_key, "best_f", kind])
        return dict(fit=held[2], founders=int(other), F=float(held[0]),
                    accept_larger=bool(decision.accept_larger), margin=float(decision.margin),
                    loso_margin=_loso_margin(decision.record()),
                    sequence_margin=float(decision.sequence.margin),
                    band=[float(x) for x in decision.band], ambiguous=bool(decision.ambiguous))

    deletion = replay("deletion", K - 1) if K > 2 else None
    addition = replay("addition", K + 1)
    own = next((item for item in recorded if item["kind"] == "deletion" and item["role"] == "larger"),
               None)
    test = own if own is not None else deletion
    fails = None if test is None else not test["accept_larger"]
    return dict(recorded=tuple(recorded), deletion_vs_best_smaller=deletion,
                addition_vs_best_larger=addition,
                own_deletion_test=(None if test is None else
                                   dict(source="recorded_round" if own is not None
                                        else "best_f_smaller_fit", **test)),
                fails_own_deletion_test=fails, ambiguous=bool(fails))


# ------------------------------------------------------------------ the pass
def count_pass(engine, geometry, start, cost, rule, budget, seed_key, additions, counter=None,
               census_prefix=None):
    """Converged fixed-K fit and converged count decisions for one component (module docstring).

    engine: the component's search.SweepEngine; geometry: count_rule.RowGeometry of its prepared
    panels; start: int64 K x B starting rows; cost: the per-founder cost c; rule: CountRuleConfig;
    seed_key: the component's bootstrap key; additions: run addition rounds (the final count step);
    counter: the component's tie-class counter (tie_rules.SharedCounter) or None; census_prefix: its
    valley-census key prefix or None. Returns a CountPass."""
    ranked_refits = int(rule.ranked_refits)
    ws = engine.workspace
    # Per-sample vectors enter the count rule restricted to the informative samples (cost's S);
    # totals F stay sums over every evidence row (the other rows add exact zeros).
    view = _CountSamples(ws.count_samples)
    incoming = len(start)
    fits, decisions, panels = [], [], []
    fit = engine.converge(start, "start", budget)
    fits.append(fit.summary)
    X = fit.rows
    scores = engine.fit_scores(fit.summary.label, X, budget)
    samples = len(scores)
    count_samples = len(view(scores))
    round_index = 0
    cleanup_index = 0
    upper = incoming + COUNT_LIMIT_MULTIPLE * incoming
    limit_reached = False
    # Best-F converged fit computed at each K: (F, panel, fit label, per-sample scores, fit).
    best_at = {}

    def key(*parts):
        return None if census_prefix is None else "/".join([census_prefix, *map(str, parts)])

    # Exact unrefitted per-sample scores of the current panel's one-founder deletions (the cleanup test and a
    # deletion round's ranking read the same panels): kept for one panel at a time.
    unrefitted_cache = {}

    def unrefitted(panel, founder):
        reduced = np.ascontiguousarray(np.delete(panel, founder, axis=0))
        token = (reduced.shape, reduced.tobytes())
        found = unrefitted_cache.get(token)
        if found is None:
            if len(unrefitted_cache) > 4 * len(panel):
                unrefitted_cache.clear()
            found = unrefitted_cache[token] = engine.per_sample(reduced, budget)
        return found

    def keep_margin(F):
        """Margin of the best-F keep, the cycle keep and the restore: max(TOL, the tie quantum at F)."""
        return max(TOL, tie_rules.quantum("refine_F", F))

    def remember(panel, held_fit, per_sample):
        F_fit = float(per_sample.sum())
        held = best_at.get(len(panel))
        if held is None or F_fit > held[0] + keep_margin(held[0]):
            best_at[len(panel)] = (F_fit, panel, held_fit.summary.label, per_sample, held_fit)

    remember(X, fit, scores)

    def tie_outcomes(kind, label, members, pick, chosen, refits, decide):
        """The decisions the other members of the selected refit's tie class would have given
        (decide(position) -> CountDecision), with the pick's own; (tied candidates, records, agree)."""
        if len(members) < 2:
            return (), None, None, {pick: decide(pick)}
        records = []
        outcomes = {}
        for position in members:
            decision = decide(position)
            outcomes[position] = decision
            records.append(dict(candidate=int(chosen[position]), F=float(refits[position].summary.F),
                                accept_larger=bool(decision.accept_larger),
                                margin=float(decision.margin)))
        agree = len({item["accept_larger"] for item in records}) == 1
        return tuple(int(chosen[p]) for p in members), tuple(records), bool(agree), outcomes

    def cleanup(X, scores, larger_fit):
        """Founders no sample uses, deleted before a deletion round (module docstring). Returns None or
        (record, fit, rows, per-sample scores) of the refitted reduced panel."""
        nonlocal cleanup_index
        quantum = tie_rules.quantum("refine_F")
        current, current_scores, removed = X, scores, []
        while len(current) > 2:
            found = None
            losses = []
            # canonical order = content order (reversed under ties.reverse_conventions())
            for founder in ties.conv_order(len(current)).tolist():
                reduced = unrefitted(current, founder)
                loss = float(np.max(np.abs(view(current_scores) - view(reduced))))
                losses.append(loss)
                if counter is not None:
                    # The band at the exact zero of an unused founder's loss (owner rule A-7).
                    counter.note("assembly.refine_zero_gain.band", inside_tie=loss <= quantum)
                if loss <= quantum and found is None:
                    found = (founder, reduced)
            tie_rules.census("assembly.refine_zero_gain.band", key("clean", cleanup_index, len(removed)), losses)
            if found is None:
                break
            founder, reduced = found
            removed.append(tie_rules.path_key(current[founder]))
            current, current_scores = np.delete(current, founder, axis=0), reduced
        if not removed:
            return None
        label = f"c{cleanup_index}"
        cleanup_index += 1
        refit = engine.converge(current, f"{label}.clean", budget)
        fits.append(refit.summary)
        refit_scores = engine.fit_scores(refit.summary.label, refit.rows, budget)
        if counter is not None:
            counter.add_counts("assembly.refine_cleanup", decisions=1)
        record = CountDecisionRecord(
            kind="cleanup", round=label, founders=len(X), skipped_by_bound=False,
            accept_larger=False, ambiguous=False, refitted=(), selected=None,
            selected_F=float(refit.summary.F), larger_fit=larger_fit, smaller_fit=refit.summary.label,
            cleanup_removed=tuple(removed))
        return record, refit, refit.rows, refit_scores

    def deletion_round(X, scores, larger_fit):
        nonlocal round_index
        K = len(X)
        label = f"r{round_index}"
        round_index += 1
        F = float(scores.sum())
        # Founders without a private cell (q = 0) are decided first, as in the join.
        without = tuple(int(f) for f in np.flatnonzero(~_own_sequence(X)))
        optimistic = band_bound = None
        if not without:
            # Every founder has a private cell: the per-sample bounds of any K - 1 panel also bound
            # every shadow panel, so a bound that proves the total-gain test for every deletion
            # proves the sequence test too; the round is skipped only when the replicate-wise form
            # of the bound also proves the band (the decision's own bootstrap stream).
            with execution.numba_threads(max(1, int(budget.threads()))):
                bounds, _, per_block_sample = upper_bound(ws.leaves, ws.offsets, ws.logs,
                                                          ws.penalty, K - 1)
            per_sample_optimistic = per_block_sample.sum(axis=0)
            optimistic = float(bounds.sum())
            tolerance = 1e-8 * max(1.0, abs(F), abs(optimistic), abs(K * cost - 2.0 * F))
            excluded, band_bound = count_rule.deletion_bound_excludes(
                view(scores), view(per_sample_optimistic), cost, tolerance=tolerance, config=rule,
                seed_key=[*seed_key, label, "deletion"])
            if excluded:
                return CountDecisionRecord(
                    kind="deletion", round=label, founders=K, skipped_by_bound=True,
                    accept_larger=True, ambiguous=False, bound="deletion_per_sample",
                    optimistic_score=optimistic, band_bound=band_bound,
                    larger_fit=larger_fit), None, None, None
        candidates = list(without) or list(range(K))
        if len(candidates) > ranked_refits:
            # Exact unrefitted per-sample scores of every candidate's deletion, ranked by the point
            # margin of the total gain (tie classes; the founder's path inside a class); the top r,
            # with the tie class the cut falls in, are converged.
            keys = [None] * K
            for d in candidates:
                keys[d] = min(_point_margins(view(scores), view(unrefitted(X, d)), cost))
            chosen = _shortlist(candidates, keys, ranked_refits, tie_rules.quantum("count_margin"),
                                [tie_rules.path_key(X[d]) for d in candidates], descending=False,
                                counter=counter, kind="assembly.refine_shortlist", census_key=key(label, "short"))
            ranking = "unrefitted_point_margin"
        else:
            keys, chosen, ranking = None, list(candidates), "all_candidates"
        refits = _fit_many(engine, [np.delete(X, d, axis=0) for d in chosen],
                           [f"{label}.del{d}" for d in chosen], budget)
        fits.extend(refit.summary for refit in refits)
        pick, members = _refit_choice([refit.summary.F for refit in refits],
                                      [tie_rules.path_key(X[d]) for d in chosen], F, counter,
                                      key(label, "deletion"))
        smaller = refits[pick]
        founder = chosen[pick]
        smaller_scores = engine.fit_scores(smaller.summary.label, smaller.rows, budget)
        shadow_scores, _, shadow_cells = _sequence_scores(engine, geometry, X, scores,
                                                          np.delete(X, founder, axis=0), budget)
        decision = count_rule.decide(view(scores), view(smaller_scores), view(shadow_scores), cost,
                                     kind="deletion", config=rule,
                                     seed_key=[*seed_key, label, "deletion"])
        tie_rules.census("assembly.count_margin", key(label, "deletion", "margin"),
                         [decision.total.margin, decision.total.loso_margin, decision.sequence.margin,
                          decision.sequence.loso_margin])

        def decide_member(position):
            if position == pick:
                return decision
            member = refits[position]
            member_scores = engine.fit_scores(member.summary.label, member.rows, budget)
            member_shadow, _, _ = _sequence_scores(engine, geometry, X, scores,
                                                   np.delete(X, chosen[position], axis=0), budget)
            return count_rule.decide(view(scores), view(member_scores), view(member_shadow), cost,
                                     kind="deletion", config=rule,
                                     seed_key=[*seed_key, label, "deletion"])

        tied, tie_records, agree, outcomes = tie_outcomes("deletion", label, members, pick, chosen, refits,
                                                          decide_member)
        resolved, flags = _tied_resolution(members, pick, outcomes, refits, X)
        if resolved != pick:
            pick = resolved
            smaller = refits[pick]
            founder = chosen[pick]
            decision = outcomes[pick]
            smaller_scores = engine.fit_scores(smaller.summary.label, smaller.rows, budget)
            _, _, shadow_cells = _sequence_scores(engine, geometry, X, scores, np.delete(X, founder, axis=0),
                                                  budget)
        hidden = hidden_failures(X, scores, founder) if decision.accept_larger else None
        entry = _decision("deletion", label, K, decision, ranking, keys, chosen, refits, pick, cost,
                          view, larger_fit=larger_fit, smaller_fit=smaller.summary.label,
                          optimistic_score=optimistic, band_bound=band_bound,
                          without_sequence=without, shadow_cells=shadow_cells,
                          hidden_sequence_failures=hidden, tied_candidates=tied,
                          tie_decisions=tie_records, tie_decisions_agree=agree, **flags)
        remember(smaller.rows, smaller, smaller_scores)
        return entry, smaller, smaller_scores, (X, smaller.rows)

    def hidden_failures(X, scores, decided):
        """Diagnostics only: founders other than the decided one whose sequence part fails the
        effective-carrier test (m <= 0 or m' <= 0 on q) at a kept deletion round."""
        failing = []
        for f in range(len(X)):
            if f == decided:
                continue
            shadow_scores, _, _ = _sequence_scores(engine, geometry, X, scores,
                                                   np.delete(X, f, axis=0), budget)
            value, loso = _point_margins(view(scores), view(shadow_scores), cost)
            if not (value > 0. and loso > 0.):
                failing.append((int(f), float(value), float(loso)))
        return tuple(failing)

    def addition_round(X, scores, smaller_fit):
        nonlocal round_index
        K = len(X)
        label = f"r{round_index}"
        round_index += 1
        F = float(scores.sum())
        with execution.numba_threads(max(1, int(budget.threads()))):
            bounds, _, per_block_sample = upper_bound(ws.leaves, ws.offsets, ws.logs, ws.penalty,
                                                      K + 1)
        optimistic = float(bounds.sum())
        tolerance = 1e-8 * max(1.0, abs(F), abs(optimistic), abs(K * cost - 2.0 * F))
        if count_rule.addition_bound_excludes(optimistic, F, cost, count_samples,
                                              tolerance=tolerance):
            return CountDecisionRecord(
                kind="addition", round=label, founders=K, skipped_by_bound=True,
                accept_larger=False, ambiguous=False, bound="addition_total",
                optimistic_score=optimistic, smaller_fit=smaller_fit), None, None, None
        # No sample's optimistic gain reaches the per-carrier cost c/(2S): no seed's refit (nor any
        # (K+1)-panel) can pass or make the call ambiguous, so the round is decided unrun.
        per_sample_optimistic = per_block_sample.sum(axis=0)
        tolerance = 1e-9 * max(1.0, abs(F), float(np.max(np.abs(per_sample_optimistic))))
        excluded, gain_bound, gain_sample = count_rule.addition_per_sample_bound_excludes(
            view(per_sample_optimistic), view(scores), cost, tolerance=tolerance)
        if excluded:
            return CountDecisionRecord(
                kind="addition", round=label, founders=K, skipped_by_bound=True,
                accept_larger=False, ambiguous=False, bound="addition_per_sample",
                optimistic_score=optimistic, per_sample_gain_bound=float(gain_bound),
                per_sample_gain_bound_sample=view.row(gain_sample),
                smaller_fit=smaller_fit), None, None, None
        starts = [np.vstack((X, X[f:f + 1])) for f in range(K)]
        responses = gains = None
        if K > ranked_refits:
            # Seeds ranked by the exact F gain of their copy's best responses (tie classes; the
            # seed's path inside a class); the top r, with the tie class the cut falls in, are
            # converged.
            states = (K + 1) * (K + 2) // 2
            per_task = 24 * (engine.blocks + 1) * samples * states + (32 << 20)
            responses = [None] * K
            for index, response in budget.concurrent(
                    [lambda child, start=start: engine.copy_response(start, child)
                     for start in starts], per_task, label="founder_refinement copy responses"):
                responses[index] = response
            gains = [response[0] for response in responses]
            chosen = _shortlist(list(range(K)), gains, ranked_refits, tie_rules.quantum("refine_F", F),
                                [tie_rules.path_key(X[f]) for f in range(K)], descending=True,
                                counter=counter, kind="assembly.refine_shortlist", census_key=key(label, "short"))
            ranking = "copy_best_response_gain"
        else:
            chosen, ranking = list(range(K)), "all_candidates"
        refits = _fit_many(engine, [starts[f] for f in chosen], [f"{label}.add{f}" for f in chosen],
                           budget)
        fits.extend(refit.summary for refit in refits)
        pick, members = _refit_choice([refit.summary.F for refit in refits],
                                      [tie_rules.path_key(X[f]) for f in chosen], F, counter,
                                      key(label, "addition"))
        larger = refits[pick]
        larger_scores = engine.fit_scores(larger.summary.label, larger.rows, budget)
        # The refit may have moved any founder: its sequence part is the gain over the refit with
        # every cell the incumbent does not use replaced by the incumbent's nearest row.
        shadow_scores, without, shadow_cells = _sequence_scores(engine, geometry, larger.rows,
                                                                larger_scores, X, budget)
        decision = count_rule.decide(view(larger_scores), view(scores), view(shadow_scores), cost,
                                     kind="addition", config=rule,
                                     seed_key=[*seed_key, label, "addition"])
        tie_rules.census("assembly.count_margin", key(label, "addition", "margin"),
                         [decision.total.margin, decision.total.loso_margin, decision.sequence.margin,
                          decision.sequence.loso_margin])

        def decide_member(position):
            if position == pick:
                return decision
            member = refits[position]
            member_scores = engine.fit_scores(member.summary.label, member.rows, budget)
            member_shadow, _, _ = _sequence_scores(engine, geometry, member.rows, member_scores, X,
                                                   budget)
            return count_rule.decide(view(member_scores), view(scores), view(member_shadow), cost,
                                     kind="addition", config=rule,
                                     seed_key=[*seed_key, label, "addition"])

        tied, tie_records, agree, outcomes = tie_outcomes("addition", label, members, pick, chosen, refits,
                                                          decide_member)
        resolved, flags = _tied_resolution(members, pick, outcomes, refits, X)
        if resolved != pick:
            pick = resolved
            larger = refits[pick]
            decision = outcomes[pick]
            larger_scores = engine.fit_scores(larger.summary.label, larger.rows, budget)
            _, without, shadow_cells = _sequence_scores(engine, geometry, larger.rows, larger_scores, X,
                                                        budget)
        entry = _decision("addition", label, K, decision, ranking, gains, chosen, refits, pick, cost,
                          view, larger_fit=larger.summary.label, smaller_fit=smaller_fit,
                          optimistic_score=optimistic, without_sequence=without,
                          shadow_cells=shadow_cells,
                          copy_responses=None if responses is None else tuple(
                              response[1] for response in responses),
                          tied_candidates=tied, tie_decisions=tie_records, tie_decisions_agree=agree,
                          **flags)
        remember(larger.rows, larger, larger_scores)
        return entry, larger, larger_scores, (larger.rows, X)

    final = fit
    # Founder count at the start and after every accepted count change, and the deletions of a
    # founder without a private cell or without a user (cleanup, not a decision): cycle diagnostics.
    sizes = [incoming]
    cleanup_steps = set()
    # Deletion rounds until one is rejected; panels of two founders are not reduced. Founders no
    # sample uses are cleaned up before every round.
    while len(X) > 2:
        cleaned = cleanup(X, scores, final.summary.label)
        if cleaned is not None:
            entry, final, X, scores = cleaned
            decisions.append(entry)
            panels.append(None)
            remember(X, final, scores)
            sizes.append(len(X))
            cleanup_steps.add(len(sizes) - 1)
            if len(X) <= 2:
                break
        entry, smaller, smaller_scores, pair = deletion_round(X, scores, final.summary.label)
        decisions.append(entry)
        panels.append(pair)
        if entry.accept_larger:
            break
        final, X, scores = smaller, smaller.rows, smaller_scores
        sizes.append(len(X))
        if entry.selected in entry.without_sequence:
            cleanup_steps.add(len(sizes) - 1)
    cycle = False
    cycle_undecided = False
    while additions:
        if len(X) >= upper:
            limit_reached = True
            break
        entry, larger, larger_scores, pair = addition_round(X, scores, final.summary.label)
        decisions.append(entry)
        panels.append(pair)
        if not entry.accept_larger:
            break
        check, smaller, smaller_scores, check_pair = deletion_round(larger.rows, larger_scores,
                                                                    larger.summary.label)
        check = replace(check, after_addition=True)
        decisions.append(check)
        panels.append(check_pair)
        if check.accept_larger:
            final, X, scores = larger, larger.rows, larger_scores
            sizes.append(len(X))
            continue
        # Adding then deleting: a count cycle. The deletion round stands (the K+1 panel is not
        # supported against its own best K refit) and the better-F of the two K panels (the
        # incumbent and that refit) is kept; the call is reported as ambiguous, and as undecided
        # when the deletion round's call is ambiguous and the cycle began with a decided deletion
        # from K + 1.
        cycle = True
        origin = count_rule.pre_cycle_count(sizes + [len(larger.rows)], len(larger.rows),
                                            cleanup_steps)
        cycle_undecided = bool(origin >= len(larger.rows) and check.ambiguous)
        if float(smaller_scores.sum()) > float(scores.sum()) + keep_margin(float(scores.sum())):
            final, X, scores = smaller, smaller.rows, smaller_scores
        flags = dict(ambiguous=True, count_cycle=True, cycle_start_count=int(origin),
                     cycle_undecided=cycle_undecided)
        decisions[-2] = replace(entry, **flags)
        decisions[-1] = replace(check, **flags)
        break
    replaced = None
    held = best_at[len(X)]
    if held[0] > float(scores.sum()) + keep_margin(float(scores.sum())):
        # The pass returned to a K it had held with a better converged panel: the count decisions
        # chose K, the fixed-K objective chooses the panel within K.
        gain = float(held[0] - float(scores.sum()))
        from_fit = final.summary.label
        final, X, scores = held[4], held[1], held[3]
        comparisons = _restored_comparisons(engine, geometry, budget, X, scores, best_at,
                                            decisions, panels, cost, rule, seed_key, view)
        replaced = BestFReplacement(from_fit=from_fit, to_fit=final.summary.label, gain=gain,
                                    **comparisons)
    if limit_reached:
        warnings.warn(f"founder refinement {engine.name}: additions reached the count limit "
                      f"K = {upper}", RuntimeWarning)
    return CountPass(
        rows=X, scores=scores, F=float(scores.sum()), founders_before=int(incoming),
        fits=tuple(fits), decisions=tuple(decisions), final_fit=final.summary.label,
        weak_junctions=final.weak_junctions, count_cycle=cycle, cycle_undecided=cycle_undecided,
        count_limit=int(upper), count_limit_reached=limit_reached, additions=bool(additions),
        ambiguous=(any(d.ambiguous for d in decisions)
                   or bool(replaced is not None and replaced.ambiguous)),
        best_f_replacement=replaced)
