"""Count pass of one component: converged count decisions between converged fixed-K fits.

Count rule (assembly/count_rule, shared with the join): a decision between a larger panel X+ and a
smaller X- keeps X+ iff the effective-carrier test passes on the total per-sample gain
d_s = F_s(X+) - F_s(X-) AND on its sequence part q_s = F_s(X+) - F_s(X+^sigma). The shadow X+^sigma
replaces every cell (block, row) of X+ that a reference panel does not use by the reference's row
nearest to it at that block; the reference is X+ minus the decided founder f for a deletion (only
f's private cells change) and the incumbent X- for an addition (every cell the refit added). A
founder without a private cell (every cell another founder uses at that block) owns no sequence:
q = 0 exactly when X+ holds one. Otherwise q is the exact shadow difference; equally near
replacements form a complete factored family. Families within the separate shadow budget
are evaluated completely; only opposing evaluated verdicts invoke the smaller-K prior.
Overflow preserves the incumbent shadow verdict and records count_unverified without
creating biological alternatives or withholding calls. Unlike the join,
refinement takes no near-identity shortcut, because its last level is final and no later level can
re-split a merged pair. Switch savings alone never justify a founder.

Pass: X_K = C(start); deletion rounds (founders without a private cell are the round's candidates
when there are any, otherwise every founder; with more than r = ranked_refits candidates the r
best-ranked by the point margin of their exact unrefitted deletion are refitted, else every one;
the best-F refit is decided; a rejection ends the deletions; K >= 2).

Ties and cleanup (release 2.1, design 5.3; fits are in canonical founder order, so founder indices are
content ranks): the refit decided is the content-smallest member (deleted founder's or copied seed's
path) of the best tie class of the refits' F (quantum tie_rules.quantum("refine_F", F)); the decision is
also evaluated for every other member of that class (complete shadow family when budgeted,
otherwise its incumbent shadow only) and recorded
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
import time
import itertools
import math
import hashlib

import numpy as np
from numba import njit, prange

from .. import count_rule
from ..count_rule.geometry import shadow_choices
from ..count_rule import shadow_bounds
from .. import tie_rules
from ...maths import ties
from .acceptance import TOL
from .objective import CENTER, forward_sites, row_max, unordered_pairs
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
    shadow_alternatives: complete nearest-row factors, evaluated verdicts and completeness per
    compared tied refit. Overflow retains the incumbent's evaluated verdict and margin;
    count_unverified records its unevaluated domain separately from biological ambiguity.
    hidden_sequence_failures: at a kept deletion round, the other founders whose sequence part
    fails ((founder, margin, loso_margin)). copy_responses: the seeds' copy best responses.
    after_addition: the deletion round that checks an accepted addition. count_cycle,
    cycle_start_count, cycle_undecided: the cycle flags of the two rounds of a count cycle.
    larger_fit, smaller_fit: labels of the two fits compared.
    tied_candidates: the candidates of the selected refit's tie class (release 2.1), in decision order
    (selected first); competitive_fits: every fit label in the winning F/margin class
    before the representative prior, including coincident primary panels; their tied
    direction variants require full count continuation. tie_decisions: per member (candidate, F, accept_larger, margin) of the decision it
    would have given; tie_decisions_agree: all members give the same point decision (None without a
    tie). Kind "cleanup" (release 2.1): founders no sample uses deleted before a deletion round
    (cleanup_removed: their row paths; selected: None; accept_larger False; not a count decision).
    Release 2.2 (tied refits decided by data, else a stated prior): ambiguous_k: the members of the
    selected refit's tie class disagree on the count decision, which then takes the smaller K (prior);
    alternatives: the distinct resulting panels (panel keys, sorted) equally supported by the data when
    the decision changes the panel, released flagged (empty when the choice is unique); prior: the rule
    that decided ("smaller_k", "margin": the member whose count margin most strongly supports the decision
    taken, "symmetric": a remaining exact tie, decided by the canonical panel key and flagged).
    fit_family_evaluations: complete observed fixed-K endpoint table when any refit
    has tied endpoints. Each row retains candidate origin, original reference, endpoint,
    F, evaluated criterion and count-margin eligibility. selected_refit_panel identifies
    the selected endpoint independently of the shared origin label. Rejected endpoints
    cannot reappear through an earlier direction-tie family.
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
    shadow_alternatives: tuple = ()
    count_unverified: tuple = ()
    hidden_sequence_failures: tuple | None = None
    copy_responses: tuple | None = None
    after_addition: bool = False
    count_cycle: bool = False
    cycle_start_count: int | None = None
    cycle_undecided: bool = False
    larger_fit: str | None = None
    smaller_fit: str | None = None
    tied_candidates: tuple = ()
    competitive_fits: tuple = ()
    fit_family_evaluations: tuple = ()
    selected_refit_panel: tuple = ()
    tie_decisions: tuple | None = None
    tie_decisions_agree: bool | None = None
    cleanup_removed: tuple = ()
    ambiguous_k: bool = False
    alternatives: tuple = ()
    unresolved_panels: tuple = ()
    prior: str | None = None
    continuation_key: str | None = None
    continuation_replayed: bool = False


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
    live_alternatives: tuple = ()
    unresolved_alternatives: tuple = ()
    continuation_families: tuple = ()
    count_unverified: tuple = ()
    branch_evaluations: int = 1
    branch_reused_fits: int = 0
    branch_seconds: float = 0.0
    branch_status: str = "none"
    selected_branch: str | None = None
    input_F0: float | None = None
    count_continuations: tuple = ()
    count_branch_evaluations: int = 0
    count_branch_seconds: float = 0.0


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


def _shadow_options(geometry, larger, reference):
    """Complete factored nearest-row family over every replaced founder cell."""
    cover = count_rule.covered_rows(np.asarray(reference, np.int64))
    choices, replaced = [], 0
    for path in larger:
        options, lost = shadow_choices(path, cover, geometry)
        choices.extend(options)
        replaced += lost
    return tuple(choices), replaced


def _shadow(geometry, larger, reference):
    """The incumbent's conventional shadow, retained when its family cannot be completed."""
    cover = count_rule.covered_rows(np.asarray(reference, np.int64))
    paths = [count_rule.shadow_path(path, cover, geometry) for path in larger]
    return np.asarray([path for path, _ in paths], np.int64), sum(lost for _, lost in paths)


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


def _shadow_decision(engine, geometry, larger, larger_scores, smaller_scores, reference,
                     budget, cost, rule, seed_key, kind, view, *, decision_options=None):
    """Complete the nearest-shadow family, or preserve the incumbent's own verdict.

    The separate computational budget never supplies count evidence: overflow
    evaluates no other shadow and leaves the evaluated decision unchanged. Only
    actual opposed verdicts invoke smaller K. All scores use one mask/seed stream;
    existing no-private-cell q=0 and optimistic round bounds remain exact shortcuts.
    """
    begun = time.perf_counter()
    without = tuple(int(f) for f in np.flatnonzero(~_own_sequence(larger)))
    if without:
        decision = count_rule.decide(view(larger_scores), view(smaller_scores), view(larger_scores),
                                     cost, kind=kind, config=rule, seed_key=seed_key)
        return decision, without, 0, dict(combinations=1, evaluated=0, complete=True,
                                         unresolved=False, no_private_sequence=True,
                                         seconds=time.perf_counter() - begun)
    choices, replaced = _shadow_options(geometry, larger, reference)
    if not replaced:
        decision = count_rule.decide(view(larger_scores), view(smaller_scores), view(larger_scores),
                                     cost, kind=kind, config=rule, seed_key=seed_key)
        return decision, without, 0, dict(combinations=1, evaluated=0, complete=True,
                                         unresolved=False, seconds=time.perf_counter() - begun)
    combinations = math.prod(map(len, choices))
    total_passes = count_rule.passes(view(larger_scores - smaller_scores), cost,
                                    len(view(larger_scores)))
    # A complete small family also supplies invariant count margins for ranking
    # tied refits, even if the total-gain test alone rejects the larger count.
    complete = combinations <= int(rule.shadow_combinations)
    paths = (itertools.product(*choices) if complete else
             [_shadow(geometry, larger, reference)[0].ravel()])
    family = (count_rule.ShadowDecisionFamily(view(larger_scores), view(smaller_scores), cost,
                                            kind=kind, config=rule, seed_key=seed_key)
              if complete and combinations > 1 else None)
    decisions = []
    for cells in paths:
        rows = np.asarray(cells, np.int64).reshape(larger.shape)
        shadow_scores = engine.per_sample(rows, budget)
        decisions.append(count_rule.decide(view(larger_scores), view(smaller_scores), view(shadow_scores),
                                          cost, kind=kind, config=rule, seed_key=seed_key)
                         if family is None else family.decide(view(shadow_scores)))
    if decision_options is not None:
        decision_options.extend(decisions)
    verdicts = {decision.accept_larger for decision in decisions}
    unresolved = total_passes and len(verdicts) > 1
    decision = min(decisions, key=lambda value: (value.accept_larger, value.margin,
                                                value.sequence.loso_margin))
    if unresolved:
        decision = replace(decision, accept_larger=False, ambiguous=True)
    details = dict(nearest_choices=choices, combinations=combinations, evaluated=len(decisions),
                   complete=complete, resolved_by_total=not total_passes, unresolved=unresolved,
                   count_unverified=not complete,
                   verdicts=tuple(value.accept_larger for value in decisions),
                   sequence_margins=tuple(value.sequence.margin for value in decisions),
                   sequence_loso_margins=tuple(value.sequence.loso_margin for value in decisions),
                   seconds=time.perf_counter() - begun)
    if not complete:
        # The ordinary decision and its nominal sequence diagnostics stay intact.
        # An optional full-domain proof updates only computational verification.
        from .search import PAIR_TABLE_BYTES
        ws = engine.workspace
        threads = max(1, int(budget.threads()))
        def upper_scores(plan):
            with execution.numba_threads(threads):
                scores = shadow_bounds.full_site_envelope(
                    ws.leaves, ws.offsets, ws.logs, plan, ws.penalty, CENTER)
            return view(scores)
        proof, audit = shadow_bounds.attempt_certificate(
            choices, larger.shape, np.diff(ws.offsets), view(larger_scores), view(smaller_scores),
            cost, incumbent=decision, config=rule, seed_key=seed_key, upper_scores=upper_scores,
            scorer_samples=ws.samples, threads=threads, full_site=True,
            max_table_bytes=PAIR_TABLE_BYTES)
        details.update(criterion_verified=proof is not None, proof_attempt=audit,
                       sequence_diagnostics='evaluated_incumbent_only')
        if proof is not None:
            details.update(count_unverified=False, count_verified_by_bound=proof)
        details['seconds'] = time.perf_counter() - begun
    return decision, without, int(replaced), details


def _unverified_shadow(details, scope):
    """Unevaluated computational work; empty panels prevent biological tie interpretation."""
    return dict(status="count_unverified", count_unverified=True, unresolved_search=True,
                cause="nearest_shadow_budget", reason="count_unverified",
                domain_size=int(details["combinations"]), evaluated=int(details["evaluated"]),
                scope=scope, panels=(), representative_prior=None)


def _shadow_flags(flags, details, refits, X):
    """Retain opposing count outcomes of every actually compared tied refit."""
    uncertain = [position for position, value in details.items() if value['unresolved']]
    if not uncertain:
        return flags
    panels = set(flags.get('unresolved_panels', ()))
    panels.add(tie_rules.panel_key(X))
    panels.update(tie_rules.panel_key(refits[position].rows) for position in uncertain)
    return dict(flags, ambiguous_k=True, unresolved_panels=tuple(sorted(panels)),
                prior='smaller_k+nearest_shadow')


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


def _count_refit_endpoints(engine, chosen, refits):
    """Complete observed fixed-K endpoints with their original count references.

    Convergence already computed both directions of each tied fit. Count-margin
    selection must see this family before a naming convention picks its primary.
    Expansion adds no search or continuation leaf. Exact Fit variants retain
    their own weak-junction records. Only surviving count outputs may continue;
    the old pre-count direction family cannot re-enter terminal alternatives.
    """
    if not any(fit.summary.tied_panels for fit in refits):
        return chosen, refits, False
    origins, endpoints = [], []
    for candidate, fit in zip(chosen, refits):
        label = fit.summary.label
        # A label-only pending vector belongs to the primary, not a companion.
        pending = engine._fit_scores.pop(label, None)
        if pending is not None:
            engine._score_cache[tie_rules.panel_key(fit.rows)] = pending
        variants = (fit,)
        if fit.summary.tied_panels:
            token = (label, engine._fit_invocations[label])
            variants = engine._fit_cache[token]
        for variant in sorted(variants, key=lambda item: tie_rules.panel_key(item.rows)):
            origins.append(candidate)
            endpoints.append(replace(variant, summary=replace(variant.summary,
                tied_panels=(), tied_scores=(), tied_switches=())))
    return origins, endpoints, True


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


def _tied_resolution(members, pick, outcomes, refits, X, engine=None, budget=None,
                     shadow_details=None):
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
        return pick, dict(competitive_fits=(refits[pick].summary.label,))
    taken_larger = {position: bool(outcomes[position].accept_larger) for position in members}
    disagree = len(set(taken_larger.values())) > 1
    # accept_larger False = the smaller panel (a deletion accepted, an addition rejected)
    candidates = [p for p in members if not taken_larger[p]] if disagree else list(members)
    larger_side = taken_larger[candidates[0]]

    def panel_of(position):
        # The panel the decision leaves: the member's refit (K + 1 rows for an addition, K - 1 for a deletion)
        # when the decision takes it, else the incumbent X.
        refit_rows = np.asarray(refits[position].rows)
        larger = taken_larger[position]
        takes_refit = larger if len(refit_rows) > len(X) else not larger
        return tie_rules.panel_key(refit_rows if takes_refit else X)

    margins = [float(outcomes[p].margin) for p in candidates]
    contents = [panel_of(p) for p in candidates]
    quantum = tie_rules.quantum("count_margin")
    result = ties.tie_classes(margins, quantum, content=contents, descending=larger_side)
    winner = candidates[result.winner]
    tied = [candidates[i] for i in result.members(int(result.classes[result.winner]))]
    distinct = sorted({panel_of(p) for p in tied})
    flags = dict(ambiguous_k=bool(disagree), prior="smaller_k" if disagree else "margin",
                 competitive_fits=tuple(sorted(refits[p].summary.label for p in tied)))
    if disagree:
        # These are the actual opposing count outcomes, before the smaller-K prior.
        flags["unresolved_panels"] = tuple(sorted(
            set(flags.get("unresolved_panels", ())) | {panel_of(p) for p in members}))
    if len(distinct) > 1:
        flags["alternatives"] = tuple(distinct)
        if engine is not None:
            counts = {panel: engine.minimum_switches(np.asarray(panel, np.int64), budget)
                      for panel in distinct}
            minimum = min(counts.values())
            eligible = [p for p in tied if counts[panel_of(p)] == minimum]
            winner = ties.conv_sorted(eligible, key=panel_of)[0]
            flags["prior"] += "+fewest_switches"
        else:
            flags["prior"] += "+symmetric"
    return winner, flags


def _eligible_count_records(X, members, outcomes, refits, shadow_options, shadow_details):
    """Actual evaluated outcomes in each verdict's winning count-margin class."""
    def panel_of(position, decision):
        rows = refits[position].rows
        take = decision.accept_larger if len(rows) > len(X) else not decision.accept_larger
        return tie_rules.panel_key(rows if take else X)

    records = []
    for position in members:
        observed = shadow_options.get(position, ())
        if shadow_details[position].get('unresolved') and observed:
            for verdict in sorted({bool(d.accept_larger) for d in observed}):
                decision = min((d for d in observed if bool(d.accept_larger) == verdict),
                               key=lambda d: (d.margin, d.sequence.loso_margin))
                records.append((position, decision, panel_of(position, decision)))
        else:
            decision = outcomes[position]
            records.append((position, decision, panel_of(position, decision)))
    eligible = []
    for verdict in sorted({bool(item[1].accept_larger) for item in records}):
        group = [item for item in records if bool(item[1].accept_larger) == verdict]
        values = [float(item[1].margin) for item in group]
        classes = ties.tie_classes(values, tie_rules.quantum('count_margin'),
                                   content=[item[2] for item in group], descending=verdict)
        eligible.extend(group[int(i)] for i in classes.members(int(classes.classes[classes.winner])))
    return records, eligible


def _endpoint_evaluations(kind, X, chosen, refits, members, outcomes, shadow_options,
                          shadow_details, selected, view):
    """Complete observed endpoint audit, before any representative convention."""
    records, eligible = _eligible_count_records(
        X, members, outcomes, refits, shadow_options, shadow_details)
    accepted = {(p, bool(d.accept_larger)) for p, d, _ in eligible}
    table = []
    for position, (origin, fit) in enumerate(zip(chosen, refits)):
        evaluated = position in outcomes
        reference = np.delete(X, origin, axis=0) if kind == 'deletion' else X
        table.append(dict(candidate=int(origin), fit=fit.summary.label,
            endpoint=tie_rules.panel_key(fit.rows), F=float(fit.summary.F),
            reference=tie_rules.panel_key(reference), best_F=evaluated,
            selected_refit=position == selected,
            criteria=tuple(dict(criterion=view.record(d), resulting_panel=panel,
                margin_eligible=(position, bool(d.accept_larger)) in accepted)
                for p, d, panel in records if p == position)))
    return tuple(table)


def _count_choice(engine, kind, label, X, members, resolved, outcomes, refits,
                  shadow_options, shadow_details, flags):
    """Eligible evaluated decision outputs, selected before replaying ordinary history.

    Compare F in the caller, then the existing count-margin class separately on
    each actually observed verdict. A computational shadow overflow supplies only
    its nominal criterion. Exact opposed shadow verdicts supply actual criteria,
    never a manufactured verdict. Within one verdict, the shadow representative
    remains its least margin, as in _shadow_decision.
    """
    def panel_of(position, decision):
        rows = refits[position].rows
        take = decision.accept_larger if len(rows) > len(X) else not decision.accept_larger
        return tie_rules.panel_key(rows if take else X)

    _, eligible = _eligible_count_records(
        X, members, outcomes, refits, shadow_options, shadow_details)
    nominal = panel_of(resolved, outcomes[resolved])
    options = {}
    for position, decision, panel in eligible:
        options.setdefault(panel, (position, decision))
        if position == resolved and panel == nominal:
            options[panel] = position, outcomes[resolved]
    if len(options) < 2:
        return resolved, outcomes[resolved], flags
    # The incumbent's normal margin/prior choice must belong to the complete family.
    assert nominal in options
    encoded = np.ascontiguousarray(X, np.int64)
    token = hashlib.sha256(repr((kind, label, encoded.shape)).encode() + encoded.tobytes()).hexdigest()
    wanted = getattr(engine, '_forced_count_choices', {}).get(token, nominal)
    if wanted not in options:
        raise ValueError('Count replay choice is not an eligible evaluated outcome')
    position, decision = options[wanted]
    panels = tuple(sorted(options))
    source = dict(panels=panels, source_decision=label, source_count_kind=kind,
                  source_count_key=token, selected_panel=wanted,
                  representative_prior=flags.get('prior'),
                  source_reason=('nearest_shadow_count' if any(d.get('unresolved')
                                 for d in shadow_details.values()) else 'tied_count_candidates'),
                  status='ambiguous_k' if len({len(p) for p in panels}) > 1 else 'same_k_tie')
    engine._count_families[token] = source
    engine._chosen_count_panels[token] = wanted
    # A direction tie in any fit from the selected verdict's winning margin
    # class still needs its own continuation; another verdict is another branch.
    competitive = tuple(sorted({refits[p].summary.label for p, d, _ in eligible
                                if bool(d.accept_larger) == bool(decision.accept_larger)}))
    flags = dict(flags, competitive_fits=competitive,
                 ambiguous_k=len({len(p) for p in panels}) > 1,
                 unresolved_panels=panels if len({len(p) for p in panels}) > 1 else (),
                 alternatives=panels if len({len(p) for p in panels}) == 1 else (),
                 continuation_key=token, continuation_replayed=wanted != nominal)
    if wanted != nominal:
        flags['prior'] = None  # exhaustive continuation is not a biological prior
    return position, decision, flags


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
        old = candidate_F[candidate]
        candidate_F[candidate] = float(value) if old is None else max(old, float(value))
    others = sorted((v for i, v in enumerate(values) if i != pick), reverse=True)
    return CountDecisionRecord(
        kind=kind, round=label, founders=int(founders), skipped_by_bound=False,
        accept_larger=bool(decision.accept_larger),
        ambiguous=bool(decision.ambiguous or extra.get("ambiguous_k")),
        margin=float(decision.margin), band=tuple(float(x) for x in decision.band),
        ranking=ranking,
        candidate_scores=None if scores is None else tuple(None if v is None else float(v)
                                                           for v in scores),
        refitted=tuple(dict.fromkeys(int(c) for c in chosen)), candidate_F=tuple(candidate_F),
        selected=int(chosen[pick]), selected_F=float(values[pick]),
        runner_up_margin_over_cost=None if not others else float((values[pick] - others[0]) / cost),
        criterion=view.record(decision), larger_fit=larger_fit, smaller_fit=smaller_fit,
        count_unverified=tuple(_unverified_shadow(item, dict(kind=kind, round=label,
                              candidate=item["candidate"], founders=int(founders),
                              **({"fit_endpoint": item["fit_endpoint"]}
                                 if "fit_endpoint" in item else {})))
                               for item in extra.get("shadow_alternatives", ())
                               if item.get("count_unverified")), **extra)


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
        decision, _, _, shadow_details = _shadow_decision(
            engine, geometry, larger, larger_scores, smaller, reference, budget, cost, rule,
            [*seed_key, "best_f", kind], kind, view)
        return dict(fit=held[2], founders=int(other), F=float(held[0]),
                    accept_larger=bool(decision.accept_larger), margin=float(decision.margin),
                    loso_margin=_loso_margin(decision.record()),
                    sequence_margin=float(decision.sequence.margin), shadow=shadow_details,
                    panels=tuple(sorted({tie_rules.panel_key(rows), tie_rules.panel_key(held[1])})),
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


def _competitive_families(entry, candidate_fits):
    """Direction families of refits in the actual best-F/best-margin class.

    A candidate whose primary coincides with another candidate is still part of
    that class. Its tied direction must face its own count shadow/margin before
    selection; replaying from the original input performs that comparison. This
    does not admit lower-F or margin-rejected historical fits.
    """
    if getattr(entry, "fit_family_evaluations", ()):
        # Complete endpoint comparison has applied the count-margin criterion;
        # its surviving alternatives are now count-choice histories.
        return ()
    labels = set(entry.competitive_fits)
    return tuple(dict(panels=tuple(fit.summary.tied_panels), source_fit=fit.summary.label,
                      source_count_round=entry.round, source_role="competitive_refit",
                      representative_prior="fewest_switches")
                 for fit in candidate_fits
                 if fit.summary.label in labels and len(fit.summary.tied_panels) > 1)


def _terminal_candidates(candidate_fits, ancestry, ancestry_labels, best_fit):
    """Preserve actual count-margin decisions when collecting final competitors."""
    allowed_panels = {tie_rules.panel_key(np.asarray(panel))
                      for family in ancestry for panel in
                      ((family['selected_panel'],) if family.get('source_count_key')
                       else family["panels"])}
    eligible_fits = []
    for candidate in candidate_fits:
        selected = (candidate.summary.label, tie_rules.panel_key(candidate.rows))
        if (candidate.summary.label in ancestry_labels or selected in ancestry_labels
                or candidate is best_fit):
            eligible_fits.append(candidate)
        elif tie_rules.panel_key(candidate.rows) in allowed_panels:
            eligible_fits.append(replace(candidate, summary=replace(
                candidate.summary, tied_panels=(), tied_scores=(), tied_switches=())))
    return eligible_fits


def _terminal_family(engine, candidate_fits, final, scores, budget):
    """Current same-K equal-F candidates, rescored together; discarded histories stay out.

    Every candidate is a converged fit in this workspace, including a fit's two
    terminal direction results. Only candidates in the returned panel's F class
    qualify; an old fit tied with itself at lower F does not qualify.
    """
    returned = tie_rules.panel_key(final.rows)
    total = float(scores.sum())
    candidates = {returned: (total, final)}
    for fit in candidate_fits:
        if len(fit.rows) != len(final.rows):
            continue
        panel = tie_rules.panel_key(fit.rows)
        if panel not in candidates or candidates[panel][1] is None:
            candidates[panel] = (float(fit.summary.F), fit)
        tied = getattr(fit.summary, "tied_panels", ())
        values = getattr(fit.summary, "tied_scores", ())
        for i, panel in enumerate(tied):
            panel = tie_rules.panel_key(np.asarray(panel))
            # Old fit records lack the panel/score correspondence. They are scored
            # afresh only when they could belong to the returned objective class.
            value = float(values[i]) if len(values) == len(tied) else float(fit.summary.F)
            candidates.setdefault(panel, (value, None))
    scale = max(abs(item[0]) for item in candidates.values())
    quantum = tie_rules.quantum("refine_F", scale)
    keys = sorted(candidates)
    values = [candidates[key][0] for key in keys]
    classes = ties.tie_classes(values, quantum, content=keys, descending=True)
    current_class = int(classes.classes[keys.index(returned)])
    possible = [keys[int(i)] for i in classes.members(current_class)]
    if len(possible) < 2:
        return final, scores, ()
    exact = {returned: scores}
    for key in possible:
        if key != returned:
            exact[key] = engine.per_sample(np.asarray(key, np.int64), budget)
    values = [float(exact[key].sum()) for key in possible]
    scale = max(abs(value) for value in values)
    classes = ties.tie_classes(values, tie_rules.quantum("refine_F", scale),
                               content=possible, descending=True)
    current_class = int(classes.classes[possible.index(returned)])
    live = tuple(sorted(possible[int(i)] for i in classes.members(current_class)))
    if len(live) < 2:
        return final, scores, ()
    switches = {key: engine.minimum_switches(np.asarray(key, np.int64), budget) for key in live}
    # converge() already chooses this prior within a two-direction tie, so at
    # least one actual Fit supplies a minimum-count representative and its weak
    # junctions. Do not attach another panel's weak-junction diagnostics.
    eligible = [key for key in live if candidates[key][1] is not None]
    best = min(switches[key] for key in eligible)
    chosen = ties.conv_sorted([key for key in eligible if switches[key] == best])[0]
    family = dict(panels=live, scores=tuple(float(exact[key].sum()) for key in live),
                  minimum_switches=tuple(switches[key] for key in live),
                  reason="same_k_equal_F", representative_prior="fewest_switches",
                  unresolved_search=False)
    return candidates[chosen][1], exact[chosen], (family,)


def _uncontinued_families(ancestry, founders, live_panels=()):
    """Only selected-path families whose alternative count continuation is unrun.

    Same-K candidates have a common final objective and are handled by
    _terminal_family. Different-K source panels cannot be called terminal ties.
    """
    result = []
    seen = set()
    for family in ancestry:
        panels = tuple(sorted({tie_rules.panel_key(np.asarray(panel))
                               for panel in family["panels"]}))
        if len(panels) < 2:
            continue
        if not family.get("source_count_key") and all(len(panel) == founders for panel in panels):
            # A demonstrated final-F gap can discard an old choice. A remaining
            # symmetric count-choice family has not had its later count tests
            # replayed merely because its old output panels still tie on F.
            if not family.get("source_decision") or len(set(panels).intersection(live_panels)) < 2:
                continue
        if panels in seen:
            continue
        seen.add(panels)
        result.append(dict(family, panels=panels, reason="unresolved_count_continuation",
                           unresolved_search=True))
    return tuple(result)


# ------------------------------------------------------------------ the pass
def _count_pass_point(engine, geometry, start, cost, rule, budget, seed_key, additions, counter=None,
                      census_prefix=None):
    """Converged fixed-K fit and converged count decisions for one component (module docstring).

    engine: the component's search.SweepEngine; geometry: count_rule.RowGeometry of its prepared
    panels; start: int64 K x B starting rows; cost: the per-founder cost c; rule: CountRuleConfig;
    seed_key: the component's bootstrap key; additions: run addition rounds (the final count step);
    counter: the component's tie-class counter (tie_rules.SharedCounter) or None; census_prefix: its
    valley-census key prefix or None. Returns a CountPass."""
    engine._count_families = {}
    engine._chosen_count_panels = {}
    ranked_refits = int(rule.ranked_refits)
    ws = engine.workspace
    # Per-sample vectors enter the count rule restricted to the informative samples (cost's S);
    # totals F stay sums over every evidence row (the other rows add exact zeros).
    view = _CountSamples(ws.count_samples)
    incoming = len(start)
    fits, decisions, panels = [], [], []
    candidate_fits, ancestry = [], []
    ancestry_labels = set()

    def capture(fit):
        fits.append(fit.summary)
        candidate_fits.append(fit)

    def adopt(fit):
        ancestry_labels.add((fit.summary.label, tie_rules.panel_key(fit.rows)))
        tied = getattr(fit.summary, "tied_panels", ())
        if len(tied) > 1:
            ancestry.append(dict(panels=tuple(tied), source_fit=fit.summary.label,
                                 representative_prior="fewest_switches"))

    def decided(entry):
        ancestry.extend(_competitive_families(entry, candidate_fits))
        source = engine._count_families.get(entry.continuation_key)
        if source is not None:
            ancestry.append(dict(source, shadow_alternatives=entry.shadow_alternatives))
        else:
            family = entry.unresolved_panels or entry.alternatives
            if len(family) > 1:
                ancestry.append(dict(panels=tuple(family), source_decision=entry.round,
                                     representative_prior=entry.prior,
                                     source_reason=("nearest_shadow_count" if any(
                                         item.get("unresolved") for item in entry.shadow_alternatives)
                                                    else "tied_count_candidates"),
                                     shadow_alternatives=entry.shadow_alternatives,
                                     status="ambiguous_k" if entry.ambiguous_k else "same_k_tie"))

    fit = engine.converge(start, "start", budget)
    capture(fit)
    adopt(fit)
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
        capture(refit)
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
        # Keep one work summary per actual convergence; the endpoint table
        # expands evidence already computed, not the number of searches run.
        fits.extend(refit.summary for refit in refits)
        chosen, refits, endpoint_family = _count_refit_endpoints(engine, chosen, refits)
        candidate_fits.extend(refits)
        pick, members = _refit_choice([refit.summary.F for refit in refits],
                                      [tie_rules.path_key(X[d]) for d in chosen], F, counter,
                                      key(label, "deletion"))
        smaller = refits[pick]
        founder = chosen[pick]
        smaller_scores = engine.fit_scores(smaller.summary.label, smaller.rows, budget)
        observed = []
        decision, _, shadow_cells, detail = _shadow_decision(
            engine, geometry, X, scores, smaller_scores, np.delete(X, founder, axis=0),
            budget, cost, rule, [*seed_key, label, "deletion"], "deletion", view, decision_options=observed)
        shadow_details = {pick: detail}
        shadow_options = {pick: observed}
        tie_rules.census("assembly.count_margin", key(label, "deletion", "margin"),
                         [decision.total.margin, decision.total.loso_margin, decision.sequence.margin,
                          decision.sequence.loso_margin])

        def decide_member(position):
            if position == pick:
                return decision
            member = refits[position]
            member_scores = engine.fit_scores(member.summary.label, member.rows, budget)
            member_observed = []
            member_decision, _, _, detail = _shadow_decision(
                engine, geometry, X, scores, member_scores, np.delete(X, chosen[position], axis=0),
                budget, cost, rule, [*seed_key, label, "deletion"], "deletion", view, decision_options=member_observed)
            shadow_details[position] = detail
            shadow_options[position] = member_observed
            return member_decision

        tied, tie_records, agree, outcomes = tie_outcomes("deletion", label, members, pick, chosen, refits,
                                                          decide_member)
        resolved, flags = _tied_resolution(members, pick, outcomes, refits, X, engine, budget, shadow_details)
        flags = _shadow_flags(flags, shadow_details, refits, X)
        resolved, selected_decision, flags = _count_choice(engine, 'deletion', label, X, members,
            resolved, outcomes, refits, shadow_options, shadow_details, flags)
        if resolved != pick or selected_decision is not decision:
            pick = resolved
            smaller = refits[pick]
            founder = chosen[pick]
            decision = selected_decision
            smaller_scores = engine.fit_scores(smaller.summary.label, smaller.rows, budget)
            _, shadow_cells = _shadow_options(geometry, X, np.delete(X, founder, axis=0))
        hidden = hidden_failures(X, scores, founder) if decision.accept_larger else None
        entry = _decision("deletion", label, K, decision, ranking, keys, chosen, refits, pick, cost,
                          view, larger_fit=larger_fit, smaller_fit=smaller.summary.label,
                          optimistic_score=optimistic, band_bound=band_bound,
                          without_sequence=without, shadow_cells=shadow_cells,
                          shadow_alternatives=tuple(dict(candidate=int(chosen[p]),
                              **({"fit_endpoint": tie_rules.panel_key(refits[p].rows)}
                                 if endpoint_family else {}), **shadow_details[p])
                                                    for p in sorted(shadow_details)),
                          fit_family_evaluations=(_endpoint_evaluations(
                              "deletion", X, chosen, refits, members, outcomes, shadow_options,
                              shadow_details, pick, view) if endpoint_family else ()),
                          selected_refit_panel=(tie_rules.panel_key(refits[pick].rows)
                                                if endpoint_family else ()),
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
        # Keep one work summary per actual convergence; the endpoint table
        # expands evidence already computed, not the number of searches run.
        fits.extend(refit.summary for refit in refits)
        chosen, refits, endpoint_family = _count_refit_endpoints(engine, chosen, refits)
        candidate_fits.extend(refits)
        pick, members = _refit_choice([refit.summary.F for refit in refits],
                                      [tie_rules.path_key(X[f]) for f in chosen], F, counter,
                                      key(label, "addition"))
        larger = refits[pick]
        larger_scores = engine.fit_scores(larger.summary.label, larger.rows, budget)
        # The refit may have moved any founder: its sequence part is the gain over the refit with
        # every cell the incumbent does not use replaced by the incumbent's nearest row.
        observed = []
        decision, without, shadow_cells, detail = _shadow_decision(
            engine, geometry, larger.rows, larger_scores, scores, X, budget, cost, rule,
            [*seed_key, label, "addition"], "addition", view, decision_options=observed)
        shadow_details = {pick: detail}
        shadow_options = {pick: observed}
        tie_rules.census("assembly.count_margin", key(label, "addition", "margin"),
                         [decision.total.margin, decision.total.loso_margin, decision.sequence.margin,
                          decision.sequence.loso_margin])

        def decide_member(position):
            if position == pick:
                return decision
            member = refits[position]
            member_scores = engine.fit_scores(member.summary.label, member.rows, budget)
            member_observed = []
            member_decision, _, _, detail = _shadow_decision(
                engine, geometry, member.rows, member_scores, scores, X, budget, cost, rule,
                [*seed_key, label, "addition"], "addition", view, decision_options=member_observed)
            shadow_details[position] = detail
            shadow_options[position] = member_observed
            return member_decision

        tied, tie_records, agree, outcomes = tie_outcomes("addition", label, members, pick, chosen, refits,
                                                          decide_member)
        resolved, flags = _tied_resolution(members, pick, outcomes, refits, X, engine, budget, shadow_details)
        flags = _shadow_flags(flags, shadow_details, refits, X)
        resolved, selected_decision, flags = _count_choice(engine, 'addition', label, X, members,
            resolved, outcomes, refits, shadow_options, shadow_details, flags)
        if resolved != pick or selected_decision is not decision:
            pick = resolved
            larger = refits[pick]
            decision = selected_decision
            larger_scores = engine.fit_scores(larger.summary.label, larger.rows, budget)
            without = tuple(int(f) for f in np.flatnonzero(~_own_sequence(larger.rows)))
            _, shadow_cells = _shadow_options(geometry, larger.rows, X)
        entry = _decision("addition", label, K, decision, ranking, gains, chosen, refits, pick, cost,
                          view, larger_fit=larger.summary.label, smaller_fit=smaller_fit,
                          optimistic_score=optimistic, without_sequence=without,
                          shadow_cells=shadow_cells,
                          shadow_alternatives=tuple(dict(candidate=int(chosen[p]),
                              **({"fit_endpoint": tie_rules.panel_key(refits[p].rows)}
                                 if endpoint_family else {}), **shadow_details[p])
                                                    for p in sorted(shadow_details)),
                          fit_family_evaluations=(_endpoint_evaluations(
                              "addition", X, chosen, refits, members, outcomes, shadow_options,
                              shadow_details, pick, view) if endpoint_family else ()),
                          selected_refit_panel=(tie_rules.panel_key(refits[pick].rows)
                                                if endpoint_family else ()),
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
            adopt(final)
            sizes.append(len(X))
            cleanup_steps.add(len(sizes) - 1)
            if len(X) <= 2:
                break
        entry, smaller, smaller_scores, pair = deletion_round(X, scores, final.summary.label)
        decisions.append(entry)
        decided(entry)
        panels.append(pair)
        if entry.accept_larger:
            break
        final, X, scores = smaller, smaller.rows, smaller_scores
        adopt(final)
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
        decided(entry)
        panels.append(pair)
        if not entry.accept_larger:
            break
        adopt(larger)
        check, smaller, smaller_scores, check_pair = deletion_round(larger.rows, larger_scores,
                                                                    larger.summary.label)
        check = replace(check, after_addition=True)
        decisions.append(check)
        decided(check)
        panels.append(check_pair)
        if check.accept_larger:
            final, X, scores = larger, larger.rows, larger_scores
            adopt(final)
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
            adopt(final)
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
        adopt(final)
        comparisons = _restored_comparisons(engine, geometry, budget, X, scores, best_at,
                                            decisions, panels, cost, rule, seed_key, view)
        replaced = BestFReplacement(from_fit=from_fit, to_fit=final.summary.label, gain=gain,
                                    **comparisons)
    # A count-margin gap is data resolution. Only the selected search ancestry,
    # its actual symmetric count outcomes, and the normal best-F restoration
    # candidate may re-enter the final comparison; an unrelated discarded fit's
    # own direction tie is not a live alternative of this component.
    eligible_fits = _terminal_candidates(candidate_fits, ancestry, ancestry_labels, best_at[len(X)][4])
    prior_fit, prior_scores, live = _terminal_family(engine, eligible_fits, final, scores, budget)
    if not np.array_equal(prior_fit.rows, X):
        from_fit, old_F = final.summary.label, float(scores.sum())
        final, X, scores = prior_fit, prior_fit.rows, prior_scores
        adopt(final)
        comparisons = _restored_comparisons(engine, geometry, budget, X, scores, best_at,
                                            decisions, panels, cost, rule, seed_key, view)
        replaced = BestFReplacement(from_fit=from_fit, to_fit=final.summary.label,
                                    gain=float(scores.sum()) - old_F, **comparisons)
    if replaced is not None:
        for comparison in (replaced.deletion_vs_best_smaller, replaced.addition_vs_best_larger):
            if comparison is not None and comparison.get("shadow", {}).get("unresolved"):
                ancestry.append(dict(panels=comparison["panels"], source_decision="best_f",
                                     source_reason="nearest_shadow_count",
                                     representative_prior="smaller_k+nearest_shadow",
                                     shadow_alternatives=(comparison["shadow"],)))
    count_unverified = [item for entry in decisions for item in entry.count_unverified]
    if replaced is not None:
        for kind, comparison in (("deletion", replaced.deletion_vs_best_smaller),
                                 ("addition", replaced.addition_vs_best_larger)):
            if comparison is not None and comparison.get("shadow", {}).get("count_unverified"):
                count_unverified.append(_unverified_shadow(comparison["shadow"],
                    dict(kind=kind, round="best_f", founders=len(X))))
    live_panels = {tie_rules.panel_key(X)}
    live_panels.update(panel for family in live for panel in family["panels"])
    unresolved = _uncontinued_families(ancestry, len(X), live_panels)
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
        best_f_replacement=replaced, live_alternatives=live,
        unresolved_alternatives=unresolved, continuation_families=tuple(ancestry),
        count_unverified=tuple(count_unverified),
        input_F0=float(fits[0].F0))


def count_pass(engine, geometry, start, cost, rule, budget, seed_key, additions, counter=None,
               census_prefix=None, alternative_starts=()):
    """At most four joint input/fit-direction/count-choice continuations.

    Every branch replays the original input, reconstructing bootstrap labels,
    round/cleanup counters, best-by-K records and cycle history before selecting
    its evaluated alternative. No intermediate panel is a fresh count-pass root.
    """
    def run(worker, incoming, incoming_seed):
        return _count_pass_point(worker, geometry, incoming, cost, rule, budget, incoming_seed, additions,
                                 counter=counter, census_prefix=worker.census_prefix)

    def branch_worker(fit_choices, count_choices, input_key):
        # Include count overrides in the existing branch checkpoint identity,
        # while preserving the original biological input and seed passed to run.
        identity = input_key if not count_choices else (input_key, tuple(sorted(count_choices.items())))
        worker = engine.branch(fit_choices, input_key=identity)
        worker._forced_count_choices = dict(count_choices)
        return worker

    def event_of(family, worker):
        token = family.get('source_count_key')
        if token is not None:
            return 'count', token
        label = family.get('source_fit')
        if label is not None:
            return 'fit', (label, worker._fit_invocations[label])
        return None

    def handled(family, state):
        event = event_of(family, state['engine'])
        if event is None:
            return False
        kind, token = event
        return token in state['count_choices' if kind == 'count' else 'choices']

    def remaining_families(state):
        return tuple(family for family in state['result'].unresolved_alternatives
                     if not handled(family, state))

    engine._forced_count_choices = {}
    initial = run(engine, start, seed_key)
    nominal_input = tie_rules.panel_key(start)
    input_keys = tuple(sorted({nominal_input, *(tie_rules.panel_key(np.asarray(panel))
                                              for panel in alternative_starts)}))
    limit = int(engine.config.alternative_branches)
    if len(input_keys) > limit:
        family = dict(panels=input_keys, reason="unresolved_branch_budget",
                      source_decision="incoming_panel", representative_prior="smaller_k+fewest_switches",
                      unresolved_search=True)
        reports = []
        for source in initial.continuation_families:
            if not source.get('source_count_key'):
                continue
            options = tuple(sorted(set(source['panels'])))
            digest = lambda panel: hashlib.sha256(repr(panel).encode()).hexdigest()
            selected = engine._chosen_count_panels[source['source_count_key']]
            terminal = digest(tie_rules.panel_key(initial.rows))
            reports.append(dict(source_count_key=source['source_count_key'],
                kind=source.get('source_count_kind'), round=source.get('source_decision'),
                source_choice_count=len(options), source_choice_Ks=tuple(len(p) for p in options),
                source_panels=tuple(digest(p) for p in options), nominal_source_panel=digest(selected),
                evaluated_terminal_histories=1, evaluated_source_choices=1, source_choices_complete=False,
                unfinished_history=True, status='unperformed_budget', final_status='unverified', resolved=False,
                retained_terminal_panels=(terminal,), retained_terminal_count=1, within_K_F_margins=(),
                representative_prior=None, replay_seconds=0., histories=(dict(branch=engine.branch_key,
                    source_panel=digest(selected), terminal_panel=terminal, terminal_K=len(initial.rows),
                    terminal_F=float(initial.F), unverified_history=True,
                    count_cycle=initial.count_cycle, count_limit_reached=initial.count_limit_reached),)))
        sources = tuple(source for source in initial.continuation_families
                        if event_of(source, engine) is not None)
        unperformed = {panel for source in sources for panel in source['panels']}
        returned = tie_rules.panel_key(initial.rows)
        live = []
        for terminal in initial.live_alternatives:
            keep = [i for i, panel in enumerate(terminal['panels'])
                    if panel == returned or panel not in unperformed]
            if len(keep) > 1:
                live.append(dict(terminal, **{key: tuple(terminal[key][i] for i in keep)
                    for key in ('panels', 'scores', 'minimum_switches') if key in terminal}))
        # Incoming-root exhaustion also prevents all fit/count source histories
        # in the nominal root from being continued. Keep every evaluated source.
        source_history = tuple(dict(source, reason='unresolved_branch_budget', unresolved_search=True)
                               for source in sources)
        return replace(initial, live_alternatives=tuple(live),
                       unresolved_alternatives=(*initial.unresolved_alternatives, family, *source_history),
                       branch_status="unresolved_budget", count_continuations=tuple(reports))
    if len(input_keys) == 1 and not any(event_of(family, engine) is not None
                                      for family in initial.continuation_families):
        return replace(initial, branch_status=("unresolved_count_continuation"
                       if initial.unresolved_alternatives else "none"))
    begun = time.perf_counter()
    pending = [dict(choices={}, count_choices={}, result=initial, engine=engine, start=start,
                    seed_key=seed_key, input_key=nominal_input)]
    evaluations, reused, count_evaluations, count_seconds = 1, 0, 0, 0.0
    for panel in input_keys:
        if panel == nominal_input:
            continue
        branch = branch_worker({}, {}, panel)
        incoming = np.asarray(panel, np.int64)
        # Same bootstrap stream; each root retains its ordinary 2 * incoming K bound.
        result = run(branch, incoming, seed_key)
        evaluations += 1
        reused += branch.reused_fits
        pending.append(dict(choices={}, count_choices={}, result=result, engine=branch, start=incoming,
                            seed_key=seed_key, input_key=panel))
    pending.sort(key=lambda state: state["input_key"])
    leaves, budget_families, count_events = [], [], []
    overflow = False
    while pending:
        state = pending.pop(0)
        point, worker = state['result'], state['engine']
        event = None
        for family in point.continuation_families:
            found = event_of(family, worker)
            if found is not None and not handled(family, state):
                event = (*found, family)
                break
        if event is None:
            leaves.append(state)
            continue
        kind, token, family = event
        options = tuple(sorted(set(family["panels"])))
        nominal = (worker._chosen_count_panels[token] if kind == 'count'
                   else worker._chosen_panels[token])
        record = None
        if kind == 'count':
            record = dict(token=token, family=family, input_key=state['input_key'],
                          fit_prefix=dict(state['choices']), count_prefix=dict(state['count_choices']),
                          nominal=nominal, options=options, overflow=False, replay_seconds=0.0)
            count_events.append(record)
        # Enumerate the full encountered class, or retain its incumbent and history.
        if len(leaves) + len(pending) + len(options) > limit:
            overflow = True
            budget_families.append(family)
            if record is not None:
                record['overflow'] = True
            leaves.append(dict(state, unfinished_budget=True))
            continue
        for panel in options:
            choices, count_choices = dict(state['choices']), dict(state['count_choices'])
            (count_choices if kind == 'count' else choices)[token] = panel
            if panel == nominal:
                pending.append(dict(state, choices=choices, count_choices=count_choices))
            else:
                branch = branch_worker(choices, count_choices, state['input_key'])
                started = time.perf_counter()
                result = run(branch, state['start'], state['seed_key'])
                seconds = time.perf_counter() - started
                evaluations += 1
                reused += branch.reused_fits
                if kind == 'count':
                    count_evaluations += 1
                    count_seconds += seconds
                    record['replay_seconds'] += seconds
                pending.append(dict(state, choices=choices, count_choices=count_choices,
                                    result=result, engine=branch))

    # Compare only outputs of the same K by F. Different supported count outcomes
    # remain a family; smaller K is the declared representative prior.
    counts = sorted({len(state["result"].rows) for state in leaves})
    by_count = {}
    for founders in counts:
        group = [state for state in leaves if len(state["result"].rows) == founders]
        values = [state["result"].F for state in group]
        keys = [tie_rules.panel_key(state["result"].rows) for state in group]
        classes = ties.tie_classes(values, tie_rules.quantum("refine_F", max(abs(v) for v in values)),
                                   content=keys, descending=True)
        by_count[founders] = [group[int(i)] for i in classes.members(int(classes.classes[classes.winner]))]
    survivors = by_count[counts[0]]
    if len(survivors) > 1:
        switches = [engine.minimum_switches(state["result"].rows, budget) for state in survivors]
        minimum = min(switches)
        survivors.sort(key=lambda state: ties.conv_key(tie_rules.panel_key(state["result"].rows)))
        chosen = next(state for state in survivors
                      if engine.minimum_switches(state["result"].rows, budget) == minimum)
    else:
        chosen = survivors[0]
    result = chosen["result"]
    def terminal_panels_of(state):
        point = state['result']
        returned = tie_rules.panel_key(point.rows)
        unperformed = {panel for family in point.continuation_families
                       if event_of(family, state['engine']) is not None and not handled(family, state)
                       for panel in family['panels']}
        panels = {returned: point.F}
        for family in point.live_alternatives:
            for panel, score in zip(family['panels'], family['scores']):
                # A fresh terminal F score does not evaluate a source panel's
                # missing count continuation. Another actual terminal leaf can
                # still establish this panel independently.
                if panel == returned or panel not in unperformed:
                    panels[panel] = score
        return panels

    final_panels = {}
    for state in by_count[counts[0]]:
        final_panels.update(terminal_panels_of(state))
    live = ()
    if len(final_panels) > 1:
        keys = tuple(sorted(final_panels))
        live = (dict(panels=keys, scores=tuple(final_panels[key] for key in keys),
                     minimum_switches=tuple(engine.minimum_switches(np.asarray(key, np.int64), budget)
                                            for key in keys), reason="same_k_equal_F",
                     representative_prior="fewest_switches", unresolved_search=False),)
    uncontinued = [family for state in leaves for family in remaining_families(state)]
    uncontinued.extend(dict(family, reason="unresolved_branch_budget", unresolved_search=True)
                       for family in budget_families)
    if len(counts) > 1:
        panels = tuple(sorted({tie_rules.panel_key(state["result"].rows)
                               for group in by_count.values() for state in group}))
        uncontinued.append(dict(panels=panels, reason="ambiguous_k",
                                representative_prior="smaller_k+fewest_switches",
                                unresolved_search=False))
    # Repeat discovery through reconverged paths does not multiply uncertainty records.
    unique = {}
    for family in uncontinued:
        key = (family["reason"], tuple(sorted(family["panels"])))
        unique.setdefault(key, family)
    unresolved = tuple(unique.values())
    status = ("unresolved_budget" if overflow else "unresolved_count_continuation" if
              any(family["unresolved_search"] for family in unresolved) else
              "ambiguous_k" if len(counts) > 1 else "same_k_tie" if live else "resolved")
    count_unverified = tuple(dict(item, branch=state["engine"].branch_key)
                             for state in leaves for item in state["result"].count_unverified)
    count_reports = []
    def digest(panel):
        return hashlib.sha256(repr(panel).encode()).hexdigest()
    for event in count_events:
        histories = []
        for state in leaves:
            if state['input_key'] != event['input_key']:
                continue
            if any(state['choices'].get(k) != v for k, v in event['fit_prefix'].items()):
                continue
            if any(state['count_choices'].get(k) != v for k, v in event['count_prefix'].items()):
                continue
            selected = getattr(state['engine'], '_chosen_count_panels', {}).get(event['token'])
            if selected not in event['options']:
                continue
            point = state['result']
            histories.append(dict(branch=state['engine'].branch_key,
                source_panel=digest(selected), terminal_panel=digest(tie_rules.panel_key(point.rows)),
                terminal_K=len(point.rows), terminal_F=float(point.F),
                terminal_candidates=tuple(dict(panel=digest(p), K=len(p), F=float(f))
                                          for p, f in terminal_panels_of(state).items()),
                unverified_history=bool(state.get('unfinished_budget') or remaining_families(state)
                                        or point.count_cycle or point.count_limit_reached),
                count_cycle=point.count_cycle, count_limit_reached=point.count_limit_reached))
        seen = {h['source_panel'] for h in histories}
        source_complete = not event['overflow'] and len(seen) == len(event['options'])
        unfinished = any(h['unverified_history'] for h in histories)
        candidates = [candidate for h in histories for candidate in h['terminal_candidates']]
        terminal_counts = sorted({c['K'] for c in candidates})
        terminal_panels = {c['panel'] for c in candidates}
        event_status = ('unperformed_budget' if event['overflow'] else 'unperformed_history' if not source_complete
                  else 'continued_with_unverified_history' if unfinished
                  else 'terminal_count_alternatives' if len(terminal_counts) > 1
                  else 'resolved_terminal_panel' if len(terminal_panels) == 1
                  else 'terminal_panels_compared_by_F')
        # Compare complete terminal outputs by the same within-K F classes as
        # the scientific selector. A representative prior never removes an
        # equal-F biological family from this report.
        retained = []
        f_margins = []
        for founders in terminal_counts:
            group = [c for c in candidates if c['K'] == founders]
            values = [c['F'] for c in group]
            keys = [c['panel'] for c in group]
            classes = ties.tie_classes(values, tie_rules.quantum('refine_F', max(abs(v) for v in values)),
                                       content=keys, descending=True)
            winning = {int(i) for i in classes.members(int(classes.classes[classes.winner]))}
            retained.extend(keys[i] for i in winning)
            other = [value for i, value in enumerate(values) if i not in winning]
            f_margins.append(dict(K=founders, margin=None if not other else max(values)-max(other)))
        retained = tuple(sorted(set(retained)))
        final_status = ('unverified' if not source_complete or unfinished else
                        'ambiguous_k' if len(terminal_counts) > 1 else
                        'same_k_tie' if len(retained) > 1 else 'resolved')
        if final_status == 'unverified':
            # Starting every choice does not complete a family whose descendants
            # hit the joint budget or ordinary count-cycle/backstop.
            family = dict(event['family'], reason='unresolved_count_continuation', unresolved_search=True)
            key = (family['reason'], tuple(sorted(family['panels'])))
            unique.setdefault(key, family)
        count_reports.append(dict(source_count_key=event['token'],
            kind=event['family'].get('source_count_kind'), round=event['family'].get('source_decision'),
            source_choice_count=len(event['options']), source_choice_Ks=tuple(len(p) for p in event['options']),
            source_panels=tuple(digest(p) for p in event['options']),
            nominal_source_panel=digest(event['nominal']), evaluated_terminal_histories=len(histories),
            evaluated_source_choices=len(seen), source_choices_complete=source_complete,
            unfinished_history=unfinished, status=event_status, histories=tuple(histories),
            retained_terminal_panels=retained, retained_terminal_count=len(retained),
            final_status=final_status, resolved=final_status == 'resolved', within_K_F_margins=tuple(f_margins),
            representative_prior=('smaller_k+fewest_switches' if len(terminal_counts) > 1 else
                                  'fewest_switches' if len(retained) > 1 else None),
            replay_seconds=event['replay_seconds']))
    unresolved = tuple(unique.values())
    if status == 'resolved' and any(family['unresolved_search'] for family in unresolved):
        status = 'unresolved_count_continuation'
    return replace(result, live_alternatives=live, unresolved_alternatives=unresolved,
                   count_unverified=count_unverified,
                   ambiguous=result.ambiguous or len(counts) > 1,
                   branch_evaluations=evaluations, branch_reused_fits=reused,
                   branch_seconds=time.perf_counter() - begun, branch_status=status,
                   selected_branch=chosen["engine"].branch_key, input_F0=initial.input_F0,
                   count_continuations=tuple(count_reports), count_branch_evaluations=count_evaluations,
                   count_branch_seconds=count_seconds)
