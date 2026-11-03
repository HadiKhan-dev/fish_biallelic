"""Count pass of one component: converged count decisions between converged fixed-K fits.

Owns count rounds and continuation order; never selects a storage route or inspects free memory.
The count driver keeps deletion/addition closures beside their per-pass state so their mutation and
bootstrap labels remain visibly ordered. Each closure implements one scientific transition.
Method: docs/methods/founder_refinement.md#count-pass (RC09–RC17).

Count rule (assembly/count_rule, shared with the join): a decision between a larger panel X+ and a
smaller X- keeps X+ iff the effective-carrier test passes on the total per-sample gain
d_s = F_s(X+) - F_s(X-) AND on its sequence part q_s = F_s(X+) - F_s(X+^sigma). The shadow X+^sigma
replaces every cell (block, row) of X+ that a reference panel does not use by the reference's row
nearest to it at that block. Release 2.8 (symmetric null, _symmetric_decision): the reference is X+
minus each member of its own canonical death tie class (its best unrefitted one-founder deletions,
founders without a private cell first), in both directions, so the same two endpoints get the same
decision whichever was the incumbent; X+ is kept only if every member's test keeps it, and a
disagreement or an unresolved member family takes the smaller K with ambiguity. A
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
a founder whenever the best-F tie-break examined another founder first). The best-F keep and the restore
compare F with max(TOL, quantum).
At the final count step only (the last level), duplicate-seed addition rounds (seeds X plus a copy
of founder f; with more than r founders the r seeds of largest exact copy best-response gain are
refitted); with the symmetric null an accepted addition needs no check deletion round, so no count
cycle arises (2.7 checked every addition and reported cycles). An optimistic bound (upper_bound) may skip a round
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

import hashlib
import itertools
import math
from dataclasses import replace

import numpy as np
from numba import prange

from haplotype_reconstruction.contracts.assembly import BestFReplacement, CountDecisionRecord, CountPass
from haplotype_reconstruction.core import ties
from haplotype_reconstruction.core.kernels import kernel

from .. import count_rule
from .. import orders as tie_rules
from ..count_rule import shadow_bounds
from ..count_rule.geometry import shadow_choices
from .acceptance import TOL
from .costs import copy_response_bytes
from .objective import CENTER, forward_sites, pair_code7, row_max, selected_alleles, transition, unordered_pairs
from .overloads import overloads_for
from .parameters import (
    COUNT_BOUND_RELATIVE,
    COUNT_FANOUT_BYTES,
    COUNT_LIMIT_MULTIPLE,
    FAMILY_BYTES,
    SAMPLE_BOUND_RELATIVE,
)
from .reductions import panel_objective
from .window import capacity, cover, locate, samples_of, site_cost, site_logs, stretch
from .workspace import sample_ranges, sample_threads

# Additions stop at the incoming count plus COUNT_LIMIT_MULTIPLE times it (reported when reached).

# A complete shadow family is scored in one prefix-sharing pass while the pass's own arrays (allele slabs, base
# alleles, scores and kernel scratch) take at most this many bytes; a larger family is scored member by member
# (engine.per_sample: the same values). The engine's workspace and caches are not counted.


# ------------------------------------------------------------------ optimistic count bound
@kernel(overloads=overloads_for(__name__, "_block_sample_scores"), cache=True)
def _block_sample_scores(haps, win, start, stop, removed):
    """Per-sample optimistic scores of one block: its rows (minus row `removed` if >= 0) as a free
    panel with the target K's switch costs (the window's), every sample's chain started fresh at the
    block (the transition into the block's first informative site is free: switch costs are >= 0)."""
    rows = np.arange(len(haps))
    if removed >= 0:
        rows = rows[rows != removed]
    first, second = unordered_pairs(len(rows))
    dos = np.empty((stop - start, len(first)), np.int8)
    for j in range(stop - start):
        for q in range(len(first)):
            dos[j, q] = pair_code7(haps[rows[first[q]], j], haps[rows[second[q]], j])
    samples = samples_of(win)
    result = np.empty(samples, np.float64)
    R = np.empty(len(rows))
    e = np.empty(7)
    for s in range(samples):
        row = np.zeros(len(first), np.float64)
        forward_sites(row, dos, start, win, s, start, stop, R, e)
        result[s] = row_max(row)
    return result


@kernel(overloads=overloads_for(__name__, "upper_bound"), cache=True, parallel=True)
def upper_bound(leaves, offsets, win, target):
    """Per-block optimistic scores bounding every panel with at most `target` distinct rows per
    block, and their per-sample form.

    Each block chooses its own panel and every sample's diplotype resets freely between blocks;
    a block with target + 1 rows takes its best one-row deletion (exact for that block), any other
    block every local row (possibly more than target). Both relaxations enlarge the feasible set,
    so the sum over blocks bounds every chromosome panel with <= target founder paths drawn from
    these prepared rows (same site mask, emissions and switch costs as fitting: costs are
    the target K's; a free panel of more rows with the same per-change costs only enlarges the
    path set). The panel-independent normalisers of the target K are not included (callers add
    workspace.normalisers(target)).
    bounds[b] is the block's total over samples; per_sample[b] lets every sample drop a different
    row of a target+1-row block (a maximum of sums becomes a sum of maxima): summed over blocks it
    bounds each sample. constrained[b] = 1 for a target+1-row block. The window (the target K's
    costs) covers one stretch of blocks at a time; the blocks of a stretch run in parallel."""
    samples = samples_of(win)
    blocks = len(leaves)
    bounds = np.empty(blocks, np.float64)
    constrained = np.zeros(blocks, np.int64)
    per_sample = np.empty((blocks, samples), np.float64)
    b0 = 0
    while b0 < blocks:
        b1 = stretch(offsets, b0, blocks, capacity(win))
        cover(win, offsets[b0], offsets[b1], 0, samples, False)
        for block_index in prange(b0, b1):
            block = np.int64(block_index)
            haps = leaves[block]
            deletion = len(haps) == target + 1
            best = -np.inf
            best_sample = np.full(samples, -np.inf)
            for removed in range(len(haps) if deletion else 1):
                scores = _block_sample_scores(
                    haps, win, offsets[block], offsets[block + 1], removed if deletion else -1
                )
                total = 0.0
                for sample in range(samples):
                    total += scores[sample]
                best = max(best, total)
                best_sample = np.maximum(best_sample, scores)
            bounds[block] = best
            per_sample[block] = best_sample
            constrained[block] = int(deletion)
        b0 = b1
    return bounds, constrained, per_sample


@kernel(overloads=overloads_for(__name__, "_shadow_codes"), cache=True, nogil=True)
def _shadow_codes(leaves, site_offsets, choices, starts):
    """shadow_bounds._dosages on coded leaves: pair codes (pair_code7) of every local row choice."""
    founders = choices.shape[1]
    states = founders * (founders + 1) // 2
    offsets = np.zeros(len(choices) + 1, np.int64)
    for block in range(len(starts) - 1):
        sites = site_offsets[block + 1] - site_offsets[block]
        for choice in range(starts[block], starts[block + 1]):
            offsets[choice + 1] = offsets[choice] + sites
    codes = np.empty((offsets[-1], states), np.int8)
    for block in range(len(starts) - 1):
        leaf = leaves[block]
        for choice in range(starts[block], starts[block + 1]):
            for site in range(leaf.shape[1]):
                state = 0
                for first in range(founders):
                    a = leaf[choices[choice, first], site]
                    for second in range(first, founders):
                        codes[offsets[choice] + site, state] = pair_code7(a, leaf[choices[choice, second], site])
                        state += 1
    return codes, offsets


@kernel(overloads=overloads_for(__name__, "_full_site_costs"), cache=True, parallel=True, nogil=True)
def _full_site_costs(win, site_offsets, dosages, dose_offsets, starts, founders):
    """shadow_bounds._full_site on the refinement objective (per-homologue transitions with the
    switch costs of K = founders, the window's): per sample the maximum over every block's local row
    choices, carried across blocks (an exact per-sample envelope of the shadow family, without the
    normalisers). Stretch by stretch of blocks, each covered before it is read."""
    samples, states = samples_of(win), dosages.shape[1]
    blocks = len(starts) - 1
    rows = np.zeros((samples, states))
    b0 = 0
    while b0 < blocks:
        b1 = stretch(site_offsets, b0, blocks, capacity(win))
        cover(win, site_offsets[b0], site_offsets[b1], 0, samples, False)
        for sample in prange(samples):
            row = rows[sample]
            trial = np.empty(states)
            target = np.empty(states)
            R = np.empty(founders)
            e = np.empty(7)
            for block in range(b0, b1):
                target[:] = -np.inf
                lo, hi = site_offsets[block], site_offsets[block + 1]
                for choice in range(starts[block], starts[block + 1]):
                    trial[:] = row
                    forward_sites(trial, dosages, lo - dose_offsets[choice], win, sample, lo, hi, R, e)
                    for state in range(states):
                        target[state] = max(target[state], trial[state])
                row[:] = target
        b0 = b1
    result = np.empty(samples, np.float64)
    for sample in prange(samples):
        result[sample] = row_max(rows[sample])
    return result


@kernel(overloads=(), inline="always")
def _family_sites(row, R, e, haps, col0, win, s, j0, j1, first, second):
    """score_direct's site step over sites [j0, j1): haps column j - col0 holds site j's coded alleles."""
    for j in range(j0, j1):
        f, k, r = locate(win, s, j)
        if not site_logs(win, s, f, k, r, e):
            continue
        transition(row, site_cost(win, s, f, k, r), R)
        for q in range(len(first)):
            row[q] += e[pair_code7(haps[first[q], j - col0], haps[second[q], j - col0])] - CENTER


@kernel(overloads=overloads_for(__name__, "_family_forward"), cache=True, parallel=True, nogil=True)
def _family_forward(base, win, s0, s1, unit_start, unit_stop, unit_offsets, unit_alleles, out):
    """Raw max-path scores of every member of a factored shadow family, sharing prefixes.

    base: int16 (K, L) coded alleles of the family's first member; unit u (sorted, disjoint site ranges
    [unit_start[u], unit_stop[u])) has n_u = unit_offsets[u + 1] - unit_offsets[u] alternative allele
    slabs unit_alleles[unit_offsets[u] + c] (K x width_u used). out: float64 (s1 - s0, M), M = prod n_u
    members in depth-first order (unit 0 slowest), for the samples [s0, s1) (the window should hold every
    site of them: a sample's depth-first walk revisits the whole component). A sample's forward row at a
    site depends only on the alleles before it, so a member shares the row of its common prefix bit for
    bit: every member's score equals score_panel's on its full allele matrix (the same site steps in the
    same order)."""
    founders, sites = base.shape
    first, second = unordered_pairs(founders)
    units = len(unit_start)
    cover(win, 0, sites, s0, s1, False)
    for s in prange(s0, s1):
        R = np.empty(founders)
        e = np.empty(7)
        stack = np.zeros((units + 1, len(first)))
        row = np.zeros(len(first))
        _family_sites(row, R, e, base, 0, win, s, 0, unit_start[0], first, second)
        stack[0, :] = row
        index = np.zeros(units, np.int64)
        depth = 0
        leaf = 0
        while True:
            row[:] = stack[depth]
            slab = unit_alleles[unit_offsets[depth] + index[depth]]
            _family_sites(
                row, R, e, slab, unit_start[depth], win, s, unit_start[depth], unit_stop[depth], first, second
            )
            following = unit_start[depth + 1] if depth + 1 < units else sites
            _family_sites(row, R, e, base, 0, win, s, unit_stop[depth], following, first, second)
            if depth + 1 < units:
                depth += 1
                stack[depth, :] = row
                index[depth] = 0
                continue
            out[s - s0, leaf] = row_max(row)
            leaf += 1
            while depth >= 0:
                index[depth] += 1
                if index[depth] < unit_offsets[depth + 1] - unit_offsets[depth]:
                    break
                index[depth] = 0
                depth -= 1
            if depth < 0:
                break


def _family_scores(engine, choices, shape, budget):
    """Per-sample scores (engine.per_sample, normalisers included) of every member of the factored
    family itertools.product(*choices) (choices: one option tuple per cell of a K x B panel, founder-major),
    in product order. Members already in the engine's exact score cache take the cached vector, as per_sample
    does; the others come from one prefix-sharing pass (_family_forward) over consecutive sample ranges, each
    at the budget's live thread share (a sample's scores are its own recursion). None when the pass's arrays
    would exceed FAMILY_BYTES (the caller then scores its members one at a time)."""
    founders, blocks = shape
    ws = engine.workspace
    result = [
        engine.ledger.scores.get(tie_rules.panel_key(np.asarray(cells, np.int64).reshape(shape)))
        for cells in itertools.product(*choices)
    ]
    if all(value is not None for value in result):
        return result
    varying = [i for i, options in enumerate(choices) if len(options) > 1]
    base_rows = np.asarray([options[0] for options in choices], np.int64).reshape(shape)
    by_block = {}
    for i in varying:
        by_block.setdefault(i % blocks, []).append(i)
    unit_blocks = sorted(by_block)
    unit_start = np.asarray([ws.offsets[b] for b in unit_blocks], np.int64)
    unit_stop = np.asarray([ws.offsets[b + 1] for b in unit_blocks], np.int64)
    width = int(max(unit_stop - unit_start))
    combos = [
        (by_block[b], list(itertools.product(*[range(len(choices[i])) for i in by_block[b]]))) for b in unit_blocks
    ]
    offsets = np.cumsum([0] + [len(unit) for _, unit in combos]).astype(np.int64)
    members = math.prod(len(unit) for _, unit in combos)
    # Slabs (int16), base alleles (int16 K x L), raw and normalised scores, and per-sample kernel scratch (a prefix
    # stack of len(combos) + 1 rows of Q states, a row and K maxima) are resident together at most.
    states = founders * (founders + 1) // 2
    nbytes = (
        2 * int(offsets[-1]) * founders * width
        + 2 * founders * int(ws.offsets[-1])
        + 16 * ws.samples * members
        + 8 * ws.samples * ((len(combos) + 2) * states + founders)
    )
    if nbytes > FAMILY_BYTES:
        return None
    with budget.reserve(nbytes, label="founder_refinement count family"):
        slabs = np.zeros((int(offsets[-1]), founders, width), np.int16)
        for u, (b, (cells, unit)) in enumerate(zip(unit_blocks, combos, strict=False)):
            leaf = ws.leaves[b]
            for c, picks in enumerate(unit):
                rows = base_rows[:, b].copy()
                for i, pick in zip(cells, picks, strict=False):
                    rows[i // blocks] = choices[i][pick]
                slabs[offsets[u] + c, :, : unit_stop[u] - unit_start[u]] = leaf[rows]
        with budget.scope(sample_threads(ws.samples, budget)):
            base = selected_alleles(ws.leaves, ws.offsets, np.ascontiguousarray(base_rows))
        raw = np.empty((ws.samples, members))
        for s0, s1, threads in sample_ranges(ws.samples, budget):
            # A streaming component reads each sample range through windows that hold every site of a
            # bounded slice of its samples (window.Windows.samples_view).
            for c0, c1 in ws.windows.sample_slices(s0, s1):
                with budget.scope(threads):
                    _family_forward(
                        base,
                        ws.windows.samples_view(founders, c0, c1),
                        c0,
                        c1,
                        unit_start,
                        unit_stop,
                        offsets,
                        slabs,
                        raw[c0:c1],
                    )
        del slabs, base
        # Depth-first leaf -> product index (cells in founder-major order, the last cell fastest).
        radix = [len(choices[i]) for i in varying]
        position = {cell: k for k, cell in enumerate(varying)}
        normalisers = ws.normalisers(founders)
        for leaf, picks in enumerate(itertools.product(*[unit for _, unit in combos])):
            digits = [0] * len(varying)
            for (cells, _), unit_picks in zip(combos, picks, strict=False):
                for cell, pick in zip(cells, unit_picks, strict=False):
                    digits[position[cell]] = pick
            index = 0
            for digit, base_count in zip(digits, radix, strict=False):
                index = index * base_count + digit
            if result[index] is None:
                result[index] = raw[:, leaf] + normalisers
    return result


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


def _shadow_decision(
    engine,
    geometry,
    larger,
    larger_scores,
    smaller_scores,
    reference,
    budget,
    cost,
    rule,
    seed_key,
    kind,
    view,
    *,
    decision_options=None,
):
    """Complete the nearest-shadow family, or preserve the incumbent's own verdict.

    The separate computational budget never supplies count evidence: overflow
    evaluates no other shadow and leaves the evaluated decision unchanged. Only
    actual opposed verdicts invoke smaller K. All scores use one mask/seed stream;
    existing no-private-cell q=0 and optimistic round bounds remain exact shortcuts.
    """
    begun = 0.0
    without = tuple(int(f) for f in np.flatnonzero(~_own_sequence(larger)))
    if without:
        decision = count_rule.decide(
            view(larger_scores),
            view(smaller_scores),
            view(larger_scores),
            cost,
            kind=kind,
            config=rule,
            seed_key=seed_key,
        )
        return (
            decision,
            without,
            0,
            dict(
                combinations=1,
                evaluated=0,
                complete=True,
                unresolved=False,
                no_private_sequence=True,
                seconds=0.0 - begun,
            ),
        )
    choices, replaced = _shadow_options(geometry, larger, reference)
    if not replaced:
        decision = count_rule.decide(
            view(larger_scores),
            view(smaller_scores),
            view(larger_scores),
            cost,
            kind=kind,
            config=rule,
            seed_key=seed_key,
        )
        return (
            decision,
            without,
            0,
            dict(combinations=1, evaluated=0, complete=True, unresolved=False, seconds=0.0 - begun),
        )
    combinations = math.prod(map(len, choices))
    total_passes = count_rule.passes(view(larger_scores - smaller_scores), cost, len(view(larger_scores)))
    # A complete small family also supplies invariant count margins for ranking
    # tied refits, even if the total-gain test alone rejects the larger count.
    complete = combinations <= int(rule.shadow_combinations)
    paths = itertools.product(*choices) if complete else [_shadow(geometry, larger, reference)[0].ravel()]
    family = (
        count_rule.ShadowDecisionFamily(
            view(larger_scores), view(smaller_scores), cost, kind=kind, config=rule, seed_key=seed_key
        )
        if complete and combinations > 1
        else None
    )
    decisions = []
    # A complete family is scored in one prefix-sharing pass (the same per-member values as per_sample).
    member_scores = _family_scores(engine, choices, larger.shape, budget) if complete and combinations > 1 else None
    for m, cells in enumerate(paths):
        rows = np.asarray(cells, np.int64).reshape(larger.shape)
        shadow_scores = engine.per_sample(rows, budget) if member_scores is None else member_scores[m]
        decisions.append(
            count_rule.decide(
                view(larger_scores),
                view(smaller_scores),
                view(shadow_scores),
                cost,
                kind=kind,
                config=rule,
                seed_key=seed_key,
            )
            if family is None
            else family.decide(view(shadow_scores))
        )
    if decision_options is not None:
        decision_options.extend(decisions)
    verdicts = {decision.accept_larger for decision in decisions}
    unresolved = total_passes and len(verdicts) > 1
    decision = min(decisions, key=lambda value: (value.accept_larger, value.margin, value.sequence.loso_margin))
    if unresolved:
        decision = replace(decision, accept_larger=False, ambiguous=True)
    details = dict(
        nearest_choices=choices,
        combinations=combinations,
        evaluated=len(decisions),
        complete=complete,
        resolved_by_total=not total_passes,
        unresolved=unresolved,
        count_unverified=not complete,
        verdicts=tuple(value.accept_larger for value in decisions),
        sequence_margins=tuple(value.sequence.margin for value in decisions),
        sequence_loso_margins=tuple(value.sequence.loso_margin for value in decisions),
        seconds=0.0 - begun,
    )
    if not complete:
        # The ordinary decision and its nominal sequence diagnostics stay intact.
        # An optional full-domain proof updates only computational verification.
        from .parameters import PAIR_TABLE_BYTES

        ws = engine.workspace
        threads = max(1, int(budget.threads()))

        def upper_scores(plan):
            founders = int(larger.shape[0])
            with budget.scope(sample_threads(ws.samples, budget)):
                dosages, dose_offsets = _shadow_codes(ws.leaves, ws.offsets, plan.choices, plan.starts)
                scores = _full_site_costs(
                    ws.windows.view(founders), ws.offsets, dosages, dose_offsets, plan.starts, founders
                )
            return view(scores + ws.normalisers(founders))

        proof, audit = shadow_bounds.attempt_certificate(
            choices,
            larger.shape,
            np.diff(ws.offsets),
            view(larger_scores),
            view(smaller_scores),
            cost,
            incumbent=decision,
            config=rule,
            seed_key=seed_key,
            upper_scores=upper_scores,
            scorer_samples=ws.samples,
            threads=threads,
            full_site=True,
            max_table_bytes=PAIR_TABLE_BYTES,
        )
        details.update(
            criterion_verified=proof is not None, proof_attempt=audit, sequence_diagnostics="evaluated_incumbent_only"
        )
        if proof is not None:
            details.update(count_unverified=False, count_verified_by_bound=proof)
        details["seconds"] = 0.0 - begun
    return decision, without, int(replaced), details


def _unverified_shadow(details, scope):
    """Unevaluated computational work; empty panels prevent biological tie interpretation."""
    return dict(
        status="count_unverified",
        count_unverified=True,
        unresolved_search=True,
        cause="nearest_shadow_budget",
        reason="count_unverified",
        domain_size=int(details["combinations"]),
        evaluated=int(details["evaluated"]),
        scope=scope,
        panels=(),
        representative_prior=None,
    )


def _shadow_flags(flags, details, refits, X):
    """Retain opposing count outcomes of every actually compared tied refit."""
    uncertain = [position for position, value in details.items() if value["unresolved"]]
    if not uncertain:
        return flags
    panels = set(flags.get("unresolved_panels", ()))
    panels.add(tie_rules.panel_key(X))
    panels.update(tie_rules.panel_key(refits[position].rows) for position in uncertain)
    return dict(flags, ambiguous_k=True, unresolved_panels=tuple(sorted(panels)), prior="smaller_k+nearest_shadow")


def _loso_margin(criterion):
    """The joint decision's leave-one-sample-out margin: the smaller of the two parts'."""
    return float(min(criterion["total"]["loso_margin"], criterion["sequence"]["loso_margin"]))


def _death_class(engine, rows, budget):
    """Founders of the larger panel's canonical unrefitted death tie class (the symmetric null):
    founders without a private cell first, otherwise all; the best tie class of unrefitted
    F(rows minus f) by the refine_F quantum. Cached per panel on the engine."""
    rows = np.ascontiguousarray(rows, np.int64)
    token = (rows.shape, rows.tobytes())
    cache = engine.ledger.death_classes
    if len(cache) > 64:
        cache.clear()
    if token in cache:
        return cache[token]
    owns = _own_sequence(rows)
    pool = np.flatnonzero(~owns) if not owns.all() else np.arange(len(rows))
    totals = [
        panel_objective(scores)
        for scores in _scores_many(
            engine, [np.delete(rows, f, axis=0) for f in pool.tolist()], budget, "founder_refinement death class"
        )
    ]
    scale = max(abs(t) for t in totals)
    result = ties.policy("assembly.refine_death_class", kind="refine_F", reverse=engine.reverse).classes(
        totals, scale=scale, content=[tie_rules.path_key(rows[f]) for f in pool.tolist()], descending=True
    )
    members = [int(pool[int(i)]) for i in result.members(int(result.classes[result.winner]))]
    cache[token] = members
    return members


def _symmetric_decision(
    engine,
    geometry,
    larger,
    larger_scores,
    smaller_scores,
    budget,
    cost,
    rule,
    seed_key,
    kind,
    view,
    *,
    decision_options=None,
):
    """Symmetric count null (release 2.8): the sequence reference is the larger panel minus its own
    canonical death candidate in either direction, so the same two endpoints get the same decision
    whichever was the incumbent. Every member of the death tie class is evaluated; disagreement takes
    smaller K with ambiguity (the existing nearest-shadow convention)."""
    begun = 0.0
    members = _death_class(engine, larger, budget)

    def member(child, founder):
        family = []
        decision, without, cells, detail = _shadow_decision(
            engine,
            geometry,
            larger,
            larger_scores,
            smaller_scores,
            np.delete(larger, founder, axis=0),
            child,
            cost,
            rule,
            seed_key,
            kind,
            view,
            decision_options=family,
        )
        return founder, decision, without, cells, detail, family

    # The members' decisions are independent (each its own family and seed stream): concurrent, in member order.
    if len(members) < 2:
        outcomes = [member(budget, founder) for founder in members]
    else:
        outcomes = [None] * len(members)
        # Each decision may build a shadow-family table (FAMILY_BYTES at most) and score panels.
        member_bytes = FAMILY_BYTES + _score_bytes(engine, len(larger))
        for wave in _waves([member_bytes] * len(members)):
            for offset, outcome in budget.concurrent(
                [lambda child, founder=members[i]: member(child, founder) for i in wave],
                member_bytes,
                label="founder_refinement death class decisions",
            ):
                outcomes[wave[offset]] = outcome
    founder, decision, without, cells, detail, _ = min(
        outcomes,
        key=lambda x: (x[1].accept_larger, x[1].margin, x[1].sequence.loso_margin, tie_rules.path_key(larger[x[0]])),
    )
    detail = dict(detail)
    disagree = len({x[1].accept_larger for x in outcomes}) > 1
    # Any member whose own nearest-shadow family is unresolved keeps the ambiguity.
    member_unresolved = any(bool(x[4].get("unresolved")) for x in outcomes)
    if disagree or member_unresolved:
        decision = replace(decision, accept_larger=False, ambiguous=True)
    detail.update(
        unresolved=bool(disagree or member_unresolved),
        sequence_null="larger_minus_death_class",
        member_unresolved=tuple(bool(x[4].get("unresolved")) for x in outcomes),
        member_details=tuple(
            dict(
                founder=int(x[0]),
                accept_larger=bool(x[1].accept_larger),
                margin=float(x[1].margin),
                unresolved=bool(x[4].get("unresolved")),
                evaluated=int(x[4].get("evaluated", 0)),
            )
            for x in outcomes
        ),
        sequence_founder=int(founder),
        death_class=tuple(members),
        death_class_verdicts=tuple(bool(x[1].accept_larger) for x in outcomes),
        symmetric_seconds=0.0 - begun,
    )
    if decision_options is not None:
        for x in outcomes:
            decision_options.extend(x[5] or [x[1]])
    return decision, without, cells, detail


def _score_bytes(engine, founders: int) -> int:
    """Transient bytes of one engine.per_sample of a K = founders panel: its int16 alleles, the pair-dosage
    table when objective.score_panel builds it (search.PAIR_TABLE_BYTES), and its result."""
    from .parameters import PAIR_TABLE_BYTES

    ws = engine.workspace
    sites = int(ws.offsets[-1])
    table = sites * founders * (founders + 1) // 2
    return 2 * founders * sites + (table if table <= PAIR_TABLE_BYTES else 0) + 8 * int(ws.samples)


# Transient bytes the count pass's concurrent scorings and shadow decisions may hold together: waves of them run
# concurrently up to this sum (one at a time when a single one exceeds it), whatever the memory admission does.


def _waves(sizes):
    """Consecutive index ranges whose summed bytes stay within COUNT_FANOUT_BYTES (at least one item each)."""
    start, total = 0, 0
    for index, size in enumerate(sizes):
        if index > start and total + size > COUNT_FANOUT_BYTES:
            yield range(start, index)
            start, total = index, 0
        total += size
    if start < len(sizes):
        yield range(start, len(sizes))


def _scores_many(engine, panels, budget, label):
    """engine.per_sample of several panels as concurrent in-process tasks, each on its live share of the budget
    (2.10: the count pass's independent scorings no longer run one after another); results in input order.
    Scores never depend on threads (fixed-order per-sample recursions). Each task holds its transient bytes
    with the run's memory governor, so concurrent scorings are paced by memory."""
    if len(panels) < 2:
        return [engine.per_sample(panel, budget) for panel in panels]
    results = [None] * len(panels)
    sizes = [_score_bytes(engine, len(panel)) for panel in panels]
    for wave in _waves(sizes):
        for offset, scores in budget.concurrent(
            [lambda child, panel=panels[i]: engine.per_sample(panel, child) for i in wave],
            [sizes[i] for i in wave],
            label=label,
        ):
            results[wave[offset]] = scores
    return results


def _fit_many(engine, starts, labels, budget):
    """Converge several independent starts concurrently; results (Fits) in input order. Each fit's
    share of the budget is re-read within its phases, so the last fits take over the finished
    ones' cores; a fit splits its share between its two searches. Its two searches reserve their
    message arrays (engine.converge: fit_bytes each, base's two per fit), so a fit reserves nothing more."""
    results = [None] * len(starts)
    ordinals = engine.bind_fits(
        len(starts), [(label, start, "fit") for label, start in zip(labels, starts, strict=False)]
    )
    for index, fit in budget.concurrent(
        [
            lambda child, start=start, label=label, ordinal=ordinal: engine.converge(
                start, label, child, ordinal=ordinal
            )
            for start, label, ordinal in zip(starts, labels, ordinals, strict=False)
        ],
        0,
        label="founder_refinement fits",
    ):
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
    for candidate, fit in zip(chosen, refits, strict=False):
        label = fit.summary.label
        # A label-only pending vector belongs to the primary, not a companion.
        pending = engine.ledger.fit_scores.pop(label, None)
        if pending is not None:
            engine.ledger.scores[tie_rules.panel_key(fit.rows)] = pending
        variants = (fit,)
        if fit.summary.tied_panels:
            token = (label, engine.ledger.invocations[label])
            variants = engine.ledger.fits[token]
        for variant in sorted(variants, key=lambda item: tie_rules.panel_key(item.rows)):
            origins.append(candidate)
            endpoints.append(
                replace(variant, summary=replace(variant.summary, tied_panels=(), tied_scores=(), tied_switches=()))
            )
    return origins, endpoints, True


def _refit_choice(values, keys, scale, counter=None, census_key=None, *, reverse, sink=None):
    """The refit decided at a count step: the content-smallest member (keys: the deleted founder's or
    copied seed's row path) of the best tie class of the refits' F (quantum at |scale|, the incumbent's
    F). Returns (position, positions of its class in decision order)."""
    quantum = tie_rules.quantum("refine_F", scale)
    result = ties.fixed_policy("assembly.refine_refit", quantum, reverse=reverse).classes(
        [float(v) for v in values], content=keys, descending=True
    )
    tie_rules.add_decision(
        counter,
        "assembly.refine_refit",
        result,
        [float(v) for v in values],
        quantum,
        tie_rules.motion("refine_F", scale),
        descending=True,
    )
    tie_rules.census("assembly.refine_refit", census_key, [-float(v) for v in values], sink=sink)
    pick = result.winner
    members = [int(i) for i in result.members(int(result.classes[pick]))]
    return pick, members


def _tied_resolution(members, pick, outcomes, refits, X, engine=None, budget=None, shadow_details=None):
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
    result = ties.fixed_policy(
        "assembly.refine_eligible", quantum, reverse=False if engine is None else engine.reverse
    ).classes(margins, content=contents, descending=larger_side)
    winner = candidates[result.winner]
    tied = [candidates[i] for i in result.members(int(result.classes[result.winner]))]
    distinct = sorted({panel_of(p) for p in tied})
    flags = dict(
        ambiguous_k=bool(disagree),
        prior="smaller_k" if disagree else "margin",
        competitive_fits=tuple(sorted(refits[p].summary.label for p in tied)),
    )
    if disagree:
        # These are the actual opposing count outcomes, before the smaller-K prior.
        flags["unresolved_panels"] = tuple(
            sorted(set(flags.get("unresolved_panels", ())) | {panel_of(p) for p in members})
        )
    if len(distinct) > 1:
        flags["alternatives"] = tuple(distinct)
        if engine is not None:
            counts = {panel: engine.minimum_switches(np.asarray(panel, np.int64), budget) for panel in distinct}
            minimum = min(counts.values())
            eligible = [p for p in tied if counts[panel_of(p)] == minimum]
            winner = ties.conv_sorted(eligible, key=panel_of, reverse=False if engine is None else engine.reverse)[0]
            flags["prior"] += "+fewest_switches"
        else:
            flags["prior"] += "+symmetric"
    return winner, flags


def _eligible_count_records(X, members, outcomes, refits, shadow_options, shadow_details, *, reverse):
    """Actual evaluated outcomes in each verdict's winning count-margin class."""

    def panel_of(position, decision):
        rows = refits[position].rows
        take = decision.accept_larger if len(rows) > len(X) else not decision.accept_larger
        return tie_rules.panel_key(rows if take else X)

    records = []
    for position in members:
        observed = shadow_options.get(position, ())
        if shadow_details[position].get("unresolved") and observed:
            for verdict in sorted({bool(d.accept_larger) for d in observed}):
                decision = min(
                    (d for d in observed if bool(d.accept_larger) == verdict),
                    key=lambda d: (d.margin, d.sequence.loso_margin),
                )
                records.append((position, decision, panel_of(position, decision)))
        else:
            decision = outcomes[position]
            records.append((position, decision, panel_of(position, decision)))
    eligible = []
    for verdict in sorted({bool(item[1].accept_larger) for item in records}):
        group = [item for item in records if bool(item[1].accept_larger) == verdict]
        values = [float(item[1].margin) for item in group]
        classes = ties.fixed_policy(
            "assembly.refine_eligible", tie_rules.quantum("count_margin"), reverse=reverse
        ).classes(values, content=[item[2] for item in group], descending=verdict)
        eligible.extend(group[int(i)] for i in classes.members(int(classes.classes[classes.winner])))
    return records, eligible


def _endpoint_evaluations(
    kind, X, chosen, refits, members, outcomes, shadow_options, shadow_details, selected, view, *, reverse
):
    """Complete observed endpoint audit, before any representative convention."""
    records, eligible = _eligible_count_records(
        X, members, outcomes, refits, shadow_options, shadow_details, reverse=reverse
    )
    accepted = {(p, bool(d.accept_larger)) for p, d, _ in eligible}
    table = []
    for position, (origin, fit) in enumerate(zip(chosen, refits, strict=False)):
        evaluated = position in outcomes
        reference = np.delete(X, origin, axis=0) if kind == "deletion" else X
        table.append(
            dict(
                candidate=int(origin),
                fit=fit.summary.label,
                endpoint=tie_rules.panel_key(fit.rows),
                F=float(fit.summary.F),
                reference=tie_rules.panel_key(reference),
                best_F=evaluated,
                selected_refit=position == selected,
                criteria=tuple(
                    dict(
                        criterion=view.record(d),
                        resulting_panel=panel,
                        margin_eligible=(position, bool(d.accept_larger)) in accepted,
                    )
                    for p, d, panel in records
                    if p == position
                ),
            )
        )
    return tuple(table)


def _count_choice(engine, kind, label, X, members, resolved, outcomes, refits, shadow_options, shadow_details, flags):
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
        X, members, outcomes, refits, shadow_options, shadow_details, reverse=engine.reverse
    )
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
    wanted = engine.ledger.forced_count_choices.get(token, nominal)
    if wanted not in options:
        raise ValueError("Count replay choice is not an eligible evaluated outcome")
    position, decision = options[wanted]
    panels = tuple(sorted(options))
    source = dict(
        panels=panels,
        source_decision=label,
        source_count_kind=kind,
        source_count_key=token,
        selected_panel=wanted,
        representative_prior=flags.get("prior"),
        source_reason=(
            "nearest_shadow_count"
            if any(d.get("unresolved") for d in shadow_details.values())
            else "tied_count_candidates"
        ),
        status="ambiguous_k" if len({len(p) for p in panels}) > 1 else "same_k_tie",
    )
    engine.ledger.count_families[token] = source
    engine.ledger.chosen_count_panels[token] = wanted
    # A direction tie in any fit from the selected verdict's winning margin
    # class still needs its own continuation; another verdict is another branch.
    competitive = tuple(
        sorted(
            {refits[p].summary.label for p, d, _ in eligible if bool(d.accept_larger) == bool(decision.accept_larger)}
        )
    )
    flags = dict(
        flags,
        competitive_fits=competitive,
        ambiguous_k=len({len(p) for p in panels}) > 1,
        unresolved_panels=panels if len({len(p) for p in panels}) > 1 else (),
        alternatives=panels if len({len(p) for p in panels}) == 1 else (),
        continuation_key=token,
        continuation_replayed=wanted != nominal,
    )
    if wanted != nominal:
        flags["prior"] = None  # exhaustive continuation is not a biological prior
    return position, decision, flags


def _shortlist(
    candidates,
    keys,
    limit,
    quantum,
    contents,
    *,
    descending,
    counter=None,
    kind="",
    census_key=None,
    reverse,
    sink=None,
):
    """The first ``limit`` candidates in tie-class order of their ranking keys (descending or not),
    extended by the tie class the cut falls in (at most 2 x limit)."""
    values = [float(keys[c]) for c in candidates]
    result = ties.fixed_policy("assembly.refine_shortlist", quantum, reverse=reverse).classes(
        values, content=contents, descending=descending
    )
    tie_rules.add_decision(counter, kind, result, values, quantum, None, descending=descending, cut=limit)
    tie_rules.census(kind, census_key, [(-v if descending else v) for v in values], sink=sink)
    return tie_rules.truncate(list(candidates), result, limit)


def _decision(
    kind,
    label,
    founders,
    decision,
    ranking,
    scores,
    chosen,
    fits,
    pick,
    cost,
    view,
    *,
    larger_fit,
    smaller_fit,
    **extra,
):
    values = [fit.summary.F for fit in fits]
    candidate_F = [None] * int(founders)
    for candidate, value in zip(chosen, values, strict=False):
        old = candidate_F[candidate]
        candidate_F[candidate] = float(value) if old is None else max(old, float(value))
    others = sorted((v for i, v in enumerate(values) if i != pick), reverse=True)
    return CountDecisionRecord(
        kind=kind,
        round=label,
        founders=int(founders),
        skipped_by_bound=False,
        accept_larger=bool(decision.accept_larger),
        ambiguous=bool(decision.ambiguous or extra.get("ambiguous_k")),
        margin=float(decision.margin),
        band=tuple(float(x) for x in decision.band),
        ranking=ranking,
        candidate_scores=None if scores is None else tuple(None if v is None else float(v) for v in scores),
        refitted=tuple(dict.fromkeys(int(c) for c in chosen)),
        candidate_F=tuple(candidate_F),
        selected=int(chosen[pick]),
        selected_F=float(values[pick]),
        runner_up_margin_over_cost=None if not others else float((values[pick] - others[0]) / cost),
        criterion=view.record(decision),
        larger_fit=larger_fit,
        smaller_fit=smaller_fit,
        count_unverified=tuple(
            _unverified_shadow(
                item,
                dict(
                    kind=kind,
                    round=label,
                    candidate=item["candidate"],
                    founders=int(founders),
                    **({"fit_endpoint": item["fit_endpoint"]} if "fit_endpoint" in item else {}),
                ),
            )
            for item in extra.get("shadow_alternatives", ())
            if item.get("count_unverified")
        ),
        **extra,
    )


def _restored_comparisons(
    engine, geometry, budget, rows, scores, best_at, decisions, panels, cost, rule, seed_key, view
):
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
    for entry, pair in zip(decisions, panels, strict=False):
        if pair is None:
            continue
        role = "larger" if np.array_equal(pair[0], rows) else "smaller" if np.array_equal(pair[1], rows) else None
        if role is not None:
            recorded.append(
                dict(
                    kind=entry.kind,
                    round=entry.round,
                    role=role,
                    accept_larger=entry.accept_larger,
                    margin=entry.margin,
                    loso_margin=_loso_margin(entry.criterion),
                    band=list(entry.band),
                    ambiguous=entry.ambiguous,
                )
            )

    def replay(kind, other):
        held = best_at.get(other)
        if held is None:
            return None
        if kind == "deletion":
            larger, larger_scores, smaller = rows, scores, held[3]
        else:
            larger, larger_scores, smaller = held[1], held[3], scores
        decision, _, _, shadow_details = _symmetric_decision(
            engine,
            geometry,
            larger,
            larger_scores,
            smaller,
            budget,
            cost,
            rule,
            [*seed_key, "best_f", kind],
            kind,
            view,
        )
        return dict(
            fit=held[2],
            founders=int(other),
            F=float(held[0]),
            accept_larger=bool(decision.accept_larger),
            margin=float(decision.margin),
            loso_margin=_loso_margin(decision.record()),
            sequence_margin=float(decision.sequence.margin),
            shadow=shadow_details,
            panels=tuple(sorted({tie_rules.panel_key(rows), tie_rules.panel_key(held[1])})),
            band=[float(x) for x in decision.band],
            ambiguous=bool(decision.ambiguous),
        )

    deletion = replay("deletion", K - 1) if K > 2 else None
    addition = replay("addition", K + 1)
    own = next((item for item in recorded if item["kind"] == "deletion" and item["role"] == "larger"), None)
    test = own if own is not None else deletion
    fails = None if test is None else not test["accept_larger"]
    return dict(
        recorded=tuple(recorded),
        deletion_vs_best_smaller=deletion,
        addition_vs_best_larger=addition,
        own_deletion_test=(
            None if test is None else dict(source="recorded_round" if own is not None else "best_f_smaller_fit", **test)
        ),
        fails_own_deletion_test=fails,
        ambiguous=bool(fails),
    )


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
    return tuple(
        dict(
            panels=tuple(fit.summary.tied_panels),
            source_fit=fit.summary.label,
            source_count_round=entry.round,
            source_role="competitive_refit",
            representative_prior="fewest_switches",
        )
        for fit in candidate_fits
        if fit.summary.label in labels and len(fit.summary.tied_panels) > 1
    )


def _terminal_candidates(candidate_fits, ancestry, ancestry_labels, best_fit):
    """Preserve actual count-margin decisions when collecting final competitors."""
    allowed_panels = {
        tie_rules.panel_key(np.asarray(panel))
        for family in ancestry
        for panel in ((family["selected_panel"],) if family.get("source_count_key") else family["panels"])
    }
    eligible_fits = []
    for candidate in candidate_fits:
        selected = (candidate.summary.label, tie_rules.panel_key(candidate.rows))
        if candidate.summary.label in ancestry_labels or selected in ancestry_labels or candidate is best_fit:
            eligible_fits.append(candidate)
        elif tie_rules.panel_key(candidate.rows) in allowed_panels:
            eligible_fits.append(
                replace(candidate, summary=replace(candidate.summary, tied_panels=(), tied_scores=(), tied_switches=()))
            )
    return eligible_fits


def _terminal_family(engine, candidate_fits, final, scores, budget):
    """Current same-K equal-F candidates, rescored together; discarded histories stay out.

    Every candidate is a converged fit in this workspace, including a fit's two
    terminal direction results. Only candidates in the returned panel's F class
    qualify; an old fit tied with itself at lower F does not qualify.
    """
    returned = tie_rules.panel_key(final.rows)
    total = panel_objective(scores)
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
    classes = ties.fixed_policy("assembly.refine_terminal", quantum, reverse=engine.reverse).classes(
        values, content=keys, descending=True
    )
    current_class = int(classes.classes[keys.index(returned)])
    possible = [keys[int(i)] for i in classes.members(current_class)]
    if len(possible) < 2:
        return final, scores, ()
    exact = {returned: scores}
    others = [key for key in possible if key != returned]
    for key, found in zip(
        others,
        _scores_many(
            engine, [np.asarray(key, np.int64) for key in others], budget, "founder_refinement terminal family"
        ),
        strict=False,
    ):
        exact[key] = found
    values = [panel_objective(exact[key]) for key in possible]
    scale = max(abs(value) for value in values)
    classes = ties.fixed_policy(
        "assembly.refine_terminal", tie_rules.quantum("refine_F", scale), reverse=engine.reverse
    ).classes(values, content=possible, descending=True)
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
    chosen = ties.conv_sorted([key for key in eligible if switches[key] == best], reverse=engine.reverse)[0]
    family = dict(
        panels=live,
        scores=tuple(panel_objective(exact[key]) for key in live),
        minimum_switches=tuple(switches[key] for key in live),
        reason="same_k_equal_F",
        representative_prior="fewest_switches",
        unresolved_search=False,
    )
    return candidates[chosen][1], exact[chosen], (family,)


def _uncontinued_families(ancestry, founders, live_panels=()):
    """Only selected-path families whose alternative count continuation is unrun.

    Same-K candidates have a common final objective and are handled by
    _terminal_family. Different-K source panels cannot be called terminal ties.
    """
    result = []
    seen = set()
    for family in ancestry:
        panels = tuple(sorted({tie_rules.panel_key(np.asarray(panel)) for panel in family["panels"]}))
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
        result.append(dict(family, panels=panels, reason="unresolved_count_continuation", unresolved_search=True))
    return tuple(result)


# ------------------------------------------------------------------ the pass
def keep_margin(F):
    """Margin of the best-F keep, the cycle keep and the restore: max(TOL, the tie quantum at F)."""
    return max(TOL, tie_rules.quantum("refine_F", F))


def _count_pass_point(
    engine, geometry, start, cost, rule, budget, seed_key, additions, counter=None, census_prefix=None
):
    """Converged fixed-K fit and converged count decisions for one component (module docstring).

    engine: the component's search.SweepEngine; geometry: count_rule.RowGeometry of its prepared
    panels; start: int64 K x B starting rows; cost: the per-founder cost c; rule: CountRuleConfig;
    seed_key: the component's bootstrap key; additions: run addition rounds (the final count step);
    counter: the component's tie-class counter (tie_rules.SharedCounter) or None; census_prefix: its
    valley-census key prefix or None. Returns a CountPass."""
    engine.ledger.start_count_pass()
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
            ancestry.append(
                dict(panels=tuple(tied), source_fit=fit.summary.label, representative_prior="fewest_switches")
            )

    def decided(entry):
        ancestry.extend(_competitive_families(entry, candidate_fits))
        source = engine.ledger.count_families.get(entry.continuation_key)
        if source is not None:
            ancestry.append(dict(source, shadow_alternatives=entry.shadow_alternatives))
        else:
            family = entry.unresolved_panels or entry.alternatives
            if len(family) > 1:
                ancestry.append(
                    dict(
                        panels=tuple(family),
                        source_decision=entry.round,
                        representative_prior=entry.prior,
                        source_reason=(
                            "nearest_shadow_count"
                            if any(item.get("unresolved") for item in entry.shadow_alternatives)
                            else "tied_count_candidates"
                        ),
                        shadow_alternatives=entry.shadow_alternatives,
                        status="ambiguous_k" if entry.ambiguous_k else "same_k_tie",
                    )
                )

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
    unrefitted_cache = engine.ledger.unrefitted

    def unrefitted(panel, founder):
        reduced = np.ascontiguousarray(np.delete(panel, founder, axis=0))
        token = (reduced.shape, reduced.tobytes())
        found = unrefitted_cache.get(token)
        if found is None:
            if len(unrefitted_cache) > 4 * len(panel):
                unrefitted_cache.clear()
            found = unrefitted_cache[token] = engine.per_sample(reduced, budget)
        return found

    def prefetch(panel, founders):
        """Fill unrefitted(panel, f) for several founders at once (concurrent scorings, the same values)."""
        missing = {}
        for founder in founders:
            reduced = np.ascontiguousarray(np.delete(panel, founder, axis=0))
            token = (reduced.shape, reduced.tobytes())
            if token not in unrefitted_cache:
                missing[token] = reduced
        if len(missing) < 2:
            return
        if len(unrefitted_cache) + len(missing) > 4 * len(panel):
            unrefitted_cache.clear()
        for token, scores in zip(
            missing,
            _scores_many(engine, list(missing.values()), budget, "founder_refinement deletion scores"),
            strict=False,
        ):
            unrefitted_cache[token] = scores

    def remember(panel, held_fit, per_sample):
        F_fit = panel_objective(per_sample)
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
            records.append(
                dict(
                    candidate=int(chosen[position]),
                    F=float(refits[position].summary.F),
                    accept_larger=bool(decision.accept_larger),
                    margin=float(decision.margin),
                )
            )
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
            order = ties.conv_order(len(current), reverse=engine.reverse).tolist()
            prefetch(current, order)
            for founder in order:
                reduced = unrefitted(current, founder)
                loss = float(np.max(np.abs(view(current_scores) - view(reduced))))
                losses.append(loss)
                if counter is not None:
                    # The band at the exact zero of an unused founder's loss (owner rule A-7).
                    counter.note("assembly.refine_zero_gain.band", inside_tie=loss <= quantum)
                if loss <= quantum and found is None:
                    found = (founder, reduced)
            tie_rules.census(
                "assembly.refine_zero_gain.band", key("clean", cleanup_index, len(removed)), losses, sink=engine.census
            )
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
            kind="cleanup",
            round=label,
            founders=len(X),
            skipped_by_bound=False,
            accept_larger=False,
            ambiguous=False,
            refitted=(),
            selected=None,
            selected_F=float(refit.summary.F),
            larger_fit=larger_fit,
            smaller_fit=refit.summary.label,
            cleanup_removed=tuple(removed),
        )
        return record, refit, refit.rows, refit_scores

    def deletion_round(X, scores, larger_fit):
        nonlocal round_index
        K = len(X)
        label = f"r{round_index}"
        round_index += 1
        F = panel_objective(scores)
        # Founders without a private cell (q = 0) are decided first, as in the join.
        without = tuple(int(f) for f in np.flatnonzero(~_own_sequence(X)))
        optimistic = band_bound = None
        if not without:
            # Every founder has a private cell: the per-sample bounds of any K - 1 panel also bound
            # every shadow panel, so a bound that proves the total-gain test for every deletion
            # proves the sequence test too; the round is skipped only when the replicate-wise form
            # of the bound also proves the band (the decision's own bootstrap stream).
            with budget.scope(sample_threads(ws.samples, budget)):
                bounds, _, per_block_sample = upper_bound(ws.leaves, ws.offsets, ws.windows.view(K - 1), K - 1)
            normalisers = ws.normalisers(K - 1)
            per_sample_optimistic = per_block_sample.sum(axis=0) + normalisers
            optimistic = float(panel_objective(bounds) + panel_objective(normalisers))
            tolerance = COUNT_BOUND_RELATIVE * max(1.0, abs(F), abs(optimistic), abs(K * cost - 2.0 * F))
            excluded, band_bound = count_rule.deletion_bound_excludes(
                view(scores),
                view(per_sample_optimistic),
                cost,
                tolerance=tolerance,
                config=rule,
                seed_key=[*seed_key, label, "deletion"],
            )
            if excluded:
                return (
                    CountDecisionRecord(
                        kind="deletion",
                        round=label,
                        founders=K,
                        skipped_by_bound=True,
                        accept_larger=True,
                        ambiguous=False,
                        bound="deletion_per_sample",
                        optimistic_score=optimistic,
                        band_bound=band_bound,
                        larger_fit=larger_fit,
                    ),
                    None,
                    None,
                    None,
                )
        candidates = list(without) or list(range(K))
        if len(candidates) > ranked_refits:
            # Exact unrefitted per-sample scores of every candidate's deletion, ranked by the point
            # margin of the total gain (tie classes; the founder's path inside a class); the top r,
            # with the tie class the cut falls in, are converged.
            keys = [None] * K
            prefetch(X, candidates)
            for d in candidates:
                keys[d] = min(_point_margins(view(scores), view(unrefitted(X, d)), cost))
            chosen = _shortlist(
                candidates,
                keys,
                ranked_refits,
                tie_rules.quantum("count_margin"),
                [tie_rules.path_key(X[d]) for d in candidates],
                descending=False,
                counter=counter,
                kind="assembly.refine_shortlist",
                census_key=key(label, "short"),
                reverse=engine.reverse,
                sink=engine.census,
            )
            ranking = "unrefitted_point_margin"
        else:
            keys, chosen, ranking = None, list(candidates), "all_candidates"
        refits = _fit_many(
            engine, [np.delete(X, d, axis=0) for d in chosen], [f"{label}.del{d}" for d in chosen], budget
        )
        # Keep one work summary per actual convergence; the endpoint table
        # expands evidence already computed, not the number of searches run.
        fits.extend(refit.summary for refit in refits)
        chosen, refits, endpoint_family = _count_refit_endpoints(engine, chosen, refits)
        candidate_fits.extend(refits)
        pick, members = _refit_choice(
            [refit.summary.F for refit in refits],
            [tie_rules.path_key(X[d]) for d in chosen],
            F,
            counter,
            key(label, "deletion"),
            reverse=engine.reverse,
            sink=engine.census,
        )
        smaller = refits[pick]
        founder = chosen[pick]
        smaller_scores = engine.fit_scores(smaller.summary.label, smaller.rows, budget)
        observed = []
        decision, _, shadow_cells, detail = _symmetric_decision(
            engine,
            geometry,
            X,
            scores,
            smaller_scores,
            budget,
            cost,
            rule,
            [*seed_key, label, "deletion"],
            "deletion",
            view,
            decision_options=observed,
        )
        shadow_details = {pick: detail}
        shadow_options = {pick: observed}
        tie_rules.census(
            "assembly.count_margin",
            key(label, "deletion", "margin"),
            [
                decision.total.margin,
                decision.total.loso_margin,
                decision.sequence.margin,
                decision.sequence.loso_margin,
            ],
            sink=engine.census,
        )

        def decide_member(position):
            if position == pick:
                return decision
            member = refits[position]
            member_scores = engine.fit_scores(member.summary.label, member.rows, budget)
            member_observed = []
            member_decision, _, _, detail = _symmetric_decision(
                engine,
                geometry,
                X,
                scores,
                member_scores,
                budget,
                cost,
                rule,
                [*seed_key, label, "deletion"],
                "deletion",
                view,
                decision_options=member_observed,
            )
            shadow_details[position] = detail
            shadow_options[position] = member_observed
            return member_decision

        tied, tie_records, agree, outcomes = tie_outcomes(
            "deletion", label, members, pick, chosen, refits, decide_member
        )
        resolved, flags = _tied_resolution(members, pick, outcomes, refits, X, engine, budget, shadow_details)
        flags = _shadow_flags(flags, shadow_details, refits, X)
        resolved, selected_decision, flags = _count_choice(
            engine, "deletion", label, X, members, resolved, outcomes, refits, shadow_options, shadow_details, flags
        )
        if resolved != pick or selected_decision is not decision:
            pick = resolved
            smaller = refits[pick]
            founder = chosen[pick]
            decision = selected_decision
            smaller_scores = engine.fit_scores(smaller.summary.label, smaller.rows, budget)
            _, shadow_cells = _shadow_options(geometry, X, np.delete(X, founder, axis=0))
        hidden = hidden_failures(X, scores, founder) if decision.accept_larger else None
        entry = _decision(
            "deletion",
            label,
            K,
            decision,
            ranking,
            keys,
            chosen,
            refits,
            pick,
            cost,
            view,
            larger_fit=larger_fit,
            smaller_fit=smaller.summary.label,
            optimistic_score=optimistic,
            band_bound=band_bound,
            without_sequence=without,
            shadow_cells=shadow_cells,
            shadow_alternatives=tuple(
                dict(
                    candidate=int(chosen[p]),
                    **({"fit_endpoint": tie_rules.panel_key(refits[p].rows)} if endpoint_family else {}),
                    **shadow_details[p],
                )
                for p in sorted(shadow_details)
            ),
            fit_family_evaluations=(
                _endpoint_evaluations(
                    "deletion",
                    X,
                    chosen,
                    refits,
                    members,
                    outcomes,
                    shadow_options,
                    shadow_details,
                    pick,
                    view,
                    reverse=engine.reverse,
                )
                if endpoint_family
                else ()
            ),
            selected_refit_panel=(tie_rules.panel_key(refits[pick].rows) if endpoint_family else ()),
            hidden_sequence_failures=hidden,
            tied_candidates=tied,
            tie_decisions=tie_records,
            tie_decisions_agree=agree,
            **flags,
        )
        remember(smaller.rows, smaller, smaller_scores)
        return entry, smaller, smaller_scores, (X, smaller.rows)

    def hidden_failures(X, scores, decided):
        """Diagnostics only: founders other than the decided one whose sequence part fails the
        effective-carrier test (m <= 0 or m' <= 0 on q) at a kept deletion round."""
        failing = []
        for f in range(len(X)):
            if f == decided:
                continue
            shadow_scores, _, _ = _sequence_scores(engine, geometry, X, scores, np.delete(X, f, axis=0), budget)
            value, loso = _point_margins(view(scores), view(shadow_scores), cost)
            if not (value > 0.0 and loso > 0.0):
                failing.append((int(f), float(value), float(loso)))
        return tuple(failing)

    def addition_round(X, scores, smaller_fit):
        nonlocal round_index
        K = len(X)
        label = f"r{round_index}"
        round_index += 1
        F = panel_objective(scores)
        with budget.scope(sample_threads(ws.samples, budget)):
            bounds, _, per_block_sample = upper_bound(ws.leaves, ws.offsets, ws.windows.view(K + 1), K + 1)
        normalisers = ws.normalisers(K + 1)
        optimistic = float(panel_objective(bounds) + panel_objective(normalisers))
        tolerance = COUNT_BOUND_RELATIVE * max(1.0, abs(F), abs(optimistic), abs(K * cost - 2.0 * F))
        if count_rule.addition_bound_excludes(optimistic, F, cost, count_samples, tolerance=tolerance):
            return (
                CountDecisionRecord(
                    kind="addition",
                    round=label,
                    founders=K,
                    skipped_by_bound=True,
                    accept_larger=False,
                    ambiguous=False,
                    bound="addition_total",
                    optimistic_score=optimistic,
                    smaller_fit=smaller_fit,
                ),
                None,
                None,
                None,
            )
        # No sample's optimistic gain reaches the per-carrier cost c/(2S): no seed's refit (nor any
        # (K+1)-panel) can pass or make the call ambiguous, so the round is decided unrun.
        per_sample_optimistic = per_block_sample.sum(axis=0) + normalisers
        tolerance = SAMPLE_BOUND_RELATIVE * max(1.0, abs(F), float(np.max(np.abs(per_sample_optimistic))))
        excluded, gain_bound, gain_sample = count_rule.addition_per_sample_bound_excludes(
            view(per_sample_optimistic), view(scores), cost, tolerance=tolerance
        )
        if excluded:
            return (
                CountDecisionRecord(
                    kind="addition",
                    round=label,
                    founders=K,
                    skipped_by_bound=True,
                    accept_larger=False,
                    ambiguous=False,
                    bound="addition_per_sample",
                    optimistic_score=optimistic,
                    per_sample_gain_bound=float(gain_bound),
                    per_sample_gain_bound_sample=view.row(gain_sample),
                    smaller_fit=smaller_fit,
                ),
                None,
                None,
                None,
            )
        starts = [np.vstack((X, X[f : f + 1])) for f in range(K)]
        responses = gains = None
        if K > ranked_refits:
            # Seeds ranked by the exact F gain of their copy's best responses (tie classes; the
            # seed's path inside a class); the top r, with the tie class the cut falls in, are
            # converged.
            per_task = copy_response_bytes(engine.blocks, samples, K + 1)
            responses = [None] * K
            ordinals = engine.bind_fits(
                K, [(f"{label}.copy{index}", start, "copy_response") for index, start in enumerate(starts)]
            )
            for index, response in budget.concurrent(
                [
                    lambda child, start=start, ordinal=ordinal: engine.copy_response(start, child, ordinal=ordinal)
                    for start, ordinal in zip(starts, ordinals, strict=False)
                ],
                per_task,
                label="founder_refinement copy responses",
            ):
                responses[index] = response
            gains = [response[0] for response in responses]
            chosen = _shortlist(
                list(range(K)),
                gains,
                ranked_refits,
                tie_rules.quantum("refine_F", F),
                [tie_rules.path_key(X[f]) for f in range(K)],
                descending=True,
                counter=counter,
                kind="assembly.refine_shortlist",
                census_key=key(label, "short"),
                reverse=engine.reverse,
                sink=engine.census,
            )
            ranking = "copy_best_response_gain"
        else:
            chosen, ranking = list(range(K)), "all_candidates"
        refits = _fit_many(engine, [starts[f] for f in chosen], [f"{label}.add{f}" for f in chosen], budget)
        # Keep one work summary per actual convergence; the endpoint table
        # expands evidence already computed, not the number of searches run.
        fits.extend(refit.summary for refit in refits)
        chosen, refits, endpoint_family = _count_refit_endpoints(engine, chosen, refits)
        candidate_fits.extend(refits)
        pick, members = _refit_choice(
            [refit.summary.F for refit in refits],
            [tie_rules.path_key(X[f]) for f in chosen],
            F,
            counter,
            key(label, "addition"),
            reverse=engine.reverse,
            sink=engine.census,
        )
        larger = refits[pick]
        larger_scores = engine.fit_scores(larger.summary.label, larger.rows, budget)
        # The refit may have moved any founder: its sequence part is the gain over the refit with
        # every cell the incumbent does not use replaced by the incumbent's nearest row.
        observed = []
        decision, without, shadow_cells, detail = _symmetric_decision(
            engine,
            geometry,
            larger.rows,
            larger_scores,
            scores,
            budget,
            cost,
            rule,
            [*seed_key, label, "addition"],
            "addition",
            view,
            decision_options=observed,
        )
        shadow_details = {pick: detail}
        shadow_options = {pick: observed}
        tie_rules.census(
            "assembly.count_margin",
            key(label, "addition", "margin"),
            [
                decision.total.margin,
                decision.total.loso_margin,
                decision.sequence.margin,
                decision.sequence.loso_margin,
            ],
            sink=engine.census,
        )

        def decide_member(position):
            if position == pick:
                return decision
            member = refits[position]
            member_scores = engine.fit_scores(member.summary.label, member.rows, budget)
            member_observed = []
            member_decision, _, _, detail = _symmetric_decision(
                engine,
                geometry,
                member.rows,
                member_scores,
                scores,
                budget,
                cost,
                rule,
                [*seed_key, label, "addition"],
                "addition",
                view,
                decision_options=member_observed,
            )
            shadow_details[position] = detail
            shadow_options[position] = member_observed
            return member_decision

        tied, tie_records, agree, outcomes = tie_outcomes(
            "addition", label, members, pick, chosen, refits, decide_member
        )
        resolved, flags = _tied_resolution(members, pick, outcomes, refits, X, engine, budget, shadow_details)
        flags = _shadow_flags(flags, shadow_details, refits, X)
        resolved, selected_decision, flags = _count_choice(
            engine, "addition", label, X, members, resolved, outcomes, refits, shadow_options, shadow_details, flags
        )
        if resolved != pick or selected_decision is not decision:
            pick = resolved
            larger = refits[pick]
            decision = selected_decision
            larger_scores = engine.fit_scores(larger.summary.label, larger.rows, budget)
            without = tuple(int(f) for f in np.flatnonzero(~_own_sequence(larger.rows)))
            _, shadow_cells = _shadow_options(geometry, larger.rows, X)
        entry = _decision(
            "addition",
            label,
            K,
            decision,
            ranking,
            gains,
            chosen,
            refits,
            pick,
            cost,
            view,
            larger_fit=larger.summary.label,
            smaller_fit=smaller_fit,
            optimistic_score=optimistic,
            without_sequence=without,
            shadow_cells=shadow_cells,
            shadow_alternatives=tuple(
                dict(
                    candidate=int(chosen[p]),
                    **({"fit_endpoint": tie_rules.panel_key(refits[p].rows)} if endpoint_family else {}),
                    **shadow_details[p],
                )
                for p in sorted(shadow_details)
            ),
            fit_family_evaluations=(
                _endpoint_evaluations(
                    "addition",
                    X,
                    chosen,
                    refits,
                    members,
                    outcomes,
                    shadow_options,
                    shadow_details,
                    pick,
                    view,
                    reverse=engine.reverse,
                )
                if endpoint_family
                else ()
            ),
            selected_refit_panel=(tie_rules.panel_key(refits[pick].rows) if endpoint_family else ()),
            copy_responses=None if responses is None else tuple(response[1] for response in responses),
            tied_candidates=tied,
            tie_decisions=tie_records,
            tie_decisions_agree=agree,
            **flags,
        )
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
        # Release 2.8: additions and deletions use the same symmetric null on the same converged
        # endpoints, so no check deletion round follows an accepted birth and no count cycle arises
        # (count_cycle and cycle_undecided stay False in the records).
        final, X, scores = larger, larger.rows, larger_scores
        adopt(final)
        sizes.append(len(X))
    replaced = None
    held = best_at[len(X)]
    if held[0] > panel_objective(scores) + keep_margin(panel_objective(scores)):
        # The pass returned to a K it had held with a better converged panel: the count decisions
        # chose K, the fixed-K objective chooses the panel within K.
        gain = float(held[0] - panel_objective(scores))
        from_fit = final.summary.label
        final, X, scores = held[4], held[1], held[3]
        adopt(final)
        comparisons = _restored_comparisons(
            engine, geometry, budget, X, scores, best_at, decisions, panels, cost, rule, seed_key, view
        )
        replaced = BestFReplacement(from_fit=from_fit, to_fit=final.summary.label, gain=gain, **comparisons)
    # A count-margin gap is data resolution. Only the selected search ancestry,
    # its actual symmetric count outcomes, and the normal best-F restoration
    # candidate may re-enter the final comparison; an unrelated discarded fit's
    # own direction tie is not a live alternative of this component.
    eligible_fits = _terminal_candidates(candidate_fits, ancestry, ancestry_labels, best_at[len(X)][4])
    prior_fit, prior_scores, live = _terminal_family(engine, eligible_fits, final, scores, budget)
    if not np.array_equal(prior_fit.rows, X):
        from_fit, old_F = final.summary.label, panel_objective(scores)
        final, X, scores = prior_fit, prior_fit.rows, prior_scores
        adopt(final)
        comparisons = _restored_comparisons(
            engine, geometry, budget, X, scores, best_at, decisions, panels, cost, rule, seed_key, view
        )
        replaced = BestFReplacement(
            from_fit=from_fit, to_fit=final.summary.label, gain=panel_objective(scores) - old_F, **comparisons
        )
    if replaced is not None:
        for comparison in (replaced.deletion_vs_best_smaller, replaced.addition_vs_best_larger):
            if comparison is not None and comparison.get("shadow", {}).get("unresolved"):
                ancestry.append(
                    dict(
                        panels=comparison["panels"],
                        source_decision="best_f",
                        source_reason="nearest_shadow_count",
                        representative_prior="smaller_k+nearest_shadow",
                        shadow_alternatives=(comparison["shadow"],),
                    )
                )
    count_unverified = [item for entry in decisions for item in entry.count_unverified]
    if replaced is not None:
        for kind, comparison in (
            ("deletion", replaced.deletion_vs_best_smaller),
            ("addition", replaced.addition_vs_best_larger),
        ):
            if comparison is not None and comparison.get("shadow", {}).get("count_unverified"):
                count_unverified.append(
                    _unverified_shadow(comparison["shadow"], dict(kind=kind, round="best_f", founders=len(X)))
                )
    live_panels = {tie_rules.panel_key(X)}
    live_panels.update(panel for family in live for panel in family["panels"])
    unresolved = _uncontinued_families(ancestry, len(X), live_panels)
    return CountPass(
        rows=X,
        scores=scores,
        F=panel_objective(scores),
        founders_before=int(incoming),
        fits=tuple(fits),
        decisions=tuple(decisions),
        final_fit=final.summary.label,
        weak_junctions=final.weak_junctions,
        count_cycle=cycle,
        cycle_undecided=cycle_undecided,
        count_limit=int(upper),
        count_limit_reached=limit_reached,
        additions=bool(additions),
        ambiguous=(any(d.ambiguous for d in decisions) or bool(replaced is not None and replaced.ambiguous)),
        best_f_replacement=replaced,
        live_alternatives=live,
        unresolved_alternatives=unresolved,
        continuation_families=tuple(ancestry),
        count_unverified=tuple(count_unverified),
        input_F0=float(fits[0].F0),
    )


def count_pass(
    engine,
    geometry,
    start,
    cost,
    rule,
    budget,
    seed_key,
    additions,
    counter=None,
    census_prefix=None,
    alternative_starts=(),
):
    """At most four joint input/fit-direction/count-choice continuations.

    Every branch replays the original input, reconstructing bootstrap labels,
    round/cleanup counters, best-by-K records and cycle history before selecting
    its evaluated alternative. No intermediate panel is a fresh count-pass root.
    """

    def run(worker, incoming, incoming_seed):
        return _count_pass_point(
            worker,
            geometry,
            incoming,
            cost,
            rule,
            budget,
            incoming_seed,
            additions,
            counter=counter,
            census_prefix=worker.census_prefix,
        )

    def branch_worker(fit_choices, count_choices, input_key):
        # Include count overrides in the existing branch checkpoint identity,
        # while preserving the original biological input and seed passed to run.
        identity = input_key if not count_choices else (input_key, tuple(sorted(count_choices.items())))
        worker = engine.branch(fit_choices, input_key=identity)
        worker.ledger.forced_count_choices = dict(count_choices)
        return worker

    def event_of(family, worker):
        token = family.get("source_count_key")
        if token is not None:
            return "count", token
        label = family.get("source_fit")
        if label is not None:
            return "fit", (label, worker.ledger.invocations[label])
        return None

    def handled(family, state):
        event = event_of(family, state["engine"])
        if event is None:
            return False
        kind, token = event
        return token in state["count_choices" if kind == "count" else "choices"]

    def remaining_families(state):
        return tuple(family for family in state["result"].unresolved_alternatives if not handled(family, state))

    engine.ledger.forced_count_choices = {}
    initial = run(engine, start, seed_key)
    nominal_input = tie_rules.panel_key(start)
    input_keys = tuple(
        sorted({nominal_input, *(tie_rules.panel_key(np.asarray(panel)) for panel in alternative_starts)})
    )
    limit = int(engine.config.alternative_branches)
    if len(input_keys) > limit:
        family = dict(
            panels=input_keys,
            reason="unresolved_branch_budget",
            source_decision="incoming_panel",
            representative_prior="smaller_k+fewest_switches",
            unresolved_search=True,
        )
        reports = []
        for source in initial.continuation_families:
            if not source.get("source_count_key"):
                continue
            options = tuple(sorted(set(source["panels"])))

            def digest(panel):
                return hashlib.sha256(repr(panel).encode()).hexdigest()

            selected = engine.ledger.chosen_count_panels[source["source_count_key"]]
            terminal = digest(tie_rules.panel_key(initial.rows))
            reports.append(
                dict(
                    source_count_key=source["source_count_key"],
                    kind=source.get("source_count_kind"),
                    round=source.get("source_decision"),
                    source_choice_count=len(options),
                    source_choice_Ks=tuple(len(p) for p in options),
                    source_panels=tuple(digest(p) for p in options),
                    nominal_source_panel=digest(selected),
                    evaluated_terminal_histories=1,
                    evaluated_source_choices=1,
                    source_choices_complete=False,
                    unfinished_history=True,
                    status="unperformed_budget",
                    final_status="unverified",
                    resolved=False,
                    retained_terminal_panels=(terminal,),
                    retained_terminal_count=1,
                    within_K_F_margins=(),
                    representative_prior=None,
                    replay_seconds=0.0,
                    histories=(
                        dict(
                            branch=engine.branch_key,
                            source_panel=digest(selected),
                            terminal_panel=terminal,
                            terminal_K=len(initial.rows),
                            terminal_F=float(initial.F),
                            unverified_history=True,
                            count_cycle=initial.count_cycle,
                            count_limit_reached=initial.count_limit_reached,
                        ),
                    ),
                )
            )
        sources = tuple(source for source in initial.continuation_families if event_of(source, engine) is not None)
        unperformed = {panel for source in sources for panel in source["panels"]}
        returned = tie_rules.panel_key(initial.rows)
        live = []
        for terminal in initial.live_alternatives:
            keep = [i for i, panel in enumerate(terminal["panels"]) if panel == returned or panel not in unperformed]
            if len(keep) > 1:
                live.append(
                    dict(
                        terminal,
                        **{
                            key: tuple(terminal[key][i] for i in keep)
                            for key in ("panels", "scores", "minimum_switches")
                            if key in terminal
                        },
                    )
                )
        # Incoming-root exhaustion also prevents all fit/count source histories
        # in the nominal root from being continued. Keep every evaluated source.
        source_history = tuple(
            dict(source, reason="unresolved_branch_budget", unresolved_search=True) for source in sources
        )
        return replace(
            initial,
            live_alternatives=tuple(live),
            unresolved_alternatives=(*initial.unresolved_alternatives, family, *source_history),
            branch_status="unresolved_budget",
            count_continuations=tuple(reports),
        )
    if len(input_keys) == 1 and not any(
        event_of(family, engine) is not None for family in initial.continuation_families
    ):
        return replace(
            initial, branch_status=("unresolved_count_continuation" if initial.unresolved_alternatives else "none")
        )
    begun = 0.0
    pending = [
        dict(
            choices={},
            count_choices={},
            result=initial,
            engine=engine,
            start=start,
            seed_key=seed_key,
            input_key=nominal_input,
        )
    ]
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
        pending.append(
            dict(
                choices={},
                count_choices={},
                result=result,
                engine=branch,
                start=incoming,
                seed_key=seed_key,
                input_key=panel,
            )
        )
    pending.sort(key=lambda state: state["input_key"])
    leaves, budget_families, count_events = [], [], []
    overflow = False
    while pending:
        state = pending.pop(0)
        point, worker = state["result"], state["engine"]
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
        nominal = worker.ledger.chosen_count_panels[token] if kind == "count" else worker.ledger.chosen_panels[token]
        record = None
        if kind == "count":
            record = dict(
                token=token,
                family=family,
                input_key=state["input_key"],
                fit_prefix=dict(state["choices"]),
                count_prefix=dict(state["count_choices"]),
                nominal=nominal,
                options=options,
                overflow=False,
                replay_seconds=0.0,
            )
            count_events.append(record)
        # Enumerate the full encountered class, or retain its incumbent and history.
        if len(leaves) + len(pending) + len(options) > limit:
            overflow = True
            budget_families.append(family)
            if record is not None:
                record["overflow"] = True
            leaves.append(dict(state, unfinished_budget=True))
            continue
        for panel in options:
            choices, count_choices = dict(state["choices"]), dict(state["count_choices"])
            (count_choices if kind == "count" else choices)[token] = panel
            if panel == nominal:
                pending.append(dict(state, choices=choices, count_choices=count_choices))
            else:
                branch = branch_worker(choices, count_choices, state["input_key"])
                started = 0.0
                result = run(branch, state["start"], state["seed_key"])
                seconds = 0.0 - started
                evaluations += 1
                reused += branch.reused_fits
                if kind == "count":
                    count_evaluations += 1
                    count_seconds += seconds
                    record["replay_seconds"] += seconds
                pending.append(dict(state, choices=choices, count_choices=count_choices, result=result, engine=branch))

    # Compare only outputs of the same K by F. Different supported count outcomes
    # remain a family; smaller K is the declared representative prior.
    counts = sorted({len(state["result"].rows) for state in leaves})
    by_count = {}
    for founders in counts:
        group = [state for state in leaves if len(state["result"].rows) == founders]
        values = [state["result"].F for state in group]
        keys = [tie_rules.panel_key(state["result"].rows) for state in group]
        classes = ties.fixed_policy(
            "assembly.refine_terminal",
            tie_rules.quantum("refine_F", max(abs(v) for v in values)),
            reverse=engine.reverse,
        ).classes(values, content=keys, descending=True)
        by_count[founders] = [group[int(i)] for i in classes.members(int(classes.classes[classes.winner]))]
    survivors = by_count[counts[0]]
    if len(survivors) > 1:
        switches = [engine.minimum_switches(state["result"].rows, budget) for state in survivors]
        minimum = min(switches)
        survivors.sort(
            key=lambda state: ties.conv_key(tie_rules.panel_key(state["result"].rows), reverse=engine.reverse)
        )
        chosen = next(state for state in survivors if engine.minimum_switches(state["result"].rows, budget) == minimum)
    else:
        chosen = survivors[0]
    result = chosen["result"]

    def terminal_panels_of(state):
        point = state["result"]
        returned = tie_rules.panel_key(point.rows)
        unperformed = {
            panel
            for family in point.continuation_families
            if event_of(family, state["engine"]) is not None and not handled(family, state)
            for panel in family["panels"]
        }
        panels = {returned: point.F}
        for family in point.live_alternatives:
            for panel, score in zip(family["panels"], family["scores"], strict=False):
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
        live = (
            dict(
                panels=keys,
                scores=tuple(final_panels[key] for key in keys),
                minimum_switches=tuple(engine.minimum_switches(np.asarray(key, np.int64), budget) for key in keys),
                reason="same_k_equal_F",
                representative_prior="fewest_switches",
                unresolved_search=False,
            ),
        )
    uncontinued = [family for state in leaves for family in remaining_families(state)]
    uncontinued.extend(
        dict(family, reason="unresolved_branch_budget", unresolved_search=True) for family in budget_families
    )
    if len(counts) > 1:
        panels = tuple(
            sorted({tie_rules.panel_key(state["result"].rows) for group in by_count.values() for state in group})
        )
        uncontinued.append(
            dict(
                panels=panels,
                reason="ambiguous_k",
                representative_prior="smaller_k+fewest_switches",
                unresolved_search=False,
            )
        )
    # Repeat discovery through reconverged paths does not multiply uncertainty records.
    unique = {}
    for family in uncontinued:
        key = (family["reason"], tuple(sorted(family["panels"])))
        unique.setdefault(key, family)
    unresolved = tuple(unique.values())
    status = (
        "unresolved_budget"
        if overflow
        else "unresolved_count_continuation"
        if any(family["unresolved_search"] for family in unresolved)
        else "ambiguous_k"
        if len(counts) > 1
        else "same_k_tie"
        if live
        else "resolved"
    )
    count_unverified = tuple(
        dict(item, branch=state["engine"].branch_key) for state in leaves for item in state["result"].count_unverified
    )
    count_reports = []

    def digest(panel):
        return hashlib.sha256(repr(panel).encode()).hexdigest()

    for event in count_events:
        histories = []
        for state in leaves:
            if state["input_key"] != event["input_key"]:
                continue
            if any(state["choices"].get(k) != v for k, v in event["fit_prefix"].items()):
                continue
            if any(state["count_choices"].get(k) != v for k, v in event["count_prefix"].items()):
                continue
            selected = state["engine"].ledger.chosen_count_panels.get(event["token"])
            if selected not in event["options"]:
                continue
            point = state["result"]
            histories.append(
                dict(
                    branch=state["engine"].branch_key,
                    source_panel=digest(selected),
                    terminal_panel=digest(tie_rules.panel_key(point.rows)),
                    terminal_K=len(point.rows),
                    terminal_F=float(point.F),
                    terminal_candidates=tuple(
                        dict(panel=digest(p), K=len(p), F=float(f)) for p, f in terminal_panels_of(state).items()
                    ),
                    unverified_history=bool(
                        state.get("unfinished_budget")
                        or remaining_families(state)
                        or point.count_cycle
                        or point.count_limit_reached
                    ),
                    count_cycle=point.count_cycle,
                    count_limit_reached=point.count_limit_reached,
                )
            )
        seen = {h["source_panel"] for h in histories}
        source_complete = not event["overflow"] and len(seen) == len(event["options"])
        unfinished = any(h["unverified_history"] for h in histories)
        candidates = [candidate for h in histories for candidate in h["terminal_candidates"]]
        terminal_counts = sorted({c["K"] for c in candidates})
        terminal_panels = {c["panel"] for c in candidates}
        event_status = (
            "unperformed_budget"
            if event["overflow"]
            else "unperformed_history"
            if not source_complete
            else "continued_with_unverified_history"
            if unfinished
            else "terminal_count_alternatives"
            if len(terminal_counts) > 1
            else "resolved_terminal_panel"
            if len(terminal_panels) == 1
            else "terminal_panels_compared_by_F"
        )
        # Compare complete terminal outputs by the same within-K F classes as
        # the scientific selector. A representative prior never removes an
        # equal-F biological family from this report.
        retained = []
        f_margins = []
        for founders in terminal_counts:
            group = [c for c in candidates if c["K"] == founders]
            values = [c["F"] for c in group]
            keys = [c["panel"] for c in group]
            classes = ties.fixed_policy(
                "assembly.refine_terminal",
                tie_rules.quantum("refine_F", max(abs(v) for v in values)),
                reverse=engine.reverse,
            ).classes(values, content=keys, descending=True)
            winning = {int(i) for i in classes.members(int(classes.classes[classes.winner]))}
            retained.extend(keys[i] for i in winning)
            other = [value for i, value in enumerate(values) if i not in winning]
            f_margins.append(dict(K=founders, margin=None if not other else max(values) - max(other)))
        retained = tuple(sorted(set(retained)))
        final_status = (
            "unverified"
            if not source_complete or unfinished
            else "ambiguous_k"
            if len(terminal_counts) > 1
            else "same_k_tie"
            if len(retained) > 1
            else "resolved"
        )
        if final_status == "unverified":
            # Starting every choice does not complete a family whose descendants
            # hit the joint budget or ordinary count-cycle/backstop.
            family = dict(event["family"], reason="unresolved_count_continuation", unresolved_search=True)
            key = (family["reason"], tuple(sorted(family["panels"])))
            unique.setdefault(key, family)
        count_reports.append(
            dict(
                source_count_key=event["token"],
                kind=event["family"].get("source_count_kind"),
                round=event["family"].get("source_decision"),
                source_choice_count=len(event["options"]),
                source_choice_Ks=tuple(len(p) for p in event["options"]),
                source_panels=tuple(digest(p) for p in event["options"]),
                nominal_source_panel=digest(event["nominal"]),
                evaluated_terminal_histories=len(histories),
                evaluated_source_choices=len(seen),
                source_choices_complete=source_complete,
                unfinished_history=unfinished,
                status=event_status,
                histories=tuple(histories),
                retained_terminal_panels=retained,
                retained_terminal_count=len(retained),
                final_status=final_status,
                resolved=final_status == "resolved",
                within_K_F_margins=tuple(f_margins),
                representative_prior=(
                    "smaller_k+fewest_switches"
                    if len(terminal_counts) > 1
                    else "fewest_switches"
                    if len(retained) > 1
                    else None
                ),
                replay_seconds=event["replay_seconds"],
            )
        )
    unresolved = tuple(unique.values())
    if status == "resolved" and any(family["unresolved_search"] for family in unresolved):
        status = "unresolved_count_continuation"
    return replace(
        result,
        live_alternatives=live,
        unresolved_alternatives=unresolved,
        count_unverified=count_unverified,
        ambiguous=result.ambiguous or len(counts) > 1,
        branch_evaluations=evaluations,
        branch_reused_fits=reused,
        branch_seconds=0.0 - begun,
        branch_status=status,
        selected_branch=chosen["engine"].branch_key,
        input_F0=initial.input_F0,
        count_continuations=tuple(count_reports),
        count_branch_evaluations=count_evaluations,
        count_branch_seconds=count_seconds,
    )


__all__ = ["upper_bound", "keep_margin", "count_pass"]
