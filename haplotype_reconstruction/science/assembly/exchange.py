"""Diplotype exchange on a refined component: another split of a diplotype that one founder row is locked into.

Owns the move's candidates (maximum-path painting, locked runs, anchors), the exchange row, the copy predicate and
the sequential acceptance by the refinement's exact objective. No storage, clocks or scheduling: the stage supplies
the engine that scores a panel (stages.assembly.exchange).
Method: docs/methods/founder_refinement.md#diplotype-exchange.

Why. A founder row is a path through the prepared rows of its blocks. When every carrier of a rare haplotype
carries it with the same partner, a block's genotype data fix the carriers' diplotype but not its split into two
rows, and block stages may hold a split that reuses the wrong neighbour. No path through the existing prepared rows
is the other split; this move builds it.

The move, on one component after its last refinement fit:

1. Paint: every sample's maximum path under the refinement objective F over unordered founder pairs (paint; the
   objective's site terms and map-derived switch costs; canonical ties).
2. Locked runs: row v is locked to u on a maximal run of sites where every sample whose path uses v is in the
   state {u, v}, u != v. Its carriers are the samples that use v anywhere in the run.
3. Anchors: every founder w (not u, not v) that a sample using v at an end of the run pairs with v at the
   neighbouring site outside it. One candidate per anchor: up to min(K - 2, 2 x carriers) per run.
4. Exchange row X on the run's objective sites: X = v where u = v; X = 1 - w where u != v; uncalled where a needed
   call is missing. Where u != v the pair {w, X} has the dosage of {u, v}; where u = v != w it is one dose off and the
   exact objective charges it. No allele is estimated from evidence (PreparedPanel.inference stays free of feedback).
5. Copy predicate: X is compatible with a founder row k when no objective site of the run has X and k both called
   and different. A candidate whose X is compatible with ANY of the K rows (v and non-partner rows included) is a
   copy and is never scored: it would put no new called allele content into the panel on the run. Missing calls
   never make X new.
6. Every non-copy candidate must improve the cohort objective, including after deleting its most-gaining sample.
   Its carrier-backed gain must cover the conditional edit cost and remain positive after deleting the strongest
   carrier (shared_exchange_support). Each comparison uses the same acceptance quantum. Commit the largest passing
   cohort gain, breaking ties by (v, start, w), then repaint and derive the candidates again.

An accepted X becomes one appended prepared row in each block of its run: the source row (the row v's path held)
on all four published fields outside the run's sites, X inside. Outside the run an uncalled allele of the appended
row is the SAME latent allele as the source row's (latent identity, component_workspace.coded_leaf), so the
objective's unknown-allele classes there are those of the source row.

Cost per round: one painting O(S L K^2); candidates are at most K locked runs per site times A anchors,
A <= min(K - 2, 2S); each is scored over the whole component by the refinement engine, O(S L K^2). Exact scoring over
the run with boundary messages would remove one factor L and is not built. Every sample is scored for every
candidate (no carrier-only shortcut). Uncalled alleles are coded in int16 as -2 - row (LEAF_ROW_CAPACITY rows per
block): trial panels are built in batches that respect it, and a committed row that would exceed it raises.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numba import prange, types

from haplotype_reconstruction.contracts.panels import PreparedPanel
from haplotype_reconstruction.core.errors import InvariantError
from haplotype_reconstruction.core.kernels import kernel
from haplotype_reconstruction.science.shared.emissions import (
    centred_likelihoods,
    predictive_likelihood,
    robust_log_likelihood,
)

from .refinement.component_workspace import LEAF_ROW_CAPACITY, coded_leaf, frozen
from .refinement.objective import pair_code7, unordered_pairs
from .refinement.parameters import ACCEPTANCE_GAIN_FLOOR

_HAPS = types.Array(types.int16, 2, "C", readonly=True)
_EVIDENCE = types.Array(types.float32, 3, "C", readonly=True)
_MASK = types.Array(types.boolean, 1, "C", readonly=True)
_HAZARDS = types.Array(types.float64, 1, "C", readonly=True)
_STATES = types.Array(types.int16, 2, "C", readonly=True)


@kernel(overloads=((_HAPS, _EVIDENCE, _MASK, _HAZARDS),), parallel=True, nogil=True, cache=True)
def paint(haps, evidence, kept, hazards):
    """int16 (S, L): the unordered founder-pair state (objective.unordered_pairs order) of every sample's maximum
    path under the refinement objective.

    haps: int16 (K, L) coded alleles of the founder paths (component_workspace.coded_leaf gathered along the
    paths); evidence: float32 (S, L, 3) the component's centred scoring view; kept: bool (L) the objective's
    sites; hazards: float64 (L) reset hazards (component_workspace.map_hazards). Site terms: the robust log of the
    pair class's centred likelihood at the sample's informative objective sites; a homologue that changes founder
    pays c = log(1 - rho (1 - 1/K)) - log(rho / K), rho from the hazard summed since the sample's previous
    informative site. Ties: stay, then one homologue, then both; the first maximiser in pair order. A site without
    evidence takes the state of the previous informative site (leading ones the first); -1 for a sample without any.
    Parallel over samples; a sample's result does not depend on the team.
    """
    founders, sites = haps.shape
    samples = evidence.shape[0]
    pairs = founders * (founders + 1) // 2
    first, second = unordered_pairs(founders)
    states = np.full((samples, sites), -1, np.int16)
    for s in prange(samples):
        informative = np.zeros(sites, np.bool_)
        count = 0
        for f in range(sites):
            if kept[f]:
                flag, _, _, _ = centred_likelihoods(evidence[s, f, 0], evidence[s, f, 1], evidence[s, f, 2])
                if flag:
                    informative[f] = True
                    count += 1
        if count == 0:
            continue
        back = np.empty((count, pairs), np.int16)
        score = np.zeros(pairs)
        new = np.empty(pairs)
        best_of = np.empty(founders)
        arg_of = np.empty(founders, np.int64)
        e = np.empty(7)
        index = 0
        mass = 0.0
        started = False
        for f in range(sites):
            if started:
                mass += hazards[f]
            if not informative[f]:
                continue
            p0, p1, p2 = np.float64(evidence[s, f, 0]), np.float64(evidence[s, f, 1]), np.float64(evidence[s, f, 2])
            for code in range(7):
                e[code] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, code))
            if not started:
                for z in range(pairs):
                    score[z] = e[pair_code7(haps[first[z], f], haps[second[z], f])]
                    back[index, z] = z
                started = True
            else:
                rho = -np.expm1(-mass)
                if rho <= 0.0:
                    c = np.inf
                elif rho >= 1.0:
                    c = 0.0
                else:
                    c = np.log1p(-rho * (1.0 - 1.0 / founders)) - np.log(rho) + np.log(float(founders))
                for a in range(founders):
                    best_of[a] = -np.inf
                    arg_of[a] = -1
                overall = -np.inf
                overall_arg = -1
                for z in range(pairs):
                    value = score[z]
                    if value > best_of[first[z]]:
                        best_of[first[z]] = value
                        arg_of[first[z]] = z
                    if value > best_of[second[z]]:
                        best_of[second[z]] = value
                        arg_of[second[z]] = z
                    if value > overall:
                        overall = value
                        overall_arg = z
                for z in range(pairs):
                    value = score[z]
                    source = z
                    a, b = first[z], second[z]
                    one = best_of[a] - c
                    one_source = arg_of[a]
                    if best_of[b] - c > one:
                        one = best_of[b] - c
                        one_source = arg_of[b]
                    if one > value:
                        value = one
                        source = one_source
                    if overall - 2.0 * c > value:
                        value = overall - 2.0 * c
                        source = overall_arg
                    new[z] = value + e[pair_code7(haps[a, f], haps[b, f])]
                    back[index, z] = source
                for z in range(pairs):
                    score[z] = new[z]
            mass = 0.0
            index += 1
        state = 0
        for z in range(1, pairs):
            if score[z] > score[state]:
                state = z
        index = count - 1
        f = sites - 1
        while f >= 0:
            states[s, f] = state
            if informative[f]:
                state = back[index, state]
                index -= 1
            f -= 1
    return states


@kernel(overloads=((_STATES, types.int64, types.int64),), parallel=True, nogil=True, cache=True)
def locked_partners(states, founders, chunks):
    """int16 (K, L): the founder u that row v is locked to at each site (every sample using v is in {u, v},
    u != v); -1 where v is unused, used with several partners, or used twice by one sample.

    states: paint's result; chunks: the number of site ranges worked in parallel (sites are independent, so the
    result does not depend on it)."""
    samples, sites = states.shape
    first, second = unordered_pairs(founders)
    partner = np.full((founders, sites), -1, np.int16)
    for chunk in prange(chunks):
        users = np.zeros(founders, np.int64)
        for f in range(sites * chunk // chunks, sites * (chunk + 1) // chunks):
            for v in range(founders):
                users[v] = 0
            for s in range(samples):
                z = states[s, f]
                if z < 0:
                    continue
                a, b = first[z], second[z]
                users[a] += 1
                if users[a] == 1:
                    partner[a, f] = b
                elif partner[a, f] != b:
                    partner[a, f] = -2
                if a != b:
                    users[b] += 1
                    if users[b] == 1:
                        partner[b, f] = a
                    elif partner[b, f] != a:
                        partner[b, f] = -2
            for v in range(founders):
                if partner[v, f] == v or partner[v, f] == -2:
                    partner[v, f] = -1
    return partner


def pair_tables(founders: int) -> tuple[np.ndarray, np.ndarray]:
    """first[q], second[q] of the unordered pairs, in the objective's order."""
    upper = np.triu_indices(int(founders))
    return upper[0].astype(np.int64), upper[1].astype(np.int64)


def candidates(states: np.ndarray, partner: np.ndarray) -> list[dict[str, Any]]:
    """Locked runs with an anchor: dicts v, u, w, start, stop (component sites, half-open), carriers (sample rows).

    states: paint's int16 (S, L); partner: locked_partners' int16 (K, L). carriers: every sample whose path uses v
    at some site of the run. One candidate per distinct anchor w: a founder (not u, not v) that a sample using v at
    an end of the run pairs with v at the neighbouring site outside it. Order: v, start, w ascending."""
    founders, sites = partner.shape
    first, second = pair_tables(founders)
    found = []
    for v in range(founders):
        row = partner[v]
        bounds = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, sites]
        for left, right in zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True):
            u = int(row[left])
            if u < 0:
                continue
            anchors: set[int] = set()
            for inside, outside in ((left, left - 1), (right - 1, right)):
                if outside < 0 or outside >= sites:
                    continue
                at_end = states[:, inside]
                users = (at_end >= 0) & ((first[at_end] == v) | (second[at_end] == v))
                beyond = states[users, outside]
                beyond = beyond[beyond >= 0]
                a, b = first[beyond], second[beyond]
                anchors.update(np.where(a == v, b, a)[(a == v) | (b == v)].tolist())
            anchors -= {u, v}
            if not anchors:
                continue
            block = states[:, left:right]
            uses = ((block >= 0) & ((first[block] == v) | (second[block] == v))).any(axis=1)
            carriers = [int(s) for s in np.flatnonzero(uses)]
            for w in sorted(anchors):
                found.append(dict(v=v, u=u, w=int(w), start=int(left), stop=int(right), carriers=carriers))
    return found


def exchange_row(v: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """int8: X = v where u = v, else 1 - w; uncalled (-1) where a needed call is missing (v and u always, w where
    u != v). Arguments are alleles 0/1, any negative value meaning uncalled."""
    v, u, w = np.asarray(v), np.asarray(u), np.asarray(w)
    result = np.full(v.shape, -1, np.int8)
    same = (u >= 0) & (v >= 0) & (u == v)
    result[same] = v[same]
    different = (u >= 0) & (v >= 0) & (u != v) & (w >= 0)
    result[different] = 1 - w[different]
    return result


def is_copy(x: np.ndarray, haps: np.ndarray, sites: np.ndarray) -> bool:
    """The copy predicate: True when SOME founder row has no called disagreement with x on the run.

    x: int (l) the exchange row on the run; haps: int (K, l) the founder rows' alleles there (negative = uncalled),
    row v included; sites: bool (l) the run's objective sites. An uncalled allele on either side agrees with
    anything, so missing calls never make x new, and a row that only erases calls of v is a copy of v."""
    x, haps = np.asarray(x), np.asarray(haps)
    both = np.asarray(sites, np.bool_)[None, :] & (haps >= 0) & (x[None, :] >= 0)
    return bool(np.any(~np.any(both & (haps != x[None, :]), axis=1)))


def with_exchange_row(panel: PreparedPanel, source: int, start: int, stop: int, alleles: np.ndarray) -> PreparedPanel:
    """The prepared panel with one exchange row appended.

    The new row is the block's row ``source`` on every field, except on the block-local sites [start, stop) of the
    run, where inference and released are ``alleles`` (int8, -1 uncalled), the ALT probability is the allele (0.5
    where uncalled) and no supporter is recorded. Inside the run a completion fill or a call at a site outside the
    objective that belonged to the source row's content is not carried over."""
    alleles = np.asarray(alleles, np.int8)
    if alleles.shape != (stop - start,) or not 0 <= start < stop <= panel.sites or not 0 <= source < panel.rows:
        raise ValueError("exchange row needs a source row and the alleles of its block-local site range")
    if panel.rows + 1 > LEAF_ROW_CAPACITY:
        raise InvariantError("a block cannot hold more prepared rows than the int16 code of uncalled alleles")

    def extended(name: str, inside: np.ndarray) -> np.ndarray:
        field = np.asarray(getattr(panel, name))
        row = field[source].copy()
        row[start:stop] = inside
        return np.vstack([field, row[None, :]])

    return replace(
        panel,
        released=extended("released", alleles),
        inference=extended("inference", alleles),
        alt_probability=extended("alt_probability", np.where(alleles >= 0, alleles, 0.5).astype(np.float64)),
        support=extended("support", np.zeros(stop - start)),
    )


def extended_identity(
    identity: np.ndarray | None, rows: int, sites: int, source: int, start: int, stop: int
) -> np.ndarray:
    """int16 (rows + 1, sites): a block's latent identities with one exchange row appended.

    identity[r, t] is the row whose unknown allele an uncalled cell (r, t) shares (None: every row its own). The
    appended row shares the source row's outside [start, stop) and has its own inside."""
    if identity is None:
        identity = np.repeat(np.arange(rows, dtype=np.int16)[:, None], sites, axis=1)
    row = np.asarray(identity)[source].copy()
    row[start:stop] = rows
    return np.vstack([identity, row[None, :]])


def coded_alleles(
    panels: Sequence[PreparedPanel], rows: np.ndarray, latent: Sequence[np.ndarray | None]
) -> np.ndarray:
    """int16 (K, L), read-only: the founder paths' coded alleles (coded_leaf of every block, gathered)."""
    parts = [coded_leaf(panel, latent[block])[rows[:, block]] for block, panel in enumerate(panels)]
    return frozen(np.concatenate(parts, axis=1), np.int16)


def _entries(candidate: dict[str, Any], x: np.ndarray, rows: np.ndarray, offsets: np.ndarray) -> list[tuple]:
    """[(block, source row, block-local start, stop, alleles)] of one candidate's exchange row."""
    v, left, right = candidate["v"], candidate["start"], candidate["stop"]
    first_block = int(np.searchsorted(offsets, left, side="right") - 1)
    last_block = int(np.searchsorted(offsets, right - 1, side="right") - 1)
    entries = []
    for block in range(first_block, last_block + 1):
        lo, hi = int(offsets[block]), int(offsets[block + 1])
        start, stop = max(left, lo), min(right, hi)
        entries.append((block, int(rows[v, block]), start - lo, stop - lo, x[start - left : stop - left]))
    return entries


def _batches(trials: list[tuple], panels: Sequence[PreparedPanel], capacity: int) -> list[list[tuple]]:
    """The round's candidates in consecutive batches whose appended rows fit ``capacity`` rows in every block."""
    batches: list[list[tuple]] = []
    current: list[tuple] = []
    load: dict[int, int] = {}
    for trial in trials:
        blocks = [entry[0] for entry in trial[1]]
        if any(panels[block].rows + 1 > capacity for block in blocks):
            raise InvariantError("a block cannot hold more prepared rows than the int16 code of uncalled alleles")
        if current and any(panels[block].rows + load.get(block, 0) + 1 > capacity for block in blocks):
            batches.append(current)
            current, load = [], {}
        current.append(trial)
        for block in blocks:
            load[block] = load.get(block, 0) + 1
    if current:
        batches.append(current)
    return batches


def shared_exchange_support(
    delta: np.ndarray, carriers: Sequence[int], founders: int, sites: int
) -> tuple[float, float, float]:
    """Carrier-backed gain, gain after the strongest carrier is removed, and the conditional edit cost in nats.

    Retain every cohort loss. Only old carriers supply positive evidence about the locked row.
    On the old paint a boundary, its inside direction, v and the rank of w given v,u identify a trial.
    """
    carrier_gain = delta[np.asarray(carriers, np.intp)]
    supported = float(np.minimum(delta, 0.0).sum() + np.maximum(carrier_gain, 0.0).sum())
    residual = supported - max(0.0, float(carrier_gain.max()))
    description = math.log(2 * (sites - 1) * founders * (founders - 2))
    return supported, residual, description


def exchange_pass(
    rows: np.ndarray,
    panels: Sequence[PreparedPanel],
    latent: Sequence[np.ndarray | None],
    evidence: np.ndarray,
    kept: np.ndarray,
    hazards: np.ndarray,
    build: Callable[[Sequence[PreparedPanel], Sequence[np.ndarray | None]], Any],
    team: Any,
    *,
    row_capacity: int = LEAF_ROW_CAPACITY,
) -> tuple[np.ndarray, list[PreparedPanel], list[np.ndarray | None], list[tuple], dict[str, Any]]:
    """Commit exchange rows one at a time until no candidate passes.

    rows: int64 (K, B) the founder paths; panels: the component's prepared panels (earlier exchange rows included);
    latent: their latent identities (None for a block without exchange rows); evidence: float32 (S, L, 3) the
    centred scoring view; kept: bool (L) the objective's sites (ComponentWorkspace.kept; exchange rows never change
    it); hazards: float64 (L); build(panels, latent) -> the refinement engine that scores a panel exactly;
    team: the caller's thread team (engine.per_sample's budget; team.threads() chunks the locked-run scan).

    Returns (rows, panels, latent, entries, report). entries: the appended rows in commitment order
    [(block, source row, start, stop, alleles)]. report: rounds (candidates, copies, scored, passing, F of the
    round's panel), accepted (v, u, w, sites, carriers, per-sample gains, F before and after), F_committed (the
    expected exact score of the returned panel; None without a commitment).
    """
    rows = np.ascontiguousarray(rows, np.int64).copy()
    panels, latent = list(panels), list(latent)
    founders = rows.shape[0]
    report: dict[str, Any] = dict(rounds=[], accepted=[], F_committed=None)
    entries_out: list[tuple] = []
    if founders < 3 or rows.shape[1] < 1:  # no third row to anchor on
        return rows, panels, latent, entries_out, report
    offsets = np.concatenate([[0], np.cumsum([panel.sites for panel in panels])]).astype(np.int64)
    positions = np.concatenate([np.asarray(panel.positions) for panel in panels])
    evidence, kept, hazards = frozen(evidence, np.float32), frozen(kept, np.bool_), frozen(hazards, np.float64)
    floor = float(ACCEPTANCE_GAIN_FLOOR)
    committed: float | None = None
    while True:
        haps = coded_alleles(panels, rows, latent)
        states = paint(haps, evidence, kept, hazards)
        partner = locked_partners(frozen(states, np.int16), founders, max(1, int(team.threads())))
        found = candidates(states, partner)
        record: dict[str, Any] = dict(candidates=len(found), copies=0, scored=0, passing=0, F=None)
        report["rounds"].append(record)
        trials = []
        for candidate in found:
            left, right = candidate["start"], candidate["stop"]
            sites = np.asarray(kept[left:right])
            run = np.where(haps[:, left:right] >= 0, haps[:, left:right], -1).astype(np.int8)
            v, u, w = (run[candidate[name]] for name in ("v", "u", "w"))
            x = np.where(sites, exchange_row(v, u, w), v).astype(np.int8)
            if is_copy(x, run, sites):
                record["copies"] += 1
                continue
            called = sites & (v >= 0) & (u >= 0) & (w >= 0)
            candidate.update(
                changed_sites=int((x != v).sum()),
                dose_off_sites=int((called & (v == u) & (w != v)).sum()),
                start_bp=int(positions[left]),
                stop_bp=int(positions[right - 1]),
            )
            trials.append((candidate, _entries(candidate, x, rows, offsets)))
        if not trials:
            break
        best = None
        for batch in _batches(trials, panels, row_capacity):
            trial_panels, trial_latent, paths = list(panels), list(latent), []
            for candidate, entries in batch:
                path = rows.copy()
                for block, source, start, stop, alleles in entries:
                    panel = trial_panels[block]
                    trial_latent[block] = extended_identity(
                        trial_latent[block], panel.rows, panel.sites, source, start, stop
                    )
                    trial_panels[block] = with_exchange_row(panel, source, start, stop, alleles)
                    path[candidate["v"], block] = panel.rows
                paths.append(path)
            engine = build(trial_panels, trial_latent)
            base = np.asarray(engine.per_sample(rows, team), np.float64)
            current = float(base.sum())
            if record["F"] is None:
                record["F"] = current
                if committed is not None:
                    # The committed panel on this round's engine: equal to its commitment up to summation order.
                    report["accepted"][-1]["F_recheck"] = current
                    if current < committed - floor:
                        raise InvariantError("an exchange commitment lowered the exact objective")
            for (candidate, entries), path in zip(batch, paths, strict=True):
                delta = np.asarray(engine.per_sample(path, team), np.float64) - base
                gain = float(delta.sum())
                without_top = float(delta.sum() - delta.max())
                candidate.update(gain=gain, gain_without_top=without_top, F_before=current, F_after=current + gain)
                record["scored"] += 1
                if gain > floor and without_top > floor:
                    backed, residual, description = shared_exchange_support(
                        delta, candidate["carriers"], founders, haps.shape[1]
                    )
                    if backed - description <= floor or residual <= floor:
                        continue
                    candidate.update(carrier_backed_gain=backed, carrier_residual=residual,
                                     edit_code_nats=description, component_sites=int(haps.shape[1]))
                    record["passing"] += 1
                    key = (-gain, candidate["v"], candidate["start"], candidate["w"])
                    if best is None or key < best[0]:
                        changes = [[int(s), float(delta[s])] for s in np.flatnonzero(delta != 0.0)]
                        best = (key, candidate, entries, changes)
        if best is None:
            break
        _, candidate, entries, changes = best
        for block, source, start, stop, alleles in entries:
            panel = panels[block]
            latent[block] = extended_identity(latent[block], panel.rows, panel.sites, source, start, stop)
            panels[block] = with_exchange_row(panel, source, start, stop, alleles)
            rows[candidate["v"], block] = panel.rows
            entries_out.append((block, source, start, stop, np.asarray(alleles, np.int8)))
        committed = float(candidate["F_after"])
        report["accepted"].append(
            dict(candidate, sample_gains=changes, rows=len(entries), round=len(report["rounds"]) - 1)
        )
    report["F_committed"] = committed
    return rows, panels, latent, entries_out, report


__all__ = [
    "candidates",
    "coded_alleles",
    "exchange_pass",
    "exchange_row",
    "extended_identity",
    "is_copy",
    "locked_partners",
    "paint",
    "pair_tables",
    "with_exchange_row",
]
