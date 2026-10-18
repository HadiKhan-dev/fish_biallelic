"""The family MAP: posterior mode of the family model by monotone exact block coordinate ascent (release 2.1).

Model (unchanged; potentials.py, factors.py, solver.py). Per chromosome: ordered genotypes G (four states per sample
and site), one frame chain F per pedigree root (does its frame match the scaffold orientation; switches only where
its chain may: painting-bin edges, probability 1/2 at component edges, never inside a component for a root with a
single child edge, state 0 at the root's anchor) and one joint four-state selector chain S per child (one bit per
parent slot; switches on the child's supported markers with the composed Haldane probabilities of
factors.selector_support, continuous across component edges).

Estimator (design/FAMILY_MAP.md, review I2): the mode of P(F, S, G_children | data) with every ROOT's genotypes
summed out per site (the Lander-Green treatment of founder alleles; allele sharing between a root's children keeps its
IBD evidence where the root's scaffold has fewer than two called alleles). Its log, J, is

    J = sum_children sum_l psi(G_cl) + sum_children [log 1/4 + chain(S_c)] + sum_roots [log 1/2 + chain(F_r)]
        + sum_roots sum_l R_rl + sum_children sum_l sum_{slots with an observed non-root parent} log copy_slot
        + (log 1/2 per unobserved slot and site),
    R_rl = log sum_g exp( psi_r(g) + log phi(g | F_rl) + sum_{children x of r} log copy_slot(x's allele | g, S_xl) ),

with phi = 1/2 for homozygotes and unoriented sites, 1 - e_phase for the heterozygote the frame matches and e_phase for
the other, copy_slot = 1 - e (transmitted allele) or e; chain terms log(1 - theta) / log(theta) per bit and interval.

Algorithm (monotone exact block coordinate ascent):
  - child block (S_x, G_x): Viterbi over x's supported markers; the emission of state s at a site is the max over G_x
    of psi + the slot terms + x's own children's copy terms, a root parent's slot term being R of that root with x's
    factor in (its cavity without x is fixed while x moves);
  - root block (F_r): Viterbi over r's frame-chain runs with run emissions sum_l R_rl(f);
  - relabelling moves (review B1, gauges.best_gauge_flips semantics): flip root r's homologue labels from a boundary
    of its frame chain to the chromosome end, re-gauged at its anchor (the side without the anchor is flipped):
    r's frame and r's bit in every child's selector. R and every other term are symmetric under it, so the change of J
    is r's frame transition plus each child's composed selector transition across the boundary. Component edges are
    candidates (frame term 0, the children's terms decide); single-child roots move only there. Applied sequentially.
  - compound moves (review I1; _compound, _relabel_compound): a seed change followed by exact windowed re-decodes of
    the blocks around it, accepted as a whole on its exact change of J (_window_objective over the terms it can alter,
    _term_masks): kind 0 removes a child's selector change pair; kind 1 swaps a non-root parent's tracks on an interval
    together with its children's bits for it; kind 2 moves the children of a root that share a change pair back to the
    root's other homologue together; the compound relabelling relabels a root at the frame boundary nearest a child's
    change of its bit and re-decodes its children and frame around it;
  - site moves (_site_move): at a Mendelian-conflict site, the exact joint maximum over a sibship's ordered genotypes
    (and a non-root parent's), selectors fixed;
  - acceptance (review B2, I5): a block's new path is split into the maximal intervals where it differs from the
    current configuration (intervals coupled through a transition are merged); each interval or move is accepted only
    if its exact score difference (one scoring routine, before and after) exceeds GAIN_QUANTUM; otherwise the current
    configuration is kept. Viterbi ties take the fewest switches, then the current state, then the lowest.
  - schedule: colours of the conflict graph (parent-child edges plus sibling cliques of each root's children, review
    M2/I2), samples in canonical order; blocks of a colour share no free variable, so the result does not depend on
    thread count; then every root's relabelling scan, the compound moves, the compound relabellings and the site
    moves, sequentially in canonical order. After every sweep the from-scratch J must have risen by the accepted gains
    (asserted). Stop when a sweep accepts nothing; reaching map_max_sweeps is an error.
  - initialisation (one top-down sum-product pass, _init_block): F = 0; children in generation order, each set to its
    exact block optimum given its parents' genotypes summed (a root's psi, phi and already initialised children; a
    non-root parent's stored conditional), so that tied parental genotype orders carry no phase evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from numba import njit, prange

from ..maths import chains

LOG_HALF = math.log(0.5)
LOG_QUARTER = math.log(0.25)
# Acceptance quantum of a step's exact gain (nats). Placed by census of interval gains (work/IE/map_iter census,
# review I5c); see docs/methods/family_phase.md.
GAIN_QUANTUM = 2.0 ** -17
# Genotype ties within one site's maximisation (sums of a few log terms): within this the current genotype is kept.
GENOTYPE_TIE = 1e-12
# Compound moves (review I1): change pairs at most COMPOUND_MAX_SPAN sites apart are candidates; the re-decode window
# widens the pair's interval by max(COMPOUND_MARGIN, min(its span, COMPOUND_WIDTH)) on each side.
COMPOUND_MAX_SPAN = 8192
COMPOUND_MARGIN = 64
COMPOUND_WIDTH = 1024
# Site moves: joint configurations of a sibship's genotypes enumerated at a Mendelian-conflict site, at most this many.
SITE_MAX_COMBOS = 4096


@dataclass
class MapState:
    """The family MAP configuration: genotype int8 [N, L] (ordered genotypes of non-root samples; roots' rows are
    unused, their genotypes being summed out), selector int8 [G, L] (joint selector state of each child group, read on
    its supported markers and carried forward between them), frame int8 [N, L] (roots' frame chains, 0 elsewhere)."""

    genotype: np.ndarray
    selector: np.ndarray
    frame: np.ndarray


@dataclass(frozen=True)
class MapResult:
    """state: the converged MapState; sweeps: sweeps run (the last accepted nothing); accepted: accepted steps per
    sweep (block intervals plus relabelling moves); objective: J after the initialisation and after every sweep;
    decisions: (kind, block, position, gain) of every evaluated step whose |gain| < 1 nat, for the gain census
    (kind 0 child interval, 1 root interval, 2 relabelling move, 3-5 compound move kinds 0-2, 6 compound relabelling,
    7 site move)."""

    state: MapState
    sweeps: int
    accepted: tuple[int, ...]
    objective: tuple[float, ...]
    decisions: tuple[tuple[int, int, int, float], ...]


@njit(cache=True, inline="always")
def _log_theta(theta):
    return math.log(theta) if theta > 0 else -np.inf


@njit(cache=True, inline="always")
def _log_stay(theta):
    return math.log(1 - theta) if theta < 1 else -np.inf


@njit(cache=True, inline="always")
def _phi(genotype, frame, orient, log_match, log_miss):
    """log phi(G | F) of a root at one site (scaffold orientation ``orient``, -1 for none)."""
    if orient < 0 or genotype == 0 or genotype == 3:
        return LOG_HALF
    matching = 1 if orient == 0 else 2
    if frame == 1:
        matching = 3 - matching
    return log_match if genotype == matching else log_miss


@njit(cache=True, inline="always")
def _allele(genotype, track):
    return (genotype >> (1 - track)) & 1


@njit(cache=True, inline="always")
def _lse4(v0, v1, v2, v3):
    top = max(max(v0, v1), max(v2, v3))
    if top == -np.inf:
        return -np.inf
    return top + math.log(((math.exp(v0 - top) + math.exp(v1 - top)) + math.exp(v2 - top)) + math.exp(v3 - top))


@njit(cache=True, inline="always")
def _transition(previous, state, theta):
    changes = previous ^ state
    count = (changes >> 1) + (changes & 1)
    if count == 0:
        return 2 * _log_stay(theta)
    if count == 1:
        return _log_theta(theta) + _log_stay(theta)
    return 2 * _log_theta(theta)


@njit(cache=True)
def _root_cavity(root, site, frame_value, exclude_group, genotype, selector, potentials, orientation, group_child,
                 child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, out):
    """out[g] = psi_r(g) + log phi(g | frame_value) + the copy terms of r's children except group ``exclude_group``
    (-2: none of them, -1: all of them)."""
    orient = orientation[root, site]
    for g in range(4):
        out[g] = potentials[root, site, g] + _phi(g, frame_value, orient, log_match, log_miss)
    if exclude_group == -2:
        return
    for index in range(child_offsets[root], child_offsets[root + 1]):
        group = child_groups[index]
        if group == exclude_group:
            continue
        slot = child_slots[index]
        child_allele = _allele(genotype[group_child[group], site], slot)
        homologue = (selector[group, site] >> (1 - slot)) & 1
        for g in range(4):
            out[g] += log_keep if _allele(g, homologue) == child_allele else log_error


@njit(cache=True)
def _root_term(root, site, frame_value, genotype, selector, potentials, orientation, group_child, child_offsets,
               child_groups, child_slots, log_keep, log_error, log_match, log_miss):
    """R_rl with frame ``frame_value`` (every child's factor in)."""
    cavity = np.empty(4)
    _root_cavity(root, site, frame_value, -1, genotype, selector, potentials, orientation, group_child, child_offsets,
                 child_groups, child_slots, log_keep, log_error, log_match, log_miss, cavity)
    return _lse4(cavity[0], cavity[1], cavity[2], cavity[3])


@njit(cache=True)
def _slot_tables(group, site, genotype, selector, frame, potentials, orientation, roots, group_parent, group_child,
                 child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, siblings, table):
    """table[slot, a, h]: group's slot term at ``site`` when the child's slot allele is a and the parent's homologue h.
    Root parent: R of the parent with the child's factor in (its cavity without the child, every other child in when
    ``siblings``, none of them otherwise); non-root parent: its explicit genotype's copy term; unobserved slot: 1/2."""
    cavity = np.empty(4)
    for slot in range(2):
        parent = group_parent[group, slot]
        if parent < 0:
            for a in range(2):
                for h in range(2):
                    table[slot, a, h] = LOG_HALF
        elif roots[parent]:
            _root_cavity(parent, site, frame[parent, site], group if siblings else -2, genotype, selector, potentials,
                         orientation, group_child, child_offsets, child_groups, child_slots, log_keep, log_error,
                         log_match, log_miss, cavity)
            for a in range(2):
                for h in range(2):
                    v0 = cavity[0] + (log_keep if _allele(0, h) == a else log_error)
                    v1 = cavity[1] + (log_keep if _allele(1, h) == a else log_error)
                    v2 = cavity[2] + (log_keep if _allele(2, h) == a else log_error)
                    v3 = cavity[3] + (log_keep if _allele(3, h) == a else log_error)
                    table[slot, a, h] = _lse4(v0, v1, v2, v3)
        else:
            parent_genotype = genotype[parent, site]
            for a in range(2):
                for h in range(2):
                    table[slot, a, h] = log_keep if _allele(parent_genotype, h) == a else log_error


@njit(cache=True)
def _own_children_terms(sample, site, genotype, selector, group_child, child_offsets, child_groups, child_slots,
                        log_keep, log_error, out):
    """out[g] += the copy terms of non-root ``sample``'s own children when it holds genotype g."""
    for index in range(child_offsets[sample], child_offsets[sample + 1]):
        group = child_groups[index]
        slot = child_slots[index]
        child_allele = _allele(genotype[group_child[group], site], slot)
        homologue = (selector[group, site] >> (1 - slot)) & 1
        for g in range(4):
            out[g] += log_keep if _allele(g, homologue) == child_allele else log_error


@njit(cache=True)
def _child_score(group, a, b, child_genotype, selector, marker_offsets, marker_sites, marker_theta, base, table):
    """The child block's terms on sites [a, b) plus its selector transitions between the last marker before a and the
    first marker at or after b (base [L, 4] and table [L, 2, 2, 2] precomputed for the block). The one routine for the
    block's before and after scores (review B2)."""
    total = 0.0
    first, stop = marker_offsets[group], marker_offsets[group + 1]
    for site in range(a, b):
        g = child_genotype[site]
        s = selector[group, site]
        total += base[site, g] + table[site, 0, _allele(g, 0), (s >> 1) & 1] + table[site, 1, _allele(g, 1), s & 1]
    markers = stop - first
    if markers < 2:
        return total
    lo = np.searchsorted(marker_sites[first:stop], a)          # first marker >= a
    hi = np.searchsorted(marker_sites[first:stop], b)          # first marker >= b
    k0 = max(lo - 1, 0)
    k1 = min(hi, markers - 1)
    for k in range(k0, k1):
        total += _transition(selector[group, marker_sites[first + k]], selector[group, marker_sites[first + k + 1]],
                             marker_theta[first + k])
    return total


@njit(cache=True)
def _block_path(group, child, lo, base, table, genotype, selector, marker_offsets, marker_sites, marker_theta, a_idx,
                b_idx, left, right, new_selector, new_genotype):
    """A child block's optimum over sites [lo, lo + len(base)) given its terms there (base [n, 4], slot tables
    [n, 2, 2, 2]): Viterbi over the group's markers a_idx..b_idx-1 (emission: max over G), joined to the pinned
    selector states ``left`` / ``right`` of the markers just outside (-1: none), the selector carried forward between
    markers (backward before the group's first marker), the genotype the argmax given the selector. Ties: fewest
    switches, then the current state, then the lowest; a genotype within GENOTYPE_TIE of the best keeps the current
    one. Fills new_selector and new_genotype [n]."""
    n = base.shape[0]
    markers = b_idx - a_idx
    path = np.zeros(max(markers, 1), dtype=np.int8)
    if markers > 0:
        emission = np.empty((markers, 4))
        for k in range(markers):
            i = marker_sites[a_idx + k] - lo
            for s in range(4):
                best = -np.inf
                for g in range(4):
                    if base[i, g] == -np.inf:
                        continue
                    value = base[i, g] + table[i, 0, _allele(g, 0), (s >> 1) & 1] + table[i, 1, _allele(g, 1), s & 1]
                    if value > best:
                        best = value
                emission[k, s] = best
        score = np.empty(4)
        new = np.empty(4)
        back = np.zeros((markers, 4), dtype=np.int8)
        for s in range(4):
            score[s] = emission[0, s] + (_transition(left, s, marker_theta[a_idx - 1]) if left >= 0 else 0.0)
        for k in range(1, markers):
            theta = marker_theta[a_idx + k - 1]
            prior_state = selector[group, marker_sites[a_idx + k - 1]]
            for s in range(4):
                top = -np.inf
                for p in range(4):
                    value = score[p] + _transition(p, s, theta)
                    if value > top:
                        top = value
                arg, fewest = 0, 3
                found = False
                for p in range(4):
                    value = score[p] + _transition(p, s, theta)
                    if value >= top - chains.CHAIN_TIE and value > -np.inf:
                        changes = p ^ s
                        count = (changes >> 1) + (changes & 1)
                        if (not found) or count < fewest or (count == fewest and p == prior_state):
                            fewest, arg, found = count, p, True
                back[k, s] = arg
                new[s] = top + emission[k, s]
            for s in range(4):
                score[s] = new[s]
        if right >= 0:
            for s in range(4):
                score[s] += _transition(s, right, marker_theta[b_idx - 1])
        top = max(max(score[0], score[1]), max(score[2], score[3]))
        state = selector[group, marker_sites[b_idx - 1]]
        if score[state] < top - chains.CHAIN_TIE:
            for s in range(4):
                if score[s] >= top - chains.CHAIN_TIE:
                    state = s
                    break
        for k in range(markers - 1, -1, -1):
            path[k] = state
            if k > 0:
                state = back[k, state]
    if left >= 0:
        current = left
    elif markers > 0:
        current = path[0]
    elif right >= 0:
        current = right
    else:
        current = selector[group, lo]
    k = 0
    for i in range(n):
        site = lo + i
        if k < markers and marker_sites[a_idx + k] == site:
            current = path[k]
            k += 1
        new_selector[i] = current
        s = current
        best, arg = -np.inf, -1
        g_now = genotype[child, site]
        for g in range(4):
            if base[i, g] == -np.inf:
                continue
            value = base[i, g] + table[i, 0, _allele(g, 0), (s >> 1) & 1] + table[i, 1, _allele(g, 1), s & 1]
            if value > best + GENOTYPE_TIE or (value >= best - GENOTYPE_TIE and g == g_now):
                if value > best:
                    best = value
                arg = g
        new_genotype[i] = arg


@njit(cache=True)
def _child_block(group, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                 child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
                 log_error, log_match, log_miss, quantum, decisions, decision_count):
    """Exact update of child block ``group`` (module docstring). Returns (accepted intervals, total gain) and records
    evaluated small decisions."""
    child = group_child[group]
    sites = genotype.shape[1]
    first, stop = marker_offsets[group], marker_offsets[group + 1]
    markers = stop - first
    base = np.empty((sites, 4))
    table = np.empty((sites, 2, 2, 2))
    slot_table = np.empty((2, 2, 2))
    row = np.empty(4)
    for site in range(sites):
        for g in range(4):
            row[g] = potentials[child, site, g]
        _own_children_terms(child, site, genotype, selector, group_child, child_offsets, child_groups, child_slots,
                            log_keep, log_error, row)
        for g in range(4):
            base[site, g] = row[g]
        _slot_tables(group, site, genotype, selector, frame, potentials, orientation, roots, group_parent, group_child,
                     child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, True,
                     slot_table)
        for k in range(2):
            for a in range(2):
                for h in range(2):
                    table[site, k, a, h] = slot_table[k, a, h]
    new_selector = np.empty(sites, dtype=np.int8)
    new_genotype = np.empty(sites, dtype=np.int8)
    _block_path(group, child, 0, base, table, genotype, selector, marker_offsets, marker_sites, marker_theta,
                marker_offsets[group], marker_offsets[group + 1], -1, -1, new_selector, new_genotype)
    # Maximal differing intervals, merged when no unchanged marker separates them; each on its own exact gain.
    accepted, total = 0, 0.0
    old_genotype = genotype[child].copy()
    old_selector = selector[group].copy()
    child_row = genotype[child]
    site = 0
    while site < sites:
        if new_selector[site] == old_selector[site] and new_genotype[site] == old_genotype[site]:
            site += 1
            continue
        a = site
        b = site
        while True:
            while b < sites and (new_selector[b] != old_selector[b] or new_genotype[b] != old_genotype[b]):
                b += 1
            nxt = b
            while nxt < sites and new_selector[nxt] == old_selector[nxt] and new_genotype[nxt] == old_genotype[nxt]:
                nxt += 1
            if nxt >= sites:
                break
            lo = np.searchsorted(marker_sites[first:stop], b)
            separated = lo < markers and marker_sites[first + lo] < nxt
            if separated:
                break
            b = nxt
        before = _child_score(group, a, b, child_row, selector, marker_offsets, marker_sites, marker_theta, base,
                              table)
        for j in range(a, b):
            selector[group, j] = new_selector[j]
            child_row[j] = new_genotype[j]
        after = _child_score(group, a, b, child_row, selector, marker_offsets, marker_sites, marker_theta, base, table)
        gain = after - before
        if abs(gain) < 1.0 and decision_count[0] < decisions.shape[0]:
            decisions[decision_count[0], 0] = 0
            decisions[decision_count[0], 1] = group
            decisions[decision_count[0], 2] = a
            decisions[decision_count[0], 3] = gain
            decision_count[0] += 1
        if gain > quantum:
            accepted += 1
            total += gain
        else:
            for j in range(a, b):
                selector[group, j] = old_selector[j]
                child_row[j] = old_genotype[j]
        site = b
    return accepted, total


@njit(cache=True)
def _init_block(group, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
                log_error, log_match, log_miss, decoded, soft, soft_index):
    """Initialisation of child block ``group`` (one top-down sum-product pass, module docstring): its optimum given
    each parent's genotype SUMMED - a root's psi, phi and its already initialised children's factors; a non-root
    parent's stored conditional (``soft``, float32 log probabilities) and its already initialised children - so that a
    parent's tied genotype orders do not act as phase evidence. Writes the block, marks it ``decoded`` and stores the
    child's own conditional (genotype summed given its new selector path) when it has children (soft_index >= 0)."""
    child = group_child[group]
    sites = genotype.shape[1]
    base = np.empty((sites, 4))
    table = np.empty((sites, 2, 2, 2))
    cavity = np.empty(4)
    for site in range(sites):
        for g in range(4):
            base[site, g] = potentials[child, site, g]
        for slot in range(2):
            parent = group_parent[group, slot]
            if parent < 0:
                for a in range(2):
                    for h in range(2):
                        table[site, slot, a, h] = LOG_HALF
                continue
            if roots[parent]:
                orient = orientation[parent, site]
                for g in range(4):
                    cavity[g] = potentials[parent, site, g] + _phi(g, frame[parent, site], orient, log_match, log_miss)
            else:
                for g in range(4):
                    cavity[g] = soft[soft_index[parent], site, g]
            for index in range(child_offsets[parent], child_offsets[parent + 1]):
                other = child_groups[index]
                if other == group or not decoded[other]:
                    continue
                other_slot = child_slots[index]
                child_allele = _allele(genotype[group_child[other], site], other_slot)
                homologue = (selector[other, site] >> (1 - other_slot)) & 1
                for g in range(4):
                    cavity[g] += log_keep if _allele(g, homologue) == child_allele else log_error
            for a in range(2):
                for h in range(2):
                    v0 = cavity[0] + (log_keep if _allele(0, h) == a else log_error)
                    v1 = cavity[1] + (log_keep if _allele(1, h) == a else log_error)
                    v2 = cavity[2] + (log_keep if _allele(2, h) == a else log_error)
                    v3 = cavity[3] + (log_keep if _allele(3, h) == a else log_error)
                    table[site, slot, a, h] = _lse4(v0, v1, v2, v3)
    new_selector = np.empty(sites, dtype=np.int8)
    new_genotype = np.empty(sites, dtype=np.int8)
    _block_path(group, child, 0, base, table, genotype, selector, marker_offsets, marker_sites, marker_theta,
                marker_offsets[group], marker_offsets[group + 1], -1, -1, new_selector, new_genotype)
    for site in range(sites):
        selector[group, site] = new_selector[site]
        genotype[child, site] = new_genotype[site]
    decoded[group] = True
    own = soft_index[child]
    if own >= 0:
        row = np.empty(4)
        for site in range(sites):
            s = new_selector[site]
            for g in range(4):
                row[g] = base[site, g] + table[site, 0, _allele(g, 0), (s >> 1) & 1] + table[site, 1, _allele(g, 1),
                                                                                          s & 1]
            norm = _lse4(row[0], row[1], row[2], row[3])
            for g in range(4):
                soft[own, site, g] = row[g] - norm


@njit(cache=True)
def _root_score(run_lo, run_hi, frame_row, starts, run_theta, run_value):
    """Root block terms of runs [run_lo, run_hi): the sum of R (run_value[run, f]) plus the frame transitions into
    run_lo and out of run_hi - 1. The one routine for the block's before and after scores."""
    runs = len(starts) - 1
    total = 0.0
    for run in range(run_lo, run_hi):
        total += run_value[run, frame_row[starts[run]]]
    for run in range(max(run_lo, 1), min(run_hi + 1, runs)):
        switched = frame_row[starts[run]] != frame_row[starts[run] - 1]
        total += _log_theta(run_theta[run - 1]) if switched else _log_stay(run_theta[run - 1])
    return total


@njit(cache=True)
def _root_block(root, potentials, genotype, selector, frame, orientation, starts, run_theta, anchor, group_child,
                child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, quantum, decisions,
                decision_count):
    """Exact update of root block ``root`` (Viterbi over its frame-chain runs); intervals of runs accepted on their own
    exact gains. Returns (accepted intervals, total gain)."""
    runs = len(starts) - 1
    run_value = np.zeros((runs, 2))
    anchor_run = -1
    for run in range(runs):
        if starts[run] <= anchor < starts[run + 1]:
            anchor_run = run
        for site in range(starts[run], starts[run + 1]):
            for f in range(2):
                run_value[run, f] += _root_term(root, site, f, genotype, selector, potentials, orientation,
                                                group_child, child_offsets, child_groups, child_slots, log_keep,
                                                log_error, log_match, log_miss)
    score = np.empty(2)
    new = np.empty(2)
    back = np.zeros((max(runs, 1), 2), dtype=np.int8)
    for f in range(2):
        score[f] = -np.inf if (anchor_run == 0 and f == 1) else run_value[0, f]
    for run in range(1, runs):
        theta = run_theta[run - 1]
        for f in range(2):
            stay, jump = score[f] + _log_stay(theta), score[1 - f] + _log_theta(theta)
            top = max(stay, jump)
            back[run, f] = f if stay >= top - chains.CHAIN_TIE else 1 - f
            new[f] = top + run_value[run, f]
            if run == anchor_run and f == 1:
                new[f] = -np.inf
        score[0], score[1] = new[0], new[1]
    top = max(score[0], score[1])
    last_now = frame[root, starts[runs] - 1]
    f = last_now if score[last_now] >= top - chains.CHAIN_TIE else (0 if score[0] >= score[1] else 1)
    path = np.empty(runs, dtype=np.int8)
    for run in range(runs - 1, -1, -1):
        path[run] = f
        if run > 0:
            f = back[run, f]
    accepted, total = 0, 0.0
    run = 0
    while run < runs:
        if path[run] == frame[root, starts[run]]:
            run += 1
            continue
        lo = run
        while run < runs and path[run] != frame[root, starts[run]]:
            run += 1
        hi = run
        before = _root_score(lo, hi, frame[root], starts, run_theta, run_value)
        saved = frame[root, starts[lo]:starts[hi]].copy()
        for r in range(lo, hi):
            for site in range(starts[r], starts[r + 1]):
                frame[root, site] = path[r]
        after = _root_score(lo, hi, frame[root], starts, run_theta, run_value)
        gain = after - before
        if abs(gain) < 1.0 and decision_count[0] < decisions.shape[0]:
            decisions[decision_count[0], 0] = 1
            decisions[decision_count[0], 1] = root
            decisions[decision_count[0], 2] = starts[lo]
            decisions[decision_count[0], 3] = gain
            decision_count[0] += 1
        if gain > quantum:
            accepted += 1
            total += gain
        else:
            frame[root, starts[lo]:starts[hi]] = saved
    return accepted, total


@njit(cache=True, parallel=True)
def _sweep_colour(blocks, is_root, node_group, potentials, genotype, selector, frame, orientation, roots, free_starts,
                  free_trans, frozen_starts, frozen_trans, frozen, anchors, group_parent, group_child, child_offsets,
                  child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error,
                  log_match, log_miss, quantum, max_decisions):
    """Exact updates of one colour's blocks (they share no free variable)."""
    count = len(blocks)
    accepted = np.zeros(count, dtype=np.int64)
    gains = np.zeros(count)
    decisions = np.zeros((count, max_decisions, 4))
    decision_counts = np.zeros((count, 1), dtype=np.int64)
    for index in prange(count):
        sample = blocks[index]
        if is_root[index]:
            starts = frozen_starts if frozen[sample] else free_starts
            trans = frozen_trans if frozen[sample] else free_trans
            accepted[index], gains[index] = _root_block(
                sample, potentials, genotype, selector, frame, orientation, starts, trans, anchors[sample], group_child,
                child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, quantum,
                decisions[index], decision_counts[index])
        else:
            accepted[index], gains[index] = _child_block(
                node_group[sample], potentials, genotype, selector, frame, orientation, roots, group_parent,
                group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta,
                log_keep, log_error, log_match, log_miss, quantum, decisions[index], decision_counts[index])
    return accepted, gains, decisions, decision_counts


@njit(cache=True)
def _boundary_terms(root, boundary, run, starts, run_theta, frame, selector, child_offsets, child_groups, child_slots,
                    marker_offsets, marker_sites, marker_theta, toggled):
    """The transition terms a relabelling of ``root`` at ``boundary`` (between runs run - 1 and run) changes: r's frame
    transition there and each child's composed selector transition across it, as they are (toggled False) or after
    the relabel (toggled True). The one routine for a move's before and after scores."""
    theta = run_theta[run - 1]
    switched = frame[root, starts[run]] != frame[root, starts[run] - 1]
    if toggled:
        switched = not switched
    total = _log_theta(theta) if switched else _log_stay(theta)
    for index in range(child_offsets[root], child_offsets[root + 1]):
        group = child_groups[index]
        slot = child_slots[index]
        first, stop = marker_offsets[group], marker_offsets[group + 1]
        k = np.searchsorted(marker_sites[first:stop], boundary, side="right")
        if k == 0 or k == stop - first:
            continue
        left = selector[group, marker_sites[first + k - 1]]
        right = selector[group, marker_sites[first + k]]
        if toggled:
            right ^= 1 << (1 - slot)
        total += _transition(left, right, marker_theta[first + k - 1])
    return total


@njit(cache=True)
def _relabel_scan(root, frame, selector, starts, run_theta, anchor, child_offsets, child_groups, child_slots,
                  marker_offsets, marker_sites, marker_theta, quantum, decisions, decision_count):
    """Every relabelling move of ``root`` (module docstring), boundaries in position order, each evaluated on the
    current configuration and applied at once if its exact gain exceeds ``quantum``. Returns (moves, total gain)."""
    runs = len(starts) - 1
    sites = frame.shape[1]
    accepted, total = 0, 0.0
    for run in range(1, runs):
        boundary = starts[run] - 1
        before = _boundary_terms(root, boundary, run, starts, run_theta, frame, selector, child_offsets, child_groups,
                                 child_slots, marker_offsets, marker_sites, marker_theta, False)
        after = _boundary_terms(root, boundary, run, starts, run_theta, frame, selector, child_offsets, child_groups,
                                child_slots, marker_offsets, marker_sites, marker_theta, True)
        gain = after - before
        if abs(gain) < 1.0 and decision_count[0] < decisions.shape[0]:
            decisions[decision_count[0], 0] = 2
            decisions[decision_count[0], 1] = root
            decisions[decision_count[0], 2] = boundary
            decisions[decision_count[0], 3] = gain
            decision_count[0] += 1
        if gain <= quantum:
            continue
        accepted += 1
        total += gain
        lo, hi = (0, boundary + 1) if (anchor >= 0 and boundary < anchor) else (boundary + 1, sites)
        for site in range(lo, hi):
            frame[root, site] ^= 1
            for index in range(child_offsets[root], child_offsets[root + 1]):
                selector[child_groups[index], site] ^= 1 << (1 - child_slots[index])
    return accepted, total


@njit(cache=True, parallel=True)
def objective(potentials, genotype, selector, frame, orientation, roots, frozen, free_starts, free_trans,
              frozen_starts, frozen_trans, group_parent, group_child, child_offsets, child_groups, child_slots,
              marker_offsets, marker_sites, marker_theta, log_keep, log_error, log_match, log_miss):
    """J of a configuration from scratch (module docstring), summed per sample and group in a fixed order."""
    samples, sites = genotype.shape
    per_sample = np.zeros(samples)
    for sample in prange(samples):
        total = 0.0
        if roots[sample]:
            starts = frozen_starts if frozen[sample] else free_starts
            trans = frozen_trans if frozen[sample] else free_trans
            total += LOG_HALF
            for site in range(sites):
                total += _root_term(sample, site, frame[sample, site], genotype, selector, potentials, orientation,
                                    group_child, child_offsets, child_groups, child_slots, log_keep, log_error,
                                    log_match, log_miss)
            for run in range(1, len(starts) - 1):
                switched = frame[sample, starts[run]] != frame[sample, starts[run] - 1]
                total += _log_theta(trans[run - 1]) if switched else _log_stay(trans[run - 1])
        else:
            for site in range(sites):
                total += potentials[sample, site, genotype[sample, site]]
        per_sample[sample] = total
    groups = len(group_child)
    per_group = np.zeros(groups)
    for group in prange(groups):
        child = group_child[group]
        total = LOG_QUARTER
        for slot in range(2):
            parent = group_parent[group, slot]
            if parent >= 0 and roots[parent]:
                continue                      # inside the parent's R
            for site in range(sites):
                if parent < 0:
                    total += LOG_HALF
                else:
                    homologue = (selector[group, site] >> (1 - slot)) & 1
                    total += (log_keep if _allele(genotype[parent, site], homologue)
                              == _allele(genotype[child, site], slot) else log_error)
        first, stop = marker_offsets[group], marker_offsets[group + 1]
        for k in range(first + 1, stop):
            total += _transition(selector[group, marker_sites[k - 1]], selector[group, marker_sites[k]],
                                 marker_theta[k - 1])
        per_group[group] = total
    return float(np.sum(per_sample) + np.sum(per_group))


# ------------------------------------------------------------------------------------------- compound moves

@njit(cache=True)
def _block_terms(group, lo, end, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                 child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, base, table):
    """A child block's terms on sites [lo, end): base [n, 4] (psi + its own children's copy terms) and slot tables
    [n, 2, 2, 2] (a root parent's R with the child's factor in, its other children in)."""
    child = group_child[group]
    slot_table = np.empty((2, 2, 2))
    row = np.empty(4)
    for i in range(end - lo):
        site = lo + i
        for g in range(4):
            row[g] = potentials[child, site, g]
        _own_children_terms(child, site, genotype, selector, group_child, child_offsets, child_groups, child_slots,
                            log_keep, log_error, row)
        for g in range(4):
            base[i, g] = row[g]
        _slot_tables(group, site, genotype, selector, frame, potentials, orientation, roots, group_parent, group_child,
                     child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, True,
                     slot_table)
        for k in range(2):
            for a in range(2):
                for h in range(2):
                    table[i, k, a, h] = slot_table[k, a, h]


@njit(cache=True)
def _window_end(group, hi, marker_offsets, marker_sites, sites):
    """End (exclusive) of a windowed re-decode of ``group`` inside [lo, hi): its first marker at or after hi (the sites
    carried from its last free marker reach it), or the chromosome end."""
    first, stop = marker_offsets[group], marker_offsets[group + 1]
    b_idx = first + np.searchsorted(marker_sites[first:stop], hi)
    return marker_sites[b_idx] if b_idx < stop else sites


@njit(cache=True)
def _window_child(group, lo, hi, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                  child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
                  log_error, log_match, log_miss):
    """Exact re-decode of child block ``group`` inside [lo, hi): its markers there free, the markers outside pinned;
    changes sites [lo, _window_end) only. Applied as found: the compound move accepts or reverts the whole."""
    sites = genotype.shape[1]
    child = group_child[group]
    first, stop = marker_offsets[group], marker_offsets[group + 1]
    a_idx = first + np.searchsorted(marker_sites[first:stop], lo)
    b_idx = first + np.searchsorted(marker_sites[first:stop], hi)
    end = marker_sites[b_idx] if b_idx < stop else sites
    n = end - lo
    if n <= 0:
        return
    base = np.empty((n, 4))
    table = np.empty((n, 2, 2, 2))
    _block_terms(group, lo, end, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                 child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, base, table)
    left = selector[group, marker_sites[a_idx - 1]] if a_idx > first else -1
    right = selector[group, marker_sites[b_idx]] if b_idx < stop else -1
    new_selector = np.empty(n, dtype=np.int8)
    new_genotype = np.empty(n, dtype=np.int8)
    _block_path(group, child, lo, base, table, genotype, selector, marker_offsets, marker_sites, marker_theta, a_idx,
                b_idx, left, right, new_selector, new_genotype)
    for i in range(n):
        selector[group, lo + i] = new_selector[i]
        genotype[child, lo + i] = new_genotype[i]


@njit(cache=True)
def _refit_genotypes(group, lo, end, potentials, genotype, selector, frame, orientation, roots, group_parent,
                     group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss):
    """Child ``group``'s genotypes on [lo, end) set to their argmax given its selector (ties keep the current one)."""
    child = group_child[group]
    n = end - lo
    if n <= 0:
        return
    base = np.empty((n, 4))
    table = np.empty((n, 2, 2, 2))
    _block_terms(group, lo, end, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                 child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, base, table)
    for i in range(n):
        s = selector[group, lo + i]
        best, arg = -np.inf, -1
        g_now = genotype[child, lo + i]
        for g in range(4):
            if base[i, g] == -np.inf:
                continue
            value = base[i, g] + table[i, 0, _allele(g, 0), (s >> 1) & 1] + table[i, 1, _allele(g, 1), s & 1]
            if value > best + GENOTYPE_TIE or (value >= best - GENOTYPE_TIE and g == g_now):
                if value > best:
                    best = value
                arg = g
        genotype[child, lo + i] = arg


@njit(cache=True)
def _window_root(root, lo, hi, potentials, genotype, selector, frame, orientation, starts, run_theta, anchor,
                 group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss):
    """Exact re-decode of root ``root``'s frame on its runs inside [lo, hi), the runs outside pinned (the anchor run
    at 0). Applied as found."""
    runs = len(starts) - 1
    r0 = np.searchsorted(starts[:runs], lo)
    r1 = r0
    while r1 < runs and starts[r1 + 1] <= hi:
        r1 += 1
    m = r1 - r0
    if m <= 0:
        return
    run_value = np.zeros((m, 2))
    anchor_run = -1
    for j in range(m):
        if starts[r0 + j] <= anchor < starts[r0 + j + 1]:
            anchor_run = j
        for site in range(starts[r0 + j], starts[r0 + j + 1]):
            for f in range(2):
                run_value[j, f] += _root_term(root, site, f, genotype, selector, potentials, orientation, group_child,
                                              child_offsets, child_groups, child_slots, log_keep, log_error,
                                              log_match, log_miss)
    left = frame[root, starts[r0] - 1] if r0 > 0 else -1
    right = frame[root, starts[r1]] if r1 < runs else -1
    score = np.empty(2)
    new = np.empty(2)
    back = np.zeros((m, 2), dtype=np.int8)
    for f in range(2):
        score[f] = run_value[0, f]
        if left >= 0:
            theta = run_theta[r0 - 1]
            score[f] += _log_stay(theta) if f == left else _log_theta(theta)
        if anchor_run == 0 and f == 1:
            score[f] = -np.inf
    for j in range(1, m):
        theta = run_theta[r0 + j - 1]
        for f in range(2):
            stay, jump = score[f] + _log_stay(theta), score[1 - f] + _log_theta(theta)
            top = max(stay, jump)
            back[j, f] = f if stay >= top - chains.CHAIN_TIE else 1 - f
            new[f] = top + run_value[j, f]
            if j == anchor_run and f == 1:
                new[f] = -np.inf
        score[0], score[1] = new[0], new[1]
    if right >= 0:
        theta = run_theta[r1 - 1]
        for f in range(2):
            score[f] += _log_stay(theta) if f == right else _log_theta(theta)
    top = max(score[0], score[1])
    last_now = frame[root, starts[r1] - 1]
    f = last_now if score[last_now] >= top - chains.CHAIN_TIE else (0 if score[0] >= score[1] else 1)
    for j in range(m - 1, -1, -1):
        for site in range(starts[r0 + j], starts[r0 + j + 1]):
            frame[root, site] = f
        if j > 0:
            f = back[j, f]


@njit(cache=True)
def _term_masks(groups, count, extra_root, roots, group_parent, group_child, child_offsets, child_groups, samples):
    """Masks of the J terms a compound move over child groups ``groups[:count]`` (and root ``extra_root``'s frame, -1
    for none) can alter: a root's R and frame terms when its frame or any child group of it may change; a non-root's
    psi when its genotype may change; a group's copy terms when it, or the non-root parent whose genotype it copies,
    may change; a group's selector transitions when it may change."""
    root_mask = np.zeros(samples, dtype=np.bool_)
    sample_mask = np.zeros(samples, dtype=np.bool_)
    group_mask = np.zeros(len(group_child), dtype=np.bool_)
    transition_mask = np.zeros(len(group_child), dtype=np.bool_)
    if extra_root >= 0:
        root_mask[extra_root] = True
    for q in range(count):
        group = groups[q]
        group_mask[group] = True
        transition_mask[group] = True
        sample_mask[group_child[group]] = True
        for slot in range(2):
            parent = group_parent[group, slot]
            if parent >= 0 and roots[parent]:
                root_mask[parent] = True
    for q in range(count):
        child = group_child[groups[q]]
        for index in range(child_offsets[child], child_offsets[child + 1]):
            group_mask[child_groups[index]] = True
    return root_mask, sample_mask, group_mask, transition_mask


@njit(cache=True)
def _window_objective(lo, hi, root_mask, sample_mask, group_mask, transition_mask, potentials, genotype, selector,
                      frame, orientation, roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans,
                      group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites,
                      marker_theta, log_keep, log_error, log_match, log_miss):
    """The terms of J (the same terms as ``objective``) that a change confined to sites [lo, hi) and to the masked
    blocks (_term_masks) can alter: their per-site terms on [lo, hi), the masked roots' frame transitions at run starts
    in [lo, hi] and the masked groups' selector transitions with a marker in [lo, hi). The one routine for a compound
    move's before and after scores."""
    samples = genotype.shape[0]
    total = 0.0
    for sample in range(samples):
        if roots[sample]:
            if not root_mask[sample]:
                continue
            starts = frozen_starts if frozen[sample] else free_starts
            trans = frozen_trans if frozen[sample] else free_trans
            for site in range(lo, hi):
                total += _root_term(sample, site, frame[sample, site], genotype, selector, potentials, orientation,
                                    group_child, child_offsets, child_groups, child_slots, log_keep, log_error,
                                    log_match, log_miss)
            runs = len(starts) - 1
            run = max(np.searchsorted(starts[:runs], lo), 1)
            while run < runs and starts[run] <= hi:
                switched = frame[sample, starts[run]] != frame[sample, starts[run] - 1]
                total += _log_theta(trans[run - 1]) if switched else _log_stay(trans[run - 1])
                run += 1
        elif sample_mask[sample]:
            for site in range(lo, hi):
                total += potentials[sample, site, genotype[sample, site]]
    for group in range(len(group_child)):
        child = group_child[group]
        if group_mask[group]:
            for slot in range(2):
                parent = group_parent[group, slot]
                if parent >= 0 and roots[parent]:
                    continue
                for site in range(lo, hi):
                    if parent < 0:
                        total += LOG_HALF
                    else:
                        homologue = (selector[group, site] >> (1 - slot)) & 1
                        total += (log_keep if _allele(genotype[parent, site], homologue)
                                  == _allele(genotype[child, site], slot) else log_error)
        if transition_mask[group]:
            first, stop = marker_offsets[group], marker_offsets[group + 1]
            a_idx = first + np.searchsorted(marker_sites[first:stop], lo)
            b_idx = first + np.searchsorted(marker_sites[first:stop], hi)
            for k in range(max(a_idx, first + 1), min(b_idx + 1, stop)):
                total += _transition(selector[group, marker_sites[k - 1]], selector[group, marker_sites[k]],
                                     marker_theta[k - 1])
    return total


@njit(cache=True)
def _compound(kind, centre, bits, a, b, lo, hi, quantum, potentials, genotype, selector, frame, orientation, roots,
              frozen, free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent,
              group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta,
              log_keep, log_error, log_match, log_miss):
    """One compound move (review I1), accepted as a whole when its exact window gain exceeds ``quantum``:
      kind 0 (switch pair): child ``centre``'s selector bits ``bits`` flipped on sites [a, b) - removing its change
        pair at a and b - and its genotypes refitted there;
      kind 1 (track swap): non-root parent ``centre``'s two tracks swapped on [a, b) together with every child's bit
        for it (its children's copy terms are unchanged; the terms joining it to its own parents are not);
      kind 2 (sibship pair): ``bits`` names one bit of centre whose parent is a root r; every child of r sharing
        centre's change pair - its bit for r constant on [a, b) and equal to centre's there, different at sites a - 1
        and b - has it flipped on [a, b) and its genotypes refitted: the children that moved to r's other homologue
        together move back together;
    then exact re-decodes inside [lo, hi) of centre's parents (root frames, non-root blocks), their other children,
    centre's own children and centre (kind 0: neighbours first, centre last; kind 1: centre first). Returns
    (gain, accepted); a rejected move is reverted."""
    sites = genotype.shape[1]
    own = node_group[centre]
    # Involved blocks: parents, siblings (the parents' other children), children.
    parents = group_parent[own]
    count = 0
    for slot in range(2):
        parent = parents[slot]
        if parent >= 0:
            count += child_offsets[parent + 1] - child_offsets[parent]
    count += child_offsets[centre + 1] - child_offsets[centre] + 3
    groups = np.full(count, -1, dtype=np.int64)
    used = 0
    groups[used] = own
    used += 1
    for slot in range(2):
        parent = parents[slot]
        if parent >= 0 and not roots[parent]:
            groups[used] = node_group[parent]
            used += 1
    sibling_lo = used
    for slot in range(2):
        parent = parents[slot]
        if parent < 0:
            continue
        for index in range(child_offsets[parent], child_offsets[parent + 1]):
            other = child_groups[index]
            seen = False
            for q in range(used):
                if groups[q] == other:
                    seen = True
            if not seen:
                groups[used] = other
                used += 1
    sibling_hi = used
    for index in range(child_offsets[centre], child_offsets[centre + 1]):
        other = child_groups[index]
        seen = False
        for q in range(used):
            if groups[q] == other:
                seen = True
        if not seen:
            groups[used] = other
            used += 1
    children_hi = used
    end = hi
    for q in range(used):
        end = max(end, _window_end(groups[q], hi, marker_offsets, marker_sites, sites))
    n = end - lo
    saved_selector = np.empty((used, n), dtype=np.int8)
    saved_genotype = np.empty((used, n), dtype=np.int8)
    for q in range(used):
        saved_selector[q] = selector[groups[q], lo:end]
        saved_genotype[q] = genotype[group_child[groups[q]], lo:end]
    saved_frame = np.empty((2, n), dtype=np.int8)
    for slot in range(2):
        parent = parents[slot]
        if parent >= 0 and roots[parent]:
            saved_frame[slot] = frame[parent, lo:end]
    groups_list, groups_count, extra_root = groups, used, -1
    root_mask, sample_mask, group_mask, transition_mask = _term_masks(
        groups_list, groups_count, extra_root, roots, group_parent, group_child, child_offsets, child_groups,
        genotype.shape[0])
    before = _window_objective(lo, end, root_mask, sample_mask, group_mask, transition_mask, potentials, genotype,
                               selector, frame, orientation, roots, frozen, free_starts, free_trans, frozen_starts,
                               frozen_trans, group_parent, group_child, child_offsets, child_groups, child_slots,
                               marker_offsets, marker_sites, marker_theta, log_keep, log_error, log_match, log_miss)
    # Seed.
    if kind == 0:
        for site in range(a, b):
            selector[own, site] ^= bits
        _refit_genotypes(own, a, b, potentials, genotype, selector, frame, orientation, roots, group_parent,
                         group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                         log_miss)
    elif kind == 1:
        for site in range(a, b):
            g = genotype[centre, site]
            genotype[centre, site] = ((g & 1) << 1) | (g >> 1)
            for index in range(child_offsets[centre], child_offsets[centre + 1]):
                selector[child_groups[index], site] ^= 1 << (1 - child_slots[index])
    else:
        root = parents[0] if bits == 2 else parents[1]
        value = (selector[own, a] & bits) != 0
        for index in range(child_offsets[root], child_offsets[root + 1]):
            other = child_groups[index]
            mask = 1 << (1 - child_slots[index])
            same = (a == 0 or ((selector[other, a - 1] & mask) != 0) != value) and (
                b == sites or ((selector[other, b] & mask) != 0) != value)
            if same:
                for site in range(a, b):
                    if ((selector[other, site] & mask) != 0) != value:
                        same = False
                        break
            if not same:
                continue
            for site in range(a, b):
                selector[other, site] ^= mask
            _refit_genotypes(other, a, b, potentials, genotype, selector, frame, orientation, roots, group_parent,
                             group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                             log_miss)
    # Re-decodes.
    for rnd in range(2):
        if kind == 1 or rnd == 1:
            _window_child(own, lo, hi, potentials, genotype, selector, frame, orientation, roots, group_parent,
                          group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites,
                          marker_theta, log_keep, log_error, log_match, log_miss)
        for slot in range(2):
            parent = parents[slot]
            if parent < 0:
                continue
            if roots[parent]:
                starts = frozen_starts if frozen[parent] else free_starts
                trans = frozen_trans if frozen[parent] else free_trans
                _window_root(parent, lo, hi, potentials, genotype, selector, frame, orientation, starts, trans,
                             anchors[parent], group_child, child_offsets, child_groups, child_slots, log_keep,
                             log_error, log_match, log_miss)
            else:
                _window_child(node_group[parent], lo, hi, potentials, genotype, selector, frame, orientation, roots,
                              group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets,
                              marker_sites, marker_theta, log_keep, log_error, log_match, log_miss)
        for q in range(sibling_lo, children_hi):
            if groups[q] == own:
                continue
            _window_child(groups[q], lo, hi, potentials, genotype, selector, frame, orientation, roots, group_parent,
                          group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites,
                          marker_theta, log_keep, log_error, log_match, log_miss)
    _window_child(own, lo, hi, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                  child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
                  log_error, log_match, log_miss)
    after = _window_objective(lo, end, root_mask, sample_mask, group_mask, transition_mask, potentials, genotype,
                              selector, frame, orientation, roots, frozen, free_starts, free_trans, frozen_starts,
                              frozen_trans, group_parent, group_child, child_offsets, child_groups, child_slots,
                              marker_offsets, marker_sites, marker_theta, log_keep, log_error, log_match, log_miss)
    gain = after - before
    if gain > quantum:
        return gain, True
    for q in range(used):
        selector[groups[q], lo:end] = saved_selector[q]
        genotype[group_child[groups[q]], lo:end] = saved_genotype[q]
    for slot in range(2):
        parent = parents[slot]
        if parent >= 0 and roots[parent]:
            frame[parent, lo:end] = saved_frame[slot]
    return gain, False


@njit(cache=True)
def _scan_pairs(kind, centre, source, mask, quantum, max_span, margin, potentials, genotype, selector, frame,
                orientation, roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group,
                group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites,
                marker_theta, log_keep, log_error, log_match, log_miss, decisions, decision_count):
    """Compound moves of ``kind`` around ``centre`` on the change pairs of group ``source``'s selector bits ``mask``
    (consecutive changes at most ``max_span`` sites apart, in position order; the state is re-read after each move).
    Returns (accepted moves, total gain)."""
    sites = genotype.shape[1]
    accepted, total = 0, 0.0
    first, stop = marker_offsets[source], marker_offsets[source + 1]
    k = first + 1
    while k < stop:
        if (selector[source, marker_sites[k]] & mask) == (selector[source, marker_sites[k - 1]] & mask):
            k += 1
            continue
        j = k + 1
        while j < stop and (selector[source, marker_sites[j]] & mask) == (selector[source, marker_sites[j - 1]] & mask):
            j += 1
        if j >= stop:
            break
        a, b = marker_sites[k], marker_sites[j]
        bits = (selector[source, a] ^ selector[source, marker_sites[k - 1]]) & mask
        usable = b - a <= max_span
        if kind == 2:
            parent = group_parent[source, 0] if bits == 2 else (group_parent[source, 1] if bits == 1 else -1)
            usable = usable and parent >= 0 and roots[parent]
        if usable:
            width = max(margin, min(b - a, COMPOUND_WIDTH))
            lo, hi = max(0, a - width), min(sites, b + width)
            gain, ok = _compound(kind, centre, bits, a, b, lo, hi, quantum, potentials, genotype, selector, frame,
                                 orientation, roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans,
                                 anchors, node_group, group_parent, group_child, child_offsets, child_groups,
                                 child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error,
                                 log_match, log_miss)
            if abs(gain) < 1.0 and decision_count[0] < decisions.shape[0]:
                decisions[decision_count[0], 0] = 3 + kind
                decisions[decision_count[0], 1] = source
                decisions[decision_count[0], 2] = a
                decisions[decision_count[0], 3] = gain
                decision_count[0] += 1
            if ok:
                accepted += 1
                total += gain
        k = j
    return accepted, total


@njit(cache=True)
def _compound_sweep(quantum, max_span, margin, potentials, genotype, selector, frame, orientation, roots, frozen,
                    free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent,
                    group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta,
                    log_keep, log_error, log_match, log_miss, decisions, decision_count):
    """Every compound move once, sequentially in canonical order (sample order, then position): kind 0 on every
    child's selector change pairs; kind 1 for every non-root parent on its own change pairs and on each child's change
    pairs of the bit for it; kind 2 on every child's change pairs of one bit whose parent is a root. Window = the
    pair's interval widened by max(``margin``, min(its span, COMPOUND_WIDTH)) on each side. Returns (accepted moves,
    total gain)."""
    accepted, total = 0, 0.0
    groups = len(group_child)
    for kind in range(3):
        for group in range(groups):
            centre = group_child[group]
            if kind == 1:
                if child_offsets[centre] == child_offsets[centre + 1]:
                    continue
                sources = np.empty(1 + child_offsets[centre + 1] - child_offsets[centre], dtype=np.int64)
                masks = np.empty_like(sources)
                sources[0], masks[0] = group, 3
                for q in range(child_offsets[centre], child_offsets[centre + 1]):
                    sources[1 + q - child_offsets[centre]] = child_groups[q]
                    masks[1 + q - child_offsets[centre]] = 1 << (1 - child_slots[q])
            else:
                sources = np.full(1, group, dtype=np.int64)
                masks = np.full(1, 3, dtype=np.int64)
            for q in range(len(sources)):
                moves, gain = _scan_pairs(kind, centre, sources[q], masks[q], quantum, max_span, margin, potentials,
                                          genotype, selector, frame, orientation, roots, frozen, free_starts,
                                          free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent,
                                          group_child, child_offsets, child_groups, child_slots, marker_offsets,
                                          marker_sites, marker_theta, log_keep, log_error, log_match, log_miss,
                                          decisions, decision_count)
                accepted += moves
                total += gain
    return accepted, total


@njit(cache=True)
def _relabel_flip(root, boundary, anchor, frame, selector, child_offsets, child_groups, child_slots):
    """Apply (or undo: it is an involution) the relabelling of ``root`` at ``boundary`` (module docstring)."""
    sites = frame.shape[1]
    lo, hi = (0, boundary + 1) if (anchor >= 0 and boundary < anchor) else (boundary + 1, sites)
    for site in range(lo, hi):
        frame[root, site] ^= 1
        for index in range(child_offsets[root], child_offsets[root + 1]):
            selector[child_groups[index], site] ^= 1 << (1 - child_slots[index])


@njit(cache=True)
def _relabel_compound(root, run, lo, hi, quantum, potentials, genotype, selector, frame, orientation, roots, frozen,
                      free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent,
                      group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites,
                      marker_theta, log_keep, log_error, log_match, log_miss):
    """Compound relabelling (review I1): relabel ``root`` at the boundary into frame run ``run``, then exact windowed
    re-decodes inside [lo, hi) of its children, its frame and its children again (a child's switch that sat elsewhere
    in the stretch where r's bit is uninformative for it moves to the boundary or vanishes). Exact gain = the
    relabelling's boundary terms + the window change; accepted as a whole above ``quantum``, else reverted. Returns
    (gain, accepted)."""
    sites = genotype.shape[1]
    starts = frozen_starts if frozen[root] else free_starts
    trans = frozen_trans if frozen[root] else free_trans
    boundary = starts[run] - 1
    anchor = anchors[root]
    before_b = _boundary_terms(root, boundary, run, starts, trans, frame, selector, child_offsets, child_groups,
                               child_slots, marker_offsets, marker_sites, marker_theta, False)
    after_b = _boundary_terms(root, boundary, run, starts, trans, frame, selector, child_offsets, child_groups,
                              child_slots, marker_offsets, marker_sites, marker_theta, True)
    _relabel_flip(root, boundary, anchor, frame, selector, child_offsets, child_groups, child_slots)
    first_child, last_child = child_offsets[root], child_offsets[root + 1]
    count = last_child - first_child
    end = hi
    for index in range(first_child, last_child):
        end = max(end, _window_end(child_groups[index], hi, marker_offsets, marker_sites, sites))
    n = end - lo
    saved_selector = np.empty((count, n), dtype=np.int8)
    saved_genotype = np.empty((count, n), dtype=np.int8)
    for q in range(count):
        group = child_groups[first_child + q]
        saved_selector[q] = selector[group, lo:end]
        saved_genotype[q] = genotype[group_child[group], lo:end]
    saved_frame = frame[root, lo:end].copy()
    groups_list = child_groups[first_child:last_child].copy()
    groups_count, extra_root = count, root
    root_mask, sample_mask, group_mask, transition_mask = _term_masks(
        groups_list, groups_count, extra_root, roots, group_parent, group_child, child_offsets, child_groups,
        genotype.shape[0])
    before = _window_objective(lo, end, root_mask, sample_mask, group_mask, transition_mask, potentials, genotype,
                               selector, frame, orientation, roots, frozen, free_starts, free_trans, frozen_starts,
                               frozen_trans, group_parent, group_child, child_offsets, child_groups, child_slots,
                               marker_offsets, marker_sites, marker_theta, log_keep, log_error, log_match, log_miss)
    for rnd in range(2):
        for index in range(first_child, last_child):
            _window_child(child_groups[index], lo, hi, potentials, genotype, selector, frame, orientation, roots,
                          group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets,
                          marker_sites, marker_theta, log_keep, log_error, log_match, log_miss)
        if rnd == 0:
            _window_root(root, lo, hi, potentials, genotype, selector, frame, orientation, starts, trans, anchor,
                         group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                         log_miss)
    after = _window_objective(lo, end, root_mask, sample_mask, group_mask, transition_mask, potentials, genotype,
                              selector, frame, orientation, roots, frozen, free_starts, free_trans, frozen_starts,
                              frozen_trans, group_parent, group_child, child_offsets, child_groups, child_slots,
                              marker_offsets, marker_sites, marker_theta, log_keep, log_error, log_match, log_miss)
    gain = (after_b - before_b) + (after - before)
    if gain > quantum:
        return gain, True
    for q in range(count):
        group = child_groups[first_child + q]
        selector[group, lo:end] = saved_selector[q]
        genotype[group_child[group], lo:end] = saved_genotype[q]
    frame[root, lo:end] = saved_frame
    _relabel_flip(root, boundary, anchor, frame, selector, child_offsets, child_groups, child_slots)
    return gain, False


@njit(cache=True)
def _relabel_compound_sweep(quantum, max_span, margin, potentials, genotype, selector, frame, orientation, roots,
                            frozen, free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group,
                            group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets,
                            marker_sites, marker_theta, log_keep, log_error, log_match, log_miss, decisions,
                            decision_count):
    """Every root (sample order) and every boundary of its frame chain nearest to a change of a child's bit for it
    (within ``max_span`` sites; boundaries in position order, each once): a compound relabelling with the window
    spanning the boundary and the triggering changes, widened by ``margin``. Returns (accepted moves, total gain)."""
    samples, sites = genotype.shape
    accepted, total = 0, 0.0
    for root in range(samples):
        if not roots[root] or child_offsets[root] == child_offsets[root + 1]:
            continue
        starts = frozen_starts if frozen[root] else free_starts
        runs = len(starts) - 1
        if runs < 2:
            continue
        reach_lo = np.full(runs, sites, dtype=np.int64)       # per boundary run: span of triggering changes
        reach_hi = np.full(runs, -1, dtype=np.int64)
        for index in range(child_offsets[root], child_offsets[root + 1]):
            group = child_groups[index]
            mask = 1 << (1 - child_slots[index])
            first, stop = marker_offsets[group], marker_offsets[group + 1]
            for k in range(first + 1, stop):
                m = marker_sites[k]
                if (selector[group, m] & mask) == (selector[group, marker_sites[k - 1]] & mask):
                    continue
                r = np.searchsorted(starts[:runs], m)            # first run starting at or after m
                best, best_d = -1, sites
                for cand in (r - 1, r):
                    if 1 <= cand < runs:
                        d = abs(starts[cand] - m)
                        if d < best_d:
                            best, best_d = cand, d
                if best < 0 or best_d > max_span:
                    continue
                reach_lo[best] = min(reach_lo[best], min(m, starts[best]))
                reach_hi[best] = max(reach_hi[best], max(m, starts[best]))
        for run in range(1, runs):
            if reach_hi[run] < 0:
                continue
            lo = max(0, reach_lo[run] - margin)
            hi = min(sites, reach_hi[run] + margin + 1)
            gain, ok = _relabel_compound(root, run, lo, hi, quantum, potentials, genotype, selector, frame, orientation,
                                         roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans, anchors,
                                         node_group, group_parent, group_child, child_offsets, child_groups,
                                         child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error,
                                         log_match, log_miss)
            if abs(gain) < 1.0 and decision_count[0] < decisions.shape[0]:
                decisions[decision_count[0], 0] = 6
                decisions[decision_count[0], 1] = root
                decisions[decision_count[0], 2] = starts[run] - 1
                decisions[decision_count[0], 3] = gain
                decision_count[0] += 1
            if ok:
                accepted += 1
                total += gain
    return accepted, total


@njit(cache=True)
def _site_move(parent, site, quantum, max_combos, potentials, genotype, selector, frame, orientation, roots, frozen,
               free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent, group_child,
               child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
               log_error, log_match, log_miss):
    """Site move (sibship genotypes): at one site, the exact joint maximum over the ordered genotypes of
    ``parent``'s children (and of the parent itself when it is not a root), selectors fixed - e.g. full siblings
    swapping which parent transmitted which allele together, which no single child block can do. Candidates are the
    genotypes of finite potential; skipped above ``max_combos`` joint configurations. The site's J terms are scored
    by _window_objective on [site, site + 1); the current configuration is kept unless the best is better by more
    than GENOTYPE_TIE, and accepted only above ``quantum``. Returns (gain, accepted)."""
    samples = genotype.shape[0]
    first, last = child_offsets[parent], child_offsets[parent + 1]
    count = last - first + (0 if roots[parent] else 1)
    members = np.empty(count, dtype=np.int64)
    groups = np.empty(last - first + (0 if roots[parent] else 1), dtype=np.int64)
    used = 0
    for index in range(first, last):
        members[used] = group_child[child_groups[index]]
        groups[used] = child_groups[index]
        used += 1
    if not roots[parent]:
        members[used] = parent
        groups[used] = node_group[parent]
        used += 1
    options = np.zeros((used, 4), dtype=np.int8)
    sizes = np.zeros(used, dtype=np.int64)
    combos = 1
    for q in range(used):
        for g in range(4):
            if potentials[members[q], site, g] > -np.inf:
                options[q, sizes[q]] = g
                sizes[q] += 1
        combos *= max(sizes[q], 1)
        if combos > max_combos:
            return 0.0, False
    if combos <= 1:
        return 0.0, False
    root_mask, sample_mask, group_mask, transition_mask = _term_masks(
        groups, used, parent if roots[parent] else -1, roots, group_parent, group_child, child_offsets, child_groups,
        samples)
    transition_mask[:] = False                      # selectors do not change
    current = np.empty(used, dtype=np.int8)
    for q in range(used):
        current[q] = genotype[members[q], site]
    before = _window_objective(site, site + 1, root_mask, sample_mask, group_mask, transition_mask, potentials,
                               genotype, selector, frame, orientation, roots, frozen, free_starts, free_trans,
                               frozen_starts, frozen_trans, group_parent, group_child, child_offsets, child_groups,
                               child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error, log_match,
                               log_miss)
    best_value = before
    best = current.copy()
    digits = np.zeros(used, dtype=np.int64)
    for _ in range(combos):
        for q in range(used):
            genotype[members[q], site] = options[q, digits[q]]
        value = _window_objective(site, site + 1, root_mask, sample_mask, group_mask, transition_mask, potentials,
                                  genotype, selector, frame, orientation, roots, frozen, free_starts, free_trans,
                                  frozen_starts, frozen_trans, group_parent, group_child, child_offsets, child_groups,
                                  child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error,
                                  log_match, log_miss)
        if value > best_value + GENOTYPE_TIE:
            best_value = value
            for q in range(used):
                best[q] = genotype[members[q], site]
        q = 0
        while q < used:
            digits[q] += 1
            if digits[q] < sizes[q]:
                break
            digits[q] = 0
            q += 1
    gain = best_value - before
    keep = best if gain > quantum else current
    for q in range(used):
        genotype[members[q], site] = keep[q]
    return gain, gain > quantum


@njit(cache=True)
def _site_sweep(quantum, max_combos, potentials, genotype, selector, frame, orientation, roots, frozen, free_starts,
                free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent, group_child, child_offsets,
                child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error, log_match,
                log_miss, decisions, decision_count):
    """Site moves at every Mendelian-conflict site (_conflicts at the sweep's start) of every root with at least two
    children and every non-root parent, sequentially (parents in sample order, then sites). Returns (accepted moves,
    total gain)."""
    conflict = _conflicts(genotype, selector, frame, potentials, orientation, roots, group_child, child_offsets,
                          child_groups, child_slots, log_keep, log_error, log_match, log_miss)
    samples, sites = genotype.shape
    accepted, total = 0, 0.0
    for parent in range(samples):
        kids = child_offsets[parent + 1] - child_offsets[parent]
        if kids == 0 or (kids < 2 and roots[parent]):
            continue
        for site in range(sites):
            if not conflict[parent, site]:
                continue
            gain, ok = _site_move(parent, site, quantum, max_combos, potentials, genotype, selector, frame,
                                  orientation, roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans,
                                  anchors, node_group, group_parent, group_child, child_offsets, child_groups,
                                  child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error,
                                  log_match, log_miss)
            if abs(gain) < 1.0 and decision_count[0] < decisions.shape[0]:
                decisions[decision_count[0], 0] = 7
                decisions[decision_count[0], 1] = parent
                decisions[decision_count[0], 2] = site
                decisions[decision_count[0], 3] = gain
                decision_count[0] += 1
            if ok:
                accepted += 1
                total += gain
    return accepted, total


def colouring(graph):
    """Colours of the conflict graph: parent-child edges plus a clique of each root's children (they share the root's
    summed-out genotype). Greedy over samples by (-degree, sample index); a list of ascending sample arrays."""
    samples = len(graph.roots)
    neighbours = [set() for _ in range(samples)]
    for parent, child in zip(graph.parents, graph.children):
        neighbours[int(parent)].add(int(child))
        neighbours[int(child)].add(int(parent))
    for root in np.flatnonzero(graph.roots):
        kids = sorted({int(c) for p, c in zip(graph.parents, graph.children) if p == root})
        for a in kids:
            neighbours[a].update(k for k in kids if k != a)
    order = sorted(range(samples), key=lambda n: (-len(neighbours[n]), n))
    colour = np.full(samples, -1, dtype=np.int64)
    for sample in order:
        used = {int(colour[m]) for m in neighbours[sample] if colour[m] >= 0}
        c = 0
        while c in used:
            c += 1
        colour[sample] = c
    return [np.flatnonzero(colour == c).astype(np.int64) for c in range(int(colour.max()) + 1 if samples else 0)]


def descent_order(graph):
    """Child groups in generation order (a child after all its observed parents), then sample index."""
    samples = len(graph.roots)
    depth = np.zeros(samples, dtype=np.int64)
    parents_of = [[] for _ in range(samples)]
    for parent, child in zip(graph.parents, graph.children):
        parents_of[int(child)].append(int(parent))
    for _ in range(samples):
        changed = False
        for child in range(samples):
            for parent in parents_of[child]:
                if depth[child] < depth[parent] + 1:
                    depth[child] = depth[parent] + 1
                    changed = True
        if not changed:
            break
    children = np.asarray(graph.group_child, dtype=np.int64)
    return np.lexsort((children, depth[children])).astype(np.int64)


def _children_of(graph, samples):
    """CSR (offsets int64 [N+1], groups int64 [E], slots int64 [E]) of each sample's child groups (canonical order:
    parent, then child group, then slot)."""
    order = np.lexsort((graph.slots, graph.edge_group, graph.parents))
    parents = graph.parents[order]
    offsets = np.searchsorted(parents, np.arange(samples + 1)).astype(np.int64)
    return offsets, graph.edge_group[order].astype(np.int64), graph.slots[order].astype(np.int64)


class FamilyMap:
    """The solver's fixed inputs for one chromosome (from a FamilyWorkspace)."""

    def __init__(self, workspace, config):
        graph = workspace.graph
        samples, sites = workspace.potentials.shape[:2]
        self.workspace, self.config, self.graph = workspace, config, graph
        self.samples, self.sites = samples, sites
        self.log_keep = math.log(1 - config.transmission_error)
        self.log_error = math.log(config.transmission_error)
        self.log_match = math.log(1 - config.scaffold_phase_error)
        self.log_miss = math.log(config.scaffold_phase_error)
        self.free_starts, self.free_trans = chains.chain_segments(workspace.phase_theta, sites)
        self.frozen_starts, self.frozen_trans = chains.chain_segments(workspace.frozen_theta, sites)
        self.child_offsets, self.child_groups, self.child_slots = _children_of(graph, samples)
        offsets, marker_sites, marker_theta = workspace.selector_support
        self.marker_offsets = np.asarray(offsets, dtype=np.int64)
        self.marker_sites = np.asarray(marker_sites, dtype=np.int64)
        self.marker_theta = np.asarray(marker_theta, dtype=np.float64)
        self.colours = colouring(graph)
        self.roots = np.asarray(graph.roots, dtype=np.bool_)
        self.descent_order = descent_order(graph)
        self.quantum = GAIN_QUANTUM
        self.phase_steps, self.phase_seconds = {}, {}

    def objective(self, state):
        w, g = self.workspace, self.graph
        return objective(w.potentials, state.genotype, state.selector, state.frame, w.orientation, self.roots,
                         w.frozen, self.free_starts, self.free_trans, self.frozen_starts, self.frozen_trans,
                         g.group_parent, g.group_child, self.child_offsets, self.child_groups, self.child_slots,
                         self.marker_offsets, self.marker_sites, self.marker_theta, self.log_keep, self.log_error,
                         self.log_match, self.log_miss)

    def initial_state(self):
        """F = 0; child blocks in generation order, each initialised by _init_block (parents' genotypes summed with
        their already initialised children)."""
        w, g = self.workspace, self.graph
        state = MapState(np.zeros((self.samples, self.sites), dtype=np.int8),
                         np.zeros((len(g.group_child), self.sites), dtype=np.int8),
                         np.zeros((self.samples, self.sites), dtype=np.int8))
        parents = np.unique(np.asarray(g.parents, dtype=np.int64))
        held = parents[~self.roots[parents]]
        soft_index = np.full(self.samples, -1, dtype=np.int64)
        soft_index[held] = np.arange(len(held))
        soft = np.zeros((max(len(held), 1), self.sites, 4), dtype=np.float32)
        decoded = np.zeros(len(g.group_child), dtype=np.bool_)
        for group in self.descent_order:
            _init_block(int(group), w.potentials, state.genotype, state.selector, state.frame, w.orientation,
                        self.roots, g.group_parent, g.group_child, self.child_offsets, self.child_groups,
                        self.child_slots, self.marker_offsets, self.marker_sites, self.marker_theta, self.log_keep,
                        self.log_error, self.log_match, self.log_miss, decoded, soft, soft_index)
        return state

    def sweep(self, state, decisions_out=None):
        """One sweep: every colour's blocks, every root's relabelling scan, the compound moves (kinds 0-2) and the
        compound relabellings, in that order. Returns (steps, total gain); per-phase accepted steps and seconds
        accumulate in self.phase_steps and self.phase_seconds."""
        w, g = self.workspace, self.graph
        steps, total = 0, 0.0

        def record(phase, moves, gain, decisions, count, started):
            nonlocal steps, total
            steps += int(moves)
            total += float(gain)
            self.phase_steps[phase] = self.phase_steps.get(phase, 0) + int(moves)
            self.phase_seconds[phase] = self.phase_seconds.get(phase, 0.0) + time.perf_counter() - started
            if decisions_out is not None:
                for row in decisions[:count]:
                    decisions_out.append((int(row[0]), int(row[1]), int(row[2]), float(row[3])))

        started = time.perf_counter()
        for blocks in self.colours:
            is_root = self.roots[blocks]
            accepted, gains, decisions, counts = _sweep_colour(
                blocks, is_root, g.node_group, w.potentials, state.genotype, state.selector, state.frame,
                w.orientation, self.roots, self.free_starts, self.free_trans, self.frozen_starts, self.frozen_trans,
                w.frozen, w.anchors, g.group_parent, g.group_child, self.child_offsets, self.child_groups,
                self.child_slots, self.marker_offsets, self.marker_sites, self.marker_theta, self.log_keep,
                self.log_error, self.log_match, self.log_miss, self.quantum, 256)
            for index in range(len(blocks)):
                record("blocks", accepted[index], gains[index], decisions[index], counts[index, 0], started)
                started = time.perf_counter()
        started = time.perf_counter()
        for root in np.flatnonzero(self.roots):
            if self.child_offsets[root] == self.child_offsets[root + 1]:
                continue
            starts, trans = ((self.frozen_starts, self.frozen_trans) if w.frozen[root]
                             else (self.free_starts, self.free_trans))
            decisions = np.zeros((256, 4))
            count = np.zeros(1, dtype=np.int64)
            moves, gain = _relabel_scan(int(root), state.frame, state.selector, starts, trans, int(w.anchors[root]),
                                        self.child_offsets, self.child_groups, self.child_slots, self.marker_offsets,
                                        self.marker_sites, self.marker_theta, self.quantum, decisions, count)
            record("relabel", moves, gain, decisions, count[0], started)
            started = time.perf_counter()
        common = (w.potentials, state.genotype, state.selector, state.frame, w.orientation, self.roots, w.frozen,
                  self.free_starts, self.free_trans, self.frozen_starts, self.frozen_trans, w.anchors, g.node_group,
                  g.group_parent, g.group_child, self.child_offsets, self.child_groups, self.child_slots,
                  self.marker_offsets, self.marker_sites, self.marker_theta, self.log_keep, self.log_error,
                  self.log_match, self.log_miss)
        started = time.perf_counter()
        decisions = np.zeros((4096, 4))
        count = np.zeros(1, dtype=np.int64)
        moves, gain = _compound_sweep(self.quantum, COMPOUND_MAX_SPAN, COMPOUND_MARGIN, *common, decisions, count)
        record("compound", moves, gain, decisions, count[0], started)
        started = time.perf_counter()
        decisions = np.zeros((4096, 4))
        count = np.zeros(1, dtype=np.int64)
        moves, gain = _relabel_compound_sweep(self.quantum, COMPOUND_WIDTH, COMPOUND_MARGIN, *common, decisions, count)
        record("compound_relabel", moves, gain, decisions, count[0], started)
        started = time.perf_counter()
        decisions = np.zeros((4096, 4))
        count = np.zeros(1, dtype=np.int64)
        moves, gain = _site_sweep(self.quantum, SITE_MAX_COMBOS, *common, decisions, count)
        record("site", moves, gain, decisions, count[0], started)
        return steps, total

    def solve(self, max_sweeps, state=None):
        """MapResult from the default start (or ``state``); RuntimeError when ``max_sweeps`` sweeps still accept a
        step, or when the from-scratch J does not rise by the accepted gains (review B2c)."""
        state = self.initial_state() if state is None else state
        values = [self.objective(state)]
        accepted, decisions = [], []
        for _ in range(int(max_sweeps)):
            steps, gain = self.sweep(state, decisions)
            accepted.append(steps)
            values.append(self.objective(state))
            tolerance = 1e-10 * abs(values[-1]) + 1e-6
            if steps and not values[-1] - values[-2] >= gain - tolerance:
                raise RuntimeError(f"family MAP: J rose by {values[-1] - values[-2]!r} but the accepted gains sum to "
                                   f"{gain!r}")
            if steps == 0:
                return MapResult(state, len(accepted), tuple(accepted), tuple(values), tuple(decisions))
        raise RuntimeError(f"family MAP did not converge in {max_sweeps} sweeps (accepted per sweep {accepted})")


# ------------------------------------------------------------------------------------------------ release

@njit(cache=True, parallel=True)
def _conditionals(potentials, genotype, selector, frame, orientation, roots, node_group, group_parent, group_child,
                  child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss):
    """Per-site log Gibbs conditionals float64 [N, L, 4] of every non-root sample's ordered genotype given the rest of
    the configuration (roots' rows: log 1/4); normalised."""
    samples, sites = genotype.shape
    out = np.empty((samples, sites, 4))
    for sample in prange(samples):
        if roots[sample]:
            for site in range(sites):
                for g in range(4):
                    out[sample, site, g] = LOG_QUARTER
            continue
        group = node_group[sample]
        table = np.empty((2, 2, 2))
        row = np.empty(4)
        for site in range(sites):
            for g in range(4):
                row[g] = potentials[sample, site, g]
            _own_children_terms(sample, site, genotype, selector, group_child, child_offsets, child_groups, child_slots,
                                log_keep, log_error, row)
            _slot_tables(group, site, genotype, selector, frame, potentials, orientation, roots, group_parent,
                         group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                         log_miss, True, table)
            s = selector[group, site]
            for g in range(4):
                row[g] += table[0, _allele(g, 0), (s >> 1) & 1] + table[1, _allele(g, 1), s & 1]
            norm = _lse4(row[0], row[1], row[2], row[3])
            for g in range(4):
                out[sample, site, g] = row[g] - norm
    return out


@njit(cache=True, inline="always")
def phase_emission(belief_row, orientation_value, phase_value, phase_error):
    """Phase-chain emission (frame matches, frame swapped) of one site from a log belief row (2.0's phase_view)."""
    orient = orientation_value
    if orient < 0:
        return .5, .5
    p = phase_error + (1 - 2 * phase_error) * phase_value
    f1 = p if orient == 0 else 1 - p
    f2 = 1 - p if orient == 0 else p
    q0 = math.exp(float(belief_row[0])) / .5
    q1 = math.exp(float(belief_row[1])) / f1
    q2 = math.exp(float(belief_row[2])) / f2
    q3 = math.exp(float(belief_row[3])) / .5
    total = ((q0 + q1) + q2) + q3
    q0 /= total
    q1 /= total
    q2 /= total
    q3 /= total
    matching_zero = phase_error + (1 - 2 * phase_error)
    matching_one = phase_error
    e10 = matching_zero if orient == 0 else 1 - matching_zero
    e20 = 1 - matching_zero if orient == 0 else matching_zero
    e11 = matching_one if orient == 0 else 1 - matching_one
    e21 = 1 - matching_one if orient == 0 else matching_one
    e0 = ((q0 * .5 + q1 * e10) + q2 * e20) + q3 * .5
    e1 = ((q0 * .5 + q1 * e11) + q2 * e21) + q3 * .5
    return e0, e1


@njit(cache=True, parallel=True)
def _frames_kernel(conditionals, frame, scaffold_orientation, roots, phase_theta, anchors, parent_counts, phase_error,
                   frozen, frozen_theta):
    """Family MAP frames int8 [N, L] (inferred, displayed): roots release their frame chain; every other sample its MAP
    frame path over painting-bin runs with its per-site Gibbs conditionals as phase_view emissions (review I4; 2.0's
    phase_view rules, FRAME_TIE for unanchored chains). Displayed = inferred for two-parent samples, 0 else."""
    samples, sites, _ = conditionals.shape
    inferred = np.empty((samples, sites), dtype=np.int8)
    displayed = np.empty_like(inferred)
    free_starts, free_trans = chains.chain_segments(phase_theta, sites)
    frozen_starts, frozen_trans = chains.chain_segments(frozen_theta, sites)
    for sample in prange(samples):
        if roots[sample]:
            for site in range(sites):
                inferred[sample, site] = frame[sample, site]
                displayed[sample, site] = 0
            continue
        starts, trans = (frozen_starts, frozen_trans) if frozen[sample] else (free_starts, free_trans)
        emissions = np.empty((sites, 2))
        for site in range(sites):
            emissions[site, 0], emissions[site, 1] = phase_emission(conditionals[sample, site],
                                                                    scaffold_orientation[sample, site], .5,
                                                                    phase_error)
        compact, anchor_run = chains.contracted_emissions(emissions, starts, anchors[sample])
        path = chains.binary_viterbi(compact, trans, anchor_run, chains.CHAIN_TIE,
                                     chains.FRAME_TIE if anchor_run < 0 else chains.CHAIN_TIE)
        for run in range(len(starts) - 1):
            flip = path[run]
            shown = flip if parent_counts[sample] >= 2 else 0
            for site in range(starts[run], starts[run + 1]):
                inferred[sample, site] = flip
                displayed[sample, site] = shown
    return inferred, displayed


@njit(cache=True)
def _parent_cavity(parent, site, exclude_group, genotype, selector, frame, potentials, orientation, roots, node_group,
                   group_parent, group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                   log_miss, out):
    """out[g]: the log weight of ``parent`` holding genotype g at ``site`` without child group ``exclude_group``'s
    factor: a root's psi, phi and other children; a non-root's psi, its own slot terms and other children."""
    if roots[parent]:
        _root_cavity(parent, site, frame[parent, site], exclude_group, genotype, selector, potentials, orientation,
                     group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss,
                     out)
        return
    for g in range(4):
        out[g] = potentials[parent, site, g]
    own = node_group[parent]
    table = np.empty((2, 2, 2))
    _slot_tables(own, site, genotype, selector, frame, potentials, orientation, roots, group_parent, group_child,
                 child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, True, table)
    s = selector[own, site]
    for g in range(4):
        out[g] += table[0, _allele(g, 0), (s >> 1) & 1] + table[1, _allele(g, 1), s & 1]
    for index in range(child_offsets[parent], child_offsets[parent + 1]):
        group = child_groups[index]
        if group == exclude_group:
            continue
        slot = child_slots[index]
        child_allele = _allele(genotype[group_child[group], site], slot)
        homologue = (selector[group, site] >> (1 - slot)) & 1
        for g in range(4):
            out[g] += log_keep if _allele(g, homologue) == child_allele else log_error


@njit(cache=True, parallel=True)
def _repair_emissions(potentials, genotype, selector, frame, orientation, roots, node_group, group_parent, group_child,
                      child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss):
    """Selector emissions float64 [G, L, 4] for the root-frame repair's votes (review I3): every observed parent's
    genotype summed with the voting child's own factor removed (its conditional cavity), the child's genotype summed
    with its potential and own children's factors; scaled to a maximum of 1 per site."""
    groups = len(group_child)
    sites = genotype.shape[1]
    emissions = np.empty((groups, sites, 4))
    for group in prange(groups):
        child = group_child[group]
        cavity = np.empty(4)
        base = np.empty(4)
        table = np.empty((2, 2, 2))
        values = np.empty(4)
        terms = np.empty(4)
        for site in range(sites):
            for slot in range(2):
                parent = group_parent[group, slot]
                if parent < 0:
                    for a in range(2):
                        for h in range(2):
                            table[slot, a, h] = LOG_HALF
                    continue
                _parent_cavity(parent, site, group, genotype, selector, frame, potentials, orientation, roots,
                               node_group, group_parent, group_child, child_offsets, child_groups, child_slots,
                               log_keep, log_error, log_match, log_miss, cavity)
                for a in range(2):
                    for h in range(2):
                        v0 = cavity[0] + (log_keep if _allele(0, h) == a else log_error)
                        v1 = cavity[1] + (log_keep if _allele(1, h) == a else log_error)
                        v2 = cavity[2] + (log_keep if _allele(2, h) == a else log_error)
                        v3 = cavity[3] + (log_keep if _allele(3, h) == a else log_error)
                        table[slot, a, h] = _lse4(v0, v1, v2, v3)
            for g in range(4):
                base[g] = potentials[child, site, g]
            _own_children_terms(child, site, genotype, selector, group_child, child_offsets, child_groups, child_slots,
                                log_keep, log_error, base)
            for s in range(4):
                for g in range(4):
                    terms[g] = base[g] + table[0, _allele(g, 0), (s >> 1) & 1] + table[1, _allele(g, 1), s & 1]
                values[s] = _lse4(terms[0], terms[1], terms[2], terms[3])
            top = max(max(values[0], values[1]), max(values[2], values[3]))
            for s in range(4):
                emissions[group, site, s] = math.exp(values[s] - top) if top > -np.inf else 1.0
    return emissions


@njit(cache=True, parallel=True)
def _conflicts(genotype, selector, frame, potentials, orientation, roots, group_child, child_offsets, child_groups,
               child_slots, log_keep, log_error, log_match, log_miss):
    """Mendelian-conflict diagnostic (review M4): bool [N, L] (parent, site) where the parent's best genotype (a
    root's mode of its summed terms, a non-root's MAP genotype) leaves at least one child's allele mistransmitted."""
    samples, sites = genotype.shape
    conflict = np.zeros((samples, sites), dtype=np.bool_)
    for parent in prange(samples):
        if child_offsets[parent] == child_offsets[parent + 1]:
            continue
        cavity = np.empty(4)
        for site in range(sites):
            if roots[parent]:
                _root_cavity(parent, site, frame[parent, site], -1, genotype, selector, potentials, orientation,
                             group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                             log_miss, cavity)
                best = 0
                for g in range(1, 4):
                    if cavity[g] > cavity[best]:
                        best = g
                g_parent = best
            else:
                g_parent = genotype[parent, site]
            for index in range(child_offsets[parent], child_offsets[parent + 1]):
                group = child_groups[index]
                slot = child_slots[index]
                homologue = (selector[group, site] >> (1 - slot)) & 1
                if _allele(g_parent, homologue) != _allele(genotype[group_child[group], site], slot):
                    conflict[parent, site] = True
                    break
    return conflict


def release_frames(family_map, state):
    """(inferred, displayed) family MAP frames int8 [N, L] of the converged configuration (_frames_kernel)."""
    w, g = family_map.workspace, family_map.graph
    conditionals = _conditionals(w.potentials, state.genotype, state.selector, state.frame, w.orientation,
                                 family_map.roots, g.node_group, g.group_parent, g.group_child,
                                 family_map.child_offsets, family_map.child_groups, family_map.child_slots,
                                 family_map.log_keep, family_map.log_error, family_map.log_match, family_map.log_miss)
    return _frames_kernel(conditionals, state.frame, w.scaffold_orientation, family_map.roots, w.phase_theta,
                          w.anchors, g.parent_counts, family_map.config.scaffold_phase_error, w.frozen,
                          w.frozen_theta)


def selector_emissions(family_map, state):
    """The root-frame repair's selector emissions float64 [G, L, 4] at the converged configuration (conditional
    cavities, review I3)."""
    w, g = family_map.workspace, family_map.graph
    return _repair_emissions(w.potentials, state.genotype, state.selector, state.frame, w.orientation, family_map.roots,
                             g.node_group, g.group_parent, g.group_child, family_map.child_offsets,
                             family_map.child_groups, family_map.child_slots, family_map.log_keep,
                             family_map.log_error, family_map.log_match, family_map.log_miss)


def conflict_sites(family_map, state):
    """Mendelian-conflict diagnostic: (parent, site) pairs (int64 [n, 2]) where the MAP needs a transmission error."""
    w = family_map.workspace
    mask = _conflicts(state.genotype, state.selector, state.frame, w.potentials, w.orientation, family_map.roots,
                      family_map.graph.group_child, family_map.child_offsets, family_map.child_groups,
                      family_map.child_slots, family_map.log_keep, family_map.log_error, family_map.log_match,
                      family_map.log_miss)
    return np.argwhere(mask).astype(np.int64)
