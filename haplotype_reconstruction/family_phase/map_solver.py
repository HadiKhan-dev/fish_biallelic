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
import os
import time

import numpy as np
from . import cavities as cavity_counts
from numba import njit, prange
from ..execution import claim_threads

from ..maths import chains, ties

LOG_HALF = math.log(0.5)
LOG_QUARTER = math.log(0.25)
# Acceptance quantum of a step's exact gain (nats). Placed by census of interval gains (work/IE/map_iter census,
# review I5c); see docs/methods/family_phase.md.
GAIN_QUANTUM = 2.0 ** -17
# Genotype ties within one site's maximisation (sums of a few log terms): within this the current genotype is kept.
GENOTYPE_TIE = 1e-12
# Compound moves (review I1): change pairs of any span, and the terminal segments between a chromosome end and the
# nearest change, are candidates (release 2.2; 2.1 capped spans at 8192 sites); the re-decode window widens the
# interval by max(COMPOUND_MARGIN, min(its span, COMPOUND_WIDTH)) on each side. Compound relabellings consider child
# changes up to RELABEL_REACH sites from a frame-chain boundary.
COMPOUND_MARGIN = 64
COMPOUND_WIDTH = 1024
RELABEL_REACH = 4096
# Compound candidates longer than LONG_SPAN sites, and terminal segments, form the second tier of the schedule (run
# when the first tier accepts nothing; release 2.2): their windowed work is long, and they are rarely needed.
LONG_SPAN = 8192
# Site moves: joint configurations of a sibship's genotypes enumerated at a Mendelian-conflict site, at most this many.
SITE_MAX_COMBOS = 4096
# Joint sibship refits inside compound seeds (release 2.2): at most this many joint configurations per site, except at
# Mendelian-conflict sites of short seeds (SITE_MAX_COMBOS); long seeds are refitted jointly at conflict sites only.
JOINT_REFIT_COMBOS = 64


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
    # Scaffold-conditioned genotypes usually have one or two finite states.
    # exp(-inf - finite_top) is exactly zero; preserve the original addition
    # order while avoiding those calls. With one finite state the sum is one.
    finite_count = ((v0 != -np.inf) + (v1 != -np.inf)
                    + (v2 != -np.inf) + (v3 != -np.inf))
    if finite_count == 1 and math.isfinite(top):
        return top + 0.0  # Same signed-zero result as top + log(1).
    e0 = 0.0 if v0 == -np.inf else math.exp(v0 - top)
    e1 = 0.0 if v1 == -np.inf else math.exp(v1 - top)
    e2 = 0.0 if v2 == -np.inf else math.exp(v2 - top)
    e3 = 0.0 if v3 == -np.inf else math.exp(v3 - top)
    return top + math.log(((e0 + e1) + e2) + e3)


@njit(cache=True, inline="always")
def _transition(previous, state, theta):
    changes = previous ^ state
    count = (changes >> 1) + (changes & 1)
    if count == 0:
        return 2 * _log_stay(theta)
    if count == 1:
        return _log_theta(theta) + _log_stay(theta)
    return 2 * _log_theta(theta)


@njit(cache=True, nogil=True)
def _root_cavity(root, site, frame_value, exclude_group, genotype, selector, potentials, orientation, group_child,
                 child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, out, cavities=None):
    """out[g] = psi_r(g) + log phi(g | frame_value) + the copy terms of r's children except group ``exclude_group``
    (-2: none of them, -1: all of them)."""
    orient = orientation[root, site]
    for g in range(4):
        out[g] = potentials[root, site, g] + _phi(g, frame_value, orient, log_match, log_miss)
    if exclude_group == -2:
        return
    if cavities is not None and cavities[0][root] >= 0:
        cavity_counts.add_terms(cavities, root, site, exclude_group, genotype, selector, log_keep, log_error, out)
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


@njit(cache=True, nogil=True)
def _root_term(root, site, frame_value, genotype, selector, potentials, orientation, group_child, child_offsets,
               child_groups, child_slots, log_keep, log_error, log_match, log_miss):
    """R_rl with frame ``frame_value`` (every child's factor in)."""
    # Fixed scalar accumulators preserve each genotype's incidence-order fold.
    # This hot site factor is also used when genotypes change inside a joint
    # enumeration, where a cached marginal would be invalid.
    orient = orientation[root, site]
    v0 = potentials[root, site, 0] + _phi(0, frame_value, orient, log_match, log_miss)
    v1 = potentials[root, site, 1] + _phi(1, frame_value, orient, log_match, log_miss)
    v2 = potentials[root, site, 2] + _phi(2, frame_value, orient, log_match, log_miss)
    v3 = potentials[root, site, 3] + _phi(3, frame_value, orient, log_match, log_miss)
    for index in range(child_offsets[root], child_offsets[root + 1]):
        group = child_groups[index]
        slot = child_slots[index]
        a = _allele(genotype[group_child[group], site], slot)
        h = (selector[group, site] >> (1 - slot)) & 1
        v0 += log_keep if _allele(0, h) == a else log_error
        v1 += log_keep if _allele(1, h) == a else log_error
        v2 += log_keep if _allele(2, h) == a else log_error
        v3 += log_keep if _allele(3, h) == a else log_error
    return _lse4(v0, v1, v2, v3)


@njit(cache=True, nogil=True)
def _slot_tables(group, site, genotype, selector, frame, potentials, orientation, roots, group_parent, group_child,
                 child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, siblings, table, cavities=None, root_scratch=None):
    """table[slot, a, h]: group's slot term at ``site`` when the child's slot allele is a and the parent's homologue h.
    Root parent: R of the parent with the child's factor in (its cavity without the child, every other child in when
    ``siblings``, none of them otherwise); non-root parent: its explicit genotype's copy term; unobserved slot: 1/2."""
    cavity = np.empty(4) if root_scratch is None else root_scratch
    for slot in range(2):
        parent = group_parent[group, slot]
        if parent < 0:
            for a in range(2):
                for h in range(2):
                    table[slot, a, h] = LOG_HALF
        elif roots[parent]:
            _root_cavity(parent, site, frame[parent, site], group if siblings else -2, genotype, selector, potentials,
                         orientation, group_child, child_offsets, child_groups, child_slots, log_keep, log_error,
                         log_match, log_miss, cavity, cavities)
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


@njit(cache=True, nogil=True)
def _own_children_terms(sample, site, genotype, selector, group_child, child_offsets, child_groups, child_slots,
                        log_keep, log_error, out, cavities=None):
    """out[g] += the copy terms of non-root ``sample``'s own children when it holds genotype g."""
    if cavities is not None and cavities[0][sample] >= 0:
        cavity_counts.add_terms(cavities, sample, site, -1, genotype, selector, log_keep, log_error, out)
        return
    for index in range(child_offsets[sample], child_offsets[sample + 1]):
        group = child_groups[index]
        slot = child_slots[index]
        child_allele = _allele(genotype[group_child[group], site], slot)
        homologue = (selector[group, site] >> (1 - slot)) & 1
        for g in range(4):
            out[g] += log_keep if _allele(g, homologue) == child_allele else log_error


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _viterbi4(emission, trans, left, left_theta, right, right_theta, prior, high):
    """Selector path over m markers (emission [m, 4] log, trans [m - 1] switch probabilities between consecutive
    markers, pinned states ``left``/``right`` of the markers just outside (-1: none) joined by left_theta/right_theta,
    prior [m] the current states). Ties: back-pointers within CHAIN_TIE take the fewest switches, then the
    predecessor's current state, then the lowest state (``high``: the highest); the final state is the current one
    within CHAIN_TIE, else the lowest (``high``: highest) within it."""
    markers = emission.shape[0]
    path = np.zeros(markers, dtype=np.int8)
    score = np.empty(4)
    new = np.empty(4)
    back = np.zeros((markers, 4), dtype=np.int8)
    for s in range(4):
        score[s] = emission[0, s] + (_transition(left, s, left_theta) if left >= 0 else 0.0)
    for k in range(1, markers):
        theta = trans[k - 1]
        prior_state = prior[k - 1]
        for s in range(4):
            top = -np.inf
            for p in range(4):
                value = score[p] + _transition(p, s, theta)
                if value > top:
                    top = value
            arg, fewest = 0, 3
            found = False
            for index in range(4):
                p = 3 - index if high else index
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
            score[s] += _transition(s, right, right_theta)
    top = max(max(score[0], score[1]), max(score[2], score[3]))
    state = prior[markers - 1]
    if score[state] < top - chains.CHAIN_TIE:
        for index in range(4):
            s = 3 - index if high else index
            if score[s] >= top - chains.CHAIN_TIE:
                state = s
                break
    for k in range(markers - 1, -1, -1):
        path[k] = state
        if k > 0:
            state = back[k, state]
    return path


@njit(cache=True, parallel=True, nogil=True)
def _block_emissions(lo, base, table, marker_sites, a_idx, b_idx, emission):
    """Independent marker emissions, called only by an already budgeted block."""
    for block in prange((b_idx - a_idx + 255) // 256):
        for k in range(256 * block, min(b_idx - a_idx, 256 * (block + 1))):
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


@njit(cache=True, parallel=True, nogil=True)
def _block_genotypes(child, lo, base, table, genotype, new_selector, new_genotype, reverse):
    """Independent genotype choices after the serial selector carry-forward."""
    for block in prange((base.shape[0] + 255) // 256):
        for i in range(256 * block, min(base.shape[0], 256 * (block + 1))):
            s = new_selector[i]
            best, arg = -np.inf, -1
            g_now = genotype[child, lo + i]
            for index in range(4):
                g = 3 - index if reverse else index
                if base[i, g] == -np.inf:
                    continue
                value = base[i, g] + table[i, 0, _allele(g, 0), (s >> 1) & 1] + table[i, 1, _allele(g, 1), s & 1]
                if value > best + GENOTYPE_TIE or (value >= best - GENOTYPE_TIE and g == g_now):
                    if value > best:
                        best = value
                    arg = g
            new_genotype[i] = arg


@njit(cache=True, nogil=True)
def _block_path(group, child, lo, base, table, genotype, selector, marker_offsets, marker_sites, marker_theta, a_idx,
                b_idx, left, right, new_selector, new_genotype, reverse=False, parallel=False):
    """A child block's optimum over sites [lo, lo + len(base)) given its terms there (base [n, 4], slot tables
    [n, 2, 2, 2]): Viterbi over the group's markers a_idx..b_idx-1 (emission: max over G), joined to the pinned
    selector states ``left`` / ``right`` of the markers just outside (-1: none), the selector carried forward between
    markers (backward before the group's first marker), the genotype the argmax given the selector. Ties: fewest
    switches, then the current state, then the lowest; a genotype within GENOTYPE_TIE of the best keeps the current
    one, else the lowest. ``reverse`` (convention reversal, maths.ties): the chain is decoded mirrored (an unavoidable
    switch lands at the rightmost tied position instead of the leftmost) and state and genotype ties go to the highest
    state. Fills new_selector and new_genotype [n]."""
    n = base.shape[0]
    markers = b_idx - a_idx
    path = np.zeros(max(markers, 1), dtype=np.int8)
    if markers > 0:
        emission = np.empty((markers, 4))
        if parallel:
            _block_emissions(lo, base, table, marker_sites, a_idx, b_idx, emission)
        else:
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
        trans = np.empty(max(markers - 1, 0))
        for k in range(markers - 1):
            trans[k] = marker_theta[a_idx + k]
        prior = np.empty(markers, dtype=np.int8)
        for k in range(markers):
            prior[k] = selector[group, marker_sites[a_idx + k]]
        left_theta = marker_theta[a_idx - 1] if left >= 0 else 0.0
        right_theta = marker_theta[b_idx - 1] if right >= 0 else 0.0
        if reverse:
            mirrored = np.empty((markers, 4))
            mirrored_trans = np.empty(max(markers - 1, 0))
            mirrored_prior = np.empty(markers, dtype=np.int8)
            for k in range(markers):
                for s in range(4):
                    mirrored[k, s] = emission[markers - 1 - k, s]
                mirrored_prior[k] = prior[markers - 1 - k]
            for k in range(markers - 1):
                mirrored_trans[k] = trans[markers - 2 - k]
            flipped = _viterbi4(mirrored, mirrored_trans, right, right_theta, left, left_theta, mirrored_prior, True)
            for k in range(markers):
                path[k] = flipped[markers - 1 - k]
        else:
            path = _viterbi4(emission, trans, left, left_theta, right, right_theta, prior, False)
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
        if parallel:
            continue
        s = current
        best, arg = -np.inf, -1
        g_now = genotype[child, site]
        for index in range(4):
            g = 3 - index if reverse else index
            if base[i, g] == -np.inf:
                continue
            value = base[i, g] + table[i, 0, _allele(g, 0), (s >> 1) & 1] + table[i, 1, _allele(g, 1), s & 1]
            if value > best + GENOTYPE_TIE or (value >= best - GENOTYPE_TIE and g == g_now):
                if value > best:
                    best = value
                arg = g
        new_genotype[i] = arg

    if parallel:
        _block_genotypes(child, lo, base, table, genotype, new_selector, new_genotype, reverse)


@njit(cache=True, nogil=True)
def _child_block(group, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                 child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
                 log_error, log_match, log_miss, quantum, decisions, decision_count, reverse=False, cavities=None, parallel_terms=False):
    """Exact update of child block ``group`` (module docstring). Returns (accepted intervals, total gain) and records
    evaluated small decisions."""
    child = group_child[group]
    sites = genotype.shape[1]
    first, stop = marker_offsets[group], marker_offsets[group + 1]
    markers = stop - first
    base = np.empty((sites, 4))
    table = np.empty((sites, 2, 2, 2))
    if parallel_terms:
        _block_terms(group, 0, sites, potentials, genotype, selector, frame, orientation, roots, group_parent,
                     group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                     log_miss, base, table, cavities)
    else:
        slot_table = np.empty((2, 2, 2))
        row = np.empty(4)
        root_scratch = np.empty(4)
        for site in range(sites):
            for g in range(4):
                row[g] = potentials[child, site, g]
            _own_children_terms(child, site, genotype, selector, group_child, child_offsets, child_groups, child_slots,
                                log_keep, log_error, row, cavities)
            for g in range(4):
                base[site, g] = row[g]
            _slot_tables(group, site, genotype, selector, frame, potentials, orientation, roots, group_parent, group_child,
                         child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, True,
                         slot_table, cavities, root_scratch)
            for k in range(2):
                for a in range(2):
                    for h in range(2):
                        table[site, k, a, h] = slot_table[k, a, h]
    new_selector = np.empty(sites, dtype=np.int8)
    new_genotype = np.empty(sites, dtype=np.int8)
    _block_path(group, child, 0, base, table, genotype, selector, marker_offsets, marker_sites, marker_theta,
                marker_offsets[group], marker_offsets[group + 1], -1, -1, new_selector, new_genotype, reverse, parallel_terms)
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


@njit(cache=True, nogil=True)
def _init_block(group, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
                log_error, log_match, log_miss, decoded, soft, soft_index, reverse=False, cavities=None):
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
            if cavities is not None and cavities[0][parent] >= 0:
                cavity_counts.add_terms(cavities, parent, site, -1, genotype, selector, log_keep, log_error, cavity)
            else:
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
                marker_offsets[group], marker_offsets[group + 1], -1, -1, new_selector, new_genotype, reverse)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _viterbi2(run_value, trans, anchor_run, left, left_theta, right, right_theta, end_now):
    """Frame path over m runs (run_value [m, 2], trans [m - 1], state 0 forced in ``anchor_run`` (-1: none), pinned
    frames ``left``/``right`` outside (-1: none)): back-pointers within CHAIN_TIE stay (fewest switches); the final
    frame is ``end_now`` (the current one) within CHAIN_TIE, else the better one."""
    m = run_value.shape[0]
    score = np.empty(2)
    new = np.empty(2)
    back = np.zeros((m, 2), dtype=np.int8)
    for f in range(2):
        score[f] = run_value[0, f]
        if left >= 0:
            score[f] += _log_stay(left_theta) if f == left else _log_theta(left_theta)
        if anchor_run == 0 and f == 1:
            score[f] = -np.inf
    for j in range(1, m):
        theta = trans[j - 1]
        for f in range(2):
            stay, jump = score[f] + _log_stay(theta), score[1 - f] + _log_theta(theta)
            top = max(stay, jump)
            back[j, f] = f if stay >= top - chains.CHAIN_TIE else 1 - f
            new[f] = top + run_value[j, f]
            if j == anchor_run and f == 1:
                new[f] = -np.inf
        score[0], score[1] = new[0], new[1]
    if right >= 0:
        for f in range(2):
            score[f] += _log_stay(right_theta) if f == right else _log_theta(right_theta)
    top = max(score[0], score[1])
    f = end_now if score[end_now] >= top - chains.CHAIN_TIE else (0 if score[0] >= score[1] else 1)
    path = np.empty(m, dtype=np.int8)
    for j in range(m - 1, -1, -1):
        path[j] = f
        if j > 0:
            f = back[j, f]
    return path


@njit(cache=True, nogil=True)
def _frame_path(run_value, trans, anchor_run, left, left_theta, right, right_theta, first_now, last_now, reverse=False):
    """_viterbi2, or under the convention reversal its mirror (an unavoidable frame switch lands at the rightmost tied
    run boundary instead of the leftmost; keep-current applies at the chain's first run)."""
    if not reverse:
        return _viterbi2(run_value, trans, anchor_run, left, left_theta, right, right_theta, last_now)
    m = run_value.shape[0]
    mirrored = np.empty((m, 2))
    mirrored_trans = np.empty(max(m - 1, 0))
    for j in range(m):
        mirrored[j, 0] = run_value[m - 1 - j, 0]
        mirrored[j, 1] = run_value[m - 1 - j, 1]
    for j in range(m - 1):
        mirrored_trans[j] = trans[m - 2 - j]
    flipped = _viterbi2(mirrored, mirrored_trans, m - 1 - anchor_run if anchor_run >= 0 else -1, right, right_theta,
                        left, left_theta, first_now)
    path = np.empty(m, dtype=np.int8)
    for j in range(m):
        path[j] = flipped[m - 1 - j]
    return path


@njit(cache=True, parallel=True, nogil=True)
def _root_run_terms(root, starts, genotype, selector, potentials, orientation, group_child,
                    child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, out):
    # Runs are independent. Keep each original site/frame accumulation order.
    for run in prange(len(starts) - 1):
        for site in range(starts[run], starts[run + 1]):
            for frame in range(2):
                out[run, frame] += _root_term(root, site, frame, genotype, selector, potentials, orientation,
                    group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss)


@njit(cache=True, nogil=True)
def _root_block(root, potentials, genotype, selector, frame, orientation, starts, run_theta, anchor, group_child,
                child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, quantum, decisions,
                decision_count, reverse=False, parallel_terms=False):
    """Exact update of root block ``root`` (Viterbi over its frame-chain runs); intervals of runs accepted on their own
    exact gains. Returns (accepted intervals, total gain)."""
    runs = len(starts) - 1
    run_value = np.zeros((runs, 2))
    anchor_run = -1
    for run in range(runs):
        if starts[run] <= anchor < starts[run + 1]:
            anchor_run = run
        if not parallel_terms:
            for site in range(starts[run], starts[run + 1]):
                for f in range(2):
                    run_value[run, f] += _root_term(root, site, f, genotype, selector, potentials, orientation,
                                                    group_child, child_offsets, child_groups, child_slots, log_keep,
                                                    log_error, log_match, log_miss)
    if parallel_terms:
        _root_run_terms(root, starts, genotype, selector, potentials, orientation, group_child,
                        child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                        log_miss, run_value)
    trans = np.empty(max(runs - 1, 0))
    for run in range(runs - 1):
        trans[run] = run_theta[run]
    path = _frame_path(run_value, trans, anchor_run, -1, 0.0, -1, 0.0, frame[root, starts[0]],
                       frame[root, starts[runs] - 1], reverse)
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


@njit(cache=True, parallel=True, nogil=True)
def _sweep_colour(blocks, is_root, node_group, potentials, genotype, selector, frame, orientation, roots, free_starts,
                  free_trans, frozen_starts, frozen_trans, frozen, anchors, group_parent, group_child, child_offsets,
                  child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error,
                  log_match, log_miss, quantum, max_decisions, reverse=False, cavities=None):
    """Exact updates of one colour's blocks (they share no free variable)."""
    count = len(blocks)
    accepted = np.zeros(count, dtype=np.int64)
    gains = np.zeros(count)
    decisions = np.zeros((count, max_decisions, 4))
    decision_counts = np.zeros((count, 1), dtype=np.int64)
    old_g = np.empty((count, genotype.shape[1]), np.int8) if cavities is not None else np.empty((0, 0), np.int8)
    old_s = np.empty_like(old_g)
    for index in prange(count):
        sample = blocks[index]
        if is_root[index]:
            starts = frozen_starts if frozen[sample] else free_starts
            trans = frozen_trans if frozen[sample] else free_trans
            accepted[index], gains[index] = _root_block(
                sample, potentials, genotype, selector, frame, orientation, starts, trans, anchors[sample], group_child,
                child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, quantum,
                decisions[index], decision_counts[index], reverse)
        else:
            if cavities is not None:
                old_g[index] = genotype[sample]
                old_s[index] = selector[node_group[sample]]
            accepted[index], gains[index] = _child_block(
                node_group[sample], potentials, genotype, selector, frame, orientation, roots, group_parent,
                group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta,
                log_keep, log_error, log_match, log_miss, quantum, decisions[index], decision_counts[index], reverse, cavities)
    if cavities is not None:
        for index in range(count):
            if not is_root[index]:
                cavity_counts.update_group(cavities, node_group[blocks[index]], 0, old_g[index], old_s[index], genotype, selector)
    return accepted, gains, decisions, decision_counts


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _relabel_scan(root, frame, selector, starts, run_theta, anchor, child_offsets, child_groups, child_slots,
                  marker_offsets, marker_sites, marker_theta, quantum, decisions, decision_count, reverse=False):
    """Every relabelling move of ``root`` (module docstring), boundaries in position order (reversed under the
    convention reversal), each evaluated on the current configuration and applied at once if its exact gain exceeds
    ``quantum``. Returns (moves, total gain)."""
    runs = len(starts) - 1
    sites = frame.shape[1]
    accepted, total = 0, 0.0
    for index in range(1, runs):
        run = runs - index if reverse else index
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


@njit(cache=True, nogil=True)
def _sum_fixed(values):
    # Keep the reduction outside a parallel function: Numba's automatic np.sum
    # reduction otherwise changes its grouping when a task claims more threads.
    total = 0.0
    for value in values:
        total += value
    return total


@njit(cache=True, parallel=True, nogil=True)
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
    return float(_sum_fixed(per_sample) + _sum_fixed(per_group))


# ------------------------------------------------------------------------------------------- compound moves

@njit(cache=True, parallel=True, nogil=True)
def _block_terms(group, lo, end, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                 child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, base, table, cavities=None):
    """A child block's terms on sites [lo, end): base [n, 4] (psi + its own children's copy terms) and slot tables
    [n, 2, 2, 2] (a root parent's R with the child's factor in, its other children in)."""
    child = group_child[group]
    for i in prange(end - lo):
        slot_table = np.empty((2, 2, 2))
        row = np.empty(4)
        root_scratch = np.empty(4)
        site = lo + i
        for g in range(4):
            row[g] = potentials[child, site, g]
        _own_children_terms(child, site, genotype, selector, group_child, child_offsets, child_groups, child_slots,
                            log_keep, log_error, row, cavities)
        for g in range(4):
            base[i, g] = row[g]
        _slot_tables(group, site, genotype, selector, frame, potentials, orientation, roots, group_parent, group_child,
                     child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, True,
                     slot_table, cavities, root_scratch)
        for k in range(2):
            for a in range(2):
                for h in range(2):
                    table[i, k, a, h] = slot_table[k, a, h]


@njit(cache=True, nogil=True)
def _window_end(group, hi, marker_offsets, marker_sites, sites):
    """End (exclusive) of a windowed re-decode of ``group`` inside [lo, hi): its first marker at or after hi (the sites
    carried from its last free marker reach it), or the chromosome end."""
    first, stop = marker_offsets[group], marker_offsets[group + 1]
    b_idx = first + np.searchsorted(marker_sites[first:stop], hi)
    return marker_sites[b_idx] if b_idx < stop else sites


@njit(cache=True, nogil=True)
def _window_child(group, lo, hi, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                  child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
                  log_error, log_match, log_miss, reverse=False):
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
                b_idx, left, right, new_selector, new_genotype, reverse)
    for i in range(n):
        selector[group, lo + i] = new_selector[i]
        genotype[child, lo + i] = new_genotype[i]


@njit(cache=True, nogil=True)
def _refit_genotypes(group, lo, end, potentials, genotype, selector, frame, orientation, roots, group_parent,
                     group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss,
                     reverse=False):
    """Child ``group``'s genotypes on [lo, end) set to their argmax given its selector (ties keep the current one,
    else the lowest; the highest under the convention reversal)."""
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
        for index in range(4):
            g = 3 - index if reverse else index
            if base[i, g] == -np.inf:
                continue
            value = base[i, g] + table[i, 0, _allele(g, 0), (s >> 1) & 1] + table[i, 1, _allele(g, 1), s & 1]
            if value > best + GENOTYPE_TIE or (value >= best - GENOTYPE_TIE and g == g_now):
                if value > best:
                    best = value
                arg = g
        genotype[child, lo + i] = arg


@njit(cache=True, nogil=True)
def _window_root(root, lo, hi, potentials, genotype, selector, frame, orientation, starts, run_theta, anchor,
                 group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss,
                 reverse=False):
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
    trans = np.empty(max(m - 1, 0))
    for j in range(m - 1):
        trans[j] = run_theta[r0 + j]
    path = _frame_path(run_value, trans, anchor_run, left, run_theta[r0 - 1] if left >= 0 else 0.0, right,
                       run_theta[r1 - 1] if right >= 0 else 0.0, frame[root, starts[r0]], frame[root, starts[r1] - 1],
                       reverse)
    for j in range(m):
        for site in range(starts[r0 + j], starts[r0 + j + 1]):
            frame[root, site] = path[j]


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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
    cavity = np.empty(4)
    for sample in range(samples):
        if roots[sample]:
            if not root_mask[sample]:
                continue
            starts = frozen_starts if frozen[sample] else free_starts
            trans = frozen_trans if frozen[sample] else free_trans
            for site in range(lo, hi):
                _root_cavity(sample, site, frame[sample, site], -1, genotype, selector, potentials, orientation,
                             group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                             log_miss, cavity)
                total += _lse4(cavity[0], cavity[1], cavity[2], cavity[3])
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


@njit(cache=True, nogil=True)
def _term_lists(groups, count, extra_root, roots, group_parent, group_child, child_offsets, child_groups, samples):
    """Sorted touched R/psi/copy/transition indices; no N- or G-sized scratch.

    Sorting deduplicates a local factor neighbourhood. The window scorer merges
    lists back into the old sample/group fold, preserving floating arithmetic.
    ``samples`` stays in the private call signature for the serial reference.
    """
    root_items = np.empty(2 * count + 1, dtype=np.int64)
    sample_items = np.empty(count, dtype=np.int64)
    nr = 0
    if extra_root >= 0:
        root_items[nr] = extra_root
        nr += 1
    capacity = count
    for q in range(count):
        child = group_child[groups[q]]
        capacity += child_offsets[child + 1] - child_offsets[child]
    copies = np.empty(capacity, dtype=np.int64)
    ng, ns = 0, 0
    for q in range(count):
        group = groups[q]
        child = group_child[group]
        if not roots[child]:
            sample_items[ns] = child
            ns += 1
        copies[ng] = group
        ng += 1
        for slot in range(2):
            parent = group_parent[group, slot]
            if parent >= 0 and roots[parent]:
                root_items[nr] = parent
                nr += 1
        for index in range(child_offsets[child], child_offsets[child + 1]):
            copies[ng] = child_groups[index]
            ng += 1
    return (np.unique(root_items[:nr]), np.unique(sample_items[:ns]),
            np.unique(copies[:ng]), np.unique(groups[:count]))


@njit(cache=True, nogil=True)
def _window_objective_lists(lo, hi, root_list, sample_list, group_list, transition_list, potentials, genotype, selector,
                      frame, orientation, roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans,
                      group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites,
                      marker_theta, log_keep, log_error, log_match, log_miss):
    """Sparse-list terms of J (the same terms as ``objective``) that a change confined to sites [lo, hi) and to the masked
    blocks (_term_masks) can alter: their per-site terms on [lo, hi), the masked roots' frame transitions at run starts
    in [lo, hi] and the masked groups' selector transitions with a marker in [lo, hi). The one routine for a compound
    move's before and after scores."""
    total = 0.0
    cavity = np.empty(4)
    # Merge the two sorted sample lists: exactly the original sample-axis fold.
    r, s = 0, 0
    while r < len(root_list) or s < len(sample_list):
        is_root = s == len(sample_list) or (r < len(root_list) and root_list[r] < sample_list[s])
        if is_root:
            sample = root_list[r]
            r += 1
            starts = frozen_starts if frozen[sample] else free_starts
            trans = frozen_trans if frozen[sample] else free_trans
            for site in range(lo, hi):
                _root_cavity(sample, site, frame[sample, site], -1, genotype, selector, potentials, orientation,
                             group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                             log_miss, cavity)
                total += _lse4(cavity[0], cavity[1], cavity[2], cavity[3])
            runs = len(starts) - 1
            run = max(np.searchsorted(starts[:runs], lo), 1)
            while run < runs and starts[run] <= hi:
                switched = frame[sample, starts[run]] != frame[sample, starts[run] - 1]
                total += _log_theta(trans[run - 1]) if switched else _log_stay(trans[run - 1])
                run += 1
        else:
            sample = sample_list[s]
            s += 1
            for site in range(lo, hi):
                total += potentials[sample, site, genotype[sample, site]]
    c, t = 0, 0
    while c < len(group_list) or t < len(transition_list):
        if t == len(transition_list) or (c < len(group_list) and group_list[c] < transition_list[t]):
            group = group_list[c]
        else:
            group = transition_list[t]
        child = group_child[group]
        if c < len(group_list) and group_list[c] == group:
            c += 1
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
        if t < len(transition_list) and transition_list[t] == group:
            t += 1
            first, stop = marker_offsets[group], marker_offsets[group + 1]
            a_idx = first + np.searchsorted(marker_sites[first:stop], lo)
            b_idx = first + np.searchsorted(marker_sites[first:stop], hi)
            for k in range(max(a_idx, first + 1), min(b_idx + 1, stop)):
                total += _transition(selector[group, marker_sites[k - 1]], selector[group, marker_sites[k]],
                                     marker_theta[k - 1])
    return total


@njit(cache=True, nogil=True)
def _pair_run(group, mask, a, b, selector, sites):
    """The change pair (c1, c2) of ``group``'s bits ``mask`` near the source pair (a, b): c1 in [a - span, b), c2 in
    (a, b + span], span = b - a, a chromosome end standing in for a change when the source interval reaches it (then
    no other change on that side). (-1, -1) when there is none."""
    span = b - a
    if a == 0:
        c1 = 0
    else:
        c1 = -1
        for site in range(max(1, a - span), b):
            if (selector[group, site] & mask) != (selector[group, site - 1] & mask):
                c1 = site
                break
        if c1 < 0:
            return -1, -1
    c2 = -1
    for site in range(c1 + 1, min(sites, b + span + 1)):
        if (selector[group, site] & mask) != (selector[group, site - 1] & mask):
            c2 = site
            break
    if b == sites:
        if c2 >= 0:
            return -1, -1
        for site in range(min(sites, b + span + 1), sites):
            if (selector[group, site] & mask) != (selector[group, site - 1] & mask):
                return -1, -1
        c2 = sites
    if c2 < 0 or c2 <= a or (c1 == 0 and c2 == sites):
        return -1, -1
    return c1, c2


@njit(cache=True, nogil=True)
def _pair_runs(root, a, b, selector, child_offsets, child_groups, child_slots, sites, runs):
    """Sibship-pair runs with tolerance (kind 2, release 2.2 M3): every child of root ``root`` whose bit for the root
    has a change pair (c1, c2) near the source pair (a, b) - c1 in [a - span, b), c2 in (a, b + span], span = b - a,
    either homologue - with a chromosome end standing in for a change when the source interval reaches it (then the
    child must have no other change on that side). Rows (group, c1, c2, mask) are written into ``runs``; returns
    their number. The source child itself qualifies with (a, b)."""
    span = b - a
    count = 0
    for index in range(child_offsets[root], child_offsets[root + 1]):
        other = child_groups[index]
        mask = 1 << (1 - child_slots[index])
        if a == 0:
            c1 = 0
        else:
            c1 = -1
            for site in range(max(1, a - span), b):
                if (selector[other, site] & mask) != (selector[other, site - 1] & mask):
                    c1 = site
                    break
            if c1 < 0:
                continue
        c2 = -1
        for site in range(c1 + 1, min(sites, b + span + 1)):
            if (selector[other, site] & mask) != (selector[other, site - 1] & mask):
                c2 = site
                break
        if b == sites:
            if c2 >= 0:
                continue                      # a later change: not a terminal run
            for site in range(min(sites, b + span + 1), sites):
                if (selector[other, site] & mask) != (selector[other, site - 1] & mask):
                    c2 = -2
                    break
            if c2 == -2:
                continue
            c2 = sites
        if c2 < 0 or c2 <= a or (c1 == 0 and c2 == sites):
            continue
        runs[count, 0] = other
        runs[count, 1] = c1
        runs[count, 2] = c2
        runs[count, 3] = mask
        count += 1
    return count


@njit(cache=True, nogil=True)
def _compound(kind, centre, bits, a, b, lo, hi, quantum, potentials, genotype, selector, frame, orientation, roots,
              frozen, free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent,
              group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta,
              log_keep, log_error, log_match, log_miss, reverse=False):
    """One compound move (review I1), accepted as a whole when its exact window gain exceeds ``quantum``:
      kind 0 (switch pair): child ``centre``'s selector bits ``bits`` flipped on sites [a, b) - removing its change
        pair at a and b (or its first or last change, when the interval reaches a chromosome end) - and its genotypes
        refitted there;
      kind 1 (track swap): non-root parent ``centre``'s two tracks swapped on [a, b) together with every child's bit
        for it (its children's copy terms are unchanged; the terms joining it to its own parents are not);
      kind 2 (sibship pair): ``bits`` names one bit of centre whose parent is a root r; every child of r with a change
        pair of its bit for r near centre's (_pair_runs: tolerance, release 2.2) has it flipped back on its own pair
        and its genotypes refitted: children that changed homologue of r together change back together;
    then the sibships of centre's parents and of centre are refitted jointly at every site of the seed interval
    (_joint_refit, release 2.2), then exact re-decodes inside the window of centre's parents (root frames, non-root
    blocks), their other children, centre's own children and centre (kind 0: neighbours first, centre last; kind 1:
    centre first). Returns (gain, accepted); a rejected move is reverted."""
    sites = genotype.shape[1]
    own = node_group[centre]
    parents = group_parent[own]
    seed_lo, seed_hi = a, b
    runs = np.zeros((max(child_offsets[parents[0] + 1] - child_offsets[parents[0]] if parents[0] >= 0 else 0,
                         child_offsets[parents[1] + 1] - child_offsets[parents[1]] if parents[1] >= 0 else 0, 1), 4),
                    dtype=np.int64)
    run_count = 0
    if kind == 3:
        # Co-located switch pairs (release 2.2): every bit of centre and of its parents' other children with a change
        # pair near centre's pair, whichever parent the bit belongs to (e.g. full and half siblings that all switch
        # their non-shared parent's bit at the same two markers).
        total = 1
        for slot in range(2):
            if parents[slot] >= 0:
                total += child_offsets[parents[slot] + 1] - child_offsets[parents[slot]]
        runs = np.zeros((2 * total, 4), dtype=np.int64)
        seen_groups = np.full(total, -1, dtype=np.int64)
        seen = 0
        for slot in range(-1, 2):
            if slot >= 0 and parents[slot] < 0:
                continue
            first = 0 if slot < 0 else child_offsets[parents[slot]]
            last = 1 if slot < 0 else child_offsets[parents[slot] + 1]
            for index in range(first, last):
                other = own if slot < 0 else child_groups[index]
                duplicate = False
                for q in range(seen):
                    if seen_groups[q] == other:
                        duplicate = True
                if duplicate:
                    continue
                seen_groups[seen] = other
                seen += 1
                for bit_slot in range(2):
                    if group_parent[other, bit_slot] < 0:
                        continue
                    mask = 1 << (1 - bit_slot)
                    c1, c2 = _pair_run(other, mask, a, b, selector, sites)
                    if c1 < 0:
                        continue
                    runs[run_count, 0] = other
                    runs[run_count, 1] = c1
                    runs[run_count, 2] = c2
                    runs[run_count, 3] = mask
                    run_count += 1
        for q in range(run_count):
            seed_lo = min(seed_lo, runs[q, 1])
            seed_hi = max(seed_hi, runs[q, 2])
        width = max(COMPOUND_MARGIN, min(seed_hi - seed_lo, COMPOUND_WIDTH))
        lo = min(lo, max(0, seed_lo - width))
        hi = max(hi, min(sites, seed_hi + width))
    if kind == 2:
        root = parents[0] if bits == 2 else parents[1]
        run_count = _pair_runs(root, a, b, selector, child_offsets, child_groups, child_slots, sites, runs)
        for q in range(run_count):
            seed_lo = min(seed_lo, runs[q, 1])
            seed_hi = max(seed_hi, runs[q, 2])
        width = max(COMPOUND_MARGIN, min(seed_hi - seed_lo, COMPOUND_WIDTH))
        lo = min(lo, max(0, seed_lo - width))
        hi = max(hi, min(sites, seed_hi + width))
    # Involved blocks: parents, siblings (the parents' other children), children.
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
    root_mask, sample_mask, group_mask, transition_mask = _term_lists(
        groups_list, groups_count, extra_root, roots, group_parent, group_child, child_offsets, child_groups,
        genotype.shape[0])
    before = _window_objective_lists(lo, end, root_mask, sample_mask, group_mask, transition_mask, potentials, genotype,
                               selector, frame, orientation, roots, frozen, free_starts, free_trans, frozen_starts,
                               frozen_trans, group_parent, group_child, child_offsets, child_groups, child_slots,
                               marker_offsets, marker_sites, marker_theta, log_keep, log_error, log_match, log_miss)
    # Seed.
    if kind == 0:
        for site in range(a, b):
            selector[own, site] ^= bits
        _refit_genotypes(own, a, b, potentials, genotype, selector, frame, orientation, roots, group_parent,
                         group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                         log_miss, reverse)
    elif kind == 1:
        for site in range(a, b):
            g = genotype[centre, site]
            genotype[centre, site] = ((g & 1) << 1) | (g >> 1)
            for index in range(child_offsets[centre], child_offsets[centre + 1]):
                selector[child_groups[index], site] ^= 1 << (1 - child_slots[index])
    else:
        # kinds 2 and 3
        for q in range(run_count):
            other, run_lo, run_hi, mask = runs[q, 0], runs[q, 1], runs[q, 2], runs[q, 3]
            for site in range(run_lo, run_hi):
                selector[other, site] ^= mask
            _refit_genotypes(other, run_lo, run_hi, potentials, genotype, selector, frame, orientation, roots,
                             group_parent, group_child, child_offsets, child_groups, child_slots, log_keep, log_error,
                             log_match, log_miss, reverse)
    # Joint sibship refit over the seed interval (M2): centre's parents and centre itself as a parent.
    refit = np.full(3 + 2 * run_count, -1, dtype=np.int64)
    refit_count = 0
    for slot in range(2):
        if parents[slot] >= 0:
            refit[refit_count] = parents[slot]
            refit_count += 1
    if child_offsets[centre + 1] > child_offsets[centre]:
        refit[refit_count] = centre
        refit_count += 1

    _joint_refit(refit, refit_count, seed_lo, seed_hi, SITE_MAX_COMBOS, potentials, genotype, selector, frame,
                 orientation, roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans, node_group,
                 group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites,
                 marker_theta, log_keep, log_error, log_match, log_miss, reverse,
                 seed_hi - seed_lo <= COMPOUND_WIDTH)
    # Re-decodes: one window, or for a long seed interval the two zones around its ends (its interior keeps the
    # seed's state; the gain is still scored over the whole window).
    zone_count = 1
    zones = np.empty((2, 2), dtype=np.int64)
    zones[0, 0], zones[0, 1] = lo, hi
    reach = COMPOUND_WIDTH + COMPOUND_MARGIN
    if seed_hi - seed_lo > 4 * reach:
        zone_count = 2
        zones[0, 0], zones[0, 1] = lo, min(hi, seed_lo + reach)
        zones[1, 0], zones[1, 1] = max(lo, seed_hi - reach), hi
    for zone in range(zone_count):
        z_lo, z_hi = zones[zone, 0], zones[zone, 1]
        for rnd in range(2):
            if kind == 1 or rnd == 1:
                _window_child(own, z_lo, z_hi, potentials, genotype, selector, frame, orientation, roots,
                              group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets,
                              marker_sites, marker_theta, log_keep, log_error, log_match, log_miss, reverse)
            for slot in range(2):
                parent = parents[slot]
                if parent < 0:
                    continue
                if roots[parent]:
                    starts = frozen_starts if frozen[parent] else free_starts
                    trans = frozen_trans if frozen[parent] else free_trans
                    _window_root(parent, z_lo, z_hi, potentials, genotype, selector, frame, orientation, starts,
                                 trans, anchors[parent], group_child, child_offsets, child_groups, child_slots,
                                 log_keep, log_error, log_match, log_miss, reverse)
                else:
                    _window_child(node_group[parent], z_lo, z_hi, potentials, genotype, selector, frame,
                                  orientation, roots, group_parent, group_child, child_offsets, child_groups,
                                  child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error,
                                  log_match, log_miss, reverse)
            for q in range(sibling_lo, children_hi):
                if groups[q] == own:
                    continue
                _window_child(groups[q], z_lo, z_hi, potentials, genotype, selector, frame, orientation, roots,
                              group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets,
                              marker_sites, marker_theta, log_keep, log_error, log_match, log_miss, reverse)
        _window_child(own, z_lo, z_hi, potentials, genotype, selector, frame, orientation, roots, group_parent,
                      group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites,
                      marker_theta, log_keep, log_error, log_match, log_miss, reverse)
    after = _window_objective_lists(lo, end, root_mask, sample_mask, group_mask, transition_mask, potentials, genotype,
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


@njit(cache=True, nogil=True)
def _scan_pairs(kind, centre, source, mask, quantum, max_span, margin, potentials, genotype, selector, frame,
                orientation, roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group,
                group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites,
                marker_theta, log_keep, log_error, log_match, log_miss, decisions, decision_count, reverse=False,
                tier=2):
    """Compound moves of ``kind`` around ``centre`` on the change pairs of group ``source``'s selector bits ``mask``
    (consecutive changes at most ``max_span`` sites apart, in position order, right to left under the convention
    reversal; the state is re-read after each move), then the two terminal segments. ``tier``: 0 = pairs of at most
    LONG_SPAN sites, 1 = longer pairs and the terminal segments (kinds 0 and 1 only), 2 = all. Returns (accepted
    moves, total gain)."""
    sites = genotype.shape[1]
    accepted, total = 0, 0.0
    first, stop = marker_offsets[source], marker_offsets[source + 1]
    k = stop - 1 if reverse else first + 1
    while (k > first) if reverse else (k < stop):
        if (selector[source, marker_sites[k]] & mask) == (selector[source, marker_sites[k - 1]] & mask):
            k = k - 1 if reverse else k + 1
            continue
        if reverse:
            # k is the pair's right change; find the change before it
            j = k - 1
            while j > first and (selector[source, marker_sites[j]] & mask) == (selector[source, marker_sites[j - 1]]
                                                                               & mask):
                j -= 1
            if j <= first:
                break
            a, b = marker_sites[j], marker_sites[k]
            bits = (selector[source, a] ^ selector[source, marker_sites[j - 1]]) & mask
        else:
            j = k + 1
            while j < stop and (selector[source, marker_sites[j]] & mask) == (selector[source, marker_sites[j - 1]]
                                                                              & mask):
                j += 1
            if j >= stop:
                break
            a, b = marker_sites[k], marker_sites[j]
            bits = (selector[source, a] ^ selector[source, marker_sites[k - 1]]) & mask
        usable = b - a <= max_span and (tier == 2 or (tier == 0) == (b - a <= LONG_SPAN))
        if kind == 3:
            usable = usable and b - a <= COMPOUND_WIDTH
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
                                 log_match, log_miss, reverse)
            if abs(gain) < 1.0 and decision_count[0] < decisions.shape[0]:
                decisions[decision_count[0], 0] = 3 + kind if kind < 3 else 8
                decisions[decision_count[0], 1] = source
                decisions[decision_count[0], 2] = a
                decisions[decision_count[0], 3] = gain
                decision_count[0] += 1
            if ok:
                accepted += 1
                total += gain
        k = j
    # Terminal segments (release 2.2): the stretch between a chromosome end and the nearest change of these bits,
    # flipped as a whole (kinds 0-2 then remove that change), left end first (right end first under the reversal).
    for side_index in range(2 if tier != 0 else 0):
        side = 1 - side_index if reverse else side_index
        k = -1
        if side == 0:
            for index in range(first + 1, stop):
                if (selector[source, marker_sites[index]] & mask) != (selector[source, marker_sites[index - 1]]
                                                                      & mask):
                    k = index
                    break
        else:
            for index in range(stop - 1, first, -1):
                if (selector[source, marker_sites[index]] & mask) != (selector[source, marker_sites[index - 1]]
                                                                      & mask):
                    k = index
                    break
        if k < 0:
            continue
        bits = (selector[source, marker_sites[k]] ^ selector[source, marker_sites[k - 1]]) & mask
        if side == 0:
            a, b = 0, marker_sites[k]
        else:
            a, b = marker_sites[k], sites
        usable = b - a <= max_span and b > a
        if kind == 2:
            parent = group_parent[source, 0] if bits == 2 else (group_parent[source, 1] if bits == 1 else -1)
            usable = usable and parent >= 0 and roots[parent]
        if not usable:
            continue
        width = max(margin, min(b - a, COMPOUND_WIDTH))
        lo, hi = max(0, a - width), min(sites, b + width)
        gain, ok = _compound(kind, centre, bits, a, b, lo, hi, quantum, potentials, genotype, selector, frame,
                             orientation, roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans,
                             anchors, node_group, group_parent, group_child, child_offsets, child_groups,
                             child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error,
                             log_match, log_miss, reverse)
        if abs(gain) < 1.0 and decision_count[0] < decisions.shape[0]:
            decisions[decision_count[0], 0] = 3 + kind if kind < 3 else 8
            decisions[decision_count[0], 1] = source
            decisions[decision_count[0], 2] = a
            decisions[decision_count[0], 3] = gain
            decision_count[0] += 1
        if ok:
            accepted += 1
            total += gain
    return accepted, total


@njit(cache=True, nogil=True)
def _compound_sweep(quantum, max_span, margin, potentials, genotype, selector, frame, orientation, roots, frozen,
                    free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent,
                    group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta,
                    log_keep, log_error, log_match, log_miss, decisions, decision_count, reverse=False, tier=2):
    """Every compound move once, sequentially in canonical order (sample order, then position; both reversed under
    the convention reversal): kind 0 on every
    child's selector change pairs; kind 1 for every non-root parent on its own change pairs and on each child's change
    pairs of the bit for it; kind 2 on every child's change pairs of one bit whose parent is a root. Window = the
    pair's interval widened by max(``margin``, min(its span, COMPOUND_WIDTH)) on each side. Returns (accepted moves,
    total gain)."""
    accepted, total = 0, 0.0
    groups = len(group_child)
    for kind in range(4 if tier != 1 else 2):
        for index in range(groups):
            group = groups - 1 - index if reverse else index
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
            for index_q in range(len(sources)):
                q = len(sources) - 1 - index_q if reverse else index_q
                moves, gain = _scan_pairs(kind, centre, sources[q], masks[q], quantum, max_span, margin, potentials,
                                          genotype, selector, frame, orientation, roots, frozen, free_starts,
                                          free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent,
                                          group_child, child_offsets, child_groups, child_slots, marker_offsets,
                                          marker_sites, marker_theta, log_keep, log_error, log_match, log_miss,
                                          decisions, decision_count, reverse, tier)
                accepted += moves
                total += gain
    return accepted, total


@njit(cache=True, nogil=True)
def _relabel_flip(root, boundary, anchor, frame, selector, child_offsets, child_groups, child_slots):
    """Apply (or undo: it is an involution) the relabelling of ``root`` at ``boundary`` (module docstring)."""
    sites = frame.shape[1]
    lo, hi = (0, boundary + 1) if (anchor >= 0 and boundary < anchor) else (boundary + 1, sites)
    for site in range(lo, hi):
        frame[root, site] ^= 1
        for index in range(child_offsets[root], child_offsets[root + 1]):
            selector[child_groups[index], site] ^= 1 << (1 - child_slots[index])


@njit(cache=True, nogil=True)
def _relabel_compound(root, run, lo, hi, quantum, potentials, genotype, selector, frame, orientation, roots, frozen,
                      free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent,
                      group_child, child_offsets, child_groups, child_slots, marker_offsets, marker_sites,
                      marker_theta, log_keep, log_error, log_match, log_miss, reverse=False):
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
    root_mask, sample_mask, group_mask, transition_mask = _term_lists(
        groups_list, groups_count, extra_root, roots, group_parent, group_child, child_offsets, child_groups,
        genotype.shape[0])
    before = _window_objective_lists(lo, end, root_mask, sample_mask, group_mask, transition_mask, potentials, genotype,
                               selector, frame, orientation, roots, frozen, free_starts, free_trans, frozen_starts,
                               frozen_trans, group_parent, group_child, child_offsets, child_groups, child_slots,
                               marker_offsets, marker_sites, marker_theta, log_keep, log_error, log_match, log_miss)
    for rnd in range(2):
        for index in range(first_child, last_child):
            _window_child(child_groups[index], lo, hi, potentials, genotype, selector, frame, orientation, roots,
                          group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets,
                          marker_sites, marker_theta, log_keep, log_error, log_match, log_miss, reverse)
        if rnd == 0:
            _window_root(root, lo, hi, potentials, genotype, selector, frame, orientation, starts, trans, anchor,
                         group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                         log_miss, reverse)
    after = _window_objective_lists(lo, end, root_mask, sample_mask, group_mask, transition_mask, potentials, genotype,
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


@njit(cache=True, nogil=True)
def _relabel_compound_sweep(quantum, max_span, margin, potentials, genotype, selector, frame, orientation, roots,
                            frozen, free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group,
                            group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets,
                            marker_sites, marker_theta, log_keep, log_error, log_match, log_miss, decisions,
                            decision_count, reverse=False):
    """Every root (sample order) and every boundary of its frame chain nearest to a change of a child's bit for it
    (within ``max_span`` sites; boundaries in position order, each once; equidistant boundaries: the left one): a
    compound relabelling with the window spanning the boundary and the triggering changes, widened by ``margin``.
    Under the convention reversal roots, boundaries and the equidistant choice are reversed. Returns (accepted moves,
    total gain)."""
    samples, sites = genotype.shape
    accepted, total = 0, 0.0
    for root_index in range(samples):
        root = samples - 1 - root_index if reverse else root_index
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
                for cand_index in range(2):
                    cand = (r - cand_index) if reverse else (r - 1 + cand_index)
                    if 1 <= cand < runs:
                        d = abs(starts[cand] - m)
                        if d < best_d:
                            best, best_d = cand, d
                if best < 0 or best_d > max_span:
                    continue
                reach_lo[best] = min(reach_lo[best], min(m, starts[best]))
                reach_hi[best] = max(reach_hi[best], max(m, starts[best]))
        for run_index in range(1, runs):
            run = runs - run_index if reverse else run_index
            if reach_hi[run] < 0:
                continue
            lo = max(0, reach_lo[run] - margin)
            hi = min(sites, reach_hi[run] + margin + 1)
            gain, ok = _relabel_compound(root, run, lo, hi, quantum, potentials, genotype, selector, frame, orientation,
                                         roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans, anchors,
                                         node_group, group_parent, group_child, child_offsets, child_groups,
                                         child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error,
                                         log_match, log_miss, reverse)
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


@njit(cache=True, nogil=True)
def _sibship(parent, roots, node_group, group_child, child_offsets, child_groups):
    """(members, groups) of ``parent``'s sibship: its children (and the parent itself when it is not a root)."""
    first, last = child_offsets[parent], child_offsets[parent + 1]
    count = last - first + (0 if roots[parent] else 1)
    members = np.empty(count, dtype=np.int64)
    groups = np.empty(count, dtype=np.int64)
    used = 0
    for index in range(first, last):
        members[used] = group_child[child_groups[index]]
        groups[used] = child_groups[index]
        used += 1
    if not roots[parent]:
        members[used] = parent
        groups[used] = node_group[parent]
    return members, groups


@njit(cache=True, nogil=True)
def _site_lists(root_mask, sample_mask, group_mask, roots, group_parent):
    """Index lists of the masked terms of one site: roots (R), non-root samples (psi), groups with a copy term to a
    non-root parent or an unobserved slot."""
    samples = len(roots)
    root_list = np.empty(samples, dtype=np.int64)
    sample_list = np.empty(samples, dtype=np.int64)
    nr, ns = 0, 0
    for sample in range(samples):
        if roots[sample]:
            if root_mask[sample]:
                root_list[nr] = sample
                nr += 1
        elif sample_mask[sample]:
            sample_list[ns] = sample
            ns += 1
    group_list = np.empty(len(group_mask), dtype=np.int64)
    ng = 0
    for group in range(len(group_mask)):
        if group_mask[group]:
            group_list[ng] = group
            ng += 1
    return root_list[:nr], sample_list[:ns], group_list[:ng]


@njit(cache=True, nogil=True)
def _site_terms(site, root_list, sample_list, group_list, potentials, genotype, selector, frame, orientation, roots,
                group_parent, group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                log_miss):
    """The J terms of one site restricted to the listed roots, samples and groups (selectors fixed): the same terms,
    in the same order, as _window_objective on [site, site + 1) with the corresponding masks and no transitions."""
    total = 0.0
    for q in range(len(root_list)):
        sample = root_list[q]
        total += _root_term(sample, site, frame[sample, site], genotype, selector, potentials, orientation,
                            group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                            log_miss)
    for q in range(len(sample_list)):
        sample = sample_list[q]
        total += potentials[sample, site, genotype[sample, site]]
    for q in range(len(group_list)):
        group = group_list[q]
        child = group_child[group]
        for slot in range(2):
            parent = group_parent[group, slot]
            if parent >= 0 and roots[parent]:
                continue
            if parent < 0:
                total += LOG_HALF
            else:
                homologue = (selector[group, site] >> (1 - slot)) & 1
                total += (log_keep if _allele(genotype[parent, site], homologue)
                          == _allele(genotype[child, site], slot) else log_error)
    return total


@njit(cache=True, nogil=True)
def _site_conflict(parent, site, genotype, selector, frame, potentials, orientation, roots, group_child,
                   child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss):
    """Whether ``parent``'s best genotype at ``site`` (a root's mode of its summed terms, a non-root's current
    genotype) leaves a child's allele mistransmitted (the per-site test of _conflicts)."""
    if roots[parent]:
        cavity = np.empty(4)
        _root_cavity(parent, site, frame[parent, site], -1, genotype, selector, potentials, orientation, group_child,
                     child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, cavity)
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
            return True
    return False


@njit(cache=True, nogil=True)
def _site_search(parent, site, max_combos, minimum_multi, potentials, genotype, selector, frame, orientation, roots,
                 frozen, free_starts, free_trans, frozen_starts, frozen_trans, node_group, group_parent, group_child,
                 child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
                 log_error, log_match, log_miss, reverse, best):
    """The exact joint maximum at one site over the ordered genotypes of ``parent``'s sibship (_sibship), selectors
    fixed: candidates are the genotypes of finite potential; nothing is searched above ``max_combos`` joint
    configurations or when fewer than ``minimum_multi`` members have more than one candidate. The site's J terms are
    scored by _site_terms (those of _window_objective on [site, site + 1)). The current configuration is kept unless another is better by
    more than GENOTYPE_TIE; among tied best configurations the first enumerated wins (the last under the convention
    reversal). Writes the best configuration into ``best`` (members' order) and leaves the genotypes as they were.
    Returns (gain, members); gain 0 when nothing was searched."""
    samples = genotype.shape[0]
    members, groups = _sibship(parent, roots, node_group, group_child, child_offsets, child_groups)
    used = len(members)
    options = np.zeros((used, 4), dtype=np.int8)
    sizes = np.zeros(used, dtype=np.int64)
    combos = 1
    multi = 0
    for q in range(used):
        best[q] = genotype[members[q], site]
        for g in range(4):
            if potentials[members[q], site, g] > -np.inf:
                options[q, sizes[q]] = g
                sizes[q] += 1
        if sizes[q] > 1:
            multi += 1
        combos *= max(sizes[q], 1)
        if combos > max_combos:
            return 0.0, members
    if combos <= 1 or multi < minimum_multi:
        return 0.0, members
    root_mask, sample_mask, group_mask, transition_mask = _term_lists(
        groups, used, parent if roots[parent] else -1, roots, group_parent, group_child, child_offsets, child_groups,
        samples)
    root_list, sample_list, group_list = (root_mask, sample_mask, group_mask)
    current = np.empty(used, dtype=np.int8)
    for q in range(used):
        current[q] = genotype[members[q], site]
    before = _site_terms(site, root_list, sample_list, group_list, potentials, genotype, selector, frame, orientation,
                         roots, group_parent, group_child, child_offsets, child_groups, child_slots, log_keep,
                         log_error, log_match, log_miss)
    best_value = before
    digits = np.zeros(used, dtype=np.int64)
    for combo in range(combos):
        code = combos - 1 - combo if reverse else combo
        for q in range(used):
            digits[q] = code % max(sizes[q], 1)
            code //= max(sizes[q], 1)
        for q in range(used):
            genotype[members[q], site] = options[q, digits[q]]
        value = _site_terms(site, root_list, sample_list, group_list, potentials, genotype, selector, frame,
                            orientation, roots, group_parent, group_child, child_offsets, child_groups, child_slots,
                            log_keep, log_error, log_match, log_miss)
        if value > best_value + GENOTYPE_TIE:
            best_value = value
            for q in range(used):
                best[q] = genotype[members[q], site]
    for q in range(used):
        genotype[members[q], site] = current[q]
    return best_value - before, members


@njit(cache=True, nogil=True)
def _site_move(parent, site, quantum, max_combos, potentials, genotype, selector, frame, orientation, roots, frozen,
               free_starts, free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent, group_child,
               child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
               log_error, log_match, log_miss, reverse=False):
    """Site move (sibship genotypes): at one site, the exact joint maximum over the ordered genotypes of
    ``parent``'s children (and of the parent itself when it is not a root), selectors fixed - e.g. full siblings
    swapping which parent transmitted which allele together, which no single child block can do (_site_search);
    accepted only above ``quantum``. Returns (gain, accepted)."""
    best = np.empty(child_offsets[parent + 1] - child_offsets[parent] + 1, dtype=np.int8)
    gain, members = _site_search(parent, site, max_combos, 1, potentials, genotype, selector, frame, orientation,
                                 roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans, node_group,
                                 group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets,
                                 marker_sites, marker_theta, log_keep, log_error, log_match, log_miss, reverse, best)
    if gain > quantum:
        for q in range(len(members)):
            genotype[members[q], site] = best[q]
        return gain, True
    return gain, False


@njit(cache=True, nogil=True)
def _joint_refit(parents, count, lo, hi, max_combos, potentials, genotype, selector, frame, orientation, roots, frozen,
                 free_starts, free_trans, frozen_starts, frozen_trans, node_group, group_parent, group_child,
                 child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
                 log_error, log_match, log_miss, reverse, all_sites):
    """Joint sibship genotype refit inside a compound seed (release 2.2, M2): at every site of [lo, hi) (only where it
    has a Mendelian conflict, _site_conflict, unless ``all_sites``) each of ``parents[:count]`` (with at least two
    children, or a non-root parent) has its sibship's genotypes set to their exact joint maximum (_site_search; at
    least two members with more than one candidate) when it beats the current configuration by more than
    GENOTYPE_TIE. Applied as found: the compound move accepts or reverts the whole."""
    best = np.empty(64, dtype=np.int8)
    for index in range(hi - lo):
        site = hi - 1 - index if reverse else lo + index
        for q in range(count):
            parent = parents[q]
            size = child_offsets[parent + 1] - child_offsets[parent]
            if size + 1 > len(best):
                best = np.empty(size + 1, dtype=np.int8)
            if size < 2 and roots[parent]:
                continue
            conflict = _site_conflict(parent, site, genotype, selector, frame, potentials, orientation, roots,
                                      group_child, child_offsets, child_groups, child_slots, log_keep, log_error,
                                      log_match, log_miss)
            if not (all_sites or conflict):
                continue
            gain, members = _site_search(parent, site,
                                         max_combos if (conflict and all_sites) else min(max_combos, JOINT_REFIT_COMBOS),
                                         2, potentials, genotype, selector, frame,
                                         orientation, roots, frozen, free_starts, free_trans, frozen_starts,
                                         frozen_trans, node_group, group_parent, group_child, child_offsets,
                                         child_groups, child_slots, marker_offsets, marker_sites, marker_theta,
                                         log_keep, log_error, log_match, log_miss, reverse, best)
            if gain > GENOTYPE_TIE:
                for m in range(len(members)):
                    genotype[members[m], site] = best[m]


@njit(cache=True, parallel=True, nogil=True)
def _site_sweep(quantum, max_combos, potentials, genotype, selector, frame, orientation, roots, frozen, free_starts,
                free_trans, frozen_starts, frozen_trans, anchors, node_group, group_parent, group_child, child_offsets,
                child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error, log_match,
                log_miss, decisions, decision_count, reverse=False):
    """Site moves at every Mendelian-conflict site (_conflicts at the sweep's start) of every root with at least two
    children and every non-root parent, sequentially (parents in sample order, then sites; both reversed under the
    convention reversal). Returns (accepted moves, total gain)."""
    conflict = _conflicts(genotype, selector, frame, potentials, orientation, roots, group_child, child_offsets,
                          child_groups, child_slots, log_keep, log_error, log_match, log_miss, reverse)
    samples, sites = genotype.shape
    accepted, total = 0, 0.0
    for parent_index in range(samples):
        parent = samples - 1 - parent_index if reverse else parent_index
        kids = child_offsets[parent + 1] - child_offsets[parent]
        if kids == 0 or (kids < 2 and roots[parent]):
            continue
        columns = np.flatnonzero(conflict[parent])
        gains = np.empty(len(columns), dtype=np.float64)
        oks = np.empty(len(columns), dtype=np.bool_)
        for index in prange(len(columns)):
            site = columns[len(columns) - 1 - index] if reverse else columns[index]
            gain, ok = _site_move(parent, site, quantum, max_combos, potentials, genotype, selector, frame,
                                  orientation, roots, frozen, free_starts, free_trans, frozen_starts, frozen_trans,
                                  anchors, node_group, group_parent, group_child, child_offsets, child_groups,
                                  child_slots, marker_offsets, marker_sites, marker_theta, log_keep, log_error,
                                  log_match, log_miss, reverse)
            gains[index], oks[index] = gain, ok
        for index in range(len(columns)):
            site = columns[len(columns) - 1 - index] if reverse else columns[index]
            gain, ok = gains[index], oks[index]
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


def colouring(graph, reverse=False):
    """The original degree-ordered greedy colours, with implicit sibling cliques.

    Each child belongs to at most two root factors. Pair multiplicities give
    exact union degrees without expanding the cliques. Root colour sets and
    their first missing colours avoid rescanning each sibling on every visit.
    Storage is O(N+E); conflicting-colour probes can still be superlinear for
    adversarial overlapping root families. No new visiting schedule is used.
    """
    samples = len(graph.roots)
    direct = [set() for _ in range(samples)]
    roots_of = [set() for _ in range(samples)]
    root_children = {}
    for parent, child in zip(graph.parents, graph.children):
        parent, child = int(parent), int(child)
        direct[parent].add(child)
        direct[child].add(parent)
        if graph.roots[parent]:
            root_children.setdefault(parent, set()).add(child)
            roots_of[child].add(parent)
    roots_of = [tuple(sorted(roots)) for roots in roots_of]
    pairs = {}
    for roots in roots_of:
        if len(roots) == 2:
            pairs[roots] = pairs.get(roots, 0) + 1
    degree = []
    for sample, roots in enumerate(roots_of):
        siblings = sum(len(root_children[root]) for root in roots)
        if len(roots) == 2:
            siblings -= pairs[roots]
        if roots:
            siblings -= 1  # the child belongs to its own root factor(s)
        degree.append(siblings + sum(not any(other in root_children[root] for root in roots)
                                     for other in direct[sample]))
    order = sorted(range(samples), key=lambda n: (-degree[n], -n if reverse else n))
    colour = np.full(samples, -1, dtype=np.int64)
    root_colours = {root: set() for root in root_children}
    root_missing = dict.fromkeys(root_children, 0)
    groups = []
    for sample in order:
        roots = roots_of[sample]
        used = {int(colour[other]) for other in direct[sample] if colour[other] >= 0}
        c = max((root_missing[root] for root in roots), default=0)
        while c in used or any(c in root_colours[root] for root in roots):
            c += 1
        colour[sample] = c
        while len(groups) <= c:
            groups.append([])
        for root in roots:
            root_colours[root].add(c)
            while root_missing[root] in root_colours[root]:
                root_missing[root] += 1
    for sample, c in enumerate(colour):
        groups[int(c)].append(sample)
    result = [np.asarray(group, dtype=np.int64) for group in groups]
    return result[::-1] if reverse else result


def initial_batches(graph, order):
    """Dependency waves preserving every original sibling/parent precedence.

    _init_block reads decoded siblings of either parent and the non-root parent's
    soft conditional. Disjoint waves neither read nor write each other's state.
    """
    last_parent = {}
    levels = np.zeros(len(graph.group_child), dtype=np.int64)
    batches = []
    for group in order:
        group = int(group)
        parents = [int(p) for p in graph.group_parent[group] if p >= 0]
        dependencies = [last_parent[p] for p in parents if p in last_parent]
        dependencies += [int(graph.node_group[p]) for p in parents if graph.node_group[p] >= 0]
        level = 1 + max((int(levels[x]) for x in dependencies), default=-1)
        levels[group] = level
        while len(batches) <= level:
            batches.append([])
        batches[level].append(group)
        for parent in parents:
            last_parent[parent] = group
    return tuple(np.asarray(batch, dtype=np.int64) for batch in batches)


@njit(cache=True, parallel=True, nogil=True)
def _init_batch(groups, potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta, log_keep,
                log_error, log_match, log_miss, decoded, soft, soft_index, reverse=False, cavities=None):
    for k in prange(len(groups)):
        _init_block(groups[k], potentials, genotype, selector, frame, orientation, roots, group_parent, group_child,
                    child_offsets, child_groups, child_slots, marker_offsets, marker_sites, marker_theta,
                    log_keep, log_error, log_match, log_miss, decoded, soft, soft_index, reverse, cavities)

    if cavities is not None:
        for group in groups:
            child = group_child[group]
            cavity_counts.update_group(cavities, group, 0, genotype[child], selector[group], genotype, selector, True)


def descent_order(graph, reverse=False):
    """Child groups in generation order (a child after all its observed parents), then sample index (reversed under
    the convention reversal)."""
    samples = len(graph.roots)
    depth = np.zeros(samples, dtype=np.int64)
    children_of = [[] for _ in range(samples)]
    indegree = np.zeros(samples, dtype=np.int64)
    for parent, child in zip(graph.parents, graph.children):
        children_of[int(parent)].append(int(child))
        indegree[child] += 1
    queue = np.flatnonzero(indegree == 0).tolist()
    for parent in queue:
        for child in children_of[parent]:
            depth[child] = max(depth[child], depth[parent] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(queue) != samples:
        raise ValueError('family phase requires an acyclic pedigree')
    children = np.asarray(graph.group_child, dtype=np.int64)
    return np.lexsort((-children if reverse else children, depth[children])).astype(np.int64)


def _children_of(graph, samples):
    """CSR (offsets int64 [N+1], groups int64 [E], slots int64 [E]) of each sample's child groups (canonical order:
    parent, then child group, then slot)."""
    order = np.lexsort((graph.slots, graph.edge_group, graph.parents))
    parents = graph.parents[order]
    offsets = np.searchsorted(parents, np.arange(samples + 1)).astype(np.int64)
    return offsets, graph.edge_group[order].astype(np.int64), graph.slots[order].astype(np.int64)


def _runs_of(mask):
    """Half-open runs [start, stop) of True in a bool vector."""
    index = np.flatnonzero(mask)
    if len(index) == 0:
        return []
    cuts = np.flatnonzero(np.diff(index) > 1)
    return list(zip(np.r_[index[0], index[cuts + 1]].tolist(), (np.r_[index[cuts], index[-1]] + 1).tolist()))


def _difference_items(state, other, roots, graph):
    """Differing runs of two configurations: (kind 'F'|'S'|'G', row, start, stop, family set) - roots' frames,
    groups' selectors, non-roots' genotypes; family = the sample, its observed parents (and its children for a
    genotype)."""
    parents_of = {int(c): [int(p) for p in graph.group_parent[graph.node_group[c]] if p >= 0] for c in graph.group_child}
    children_of = {}
    for parent, child in zip(graph.parents, graph.children):
        children_of.setdefault(int(parent), []).append(int(child))
    items = []
    for root in np.flatnonzero(roots):
        for lo, hi in _runs_of(state.frame[root] != other.frame[root]):
            items.append(("F", int(root), lo, hi, frozenset((int(root),))))
    for group, child in enumerate(graph.group_child):
        child = int(child)
        family = frozenset((child, *parents_of[child]))
        for lo, hi in _runs_of(state.selector[group] != other.selector[group]):
            items.append(("S", group, lo, hi, family))
        for lo, hi in _runs_of(state.genotype[child] != other.genotype[child]):
            items.append(("G", child, lo, hi, family | frozenset(children_of.get(child, ()))))
    return items


def _difference_clusters(items, margin):
    """Clusters of difference items joined by family overlap and site proximity (gap <= margin): a list of
    (first site, member indices), in position order."""
    count = len(items)
    parent = list(range(count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    factors = {}
    for i, item in enumerate(items):
        for key in item[4]:
            factors.setdefault(key, []).append(i)
    for members in factors.values():
        ordered = sorted(members, key=lambda i: (items[i][2], items[i][3], i))
        representative, right = -1, -1
        for i in ordered:
            left, stop = items[i][2], items[i][3]
            if representative >= 0 and left <= right + margin:
                parent[find(i)] = find(representative)
                right = max(right, stop)
            else:
                representative, right = i, stop
    groups = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    clusters = [(min(items[i][2] for i in members), sorted(members)) for members in groups.values()]
    return sorted(clusters, key=lambda c: (c[0], c[1]))


def _apply_items(state, other, items, members):
    for i in members:
        kind, row, lo, hi, _ = items[i]
        if kind == "F":
            state.frame[row, lo:hi] = other.frame[row, lo:hi]
        elif kind == "S":
            state.selector[row, lo:hi] = other.selector[row, lo:hi]
        else:
            state.genotype[row, lo:hi] = other.genotype[row, lo:hi]


def _save_items(state, items, members):
    saved = []
    for i in members:
        kind, row, lo, hi, _ = items[i]
        source = state.frame if kind == "F" else (state.selector if kind == "S" else state.genotype)
        saved.append(source[row, lo:hi].copy())
    return saved


def _restore_items(state, items, members, saved):
    for i, values in zip(members, saved):
        kind, row, lo, hi, _ = items[i]
        target = state.frame if kind == "F" else (state.selector if kind == "S" else state.genotype)
        target[row, lo:hi] = values


class FamilyMap:
    """The solver's fixed inputs for one chromosome (from a FamilyWorkspace). ``reverse`` overrides the convention
    reversal flag (None: maths.ties.reverse_conventions(); the two-schedule estimator builds its mirror with it)."""

    def __init__(self, workspace, config, reverse=None):
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
        # Convention reversal (release 2.2 acceptance test; maths.ties): every arbitrary order and tie convention of
        # the solver runs reversed; the stated priors (fewest switches, keep-current inside a tie class) do not.
        self.reverse = bool(ties.reverse_conventions()) if reverse is None else bool(reverse)
        self.colours = colouring(graph, self.reverse)
        self.roots = np.asarray(graph.roots, dtype=np.bool_)
        self.descent_order = descent_order(graph, self.reverse)
        self.initial_batches = initial_batches(graph, self.descent_order)
        self.quantum = GAIN_QUANTUM
        self.phase_steps, self.phase_seconds = {}, {}
        self.compound_schedule = os.environ.get('HAPLOTYPES_COMPOUND_SCHEDULE', 'coloured')
        if self.compound_schedule not in ('serial', 'coloured'):
            raise ValueError('HAPLOTYPES_COMPOUND_SCHEDULE must be serial or coloured')
        self.compound_layout = []
        self.cavity_mode = os.environ.get('HAPLOTYPES_CAVITY_MODE', 'counts')
        if self.cavity_mode not in ('counts', 'fold'):
            raise ValueError('HAPLOTYPES_CAVITY_MODE must be counts or fold')

    def cavities(self, state, decoded=None):
        if self.cavity_mode == 'fold':
            return None
        return cavity_counts.build(state.genotype, state.selector, self.graph, self.child_offsets,
                                   self.child_groups, self.child_slots, decoded)

    def objective(self, state):
        claim_threads(max_threads=self.samples)
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
        cavities = self.cavities(state, decoded)
        for batch in self.initial_batches:
            claim_threads(max_threads=len(batch))
            _init_batch(batch, w.potentials, state.genotype, state.selector, state.frame, w.orientation,
                        self.roots, g.group_parent, g.group_child, self.child_offsets, self.child_groups,
                        self.child_slots, self.marker_offsets, self.marker_sites, self.marker_theta, self.log_keep,
                        self.log_error, self.log_match, self.log_miss, decoded, soft, soft_index, self.reverse, cavities)
        return state

    def sweep(self, state, decisions_out=None, tier=0):
        """One sweep. Tier 0: every colour's blocks, every root's relabelling scan, the compound moves (kinds 0-2,
        pairs of at most LONG_SPAN sites), the compound relabellings and the site moves, in that order. Tier 1: the
        long compound moves (longer pairs and terminal segments). Returns (steps, total gain); per-phase accepted
        steps and seconds accumulate in self.phase_steps and self.phase_seconds."""
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

        common = (w.potentials, state.genotype, state.selector, state.frame, w.orientation, self.roots, w.frozen,
                  self.free_starts, self.free_trans, self.frozen_starts, self.frozen_trans, w.anchors, g.node_group,
                  g.group_parent, g.group_child, self.child_offsets, self.child_groups, self.child_slots,
                  self.marker_offsets, self.marker_sites, self.marker_theta, self.log_keep, self.log_error,
                  self.log_match, self.log_miss)
        if tier == 1:
            claim_threads(max_threads=1)
            started = time.perf_counter()
            decisions = np.zeros((4096, 4))
            count = np.zeros(1, dtype=np.int64)
            if self.compound_schedule == 'coloured':
                from .compound_schedule import sweep as compound_sweep
                moves, gain = compound_sweep(self, state, decisions, count, tier=1)
            else:
                moves, gain = _compound_sweep(self.quantum, self.sites, COMPOUND_MARGIN, *common, decisions, count,
                                              self.reverse, 1)
            record("compound_long", moves, gain, decisions, count[0], started)
            return steps, total
        started = time.perf_counter()
        cavities = self.cavities(state)
        for blocks in self.colours:
            claim_threads(max_threads=len(blocks))
            is_root = self.roots[blocks]
            from ..execution.budget import current_budget
            if self.sites >= 1024 and len(blocks) < current_budget().threads():
                from .block_schedule import sweep_colour
                accepted, gains, decisions, counts = sweep_colour(self, state, blocks, cavities)
            else:
                accepted, gains, decisions, counts = _sweep_colour(
                    blocks, is_root, g.node_group, w.potentials, state.genotype, state.selector, state.frame,
                    w.orientation, self.roots, self.free_starts, self.free_trans, self.frozen_starts, self.frozen_trans,
                    w.frozen, w.anchors, g.group_parent, g.group_child, self.child_offsets, self.child_groups,
                    self.child_slots, self.marker_offsets, self.marker_sites, self.marker_theta, self.log_keep,
                    self.log_error, self.log_match, self.log_miss, self.quantum, 256, self.reverse, cavities)
            for index in range(len(blocks)):
                record("blocks", accepted[index], gains[index], decisions[index], counts[index, 0], started)
                started = time.perf_counter()
        started = time.perf_counter()
        claim_threads(max_threads=1)
        del cavities
        root_order = np.flatnonzero(self.roots)
        for root in (root_order[::-1] if self.reverse else root_order):
            if self.child_offsets[root] == self.child_offsets[root + 1]:
                continue
            starts, trans = ((self.frozen_starts, self.frozen_trans) if w.frozen[root]
                             else (self.free_starts, self.free_trans))
            decisions = np.zeros((256, 4))
            count = np.zeros(1, dtype=np.int64)
            moves, gain = _relabel_scan(int(root), state.frame, state.selector, starts, trans, int(w.anchors[root]),
                                        self.child_offsets, self.child_groups, self.child_slots, self.marker_offsets,
                                        self.marker_sites, self.marker_theta, self.quantum, decisions, count,
                                        self.reverse)
            record("relabel", moves, gain, decisions, count[0], started)
            started = time.perf_counter()
        started = time.perf_counter()
        decisions = np.zeros((4096, 4))
        count = np.zeros(1, dtype=np.int64)
        if self.compound_schedule == 'coloured':
            from .compound_schedule import sweep as compound_sweep
            moves, gain = compound_sweep(self, state, decisions, count, tier=0)
        else:
            moves, gain = _compound_sweep(self.quantum, self.sites, COMPOUND_MARGIN, *common, decisions, count,
                                          self.reverse, 0)
        record("compound", moves, gain, decisions, count[0], started)
        started = time.perf_counter()
        decisions = np.zeros((4096, 4))
        count = np.zeros(1, dtype=np.int64)
        if self.compound_schedule == 'coloured':
            moves, gain = compound_sweep(self, state, decisions, count, relabel=True)
        else:
            moves, gain = _relabel_compound_sweep(self.quantum, RELABEL_REACH, COMPOUND_MARGIN, *common, decisions,
                                                  count, self.reverse)
        record("compound_relabel", moves, gain, decisions, count[0], started)
        started = time.perf_counter()
        decisions = np.zeros((4096, 4))
        count = np.zeros(1, dtype=np.int64)
        claim_threads()
        moves, gain = _site_sweep(self.quantum, SITE_MAX_COMBOS, *common, decisions, count, self.reverse)
        record("site", moves, gain, decisions, count[0], started)
        return steps, total

    def solve(self, max_sweeps, state=None, long_tier=True):
        """MapResult from the default start (or ``state``); RuntimeError when ``max_sweeps`` sweeps still accept a
        step, or when the from-scratch J does not rise by the accepted gains (review B2c). ``long_tier`` False stops
        when a tier-0 sweep accepts nothing (no long compound moves)."""
        state = self.initial_state() if state is None else state
        values = [self.objective(state)]
        accepted, decisions = [], []
        tier = 0
        for _ in range(int(max_sweeps)):
            steps, gain = self.sweep(state, decisions, tier)
            accepted.append(steps)
            values.append(self.objective(state))
            tolerance = 1e-10 * abs(values[-1]) + 1e-6
            if steps and not values[-1] - values[-2] >= gain - tolerance:
                raise RuntimeError(f"family MAP: J rose by {values[-1] - values[-2]!r} but the accepted gains sum to "
                                   f"{gain!r}")
            # Two tiers (release 2.2): tier-0 sweeps until one accepts nothing, then one tier-1 sweep (long compound
            # moves); converged when that accepts nothing too, else back to tier 0.
            if steps == 0 and (tier == 1 or not long_tier):
                return MapResult(state, len(accepted), tuple(accepted), tuple(values), tuple(decisions))
            tier = 1 if (steps == 0 and tier == 0) else 0
        raise RuntimeError(f"family MAP did not converge in {max_sweeps} sweeps (accepted per sweep {accepted})")

    def solve_symmetric(self, max_sweeps):
        """Unordered two-schedule search, with common-start independent transplants.

        The unique better endpoint owns the continuation schedule. A tied pair gets
        equal continuation effort; tied final alternatives are retained for release,
        never interpreted as evidence for the representative's biological phase.
        """
        first = self.solve(max_sweeps, long_tier=False)
        mirror = FamilyMap(self.workspace, self.config, reverse=not self.reverse)
        second = mirror.solve(max_sweeps, long_tier=False)
        endpoints = ((self, first), (mirror, second))
        scores = [r.objective[-1] for _, r in endpoints]
        best = ties.tie_classes(scores, self.quantum, descending=True)
        started = time.perf_counter()
        branches, merge_rows = [], []
        # At most four complete continuations: no unbounded archive of chromosome
        # states. Unvisited combinations are recorded as search incompleteness;
        # only evaluated release differences can mask called phase.
        continuation_bound = 4
        endpoint_indices = np.flatnonzero(best.classes == 0)
        per_endpoint = continuation_bound // len(endpoint_indices)
        for index in endpoint_indices:
            owner, base = endpoints[index]
            other = endpoints[1 - index][1]
            results, rows = owner.continue_endpoint(max_sweeps, base, other, per_endpoint)
            branches.extend((owner.reverse, result) for result in results)
            merge_rows.extend(rows)
        ranked = ties.tie_classes([r.objective[-1] for _, r in branches], self.quantum, descending=True)
        selected = np.flatnonzero(ranked.classes == 0)
        retained = [branches[i] for i in selected]
        # This order only labels the stored representative. All tied alternatives
        # are passed to release and their disagreements remain ambiguity.
        retained.sort(key=lambda pair: (pair[1].state.frame.tobytes(), pair[1].state.genotype.tobytes(),
                                       pair[1].state.selector.tobytes(), pair[0]))
        self.release_alternatives = tuple(retained)
        self.release_reverse, final = retained[0]
        elapsed = time.perf_counter() - started
        self.phase_seconds["merge_continuation"] = elapsed
        record = dict(J_schedule=float(scores[0]), J_mirror=float(scores[1]),
                      base="tied" if best.tied_winner else ("schedule" if best.classes[0] == 0 else "mirror"),
                      tied_base=bool(best.tied_winner), branches=len(branches),
                      release_alternatives=len(retained), continuation_seconds=elapsed,
                      continuation_bound=continuation_bound,
                      bounded_out=any(r["bounded_out"] for r in merge_rows),
                      clusters=sum(r["clusters"] for r in merge_rows),
                      transplanted=sum(r["transplanted"] for r in merge_rows),
                      tied_transplants=sum(r["tied_transplants"] for r in merge_rows),
                      continuations=merge_rows, J_release=float(final.objective[-1]))
        return final, record

    def continue_endpoint(self, max_sweeps, base, other, configuration_limit=4):
        """Continue one endpoint in its own coordinates and schedule.

        Independent exact-J improvements are accepted from one common base;
        tied interacting transplants receive bounded, equal continuation effort.
        Return every evaluated result so the caller ranks the complete endpoint
        set once, using the shared tie definition.
        """
        state, taken, tied, candidates = self.merge_end_points(base.state, other.state)
        items, tied_clusters = self.merge_tied
        combinations = 1 << len(tied_clusters)
        bounded_out = combinations > configuration_limit
        if bounded_out:
            combinations = 1
        results, rows = [], []
        for bits in range(combinations):
            seed = MapState(state.genotype.copy(), state.selector.copy(), state.frame.copy())
            for k, cluster in enumerate(tied_clusters):
                if (bits >> k) & 1:
                    _apply_items(seed, other.state, items, cluster[1])
            final = self.solve(max_sweeps, state=seed,
                               long_tier=bool(getattr(self.config, "map_long_moves", False)))
            results.append(final)
            rows.append(dict(reverse=self.reverse, clusters=candidates, transplanted=taken,
                             tied_transplants=tied, tied_configuration=bits,
                             combination_log2=len(tied_clusters), evaluated_configurations=combinations,
                             bounded_out=bool(bounded_out), J=float(final.objective[-1])))
        return results, rows

    def difference_clusters(self, items):
        """Difference runs connected through an actual J factor.

        Root marginal factors join all of a root's children. Selector transitions
        join the neighbouring *supported* markers, however long their gap. The
        returned clusters therefore have additive transplant gains: each can be
        evaluated from the same base, and all strict improvements applied together.
        A cluster transplants the two endpoints' joint configuration as one move.
        """
        g = self.graph
        footprints = []
        for kind, row, lo, hi, _ in items:
            keys = set()
            left, right = lo, hi
            if kind == "F":
                keys.add(("R", row))
                left, right = max(0, lo - 1), min(self.sites, hi + 1)
            elif kind == "S":
                keys.add(("C", row))
                keys.add(("T", row))
                for p in g.group_parent[row]:
                    if p >= 0 and self.roots[p]:
                        keys.add(("R", int(p)))
                markers = self.marker_sites[self.marker_offsets[row]:self.marker_offsets[row + 1]]
                a, b = np.searchsorted(markers, (lo, hi))
                if a > 0:
                    left = min(left, int(markers[a - 1]))
                if b < len(markers):
                    right = max(right, int(markers[b]) + 1)
            else:
                keys.add(("G", row))
                own = int(g.node_group[row])
                if own >= 0:
                    keys.add(("C", own))
                    for p in g.group_parent[own]:
                        if p >= 0 and self.roots[p]:
                            keys.add(("R", int(p)))
                for k in range(self.child_offsets[row], self.child_offsets[row + 1]):
                    keys.add(("C", int(self.child_groups[k])))
            footprints.append((kind, row, left, right, frozenset(keys)))
        groups = _difference_clusters(footprints, 0)
        return [(min(items[i][2] for i in members), members) for _, members in groups]

    def merge_end_points(self, base, other):
        """Joint endpoint transplants, independent under J, all scored from base.

        A tied transplant is kept in the current endpoint; solve_symmetric also
        continues the opposite endpoint whenever their total scores are tied.
        There is no best-first or leftmost/rightmost winning cluster.
        """
        state = MapState(base.genotype.copy(), base.selector.copy(), base.frame.copy())
        items = _difference_items(state, other, self.roots, self.graph)
        clusters = self.difference_clusters(items)
        gains = [self._transplant_gain(state, other, items, cluster) for cluster in clusters]
        taken = tied = 0
        tied_clusters = []
        for gain, cluster in zip(gains, clusters):
            comparison = ties.tie_classes([0.0, gain], self.quantum, descending=True)
            if comparison.tied_winner:
                tied += 1
                tied_clusters.append(cluster)
            elif comparison.classes[1] == 0:
                _apply_items(state, other, items, cluster[1])
                taken += 1
        self.merge_tied = (items, tied_clusters)
        return state, taken, tied, len(clusters)

    def _transplant_gain(self, state, other, items, cluster):
        """Exact change of J when ``cluster``'s cells take ``other``'s values (state restored)."""
        w, g = self.workspace, self.graph
        lo, members = cluster[0], cluster[1]
        hi = max(items[i][3] for i in members)
        root_mask = set()
        sample_mask = set()
        group_mask = set()
        transition_mask = set()
        end = hi
        for i in members:
            kind, row = items[i][0], items[i][1]
            if kind == "F":
                root_mask.add(int(row))
            elif kind == "S":
                group_mask.add(int(row))
                transition_mask.add(int(row))
                end = max(end, int(_window_end(row, hi, self.marker_offsets, self.marker_sites, self.sites)))
                for parent in g.group_parent[row]:
                    if parent >= 0 and self.roots[parent]:
                        root_mask.add(int(parent))
            else:
                sample_mask.add(int(row))
                own = g.node_group[row]
                if own >= 0:
                    group_mask.add(int(own))
                    for parent in g.group_parent[own]:
                        if parent >= 0 and self.roots[parent]:
                            root_mask.add(int(parent))
                for index in range(self.child_offsets[row], self.child_offsets[row + 1]):
                    group_mask.add(int(self.child_groups[index]))
        root_mask = np.asarray(sorted(root_mask), np.int64)
        sample_mask = np.asarray(sorted(x for x in sample_mask if not self.roots[x]), np.int64)
        group_mask = np.asarray(sorted(group_mask), np.int64)
        transition_mask = np.asarray(sorted(transition_mask), np.int64)
        end = min(self.sites, end + 1)
        args = (w.potentials, state.genotype, state.selector, state.frame, w.orientation, self.roots, w.frozen,
                self.free_starts, self.free_trans, self.frozen_starts, self.frozen_trans, g.group_parent,
                g.group_child, self.child_offsets, self.child_groups, self.child_slots, self.marker_offsets,
                self.marker_sites, self.marker_theta, self.log_keep, self.log_error, self.log_match, self.log_miss)
        before = _window_objective_lists(lo, end, root_mask, sample_mask, group_mask, transition_mask, *args)
        saved = _save_items(state, items, members)
        _apply_items(state, other, items, members)
        after = _window_objective_lists(lo, end, root_mask, sample_mask, group_mask, transition_mask, *args)
        _restore_items(state, items, members, saved)
        return after - before


# ------------------------------------------------------------------------------------------------ release

@njit(cache=True, parallel=True, nogil=True)
def _conditionals(potentials, genotype, selector, frame, orientation, roots, node_group, group_parent, group_child,
                  child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, cavities=None):
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
        root_scratch = np.empty(4)
        for site in range(sites):
            for g in range(4):
                row[g] = potentials[sample, site, g]
            _own_children_terms(sample, site, genotype, selector, group_child, child_offsets, child_groups, child_slots,
                                log_keep, log_error, row, cavities)
            _slot_tables(group, site, genotype, selector, frame, potentials, orientation, roots, group_parent,
                         group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                         log_miss, True, table, cavities, root_scratch)
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


@njit(cache=True, parallel=True, nogil=True)
def _frames_kernel(conditionals, frame, scaffold_orientation, roots, phase_theta, anchors, parent_counts, phase_error,
                   frozen, frozen_theta, reverse=False):
    """Family MAP frames int8 [N, L] (inferred, displayed) and, for non-roots, the released ambiguity bool [N, L]:
    roots release their frame chain; every other sample its MAP frame path over painting-bin runs with its per-site
    Gibbs conditionals as phase_view emissions (review I4; 2.0's phase_view rules, FRAME_TIE for unanchored chains).
    Displayed = inferred for two-parent samples, 0 else. Ambiguity (release 2.2): the sites where the two extreme tie
    placements of the chain (decoded forward and mirrored, maths.chains reverse) differ; the released path is the
    forward one (the mirrored one under the convention reversal). Also returns ``evidence`` float32 [N, L]: each
    non-root site's phase_view log-likelihood ratio log P(frame 1) - log P(frame 0) (0: no frame evidence)."""
    samples, sites, _ = conditionals.shape
    inferred = np.empty((samples, sites), dtype=np.int8)
    displayed = np.empty_like(inferred)
    undetermined = np.zeros((samples, sites), dtype=np.bool_)
    evidence = np.zeros((samples, sites), dtype=np.float32)
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
            evidence[sample, site] = math.log(emissions[site, 1]) - math.log(emissions[site, 0])
        compact, anchor_run = chains.contracted_emissions(emissions, starts, anchors[sample])
        final_tie = chains.FRAME_TIE if anchor_run < 0 else chains.CHAIN_TIE
        forward = chains.binary_viterbi(compact, trans, anchor_run, chains.CHAIN_TIE, final_tie, False)
        mirrored = chains.binary_viterbi(compact, trans, anchor_run, chains.CHAIN_TIE, final_tie, True)
        for run in range(len(starts) - 1):
            flip = mirrored[run] if reverse else forward[run]
            shown = flip if parent_counts[sample] >= 2 else 0
            tied = forward[run] != mirrored[run]
            for site in range(starts[run], starts[run + 1]):
                inferred[sample, site] = flip
                displayed[sample, site] = shown
                undetermined[sample, site] = tied
    return inferred, displayed, undetermined, evidence


@njit(cache=True, parallel=True, nogil=True)
def _root_undetermined(potentials, genotype, selector, frame, orientation, roots, frozen, free_starts, free_trans,
                       frozen_starts, frozen_trans, anchors, group_child, child_offsets, child_groups, child_slots,
                       log_keep, log_error, log_match, log_miss):
    """bool [N, L]: for every root, the sites where either extreme tie placement of its frame chain at the converged
    configuration (root block Viterbi forward and mirrored, _frame_path) differs from its released frame (release 2.2
    released ambiguity; e.g. a frame switch inside a stretch without scaffold-heterozygous sites of the root)."""
    samples, sites = genotype.shape
    out = np.zeros((samples, sites), dtype=np.bool_)
    for sample in prange(samples):
        if not roots[sample]:
            continue
        starts = frozen_starts if frozen[sample] else free_starts
        run_theta = frozen_trans if frozen[sample] else free_trans
        runs = len(starts) - 1
        run_value = np.zeros((runs, 2))
        anchor_run = -1
        for run in range(runs):
            if starts[run] <= anchors[sample] < starts[run + 1]:
                anchor_run = run
            for site in range(starts[run], starts[run + 1]):
                for f in range(2):
                    run_value[run, f] += _root_term(sample, site, f, genotype, selector, potentials, orientation,
                                                    group_child, child_offsets, child_groups, child_slots, log_keep,
                                                    log_error, log_match, log_miss)
        trans = np.empty(max(runs - 1, 0))
        for run in range(runs - 1):
            trans[run] = run_theta[run]
        left = _frame_path(run_value, trans, anchor_run, -1, 0.0, -1, 0.0, frame[sample, starts[0]],
                           frame[sample, starts[runs] - 1], False)
        right = _frame_path(run_value, trans, anchor_run, -1, 0.0, -1, 0.0, frame[sample, starts[0]],
                            frame[sample, starts[runs] - 1], True)
        for run in range(runs):
            f = frame[sample, starts[run]]
            if left[run] != f or right[run] != f:
                for site in range(starts[run], starts[run + 1]):
                    out[sample, site] = True
    return out


@njit(cache=True, nogil=True)
def _parent_cavity(parent, site, exclude_group, genotype, selector, frame, potentials, orientation, roots, node_group,
                   group_parent, group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                   log_miss, out, cavities=None):
    """out[g]: the log weight of ``parent`` holding genotype g at ``site`` without child group ``exclude_group``'s
    factor: a root's psi, phi and other children; a non-root's psi, its own slot terms and other children."""
    if roots[parent]:
        _root_cavity(parent, site, frame[parent, site], exclude_group, genotype, selector, potentials, orientation,
                     group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss,
                     out, cavities)
        return
    for g in range(4):
        out[g] = potentials[parent, site, g]
    own = node_group[parent]
    table = np.empty((2, 2, 2))
    _slot_tables(own, site, genotype, selector, frame, potentials, orientation, roots, group_parent, group_child,
                 child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, True, table, cavities)
    s = selector[own, site]
    for g in range(4):
        out[g] += table[0, _allele(g, 0), (s >> 1) & 1] + table[1, _allele(g, 1), s & 1]
    if cavities is not None and cavities[0][parent] >= 0:
        cavity_counts.add_terms(cavities, parent, site, exclude_group, genotype, selector, log_keep, log_error, out)
        return
    for index in range(child_offsets[parent], child_offsets[parent + 1]):
        group = child_groups[index]
        if group == exclude_group:
            continue
        slot = child_slots[index]
        child_allele = _allele(genotype[group_child[group], site], slot)
        homologue = (selector[group, site] >> (1 - slot)) & 1
        for g in range(4):
            out[g] += log_keep if _allele(g, homologue) == child_allele else log_error


@njit(cache=True, parallel=True, nogil=True)
def _repair_emissions(potentials, genotype, selector, frame, orientation, roots, node_group, group_parent, group_child,
                      child_offsets, child_groups, child_slots, log_keep, log_error, log_match, log_miss, cavities=None):
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
                               log_keep, log_error, log_match, log_miss, cavity, cavities)
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
                                log_keep, log_error, base, cavities)
            for s in range(4):
                for g in range(4):
                    terms[g] = base[g] + table[0, _allele(g, 0), (s >> 1) & 1] + table[1, _allele(g, 1), s & 1]
                values[s] = _lse4(terms[0], terms[1], terms[2], terms[3])
            top = max(max(values[0], values[1]), max(values[2], values[3]))
            for s in range(4):
                emissions[group, site, s] = math.exp(values[s] - top) if top > -np.inf else 1.0
    return emissions


@njit(cache=True, parallel=True, nogil=True)
def _conflicts(genotype, selector, frame, potentials, orientation, roots, group_child, child_offsets, child_groups,
               child_slots, log_keep, log_error, log_match, log_miss, reverse=False):
    """Mendelian-conflict diagnostic (review M4): bool [N, L] (parent, site) where the parent's best genotype (a
    root's mode of its summed terms, the lowest on an exact tie, the highest under the convention reversal; a
    non-root's MAP genotype) leaves at least one child's allele mistransmitted."""
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
                best = 3 if reverse else 0
                for index in range(1, 4):
                    g = 3 - index if reverse else index
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
    """(inferred, displayed) family MAP frames int8 [N, L] of the converged configuration (_frames_kernel), the
    released ambiguity bool [N, L] of the inferred frames (roots: _root_undetermined; others: _frames_kernel) and the
    per-site frame evidence float32 [N, L] of non-roots' phase_view emissions (log-likelihood ratio of frame 1)."""
    w, g = family_map.workspace, family_map.graph
    conditionals = _conditionals(w.potentials, state.genotype, state.selector, state.frame, w.orientation,
                                 family_map.roots, g.node_group, g.group_parent, g.group_child,
                                 family_map.child_offsets, family_map.child_groups, family_map.child_slots,
                                 family_map.log_keep, family_map.log_error, family_map.log_match, family_map.log_miss, family_map.cavities(state))
    inferred, displayed, undetermined, evidence = _frames_kernel(
        conditionals, state.frame, w.scaffold_orientation, family_map.roots, w.phase_theta, w.anchors,
        g.parent_counts, family_map.config.scaffold_phase_error, w.frozen, w.frozen_theta, family_map.reverse)
    del conditionals
    undetermined |= _root_undetermined(
        w.potentials, state.genotype, state.selector, state.frame, w.orientation, family_map.roots, w.frozen,
        family_map.free_starts, family_map.free_trans, family_map.frozen_starts, family_map.frozen_trans, w.anchors,
        g.group_child, family_map.child_offsets, family_map.child_groups, family_map.child_slots, family_map.log_keep,
        family_map.log_error, family_map.log_match, family_map.log_miss)
    return inferred, displayed, undetermined, evidence


def selector_emissions(family_map, state):
    """The root-frame repair's selector emissions float64 [G, L, 4] at the converged configuration (conditional
    cavities, review I3)."""
    w, g = family_map.workspace, family_map.graph
    return _repair_emissions(w.potentials, state.genotype, state.selector, state.frame, w.orientation, family_map.roots,
                             g.node_group, g.group_parent, g.group_child, family_map.child_offsets,
                             family_map.child_groups, family_map.child_slots, family_map.log_keep,
                             family_map.log_error, family_map.log_match, family_map.log_miss, family_map.cavities(state))


def conflict_sites(family_map, state):
    """Mendelian-conflict diagnostic: (parent, site) pairs (int64 [n, 2]) where the MAP needs a transmission error."""
    w = family_map.workspace
    mask = _conflicts(state.genotype, state.selector, state.frame, w.potentials, w.orientation, family_map.roots,
                      family_map.graph.group_child, family_map.child_offsets, family_map.child_groups,
                      family_map.child_slots, family_map.log_keep, family_map.log_error, family_map.log_match,
                      family_map.log_miss, family_map.reverse)
    return np.argwhere(mask).astype(np.int64)
