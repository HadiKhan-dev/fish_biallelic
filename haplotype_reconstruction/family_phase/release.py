"""Evidence for displayed phase-set joins; released point estimates stay unchanged.

Long disagreements between the existing MAP phase view and polish delimit phase
sets. Painting joins compare the two relative flank orientations with equal
family context and effort under unchanged J. These scores never flip calls.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
import time

import numpy as np
from numba import njit, prange

from ..execution import claim_threads, reserve_memory
from ..execution.budget import current_budget
from ..maths import ties
from . import map_solver as ms
from .solver import family_graph
from .work_bounds import BoundedSet, FamilyWorkBound, configured_bound, witness_family

REFIT_SWEEPS = 4


@dataclass(frozen=True, order=True)
class Candidate:
    sample: int
    start: int
    stop: int


@njit(cache=True, nogil=True)
def _constrained_block(group, lo, hi, constraint, potentials, genotype, selector, frame, orientation, roots,
                       group_parent, group_child, child_offsets, child_groups, child_slots, marker_offsets,
                       marker_sites, marker_theta, log_keep, log_error, log_match, log_miss, reverse, cavities=None):
    child = group_child[group]
    first, stop = marker_offsets[group], marker_offsets[group + 1]
    a = first + np.searchsorted(marker_sites[first:stop], lo)
    b = first + np.searchsorted(marker_sites[first:stop], hi)
    end = marker_sites[b] if b < stop else genotype.shape[1]
    n = end - lo
    if not n:
        return
    base, table = np.empty((n, 4)), np.empty((n, 2, 2, 2))
    ms._block_terms(group, np.int64(lo), end, potentials, genotype, selector, frame, orientation, roots, group_parent,
                    group_child, child_offsets, child_groups, child_slots, log_keep, log_error, log_match,
                    log_miss, base, table, cavities)
    for j in range(n):
        fixed = constraint[child, lo + j]
        if fixed >= 0:
            for k in range(4):
                if k != fixed:
                    base[j, k] = -np.inf
    left = selector[group, marker_sites[a - 1]] if a > first else -1
    right = selector[group, marker_sites[b]] if b < stop else -1
    new_s, new_g = np.empty(n, dtype=np.int8), np.empty(n, dtype=np.int8)
    ms._block_path(group, child, np.int64(lo), base, table, genotype, selector, marker_offsets, marker_sites, marker_theta,
                   a, b, np.int64(left), np.int64(right), new_s, new_g, np.bool_(reverse), np.bool_(True))
    if cavities is not None:
        old_g = genotype[child, lo:end].copy()
        old_s = selector[group, lo:end].copy()
    selector[group, lo:end] = new_s
    genotype[child, lo:end] = new_g

    if cavities is not None:
        ms.cavity_counts.update_group(cavities, group, np.int64(lo), old_g, old_s, genotype, selector, np.bool_(False))


@njit(cache=True, parallel=True, nogil=True)
def _joint_genotypes(parents, lo, hi, constraint, potentials, genotype, selector, frame, orientation, roots,
                     node_group, group_parent, group_child, child_offsets, child_groups, child_slots,
                     log_keep, log_error, log_match, log_miss, reverse):
    """Exact bounded sibship refits at conflicts, respecting display constraints.

    A shared root marginal couples siblings even when neither is a parent of the
    proposed sample. Independent child updates cannot cross that joint barrier.
    The factors and finite-state bound are the MAP site move's; J is unchanged.
    """
    for parent in parents:
        members, groups = ms._sibship(parent, roots, node_group, group_child, child_offsets, child_groups)
        if len(members) < 2:
            continue
        masks = ms._term_lists(groups, len(groups), parent if roots[parent] else -1, roots,
                               group_parent, group_child, child_offsets, child_groups, genotype.shape[0])
        rl, sl, gl = masks[:3]
        count = len(members)
        # Parent order is unchanged. Sites are independent conditional factors;
        # private scratch is reused within a bounded marker chunk, then joined
        # before the next parent sees this parent's updates.
        for block in prange((hi - lo + 255) // 256):
            options, sizes = np.zeros((count, 4), np.int8), np.zeros(count, np.int64)
            current, best = np.empty(count, np.int8), np.empty(count, np.int8)
            for j in range(lo + 256 * block, min(hi, lo + 256 * (block + 1))):
                if not ms._site_conflict(parent, j, genotype, selector, frame, potentials, orientation, roots,
                                         group_child, child_offsets, child_groups, child_slots, log_keep, log_error,
                                         log_match, log_miss):
                    continue
                combos = 1
                for q in range(count):
                    sample = members[q]
                    current[q] = best[q] = genotype[sample, j]
                    sizes[q] = 0
                    for k in range(4):
                        if (potentials[sample, j, k] > -np.inf
                                and (constraint[sample, j] < 0 or constraint[sample, j] == k)):
                            options[q, sizes[q]] = k
                            sizes[q] += 1
                    combos *= sizes[q]
                    if combos > ms.SITE_MAX_COMBOS:
                        break
                if combos > ms.SITE_MAX_COMBOS or combos <= 1:
                    continue
                value = ms._site_terms(j, rl, sl, gl, potentials, genotype, selector, frame, orientation, roots,
                                       group_parent, group_child, child_offsets, child_groups, child_slots,
                                       log_keep, log_error, log_match, log_miss)
                best_value = value
                for index in range(combos):
                    code = combos - 1 - index if reverse else index
                    for q in range(count):
                        genotype[members[q], j] = options[q, code % sizes[q]]
                        code //= sizes[q]
                    value = ms._site_terms(j, rl, sl, gl, potentials, genotype, selector, frame, orientation, roots,
                                           group_parent, group_child, child_offsets, child_groups, child_slots,
                                           log_keep, log_error, log_match, log_miss)
                    if value > best_value + ms.GENOTYPE_TIE:
                        best_value = value
                        for q in range(count):
                            best[q] = genotype[members[q], j]
                for q in range(count):
                    genotype[members[q], j] = best[q]

class FamilyProfile:
    """Reusable local work arrays; all profiles reset from an immutable common start."""

    def __init__(self, fm, state, scaffold, displayed, reverse, shared=None, undetermined=None):
        self.fm, self.base, self.scaffold, self.displayed = fm, state, scaffold, displayed
        self.reverse = bool(reverse)
        self.state = state  # immutable common start until a direct reference score needs scratch
        self.constraint = None
        self.layouts = {}
        if shared is not None:
            self.het, self.calls, self.families = shared.het, shared.calls, shared.families
            self.family_roots, self.joint_parents = shared.family_roots, shared.joint_parents
            self.family_rows, self.family_failures = shared.family_rows, shared.family_failures
            self.work_bound = shared.work_bound
            return
        self.het = np.all(scaffold >= 0, axis=2) & (scaffold[..., 0] != scaffold[..., 1])
        if undetermined is not None:
            self.het &= ~undetermined
        self.calls = np.take_along_axis(scaffold, np.stack((displayed, 1 ^ displayed), axis=-1), axis=-1)
        self.families, self.family_roots, self.joint_parents = {}, {}, {}
        self.family_rows, self.family_failures = {}, set()
        self.work_bound = configured_bound()

    def _family(self, sample):
        if sample in self.family_failures:
            raise FamilyWorkBound(self.work_bound)
        if sample not in self.families:
            try:
                groups, roots, parents, rows = witness_family(self.fm, sample, self.work_bound)
            except FamilyWorkBound:
                self.family_failures.add(sample)
                raise
            self.families[sample], self.family_roots[sample] = groups, roots
            self.joint_parents[sample], self.family_rows[sample] = parents, rows


    def scope(self, proposals):
        fm, g = self.fm, self.fm.graph
        for p in proposals:
            self._family(p.sample)
        BoundedSet((x for p in proposals for x in self.family_rows[p.sample]), self.work_bound)
        groups = np.array(sorted({x for p in proposals for x in self.families[p.sample]}), dtype=np.int64)
        a, b = min(p.start for p in proposals), max(p.stop for p in proposals)
        padding = max(ms.COMPOUND_MARGIN, min(b - a, ms.COMPOUND_WIDTH))
        lo, hi = max(0, a - padding), min(fm.sites, b + padding)
        roots = np.array(sorted({x for p in proposals for x in self.family_roots[p.sample]}), dtype=np.int64)
        parents = np.array(sorted({x for p in proposals for x in self.joint_parents[p.sample]}), dtype=np.int64)
        end = max(int(ms._window_end(int(x), hi, fm.marker_offsets, fm.marker_sites, fm.sites)) for x in groups)
        moving_roots = []
        for root in roots:
            starts = fm.frozen_starts if fm.workspace.frozen[root] else fm.free_starts
            a, b = np.searchsorted(starts[:-1], (lo, hi))
            anchor = np.searchsorted(starts, fm.workspace.anchors[root], side='right') - 1
            if b > a and not (b == a + 1 and a == anchor):
                moving_roots.append(int(root))
                end = max(end, int(starts[b]))
        roots = np.asarray(moving_roots, dtype=np.int64)
        # Sparse factor indices: no N-sized mask per witness.
        sample_terms = {int(g.group_child[x]) for x in groups}
        root_terms = {int(p) for x in groups for p in g.group_parent[x] if p >= 0 and fm.roots[p]}
        copy_terms = set(map(int, groups))
        copy_terms.update(int(x) for child in sample_terms
                          for x in fm.child_groups[fm.child_offsets[child]:fm.child_offsets[child + 1]])
        masks = tuple(np.asarray(sorted(x), dtype=np.int64)
                      for x in (root_terms, sample_terms, copy_terms, set(map(int, groups))))
        # All affected site factors, including each shared root marginal. The
        # previous/next supported markers bound the selector transition factors.
        left = lo
        for x in groups:
            markers = fm.marker_sites[fm.marker_offsets[x]:fm.marker_offsets[x + 1]]
            k = np.searchsorted(markers, lo)
            if k:
                left = min(left, int(markers[k - 1]))
        keys = frozenset((kind, int(i)) for kind, mask in zip(('R', 'G', 'C', 'T'), masks)
                         for i in mask)
        return groups, lo, hi, end, masks, roots, parents, (left, min(fm.sites, end + 1), keys)

    def score(self, proposals, flips, scope=None, *, first_joint=None, capture_first=False):
        if self.constraint is None:
            self.state = ms.MapState(self.base.genotype.copy(), self.base.selector.copy(), self.base.frame.copy())
            self.constraint = np.full(self.base.genotype.shape, -1, dtype=np.int8)
        fm, w, g, state = self.fm, self.fm.workspace, self.fm.graph, self.state
        groups, lo, hi, end, masks, roots, parents, _ = self.scope(proposals) if scope is None else scope
        masks = tuple(_indices(x) for x in masks)
        children = g.group_child[groups]
        state.genotype[children, lo:end] = self.base.genotype[children, lo:end]
        state.selector[groups, lo:end] = self.base.selector[groups, lo:end]
        state.frame[roots, lo:end] = self.base.frame[roots, lo:end]
        # Each sample's complete window is constrained to the common display,
        # with XOR composition for overlapping interval proposals.
        samples = sorted({p.sample for p in proposals})
        for sample in samples:
            desired = (2 * self.calls[sample, lo:end, 0] + self.calls[sample, lo:end, 1]).copy()
            flip = np.zeros(end - lo, dtype=np.bool_)
            for p, use in zip(proposals, flips):
                if use and p.sample == sample:
                    flip[p.start - lo:p.stop - lo] ^= True
            desired[flip] = ((desired[flip] & 1) << 1) | ((desired[flip] >> 1) & 1)
            fixed = self.het[sample, lo:end]
            self.constraint[sample, lo:end] = np.where(fixed, desired, -1)
            state.genotype[sample, lo:end][fixed] = desired[fixed]
        common = (self.constraint, w.potentials, state.genotype, state.selector, state.frame, w.orientation,
                  fm.roots, g.group_parent, g.group_child, fm.child_offsets, fm.child_groups, fm.child_slots,
                  fm.marker_offsets, fm.marker_sites, fm.marker_theta, fm.log_keep, fm.log_error,
                  fm.log_match, fm.log_miss)
        args = (w.potentials, state.genotype, state.selector, state.frame, w.orientation, fm.roots, w.frozen,
                fm.free_starts, fm.free_trans, fm.frozen_starts, fm.frozen_trans, g.group_parent, g.group_child,
                fm.child_offsets, fm.child_groups, fm.child_slots, fm.marker_offsets, fm.marker_sites,
                fm.marker_theta, fm.log_keep, fm.log_error, fm.log_match, fm.log_miss)
        joint_args = (self.constraint, w.potentials, state.genotype, state.selector, state.frame, w.orientation,
                      fm.roots, g.node_group, g.group_parent, g.group_child, fm.child_offsets, fm.child_groups,
                      fm.child_slots, fm.log_keep, fm.log_error, fm.log_match, fm.log_miss, self.reverse)
        previous = ms._window_objective_lists(lo, min(fm.sites, end + 1), *masks, *args)
        score = previous
        cavities = None
        for sweep in range(REFIT_SWEEPS):
            claim_threads()
            if sweep == 0 and first_joint is not None:
                # Only the first joint pass is column-local. The two complete
                # parent-ordered passes were made from this same common start.
                state.genotype[:, lo:end] = first_joint[0]
                for proposal, use in zip(proposals, flips):
                    if use:
                        a, b = proposal.start, proposal.stop
                        state.genotype[:, a:b] = first_joint[1][:, a - lo:b - lo]
            else:
                _joint_genotypes(parents, lo, end, *joint_args)
            if capture_first:
                # Used on a private, disposable profile; no later refit is run.
                return state.genotype[:, lo:end].copy()
            if fm.cavity_mode == 'counts':
                if cavities is None:
                    cavities = fm.cavities(state)
                else:
                    ms.cavity_counts.refresh(cavities, state.genotype, state.selector, fm.child_offsets,
                                             fm.child_groups, fm.child_slots, None)
            order = groups[::-1] if self.reverse ^ bool(sweep % 2) else groups
            for group in order:
                claim_threads()
                _constrained_block(int(group), lo, hi, *common, self.reverse, cavities)
            for root in roots:
                starts, trans = ((fm.frozen_starts, fm.frozen_trans) if w.frozen[root]
                                  else (fm.free_starts, fm.free_trans))
                ms._window_root(int(root), lo, hi, w.potentials, state.genotype, state.selector, state.frame,
                                w.orientation, starts, trans, int(w.anchors[root]), g.group_child, fm.child_offsets,
                                fm.child_groups, fm.child_slots, fm.log_keep, fm.log_error, fm.log_match,
                                fm.log_miss, self.reverse)
            score = ms._window_objective_lists(lo, min(fm.sites, end + 1), *masks, *args)
            if score < previous - fm.quantum:
                raise RuntimeError(f"family release refit decreased J: {score - previous}")
            if score <= previous + fm.quantum:
                break
            previous = score
        for sample in samples:
            self.constraint[sample, lo:end] = -1
        # Restore immediately: a later, overlapping scope must see the common
        # start also outside its own reset window.
        state.genotype[children, lo:end] = self.base.genotype[children, lo:end]
        state.selector[groups, lo:end] = self.base.selector[groups, lo:end]
        state.frame[roots, lo:end] = self.base.frame[roots, lo:end]
        return float(score)


def _local_bounds(profile, scope):
    """Include every external state read by the original window scorer/decoder.

    In addition to the refit window, the objective reads a preceding and following
    supported selector marker and the root transition at its right endpoint.
    No selector reset or new evidence boundary is introduced by this crop.
    """
    fm = profile.fm
    groups, lo, hi, end, masks, roots, parents, footprint = scope
    score_hi = min(fm.sites, end + 1)
    left, right = max(0, lo - 1), min(fm.sites, score_hi + 1)
    for group in _indices(masks[3]):
        markers = fm.marker_sites[fm.marker_offsets[group]:fm.marker_offsets[group + 1]]
        a, b = np.searchsorted(markers, (lo, score_hi))
        if a:
            left = min(left, int(markers[a - 1]))
        if b < len(markers):
            right = max(right, int(markers[b]) + 1)
    return left, right


def _indices(mask):
    return np.flatnonzero(mask) if mask.dtype == np.bool_ else mask


def _mask(indices, size):
    result = np.zeros(size, dtype=np.bool_)
    result[indices] = True
    return result


def _factor_layout(profile, scope):
    """Complete read-factor neighbourhood; boundary nodes keep their original root status.

    Only groups actually read are retained. Every updated child's outgoing copy
    factors and every scored root's full marginal are included, without recursively
    pulling in irrelevant ancestors of pinned boundary genotypes.
    """
    fm, graph = profile.fm, profile.fm.graph
    groups, _, _, _, masks, roots, parents, _ = scope
    ids = tuple(tuple(map(int, _indices(x))) for x in masks)
    key = (tuple(map(int, groups)), ids, tuple(map(int, parents)))
    cached = profile.layouts.get(key)
    if cached is not None:
        return cached
    needed_groups = set(ids[2]) | set(ids[3]) | set(map(int, groups))
    for root in ids[0]:
        needed_groups.update(map(int, fm.child_groups[fm.child_offsets[root]:fm.child_offsets[root + 1]]))
    # Joint refits read the complete sibship, the parent's own group, and the
    # outgoing factors of every member. Scope construction already includes its
    # mutable members; these additions also preserve their pinned boundary reads.
    for parent in parents:
        needed_groups.update(map(int, fm.child_groups[fm.child_offsets[parent]:fm.child_offsets[parent + 1]]))
        if not fm.roots[parent]:
            needed_groups.add(int(graph.node_group[parent]))
    rows = set(ids[0]) | set(ids[1]) | set(map(int, parents))
    for group in needed_groups:
        rows.add(int(graph.group_child[group]))
        rows.update(int(p) for p in graph.group_parent[group] if p >= 0)
    samples = np.asarray(sorted(rows), dtype=np.int64)
    group_ids = np.asarray(sorted(needed_groups), dtype=np.int64)
    sm = {int(old): i for i, old in enumerate(samples)}
    gm = {int(old): i for i, old in enumerate(group_ids)}
    pp, cc, ss = [], [], []
    for old in group_ids:
        for slot, parent in enumerate(graph.group_parent[old]):
            if parent >= 0:
                pp.append(sm[int(parent)]); cc.append(sm[int(graph.group_child[old])]); ss.append(slot)
    compact = family_graph(pp, cc, ss, len(samples))
    compact = replace(compact, roots=graph.roots[samples], parent_counts=graph.parent_counts[samples])
    children = ms._children_of(compact, len(samples))
    cached = (samples, group_ids, sm, gm, compact, children)
    profile.layouts[key] = cached
    return cached


def _local_profile(profile, proposals, scope):
    """Exact family/factor and marker view, preserving original arithmetic order."""
    left, right = _local_bounds(profile, scope)
    samples, group_ids, sm, gm, graph, children = _factor_layout(profile, scope)
    fm = copy(profile.fm)
    fm.samples, fm.sites, fm.graph = len(samples), right - left, graph
    fm.roots = graph.roots
    fm.child_offsets, fm.child_groups, fm.child_slots = children
    fm.workspace = copy(profile.fm.workspace)
    w = fm.workspace
    w.graph = graph
    w.potentials = w.potentials[samples, left:right]
    w.orientation = w.orientation[samples, left:right]
    w.anchors = w.anchors[samples] - left
    w.frozen = w.frozen[samples]
    for prefix in ('free', 'frozen'):
        starts = getattr(profile.fm, prefix + '_starts')
        trans = getattr(profile.fm, prefix + '_trans')
        indices = np.flatnonzero((starts > left) & (starts < right))
        setattr(fm, prefix + '_starts', np.r_[0, starts[indices] - left, right - left].astype(np.int64))
        setattr(fm, prefix + '_trans', trans[indices - 1])
    sites, thetas, offsets = [], [], [0]
    for group in group_ids:
        a, b = profile.fm.marker_offsets[group:group + 2]
        markers = profile.fm.marker_sites[a:b]
        lo, hi = np.searchsorted(markers, (left, right))
        sites.append(markers[lo:hi] - left)
        thetas.append(profile.fm.marker_theta[a + lo:a + hi])
        offsets.append(offsets[-1] + hi - lo)
    fm.marker_offsets = np.asarray(offsets, dtype=np.int64)
    fm.marker_sites = np.concatenate(sites) if sites else np.empty(0, dtype=np.int64)
    fm.marker_theta = np.concatenate(thetas) if thetas else np.empty(0, dtype=np.float64)
    local = copy(profile)
    local.fm = fm
    local.base = ms.MapState(profile.base.genotype[samples, left:right],
                            profile.base.selector[group_ids, left:right],
                            profile.base.frame[samples, left:right])
    local.state, local.constraint = local.base, None
    local.calls = profile.calls[samples, left:right]
    local.het = profile.het[samples, left:right]
    local.scaffold = profile.scaffold[samples, left:right]
    local.displayed = profile.displayed[samples, left:right]
    groups, lo, hi, end, masks, roots, parents, footprint = scope
    masks = tuple(np.asarray([(sm if k < 2 else gm)[int(x)] for x in _indices(mask)], np.int64)
                  for k, mask in enumerate(masks))
    scope = (np.asarray([gm[int(x)] for x in groups], np.int64), lo - left, hi - left, end - left, masks,
             np.asarray([sm[int(x)] for x in roots], np.int64),
             np.asarray([sm[int(x)] for x in parents], np.int64),
             (footprint[0] - left, footprint[1] - left, footprint[2]))
    proposals = tuple(Candidate(sm[p.sample], p.start - left, p.stop - left) for p in proposals)
    return local, proposals, scope


def _share_first_pass(proposals, configs):
    """Four configurations of two disjoint flanks on one sample, unordered.

    Every column's complete first-pass input equals either 00 or 11. Different
    samples, overlaps, or other configuration sets use the reference path.
    """
    return (len(proposals) == 2 and proposals[0].sample == proposals[1].sample
            and (proposals[0].stop <= proposals[1].start
                 or proposals[1].stop <= proposals[0].start)
            and len(configs) == 4
            and set(map(tuple, configs)) == {(False, False), (False, True),
                                           (True, False), (True, True)})


def _score_batch(profile, jobs):
    """Independent, equal-effort profiles under the execution layer's live budget.

    Scratch scales with the actual window, never with a chromosome per worker.
    Scoring jobs keep input order; the four configurations use the same local
    start, outside messages, refits and arithmetic as the reference implementation.
    """
    if not jobs:
        return []
    # Compact immutable likelihood rows are copies, not NumPy views. Charge
    # them once while at least one configuration uses the template; charge each
    # mutable state separately. Retire idle templates immediately so their arrays
    # cannot survive their scratch credits while later configurations wait.
    import threading
    changed = threading.Condition()
    templates = [None] * len(jobs)
    users = [0] * len(jobs)
    sharing = [_share_first_pass(proposals, configs) for proposals, configs, _ in jobs]
    shared_bytes, private_bytes = [], []
    for index, (_, _, scope) in enumerate(jobs):
        a, b = _local_bounds(profile, scope)
        samples, group_ids, _, _, _, local_children = _factor_layout(profile, scope)
        marker_count = 0
        for group in group_ids:
            x, y = profile.fm.marker_offsets[group:group + 2]
            first, last = np.searchsorted(profile.fm.marker_sites[x:y], (a, b))
            marker_count += int(last - first)
        # Immutable row copies and marker construction temporaries; private
        # G/F/constraint/S, block tables, paths and bounded site configurations.
        shared_bytes.append(int((b - a) * ((48 + 2 * sharing[index]) * len(samples) + len(group_ids))
                                + 32 * marker_count + 128 * (len(samples) + len(group_ids))))
        cached_parents = (int(np.count_nonzero(np.diff(local_children[0]) >= 2))
                          if profile.fm.cavity_mode == 'counts' else 0)
        private_bytes.append(int((b - a) * (3 * len(samples) + len(group_ids) + 256
                                           + 16 * cached_parents) + 4096))
    # A shared job owns two cached family windows and up to four private
    # configurations. Its numerical phases share the job's live CPU budget;
    # preparation runs outside the template lock.
    private_bytes = [size * (4 if share else 1) for size, share in zip(private_bytes, sharing)]
    tasks = [(index, None) for index in range(len(jobs)) if sharing[index]]
    tasks += [(index, bits) for index, job in enumerate(jobs)
              if not sharing[index] for bits in job[1]]
    if not tasks:
        return [[] for _ in jobs]
    largest = max(shared_bytes[i] + private_bytes[i] for i, _ in tasks)
    # One extra chromosome-sized mutable state (G/F/constraint/S), or the
    # largest family/window with all its configurations if larger. This is a storage budget,
    # not an inference bound: every configuration receives the original effort.
    target = max(max(shared_bytes[i] + (1 if sharing[i] else len(job[1])) * private_bytes[i]
                     for i, job in enumerate(jobs)),
                 profile.base.genotype.nbytes * 3 + profile.base.selector.nbytes)
    diagnostics = dict(target_bytes=target, minimum_bytes=largest, granted_bytes=0,
                       peak_charged_bytes=0, waits=0, wait_seconds=0.0, template_builds=0,
                       jobs=len(jobs), configurations=sum(len(job[1]) for job in jobs),
                       shared_first_pass_jobs=sum(sharing))
    occupied = 0
    error = None
    with reserve_memory(target, minimum=largest, envelope=True,
                        label='family orientation scratch') as scratch:
        diagnostics['granted_bytes'] = scratch.nbytes
        def evaluate(task):
            nonlocal occupied, error
            index, bits = task
            with changed:
                began = None
                while True:
                    if error is not None:
                        raise error
                    needed = private_bytes[index] + (shared_bytes[index] if users[index] == 0 else 0)
                    if occupied + needed <= scratch.nbytes:
                        break
                    if began is None:
                        began = time.perf_counter()
                        diagnostics['waits'] += 1
                    changed.wait()
                if began is not None:
                    diagnostics['wait_seconds'] += time.perf_counter() - began
                occupied += needed
                diagnostics['peak_charged_bytes'] = max(diagnostics['peak_charged_bytes'], occupied)
                users[index] += 1
                try:
                    if templates[index] is None:
                        proposals, _, scope = jobs[index]
                        templates[index] = _local_profile(profile, proposals, scope)
                        diagnostics['template_builds'] += 1
                    template, proposals, scope = templates[index]
                except BaseException as exc:
                    error = exc
                    changed.notify_all()
                    raise
            local = copy(template)
            try:
                if bits is not None:
                    return local.score(proposals, bits, scope)
                def pass_task(orientation):
                    def run(budget):
                        return copy(template).score(proposals, (orientation, orientation), scope,
                                                    capture_first=True)
                    return run
                first = [None, None]
                for k, value in current_budget().concurrent(
                        [pass_task(False), pass_task(True)], 0, label='family shared first passes'):
                    first[k] = value
                def score_task(config):
                    def run(budget):
                        return copy(template).score(proposals, config, scope, first_joint=first)
                    return run
                scores = [None] * len(jobs[index][1])
                for k, value in current_budget().concurrent(
                        [score_task(config) for config in jobs[index][1]], 0,
                        label='family shared configurations'):
                    scores[k] = value
                del first
                return scores
            finally:
                # Drop owned references before making their credits reusable.
                del local, template, proposals, scope
                with changed:
                    occupied -= private_bytes[index]
                    users[index] -= 1
                    if users[index] == 0:
                        templates[index] = None
                        occupied -= shared_bytes[index]
                    changed.notify_all()
        # The enclosing reservation covers every admitted array. CPU workers
        # retain the production live allowance at each numerical kernel boundary.
        # Each long configuration can use free cores inside marker-local terms;
        # shorter independent configurations still share the live CPU budget.
        def function(task):
            def run(budget):
                claim_threads()
                return evaluate(task)
            return run
        values = [None] * len(tasks)
        # Admit scratch before CPU leases. A task waiting for a family template
        # must not strand a numerical core while an admitted job could use it.
        # Each batch fits even if every private state is live simultaneously.
        start = 0
        while start < len(tasks):
            stop, charged, present = start, 0, set()
            while stop < len(tasks):
                index, _ = tasks[stop]
                extra = private_bytes[index] + (0 if index in present else shared_bytes[index])
                if charged + extra > scratch.nbytes:
                    break
                charged += extra
                present.add(index)
                stop += 1
            assert stop > start
            for offset, value in current_budget().concurrent(
                    [function(task) for task in tasks[start:stop]], 0,
                    label='family orientation configurations'):
                values[start + offset] = value
            start = stop
    profile.scratch_diagnostics = diagnostics
    out = [[] for _ in jobs]
    for (index, bits), value in zip(tasks, values):
        if bits is None:
            out[index] = value
        else:
            out[index].append(value)
    return out



def disagreement_breaks(fm, scaffold, displayed, family_displayed, components):
    """Unordered long MAP/polish disagreements, bounded by assembly components.

    A whole-component strand swap changes no relative phase within the component.
    Reversing the order of these two model outputs gives the same difference set.
    The existing long-tract span is a scale definition, not a confidence margin.
    """
    breaks = {i: set() for i in range(fm.samples)}
    runs = set()
    cuts = np.r_[0, np.flatnonzero(components[1:] != components[:-1]) + 1, fm.sites]
    for sample in np.flatnonzero(fm.graph.parent_counts == 2):
        sample = int(sample)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            lo, hi = int(lo), int(hi)
            if components[lo] < 0:
                continue
            different = displayed[sample, lo:hi] != family_displayed[sample, lo:hi]
            observable = scaffold[sample, lo:hi, 0] != scaffold[sample, lo:hi, 1]
            # Switch locations inside homozygous/fully missing stretches do not
            # identify phase. A strand swap of every observable ordered call is
            # still a whole-component swap even if raw frame XOR has gaps.
            if np.all(different[observable]):
                continue
            for a, b in ms._runs_of(different):
                a, b = a + lo, b + lo
                called = scaffold[sample, a:b]
                if (b - a <= ms.LONG_SPAN or not np.any(np.all(called >= 0, axis=-1)
                                                       & (called[:, 0] != called[:, 1]))):
                    continue
                runs.add((sample, a, b))
                for cut in (a, b):
                    if lo < cut < hi:
                        breaks[sample].add(cut - 1)
    return breaks, sorted(runs)


def phase_sets(fm, state, scaffold, displayed, family_displayed, components, painting_ties, *, reverse,
               undetermined=None):
    """(display phase breaks, evidence record); no point-estimate mutation.

    J and polish remain distinct estimators. Their long disagreements become
    display breaks. Each painting join separately requires a family witness:
    both flanks supported and their relative orientation uniquely supported.
    """
    started = time.perf_counter()
    breaks, disagreements = disagreement_breaks(fm, scaffold, displayed, family_displayed, components)
    profile = FamilyProfile(fm, state, scaffold, displayed, reverse, undetermined=undetermined)
    joins, join_jobs = [], []
    cuts = np.r_[0, np.flatnonzero(components[1:] != components[:-1]) + 1, fm.sites]
    for sample in np.flatnonzero(fm.graph.parent_counts == 2):
        sample = int(sample)
        own = np.asarray(painting_ties.get(sample, ()), dtype=np.int64)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if components[lo] < 0:
                continue
            inner = own[(own >= lo) & (own < hi - 1)]
            bounds = np.r_[lo, inner + 1, hi]
            for k, boundary in enumerate(inner):
                left = Candidate(sample, int(bounds[k]), int(bounds[k + 1]))
                right = Candidate(sample, int(bounds[k + 1]), int(bounds[k + 2]))
                has_left = bool(np.any(profile.het[sample, left.start:left.stop]))
                has_right = bool(np.any(profile.het[sample, right.start:right.stop]))
                job = None
                bounded = False
                if has_left and has_right:
                    try:
                        scope = profile.scope((left, right))
                    except FamilyWorkBound:
                        bounded = True
                    else:
                        job = len(join_jobs)
                        join_jobs.append(((left, right), ((False, False), (False, True), (True, False), (True, True)), scope))
                joins.append(dict(sample=sample, boundary=int(boundary), joined=False, relative_margin=None,
                                  left_margin=None, right_margin=None, left_observed=has_left,
                                  right_observed=has_right, scores=[], job=job, start=left.start, stop=right.stop,
                                  computational_bound=bounded, family_work_bound=profile.work_bound))
    join_scores = _score_batch(profile, join_jobs)
    for row in joins:
        job = row.pop('job')
        start, stop = row.pop('start'), row.pop('stop')
        if job is not None:
            values = join_scores[job]
            row.update(relative_witness(values, fm.quantum))
        if not row['joined']:
            breaks[row['sample']].add(row['boundary'])
    record = dict(policy="map_polish_disagreement", disagreements=disagreements,
                  long_span=ms.LONG_SPAN, refit_sweeps=REFIT_SWEEPS,
                  painting_joins=joins, computational_bound=sum(x['computational_bound'] for x in joins),
                  family_work_bound=profile.work_bound, seconds=time.perf_counter() - started)
    return breaks, record


def relative_witness(values, quantum):
    """Four equal-effort orientations: current relative order and both flanks supported."""
    same, opposite = max(values[0], values[3]), max(values[1], values[2])
    relative = ties.tie_classes([same, opposite], quantum, descending=True)
    left_values = [max(values[0], values[1]), max(values[2], values[3])]
    right_values = [max(values[0], values[2]), max(values[1], values[3])]
    left_rank = ties.tie_classes(left_values, quantum, descending=True)
    right_rank = ties.tie_classes(right_values, quantum, descending=True)
    return dict(relative_margin=same - opposite, left_margin=left_values[0] - left_values[1],
                right_margin=right_values[0] - right_values[1], scores=values,
                joined=bool(not relative.tied_winner and relative.classes[0] == 0
                            and not left_rank.tied_winner and not right_rank.tied_winner))


def uncertain_gaps(components, undetermined, breaks):
    """Supported flank pieces around each displayed-uncertain run inside an existing PS.

    Runs at a phase-set end have no two-flank continuity claim. The witness uses
    only the adjacent determined pieces, never another uncertain stretch.
    """
    sites = len(components)
    component_cuts = np.flatnonzero(components[1:] != components[:-1]) + 1
    gaps = []
    for sample in range(undetermined.shape[0]):
        bounds = np.unique(np.r_[0, component_cuts, np.asarray(sorted(breaks[sample]), dtype=np.int64) + 1, sites])
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if components[lo] < 0:
                continue
            runs = [(a + lo, b + lo) for a, b in ms._runs_of(undetermined[sample, lo:hi])]
            for k, (a, b) in enumerate(runs):
                if a == lo or b == hi:
                    continue
                start = runs[k - 1][1] if k else lo
                stop = runs[k + 1][0] if k + 1 < len(runs) else hi
                gaps.append((int(sample), int(start), int(a), int(b), int(stop)))
    return gaps


def gap_witnesses(fm, state, scaffold, displayed, undetermined, gaps, repair, one_parent_evidence):
    """Witnesses for the final union mask, including ambiguity projected from other modes.

    Two-parent fits leave all displayed-undetermined target cells unconstrained.
    Roots reuse accepted repair votes; one-parent samples reuse their per-site
    orientation evidence. No point estimate or decoder state is changed.
    """
    from ..maths import chains
    if not gaps:
        return []
    profile = None
    supported_het = np.all(scaffold >= 0, axis=2) & (scaffold[..., 0] != scaffold[..., 1]) & ~undetermined
    rows, jobs = [], []
    decisions = (*repair.moves, *repair.reverts, *repair.transfers)
    positions = fm.workspace.positions
    for sample, lo, a, b, hi in gaps:
        row = dict(sample=sample, start=a, stop=b, boundary=b - 1, joined=False,
                   relative_margin=None, left_margin=None, right_margin=None, scores=[], kind='unsupported')
        left = np.flatnonzero(supported_het[sample, lo:a]) + lo
        right = np.flatnonzero(supported_het[sample, b:hi]) + b
        row['left_observed'], row['right_observed'] = bool(len(left)), bool(len(right))
        row['relative_frame'] = int(displayed[sample, left[-1]] ^ displayed[sample, right[0]]) if len(left) and len(right) else None
        job = None
        if len(left) and len(right):
            if fm.graph.parent_counts[sample] == 2:
                if profile is None:
                    profile = FamilyProfile(fm, state, scaffold, displayed, fm.reverse, undetermined=undetermined)
                proposals = (Candidate(sample, lo, a), Candidate(sample, b, hi))
                try:
                    scope = profile.scope(proposals)
                except FamilyWorkBound:
                    row.update(kind='computational_bound', family_work_bound=profile.work_bound)
                else:
                    job = len(jobs)
                    jobs.append((proposals, ((False, False), (False, True), (True, False), (True, True)), scope))
                    row['kind'] = 'family_profile'
            elif fm.graph.parent_counts[sample] == 1:
                evidence = one_parent_evidence[sample]
                # Compare evidence with the actually released orientation at each
                # supported marker. The sign correction is the existing frame rule.
                lmargin = float(np.sum(np.where(displayed[sample, left] == 1, evidence[left], -evidence[left]), dtype=np.float64))
                rmargin = float(np.sum(np.where(displayed[sample, right] == 1, evidence[right], -evidence[right]), dtype=np.float64))
                row.update(kind='one_parent_evidence', left_margin=lmargin, right_margin=rmargin,
                           joined=bool(lmargin > chains.CHAIN_TIE and rmargin > chains.CHAIN_TIE))
            else:
                # Every uncertain cell must be covered by accepted repair decisions;
                # projected uncertainty without such a vote has no witness.
                own = [d for d in decisions if d.sample == sample and d.interval_left_bp < positions[b]
                       and d.interval_right_bp > positions[a - 1]]
                covered = np.zeros(b - a, dtype=np.bool_)
                for d in own:
                    covered |= (positions[a:b] > d.interval_left_bp) & (positions[a:b] < d.interval_right_bp)
                expected = 0 if repair.display is None else int(repair.display[sample, left[-1]] ^ repair.display[sample, right[0]])
                row.update(kind='root_repair', joined=bool(own and np.all(covered) and row['relative_frame'] == expected),
                           repair_witnesses=[dict(kind=d.kind, boundary=d.boundary, prior=d.prior, total=d.total,
                                                  child_gains=d.child_gains, child_net_gains=d.child_net_gains) for d in own])
        row['job'] = job
        rows.append(row)
    values = _score_batch(profile, jobs) if jobs else []
    for row in rows:
        job = row.pop('job')
        if job is not None:
            row.update(relative_witness(values[job], fm.quantum))
    return rows
