"""Windowed root-frame repair on the final family state.

The scaffold frame of a pedigree root (the track order of its painting, used by the polish and the displayed
phase) can be wrong over the rest of a component: panel rows that exchange founder identity, painting
abstentions or ties between interchangeable rows, segregating deletions. Every child of the root then changes
homologue near the error. Where the root's own data cannot place that change, the children's changes spread
over a stretch, so beyond the reach of the family MAP's relabelling and compound moves (map_solver.py) the MAP
keeps the scaffold frame. The MAP can also change a root's internal frame (the family MAP frame, which
recombination decodes) where the scaffold is right.

Votes. For root P, a bin edge e inside a component and each child edge of P, the child's joint selector chain
(selector emissions from conditional cavities at the family MAP, map_solver.selector_emissions, and the
selector transitions) is re-decoded with free ends after
relabelling P's homologues from e on:
  - exact: the change of the max-path log score on the window +-W around e; at a flagged edge (where the root's
    own data do not carry the scaffold orientation, ``orientation_flags``) also the best such change at any
    boundary of the flagged stretch around e ("inside");
  - net: the same change with the evidence inside a location zone left out and the zone crossed by one
    transition priced like e, so only the flanks (W - tolerance wide) count: does the child's homologue
    differ between the flanks? The zone is e +- ``root_frame_location_bp``, or the flagged stretch widened by
    that tolerance.
A child supports a change near e if it changes homologue there (exact, or inside at a flagged edge, >= tau) and
its flanks do not deny a net change (net > -tau); it opposes if its flanks show no net change (net <= -tau).

Move (internal frame and scaffold together). Where P's family MAP frame agrees with its scaffold over the whole
net window, both are relabelled from e to the end of the component if at least ``root_frame_minimum_children``
children support it and none opposes (a frame error moves every child, all-or-none, while real crossovers and
reference-order junctions switch children binomially), the total at e (the prior log-odds of a scaffold frame
change, 0 at flagged edges, plus every child's exact gain) is positive, no other parent with that many children
of its own is mixed inside the two painting bins next to e (junction guard; children shared with P are left
out), and e lies outside the protected intervals. Rounds repeat on the relabelled chains.

Belief-propagation frame changes. At a change e of P's family MAP frame against its scaffold (no other such
change in the net window) the votes for undoing it decide, with the same guard and protection: unanimous
support and a positive total revert the internal frame (the scaffold was right); at least
``root_frame_minimum_children`` opponents, no supporter and a negative total copy the change into the scaffold
(transfer: the displayed frame then agrees with the decoded one).

Children's frames. A child with one observed parent releases its family MAP frame, which the family MAP
aligned with the parent's homologue labels; in stretches where the child's genotype does not place the
parent's homologue that frame can carry the root's frame error and cancel it in the decoder. When a root's
internal frame changes, such children's frames follow it (``follow_child_frames``).

All inputs are computed from the converged family MAP without changing it, so a recomputed stage makes the same
decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from numba import njit

from ..maths import chains
from . import gauges


@dataclass(frozen=True)
class FrameDecision:
    """One accepted move, revert or transfer of a root's frame at marker boundary ``boundary``."""

    kind: str
    sample: int
    boundary: int
    left_bp: int
    right_bp: int
    component_end_bp: int
    prior: float
    total: float
    flagged: bool
    child_gains: tuple[float, ...]
    child_net_gains: tuple[float, ...]
    round: int = -1


@dataclass(frozen=True)
class ChildFrameChange:
    """A one-parent child's frame following its root (kind follow | cancel | keep | not_adjusted_parent)."""

    sample: int
    parent: int
    kind: str
    start_bp: int = -1
    end_bp: int = -1
    heterozygous_sites: int = 0
    switch_bp: int = -1


@dataclass(frozen=True)
class GuardBlock:
    """A decision blocked by the junction guard: ``blocking_parent`` is mixed near the boundary."""

    sample: int
    boundary: int
    left_bp: int
    blocking_parent: int
    blocking_child_gains: tuple[float, ...]


@dataclass(frozen=True)
class FrameRepair:
    """Decisions of one chromosome; masks are None when nothing was accepted.

    internal int8 [N, L]: relabelling of the decoder frame (moves ^ reverts for roots, plus the following
    frames of their one-parent children); display int8 [N, L]: relabelling of the scaffold (moves ^ transfers).
    """

    moves: tuple[FrameDecision, ...] = ()
    reverts: tuple[FrameDecision, ...] = ()
    transfers: tuple[FrameDecision, ...] = ()
    child_frames: tuple[ChildFrameChange, ...] = ()
    rejected: tuple[tuple[str, int], ...] = ()
    guard_blocks: tuple[GuardBlock, ...] = ()
    flagged_edges: tuple[tuple[int, int], ...] = ()
    internal: np.ndarray | None = field(default=None, repr=False)
    display: np.ndarray | None = field(default=None, repr=False)


@njit(cache=True)
def window_gains(emission, theta, lo, hi, boundaries, mask):
    """Max-path score change of a selector chain when ``mask`` is relabelled after each boundary.

    The four-state chain (emission columns are joint selector states) is decoded on sites [lo, hi) with free
    ends. For each boundary i (between sites i and i + 1, lo <= i < hi - 1) the result is the max over paths
    with state s reading emission s ^ mask on [i + 1, hi), minus the unrelabelled maximum. Relabelling a whole
    suffix permutes its backward scores, so one forward and one backward pass serve every boundary.
    """
    n = hi - lo
    forward = np.empty((n, 4))
    backward = np.empty((n, 4))
    score = np.empty(4)
    for state in range(4):
        score[state] = math.log(max(1e-300, float(emission[lo, state])))
    score -= np.max(score)
    forward[0] = score
    for k in range(1, n):
        chains.max_step4(score, theta[lo + k - 1])
        for state in range(4):
            score[state] += math.log(max(1e-300, float(emission[lo + k, state])))
        score -= np.max(score)
        forward[k] = score
    for state in range(4):
        score[state] = math.log(max(1e-300, float(emission[hi - 1, state])))
    score -= np.max(score)
    backward[n - 1] = score
    for k in range(n - 2, -1, -1):
        chains.max_step4(score, theta[lo + k])
        for state in range(4):
            score[state] += math.log(max(1e-300, float(emission[lo + k, state])))
        score -= np.max(score)
        backward[k] = score
    result = np.empty(len(boundaries))
    for index in range(len(boundaries)):
        k = boundaries[index] - lo
        for state in range(4):
            score[state] = forward[k, state]
        chains.max_step4(score, theta[lo + k])
        current = -np.inf
        relabelled = -np.inf
        for state in range(4):
            current = max(current, score[state] + backward[k + 1, state])
            relabelled = max(relabelled, score[state] + backward[k + 1, state ^ mask])
        result[index] = relabelled - current
    return result


@njit(cache=True)
def zone_gain(emission, theta, lo, zone_lo, zone_hi, hi, edge_theta, mask):
    """Max-path score change when ``mask`` is relabelled on [zone_hi, hi), evidence on [zone_lo, zone_hi) left out.

    The flanks [lo, zone_lo) and [zone_hi, hi) are decoded with free outer ends and joined by one transition
    with probability ``edge_theta`` per bit, so the result depends only on whether the selector state differs
    between the flanks. 0 when a flank is empty.
    """
    if zone_lo <= lo or zone_hi >= hi:
        return 0.
    left = np.empty(4)
    for state in range(4):
        left[state] = math.log(max(1e-300, float(emission[lo, state])))
    left -= np.max(left)
    for k in range(lo + 1, zone_lo):
        chains.max_step4(left, theta[k - 1])
        for state in range(4):
            left[state] += math.log(max(1e-300, float(emission[k, state])))
        left -= np.max(left)
    right = np.empty(4)
    for state in range(4):
        right[state] = math.log(max(1e-300, float(emission[hi - 1, state])))
    right -= np.max(right)
    for k in range(hi - 2, zone_hi - 1, -1):
        chains.max_step4(right, theta[k])
        for state in range(4):
            right[state] += math.log(max(1e-300, float(emission[k, state])))
        right -= np.max(right)
    chains.max_step4(left, edge_theta)
    current = -np.inf
    relabelled = -np.inf
    for state in range(4):
        current = max(current, left[state] + right[state])
        relabelled = max(relabelled, left[state] + right[state ^ mask])
    return relabelled - current


def orientation_flags(scaffold, bins, components, sample, tie_boundaries=None):
    """Bin edges where the root's own data do not carry the scaffold orientation.

    Flagged: edges bounding or inside bins without any scaffold heterozygous site of the sample (abstentions,
    identical rows, hemizygous or missing stretches), and painting tie boundaries.
    """
    ref = scaffold[sample]
    het = (ref[:, 0] >= 0) & (ref[:, 1] >= 0) & (ref[:, 0] != ref[:, 1])
    valid = bins >= 0
    counts = np.bincount(bins[valid], weights=het[valid], minlength=int(bins.max()) + 1 if valid.any() else 0)
    left, right = bins[:-1], bins[1:]
    edge = (left != right) & (left >= 0) & (right >= 0) & (components[:-1] == components[1:])
    flags = np.zeros(len(bins) - 1, dtype=np.bool_)
    flags[edge] = (counts[left[edge]] == 0) | (counts[right[edge]] == 0)
    if tie_boundaries is not None and len(tie_boundaries):
        flags[tie_boundaries] = True
    return flags


def _component_runs(components):
    sites = len(components)
    cuts = np.r_[0, np.flatnonzero(components[1:] != components[:-1]) + 1, sites]
    start = np.empty(sites, dtype=np.int64)
    stop = np.empty(sites, dtype=np.int64)
    for a, b in zip(cuts[:-1], cuts[1:]):
        start[a:b], stop[a:b] = a, b
    return start, stop


def repair_root_frames(workspace, emissions, family_frame, config, *, tie_boundaries, protected):
    """Decide root frame moves, reverts and transfers on the final family state (module docstring).

    ``emissions`` float64 [G, L, 4] are the final selector emissions (map_solver.selector_emissions; relabelled in
    place here); ``family_frame`` int8 [N, L] the family MAP (inferred) frames; ``tie_boundaries`` {sample:
    marker boundaries}; ``protected`` int64 (n, 2) closed bp intervals. Masks are relative to the scaffold.
    """
    cfg = config
    graph = workspace.graph
    pos = np.asarray(workspace.positions, dtype=np.int64)
    comps = np.asarray(workspace.components)
    theta = np.asarray(workspace.theta, dtype=np.float64)
    phase_theta = np.asarray(workspace.phase_theta, dtype=np.float64)
    parents, children = graph.parents, graph.children
    samples, sites = family_frame.shape
    kmin = int(cfg.root_frame_minimum_children)
    tau = float(cfg.root_frame_informative_nats)
    width = float(cfg.root_frame_window_bp)
    tolerance = float(cfg.root_frame_location_bp)
    flank = width - tolerance
    rejected, guard_blocks, flagged_edges = {}, [], []
    moves, reverts, transfers = [], [], []
    edges_of = {int(p): np.flatnonzero(parents == p) for p in np.unique(parents)}
    guard_parents = sorted(p for p, edges in edges_of.items() if len(edges) >= kmin)
    roots = [p for p in guard_parents if graph.roots[p]]
    if not roots or sites < 2:
        return FrameRepair()
    edge_group = graph.edge_group
    edge_mask = (1 << (1 - graph.slots)).astype(np.int64)

    run_start, run_stop = _component_runs(comps)
    candidate = (phase_theta > 0) & (phase_theta < .5) & (comps[:-1] >= 0)
    bin_edges = np.flatnonzero(candidate)
    log_odds = np.zeros(sites - 1)
    log_odds[candidate] = np.log(phase_theta[candidate] / (1 - phase_theta[candidate]))
    prior_odds, flags_of = {}, {}
    for root in roots:
        odds = log_odds.copy()
        flags = orientation_flags(workspace.scaffold, np.asarray(workspace.bins), comps, root,
                                  tie_boundaries.get(root))
        flags &= candidate
        odds[flags] = 0.
        flags_of[root] = flags
        flagged_edges.append((root, int(flags.sum())))
        prior_odds[root] = odds
    ranges = np.asarray(protected, dtype=np.int64).reshape(-1, 2)

    def window(i):
        lo = max(int(run_start[i]), int(np.searchsorted(pos, pos[i] - width, side="left")))
        hi = min(int(run_stop[i]), int(np.searchsorted(pos, pos[i + 1] + width, side="right")))
        return lo, hi

    def neighbour_edges(i):
        """The bin edges before and after i in its component (component ends otherwise)."""
        k = int(np.searchsorted(bin_edges, i))
        start = run_start[i]
        before = int(bin_edges[k - 1]) if k > 0 and run_start[bin_edges[k - 1]] == start else None
        after = (int(bin_edges[k + 1]) if k + 1 < len(bin_edges) and run_start[bin_edges[k + 1]] == start
                 else None)
        return before, after

    def flagged_stretch(parent, i):
        """Boundaries (left, right) of the bins around the run of flagged edges containing i, or None."""
        flags = flags_of.get(parent)
        if flags is None or not flags[i]:
            return None
        start = run_start[i]
        a = b = int(np.searchsorted(bin_edges, i))
        while a > 0 and run_start[bin_edges[a - 1]] == start and flags[bin_edges[a - 1]]:
            a -= 1
        while b + 1 < len(bin_edges) and run_start[bin_edges[b + 1]] == start and flags[bin_edges[b + 1]]:
            b += 1
        left = neighbour_edges(int(bin_edges[a]))[0]
        right = neighbour_edges(int(bin_edges[b]))[1]
        return (int(run_start[i]) if left is None else left,
                int(run_stop[i]) - 2 if right is None else right)

    def votes(parent, i, cache):
        """Per child edge of ``parent`` at bin edge i: exact and net gains, support, opposition and the net
        window; None when the zone reaches an end of the component (no flank on one side)."""
        key = ("votes", parent, i)
        if key not in cache:
            wlo, whi = window(i)
            stretch = flagged_stretch(parent, i)
            left, right = (i, i) if stretch is None else stretch
            start_bp, stop_bp = pos[left] - tolerance, pos[right + 1] + tolerance
            lo = max(int(run_start[i]), int(np.searchsorted(pos, start_bp - flank, side="left")))
            hi = min(int(run_stop[i]), int(np.searchsorted(pos, stop_bp + flank, side="right")))
            zone_lo = max(lo, int(np.searchsorted(pos, start_bp, side="left")))
            zone_hi = min(hi, int(np.searchsorted(pos, stop_bp, side="right")))
            if zone_lo <= lo or zone_hi >= hi:
                cache[key] = None
                return None
            boundaries = np.arange(max(wlo, left), min(whi - 2, right) + 1, dtype=np.int64)
            at = int(np.searchsorted(boundaries, i))
            exact, net, inside = [], [], []
            for edge in edges_of[parent]:
                array, mask = emissions[edge_group[edge]], int(edge_mask[edge])
                values = window_gains(array, theta, wlo, whi, boundaries, mask)
                exact.append(values[at])
                inside.append(values.max())
                net.append(zone_gain(array, theta, lo, zone_lo, zone_hi, hi, theta[i], mask))
            exact, net, inside = (np.asarray(x, dtype=np.float64) for x in (exact, net, inside))
            support = (inside >= tau) & (net > -tau)
            cache[key] = (exact, net, support, net <= -tau, (lo, hi))
        return cache[key]

    def local_gains(parent, i, cache):
        """Per child edge: the best gain at a marker boundary inside the two bins next to i."""
        key = ("local", parent, i)
        if key not in cache:
            lo, hi = window(i)
            near = np.arange(max(lo, int(np.searchsorted(pos, pos[i] - tolerance, side="left"))),
                             min(hi - 1, int(np.searchsorted(pos, pos[i + 1] + tolerance, side="right"))))
            before, after = neighbour_edges(i)
            first = lo if before is None else before
            last = hi - 2 if after is None else after
            same_bin = (near >= max(first, lo)) & (near <= min(last, hi - 2))
            # A boundary near a clipped window end is close to a free gauge (no evidence on one side, gain
            # ~0): it counts only with at least W - tolerance of window on both sides.
            evidenced = (pos[near] - pos[lo] >= flank) & (pos[hi - 1] - pos[near + 1] >= flank)
            at = int(np.searchsorted(near, i))
            local = []
            for edge in edges_of[parent]:
                values = window_gains(emissions[edge_group[edge]], theta, lo, hi, near, int(edge_mask[edge]))
                local.append(max(values[at], values[same_bin & evidenced].max(initial=-np.inf)))
            cache[key] = np.asarray(local)
        return cache[key]

    def protected_edge(i):
        return bool(len(ranges) and np.any((ranges[:, 0] <= pos[i + 1]) & (ranges[:, 1] >= pos[i])))

    def mixed_parent(parent, i, cache):
        # A child shared with ``parent`` has one joint selector chain, where a frame error of ``parent`` can be
        # absorbed by the other parent's bit: it is not independent evidence about the other parent's pattern.
        shared = set(children[edges_of[parent]].tolist())
        for other in guard_parents:
            if other == parent:
                continue
            own = np.asarray([int(children[edge]) not in shared for edge in edges_of[other]])
            best = local_gains(other, i, cache)[own]
            support, oppose = int(np.sum(best >= tau)), int(np.sum(best <= -tau))
            # One changing child is an isolated crossover or a poorly informed meiosis; two changing in the
            # same bin while another does not is the binomial pattern of a transmission change at a junction.
            if support + oppose >= kmin and support >= 2 and oppose:
                return GuardBlock(parent, i, int(pos[i]), other, tuple(round(float(v), 3) for v in best))
        return None

    def reject(reason, block=None):
        rejected[reason] = rejected.get(reason, 0) + 1
        if block is not None:
            guard_blocks.append(block)

    def record(kind, root, i, prior, total, exact, net, round_index=-1):
        flags = flags_of.get(root)
        return FrameDecision(kind, root, i, int(pos[i]), int(pos[i + 1]), int(pos[run_stop[i] - 1]),
                             float(prior), float(total), bool(flags is not None and flags[i]),
                             tuple(round(float(v), 3) for v in exact),
                             tuple(round(float(v), 3) for v in net), round_index)

    def bit_paths(edges, cache):
        """MAP selector bit of each edge (its parent's homologue) from its group's chain, as gauges computes."""
        groups = sorted({int(edge_group[edge]) for edge in edges} - set(cache))
        if groups:
            paths = gauges.selector_paths(emissions, theta, np.asarray(groups, dtype=np.int64))
            for group in groups:
                cache[group] = paths[group]
        return {int(edge): (cache[int(edge_group[edge])] & edge_mask[edge]) != 0 for edge in edges}

    move = np.zeros((samples, sites), dtype=np.int8)
    done = set()
    for round_index in range(int(cfg.root_frame_max_rounds)):
        cache, paths = {}, {}
        # A round relabels chains only after all its paths are taken.
        bits = bit_paths([edge for root in roots for edge in edges_of[root]], paths)
        switches = {}
        for root in roots:
            for edge in edges_of[root]:
                change = np.flatnonzero(bits[int(edge)][1:] != bits[int(edge)][:-1])
                switches[int(edge)] = change[comps[change] == comps[change + 1]]
        proposals = []
        for root in roots:
            frame = family_frame[root]
            # Candidates: the bin edges next to each child's current selector change.
            near = set()
            for edge in edges_of[root]:
                for boundary in switches[int(edge)]:
                    k = int(np.searchsorted(bin_edges, boundary))
                    near.update(int(bin_edges[j]) for j in (k - 1, k) if 0 <= j < len(bin_edges))
            for i in sorted(near):
                if (root, i) in done:
                    continue
                lo, hi = window(i)
                changing = sum(bool(np.any((s >= lo) & (s < hi - 1)))
                               for s in (switches[int(edge)] for edge in edges_of[root]))
                if changing < kmin:
                    continue
                vote = votes(root, i, cache)
                if vote is None:
                    continue
                exact, net, support, oppose, (lo, hi) = vote
                # A family-MAP frame change against the scaffold inside the net window is decided at that
                # change (revert or transfer) below.
                if np.any(frame[lo:hi] != frame[lo]):
                    continue
                total = prior_odds[root][i] + float(exact.sum())
                if np.sum(support) < kmin or np.any(oppose) or total <= 0:
                    continue
                proposals.append((-total, root, i, exact, net, lo, hi))
        accepted = []
        for negative_total, root, i, exact, net, lo, hi in sorted(proposals, key=lambda x: x[:3]):
            # One move per root and net window in a round: edges of one flagged stretch share their votes,
            # and the next round sees the result.
            if any(r == root and lo < b and a < hi for r, _, _, a, b in accepted):
                continue
            done.add((root, i))
            if protected_edge(i):
                reject("protected")
                continue
            blocking = mixed_parent(root, i, cache)
            if blocking is not None:
                reject("junction_guard", blocking)
                continue
            accepted.append((root, i, record("move", root, i, prior_odds[root][i], -negative_total, exact, net,
                                             round_index), lo, hi))
        if not accepted:
            break
        for root, i, item, _, _ in accepted:
            stop = int(run_stop[i])
            move[root, i + 1:stop] ^= 1
            for edge in edges_of[root]:
                group, mask = int(edge_group[edge]), int(edge_mask[edge])
                emissions[group, i + 1:stop] = emissions[group, i + 1:stop][:, np.arange(4) ^ mask]
            moves.append(item)

    revert = np.zeros((samples, sites), dtype=np.int8)
    transfer = np.zeros((samples, sites), dtype=np.int8)
    cache = {}
    for root in roots:
        frame = family_frame[root]
        for i in map(int, np.flatnonzero(candidate & (frame[1:] != frame[:-1]))):
            vote = votes(root, i, cache)
            if vote is None:
                continue
            exact, net, support, oppose, (lo, hi) = vote
            if np.count_nonzero(frame[lo + 1:hi] != frame[lo:hi - 1]) != 1:
                continue  # several family-MAP changes in the window: left as they are
            prior = -prior_odds[root][i]
            total = prior + float(exact.sum())
            if np.sum(support) >= kmin and not np.any(oppose) and total > 0:
                kind, target, decisions = "revert", revert, reverts
            elif np.sum(oppose) >= kmin and not np.any(support) and total < 0:
                kind, target, decisions = "transfer", transfer, transfers
            else:
                continue
            if protected_edge(i):
                reject("protected")
                continue
            blocking = mixed_parent(root, i, cache)
            if blocking is not None:
                reject("junction_guard", blocking)
                continue
            target[root, i + 1:int(run_stop[i])] ^= 1
            decisions.append(record(kind, root, i, prior, total, exact, net))
    report = dict(rejected=tuple(sorted(rejected.items())), guard_blocks=tuple(guard_blocks),
                  flagged_edges=tuple(flagged_edges))
    if not (moves or reverts or transfers):
        return FrameRepair(**report)
    internal = move ^ revert
    follow, child_frames = follow_child_frames(
        workspace.scaffold, family_frame, internal, parents, children, graph.slots, comps, pos)
    return FrameRepair(tuple(moves), tuple(reverts), tuple(transfers), tuple(child_frames), **report,
                       internal=internal ^ follow, display=move ^ transfer)


def follow_child_frames(scaffold, frame, internal, parents, children, slots, components, positions):
    """Frames of one-parent children that follow their root's internal frame changes.

    A sample with one observed parent releases its family MAP frame (``frame``), and recombination decodes its
    meiosis from it: at a parent-heterozygous site where the child is homozygous the observation (transmitted
    allele against the parent's homologue 0) is data; where the child is heterozygous it follows the child's
    frame, which the family MAP aligned with the parent's homologue labels. Inside a run of such sites
    between homozygous ones the child's own data do not place the parent's homologue, so its frame can carry
    the root's frame error and cancel it. ``internal`` relabels roots' homologues (1 = relabelled). For every
    run that contains a relabelling change, the child's frame is changed so that its observations keep the
    family MAP's pattern relative to the homozygous sites on both sides: where the label is equal at the two
    bounding sites (or one side reaches a component end) the frame is relabelled with the root; where it
    differs, the data on the two sides change parity, so the run's own observation switch nearest to the label
    change is removed, or, if the run has none, the switch at the label change is kept (the data require one).
    Children with children of their own (their frame is also a parental frame) are recorded, not changed.
    Returns (mask int8 [N, L], records).
    """
    ref = np.asarray(scaffold)
    frame = np.asarray(frame)
    samples, sites = frame.shape
    comps = np.asarray(components)
    pos = np.asarray(positions)
    run_start, run_stop = _component_runs(comps)
    observed = np.bincount(children, minlength=samples)
    has_children = np.bincount(parents, minlength=samples) > 0
    follow = np.zeros((samples, sites), dtype=np.int8)
    records = []
    for edge in range(len(parents)):
        parent, child, slot = int(parents[edge]), int(children[edge]), int(slots[edge])
        label = np.asarray(internal[parent], dtype=np.int8)
        if observed[child] != 1 or not label.any():
            continue
        if has_children[child]:
            records.append(ChildFrameChange(child, parent, "not_adjusted_parent"))
            continue
        p, c = ref[parent], ref[child]
        used = np.flatnonzero((p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] != p[:, 1])
                              & (c[:, 0] >= 0) & (c[:, 1] >= 0) & (comps >= 0))
        if len(used) == 0:
            continue
        data = c[used, 0] == c[used, 1]
        observation = c[used, slot ^ frame[child, used]] != p[used, frame[parent, used]]
        mark = label[used]
        for block in np.split(np.arange(len(used)), np.flatnonzero(np.diff(comps[used]) != 0) + 1):
            anchors = block[data[block]]
            runs = set()
            # The runs of heterozygous sites (between homozygous ones) that contain a label change.
            for q in block[:-1][mark[block[1:]] != mark[block[:-1]]]:
                j = int(np.searchsorted(anchors, q, side="right"))
                runs.add((int(anchors[j - 1]) if j else None, int(anchors[j]) if j < len(anchors) else None))
            for left, right in sorted(runs, key=lambda run: -1 if run[0] is None else run[0]):
                lo = int(block[0]) if left is None else left + 1
                hi = int(block[-1]) + 1 if right is None else right
                if hi <= lo:
                    continue  # no heterozygous site between these homozygous ones
                ext = np.arange(lo if left is None else left, hi if right is None else right + 1)
                marks = mark[ext]
                start = int(run_start[used[lo]]) if left is None else int(used[left]) + 1
                stop = int(run_stop[used[lo]]) if right is None else int(used[right])
                span = label[start:stop]
                switch_bp = -1
                if left is None or right is None or mark[left] == mark[right]:
                    bound = span[0] if left is None and right is None else (
                        mark[right] if left is None else mark[left])
                    adjust, kind = span ^ bound, "follow"
                else:
                    values = observation[ext]
                    switches = np.flatnonzero(values[1:] != values[:-1])
                    changes = np.flatnonzero(marks[1:] != marks[:-1])
                    if len(switches):
                        k = int(switches[np.argmin(np.abs(switches[:, None] - changes[None, :]).min(axis=1))])
                        kind = "cancel"
                    else:
                        k, kind = int(changes[0]), "keep"
                    # The parity step goes where the label changes between the two observations around the
                    # switch, else where either frame does.
                    a, b = int(used[ext[k]]), int(used[ext[k + 1]])
                    steps = np.flatnonzero(label[a + 1:b + 1] != label[a:b])
                    if not len(steps):
                        steps = np.flatnonzero((frame[child, a + 1:b + 1] != frame[child, a:b])
                                               | (frame[parent, a + 1:b + 1] != frame[parent, a:b]))
                    split = a + 1 + int(steps[0]) if len(steps) else b
                    adjust = span ^ mark[left] ^ (np.arange(start, stop) >= split).astype(np.int8)
                    switch_bp = int(pos[split])
                follow[child, start:stop] = adjust
                if adjust.any() or kind == "keep":
                    changed = np.flatnonzero(adjust)
                    records.append(ChildFrameChange(
                        child, parent, kind,
                        int(pos[start + changed[0]] if len(changed) else pos[used[lo]]),
                        int(pos[start + changed[-1]] if len(changed) else pos[used[hi - 1]]),
                        int(np.count_nonzero(adjust[used[lo:hi] - start])), switch_bp))
    return follow, records


def swap_tracks(scaffold, display):
    """Scaffold alleles with the two tracks exchanged wherever ``display`` is 1."""
    corrected = np.array(scaffold, copy=True)
    for sample in np.flatnonzero(np.any(display, axis=1)):
        flip = display[sample].astype(np.bool_)
        corrected[sample, flip] = corrected[sample, flip][:, ::-1]
    return corrected
