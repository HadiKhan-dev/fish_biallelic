"""Windowed root frame repair on the final family state.

The scaffold frame of a pedigree root (the track order of its painting, used
by the polish and the displayed phase) can be wrong over the rest of a
component: panel rows that exchange founder identity, painting abstentions or
ties between interchangeable rows, segregating deletions. Every child of the
root then changes homologue near the error. Where the root's own data cannot
place that change, the children's changes spread over a stretch, so the
single-boundary gauge move (messages.best_gauge_flips) gains nothing and
belief propagation settles on the scaffold frame. Belief propagation can also
change a root's internal frame (the family MAP, which recombination decodes)
where the scaffold is right.

Votes. For root P, a bin edge e inside a component and each child edge of P,
the child's joint selector chain (the family model's copy-factor emissions and
selector transitions) is re-decoded with free ends after relabelling P's
homologues from e on:
  - exact: the change of the max-path log score on the window +-W around e;
    at a flagged edge (where the root's own data do not carry the scaffold
    orientation, ``orientation_flags``) also the best such change at any
    boundary of the flagged stretch around e ("inside");
  - net: the same change with the evidence inside a location zone left out
    and the zone crossed by one transition priced like e, so only the
    flanks (W - tolerance wide) count: does the child's homologue differ
    between the flanks? A persistent change anywhere in the zone counts, a
    short tract (two changes) inside it does not. The zone is e +-
    ``root_frame_location_bp``, or the flagged stretch widened by that
    tolerance.
A child supports a change near e if it changes homologue there (exact, or
inside at a flagged edge, >= tau) and its flanks do not deny a net change
(net > -tau); it opposes if its flanks show no net change (net <= -tau).
Children whose own change lies a little away from e (scaffold content between
that the root's data do not support) are neutral, not opponents.

Move (internal frame and scaffold together). Where P's family MAP frame agrees
with its scaffold over the whole net window, both are relabelled from e to the
end of the component if at least ``root_frame_minimum_children`` children
support it and none opposes (a frame error moves every child, all-or-none,
while real crossovers and reference-order junctions switch children
binomially), the total at e (the prior log-odds of a scaffold frame change, 0
at flagged edges, plus every child's exact gain) is positive, no other parent
with that many children of its own is mixed inside the two painting bins next
to e (junction guard; children shared with P are left out, as their joint
chain can absorb P's change in the other parent's bit), and e lies outside
supplied protected intervals. Rounds repeat on the relabelled chains.

Belief-propagation frame changes. At a change e of P's family MAP frame against
its scaffold (no other such change in the net window) the votes for undoing
it decide, with the same guard and protection: unanimous support and a
positive total revert the internal frame (the scaffold was right); at least
``root_frame_minimum_children`` opponents, no supporter and a negative total
copy the change into the scaffold (transfer: the displayed frame then agrees
with the decoded one).

Children's frames. A child with one observed parent releases its family MAP
frame, which belief propagation aligned with the parent's homologue labels; in
stretches where the child's genotype does not place the parent's homologue
(heterozygous wherever the parent is) that frame can carry the root's frame
error and cancel it in the decoder. When a root's internal frame changes, such
children's frames follow it (``follow_child_frames``), so a repaired root does
not create a crossover call that the child's data do not require.

All inputs are recomputed from the final family messages without changing
them, so a resumed stage makes the same decisions.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from numba import njit

from . import factors, messages as refinement_messages


PROTECTED_INTERVALS_ENV = "BHD_FAMILY_PHASE_PROTECTED_INTERVALS"


@njit(cache=True, inline="always")
def _transition(score, t, bits):
    """Max-product step of ``bits`` independent selector bits, each changing with probability t."""
    stay = math.log1p(-t)
    jump = math.log(t) if t > 0 else -np.inf
    for b in range(bits):
        bit = 1 << b
        for state in range(len(score)):
            if state & bit:
                continue
            a = score[state]
            c = score[state | bit]
            score[state] = max(a + stay, c + jump)
            score[state | bit] = max(c + stay, a + jump)


@njit(cache=True)
def window_gains(emission, theta, lo, hi, boundaries, mask, bits):
    """Max-path score change of a joint selector chain when ``mask`` is relabelled after a boundary.

    The chain has ``bits`` independent selector bits (emission columns are
    joint states) and is decoded on sites [lo, hi) with free ends. For each
    boundary i (between sites i and i + 1, lo <= i < hi - 1) the result is
    max over paths with state s reading emission s ^ mask on [i + 1, hi),
    minus the unrelabelled maximum. Relabelling a whole suffix permutes its
    backward scores, so one forward and one backward pass serve every boundary.
    """
    states = 1 << bits
    n = hi - lo
    forward = np.empty((n, states))
    backward = np.empty((n, states))
    score = np.empty(states)
    for state in range(states):
        score[state] = math.log(max(1e-300, float(emission[lo, state])))
    score -= np.max(score)
    forward[0] = score
    for k in range(1, n):
        _transition(score, theta[lo + k - 1], bits)
        for state in range(states):
            score[state] += math.log(max(1e-300, float(emission[lo + k, state])))
        score -= np.max(score)
        forward[k] = score
    for state in range(states):
        score[state] = math.log(max(1e-300, float(emission[hi - 1, state])))
    score -= np.max(score)
    backward[n - 1] = score
    for k in range(n - 2, -1, -1):
        _transition(score, theta[lo + k], bits)
        for state in range(states):
            score[state] += math.log(max(1e-300, float(emission[lo + k, state])))
        score -= np.max(score)
        backward[k] = score
    result = np.empty(len(boundaries))
    for index in range(len(boundaries)):
        k = boundaries[index] - lo
        for state in range(states):
            score[state] = forward[k, state]
        _transition(score, theta[lo + k], bits)
        current = -np.inf
        relabelled = -np.inf
        for state in range(states):
            current = max(current, score[state] + backward[k + 1, state])
            relabelled = max(relabelled, score[state] + backward[k + 1, state ^ mask])
        result[index] = relabelled - current
    return result


@njit(cache=True)
def zone_gain(emission, theta, lo, zone_lo, zone_hi, hi, edge_theta, mask, bits):
    """Max-path score change when ``mask`` is relabelled on [zone_hi, hi), evidence on [zone_lo, zone_hi) left out.

    The flanks [lo, zone_lo) and [zone_hi, hi) of a joint selector chain are
    decoded with free outer ends and joined by one transition with
    probability ``edge_theta`` per bit, so the result depends only on whether
    the selector state differs between the flanks. 0 when a flank is empty.
    """
    if zone_lo <= lo or zone_hi >= hi:
        return 0.
    states = 1 << bits
    left = np.empty(states)
    for state in range(states):
        left[state] = math.log(max(1e-300, float(emission[lo, state])))
    left -= np.max(left)
    for k in range(lo + 1, zone_lo):
        _transition(left, theta[k - 1], bits)
        for state in range(states):
            left[state] += math.log(max(1e-300, float(emission[k, state])))
        left -= np.max(left)
    right = np.empty(states)
    for state in range(states):
        right[state] = math.log(max(1e-300, float(emission[hi - 1, state])))
    right -= np.max(right)
    for k in range(hi - 2, zone_hi - 1, -1):
        _transition(right, theta[k], bits)
        for state in range(states):
            right[state] += math.log(max(1e-300, float(emission[k, state])))
        right -= np.max(right)
    _transition(left, edge_theta, bits)
    current = -np.inf
    relabelled = -np.inf
    for state in range(states):
        current = max(current, left[state] + right[state])
        relabelled = max(relabelled, left[state] + right[state ^ mask])
    return relabelled - current


def read_protected_intervals(path):
    """Read a breakpoint TSV (chrom, start, end, reason; header and ``#`` comments allowed).

    This is the order-breakpoint format of the recombination stage. Rows whose
    reason starts with ``reversed_segment`` flag a reversed segment rather than
    a breakpoint and are not protected. Returns {contig: int64 array (n, 2)},
    closed bp intervals.
    """
    intervals = {}
    with open(path) as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split("\t")
            if len(fields) < 3:
                raise ValueError(f"protected interval row needs chrom, start, end: {text!r}")
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError:
                if not intervals and fields[1].strip().lower() == "start":
                    continue
                raise
            reason = fields[3].strip() if len(fields) > 3 else ""
            if reason.startswith("reversed_segment"):
                continue
            if end < start:
                raise ValueError(f"protected interval ends before it starts: {text!r}")
            intervals.setdefault(fields[0].strip(), []).append((start, end))
    return {contig: np.asarray(sorted(rows), dtype=np.int64).reshape((-1, 2))
            for contig, rows in intervals.items()}


def _released_track_changes(before, after, local_keys, bin_index, n_classes, class_of):
    """(label changes, allele-visible changes) of one track across a chunk boundary.

    ``before``/``after`` are released candidate row sets (() = unknown). A
    track keeps a row iff the sets intersect; it keeps its alleles iff some
    members are the same row or their classes share a local key below
    ``n_classes`` in the bin before or after the boundary. O(K).
    """
    if not before or not after:
        changed = int(bool(before) != bool(after))
        return changed, changed
    if set(before).intersection(after):
        return 0, 0
    marked = set()
    for row in before:
        for shared_bin in (bin_index - 1, bin_index):
            key = int(local_keys[shared_bin, class_of[row]])
            if key < n_classes:
                marked.add((shared_bin, key))
    for row in after:
        for shared_bin in (bin_index - 1, bin_index):
            key = int(local_keys[shared_bin, class_of[row]])
            if key < n_classes and (shared_bin, key) in marked:
                return 1, 0
    return 1, 1


def _continuation_changes(left, right, swapped, local_keys, bin_index, n_classes, class_of):
    """Summed (label, allele-visible) changes of the direct or swapped continuation."""
    label = visible = 0
    for track in (0, 1):
        label_change, visible_change = _released_track_changes(
            left.members(track), right.members(1 - track if swapped else track),
            local_keys, bin_index, n_classes, class_of)
        label += label_change
        visible += visible_change
    return label, visible


def painting_tie_boundaries(painting_checkpoint, positions, samples):
    """Marker boundaries where a painting tie, not data, set the scaffold's track order.

    Diploid painting emissions are symmetric in track order and the label
    transitions after a boundary are invariant to swapping both tracks, so the
    track order of everything after a released chunk boundary is a tie whenever
    the ordered and the swapped continuation change equally many labels: both
    tracks change, a homozygous or abstained pair (-1, -1) is entered or left.
    With ragged diagnostics, tracks are compared by their released candidate
    rows (chunk members) and, among equal label changes, by allele-visible
    changes (local allele-equivalence keys); a boundary is a tie unless the
    scaffold's continuation is lexicographically strictly better than the
    swapped one. Boundary i separates sites i and i + 1.
    """
    pos = np.asarray(positions, dtype=np.int64)
    found = {int(sample): [] for sample in samples}
    for component in painting_checkpoint.painting_bundle.components:
        paintings = component.painting.samples
        diagnostics = component.ragged_diagnostics
        if diagnostics is not None:
            local_keys = np.asarray(diagnostics.local_allele_class_keys)
            bin_edges = np.asarray(diagnostics.bin_edges)
            n_classes = len(diagnostics.equivalence_classes)
            class_of = {int(row): index for index, members in enumerate(diagnostics.equivalence_classes)
                        for row in members}
        for sample, boundaries in found.items():
            chunks = paintings[sample].chunks
            for left, right in zip(chunks[:-1], chunks[1:]):
                if left.end != right.start:
                    continue
                if diagnostics is None:
                    x0, x1, y0, y1 = left.hap1, left.hap2, right.hap1, right.hap2
                    if (x0 != y0) + (x1 != y1) != (x0 != y1) + (x1 != y0):
                        continue
                else:
                    bin_index = int(np.searchsorted(bin_edges, right.start, side="right")) - 1
                    direct, crossed = (
                        _continuation_changes(left, right, swapped, local_keys, bin_index, n_classes, class_of)
                        for swapped in (False, True))
                    if direct < crossed:
                        continue
                index = int(np.searchsorted(pos, right.start, side="left"))
                if 0 < index < len(pos):
                    boundaries.append(index - 1)
    return {sample: np.unique(np.asarray(values, dtype=np.int64))
            for sample, values in found.items()}


def orientation_flags(reference, bins, components, sample, tie_boundaries=None):
    """Bin edges where the root's own data do not carry the scaffold orientation.

    Flagged: edges bounding or inside bins without any scaffold heterozygous
    site of the sample (abstentions, identical rows, hemizygous or missing
    stretches), and painting tie boundaries.
    """
    ref = reference[sample]
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


class _Chains:
    """Joint selector chains of the family model and each pedigree edge's selector bit."""

    def __init__(self, workspace, messages, emissions):
        clusters = messages.branch_clusters
        cluster_of = {}
        for index, cluster in enumerate(clusters):
            a, b = map(int, cluster.groups)
            cluster_of[a] = (index, 2)
            cluster_of[b] = (index, 0)
        # Ordinary chains are views of ``emissions`` (relabel edits them in place).
        self.emissions = emissions
        self.arrays, self.bits, self.groups = [], [], []
        keys = {}
        edges = len(workspace.parents)
        self.chain = np.empty(edges, dtype=np.int64)
        self.mask = np.empty(edges, dtype=np.int64)
        for edge in range(edges):
            group = int(workspace.edge_group[edge])
            mask = 1 << (1 - int(workspace.slots[edge]))
            if group in cluster_of:
                index, shift = cluster_of[group]
                key, array, bits = ("cluster", index), clusters[index].emissions, 4
                mask <<= shift
            else:
                key, array, bits = ("group", group), emissions[group], 2
            if key not in keys:
                keys[key] = len(self.arrays)
                self.arrays.append(array)
                self.bits.append(bits)
                self.groups.append(group)
            self.chain[edge] = keys[key]
            self.mask[edge] = mask

    def prefetch(self, edges, theta, cache):
        """Put the MAP paths of these edges' chains in ``cache``, as bit_path would.

        The ordinary chains are independent and run concurrently.
        """
        chains = sorted({int(self.chain[edge]) for edge in edges} - set(cache))
        ordinary = [chain for chain in chains if self.bits[chain] == 2]
        if ordinary:
            groups = np.asarray([self.groups[chain] for chain in ordinary], dtype=np.int64)
            # Only the requested rows are written and read (as in selector_maps).
            paths = np.empty(self.emissions.shape[:2], dtype=np.int8)
            refinement_messages.joint_selector_maps(self.emissions, theta, groups, paths)
            for chain, group in zip(ordinary, groups):
                cache[chain] = paths[group]
        for chain in chains:
            if self.bits[chain] != 2:
                cache[chain] = refinement_messages.branch_map(self.arrays[chain], theta)

    def bit_path(self, edge, theta, cache):
        chain = int(self.chain[edge])
        if chain not in cache:
            array = self.arrays[chain]
            cache[chain] = (refinement_messages.paired_map(array, theta) if self.bits[chain] == 2
                            else refinement_messages.branch_map(array, theta))
        return (cache[chain] & self.mask[edge]) != 0

    def relabel(self, edge, start, stop):
        chain = int(self.chain[edge])
        array = self.arrays[chain]
        order = np.arange(array.shape[1]) ^ int(self.mask[edge])
        array[start:stop] = array[start:stop][:, order]


def _component_runs(components):
    sites = len(components)
    cuts = np.r_[0, np.flatnonzero(components[1:] != components[:-1]) + 1, sites]
    start = np.empty(sites, dtype=np.int64)
    stop = np.empty(sites, dtype=np.int64)
    for a, b in zip(cuts[:-1], cuts[1:]):
        start[a:b], stop[a:b] = a, b
    return start, stop


def repair_root_frames(workspace, messages, inferred_phase_map, config, *,
                       tie_boundaries=None, protected=None):
    """Decide root frame moves, reverts and transfers on the final family state (module docstring).

    Returns masks over (samples, sites) relative to the original scaffold:
    ``internal`` (moves ^ reverts for roots and the following frames of their
    one-parent children, the decoder frame) and ``display`` (moves ^
    transfers, the scaffold track order), plus decision records; both masks
    are None when nothing is accepted.
    """
    cfg = config
    pos = np.asarray(workspace.positions, dtype=np.int64)
    comps = np.asarray(workspace.components)
    theta = np.asarray(workspace.theta, dtype=np.float64)
    phase_theta = np.asarray(workspace.phase_theta, dtype=np.float64)
    parents = np.asarray(workspace.parents)
    children = np.asarray(workspace.children)
    family_frame = np.asarray(inferred_phase_map)
    samples, sites = family_frame.shape
    kmin = int(cfg.root_frame_minimum_children)
    tau = float(cfg.root_frame_informative_nats)
    width = float(cfg.root_frame_window_bp)
    tolerance = float(cfg.root_frame_location_bp)
    flank = width - tolerance
    report = {"moves": [], "reverts": [], "transfers": [], "rejected": {}, "flagged_edges": {},
              "child_frames": []}
    edges_of = {int(p): np.flatnonzero(parents == p) for p in np.unique(parents)}
    guard_parents = sorted(p for p, edges in edges_of.items() if len(edges) >= kmin)
    roots = [p for p in guard_parents if workspace.roots[p]]
    if not roots or sites < 2:
        return dict(report, internal=None, display=None)

    # Final selector emissions: a copy sweep without damping recomputes them
    # from the final messages and changes no message.
    belief = factors.joint_beliefs(workspace.base, workspace.orientation, messages.phase_match,
        messages.parent_log, messages.child_log, workspace.node_group, workspace.offsets,
        workspace.adjacency, workspace.children, cfg.scaffold_phase_error)
    emissions, _ = refinement_messages.copy_messages(belief, messages, workspace.gp, workspace.ge,
        workspace.gc, cfg.transmission_error, 0., False, workspace.factor_support, None)
    del belief
    chains = _Chains(workspace, messages, emissions)

    run_start, run_stop = _component_runs(comps)
    candidate = (phase_theta > 0) & (phase_theta < .5) & (comps[:-1] >= 0)
    bin_edges = np.flatnonzero(candidate)
    log_odds = np.zeros(sites - 1)
    log_odds[candidate] = np.log(phase_theta[candidate] / (1 - phase_theta[candidate]))
    prior_odds, flags_of = {}, {}
    for root in roots:
        odds = log_odds.copy()
        if cfg.root_frame_support_flags:
            flags = orientation_flags(workspace.reference, np.asarray(workspace.bins), comps, root,
                                      None if tie_boundaries is None else tie_boundaries.get(root))
            flags &= candidate
            odds[flags] = 0.
            flags_of[root] = flags
            report["flagged_edges"][root] = int(flags.sum())
        prior_odds[root] = odds
    ranges = np.zeros((0, 2), dtype=np.int64) if protected is None else np.asarray(protected, dtype=np.int64)

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
        """Per child edge of ``parent`` at bin edge i: exact, net and inside gains (module
        docstring), support and opposition, and the net window; None when the zone reaches an
        end of the component (no flank on one side: no net vote, no decision)."""
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
                chain = int(chains.chain[edge])
                array, mask, bits = chains.arrays[chain], int(chains.mask[edge]), chains.bits[chain]
                values = window_gains(array, theta, wlo, whi, boundaries, mask, bits)
                exact.append(values[at])
                inside.append(values.max())
                net.append(zone_gain(array, theta, lo, zone_lo, zone_hi, hi, theta[i], mask, bits))
            exact, net, inside = (np.asarray(x, dtype=np.float64) for x in (exact, net, inside))
            # Support: the child changes homologue at e (anywhere in a flagged
            # stretch) and its flanks do not deny a net change; opposition: its
            # flanks show no net change (a short tract is not a frame change).
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
            # A boundary near a clipped window end is close to a free gauge (no
            # evidence on one side, gain ~0): it counts only with at least
            # W - tolerance of window on both sides.
            evidenced = (pos[near] - pos[lo] >= flank) & (pos[hi - 1] - pos[near + 1] >= flank)
            at = int(np.searchsorted(near, i))
            local = []
            for edge in edges_of[parent]:
                chain = int(chains.chain[edge])
                values = window_gains(chains.arrays[chain], theta, lo, hi, near,
                                      int(chains.mask[edge]), chains.bits[chain])
                local.append(max(values[at], values[same_bin & evidenced].max(initial=-np.inf)))
            cache[key] = np.asarray(local)
        return cache[key]

    def protected_edge(i):
        return bool(len(ranges) and np.any((ranges[:, 0] <= pos[i + 1]) & (ranges[:, 1] >= pos[i])))

    def mixed_parent(parent, i, cache):
        # A child shared with ``parent`` has one joint selector chain, where a
        # frame error of ``parent`` can be absorbed by the other parent's bit:
        # it is not independent evidence about the other parent's pattern.
        shared = set(children[edges_of[parent]].tolist())
        for other in guard_parents:
            if other == parent:
                continue
            own = np.asarray([int(children[edge]) not in shared for edge in edges_of[other]])
            best = local_gains(other, i, cache)[own]
            support, oppose = int(np.sum(best >= tau)), int(np.sum(best <= -tau))
            # One changing child is an isolated crossover or a poorly informed
            # meiosis; two changing in the same bin while another does not is
            # the binomial pattern of a transmission change at a junction.
            if support + oppose >= kmin and support >= 2 and oppose:
                return dict(sample=parent, boundary=i, left_bp=int(pos[i]), blocking_parent=other,
                            blocking_child_gains=[round(float(v), 3) for v in best])
        return None

    def reject(reason, detail=None):
        report["rejected"][reason] = report["rejected"].get(reason, 0) + 1
        if detail is not None:
            report.setdefault("guard_details", []).append(detail)

    def record(kind, root, i, prior, total, exact, net, **extra):
        flags = flags_of.get(root)
        return dict(kind=kind, sample=root, boundary=i, left_bp=int(pos[i]), right_bp=int(pos[i + 1]),
                    component_end_bp=int(pos[run_stop[i] - 1]), prior=float(prior), total=float(total),
                    flagged=bool(flags is not None and flags[i]),
                    child_gains=[round(float(v), 3) for v in exact],
                    child_net_gains=[round(float(v), 3) for v in net], **extra)

    move = np.zeros((samples, sites), dtype=np.int8)
    done = set()
    for round_index in range(int(cfg.root_frame_max_rounds)):
        cache, paths, switches = {}, {}, {}
        # A round relabels chains only after all its paths are taken.
        chains.prefetch([edge for root in roots for edge in edges_of[root]], theta, paths)
        for root in roots:
            for edge in edges_of[root]:
                bits = chains.bit_path(edge, theta, paths)
                change = np.flatnonzero(bits[1:] != bits[:-1])
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
                # A family-MAP frame change against the scaffold inside the net
                # window is decided at that change (revert or transfer) below.
                if np.any(frame[lo:hi] != frame[lo]):
                    continue
                total = prior_odds[root][i] + float(exact.sum())
                if np.sum(support) < kmin or np.any(oppose) or total <= 0:
                    continue
                proposals.append((-total, root, i, exact, net, lo, hi))
        accepted = []
        for negative_total, root, i, exact, net, lo, hi in sorted(proposals, key=lambda x: x[:3]):
            # One move per root and net window in a round: edges of one flagged
            # stretch share their votes, and the next round sees the result.
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
                                             round=round_index), lo, hi))
        if not accepted:
            break
        for root, i, item, _, _ in accepted:
            stop = int(run_stop[i])
            move[root, i + 1:stop] ^= 1
            for edge in edges_of[root]:
                chains.relabel(edge, i + 1, stop)
            report["moves"].append(item)

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
                kind, target = "revert", revert
            elif np.sum(oppose) >= kmin and not np.any(support) and total < 0:
                kind, target = "transfer", transfer
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
            report[kind + "s"].append(record(kind, root, i, prior, total, exact, net))
    if not (report["moves"] or report["reverts"] or report["transfers"]):
        return dict(report, internal=None, display=None)
    internal = move ^ revert
    follow, report["child_frames"] = follow_child_frames(
        workspace.reference, family_frame, internal, parents, children, workspace.slots, comps, pos)
    return dict(report, internal=internal ^ follow, display=move ^ transfer)


def follow_child_frames(reference, frame, internal, parents, children, slots, components, positions):
    """Frames of one-parent children that follow their root's internal frame changes.

    Under the scaffold polish a sample with one observed parent releases its
    family MAP frame (``frame``), and recombination decodes its meiosis from it:
    at a parent-heterozygous site where the child is homozygous the observation
    (transmitted allele against the parent's homologue 0) is data; where the
    child is heterozygous it follows the child's frame, which belief propagation
    aligned with the parent's homologue labels. Inside a run of such sites
    between homozygous ones the child's own data do not place the parent's
    homologue, so its frame can carry the root's frame error and cancel it.
    ``internal`` relabels roots' homologues (1 = relabelled). For every run that
    contains a relabelling change, the child's frame is changed so that its
    observations keep belief propagation's pattern relative to the homozygous
    sites on both sides: where the label is equal at the two bounding sites (or
    one side reaches a component end) the frame is relabelled with the root;
    where it differs, the data on the two sides change parity, so the run's own
    observation switch nearest to the label change is removed, or, if the run
    has none, the switch at the label change is kept (the data require one).
    Children with children of their own (their frame is also a parental frame)
    are recorded, not changed. Returns (mask, records).
    """
    ref = np.asarray(reference)
    frame = np.asarray(frame)
    samples, sites = frame.shape
    parents, children, slots = (np.asarray(x, dtype=np.int64) for x in (parents, children, slots))
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
            records.append(dict(sample=child, parent=parent, kind="not_adjusted_parent"))
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
                    # The parity step goes where the label changes between the two
                    # observations around the switch, else where either frame does.
                    a, b = int(used[ext[k]]), int(used[ext[k + 1]])
                    steps = np.flatnonzero(label[a + 1:b + 1] != label[a:b])
                    if not len(steps):
                        steps = np.flatnonzero((frame[child, a + 1:b + 1] != frame[child, a:b])
                                               | (frame[parent, a + 1:b + 1] != frame[parent, a:b]))
                    split = a + 1 + int(steps[0]) if len(steps) else b
                    adjust = span ^ mark[left] ^ (np.arange(start, stop) >= split).astype(np.int8)
                follow[child, start:stop] = adjust
                if adjust.any() or kind == "keep":
                    changed = np.flatnonzero(adjust)
                    item = dict(sample=child, parent=parent, kind=kind,
                                start_bp=int(pos[start + changed[0]] if len(changed) else pos[used[lo]]),
                                end_bp=int(pos[start + changed[-1]] if len(changed) else pos[used[hi - 1]]),
                                heterozygous_sites=int(np.count_nonzero(adjust[used[lo:hi] - start])))
                    if kind != "follow":
                        item["switch_bp"] = int(pos[split])
                    records.append(item)
    return follow, records


def swap_tracks(reference, display):
    """Scaffold alleles with the two tracks exchanged wherever ``display`` is 1."""
    corrected = np.array(reference, copy=True)
    for sample in np.flatnonzero(np.any(display, axis=1)):
        flip = display[sample].astype(np.bool_)
        corrected[sample, flip] = corrected[sample, flip][:, ::-1]
    return corrected


def finish_repaired_phase(reference, family, repair, positions, bins, components,
                          parents, children, slots, *, phase_config, chromosome_map, finish):
    """Polish against the repaired scaffold; return frames in original scaffold coordinates.

    Roots' tracks are exchanged where the display mask is 1; relative to that
    scaffold a root's internal frame is family MAP ^ internal ^ display. The
    returned phase maps and root-edge selectors are XORed back with the
    display mask, so the product keeps the painting's coordinate frame:
    displayed = display mask, internal = family MAP ^ internal mask (the
    internal mask also carries one-parent children's following frames, whose
    display mask is 0). Allele calls are taken in the displayed orientation.
    """
    display, internal = repair["display"], repair["internal"]
    corrected = swap_tracks(reference, display)
    shifted = replace(family, inferred_phase_map=family.inferred_phase_map ^ internal ^ display)
    point = finish(corrected, shifted, positions, bins, components, parents, children, slots,
                   config=phase_config, chromosome_map=chromosome_map)
    result = point.conditional_result
    result = replace(result, phase_map=result.phase_map ^ display,
                     selectors=result.selectors ^ display[np.asarray(parents)])
    return replace(point, phase_map=point.phase_map ^ display, conditional_result=result)
