"""Agree both convention solves and release meiosis chains (FP20, FP21)."""

import numpy as np
from numba import types

from ...contracts.phase import AgreedRelease, SampleRelease
from ...core.kernels import kernel
from .parameters import MEIOSIS_RESET_TOLERANCE, ORIGIN_BIN_BATCH


def set_start_bins(set_starts, site_bin, B):
    """(reset bool [B], blocked bool [B]) of one sample's released phase-set starts (indices into the component's
    released sites): a meiosis chain resets at the bin holding each start; a start inside a bin (after its first
    released site) also blocks that bin, whose sites belong to two sets, and resets after it."""
    reset = np.zeros(B, dtype=bool)
    blocked = np.zeros(B, dtype=bool)
    for i in set_starts:
        b = int(site_bin[i])
        reset[b] = True
        if i > 0 and site_bin[i - 1] == b:
            blocked[b] = True
            if b + 1 < B:
                reset[b + 1] = True
    return reset, blocked


def edge_chains(model, mirror, chains, mirrored_chains, theta_m, starts, tau, *, map_work=map):
    """{(parent, child, slot): (source int8 [B], support float32 [B], reset bool [B])} for every linked homologue of
    the component, from both solves' release chains (release_sample's "chain").

    A chain resets at the component's first bin, at meiosis resets (theta_m = 1/2) and at the parent's and the
    child's released phase-set starts (set_start_bins). Source 0/1 refers to the parent's homologue order: slot
    order for a parent with a linked parent, else each solve's own gauge, so within a segment the mirrored solve's
    sources are aligned to the default solve by the parent's labels (a segment whose two label orders disagree is
    blocked), and a bin is released only where the parent's aligned label pair is the same in both solves, so that
    a source names the same copied labels. The released source of a bin is the parent homologue that both aligned
    solves give a posterior of at least ``tau`` (else -1), its support the smaller of the two posteriors.
    """
    B = model.B
    base = np.zeros(B, dtype=bool)
    base[0] = True
    reset_decision = np.asarray(theta_m) >= 0.5 - MEIOSIS_RESET_TOLERANCE
    base[1:] |= reset_decision
    def release_edge(edge):
        p, c, k = edge
        a, b = chains[c][k], mirrored_chains[c][k]
        reset = base | starts[p][0] | starts[c][0]
        blocked = starts[p][1] | starts[c][1]
        flip = np.zeros(B, dtype=bool)
        X, Y = model.X[p], mirror.X[p]
        if np.all(model.par[p] < 0):
            segment = np.cumsum(reset)
            comparable = (
                (X[0] != X[1])
                & (np.minimum(X[0], X[1]) == np.minimum(Y[0], Y[1]))
                & (np.maximum(X[0], X[1]) == np.maximum(Y[0], Y[1]))
            )
            swapped = comparable & (Y[0] == X[1])
            # One integer count per segment. The tests below are the per-segment orientation decisions
            # applied to every bin, without rescanning all bins for each segment.
            size = int(segment[-1]) + 1
            n_comparable = np.bincount(segment[comparable], minlength=size)
            n_swapped = np.bincount(segment[swapped], minlength=size)
            has_swaps = n_swapped > 0
            mixed = n_swapped < n_comparable
            blocked |= (has_swaps & mixed)[segment]
            flip = (has_swaps & ~mixed)[segment]
        blocked |= (np.where(flip, Y[1], Y[0]) != X[0]) | (np.where(flip, Y[0], Y[1]) != X[1])
        va = np.where(a[0] >= tau, 0, np.where(a[1] >= tau, 1, -1))
        vb = np.where(b[0] >= tau, 0, np.where(b[1] >= tau, 1, -1))
        vb = np.where((vb >= 0) & flip, 1 - vb, vb)
        source = np.where((va >= 0) & (va == vb) & ~blocked, va, -1).astype(np.int8)
        support = np.where(source >= 0, np.minimum(np.maximum(a[0], a[1]), np.maximum(b[0], b[1])), 0.0).astype(
            np.float32
        )
        return edge, (source, support, reset)

    edges = [(int(model.par[c, k]), c, k) for c, k in zip(*np.nonzero(model.par >= 0), strict=True)]
    return dict(map_work(release_edge, edges))


def agreement(first: SampleRelease, second: SampleRelease, site_bin: np.ndarray) -> AgreedRelease:
    """One sample's release from two solves under opposite tie conventions (released outputs must not depend on a
    convention): genotypes where both call the same genotype (else unknown); a heterozygote is phased where both phase
    it; phase sets break at both solves' breaks and wherever their relative orientation changes (so within a set the
    two agree up to one global flip, and the first solve's orientation is released). Returns (calls, undetermined,
    site indices where a new phase set starts) and, when both releases carry scores, float32 [2, m]: the smaller of
    the two solves' scores, -1 where the agreement releases no genotype (row 0) or no phased heterozygote (row 1)."""
    calls_a, und_a, breaks_a = first.calls, first.undetermined, first.set_breaks
    calls_b, und_b, breaks_b = second.calls, second.undetermined, second.set_breaks
    scores_a, scores_b = first.scores, second.scores
    m = len(site_bin)
    starts = np.zeros(m, dtype=bool)
    cut_bins = np.asarray(sorted(set(breaks_a) | set(breaks_b)), dtype=np.int64)
    first_sites = np.searchsorted(site_bin, cut_bins, side="left")
    starts[first_sites[first_sites < m]] = True
    scores = np.minimum(scores_a, scores_b) if scores_a is not None and scores_b is not None else None
    calls, undetermined = _agree_sites(calls_a, und_a, calls_b, und_b, starts, scores)
    return AgreedRelease(calls, undetermined, np.flatnonzero(starts), scores)


def high_noncopy(models, releases, threshold, *, map_work=map):
    """Bins where each solve has at least one track with high propagated noncopy support."""

    def scan_bins(start):
        stop = min(start + ORIGIN_BIN_BATCH, models[0].B)
        bins = slice(start, stop)
        high = np.ones((models[0].N, stop - start), dtype=bool)
        for model, solved in zip(models, releases, strict=True):
            support = np.zeros((model.N, 2, stop - start), dtype=np.float64)
            # A child's support depends on both parental tracks in this solve. Independent bin batches run together.
            for child in model.order:
                for slot, parent in enumerate(model.par[child]):
                    if parent >= 0:
                        q0, q1 = solved[child].chain[slot, :, bins]
                        support[child, slot] = (1.0 - (q0 + q1)) + q0 * support[parent, 0] + q1 * support[parent, 1]
            high &= (support >= threshold).any(axis=1)
        return high

    return np.concatenate(tuple(map_work(scan_bins, range(0, models[0].B, ORIGIN_BIN_BATCH))), axis=1)


def _disputed_bins(high, frame_starts, site_bin, parity):
    """Select the sites in determinate, opposing bins of a bounded high-support tract."""
    # Number runs on the full bin axis so a missing whole run cannot connect two observed stretches.
    run = np.cumsum(np.r_[False, high[1:] != high[:-1]])
    new_group = np.r_[True, site_bin[1:] != site_bin[:-1], False]
    new_group[frame_starts] = True
    group_starts = np.flatnonzero(new_group[:-1])
    group_bins = site_bin[group_starts]
    group_runs = run[group_bins]
    group_frames = np.searchsorted(frame_starts, group_starts, side="right")
    zero = np.logical_or.reduceat(parity == 0, group_starts)
    one = np.logical_or.reduceat(parity == 1, group_starts)
    informative = zero ^ one
    changes = (group_frames[1:] != group_frames[:-1]) | (group_runs[1:] != group_runs[:-1])
    stretches = np.r_[0, 1 + np.flatnonzero(changes), len(group_starts)]
    mask = np.zeros(len(group_starts), dtype=bool)
    for preceding, start, stop, following in zip(stretches, stretches[1:], stretches[2:], stretches[3:]):
        if not high[group_bins[start]] or group_frames[start - 1] != group_frames[stop]:
            continue
        if group_runs[start] - group_runs[start - 1] != 1 or group_runs[stop] - group_runs[start] != 1:
            continue
        left = np.flatnonzero(informative[preceding:start])
        right = np.flatnonzero(informative[stop:following])
        if len(left) and len(right):
            flank_order = one[preceding + left[-1]]
            if flank_order == one[stop + right[0]]:
                mask[start:stop] = informative[start:stop] & (one[start:stop] != flank_order)
    return np.repeat(mask, np.diff(np.r_[group_starts, len(site_bin)]))


def withhold_local_phase(agreed, high, site_bin, tie_starts, paint_rows, paint_alleles, panel_cols):
    """Update phase flags and scores of the newly agreed release in the native phase frames."""
    if not len(site_bin) or not high.any():
        return
    g0, g1 = agreed.calls.T
    phased = (g0 >= 0) & (g1 >= 0) & (g0 != g1) & ~agreed.undetermined
    a, b = paint_rows[:, site_bin]
    a0, a1 = paint_alleles[a, panel_cols], paint_alleles[b, panel_cols]
    comparable = phased & (a >= 0) & (b >= 0) & (a0 >= 0) & (a1 >= 0) & (a0 != a1)
    parity = np.where(comparable, (a0 != g0).astype(np.int8), np.int8(-1))
    frame_starts = np.union1d(agreed.set_starts, tie_starts)
    mask = _disputed_bins(high, frame_starts, site_bin, parity) & phased
    agreed.undetermined[mask] = True
    if agreed.scores is not None:
        agreed.scores[1, mask] = -1.0


@kernel(
    overloads=tuple(
        (
            types.Array(types.int8, 2, "C", readonly=readonly),
            types.Array(types.boolean, 1, "C", readonly=readonly),
            types.Array(types.int8, 2, "C", readonly=readonly),
            types.Array(types.boolean, 1, "C", readonly=readonly),
            types.Array(types.boolean, 1, "C"),
            score_type,
        )
        for readonly in (False, True)
        for score_type in (types.none, types.Array(types.float32, 2, "C"))
    ),
    cache=True,
    nogil=True,
)
def _agree_sites(calls_a, und_a, calls_b, und_b, starts, scores):
    """Scan the fixed cuts and agreed heterozygotes in site order, once per sample."""
    calls = np.empty_like(calls_a)
    undetermined = np.empty(len(starts), dtype=np.bool_)
    previous_flip = -1
    for site in range(len(starts)):
        if starts[site]:
            previous_flip = -1
        a0, a1 = calls_a[site]
        b0, b1 = calls_b[site]
        gt_a = int(a0) + int(a1) if a0 >= 0 and a1 >= 0 else -1
        gt_b = int(b0) + int(b1) if b0 >= 0 and b1 >= 0 else -1
        agree = gt_a == gt_b
        calls[site, 0] = a0 if agree else -1
        calls[site, 1] = a1 if agree else -1
        het = agree and gt_a == 1
        phased = het and not und_a[site] and not und_b[site]
        undetermined[site] = het and not phased
        if phased:
            flip = int(a0 != b0)
            if previous_flip >= 0 and flip != previous_flip:
                starts[site] = True
            previous_flip = flip
        if scores is not None:
            if not agree or gt_a < 0:
                scores[0, site] = -1.0
            if not phased:
                scores[1, site] = -1.0
    starts[:1] = False
    return calls, undetermined


__all__ = ["agreement", "edge_chains", "high_noncopy", "set_start_bins", "withhold_local_phase"]
