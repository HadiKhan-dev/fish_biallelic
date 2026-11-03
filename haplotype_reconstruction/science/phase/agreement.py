"""Agree both convention solves and release meiosis chains (P0 FP20, FP21)."""

import numpy as np

from ...contracts.phase import AgreedRelease, SampleRelease
from .diagnostics import DecisionTrace, PhaseTrace
from .parameters import MEIOSIS_RESET_TOLERANCE


def _set_start_bins(set_starts, site_bin, B):
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


def _edge_chains(model, mirror, chains, mirrored_chains, theta_m, starts, tau, *, audit=None):
    """{(parent, child, slot): (source int8 [B], support float32 [B], reset bool [B])} for every linked homologue of
    the component, from both solves' release chains (release_sample's "chain").

    A chain resets at the component's first bin, at meiosis resets (theta_m = 1/2) and at the parent's and the
    child's released phase-set starts (_set_start_bins). Source 0/1 refers to the parent's homologue order: slot
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
    if audit is not None:
        audit.difference("reset", theta_m, 0.5 - MEIOSIS_RESET_TOLERANCE, reset_decision)
    out = {}
    for c in range(model.N):
        for k in (0, 1):
            p = int(model.par[c, k])
            if p < 0:
                continue
            a, b = chains[c][k], mirrored_chains[c][k]
            reset = base | starts[p][0] | starts[c][0]
            blocked = starts[p][1] | starts[c][1]
            segment = np.cumsum(reset)
            flip = np.zeros(B, dtype=bool)
            if np.all(model.par[p] < 0):
                X, Y = model.X[p], mirror.X[p]
                comparable = (
                    (X[0] != X[1])
                    & (np.minimum(X[0], X[1]) == np.minimum(Y[0], Y[1]))
                    & (np.maximum(X[0], X[1]) == np.maximum(Y[0], Y[1]))
                )
                swapped = comparable & (Y[0] == X[1])
                for g in np.unique(segment[comparable]):
                    inside = segment == g
                    n_swapped = int(np.sum(swapped & inside))
                    if audit is not None and audit.extended:
                        count = int(np.sum(comparable & inside))
                        audit.emit("alignment", 1 if 0 < n_swapped < count else 2 if n_swapped else 0, n_swapped, count)
                    if 0 < n_swapped < int(np.sum(comparable & inside)):
                        blocked = blocked | inside
                    elif n_swapped:
                        flip |= inside
            X, Y = model.X[p], mirror.X[p]
            blocked = blocked | (np.where(flip, Y[1], Y[0]) != X[0]) | (np.where(flip, Y[0], Y[1]) != X[1])
            if audit is not None:
                for array in (a, b):
                    cutoff = np.asarray(tau, dtype=array.dtype)
                    for row in array:
                        audit.emit("chain", row >= cutoff, row.astype(np.float64), float(cutoff))
            va = np.where(a[0] >= tau, 0, np.where(a[1] >= tau, 1, -1))
            vb = np.where(b[0] >= tau, 0, np.where(b[1] >= tau, 1, -1))
            vb = np.where((vb >= 0) & flip, 1 - vb, vb)
            source = np.where((va >= 0) & (va == vb) & ~blocked, va, -1).astype(np.int8)
            support = np.where(source >= 0, np.minimum(np.maximum(a[0], a[1]), np.maximum(b[0], b[1])), 0.0).astype(
                np.float32
            )
            out[(p, c, k)] = (source, support, reset)
    return out


def _agreement(first: SampleRelease, second: SampleRelease, site_bin: np.ndarray) -> AgreedRelease:
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
    gt_a = np.where((calls_a >= 0).all(axis=1), calls_a.sum(axis=1, dtype=np.int64), -1)
    gt_b = np.where((calls_b >= 0).all(axis=1), calls_b.sum(axis=1, dtype=np.int64), -1)
    agree = gt_a == gt_b
    calls = np.where(agree[:, None], calls_a, np.int8(-1)).astype(np.int8)
    het = agree & (gt_a == 1)
    phased = het & ~und_a & ~und_b
    starts = np.zeros(m, dtype=bool)
    cut_bins = np.asarray(sorted(set(breaks_a) | set(breaks_b)), dtype=np.int64)
    first_sites = np.searchsorted(site_bin, cut_bins, side="left")
    starts[first_sites[first_sites < m]] = True
    idx = np.flatnonzero(phased)
    if len(idx) > 1:
        flipped = (calls_a[idx, 0] != calls_b[idx, 0]).astype(np.int8)
        same_set = np.diff(np.cumsum(starts)[idx]) == 0
        starts[idx[1:][(np.diff(flipped) != 0) & same_set]] = True
    starts[:1] = False
    if scores_a is not None and scores_b is not None:
        scores = np.minimum(scores_a, scores_b)
        scores[0, ~agree | (gt_a < 0)] = -1.0  # no genotype released
        scores[1, ~phased] = -1.0  # no phased heterozygote released
        return AgreedRelease(calls, het & ~phased, np.flatnonzero(starts), scores)
    return AgreedRelease(calls, het & ~phased, np.flatnonzero(starts), None)


__all__ = []


def edge_chains(model, mirror, chains, mirrored_chains, theta_m, starts, tau, *, trace=False):
    """Validation path for aligned chain release, optionally observing cutoffs and alignment segments."""
    audit = DecisionTrace(extended=True) if trace else None
    value = _edge_chains(model, mirror, chains, mirrored_chains, theta_m, starts, tau, audit=audit)
    return PhaseTrace(value, tuple(audit.rows)) if trace else value
