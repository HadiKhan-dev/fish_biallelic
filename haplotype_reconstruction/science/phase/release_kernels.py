"""Exact conditional block and site posteriors for final-state release (FP15/FP16)."""

import math

import numpy as np
from numba import types

from ...core.kernels import kernel
from ...core.numerics import log_add
from .kernels import gauge_children_terms, slot_init, slot_label, slot_log_transition, subtree_terms
from .parameters import IMPOSSIBLE_LOG_WEIGHT, POSTERIOR_LOG_FLOOR

NEG = IMPOSSIBLE_LOG_WEIGHT

_OVERLOADS = {
    "_lse": ((types.float64, types.float64),),
    "block_tables": (
        (
            types.int64,
            types.Array(types.int64, 1, "C", False, aligned=True),
            types.Array(types.int64, 1, "C", False, aligned=True),
            types.Array(types.float64, 4, "C", False, aligned=True),
            types.Array(types.int64, 3, "C", False, aligned=True),
            types.Array(types.int8, 3, "C", False, aligned=True),
            types.Array(types.int64, 2, "C", False, aligned=True),
            types.Array(types.int64, 1, "C", False, aligned=True),
            types.Array(types.int64, 1, "C", False, aligned=True),
            types.Array(types.int8, 1, "C", False, aligned=True),
            types.int64,
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.float64,
            types.float64,
        ),
    ),
    "_step": (
        (
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 2, "C", False, aligned=True),
            types.Array(types.float64, 2, "C", False, aligned=True),
            types.Array(types.float64, 2, "C", False, aligned=True),
            types.int64,
        ),
    ),
    "_step_back": (
        (
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 2, "C", False, aligned=True),
            types.Array(types.float64, 2, "C", False, aligned=True),
            types.Array(types.float64, 2, "C", False, aligned=True),
            types.int64,
        ),
    ),
    "forward_backward": (
        (
            types.Array(types.float64, 4, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.int64,
            types.Array(types.boolean, 2, "C", False, aligned=True),
        ),
    ),
    "chain_messages": (
        (
            types.Array(types.float64, 4, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
        ),
    ),
    "posterior_bins": (
        (
            types.Array(types.float64, 4, "C", False, aligned=True),
            types.Array(types.float64, 4, "C", False, aligned=True),
        ),
    ),
    "restricted": (
        (
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.boolean, 2, "C", False, aligned=True),
        ),
    ),
    "forward_from": (
        (
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 4, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.int64,
            types.int64,
        ),
    ),
    "backward_to": (
        (
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 4, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.int64,
            types.int64,
            types.Array(types.boolean, 2, "C", False, aligned=True),
        ),
    ),
    "_flip_pair": (
        (
            types.float64,
            types.float64,
            types.float64,
            types.float64,
            types.float32,
            types.float32,
            types.float32,
            types.float64,
            types.Array(types.float64, 2, "C", False, aligned=True),
            types.int64,
        ),
    ),
    "site_posteriors": (
        (
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.int64, 1, "C", False, aligned=True),
            types.Array(types.int8, 2, "C", False, aligned=True),
            types.Array(types.float32, 2, "C", False, aligned=True),
            types.float64,
            types.Array(types.float64, 2, "C", False, aligned=True),
            types.Array(types.int64, 2, "C", False, aligned=True),
            types.Array(types.float64, 2, "C", False, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.int64, 1, "C", False, aligned=True),
            types.Array(types.int8, 2, "F", False, aligned=True),
            types.Array(types.float32, 2, "C", False, aligned=True),
            types.float64,
            types.Array(types.float64, 2, "C", False, aligned=True),
            types.Array(types.int64, 2, "C", False, aligned=True),
            types.Array(types.float64, 2, "C", False, aligned=True),
        ),
    ),
    "label_pairs": (
        (
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.int64, 2, "C", False, aligned=True),
            types.Array(types.int64, 2, "C", False, aligned=True),
            types.int64,
        ),
    ),
    "orientation": (
        (
            types.Array(types.float64, 2, "C", False, aligned=True),
            types.Array(types.int64, 1, "C", False, aligned=True),
            types.float64,
            types.int64,
        ),
    ),
}


@kernel(overloads=_OVERLOADS["_lse"], cache=True, nogil=True)
def _lse(a, b):
    if a < b:
        a, b = b, a
    if b <= NEG:
        return a
    return log_add(a, b)


@kernel(overloads=_OVERLOADS["block_tables"], cache=True, nogil=True)
def block_tables(
    n,
    kinds,
    parents,
    E,
    X,
    V,
    par,
    desc,
    child_tracks,
    mark,
    S,
    lstay,
    lmove,
    mstay,
    mmove,
    ain,
    astay,
    bout,
    init_copy,
    init_bottom,
):
    """The block's factors for forward_backward: em [B, n0, n1, G] (bin 0 includes the slots' initial terms), the
    slot transitions T0, T1 and the gauge transition TG per bin edge, and each slot state's label per bin."""
    B = E.shape[3]
    n0 = S + 2 if kinds[0] == 1 else S
    n1 = S + 2 if kinds[1] == 1 else S
    G = 2 if len(child_tracks) > 0 else 1
    em = np.empty((B, n0, n1, G))
    T0 = np.zeros((B, n0, n0))
    T1 = np.zeros((B, n1, n1))
    TG = np.zeros((B, 2, 2))
    labs0 = np.empty((B, n0), dtype=np.int64)
    labs1 = np.empty((B, n1), dtype=np.int64)
    F0 = np.zeros(S)
    F1 = np.zeros(S)
    Pm = np.zeros((S, S))
    for b in range(B):
        for s0 in range(n0):
            labs0[b, s0] = slot_label(kinds[0], parents[0], s0, X, b)
        for s1 in range(n1):
            labs1[b, s1] = slot_label(kinds[1], parents[1], s1, X, b)
        for g in range(G):
            hp = subtree_terms(n, b, g, E, X, V, par, desc, mark, F0, F1, Pm)
            for s0 in range(n0):
                l0 = labs0[b, s0]
                for s1 in range(n1):
                    l1 = labs1[b, s1]
                    e = E[n, l0, l1, b] + F0[l0] + F1[l1]
                    if hp:
                        e += Pm[l0, l1]
                    if b == 0:
                        e += slot_init(kinds[0], s0, init_copy, init_bottom) + slot_init(
                            kinds[1], s1, init_copy, init_bottom
                        )
                    em[b, s0, s1, g] = e
        if b == 0:
            continue
        for r in range(n0):
            for s in range(n0):
                T0[b, r, s] = slot_log_transition(kinds[0], r, s, b, lstay, lmove, mstay, mmove, ain, astay, bout, S)
        for r in range(n1):
            for s in range(n1):
                T1[b, r, s] = slot_log_transition(kinds[1], r, s, b, lstay, lmove, mstay, mmove, ain, astay, bout, S)
        if G == 2:
            for h in range(2):
                for g in range(2):
                    TG[b, h, g] = gauge_children_terms(b, h, g, V, child_tracks, mstay, mmove)
    return em, T0, T1, TG, labs0, labs1


@kernel(overloads=_OVERLOADS["_step"], cache=True, nogil=True)
def _step(prev, T0b, T1b, TGb, G):
    """log sum over predecessors of prev[r0, r1, h] + T0[r0, s0] + T1[r1, s1] + TG[h, g] -> out[s0, s1, g]."""
    n0, n1 = prev.shape[0], prev.shape[1]
    A = np.full((n0, n1, G), NEG)
    for s0 in range(n0):
        for r1 in range(n1):
            for h in range(G):
                acc = NEG
                for r0 in range(n0):
                    acc = _lse(acc, prev[r0, r1, h] + T0b[r0, s0])
                A[s0, r1, h] = acc
    Bm = np.full((n0, n1, G), NEG)
    for s0 in range(n0):
        for s1 in range(n1):
            for h in range(G):
                acc = NEG
                for r1 in range(n1):
                    acc = _lse(acc, A[s0, r1, h] + T1b[r1, s1])
                Bm[s0, s1, h] = acc
    out = np.full((n0, n1, G), NEG)
    for s0 in range(n0):
        for s1 in range(n1):
            for g in range(G):
                acc = NEG
                for h in range(G):
                    acc = _lse(acc, Bm[s0, s1, h] + TGb[h, g])
                out[s0, s1, g] = acc
    return out


@kernel(overloads=_OVERLOADS["_step_back"], cache=True, nogil=True)
def _step_back(nxt, T0b, T1b, TGb, G):
    """beta[r0, r1, h] = log sum over successors of T0[r0, s0] + T1[r1, s1] + TG[h, g] + nxt[s0, s1, g]."""
    n0, n1 = nxt.shape[0], nxt.shape[1]
    A = np.full((n0, n1, G), NEG)
    for s0 in range(n0):
        for s1 in range(n1):
            for h in range(G):
                acc = NEG
                for g in range(G):
                    acc = _lse(acc, nxt[s0, s1, g] + TGb[h, g])
                A[s0, s1, h] = acc
    Bm = np.full((n0, n1, G), NEG)
    for s0 in range(n0):
        for r1 in range(n1):
            for h in range(G):
                acc = NEG
                for s1 in range(n1):
                    acc = _lse(acc, A[s0, s1, h] + T1b[r1, s1])
                Bm[s0, r1, h] = acc
    out = np.full((n0, n1, G), NEG)
    for r0 in range(n0):
        for r1 in range(n1):
            for h in range(G):
                acc = NEG
                for s0 in range(n0):
                    acc = _lse(acc, Bm[s0, r1, h] + T0b[r0, s0])
                out[r0, r1, h] = acc
    return out


@kernel(overloads=_OVERLOADS["forward_backward"], cache=True, nogil=True)
def forward_backward(em, T0, T1, TG, anchor_bin, anchor_mask):
    """Posteriors post[b, s0, s1] (gauge summed). If anchor_bin >= 0, the state at that bin is restricted to
    anchor_mask[s0, s1] (True = allowed). The gauge starts at 0, as in the Viterbi block."""
    B, n0, n1, G = em.shape
    alpha = np.full((B, n0, n1, G), NEG)
    for s0 in range(n0):
        for s1 in range(n1):
            alpha[0, s0, s1, 0] = em[0, s0, s1, 0]
    if anchor_bin == 0:
        for s0 in range(n0):
            for s1 in range(n1):
                if not anchor_mask[s0, s1]:
                    for g in range(G):
                        alpha[0, s0, s1, g] = NEG
    for b in range(1, B):
        step = _step(alpha[b - 1], T0[b], T1[b], TG[b], G)
        for s0 in range(n0):
            for s1 in range(n1):
                for g in range(G):
                    v = step[s0, s1, g] + em[b, s0, s1, g]
                    if b == anchor_bin and not anchor_mask[s0, s1]:
                        v = NEG
                    alpha[b, s0, s1, g] = v
    beta = np.zeros((B, n0, n1, G))
    for b in range(B - 2, -1, -1):
        nxt = np.empty((n0, n1, G))
        for s0 in range(n0):
            for s1 in range(n1):
                for g in range(G):
                    v = beta[b + 1, s0, s1, g] + em[b + 1, s0, s1, g]
                    if b + 1 == anchor_bin and not anchor_mask[s0, s1]:
                        v = NEG
                    nxt[s0, s1, g] = v
        beta[b] = _step_back(nxt, T0[b + 1], T1[b + 1], TG[b + 1], G)
    return posterior_bins(alpha, beta)


@kernel(overloads=_OVERLOADS["posterior_bins"], cache=True, nogil=True)
def posterior_bins(alpha, beta):
    """Posteriors post[b, s0, s1] (gauge summed) of the bins whose forward and backward messages are alpha[b] and
    beta[b] ([m, n0, n1, G] each); every bin is normalised by its own total."""
    B, n0, n1, G = alpha.shape
    post = np.zeros((B, n0, n1))
    for b in range(B):
        z = NEG
        for s0 in range(n0):
            for s1 in range(n1):
                for g in range(G):
                    z = _lse(z, alpha[b, s0, s1, g] + beta[b, s0, s1, g])
        for s0 in range(n0):
            for s1 in range(n1):
                acc = 0.0
                for g in range(G):
                    v = alpha[b, s0, s1, g] + beta[b, s0, s1, g] - z
                    if v > POSTERIOR_LOG_FLOOR:
                        acc += math.exp(v)
                post[b, s0, s1] = acc
    return post


# The chain restricted at one bin shares most of its messages with the unrestricted chain: the restriction acts on the
# state at the anchor bin only, so alpha before the anchor and beta at and after it are the unrestricted ones. The four
# kernels below give the others where they are needed, by the recursion of forward_backward (the same operations in
# the same order: the posteriors assembled from them are those of forward_backward with that anchor).
@kernel(overloads=_OVERLOADS["chain_messages"], cache=True, nogil=True)
def chain_messages(em, T0, T1, TG):
    """(alpha, beta), [B, n0, n1, G] each: the forward and backward messages of the unrestricted chain
    (forward_backward's, anchor_bin = -1). The gauge starts at 0."""
    B, n0, n1, G = em.shape
    alpha = np.full((B, n0, n1, G), NEG)
    for s0 in range(n0):
        for s1 in range(n1):
            alpha[0, s0, s1, 0] = em[0, s0, s1, 0]
    for b in range(1, B):
        step = _step(alpha[b - 1], T0[b], T1[b], TG[b], G)
        for s0 in range(n0):
            for s1 in range(n1):
                for g in range(G):
                    alpha[b, s0, s1, g] = step[s0, s1, g] + em[b, s0, s1, g]
    beta = np.zeros((B, n0, n1, G))
    for b in range(B - 2, -1, -1):
        nxt = np.empty((n0, n1, G))
        for s0 in range(n0):
            for s1 in range(n1):
                for g in range(G):
                    nxt[s0, s1, g] = beta[b + 1, s0, s1, g] + em[b + 1, s0, s1, g]
        beta[b] = _step_back(nxt, T0[b + 1], T1[b + 1], TG[b + 1], G)
    return alpha, beta


@kernel(overloads=_OVERLOADS["restricted"], cache=True, nogil=True)
def restricted(state, anchor_mask):
    """The forward message ``state`` [n0, n1, G] of a bin with every (s0, s1) outside anchor_mask made impossible:
    alpha at the anchor bin of the chain restricted there."""
    n0, n1, G = state.shape
    out = state.copy()
    for s0 in range(n0):
        for s1 in range(n1):
            if not anchor_mask[s0, s1]:
                for g in range(G):
                    out[s0, s1, g] = NEG
    return out


@kernel(overloads=_OVERLOADS["forward_from"], cache=True, nogil=True)
def forward_from(prev, em, T0, T1, TG, lo, hi):
    """alpha [hi - lo, n0, n1, G] of the bins [lo, hi), continuing the forward recursion from prev = alpha[lo - 1]."""
    n0, n1, G = prev.shape
    out = np.empty((hi - lo, n0, n1, G))
    last = prev
    for b in range(lo, hi):
        step = _step(last, T0[b], T1[b], TG[b], G)
        for s0 in range(n0):
            for s1 in range(n1):
                for g in range(G):
                    out[b - lo, s0, s1, g] = step[s0, s1, g] + em[b, s0, s1, g]
        last = out[b - lo]
    return out


@kernel(overloads=_OVERLOADS["backward_to"], cache=True, nogil=True)
def backward_to(beta_anchor, em, T0, T1, TG, lo, anchor, anchor_mask):
    """beta [anchor - lo, n0, n1, G] of the bins [lo, anchor) of the chain whose state at ``anchor`` is restricted to
    anchor_mask, from beta_anchor = beta at the anchor bin (the unrestricted chain's)."""
    n0, n1, G = beta_anchor.shape
    out = np.empty((anchor - lo, n0, n1, G))
    for b in range(anchor - 1, lo - 1, -1):
        nxt = np.empty((n0, n1, G))
        for s0 in range(n0):
            for s1 in range(n1):
                for g in range(G):
                    if b + 1 == anchor:
                        v = beta_anchor[s0, s1, g] + em[anchor, s0, s1, g]
                        if not anchor_mask[s0, s1]:
                            v = NEG
                    else:
                        v = out[b + 1 - lo, s0, s1, g] + em[b + 1, s0, s1, g]
                    nxt[s0, s1, g] = v
        out[b - lo] = _step_back(nxt, T0[b + 1], T1[b + 1], TG[b + 1], G)
    return out


@kernel(overloads=_OVERLOADS["_flip_pair"], cache=True, nogil=True)
def _flip_pair(f1x, f1y, keep, epsilon, g0, g1, g2, weight, out, i):
    """Adds weight x the ordered allele posterior of one label pair (P(homologue 0 carries allele 1) = f1x, homologue
    1: f1y), with independent per-homologue flips, given the site's reads to out[:, i] (site_posteriors' law)."""
    w00 = (keep * (1.0 - f1x) * (1.0 - f1y) + epsilon / 3.0) * g0
    w01 = (keep * (1.0 - f1x) * f1y + epsilon / 6.0) * g1
    w10 = (keep * f1x * (1.0 - f1y) + epsilon / 6.0) * g1
    w11 = (keep * f1x * f1y + epsilon / 3.0) * g2
    z = w00 + w01 + w10 + w11
    if z <= 0.0:
        return
    f = weight / z
    out[0, i] += f * w00
    out[1, i] += f * w01
    out[2, i] += f * w10
    out[3, i] += f * w11


@kernel(overloads=_OVERLOADS["site_posteriors"], cache=True, nogil=True)
def site_posteriors(P, site_bin, alleles, gl, epsilon, flips, own_state, own_flip):
    """float64 [4, m]: ordered allele posteriors (00, 01, 10, 11) of each site under the release's site model.

    Given the bin's ordered label pair (x, y) (posterior P [B, S, S]) with panel alleles a_x, a_y (alleles [S, m]; a
    pair with an unknown allele, -1, contributes to none, so its mass leaves the call undecided), the sample's ordered
    alleles (u, v) have probability (1 - epsilon) f_x(u) f_y(v) + epsilon U(u, v): each homologue carries its label's
    panel allele except for a flip of that allele with probability d_x (f_x(a_x) = 1 - d_x, f_x(1 - a_x) = d_x: the
    capped empirical-Bayes per-copy uncertainty conditional on the fitted carriers and panel, allele_evidence.py),
    and with mixture mass
    epsilon the pair is redrawn from a uniform genotype (ordered weights 1/3, 1/6, 1/6, 1/3: a label error shorter than
    a bin, a conversion or mutation; the mass is not itself the probability of a change). d_x is flips[x, i], except
    for the labels of the sample's own MAP pair at the site (own_state [2, m], -1 none), which take own_flip [2, m]
    (their flips with the sample's direct own likelihood term removed). The selected allele determines the cap.
    The copies flip independently per homologue, including (x, x); this is not exact integration over one shared
    latent allele. Carrier and panel selection are not integrated out, and this is not a cross-fitted refit.
    With every d = 0 the selected panel alleles have no flip uncertainty. The sample's own
    site likelihood gl [m, 3] (P(reads | genotype), flat where unobserved) weighs the four ordered states given each
    pair, normalised per pair; the direct own term enters here, while indirect selection dependence remains:
    out[:, i] = sum over known pairs of
    P[b, x, y] P(alleles | x, y, reads).
    A pair's normalised posterior depends only on its two homologues' allele-1 probabilities, so pairs are summed by
    them before normalising: pairs of two labels without a flip by their allele pattern (four normalisations per
    site), pairs with one flipped label by that label and the other's allele (two per flipped label and side), and
    only pairs of two flipped labels one by one. O(m S^2) additions and O(m (4 + 4 F + F^2)) normalisations for F
    flipped labels at the site.
    """
    S = P.shape[1]
    m = site_bin.shape[0]
    out = np.zeros((4, m))
    mass = np.zeros(4)
    flip = np.zeros(S)
    side0 = np.zeros((S, 2))  # flipped label on homologue 0, mass by homologue 1's allele
    side1 = np.zeros((S, 2))  # flipped label on homologue 1, mass by homologue 0's allele
    keep = 1.0 - epsilon
    for i in range(m):
        b = site_bin[i]
        mass[:] = 0.0
        g0 = float(gl[i, 0])
        g1 = float(gl[i, 1])
        g2 = float(gl[i, 2])
        for x in range(S):
            flip[x] = flips[x, i]
        for t in range(2):
            if own_state[t, i] >= 0:
                flip[own_state[t, i]] = own_flip[t, i]
        side0[:] = 0.0
        side1[:] = 0.0
        for x in range(S):
            ax = alleles[x, i]
            if ax < 0:
                continue
            dx = flip[x]
            for y in range(S):
                ay = alleles[y, i]
                if ay < 0:
                    continue
                dy = flip[y]
                p = P[b, x, y]
                if dx == 0.0:
                    if dy == 0.0:
                        mass[2 * ax + ay] += p
                    else:
                        side1[y, ax] += p
                elif dy == 0.0:
                    side0[x, ay] += p
                elif p != 0.0:
                    _flip_pair(
                        dx if ax == 0 else 1.0 - dx, dy if ay == 0 else 1.0 - dy, keep, epsilon, g0, g1, g2, p, out, i
                    )
        for x in range(S):
            dx = flip[x]
            if dx == 0.0 or alleles[x, i] < 0:
                continue
            f1 = dx if alleles[x, i] == 0 else 1.0 - dx  # P(the flipped label's homologue carries allele 1)
            for c in range(2):
                if side0[x, c] != 0.0:
                    _flip_pair(f1, float(c), keep, epsilon, g0, g1, g2, side0[x, c], out, i)
                if side1[x, c] != 0.0:
                    _flip_pair(float(c), f1, keep, epsilon, g0, g1, g2, side1[x, c], out, i)
        for h in range(4):
            if mass[h] == 0.0:
                continue
            w00 = (epsilon / 3.0 + (keep if h == 0 else 0.0)) * g0
            w01 = (epsilon / 6.0 + (keep if h == 1 else 0.0)) * g1
            w10 = (epsilon / 6.0 + (keep if h == 2 else 0.0)) * g1
            w11 = (epsilon / 3.0 + (keep if h == 3 else 0.0)) * g2
            z = w00 + w01 + w10 + w11
            if z <= 0.0:
                continue
            f = mass[h] / z
            out[0, i] += f * w00
            out[1, i] += f * w01
            out[2, i] += f * w10
            out[3, i] += f * w11
    return out


@kernel(overloads=_OVERLOADS["label_pairs"], cache=True, nogil=True)
def label_pairs(post, labels0, labels1, S):
    """float64 [B, S, S]: the ordered label-pair posteriors from the block-state posteriors post [B, n0, n1], each
    slot state's label per bin in labels0 [B, n0] and labels1 [B, n1]; states are added in their order."""
    B, n0, n1 = post.shape
    out = np.zeros((B, S, S))
    for b in range(B):
        for s0 in range(n0):
            for s1 in range(n1):
                out[b, labels0[b, s0], labels1[b, s1]] += post[b, s0, s1]
    return out


@kernel(overloads=_OVERLOADS["orientation"], cache=True, nogil=True)
def orientation(ordered, site_bin, tau_g, B):
    """float64 [B]: per bin, the median over its sites (``site_bin``, a bin's sites consecutive) with P(heterozygous)
    >= tau_g under the ordered allele posteriors ``ordered`` [4, m] of the orientation certainty max(P01, P10) /
    (P01 + P10); NaN in a bin without such a site."""
    out = np.full(B, np.nan)
    certainty = np.empty(len(site_bin))
    used = 0
    last = -1
    for i in range(len(site_bin)):
        het = ordered[1, i] + ordered[2, i]
        if het >= tau_g:
            b = site_bin[i]
            if b != last and used:
                out[last] = np.median(certainty[:used])
                used = 0
            last = b
            certainty[used] = max(ordered[1, i], ordered[2, i]) / het
            used += 1
    if used:
        out[last] = np.median(certainty[:used])
    return out


__all__ = [
    "backward_to",
    "block_tables",
    "chain_messages",
    "forward_backward",
    "forward_from",
    "label_pairs",
    "orientation",
    "posterior_bins",
    "restricted",
    "site_posteriors",
]
