"""Numba kernels of the family-phase model (docs/methods/family_phase.md).

Owns FP03-FP04 and FP15-FP16 recurrences; must not read storage or schedule work.
Style exception: hot recurrences remain whole, preserving the three-factor
contractions and accumulator grouping; the axis letters are the model's notation.


One component's model: N samples (rows of E, X, V; latent nodes would be rows without reads), B painting bins, S
labels (the painting states: trajectory classes + BACKGROUND).
  E float64 [N, S, S, B]   the painting's tempered log emission of an ordered label pair (symmetric in the pair)
  X int64   [N, 2, B]      label of each homologue
  V int8    [N, 2, B]      a linked homologue's source: 0/1 = copy of that parent homologue, 2 = not a copy
  par int64 [N, 2]         slot-k parent (-1 = no linked parent: the homologue is a free label path)
Per bin edge e (between bins e and e + 1), log probabilities: lstay/lmove (a free label keeps its label / moves to one
particular other label), mstay/mmove (a meiosis chain keeps / changes the transmitted parental homologue), ain/astay
(a linked homologue enters / stays out of the not-a-copy state), bout (it leaves it, to either parental homologue with
1/2 each). Entering not-a-copy draws its label uniformly, so a block's change of a parent's labels never alters a
descendant's transition terms (blocks are exact). A linked homologue starts (bin 0) as a copy of either parental
homologue with log probability init_copy each, or not a copy with label x with init_bottom each (the chain's
stationary not-a-copy mass spread over the labels); a free label starts uniformly (a constant, omitted).

A sample's block is the chain of (source of homologue 0, source of homologue 1, gauge g of its children): g = 1 reads
every child's chain for this sample flipped from that bin on, so one exact HMM contains every "relabel this sample's
homologues from here" move. block_viterbi (max-product) and block_tables + forward_backward (sum-product) evaluate
the same factors; the transition is the product of the three factors, contracted one factor at a time.
"""

import math

import numpy as np
from numba import types

from ...core.kernels import kernel
from .parameters import IMPOSSIBLE_LOG_WEIGHT

NEG = IMPOSSIBLE_LOG_WEIGHT


# Declared kernel signatures; all calls remain lazily compiled.
_OVERLOADS = {
    "subtree_terms": (
        (
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.float64, 4, "C", False, aligned=True),
            types.Array(types.int64, 3, "C", False, aligned=True),
            types.Array(types.int8, 3, "C", False, aligned=True),
            types.Array(types.int64, 2, "C", False, aligned=True),
            types.Array(types.int64, 1, "C", False, aligned=True),
            types.Array(types.int8, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 2, "C", False, aligned=True),
        ),
    ),
    "slot_label": (
        (types.int64, types.int64, types.int64, types.Array(types.int64, 3, "C", False, aligned=True), types.int64),
    ),
    "slot_log_transition": (
        (
            types.literal(1),
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.int64,
        ),
        (
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.int64,
        ),
    ),
    "slot_init": ((types.int64, types.int64, types.float64, types.float64),),
    "gauge_children_terms": (
        (
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.int8, 3, "C", False, aligned=True),
            types.Array(types.int64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
            types.Array(types.float64, 1, "C", False, aligned=True),
        ),
    ),
    "block_viterbi": (
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
            types.Array(types.int32, 1, "C", False, aligned=True),
            types.Array(types.int32, 1, "C", False, aligned=True),
            types.float64,
            types.boolean,
        ),
    ),
    "objective": (
        (
            types.Array(types.float64, 4, "C", False, aligned=True),
            types.Array(types.int64, 3, "C", False, aligned=True),
            types.Array(types.int8, 3, "C", False, aligned=True),
            types.Array(types.int64, 2, "C", False, aligned=True),
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
    "row_terms": (
        (
            types.Array(types.float64, 4, "C", False, aligned=True),
            types.Array(types.int64, 3, "C", False, aligned=True),
            types.Array(types.int8, 3, "C", False, aligned=True),
            types.Array(types.int64, 2, "C", False, aligned=True),
            types.Array(types.int64, 1, "C", False, aligned=True),
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
    "evidence_terms": (
        (
            types.Array(types.int64, 2, "C", False, aligned=True),
            types.Array(types.int64, 2, "C", False, aligned=True),
            types.Array(types.int8, 2, "C", False, aligned=True),
            types.Array(types.float64, 3, "C", False, aligned=True),
            types.Array(types.float64, 2, "C", False, aligned=True),
        ),
    ),
}


@kernel(overloads=_OVERLOADS["evidence_terms"], cache=True, nogil=True)
def evidence_terms(lab0, lab1, alleles, lg, out):
    """Carrier sums for one site chunk, added into zero-initialized ``out`` float64 [S, c].
    lab0, lab1 int64 [R, c] are the carriers' MAP labels; alleles int8 [S, c]; lg float64 [R, c, 3] contains robust
    log likelihoods. The contrasts implement the same law as allele_evidence._terms. Each site's cells receive
    homologue-0 terms of rows 0..R-1, then homologue-1 terms of rows 0..R-1. The kernel releases the GIL so
    disjoint site chunks can run concurrently."""
    R, c = lab0.shape
    for j in range(c):
        for r in range(R):
            x0 = lab0[r, j]
            x1 = lab1[r, j]
            a0 = alleles[x0, j]
            a1 = alleles[x1, j]
            same = x0 == x1
            if a0 >= 0 and (same or a1 >= 0):
                zero = lg[r, j, 0]
                one = lg[r, j, 1]
                two = lg[r, j, 2]
                if same:
                    out[x0, j] += two - zero
                elif a1 >= 1:
                    out[x0, j] += two - one
                else:
                    out[x0, j] += one - zero
        for r in range(R):
            x0 = lab0[r, j]
            x1 = lab1[r, j]
            a0 = alleles[x0, j]
            a1 = alleles[x1, j]
            if a1 >= 0 and a0 >= 0 and x0 != x1:
                one = lg[r, j, 1]
                if a0 >= 1:
                    out[x1, j] += lg[r, j, 2] - one
                else:
                    out[x1, j] += one - lg[r, j, 0]


@kernel(overloads=_OVERLOADS["subtree_terms"], cache=True, nogil=True)
def subtree_terms(n, b, g, E, X, V, par, desc, mark, F0, F1, P):
    """F0[x], F1[x], P[x0, x1]: summed emissions at bin b of n's descendant samples whose homologues copy n's
    homologue 0 / 1 (as functions of that homologue's label), with n's children's chains read through gauge g.
    ``desc`` = n's descendant samples in topological order; ``mark`` int8[2N] scratch (-1 outside)."""
    S = F0.shape[0]
    for x in range(S):
        F0[x] = 0.0
        F1[x] = 0.0
    has_pair = False
    for i in range(len(desc)):
        m = desc[i]
        for k in range(2):
            t = 2 * m + k
            p = par[m, k]
            v = V[m, k, b]
            if p < 0 or v > 1:
                mark[t] = -1
            elif p == n:
                mark[t] = v ^ g
            else:
                mark[t] = mark[2 * p + v]
        a = mark[2 * m]
        c = mark[2 * m + 1]
        if a < 0 and c < 0:
            continue
        la = X[m, 0, b]
        lc = X[m, 1, b]
        if a >= 0 and c < 0:
            for x in range(S):
                if a == 0:
                    F0[x] += E[m, x, lc, b]
                else:
                    F1[x] += E[m, x, lc, b]
        elif a < 0 and c >= 0:
            for x in range(S):
                if c == 0:
                    F0[x] += E[m, la, x, b]
                else:
                    F1[x] += E[m, la, x, b]
        elif a == c:
            for x in range(S):
                if a == 0:
                    F0[x] += E[m, x, x, b]
                else:
                    F1[x] += E[m, x, x, b]
        else:
            if not has_pair:
                for x0 in range(S):
                    for x1 in range(S):
                        P[x0, x1] = 0.0
                has_pair = True
            for x0 in range(S):
                for x1 in range(S):
                    if a == 0:
                        P[x0, x1] += E[m, x0, x1, b]
                    else:
                        P[x0, x1] += E[m, x1, x0, b]
    for i in range(len(desc)):
        m = desc[i]
        mark[2 * m] = -1
        mark[2 * m + 1] = -1
    return has_pair


@kernel(overloads=_OVERLOADS["slot_label"], cache=True, nogil=True)
def slot_label(kind, p, s, X, b):
    """Label of a homologue in source state s: observed slot (kind 1): s in {0,1} -> parent's homologue s,
    s >= 2 -> bottom with label s - 2; free slot (kind 0): label s."""
    if kind == 1:
        if s < 2:
            return X[p, s, b]
        return s - 2
    return s


@kernel(overloads=_OVERLOADS["slot_log_transition"], cache=True, nogil=True)
def slot_log_transition(kind, s, t, b, lstay, lmove, mstay, mmove, ain, astay, bout, nlabels):
    """log P(source s at bin b-1 -> t at bin b) for one homologue slot (b >= 1)."""
    if kind == 0:
        return lstay[b - 1] if s == t else lmove[b - 1]
    if s < 2 and t < 2:
        return astay[b - 1] + (mstay[b - 1] if s == t else mmove[b - 1])
    if s < 2 and t >= 2:
        # entering bottom: the free label is drawn uniformly (independent of the copied label, so that a block's
        # change of a parent's labels never alters its descendants' transition terms)
        return ain[b - 1] - math.log(nlabels)
    if s >= 2 and t >= 2:
        return math.log(1.0 - math.exp(bout[b - 1])) + (lstay[b - 1] if s == t else lmove[b - 1])
    return bout[b - 1] + math.log(0.5)


@kernel(overloads=_OVERLOADS["slot_init"], cache=True, nogil=True)
def slot_init(kind, s, init_copy, init_bottom):
    """Initial log probability of slot state s at bin 0 (0 for a free slot: its uniform start is a constant)."""
    if kind == 0:
        return 0.0
    return init_copy if s < 2 else init_bottom


@kernel(overloads=_OVERLOADS["gauge_children_terms"], cache=True, nogil=True)
def gauge_children_terms(b, g0, g1, V, child_tracks, mstay, mmove):
    """Sum over the supplied child tracks of the chain transition term at edge (b-1, b) with the children's chains read
    through gauge g0 at b-1 and g1 at b (only 0/1 -> 0/1 transitions depend on the gauge)."""
    total = 0.0
    for i in range(len(child_tracks)):
        t = child_tracks[i]
        c = t // 2
        k = t % 2
        v0 = V[c, k, b - 1]
        v1 = V[c, k, b]
        if v0 > 1 or v1 > 1:
            continue
        total += mstay[b - 1] if (v0 ^ g0) == (v1 ^ g1) else mmove[b - 1]
    return total


@kernel(overloads=_OVERLOADS["block_viterbi"], cache=True, nogil=True)
def block_viterbi(
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
    cur0,
    cur1,
    tie,
    rev,
):
    """Exact factorised max-product for one sample block: phase.block.viterbi.

    Tie rule:
    in each of the three stages the predecessor is scanned in
    ascending index order (descending when `rev`); with `best = NEG`, `arg = 0`, a candidate replaces the incumbent when
    `val > best + tie` or (`|val − best| <= tie` and the candidate is the slot's current state at bin b − 1 (`c0`/`c1`)
    and `arg` is not already it); stage 3's preferred predecessor is `h == g`. `best` is the **retained** argument's
    value, not the stage maximum (drift ≤ `tie` per stage, can chain across bins).
    At the last bin the scan over (s0, s1,
    g) (each index reversed when `rev`) keeps the earliest strict improvement beyond `tie`, and (cur0[B−1], cur1[B−1],
    g = 0) wins any tie. `tie` = `tie_quantum` 1e-9 nats; `NEG` = −1e300.

    Bin zero permits only gauge 0, even with children. Descendant scratch marks
    are cleared by subtree_terms at every call. Returns path0, path1, gpath and
    retained score; int16 source and int8 gauge back-pointers accommodate K30.
    This incumbent policy belongs here; quantised class policies are different.
    """
    B = E.shape[3]
    n0 = S + 2 if kinds[0] == 1 else S
    n1 = S + 2 if kinds[1] == 1 else S
    G = 2 if len(child_tracks) > 0 else 1
    score = np.full((n0, n1, G), NEG)
    em = np.empty((n0, n1, G))
    argA = np.empty((B, n0, n1, G), dtype=np.int16)
    argB = np.empty((B, n0, n1, G), dtype=np.int16)
    argC = np.empty((B, n0, n1, G), dtype=np.int8)
    A = np.empty((n0, n1, G))
    Bm = np.empty((n0, n1, G))
    T0 = np.empty((n0, n0))
    T1 = np.empty((n1, n1))
    TG = np.zeros((2, 2))
    F0 = np.zeros((2, S))
    F1 = np.zeros((2, S))
    Pm = np.zeros((2, S, S))
    hp = np.zeros(2, dtype=np.bool_)
    tmpF0 = np.zeros(S)
    tmpF1 = np.zeros(S)
    tmpP = np.zeros((S, S))
    lab0 = np.empty(n0, dtype=np.int64)
    lab1 = np.empty(n1, dtype=np.int64)
    for b in range(B):
        for g in range(G):
            hp[g] = subtree_terms(n, b, g, E, X, V, par, desc, mark, tmpF0, tmpF1, tmpP)
            for x in range(S):
                F0[g, x] = tmpF0[x]
                F1[g, x] = tmpF1[x]
            if hp[g]:
                for x0 in range(S):
                    for x1 in range(S):
                        Pm[g, x0, x1] = tmpP[x0, x1]
        for s0 in range(n0):
            lab0[s0] = slot_label(kinds[0], parents[0], s0, X, b)
        for s1 in range(n1):
            lab1[s1] = slot_label(kinds[1], parents[1], s1, X, b)
        for s0 in range(n0):
            l0 = lab0[s0]
            for s1 in range(n1):
                l1 = lab1[s1]
                for g in range(G):
                    e = E[n, l0, l1, b] + F0[g, l0] + F1[g, l1]
                    if hp[g]:
                        e += Pm[g, l0, l1]
                    em[s0, s1, g] = e
        if b == 0:
            for s0 in range(n0):
                i0 = slot_init(kinds[0], s0, init_copy, init_bottom)
                for s1 in range(n1):
                    i1 = slot_init(kinds[1], s1, init_copy, init_bottom)
                    for g in range(G):
                        score[s0, s1, g] = em[s0, s1, g] + i0 + i1 if g == 0 else NEG
            continue
        for r0 in range(n0):
            for s0 in range(n0):
                T0[r0, s0] = slot_log_transition(kinds[0], r0, s0, b, lstay, lmove, mstay, mmove, ain, astay, bout, S)
        for r1 in range(n1):
            for s1 in range(n1):
                T1[r1, s1] = slot_log_transition(kinds[1], r1, s1, b, lstay, lmove, mstay, mmove, ain, astay, bout, S)
        if G == 2:
            for h in range(2):
                for g in range(2):
                    TG[h, g] = gauge_children_terms(b, h, g, V, child_tracks, mstay, mmove)
        c0 = cur0[b - 1]
        c1 = cur1[b - 1]
        # stage 1: previous source 0
        for s0 in range(n0):
            for r1 in range(n1):
                for h in range(G):
                    best = NEG
                    arg = 0
                    for i in range(n0):
                        r0 = n0 - 1 - i if rev else i
                        val = score[r0, r1, h] + T0[r0, s0]
                        preferred = r0 == c0 and arg != c0
                        taken = val > best + tie or (abs(val - best) <= tie and preferred)
                        if taken:
                            best = val
                            arg = r0
                    A[s0, r1, h] = best
                    argA[b, s0, r1, h] = arg
        # stage 2: previous source 1
        for s0 in range(n0):
            for s1 in range(n1):
                for h in range(G):
                    best = NEG
                    arg = 0
                    for i in range(n1):
                        r1 = n1 - 1 - i if rev else i
                        val = A[s0, r1, h] + T1[r1, s1]
                        preferred = r1 == c1 and arg != c1
                        taken = val > best + tie or (abs(val - best) <= tie and preferred)
                        if taken:
                            best = val
                            arg = r1
                    Bm[s0, s1, h] = best
                    argB[b, s0, s1, h] = arg
        # stage 3: previous gauge
        for s0 in range(n0):
            for s1 in range(n1):
                for g in range(G):
                    best = NEG
                    arg = 0
                    for i in range(G):
                        h = G - 1 - i if rev else i
                        val = Bm[s0, s1, h] + TG[h, g]
                        preferred = h == g and arg != g
                        taken = val > best + tie or (abs(val - best) <= tie and preferred)
                        if taken:
                            best = val
                            arg = h
                    score[s0, s1, g] = best + em[s0, s1, g]
                    argC[b, s0, s1, g] = arg
    best = NEG
    a0 = 0
    a1 = 0
    ag = 0
    for i0 in range(n0):
        s0 = n0 - 1 - i0 if rev else i0
        for i1 in range(n1):
            s1 = n1 - 1 - i1 if rev else i1
            for ig in range(G):
                g = G - 1 - ig if rev else ig
                val = score[s0, s1, g]
                keep = s0 == cur0[B - 1] and s1 == cur1[B - 1] and g == 0
                preferred = keep
                taken = val > best + tie or (abs(val - best) <= tie and preferred)
                if taken:
                    best = val
                    a0 = s0
                    a1 = s1
                    ag = g
    path0 = np.empty(B, dtype=np.int32)
    path1 = np.empty(B, dtype=np.int32)
    gpath = np.empty(B, dtype=np.int32)
    s0, s1, g = a0, a1, ag
    for b in range(B - 1, -1, -1):
        path0[b] = s0
        path1[b] = s1
        gpath[b] = g
        if b == 0:
            break
        h = argC[b, s0, s1, g]
        r1 = argB[b, s0, s1, h]
        r0 = argA[b, s0, r1, h]
        s0, s1, g = r0, r1, h
    return path0, path1, gpath, best


@kernel(overloads=_OVERLOADS["objective"], cache=True, nogil=True)
def objective(E, X, V, par, lstay, lmove, mstay, mmove, ain, astay, bout, init_copy, init_bottom):
    """The joint log posterior (up to constants) of the current configuration: the sum of every sample's terms
    (row_terms), O(N B). The ascent evaluates moves by row_terms of the samples they change; this is its reference."""
    N = X.shape[0]
    B = X.shape[2]
    total = 0.0
    for n in range(N):
        for b in range(B):
            total += E[n, X[n, 0, b], X[n, 1, b], b]
        for k in range(2):
            p = par[n, k]
            if p >= 0:
                total += init_copy if V[n, k, 0] < 2 else init_bottom
            for b in range(1, B):
                if p < 0:
                    total += lstay[b - 1] if X[n, k, b] == X[n, k, b - 1] else lmove[b - 1]
                else:
                    s = V[n, k, b - 1]
                    t = V[n, k, b]
                    ss = s if s < 2 else 2 + X[n, k, b - 1]
                    tt = t if t < 2 else 2 + X[n, k, b]
                    total += slot_log_transition(1, ss, tt, b, lstay, lmove, mstay, mmove, ain, astay, bout, E.shape[1])
    return total


@kernel(overloads=_OVERLOADS["row_terms"], cache=True, nogil=True)
def row_terms(E, X, V, par, rows, lstay, lmove, mstay, mmove, ain, astay, bout, init_copy, init_bottom):
    """float64 [R]: the objective's terms of samples ``rows`` [R] in the configuration X int64 [R, 2, B], V int8
    [R, 2, B] (row i holds sample rows[i]): its emissions, its homologues' label or chain transitions and its linked
    homologues' initial terms. A sample's terms read only its own labels and chains (a linked homologue's transitions
    depend on its source states, not on the copied labels: ⊥ entry draws a uniform label), so J is the sum of every
    sample's terms and a move changes J by the change in the terms of the samples whose labels or chains it alters.
    O(R B)."""
    R = rows.shape[0]
    B = X.shape[2]
    out = np.zeros(R)
    for i in range(R):
        n = rows[i]
        total = 0.0
        for b in range(B):
            total += E[n, X[i, 0, b], X[i, 1, b], b]
        for k in range(2):
            p = par[n, k]
            if p >= 0:
                total += init_copy if V[i, k, 0] < 2 else init_bottom
            for b in range(1, B):
                if p < 0:
                    total += lstay[b - 1] if X[i, k, b] == X[i, k, b - 1] else lmove[b - 1]
                else:
                    s = V[i, k, b - 1]
                    t = V[i, k, b]
                    ss = s if s < 2 else 2 + X[i, k, b - 1]
                    tt = t if t < 2 else 2 + X[i, k, b]
                    total += slot_log_transition(1, ss, tt, b, lstay, lmove, mstay, mmove, ain, astay, bout, E.shape[1])
        out[i] = total
    return out


__all__ = [
    "evidence_terms",
    "subtree_terms",
    "slot_label",
    "slot_log_transition",
    "slot_init",
    "gauge_children_terms",
    "block_viterbi",
    "objective",
    "row_terms",
]
