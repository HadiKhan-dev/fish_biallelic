"""Numba kernels of the family-phase model (release 3.0 rebuild; docs/methods/family_phase.md).

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
from numba import njit

NEG = -1e300


@njit(cache=True)
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


@njit(cache=True)
def slot_label(kind, p, s, X, b):
    """Label of a homologue in source state s: observed slot (kind 1): s in {0,1} -> parent's homologue s,
    s >= 2 -> bottom with label s - 2; free slot (kind 0): label s."""
    if kind == 1:
        if s < 2:
            return X[p, s, b]
        return s - 2
    return s


@njit(cache=True)
def slot_log_transition(kind, p, s, t, X, b, lstay, lmove, mstay, mmove, ain, astay, bout, nlabels):
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


@njit(cache=True)
def slot_init(kind, s, init_copy, init_bottom):
    """Initial log probability of slot state s at bin 0 (0 for a free slot: its uniform start is a constant)."""
    if kind == 0:
        return 0.0
    return init_copy if s < 2 else init_bottom


@njit(cache=True)
def gauge_children_terms(n, b, g0, g1, V, child_tracks, mstay, mmove):
    """Sum over n's child tracks of the chain transition term at edge (b-1, b) with the children's chains read
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


@njit(cache=True)
def block_viterbi(n, kinds, parents, E, X, V, par, desc, child_tracks, mark, S,
                  lstay, lmove, mstay, mmove, ain, astay, bout, init_copy, init_bottom, cur0, cur1, tie, rev):
    """Exact Viterbi of sample n's block over (source 0, source 1, gauge) by factorised max-product (one stage per
    factor of the product transition). Returns (path0, path1, gpath, score). Ties keep the current source state
    (cur0/cur1) and gauge 0 (no change; stated priors), else the lowest index (the highest when ``rev``: the
    convention reversal, maths.ties.reverse_conventions)."""
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
                T0[r0, s0] = slot_log_transition(kinds[0], parents[0], r0, s0, X, b, lstay, lmove, mstay, mmove,
                                                 ain, astay, bout, S)
        for r1 in range(n1):
            for s1 in range(n1):
                T1[r1, s1] = slot_log_transition(kinds[1], parents[1], r1, s1, X, b, lstay, lmove, mstay, mmove,
                                                 ain, astay, bout, S)
        if G == 2:
            for h in range(2):
                for g in range(2):
                    TG[h, g] = gauge_children_terms(n, b, h, g, V, child_tracks, mstay, mmove)
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
                        if val > best + tie or (abs(val - best) <= tie and r0 == c0 and arg != c0):
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
                        if val > best + tie or (abs(val - best) <= tie and r1 == c1 and arg != c1):
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
                        if val > best + tie or (abs(val - best) <= tie and h == g and arg != g):
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
                if val > best + tie or (abs(val - best) <= tie and keep):
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


@njit(cache=True)
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
                    total += slot_log_transition(1, p, ss, tt, X, b, lstay, lmove, mstay, mmove, ain, astay, bout,
                                                 E.shape[1])
    return total


@njit(cache=True)
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
                    total += slot_log_transition(1, p, ss, tt, X, b, lstay, lmove, mstay, mmove, ain, astay, bout,
                                                 E.shape[1])
        out[i] = total
    return out


@njit(cache=True)
def _lse(a, b):
    if a < b:
        a, b = b, a
    if b <= NEG:
        return a
    return a + math.log1p(math.exp(b - a))


@njit(cache=True)
def block_tables(n, kinds, parents, E, X, V, par, desc, child_tracks, mark, S,
                 lstay, lmove, mstay, mmove, ain, astay, bout, init_copy, init_bottom):
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
                        e += slot_init(kinds[0], s0, init_copy, init_bottom) + slot_init(kinds[1], s1, init_copy,
                                                                                           init_bottom)
                    em[b, s0, s1, g] = e
        if b == 0:
            continue
        for r in range(n0):
            for s in range(n0):
                T0[b, r, s] = slot_log_transition(kinds[0], parents[0], r, s, X, b, lstay, lmove, mstay, mmove,
                                                  ain, astay, bout, S)
        for r in range(n1):
            for s in range(n1):
                T1[b, r, s] = slot_log_transition(kinds[1], parents[1], r, s, X, b, lstay, lmove, mstay, mmove,
                                                  ain, astay, bout, S)
        if G == 2:
            for h in range(2):
                for g in range(2):
                    TG[b, h, g] = gauge_children_terms(n, b, h, g, V, child_tracks, mstay, mmove)
    return em, T0, T1, TG, labs0, labs1


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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
                    if v > -700:
                        acc += math.exp(v)
                post[b, s0, s1] = acc
    return post


@njit(cache=True)
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


@njit(cache=True)
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
    With every d = 0 this is the 3.3 model. The sample's own
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
    side0 = np.zeros((S, 2))                        # flipped label on homologue 0, mass by homologue 1's allele
    side1 = np.zeros((S, 2))                        # flipped label on homologue 1, mass by homologue 0's allele
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
                    _flip_pair(dx if ax == 0 else 1.0 - dx, dy if ay == 0 else 1.0 - dy, keep, epsilon, g0, g1, g2,
                               p, out, i)
        for x in range(S):
            dx = flip[x]
            if dx == 0.0 or alleles[x, i] < 0:
                continue
            f1 = dx if alleles[x, i] == 0 else 1.0 - dx     # P(the flipped label's homologue carries allele 1)
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


@njit(cache=True)
def project_alleles(P, site_bin, alleles):
    """float64 [4, m]: ordered allele posteriors (00, 01, 10, 11) of each site from its bin's label-pair posterior
    P [B, S, S] and the labels' alleles [S, m] (-1 unknown: contributes to none). O(m S^2) time, no [m, S, S] copy."""
    S = P.shape[1]
    m = site_bin.shape[0]
    out = np.zeros((4, m))
    for i in range(m):
        b = site_bin[i]
        for x in range(S):
            ax = alleles[x, i]
            if ax < 0:
                continue
            for y in range(S):
                ay = alleles[y, i]
                if ay >= 0:
                    out[2 * ax + ay, i] += P[b, x, y]
    return out
