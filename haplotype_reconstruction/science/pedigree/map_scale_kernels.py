"""Read the painted classes of the painting-based map-scale estimate (rev 5, DESIGN_MAPSCALE): switch-interval
candidates, the guard's pair observations and IBD-count chains, and the sibs' frames.

Every kernel reads the released painting of one relatedness family at a time (classes and keys only, no emission;
the frame chains on emissions, with the read state, are map_scale_chains). Rates are genome-wide (cM/Mb, Morgans per
bp = r 1e-8), Haldane switches between observations. Inputs are C-contiguous writable arrays (the wrappers in
map_scale_frames copy them); outputs are freshly allocated.

Method: docs/methods/pedigree.md#map-scale.
"""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import kernel

__all__ = [
    "switch_candidates",
    "pair_observations",
    "pair_fullsib_loglik",
    "pair_alternative_loglik",
    "sib_frames",
]


def _a(dtype, ndim):
    return types.Array(dtype, ndim, "C", readonly=False, aligned=True)


_u8_3 = _a(types.uint8, 3)
_i32_2, _i32_3 = _a(types.int32, 2), _a(types.int32, 3)
_i64_1 = _a(types.int64, 1)
_b1 = _a(types.boolean, 1)
_f1 = _a(types.float64, 1)
# Graph workers attach immutable component arrays. Member and slot indices are allocated inside the worker.
_ro_keys = types.Array(types.int32, 3, "C", readonly=True)
_ro_centers = types.Array(types.float64, 1, "C", readonly=True)

# painting.TrackStatus values (contracts.painting): firm stage-1 tracks and switch intervals.
_NAMED = 2
_POOLED = 3
_SWITCH = 7


@kernel(overloads=((_u8_3, _i32_3, _i32_2, types.int64, _b1),), cache=True, nogil=True)
def switch_candidates(status, classes, keys_table, n_classes, eligible):
    """For tracks in a SWITCH_INTERVAL run flanked by firm (named or pooled) bins: the bin-local keys of the left- and
    right-flank classes at every bin of the run (int32 [N, 2, B] each; -1 when absent or not a class key)."""
    N, _, B = status.shape
    c1 = np.full((N, 2, B), -1, np.int32)
    c2 = np.full((N, 2, B), -1, np.int32)
    for s in range(N):
        if not eligible[s]:
            continue
        for t in range(2):
            b = 0
            while b < B:
                if status[s, t, b] != _SWITCH:
                    b += 1
                    continue
                e = b
                while e < B and status[s, t, e] == _SWITCH:
                    e += 1
                lb = b - 1
                rb = e
                if lb >= 0 and rb < B and (status[s, t, lb] == _NAMED or status[s, t, lb] == _POOLED) and (
                        status[s, t, rb] == _NAMED or status[s, t, rb] == _POOLED):
                    A = classes[s, t, lb]
                    Z = classes[s, t, rb]
                    if A >= 0 and Z >= 0 and A < keys_table.shape[1] and Z < keys_table.shape[1]:
                        for x in range(b, e):
                            k1 = keys_table[x, A]
                            k2 = keys_table[x, Z]
                            if k1 < n_classes:
                                c1[s, t, x] = k1
                            if k2 < n_classes and k2 != k1:
                                c2[s, t, x] = k2
                b = e
    return c1, c2


@kernel(overloads=((_i32_3, _i32_3, _i32_3, _i64_1, _i64_1, _f1),
                   (_ro_keys, _ro_keys, _ro_keys, _i64_1, _i64_1, _ro_centers)), cache=True, nogil=True)
def pair_observations(keys, cand1, cand2, members, carriers, centers):
    """Shared-key observations of every member pair (a, c), a < c, in pair order: at a bin where the OTHER carriers
    (members and slot samples, firm keys only) show exactly four distinct keys and all four tracks of the pair carry a
    key (or a switch candidate) inside them, the bit mask of the pair's possible shared counts over all completions.
    Returns (positions float64 [P B], masks int8 [P B], lengths int64 [P]); pair p's observations are rows
    [p B, p B + lengths[p])."""
    n = len(members)
    B = keys.shape[2]
    P = n * (n - 1) // 2
    max_key = 0
    for m in range(n):
        for t in range(2):
            for b in range(B):
                if keys[members[m], t, b] > max_key:
                    max_key = keys[members[m], t, b]
                if cand1[members[m], t, b] > max_key:
                    max_key = cand1[members[m], t, b]
                if cand2[members[m], t, b] > max_key:
                    max_key = cand2[members[m], t, b]
    for m in range(len(carriers)):
        for t in range(2):
            for b in range(B):
                if keys[carriers[m], t, b] > max_key:
                    max_key = keys[carriers[m], t, b]
    K = max_key + 1
    counts = np.zeros(K, np.int64)
    lens = np.zeros(P, np.int64)
    pos_out = np.empty(P * B, np.float64)
    obs_out = np.empty(P * B, np.int8)
    cand = np.full((4, 2), -1, np.int64)
    smp4 = np.empty(4, np.int64)
    trk4 = np.empty(4, np.int64)
    for b in range(B):
        counts[:] = 0
        for m in range(n):
            for t in range(2):
                k = keys[members[m], t, b]
                if k >= 0:
                    counts[k] += 1
        for m in range(len(carriers)):
            for t in range(2):
                k = keys[carriers[m], t, b]
                if k >= 0:
                    counts[k] += 1
        p = 0
        for a in range(n):
            for c in range(a + 1, n):
                smp4[0] = members[a]
                smp4[1] = members[a]
                smp4[2] = members[c]
                smp4[3] = members[c]
                trk4[0] = 0
                trk4[1] = 1
                trk4[2] = 0
                trk4[3] = 1
                for q in range(4):
                    k = keys[smp4[q], trk4[q], b]
                    if k >= 0:
                        counts[k] -= 1
                distinct = 0
                for k in range(K):
                    if counts[k] > 0:
                        distinct += 1
                if distinct == 4:
                    ok = True
                    for q in range(4):
                        cand[q, 0] = -1
                        cand[q, 1] = -1
                        k = keys[smp4[q], trk4[q], b]
                        if k >= 0:
                            if counts[k] > 0:
                                cand[q, 0] = k
                        else:
                            k1 = cand1[smp4[q], trk4[q], b]
                            k2 = cand2[smp4[q], trk4[q], b]
                            if k1 >= 0 and counts[k1] > 0:
                                cand[q, 0] = k1
                            if k2 >= 0 and counts[k2] > 0:
                                if cand[q, 0] < 0:
                                    cand[q, 0] = k2
                                else:
                                    cand[q, 1] = k2
                        if cand[q, 0] < 0:
                            ok = False
                    if ok:
                        mask = 0
                        for i0 in range(2):
                            x0 = cand[0, i0]
                            if x0 < 0:
                                continue
                            for i1 in range(2):
                                x1 = cand[1, i1]
                                if x1 < 0:
                                    continue
                                for i2 in range(2):
                                    y0 = cand[2, i2]
                                    if y0 < 0:
                                        continue
                                    for i3 in range(2):
                                        y1 = cand[3, i3]
                                        if y1 < 0:
                                            continue
                                        s1 = (1 if x0 == y0 else 0) + (1 if x1 == y1 else 0)
                                        s2 = (1 if x0 == y1 else 0) + (1 if x1 == y0 else 0)
                                        mask |= 1 << (s1 if s1 > s2 else s2)
                        pos_out[p * B + lens[p]] = centers[b]
                        obs_out[p * B + lens[p]] = mask
                        lens[p] += 1
                for q in range(4):
                    k = keys[smp4[q], trk4[q], b]
                    if k >= 0:
                        counts[k] += 1
                p += 1
    return pos_out, obs_out, lens


@kernel(overloads=((_i64_1, _f1, _i64_1, _f1, _f1),), cache=True, parallel=True, nogil=True)
def pair_fullsib_loglik(offsets, pos, obs, r_grid, eps_grid):
    """Full-sib IBD-count chain of each pair sequence [S, R, E]: (I_u, I_v) independent bits flipping at 2 per Morgan;
    emission 1 - eps when the observed mask allows the state's shared count, else eps / 2."""
    S = len(offsets) - 1
    R = len(r_grid)
    E = len(eps_grid)
    out = np.zeros((S, R, E))
    for s in prange(S):
        lo = offsets[s]
        hi = offsets[s + 1]
        if hi - lo < 1:
            continue
        for ri in range(R):
            rate = r_grid[ri] * 1e-8
            for ei in range(E):
                eps = eps_grid[ei]
                em_ok = 1.0 - eps
                em_bad = eps / 2.0
                a0 = 0.25
                a1 = 0.25
                a2 = 0.25
                a3 = 0.25
                ll = 0.0
                for x in range(lo, hi):
                    if x > lo:
                        d = (pos[x] - pos[x - 1]) * rate
                        q = 0.5 * (1.0 - math.exp(-4.0 * d))
                        st = 1.0 - q
                        b0 = st * a0 + q * a1
                        b1 = q * a0 + st * a1
                        b2 = st * a2 + q * a3
                        b3 = q * a2 + st * a3
                        a0 = st * b0 + q * b2
                        a2 = q * b0 + st * b2
                        a1 = st * b1 + q * b3
                        a3 = q * b1 + st * b3
                    c = obs[x]
                    e0 = em_ok if c & 1 else em_bad
                    e1 = em_ok if c & 2 else em_bad
                    e2 = em_ok if c & 4 else em_bad
                    a0 *= e0
                    a1 *= e1
                    a2 *= e1
                    a3 *= e2
                    tot = a0 + a1 + a2 + a3
                    ll += math.log(tot)
                    a0 /= tot
                    a1 /= tot
                    a2 /= tot
                    a3 /= tot
                out[s, ri, ei] = ll
    return out


@kernel(overloads=((_i64_1, _f1, _i64_1, _f1, _f1),), cache=True, parallel=True, nogil=True)
def pair_alternative_loglik(offsets, pos, obs, r_grid, eps_grid):
    """Relationship alternatives of each pair sequence: half sibs (one IBD bit flipping at 2 per Morgan) [S, R, E] and
    parent-offspring (one shared copy everywhere) [S, E]."""
    S = len(offsets) - 1
    R = len(r_grid)
    E = len(eps_grid)
    hs = np.zeros((S, R, E))
    po = np.zeros((S, E))
    for s in prange(S):
        lo = offsets[s]
        hi = offsets[s + 1]
        for ei in range(E):
            eps = eps_grid[ei]
            ll = 0.0
            for x in range(lo, hi):
                ll += math.log(1.0 - eps if obs[x] & 2 else eps / 2.0)
            po[s, ei] = ll
        for ri in range(R):
            rate = r_grid[ri] * 1e-8
            for ei in range(E):
                eps = eps_grid[ei]
                a0 = 0.5
                a1 = 0.5
                ll = 0.0
                for x in range(lo, hi):
                    if x > lo:
                        d = (pos[x] - pos[x - 1]) * rate
                        q = 0.5 * (1.0 - math.exp(-4.0 * d))
                        b0 = (1.0 - q) * a0 + q * a1
                        b1 = q * a0 + (1.0 - q) * a1
                        a0 = b0
                        a1 = b1
                    c = obs[x]
                    a0 *= (1.0 - eps) if c & 1 else eps / 2.0
                    a1 *= (1.0 - eps) if c & 2 else eps / 2.0
                    tot = a0 + a1
                    ll += math.log(tot)
                    a0 /= tot
                    a1 /= tot
                hs[s, ri, ei] = ll
    return hs, po


@kernel(overloads=((_i32_3, _i32_3, _i64_1, _i64_1, _f1),
                   (_ro_keys, _ro_keys, _i64_1, _i64_1, _ro_centers)), cache=True, nogil=True)
def sib_frames(cls, key, members, slots, centers):
    """Determined frames of every target member (strict linking). At a bin where the OTHER carriers (members and slot
    samples, firm tracks only) show exactly four classes with four distinct keys, and the other members' co-occurrence
    leaves exactly one supported 2 + 2 split, the two parents' homologue classes are labelled and linked to the
    previous determined bin by shared classes (a part that changed a class across an undetermined stretch needs both
    classes shared; an ambiguous link starts new chains). Returns per frame the target index (into members), the
    chain id of parent 0's homologue chain, the bin and the four labels (parent 0 homologues 0, 1; parent 1
    homologues 0, 1)."""
    n = len(members)
    B = cls.shape[2]
    fr_t = np.empty(n * B, np.int64)
    fr_c = np.empty(n * B, np.int64)
    fr_b = np.empty(n * B, np.int64)
    fr_lab = np.empty((n * B, 4), np.int64)
    nf = 0
    chain_next = 0
    D = np.empty(4, np.int64)
    Dk = np.empty(4, np.int64)
    for ti in range(n):
        target = members[ti]
        have_prev = False
        b_prev = -10
        prev = np.full((2, 2), -1, np.int64)
        chain = np.array([-1, -1], np.int64)
        for b in range(B):
            nd = 0
            bad = False
            for j in range(n + len(slots)):
                s = members[j] if j < n else slots[j - n]
                if s == target:
                    continue
                for t in range(2):
                    c = cls[s, t, b]
                    if c < 0:
                        continue
                    k = key[s, t, b]
                    found = False
                    for q in range(nd):
                        if D[q] == c:
                            found = True
                            if Dk[q] != k:
                                bad = True
                            break
                    if not found:
                        if nd == 4:
                            bad = True
                            break
                        D[nd] = c
                        Dk[nd] = k
                        nd += 1
                if bad:
                    break
            if bad or nd != 4:
                continue
            distinct_keys = True
            for q in range(4):
                for r_ in range(q + 1, 4):
                    if Dk[q] == Dk[r_]:
                        distinct_keys = False
            if not distinct_keys:
                continue
            edge = np.zeros((4, 4), np.bool_)
            for j in range(n):
                s = members[j]
                if s == target:
                    continue
                c0 = cls[s, 0, b]
                c1 = cls[s, 1, b]
                if c0 < 0 or c1 < 0:
                    continue
                i0 = -1
                i1 = -1
                for q in range(4):
                    if D[q] == c0:
                        i0 = q
                    if D[q] == c1:
                        i1 = q
                if i0 >= 0 and i1 >= 0 and i0 != i1:
                    edge[i0, i1] = True
                    edge[i1, i0] = True
            ok_split = -1
            n_ok = 0
            for sp in range(3):
                a0 = 0
                a1 = sp + 1
                o = np.empty(2, np.int64)
                w = 0
                for q in range(1, 4):
                    if q != a1:
                        o[w] = q
                        w += 1
                if edge[a0, a1] or edge[o[0], o[1]]:
                    continue
                sup = True
                for q in (a0, a1):
                    if not (edge[q, o[0]] or edge[q, o[1]]):
                        sup = False
                for q in (o[0], o[1]):
                    if not (edge[q, a0] or edge[q, a1]):
                        sup = False
                if sup:
                    n_ok += 1
                    ok_split = sp
            if n_ok != 1:
                continue
            a1 = ok_split + 1
            o0 = -1
            o1 = -1
            for q in range(1, 4):
                if q != a1:
                    if o0 < 0:
                        o0 = q
                    else:
                        o1 = q
            parts = np.empty((2, 2), np.int64)
            parts[0, 0] = D[0]
            parts[0, 1] = D[a1]
            parts[1, 0] = D[o0]
            parts[1, 1] = D[o1]
            lab = np.empty((2, 2), np.int64)
            if not have_prev:
                lab[:, :] = parts
                chain[0] = chain_next
                chain[1] = chain_next + 1
                chain_next += 2
            else:
                share = np.zeros((2, 2), np.int64)
                for pn in range(2):
                    for po in range(2):
                        for x in range(2):
                            for y in range(2):
                                if parts[pn, x] == prev[po, y]:
                                    share[pn, po] += 1
                ok_link = True
                mapping = np.array([-1, -1], np.int64)
                for pn in range(2):
                    if share[pn, 0] > 0 and share[pn, 1] == 0:
                        mapping[pn] = 0
                    elif share[pn, 1] > 0 and share[pn, 0] == 0:
                        mapping[pn] = 1
                    else:
                        ok_link = False
                if ok_link and mapping[0] == mapping[1]:
                    ok_link = False
                if ok_link and b - b_prev > 1:
                    # strict: a part that changed a class across an undetermined stretch may have lost its phase
                    for pn in range(2):
                        if share[pn, mapping[pn]] < 2:
                            ok_link = False
                if not ok_link:
                    lab[:, :] = parts
                    chain[0] = chain_next
                    chain[1] = chain_next + 1
                    chain_next += 2
                else:
                    for pn in range(2):
                        po = mapping[pn]
                        x0 = parts[pn, 0]
                        x1 = parts[pn, 1]
                        if x0 == prev[po, 0] or x1 == prev[po, 1]:
                            lab[po, 0] = x0
                            lab[po, 1] = x1
                        else:
                            lab[po, 0] = x1
                            lab[po, 1] = x0
            prev[:, :] = lab
            have_prev = True
            b_prev = b
            fr_t[nf] = ti
            fr_c[nf] = chain[0]
            fr_b[nf] = b
            fr_lab[nf, 0] = lab[0, 0]
            fr_lab[nf, 1] = lab[0, 1]
            fr_lab[nf, 2] = lab[1, 0]
            fr_lab[nf, 3] = lab[1, 1]
            nf += 1
    return fr_t[:nf], fr_c[:nf], fr_b[:nf], fr_lab[:nf]
