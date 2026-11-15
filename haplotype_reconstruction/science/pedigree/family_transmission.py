"""Family transmission likelihoods of a released parent edge on scheduled paintings (PD29).

One painting segment (B bins of one component, T whole-component trajectories with bin-local classes tc[T, B]):
two latent (unsequenced) diploids, each with two homologue paths over the trajectories under PD06's founder-mosaic
law (start 1/T; stay 1 - theta; change to a named other trajectory theta / (T - 1)). Each observed member is a
direct observation of latent 0 or 1 (role 0, 1), a child of the pair (role 2: a paternal selector over latent 0's
homologues and a maternal selector over latent 1's, each switching with the meiotic probability theta), ignored
(role 3), an outsider explained by its own best mosaic (role 4) or a half child of latent 0 or 1 (roles 5, 6: one
homologue through a meiotic selector over that latent's homologues, the other a free founder mosaic under the same
law). Emission of an unordered painted pair {x, y} given true classes (u, v): f(x,u) f(y,v) + f(y,u) f(x,v)
(x != y) or f(x,u) f(x,v), with f = 1 - delta when equal and delta otherwise (PD06's convention); bins without both
labels emit 1. The profile log likelihood maximises over the latent paths and the flexible members' roles (MAP-EM
with exact selector posteriors, pair-path Viterbi M steps, exact homologue-label flip moves and role updates;
monotone); member selectors are summed exactly.

Method: docs/methods/pedigree.md#pd29
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from ...core.kernels import kernel
from ...core.random import stream
from .kernels.signatures import OVERLOADS

NEG = -1e300


@kernel(overloads=OVERLOADS["family_transmission._f"], cache=True, nogil=True)
def _f(a, b, delta):
    return 1.0 - delta if a == b else delta


@kernel(overloads=OVERLOADS["family_transmission._emit"], cache=True, nogil=True)
def _emit(x, y, u, v, delta):
    if x == y:
        return _f(x, u, delta) * _f(x, v, delta)
    return _f(x, u, delta) * _f(y, v, delta) + _f(y, u, delta) * _f(x, v, delta)


@kernel(overloads=OVERLOADS["family_transmission._switch"], cache=True, nogil=True)
def _switch(theta, floor):
    return min(0.5, max(floor, theta))


@kernel(overloads=OVERLOADS["family_transmission._child_fb"], cache=True, nogil=True)
def _child_fb(tc, cls, known, paths, theta, floor, delta, want_gamma, gamma, inc):
    """Forward-backward of one child member (cls [B,2], known [B]) given latent paths [2,2,B] (trajectories).
    Adds per-bin log scaling to inc[B]; fills gamma[B,4] (state = 2*s + r) if want_gamma. Returns log lik."""
    B = cls.shape[0]
    fw = np.empty((B, 4))
    scale = np.empty(B)
    total = 0.0
    for t in range(B):
        e = np.ones(4)
        if known[t]:
            for s in range(2):
                u = tc[paths[0, s, t], t]
                for r in range(2):
                    v = tc[paths[1, r, t], t]
                    e[2 * s + r] = _emit(cls[t, 0], cls[t, 1], u, v, delta)
        if t == 0:
            for q in range(4):
                fw[t, q] = 0.25 * e[q]
        else:
            th = _switch(theta[t - 1], floor)
            st = 1.0 - th
            for q in range(4):
                s, r = q >> 1, q & 1
                acc = 0.0
                for p in range(4):
                    ps, pr = p >> 1, p & 1
                    w = (st if ps == s else th) * (st if pr == r else th)
                    acc += fw[t - 1, p] * w
                fw[t, q] = acc * e[q]
        z = fw[t, 0] + fw[t, 1] + fw[t, 2] + fw[t, 3]
        if z <= 0.0:
            return -np.inf
        for q in range(4):
            fw[t, q] /= z
        scale[t] = z
        lz = math.log(z)
        inc[t] += lz
        total += lz
    if want_gamma:
        bw = np.ones(4)
        for t in range(B - 1, -1, -1):
            g = 0.0
            for q in range(4):
                gamma[t, q] = fw[t, q] * bw[q]
                g += gamma[t, q]
            for q in range(4):
                gamma[t, q] /= g
            if t > 0:
                e = np.ones(4)
                if known[t]:
                    for s in range(2):
                        u = tc[paths[0, s, t], t]
                        for r in range(2):
                            v = tc[paths[1, r, t], t]
                            e[2 * s + r] = _emit(cls[t, 0], cls[t, 1], u, v, delta)
                th = _switch(theta[t - 1], floor)
                st = 1.0 - th
                nb = np.zeros(4)
                for p in range(4):
                    ps, pr = p >> 1, p & 1
                    acc = 0.0
                    for q in range(4):
                        s, r = q >> 1, q & 1
                        w = (st if ps == s else th) * (st if pr == r else th)
                        acc += w * e[q] * bw[q]
                    nb[p] = acc / scale[t]
                bw = nb
    return total



@kernel(overloads=OVERLOADS["family_transmission._half_fb"], cache=True, nogil=True)
def _half_fb(tc, cls, known, paths, latent, theta, floor, delta, want_gamma, gamma, inc):
    """Half child of one latent diploid: homologue A copied from latent ``latent`` by a meiotic selector s, homologue
    B a free founder mosaic f (PD06 law). States (s, f); gamma[B, 2, T] posterior if want_gamma; log lik summed."""
    T = tc.shape[0]
    B = cls.shape[0]
    fw = np.empty((B, 2, T))
    scale = np.empty(B)
    total = 0.0
    for t in range(B):
        if t == 0:
            for sl in range(2):
                for f in range(T):
                    e = 1.0
                    if known[t]:
                        e = _emit(cls[t, 0], cls[t, 1], tc[paths[latent, sl, t], t], tc[f, t], delta)
                    fw[t, sl, f] = 0.5 / T * e
        else:
            th = _switch(theta[t - 1], floor)
            st = 1.0 - th
            ch = th / (T - 1) if T > 1 else 0.0
            for sl in range(2):
                sel = np.empty(T)
                tot = 0.0
                for f in range(T):
                    sel[f] = fw[t - 1, sl, f] * st + fw[t - 1, 1 - sl, f] * th
                    tot += sel[f]
                for f in range(T):
                    v = sel[f] * (1.0 - th) + (tot - sel[f]) * ch
                    e = 1.0
                    if known[t]:
                        e = _emit(cls[t, 0], cls[t, 1], tc[paths[latent, sl, t], t], tc[f, t], delta)
                    fw[t, sl, f] = v * e
        z = 0.0
        for sl in range(2):
            for f in range(T):
                z += fw[t, sl, f]
        if z <= 0.0:
            return -np.inf
        for sl in range(2):
            for f in range(T):
                fw[t, sl, f] /= z
        scale[t] = z
        lz = math.log(z)
        inc[t] += lz
        total += lz
    if want_gamma:
        bw = np.ones((2, T))
        for t in range(B - 1, -1, -1):
            g = 0.0
            for sl in range(2):
                for f in range(T):
                    gamma[t, sl, f] = fw[t, sl, f] * bw[sl, f]
                    g += gamma[t, sl, f]
            for sl in range(2):
                for f in range(T):
                    gamma[t, sl, f] /= g
            if t > 0:
                th = _switch(theta[t - 1], floor)
                st = 1.0 - th
                ch = th / (T - 1) if T > 1 else 0.0
                eb = np.empty((2, T))
                for sl in range(2):
                    for f in range(T):
                        e = 1.0
                        if known[t]:
                            e = _emit(cls[t, 0], cls[t, 1], tc[paths[latent, sl, t], t], tc[f, t], delta)
                        eb[sl, f] = e * bw[sl, f]
                # transpose of mosaic step then selector step
                mo = np.empty((2, T))
                for sl in range(2):
                    tot = 0.0
                    for f in range(T):
                        tot += eb[sl, f]
                    for f in range(T):
                        mo[sl, f] = eb[sl, f] * (1.0 - th) + (tot - eb[sl, f]) * ch
                for sl in range(2):
                    for f in range(T):
                        bw[sl, f] = (mo[sl, f] * st + mo[1 - sl, f] * th) / scale[t]
    return total

@kernel(overloads=OVERLOADS["family_transmission._prior_inc"], cache=True, nogil=True)
def _prior_inc(paths, which, T, theta, floor, inc):
    """Founder-mosaic prior of the latent homologue paths of latent diploids flagged in which[2]; adds to inc."""
    B = paths.shape[2]
    total = 0.0
    lt = -math.log(T)
    for l in range(2):
        if not which[l]:
            continue
        for h in range(2):
            inc[0] += lt
            total += lt
            for t in range(1, B):
                th = _switch(theta[t - 1], floor)
                if paths[l, h, t] == paths[l, h, t - 1]:
                    v = math.log(1.0 - th)
                else:
                    v = math.log(th / max(1, T - 1)) if T > 1 else NEG
                inc[t] += v
                total += v
    return total


@kernel(overloads=OVERLOADS["family_transmission._direct_inc"], cache=True, nogil=True)
def _direct_inc(tc, cls, known, paths, latent, delta, inc):
    B = cls.shape[0]
    total = 0.0
    for t in range(B):
        if known[t]:
            v = math.log(_emit(cls[t, 0], cls[t, 1], tc[paths[latent, 0, t], t], tc[paths[latent, 1, t], t], delta))
            inc[t] += v
            total += v
    return total


@kernel(overloads=OVERLOADS["family_transmission.evaluate"], cache=True, nogil=True)
def evaluate(tc, mcls, mknown, roles, paths, which, theta, floor, delta, inc, alone_inc, role_lp):
    """Exact log P(data, paths, roles) (members' selectors summed); per-bin increments added to inc[B].
    Role 4 (outsider) contributes the member's own best mosaic fit alone_inc[m]; role_lp[m] is the log prior of the
    member's current role (credited to bin 0)."""
    T = tc.shape[0]
    B = tc.shape[1]
    total = _prior_inc(paths, which, T, theta, floor, inc)
    dummy = np.empty((1, 4))
    dummy3 = np.empty((1, 1, 1))
    for m in range(len(roles)):
        inc[0] += role_lp[m]
        total += role_lp[m]
        if roles[m] == 4:
            for t in range(B):
                inc[t] += alone_inc[m, t]
                total += alone_inc[m, t]
            continue
        if roles[m] == 0:
            total += _direct_inc(tc, mcls[m], mknown[m], paths, 0, delta, inc)
        elif roles[m] == 1:
            total += _direct_inc(tc, mcls[m], mknown[m], paths, 1, delta, inc)
        elif roles[m] == 2:
            total += _child_fb(tc, mcls[m], mknown[m], paths, theta, floor, delta, False, dummy, inc)
        elif roles[m] == 5 or roles[m] == 6:
            total += _half_fb(tc, mcls[m], mknown[m], paths, roles[m] - 5, theta, floor, delta, False, dummy3, inc)
    return total


@kernel(overloads=OVERLOADS["family_transmission._mstep"], cache=True, nogil=True)
def _mstep(tc, mcls, mknown, roles, gammas, paths, latent, theta, floor, delta, maxmode, hgam):
    """Viterbi over ordered homologue pairs (a, b) of one latent diploid with the other latent fixed.
    Maximises sum_t expected complete log emission + mosaic prior. Returns new paths [2, B] for that latent."""
    T = tc.shape[0]
    B = tc.shape[1]
    other = 1 - latent
    V = np.empty((T, T))
    back = np.empty((B, T, T), dtype=np.int32)
    score = np.zeros((T, T))
    for t in range(B):
        # emission score of every pair at bin t
        for a in range(T):
            for b in range(T):
                score[a, b] = 0.0
        for m in range(len(roles)):
            if not mknown[m, t]:
                continue
            x = mcls[m, t, 0]
            y = mcls[m, t, 1]
            if roles[m] == latent:
                for a in range(T):
                    ua = tc[a, t]
                    for b in range(T):
                        score[a, b] += math.log(_emit(x, y, ua, tc[b, t], delta))
            elif roles[m] == 5 + latent:
                # half child of this latent: its slot-s homologue against the free mosaic's posterior
                for slot in range(2):
                    for a in range(T):
                        ua = tc[a, t]
                        acc = 0.0
                        for f in range(T):
                            g = hgam[m, t, slot, f]
                            if g > 0.0:
                                acc += g * math.log(_emit(x, y, ua, tc[f, t], delta))
                        if slot == 0:
                            for b in range(T):
                                score[a, b] += acc
                        else:
                            for b in range(T):
                                score[b, a] += acc
            elif roles[m] == 2 and maxmode:
                # initialisation: each child explained by its best selector state at this bin (no continuity)
                for a in range(T):
                    ua = tc[a, t]
                    for b in range(T):
                        ub = tc[b, t]
                        best = NEG
                        for o in range(2):
                            vo = tc[paths[other, o, t], t]
                            va = math.log(_emit(x, y, ua, vo, delta))
                            vb = math.log(_emit(x, y, ub, vo, delta))
                            if va > best:
                                best = va
                            if vb > best:
                                best = vb
                        score[a, b] += best
            elif roles[m] == 2:
                # child: homologue slot s of latent 0 (paternal) or r of latent 1 (maternal)
                for slot in range(2):
                    for a in range(T):
                        ua = tc[a, t]
                        acc = 0.0
                        for o in range(2):
                            vo = tc[paths[other, o, t], t]
                            if latent == 0:
                                g = gammas[m, t, 2 * slot + o]
                            else:
                                g = gammas[m, t, 2 * o + slot]
                            if g > 0.0:
                                if latent == 0:
                                    acc += g * math.log(_emit(x, y, ua, vo, delta))
                                else:
                                    acc += g * math.log(_emit(x, y, vo, ua, delta))
                        if slot == 0:
                            for b in range(T):
                                score[a, b] += acc
                        else:
                            for b in range(T):
                                score[b, a] += acc
        if t == 0:
            for a in range(T):
                for b in range(T):
                    V[a, b] = score[a, b] - 2.0 * math.log(T)
            continue
        th = _switch(theta[t - 1], floor)
        ls = math.log(1.0 - th)
        lc = math.log(th / (T - 1)) if T > 1 else NEG
        rowmax = np.empty(T)
        rowarg = np.empty(T, dtype=np.int64)
        colmax = np.empty(T)
        colarg = np.empty(T, dtype=np.int64)
        gmax = NEG
        ga = 0
        gb = 0
        for a in range(T):
            rowmax[a] = NEG
            colmax[a] = NEG
        for a in range(T):
            for b in range(T):
                v = V[a, b]
                if v > rowmax[a]:
                    rowmax[a] = v
                    rowarg[a] = b
                if v > colmax[b]:
                    colmax[b] = v
                    colarg[b] = a
                if v > gmax:
                    gmax = v
                    ga = a
                    gb = b
        NV = np.empty((T, T))
        for a in range(T):
            for b in range(T):
                best = V[a, b] + 2.0 * ls
                arg = a * T + b
                c = rowmax[a] + ls + lc
                if c > best:
                    best = c
                    arg = a * T + rowarg[a]
                c = colmax[b] + lc + ls
                if c > best:
                    best = c
                    arg = colarg[b] * T + b
                c = gmax + 2.0 * lc
                if c > best:
                    best = c
                    arg = ga * T + gb
                NV[a, b] = best + score[a, b]
                back[t, a, b] = arg
        V = NV
    # traceback (first maximum in (a, b) order)
    out = np.empty((2, B), dtype=np.int64)
    best = NEG
    ba = 0
    bb = 0
    for a in range(T):
        for b in range(T):
            if V[a, b] > best:
                best = V[a, b]
                ba = a
                bb = b
    out[0, B - 1] = ba
    out[1, B - 1] = bb
    for t in range(B - 1, 0, -1):
        arg = back[t, out[0, t], out[1, t]]
        out[0, t - 1] = arg // T
        out[1, t - 1] = arg % T
    return out


@kernel(overloads=OVERLOADS["family_transmission._emis4"], cache=True, nogil=True)
def _emis4(tc, cls, known, paths, t, delta, e):
    for q in range(4):
        e[q] = 1.0
    if known[t]:
        for s in range(2):
            u = tc[paths[0, s, t], t]
            for r in range(2):
                e[2 * s + r] = _emit(cls[t, 0], cls[t, 1], u, tc[paths[1, r, t], t], delta)


@kernel(overloads=OVERLOADS["family_transmission._half_flip"], cache=True, nogil=True)
def _half_flip(tc, cls, known, paths, latent, theta, floor, delta, gains):
    """Add to gains[t - 1] the exact log-likelihood change, for one half child of ``latent``, of swapping that
    latent's homologue labels on every bin from t on: its selector transition at boundary t inverts (stay <-> switch)
    while its free mosaic is unchanged; exact through the pairwise posteriors of the (selector, mosaic) chain."""
    T = tc.shape[0]
    B = cls.shape[0]
    fw = np.empty((B, 2, T))
    scale = np.empty(B)
    for t in range(B):
        if t == 0:
            for sl in range(2):
                for f in range(T):
                    e = 1.0
                    if known[t]:
                        e = _emit(cls[t, 0], cls[t, 1], tc[paths[latent, sl, t], t], tc[f, t], delta)
                    fw[t, sl, f] = 0.5 / T * e
        else:
            th = _switch(theta[t - 1], floor)
            st = 1.0 - th
            ch = th / (T - 1) if T > 1 else 0.0
            for sl in range(2):
                tot = 0.0
                for f in range(T):
                    tot += fw[t - 1, sl, f] * st + fw[t - 1, 1 - sl, f] * th
                for f in range(T):
                    sel = fw[t - 1, sl, f] * st + fw[t - 1, 1 - sl, f] * th
                    e = 1.0
                    if known[t]:
                        e = _emit(cls[t, 0], cls[t, 1], tc[paths[latent, sl, t], t], tc[f, t], delta)
                    fw[t, sl, f] = (sel * (1.0 - th) + (tot - sel) * ch) * e
        z = 0.0
        for sl in range(2):
            for f in range(T):
                z += fw[t, sl, f]
        if z <= 0.0:
            return
        for sl in range(2):
            for f in range(T):
                fw[t, sl, f] /= z
        scale[t] = z
    bw = np.ones((2, T))
    g = np.empty((2, T))
    mixed = np.empty((2, T))
    mo = np.empty((2, T))
    for t in range(B - 1, 0, -1):
        th = _switch(theta[t - 1], floor)
        st = 1.0 - th
        ch = th / (T - 1) if T > 1 else 0.0
        for sl in range(2):
            for f in range(T):
                e = 1.0
                if known[t]:
                    e = _emit(cls[t, 0], cls[t, 1], tc[paths[latent, sl, t], t], tc[f, t], delta)
                g[sl, f] = e * bw[sl, f]
        # mosaic step from t - 1: mixed[s, f'] = sum_f fw[t-1, s, f] Mos(f, f')
        for sl in range(2):
            tot = 0.0
            for f in range(T):
                tot += fw[t - 1, sl, f]
            for f in range(T):
                mixed[sl, f] = fw[t - 1, sl, f] * (1.0 - th) + (tot - fw[t - 1, sl, f]) * ch
        a00 = 0.0
        a01 = 0.0
        a10 = 0.0
        a11 = 0.0
        for f in range(T):
            a00 += mixed[0, f] * g[0, f]
            a01 += mixed[0, f] * g[1, f]
            a10 += mixed[1, f] * g[0, f]
            a11 += mixed[1, f] * g[1, f]
        den = st * (a00 + a11) + th * (a01 + a10)
        num = th * (a00 + a11) + st * (a01 + a10)
        if den > 0.0 and num > 0.0:
            gains[t - 1] += math.log(num / den)
        # backward message to t - 1 (scaled as the forward pass)
        for sl in range(2):
            tot = 0.0
            for f in range(T):
                tot += g[sl, f]
            for f in range(T):
                mo[sl, f] = g[sl, f] * (1.0 - th) + (tot - g[sl, f]) * ch
        for sl in range(2):
            for f in range(T):
                bw[sl, f] = (mo[sl, f] * st + mo[1 - sl, f] * th) / scale[t]


@kernel(overloads=OVERLOADS["family_transmission.flip_gains"], cache=True, nogil=True)
def flip_gains(tc, mcls, mknown, roles, paths, latent, theta, floor, delta):
    """Exact log-likelihood change of swapping the two homologue labels of one latent diploid on every bin after
    boundary t (t = 0 .. B-2): the children's selector transition on that latent inverts at t (exact through the
    pairwise posteriors, the likelihood being linear in one transition matrix) and the latent's mosaic prior
    changes at t; direct observations are unordered and unchanged; half children of the latent add their own
    exact selector term (_half_flip)."""
    T = tc.shape[0]
    B = tc.shape[1]
    gains = np.zeros(B - 1)
    fw = np.empty((B, 4))
    scale = np.empty(B)
    bw = np.empty((B, 4))
    e = np.empty(4)
    for m in range(len(roles)):
        if roles[m] == 5 + latent:
            _half_flip(tc, mcls[m], mknown[m], paths, latent, theta, floor, delta, gains)
            continue
        if roles[m] != 2:
            continue
        cls = mcls[m]
        known = mknown[m]
        for t in range(B):
            _emis4(tc, cls, known, paths, t, delta, e)
            if t == 0:
                for q in range(4):
                    fw[t, q] = 0.25 * e[q]
            else:
                th = _switch(theta[t - 1], floor)
                st = 1.0 - th
                for q in range(4):
                    acc = 0.0
                    for p in range(4):
                        w = (st if (p >> 1) == (q >> 1) else th) * (st if (p & 1) == (q & 1) else th)
                        acc += fw[t - 1, p] * w
                    fw[t, q] = acc * e[q]
            z = fw[t, 0] + fw[t, 1] + fw[t, 2] + fw[t, 3]
            for q in range(4):
                fw[t, q] /= z
            scale[t] = z
        for q in range(4):
            bw[B - 1, q] = 1.0
        for t in range(B - 1, 0, -1):
            _emis4(tc, cls, known, paths, t, delta, e)
            th = _switch(theta[t - 1], floor)
            st = 1.0 - th
            ratio_stay = th / st
            ratio_switch = st / th
            num = 0.0
            for p in range(4):
                acc = 0.0
                for q in range(4):
                    same_s = (p >> 1) == (q >> 1)
                    same_r = (p & 1) == (q & 1)
                    w = (st if same_s else th) * (st if same_r else th)
                    x = w * e[q] * bw[t, q]
                    acc += x
                    same = same_s if latent == 0 else same_r
                    num += fw[t - 1, p] * x * (ratio_stay if same else ratio_switch)
                bw[t - 1, p] = acc / scale[t]
            # sum_pq xi(p,q) r(p,q) = num / scale[t] (xi normalised)
            gains[t - 1] += math.log(num / scale[t])
    # prior change of the latent's own homologue paths at each boundary
    if T > 1:
        for t in range(B - 1):
            th = _switch(theta[t], floor)
            ls = math.log(1.0 - th)
            lc = math.log(th / (T - 1))
            a0, b0 = paths[latent, 0, t], paths[latent, 1, t]
            a1, b1 = paths[latent, 0, t + 1], paths[latent, 1, t + 1]
            old = (ls if a0 == a1 else lc) + (ls if b0 == b1 else lc)
            new = (ls if a0 == b1 else lc) + (ls if b0 == a1 else lc)
            gains[t] += new - old
    return gains


@kernel(overloads=OVERLOADS["family_transmission.fit"], cache=True, nogil=True)
def fit(tc, mcls, mknown, roles0, paths0, which, theta, floor, delta, max_iter, alone_inc, flex, allowed, lpr):
    """MAP-EM over the latent paths of the latent diploids flagged in which[2] and the roles of flexible members.
    lpr[5] are the role log priors (0, 1 direct of latent 0/1; 2 child; 4 outsider); a flexible member may take
    the roles flagged in allowed[5]; fixed members pay no role prior. Returns (log lik, inc[B], paths, iterations,
    roles). Monotone: no step lowers log P(data, paths, roles)."""
    B = tc.shape[1]
    paths = paths0.copy()
    roles = roles0.copy()
    n = len(roles)
    role_lp = np.zeros(n)
    for m in range(n):
        if flex[m]:
            role_lp[m] = lpr[roles[m]]
    gammas = np.zeros((n, B, 4))
    T = tc.shape[0]
    hgam = np.zeros((n, B, 2, T))
    inc = np.zeros(B)
    current = evaluate(tc, mcls, mknown, roles, paths, which, theta, floor, delta, inc, alone_inc, role_lp)
    it = 0
    for it in range(max_iter):
        improved = False
        for latent in range(2):
            if not which[latent]:
                continue
            junk = np.zeros(B)
            for m in range(n):
                if roles[m] == 2:
                    _child_fb(tc, mcls[m], mknown[m], paths, theta, floor, delta, True, gammas[m], junk)
                elif roles[m] == 5 + latent:
                    _half_fb(tc, mcls[m], mknown[m], paths, latent, theta, floor, delta, True, hgam[m], junk)
            new = _mstep(tc, mcls, mknown, roles, gammas, paths, latent, theta, floor, delta, False, hgam)
            trial = paths.copy()
            trial[latent, 0, :] = new[0]
            trial[latent, 1, :] = new[1]
            tinc = np.zeros(B)
            value = evaluate(tc, mcls, mknown, roles, trial, which, theta, floor, delta, tinc, alone_inc, role_lp)
            if value > current + 1e-9:
                paths = trial
                current = value
                inc = tinc
                improved = True
        if not improved:
            # roles of flexible members given the paths
            junk = np.zeros(B)
            dummy = np.empty((1, 4))
            dummy3 = np.empty((1, 1, 1))
            trial_roles = roles.copy()
            trial_lp = role_lp.copy()
            for m in range(n):
                if not flex[m]:
                    continue
                best_v = -np.inf
                best_r = roles[m]
                for r in (2, 0, 1, 4, 5, 6):
                    if not allowed[r]:
                        continue
                    if r == 2:
                        v = _child_fb(tc, mcls[m], mknown[m], paths, theta, floor, delta, False, dummy, junk)
                    elif r == 4:
                        v = alone_inc[m].sum()
                    elif r >= 5:
                        if not which[r - 5]:
                            continue
                        v = _half_fb(tc, mcls[m], mknown[m], paths, r - 5, theta, floor, delta, False, dummy3, junk)
                    else:
                        if not which[r]:
                            continue
                        v = _direct_inc(tc, mcls[m], mknown[m], paths, r, delta, junk)
                    v += lpr[r]
                    if v > best_v + 1e-9:
                        best_v = v
                        best_r = r
                trial_roles[m] = best_r
                trial_lp[m] = lpr[best_r]
            tinc = np.zeros(B)
            value = evaluate(tc, mcls, mknown, trial_roles, paths, which, theta, floor, delta, tinc, alone_inc, trial_lp)
            if value > current + 1e-9:
                roles = trial_roles
                role_lp = trial_lp
                current = value
                inc = tinc
                improved = True
        if not improved:
            # label-flip moves (one per latent per round), then back to EM
            for latent in range(2):
                if not which[latent]:
                    continue
                if B < 2:
                    continue  # a one-bin segment has no boundary to flip at
                g = flip_gains(tc, mcls, mknown, roles, paths, latent, theta, floor, delta)
                t = np.argmax(g)
                if g[t] > 1e-6:
                    trial = paths.copy()
                    for b in range(t + 1, B):
                        trial[latent, 0, b] = paths[latent, 1, b]
                        trial[latent, 1, b] = paths[latent, 0, b]
                    tinc = np.zeros(B)
                    value = evaluate(tc, mcls, mknown, roles, trial, which, theta, floor, delta, tinc, alone_inc,
                                     role_lp)
                    if value > current + 1e-9:
                        paths = trial
                        current = value
                        inc = tinc
                        improved = True
        if not improved:
            break
    return current, inc, paths, it + 1, roles


@kernel(overloads=OVERLOADS["family_transmission.init_latent"], cache=True, nogil=True)
def init_latent(tc, mcls, mknown, roles, paths, latent, theta, floor, delta):
    """Initial pair path of one latent diploid: the M step under uniform member selector posteriors."""
    n = len(roles)
    B = tc.shape[1]
    gammas = np.full((n, B, 4), 0.25)
    hgam = np.zeros((n, B, 2, tc.shape[0]))
    return _mstep(tc, mcls, mknown, roles, gammas, paths, latent, theta, floor, delta, True, hgam)


def _closest(paths, latent_pair, cpairs):
    """Index (0/1) of the latent diploid whose unordered homologue pair matches C's schedule in more bins."""
    agree = []
    for l in range(2):
        a, b = paths[l, 0], paths[l, 1]
        agree.append(int(np.sum(((a == cpairs[0]) & (b == cpairs[1])) | ((a == cpairs[1]) & (b == cpairs[0])))))
    return int(np.argmax(agree))


def _complement(p, cpairs):
    """PAR start from a REL/UNR solution: X = C's pair, Y = per bin the homologue of each latent diploid that C's
    schedule does not use (the first homologue when neither or both match)."""
    q = np.empty_like(p)
    q[0] = cpairs
    for l in range(2):
        a, b = p[l, 0], p[l, 1]
        uses_a = (a == cpairs[0]) | (a == cpairs[1])
        q[1, l] = np.where(uses_a, b, a)
    return q


def family_models(tc, mcls, mknown, cpairs, theta, floor, delta, max_iter=30, rounds=3, rho=0.01, n_random=3,
                  seed=None):
    """Profile log likelihoods and per-bin increments of PAR, REL and UNR for members [C, S] + witnesses.

    cpairs int64 [2, B]: C's scheduled trajectory pair. S (member 1) is always a child; each witness (members 2..)
    is a child of the latent pair, or (log prior rho each) a direct observation of a latent diploid (a sequenced
    parent of the family; under PAR only latent 1, the other parent) or an outsider explained by its own mosaic
    (a relative that is not a sib). Every model is refitted from starts derived from every other model's best
    latent paths (direct, homologues crossed between the diploids, maternal pair swapped, and for PAR the
    complement of C's homologues) until no model improves (at most ``rounds``). Returns dict(name -> (loglik,
    inc[B], paths, iterations, roles))."""
    B = tc.shape[1]
    n = len(mcls)
    k = n - 1
    yes = np.array([True, True])
    one = np.array([True, False])
    p0 = np.zeros((2, 2, B), dtype=np.int64)
    # each member alone (one latent diploid observed directly; exact by a single Viterbi step)
    alone_inc = np.zeros((n, B))
    alone_paths = np.zeros((n, 2, B), dtype=np.int64)
    none_inc = np.zeros((n, B))
    nflex = np.zeros(n, dtype=np.bool_)
    lpr0 = np.zeros(7)
    allow0 = np.zeros(7, dtype=np.bool_)
    for m in range(n):
        q = p0.copy()
        r = np.full(n, 3, dtype=np.int64)
        r[m] = 0
        res = fit(tc, mcls, mknown, r, q, one, theta, floor, delta, max_iter, none_inc, nflex, allow0, lpr0)
        alone_inc[m] = res[1]
        alone_paths[m] = res[2][0]
    calone = (float(alone_inc[0].sum()), alone_inc[0].copy(), None, 1, None)
    # v8b: half child of latent 0 or 1 (roles 5, 6) are flexible roles too; child prior 1 - 5 rho
    lc = math.log(1.0 - 5.0 * rho)
    lr = math.log(rho)
    lpr = np.array([lr, lr, lc, 0.0, lr, lr, lr])
    flex = np.zeros(n, dtype=np.bool_)
    flex[2:] = True
    allowed = dict(PAR=np.array([False, True, True, False, True, True, True]),
                   UNR=np.array([True, True, True, False, True, True, True]),
                   REL=np.array([True, True, True, False, True, True, True]))
    roles = dict(PAR=np.array([0] + [2] * k, dtype=np.int64), UNR=np.array([3] + [2] * k, dtype=np.int64),
                 REL=np.array([2] + [2] * k, dtype=np.int64))
    p0[0] = cpairs
    p0[1] = cpairs
    pp = p0.copy()
    pp[1] = init_latent(tc, mcls, mknown, roles["PAR"], pp, 1, theta, floor, delta)

    def run(name, start, start_roles):
        return fit(tc, mcls, mknown, start_roles, start.copy(), yes, theta, floor, delta, max_iter, alone_inc, flex,
                   allowed[name], lpr)

    best = dict(PAR=run("PAR", pp, roles["PAR"]))

    def crossed(p):
        q = p.copy()
        q[0, 1], q[1, 0] = p[1, 0].copy(), p[0, 1].copy()
        return q

    def swapped(p):
        q = p.copy()
        q[1, 0], q[1, 1] = p[1, 1].copy(), p[1, 0].copy()
        return q

    def starts_for(name):
        out = []
        for src, res in best.items():
            p = res[2]
            if name == "PAR":
                j = _closest(p, None, cpairs)
                s2 = np.empty_like(p)
                s2[0] = p[j]
                s2[1] = p[1 - j]
                out += [s2, _complement(p, cpairs)]
            else:
                out += [p, crossed(p), crossed(swapped(p))]
        return out

    def adopt(name):
        # joint moves: one witness becomes a direct observation of a latent diploid whose paths are reset to that
        # witness's own best mosaic (a sequenced parent inside the family, e.g. an unreleased co-parent)
        res0 = best[name]
        changed = False
        targets = (1,) if name == "PAR" else (0, 1)
        for m in range(2, n):
            for j in targets:
                q = res0[2].copy()
                q[j] = alone_paths[m]
                r = res0[4].copy()
                r[m] = j
                res = run(name, q, r)
                if res[0] > best[name][0] + 1e-9:
                    best[name] = res
                    changed = True
            # v8b: the witness becomes a half child of either latent diploid (paths kept; EM fills the free slot)
            for j in (0, 1):
                r = best[name][4].copy()
                if r[m] == 5 + j:
                    continue
                r[m] = 5 + j
                res = run(name, best[name][2].copy(), r)
                if res[0] > best[name][0] + 1e-9:
                    best[name] = res
                    changed = True
        return changed

    def seeded():
        # v7: member-seeded starts (S3): latent diploids initialised at members' own best mosaics, so the
        # alternatives are not reached only through PAR's solution
        k_seed = min(n, 4)
        for name in ("UNR", "REL"):
            for i in range(k_seed):
                for j in range(i + 1, k_seed):
                    q = np.empty_like(p0)
                    q[0] = alone_paths[i]
                    q[1] = alone_paths[j]
                    res = run(name, q, roles[name])
                    if res[0] > best.get(name, (-np.inf,))[0] + 1e-9:
                        best[name] = res
        for m in range(1, k_seed):
            q = p0.copy()
            q[1] = alone_paths[m]
            res = run("PAR", q, roles["PAR"])
            if res[0] > best["PAR"][0] + 1e-9:
                best["PAR"] = res

    def perturbed(rng):
        # v7: deterministic random block perturbations of each model's best paths (S3), seeded from the data
        changed = False
        T = tc.shape[0]
        for name in ("UNR", "REL", "PAR"):
            for _ in range(n_random):
                q = best[name][2].copy()
                lo = int(rng.integers(0, B))
                hi = min(B, lo + max(1, int(B * rng.uniform(0.1, 0.4))))
                lat = 1 if name == "PAR" else int(rng.integers(0, 2))
                h = int(rng.integers(0, 2))
                q[lat, h, lo:hi] = int(rng.integers(0, T))
                res = run(name, q, best[name][4])
                if res[0] > best[name][0] + 1e-9:
                    best[name] = res
                    changed = True
        return changed

    seeded()
    digest = hashlib.sha256(np.ascontiguousarray(mcls).tobytes() + np.ascontiguousarray(mknown).tobytes()).digest()
    rng = stream("pedigree.family_transmission.starts", seed=seed, key=int.from_bytes(digest[:8], "little"))
    tried = set()
    for _ in range(rounds):
        changed = False
        for name in ("UNR", "REL", "PAR"):
            for s0 in starts_for(name):
                key = (name, s0.tobytes())
                if key in tried:
                    continue
                tried.add(key)
                res = run(name, s0, roles[name])
                if name not in best or res[0] > best[name][0] + 1e-9:
                    best[name] = res
                    changed = True
        if n > 2:
            for name in ("UNR", "REL", "PAR"):
                changed = adopt(name) or changed
        changed = perturbed(rng) or changed
        if not changed:
            break
    return dict(C=calone, PAR=best["PAR"], UNR=best["UNR"], REL=best["REL"])


TESTED, KEPT, WITHHELD, UNTESTED = "tested", "kept", "withheld", "untested"
BASIS = "transmission"


def segments(theta: np.ndarray, unit_morgans: float, tail_morgans: float) -> tuple[np.ndarray, np.ndarray]:
    """Component-local evidence segments: int64 [B] segment of each bin and float64 [n] working Morgans.

    Consecutive pieces of ``unit_morgans`` of the component's working map (Haldane distances of its selector
    probabilities, as the material units); a final piece shorter than ``tail_morgans`` joins the previous one."""
    theta = np.asarray(theta, dtype=np.float64)
    distance = -0.5 * np.log1p(-2.0 * np.minimum(theta, 0.499999))
    position = np.concatenate(([0.0], np.cumsum(distance)))
    total = float(position[-1])
    count = max(1, int(math.floor(total / unit_morgans + (1.0 - tail_morgans / unit_morgans))))
    segment = np.minimum(np.floor(position / unit_morgans).astype(np.int64), count - 1)
    bounds = np.minimum(np.arange(count + 1, dtype=np.float64) * unit_morgans, total)
    bounds[-1] = total
    lengths = np.diff(bounds)
    used = np.zeros(count, dtype=np.bool_)
    used[segment] = True
    if not used.all():
        # one bin step longer than a unit leaves units without bins (unit 0 always holds bin 0): their Morgans join
        # the last unit before them that holds bins, so every segment has data and the total is kept
        renumber = np.cumsum(used) - 1
        owner = np.maximum.accumulate(np.where(used, np.arange(count), 0))
        merged = np.zeros(int(used.sum()))
        np.add.at(merged, renumber[owner], lengths)
        segment, lengths = renumber[segment], merged
    return segment, lengths


def segment_models(tc, mcls, mknown, cpairs, theta, floor, delta, ft, seed):
    """(PAR, REL, UNR) profile log likelihoods of one segment for members [C, S, witnesses...] (UNR includes C's own
    mosaic): the best, per model, of two searches from data-derived starts, the witnesses ordered by a digest of
    each witness's painted classes and known bits and in the reverse of that order. Each search only raises the
    profile, so the value is the better of two local optima; it depends neither on the input witness order nor on
    sample names (witnesses with identical data are interchangeable)."""
    n = mcls.shape[0]
    order = sorted(range(2, n), key=lambda w: hashlib.sha256(np.ascontiguousarray(mcls[w]).tobytes()
                                                             + np.ascontiguousarray(mknown[w]).tobytes()).digest())
    best = None
    for witnesses in ((order, order[::-1]) if n > 3 else (order,)):
        idx = np.asarray([0, 1] + witnesses, dtype=np.int64)
        res = family_models(tc, np.ascontiguousarray(mcls[idx]), np.ascontiguousarray(mknown[idx]), cpairs, theta,
                            floor, delta, max_iter=ft.maximum_iterations, rounds=ft.start_rounds, rho=ft.role_prior,
                            n_random=ft.random_starts, seed=seed)
        value = (res["PAR"][0], res["REL"][0], res["C"][0] + res["UNR"][0])
        best = value if best is None else tuple(max(a, b) for a, b in zip(best, value, strict=True))
    return best


def score_component(schedule, theta: np.ndarray, members: tuple[int, ...], parameters, delta: float,
                    floor: float) -> tuple[np.ndarray, np.ndarray]:
    """(float64 [n, 3] PAR, REL, UNR profile log likelihoods, float64 [n] Morgans) of one component's segments for
    members (C, S, witnesses...); every segment is fitted independently (segment_models)."""
    from .junctions import ComponentSchedule  # noqa: F401  (type of ``schedule``)

    ft = parameters.family_transmission
    m = parameters.material
    if schedule.classes.shape[1] == 0:
        return np.zeros((0, 3)), np.zeros(0)
    trajectories = _trajectory_classes(schedule)
    seg, lengths = segments(theta, m.unit_morgans, m.tail_morgans)
    idx = np.asarray(members, dtype=np.int64)
    out = np.zeros((len(lengths), 3))
    for u in range(len(lengths)):
        bins = np.flatnonzero(seg == u)
        lo, hi = int(bins[0]), int(bins[-1]) + 1
        out[u] = segment_models(
            np.ascontiguousarray(trajectories[:, lo:hi], dtype=np.int64),
            np.ascontiguousarray(schedule.classes[idx, lo:hi], dtype=np.int64),
            np.ascontiguousarray(schedule.known[idx, lo:hi], dtype=np.bool_),
            np.ascontiguousarray(schedule.pairs[idx[0], lo:hi].T, dtype=np.int64),
            np.array(theta[lo:hi - 1], dtype=np.float64, order="C", copy=True),
            float(floor), float(delta), ft, parameters.resampling.bootstrap_seed,
        )
    return out, lengths


def _trajectory_classes(schedule) -> np.ndarray:
    """int64 [T, B] bin-local class of every whole-component trajectory, recovered from the schedule."""
    if getattr(schedule, "trajectory_classes", None) is not None:
        return np.asarray(schedule.trajectory_classes, dtype=np.int64)
    raise ValueError("family transmission needs the schedule's trajectory class table")


def request(sample_ids, contigs, before, guard_record, relatedness, relationship_request, relationship_records,
            observed_counts, parameters):
    """(TransmissionRequest, witnesses per guard vote) after the guard's fixed point (PD29).

    Witnesses of a retained edge: the guard's tested members of the child's relatedness family (linked members
    removed against the retained edges), minus those failing the guard's family check, minus those with a retained
    parent other than C or a retained parent of S; at most ``maximum_witnesses`` by relatedness member support, then
    observed-site counts in configured contig order, then the witness's own family-check margins (no sample names;
    complete ties keep the guard's member order). The guard's 'both hypotheses fail' set-aside is not applied: it
    depends on C, the hypothesis under test."""
    from ...contracts.pedigree import TransmissionRequest
    from . import guard

    ft = parameters.family_transmission
    rp = parameters.relationship
    withheld = guard_record.withheld
    retained = {(int(p), int(c)) for p, c in zip(before.parents.tolist(), before.children.tolist(), strict=True)}
    retained -= set(withheld)
    parents_of: dict[int, set[int]] = {}
    for p, c in retained:
        parents_of.setdefault(c, set()).add(p)
    trio = guard.summed_margins(relationship_request, relationship_records)[1] if relationship_records else {}
    support = np.nan_to_num(np.asarray(relatedness.member_support, dtype=np.float64), nan=0.0)
    counts = np.asarray(observed_counts, dtype=np.int64)
    witnesses: dict[tuple[int, int], int] = {}
    tested: list[tuple[int, int, list[int]]] = []
    for vote in guard_record.votes:
        f = vote.family
        if vote.call == guard.WITHHELD:
            continue
        c, p = int(f.child), int(f.parent)
        partners = guard.check_partner_map(f, rp.sibship_check_partners)
        keep = []
        content: dict[int, tuple[float, ...]] = {}
        for w in f.tested:
            low = min(support[c], support[w])
            band = int(low >= rp.support_bounds[0]) + int(low >= rp.support_bounds[1])
            price = max(0.0, math.log(rp.sibship_prior[band] / (1.0 - rp.sibship_prior[band])))
            others = partners[w]
            if others and float(np.median([trio[tuple(sorted((c, w, o)))] for o in others])) <= -price:
                continue
            if parents_of.get(int(w), set()) - {p} - parents_of.get(c, set()):
                continue
            keep.append(int(w))
            # content key: the witness's own family-check margins (data, independent of C and of sample names)
            content[int(w)] = tuple(sorted(-round(float(trio[tuple(sorted((c, w, o)))]), 9) for o in others))
        keep.sort(key=lambda w: (-float(support[w]), tuple(-int(x) for x in counts[:, w]), content[w]))
        keep = keep[: ft.maximum_witnesses]
        witnesses[(p, c)] = len(keep)
        if len(keep) >= ft.minimum_witnesses:
            tested.append((c, p, keep))
    tested.sort(key=lambda row: (row[0], row[1]))  # canonical edge order (child, parent): vote order is immaterial
    parents = [p for _, p, _ in tested]
    children = [c for c, _, _ in tested]
    members = [w for _, _, keep in tested for w in keep]
    offsets = np.concatenate(([0], np.cumsum([len(keep) for _, _, keep in tested]))).astype(np.int64)
    req = TransmissionRequest(
        tuple(sample_ids), tuple(contigs), np.asarray(parents, dtype=np.int64), np.asarray(children, dtype=np.int64),
        offsets, np.asarray(members, dtype=np.int64),
    )
    return req, witnesses


def robust_log_odds(difference: float, error_probability: float) -> float:
    """PD06's robust per-unit log odds of a log-likelihood difference (capped at ln((2 - eps) / eps))."""
    if difference == 0.0:
        return 0.0
    keep = 1.0 - error_probability
    floor = 0.5 * error_probability
    scaled = math.exp(-abs(difference))
    value = math.log(keep / (1.0 + scaled) + floor) - math.log(keep * scaled / (1.0 + scaled) + floor)
    return value if difference > 0.0 else -value


def apply(guard_record, req, witnesses, evidence, parameters):
    """GuardRecord with every retained edge's PD29 vote; a withheld edge is withdrawn one pass after the guard's
    fixed point (no iteration: the guard's cleaning is not recomputed)."""
    from dataclasses import replace

    from .orientation import unit_failure

    rp = parameters.relationship
    ft = parameters.family_transmission
    for record in evidence:
        if record.request_digest != req.digest or len(record.offsets) != len(req.parents) + 1:
            raise ValueError("family transmission evidence differs from its request")
    order = {(int(p), int(c)): e for e, (p, c) in enumerate(zip(req.parents.tolist(), req.children.tolist()))}
    votes = []
    for vote in guard_record.votes:
        f = vote.family
        if vote.call == "withheld":
            votes.append(vote)
            continue
        key = (int(f.parent), int(f.child))
        k = witnesses[key]
        if k < ft.minimum_witnesses or key not in order:
            votes.append(replace(vote, transmission_call=UNTESTED, transmission_witnesses=int(k)))
            continue
        e = order[key]
        ll = np.concatenate([r.log_likelihoods[r.offsets[e]:r.offsets[e + 1]] for r in evidence], axis=0)
        morgans = np.concatenate([r.morgans[r.offsets[e]:r.offsets[e + 1]] for r in evidence])
        total = ll.sum(axis=0)
        alternative = 1 if total[1] >= total[2] else 2
        eps = unit_failure(morgans, rp.failure_region_rate, rp.failure_region_length)
        score = float(sum(robust_log_odds(float(d), float(x)) for d, x in zip(ll[:, 0] - ll[:, alternative], eps)))
        changes = dict(transmission_witnesses=int(k), transmission_segments=int(len(morgans)),
                       transmission_score=score, transmission_alternative="REL" if alternative == 1 else "UNR")
        if score < -rp.withholding_score:
            votes.append(replace(vote, call="withheld", evidence_call="withheld", relationship_basis="",
                                 withholding_basis=BASIS, withdrawal_iteration=guard_record.cleaning_iterations + 1,
                                 transmission_call=WITHHELD, **changes))
        else:
            votes.append(replace(vote, transmission_call=KEPT, **changes))
    return replace(guard_record, votes=tuple(votes))
