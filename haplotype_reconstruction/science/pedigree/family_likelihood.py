"""Family transmission likelihoods of a released parent edge on scheduled paintings (PD29).

One painting segment (B bins of one component, T whole-component trajectories with bin-local classes tc[T, B]):
two latent (unsequenced) diploids, each with two homologue paths over the trajectories under PD06's founder-mosaic
law (start 1/T; stay 1 - theta; change to a named other trajectory theta / (T - 1)). Each observed member is a
direct observation of latent 0 or 1 (role 0, 1), a child of the pair (role 2: a paternal selector over latent 0's
homologues and a maternal selector over latent 1's, each switching with the meiotic probability theta), ignored
(role 3), an outsider explained by its own best mosaic (role 4) or a half child of latent 0 or 1 (roles 5, 6: one
homologue through a meiotic selector over that latent's homologues, the other a free founder mosaic under the same
law). Emission of an unordered painted pair {x, y} given true classes (u, v): the probability of the member labels'
error pattern under the orientation x <- u, y <- v, plus that of x <- v, y <- u when x != y (_emit); bins without
both labels emit 1. Each member and bin carries its error-pattern law (4.22 R4b;
family_transmission.member_allowances): both right, label 0 wrong only, label 1 wrong only, both wrong, a wrong
label's whole mass given to whichever class a hypothesis needs, from the reference painting's untempered label-error
statistic (u, v_0, v_1) under the 4.20 floor (only the member's labels can be wrong: the latent trajectories'
classes are not painted); ``mdelta`` float64 [M, B, 4] per member, ``dm`` float64 [B, 4] for one member. The profile
log likelihood maximises over the latent paths and the flexible members' roles (MAP-EM with exact selector
posteriors, pair-path Viterbi M steps, exact homologue-label flip moves and role updates; monotone); member
selectors are summed exactly.

Method: docs/methods/pedigree.md#pd29
"""

from __future__ import annotations

import math

import numpy as np

from ...core.kernels import kernel
from .kernels.family_transmission import OVERLOADS

NEG = -1e300


@kernel(overloads=OVERLOADS["family_likelihood._emit"], cache=True, nogil=True)
def _emit(x, y, u, v, rr, wr, rw, ww):
    """Emission of a member's painted unordered pair {x, y} (label 0 = x, label 1 = y) given true classes (u, v):
    the error-pattern probabilities (rr both right, wr label 0 wrong only, rw label 1 wrong only, ww both wrong;
    member_allowances) of the orientation x <- u, y <- v, plus those of x <- v, y <- u
    when x != y. Exactly one pattern applies per orientation (selected, never summed with zero weights)."""
    if x == u:
        first = rr if y == v else rw
    else:
        first = wr if y == v else ww
    if x == y:
        return first
    if x == v:
        second = rr if y == u else rw
    else:
        second = wr if y == u else ww
    return first + second


@kernel(overloads=OVERLOADS["family_likelihood._switch"], cache=True, nogil=True)
def _switch(theta, floor):
    return min(0.5, max(floor, theta))


@kernel(overloads=OVERLOADS["family_likelihood._child_fb"], cache=True, nogil=True)
def _child_fb(tc, cls, known, paths, theta, floor, dm, want_gamma, gamma, inc):
    """Forward-backward of one child member (cls [B,2], known [B]) given latent paths [2,2,B] (trajectories).
    Adds per-bin log scaling to inc[B]; fills gamma[B,4] (state = 2*s + r) if want_gamma. Returns log lik."""
    B = cls.shape[0]
    fw = np.empty((B, 4))
    scale = np.empty(B)
    total = 0.0
    emissions = np.ones((B, 4))
    for t in range(B):
        e = emissions[t]
        if known[t]:
            for s in range(2):
                u = tc[paths[0, s, t], t]
                for r in range(2):
                    v = tc[paths[1, r, t], t]
                    e[2 * s + r] = _emit(cls[t, 0], cls[t, 1], u, v, dm[t, 0], dm[t, 1], dm[t, 2], dm[t, 3])
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
        nb = np.empty(4)
        for t in range(B - 1, -1, -1):
            g = 0.0
            for q in range(4):
                gamma[t, q] = fw[t, q] * bw[q]
                g += gamma[t, q]
            for q in range(4):
                gamma[t, q] /= g
            if t > 0:
                e = emissions[t]
                th = _switch(theta[t - 1], floor)
                st = 1.0 - th
                for p in range(4):
                    ps, pr = p >> 1, p & 1
                    acc = 0.0
                    for q in range(4):
                        s, r = q >> 1, q & 1
                        w = (st if ps == s else th) * (st if pr == r else th)
                        acc += w * e[q] * bw[q]
                    nb[p] = acc / scale[t]
                bw, nb = nb, bw
    return total



@kernel(overloads=OVERLOADS["family_likelihood._half_fb"], cache=True, nogil=True)
def _half_fb(tc, cls, known, paths, latent, theta, floor, dm, want_gamma, gamma, inc):
    """Half child of one latent diploid: homologue A copied from latent ``latent`` by a meiotic selector s, homologue
    B a free founder mosaic f (PD06 law). States (s, f); gamma[B, 2, T] posterior if want_gamma; log lik summed."""
    T = tc.shape[0]
    B = cls.shape[0]
    fw = np.empty((B, 2, T))
    scale = np.empty(B)
    emissions = np.empty((B if want_gamma else 0, 2, T))
    total = 0.0
    sel = np.empty(T)
    eb = np.empty((2, T))
    mo = np.empty((2, T))
    for t in range(B):
        if t == 0:
            for sl in range(2):
                for f in range(T):
                    e = 1.0
                    if known[t]:
                        e = _emit(cls[t, 0], cls[t, 1], tc[paths[latent, sl, t], t], tc[f, t],
                                dm[t, 0], dm[t, 1], dm[t, 2], dm[t, 3])
                    if want_gamma:
                        emissions[t, sl, f] = e
                    fw[t, sl, f] = 0.5 / T * e
        else:
            th = _switch(theta[t - 1], floor)
            st = 1.0 - th
            ch = th / (T - 1) if T > 1 else 0.0
            for sl in range(2):
                tot = 0.0
                for f in range(T):
                    sel[f] = fw[t - 1, sl, f] * st + fw[t - 1, 1 - sl, f] * th
                    tot += sel[f]
                for f in range(T):
                    v = sel[f] * (1.0 - th) + (tot - sel[f]) * ch
                    e = 1.0
                    if known[t]:
                        e = _emit(cls[t, 0], cls[t, 1], tc[paths[latent, sl, t], t], tc[f, t],
                                dm[t, 0], dm[t, 1], dm[t, 2], dm[t, 3])
                    if want_gamma:
                        emissions[t, sl, f] = e
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
                for sl in range(2):
                    for f in range(T):
                        eb[sl, f] = emissions[t, sl, f] * bw[sl, f]
                # transpose of mosaic step then selector step
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

@kernel(overloads=OVERLOADS["family_likelihood._prior_inc"], cache=True, nogil=True)
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


@kernel(overloads=OVERLOADS["family_likelihood._direct_inc"], cache=True, nogil=True)
def _direct_inc(tc, cls, known, paths, latent, dm, inc):
    B = cls.shape[0]
    total = 0.0
    for t in range(B):
        if known[t]:
            v = math.log(_emit(cls[t, 0], cls[t, 1], tc[paths[latent, 0, t], t], tc[paths[latent, 1, t], t],
                    dm[t, 0], dm[t, 1], dm[t, 2], dm[t, 3]))
            inc[t] += v
            total += v
    return total


@kernel(overloads=OVERLOADS["family_likelihood.evaluate"], cache=True, nogil=True)
def evaluate(tc, mcls, mknown, roles, paths, which, theta, floor, mdelta, inc, alone_inc, role_lp):
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
            total += _direct_inc(tc, mcls[m], mknown[m], paths, 0, mdelta[m], inc)
        elif roles[m] == 1:
            total += _direct_inc(tc, mcls[m], mknown[m], paths, 1, mdelta[m], inc)
        elif roles[m] == 2:
            total += _child_fb(tc, mcls[m], mknown[m], paths, theta, floor, mdelta[m], False, dummy, inc)
        elif roles[m] == 5 or roles[m] == 6:
            total += _half_fb(tc, mcls[m], mknown[m], paths, roles[m] - 5, theta, floor, mdelta[m], False, dummy3, inc)
    return total


@kernel(overloads=OVERLOADS["family_likelihood._prior_cached"], cache=True, nogil=True)
def _prior_cached(paths, which, T, transition, inc):
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
                v = transition[t - 1, 0 if paths[l, h, t] == paths[l, h, t - 1] else 1]
                inc[t] += v
                total += v
    return total


@kernel(overloads=OVERLOADS["family_likelihood._member_value"], cache=True, nogil=True)
def _member_value(tc, mcls, mknown, paths, theta, floor, mdelta, alone_inc, m, r, valid, values, increments):
    """One role's exact member likelihood, cached until a latent path it uses changes."""
    if valid[m, r]:
        return values[m, r]
    B = tc.shape[1]
    if r == 4:
        value = alone_inc[m].sum()
    else:
        for t in range(B):
            increments[m, r, t] = 0.0
        if r == 2:
            dummy = np.empty((1, 4))
            value = _child_fb(tc, mcls[m], mknown[m], paths, theta, floor, mdelta[m], False, dummy, increments[m, r])
        elif r >= 5:
            dummy3 = np.empty((1, 1, 1))
            value = _half_fb(tc, mcls[m], mknown[m], paths, r - 5, theta, floor, mdelta[m], False, dummy3,
                             increments[m, r])
        else:
            value = _direct_inc(tc, mcls[m], mknown[m], paths, r, mdelta[m], increments[m, r])
    values[m, r] = value
    valid[m, r] = True
    return value


@kernel(overloads=OVERLOADS["family_likelihood._evaluate_cached"], cache=True, nogil=True)
def _evaluate_cached(tc, mcls, mknown, roles, paths, which, theta, floor, mdelta, inc, alone_inc, role_lp,
                     valid, values, increments, transition):
    """Evaluate with reusable member terms, retaining evaluate's exact total and per-bin addition order.

    In particular, outsiders add each bin to the total (not their pre-summed member likelihood). The latent
    prior is still evaluated in its original homologue/bin order. The caller owns and invalidates the cache.
    """
    T, B = tc.shape
    total = _prior_cached(paths, which, T, transition, inc)
    for m in range(len(roles)):
        inc[0] += role_lp[m]
        total += role_lp[m]
        r = roles[m]
        if r == 4:
            for t in range(B):
                inc[t] += alone_inc[m, t]
                total += alone_inc[m, t]
        elif r != 3:
            total += _member_value(tc, mcls, mknown, paths, theta, floor, mdelta, alone_inc, m, r,
                                   valid, values, increments)
            for t in range(B):
                if r >= 2 or mknown[m, t]:
                    inc[t] += increments[m, r, t]
    return total
