"""Optimize latent homologue paths and member roles for a family transmission model."""

from __future__ import annotations

import math

import numpy as np

from ...core.kernels import kernel
from .family_likelihood import NEG, _child_fb, _emit, _evaluate_cached, _half_fb, _member_value, _switch
from .kernels.family_transmission import OVERLOADS


@kernel(overloads=OVERLOADS["family_fit._prepare"], cache=True, nogil=True)
def _prepare(tc, mcls, mknown, theta, floor, mdelta):
    """Fixed member/trajectory-pair log emissions, reused by all starts and M steps of this search.

    The table uses _emit and math.log unchanged; every subsequent likelihood reduction retains its original
    member, slot and state order. Unknown bins are never read by the M step. Also prepare the first trajectory
    of each bin-local class and the exact mosaic transition logs shared by every fit.
    """
    T, B = tc.shape
    representatives = np.empty((B, T), dtype=np.int64)
    transition = np.empty((B - 1, 2))
    for t in range(B):
        for a in range(T):
            representatives[t, a] = a
            for b in range(a):
                if tc[a, t] == tc[b, t]:
                    representatives[t, a] = representatives[t, b]
                    break
        if t > 0:
            th = _switch(theta[t - 1], floor)
            transition[t - 1, 0] = math.log(1.0 - th)
            transition[t - 1, 1] = math.log(th / (T - 1)) if T > 1 else NEG
    out = np.empty((len(mcls), B, T, T))
    for m in range(len(mcls)):
        for t in range(B):
            if not mknown[m, t]:
                continue
            x, y = mcls[m, t, 0], mcls[m, t, 1]
            for a in range(T):
                for b in range(T):
                    out[m, t, a, b] = math.log(_emit(x, y, tc[a, t], tc[b, t],
                            mdelta[m, t, 0], mdelta[m, t, 1], mdelta[m, t, 2], mdelta[m, t, 3]))
    return out, representatives, transition


@kernel(overloads=OVERLOADS["family_fit._pair_transition"], cache=True, nogil=True, inline="always")
def _pair_transition(V, score, back, t, ls, lc, rowmax, rowarg, colmax, colarg, NV, T):
    """One pair Viterbi transition, preserving the released stay/row/column/global comparison order.

    Inlining the common fixed dimensions lets the compiler unroll these short loops; the general T path is
    the very same recurrence. T is an implementation dimension, never a model limit.
    """
    gmax = NEG
    ga = 0
    gb = 0
    for a in range(T):
        colmax[a] = NEG
    for a in range(T):
        row_best = NEG
        row_arg = 0
        for b in range(T):
            v = V[a, b]
            if v > row_best:
                row_best = v
                row_arg = b
            if v > colmax[b]:
                colmax[b] = v
                colarg[b] = a
            if v > gmax:
                gmax = v
                ga = a
                gb = b
        rowmax[a] = row_best
        if row_best > NEG:
            rowarg[a] = row_arg
    # These candidate values are identical for every cell of a row/column. Materialize them once;
    # addition order and the stay, row, column, global comparison sequence below remain unchanged.
    for a in range(T):
        rowmax[a] = rowmax[a] + ls + lc
        colmax[a] = colmax[a] + lc + ls
    global_value = gmax + 2.0 * lc
    global_arg = ga * T + gb
    for a in range(T):
        for b in range(T):
            best = V[a, b] + 2.0 * ls
            arg = a * T + b
            c = rowmax[a]
            if c > best:
                best = c
                arg = a * T + rowarg[a]
            c = colmax[b]
            if c > best:
                best = c
                arg = colarg[b] * T + b
            c = global_value
            if c > best:
                best = c
                arg = global_arg
            NV[a, b] = best + score[a, b]
            back[t, a, b] = arg


@kernel(overloads=OVERLOADS["family_fit._mstep"], cache=True, nogil=True)
def _mstep(tc, mcls, mknown, roles, gammas, paths, latent, theta, floor, mdelta, maxmode, hgam, loge,
           representatives, transition):
    """Viterbi over ordered homologue pairs (a, b) of one latent diploid with the other latent fixed.
    Maximises sum_t expected complete log emission + mosaic prior. Returns new paths [2, B] for that latent."""
    T = tc.shape[0]
    B = tc.shape[1]
    other = 1 - latent
    V = np.empty((T, T))
    back = np.empty((B, T, T), dtype=np.int32)
    score = np.zeros((T, T))
    rowmax = np.empty(T)
    rowarg = np.empty(T, dtype=np.int64)
    colmax = np.empty(T)
    colarg = np.empty(T, dtype=np.int64)
    NV = np.empty((T, T))
    accs = np.empty((2, T))
    classes = np.empty(T, dtype=np.int64)
    unique = np.empty(T, dtype=np.int64)
    for t in range(B):
        # Only emission scores are compressed. Every distinct trajectory remains in the Viterbi recurrence,
        # with its original prior, index, transition and tie order. Dense class indices keep score writes local.
        count = 0
        for a in range(T):
            representative = representatives[t, a]
            if representative == a:
                unique[count] = a
                classes[a] = count
                count += 1
            else:
                classes[a] = classes[representative]
        for a in range(count):
            for b in range(count):
                score[a, b] = 0.0
        for m in range(len(roles)):
            if not mknown[m, t]:
                continue
            if roles[m] == latent:
                for a in range(count):
                    for b in range(count):
                        score[a, b] += loge[m, t, unique[a], unique[b]]
            elif roles[m] == 5 + latent:
                for slot in range(2):
                    for a in range(count):
                        acc = 0.0
                        for f in range(T):
                            g = hgam[m, t, slot, f]
                            if g > 0.0:
                                acc += g * loge[m, t, unique[a], f]
                        accs[slot, a] = acc
                for a in range(count):
                    for b in range(count):
                        score[a, b] += accs[0, a]
                        score[a, b] += accs[1, b]
            elif roles[m] == 2 and maxmode:
                for a in range(count):
                    for b in range(count):
                        best = NEG
                        for o in range(2):
                            va = loge[m, t, unique[a], paths[other, o, t]]
                            vb = loge[m, t, unique[b], paths[other, o, t]]
                            if va > best:
                                best = va
                            if vb > best:
                                best = vb
                        score[a, b] += best
            elif roles[m] == 2:
                for slot in range(2):
                    for a in range(count):
                        acc = 0.0
                        for o in range(2):
                            if latent == 0:
                                g = gammas[m, t, 2 * slot + o]
                            else:
                                g = gammas[m, t, 2 * o + slot]
                            if g > 0.0:
                                if latent == 0:
                                    acc += g * loge[m, t, unique[a], paths[other, o, t]]
                                else:
                                    acc += g * loge[m, t, paths[other, o, t], unique[a]]
                        accs[slot, a] = acc
                for a in range(count):
                    for b in range(count):
                        score[a, b] += accs[0, a]
                        score[a, b] += accs[1, b]
        if t == 0:
            for a in range(T):
                for b in range(T):
                    V[a, b] = score[classes[a], classes[b]] - 2.0 * math.log(T)
            continue
        # Expand into the next-value buffer. Each transition reads its own cell's score before replacing it.
        for a in range(T):
            for b in range(T):
                NV[a, b] = score[classes[a], classes[b]]
        ls = transition[t - 1, 0]
        lc = transition[t - 1, 1]
        if T == 6:
            _pair_transition(V, NV, back, t, ls, lc, rowmax, rowarg, colmax, colarg, NV, 6)
        elif T == 8:
            _pair_transition(V, NV, back, t, ls, lc, rowmax, rowarg, colmax, colarg, NV, 8)
        else:
            _pair_transition(V, NV, back, t, ls, lc, rowmax, rowarg, colmax, colarg, NV, T)
        V, NV = NV, V
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


@kernel(overloads=OVERLOADS["family_fit._emis4"], cache=True, nogil=True)
def _emis4(tc, cls, known, paths, t, dm, e):
    for q in range(4):
        e[q] = 1.0
    if known[t]:
        for s in range(2):
            u = tc[paths[0, s, t], t]
            for r in range(2):
                e[2 * s + r] = _emit(cls[t, 0], cls[t, 1], u, tc[paths[1, r, t], t],
                        dm[t, 0], dm[t, 1], dm[t, 2], dm[t, 3])


@kernel(overloads=OVERLOADS["family_fit._half_flip"], cache=True, nogil=True)
def _half_flip(tc, cls, known, paths, latent, theta, floor, dm, gains):
    """Add to gains[t - 1] the exact log-likelihood change, for one half child of ``latent``, of swapping that
    latent's homologue labels on every bin from t on: its selector transition at boundary t inverts (stay <-> switch)
    while its free mosaic is unchanged; exact through the pairwise posteriors of the (selector, mosaic) chain."""
    T = tc.shape[0]
    B = cls.shape[0]
    fw = np.empty((B, 2, T))
    scale = np.empty(B)
    emissions = np.empty((B, 2, T))
    for t in range(B):
        if t == 0:
            for sl in range(2):
                for f in range(T):
                    e = 1.0
                    if known[t]:
                        e = _emit(cls[t, 0], cls[t, 1], tc[paths[latent, sl, t], t], tc[f, t],
                                dm[t, 0], dm[t, 1], dm[t, 2], dm[t, 3])
                    emissions[t, sl, f] = e
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
                        e = _emit(cls[t, 0], cls[t, 1], tc[paths[latent, sl, t], t], tc[f, t],
                                dm[t, 0], dm[t, 1], dm[t, 2], dm[t, 3])
                    emissions[t, sl, f] = e
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
                g[sl, f] = emissions[t, sl, f] * bw[sl, f]
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


@kernel(overloads=OVERLOADS["family_fit._flip_gains_both"], cache=True, nogil=True)
def _flip_gains_both(tc, mcls, mknown, roles, paths, theta, floor, mdelta):
    """Exact log-likelihood changes [latent, boundary] for both latent diploids, sharing selector messages.

    Swapping homologue labels on every bin after boundary t (t = 0 .. B-2): the children's selector transition on that latent inverts at t (exact through the
    pairwise posteriors, the likelihood being linear in one transition matrix) and the latent's mosaic prior
    changes at t; direct observations are unordered and unchanged; half children of the latent add their own
    exact selector term (_half_flip)."""
    T = tc.shape[0]
    B = tc.shape[1]
    gains = np.zeros((2, B - 1))
    fw = np.empty((B, 4))
    scale = np.empty(B)
    bw = np.empty((B, 4))
    emissions = np.empty((B, 4))
    for m in range(len(roles)):
        if roles[m] >= 5:
            latent = roles[m] - 5
            _half_flip(tc, mcls[m], mknown[m], paths, latent, theta, floor, mdelta[m], gains[latent])
            continue
        if roles[m] != 2:
            continue
        cls = mcls[m]
        known = mknown[m]
        for t in range(B):
            e = emissions[t]
            _emis4(tc, cls, known, paths, t, mdelta[m], e)
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
            e = emissions[t]
            th = _switch(theta[t - 1], floor)
            st = 1.0 - th
            ratio_stay = th / st
            ratio_switch = st / th
            num0 = 0.0
            num1 = 0.0
            for p in range(4):
                acc = 0.0
                for q in range(4):
                    same_s = (p >> 1) == (q >> 1)
                    same_r = (p & 1) == (q & 1)
                    w = (st if same_s else th) * (st if same_r else th)
                    x = w * e[q] * bw[t, q]
                    acc += x
                    num0 += fw[t - 1, p] * x * (ratio_stay if same_s else ratio_switch)
                    num1 += fw[t - 1, p] * x * (ratio_stay if same_r else ratio_switch)
                bw[t - 1, p] = acc / scale[t]
            # sum_pq xi(p,q) r(p,q) = num / scale[t] (xi normalised)
            gains[0, t - 1] += math.log(num0 / scale[t])
            gains[1, t - 1] += math.log(num1 / scale[t])
    for latent in range(2):
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
                gains[latent, t] += new - old
    return gains


@kernel(overloads=OVERLOADS["family_fit.flip_gains"], cache=True, nogil=True)
def flip_gains(tc, mcls, mknown, roles, paths, latent, theta, floor, mdelta):
    """Exact label-flip gains for one latent; both selector terms share the same messages."""
    return _flip_gains_both(tc, mcls, mknown, roles, paths, theta, floor, mdelta)[latent]


@kernel(overloads=OVERLOADS["family_fit.fit"], cache=True, nogil=True)
def fit(tc, mcls, mknown, roles0, paths0, which, theta, floor, mdelta, max_iter, alone_inc, flex, allowed, lpr, loge,
        representatives, transition):
    """MAP-EM over the latent paths of the latent diploids flagged in which[2] and the roles of flexible members.
    lpr[7] are the role log priors (0, 1 direct; 2 child; 4 outsider; 5, 6 half child); a flexible member may take
    the roles flagged in allowed[7]; fixed members pay no role prior. Returns (log lik, inc[B], paths, iterations,
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
    gamma_valid = np.zeros(n, dtype=np.bool_)
    half_valid = np.zeros(n, dtype=np.bool_)
    T = tc.shape[0]
    hgam = np.zeros((n, B, 2, T))
    inc = np.zeros(B)
    valid = np.zeros((n, 7), dtype=np.bool_)
    values = np.zeros((n, 7))
    increments = np.zeros((n, 7, B))
    # The initial evaluation and first E step share their forward pass.
    for m in range(n):
        r = roles[m]
        if r == 2:
            values[m, r] = _child_fb(tc, mcls[m], mknown[m], paths, theta, floor, mdelta[m], True, gammas[m],
                                      increments[m, r])
            valid[m, r] = True
            gamma_valid[m] = True
        elif r >= 5:
            values[m, r] = _half_fb(tc, mcls[m], mknown[m], paths, r - 5, theta, floor, mdelta[m], True, hgam[m],
                                     increments[m, r])
            valid[m, r] = True
            half_valid[m] = True
    current = _evaluate_cached(tc, mcls, mknown, roles, paths, which, theta, floor, mdelta, inc, alone_inc, role_lp,
                               valid, values, increments, transition)
    it = 0
    for it in range(max_iter):
        improved = False
        for latent in range(2):
            if not which[latent]:
                continue
            for m in range(n):
                if roles[m] == 2 and not gamma_valid[m]:
                    increments[m, 2, :] = 0.0
                    values[m, 2] = _child_fb(tc, mcls[m], mknown[m], paths, theta, floor, mdelta[m], True, gammas[m],
                                              increments[m, 2])
                    valid[m, 2] = True
                    gamma_valid[m] = True
                elif roles[m] == 5 + latent and not half_valid[m]:
                    increments[m, 5 + latent, :] = 0.0
                    values[m, 5 + latent] = _half_fb(tc, mcls[m], mknown[m], paths, latent, theta, floor, mdelta[m],
                                                     True, hgam[m], increments[m, 5 + latent])
                    valid[m, 5 + latent] = True
                    half_valid[m] = True
            new = _mstep(tc, mcls, mknown, roles, gammas, paths, latent, theta, floor, mdelta, False, hgam, loge,
                         representatives, transition)
            if np.array_equal(new, paths[latent]):
                continue  # the exact same paths have the exact same evaluation
            trial = paths.copy()
            trial[latent, 0, :] = new[0]
            trial[latent, 1, :] = new[1]
            tinc = np.zeros(B)
            tvalid = valid.copy()
            tvalues = values.copy()
            tincrements = increments.copy()
            tvalid[:, 2] = False
            tvalid[:, latent] = False
            tvalid[:, 5 + latent] = False
            value = _evaluate_cached(tc, mcls, mknown, roles, trial, which, theta, floor, mdelta, tinc, alone_inc,
                                     role_lp, tvalid, tvalues, tincrements, transition)
            if value > current + 1e-9:
                paths = trial
                valid, values, increments = tvalid, tvalues, tincrements
                gamma_valid[:] = False
                for m in range(n):
                    if roles[m] == 5 + latent:
                        half_valid[m] = False
                current = value
                inc = tinc
                improved = True
        if not improved:
            # roles of flexible members given the paths
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
                    if r >= 5 and not which[r - 5]:
                        continue
                    if r < 2 and not which[r]:
                        continue
                    v = _member_value(tc, mcls, mknown, paths, theta, floor, mdelta, alone_inc, m, r,
                                      valid, values, increments)
                    v += lpr[r]
                    if v > best_v + 1e-9:
                        best_v = v
                        best_r = r
                trial_roles[m] = best_r
                trial_lp[m] = lpr[best_r]
            if not np.array_equal(trial_roles, roles):
                tinc = np.zeros(B)
                value = _evaluate_cached(tc, mcls, mknown, trial_roles, paths, which, theta, floor, mdelta, tinc,
                                         alone_inc, trial_lp, valid, values, increments, transition)
                if value > current + 1e-9:
                    roles = trial_roles
                    role_lp = trial_lp
                    gamma_valid[:] = False
                    half_valid[:] = False
                    current = value
                    inc = tinc
                    improved = True
        if not improved:
            # The two latent flip gains share messages until an accepted flip changes the paths.
            flips_valid = False
            gains = np.empty((2, 0))
            for latent in range(2):
                if not which[latent]:
                    continue
                if B < 2:
                    continue  # a one-bin segment has no boundary to flip at
                if not flips_valid:
                    gains = _flip_gains_both(tc, mcls, mknown, roles, paths, theta, floor, mdelta)
                    flips_valid = True
                g = gains[latent]
                t = np.argmax(g)
                if g[t] > 1e-6:
                    trial = paths.copy()
                    for b in range(t + 1, B):
                        trial[latent, 0, b] = paths[latent, 1, b]
                        trial[latent, 1, b] = paths[latent, 0, b]
                    tinc = np.zeros(B)
                    tvalid = valid.copy()
                    tvalues = values.copy()
                    tincrements = increments.copy()
                    tvalid[:, 2] = False
                    tvalid[:, latent] = False
                    tvalid[:, 5 + latent] = False
                    value = _evaluate_cached(tc, mcls, mknown, roles, trial, which, theta, floor, mdelta, tinc,
                                             alone_inc, role_lp, tvalid, tvalues, tincrements, transition)
                    if value > current + 1e-9:
                        paths = trial
                        flips_valid = False
                        valid, values, increments = tvalid, tvalues, tincrements
                        gamma_valid[:] = False
                        for m in range(n):
                            if roles[m] == 5 + latent:
                                half_valid[m] = False
                        current = value
                        inc = tinc
                        improved = True
        if not improved:
            break
    return current, inc, paths, it + 1, roles


@kernel(overloads=OVERLOADS["family_fit.init_latent"], cache=True, nogil=True)
def init_latent(tc, mcls, mknown, roles, paths, latent, theta, floor, mdelta, loge, representatives, transition):
    """Initial pair path of one latent diploid: the M step under uniform member selector posteriors."""
    n = len(roles)
    B = tc.shape[1]
    gammas = np.full((n, B, 4), 0.25)
    hgam = np.zeros((n, B, 2, tc.shape[0]))
    return _mstep(tc, mcls, mknown, roles, gammas, paths, latent, theta, floor, mdelta, True, hgam, loge,
                  representatives, transition)
