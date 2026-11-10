"""Evaluate full-sib families and their gain screens.

Member sums precede parent laws; float32 screens fall back to exact evaluation within their error allowance.
Method: docs/methods/relatedness.md.
"""

from __future__ import annotations

import numba
import numpy as np

from ...core.kernels import kernel
from .pair_index import pair_indices
from .parameters import LOG_ZERO, SCALE_EXPONENT, SCREEN_FLOOR, SCREEN_SITE_TILE

_b1C = numba.types.Array(numba.boolean, 1, "C", readonly=False)
_float323C = numba.types.Array(numba.float32, 3, "C", readonly=False)
_float641C = numba.types.Array(numba.float64, 1, "C", readonly=False)
_float642C = numba.types.Array(numba.float64, 2, "C", readonly=False)
_float643C = numba.types.Array(numba.float64, 3, "C", readonly=False)
_int641C = numba.types.Array(numba.int64, 1, "C", readonly=False)
_int642C = numba.types.Array(numba.int64, 2, "C", readonly=False)

_SCREEN_FLOOR = SCREEN_FLOOR
_SCALE_UP = 2.0**SCALE_EXPONENT
_SCALE_DOWN = 2.0**-SCALE_EXPONENT
_LOG_SCALE = float(SCALE_EXPONENT) * np.log(2.0)


@kernel(overloads=((numba.float64,) * 5,), inline="always", cache=True, nogil=True)
def _family_value(member_sum, change, sign, parent_u, parent_v):
    """The single L13 genotype expression, without a per-site temporary allocation."""
    return (parent_u + parent_v) + (member_sum + sign * change)


@kernel(
    overloads=((_float641C, _float641C, numba.float64, _float641C, _float641C),),
    inline="always",
    cache=True,
    nogil=True,
)
def _family_site(S, D, sign, Au, Av):
    """L13: member sum first; retain the genotype, exponential and site reduction orders."""
    best = LOG_ZERO
    for gu in range(3):
        for gv in range(3):
            k = 3 * gu + gv
            value = _family_value(S[k], D[k], sign, Au[gu], Av[gv])
            if value > best:
                best = value
    acc = 0.0
    for gu in range(3):
        for gv in range(3):
            k = 3 * gu + gv
            acc += np.exp(_family_value(S[k], D[k], sign, Au[gu], Av[gv]) - best)
    return best + np.log(acc)


@kernel(
    overloads=(
        (
            _float642C,
            _float642C,
            numba.float64,
            _float642C,
            _float642C,
            numba.types.literal(0),
            numba.int64,
        ),
        (
            _float642C,
            _float642C,
            numba.float64,
            _float642C,
            _float642C,
            numba.int64,
            numba.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
def family_loglik(S, D, sign, Au, Av, lo, hi):
    """sum over sites [lo, hi) of log sum_{g_u, g_v} exp(Au[g_u] + Av[g_v] + (S + sign D)[3 g_u + g_v]): a full-sib
    family with parent laws Au, Av whose members' child terms sum to S (D added or removed without a copy)."""
    total = 0.0
    for i in range(lo, hi):
        total += _family_site(S[i], D[i], sign, Au[i], Av[i])
    return total


@kernel(
    overloads=(
        (
            _float642C,
            _float642C,
            _float642C,
            numba.types.literal(0),
            numba.int64,
        ),
        (
            _float642C,
            _float642C,
            _float642C,
            numba.int64,
            numba.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
def half_sib_loglik(SF, SG, logpi, lo, hi):
    """Two latent-parent families sharing ONE latent parent u (their other parents independent): per site
    log sum_{g_u} pi(g_u) [sum_{g_v} pi(g_v) exp SF[3 g_u + g_v]] [sum_{g_w} pi(g_w) exp SG[3 g_u + g_w]]."""
    total = 0.0
    a = np.empty(3)
    for i in range(lo, hi):
        for gu in range(3):
            mf = LOG_ZERO
            mg = LOG_ZERO
            for gv in range(3):
                vf = logpi[i, gv] + SF[i, 3 * gu + gv]
                vg = logpi[i, gv] + SG[i, 3 * gu + gv]
                if vf > mf:
                    mf = vf
                if vg > mg:
                    mg = vg
            af = 0.0
            ag = 0.0
            for gv in range(3):
                af += np.exp(logpi[i, gv] + SF[i, 3 * gu + gv] - mf)
                ag += np.exp(logpi[i, gv] + SG[i, 3 * gu + gv] - mg)
            a[gu] = logpi[i, gu] + mf + np.log(af) + mg + np.log(ag)
        m = max(a[0], max(a[1], a[2]))
        total += m + np.log(np.exp(a[0] - m) + np.exp(a[1] - m) + np.exp(a[2] - m))
    return total


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _int642C,
            _int641C,
            _float642C,
            _float643C,
            numba.int64,
            numba.int64,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def family_logliks(S, slots, families, logpi, logpost, lo, hi, out):
    """out[j] = family_loglik of family families[j] with its slot laws, over sites [lo, hi)."""
    for j in numba.prange(families.shape[0]):
        f = families[j]
        Au = logpi if slots[f, 0] < 0 else logpost[slots[f, 0]]
        Av = logpi if slots[f, 1] < 0 else logpost[slots[f, 1]]
        out[j] = family_loglik(S[f], S[f], 0.0, Au, Av, lo, hi)


@kernel(
    overloads=(
        (
            _float642C,
            _float642C,
            _float643C,
            _b1C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def slot_gains(S_f, Au, logpost, allowed, out):
    """out[p] = family_loglik(S_f, Au, logpost[p]) over all sites for each allowed sample p (NaN otherwise): the
    family with its free parent slot filled by sequenced sample p."""
    M = S_f.shape[0]
    for p in numba.prange(logpost.shape[0]):
        out[p] = family_loglik(S_f, S_f, 0.0, Au, logpost[p], 0, M) if allowed[p] else np.nan


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float643C,
            _int642C,
            _float641C,
            _float642C,
            _float643C,
            _int641C,
            _int641C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def join_entries(S, C, slots, ll, logpi, logpost, families, samples, out):
    """out[t] = family_loglik(S[f] + C[c], f's slot laws) - ll[f] over all sites, f = families[t], c = samples[t]
    (join_gains for one sample)."""
    M = logpi.shape[0]
    for t in numba.prange(families.shape[0]):
        f = families[t]
        Au = logpi if slots[f, 0] < 0 else logpost[slots[f, 0]]
        Av = logpi if slots[f, 1] < 0 else logpost[slots[f, 1]]
        out[t] = family_loglik(S[f], C[samples[t]], 1.0, Au, Av, 0, M) - ll[f]


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float643C,
            _int642C,
            _float641C,
            _float642C,
            _float643C,
            _int641C,
            _int641C,
            _b1C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def leave_entries(S, C, slots, ll, logpi, logpost, families, samples, single, out):
    """out[t] = the change of family f = families[t] when sample c = samples[t] leaves it: -ll[f] for a one-member
    family (single[t]), else family_loglik(S[f] - C[c], f's slot laws) - ll[f]."""
    M = logpi.shape[0]
    for t in numba.prange(families.shape[0]):
        f = families[t]
        if single[t]:
            out[t] = -ll[f]
        else:
            Au = logpi if slots[f, 0] < 0 else logpost[slots[f, 0]]
            Av = logpi if slots[f, 1] < 0 else logpost[slots[f, 1]]
            out[t] = family_loglik(S[f], C[samples[t]], -1.0, Au, Av, 0, M) - ll[f]


@kernel(
    overloads=(
        (
            _float642C,
            _float643C,
            _int641C,
            _float642C,
            _float643C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def member_parent_logliks(S_f, C, members, Au, logpost, out):
    """out[j] = family_loglik(S_f - C[p], Au, logpost[p]) over all sites, p = members[j]: the family without member p
    and with p in its free parent slot."""
    M = S_f.shape[0]
    for j in numba.prange(members.shape[0]):
        p = members[j]
        out[j] = family_loglik(S_f, C[p], -1.0, Au, logpost[p], 0, M)


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float641C,
            _int642C,
            _int642C,
            _float642C,
            _float643C,
            numba.float64,
            _float642C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def merge_gains(S, ll, rows, parents, logpi, logpost, tolerance, out):
    """pair_gains for the merge step: out[t, 0] the full-sib merge gain; out[t, 1] the half-sib link gain, evaluated
    only when both merged slots are latent and the full-sib gain exceeds ``tolerance`` (NaN otherwise: such a pair
    is no merge offer either way)."""
    M = logpi.shape[0]
    for t in numba.prange(rows.shape[0]):
        f, g = rows[t, 0], rows[t, 1]
        pu, pv = parents[t, 0], parents[t, 1]
        Au = logpi if pu < 0 else logpost[pu]
        Av = logpi if pv < 0 else logpost[pv]
        out[t, 0] = family_loglik(S[f], S[g], 1.0, Au, Av, 0, M) - ll[f] - ll[g]
        if pu < 0 and pv < 0 and out[t, 0] > tolerance:
            out[t, 1] = half_sib_loglik(S[f], S[g], logpi, 0, M) - ll[f] - ll[g]
        else:
            out[t, 1] = np.nan


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float641C,
            _int642C,
            _float642C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def half_sib_gains(S, ll, rows, logpi, out):
    """out[t] = half_sib_loglik(S[f], S[g]) - ll[f] - ll[g] over all sites, (f, g) = rows[t]."""
    M = logpi.shape[0]
    for t in numba.prange(rows.shape[0]):
        f, g = rows[t, 0], rows[t, 1]
        out[t] = half_sib_loglik(S[f], S[g], logpi, 0, M) - ll[f] - ll[g]


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float643C,
            _int642C,
            _int641C,
            _float641C,
            _int641C,
            _float642C,
            _float643C,
            numba.int64,
            numba.int64,
            _float642C,
            _float641C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def unit_terms(S, C, slots, families, base, family_of, logpi, logpost, lo, hi, joins, rest, alone):
    """One support unit (sites [lo, hi)): joins[j, c] = family_loglik(S[f] + C[c], f's laws) - base[j] for every
    live family f = families[j] and sample c; rest[c] = family_loglik(S[f] - C[c], f's laws) for c's own family
    f = families[family_of[c]] (0 for a one-member family, set by the caller); alone[c] = c alone with latent
    parents."""
    F = families.shape[0]
    N = C.shape[0]
    for t in numba.prange(F * N + 2 * N):
        if t < F * N:
            j = t // N
            c = t - j * N
            f = families[j]
            Au = logpi if slots[f, 0] < 0 else logpost[slots[f, 0]]
            Av = logpi if slots[f, 1] < 0 else logpost[slots[f, 1]]
            joins[j, c] = family_loglik(S[f], C[c], 1.0, Au, Av, lo, hi) - base[j]
        elif t < F * N + N:
            c = t - F * N
            f = families[family_of[c]]
            Au = logpi if slots[f, 0] < 0 else logpost[slots[f, 0]]
            Av = logpi if slots[f, 1] < 0 else logpost[slots[f, 1]]
            rest[c] = family_loglik(S[f], C[c], -1.0, Au, Av, lo, hi)
        else:
            c = t - F * N - N
            alone[c] = family_loglik(C[c], C[c], 0.0, logpi, logpi, lo, hi)


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _int642C,
            _int641C,
            _float642C,
            _float643C,
            numba.int64,
            numba.int64,
            _float642C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def latent_alternatives(S, slots, families, logpi, logpost, lo, hi, out):
    """out[j, k] = family_loglik of family f = families[j] over sites [lo, hi) with slot k latent and the other slot
    as it is (NaN for a latent slot k): the alternative a sequenced parent must beat."""
    for t in numba.prange(2 * families.shape[0]):
        j = t // 2
        k = t - 2 * j
        f = families[j]
        if slots[f, k] < 0:
            out[j, k] = np.nan
        else:
            Au = logpi if (k == 0 or slots[f, 0] < 0) else logpost[slots[f, 0]]
            Av = logpi if (k == 1 or slots[f, 1] < 0) else logpost[slots[f, 1]]
            out[j, k] = family_loglik(S[f], S[f], 0.0, Au, Av, lo, hi)


@kernel(
    overloads=(
        (
            _float643C,
            _float642C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def alone_logliks(C, logpi, out):
    """out[c] = family_loglik(C[c], C[c], 0.0, logpi, logpi) over all sites: sample c alone with latent parents."""
    M = C.shape[1]
    for c in numba.prange(C.shape[0]):
        out[c] = family_loglik(C[c], C[c], 0.0, logpi, logpi, 0, M)


@kernel(
    overloads=(
        (
            _float642C,
            _float642C,
            numba.float64,
            _float642C,
            _float642C,
            numba.int64,
            numba.int64,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def family_loglik_parallel(S, D, sign, Au, Av, lo, hi):
    """family_loglik with its sites' terms evaluated in parallel and summed in site order (the same value)."""
    n = hi - lo
    terms = np.empty(n)
    for t in numba.prange(n):
        i = lo + t
        terms[t] = _family_site(S[i], D[i], sign, Au[i], Av[i])
    total = 0.0
    for t in range(n):
        total += terms[t]
    return total


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _int641C,
            _float642C,
            _float642C,
            _float642C,
            _float323C,
            _float642C,
            _float323C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def screen_shapes(S, families, W2, logpi, m, A, y, Y):
    """For each family f in families (latent slots): m[f, i] = max_k (S[f][i, k] + W2[i, k]) and
    A[f, i, k] = exp(S[f][i, k] + W2[i, k] - m[f, i]) (W2 half the latent pair prior: the full-sib merge screen);
    y[f, i] = max_k S[f][i, k] and Y[f, i, gu] = sum_gv pi(gv) exp(S[f][i, 3 gu + gv] - y[f, i]) (the half-sib
    screen). A and Y in float32."""
    M = logpi.shape[0]
    F = families.shape[0]
    # Fetch each shared row once per site tile: typed-list access retains the array, which must stay outside
    # the scalar inner loops. Sites remain independent and each output cell uses its site-local terms.
    tiles = (M + SCREEN_SITE_TILE - 1) // SCREEN_SITE_TILE
    for t in numba.prange(F * tiles):
        j = t // tiles
        tile = t - j * tiles
        f = families[j]
        Sf = S[f]
        for i in range(tile * SCREEN_SITE_TILE, min(M, (tile + 1) * SCREEN_SITE_TILE)):
            best = LOG_ZERO
            top = LOG_ZERO
            for k in range(9):
                v = Sf[i, k] + W2[i, k]
                if v > best:
                    best = v
                if Sf[i, k] > top:
                    top = Sf[i, k]
            m[f, i] = best
            y[f, i] = top
            for k in range(9):
                A[f, i, k] = np.float32(np.exp(Sf[i, k] + W2[i, k] - best))
            for gu in range(3):
                acc = 0.0
                for gv in range(3):
                    acc += np.exp(logpi[i, gv] + Sf[i, 3 * gu + gv] - top)
                Y[f, i, gu] = np.float32(acc)


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float642C,
            _float642C,
            numba.int64,
            numba.int64,
            numba.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _merge_site(S, W, m, f, g, i):
    """The exact log-domain value of one site of the merged family, relative to m[f, i] + m[g, i]."""
    Sf, Sg = S[f], S[g]
    best = LOG_ZERO
    for k in range(9):
        v = W[i, k] + Sf[i, k] + Sg[i, k]
        if v > best:
            best = v
    acc = 0.0
    for k in range(9):
        acc += np.exp(W[i, k] + Sf[i, k] + Sg[i, k] - best)
    return best + np.log(acc) - m[f, i] - m[g, i]


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float642C,
            _float642C,
            _float323C,
            numba.int64,
            numba.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _merge_estimate(S, W, m, A, f, g):
    """sum_i log sum_k A[f, i, k] A[g, i, k], with log-domain sites where the linear sum is too small."""
    M = A.shape[1]
    P = 1.0
    E = 0
    X = 0.0
    for i in range(M):
        d = 0.0
        for k in range(9):
            d += np.float64(A[f, i, k]) * np.float64(A[g, i, k])
        if d < _SCREEN_FLOOR:
            X += _merge_site(S, W, m, f, g, i)
        else:
            P *= d
            if P < _SCALE_DOWN:
                P *= _SCALE_UP
                E -= 1
            elif P > _SCALE_UP:
                P *= _SCALE_DOWN
                E += 1
    return np.log(P) + E * _LOG_SCALE + X


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float642C,
            _float642C,
            _float323C,
            _float641C,
            numba.int64,
            numba.int64,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def screen_all(S, W, m, A, base, n, tile, out):
    """out in packed upper-triangle order (f < g < n): the screen's latent full-sib merge gain estimate,
    base[f] + base[g] + sum_i log sum_k A[f, i, k] A[g, i, k] (base[f] = sum_i m[f, i] - ll[f]). Tiles of tile
    x tile families, each tile's rows read once, sites in order."""
    M = A.shape[1]
    tiles = (n + tile - 1) // tile
    for t in numba.prange(tiles * tiles):
        a = t // tiles
        b = t - a * tiles
        if b < a:
            continue
        f0, f1 = a * tile, min(n, a * tile + tile)
        g0, g1 = b * tile, min(n, b * tile + tile)
        P = np.ones((tile, tile))
        E = np.zeros((tile, tile), dtype=np.int64)
        X = np.zeros((tile, tile))
        for i in range(M):
            for f in range(f0, f1):
                for g in range(max(g0, f + 1), g1):
                    d = 0.0
                    for k in range(9):
                        d += np.float64(A[f, i, k]) * np.float64(A[g, i, k])
                    x, y = f - f0, g - g0
                    if d < _SCREEN_FLOOR:
                        X[x, y] += _merge_site(S, W, m, f, g, i)
                    else:
                        P[x, y] *= d
                        if P[x, y] < _SCALE_DOWN:
                            P[x, y] *= _SCALE_UP
                            E[x, y] -= 1
                        elif P[x, y] > _SCALE_UP:
                            P[x, y] *= _SCALE_DOWN
                            E[x, y] += 1
        for f in range(f0, f1):
            for g in range(max(g0, f + 1), g1):
                x, y = f - f0, g - g0
                index = f * (2 * n - f - 1) // 2 + g - f - 1
                out[index] = base[f] + base[g] + np.log(P[x, y]) + E[x, y] * _LOG_SCALE + X[x, y]


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float642C,
            _float642C,
            _float323C,
            _float641C,
            numba.int64,
            _int641C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def screen_one(S, W, m, A, base, f, others, out):
    """out[j] = the screen's estimate of the latent full-sib merge gain of family f and family others[j]."""
    for j in numba.prange(others.shape[0]):
        g = others[j]
        out[j] = base[f] + base[g] + _merge_estimate(S, W, m, A, f, g)


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float642C,
            _float642C,
            _float642C,
            _float323C,
            _float641C,
            _int641C,
            _int641C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def half_screen(S, logpi, pi, y, Y, base, first, second, out):
    """out[t] = the screen's estimate of the half-sib link gain of families f = first[t], g = second[t]:
    base[f] + base[g] + sum_i log sum_gu pi(gu) Y[f, i, gu] Y[g, i, gu] (base[f] = sum_i y[f, i] - ll[f])."""
    M = Y.shape[1]
    for t in numba.prange(first.shape[0]):
        f, g = first[t], second[t]
        Sf, Sg = S[f], S[g]
        P = 1.0
        E = 0
        X = 0.0
        for i in range(M):
            d = (
                pi[i, 0] * np.float64(Y[f, i, 0]) * np.float64(Y[g, i, 0])
                + pi[i, 1] * np.float64(Y[f, i, 1]) * np.float64(Y[g, i, 1])
                + pi[i, 2] * np.float64(Y[f, i, 2]) * np.float64(Y[g, i, 2])
            )
            if d < _SCREEN_FLOOR:
                a0 = 0.0
                a1 = 0.0
                a2 = 0.0
                for gu in range(3):
                    mf = LOG_ZERO
                    mg = LOG_ZERO
                    for gv in range(3):
                        vf = logpi[i, gv] + Sf[i, 3 * gu + gv]
                        vg = logpi[i, gv] + Sg[i, 3 * gu + gv]
                        if vf > mf:
                            mf = vf
                        if vg > mg:
                            mg = vg
                    af = 0.0
                    ag = 0.0
                    for gv in range(3):
                        af += np.exp(logpi[i, gv] + Sf[i, 3 * gu + gv] - mf)
                        ag += np.exp(logpi[i, gv] + Sg[i, 3 * gu + gv] - mg)
                    value = logpi[i, gu] + mf + np.log(af) + mg + np.log(ag)
                    if gu == 0:
                        a0 = value
                    elif gu == 1:
                        a1 = value
                    else:
                        a2 = value
                top = max(a0, max(a1, a2))
                X += top + np.log(np.exp(a0 - top) + np.exp(a1 - top) + np.exp(a2 - top)) - y[f, i] - y[g, i]
            else:
                P *= d
                if P < _SCALE_DOWN:
                    P *= _SCALE_UP
                    E -= 1
                elif P > _SCALE_UP:
                    P *= _SCALE_DOWN
                    E += 1
        out[t] = base[f] + base[g] + np.log(P) + E * _LOG_SCALE + X


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C),
            _float641C,
            _int641C,
            _int641C,
            _float642C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def full_sib_gains(S, ll, first, second, logpi, out):
    """out[t] = the latent full-sib merge gain of families first[t] < second[t] (pair_gains' first column)."""
    M = logpi.shape[0]
    for t in numba.prange(first.shape[0]):
        f, g = first[t], second[t]
        out[t] = family_loglik(S[f], S[g], 1.0, logpi, logpi, 0, M) - ll[f] - ll[g]


@kernel(
    overloads=(
        (
            _float643C,
            _float323C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def posterior_screen(logpost, out):
    """out[p, i, g] = exp(logpost[p, i, g]) in float32: each sample's genotype posterior, for the slot screen."""
    N, M = logpost.shape[0], logpost.shape[1]
    for t in numba.prange(N * M):
        p = t // M
        i = t - p * M
        for g in range(3):
            out[p, i, g] = np.float32(np.exp(logpost[p, i, g]))


@kernel(
    overloads=(
        (
            _float642C,
            _float642C,
            _float323C,
            _float643C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def slot_screen(S_f, Au, posterior, logpost, out):
    """out[p] = the screen's estimate of slot_gains' family_loglik(S_f, S_f, 0, Au, logpost[p]) for every sample p:
    sum_i [r_i + log sum_gv posterior[p, i, gv] R[i, gv]], R[i, gv] = sum_gu exp(Au[i, gu] + S_f[i, 3 gu + gv] - r_i)
    (r_i the largest exponent)."""
    M = S_f.shape[0]
    N = posterior.shape[0]
    R = np.empty((M, 3))
    r = np.empty(M)
    for i in numba.prange(M):
        best = LOG_ZERO
        for gu in range(3):
            for gv in range(3):
                v = Au[i, gu] + S_f[i, 3 * gu + gv]
                if v > best:
                    best = v
        r[i] = best
        for gv in range(3):
            acc = 0.0
            for gu in range(3):
                acc += np.exp(Au[i, gu] + S_f[i, 3 * gu + gv] - best)
            R[i, gv] = acc
    rsum = 0.0
    for i in range(M):
        rsum += r[i]
    for p in numba.prange(N):
        P = 1.0
        E = 0
        X = 0.0
        for i in range(M):
            d = (
                np.float64(posterior[p, i, 0]) * R[i, 0]
                + np.float64(posterior[p, i, 1]) * R[i, 1]
                + np.float64(posterior[p, i, 2]) * R[i, 2]
            )
            if d < _SCREEN_FLOOR:
                best = LOG_ZERO
                for gu in range(3):
                    for gv in range(3):
                        v = Au[i, gu] + logpost[p, i, gv] + S_f[i, 3 * gu + gv]
                        if v > best:
                            best = v
                acc = 0.0
                for gu in range(3):
                    for gv in range(3):
                        acc += np.exp(Au[i, gu] + logpost[p, i, gv] + S_f[i, 3 * gu + gv] - best)
                X += best + np.log(acc) - r[i]
            else:
                P *= d
                if P < _SCALE_DOWN:
                    P *= _SCALE_UP
                    E -= 1
                elif P > _SCALE_UP:
                    P *= _SCALE_DOWN
                    E += 1
        out[p] = rsum + np.log(P) + E * _LOG_SCALE + X


@kernel(
    overloads=(
        (
            _float643C,
            _float642C,
            _float323C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def sample_screen(C, n, G):
    """n[c, i] = max_k C[c, i, k] and G[c, i, k] = exp(C[c, i, k] - n[c, i]) (float32): each sample's child terms in
    the linear domain, for the join screen."""
    N, M = C.shape[0], C.shape[1]
    for t in numba.prange(N * M):
        c = t // M
        i = t - c * M
        best = LOG_ZERO
        for k in range(9):
            if C[c, i, k] > best:
                best = C[c, i, k]
        n[c, i] = best
        for k in range(9):
            G[c, i, k] = np.float32(np.exp(C[c, i, k] - best))


@kernel(
    overloads=(
        (
            _float642C,
            _float642C,
            _float642C,
            _float641C,
            _float642C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def family_screen(S_f, Au, Av, h, H):
    """h[i] = max_k (Au[i, gu] + Av[i, gv] + S_f[i, k]) and H[i, k] = exp(Au + Av + S_f - h[i]) (k = 3 gu + gv): a
    family's terms with its parent laws in the linear domain, for the join screen."""
    M = S_f.shape[0]
    for i in numba.prange(M):
        best = LOG_ZERO
        for gu in range(3):
            for gv in range(3):
                v = Au[i, gu] + Av[i, gv] + S_f[i, 3 * gu + gv]
                if v > best:
                    best = v
        h[i] = best
        for gu in range(3):
            for gv in range(3):
                H[i, 3 * gu + gv] = np.exp(Au[i, gu] + Av[i, gv] + S_f[i, 3 * gu + gv] - best)


@kernel(
    overloads=(
        (
            _float642C,
            _float642C,
            _float642C,
            _float641C,
            _float642C,
            numba.float64,
            _float643C,
            _float642C,
            _float323C,
            _float641C,
            _float641C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def join_screen(S_f, Au, Av, h, H, base, C, n, G, nsum, out):
    """out[c] = the screen's estimate of join_entries' family_loglik(S_f + C[c], Au, Av) - ll_f for every sample c:
    base + nsum[c] + sum_i log sum_k H[i, k] G[c, i, k] (base = sum_i h[i] - ll_f, nsum[c] = sum_i n[c, i])."""
    M = S_f.shape[0]
    N = G.shape[0]
    for c in numba.prange(N):
        P = 1.0
        E = 0
        X = 0.0
        for i in range(M):
            d = 0.0
            for k in range(9):
                d += H[i, k] * np.float64(G[c, i, k])
            if d < _SCREEN_FLOOR:
                best = LOG_ZERO
                for gu in range(3):
                    for gv in range(3):
                        v = Au[i, gu] + Av[i, gv] + S_f[i, 3 * gu + gv] + C[c, i, 3 * gu + gv]
                        if v > best:
                            best = v
                acc = 0.0
                for gu in range(3):
                    for gv in range(3):
                        acc += np.exp(Au[i, gu] + Av[i, gv] + S_f[i, 3 * gu + gv] + C[c, i, 3 * gu + gv] - best)
                X += best + np.log(acc) - h[i] - n[c, i]
            else:
                P *= d
                if P < _SCALE_DOWN:
                    P *= _SCALE_UP
                    E -= 1
                elif P > _SCALE_UP:
                    P *= _SCALE_DOWN
                    E += 1
        out[c] = base + nsum[c] + np.log(P) + E * _LOG_SCALE + X


__all__ = [
    "family_loglik",
    "half_sib_loglik",
    "family_logliks",
    "slot_gains",
    "pair_indices",
    "join_entries",
    "leave_entries",
    "member_parent_logliks",
    "merge_gains",
    "half_sib_gains",
    "unit_terms",
    "latent_alternatives",
    "alone_logliks",
    "family_loglik_parallel",
    "screen_shapes",
    "screen_all",
    "screen_one",
    "half_screen",
    "full_sib_gains",
    "posterior_screen",
    "slot_screen",
    "sample_screen",
    "family_screen",
    "join_screen",
]
