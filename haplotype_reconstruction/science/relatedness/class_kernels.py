"""Evaluate full-sib families and half-sib links with ancestry classes on the latent parent slots (rl21).

With two source populations a parent is described at a site by one of G configurations (its local ancestry state
with the alleles on its homologues, ancestry.CONFIGURATIONS), a sample's child terms run over the G x G parental
configuration pairs, and a latent parent's law over its configurations depends on its ancestry class, which is
constant over the family's sites. A family's term is therefore the marginal over the class pair of its latent slots:

    log sum_{c_u, c_v} w(c_u) w(c_v) prod_i sum_{h_u, h_v} pi_{c_u, i}(h_u) pi_{c_v, i}(h_v) exp S_i[G h_u + h_v]

(a sequenced slot is one fixed law, the sample's configuration posterior). The kernels return the per-class-pair
log-likelihoods. Per site the G x G exponentials are taken once and every class pair is a linear-domain product,
rescaled by powers of two; a site term below the floor is added in the log domain. These kernels run only with two
source populations; one source runs family_kernels unchanged.
Method: docs/methods/relatedness.md#rl21.
"""

from __future__ import annotations

import numba
import numpy as np

from ...core.kernels import kernel
from .parameters import LOG_ZERO, SCALE_EXPONENT, SCREEN_FLOOR, TERM_FLOOR

_float642C = numba.types.Array(numba.float64, 2, "C", readonly=False)
_float643C = numba.types.Array(numba.float64, 3, "C", readonly=False)
_float644C = numba.types.Array(numba.float64, 4, "C", readonly=False)
_float641C = numba.types.Array(numba.float64, 1, "C", readonly=False)
_int641C = numba.types.Array(numba.int64, 1, "C", readonly=False)

_FLOOR = SCREEN_FLOOR
_TINY = TERM_FLOOR
_SCALE_UP = 2.0**SCALE_EXPONENT
_SCALE_DOWN = 2.0**-SCALE_EXPONENT
_LOG_SCALE = float(SCALE_EXPONENT) * np.log(2.0)

# What a task evaluates: the sums of family ``family[t]``, those sums plus or minus sample ``other[t]``'s child terms,
# those sums plus family ``other[t]``'s sums, or sample ``family[t]``'s child terms alone.
FAMILY = 0
FAMILY_SAMPLE = 1
FAMILY_FAMILY = 2
SAMPLE = 3


@kernel(
    overloads=(
        (
            _float642C, _float642C, numba.float64, numba.int64, numba.int64, _float643C, _float643C, numba.int64,
            numba.int64, _float642C,
        ),
    ),
    cache=True,
    nogil=True,
)
def class_family_table(base, change, sign, u, v, laws, slot_laws, lo, hi, table):
    """table[a, b] = sum over sites [lo, hi) of log sum_{h_u, h_v} P_a(h_u) Q_b(h_v) exp((base + sign change)[G h_u +
    h_v]) for every law a of slot u and b of slot v: the classes' laws ``laws`` [classes, M, G] (probabilities) for a
    latent slot (u or v < 0), else the one law exp(slot_laws[u]) (log probabilities [N, M, G]). table is [n_u, n_v]
    with n = classes or 1. Sites in order; each class pair's product is rescaled by powers of two."""
    nu, nv = table.shape[0], table.shape[1]
    G = laws.shape[2]
    P = np.ones((nu, nv))
    E = np.zeros((nu, nv), dtype=np.int64)
    X = np.zeros((nu, nv))
    e = np.empty(G * G)
    pu = np.empty((nu, G))
    pv = np.empty(G)
    R = np.empty((nv, G))
    tops = 0.0
    for i in range(lo, hi):
        top = LOG_ZERO
        for k in range(G * G):
            e[k] = base[i, k] + sign * change[i, k]
            if e[k] > top:
                top = e[k]
        tops += top
        for k in range(G * G):
            e[k] = np.exp(e[k] - top)
        if u < 0:
            for a in range(nu):
                for g in range(G):
                    pu[a, g] = laws[a, i, g]
        else:
            for g in range(G):
                pu[0, g] = np.exp(slot_laws[u, i, g])
        for b in range(nv):
            if v < 0:
                for h in range(G):
                    pv[h] = laws[b, i, h]
            else:
                for h in range(G):
                    pv[h] = np.exp(slot_laws[v, i, h])
            for g in range(G):
                acc = 0.0
                for h in range(G):
                    acc += pv[h] * e[G * g + h]
                R[b, g] = acc
        for a in range(nu):
            for b in range(nv):
                d = 0.0
                for g in range(G):
                    d += pu[a, g] * R[b, g]
                if d < _FLOOR:
                    X[a, b] += np.log(max(d, _TINY))
                else:
                    P[a, b] *= d
                    if P[a, b] < _SCALE_DOWN:
                        P[a, b] *= _SCALE_UP
                        E[a, b] -= 1
    for a in range(nu):
        for b in range(nv):
            table[a, b] = tops + np.log(P[a, b]) + E[a, b] * _LOG_SCALE + X[a, b]


@kernel(
    overloads=(
        (
            numba.types.ListType(_float642C), _float643C, _int641C, _int641C, _int641C, _float641C, _int641C,
            _int641C, _float643C, _float643C, numba.int64, numba.int64, _float643C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def class_family_tasks(S, C, kind, family, other, sign, slot_u, slot_v, laws, slot_laws, lo, hi, out):
    """out[t, a, b] = class_family_table of task t over sites [lo, hi): member sums S[family[t]] (kind FAMILY),
    S[family[t]] + sign[t] C[other[t]] (FAMILY_SAMPLE), S[family[t]] + S[other[t]] (FAMILY_FAMILY) or C[family[t]]
    alone (SAMPLE), with slots slot_u[t], slot_v[t]: < 0 latent (one entry per class), else the one law
    slot_laws[slot] (entry 0; the other entries are LOG_ZERO). Tasks are independent; each is one thread's work in
    site order."""
    classes = laws.shape[0]
    for t in numba.prange(kind.shape[0]):
        k = kind[t]
        base = C[family[t]] if k == SAMPLE else S[family[t]]
        if k == FAMILY_SAMPLE:
            change = C[other[t]]
            s = sign[t]
        elif k == FAMILY_FAMILY:
            change = S[other[t]]
            s = 1.0
        else:
            change = base
            s = 0.0
        u, v = slot_u[t], slot_v[t]
        nu = classes if u < 0 else 1
        nv = classes if v < 0 else 1
        table = np.empty((nu, nv))
        class_family_table(base, change, s, u, v, laws, slot_laws, lo, hi, table)
        for a in range(classes):
            for b in range(classes):
                out[t, a, b] = table[a, b] if (a < nu and b < nv) else LOG_ZERO


@kernel(
    overloads=((numba.types.ListType(_float642C), _int641C, _int641C, _float643C, numba.int64, numba.int64, _float644C),),
    parallel=True,
    cache=True,
    nogil=True,
)
def class_half_sib_tasks(S, first, second, laws, lo, hi, out):
    """out[t, a, b, c] = the log-likelihood over sites [lo, hi) of latent families f = first[t], g = second[t]
    sharing ONE latent parent of class a, their other parents of classes b (f's) and c (g's): sum_i log
    sum_{h_u} pi_a(h_u) [sum_{h_v} pi_b(h_v) exp S_f[G h_u + h_v]] [sum_{h_w} pi_c(h_w) exp S_g[G h_u + h_w]]."""
    n = laws.shape[0]
    G = laws.shape[2]
    for t in numba.prange(first.shape[0]):
        Sf, Sg = S[first[t]], S[second[t]]
        P = np.ones((n, n, n))
        E = np.zeros((n, n, n), dtype=np.int64)
        X = np.zeros((n, n, n))
        ef = np.empty(G * G)
        eg = np.empty(G * G)
        Rf = np.empty((n, G))
        Rg = np.empty((n, G))
        tops = 0.0
        for i in range(lo, hi):
            tf = LOG_ZERO
            tg = LOG_ZERO
            for k in range(G * G):
                if Sf[i, k] > tf:
                    tf = Sf[i, k]
                if Sg[i, k] > tg:
                    tg = Sg[i, k]
            tops += tf + tg
            for k in range(G * G):
                ef[k] = np.exp(Sf[i, k] - tf)
                eg[k] = np.exp(Sg[i, k] - tg)
            for b in range(n):
                for g in range(G):
                    af = 0.0
                    ag = 0.0
                    for h in range(G):
                        af += laws[b, i, h] * ef[G * g + h]
                        ag += laws[b, i, h] * eg[G * g + h]
                    Rf[b, g] = af
                    Rg[b, g] = ag
            for a in range(n):
                for b in range(n):
                    for c in range(n):
                        d = 0.0
                        for g in range(G):
                            d += laws[a, i, g] * Rf[b, g] * Rg[c, g]
                        if d < _FLOOR:
                            X[a, b, c] += np.log(max(d, _TINY))
                        else:
                            P[a, b, c] *= d
                            if P[a, b, c] < _SCALE_DOWN:
                                P[a, b, c] *= _SCALE_UP
                                E[a, b, c] -= 1
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    out[t, a, b, c] = tops + np.log(P[a, b, c]) + E[a, b, c] * _LOG_SCALE + X[a, b, c]


__all__ = [
    "FAMILY",
    "FAMILY_SAMPLE",
    "FAMILY_FAMILY",
    "SAMPLE",
    "class_family_table",
    "class_family_tasks",
    "class_half_sib_tasks",
]
