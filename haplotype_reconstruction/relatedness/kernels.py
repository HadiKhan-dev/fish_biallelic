"""Kernels of the relatedness stage: the full-sib family likelihood and the pairwise IBD chain.

Genotypes are coded 0, 1, 2 (copies of the alternative allele). A family's two parents have genotype pair index
k = 3 g_u + g_v; ``TRANSMISSION[k, g]`` is the Mendelian probability of a child genotype g. A parent's genotype law
(log, per site) is the Hardy-Weinberg prior for a latent parent, or a sequenced parent's posterior from its own reads.
Site arrays are concatenated over chromosomes; ``lo, hi`` select a contiguous range (one chromosome, or all).
"""
from __future__ import annotations

import numba
import numpy as np

TRANSMISSION = np.zeros((9, 3))
for _gu in range(3):
    for _gv in range(3):
        _a, _b = _gu / 2.0, _gv / 2.0
        TRANSMISSION[3 * _gu + _gv] = [(1 - _a) * (1 - _b), _a * (1 - _b) + (1 - _a) * _b, _a * _b]

# IBD chain state masks: all states; no IBD0; no IBD2. log L(all) - log L(no IBD0) is the evidence for IBD0
# stretches (a parent and child have none); log L(all) - log L(no IBD2) the evidence for IBD2 stretches.
IBD_MASKS = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 0]], dtype=np.bool_)


def allele_frequency(gl, observed, iterations=6):
    """float64 [L]: the alternative allele frequency under Hardy-Weinberg (EM over the samples' genotype likelihoods
    gl [N, L, 3]; unobserved cells carry the prior), clipped to [1e-4, 1 - 1e-4]."""
    p = np.full(gl.shape[1], 0.5)
    for _ in range(iterations):
        prior = np.stack([(1 - p) ** 2, 2 * p * (1 - p), p ** 2], axis=1)[None]
        post = gl * prior
        post /= np.maximum(post.sum(axis=2, keepdims=True), 1e-300)
        post = np.where(observed[..., None], post, prior)
        p = np.clip(post[..., 1].mean(0) / 2 + post[..., 2].mean(0), 1e-4, 1 - 1e-4)
    return p


def family_terms(gl, observed, p):
    """(child terms float64 [N, M, 9], log prior float64 [M, 3], log posterior float64 [N, M, 3]) at the given
    sites (gl [N, M, 3], observed [N, M], p [M]).

    Child terms: a sample's log probability of its reads given each parent genotype pair (genotype likelihoods
    scaled to a maximum of 1 per cell; unobserved cells contribute 0). Log prior: Hardy-Weinberg genotype
    probabilities (a latent parent's law). Log posterior: each sample's genotype posterior from its own reads under
    that prior (its law as a sequenced parent; a family conditions on it, so its reads are counted once, in its
    own family)."""
    g = gl / np.maximum(gl.max(axis=2, keepdims=True), 1e-300)
    g = np.where(observed[..., None], g, 1.0)
    terms = np.log(np.maximum(np.einsum("nmg,kg->nmk", g, TRANSMISSION), 1e-300))
    prior = np.stack([(1 - p) ** 2, 2 * p * (1 - p), p ** 2], axis=1)
    post = g * prior[None]
    post /= post.sum(axis=2, keepdims=True)
    return (np.ascontiguousarray(terms), np.ascontiguousarray(np.log(prior)),
            np.ascontiguousarray(np.log(np.maximum(post, 1e-300))))


@numba.njit(cache=True)
def family_loglik(S, D, sign, Au, Av, lo, hi):
    """sum over sites [lo, hi) of log sum_{g_u, g_v} exp(Au[g_u] + Av[g_v] + (S + sign D)[3 g_u + g_v]): a full-sib
    family with parent laws Au, Av whose members' child terms sum to S (D added or removed without a copy)."""
    total = 0.0
    for i in range(lo, hi):
        best = -1e300
        for gu in range(3):
            for gv in range(3):
                v = Au[i, gu] + Av[i, gv] + S[i, 3 * gu + gv] + sign * D[i, 3 * gu + gv]
                if v > best:
                    best = v
        acc = 0.0
        for gu in range(3):
            for gv in range(3):
                acc += np.exp(Au[i, gu] + Av[i, gv] + S[i, 3 * gu + gv] + sign * D[i, 3 * gu + gv] - best)
        total += best + np.log(acc)
    return total


@numba.njit(cache=True)
def half_sib_loglik(SF, SG, logpi, lo, hi):
    """Two latent-parent families sharing ONE latent parent u (their other parents independent): per site
    log sum_{g_u} pi(g_u) [sum_{g_v} pi(g_v) exp SF[3 g_u + g_v]] [sum_{g_w} pi(g_w) exp SG[3 g_u + g_w]]."""
    total = 0.0
    a = np.empty(3)
    for i in range(lo, hi):
        for gu in range(3):
            mf = -1e300
            mg = -1e300
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


@numba.njit(parallel=True, cache=True)
def family_logliks(S, slots, families, logpi, logpost, lo, hi, out):
    """out[j] = family_loglik of family families[j] with its slot laws, over sites [lo, hi)."""
    for j in numba.prange(families.shape[0]):
        f = families[j]
        Au = logpi if slots[f, 0] < 0 else logpost[slots[f, 0]]
        Av = logpi if slots[f, 1] < 0 else logpost[slots[f, 1]]
        out[j] = family_loglik(S[f], S[f], 0.0, Au, Av, lo, hi)


@numba.njit(parallel=True, cache=True)
def pair_gains(S, ll, rows, parents, logpi, logpost, out):
    """For each row t (families f, g = rows[t]; merged slot parents parents[t], -1 latent): out[t, 0] = the
    log-likelihood gain of merging f and g into one full-sib family; out[t, 1] = the gain of a half-sib link instead
    (both families latent and sharing one latent parent; NaN unless both merged slots are latent)."""
    M = S.shape[1]
    for t in numba.prange(rows.shape[0]):
        f, g = rows[t, 0], rows[t, 1]
        pu, pv = parents[t, 0], parents[t, 1]
        Au = logpi if pu < 0 else logpost[pu]
        Av = logpi if pv < 0 else logpost[pv]
        out[t, 0] = family_loglik(S[f], S[g], 1.0, Au, Av, 0, M) - ll[f] - ll[g]
        if pu < 0 and pv < 0:
            out[t, 1] = half_sib_loglik(S[f], S[g], logpi, 0, M) - ll[f] - ll[g]
        else:
            out[t, 1] = np.nan


@numba.njit(parallel=True, cache=True)
def join_gains(S, base, slots, families, logpi, logpost, D, sign, lo, hi, out):
    """out[j] = family_loglik(S[f] + sign D, f's slot laws) - base[j] over sites [lo, hi), f = families[j]."""
    for j in numba.prange(families.shape[0]):
        f = families[j]
        Au = logpi if slots[f, 0] < 0 else logpost[slots[f, 0]]
        Av = logpi if slots[f, 1] < 0 else logpost[slots[f, 1]]
        out[j] = family_loglik(S[f], D, sign, Au, Av, lo, hi) - base[j]


@numba.njit(parallel=True, cache=True)
def slot_gains(S_f, Au, logpost, allowed, out):
    """out[p] = family_loglik(S_f, Au, logpost[p]) over all sites for each allowed sample p (NaN otherwise): the
    family with its free parent slot filled by sequenced sample p."""
    M = S_f.shape[0]
    for p in numba.prange(logpost.shape[0]):
        out[p] = family_loglik(S_f, S_f, 0.0, Au, logpost[p], 0, M) if allowed[p] else np.nan


@numba.njit(parallel=True, cache=True)
def pair_ibd(gl, observed, p, stay, pairs, masks, posterior, loglik):
    """Pairwise 3-state IBD chain (IBD0, IBD1, IBD2; Jacquard emissions at allele frequency p under Hardy-Weinberg;
    sites not observed in both samples emit 1) for each pair (pairs[t, 0], pairs[t, 1]) over one chromosome's sites.
    stay[l]: the probability that the chain keeps its state from site l - 1 to l (exp(-rate x Morgans)); a leaving
    chain moves to the other allowed states equally. Accumulates posterior[t, 0:3] += the posterior state sums over
    jointly observed sites, posterior[t, 3] += their count, and loglik[t, m] += the forward log-likelihood with only
    the states allowed by masks[m]. Genotype likelihoods need no normalisation: every state of a pair uses the
    same reads."""
    L = gl.shape[1]
    M = masks.shape[0]
    for t in numba.prange(pairs.shape[0]):
        i, j = pairs[t, 0], pairs[t, 1]
        em = np.empty((L, 3))
        for l in range(L):
            if observed[i, l] and observed[j, l]:
                q = 1.0 - p[l]
                pp = p[l]
                h0, h1, h2 = q * q, 2.0 * pp * q, pp * pp
                a0, a1, a2 = gl[i, l, 0], gl[i, l, 1], gl[i, l, 2]
                b0, b1, b2 = gl[j, l, 0], gl[j, l, 1], gl[j, l, 2]
                em[l, 0] = (h0 * a0 + h1 * a1 + h2 * a2) * (h0 * b0 + h1 * b1 + h2 * b2)
                em[l, 1] = (q * q * q * a0 * b0 + pp * pp * pp * a2 * b2 + pp * q * a1 * b1
                            + pp * q * q * (a0 * b1 + a1 * b0) + pp * pp * q * (a2 * b1 + a1 * b2))
                em[l, 2] = h0 * a0 * b0 + h1 * a1 * b1 + h2 * a2 * b2
            else:
                em[l, 0] = 1.0
                em[l, 1] = 1.0
                em[l, 2] = 1.0
        # the full chain: forward-backward posteriors
        alpha = np.empty((L, 3))
        scale = np.empty(L)
        s = 0.0
        for k in range(3):
            alpha[0, k] = em[0, k] / 3.0
            s += alpha[0, k]
        scale[0] = s
        for k in range(3):
            alpha[0, k] /= s
        for l in range(1, L):
            move = (1.0 - stay[l]) / 2.0
            s = 0.0
            for k in range(3):
                alpha[l, k] = (alpha[l - 1, k] * stay[l] + (1.0 - alpha[l - 1, k]) * move) * em[l, k]
                s += alpha[l, k]
            scale[l] = s
            for k in range(3):
                alpha[l, k] /= s
        b0 = 1.0
        b1 = 1.0
        b2 = 1.0
        for l in range(L - 1, -1, -1):
            g0, g1, g2 = alpha[l, 0] * b0, alpha[l, 1] * b1, alpha[l, 2] * b2
            gs = g0 + g1 + g2
            if observed[i, l] and observed[j, l]:
                posterior[t, 0] += g0 / gs
                posterior[t, 1] += g1 / gs
                posterior[t, 2] += g2 / gs
                posterior[t, 3] += 1.0
            if l > 0:
                move = (1.0 - stay[l]) / 2.0
                e0, e1, e2 = em[l, 0] * b0, em[l, 1] * b1, em[l, 2] * b2
                tot = e0 + e1 + e2
                sc = scale[l]
                b0 = (e0 * stay[l] + (tot - e0) * move) / sc
                b1 = (e1 * stay[l] + (tot - e1) * move) / sc
                b2 = (e2 * stay[l] + (tot - e2) * move) / sc
        # the masked chains: forward log-likelihoods
        for m in range(M):
            allowed = 0
            for k in range(3):
                if masks[m, k]:
                    allowed += 1
            a = np.zeros(3)
            for k in range(3):
                if masks[m, k]:
                    a[k] = 1.0 / allowed
            total = 0.0
            for l in range(L):
                if l > 0 and allowed > 1:
                    move = (1.0 - stay[l]) / (allowed - 1)
                    tot = a[0] + a[1] + a[2]
                    n0 = a[0] * stay[l] + (tot - a[0]) * move if masks[m, 0] else 0.0
                    n1 = a[1] * stay[l] + (tot - a[1]) * move if masks[m, 1] else 0.0
                    n2 = a[2] * stay[l] + (tot - a[2]) * move if masks[m, 2] else 0.0
                    a[0] = n0
                    a[1] = n1
                    a[2] = n2
                a[0] *= em[l, 0]
                a[1] *= em[l, 1]
                a[2] *= em[l, 2]
                s = a[0] + a[1] + a[2]
                total += np.log(s)
                a[0] /= s
                a[1] /= s
                a[2] /= s
            loglik[t, m] += total
