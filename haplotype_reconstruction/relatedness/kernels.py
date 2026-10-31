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
    gl [N, L, 3]; unobserved cells carry the prior), clipped to [1e-4, 1 - 1e-4].

    Release 3.4: one parallel pass per site (allele_frequency_sites), a mathematically equivalent per-site form of the
    array expression it replaced (allele_frequency_reference), without [N, L, 3] temporaries. It agrees bitwise on the
    production replays checked (all contigs of the validation runs); other array shapes can differ in the last bit,
    as numpy's own reductions may associate differently."""
    out = np.empty(gl.shape[1])
    allele_frequency_sites(np.ascontiguousarray(gl, dtype=np.float64), np.ascontiguousarray(observed, dtype=np.bool_),
                           iterations, out)
    return out


@numba.njit(parallel=True, cache=True)
def allele_frequency_sites(gl, observed, iterations, out):
    """allele_frequency per site: out[l] after ``iterations`` EM steps from 0.5 (posterior sums over the samples in
    sample order; a cell's genotype weights are summed in genotype order)."""
    N, L = gl.shape[0], gl.shape[1]
    for l in numba.prange(L):
        p = 0.5
        for _ in range(iterations):
            q = 1.0 - p
            prior0 = q * q
            prior1 = 2.0 * p * q
            prior2 = p * p
            het = 0.0
            hom = 0.0
            for n in range(N):
                if observed[n, l]:
                    x0 = gl[n, l, 0] * prior0
                    x1 = gl[n, l, 1] * prior1
                    x2 = gl[n, l, 2] * prior2
                    total = max((x0 + x1) + x2, 1e-300)
                    het += x1 / total
                    hom += x2 / total
                else:
                    het += prior1
                    hom += prior2
            p = min(max(het / N / 2 + hom / N, 1e-4), 1 - 1e-4)
        out[l] = p


def allele_frequency_reference(gl, observed, iterations=6):
    """allele_frequency as an array expression (the implementation before release 3.4; validation only)."""
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
    M = logpi.shape[0]
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
def sample_gains(S_f, C, samples, sign, Au, Av, base, out):
    """out[j] = family_loglik(S_f + sign C[samples[j]], Au, Av) - base over all sites: each listed sample joining
    (sign 1) or leaving (sign -1) one family with parent laws Au, Av."""
    M = S_f.shape[0]
    for j in numba.prange(samples.shape[0]):
        out[j] = family_loglik(S_f, C[samples[j]], sign, Au, Av, 0, M) - base


@numba.njit(parallel=True, cache=True)
def slot_gains(S_f, Au, logpost, allowed, out):
    """out[p] = family_loglik(S_f, Au, logpost[p]) over all sites for each allowed sample p (NaN otherwise): the
    family with its free parent slot filled by sequenced sample p."""
    M = S_f.shape[0]
    for p in numba.prange(logpost.shape[0]):
        out[p] = family_loglik(S_f, S_f, 0.0, Au, logpost[p], 0, M) if allowed[p] else np.nan


@numba.njit(cache=True)
def pair_indices(n, lo, hi):
    """Unordered sample pairs [lo, hi) in row-major upper-triangle order, without triu temporaries."""
    out = np.empty((hi - lo, 2), dtype=np.int64)
    f, offset = 0, 0
    while f < n - 1 and offset + n - f - 1 <= lo:
        offset += n - f - 1
        f += 1
    g = f + 1 + lo - offset
    for t in range(hi - lo):
        out[t, 0], out[t, 1] = f, g
        g += 1
        if g == n:
            f += 1
            g = f + 1
    return out


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
    P = pairs.shape[0]
    # Release 3.4: contiguous blocks of pairs share their chain buffers (allocated once per block, not per pair)
    blocks = min(P, 16 * numba.get_num_threads())
    for block in numba.prange(blocks):
        em = np.empty((L, 3))
        alpha = np.empty((L, 3))
        scale = np.empty(L)
        for t in range(block * P // blocks, (block + 1) * P // blocks):
            _pair_chain(gl, observed, p, stay, pairs[t, 0], pairs[t, 1], masks, M, L, em, alpha, scale, posterior[t],
                        loglik[t])


@numba.njit(cache=True)
def _pair_chain(gl, observed, p, stay, i, j, masks, M, L, em, alpha, scale, posterior, loglik):
    """pair_ibd for one pair (i, j), in the buffers em, alpha [L, 3] and scale [L]; accumulates into its rows
    posterior [4] and loglik [M]. Release 3.4: the emissions, the full chain's forward pass and the masked chains run
    in one pass over the sites (each chain's arithmetic and order as before; independent chains overlap)."""
    a = np.zeros((M, 3))
    allowed = np.zeros(M, dtype=np.int64)
    for m in range(M):
        for k in range(3):
            if masks[m, k]:
                allowed[m] += 1
        for k in range(3):
            if masks[m, k]:
                a[m, k] = 1.0 / allowed[m]
    total = np.zeros(M)
    for l in range(L):
        if observed[i, l] and observed[j, l]:
            q = 1.0 - p[l]
            pp = p[l]
            h0, h1, h2 = q * q, 2.0 * pp * q, pp * pp
            a0, a1, a2 = gl[i, l, 0], gl[i, l, 1], gl[i, l, 2]
            b0, b1, b2 = gl[j, l, 0], gl[j, l, 1], gl[j, l, 2]
            e0 = (h0 * a0 + h1 * a1 + h2 * a2) * (h0 * b0 + h1 * b1 + h2 * b2)
            e1 = (q * q * q * a0 * b0 + pp * pp * pp * a2 * b2 + pp * q * a1 * b1
                  + pp * q * q * (a0 * b1 + a1 * b0) + pp * pp * q * (a2 * b1 + a1 * b2))
            e2 = h0 * a0 * b0 + h1 * a1 * b1 + h2 * a2 * b2
        else:
            e0 = 1.0
            e1 = 1.0
            e2 = 1.0
        em[l, 0] = e0
        em[l, 1] = e1
        em[l, 2] = e2
        # the full chain: forward pass
        if l == 0:
            s = 0.0
            for k in range(3):
                alpha[0, k] = em[0, k] / 3.0
                s += alpha[0, k]
            scale[0] = s
            for k in range(3):
                alpha[0, k] /= s
        else:
            move = (1.0 - stay[l]) / 2.0
            s = 0.0
            for k in range(3):
                alpha[l, k] = (alpha[l - 1, k] * stay[l] + (1.0 - alpha[l - 1, k]) * move) * em[l, k]
                s += alpha[l, k]
            scale[l] = s
            for k in range(3):
                alpha[l, k] /= s
        # the masked chains: forward log-likelihoods
        for m in range(M):
            if l > 0 and allowed[m] > 1:
                move = (1.0 - stay[l]) / (allowed[m] - 1)
                tot = a[m, 0] + a[m, 1] + a[m, 2]
                n0 = a[m, 0] * stay[l] + (tot - a[m, 0]) * move if masks[m, 0] else 0.0
                n1 = a[m, 1] * stay[l] + (tot - a[m, 1]) * move if masks[m, 1] else 0.0
                n2 = a[m, 2] * stay[l] + (tot - a[m, 2]) * move if masks[m, 2] else 0.0
                a[m, 0] = n0
                a[m, 1] = n1
                a[m, 2] = n2
            a[m, 0] *= e0
            a[m, 1] *= e1
            a[m, 2] *= e2
            s = a[m, 0] + a[m, 1] + a[m, 2]
            total[m] += np.log(s)
            a[m, 0] /= s
            a[m, 1] /= s
            a[m, 2] /= s
    for m in range(M):
        loglik[m] += total[m]
    # the full chain: backward pass and posteriors
    b0 = 1.0
    b1 = 1.0
    b2 = 1.0
    for l in range(L - 1, -1, -1):
        g0, g1, g2 = alpha[l, 0] * b0, alpha[l, 1] * b1, alpha[l, 2] * b2
        gs = g0 + g1 + g2
        if observed[i, l] and observed[j, l]:
            posterior[0] += g0 / gs
            posterior[1] += g1 / gs
            posterior[2] += g2 / gs
            posterior[3] += 1.0
        if l > 0:
            move = (1.0 - stay[l]) / 2.0
            e0, e1, e2 = em[l, 0] * b0, em[l, 1] * b1, em[l, 2] * b2
            tot = e0 + e1 + e2
            sc = scale[l]
            b0 = (e0 * stay[l] + (tot - e0) * move) / sc
            b1 = (e1 * stay[l] + (tot - e1) * move) / sc
            b2 = (e2 * stay[l] + (tot - e2) * move) / sc


# Release 3.4 (relatedness speed-up): kernels of the incremental family search (relatedness.families). Each value is
# the same family_loglik / half_sib_loglik expression, in the same order, as the kernels above compute it, so the
# search's numbers do not depend on which kernel computed them; they differ only in which entries they evaluate.


@numba.njit(parallel=True, cache=True)
def join_entries(S, C, slots, ll, logpi, logpost, families, samples, out):
    """out[t] = family_loglik(S[f] + C[c], f's slot laws) - ll[f] over all sites, f = families[t], c = samples[t]
    (join_gains for one sample)."""
    M = logpi.shape[0]
    for t in numba.prange(families.shape[0]):
        f = families[t]
        Au = logpi if slots[f, 0] < 0 else logpost[slots[f, 0]]
        Av = logpi if slots[f, 1] < 0 else logpost[slots[f, 1]]
        out[t] = family_loglik(S[f], C[samples[t]], 1.0, Au, Av, 0, M) - ll[f]


@numba.njit(parallel=True, cache=True)
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


@numba.njit(parallel=True, cache=True)
def member_parent_logliks(S_f, C, members, Au, logpost, out):
    """out[j] = family_loglik(S_f - C[p], Au, logpost[p]) over all sites, p = members[j]: the family without member p
    and with p in its free parent slot."""
    M = S_f.shape[0]
    for j in numba.prange(members.shape[0]):
        p = members[j]
        out[j] = family_loglik(S_f, C[p], -1.0, Au, logpost[p], 0, M)


@numba.njit(parallel=True, cache=True)
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


@numba.njit(parallel=True, cache=True)
def half_sib_gains(S, ll, rows, logpi, out):
    """out[t] = half_sib_loglik(S[f], S[g]) - ll[f] - ll[g] over all sites, (f, g) = rows[t]."""
    M = logpi.shape[0]
    for t in numba.prange(rows.shape[0]):
        f, g = rows[t, 0], rows[t, 1]
        out[t] = half_sib_loglik(S[f], S[g], logpi, 0, M) - ll[f] - ll[g]


@numba.njit(parallel=True, cache=True)
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


@numba.njit(parallel=True, cache=True)
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


@numba.njit(parallel=True, cache=True)
def alone_logliks(C, logpi, out):
    """out[c] = family_loglik(C[c], C[c], 0.0, logpi, logpi) over all sites: sample c alone with latent parents."""
    M = C.shape[1]
    for c in numba.prange(C.shape[0]):
        out[c] = family_loglik(C[c], C[c], 0.0, logpi, logpi, 0, M)


# The screens: estimates of a gain from each family's and sample's site terms in the linear domain (rescaled per site
# to their largest term, in float32), so that a site's sum of exponentials is a short dot product; a site whose sum is
# too small for float32 is evaluated in the log domain. The search sets aside a candidate only when its estimate is
# beyond the decision by more than the screen margin (relatedness.families), and evaluates the others exactly.
_SCREEN_FLOOR = 1e-30
_SCALE_UP = 2.0 ** 500
_SCALE_DOWN = 2.0 ** -500
_LOG_SCALE = 500.0 * np.log(2.0)


@numba.njit(parallel=True, cache=True)
def family_loglik_parallel(S, D, sign, Au, Av, lo, hi):
    """family_loglik with its sites' terms evaluated in parallel and summed in site order (the same value)."""
    n = hi - lo
    terms = np.empty(n)
    for t in numba.prange(n):
        i = lo + t
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
        terms[t] = best + np.log(acc)
    total = 0.0
    for t in range(n):
        total += terms[t]
    return total


@numba.njit(parallel=True, cache=True)
def screen_shapes(S, families, W2, logpi, m, A, y, Y):
    """For each family f in families (latent slots): m[f, i] = max_k (S[f][i, k] + W2[i, k]) and
    A[f, i, k] = exp(S[f][i, k] + W2[i, k] - m[f, i]) (W2 half the latent pair prior: the full-sib merge screen);
    y[f, i] = max_k S[f][i, k] and Y[f, i, gu] = sum_gv pi(gv) exp(S[f][i, 3 gu + gv] - y[f, i]) (the half-sib
    screen). A and Y in float32."""
    M = logpi.shape[0]
    F = families.shape[0]
    # Fetch each shared row once per site tile: typed-list access retains the array, which must stay outside
    # the scalar inner loops. Each site's arithmetic and every output cell are unchanged.
    tiles = (M + 255) // 256
    for t in numba.prange(F * tiles):
        j = t // tiles
        tile = t - j * tiles
        f = families[j]
        Sf = S[f]
        for i in range(tile * 256, min(M, (tile + 1) * 256)):
            best = -1e300
            top = -1e300
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


@numba.njit(cache=True)
def _merge_site(S, W, m, f, g, i):
    """The exact log-domain value of one site of the merged family, relative to m[f, i] + m[g, i]."""
    Sf, Sg = S[f], S[g]
    best = -1e300
    for k in range(9):
        v = W[i, k] + Sf[i, k] + Sg[i, k]
        if v > best:
            best = v
    acc = 0.0
    for k in range(9):
        acc += np.exp(W[i, k] + Sf[i, k] + Sg[i, k] - best)
    return best + np.log(acc) - m[f, i] - m[g, i]


@numba.njit(cache=True)
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


@numba.njit(parallel=True, cache=True)
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


@numba.njit(parallel=True, cache=True)
def screen_one(S, W, m, A, base, f, others, out):
    """out[j] = the screen's estimate of the latent full-sib merge gain of family f and family others[j]."""
    for j in numba.prange(others.shape[0]):
        g = others[j]
        out[j] = base[f] + base[g] + _merge_estimate(S, W, m, A, f, g)


@numba.njit(parallel=True, cache=True)
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
            d = (pi[i, 0] * np.float64(Y[f, i, 0]) * np.float64(Y[g, i, 0])
                 + pi[i, 1] * np.float64(Y[f, i, 1]) * np.float64(Y[g, i, 1])
                 + pi[i, 2] * np.float64(Y[f, i, 2]) * np.float64(Y[g, i, 2]))
            if d < _SCREEN_FLOOR:
                a0 = 0.0
                a1 = 0.0
                a2 = 0.0
                for gu in range(3):
                    mf = -1e300
                    mg = -1e300
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


@numba.njit(parallel=True, cache=True)
def full_sib_gains(S, ll, first, second, logpi, out):
    """out[t] = the latent full-sib merge gain of families first[t] < second[t] (pair_gains' first column)."""
    M = logpi.shape[0]
    for t in numba.prange(first.shape[0]):
        f, g = first[t], second[t]
        out[t] = family_loglik(S[f], S[g], 1.0, logpi, logpi, 0, M) - ll[f] - ll[g]


@numba.njit(parallel=True, cache=True)
def posterior_screen(logpost, out):
    """out[p, i, g] = exp(logpost[p, i, g]) in float32: each sample's genotype posterior, for the slot screen."""
    N, M = logpost.shape[0], logpost.shape[1]
    for t in numba.prange(N * M):
        p = t // M
        i = t - p * M
        for g in range(3):
            out[p, i, g] = np.float32(np.exp(logpost[p, i, g]))


@numba.njit(parallel=True, cache=True)
def slot_screen(S_f, Au, posterior, logpost, out):
    """out[p] = the screen's estimate of slot_gains' family_loglik(S_f, S_f, 0, Au, logpost[p]) for every sample p:
    sum_i [r_i + log sum_gv posterior[p, i, gv] R[i, gv]], R[i, gv] = sum_gu exp(Au[i, gu] + S_f[i, 3 gu + gv] - r_i)
    (r_i the largest exponent)."""
    M = S_f.shape[0]
    N = posterior.shape[0]
    R = np.empty((M, 3))
    r = np.empty(M)
    for i in numba.prange(M):
        best = -1e300
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
            d = (np.float64(posterior[p, i, 0]) * R[i, 0] + np.float64(posterior[p, i, 1]) * R[i, 1]
                 + np.float64(posterior[p, i, 2]) * R[i, 2])
            if d < _SCREEN_FLOOR:
                best = -1e300
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


@numba.njit(parallel=True, cache=True)
def sample_screen(C, n, G):
    """n[c, i] = max_k C[c, i, k] and G[c, i, k] = exp(C[c, i, k] - n[c, i]) (float32): each sample's child terms in
    the linear domain, for the join screen."""
    N, M = C.shape[0], C.shape[1]
    for t in numba.prange(N * M):
        c = t // M
        i = t - c * M
        best = -1e300
        for k in range(9):
            if C[c, i, k] > best:
                best = C[c, i, k]
        n[c, i] = best
        for k in range(9):
            G[c, i, k] = np.float32(np.exp(C[c, i, k] - best))


@numba.njit(parallel=True, cache=True)
def family_screen(S_f, Au, Av, h, H):
    """h[i] = max_k (Au[i, gu] + Av[i, gv] + S_f[i, k]) and H[i, k] = exp(Au + Av + S_f - h[i]) (k = 3 gu + gv): a
    family's terms with its parent laws in the linear domain, for the join screen."""
    M = S_f.shape[0]
    for i in numba.prange(M):
        best = -1e300
        for gu in range(3):
            for gv in range(3):
                v = Au[i, gu] + Av[i, gv] + S_f[i, 3 * gu + gv]
                if v > best:
                    best = v
        h[i] = best
        for gu in range(3):
            for gv in range(3):
                H[i, 3 * gu + gv] = np.exp(Au[i, gu] + Av[i, gv] + S_f[i, 3 * gu + gv] - best)


@numba.njit(parallel=True, cache=True)
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
                best = -1e300
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
