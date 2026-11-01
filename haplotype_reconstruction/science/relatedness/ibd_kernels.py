"""Evaluate independent pairwise three-state IBD chains.

Owns the relocated ibd_kernels units. No storage, environment or runtime policy.
Method: docs/methods/relatedness.md; WP_RL1. Legacy arithmetic retained for relocation.
"""

from __future__ import annotations

import numba
import numpy as np

from ...core.kernels import kernel

_b2C = numba.types.Array(numba.boolean, 2, "C", readonly=False)
_float641A = numba.types.Array(numba.float64, 1, "A", readonly=False)
_float641C = numba.types.Array(numba.float64, 1, "C", readonly=False)
_float642C = numba.types.Array(numba.float64, 2, "C", readonly=False)
_float642F = numba.types.Array(numba.float64, 2, "F", readonly=False)
_float643C = numba.types.Array(numba.float64, 3, "C", readonly=False)
_int642C = numba.types.Array(numba.int64, 2, "C", readonly=False)

IBD_MASKS = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 0]], dtype=np.bool_)


@kernel(
    overloads=(
        (
            numba.int64,
            numba.int64,
            numba.int64,
        ),
    ),
    cache=True,
)
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


@kernel(
    overloads=(
        (
            _float643C,
            _b2C,
            _float641C,
            _float641C,
            _int642C,
            _b2C,
            _float642C,
            _float642C,
        ),
        (
            _float643C,
            _b2C,
            _float641C,
            _float641C,
            _int642C,
            _b2C,
            _float642F,
            _float642C,
        ),
    ),
    parallel=True,
    cache=False,
)
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
            _pair_chain(
                gl, observed, p, stay, pairs[t, 0], pairs[t, 1], masks, M, L, em, alpha, scale, posterior[t], loglik[t]
            )


@kernel(
    overloads=(
        (
            _float643C,
            _b2C,
            _float641C,
            _float641C,
            numba.int64,
            numba.int64,
            _b2C,
            numba.int64,
            numba.int64,
            _float642C,
            _float642C,
            _float641C,
            _float641A,
            _float641C,
        ),
        (
            _float643C,
            _b2C,
            _float641C,
            _float641C,
            numba.int64,
            numba.int64,
            _b2C,
            numba.int64,
            numba.int64,
            _float642C,
            _float642C,
            _float641C,
            _float641C,
            _float641C,
        ),
    ),
    cache=True,
)
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
            e1 = (
                q * q * q * a0 * b0
                + pp * pp * pp * a2 * b2
                + pp * q * a1 * b1
                + pp * q * q * (a0 * b1 + a1 * b0)
                + pp * pp * q * (a2 * b1 + a1 * b2)
            )
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


__all__ = ["pair_indices", "pair_ibd"]
