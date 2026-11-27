"""Set continuation across emission-tied child transmission passages (FP20)."""

import math

import numpy as np
from numba import types

from ...core.kernels import kernel


@kernel(
    overloads=(
        (
            types.int64[:, :, ::1],
            types.int8[:, :, ::1],
            types.int64[::1],
            types.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _tied_passages(labels, sources, parents, child):
    """Consecutive distinct, fully copied anchors around at least one equal-label bin."""
    bins = labels.shape[2]
    pairs = np.empty((bins // 2, 2), dtype=np.int64)
    count = 0
    left = -1
    equal_between = False
    for b in range(bins):
        if labels[child, 0, b] == labels[child, 1, b]:
            equal_between = True
        elif sources[child, 0, b] < 2 and sources[child, 1, b] < 2:
            if left >= 0 and equal_between:
                for anchor in (left, b):
                    other0 = labels[parents[0], 1 - sources[child, 0, anchor], anchor]
                    other1 = labels[parents[1], 1 - sources[child, 1, anchor], anchor]
                    if other0 == labels[child, 1, anchor] and other1 == labels[child, 0, anchor]:
                        pairs[count, 0], pairs[count, 1] = left, b
                        count += 1
                        break
            left = b
            equal_between = False
    return pairs[:count]


@kernel(
    overloads=tuple(
        (
            types.int64[:, :, ::1],
            types.int64[::1],
            types.Array(types.int8, 2, allele_layout),
            types.float32[:, ::1],
            types.float64[:, ::1],
            types.float64,
            types.float64[:, ::1],
            types.int64[:, ::1],
            types.float64[:, ::1],
        )
        for allele_layout in ("C", "F")
    ),
    cache=True,
    nogil=True,
)
def _bin_evidence(labels, bins, alleles, gl, cohort, eps, flips, own_state, own_flip):
    """Sum own-site log likelihoods for the four source pairs."""
    em = np.zeros((len(labels), 4), np.float64)
    p = np.empty((2, 2))
    for i, b in enumerate(bins):
        valid = True
        for k in range(2):
            for h in range(2):
                z = labels[b, k, h]
                a = alleles[z, i]
                if a < 0:
                    valid = False
                f = flips[z, i]
                for t in range(2):
                    if own_state[t, i] == z:
                        f = own_flip[t, i]
                p[k, h] = 1 - f if a == 1 else f
        if not valid:
            continue
        g0, g1, g2 = float(gl[i, 0]), float(gl[i, 1]), float(gl[i, 2])
        background = cohort[i, 0] * g0 + cohort[i, 1] * g1 + cohort[i, 2] * g2
        for h in range(4):
            u, v = p[0, h // 2], p[1, h % 2]
            # Symmetric arithmetic for exact exchangeability of the two copies.
            mass = (1 - u) * (1 - v) * g0 + ((1 - u) * v + u * (1 - v)) * g1 + u * v * g2
            em[b, h] += math.log((1 - eps) * mass + eps * background)
    return em


@kernel(
    overloads=(
        (
            types.float64[:, ::1],
            types.float64[:, ::1],
            types.int64[:, :, ::1],
            types.int64[:, ::1],
            types.int64[:, ::1],
        ),
    ),
    cache=True,
    nogil=True,
)
def _continuation_odds(em, switch, labels, focal, pairs):
    """Log odds for matching versus differing endpoint reversals, using both flanks."""
    B = len(em)
    trans = np.empty((B - 1, 4, 4))
    for b in range(B - 1):
        for a in range(4):
            for h in range(4):
                s0, s1 = switch[0, b], switch[1, b]
                trans[b, a, h] = math.log(s0 if a // 2 != h // 2 else 1 - s0) + math.log(
                    s1 if a % 2 != h % 2 else 1 - s1
                )
    alpha = np.empty_like(em)
    beta = np.empty_like(em)
    alpha[0] = em[0] - max(em[0])  # uniform initial law, normalized by a common constant
    beta[-1] = 0.0
    for j in range(1, B):
        b = B - 1 - j
        for h in range(4):
            f = -np.inf
            g = -np.inf
            for a in range(4):
                f = np.logaddexp(f, alpha[j - 1, a] + trans[j - 1, a, h])
                g = np.logaddexp(g, trans[b, h, a] + em[b + 1, a] + beta[b + 1, a])
            alpha[j, h] = f + em[j, h]
            beta[b, h] = g
        alpha[j] -= max(alpha[j])
        beta[b] -= max(beta[b])
    margins = np.empty(len(pairs))
    for i, (left, right) in enumerate(pairs):
        # Two restricted forwards share every normalization constant.
        m = np.full((2, 4), -np.inf)
        for a in range(4):
            for u in range(2):
                if labels[left, 0, a // 2] == focal[u, left] and labels[left, 1, a % 2] == focal[1 - u, left]:
                    m[u, a] = alpha[left, a]
        n = np.empty_like(m)
        for b in range(left + 1, right + 1):
            for u in range(2):
                for h in range(4):
                    f = -np.inf
                    for a in range(4):
                        f = np.logaddexp(f, m[u, a] + trans[b - 1, a, h])
                    n[u, h] = f + em[b, h]
            m[:] = n - np.max(n)
        mass = np.full(2, -np.inf)
        for h in range(4):
            for u in range(2):
                if labels[right, 0, h // 2] == focal[u, right] and labels[right, 1, h % 2] == focal[1 - u, right]:
                    for v in range(2):
                        mass[u ^ v] = np.logaddexp(mass[u ^ v], m[v, h] + beta[right, h])
        margins[i] = mass[0] - mass[1]
    return margins


def transmission_breaks(model, child, ctx, flips):
    """Right anchors below the set-continuation odds, in ascending bin order."""
    parents = model.par[child]
    if parents[0] < 0 or parents[1] < 0:
        return ()
    passages = _tied_passages(model.X, model.V, parents, child)
    if len(passages) == 0:
        return ()

    fitted = model.X[child]
    copied = model.V[child] < 2
    labels = np.empty((model.B, 2, 2), dtype=np.int64)
    rate = np.exp(model.mmove)
    switch = np.empty((2, len(rate)))
    for slot, parent in enumerate(parents):
        labels[:, slot, :] = np.where(copied[slot, :, None], model.X[parent].T, fitted[slot, :, None])
        switch[slot] = np.where(copied[slot, 1:] & copied[slot, :-1], rate, 0.5)

    # Native release maps independent child/solve pairs in parallel. Within each
    # pair the recursions depend on adjacent bins, and passages have disjoint interiors.
    emissions = _bin_evidence(
        labels, ctx.site_bin, ctx.site_alleles, ctx.site_gl, ctx.site_cohort, ctx.robust_epsilon, *flips
    )
    odds = _continuation_odds(emissions, switch, labels, fitted, passages)
    posterior = ctx.config.set_posterior
    threshold = math.log(posterior / (1 - posterior))
    return tuple(passages[odds < threshold, 1].tolist())
