"""One emission family for the observed allele depths of every read model.

Per sample s, a cell with depth d and a ALT reads (r = d - a REF reads) has three genotype
kernels (the binomial coefficient C(d, a) is common to all three and omitted):

    g = 0 (REF/REF): binomial(a; e)
    g = 1 (REF/ALT): beta-binomial(a; mean q, inverse concentration k)
    g = 2 (ALT/ALT): binomial(a; 1 - e)

with e = expit(error logit), balance = expit(balance logit), q = e + (1 - 2e) balance (so
e < q < 1 - e) and k = 1 / kappa the heterozygote dispersion (0 = binomial heterozygotes). Integer
depths let every term be a rising product, log B(m/c + a, (1-m)/c + r) / B(m/c, (1-m)/c) =
sum_{j<a} log(m + j c) + sum_{j<r} log(1 - m + j c) - sum_{j<d} log(1 + j c); the eight cumulative
sums of :func:`prefix` give these terms and their derivatives, exactly at the binomial limit c = 0
(no log-gamma cancellation; the homozygote kernels are that limit). The models of ``fit.py`` are
special cases: shared or per-sample binomial (k = 0) and sample beta-binomial heterozygotes
(k > 0).

Kernels write one row per sample (or per cell); callers reduce them in sample order, so every
result is independent of the thread count. No fast-math.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _logsum3(a, b, c):
    high = max(a, b, c)
    return high + math.log(math.exp(a - high) + math.exp(b - high) + math.exp(c - high))


@njit(cache=True, nogil=True)
def prefix(mean, inverse_concentration, maximum_depth):
    """Cumulative sums over j < n (n = 0..maximum_depth) of a beta-binomial component.

    Rows: log(m + j c), log(1 - m + j c), log1p(j c), 1/(m + j c), 1/(1 - m + j c),
    j/(m + j c), j/(1 - m + j c), j/(1 + j c); m = mean, c = inverse concentration.
    """
    out = np.zeros((8, maximum_depth + 1))
    for j in range(maximum_depth):
        shift = j * inverse_concentration
        first, second, total = mean + shift, 1. - mean + shift, 1. + shift
        out[0, j + 1] = out[0, j] + math.log(first)
        out[1, j + 1] = out[1, j] + math.log(second)
        out[2, j + 1] = out[2, j] + math.log1p(shift)
        out[3, j + 1] = out[3, j] + 1. / first
        out[4, j + 1] = out[4, j] + 1. / second
        out[5, j + 1] = out[5, j] + j / first
        out[6, j + 1] = out[6, j] + j / second
        out[7, j + 1] = out[7, j] + j / total
    return out


@njit(cache=True, nogil=True)
def sample_objective(local, het_inverse, counts, depth, alt, mean, variance):
    """Penalised log likelihood of one sample's histogram and its gradient.

    local: (error logit, balance logit, mixing logit 0, mixing logit 1); the genotype-mixture
    logits are relative to genotype 2. het_inverse = 1/kappa (0: binomial heterozygotes).
    counts: histogram over the (depth, alt) cells. mean, variance: normal prior on the two logits
    (variance 0 = none). Gradient order: error logit, balance logit, mixing 0, mixing 1, log kappa.
    """
    e = 1. / (1. + math.exp(-local[0]))
    balance = 1. / (1. + math.exp(-local[1]))
    q = e + (1. - 2. * e) * balance
    normaliser = _logsum3(local[2], local[3], 0.)
    log_mix0, log_mix1, log_mix2 = local[2] - normaliser, local[3] - normaliser, -normaliser
    maximum_depth = 0
    for cell in range(len(counts)):
        if counts[cell] != 0. and depth[cell] > maximum_depth:
            maximum_depth = int(depth[cell])
    het = prefix(q, het_inverse, maximum_depth)
    hom = prefix(e, 0., maximum_depth)
    gradient = np.zeros(5)
    value, weight, mass0, mass1 = 0., 0., 0., 0.
    for cell in range(len(counts)):
        count = counts[cell]
        if count == 0.:
            continue
        d, a = int(depth[cell]), int(alt[cell])
        r = d - a
        l0 = hom[0, a] + hom[1, r] - hom[2, d] + log_mix0
        l1 = het[0, a] + het[1, r] - het[2, d] + log_mix1
        l2 = hom[0, r] + hom[1, a] - hom[2, d] + log_mix2
        z = _logsum3(l0, l1, l2)
        r0, r1, r2 = count * math.exp(l0 - z), count * math.exp(l1 - z), count * math.exp(l2 - z)
        value += count * z
        weight += count
        mass0 += r0
        mass1 += r1
        de0 = hom[3, a] - hom[4, r]
        de2 = hom[3, r] - hom[4, a]
        dq = het[3, a] - het[4, r]
        gradient[0] += (r0 * de0 + r2 * de2 + r1 * dq * (1. - 2. * balance)) * e * (1. - e)
        gradient[1] += r1 * dq * (1. - 2. * e) * balance * (1. - balance)
        gradient[4] -= r1 * het_inverse * (het[5, a] + het[6, r] - het[7, d])
    gradient[2] = mass0 - weight * math.exp(log_mix0)
    gradient[3] = mass1 - weight * math.exp(log_mix1)
    for j in range(2):
        if variance[j] > 0.:
            delta = local[j] - mean[j]
            value -= delta * delta / (2. * variance[j])
            gradient[j] -= delta / variance[j]
    return value, gradient


@njit(cache=True, nogil=True)
def samples_objective_serial(local, het_inverse, hist, depth, alt, mean, variance):
    """sample_objective for every sample (rows), on the calling thread (no Numba team)."""
    values = np.empty(len(hist))
    gradients = np.empty((len(hist), 5))
    for sample in range(len(hist)):
        values[sample], gradients[sample] = sample_objective(
            local[sample], het_inverse, hist[sample], depth, alt, mean, variance)
    return values, gradients


@njit(cache=True, nogil=True, parallel=True)
def samples_objective_parallel(local, het_inverse, hist, depth, alt, mean, variance):
    """sample_objective for every sample (rows), samples in parallel on the current Numba team."""
    values = np.empty(len(hist))
    gradients = np.empty((len(hist), 5))
    for sample in prange(len(hist)):
        values[sample], gradients[sample] = sample_objective(
            local[sample], het_inverse, hist[sample], depth, alt, mean, variance)
    return values, gradients


@njit(cache=True, nogil=True, parallel=True)
def em_statistics(hist, depth, alt, error, heterozygote, mixture):
    """E-step of the shared binomial model: per-sample log likelihood, genotype mass and moments.

    Moments: wrong-allele reads and depth in homozygotes, ALT reads and depth in heterozygotes
    (the error and heterozygote-fraction numerators/denominators of the M-step). Sample-wise
    outputs keep the pooled reduction deterministic.
    """
    values = np.zeros(len(hist))
    mass = np.zeros((len(hist), 3))
    moments = np.zeros((len(hist), 4))
    probability = np.array([error, heterozygote, 1. - error])
    log_probability = np.log(probability)
    log_complement = np.log1p(-probability)
    for sample in prange(len(hist)):
        log_mixture = np.log(np.maximum(mixture[sample], 1e-300))
        for cell in range(len(depth)):
            count = hist[sample, cell]
            if count == 0.:
                continue
            d, a = depth[cell], alt[cell]
            l0 = a * log_probability[0] + (d - a) * log_complement[0] + log_mixture[0]
            l1 = a * log_probability[1] + (d - a) * log_complement[1] + log_mixture[1]
            l2 = a * log_probability[2] + (d - a) * log_complement[2] + log_mixture[2]
            z = _logsum3(l0, l1, l2)
            r0 = count * math.exp(l0 - z)
            r1 = count * math.exp(l1 - z)
            r2 = count * math.exp(l2 - z)
            values[sample] += count * z
            mass[sample, 0] += r0
            mass[sample, 1] += r1
            mass[sample, 2] += r2
            moments[sample, 0] += r0 * a + r2 * (d - a)
            moments[sample, 1] += (r0 + r2) * d
            moments[sample, 2] += r1 * a
            moments[sample, 3] += r1 * d
    return values, mass, moments


@njit(cache=True, inline="always")
def cell_probabilities(hom, het, alt, ref, depth):
    """Normalised P(reads | g), g = 0, 1, 2, of one cell with ``alt`` + ``ref`` = ``depth`` > 0 reads from the
    homozygote and heterozygote prefix tables: the three beta-binomial kernels with the maximum shifted out,
    exponentiated and divided by their sum."""
    l0 = hom[0, alt] + hom[1, ref] - hom[2, depth]
    l1 = het[0, alt] + het[1, ref] - het[2, depth]
    l2 = hom[0, ref] + hom[1, alt] - hom[2, depth]
    high = max(l0, l1, l2)
    p0, p1, p2 = math.exp(l0 - high), math.exp(l1 - high), math.exp(l2 - high)
    total = p0 + p1 + p2
    return p0 / total, p1 / total, p2 / total


@njit(cache=True, nogil=True)
def genotype_probabilities(error, heterozygote, het_inverse, depth, alt):
    """Normalised P(reads | g) of one sample for the cells (depth[i], alt[i]): float64[cells, 3].

    The three kernels are normalised over g (max shift, exp, divide); a zero-depth cell is
    exactly (1/3, 1/3, 1/3). ``heterozygote`` is q (the heterozygote ALT mean).
    """
    maximum_depth = 0
    for cell in range(len(depth)):
        if depth[cell] > maximum_depth:
            maximum_depth = int(depth[cell])
    hom = prefix(error, 0., maximum_depth)
    het = prefix(heterozygote, het_inverse, maximum_depth)
    out = np.empty((len(depth), 3))
    for cell in range(len(depth)):
        d, a = int(depth[cell]), int(alt[cell])
        if d == 0:
            out[cell, 0] = out[cell, 1] = out[cell, 2] = 1. / 3.
            continue
        out[cell, 0], out[cell, 1], out[cell, 2] = cell_probabilities(hom, het, a, d - a, d)
    return out


@njit(cache=True, parallel=True)
def gather_likelihoods(counts, table, cap):
    """Fill (N, L, 3) likelihoods from per-sample lookup tables (depth <= cap; deeper cells 0).

    table[s, d, a] = genotype_probabilities of sample s at (d, a). Independent sample x
    4096-site tiles in parallel.
    """
    n, length = counts.shape[:2]
    out = np.empty((n, length, 3), np.float64)
    chunks = (length + 4095) // 4096
    for task in prange(n * chunks):
        sample, chunk = task // chunks, task % chunks
        for site in range(chunk * 4096, min(length, (chunk + 1) * 4096)):
            ref = int(counts[sample, site, 0])
            alt = int(counts[sample, site, 1])
            depth = ref + alt
            if depth <= cap:
                for g in range(3):
                    out[sample, site, g] = table[sample, depth, alt, g]
            else:
                for g in range(3):
                    out[sample, site, g] = 0.
    return out


@njit(cache=True, nogil=True, parallel=True)
def likelihoods_by_sample(counts, error, heterozygote, het_inverse, lookup_depth):
    """Apply the same lookup/direct kernels with one lookup table per active sample.

    Prefix sums and cell normalisation have the same evaluation order as
    ``calibrate.likelihoods``. Only table lifetime changes: O(threads * cap**2)
    scratch instead of O(samples * cap**2), with no second contig-sized output.
    """
    n, length = counts.shape[:2]
    out = np.empty((n, length, 3), np.float64)
    for sample in prange(n):
        deepest = 0
        for site in range(length):
            depth = int(counts[sample, site, 0]) + int(counts[sample, site, 1])
            deepest = max(deepest, depth)
        cap = min(lookup_depth, deepest)
        hom = prefix(error[sample], 0., deepest)
        het = prefix(heterozygote[sample], het_inverse, deepest)
        table = np.empty((cap + 1, cap + 1, 3), np.float64)
        table[0, 0, :] = 1. / 3.
        for depth in range(1, cap + 1):
            for alt in range(depth + 1):
                table[depth, alt, 0], table[depth, alt, 1], table[depth, alt, 2] = (
                    cell_probabilities(hom, het, alt, depth - alt, depth))
        for site in range(length):
            ref = int(counts[sample, site, 0])
            alt = int(counts[sample, site, 1])
            depth = ref + alt
            if depth <= cap:
                for genotype in range(3):
                    out[sample, site, genotype] = table[depth, alt, genotype]
            else:
                out[sample, site, 0], out[sample, site, 1], out[sample, site, 2] = (
                    cell_probabilities(hom, het, alt, ref, depth))
    return out


@njit(cache=True, nogil=True, parallel=True)
def shared_nuisance_curvature(local, hist, depth, alt):
    """Negative-LL mixture curvature, evaluated through three pair covariances.

    Cov(pi)-Cov(r) has off-diagonal pair terms r_i*r_j-pi_i*pi_j.
    Form the diagonal as the sum over the other categories, preserving the
    contribution of a nearly absent third category without 1-p cancellation.
    The binomial kernels and posterior weights are exactly those of the objective.
    """
    blocks = np.zeros((len(hist), 2, 2))
    coupling = np.zeros((len(hist), 2, 2))
    for sample in prange(len(hist)):
        x = local[sample]
        e = 1. / (1. + math.exp(-x[0]))
        balance = 1. / (1. + math.exp(-x[1]))
        q = e + (1. - 2. * e) * balance
        z = _logsum3(x[2], x[3], 0.)
        lm0, lm1, lm2 = x[2] - z, x[3] - z, -z
        p0, p1, p2 = math.exp(lm0), math.exp(lm1), math.exp(lm2)
        maximum_depth = 0
        for cell in range(len(depth)):
            if hist[sample, cell] != 0.:
                maximum_depth = max(maximum_depth, int(depth[cell]))
        hom, het = prefix(e, 0., maximum_depth), prefix(q, 0., maximum_depth)
        w01, w02, w12 = 0., 0., 0.
        for cell in range(len(depth)):
            count = hist[sample, cell]
            if count == 0.:
                continue
            d, a = int(depth[cell]), int(alt[cell]); r = d - a
            l0 = hom[0, a] + hom[1, r] - hom[2, d] + lm0
            l1 = het[0, a] + het[1, r] - het[2, d] + lm1
            l2 = hom[0, r] + hom[1, a] - hom[2, d] + lm2
            z = _logsum3(l0, l1, l2)
            r0, r1, r2 = math.exp(l0-z), math.exp(l1-z), math.exp(l2-z)
            w01 += count * (p0*p1-r0*r1)
            w02 += count * (p0*p2-r0*r2)
            w12 += count * (p1*p2-r1*r2)
            de0 = hom[3, a] - hom[4, r]
            de2 = hom[3, r] - hom[4, a]
            dq = het[3, a] - het[4, r]
            s0 = de0 * e * (1. - e)
            s2 = de2 * e * (1. - e)
            s1 = dq * (1. - 2. * balance) * e * (1. - e)
            t1 = dq * (1. - 2. * e) * balance * (1. - balance)
            coupling[sample, 0, 0] -= count * (r0*r1*(s0-s1)+r0*r2*(s0-s2))
            coupling[sample, 1, 0] -= count * (r0*r1*(s1-s0)+r1*r2*(s1-s2))
            coupling[sample, 0, 1] += count * r0*r1*t1
            coupling[sample, 1, 1] -= count * (r0*r1+r1*r2)*t1
        blocks[sample, 0, 0] = w01 + w02
        blocks[sample, 1, 1] = w01 + w12
        blocks[sample, 0, 1] = blocks[sample, 1, 0] = -w01
    return blocks, coupling
