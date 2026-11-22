"""The painting's read matrices: the read state's coupling to the founder labels (science.shared.reads).

The read state couples to the founder labels only through each site's alleles. For a sample, an ordered label pair
(x1, x2) and a site l of a bin, the GL-anchored read factor of class s is

    R_l(s) = [(1 - eps) sum_a W(a | x1, x2) L_a kappa_s(a) + eps B] / G,
    kappa_s(a) = A_l(a, s) / sum_s' pi_l(s') A_l(a, s'),
    G = (1 - eps) sum_a W(a | x1, x2) L_a + eps B,    f_l(s) = R_l(s) ** w   (w = 1 / the bin's design effect)

with A the class likelihood of the site's REF/ALT counts (science.shared.reads.class_likelihoods), W the pair's
configuration law (class alleles, BACKGROUND and uncalled alleles at the background frequency, two tracks of one
class sharing one allele), L the genotype likelihoods, B the cohort genotype law's likelihood (the outlier branch,
science.shared.observation: an outlier site's reads do not follow the read model) and pi_l the class law at the site.
sum_s pi_l(s) R_l(s) = 1, and R = 1 at an unobserved site and wherever the pair's alleles agree. Each site's factor is
tempered with its bin's weight w: today's tempered scalar emission carries G ** w, the read matrix R ** w.

The bin's read matrix is M_b = prod over its sites l (in order) of T_l diag(f_l), T_l the class operator from the
previous selected site (composed over skipped evidence sites; the first site's operator links the bin to the previous
bin, the identity at the component's first site). The painting chain multiplies today's tempered scalar bin emission
by M_b: with no read sharing (rows of T_l equal pi_l) and w = 1 every product with a class vector reduces to today's
emission exactly.

Matrices are built per unordered label pair; the mirror pair (x2, x1) reads the same matrix with the homologues'
classes swapped (swap_classes)."""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import kernel

__all__ = ["swap_classes", "pair_representatives", "read_matrices", "hidden_read_matrices", "deep_hidden_log"]


def swap_classes(minor: int) -> np.ndarray:
    """Class permutation exchanging homologues 1 and 2 (science.shared.reads.split_class_map's order)."""
    n_classes = 2 * minor + 3
    perm = np.arange(n_classes)
    for j in range(minor + 1):
        perm[1 + j] = minor + 2 + j
        perm[minor + 2 + j] = 1 + j
    return perm


_I1 = types.Array(types.int64, 1, "C")
_B2 = types.Array(types.boolean, 2, "C")
_F1 = types.Array(types.float64, 1, "C")
_F2 = types.Array(types.float64, 2, "C")
_F3 = types.Array(types.float64, 3, "C")
_F4 = types.Array(types.float64, 4, "C")
_M5 = types.Array(types.float32, 5, "C")
_I2 = types.Array(types.int64, 2, "C")
_I1R = types.Array(types.int64, 1, "C", readonly=True)
_B2R = types.Array(types.boolean, 2, "C", readonly=True)
_F3R = types.Array(types.float64, 3, "C", readonly=True)
# Bins per work item of the read-matrix kernels (dimension-only scheduling: no effect on any value).
MATRIX_CHUNK_BINS = 64
_OVERLOADS = {
    "_matrices": (
        (_F4, _F3, _F4, _B2, _F2, _F1, _F3, _F2, types.float64, _I1, _I1, _I1, _I1, _F2, _I2, _M5, _F3),
        (_F4, _F3, _F4, _B2R, _F2, _F1, _F3R, _F2, types.float64, _I1, _I1, _I1, _I1, _F2, _I2, _M5, _F3),
    ),
    "_pair_representatives": (
        (_F2, _I1, _I1, _I1, _I1, _I2),
        (_F2, _I1R, _I1R, _I1, _I1, _I2),
    ),
}


@kernel(overloads=_OVERLOADS["_pair_representatives"], cache=True, parallel=True, nogil=True)
def _pair_representatives(alt, bin_starts, bin_stops, first_label, second_label, out):
    """out[U, B]: per bin, the first unordered pair whose read matrix equals this pair's by construction: the same
    ALT probabilities of its first label and of its second label at every site of the bin, and the same 'one named
    class' flag (x1 == x2, not BACKGROUND). A pair's factors depend on the labels only through these, so pairs with
    one representative have identical matrices and log scales at every sample (unobserved sites contribute one
    whatever the pair)."""
    n_labels = alt.shape[0]
    n_pairs = len(first_label)
    for b in prange(len(bin_starts)):
        low = bin_starts[b]
        high = bin_stops[b]
        group = np.empty(n_labels, dtype=np.int64)          # the first label with this label's ALT values in the bin
        for x in range(n_labels):
            group[x] = x
            for y in range(x):
                if group[y] != y:
                    continue
                equal = True
                for site in range(low, high):
                    if alt[x, site] != alt[y, site]:
                        equal = False
                        break
                if equal:
                    group[x] = y
                    break
        table = np.full((2, n_labels, n_labels), -1, dtype=np.int64)
        for u in range(n_pairs):
            x1 = first_label[u]
            x2 = second_label[u]
            same = 1 if (x1 == x2 and x1 < n_labels - 1) else 0
            g1 = group[x1]
            g2 = group[x2]
            if table[same, g1, g2] < 0:
                table[same, g1, g2] = u
            out[u, b] = table[same, g1, g2]


def pair_representatives(alt, bin_starts, bin_stops, first, second):
    """int64[U, B]: _pair_representatives of the unordered pairs (first, second) over the bins."""
    out = np.empty((len(first), len(bin_starts)), dtype=np.int64)
    _pair_representatives(np.ascontiguousarray(alt, dtype=np.float64), np.asarray(bin_starts, dtype=np.int64),
                          np.asarray(bin_stops, dtype=np.int64), np.asarray(first, dtype=np.int64),
                          np.asarray(second, dtype=np.int64), out)
    return out


@kernel(overloads=_OVERLOADS["_matrices"], cache=True, parallel=True, nogil=True)
def _matrices(class_lik, site_law, operators, observed, alt, background, gl, cohort, epsilon, bin_starts, bin_stops,
              first_label, second_label, weight, representative, out, out_log):
    """out[S, U, B, C, C] float32. class_lik [S, L, 4, C]; site_law [L, S, C]; operators [L, S, C, C] between
    consecutive selected sites; observed [S, L]; alt [X, L] label ALT probabilities (BACKGROUND last); background
    [L]; gl [S, L, 3] the (unmixed) genotype likelihoods; cohort [L, 3] cohort genotype frequencies; first_label /
    second_label [U] the unordered pairs (first <= second); weight [S, B] the bins' tempering weights;
    representative [U, B] (_pair_representatives): a pair-bin whose representative is another pair is not computed
    but copied from it (the same arithmetic on the same values: identical).

    Work items are (sample, run of MATRIX_CHUNK_BINS bins), each walking its sites once: the pair-independent parts of
    a site (kappa_s(a), the cohort and pair-free terms) are formed once and every representative pair's running
    matrix is advanced with them, in the order and arithmetic of a pair-by-pair walk."""
    n_samples, n_sites = observed.shape
    n_pairs = len(first_label)
    n_bins = len(bin_starts)
    n_classes = class_lik.shape[3]
    n_labels = alt.shape[0]
    n_chunks = (n_bins + MATRIX_CHUNK_BINS - 1) // MATRIX_CHUNK_BINS
    for task in prange(n_samples * n_chunks):
        sample = task // n_chunks
        chunk = task % n_chunks
        matrix = np.empty((n_pairs, n_classes, n_classes))
        log_scale = np.empty(n_pairs)
        work = np.empty((n_classes, n_classes))
        factor = np.empty(n_classes)
        kappa = np.empty((4, n_classes))
        for b in range(chunk * MATRIX_CHUNK_BINS, min(n_bins, (chunk + 1) * MATRIX_CHUNK_BINS)):
            for pair in range(n_pairs):
                if representative[pair, b] != pair:
                    continue
                log_scale[pair] = 0.0
                for i in range(n_classes):
                    for j in range(n_classes):
                        matrix[pair, i, j] = 1.0 if i == j else 0.0
            for site in range(bin_starts[b], bin_stops[b]):
                seen = observed[sample, site]
                g0 = gl[sample, site, 0]
                g1 = gl[sample, site, 1]
                g2 = gl[sample, site, 2]
                b_cohort = 0.0
                if seen:
                    # the pair-free parts: the outlier branch's likelihood and kappa_s(a) of every configuration
                    b_cohort = cohort[site, 0] * g0 + cohort[site, 1] * g1 + cohort[site, 2] * g2
                    for config in range(4):
                        mean = 0.0
                        for s in range(n_classes):
                            mean += site_law[site, sample, s] * class_lik[sample, site, config, s]
                        for s in range(n_classes):
                            kappa[config, s] = class_lik[sample, site, config, s] / mean if mean > 0.0 else 1.0
                tempered = weight[sample, b]
                first_site = site == bin_starts[b]
                for pair in range(n_pairs):
                    if representative[pair, b] != pair:
                        continue
                    if seen:
                        x1 = first_label[pair]
                        x2 = second_label[pair]
                        q1 = alt[x1, site]
                        q2 = alt[x2, site]
                        if x1 == x2 and x1 < n_labels - 1:
                            w00, w01, w10, w11 = 1.0 - q1, 0.0, 0.0, q1
                        else:
                            w00, w01, w10, w11 = (1.0 - q1) * (1.0 - q2), (1.0 - q1) * q2, q1 * (1.0 - q2), q1 * q2
                        # per-configuration ratio: R_s = [(1 - eps) sum_a W(a) L_a kappa_s(a) + eps B] / G,
                        # kappa_s(a) = A_s(a) / sum_s' pi(s') A_s'(a); the outlier branch carries no read dependence
                        p_pair = w00 * g0 + (w01 + w10) * g1 + w11 * g2
                        mixed = (1.0 - epsilon) * p_pair + epsilon * b_cohort
                        for s in range(n_classes):
                            factor[s] = 0.0
                        for config in range(4):
                            wa = w00 if config == 0 else (w01 if config == 1 else (w10 if config == 2 else w11))
                            if wa == 0.0:
                                continue
                            la = g0 if config == 0 else (g2 if config == 3 else g1)
                            for s in range(n_classes):
                                factor[s] += wa * la * kappa[config, s]
                        for s in range(n_classes):
                            ratio = ((1.0 - epsilon) * factor[s] + epsilon * b_cohort) / mixed if mixed > 0.0 else 1.0
                            factor[s] = ratio ** tempered      # E_s^w = G^w R_s^w: today's tempered scalar carries G^w
                    else:
                        for s in range(n_classes):
                            factor[s] = 1.0
                    # matrix <- matrix @ T_site @ diag(factor) (T of the component's first selected site: identity);
                    # at the bin's first site the matrix is the identity and the product is T itself (exactly)
                    if first_site:
                        for i in range(n_classes):
                            for j in range(n_classes):
                                work[i, j] = operators[site, sample, i, j] * factor[j]
                    else:
                        for i in range(n_classes):
                            for j in range(n_classes):
                                acc = 0.0
                                for k in range(n_classes):
                                    acc += matrix[pair, i, k] * operators[site, sample, k, j]
                                work[i, j] = acc * factor[j]
                    # rescale against over/underflow along the bin; the scale is kept (it differs between label pairs)
                    top = 0.0
                    for i in range(n_classes):
                        for j in range(n_classes):
                            if work[i, j] > top:
                                top = work[i, j]
                    scale = 1.0 / top if top > 0.0 else 1.0
                    if top > 0.0:
                        log_scale[pair] += np.log(top)
                    for i in range(n_classes):
                        for j in range(n_classes):
                            matrix[pair, i, j] = work[i, j] * scale
            for pair in range(n_pairs):
                r = representative[pair, b]
                for i in range(n_classes):
                    for j in range(n_classes):
                        out[sample, pair, b, i, j] = matrix[r, i, j]
                out_log[sample, pair, b] = log_scale[r]


def read_matrices(class_lik, site_law, operators, observed, alt, background, gl, cohort, epsilon, bin_starts,
                  bin_stops, weight=None):
    """(matrices float32[S, U, B, C, C], log_scale float64[S, U, B], first, second): the read matrices of every
    unordered label pair (U = X (X + 1) / 2, row-major upper triangle) and bin, each rescaled along its product, and
    the log of the removed scale: the bin's read factor is exp(log_scale) x matrix, so log_scale is added to the
    pair's bin log emission (it differs between pairs and must not be dropped). ``weight`` [S, B]: the bins'
    tempering weights (None: one, no tempering)."""
    n_labels = alt.shape[0]
    first, second = np.triu_indices(n_labels)
    n_samples, n_sites = observed.shape
    n_classes = class_lik.shape[3]
    out = np.empty((n_samples, len(first), len(bin_starts), n_classes, n_classes), dtype=np.float32)
    out_log = np.empty((n_samples, len(first), len(bin_starts)), dtype=np.float64)
    if weight is None:
        weight = np.ones((n_samples, len(bin_starts)))
    representative = pair_representatives(alt, bin_starts, bin_stops, first, second)
    _matrices(np.ascontiguousarray(class_lik), np.ascontiguousarray(site_law), np.ascontiguousarray(operators),
              np.ascontiguousarray(observed), np.ascontiguousarray(alt), np.array(background, dtype=np.float64),
              np.ascontiguousarray(gl), np.ascontiguousarray(cohort), float(epsilon),
              np.array(bin_starts, dtype=np.int64), np.array(bin_stops, dtype=np.int64),
              first.astype(np.int64), second.astype(np.int64), np.ascontiguousarray(weight, dtype=np.float64),
              representative, out, out_log)
    return out, out_log, first, second


# ----------------------------------------------------------------------------------------- hidden blocks (calibration)
_B1 = types.Array(types.boolean, 1, "C")
_OVERLOADS.update({
    "_hidden_matrices": ((_F4, _F3, _F4, _B2, _B1, _F2, _F3, _F2, types.float64, _I1, _I1, _I1, _I1, _F2, _I2, _M5,
                          _F3),),
    "_deep_hidden_log": ((_I2, _I2, _F1, _B2, _B1, _F2, _F2, types.float64, _I1, _I1, _I1, _I1, _F3),),
})


@kernel(overloads=_OVERLOADS["_hidden_matrices"], cache=True, parallel=True, nogil=True)
def _hidden_matrices(class_lik, site_law, operators, observed, hidden, alt, gl, cohort, epsilon, bin_starts, bin_stops,
                     first_label, second_label, weight, representative, out, out_log):
    """out[S, U, B, C, C] float32: the read matrices of _matrices with the bin's hidden sites (``hidden`` bool[L], the
    calibration fold's hidden part) predicted under the normalised hidden law. A visible observed site (observed and
    not hidden) contributes its tempered GL-anchored factor exactly as in _matrices; a hidden observed site contributes
    the probability of its ALT count given the class s, weight one:
    F_s = (1 - eps) sum_a W(a | x1, x2) A_s(a) + eps sum_g q_g sum_t pi(t) A_t(a(g)),
    A the class likelihoods (science.shared.reads: for each class and ordered configuration a proper PMF of the ALT
    count at the site's depth), W the pair's configuration law (two tracks of one named class share one allele), q
    the site's cohort genotype law and pi its class law (the outlier term carries no read dependence). ``observed``
    holds the ORIGINAL flags (hidden sites included). ``representative`` as in _matrices (the hidden flags are one
    per site, the same for every pair)."""
    n_samples, n_sites = observed.shape
    n_pairs = len(first_label)
    n_bins = len(bin_starts)
    n_classes = class_lik.shape[3]
    n_labels = alt.shape[0]
    n_chunks = (n_bins + MATRIX_CHUNK_BINS - 1) // MATRIX_CHUNK_BINS
    for task in prange(n_samples * n_chunks):
        sample = task // n_chunks
        chunk = task % n_chunks
        matrix = np.empty((n_pairs, n_classes, n_classes))
        log_scale = np.empty(n_pairs)
        work = np.empty((n_classes, n_classes))
        factor = np.empty(n_classes)
        kappa = np.empty((4, n_classes))
        for b in range(chunk * MATRIX_CHUNK_BINS, min(n_bins, (chunk + 1) * MATRIX_CHUNK_BINS)):
            for pair in range(n_pairs):
                if representative[pair, b] != pair:
                    continue
                log_scale[pair] = 0.0
                for i in range(n_classes):
                    for j in range(n_classes):
                        matrix[pair, i, j] = 1.0 if i == j else 0.0
            for site in range(bin_starts[b], bin_stops[b]):
                seen = observed[sample, site]
                masked = hidden[site]
                g0 = gl[sample, site, 0]
                g1 = gl[sample, site, 1]
                g2 = gl[sample, site, 2]
                b_cohort = 0.0
                outlier = 0.0
                if seen and masked:
                    for t in range(n_classes):
                        law = site_law[site, sample, t]
                        outlier += law * (cohort[site, 0] * class_lik[sample, site, 0, t]
                                          + cohort[site, 1] * class_lik[sample, site, 1, t]
                                          + cohort[site, 2] * class_lik[sample, site, 3, t])
                elif seen:
                    b_cohort = cohort[site, 0] * g0 + cohort[site, 1] * g1 + cohort[site, 2] * g2
                    for config in range(4):
                        mean = 0.0
                        for s in range(n_classes):
                            mean += site_law[site, sample, s] * class_lik[sample, site, config, s]
                        for s in range(n_classes):
                            kappa[config, s] = class_lik[sample, site, config, s] / mean if mean > 0.0 else 1.0
                tempered = weight[sample, b]
                first_site = site == bin_starts[b]
                for pair in range(n_pairs):
                    if representative[pair, b] != pair:
                        continue
                    if seen:
                        x1 = first_label[pair]
                        x2 = second_label[pair]
                        q1 = alt[x1, site]
                        q2 = alt[x2, site]
                        if x1 == x2 and x1 < n_labels - 1:
                            w00, w01, w10, w11 = 1.0 - q1, 0.0, 0.0, q1
                        else:
                            w00, w01, w10, w11 = (1.0 - q1) * (1.0 - q2), (1.0 - q1) * q2, q1 * (1.0 - q2), q1 * q2
                        if masked:
                            for s in range(n_classes):
                                factor[s] = (1.0 - epsilon) * (
                                    w00 * class_lik[sample, site, 0, s] + w01 * class_lik[sample, site, 1, s]
                                    + w10 * class_lik[sample, site, 2, s] + w11 * class_lik[sample, site, 3, s]
                                ) + epsilon * outlier
                        else:
                            p_pair = w00 * g0 + (w01 + w10) * g1 + w11 * g2
                            mixed = (1.0 - epsilon) * p_pair + epsilon * b_cohort
                            for s in range(n_classes):
                                factor[s] = 0.0
                            for config in range(4):
                                wa = w00 if config == 0 else (w01 if config == 1 else (w10 if config == 2 else w11))
                                if wa == 0.0:
                                    continue
                                la = g0 if config == 0 else (g2 if config == 3 else g1)
                                for s in range(n_classes):
                                    factor[s] += wa * la * kappa[config, s]
                            for s in range(n_classes):
                                ratio = (((1.0 - epsilon) * factor[s] + epsilon * b_cohort) / mixed
                                         if mixed > 0.0 else 1.0)
                                factor[s] = ratio ** tempered
                    else:
                        for s in range(n_classes):
                            factor[s] = 1.0
                    if first_site:                          # the identity times T: T exactly (as in _matrices)
                        for i in range(n_classes):
                            for j in range(n_classes):
                                work[i, j] = operators[site, sample, i, j] * factor[j]
                    else:
                        for i in range(n_classes):
                            for j in range(n_classes):
                                acc = 0.0
                                for k in range(n_classes):
                                    acc += matrix[pair, i, k] * operators[site, sample, k, j]
                                work[i, j] = acc * factor[j]
                    top = 0.0
                    for i in range(n_classes):
                        for j in range(n_classes):
                            if work[i, j] > top:
                                top = work[i, j]
                    scale = 1.0 / top if top > 0.0 else 1.0
                    if top > 0.0:
                        log_scale[pair] += np.log(top)
                    for i in range(n_classes):
                        for j in range(n_classes):
                            matrix[pair, i, j] = work[i, j] * scale
            for pair in range(n_pairs):
                r = representative[pair, b]
                for i in range(n_classes):
                    for j in range(n_classes):
                        out[sample, pair, b, i, j] = matrix[r, i, j]
                out_log[sample, pair, b] = log_scale[r]


@kernel(overloads=_OVERLOADS["_deep_hidden_log"], cache=True, parallel=True, nogil=True)
def _deep_hidden_log(ref, alt_count, error, observed, hidden, alt, cohort, epsilon, bin_starts, bin_stops,
                     first_label, second_label, out):
    """out[S, U, B]: the sum over the bin's hidden observed sites of log F of a sample bypassing the read state (every
    called depth at the deep bound: its class is 'balanced' with certainty; or a library without read sharing: each
    read comes from either homologue independently, the same law), F = (1 - eps) sum_a W(a | x1, x2)
    Bin(k; n, (p_g1 + p_g2) / 2) + eps sum_g q_g Bin(k; n, p_g), the balanced class's count PMF (the closed form of
    science.shared.reads) with the per-read other-allele rate e (p_0 = e, p_1 = 1 - e, p_het = 1/2)."""
    n_samples = ref.shape[0]
    n_pairs = len(first_label)
    n_labels = alt.shape[0]
    for task in prange(n_samples * n_pairs):
        sample = task // n_pairs
        pair = task % n_pairs
        x1 = first_label[pair]
        x2 = second_label[pair]
        same = x1 == x2 and x1 < n_labels - 1
        e = error[sample]
        for b in range(len(bin_starts)):
            acc = 0.0
            for site in range(bin_starts[b], bin_stops[b]):
                if not hidden[site] or not observed[sample, site]:
                    continue
                k = alt_count[sample, site]
                n = ref[sample, site] + k
                log_choose = math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)
                hom0 = math.exp(log_choose + k * math.log(e) + (n - k) * math.log(1.0 - e))
                hom1 = math.exp(log_choose + k * math.log(1.0 - e) + (n - k) * math.log(e))
                het = math.exp(log_choose + n * math.log(0.5))
                q1 = alt[x1, site]
                q2 = alt[x2, site]
                if same:
                    pair_law = (1.0 - q1) * hom0 + q1 * hom1
                else:
                    pair_law = ((1.0 - q1) * (1.0 - q2) * hom0 + ((1.0 - q1) * q2 + q1 * (1.0 - q2)) * het
                                + q1 * q2 * hom1)
                outlier = cohort[site, 0] * hom0 + cohort[site, 1] * het + cohort[site, 2] * hom1
                acc += math.log((1.0 - epsilon) * pair_law + epsilon * outlier)
            out[sample, pair, b] = acc


def hidden_read_matrices(class_lik, site_law, operators, observed, hidden, alt, gl, cohort, epsilon, bin_starts,
                         bin_stops, weight):
    """(matrices float32[S, U, B, C, C], log scales float64[S, U, B]): read_matrices with the hidden sites predicted
    under the normalised hidden law (_hidden_matrices); ``observed`` the ORIGINAL flags, ``hidden`` bool[L]."""
    n_labels = alt.shape[0]
    first, second = np.triu_indices(n_labels)
    n_samples = observed.shape[0]
    n_classes = class_lik.shape[3]
    out = np.empty((n_samples, len(first), len(bin_starts), n_classes, n_classes), dtype=np.float32)
    out_log = np.empty((n_samples, len(first), len(bin_starts)), dtype=np.float64)
    _hidden_matrices(np.ascontiguousarray(class_lik), np.ascontiguousarray(site_law), np.ascontiguousarray(operators),
                     np.ascontiguousarray(observed), np.array(hidden, dtype=np.bool_), np.ascontiguousarray(alt),
                     np.ascontiguousarray(gl), np.ascontiguousarray(cohort), float(epsilon),
                     np.array(bin_starts, dtype=np.int64), np.array(bin_stops, dtype=np.int64), first.astype(np.int64),
                     second.astype(np.int64), np.ascontiguousarray(weight, dtype=np.float64),
                     pair_representatives(alt, bin_starts, bin_stops, first, second), out, out_log)
    return out, out_log


def deep_hidden_log(ref, alt_count, error, observed, hidden, alt, cohort, epsilon, bin_starts, bin_stops):
    """float64[S, U, B]: the hidden blocks' log probabilities of samples bypassing the read state (_deep_hidden_log)."""
    first, second = np.triu_indices(alt.shape[0])
    out = np.zeros((ref.shape[0], len(first), len(bin_starts)))
    _deep_hidden_log(np.ascontiguousarray(ref, dtype=np.int64), np.ascontiguousarray(alt_count, dtype=np.int64),
                     np.array(error, dtype=np.float64), np.ascontiguousarray(observed),
                     np.array(hidden, dtype=np.bool_),
                     np.ascontiguousarray(alt), np.ascontiguousarray(cohort), float(epsilon),
                     np.array(bin_starts, dtype=np.int64), np.array(bin_stops, dtype=np.int64), first.astype(np.int64),
                     second.astype(np.int64), out)
    return out
