"""Site count factors and scalar calibration for the painting read state."""

from __future__ import annotations

import math

import numpy as np
from numba import prange, types

from ...core.kernels import kernel

__all__ = ["swap_classes", "pair_representatives", "deep_hidden_log"]


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
_I2 = types.Array(types.int64, 2, "C")
_I1R = types.Array(types.int64, 1, "C", readonly=True)
# Bins per work item of the structured read projections (dimension-only scheduling: no effect on any value; the bin
# correction's carry is a separate pass over (sample, pair), read_operator.carry_correction).
MATRIX_CHUNK_BINS = 64
_OVERLOADS = {
    "_untempered_ratios": ((_F2, types.float64, types.float64, types.float64, types.float64, types.float64,
                            types.float64, types.float64, types.float64, types.float64, _F1),),
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
    one representative share the same own-class bin product, before the correction from each pair's carried law."""
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


@kernel(overloads=((_F4, _F3, types.int64, types.int64, _F2),), cache=True, inline="always", nogil=True)
def own_count_ratios(class_lik, law, sample, site, out):
    """Normalize each own genotype-configuration count law against the site's class law."""
    own = law.shape[2]
    for config in range(4):
        mean = 0.0
        for c in range(own):
            mean += law[site, sample, c] * class_lik[sample, site, config, c]
        for c in range(own):
            out[config, c] = class_lik[sample, site, config, c] / mean if mean > 0.0 else 1.0


@kernel(overloads=((types.int64, types.int64, types.float64, types.float64, types.float64, types.float64,
                    types.float64),), cache=True, inline="always", nogil=True)
def donor_ratio(config, allele, alt_count, depth, error, loss, expected):
    """The site's ratio for one covering read being a donor read carrying ``allele`` (0 REF, 1 ALT) instead of one
    of the sample's own reads, under the ordered allele configuration ``config`` of the sample's homologues (0 = 00,
    1 = 01, 2 = 10, 3 = 11; own per-read ALT probability ``expected``), at ``alt_count`` ALT of ``depth`` called
    reads with per-read other-allele rate ``error`` and loss ``loss``. One (no evidence) for an allele the
    configuration carries: such a read could equally be one of the sample's own reads, and only an allele neither
    homologue carries identifies a foreign read (docs/methods/painting.md, Foreign reads)."""
    if depth <= 0.0 or allele == config // 2 or allele == config % 2:
        return 1.0
    donor = error + (1.0 - 2.0 * error) * allele
    called = alt_count / depth
    return loss + (1.0 - loss) * (called * donor / expected + (1.0 - called) * (1.0 - donor) / (1.0 - expected))


@kernel(overloads=((types.float64, types.float64, types.boolean),), cache=True, inline="always", nogil=True)
def configuration_weights(first, second, repeated_named):
    """Alleles of a repeated named label share a draw. BACKGROUND draws independently."""
    if repeated_named:
        return 1.0 - first, 0.0, 0.0, first
    return (1.0 - first) * (1.0 - second), (1.0 - first) * second, first * (1.0 - second), first * second


@kernel(overloads=_OVERLOADS["_untempered_ratios"], cache=True, inline="always", nogil=True)
def _untempered_ratios(kappa, g0, g1, g2, b_cohort, epsilon, w00, w01, w10, w11, out):
    """out[C] = the site's untempered GL-anchored read factor R(s) of a pair with configuration law (w00, w01, w10,
    w11), before tempering."""
    n_classes = out.shape[0]
    p_pair = w00 * g0 + (w01 + w10) * g1 + w11 * g2
    mixed = (1.0 - epsilon) * p_pair + epsilon * b_cohort
    for s in range(n_classes):
        out[s] = 0.0
    for config in range(4):
        wa = w00 if config == 0 else (w01 if config == 1 else (w10 if config == 2 else w11))
        if wa == 0.0:
            continue
        la = g0 if config == 0 else (g2 if config == 3 else g1)
        for s in range(n_classes):
            out[s] += wa * la * kappa[config, s]
    for s in range(n_classes):
        out[s] = ((1.0 - epsilon) * out[s] + epsilon * b_cohort) / mixed if mixed > 0.0 else 1.0


_B1 = types.Array(types.boolean, 1, "C")
_OVERLOADS["_deep_hidden_log"] = (
    (_I2, _I2, _F1, _B2, _B1, _F2, _F2, types.float64, _I1, _I1, _I1, _I1, _F3),
)


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
