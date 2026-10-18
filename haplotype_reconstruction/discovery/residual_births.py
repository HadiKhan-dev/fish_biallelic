"""Soft-residual birth candidates of a complete panel (the search's residual phase).

For a fitted complete panel, every sample's diplotype posterior over unordered row pairs (a
softmax of the summed log genotype likelihood of each hard pair, uniform pair prior) gives each
row an expected copy count. Every row with at least ``min_soft_responsibility`` expected copies
in a sample is *subtracted* from that sample: the residual strand's ALT probability at an
observed site is P(GL | subtracted allele + residual allele), normalised over the residual
allele. Residual strands are called at ``residual_call_probability``, clustered by missing-aware
Hamming distance (HDBSCAN, then complete linkage on HDBSCAN's noise; every cluster's diameter at
most ``max_cluster_hamming``), globally and within each (subtracted row, dominant partner)
route. Each cluster's consensus pools its samples' responsibility-weighted log odds with one
capped vote per sample; a consensus is emitted when enough distinct and effective samples
support it, enough of it is called, and it is no duplicate of the panel or an earlier emission.
Emission order (release 2.1): more distinct supporting samples first, then the consensus's hardened
0/1 row (> 1/2 is ALT; within the ``probability`` quantum of 1/2 the population ALT fraction decides,
as the search hardens it), then source order -- exact discrete keys (2.0 sorted by a digest of the
probabilities rounded to 12 decimals, which 1e-12 GL noise reshuffled). The order decides which of
two near-duplicate consensus rows is kept, so the better supported one is (measured on stress_n80:
fewer blocks losing a founder than with the hardened row first, design A-6). A record's dominant
partner is the largest partner mass beyond a relative band of the ``probability`` quantum (lower
row on ties). The candidates are soft (ALT probabilities); the search hardens them into birth
starts and its cavity score decides.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from ..maths.ties import tie_quantum

# Bands of posterior-mass decisions (probability-kind quantum): the dominant-partner choice (relative to
# max(1, mass)), and the record-weight and effective-support thresholds, where symmetric evidence puts masses
# exactly at the threshold (e.g. a posterior split exactly in half, effective support exactly 1.5): values within
# the band count as at the threshold, which passes (2.0's decision at exact equality; 2.0 used a 1e-12 fuzz, the
# 1e-12-noise scale).
PARTNER_BAND = tie_quantum(0.0, "probability")
MASS_BAND = tie_quantum(0.0, "probability")
# Values within this band of 1/2 are 1/2 (probability-kind quantum; 2.0 compared exactly or with 1e-12, the 1e-12-noise
# scale). Inside the band a consensus row takes the population ALT fraction, and a population ALT fraction itself
# inside the band takes the rule's decision at exactly 1/2 (2.0's: REF for "> 1/2" rules, ALT for ">= 1/2").
HALF_BAND = tie_quantum(0.0, "probability")


def note_band(counter, kind, values, threshold, band):
    """Count one batch of threshold decisions for the stage records (counter: maths.ties.TieCounter or None):
    decisions = the values compared, inside_tie = those strictly inside the band (0 < |value - threshold| <= band:
    where the band, not the exact comparison, decided)."""
    if counter is None:
        return
    values = np.asarray(values, dtype=np.float64)
    distance = np.abs(values - threshold)
    entry = counter.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
    entry["decisions"] += int(values.size)
    entry["inside_tie"] += int(np.count_nonzero((distance > 0.0) & (distance <= band)))


def half_band_rows(values, fallback):
    """Binary rows of probabilities: > 1/2 is 1, < 1/2 is 0; within HALF_BAND of 1/2 the
    ``fallback`` row (broadcast) decides."""
    values = np.asarray(values, dtype=np.float64)
    rows = values > 0.5
    tied = np.abs(values - 0.5) <= HALF_BAND
    rows[tied] = np.broadcast_to(fallback, rows.shape)[tied]
    return rows


def alt_fraction_rule(population_alt, at_half: bool):
    """population ALT fraction > 1/2 (outside the band); within HALF_BAND of 1/2: ``at_half``."""
    population_alt = np.asarray(population_alt, dtype=np.float64)
    rows = population_alt > 0.5
    rows[np.abs(population_alt - 0.5) <= HALF_BAND] = at_half
    return rows


def hardening_fallback(likelihoods):
    """The ALT/REF row hardening uses within HALF_BAND of 1/2: the population ALT fraction of all
    samples >= 1/2 (ALT within HALF_BAND of 1/2, 2.0's decision at exactly 1/2)."""
    marginal = np.mean(0.5 * likelihoods[:, :, 1] + likelihoods[:, :, 2], axis=0)
    return alt_fraction_rule(marginal, True)


# Elementary functions (release 2.1): discovery takes log and exp from the C library's scalar functions (math.log /
# math.exp in Numba kernels), never from NumPy's float64 ufuncs. NumPy picks those ufuncs' SIMD code paths at run time
# from the CPU (AVX-512 vs AVX2 kernels), and they differ from the scalar function by an ulp in ~0.3% (log) and ~5%
# (exp) of values, so the same evidence gave different soft-residual candidates on different CPUs (actm3 chr8:204 under
# env_haswell). The scalar functions, element-wise arithmetic and fixed-order sums give the same bits on every x86-64
# CPU with FMA (all CSD3 node types); discovery makes no BLAS calls.
@njit(cache=True, nogil=True)
def scalar_log_floored(values, floor):
    """float64 array of log(max(value, floor)) for a C-contiguous float64 array (C library scalar log)."""
    flat = values.reshape(-1)
    out = np.empty(flat.size, dtype=np.float64)
    for index in range(flat.size):
        value = flat[index]
        out[index] = math.log(value if value > floor else floor)
    return out.reshape(values.shape)


@njit(cache=True, nogil=True)
def _exp_inplace(values):
    """values[i, j] = exp(values[i, j]) for a 2-D float64 array of any layout (C library scalar exp)."""
    rows, columns = values.shape
    for row in range(rows):
        for column in range(columns):
            values[row, column] = math.exp(values[row, column])


@njit(cache=True, nogil=True)
def _logistic_where(total, observed):
    """ALT probability 1 / (1 + exp(-total)) where observed (0.5 elsewhere), evaluated on the stable side of 0 with
    the C library scalar exp."""
    out = np.full(total.shape[0], 0.5, dtype=np.float64)
    for site in range(total.shape[0]):
        if observed[site]:
            value = total[site]
            if value >= 0.0:
                out[site] = 1.0 / (1.0 + math.exp(-value))
            else:
                exponential = math.exp(value)
                out[site] = exponential / (1.0 + exponential)
    return out


def _splitmix64(values):
    """splitmix64 of uint64 values (wrapping arithmetic): a fixed pseudo-random content key."""
    z = values + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def exact_hdbscan_labels(distance, min_cluster_size, min_samples, allow_single_cluster):
    """int64 HDBSCAN labels (-1 = noise) of a precomputed float64 distance matrix: the hdbscan package's algorithm
    for ``metric="precomputed"`` (mutual reachability, Prim's minimum spanning tree, single linkage, condensed tree,
    excess-of-mass selection; alpha 1, no epsilon or persistence) with one change (release 2.1): the tree's edges are
    ordered by weight and, among equal weights, by a fixed pseudo-random key of their two record indices (splitmix64).
    The package orders them with NumPy's default argsort, which is unstable and whose order among equal weights depends
    on the CPU's SIMD sort kernel (AVX-512 vs AVX2); equal weights are the rule for Hamming distances, and the order of
    equal-weight merges shapes the condensed tree, so the clusters depended on the CPU (actm3 chr8:204 under
    env_haswell). The key depends only on the (canonical) record order and keeps the unstable sort's effectively random
    resolution of equal-weight merges, with which 2.1 was validated; a stable order would resolve every multi-way merge
    as a chain, so equidistant records would fall out of the growing cluster instead of pairing up (fewer residual
    clusters and soft candidates, measured as a downward shift of the selected K on TroMau)."""
    from hdbscan._hdbscan_linkage import label, mst_linkage_core
    from hdbscan._hdbscan_reachability import mutual_reachability
    from hdbscan.hdbscan_ import _tree_to_labels
    reachability = np.array(distance, dtype=np.float64, order="C", copy=True)
    size = reachability.shape[0]
    min_samples = max(1, min(size - 1, int(min_samples)))
    reachability = mutual_reachability(reachability, min_samples, 1.0)
    tree = mst_linkage_core(reachability)
    first = np.minimum(tree[:, 0], tree[:, 1]).astype(np.uint64)
    second = np.maximum(tree[:, 0], tree[:, 1]).astype(np.uint64)
    tree = tree[np.lexsort((_splitmix64((first << np.uint64(32)) | second), tree[:, 2])), :]
    labels = _tree_to_labels(None, label(tree), int(min_cluster_size), "eom", bool(allow_single_cluster), False,
                             0.0, 0.0, 0, float("inf"))[0]
    return np.asarray(labels, dtype=np.int64)


# Residual strand hard-call sentinel.
_UNCALLED = -1
# Bounds of a residual strand's log odds: release 2.0 clipped the ALT probability to [tiny, 1 - eps] before taking
# log(p) - log1p(-p), so its log odds lay in [log tiny, log((1 - eps) / eps)]. 2.1 computes the log odds exactly as
# log(residual ALT mass) - log(residual REF mass) within the same bounds: from p itself they lost all precision as
# p -> 1 (1 - p carries ~1e-16 absolute error), and 1e-12 GL noise moved a record's log odds by 0.15 nats there
# (TroMau chr7:665, chr17:425), flipping near-cancelling consensus sites.
_LOG_ODDS_MAX = math.log((1.0 - np.finfo(np.float64).eps) / np.finfo(np.float64).eps)
_LOG_ODDS_MIN = math.log(np.finfo(np.float64).tiny) - math.log1p(-np.finfo(np.float64).tiny)
# Bound of the largest temporary of the diplotype responsibility (bytes).
_PREDICTIVE_CHUNK_BYTES = 32 * 1024 * 1024


class ResidualInputs:
    """Evidence-only inputs of one block: likelihood (unobserved cells exactly 1/3), its floored
    logarithm, the observation mask, and an exact cache of residual clusterings (identical
    distance matrices recur across proposals of one block search)."""

    def __init__(self, likelihood, observed):
        self.likelihood = np.ascontiguousarray(likelihood, dtype=np.float64)
        self.log_likelihood = scalar_log_floored(self.likelihood, np.finfo(np.float64).tiny)
        self.observed = np.ascontiguousarray(observed, dtype=np.bool_)
        self.cluster_cache: dict = {}


@njit(cache=True, nogil=True)
def _gather_binary_predictive(likelihood, haplotypes, pair_i, pair_j, predictive):
    """One exact log likelihood per hard diplotype and site, into an F-ordered output."""
    n_samples = likelihood.shape[0]
    n_pairs = pair_i.shape[0]
    n_sites = haplotypes.shape[1]
    for site_index in range(n_sites):
        for pair_index in range(n_pairs):
            genotype = haplotypes[pair_i[pair_index], site_index] + haplotypes[pair_j[pair_index], site_index]
            for sample_index in range(n_samples):
                predictive[sample_index, pair_index, site_index] = likelihood[sample_index, site_index, genotype]


def _diplotype_responsibility(log_likelihood, haplotypes, pair_i, pair_j):
    """float64[N, pairs] posterior of each sample over the unordered row pairs (uniform prior).

    The per-pair log emission is a NumPy sum over sites of an F-ordered (samples, pairs, sites)
    array (fixed summation order); samples are chunked to bound that array, pairs never are
    (chunking pairs would change the array's site stride and hence the summation order).
    """
    n_samples = log_likelihood.shape[0]
    n_pairs = len(pair_i)
    if n_pairs == 1:
        return np.ones((n_samples, 1), dtype=np.float64)
    n_sites = haplotypes.shape[1]
    bytes_per_sample = max(1, n_pairs * n_sites * 8)
    samples_per_chunk = max(1, min(n_samples, _PREDICTIVE_CHUNK_BYTES // bytes_per_sample))
    log_emission = np.empty((n_samples, n_pairs), dtype=np.float64,
                            order="F" if n_samples > 1 else "C")
    for start in range(0, n_samples, samples_per_chunk):
        stop = min(start + samples_per_chunk, n_samples)
        log_predictive = np.empty((stop - start, n_pairs, n_sites), dtype=np.float64, order="F")
        _gather_binary_predictive(log_likelihood[start:stop], haplotypes, pair_i, pair_j,
                                  log_predictive)
        log_emission[start:stop] = np.sum(log_predictive, axis=2)
    row_maximum = np.max(log_emission, axis=1, keepdims=True)
    responsibility = log_emission - row_maximum
    _exp_inplace(responsibility)
    responsibility /= np.sum(responsibility, axis=1, keepdims=True)
    return np.ascontiguousarray(responsibility)


@njit(cache=True, nogil=True)
def _residual_records(responsibility, pair_i, pair_j, n_rows, likelihood, observed, haplotypes,
                      call_probability, minimum_responsibility, partner_band, mass_band):
    """Residual strands of every (sample, subtracted row) with enough expected copies.

    Returns sample, subtracted row, dominant partner (largest posterior partner mass of the
    subtracted row; a later row replaces the current one only when its mass exceeds the current
    by more than partner_band x max(1, current): lowest index on ties), weight min(1, expected
    copies), the residual ALT probability per site (0.5 where unobserved), hard calls (-1
    uncalled), the compatible mask (observed sites with positive residual mass) and the residual
    ALT log odds log(r1) - log(r0) at compatible sites within [_LOG_ODDS_MIN, _LOG_ODDS_MAX] (0
    elsewhere). A record needs weight + mass_band >= minimum_responsibility.
    """
    n_samples, n_states = responsibility.shape
    n_sites = haplotypes.shape[1]
    maximum_records = n_samples * n_rows
    sample_indices = np.empty(maximum_records, dtype=np.int64)
    subtracted = np.empty(maximum_records, dtype=np.int64)
    partners = np.empty(maximum_records, dtype=np.int64)
    weights = np.empty(maximum_records, dtype=np.float64)
    copy_mass = np.zeros(n_rows, dtype=np.float64)
    partner_mass = np.zeros((n_rows, n_rows), dtype=np.float64)
    count = 0
    for sample in range(n_samples):
        copy_mass.fill(0.0)
        partner_mass.fill(0.0)
        for state in range(n_states):
            first = pair_i[state]
            second = pair_j[state]
            mass = responsibility[sample, state]
            if first == second:
                copy_mass[first] += 2.0 * mass
                partner_mass[first, first] += 2.0 * mass
            else:
                copy_mass[first] += mass
                copy_mass[second] += mass
                partner_mass[first, second] += mass
                partner_mass[second, first] += mass
        for row in range(n_rows):
            expected = copy_mass[row]
            weight = min(1.0, expected)
            if weight + mass_band < minimum_responsibility:
                continue
            partner = 0
            best = partner_mass[row, 0]
            for other in range(1, n_rows):
                band = partner_band * (best if best > 1.0 else 1.0)
                if partner_mass[row, other] > best + band:
                    partner = other
                    best = partner_mass[row, other]
            sample_indices[count] = sample
            subtracted[count] = row
            partners[count] = partner
            weights[count] = weight
            count += 1
    soft_alt = np.full((count, n_sites), 0.5, dtype=np.float64)
    hard_calls = np.full((count, n_sites), _UNCALLED, dtype=np.int8)
    compatible = np.zeros((count, n_sites), dtype=np.bool_)
    log_odds = np.zeros((count, n_sites), dtype=np.float64)
    for record in range(count):
        sample = sample_indices[record]
        row = subtracted[record]
        for site in range(n_sites):
            q = float(haplotypes[row, site])
            residual_zero = (1.0 - q) * likelihood[sample, site, 0] + q * likelihood[sample, site, 1]
            residual_one = (1.0 - q) * likelihood[sample, site, 1] + q * likelihood[sample, site, 2]
            denominator = residual_zero + residual_one
            if observed[sample, site] and denominator > 0.0:
                value = residual_one / denominator
                soft_alt[record, site] = value
                compatible[record, site] = True
                if residual_zero <= 0.0:
                    odds = _LOG_ODDS_MAX
                elif residual_one <= 0.0:
                    odds = _LOG_ODDS_MIN
                else:
                    odds = math.log(residual_one) - math.log(residual_zero)
                    if odds > _LOG_ODDS_MAX:
                        odds = _LOG_ODDS_MAX
                    elif odds < _LOG_ODDS_MIN:
                        odds = _LOG_ODDS_MIN
                log_odds[record, site] = odds
                if value >= call_probability:
                    hard_calls[record, site] = 1
                if value <= 1.0 - call_probability:
                    hard_calls[record, site] = 0
    return (sample_indices[:count], subtracted[:count], partners[:count], weights[:count],
            soft_alt, hard_calls, compatible, log_odds)


@njit(cache=True, inline="always")
def _popcount(value):
    """64-bit population count (SWAR reduction)."""
    value = value - ((value >> np.uint64(1)) & np.uint64(0x5555555555555555))
    value = ((value & np.uint64(0x3333333333333333))
             + ((value >> np.uint64(2)) & np.uint64(0x3333333333333333)))
    value = (value + (value >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((value * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def _hamming_distances(hard_calls, minimum_joint):
    """Missing-aware Hamming distance of every record pair: mismatches over jointly called
    sites, 1 when fewer than ``minimum_joint`` sites are called in both (bit-packed)."""
    n_records, n_sites = hard_calls.shape
    n_words = (n_sites + 63) // 64
    known_bits = np.zeros((n_records, n_words), dtype=np.uint64)
    allele_bits = np.zeros((n_records, n_words), dtype=np.uint64)
    for record in range(n_records):
        for site in range(n_sites):
            value = hard_calls[record, site]
            if value != _UNCALLED:
                word = site >> 6
                bit = np.uint64(1) << np.uint64(site & 63)
                known_bits[record, word] |= bit
                if value == 1:
                    allele_bits[record, word] |= bit
    distance = np.zeros((n_records, n_records), dtype=np.float64)
    for first in range(n_records - 1):
        for second in range(first + 1, n_records):
            n_joint = 0
            n_mismatch = 0
            for word in range(n_words):
                joint = known_bits[first, word] & known_bits[second, word]
                mismatch = (allele_bits[first, word] ^ allele_bits[second, word]) & joint
                n_joint += _popcount(joint)
                n_mismatch += _popcount(mismatch)
            value = 1.0 if n_joint < minimum_joint else n_mismatch / n_joint
            distance[first, second] = value
            distance[second, first] = value
    return distance


def _cluster(distance, maximum_hamming):
    """Cohesive groups of residual records: HDBSCAN's clusters within the diameter bound, then
    complete linkage (cut at the bound) on the remaining records. Returns a tuple of member
    tuples (HDBSCAN clusters by label, then complete-link groups by label)."""
    n_records = distance.shape[0]
    if n_records < 2:
        return ()
    if n_records == 2:
        pair_distance = float(distance[0, 1])
        return ((0, 1),) if pair_distance <= maximum_hamming + 1e-12 else ()
    labels = exact_hdbscan_labels(distance, min_cluster_size=2, min_samples=1, allow_single_cluster=True)
    clusters = []
    noise = set(np.flatnonzero(labels == -1).tolist())
    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        members = tuple(np.flatnonzero(labels == label).tolist())
        if len(members) < 2:
            noise.update(members)
        elif float(np.max(distance[np.ix_(members, members)])) <= maximum_hamming + 1e-12:
            clusters.append(members)
        else:
            # HDBSCAN is a density algorithm, not a complete-link guarantee.
            noise.update(members)
    remaining = tuple(sorted(noise))
    if len(remaining) >= 2:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        hierarchy = linkage(squareform(distance[np.ix_(remaining, remaining)], checks=False),
                            method="complete")
        group_labels = fcluster(hierarchy, t=float(maximum_hamming), criterion="distance")
        for group_label in sorted(np.unique(group_labels)):
            local = np.flatnonzero(group_labels == group_label)
            if len(local) < 2:
                continue
            members = tuple(remaining[int(index)] for index in local)
            if float(np.max(distance[np.ix_(members, members)])) <= maximum_hamming + 1e-12:
                clusters.append(members)
    return tuple(clusters)


def _cluster_cached(distance, maximum_hamming, cache):
    value = np.ascontiguousarray(distance)
    key = (float(maximum_hamming), value.shape, value.tobytes())
    clusters = cache.get(key)
    if clusters is None:
        clusters = cache[key] = _cluster(distance, maximum_hamming)
    return clusters


@njit(cache=True)
def _pooled_log_odds(members, sample_indices, weights, compatible, log_odds):
    """Per site, the sum over samples of min(1, w_s) x (the sample's responsibility-weighted
    mean log odds over its records), with the number of contributing samples. ``members`` are
    ordered by sample."""
    n_sites = log_odds.shape[1]
    total = np.zeros(n_sites, dtype=np.float64)
    contributing = np.zeros(n_sites, dtype=np.int32)
    weighted = np.zeros(n_sites, dtype=np.float64)
    weight_at_site = np.zeros(n_sites, dtype=np.float64)
    if members.shape[0] == 0:
        return total, contributing
    current = sample_indices[members[0]]
    for position in range(members.shape[0]):
        record = members[position]
        sample = sample_indices[record]
        if sample != current:
            for site in range(n_sites):
                weight = weight_at_site[site]
                if weight > 0.0:
                    sample_weight = 1.0 if weight > 1.0 else weight
                    total[site] += sample_weight * (weighted[site] / weight)
                    contributing[site] += 1
                    weighted[site] = 0.0
                    weight_at_site[site] = 0.0
            current = sample
        record_weight = weights[record]
        for site in range(n_sites):
            if compatible[record, site]:
                weighted[site] += record_weight * log_odds[record, site]
                weight_at_site[site] += record_weight
    for site in range(n_sites):
        weight = weight_at_site[site]
        if weight > 0.0:
            sample_weight = 1.0 if weight > 1.0 else weight
            total[site] += sample_weight * (weighted[site] / weight)
            contributing[site] += 1
    return total, contributing


def _consensus(members, sample_indices, weights, compatible, log_odds, call_probability):
    """(ALT probability per site, fraction of sites called at ``call_probability``)."""
    total, contributing = _pooled_log_odds(
        np.ascontiguousarray(members, dtype=np.int64), sample_indices, weights, compatible,
        log_odds)
    observed = contributing > 0
    posterior = _logistic_where(total, observed)
    confident = observed & ((posterior >= call_probability) | (posterior <= 1.0 - call_probability))
    return np.ascontiguousarray(posterior), float(np.mean(confident))


@njit(cache=True)
def _closest(candidate, existing, minimum_joint, n_sites, known_probability):
    """Nearest existing row by missing-aware Hamming distance over sites called in both
    (rounded ALT probabilities beyond ``known_probability``); ties within 1e-12 prefer the
    larger joint-called fraction. Returns (found, distance, joint fraction, coverage of the
    candidate's calls, coverage of the row's calls)."""
    lower = 1.0 - known_probability + 1e-12
    upper = known_probability - 1e-12
    candidate_known = np.zeros(n_sites, dtype=np.bool_)
    candidate_calls = np.empty(n_sites, dtype=np.float64)
    n_candidate_known = 0
    for site in range(n_sites):
        value = candidate[site]
        known = value <= lower or value >= upper
        candidate_known[site] = known
        if known:
            candidate_calls[site] = np.rint(value)
            n_candidate_known += 1
    found = False
    closest_distance = 0.0
    closest_joint = 0.0
    closest_candidate_coverage = 0.0
    closest_other_coverage = 0.0
    for other in range(existing.shape[0]):
        n_other_known = 0
        n_joint = 0
        n_mismatch = 0
        for site in range(n_sites):
            other_value = existing[other, site]
            if not (other_value <= lower or other_value >= upper):
                continue
            n_other_known += 1
            if candidate_known[site]:
                n_joint += 1
                if candidate_calls[site] != np.rint(other_value):
                    n_mismatch += 1
        if n_joint < minimum_joint:
            continue
        distance = n_mismatch / n_joint
        joint_fraction = n_joint / n_sites
        if (not found or distance < closest_distance - 1e-12
                or (abs(distance - closest_distance) <= 1e-12 and joint_fraction > closest_joint)):
            found = True
            closest_distance = distance
            closest_joint = joint_fraction
            closest_candidate_coverage = n_joint / max(1, n_candidate_known)
            closest_other_coverage = n_joint / max(1, n_other_known)
    return (found, closest_distance, closest_joint, closest_candidate_coverage,
            closest_other_coverage)


def _hardened_key(candidate, fallback) -> bytes:
    """The emission-order key of a soft candidate: its hardened 0/1 row as bytes (> 1/2 is ALT;
    within the probability quantum of 1/2 the ``fallback`` row decides, as moves.half_band_rows)."""
    return np.ascontiguousarray(half_band_rows(candidate, fallback), dtype=np.uint8).tobytes()


def soft_candidates(haplotypes, inputs: ResidualInputs, config, counter=None) -> np.ndarray:
    """float64[C, L] soft birth candidates of a complete binary panel, in emission order
    (``config``: ResidualBirthConfig; the block's sites are all kept; ``counter``: optional TieCounter of the
    band decisions, kinds discovery.residual.half_band and discovery.residual.mass_band)."""
    rows = np.ascontiguousarray(haplotypes, dtype=np.int64)
    k, n_sites = rows.shape
    pair_i, pair_j = np.triu_indices(k)
    pair_i = np.ascontiguousarray(pair_i, dtype=np.int64)
    pair_j = np.ascontiguousarray(pair_j, dtype=np.int64)
    responsibility = _diplotype_responsibility(inputs.log_likelihood, rows, pair_i, pair_j)
    (sample_indices, subtracted, partners, weights, soft_alt, hard_calls,
     compatible, log_odds) = _residual_records(
        responsibility, pair_i, pair_j, k, inputs.likelihood, inputs.observed, rows,
        float(config.residual_call_probability), float(config.min_soft_responsibility),
        float(PARTNER_BAND), float(MASS_BAND))
    n_records = len(sample_indices)
    # Sources: global clusters, then clusters within each (subtracted row, partner) route.
    sources = []
    distance = None
    if n_records >= 2:
        minimum_joint = max(1, int(math.ceil(config.min_residual_joint_known_fraction * n_sites)))
        distance = _hamming_distances(np.ascontiguousarray(hard_calls), minimum_joint)
        sources.extend(_cluster_cached(distance, config.max_cluster_hamming,
                                       inputs.cluster_cache))
    routes: dict[tuple[int, int], list[int]] = {}
    for index in range(n_records):
        routes.setdefault((int(subtracted[index]), int(partners[index])), []).append(index)
    for route in sorted(routes):
        pool = tuple(sorted(routes[route]))
        if len(pool) < 2:
            continue
        sub_distance = distance[np.ix_(pool, pool)]
        for local in _cluster_cached(sub_distance, config.max_cluster_hamming,
                                     inputs.cluster_cache):
            sources.append(tuple(pool[index] for index in local))
    if not sources:
        return np.empty((0, n_sites), dtype=np.float64)
    fallback = hardening_fallback(inputs.likelihood)
    log_odds = np.ascontiguousarray(log_odds)
    # One consensus per distinct member set, in emission order (more distinct supporting samples,
    # hardened row, source order).
    prepared = []
    consensus_by_members: dict[tuple, tuple] = {}
    for source_index, members in enumerate(sources):
        ordered = tuple(sorted(members, key=lambda index: (
            sample_indices[index], subtracted[index], partners[index], index)))
        cached = consensus_by_members.get(ordered)
        if cached is None:
            by_sample: dict[int, float] = {}
            for index in ordered:
                sample = int(sample_indices[index])
                by_sample[sample] = min(1.0, by_sample.get(sample, 0.0) + float(weights[index]))
            candidate, known_fraction = _consensus(
                ordered, sample_indices, weights, compatible, log_odds,
                float(config.candidate_call_probability))
            cached = consensus_by_members[ordered] = (
                _hardened_key(candidate, fallback), len(by_sample), math.fsum(by_sample.values()),
                candidate, known_fraction)
            note_band(counter, "discovery.residual.half_band", candidate, 0.5, HALF_BAND)
            note_band(counter, "discovery.residual.mass_band", [cached[2]],
                      float(config.min_effective_sample_support), MASS_BAND)
        prepared.append((cached, source_index))
    prepared.sort(key=lambda item: (-item[0][1], item[0][0], item[1]))
    existing = [np.asarray(row, dtype=np.float64) for row in rows]
    emitted = []
    minimum_dedup_joint = max(1, int(math.ceil(config.min_dedup_joint_known_fraction * n_sites)))
    for (digest, unique_support, effective_support, candidate, known_fraction), _ in prepared:
        if unique_support < config.min_unique_sample_support:
            continue
        if effective_support + MASS_BAND < config.min_effective_sample_support:
            continue
        if known_fraction < config.min_candidate_known_fraction:
            continue
        found, closest_distance, _joint, candidate_coverage, other_coverage = _closest(
            candidate, np.ascontiguousarray(np.stack(existing)), minimum_dedup_joint, n_sites,
            float(config.candidate_call_probability))
        if (found and closest_distance <= config.dedup_hamming_fraction + 1e-12
                and candidate_coverage >= config.min_dedup_bidirectional_coverage
                and other_coverage >= config.min_dedup_bidirectional_coverage):
            continue
        emitted.append(candidate)
        existing.append(candidate)
    if not emitted:
        return np.empty((0, n_sites), dtype=np.float64)
    return np.ascontiguousarray(np.stack(emitted))
