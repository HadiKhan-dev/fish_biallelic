"""Quadratic sum-product scans for a block-local normal/error-tract model.

The quality indicator follows a two-state continuous-distance Markov process.
Ancestry transitions apply in both quality states. Each block initializes
quality at stationarity and marginalizes it at the opposite boundary; the
backward scan is the adjoint of that same operator, not a reversed heuristic.
"""
import math
import threading
import numpy as np
from numba import njit, prange
from . import micro_hmm_log
from ..core import config as core_config

# Numerical range guards, not confidence thresholds.
_MIN_SCALED_MASS = 1e-250
_MAX_LOG_RANGE = 500.0
# Samples per lane block of the scans: a work split only, results do not depend on it.
_SCAN_LANE_WIDTH = 32


@njit(cache=True, nogil=True)
def _transition_weights(positions, rate, haps, genetic_distances):
    """Preserve relative zero/single-switch weights, normalized to row mass 1.

    The existing approximation conditions on at most one homologue switching
    between retained markers. Double-switch and rate-model changes are separate.
    """
    stay = np.empty(len(positions) - 1)
    switch = np.empty_like(stay)
    for site in range(len(stay)):
        theta = max(1, positions[site + 1] - positions[site]) * rate
        if genetic_distances is not None:
            theta = genetic_distances[site]
        theta = min(theta, 0.5)
        if theta < 1e-15 or haps == 1:
            stay[site], switch[site] = 1.0, 0.0
        else:
            same = (1.0 - theta) ** 2
            other = theta * (1.0 - theta) / (haps - 1)
            mass = same + 2.0 * (haps - 1) * other
            stay[site], switch[site] = same / mass, other / mass
    return stay, switch


def _quality_transitions(positions, fraction, tract_bp):
    """Exact two-state CTMC transition probabilities at each physical interval.

    Exit intensity is 1/L; entry intensity is p/((1-p)*L). This yields stationary
    error fraction p and mean uninterrupted error-tract length L base pairs.
    """
    distances = np.diff(np.asarray(positions, dtype=np.float64))
    moved = -np.expm1(-distances / ((1.0 - fraction) * tract_bp))
    enter = fraction * moved
    leave = (1.0 - fraction) * moved
    return np.ascontiguousarray(np.column_stack((1 - enter, enter, leave, 1 - leave)))


@njit(cache=True, parallel=True, nogil=True)
def _emission_weights(tensor, haps):
    samples, sites, _ = tensor.shape
    folded = haps * (haps + 1) // 2
    weights = np.empty((samples, sites, folded), dtype=np.float64)
    unsafe = np.zeros(samples, dtype=np.bool_)
    for sample in prange(samples):
        for site in range(sites):
            k = 0
            for a in range(haps):
                for b in range(a, haps):
                    value = np.float64(tensor[sample, site, a * haps + b])
                    if not np.isfinite(value) or abs(value) > _MAX_LOG_RANGE:
                        unsafe[sample] = True
                        value = 0.0
                    weights[sample, site, k] = math.exp(value)
                    k += 1
    return weights, unsafe


@njit(cache=True, nogil=True)
def _to_scan_lanes(weights, error_weights, width):
    """Lane copies wl[block, site, category, lane] and ewl[block, site, lane] of a
    block's emission weights; padding lanes are never read."""
    samples, sites, categories = weights.shape
    blocks = (samples + width - 1) // width
    wl = np.ones((blocks, sites, categories, width))
    ewl = np.ones((blocks, sites, width))
    for sample in range(samples):
        block, lane = sample // width, sample % width
        for site in range(sites):
            for f in range(categories):
                wl[block, site, f, lane] = weights[sample, site, f]
            ewl[block, site, lane] = error_weights[sample, site]
    return wl, ewl


@njit(cache=True, inline='always')
def _scan_pair(a, b, haps):
    """Upper-triangle index of the unordered homologue pair (min(a, b), max(a, b))."""
    if a > b:
        a, b = b, a
    return a * haps - a * (a - 1) // 2 + (b - a)


@njit(cache=True, error_model='numpy')
def _scan_block(block, lanes, wl, ewl, invalid_emission, stay, switch, quality, priors, haps, backward,
                emission_indices, error_fraction, result, unsafe, normal, error, nxt_normal, nxt_error,
                ex_normal, ex_error, vec, log_scale):
    """Scaled sum-product scan of the first ``lanes`` samples of one lane block.

    Every inner loop runs over the block's lanes (vectorized across samples);
    per sample the operations, operands and their order do not depend on its
    lane. The normal and error state matrices are symmetric: each keeps its
    upper triangle (pair order), so a normalisation divides each unordered
    cell once. A sample is unsafe (the log-domain fallback rescores it) on an
    invalid emission, a non-finite prior, a prior range above _MAX_LOG_RANGE,
    a non-positive or non-finite scale, or a scaled mass below
    _MIN_SCALED_MASS. An unsafe lane keeps computing values that are never
    read; logarithms are taken only while a lane is safe. vec rows: 0 scale,
    1 prefix, 2 suffix, 3 largest prior.
    """
    sites = wl.shape[1]
    pairs = haps * (haps + 1) // 2
    base = block * wl.shape[3]
    for j in range(lanes):
        sample = base + j
        flag = invalid_emission[sample]
        prior_max, prior_min = -np.inf, np.inf
        if not flag:
            for a in range(haps):
                for b in range(a, haps):
                    prior = priors[sample, a * haps + b]
                    if not np.isfinite(prior):
                        flag = True
                    prior_max = max(prior_max, prior)
                    prior_min = min(prior_min, prior)
            if flag or prior_max - prior_min > _MAX_LOG_RANGE:
                flag = True
        unsafe[sample] = flag
        vec[3, j] = prior_max
    first = sites - 1 if backward else 0
    for j in range(lanes):
        vec[0, j] = 0.0
    k = 0
    for a in range(haps):
        for b in range(a, haps):
            f = emission_indices[first, k]
            for j in range(lanes):
                sample = base + j
                prior = math.exp(priors[sample, a * haps + b] - vec[3, j]) if not unsafe[sample] else 0.0
                n = prior * wl[block, first, f, j]
                e = prior * ewl[block, first, j]
                if not backward:
                    n *= 1.0 - error_fraction
                    e *= error_fraction
                normal[k, j] = n
                error[k, j] = e
                # numba's max(scale, n, e): two selects.
                scale = vec[0, j]
                scale = n if n > scale else scale
                scale = e if e > scale else scale
                vec[0, j] = scale
            k += 1
    for j in range(lanes):
        sample = base + j
        scale = vec[0, j]
        if scale <= 0.0 or not np.isfinite(scale):
            unsafe[sample] = True
            log_scale[j] = 0.0
        elif not unsafe[sample]:
            log_scale[j] = vec[3, j] + math.log(scale)
    for p in range(pairs):
        for j in range(lanes):
            n = normal[p, j] / vec[0, j]
            e = error[p, j] / vec[0, j]
            normal[p, j] = n
            error[p, j] = e
            if n < _MIN_SCALED_MASS or (error_fraction > 0.0 and e < _MIN_SCALED_MASS):
                unsafe[base + j] = True
    for step in range(1, sites):
        site = sites - 1 - step if backward else step
        interval = site if backward else site - 1
        # Row exclusions of both symmetric matrices, ex[row, col] = sum of the
        # row's other entries as a prefix plus a suffix sum (no cancellation
        # from subtracting a dominant state); M[row, col] is read as M[min, max].
        for row in range(haps):
            for j in range(lanes):
                vec[1, j] = 0.0
                vec[2, j] = 0.0
            for col in range(haps):
                p = _scan_pair(row, col, haps)
                for j in range(lanes):
                    ex_normal[row, col, j] = vec[1, j]
                    vec[1, j] += normal[p, j]
            for col in range(haps - 1, -1, -1):
                p = _scan_pair(row, col, haps)
                for j in range(lanes):
                    ex_normal[row, col, j] += vec[2, j]
                    vec[2, j] += normal[p, j]
            for j in range(lanes):
                vec[1, j] = 0.0
                vec[2, j] = 0.0
            for col in range(haps):
                p = _scan_pair(row, col, haps)
                for j in range(lanes):
                    ex_error[row, col, j] = vec[1, j]
                    vec[1, j] += error[p, j]
            for col in range(haps - 1, -1, -1):
                p = _scan_pair(row, col, haps)
                for j in range(lanes):
                    ex_error[row, col, j] += vec[2, j]
                    vec[2, j] += error[p, j]
        nn, ne, en, ee = quality[interval]
        st, sw = stay[interval], switch[interval]
        for j in range(lanes):
            vec[0, j] = 0.0
        k = 0
        for a in range(haps):
            for b in range(a, haps):
                f = emission_indices[site, k]
                for j in range(lanes):
                    hn = st * normal[k, j] + sw * (ex_normal[a, b, j] + ex_normal[b, a, j])
                    he = st * error[k, j] + sw * (ex_error[a, b, j] + ex_error[b, a, j])
                    if backward:
                        n = nn * hn + ne * he
                        e = en * hn + ee * he
                    else:
                        n = nn * hn + en * he
                        e = ne * hn + ee * he
                    n *= wl[block, site, f, j]
                    e *= ewl[block, site, j]
                    nxt_normal[k, j] = n
                    nxt_error[k, j] = e
                    scale = vec[0, j]
                    scale = n if n > scale else scale
                    scale = e if e > scale else scale
                    vec[0, j] = scale
                k += 1
        for j in range(lanes):
            sample = base + j
            scale = vec[0, j]
            if scale <= 0.0 or not np.isfinite(scale):
                unsafe[sample] = True
            elif not unsafe[sample]:
                log_scale[j] += math.log(scale)
        for p in range(pairs):
            for j in range(lanes):
                n = nxt_normal[p, j] / vec[0, j]
                e = nxt_error[p, j] / vec[0, j]
                normal[p, j] = n
                error[p, j] = e
                if n < _MIN_SCALED_MASS or (error_fraction > 0.0 and e < _MIN_SCALED_MASS):
                    unsafe[base + j] = True
    for j in range(lanes):
        sample = base + j
        if not unsafe[sample]:
            k = 0
            for a in range(haps):
                for b in range(a, haps):
                    value = ((1 - error_fraction) * normal[k, j] + error_fraction * error[k, j]
                             if backward else normal[k, j] + error[k, j])
                    score = math.log(value) + log_scale[j]
                    result[sample, a * haps + b] = result[sample, b * haps + a] = score
                    k += 1


@njit(cache=True, parallel=True, error_model='numpy', nogil=True)
def _scaled_scan(wl, ewl, invalid_emission, stay, switch, quality, priors, haps, backward, emission_indices,
                 error_fraction, samples):
    """Every sample's scan scores (-inf rows for unsafe samples) and unsafe flags;
    lane blocks run in parallel."""
    blocks, width = wl.shape[0], wl.shape[3]
    pairs = haps * (haps + 1) // 2
    result = np.full((samples, haps * haps), -np.inf)
    unsafe = np.empty(samples, dtype=np.bool_)
    for block_index in prange(blocks):
        block = np.int64(block_index)
        lanes = min(width, samples - block * width)
        normal, error = np.empty((pairs, width)), np.empty((pairs, width))
        nxt_normal, nxt_error = np.empty((pairs, width)), np.empty((pairs, width))
        ex_normal, ex_error = np.empty((haps, haps, width)), np.empty((haps, haps, width))
        vec, log_scale = np.empty((4, width)), np.empty(width)
        _scan_block(block, lanes, wl, ewl, invalid_emission, stay, switch, quality, priors, haps, backward,
                    emission_indices, error_fraction, result, unsafe, normal, error, nxt_normal, nxt_error,
                    ex_normal, ex_error, vec, log_scale)
    return result, unsafe


class PreparedBlockScans:
    """Fixed block data and cached zero-prior scores for one transition mesh.

    Compact input starts with three genotype log weights and a pair lookup. Its error
    emission averages those three weights: an independent uniformly drawn
    genotype, not an average over the represented founders. For dense diagnostic
    inputs supply error_log_emissions in the same per-site likelihood scale;
    the default 1/3 assumes normalized genotype likelihoods.

    The mesh's concurrent gap fits and recursions share one instance per block:
    each zero-prior score is computed once (under its direction's lock) and the
    call counters stay exact.
    """
    def __init__(self, tensor, positions, rate, definitions, haps, distances=None,
                 dosages=None, *, error_fraction=None, error_tract_bp=None,
                 error_log_emissions=None):
        if tensor.shape[1] == 0:
            raise ValueError("a micro-HMM block needs at least one retained marker")
        self.error_fraction = (core_config.LINKER_ERROR_FRACTION if error_fraction is None
                               else float(error_fraction))
        tract_bp = core_config.LINKER_ERROR_TRACT_BP if error_tract_bp is None else float(error_tract_bp)
        if not 0 <= self.error_fraction < 1 or not np.isfinite(tract_bp) or tract_bp <= 0:
            raise ValueError("error fraction must be in [0,1), tract length finite and positive")
        self.tensor, self.positions, self.rate = tensor, positions, float(rate)
        self.definitions, self.haps, self.distances = definitions, int(haps), distances
        self.dosages = dosages
        upper = np.array([a * haps + b for a in range(haps) for b in range(a, haps)])
        if dosages is None:
            weights, self.invalid_emission = _emission_weights(tensor, self.haps)
            self.emission_indices = np.broadcast_to(
                np.arange(len(upper)), (len(positions), len(upper)))
            self.log_emission_indices = np.broadcast_to(upper, self.emission_indices.shape)
        else:
            self.emission_indices = np.ascontiguousarray(dosages[:, upper])
            self.log_emission_indices = self.emission_indices
            values = tensor.astype(np.float64)
            invalid = ~np.isfinite(values) | (np.abs(values) > _MAX_LOG_RANGE)
            self.invalid_emission = np.any(invalid, axis=(1, 2))
            values[invalid] = 0.0
            weights = np.exp(values)
        if error_log_emissions is None:
            if dosages is None:
                error_log_emissions = np.full(tensor.shape[:2], -math.log(3.0))
            else:
                # Log-sum-exp retains the original common observation scale,
                # including for seven-category partial-founder inputs. Only the
                # first three entries are genotypes, not all predictive categories.
                error_log_emissions = np.logaddexp.reduce(
                    tensor[:,:,:3].astype(np.float64), axis=2) - math.log(3.0)
        self.error_log_emissions = np.ascontiguousarray(error_log_emissions)
        error_weights = np.exp(self.error_log_emissions)
        self.invalid_emission |= np.any(
            ~np.isfinite(error_weights) | (error_weights <= 0), axis=1)
        # The scans read the weights in lane layout (samples side by side).
        self.lane_weights, self.lane_error_weights = _to_scan_lanes(
            np.ascontiguousarray(weights), error_weights, _SCAN_LANE_WIDTH)
        self.stay, self.switch = _transition_weights(positions, self.rate, self.haps, distances)
        self.quality = _quality_transitions(positions, self.error_fraction, tract_bp)
        self.zero_priors = np.zeros((tensor.shape[0], self.haps * self.haps))
        self.fixed_scores = [None, None]
        self.scan_calls = self.cached_calls = self.fallback_samples = 0
        self._fixed_locks = (threading.Lock(), threading.Lock())
        self._count_lock = threading.Lock()

    def scan(self, priors=None, *, backward=False):
        direction = int(backward)
        if priors is not None and np.all(np.isfinite(priors[:, 0])) and np.all(
                priors == priors[:,:1]):
            return self.scan(backward=backward) + priors[:,:1]
        if priors is None:
            with self._fixed_locks[direction]:
                if self.fixed_scores[direction] is None:
                    self.fixed_scores[direction] = self._scan(self.zero_priors, backward)
                else:
                    with self._count_lock:
                        self.cached_calls += 1
                return self.fixed_scores[direction]
        return self._scan(priors, backward)

    def _scan(self, incoming, backward):
        result, unsafe = _scaled_scan(
            self.lane_weights, self.lane_error_weights, self.invalid_emission, self.stay, self.switch,
            self.quality, incoming, self.haps, backward, self.emission_indices, self.error_fraction,
            len(self.invalid_emission))
        count = int(np.count_nonzero(unsafe))
        with self._count_lock:
            self.scan_calls += 1
            self.fallback_samples += count
        if count:
            selected = slice(None) if count == len(unsafe) else unsafe
            result[unsafe] = micro_hmm_log.scan_sum_product(
                self.tensor[selected], self.error_log_emissions[selected], self.stay, self.switch,
                self.quality, incoming[selected], self.haps, backward,
                self.log_emission_indices, self.error_fraction)
        return result
