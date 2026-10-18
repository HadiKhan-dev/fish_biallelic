"""Within-block sum-product scans of the linker's normal/error-tract model (base micro_hmm + micro_hmm_log).

Each sample is a diploid pair of founder copies inside one block. Between adjacent emission sites at most one
of the two copies switches founder (relative zero/one-switch weights of a per-meiosis recombination fraction,
normalised to row mass 1). A two-state quality indicator follows a continuous-distance Markov process: in the
normal state the pair emits through its predictive class (``linker.emissions``), in the error state a
uniformly drawn genotype (the mean of the three genotype likelihoods); ancestry transitions apply in both
states. Each block starts the quality indicator at stationarity and marginalises it at the opposite end; the
backward scan is the adjoint of the same operator (not a reversed heuristic).

A scan maps a log prior over ordered founder pairs (S x K^2) at the block's first (forward) or last (backward)
site to the log likelihood of the block's data and the pair at the opposite end, for every sample. The scaled
scan works on 32-sample lanes (vectorised across samples) with O(K^2) row-exclusion sums per site; a sample
whose numbers leave the safe range is rescored by the log-domain scan of the same model.
"""
from __future__ import annotations

import math
import threading

import numpy as np
from numba import njit, prange

# Numerical range guards, not confidence thresholds.
_MIN_SCALED_MASS = 1e-250
_MAX_LOG_RANGE = 500.0
# Samples per lane block of the scans: a work split only, results do not depend on it.
_LANE_WIDTH = 32
# Site x unordered-pair cells of one lane block from which a scan that shares the budget with a concurrent
# recursion re-reads its thread share between rounds of lane blocks (about 10 ms of work per lane block; a
# smaller scan runs in one parallel call, where an extra launch would cost more than it could gain).
_MIN_ROUND_CELLS = 50_000


@njit(cache=True, nogil=True)
def _transition_weights(positions, rate, founders, morgans):
    """Stay and per-other-founder switch weights of each interval between adjacent emission sites.

    One homologue switches between retained markers at most (the approximation conditions on zero or one
    switch). theta = max(1, distance bp) x rate, or the interval's genetic distance (Morgans) when a map is
    given, capped at 0.5; the weights (1 - theta)^2 and theta (1 - theta) / (K - 1) are normalised to row
    mass 1.
    """
    stay = np.empty(len(positions) - 1)
    switch = np.empty_like(stay)
    for site in range(len(stay)):
        theta = max(1, positions[site + 1] - positions[site]) * rate
        if morgans is not None:
            theta = morgans[site]
        theta = min(theta, 0.5)
        if theta < 1e-15 or founders == 1:
            stay[site], switch[site] = 1.0, 0.0
        else:
            same = (1.0 - theta) ** 2
            other = theta * (1.0 - theta) / (founders - 1)
            mass = same + 2.0 * (founders - 1) * other
            stay[site], switch[site] = same / mass, other / mass
    return stay, switch


def _quality_transitions(positions, fraction, tract_bp):
    """float64[L - 1, 4] exact two-state CTMC transitions (normal->normal, normal->error, error->normal,
    error->error) of each physical interval: exit intensity 1/L_tract, entry intensity p / ((1 - p) L_tract),
    so the stationary error fraction is p and the mean error tract is L_tract bp."""
    distances = np.diff(np.asarray(positions, dtype=np.float64))
    moved = -np.expm1(-distances / ((1.0 - fraction) * tract_bp))
    enter = fraction * moved
    leave = (1.0 - fraction) * moved
    return np.ascontiguousarray(np.column_stack((1 - enter, enter, leave, 1 - leave)))


@njit(cache=True, nogil=True)
def _to_lanes(weights, error_weights, width):
    """Lane copies wl[block, site, class, lane] and ewl[block, site, lane] of a block's emission weights;
    padding lanes are never read."""
    samples, sites, classes = weights.shape
    blocks = (samples + width - 1) // width
    wl = np.ones((blocks, sites, classes, width))
    ewl = np.ones((blocks, sites, width))
    for sample in range(samples):
        block, lane = sample // width, sample % width
        for site in range(sites):
            for f in range(classes):
                wl[block, site, f, lane] = weights[sample, site, f]
            ewl[block, site, lane] = error_weights[sample, site]
    return wl, ewl


@njit(cache=True, inline='always')
def _pair_index(a, b, founders):
    """Upper-triangle index of the unordered founder pair (min(a, b), max(a, b))."""
    if a > b:
        a, b = b, a
    return a * founders - a * (a - 1) // 2 + (b - a)


@njit(cache=True, error_model='numpy')
def _scan_lane_block(block, lanes, wl, ewl, invalid_emission, stay, switch, quality, priors, founders, backward,
                     emission_indices, error_fraction, result, unsafe, normal, error, nxt_normal, nxt_error,
                     ex_normal, ex_error, vec, log_scale):
    """Scaled sum-product scan of the first ``lanes`` samples of one lane block.

    Every inner loop runs over the block's lanes (vectorised across samples); per sample the operations,
    operands and their order do not depend on its lane. The normal and error state matrices are symmetric:
    each keeps its upper triangle (pair order), so a normalisation divides each unordered cell once. A sample
    is unsafe (the log-domain scan rescores it) on an invalid emission, a non-finite prior, a prior range above
    _MAX_LOG_RANGE, a non-positive or non-finite scale, or a scaled mass below _MIN_SCALED_MASS. An unsafe lane
    keeps computing values that are never read; logarithms are taken only while a lane is safe. vec rows:
    0 scale, 1 prefix, 2 suffix, 3 largest prior.
    """
    sites = wl.shape[1]
    pairs = founders * (founders + 1) // 2
    base = block * wl.shape[3]
    for j in range(lanes):
        sample = base + j
        flag = invalid_emission[sample]
        prior_max, prior_min = -np.inf, np.inf
        if not flag:
            for a in range(founders):
                for b in range(a, founders):
                    prior = priors[sample, a * founders + b]
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
    for a in range(founders):
        for b in range(a, founders):
            f = emission_indices[first, k]
            for j in range(lanes):
                sample = base + j
                prior = math.exp(priors[sample, a * founders + b] - vec[3, j]) if not unsafe[sample] else 0.0
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
        # Row exclusions of both symmetric matrices, ex[row, col] = sum of the row's other entries as a
        # prefix plus a suffix sum (no cancellation from subtracting a dominant state); M[row, col] is read
        # as M[min, max].
        for row in range(founders):
            for j in range(lanes):
                vec[1, j] = 0.0
                vec[2, j] = 0.0
            for col in range(founders):
                p = _pair_index(row, col, founders)
                for j in range(lanes):
                    ex_normal[row, col, j] = vec[1, j]
                    vec[1, j] += normal[p, j]
            for col in range(founders - 1, -1, -1):
                p = _pair_index(row, col, founders)
                for j in range(lanes):
                    ex_normal[row, col, j] += vec[2, j]
                    vec[2, j] += normal[p, j]
            for j in range(lanes):
                vec[1, j] = 0.0
                vec[2, j] = 0.0
            for col in range(founders):
                p = _pair_index(row, col, founders)
                for j in range(lanes):
                    ex_error[row, col, j] = vec[1, j]
                    vec[1, j] += error[p, j]
            for col in range(founders - 1, -1, -1):
                p = _pair_index(row, col, founders)
                for j in range(lanes):
                    ex_error[row, col, j] += vec[2, j]
                    vec[2, j] += error[p, j]
        nn, ne, en, ee = quality[interval]
        st, sw = stay[interval], switch[interval]
        for j in range(lanes):
            vec[0, j] = 0.0
        k = 0
        for a in range(founders):
            for b in range(a, founders):
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
            for a in range(founders):
                for b in range(a, founders):
                    value = ((1 - error_fraction) * normal[k, j] + error_fraction * error[k, j]
                             if backward else normal[k, j] + error[k, j])
                    score = math.log(value) + log_scale[j]
                    result[sample, a * founders + b] = result[sample, b * founders + a] = score
                    k += 1


@njit(cache=True, parallel=True, error_model='numpy', nogil=True)
def _scaled_scan(first, stop, wl, ewl, invalid_emission, stay, switch, quality, priors, founders, backward,
                 emission_indices, error_fraction, result, unsafe):
    """Scan scores (rows of unsafe samples stay as they are) and unsafe flags of the samples of lane blocks
    first .. stop - 1, written into ``result`` and ``unsafe``; the lane blocks run in parallel."""
    width = wl.shape[3]
    samples = result.shape[0]
    pairs = founders * (founders + 1) // 2
    for block_index in prange(first, stop):
        block = np.int64(block_index)
        lanes = min(width, samples - block * width)
        normal, error = np.empty((pairs, width)), np.empty((pairs, width))
        nxt_normal, nxt_error = np.empty((pairs, width)), np.empty((pairs, width))
        ex_normal, ex_error = np.empty((founders, founders, width)), np.empty((founders, founders, width))
        vec, log_scale = np.empty((4, width)), np.empty(width)
        _scan_lane_block(block, lanes, wl, ewl, invalid_emission, stay, switch, quality, priors, founders,
                         backward, emission_indices, error_fraction, result, unsafe, normal, error, nxt_normal,
                         nxt_error, ex_normal, ex_error, vec, log_scale)


@njit(cache=True)
def _row_exclusions(values, out):
    """out[row, col] = log sum of the row's other entries (prefix + suffix log sums, no subtraction)."""
    for row in range(values.shape[0]):
        prefix = -np.inf
        for col in range(values.shape[1]):
            out[row, col] = prefix
            prefix = np.logaddexp(prefix, values[row, col])
        suffix = -np.inf
        for col in range(values.shape[1] - 1, -1, -1):
            out[row, col] = np.logaddexp(out[row, col], suffix)
            suffix = np.logaddexp(suffix, values[row, col])


@njit(cache=True, parallel=True, nogil=True)
def _log_scan(log_emissions, error_logs, stay, switch, quality, priors, founders, backward, emission_indices,
              error_fraction):
    """The same scan in the log domain (the numerical fallback for unsafe samples, not a different model);
    prefix/suffix log sums exclude a homologue without subtracting a dominant term or cubic loops."""
    samples, sites, _ = log_emissions.shape
    output = np.full((samples, founders * founders), -np.inf)
    log_stay, log_switch, log_quality = np.log(stay), np.log(switch), np.log(quality)
    lp = math.log(error_fraction) if error_fraction > 0 else -np.inf
    ln = math.log1p(-error_fraction)
    for sample in prange(samples):
        normal = np.empty((founders, founders))
        error = np.empty_like(normal)
        next_normal = np.empty_like(normal)
        next_error = np.empty_like(normal)
        xn = np.empty_like(normal)
        xe = np.empty_like(normal)
        first = sites - 1 if backward else 0
        k = 0
        for a in range(founders):
            for b in range(a, founders):
                prior = priors[sample, a * founders + b]
                n = prior + np.float64(log_emissions[sample, first, emission_indices[first, k]])
                e = prior + error_logs[sample, first]
                if not backward:
                    n += ln
                    e += lp
                normal[a, b] = normal[b, a] = n
                error[a, b] = error[b, a] = e
                k += 1
        for step in range(1, sites):
            site = sites - 1 - step if backward else step
            interval = site if backward else site - 1
            _row_exclusions(normal, xn)
            _row_exclusions(error, xe)
            nn, ne, en, ee = log_quality[interval]
            k = 0
            for a in range(founders):
                for b in range(a, founders):
                    hn = np.logaddexp(log_stay[interval] + normal[a, b],
                                      log_switch[interval] + np.logaddexp(xn[a, b], xn[b, a]))
                    he = np.logaddexp(log_stay[interval] + error[a, b],
                                      log_switch[interval] + np.logaddexp(xe[a, b], xe[b, a]))
                    if backward:
                        n = np.logaddexp(nn + hn, ne + he)
                        e = np.logaddexp(en + hn, ee + he)
                    else:
                        n = np.logaddexp(nn + hn, en + he)
                        e = np.logaddexp(ne + hn, ee + he)
                    n += np.float64(log_emissions[sample, site, emission_indices[site, k]])
                    e += error_logs[sample, site]
                    next_normal[a, b] = next_normal[b, a] = n
                    next_error[a, b] = next_error[b, a] = e
                    k += 1
            normal, next_normal = next_normal, normal
            error, next_error = next_error, error
        for a in range(founders):
            for b in range(a, founders):
                value = (np.logaddexp(normal[a, b] + ln, error[a, b] + lp) if backward
                         else np.logaddexp(normal[a, b], error[a, b]))
                output[sample, a * founders + b] = output[sample, b * founders + a] = value
    return output


class BlockScans:
    """One block's fixed scan data and its cached zero-prior scores, shared by every gap fit of a batch.

    Built once per block from the block's log emissions (float32[S, L, C], C = 3 or 7 classes), the pair
    codes (uint8[L, K * K]), the emission sites' positions (int64[L], L >= 1) and either a per-bp rate or the
    intervals' genetic distances. The error emission of a sample at a site is the mean of its three genotype
    likelihoods (log-sum-exp of the first three classes minus log 3): an independent uniformly drawn genotype.

    ``scan(priors, backward=..., threads=...)`` returns float64[S, K * K] scores. A prior that is the same for
    every pair of a sample (finite) is the zero-prior scan plus that constant; the zero-prior scan of each
    direction is computed once (under the direction's lock) and shared by the concurrent gap fits and
    recursions. Returned arrays are read-only by convention (the cached one is shared). ``threads``: None runs
    all lane blocks in one parallel call on the calling thread's Numba team; a callable (which sizes the team
    and returns its size) is re-read between rounds of that many lane blocks when a lane block is large
    (_MIN_ROUND_CELLS), so a scan running beside another takes up threads its peer frees (lane blocks are
    independent: results do not change).
    """

    def __init__(self, log_emissions, codes, positions, founders, rate, morgans, error_fraction, error_tract_bp):
        if log_emissions.shape[1] == 0:
            raise ValueError("a linker block needs at least one emission site")
        self.founders = int(founders)
        self.error_fraction = float(error_fraction)
        self.log_emissions = log_emissions
        upper = np.array([a * self.founders + b for a in range(self.founders) for b in range(a, self.founders)])
        self.emission_indices = np.ascontiguousarray(codes[:, upper])
        values = log_emissions.astype(np.float64)
        invalid = ~np.isfinite(values) | (np.abs(values) > _MAX_LOG_RANGE)
        invalid_emission = np.any(invalid, axis=(1, 2))
        values[invalid] = 0.0
        weights = np.exp(values)
        del values, invalid
        # The log-sum-exp keeps the common observation scale (only the first three classes are genotypes).
        self.error_logs = np.ascontiguousarray(
            np.logaddexp.reduce(log_emissions[:, :, :3].astype(np.float64), axis=2) - math.log(3.0))
        error_weights = np.exp(self.error_logs)
        invalid_emission |= np.any(~np.isfinite(error_weights) | (error_weights <= 0), axis=1)
        self.invalid_emission = invalid_emission
        # The scans read the weights in lane layout (samples side by side).
        self.lane_weights, self.lane_error_weights = _to_lanes(
            np.ascontiguousarray(weights), error_weights, _LANE_WIDTH)
        del weights, error_weights
        self.stay, self.switch = _transition_weights(positions, float(rate), self.founders, morgans)
        self.quality = _quality_transitions(positions, self.error_fraction, float(error_tract_bp))
        # Rounds of lane blocks are worth it only for scans with a large lane block (see _MIN_ROUND_CELLS).
        self._rounds = len(positions) * (self.founders * (self.founders + 1) // 2) >= _MIN_ROUND_CELLS
        self._fixed = [None, None]
        self._locks = (threading.Lock(), threading.Lock())

    @property
    def samples(self) -> int:
        return int(self.invalid_emission.shape[0])

    def scan(self, priors=None, *, backward=False, threads=None):
        """float64[S, K * K] log scores of the block given log pair priors at its entry (None = zero priors)."""
        if priors is not None and np.all(np.isfinite(priors[:, 0])) and np.all(priors == priors[:, :1]):
            return self.scan(backward=backward, threads=threads) + priors[:, :1]
        if priors is None:
            direction = int(backward)
            with self._locks[direction]:
                if self._fixed[direction] is None:
                    self._fixed[direction] = self._scan(
                        np.zeros((self.samples, self.founders * self.founders)), backward, threads)
                return self._fixed[direction]
        return self._scan(priors, backward, threads)

    def _scan(self, priors, backward, threads=None):
        result = np.full((self.samples, self.founders * self.founders), -np.inf)
        unsafe = np.empty(self.samples, dtype=np.bool_)
        blocks = self.lane_weights.shape[0]
        first = 0
        rounds = threads is not None and self._rounds
        while first < blocks:
            stop = min(blocks, first + max(1, int(threads()))) if rounds else blocks
            _scaled_scan(first, stop, self.lane_weights, self.lane_error_weights, self.invalid_emission, self.stay,
                         self.switch, self.quality, priors, self.founders, backward, self.emission_indices,
                         self.error_fraction, result, unsafe)
            first = stop
        count = int(np.count_nonzero(unsafe))
        if count:
            selected = slice(None) if count == len(unsafe) else unsafe
            result[unsafe] = _log_scan(
                self.log_emissions[selected], self.error_logs[selected], self.stay, self.switch, self.quality,
                priors[selected], self.founders, backward, self.emission_indices, self.error_fraction)
        return result
