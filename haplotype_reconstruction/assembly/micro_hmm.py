"""Quadratic sum-product scans for a block-local normal/error-tract model.

The quality indicator follows a two-state continuous-distance Markov process.
Ancestry transitions apply in both quality states. Each block initializes
quality at stationarity and marginalizes it at the opposite boundary; the
backward scan is the adjoint of that same operator, not a reversed heuristic.
"""
import math
import numpy as np
from numba import njit, prange
from . import micro_hmm_log
from ..core import config as core_config

# Numerical range guards, not confidence thresholds.
_MIN_SCALED_MASS = 1e-250
_MAX_LOG_RANGE = 500.0


@njit(cache=True)
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
            same = (1.0 - theta)**2
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
    return np.ascontiguousarray(np.column_stack((1-enter, enter, leave, 1-leave)))


@njit(cache=True, parallel=True)
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


@njit(cache=True, inline='always')
def _row_exclusions(values, output):
    # Prefix/suffix sums avoid cancellation from subtracting a dominant state.
    for row in range(values.shape[0]):
        prefix = 0.0
        for col in range(values.shape[1]):
            output[row, col] = prefix
            prefix += values[row, col]
        suffix = 0.0
        for col in range(values.shape[1] - 1, -1, -1):
            output[row, col] += suffix
            suffix += values[row, col]


@njit(cache=True, parallel=True)
def _scaled_scan(weights, error_weights, invalid_emission, stay, switch, quality,
                 priors, haps, backward, emission_indices, error_fraction):
    samples, sites, _ = weights.shape
    result = np.full((samples, haps * haps), -np.inf)
    unsafe = invalid_emission.copy()
    for sample in prange(samples):
        if unsafe[sample]:
            continue
        normal = np.empty((haps, haps))
        error = np.empty_like(normal)
        next_normal = np.empty_like(normal)
        next_error = np.empty_like(normal)
        excluded_normal = np.empty_like(normal)
        excluded_error = np.empty_like(normal)
        prior_max, prior_min = -np.inf, np.inf
        for a in range(haps):
            for b in range(a, haps):
                prior = priors[sample, a * haps + b]
                if not np.isfinite(prior):
                    unsafe[sample] = True
                prior_max = max(prior_max, prior)
                prior_min = min(prior_min, prior)
        if unsafe[sample] or prior_max - prior_min > _MAX_LOG_RANGE:
            unsafe[sample] = True
            continue
        first = sites - 1 if backward else 0
        scale = 0.0
        k = 0
        for a in range(haps):
            for b in range(a, haps):
                prior = math.exp(priors[sample, a * haps + b] - prior_max)
                n = prior * weights[sample, first, emission_indices[first, k]]
                e = prior * error_weights[sample, first]
                if not backward:
                    n *= 1.0 - error_fraction
                    e *= error_fraction
                normal[a, b] = normal[b, a] = n
                error[a, b] = error[b, a] = e
                scale = max(scale, n, e)
                k += 1
        if scale <= 0.0 or not np.isfinite(scale):
            unsafe[sample] = True
            continue
        log_scale = prior_max + math.log(scale)
        for a in range(haps):
            for b in range(haps):
                normal[a, b] /= scale
                error[a, b] /= scale
                if normal[a, b] < _MIN_SCALED_MASS or (
                        error_fraction > 0.0 and error[a, b] < _MIN_SCALED_MASS):
                    unsafe[sample] = True
        for step in range(1, sites):
            if unsafe[sample]:
                break
            site = sites - 1 - step if backward else step
            interval = site if backward else site - 1
            _row_exclusions(normal, excluded_normal)
            _row_exclusions(error, excluded_error)
            nn, ne, en, ee = quality[interval]
            scale = 0.0
            k = 0
            for a in range(haps):
                for b in range(a, haps):
                    hn = stay[interval] * normal[a, b] + switch[interval] * (
                        excluded_normal[a, b] + excluded_normal[b, a])
                    he = stay[interval] * error[a, b] + switch[interval] * (
                        excluded_error[a, b] + excluded_error[b, a])
                    if backward:
                        n = nn * hn + ne * he
                        e = en * hn + ee * he
                    else:
                        n = nn * hn + en * he
                        e = ne * hn + ee * he
                    n *= weights[sample, site, emission_indices[site, k]]
                    e *= error_weights[sample, site]
                    next_normal[a, b] = next_normal[b, a] = n
                    next_error[a, b] = next_error[b, a] = e
                    scale = max(scale, n, e)
                    k += 1
            if scale <= 0.0 or not np.isfinite(scale):
                unsafe[sample] = True
                break
            log_scale += math.log(scale)
            for a in range(haps):
                for b in range(haps):
                    next_normal[a, b] /= scale
                    next_error[a, b] /= scale
                    if next_normal[a, b] < _MIN_SCALED_MASS or (
                            error_fraction > 0.0 and next_error[a, b] < _MIN_SCALED_MASS):
                        unsafe[sample] = True
            normal, next_normal = next_normal, normal
            error, next_error = next_error, error
        if not unsafe[sample]:
            for a in range(haps):
                for b in range(a, haps):
                    value = ((1-error_fraction)*normal[a, b] + error_fraction*error[a, b]
                             if backward else normal[a, b] + error[a, b])
                    score = math.log(value) + log_scale
                    result[sample, a*haps+b] = result[sample, b*haps+a] = score
    return result, unsafe


class PreparedBlockScans:
    """Fixed block data and cached zero-prior scores for one transition mesh.

    Compact input starts with three genotype log weights and a pair lookup. Its error
    emission averages those three weights: an independent uniformly drawn
    genotype, not an average over the represented founders. For dense diagnostic
    inputs supply error_log_emissions in the same per-site likelihood scale;
    the default 1/3 assumes normalized genotype likelihoods.
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
        upper = np.array([a*haps+b for a in range(haps) for b in range(a,haps)])
        if dosages is None:
            self.weights, self.invalid_emission = _emission_weights(tensor, self.haps)
            self.emission_indices = np.broadcast_to(
                np.arange(len(upper)), (len(positions),len(upper)))
            self.log_emission_indices = np.broadcast_to(upper, self.emission_indices.shape)
        else:
            self.emission_indices = np.ascontiguousarray(dosages[:,upper])
            self.log_emission_indices = self.emission_indices
            values = tensor.astype(np.float64)
            invalid = ~np.isfinite(values) | (np.abs(values) > _MAX_LOG_RANGE)
            self.invalid_emission = np.any(invalid, axis=(1,2))
            values[invalid] = 0.0
            self.weights = np.exp(values)
        if error_log_emissions is None:
            if dosages is None:
                error_log_emissions = np.full(tensor.shape[:2], -math.log(3.0))
            else:
                # Log-sum-exp retains the original common observation scale,
                # including for seven-category partial-founder inputs. Only the
                # first three entries are genotypes, not all predictive categories.
                error_log_emissions = np.logaddexp.reduce(
                    tensor[:, :, :3].astype(np.float64), axis=2) - math.log(3.0)
        self.error_log_emissions = np.ascontiguousarray(error_log_emissions)
        self.error_weights = np.exp(self.error_log_emissions)
        self.invalid_emission |= np.any(
            ~np.isfinite(self.error_weights) | (self.error_weights <= 0), axis=1)
        self.stay, self.switch = _transition_weights(positions,self.rate,self.haps,distances)
        self.quality = _quality_transitions(positions,self.error_fraction,tract_bp)
        self.zero_priors = np.zeros((tensor.shape[0],self.haps*self.haps))
        self.fixed_scores = [None,None]
        self.scan_calls = self.cached_calls = self.fallback_samples = 0

    def scan(self, priors=None, *, backward=False):
        direction = int(backward)
        if priors is not None and np.all(np.isfinite(priors[:,0])) and np.all(
                priors == priors[:,:1]):
            return self.scan(backward=backward) + priors[:,:1]
        if priors is None and self.fixed_scores[direction] is not None:
            self.cached_calls += 1
            return self.fixed_scores[direction]
        incoming = self.zero_priors if priors is None else priors
        result, unsafe = _scaled_scan(
            self.weights,self.error_weights,self.invalid_emission,self.stay,self.switch,
            self.quality,incoming,self.haps,backward,self.emission_indices,self.error_fraction)
        self.scan_calls += 1
        count = int(np.count_nonzero(unsafe))
        self.fallback_samples += count
        if count:
            selected = slice(None) if count == len(unsafe) else unsafe
            result[unsafe] = micro_hmm_log.scan_sum_product(
                self.tensor[selected],self.error_log_emissions[selected],self.stay,self.switch,
                self.quality,incoming[selected],self.haps,backward,
                self.log_emission_indices,self.error_fraction)
        if priors is None:
            self.fixed_scores[direction] = result
        return result


def scan_distance_aware_forward(tensor, positions, rate, definitions, priors, haps,
                                genetic_distances=None, **model_options):
    return PreparedBlockScans(
        tensor,positions,rate,definitions,haps,genetic_distances,**model_options).scan(priors)


def scan_distance_aware_backward(tensor, positions, rate, definitions, priors, haps,
                                 genetic_distances=None, **model_options):
    return PreparedBlockScans(
        tensor,positions,rate,definitions,haps,genetic_distances,**model_options).scan(
            priors,backward=True)
