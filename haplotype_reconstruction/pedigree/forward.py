"""M0 / M1 / M2 forward likelihoods of a child's genotype likelihoods (projected quadratic model).

The hidden state of a child at a bin is the ordered pair of transmitted founder states (S^2 states):
under M0 both come from the null source (a copying chain that starts from, and redraws from, the
leave-one-out frequency of the founder states the other samples transmit), under M1 the
first comes from the candidate parent's projected transmission (pi, bridges), under M2 both come from
the two parents. Emissions multiply the child's per-marker
likelihoods of the two transmitted alleles (with mismatch probability) over its observed markers
of the bin. Each score pools two views with equal weight: tempered (bin emissions raised to the
information exponent) and full strength, run in one pass that shares the emission products. Their
average is a composite score, not two independent data sets or a calibrated posterior.

Called founder alleles project to the same 0/1 value in every candidate, so their emission products
are cached per child and bin (NaN marks entries needing the candidate-specific path) in batches of
children bounded by a 256 MiB workspace.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .projection import Projection

# Temporary per-call cache of shared emission products, tiled over children (not a setting).
COMMON_EMISSION_CACHE_BYTES = 256 << 20
_SMALLEST_NORMAL = 2.2250738585072014e-308


@njit(cache=True, inline="always")
def _site_likelihood(gl, coefficient, first_alt, second_alt, mismatch):
    value = coefficient[0] + coefficient[1] * (first_alt + second_alt) + coefficient[2] * first_alt * second_alt
    if value > 1e-8:
        return value
    first = mismatch + (1.0 - 2.0 * mismatch) * first_alt
    second = mismatch + (1.0 - 2.0 * mismatch) * second_alt
    p0 = (1.0 - first) * (1.0 - second)
    p2 = first * second
    p1 = 1.0 - p0 - p2
    return 3.0 * (gl[0] * p0 + gl[1] * p1 + gl[2] * p2)


def child_coefficients(gl: np.ndarray, mismatch: float) -> np.ndarray:
    """float64 [N, M, 3]: L(a, b) = c0 + c1 (a + b) + c2 a b for transmitted ALT probabilities a, b."""
    g0, g1, g2 = gl[:, :, 0], gl[:, :, 1], gl[:, :, 2]
    linear = g1 - g0
    quadratic = g0 - 2.0 * g1 + g2
    inherited = 1.0 - 2.0 * mismatch
    coefficient = np.empty(gl.shape, dtype=np.float64)
    coefficient[:, :, 0] = 3.0 * (g0 + 2.0 * mismatch * linear + mismatch * mismatch * quadratic)
    coefficient[:, :, 1] = 3.0 * inherited * (linear + mismatch * quadratic)
    coefficient[:, :, 2] = 3.0 * inherited * inherited * quadratic
    return coefficient


@njit(cache=True, parallel=True)
def observed_sites(observed, bin_start, bin_stop):
    """CSR of each child's observed markers per bin: (sites, start [N, B], stop [N, B])."""
    children = observed.shape[0]
    bins = len(bin_start)
    start = np.empty((children, bins), dtype=np.int64)
    stop = np.empty((children, bins), dtype=np.int64)
    for task in prange(children * bins):
        child = task // bins
        block = task - child * bins
        count = 0
        for site in range(int(bin_start[block]), int(bin_stop[block])):
            count += int(observed[child, site])
        stop[child, block] = count
    cursor = 0
    for child in range(children):
        for block in range(bins):
            count = stop[child, block]
            start[child, block] = cursor
            cursor += count
            stop[child, block] = cursor
    sites = np.empty(cursor, dtype=np.int64)
    for child in prange(children):
        for block in range(bins):
            destination = start[child, block]
            for site in range(int(bin_start[block]), int(bin_stop[block])):
                if observed[child, site]:
                    sites[destination] = site
                    destination += 1
    return sites, start, stop


@njit(cache=True, parallel=True)
def _common_products(state_alt, child_gl, coefficient, site_index, site_start, site_stop,
                     mismatch, child_start, child_stop):
    """Emission products of called-founder state pairs per child and bin; NaN = candidate-specific.

    Missing named alleles and BACKGROUND keep candidate-specific emissions; the site order,
    likelihood and underflow fallback of the general path stay authoritative.
    """
    bins = site_start.shape[1]
    states = state_alt.shape[0]
    children = child_stop - child_start
    result = np.full((children, bins, states, states), np.nan)
    for task in prange(children * bins):
        local_child, block = task // bins, task % bins
        child = child_start + local_child
        for first in range(states - 1):
            for second in range(states - 1):
                product = 1.0
                valid = True
                for slot in range(site_start[child, block], site_stop[child, block]):
                    site = site_index[slot]
                    a, b = state_alt[first, site], state_alt[second, site]
                    if not ((a == 0.0 or a == 1.0) and (b == 0.0 or b == 1.0)):
                        valid = False
                        break
                    value = _site_likelihood(child_gl[child, site], coefficient[child, site], a, b,
                                             mismatch)
                    if value <= 0.0:
                        valid = False
                        break
                    product *= value
                    if not np.isfinite(product) or product < _SMALLEST_NORMAL:
                        valid = False
                        break
                if valid:
                    result[local_child, block, first, second] = product
    return result


@njit(cache=True, parallel=True)
def _powered_products(products, exponent, child_start):
    """The cached products raised to each child's fractional bin exponent (NaN otherwise)."""
    powered = np.full(products.shape, np.nan)
    children, bins, states, _ = products.shape
    for task in prange(children * bins):
        child, block = task // bins, task % bins
        eta = exponent[child_start + child, block]
        if 0.0 < eta < 1.0:
            for first in range(states):
                for second in range(states):
                    value = products[child, block, first, second]
                    if np.isfinite(value) and value > 0.0:
                        result = math.pow(value, eta)
                        if np.isfinite(result) and result > 0.0:
                            powered[child, block, first, second] = result
    return powered


@njit(cache=True)
def _normalise(values):
    total = np.sum(values)
    if total <= 0.0 or not np.isfinite(total):
        return -np.inf
    values /= total
    return math.log(total)


@njit(cache=True)
def _apply_probability_emission(current, emission, exponent, powered=None):
    maximum = np.max(emission)
    if maximum <= 0.0 or not np.isfinite(maximum):
        return -np.inf
    inverse = 1.0 / maximum
    states = current.shape[0]
    log_maximum = math.log(maximum)
    common_scale = 0.0
    if powered is not None and abs(exponent * log_maximum) < 300.0:
        common_scale = math.pow(maximum, exponent)
    if exponent == 1.0:
        for first in range(states):
            for second in range(states):
                current[first, second] *= emission[first, second] * inverse
    else:
        for first in range(states):
            for second in range(states):
                base = emission[first, second] * inverse
                saved = np.nan if powered is None else powered[first, second]
                # Keep the underflow path and arbitrary exponents; ordinary fractional powers share
                # their numerator through the cache.
                if common_scale > 0.0 and np.isfinite(saved) and base >= _SMALLEST_NORMAL:
                    value = saved / common_scale
                else:
                    value = math.pow(base, exponent)
                current[first, second] *= value
    increment = _normalise(current)
    return increment + exponent * log_maximum if np.isfinite(increment) else -np.inf


@njit(cache=True)
def _apply_log_emission(current, log_emission):
    maximum = np.max(log_emission)
    if not np.isfinite(maximum):
        return -np.inf
    states = current.shape[0]
    for first in range(states):
        for second in range(states):
            current[first, second] *= math.exp(log_emission[first, second] - maximum)
    increment = _normalise(current)
    return increment + maximum if np.isfinite(increment) else -np.inf


@njit(cache=True)
def _apply_star_bridge(current, output, stay, alpha, row_arm, pivot, axis):
    """Exact Hall-star support: only the pivot has a nonzero beta; its two sums keep the general
    operator's ascending prefix and descending suffix order."""
    states = current.shape[0]
    if axis == 0:
        for second in range(states):
            left = 0.0
            for source in range(pivot):
                left += current[source, second] * alpha[source]
            right = 0.0
            for source in range(states - 1, pivot, -1):
                right += current[source, second] * alpha[source]
            pivot_value = current[pivot, second]
            for destination in range(states):
                value = stay[destination] * current[destination, second]
                if destination == pivot:
                    value += left + right
                value += pivot_value * row_arm[destination]
                output[destination, second] = value
    else:
        for first in range(states):
            left = 0.0
            for source in range(pivot):
                left += current[first, source] * alpha[source]
            right = 0.0
            for source in range(states - 1, pivot, -1):
                right += current[first, source] * alpha[source]
            pivot_value = current[first, pivot]
            for destination in range(states):
                value = stay[destination] * current[first, destination]
                if destination == pivot:
                    value += left + right
                value += pivot_value * row_arm[destination]
                output[first, destination] = value


@njit(cache=True)
def _apply_bridge(current, output, stay, alpha, beta, row_arm, column_arm, pivot, axis, prefix,
                  suffix, branch):
    """One compressed bridge on one selector axis with cancellation-free leave-one-out sums."""
    if branch == 3:
        _apply_star_bridge(current, output, stay, alpha, row_arm, pivot, axis)
        return
    states = current.shape[0]
    if axis == 0:
        for second in range(states):
            prefix[0] = 0.0
            for source in range(states):
                prefix[source + 1] = prefix[source] + current[source, second] * alpha[source]
            suffix[states] = 0.0
            for source in range(states - 1, -1, -1):
                suffix[source] = suffix[source + 1] + current[source, second] * alpha[source]
            column_arm_total = 0.0
            if pivot >= 0:
                for source in range(states):
                    column_arm_total += current[source, second] * column_arm[source]
            for destination in range(states):
                value = (stay[destination] * current[destination, second]
                         + beta[destination] * (prefix[destination] + suffix[destination + 1]))
                if pivot >= 0:
                    value += current[pivot, second] * row_arm[destination]
                    if destination == pivot:
                        value += column_arm_total
                output[destination, second] = value
    else:
        for first in range(states):
            prefix[0] = 0.0
            for source in range(states):
                prefix[source + 1] = prefix[source] + current[first, source] * alpha[source]
            suffix[states] = 0.0
            for source in range(states - 1, -1, -1):
                suffix[source] = suffix[source + 1] + current[first, source] * alpha[source]
            column_arm_total = 0.0
            if pivot >= 0:
                for source in range(states):
                    column_arm_total += current[first, source] * column_arm[source]
            for destination in range(states):
                value = (stay[destination] * current[first, destination]
                         + beta[destination] * (prefix[destination] + suffix[destination + 1]))
                if pivot >= 0:
                    value += current[first, pivot] * row_arm[destination]
                    if destination == pivot:
                        value += column_arm_total
                output[first, destination] = value


@njit(cache=True)
def _apply_excursion(current, output, probability, axis):
    """Complete T' = (1 - k) T + k U after a bridge mapped ``current`` to ``output``."""
    states = current.shape[0]
    keep = 1.0 - probability
    if axis == 0:
        for second in range(states):
            total = 0.0
            for source in range(states):
                total += current[source, second]
            share = probability * total / states
            for destination in range(states):
                output[destination, second] = keep * output[destination, second] + share
    else:
        for first in range(states):
            total = 0.0
            for source in range(states):
                total += current[first, source]
            share = probability * total / states
            for destination in range(states):
                output[first, destination] = keep * output[first, destination] + share


@njit(cache=True)
def _apply_null(current, output, diagonal, off, frequency, axis):
    """One null-slot step: stay with weight diagonal - off, else draw the destination from ``frequency``.

    With uniform frequencies this is the symmetric kernel diagonal / off (rows sum to diagonal + (S - 1) off).
    """
    states = current.shape[0]
    coefficient = diagonal - off
    move = states * off
    if axis == 0:
        for second in range(states):
            total = 0.0
            for source in range(states):
                total += current[source, second]
            for destination in range(states):
                output[destination, second] = (coefficient * current[destination, second]
                                               + move * frequency[destination] * total)
    else:
        for first in range(states):
            total = 0.0
            for source in range(states):
                total += current[first, source]
            for destination in range(states):
                output[first, destination] = (coefficient * current[first, destination]
                                              + move * frequency[destination] * total)


@njit(cache=True)
def _advance(current, work, child, first_parent, second_parent, mode, boundary, stay, alpha, beta,
             row_arm, column_arm, pivot, excursion, branch, null_diagonal, null_off, null_frequency, prefix,
             suffix):
    """Cross one bin edge on both selector axes; the result returns to ``current``."""
    if mode >= 1:
        _apply_bridge(current, work, stay[first_parent, boundary], alpha[first_parent, boundary],
                      beta[first_parent, boundary], row_arm[first_parent, boundary],
                      column_arm[first_parent, boundary], pivot[first_parent, boundary], 0, prefix,
                      suffix, branch[first_parent, boundary])
        probability = excursion[first_parent, boundary]
        if probability > 0.0:
            _apply_excursion(current, work, probability, 0)
    else:
        _apply_null(current, work, null_diagonal[child, boundary], null_off[child, boundary],
                    null_frequency[child, boundary + 1], 0)
    if mode == 2:
        _apply_bridge(work, current, stay[second_parent, boundary], alpha[second_parent, boundary],
                      beta[second_parent, boundary], row_arm[second_parent, boundary],
                      column_arm[second_parent, boundary], pivot[second_parent, boundary], 1, prefix,
                      suffix, branch[second_parent, boundary])
        probability = excursion[second_parent, boundary]
        if probability > 0.0:
            _apply_excursion(work, current, probability, 1)
    else:
        _apply_null(work, current, null_diagonal[child, boundary], null_off[child, boundary],
                    null_frequency[child, boundary + 1], 1)


@njit(cache=True)
def _score_path(child, first_parent, second_parent, mode, pi, candidate_alt, stay, alpha, beta,
                row_arm, column_arm, pivot, excursion, branch, state_alt, child_gl, coefficient,
                site_index, site_start, site_stop, exponent, null_diagonal, null_off, null_frequency,
                mismatch, common_products, first_cached_child, common_powered):
    """(tempered, full) forward log likelihood of one child under one configuration.

    The views keep their own forward state, scaling and underflow path and each stops at its first
    non-finite increment; they share the observed-marker emission products (or their logarithms),
    which the full view applies with exponent one.
    """
    states = state_alt.shape[0]
    bins = exponent.shape[1]
    tempered = np.empty((states, states), dtype=np.float64)
    full = np.empty_like(tempered)
    work = np.empty_like(tempered)
    scratch = np.empty_like(tempered)
    prefix = np.empty(states + 1, dtype=np.float64)
    suffix = np.empty(states + 1, dtype=np.float64)
    for first in range(states):
        for second in range(states):
            if mode == 0:
                tempered[first, second] = null_frequency[child, 0, first] * null_frequency[child, 0, second]
            elif mode == 1:
                tempered[first, second] = pi[first_parent, 0, first] * null_frequency[child, 0, second]
            else:
                tempered[first, second] = pi[first_parent, 0, first] * pi[second_parent, 0, second]
            full[first, second] = tempered[first, second]
    tempered_log = 0.0
    full_log = 0.0
    tempered_alive = True
    full_alive = True
    for block in range(bins):
        if block > 0:
            boundary = block - 1
            if tempered_alive:
                _advance(tempered, scratch, child, first_parent, second_parent, mode, boundary,
                         stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
                         null_diagonal, null_off, null_frequency, prefix, suffix)
            if full_alive:
                _advance(full, scratch, child, first_parent, second_parent, mode, boundary,
                         stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
                         null_diagonal, null_off, null_frequency, prefix, suffix)

        stable_product = True
        for first in range(states):
            for second in range(states):
                if common_products is not None:
                    saved = common_products[child - first_cached_child, block, first, second]
                    if np.isfinite(saved):
                        work[first, second] = saved
                        continue
                product = 1.0
                for slot in range(site_start[child, block], site_stop[child, block]):
                    site = site_index[slot]
                    first_alt = (candidate_alt[first_parent, site, first] if mode >= 1
                                 else state_alt[first, site])
                    second_alt = (candidate_alt[second_parent, site, second] if mode == 2
                                  else state_alt[second, site])
                    value = _site_likelihood(child_gl[child, site], coefficient[child, site],
                                             first_alt, second_alt, mismatch)
                    if value <= 0.0:
                        product = 0.0
                        break
                    product *= value
                    if not np.isfinite(product) or product < _SMALLEST_NORMAL:
                        stable_product = False
                        break
                work[first, second] = product
                if not stable_product:
                    break
            if not stable_product:
                break
        if not stable_product:
            for first in range(states):
                for second in range(states):
                    log_emission = 0.0
                    for slot in range(site_start[child, block], site_stop[child, block]):
                        site = site_index[slot]
                        first_alt = (candidate_alt[first_parent, site, first] if mode >= 1
                                     else state_alt[first, site])
                        second_alt = (candidate_alt[second_parent, site, second] if mode == 2
                                      else state_alt[second, site])
                        value = _site_likelihood(child_gl[child, site], coefficient[child, site],
                                                 first_alt, second_alt, mismatch)
                        if value <= 0.0:
                            log_emission = -np.inf
                            break
                        log_emission += math.log(value)
                    work[first, second] = log_emission

        if tempered_alive:
            eta = exponent[child, block]
            powered = (None if common_powered is None
                       else common_powered[child - first_cached_child, block])
            if eta == 0.0:
                # A zero exponent makes the tempered emission uniform.
                scratch[:] = 1.0
                increment = _apply_probability_emission(tempered, scratch, eta, powered)
            elif stable_product:
                increment = _apply_probability_emission(tempered, work, eta, powered)
            else:
                for first in range(states):
                    for second in range(states):
                        scratch[first, second] = eta * work[first, second]
                increment = _apply_log_emission(tempered, scratch)
            if np.isfinite(increment):
                tempered_log += increment
            else:
                tempered_alive = False
                tempered_log = -np.inf
        if full_alive:
            if stable_product:
                increment = _apply_probability_emission(full, work, 1.0, None)
            else:
                increment = _apply_log_emission(full, work)
            if np.isfinite(increment):
                full_log += increment
            else:
                full_alive = False
                full_log = -np.inf
        if not tempered_alive and not full_alive:
            break
    return tempered_log, full_log


@njit(cache=True, parallel=True)
def _m0_kernel(pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
               state_alt, child_gl, coefficient, site_index, site_start, site_stop, exponent,
               null_diagonal, null_off, null_frequency, mismatch, eligible_child):
    tempered = np.empty(child_gl.shape[0], dtype=np.float64)
    full = np.empty(child_gl.shape[0], dtype=np.float64)
    for child in prange(child_gl.shape[0]):
        if eligible_child[child]:
            tempered[child], full[child] = _score_path(
                child, 0, 0, 0, pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot,
                excursion, branch, state_alt, child_gl, coefficient, site_index, site_start,
                site_stop, exponent, null_diagonal, null_off, null_frequency, mismatch, None, 0, None)
        else:
            tempered[child] = -np.inf
            full[child] = -np.inf
    return tempered, full


@njit(cache=True, parallel=True)
def _m1_kernel(pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
               available, state_alt, child_gl, coefficient, site_index, site_start, site_stop,
               exponent, null_diagonal, null_off, null_frequency, mismatch, eligible_edge, tempered_m0, full_m0,
               common_products, child_start, child_stop, common_powered):
    children = child_stop - child_start
    candidates = pi.shape[0]
    tempered = np.empty((children, candidates), dtype=np.float64)
    full = np.empty((children, candidates), dtype=np.float64)
    for task in prange(children * candidates):
        local_child = task // candidates
        child = child_start + local_child
        parent = task % candidates
        if not eligible_edge[child, parent]:
            tempered[local_child, parent] = -np.inf
            full[local_child, parent] = -np.inf
        elif not available[parent]:
            tempered[local_child, parent] = tempered_m0[child]
            full[local_child, parent] = full_m0[child]
        else:
            tempered[local_child, parent], full[local_child, parent] = _score_path(
                child, parent, 0, 1, pi, candidate_alt, stay, alpha, beta, row_arm, column_arm,
                pivot, excursion, branch, state_alt, child_gl, coefficient, site_index, site_start,
                site_stop, exponent, null_diagonal, null_off, null_frequency, mismatch, common_products,
                child_start, common_powered)
    return tempered, full


@njit(cache=True, parallel=True)
def _m2_kernel(trios, pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot, excursion,
               branch, state_alt, child_gl, coefficient, site_index, site_start, site_stop, exponent,
               null_diagonal, null_off, null_frequency, mismatch, common_products, first_cached_child,
               common_powered):
    tempered = np.empty(len(trios), dtype=np.float64)
    full = np.empty(len(trios), dtype=np.float64)
    for row in prange(len(trios)):
        tempered[row], full[row] = _score_path(
            trios[row, 0], trios[row, 1], trios[row, 2], 2, pi, candidate_alt, stay, alpha, beta,
            row_arm, column_arm, pivot, excursion, branch, state_alt, child_gl, coefficient,
            site_index, site_start, site_stop, exponent, null_diagonal, null_off, null_frequency, mismatch,
            common_products, first_cached_child, common_powered)
    return tempered, full


def null_frequencies(pi: np.ndarray, available: np.ndarray, prior_samples: float) -> np.ndarray:
    """float64 [N, B, S]: per child, the leave-one-out transmitted-state frequency of the other informative
    samples with a uniform pseudo-count regulariser of ``prior_samples`` samples (see ChildTerms)."""
    states = pi.shape[2]
    weight = np.asarray(available, dtype=np.float64)
    total = np.einsum("n,nbs->bs", weight, pi)
    own = weight[:, None, None] * pi
    counts = np.maximum(total[None] - own, 0.0) + prior_samples / states
    return np.ascontiguousarray(counts / np.sum(counts, axis=2, keepdims=True))


class ChildTerms:
    """Child-side scoring inputs of one component, shared by the M0/M1 screen and the M2 score.

    gl float64 [N, M, 3], coefficient [N, M, 3], observed-marker CSR (site_index, site_start,
    site_stop), exponent float64 [N, B], null transition coefficients [N, B-1] (diagonal, off),
    null_frequency float64 [N, B, S] (the null source's initial and redraw law per child and bin), mismatch, the
    cache batch size and whether any exponent is fractional.

    The null source of a child's unobserved parent slot starts from, and redraws from, the frequency at which the
    other samples transmit each founder state, f_cb(s) = (sum_{j != c} a_j pi_jb(s) + w / S) /
    (sum_{j != c} a_j + w), over the samples j with an informative painting site (a_j = 1), with a uniform
    pseudo-count of w = ``null_prior_samples`` samples (a regulariser): an empirical sample-frequency null at
    founder-state resolution. Across a bin edge the slot keeps its state with the weight keep = diagonal - off of the
    painting's single-homologue copying kernel mixed with the meiotic selector, and otherwise draws a state from
    f_c,b+1 (with uniform frequencies, exactly the symmetric diagonal / off kernel). Its marginal therefore follows
    m_c0 = f_c0, m_c,b+1 = keep m_cb + (1 - keep) f_c,b+1, not f_cb at every bin. Leaving the child out removes its
    own row from the average, not its influence on the shared paintings.
    """

    def __init__(self, projection: Projection, gl: np.ndarray, observed: np.ndarray,
                 exponent: np.ndarray, selector_switch_probability: np.ndarray, same: np.ndarray,
                 one: np.ndarray, two: np.ndarray, mismatch: float, null_prior_samples: float):
        children = gl.shape[0]
        states = projection.n_states
        bins = projection.n_bins
        self.gl = gl
        self.coefficient = child_coefficients(gl, mismatch)
        self.site_index, self.site_start, self.site_stop = observed_sites(
            observed, projection.bin_start, projection.bin_stop)
        self.exponent = exponent
        self.mismatch = float(mismatch)
        selector = np.broadcast_to(selector_switch_probability, (children, bins - 1))
        self.null_diagonal = np.ascontiguousarray(
            (1.0 - selector) * (same[None, :] + (states - 1) * one[None, :]) + selector / states)
        self.null_off = np.ascontiguousarray(
            (1.0 - selector) * (one[None, :] + (states - 1) * two[None, :]) + selector / states)
        # Persistence weight keep = diagonal - off >= 0: a saturated kernel (theoretically diagonal == off: no
        # persistence) can round a few ULP below zero; that rounding is clipped to zero persistence (diagonal = off =
        # 1 / S, rows summing to one), a materially negative keep is an error. The certificate needs keep >= 0.
        keep = self.null_diagonal - self.null_off
        rounding = 64.0 * np.finfo(np.float64).eps * states * np.maximum(self.null_diagonal, self.null_off)
        if np.any(keep < -rounding):
            raise ValueError("null source kernel prefers a specific other state to staying")
        saturated = keep < 0.0
        if np.any(saturated):
            self.null_diagonal = np.ascontiguousarray(np.where(saturated, 1.0 / states, self.null_diagonal))
            self.null_off = np.ascontiguousarray(np.where(saturated, 1.0 / states, self.null_off))
        if not null_prior_samples > 0.0:
            raise ValueError("null_prior_samples must be positive")
        self.null_frequency = null_frequencies(projection.pi, projection.available, float(null_prior_samples))
        self.fractional = bool(np.any((exponent > 0.0) & (exponent < 1.0)))
        cache_children = COMMON_EMISSION_CACHE_BYTES // max(1, bins * states * states * 16)
        self.cached = cache_children > 0
        self.batch = min(children, cache_children) if cache_children else children

    def common(self, projection: Projection, child_start: int, child_stop: int):
        """(cached products, powered products) of a child batch, or (None, None) without a cache."""
        if not self.cached:
            return None, None
        products = _common_products(projection.state_alt, self.gl, self.coefficient,
                                    self.site_index, self.site_start, self.site_stop,
                                    self.mismatch, child_start, child_stop)
        powered = (_powered_products(products, self.exponent, child_start)
                   if self.fractional else None)
        return products, powered


def _model(projection: Projection) -> tuple:
    return (projection.pi, projection.candidate_alt, projection.stay, projection.alpha,
            projection.beta, projection.row_arm, projection.column_arm, projection.pivot,
            projection.excursion, projection.branch)


def screen(projection: Projection, terms: ChildTerms, eligible_children: np.ndarray,
           eligible_parents: np.ndarray, claim_threads=lambda: None) -> tuple[np.ndarray, np.ndarray]:
    """Pooled M0 [N] and all-pairs M1 [N, N] log likelihoods (-inf off the eligible universe)."""
    children = terms.gl.shape[0]
    candidates = projection.pi.shape[0]
    eligible_edge = np.ascontiguousarray(eligible_parents & eligible_children[:, None])
    common_args = (projection.state_alt, terms.gl, terms.coefficient, terms.site_index,
                   terms.site_start, terms.site_stop, terms.exponent, terms.null_diagonal,
                   terms.null_off, terms.null_frequency, terms.mismatch)
    claim_threads()
    zero = _m0_kernel(*_model(projection), *common_args, np.ascontiguousarray(eligible_children))
    one = tuple(np.empty((children, candidates), dtype=np.float64) for _ in range(2))
    for child_start in range(0, children, terms.batch):
        claim_threads()
        child_stop = min(children, child_start + terms.batch)
        products, powered = terms.common(projection, child_start, child_stop)
        claim_threads()
        one[0][child_start:child_stop], one[1][child_start:child_stop] = _m1_kernel(
            *_model(projection), projection.available, *common_args, eligible_edge, *zero,
            products, child_start, child_stop, powered)
        del products, powered
    return 0.5 * (zero[0] + zero[1]), 0.5 * (one[0] + one[1])


def score_pairs(projection: Projection, terms: ChildTerms, trios: np.ndarray, zero: np.ndarray,
                one: np.ndarray, claim_threads=lambda: None) -> np.ndarray:
    """Pooled M2 log likelihoods [T] of the panel rows; a row with an evidence-free parent reduces
    to the other parent's M1 (or M0) screen value."""
    children = terms.gl.shape[0]
    first_available = projection.available[trios[:, 1]]
    second_available = projection.available[trios[:, 2]]
    active = first_available & second_available
    two = np.empty(len(trios), dtype=np.float64)
    for row in np.flatnonzero(~active):
        child, first, second = trios[row]
        two[row] = (one[child, first] if first_available[row] else
                    one[child, second] if second_available[row] else zero[child])
    active_rows = np.flatnonzero(active)
    if not len(active_rows):
        return two
    active_children = trios[active_rows, 0]
    common_args = (projection.state_alt, terms.gl, terms.coefficient, terms.site_index,
                   terms.site_start, terms.site_stop, terms.exponent, terms.null_diagonal,
                   terms.null_off, terms.null_frequency, terms.mismatch)
    for child_start in range(0, children, terms.batch):
        child_stop = min(children, child_start + terms.batch)
        batch_rows = active_rows[(active_children >= child_start) & (active_children < child_stop)]
        if not len(batch_rows):
            continue
        claim_threads()
        products, powered = terms.common(projection, child_start, child_stop)
        claim_threads()
        tempered, full = _m2_kernel(np.ascontiguousarray(trios[batch_rows]), *_model(projection),
                                    *common_args, products, child_start, powered)
        two[batch_rows] = 0.5 * (tempered + full)
        del products, powered
    return two
