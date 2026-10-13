"""Quadratic M0/M1/M2 transmission likelihood kernels."""

import math
import numpy as np
from numba import prange
from ..core import parallel

parallel.ensure_numba_registry_warmup()
njit = parallel.original_njit


@njit(cache=True, inline="always", fastmath=False)
def _site_likelihood(gl, coefficient, first_alt, second_alt, mismatch):
    value = (
        coefficient[0]
        + coefficient[1] * (first_alt + second_alt)
        + coefficient[2] * first_alt * second_alt
    )
    if value > 1e-8:
        return value
    first = mismatch + (1.0 - 2.0 * mismatch) * first_alt
    second = mismatch + (1.0 - 2.0 * mismatch) * second_alt
    p0 = (1.0 - first) * (1.0 - second)
    p2 = first * second
    p1 = 1.0 - p0 - p2
    return 3.0 * (gl[0] * p0 + gl[1] * p1 + gl[2] * p2)


@njit(cache=True, parallel=True, fastmath=False)
def _common_emission_products(
    state_alt, child_gl, coefficient, observed_site, observed_start,
    observed_stop, mismatch, child_start, child_stop,
):
    """Share exact called-founder products; NaN requests the general path.

    A called named allele projects to the same 0/1 value in every candidate.
    Missing named alleles and BACKGROUND retain candidate-specific emissions.
    The original site order, likelihood calculation and underflow fallback
    remain authoritative. This cache never removes an HMM state.
    """
    bins = observed_start.shape[1]
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
                for slot in range(
                    observed_start[child, block], observed_stop[child, block]
                ):
                    site = observed_site[slot]
                    a, b = state_alt[first, site], state_alt[second, site]
                    if not ((a == 0.0 or a == 1.0)
                            and (b == 0.0 or b == 1.0)):
                        valid = False
                        break
                    value = _site_likelihood(
                        child_gl[child, site], coefficient[child, site],
                        a, b, mismatch,
                    )
                    if value <= 0.0:
                        valid = False
                        break
                    product *= value
                    if (not np.isfinite(product)
                            or product < 2.2250738585072014e-308):
                        valid = False
                        break
                if valid:
                    result[local_child, block, first, second] = product
    return result


@njit(cache=True, parallel=True, fastmath=False)
def _power_common_products(products, exponent, child_start):
    """Reuse called-founder emission powers across the candidate panel."""
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


@njit(cache=True, fastmath=False)
def _normalise_forward(values):
    total = np.sum(values)
    if total <= 0.0 or not np.isfinite(total):
        return -np.inf
    values /= total
    return math.log(total)


@njit(cache=True, fastmath=False)
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
                # Preserve the original underflow path and arbitrary-exponent
                # fallback. Ordinary fractional powers share their numerator.
                if (common_scale > 0.0 and np.isfinite(saved)
                        and base >= 2.2250738585072014e-308):
                    value = saved / common_scale
                else:
                    value = math.pow(base, exponent)
                current[first, second] *= value
    increment = _normalise_forward(current)
    return (
        increment + exponent * log_maximum
        if np.isfinite(increment)
        else -np.inf
    )


@njit(cache=True, fastmath=False)
def _apply_log_emission(current, log_emission):
    maximum = np.max(log_emission)
    if not np.isfinite(maximum):
        return -np.inf
    states = current.shape[0]
    for first in range(states):
        for second in range(states):
            current[first, second] *= math.exp(
                log_emission[first, second] - maximum
            )
    increment = _normalise_forward(current)
    return increment + maximum if np.isfinite(increment) else -np.inf


@njit(cache=True, fastmath=False)
def _apply_star_bridge_axis(current, output, stay, alpha, row_arm, pivot, axis):
    """Apply exact Hall-star support without general leave-one-out arrays.

    Only the pivot has a nonzero beta. Its two sums retain the general
    operator's ascending prefix and descending suffix order. All other
    destinations receive their diagonal and the pivot's outgoing row arm.
    """
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


@njit(cache=True, fastmath=False)
def _apply_bridge_axis(
    current, output, stay, alpha, beta, row_arm, column_arm, large, axis,
    prefix, suffix, branch=-1,
):
    """Apply one compressed bridge with cancellation-free leave-one-out sums."""

    if branch == 3:
        _apply_star_bridge_axis(current, output, stay, alpha, row_arm, large, axis)
        return

    states = current.shape[0]
    if axis == 0:
        for second in range(states):
            prefix[0] = 0.0
            for source in range(states):
                prefix[source + 1] = (
                    prefix[source] + current[source, second] * alpha[source]
                )
            suffix[states] = 0.0
            for source in range(states - 1, -1, -1):
                suffix[source] = (
                    suffix[source + 1]
                    + current[source, second] * alpha[source]
                )
            column_arm_total = 0.0
            if large >= 0:
                for source in range(states):
                    column_arm_total += (
                        current[source, second] * column_arm[source]
                    )
            for destination in range(states):
                value = (
                    stay[destination] * current[destination, second]
                    + beta[destination]
                    * (prefix[destination] + suffix[destination + 1])
                )
                if large >= 0:
                    value += current[large, second] * row_arm[destination]
                    if destination == large:
                        value += column_arm_total
                output[destination, second] = value
    else:
        for first in range(states):
            prefix[0] = 0.0
            for source in range(states):
                prefix[source + 1] = (
                    prefix[source] + current[first, source] * alpha[source]
                )
            suffix[states] = 0.0
            for source in range(states - 1, -1, -1):
                suffix[source] = (
                    suffix[source + 1]
                    + current[first, source] * alpha[source]
                )
            column_arm_total = 0.0
            if large >= 0:
                for source in range(states):
                    column_arm_total += (
                        current[first, source] * column_arm[source]
                    )
            for destination in range(states):
                value = (
                    stay[destination] * current[first, destination]
                    + beta[destination]
                    * (prefix[destination] + suffix[destination + 1])
                )
                if large >= 0:
                    value += current[first, large] * row_arm[destination]
                    if destination == large:
                        value += column_arm_total
                output[first, destination] = value


@njit(cache=True, fastmath=False)
def _apply_excursion_axis(current, output, probability, axis):
    """Complete T' = (1-k) T + k U after a bridge mapped current to output."""

    states = current.shape[0]
    keep = 1.0 - probability
    if axis == 0:
        for second in range(states):
            total = 0.0
            for source in range(states):
                total += current[source, second]
            share = probability * total / states
            for destination in range(states):
                output[destination, second] = (
                    keep * output[destination, second] + share
                )
    else:
        for first in range(states):
            total = 0.0
            for source in range(states):
                total += current[first, source]
            share = probability * total / states
            for destination in range(states):
                output[first, destination] = (
                    keep * output[first, destination] + share
                )


@njit(cache=True, fastmath=False)
def _apply_null_axis(current, output, diagonal, off, axis):
    states = current.shape[0]
    coefficient = diagonal - off
    if axis == 0:
        for second in range(states):
            total = 0.0
            for source in range(states):
                total += current[source, second]
            for destination in range(states):
                output[destination, second] = (
                    coefficient * current[destination, second] + off * total
                )
    else:
        for first in range(states):
            total = 0.0
            for source in range(states):
                total += current[first, source]
            for destination in range(states):
                output[first, destination] = (
                    coefficient * current[first, destination] + off * total
                )


@njit(cache=True, fastmath=False)
def _advance_selectors(
    current, work, child, first_parent, second_parent, mode, boundary,
    bridge_left, bridge_right, bridge_diagonal, bridge_row_arm, bridge_column_arm,
    bridge_large_index, bridge_excursion, null_diagonal, null_off, bridge_branch,
    prefix, suffix,
):
    """Cross one bin boundary on both selector axes; the result returns to ``current``."""
    if mode >= 1:
        _apply_bridge_axis(
            current,
            work,
            bridge_left[first_parent, boundary],
            bridge_right[first_parent, boundary],
            bridge_diagonal[first_parent, boundary],
            bridge_row_arm[first_parent, boundary],
            bridge_column_arm[first_parent, boundary],
            bridge_large_index[first_parent, boundary],
            0, prefix, suffix,
            -1 if bridge_branch is None else bridge_branch[first_parent, boundary],
        )
        excursion = bridge_excursion[first_parent, boundary]
        if excursion > 0.0:
            _apply_excursion_axis(current, work, excursion, 0)
    else:
        _apply_null_axis(
            current,
            work,
            null_diagonal[child, boundary],
            null_off[child, boundary],
            0,
        )
    if mode == 2:
        _apply_bridge_axis(
            work,
            current,
            bridge_left[second_parent, boundary],
            bridge_right[second_parent, boundary],
            bridge_diagonal[second_parent, boundary],
            bridge_row_arm[second_parent, boundary],
            bridge_column_arm[second_parent, boundary],
            bridge_large_index[second_parent, boundary],
            1, prefix, suffix,
            -1 if bridge_branch is None else bridge_branch[second_parent, boundary],
        )
        excursion = bridge_excursion[second_parent, boundary]
        if excursion > 0.0:
            _apply_excursion_axis(work, current, excursion, 1)
    else:
        _apply_null_axis(
            work,
            current,
            null_diagonal[child, boundary],
            null_off[child, boundary],
            1,
        )


@njit(cache=True, fastmath=False)
def _score_path(
    child,
    first_parent,
    second_parent,
    mode,
    pi,
    candidate_alt,
    bridge_left,
    bridge_right,
    bridge_diagonal,
    bridge_off,
    bridge_row_arm,
    bridge_column_arm,
    bridge_large_index,
    bridge_excursion,
    state_alt,
    child_gl,
    child_coefficient,
    observed_site,
    observed_start,
    observed_stop,
    exponent,
    null_diagonal,
    null_off,
    mismatch,
    common_products=None,
    first_cached_child=0,
    bridge_branch=None,
    common_powered=None,
):
    """Forward log scores of the tempered (bin ``exponent``) and full-strength view.

    The two recurrences are independent: each keeps its own forward state,
    scaling and underflow path, and a view stops at its first non-finite
    increment. They share only the observed-site emission products (or
    their logarithms), which do not depend on the exponent; the full view
    applies them with exponent one.
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
                tempered[first, second] = 1.0 / (states * states)
            elif mode == 1:
                tempered[first, second] = pi[first_parent, 0, first] / states
            else:
                tempered[first, second] = (
                    pi[first_parent, 0, first] * pi[second_parent, 0, second]
                )
            full[first, second] = tempered[first, second]
    tempered_log = 0.0
    full_log = 0.0
    tempered_alive = True
    full_alive = True
    for block in range(bins):
        if block > 0:
            boundary = block - 1
            if tempered_alive:
                _advance_selectors(
                    tempered, scratch, child, first_parent, second_parent, mode,
                    boundary, bridge_left, bridge_right, bridge_diagonal,
                    bridge_row_arm, bridge_column_arm, bridge_large_index,
                    bridge_excursion, null_diagonal, null_off, bridge_branch,
                    prefix, suffix,
                )
            if full_alive:
                _advance_selectors(
                    full, scratch, child, first_parent, second_parent, mode,
                    boundary, bridge_left, bridge_right, bridge_diagonal,
                    bridge_row_arm, bridge_column_arm, bridge_large_index,
                    bridge_excursion, null_diagonal, null_off, bridge_branch,
                    prefix, suffix,
                )

        stable_product = True
        for first in range(states):
            for second in range(states):
                if common_products is not None:
                    saved = common_products[
                        child - first_cached_child, block, first, second
                    ]
                    if np.isfinite(saved):
                        work[first, second] = saved
                        continue
                product = 1.0
                for slot in range(
                    observed_start[child, block], observed_stop[child, block]
                ):
                    site = observed_site[slot]
                    first_alt = (
                        candidate_alt[first_parent, site, first]
                        if mode >= 1
                        else state_alt[first, site]
                    )
                    second_alt = (
                        candidate_alt[second_parent, site, second]
                        if mode == 2
                        else state_alt[second, site]
                    )
                    value = _site_likelihood(
                        child_gl[child, site],
                        child_coefficient[child, site],
                        first_alt,
                        second_alt,
                        mismatch,
                    )
                    if value <= 0.0:
                        product = 0.0
                        break
                    product *= value
                    if not np.isfinite(product) or product < 2.2250738585072014e-308:
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
                    for slot in range(
                        observed_start[child, block], observed_stop[child, block]
                    ):
                        site = observed_site[slot]
                        first_alt = (
                            candidate_alt[first_parent, site, first]
                            if mode >= 1
                            else state_alt[first, site]
                        )
                        second_alt = (
                            candidate_alt[second_parent, site, second]
                            if mode == 2
                            else state_alt[second, site]
                        )
                        value = _site_likelihood(
                            child_gl[child, site],
                            child_coefficient[child, site],
                            first_alt,
                            second_alt,
                            mismatch,
                        )
                        if value <= 0.0:
                            log_emission = -np.inf
                            break
                        log_emission += math.log(value)
                    work[first, second] = log_emission

        if tempered_alive:
            eta = exponent[child, block]
            if eta == 0.0:
                # A zero exponent makes the tempered emission uniform.
                scratch[:] = 1.0
                powered = (None if common_powered is None else
                           common_powered[child - first_cached_child, block])
                increment = _apply_probability_emission(tempered, scratch, eta, powered)
            elif stable_product:
                powered = (None if common_powered is None else
                           common_powered[child - first_cached_child, block])
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


@njit(cache=True, parallel=True, fastmath=False)
def _score_m0_kernel(
    pi, candidate_alt, bridge_left, bridge_right, bridge_diagonal, bridge_off,
    bridge_row_arm, bridge_column_arm, bridge_large_index, bridge_excursion,
    state_alt, child_gl, child_coefficient, observed_site, observed_start,
    observed_stop, exponent, null_diagonal, null_off, mismatch, eligible_child,
):
    tempered = np.empty(child_gl.shape[0], dtype=np.float64)
    full = np.empty(child_gl.shape[0], dtype=np.float64)
    for child in prange(child_gl.shape[0]):
        if eligible_child[child]:
            tempered[child], full[child] = _score_path(
                child, 0, 0, 0, pi, candidate_alt, bridge_left, bridge_right,
                bridge_diagonal, bridge_off, bridge_row_arm, bridge_column_arm,
                bridge_large_index, bridge_excursion, state_alt, child_gl,
                child_coefficient, observed_site, observed_start, observed_stop,
                exponent, null_diagonal, null_off, mismatch,
            )
        else:
            tempered[child] = -np.inf
            full[child] = -np.inf
    return tempered, full


@njit(cache=True, parallel=True, fastmath=False)
def _score_m1_kernel(
    pi, candidate_alt, bridge_left, bridge_right, bridge_diagonal, bridge_off,
    bridge_row_arm, bridge_column_arm, bridge_large_index, bridge_excursion,
    available, state_alt, child_gl, child_coefficient, observed_site,
    observed_start, observed_stop, exponent, null_diagonal, null_off, mismatch,
    eligible_edge, tempered_m0, full_m0, common_products, child_start, child_stop,
    bridge_branch=None, common_powered=None,
):
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
                child, parent, 0, 1, pi, candidate_alt, bridge_left,
                bridge_right, bridge_diagonal, bridge_off, bridge_row_arm,
                bridge_column_arm, bridge_large_index, bridge_excursion,
                state_alt, child_gl,
                child_coefficient, observed_site, observed_start, observed_stop,
                exponent, null_diagonal, null_off, mismatch,
                common_products, child_start, bridge_branch, common_powered,
            )
    return tempered, full


@njit(cache=True, parallel=True, fastmath=False)
def _score_m2_kernel(
    trios, pi, candidate_alt, bridge_left, bridge_right, bridge_diagonal,
    bridge_off, bridge_row_arm, bridge_column_arm, bridge_large_index,
    bridge_excursion,
    state_alt, child_gl, child_coefficient, observed_site,
    observed_start, observed_stop, exponent, null_diagonal, null_off, mismatch,
    common_products, first_cached_child, bridge_branch=None, common_powered=None,
):
    tempered = np.empty(len(trios), dtype=np.float64)
    full = np.empty(len(trios), dtype=np.float64)
    for row in prange(len(trios)):
        child = trios[row, 0]
        first = trios[row, 1]
        second = trios[row, 2]
        tempered[row], full[row] = _score_path(
            child, first, second, 2, pi, candidate_alt, bridge_left,
            bridge_right, bridge_diagonal, bridge_off, bridge_row_arm,
            bridge_column_arm, bridge_large_index, bridge_excursion,
            state_alt, child_gl,
            child_coefficient, observed_site, observed_start, observed_stop,
            exponent, null_diagonal, null_off, mismatch,
            common_products, first_cached_child, bridge_branch, common_powered,
        )
    return tempered, full
