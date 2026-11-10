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

Method: docs/methods/pedigree.md#child-forward-model-and-composite-score
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numba import prange

from ...core.errors import InvariantError, SettingsError
from ...core.kernels import kernel
from .kernels.signatures import OVERLOADS

# Temporary per-call cache of shared emission products, tiled over children (not a setting).
from .parameters import COMMON_EMISSION_CACHE_BYTES, NULL_ROUNDING_ULPS, SITE_POLYNOMIAL_THRESHOLD
from .projection import Projection

_SMALLEST_NORMAL = 2.2250738585072014e-308


@kernel(overloads=OVERLOADS["forward.site_likelihood"], cache=True, inline="always", nogil=True)
def site_likelihood(gl, coefficient, first_alt, second_alt, mismatch):
    value = coefficient[0] + coefficient[1] * (first_alt + second_alt) + coefficient[2] * first_alt * second_alt
    if value > SITE_POLYNOMIAL_THRESHOLD:
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


@kernel(overloads=OVERLOADS["forward.observed_sites"], cache=True, parallel=True, nogil=True)
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


@kernel(overloads=OVERLOADS["forward._common_products"], cache=True, parallel=True, nogil=True)
def _common_products(
    state_alt, child_gl, coefficient, site_index, site_start, site_stop, mismatch, child_start, child_stop
):
    """Emission products of called-founder state pairs per child and bin; NaN = the general path.

    Missing named alleles (BACKGROUND's frequency) and BACKGROUND (candidate-specific) take the general path, whose
    site order, likelihood and underflow fallback stay authoritative.
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
                    value = site_likelihood(child_gl[child, site], coefficient[child, site], a, b, mismatch)
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


@kernel(overloads=OVERLOADS["forward._fill_named_first_products"], cache=True, parallel=True, nogil=True)
def _fill_named_first_products(
    products, state_alt, child_gl, coefficient, site_index, site_start, site_stop, mismatch, child_start
):
    """Complete, in place, the cached products that no candidate changes when the first slot transmits a named state
    and the second slot is the child's null source (M0/M1 configurations).

    A named state passes through every candidate's projection unchanged (projection._transmitted_alt), so under M0
    and M1 the entries with a named first state depend on the child, bin and state pair only, also where the second
    state is BACKGROUND or a named allele is uncalled (which _common_products leaves NaN). Each such entry is
    computed exactly as score_path's general path computes it (site order, likelihood, the zero-likelihood rule);
    an entry that path would find unstable (underflow) stays NaN, so the general path still meets it and switches
    to log space. The powered (tempered) cache is not extended: those entries keep their direct power.
    Not valid for M2, whose second slot is a candidate's.
    """
    children, bins, states, _ = products.shape
    for task in prange(children * bins):
        local_child, block = task // bins, task % bins
        child = child_start + local_child
        for first in range(states - 1):
            for second in range(states):
                if np.isfinite(products[local_child, block, first, second]):
                    continue
                product = 1.0
                stable = True
                for slot in range(site_start[child, block], site_stop[child, block]):
                    site = site_index[slot]
                    value = site_likelihood(
                        child_gl[child, site],
                        coefficient[child, site],
                        state_alt[first, site],
                        state_alt[second, site],
                        mismatch,
                    )
                    if value <= 0.0:
                        product = 0.0
                        break
                    product *= value
                    if not np.isfinite(product) or product < _SMALLEST_NORMAL:
                        stable = False
                        break
                if stable:
                    products[local_child, block, first, second] = product


@kernel(overloads=OVERLOADS["forward.normalized_power"], cache=True, inline="always", nogil=True)
def normalized_power(emission, maximum, exponent):
    """The probability-domain normalized power, including eta one."""
    base = emission / maximum
    return base if exponent == 1.0 else math.pow(base, exponent)


@kernel(overloads=OVERLOADS["forward._powered_products"], cache=True, parallel=True, nogil=True)
def _powered_products(products, exponent, child_start):
    """Cache normalized powers, plus their exact normalization maximum.

    The final row stores the maximum in column zero. A consumer may reuse an
    entry only when its complete emission matrix has exactly that maximum;
    otherwise it evaluates the same normalized power directly. Unknown products
    stay NaN. This makes cache capacity and completion arithmetically neutral.
    """
    children, bins, states, _ = products.shape
    powered = np.full((children, bins, states + 1, states), np.nan)
    for task in prange(children * bins):
        child, block = task // bins, task % bins
        eta = exponent[child_start + child, block]
        maximum = 0.0
        for first in range(states):
            for second in range(states):
                value = products[child, block, first, second]
                if np.isfinite(value):
                    maximum = max(maximum, value)
        if maximum > 0.0 and 0.0 < eta < 1.0:
            powered[child, block, states, 0] = maximum
            for first in range(states):
                for second in range(states):
                    value = products[child, block, first, second]
                    if np.isfinite(value) and value >= 0.0:
                        powered[child, block, first, second] = normalized_power(value, maximum, eta)
    return powered


@kernel(overloads=OVERLOADS["forward._normalise"], cache=True, nogil=True)
def _normalise(values):
    total = np.sum(values)
    if total <= 0.0 or not np.isfinite(total):
        return -np.inf
    values /= total
    return math.log(total)


@kernel(overloads=OVERLOADS["forward._apply_probability_emission"], cache=True, nogil=True)
def _apply_probability_emission(current, emission, exponent, powered=None):
    """Apply pow(emission / maximum, exponent), then normalize the pair mass.

    Divide by the maximum before taking the power. Even exponent zero takes
    this normalization path, retaining its log(total) increment.
    """
    maximum = np.max(emission)
    if maximum <= 0.0 or not np.isfinite(maximum):
        return -np.inf
    states = current.shape[0]
    log_maximum = math.log(maximum)
    cached = powered is not None and powered.shape[0] == states + 1 and powered[states, 0] == maximum
    for first in range(states):
        for second in range(states):
            value = np.nan if powered is None else powered[first, second]
            if not cached or not np.isfinite(value):
                value = normalized_power(emission[first, second], maximum, exponent)
            current[first, second] *= value
    increment = _normalise(current)
    return increment + exponent * log_maximum if np.isfinite(increment) else -np.inf


@kernel(overloads=OVERLOADS["forward._apply_log_emission"], cache=True, nogil=True)
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


@kernel(overloads=OVERLOADS["forward._apply_star_bridge"], cache=True, nogil=True)
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


@kernel(overloads=OVERLOADS["forward.apply_bridge"], cache=True, nogil=True)
def apply_bridge(current, output, stay, alpha, beta, row_arm, column_arm, pivot, axis, prefix, suffix, branch):
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
                value = stay[destination] * current[destination, second] + beta[destination] * (
                    prefix[destination] + suffix[destination + 1]
                )
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
                value = stay[destination] * current[first, destination] + beta[destination] * (
                    prefix[destination] + suffix[destination + 1]
                )
                if pivot >= 0:
                    value += current[first, pivot] * row_arm[destination]
                    if destination == pivot:
                        value += column_arm_total
                output[first, destination] = value


@kernel(overloads=OVERLOADS["forward.apply_excursion"], cache=True, nogil=True)
def apply_excursion(current, output, probability, axis):
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


@kernel(overloads=OVERLOADS["forward.apply_null"], cache=True, nogil=True)
def apply_null(current, output, diagonal, off, frequency, axis):
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
                output[destination, second] = (
                    coefficient * current[destination, second] + move * frequency[destination] * total
                )
    else:
        for first in range(states):
            total = 0.0
            for source in range(states):
                total += current[first, source]
            for destination in range(states):
                output[first, destination] = (
                    coefficient * current[first, destination] + move * frequency[destination] * total
                )


@kernel(overloads=OVERLOADS["forward.advance"], cache=True, nogil=True)
def advance(
    current,
    work,
    child,
    first_parent,
    second_parent,
    mode,
    boundary,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    excursion,
    branch,
    null_diagonal,
    null_off,
    null_frequency,
    prefix,
    suffix,
):
    """Cross one bin edge on both selector axes; the result returns to ``current``."""
    if mode >= 1:
        apply_bridge(
            current,
            work,
            stay[first_parent, boundary],
            alpha[first_parent, boundary],
            beta[first_parent, boundary],
            row_arm[first_parent, boundary],
            column_arm[first_parent, boundary],
            pivot[first_parent, boundary],
            0,
            prefix,
            suffix,
            branch[first_parent, boundary],
        )
        probability = excursion[first_parent, boundary]
        if probability > 0.0:
            apply_excursion(current, work, probability, 0)
    else:
        apply_null(
            current,
            work,
            null_diagonal[child, boundary],
            null_off[child, boundary],
            null_frequency[child, boundary + 1],
            0,
        )
    if mode == 2:
        apply_bridge(
            work,
            current,
            stay[second_parent, boundary],
            alpha[second_parent, boundary],
            beta[second_parent, boundary],
            row_arm[second_parent, boundary],
            column_arm[second_parent, boundary],
            pivot[second_parent, boundary],
            1,
            prefix,
            suffix,
            branch[second_parent, boundary],
        )
        probability = excursion[second_parent, boundary]
        if probability > 0.0:
            apply_excursion(work, current, probability, 1)
    else:
        apply_null(
            work,
            current,
            null_diagonal[child, boundary],
            null_off[child, boundary],
            null_frequency[child, boundary + 1],
            1,
        )


@kernel(overloads=OVERLOADS["forward.score_path"], cache=True, nogil=True)
def score_path(
    child,
    first_parent,
    second_parent,
    mode,
    pi,
    candidate_alt,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    excursion,
    branch,
    state_alt,
    child_gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    exponent,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    common_products,
    first_cached_child,
    common_powered,
):
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
                advance(
                    tempered,
                    scratch,
                    child,
                    first_parent,
                    second_parent,
                    mode,
                    boundary,
                    stay,
                    alpha,
                    beta,
                    row_arm,
                    column_arm,
                    pivot,
                    excursion,
                    branch,
                    null_diagonal,
                    null_off,
                    null_frequency,
                    prefix,
                    suffix,
                )
            if full_alive:
                advance(
                    full,
                    scratch,
                    child,
                    first_parent,
                    second_parent,
                    mode,
                    boundary,
                    stay,
                    alpha,
                    beta,
                    row_arm,
                    column_arm,
                    pivot,
                    excursion,
                    branch,
                    null_diagonal,
                    null_off,
                    null_frequency,
                    prefix,
                    suffix,
                )

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
                    first_alt = candidate_alt[first_parent, site, first] if mode >= 1 else state_alt[first, site]
                    second_alt = candidate_alt[second_parent, site, second] if mode == 2 else state_alt[second, site]
                    value = site_likelihood(
                        child_gl[child, site], coefficient[child, site], first_alt, second_alt, mismatch
                    )
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
                        first_alt = candidate_alt[first_parent, site, first] if mode >= 1 else state_alt[first, site]
                        second_alt = (
                            candidate_alt[second_parent, site, second] if mode == 2 else state_alt[second, site]
                        )
                        value = site_likelihood(
                            child_gl[child, site], coefficient[child, site], first_alt, second_alt, mismatch
                        )
                        if value <= 0.0:
                            log_emission = -np.inf
                            break
                        log_emission += math.log(value)
                    work[first, second] = log_emission

        if tempered_alive:
            eta = exponent[child, block]
            powered = None if common_powered is None else common_powered[child - first_cached_child, block]
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


@kernel(overloads=OVERLOADS["forward.named_column_powers"], cache=True, parallel=True, nogil=True)
def named_column_powers(products, exponent, child_start):
    """For the M1 screen's completed cache (_fill_named_first_products): per child and bin, the largest named-first
    product (NaN unless every named-first entry is cached) and, for each named first state, the tempered power
    pow(product[first, BACKGROUND] / that maximum, exponent) that _apply_probability_emission takes for the
    BACKGROUND-second entry when the bin's maximum is that same value (fractional exponents only; NaN otherwise)."""
    children, bins, states, _ = products.shape
    last = states - 1
    maximum = np.full((children, bins), np.nan)
    powered = np.full((children, bins, last), np.nan)
    for task in prange(children * bins):
        child, block = task // bins, task % bins
        largest = -np.inf
        complete = True
        for first in range(last):
            for second in range(states):
                value = products[child, block, first, second]
                if not np.isfinite(value):
                    complete = False
                    break
                largest = max(largest, value)
            if not complete:
                break
        if not complete:
            continue
        maximum[child, block] = largest
        eta = exponent[child_start + child, block]
        if largest > 0.0 and np.isfinite(largest) and eta != 1.0 and eta != 0.0:
            for first in range(last):
                powered[child, block, first] = normalized_power(products[child, block, first, last], largest, eta)
    return maximum, powered


@kernel(overloads=OVERLOADS["forward._apply_tempered_m1"], cache=True, nogil=True)
def _apply_tempered_m1(current, emission, exponent, powered, named_maximum, column_powered):
    """_apply_probability_emission for a fractional exponent with the M1 screen's caches: the same values, except that
    a BACKGROUND-second entry with a named first state takes its cached power when the bin's maximum equals the
    named-first maximum (the power is then of the identical base)."""
    maximum = np.max(emission)
    if maximum <= 0.0 or not np.isfinite(maximum):
        return -np.inf
    states = current.shape[0]
    last = states - 1
    log_maximum = math.log(maximum)
    cached = powered.shape[0] == states + 1 and powered[states, 0] == maximum
    column = maximum == named_maximum
    for first in range(states):
        for second in range(states):
            saved = powered[first, second]
            if cached and np.isfinite(saved):
                value = saved
            elif column and second == last and first < last and np.isfinite(column_powered[first]):
                value = column_powered[first]
            else:
                value = normalized_power(emission[first, second], maximum, exponent)
            current[first, second] *= value
    increment = _normalise(current)
    return increment + exponent * log_maximum if np.isfinite(increment) else -np.inf


@kernel(overloads=OVERLOADS["forward.score_m1_cached"], cache=True, nogil=True)
def score_m1_cached(
    child,
    parent,
    pi,
    candidate_alt,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    excursion,
    branch,
    state_alt,
    child_gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    exponent,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    common_products,
    first_cached_child,
    common_powered,
    named_maximum,
    column_powered,
):
    """score_path(child, parent, 0, 1, ...) with the M1 screen's completed caches: the same
    forward recursion, emission values and operations, so the same (tempered, full) log likelihoods.

    Where every named-first product of the bin is cached, the candidate's BACKGROUND row is accumulated site by site
    for all second states together (each entry still multiplies its own site likelihoods in site order, with the
    zero rule; any underflow sends the bin to the log path, which recomputes every entry); otherwise the
    general loop of score_path runs unchanged.
    """
    states = state_alt.shape[0]
    last = states - 1
    bins = exponent.shape[1]
    local = child - first_cached_child
    tempered = np.empty((states, states), dtype=np.float64)
    full = np.empty_like(tempered)
    work = np.empty_like(tempered)
    scratch = np.empty_like(tempered)
    prefix = np.empty(states + 1, dtype=np.float64)
    suffix = np.empty(states + 1, dtype=np.float64)
    row = np.empty(states, dtype=np.float64)
    zeroed = np.empty(states, dtype=np.bool_)
    for first in range(states):
        for second in range(states):
            tempered[first, second] = pi[parent, 0, first] * null_frequency[child, 0, second]
            full[first, second] = tempered[first, second]
    tempered_log = 0.0
    full_log = 0.0
    tempered_alive = True
    full_alive = True
    for block in range(bins):
        if block > 0:
            boundary = block - 1
            if tempered_alive:
                advance(
                    tempered,
                    scratch,
                    child,
                    parent,
                    0,
                    1,
                    boundary,
                    stay,
                    alpha,
                    beta,
                    row_arm,
                    column_arm,
                    pivot,
                    excursion,
                    branch,
                    null_diagonal,
                    null_off,
                    null_frequency,
                    prefix,
                    suffix,
                )
            if full_alive:
                advance(
                    full,
                    scratch,
                    child,
                    parent,
                    0,
                    1,
                    boundary,
                    stay,
                    alpha,
                    beta,
                    row_arm,
                    column_arm,
                    pivot,
                    excursion,
                    branch,
                    null_diagonal,
                    null_off,
                    null_frequency,
                    prefix,
                    suffix,
                )

        stable_product = True
        complete = np.isfinite(named_maximum[local, block])
        if complete:
            for first in range(last):
                for second in range(states):
                    work[first, second] = common_products[local, block, first, second]
            for second in range(states):
                row[second] = 1.0
                zeroed[second] = False
            for slot in range(site_start[child, block], site_stop[child, block]):
                site = site_index[slot]
                first_alt = candidate_alt[parent, site, last]
                for second in range(states):
                    if zeroed[second]:
                        continue
                    value = site_likelihood(
                        child_gl[child, site], coefficient[child, site], first_alt, state_alt[second, site], mismatch
                    )
                    if value <= 0.0:
                        row[second] = 0.0
                        zeroed[second] = True
                        continue
                    row[second] *= value
                    if not np.isfinite(row[second]) or row[second] < _SMALLEST_NORMAL:
                        stable_product = False
                        break
                if not stable_product:
                    break
            for second in range(states):
                work[last, second] = row[second]
        else:
            for first in range(states):
                for second in range(states):
                    saved = common_products[local, block, first, second]
                    if np.isfinite(saved):
                        work[first, second] = saved
                        continue
                    product = 1.0
                    for slot in range(site_start[child, block], site_stop[child, block]):
                        site = site_index[slot]
                        value = site_likelihood(
                            child_gl[child, site],
                            coefficient[child, site],
                            candidate_alt[parent, site, first],
                            state_alt[second, site],
                            mismatch,
                        )
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
                        value = site_likelihood(
                            child_gl[child, site],
                            coefficient[child, site],
                            candidate_alt[parent, site, first],
                            state_alt[second, site],
                            mismatch,
                        )
                        if value <= 0.0:
                            log_emission = -np.inf
                            break
                        log_emission += math.log(value)
                    work[first, second] = log_emission

        if tempered_alive:
            eta = exponent[child, block]
            powered = common_powered[local, block]
            if eta == 0.0:
                # A zero exponent makes the tempered emission uniform.
                scratch[:] = 1.0
                increment = _apply_probability_emission(tempered, scratch, eta, powered)
            elif stable_product and complete and eta != 1.0:
                increment = _apply_tempered_m1(
                    tempered, work, eta, powered, named_maximum[local, block], column_powered[local, block]
                )
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


@kernel(overloads=OVERLOADS["forward._m0_kernel"], cache=True, parallel=True, nogil=True)
def _m0_kernel(
    pi,
    candidate_alt,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    excursion,
    branch,
    state_alt,
    child_gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    exponent,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    eligible_child,
):
    tempered = np.empty(child_gl.shape[0], dtype=np.float64)
    full = np.empty(child_gl.shape[0], dtype=np.float64)
    for child in prange(child_gl.shape[0]):
        if eligible_child[child]:
            tempered[child], full[child] = score_path(
                child,
                0,
                0,
                0,
                pi,
                candidate_alt,
                stay,
                alpha,
                beta,
                row_arm,
                column_arm,
                pivot,
                excursion,
                branch,
                state_alt,
                child_gl,
                coefficient,
                site_index,
                site_start,
                site_stop,
                exponent,
                null_diagonal,
                null_off,
                null_frequency,
                mismatch,
                None,
                0,
                None,
            )
        else:
            tempered[child] = -np.inf
            full[child] = -np.inf
    return tempered, full


@kernel(overloads=OVERLOADS["forward._m2_kernel"], cache=True, parallel=True, nogil=True)
def _m2_kernel(
    trios,
    pi,
    candidate_alt,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    excursion,
    branch,
    state_alt,
    child_gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    exponent,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    common_products,
    first_cached_child,
    common_powered,
):
    tempered = np.empty(len(trios), dtype=np.float64)
    full = np.empty(len(trios), dtype=np.float64)
    for row in prange(len(trios)):
        tempered[row], full[row] = score_path(
            trios[row, 0],
            trios[row, 1],
            trios[row, 2],
            2,
            pi,
            candidate_alt,
            stay,
            alpha,
            beta,
            row_arm,
            column_arm,
            pivot,
            excursion,
            branch,
            state_alt,
            child_gl,
            coefficient,
            site_index,
            site_start,
            site_stop,
            exponent,
            null_diagonal,
            null_off,
            null_frequency,
            mismatch,
            common_products,
            first_cached_child,
            common_powered,
        )
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

    def __init__(
        self,
        projection: Projection,
        gl: np.ndarray,
        observed: np.ndarray,
        exponent: np.ndarray,
        selector_switch_probability: np.ndarray,
        same: np.ndarray,
        one: np.ndarray,
        two: np.ndarray,
        mismatch: float,
        null_prior_samples: float,
    ):
        children = gl.shape[0]
        states = projection.n_states
        bins = projection.n_bins
        self.gl = gl
        self.coefficient = child_coefficients(gl, mismatch)
        self.site_index, self.site_start, self.site_stop = observed_sites(
            observed, projection.bin_start, projection.bin_stop
        )
        self.exponent = exponent
        self.mismatch = float(mismatch)
        selector = np.broadcast_to(selector_switch_probability, (children, bins - 1))
        self.null_diagonal = np.ascontiguousarray(
            (1.0 - selector) * (same[None, :] + (states - 1) * one[None, :]) + selector / states
        )
        self.null_off = np.ascontiguousarray(
            (1.0 - selector) * (one[None, :] + (states - 1) * two[None, :]) + selector / states
        )
        # Persistence weight keep = diagonal - off >= 0: a saturated kernel (theoretically diagonal == off: no
        # persistence) can round a few ULP below zero; that rounding is clipped to zero persistence (diagonal = off =
        # 1 / S, rows summing to one), a materially negative keep is an error. The certificate needs keep >= 0.
        keep = self.null_diagonal - self.null_off
        rounding = (
            NULL_ROUNDING_ULPS * np.finfo(np.float64).eps * states * np.maximum(self.null_diagonal, self.null_off)
        )
        if np.any(keep < -rounding):
            raise InvariantError("null source kernel prefers a specific other state to staying")
        saturated = keep < 0.0
        if np.any(saturated):
            self.null_diagonal = np.ascontiguousarray(np.where(saturated, 1.0 / states, self.null_diagonal))
            self.null_off = np.ascontiguousarray(np.where(saturated, 1.0 / states, self.null_off))
        if not null_prior_samples > 0.0:
            raise SettingsError("null_prior_samples must be positive")
        self.null_frequency = null_frequencies(projection.pi, projection.available, float(null_prior_samples))
        self.fractional = bool(np.any((exponent > 0.0) & (exponent < 1.0)))
        cache_children = COMMON_EMISSION_CACHE_BYTES // max(1, bins * states * (2 * states + 1) * 8)
        self.cached = cache_children > 0
        self.batch = min(children, cache_children) if cache_children else children

    def common(self, projection: Projection, child_start: int, child_stop: int):
        """(cached products, powered products) of a child batch, or (None, None) without a cache."""
        if not self.cached:
            return None, None
        products = _common_products(
            projection.state_alt,
            self.gl,
            self.coefficient,
            self.site_index,
            self.site_start,
            self.site_stop,
            self.mismatch,
            child_start,
            child_stop,
        )
        powered = _powered_products(products, self.exponent, child_start) if self.fractional else None
        return products, powered

    def common_null_second(self, projection: Projection, child_start: int, child_stop: int):
        """``common`` for M0/M1 configurations only (the M1 screen): the products also hold every entry with a named
        first state (_fill_named_first_products); the powered cache is ``common``'s."""
        products, powered = self.common(projection, child_start, child_stop)
        if products is not None:
            _fill_named_first_products(
                products,
                projection.state_alt,
                self.gl,
                self.coefficient,
                self.site_index,
                self.site_start,
                self.site_stop,
                self.mismatch,
                child_start,
            )
        return products, powered


def model_arguments(projection: Projection) -> tuple:
    return (
        projection.pi,
        projection.candidate_alt,
        projection.stay,
        projection.alpha,
        projection.beta,
        projection.row_arm,
        projection.column_arm,
        projection.pivot,
        projection.excursion,
        projection.branch,
    )


def component_m2(
    projection: Projection,
    terms: ChildTerms,
    trios: np.ndarray,
    zero: np.ndarray,
    one: np.ndarray,
    claim_threads=lambda: None,
) -> np.ndarray:
    """Pooled M2 log likelihoods [T] of the panel rows; a row with an evidence-free parent reduces
    to the other parent's M1 (or M0) screen value."""
    children = terms.gl.shape[0]
    first_available = projection.available[trios[:, 1]]
    second_available = projection.available[trios[:, 2]]
    active = first_available & second_available
    two = np.empty(len(trios), dtype=np.float64)
    for row in np.flatnonzero(~active):
        child, first, second = trios[row]
        two[row] = (
            one[child, first] if first_available[row] else one[child, second] if second_available[row] else zero[child]
        )
    active_rows = np.flatnonzero(active)
    if not len(active_rows):
        return two
    active_children = trios[active_rows, 0]
    common_args = (
        projection.state_alt,
        terms.gl,
        terms.coefficient,
        terms.site_index,
        terms.site_start,
        terms.site_stop,
        terms.exponent,
        terms.null_diagonal,
        terms.null_off,
        terms.null_frequency,
        terms.mismatch,
    )
    for child_start in range(0, children, terms.batch):
        child_stop = min(children, child_start + terms.batch)
        batch_rows = active_rows[(active_children >= child_start) & (active_children < child_stop)]
        if not len(batch_rows):
            continue
        claim_threads()
        products, powered = terms.common(projection, child_start, child_stop)
        claim_threads()
        tempered, full = _m2_kernel(
            np.ascontiguousarray(trios[batch_rows]),
            *model_arguments(projection),
            *common_args,
            products,
            child_start,
            powered,
        )
        two[batch_rows] = 0.5 * (tempered + full)
        del products, powered
    return two


@kernel(overloads=OVERLOADS["forward._dyads"], cache=True, parallel=True, nogil=True)
def _dyads(
    pairs,
    pi,
    candidate_alt,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    excursion,
    branch,
    available,
    state_alt,
    gl,
    coefficient,
    indices,
    starts,
    stops,
    exponent,
    null_diag,
    null_off,
    null_frequency,
    mismatch,
    zero,
    products,
    first_child,
    powered,
):
    output = np.empty(len(pairs))
    for i in prange(len(pairs)):
        c, p = pairs[i]
        if not available[p]:
            output[i] = zero[c]
        else:
            a, b = score_path(
                c,
                p,
                0,
                1,
                pi,
                candidate_alt,
                stay,
                alpha,
                beta,
                row_arm,
                column_arm,
                pivot,
                excursion,
                branch,
                state_alt,
                gl,
                coefficient,
                indices,
                starts,
                stops,
                exponent,
                null_diag,
                null_off,
                null_frequency,
                mismatch,
                products,
                first_child,
                powered,
            )
            output[i] = 0.5 * (a + b)
    return output


@kernel(overloads=OVERLOADS["forward._dyads_cached"], cache=True, parallel=True, nogil=True)
def _dyads_cached(
    pairs,
    pi,
    candidate_alt,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    excursion,
    branch,
    available,
    state_alt,
    gl,
    coefficient,
    indices,
    starts,
    stops,
    exponent,
    null_diag,
    null_off,
    null_frequency,
    mismatch,
    zero,
    products,
    first_child,
    powered,
    named_maximum,
    column_powered,
):
    """_dyads with the completed M1 caches (score_m1_cached: the same values)."""
    output = np.empty(len(pairs))
    for i in prange(len(pairs)):
        c, p = pairs[i]
        if not available[p]:
            output[i] = zero[c]
        else:
            a, b = score_m1_cached(
                c,
                p,
                pi,
                candidate_alt,
                stay,
                alpha,
                beta,
                row_arm,
                column_arm,
                pivot,
                excursion,
                branch,
                state_alt,
                gl,
                coefficient,
                indices,
                starts,
                stops,
                exponent,
                null_diag,
                null_off,
                null_frequency,
                mismatch,
                products,
                first_child,
                powered,
                named_maximum,
                column_powered,
            )
            output[i] = 0.5 * (a + b)
    return output


def terms_args(p, t):
    return (
        p.state_alt,
        t.gl,
        t.coefficient,
        t.site_index,
        t.site_start,
        t.site_stop,
        t.exponent,
        t.null_diagonal,
        t.null_off,
        t.null_frequency,
        t.mismatch,
    )


def component_m0(
    p: Projection, t: ChildTerms, eligible_children: np.ndarray, claim_threads: Callable[[], int | None]
) -> np.ndarray:
    claim_threads()
    a, b = _m0_kernel(*model_arguments(p), *terms_args(p, t), np.ascontiguousarray(eligible_children))
    return 0.5 * (a + b)


# Pair chunks per M1 query call: the calling thread re-reads its CPU share before each chunk, so a
# chromosome whose query outlasts its peers grows into the CPUs they free (a kernel's team is fixed for the whole call).
# Pairs are scored independently, so chunking does not change any value.
_CLAIM_CHUNKS = 32
_CLAIM_CHUNK_MINIMUM = 256


def component_m1(
    p: Projection,
    t: ChildTerms,
    pairs: np.ndarray,
    zero_scores: np.ndarray,
    first_child: int,
    products: np.ndarray | None,
    powered: np.ndarray | None,
    claim_threads: Callable[[], int | None],
    columns: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    pairs = np.ascontiguousarray(pairs, dtype=np.int64)
    model, terms = model_arguments(p), terms_args(p, t)
    output = np.empty(len(pairs))
    step = max(_CLAIM_CHUNK_MINIMUM, -(-len(pairs) // _CLAIM_CHUNKS))
    for start in range(0, max(1, len(pairs)), step):
        claim_threads()
        chunk = pairs[start : start + step]
        if columns is not None and products is not None and powered is not None:
            output[start : start + len(chunk)] = _dyads_cached(
                chunk, *model, p.available, *terms, zero_scores, products, first_child, powered, *columns
            )
        else:
            output[start : start + len(chunk)] = _dyads(
                chunk, *model, p.available, *terms, zero_scores, products, first_child, powered
            )
    return output
