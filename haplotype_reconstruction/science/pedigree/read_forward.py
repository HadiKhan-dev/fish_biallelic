"""M0 / M1 / M2 forward likelihoods of a child with its read state (LR phase 2, Tier 2).

The hidden state of a child at a bin is (f, s, c): the ordered pair of transmitted founder states (forward.py: under
M0 both from the null source, under M1 the first from the candidate parent's projected transmission, under M2 both
from the two parents) and the read class c of the child at its last observed marker (science.shared.reads, carried by
the painting's read state; C classes, C = 1 without it). The founder pair moves at bin edges (the bridges and the null
source of forward.advance, applied to every class alike); the class moves between the child's observed markers by
the composed class operators of its ChildReads (child_reads), whatever the founders do. Founder and class transitions
act on different coordinates, so each class operator is applied at the observed marker it leads into.

At an observed marker with transmitted ALT probabilities (a, b) of slots 1 and 2 (slot 1 = homologue 1) the child's
emission is E_c(a, b) = G(a, b) R_c(a, b): G the scalar site likelihood (forward.site_likelihood) and R_c the
GL-anchored read ratio (child_reads). The scalar part keeps forward.py's per-bin products, caches and underflow path;
the class part multiplies each state pair's class vector by the marker's operator and ratio (carry_bin). Each score
pools two views with equal weight, as forward.py: full strength (E itself) and tempered, where the information
exponent eta acts per site: G ** eta (the product's power, as before) times the tempered read factor
(tempered_factor: R ** eta, the painting's own tempered read factor). Without read state (one class,
or a child bypassing it) every chain is forward.py's scalar chain exactly.

Method: docs/methods/pedigree.md#child-read-state
"""

from __future__ import annotations

import math

import numpy as np

from ...core.kernels import kernel
from .forward import (
    _SMALLEST_NORMAL,
    ChildTerms,
    advance,
    normalized_power,
    site_likelihood,
)
from .kernels.read_chain import OVERLOADS
from .kernels.read_movement import OVERLOADS as MOVEMENT_OVERLOADS
from .read_transfer import carry_bin


@kernel(overloads=OVERLOADS["read_forward._normalise"], cache=True, nogil=True)
def _normalise(values):
    total = np.sum(values)
    if total <= 0.0 or not np.isfinite(total):
        return -np.inf
    values /= total
    return math.log(total)


@kernel(overloads=OVERLOADS["read_forward._apply_probability_emission"], cache=True, nogil=True)
def _apply_probability_emission(current, emission, exponent, powered=None):
    """Apply pow(emission / maximum, exponent) of each state pair to all its classes, then normalize the mass.

    Divide by the maximum before taking the power. Even exponent zero takes this normalization path, retaining its
    log(total) increment."""
    maximum = np.max(emission)
    if maximum <= 0.0 or not np.isfinite(maximum):
        return -np.inf
    classes = current.shape[0]
    states = current.shape[1]
    log_maximum = math.log(maximum)
    cached = powered is not None and powered.shape[0] == states + 1 and powered[states, 0] == maximum
    for first in range(states):
        for second in range(states):
            value = np.nan if powered is None else powered[first, second]
            if not cached or not np.isfinite(value):
                value = normalized_power(emission[first, second], maximum, exponent)
            for c in range(classes):
                current[c, first, second] *= value
    increment = _normalise(current)
    return increment + exponent * log_maximum if np.isfinite(increment) else -np.inf


@kernel(overloads=OVERLOADS["read_forward._apply_log_emission"], cache=True, nogil=True)
def _apply_log_emission(current, log_emission):
    maximum = np.max(log_emission)
    if not np.isfinite(maximum):
        return -np.inf
    classes = current.shape[0]
    states = current.shape[1]
    for first in range(states):
        for second in range(states):
            value = math.exp(log_emission[first, second] - maximum)
            for c in range(classes):
                current[c, first, second] *= value
    increment = _normalise(current)
    return increment + maximum if np.isfinite(increment) else -np.inf


@kernel(overloads=OVERLOADS["read_forward._apply_tempered_m1"], cache=True, nogil=True)
def _apply_tempered_m1(current, emission, exponent, powered, named_maximum, column_powered):
    """_apply_probability_emission for a fractional exponent with the M1 screen's caches: the same values, except that
    a BACKGROUND-second entry with a named first state takes its cached power when the bin's maximum equals the
    named-first maximum (the power is then of the identical base)."""
    maximum = np.max(emission)
    if maximum <= 0.0 or not np.isfinite(maximum):
        return -np.inf
    classes = current.shape[0]
    states = current.shape[1]
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
            for c in range(classes):
                current[c, first, second] *= value
    increment = _normalise(current)
    return increment + exponent * log_maximum if np.isfinite(increment) else -np.inf


@kernel(overloads=MOVEMENT_OVERLOADS["read_forward.advance_classes"], cache=True, nogil=True)
def advance_classes(
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
    """forward.advance on every class of ``current`` [C, S, S] (the founder transition does not move the class)."""
    for c in range(current.shape[0]):
        advance(
            current[c],
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
        )


@kernel(overloads=OVERLOADS["read_forward._initial_state"], cache=True, inline="always", nogil=True)
def _initial_state(current, child, first_parent, second_parent, mode, pi, null_frequency, read_initial):
    """The first bin's state: the slots' initial laws times the child's class law at its first marker."""
    classes = current.shape[0]
    states = current.shape[1]
    for first in range(states):
        for second in range(states):
            if mode == 0:
                prior = null_frequency[child, 0, first] * null_frequency[child, 0, second]
            elif mode == 1:
                prior = pi[first_parent, 0, first] * null_frequency[child, 0, second]
            else:
                prior = pi[first_parent, 0, first] * pi[second_parent, 0, second]
            for c in range(classes):
                current[c, first, second] = prior * read_initial[child, c]


@kernel(overloads=OVERLOADS["read_forward._bin_products"], cache=True, inline="always", nogil=True)
def _bin_products(
    work,
    child,
    first_parent,
    second_parent,
    mode,
    block,
    candidate_alt,
    state_alt,
    child_gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    mismatch,
    common_products,
    first_cached_child,
):
    """The scalar bin emissions of every state pair into ``work`` [S, S] (forward.py's products: the cached ones where
    finite, the site order, likelihood and zero rule otherwise); returns False when a product underflows, ``work``
    then holding the log emissions of every pair."""
    states = work.shape[0]
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
                value = site_likelihood(child_gl[child, site], coefficient[child, site], first_alt, second_alt,
                                        mismatch)
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
                    second_alt = candidate_alt[second_parent, site, second] if mode == 2 else state_alt[second, site]
                    value = site_likelihood(
                        child_gl[child, site], coefficient[child, site], first_alt, second_alt, mismatch
                    )
                    if value <= 0.0:
                        log_emission = -np.inf
                        break
                    log_emission += math.log(value)
                work[first, second] = log_emission
    return stable_product


@kernel(overloads=OVERLOADS["read_forward._tempered_increment"], cache=True, inline="always", nogil=True)
def _tempered_increment(tempered, work, scratch, eta, stable_product, powered):
    """The tempered view's scalar part of one bin (the bin product's power; uniform at exponent zero); returns its
    log increment."""
    states = work.shape[0]
    if eta == 0.0:
        # A zero exponent makes the tempered emission uniform.
        scratch[:] = 1.0
        return _apply_probability_emission(tempered, scratch, eta, powered)
    if stable_product:
        return _apply_probability_emission(tempered, work, eta, powered)
    for first in range(states):
        for second in range(states):
            scratch[first, second] = eta * work[first, second]
    return _apply_log_emission(tempered, scratch)


@kernel(overloads=OVERLOADS["read_forward.score_path"], cache=True, nogil=True)
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
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
    read_powered,
    class_cache=None,
):
    """(tempered, full) forward log likelihood of one child under one configuration, its read class carried
    (``class_cache``: class_transfers of the child's batch starting at ``first_cached_child``, or None).

    The views keep their own forward state, scaling and underflow path and each stops at its first non-finite
    increment; they share the observed-marker scalar emission products (or their logarithms), which the full view
    applies with exponent one; each carries its own class part (carry_bin)."""
    states = state_alt.shape[0]
    bins = exponent.shape[1]
    classes = read_initial.shape[1]
    tempered = np.empty((classes, states, states), dtype=np.float64)
    full = np.empty_like(tempered)
    mix = np.empty_like(tempered)
    work = np.empty((states, states), dtype=np.float64)
    scratch = np.empty_like(work)
    prefix = np.empty(states + 1, dtype=np.float64)
    suffix = np.empty(states + 1, dtype=np.float64)
    factor = np.empty(classes, dtype=np.float64)
    column = np.empty(classes, dtype=np.float64)
    moved = np.empty(classes, dtype=np.float64)
    first_source = first_parent if mode >= 1 else -1
    second_source = second_parent if mode == 2 else -1
    _initial_state(tempered, child, first_parent, second_parent, mode, pi, null_frequency, read_initial)
    full[:] = tempered
    tempered_log = 0.0
    full_log = 0.0
    tempered_alive = True
    full_alive = True
    for block in range(bins):
        if block > 0:
            boundary = block - 1
            if tempered_alive:
                advance_classes(tempered[:read_width[child, block]], scratch, child, first_parent, second_parent, mode, boundary, stay, alpha,
                                beta, row_arm, column_arm, pivot, excursion, branch, null_diagonal, null_off,
                                null_frequency, prefix, suffix)
            if full_alive:
                advance_classes(full[:read_width[child, block]], scratch, child, first_parent, second_parent, mode, boundary, stay, alpha, beta,
                                row_arm, column_arm, pivot, excursion, branch, null_diagonal, null_off,
                                null_frequency, prefix, suffix)
        stable_product = _bin_products(work, child, first_parent, second_parent, mode, block, candidate_alt, state_alt,
                                       child_gl, coefficient, site_index, site_start, site_stop, mismatch,
                                       common_products, first_cached_child)
        if tempered_alive:
            eta = exponent[child, block]
            carry_bin(tempered, mix, factor, column, child, block, first_source, second_source, candidate_alt,
                      state_alt, child_gl, site_index, site_start, site_stop, read_offset, read_reset, read_width, read_operator,
                      read_ratio_table, read_powered, True, eta, class_cache, child - first_cached_child,
                      moved)
            powered = None if common_powered is None else common_powered[child - first_cached_child, block]
            increment = _tempered_increment(tempered[:read_width[child, block + 1]], work, scratch, eta, stable_product, powered)
            if np.isfinite(increment):
                tempered_log += increment
            else:
                tempered_alive = False
                tempered_log = -np.inf
        if full_alive:
            carry_bin(full, mix, factor, column, child, block, first_source, second_source, candidate_alt, state_alt,
                      child_gl, site_index, site_start, site_stop, read_offset, read_reset, read_width, read_operator, read_ratio_table,
                      read_powered, False, 1.0, class_cache, child - first_cached_child, moved)
            if stable_product:
                increment = _apply_probability_emission(full[:read_width[child, block + 1]], work, 1.0, None)
            else:
                increment = _apply_log_emission(full[:read_width[child, block + 1]], work)
            if np.isfinite(increment):
                full_log += increment
            else:
                full_alive = False
                full_log = -np.inf
        if not tempered_alive and not full_alive:
            break
    return tempered_log, full_log


def read_arguments(t: ChildTerms) -> tuple:
    """The children's read-state arguments of the chain kernels (ChildTerms' read terms)."""
    return t.read_offset, t.read_initial, t.read_reset, t.read_width, t.read_operator, t.read_ratio, t.read_powered


@kernel(overloads=OVERLOADS["read_forward.score_m1_cached"], cache=True, nogil=True)
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
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
    read_powered,
    class_cache=None,
):
    """score_path(child, parent, 0, 1, ...) with the M1 screen's completed caches: the same forward recursion,
    emission values and operations, so the same (tempered, full) log likelihoods.

    Where every named-first product of the bin is cached, the candidate's BACKGROUND row is accumulated site by site
    for all second states together (each entry still multiplies its own site likelihoods in site order, with the
    zero rule; any underflow sends the bin to the log path, which recomputes every entry); otherwise the
    general loop of score_path runs unchanged."""
    states = state_alt.shape[0]
    last = states - 1
    bins = exponent.shape[1]
    classes = read_initial.shape[1]
    local = child - first_cached_child
    tempered = np.empty((classes, states, states), dtype=np.float64)
    full = np.empty_like(tempered)
    mix = np.empty_like(tempered)
    work = np.empty((states, states), dtype=np.float64)
    scratch = np.empty_like(work)
    prefix = np.empty(states + 1, dtype=np.float64)
    suffix = np.empty(states + 1, dtype=np.float64)
    row = np.empty(states, dtype=np.float64)
    zeroed = np.empty(states, dtype=np.bool_)
    factor = np.empty(classes, dtype=np.float64)
    column = np.empty(classes, dtype=np.float64)
    moved = np.empty(classes, dtype=np.float64)
    _initial_state(tempered, child, parent, 0, 1, pi, null_frequency, read_initial)
    full[:] = tempered
    tempered_log = 0.0
    full_log = 0.0
    tempered_alive = True
    full_alive = True
    for block in range(bins):
        if block > 0:
            boundary = block - 1
            if tempered_alive:
                advance_classes(tempered[:read_width[child, block]], scratch, child, parent, 0, 1, boundary, stay, alpha, beta, row_arm,
                                column_arm, pivot, excursion, branch, null_diagonal, null_off, null_frequency, prefix,
                                suffix)
            if full_alive:
                advance_classes(full[:read_width[child, block]], scratch, child, parent, 0, 1, boundary, stay, alpha, beta, row_arm, column_arm,
                                pivot, excursion, branch, null_diagonal, null_off, null_frequency, prefix, suffix)

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
            carry_bin(tempered, mix, factor, column, child, block, parent, -1, candidate_alt, state_alt, child_gl,
                      site_index, site_start, site_stop, read_offset, read_reset, read_width, read_operator, read_ratio_table, read_powered,
                      True, eta, class_cache, local, moved)
            powered = common_powered[local, block]
            if eta == 0.0:
                # A zero exponent makes the tempered emission uniform.
                scratch[:] = 1.0
                increment = _apply_probability_emission(tempered[:read_width[child, block + 1]], scratch, eta, powered)
            elif stable_product and complete and eta != 1.0:
                increment = _apply_tempered_m1(
                    tempered[:read_width[child, block + 1]], work, eta, powered, named_maximum[local, block], column_powered[local, block]
                )
            elif stable_product:
                increment = _apply_probability_emission(tempered[:read_width[child, block + 1]], work, eta, powered)
            else:
                for first in range(states):
                    for second in range(states):
                        scratch[first, second] = eta * work[first, second]
                increment = _apply_log_emission(tempered[:read_width[child, block + 1]], scratch)
            if np.isfinite(increment):
                tempered_log += increment
            else:
                tempered_alive = False
                tempered_log = -np.inf
        if full_alive:
            carry_bin(full, mix, factor, column, child, block, parent, -1, candidate_alt, state_alt, child_gl,
                      site_index, site_start, site_stop, read_offset, read_reset, read_width, read_operator, read_ratio_table, read_powered,
                      False, 1.0, class_cache, local, moved)
            if stable_product:
                increment = _apply_probability_emission(full[:read_width[child, block + 1]], work, 1.0, None)
            else:
                increment = _apply_log_emission(full[:read_width[child, block + 1]], work)
            if np.isfinite(increment):
                full_log += increment
            else:
                full_alive = False
                full_log = -np.inf
        if not tempered_alive and not full_alive:
            break
    return tempered_log, full_log
