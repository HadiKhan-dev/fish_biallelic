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
the class part multiplies each state pair's class vector by the marker's operator and ratio (carry_bin). Each
configuration has one likelihood, every observed marker entering once at full strength (4.22, R4 1a). Without read
state (one class, or a child bypassing it) every chain is forward.py's scalar chain exactly.

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
def _apply_probability_emission(current, emission):
    """Apply emission / maximum of each state pair to all its classes, then normalize the mass; returns the log
    increment."""
    maximum = np.max(emission)
    if maximum <= 0.0 or not np.isfinite(maximum):
        return -np.inf
    classes = current.shape[0]
    states = current.shape[1]
    log_maximum = math.log(maximum)
    for first in range(states):
        for second in range(states):
            emission[first, second] /= maximum
    for c in range(classes):
        for first in range(states):
            for second in range(states):
                current[c, first, second] *= emission[first, second]
    increment = _normalise(current)
    return increment + log_maximum if np.isfinite(increment) else -np.inf


@kernel(overloads=OVERLOADS["read_forward._apply_log_emission"], cache=True, nogil=True)
def _apply_log_emission(current, log_emission):
    maximum = np.max(log_emission)
    if not np.isfinite(maximum):
        return -np.inf
    classes = current.shape[0]
    states = current.shape[1]
    for first in range(states):
        for second in range(states):
            log_emission[first, second] = math.exp(log_emission[first, second] - maximum)
    for c in range(classes):
        for first in range(states):
            for second in range(states):
                current[c, first, second] *= log_emission[first, second]
    increment = _normalise(current)
    return increment + maximum if np.isfinite(increment) else -np.inf


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
    background_row=None,
    reuse_background_row=False,
):
    """The scalar bin emissions of every state pair into ``work`` [S, S] (forward.py's products: the cached ones where
    finite, the site order, likelihood and zero rule otherwise); returns False when a product underflows, ``work``
    then holding the log emissions of every pair. A tile can share the first parent's BACKGROUND by named-state
    products with the next row for the same child, bin and parent. The two BACKGROUND alleles remain pair-specific."""
    states = work.shape[0]
    stable_product = True
    for first in range(states):
        for second in range(states):
            if background_row is not None and reuse_background_row and first == states - 1 and second < states - 1:
                work[first, second] = background_row[second]
                continue
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
    if stable_product:
        if background_row is not None and not reuse_background_row:
            for second in range(states - 1):
                background_row[second] = work[states - 1, second]
    else:
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
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    common_products,
    first_cached_child,
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
    class_cache=None,
):
    """The forward log likelihood of one child under one configuration, its read class carried (``class_cache``:
    class_transfers of the child's batch starting at ``first_cached_child``, or None). The chain stops at its first
    non-finite increment."""
    states = state_alt.shape[0]
    bins = site_start.shape[1]
    classes = read_initial.shape[1]
    full = np.empty((classes, states, states), dtype=np.float64)
    mix = np.empty_like(full)
    work = np.empty((states, states), dtype=np.float64)
    scratch = np.empty_like(work)
    prefix = np.empty(states + 1, dtype=np.float64)
    suffix = np.empty(states + 1, dtype=np.float64)
    factor = np.empty(classes, dtype=np.float64)
    column = np.empty(classes, dtype=np.float64)
    moved = np.empty(classes, dtype=np.float64)
    first_source = first_parent if mode >= 1 else -1
    second_source = second_parent if mode == 2 else -1
    _initial_state(full, child, first_parent, second_parent, mode, pi, null_frequency, read_initial)
    full_log = 0.0
    for block in range(bins):
        if block > 0:
            advance_classes(full[:read_width[child, block]], scratch, child, first_parent, second_parent, mode,
                            block - 1, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch, null_diagonal,
                            null_off, null_frequency, prefix, suffix)
        stable_product = _bin_products(work, child, first_parent, second_parent, mode, block, candidate_alt, state_alt,
                                       child_gl, coefficient, site_index, site_start, site_stop, mismatch,
                                       common_products, first_cached_child)
        carry_bin(full, mix, factor, column, child, block, first_source, second_source, candidate_alt, state_alt,
                  child_gl, site_index, site_start, site_stop, read_offset, read_reset, read_width, read_operator,
                  read_ratio_table, class_cache, child - first_cached_child, moved)
        if stable_product:
            increment = _apply_probability_emission(full[:read_width[child, block + 1]], work)
        else:
            increment = _apply_log_emission(full[:read_width[child, block + 1]], work)
        if not np.isfinite(increment):
            return -np.inf
        full_log += increment
    return full_log


def read_arguments(t: ChildTerms) -> tuple:
    """The children's read-state arguments of the chain kernels (ChildTerms' read terms)."""
    return t.read_offset, t.read_initial, t.read_reset, t.read_width, t.read_operator, t.read_ratio
