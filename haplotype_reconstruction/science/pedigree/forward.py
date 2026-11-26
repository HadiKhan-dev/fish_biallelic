"""The child's scalar emission model, the configuration chains' transitions and caches, and ChildTerms (the
configuration chains themselves, which also carry the child's read state, are in read_forward).

The hidden state of a child at a bin is the ordered pair of transmitted founder states (S^2 states):
under M0 both come from the null source (a copying chain that starts from, and redraws from, the
leave-one-out frequency of the founder states the other samples transmit), under M1 the
first comes from the candidate parent's projected transmission (pi, bridges), under M2 both come from
the two parents. Emissions multiply the child's per-marker
likelihoods of the two transmitted alleles (with mismatch probability) over its observed markers
of the bin. Each configuration has one likelihood: every observed marker enters once, at full strength (4.22,
R4 1a: no tempered view, information exponent or pooled average).

Called founder alleles project to the same 0/1 value in every candidate, so their emission products
are cached per child and bin (NaN marks entries needing the candidate-specific path) in batches of
children bounded by a 256 MiB workspace.

Method: docs/methods/pedigree.md#child-forward-model
"""

from __future__ import annotations

import numpy as np
from numba import prange

from ...contracts.pedigree import ChildReads
from ...core.errors import InvariantError, SettingsError
from ...core.kernels import kernel
from .child_reads import class_layout, resident_terms
from .kernels.forward import OVERLOADS
from .numerical_parameters import NULL_ROUNDING_ULPS, SITE_POLYNOMIAL_THRESHOLD
from .projection import Projection

# Temporary per-call cache of shared emission products, tiled over children (not a setting).
from .workspace_parameters import CLASS_TRANSFER_CACHE_BYTES, COMMON_EMISSION_CACHE_BYTES

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

    A named state has the same ALT probability in every projection, even where a panel allele is missing.
    BACKGROUND (candidate-specific) takes the general path. Its site order, likelihood and underflow rule also
    apply here, so a finite cached product is the exact value the general path would have computed.
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
                    value = site_likelihood(child_gl[child, site], coefficient[child, site], a, b, mismatch)
                    if value <= 0.0:
                        product = 0.0
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
    to log space.
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
    site_stop), null transition coefficients [N, B-1] (diagonal, off), null_frequency float64 [N, B, S] (the null
    source's initial and redraw law per child and bin), mismatch and the cache batch size; the children's read state
    (child_reads.resident_terms of the component's ChildReads, one class without it): read_offset, read_initial,
    read_operator and read_ratio, carried by the configuration chains (read_forward).

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
        selector_switch_probability: np.ndarray,
        same: np.ndarray,
        one: np.ndarray,
        two: np.ndarray,
        null_prior_samples: float,
        reads: ChildReads | None = None,
    ):
        # ``gl`` are the children's mixed likelihoods (the painting's outlier rate against the cohort law,
        # science.shared.observation): a genotype redraw is in them, so the kernels' allele-flip term is zero (A8).
        mismatch = 0.0
        children = gl.shape[0]
        states = projection.n_states
        bins = projection.n_bins
        self.gl = gl
        self.coefficient = child_coefficients(gl, mismatch)
        self.site_index, self.site_start, self.site_stop = observed_sites(
            observed, projection.bin_start, projection.bin_stop
        )
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
        self.read_offset, self.read_initial, self.read_operator, self.read_ratio = resident_terms(
            reads, children, self.site_start, self.site_stop
        )
        self.n_classes = int(self.read_initial.shape[1])
        self.read_reset, self.read_width = class_layout(
            self.read_operator, self.read_offset, self.site_start, self.site_stop, self.n_classes)
        cache_children = COMMON_EMISSION_CACHE_BYTES // max(1, bins * states * states * 8)
        self.cached = cache_children > 0
        self.batch = min(children, cache_children) if cache_children else children
        # the read state's class-transfer cache (class_cache): one child batch within CLASS_TRANSFER_CACHE_BYTES, the
        # batch then shared with the scalar caches; none when one child's would not fit (scored per site)
        class_bytes = bins * states * states * self.n_classes * self.n_classes * 8 if self.n_classes > 1 else 0
        # Only the fixed-width own block admits a class matrix per state pair. The extended
        # chain keeps its O(C^2 + K) gap and ratio descriptors at each observed slot.
        fixed_width = self.read_operator.shape[1] == self.read_operator.shape[2]
        self.class_cached = fixed_width and 0 < class_bytes <= CLASS_TRANSFER_CACHE_BYTES
        if self.class_cached:
            self.batch = max(1, min(self.batch, CLASS_TRANSFER_CACHE_BYTES // class_bytes))
        # Dense reuse for short structured vectors, with bounded foreign width relative to the own block.
        # Wider chains keep structured gap actions, so the cache never adds a fourth power of founder count.
        compact_width = self.n_classes <= 2 * self.read_operator.shape[1]
        self.trio_batch = self.batch
        self.trio_class_cached = self.class_cached or (
            not fixed_width and compact_width and 0 < class_bytes <= CLASS_TRANSFER_CACHE_BYTES
        )
        if self.trio_class_cached:
            self.trio_batch = max(1, min(self.batch, CLASS_TRANSFER_CACHE_BYTES // class_bytes))

    def common(self, projection: Projection, child_start: int, child_stop: int):
        """The cached emission products of a child batch, or None without a cache."""
        if not self.cached:
            return None
        return _common_products(
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

    def class_cache(self, projection: Projection, child_start: int, child_stop: int, *, trios: bool = False):
        """The read-class transfer cache of this child batch, or None when it is disabled for this width.

        Ranking and block scoring can also request a bounded bank for short structured vectors (trios=True).
        """
        if not (self.trio_class_cached if trios else self.class_cached):
            return None
        from .read_transfer import class_transfers

        return class_transfers(
            int(child_start), int(child_stop), projection.state_alt, self.gl, self.site_index, self.site_start,
            self.site_stop, self.read_offset, self.read_reset, self.read_width, self.read_operator, self.read_ratio,
        )

    def common_null_second(self, projection: Projection, child_start: int, child_stop: int):
        """``common`` for M0/M1 configurations only (the M1 screen): the products also hold every entry with a named
        first state (_fill_named_first_products)."""
        products = self.common(projection, child_start, child_stop)
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
        return products


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


def terms_args(p, t):
    return (
        p.state_alt,
        t.gl,
        t.coefficient,
        t.site_index,
        t.site_start,
        t.site_stop,
        t.null_diagonal,
        t.null_off,
        t.null_frequency,
        t.mismatch,
    )
