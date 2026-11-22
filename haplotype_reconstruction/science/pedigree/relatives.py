"""Score and aggregate relative-proxy contrasts on prepared chromosome inputs.

Second-degree and full-sib forward models share the prepared resident projection and
public kernels, and carry the child's read state as its M1 and M2 chains do (read_forward). Availability gates only the candidate parent. Component order
is fixed; report aggregation uses the declared configured-contig sum.

Method: docs/methods/pedigree.md#relative-proxy-contrasts-of-released-edges
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numba
import numpy as np
from numba import prange

from ...contracts.pedigree import PreparedComponent, RelativeContrasts
from ...core.kernels import kernel
from ...core.numerics import ordered_sum
from . import resident
from .forward import apply_bridge, apply_excursion, apply_null, model_arguments, site_likelihood
from .kernels.relatives import OVERLOADS
from .parameters import PedigreeParameters
from .read_forward import read_arguments, score_path
from .read_transfer import carry_bin

__all__ = ["aggregate", "component_contrasts", "PROXY_MODEL"]
PROXY_MODEL = "relative-proxy-v1"


def aggregate(
    pairs: np.ndarray, views: tuple[tuple[str, ...], ...], second: np.ndarray, full: np.ndarray, usable: np.ndarray
) -> RelativeContrasts:
    """Reduce float64[C,Q] contrasts over usable contigs in configured order.

    No usable chromosome gives NaN contrasts, zero counts. Negative contrasts use
    a strict zero cut-off, solely for the report. Inputs are borrowed read-only.
    """
    n = len(pairs)
    d2, df, minimum = (np.full(n, np.nan) for _ in range(3))
    count, n2, nf = (np.zeros(n, dtype=np.int64) for _ in range(3))
    for edge in range(n):
        use = usable[:, edge]
        a, b = second[use, edge], full[use, edge]
        count[edge] = len(a)
        n2[edge], nf[edge] = np.count_nonzero(a < 0), np.count_nonzero(b < 0)
        if len(a):
            d2[edge], df[edge], minimum[edge] = ordered_sum(a), ordered_sum(b), float(a.min())
    return RelativeContrasts(np.array(pairs, dtype=np.int64, copy=True), tuple(views), d2, df, count, n2, nf, minimum)


@kernel(overloads=OVERLOADS["relatives._second_degree"], cache=True, nogil=True)
def _second_degree(
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
    gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    exponent,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    selector,
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
    read_powered,
    class_cache,
    first_child,
    view,
):
    """Second-degree relative proxy: one homologue of the child is IBD with the candidate's transmission chain in
    tracts (stationary 1/2, flipping with the meiotic selector probability per bin edge), null outside, the other
    homologue null; the child's read class carried as in M1 (read_transfer.carry_bin). Composite score (1/2 tempered +
    1/2 full), as M1."""
    states = state_alt.shape[0]
    bins = exponent.shape[1]
    classes = read_initial.shape[1]
    prefix = np.empty(states + 1)
    suffix = np.empty(states + 1)
    mix = np.empty((classes, states, states))
    factor = np.empty(classes)
    column = np.empty(classes)
    moved = np.empty(classes)
    layer = np.zeros((2, classes, states, states))
    work = np.empty((states, states))
    for c in range(classes):
        for a in range(states):
            for b in range(states):
                layer[1, c, a, b] = 0.5 * pi[parent, 0, a] * null_frequency[child, 0, b] * read_initial[child, c]
                layer[0, c, a, b] = (
                    0.5 * null_frequency[child, 0, a] * null_frequency[child, 0, b] * read_initial[child, c]
                )
    total_log = 0.0
    for block in range(bins):
        if block > 0:
            k = block - 1
            q = selector[k]
            for c in range(read_width[child, block]):
                m1 = np.zeros(states)
                m0 = np.zeros(states)
                for a in range(states):
                    for b in range(states):
                        m1[b] += layer[1, c, a, b]
                        m0[b] += layer[0, c, a, b]
                # IBD layer: p's bridge (+ excursion) on axis 0
                apply_bridge(
                    layer[1, c],
                    work,
                    stay[parent, k],
                    alpha[parent, k],
                    beta[parent, k],
                    row_arm[parent, k],
                    column_arm[parent, k],
                    pivot[parent, k],
                    0,
                    prefix,
                    suffix,
                    branch[parent, k],
                )
                if excursion[parent, k] > 0.0:
                    apply_excursion(layer[1, c], work, excursion[parent, k], 0)
                ibd = work.copy()
                # non-IBD layer: null on axis 0
                apply_null(
                    layer[0, c], work, null_diagonal[child, k], null_off[child, k], null_frequency[child, k + 1], 0
                )
                non = work.copy()
                for a in range(states):
                    for b in range(states):
                        layer[1, c, a, b] = (1 - q) * ibd[a, b] + q * pi[parent, k + 1, a] * m0[b]
                        layer[0, c, a, b] = (1 - q) * non[a, b] + q * null_frequency[child, k + 1, a] * m1[b]
                for z in range(2):
                    apply_null(
                        layer[z, c], work, null_diagonal[child, k], null_off[child, k],
                        null_frequency[child, k + 1], 1
                    )
                    layer[z, c] = work.copy()
        eta = exponent[child, block] if view == 0 else 1.0
        for z in range(2):
            carry_bin(layer[z], mix, factor, column, child, block, parent if z == 1 else -1, -1, candidate_alt,
                      state_alt, gl, site_index, site_start, site_stop, read_offset, read_reset, read_width, read_operator,
                      read_ratio_table, read_powered, view == 0, eta, class_cache, child - first_child, moved)
        # emissions (log space) per layer
        peak = -np.inf
        logs = np.empty((2, states, states))
        for z in range(2):
            for a in range(states):
                for b in range(states):
                    v = 0.0
                    for slot in range(site_start[child, block], site_stop[child, block]):
                        site = site_index[slot]
                        first = candidate_alt[parent, site, a] if z == 1 else state_alt[a, site]
                        x = site_likelihood(
                            gl[child, site], coefficient[child, site], first, state_alt[b, site], mismatch
                        )
                        if x <= 0.0:
                            v = -np.inf
                            break
                        v += math.log(x)
                    logs[z, a, b] = eta * v
                    if logs[z, a, b] > peak:
                        peak = logs[z, a, b]
        s = 0.0
        for z in range(2):
            for a in range(states):
                for b in range(states):
                    value = math.exp(logs[z, a, b] - peak) if np.isfinite(logs[z, a, b]) else 0.0
                    for c in range(read_width[child, block + 1]):
                        layer[z, c, a, b] *= value
                        s += layer[z, c, a, b]
        if s <= 0.0:
            total_log = -np.inf
            break
        layer[:, :read_width[child, block + 1]] /= s
        total_log += peak + math.log(s)
    return total_log


@kernel(overloads=OVERLOADS["relatives._full_sib"], cache=True, nogil=True)
def _full_sib(
    child,
    donor,
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
    site_index,
    site_start,
    site_stop,
    exponent,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    selector,
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
    read_powered,
    class_cache,
    first_child,
    view,
):
    """Full-sib relationship model: each of the child's homologues is IBD with the donor's transmission chain
    in its own tracts (stationary 1/2, flipping with probability 2 theta (1 - theta) per bin edge, two meioses), null
    outside; the child's read class carried as in M2 (read_transfer.carry_bin). Composite score (1/2 tempered + 1/2
    full)."""
    states = state_alt.shape[0]
    bins = exponent.shape[1]
    classes = read_initial.shape[1]
    prefix = np.empty(states + 1)
    suffix = np.empty(states + 1)
    mix = np.empty((classes, states, states))
    factor = np.empty(classes)
    column = np.empty(classes)
    moved = np.empty(classes)
    layer = np.zeros((2, 2, classes, states, states))
    work = np.empty((states, states))
    for y1 in range(2):
        for y2 in range(2):
            for a in range(states):
                for b in range(states):
                    f1 = pi[donor, 0, a] if y1 == 1 else null_frequency[child, 0, a]
                    f2 = pi[donor, 0, b] if y2 == 1 else null_frequency[child, 0, b]
                    for c in range(classes):
                        layer[y1, y2, c, a, b] = 0.25 * f1 * f2 * read_initial[child, c]
    total_log = 0.0
    for block in range(bins):
        if block > 0:
            k = block - 1
            q = min(0.5, 2.0 * selector[k] * (1.0 - selector[k]))
            for y1 in range(2):
                for y2 in range(2):
                    for c in range(read_width[child, block]):
                        cur = layer[y1, y2, c]
                        if y1 == 1:
                            apply_bridge(
                                cur,
                                work,
                                stay[donor, k],
                                alpha[donor, k],
                                beta[donor, k],
                                row_arm[donor, k],
                                column_arm[donor, k],
                                pivot[donor, k],
                                0,
                                prefix,
                                suffix,
                                branch[donor, k],
                            )
                            if excursion[donor, k] > 0.0:
                                apply_excursion(cur, work, excursion[donor, k], 0)
                        else:
                            apply_null(
                                cur, work, null_diagonal[child, k], null_off[child, k],
                                null_frequency[child, k + 1], 0
                            )
                        cur2 = work.copy()
                        if y2 == 1:
                            apply_bridge(
                                cur2,
                                work,
                                stay[donor, k],
                                alpha[donor, k],
                                beta[donor, k],
                                row_arm[donor, k],
                                column_arm[donor, k],
                                pivot[donor, k],
                                1,
                                prefix,
                                suffix,
                                branch[donor, k],
                            )
                            if excursion[donor, k] > 0.0:
                                apply_excursion(cur2, work, excursion[donor, k], 1)
                        else:
                            apply_null(
                                cur2, work, null_diagonal[child, k], null_off[child, k],
                                null_frequency[child, k + 1], 1
                            )
                        layer[y1, y2, c] = work.copy()
            # IBD switching per slot: entering IBD draws from the donor's marginal, leaving from the null
            new = np.zeros_like(layer)
            for y1 in range(2):
                for y2 in range(2):
                    t1 = 1 - y1
                    t2 = 1 - y2
                    for c in range(read_width[child, block]):
                        m_rows = np.zeros(states)
                        m_cols = np.zeros(states)
                        for a in range(states):
                            for b in range(states):
                                m_rows[b] += layer[y1, y2, c, a, b]
                                m_cols[a] += layer[y1, y2, c, a, b]
                        for a in range(states):
                            for b in range(states):
                                new[y1, y2, c, a, b] += (1 - q) * (1 - q) * layer[y1, y2, c, a, b]
                        # slot 1 flips
                        for a in range(states):
                            g = pi[donor, k + 1, a] if t1 == 1 else null_frequency[child, k + 1, a]
                            for b in range(states):
                                new[t1, y2, c, a, b] += q * (1 - q) * g * m_rows[b]
                        # slot 2 flips
                        for b in range(states):
                            g = pi[donor, k + 1, b] if t2 == 1 else null_frequency[child, k + 1, b]
                            for a in range(states):
                                new[y1, t2, c, a, b] += q * (1 - q) * g * m_cols[a]
                        # both flip
                        tot = 0.0
                        for a in range(states):
                            for b in range(states):
                                tot += layer[y1, y2, c, a, b]
                        for a in range(states):
                            g1 = pi[donor, k + 1, a] if t1 == 1 else null_frequency[child, k + 1, a]
                            for b in range(states):
                                g2 = pi[donor, k + 1, b] if t2 == 1 else null_frequency[child, k + 1, b]
                                new[t1, t2, c, a, b] += q * q * g1 * g2 * tot
            layer = new
        eta = exponent[child, block] if view == 0 else 1.0
        for y1 in range(2):
            for y2 in range(2):
                carry_bin(layer[y1, y2], mix, factor, column, child, block, donor if y1 == 1 else -1,
                          donor if y2 == 1 else -1, candidate_alt, state_alt, gl, site_index, site_start,
                          site_stop, read_offset, read_reset, read_width, read_operator, read_ratio_table, read_powered, view == 0, eta,
                          class_cache, child - first_child, moved)
        peak = -np.inf
        logs = np.empty((2, 2, states, states))
        for y1 in range(2):
            for y2 in range(2):
                for a in range(states):
                    for b in range(states):
                        v = 0.0
                        for slot in range(site_start[child, block], site_stop[child, block]):
                            site = site_index[slot]
                            f1 = candidate_alt[donor, site, a] if y1 == 1 else state_alt[a, site]
                            f2 = candidate_alt[donor, site, b] if y2 == 1 else state_alt[b, site]
                            x = site_likelihood(gl[child, site], coefficient[child, site], f1, f2, mismatch)
                            if x <= 0.0:
                                v = -np.inf
                                break
                            v += math.log(x)
                        logs[y1, y2, a, b] = eta * v
                        if logs[y1, y2, a, b] > peak:
                            peak = logs[y1, y2, a, b]
        s = 0.0
        for y1 in range(2):
            for y2 in range(2):
                for a in range(states):
                    for b in range(states):
                        val = logs[y1, y2, a, b]
                        value = math.exp(val - peak) if np.isfinite(val) else 0.0
                        for c in range(read_width[child, block + 1]):
                            layer[y1, y2, c, a, b] *= value
                            s += layer[y1, y2, c, a, b]
        if s <= 0.0:
            total_log = -np.inf
            break
        layer[:, :, :read_width[child, block + 1]] /= s
        total_log += peak + math.log(s)
    return total_log


@kernel(overloads=OVERLOADS["relatives._proxy_kernel"], cache=True, parallel=True, nogil=True)
def _proxy_kernel(
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
    state_alt,
    gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    exponent,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    selector,
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
    read_powered,
    class_cache,
    first_child,
):
    scores = np.empty((len(pairs), 2, 2))
    # Each proxy has independent tempered and full chains. A batch of 30 pairs yields 120 independent
    # chains before the fixed two-view average. At 19 threads, famhs/bc8 chr19 went from .76/.74
    # core use to .83/.81. The marker recursion within each chain is sequential.
    for unit in prange(4 * len(pairs)):
        i = unit // 4
        view = np.int64((unit // 2) % 2)
        if unit % 2 == 0:
            scores[i, 0, view] = _second_degree(
                pairs[i, 0],
                pairs[i, 1],
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
                site_index,
                site_start,
                site_stop,
                exponent,
                null_diagonal,
                null_off,
                null_frequency,
                mismatch,
                selector,
                read_offset,
                read_initial,
                read_reset, read_width, read_operator,
                read_ratio_table,
                read_powered,
                class_cache,
                first_child,
                view,
            )
        else:
            scores[i, 1, view] = _full_sib(
                pairs[i, 0],
                pairs[i, 1],
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
                site_index,
                site_start,
                site_stop,
                exponent,
                null_diagonal,
                null_off,
                null_frequency,
                mismatch,
                selector,
                read_offset,
                read_initial,
                read_reset, read_width, read_operator,
                read_ratio_table,
                read_powered,
                class_cache,
                first_child,
                view,
            )
    second = np.empty(len(pairs))
    full = np.empty(len(pairs))
    for i in range(len(pairs)):
        second[i] = 0.5 * (scores[i, 0, 0] + scores[i, 0, 1])
        full[i] = 0.5 * (scores[i, 1, 0] + scores[i, 1, 1])
    return second, full


@kernel(overloads=OVERLOADS["relatives._m1_pairs"], cache=True, parallel=True, nogil=True)
def _m1_pairs(
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
    state_alt,
    gl,
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
    class_cache,
):
    out = np.empty(len(pairs))
    for i in prange(len(pairs)):
        tempered, full = score_path(
            pairs[i, 0],
            pairs[i, 1],
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
            class_cache,
        )
        out[i] = 0.5 * (tempered + full)
    return out


def component_contrasts(
    component: PreparedComponent,
    config: PedigreeParameters,
    pairs: np.ndarray,
    claim_threads: Callable[[], int | None] = lambda: None,
    *,
    reverse: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(M1 - second-degree proxy, M1 - full-sib proxy, usable) float64/float64/bool [E] of one prepared component
    for the (child, candidate) ``pairs`` int64 [E, 2]
    zero where the candidate has no evidence here."""
    pairs = np.ascontiguousarray(pairs, dtype=np.int64)
    second = np.zeros(len(pairs))
    full = np.zeros(len(pairs))
    state = resident.resident(component, config, claim_threads, reverse=reverse, include_pair_masses=False)
    projection, terms = state.projection, state.terms
    usable = np.asarray(projection.available, dtype=bool)[pairs[:, 1]] if len(pairs) else np.zeros(0, bool)
    rows = np.flatnonzero(usable)
    if not len(rows):
        return second, full, usable
    common = (
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
    m1 = np.zeros(len(pairs))
    proxy_second = np.zeros(len(pairs))
    proxy_full = np.zeros(len(pairs))
    selector = np.ascontiguousarray(component.selector_switch_probability)
    children = terms.gl.shape[0]
    for child_start in range(0, children, terms.batch):
        child_stop = min(children, child_start + terms.batch)
        batch = rows[(pairs[rows, 0] >= child_start) & (pairs[rows, 0] < child_stop)]
        if not len(batch):
            continue
        claim_threads()
        products, powered = terms.common(projection, child_start, child_stop)
        classes = terms.class_cache(projection, child_start, child_stop)
        claim_threads()
        chosen = np.ascontiguousarray(pairs[batch])
        # a batch holds few pairs of unequal cost: hand the units out one at a time
        with numba.parallel_chunksize(1):
            m1[batch] = _m1_pairs(
                chosen, *model_arguments(projection), *common, products, child_start, powered,
                *read_arguments(terms), classes,
            )
            claim_threads()
            # the proxies read the batch's class transfers too (read_transfer.carry_bin)
            proxy_second[batch], proxy_full[batch] = _proxy_kernel(
                chosen, *model_arguments(projection), *common, selector, *read_arguments(terms), classes,
                child_start,
            )
        del products, powered, classes
    second[rows] = m1[rows] - proxy_second[rows]
    full[rows] = m1[rows] - proxy_full[rows]
    return second, full, usable
