"""Block certificates: exact null slot and relaxed uncertain parental alleles.
Method: docs/methods/pedigree.md#candidate-panel
"""

import math

import numpy as np
from numba import prange

from ...core.kernels import kernel
from .forward import site_likelihood
from .kernels.signatures import OVERLOADS
from .parameters import CERTIFICATE_GL_ULPS

EPS = np.finfo(float).eps


@kernel(overloads=OVERLOADS["certificates.vector_bridge"], cache=True, nogil=True)
def vector_bridge(current, stay, alpha, beta, row_arm, column_arm, pivot, branch, excursion):
    """The existing cancellation-free bridge on a single vector, O(K)."""
    n = len(current)
    out = np.empty(n)
    if branch == 3:
        left = right = 0.0
        for source in range(pivot):
            left += current[source] * alpha[source]
        for source in range(n - 1, pivot, -1):
            right += current[source] * alpha[source]
        for dest in range(n):
            value = stay[dest] * current[dest]
            if dest == pivot:
                value += left + right
            out[dest] = value + current[pivot] * row_arm[dest]
    else:
        prefix = np.zeros(n + 1)
        suffix = np.zeros(n + 1)
        for source in range(n):
            prefix[source + 1] = prefix[source] + current[source] * alpha[source]
        for source in range(n - 1, -1, -1):
            suffix[source] = suffix[source + 1] + current[source] * alpha[source]
        column = 0.0
        if pivot >= 0:
            for source in range(n):
                column += current[source] * column_arm[source]
        for dest in range(n):
            value = stay[dest] * current[dest] + beta[dest] * (prefix[dest] + suffix[dest + 1])
            if pivot >= 0:
                value += current[pivot] * row_arm[dest]
                if dest == pivot:
                    value += column
            out[dest] = value
    if excursion > 0:
        share = excursion * np.sum(current) / n
        out = (1.0 - excursion) * out + share
    return out


@kernel(overloads=OVERLOADS["certificates.dense_bridges"], cache=True, parallel=True, nogil=True)
def dense_bridges(stay, alpha, beta, row_arm, column_arm, pivot, branch, excursion):
    n, bins, k = stay.shape
    out = np.empty((n, bins, k, k))
    for item in prange(n * bins):
        p, b = item // bins, item % bins
        for source in range(k):
            unit = np.zeros(k)
            unit[source] = 1.0
            out[p, b, source] = vector_bridge(
                unit,
                stay[p, b],
                alpha[p, b],
                beta[p, b],
                row_arm[p, b],
                column_arm[p, b],
                pivot[p, b],
                branch[p, b],
                excursion[p, b],
            )
    return out


@kernel(overloads=OVERLOADS["certificates.power_common"], cache=True, parallel=True, nogil=True)
def power_common(common, children, exponent):
    count, bins, k, _ = common.shape
    powered = np.empty((count, bins, 2, k, k))
    scales = np.zeros((count, bins, 2))
    for task in prange(count * bins):
        ci, b = task // bins, task % bins
        maximum = -np.inf
        for s in range(k):
            for t in range(k):
                x = common[ci, b, s, t]
                if np.isfinite(x):
                    maximum = max(maximum, x)
        if not np.isfinite(maximum):
            maximum = 0.0
        for view in range(2):
            eta = exponent[children[ci], b] if view == 0 else 1.0
            scales[ci, b, view] = eta * maximum
            for s in range(k):
                for t in range(k):
                    x = common[ci, b, s, t]
                    powered[ci, b, view, s, t] = math.exp(eta * (x - maximum)) if np.isfinite(x) else np.nan
    return powered, scales


@kernel(overloads=OVERLOADS["certificates.block_upper"], cache=True, parallel=True, nogil=True)
def block_upper(
    children,
    first,
    stop,
    refs,
    tv,
    mass,
    ref_mass,
    null_diag,
    null_off,
    null_frequency,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    branch,
    excursion,
    powered,
    scales,
):
    """Apply the finite global emission envelope; no unreachable allele-envelope fallback."""
    groups = len(refs)
    k = powered.shape[3]
    length = stop - first
    output = np.empty((len(children), groups))
    for task in prange(len(children) * groups):
        ci, g = task // groups, task % groups
        c = children[ci]
        p = refs[g]
        emission = np.empty((length, 2, k, k))
        for b in range(first, stop):
            for s in range(k):
                for t in range(k):
                    emission[b - first, 0, s, t] = powered[ci, b, 0, s, t]
                    emission[b - first, 1, s, t] = powered[ci, b, 1, s, t]
        score = 0.0
        value = np.empty((k, k))
        work = np.empty_like(value)
        ref = np.empty_like(value)
        prefix = np.empty((k + 1, k))
        suffix = np.empty_like(prefix)
        arm = np.empty(k)
        lo_v = np.empty(k)
        hi_v = np.empty(k)
        sum_v = np.empty(k)
        for view in range(2):
            value[:] = 1.0
            for b in range(stop - 1, first - 1, -1):
                local = b - first
                for s in range(k):
                    for t in range(k):
                        value[s, t] *= emission[local, view, s, t]
                score += 0.5 * scales[ci, b, view]
                if b:
                    # Null slot (exact): stay with nd - no >= 0 (ChildTerms checks), else move to a state drawn
                    # from the child's null frequencies at bin b; every term is nonnegative.
                    keep = null_diag[c, b - 1] - null_off[c, b - 1]
                    move = k * null_off[c, b - 1]
                    for s in range(k):
                        total = 0.0
                        for t in range(k):
                            total += null_frequency[c, b, t] * value[s, t]
                        for t in range(k):
                            work[s, t] = keep * value[s, t] + move * total
                    boundary = b - 1
                    piv = pivot[p, boundary]
                    kind = branch[p, boundary]
                    for t in range(k):
                        lo_v[t] = np.inf
                        hi_v[t] = 0.0
                        sum_v[t] = 0.0
                        arm[t] = 0.0
                        for s in range(k):
                            lo_v[t] = min(lo_v[t], work[s, t])
                            hi_v[t] = max(hi_v[t], work[s, t])
                            sum_v[t] += work[s, t]
                            if piv >= 0:
                                arm[t] += row_arm[p, boundary, s] * work[s, t]
                    if kind == 3:
                        for s in range(k):
                            for t in range(k):
                                ref[s, t] = stay[p, boundary, s] * work[s, t]
                                if s == piv:
                                    ref[s, t] += arm[t]
                                else:
                                    ref[s, t] += alpha[p, boundary, s] * work[piv, t]
                    else:
                        prefix[0, :] = 0.0
                        suffix[k, :] = 0.0
                        for s in range(k):
                            for t in range(k):
                                prefix[s + 1, t] = prefix[s, t] + beta[p, boundary, s] * work[s, t]
                        for s in range(k - 1, -1, -1):
                            for t in range(k):
                                suffix[s, t] = suffix[s + 1, t] + beta[p, boundary, s] * work[s, t]
                        for s in range(k):
                            for t in range(k):
                                x = stay[p, boundary, s] * work[s, t] + alpha[p, boundary, s] * (
                                    prefix[s, t] + suffix[s + 1, t]
                                )
                                if piv >= 0:
                                    x += column_arm[p, boundary, s] * work[piv, t]
                                    if s == piv:
                                        x += arm[t]
                                ref[s, t] = x
                    ex = excursion[p, boundary]
                    for s in range(k):
                        for t in range(k):
                            r = ref[s, t]
                            if ex > 0.0:
                                r = (1 - ex) * r + ex * sum_v[t] / k
                            bounded = r / ref_mass[g, local, s] + tv[g, local, s] * (hi_v[t] - lo_v[t])
                            value[s, t] = mass[g, local, s] * min(hi_v[t], bounded)
                maximum = np.max(value)
                if maximum <= 0.0 or not np.isfinite(maximum):
                    score = np.inf
                    break
                value /= maximum
                score += 0.5 * math.log(maximum)
        output[ci, g] = score
    return output


@kernel(overloads=OVERLOADS["certificates.global_common_logs"], cache=True, parallel=True, nogil=True)
def global_common_logs(children, state_alt, gl, coefficient, indices, starts, stops, mismatch):
    """Relax only uncertain parental alleles to [0,1]; retain the null chain.

    All groups share the complete emission operator. Exact known
    founder alleles stay fixed. This is a weaker bound, never a point model.
    """
    bins = starts.shape[1]
    k = len(state_alt)
    result = np.empty((len(children), bins, k, k))
    for item in prange(len(children) * bins):
        ci, b = item // bins, item % bins
        c = children[ci]
        for s in range(k):
            for t in range(k):
                total = 0.0
                for slot in range(starts[c, b], stops[c, b]):
                    site = indices[slot]
                    a = state_alt[s, site]
                    other = state_alt[t, site]
                    if a == 0.0 or a == 1.0:
                        likelihood = site_likelihood(gl[c, site], coefficient[c, site], a, other, mismatch)
                    else:
                        likelihood = max(
                            site_likelihood(gl[c, site], coefficient[c, site], 0.0, other, mismatch),
                            site_likelihood(gl[c, site], coefficient[c, site], 1.0, other, mismatch),
                        )
                    total += math.log(likelihood + CERTIFICATE_GL_ULPS * EPS * max(1.0, 3 * np.max(gl[c, site])))
                result[ci, b, s, t] = total
    return result
