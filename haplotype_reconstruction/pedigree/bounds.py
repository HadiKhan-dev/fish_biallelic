"""Private feasibility bounds: retain the parental K-state chain, relax the null allele.

For each parental state/site, minimum/maximum emission over the other allele bounds
all null-selector paths. At uncertain parental alleles the extrema include both
parental allele endpoints too. Products, positive transitions and the two original
view exponents preserve these inequalities. No biological cutoff or candidate cap.

The mathematical inequalities are exact. The implementation reports an explicit
conservative float64 operation-count pad, tested against the exact reference; this
is not a formally verified libm directed-rounding implementation. No production
release path uses this module. Fail any reference coverage violation.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from .forward import _site_likelihood

_EPS = np.finfo(np.float64).eps


@njit(cache=True, parallel=True)
def emission_envelopes(state_alt, gl, coefficient, indices, starts, stops, mismatch):
    samples, bins = starts.shape
    states = state_alt.shape[0]
    lower = np.zeros((samples, bins, states))
    upper = np.zeros_like(lower)
    for task in prange(samples * bins):
        child, block = task // bins, task % bins
        for slot in range(starts[child, block], stops[child, block]):
            site = indices[slot]
            corners = np.empty((2, 2))
            for a in range(2):
                for b in range(2):
                    corners[a, b] = _site_likelihood(gl[child, site], coefficient[child, site],
                                                     float(a), float(b), mismatch)
            # Absolute polynomial/fallback evaluation allowance, before taking logs.
            pad = 64 * _EPS * max(1., 3 * np.max(gl[child, site]))
            for state in range(states):
                allele = state_alt[state, site]
                if allele == 0. or allele == 1.:
                    lo = min(corners[int(allele), 0], corners[int(allele), 1])
                    hi = max(corners[int(allele), 0], corners[int(allele), 1])
                else:
                    lo, hi = np.min(corners), np.max(corners)
                lower[child, block, state] += math.log(max(lo - pad, 0.)) if lo > pad else -np.inf
                upper[child, block, state] += math.log(hi + pad)
    return lower, upper


@njit(cache=True)
def vector_bridge(current, stay, alpha, beta, row_arm, column_arm, pivot, branch, excursion):
    """The existing cancellation-free bridge on a single vector, O(K)."""
    n = len(current)
    out = np.empty(n)
    if branch == 3:
        left = right = 0.
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
        column = 0.
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
        out = (1. - excursion) * out + share
    return out


@njit(cache=True)
def _path(child, parent, emission, pi, stay, alpha, beta, row_arm, column_arm, pivot,
          branch, excursion, exponent, null_diagonal, null_off):
    bins, states = emission.shape[1:]
    message = np.empty((2, states))
    message[0] = pi[parent, 0]
    message[1] = pi[parent, 0]
    score = np.zeros(2)
    for block in range(bins):
        for view in range(2):
            if block:
                k = block - 1
                message[view] = vector_bridge(message[view], stay[parent,k], alpha[parent,k],
                    beta[parent,k], row_arm[parent,k], column_arm[parent,k], pivot[parent,k],
                    branch[parent,k], excursion[parent,k])
                # Null transition has a common row sum. Retain its computed mass,
                # rather than assume numerical row normalization is perfect.
                score[view] += math.log(null_diagonal[child,k] + (states-1)*null_off[child,k])
            eta = exponent[child,block] if view == 0 else 1.
            if eta != 0:
                maximum = np.max(emission[child,block]) * eta
                if not np.isfinite(maximum):
                    return -np.inf
                for state in range(states):
                    message[view,state] *= math.exp(eta*emission[child,block,state]-maximum)
                score[view] += maximum
            total = np.sum(message[view])
            if total <= 0:
                return -np.inf
            message[view] /= total
            score[view] += math.log(total)
    return .5*(score[0]+score[1])


@njit(cache=True, parallel=True)
def _batch(first, stop, low_emission, high_emission, pi, stay, alpha, beta, row_arm, column_arm,
           pivot, branch, excursion, exponent, null_diagonal, null_off, available, zero, sites):
    n = len(pi)
    lo = np.full((stop-first,n),-np.inf)
    hi = np.full_like(lo,-np.inf)
    pad = np.zeros_like(lo)
    for task in prange((stop-first)*n):
        c,p = first+task//n, task%n
        if c==p:
            continue
        if not available[p]:
            lo[c-first,p]=hi[c-first,p]=zero[c]
            continue
        lower = _path(c,p,low_emission,pi,stay,alpha,beta,row_arm,column_arm,pivot,branch,
                      excursion,exponent,null_diagonal,null_off)
        upper = _path(c,p,high_emission,pi,stay,alpha,beta,row_arm,column_arm,pivot,branch,
                      excursion,exponent,null_diagonal,null_off)
        finite_size = max(1.,abs(upper))
        if np.isfinite(lower):
            finite_size=max(finite_size,abs(lower))
        error = 512*_EPS*(sites + 8*pi.shape[1]*pi.shape[2])*finite_size
        lo[c-first,p]=lower-error
        hi[c-first,p]=upper+error
        pad[c-first,p]=error
    return lo,hi,pad


def screen_bounds(projection, terms, zero, claim_threads=lambda: None, batch=8, progress=None):
    """All-pair feasibility intervals. O(N M K + N² B K), still quadratic in N."""
    claim_threads()
    low,high=emission_envelopes(projection.state_alt,terms.gl,terms.coefficient,terms.site_index,
                               terms.site_start,terms.site_stop,terms.mismatch)
    n=len(projection.pi)
    lower=np.full((n,n),-np.inf);upper=lower.copy();padding=np.zeros_like(lower)
    for first in range(0,n,batch):
        stop=min(n,first+batch);claim_threads()
        lower[first:stop],upper[first:stop],padding[first:stop]=_batch(first,stop,low,high,
            projection.pi,projection.stay,projection.alpha,projection.beta,projection.row_arm,
            projection.column_arm,projection.pivot,projection.branch,projection.excursion,
            terms.exponent,terms.null_diagonal,terms.null_off,projection.available,zero,terms.gl.shape[1])
        if progress is not None:progress(stop,n)
    return lower,upper,padding
