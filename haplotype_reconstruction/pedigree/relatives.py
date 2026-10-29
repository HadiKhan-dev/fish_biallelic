"""Released-edge relative-proxy contrasts (release 2.17): decision-free diagnostics of a parent call.

For each released child-candidate edge and each scored chromosome: the production M1 score (the candidate as the
child's observed parent, the other slot null; forward._score_path, the screen's kernel) minus each of two
relationship proxies with simplifying IBD assumptions:

- second degree: one homologue of the child is IBD with the candidate's transmission chain in tracts (stationary
  1/2, switching with the meiotic selector probability per bin edge), the other homologue null;
- full sib: each homologue is IBD with the candidate's chain in its own tracts (stationary 1/2, flipping with
  probability 2 theta (1 - theta) per bin edge), null outside.

Every score is the composite of the tempered and the full genetic view, as M1 and M2. A chromosome counts as usable
when the candidate has evidence in at least one of its components (elsewhere every model is the child's M0 and the
contrast is zero). These are uncalibrated composite genetic-score contrasts conditional on the paintings: positive
favours the candidate-parent model over the proxy; they change no call or tier and are not posterior parentage
probabilities, and the proxies are not exhaustive alternatives (no inbreeding loops, no other relationships).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit, prange

from .forward import _apply_bridge, _apply_excursion, _apply_null, _model, _score_path, _site_likelihood

PROXY_MODEL = "relative-proxy-v1"
COLUMNS = ("Sample", "CandidateParent", "Views", "ProxyModel", "M1MinusSecondDegreeProxyNats",
           "M1MinusFullSibProxyNats", "UsableChromosomes", "NegativeChromosomesSecondDegree",
           "NegativeChromosomesFullSib", "MinChromosomeContrastSecondDegree")


@njit(cache=True)
def _second_degree(child, parent, pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
              state_alt, gl, coefficient, site_index, site_start, site_stop, exponent, null_diagonal, null_off,
              null_frequency, mismatch, selector):
    """Second-degree relative proxy: one homologue of the child is IBD with the candidate's transmission chain in
    tracts (stationary 1/2, flipping with the meiotic selector probability per bin edge), null outside
    the other
    homologue null. Composite score (1/2 tempered + 1/2 full), as M1."""
    states = state_alt.shape[0]
    bins = exponent.shape[1]
    out = np.zeros(2)
    prefix = np.empty(states + 1)
    suffix = np.empty(states + 1)
    for view in range(2):
        layer = np.zeros((2, states, states))
        work = np.empty((states, states))
        for a in range(states):
            for b in range(states):
                layer[1, a, b] = 0.5 * pi[parent, 0, a] * null_frequency[child, 0, b]
                layer[0, a, b] = 0.5 * null_frequency[child, 0, a] * null_frequency[child, 0, b]
        total_log = 0.0
        for block in range(bins):
            if block > 0:
                k = block - 1
                q = selector[k]
                m1 = np.zeros(states)
                m0 = np.zeros(states)
                for a in range(states):
                    for b in range(states):
                        m1[b] += layer[1, a, b]
                        m0[b] += layer[0, a, b]
                # IBD layer: p's bridge (+ excursion) on axis 0
                _apply_bridge(layer[1], work, stay[parent, k], alpha[parent, k], beta[parent, k], row_arm[parent, k],
                              column_arm[parent, k], pivot[parent, k], 0, prefix, suffix, branch[parent, k])
                if excursion[parent, k] > 0.0:
                    _apply_excursion(layer[1], work, excursion[parent, k], 0)
                ibd = work.copy()
                # non-IBD layer: null on axis 0
                _apply_null(layer[0], work, null_diagonal[child, k], null_off[child, k], null_frequency[child, k + 1], 0)
                non = work.copy()
                for a in range(states):
                    for b in range(states):
                        layer[1, a, b] = (1 - q) * ibd[a, b] + q * pi[parent, k + 1, a] * m0[b]
                        layer[0, a, b] = (1 - q) * non[a, b] + q * null_frequency[child, k + 1, a] * m1[b]
                for z in range(2):
                    _apply_null(layer[z], work, null_diagonal[child, k], null_off[child, k],
                                null_frequency[child, k + 1], 1)
                    layer[z] = work.copy()
            eta = exponent[child, block] if view == 0 else 1.0
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
                            x = _site_likelihood(gl[child, site], coefficient[child, site], first,
                                                 state_alt[b, site], mismatch)
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
                        layer[z, a, b] *= math.exp(logs[z, a, b] - peak) if np.isfinite(logs[z, a, b]) else 0.0
                        s += layer[z, a, b]
            if s <= 0.0:
                total_log = -np.inf
                break
            layer /= s
            total_log += peak + math.log(s)
        out[view] = total_log
    return 0.5 * (out[0] + out[1])




@njit(cache=True)
def _full_sib(child, donor, pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
              state_alt, gl, coefficient, site_index, site_start, site_stop, exponent, null_diagonal, null_off,
              null_frequency, mismatch, selector):
    """Full-sib relationship model: each of the child's homologues is IBD with the donor's transmission chain
    in its own tracts (stationary 1/2, flipping with probability 2 theta (1 - theta) per bin edge, two meioses), null
    outside. Composite score (1/2 tempered + 1/2 full)."""
    states = state_alt.shape[0]
    bins = exponent.shape[1]
    out = np.zeros(2)
    prefix = np.empty(states + 1)
    suffix = np.empty(states + 1)
    for view in range(2):
        layer = np.zeros((2, 2, states, states))
        work = np.empty((states, states))
        for y1 in range(2):
            for y2 in range(2):
                for a in range(states):
                    for b in range(states):
                        f1 = pi[donor, 0, a] if y1 == 1 else null_frequency[child, 0, a]
                        f2 = pi[donor, 0, b] if y2 == 1 else null_frequency[child, 0, b]
                        layer[y1, y2, a, b] = 0.25 * f1 * f2
        total_log = 0.0
        for block in range(bins):
            if block > 0:
                k = block - 1
                q = min(0.5, 2.0 * selector[k] * (1.0 - selector[k]))
                for y1 in range(2):
                    for y2 in range(2):
                        cur = layer[y1, y2]
                        if y1 == 1:
                            _apply_bridge(cur, work, stay[donor, k], alpha[donor, k], beta[donor, k], row_arm[donor, k],
                                          column_arm[donor, k], pivot[donor, k], 0, prefix, suffix, branch[donor, k])
                            if excursion[donor, k] > 0.0:
                                _apply_excursion(cur, work, excursion[donor, k], 0)
                        else:
                            _apply_null(cur, work, null_diagonal[child, k], null_off[child, k],
                                        null_frequency[child, k + 1], 0)
                        cur2 = work.copy()
                        if y2 == 1:
                            _apply_bridge(cur2, work, stay[donor, k], alpha[donor, k], beta[donor, k], row_arm[donor, k],
                                          column_arm[donor, k], pivot[donor, k], 1, prefix, suffix, branch[donor, k])
                            if excursion[donor, k] > 0.0:
                                _apply_excursion(cur2, work, excursion[donor, k], 1)
                        else:
                            _apply_null(cur2, work, null_diagonal[child, k], null_off[child, k],
                                        null_frequency[child, k + 1], 1)
                        layer[y1, y2] = work.copy()
                # IBD switching per slot: entering IBD draws from the donor's marginal, leaving from the null
                new = np.zeros_like(layer)
                for y1 in range(2):
                    for y2 in range(2):
                        m_rows = np.zeros(states)
                        m_cols = np.zeros(states)
                        for a in range(states):
                            for b in range(states):
                                m_rows[b] += layer[y1, y2, a, b]
                                m_cols[a] += layer[y1, y2, a, b]
                        for a in range(states):
                            for b in range(states):
                                v = layer[y1, y2, a, b]
                                new[y1, y2, a, b] += (1 - q) * (1 - q) * v
                        # slot 1 flips
                        t1 = 1 - y1
                        for a in range(states):
                            g = pi[donor, k + 1, a] if t1 == 1 else null_frequency[child, k + 1, a]
                            for b in range(states):
                                new[t1, y2, a, b] += q * (1 - q) * g * m_rows[b]
                        # slot 2 flips
                        t2 = 1 - y2
                        for b in range(states):
                            g = pi[donor, k + 1, b] if t2 == 1 else null_frequency[child, k + 1, b]
                            for a in range(states):
                                new[y1, t2, a, b] += q * (1 - q) * g * m_cols[a]
                        # both flip
                        tot = 0.0
                        for a in range(states):
                            for b in range(states):
                                tot += layer[y1, y2, a, b]
                        for a in range(states):
                            g1 = pi[donor, k + 1, a] if t1 == 1 else null_frequency[child, k + 1, a]
                            for b in range(states):
                                g2 = pi[donor, k + 1, b] if t2 == 1 else null_frequency[child, k + 1, b]
                                new[t1, t2, a, b] += q * q * g1 * g2 * tot
                layer = new
            eta = exponent[child, block] if view == 0 else 1.0
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
                                x = _site_likelihood(gl[child, site], coefficient[child, site], f1, f2, mismatch)
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
                            layer[y1, y2, a, b] *= math.exp(val - peak) if np.isfinite(val) else 0.0
                            s += layer[y1, y2, a, b]
            if s <= 0.0:
                total_log = -np.inf
                break
            layer /= s
            total_log += peak + math.log(s)
        out[view] = total_log
    return 0.5 * (out[0] + out[1])




@njit(cache=True, parallel=True)
def _proxy_kernel(pairs, pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
                  state_alt, gl, coefficient, site_index, site_start, site_stop, exponent, null_diagonal,
                  null_off, null_frequency, mismatch, selector):
    second = np.empty(len(pairs))
    full = np.empty(len(pairs))
    for i in prange(len(pairs)):
        second[i] = _second_degree(pairs[i, 0], pairs[i, 1], pi, candidate_alt, stay, alpha, beta, row_arm,
                                   column_arm, pivot, excursion, branch, state_alt, gl, coefficient, site_index,
                                   site_start, site_stop, exponent, null_diagonal, null_off, null_frequency,
                                   mismatch, selector)
        full[i] = _full_sib(pairs[i, 0], pairs[i, 1], pi, candidate_alt, stay, alpha, beta, row_arm, column_arm,
                            pivot, excursion, branch, state_alt, gl, coefficient, site_index, site_start,
                            site_stop, exponent, null_diagonal, null_off, null_frequency, mismatch, selector)
    return second, full


@njit(cache=True, parallel=True)
def _m1_pairs(pairs, pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
              state_alt, gl, coefficient, site_index, site_start, site_stop, exponent, null_diagonal, null_off,
              null_frequency, mismatch, common_products, first_cached_child, common_powered):
    out = np.empty(len(pairs))
    for i in prange(len(pairs)):
        tempered, full = _score_path(pairs[i, 0], pairs[i, 1], 0, 1, pi, candidate_alt, stay, alpha, beta,
                                     row_arm, column_arm, pivot, excursion, branch, state_alt, gl, coefficient,
                                     site_index, site_start, site_stop, exponent, null_diagonal, null_off,
                                     null_frequency, mismatch, common_products, first_cached_child,
                                     common_powered)
        out[i] = 0.5 * (tempered + full)
    return out


def component_contrasts(component, config, pairs, claim_threads=lambda: None):
    """(M1 - second-degree proxy, M1 - full-sib proxy, usable) float64/float64/bool [E] of one prepared component
    for the (child, candidate) ``pairs`` int64 [E, 2]
    zero where the candidate has no evidence here."""
    from . import scoring
    pairs = np.ascontiguousarray(pairs, dtype=np.int64)
    second = np.zeros(len(pairs))
    full = np.zeros(len(pairs))
    state = scoring.resident(component, config, claim_threads)
    projection, terms = state.projection, state.terms
    usable = np.asarray(projection.available, dtype=bool)[pairs[:, 1]] if len(pairs) else np.zeros(0, bool)
    rows = np.flatnonzero(usable)
    if not len(rows):
        return second, full, usable
    common = (projection.state_alt, terms.gl, terms.coefficient, terms.site_index, terms.site_start,
              terms.site_stop, terms.exponent, terms.null_diagonal, terms.null_off, terms.null_frequency,
              terms.mismatch)
    m1 = np.zeros(len(pairs))
    children = terms.gl.shape[0]
    for child_start in range(0, children, terms.batch):
        child_stop = min(children, child_start + terms.batch)
        batch = rows[(pairs[rows, 0] >= child_start) & (pairs[rows, 0] < child_stop)]
        if not len(batch):
            continue
        claim_threads()
        products, powered = terms.common(projection, child_start, child_stop)
        claim_threads()
        m1[batch] = _m1_pairs(np.ascontiguousarray(pairs[batch]), *_model(projection), *common, products,
                              child_start, powered)
        del products, powered
    claim_threads()
    proxy_second, proxy_full = _proxy_kernel(np.ascontiguousarray(pairs[rows]), *_model(projection), *common,
                                             np.ascontiguousarray(component.selector_switch_probability))
    second[rows] = m1[rows] - proxy_second
    full[rows] = m1[rows] - proxy_full
    return second, full, usable


def contrast_table(sample_ids, pairs, views, second, full, usable) -> pd.DataFrame:
    """One row per released child-candidate edge: ``second``, ``full`` float64 [C, E] per scored chromosome,
    ``usable`` bool [C, E]; ``views`` per edge: the views releasing it (tier_b, tier_b_partial, complete)."""
    rows = []
    for e, (child, parent) in enumerate(np.asarray(pairs, dtype=np.int64).tolist()):
        use = usable[:, e]
        d2, df = second[use, e], full[use, e]
        rows.append(dict(Sample=sample_ids[child], CandidateParent=sample_ids[parent], Views=";".join(views[e]),
                         ProxyModel=PROXY_MODEL, M1MinusSecondDegreeProxyNats=float(d2.sum()) if len(d2) else np.nan,
                         M1MinusFullSibProxyNats=float(df.sum()) if len(df) else np.nan,
                         UsableChromosomes=int(use.sum()), NegativeChromosomesSecondDegree=int((d2 < 0).sum()),
                         NegativeChromosomesFullSib=int((df < 0).sum()),
                         MinChromosomeContrastSecondDegree=float(d2.min()) if len(d2) else np.nan))
    return pd.DataFrame(rows, columns=list(COLUMNS))
