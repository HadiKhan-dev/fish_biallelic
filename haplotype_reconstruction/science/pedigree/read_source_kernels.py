"""Compute oriented source laws and background ALT probabilities with the painting's read state."""

from __future__ import annotations

import math

import numpy as np
from numba import prange

from ...core.kernels import kernel
from ..painting.api import apply_bin
from .kernels.preparation import OVERLOADS
from .read_gaps import configuration_ratios, site_adjoint, site_row
from .sources import _hamming_contract


@kernel(overloads=OVERLOADS["read_source_kernels._exclusion_sums"], cache=True, nogil=True)
def _exclusion_sums(matrix):
    """Row-without, column-without and complement leave-one-out sums, cancellation-free, O(S^2)."""
    states = matrix.shape[0]
    row_without = np.empty_like(matrix)
    column_without = np.empty_like(matrix)
    complement = np.empty_like(matrix)
    prefix = np.empty(states + 1, dtype=np.float64)
    suffix = np.empty(states + 1, dtype=np.float64)
    for row in range(states):
        prefix[0] = 0.0
        for column in range(states):
            prefix[column + 1] = prefix[column] + matrix[row, column]
        suffix[states] = 0.0
        for column in range(states - 1, -1, -1):
            suffix[column] = suffix[column + 1] + matrix[row, column]
        for column in range(states):
            row_without[row, column] = prefix[column] + suffix[column + 1]
    for column in range(states):
        prefix[0] = 0.0
        for row in range(states):
            prefix[row + 1] = prefix[row] + matrix[row, column]
        suffix[states] = 0.0
        for row in range(states - 1, -1, -1):
            suffix[row] = suffix[row + 1] + matrix[row, column]
        for row in range(states):
            column_without[row, column] = prefix[row] + suffix[row + 1]
    for row in range(states):
        prefix[0] = 0.0
        for column in range(states):
            prefix[column + 1] = prefix[column] + column_without[row, column]
        suffix[states] = 0.0
        for column in range(states - 1, -1, -1):
            suffix[column] = suffix[column + 1] + column_without[row, column]
        for column in range(states):
            complement[row, column] = prefix[column] + suffix[column + 1]
    return row_without, column_without, complement


@kernel(overloads=OVERLOADS["read_source_kernels._apply_pairs"], cache=True, nogil=True)
def _apply_pairs(emission, read, read_log, tape, exponent, pair_index, perm, available, sample, block,
                 entering, out, backward, exit_weight):
    """Apply each ordered pair's bin operator with one common gauge across pairs.

    For the forward posterior, include the exit backward weight before choosing the gauge.
    """
    classes, states, _ = entering.shape
    gauges = np.empty((states, states))
    row = np.empty(classes)
    moved = np.empty(classes)
    scratch = np.empty((4, classes))
    top = -np.inf
    for first in range(states):
        for second in range(states):
            for c in range(classes):
                row[c] = entering[c, first, second]
            if available[sample]:
                pair = pair_index[first, second]
                gauge = apply_bin(read, tape, exponent, sample, pair, block, first > second, perm,
                                  row, moved, scratch, backward)
                gauge += emission[sample, block, first, second] + read_log[sample, pair, block]
            else:
                moved[:] = 0.0
                moved[0] = row[0]
                gauge = 0.0
            peak = 0.0
            for c in range(classes):
                if not backward:
                    moved[c] *= exit_weight[c, first, second]
                peak = max(peak, moved[c])
            if peak > 0.0 and np.isfinite(gauge):
                gauge += math.log(peak)
                for c in range(classes):
                    out[c, first, second] = moved[c] / peak
            else:
                gauge = -np.inf
                out[:, first, second] = 0.0
            gauges[first, second] = gauge
            top = max(top, gauge)
    if np.isfinite(top):
        for first in range(states):
            for second in range(states):
                scale = math.exp(gauges[first, second] - top)
                for c in range(classes):
                    out[c, first, second] *= scale
    return np.isfinite(top)


@kernel(overloads=OVERLOADS["read_source_kernels._backward"], cache=True, parallel=True, nogil=True)
def _backward(emission, read, read_log, tape, exponent, pair_index, perm, available, same, one, two, beta, failed):
    """Scaled backward messages. Hamming movement acts independently on each read class."""
    candidates, bins, states, _ = emission.shape
    classes = beta.shape[2]
    valid = np.ones(candidates, dtype=np.bool_)
    for candidate in prange(candidates):
        right = np.empty((classes, states, states))
        rows = np.empty(states)
        columns = np.empty(states)
        beta[candidate, bins - 1] = 1.0
        for boundary in range(bins - 2, -1, -1):
            exit_beta = beta[candidate, boundary + 1]
            if not _apply_pairs(emission, read, read_log, tape, exponent, pair_index, perm, available,
                                np.int64(candidate), boundary + 1, exit_beta, right, True, exit_beta):
                valid[candidate] = False
                failed[candidate] = boundary
                break
            for i in range(classes):
                _hamming_contract(
                    right[i], same[boundary], one[boundary], two[boundary], beta[candidate, boundary, i], rows, columns
                )
            maximum = np.max(beta[candidate, boundary])
            if maximum <= 0.0 or not np.isfinite(maximum):
                valid[candidate] = False
                failed[candidate] = boundary
                break
            beta[candidate, boundary] /= maximum
    return valid


@kernel(overloads=OVERLOADS["read_source_kernels._class_flows"], cache=True, nogil=True)
def _class_flows(posterior, rho, w0, w1, w2, tau, diagonal, outgoing, incoming, contracted):
    """One read class of a boundary: adds its transmitted-coordinate flows to ``diagonal`` / ``outgoing`` /
    ``incoming`` [S] and writes ``contracted`` [S, S] = sum_u A(u, v) P(u, c) / beta(u, c) (the propagation's input).
    ``posterior`` [S, S] the oriented law's class slice, ``rho`` [S, S] the class's right weights r(v, c), whose Hamming
    contraction is beta(u, c) up to the boundary's scale. Returns False when a state with mass has no support."""
    states = posterior.shape[0]
    row_rho = np.sum(rho, axis=1)
    column_rho = np.sum(rho, axis=0)
    rho_row_without, rho_column_without, rho_complement = _exclusion_sums(rho)
    scaled = np.empty((states, states), dtype=np.float64)
    for first in range(states):
        for second in range(states):
            cell = rho[first, second]
            denominator = (
                w0 * cell
                + w1 * (rho_row_without[first, second] + rho_column_without[first, second])
                + w2 * rho_complement[first, second]
            )
            if denominator <= 0.0 or not np.isfinite(denominator):
                # a class without continuation carries no mass (its backward vector is zero)
                if posterior[first, second] != 0.0:
                    return False
                scaled[first, second] = 0.0
            else:
                scaled[first, second] = posterior[first, second] / denominator
    for first in range(states):
        direct_same = 0.0
        swapped_same = 0.0
        outgoing_direct = 0.0
        outgoing_swapped = 0.0
        swap_change_base = w1 * rho_row_without[first, first] + w2 * rho_complement[first, first]
        for second in range(states):
            u = scaled[first, second]
            direct_same += u * (w1 * row_rho[first] + (w0 - w1) * rho[first, second])
            if second == first:
                swapped_same += u * (w1 * column_rho[first] + (w0 - w1) * rho[first, first])
            else:
                swapped_same += u * (w2 * column_rho[first] + (w1 - w2) * rho[first, first])
            direct_change = w1 * rho_column_without[first, second] + w2 * rho_complement[first, second]
            swap_change = swap_change_base
            if second != first:
                swap_change += (w0 - w1) * rho[first, second] + (w1 - w2) * rho_column_without[first, second]
            outgoing_direct += u * direct_change
            outgoing_swapped += u * swap_change
        diagonal[first] += (1.0 - tau) * direct_same + tau * swapped_same
        outgoing[first] += (1.0 - tau) * outgoing_direct + tau * outgoing_swapped
    scaled_row_without, scaled_column_without, scaled_complement = _exclusion_sums(scaled)
    for destination in range(states):
        other_source_total = 0.0
        for other in range(states):
            other_source_total += scaled_column_without[destination, other]
        direct_incoming = w2 * row_rho[destination] * other_source_total
        correction = 0.0
        for other in range(states):
            correction += rho[destination, other] * scaled_column_without[destination, other]
        direct_incoming += (w1 - w2) * correction
        swapped_incoming = 0.0
        for source in range(states):
            if source == destination:
                continue
            source_to_destination = scaled[source, destination]
            source_other = scaled_row_without[source, destination]
            swapped_incoming += column_rho[destination] * (
                w1 * source_to_destination + w2 * source_other
            ) + rho[source, destination] * ((w0 - w1) * source_to_destination + (w1 - w2) * source_other)
        incoming[destination] += (1.0 - tau) * direct_incoming + tau * swapped_incoming
    for first in range(states):
        for second in range(states):
            contracted[first, second] = (
                w0 * scaled[first, second]
                + w1 * (scaled_row_without[first, second] + scaled_column_without[first, second])
                + w2 * scaled_complement[first, second]
            )
    return True


@kernel(overloads=OVERLOADS["read_source_kernels._oriented"], cache=True, parallel=True, nogil=True)
def _oriented(
    emission, read, read_log, tape, exponent, pair_index, perm, available, initial, same, one, two, selector,
    beta, pair_first, pair_second,
    pair_mass, pi, diagonal_flow, outgoing_flow, incoming_flow, background_law, entering, failed,
):
    """The transmission-oriented propagation of every candidate (prange) and its class-free outputs: pair_mass
    [N, B, F], pi [N, B, S], the flows [N, B-1, S], background_law [N, B, S] = P_b(x = BACKGROUND, y) and entering
    [N, B, C, S] the class vector entering bin b of the oriented pair (BACKGROUND, y) (any positive scale per y).
    Returns valid bool [N]; ``failed`` [N] the boundary where mass was lost (-1 the first bin)."""
    candidates, bins, states, _ = emission.shape
    classes = initial.shape[1]
    features = len(pair_first)
    background = states - 1
    valid = np.ones(candidates, dtype=np.bool_)
    for candidate in prange(candidates):
        law = np.empty((classes, states, states), dtype=np.float64)
        work = np.empty((classes, states, states), dtype=np.float64)
        contracted = np.empty((classes, states, states), dtype=np.float64)
        right = np.empty((classes, states, states), dtype=np.float64)
        free = np.empty((states, states), dtype=np.float64)
        diagonal = np.empty(states, dtype=np.float64)
        outgoing = np.empty(states, dtype=np.float64)
        incoming = np.empty(states, dtype=np.float64)
        symmetric_initial = np.empty(classes, dtype=np.float64)
        for i in range(classes):
            symmetric_initial[i] = 0.5 * (initial[candidate, i] + initial[candidate, perm[i]])
        for i in range(classes):
            contracted[i] = symmetric_initial[i]
        exit_beta = beta[candidate, 0]
        if not _apply_pairs(emission, read, read_log, tape, exponent, pair_index, perm, available,
                            np.int64(candidate), 0, contracted, work, False, exit_beta):
            valid[candidate] = False
            failed[candidate] = -1
            continue
        # unphased source evidence has an arbitrary homologue gauge: the first bin's law is the mean of the two
        # orientations, (a, b, c) and (b, a, J c)
        total = 0.0
        for j in range(classes):
            for first in range(states):
                for second in range(states):
                    value = 0.5 * (work[j, first, second] + work[perm[j], second, first])
                    law[j, first, second] = value
                    total += value
        if total <= 0.0 or not np.isfinite(total):
            valid[candidate] = False
            failed[candidate] = -1
            continue
        law /= total
        for block in range(bins):
            if block > 0:
                boundary = block - 1
                w0 = same[boundary]
                w1 = one[boundary]
                w2 = two[boundary]
                tau = selector[boundary]
                exit_beta = beta[candidate, block]
                if not _apply_pairs(emission, read, read_log, tape, exponent, pair_index, perm, available,
                                    np.int64(candidate), block, exit_beta, right, True, exit_beta):
                    valid[candidate] = False
                    failed[candidate] = boundary
                    break
                diagonal[:] = 0.0
                outgoing[:] = 0.0
                incoming[:] = 0.0
                supported = True
                for i in range(classes):
                    if not _class_flows(law[i], right[i], w0, w1, w2, tau, diagonal, outgoing, incoming,
                                        contracted[i]):
                        supported = False
                        break
                if not supported or not _apply_pairs(
                    emission, read, read_log, tape, exponent, pair_index, perm, available,
                    np.int64(candidate), block, contracted, work, False, exit_beta,
                ):
                    valid[candidate] = False
                    failed[candidate] = boundary
                    break
                total = 0.0
                for j in range(classes):
                    for first in range(states):
                        for second in range(states):
                            value = (1.0 - tau) * work[j, first, second] + tau * work[perm[j], second, first]
                            law[j, first, second] = value
                            total += value
                if total <= 0.0 or not np.isfinite(total):
                    valid[candidate] = False
                    failed[candidate] = boundary
                    break
                law /= total
                # The forward action has its own gauge. The flows already share the right-message gauge.
                flow_total = diagonal.sum() + outgoing.sum()
                for state in range(states):
                    diagonal_flow[candidate, boundary, state] = diagonal[state] / flow_total
                    outgoing_flow[candidate, boundary, state] = outgoing[state] / flow_total
                    incoming_flow[candidate, boundary, state] = incoming[state] / flow_total
                for i in range(classes):
                    for other in range(states):
                        entering[candidate, block, i, other] = (1.0 - tau) * contracted[i, background, other] + (
                            tau * contracted[perm[i], other, background]
                        )
            else:
                for i in range(classes):
                    for other in range(states):
                        entering[candidate, 0, i, other] = symmetric_initial[i]
            for first in range(states):
                for second in range(states):
                    value = 0.0
                    for i in range(classes):
                        value += law[i, first, second]
                    free[first, second] = value
            for first in range(states):
                value = 0.0
                for second in range(states):
                    value += free[first, second]
                pi[candidate, block, first] = value
                background_law[candidate, block, first] = free[background, first]
            for feature in range(features):
                first = pair_first[feature]
                second = pair_second[feature]
                value = free[first, second]
                if first != second:
                    value += free[second, first]
                pair_mass[candidate, block, feature] = value
    return valid


@kernel(overloads=OVERLOADS["read_source_kernels._site_factor"], cache=True, inline="always", nogil=True)
def _site_factor(q1, q2, raw_gl, b_cohort, kappa, epsilon, tempered, out):
    """Tempered GL-anchored factor for an observed (BACKGROUND, partner) configuration."""
    classes = out.shape[0]
    w00, w01, w10, w11 = (1.0 - q1) * (1.0 - q2), (1.0 - q1) * q2, q1 * (1.0 - q2), q1 * q2
    g0, g1, g2 = raw_gl[0], raw_gl[1], raw_gl[2]
    mixed = (1.0 - epsilon) * (w00 * g0 + (w01 + w10) * g1 + w11 * g2) + epsilon * b_cohort
    for s in range(classes):
        out[s] = 0.0
    for config in range(4):
        wa = w00 if config == 0 else (w01 if config == 1 else (w10 if config == 2 else w11))
        if wa == 0.0:
            continue
        la = g0 if config == 0 else (g2 if config == 3 else g1)
        for s in range(classes):
            out[s] += wa * la * kappa[config, s]
    for s in range(classes):
        ratio = ((1.0 - epsilon) * out[s] + epsilon * b_cohort) / mixed if mixed > 0.0 else 1.0
        out[s] = ratio**tempered


@kernel(overloads=OVERLOADS["read_source_kernels._background_alt"], cache=True, parallel=True, nogil=True)
def _background_alt(
    background_law, entering, beta, has_reads, marker_gl, marker_observed, state_alt, marker_site, marker_start,
    marker_stop, site_start, site_stop, alt, composed, laws, class_lik, gl, seen, cohort, epsilon, weight,
    counts, error, loss, survival, absent, depth, result, lost,
):
    """result [N, M]: BACKGROUND's transmitted ALT probability per candidate and marker (one task per candidate and
    bin). A fractional BACKGROUND frequency q is conditioned on the candidate's reads: sum over the untransmitted
    state y of P_b(BACKGROUND, y) times, with the read state, the class posterior at the marker's site (forward
    from the oriented pair's entering vector through the bin's sites, backward from beta_b) averaged
    P(transmitted ALT | y, class, the site's reads); without it the site's mixed genotype likelihoods (kappa = 1).
    ``lost`` bool [N, B]: a partner y with positive mass whose class posterior at a marker vanished (numerical loss,
    reported by the caller; never silently dropped). ``depth`` [N] the foreign depth: a candidate without any (QC A's
    corrected rows) has unreachable foreign classes, whose factors are zero at its observed sites as in the painting's
    site tape (painting.read_operator._make_tape), so they never set this chain's normalisations."""
    candidates, bins, states = background_law.shape
    classes = entering.shape[2]
    background = states - 1
    for task in prange(candidates * bins):
        candidate = task // bins
        block = task % bins
        first_marker = marker_start[block]
        last_marker = marker_stop[block]
        conditioned = False
        for marker in range(first_marker, last_marker):
            q = state_alt[background, marker]
            result[candidate, marker] = q
            if q != 0.0 and q != 1.0:
                conditioned = True
        if not conditioned:
            continue
        if not has_reads[candidate]:
            for marker in range(first_marker, last_marker):
                q_first = state_alt[background, marker]
                if q_first == 0.0 or q_first == 1.0:
                    continue
                if marker_observed[candidate, marker]:
                    w0 = marker_gl[candidate, marker, 0]
                    w1 = marker_gl[candidate, marker, 1]
                    w2 = marker_gl[candidate, marker, 2]
                else:
                    w0 = 1.0
                    w1 = 1.0
                    w2 = 1.0
                numerator = 0.0
                marginal = 0.0
                for second in range(states):
                    q_second = state_alt[second, marker]
                    p00 = (1.0 - q_first) * (1.0 - q_second)
                    p10 = q_first * (1.0 - q_second)
                    p01 = (1.0 - q_first) * q_second
                    p11 = q_first * q_second
                    denominator = w0 * p00 + w1 * (p10 + p01) + w2 * p11
                    conditional = (w1 * p10 + w2 * p11) / denominator if denominator > 0.0 else q_first
                    mass = background_law[candidate, block, second]
                    marginal += mass
                    numerator += mass * conditional
                result[candidate, marker] = numerator / marginal if marginal > 0.0 else q_first
            continue
        first_site = site_start[block]
        count = site_stop[block] - first_site
        tempered = weight[candidate, block]
        factors = np.empty((count, classes), dtype=np.float64)
        forward = np.empty((count, classes), dtype=np.float64)
        vector = np.empty(classes, dtype=np.float64)
        work = np.empty(classes, dtype=np.float64)
        kappa = np.empty((count, 4, classes), dtype=np.float64)
        cohort_gl = np.empty(count, dtype=np.float64)
        own = laws.shape[2]
        extended = classes > own
        # A named founder's allele is 0, 1 or the local BACKGROUND frequency.
        # Reuse its site factor across partners with the same allele probability.
        saved = np.empty((count, 3, classes), dtype=np.float64)
        computed = np.zeros((count, 3), dtype=np.bool_)
        for k in range(count):
            site = first_site + k
            if not seen[candidate, site]:
                continue
            g0, g1, g2 = gl[candidate, site]
            cohort_gl[k] = cohort[site, 0] * g0 + cohort[site, 1] * g1 + cohort[site, 2] * g2
            configuration_ratios(class_lik[candidate, site], laws[site, candidate], alt, site,
                                 counts, error, loss, candidate, kappa[k])
        numerator = np.zeros(last_marker - first_marker, dtype=np.float64)
        marginal = np.zeros(last_marker - first_marker, dtype=np.float64)
        for second in range(states):
            mass = background_law[candidate, block, second]
            if mass <= 0.0:
                continue
            for k in range(count):
                site = first_site + k
                if not seen[candidate, site]:
                    factors[k] = 1.0
                    continue
                q1, q2 = alt[background, site], alt[second, site]
                if q2 == 0.0 or q2 == 1.0:
                    allele = int(q2)
                else:
                    allele = 2 if q2 == q1 else -1
                if allele < 0:
                    _site_factor(q1, q2, gl[candidate, site], cohort_gl[k], kappa[k], epsilon, tempered, factors[k])
                else:
                    if not computed[k, allele]:
                        _site_factor(q1, q2, gl[candidate, site], cohort_gl[k], kappa[k], epsilon, tempered,
                                     saved[k, allele])
                        computed[k, allele] = True
                    factors[k] = saved[k, allele]
            if extended:
                # The two donor labels are excluded even at an unobserved emitting site.
                factors[:, own + background] = 0.0
                factors[:, own + second] = 0.0
                if depth[candidate] == 0.0:
                    for k in range(count):
                        if seen[candidate, first_site + k]:
                            factors[k, own:] = 0.0
            for i in range(classes):
                vector[i] = entering[candidate, block, i, second]
            for k in range(count):
                site = first_site + k
                phi = survival[candidate, site] if extended else 1.0
                q = absent[candidate, site] if extended else 1.0
                site_row(composed[site, candidate], laws[site, candidate], phi, q, vector, work)
                top = 0.0
                for j in range(classes):
                    work[j] *= factors[k, j]
                    top = max(top, work[j])
                for j in range(classes):
                    vector[j] = work[j] / top if top > 0.0 else work[j]
                    forward[k, j] = vector[j]
            for i in range(classes):
                vector[i] = beta[candidate, block, i, background, second]
            marker = last_marker - 1
            for k in range(count - 1, -1, -1):
                site = first_site + k
                if marker >= first_marker and marker_site[marker] == site:
                    q1 = alt[background, site]
                    if q1 != 0.0 and q1 != 1.0:
                        # the class posterior at the site, and each class's transmitted-ALT conditional
                        norm = 0.0
                        for i in range(classes):
                            norm += forward[k, i] * vector[i]
                        if not (norm > 0.0 and np.isfinite(norm)):
                            lost[candidate, block] = True
                        else:
                            if seen[candidate, site]:
                                q2 = alt[second, site]
                                w00 = (1.0 - q1) * (1.0 - q2)
                                w01 = (1.0 - q1) * q2
                                w10 = q1 * (1.0 - q2)
                                w11 = q1 * q2
                                g0 = gl[candidate, site, 0]
                                g1 = gl[candidate, site, 1]
                                g2 = gl[candidate, site, 2]
                                b_cohort = cohort_gl[k]
                                value = 0.0
                                for i in range(classes):
                                    e00 = (1.0 - epsilon) * g0 * kappa[k, 0, i] + epsilon * b_cohort
                                    e01 = (1.0 - epsilon) * g1 * kappa[k, 1, i] + epsilon * b_cohort
                                    e10 = (1.0 - epsilon) * g1 * kappa[k, 2, i] + epsilon * b_cohort
                                    e11 = (1.0 - epsilon) * g2 * kappa[k, 3, i] + epsilon * b_cohort
                                    denominator = w00 * e00 + w01 * e01 + w10 * e10 + w11 * e11
                                    conditional = (w10 * e10 + w11 * e11) / denominator if denominator > 0.0 else q1
                                    value += forward[k, i] * vector[i] * conditional
                                value /= norm
                            else:
                                value = q1
                            numerator[marker - first_marker] += mass * value
                            marginal[marker - first_marker] += mass
                    marker -= 1
                phi = survival[candidate, site] if extended else 1.0
                q = absent[candidate, site] if extended else 1.0
                for i in range(classes):
                    vector[i] *= factors[k, i]
                site_adjoint(composed[site, candidate], laws[site, candidate], phi, q, vector, work)
                top = np.max(work)
                for i in range(classes):
                    vector[i] = work[i] / top if top > 0.0 else work[i]
        for marker in range(first_marker, last_marker):
            q_first = state_alt[background, marker]
            if q_first == 0.0 or q_first == 1.0:
                continue
            total = marginal[marker - first_marker]
            result[candidate, marker] = numerator[marker - first_marker] / total if total > 0.0 else q_first
