"""Transmitted-state projection of the candidate-parent source chains (projected quadratic model).

A parent transmits one of its two painted homologues at each bin; which one is a selector that
switches at meiotic crossovers (probability ``selector_switch_probability`` per bin edge). The exact
law of the transmitted founder state couples the ordered-diplotype chain with the selector, an S^4
object for two parents. The projection keeps, per candidate and bin, the transmitted-state marginal
pi (S values), its transmitted ALT probability at each marker conditional on the candidate's own
reads, and between bins a maximum-entropy bridge matching the exact adjacent marginals and the exact
same-state (persistence) flow, so a trio is scored with O(S^2) forward states.

Bridge operator per boundary (row = source state, column = destination): T = diag(stay) +
alpha beta^T off the diagonal (+ one pivot row arm and column arm), solved by bisection on the
maximum-entropy root; branch 0 diagonal, 1 all-small, 2 one-large, 3 Hall star.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from ..maths import ties
from ..painting.states import background_alt_frequency, state_alt_probabilities
from .records import PreparedComponent

# A GL row whose max - min exceeds this is informative (base pedigree definition).
INFORMATIVE_GL_SPREAD = 1e-12
# Exact Hall equality of a bridge (release 2.1, review R12). Where the equality holds mathematically (all residual flow
# through one state) the computed Hall gap is the rounding of the two differences that form it: <= 2 ULP of their terms
# on TroMau chr3 (5,009 of 147,320 bridges), and 6,923 bridges flipped between exactly 0 and such a rounding value
# under iid 1e-9 noise, switching between the star and the one-large parameterisation (M1/M2 scores moved by up to
# 1.4e-4 / 0.07 nats through the rows of near-unreachable states). A difference within HALL_ULPS ULP of its larger term
# is that equality up to rounding (the same 64 ULP as the feasibility tolerance); genuine gaps move by <= 3e-16 under
# the same noise. The scale is per difference: a gap formed by tiny flows is compared with those flows, not with 1.
HALL_ULPS = 64.0
# The Hall state among states whose loads tie up to rounding (release 2.1). Such ties are the rule when the residual
# flow is an exchange between two states (both loads equal 1 up to rounding: 1,500-1,900 bridges per TroMau contig);
# the choice then decides only the rows of near-unreachable states (masses far below rounding of the flows), which the
# star sends through the Hall state. In the exact transmitted chain those states leave to, and are entered from, the
# state that holds the mass at the boundary, so the Hall state is the tied state with the largest mass pi_b + pi_b+1
# (masses within HUB_MASS_TIE are tied: then the lowest state). 2.0 took the first maximum of the rounded loads.
HUB_MASS_TIE = 2.0 ** -20


@dataclass(frozen=True)
class Projection:
    """Projected transmission model of one component (C candidates = N samples).

    pi:            float64 [N, B, S] transmitted-state marginal per candidate and bin.
    candidate_alt: float64 [N, M, S] transmitted ALT probability per candidate, marker, state.
    stay, alpha, beta, row_arm, column_arm: float64 [N, B-1, S] bridge operator parts.
    pivot:         int64 [N, B-1] operator pivot state (-1 none); branch int8 [N, B-1].
    excursion:     float64 [N, B-1] weight k of the uniform selector kernel mixed into each bridge,
                   T' = (1 - k) T + k U (switch probability where the parent has no support).
    state_alt:     float64 [S, M] ALT probability of each state at each marker (BACKGROUND last).
    bin_start, bin_stop: int64 [B] marker ranges of the bins.
    available:     bool [N] candidates with source evidence.
    """

    pi: np.ndarray
    candidate_alt: np.ndarray
    stay: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    row_arm: np.ndarray
    column_arm: np.ndarray
    pivot: np.ndarray
    branch: np.ndarray
    excursion: np.ndarray
    state_alt: np.ndarray
    bin_start: np.ndarray
    bin_stop: np.ndarray
    available: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.pi.shape[1])

    @property
    def n_states(self) -> int:
        return int(self.pi.shape[2])


@njit(cache=True, parallel=True)
def _symmetrise(posterior):
    """Average each ordered diplotype with its mirror and renormalise per bin (in place)."""
    candidates, bins, states, _ = posterior.shape
    valid = np.ones(candidates, dtype=np.bool_)
    for candidate in prange(candidates):
        for block in range(bins):
            for first in range(states):
                for second in range(first + 1, states):
                    value = 0.5 * (posterior[candidate, block, first, second]
                                   + posterior[candidate, block, second, first])
                    posterior[candidate, block, first, second] = value
                    posterior[candidate, block, second, first] = value
            total = 0.0
            for first in range(states):
                for second in range(states):
                    total += posterior[candidate, block, first, second]
            if total <= 0.0 or not np.isfinite(total):
                valid[candidate] = False
            else:
                posterior[candidate, block] /= total
    return valid


@njit(cache=True, parallel=True)
def _transmitted_alt(candidate_gl, candidate_observed, state_alt, posterior, site_to_bin,
                     background_index, epsilon):
    """P(transmitted allele = ALT | transmitted state, own reads), the untransmitted homologue
    marginalised over the source posterior, without an S^2 per-site array.

    Only BACKGROUND (no panel haplotype) is conditioned on the candidate's reads. A named class transmits its panel
    allele, or BACKGROUND's ALT frequency where the panel left it uncalled (release 2.16): the panel leaves alleles
    uncalled where the data are noisy, and there the candidate's own GL rows are noisy too (simulations: heterozygous-
    looking at ~40% of truly homozygous such sites), so conditioning on them cost true parents tens of nats per
    chromosome against M0 and the relative models."""
    candidates, sites, _ = candidate_gl.shape
    states = state_alt.shape[0]
    result = np.empty((candidates, sites, states), dtype=np.float64)
    for task in prange(candidates * sites * states):
        first = task % states
        quotient = task // states
        site = quotient % sites
        candidate = quotient // sites
        block = site_to_bin[site]
        q_first = state_alt[first, site]
        # A deterministic transmitted allele cannot change on conditioning (the shared emission
        # cache relies on these passing through unchanged); named classes are never conditioned.
        if q_first == 0.0 or q_first == 1.0 or first < background_index:
            result[candidate, site, first] = q_first
            continue
        if candidate_observed[candidate, site]:
            w0 = (1.0 - epsilon) * candidate_gl[candidate, site, 0] + epsilon / 3.0
            w1 = (1.0 - epsilon) * candidate_gl[candidate, site, 1] + epsilon / 3.0
            w2 = (1.0 - epsilon) * candidate_gl[candidate, site, 2] + epsilon / 3.0
        else:
            w0 = 1.0
            w1 = 1.0
            w2 = 1.0
        numerator = 0.0
        marginal = 0.0
        for second in range(states):
            q_second = state_alt[second, site]
            if first == second and first < background_index:
                p00 = 1.0 - q_first
                p10 = 0.0
                p01 = 0.0
                p11 = q_first
            else:
                p00 = (1.0 - q_first) * (1.0 - q_second)
                p10 = q_first * (1.0 - q_second)
                p01 = (1.0 - q_first) * q_second
                p11 = q_first * q_second
            denominator = w0 * p00 + w1 * (p10 + p01) + w2 * p11
            conditional = (w1 * p10 + w2 * p11) / denominator if denominator > 0.0 else q_first
            mass = posterior[candidate, block, first, second]
            marginal += mass
            numerator += mass * conditional
        result[candidate, site, first] = numerator / marginal if marginal > 0.0 else q_first
    return result


@njit(cache=True)
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


@njit(cache=True, parallel=True)
def _selected_flows(posterior, right_weight, same, one, two, selector):
    """Propagate the selected (transmitted) diplotypes and the positive adjacent selected-state
    flows: same-state (diagonal), outgoing-change and incoming-change flow per boundary."""
    candidates, bins, states, _ = posterior.shape
    shape = (candidates, bins - 1, states)
    diagonal_flow = np.empty(shape, dtype=np.float64)
    outgoing_change_flow = np.empty(shape, dtype=np.float64)
    incoming_change_flow = np.empty(shape, dtype=np.float64)
    valid = np.ones((candidates, bins - 1), dtype=np.bool_)
    for candidate in prange(candidates):
        for boundary in range(bins - 1):
            w0 = same[boundary]
            w1 = one[boundary]
            w2 = two[boundary]
            rho = right_weight[candidate, boundary]
            row_rho = np.sum(rho, axis=1)
            column_rho = np.sum(rho, axis=0)
            rho_row_without, rho_column_without, rho_complement = _exclusion_sums(rho)
            scaled = np.empty((states, states), dtype=np.float64)
            for first in range(states):
                for second in range(states):
                    cell = rho[first, second]
                    denominator = (w0 * cell + w1 * (rho_row_without[first, second]
                                                     + rho_column_without[first, second])
                                   + w2 * rho_complement[first, second])
                    if denominator <= 0.0 or not np.isfinite(denominator):
                        valid[candidate, boundary] = False
                        scaled[first, second] = 0.0
                    else:
                        scaled[first, second] = posterior[candidate, boundary, first, second] / denominator

            tau = selector[candidate, boundary]
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
                    direct_change = (w1 * rho_column_without[first, second]
                                     + w2 * rho_complement[first, second])
                    swap_change = swap_change_base
                    if second != first:
                        swap_change += ((w0 - w1) * rho[first, second]
                                        + (w1 - w2) * rho_column_without[first, second])
                    outgoing_direct += u * direct_change
                    outgoing_swapped += u * swap_change
                diagonal_flow[candidate, boundary, first] = (1.0 - tau) * direct_same + tau * swapped_same
                outgoing_change_flow[candidate, boundary, first] = (
                    (1.0 - tau) * outgoing_direct + tau * outgoing_swapped)

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
                    swapped_incoming += (
                        column_rho[destination] * (w1 * source_to_destination + w2 * source_other)
                        + rho[source, destination] * ((w0 - w1) * source_to_destination
                                                      + (w1 - w2) * source_other))
                incoming_change_flow[candidate, boundary, destination] = (
                    (1.0 - tau) * direct_incoming + tau * swapped_incoming)

            direct = np.empty((states, states), dtype=np.float64)
            for first in range(states):
                for second in range(states):
                    cell = scaled[first, second]
                    direct[first, second] = rho[first, second] * (
                        w0 * cell + w1 * (scaled_row_without[first, second]
                                          + scaled_column_without[first, second])
                        + w2 * scaled_complement[first, second])
            total = 0.0
            for first in range(states):
                for second in range(states):
                    value = (1.0 - tau) * direct[first, second] + tau * direct[second, first]
                    posterior[candidate, boundary + 1, first, second] = value
                    total += value
            if total <= 0.0 or not np.isfinite(total):
                valid[candidate, boundary] = False
            else:
                posterior[candidate, boundary + 1] /= total
                diagonal_flow[candidate, boundary] /= total
                outgoing_change_flow[candidate, boundary] /= total
                incoming_change_flow[candidate, boundary] /= total
    return diagonal_flow, outgoing_change_flow, incoming_change_flow, valid


@njit(cache=True, inline="always")
def _small_diagonal_product(q, a, b):
    """Stable small root of the excluded diagonal product x_i y_i."""
    if q <= 0.0 or a <= 0.0 or b <= 0.0:
        return 0.0
    sqrt_a = math.sqrt(a)
    sqrt_b = math.sqrt(b)
    plus = sqrt_a + sqrt_b
    minus = sqrt_a - sqrt_b
    first = max(0.0, 1.0 - q * plus * plus)
    second = max(0.0, 1.0 - q * minus * minus)
    denominator = 1.0 - q * (a + b) + math.sqrt(first * second)
    return 2.0 * q * a * b / denominator if denominator > 0.0 else 0.0


@njit(cache=True, inline="always")
def _branch_function(q, a, b, large_index):
    total_u = 0.0
    large_u = 0.0
    for state in range(len(a)):
        value = _small_diagonal_product(q, a[state], b[state])
        total_u += value
        if state == large_index:
            large_u = value
    if large_index < 0:
        return q * (1.0 + total_u) - 1.0
    complement_a = 0.0
    complement_b = 0.0
    for state in range(len(a)):
        if state != large_index:
            complement_a += a[state]
            complement_b += b[state]
    # The larger independently accumulated complement keeps tiny positive Hall slack when
    # 1 - a_h - b_h would round to zero.
    base = max(0.0, complement_a - b[large_index], complement_b - a[large_index])
    return base + total_u - 2.0 * large_u


@njit(cache=True)
def _bisect_root(a, b, large_index, q_max, max_iterations):
    lower = 0.0
    upper = q_max
    lower_value = _branch_function(lower, a, b, large_index)
    upper_value = _branch_function(upper, a, b, large_index)
    increasing = large_index < 0
    used = 0
    for iteration in range(min(128, max_iterations)):
        midpoint = 0.5 * (lower + upper)
        if midpoint == lower or midpoint == upper:
            break
        value = _branch_function(midpoint, a, b, large_index)
        if increasing:
            if value < 0.0:
                lower, lower_value = midpoint, value
            else:
                upper, upper_value = midpoint, value
        else:
            if value > 0.0:
                lower, lower_value = midpoint, value
            else:
                upper, upper_value = midpoint, value
        used = iteration + 1
    if abs(lower_value) <= abs(upper_value):
        return lower, used
    return upper, used


def _bridges(pi, diagonal_flow, outgoing_change_flow, incoming_change_flow, tolerance, max_iterations,
             reverse=None):
    """Every normalised zero-diagonal maximum-entropy completion (one task per candidate x boundary).

    Among states tied up to rounding (Hall state, bracketing large branch, pivot) the lowest state is taken, or the
    highest under the convention reversal (``reverse``; default maths.ties.reverse_conventions(), release 2.2)."""
    if reverse is None:
        reverse = ties.reverse_conventions()
    return _bridges_kernel(pi, diagonal_flow, outgoing_change_flow, incoming_change_flow, tolerance,
                           max_iterations, bool(reverse))


@njit(cache=True, parallel=True)
def _bridges_kernel(pi, diagonal_flow, outgoing_change_flow, incoming_change_flow, tolerance,
                    max_iterations, reverse):
    """:func:`_bridges` (the kernel)."""
    candidates, bins, states = pi.shape
    boundaries = bins - 1
    stay = np.zeros((candidates, boundaries, states), dtype=np.float64)
    alpha = np.zeros_like(stay)
    beta = np.zeros_like(stay)
    row_arm = np.zeros_like(stay)
    column_arm = np.zeros_like(stay)
    branch = np.zeros((candidates, boundaries), dtype=np.int8)
    operator_pivot = np.full((candidates, boundaries), -1, dtype=np.int64)
    valid = np.ones((candidates, boundaries), dtype=np.bool_)
    epsilon = np.finfo(np.float64).eps
    smallest_normal = np.finfo(np.float64).tiny
    feasibility_tolerance = 64.0 * epsilon
    for task in prange(candidates * boundaries):
        candidate = task // boundaries
        boundary = task % boundaries
        row = np.empty(states, dtype=np.float64)
        column = np.empty(states, dtype=np.float64)
        row_mass = np.empty(states, dtype=np.float64)
        row_total = 0.0
        column_total = 0.0
        for state in range(states):
            source_mass = pi[candidate, boundary, state]
            diagonal_mass = diagonal_flow[candidate, boundary, state]
            row[state] = outgoing_change_flow[candidate, boundary, state]
            column[state] = incoming_change_flow[candidate, boundary, state]
            if row[state] < -feasibility_tolerance or column[state] < -feasibility_tolerance:
                valid[candidate, boundary] = False
            row[state] = max(0.0, row[state])
            column[state] = max(0.0, column[state])
            row_total += row[state]
            column_total += column[state]
            # A subnormal source mass keeps too few bits for flow / mass ratios to sum to one, so
            # such a row is normalised by its own retained outgoing flow.
            row_mass[state] = (source_mass if source_mass >= smallest_normal
                               else diagonal_mass + row[state])
            stay[candidate, boundary, state] = (diagonal_mass / row_mass[state]
                                                if row_mass[state] > 0.0 else 1.0)
        # The totals agree algebraically; the row total is the shared residual R.
        if row_total == 0.0:
            if column_total != 0.0:
                valid[candidate, boundary] = False
            continue
        total_difference = abs(row_total - column_total)
        if (total_difference > feasibility_tolerance * max(row_total, column_total)
                and total_difference > feasibility_tolerance):
            valid[candidate, boundary] = False
        residual_total = row_total
        a = np.empty(states, dtype=np.float64)
        b = np.empty(states, dtype=np.float64)
        for state in range(states):
            a[state] = row[state] / residual_total
            b[state] = column[state] / residual_total
        b_sum = np.sum(b)
        if b_sum <= 0.0 or not np.isfinite(b_sum):
            valid[candidate, boundary] = False
            continue
        # b is not renormalised independently (that could erase a positive one-ULP Hall gap).
        hall_index = -1
        max_load = -1.0
        max_l = 0.0
        for state in range(states):
            load = a[state] + b[state]
            if load > 1.0 + feasibility_tolerance:
                valid[candidate, boundary] = False
            if load > max_load:
                max_load = load
            root_sum = math.sqrt(a[state]) + math.sqrt(b[state])
            max_l = max(max_l, root_sum * root_sum)
        # The Hall state (release 2.1): among the states whose load is within HALL_ULPS of the maximum (tied up to
        # rounding, e.g. the two states of an exchange), the one with the largest mass across the boundary, then the
        # lowest state (HUB_MASS_TIE).
        load_floor = max_load - HALL_ULPS * epsilon * max(1.0, max_load)
        hub_mass = -1.0
        for state in range(states):
            if a[state] + b[state] >= load_floor:
                hub_mass = max(hub_mass, pi[candidate, boundary, state] + pi[candidate, boundary + 1, state])
        for rank in range(states):
            state = states - 1 - rank if reverse else rank
            if (a[state] + b[state] >= load_floor
                    and pi[candidate, boundary, state] + pi[candidate, boundary + 1, state] >= hub_mass - HUB_MASS_TIE):
                hall_index = state
                break
        if not valid[candidate, boundary] or max_l <= 0.0:
            continue
        complement_a = 0.0
        complement_b = 0.0
        for state in range(states):
            if state != hall_index:
                complement_a += a[state]
                complement_b += b[state]
        gap_from_a = complement_a - b[hall_index]
        gap_from_b = complement_b - a[hall_index]
        if gap_from_a < -feasibility_tolerance or gap_from_b < -feasibility_tolerance:
            valid[candidate, boundary] = False
            continue
        hall_gap = max(0.0, gap_from_a, gap_from_b)
        equality = (gap_from_a <= HALL_ULPS * epsilon * max(complement_a, b[hall_index])
                    and gap_from_b <= HALL_ULPS * epsilon * max(complement_b, a[hall_index]))

        chosen_large = -1
        q = 0.0
        if states == 2 or hall_gap == 0.0 or equality:
            # Exact Hall equality has a unique star support (always for S = 2); gaps within HALL_ULPS of their terms
            # are that equality up to rounding (release 2.1).
            branch[candidate, boundary] = 3
            source_mass = row_mass[hall_index]
            operator_pivot[candidate, boundary] = hall_index
            beta[candidate, boundary, hall_index] = 1.0
            for state in range(states):
                if state == hall_index:
                    continue
                row_arm[candidate, boundary, state] = (residual_total * b[state] / source_mass
                                                       if source_mass > 0.0 else 0.0)
                state_mass = row_mass[state]
                # R a reconstructs this row's outgoing flow; a subnormal row uses the flow itself.
                off_mass = (residual_total * a[state]
                            if pi[candidate, boundary, state] >= smallest_normal else row[state])
                alpha[candidate, boundary, state] = off_mass / state_mass if state_mass > 0.0 else 0.0
        else:
            q_max = 1.0 / max_l
            all_small_at_max = _branch_function(q_max, a, b, -1)
            # A maximum load rounded to one with positive complementary slack is the limiting
            # one-large branch, not the singular all-small endpoint.
            rounded_positive_hall_gap = max_load >= 1.0 and hall_gap > 0.0
            if all_small_at_max >= -feasibility_tolerance and not rounded_positive_hall_gap:
                q, _ = _bisect_root(a, b, -1, q_max, max_iterations)
                branch[candidate, boundary] = 1
            else:
                # Tied maximisers of L first, then every state.
                selected = -1
                for search_pass in range(2):
                    for rank in range(states):
                        state = states - 1 - rank if reverse else rank
                        root_sum = math.sqrt(a[state]) + math.sqrt(b[state])
                        state_l = root_sum * root_sum
                        if search_pass == 0 and (max_l - state_l
                                                 > feasibility_tolerance * max(1.0, max_l)):
                            continue
                        at_zero = _branch_function(0.0, a, b, state)
                        at_max = _branch_function(q_max, a, b, state)
                        if at_zero >= -feasibility_tolerance and at_max <= feasibility_tolerance:
                            selected = state
                            break
                    if selected >= 0:
                        break
                if selected < 0:
                    # No large branch brackets a root: H0(q_max) < 0 is summation drift at the
                    # adjoining all-small endpoint; use that same maximum-entropy endpoint.
                    q = q_max
                    branch[candidate, boundary] = 1
                else:
                    chosen_large = selected
                    q, _ = _bisect_root(a, b, chosen_large, q_max, max_iterations)
                    branch[candidate, boundary] = 2

            u = np.empty(states, dtype=np.float64)
            qstar = np.empty(states, dtype=np.float64)
            for state in range(states):
                u[state] = _small_diagonal_product(q, a[state], b[state])
                qstar[state] = b[state] + u[state]
            small_total = 0.0
            if chosen_large >= 0:
                for state in range(states):
                    if state != chosen_large:
                        small_total += qstar[state]
                for state in range(states):
                    if state == chosen_large:
                        beta[candidate, boundary, state] = max(0.0, 1.0 - q * small_total)
                    else:
                        beta[candidate, boundary, state] = q * qstar[state]
            else:
                for state in range(states):
                    beta[candidate, boundary, state] = qstar[state]
            # The pivot: the lowest state whose beta is within HALL_ULPS of the largest (release 2.1).
            top_beta = beta[candidate, boundary, 0]
            for state in range(1, states):
                top_beta = max(top_beta, beta[candidate, boundary, state])
            pivot = 0
            for rank in range(states):
                state = states - 1 - rank if reverse else rank
                if beta[candidate, boundary, state] >= top_beta - HALL_ULPS * epsilon * abs(top_beta):
                    pivot = state
                    break
            pivot_value = beta[candidate, boundary, pivot]
            if pivot_value <= 0.0 or not np.isfinite(pivot_value):
                valid[candidate, boundary] = False
                continue
            operator_pivot[candidate, boundary] = pivot
            beta[candidate, boundary] /= pivot_value
            for source in range(states):
                source_mass = row_mass[source]
                off_fraction = row[source] / source_mass if source_mass > 0.0 else 0.0
                if source == pivot and pivot == chosen_large:
                    denominator = small_total
                    if denominator <= 0.0:
                        if off_fraction > feasibility_tolerance:
                            valid[candidate, boundary] = False
                        continue
                    for destination in range(states):
                        if destination != source:
                            row_arm[candidate, boundary, destination] = (
                                off_fraction * qstar[destination] / denominator)
                    continue
                denominator = 0.0
                for destination in range(states):
                    if destination != source:
                        denominator += beta[candidate, boundary, destination]
                if denominator <= 0.0:
                    if off_fraction > feasibility_tolerance:
                        valid[candidate, boundary] = False
                    continue
                if source == pivot:
                    for destination in range(states):
                        if destination != source:
                            row_arm[candidate, boundary, destination] = (
                                off_fraction * beta[candidate, boundary, destination] / denominator)
                else:
                    alpha[candidate, boundary, source] = off_fraction / denominator

        # Original-scale check: reachable rows are stochastic, the adjacent marginal and the
        # diagonal joint flow are reproduced, unreachable rows are canonical identity rows.
        post_tolerance = max(8.0 * tolerance, 256.0 * epsilon)
        h = operator_pivot[candidate, boundary]
        maximum_residual = 0.0
        for source in range(states):
            source_mass = pi[candidate, boundary, source]
            if row_mass[source] == 0.0:
                stay[candidate, boundary, source] = 1.0
                alpha[candidate, boundary, source] = 0.0
                column_arm[candidate, boundary, source] = 0.0
            row_probability = stay[candidate, boundary, source]
            if row_probability < -post_tolerance:
                valid[candidate, boundary] = False
            for destination in range(states):
                if source != destination:
                    row_probability += alpha[candidate, boundary, source] * beta[candidate, boundary, destination]
                    if source == h:
                        row_probability += row_arm[candidate, boundary, destination]
                    if destination == h:
                        row_probability += column_arm[candidate, boundary, source]
            if row_mass[source] > 0.0:
                maximum_residual = max(maximum_residual, abs(row_probability - 1.0))
            maximum_residual = max(maximum_residual, abs(
                source_mass * stay[candidate, boundary, source]
                - diagonal_flow[candidate, boundary, source]))
        for destination in range(states):
            observed = pi[candidate, boundary, destination] * stay[candidate, boundary, destination]
            for source in range(states):
                if source == destination:
                    continue
                probability = alpha[candidate, boundary, source] * beta[candidate, boundary, destination]
                if source == h:
                    probability += row_arm[candidate, boundary, destination]
                if destination == h:
                    probability += column_arm[candidate, boundary, source]
                if probability < -post_tolerance:
                    valid[candidate, boundary] = False
                observed += pi[candidate, boundary, source] * probability
            maximum_residual = max(maximum_residual,
                                   abs(observed - pi[candidate, boundary + 1, destination]))
        if maximum_residual > post_tolerance:
            valid[candidate, boundary] = False
    return stay, alpha, beta, row_arm, column_arm, branch, operator_pivot, valid


def informative_marker_counts(gl: np.ndarray, observed: np.ndarray, bin_start: np.ndarray,
                              bin_stop: np.ndarray) -> np.ndarray:
    """int64 [N, B]: observed informative markers (GL spread > INFORMATIVE_GL_SPREAD) per bin."""
    informative = observed & (np.ptp(gl, axis=2) > INFORMATIVE_GL_SPREAD)
    cumulative = np.zeros((gl.shape[0], gl.shape[1] + 1), dtype=np.int64)
    np.cumsum(informative, axis=1, out=cumulative[:, 1:])
    return cumulative[:, bin_stop] - cumulative[:, bin_start]


def project(component: PreparedComponent, posterior: np.ndarray, *, source_gap_support_sites: int,
            bridge_tolerance: float, bridge_max_iterations: int) -> Projection:
    """Projected model of one component from its exact source marginals [N, B, S, S].

    ``posterior`` is overwritten (homologue-symmetrised, then the selected chain's propagation).
    """
    factors, transition = component.factors, component.transition
    candidates, bins, states, _ = posterior.shape
    bin_stop = np.cumsum(component.bin_sizes).astype(np.int64)
    bin_start = bin_stop - component.bin_sizes
    background = background_alt_frequency(component.named_alleles)
    state_alt = np.empty((states, component.informative_markers), dtype=np.float64)
    state_alt[:-1] = state_alt_probabilities(component.named_alleles, background)
    state_alt[-1] = background
    gl, observed = component.genotype_likelihoods, component.observed
    if np.any(~_symmetrise(posterior)):
        raise FloatingPointError("source posterior marginal lost all mass")
    selector = np.broadcast_to(component.selector_switch_probability,
                               (candidates, bins - 1)).copy()
    excursion = np.zeros((candidates, bins - 1), dtype=np.float64)
    if source_gap_support_sites > 0 and bins > 1:
        # A parent bin without informative markers carries only the painter's interpolation (its copying
        # process; before 2.14 an imposed switch-suppression prior): such bridges take the meiotic
        # switch probability k as a uniform excursion weight, scaled by 1 - min(1, n / N) for the smaller
        # adjacent count n.
        per_bin = informative_marker_counts(gl, observed, bin_start, bin_stop)
        adjacent = np.minimum(per_bin[:, :-1], per_bin[:, 1:])
        excursion = selector * (1.0 - np.minimum(1.0, adjacent / source_gap_support_sites))
    same, one, two = transition.same, transition.one_change, transition.two_changes
    diagonal_flow, outgoing, incoming, valid = _selected_flows(
        posterior, factors.right_weight, same, one, two, selector)
    if np.any(~valid):
        raise FloatingPointError("selected-oriented posterior propagation lost mass")
    pi = np.ascontiguousarray(np.sum(posterior, axis=3))
    site_to_bin = np.repeat(np.arange(bins, dtype=np.int64), component.bin_sizes)
    candidate_alt = _transmitted_alt(gl, observed, state_alt, posterior, site_to_bin, states - 1,
                                     float(component.robustness_epsilon))
    stay, alpha, beta, row_arm, column_arm, branch, pivot, valid = _bridges(
        pi, diagonal_flow, outgoing, incoming, float(bridge_tolerance), int(bridge_max_iterations))
    if np.any(~valid):
        raise FloatingPointError("persistence-constrained projected bridge did not converge")
    return Projection(pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot, branch,
                      np.ascontiguousarray(excursion), state_alt, bin_start, bin_stop,
                      factors.available)
