"""Source-marginal projection and persistence-constrained bridge kernels."""

import math
import numpy as np
from numba import prange
from..core import parallel

parallel.ensure_numba_registry_warmup()
njit = parallel.original_njit


@njit(cache=True, parallel=True, fastmath=False)
def _symmetrise_source_marginals(posterior):
    candidates, bins, states, _ = posterior.shape
    valid = np.ones(candidates, dtype=np.bool_)
    for candidate in prange(candidates):
        for block in range(bins):
            for first in range(states):
                for second in range(first + 1, states):
                    value = 0.5 * (
                        posterior[candidate, block, first, second]
                        + posterior[candidate, block, second, first]
                    )
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


@njit(cache=True, parallel=True, fastmath=False)
def _project_transmitted_alt(
    candidate_gl,
    candidate_observed,
    state_alt,
    posterior,
    site_to_bin,
    background_index,
    epsilon,
):
    """Collapse the untransmitted source state without an S**2 site array."""

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
        # Deterministic transmitted alleles cannot change upon conditioning.
        if q_first == 0.0 or q_first == 1.0:
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
            conditional = (
                (w1 * p10 + w2 * p11) / denominator
                if denominator > 0.0
                else q_first
            )
            mass = posterior[candidate, block, first, second]
            marginal += mass
            numerator += mass * conditional
        result[candidate, site, first] = (
            numerator / marginal if marginal > 0.0 else q_first
        )
    return result


@njit(cache=True, fastmath=False)
def _positive_exclusion_sums(matrix):
    """Positive row/column/complement leave-one-out sums in O(S**2)."""

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
            row_without[row, column] = (
                prefix[column] + suffix[column + 1]
            )
    for column in range(states):
        prefix[0] = 0.0
        for row in range(states):
            prefix[row + 1] = prefix[row] + matrix[row, column]
        suffix[states] = 0.0
        for row in range(states - 1, -1, -1):
            suffix[row] = suffix[row + 1] + matrix[row, column]
        for row in range(states):
            column_without[row, column] = (
                prefix[row] + suffix[row + 1]
            )
    for row in range(states):
        prefix[0] = 0.0
        for column in range(states):
            prefix[column + 1] = (
                prefix[column] + column_without[row, column]
            )
        suffix[states] = 0.0
        for column in range(states - 1, -1, -1):
            suffix[column] = (
                suffix[column + 1] + column_without[row, column]
            )
        for column in range(states):
            complement[row, column] = (
                prefix[column] + suffix[column + 1]
            )
    return row_without, column_without, complement


@njit(cache=True, parallel=True, fastmath=False)
def _oriented_posterior_and_diagonal_flow(
    posterior, right_weight, same, one, two, selector
):
    """Propagate selected diplotypes and positive adjacent selected-state flows."""

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
            rho_row_without, rho_column_without, rho_complement = (
                _positive_exclusion_sums(rho)
            )
            scaled = np.empty((states, states), dtype=np.float64)
            for first in range(states):
                for second in range(states):
                    cell = rho[first, second]
                    denominator = (
                        w0 * cell
                        + w1 * (rho_row_without[first, second]
                                + rho_column_without[first, second])
                        + w2 * rho_complement[first, second]
                    )
                    if denominator <= 0.0 or not np.isfinite(denominator):
                        valid[candidate, boundary] = False
                        scaled[first, second] = 0.0
                    else:
                        scaled[first, second] = (
                            posterior[candidate, boundary, first, second]
                            / denominator
                        )

            tau = selector[candidate, boundary]
            for first in range(states):
                direct_same = 0.0
                swapped_same = 0.0
                outgoing_direct = 0.0
                outgoing_swapped = 0.0
                swap_change_base = (
                    w1 * rho_row_without[first, first]
                    + w2 * rho_complement[first, first]
                )
                for second in range(states):
                    u = scaled[first, second]
                    direct_same += u * (
                        w1 * row_rho[first]
                        + (w0 - w1) * rho[first, second]
                    )
                    if second == first:
                        swapped_same += u * (
                            w1 * column_rho[first]
                            + (w0 - w1) * rho[first, first]
                        )
                    else:
                        swapped_same += u * (
                            w2 * column_rho[first]
                            + (w1 - w2) * rho[first, first]
                        )

                    direct_change = (
                        w1 * rho_column_without[first, second]
                        + w2 * rho_complement[first, second]
                    )
                    swap_change = swap_change_base
                    if second != first:
                        swap_change += (
                            (w0 - w1) * rho[first, second]
                            + (w1 - w2) * rho_column_without[first, second]
                        )
                    outgoing_direct += u * direct_change
                    outgoing_swapped += u * swap_change

                diagonal_flow[candidate, boundary, first] = (
                    (1.0 - tau) * direct_same + tau * swapped_same
                )
                outgoing_change_flow[candidate, boundary, first] = (
                    (1.0 - tau) * outgoing_direct
                    + tau * outgoing_swapped
                )

            scaled_row_without, scaled_column_without, scaled_complement = (
                _positive_exclusion_sums(scaled)
            )
            for destination in range(states):
                other_source_total = 0.0
                for other in range(states):
                    other_source_total += scaled_column_without[
                        destination, other
                    ]
                direct_incoming = w2 * row_rho[destination] * other_source_total
                correction = 0.0
                for other in range(states):
                    correction += (rho[destination, other]
                                   * scaled_column_without[destination, other])
                direct_incoming += (w1 - w2) * correction

                swapped_incoming = 0.0
                for source in range(states):
                    if source == destination:
                        continue
                    source_to_destination = scaled[source, destination]
                    source_other = scaled_row_without[source, destination]
                    swapped_incoming += (
                        column_rho[destination]
                        * (
                            w1 * source_to_destination
                            + w2 * source_other
                        )
                        + rho[source, destination]
                        * (
                            (w0 - w1) * source_to_destination
                            + (w1 - w2) * source_other
                        )
                    )
                incoming_change_flow[candidate, boundary, destination] = (
                    (1.0 - tau) * direct_incoming
                    + tau * swapped_incoming
                )

            direct = np.empty((states, states), dtype=np.float64)
            for first in range(states):
                for second in range(states):
                    cell = scaled[first, second]
                    direct[first, second] = rho[first, second] * (
                        w0 * cell
                        + w1 * (scaled_row_without[first, second]
                                + scaled_column_without[first, second])
                        + w2 * scaled_complement[first, second]
                    )
            total = 0.0
            for first in range(states):
                for second in range(states):
                    value = (
                        (1.0 - tau) * direct[first, second]
                        + tau * direct[second, first]
                    )
                    posterior[candidate, boundary + 1, first, second] = value
                    total += value
            if total <= 0.0 or not np.isfinite(total):
                valid[candidate, boundary] = False
            else:
                posterior[candidate, boundary + 1] /= total
                diagonal_flow[candidate, boundary] /= total
                outgoing_change_flow[candidate, boundary] /= total
                incoming_change_flow[candidate, boundary] /= total
    return (
        diagonal_flow,
        outgoing_change_flow,
        incoming_change_flow,
        valid,
    )


@njit(cache=True, inline="always", fastmath=False)
def _small_maxent_diagonal_product(q, a, b):
    """Stable small root for the excluded diagonal product x_i*y_i."""

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


@njit(cache=True, inline="always", fastmath=False)
def _maxent_branch_function(q, a, b, large_index):
    total_u = 0.0
    large_u = 0.0
    for state in range(len(a)):
        value = _small_maxent_diagonal_product(q, a[state], b[state])
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
    base_from_a = complement_a - b[large_index]
    base_from_b = complement_b - a[large_index]
    # The larger independently accumulated complement preserves tiny positive
    # Hall slack when 1-a_h-b_h would round to zero.
    base = max(0.0, base_from_a, base_from_b)
    return base + total_u - 2.0 * large_u


@njit(cache=True, fastmath=False)
def _bisect_maxent_root(a, b, large_index, q_max, max_iterations):
    lower = 0.0
    upper = q_max
    lower_value = _maxent_branch_function(lower, a, b, large_index)
    upper_value = _maxent_branch_function(upper, a, b, large_index)
    increasing = large_index < 0
    used = 0
    for iteration in range(min(128, max_iterations)):
        midpoint = 0.5 * (lower + upper)
        if midpoint == lower or midpoint == upper:
            break
        value = _maxent_branch_function(midpoint, a, b, large_index)
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


@njit(cache=True, parallel=True, fastmath=False)
def _maximum_entropy_bridge_kernel(
    pi, diagonal_flow, outgoing_change_flow, incoming_change_flow,
    tolerance, max_iterations,
):
    """Solve every normalized zero-diagonal maximum-entropy completion."""

    candidates, bins, states = pi.shape
    boundaries = bins - 1
    stay = np.zeros((candidates, boundaries, states), dtype=np.float64)
    alpha = np.zeros_like(stay)
    beta = np.zeros_like(stay)
    row_arm = np.zeros_like(stay)
    column_arm = np.zeros_like(stay)
    branch = np.zeros((candidates, boundaries), dtype=np.int8)
    large_index = np.full((candidates, boundaries), -1, dtype=np.int64)
    iterations = np.zeros((candidates, boundaries), dtype=np.int64)
    operator_pivot = np.full((candidates, boundaries), -1, dtype=np.int64)
    residuals = np.zeros((candidates, boundaries), dtype=np.float64)
    valid = np.ones((candidates, boundaries), dtype=np.bool_)
    epsilon = np.finfo(np.float64).eps
    feasibility_tolerance = 64.0 * epsilon

    for task in prange(candidates * boundaries):
        candidate = task // boundaries
        boundary = task % boundaries
        row = np.empty(states, dtype=np.float64)
        column = np.empty(states, dtype=np.float64)
        row_total = 0.0
        column_total = 0.0
        for state in range(states):
            source_mass = pi[candidate, boundary, state]
            diagonal_mass = diagonal_flow[candidate, boundary, state]
            row[state] = outgoing_change_flow[candidate, boundary, state]
            column[state] = incoming_change_flow[candidate, boundary, state]
            if row[state] < -feasibility_tolerance:
                valid[candidate, boundary] = False
            if column[state] < -feasibility_tolerance:
                valid[candidate, boundary] = False
            row[state] = max(0.0, row[state])
            column[state] = max(0.0, column[state])
            row_total += row[state]
            column_total += column[state]
            stay[candidate, boundary, state] = (
                diagonal_mass / source_mass if source_mass > 0.0 else 1.0
            )

        # The totals are identical algebraically. Retain every positive row
        # residual, including subnormal-scale totals, and use it as the shared R.
        if row_total == 0.0:
            if column_total != 0.0:
                valid[candidate, boundary] = False
            continue
        total_difference = abs(row_total - column_total)
        if total_difference > feasibility_tolerance * max(row_total, column_total):
            # Absolute posterior normalization drift can dominate an extremely
            # tiny residual. It is checked again against the original marginals.
            if total_difference > feasibility_tolerance:
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
        # Do not independently renormalize b: doing so can erase a positive
        # one-ULP Hall gap. The shared R and original-scale postcheck are
        # authoritative for the harmless summation drift.

        hall_index = -1
        max_load = -1.0
        max_l = 0.0
        for state in range(states):
            load = a[state] + b[state]
            if load > 1.0 + feasibility_tolerance:
                valid[candidate, boundary] = False
            if load > max_load:
                max_load = load
                hall_index = state
            root_sum = math.sqrt(a[state]) + math.sqrt(b[state])
            max_l = max(max_l, root_sum * root_sum)
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
        if (
            gap_from_a < -feasibility_tolerance
            or gap_from_b < -feasibility_tolerance
        ):
            valid[candidate, boundary] = False
            continue
        # Taking the larger independently evaluated complement gap preserves
        # any positive one-ULP slack that one evaluation still resolves.
        hall_gap = max(0.0, gap_from_a, gap_from_b)

        chosen_large = -1
        q = 0.0
        used = 0
        if states == 2 or hall_gap == 0.0:
            # Exact Hall equality has a unique star support. S=2 always has
            # this support; a positive complementary gap is never a star.
            branch[candidate, boundary] = 3
            large_index[candidate, boundary] = hall_index
            source_mass = pi[candidate, boundary, hall_index]
            operator_pivot[candidate, boundary] = hall_index
            beta[candidate, boundary, hall_index] = 1.0
            for state in range(states):
                if state == hall_index:
                    continue
                row_arm[candidate, boundary, state] = (
                    residual_total * b[state] / source_mass
                    if source_mass > 0.0 else 0.0
                )
                state_mass = pi[candidate, boundary, state]
                alpha[candidate, boundary, state] = (
                    residual_total * a[state] / state_mass
                    if state_mass > 0.0 else 0.0
                )
        else:
            q_max = 1.0 / max_l
            all_small_at_max = _maxent_branch_function(q_max, a, b, -1)
            # If direct addition rounded the maximum load to one but the
            # complementary calculation retains positive slack, this is the
            # limiting one-large branch, not the singular all-small endpoint.
            rounded_positive_hall_gap = (
                max_load >= 1.0 and hall_gap > 0.0
            )
            if (
                all_small_at_max >= -feasibility_tolerance
                and not rounded_positive_hall_gap
            ):
                q, used = _bisect_maxent_root(
                    a, b, -1, q_max, max_iterations
                )
                branch[candidate, boundary] = 1
                # The one-large root normally belongs to a maximizer of L.
            else:
                # Try tied maximizers first, then every state defensively.
                selected = -1
                for search_pass in range(2):
                    for state in range(states):
                        root_sum = math.sqrt(a[state]) + math.sqrt(b[state])
                        state_l = root_sum * root_sum
                        if search_pass == 0 and (
                            max_l - state_l
                            > feasibility_tolerance * max(1.0, max_l)
                        ):
                            continue
                        at_zero = _maxent_branch_function(0.0, a, b, state)
                        at_max = _maxent_branch_function(q_max, a, b, state)
                        if at_zero >= -feasibility_tolerance and at_max <= feasibility_tolerance:
                            selected = state
                            break
                    if selected >= 0:
                        break
                if selected < 0:
                    # No large branch brackets a root: H0(qmax)'s negative
                    # sign is summation drift at the adjoining all-small
                    # endpoint. Use that same maximum-entropy endpoint.
                    q = q_max
                    branch[candidate, boundary] = 1
                else:
                    chosen_large = selected
                    q, used = _bisect_maxent_root(
                        a, b, chosen_large, q_max, max_iterations
                    )
                    branch[candidate, boundary] = 2
                    large_index[candidate, boundary] = chosen_large
            iterations[candidate, boundary] = used


            u = np.empty(states, dtype=np.float64)
            qstar = np.empty(states, dtype=np.float64)
            for state in range(states):
                u[state] = _small_maxent_diagonal_product(
                    q, a[state], b[state]
                )
                qstar[state] = b[state] + u[state]

            small_total = 0.0
            if chosen_large >= 0:
                for state in range(states):
                    if state != chosen_large:
                        small_total += qstar[state]
                for state in range(states):
                    if state == chosen_large:
                        beta[candidate, boundary, state] = max(
                            0.0, 1.0 - q * small_total
                        )
                    else:
                        beta[candidate, boundary, state] = q * qstar[state]
            else:
                for state in range(states):
                    beta[candidate, boundary, state] = qstar[state]

            pivot = 0
            for state in range(1, states):
                if (
                    beta[candidate, boundary, state]
                    > beta[candidate, boundary, pivot]
                ):
                    pivot = state
            pivot_value = beta[candidate, boundary, pivot]
            if pivot_value <= 0.0 or not np.isfinite(pivot_value):
                valid[candidate, boundary] = False
                continue
            operator_pivot[candidate, boundary] = pivot
            beta[candidate, boundary] /= pivot_value

            for source in range(states):
                source_mass = pi[candidate, boundary, source]
                off_fraction = (
                    row[source] / source_mass if source_mass > 0.0 else 0.0
                )
                if source == pivot and pivot == chosen_large:
                    denominator = small_total
                    if denominator <= 0.0:
                        if off_fraction > feasibility_tolerance:
                            valid[candidate, boundary] = False
                        continue
                    for destination in range(states):
                        if destination != source:
                            row_arm[candidate, boundary, destination] = (
                                off_fraction * qstar[destination] / denominator
                            )
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
                                off_fraction
                                * beta[candidate, boundary, destination]
                                / denominator
                            )
                else:
                    alpha[candidate, boundary, source] = (
                        off_fraction / denominator
                    )

        # Original-scale release check: reachable rows are stochastic, the
        # requested adjacent marginal and diagonal joint flow are exact, and
        # unreachable rows are canonical identity rows.
        post_tolerance = max(8.0 * tolerance, 256.0 * epsilon)
        h = operator_pivot[candidate, boundary]
        maximum_residual = 0.0
        for source in range(states):
            source_mass = pi[candidate, boundary, source]
            if source_mass == 0.0:
                stay[candidate, boundary, source] = 1.0
                alpha[candidate, boundary, source] = 0.0
                column_arm[candidate, boundary, source] = 0.0
            row_probability = stay[candidate, boundary, source]
            if row_probability < -post_tolerance:
                valid[candidate, boundary] = False

            for destination in range(states):
                if source != destination:
                    row_probability += (
                        alpha[candidate, boundary, source]
                        * beta[candidate, boundary, destination]
                    )
                    if source == h:
                        row_probability += row_arm[
                            candidate, boundary, destination
                        ]
                    if destination == h:
                        row_probability += column_arm[
                            candidate, boundary, source
                        ]
            if source_mass > 0.0:
                maximum_residual = max(
                    maximum_residual, abs(row_probability - 1.0)
                )
            diagonal_observed = (
                source_mass * stay[candidate, boundary, source]
            )
            maximum_residual = max(
                maximum_residual,
                abs(
                    diagonal_observed
                    - diagonal_flow[candidate, boundary, source]
                ),
            )
        for destination in range(states):
            observed = (
                pi[candidate, boundary, destination]
                * stay[candidate, boundary, destination]
            )
            for source in range(states):
                if source == destination:
                    continue
                probability = (
                    alpha[candidate, boundary, source]
                    * beta[candidate, boundary, destination]
                )
                if source == h:
                    probability += row_arm[candidate, boundary, destination]
                if destination == h:
                    probability += column_arm[candidate, boundary, source]
                if probability < -post_tolerance:
                    valid[candidate, boundary] = False
                observed += pi[candidate, boundary, source] * probability
            maximum_residual = max(
                maximum_residual,
                abs(observed - pi[candidate, boundary + 1, destination]),
            )
        residuals[candidate, boundary] = maximum_residual
        if maximum_residual > post_tolerance:
            valid[candidate, boundary] = False

    return (
        stay,
        alpha,
        beta,
        row_arm,
        column_arm,
        branch,
        large_index,
        operator_pivot,
        iterations,
        residuals,
        valid,
    )
