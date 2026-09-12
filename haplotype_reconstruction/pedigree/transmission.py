"""pedigree / transmission for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import dataclass
import math
import time
from numba import prange
import numpy as np
import haplotype_reconstruction.core.parallel as core_parallel

core_parallel.ensure_numba_registry_warmup()


njit = core_parallel.original_njit


APPROXIMATION_NAME = "transmitted-marginal-persistence-maxent-quadratic"

# Temporary per-call cache, tiled over children; not a scientific setting.
_COMMON_EMISSION_CACHE_BYTES = 256 << 20


@dataclass(frozen=True)
class ProjectedRaggedQuadraticModel:
    """Candidate-specific O(S) projection retained for quadratic scoring."""

    transmitted_state_probability: np.ndarray  # candidates, bins, S
    transmitted_alt_probability: np.ndarray  # candidates, selected sites, S
    bridge_stay_probability: np.ndarray  # candidates, boundaries, S
    bridge_offdiag_left_scale: np.ndarray  # candidates, boundaries, S
    bridge_offdiag_right_scale: np.ndarray  # candidates, boundaries, S
    bridge_row_arm: np.ndarray  # candidates, boundaries, S
    bridge_column_arm: np.ndarray  # candidates, boundaries, S
    bridge_branch: np.ndarray  # 0 diagonal, 1 all-small, 2 one-large, 3 star
    transition: object
    bridge_large_index: np.ndarray  # -1 unless one-large/star
    bridge_operator_pivot: np.ndarray  # bounded-factor operator pivot
    bridge_same_state_joint_mass: np.ndarray  # candidates, boundaries, S
    state_alt_probability: np.ndarray  # S, selected sites (bin-major)
    selected_input_order: np.ndarray
    bin_start: np.ndarray
    bin_stop: np.ndarray
    available: np.ndarray
    informative_site_count: np.ndarray
    sinkhorn_iterations: np.ndarray
    maximum_bridge_marginal_residual: float
    preparation_seconds: float

    @property
    def n_candidates(self) -> int:
        return int(self.transmitted_state_probability.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.transmitted_state_probability.shape[1])

    @property
    def n_states(self) -> int:
        return int(self.transmitted_state_probability.shape[2])

    @property
    def n_sites(self) -> int:
        return int(self.transmitted_alt_probability.shape[1])

    @property
    def retained_bytes(self) -> int:
        arrays = (
            self.transmitted_state_probability,
            self.transmitted_alt_probability,
            self.bridge_stay_probability,
            self.bridge_offdiag_left_scale,
            self.bridge_offdiag_right_scale,
            self.bridge_row_arm,
            self.bridge_column_arm,
            self.bridge_branch,
            self.bridge_large_index,
            self.bridge_operator_pivot,
            self.bridge_same_state_joint_mass,
            self.state_alt_probability,
            self.selected_input_order,
            self.bin_start,
            self.bin_stop,
            self.available,
            self.informative_site_count,
            self.sinkhorn_iterations,
        )
        return int(sum(value.nbytes for value in arrays))


@dataclass(frozen=True)
class QuadraticComplexityDiagnostic:
    """Explicit state/workspace contract for every child, edge, and trio."""

    n_states: int
    projected_hidden_state_count: int
    exact_m2_hidden_state_count: int
    float64_workspace_values_per_task: int
    peak_working_bytes_per_task: int
    time_complexity_per_task: str
    memory_complexity_per_task: str


@dataclass(frozen=True)
class ProjectedRaggedQuadraticScores:
    """Projected M0/M1/M2 batch scores and approximation diagnostics."""

    zero_observed: np.ndarray
    one_observed: np.ndarray
    two_observed: np.ndarray
    candidate_source_available: np.ndarray
    candidate_source_informative_site_count: np.ndarray
    child_informative_site_count: np.ndarray
    m2_active_trio_count: int
    m2_reduced_trio_count: int
    approximation_name: str
    complexity: QuadraticComplexityDiagnostic
    projection_retained_bytes: int
    maximum_bridge_marginal_residual: float
    projection_seconds: float
    scoring_seconds: float
    m0_scoring_seconds: float
    m1_scoring_seconds: float
    m2_scoring_seconds: float
    reused_lower_order_scores: bool


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


def prepare_projected_ragged_quadratic(
    factors: pedigree_sources.RaggedSourceBatchFactors,
    model,
    candidate_genotype_likelihoods: np.ndarray,
    candidate_observed: np.ndarray,
    *,
    selected_site_indices: np.ndarray | None = None,
    precomputed_source_marginals: np.ndarray | None = None,
    candidate_selector_switch_probability=0.01,
    sinkhorn_tolerance: float = 2e-13,
    sinkhorn_max_iterations: int = 128,
) -> ProjectedRaggedQuadraticModel:
    """Prepare the transmitted-marginal approximation from exact factors."""

    started = time.perf_counter()
    candidates = factors.n_candidates
    states = factors.n_states
    bins = factors.n_bins
    if int(model.n_states) != states or int(model.n_bins) != bins:
        raise ValueError("model, transition factors, and bin counts disagree")
    order, state_alt, bin_start, bin_stop = pedigree_sources._compact_model_arrays(
        model, selected_site_indices
    )
    candidate_gl = pedigree_sources._normalise_gl(candidate_genotype_likelihoods, "candidate GL")
    if candidate_gl.shape[:2] != (candidates, len(order)):
        raise ValueError("candidate GL shape disagrees with factors/model")
    observed = np.asarray(candidate_observed, dtype=np.bool_)
    if observed.shape != candidate_gl.shape[:2]:
        raise ValueError("candidate observed mask has wrong shape")
    candidate_gl = np.ascontiguousarray(candidate_gl[:, order])
    observed = np.ascontiguousarray(observed[:, order])
    if not np.isfinite(sinkhorn_tolerance) or sinkhorn_tolerance <= 0.0:
        raise ValueError("sinkhorn_tolerance must be finite and positive")
    if (
        isinstance(sinkhorn_max_iterations, bool)
        or int(sinkhorn_max_iterations) != sinkhorn_max_iterations
        or sinkhorn_max_iterations < 1
    ):
        raise ValueError("sinkhorn_max_iterations must be a positive integer")

    if precomputed_source_marginals is None:
        posterior = np.ascontiguousarray(pedigree_sources.source_posterior_marginals(factors))
    else:
        posterior = np.array(
            precomputed_source_marginals, dtype=np.float64, order="C", copy=True
        )
        if posterior.shape != (candidates, bins, states, states):
            raise ValueError("precomputed source marginals have wrong shape")
    if np.any(~_symmetrise_source_marginals(posterior)):
        raise FloatingPointError("source posterior marginal lost all mass")
    selector = pedigree_sources._probability_matrix(
        candidate_selector_switch_probability,
        candidates,
        bins - 1,
        "candidate selector switch",
    )
    transition = factors.transition
    same = np.ascontiguousarray(np.asarray(transition.same, dtype=np.float64))
    one = np.ascontiguousarray(np.asarray(transition.one_change, dtype=np.float64))
    two = np.ascontiguousarray(np.asarray(transition.two_changes, dtype=np.float64))
    if any(value.shape != (bins - 1,) for value in (same, one, two)):
        raise ValueError("factor transition boundary count is wrong")
    (
        diagonal_flow,
        outgoing_change_flow,
        incoming_change_flow,
        propagation_valid,
    ) = _oriented_posterior_and_diagonal_flow(
        posterior,
        np.ascontiguousarray(np.asarray(factors.right_weight, dtype=np.float64)),
        same,
        one,
        two,
        selector,
    )
    if np.any(~propagation_valid):
        raise FloatingPointError("selected-oriented posterior propagation lost mass")
    pi = np.ascontiguousarray(np.sum(posterior, axis=3))
    site_to_bin = np.empty(len(order), dtype=np.int64)
    for block in range(bins):
        site_to_bin[int(bin_start[block]):int(bin_stop[block])] = block
    transmitted_alt = _project_transmitted_alt(
        candidate_gl,
        observed,
        state_alt,
        posterior,
        np.ascontiguousarray(site_to_bin),
        int(model.background_index),
        float(factors.robustness_epsilon),
    )
    (
        stay,
        alpha,
        beta,
        row_arm,
        column_arm,
        bridge_branch,
        large_index,
        operator_pivot,
        iterations,
        residuals,
        valid,
    ) = (
        _maximum_entropy_bridge_kernel(
            pi,
            diagonal_flow,
            outgoing_change_flow,
            incoming_change_flow,
            float(sinkhorn_tolerance),
            int(sinkhorn_max_iterations),
        )
    )
    if np.any(~valid):
        raise FloatingPointError(
            "persistence-constrained projected bridge did not converge"
        )
    return ProjectedRaggedQuadraticModel(
        transmitted_state_probability=pi,
        transmitted_alt_probability=np.ascontiguousarray(transmitted_alt),
        bridge_stay_probability=np.ascontiguousarray(stay),
        bridge_offdiag_left_scale=np.ascontiguousarray(alpha),
        bridge_offdiag_right_scale=np.ascontiguousarray(beta),
        bridge_row_arm=np.ascontiguousarray(row_arm),
        bridge_column_arm=np.ascontiguousarray(column_arm),
        bridge_branch=np.ascontiguousarray(bridge_branch),
        bridge_large_index=np.ascontiguousarray(large_index),
        bridge_operator_pivot=np.ascontiguousarray(operator_pivot),
        bridge_same_state_joint_mass=np.ascontiguousarray(diagonal_flow),
        transition=transition,
        state_alt_probability=state_alt,
        selected_input_order=np.ascontiguousarray(order),
        bin_start=np.ascontiguousarray(bin_start),
        bin_stop=np.ascontiguousarray(bin_stop),
        available=np.ascontiguousarray(np.asarray(factors.available, dtype=np.bool_)),
        informative_site_count=np.ascontiguousarray(
            np.asarray(factors.informative_site_count, dtype=np.int64)
        ),
        sinkhorn_iterations=np.ascontiguousarray(iterations),
        maximum_bridge_marginal_residual=(
            float(np.max(residuals)) if residuals.size else 0.0
        ),
        preparation_seconds=time.perf_counter() - started,
    )


@njit(cache=True, inline="always", fastmath=False)
def _site_likelihood(gl, coefficient, first_alt, second_alt, mismatch):
    value = (
        coefficient[0]
        + coefficient[1] * (first_alt + second_alt)
        + coefficient[2] * first_alt * second_alt
    )
    if value > 1e-8:
        return value
    first = mismatch + (1.0 - 2.0 * mismatch) * first_alt
    second = mismatch + (1.0 - 2.0 * mismatch) * second_alt
    p0 = (1.0 - first) * (1.0 - second)
    p2 = first * second
    p1 = 1.0 - p0 - p2
    return 3.0 * (gl[0] * p0 + gl[1] * p1 + gl[2] * p2)


@njit(cache=True, parallel=True, fastmath=False)
def _common_emission_products(
    state_alt, child_gl, coefficient, observed_site, observed_start,
    observed_stop, mismatch, child_start, child_stop,
):
    """Share exact called-founder products; NaN requests the general path.

    A called named allele projects to the same 0/1 value in every candidate.
    Missing named alleles and BACKGROUND retain candidate-specific emissions.
    The original site order, likelihood calculation and underflow fallback
    remain authoritative. This cache never removes an HMM state.
    """
    bins = observed_start.shape[1]
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
                for slot in range(
                    observed_start[child, block], observed_stop[child, block]
                ):
                    site = observed_site[slot]
                    a, b = state_alt[first, site], state_alt[second, site]
                    if not ((a == 0.0 or a == 1.0)
                            and (b == 0.0 or b == 1.0)):
                        valid = False
                        break
                    value = _site_likelihood(
                        child_gl[child, site], coefficient[child, site],
                        a, b, mismatch,
                    )
                    if value <= 0.0:
                        valid = False
                        break
                    product *= value
                    if (not np.isfinite(product)
                            or product < 2.2250738585072014e-308):
                        valid = False
                        break
                if valid:
                    result[local_child, block, first, second] = product
    return result


@njit(cache=True, fastmath=False)
def _normalise_forward(values):
    total = np.sum(values)
    if total <= 0.0 or not np.isfinite(total):
        return -np.inf
    values /= total
    return math.log(total)


@njit(cache=True, fastmath=False)
def _apply_probability_emission(current, emission, exponent):
    maximum = np.max(emission)
    if maximum <= 0.0 or not np.isfinite(maximum):
        return -np.inf
    inverse = 1.0 / maximum
    states = current.shape[0]
    if exponent == 1.0:
        for first in range(states):
            for second in range(states):
                current[first, second] *= emission[first, second] * inverse
    else:
        for first in range(states):
            for second in range(states):
                current[first, second] *= math.pow(
                    emission[first, second] * inverse, exponent
                )
    increment = _normalise_forward(current)
    return (
        increment + exponent * math.log(maximum)
        if np.isfinite(increment)
        else -np.inf
    )


@njit(cache=True, fastmath=False)
def _apply_log_emission(current, log_emission):
    maximum = np.max(log_emission)
    if not np.isfinite(maximum):
        return -np.inf
    states = current.shape[0]
    for first in range(states):
        for second in range(states):
            current[first, second] *= math.exp(
                log_emission[first, second] - maximum
            )
    increment = _normalise_forward(current)
    return increment + maximum if np.isfinite(increment) else -np.inf


@njit(cache=True, fastmath=False)
def _apply_star_bridge_axis(current, output, stay, alpha, row_arm, pivot, axis):
    """Apply exact Hall-star support without general leave-one-out arrays.

    Only the pivot has a nonzero beta. Its two sums retain the general
    operator's ascending prefix and descending suffix order. All other
    destinations receive their diagonal and the pivot's outgoing row arm.
    """
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


@njit(cache=True, fastmath=False)
def _apply_bridge_axis(
    current, output, stay, alpha, beta, row_arm, column_arm, large, axis,
    prefix, suffix, branch=-1,
):
    """Apply one compressed bridge with cancellation-free leave-one-out sums."""

    if branch == 3:
        _apply_star_bridge_axis(current, output, stay, alpha, row_arm, large, axis)
        return

    states = current.shape[0]
    if axis == 0:
        for second in range(states):
            prefix[0] = 0.0
            for source in range(states):
                prefix[source + 1] = (
                    prefix[source] + current[source, second] * alpha[source]
                )
            suffix[states] = 0.0
            for source in range(states - 1, -1, -1):
                suffix[source] = (
                    suffix[source + 1]
                    + current[source, second] * alpha[source]
                )
            column_arm_total = 0.0
            if large >= 0:
                for source in range(states):
                    column_arm_total += (
                        current[source, second] * column_arm[source]
                    )
            for destination in range(states):
                value = (
                    stay[destination] * current[destination, second]
                    + beta[destination]
                    * (prefix[destination] + suffix[destination + 1])
                )
                if large >= 0:
                    value += current[large, second] * row_arm[destination]
                    if destination == large:
                        value += column_arm_total
                output[destination, second] = value
    else:
        for first in range(states):
            prefix[0] = 0.0
            for source in range(states):
                prefix[source + 1] = (
                    prefix[source] + current[first, source] * alpha[source]
                )
            suffix[states] = 0.0
            for source in range(states - 1, -1, -1):
                suffix[source] = (
                    suffix[source + 1]
                    + current[first, source] * alpha[source]
                )
            column_arm_total = 0.0
            if large >= 0:
                for source in range(states):
                    column_arm_total += (
                        current[first, source] * column_arm[source]
                    )
            for destination in range(states):
                value = (
                    stay[destination] * current[first, destination]
                    + beta[destination]
                    * (prefix[destination] + suffix[destination + 1])
                )
                if large >= 0:
                    value += current[first, large] * row_arm[destination]
                    if destination == large:
                        value += column_arm_total
                output[first, destination] = value


@njit(cache=True, fastmath=False)
def _apply_null_axis(current, output, diagonal, off, axis):
    states = current.shape[0]
    coefficient = diagonal - off
    if axis == 0:
        for second in range(states):
            total = 0.0
            for source in range(states):
                total += current[source, second]
            for destination in range(states):
                output[destination, second] = (
                    coefficient * current[destination, second] + off * total
                )
    else:
        for first in range(states):
            total = 0.0
            for source in range(states):
                total += current[first, source]
            for destination in range(states):
                output[first, destination] = (
                    coefficient * current[first, destination] + off * total
                )


@njit(cache=True, fastmath=False)
def _score_path(
    child,
    first_parent,
    second_parent,
    mode,
    pi,
    candidate_alt,
    bridge_left,
    bridge_right,
    bridge_diagonal,
    bridge_off,
    bridge_row_arm,
    bridge_column_arm,
    bridge_large_index,
    state_alt,
    child_gl,
    child_coefficient,
    observed_site,
    observed_start,
    observed_stop,
    exponent,
    null_diagonal,
    null_off,
    mismatch,
    common_products=None,
    first_cached_child=0,
    bridge_branch=None,
):
    states = state_alt.shape[0]
    bins = exponent.shape[1]
    current = np.empty((states, states), dtype=np.float64)
    work = np.empty_like(current)
    prefix = np.empty(states + 1, dtype=np.float64)
    suffix = np.empty(states + 1, dtype=np.float64)
    for first in range(states):
        for second in range(states):
            if mode == 0:
                current[first, second] = 1.0 / (states * states)
            elif mode == 1:
                current[first, second] = pi[first_parent, 0, first] / states
            else:
                current[first, second] = (
                    pi[first_parent, 0, first] * pi[second_parent, 0, second]
                )
    total_log = 0.0
    for block in range(bins):
        if block > 0:
            boundary = block - 1
            if mode >= 1:
                _apply_bridge_axis(
                    current,
                    work,
                    bridge_left[first_parent, boundary],
                    bridge_right[first_parent, boundary],
                    bridge_diagonal[first_parent, boundary],
                    bridge_row_arm[first_parent, boundary],
                    bridge_column_arm[first_parent, boundary],
                    bridge_large_index[first_parent, boundary],
                    0, prefix, suffix,
                    -1 if bridge_branch is None else bridge_branch[first_parent, boundary],
                )
            else:
                _apply_null_axis(
                    current,
                    work,
                    null_diagonal[child, boundary],
                    null_off[child, boundary],
                    0,
                )
            current, work = work, current
            if mode == 2:
                _apply_bridge_axis(
                    current,
                    work,
                    bridge_left[second_parent, boundary],
                    bridge_right[second_parent, boundary],
                    bridge_diagonal[second_parent, boundary],
                    bridge_row_arm[second_parent, boundary],
                    bridge_column_arm[second_parent, boundary],
                    bridge_large_index[second_parent, boundary],
                    1, prefix, suffix,
                    -1 if bridge_branch is None else bridge_branch[second_parent, boundary],
                )
            else:
                _apply_null_axis(
                    current,
                    work,
                    null_diagonal[child, boundary],
                    null_off[child, boundary],
                    1,
                )
            current, work = work, current

        eta = exponent[child, block]
        stable_product = True
        if eta == 0.0:
            work[:] = 1.0
        else:
            for first in range(states):
                for second in range(states):
                    if common_products is not None:
                        saved = common_products[
                            child - first_cached_child, block, first, second
                        ]
                        if np.isfinite(saved):
                            work[first, second] = saved
                            continue
                    product = 1.0
                    for slot in range(
                        observed_start[child, block], observed_stop[child, block]
                    ):
                        site = observed_site[slot]
                        first_alt = (
                            candidate_alt[first_parent, site, first]
                            if mode >= 1
                            else state_alt[first, site]
                        )
                        second_alt = (
                            candidate_alt[second_parent, site, second]
                            if mode == 2
                            else state_alt[second, site]
                        )
                        value = _site_likelihood(
                            child_gl[child, site],
                            child_coefficient[child, site],
                            first_alt,
                            second_alt,
                            mismatch,
                        )
                        if value <= 0.0:
                            product = 0.0
                            break
                        product *= value
                        if not np.isfinite(product) or product < 2.2250738585072014e-308:
                            stable_product = False
                            break
                    work[first, second] = product
                    if not stable_product:
                        break
                if not stable_product:
                    break
        if stable_product:
            increment = _apply_probability_emission(current, work, eta)
        else:
            for first in range(states):
                for second in range(states):
                    log_emission = 0.0
                    for slot in range(
                        observed_start[child, block], observed_stop[child, block]
                    ):
                        site = observed_site[slot]
                        first_alt = (
                            candidate_alt[first_parent, site, first]
                            if mode >= 1
                            else state_alt[first, site]
                        )
                        second_alt = (
                            candidate_alt[second_parent, site, second]
                            if mode == 2
                            else state_alt[second, site]
                        )
                        value = _site_likelihood(
                            child_gl[child, site],
                            child_coefficient[child, site],
                            first_alt,
                            second_alt,
                            mismatch,
                        )
                        if value <= 0.0:
                            log_emission = -np.inf
                            break
                        log_emission += math.log(value)
                    work[first, second] = eta * log_emission
            increment = _apply_log_emission(current, work)
        if not np.isfinite(increment):
            return -np.inf
        total_log += increment
    return total_log


@njit(cache=True, parallel=True, fastmath=False)
def _score_m0_kernel(
    pi, candidate_alt, bridge_left, bridge_right, bridge_diagonal, bridge_off,
    bridge_row_arm, bridge_column_arm, bridge_large_index,
    state_alt, child_gl, child_coefficient, observed_site, observed_start,
    observed_stop, exponent, null_diagonal, null_off, mismatch, eligible_child,
):
    output = np.empty(child_gl.shape[0], dtype=np.float64)
    for child in prange(child_gl.shape[0]):
        output[child] = (
            _score_path(
                child, 0, 0, 0, pi, candidate_alt, bridge_left, bridge_right,
                bridge_diagonal, bridge_off, bridge_row_arm, bridge_column_arm,
                bridge_large_index, state_alt, child_gl,
                child_coefficient, observed_site, observed_start, observed_stop,
                exponent, null_diagonal, null_off, mismatch,
            )
            if eligible_child[child]
            else -np.inf
        )
    return output


@njit(cache=True, parallel=True, fastmath=False)
def _score_m1_kernel(
    pi, candidate_alt, bridge_left, bridge_right, bridge_diagonal, bridge_off,
    bridge_row_arm, bridge_column_arm, bridge_large_index,
    available, state_alt, child_gl, child_coefficient, observed_site,
    observed_start, observed_stop, exponent, null_diagonal, null_off, mismatch,
    eligible_edge, m0, common_products, child_start, child_stop, bridge_branch=None,
):
    children = child_stop - child_start
    candidates = pi.shape[0]
    output = np.empty((children, candidates), dtype=np.float64)
    for task in prange(children * candidates):
        local_child = task // candidates
        child = child_start + local_child
        parent = task % candidates
        if not eligible_edge[child, parent]:
            output[local_child, parent] = -np.inf
        elif not available[parent]:
            output[local_child, parent] = m0[child]
        else:
            output[local_child, parent] = _score_path(
                child, parent, 0, 1, pi, candidate_alt, bridge_left,
                bridge_right, bridge_diagonal, bridge_off, bridge_row_arm,
                bridge_column_arm, bridge_large_index, state_alt, child_gl,
                child_coefficient, observed_site, observed_start, observed_stop,
                exponent, null_diagonal, null_off, mismatch,
                common_products, child_start, bridge_branch,
            )
    return output


@njit(cache=True, parallel=True, fastmath=False)
def _score_m2_kernel(
    trios, pi, candidate_alt, bridge_left, bridge_right, bridge_diagonal,
    bridge_off, bridge_row_arm, bridge_column_arm, bridge_large_index,
    state_alt, child_gl, child_coefficient, observed_site,
    observed_start, observed_stop, exponent, null_diagonal, null_off, mismatch,
    common_products, first_cached_child, bridge_branch=None,
):
    output = np.empty(len(trios), dtype=np.float64)
    for row in prange(len(trios)):
        child = trios[row, 0]
        first = trios[row, 1]
        second = trios[row, 2]
        output[row] = _score_path(
            child, first, second, 2, pi, candidate_alt, bridge_left,
            bridge_right, bridge_diagonal, bridge_off, bridge_row_arm,
            bridge_column_arm, bridge_large_index, state_alt, child_gl,
            child_coefficient, observed_site, observed_start, observed_stop,
            exponent, null_diagonal, null_off, mismatch,
            common_products, first_cached_child, bridge_branch,
        )
    return output


def _same_transition(first, second) -> bool:
    try:
        return (
            int(first.n_states) == int(second.n_states)
            and float(first.double_recomb_factor)
            == float(second.double_recomb_factor)
            and np.array_equal(np.asarray(first.same), np.asarray(second.same))
            and np.array_equal(
                np.asarray(first.one_change), np.asarray(second.one_change)
            )
            and np.array_equal(
                np.asarray(first.two_changes), np.asarray(second.two_changes)
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _null_transition_coefficients(projected, transition, selector):
    states = projected.n_states
    boundaries = projected.n_bins - 1
    same = np.asarray(transition.same, dtype=np.float64)
    one = np.asarray(transition.one_change, dtype=np.float64)
    two = np.asarray(transition.two_changes, dtype=np.float64)
    diagonal = (
        (1.0 - selector) * (same[None, :] + (states - 1) * one[None, :])
        + selector / states
    )
    off = (
        (1.0 - selector) * (one[None, :] + (states - 1) * two[None, :])
        + selector / states
    )
    if diagonal.shape[1] != boundaries:
        raise ValueError("transition and projection boundary counts disagree")
    return np.ascontiguousarray(diagonal), np.ascontiguousarray(off)


def score_projected_ragged_quadratic(
    projected: ProjectedRaggedQuadraticModel,
    transition,
    child_genotype_likelihoods: np.ndarray,
    child_observed: np.ndarray,
    child_information_exponent: np.ndarray,
    trios: np.ndarray,
    *,
    eligible_children: np.ndarray | None = None,
    eligible_parent_edges: np.ndarray | None = None,
    null_selector_switch_probability=0.01,
    mismatch_probability: float = 0.01,
    reuse_scores: ProjectedRaggedQuadraticScores | None = None,
    uniform_tolerance: float = 1e-12,
) -> ProjectedRaggedQuadraticScores:
    """Score projected M0/M1/M2 batches with O(S**2) task workspaces."""

    started = time.perf_counter()
    if not _same_transition(projected.transition, transition):
        raise ValueError("requested transition differs from projected transition")
    transition = projected.transition
    if reuse_scores is not None and not isinstance(
        reuse_scores, ProjectedRaggedQuadraticScores
    ):
        raise ValueError("reuse_scores must be ProjectedRaggedQuadraticScores")

    children_gl = pedigree_sources._normalise_gl(child_genotype_likelihoods, "child GL")
    children = children_gl.shape[0]
    if children_gl.shape[1] != projected.n_sites:
        raise ValueError("child GL site count disagrees with projection")
    child_seen = np.asarray(child_observed, dtype=np.bool_)
    if child_seen.shape != children_gl.shape[:2]:
        raise ValueError("child observed mask has wrong shape")
    order = projected.selected_input_order
    children_gl = np.ascontiguousarray(children_gl[:, order])
    child_seen = np.ascontiguousarray(child_seen[:, order])
    exponent = np.asarray(child_information_exponent, dtype=np.float64)
    if (
        exponent.shape != (children, projected.n_bins)
        or np.any(~np.isfinite(exponent))
        or np.any(exponent < 0.0)
    ):
        raise ValueError(
            "child information exponent must be nonnegative shape (children, bins)"
        )
    exponent = np.ascontiguousarray(exponent)
    if not np.isfinite(mismatch_probability) or not 0.0 <= mismatch_probability < 0.5:
        raise ValueError("mismatch_probability must lie in [0, .5)")
    if not np.isfinite(uniform_tolerance) or uniform_tolerance < 0.0:
        raise ValueError("uniform_tolerance must be nonnegative")

    if eligible_children is None:
        eligible_child = np.ones(children, dtype=np.bool_)
    else:
        eligible_child = np.asarray(eligible_children, dtype=np.bool_)
        if eligible_child.shape != (children,):
            raise ValueError("eligible_children must have shape (children,)")
        eligible_child = np.ascontiguousarray(eligible_child)
    if eligible_parent_edges is None:
        eligible_edge = np.ones(
            (children, projected.n_candidates), dtype=np.bool_
        )
    else:
        eligible_edge = np.asarray(eligible_parent_edges, dtype=np.bool_)
        if eligible_edge.shape != (children, projected.n_candidates):
            raise ValueError(
                "eligible_parent_edges must have shape (children, candidates)"
            )
        eligible_edge = np.ascontiguousarray(eligible_edge).copy()
    eligible_edge &= eligible_child[:, None]
    diagonal_count = min(children, projected.n_candidates)
    eligible_edge[np.arange(diagonal_count), np.arange(diagonal_count)] = False

    trio_array = np.asarray(trios, dtype=np.int64)
    if trio_array.ndim != 2 or trio_array.shape[1] != 3:
        raise ValueError("trios must have shape (rows, 3)")
    if len(trio_array) and (
        np.any(trio_array[:, 0] < 0)
        or np.any(trio_array[:, 0] >= children)
        or np.any(trio_array[:, 1:] < 0)
        or np.any(trio_array[:, 1:] >= projected.n_candidates)
        or np.any(trio_array[:, 1] == trio_array[:, 2])
    ):
        raise ValueError("trio row contains an invalid child or parent")
    if len(trio_array) and (
        np.any(~eligible_child[trio_array[:, 0]])
        or np.any(~eligible_edge[trio_array[:, 0], trio_array[:, 1]])
        or np.any(~eligible_edge[trio_array[:, 0], trio_array[:, 2]])
    ):
        raise ValueError("trio row violates child/parent eligibility")
    trio_array = np.ascontiguousarray(trio_array)

    null_selector = pedigree_sources._probability_matrix(
        null_selector_switch_probability,
        children,
        projected.n_bins - 1,
        "null selector switch",
    )
    null_diagonal, null_off = _null_transition_coefficients(
        projected, transition, null_selector
    )
    coefficient = pedigree_sources._child_likelihood_coefficients(
        children_gl, float(mismatch_probability)
    )
    observed_site, observed_start, observed_stop = pedigree_sources._compact_observed_sites(
        child_seen, projected.bin_start, projected.bin_stop
    )

    # Keep temporary emission storage bounded even for large founder panels.
    # If a single child's cache cannot fit, use the same quadratic scorer
    # without caching, rather than exceeding the workspace allowance.
    per_child_bytes = projected.n_bins * projected.n_states**2 * 8
    cache_children = _COMMON_EMISSION_CACHE_BYTES // max(1, per_child_bytes)
    child_batch_size = min(children, cache_children) if cache_children else children
    common_args = (
        projected.state_alt_probability, children_gl, coefficient,
        observed_site, observed_start, observed_stop, float(mismatch_probability),
    )

    args = (
        projected.transmitted_state_probability,
        projected.transmitted_alt_probability,
        projected.bridge_stay_probability,
        projected.bridge_offdiag_left_scale,
        projected.bridge_offdiag_right_scale,
        projected.bridge_same_state_joint_mass,
        projected.bridge_row_arm,
        projected.bridge_column_arm,
        projected.bridge_operator_pivot,
    )
    if reuse_scores is None:
        m0_started = time.perf_counter()
        zero = _score_m0_kernel(
            *args,
            projected.state_alt_probability,
            children_gl,
            coefficient,
            observed_site,
            observed_start,
            observed_stop,
            exponent,
            null_diagonal,
            null_off,
            float(mismatch_probability),
            eligible_child,
        )
        m0_seconds = time.perf_counter() - m0_started
        m1_started = time.perf_counter()
        one = np.empty((children, projected.n_candidates), dtype=np.float64)
        for child_start in range(0, children, child_batch_size):
            child_stop = min(children, child_start + child_batch_size)
            common_products = (
                _common_emission_products(*common_args, child_start, child_stop)
                if cache_children else None
            )
            one[child_start:child_stop] = _score_m1_kernel(
                *args,
                projected.available,
                projected.state_alt_probability,
                children_gl,
                coefficient,
                observed_site,
                observed_start,
                observed_stop,
                exponent,
                null_diagonal,
                null_off,
                float(mismatch_probability),
                eligible_edge,
                zero, common_products, child_start, child_stop,
                projected.bridge_branch,
            )
            del common_products
        m1_seconds = time.perf_counter() - m1_started
    else:
        zero = np.asarray(reuse_scores.zero_observed, dtype=np.float64)
        one = np.asarray(reuse_scores.one_observed, dtype=np.float64)
        if zero.shape != (children,) or one.shape != (
            children, projected.n_candidates
        ):
            raise ValueError("reuse_scores M0/M1 arrays have incompatible shape")
        if (
            np.any(np.isnan(zero))
            or np.any(np.isposinf(zero))
            or np.any(np.isnan(one))
            or np.any(np.isposinf(one))
        ):
            raise ValueError("reuse_scores M0/M1 arrays contain invalid values")
        if (
            not np.array_equal(
                reuse_scores.candidate_source_available, projected.available
            )
            or not np.array_equal(
                reuse_scores.candidate_source_informative_site_count,
                projected.informative_site_count,
            )
        ):
            raise ValueError("reuse_scores candidate metadata are incompatible")
        zero = zero.copy()
        one = one.copy()
        m0_seconds = 0.0
        m1_seconds = 0.0
    m2_started = time.perf_counter()
    available1 = (
        projected.available[trio_array[:, 1]]
        if len(trio_array)
        else np.empty(0, dtype=np.bool_)
    )
    available2 = (
        projected.available[trio_array[:, 2]]
        if len(trio_array)
        else np.empty(0, dtype=np.bool_)
    )
    active = available1 & available2
    two = np.empty(len(trio_array), dtype=np.float64)
    for row in np.flatnonzero(~active):
        child, first, second = trio_array[row]
        if available1[row]:
            two[row] = one[child, first]
        elif available2[row]:
            two[row] = one[child, second]
        else:
            two[row] = zero[child]
    active_rows = np.flatnonzero(active)
    if len(active_rows):
        active_children = trio_array[active_rows, 0]
        for child_start in range(0, children, child_batch_size):
            child_stop = min(children, child_start + child_batch_size)
            batch_rows = active_rows[
                (active_children >= child_start) & (active_children < child_stop)
            ]
            if not len(batch_rows):
                continue
            common_products = (
                _common_emission_products(*common_args, child_start, child_stop)
                if cache_children else None
            )
            two[batch_rows] = _score_m2_kernel(
                np.ascontiguousarray(trio_array[batch_rows]),
                *args,
                projected.state_alt_probability,
                children_gl,
                coefficient,
                observed_site,
                observed_start,
                observed_stop,
                exponent,
                null_diagonal,
                null_off,
                float(mismatch_probability),
                common_products, child_start, projected.bridge_branch,
            )

            del common_products

    m2_seconds = time.perf_counter() - m2_started
    informative = child_seen & (
        np.ptp(children_gl, axis=2) > float(uniform_tolerance)
    )
    states = projected.n_states
    workspace_values = 2 * states * states + 2 * (states + 1)
    complexity = QuadraticComplexityDiagnostic(
        n_states=states,
        projected_hidden_state_count=states * states,
        exact_m2_hidden_state_count=states**4,
        float64_workspace_values_per_task=workspace_values,
        peak_working_bytes_per_task=workspace_values * np.dtype(np.float64).itemsize,
        time_complexity_per_task="O((observed_sites + bins) * S^2)",
        memory_complexity_per_task="O(S^2)",
    )
    return ProjectedRaggedQuadraticScores(
        zero_observed=zero,
        one_observed=one,
        two_observed=two,
        candidate_source_available=projected.available.copy(),
        candidate_source_informative_site_count=(
            projected.informative_site_count.copy()
        ),
        child_informative_site_count=np.sum(informative, axis=1, dtype=np.int64),
        m2_active_trio_count=int(np.sum(active)),
        m2_reduced_trio_count=int(len(trio_array) - np.sum(active)),
        approximation_name=APPROXIMATION_NAME,
        complexity=complexity,
        projection_retained_bytes=projected.retained_bytes,
        maximum_bridge_marginal_residual=(
            projected.maximum_bridge_marginal_residual
        ),
        projection_seconds=projected.preparation_seconds,
        scoring_seconds=time.perf_counter() - started,
        m0_scoring_seconds=m0_seconds,
        m1_scoring_seconds=m1_seconds,
        m2_scoring_seconds=m2_seconds,
        reused_lower_order_scores=reuse_scores is not None,
    )

import haplotype_reconstruction.pedigree.sources as pedigree_sources
