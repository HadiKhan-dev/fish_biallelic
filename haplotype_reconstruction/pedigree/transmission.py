"""Prepare projected ragged parent sources and score M0/M1/M2 evidence.

Numerical projection and scoring kernels live in the two transmission modules;
this module owns their input validation, public result types and orchestration.
"""
from __future__ import annotations


from dataclasses import dataclass
import time
import numpy as np
import haplotype_reconstruction.core.parallel as core_parallel

core_parallel.ensure_numba_registry_warmup()


from.transmission_projection import (
    _symmetrise_source_marginals,
    _project_transmitted_alt,
    _positive_exclusion_sums,
    _oriented_posterior_and_diagonal_flow,
    _small_maxent_diagonal_product,
    _maxent_branch_function,
    _bisect_maxent_root,
    _maximum_entropy_bridge_kernel,
)

from.transmission_scoring import (
    _site_likelihood,
    _common_emission_products,
    _power_common_products,
    _normalise_forward,
    _apply_probability_emission,
    _apply_log_emission,
    _apply_star_bridge_axis,
    _apply_bridge_axis,
    _apply_null_axis,
    _score_path,
    _score_m0_kernel,
    _score_m1_kernel,
    _score_m2_kernel,
)


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


def prepare_projected_ragged_quadratic(
    factors: pedigree_sources.RaggedSourceBatchFactors,
    model,
    candidate_genotype_likelihoods: np.ndarray,
    candidate_observed: np.ndarray,
    *,
    selected_site_indices: np.ndarray | None=None,
    precomputed_source_marginals: np.ndarray | None=None,
    candidate_selector_switch_probability=0.01,
    sinkhorn_tolerance: float=2e-13,
    sinkhorn_max_iterations: int=128,
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
        (1.0 - selector) * (same[None,:] + (states - 1) * one[None,:])
        + selector / states
    )
    off = (
        (1.0 - selector) * (one[None,:] + (states - 1) * two[None,:])
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
    eligible_children: np.ndarray | None=None,
    eligible_parent_edges: np.ndarray | None=None,
    null_selector_switch_probability=0.01,
    mismatch_probability: float=0.01,
    reuse_scores: ProjectedRaggedQuadraticScores | None=None,
    uniform_tolerance: float=1e-12,
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
    per_child_bytes = projected.n_bins * projected.n_states ** 2 * 16
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
            common_powered = (_power_common_products(common_products, exponent, child_start)
                              if common_products is not None else None)
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
                projected.bridge_branch, common_powered,
            )
            del common_products, common_powered
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
            common_powered = (_power_common_products(common_products, exponent, child_start)
                              if common_products is not None else None)
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
                common_products, child_start, projected.bridge_branch, common_powered,
            )

            del common_products, common_powered

    m2_seconds = time.perf_counter() - m2_started
    informative = child_seen & (
        np.ptp(children_gl, axis=2) > float(uniform_tolerance)
    )
    states = projected.n_states
    workspace_values = 2 * states * states + 2 * (states + 1)
    complexity = QuadraticComplexityDiagnostic(
        n_states=states,
        projected_hidden_state_count=states * states,
        exact_m2_hidden_state_count=states ** 4,
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
