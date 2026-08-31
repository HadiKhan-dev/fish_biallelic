"""Fast deterministic one-dimensional Gaussian-mixture model selection.

This module specializes the small Gaussian-mixture fits used by pedigree
ancestry-depth inference.  The statistical model and defaults match the
existing scikit-learn ``GaussianMixture`` path: k-means initialization, full
component covariances, ten initializations, EM tolerance ``1e-3``, covariance
regularization ``1e-3``, and BIC selection.  Scalar arithmetic and reuse of
duplicate k-means partitions avoid estimator-framework overhead.  Compiled
scalar Lloyd and EM loops remove small-array allocation overhead when the fit
is repeated for every bootstrap replicate.

The input is expected to have already been standardized by the caller.  The
returned components are ordered by increasing mean because ancestry-depth
components have that ordinal interpretation downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit


_DEFAULT_N_INIT = 10
_DEFAULT_MAX_ITER = 500
_DEFAULT_REG_COVAR = 1e-3
_DEFAULT_TOLERANCE = 1e-3
_KMEANS_MAX_ITER = 300
_KMEANS_TOLERANCE = 1e-4
_LOG_2_PI = math.log(2.0 * math.pi)
_EPSILON_COUNT = 10.0 * np.finfo(np.float64).eps


@dataclass(frozen=True)
class GaussianMixture1DSelection:
    """BIC-selected 1-D Gaussian mixture and candidate-fit diagnostics.

    ``selected_component_count == 0`` denotes the same failure state as an
    all-nonconverged scikit-learn candidate set.  In that state the parameter
    arrays are empty and ``selected_bic`` is positive infinity.
    """

    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    selected_component_count: int
    selected_bic: float
    tested_bics: tuple[float, ...]
    converged: bool
    n_iter: int


@dataclass(frozen=True)
class _FixedComponentFit:
    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    lower_bound: float
    converged: bool
    n_iter: int


def _kmeans_assignment_cost_1d(
    values: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Return sklearn Lloyd's center-dependent assignment cost.

    sklearn omits the common ``values**2`` term from squared distances before
    taking the argmin.  Keeping the same evaluation matters for exactly tied
    partitions of discrete junction burdens.
    """
    return (
        centers[None, :] * centers[None, :]
        - 2.0 * values[:, None] * centers[None, :]
    )


def _kmeans_plusplus_centers_1d(
    values: np.ndarray,
    component_count: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Match sklearn's greedy k-means++ seeding for unit sample weights."""
    sample_count = len(values)
    centers = np.empty(component_count, dtype=np.float64)
    probabilities = np.full(sample_count, 1.0 / sample_count, dtype=np.float64)
    first = int(random_state.choice(sample_count, p=probabilities))
    centers[0] = values[first]

    closest_squared_distance = -2.0 * centers[0] * values
    closest_squared_distance += centers[0] * centers[0]
    closest_squared_distance += values * values
    np.maximum(closest_squared_distance, 0.0, out=closest_squared_distance)
    current_potential = float(np.sum(closest_squared_distance))
    local_trials = 2 + int(math.log(component_count))
    for component in range(1, component_count):
        trial_values = random_state.uniform(size=local_trials) * current_potential
        cumulative = np.cumsum(closest_squared_distance, dtype=np.float64)
        candidate_indices = np.searchsorted(cumulative, trial_values)
        np.clip(
            candidate_indices,
            None,
            sample_count - 1,
            out=candidate_indices,
        )
        candidates = values[candidate_indices]
        candidate_squared_distance = -2.0 * candidates[:, None] * values[None, :]
        candidate_squared_distance += candidates[:, None] * candidates[:, None]
        candidate_squared_distance += values[None, :] * values[None, :]
        np.maximum(
            candidate_squared_distance, 0.0, out=candidate_squared_distance
        )
        np.minimum(
            closest_squared_distance[None, :],
            candidate_squared_distance,
            out=candidate_squared_distance,
        )
        candidate_potentials = np.sum(candidate_squared_distance, axis=1)
        best_trial = int(np.argmin(candidate_potentials))
        best_candidate = int(candidate_indices[best_trial])
        centers[component] = values[best_candidate]
        closest_squared_distance = candidate_squared_distance[best_trial]
        current_potential = float(candidate_potentials[best_trial])
    return centers


def _relocate_empty_kmeans_clusters_1d(
    values: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Apply sklearn's farthest-point relocation for empty Lloyd clusters."""
    empty = np.flatnonzero(counts == 0.0)
    if not len(empty):
        return
    assigned_squared_distance = (values - centers[labels]) ** 2
    # ``argsort(...)[::-1]`` is deterministic; exact tie order is immaterial
    # for the supported path because k does not exceed the distinct values.
    farthest = np.argsort(assigned_squared_distance, kind="stable")[::-1]
    for empty_component, sample in zip(empty, farthest):
        source_component = int(labels[sample])
        sums[source_component] -= values[sample]
        counts[source_component] -= 1.0
        sums[empty_component] = values[sample]
        counts[empty_component] = 1.0


def _kmeans_labels_1d(
    values: np.ndarray,
    component_count: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Return labels from sklearn-compatible centered 1-D Lloyd k-means."""
    centered = values - float(np.mean(values))
    centers = _kmeans_plusplus_centers_1d(
        centered, component_count, random_state
    )
    tolerance = float(np.var(centered)) * _KMEANS_TOLERANCE
    return _lloyd_labels_1d(
        centered,
        centers,
        tolerance,
        _KMEANS_MAX_ITER,
    )


@njit(cache=True, fastmath=False, nogil=True)
def _lloyd_labels_1d(
    values: np.ndarray,
    initial_centers: np.ndarray,
    tolerance: float,
    max_iter: int,
) -> np.ndarray:
    """Compiled scalar equivalent of sklearn's one-dimensional Lloyd loop."""
    sample_count = len(values)
    component_count = len(initial_centers)
    centers = initial_centers.copy()
    new_centers = np.empty(component_count, dtype=np.float64)
    labels = np.full(sample_count, -1, dtype=np.int32)
    old_labels = labels.copy()
    counts = np.empty(component_count, dtype=np.float64)
    sums = np.empty(component_count, dtype=np.float64)
    relocated = np.empty(sample_count, dtype=np.bool_)
    strict_convergence = False

    for _ in range(max_iter):
        counts[:] = 0.0
        sums[:] = 0.0
        for sample in range(sample_count):
            value = values[sample]
            best_component = 0
            best_cost = (
                centers[0] * centers[0] - 2.0 * value * centers[0]
            )
            for component in range(1, component_count):
                cost = (
                    centers[component] * centers[component]
                    - 2.0 * value * centers[component]
                )
                if cost < best_cost:
                    best_component = component
                    best_cost = cost
            labels[sample] = best_component
            counts[best_component] += 1.0
            sums[best_component] += value

        # This is normally unreachable because ancestry-depth fitting limits k
        # to the distinct observations.  Retain sklearn's deterministic
        # farthest-point relocation for the general helper contract.
        relocated[:] = False
        for empty_component in range(component_count):
            if counts[empty_component] != 0.0:
                continue
            farthest_sample = -1
            farthest_distance = -1.0
            for sample in range(sample_count):
                if relocated[sample]:
                    continue
                source_component = labels[sample]
                difference = values[sample] - centers[source_component]
                distance = difference * difference
                # Reversing a stable ascending sort selects the highest sample
                # index first when distances tie.
                if distance >= farthest_distance:
                    farthest_sample = sample
                    farthest_distance = distance
            relocated[farthest_sample] = True
            source_component = labels[farthest_sample]
            sums[source_component] -= values[farthest_sample]
            counts[source_component] -= 1.0
            sums[empty_component] = values[farthest_sample]
            counts[empty_component] = 1.0

        center_shift = 0.0
        for component in range(component_count):
            new_centers[component] = sums[component] / counts[component]
            difference = new_centers[component] - centers[component]
            center_shift += difference * difference
        centers[:] = new_centers

        same_labels = True
        for sample in range(sample_count):
            if labels[sample] != old_labels[sample]:
                same_labels = False
                break
        if same_labels:
            strict_convergence = True
            break
        if center_shift <= tolerance:
            break
        old_labels[:] = labels

    if not strict_convergence:
        for sample in range(sample_count):
            value = values[sample]
            best_component = 0
            best_cost = (
                centers[0] * centers[0] - 2.0 * value * centers[0]
            )
            for component in range(1, component_count):
                cost = (
                    centers[component] * centers[component]
                    - 2.0 * value * centers[component]
                )
                if cost < best_cost:
                    best_component = component
                    best_cost = cost
            labels[sample] = best_component
    return labels


def _initial_parameters(
    values: np.ndarray,
    labels: np.ndarray,
    component_count: int,
    reg_covar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    responsibilities = np.equal(
        labels[:, None], np.arange(component_count)[None, :]
    ).astype(np.float64)
    weights, means, variances = _estimate_parameters(
        values, responsibilities, reg_covar
    )
    # sklearn divides the initialized effective counts by n, while subsequent
    # M steps normalize their sum.  Retain that tiny distinction exactly.
    weights /= float(len(values))
    return weights, means, variances


def _estimate_parameters(
    values: np.ndarray,
    responsibilities: np.ndarray,
    reg_covar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epsilon_count = 10.0 * np.finfo(responsibilities.dtype).eps
    effective_counts = np.sum(responsibilities, axis=0) + epsilon_count
    means = (responsibilities.T @ values) / effective_counts
    differences = values[:, None] - means[None, :]
    variances = (
        np.sum(responsibilities * differences * differences, axis=0)
        / effective_counts
        + reg_covar
    )
    return effective_counts, means, variances


def _e_step(
    values: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    precision_cholesky = 1.0 / np.sqrt(variances)
    transformed = (
        values[:, None] * precision_cholesky[None, :]
        - means[None, :] * precision_cholesky[None, :]
    )
    weighted_log_probability = (
        -0.5 * (_LOG_2_PI + transformed * transformed)
        + np.log(precision_cholesky)[None, :]
        + np.log(weights)[None, :]
    )
    row_maximum = np.max(weighted_log_probability, axis=1)
    log_probability = row_maximum + np.log(
        np.sum(
            np.exp(weighted_log_probability - row_maximum[:, None]),
            axis=1,
        )
    )
    log_responsibilities = (
        weighted_log_probability - log_probability[:, None]
    )
    return (
        float(np.mean(log_probability)),
        log_responsibilities,
        log_probability,
    )


def _fit_fixed_components(
    values: np.ndarray,
    component_count: int,
    seed: int,
    n_init: int,
    max_iter: int,
    reg_covar: float,
    tolerance: float,
) -> _FixedComponentFit:
    random_state = np.random.RandomState(int(seed))
    best_fit: _FixedComponentFit | None = None
    maximum_lower_bound = -np.inf
    fitted_partitions: set[tuple[int, ...]] = set()

    for _ in range(n_init):
        labels = _kmeans_labels_1d(values, component_count, random_state)
        # Component permutations initialize the same mixture.  Canonicalize
        # by mean to avoid repeating deterministic EM for duplicate k-means
        # partitions without changing the best initialization.
        label_means = np.asarray(
            [np.mean(values[labels == k]) for k in range(component_count)]
        )
        component_order = np.argsort(label_means, kind="stable")
        inverse_order = np.empty(component_count, dtype=np.int64)
        inverse_order[component_order] = np.arange(component_count)
        canonical_labels = inverse_order[labels]
        partition = tuple(int(value) for value in canonical_labels)
        if partition in fitted_partitions:
            continue
        fitted_partitions.add(partition)

        weights, means, variances = _initial_parameters(
            values, canonical_labels, component_count, reg_covar
        )
        (
            weights,
            means,
            variances,
            lower_bound,
            converged,
            iteration,
        ) = _em_from_parameters_1d(
            values,
            weights,
            means,
            variances,
            max_iter,
            reg_covar,
            tolerance,
        )

        if lower_bound > maximum_lower_bound or best_fit is None:
            maximum_lower_bound = lower_bound
            best_fit = _FixedComponentFit(
                means=means.copy(),
                variances=variances.copy(),
                weights=weights.copy(),
                lower_bound=float(lower_bound),
                converged=converged,
                n_iter=int(iteration),
            )

    assert best_fit is not None
    return best_fit


@njit(cache=True, fastmath=False, nogil=True)
def _em_from_parameters_1d(
    values: np.ndarray,
    initial_weights: np.ndarray,
    initial_means: np.ndarray,
    initial_variances: np.ndarray,
    max_iter: int,
    reg_covar: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, int]:
    """Run sklearn-compatible 1-D full-covariance EM from fixed parameters."""
    sample_count = len(values)
    component_count = len(initial_weights)
    weights = initial_weights.copy()
    means = initial_means.copy()
    variances = initial_variances.copy()
    responsibilities = np.empty(
        (sample_count, component_count), dtype=np.float64
    )
    precision_cholesky = np.empty(component_count, dtype=np.float64)
    effective_counts = np.empty(component_count, dtype=np.float64)
    new_means = np.empty(component_count, dtype=np.float64)
    new_variances = np.empty(component_count, dtype=np.float64)
    lower_bound = -np.inf
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        previous_lower_bound = lower_bound
        for component in range(component_count):
            precision_cholesky[component] = 1.0 / math.sqrt(
                variances[component]
            )

        lower_bound_sum = 0.0
        for sample in range(sample_count):
            value = values[sample]
            row_maximum = -np.inf
            for component in range(component_count):
                precision = precision_cholesky[component]
                transformed = (
                    value * precision - means[component] * precision
                )
                score = -0.5 * (
                    _LOG_2_PI + transformed * transformed
                )
                score += math.log(precision)
                score += math.log(weights[component])
                responsibilities[sample, component] = score
                if score > row_maximum:
                    row_maximum = score

            exponential_sum = 0.0
            for component in range(component_count):
                exponential_sum += math.exp(
                    responsibilities[sample, component] - row_maximum
                )
            log_probability = row_maximum + math.log(exponential_sum)
            lower_bound_sum += log_probability
            for component in range(component_count):
                responsibilities[sample, component] = math.exp(
                    responsibilities[sample, component] - log_probability
                )
        lower_bound = lower_bound_sum / sample_count

        effective_counts[:] = _EPSILON_COUNT
        new_means[:] = 0.0
        for sample in range(sample_count):
            value = values[sample]
            for component in range(component_count):
                responsibility = responsibilities[sample, component]
                effective_counts[component] += responsibility
                new_means[component] += responsibility * value
        for component in range(component_count):
            new_means[component] /= effective_counts[component]

        new_variances[:] = 0.0
        for sample in range(sample_count):
            value = values[sample]
            for component in range(component_count):
                difference = value - new_means[component]
                new_variances[component] += (
                    responsibilities[sample, component]
                    * difference
                    * difference
                )
        weight_sum = 0.0
        for component in range(component_count):
            new_variances[component] = (
                new_variances[component] / effective_counts[component]
                + reg_covar
            )
            weight_sum += effective_counts[component]
        for component in range(component_count):
            weights[component] = effective_counts[component] / weight_sum
            means[component] = new_means[component]
            variances[component] = new_variances[component]

        if abs(lower_bound - previous_lower_bound) < tolerance:
            converged = True
            break

    return (
        weights,
        means,
        variances,
        lower_bound,
        converged,
        iteration,
    )


def _bic(values: np.ndarray, fit: _FixedComponentFit) -> float:
    _, _, log_probability = _e_step(
        values, fit.weights, fit.means, fit.variances
    )
    component_count = len(fit.means)
    parameter_count = 3 * component_count - 1
    return float(
        -2.0 * np.sum(log_probability)
        + parameter_count * math.log(len(values))
    )


def fit_bic_selected_gaussian_mixture_1d(
    values: np.ndarray,
    maximum_components: int,
    seed: int,
    *,
    n_init: int = _DEFAULT_N_INIT,
    max_iter: int = _DEFAULT_MAX_ITER,
    reg_covar: float = _DEFAULT_REG_COVAR,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> GaussianMixture1DSelection:
    """Fit and BIC-select deterministic Gaussian mixtures with 1..k components.

    Parameters mirror the ancestry-depth sklearn reference path.  Candidate
    component counts are always tested in increasing order.  The fixed seed is
    restarted for every candidate count, as occurs when constructing one
    ``GaussianMixture(random_state=seed)`` estimator per count.
    """
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or len(observations) < 2:
        raise ValueError("values must be a one-dimensional vector of length >= 2")
    if np.any(~np.isfinite(observations)):
        raise ValueError("values must be finite")
    if not 1 <= int(maximum_components) <= len(observations):
        raise ValueError("maximum_components must be between 1 and sample count")
    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive")
    if reg_covar < 0.0 or tolerance < 0.0:
        raise ValueError("reg_covar and tolerance must be non-negative")

    fits: list[_FixedComponentFit | None] = []
    bics: list[float] = []
    for component_count in range(1, int(maximum_components) + 1):
        fit = _fit_fixed_components(
            observations,
            component_count,
            int(seed),
            int(n_init),
            int(max_iter),
            float(reg_covar),
            float(tolerance),
        )
        if not fit.converged:
            fits.append(None)
            bics.append(np.inf)
        else:
            fits.append(fit)
            bics.append(_bic(observations, fit))

    finite = np.flatnonzero(np.isfinite(bics))
    if not len(finite):
        return GaussianMixture1DSelection(
            means=np.empty(0, dtype=np.float64),
            variances=np.empty(0, dtype=np.float64),
            weights=np.empty(0, dtype=np.float64),
            selected_component_count=0,
            selected_bic=np.inf,
            tested_bics=tuple(float(value) for value in bics),
            converged=False,
            n_iter=0,
        )

    selected_index = int(finite[np.argmin(np.asarray(bics)[finite])])
    selected_fit = fits[selected_index]
    assert selected_fit is not None
    order = np.argsort(selected_fit.means, kind="stable")
    return GaussianMixture1DSelection(
        means=selected_fit.means[order],
        variances=selected_fit.variances[order],
        weights=selected_fit.weights[order],
        selected_component_count=selected_index + 1,
        selected_bic=float(bics[selected_index]),
        tested_bics=tuple(float(value) for value in bics),
        converged=True,
        n_iter=selected_fit.n_iter,
    )
