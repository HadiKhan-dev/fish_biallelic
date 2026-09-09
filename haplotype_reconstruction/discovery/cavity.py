"""discovery / cavity for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence
import numpy as np
from numba import njit, prange


@dataclass(frozen=True)
class CavitySelectionConfig:
    """Numerical and state-prior settings for cavity K comparison."""

    state_concentration: float = 0.5
    wildcard_prior_mass: float = 0.02
    double_wildcard_fraction: float = 0.25
    mean_field_max_iter: int = 100
    mean_field_tolerance: float = 1e-10
    likelihood_floor: float = 1e-300
    max_modes_per_k: int | None = None
    founder_inference: str = "mean_field"
    apply_unordered_haplotype_set_code: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.state_concentration) \
                or self.state_concentration <= 0.0:
            raise ValueError("state_concentration must be finite and positive")
        if not 0.0 < self.wildcard_prior_mass < 1.0:
            raise ValueError("wildcard_prior_mass must lie in (0, 1)")
        if not 0.0 < self.double_wildcard_fraction < 1.0:
            raise ValueError("double_wildcard_fraction must lie in (0, 1)")
        if isinstance(self.mean_field_max_iter, bool) \
                or int(self.mean_field_max_iter) < 1:
            raise ValueError("mean_field_max_iter must be a positive integer")
        if not math.isfinite(self.mean_field_tolerance) \
                or self.mean_field_tolerance <= 0.0:
            raise ValueError(
                "mean_field_tolerance must be finite and positive"
            )
        if not math.isfinite(self.likelihood_floor) \
                or not 0.0 < self.likelihood_floor < 1.0:
            raise ValueError("likelihood_floor must lie strictly in (0, 1)")
        if self.founder_inference not in {
            "mean_field", "anchored_pseudolikelihood"
        }:
            raise ValueError(
                "founder_inference must be mean_field or "
                "anchored_pseudolikelihood"
            )
        if not isinstance(self.apply_unordered_haplotype_set_code, bool):
            raise ValueError(
                "apply_unordered_haplotype_set_code must be boolean"
            )
        if self.max_modes_per_k is not None and (
            isinstance(self.max_modes_per_k, bool)
            or int(self.max_modes_per_k) < 1
        ):
            raise ValueError(
                "max_modes_per_k must be None or a positive integer"
            )


@dataclass(frozen=True)
class HybridCavitySelectionConfig:
    """Settings for anchored screening followed by mean-field refinement.

    The hybrid workflow deliberately fixes both the one-mode-per-K cap and
    the unordered haplotype-set regularizer.  Only numerical cavity settings
    are exposed here so callers cannot accidentally turn the documented
    two-stage objective into a different selection procedure.
    """

    state_concentration: float = 0.5
    wildcard_prior_mass: float = 0.02
    double_wildcard_fraction: float = 0.25
    mean_field_max_iter: int = 100
    mean_field_tolerance: float = 1e-10
    likelihood_floor: float = 1e-300

    def __post_init__(self) -> None:
        # Reuse the single-stage validation rather than maintaining a second
        # subtly different definition of the same numerical domain.
        self._stage_config("mean_field")

    def _stage_config(self, founder_inference: str) -> CavitySelectionConfig:
        return CavitySelectionConfig(
            state_concentration=self.state_concentration,
            wildcard_prior_mass=self.wildcard_prior_mass,
            double_wildcard_fraction=self.double_wildcard_fraction,
            mean_field_max_iter=self.mean_field_max_iter,
            mean_field_tolerance=self.mean_field_tolerance,
            likelihood_floor=self.likelihood_floor,
            max_modes_per_k=1,
            founder_inference=founder_inference,
            apply_unordered_haplotype_set_code=True,
        )


@dataclass(frozen=True)
class CavityKDiagnostic:
    """Auditable score and numerical diagnostics for one represented K."""

    k: int
    cavity_log_predictive: float
    mean_sample_log_predictive: float
    log_k_prior: float
    log_haplotype_set_prior: float
    log_score: float
    log_weight: float
    pseudo_probability: float
    best_total_nll: float
    best_mode_digest: str
    n_input_modes: int
    n_unique_modes: int
    n_duplicate_modes_removed: int
    n_mean_field_site_fits: int
    n_mean_field_not_converged: int
    mean_mean_field_iterations: float
    n_zero_support_founder_cavities: int
    mean_founder_allele_entropy_nats: float
    n_alternate_initialization_wins: int
    mean_initialization_elbo_spread: float
    uniform_mode_log_mean_exp: float
    n_modes_scored: int
    n_modes_omitted_by_cap: int

    @property
    def probability(self) -> float:
        """Compatibility alias for normalized pseudo-probability."""

        return self.pseudo_probability


@dataclass(frozen=True)
class CavityModeDiagnostic:
    """Cavity score and numerical behavior of one canonical unique mode."""

    k: int
    mode_digest: str
    total_nll: float
    cavity_log_predictive: float
    mean_sample_log_predictive: float
    n_mean_field_site_fits: int
    n_mean_field_not_converged: int
    mean_mean_field_iterations: float
    n_zero_support_founder_cavities: int
    mean_founder_allele_entropy_nats: float
    n_alternate_initialization_wins: int
    mean_initialization_elbo_spread: float
    selected_within_k: bool


@dataclass(frozen=True)
class HybridCavityDiagnostic:
    """Complete audit trail for the two-stage regularized selector."""

    anchored_k_diagnostics: tuple[CavityKDiagnostic, ...]
    anchored_mode_diagnostics: tuple[CavityModeDiagnostic, ...]
    anchored_ranked_k: tuple[int, ...]
    shortlist_rule: str
    shortlisted_k: tuple[int, ...]
    refined_k_diagnostics: tuple[CavityKDiagnostic, ...]
    refined_mode_diagnostics: tuple[CavityModeDiagnostic, ...]
    refined_ranked_k: tuple[int, ...]
    final_winner_k: int
    final_runner_up_k: int | None
    selected_mode_digest: str
    boundary_caveat: str
    calibration_caveat: str
    full_data_selection_leakage_caveat: str

    @property
    def anchored_mode_digest_by_k(self) -> Mapping[int, str]:
        return {
            diagnostic.k: diagnostic.best_mode_digest
            for diagnostic in self.anchored_k_diagnostics
        }

    @property
    def refined_mode_digest_by_k(self) -> Mapping[int, str]:
        return {
            diagnostic.k: diagnostic.best_mode_digest
            for diagnostic in self.refined_k_diagnostics
        }


@dataclass(frozen=True)
class CavitySelection:
    """Result of fixed-A leave-one-sample cavity comparison across K."""

    method: str
    map_k: int
    runner_up_k: int | None
    selected_mode_digest: str
    k_diagnostics: tuple[CavityKDiagnostic, ...]
    mode_diagnostics: tuple[CavityModeDiagnostic, ...]
    support_selected_from_full_data: bool
    assignments_selected_from_full_data: bool
    selection_leakage: bool
    weights_are_calibrated: bool
    k_prior: str
    boundary_limited: bool
    founder_inference: str
    apply_unordered_haplotype_set_code: bool
    mode_cap_per_k: int | None
    mode_cap_applied: bool
    all_mean_field_converged: bool
    interpretation: str
    n_samples: int
    n_sites: int
    hybrid_diagnostic: HybridCavityDiagnostic | None = None

    @property
    def probability_by_k(self) -> Mapping[int, float]:
        return {
            diagnostic.k: diagnostic.pseudo_probability
            for diagnostic in self.k_diagnostics
        }

    @property
    def log_score_by_k(self) -> Mapping[int, float]:
        return {
            diagnostic.k: diagnostic.log_score
            for diagnostic in self.k_diagnostics
        }

    @property
    def log_weight_by_k(self) -> Mapping[int, float]:
        return {
            diagnostic.k: diagnostic.log_weight
            for diagnostic in self.k_diagnostics
        }


@dataclass(frozen=True)
class _PreparedMode:
    k: int
    haplotypes: np.ndarray
    assignments: np.ndarray
    total_nll: float
    digest: str
    raw_mode: Any


class _CavityScoringWorkspace:
    """Evidence-local inputs shared by repeated cavity-score batches.

    Reversible search scores a succession of newly discovered modes against
    one immutable likelihood tensor and one immutable configuration.  The
    normalized log likelihoods and K-specific state geometry do not depend on
    the mode, so preparing them once avoids repeating validation, flooring and
    triangular-state construction at every search refresh.
    """

    __slots__ = (
        "evidence_reference",
        "likelihood",
        "config",
        "log_likelihood",
        "rw_log_likelihood",
        "ww_log_emission",
        "state_geometry_by_k",
    )

    def __init__(self, evidence: np.ndarray, config: CavitySelectionConfig):
        self.evidence_reference = evidence
        self.likelihood = _validate_evidence(evidence)
        self.config = config
        self.log_likelihood = _floor_log_likelihood_kernel(
            self.likelihood, float(config.likelihood_floor)
        )
        (
            self.rw_log_likelihood,
            self.ww_log_emission,
        ) = _build_evidence_emission_cache_kernel(
            self.likelihood, float(config.likelihood_floor)
        )
        self.state_geometry_by_k: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    def geometry(self, k: int):
        value = self.state_geometry_by_k.get(int(k))
        if value is None:
            value = _build_state_geometry(int(k), self.config)
            self.state_geometry_by_k[int(k)] = value
        return value


def _prepare_cavity_scoring_workspace(
    evidence: np.ndarray,
    config: CavitySelectionConfig,
) -> _CavityScoringWorkspace:
    """Prepare one private workspace for repeated scores of one evidence."""

    return _CavityScoringWorkspace(evidence, config)


def _validate_evidence(evidence: np.ndarray) -> np.ndarray:
    return core_genotypes.validate_normalized_genotype_evidence(evidence)


def _normalized_log_weights(
    log_scores: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Normalize finite scores in log space."""

    values = tuple(float(value) for value in log_scores)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("log_scores must be a non-empty finite sequence")
    maximum = max(values)
    log_normalizer = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in values)
    )
    log_weights = tuple(value - log_normalizer for value in values)
    return log_weights, tuple(math.exp(value) for value in log_weights)


def _canonical_prepared_mode(
    mode: Any,
    k: int,
    n_samples: int,
    n_sites: int,
) -> _PreparedMode:
    if not hasattr(mode, "haplotypes") or not hasattr(mode, "assignments"):
        raise TypeError("cavity scoring requires rich modes with assignments")
    if not hasattr(mode, "total_nll"):
        raise TypeError("cavity scoring requires modes with total_nll")
    haplotypes = np.asarray(mode.haplotypes, dtype=np.int64)
    assignments = np.asarray(mode.assignments, dtype=np.int64)
    if haplotypes.shape != (k, n_sites):
        raise ValueError(
            f"a K={k} mode must have haplotypes shaped ({k}, {n_sites})"
        )
    if np.any((haplotypes != 0) & (haplotypes != 1)):
        raise ValueError("cavity scoring requires hard binary haplotypes")
    if assignments.shape != (n_samples, 2):
        raise ValueError(
            "mode assignments must have shape (evidence samples, 2)"
        )
    if np.any(assignments < 0) or np.any(assignments > k):
        raise ValueError("assignment index lies outside [0, K]")

    (
        canonical_haplotypes,
        canonical_assignments,
        _order,
        _inverse,
        _canonical_key,
    ) = discovery_objectives.canonicalize_binary_panel(haplotypes, assignments)
    if k > 1 and any(
        np.array_equal(
            canonical_haplotypes[index - 1],
            canonical_haplotypes[index],
        )
        for index in range(1, k)
    ):
        raise ValueError("a complete mode cannot contain duplicate rows")

    try:
        total_nll = float(mode.total_nll)
    except (TypeError, ValueError) as error:
        raise TypeError("mode total_nll must be numeric") from error
    if not math.isfinite(total_nll):
        raise ValueError("mode total_nll must be finite")
    canonical = discovery_modes._canonicalize_mode(mode, k)
    if not np.array_equal(canonical.haplotypes, canonical_haplotypes):
        raise AssertionError("canonical haplotype implementations disagree")
    canonical_haplotypes.setflags(write=False)
    canonical_assignments.setflags(write=False)
    return _PreparedMode(
        k=k,
        haplotypes=canonical_haplotypes,
        assignments=canonical_assignments,
        total_nll=total_nll,
        digest=canonical.digest,
        raw_mode=mode,
    )


def _prepare_modes(
    modes_by_k: Mapping[int, Sequence[Any]],
    n_samples: int,
    n_sites: int,
) -> tuple[
    dict[int, tuple[_PreparedMode, ...]],
    dict[int, tuple[int, int]],
]:
    if not isinstance(modes_by_k, Mapping) or not modes_by_k:
        raise TypeError(
            "modes_by_k must be a non-empty mapping from K to mode sequences"
        )
    prepared_by_k: dict[int, tuple[_PreparedMode, ...]] = {}
    counts: dict[int, tuple[int, int]] = {}
    for raw_k in sorted(modes_by_k):
        if isinstance(raw_k, bool) or int(raw_k) != raw_k or int(raw_k) < 1:
            raise ValueError("mode-map keys must be positive integer K values")
        k = int(raw_k)
        raw_modes = modes_by_k[raw_k]
        sequence = (
            (raw_modes,)
            if isinstance(raw_modes, np.ndarray)
            else tuple(raw_modes)
        )
        if not sequence:
            raise ValueError(f"no modes were supplied for K={k}")
        unique: dict[str, _PreparedMode] = {}
        for raw_mode in sequence:
            prepared = _canonical_prepared_mode(
                raw_mode, k, n_samples, n_sites
            )
            previous = unique.get(prepared.digest)
            if previous is None or prepared.total_nll < previous.total_nll:
                unique[prepared.digest] = prepared
        counts[k] = (len(sequence), len(unique))
        prepared_by_k[k] = tuple(
            unique[digest] for digest in sorted(unique)
        )
    return prepared_by_k, counts


def _build_state_geometry(
    k: int,
    config: CavitySelectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_i, pair_j = np.triu_indices(k)
    n_rr = len(pair_i)
    n_states = n_rr + k + 1
    kind = np.empty(n_states, dtype=np.int64)
    first = np.full(n_states, -1, dtype=np.int64)
    second = np.full(n_states, -1, dtype=np.int64)
    kind[:n_rr] = 0  # RR
    first[:n_rr] = pair_i
    second[:n_rr] = pair_j
    kind[n_rr:n_rr + k] = 1  # RW
    first[n_rr:n_rr + k] = np.arange(k)
    kind[-1] = 2  # WW

    diagonal = pair_i == pair_j
    base = np.empty(n_states, dtype=np.float64)
    omega = config.wildcard_prior_mass
    delta = config.double_wildcard_fraction
    base[:n_rr] = (
        (1.0 - omega)
        * np.where(diagonal, 1.0, 2.0)
        / float(k * k)
    )
    base[n_rr:n_rr + k] = omega * (1.0 - delta) / k
    base[-1] = omega * delta
    base /= np.sum(base)
    return tuple(
        np.ascontiguousarray(value)
        for value in (kind, first, second, base)
    )


@njit(cache=True)
def _stable_sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


@njit(cache=True)
def _logaddexp_values(values: np.ndarray) -> float:
    maximum = -math.inf
    for index in range(values.shape[0]):
        if values[index] > maximum:
            maximum = values[index]
    total = 0.0
    for index in range(values.shape[0]):
        total += math.exp(values[index] - maximum)
    return maximum + math.log(total)


@njit(cache=True)
def _floor_log_likelihood_kernel(
    likelihood: np.ndarray,
    likelihood_floor: float,
) -> np.ndarray:
    """Apply the historical floor/log loop once for all candidate modes."""

    n_samples, n_sites, _ = likelihood.shape
    result = np.empty_like(likelihood)
    for sample in range(n_samples):
        for site in range(n_sites):
            for genotype in range(3):
                value = likelihood[sample, site, genotype]
                if value < likelihood_floor:
                    value = likelihood_floor
                result[sample, site, genotype] = math.log(value)
    return result


@njit(cache=True)
def _build_evidence_emission_cache_kernel(
    likelihood: np.ndarray,
    likelihood_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cache evidence-only wildcard terms shared by every scored mode."""

    n_samples, n_sites, _ = likelihood.shape
    rw_log_likelihood = np.empty(
        (n_samples, n_sites, 2), dtype=np.float64
    )
    ww_log_emission = np.empty((n_samples, n_sites), dtype=np.float64)
    for sample in range(n_samples):
        for site in range(n_sites):
            g0 = likelihood[sample, site, 0]
            g1 = likelihood[sample, site, 1]
            g2 = likelihood[sample, site, 2]
            mix0 = 0.5 * (g0 + g1)
            mix1 = 0.5 * (g1 + g2)
            ww = 0.25 * g0 + 0.5 * g1 + 0.25 * g2
            if mix0 < likelihood_floor:
                mix0 = likelihood_floor
            if mix1 < likelihood_floor:
                mix1 = likelihood_floor
            if ww < likelihood_floor:
                ww = likelihood_floor
            rw_log_likelihood[sample, site, 0] = math.log(mix0)
            rw_log_likelihood[sample, site, 1] = math.log(mix1)
            ww_log_emission[sample, site] = math.log(ww)
    return rw_log_likelihood, ww_log_emission


@njit(cache=True, parallel=True)
def _score_mode_cavity_kernel(
    likelihood: np.ndarray,
    log_likelihood: np.ndarray,
    rw_log_likelihood: np.ndarray,
    ww_log_emission: np.ndarray,
    haplotypes: np.ndarray,
    assignments: np.ndarray,
    state_kind: np.ndarray,
    state_first: np.ndarray,
    state_second: np.ndarray,
    prior_base: np.ndarray,
    state_concentration: float,
    mean_field_max_iter: int,
    mean_field_tolerance: float,
    likelihood_floor: float,
) -> tuple[np.ndarray, int, int, int, float, int, float]:
    """Compiled fixed-A cavity score for one complete mode.

    Held-out samples are conditionally independent once the complete-data
    unary/pair potentials and assignment counts have been constructed. Keep
    each held-out calculation's scratch arrays and diagnostics private, then
    reduce diagnostics in sample-index order after the parallel region.
    """

    n_samples, n_sites, _ = likelihood.shape
    k = haplotypes.shape[0]
    n_states = state_kind.shape[0]
    wildcard = k
    log_half = -math.log(2.0)

    # Count support and active real-real assignment edges first.  Building the
    # potentials in a second sample-ordered pass retains the historical sum
    # order while avoiding a dense K x K pair tensor for unused edges.
    pair_counts = np.zeros((k, k), dtype=np.int64)
    founder_support = np.zeros(k, dtype=np.int64)
    state_counts = np.zeros(n_states, dtype=np.int64)

    # Dense lookup is tiny at the intended K<=16 and avoids repeatedly
    # deriving triangular-state offsets in the hot held-out loop.
    state_lookup = np.empty((k + 1, k + 1), dtype=np.int64)
    state_lookup[:, :] = -1
    for state in range(n_states):
        kind = state_kind[state]
        if kind == 0:
            a = state_first[state]
            b = state_second[state]
        elif kind == 1:
            a = state_first[state]
            b = wildcard
        else:
            a = wildcard
            b = wildcard
        state_lookup[a, b] = state

    for sample in range(n_samples):
        a = assignments[sample, 0]
        b = assignments[sample, 1]
        state_counts[state_lookup[a, b]] += 1
        if a < wildcard:
            founder_support[a] += 1
        if b < wildcard and b != a:
            founder_support[b] += 1
        if a < b and b < wildcard:
            pair_counts[a, b] += 1

    # Retain the historical ascending-other and ascending-pair orders while
    # storing only pair factors used by at least one real-real assignment.
    neighbours = np.empty((k, k), dtype=np.int64)
    neighbour_counts = np.zeros(k, dtype=np.int64)
    edge_first = np.empty(k * (k - 1) // 2, dtype=np.int64)
    edge_second = np.empty(k * (k - 1) // 2, dtype=np.int64)
    edge_lookup = np.empty((k, k), dtype=np.int64)
    edge_lookup[:, :] = -1
    n_edges = 0
    for first_founder in range(k):
        for second_founder in range(first_founder + 1, k):
            if pair_counts[first_founder, second_founder] == 0:
                continue
            neighbours[first_founder, neighbour_counts[first_founder]] = (
                second_founder
            )
            neighbour_counts[first_founder] += 1
            neighbours[second_founder, neighbour_counts[second_founder]] = (
                first_founder
            )
            neighbour_counts[second_founder] += 1
            edge_first[n_edges] = first_founder
            edge_second[n_edges] = second_founder
            edge_lookup[first_founder, second_founder] = n_edges
            n_edges += 1

    unary = np.empty((k, n_sites, 2), dtype=np.float64)
    pair = np.zeros((n_edges, n_sites, 4), dtype=np.float64)
    for founder in range(k):
        for site in range(n_sites):
            unary[founder, site, 0] = log_half
            unary[founder, site, 1] = log_half
    for sample in range(n_samples):
        a = assignments[sample, 0]
        b = assignments[sample, 1]
        if a == b and a < wildcard:
            for site in range(n_sites):
                unary[a, site, 0] += log_likelihood[sample, site, 0]
                unary[a, site, 1] += log_likelihood[sample, site, 2]
        elif a < b and b < wildcard:
            edge = edge_lookup[a, b]
            for site in range(n_sites):
                pair[edge, site, 0] += log_likelihood[sample, site, 0]
                pair[edge, site, 1] += log_likelihood[sample, site, 1]
                pair[edge, site, 2] += log_likelihood[sample, site, 1]
                pair[edge, site, 3] += log_likelihood[sample, site, 2]
        elif a < wildcard and b == wildcard:
            for site in range(n_sites):
                unary[a, site, 0] += rw_log_likelihood[sample, site, 0]
                unary[a, site, 1] += rw_log_likelihood[sample, site, 1]

    sample_scores = np.zeros(n_samples, dtype=np.float64)
    iterations_by_heldout = np.zeros(n_samples, dtype=np.int64)
    not_converged_by_heldout = np.zeros(n_samples, dtype=np.int64)
    zero_support_by_heldout = np.zeros(n_samples, dtype=np.int64)
    entropy_by_heldout = np.zeros(n_samples, dtype=np.float64)
    alternate_wins_by_heldout = np.zeros(n_samples, dtype=np.int64)
    initialization_spread_by_heldout = np.zeros(
        n_samples, dtype=np.float64
    )
    # The common denominator N-1+alpha is shared by held-out samples/states.
    log_prior_denominator = math.log(
        n_samples - 1 + state_concentration
    )
    # Every held-out prior differs from the full count vector at one state.
    # Cache both alternatives once per mode rather than taking the same logs
    # for every sample.
    full_log_state_mass = np.empty(n_states, dtype=np.float64)
    deleted_log_state_mass = np.empty(n_states, dtype=np.float64)
    for state in range(n_states):
        prior_mass = state_concentration * prior_base[state]
        full_log_state_mass[state] = math.log(
            state_counts[state] + prior_mass
        )
        if state_counts[state] > 0:
            deleted_log_state_mass[state] = math.log(
                state_counts[state] - 1 + prior_mass
            )
        else:
            deleted_log_state_mass[state] = 0.0

    for heldout in prange(n_samples):
        heldout_iterations = 0
        heldout_not_converged = 0
        heldout_zero_support = 0
        heldout_entropy = 0.0
        heldout_alternate_wins = 0
        heldout_initialization_spread = 0.0
        q = np.empty(k, dtype=np.float64)
        candidate_q = np.empty((3, k), dtype=np.float64)
        candidate_elbo = np.empty(3, dtype=np.float64)
        candidate_converged = np.empty(3, dtype=np.uint8)
        candidate_q_log_q = np.empty((3, k), dtype=np.float64)
        candidate_one_minus_q_log_one_minus_q = np.empty(
            (3, k), dtype=np.float64
        )
        emissions = np.empty(n_states, dtype=np.float64)
        log_state_predictive = np.empty(n_states, dtype=np.float64)
        # Held-out corrections depend on the sample and site, but not on the
        # mean-field initialization or iteration.  Materialize them once per
        # site so the three deterministic starts reuse exactly the same
        # corrected floating-point values.  Directed differences retain the
        # historical ascending-neighbour Gauss-Seidel traversal below.
        cavity_unary = np.empty((k, 2), dtype=np.float64)
        cavity_unary_logit = np.empty(k, dtype=np.float64)
        directed_difference_zero = np.empty((k, k), dtype=np.float64)
        directed_difference_one = np.empty((k, k), dtype=np.float64)
        cavity_edge_potential = np.empty((n_edges, 4), dtype=np.float64)
        held_a = assignments[heldout, 0]
        held_b = assignments[heldout, 1]
        held_state = state_lookup[held_a, held_b]
        held_support_a = held_a if held_a < wildcard else -1
        held_support_b = (
            held_b if held_b < wildcard and held_b != held_a else -1
        )
        remaining_support_by_founder = np.empty(k, dtype=np.int64)
        supported_founders = np.empty(k, dtype=np.int64)
        n_supported = 0
        n_zero_support = 0
        for founder in range(k):
            remaining_support = founder_support[founder]
            if founder == held_support_a or founder == held_support_b:
                remaining_support -= 1
            remaining_support_by_founder[founder] = remaining_support
            if remaining_support > 0:
                supported_founders[n_supported] = founder
                n_supported += 1
            else:
                n_zero_support += 1
        heldout_zero_support = n_zero_support * n_sites
        for state in range(n_states):
            log_state_predictive[state] = full_log_state_mass[state]
            emissions[state] = 0.0
        log_state_predictive[held_state] = deleted_log_state_mass[held_state]

        for site in range(n_sites):
            g0 = likelihood[heldout, site, 0]
            g1 = likelihood[heldout, site, 1]
            g2 = likelihood[heldout, site, 2]

            held_log_mix0 = 0.0
            held_log_mix1 = 0.0
            if held_a < wildcard and held_b == wildcard:
                held_log_mix0 = rw_log_likelihood[heldout, site, 0]
                held_log_mix1 = rw_log_likelihood[heldout, site, 1]

            for founder in range(k):
                u0 = unary[founder, site, 0]
                u1 = unary[founder, site, 1]
                if held_a == held_b and held_a == founder:
                    u0 -= log_likelihood[heldout, site, 0]
                    u1 -= log_likelihood[heldout, site, 2]
                elif held_a == founder and held_b == wildcard:
                    u0 -= held_log_mix0
                    u1 -= held_log_mix1
                cavity_unary[founder, 0] = u0
                cavity_unary[founder, 1] = u1
                cavity_unary_logit[founder] = u1 - u0

            for edge_index in range(n_edges):
                first_founder = edge_first[edge_index]
                second_founder = edge_second[edge_index]
                v00 = pair[edge_index, site, 0]
                v01 = pair[edge_index, site, 1]
                v10 = pair[edge_index, site, 2]
                v11 = pair[edge_index, site, 3]
                if (
                    held_a == first_founder
                    and held_b == second_founder
                ):
                    v00 -= log_likelihood[heldout, site, 0]
                    v01 -= log_likelihood[heldout, site, 1]
                    v10 -= log_likelihood[heldout, site, 1]
                    v11 -= log_likelihood[heldout, site, 2]
                cavity_edge_potential[edge_index, 0] = v00
                cavity_edge_potential[edge_index, 1] = v01
                cavity_edge_potential[edge_index, 2] = v10
                cavity_edge_potential[edge_index, 3] = v11
                directed_difference_zero[
                    first_founder, second_founder
                ] = v10 - v00
                directed_difference_one[
                    first_founder, second_founder
                ] = v11 - v01
                directed_difference_zero[
                    second_founder, first_founder
                ] = v01 - v00
                directed_difference_one[
                    second_founder, first_founder
                ] = v11 - v10

            candidate_converged[:] = 0
            for start_index in range(3):
                for founder in range(k):
                    if (
                        remaining_support_by_founder[founder] == 0
                        or start_index == 2
                    ):
                        q[founder] = 0.5
                    elif start_index == 0:
                        q[founder] = float(haplotypes[founder, site])
                    else:
                        q[founder] = 1.0 - float(
                            haplotypes[founder, site]
                        )

                converged = False
                used_iterations = 0
                for iteration in range(mean_field_max_iter):
                    maximum_change = 0.0
                    for supported_index in range(n_supported):
                        founder = supported_founders[supported_index]
                        # An isolated founder reaches its exact unary update on
                        # the first sweep; subsequent sweeps would repeat the
                        # same sigmoid and contribute an exact zero change.
                        if neighbour_counts[founder] == 0 and iteration > 0:
                            continue
                        logit = cavity_unary_logit[founder]

                        for neighbour_index in range(neighbour_counts[founder]):
                            other = neighbours[founder, neighbour_index]
                            qo = q[other]
                            logit += (1.0 - qo) * (
                                directed_difference_zero[founder, other]
                            )
                            logit += qo * (
                                directed_difference_one[founder, other]
                            )

                        updated = _stable_sigmoid(logit)
                        change = abs(updated - q[founder])
                        if change > maximum_change:
                            maximum_change = change
                        q[founder] = updated
                    used_iterations = iteration + 1
                    if maximum_change <= mean_field_tolerance:
                        converged = True
                        break
                heldout_iterations += used_iterations
                if converged:
                    candidate_converged[start_index] = 1

                elbo = 0.0
                for founder in range(k):
                    u0 = cavity_unary[founder, 0]
                    u1 = cavity_unary[founder, 1]
                    probability = q[founder]
                    elbo += (1.0 - probability) * u0 + probability * u1
                    q_log_q = 0.0
                    one_minus_q_log_one_minus_q = 0.0
                    if probability > 0.0 and probability < 1.0:
                        q_log_q = probability * math.log(probability)
                        one_minus_q_log_one_minus_q = (
                            (1.0 - probability) * math.log(
                                1.0 - probability
                            )
                        )
                        elbo -= q_log_q
                        elbo -= one_minus_q_log_one_minus_q
                    candidate_q_log_q[start_index, founder] = q_log_q
                    candidate_one_minus_q_log_one_minus_q[
                        start_index, founder
                    ] = one_minus_q_log_one_minus_q
                for edge_index in range(n_edges):
                    first_founder = edge_first[edge_index]
                    second_founder = edge_second[edge_index]
                    v00 = cavity_edge_potential[edge_index, 0]
                    v01 = cavity_edge_potential[edge_index, 1]
                    v10 = cavity_edge_potential[edge_index, 2]
                    v11 = cavity_edge_potential[edge_index, 3]
                    q_first = q[first_founder]
                    q_second = q[second_founder]
                    elbo += (1.0 - q_first) * (1.0 - q_second) * v00
                    elbo += (1.0 - q_first) * q_second * v01
                    elbo += q_first * (1.0 - q_second) * v10
                    elbo += q_first * q_second * v11
                candidate_elbo[start_index] = elbo
                candidate_q[start_index, :] = q

            any_converged = False
            best_start = 0
            best_elbo = -math.inf
            minimum_converged_elbo = math.inf
            for start_index in range(3):
                if candidate_converged[start_index] == 1:
                    any_converged = True
                    value = candidate_elbo[start_index]
                    better = value > best_elbo + 1e-12
                    if abs(value - best_elbo) <= 1e-12:
                        for founder in range(k):
                            difference = (
                                candidate_q[start_index, founder]
                                - candidate_q[best_start, founder]
                            )
                            if difference < -1e-12:
                                better = True
                                break
                            if difference > 1e-12:
                                break
                    if better:
                        best_elbo = value
                        best_start = start_index
                    if value < minimum_converged_elbo:
                        minimum_converged_elbo = value
            if not any_converged:
                heldout_not_converged += 1
                best_start = 0
                best_elbo = candidate_elbo[0]
                for start_index in range(1, 3):
                    value = candidate_elbo[start_index]
                    better = value > best_elbo + 1e-12
                    if abs(value - best_elbo) <= 1e-12:
                        for founder in range(k):
                            difference = (
                                candidate_q[start_index, founder]
                                - candidate_q[best_start, founder]
                            )
                            if difference < -1e-12:
                                better = True
                                break
                            if difference > 1e-12:
                                break
                    if better:
                        best_elbo = value
                        best_start = start_index
            else:
                heldout_initialization_spread += (
                    best_elbo - minimum_converged_elbo
                )
            if best_start != 0:
                heldout_alternate_wins += 1
            q[:] = candidate_q[best_start, :]

            for founder in range(k):
                probability = q[founder]
                if probability > 0.0 and probability < 1.0:
                    heldout_entropy -= candidate_q_log_q[
                        best_start, founder
                    ]
                    heldout_entropy -= (
                        candidate_one_minus_q_log_one_minus_q[
                            best_start, founder
                        ]
                    )

            # Geometry is ordered RR (row-major upper triangle), RW, WW.
            # Specialize those runs while retaining each state's index and
            # the historical arithmetic within its emission.
            state = 0
            for first in range(k):
                qi = q[first]
                predictive = (1.0 - qi) * g0 + qi * g2
                if predictive < likelihood_floor:
                    predictive = likelihood_floor
                emissions[state] += math.log(predictive)
                state += 1
                for second in range(first + 1, k):
                    qj = q[second]
                    p0 = (1.0 - qi) * (1.0 - qj)
                    p2 = qi * qj
                    p1 = 1.0 - p0 - p2
                    predictive = p0 * g0 + p1 * g1 + p2 * g2
                    if predictive < likelihood_floor:
                        predictive = likelihood_floor
                    emissions[state] += math.log(predictive)
                    state += 1
            for first in range(k):
                qi = q[first]
                predictive = (
                    0.5 * (1.0 - qi) * g0
                    + 0.5 * g1
                    + 0.5 * qi * g2
                )
                if predictive < likelihood_floor:
                    predictive = likelihood_floor
                emissions[state] += math.log(predictive)
                state += 1
            emissions[state] += ww_log_emission[heldout, site]

        for state in range(n_states):
            log_state_predictive[state] += emissions[state]
            log_state_predictive[state] -= log_prior_denominator
        sample_scores[heldout] = _logaddexp_values(log_state_predictive)

        iterations_by_heldout[heldout] = heldout_iterations
        not_converged_by_heldout[heldout] = heldout_not_converged
        zero_support_by_heldout[heldout] = heldout_zero_support
        entropy_by_heldout[heldout] = heldout_entropy
        alternate_wins_by_heldout[heldout] = heldout_alternate_wins
        initialization_spread_by_heldout[heldout] = (
            heldout_initialization_spread
        )

    # Reduce audit quantities in historical held-out sample order.
    total_iterations = 0
    not_converged = 0
    zero_support_cavities = 0
    entropy_sum = 0.0
    alternate_initialization_wins = 0
    initialization_spread_sum = 0.0
    for heldout in range(n_samples):
        total_iterations += iterations_by_heldout[heldout]
        not_converged += not_converged_by_heldout[heldout]
        zero_support_cavities += zero_support_by_heldout[heldout]
        entropy_sum += entropy_by_heldout[heldout]
        alternate_initialization_wins += alternate_wins_by_heldout[heldout]
        initialization_spread_sum += (
            initialization_spread_by_heldout[heldout]
        )

    return (
        sample_scores,
        total_iterations,
        not_converged,
        zero_support_cavities,
        entropy_sum,
        alternate_initialization_wins,
        initialization_spread_sum,
    )


@njit(cache=True)
def _infer_anchored_cavity_q_kernel(
    likelihood: np.ndarray,
    log_likelihood: np.ndarray,
    haplotypes: np.ndarray,
    assignments: np.ndarray,
    likelihood_floor: float,
) -> tuple[np.ndarray, int, float]:
    """Founder probabilities from hard-partner conditional pseudolikelihood."""

    n_samples, n_sites, _ = likelihood.shape
    k = haplotypes.shape[0]
    wildcard = k
    total_delta = np.zeros((k, n_sites), dtype=np.float64)
    founder_support = np.zeros(k, dtype=np.int64)
    for sample in range(n_samples):
        a = assignments[sample, 0]
        b = assignments[sample, 1]
        if a < wildcard:
            founder_support[a] += 1
        if b < wildcard and b != a:
            founder_support[b] += 1
        if a == b and a < wildcard:
            for site in range(n_sites):
                total_delta[a, site] += (
                    log_likelihood[sample, site, 2]
                    - log_likelihood[sample, site, 0]
                )
        elif a < b and b < wildcard:
            for site in range(n_sites):
                ha = haplotypes[a, site]
                hb = haplotypes[b, site]
                total_delta[a, site] += (
                    log_likelihood[sample, site, hb + 1]
                    - log_likelihood[sample, site, hb]
                )
                total_delta[b, site] += (
                    log_likelihood[sample, site, ha + 1]
                    - log_likelihood[sample, site, ha]
                )
        elif a < wildcard and b == wildcard:
            for site in range(n_sites):
                g0 = likelihood[sample, site, 0]
                g1 = likelihood[sample, site, 1]
                g2 = likelihood[sample, site, 2]
                mix0 = 0.5 * (g0 + g1)
                mix1 = 0.5 * (g1 + g2)
                if mix0 < likelihood_floor:
                    mix0 = likelihood_floor
                if mix1 < likelihood_floor:
                    mix1 = likelihood_floor
                total_delta[a, site] += math.log(mix1) - math.log(mix0)

    q_all = np.empty((n_samples, n_sites, k), dtype=np.float64)
    zero_support_cavities = 0
    entropy_sum = 0.0
    for heldout in range(n_samples):
        a = assignments[heldout, 0]
        b = assignments[heldout, 1]
        for site in range(n_sites):
            for founder in range(k):
                remaining_support = founder_support[founder]
                if founder == a or (founder == b and b != a):
                    remaining_support -= 1
                if remaining_support == 0:
                    q = 0.5
                    zero_support_cavities += 1
                else:
                    delta = total_delta[founder, site]
                    if a == b and a == founder:
                        delta -= (
                            log_likelihood[heldout, site, 2]
                            - log_likelihood[heldout, site, 0]
                        )
                    elif a < b and b < wildcard:
                        if founder == a:
                            hb = haplotypes[b, site]
                            delta -= (
                                log_likelihood[heldout, site, hb + 1]
                                - log_likelihood[heldout, site, hb]
                            )
                        elif founder == b:
                            ha = haplotypes[a, site]
                            delta -= (
                                log_likelihood[heldout, site, ha + 1]
                                - log_likelihood[heldout, site, ha]
                            )
                    elif a == founder and b == wildcard:
                        g0 = likelihood[heldout, site, 0]
                        g1 = likelihood[heldout, site, 1]
                        g2 = likelihood[heldout, site, 2]
                        mix0 = 0.5 * (g0 + g1)
                        mix1 = 0.5 * (g1 + g2)
                        if mix0 < likelihood_floor:
                            mix0 = likelihood_floor
                        if mix1 < likelihood_floor:
                            mix1 = likelihood_floor
                        delta -= math.log(mix1) - math.log(mix0)
                    q = _stable_sigmoid(delta)
                q_all[heldout, site, founder] = q
                if q > 0.0 and q < 1.0:
                    entropy_sum -= q * math.log(q)
                    entropy_sum -= (1.0 - q) * math.log(1.0 - q)
    return q_all, zero_support_cavities, entropy_sum


@njit(cache=True)
def _score_given_cavity_q_kernel(
    likelihood: np.ndarray,
    q_all: np.ndarray,
    assignments: np.ndarray,
    state_kind: np.ndarray,
    state_first: np.ndarray,
    state_second: np.ndarray,
    prior_base: np.ndarray,
    state_concentration: float,
    likelihood_floor: float,
) -> np.ndarray:
    """Marginalize RR/RW/WW states for supplied cavity founder probabilities."""

    n_samples, n_sites, _ = likelihood.shape
    k = q_all.shape[2]
    wildcard = k
    n_states = state_kind.shape[0]
    state_lookup = np.empty((k + 1, k + 1), dtype=np.int64)
    state_lookup[:, :] = -1
    state_counts = np.zeros(n_states, dtype=np.int64)
    for state in range(n_states):
        kind = state_kind[state]
        if kind == 0:
            a = state_first[state]
            b = state_second[state]
        elif kind == 1:
            a = state_first[state]
            b = wildcard
        else:
            a = wildcard
            b = wildcard
        state_lookup[a, b] = state
    for sample in range(n_samples):
        state_counts[state_lookup[
            assignments[sample, 0], assignments[sample, 1]
        ]] += 1

    sample_scores = np.empty(n_samples, dtype=np.float64)
    state_values = np.empty(n_states, dtype=np.float64)
    for heldout in range(n_samples):
        held_state = state_lookup[
            assignments[heldout, 0], assignments[heldout, 1]
        ]
        prior_denominator = math.log(n_samples - 1 + state_concentration)
        for state in range(n_states):
            count = state_counts[state]
            if state == held_state:
                count -= 1
            state_values[state] = (
                math.log(count + state_concentration * prior_base[state])
                - prior_denominator
            )
        for site in range(n_sites):
            g0 = likelihood[heldout, site, 0]
            g1 = likelihood[heldout, site, 1]
            g2 = likelihood[heldout, site, 2]
            for state in range(n_states):
                kind = state_kind[state]
                first = state_first[state]
                second = state_second[state]
                if kind == 0:
                    qi = q_all[heldout, site, first]
                    if first == second:
                        predictive = (1.0 - qi) * g0 + qi * g2
                    else:
                        qj = q_all[heldout, site, second]
                        p0 = (1.0 - qi) * (1.0 - qj)
                        p2 = qi * qj
                        predictive = (
                            p0 * g0 + (1.0 - p0 - p2) * g1 + p2 * g2
                        )
                elif kind == 1:
                    qi = q_all[heldout, site, first]
                    predictive = (
                        0.5 * (1.0 - qi) * g0
                        + 0.5 * g1
                        + 0.5 * qi * g2
                    )
                else:
                    predictive = 0.25 * g0 + 0.5 * g1 + 0.25 * g2
                if predictive < likelihood_floor:
                    predictive = likelihood_floor
                state_values[state] += math.log(predictive)
        sample_scores[heldout] = _logaddexp_values(state_values)
    return sample_scores


def _score_prepared_mode(
    evidence: np.ndarray,
    log_likelihood: np.ndarray,
    rw_log_likelihood: np.ndarray,
    ww_log_emission: np.ndarray,
    mode: _PreparedMode,
    config: CavitySelectionConfig,
    state_geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, int, int, int, float, int, float]:
    kind, first, second, prior_base = state_geometry
    haplotypes = np.ascontiguousarray(mode.haplotypes, dtype=np.int64)
    assignments = np.ascontiguousarray(mode.assignments, dtype=np.int64)
    if config.founder_inference == "anchored_pseudolikelihood":
        q_all, zero_support, entropy = _infer_anchored_cavity_q_kernel(
            evidence,
            log_likelihood,
            haplotypes,
            assignments,
            float(config.likelihood_floor),
        )
        sample_scores = _score_given_cavity_q_kernel(
            evidence,
            q_all,
            assignments,
            kind,
            first,
            second,
            prior_base,
            float(config.state_concentration),
            float(config.likelihood_floor),
        )
        iterations = 0
        not_converged = 0
        alternate_wins = 0
        initialization_spread = 0.0
    else:
        values = _score_mode_cavity_kernel(
            evidence,
            log_likelihood,
            rw_log_likelihood,
            ww_log_emission,
            haplotypes,
            assignments,
            kind,
            first,
            second,
            prior_base,
            float(config.state_concentration),
            int(config.mean_field_max_iter),
            float(config.mean_field_tolerance),
            float(config.likelihood_floor),
        )
        (
            sample_scores,
            iterations,
            not_converged,
            zero_support,
            entropy,
            alternate_wins,
            initialization_spread,
        ) = values
    total = math.fsum(float(value) for value in sample_scores)
    n_site_fits = evidence.shape[0] * evidence.shape[1]
    mean_iterations = iterations / n_site_fits
    entropy_denominator = n_site_fits * mode.k
    mean_entropy = entropy / entropy_denominator
    return (
        total,
        mean_iterations,
        n_site_fits,
        int(not_converged),
        int(zero_support),
        mean_entropy,
        int(alternate_wins),
        initialization_spread / n_site_fits,
    )


def select_cavity_predictive_k(
    evidence: np.ndarray,
    modes_by_k: Mapping[int, Sequence[Any]],
    *,
    config: CavitySelectionConfig | None = None,
    _workspace: _CavityScoringWorkspace | None = None,
) -> CavitySelection:
    """Compare represented K values using fixed-A sample cavities.

    Every canonical unique input mode is scored.  The highest cavity score at
    each K is the operational within-K choice; exact score ties use the stable
    canonical digest.  The uniform represented-mode log-mean-exp is reported
    only as a sensitivity summary.  The resulting weights remain
    pseudo-weights because the represented support and fixed assignments saw
    all evidence.
    """

    settings = CavitySelectionConfig() if config is None else config
    if not isinstance(settings, CavitySelectionConfig):
        raise TypeError("config must be a CavitySelectionConfig")
    if _workspace is None:
        likelihood = _validate_evidence(evidence)
        log_likelihood = _floor_log_likelihood_kernel(
            likelihood, float(settings.likelihood_floor)
        )
        rw_log_likelihood, ww_log_emission = (
            _build_evidence_emission_cache_kernel(
                likelihood, float(settings.likelihood_floor)
            )
        )
        state_geometry_by_k: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
    else:
        if not isinstance(_workspace, _CavityScoringWorkspace):
            raise TypeError("_workspace must be a _CavityScoringWorkspace")
        if _workspace.evidence_reference is not evidence:
            raise ValueError("cavity workspace was prepared for other evidence")
        if _workspace.config != settings:
            raise ValueError("cavity workspace configuration does not match")
        likelihood = _workspace.likelihood
        log_likelihood = _workspace.log_likelihood
        rw_log_likelihood = _workspace.rw_log_likelihood
        ww_log_emission = _workspace.ww_log_emission
        state_geometry_by_k = _workspace.state_geometry_by_k
    n_samples, n_sites, _ = likelihood.shape
    modes, counts = _prepare_modes(modes_by_k, n_samples, n_sites)

    scored_modes_by_k: dict[int, tuple[_PreparedMode, ...]] = {}
    for k, available_modes in modes.items():
        if (
            settings.max_modes_per_k is None
            or len(available_modes) <= settings.max_modes_per_k
        ):
            scored_modes_by_k[k] = available_modes
        else:
            ranked_modes = sorted(
                available_modes,
                key=lambda mode: (mode.total_nll, mode.digest),
            )
            scored_modes_by_k[k] = tuple(
                ranked_modes[:settings.max_modes_per_k]
            )
    components: dict[
        tuple[int, str], tuple[float, float, int, int, int, float, int, float]
    ] = {}
    best_by_k: dict[int, _PreparedMode] = {}
    uniform_log_mean_exp: dict[int, float] = {}
    log_haplotype_set_prior: dict[int, float] = {}
    log_scores = []
    k_values = tuple(sorted(modes))
    for k in k_values:
        if k not in state_geometry_by_k:
            state_geometry_by_k[k] = _build_state_geometry(k, settings)
    for k in k_values:
        mode_scores = []
        for mode in scored_modes_by_k[k]:
            component = _score_prepared_mode(
                likelihood,
                log_likelihood,
                rw_log_likelihood,
                ww_log_emission,
                mode,
                settings,
                state_geometry_by_k[k],
            )
            components[(k, mode.digest)] = component
            mode_scores.append(component[0])
        best_index = min(
            range(len(scored_modes_by_k[k])),
            key=lambda index: (
                -mode_scores[index], scored_modes_by_k[k][index].digest
            ),
        )
        best_by_k[k] = scored_modes_by_k[k][best_index]
        cavity_total = mode_scores[best_index]
        maximum_mode_score = max(mode_scores)
        uniform_log_mean_exp[k] = (
            maximum_mode_score
            + math.log(math.fsum(
                math.exp(value - maximum_mode_score)
                for value in mode_scores
            ))
            - math.log(len(mode_scores))
        )
        set_prior = (
            -discovery_objectives.log_binary_haplotype_set_count(n_sites, k)
            if settings.apply_unordered_haplotype_set_code
            else 0.0
        )
        log_haplotype_set_prior[k] = set_prior
        log_scores.append(
            cavity_total - math.log(k) - math.log(k + 1) + set_prior
        )
    log_scores_array = np.asarray(log_scores, dtype=np.float64)
    normalized_logs, normalized_probabilities = _normalized_log_weights(log_scores_array)
    log_weights = np.asarray(normalized_logs, dtype=np.float64)
    probabilities = np.asarray(normalized_probabilities, dtype=np.float64)
    ranking = sorted(
        range(len(k_values)),
        key=lambda index: (-log_scores_array[index], k_values[index]),
    )

    diagnostics = []
    mode_diagnostics = []
    for index, k in enumerate(k_values):
        best_mode = best_by_k[k]
        (
            total,
            mean_iter,
            n_fits,
            nonconv,
            zero_support,
            entropy,
            alternate_wins,
            initialization_spread,
        ) = components[(k, best_mode.digest)]
        n_input, n_unique = counts[k]
        diagnostics.append(CavityKDiagnostic(
            k=k,
            cavity_log_predictive=total,
            mean_sample_log_predictive=total / n_samples,
            log_k_prior=-math.log(k) - math.log(k + 1),
            log_haplotype_set_prior=log_haplotype_set_prior[k],
            log_score=float(log_scores_array[index]),
            log_weight=float(log_weights[index]),
            pseudo_probability=float(probabilities[index]),
            best_total_nll=best_mode.total_nll,
            best_mode_digest=best_mode.digest,
            n_input_modes=n_input,
            n_unique_modes=n_unique,
            n_duplicate_modes_removed=n_input - n_unique,
            n_mean_field_site_fits=n_fits,
            n_mean_field_not_converged=nonconv,
            mean_mean_field_iterations=mean_iter,
            n_zero_support_founder_cavities=zero_support,
            mean_founder_allele_entropy_nats=entropy,
            n_alternate_initialization_wins=alternate_wins,
            mean_initialization_elbo_spread=initialization_spread,
            uniform_mode_log_mean_exp=uniform_log_mean_exp[k],
            n_modes_scored=len(scored_modes_by_k[k]),
            n_modes_omitted_by_cap=(
                n_unique - len(scored_modes_by_k[k])
            ),
        ))
        for mode in scored_modes_by_k[k]:
            component = components[(k, mode.digest)]
            mode_diagnostics.append(CavityModeDiagnostic(
                k=k,
                mode_digest=mode.digest,
                total_nll=mode.total_nll,
                cavity_log_predictive=component[0],
                mean_sample_log_predictive=component[0] / n_samples,
                n_mean_field_site_fits=component[2],
                n_mean_field_not_converged=component[3],
                mean_mean_field_iterations=component[1],
                n_zero_support_founder_cavities=component[4],
                mean_founder_allele_entropy_nats=component[5],
                n_alternate_initialization_wins=component[6],
                mean_initialization_elbo_spread=component[7],
                selected_within_k=(mode.digest == best_mode.digest),
            ))
    map_k = k_values[ranking[0]]
    mode_scope_interpretation = (
        "All canonical unique modes were cavity-scored. "
        if settings.max_modes_per_k is None
        else (
            f"At most {settings.max_modes_per_k} canonical modes per K were "
            "cavity-scored after explicit minimum-full-data-NLL screening; "
            "this additional screening is selection-leakage affected. "
        )
    )
    founder_interpretation = (
        "Founder fields use deterministic mean-field coordinate ascent. "
        if settings.founder_inference == "mean_field"
        else (
            "Founder fields use anchored pseudolikelihood: each distinct-pair "
            "factor is conditioned on the fitted hard partner allele before "
            "the held-out contribution is deleted. "
        )
    )
    regularization_interpretation = (
        "The objective additionally subtracts log C(2^L,K), the ideal code "
        "length for an unordered set of K distinct binary L-site haplotypes. "
        "This regularized fixed-A pseudo-predictive objective is intended "
        "to counter residual full-data mode-support and assignment-selection "
        "leakage. It is not "
        "calibrated leave-one-out evidence or an MDL result, and it does not "
        "claim that double counting is absent. "
        if settings.apply_unordered_haplotype_set_code
        else "No haplotype-set complexity code is applied. "
    )
    return CavitySelection(
        method=(
            "regularized_fixed_assignment_cavity_with_unordered_set_code"
            if settings.apply_unordered_haplotype_set_code
            else "fixed_assignment_leave_one_sample_cavity"
        ),
        map_k=map_k,
        runner_up_k=(
            None if len(ranking) == 1 else k_values[ranking[1]]
        ),
        selected_mode_digest=best_by_k[map_k].digest,
        k_diagnostics=tuple(diagnostics),
        mode_diagnostics=tuple(mode_diagnostics),
        support_selected_from_full_data=True,
        assignments_selected_from_full_data=True,
        selection_leakage=True,
        weights_are_calibrated=False,
        k_prior="telescoping_1_over_k_k_plus_1_applied_once",
        boundary_limited=(map_k == max(k_values)),
        founder_inference=settings.founder_inference,
        apply_unordered_haplotype_set_code=(
            settings.apply_unordered_haplotype_set_code
        ),
        mode_cap_per_k=settings.max_modes_per_k,
        mode_cap_applied=any(
            len(scored_modes_by_k[k]) < len(modes[k]) for k in k_values
        ),
        all_mean_field_converged=all(
            item.n_mean_field_not_converged == 0
            for item in mode_diagnostics
        ),
        interpretation=(
            "Fixed-assignment leave-one-sample classification-cavity score. "
            "Each sample's exact MRF likelihood factor and fitted-state count "
            "are deleted before predicting it. "
            + founder_interpretation
            + "All RR/RW/WW states are marginalized. The telescoping K prior "
            "is applied exactly once. "
            + mode_scope_interpretation
            + regularization_interpretation
            + "The maximum cavity "
            "score is used within K; uniform mode log-mean-exp is diagnostic "
            "only. Mode support and A were selected from the full data, so "
            "normalized values are uncalibrated selection-leakage-affected "
            "pseudo-weights, not posteriors."
        ),
        n_samples=n_samples,
        n_sites=n_sites,
    )

import haplotype_reconstruction.core.genotypes as core_genotypes
import haplotype_reconstruction.discovery.modes as discovery_modes
import haplotype_reconstruction.discovery.objectives as discovery_objectives
