"""Block-level adapter for cap-free reversible cavity discovery.

This module deliberately keeps the reversible controller independent of the
legacy robust-discovery and exhaustive fixed-K search paths.  It converts the
kept allele depths to normalized raw genotype likelihoods, invokes one
adaptive complete-panel search, and materializes the exact selected H/A state
through the established cavity result contract.  It does not enumerate K,
does not use metadata or truth, and does not inject a legacy-discovered panel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import math
from typing import Any, Mapping, Sequence

import numpy as np

from bhd_genotype_evidence import (
    allele_depths_to_raw_genotype_likelihoods,
)
from bhd_factorization_modes import FactorizationMode
from bhd_reversible_cavity import (
    ReversibleCavitySearchConfig,
    as_cavity_selection,
    search_reversible_cavity,
)


RAW_EVIDENCE_MODE = "raw_likelihood"
CAVITY_SCORE_CALIBRATION = (
    "uncalibrated_selection_leakage_affected_pseudo_score"
)
CAVITY_WEIGHT_CALIBRATION = (
    "uncalibrated_selection_leakage_affected_pseudo_weight"
)


class ReversibleDiscoveryError(RuntimeError):
    """Raised when reversible discovery cannot process a supported block."""


def _readonly(
    value: np.ndarray,
    dtype: np.dtype[Any] | type | None = None,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _raw_genotype_likelihoods(
    reads: np.ndarray,
    read_error_probability: float,
) -> np.ndarray:
    """Return normalized raw genotype likelihoods with no HWE prior."""

    return allele_depths_to_raw_genotype_likelihoods(
        reads, read_error_probability,
        require_nonempty=True, require_integer=True,
    )


@dataclass(frozen=True)
class ReversibleCandidateSearchDiagnostic:
    """Compact provenance for the adaptive search used by this adapter.

    This intentionally does not mimic the exhaustive-search candidate
    diagnostic: no robust base panel, K grid, pre-cleanup subset sweep, or
    proposal-seed cap is part of this execution path.
    """

    search_kind: str
    data_start_count: int
    supplied_panel_start_count: int
    supplied_candidate_row_count: int
    natural_k_ceiling: int
    exact_score_evaluations: int
    exact_score_cache_hits: int
    stop_reason: str
    search_limited: bool
    search_limit_reasons: tuple[str, ...]
    local_neighbourhood_certified: bool
    local_certificate_scope: str
    search_interpretation: str


@dataclass(frozen=True)
class CavityModeSupport:
    """Immutable wrapper around exact rich modes from one full-data search."""

    modes_by_k: tuple[tuple[int, tuple[FactorizationMode, ...]], ...]

    @classmethod
    def from_mapping(
        cls,
        modes_by_k: Mapping[int, Sequence[FactorizationMode]],
    ) -> "CavityModeSupport":
        return cls(tuple(
            (int(k), tuple(modes_by_k[k])) for k in sorted(modes_by_k)
        ))

    def __post_init__(self) -> None:
        k_values = tuple(k for k, _modes in self.modes_by_k)
        if not k_values or k_values != tuple(sorted(set(k_values))):
            raise ValueError("cavity mode support must have unique sorted K")
        for k, modes in self.modes_by_k:
            if k < 1 or not modes:
                raise ValueError("every represented K needs at least one mode")
            if any(
                not isinstance(mode, FactorizationMode) or mode.k != k
                for mode in modes
            ):
                raise TypeError("cavity support requires rich modes at their K")

    @property
    def k_values(self) -> tuple[int, ...]:
        return tuple(k for k, _modes in self.modes_by_k)

    def as_mapping(self) -> Mapping[int, tuple[FactorizationMode, ...]]:
        """Return a fresh mapping while retaining exact mode objects."""

        return dict(self.modes_by_k)

    def modes(self, k: int) -> tuple[FactorizationMode, ...]:
        for represented_k, modes in self.modes_by_k:
            if represented_k == k:
                return modes
        raise KeyError(k)


@dataclass(frozen=True)
class CavityDiscoveryDiagnostics:
    """Audit summary for one full-data rich search and cavity selection."""

    status: str
    inference_kind: str
    search_pass_count: int
    represented_k_values: tuple[int, ...]
    selected_k: int
    runner_up_k: int | None
    selected_mode_digest: str
    selection_method: str
    selection_config_type: str
    candidate_search: ReversibleCandidateSearchDiagnostic
    terminalize_precleanup_seed_modes: bool
    terminalize_proposal_seed_modes: bool
    apply_terminal_merge_repairs: bool
    min_supporters_for_confidence: int
    genotype_evidence_mode: str
    genotype_evidence_interpretation: str
    cavity_score_calibration: str
    cavity_weight_calibration: str
    cavity_scores_are_calibrated: bool
    cavity_weights_are_calibrated: bool
    support_selected_from_full_data: bool
    assignments_selected_from_full_data: bool
    selection_leakage: bool
    boundary_limited: bool
    uncertainty_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CavityMaterializedBlockData:
    """Legacy-compatible arrays materialized from one exact selected mode."""

    positions: np.ndarray
    haplotype_probability_arrays: tuple[np.ndarray, ...]
    reads_count_matrix: np.ndarray
    keep_flags: np.ndarray
    probs_array: np.ndarray
    discrete_haps: np.ndarray
    per_site_confidence: np.ndarray
    n_site_supporters: np.ndarray
    pair_assignments: np.ndarray
    wildcard_slots: np.ndarray
    wildcard_mass: float
    uncertainty_flag: bool
    K_final: int
    selected_mode: FactorizationMode
    selected_mode_digest: str
    materialization_iterations: int
    selected_mode_iterations: int
    selected_mode_nll: float
    uncertainty_reasons: tuple[str, ...]
    diagnostics: CavityDiscoveryDiagnostics
    selection: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", _readonly(self.positions))
        object.__setattr__(
            self, "reads_count_matrix", _readonly(self.reads_count_matrix)
        )
        object.__setattr__(
            self, "keep_flags", _readonly(self.keep_flags, np.int64)
        )
        object.__setattr__(
            self, "probs_array", _readonly(self.probs_array, np.float64)
        )
        object.__setattr__(
            self, "discrete_haps", _readonly(self.discrete_haps, np.int64)
        )
        object.__setattr__(
            self,
            "per_site_confidence",
            _readonly(self.per_site_confidence, np.float64),
        )
        object.__setattr__(
            self,
            "n_site_supporters",
            _readonly(self.n_site_supporters, np.int64),
        )
        object.__setattr__(
            self,
            "pair_assignments",
            _readonly(self.pair_assignments, np.int64),
        )
        object.__setattr__(
            self,
            "wildcard_slots",
            _readonly(self.wildcard_slots, np.int64),
        )
        object.__setattr__(
            self,
            "haplotype_probability_arrays",
            tuple(
                _readonly(value, np.float64)
                for value in self.haplotype_probability_arrays
            ),
        )
        self.validate()

    @property
    def haplotypes(self) -> dict[int, np.ndarray]:
        return {
            index: value
            for index, value in enumerate(self.haplotype_probability_arrays)
        }

    def validate(self) -> None:
        reads = np.asarray(self.reads_count_matrix)
        if reads.ndim != 3 or reads.shape[2] != 2:
            raise AssertionError("reads_count_matrix has the wrong shape")
        n_samples, n_sites, _ = reads.shape
        k = int(self.K_final)
        if not isinstance(self.selected_mode, FactorizationMode):
            raise TypeError("selected_mode must be a FactorizationMode")
        if self.selected_mode.k != k:
            raise AssertionError("selected mode and K_final disagree")
        if np.asarray(self.positions).shape != (n_sites,):
            raise AssertionError("positions and reads disagree")
        if np.asarray(self.keep_flags).shape != (n_sites,):
            raise AssertionError("keep_flags and reads disagree")
        if np.asarray(self.probs_array).shape != (n_samples, n_sites, 3):
            raise AssertionError("probs_array has the wrong shape")
        expected_k_site = (k, n_sites)
        for name in (
            "discrete_haps",
            "per_site_confidence",
            "n_site_supporters",
        ):
            if np.asarray(getattr(self, name)).shape != expected_k_site:
                raise AssertionError(f"{name} and K_final disagree")
        if len(self.haplotype_probability_arrays) != k:
            raise AssertionError("public haplotypes and K_final disagree")
        if any(
            np.asarray(value).shape != (n_sites, 2)
            for value in self.haplotype_probability_arrays
        ):
            raise AssertionError("a public haplotype has the wrong shape")
        if np.asarray(self.pair_assignments).shape != (n_samples, 2):
            raise AssertionError("pair_assignments has the wrong shape")
        if not np.array_equal(
            self.pair_assignments, self.selected_mode.assignments
        ):
            raise AssertionError("materialization changed selected assignments")
        if np.asarray(self.wildcard_slots).shape != (n_samples,):
            raise AssertionError("wildcard_slots has the wrong shape")
        if not np.array_equal(
            self.wildcard_slots, self.selected_mode.wildcard_slots
        ):
            raise AssertionError("materialization changed wildcard slots")
        expected_mass = float(
            np.sum(self.wildcard_slots, dtype=np.float64)
            / max(2 * n_samples, 1)
        )
        if not math.isclose(
            float(self.wildcard_mass),
            expected_mass,
            rel_tol=0.0,
            abs_tol=np.finfo(np.float64).eps,
        ):
            raise AssertionError("wildcard mass and selected mode disagree")
        if self.materialization_iterations != 0:
            raise AssertionError("cavity materialization must not refit")
        if self.selected_mode_iterations != self.selected_mode.n_iter:
            raise AssertionError("selected-mode iteration provenance disagrees")
        if self.selected_mode_nll != self.selected_mode.total_nll:
            raise AssertionError("selected-mode NLL provenance disagrees")
        threshold = self.diagnostics.min_supporters_for_confidence
        low_support = self.n_site_supporters < threshold
        if np.any(self.discrete_haps[low_support] != -1):
            raise AssertionError("low-support discrete cells must be masked")
        for index, value in enumerate(self.haplotype_probability_arrays):
            if np.any(value[low_support[index]] != 0.5):
                raise AssertionError(
                    "low-support public cells must be (0.5, 0.5)"
                )

    def to_block_result(self, block_result_class: type | None = None) -> Any:
        """Construct the historical result type without changing H or A."""

        self.validate()
        if block_result_class is None:
            from block_haplotypes import BlockResult as block_result_class

        parameters = inspect.signature(block_result_class).parameters.values()
        accepts_mode = any(
            parameter.name == "genotype_evidence_mode"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        kwargs = {
            "keep_flags": self.keep_flags,
            "probs_array": self.probs_array,
        }
        if accepts_mode:
            kwargs["genotype_evidence_mode"] = RAW_EVIDENCE_MODE
        result = block_result_class(
            self.positions,
            self.haplotypes,
            self.reads_count_matrix,
            **kwargs,
        )
        if not accepts_mode:
            result.genotype_evidence_mode = RAW_EVIDENCE_MODE
        result.discrete_haps = self.discrete_haps
        result.per_site_confidence = self.per_site_confidence
        result.n_site_supporters = self.n_site_supporters
        result.pair_assignments = self.pair_assignments
        result.wildcard_slots = self.wildcard_slots
        result.wildcard_mass = self.wildcard_mass
        result.uncertainty_flag = self.uncertainty_flag
        result.K_final = self.K_final
        result.growth_history = []
        keep_mask = self.keep_flags > 0
        precleanup = np.full_like(self.discrete_haps, -1)
        precleanup[:, keep_mask] = self.selected_mode.haplotypes
        result.precleanup_candidate_discrete_haps = precleanup
        result.precleanup_candidate_k = self.K_final
        result.cavity_discovery_diagnostics = asdict(self.diagnostics)
        result.cavity_selection = self.selection
        result.cavity_selected_mode = self.selected_mode
        result.cavity_selected_mode_digest = self.selected_mode_digest
        result.cavity_materialization_iterations = (
            self.materialization_iterations
        )
        result.cavity_selected_mode_iterations = self.selected_mode_iterations
        result.cavity_selected_mode_nll = self.selected_mode_nll
        result.cavity_score_calibration = CAVITY_SCORE_CALIBRATION
        result.cavity_weight_calibration = CAVITY_WEIGHT_CALIBRATION
        result.cavity_materialization_uncertainty_reasons = (
            self.uncertainty_reasons
        )
        return result


@dataclass(frozen=True)
class CavityBlockDiscoveryResult:
    """One-search cavity selection retaining its exact rich mode payload."""

    positions: np.ndarray
    reads_count_matrix: np.ndarray
    keep_flags: np.ndarray
    raw_genotype_likelihoods_kept: np.ndarray
    mode_support: CavityModeSupport
    selection: Any
    selected_mode: FactorizationMode
    diagnostics: CavityDiscoveryDiagnostics
    config: ReversibleCavitySearchConfig

    def __post_init__(self) -> None:
        positions = _readonly(self.positions)
        reads = _readonly(self.reads_count_matrix)
        flags = _readonly(self.keep_flags, np.int64)
        evidence = _readonly(
            self.raw_genotype_likelihoods_kept, np.float64
        )
        if reads.ndim != 3 or reads.shape[2] != 2:
            raise ValueError("reads_count_matrix has the wrong shape")
        if positions.shape != (reads.shape[1],):
            raise ValueError("positions and reads disagree")
        if flags.shape != (reads.shape[1],) or not np.any(flags > 0):
            raise ValueError("keep_flags must retain at least one site")
        if evidence.shape != (reads.shape[0], int(np.sum(flags > 0)), 3):
            raise ValueError("kept raw genotype likelihoods have wrong shape")
        if not isinstance(self.mode_support, CavityModeSupport):
            raise TypeError("mode_support must be a CavityModeSupport")
        if not isinstance(self.selected_mode, FactorizationMode):
            raise TypeError("selected_mode must be a FactorizationMode")
        if self.selected_mode.k != int(self.selection.map_k):
            raise AssertionError("selection and exact selected mode disagree")
        if self.selected_mode.n_sites != evidence.shape[1]:
            raise AssertionError("selected mode and kept sites disagree")
        if self.selected_mode.assignments.shape != (reads.shape[0], 2):
            raise AssertionError("selected mode and samples disagree")
        if not any(
            mode is self.selected_mode
            for mode in self.mode_support.modes(self.selected_mode.k)
        ):
            raise AssertionError("selected mode is not exact search support")
        if bool(getattr(self.selection, "weights_are_calibrated", True)):
            raise ValueError("cavity weights must be labelled uncalibrated")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "reads_count_matrix", reads)
        object.__setattr__(self, "keep_flags", flags)
        object.__setattr__(self, "raw_genotype_likelihoods_kept", evidence)

    @property
    def selected_k(self) -> int:
        return self.selected_mode.k

    @property
    def cavity_pseudo_probability_by_k(self) -> Mapping[int, float]:
        return dict(self.selection.probability_by_k)

    def materialize(self) -> CavityMaterializedBlockData:
        """Expand the exact selected mode without any fitted-state update."""

        mode = self.selected_mode
        keep_mask = self.keep_flags > 0
        from block_haplotypes import (
            _compute_per_site_confidence,
            _discrete_haps_to_prob_arrays,
        )

        confidence_kept, supporters_kept = _compute_per_site_confidence(
            self.raw_genotype_likelihoods_kept,
            mode.haplotypes,
            mode.assignments,
            self.config.lambda_wildcard_penalty,
            min_supporters=self.config.min_supporters_for_confidence,
        )
        n_sites = len(self.positions)
        h_full = np.zeros((mode.k, n_sites), dtype=np.int64)
        confidence_full = np.zeros((mode.k, n_sites), dtype=np.float64)
        supporters_full = np.zeros((mode.k, n_sites), dtype=np.int64)
        h_full[:, keep_mask] = mode.haplotypes
        confidence_full[:, keep_mask] = confidence_kept
        supporters_full[:, keep_mask] = supporters_kept
        public = _discrete_haps_to_prob_arrays(
            h_full,
            n_sites,
            keep_mask,
            confidence_full,
            supporters_full,
            self.config.min_supporters_for_confidence,
        )
        h_masked = h_full.copy()
        h_masked[
            supporters_full < self.config.min_supporters_for_confidence
        ] = -1
        full_probabilities = _raw_genotype_likelihoods(
            self.reads_count_matrix, self.config.read_error_probability
        )
        wildcard_mass = float(
            np.sum(mode.wildcard_slots, dtype=np.float64)
            / max(2 * len(self.reads_count_matrix), 1)
        )
        reasons = list(self.diagnostics.uncertainty_reasons)
        if wildcard_mass > 0.0:
            reasons.append("selected_mode_uses_wildcard_copies")
        return CavityMaterializedBlockData(
            positions=self.positions,
            haplotype_probability_arrays=tuple(
                public[index] for index in range(mode.k)
            ),
            reads_count_matrix=self.reads_count_matrix,
            keep_flags=self.keep_flags,
            probs_array=full_probabilities,
            discrete_haps=h_masked,
            per_site_confidence=confidence_full,
            n_site_supporters=supporters_full,
            pair_assignments=mode.assignments,
            wildcard_slots=mode.wildcard_slots,
            wildcard_mass=wildcard_mass,
            uncertainty_flag=bool(reasons),
            K_final=mode.k,
            selected_mode=mode,
            selected_mode_digest=self.diagnostics.selected_mode_digest,
            materialization_iterations=0,
            selected_mode_iterations=mode.n_iter,
            selected_mode_nll=mode.total_nll,
            uncertainty_reasons=tuple(dict.fromkeys(reasons)),
            diagnostics=self.diagnostics,
            selection=self.selection,
        )

    def materialize_state(self) -> CavityMaterializedBlockData:
        """Compatibility spelling for exact selected-mode materialization."""

        return self.materialize()

    def to_block_result(
        self,
        *,
        block_result_class: type | None = None,
    ) -> Any:
        return self.materialize().to_block_result(block_result_class)


def _validate_reversible_inputs(
    positions: np.ndarray,
    reads_array: np.ndarray,
    keep_flags: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the supported block contract without imposing a K envelope."""

    positions_value = np.ascontiguousarray(np.asarray(positions))
    reads = np.ascontiguousarray(np.asarray(reads_array))
    if reads.ndim != 3 or reads.shape[2] != 2:
        raise ValueError("reads_array must have shape (samples, sites, 2)")
    if not np.issubdtype(reads.dtype, np.integer) or np.any(reads < 0):
        raise ValueError(
            "reads_array must contain non-negative integer counts"
        )
    if positions_value.shape != (reads.shape[1],):
        raise ValueError("positions must match the reads site dimension")
    if reads.shape[0] < 1 or reads.shape[1] < 1:
        raise ValueError("reads must contain samples and sites")
    if int(np.sum(reads, dtype=np.int64)) <= 0:
        raise ReversibleDiscoveryError("at least one read is required")
    flags = (
        np.ones(reads.shape[1], dtype=np.int64)
        if keep_flags is None
        else np.ascontiguousarray(np.asarray(keep_flags), dtype=np.int64)
    )
    if flags.shape != (reads.shape[1],) or not np.any(flags > 0):
        raise ValueError("keep_flags must retain at least one site")
    return positions_value, reads, flags


def discover_block_reversible_cavity(
    positions: np.ndarray,
    reads_array: np.ndarray,
    keep_flags: np.ndarray | None = None,
    *,
    config: ReversibleCavitySearchConfig | None = None,
) -> CavityBlockDiscoveryResult:
    """Discover and select one block panel by adaptive reversible search.

    The only K ceiling is the finite identifiable state-space ceiling computed
    by :func:`search_reversible_cavity`.  Operational proposal, scoring, and
    expansion budgets are reported as search limitations rather than treated
    as biological bounds on founder count.
    """

    settings = ReversibleCavitySearchConfig() if config is None else config
    if not isinstance(settings, ReversibleCavitySearchConfig):
        raise TypeError("config must be a ReversibleCavitySearchConfig")
    positions_value, reads, flags = _validate_reversible_inputs(
        positions, reads_array, keep_flags
    )
    keep_mask = flags > 0
    evidence_kept = np.ascontiguousarray(
        _raw_genotype_likelihoods(
            reads, settings.read_error_probability
        )[:, keep_mask, :]
    )
    depths_kept = np.ascontiguousarray(reads[:, keep_mask, :])

    search = search_reversible_cavity(
        evidence_kept,
        allele_depths=depths_kept,
        config=settings,
    )
    selection = as_cavity_selection(search)
    scored_modes: dict[int, list[Any]] = {}
    for score in search.visited_scores:
        scored_modes.setdefault(int(score.k), []).append(score.mode)
    mode_support = CavityModeSupport.from_mapping(scored_modes)

    uncertainty_reasons = [
        "cavity_scores_and_weights_are_uncalibrated",
        "mode_support_and_assignments_selected_from_full_data",
    ]
    if search.search_limited:
        uncertainty_reasons.append("reversible_search_operationally_limited")
        uncertainty_reasons.extend(
            f"reversible_search_limit:{reason}"
            for reason in search.search_limit_reasons
        )
    if selection.boundary_limited:
        uncertainty_reasons.append(
            "selected_k_at_natural_identifiability_ceiling"
        )
    if selection.mode_cap_applied:
        uncertainty_reasons.append(
            "minimum_full_data_nll_representative_selected_within_each_k"
        )
    if not selection.all_mean_field_converged:
        uncertainty_reasons.append(
            "some_cavity_founder_fits_did_not_converge"
        )

    candidate_diagnostic = ReversibleCandidateSearchDiagnostic(
        search_kind="adaptive_reversible_complete_panel_search",
        data_start_count=int(search.data_start_count),
        supplied_panel_start_count=int(search.supplied_panel_start_count),
        supplied_candidate_row_count=int(search.candidate_row_count),
        natural_k_ceiling=int(search.natural_k_ceiling),
        exact_score_evaluations=int(search.exact_score_evaluations),
        exact_score_cache_hits=int(search.exact_score_cache_hits),
        stop_reason=str(search.stop_reason),
        search_limited=bool(search.search_limited),
        search_limit_reasons=tuple(search.search_limit_reasons),
        local_neighbourhood_certified=bool(
            search.local_certificate
            .certified_generated_neighbourhood_local_optimum
        ),
        local_certificate_scope=str(search.local_certificate.scope),
        search_interpretation=str(search.search_interpretation),
    )
    diagnostics = CavityDiscoveryDiagnostics(
        status="selected_with_uncalibrated_reversible_cavity_pseudo_weights",
        inference_kind="full_data_adaptive_reversible_cavity",
        search_pass_count=1,
        represented_k_values=tuple(sorted(scored_modes)),
        selected_k=int(selection.map_k),
        runner_up_k=(
            None
            if selection.runner_up_k is None
            else int(selection.runner_up_k)
        ),
        selected_mode_digest=str(selection.selected_mode_digest),
        selection_method=str(selection.method),
        selection_config_type=type(settings.cavity).__name__,
        candidate_search=candidate_diagnostic,
        terminalize_precleanup_seed_modes=False,
        terminalize_proposal_seed_modes=False,
        apply_terminal_merge_repairs=False,
        min_supporters_for_confidence=(
            settings.min_supporters_for_confidence
        ),
        genotype_evidence_mode=RAW_EVIDENCE_MODE,
        genotype_evidence_interpretation=(
            "normalized_raw_read_genotype_likelihoods_at_kept_sites"
        ),
        cavity_score_calibration=CAVITY_SCORE_CALIBRATION,
        cavity_weight_calibration=CAVITY_WEIGHT_CALIBRATION,
        cavity_scores_are_calibrated=False,
        cavity_weights_are_calibrated=False,
        support_selected_from_full_data=bool(
            selection.support_selected_from_full_data
        ),
        assignments_selected_from_full_data=bool(
            selection.assignments_selected_from_full_data
        ),
        selection_leakage=bool(selection.selection_leakage),
        boundary_limited=bool(selection.boundary_limited),
        uncertainty_reasons=tuple(dict.fromkeys(uncertainty_reasons)),
    )
    return CavityBlockDiscoveryResult(
        positions=positions_value,
        reads_count_matrix=reads,
        keep_flags=flags,
        raw_genotype_likelihoods_kept=evidence_kept,
        mode_support=mode_support,
        selection=selection,
        selected_mode=search.selected_mode,
        diagnostics=diagnostics,
        # CavityBlockDiscoveryResult only needs the shared numerical and
        # materialization fields exposed directly by the reversible settings.
        config=settings,
    )


__all__ = [
    "CavityBlockDiscoveryResult",
    "CavityDiscoveryDiagnostics",
    "CavityMaterializedBlockData",
    "CavityModeSupport",
    "ReversibleCandidateSearchDiagnostic",
    "ReversibleDiscoveryError",
    "discover_block_reversible_cavity",
]
