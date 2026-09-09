"""discovery / blocks for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import asdict, dataclass
import inspect
import math
from typing import Any, Mapping, Sequence
import numpy as np
import ctypes
import gc
import os
import numba
import haplotype_reconstruction.discovery.modes as discovery_modes
import haplotype_reconstruction.discovery.search as discovery_search

RAW_EVIDENCE_MODE = "raw_likelihood"


STAGE1_BACKEND = "reversible_cavity_depth_observation_v1"


CAVITY_SCORE_CALIBRATION = (
    "uncalibrated_selection_leakage_affected_pseudo_score"
)


def _nonnegative_env_int(name, default):
    """Read a non-negative operational tuning value from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return value if value >= 0 else int(default)


CAVITY_WEIGHT_CALIBRATION = (
    "uncalibrated_selection_leakage_affected_pseudo_weight"
)


_MALLOC_TRIM_RSS_BYTES = 1024 * 1024 * _nonnegative_env_int(
    "BHD_MALLOC_TRIM_RSS_MB", 1536
)


class ReversibleDiscoveryError(RuntimeError):
    """Raised when reversible discovery cannot process a supported block."""


_MALLOC_TRIM_INTERVAL = _nonnegative_env_int(
    "BHD_MALLOC_TRIM_EVERY_BLOCKS", 32
)


def _readonly(
    value: np.ndarray,
    dtype: np.dtype[Any] | type | None = None,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


_BLOCKS_SINCE_MALLOC_TRIM = 0


def _raw_genotype_likelihoods(
    reads: np.ndarray,
    read_error_probability: float,
) -> np.ndarray:
    """Return normalized raw genotype likelihoods with no HWE prior."""

    return core_genotypes.allele_depths_to_raw_genotype_likelihoods(
        reads, read_error_probability,
        require_nonempty=True, require_integer=True,
    )


try:
    _RSS_PAGE_SIZE = int(os.sysconf("SC_PAGE_SIZE"))
except (AttributeError, OSError, ValueError):
    _RSS_PAGE_SIZE = 0


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


try:
    _libc = ctypes.CDLL("libc.so.6")

    def _trim_process_heap():
        _libc.malloc_trim(0)
except OSError:
    def _trim_process_heap():
        pass


@dataclass(frozen=True)
class CavityModeSupport:
    """Immutable wrapper around exact rich modes from one full-data search."""

    modes_by_k: tuple[tuple[int, tuple[discovery_modes.FactorizationMode, ...]], ...]

    @classmethod
    def from_mapping(
        cls,
        modes_by_k: Mapping[int, Sequence[discovery_modes.FactorizationMode]],
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
                not isinstance(mode, discovery_modes.FactorizationMode) or mode.k != k
                for mode in modes
            ):
                raise TypeError("cavity support requires rich modes at their K")

    @property
    def k_values(self) -> tuple[int, ...]:
        return tuple(k for k, _modes in self.modes_by_k)

    def as_mapping(self) -> Mapping[int, tuple[discovery_modes.FactorizationMode, ...]]:
        """Return a fresh mapping while retaining exact mode objects."""

        return dict(self.modes_by_k)

    def modes(self, k: int) -> tuple[discovery_modes.FactorizationMode, ...]:
        for represented_k, modes in self.modes_by_k:
            if represented_k == k:
                return modes
        raise KeyError(k)


def _current_process_rss_bytes():
    """Return current Linux RSS cheaply; zero when unavailable."""
    if not _RSS_PAGE_SIZE:
        return 0
    try:
        with open("/proc/self/statm", "rt", encoding="ascii") as handle:
            fields = handle.readline().split()
        return int(fields[1]) * _RSS_PAGE_SIZE
    except (OSError, ValueError, IndexError):
        return 0


@dataclass(frozen=True)
class CavityDiscoveryDiagnostics:
    """Audit summary for one full-data rich search and cavity selection."""

    status: str
    inference_kind: str
    represented_k_values: tuple[int, ...]
    selected_k: int
    runner_up_k: int | None
    selected_mode_digest: str
    selection_method: str
    selection_config_type: str
    candidate_search: ReversibleCandidateSearchDiagnostic
    observation_model: str
    min_soft_unique_sample_support: int
    min_directional_supporters: int
    min_hard_call_pseudo_probability: float
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


def _maybe_malloc_trim(*, completed_block=False):
    """Trim only for high RSS or periodically after completed blocks.

    Avoid discarding reusable arenas after every block; trim only when the
    process is large or has completed the configured number of blocks.
    """
    global _BLOCKS_SINCE_MALLOC_TRIM
    if completed_block:
        _BLOCKS_SINCE_MALLOC_TRIM += 1
    over_rss_limit = (
        _MALLOC_TRIM_RSS_BYTES > 0
        and _current_process_rss_bytes() >= _MALLOC_TRIM_RSS_BYTES
    )
    periodic = (
        completed_block
        and _MALLOC_TRIM_INTERVAL > 0
        and _BLOCKS_SINCE_MALLOC_TRIM >= _MALLOC_TRIM_INTERVAL
    )
    if over_rss_limit or periodic:
        _trim_process_heap()
        _BLOCKS_SINCE_MALLOC_TRIM = 0


@dataclass(frozen=True)
class CavityMaterializedBlockData:
    """Canonical public arrays materialized from one exact selected mode."""

    positions: np.ndarray
    haplotype_probability_arrays: tuple[np.ndarray, ...]
    reads_count_matrix: np.ndarray
    keep_flags: np.ndarray
    probs_array: np.ndarray
    discrete_haps: np.ndarray
    founder_allele_pseudo_confidence: np.ndarray
    n_directional_site_supporters: np.ndarray
    founder_alt_pseudo_probability: np.ndarray
    founder_log_pseudo_odds: np.ndarray
    sample_has_observed_kept_depth: np.ndarray
    pair_assignments: np.ndarray
    wildcard_slots: np.ndarray
    wildcard_mass: float
    uncertainty_flag: bool
    K_final: int
    selected_mode: discovery_modes.FactorizationMode
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
            "founder_allele_pseudo_confidence",
            _readonly(self.founder_allele_pseudo_confidence, np.float64),
        )
        object.__setattr__(
            self,
            "n_directional_site_supporters",
            _readonly(self.n_directional_site_supporters, np.int64),
        )
        object.__setattr__(
            self,
            "founder_alt_pseudo_probability",
            _readonly(self.founder_alt_pseudo_probability, np.float64),
        )
        object.__setattr__(
            self,
            "founder_log_pseudo_odds",
            _readonly(self.founder_log_pseudo_odds, np.float64),
        )
        object.__setattr__(
            self,
            "sample_has_observed_kept_depth",
            _readonly(self.sample_has_observed_kept_depth, np.bool_),
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
        if not isinstance(self.selected_mode, discovery_modes.FactorizationMode):
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
            "founder_allele_pseudo_confidence",
            "n_directional_site_supporters",
            "founder_alt_pseudo_probability",
            "founder_log_pseudo_odds",
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
        if (
            np.asarray(self.sample_has_observed_kept_depth).shape
            != (n_samples,)
        ):
            raise AssertionError(
                "sample_has_observed_kept_depth has the wrong shape"
            )
        resolved = np.asarray(
            self.sample_has_observed_kept_depth, dtype=np.bool_
        )
        expected_mass = float(
            np.sum(self.wildcard_slots[resolved], dtype=np.float64)
            / max(2 * int(np.sum(resolved)), 1)
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
        minimum_support = self.diagnostics.min_directional_supporters
        minimum_probability = (
            self.diagnostics.min_hard_call_pseudo_probability
        )
        pseudo_probability = self.founder_alt_pseudo_probability
        expected_called = (
            (self.n_directional_site_supporters >= minimum_support)
            & (
                np.maximum(pseudo_probability, 1.0 - pseudo_probability)
                >= minimum_probability
            )
        )
        called = self.discrete_haps >= 0
        if not np.array_equal(called, expected_called):
            raise AssertionError(
                "founder hard calls disagree with the release rule"
            )
        expected_allele = (pseudo_probability >= 0.5).astype(np.int64)
        if np.any(self.discrete_haps[called] != expected_allele[called]):
            raise AssertionError("called alleles oppose their pseudo-evidence")
        for index, value in enumerate(self.haplotype_probability_arrays):
            if np.any(value[~called[index]] != 0.5):
                raise AssertionError(
                    "unknown public cells must be (0.5, 0.5)"
                )

    def to_block_result(self, block_result_class: type | None = None) -> Any:
        """Construct the public block result without changing H or A."""

        self.validate()
        if block_result_class is None:
            block_result_class = core_haplotypes.BlockResult

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
        result.founder_allele_pseudo_confidence = (
            self.founder_allele_pseudo_confidence
        )
        result.n_directional_site_supporters = (
            self.n_directional_site_supporters
        )
        result.pair_assignments = self.pair_assignments
        result.founder_alt_pseudo_probability = (
            self.founder_alt_pseudo_probability
        )
        result.founder_log_pseudo_odds = self.founder_log_pseudo_odds
        result.sample_has_observed_kept_depth = (
            self.sample_has_observed_kept_depth
        )
        result.wildcard_slots = self.wildcard_slots
        result.wildcard_mass = self.wildcard_mass
        result.uncertainty_flag = self.uncertainty_flag
        result.K_final = self.K_final
        result.growth_history = []
        keep_mask = self.keep_flags > 0
        precleanup = np.full_like(self.discrete_haps, -1)
        precleanup[:, keep_mask] = self.discrete_haps[:, keep_mask]
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


def _init_block_worker(
    total_cores,
    active_counter,
    extra_counter=None,
    started_counter=None,
    participant_counter=None,
    batch_generation=None,
    batch_task_count=None,
    startup_target=None,
    startup_ready=None,
):
    """Initializer for worker processes — sets up dynamic numba thread
    allocation based on number of currently-active workers.

    Wires dynamic_threads' shared dynamic-thread state, which is read by
    dynamic_threads.apply_dynamic_threads() at every
    phase boundary across the discovery kernels. That lets a straggler
    block grow into cores freed as its peers finish, instead of
    being pinned for its whole run to the thread count it got at start.
    extra_counter drives the remainder distribution (total threads in use ==
    total_cores, zero idle cores); None falls back to floor-only."""
    try:
        os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
        numba.config.NUMBA_NUM_THREADS = total_cores
        numba.set_num_threads(1)
    except Exception:
        pass
    # Every discovery phase boundary re-checks the same pool-wide state.
    core_parallel.set_dynamic_thread_state(
        total_cores,
        active_counter,
        extra_counter,
        started_counter,
        participant_counter,
        batch_generation,
        batch_task_count,
        startup_target,
        startup_ready,
    )


@dataclass(frozen=True)
class CavityBlockDiscoveryResult:
    """One-search cavity selection retaining its exact rich mode payload."""

    positions: np.ndarray
    reads_count_matrix: np.ndarray
    keep_flags: np.ndarray
    raw_genotype_likelihoods_kept: np.ndarray
    mode_support: CavityModeSupport
    selection: Any
    selected_mode: discovery_modes.FactorizationMode
    diagnostics: CavityDiscoveryDiagnostics
    config: discovery_search.ReversibleCavitySearchConfig

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
        if not isinstance(self.selected_mode, discovery_modes.FactorizationMode):
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
        """Expand the exact selected mode without refitting H or A."""

        mode = self.selected_mode
        keep_mask = self.keep_flags > 0
        n_sites = len(self.positions)
        observed_kept = np.ascontiguousarray(
            np.sum(self.reads_count_matrix[:, keep_mask, :], axis=2) > 0
        )
        (
            q_kept,
            supporters_kept,
            log_pseudo_odds_kept,
            _hard_mask_kept,
            hard_values_kept,
        ) = core_haplotypes._materialize_founder_site_pseudo_evidence(
            self.raw_genotype_likelihoods_kept,
            mode.haplotypes,
            mode.assignments,
            observed_kept,
            self.config.lambda_wildcard_penalty,
            self.config.min_directional_supporters,
            self.config.min_hard_call_pseudo_probability,
        )
        q_full = np.full((mode.k, n_sites), 0.5, dtype=np.float64)
        log_pseudo_odds_full = np.zeros(
            (mode.k, n_sites), dtype=np.float64
        )
        supporters_full = np.zeros((mode.k, n_sites), dtype=np.int64)
        h_masked = np.full((mode.k, n_sites), -1, dtype=np.int64)
        q_full[:, keep_mask] = q_kept
        log_pseudo_odds_full[:, keep_mask] = log_pseudo_odds_kept
        supporters_full[:, keep_mask] = supporters_kept
        h_masked[:, keep_mask] = hard_values_kept
        confidence_full = np.maximum(q_full, 1.0 - q_full)
        public = {}
        for founder in range(mode.k):
            values = np.full((n_sites, 2), 0.5, dtype=np.float64)
            called = h_masked[founder] >= 0
            values[called, 0] = 1.0 - q_full[founder, called]
            values[called, 1] = q_full[founder, called]
            public[founder] = values
        sample_has_observed_kept_depth = np.ascontiguousarray(
            np.any(observed_kept, axis=1)
        )

        full_probabilities = _raw_genotype_likelihoods(
            self.reads_count_matrix, self.config.read_error_probability
        )
        wildcard_mass = float(
            np.sum(
                mode.wildcard_slots[sample_has_observed_kept_depth],
                dtype=np.float64,
            )
            / max(2 * int(np.sum(sample_has_observed_kept_depth)), 1)
        )
        reasons = list(self.diagnostics.uncertainty_reasons)
        reasons.append(
            "founder_allele_confidence_is_fixed_assignment_capped_"
            "pseudo_probability"
        )
        unresolved_count = int(np.sum(~sample_has_observed_kept_depth))
        if unresolved_count:
            reasons.append("samples_without_kept_site_depth_are_unresolved")
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
            founder_allele_pseudo_confidence=confidence_full,
            n_directional_site_supporters=supporters_full,
            founder_alt_pseudo_probability=q_full,
            founder_log_pseudo_odds=log_pseudo_odds_full,
            sample_has_observed_kept_depth=sample_has_observed_kept_depth,
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

    def to_block_result(
        self,
        *,
        block_result_class: type | None = None,
    ) -> Any:
        return self.materialize().to_block_result(block_result_class)


def _validate_block_parallelism(
    num_processes: int,
    total_numba_threads: int | None,
) -> tuple[int, int]:
    if (
        isinstance(num_processes, bool)
        or int(num_processes) != num_processes
        or int(num_processes) < 1
    ):
        raise ValueError("num_processes must be a positive integer")
    processes = int(num_processes)
    total_threads = (
        processes
        if total_numba_threads is None
        else total_numba_threads
    )
    if (
        isinstance(total_threads, bool)
        or int(total_threads) != total_threads
        or int(total_threads) < processes
    ):
        raise ValueError(
            "total_numba_threads must be an integer at least as large as "
            "num_processes"
        )
    total_threads = int(total_threads)
    available_cpus = len(os.sched_getaffinity(0))
    if processes > available_cpus or total_threads > available_cpus:
        raise ValueError(
            "block workers and Numba thread budget must lie within the "
            "current CPU affinity"
        )
    return processes, total_threads


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


class BlockDiscoveryPool:
    """Reusable block-worker pool with one bounded dynamic thread budget."""

    def __init__(
        self,
        num_processes: int,
        total_numba_threads: int | None = None,
    ) -> None:
        (
            self.num_processes,
            self.total_numba_threads,
        ) = _validate_block_parallelism(
            num_processes, total_numba_threads
        )
        self._closed = False
        self._active_counter = core_parallel.forkserver_context.Value("i", 0)
        self._extra_counter = core_parallel.forkserver_context.Value("i", 0)
        self._started_counter = core_parallel.forkserver_context.Value("i", 0)
        self._participant_counter = core_parallel.forkserver_context.Value("i", 0)
        self._batch_generation = core_parallel.forkserver_context.Value("i", 0)
        self._batch_task_count = core_parallel.forkserver_context.Value("i", 0)
        self._startup_target = core_parallel.forkserver_context.Value("i", 1)
        self._startup_ready = core_parallel.forkserver_context.Value("i", 0)
        self._batch_in_progress = False

        # Prevent forkserver workers from re-executing a pipeline entry point.
        with core_parallel.main_module_guard():
            self._pool = core_parallel.ForkserverPool(
                processes=self.num_processes,
                initializer=_init_block_worker,
                initargs=(
                    self.total_numba_threads,
                    self._active_counter,
                    self._extra_counter,
                    self._started_counter,
                    self._participant_counter,
                    self._batch_generation,
                    self._batch_task_count,
                    self._startup_target,
                    self._startup_ready,
                ),
            )

    @staticmethod
    def _store_counter(counter, value):
        with counter.get_lock():
            counter.get_obj().value = int(value)

    def _prepare_task_batch(self, n_tasks):
        if self._batch_in_progress:
            raise RuntimeError(
                "block-discovery pool already has an unfinished task batch"
            )
        if self._active_counter.value != 0:
            raise RuntimeError(
                "block-discovery workers from the preceding batch are still active"
            )
        task_count = int(n_tasks)
        initial_target = min(task_count, self.num_processes)
        self._store_counter(self._started_counter, 0)
        self._store_counter(self._participant_counter, 0)
        self._store_counter(self._batch_task_count, task_count)
        self._store_counter(self._startup_target, initial_target)
        self._store_counter(self._startup_ready, initial_target <= 1)
        with self._batch_generation.get_lock():
            generation = self._batch_generation.get_obj()
            generation.value += 1

    def imap_unordered(self, tasks):
        if self._closed:
            raise RuntimeError("block-discovery pool is closed")
        if not hasattr(tasks, "__len__"):
            tasks = tuple(tasks)
        self._prepare_task_batch(len(tasks))
        try:
            raw_results = self._pool.imap_unordered(
                _worker_generate_block_direct, tasks, chunksize=1
            )
        except Exception:
            raise
        self._batch_in_progress = True

        def consume_batch():
            completed = False
            try:
                for result in raw_results:
                    yield result
                completed = True
            finally:
                # An abandoned or failed iterator may still have queued work.
                # Keep the pool guarded in that case; close/terminate remains
                # available, but a new batch cannot corrupt the startup gate.
                if completed:
                    self._batch_in_progress = False

        return consume_batch()

    def close(self) -> None:
        if self._closed:
            return
        self._pool.close()
        self._pool.join()
        self._closed = True

    def terminate(self) -> None:
        if self._closed:
            return
        self._pool.terminate()
        self._pool.join()
        self._closed = True

    def __enter__(self) -> "BlockDiscoveryPool":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()


def discover_block_reversible_cavity(
    positions: np.ndarray,
    reads_array: np.ndarray,
    keep_flags: np.ndarray | None = None,
    *,
    config: discovery_search.ReversibleCavitySearchConfig | None = None,
) -> CavityBlockDiscoveryResult:
    """Discover and select one block panel by adaptive reversible search.

    The only K ceiling is the finite identifiable state-space ceiling computed
    by :func:`search_reversible_cavity`.  Operational proposal, scoring, and
    expansion budgets are reported as search limitations rather than treated
    as biological bounds on founder count.
    """

    settings = discovery_search.ReversibleCavitySearchConfig() if config is None else config
    if not isinstance(settings, discovery_search.ReversibleCavitySearchConfig):
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

    search = discovery_search.search_reversible_cavity(
        evidence_kept,
        allele_depths=depths_kept,
        config=settings,
    )
    selection = discovery_search.as_cavity_selection(search)
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
        observation_model="positive_allele_depth_is_observed",
        min_soft_unique_sample_support=(
            settings.min_soft_unique_sample_support
        ),
        min_directional_supporters=settings.min_directional_supporters,
        min_hard_call_pseudo_probability=(
            settings.min_hard_call_pseudo_probability
        ),
        genotype_evidence_mode=RAW_EVIDENCE_MODE,
        genotype_evidence_interpretation=(
            "normalized_raw_read_genotype_likelihoods_at_kept_sites; "
            "founder hard-call confidence is an uncalibrated capped "
            "fixed-assignment pseudo-probability"
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


def _block_has_informative_retained_data(positions, reads, keep_flags):
    """Return whether a block can support founder discovery."""

    positions_value = np.asarray(positions)
    reads_value = np.asarray(reads)
    if reads_value.ndim != 3 or reads_value.shape[2] != 2:
        raise ValueError("reads must have shape (samples, sites, 2)")
    if positions_value.shape != (reads_value.shape[1],):
        raise ValueError("positions must match the reads site dimension")
    if reads_value.shape[0] < 1:
        raise ValueError("reads must contain at least one sample")
    if len(positions_value) == 0:
        return False
    retained = (
        np.ones(len(positions_value), dtype=np.bool_)
        if keep_flags is None
        else np.asarray(keep_flags) > 0
    )
    if retained.shape != (len(positions_value),):
        raise ValueError("keep_flags must match positions")
    if not np.any(retained):
        return False
    return bool(np.any(reads_value[:, retained, :] > 0))


def _worker_generate_block_direct(args):
    """Discover one informative block and return ``(input_index, result)``."""

    core_parallel.increment_active()
    core_parallel.apply_dynamic_threads()
    try:
        (
            block_index,
            positions,
            reads,
            keep_flags,
            discovery_config,
            discard_reads_after,
        ) = args
        if not _block_has_informative_retained_data(
            positions, reads, keep_flags
        ):
            return block_index, None


        discovery = discover_block_reversible_cavity(
            positions,
            reads,
            keep_flags=keep_flags,
            config=discovery_config,
        )
        result = discovery.to_block_result(block_result_class=core_haplotypes.BlockResult)
        if discard_reads_after:
            result.reads_count_matrix = None
        _maybe_malloc_trim(completed_block=True)
        return block_index, result
    finally:
        core_parallel.release_dynamic_extra()
        core_parallel.decrement_active()


def generate_all_block_haplotypes(
    genomic_data,
    *,
    num_processes=16,
    discard_reads_after=True,
    total_numba_threads=None,
    block_pool=None,
    discovery_config=None,
):
    """Discover founder haplotypes in every informative input block.

    Empty blocks, blocks with no retained sites, and blocks with no reads at
    retained sites are omitted. ``discovery_config`` defaults to the canonical
    calibrated :class:`ReversibleCavitySearchConfig`.
    """

    from tqdm import tqdm

    config = (
        discovery_search.ReversibleCavitySearchConfig()
        if discovery_config is None
        else discovery_config
    )
    if not isinstance(config, discovery_search.ReversibleCavitySearchConfig):
        raise TypeError(
            "discovery_config must be a ReversibleCavitySearchConfig"
        )
    num_processes, total_numba_threads = _validate_block_parallelism(
        num_processes, total_numba_threads
    )
    if block_pool is not None:
        if not isinstance(block_pool, BlockDiscoveryPool):
            raise TypeError("block_pool must be a BlockDiscoveryPool")
        if (
            block_pool.num_processes != num_processes
            or block_pool.total_numba_threads != total_numba_threads
        ):
            raise ValueError(
                "block_pool worker and thread budgets must match this call"
            )

    tasks = []
    for index in range(len(genomic_data)):
        positions, reads, keep_flags = genomic_data[index]
        tasks.append((
            index,
            positions,
            reads,
            keep_flags,
            config,
            bool(discard_reads_after),
        ))

    def collect(pool):
        return list(tqdm(
            pool.imap_unordered(tasks),
            total=len(tasks),
            desc="Block haplotypes",
        ))

    if block_pool is None:
        with BlockDiscoveryPool(
            num_processes, total_numba_threads
        ) as local_pool:
            indexed_results = collect(local_pool)
    else:
        indexed_results = collect(block_pool)

    indexed_results.sort(key=lambda item: item[0])
    results = [
        result for _index, result in indexed_results if result is not None
    ]
    if discard_reads_after:
        gc.collect()
    return core_haplotypes.BlockResults(results)

import haplotype_reconstruction.core.genotypes as core_genotypes
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.parallel as core_parallel
