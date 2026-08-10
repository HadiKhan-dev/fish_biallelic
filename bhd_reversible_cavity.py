"""Cap-free reversible search for regularized cavity-selected panels.

The controller in this module searches *complete* :class:`FactorizationMode`
objects.  It deliberately does not enumerate ``K=1..Kmax`` and does not call
the recovery-mixture K sweep.  Instead, it starts from several data-derived
factorisations and follows deterministic local moves to ``K-1``, ``K``,
``K+1`` and ``K+2``.  Downward moves remain available after every upward
move, so an early growth decision is not irreversible.

Candidate haplotypes are anonymous proposal rows.  They may seed a K=1 fit
or be appended to an existing complete panel, but their count is never
interpreted as K.  The only K ceiling is the identifiable finite state space,
``min(2 * n_samples, 2**n_sites)``.  There is intentionally no public K cap.

At each represented K, the deterministic minimum-full-data-NLL panel is the
only representative compared scientifically.  That comparison uses the full
mean-field fixed-assignment cavity score with the telescoping K prior and
unordered binary-haplotype-set code.  Scores are cached by canonical hard
panel.  As in :mod:`bhd_cavity_selection`, they are selection-leakage-affected,
uncalibrated pseudo-scores rather than calibrated posterior probabilities.

Finite proposal, score and expansion budgets are operational controls.  If
one binds, the result says ``search_limited=True``; it never reports that
event as a scientific K boundary.  The local certificate is correspondingly
limited to the generated reversible neighbourhood and never claims a global
optimum over binary panels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from bhd_cavity_selection import (
    CavityKDiagnostic,
    CavityModeDiagnostic,
    CavitySelection,
    _prepare_cavity_scoring_workspace,
    _normalized_log_weights,
    CavitySelectionConfig,
    HybridCavitySelectionConfig,
    select_cavity_predictive_k,
)
from bhd_factorization_modes import (
    FactorizationMode,
    FixedKPanelFitConfig,
    _canonicalize_mode,
    _death_starts,
    _fit_starts_with_synchronized_endpoints,
)
from bhd_mode_canonicalization import exact_unique_binary_rows


_SCORE_TOLERANCE = 1e-9
_SUPPORTED_MOVE_OFFSETS = (-1, 0, 1, 2)


@dataclass(frozen=True)
class _GrowthInputs:
    oracle_nll: np.ndarray
    decisiveness: np.ndarray
    dosage_by_sample: np.ndarray
    seed_haplotypes_by_sample: np.ndarray



@dataclass(frozen=True)
class ReversibleCavitySearchConfig:
    """Numerical settings and finite operational search budgets.

    None of these fields is a scientific upper bound on K.  ``beam_width``
    controls how many deterministic low-NLL basins at the currently selected
    K may be expanded.  Basin companions are evaluated only after the
    minimum-NLL representative's ordinary route stalls.
    """

    beam_width: int = 3
    max_expansions: int = 32
    max_exact_scores: int = 96
    max_proposals_per_expansion: int = 128
    data_start_beam_width: int = 6
    n_data_seed_modes: int = 8
    max_candidate_start_rows: int = 24
    max_replacement_children_per_mode: int = 24
    lambda_wildcard_penalty: float = 0.5
    read_error_probability: float = 0.02
    min_supporters_for_confidence: int = 2
    coordinate_descent_max_iter: int = 50
    soft_seed_min_cluster_size: int = 2
    apply_gauge_rewire: bool = True
    exact_cut_max_k: int = 12
    max_cut_ties: int = 4
    score_tolerance: float = _SCORE_TOLERANCE
    cavity: HybridCavitySelectionConfig = HybridCavitySelectionConfig()

    def __post_init__(self) -> None:
        for name in (
            "beam_width",
            "max_expansions",
            "max_exact_scores",
            "max_proposals_per_expansion",
            "data_start_beam_width",
            "n_data_seed_modes",
            "max_candidate_start_rows",
            "max_replacement_children_per_mode",
            "min_supporters_for_confidence",
            "coordinate_descent_max_iter",
            "soft_seed_min_cluster_size",
            "exact_cut_max_k",
            "max_cut_ties",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not math.isfinite(self.lambda_wildcard_penalty)
            or self.lambda_wildcard_penalty < 0.0
        ):
            raise ValueError(
                "lambda_wildcard_penalty must be finite and non-negative"
            )
        if not 0.0 < self.read_error_probability < 0.5:
            raise ValueError(
                "read_error_probability must lie strictly in (0, 0.5)"
            )
        if not math.isfinite(self.score_tolerance) or self.score_tolerance < 0:
            raise ValueError("score_tolerance must be finite and non-negative")
        if not isinstance(self.apply_gauge_rewire, bool):
            raise TypeError("apply_gauge_rewire must be boolean")
        if not isinstance(self.cavity, HybridCavitySelectionConfig):
            raise TypeError("cavity must be a HybridCavitySelectionConfig")


@dataclass(frozen=True)
class AnchoredModeScore:
    """Cheap regularized screen score for one canonical complete panel."""

    mode: FactorizationMode
    mode_digest: str
    cavity_log_predictive: float
    log_k_prior: float
    log_haplotype_set_prior: float
    log_score: float
    diagnostic: CavityModeDiagnostic


@dataclass(frozen=True)
class ReversibleModeScore:
    """Cached full mean-field regularized score for one visited panel."""

    mode: FactorizationMode
    mode_digest: str
    anchored_cavity_log_predictive: float
    anchored_log_score: float
    cavity_log_predictive: float
    log_k_prior: float
    log_haplotype_set_prior: float
    log_score: float
    mean_field_diagnostic: CavityModeDiagnostic
    anchored_diagnostic: CavityModeDiagnostic

    @property
    def k(self) -> int:
        return self.mode.k


@dataclass(frozen=True)
class MoveScore:
    """Best exact-scored generated neighbour at one relative K."""

    offset: int
    k: int
    mode_digest: str
    log_score: float


@dataclass(frozen=True)
class SearchStepDiagnostic:
    """One best-first expansion and its operational accounting."""

    expansion: int
    round_index: int
    expanded_k: int
    expanded_mode_digest: str
    expanded_log_score: float
    proposed_by_offset: tuple[tuple[int, int], ...]
    proposals_after_budget: int
    proposal_budget_omissions: int
    anchored_scores_computed: int
    exact_cache_hits: int
    exact_scores_computed: int
    anchored_screen_omissions: int
    exact_budget_omissions: int
    improving_exact_neighbours: int
    incumbent_k: int
    incumbent_mode_digest: str
    incumbent_log_score: float
    proposal_generation_complete: bool
    all_generated_neighbours_exact_scored: bool
    best_exact_neighbours: tuple[MoveScore, ...]
    limitation_reasons: tuple[str, ...]


@dataclass(frozen=True)
class LocalSearchCertificate:
    """Certificate over the selected panel's generated local neighbourhood."""

    incumbent_k: int
    incumbent_mode_digest: str
    incumbent_log_score: float
    incumbent_was_expanded: bool
    generated_move_offsets: tuple[int, ...]
    exact_scored_move_offsets: tuple[int, ...]
    best_exact_neighbours: tuple[MoveScore, ...]
    proposal_generation_complete: bool
    all_generated_neighbours_exact_scored: bool
    no_improving_generated_neighbour: bool | None
    certified_generated_neighbourhood_local_optimum: bool
    limitation_reasons: tuple[str, ...]
    scope: str


@dataclass(frozen=True)
class ReversibleCavitySearchResult:
    """Visited support and selected complete panel from reversible search."""

    selected: ReversibleModeScore
    runner_up: ReversibleModeScore | None
    visited_scores: tuple[ReversibleModeScore, ...]
    anchored_scores: tuple[AnchoredModeScore, ...]
    visited_modes_by_k: tuple[tuple[int, tuple[FactorizationMode, ...]], ...]
    best_score_by_k: tuple[tuple[int, float], ...]
    pseudo_probability_by_k: tuple[tuple[int, float], ...]
    search_steps: tuple[SearchStepDiagnostic, ...]
    local_certificate: LocalSearchCertificate
    natural_k_ceiling: int
    search_limited: bool
    search_limit_reasons: tuple[str, ...]
    boundary_limited: bool
    stop_reason: str
    data_start_count: int
    overcomplete_data_start_count: int
    supplied_panel_start_count: int
    candidate_row_count: int
    exact_score_evaluations: int
    exact_score_cache_hits: int
    anchored_score_evaluations: int
    high_k_tail_mass_upper_bound: float
    high_k_tail_log_mass_upper_bound: float
    high_k_tail_pseudo_probability_upper_bound: float
    high_k_tail_bound_interpretation: str
    objective_interpretation: str
    search_interpretation: str

    @property
    def selected_mode(self) -> FactorizationMode:
        return self.selected.mode

    @property
    def selected_k(self) -> int:
        return self.selected.k

    def modes(self, k: int) -> tuple[FactorizationMode, ...]:
        for represented_k, modes in self.visited_modes_by_k:
            if represented_k == k:
                return modes
        raise KeyError(k)


@dataclass(frozen=True)
class _StageModeScore:
    mode: FactorizationMode
    digest: str
    cavity_log_predictive: float
    log_k_prior: float
    log_haplotype_set_prior: float
    log_score: float
    diagnostic: CavityModeDiagnostic


def _natural_k_ceiling(n_samples: int, n_sites: int) -> int:
    sample_ceiling = 2 * int(n_samples)
    if n_sites < sample_ceiling.bit_length():
        return min(sample_ceiling, 1 << int(n_sites))
    return sample_ceiling


def _canonical_candidate_rows(
    candidate_haplotypes: np.ndarray | Sequence[np.ndarray] | None,
    n_sites: int,
) -> tuple[np.ndarray, ...]:
    if candidate_haplotypes is None:
        return ()
    matrix = np.asarray(candidate_haplotypes)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != n_sites:
        raise ValueError(
            "candidate_haplotypes must have shape (candidates, sites)"
        )
    if np.any((matrix != 0) & (matrix != 1)):
        raise ValueError("candidate haplotypes must be hard binary rows")
    unique_rows = exact_unique_binary_rows(
        np.asarray(matrix, dtype=np.int64)
    )
    return tuple(np.ascontiguousarray(row) for row in unique_rows)


def _stage_config(
    settings: HybridCavitySelectionConfig,
    founder_inference: str,
) -> CavitySelectionConfig:
    """Construct the uncapped regularized stage used by this search."""

    return CavitySelectionConfig(
        state_concentration=settings.state_concentration,
        wildcard_prior_mass=settings.wildcard_prior_mass,
        double_wildcard_fraction=settings.double_wildcard_fraction,
        mean_field_max_iter=settings.mean_field_max_iter,
        mean_field_tolerance=settings.mean_field_tolerance,
        likelihood_floor=settings.likelihood_floor,
        max_modes_per_k=None,
        founder_inference=founder_inference,
        apply_unordered_haplotype_set_code=True,
    )


def _score_stage(
    evidence: np.ndarray,
    modes: Sequence[FactorizationMode],
    config: CavitySelectionConfig,
    cavity_workspace: Any | None = None,
) -> tuple[_StageModeScore, ...]:
    # Refresh the shared block-pool allocation immediately before the
    # parallel held-out cavity kernel, so late stragglers can claim cores
    # released since their preceding proposal fit.
    import dynamic_threads
    dynamic_threads.apply_dynamic_threads()
    if not modes:
        return ()
    grouped: dict[int, list[FactorizationMode]] = {}
    for mode in modes:
        grouped.setdefault(mode.k, []).append(mode)
    selection = select_cavity_predictive_k(
        evidence, grouped, config=config, _workspace=cavity_workspace
    )
    diagnostics = {
        (item.k, item.mode_digest): item
        for item in selection.mode_diagnostics
    }
    k_components = {
        item.k: (item.log_k_prior, item.log_haplotype_set_prior)
        for item in selection.k_diagnostics
    }
    scored: list[_StageModeScore] = []
    for mode in modes:
        digest = _canonicalize_mode(mode, mode.k).digest
        diagnostic = diagnostics[(mode.k, digest)]
        log_k_prior, log_set_prior = k_components[mode.k]
        scored.append(_StageModeScore(
            mode=mode,
            digest=digest,
            cavity_log_predictive=diagnostic.cavity_log_predictive,
            log_k_prior=log_k_prior,
            log_haplotype_set_prior=log_set_prior,
            log_score=(
                diagnostic.cavity_log_predictive
                + log_k_prior
                + log_set_prior
            ),
            diagnostic=diagnostic,
        ))
    return tuple(scored)


def _mode_order(mode: FactorizationMode) -> tuple[float, int, bytes]:
    return (mode.total_nll, mode.k, mode.canonical_key)


def _deduplicate(modes: Sequence[FactorizationMode]) -> tuple[FactorizationMode, ...]:
    from bhd_factorization_modes import _deduplicate_modes

    return _deduplicate_modes(modes)


def _internal_move_config(
    config: ReversibleCavitySearchConfig,
) -> FixedKPanelFitConfig:
    return FixedKPanelFitConfig(
        lambda_wildcard_penalty=config.lambda_wildcard_penalty,
        coordinate_descent_max_iter=config.coordinate_descent_max_iter,
    )


def _prepare_growth_inputs(evidence: np.ndarray) -> _GrowthInputs:
    """Prepare the evidence-only K-growth quantities once per search."""

    max_genotype_probability = np.max(evidence, axis=2)
    oracle_nll = -np.sum(
        np.log(np.maximum(max_genotype_probability, np.finfo(np.float64).tiny)),
        axis=1,
    )
    decisiveness = np.sum(max_genotype_probability, axis=1)
    dosage_by_sample = np.argmax(evidence, axis=2)
    population_alt_frequency = (
        evidence[..., 1].mean(axis=0) * 0.5
        + evidence[..., 2].mean(axis=0)
    )
    seed_haplotypes_by_sample = np.zeros_like(dosage_by_sample)
    seed_haplotypes_by_sample[dosage_by_sample == 2] = 1
    heterozygous = dosage_by_sample == 1
    seed_haplotypes_by_sample[
        heterozygous & (population_alt_frequency[None, :] > 0.5)
    ] = 1
    for value in (
        oracle_nll, decisiveness, dosage_by_sample, seed_haplotypes_by_sample
    ):
        value.setflags(write=False)
    return _GrowthInputs(
        oracle_nll=oracle_nll,
        decisiveness=decisiveness,
        dosage_by_sample=dosage_by_sample,
        seed_haplotypes_by_sample=seed_haplotypes_by_sample,
    )


def _fit_panel_starts(
    evidence: np.ndarray,
    starts: Sequence[np.ndarray],
    move_config: FixedKPanelFitConfig,
    workspace: Any,
) -> tuple[FactorizationMode, ...]:
    if not starts:
        return ()
    raw, refitted = _fit_starts_with_synchronized_endpoints(
        evidence, starts, move_config, workspace
    )
    return _deduplicate((*raw, *refitted))


def _ordered_binary_candidate_rows(
    candidates: np.ndarray, evidence: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Harden soft candidates without discarding their support-ranked order."""

    soft = np.asarray(candidates, dtype=np.float64)
    if soft.ndim != 2 or soft.shape[1] != evidence.shape[1]:
        raise ValueError("soft residual candidates must match evidence sites")
    marginal = np.mean(
        0.5 * evidence[:, :, 1] + evidence[:, :, 2], axis=0
    )
    binary = soft > 0.5
    ties = np.isclose(soft, 0.5, rtol=0.0, atol=1e-12)
    binary[ties] = np.broadcast_to(marginal >= 0.5, binary.shape)[ties]
    unique: dict[bytes, np.ndarray] = {}
    for row in np.asarray(binary, dtype=np.int64):
        contiguous = np.ascontiguousarray(row, dtype=np.int64)
        unique.setdefault(
            contiguous.astype(np.int8, copy=False).tobytes(), contiguous
        )
    return tuple(unique.values())


def _birth_delete_replacements(
    parent: FactorizationMode,
    births: Sequence[FactorizationMode],
    evidence: np.ndarray,
    settings: ReversibleCavitySearchConfig,
    move_config: Any,
    workspace: Any,
) -> tuple[tuple[FactorizationMode, ...], int]:
    """Fit same-K panels obtained by adding a row and deleting any row.

    These moves make a birth reversible even when the useful proposal must
    replace, rather than merely augment, an incumbent row.  They are called
    only after a direct upward move fails to improve the regularized score.
    """

    row_width = parent.n_sites
    descriptors = []
    seen: set[bytes] = set()
    for birth in sorted(_deduplicate(births), key=_mode_order):
        if birth.k != parent.k + 1:
            continue
        for deleted in range(birth.k):
            offset = deleted * row_width
            key = (
                birth.canonical_key[:offset]
                + birth.canonical_key[offset + row_width:]
            )
            if key in seen:
                continue
            seen.add(key)
            descriptors.append((birth, deleted))
    limit = settings.max_replacement_children_per_mode
    omitted = max(0, len(descriptors) - limit)
    starts = [
        np.ascontiguousarray(
            np.delete(birth.haplotypes, deleted, axis=0), dtype=np.int64
        )
        for birth, deleted in descriptors[:limit]
    ]
    fitted = _fit_panel_starts(evidence, starts, move_config, workspace)
    return tuple(mode for mode in fitted if mode.k == parent.k), omitted


def _v2_soft_births(
    mode: FactorizationMode,
    evidence: np.ndarray,
    reads: np.ndarray,
    ceiling: int,
    settings: ReversibleCavitySearchConfig,
    move_config: Any,
    workspace: Any,
    residual_input_workspace: Any,
    soft_cache: dict[tuple[bytes, bytes], Any] | None = None,
) -> tuple[tuple[FactorizationMode, ...], int, int, tuple[str, ...]]:
    """Batch-fit proposal-D/base-plus-soft residual rows for one mode."""

    if mode.k >= ceiling:
        return (), 0, 0, ()
    cache_key = (mode.canonical_key, mode.assignments.tobytes())
    if soft_cache is not None:
        cached = soft_cache.get(cache_key)
        if cached is not None:
            return cached

    def finish(value):
        if soft_cache is not None:
            soft_cache[cache_key] = value
        return value

    from types import SimpleNamespace
    from bhd_candidate_pool import augment_assigned_residual_candidates

    adapter = SimpleNamespace(
        discrete_haps=mode.haplotypes,
        pair_assignments=mode.assignments,
        K_final=mode.k,
        keep_flags=np.ones(mode.n_sites, dtype=np.int8),
        precleanup_candidate_discrete_haps=mode.haplotypes,
        precleanup_candidate_k=mode.k,
        haplotypes={},
    )
    augmented = augment_assigned_residual_candidates(
        adapter,
        reads,
        base_candidates=mode.haplotypes.astype(float),
        read_error_probability=settings.read_error_probability,
        proposal_mode="D",
        include_assigned_hard_candidates=False,
        compute_excluded_assigned_hard_diagnostics=False,
        minimum_soft_unique_sample_support=(
            settings.min_supporters_for_confidence
        ),
        residual_input_workspace=residual_input_workspace,
        binary_panel_fast_path=True,
    )
    soft = np.asarray(
        augmented.candidates[augmented.n_base_candidates:], dtype=np.float64
    )
    if len(soft) == 0:
        return finish(((), 0, 0, ()))
    for index in range(augmented.n_base_candidates, len(augmented.candidates)):
        provenance = augmented.candidate_provenance[index]
        diagnostic_index = provenance.proposal_diagnostic_index
        if diagnostic_index is None:
            raise AssertionError("residual proposal lacks diagnostic provenance")
        diagnostic = augmented.proposal_diagnostics[diagnostic_index]
        if diagnostic.proposal_mode == "assigned_hard":
            raise AssertionError("base_plus_soft retained a hard proposal")

    rows = _ordered_binary_candidate_rows(soft, evidence)
    existing = {
        np.asarray(row, dtype=np.int8).tobytes() for row in mode.haplotypes
    }
    starts = [
        np.vstack((mode.haplotypes, row[None, :]))
        for row in rows
        if np.asarray(row, dtype=np.int8).tobytes() not in existing
    ]
    emitted = len(starts)
    omitted = max(0, emitted - settings.max_proposals_per_expansion)
    retained = starts[:settings.max_proposals_per_expansion]
    fitted = _fit_panel_starts(evidence, retained, move_config, workspace)
    limits = ("soft_residual_proposal_budget",) if omitted else ()
    return finish((
        tuple(child for child in fitted if child.k == mode.k + 1),
        emitted,
        omitted,
        limits,
    ))


def _v2_ordinary_moves(
    mode: FactorizationMode,
    evidence: np.ndarray,
    candidate_rows: Sequence[np.ndarray],
    ceiling: int,
    settings: ReversibleCavitySearchConfig,
    move_config: Any,
    workspace: Any,
    growth_inputs: _GrowthInputs,
) -> tuple[
    tuple[FactorizationMode, ...],
    tuple[FactorizationMode, ...],
    tuple[tuple[int, int], ...],
    int,
    tuple[str, ...],
]:
    """Generate synchronized/refitted K-1, K and K+1 ordinary moves."""

    from bhd_factorization_modes import (
        _expand_one_complete_mode,
        _propose_bipartite_gauge_starts,
        assignment_graph,
        maximum_cut_partitions,
    )

    buckets: dict[int, list[np.ndarray]] = {-1: [], 0: [], 1: []}
    limitations: list[str] = []
    if mode.k > 1:
        buckets[-1].extend(_death_starts(mode))
    if settings.apply_gauge_rewire and mode.k > 1:
        weights = assignment_graph(mode)
        partitions = maximum_cut_partitions(
            weights,
            exact_max_k=settings.exact_cut_max_k,
            max_ties=settings.max_cut_ties + 1,
        )
        if len(partitions) > settings.max_cut_ties:
            limitations.append("gauge_max_cut_ties_truncated")
        if mode.k > settings.exact_cut_max_k:
            limitations.append("gauge_cut_heuristic")
        buckets[0].extend(
            proposal.haplotypes
            for proposal in _propose_bipartite_gauge_starts(
                mode,
                evidence,
                exact_cut_max_k=settings.exact_cut_max_k,
                max_cut_ties=settings.max_cut_ties,
                assignment_weights=weights,
                cut_partitions=partitions,
            )
        )
    if mode.k < ceiling:
        existing = {
            np.asarray(row, dtype=np.int8).tobytes()
            for row in mode.haplotypes
        }
        buckets[1].extend(
            np.vstack((mode.haplotypes, row[None, :]))
            for row in candidate_rows
            if np.asarray(row, dtype=np.int8).tobytes() not in existing
        )

    unique: dict[int, list[np.ndarray]] = {}
    for offset, starts in buckets.items():
        by_key: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
        for start in starts:
            value = np.ascontiguousarray(start, dtype=np.int64)
            key = (tuple(int(item) for item in value.shape), value.tobytes())
            by_key.setdefault(key, value)
        unique[offset] = [by_key[key] for key in sorted(by_key)]
    total = sum(len(starts) for starts in unique.values())
    retained: list[np.ndarray] = []
    index = 0
    while len(retained) < min(total, settings.max_proposals_per_expansion):
        added = False
        for offset in (-1, 0, 1):
            if index < len(unique[offset]):
                retained.append(unique[offset][index])
                added = True
                if len(retained) == settings.max_proposals_per_expansion:
                    break
        if not added:
            break
        index += 1
    omitted = total - len(retained)
    if omitted:
        limitations.append("proposal_budget")

    modes = list(_fit_panel_starts(evidence, retained, move_config, workspace))
    if mode.k < ceiling:
        data_births, _attempts, _fallbacks = _expand_one_complete_mode(
            mode,
            evidence,
            settings.lambda_wildcard_penalty,
            settings.coordinate_descent_max_iter,
            fit_workspace=workspace,
            oracle_nll=growth_inputs.oracle_nll,
            decisiveness=growth_inputs.decisiveness,
            dosage_by_sample=growth_inputs.dosage_by_sample,
            seed_haplotypes_by_sample=growth_inputs.seed_haplotypes_by_sample,
        )
        modes.extend(child for child in data_births if child.k == mode.k + 1)
    modes = list(_deduplicate(modes))
    births = tuple(child for child in modes if child.k == mode.k + 1)
    counts = tuple(
        (
            offset,
            sum(child.k - mode.k == offset for child in modes),
        )
        for offset in _SUPPORTED_MOVE_OFFSETS
    )
    return (
        tuple(modes),
        births,
        counts,
        omitted,
        tuple(dict.fromkeys(limitations)),
    )


def _score_sort_key(score: ReversibleModeScore) -> tuple[float, int, bytes]:
    return (-score.log_score, score.k, score.mode.canonical_key)




def search_reversible_cavity(
    evidence: np.ndarray,
    seed_haplotypes: Sequence[np.ndarray | FactorizationMode] = (),
    *,
    candidate_haplotypes: np.ndarray | Sequence[np.ndarray] | None = None,
    allele_depths: np.ndarray,
    config: ReversibleCavitySearchConfig | None = None,
) -> ReversibleCavitySearchResult:
    """Run adaptive reversible search without an enumerated K grid.

    Full mean-field cavity scoring is applied only to the deterministic
    minimum-full-data-NLL representative at each represented K.  Up to
    ``beam_width`` low-NLL basins at the selected K are nevertheless expanded:
    the representative first, companions only after its ordinary route stalls,
    and residual/paired rescue only after every ordinary beam route stalls.
    """

    settings = ReversibleCavitySearchConfig() if config is None else config
    from bhd_factorization_modes import (
        _expand_one_complete_mode,
        _initial_complete_modes,
        _validate_evidence,
    )
    from bhd_candidate_pool import prepare_residual_inputs
    from bhd_fit import _prepare_fixed_k_fit_workspace

    likelihood = _validate_evidence(evidence)
    n_samples, n_sites, _ = likelihood.shape
    reads = np.asarray(allele_depths)
    if reads.shape != (n_samples, n_sites, 2):
        raise ValueError("allele_depths must match evidence samples/sites")
    if np.any(~np.isfinite(reads)) or np.any(reads < 0):
        raise ValueError("allele_depths must be finite and non-negative")

    ceiling = _natural_k_ceiling(n_samples, n_sites)
    candidate_rows = _canonical_candidate_rows(candidate_haplotypes, n_sites)
    move_config = _internal_move_config(settings)
    workspace = _prepare_fixed_k_fit_workspace(
        likelihood, settings.lambda_wildcard_penalty
    )
    growth_inputs = _prepare_growth_inputs(likelihood)
    residual_input_workspace = prepare_residual_inputs(
        reads, settings.read_error_probability,
        likelihood=likelihood,
    )
    data_modes = _initial_complete_modes(
        likelihood,
        settings.data_start_beam_width,
        settings.n_data_seed_modes,
        settings.soft_seed_min_cluster_size,
        settings.lambda_wildcard_penalty,
        settings.coordinate_descent_max_iter,
        fit_workspace=workspace,
    )
    if any(mode.k != 1 for mode in data_modes):
        raise AssertionError("data-derived starts must remain independent K=1 fits")

    panel_starts: list[np.ndarray] = []
    for seed in seed_haplotypes:
        source = seed.haplotypes if isinstance(seed, FactorizationMode) else seed
        haplotypes = np.ascontiguousarray(np.asarray(source), dtype=np.int64)
        if (
            haplotypes.ndim != 2
            or haplotypes.shape[1] != n_sites
            or not 1 <= len(haplotypes) <= ceiling
        ):
            raise ValueError("invalid complete-panel seed")
        if np.any((haplotypes != 0) & (haplotypes != 1)):
            raise ValueError("complete-panel seeds must be binary")
        # A supplied FactorizationMode contributes H only.  Assignments and
        # costs are rebuilt from the current evidence below.
        panel_starts.append(haplotypes)
    panel_starts.extend(row[None, :] for row in candidate_rows)

    limits: list[str] = []
    if len(panel_starts) > settings.max_candidate_start_rows:
        panel_starts = panel_starts[:settings.max_candidate_start_rows]
        limits.append("initial_proposal_budget")
    initial_modes = _fit_panel_starts(
        likelihood, panel_starts, move_config, workspace
    )

    archive: dict[bytes, FactorizationMode] = {}
    representatives: dict[int, FactorizationMode] = {}
    score_cache: dict[bytes, ReversibleModeScore] = {}
    scores: dict[int, ReversibleModeScore] = {}
    exact_evaluations = 0
    exact_cache_hits = 0
    mean_field_nonconvergence = False
    soft_birth_cache: dict[tuple[bytes, bytes], Any] = {}

    def add_modes(values: Sequence[FactorizationMode]) -> None:
        for mode in values:
            if mode.k > ceiling:
                raise AssertionError("proposal exceeded natural K ceiling")
            previous = archive.get(mode.canonical_key)
            if previous is None or _mode_order(mode) < _mode_order(previous):
                archive[mode.canonical_key] = mode

    add_modes((*data_modes, *initial_modes))
    if not archive:
        raise RuntimeError("reversible search produced no initial complete mode")
    exact_config = _stage_config(settings.cavity, "mean_field")
    cavity_workspace = _prepare_cavity_scoring_workspace(
        likelihood, exact_config
    )

    def refresh(preferred_k: int) -> bool:
        """Score changed minimum-NLL representatives within the exact budget."""

        nonlocal exact_evaluations, exact_cache_hits, mean_field_nonconvergence
        grouped: dict[int, list[FactorizationMode]] = {}
        for mode in archive.values():
            grouped.setdefault(mode.k, []).append(mode)
        changed: list[FactorizationMode] = []
        for k in sorted(grouped):
            representative = min(grouped[k], key=_mode_order)
            previous = representatives.get(k)
            if previous is None or previous.canonical_key != representative.canonical_key:
                changed.append(representative)
        changed.sort(
            key=lambda mode: (
                abs(mode.k - preferred_k),
                mode.k,
                mode.total_nll,
                mode.canonical_key,
            )
        )
        novel: list[FactorizationMode] = []
        for mode in changed:
            cached = score_cache.get(mode.canonical_key)
            if cached is None:
                novel.append(mode)
            else:
                exact_cache_hits += 1
                representatives[mode.k] = mode
                scores[mode.k] = cached
        room = max(0, settings.max_exact_scores - exact_evaluations)
        retained = novel[:room]
        omitted = novel[room:]
        for stage in _score_stage(
            likelihood, retained, exact_config, cavity_workspace
        ):
            score = ReversibleModeScore(
                mode=stage.mode,
                mode_digest=stage.digest,
                anchored_cavity_log_predictive=math.nan,
                anchored_log_score=math.nan,
                cavity_log_predictive=stage.cavity_log_predictive,
                log_k_prior=stage.log_k_prior,
                log_haplotype_set_prior=stage.log_haplotype_set_prior,
                log_score=stage.log_score,
                mean_field_diagnostic=stage.diagnostic,
                anchored_diagnostic=stage.diagnostic,
            )
            score_cache[stage.mode.canonical_key] = score
            representatives[stage.mode.k] = stage.mode
            scores[stage.mode.k] = score
            if stage.diagnostic.n_mean_field_not_converged:
                mean_field_nonconvergence = True
        exact_evaluations += len(retained)
        return bool(omitted)

    if refresh(1):
        limits.append("exact_score_budget")
    if mean_field_nonconvergence:
        limits.append("mean_field_nonconvergence")

    ordinary_expanded: set[bytes] = set()
    residual_expanded: set[bytes] = set()
    ordinary_complete: dict[bytes, bool] = {}
    residual_complete: dict[bytes, bool] = {}
    generated_offsets: dict[bytes, set[int]] = {}
    audit_no_improvement: set[bytes] = set()
    search_steps: list[SearchStepDiagnostic] = []
    residual_calls = 0
    residual_candidates = 0
    stop_reason = "expansion_budget_exhausted"

    def selected_score() -> ReversibleModeScore:
        return min(scores.values(), key=_score_sort_key)

    def selected_beam(k: int) -> tuple[FactorizationMode, ...]:
        modes = sorted(
            (mode for mode in archive.values() if mode.k == k),
            key=_mode_order,
        )
        return tuple(modes[:settings.beam_width])

    for expansion in range(settings.max_expansions):
        before = selected_score()
        beam = selected_beam(before.k)
        parent = next(
            (mode for mode in beam if mode.canonical_key not in ordinary_expanded),
            None,
        )
        phase = "ordinary"
        if parent is None:
            parent = next(
                (mode for mode in beam if mode.canonical_key not in residual_expanded),
                None,
            )
            phase = "residual"
        if parent is None:
            stop_reason = (
                "generated_neighbourhood_local_optimum"
                if not limits
                else "incumbent_neighbourhood_search_limited"
            )
            break

        local_limits: list[str] = []
        exact_before = exact_evaluations
        cache_before = exact_cache_hits
        proposed_counts: tuple[tuple[int, int], ...] = ()
        generated: list[FactorizationMode] = []
        proposal_omissions = 0

        if phase == "ordinary":
            ordinary_expanded.add(parent.canonical_key)
            (
                ordinary_modes,
                ordinary_births,
                proposed_counts,
                proposal_omissions,
                move_limits,
            ) = _v2_ordinary_moves(
                parent,
                likelihood,
                candidate_rows,
                ceiling,
                settings,
                move_config,
                workspace,
                growth_inputs,
            )
            generated.extend(ordinary_modes)
            local_limits.extend(move_limits)
            add_modes(ordinary_modes)
            if refresh(parent.k):
                local_limits.append("exact_score_budget")
            after_direct = selected_score()
            direct_improved = (
                after_direct.log_score
                > before.log_score + settings.score_tolerance
            )
            if not direct_improved:
                replacements, replacement_omissions = _birth_delete_replacements(
                    parent,
                    ordinary_births,
                    likelihood,
                    settings,
                    move_config,
                    workspace,
                )
                generated.extend(replacements)
                proposal_omissions += replacement_omissions
                if replacement_omissions:
                    local_limits.append("replacement_proposal_budget")
                add_modes(replacements)
                if refresh(parent.k):
                    local_limits.append("exact_score_budget")
            ordinary_complete[parent.canonical_key] = not local_limits
        else:
            residual_expanded.add(parent.canonical_key)
            soft_births, emitted, omitted, soft_limits = _v2_soft_births(
                parent,
                likelihood,
                reads,
                ceiling,
                settings,
                move_config,
                workspace,
                residual_input_workspace,
                soft_birth_cache,
            )
            residual_calls += 1
            residual_candidates += emitted
            proposal_omissions += omitted
            local_limits.extend(soft_limits)
            generated.extend(soft_births)
            add_modes(soft_births)
            # Refresh is deliberately unconditional: a retained soft proposal
            # must be considered even when other proposals were capped.
            if refresh(parent.k):
                local_limits.append("exact_score_budget")
            after_soft = selected_score()
            soft_improved = (
                after_soft.log_score
                > before.log_score + settings.score_tolerance
            )
            if not soft_improved:
                replacements, replacement_omissions = _birth_delete_replacements(
                    parent,
                    soft_births,
                    likelihood,
                    settings,
                    move_config,
                    workspace,
                )
                generated.extend(replacements)
                proposal_omissions += replacement_omissions
                if replacement_omissions:
                    local_limits.append("replacement_proposal_budget")
                add_modes(replacements)
                if refresh(parent.k):
                    local_limits.append("exact_score_budget")

            after_replacement = selected_score()
            if (
                after_replacement.log_score
                <= before.log_score + settings.score_tolerance
                and parent.k + 2 <= ceiling
                and parent.k + 1 in representatives
            ):
                bridge = representatives[parent.k + 1]
                bridge_children, _attempts, _fallbacks = _expand_one_complete_mode(
                    bridge,
                    likelihood,
                    settings.lambda_wildcard_penalty,
                    settings.coordinate_descent_max_iter,
                    fit_workspace=workspace,
                    oracle_nll=growth_inputs.oracle_nll,
                    decisiveness=growth_inputs.decisiveness,
                    dosage_by_sample=growth_inputs.dosage_by_sample,
                    seed_haplotypes_by_sample=growth_inputs.seed_haplotypes_by_sample,
                )
                paired = [
                    mode for mode in bridge_children
                    if mode.k == parent.k + 2
                ]
                bridge_soft, emitted, omitted, bridge_limits = _v2_soft_births(
                    bridge,
                    likelihood,
                    reads,
                    ceiling,
                    settings,
                    move_config,
                    workspace,
                    residual_input_workspace,
                    soft_birth_cache,
                )
                residual_calls += 1
                residual_candidates += emitted
                proposal_omissions += omitted
                local_limits.extend(bridge_limits)
                paired.extend(
                    mode for mode in bridge_soft if mode.k == parent.k + 2
                )
                paired_modes = _deduplicate(paired)
                generated.extend(paired_modes)
                add_modes(paired_modes)
                if refresh(parent.k):
                    local_limits.append("exact_score_budget")

            residual_complete[parent.canonical_key] = not local_limits
            after = selected_score()
            if (
                after.mode.canonical_key == parent.canonical_key
                and after.log_score
                <= before.log_score + settings.score_tolerance
            ):
                audit_no_improvement.add(parent.canonical_key)

        limits.extend(local_limits)
        if mean_field_nonconvergence:
            limits.append("mean_field_nonconvergence")
        offsets = generated_offsets.setdefault(parent.canonical_key, set())
        offsets.update(
            mode.k - parent.k
            for mode in generated
            if mode.k - parent.k in _SUPPORTED_MOVE_OFFSETS
        )
        after = selected_score()
        best_neighbours = tuple(
            MoveScore(
                offset=offset,
                k=parent.k + offset,
                mode_digest=scores[parent.k + offset].mode_digest,
                log_score=scores[parent.k + offset].log_score,
            )
            for offset in sorted(offsets)
            if parent.k + offset in scores
        )
        search_steps.append(SearchStepDiagnostic(
            expansion=expansion,
            round_index=expansion,
            expanded_k=parent.k,
            expanded_mode_digest=_canonicalize_mode(parent, parent.k).digest,
            expanded_log_score=before.log_score,
            proposed_by_offset=proposed_counts,
            proposals_after_budget=len(generated),
            proposal_budget_omissions=proposal_omissions,
            anchored_scores_computed=0,
            exact_cache_hits=exact_cache_hits - cache_before,
            exact_scores_computed=exact_evaluations - exact_before,
            anchored_screen_omissions=0,
            exact_budget_omissions=int("exact_score_budget" in local_limits),
            improving_exact_neighbours=sum(
                item.log_score > before.log_score + settings.score_tolerance
                for item in best_neighbours
            ),
            incumbent_k=after.k,
            incumbent_mode_digest=after.mode_digest,
            incumbent_log_score=after.log_score,
            proposal_generation_complete=not local_limits,
            all_generated_neighbours_exact_scored=(
                "exact_score_budget" not in local_limits
            ),
            best_exact_neighbours=best_neighbours,
            limitation_reasons=tuple(dict.fromkeys(local_limits)),
        ))
    else:
        limits.append("expansion_budget")

    if mean_field_nonconvergence:
        limits.append("mean_field_nonconvergence")
    unique_limits = tuple(dict.fromkeys(limits))
    ordered = tuple(sorted(scores.values(), key=_score_sort_key))
    selected = ordered[0]
    _log_weights, probabilities = _normalized_log_weights(
        tuple(score.log_score for score in ordered)
    )
    by_k = tuple(sorted((score.k, score.log_score) for score in ordered))
    probability_by_score_k = {
        score.k: probability for score, probability in zip(ordered, probabilities)
    }
    pseudo = tuple(
        (k, probability_by_score_k[k]) for k, _score in by_k
    )
    grouped: dict[int, list[FactorizationMode]] = {}
    for mode in archive.values():
        grouped.setdefault(mode.k, []).append(mode)

    selected_key = selected.mode.canonical_key
    selected_offsets = tuple(sorted(generated_offsets.get(selected_key, set())))
    exact_offsets = tuple(
        offset
        for offset in selected_offsets
        if selected.k + offset in scores
    )
    best_exact = tuple(
        MoveScore(
            offset=offset,
            k=selected.k + offset,
            mode_digest=scores[selected.k + offset].mode_digest,
            log_score=scores[selected.k + offset].log_score,
        )
        for offset in exact_offsets
    )
    incumbent_expanded = (
        selected_key in ordinary_expanded and selected_key in residual_expanded
    )
    generation_complete = (
        ordinary_complete.get(selected_key, False)
        and residual_complete.get(selected_key, False)
    )
    all_exact = all(selected.k + offset in scores for offset in selected_offsets)
    no_improvement = (
        True if selected_key in audit_no_improvement else None
    )
    certified = bool(
        incumbent_expanded
        and generation_complete
        and all_exact
        and no_improvement
        and not unique_limits
        and not mean_field_nonconvergence
    )
    certificate = LocalSearchCertificate(
        incumbent_k=selected.k,
        incumbent_mode_digest=selected.mode_digest,
        incumbent_log_score=selected.log_score,
        incumbent_was_expanded=incumbent_expanded,
        generated_move_offsets=selected_offsets,
        exact_scored_move_offsets=exact_offsets,
        best_exact_neighbours=best_exact,
        proposal_generation_complete=generation_complete,
        all_generated_neighbours_exact_scored=all_exact,
        no_improving_generated_neighbour=no_improvement,
        certified_generated_neighbourhood_local_optimum=certified,
        limitation_reasons=unique_limits,
        scope=(
            "Configured low-NLL beam at the selected K, with death, gauge, "
            "ordinary and proposal-D soft births, lazy birth-delete "
            "replacements, and a minimum-NLL K+1 paired bridge. This is not "
            "a global certificate over all binary panels."
        ),
    )
    at_ceiling = selected.k == ceiling
    return ReversibleCavitySearchResult(
        selected=selected,
        runner_up=ordered[1] if len(ordered) > 1 else None,
        visited_scores=tuple(sorted(ordered, key=lambda item: item.k)),
        anchored_scores=(),
        visited_modes_by_k=tuple(
            (k, tuple(sorted(grouped[k], key=_mode_order)))
            for k in sorted(grouped)
        ),
        best_score_by_k=by_k,
        pseudo_probability_by_k=pseudo,
        search_steps=tuple(search_steps),
        local_certificate=certificate,
        natural_k_ceiling=ceiling,
        search_limited=bool(unique_limits),
        search_limit_reasons=unique_limits,
        boundary_limited=at_ceiling,
        stop_reason=stop_reason,
        data_start_count=len(data_modes),
        overcomplete_data_start_count=0,
        supplied_panel_start_count=len(seed_haplotypes),
        candidate_row_count=len(candidate_rows),
        exact_score_evaluations=exact_evaluations,
        exact_score_cache_hits=exact_cache_hits,
        anchored_score_evaluations=0,
        high_k_tail_mass_upper_bound=math.nan,
        high_k_tail_log_mass_upper_bound=math.nan,
        high_k_tail_pseudo_probability_upper_bound=math.nan,
        high_k_tail_bound_interpretation=(
            "No tail bound: adaptive support is not an enumerated K prefix."
        ),
        objective_interpretation=(
            "Only the deterministic minimum-full-data-NLL representative per "
            "represented K is mean-field cavity-scored. Scores retain "
            "full-data selection leakage and normalized values are "
            "uncalibrated pseudo-weights."
        ),
        search_interpretation=(
            "Cap-free-in-K adaptive reversible search with lazy basin, "
            "replacement, proposal-D residual and paired-bridge expansion "
            f"({residual_calls} residual calls, {residual_candidates} emitted "
            "novel rows). Finite work budgets are operational limitations; "
            "the natural identifiable ceiling alone sets boundary_limited."
        ),
    )


def as_cavity_selection(
    result: ReversibleCavitySearchResult,
) -> CavitySelection:
    """Build the existing cavity-selection schema from cached exact scores."""

    if not isinstance(result, ReversibleCavitySearchResult):
        raise TypeError("result must be a ReversibleCavitySearchResult")
    scores_by_k: dict[int, list[ReversibleModeScore]] = {}
    for score in result.visited_scores:
        scores_by_k.setdefault(score.k, []).append(score)
    probability_by_k = dict(result.pseudo_probability_by_k)
    normalized_logs, _probabilities = _normalized_log_weights(
        tuple(score.log_score for score in result.visited_scores)
    )
    log_weight_by_k = {
        score.k: log_weight
        for score, log_weight in zip(result.visited_scores, normalized_logs)
    }
    best_by_k = {
        k: min(
            scores,
            key=lambda item: (
                -item.log_score, item.mode.canonical_key
            ),
        )
        for k, scores in scores_by_k.items()
    }
    ranked_k = sorted(
        best_by_k,
        key=lambda k: (-best_by_k[k].log_score, k),
    )
    k_diagnostics: list[CavityKDiagnostic] = []
    mode_diagnostics: list[CavityModeDiagnostic] = []
    n_samples = len(result.selected.mode.assignments)
    for k in sorted(scores_by_k):
        scores = scores_by_k[k]
        best = best_by_k[k]
        values = [score.cavity_log_predictive for score in scores]
        maximum = max(values)
        uniform_log_mean_exp = (
            maximum
            + math.log(math.fsum(math.exp(value - maximum) for value in values))
            - math.log(len(values))
        )
        diagnostic = best.mean_field_diagnostic
        probability = probability_by_k[k]
        k_diagnostics.append(CavityKDiagnostic(
            k=k,
            cavity_log_predictive=best.cavity_log_predictive,
            mean_sample_log_predictive=best.cavity_log_predictive / n_samples,
            log_k_prior=best.log_k_prior,
            log_haplotype_set_prior=best.log_haplotype_set_prior,
            log_score=best.log_score,
            log_weight=log_weight_by_k[k],
            pseudo_probability=probability,
            best_total_nll=best.mode.total_nll,
            best_mode_digest=best.mode_digest,
            n_input_modes=len(result.modes(k)),
            n_unique_modes=len(result.modes(k)),
            n_duplicate_modes_removed=0,
            n_mean_field_site_fits=diagnostic.n_mean_field_site_fits,
            n_mean_field_not_converged=diagnostic.n_mean_field_not_converged,
            mean_mean_field_iterations=diagnostic.mean_mean_field_iterations,
            n_zero_support_founder_cavities=(
                diagnostic.n_zero_support_founder_cavities
            ),
            mean_founder_allele_entropy_nats=(
                diagnostic.mean_founder_allele_entropy_nats
            ),
            n_alternate_initialization_wins=(
                diagnostic.n_alternate_initialization_wins
            ),
            mean_initialization_elbo_spread=(
                diagnostic.mean_initialization_elbo_spread
            ),
            uniform_mode_log_mean_exp=uniform_log_mean_exp,
            n_modes_scored=len(scores),
            n_modes_omitted_by_cap=max(0, len(result.modes(k)) - 1),
        ))
        for score in scores:
            mode_diagnostics.append(replace(
                score.mean_field_diagnostic,
                selected_within_k=(score.mode_digest == best.mode_digest),
            ))
    return CavitySelection(
        method="reversible_regularized_fixed_assignment_cavity",
        map_k=result.selected_k,
        runner_up_k=None if len(ranked_k) == 1 else ranked_k[1],
        selected_mode_digest=result.selected.mode_digest,
        k_diagnostics=tuple(k_diagnostics),
        mode_diagnostics=tuple(mode_diagnostics),
        support_selected_from_full_data=True,
        assignments_selected_from_full_data=True,
        selection_leakage=True,
        weights_are_calibrated=False,
        k_prior="telescoping_1_over_k_k_plus_1_applied_once",
        boundary_limited=result.boundary_limited,
        founder_inference="mean_field",
        apply_unordered_haplotype_set_code=True,
        mode_cap_per_k=1,
        mode_cap_applied=any(
            len(modes) > 1 for _k, modes in result.visited_modes_by_k
        ),
        all_mean_field_converged=all(
            item.n_mean_field_not_converged == 0
            for item in mode_diagnostics
        ),
        interpretation=(
            "Cached full mean-field fixed-assignment cavity scores from "
            "deterministic reversible complete-panel search. The telescoping "
            "K prior and unordered binary-haplotype-set code are applied once. "
            "Mode support and assignments used full data; normalized values "
            "are uncalibrated selection-leakage-affected pseudo-weights. "
            "The deterministic minimum-full-data-NLL representative is the "
            "single scored mode at each represented K; archived alternatives "
            "are reported as per-K cap omissions by this compatibility "
            "schema. Operational search limits are recorded on the source "
            "result and are not represented as a K boundary."
        ),
        n_samples=n_samples,
        n_sites=result.selected.mode.n_sites,
        hybrid_diagnostic=None,
    )


def self_test() -> Mapping[str, Any]:
    """Run focused controller regressions on bounded synthetic evidence."""

    h2 = np.asarray([[0, 0, 1], [1, 1, 0]], dtype=np.int64)
    sample_rows = h2[np.asarray([0, 0, 1, 1, 0, 1])]
    evidence = np.full((len(sample_rows), h2.shape[1], 3), 0.002)
    allele_depths = np.zeros((len(sample_rows), h2.shape[1], 2), dtype=np.int64)
    for sample, row in enumerate(sample_rows):
        allele_depths[sample, :, 0] = np.where(row == 0, 20, 0)
        allele_depths[sample, :, 1] = np.where(row == 1, 20, 0)
        evidence[sample, np.arange(h2.shape[1]), 2 * row] = 0.996
    evidence /= np.sum(evidence, axis=2, keepdims=True)

    settings = ReversibleCavitySearchConfig(
        beam_width=1,
        max_expansions=4,
        max_exact_scores=40,
        max_proposals_per_expansion=16,
        data_start_beam_width=2,
        n_data_seed_modes=2,
        max_candidate_start_rows=2,
        max_replacement_children_per_mode=4,
        coordinate_descent_max_iter=12,
        apply_gauge_rewire=False,
        cavity=HybridCavitySelectionConfig(mean_field_max_iter=40),
    )
    try:
        search_reversible_cavity(evidence, config=settings)
    except TypeError as error:
        assert "allele_depths" in str(error)
    else:
        raise AssertionError("allele_depths must be a required input")

    result = search_reversible_cavity(
        evidence,
        (h2,),
        candidate_haplotypes=h2,
        allele_depths=allele_depths,
        config=settings,
    )
    assert result.visited_scores
    assert result.overcomplete_data_start_count == 0
    selection = as_cavity_selection(result)
    assert selection.map_k == result.selected_k
    assert selection.selected_mode_digest == result.selected.mode_digest
    assert selection.boundary_limited == result.boundary_limited
    assert selection.mode_cap_per_k == 1
    assert selection.mode_cap_applied == any(
        len(modes) > 1 for _k, modes in result.visited_modes_by_k
    )
    assert result.natural_k_ceiling == min(2 * len(evidence), 2 ** h2.shape[1])
    assert result.exact_score_evaluations >= len(result.visited_scores)
    assert all(math.isfinite(score.log_score) for score in result.visited_scores)
    assert all(
        score.mean_field_diagnostic.mode_digest == score.mode_digest
        for score in result.visited_scores
    )
    assert math.isclose(
        math.fsum(value for _k, value in result.pseudo_probability_by_k),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert "max_k" not in ReversibleCavitySearchConfig.__dataclass_fields__

    # A one-site, one-sample problem has natural ceiling K=2.  Neither direct
    # nor paired residual growth may create an archived K=3 state.
    tiny_evidence = np.asarray(
        [
            [[0.996, 0.002, 0.002]],
            [[0.002, 0.002, 0.996]],
        ],
        dtype=np.float64,
    )
    tiny_reads = np.asarray([[[20, 0]], [[0, 20]]], dtype=np.int64)
    ceiling_result = search_reversible_cavity(
        tiny_evidence,
        (np.asarray([[0], [1]], dtype=np.int64),),
        allele_depths=tiny_reads,
        config=ReversibleCavitySearchConfig(
            beam_width=1,
            max_expansions=2,
            max_exact_scores=12,
            max_proposals_per_expansion=4,
            data_start_beam_width=1,
            n_data_seed_modes=1,
            max_candidate_start_rows=1,
            max_replacement_children_per_mode=2,
            coordinate_descent_max_iter=4,
            apply_gauge_rewire=False,
            cavity=HybridCavitySelectionConfig(mean_field_max_iter=10),
        ),
    )
    assert ceiling_result.natural_k_ceiling == 2
    assert all(k <= 2 for k, _modes in ceiling_result.visited_modes_by_k)

    # Mean-field nonconvergence is an operational limitation and can never
    # support a generated-neighbourhood certificate.
    nonconverged = search_reversible_cavity(
        evidence,
        (h2,),
        allele_depths=allele_depths,
        config=ReversibleCavitySearchConfig(
            beam_width=1,
            max_expansions=1,
            max_exact_scores=20,
            max_proposals_per_expansion=8,
            data_start_beam_width=1,
            n_data_seed_modes=1,
            max_candidate_start_rows=1,
            max_replacement_children_per_mode=2,
            coordinate_descent_max_iter=4,
            apply_gauge_rewire=False,
            cavity=HybridCavitySelectionConfig(
                mean_field_max_iter=1,
                mean_field_tolerance=1e-15,
            ),
        ),
    )
    assert "mean_field_nonconvergence" in nonconverged.search_limit_reasons
    assert not nonconverged.local_certificate.certified_generated_neighbourhood_local_optimum

    log_weights, probabilities = _normalized_log_weights((0.0, -1000.0))
    assert all(math.isfinite(value) for value in log_weights)
    assert probabilities[0] == 1.0
    assert probabilities[1] == 0.0

    return {
        "status": "ok",
        "selected_k": result.selected_k,
        "visited_modes": len(result.visited_scores),
        "represented_k": tuple(k for k, _modes in result.visited_modes_by_k),
        "search_limited": result.search_limited,
        "natural_k_ceiling": result.natural_k_ceiling,
        "cap_free_config": True,
        "required_allele_depths": True,
        "mean_field_nonconvergence_withholds_certificate": True,
        "natural_ceiling_enforced": True,
        "underflow_safe_log_weights": True,
        "candidate_rows_are_anonymous_proposals": True,
        "overcomplete_panel_start_removed": True,
        "cached_cavity_selection_adapter": True,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("pass --self-test")
    print(json.dumps(self_test(), indent=2, sort_keys=True))
