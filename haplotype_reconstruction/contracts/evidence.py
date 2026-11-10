"""Immutable observation-model evidence and typed quality/calibration decisions.

Counts and raw likelihoods keep their values in the founder view; consumers must
consult observed or materialise neutral cells. The sample axis is canonical run
order. No model fitting, I/O, memory reservation, logging or environment access.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.science.evidence.quality.parameters import SampleQualityParameters

from ..core.ties import TieCounts
from .likelihoods import CodedLikelihoods
from .observations import PairArray
from .roles import freeze_record_arrays, frozen_array, record_field, schema_field


@dataclass(frozen=True)
class ContigEvidence:
    """One contig's evidence (record ``evidence``): the stage boundary every later stage reads.

    contig, sample_ids: as in :class:`ContigReads` (N samples).
    positions: int64[L] sites kept by the site-quality filter (all input sites when the filter is
        off or not applied), strictly increasing.
    counts: dense int32[N, L, 2] or a pair-backed source of the same exact REF/ALT
        read depths at these sites, or None when the record was
        loaded without them (``evidence.records.load(..., counts=False)``; most stages need only
        likelihoods and the observation mask).
    likelihoods: dense float64[N, L, 3] or CodedLikelihoods with that logical shape.
        Explicit gathers return identical float64 P(reads | genotype g) under the contig's
        fitted read model (fixed binomial when calibration falls back), normalised over g;
        zero depth is exactly (1/3, 1/3, 1/3). No genotype prior or implicit ndarray
        conversion is applied. Evidence-v2 stores automatic per-sample codebooks only
        when exact table + codes and maps beat the joint 33-byte dense cell; other samples remain dense.
    observed: dense bool[N, L] or a pair-backed positive-depth source (REF + ALT > 0).
        Coded samples reconstruct counts and flags from the same pairs/codes as GL;
        only dense fallback rows retain separate count and mask arrays.
    discovery_sites: bool[L] sites the discovery search uses: total depth over all N samples at
        least ``DiscoverySiteRule.minimum_total_depth`` (5) reads, whatever the cohort size
        (``evidence.config.DiscoverySiteRule``; base loader ``keep_flags``).
    founder_excluded: bool[N] samples flagged by the contamination QC on this contig (all False
        when the QC is off or did not apply). Founder construction sees them unobserved
        (:meth:`founder_view`); painting and later stages use every sample's evidence.
    block_spans: int64[B, 2] discovery blocks as [start, stop) indices into the site axis:
        consecutive non-overlapping windows of ``EvidenceConfig.block_sites`` sites (200; the last
        block may be shorter).
    """

    schema: int = schema_field(2)
    payload_identity: str = record_field("obs", dtype="str", kw_only=True)
    contig: str = record_field("obs", axes=(), units="1", dtype="str")
    sample_ids: tuple[str, ...] = record_field("obs", axes=("N",), units="1", dtype="str")
    positions: NDArray[np.int64] = record_field("obs", axes=("L",), units="bp", dtype="int64")
    counts: NDArray[np.int32] | PairArray | None = record_field(
        "obs", axes=("N", "L", "2"), units="reads", dtype="int32"
    )
    likelihoods: NDArray[np.float64] | CodedLikelihoods = record_field(
        "evidence", axes=("N", "L", "3"), units="probability", dtype="float64"
    )
    observed: NDArray[np.bool_] | PairArray = record_field("evidence", axes=("N", "L"), units="1", dtype="bool")
    discovery_sites: NDArray[np.bool_] = record_field("state", axes=("L",), units="1", dtype="bool")
    founder_excluded: NDArray[np.bool_] = record_field("state", axes=("N",), units="1", dtype="bool")
    block_spans: NDArray[np.int64] = record_field("state", axes=("B", "2"), units="site", dtype="int64")

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(map(str, self.sample_ids)))
        object.__setattr__(self, "positions", frozen_array(self.positions, np.int64))
        if self.counts is not None and not isinstance(self.counts, PairArray):
            object.__setattr__(self, "counts", frozen_array(self.counts, np.int32))
        if not isinstance(self.likelihoods, CodedLikelihoods):
            object.__setattr__(self, "likelihoods", frozen_array(self.likelihoods, np.float64))
        if not isinstance(self.observed, PairArray):
            object.__setattr__(self, "observed", frozen_array(self.observed, np.bool_))
        object.__setattr__(self, "discovery_sites", frozen_array(self.discovery_sites, np.bool_))
        object.__setattr__(self, "founder_excluded", frozen_array(self.founder_excluded, np.bool_))
        spans = np.asarray(self.block_spans, np.int64)
        if spans.ndim != 2 or spans.shape[1] != 2:
            spans = spans.reshape(-1, 2)
        object.__setattr__(self, "block_spans", frozen_array(spans, np.int64))

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def n_sites(self) -> int:
        return len(self.positions)

    @property
    def n_blocks(self) -> int:
        return len(self.block_spans)

    def block(self, index: int) -> slice:
        """Site-axis slice of discovery block ``index``."""
        start, stop = self.block_spans[index]
        return slice(int(start), int(stop))

    def founder_view(self) -> ContigEvidence:
        """Founder construction's missing-data view (discovery through founder refit).

        Excluded rows have no observations. Immutable counts and likelihoods are shared:
        consumers must mask unobserved cells before reading them, or materialise their
        neutral values (zero counts, likelihoods 1/3) in the block/view they already copy.
        Site axis, discovery sites and blocks are unchanged. No exclusions returns self.
        """
        if not self.founder_excluded.any():
            return self
        if isinstance(self.observed, PairArray):
            excluded = self.founder_excluded
            if self.observed.excluded is not None:
                excluded = excluded | self.observed.excluded
            return replace(self, observed=replace(self.observed, excluded=excluded))
        observed = np.array(self.observed, copy=True)
        observed[self.founder_excluded] = False
        return replace(self, observed=observed)


@dataclass(frozen=True)
class FixedFitDetails:
    """Fixed fallback observation-model error; no fitted candidate."""

    error: float = record_field("state", axes=(), units="probability", dtype="float64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SharedStart:
    """One shared-model start; None fields were not reached after a recorded failure."""

    initial: tuple[float, ...] = record_field("state", axes=(), units="1", dtype="tuple[float, ...]")
    em_log_likelihood: float | None = record_field("state", axes=(), units="nats", dtype="float | None", default=None)
    em_iterations: int | None = record_field("state", axes=(), units="1", dtype="int | None", default=None)
    train_log_likelihood: float | None = record_field(
        "state", axes=(), units="nats", dtype="float | None", default=None
    )
    polish_iterations: int | None = record_field("state", axes=(), units="1", dtype="int | None", default=None)
    polish_converged: bool | None = record_field("state", axes=(), units="1", dtype="bool | None", default=None)
    polish_termination: str | None = record_field("state", axes=(), units="1", dtype="str | None", default=None)
    failure: str | None = record_field("state", axes=(), units="1", dtype="str | None", default=None)
    phase: str | None = record_field("state", axes=(), units="1", dtype="str | None", default=None)
    em_vector: tuple[float, ...] | None = record_field(
        "state", axes=(), units="1", dtype="tuple[float, ...] | None", default=None
    )
    vector: tuple[float, ...] | None = record_field(
        "state", axes=(), units="1", dtype="tuple[float, ...] | None", default=None
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SharedDistance:
    """Relative parameter distances between successful starts."""

    first: int = record_field("state", axes=(), units="1", dtype="int")
    second: int = record_field("state", axes=(), units="1", dtype="int")
    shared: float = record_field("state", axes=(), units="1", dtype="float64")
    nuisance: float = record_field("state", axes=(), units="1", dtype="float64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SharedFitDiagnostics:
    """All recorded start failures and equivalence decisions; scientific fallback state."""

    starts: tuple[SharedStart, ...] = record_field("state", axes=(), units="1", dtype="tuple[SharedStart, ...]")
    pairwise_distances: tuple[SharedDistance, ...] = record_field(
        "state", axes=(), units="1", dtype="tuple[SharedDistance, ...]"
    )
    reason: str | None = record_field("state", axes=(), units="1", dtype="str | None", default=None)
    successful_start_indices: tuple[int, ...] = record_field(
        "state", axes=(), units="1", dtype="tuple[int, ...]", default=()
    )
    best_successful_start: int | None = record_field("state", axes=(), units="1", dtype="int | None", default=None)
    likelihood_tie_nats: float | None = record_field("state", axes=(), units="nats", dtype="float | None", default=None)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class StationaryStart:
    """One successfully polished shared-model endpoint."""

    train_log_likelihood: float = record_field("state", axes=(), units="nats", dtype="float64")
    em_iterations: int = record_field("state", axes=(), units="1", dtype="int")
    polish_iterations: int = record_field("state", axes=(), units="1", dtype="int")
    polish_converged: bool = record_field("state", axes=(), units="1", dtype="bool")
    polish_termination: str = record_field("state", axes=(), units="1", dtype="str")
    gradient_max: float = record_field("state", axes=(), units="1", dtype="float64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SharedFitDetails:
    """Shared-binomial fit endpoints, distances, warnings and fallbacks.

    fit_diagnostics records every start and distance, including successful fits
    without warnings. Producers supply these operands explicitly; the empty default
    represents absent diagnostics. shared_fit_diagnostics retains warning/fallback evidence.
    """

    message: str = record_field("diagnostic", axes=(), units="1", dtype="str")
    iterations: int = record_field("diagnostic", axes=(), units="1", dtype="int")
    train_log_likelihood: float = record_field("state", axes=(), units="nats", dtype="float64")
    failed_starts: tuple[str, ...] = record_field("state", axes=(), units="1", dtype="tuple[str, ...]")
    starts_in_band: int = record_field("report", axes=(), units="1", dtype="int")
    stationary_relative_distance: float = record_field("report", axes=(), units="1", dtype="float64")
    stationary_shared_relative_distance: float = record_field("report", axes=(), units="1", dtype="float64")
    stationary_nuisance_relative_distance: float = record_field("report", axes=(), units="1", dtype="float64")
    stationary_shared_resolution: float = record_field("report", axes=(), units="1", dtype="float64")
    stationary_nuisance_resolution: float = record_field("report", axes=(), units="1", dtype="float64")
    stationary_representative: str = record_field("report", axes=(), units="1", dtype="str")
    stationary_sample_order: str = record_field("report", axes=(), units="1", dtype="str")
    stationary_starts: tuple[StationaryStart, ...] = record_field(
        "state", axes=(), units="1", dtype="tuple[StationaryStart, ...]"
    )
    fit_diagnostics: SharedFitDiagnostics = record_field(
        "state",
        dtype="SharedFitDiagnostics",
        kw_only=True,
        default_factory=lambda: SharedFitDiagnostics(starts=(), pairwise_distances=()),
    )
    shared_fit_warning: str | None = record_field("state", axes=(), units="1", dtype="str | None", default=None)
    shared_fit_diagnostics: SharedFitDiagnostics | None = record_field(
        "state", axes=(), units="1", dtype="SharedFitDiagnostics | None", default=None
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PooledFitDetails:
    """Sample-binomial empirical Bayes fit; hyperparameters affect held-out scores."""

    sample_fit_failures: int = record_field("state", axes=(), units="1", dtype="int")
    polish_failures: int = record_field("state", axes=(), units="1", dtype="int")
    hyper_mean: NDArray[np.float64] = record_field("state", axes=("2",), units="1", dtype="float64")
    hyper_variance: NDArray[np.float64] = record_field("state", axes=("2",), units="1", dtype="float64")
    identifiable_sample_effects: tuple[int, int] = record_field("report", axes=(), units="1", dtype="tuple[int, int]")
    approximation: str = record_field("report", axes=(), units="1", dtype="str")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class JointFitDetails:
    """Joint sample fit and shared heterozygote concentration."""

    hyper_mean: NDArray[np.float64] = record_field("state", axes=("2",), units="1", dtype="float64")
    hyper_variance: NDArray[np.float64] = record_field("state", axes=("2",), units="1", dtype="float64")
    polish_steps: int = record_field("state", axes=(), units="1", dtype="int")
    polish_converged: bool = record_field("state", axes=(), units="1", dtype="bool")
    polish_shift: float = record_field("report", axes=(), units="1", dtype="float64")
    message: str = record_field("diagnostic", axes=(), units="1", dtype="str")
    iterations: int = record_field("diagnostic", axes=(), units="1", dtype="int")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ReadModelFit:
    """One fitted observation model of a contig's N samples.

    error_logit, balance_logit: float64[N]; e = expit(error_logit) is the homozygote error
    (REF/REF reads ALT with mean e, ALT/ALT reads REF with mean e), balance = expit(balance_logit)
    places the heterozygote ALT mean q = e + (1 - 2e) balance between e and 1 - e (homozygotes
    are binomial). log_concentration: shared heterozygote log kappa (beta-binomial
    heterozygotes), None for binomial heterozygotes. mixing: float64[N, 2] genotype-mixture logits
    relative to genotype 2 (training nuisances; never a genotype prior). success: the optimiser's
    own convergence flag (a diagnostic; selection does not use it). details: JSON-compatible fit
    diagnostics.
    """

    name: str = record_field("call", axes=(), units="1", dtype="str")
    error_logit: NDArray[np.float64] = record_field("state", axes=("N",), units="logit", dtype="float64")
    balance_logit: NDArray[np.float64] = record_field("state", axes=("N",), units="logit", dtype="float64")
    log_concentration: float | None = record_field("state", axes=(), units="1", dtype="float | None")
    mixing: NDArray[np.float64] = record_field("state", axes=("N", "2"), units="logit", dtype="float64")
    success: bool = record_field("diagnostic", axes=(), units="1", dtype="bool")
    details: FixedFitDetails | SharedFitDetails | PooledFitDetails | JointFitDetails = record_field(
        "state", axes=(), units="1", dtype="FixedFitDetails | SharedFitDetails | PooledFitDetails | JointFitDetails"
    )

    @property
    def n_samples(self) -> int:
        return len(self.error_logit)

    @property
    def heterozygote_inverse_concentration(self) -> float:
        return 0.0 if self.log_concentration is None else float(np.exp(-self.log_concentration))

    def local(self) -> np.ndarray:
        """float64[N, 4]: error logit, balance logit, mixing logit 0, mixing logit 1."""
        return np.column_stack((self.error_logit, self.balance_logit, self.mixing))

    def error_and_heterozygote(self) -> tuple[np.ndarray, np.ndarray]:
        """(e, q) per sample: the homozygote error and heterozygote ALT mean."""
        error = 1.0 / (1.0 + np.exp(-np.asarray(self.error_logit, float)))
        balance = 1.0 / (1.0 + np.exp(-np.asarray(self.balance_logit, float)))
        return (error, error + (1.0 - 2.0 * error) * balance)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ReadModelComparison:
    """A held-out comparison; accepted is the released model-choice call."""

    candidate: str = record_field("report", axes=(), units="1", dtype="str")
    against: str = record_field("report", axes=(), units="1", dtype="str")
    gain: float = record_field("report", axes=(), units="nats", dtype="float64")
    standard_error: float = record_field("report", axes=(), units="nats", dtype="float64")
    blocks: int = record_field("report", axes=(), units="1", dtype="int")
    z: float = record_field("report", axes=(), units="1", dtype="float64")
    accepted: bool = record_field("call", axes=(), units="1", dtype="bool")

    exposure: int = record_field("report", dtype="int", units="count")
    largest_block_share: float = record_field("report", dtype="float64", units="share")
    exposure_centred_error: float = record_field("report", dtype="float64", units="nats")
    leave_one_out_flips: int = record_field("report", dtype="int", units="count")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ReadModelDecision:
    """Calibrated-model decision, including produced candidates and exact fallback reasons."""

    applied: bool = record_field("call", axes=(), units="1", dtype="bool")
    selected: str | None = record_field("call", axes=(), units="1", dtype="str | None")
    fallback_reason: str | None = record_field("call", axes=(), units="1", dtype="str | None")
    fold_observations: tuple[int, ...] = record_field("report", axes=(), units="count", dtype="tuple[int, ...]")
    observations_used: int = record_field("report", axes=(), units="count", dtype="int")
    high_depth_excluded: int = record_field("report", axes=(), units="count", dtype="int")
    candidates: Mapping[str, ReadModelFit] = record_field(
        "state", axes=(), units="1", dtype="Mapping[str, ReadModelFit]", default_factory=dict
    )
    scores: Mapping[str, tuple[float, float | None]] = record_field(
        "report", axes=(), units="nats", dtype="Mapping[str, tuple[float, float | None]]", default_factory=dict
    )
    complex_fit_failure: str | None = record_field("state", axes=(), units="1", dtype="str | None", default=None)
    shared_fit_warning: str | None = record_field("state", axes=(), units="1", dtype="str | None", default=None)
    shared_fit_diagnostics: SharedFitDiagnostics | None = record_field(
        "state", axes=(), units="1", dtype="SharedFitDiagnostics | None", default=None
    )
    comparisons: tuple[ReadModelComparison, ...] = record_field(
        "report", axes=(), units="1", dtype="tuple[ReadModelComparison, ...]", default=()
    )
    tie_events: TieCounts = record_field(
        "diagnostic", axes=(), units="1", dtype="TieCounts", default_factory=lambda: TieCounts(())
    )
    model: str = record_field("report", axes=(), units="1", dtype="str", default="heldout-read-model-v7")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class InheritedSegment:
    """A region of heterozygous-like near-private events of one (sample, partner) pair.

    sample, partner: sample-axis indices (partner -1: private to the sample). start, end: bp of the
    first and last event cell. events: fragment events; cells: every near-private cell of the pair
    in the region; minor_read_fraction (and the partner's, None when private).
    """

    sample: int = record_field("report", axes=(), units="1", dtype="int")
    partner: int = record_field("report", axes=(), units="1", dtype="int")
    start: int = record_field("report", axes=(), units="bp", dtype="int")
    end: int = record_field("report", axes=(), units="bp", dtype="int")
    events: float = record_field("report", axes=(), units="1", dtype="float64")
    cells: int = record_field("report", axes=(), units="1", dtype="int")
    minor_read_fraction: float = record_field("report", axes=(), units="1", dtype="float64")
    partner_minor_read_fraction: float | None = record_field("report", axes=(), units="1", dtype="float64 | None")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SampleQualityDecision:
    """One contig's quality exclusion over the analysed samples (arrays of length N).

    applied: the QC could run (else ``reason``; nothing is flagged). flagged bool[N].
    Per sample (NaN where undefined): lrt (T, the likelihood-ratio statistic of H1 against H0 with the artefact
    components), plain_lrt (the same statistic without artefact components; a diagnostic, NaN unless requested),
    foreign_fraction (a at the maximum), foreign_share (c at the maximum), heterozygote_fit (the own-heterozygote ALT
    mean at the null maximum), heterozygote_dispersion (its rho), error_fit (the mean error rate of the null's fitted
    error distribution), artefact_share (the share of eligible-cell weight the null assigns to local artefacts) and
    artefact_peak (the null's largest regional artefact share pi_R); counts: observed_cells,
    eligible_cells, eligible_windows (the sum of the composite
    weights), minor_cells (eligible cells with a minor read), near_private_cells. Scalars: error_rate (e),
    heterozygote_balance (h0), balance_interval ([h_lo, h_hi]) and balance_reason (r4's reason when the interval was
    widened, else None), significance_threshold (u_N), minimum_fraction (a_min), fraction_ceiling (a_hi),
    carrier_level (q / (N - 1)), artefact_regions (the regions actually cut); None when not applied.
    """

    applied: bool = record_field("report", axes=(), units="1", dtype="bool")
    reason: str | None = record_field("report", axes=(), units="1", dtype="str | None")
    passing_sites: int = record_field("report", axes=(), units="1", dtype="int")
    heterozygote_balance: float = record_field("report", axes=(), units="1", dtype="float")
    balance_interval: tuple[float, float] = record_field("state", axes=("2",), units="1", dtype="tuple[float,float]")
    balance_reason: str | None = record_field("report", axes=(), units="1", dtype="str | None")
    error_rate: float = record_field("report", axes=(), units="1", dtype="float")
    flagged: NDArray[np.bool_] = record_field("call", axes=("N",), units="1", dtype="bool")
    lrt: NDArray[np.float64] = record_field("report", axes=("N",), units="nats", dtype="float64")
    plain_lrt: NDArray[np.float64] = record_field("report", axes=("N",), units="nats", dtype="float64")
    foreign_fraction: NDArray[np.float64] = record_field("report", axes=("N",), units="1", dtype="float64")
    foreign_share: NDArray[np.float64] = record_field("report", axes=("N",), units="1", dtype="float64")
    heterozygote_fit: NDArray[np.float64] = record_field("report", axes=("N",), units="1", dtype="float64")
    heterozygote_dispersion: NDArray[np.float64] = record_field("report", axes=("N",), units="1", dtype="float64")
    error_fit: NDArray[np.float64] = record_field("report", axes=("N",), units="1", dtype="float64")
    artefact_share: NDArray[np.float64] = record_field("report", axes=("N",), units="1", dtype="float64")
    artefact_peak: NDArray[np.float64] = record_field("report", axes=("N",), units="1", dtype="float64")
    observed_cells: NDArray[np.int64] = record_field("report", axes=("N",), units="count", dtype="int64")
    eligible_cells: NDArray[np.int64] = record_field("report", axes=("N",), units="count", dtype="int64")
    eligible_windows: NDArray[np.float64] = record_field("report", axes=("N",), units="1", dtype="float64")
    minor_cells: NDArray[np.int64] = record_field("report", axes=("N",), units="count", dtype="int64")
    near_private_cells: NDArray[np.int64] = record_field("report", axes=("N",), units="count", dtype="int64")
    significance_threshold: float | None = record_field("report", axes=(), units="1", dtype="float | None")
    minimum_fraction: float | None = record_field("report", axes=(), units="1", dtype="float | None")
    fraction_ceiling: float | None = record_field("report", axes=(), units="1", dtype="float | None")
    carrier_level: float | None = record_field("report", axes=(), units="1", dtype="float | None")
    artefact_regions: int = record_field("report", axes=(), units="1", dtype="int")
    segments: tuple[InheritedSegment, ...] = record_field(
        "report", axes=(), units="1", dtype="tuple[InheritedSegment, ...]", default=()
    )
    config: SampleQualityParameters = record_field(
        "report", axes=(), units="1", dtype="SampleQualityParameters", default_factory=SampleQualityParameters
    )
    tie_events: TieCounts = record_field(
        "diagnostic", axes=(), units="1", dtype="TieCounts", default_factory=lambda: TieCounts(())
    )

    version: str = record_field("report", dtype="str", default="private-allele-artefact-mixture-qc-v7")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class DepthCore:
    """Persisted depth core and accepted window. Optional peak details are recorded only when supplied."""

    mode: float = record_field("report", axes=(), units="1", dtype="float64")
    sd_left: float = record_field("report", axes=(), units="1", dtype="float64")
    sd_right: float = record_field("report", axes=(), units="1", dtype="float64")
    core_weight: float = record_field("report", axes=(), units="1", dtype="float64")
    low: float = record_field("report", axes=(), units="1", dtype="float64")
    high: float = record_field("report", axes=(), units="1", dtype="float64")
    poisson_sd: float = record_field("report", axes=(), units="1", dtype="float64")
    peak_density: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    tied_mode: bool | None = record_field("report", axes=(), units="1", dtype="bool | None", default=None)
    bins: int | None = record_field("report", axes=(), units="1", dtype="int | None", default=None)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class HeterogeneityCore:
    """Fitted log-depth heterogeneity core, or explicit no-fit reason."""

    mode: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    sd_log_left: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    sd_log_right: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    high: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    core_weight: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    applied: bool | None = record_field("report", axes=(), units="1", dtype="bool | None", default=None)
    reason: str | None = record_field("report", axes=(), units="1", dtype="str | None", default=None)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ErrorMSteps:
    """Counts of r3 Newton outcomes; omitted counters mean zero occurrences."""

    stationary: int = record_field("report", dtype="int", units="count", default=0)
    precision: int = record_field("report", dtype="int", units="count", default=0)
    line_search: int = record_field("report", dtype="int", units="count", default=0)
    iterations: int = record_field("report", dtype="int", units="count", default=0)
    worst_gradient: float = record_field("report", dtype="float64", default=0.0)


@dataclass(frozen=True)
class ErrorMixture:
    """r3 mixture; eps0 is consumed by sample quality, with explicit no-fit status."""

    applied: bool = record_field("report", axes=(), units="1", dtype="bool")
    eps0: float | None = record_field("state", axes=(), units="1", dtype="float | None", default=None)
    pi_clean: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    eps1: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    shape: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    iterations: int | None = record_field("report", axes=(), units="1", dtype="int | None", default=None)
    m_steps: ErrorMSteps | None = record_field("report", dtype="ErrorMSteps | None", default=None)
    heterozygote_leakage: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    clean_leakage: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    reason: str | None = record_field("report", axes=(), units="1", dtype="str | None", default=None)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class BalanceStart:
    """One r4 start including certification and deterministic winner operands."""

    start: tuple[float, ...] = record_field("state", axes=(), units="1", dtype="tuple[float, ...]")
    parameters: tuple[float, ...] = record_field("state", axes=(), units="1", dtype="tuple[float, ...]")
    objective: float = record_field("state", axes=(), units="1", dtype="float64")
    projected_gradient: float = record_field("state", axes=(), units="1", dtype="float64")
    converged: bool = record_field("state", axes=(), units="1", dtype="bool")
    retries: int = record_field("state", axes=(), units="1", dtype="int")
    polish_iterations: int = record_field("state", axes=(), units="1", dtype="int")
    minimum_curvature: float | None = record_field("state", axes=(), units="1", dtype="float64 | None")
    optimizer_success: bool = record_field("diagnostic", axes=(), units="1", dtype="bool")
    message: str = record_field("diagnostic", axes=(), units="1", dtype="str")
    iterations: int = record_field("diagnostic", axes=(), units="1", dtype="int")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class BalanceMixture:
    """r4 mixture and explicit no-fit/ambiguous class-label state."""

    applied: bool = record_field("report", axes=(), units="1", dtype="bool")
    h0: float = record_field("state", axes=(), units="probability", dtype="float64")
    artefact_mean: float | None = record_field(
        "state", axes=(), units="probability", dtype="float | None", default=None
    )
    reason: str | None = record_field("state", axes=(), units="1", dtype="str | None", default=None)
    model: str | None = record_field("report", axes=(), units="1", dtype="str | None", default=None)
    h0_source: str | None = record_field("report", axes=(), units="1", dtype="str | None", default=None)
    pi_clean: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    clean_inverse_concentration: float | None = record_field(
        "report", axes=(), units="1", dtype="float | None", default=None
    )
    clean_sd: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    artefact_inverse_concentration: float | None = record_field(
        "report", axes=(), units="1", dtype="float | None", default=None
    )
    artefact_sd: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    binomial_sd_at_median_reads: float | None = record_field(
        "report", axes=(), units="1", dtype="float | None", default=None
    )
    separation_scale: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    objective: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    projected_gradient: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    clean_binomial: bool | None = record_field("report", axes=(), units="1", dtype="bool | None", default=None)
    artefact_binomial: bool | None = record_field("report", axes=(), units="1", dtype="bool | None", default=None)
    clean_concentration: float | None = record_field("report", axes=(), units="1", dtype="float | None", default=None)
    artefact_concentration: float | None = record_field(
        "report", axes=(), units="1", dtype="float | None", default=None
    )
    starts: tuple[BalanceStart, ...] = record_field(
        "state", axes=(), units="1", dtype="tuple[BalanceStart, ...]", default=()
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SiteQualityFit:
    """Fitted site-quality records; absent fits remain None, never invented estimates."""

    records: int | None = record_field("report", axes=(), units="1", dtype="int | None", default=None)
    samples: int | None = record_field("report", axes=(), units="1", dtype="int | None", default=None)
    snp: int | None = record_field("report", axes=(), units="1", dtype="int | None", default=None)
    depth: DepthCore | None = record_field("report", axes=(), units="1", dtype="DepthCore | None", default=None)
    heterogeneity: HeterogeneityCore | None = record_field(
        "report", axes=(), units="1", dtype="HeterogeneityCore | None", default=None
    )
    error: ErrorMixture | None = record_field("state", axes=(), units="1", dtype="ErrorMixture | None", default=None)
    balance: BalanceMixture | None = record_field(
        "state", axes=(), units="1", dtype="BalanceMixture | None", default=None
    )
    tie_events: TieCounts = record_field(
        "diagnostic", axes=(), units="1", dtype="TieCounts", default_factory=lambda: TieCounts(())
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SiteQualityDecision:
    """Site removal decisions on the input site axis; sample QC is stored separately."""

    enabled: bool = record_field("call", axes=(), units="1", dtype="bool")
    applied: bool = record_field("call", axes=(), units="1", dtype="bool")
    reason: str | None = record_field("call", axes=(), units="1", dtype="str | None")
    kept: NDArray[np.bool_] = record_field("call", axes=("M_in",), units="1", dtype="bool_")
    rule_bits: NDArray[np.uint8] | None = record_field(
        "call", axes=("M_in",), units="1", dtype="uint8 | None", default=None
    )
    applied_rules: tuple[str, ...] = record_field("call", axes=(), units="1", dtype="tuple[str, ...]", default=())
    flagged: Mapping[str, int] = record_field(
        "report", axes=(), units="1", dtype="Mapping[str, int]", default_factory=dict
    )
    removed_only_by: Mapping[str, int] = record_field(
        "report", axes=(), units="1", dtype="Mapping[str, int]", default_factory=dict
    )
    parameters: SiteQualityFit = record_field(
        "state", axes=(), units="1", dtype="SiteQualityFit", default_factory=SiteQualityFit
    )
    r5_excluded_samples: int = record_field("report", axes=(), units="1", dtype="int", default=0)
    version: str = record_field("report", axes=(), units="1", dtype="str", default="self-calibrated-site-quality-v4")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class EvidenceDecisions:
    """One contig QC and read-model decisions; sample_quality=None means QC did not run."""

    schema: int = schema_field(1)
    contig: str = record_field("obs", axes=(), units="1", dtype="str")
    input_positions: NDArray[np.int64] = record_field("obs", axes=("M_in",), units="bp", dtype="int64")
    dropped: Mapping[str, int] = record_field("report", axes=(), units="1", dtype="Mapping[str, int]")
    site_quality: SiteQualityDecision = record_field("call", axes=(), units="1", dtype="SiteQualityDecision")
    sample_quality: SampleQualityDecision | None = record_field(
        "call", axes=(), units="1", dtype="SampleQualityDecision | None"
    )
    read_model: ReadModelDecision = record_field("call", axes=(), units="1", dtype="ReadModelDecision")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class EvidenceDiagnostics:
    """Runtime and fit diagnostic summaries without scientific decisions."""

    schema: int = schema_field(1)
    timings: Mapping[str, float] = record_field(
        "diagnostic", axes=(), units="seconds", dtype="Mapping[str, float]", default_factory=dict
    )
    tie_events: TieCounts = record_field(
        "diagnostic", axes=(), units="1", dtype="TieCounts", default_factory=lambda: TieCounts(())
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)
