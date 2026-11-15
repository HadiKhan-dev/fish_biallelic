"""Relatedness family calls and their measured fit summaries.

Only record definitions live here: no relatedness estimation, canonicalisation or storage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .roles import freeze_record_arrays, record_field, schema_field


@dataclass(frozen=True)
class RecombinationRateEstimate:
    """The genome-wide recombination rate estimated from full-sib families, its gate and the working-map decision.

    One fixed rate in cM/Mb for the whole genome; per-chromosome values are diagnostics and never enter a map.
    Family rows (R) cover every relatedness family of at least two members, in family order. ``decision`` is
    ``adopted`` (the fallback rate of the working map, on contigs without a map file and beyond every file's knots,
    is ``working_rate_cm_per_mb``, which then differs from the supplied rate; only with rate_estimate = adopt) or
    ``kept`` (the supplied map is the working map). ``map_free_contigs``: the contigs the estimate is fitted on when
    some contigs have map files (a fully file-covered genome is fitted on every contig, as a diagnostic). NaN marks an estimate or interval that does not exist.
    """

    estimate_cm_per_mb: float = record_field("diagnostic", units="cM/Mb", dtype="float")
    interval_low_cm_per_mb: float = record_field("diagnostic", units="cM/Mb", dtype="float")
    interval_high_cm_per_mb: float = record_field("diagnostic", units="cM/Mb", dtype="float")
    interval_kind: str = record_field("diagnostic", dtype="str")
    meioses: int = record_field("diagnostic", units="count", dtype="int")
    determined_sites: float = record_field("diagnostic", units="site", dtype="float")
    error_rate: float = record_field("diagnostic", units="share", dtype="float")
    artefact_rate_per_mb: float = record_field("diagnostic", units="1/Mb", dtype="float")
    artefact_length_kb: float = record_field("diagnostic", units="kb", dtype="float")
    at_grid_edge: bool = record_field("diagnostic", dtype="bool")
    contigs: tuple[str, ...] = record_field("obs", axes=("C",), dtype="str")
    per_chromosome_cm_per_mb: NDArray[np.float64] = record_field(
        "diagnostic", axes=("C",), units="cM/Mb", dtype="float64"
    )
    family: NDArray[np.int64] = record_field("diagnostic", axes=("R",), dtype="int64")
    family_members: NDArray[np.int64] = record_field("diagnostic", axes=("R",), units="count", dtype="int64")
    family_model: tuple[str, ...] = record_field("diagnostic", axes=("R",), dtype="str")
    family_pairs: NDArray[np.int64] = record_field("diagnostic", axes=("R",), units="count", dtype="int64")
    family_sites: NDArray[np.float64] = record_field("diagnostic", axes=("R",), units="site", dtype="float64")
    family_estimate_cm_per_mb: NDArray[np.float64] = record_field(
        "diagnostic", axes=("R",), units="cM/Mb", dtype="float64"
    )
    family_status: tuple[str, ...] = record_field("diagnostic", axes=("R",), dtype="str")
    family_inconsistency: NDArray[np.float64] = record_field("diagnostic", axes=("R",), units="nats", dtype="float64")
    gate_passed: bool = record_field("call", dtype="bool")
    gate_reason: str = record_field("diagnostic", dtype="str")
    supplied_rate_cm_per_mb: float = record_field("obs", units="cM/Mb", dtype="float")
    rate_estimate: str = record_field("obs", dtype="str")
    map_free_contigs: tuple[str, ...] = record_field("obs", dtype="str")
    decision: str = record_field("call", dtype="str")
    decision_reason: str = record_field("diagnostic", dtype="str")
    working_rate_cm_per_mb: float = record_field("call", units="cM/Mb", dtype="float")
    warning: str = record_field("report", dtype="str")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SourceStructure:
    """The two-source ancestry model fitted before the family fit (rl21); absent with one source population.

    ``sample_shares`` [N, 3]: each sample's estimated shares of the ancestry states (both homologues from source
    A, one from each, both from B; which source is called A is arbitrary). ``shares`` [T]: the homologue types'
    shares of source B; ``classes`` [K, 2]: each latent-parent class's two types. ``slot_class`` [F, 2]: the most
    probable class of each family's latent slots under the family's class posterior (-1 for a sequenced slot),
    ``slot_class_probability`` [F] that class pair's posterior probability. ``differentiated_fraction``: the share
    of family sites whose source frequencies differ by more than the reporting bound (a description; no site is
    removed). ``loglik``: the fit's log likelihood on its ``fit_sites`` sites after ``iterations`` EM iterations;
    ``start_spread``: the largest difference between the seeded starts' log likelihoods (0 when they agree).
    ``strong_fraction``: the share of family sites that are strongly differentiated between the sources, and
    ``depth``: the cohort's estimated reads per family site and sample; ``warnings``: the texts raised from them
    (the cohort does not look like a two-source cross or its sources are not identified; the depth is below what
    the setting was validated at), reported in the run log and the report. No result depends on them.
    """

    populations: int = record_field("obs", units="count", dtype="int")
    shares: tuple[float, ...] = record_field("obs", axes=("T",), units="share", dtype="tuple[float, ...]")
    classes: NDArray[np.int64] = record_field("obs", axes=("K", "2"), units="1", dtype="int64")
    sample_shares: NDArray[np.float64] = record_field("call", axes=("N", "3"), units="share", dtype="float64")
    slot_class: NDArray[np.int64] = record_field("diagnostic", axes=("F", "2"), units="1", dtype="int64")
    slot_class_probability: NDArray[np.float64] = record_field(
        "diagnostic", axes=("F",), units="probability", dtype="float64"
    )
    differentiated_fraction: float = record_field("diagnostic", units="share", dtype="float")
    fit_sites: int = record_field("diagnostic", units="site", dtype="int")
    loglik: float = record_field("diagnostic", units="nats", dtype="float")
    start_spread: float = record_field("diagnostic", units="nats", dtype="float")
    iterations: int = record_field("state", units="count", dtype="int")
    strong_fraction: float = record_field("diagnostic", units="share", dtype="float", default=float("nan"))
    depth: float = record_field("diagnostic", units="reads", dtype="float", default=float("nan"))
    warnings: tuple[str, ...] = record_field("report", dtype="tuple[str, ...]", default=())

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class RelatednessFamilies:
    """Full-sib families with aligned parent-slot/support and half-sib tables.

    Families are numbered by smallest member; half_sib and gains share one sorted
    row order. ``parents`` holds sequenced slot samples whose genotypes serve the family model as a parent's
    would. A slot sample may be a parent, child (the two directions tie for a one-member family), sib or another
    close relative. Parent calls, direction and their support belong to the pedigree stage.
    NaN slot support denotes a latent slot, and search_margin may be
    NaN when the alternate search was not measured. Canonicalisation is RL1 science.
    member_support / parent_support (FamilySupport, ParentSlot*Support): Morgan-weighted shares of the
    ``support_units`` pooled 1-Morgan units of the family sites' working map (DESIGN_417 D5).
    ``rate`` is the genome-wide recombination-rate estimate from these families (rl20); records built outside the
    relatedness stage may omit it. ``sources`` is the ancestry model of a cohort declared to descend from two source
    populations (rl21), None with one.
    """

    schema: int = schema_field(3)
    sample_ids: tuple[str, ...] = record_field("obs", axes=("N",), units="1", dtype="str")
    contigs: tuple[str, ...] = record_field("obs", axes=("C",), units="1", dtype="str")
    family: NDArray[np.int64] = record_field("call", axes=("N",), units="1", dtype="int64")
    parents: NDArray[np.int64] = record_field("call", axes=("F", "2"), units="1", dtype="int64")
    member_support: NDArray[np.float64] = record_field("call", axes=("N",), units="share", dtype="float64")
    parent_support: NDArray[np.float64] = record_field("call", axes=("F", "2"), units="share", dtype="float64")
    half_sib: NDArray[np.int64] = record_field("call", axes=("H", "2"), units="1", dtype="int64")
    half_sib_gain: NDArray[np.float64] = record_field("call", axes=("H",), units="nats", dtype="float64")
    loglik: float = record_field("diagnostic", axes=(), units="nats", dtype="float")
    sweeps: int = record_field("state", axes=(), units="count", dtype="int")
    converged: bool = record_field("state", axes=(), units="1", dtype="bool")
    family_sites: int = record_field("diagnostic", axes=(), units="site", dtype="int")
    chromosome_sites: tuple[int, ...] = record_field("diagnostic", axes=(), units="site", dtype="tuple[int, ...]")
    support_units: int = record_field("diagnostic", axes=(), units="1", dtype="int", default=0)
    search: str = record_field("state", axes=(), units="1", dtype="str", default="")
    search_margin: float = record_field("diagnostic", axes=(), units="nats", dtype="float", default=float("nan"))
    rate: RecombinationRateEstimate | None = record_field(
        "diagnostic", axes=(), units="1", dtype="RecombinationRateEstimate", default=None
    )
    sources: SourceStructure | None = record_field(
        "diagnostic", axes=(), units="1", dtype="SourceStructure", default=None
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class RelatednessDiagnostics:
    """Measured family-fit summaries; termination and fallback remain in families."""

    loglik: float = record_field("diagnostic", axes=(), units="nats", dtype="float")
    family_sites: int = record_field("diagnostic", axes=(), units="site", dtype="int")
    chromosome_sites: tuple[int, ...] = record_field("diagnostic", axes=(), units="site", dtype="tuple[int, ...]")
    support_units: int = record_field("diagnostic", axes=(), units="1", dtype="int", default=0)
    search_margin: float = record_field("diagnostic", axes=(), units="nats", dtype="float", default=float("nan"))

    def __post_init__(self) -> None:
        freeze_record_arrays(self)
