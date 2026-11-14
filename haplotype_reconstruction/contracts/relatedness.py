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
    ``adopted`` (contigs without a map file use ``working_rate_cm_per_mb``, which then differs from the supplied
    rate) or ``kept`` (the supplied map is the working map). NaN marks an estimate or interval that does not exist.
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
    infer_rate: bool = record_field("obs", dtype="bool")
    map_free_contigs: tuple[str, ...] = record_field("obs", dtype="str")
    decision: str = record_field("call", dtype="str")
    decision_reason: str = record_field("diagnostic", dtype="str")
    working_rate_cm_per_mb: float = record_field("call", units="cM/Mb", dtype="float")
    warning: str = record_field("report", dtype="str")

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
    relatedness stage may omit it.
    """

    schema: int = schema_field(2)
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
