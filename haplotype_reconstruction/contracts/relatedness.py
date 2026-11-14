"""Relatedness family calls and their measured fit summaries.

Only record definitions live here: no relatedness estimation, canonicalisation or storage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .roles import freeze_record_arrays, record_field, schema_field


@dataclass(frozen=True)
class RelatednessFamilies:
    """Full-sib families with aligned parent-slot/support and half-sib tables.

    Families are numbered by smallest member; half_sib and gains share one sorted
    row order. ``parents`` holds sequenced slot samples whose genotypes serve the family model as a parent's
    would. A slot sample may be a parent, child (the two directions tie for a one-member family), sib or another
    close relative. Parent calls, direction and their support belong to the pedigree stage.
    NaN slot support denotes a latent slot, and search_margin may be
    NaN when the alternate search was not measured. Canonicalisation is RL1 science.
    """

    schema: int = schema_field(1)
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
