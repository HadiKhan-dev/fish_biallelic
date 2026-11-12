"""Separate family calls, optional pair reports, and their immutable reader view.

Separate products prevent optional pair reporting from changing the family product.
Only record definitions and immutable product views live here.
No relatedness estimation, canonicalisation, storage, or optional-report imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .roles import freeze_record_arrays, record_field, schema_field


@dataclass(frozen=True)
class RelatednessFamilies:
    """Full-sib families with aligned parent/support and half-sib tables.

    Families are numbered by smallest member; half_sib and gains share one sorted
    row order. NaN parent support denotes a latent slot, and search_margin may be
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
class RelatednessPairs:
    """Optional pairwise report; no family inference consumes these arrays."""

    schema: int = schema_field(1)
    pairs: NDArray[np.int64] = record_field("report", axes=("P", "2"), units="1", dtype="int64")
    ibd: NDArray[np.float64] = record_field("report", axes=("P", "3"), units="probability", dtype="float64")
    ibd_sites: NDArray[np.float64] = record_field("report", axes=("P",), units="site", dtype="float64")
    lr_ibd0: NDArray[np.float64] = record_field("report", axes=("P",), units="nats", dtype="float64")
    lr_ibd2: NDArray[np.float64] = record_field("report", axes=("P",), units="nats", dtype="float64")

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


@dataclass(frozen=True)
class Relatedness:
    """Reader view of independent family and optional pair products; never stored.

    Reconstructed by results from families and, only when requested, pairs. Arrays
    belong to those immutable products and have their lifetime.
    """

    __payload_view__: ClassVar[bool] = True
    families: RelatednessFamilies = record_field("call", dtype="RelatednessFamilies", stored=False)
    pairs: RelatednessPairs | None = record_field("report", dtype="RelatednessPairs | None", stored=False, default=None)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.families.sample_ids

    @property
    def contigs(self) -> tuple[str, ...]:
        return self.families.contigs

    @property
    def family(self) -> NDArray[np.int64]:
        return self.families.family

    @property
    def member_support(self) -> NDArray[np.float64]:
        return self.families.member_support

    @property
    def parents(self) -> NDArray[np.int64]:
        return self.families.parents
