"""Discovery products and scientific search status (DI23/25/27).

Search omissions and backstops are state, ambiguity is a call. Timings and tie-event diagnostics live
in science.discovery.diagnostics, outside these products. No I/O or science.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..core.errors import InvariantError
from .roles import freeze_record_arrays, record_field, schema_field

__all__ = ["DiscoveryPlan", "DiscoveryOmission", "DiscoverySearchStatus", "DiscoveredBlock", "DiscoveryBlockResult"]


@dataclass(frozen=True)
class DiscoveryPlan:
    """State of a contig's ordered blocks: indices are exactly range(blocks)."""

    schema: int = schema_field(1)
    blocks: int = record_field("state", units="block", dtype="int")


@dataclass(frozen=True)
class DiscoveryOmission:
    """A whole omitted search class/component, in first-occurrence order (DI25)."""

    kind: str = record_field("state", dtype="str")
    cause: str = record_field("state", dtype="str")
    count: int = record_field("state", dtype="int", units="count")
    scope: str = record_field("state", dtype="str")
    count_kind: str = record_field("state", dtype="str")


@dataclass(frozen=True)
class DiscoverySearchStatus:
    """Scientific search state and released ambiguity (DI25).

    k_values int16[R] and k_log_scores float64[R] describe completely scored K
    classes. Counts, limits, omissions, heuristic cuts, certification, termination
    and mean-field failure counts describe the attempted search. No omitted
    search is diagnostic. score_margin is inf when no competing K exists;
    archive/failure counts may be None when not measured. Timings and tie events
    are supplied separately by the science diagnostics.
    """

    k_values: NDArray[np.int16] = record_field("state", axes=("R",), units="1", dtype="int16")
    k_log_scores: NDArray[np.float64] = record_field("state", axes=("R",), units="nats", dtype="float64")
    natural_k_ceiling: int = record_field("state", axes=(), units="1", dtype="int")
    expansions: int = record_field("state", axes=(), units="1", dtype="int")
    data_starts: int = record_field("state", axes=(), units="1", dtype="int")
    exact_scores: int = record_field("state", axes=(), units="1", dtype="int")
    exact_score_cache_hits: int = record_field("state", axes=(), units="1", dtype="int")
    residual_calls: int = record_field("state", axes=(), units="1", dtype="int")
    residual_rows: int = record_field("state", axes=(), units="1", dtype="int")
    limits: tuple[str, ...] = record_field("state", axes=(), units="1", dtype="tuple[str, ...]")
    heuristic_gauge_cuts: bool = record_field("state", axes=(), units="1", dtype="bool")
    stop_reason: str = record_field("state", axes=(), units="1", dtype="str")
    certified: bool = record_field("state", axes=(), units="1", dtype="bool")
    tie_quantum: float = record_field("state", axes=(), units="nats", dtype="float64", default=0.0)
    ambiguous: bool = record_field("call", axes=(), units="1", dtype="bool", default=False)
    tied_alternatives: int = record_field("call", axes=(), units="1", dtype="int", default=0)
    score_margin: float = record_field("state", axes=(), units="nats", dtype="float64", default=math.inf)
    search_incomplete: bool = record_field("state", axes=(), units="1", dtype="bool", default=False)
    omissions: tuple[DiscoveryOmission, ...] = record_field(
        "state", axes=(), units="1", dtype="tuple[DiscoveryOmission, ...]", default=()
    )
    unresolved: bool = record_field("call", axes=(), units="1", dtype="bool", default=False)
    archive_max_k: int | None = record_field("state", axes=(), units="1", dtype="int | None", default=None)
    selected_mean_field_failures: int | None = record_field(
        "state", axes=(), units="1", dtype="int | None", default=None
    )
    other_mean_field_failures: int | None = record_field("state", axes=(), units="1", dtype="int | None", default=None)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class DiscoveredBlock:
    """One informative block, preserving DI23/25/27 semantics.

    latent int8[K,L_kept] contains the representative's binary rows; latent_unknown
    marks cells with no positively weighted observed carrier. Released calls are
    conservative across alternatives with distinct known content, always at the
    winning K. support is int32, not a posterior. nll is the selected solution's
    fsum negative log likelihood; inf and K=0 denote unresolved search.
    ambiguous/tied_alternatives count the winning set before deduplication and
    are stored calls projected from search (decoding checks consistency).
    latent_unknown=None denotes an unavailable unknown-copy view.
    """

    schema: int = schema_field(1)
    block: int = record_field("obs", axes=(), units="1", dtype="int")
    site_start: int = record_field("obs", axes=(), units="1", dtype="int")
    positions: NDArray[np.int64] = record_field("obs", axes=("L",), units="bp", dtype="int64")
    kept: NDArray[np.bool_] = record_field("obs", axes=("L",), units="1", dtype="bool")
    latent: NDArray[np.int8] = record_field("state", axes=("K", "L_kept"), units="1", dtype="int8")
    calls: NDArray[np.int8] = record_field("call", axes=("K", "L"), units="1", dtype="int8")
    alt_probability: NDArray[np.float64] = record_field("state", axes=("K", "L"), units="probability", dtype="float64")
    support: NDArray[np.int32] = record_field("report", axes=("K", "L"), units="count", dtype="int32")
    unknown_mass: float = record_field("state", axes=(), units="1", dtype="float")
    nll: float = record_field("state", axes=(), units="nats", dtype="float64")
    search: DiscoverySearchStatus = record_field("state", axes=(), units="1", dtype="DiscoverySearchStatus")
    latent_unknown: NDArray[np.bool_] | None = record_field(
        "state", axes=("K", "L_kept"), units="1", dtype="bool", default=None
    )
    tied_panels: tuple[NDArray[np.int8], ...] = record_field(
        "call", axes=("T", "K", "L_kept"), units="1", dtype="tuple[int8]", default=()
    )
    tied_panel_nll: NDArray[np.float64] = record_field(
        "state", axes=("T",), units="nats", dtype="float64", default_factory=lambda: np.empty(0, np.float64)
    )
    tied_unknown: tuple[NDArray[np.bool_], ...] = record_field(
        "call", axes=("T", "K", "L_kept"), units="1", dtype="tuple[bool]", default=()
    )

    ambiguous: bool = record_field("call", dtype="bool", init=False)
    tied_alternatives: int = record_field("call", dtype="int", init=False)
    unresolved: bool = record_field("call", dtype="bool", init=False)

    @property
    def founders(self) -> int:
        return int(self.latent.shape[0])

    def __post_init__(self) -> None:
        object.__setattr__(self, "tied_panel_nll", np.asarray(self.tied_panel_nll, dtype=np.float64))
        for name in ("ambiguous", "tied_alternatives", "unresolved"):
            object.__setattr__(self, name, getattr(self.search, name))
        freeze_record_arrays(self)


@dataclass(frozen=True)
class DiscoveryBlockResult:
    """Stored completion, distinguishing absent information from a scored/search-limited block.

    kind='uninformative' requires result=None; kind='discovered' requires a block.
    An unresolved block remains discovered, with its explicit search.unresolved call.
    """

    schema: int = schema_field(1)
    block: int = record_field("state", dtype="int", units="block")
    kind: Literal["uninformative", "discovered"] = record_field("state", dtype="str")
    result: DiscoveredBlock | None = record_field("state", dtype="DiscoveredBlock | None")

    def __post_init__(self) -> None:
        if self.kind not in ("uninformative", "discovered"):
            raise InvariantError("unknown discovery result kind")
        if (self.result is None) != (self.kind == "uninformative"):
            raise InvariantError("discovery result kind disagrees with its block")
        if self.result is not None and self.result.block != self.block:
            raise InvariantError("discovery result block index disagrees")
