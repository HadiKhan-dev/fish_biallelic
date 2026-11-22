"""Founder-refit records and explicit derived-panel storage boundary (FR units).

Released/inference alleles are stored arrays; ComponentPanel is rebuilt from the
painting record by the adapter. No refit computation, I/O or painting imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.errors import InvariantError
from .panels import ComponentPanel
from .roles import freeze_record_arrays, record_field, schema_field


@dataclass(frozen=True)
class ComponentRefit:
    """Refit calls, inference snapshot and row-major changed cells.

    changed_rows holds int64 row indices.
    panel is a derived view reconstructed from the painted panel plus these
    released/inference arrays; it is never stored.
    """

    released: NDArray[np.int8] = record_field("call", axes=("K", "L"), units="1", dtype="int8")
    inference: NDArray[np.int8] = record_field("state", axes=("K", "L"), units="1", dtype="int8")
    changed_rows: NDArray[np.int64] = record_field("report", axes=("Q",), units="1", dtype="int64")
    changed_sites: NDArray[np.int64] = record_field("report", axes=("Q",), units="site", dtype="int64")
    gain_nats: NDArray[np.float64] = record_field("report", axes=("Q",), units="nats", dtype="float64")
    filled: NDArray[np.bool_] = record_field("report", axes=("Q",), units="1", dtype="bool")

    panel: ComponentPanel | None = record_field("state", dtype="ComponentPanel | None", stored=False, default=None)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class FounderRefit:
    """One contig refit; no changes to the immutable painting or pedigree inputs."""

    schema: int = schema_field(1)
    contig: str = record_field("obs", axes=(), units="1", dtype="str")
    sample_ids: tuple[str, ...] = record_field("obs", axes=("N",), units="1", dtype="str")
    components: tuple[ComponentRefit, ...] = record_field("call", axes=("C",), units="1", dtype="ComponentRefit")
    excluded_samples: tuple[str, ...] = record_field("state", axes=(), units="1", dtype="str")
    # 4.21 acceptance rule (ADVICE2 sections 12 and 15.1), per carrier class (science.refit.parameters.CARRIER_CLASS_EDGES:
    # one carrier, 2, 3-4, 5-8, 9-16, 17-32, 33+): start-state called cells and those with a positive gain, the pooled
    # class's discordance coefficient (a working coefficient, not a probability of error) and penalty (+inf / NaN where
    # nothing can depart), whether the coefficient sits at a bound, changed and filled cells; and the residual
    # certificate: sites cut off by the coordinate-sweep guard, single-row moves still eligible and supported after the
    # sweeps, moves refused by support alone (the discarded repairs) and moves eligible after the fills.
    class_cells: np.ndarray | None = record_field(
        "report", axes=("7",), units="1", dtype="int64", default=None, kw_only=True
    )
    class_positive: np.ndarray | None = record_field(
        "report", axes=("7",), units="1", dtype="int64", default=None, kw_only=True
    )
    class_discordance: np.ndarray | None = record_field(
        "report", axes=("7",), units="1", dtype="float64", default=None, kw_only=True
    )
    class_penalty: np.ndarray | None = record_field(
        "report", axes=("7",), units="nats", dtype="float64", default=None, kw_only=True
    )
    class_bound: np.ndarray | None = record_field(
        "report", axes=("7",), units="1", dtype="bool", default=None, kw_only=True
    )
    class_changed: np.ndarray | None = record_field(
        "report", axes=("7",), units="1", dtype="int64", default=None, kw_only=True
    )
    class_filled: np.ndarray | None = record_field(
        "report", axes=("7",), units="1", dtype="int64", default=None, kw_only=True
    )
    guard_sites: int = record_field("report", axes=(), units="1", dtype="int64", default=0, kw_only=True)
    eligible_remaining: int = record_field("report", axes=(), units="1", dtype="int64", default=0, kw_only=True)
    refused_by_support: int = record_field("report", axes=(), units="1", dtype="int64", default=0, kw_only=True)
    eligible_after_fills: int = record_field("report", axes=(), units="1", dtype="int64", default=0, kw_only=True)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)

    @property
    def panels(self) -> tuple[ComponentPanel, ...]:
        """Derived panels, requiring the adapter to reconstruct every component view."""
        result = []
        for component in self.components:
            if component.panel is None:
                raise InvariantError("refit panels were not reconstructed")
            result.append(component.panel)
        return tuple(result)
