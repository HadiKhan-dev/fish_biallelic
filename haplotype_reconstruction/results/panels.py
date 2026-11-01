"""Expose founder stage calls and row paths before export's withholding or renaming.

Owns the StagePanel reader view required by WP_EVA1 section 4.5. No genotype
threshold or release rule is applied here: calls are those already released by
the original stage on its own component, founder row and site axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..contracts.roles import record_field
from ..core.errors import InvariantError

__all__ = ["StagePanel", "FOUNDER_STAGES"]

FOUNDER_STAGES = (
    "discovery",
    "local_selection_initial",
    "local_selection_l1",
    "local_selection_l2",
    "assembly_l1",
    "assembly_l2",
    "assembly_l3",
    "assembly_l4",
    "founder_refinement",
    "painting",
    "founder_refit",
)


@dataclass(frozen=True)
class StagePanel:
    """Read-only original axes: bp positions[L], hard calls[K,L], optional paths[K,B].

    Atomic offsets[B] index the first site of each block. Arrays retain their exact
    dtypes and are borrowed through read-only views, without changing caller flags.
    Malformed dimensions, dtypes or paths raise InvariantError; no calls are altered.
    """

    positions: NDArray[np.int64] = record_field("obs", axes=("L",), units="bp", dtype="int64", stored=False)
    calls: NDArray[np.int8] = record_field("call", axes=("K", "L"), dtype="int8", stored=False)
    atomic_offsets: NDArray[np.int64] | None = record_field(
        "state", axes=("B",), units="index", dtype="int64|None", stored=False, default=None
    )
    rows: NDArray[np.integer] | None = record_field(
        "state", axes=("K", "B"), units="index", dtype="int|None", stored=False, default=None
    )

    def __post_init__(self) -> None:
        if self.positions.dtype != np.int64 or self.positions.ndim != 1:
            raise InvariantError("stage-panel positions must be int64[L] bp")
        if self.calls.dtype != np.int8 or self.calls.ndim != 2 or self.calls.shape[1] != self.positions.size:
            raise InvariantError("stage-panel calls must be int8[K,L] on the original site axis")
        if (self.atomic_offsets is None) != (self.rows is None):
            raise InvariantError("stage-panel offsets and row paths must be supplied together")
        if self.atomic_offsets is not None and self.rows is not None:
            if self.atomic_offsets.dtype != np.int64 or self.atomic_offsets.ndim != 1:
                raise InvariantError("stage-panel atomic_offsets must be int64[B]")
            if self.rows.dtype.kind not in "iu" or self.rows.shape != (len(self.calls), len(self.atomic_offsets)):
                raise InvariantError("stage-panel row paths must be integer[K,B]")
        for name in ("positions", "calls", "atomic_offsets", "rows"):
            array = getattr(self, name)
            if array is not None:
                view = array.view()
                view.flags.writeable = False
                object.__setattr__(self, name, view)
