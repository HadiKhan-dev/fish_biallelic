"""Declare the execution and storage views consumed by the refinement science.

Owns structural protocols, not scheduling, allocations or persistent records.
Method: docs/methods/founder_refinement.md#workspace (RC01, RC04–RC07).
A scope controls kernel threads at the stage boundary; every map returns in input
order and joins workers before propagating failure. Window reads borrow buffers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.contracts.genetic_map import ChromosomeGeneticMap

_T = TypeVar("_T")
_R = TypeVar("_R")

__all__ = [
    "RefinementExecution",
    "EvidenceWindows",
    "ComponentWorkspaceView",
    "RefinementRecords",
]


class RefinementExecution(Protocol):
    """A declared CPU share with scoped kernel execution and ordered task mapping.

    ``sample_work`` is the work of one sample's component messages in site-diplotype cells (sites x K^2), or 0 when
    unknown; it sets how many samples one thread of a sample kernel takes (core.kernels.sample_team)."""

    sample_work: int

    def threads(self) -> int: ...
    def scope(self, cores: int | None = None) -> AbstractContextManager[None]: ...
    def map(self, function: Callable[[_T], _R], items: list[_T]) -> list[_R]: ...
    def thread_map(
        self,
        function: Callable[[_T], _R],
        items: Iterable[_T],
        *,
        item_bytes: Sequence[int] | None = None,
        label: str = "",
    ) -> list[_R]: ...
    def reserve(self, size: int, *, label: str = "") -> AbstractContextManager[None]: ...
    def concurrent(
        self,
        functions: Sequence[Callable[[RefinementExecution], _R]],
        sizes: int | Sequence[int] = 0,
        *,
        label: str = "",
    ) -> Iterator[tuple[int, _R]]: ...


class EvidenceWindows(Protocol):
    """Evidence/cost views per K; leases protect their backing buffers."""

    resident: bool
    capacity: int

    def view(self, founders: int, reverse: bool = False) -> tuple: ...
    def samples_view(self, founders: int, s0: int, s1: int, reverse: bool = False) -> tuple: ...
    def lease(self, founders: int) -> AbstractContextManager[object]: ...
    def sample_slices(self, s0: int, s1: int) -> list[tuple[int, int]]: ...
    def window_bytes(self) -> int: ...


class RefinementRecords(Protocol):
    """A stage-owned, identity-checked scope; science never opens a container."""

    def load(self, name: str) -> Mapping[str, object] | None: ...
    def save(self, name: str, payload: Mapping[str, object]) -> object: ...
    def scoped(self, sub: str, *, nthreads: int | None = None) -> RefinementRecords: ...


class ComponentWorkspaceView(Protocol):
    """Fixed component arrays and storage capabilities used by full fits/counts.

    Arrays are immutable after construction; normalisers are filled lazily by K
    and published under a lock. A window view has the evidence lease's lifetime.
    """

    leaves: Sequence[NDArray[np.int16]]
    offsets: NDArray[np.int64]
    evidence: NDArray[np.float32]
    kept: NDArray[np.bool_]
    partial: NDArray[np.bool_]
    hazards: NDArray[np.float64]
    emissions: Sequence[NDArray[np.float64]]
    count_samples: NDArray[np.int64] | None
    samples: int
    bin_size: int
    penalty: float
    windows: EvidenceWindows
    message_step: int | None
    genetic_map: ChromosomeGeneticMap
    reset_rate: float
    map_probability_floor: float

    @property
    def blocks(self) -> int: ...
    @property
    def count_sample_count(self) -> int: ...
    def normalisers(self, founders: int) -> NDArray[np.float64]: ...
