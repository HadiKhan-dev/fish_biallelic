"""The search state consumed by the count drivers (WP AS4 §6 / AS5 §4).

The contract is frozen before moving the drivers. PanelSearch owns the arrays and
caches; the drivers mutate only the census, history, tags and family lists below.
A panel change always goes through set_panel, preserving canonical slot order.
Concurrent work returns results in input order. Scores are float64[S], and rows
are int64[K,B]. Their sample, founder and block axes are never reordered here.

Owns: the typed search-state protocol shared by count drivers.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: WP_AS4 §6 and WP_AS5 §6.
Identity: Panel writes pass through set_panel; census/history/tag mutations keep the native driver order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import numpy as np

from haplotype_reconstruction.core import ties
from haplotype_reconstruction.core.ties import TieSettings

DEFAULT_TIE_SETTINGS = TieSettings()

if TYPE_CHECKING:
    from haplotype_reconstruction.core.ties import TieCounter

    from ..count_rule.geometry import RowGeometry
    from .certify import PanelForwards
    from .count_step import CountStep

Path = tuple[int, ...]
Panel = list[Path]
PanelKey = tuple[Path, ...]
T = TypeVar("T")
U = TypeVar("U")


def search_tie_policy(
    name: str, quantum_kind: Literal["join_ll", "join_ll_wide", "join_proxy"], settings: TieSettings
) -> ties.TiePolicy:
    """AJ14/AJ21: retain the three native search scales with caller reversal."""
    if quantum_kind == "join_ll":
        return ties.policy(name, kind="join_ll", reverse=settings.reverse)
    if quantum_kind == "join_ll_wide":
        return ties.policy(name, kind="join_ll_wide", reverse=settings.reverse)
    if quantum_kind == "join_proxy":
        return ties.policy(name, kind="join_proxy", reverse=settings.reverse)
    raise ValueError(f"unknown join tie scale: {quantum_kind}")


class SearchState(Protocol):
    """Mutable, task-local search state; no storage or runtime effects are required.

    CountCensus is the frozen output snapshot; count is its working accumulator.
    The lock protects family append operations during duplicate-seed refits.
    Only count_step increments count_steps. No driver writes selected or its
    score arrays directly. claim is a boundary callback, never a science choice.
    """

    selected: Panel
    baseline_scores: np.ndarray
    likelihood: float
    candidates: Panel
    born: set[Path]
    births_frozen: bool
    count_states: set[PanelKey]
    sizes: list[int]
    cleanup_steps: set[int]
    count: dict[str, Any]
    events: list[dict[str, Any]]
    refit_events: dict[str, list[dict[str, Any]]] | None
    release_families: list[dict[str, Any]]
    seed_families: dict[str, list[dict[str, Any]]]
    count_steps: int
    count_backstop: int
    sweep_backstop: int
    birth_sets: int
    block_offsets: np.ndarray
    tie_counter: TieCounter
    tie_settings: TieSettings
    tie_lock: Lock
    geometry: RowGeometry
    near_identity: float
    counter: CountStep
    # The batch linker's path score (None: content alone decides inside tie classes).
    linkage: Callable[[Path], float] | None

    def set_panel(self, panel: Sequence[Path], scores: np.ndarray, likelihood: float) -> None: ...
    def panel_forwards(self, panel: Sequence[Path]) -> PanelForwards: ...
    def certify(self, *args: Any, **kwargs: Any) -> Any: ...
    def concurrent(self, function: Callable[[T], U], items: Sequence[T]) -> list[U]: ...
    def claim(self) -> int: ...
    def prefer_panels(
        self, kind: str, panels: Sequence[Panel], scores: Sequence[float], source: str | None = None
    ) -> int: ...
    def class_order(
        self,
        kind: str,
        items: Sequence[T],
        value: Callable[[T], float],
        content: Callable[[T], Any],
        *,
        census_key: str | None = None,
        scale: float | None = None,
        quantum_kind: Literal["join_ll", "join_ll_wide", "join_proxy"] = "join_ll",
        link: Callable[[T], float | None] | None = None,
        panel: Callable[[T], Sequence[Path]] | None = None,
    ) -> Any: ...
    def class_winner(
        self,
        kind: str,
        items: Sequence[T],
        value: Callable[[T], float],
        content: Callable[[T], Any],
        *,
        census_key: str | None = None,
        scale: float | None = None,
        quantum_kind: Literal["join_ll", "join_ll_wide", "join_proxy"] = "join_ll",
        link: Callable[[T], float | None] | None = None,
        panel: Callable[[T], Sequence[Path]] | None = None,
    ) -> T | None: ...
    def panel_link(self, panel: Sequence[Path]) -> float | None: ...
    def minimum_switch_count(self, panel: Sequence[Path] | None = None) -> int: ...
    def rows(self, paths: Sequence[Path]) -> np.ndarray: ...


__all__ = ["DEFAULT_TIE_SETTINGS", "T", "U", "search_tie_policy", "SearchState"]
