"""Collect discovery work counts without affecting a scientific decision.

Owns typed diagnostic options, clique records and block receipts (P0 DI29).
Must not read clocks, environment or files; stage tasks supply elapsed times.
Method: docs/methods/discovery.md#search-status. Counts never select a panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...contracts.roles import record_field

if TYPE_CHECKING:
    from .search import SearchSummary

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

from ...core.ties import TieCount, TieCounts

__all__ = [
    "DiagnosticOptions",
    "CliqueStatistic",
    "DiscoveryDiagnostics",
    "BlockDiagnostics",
    "diagnostic_block",
    "diagnostic_phase",
    "backstop_counts",
]


@dataclass(frozen=True)
class DiagnosticOptions:
    """Explicit opt-in for per-pool clique work records; no effect on candidate generation."""

    clique_statistics: bool = False


@dataclass(frozen=True)
class CliqueStatistic:
    """One residual pool's packed intersection work, preserving the DI29 TSV quantities."""

    pool: int = record_field("diagnostic", dtype="int", units="index")
    work: int = record_field("diagnostic", dtype="int", units="intersections")
    overflow: bool = record_field("diagnostic", dtype="bool", units="flag")
    cliques: int = record_field("diagnostic", dtype="int", units="count", default=0)
    enumerated: bool = record_field("diagnostic", dtype="bool", units="flag", default=True)


@dataclass(frozen=True)
class DiscoveryDiagnostics:
    """A block's returned diagnostic record; timings are filled only by the stage boundary."""

    tie_events: TieCounts = record_field("diagnostic", dtype="TieCounts", default_factory=lambda: TieCounts(()))
    fit_schedule_counts: tuple[tuple[str, int], ...] = record_field(
        "diagnostic", dtype="tuple[tuple[str,int]]", default=()
    )
    endpoint_family_counts: tuple[tuple[str, int], ...] = record_field(
        "diagnostic", dtype="tuple[tuple[str,int]]", default=()
    )
    clique_statistics: tuple[CliqueStatistic, ...] = record_field(
        "diagnostic", dtype="tuple[CliqueStatistic]", default=()
    )
    wall_seconds: float | None = record_field("diagnostic", dtype="float|None", units="seconds", default=None)
    cpu_seconds: float | None = record_field("diagnostic", dtype="float|None", units="seconds", default=None)


@dataclass
class BlockDiagnostics:
    """Caller-owned, mutable accumulator of optional clique statistics during one search.

    record() appends only when enabled. snapshot() returns independent immutable records.
    A diagnostic phase is descriptive; timing it is the stage's responsibility.
    """

    options: DiagnosticOptions = field(default_factory=DiagnosticOptions)
    clique_statistics: list[CliqueStatistic] = field(default_factory=list)
    tie_events: TieCounts = field(default_factory=lambda: TieCounts(()))

    def record(self, pool: int, work: int, overflow: bool, cliques: int = 0, enumerated: bool = True) -> None:
        if self.options.clique_statistics:
            self.clique_statistics.append(
                CliqueStatistic(int(pool), int(work), bool(overflow), int(cliques), bool(enumerated))
            )

    def capture_search(
        self, summary: SearchSummary, *, tie_events: Mapping[str, Mapping[str, int]] | None = None
    ) -> None:
        """Copy the search/release's diagnostic counters without retaining mutable state."""
        counts = summary.tie_events if tie_events is None else tie_events
        self.tie_events = TieCounts(tuple(TieCount(kind, **values) for kind, values in sorted(counts.items())))

    fit_schedule_counts: tuple[tuple[str, int], ...] = ()
    endpoint_family_counts: tuple[tuple[str, int], ...] = ()

    def capture_fit_counts(self, counts: Mapping[str, int], endpoint_families: Mapping[str, int] | None = None) -> None:
        """Freeze workspace work counters; these values never feed scientific selection."""
        self.fit_schedule_counts = tuple(sorted((str(key), int(value)) for key, value in counts.items()))
        if endpoint_families is not None:
            self.endpoint_family_counts = tuple(
                sorted((str(key), int(value)) for key, value in endpoint_families.items())
            )

    def snapshot(self) -> DiscoveryDiagnostics:
        return DiscoveryDiagnostics(
            tie_events=self.tie_events,
            clique_statistics=tuple(self.clique_statistics),
            fit_schedule_counts=self.fit_schedule_counts,
            endpoint_family_counts=self.endpoint_family_counts,
        )

    def summary_fields(self) -> dict:
        """No timings originate in science; retained until DI2 finishes the summary split."""
        return {}


@contextmanager
def diagnostic_block(diagnostics: BlockDiagnostics | None = None) -> Iterator[BlockDiagnostics]:
    """Use an explicit block accumulator, allocating an empty one when omitted."""
    yield diagnostics if diagnostics is not None else BlockDiagnostics()


@contextmanager
def diagnostic_phase(name: str, diagnostics: BlockDiagnostics | None = None) -> Iterator[BlockDiagnostics | None]:
    """Mark an algorithmic phase without consulting a clock or implicit task state."""
    yield diagnostics


def backstop_counts(summary) -> dict[str, int]:
    """Return exact block indicators for ranked truncations and omitted tied classes (DI25)."""
    get = summary.get if isinstance(summary, Mapping) else lambda key, default=None: getattr(summary, key, default)
    causes = {item["cause"] if isinstance(item, Mapping) else item.cause for item in get("omissions", ())}
    expansion_ranked = "expansion_ranked_truncation" in causes
    expansion_tied = "expansion_tie_class_at_budget" in causes
    score_ranked = "exact_score_ranked_truncation" in causes
    score_tied = "exact_score_tie_class_at_budget" in causes
    return dict(
        expansion_backstop=int(get("stop_reason") == "expansion_backstop" or expansion_ranked or expansion_tied),
        exact_score_backstop=int(score_ranked or score_tied),
        expansion_ranked_truncation=int(expansion_ranked),
        expansion_tie_class_at_budget=int(expansion_tied),
        exact_score_ranked_truncation=int(score_ranked),
        exact_score_tie_class_at_budget=int(score_tied),
    )
