"""Task-local producer receipts for recombination's actual branch evaluations.

The collector only records operands and the result of a branch already evaluated
by its producer. Oracle code independently evaluates those operands and assigns
margins; science does not import the oracle, inspect source or use global tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ...contracts.recombination import RecombinationSummary
from ...contracts.roles import record_field

T = TypeVar("T")


@dataclass(frozen=True)
class Traced(Generic[T]):
    """One producer value and the ordered operands it actually visited."""

    value: T = record_field("diagnostic", units="1", dtype="T", stored=False)
    audit: RecombinationTrace = record_field("diagnostic", units="1", dtype="RecombinationTrace", stored=False)


@dataclass
class RecombinationTrace:
    """Mutable collector owned by one contig invocation; edge is its current meiosis."""

    events: list[dict[str, Any]] = record_field(
        "diagnostic", units="1", dtype="list[dict[str, Any]]", stored=False, default_factory=list
    )
    profiles: list[dict[str, Any]] = record_field(
        "diagnostic", units="1", dtype="list[dict[str, Any]]", stored=False, default_factory=list
    )
    pairing: list[dict[str, Any]] = record_field(
        "diagnostic", units="1", dtype="list[dict[str, Any]]", stored=False, default_factory=list
    )
    followed: list[Any] = record_field("diagnostic", units="1", dtype="list[Any]", stored=False, default_factory=list)
    axis: dict[str, Any] = record_field(
        "diagnostic", units="1", dtype="dict[str, Any]", stored=False, default_factory=dict
    )
    bin_axis: Any = record_field("diagnostic", units="1", dtype="Any", stored=False, default_factory=list)
    edge: int = record_field("diagnostic", units="1", dtype="int", stored=False, default=-1)

    def emit(self, name: str, outcome: int | bool, **args: Any) -> None:
        """Append one observed branch in execution order, without re-deciding it."""
        self.events.append(
            dict(id="rc1." + name, event=len(self.events), args=args, outcome=int(outcome), edge=self.edge)
        )


@dataclass(frozen=True)
class RecombinationWitness:
    """Typed comparison view of real records and their actual producer receipts."""

    protocol: str = record_field("obs", units="1", dtype="str")
    convention: bool = record_field("state", units="1", dtype="bool")
    records: list[Any] = record_field("call", units="1", dtype="Recombination")
    trace: list[dict[str, Any]] = record_field("state", units="", dtype="producer-events")


__all__ = ["RecombinationSummary", "RecombinationTrace", "Traced", "RecombinationWitness"]
