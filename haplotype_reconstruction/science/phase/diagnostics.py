"""Count released phase sets and retain native solver diagnostics without execution timing.

The native history-step count is len(history)-1; repeated sweeps() calls each
append their initial objective, so this is deliberately not an iteration count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...contracts.phase import MirroredSolveDiagnostics, PairingDiagnostics, PhaseComponentState
from ...contracts.roles import record_field
from .parameters import PhaseParameters


def phase_set_count(phased: np.ndarray, component_ids: np.ndarray, breaks: list[list[int]]) -> int:
    """Count released sets containing a phased heterozygote, split at components and starts (FP24)."""
    length = len(component_ids)
    edge = np.zeros(length, dtype=bool)
    edge[1:] = component_ids[1:] != component_ids[:-1]
    count = 0
    for sample, sample_breaks in enumerate(breaks):
        sites = phased[sample] & (component_ids >= 0)
        if not sites.any():
            continue
        cut = edge.copy()
        starts = np.asarray(sorted(set(sample_breaks)), dtype=np.int64) + 1
        cut[starts[(starts > 0) & (starts < length)]] = True
        count += len(np.unique(np.cumsum(cut)[sites]))
    return count


def solve_diagnostics(
    state: PhaseComponentState, parameters: PhaseParameters
) -> tuple[MirroredSolveDiagnostics, PairingDiagnostics]:
    """Borrow both final histories and count actual pairing decisions at the native ambiguity cut."""
    first, mirror = state.result.solves
    cutoff = -float(np.log(parameters.set_posterior / (1.0 - parameters.set_posterior)))
    mirrored = MirroredSolveDiagnostics(
        len(mirror.history) - 1,
        mirror.settled,
        mirror.history,
        len(state.junctions[1]),
        first.settled,
        len(state.junctions[0]),
    )
    pairings = PairingDiagnostics(
        len(first.pairings),
        len(first.pairing_moves),
        len(mirror.pairing_moves),
        len({(p, j) for solve in (first, mirror) for p, j, gain in solve.pairings if gain > cutoff}),
        sum(len(bins) for bins in state.pairing.values()),
    )
    return mirrored, pairings


@dataclass
class SolveTrace:
    """Task-owned receipts emitted at the actual ascent branches; never used by the solver."""

    sweeps: list[dict[str, Any]] = record_field(
        "diagnostic", units="1", dtype="list[dict[str, Any]]", stored=False, default_factory=list
    )
    passes: list[dict[str, Any]] = record_field(
        "diagnostic", units="1", dtype="list[dict[str, Any]]", stored=False, default_factory=list
    )
    gains: list[list[float]] = record_field(
        "diagnostic", units="1", dtype="list[list[float]]", stored=False, default_factory=list
    )
    cap_events: list[dict[str, Any]] = record_field(
        "diagnostic", units="1", dtype="list[dict[str, Any]]", stored=False, default_factory=list
    )
    initial: dict[str, Any] = record_field(
        "diagnostic", units="1", dtype="dict[str, Any]", stored=False, default_factory=dict
    )
    events: list[dict[str, Any]] = record_field(
        "diagnostic", units="1", dtype="list[dict[str, Any]]", stored=False, default_factory=list
    )
    final_nonkeeping_pass: bool = record_field("diagnostic", units="1", dtype="bool", stored=False, default=False)

    def emit(self, kind: str, outcome: bool | int, **operands: Any) -> None:
        self.events.append(dict(id=kind, event=len(self.events), outcome=int(outcome), operands=operands))

    def finish(self, model, objective):
        """Borrow the final producer state and copy only mutable arrays into the receipt."""
        return dict(
            X=model.X.copy(),
            V=model.V.copy(),
            history=np.asarray(model.history),
            settled=bool(model.settled),
            objective=float(objective),
            sweeps=self.sweeps,
            passes=self.passes,
            gains=np.asarray(self.gains),
            final_nonkeeping_pass=self.final_nonkeeping_pass,
            max_sweeps=model.config.max_sweeps,
            cap_events=self.cap_events,
            initial=self.initial,
        )


@dataclass
class DecisionTrace:
    """Raw actual producer operands with one scalar-event sequence across the entire component.

    The collector never chooses a branch. Producers pass their evaluated boolean
    or selected index and the exact operands; the oracle independently re-decides.
    Vector predicates use C order, at their original point in execution.
    """

    rows: list[dict[str, Any]] = record_field(
        "diagnostic", units="1", dtype="list[dict[str, Any]]", stored=False, default_factory=list
    )
    next_event: int = record_field("diagnostic", units="1", dtype="int", stored=False, default=0)
    extended: bool = record_field("diagnostic", units="1", dtype="bool", stored=False, default=False)

    def emit(self, name, outcome, *operands):
        args = [np.asarray(x).copy().reshape(-1) for x in operands]
        result = np.asarray(outcome).copy().reshape(-1)
        count = result.size
        args = [np.full(count, x.item()) if x.size == 1 and count != 1 else x for x in args]
        if any(x.size != count for x in args):
            raise ValueError("phase trace operand axes differ")
        self.rows.append(
            dict(
                id=name,
                events=np.arange(self.next_event, self.next_event + count, dtype=np.int64),
                outcome=result,
                operands=args,
            )
        )
        self.next_event += count

    def difference(self, name, value, cutoff, outcome):
        self.emit(name, outcome, np.asarray(value, dtype=np.float64) - np.asarray(cutoff, dtype=np.float64))
        self.rows[-1]["raw"] = [np.asarray(value).copy(), np.asarray(cutoff).copy()]

    def argmax(self, pg, selected):
        ordered = np.sort(pg, axis=0)
        self.emit("genotype_argmax", selected, selected, ordered[-1] - ordered[-2])
        self.rows[-1]["raw"] = [pg.copy()]

    def events(self):
        names = {row["id"] for row in self.rows}
        return {name: np.concatenate([r["events"] for r in self.rows if r["id"] == name]) for name in sorted(names)}

    def operands(self):
        values = {}
        for row in self.rows:
            values.setdefault(row["id"], [[] for _ in row["operands"]])
            for dest, value in zip(values[row["id"]], row["operands"], strict=True):
                dest.append(value)
        return {name: [np.concatenate(parts) for parts in args] for name, args in values.items()}


@dataclass(frozen=True)
class PhaseTrace:
    """An actual numerical result and its task-local producer receipts; no alternate algorithm."""

    value: Any = record_field("diagnostic", units="1", dtype="Any", stored=False)
    events: tuple[dict[str, Any], ...] = record_field(
        "diagnostic", units="1", dtype="tuple[dict[str, Any], ...]", stored=False
    )
