"""Typed local-selection boundaries with 3.5 field meanings (LS inventories).

Fits retain int64 panels; released local calls are int8. Scientific backstops and
selection termination remain state. Context proposal views repeat multiplicity
exactly as 3.5; no likelihood, search, storage or orchestration lives here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.ties import TieCounts
from .panels import LocalPanel
from .roles import freeze_record_arrays, record_field, schema_field

__all__ = [
    "PathFit",
    "RoundTrace",
    "SearchOutcome",
    "PathRelease",
    "BankSummary",
    "ContextProposal",
    "RefitSummary",
    "PackedPanel",
    "PackedFit",
    "SelectionMemo",
    "MemoHits",
    "SelectionSummary",
    "BlockSelection",
    "SelectionProposals",
    "SelectionSelected",
]


@dataclass(frozen=True)
class PathFit:
    """A fitted (panel, frequencies) of a block.

    panel: int64[K, L] binary rows in canonical (lexicographic) order; frequencies: float64[K]
    (sum 1). allele_probability: float64[K, L] probability of ALT per founder allele (the exact
    one-bit conditional posterior); occupancy: float64[K, L] expected copies per site (both are
    None while a fit is restored from a memo). log_likelihood, objective (log_likelihood +
    regularizer): nats. converged: ended at a conditional fixed point; termination:
    "conditional_fixed_point", "row_collision_requires_outer_selection", "update_backstop", or
    (release 2.2) "tied_flips" / "tied_exact_flips": the best moves tied and led to different panels,
    so the ascent stopped (keep-current; converged is True). termination always describes the trajectory.
    selected: "endpoint" (the trajectory's last state) or "trajectory_best" (an earlier state that the
    caller's selection objective prefers by more than the decision tolerance; it is then refitted in its
    frequencies at the fixed panel and is NOT a conditional fixed point: converged is False).
    selected_update: the GEM update after which the selected state was reached (-1: the start).
    endpoint_selection, selected_selection: the selection objective of the endpoint and of the returned
    state (None without a selection objective).
    context_snaps (release 2.12): one-bit flips of a context round's fit taken by the selection objective
    after the GEM (``_context_snaps``); a fit with snaps is not a conditional fixed point (converged False).
    snap_ties: the snap steps decided inside a tie class (by content).
    """

    panel: NDArray[np.int64] = record_field("state", axes=("K", "L_kept"), units="1", dtype="int64")
    frequencies: NDArray[np.float64] = record_field("state", axes=("K",), units="probability", dtype="float64")
    allele_probability: NDArray[np.float64] | None = record_field(
        "state", axes=("K", "L_kept"), units="probability", dtype="float64"
    )
    occupancy: NDArray[np.float64] | None = record_field("state", axes=("K", "L_kept"), units="copies", dtype="float64")
    log_likelihood: float = record_field("state", axes=(), units="nats", dtype="float64")
    objective: float = record_field("state", axes=(), units="nats", dtype="float64")
    converged: bool = record_field("state", axes=(), units="1", dtype="bool")
    termination: str = record_field("state", axes=(), units="1", dtype="str")
    selected: str = record_field("state", axes=(), units="1", dtype="str", default="endpoint")
    selected_update: int = record_field("state", axes=(), units="1", dtype="int", default=-1)
    endpoint_selection: float | None = record_field(
        "state", axes=(), units="nats", dtype="float64 | None", default=None
    )
    selected_selection: float | None = record_field(
        "state", axes=(), units="nats", dtype="float64 | None", default=None
    )
    context_snaps: int = record_field("state", axes=(), units="1", dtype="int", default=0)
    snap_ties: int = record_field("state", axes=(), units="1", dtype="int", default=0)

    @property
    def update_backstop(self) -> bool:
        return self.termination == "update_backstop"

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class RoundTrace:
    """One search round: proposals per kind, refits, K before and of the best refit, its gain,
    kind and rank within its kind."""

    proposals: tuple[tuple[str, int], ...] = record_field(
        "state", axes=(), units="1", dtype="tuple[tuple[str, int], ...]"
    )
    refits: int = record_field("state", axes=(), units="1", dtype="int")
    old_k: int = record_field("state", axes=(), units="1", dtype="int")
    best_k: int = record_field("state", axes=(), units="1", dtype="int")
    gain: float = record_field("state", axes=(), units="nats", dtype="float64")
    kind: str = record_field("state", axes=(), units="1", dtype="str")
    winner_rank: int = record_field("state", axes=(), units="1", dtype="int")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SearchOutcome:
    """A block search's result: the fit, its selection objective, rounds, how it started
    ("start", "incumbent" or "reused"), the round backstop, the context founders and free rows,
    the reuse key of its inputs and work counts. ``start_was_incumbent_panel``: the start's fit
    reached the incumbent's panel (release 2.1; the search then continued from the incumbent)."""

    schema: int = schema_field(1)
    fit: PathFit = record_field("state", axes=(), units="1", dtype="PathFit")
    selection_objective: float = record_field("state", axes=(), units="nats", dtype="float64")
    trace: tuple[RoundTrace, ...] = record_field("state", axes=(), units="1", dtype="tuple[RoundTrace, ...]")
    search_start: str = record_field("state", axes=(), units="1", dtype="str")
    round_backstop: bool = record_field("state", axes=(), units="1", dtype="bool")
    context_founders: int | None = record_field("state", axes=(), units="1", dtype="int | None")
    free_rows: int = record_field("state", axes=(), units="1", dtype="int")
    input_key: str = record_field("state", axes=(), units="1", dtype="str")
    work: tuple[tuple[str, int], ...] = record_field(
        "diagnostic", axes=(), units="1", dtype="tuple[tuple[str, int], ...]"
    )
    start_was_incumbent_panel: bool = record_field("state", axes=(), units="1", dtype="bool", default=False)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PathRelease:
    """calls int8[K, L_kept] (-1 withheld); probability float64[K, L_kept] of the current
    allele; supporters int64[K, L_kept] favourable carriers by count; unknown_copy_fraction:
    posterior unknown-state occupancy over all copies (report); has_depth bool[N];
    unknown_slots int8[N]; unknown_mass float; favourable_band_cells: observed carrier cells whose
    gain lies inside the favourable band (diagnostic, release 2.1; 0 in older records)."""

    calls: NDArray[np.int8] = record_field("call", axes=("K", "L_kept"), units="1", dtype="int8")
    probability: NDArray[np.float64] = record_field("state", axes=("K", "L_kept"), units="probability", dtype="float64")
    supporters: NDArray[np.int64] = record_field("state", axes=("K", "L_kept"), units="count", dtype="int64")
    unknown_copy_fraction: float = record_field("report", axes=(), units="probability", dtype="float64")
    has_depth: NDArray[np.bool_] = record_field("state", axes=("N",), units="1", dtype="bool")
    unknown_slots: NDArray[np.int8] = record_field("state", axes=("N",), units="copies", dtype="int8")
    unknown_mass: float = record_field("state", axes=(), units="1", dtype="float")
    favourable_band_cells: int = record_field("diagnostic", axes=(), units="1", dtype="int", default=0)
    context_cells: int = record_field("state", axes=(), units="1", dtype="int", default=0)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class BankSummary:
    """bank_rows: distinct candidate rows; fitted_starts / scored: novel starts fitted and
    panels admitted; selected_k: the BIC-selected K; rounds: neighbourhood rounds; budget_bound:
    a reported backstop was hit (start, score or round backstop). Release 2.2: selected_tied:
    members of the final BIC tie set (all join the bank); starts: distinct path-search starts (the
    cavity-best tie set over all sources). Both default to 1 (2.1 records)."""

    bank_rows: int = record_field("state", axes=(), units="1", dtype="int")
    fitted_starts: int = record_field("state", axes=(), units="1", dtype="int")
    scored: int = record_field("state", axes=(), units="1", dtype="int")
    selected_k: int = record_field("state", axes=(), units="1", dtype="int")
    rounds: int = record_field("state", axes=(), units="1", dtype="int")
    budget_bound: bool = record_field("state", axes=(), units="1", dtype="bool")
    selected_tied: int = record_field("state", axes=(), units="1", dtype="int", default=1)
    starts: int = record_field("state", axes=(), units="1", dtype="int", default=1)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ContextProposal:
    """One block's proposal of a context round.

    covered: a context component covered the block; then ``calls`` are the distinct local rows
    of the component's refitted founders (lexicographic order; -1 uncalled), ``multiplicity``
    the number of the component's assembled founders carrying each, ``posterior_alt``/``support``
    the refit's ALT posterior and carrier support, ``unknown_slots`` the MAP number of background
    copies per sample (2 without depth), ``witnessed`` (release 2.12) where the refit itself released
    the call (site posterior and carrier support passed; not a completion-filled fallback; None in
    records written before 2.12). Otherwise ``calls`` are the block's current selected calls
    and the other fields None.
    """

    block: int = record_field("state", axes=(), units="1", dtype="int")
    covered: bool = record_field("state", axes=(), units="1", dtype="bool")
    calls: NDArray[np.int8] = record_field("state", axes=("R", "L"), units="1", dtype="int8")
    posterior_alt: NDArray[np.float64] | None = record_field(
        "report", axes=("R", "L"), units="probability", dtype="float64", default=None
    )
    support: NDArray[np.float64] | None = record_field(
        "report", axes=("R", "L"), units="expected count", dtype="float64", default=None
    )
    unknown_slots: NDArray[np.int8] | None = record_field(
        "report", axes=("N",), units="copies", dtype="int8", default=None
    )
    multiplicity: NDArray[np.int64] | None = record_field(
        "state", axes=("R",), units="count", dtype="int64", default=None
    )
    witnessed: NDArray[np.bool_] | None = record_field("state", axes=("R", "L"), units="1", dtype="bool", default=None)

    @property
    def founders(self) -> NDArray[np.int8] | None:
        """The context founders of the path objective's code: one local row per assembled founder
        (release 2.12). Founders whose local rows are equal stay separate
        claimers, so a one-call change that makes two founders' rows equal moves the code by an edit,
        not by a whole free row; the proposal panel itself keeps the distinct rows."""
        if not self.covered:
            return None
        if self.multiplicity is None:
            return self.calls
        return np.repeat(self.calls, np.asarray(self.multiplicity, dtype=np.int64), axis=0)

    @property
    def founder_witness(self) -> NDArray[np.bool_] | None:
        """``witnessed`` aligned with ``founders`` (one row per assembled founder); None when absent."""
        if not self.covered or self.witnessed is None:
            return None
        if self.multiplicity is None:
            return self.witnessed
        return np.repeat(self.witnessed, np.asarray(self.multiplicity, dtype=np.int64), axis=0)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class RefitSummary:
    """components refitted, components skipped (block coverage incomplete), released alleles
    that changed a published call, alleles newly called, site-posterior totals, the largest number of
    distinct refitted founder rows in one component (not the dictionary code's claiming slots);
    tie_events (release 2.1, maths.ties TieCounter record; one decision per refitted site): sites
    with a band decision of the release ("ls.refit.half.band": a carrier probability within the
    band of an exact 1/2; "ls.refit.support.band": a founder's support mass within the band of the
    threshold, its count below it)."""

    components: int = record_field("report", axes=(), units="1", dtype="int")
    skipped: int = record_field("report", axes=(), units="1", dtype="int")
    allele_changes: int = record_field("report", axes=(), units="1", dtype="int")
    filled_alleles: int = record_field("report", axes=(), units="1", dtype="int")
    site_posterior: tuple[tuple[str, int], ...] = record_field(
        "report", axes=(), units="1", dtype="tuple[tuple[str, int], ...]"
    )
    max_founders: int = record_field("report", axes=(), units="1", dtype="int")
    tie_events: TieCounts | None = record_field(
        "diagnostic", axes=(), units="1", dtype="TieCounts | None", default=None
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PackedPanel:
    """Exact bit-packed panel rows and the unpadded site count."""

    sites: int = record_field("state", dtype="int", units="site")
    rows: NDArray[np.uint8] = record_field("state", axes=("K", "L_bytes"), dtype="uint8", units="packed bits")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PackedFit:
    """Memo fit: panel=(site count, packbits rows); posteriors are recomputed."""

    panel: PackedPanel = record_field("state", dtype="PackedPanel")
    frequencies: NDArray[np.float64] = record_field("state", axes=("K",), units="probability", dtype="float64")
    log_likelihood: float = record_field("state", axes=(), units="nats", dtype="float64")
    objective: float = record_field("state", axes=(), units="nats", dtype="float64")
    converged: bool = record_field("state", axes=(), units="1", dtype="bool")
    termination: str = record_field("state", axes=(), units="1", dtype="str")
    selected: str = record_field("state", axes=(), units="1", dtype="str", default="endpoint")
    selected_update: int = record_field("state", axes=(), units="1", dtype="int", default=-1)
    endpoint_selection: float | None = record_field(
        "state", axes=(), units="nats", dtype="float64 | None", default=None
    )
    selected_selection: float | None = record_field(
        "state", axes=(), units="nats", dtype="float64 | None", default=None
    )
    context_snaps: int = record_field("state", axes=(), units="1", dtype="int", default=0)
    snap_ties: int = record_field("state", axes=(), units="1", dtype="int", default=0)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SelectionMemo:
    """Cross-round state. Keys are lowercase hex of exact 3.5 content-key bytes.

    Fits keep packed rows, scores are likelihoods, bounds retain proposal order.
    Round initial/l1 only; memo omission in l2 is an explicit storage choice.
    """

    schema: int = schema_field(1)
    fits: Mapping[str, PackedFit] = record_field(
        "state", axes=(), units="1", dtype="mapping[str,PackedFit]", default_factory=dict
    )
    scores: Mapping[str, float] = record_field(
        "state", axes=(), units="nats", dtype="mapping[str,float64]", default_factory=dict
    )
    bounds: Mapping[str, tuple[float, ...]] = record_field(
        "state", axes=(), units="nats", dtype="mapping[str,tuple[float64]]", default_factory=dict
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class MemoHits:
    """Reported deterministic memo hits; no scientific choice reads these counts."""

    fits: int = record_field("diagnostic", axes=(), units="count", dtype="int", default=0)
    scores: int = record_field("diagnostic", axes=(), units="count", dtype="int", default=0)
    bounds: int = record_field("diagnostic", axes=(), units="count", dtype="int", default=0)
    completed: int = record_field("diagnostic", axes=(), units="count", dtype="int", default=0)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SelectionSummary:
    """Typed block outcome; seconds live in run records, scientific fallbacks remain state."""

    incumbent_retained: bool = record_field("state", axes=(), units="1", dtype="bool")
    release_reused: bool = record_field("diagnostic", axes=(), units="1", dtype="bool")
    bank: BankSummary | None = record_field("state", axes=(), units="1", dtype="BankSummary | None")
    attempted_objective: float = record_field("state", axes=(), units="nats", dtype="float64")
    selection_objective: float = record_field("state", axes=(), units="nats", dtype="float64")
    search_start: str = record_field("state", axes=(), units="1", dtype="str")
    search_rounds: int = record_field("state", axes=(), units="count", dtype="int")
    start_was_incumbent_panel: bool = record_field("state", axes=(), units="1", dtype="bool")
    incumbent_margin: float | None = record_field("state", axes=(), units="nats", dtype="float | None")
    round_backstop: bool = record_field("state", axes=(), units="1", dtype="bool")
    update_backstop: bool = record_field("state", axes=(), units="1", dtype="bool")
    trajectory_best: bool = record_field("state", axes=(), units="1", dtype="bool")
    context_snaps: int = record_field("state", axes=(), units="count", dtype="int")
    context_founders: int | None = record_field("state", axes=(), units="count", dtype="int | None")
    free_rows: int = record_field("state", axes=(), units="count", dtype="int")
    starts: int = record_field("state", axes=(), units="count", dtype="int")
    searched_starts: int = record_field("state", axes=(), units="count", dtype="int")
    tied_alternatives: int = record_field("state", axes=(), units="count", dtype="int")
    withheld_by_ties: int = record_field("state", axes=(), units="count", dtype="int")
    context_witnessed: int = record_field("state", axes=(), units="count", dtype="int")
    cut_overflow: int = record_field("state", axes=(), units="count", dtype="int")
    tie_events: TieCounts = record_field(
        "diagnostic", axes=(), units="1", dtype="TieCounts", default_factory=lambda: TieCounts(())
    )
    memo_hits: MemoHits = record_field("diagnostic", axes=(), units="1", dtype="MemoHits", default_factory=MemoHits)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class BlockSelection:
    """One block outcome with immutable selected panel, release and optional next-round memo."""

    schema: int = schema_field(1)
    index: int = record_field("state", axes=(), units="block", dtype="int")
    panel: LocalPanel = record_field("call", axes=(), units="1", dtype="LocalPanel")
    outcome: SearchOutcome = record_field("state", axes=(), units="1", dtype="SearchOutcome")
    release: PathRelease = record_field("call", axes=(), units="1", dtype="PathRelease")
    memo: SelectionMemo | None = record_field("state", axes=(), units="1", dtype="SelectionMemo | None")
    summary: SelectionSummary = record_field("report", axes=(), units="1", dtype="SelectionSummary")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SelectionProposals:
    """Context-round proposals in panel order and refit summary."""

    schema: int = schema_field(1)
    proposals: tuple[ContextProposal, ...] = record_field("state", axes=("B",), units="1", dtype="ContextProposal")
    summary: RefitSummary = record_field("report", axes=(), units="1", dtype="RefitSummary")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SelectionSelected:
    """Completed round: ordered block indices and released panels."""

    schema: int = schema_field(1)
    indices: NDArray[np.int64] = record_field("state", axes=("B",), units="block", dtype="int64")
    panels: tuple[LocalPanel, ...] = record_field("call", axes=("B",), units="1", dtype="LocalPanel")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)
