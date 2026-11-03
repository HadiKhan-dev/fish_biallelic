"""Describe immutable assembly hierarchy, release and refinement boundaries.

Owns refinement inputs, moves and results (RS01–RS12; required RC02/RC04–RC08).
Must not choose storage, run kernels, read checkpoints or select a founder count.
Method: docs/methods/founder_refinement.md#fixed-k-search.

Producer-owned arrays are read-only by contract; a sweep copies founder rows before
editing them. Evidence, coded leaves and surrogate emissions stay in original sample,
site and prepared-row order. No founder relabelling occurs within a sweep. Canonical
ordering belongs to best-of-two selection. These records do not claim convergence
beyond the offered moves of this one sweep.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ..core.ties import TieCensus, TieCounts
from .genetic_map import ChromosomeGeneticMap
from .panels import ComponentPanel, FounderComponent, PreparedPanel
from .roles import freeze_record_arrays, record_field, schema_field

__all__ = ["RefinementContext", "RefinementProblem", "RefinementSweep", "RefinementProposal", "RefinementSelection"]


@dataclass(frozen=True)
class RefinementContext:
    """Fully contained L1 span alphabets, sorted by local (start, stop).

    WP_AS8 4.10 excludes edge-crossing spans. Coverage of B is not required;
    e.g. [(5,15)] is valid for a 20-block component. No absent edge is filled.

    groups: int64[U,2] local half-open block spans. paths[u]: int64[A_u,B_u]
    distinct L1 prepared-row paths, refined before unrefined. No mutation occurs.
    """

    groups: NDArray[np.int64] = record_field("state", axes=("U", "2"), units="block", dtype="int64")
    paths: tuple[NDArray[np.int64], ...] = record_field("state", axes=("U", "A_u", "B_u"), dtype="int64")


@dataclass(frozen=True)
class RefinementProblem:
    """Complete numerical inputs after RC01/RC03 preparation, before a fixed-K sweep.

    Evidence is the float32 assembly scoring copy; uniform cells are uninformative.
    leaves[b] codes known alleles as 0/1 and an unknown local row r as -2-r.
    emissions[b] addresses prepared pairs by max(a,b)*(max(a,b)+1)//2+min(a,b),
    whereas exact objective states enumerate a<=b lexicographically. normalisers
    are the per-sample start/stay constants for this K, included in exact scores.
    No arrays are mutated; output rows have separate ownership.
    """

    rows: NDArray[np.int64] = record_field("state", axes=("K", "B"), dtype="int64")
    evidence: NDArray[np.float32] = record_field("evidence", axes=("N", "L", "3"), dtype="float32")
    leaves: tuple[NDArray[np.int16], ...] = record_field("state", axes=("B", "R_b", "L_b"), dtype="int16")
    offsets: NDArray[np.int64] = record_field("state", axes=("B+1",), units="site", dtype="int64")
    kept: NDArray[np.bool_] = record_field("state", axes=("L",), dtype="bool")
    partial: NDArray[np.bool_] = record_field("state", axes=("L",), dtype="bool")
    hazards: NDArray[np.float64] = record_field("state", axes=("L",), units="reset hazard", dtype="float64")
    emissions: tuple[NDArray[np.float64], ...] = record_field(
        "state", axes=("B", "N", "R_b*(R_b+1)/2", "M_b"), units="nats", dtype="float64"
    )
    normalisers: NDArray[np.float64] = record_field("state", axes=("N",), units="nats", dtype="float64")
    context: RefinementContext | None = record_field("state", dtype="RefinementContext | None")
    penalty: float = record_field("state", units="nats", dtype="float64")
    source_start: int = record_field("state", units="block", dtype="int64")
    bin_size: int = record_field("state", units="site", dtype="int64")
    schema: int = schema_field(1)


@dataclass(frozen=True)
class RefinementProposal:
    """One surrogate proposal, in template then focal order; rows cover [start,stop).

    Founder is the sweep-start label, remapped by exchanges during acceptance.
    All block coordinates and rows are in forward orientation. No mutation occurs.
    """

    kind: Literal["fine", "coarse"] = record_field("state", dtype="str")
    start: int = record_field("state", units="block", dtype="int64")
    stop: int = record_field("state", units="block", dtype="int64")
    founder: int = record_field("state", dtype="int64")
    rows: NDArray[np.int64] = record_field("state", axes=("stop-start",), dtype="int64")


@dataclass(frozen=True)
class RefinementSweep:
    """Result of one sweep, including every steering counter and ordered decision receipt.

    rows retain the search's founder label frame; scores include K's normalisers.
    decision_trace is float64[D,10]: event, junction, founder, partner, start, stop,
    proposal, operand (nats), cutoff (nats), verdict (0/1). Index columns are exact
    integer-valued floats; the adapter splits their comparison roles explicitly.
    Events 0/1 test exchange/content gains; 2 tests a guard residual; 5 tests guard E; 3/4 record an
    accepted exchange/content move. Trace order is execution order, including ties.
    lex_next steers the next sweep's message mode. A computational work stop raises
    WindowWorkLimit and returns no usable result, never a biological alternative.
    """

    rows: NDArray[np.int64] = record_field("state", axes=("K", "B"), dtype="int64")
    scores: NDArray[np.float64] = record_field("state", axes=("N",), units="nats", dtype="float64")
    F0: float = record_field("state", units="nats", dtype="float64")
    F: float = record_field("state", units="nats", dtype="float64")
    gain: float = record_field("state", units="nats", dtype="float64")
    accounted: float = record_field("state", units="nats", dtype="float64")
    deviation: float = record_field("report", units="nats", dtype="float64")
    accepted: int = record_field("state", dtype="int64")
    converged: bool = record_field("state", dtype="bool")
    lex_next: bool = record_field("state", dtype="bool")
    counters: AcceptanceCounters = record_field("state", dtype="AcceptanceCounters")
    proposals: tuple[RefinementProposal, ...] = record_field("state", dtype="tuple[RefinementProposal]")
    proposal_statistics: ProposalCounters = record_field("state", dtype="ProposalCounters")
    window_work: WindowWorkCounters = record_field("state", dtype="WindowWorkCounters")
    window_domains: tuple[WindowDomain, ...] = record_field("state", dtype="tuple[WindowDomain]")
    junction_gain: NDArray[np.float64] = record_field("report", axes=("B+1",), units="nats", dtype="float64")
    junction_pair: NDArray[np.int64] = record_field("report", axes=("B+1", "2"), dtype="int64")
    weak_junctions: tuple[RefinementWeakJunction, ...] = record_field("report", dtype="tuple[WeakJunction]")
    decision_trace: RefinementDecisionTrace = record_field("state", dtype="RefinementDecisionTrace")
    schema: int = schema_field(1)


@dataclass(frozen=True)
class RefinementSelection:
    """RS12 representative of two completed directional searches, with tied alternatives.

    F ties use the refine_F policy, then minimum exact-optimal switches as a prior;
    the arbitrary content convention applies only after that prior. Chosen rows
    and alternatives are canonical, with multiplicity preserved. No input mutates.
    """

    rows: NDArray[np.int64] = record_field("state", axes=("K", "B"), dtype="int64")
    first: Literal["fwd", "rev"] = record_field("state", dtype="str")
    tied_panels: tuple[tuple[tuple[int, ...], ...], ...] = record_field("call", dtype="tuple[PanelKey]")
    tied_scores: tuple[float, ...] = record_field("state", units="nats", dtype="float64")
    tied_switches: tuple[int, ...] = record_field("state", dtype="int64")
    minimum_switches: int | None = record_field("state", dtype="int64 | None")
    schema: int = schema_field(1)


# PILOT-R neutral comparison views. These are adapter views of a RefinementSweep,
# not separately persisted products; their schema is the enclosing sweep's schema.
@dataclass(frozen=True)
class RefinementDecisionTrace:
    """Separate exact trace indices from the two nats columns without losing order."""

    indices: NDArray[np.int64] = record_field("state", dtype="int64")
    operand: NDArray[np.float64] = record_field("state", units="nats", dtype="float64")
    cutoff: NDArray[np.float64] = record_field("state", units="nats", dtype="float64")


@dataclass(frozen=True)
class RefinementWeakJunction:
    """First-look weak junction interval, representative pair and exact gain."""

    first: int = record_field("report", units="block", dtype="int64")
    last: int = record_field("report", units="block", dtype="int64")
    pair: tuple[int, int] = record_field("report", dtype="tuple[int64,int64]")
    gain: float = record_field("report", units="nats", dtype="float64")


class RefinementCounterAccess:
    """Read-only named access for incremental callers of typed numerical records.

    Fields remain explicitly declared and role-bearing. This is not an open
    dictionary; unknown names raise KeyError and no caller can add a field.
    """

    def __getitem__(self, name: str) -> Any:
        try:
            return getattr(self, name)
        except AttributeError as error:
            raise KeyError(name) from error

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    def keys(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cast(Any, self)))


@dataclass(frozen=True)
class AcceptanceCounters(RefinementCounterAccess):
    """All acceptance counters; gain and saved switch-cost sums are measured in nats."""

    fine: int = record_field("state", dtype="int64")
    coarse: int = record_field("state", dtype="int64")
    exchange: int = record_field("state", dtype="int64")
    fine_gain: float = record_field("state", units="nats", dtype="float64")
    coarse_gain: float = record_field("state", units="nats", dtype="float64")
    exchange_gain: float = record_field("state", units="nats", dtype="float64")
    evaluated: int = record_field("state", dtype="int64")
    false_positive: int = record_field("state", dtype="int64")
    guard_rejected: int = record_field("state", dtype="int64")
    guard_shared_tests: int = record_field("state", dtype="int64")
    guard_relaxed: int = record_field("state", dtype="int64")
    guard_shared_switches: float = record_field("state", units="nats", dtype="float64")
    guard_traced_blocks: int = record_field("state", dtype="int64")
    chain_blocks: int = record_field("state", dtype="int64")
    ahead_blocks: int = record_field("state", dtype="int64")
    changed_blocks: int = record_field("state", dtype="int64")
    exchange_tie: int = record_field("state", dtype="int64")
    content_tie: int = record_field("state", dtype="int64")
    tie_wide: int = record_field("state", dtype="int64")
    content_rounds: int = record_field("state", dtype="int64")
    round_cap: int = record_field("state", dtype="int64")
    stale_evaluated: int = record_field("state", dtype="int64")
    exchange_cap: int = record_field("state", dtype="int64")
    lex_items: int = record_field("state", dtype="int64")
    lex_abort: int = record_field("state", dtype="int64")

    bridge: int = record_field("state", dtype="int64")
    bridge_gain: float = record_field("state", units="nats", dtype="float64")
    escape: int = record_field("state", dtype="int64")
    escape_gain: float = record_field("state", units="nats", dtype="float64")
    escape_offers: int = record_field("state", dtype="int64")
    escape_windows: int = record_field("state", dtype="int64")
    escape_beam_cut: int = record_field("state", dtype="int64")
    escape_beam_binding: int = record_field("state", dtype="int64")
    long: int = record_field("state", dtype="int64")
    long_gain: float = record_field("state", units="nats", dtype="float64")
    long_offers: int = record_field("state", dtype="int64")
    long_windows: int = record_field("state", dtype="int64")
    long_beam_cut: int = record_field("state", dtype="int64")
    long_beam_binding: int = record_field("state", dtype="int64")
    long_candidates: int = record_field("state", dtype="int64")


# AS7/AS8: bound window-work identity. This section is owned by pr40.
@dataclass(frozen=True, order=True)
class WindowOrdinal:
    """Bind a template to deterministic driver order before any cost scheduling.

    Fit is the driver invocation ordinal; direction is forward 0/reverse 1;
    sweep is the search sweep; template is its original fine-then-coarse index.
    Convention remains on the request/domain and never changes these ordinals.
    """

    fit: int = record_field("state", dtype="int64")
    direction: int = record_field("state", dtype="int64")
    sweep: int = record_field("state", dtype="int64")
    template: int = record_field("state", dtype="int64")


# The neutral comparison view and the actual product now use the same record.
RefinementAcceptanceCounters = AcceptanceCounters


@dataclass(frozen=True)
class ProposalCounters(RefinementCounterAccess):
    """Outcome-steering proposal counts, including reuse and bounded programme work."""

    windows: int = record_field("state", dtype="int64")
    certified: int = record_field("state", dtype="int64")
    dp: int = record_field("state", dtype="int64")
    fixing_dps: int = record_field("state", dtype="int64")
    max_frontier: int = record_field("state", dtype="int64")
    fallbacks: int = record_field("state", dtype="int64")
    changed: int = record_field("state", dtype="int64")
    templates_solved: int = record_field("state", dtype="int64")
    templates_reused: int = record_field("state", dtype="int64")
    beam_truncations: int = record_field("state", dtype="int64")
    beam_fallbacks: int = record_field("state", dtype="int64")


@dataclass(frozen=True)
class WindowWorkCounters(RefinementCounterAccess):
    """Logical-prefix native work totals; units are expanded Python-integer charges."""

    window_work_limits: int = record_field("state", dtype="int")
    window_work_calls: int = record_field("state", dtype="int")
    window_work_units: int = record_field("state", dtype="int")
    window_work_retries: int = record_field("state", dtype="int")
    window_work_discarded_focals: int = record_field("state", dtype="int")


@dataclass(frozen=True)
class WindowWorkShape:
    """Dimensions defining the native per-template computational budget."""

    samples: int = record_field("state", dtype="int")
    units: int = record_field("state", dtype="int")
    bins: int = record_field("state", dtype="int")
    founders: int = record_field("state", dtype="int")
    alphabet: int = record_field("state", dtype="int")
    height: int = record_field("state", dtype="int")


@dataclass(frozen=True)
class WindowAccount(RefinementCounterAccess):
    """One terminal cumulative account, shared across all focals and retries.

    Expanded work equals raw units times work_quantum. Phase arrays retain all
    sixteen native debit phases. No physical scheduling order enters an account.
    """

    status: int = record_field("state", dtype="int")
    work_limit: bool = record_field("state", dtype="bool")
    work_phase: int = record_field("state", dtype="int")
    work_retries: int = record_field("state", dtype="int")
    work_discarded: int = record_field("state", dtype="int")
    account_scope: str = record_field("state", dtype="str")
    work_budget: int = record_field("state", dtype="int")
    work_used: int = record_field("state", dtype="int")
    phase_work: tuple[int, ...] = record_field("state", axes=("16",), dtype="int")
    work_quantum: int = record_field("state", dtype="int")
    work_used_units: int = record_field("state", dtype="int")
    phase_work_units: tuple[int, ...] = record_field("state", axes=("16",), dtype="int")
    work_coefficient: int = record_field("state", dtype="int")
    work_shape: WindowWorkShape = record_field("state", dtype="WindowWorkShape")


@dataclass(frozen=True)
class WindowRangeAlphabet:
    """A fine window's complete prepared-row range per unit."""

    encoding: str = record_field("state", dtype="str")
    sizes: tuple[int, ...] = record_field("state", dtype="int")


@dataclass(frozen=True)
class WindowPathAlphabet:
    """A coarse window's complete ordered prepared-row path alphabet per unit."""

    encoding: str = record_field("state", dtype="str")
    paths: tuple[tuple[tuple[int, ...], ...], ...] = record_field("state", dtype="int")


@dataclass(frozen=True)
class WindowDomain(RefinementCounterAccess):
    """Whole unfinished conditional domain; never an evaluated biological family."""

    kind: str = record_field("state", dtype="str")
    block_start: int = record_field("state", units="block", dtype="int64")
    block_stop: int = record_field("state", units="block", dtype="int64")
    source_component_start: int = record_field("state", units="block", dtype="int64")
    source_component_stop: int = record_field("state", units="block", dtype="int64")
    source_panel: tuple[tuple[int, ...], ...] = record_field("state", dtype="int64")
    focals: tuple[int, ...] = record_field("state", dtype="int64")
    unit_spans: tuple[tuple[int, int], ...] = record_field("state", units="block", dtype="int64")
    alphabet: WindowRangeAlphabet | WindowPathAlphabet = record_field("state", dtype="WindowAlphabet")
    paths_per_focal: int = record_field("state", dtype="int64")
    convention_reversed: bool = record_field("state", dtype="bool")
    complete_conditional_domain: bool = record_field("state", dtype="bool")
    evaluated: bool = record_field("state", dtype="bool")
    meaning: str = record_field("state", dtype="str")
    account: WindowAccount = record_field("state", dtype="WindowAccount")


__all__ += [
    "AcceptanceCounters",
    "ProposalCounters",
    "WindowWorkCounters",
    "WindowOrdinal",
    "WindowWorkShape",
    "WindowAccount",
    "WindowDomain",
    "WindowRangeAlphabet",
    "WindowPathAlphabet",
    "RefinementAcceptanceCounters",
    "RefinementWeakJunction",
    "RefinementDecisionTrace",
]

# ---------------------------------------------------------------------------
# AS6: assembly hierarchy and founder release (WP_AS6; sole integrator as40c).
# Existing PILOT-R records above retain their fields and semantics.


@dataclass(frozen=True)
class LinkerSettings:
    """Plain linker values crossing the join task boundary; no science imports."""

    max_sites: int = record_field("state", dtype="int64")
    max_iterations: int = record_field("state", dtype="int64")
    robustness: float = record_field("state", dtype="float64")
    error_fraction: float = record_field("state", dtype="float64")
    error_tract_bp: float = record_field("state", units="bp", dtype="float64")
    min_log_probability: float = record_field("state", units="nats", dtype="float64")
    pseudo_count: float = record_field("state", dtype="float64")
    uniform_mix: float = record_field("state", dtype="float64")
    min_change: float = record_field("state", dtype="float64")
    learning_rate_decay: float = record_field("state", dtype="float64")
    minimum_learning_rate: float = record_field("state", dtype="float64")


@dataclass(frozen=True)
class JoinSearchSettings:
    """The four plain search controls carried by a join task."""

    paths_per_endpoint: int = record_field("state", dtype="int64")
    full_scores_per_kind: int = record_field("state", dtype="int64")
    max_bins: int = record_field("state", dtype="int64")
    tensor_budget_mb: int = record_field("diagnostic", units="MiB", dtype="int64")


@dataclass(frozen=True)
class CountRuleSettings:
    """Plain effective-carrier controls, shared by join and refinement tasks."""

    bootstrap_resamples: int = record_field("state", dtype="int64")
    bootstrap_level: float = record_field("state", dtype="float64")
    near_identity: float = record_field("state", dtype="float64")
    shadow_combinations: int = record_field("state", dtype="int64")
    schema: int = schema_field(2)


@dataclass(frozen=True)
class JoinSettings:
    """All scientific settings of one batch, including the explicit tie convention.

    The map covers the task's absolute genomic positions. L1 ignores the horizon
    controls; later levels use them. The three nested records contain values only:
    owning science constructors rebuild their parameter dataclasses from them.
    """

    linker: LinkerSettings = record_field("state", dtype="LinkerSettings")
    join: JoinSearchSettings = record_field("state", dtype="JoinSearchSettings")
    count_rule: CountRuleSettings = record_field("state", dtype="CountRuleSettings")
    cost_scale: float = record_field("state", dtype="float64")
    gap_horizon_scale: float = record_field("state", dtype="float64")
    recombination_tolerance: float = record_field("state", dtype="float64")
    level_one: bool = record_field("state", dtype="bool")
    genetic_map: ChromosomeGeneticMap = record_field("state", dtype="ChromosomeGeneticMap")
    reverse: bool = record_field("state", default=False, kw_only=True, dtype="bool")
    tie_keep: tuple[str, ...] = record_field("state", default=(), kw_only=True, dtype="tuple[str, ...]")
    census: Callable[[TieCensus], None] | None = record_field(
        "diagnostic", default=None, kw_only=True, stored=False, dtype="census capability"
    )


@dataclass(frozen=True)
class JoinInputs:
    """Materialised blocks and float32 evidence on one common join domain.

    blocks are in genomic order; gl[S,L,3] reads pre-fill inference, never released
    fills. offsets[n+1] delimit concatenated blocks. samples[S] are the selected
    sample indices on the enclosing evidence axis, ascending. Arrays are borrowed
    read-only; block_gl returns a view and nothing is mutated.
    """

    blocks: tuple[ComponentPanel, ...] = record_field("state", dtype="tuple[ComponentPanel, ...]")
    gl: NDArray[np.float32] = record_field("evidence", axes=("S", "L", "3"), dtype="float32")
    offsets: NDArray[np.int64] = record_field("state", axes=("n+1",), units="site", dtype="int64")
    samples: NDArray[np.int64] = record_field("state", axes=("S",), dtype="int64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)

    def block_gl(self, index: int) -> NDArray[np.float32]:
        """Return the S by L_index by 3 evidence view for one input component."""
        return self.gl[:, self.offsets[index] : self.offsets[index + 1]]

    @property
    def sample_count(self) -> int:
        """Number of informative samples in the common likelihood population."""
        return int(self.gl.shape[0])


@dataclass(frozen=True)
class ReleaseFamily:
    """One evaluated biological family or an unverified continuation (U13/U16).

    panels are prepared-row matrices on [start,stop). Nested values, replay_panels,
    cycle_excluded_panels, selected_panel, source_choices, shadow and switch keys
    retain the producer's coordinates: join-local paths for joins; prepared-row
    paths for refinement. Re-basing the family never re-bases that provenance.

    The sparse payload view retains 3.5 metadata, including explicit None.
    This is a value record, not a Mapping subclass: its scientific values field
    must not shadow Mapping.values(), which orchestration traversals call.
    This is needed because absence and None differ for fallback status/provenance
    reads. _present preserves sparse source-field presence across typed payload round trips. Unknown
    payload fields are rejected. Arrays and nested source metadata are borrowed
    read-only; dataclasses.replace is the update operation.
    """

    start: int = record_field("state", units="block", dtype="int64")
    stop: int = record_field("state", units="block", dtype="int64")
    panels: tuple[NDArray[np.int32], ...] = record_field("call", axes=("A", "K_a", "B"), dtype="int32")
    reason: str = record_field("state", dtype="str")
    status: str = record_field("state", dtype="str")
    unresolved_search: bool = record_field("state", dtype="bool")
    source_decision: str | None = record_field("state", default=None, dtype="str")
    objective: str | None = record_field("state", default=None, dtype="str")
    values: Any = record_field("state", default=None, dtype="source provenance")
    minimum_switches: Any = record_field("state", default=None, dtype="source provenance")
    representative_prior: str | None = record_field("state", default=None, dtype="str")
    replay_panels: Any = record_field("state", default=None, dtype="source provenance")
    cycle_excluded_panels: Any = record_field("state", default=None, dtype="source provenance")
    selected_panel: Any = record_field("state", default=None, dtype="source provenance")
    source_passes: Any = record_field("state", default=None, dtype="source provenance")
    enumeration_scope: Any = record_field("state", default=None, dtype="source provenance")
    source_replay_complete: bool | None = record_field("state", default=None, dtype="bool")
    source_search_certified: bool | None = record_field("state", default=None, dtype="bool")
    source_choices: Any = record_field("state", default=None, dtype="source provenance")
    shadow: Any = record_field("state", default=None, dtype="source provenance")
    shadow_alternatives: Any = record_field("state", default=None, dtype="source provenance")
    count_unverified: bool | None = record_field("diagnostic", default=None, dtype="bool")
    cause: str | None = record_field("state", default=None, dtype="str")
    domain_size: int | None = record_field("state", default=None, dtype="int")
    evaluated: int | None = record_field("state", default=None, dtype="int")
    scope: Any = record_field("state", default=None, dtype="source provenance")
    source_count_key: Any = record_field("state", default=None, dtype="source provenance")
    source_count_kind: str | None = record_field("state", default=None, dtype="str")
    source_count_round: int | None = record_field("state", default=None, dtype="int")
    source_role: str | None = record_field("state", default=None, dtype="str")
    source_fit: Any = record_field("state", default=None, dtype="source provenance")
    source_reason: str | None = record_field("state", default=None, dtype="str")
    scores: Any = record_field("state", default=None, dtype="source provenance")
    branch: str | None = record_field("state", default=None, dtype="str")
    input_families: Any = record_field("state", default=None, dtype="source provenance")
    joint_combinations: int | None = record_field("state", default=None, dtype="int")
    lost_usable_sites: int | None = record_field("state", default=None, dtype="int")
    usable_site_reference: str | None = record_field("state", default=None, dtype="str")
    source_level: int | None = record_field("state", default=None, dtype="int")
    window_domains: Any = record_field("state", default=None, dtype="source provenance")
    window_domains_scope: Any = record_field("state", default=None, dtype="source provenance")
    domain_basis: Any = record_field("state", default=None, dtype="source provenance")
    _present: tuple[str, ...] = record_field("state", default=(), repr=False, compare=False, dtype="tuple[str, ...]")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)

    def __iter__(self) -> Iterator[str]:
        return iter(
            self._present
            or tuple(
                item.name
                for item in fields(self)
                if not item.name.startswith("_") and (item.default is MISSING or getattr(self, item.name) is not None)
            )
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getitem__(self, key: str) -> Any:
        if key not in tuple(self):
            raise KeyError(key)
        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        """Sparse payload names; values remains the scientific objective field."""
        return tuple(self)

    def items(self) -> tuple[tuple[str, Any], ...]:
        return tuple((key, self[key]) for key in self)

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in tuple(self) else default

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | ReleaseFamily) -> ReleaseFamily:
        """Decode known sparse source fields; reject unrecognised provenance."""
        names = {item.name for item in fields(cls) if not item.name.startswith("_")}
        unknown = set(value) - names
        if unknown:
            raise ValueError(f"ReleaseFamily has unknown fields: {sorted(unknown)}")
        required: dict[str, Any] = dict(panels=(), reason="", status="", unresolved_search=False)
        required.update(value)
        return cls(**required, _present=tuple(value))

    def updated(self, **values: Any) -> ReleaseFamily:
        """Return an updated family, preserving source field order and presence."""
        data = dict(self)
        data.update(values)
        return self.from_mapping(data)

    @property
    def kind(self) -> Literal["biological", "unverified", "diagnostic"]:
        """Classify by the published status; count-proof diagnostics are separate."""
        return release_family_kind(self.status)


def release_family_kind(status: str) -> Literal["biological", "unverified", "diagnostic"]:
    """Return a family's comparison kind from its declared status (WP_AS6 §5)."""
    if status == "count_unverified":
        return "diagnostic"
    if status.startswith("unresolved") or status == "continuation_unverified":
        return "unverified"
    return "biological"


@dataclass(frozen=True)
class JoinTask:
    """One batch over relative prepared-panel ranges and a shared sample domain.

    Samples index the enclosing scoring array. Persisted identity normalises
    samples and site_start to a batch-local axis and binds its evidence content.
    Common masks apply even for a single configuration and branch overflow.
    """

    index: int = record_field("diagnostic", dtype="int64")
    components: tuple[FounderComponent, ...] = record_field("state", dtype="tuple[FounderComponent, ...]")
    prepared: tuple[PreparedPanel, ...] = record_field("state", dtype="tuple[PreparedPanel, ...]")
    samples: NDArray[np.int64] = record_field("state", axes=("S",), dtype="int64")
    settings: JoinSettings = record_field("state", dtype="JoinSettings")
    branches: tuple[tuple[FounderComponent, ...], ...] = record_field(
        "state", default=(), dtype="tuple[tuple[FounderComponent, ...], ...]"
    )
    branch_sources: tuple[ReleaseFamily, ...] = record_field("state", default=(), dtype="tuple[ReleaseFamily, ...]")
    common_mask_lost: int = record_field("state", default=0, dtype="int64")
    common_usable: tuple[NDArray[np.bool_], ...] = record_field(
        "state", default=(), dtype="tuple[NDArray[np.bool_], ...]"
    )
    common_usable_lost: int = record_field("state", default=0, dtype="int64")
    overflow: bool = record_field("state", default=False, dtype="bool")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


class JoinSummaryView(Protocol):
    """Diagnostic/steering fields consumed when complete branch results combine (read-only: JoinSummary is frozen).

    ``tie_events``: a fresh search's TieCounter record (TieCounts); a decoded summary's plain mapping.
    """

    @property
    def phase_seconds(self) -> Mapping[str, float]: ...
    @property
    def work_counters(self) -> Mapping[str, int]: ...
    @property
    def total_full_scores(self) -> int: ...
    @property
    def count_exact_scores(self) -> int: ...
    @property
    def count_exact_sample_scores(self) -> int: ...
    @property
    def beam_seconds(self) -> float: ...
    @property
    def events(self) -> tuple[Mapping[str, Any], ...]: ...
    @property
    def tie_events(self) -> TieCounts | Mapping[str, Any]: ...

    def record(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class JoinRecord:
    """A completed batch's prepared-row paths and ordered release families."""

    index: int = record_field("diagnostic", dtype="int64")
    rows: NDArray[np.int32] = record_field("state", axes=("K", "B"), dtype="int32")
    log_likelihood: float = record_field("state", units="nats", dtype="float64")
    summary: JoinSummaryView | None = record_field("state", dtype="JoinSummary")
    seconds: float = record_field("diagnostic", units="s", default=0.0, dtype="float64")
    release_alternatives: tuple[ReleaseFamily, ...] = record_field(
        "state", default=(), dtype="tuple[ReleaseFamily, ...]"
    )
    minimum_switches: int | None = record_field("state", default=None, dtype="int")
    shadow: Mapping[str, Any] | None = record_field("state", default=None, dtype="Mapping[str, Any] | None")
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        rows = self.rows.view()
        rows.flags.writeable = False
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True)
class LevelRecord:
    """One level's plan and pre-refinement stop decision, in batch order (U02/U15)."""

    level: int = record_field("state", dtype="int64")
    batch_size: int = record_field("state", dtype="int64")
    components_before: int = record_field("state", dtype="int64")
    components_after: int = record_field("state", dtype="int64")
    stop_reason: str | None = record_field("state", dtype="str")
    plan: tuple[tuple[int, int, str | None], ...] = record_field(
        "state", dtype="tuple[tuple[int, int, str | None], ...]"
    )
    joins: tuple[JoinSummaryView, ...] = record_field("state", dtype="JoinSummary", axes=("J",))
    log_likelihoods: tuple[float, ...] = record_field("state", units="nats", dtype="tuple[float, ...]")
    reused_joins: int = record_field("diagnostic", dtype="int64")
    boundary_joint: tuple[int, ...] = record_field("state", dtype="tuple[int, ...]")
    refinement: Mapping[str, Any] | None = record_field("state", default=None, dtype="LevelSummary payload")
    elapsed_seconds: float = record_field("diagnostic", units="s", default=0.0, dtype="float64")
    refinement_elapsed_seconds: float | None = record_field("diagnostic", units="s", default=None, dtype="float64")
    shadows: tuple[Mapping[str, Any], ...] = record_field(
        "diagnostic", default=(), dtype="tuple[Mapping[str, Any], ...]"
    )
    schema: int = schema_field(1)


@dataclass(frozen=True)
class ReleaseAxes:
    """Prepared block range, row paths and atomic offsets of one refit component."""

    start: int = record_field("state", units="block", dtype="int64")
    stop: int = record_field("state", units="block", dtype="int64")
    rows: NDArray[np.int32] = record_field("state", axes=("K", "B"), dtype="int32")
    atomic_offsets: NDArray[np.int64] = record_field("state", axes=("B",), units="site", dtype="int64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ReleaseSegment:
    """One released interval, including its source row order and withholding counts."""

    positions: NDArray[np.int64] = record_field("obs", axes=("L",), units="bp", dtype="int64")
    calls: NDArray[np.int8] = record_field("call", axes=("K", "L"), dtype="int8")
    source_component: int = record_field("call", dtype="int64")
    segment_index: int = record_field("call", dtype="int64")
    block_start: int = record_field("call", units="block", dtype="int64")
    block_stop: int = record_field("call", units="block", dtype="int64")
    source_founder_rows: tuple[int, ...] = record_field("call", dtype="tuple[int, ...]")
    ambiguity: tuple[Mapping[str, Any], ...] = record_field("call", dtype="tuple[Mapping[str, Any], ...]")
    continuation_unverified: tuple[Mapping[str, Any], ...] = record_field("call", dtype="tuple[Mapping[str, Any], ...]")
    count_unverified: tuple[Mapping[str, Any], ...] = record_field("diagnostic", dtype="tuple[Mapping[str, Any], ...]")
    withheld_ambiguous_alleles: int = record_field("call", dtype="int64")
    r5_withheld_rows: tuple[int, ...] = record_field("call", dtype="tuple[int, ...]")
    first_position: int = record_field("report", units="bp", dtype="int64")
    last_position: int = record_field("report", units="bp", dtype="int64")
    markers: int = record_field("report", dtype="int64")
    called_alleles: int = record_field("report", dtype="int64")
    missing_alleles: int = record_field("report", dtype="int64")
    withheld_unobserved_alleles: int = record_field("report", dtype="int64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class FounderIdentityFlag:
    """Diagnostic local duplicate-partner change, remapped to release segments.

    block is global; positions use the source component's local atomic offsets.
    left_partners preserve source-row enumeration, renamed through
    left_partner_component; right_partners through export_component.
    """

    export_component: int = record_field("report", dtype="int64")
    founder_row: int = record_field("report", dtype="int64")
    source_component: int = record_field("report", dtype="int64")
    source_row: int = record_field("report", dtype="int64")
    block: int = record_field("report", units="block", dtype="int64")
    left_position: int = record_field("report", units="bp", dtype="int64")
    right_position: int = record_field("report", units="bp", dtype="int64")
    left_local_row: int = record_field("report", dtype="int64")
    right_local_row: int = record_field("report", dtype="int64")
    left_partner_component: int = record_field("report", dtype="int64")
    left_partners: tuple[int, ...] = record_field("report", dtype="tuple[int, ...]")
    right_partners: tuple[int, ...] = record_field("report", dtype="tuple[int, ...]")
    at_release_break: bool = record_field("report", dtype="bool")
    reason: str = record_field("report", default="local_duplicate_partner_change", dtype="str")
    method: str = record_field("report", default="duplicate-partner-v1", dtype="str")
    status: str = record_field("report", default="diagnostic_only", dtype="str")


@dataclass(frozen=True)
class FounderRelease:
    """Option-free contig release published by release[c], formatted by outputs.

    Calls include carrier and assembly masks. Identity flags are already renamed;
    outputs computes no science. Work counters are diagnostic; timers belong to
    stage run records. Input arrays are borrowed read-only and never modified.
    """

    segments: tuple[ReleaseSegment, ...] = record_field("call", dtype="tuple[ReleaseSegment, ...]")
    carrier_withheld_by_component: tuple[int, ...] = record_field("report", dtype="tuple[int, ...]")
    unobserved_founder_alleles: int = record_field("report", dtype="int64")
    founder_continuity_breaks: int = record_field("call", dtype="int64")
    assembly_ambiguous_alleles: int = record_field("call", dtype="int64")
    assembly_ambiguity_metadata: str = record_field("call", dtype="str")
    assembly_continuation_unverified_records: int = record_field("call", dtype="int64")
    assembly_count_unverified_records: int = record_field("diagnostic", dtype="int64")
    identity_flags: tuple[FounderIdentityFlag, ...] = record_field(
        "report", default=(), dtype="tuple[FounderIdentityFlag, ...]"
    )
    refit_ambiguous_alleles: int = record_field("report", default=0, dtype="int64")
    work_counters: Mapping[str, int] = record_field("diagnostic", default_factory=dict, dtype="Mapping[str, int]")
    status: str = record_field("call", default="recorded", dtype="str")
    schema: int = schema_field(1)


__all__ += [
    "LinkerSettings",
    "JoinSearchSettings",
    "CountRuleSettings",
    "JoinSettings",
    "JoinInputs",
    "ReleaseFamily",
    "release_family_kind",
    "JoinTask",
    "JoinRecord",
    "LevelRecord",
    "ReleaseAxes",
    "ReleaseSegment",
    "FounderIdentityFlag",
    "FounderRelease",
]


# AS6 publication products. Prepared panels are hoisted to completion for final products.
@dataclass(frozen=True)
class CompletionProduct:
    """Immutable prepared alphabet and typed completion summary payload."""

    prepared: tuple[PreparedPanel, ...] = record_field("call", dtype="tuple[PreparedPanel, ...]")
    summary: Mapping[str, Any] = record_field("report", dtype="Mapping[str, Any]")
    schema: int = schema_field(1)


@dataclass(frozen=True)
class JoinedLevel:
    """Checkpoint after every batch has been published and composed in range order."""

    components: tuple[FounderComponent, ...] = record_field("state", dtype="tuple[FounderComponent, ...]")
    record: LevelRecord = record_field("state", dtype="LevelRecord")
    release_alternatives: tuple[ReleaseFamily, ...] = record_field("state", dtype="tuple[ReleaseFamily, ...]")
    schema: int = schema_field(1)


@dataclass(frozen=True)
class RefinedLevel:
    """Refined components and families after one final-role hierarchy level."""

    components: tuple[FounderComponent, ...] = record_field("state", dtype="tuple[FounderComponent, ...]")
    summary: Mapping[str, Any] = record_field("report", dtype="Mapping[str, Any]")
    release_alternatives: tuple[ReleaseFamily, ...] = record_field("state", dtype="tuple[ReleaseFamily, ...]")
    schema: int = schema_field(1)


@dataclass(frozen=True)
class AssemblyProduct:
    """Complete assembly; prepared alphabet is supplied from the completion record."""

    components: tuple[FounderComponent, ...] = record_field("call", dtype="tuple[FounderComponent, ...]")
    levels: tuple[LevelRecord, ...] = record_field("report", dtype="tuple[LevelRecord, ...]")
    release_alternatives: tuple[ReleaseFamily, ...] = record_field("call", dtype="tuple[ReleaseFamily, ...]")
    prepared: tuple[PreparedPanel, ...] = record_field(
        "call", stored=False, default=(), dtype="tuple[PreparedPanel, ...]"
    )
    schema: int = schema_field(1)


__all__ += ["CompletionProduct", "JoinedLevel", "RefinedLevel", "AssemblyProduct"]


# AS7/AS8: integrated pr40 handoff; source SHA256 0753ec91e822a7580c1de10d5680cd001c687b340752836fdb5fa707728d40df
@dataclass(frozen=True)
class WeakJunction:
    """Native fit weak-junction record; persisted name best_gain is preserved."""

    first: int = record_field("report", dtype="int", units="block")
    last: int = record_field("report", dtype="int", units="block")
    pair: tuple[int, int] = record_field("report", dtype="int")
    best_gain: float = record_field("report", dtype="float64", units="nats")


@dataclass(frozen=True)
class FitAcceptanceTotals(RefinementCounterAccess):
    """Complete native fixed-K diagnostic state; no open-ended keys."""

    fine: int = record_field("state", dtype="int", units="1")
    coarse: int = record_field("state", dtype="int", units="1")
    exchange: int = record_field("state", dtype="int", units="1")
    bridge: int = record_field("state", dtype="int", units="1")
    escape: int = record_field("state", dtype="int", units="1")
    long: int = record_field("state", dtype="int", units="1")
    fine_gain: float = record_field("state", dtype="float64", units="nats")
    coarse_gain: float = record_field("state", dtype="float64", units="nats")
    exchange_gain: float = record_field("state", dtype="float64", units="nats")
    bridge_gain: float = record_field("state", dtype="float64", units="nats")
    escape_gain: float = record_field("state", dtype="float64", units="nats")
    long_gain: float = record_field("state", dtype="float64", units="nats")
    long_offers: int = record_field("state", dtype="int", units="1")
    escape_offers: int = record_field("state", dtype="int", units="1")
    escape_windows: int = record_field("state", dtype="int", units="1")
    escape_beam_cut: int = record_field("state", dtype="int", units="1")
    escape_beam_binding: int = record_field("state", dtype="int", units="1")
    long_windows: int = record_field("state", dtype="int", units="1")
    long_beam_cut: int = record_field("state", dtype="int", units="1")
    long_beam_binding: int = record_field("state", dtype="int", units="1")
    long_candidates: int = record_field("state", dtype="int", units="1")
    evaluated: int = record_field("state", dtype="int", units="1")
    false_positive: int = record_field("state", dtype="int", units="1")
    guard_rejected: int = record_field("state", dtype="int", units="1")
    guard_shared_tests: int = record_field("state", dtype="int", units="1")
    guard_relaxed: int = record_field("state", dtype="int", units="1")
    guard_shared_switches: float = record_field("state", dtype="float64", units="nats")
    guard_traced_blocks: int = record_field("state", dtype="int", units="1")
    changed_blocks: int = record_field("state", dtype="int", units="1")
    content_rounds: int = record_field("state", dtype="int", units="1")
    round_cap: int = record_field("state", dtype="int", units="1")
    stale_evaluated: int = record_field("state", dtype="int", units="1")
    exchange_cap: int = record_field("state", dtype="int", units="1")
    lex_items: int = record_field("state", dtype="int", units="1")
    lex_abort: int = record_field("state", dtype="int", units="1")


@dataclass(frozen=True)
class FitWorkCounters(RefinementCounterAccess):
    """Complete native fixed-K diagnostic state; no open-ended keys."""

    windows: int = record_field("state", dtype="int", units="1")
    templates_solved: int = record_field("state", dtype="int", units="1")
    templates_reused: int = record_field("state", dtype="int", units="1")
    dp: int = record_field("state", dtype="int", units="1")
    fixing_dps: int = record_field("state", dtype="int", units="1")
    beam_truncations: int = record_field("state", dtype="int", units="1")
    beam_fallbacks: int = record_field("state", dtype="int", units="1")
    guard_shared_tests: int = record_field("state", dtype="int", units="1")
    guard_traced_blocks: int = record_field("state", dtype="int", units="1")
    content_rounds: int = record_field("state", dtype="int", units="1")
    round_cap: int = record_field("state", dtype="int", units="1")
    stale_evaluated: int = record_field("state", dtype="int", units="1")
    exchange_cap: int = record_field("state", dtype="int", units="1")
    escape_windows: int = record_field("state", dtype="int", units="1")
    escape_beam_cut: int = record_field("state", dtype="int", units="1")
    escape_beam_binding: int = record_field("state", dtype="int", units="1")
    long_windows: int = record_field("state", dtype="int", units="1")
    long_beam_cut: int = record_field("state", dtype="int", units="1")
    long_beam_binding: int = record_field("state", dtype="int", units="1")
    long_candidates: int = record_field("state", dtype="int", units="1")
    lex_items: int = record_field("state", dtype="int", units="1")
    lex_abort: int = record_field("state", dtype="int", units="1")


@dataclass(frozen=True)
class FitSummary:
    """One converged fixed-K fit (the better of its forward-first and reverse-first searches)."""

    label: str = record_field("state", dtype="str", units="1")
    founders: int = record_field("call", dtype="int", units="1")
    F0: float = record_field("state", dtype="float64", units="nats")
    F: float = record_field("state", dtype="float64", units="nats")
    converged: bool = record_field("call", dtype="bool", units="1")
    sweeps: int = record_field("state", dtype="int", units="1")
    first: str = record_field("state", dtype="str", units="1")
    F_fwd: float = record_field("state", dtype="float64", units="nats")
    F_rev: float | None = record_field("state", dtype="float64", units="nats")
    max_frontier: int = record_field("state", dtype="int", units="1")
    frontier_fallbacks: int = record_field("state", dtype="int", units="1")
    accepted: FitAcceptanceTotals = record_field("state", dtype="FitAcceptanceTotals", units="1")
    max_deviation: float = record_field("state", dtype="float64", units="nats")
    wall: float = record_field("diagnostic", dtype="float64", units="seconds")
    worker_cpu: float = record_field("diagnostic", dtype="float64", units="seconds")
    thread_cpu: float = record_field("diagnostic", dtype="float64", units="seconds")
    tied_panels: tuple[tuple[tuple[int, ...], ...], ...] = record_field(
        "state", dtype="int64", axes=("A", "K", "B"), units="1", default=()
    )
    tied_scores: tuple[float, ...] = record_field("state", dtype="float64", axes=("A",), units="nats", default=())
    tied_switches: tuple[int, ...] = record_field("state", dtype="int64", axes=("A",), units="switch", default=())
    minimum_switches: int | None = record_field("state", dtype="int", units="1", default=None)
    phase_seconds: Mapping[str, float] = record_field(
        "diagnostic", dtype="float64", units="seconds", default_factory=dict
    )
    work_counters: FitWorkCounters = record_field("state", dtype="FitWorkCounters", units="1", default_factory=dict)


@dataclass(frozen=True)
class Fit:
    """A converged fit: its panel (int64 K x B prepared-row indices), summary and weak junctions."""

    rows: NDArray[np.int64] = record_field("call", dtype="int64", units="1", axes=("K", "B"))
    summary: FitSummary = record_field("state", dtype="FitSummary", units="1")
    weak_junctions: tuple[WeakJunction, ...] = record_field("state", dtype="tuple[WeakJunction, ...]", units="1")


@dataclass(frozen=True)
class CountDecisionRecord:
    """One count round of a component's count pass."""

    kind: str = record_field("state", dtype="str", units="1")
    round: str = record_field("state", dtype="str", units="1")
    founders: int = record_field("call", dtype="int", units="1")
    skipped_by_bound: bool = record_field("state", dtype="bool", units="1")
    accept_larger: bool = record_field("call", dtype="bool", units="1")
    ambiguous: bool = record_field("call", dtype="bool", units="1")
    bound: str | None = record_field("state", dtype="str", units="1", default=None)
    optimistic_score: float | None = record_field("state", dtype="float64", units="nats", default=None)
    per_sample_gain_bound: float | None = record_field("state", dtype="float64", units="nats", default=None)
    per_sample_gain_bound_sample: int | None = record_field("state", dtype="int", units="1", default=None)
    band_bound: float | None = record_field("state", dtype="float64", units="founder_cost", default=None)
    margin: float | None = record_field("state", dtype="float64", units="founder_cost", default=None)
    band: tuple[float, float] | None = record_field(
        "state", dtype="tuple[float, float] | None", units="founder_cost", default=None
    )
    refitted: tuple[int, ...] = record_field("state", dtype="tuple[int, ...]", units="1", default=())
    candidate_F: tuple | None = record_field("state", dtype="tuple | None", units="1", default=None)
    selected: int | None = record_field("state", dtype="int", units="1", default=None)
    selected_F: float | None = record_field("state", dtype="float64", units="nats", default=None)
    runner_up_margin_over_cost: float | None = record_field(
        "state", dtype="float64", units="founder_cost", default=None
    )
    criterion: dict[str, Any] | None = record_field("state", dtype="dict[str, Any] | None", units="1", default=None)
    without_sequence: tuple[int, ...] = record_field("state", dtype="tuple[int, ...]", units="1", default=())
    shadow_cells: int = record_field("state", dtype="int", units="1", default=0)
    shadow_alternatives: tuple = record_field("state", dtype="tuple", units="1", default=())
    count_unverified: tuple = record_field("diagnostic", dtype="tuple", units="1", default=())
    hidden_sequence_failures: tuple | None = record_field("state", dtype="tuple | None", units="1", default=None)
    after_addition: bool = record_field("state", dtype="bool", units="1", default=False)
    count_cycle: bool = record_field("call", dtype="bool", units="1", default=False)
    cycle_start_count: int | None = record_field("state", dtype="int", units="1", default=None)
    cycle_undecided: bool = record_field("call", dtype="bool", units="1", default=False)
    larger_fit: str | None = record_field("state", dtype="str", units="1", default=None)
    smaller_fit: str | None = record_field("state", dtype="str", units="1", default=None)
    tied_candidates: tuple = record_field("state", dtype="tuple", units="1", default=())
    competitive_fits: tuple[str, ...] = record_field("state", dtype="str", axes=("C",), units="1", default=())
    fit_family_evaluations: tuple = record_field("state", dtype="tuple", units="1", default=())
    selected_refit_panel: tuple = record_field("state", dtype="tuple", units="1", default=())
    tie_decisions: tuple | None = record_field("state", dtype="tuple | None", units="1", default=None)
    tie_decisions_agree: bool | None = record_field("state", dtype="bool", units="1", default=None)
    ambiguous_k: bool = record_field("state", dtype="bool", units="1", default=False)
    alternatives: tuple = record_field("state", dtype="tuple", units="1", default=())
    unresolved_panels: tuple = record_field("state", dtype="tuple", units="1", default=())
    prior: str | None = record_field("call", dtype="str", units="1", default=None)
    continuation_key: str | None = record_field("state", dtype="str", units="1", default=None)
    continuation_replayed: bool = record_field("state", dtype="bool", units="1", default=False)
    schema: int = schema_field(2)


@dataclass(frozen=True)
class BestFReplacement:
    """The count pass returned to a K it had held with a better converged panel."""

    from_fit: str = record_field("state", dtype="str", units="1")
    to_fit: str = record_field("state", dtype="str", units="1")
    gain: float = record_field("state", dtype="float64", units="nats")
    recorded: tuple[Mapping[str, Any], ...] = record_field("state", dtype="recorded count comparisons", units="1")
    deletion_vs_best_smaller: dict[str, Any] | None = record_field("state", dtype="dict[str, Any] | None", units="1")
    addition_vs_best_larger: dict[str, Any] | None = record_field("state", dtype="dict[str, Any] | None", units="1")
    own_deletion_test: dict[str, Any] | None = record_field("state", dtype="dict[str, Any] | None", units="1")
    fails_own_deletion_test: bool | None = record_field("state", dtype="bool", units="1")
    ambiguous: bool = record_field("call", dtype="bool", units="1")


@dataclass(frozen=True)
class CountPass:
    """Result of count_pass: the decided panel and the pass's record."""

    rows: NDArray[np.int64] = record_field("call", dtype="int64", units="1", axes=("K", "B"))
    scores: NDArray[np.float64] = record_field("state", dtype="float64", units="nats", axes=("N",))
    F: float = record_field("state", dtype="float64", units="nats")
    founders_before: int = record_field("state", dtype="int", units="1")
    fits: tuple[FitSummary, ...] = record_field("state", dtype="FitSummary", units="1")
    decisions: tuple[CountDecisionRecord, ...] = record_field(
        "state", dtype="tuple[CountDecisionRecord, ...]", units="1"
    )
    final_fit: str = record_field("call", dtype="str", units="1")
    weak_junctions: tuple[WeakJunction, ...] = record_field("state", dtype="WeakJunction", units="1")
    count_cycle: bool = record_field("call", dtype="bool", units="1")
    cycle_undecided: bool = record_field("call", dtype="bool", units="1")
    count_limit: int = record_field("state", dtype="int", units="1")
    count_limit_reached: bool = record_field("call", dtype="bool", units="1")
    additions: bool = record_field("state", dtype="bool", units="1")
    ambiguous: bool = record_field("call", dtype="bool", units="1")
    best_f_replacement: BestFReplacement | None = record_field("state", dtype="BestFReplacement | None", units="1")
    live_alternatives: tuple = record_field("state", dtype="tuple", units="1", default=())
    unresolved_alternatives: tuple = record_field("state", dtype="tuple", units="1", default=())
    continuation_families: tuple = record_field("state", dtype="tuple", units="1", default=())
    count_unverified: tuple = record_field("diagnostic", dtype="tuple", units="1", default=())
    branch_evaluations: int = record_field("state", dtype="int", units="1", default=1)
    branch_reused_fits: int = record_field("state", dtype="int", units="1", default=0)
    branch_seconds: float = record_field("diagnostic", dtype="float64", units="seconds", default=0.0)
    branch_status: str = record_field("call", dtype="str", units="1", default="none")
    selected_branch: str | None = record_field("call", dtype="str", units="1", default=None)
    input_F0: float | None = record_field("state", dtype="float64", units="nats", default=None)
    count_continuations: tuple = record_field("state", dtype="tuple", units="1", default=())
    count_branch_evaluations: int = record_field("state", dtype="int", units="1", default=0)
    count_branch_seconds: float = record_field("diagnostic", dtype="float64", units="seconds", default=0.0)


@dataclass(frozen=True)
class ComponentRefinement:
    """One component's refinement at one level (the content of its component record)."""

    component: int = record_field("state", dtype="int", units="1")
    start: int = record_field("state", dtype="int", units="1")
    stop: int = record_field("state", dtype="int", units="1")
    changed: bool = record_field("call", dtype="bool", units="1")
    reason: str | None = record_field("call", dtype="str", units="1")
    founders_before: int = record_field("state", dtype="int", units="1")
    founders: int = record_field("call", dtype="int", units="1")
    F0: float | None = record_field("state", dtype="float64", units="nats", default=None)
    F: float | None = record_field("state", dtype="float64", units="nats", default=None)
    changed_local_rows: int | None = record_field("state", dtype="int", units="1", default=None)
    bin_size: int | None = record_field("state", dtype="int", units="1", default=None)
    coarse_units: int | None = record_field("state", dtype="int", units="1", default=None)
    cost: float | None = record_field("state", dtype="float64", units="1", default=None)
    decisions: tuple[CountDecisionRecord, ...] = record_field(
        "state", dtype="CountDecisionRecord", units="1", default=()
    )
    fits: tuple[FitSummary, ...] = record_field("state", dtype="FitSummary", units="1", default=())
    count_cycle: bool = record_field("call", dtype="bool", units="1", default=False)
    cycle_undecided: bool = record_field("call", dtype="bool", units="1", default=False)
    count_limit: int | None = record_field("state", dtype="int", units="1", default=None)
    count_limit_reached: bool = record_field("call", dtype="bool", units="1", default=False)
    ambiguous: bool = record_field("call", dtype="bool", units="1", default=False)
    final_fit: str | None = record_field("call", dtype="str", units="1", default=None)
    best_f_replacement: BestFReplacement | None = record_field(
        "state", dtype="BestFReplacement | None", units="1", default=None
    )
    weak_label_junctions: tuple[WeakJunction, ...] = record_field("state", dtype="WeakJunction", units="1", default=())
    elapsed_seconds: float = record_field("diagnostic", dtype="float64", units="seconds", default=0.0)
    tie_events: dict[str, Any] = record_field("state", dtype="dict[str, Any]", units="1", default_factory=dict)
    alternatives: tuple = record_field("state", dtype="tuple", units="1", default=())
    release_alternatives: tuple = record_field("state", dtype="tuple", units="1", default=())
    input_alternatives: tuple = record_field("state", dtype="tuple", units="1", default=())
    branch_evaluations: int = record_field("state", dtype="int", units="1", default=1)
    branch_reused_fits: int = record_field("state", dtype="int", units="1", default=0)
    branch_seconds: float = record_field("diagnostic", dtype="float64", units="seconds", default=0.0)
    branch_status: str = record_field("call", dtype="str", units="1", default="none")
    selected_branch: str | None = record_field("call", dtype="str", units="1", default=None)
    count_continuations: tuple = record_field("state", dtype="tuple", units="1", default=())
    count_branch_evaluations: int = record_field("state", dtype="int", units="1", default=0)
    count_branch_seconds: float = record_field("diagnostic", dtype="float64", units="seconds", default=0.0)
    phase_seconds: dict[str, Any] = record_field(
        "diagnostic", dtype="dict[str, Any]", units="seconds", default_factory=dict
    )
    work_counters: dict[str, Any] = record_field("state", dtype="dict[str, Any]", units="1", default_factory=dict)
    polish: dict[str, Any] = record_field("state", dtype="dict[str, Any]", units="1", default_factory=dict)

    window_stop: WindowStop | None = record_field("report", default=None, dtype="WindowStop | None")


@dataclass(frozen=True)
class LevelSummary:
    """Census of one refinement level (its level record; logged as one line, LevelSummary.line)."""

    level: int = record_field("state", dtype="int", units="1")
    additions: bool = record_field("state", dtype="bool", units="1")
    min_sites_per_bin: int = record_field("state", dtype="int", units="1")
    elapsed_seconds: float = record_field("diagnostic", dtype="float64", units="seconds")
    components: int = record_field("state", dtype="int", units="1")
    fits: int = record_field("state", dtype="int", units="1")
    nonconverged_fits: int = record_field("state", dtype="int", units="1")
    max_sweeps: int = record_field("state", dtype="int", units="1")
    mean_sweeps: float = record_field("state", dtype="float64", units="1")
    max_frontier: int = record_field("state", dtype="int", units="1")
    frontier_fallbacks: int = record_field("state", dtype="int", units="1")
    count_limit_hits: int = record_field("state", dtype="int", units="1")
    count_cycles: int = record_field("state", dtype="int", units="1")
    count_cycles_undecided: int = record_field("state", dtype="int", units="1")
    best_f_replacements: int = record_field("state", dtype="int", units="1")
    weak_label_junctions: int = record_field("state", dtype="int", units="1")
    deletions_without_sequence: int = record_field("state", dtype="int", units="1")
    decided_by_sequence_part: int = record_field("state", dtype="int", units="1")
    hidden_sequence_failures: int = record_field("state", dtype="int", units="1")
    ambiguous_count_calls: tuple[Mapping[str, Any], ...] = record_field(
        "state", dtype="ambiguous count comparisons", units="1"
    )
    phase_seconds: dict[str, Any] = record_field(
        "diagnostic", dtype="dict[str, Any]", units="seconds", default_factory=dict
    )
    work_counters: dict[str, Any] = record_field("state", dtype="dict[str, Any]", units="1", default_factory=dict)

    def line(self) -> str:
        """The level's census log line (backstops, cycles and ambiguity are never silent)."""
        return (
            f"  [founder refinement L{self.level}] converged search: {self.fits} fits, "
            f"{self.nonconverged_fits} at the sweep limit, max sweeps {self.max_sweeps}, "
            f"max window frontier {self.max_frontier} ({self.frontier_fallbacks} frontier-cap "
            f"fallbacks), count limit hits {self.count_limit_hits}, count cycles "
            f"{self.count_cycles} ({self.count_cycles_undecided} undecided), ambiguous count "
            f"calls {len(self.ambiguous_count_calls)}, best-F panel replacements "
            f"{self.best_f_replacements}, weak label junctions {self.weak_label_junctions}, "
            f"deletions without sequence {self.deletions_without_sequence}, decided by the "
            f"sequence part {self.decided_by_sequence_part}, kept founders failing the "
            f"sequence test {self.hidden_sequence_failures}"
        )


__all__ += [
    "WeakJunction",
    "FitSummary",
    "Fit",
    "CountDecisionRecord",
    "BestFReplacement",
    "CountPass",
    "ComponentRefinement",
    "LevelSummary",
]


# AS8 logical-prefix records, supplied by pr40 (2026-11-01 07:46).
@dataclass(frozen=True)
class FitOrdinalBinding:
    ordinal: int = record_field("state", dtype="int")
    label: str = record_field("state", dtype="str")
    input_panel: tuple[tuple[int, ...], ...] = record_field("state", dtype="int")
    kind: str = record_field("state", dtype="str")
    branch: str | None = record_field("state", dtype="str")


@dataclass(frozen=True)
class LogicalWindowReceipt:
    ordinal: WindowOrdinal = record_field("state", dtype="WindowOrdinal")
    account: WindowAccount | None = record_field("state", dtype="WindowAccount | None")
    reused: bool = record_field("state", dtype="bool")


@dataclass(frozen=True)
class PhysicalWindowWork:
    window_work_limits: int = record_field("diagnostic", dtype="int")
    window_work_calls: int = record_field("diagnostic", dtype="int")
    window_work_units: int = record_field("diagnostic", dtype="int")
    window_work_retries: int = record_field("diagnostic", dtype="int")
    window_work_discarded_focals: int = record_field("diagnostic", dtype="int")


@dataclass(frozen=True)
class WindowStop:
    first: WindowOrdinal = record_field("state", dtype="WindowOrdinal")
    fits: tuple[FitOrdinalBinding, ...] = record_field("state", dtype="tuple[FitOrdinalBinding, ...]")
    prefix: tuple[LogicalWindowReceipt, ...] = record_field("state", dtype="tuple[LogicalWindowReceipt, ...]")
    work: WindowWorkCounters = record_field("state", dtype="WindowWorkCounters")
    domain: WindowDomain = record_field("report", dtype="WindowDomain")
    physical_work: PhysicalWindowWork = record_field("diagnostic", dtype="PhysicalWindowWork")


__all__ += ["FitOrdinalBinding", "LogicalWindowReceipt", "PhysicalWindowWork", "WindowStop"]


__all__ += ["FitAcceptanceTotals", "FitWorkCounters"]


# AS7/AS8 sweep history, moved from refinement/diagnostics.py (TEMP-AS6; asm40c, 1 Nov).
@dataclass(frozen=True)
class SweepHistory(AcceptanceCounters):
    """One completed native sweep, with flat counter names retained by its codec.

    Proposal records retain their sparse native polish subrecords; ordinary and
    polish sweeps never manufacture each other's absent keys. Timings are diagnostic.
    """

    sweep: int = record_field("state", dtype="int", units="1")
    reverse: bool = record_field("state", dtype="bool", units="1")
    F: float = record_field("state", dtype="float64", units="nats")
    dF: float = record_field("state", dtype="float64", units="nats")
    dF_accounted: float = record_field("state", dtype="float64", units="nats")
    deviation: float = record_field("state", dtype="float64", units="nats")
    accepted: int = record_field("state", dtype="int", units="1")
    proposals: Mapping[str, Any] = record_field("state", dtype="Mapping", units="1")
    wall: float = record_field("diagnostic", dtype="float64", units="seconds")
    message_wall: float = record_field("diagnostic", dtype="float64", units="seconds")
    proposal_wall: float = record_field("diagnostic", dtype="float64", units="seconds")
    pass_wall: float = record_field("diagnostic", dtype="float64", units="seconds")


__all__ += ["SweepHistory"]
