"""Declare the component phase seam and the released per-contig phase product.

Owns frozen records, axis/dtype/role metadata and no computation or storage.
Method: docs/methods/family_phase.md (P0 FP01-FP24; CONTRACTS_40 §10).
Arrays are owned by the caller/result and borrowed read-only during computation;
frozen records do not freeze NumPy buffers. Labels retain painting row order,
samples retain input order, and bin/site coordinates are never truth-aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pedigree import ParentEdges
from .roles import record_field

__all__ = [
    "PhaseObservations",
    "PhaseStates",
    "PhaseBins",
    "PhaseTransitions",
    "PhasePainting",
    "PhaseRefit",
    "PhaseMap",
    "PhaseRates",
    "PhaseInputIdentities",
    "PhaseComponentInputs",
    "PhaseSolve",
    "AlleleUncertainty",
    "SampleRelease",
    "AgreedRelease",
    "MeiosisChain",
    "PhaseComponentResult",
    "PhasedSamples",
    "PhaseDiagnostics",
    "MirroredSolveDiagnostics",
    "PairingDiagnostics",
    "PhaseComponentState",
]


@dataclass(frozen=True)
class PhaseObservations:
    """Component-local evidence, preserving the absolute site and run sample order."""

    sample_ids: tuple[str, ...] = field(metadata={"role": "obs", "axes": ("N",), "units": "1", "dtype": "str"})
    positions: np.ndarray = field(metadata={"role": "obs", "axes": ("L",), "units": "bp", "dtype": "int64"})
    absolute_site_indices: np.ndarray = field(metadata={"role": "obs", "axes": ("L",), "units": "1", "dtype": "int64"})
    likelihoods: np.ndarray = field(
        metadata={"role": "evidence", "axes": ("N", "L", "3"), "units": "1", "dtype": "float64"}
    )
    observed: np.ndarray = field(metadata={"role": "evidence", "axes": ("N", "L"), "units": "1", "dtype": "bool"})
    excluded: np.ndarray = field(metadata={"role": "call", "axes": ("N",), "units": "1", "dtype": "bool"})


@dataclass(frozen=True)
class PhaseStates:
    """Painting labels: trajectory classes followed by BACKGROUND; selected site axis."""

    positions: np.ndarray = field(metadata={"role": "obs", "axes": ("Ls",), "units": "bp", "dtype": "int64"})
    named_alleles: np.ndarray = field(metadata={"role": "state", "axes": ("C", "Ls"), "units": "1", "dtype": "int8"})
    classes: tuple[tuple[int, ...], ...] = field(
        metadata={"role": "state", "axes": ("C",), "units": "1", "dtype": "row_indices"}
    )
    background_frequency: np.ndarray = field(
        metadata={"role": "state", "axes": ("Ls",), "units": "1", "dtype": "float64"}
    )
    n_states: int = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "int"})


@dataclass(frozen=True)
class PhaseBins:
    """Half-open painting bins, with their selected-site offsets and mean bp centres."""

    starts: np.ndarray = field(metadata={"role": "state", "axes": ("M",), "units": "1", "dtype": "int64"})
    stops: np.ndarray = field(metadata={"role": "state", "axes": ("M",), "units": "1", "dtype": "int64"})
    centers: np.ndarray = field(metadata={"role": "state", "axes": ("M",), "units": "bp", "dtype": "float64"})
    edges: np.ndarray = field(metadata={"role": "state", "axes": ("M+1",), "units": "bp", "dtype": "int64"})


@dataclass(frozen=True)
class PhaseTransitions:
    """The recorded painting copying-process factors consumed by phase."""

    log_same: np.ndarray = field(metadata={"role": "state", "axes": ("M-1",), "units": "nats", "dtype": "float64"})
    site_correlation: float = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "float64"})


@dataclass(frozen=True)
class PhasePainting:
    """Painting fields read by phase; labels and candidate row sets keep their original identities."""

    sample_ids: tuple[str, ...] = field(
        kw_only=True, metadata={"role": "obs", "axes": ("N",), "units": "1", "dtype": "str"}
    )
    states: PhaseStates = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "PhaseStates"})
    bins: PhaseBins = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "PhaseBins"})
    transitions: PhaseTransitions = field(
        metadata={"role": "state", "axes": (), "units": "1", "dtype": "PhaseTransitions"}
    )
    local_keys: np.ndarray = field(metadata={"role": "state", "axes": ("M", "S"), "units": "1", "dtype": "int32"})
    eligible: np.ndarray = field(metadata={"role": "call", "axes": ("N",), "units": "1", "dtype": "bool"})
    labels: np.ndarray = field(metadata={"role": "call", "axes": ("N", "2", "M"), "units": "1", "dtype": "int32"})
    classes: np.ndarray = field(metadata={"role": "call", "axes": ("N", "2", "M"), "units": "1", "dtype": "int32"})
    status: np.ndarray = field(metadata={"role": "call", "axes": ("N", "2", "M"), "units": "1", "dtype": "uint8"})
    member_ids: np.ndarray = field(metadata={"role": "call", "axes": ("N", "2", "M"), "units": "1", "dtype": "int32"})
    member_sets: tuple[tuple[int, ...], ...] = field(
        metadata={"role": "call", "axes": (), "units": "1", "dtype": "row_sets"}
    )
    packed_emissions: np.ndarray | None = field(
        metadata={"role": "evidence", "axes": ("N", "S*(S+1)/2", "M"), "units": "nats", "dtype": "float64?"}
    )
    robustness_epsilon: float = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "float64"})
    emission_log_floor: float = field(metadata={"role": "state", "axes": (), "units": "nats", "dtype": "float64"})


@dataclass(frozen=True)
class PhaseRefit:
    """Refit scaffold and exact chromosome-wide prior counts (all components, called cells only)."""

    positions: np.ndarray = field(metadata={"role": "obs", "axes": ("Lr",), "units": "bp", "dtype": "int64"})
    inference: np.ndarray = field(metadata={"role": "state", "axes": ("K", "Lr"), "units": "1", "dtype": "int8"})
    prior_ones: int = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "int64"})
    prior_called: int = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "int64"})


@dataclass(frozen=True)
class PhaseRates:
    """Named per-bin-edge probabilities; storage never precedes their decisions."""

    theta_l: np.ndarray = record_field("state", axes=("M-1",), units="probability", dtype="float64")
    theta_m: np.ndarray = record_field("state", axes=("M-1",), units="probability", dtype="float64")
    alpha: np.ndarray = record_field("state", axes=("M-1",), units="probability", dtype="float64")
    beta: np.ndarray = record_field("state", axes=("M-1",), units="probability", dtype="float64")


@dataclass(frozen=True)
class PhaseMap:
    """Explicit genetic distances between successive bin centres and protected bp intervals."""

    distance_morgans: np.ndarray = field(
        metadata={"role": "state", "axes": ("M-1",), "units": "Morgans", "dtype": "float64"}
    )
    protected: tuple[tuple[int, int], ...] = field(
        metadata={"role": "state", "axes": ("P", "2"), "units": "bp", "dtype": "int64"}
    )


@dataclass(frozen=True)
class PhaseInputIdentities:
    """Upstream derivation identities supplied by the canonical stage reader.

    The pedigree content identity and shared map semantic identity are verified
    at the adapter boundary; the store validates the other three record identities.
    Neutral historical component fixtures have no store identities and use None.
    """

    evidence: str = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "str"})
    painting: str = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "str"})
    refit: str = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "str"})
    pedigree: str = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "str"})
    genetic_map: str = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "str"})


@dataclass(frozen=True)
class PhaseComponentInputs:
    """Complete scientific inputs of one component; no task, storage route or worker count."""

    identities: PhaseInputIdentities | None = field(
        default=None,
        kw_only=True,
        metadata={"role": "state", "axes": (), "units": "1", "dtype": "PhaseInputIdentities?"},
    )
    contig: str = field(metadata={"role": "obs", "axes": (), "units": "1", "dtype": "str"})
    component: int = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "int"})
    observations: PhaseObservations = field(
        metadata={"role": "evidence", "axes": (), "units": "1", "dtype": "PhaseObservations"}
    )
    painting: PhasePainting = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "PhasePainting"})
    refit: PhaseRefit = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "PhaseRefit"})
    pedigree: ParentEdges = field(metadata={"role": "call", "axes": (), "units": "1", "dtype": "ParentEdges"})
    genetic_map: PhaseMap = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "PhaseMap"})


@dataclass(frozen=True)
class PhaseSolve:
    """Final labels and copy sources of one convention solve; all are inference state."""

    X: np.ndarray = field(metadata={"role": "state", "axes": ("N", "2", "M"), "units": "1", "dtype": "int64"})
    V: np.ndarray = field(metadata={"role": "state", "axes": ("N", "2", "M"), "units": "1", "dtype": "int8"})
    history: tuple[float, ...] = field(metadata={"role": "state", "axes": (), "units": "nats", "dtype": "float64"})
    settled: bool = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "bool"})
    pairings: tuple[tuple[int, int, float], ...] = field(
        metadata={"role": "state", "axes": (), "units": "1", "dtype": "pairing_gains"}
    )
    pairing_moves: tuple[tuple[int, int, float], ...] = field(
        metadata={"role": "state", "axes": (), "units": "1", "dtype": "accepted_pairings"}
    )


@dataclass(frozen=True)
class AlleleUncertainty:
    """CAL2 evidence and selected-allele disagreement in fixed sample/homologue order."""

    evidence: np.ndarray = field(metadata={"role": "state", "axes": ("S", "L"), "units": "nats", "dtype": "float64"})
    flips: np.ndarray = field(metadata={"role": "state", "axes": ("S", "L"), "units": "1", "dtype": "float64"})
    logit_prior: float = field(metadata={"role": "state", "axes": (), "units": "nats", "dtype": "float64"})


@dataclass(frozen=True)
class SampleRelease:
    """One solve's sample release; chain probabilities remain float64 until the release test."""

    calls: np.ndarray = field(metadata={"role": "call", "axes": ("L", "2"), "units": "1", "dtype": "int8"})
    undetermined: np.ndarray = field(metadata={"role": "call", "axes": ("L",), "units": "1", "dtype": "bool"})
    set_breaks: tuple[int, ...] = field(metadata={"role": "call", "axes": (), "units": "1", "dtype": "bin_indices"})
    scores: np.ndarray | None = field(
        metadata={"role": "report", "axes": ("2", "L"), "units": "1", "dtype": "float32?"}
    )
    chain: np.ndarray | None = field(
        metadata={"role": "state", "axes": ("2", "2", "M"), "units": "1", "dtype": "float64?"}
    )


@dataclass(frozen=True)
class AgreedRelease:
    """Both solves' agreement in default orientation, with new-set site indices."""

    calls: np.ndarray = field(metadata={"role": "call", "axes": ("L", "2"), "units": "1", "dtype": "int8"})
    undetermined: np.ndarray = field(metadata={"role": "call", "axes": ("L",), "units": "1", "dtype": "bool"})
    set_starts: np.ndarray = field(metadata={"role": "call", "axes": (), "units": "1", "dtype": "int64"})
    scores: np.ndarray | None = field(
        metadata={"role": "report", "axes": ("2", "L"), "units": "1", "dtype": "float32?"}
    )


@dataclass(frozen=True)
class MeiosisChain:
    """Released parental source per bin; float32 support is storage after the float64 decision."""

    source: np.ndarray = field(metadata={"role": "state", "axes": ("M",), "units": "1", "dtype": "int8"})
    support: np.ndarray = field(metadata={"role": "state", "axes": ("M",), "units": "1", "dtype": "float32"})
    reset: np.ndarray = field(metadata={"role": "state", "axes": ("M",), "units": "1", "dtype": "bool"})


@dataclass(frozen=True)
class PhaseComponentResult:
    """Component release and the two final solves, ready for a contig adapter."""

    solves: tuple[PhaseSolve, PhaseSolve] = field(
        metadata={"role": "state", "axes": (), "units": "1", "dtype": "PhaseSolve[2]"}
    )
    site_indices: np.ndarray = field(metadata={"role": "state", "axes": ("L",), "units": "1", "dtype": "int64"})
    site_bins: np.ndarray = field(metadata={"role": "state", "axes": ("L",), "units": "1", "dtype": "int64"})
    site_alleles: np.ndarray = field(metadata={"role": "state", "axes": ("S", "L"), "units": "1", "dtype": "int8"})
    uncertainty: tuple[AlleleUncertainty, AlleleUncertainty] = field(
        metadata={"role": "state", "axes": (), "units": "1", "dtype": "AlleleUncertainty[2]"}
    )
    releases: tuple[tuple[SampleRelease, ...], tuple[SampleRelease, ...]] = field(
        metadata={"role": "call", "axes": (), "units": "1", "dtype": "SampleRelease[2,N]"}
    )
    agreed: tuple[AgreedRelease, ...] = field(
        metadata={"role": "call", "axes": ("N",), "units": "1", "dtype": "AgreedRelease"}
    )
    chains: tuple[MeiosisChain, ...] = field(
        metadata={"role": "state", "axes": ("E",), "units": "1", "dtype": "MeiosisChain"}
    )


@dataclass(frozen=True)
class MirroredSolveDiagnostics:
    """Both convergence states and junction counts, retaining mirror objective history."""

    sweeps: int = record_field("report", units="count", dtype="int")
    settled: bool = record_field("state", units="1", dtype="bool")
    objective: tuple[float, ...] = record_field("report", units="nats", dtype="float64")
    junction_candidates: int = record_field("report", units="count", dtype="int")
    default_settled: bool = record_field("state", units="1", dtype="bool")
    default_junction_candidates: int = record_field("report", units="count", dtype="int")


@dataclass(frozen=True)
class PairingDiagnostics:
    """Joint pairing candidates, accepted moves and the ambiguity breaks they add."""

    candidates: int = record_field("report", units="count", dtype="int")
    moves: int = record_field("report", units="count", dtype="int")
    mirrored_moves: int = record_field("report", units="count", dtype="int")
    ambiguous: int = record_field("report", units="count", dtype="int")
    breaks: int = record_field("report", units="count", dtype="int")


@dataclass(frozen=True)
class PhaseDiagnostics:
    """Scientific release counts and both solve histories; execution timing stays outside.

    Component entries remain in upstream component order. A skipped component
    has a zero history-step count, settled=True and objective=(); its optional
    mirror/pairing entry is None, the explicit form of 3.5's empty dictionary.
    """

    history_steps: tuple[int, ...] = record_field("report", units="count", dtype="int")
    settled: tuple[bool, ...] = record_field("state", units="1", dtype="bool")
    objective: tuple[tuple[float, ...], ...] = record_field("report", units="nats", dtype="float64")
    junction_candidates: tuple[int, ...] = record_field("report", units="count", dtype="int")
    withheld_joins: int = record_field("report", units="count", dtype="int")
    tie_breaks: int = record_field("report", units="count", dtype="int")
    called_alleles: int = record_field("report", units="count", dtype="int")
    phased_heterozygotes: int = record_field("report", units="count", dtype="int")
    unphased_heterozygotes: int = record_field("report", units="count", dtype="int")
    phase_sets: int = record_field("report", units="count", dtype="int")
    mirrored_solves: tuple[MirroredSolveDiagnostics | None, ...] = record_field(
        "report", dtype="MirroredSolveDiagnostics?"
    )
    pairings: tuple[PairingDiagnostics | None, ...] = record_field("report", dtype="PairingDiagnostics?")

    @property
    def sweeps(self) -> tuple[int, ...]:
        """Legacy report spelling for len(history)-1, including repeated sweep initial objectives."""
        return self.history_steps

    @property
    def converged(self) -> bool:
        """Every component settled before the sweep cap in both ascents."""
        return all(self.settled)


@dataclass(frozen=True)
class PhasedSamples:
    """Per-contig released phase, CONTRACTS_40 §10; schema excludes the obsolete all-zero frames."""

    contig: str = field(metadata={"role": "obs", "axes": (), "units": "1", "dtype": "str"})
    sample_ids: tuple[str, ...] = field(metadata={"role": "obs", "axes": ("N",), "units": "1", "dtype": "str"})
    positions: np.ndarray = field(metadata={"role": "obs", "axes": ("L",), "units": "bp", "dtype": "int64"})
    component_ids: np.ndarray = field(metadata={"role": "state", "axes": ("L",), "units": "1", "dtype": "int32"})
    bin_edges: np.ndarray = field(metadata={"role": "state", "axes": ("M",), "units": "bp", "dtype": "int64"})
    bin_edge_offsets: np.ndarray = field(metadata={"role": "state", "axes": ("C+1",), "units": "1", "dtype": "int64"})
    bin_centers: np.ndarray = field(metadata={"role": "state", "axes": ("M-C",), "units": "bp", "dtype": "float64"})
    calls: np.ndarray = field(metadata={"role": "call", "axes": ("N", "L", "2"), "units": "1", "dtype": "int8"})
    coverage: np.ndarray = field(metadata={"role": "call", "axes": ("Rc", "2"), "units": "1", "dtype": "int64"})
    coverage_offsets: np.ndarray = field(metadata={"role": "call", "axes": ("N+1",), "units": "1", "dtype": "int64"})
    edges_identity: str = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "str"})
    frame_undetermined: np.ndarray = field(metadata={"role": "call", "axes": ("N", "L"), "units": "1", "dtype": "int8"})
    phase_breaks: np.ndarray = field(metadata={"role": "call", "axes": ("Q",), "units": "1", "dtype": "int64"})
    phase_break_offsets: np.ndarray = field(metadata={"role": "call", "axes": ("N+1",), "units": "1", "dtype": "int64"})
    chain_source: np.ndarray = field(metadata={"role": "state", "axes": ("E", "M-C"), "units": "1", "dtype": "int8"})
    chain_support: np.ndarray = field(
        metadata={"role": "state", "axes": ("E", "M-C"), "units": "1", "dtype": "float32"}
    )
    chain_reset: np.ndarray = field(metadata={"role": "state", "axes": ("E", "M-C"), "units": "1", "dtype": "bool"})
    diagnostics: PhaseDiagnostics = record_field("report", axes=(), units="1", dtype="PhaseDiagnostics", kw_only=True)
    schema: int = field(
        default=1, kw_only=True, metadata={"role": "diagnostic", "axes": (), "units": "1", "dtype": "int"}
    )

    def component_bin_edges(self, component: int) -> np.ndarray:
        """Borrow the component's unchanged half-open painting/fallback bin edges."""
        return self.bin_edges[self.bin_edge_offsets[component] : self.bin_edge_offsets[component + 1]]


@dataclass(frozen=True)
class PhaseComponentState:
    """Decision-bearing component intermediates for oracle replay and adjacent-unit composition.

    These are inference state, not execution diagnostics. The seam exposes them
    because final calls alone cannot prove that the same solve path was followed.
    """

    emissions: np.ndarray = field(
        metadata={"role": "evidence", "axes": ("N", "S", "S", "M"), "units": "nats", "dtype": "float64"}
    )
    rates: PhaseRates = field(metadata={"role": "state", "axes": (), "units": "1", "dtype": "PhaseRates"})
    parents: np.ndarray = field(metadata={"role": "call", "axes": ("N", "2"), "units": "1", "dtype": "int64"})
    junctions: tuple[tuple[tuple[int, int, int, int], ...], ...] = field(
        metadata={"role": "state", "axes": (), "units": "1", "dtype": "junctions[2]"}
    )
    withheld: dict[int, tuple[int, ...]] = field(
        metadata={"role": "call", "axes": (), "units": "1", "dtype": "sample_bin_indices"}
    )
    pairing: dict[int, tuple[int, ...]] = field(
        metadata={"role": "call", "axes": (), "units": "1", "dtype": "sample_bin_indices"}
    )
    break_bins: tuple[tuple[int, ...], ...] = field(
        metadata={"role": "call", "axes": (), "units": "1", "dtype": "sample_bin_indices"}
    )
    tie_cuts: tuple[tuple[int, ...], ...] = field(
        metadata={"role": "call", "axes": (), "units": "1", "dtype": "sample_bin_indices"}
    )
    starts: tuple[tuple[np.ndarray, np.ndarray], ...] = field(
        metadata={"role": "state", "axes": (), "units": "1", "dtype": "reset_and_blocked[N,M]"}
    )
    result: PhaseComponentResult = field(
        metadata={"role": "call", "axes": (), "units": "1", "dtype": "PhaseComponentResult"}
    )


# WP_OU1 section 4.5: bit 1 marks the displayed phase ambiguity in int8 frame flags.
DISPLAYED_UNDETERMINED = 2


@dataclass(frozen=True)
class PairingTrial:
    """A measured alternative pairing; parent/bin are indices and gain is in nats."""

    parent: int = record_field("state", units="1", dtype="int64")
    bin: int = record_field("state", units="1", dtype="int64")
    gain: float = record_field("state", units="nats", dtype="float64")


@dataclass(frozen=True)
class SolveSnapshot:
    """Labels and source states at a named point of the actual ascent."""

    kind: str = record_field("state", units="1", dtype="str")
    X: np.ndarray = record_field("state", units="1", dtype="int64")
    V: np.ndarray = record_field("state", units="1", dtype="int8")


@dataclass(frozen=True)
class BlockStep:
    """Actual block winner and retained score, including the stage of initialisation."""

    kind: str = record_field("state", units="1", dtype="str")
    phase: str = record_field("state", units="1", dtype="str")
    sample: int = record_field("state", units="1", dtype="int64")
    path0: np.ndarray = record_field("state", units="1", dtype="int32")
    path1: np.ndarray = record_field("state", units="1", dtype="int32")
    gauge: np.ndarray = record_field("state", units="1", dtype="int32")
    score: float = record_field("state", units="nats", dtype="float64")


@dataclass(frozen=True)
class GainStep:
    """Measured changed rows and the actual sweep/keeping cutoff."""

    kind: str = record_field("state", units="1", dtype="str")
    phase: str = record_field("state", units="1", dtype="str")
    rows: np.ndarray = record_field("state", units="1", dtype="int64")
    gain: float = record_field("state", units="nats", dtype="float64")
    changed: np.ndarray = record_field("state", units="1", dtype="int64")
    threshold: float = record_field("state", units="nats", dtype="float64")
    above: bool = record_field("state", units="1", dtype="bool")


@dataclass(frozen=True)
class ObjectiveStep(SolveSnapshot):
    """The full joint objective at the recorded label/source configuration."""

    value: float = record_field("state", units="nats", dtype="float64")


@dataclass(frozen=True)
class PairingStep(SolveSnapshot):
    """Final candidates and state from a real pairing pass, including keep=False."""

    keep: bool = record_field("state", units="1", dtype="bool")
    kept: int = record_field("state", units="1", dtype="int64")
    records: tuple[PairingTrial, ...] = record_field("state", units="1", dtype="PairingTrial")


@dataclass(frozen=True)
class AmbiguityStep:
    """Actual ambiguity cutoff; deliberately independent of the keeping gain quantum."""

    kind: str = record_field("state", units="1", dtype="str")
    sample: int = record_field("state", units="1", dtype="int64")
    gain: float = record_field("state", units="nats", dtype="float64")
    threshold: float = record_field("state", units="nats", dtype="float64")
    above: bool = record_field("state", units="1", dtype="bool")


@dataclass(frozen=True)
class SolveReceipt:
    """A final solve plus named snapshots emitted while its native transitions execute."""

    X: np.ndarray = record_field("state", units="1", dtype="int64")
    V: np.ndarray = record_field("state", units="1", dtype="int8")
    history: tuple[float, ...] = record_field("state", units="nats", dtype="float64")
    settled: bool = record_field("state", units="1", dtype="bool")
    pairings: tuple[PairingTrial, ...] = record_field("state", units="1", dtype="PairingTrial")
    pairing_moves: tuple[PairingTrial, ...] = record_field("state", units="1", dtype="PairingTrial")
    trace: tuple[SolveSnapshot | BlockStep | GainStep | ObjectiveStep | PairingStep | AmbiguityStep, ...] = (
        record_field("state", units="1", dtype="ascent_receipts")
    )


@dataclass(frozen=True)
class StoredSampleRelease:
    """Explicit baseline storage projection: calls and scores unchanged, raw chain retained elsewhere.

    A float32 chain here is a comparison of stored values, never an input to the
    4.0 source threshold; release_details.chain64 holds its original float64.
    """

    calls: np.ndarray = record_field("call", units="1", dtype="int8")
    undetermined: np.ndarray = record_field("call", units="1", dtype="bool")
    set_breaks: tuple[int, ...] = record_field("call", units="1", dtype="bin_indices")
    scores: np.ndarray | None = record_field("report", units="1", dtype="float32?")
    chain: np.ndarray | None = record_field("state", units="1", dtype="float32?")


@dataclass(frozen=True)
class CalibrationReceipt:
    """CAL2 aggregate and flips; the shared chromosome logit prior is retained once at the root."""

    evidence: np.ndarray = record_field("state", units="nats", dtype="float64")
    flips: np.ndarray = record_field("state", units="probability", dtype="float64")


@dataclass(frozen=True)
class SamplePosteriorReceipt:
    """Raw source probabilities and the sample's leave-one-out state/flip evidence."""

    chain64: np.ndarray = record_field("state", units="probability", dtype="float64")
    own_state: np.ndarray = record_field("state", units="1", dtype="int64")
    own_flip: np.ndarray = record_field("state", units="probability", dtype="float64")


@dataclass(frozen=True)
class DecisionMargins:
    """Actual release margins, in encounter order within each registered decision kind."""

    anchor: np.ndarray = record_field("state", units="probability difference", dtype="float64")
    stop_het: np.ndarray = record_field("state", units="probability difference", dtype="float64")
    stop_orientation: np.ndarray = record_field("state", units="probability difference", dtype="float64")
    genotype: np.ndarray = record_field("state", units="probability difference", dtype="float64")
    genotype_argmax: np.ndarray = record_field("state", units="1", dtype="int64")
    genotype_gap: np.ndarray = record_field("state", units="probability difference", dtype="float64")
    phase: np.ndarray = record_field("state", units="probability difference", dtype="float64")
    orientation: np.ndarray = record_field("state", units="probability difference", dtype="float64")
    orientation_het: np.ndarray = record_field("state", units="probability difference", dtype="float64")
    bridge: np.ndarray = record_field("state", units="probability difference", dtype="float64")
    pin: np.ndarray = record_field("state", units="nats", dtype="float64")

    gain: np.ndarray = record_field("state", units="nats", dtype="float64")
    pairing: np.ndarray = record_field("state", units="nats", dtype="float64")
    ambiguity: np.ndarray = record_field("state", units="nats", dtype="float64")


@dataclass(frozen=True)
class PhaseComponentReceipt:
    """Complete component observation contract with canonical named numerical outputs.

    E and the recomputation are evidence; no solver state is relabelled as a call.
    Every event index is in one invocation's execution order. The adapter's only
    numerical projection is the declared float32 stored chain, after release.
    """

    E: np.ndarray = record_field("evidence", units="nats", dtype="float64")
    recomputed_emissions: np.ndarray = record_field("evidence", units="nats", dtype="float64")
    rates: PhaseRates = record_field("state", units="1", dtype="PhaseRates")
    par: np.ndarray = record_field("call", units="1", dtype="int64")
    solves: tuple[SolveReceipt, ...] = record_field("state", units="1", dtype="SolveReceipt")
    junctions: tuple = record_field("state", units="1", dtype="junctions")
    withheld: dict = record_field("call", units="1", dtype="sample_bin_indices")
    pairing: dict = record_field("call", units="1", dtype="sample_bin_indices")
    site_idx: np.ndarray = record_field("state", units="1", dtype="int64")
    site_bin: np.ndarray = record_field("state", units="1", dtype="int64")
    site_alleles: np.ndarray = record_field("state", units="1", dtype="int8")
    break_bins: tuple = record_field("call", units="1", dtype="sample_bin_indices")
    tie_cuts: tuple = record_field("call", units="1", dtype="sample_bin_indices")
    prior: float = record_field("state", units="nats", dtype="float64")
    uncertainty: tuple[CalibrationReceipt, ...] = record_field("state", units="1", dtype="CalibrationReceipt")
    releases: tuple[tuple[StoredSampleRelease, ...], ...] = record_field("call", units="1", dtype="StoredSampleRelease")
    release_details: tuple[tuple[SamplePosteriorReceipt, ...], ...] = record_field(
        "state", units="1", dtype="SamplePosteriorReceipt"
    )
    agreed: tuple[AgreedRelease, ...] = record_field("call", units="1", dtype="AgreedRelease")
    starts: tuple = record_field("state", units="1", dtype="reset_and_blocked")
    chains: dict = record_field("state", units="1", dtype="MeiosisChain")
    decisions: DecisionMargins = record_field("state", units="1", dtype="DecisionMargins")
    decision_events: dict[str, np.ndarray] = record_field("state", units="1", dtype="int64")


@dataclass(frozen=True)
class ViterbiCandidates:
    """Actual retained-score candidate evaluations in all four native scan stages."""

    stage: np.ndarray = record_field("state", units="1", dtype="int64")
    bin: np.ndarray = record_field("state", units="1", dtype="int64")
    destination: np.ndarray = record_field("state", units="1", dtype="int64")
    candidate: np.ndarray = record_field("state", units="1", dtype="int64")
    terminal: np.ndarray = record_field("state", units="1", dtype="int64")
    current_terminal: np.ndarray = record_field("state", units="1", dtype="int64")
    arg_before: np.ndarray = record_field("state", units="1", dtype="int64")
    incumbent: np.ndarray = record_field("state", units="1", dtype="int64")
    value: np.ndarray = record_field("state", units="nats", dtype="float64")
    best: np.ndarray = record_field("state", units="nats", dtype="float64")
    quantum: np.ndarray = record_field("state", units="nats", dtype="float64")
    preferred: np.ndarray = record_field("state", units="1", dtype="bool")
    taken: np.ndarray = record_field("state", units="1", dtype="bool")


@dataclass(frozen=True)
class ViterbiReceipt:
    """A bounded actual block recurrence and its initial-gauge and branch receipts."""

    path0: np.ndarray = record_field("state", units="1", dtype="int32")
    path1: np.ndarray = record_field("state", units="1", dtype="int32")
    gauge: np.ndarray = record_field("state", units="1", dtype="int32")
    score: float = record_field("state", units="nats", dtype="float64")
    trace: ViterbiCandidates = record_field("state", units="1", dtype="ViterbiCandidates")
    initial_scores: np.ndarray = record_field("state", units="nats", dtype="float64")
    compiled_identical: bool = record_field("state", units="1", dtype="bool")
    scratch_clean: bool = record_field("state", units="1", dtype="bool")


@dataclass(frozen=True)
class PosteriorReceipt:
    """Actual posterior terms at the strict -700 cutoff and the resulting probabilities."""

    post: np.ndarray = record_field("state", units="probability", dtype="float64")
    log_posterior: np.ndarray = record_field("state", units="nats", dtype="float64")
    indices: np.ndarray = record_field("state", units="1", dtype="int64")
    compiled_identical: bool = record_field("state", units="1", dtype="bool")


@dataclass(frozen=True)
class OrientationReceipt:
    """Actual site-inclusion operands and complete-bin orientation medians."""

    values: np.ndarray = record_field("state", units="probability", dtype="float64")
    P01: np.ndarray = record_field("state", units="probability", dtype="float64")
    P10: np.ndarray = record_field("state", units="probability", dtype="float64")
    cutoff: np.ndarray = record_field("state", units="probability", dtype="float64")


@dataclass(frozen=True)
class ConfigurationReceipt:
    """Actual set-posterior guard operands and observed constructor outcome."""

    posterior: float = record_field("state", units="probability", dtype="float64")
    ceiling: float = record_field("state", units="probability", dtype="float64")
    raised: bool = record_field("state", units="1", dtype="bool")
    message: str = record_field("state", units="1", dtype="str")


@dataclass(frozen=True)
class AscentInitialReceipt:
    """Actual state and objective after independent initialisation."""

    X: np.ndarray = record_field("state", units="1", dtype="int64")
    V: np.ndarray = record_field("state", units="1", dtype="int8")
    objective: float = record_field("state", units="nats", dtype="float64")


@dataclass(frozen=True)
class SweepReceipt:
    """One completed native global sweep; index is the loop index, never block-score scratch."""

    index: int = record_field("state", units="1", dtype="int64")
    accepted: int = record_field("state", units="1", dtype="int64")
    objective: float = record_field("state", units="nats", dtype="float64")
    X: np.ndarray = record_field("state", units="1", dtype="int64")
    V: np.ndarray = record_field("state", units="1", dtype="int8")


@dataclass(frozen=True)
class PairingPassReceipt:
    """Actual keeping or final nonkeeping pass with before/after states and candidates."""

    keep: bool = record_field("state", units="1", dtype="bool")
    before: float = record_field("state", units="nats", dtype="float64")
    X_before: np.ndarray = record_field("state", units="1", dtype="int64")
    V_before: np.ndarray = record_field("state", units="1", dtype="int8")
    kept: int = record_field("state", units="1", dtype="int64")
    after: float = record_field("state", units="nats", dtype="float64")
    records: tuple[PairingTrial, ...] = record_field("state", units="1", dtype="PairingTrial")
    X_after: np.ndarray = record_field("state", units="1", dtype="int64")
    V_after: np.ndarray = record_field("state", units="1", dtype="int8")


@dataclass(frozen=True)
class CapReceipt:
    """Actual loop usage and improvements distinguish settled exit from cap exhaustion."""

    loop: str = record_field("state", units="1", dtype="str")
    used: int = record_field("state", units="1", dtype="int64")
    limit: int = record_field("state", units="1", dtype="int64")
    improvements: int = record_field("state", units="1", dtype="int64")


@dataclass(frozen=True)
class AscentGainReceipt:
    """Sample indices and measured gain/quantum operands on separate typed axes."""

    samples: np.ndarray = record_field("state", units="1", dtype="int64")
    values: np.ndarray = record_field("state", units="nats", dtype="float64")
    cutoffs: np.ndarray = record_field("state", units="nats", dtype="float64")


@dataclass(frozen=True)
class AscentReceipt:
    """One complete native ascent with both budgets and the final ambiguity pass."""

    X: np.ndarray = record_field("state", units="1", dtype="int64")
    V: np.ndarray = record_field("state", units="1", dtype="int8")
    history: np.ndarray = record_field("state", units="nats", dtype="float64")
    settled: bool = record_field("state", units="1", dtype="bool")
    objective: float = record_field("state", units="nats", dtype="float64")
    sweeps: tuple[SweepReceipt, ...] = record_field("state", units="1", dtype="SweepReceipt")
    passes: tuple[PairingPassReceipt, ...] = record_field("state", units="1", dtype="PairingPassReceipt")
    gains: AscentGainReceipt = record_field("state", units="1", dtype="AscentGainReceipt")
    final_nonkeeping_pass: bool = record_field("state", units="1", dtype="bool")
    max_sweeps: int = record_field("state", units="1", dtype="int64")
    cap_events: tuple[CapReceipt, ...] = record_field("state", units="1", dtype="CapReceipt")
    initial: AscentInitialReceipt = record_field("state", units="1", dtype="AscentInitialReceipt")
    uninstrumented_identical: bool = record_field("state", units="1", dtype="bool")


@dataclass(frozen=True)
class PhaseMechanismReceipt:
    """One selected mechanism and its actual-producer receipts; other branches are explicit empty containers."""

    kind: str = record_field("state", units="1", dtype="str")
    block: ViterbiReceipt | dict = record_field("state", units="1", dtype="ViterbiReceipt")
    solve: AscentReceipt | dict = record_field("state", units="1", dtype="AscentReceipt")
    posterior: PosteriorReceipt | dict = record_field("state", units="1", dtype="PosteriorReceipt")
    orientation: OrientationReceipt | dict = record_field("state", units="1", dtype="OrientationReceipt")
    config: tuple[ConfigurationReceipt, ...] = record_field("state", units="1", dtype="ConfigurationReceipt")


@dataclass(frozen=True)
class ReleaseTables:
    """Owned factors for one sample's exact block posterior; label axes preserve native scan order."""

    emissions: np.ndarray = record_field("state", units="nats", dtype="float64")
    transition0: np.ndarray = record_field("state", units="nats", dtype="float64")
    transition1: np.ndarray = record_field("state", units="nats", dtype="float64")
    gauge_transition: np.ndarray = record_field("state", units="nats", dtype="float64")
    labels0: np.ndarray = record_field("state", units="1", dtype="int64")
    labels1: np.ndarray = record_field("state", units="1", dtype="int64")
    kinds: np.ndarray = record_field("state", units="1", dtype="int64")


@dataclass(frozen=True)
class PhaseSummary:
    """Native per-contig summary columns; elapsed time is supplied from execution records."""

    contig: str = record_field("obs", units="1", dtype="str")
    samples: int = record_field("report", units="count", dtype="int64")
    sites: int = record_field("report", units="count", dtype="int64")
    components: int = record_field("report", units="count", dtype="int64")
    sweeps: int = record_field("report", units="count", dtype="int64")
    settled: bool = record_field("report", units="1", dtype="bool")
    objective: float = record_field("report", units="nats", dtype="float64")
    junction_candidates: int = record_field("report", units="count", dtype="int64")
    withheld_joins: int = record_field("report", units="count", dtype="int64")
    tie_breaks: int = record_field("report", units="count", dtype="int64")
    called_alleles: int = record_field("report", units="count", dtype="int64")
    phased_heterozygotes: int = record_field("report", units="count", dtype="int64")
    unphased_heterozygotes: int = record_field("report", units="count", dtype="int64")
    phase_sets: int = record_field("report", units="count", dtype="int64")
    elapsed_seconds: float | None = record_field("diagnostic", units="seconds", dtype="float64 | None", default=None)


@dataclass(frozen=True)
class PhaseBoundaryOperands:
    """Actual branch order and numerical operands, separated by physical units."""

    events: np.ndarray = record_field("state", units="1", dtype="int64")
    indices: tuple[np.ndarray, ...] = record_field("state", units="1", dtype="int64")
    probabilities: tuple[np.ndarray, ...] = record_field("state", units="probability", dtype="float64")
    log_likelihoods: tuple[np.ndarray, ...] = record_field("state", units="nats", dtype="float64")
    raw_probabilities: tuple[np.ndarray, ...] = record_field("state", units="probability", dtype="float64")
    raw_log_likelihoods: tuple[np.ndarray, ...] = record_field("state", units="nats", dtype="float64")


@dataclass(frozen=True)
class PhaseBoundaryReceipt:
    """Bounded actual PH2 producer results and the raw decisions that produced them."""

    kind: str = record_field("state", units="1", dtype="str")
    pinned: tuple[bool, ...] = record_field("call", units="1", dtype="bool")
    junctions: tuple[tuple[tuple[int, ...], ...], ...] = record_field("call", units="1", dtype="int")
    chains: tuple[dict[str, MeiosisChain], ...] = record_field("state", units="1", dtype="MeiosisChain")
    flips: tuple[np.ndarray, ...] = record_field("state", units="probability", dtype="float64")
    ties: tuple[dict[str, tuple[int, ...]], ...] = record_field("call", units="1", dtype="int")
    bridged: tuple[tuple[int, ...], ...] = record_field("call", units="1", dtype="int")
    releases: tuple[StoredSampleRelease, ...] = record_field("call", units="1", dtype="StoredSampleRelease")
    trace: dict[str, PhaseBoundaryOperands] = record_field("state", units="1", dtype="PhaseBoundaryOperands")
