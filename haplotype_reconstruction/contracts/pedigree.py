"""Carry decision inputs and released pedigree calls across the fit-to-release seam.

Owns the pedigree decision records (PD01-PD27). No storage access or scoring is performed.
Method: docs/methods/pedigree.md#products-and-published-tables. Inputs are borrowed read-only; results own their arrays.
Ordered axes and roles are field metadata.
Stage decoding freezes arrays; scientific entry points never mutate borrowed inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import Field, dataclass, fields
from typing import Any, ClassVar

import numpy as np

from ..core.errors import InputError
from ..core.ties import TieCounts
from .roles import freeze_record_arrays, record_field, schema_field

M0, M1, M2 = 0, 1, 2
UNRESOLVED = NO_PARENT = -1
STATE_NAMES = ("zero_observed_parents", "one_observed_parent", "two_observed_parents")


@dataclass(frozen=True)
class CandidatePanel:
    """Genome-wide fixed M2 panel.

    trios:         int64 [T, 3] rows (child, first, second), first < second, sorted by child then pair;
                   the canonical retained pairs from the bounded genome-wide selective search.
    screen_scores: float64 [N, N] robust screen utility [child, parent]; -inf off the eligible edges.
    capped_children: the children whose admitted pairs exceeded PanelParameters.max_pairs_per_child (ascending); their
                   panel rows are the best pairs by bound gap, whole tie classes kept.
    """

    trios: np.ndarray = record_field("state", axes=("T", "3"), units="1", dtype="int64")
    screen_scores: np.ndarray = record_field("report", axes=("N", "N"), units="1", dtype="float64")
    capped_children: tuple[int, ...] = record_field("report", axes=("C",), units="1", dtype="int64", default=())
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def edge_index(edges: np.ndarray, n_samples: int, children: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """int64 positions of the (child, parent) pairs on the edge axis ``edges`` (GenomeScores.edges, sorted
    child-major); -1 for a pair that is not an eligible edge."""
    keys = edges[:, 0] * n_samples + edges[:, 1]
    wanted = np.asarray(children, dtype=np.int64) * n_samples + np.asarray(parents, dtype=np.int64)
    index = np.searchsorted(keys, wanted)
    found = index < len(keys)
    found[found] = keys[index[found]] == wanted[found]
    return np.where(found, index, -1).astype(np.int64)


@dataclass(frozen=True)
class GenomeScores:
    """Decision input: every informative contig's scores stacked in contig order (not a record).

    contigs: informative contigs (C); arrays carry a leading contig axis C.
    edges:   int64 [E, 2] the eligible (child, parent) edges, child-major with parents ascending
             (np.argwhere(eligibility.parents)). one, edge_matched and edge_exposed are float64 [C, E] on this
             axis: the chromosome records' [N, N] entries off the eligible edges (M1 -inf, exposure
             of non-edges and self coverage) are never read by the decision or the report.
    zero float64 [C, N]; two, pair_explained, pair_exposed float64 [C, T] (panel rows); total_bins float64 [C];
    informative_markers int64 [C]; junctions / callable_bins float64 [C, N]; junction_dyads int64 [D, 2] and
    transmission_log_likelihood float64 [C, D] in nats (ChromosomeScores);
    scored_morgans float64 [C] (NaN where unknown);
    working_morgans float64 [C]: sum of -0.5 ln(1 - 2 theta) over scored components' selector boundaries,
    the configured working-map distance of PS04. It is NOT measured exposure, and never enters a gate or
    decision; reports alone use it to describe painted junctions per Morgan.
    edge_exposed_morgans float64 [C, E] and pair_exposed_morgans float64 [C, T] (ChromosomeScores; 0 in earlier
    records).
    """

    sample_ids: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    contigs: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    panel: CandidatePanel = record_field("evidence", axes=(), units="1", dtype="CandidatePanel")
    edges: np.ndarray = record_field("evidence", axes=("E", "2"), units="1", dtype="int64")
    zero: np.ndarray = record_field("evidence", axes=("C", "N"), units="nats", dtype="float64")
    one: np.ndarray = record_field("evidence", axes=("C", "E"), units="nats", dtype="float64")
    two: np.ndarray = record_field("evidence", axes=("C", "T"), units="nats", dtype="float64")
    edge_matched: np.ndarray = record_field("evidence", axes=("C", "E"), units="painting bins", dtype="float64")
    edge_exposed: np.ndarray = record_field("evidence", axes=("C", "E"), units="painting bins", dtype="float64")
    pair_explained: np.ndarray = record_field("evidence", axes=("C", "T"), units="painting bins", dtype="float64")
    pair_exposed: np.ndarray = record_field("evidence", axes=("C", "T"), units="painting bins", dtype="float64")
    total_bins: np.ndarray = record_field("evidence", axes=("C",), units="painting bins", dtype="float64")
    junctions: np.ndarray = record_field("evidence", axes=("C", "N"), units="painting bins", dtype="float64")
    callable_bins: np.ndarray = record_field("evidence", axes=("C", "N"), units="painting bins", dtype="float64")
    informative_markers: np.ndarray = record_field("obs", axes=("C",), units="1", dtype="int64")
    junction_dyads: np.ndarray = record_field("evidence", axes=("D", "2"), units="1", dtype="int64")
    transmission_log_likelihood: np.ndarray = record_field("evidence", axes=("C", "D"), units="nats", dtype="float64")
    scored_morgans: np.ndarray | None = record_field(
        "evidence", axes=("C",), units="Morgans", dtype="float64", default=None
    )
    edge_exposed_morgans: np.ndarray | None = record_field(
        "evidence", axes=("C", "E"), units="Morgans", dtype="float64", default=None
    )
    pair_exposed_morgans: np.ndarray | None = record_field(
        "evidence", axes=("C", "T"), units="Morgans", dtype="float64", default=None
    )

    working_morgans: np.ndarray = record_field(
        "report", axes=("C",), units="Morgans", dtype="float64", kw_only=True
    )

    def edge_starts(self) -> np.ndarray:
        """int64 [N + 1]: child c's eligible edges are edges[starts[c]:starts[c + 1]] (parents ascending)."""
        return np.searchsorted(self.edges[:, 0], np.arange(len(self.sample_ids) + 1)).astype(np.int64)


@dataclass(frozen=True)
class ExclusionEvidence:
    """Full-marker exclusion of one contig (record ``pedigree/exclusion``).

    dyads:          int64 [D, 2] (child, candidate parent) in target order.
    log_e:          float64 [D] exact log E, or a certified lower endpoint.
    log_e_upper:    optional float64 [D] upper endpoint; None means exact evidence without a separate upper endpoint.
    observed_sites: int64 [D] sites observed in both samples.
    markers:        markers of the contig; pattern_fallback_samples: samples scored without the
                    exact repeated-pattern cache (more than its cap of distinct GL triples).
    """

    contig: str = record_field("state", axes=(), units="1", dtype="str")
    dyads: np.ndarray = record_field("state", axes=("D", "2"), units="1", dtype="int64")
    log_e: np.ndarray = record_field("state", axes=("D",), units="nats", dtype="float64")
    observed_sites: np.ndarray = record_field("obs", axes=("D",), units="1", dtype="int64")
    markers: int = record_field("obs", axes=(), units="1", dtype="int")
    pattern_fallback_samples: int = record_field("diagnostic", axes=(), units="1", dtype="int")
    log_e_upper: np.ndarray | None = record_field("state", axes=("D",), units="nats", dtype="float64", default=None)
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PedigreeCalls:
    """One relationship view over the run's samples.

    state:   int8 [N] observed-parent count 0/1/2, or UNRESOLVED (-1).
    parents: int64 [N, 2] parent sample indices, NO_PARENT (-1) for an empty or unresolved slot;
             parent order is arbitrary (sample-array order of the configuration).
    status:  inference status per sample (e.g. "tier_b_supported_one_observed_parent").
    """

    state: np.ndarray = record_field("call", axes=("N",), units="1", dtype="int8")
    parents: np.ndarray = record_field("call", axes=("N", "2"), units="1", dtype="int64")
    status: tuple[str, ...] = record_field("call", axes=(), units="1", dtype="tuple[str, ...]")


@dataclass(frozen=True)
class ParentEdges:
    """Supported Tier-B parent -> child edges: the pedigree input of family phase and recombination.

    parents, children, slots: int64 [E]; ``slots`` is 0 for the child's Parent1 column, 1 for Parent2.
    Edges are ordered by child, then slot. An edge does not establish whether the child's other
    parent was sampled; unreleased candidates never appear.
    observed_parent_minimum / observed_parent_maximum: int64 [N] bounds of each sample's
    observed-parent count (minimum = released edges, maximum = 2 unless the count is resolved).
    """

    sample_ids: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    parents: np.ndarray = record_field("call", axes=("Q",), units="1", dtype="int64")
    children: np.ndarray = record_field("call", axes=("Q",), units="1", dtype="int64")
    slots: np.ndarray = record_field("call", axes=("Q",), units="1", dtype="int64")
    observed_parent_minimum: np.ndarray = record_field("call", axes=("N",), units="1", dtype="int64")
    observed_parent_maximum: np.ndarray = record_field("call", axes=("N",), units="1", dtype="int64")

    @property
    def identity(self) -> str:
        """SHA-256 of the sample axis and the edge list (what conditioning depends on)."""
        digest = hashlib.sha256(json.dumps(list(self.sample_ids)).encode())
        for array in (self.parents, self.children, self.slots):
            digest.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class TierReach:
    """What the chromosome resampling can release with this many informative chromosomes.

    A non-M0 configuration is testable in a fit only when its edges are exposed on chromosomes holding at least
    ExposureParameters.minimum_exposed_units map units, which caps the achievable support:
    max_bootstrap_support = fraction of this run's bootstrap draws whose distinct chromosomes hold that many units;
    max_loco_support = fraction of the leave-one-out fits that do. ``minimum_exposed_contigs`` is the number of
    chromosomes shorter than one unit that the gate needs; ``minimum_contigs_tier_*`` count such chromosomes too
    (longer chromosomes need fewer). informative_units: the run's informative chromosomes' map units.
    tier_*_reachable: whether those caps meet the tier thresholds. minimum_contigs_tier_*: the
    smallest chromosome count for which they would (exact combinatorics). ``statement``: one line
    for the log and reports.
    """

    informative_contigs: int = record_field("obs", axes=(), units="1", dtype="int")
    minimum_exposed_contigs: int = record_field("report", axes=(), units="1", dtype="int")
    max_bootstrap_support: float = record_field("report", axes=(), units="1", dtype="float")
    max_loco_support: float = record_field("report", axes=(), units="1", dtype="float")
    tier_a_reachable: bool = record_field("report", axes=(), units="1", dtype="bool")
    tier_b_reachable: bool = record_field("report", axes=(), units="1", dtype="bool")
    minimum_contigs_tier_a: int = record_field("report", axes=(), units="1", dtype="int")
    minimum_contigs_tier_b: int = record_field("report", axes=(), units="1", dtype="int")
    statement: str = record_field("report", axes=(), units="1", dtype="str")
    informative_units: float = record_field("report", axes=(), units="map units", dtype="float", default=float("nan"))
    unit_morgans: float = record_field("report", axes=(), units="Morgans", dtype="float", default=float("nan"))


@dataclass(frozen=True)
class Alternatives:
    """M0/M1/M2 configuration rows (A rows) grouped by child.

    rows:            int64 [A, 3] (child, parent, parent); NO_PARENT in empty slots. Per child: the M0
                     row, M1 rows (parents ascending), then M2 rows (panel order).
    states:          int8 [A] observed-parent count of each row.
    child_rows:      per sample, int64 row indices (empty for samples that are not targets).
    m0_row:          int64 [A] row index of the same child's M0 row.
    panel_row:       int64 [A] index of an M2 row in the panel trios, -1 for M0/M1 rows.
    log_likelihoods: float64 [C, A] per-chromosome composite log likelihood of each row.
    full_counts:     int64 [N, 3] eligible universe size per state (identity-prior denominator).
    scored_counts:   int64 [N, 3] rows actually scored per state (M2 is screened).
    """

    rows: np.ndarray = record_field("state", axes=("A", "3"), units="1", dtype="int64")
    states: np.ndarray = record_field("state", axes=("A",), units="1", dtype="int8")
    child_rows: tuple[np.ndarray, ...] = record_field("state", axes=("N", "A_child"), units="1", dtype="int64")
    m0_row: np.ndarray = record_field("state", axes=("A",), units="1", dtype="int64")
    panel_row: np.ndarray = record_field("state", axes=("A",), units="1", dtype="int64")
    log_likelihoods: np.ndarray = record_field("state", axes=("C", "A"), units="nats", dtype="float64")
    full_counts: np.ndarray = record_field("state", axes=("N", "3"), units="1", dtype="int64")
    scored_counts: np.ndarray = record_field("state", axes=("N", "3"), units="1", dtype="int64")

    @property
    def n_samples(self) -> int:
        return len(self.child_rows)

@dataclass(frozen=True)
class JunctionEvidence:
    """Per-chromosome painted junction data of the informative chromosomes (weight independent).

    junctions:     float64 [C, N] minimum founder switches of each painting.
    callable_bins: float64 [C, N] observed haplotype-bin labels behind them.
    total_bins:    float64 [C] painting bins (two haplotypes each).
    pair_child, pair_parent: int64 [U] the unordered dyads with both directions oriented, as (larger index,
                   smaller index) like the old [child, parent] (parent < child) triangle.
    pair_log_odds: float64 [C, U] per-chromosome log odds of pair_parent -> pair_child against the reverse;
                   ``compared`` bool [C, U] marks entries with evidence. Computed once; resamples only reweight
                   it. Stored per dyad, not as [C, N, N] triangles.
    """

    junctions: np.ndarray = record_field("evidence", axes=("C", "N"), units="painting bins", dtype="float64")
    callable_bins: np.ndarray = record_field("evidence", axes=("C", "N"), units="painting bins", dtype="float64")
    total_bins: np.ndarray = record_field("evidence", axes=("C",), units="painting bins", dtype="float64")
    pair_child: np.ndarray = record_field("evidence", axes=("U",), units="1", dtype="int64")
    pair_parent: np.ndarray = record_field("evidence", axes=("U",), units="1", dtype="int64")
    pair_log_odds: np.ndarray = record_field("evidence", axes=("C", "U"), units="nats", dtype="float64")
    compared: np.ndarray = record_field("evidence", axes=("C", "U"), units="1", dtype="bool")


@dataclass(frozen=True)
class Orientation:
    """Direction evidence of one (re)weighted fit.

    order_probability: float64 [N, N] [child, parent] paired junction-density contrast in [0, 1]
                       (0.5 = no information); orders the graph search, reported, never scored.
    log_odds:          float64 [N, N] [child, parent] weighted sum of the chromosome log odds of
                       parent -> child against child -> parent (antisymmetric); score evidence.
    adjusted_burden:   float64 [N] weighted junctions / callability fraction (report).
    callability:       float64 [N] weighted callable bins / (2 x weighted total bins) (report).
    """

    order_probability: np.ndarray = record_field("state", axes=("N", "N"), units="1", dtype="float64")
    log_odds: np.ndarray = record_field("state", axes=("N", "N"), units="nats", dtype="float64")
    adjusted_burden: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    callability: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")


@dataclass(frozen=True)
class ExposureEvidence:
    """Posterior-expected exposure of every eligible edge and panel pair, per informative chromosome.

    edge_exposed: float64 [C, E] on the eligible edges (GenomeScores.edges); pair_exposed: float64 [C, T];
    total_bins: float64 [C].
    edge_present / pair_present: uint64 [W, E] / [W, T] packed per-chromosome flags of
    exposure > 0 and >= the minimum bins (W = ceil(C / 64)).
    units: float64 [C] map units of each chromosome (map_units of its scored length); edge_units / pair_units:
    float64 [C, E] / [C, T] the units each edge / pair earns there (map_units of its exposed map length:
    GenomeScores.edge_exposed_morgans / pair_exposed_morgans); a fit sums them over its exposed chromosomes.
    row_edges: int64 [A, 2] edge-axis positions of each configuration row's (child, parent) edges, one per
    parent slot; -1 for an empty slot.
    """

    edge_exposed: np.ndarray = record_field("evidence", axes=("C", "E"), units="painting bins", dtype="float64")
    pair_exposed: np.ndarray = record_field("evidence", axes=("C", "T"), units="painting bins", dtype="float64")
    total_bins: np.ndarray = record_field("evidence", axes=("C",), units="painting bins", dtype="float64")
    edge_present: np.ndarray = record_field("evidence", axes=("ceil(C/64)", "E"), units="1", dtype="uint64")
    pair_present: np.ndarray = record_field("evidence", axes=("ceil(C/64)", "T"), units="1", dtype="uint64")
    row_edges: np.ndarray = record_field("evidence", axes=("A", "2"), units="1", dtype="int64")
    units: np.ndarray = record_field("evidence", axes=("C",), units="1", dtype="float64")
    edge_units: np.ndarray = record_field("evidence", axes=("C", "E"), units="1", dtype="float64")
    pair_units: np.ndarray = record_field("evidence", axes=("C", "T"), units="1", dtype="float64")


@dataclass(frozen=True)
class DecisionInputs:
    """What every weighted fit reads; shared read-only with resampling workers."""

    alternatives: Alternatives = record_field("evidence", axes=(), units="1", dtype="Alternatives")
    exposure: ExposureEvidence = record_field("evidence", axes=(), units="1", dtype="ExposureEvidence")
    junctions: JunctionEvidence = record_field("evidence", axes=(), units="1", dtype="JunctionEvidence")


@dataclass(frozen=True)
class Selection:
    """Decisions of one fit over the N samples (A configuration rows).

    state_log_evidence, state_scores (+ log prior), state_support (softmax): float64 [N, 3].
    decision_scores: float64 [A] state score + identity score - the state's best identity score.
    priors: float64 [N, 3] fixed state priors used.
    local_state int8 [N], local_row int64 [N]: marginal winners (UNRESOLVED / -1 = none).
    graph_row int64 [N]: acyclic selection (-1 = none); tie_conflict / direction_resolved bool [N];
    role_probability float64 [N] (NaN without a graph row).
    state_margin, identity_margin float64 [N] (identity NaN unless the state was resolved).
    unresolved_reason: per sample None or the reason no local configuration was chosen.
    """

    state_log_evidence: np.ndarray = record_field("state", axes=("N", "3"), units="nats", dtype="float64")
    state_scores: np.ndarray = record_field("state", axes=("N", "3"), units="nats", dtype="float64")
    state_support: np.ndarray = record_field("state", axes=("N", "3"), units="1", dtype="float64")
    decision_scores: np.ndarray = record_field("state", axes=("A",), units="nats", dtype="float64")
    priors: np.ndarray = record_field("state", axes=("N", "3"), units="1", dtype="float64")
    local_state: np.ndarray = record_field("state", axes=("N",), units="1", dtype="int8")
    local_row: np.ndarray = record_field("state", axes=("N",), units="1", dtype="int64")
    graph_row: np.ndarray = record_field("state", axes=("N",), units="1", dtype="int64")
    tie_conflict: np.ndarray = record_field("state", axes=("N",), units="1", dtype="bool")
    direction_resolved: np.ndarray = record_field("state", axes=("N",), units="1", dtype="bool")
    role_probability: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    state_margin: np.ndarray = record_field("state", axes=("N",), units="nats", dtype="float64")
    identity_margin: np.ndarray = record_field("state", axes=("N",), units="nats", dtype="float64")
    unresolved_reason: tuple[str | None, ...] = record_field(
        "state", axes=(), units="1", dtype="tuple[str | None, ...]"
    )

    graph_passes_used: int = record_field("diagnostic", dtype="int", default=0, kw_only=True)
    graph_pass_budget_exhausted: bool = record_field("diagnostic", dtype="bool", default=False, kw_only=True)


@dataclass(frozen=True)
class LocalGraphChoices:
    """Local marginal winners and their confidence, before acyclic graph selection (PD13)."""

    rows: dict[int, int] = record_field("state", axes=("selected_child",), dtype="int64")
    state_margins: np.ndarray = record_field("state", axes=("N",), units="nats", dtype="float64")
    identity_margins: np.ndarray = record_field("state", axes=("N",), units="nats", dtype="float64")


@dataclass(frozen=True)
class GraphSelection:
    """rows: child -> selected configuration row; direction_resolved: children whose local row was
    displaced by a non-M0 alternative; role_probability: child -> orientation support of its row."""

    rows: dict[int, int] = record_field("state", axes=("selected_child",), units="1", dtype="int64")
    direction_resolved: frozenset[int] = record_field("state", axes=(), units="1", dtype="frozenset[int]")
    role_probability: dict[int, float] = record_field(
        "state", axes=("selected_child",), units="probability", dtype="float64"
    )

    graph_passes_used: int = record_field("diagnostic", dtype="int", default=0, kw_only=True)
    graph_pass_budget_exhausted: bool = record_field("diagnostic", dtype="bool", default=False, kw_only=True)


@dataclass(frozen=True)
class EdgeCounts:
    """Integer counts per (child, parent) edge named by some configuration row, read as ``counts[child, parent]``
    (0 for an edge no row names). keys int64 [E] sorted child * N + parent; values int64 [E] (one value per edge)."""

    keys: np.ndarray = record_field("call", axes=("Q",), units="1", dtype="int64")
    values: np.ndarray = record_field("call", axes=("Q",), units="1", dtype="int64")
    samples: int = record_field("call", axes=(), units="1", dtype="int")

    def __getitem__(self, edge: tuple[int, int]) -> int:
        child, parent = edge
        key = int(child) * self.samples + int(parent)
        index = int(np.searchsorted(self.keys, key))
        return int(self.values[index]) if index < len(self.keys) and self.keys[index] == key else 0


@dataclass(frozen=True)
class SupportCounts:
    """Integer counts over resampled fits.

    local_configuration / graph_configuration int64 [A]: fits whose local / graph row is the row.
    local_state int64 [N, 3]: fits calling each marginal state.
    local_parent / graph_parent EdgeCounts: fits whose local / graph row names [child, parent].
    fits: number of fits (the denominator); nonconverged: fits that abstained.
    """

    local_configuration: np.ndarray = record_field("call", axes=("A",), units="1", dtype="int64")
    graph_configuration: np.ndarray = record_field("call", axes=("A",), units="1", dtype="int64")
    local_state: np.ndarray = record_field("call", axes=("N", "3"), units="1", dtype="int64")
    local_parent: EdgeCounts = record_field("call", axes=(), units="1", dtype="EdgeCounts")
    graph_parent: EdgeCounts = record_field("call", axes=(), units="1", dtype="EdgeCounts")
    fits: int = record_field("call", axes=(), units="1", dtype="int")
    nonconverged: int = record_field("call", axes=(), units="1", dtype="int")

    @classmethod
    def zeros(cls, n_alternatives: int, n_samples: int, edges: np.ndarray) -> SupportCounts:
        return cls(
            np.zeros(n_alternatives, np.int64),
            np.zeros(n_alternatives, np.int64),
            np.zeros((n_samples, 3), np.int64),
            EdgeCounts(edges, np.zeros(len(edges), np.int64), n_samples),
            EdgeCounts(edges, np.zeros(len(edges), np.int64), n_samples),
            0,
            0,
        )

    def fraction(self, count: int) -> float:
        """count / fits; 0 without fits."""
        return float(count / self.fits) if self.fits > 0 else 0.0


@dataclass(frozen=True)
class Resampling:
    bootstrap: SupportCounts = record_field("call", axes=(), units="1", dtype="SupportCounts")
    loco: SupportCounts = record_field("call", axes=(), units="1", dtype="SupportCounts")
    multiplicities: np.ndarray = record_field("call", axes=("R", "C"), units="1", dtype="float64")
    worker_count: int = record_field("diagnostic", axes=(), units="1", dtype="int")
    bootstrap_graph_budget_exhausted: int = record_field("diagnostic", dtype="int", default=0, kw_only=True)
    loco_graph_budget_exhausted: int = record_field("diagnostic", dtype="int", default=0, kw_only=True)
    maximum_graph_passes_used: int = record_field("diagnostic", dtype="int", default=0, kw_only=True)


@dataclass(frozen=True)
class ChildEvidence:
    """Per-child quantities the release rules read (N samples; NaN where undefined).

    local_parents / graph_parents: int64 [N, 2] parents of the local / graph row (NO_PARENT).
    graph_state: int8 [N]; graph_conflict: bool [N] (tie conflict or graph row != local row).
    informative_units: float64 [N] map units of the chromosomes whose own evidence picks a unique state.
    state_bootstrap / state_loco: support of the local state.
    configuration_*: support of the local row as local / graph configuration.
    parent_bootstrap / parent_loco: float64 [N, 2] local-parent support per slot.
    graph_parent_bootstrap / graph_parent_loco: float64 [N, 2] graph-parent support per slot.
    graph_configuration_bootstrap / _loco: support of the graph row as graph configuration.
    orientation_resolved: bool [N, 2] per local parent slot (True for empty slots).
    m0_state_agreement / m0_graph_agreement / m0_veto: bool [N] prior-sensitivity checks of M0.
    """

    local_parents: np.ndarray = record_field("state", axes=("N", "2"), units="1", dtype="int64")
    graph_parents: np.ndarray = record_field("state", axes=("N", "2"), units="1", dtype="int64")
    graph_state: np.ndarray = record_field("state", axes=("N",), units="1", dtype="int8")
    graph_conflict: np.ndarray = record_field("state", axes=("N",), units="1", dtype="bool")
    informative_units: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    state_bootstrap: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    state_loco: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    configuration_bootstrap: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    configuration_loco: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    configuration_graph_bootstrap: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    configuration_graph_loco: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    parent_bootstrap: np.ndarray = record_field("state", axes=("N", "2"), units="1", dtype="float64")
    parent_loco: np.ndarray = record_field("state", axes=("N", "2"), units="1", dtype="float64")
    graph_parent_bootstrap: np.ndarray = record_field("state", axes=("N", "2"), units="1", dtype="float64")
    graph_parent_loco: np.ndarray = record_field("state", axes=("N", "2"), units="1", dtype="float64")
    graph_configuration_bootstrap: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    graph_configuration_loco: np.ndarray = record_field("state", axes=("N",), units="1", dtype="float64")
    orientation_resolved: np.ndarray = record_field("state", axes=("N", "2"), units="1", dtype="bool")
    m0_state_agreement: np.ndarray = record_field("state", axes=("N",), units="1", dtype="bool")
    m0_graph_agreement: np.ndarray = record_field("state", axes=("N",), units="1", dtype="bool")
    m0_veto: np.ndarray = record_field("state", axes=("N",), units="1", dtype="bool")


@dataclass(frozen=True)
class TierCalls:
    """Exact configuration view of one tier (or the complete DAG view).

    state int8 [N] (UNRESOLVED), row int64 [N] (-1 = identity not released), status tuple[str];
    for tiers also state_pass / identity_pass / orientation_withheld bool [N] and parent_flags
    bool [N, 2] (per local parent slot: resampling-stable and oriented).
    """

    state: np.ndarray = record_field("call", axes=("N",), units="1", dtype="int8")
    row: np.ndarray = record_field("call", axes=("N",), units="1", dtype="int64")
    status: tuple[str, ...] = record_field("call", axes=(), units="1", dtype="tuple[str, ...]")
    state_pass: np.ndarray | None = record_field("call", axes=("N",), units="1", dtype="bool", default=None)
    identity_pass: np.ndarray | None = record_field("call", axes=("N",), units="1", dtype="bool", default=None)
    orientation_withheld: np.ndarray | None = record_field("call", axes=("N",), units="1", dtype="bool", default=None)
    parent_flags: np.ndarray | None = record_field("call", axes=("N", "2"), units="1", dtype="bool", default=None)


@dataclass(frozen=True)
class PartialEdges:
    """Edge-level release of one tier.

    parents int64 [N, 2] released or exact-configuration parents per slot (NO_PARENT otherwise);
    supported bool [N, 2]; exact bool [N] (exact configuration released); count_minimum /
    count_maximum int64 [N] observed-parent count bounds; state int8 [N] (from the exact view);
    status tuple[str]; release_evidence tuple[str] ("chromosome_resampling" or, after the
    full-marker release, "mendelian_exclusion_and_supported_direction").
    """

    parents: np.ndarray = record_field("call", axes=("N", "2"), units="1", dtype="int64")
    supported: np.ndarray = record_field("call", axes=("N", "2"), units="1", dtype="bool")
    exact: np.ndarray = record_field("call", axes=("N",), units="1", dtype="bool")
    count_minimum: np.ndarray = record_field("call", axes=("N",), units="1", dtype="int64")
    count_maximum: np.ndarray = record_field("call", axes=("N",), units="1", dtype="int64")
    state: np.ndarray = record_field("call", axes=("N",), units="1", dtype="int8")
    status: tuple[str, ...] = record_field("call", axes=(), units="1", dtype="tuple[str, ...]")
    release_evidence: tuple[str, ...] = record_field("call", axes=(), units="1", dtype="tuple[str, ...]")


@dataclass(frozen=True)
class CandidateSets:
    """Tier-B candidate sets: parents [N, 2] (supported singletons), side candidates and the
    configuration candidates (tuples of sample indices), exact flags and statuses."""

    parents: np.ndarray = record_field("report", axes=("N", "2"), units="1", dtype="int64")
    side_candidates: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = record_field(
        "report", axes=(), units="1", dtype="tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]"
    )
    configurations: tuple[tuple[tuple[int, ...], ...], ...] = record_field(
        "report", axes=(), units="1", dtype="tuple[tuple[tuple[int, ...], ...], ...]"
    )
    exact: np.ndarray = record_field("report", axes=("N",), units="1", dtype="bool")
    status: tuple[str, ...] = record_field("report", axes=(), units="1", dtype="tuple[str, ...]")


@dataclass(frozen=True)
class CountReleaseBasis:
    """The fitted views and eligibility that the post-guard parent count may use (PD22)."""

    tier_b: TierCalls = record_field("call", dtype="TierCalls")
    partial_b: PartialEdges = record_field("call", dtype="PartialEdges")
    complete: TierCalls = record_field("state", dtype="TierCalls")
    graph_conflict: np.ndarray = record_field("state", axes=("N",), dtype="bool")
    informative_units: np.ndarray = record_field("state", axes=("N",), units="map units", dtype="float64")
    rows: np.ndarray = record_field("state", axes=("A", "3"), dtype="int64")
    eligibility: ParentEligibility = record_field("state", dtype="ParentEligibility")


@dataclass(frozen=True)
class CountExclusion:
    """Genome-wide exclusion bounds for the count's target dyads (PD22)."""

    dyads: np.ndarray = record_field("state", axes=("D", "2"), dtype="int64")
    log_e: np.ndarray = record_field("state", axes=("D",), units="nats", dtype="float64")
    observed_sites: np.ndarray = record_field("obs", axes=("D",), units="count", dtype="int64")
    log_e_upper: np.ndarray | None = record_field("state", axes=("D",), units="nats", dtype="float64", default=None)


@dataclass(frozen=True)
class CountDiagnostic:
    """One target child's count evidence, using sample indices instead of report labels."""

    child: int = record_field("state", dtype="int")
    eligible_candidates: int = record_field("state", units="count", dtype="int")
    compatible: tuple[int, ...] = record_field("state", dtype="tuple[int, ...]")
    reverse: tuple[int, ...] = record_field("state", dtype="tuple[int, ...]")
    remaining: tuple[int, ...] = record_field("state", dtype="tuple[int, ...]")
    weakest_log_e: float = record_field("report", units="nats", dtype="float")
    threshold_log_e: float = record_field("state", units="nats", dtype="float")
    complete_evidence: bool = record_field("state", dtype="bool")
    released: bool = record_field("call", dtype="bool")
    uses_supported_directions: bool = record_field("state", dtype="bool")
    weakest_lower: float | None = record_field("report", units="nats", dtype="float | None", default=None)
    weakest_upper: float | None = record_field("report", units="nats", dtype="float | None", default=None)
    weakest_exact: bool | None = record_field("report", dtype="bool | None", default=None)


@dataclass(frozen=True)
class CountRelease:
    """Result of the full-marker release.

    tier_b / partial_b: the final Tier-B views (released children: row = the complete graph row,
    status "tier_b_fixed_genome_directed_exclusion", release evidence
    "mendelian_exclusion_and_supported_direction"); released bool [N]; kept: per child the
    remaining candidate (0 or 1 sample index). ``records``: one typed record per target child.
    dyads / log_e / observed_sites: the evidence used (possibly empty).
    """

    tier_b: TierCalls = record_field("call", axes=(), units="1", dtype="TierCalls")
    partial_b: PartialEdges = record_field("call", axes=(), units="1", dtype="PartialEdges")
    released: np.ndarray = record_field("call", axes=("N",), units="1", dtype="bool")
    kept: tuple[tuple[int, ...], ...] = record_field("call", axes=(), units="1", dtype="tuple[tuple[int, ...], ...]")
    records: tuple[CountDiagnostic, ...] = record_field("state", dtype="tuple[CountDiagnostic, ...]")
    dyads: np.ndarray = record_field("call", axes=("D", "2"), units="1", dtype="int64")
    log_e: np.ndarray = record_field("call", axes=("D",), units="nats", dtype="float64")
    observed_sites: np.ndarray = record_field("obs", axes=("D",), units="1", dtype="int64")
    log_e_upper: np.ndarray | None = record_field("call", axes=("D",), units="nats", dtype="float64", default=None)


@dataclass(frozen=True)
class Decision:
    """Everything the release, exclusion and report steps read.

    scores: genome-floor transformed chromosome scores; inputs: the fit inputs;
    full / orientation / adjusted (state, identity scores) / trace: the
    full-data fit and its per-row terms; sensitivity: (priors, selection) per alternative prior;
    informative: int64 [N] chromosomes whose own evidence picks a unique state; informative_units: float64 [N]
    their map units (read by the tier rules).
    """

    scores: GenomeScores = record_field("state", axes=(), units="1", dtype="GenomeScores")
    eligibility: ParentEligibility = record_field("state", axes=(), units="1", dtype="ParentEligibility")
    inputs: DecisionInputs = record_field("state", axes=(), units="1", dtype="DecisionInputs")
    full: Selection = record_field("state", axes=(), units="1", dtype="Selection")
    orientation: Orientation = record_field("state", axes=(), units="1", dtype="Orientation")
    adjusted: tuple[np.ndarray, np.ndarray] = record_field(
        "state", axes=("score_kind", "A"), units="nats", dtype="float64"
    )
    trace: DecisionTrace = record_field("state", dtype="DecisionTrace")
    sensitivity: tuple[tuple[tuple[float, float, float], Selection], ...] = record_field(
        "state", axes=(), units="1", dtype="tuple[tuple[tuple[float, float, float], Selection], ...]"
    )
    resampling: Resampling = record_field("state", axes=(), units="1", dtype="Resampling")
    informative: np.ndarray = record_field("state", axes=("N",), units="1", dtype="int64")
    evidence: ChildEvidence = record_field("state", axes=(), units="1", dtype="ChildEvidence")
    tier_a: TierCalls = record_field("state", axes=(), units="1", dtype="TierCalls")
    tier_b: TierCalls = record_field("state", axes=(), units="1", dtype="TierCalls")
    complete: TierCalls = record_field("state", axes=(), units="1", dtype="TierCalls")
    partial_a: PartialEdges = record_field("state", axes=(), units="1", dtype="PartialEdges")
    partial_b: PartialEdges = record_field("state", axes=(), units="1", dtype="PartialEdges")
    candidates: CandidateSets = record_field("state", axes=(), units="1", dtype="CandidateSets")
    reach: TierReach = record_field("state", axes=(), units="1", dtype="TierReach")
    family_solve: FamilySolveRecord = record_field("state", axes=(), units="1", dtype="FamilySolveRecord")
    tie_events: TieCounts = record_field("diagnostic", dtype="TieCounts")
    informative_units: np.ndarray = record_field("state", axes=("N",), units="map units", dtype="float64")

    @property
    def alternatives(self) -> Alternatives:
        return self.inputs.alternatives


@dataclass(frozen=True)
class EdgeFamily:
    """The members of one released edge's test (sample indices): ``supported`` = c's whole relatedness family
    without c and p; ``dropped`` = (member, reason) pairs; ``tested`` = the remaining members."""

    parent: int = record_field("state", axes=(), units="1", dtype="int")
    child: int = record_field("state", axes=(), units="1", dtype="int")
    supported: tuple[int, ...] = record_field("state", axes=(), units="1", dtype="tuple[int, ...]")
    dropped: tuple[tuple[int, str], ...] = record_field(
        "state", axes=(), units="1", dtype="tuple[tuple[int, str], ...]"
    )
    tested: tuple[int, ...] = record_field("state", axes=(), units="1", dtype="tuple[int, ...]")

    ring: tuple[int, ...] = record_field("state", units="1", dtype="tuple[int, ...]", default=())


@dataclass(frozen=True)
class GuardVote:
    """One edge's final relationship evidence and irreversible withdrawal (PD23)."""

    family: EdgeFamily = record_field("state", dtype="EdgeFamily")
    set_aside: int = record_field("state", units="count", dtype="int")
    against: int = record_field("state", units="count", dtype="int")
    supporting: int = record_field("state", units="count", dtype="int")
    pair_score: float = record_field("evidence", units="nats", dtype="float64")
    family_score: float = record_field("evidence", units="nats", dtype="float64")
    score: float = record_field("evidence", units="nats", dtype="float64")
    relationship_basis: str = record_field("call", dtype="str")
    call: str = record_field("call", dtype="str")
    evidence_call: str = record_field("call", dtype="str")
    withdrawal_iteration: int = record_field("state", dtype="int")
    withholding_basis: str = record_field("state", dtype="str")


@dataclass(frozen=True)
class GuardRecord:
    """Final sibling-guard votes, in the input edge order; no report tables."""

    votes: tuple[GuardVote, ...] = record_field("state", axes=(), units="1", dtype="tuple[GuardVote, ...]")
    cleaning_iterations: int = record_field("state", axes=(), units="1", dtype="int")
    cleaning_converged: bool = record_field("state", axes=(), units="1", dtype="bool")
    cleaning_iteration_bound: int = record_field("state", axes=(), units="1", dtype="int")

    @property
    def withheld(self) -> frozenset[tuple[int, int]]:
        return frozenset((v.family.parent, v.family.child) for v in self.votes if v.call == "withheld")


@dataclass(frozen=True)
class GuardRelatedness:
    """The only relatedness state read by the guard: sample axis, family and support."""

    sample_ids: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    family: np.ndarray = record_field("evidence", axes=("N",), units="1", dtype="int64")
    member_support: np.ndarray = record_field("evidence", axes=("N",), units="1", dtype="float64")


@dataclass(frozen=True)
class Pedigree:
    """Complete released pedigree and retained numerical state, published atomically as schema 7 with working-map
    diagnostics, joint transmission direction likelihoods and columnar exclusion evidence."""

    sample_ids: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    contigs: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    informative_contigs: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    omitted: dict[str, str] = record_field("call", axes=(), units="1", dtype="dict[str, str]")
    calls: PedigreeCalls = record_field("call", axes=(), units="1", dtype="PedigreeCalls")
    edges: ParentEdges = record_field("call", axes=(), units="1", dtype="ParentEdges")
    tier_a: PedigreeCalls = record_field("call", axes=(), units="1", dtype="PedigreeCalls")
    complete: PedigreeCalls = record_field("call", axes=(), units="1", dtype="PedigreeCalls")
    reach: TierReach = record_field("report", axes=(), units="1", dtype="TierReach")
    eligibility: EligibilityRecord = record_field("state", axes=(), units="1", dtype="EligibilityRecord")
    panel: CandidatePanel = record_field("state", axes=(), units="1", dtype="CandidatePanel")
    support: SupportRecord = record_field("call", axes=(), units="1", dtype="SupportRecord")
    sibling_guard: GuardRecord = record_field("call", axes=(), units="1", dtype="GuardRecord")
    views: PedigreeViews = record_field("report", dtype="PedigreeViews", kw_only=True)
    family_solve: FamilySolveRecord = record_field("state", dtype="FamilySolveRecord", kw_only=True)
    guard_summary: GuardSummary = record_field("report", dtype="GuardSummary", kw_only=True)
    exclusion: ExclusionSummary = record_field("state", dtype="ExclusionSummary", kw_only=True)
    retained: PedigreeRetained = record_field("state", dtype="PedigreeRetained", kw_only=True)
    reports_complete: bool = record_field("diagnostic", dtype="bool", kw_only=True)
    schema: int = schema_field(7)


@dataclass(frozen=True)
class PedigreeResult:
    """Released science plus explicit pre-guard state used by report projections."""

    pedigree: Pedigree = record_field("state", axes=(), units="1", dtype="Pedigree")
    decision: Decision = record_field("state", axes=(), units="1", dtype="Decision")
    count_release: CountRelease = record_field("state", axes=(), units="1", dtype="CountRelease")
    orientation_ranks: np.ndarray = record_field("state", axes=("N",), units="1", dtype="int64")
    exclusion_summary: ExclusionSummary = record_field("state", axes=(), units="1", dtype="ExclusionSummary")


POLICY = "every_sequenced_sample"


@dataclass(frozen=True)
class ParentEligibility:
    """The candidate universe over the ordered sample axis (never parentage truth).

    sample_ids: ordered sample IDs, identical to the run's sample axis (N).
    children:   bool [N]; samples whose observed parents are inferred (every sample).
    parents:    bool [N, N]; [child, parent] admissible M1 edges (every pair of distinct samples).
    """

    sample_ids: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    children: np.ndarray = record_field("state", axes=("N",), units="1", dtype="bool")
    parents: np.ndarray = record_field("state", axes=("N", "N"), units="1", dtype="bool")

    @classmethod
    def unrestricted(cls, sample_ids: Sequence[str]) -> ParentEligibility:
        """Every sample is a target and every other sample a candidate."""
        samples = tuple(str(value) for value in sample_ids)
        if len(samples) != len(set(samples)):
            raise InputError("sample IDs must be unique")
        n = len(samples)
        children = np.ones(n, dtype=np.bool_)
        parents = ~np.eye(n, dtype=np.bool_)
        children.setflags(write=False)
        parents.setflags(write=False)
        return cls(samples, children, parents)

    # ---------------------------------------------------------------- queries
    def pair_allowed(self, child: int, first: int, second: int) -> bool:
        """Exact M2 membership of one unordered pair."""
        return first != second and bool(self.parents[child, first] and self.parents[child, second])

    def pair_mask(self, child: int | np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Vectorised ``pair_allowed``."""
        return (first != second) & self.parents[child, first] & self.parents[child, second]

    def parent_counts(self) -> np.ndarray:
        """int64 [N]: eligible M1 candidates per sample."""
        return np.count_nonzero(self.parents, axis=1).astype(np.int64)

    def pair_counts(self) -> np.ndarray:
        """int64 [N]: size of each sample's unordered M2 universe (the identity-prior denominator)."""
        counts = self.parent_counts()
        return counts * (counts - 1) // 2

    def identity(self) -> str:
        """SHA-256 of the universe and its masks (a cache identity, not an integrity check)."""
        fields = {"sample_ids": list(self.sample_ids), "policy": POLICY}
        digest = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())
        for name, array in (("children", self.children), ("parents", self.parents)):
            digest.update(name.encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def summary(self) -> EligibilityRecord:
        """Small JSON-safe counts for the product and reports."""
        targets = np.flatnonzero(self.children)
        parent_counts = self.parent_counts()[targets]
        pair_counts = self.pair_counts()[targets]
        return EligibilityRecord(
            policy=POLICY,
            sample_count=len(self.sample_ids),
            eligible_child_count=int(len(targets)),
            candidate_parent_sample_count=int(np.count_nonzero(np.any(self.parents, axis=0))),
            minimum_parent_candidates_per_target=int(parent_counts.min()) if len(targets) else 0,
            maximum_parent_candidates_per_target=int(parent_counts.max(initial=0)),
            minimum_parent_pairs_per_target=int(pair_counts.min()) if len(targets) else 0,
            maximum_parent_pairs_per_target=int(pair_counts.max(initial=0)),
            individual_parentage_ground_truth=False,
            sample_metadata=False,
            parent_slot_basis=(
                "unordered: canonical order by observed sites per contig, then sample name; "
                "no sex is recorded or inferred"
            ),
        )


class SummaryMapping(Mapping[str, object]):
    """Named dataclass fields exposed to existing table renderers; absent optional fields stay absent."""

    # Dataclass subclasses supply the schema; the mixin is not itself a product record.
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __getitem__(self, key: str) -> object:
        if key not in {f.name for f in fields(self)} or getattr(self, key) is None:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return (f.name for f in fields(self) if getattr(self, f.name) is not None)

    def __len__(self) -> int:
        return sum(getattr(self, f.name) is not None for f in fields(self))


@dataclass(frozen=True)
class FamilySolveRecord(SummaryMapping):
    """The full solve and abstaining-resample diagnostics (PD10/PD16)."""

    iterations: int = record_field("state", axes=(), units="1", dtype="int")
    residual: float = record_field("state", axes=(), units="nats", dtype="float")
    tolerance: float = record_field("state", axes=(), units="1", dtype="float")
    maximum_iterations: int = record_field("state", axes=(), units="1", dtype="int")
    retry_iterations: int = record_field("state", axes=(), units="1", dtype="int")
    retry_limit: int = record_field("state", axes=(), units="1", dtype="int")
    retry_damping: float = record_field("state", axes=(), units="1", dtype="float")
    frozen_path_preparation_passes: int = record_field("state", axes=(), units="1", dtype="int")
    nonconverged_bootstrap_replicates: int = record_field("state", axes=(), units="1", dtype="int")
    nonconverged_loco_replicates: int = record_field("state", axes=(), units="1", dtype="int")


@dataclass(frozen=True)
class EligibilityRecord(SummaryMapping):
    """The unrestricted candidate universe, explicitly disavowing metadata or truth (PD01)."""

    policy: str = record_field("state", axes=(), units="1", dtype="str")
    sample_count: int = record_field("state", axes=(), units="1", dtype="int")
    eligible_child_count: int = record_field("state", axes=(), units="1", dtype="int")
    candidate_parent_sample_count: int = record_field("state", axes=(), units="1", dtype="int")
    minimum_parent_candidates_per_target: int = record_field("state", axes=(), units="1", dtype="int")
    maximum_parent_candidates_per_target: int = record_field("state", axes=(), units="1", dtype="int")
    minimum_parent_pairs_per_target: int = record_field("state", axes=(), units="1", dtype="int")
    maximum_parent_pairs_per_target: int = record_field("state", axes=(), units="1", dtype="int")
    individual_parentage_ground_truth: bool = record_field("state", axes=(), units="1", dtype="bool")
    sample_metadata: bool = record_field("state", axes=(), units="1", dtype="bool")
    parent_slot_basis: str = record_field("state", axes=(), units="1", dtype="str")


@dataclass(frozen=True)
class ExclusionSummary(SummaryMapping):
    """The count evidence's model and scoring coverage (PD22); no scoring implementation."""

    model: str = record_field("report", axes=(), units="1", dtype="str")
    alpha: float = record_field("report", axes=(), units="1", dtype="float")
    genotype_replacement_probability: float = record_field("report", axes=(), units="1", dtype="float")
    conditional_on_supported_edge_directions: bool = record_field("report", axes=(), units="1", dtype="bool")
    candidate_dyads: int = record_field("report", axes=(), units="1", dtype="int")
    contigs: int = record_field("obs", axes=(), units="1", dtype="int")
    status: str = record_field("report", axes=(), units="1", dtype="str")
    released: int = record_field("report", axes=(), units="1", dtype="int")
    chromosomes: list[dict[str, str | int]] | None = record_field(
        "report", axes=(), units="1", dtype="list[dict[str, str | int]] | None", default=None
    )


@dataclass(frozen=True)
class RelativeContrasts:
    """Report-only relative proxies; arrays own their storage, ordered by query pair Q (PD26)."""

    pairs: np.ndarray = record_field("report", axes=("Q", "2"), dtype="int64")
    views: tuple[tuple[str, ...], ...] = record_field("report", axes=("Q",), dtype="tuple[str]")
    second_degree: np.ndarray = record_field("report", axes=("Q",), units="nats", dtype="float64")
    full_sib: np.ndarray = record_field("report", axes=("Q",), units="nats", dtype="float64")
    usable_chromosomes: np.ndarray = record_field("report", axes=("Q",), dtype="int64")
    negative_second: np.ndarray = record_field("report", axes=("Q",), dtype="int64")
    negative_full: np.ndarray = record_field("report", axes=("Q",), dtype="int64")
    minimum_second: np.ndarray = record_field("report", axes=("Q",), units="nats", dtype="float64")


@dataclass(frozen=True)
class DecisionTrace(SummaryMapping):
    """Full-fit factors and the actual final reciprocal solve (PD05/PD08-PD10)."""

    genetic: np.ndarray = record_field("state", axes=("A",), units="nats", dtype="float64")
    structural_state: np.ndarray = record_field("state", axes=("A",), units="nats", dtype="float64")
    structural_identity: np.ndarray = record_field("state", axes=("A",), units="nats", dtype="float64")
    testable: np.ndarray = record_field("state", axes=("A",), dtype="bool")
    direction: np.ndarray = record_field("state", axes=("A",), units="nats", dtype="float64")
    ancestry_paths: np.ndarray = record_field("state", axes=("A",), units="nats", dtype="float64")
    reciprocal_family: np.ndarray = record_field("state", axes=("A",), units="nats", dtype="float64")
    orientation_log_odds: np.ndarray = record_field("state", axes=("N", "N"), units="nats", dtype="float64")
    final_family_iterations: int = record_field("state", dtype="int")
    final_family_retry_iterations: int = record_field("state", dtype="int")
    final_family_residual: float = record_field("state", units="nats", dtype="float64")


@dataclass(frozen=True)
class StructuralCompatibility:
    """PD25's numeric structural views on eligible-edge, panel-row and alternative axes.

    Ratios are NaN where their exposure denominator is zero. Compatible/testable
    refer to every alternative, while evaluable refers to each child. This work
    record makes the producer's report roles and units explicit before row views.
    """

    coverage: np.ndarray = record_field("report", axes=("E",), units="probability", dtype="float64")
    explainability: np.ndarray = record_field("report", axes=("T",), units="probability", dtype="float64")
    compatible: np.ndarray = record_field("report", axes=("A",), dtype="bool")
    testable: np.ndarray = record_field("report", axes=("A",), dtype="bool")
    evaluable: np.ndarray = record_field("report", axes=("N",), dtype="bool")

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.coverage, self.explainability, self.compatible, self.testable, self.evaluable))


@dataclass(frozen=True)
class ConfigurationSupport:
    """One selected support-set entry; formatting and sample names belong to OU2."""

    state: int = record_field("report", dtype="int8")
    parents: tuple[int, ...] = record_field("report", dtype="tuple[int, ...]")
    probability: float = record_field("report", dtype="float64")


@dataclass(frozen=True)
class RelationshipView:
    """Numeric PD25 row; sample/parent integers index the product sample axis (-1 missing)."""

    sample: int = record_field("report", units="1", dtype="int")
    parent1: int = record_field("report", units="1", dtype="int")
    parent2: int = record_field("report", units="1", dtype="int")
    parent_state: int = record_field("report", units="1", dtype="int")
    observed_parent_count: float = record_field("report", units="1", dtype="float64")
    inference_status: str = record_field("report", units="1", dtype="str")
    release_evidence: str | None = record_field("report", dtype="str | None", default=None)
    parent_slot_basis: str | None = record_field("report", dtype="str | None", default=None)
    parents_before_guard: tuple[int, int] | None = record_field("report", dtype="tuple[int, int] | None", default=None)

    relationship_basis: str = record_field("report", dtype="str", default="")


@dataclass(frozen=True)
class PartialRelationshipView:
    """Numeric PD25 row; sample/parent integers index the product sample axis (-1 missing)."""

    sample: int = record_field("report", units="1", dtype="int")
    parent1: int = record_field("report", units="1", dtype="int")
    parent2: int = record_field("report", units="1", dtype="int")
    parent_state: int = record_field("report", units="1", dtype="int")
    observed_parent_count: float = record_field("report", units="1", dtype="float64")
    inference_status: str = record_field("report", units="1", dtype="str")
    parent1_supported: bool = record_field("report", units="1", dtype="bool")
    parent2_supported: bool = record_field("report", units="1", dtype="bool")
    exact_configuration_resolved: bool = record_field("report", units="1", dtype="bool")
    observed_parent_count_minimum: int = record_field("report", units="1", dtype="int")
    observed_parent_count_maximum: int = record_field("report", units="1", dtype="int")
    release_evidence: str | None = record_field("report", dtype="str | None", default=None)
    parent_slot_basis: str | None = record_field("report", dtype="str | None", default=None)
    parents_before_guard: tuple[int, int] | None = record_field("report", dtype="tuple[int, int] | None", default=None)

    relationship_basis: str = record_field("report", dtype="str", default="")


@dataclass(frozen=True)
class CandidateView:
    """Numeric PD25 row; sample/parent integers index the product sample axis (-1 missing)."""

    sample: int = record_field("report", units="1", dtype="int")
    parent_state: int = record_field("report", units="1", dtype="int")
    observed_parent_count: float = record_field("report", units="1", dtype="float64")
    parent1: int = record_field("report", units="1", dtype="int")
    parent2: int = record_field("report", units="1", dtype="int")
    parent1_candidates: tuple[int, ...] = record_field("report", units="1", dtype="tuple[int, ...]")
    parent2_candidates: tuple[int, ...] = record_field("report", units="1", dtype="tuple[int, ...]")
    configuration_candidates: tuple[tuple[int, ...], ...] = record_field(
        "report", units="1", dtype="tuple[tuple[int, ...], ...]"
    )
    exact_configuration_resolved: bool = record_field("report", units="1", dtype="bool")
    inference_status: str = record_field("report", units="1", dtype="str")
    parent1_supported: bool = record_field("report", units="1", dtype="bool")
    parent2_supported: bool = record_field("report", units="1", dtype="bool")
    observed_parent_count_minimum: int = record_field("report", units="1", dtype="int")
    observed_parent_count_maximum: int = record_field("report", units="1", dtype="int")
    tier_b_final_state_call: bool = record_field("report", dtype="bool", default=False)
    tier_b_release_evidence: str = record_field("report", dtype="str", default="")
    parent_slot_basis: str | None = record_field("report", dtype="str | None", default=None)
    parents_before_guard: tuple[int, int] | None = record_field("report", dtype="tuple[int, int] | None", default=None)


@dataclass(frozen=True)
class DiagnosticView:
    """Numeric PD25 row; sample/parent integers index the product sample axis (-1 missing)."""

    sample: int = record_field("report", units="1", dtype="int")
    m0_prior_sensitivity_state_agreement: bool = record_field("report", units="1", dtype="bool")
    m0_prior_sensitivity_graph_agreement: bool = record_field("report", units="1", dtype="bool")
    m0_prior_sensitivity_tier_veto: bool = record_field("report", units="1", dtype="bool")
    structure_child_evaluable: bool = record_field("report", units="1", dtype="bool")
    structure_selected_row_exposure_testable: bool = record_field("report", units="1", dtype="bool")
    structure_selected_row_cx_compatible: bool = record_field("report", units="1", dtype="bool")
    structure_parent1_coverage: float = record_field("report", units="1", dtype="float64")
    structure_parent2_coverage: float = record_field("report", units="1", dtype="float64")
    structure_pair_explainability: float = record_field("report", units="1", dtype="float64")
    structure_parent1_direction_probability: float = record_field("report", units="1", dtype="float64")
    structure_parent2_direction_probability: float = record_field("report", units="1", dtype="float64")
    best_m1_parent: int = record_field("report", units="1", dtype="int")
    best_m1_conditional_identity_margin: float = record_field("report", units="nats", dtype="float64")
    best_m2_parent1: int = record_field("report", units="1", dtype="int")
    best_m2_parent2: int = record_field("report", units="1", dtype="int")
    best_m2_conditional_identity_margin: float = record_field("report", units="nats", dtype="float64")
    local_parent1: int = record_field("report", units="1", dtype="int")
    local_parent2: int = record_field("report", units="1", dtype="int")
    eligible_child: bool = record_field("report", units="1", dtype="bool")
    eligible_parent_count: int = record_field("report", units="1", dtype="int")
    eligible_parent_pair_count: int = record_field("report", units="1", dtype="int")
    complete_parent1: int = record_field("report", units="1", dtype="int")
    complete_parent2: int = record_field("report", units="1", dtype="int")
    selected_parent_state: int = record_field("report", units="1", dtype="int")
    local_winner_parent_state: int = record_field("report", units="1", dtype="int")
    observed_parent_count: float = record_field("report", units="1", dtype="float64")
    local_observed_parent_count: float = record_field("report", units="1", dtype="float64")
    identifiable: bool = record_field("report", units="1", dtype="bool")
    dag_displaced: bool = record_field("report", units="1", dtype="bool")
    graph_conflict: bool = record_field("report", units="1", dtype="bool")
    graph_tie_conflict: bool = record_field("report", units="1", dtype="bool")
    graph_direction_resolved_alternative: bool = record_field("report", units="1", dtype="bool")
    local_parent_role_probability: float = record_field("report", units="1", dtype="float64")
    selected_parent_role_probability: float = record_field("report", units="1", dtype="float64")
    raw_ancestry_junction_count: float = record_field("report", units="1", dtype="float64")
    callable_ancestry_haplotype_bin_count: float = record_field("report", units="1", dtype="float64")
    adjusted_ancestry_junction_burden: float = record_field("report", units="1", dtype="float64")
    ancestry_painting_callability_fraction: float = record_field("report", units="1", dtype="float64")
    informative_contig_count: int = record_field("report", units="1", dtype="int")
    informative_map_units: float = record_field("report", units="1", dtype="float64")
    local_state_bootstrap_fraction: float = record_field("report", units="1", dtype="float64")
    local_configuration_bootstrap_fraction: float = record_field("report", units="1", dtype="float64")
    graph_configuration_bootstrap_fraction: float = record_field("report", units="1", dtype="float64")
    selected_graph_configuration_bootstrap_fraction: float = record_field("report", units="1", dtype="float64")
    selected_graph_parent1_bootstrap_fraction: float = record_field("report", units="1", dtype="float64")
    selected_graph_parent2_bootstrap_fraction: float = record_field("report", units="1", dtype="float64")
    parent1_bootstrap_fraction: float = record_field("report", units="1", dtype="float64")
    parent2_bootstrap_fraction: float = record_field("report", units="1", dtype="float64")
    local_state_loco_fraction: float = record_field("report", units="1", dtype="float64")
    local_configuration_loco_fraction: float = record_field("report", units="1", dtype="float64")
    graph_configuration_loco_fraction: float = record_field("report", units="1", dtype="float64")
    selected_graph_configuration_loco_fraction: float = record_field("report", units="1", dtype="float64")
    selected_graph_parent1_loco_fraction: float = record_field("report", units="1", dtype="float64")
    selected_graph_parent2_loco_fraction: float = record_field("report", units="1", dtype="float64")
    parent1_loco_fraction: float = record_field("report", units="1", dtype="float64")
    parent2_loco_fraction: float = record_field("report", units="1", dtype="float64")
    state_winner_margin: float = record_field("report", units="nats", dtype="float64")
    conditional_identity_margin: float = record_field("report", units="nats", dtype="float64")
    unconstrained_winner_margin: float = record_field("report", units="nats", dtype="float64")
    selected_aggregate_utility: float = record_field("report", units="nats", dtype="float64")
    selected_minus_unconstrained_best: float = record_field("report", units="1", dtype="float64")
    state_log_evidence0: float = record_field("report", units="nats", dtype="float64")
    state_log_evidence1: float = record_field("report", units="nats", dtype="float64")
    state_log_evidence2: float = record_field("report", units="nats", dtype="float64")
    state_support0: float = record_field("report", units="1", dtype="float64")
    state_support1: float = record_field("report", units="1", dtype="float64")
    state_support2: float = record_field("report", units="1", dtype="float64")
    scored_candidate_count2: int = record_field("report", units="1", dtype="int")
    m2_candidate_screen_incomplete: bool = record_field("report", units="1", dtype="bool")
    prior_sensitivity_local_state_agreement_fraction: float = record_field("report", units="1", dtype="float64")
    prior_sensitivity_local_identity_agreement_fraction: float = record_field("report", units="1", dtype="float64")
    prior_sensitivity_graph_agreement_fraction: float = record_field("report", units="1", dtype="float64")
    prior_sensitivity_selected_state_minimum_support: float = record_field("report", units="1", dtype="float64")
    inference_status: str = record_field("report", units="1", dtype="str")
    tier_a_state_call: bool = record_field("report", units="1", dtype="bool")
    tier_b_state_call: bool = record_field("report", units="1", dtype="bool")
    tier_a_exact_configuration: bool = record_field("report", units="1", dtype="bool")
    tier_b_exact_configuration: bool = record_field("report", units="1", dtype="bool")
    tier_a_exact_pair: bool = record_field("report", units="1", dtype="bool")
    tier_b_exact_pair: bool = record_field("report", units="1", dtype="bool")
    tier_a_parent1: bool = record_field("report", units="1", dtype="bool")
    tier_a_parent2: bool = record_field("report", units="1", dtype="bool")
    tier_b_parent1: bool = record_field("report", units="1", dtype="bool")
    tier_b_parent2: bool = record_field("report", units="1", dtype="bool")
    configuration_support_set: tuple[ConfigurationSupport, ...] = record_field(
        "report", units="1", dtype="tuple[ConfigurationSupport, ...]"
    )
    parent1_candidate_set: tuple[int, ...] = record_field("report", units="1", dtype="tuple[int, ...]")
    parent2_candidate_set: tuple[int, ...] = record_field("report", units="1", dtype="tuple[int, ...]")
    evidence_configuration_set: tuple[tuple[int, ...], ...] = record_field(
        "report", units="1", dtype="tuple[tuple[int, ...], ...]"
    )
    tier_b_final_state_call: bool = record_field("report", dtype="bool", default=False)
    tier_b_release_evidence: str = record_field("report", dtype="str", default="")


@dataclass(frozen=True)
class ParentStateView:
    """Numeric PD25 row; sample/parent integers index the product sample axis (-1 missing)."""

    sample: int = record_field("report", units="1", dtype="int")
    selected_parent_state: int = record_field("report", units="1", dtype="int")
    local_winner_parent_state: int = record_field("report", units="1", dtype="int")
    observed_parent_count: float = record_field("report", units="1", dtype="float64")
    local_observed_parent_count: float = record_field("report", units="1", dtype="float64")
    inference_status: str = record_field("report", units="1", dtype="str")
    dag_displaced: bool = record_field("report", units="1", dtype="bool")
    graph_conflict: bool = record_field("report", units="1", dtype="bool")
    graph_tie_conflict: bool = record_field("report", units="1", dtype="bool")
    graph_direction_resolved_alternative: bool = record_field("report", units="1", dtype="bool")
    selected_parent_role_probability: float = record_field("report", units="1", dtype="float64")
    tier_a_state_call: bool = record_field("report", units="1", dtype="bool")
    tier_b_state_call: bool = record_field("report", units="1", dtype="bool")
    tier_b_final_state_call: bool = record_field("report", units="1", dtype="bool")
    tier_b_release_evidence: str = record_field("report", units="1", dtype="str")


@dataclass(frozen=True)
class CallExplanationView:
    """Numeric PD25 row; sample/parent integers index the product sample axis (-1 missing)."""

    sample: int = record_field("report", units="1", dtype="int")
    parent_state: int = record_field("report", units="1", dtype="int")
    observed_parent_count: float = record_field("report", units="1", dtype="float64")
    parent1: int = record_field("report", units="1", dtype="int")
    parent2: int = record_field("report", units="1", dtype="int")
    parent1_candidates: tuple[int, ...] = record_field("report", units="1", dtype="tuple[int, ...]")
    parent2_candidates: tuple[int, ...] = record_field("report", units="1", dtype="tuple[int, ...]")
    configuration_candidates: tuple[tuple[int, ...], ...] = record_field(
        "report", units="1", dtype="tuple[tuple[int, ...], ...]"
    )
    exact_configuration_resolved: bool = record_field("report", units="1", dtype="bool")
    inference_status: str = record_field("report", units="1", dtype="str")
    parent1_supported: bool = record_field("report", units="1", dtype="bool")
    parent2_supported: bool = record_field("report", units="1", dtype="bool")
    observed_parent_count_minimum: int = record_field("report", units="1", dtype="int")
    observed_parent_count_maximum: int = record_field("report", units="1", dtype="int")
    local_winner_parent_state: int = record_field("report", units="1", dtype="int")
    selected_parent_state: int = record_field("report", units="1", dtype="int")
    local_parent1: int = record_field("report", units="1", dtype="int")
    local_parent2: int = record_field("report", units="1", dtype="int")
    graph_conflict: bool = record_field("report", units="1", dtype="bool")
    state_winner_margin: float = record_field("report", units="nats", dtype="float64")
    conditional_identity_margin: float = record_field("report", units="nats", dtype="float64")
    local_state_bootstrap_fraction: float = record_field("report", units="1", dtype="float64")
    local_state_loco_fraction: float = record_field("report", units="1", dtype="float64")
    eligible_m2_pairs: int = record_field("report", units="1", dtype="int")
    scored_m2_pairs: int = record_field("report", units="1", dtype="int")
    incomplete_m2_search: bool = record_field("report", units="1", dtype="bool")
    graph_changed_local_configuration: bool = record_field("report", units="1", dtype="bool")
    interpretation: str = record_field("report", units="1", dtype="str")
    tier_b_final_state_call: bool = record_field("report", dtype="bool", default=False)
    tier_b_release_evidence: str = record_field("report", dtype="str", default="")


@dataclass(frozen=True)
class AlternativeExplanationView:
    """Numeric PD25 row; sample/parent integers index the product sample axis (-1 missing)."""

    sample: int = record_field("report", units="1", dtype="int")
    parent_state: int = record_field("report", units="1", dtype="int")
    parent1: int = record_field("report", units="1", dtype="int")
    parent2: int = record_field("report", units="1", dtype="int")
    local_winner: bool = record_field("report", units="1", dtype="bool")
    graph_winner: bool = record_field("report", units="1", dtype="bool")
    within_state_rank: int | None = record_field("report", units="1", dtype="int | None")
    genetic_log_evidence: float = record_field("report", units="nats", dtype="float64")
    structural_state_log_evidence: float = record_field("report", units="nats", dtype="float64")
    structural_identity_eligible: bool = record_field("report", units="1", dtype="bool")
    direction_log_contribution: float = record_field("report", units="nats", dtype="float64")
    reciprocal_family_log_contribution: float = record_field("report", units="nats", dtype="float64")
    ancestry_path_log_contribution: float = record_field("report", units="nats", dtype="float64")
    adjusted_state_log_evidence: float = record_field("report", units="nats", dtype="float64")
    adjusted_identity_log_evidence: float = record_field("report", units="nats", dtype="float64")
    parent_state_log_prior: float = record_field("report", units="nats", dtype="float64")
    log_eligible_identity_multiplicity: float = record_field("report", units="nats", dtype="float64")
    marginal_state_log_evidence: float = record_field("report", units="nats", dtype="float64")
    graph_decision_utility: float = record_field("report", units="nats", dtype="float64")


@dataclass(frozen=True)
class SearchDiagnosticView:
    """Numeric PD25 row; sample/parent integers index the product sample axis (-1 missing)."""

    sample: int = record_field("report", units="1", dtype="int")
    eligible_parents: int = record_field("report", units="1", dtype="int")
    scored_m2_pairs: int = record_field("report", units="1", dtype="int")
    m2_panel_parents: int = record_field("report", units="1", dtype="int")
    screen_boundary_rank: int = record_field("report", units="1", dtype="int")
    screen_boundary_gap: float = record_field("report", units="nats", dtype="float64")
    omitted_parent_count: int = record_field("report", units="1", dtype="int")
    strongest_omitted_parent: int = record_field("report", units="1", dtype="int")
    parent1_screen_rank: float = record_field("report", units="1", dtype="float64")
    parent2_screen_rank: float = record_field("report", units="1", dtype="float64")
    selected_parent_at_screen_boundary: bool = record_field("report", units="1", dtype="bool")
    m2_panel_capped: bool = record_field("report", units="1", dtype="bool", default=False)


@dataclass(frozen=True)
class EvidenceSummaryView:
    """Numeric PD25 row; sample/parent integers index the product sample axis (-1 missing)."""

    contig: str = record_field("report", units="1", dtype="str")
    informative_markers: int = record_field("report", units="1", dtype="int")
    mean_minimum_ancestry_junctions: float = record_field("report", units="painting bins", dtype="float64")
    mean_callable_haplotype_bins: float = record_field("report", units="1", dtype="float64")


@dataclass(frozen=True)
class ChromosomeSummaryView:
    """Numeric PD25 row; sample/parent integers index the product sample axis (-1 missing)."""

    contig: str = record_field("report", units="1", dtype="str")
    component_count: int = record_field("report", units="1", dtype="int")
    scored_component_count: int = record_field("report", units="1", dtype="int")
    informative_markers: int = record_field("report", units="1", dtype="int")
    omitted_reason: str | None = record_field("report", units="1", dtype="str | None")


@dataclass(frozen=True)
class ExclusionEvidenceView:
    """Numeric PD25 exclusion evidence, one entry per screened dyad D in construction order; arrays own their storage.

    sample / candidate_parent index the product sample axis. With exclusion intervals (PedigreeViews.
    exclusion_intervals), log_e is the exact value where the interval is a point and NaN otherwise, and
    log_e_lower / log_e_upper / log_e_exact hold the interval; without them those three are None and log_e is exact.
    One record per view rather than one object per dyad: N = 2,500 screens about 4 million dyads.
    """

    sample: np.ndarray = record_field("report", axes=("D",), units="1", dtype="int64")
    candidate_parent: np.ndarray = record_field("report", axes=("D",), units="1", dtype="int64")
    log_e: np.ndarray = record_field("report", axes=("D",), units="nats", dtype="float64")
    observed_sites: np.ndarray = record_field("report", axes=("D",), units="1", dtype="int64")
    log_e_lower: np.ndarray | None = record_field("report", axes=("D",), units="nats", dtype="float64", default=None)
    log_e_upper: np.ndarray | None = record_field("report", axes=("D",), units="nats", dtype="float64", default=None)
    log_e_exact: np.ndarray | None = record_field("report", axes=("D",), dtype="bool", default=None)

    def __len__(self) -> int:
        return len(self.sample)


@dataclass(frozen=True)
class PedigreeViews:
    """All 17 PD25 views before rendering; no DataFrames, names or formatted support sets.

    Rows preserve baseline construction order. Relative/guard inputs are absent only
    before their explicit composition step, never repaired by the product reader.
    """

    tier_b_relationships: tuple[RelationshipView, ...] = record_field(
        "report", axes=(), dtype="tuple[RelationshipView, ...]"
    )
    tier_b_partial_relationships: tuple[PartialRelationshipView, ...] = record_field(
        "report", axes=(), dtype="tuple[PartialRelationshipView, ...]"
    )
    tier_a_relationships: tuple[RelationshipView, ...] = record_field(
        "report", axes=(), dtype="tuple[RelationshipView, ...]"
    )
    tier_a_partial_relationships: tuple[PartialRelationshipView, ...] = record_field(
        "report", axes=(), dtype="tuple[PartialRelationshipView, ...]"
    )
    complete_relationships: tuple[RelationshipView, ...] = record_field(
        "report", axes=(), dtype="tuple[RelationshipView, ...]"
    )
    tier_b_candidate_sets: tuple[CandidateView, ...] = record_field(
        "report", axes=(), dtype="tuple[CandidateView, ...]"
    )
    parent_state_calls: tuple[ParentStateView, ...] = record_field(
        "report", axes=(), dtype="tuple[ParentStateView, ...]"
    )
    diagnostics: tuple[DiagnosticView, ...] = record_field("report", axes=(), dtype="tuple[DiagnosticView, ...]")
    call_explanations: tuple[CallExplanationView, ...] = record_field(
        "report", axes=(), dtype="tuple[CallExplanationView, ...]"
    )
    alternative_explanations: tuple[AlternativeExplanationView, ...] = record_field(
        "report", axes=(), dtype="tuple[AlternativeExplanationView, ...]"
    )
    search_diagnostics: tuple[SearchDiagnosticView, ...] = record_field(
        "report", axes=(), dtype="tuple[SearchDiagnosticView, ...]"
    )
    exclusion_evidence: ExclusionEvidenceView = record_field("report", axes=(), dtype="ExclusionEvidenceView")
    exclusion_diagnostics: tuple[CountDiagnostic, ...] = record_field(
        "report", axes=(), dtype="tuple[CountDiagnostic, ...]"
    )
    evidence_summary: tuple[EvidenceSummaryView, ...] = record_field(
        "report", axes=(), dtype="tuple[EvidenceSummaryView, ...]"
    )
    chromosome_summary: tuple[ChromosomeSummaryView, ...] = record_field(
        "report", axes=(), dtype="tuple[ChromosomeSummaryView, ...]"
    )
    relative_contrasts: RelativeContrasts | None = record_field("report", axes=(), dtype="RelativeContrasts | None")
    sibling_guard: GuardView | None = record_field("report", dtype="GuardView | None")
    exclusion_intervals: bool = record_field("report", dtype="bool", default=False)


@dataclass(frozen=True)
class GuardViewRow:
    """Final PD23 relationship evidence, in original edge order."""

    parent: int = record_field("report", dtype="int64")
    child: int = record_field("report", dtype="int64")
    witnesses: int = record_field("report", dtype="int64")
    set_aside: int = record_field("report", dtype="int64")
    against: int = record_field("report", dtype="int64")
    supporting: int = record_field("report", dtype="int64")
    pair_score: float = record_field("report", units="nats", dtype="float64")
    family_score: float = record_field("report", units="nats", dtype="float64")
    score: float = record_field("report", units="nats", dtype="float64")
    call: str = record_field("report", dtype="str")
    evidence_call: str = record_field("report", dtype="str")
    withdrawal_iteration: int = record_field("report", dtype="int64")
    withholding_basis: str = record_field("report", dtype="str")
    relationship_basis: str = record_field("report", dtype="str")


@dataclass(frozen=True)
class GuardView:
    """Final guard rows and the convergence receipt; no reconstruction of unreported membership."""

    rows: tuple[GuardViewRow, ...] = record_field("report", dtype="tuple[GuardViewRow, ...]")
    cleaning_iterations: int = record_field("diagnostic", dtype="int64")
    cleaning_converged: bool = record_field("diagnostic", dtype="bool")
    cleaning_iteration_bound: int = record_field("diagnostic", dtype="int64")


@dataclass(frozen=True)
class PedigreeRetained:
    """Lossless state of the composed decision; borrowed arrays share the codec's array objects.

    Retains every PedigreeResult field not already published directly on Pedigree.
    These fields are never recomputed to repair a resumed product.
    """

    decision: Decision = record_field("state", dtype="Decision")
    count_release: CountRelease = record_field("state", dtype="CountRelease")
    orientation_ranks: np.ndarray = record_field("state", axes=("N",), dtype="int64")
    exclusion_request: ExclusionRequest = record_field("state", dtype="ExclusionRequest")


@dataclass(frozen=True)
class PedigreeReportInputs:
    """Report results supplied after count release and before guard application."""

    search: tuple[SearchDiagnosticView, ...] = record_field("report", dtype="tuple[SearchDiagnosticView, ...]")
    chromosomes: tuple[ChromosomeSummaryView, ...] = record_field("report", dtype="tuple[ChromosomeSummaryView, ...]")
    relatives: RelativeContrasts = record_field("report", dtype="RelativeContrasts")


@dataclass(frozen=True)
class RelationshipRequest:
    """All unordered dyads/trios any guard iteration can need, frozen once."""

    sample_ids: tuple[str, ...] = record_field("obs", dtype="tuple[str, ...]")
    contigs: tuple[str, ...] = record_field("obs", dtype="tuple[str, ...]")
    dyads: np.ndarray = record_field("state", axes=("D", "2"), dtype="int64")
    trios: np.ndarray = record_field("state", axes=("T", "3"), dtype="int64")

    def __post_init__(self) -> None:
        for rows, width in ((self.dyads, 2), (self.trios, 3)):
            if rows.ndim != 2 or rows.shape[1] != width or rows.dtype != np.int64:
                raise InputError("relationship request has invalid sequence axes")
            if len(rows) and (
                np.any(rows < 0) or np.any(rows >= len(self.sample_ids)) or np.any(np.diff(rows, axis=1) <= 0)
            ):
                raise InputError("relationship request requires distinct sorted sample indices")
            if len(set(map(tuple, rows.tolist()))) != len(rows):
                raise InputError("relationship request contains duplicate sequences")
        freeze_record_arrays(self)


@dataclass(frozen=True)
class RelationshipChance:
    """Cohort-derived chance laws and sampled triples for one independent component."""

    component_index: int = record_field("obs", dtype="int64")
    pair_chance: np.ndarray = record_field("evidence", axes=("B",), units="probability", dtype="float64")
    pair_persistence: float = record_field("evidence", units="probability", dtype="float64")
    trio_chance: np.ndarray = record_field("evidence", axes=("B",), units="probability", dtype="float64")
    trio_persistence: float = record_field("evidence", units="probability", dtype="float64")
    reference_triples: np.ndarray = record_field("state", axes=("K", "3"), dtype="int64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class RelationshipEvidence:
    """Four process log likelihoods summed over this contig's independent components."""

    contig: str = record_field("obs", dtype="str")
    sample_ids: tuple[str, ...] = record_field("obs", dtype="tuple[str, ...]")
    dyads: np.ndarray = record_field("state", axes=("D", "2"), dtype="int64")
    trios: np.ndarray = record_field("state", axes=("T", "3"), dtype="int64")
    pair_log_likelihoods: np.ndarray = record_field("evidence", axes=("D", "4"), units="nats", dtype="float64")
    trio_log_likelihoods: np.ndarray = record_field("evidence", axes=("T", "4"), units="nats", dtype="float64")
    components: tuple[RelationshipChance, ...] = record_field("evidence", dtype="tuple[RelationshipChance, ...]")

    def __post_init__(self) -> None:
        if self.pair_log_likelihoods.shape != (len(self.dyads), 4) or self.trio_log_likelihoods.shape != (
            len(self.trios),
            4,
        ):
            raise InputError("relationship likelihood axes differ from request")
        if np.any(~np.isfinite(self.pair_log_likelihoods[:, 0])) or np.any(
            ~np.isfinite(self.trio_log_likelihoods[:, 0])
        ):
            raise InputError("feasible relationship likelihoods must be finite")
        if np.any(np.isnan(self.pair_log_likelihoods)) or np.any(np.isnan(self.trio_log_likelihoods)):
            raise InputError("relationship likelihoods contain NaN")
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ExclusionRequest:
    """Count-only exclusion query and the independently scored relationship request."""

    sample_ids: tuple[str, ...] = record_field("obs", dtype="tuple[str, ...]")
    contigs: tuple[str, ...] = record_field("obs", dtype="tuple[str, ...]")
    dyads: np.ndarray = record_field("state", axes=("D", "2"), dtype="int64")
    target_count: int = record_field("state", dtype="int64")
    before_edges: ParentEdges = record_field("state", dtype="ParentEdges")
    relationship: RelationshipRequest = record_field("state", dtype="RelationshipRequest")

    @property
    def dyad_digest(self) -> str:
        """Bind the ordered dyads and sample axis; distinct from the product derivation."""
        digest = hashlib.sha256(json.dumps(list(self.sample_ids)).encode())
        digest.update(np.ascontiguousarray(self.dyads, dtype=np.int64).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class PedigreePrecheck:
    """The configured/informative chromosome requirements; warnings do not alter science."""

    samples: int = record_field("obs", dtype="int64")
    contigs: int = record_field("obs", dtype="int64")
    minimum_scored_contigs: int = record_field("state", dtype="int64")
    tier_b_minimum_contigs: int = record_field("state", dtype="int64")
    tier_b_warning: bool = record_field("diagnostic", dtype="bool")
    basis: str = record_field("diagnostic", dtype="str")


@dataclass(frozen=True)
class ReleasedDecision:
    """Pure post-count/slot/guard state awaiting report inputs and publication."""

    decision: Decision = record_field("state", dtype="Decision")
    request: ExclusionRequest = record_field("state", dtype="ExclusionRequest")
    count_release: CountRelease = record_field("state", dtype="CountRelease")
    orientation_ranks: np.ndarray = record_field("state", axes=("N",), dtype="int64")
    pre_guard_calls: PedigreeCalls = record_field("state", dtype="PedigreeCalls")
    calls: PedigreeCalls = record_field("call", dtype="PedigreeCalls")
    tier_a: PedigreeCalls = record_field("call", dtype="PedigreeCalls")
    edges: ParentEdges = record_field("call", dtype="ParentEdges")
    complete: PedigreeCalls = record_field("call", dtype="PedigreeCalls")
    guard: GuardRecord = record_field("call", dtype="GuardRecord")
    exclusion: ExclusionSummary = record_field("state", dtype="ExclusionSummary")


# Pedigree report and publication boundary.


@dataclass(frozen=True)
class HammingTransition:
    """Painting's normalised ordered-diplotype transition at every bin boundary of a component.

    same, one_change, two_changes: float64 [B-1], the probability of moving from an ordered diplotype
    to one specific diplotype that keeps both founders, changes exactly one, or changes both.
    Each row of the S^2 x S^2 matrix sums to one: same + 2 (S-1) one_change + (S-1)^2 two_changes."""

    same: np.ndarray = record_field("state", axes=("B-1",), units="1", dtype="float64")
    one_change: np.ndarray = record_field("state", axes=("B-1",), units="1", dtype="float64")
    two_changes: np.ndarray = record_field("state", axes=("B-1",), units="1", dtype="float64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SourceFactors:
    """Exact source-posterior chain of every sample as a candidate parent (pedigree's reduced source model).

    initial:      float64 [N, S, S] normalised ordered-diplotype law of bin 0 (homologue-symmetrised).
    right_weight: float64 [N, B-1, S, S] backward factors r_b(v), max-normalised per boundary, so that
                  K_b(u, v) = A_b(u, v) r_b(v) / sum_w A_b(u, w) r_b(w) is the posterior chain kernel.
    available:    bool [N] samples with at least one informative painting site; others keep a uniform
                  initial law and unit right weights."""

    initial: np.ndarray = record_field("state", axes=("N", "S", "S"), units="1", dtype="float64")
    right_weight: np.ndarray = record_field("state", axes=("N", "B-1", "S", "S"), units="1", dtype="float64")
    available: np.ndarray = record_field("state", axes=("N",), units="1", dtype="bool")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PreparedComponent:
    """One painting component prepared for transmission scoring.

    component_index:   index of the painting component on its contig.
    named_alleles:     int8 [S-1, M] founder trajectory classes at the selected markers, -1 = no call.
    bin_sizes:         int64 [B] selected markers per painting bin (>= 1), markers bin-major.
    anchored_states:   bool [S] classes called somewhere in the component (BACKGROUND last, False).
    transition:        the painting's source transition (HammingTransition).
    factors:           exact source-posterior factors (SourceFactors).
    robustness_epsilon: painting emission robustness weight (uniform genotype mixture).
    selector_switch_probability: float64 [B-1] meiotic switch probability per boundary (pedigree map).
    measured_morgans:  float64 [B-1] measured map length (Morgans) between adjacent bin centres
                       (PreparedComponent.measured_coordinates): 0 where no supplied map or stated rate
                       measures it.
    genotype_likelihoods: float64 [N, M, 3] normalised raw GLs at the selected markers.
    observed:          bool [N, M] read depth > 0 at the selected markers.
    state_classes:     int16 [N, B, 2] painting MAP state class per sample, bin and track; -1 = unknown
                       (abstention, BACKGROUND or unanchored trajectory). Junction-count input.
    information_exponent: float64 [N, B] tempered-view emission exponent of each child and bin."""

    component_index: int = record_field("state", axes=(), units="1", dtype="int64")
    named_alleles: np.ndarray = record_field("state", axes=("S-1", "M"), units="1", dtype="int8")
    bin_sizes: np.ndarray = record_field("state", axes=("B",), units="1", dtype="int64")
    anchored_states: np.ndarray = record_field("state", axes=("S",), units="1", dtype="bool")
    transition: HammingTransition = record_field("state", axes=(), units="1", dtype="HammingTransition")
    factors: SourceFactors = record_field("state", axes=(), units="1", dtype="SourceFactors")
    robustness_epsilon: float = record_field("state", axes=(), units="1", dtype="float64")
    selector_switch_probability: np.ndarray = record_field("state", axes=("B-1",), units="1", dtype="float64")
    measured_morgans: np.ndarray = record_field("state", axes=("B-1",), units="Morgans", dtype="float64")
    genotype_likelihoods: np.ndarray = record_field("state", axes=("N", "M", "3"), units="probability", dtype="float64")
    observed: np.ndarray = record_field("state", axes=("N", "M"), units="1", dtype="bool")
    state_classes: np.ndarray = record_field("state", axes=("N", "B", "2"), units="1", dtype="int16")
    information_exponent: np.ndarray = record_field("state", axes=("N", "B"), units="1", dtype="float64")

    @property
    def n_states(self) -> int:
        return int(self.anchored_states.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.bin_sizes.shape[0])

    @property
    def informative_markers(self) -> int:
        return int(self.named_alleles.shape[1])

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PreparedChromosome:
    """Prepared components of one contig (record ``pedigree/prepared``).

    component_count: painting components on the contig; ``components`` keeps only components with
    observed non-uniform evidence at a selected marker, in painting order. ``omitted_reason`` is
    "no_observed_nonuniform_component_evidence" when none remains, else None."""

    contig: str = record_field("state", axes=(), units="1", dtype="str")
    component_count: int = record_field("state", axes=(), units="1", dtype="int64")
    components: tuple[PreparedComponent, ...] = record_field(
        "state", axes=("components",), units="1", dtype="tuple[PreparedComponent, ...]"
    )
    omitted_reason: str | None = record_field("state", axes=(), units="1", dtype="str | None", default=None)

    @property
    def informative_markers(self) -> int:
        return sum(component.informative_markers for component in self.components)

    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ChromosomeScores:
    """Composite evidence of one informative contig, summed over its components (``pedigree/scores``).

    zero [N], one [N, N], two [T]: M0 / M1 / M2 log likelihoods (panel row order); -inf where ineligible.
    edge_matched, edge_exposed: float64 [N, N] symmetric posterior-expected painting bins in which the
        pair shares an anchored founder / is testable for one (diagnostic / gate).
    pair_explained, pair_exposed: float64 [T] the same for a child and both panel parents.
    total_bins:    painting bins of the scored components.
    junctions:     int64 [N] minimum founder switches of each painting (phase-invariant).
    callable_bins: int64 [N] observed haplotype-bin labels behind the junction count.
    informative_markers: selected markers of the scored components.
    component_count, scored_component_count: painting components / prepared components.
    junction_dyads: int64 [D, 2] (child c, parent p) dyads the decision orients: both directions of every
        eligible edge, in row-major order (the same for every chromosome of a run).
    transmission_log_likelihood: float64 [D] joint log likelihood (nats) of the two samples' scheduled
        homologues under c <- p (science.pedigree.transmission.log_likelihood).
    scored_morgans: measured map length (Morgans) between adjacent bins of the scored components
        (PreparedComponent.measured_morgans; each interval at most ExposureParameters.unit_morgans;
        0 without a measured map).
    working_morgans: configured working-map distance, sum of -0.5 ln(1 - 2 theta) over the scored
        components' selector boundaries (PS04). This is NOT measured exposure and never enters a gate or
        decision; it is retained only for reports.
    edge_exposed_morgans: float64 [N, N] symmetric, pair_exposed_morgans: float64 [T]: the measured map length
        over which the pair (child and both panel parents) is exposed: each measured interval between adjacent
        bins counts as far as it is exposed at both ends."""

    contig: str = record_field("state", axes=(), units="1", dtype="str")
    zero: np.ndarray = record_field("state", axes=("N",), units="nats", dtype="float64")
    one_floored: np.ndarray = record_field("state", axes=("N", "N"), units="nats", dtype="float64")
    two: np.ndarray = record_field("state", axes=("T",), units="nats", dtype="float64")
    edge_matched: np.ndarray = record_field("state", axes=("N", "N"), units="painting bins", dtype="float64")
    edge_exposed: np.ndarray = record_field("state", axes=("N", "N"), units="painting bins", dtype="float64")
    pair_explained: np.ndarray = record_field("state", axes=("T",), units="painting bins", dtype="float64")
    pair_exposed: np.ndarray = record_field("state", axes=("T",), units="painting bins", dtype="float64")
    total_bins: float = record_field("state", axes=(), units="painting bins", dtype="float64")
    junctions: np.ndarray = record_field("state", axes=("N",), units="painting bins", dtype="int64")
    callable_bins: np.ndarray = record_field("state", axes=("N",), units="painting bins", dtype="int64")
    informative_markers: int = record_field("state", axes=(), units="1", dtype="int64")
    component_count: int = record_field("state", axes=(), units="1", dtype="int64")
    scored_component_count: int = record_field("state", axes=(), units="1", dtype="int64")
    junction_dyads: np.ndarray = record_field("state", axes=("D", "2"), units="1", dtype="int64")
    transmission_log_likelihood: np.ndarray = record_field("state", axes=("D",), units="nats", dtype="float64")
    scored_morgans: float = record_field("state", axes=(), units="Morgans", dtype="float64")
    edge_exposed_morgans: np.ndarray = record_field("state", axes=("N", "N"), units="Morgans", dtype="float64")
    pair_exposed_morgans: np.ndarray = record_field("state", axes=("T",), units="Morgans", dtype="float64")
    working_morgans: float = record_field("report", axes=(), units="Morgans", dtype="float64", kw_only=True)
    schema: int = schema_field(3)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class CompactM1:
    """Fixed genome-profile retained scores and the counted default floor.

    Rows are child-major; parents ascend within child. values/ chromosome_reference
    only gather the already-floored evidence and never apply the genome profile twice.
    """

    indptr: np.ndarray = record_field("state", axes=("N+1",), dtype="int64")
    parents: np.ndarray = record_field("state", axes=("R",), dtype="int64")
    raw: np.ndarray = record_field("state", axes=("C", "R"), units="nats", dtype="float64")
    best: np.ndarray = record_field("state", axes=("N",), units="nats", dtype="float64")
    cutoff: np.ndarray = record_field("state", axes=("N",), units="nats", dtype="float64")
    chromosome_best: np.ndarray = record_field("state", axes=("C", "N"), units="nats", dtype="float64")
    floors: np.ndarray = record_field("state", axes=("C", "N"), units="nats", dtype="float64")
    markers: np.ndarray = record_field("obs", axes=("C",), dtype="float64")
    eligible_counts: np.ndarray = record_field("state", axes=("N",), dtype="int64")

    @property
    def n_samples(self) -> int:
        return len(self.indptr) - 1

    @property
    def n_contigs(self) -> int:
        return len(self.markers)

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in vars(self).values())

    def retained(self, child: int) -> np.ndarray:
        return self.parents[self.indptr[child] : self.indptr[child + 1]]

    def values(self, child: int, parents: np.ndarray) -> np.ndarray:
        """C x requested identities. Caller supplies legitimate eligible parents."""
        parents = np.asarray(parents, dtype=np.int64)
        first, stop = self.indptr[child : child + 2]
        retained = self.parents[first:stop]
        output = np.broadcast_to(self.floors[:, child, None], (self.n_contigs, len(parents))).copy()
        where = np.searchsorted(retained, parents)
        possible = np.flatnonzero(where < len(retained))
        hits = possible[retained[where[possible]] == parents[possible]]
        output[:, hits] = self.raw[:, first + where[hits]]
        return output

    def chromosome_reference(self, chromosome: int, eligibility: ParentEligibility) -> np.ndarray:
        """One dense chromosome's score and exposure arrays."""
        output = np.full((self.n_samples, self.n_samples), -np.inf)
        for child in np.flatnonzero(eligibility.children):
            parents = np.flatnonzero(eligibility.parents[child])
            output[child, parents] = self.floors[chromosome, child]
            first, stop = self.indptr[child : child + 2]
            output[child, self.parents[first:stop]] = self.raw[chromosome, first:stop]
        return output

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ScreenBatchCounts:
    """PS12 query counts of a child batch; scientific work receipts, not timings."""

    global_queries: int = record_field("state", dtype="int64")
    local_max_queries: int = record_field("state", dtype="int64")
    rounds: int = record_field("state", dtype="int64")
    seed_queries: int = record_field("state", dtype="int64")
    B_frontier_queries: int = record_field("state", dtype="int64")
    floor_frontier_queries: int = record_field("state", dtype="int64")
    nonfinite_upper_pairs: int = record_field("state", dtype="int64")
    eligible_chromosome_pairs: int = record_field("state", dtype="int64")
    queried_chromosome_pairs: int = record_field("state", dtype="int64")
    retained_pairs: int = record_field("state", dtype="int64")


@dataclass(frozen=True)
class ScreenPhase:
    """One contig's execution phase, separate from selection counts and identity."""

    contig: str = record_field("diagnostic", dtype="str")
    phase: str = record_field("diagnostic", dtype="str")
    wall_seconds: float = record_field("diagnostic", units="s", dtype="float64")
    cpu_seconds: float = record_field("diagnostic", units="s", dtype="float64")
    state_bytes: int = record_field("diagnostic", units="bytes", dtype="int64")
    retained: bool = record_field("diagnostic", dtype="bool")


@dataclass(frozen=True)
class ScreenWork:
    """Ordered PS12 batch receipts and separate operational phase measurements."""

    batches: tuple[ScreenBatchCounts, ...] = record_field("state", dtype="tuple[ScreenBatchCounts, ...]")
    phases: tuple[ScreenPhase, ...] = record_field("diagnostic", dtype="tuple[ScreenPhase, ...]", default=())


@dataclass(frozen=True)
class ScreenRecord:
    """Genome M1 profile with raw M0 and explicit scientific/operational work (PS12)."""

    contigs: tuple[str, ...] = record_field("obs", dtype="tuple[str, ...]")
    compact: CompactM1 = record_field("state", dtype="CompactM1")
    zero: np.ndarray = record_field("state", axes=("C", "N"), units="nats", dtype="float64")
    work: ScreenWork = record_field("state", dtype="ScreenWork")
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PairBounds:
    """Upward-rounded pair bounds in canonical child/retained-pair order (PS14)."""

    contig: str = record_field("obs", dtype="str")
    bounds: np.ndarray = record_field("state", axes=("P",), units="nats", dtype="float32")
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class RelativeRequest:
    """Sorted child-parent pairs and their baseline pre-guard report-view memberships."""

    pairs: np.ndarray = record_field("report", axes=("Q", "2"), dtype="int64")
    views: tuple[tuple[str, ...], ...] = record_field("report", dtype="tuple[tuple[str, ...], ...]")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class GuardSummary:
    """Family-exclusion summary computed before publication."""

    model: str = record_field("report", dtype="str")
    edges: int = record_field("report", dtype="int64")
    tested: int = record_field("report", dtype="int64")
    withheld: int = record_field("call", dtype="int64")
    cleaning_iterations: int = record_field("diagnostic", dtype="int64")
    cleaning_converged: bool = record_field("diagnostic", dtype="bool")
    cleaning_iteration_bound: int = record_field("diagnostic", dtype="int64")
    prior_veto_abstentions: int = record_field("report", dtype="int64")


@dataclass(frozen=True)
class SupportFractions:
    """Count/fits on the exact axes of the paired SupportCounts; zero when fits is zero."""

    local_configuration: np.ndarray = record_field("call", axes=("A",), dtype="float64")
    graph_configuration: np.ndarray = record_field("call", axes=("A",), dtype="float64")
    local_state: np.ndarray = record_field("call", axes=("N", "3"), dtype="float64")
    local_parent: np.ndarray = record_field("call", axes=("Q",), dtype="float64")
    graph_parent: np.ndarray = record_field("call", axes=("Q",), dtype="float64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SupportRecord:
    """Both raw support counts and exact fractions, without recomputation on resume.

    Parent fraction axes are the corresponding counts' sorted child*N+parent keys.
    The bootstrap multiplicity matrix preserves the actual PCG64 draw stream.
    """

    bootstrap: SupportCounts = record_field("call", dtype="SupportCounts")
    loco: SupportCounts = record_field("call", dtype="SupportCounts")
    multiplicities: np.ndarray = record_field("call", axes=("R", "C"), dtype="float64")
    worker_count: int = record_field("diagnostic", dtype="int")
    bootstrap_graph_budget_exhausted: int = record_field("diagnostic", dtype="int", default=0, kw_only=True)
    loco_graph_budget_exhausted: int = record_field("diagnostic", dtype="int", default=0, kw_only=True)
    maximum_graph_passes_used: int = record_field("diagnostic", dtype="int", default=0, kw_only=True)
    bootstrap_fractions: SupportFractions = record_field("call", dtype="SupportFractions")
    loco_fractions: SupportFractions = record_field("call", dtype="SupportFractions")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)
