"""Pedigree records: typed products, their checkpoint keys and identities.

Per contig (one current record each, bound to its stage identity):
  pedigree/prepared   PreparedChromosome   painting + raw evidence -> transmission inputs
  pedigree/screen     ChromosomeScreen     M0 and all-pairs M1 per component (panel input)
  pedigree/scores     ChromosomeScores     RAW M0/M1/M2 + exposure + junctions for the fixed panel
  pedigree/exclusion  ExclusionEvidence    full-marker Mendelian exclusion of the target dyads
Run:
  pedigree            PedigreeProduct      decisions, typed views for family phase / evaluation, reports

Axes: N samples in run order, S = founder trajectory classes + BACKGROUND (last), B painting bins of a
component, M selected markers of a component (bin-major), T rows of the M2 candidate panel, C informative
chromosomes. Log likelihoods are natural logs of composite (pooled tempered/full) forward likelihoods.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd

from ..evidence.records import EVIDENCE as EVIDENCE_KEY
from ..painting.records import KEY as PAINTING_KEY
from ..store.identity import array_digest, json_digest, stage_identity

STAGE = "pedigree"
PREPARED = "pedigree/prepared"
SCREEN = "pedigree/screen"
SCORES = "pedigree/scores"
SELECTIVE_SCREEN = "pedigree/selective_screen"
FLOORED_SCORES = "pedigree/floored_scores"
EXCLUSION = "pedigree/exclusion"
PRODUCT = "pedigree"
# 2: release 2.2 released parent order (pedigree.slots: dam/sire or canonical), eligibility sexes.
SCHEMA_VERSION = 3

# Parent states: number of parents among the sequenced candidates (M0 is not "founder generation").
M0, M1, M2 = 0, 1, 2
STATE_NAMES = ("zero_observed_parents", "one_observed_parent", "two_observed_parents")
UNRESOLVED = -1     # parent state not called
NO_PARENT = -1      # empty parent slot / external parent


# ------------------------------------------------------------------------------- preparation
@dataclass(frozen=True)
class HammingTransition:
    """Painting's normalised ordered-diplotype transition at every bin boundary of a component.

    same, one_change, two_changes: float64 [B-1], the probability of moving from an ordered diplotype
    to one specific diplotype that keeps both founders, changes exactly one, or changes both.
    Each row of the S^2 x S^2 matrix sums to one: same + 2 (S-1) one_change + (S-1)^2 two_changes.
    """

    same: np.ndarray
    one_change: np.ndarray
    two_changes: np.ndarray


@dataclass(frozen=True)
class SourceFactors:
    """Exact source-posterior chain of every sample as a candidate parent (pedigree's reduced source model).

    initial:      float64 [N, S, S] normalised ordered-diplotype law of bin 0 (homologue-symmetrised).
    right_weight: float64 [N, B-1, S, S] backward factors r_b(v), max-normalised per boundary, so that
                  K_b(u, v) = A_b(u, v) r_b(v) / sum_w A_b(u, w) r_b(w) is the posterior chain kernel.
    available:    bool [N] samples with at least one informative painting site; others keep a uniform
                  initial law and unit right weights.
    """

    initial: np.ndarray
    right_weight: np.ndarray
    available: np.ndarray


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
    genotype_likelihoods: float64 [N, M, 3] normalised raw GLs at the selected markers.
    observed:          bool [N, M] read depth > 0 at the selected markers.
    state_classes:     int16 [N, B, 2] painting MAP state class per sample, bin and track; -1 = unknown
                       (abstention, BACKGROUND or unanchored trajectory). Junction-count input.
    information_exponent: float64 [N, B] tempered-view emission exponent of each child and bin.
    """

    component_index: int
    named_alleles: np.ndarray
    bin_sizes: np.ndarray
    anchored_states: np.ndarray
    transition: HammingTransition
    factors: SourceFactors
    robustness_epsilon: float
    selector_switch_probability: np.ndarray
    genotype_likelihoods: np.ndarray
    observed: np.ndarray
    state_classes: np.ndarray
    information_exponent: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.anchored_states.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.bin_sizes.shape[0])

    @property
    def informative_markers(self) -> int:
        return int(self.named_alleles.shape[1])


@dataclass(frozen=True)
class PreparedChromosome:
    """Prepared components of one contig (record ``pedigree/prepared``).

    component_count: painting components on the contig; ``components`` keeps only components with
    observed non-uniform evidence at a selected marker, in painting order. ``omitted_reason`` is
    "no_observed_nonuniform_component_evidence" when none remains, else None.
    """

    contig: str
    component_count: int
    components: tuple[PreparedComponent, ...]
    omitted_reason: str | None = None

    @property
    def informative_markers(self) -> int:
        return sum(component.informative_markers for component in self.components)


# ------------------------------------------------------------------------------- screening / scoring
@dataclass(frozen=True)
class ComponentScreen:
    """Pooled M0 and all-pairs M1 composite log likelihoods of one component.

    zero: float64 [N], -inf for samples that are not eligible children.
    one:  float64 [N, N] [child, parent], -inf off the eligible edges (and on the diagonal).
    """

    zero: np.ndarray
    one: np.ndarray


@dataclass(frozen=True)
class ChromosomeScreen:
    """Screen of one informative contig (record ``pedigree/screen``); components as prepared."""

    contig: str
    components: tuple[ComponentScreen, ...]
    informative_markers: int

    def pair_scores(self) -> np.ndarray:
        """float64 [N, N]: the contig's M1 screen, the sum over its components."""
        return np.sum(np.stack([component.one for component in self.components]), axis=0)


@dataclass(frozen=True)
class CandidatePanel:
    """Genome-wide fixed M2 panel.

    trios:         int64 [T, 3] rows (child, first, second), first < second, sorted by child then pair;
                   every eligible pair among the child's top ``parents_per_child`` screen parents.
    screen_scores: float64 [N, N] robust screen utility [child, parent]; -inf off the eligible edges.
    """

    trios: np.ndarray
    screen_scores: np.ndarray


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
    junction_log_cdf: float64 [D] log P(the junctions that c inherits from p and detects are at most c's own
        count) under linked transmission (pedigree.junctions.linked_log_cdf, release 2.16).
    """

    contig: str
    zero: np.ndarray
    one: np.ndarray
    two: np.ndarray
    edge_matched: np.ndarray
    edge_exposed: np.ndarray
    pair_explained: np.ndarray
    pair_exposed: np.ndarray
    total_bins: float
    junctions: np.ndarray
    callable_bins: np.ndarray
    informative_markers: int
    component_count: int
    scored_component_count: int
    junction_dyads: np.ndarray
    junction_log_cdf: np.ndarray


# ChromosomeScores fields stacked whole ([C, N], [C, T], [C, D]) and the [N, N] fields of which the genome
# scores keep only the eligible edges ([C, E]).
_STACKED_FIELDS = ("zero", "two", "pair_explained", "pair_exposed", "junctions", "callable_bins",
                   "junction_log_cdf")
_EDGE_FIELDS = ("one", "edge_matched", "edge_exposed")


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
             axis (release 2.16): the chromosome records' [N, N] entries off the eligible edges (M1 -inf, exposure
             of non-edges and self coverage) are never read by the decision or the report.
    zero float64 [C, N]; two, pair_explained, pair_exposed float64 [C, T] (panel rows); total_bins float64 [C];
    informative_markers int64 [C]; junctions / callable_bins float64 [C, N]; junction_dyads int64 [D, 2] and
    junction_log_cdf float64 [C, D] (ChromosomeScores).
    """

    sample_ids: tuple[str, ...]
    contigs: tuple[str, ...]
    panel: CandidatePanel
    edges: np.ndarray
    zero: np.ndarray
    one: np.ndarray
    two: np.ndarray
    edge_matched: np.ndarray
    edge_exposed: np.ndarray
    pair_explained: np.ndarray
    pair_exposed: np.ndarray
    total_bins: np.ndarray
    junctions: np.ndarray
    callable_bins: np.ndarray
    informative_markers: np.ndarray
    junction_dyads: np.ndarray
    junction_log_cdf: np.ndarray

    @classmethod
    def stack(cls, sample_ids: tuple[str, ...], panel: CandidatePanel, chromosomes,
              eligibility) -> "GenomeScores":
        """Stack the chromosome records in contig order on the edges of ``eligibility`` (ParentEligibility).

        ``chromosomes`` (ChromosomeScores) is iterated once, so it may decode the records one at a time: only each
        record's eligible-edge entries of its [N, N] arrays are kept, and no [C, N, N] array is formed.
        """
        n_samples = len(sample_ids)
        edges = np.argwhere(eligibility.parents).astype(np.int64)
        flat = edges[:, 0] * n_samples + edges[:, 1]
        columns = {name: [] for name in _STACKED_FIELDS + _EDGE_FIELDS}
        contigs, total_bins, markers = [], [], []
        dyads = np.zeros((0, 2), dtype=np.int64)
        for value in chromosomes:
            # The oriented dyads follow the eligibility only, so they are the same on every chromosome.
            # Decoded records can share one buffer among all their arrays (store codec), so every kept field is
            # copied: a view would keep the record's dense [N, N] arrays alive.
            if not contigs:
                dyads = np.array(value.junction_dyads, dtype=np.int64, copy=True)
            elif not np.array_equal(value.junction_dyads, dyads):
                raise ValueError("chromosome scores orient different dyads")
            contigs.append(value.contig)
            total_bins.append(value.total_bins)
            markers.append(value.informative_markers)
            for name in _STACKED_FIELDS:
                columns[name].append(np.array(getattr(value, name), copy=True))
            for name in _EDGE_FIELDS:
                columns[name].append(np.asarray(getattr(value, name), dtype=np.float64).reshape(-1)[flat])
            del value  # before the next record is decoded

        def stacked(name, dtype=np.float64):
            return np.ascontiguousarray(np.stack(columns.pop(name)), dtype=dtype)
        return cls(
            sample_ids=tuple(sample_ids), contigs=tuple(contigs), panel=panel, edges=edges,
            zero=stacked("zero"), one=stacked("one"), two=stacked("two"),
            edge_matched=stacked("edge_matched"), edge_exposed=stacked("edge_exposed"),
            pair_explained=stacked("pair_explained"), pair_exposed=stacked("pair_exposed"),
            total_bins=np.asarray(total_bins, dtype=np.float64),
            junctions=stacked("junctions"), callable_bins=stacked("callable_bins"),
            informative_markers=np.asarray(markers, dtype=np.int64),
            junction_dyads=dyads, junction_log_cdf=stacked("junction_log_cdf"))

    def edge_starts(self) -> np.ndarray:
        """int64 [N + 1]: child c's eligible edges are edges[starts[c]:starts[c + 1]] (parents ascending)."""
        return np.searchsorted(self.edges[:, 0], np.arange(len(self.sample_ids) + 1)).astype(np.int64)


# ------------------------------------------------------------------------------- exclusion
@dataclass(frozen=True)
class ExclusionEvidence:
    """Full-marker exclusion of one contig (record ``pedigree/exclusion``).

    dyads:          int64 [D, 2] (child, candidate parent) in target order.
    log_e:          float64 [D] exact log E, or a certified lower endpoint.
    log_e_upper:    optional float64 [D] upper endpoint; None means legacy exact evidence.
    observed_sites: int64 [D] sites observed in both samples.
    markers:        markers of the contig; pattern_fallback_samples: samples scored without the
                    exact repeated-pattern cache (more than its cap of distinct GL triples).
    """

    contig: str
    dyads: np.ndarray
    log_e: np.ndarray
    observed_sites: np.ndarray
    markers: int
    pattern_fallback_samples: int
    log_e_upper: np.ndarray | None = None


# ------------------------------------------------------------------------------- product
@dataclass(frozen=True)
class PedigreeCalls:
    """One relationship view over the run's samples.

    state:   int8 [N] observed-parent count 0/1/2, or UNRESOLVED (-1).
    parents: int64 [N, 2] parent sample indices, NO_PARENT (-1) for an empty or unresolved slot;
             parent order is arbitrary (sample-array order of the configuration).
    status:  inference status per sample (e.g. "tier_b_supported_one_observed_parent").
    """

    state: np.ndarray
    parents: np.ndarray
    status: tuple[str, ...]


@dataclass(frozen=True)
class ParentEdges:
    """Supported Tier-B parent -> child edges: the pedigree input of family phase and recombination.

    parents, children, slots: int64 [E]; ``slots`` is 0 for the child's Parent1 column, 1 for Parent2.
    Edges are ordered by child, then slot. An edge does not establish whether the child's other
    parent was sampled; unreleased candidates never appear.
    observed_parent_minimum / observed_parent_maximum: int64 [N] bounds of each sample's
    observed-parent count (minimum = released edges, maximum = 2 unless the count is resolved).
    """

    sample_ids: tuple[str, ...]
    parents: np.ndarray
    children: np.ndarray
    slots: np.ndarray
    observed_parent_minimum: np.ndarray
    observed_parent_maximum: np.ndarray

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

    A non-M0 configuration is testable in a fit only when its edges are exposed on at least
    ``minimum_exposed_contigs`` distinct chromosomes, which caps the achievable support:
    max_bootstrap_support = fraction of this run's bootstrap draws with that many distinct
    chromosomes; max_loco_support = 1 if C - 1 >= minimum_exposed_contigs else 0.
    tier_*_reachable: whether those caps meet the tier thresholds. minimum_contigs_tier_*: the
    smallest chromosome count for which they would (exact combinatorics). ``statement``: one line
    for the log and reports.
    """

    informative_contigs: int
    minimum_exposed_contigs: int
    max_bootstrap_support: float
    max_loco_support: float
    tier_a_reachable: bool
    tier_b_reachable: bool
    minimum_contigs_tier_a: int
    minimum_contigs_tier_b: int
    statement: str


@dataclass(frozen=True)
class PedigreeReport:
    """Report tables (written by the reporting layer; one CSV each, same name)."""

    tier_b_relationships: pd.DataFrame
    tier_b_partial_relationships: pd.DataFrame
    tier_a_relationships: pd.DataFrame
    tier_a_partial_relationships: pd.DataFrame
    complete_relationships: pd.DataFrame
    tier_b_candidate_sets: pd.DataFrame
    parent_state_calls: pd.DataFrame
    diagnostics: pd.DataFrame
    call_explanations: pd.DataFrame
    alternative_explanations: pd.DataFrame
    search_diagnostics: pd.DataFrame
    exclusion_evidence: pd.DataFrame
    exclusion_diagnostics: pd.DataFrame
    evidence_summary: pd.DataFrame
    chromosome_summary: pd.DataFrame
    # Release 2.17: released-edge relative-proxy contrasts (pedigree.relatives; None in earlier records).
    relative_contrasts: pd.DataFrame | None = None


@dataclass(frozen=True)
class PedigreeProduct:
    """Genome-wide pedigree (record ``pedigree``).

    sample_ids: run sample axis (N); contigs: configured contigs in order; informative_contigs: the
    scored ones (others: ``omitted`` {contig: reason}).
    calls:    Tier-B configuration calls after the full-marker M0/M1 release (evaluation reads these).
    edges:    supported Tier-B edges (family phase and recombination read only these).
    tier_a, complete: the strict Tier-A view and the full-data acyclic (unreleased) view.
    reach:    the chromosome-count requirement of Tier A/B for this run (TierReach).
    calibration, family_solve, exclusion, eligibility: small JSON-safe summaries.
    panel:    the fixed M2 candidate panel (trios and screen utilities).
    report:   report tables.
    tie_events: full-data-fit decisions taken inside a tie (``maths.ties.TieCounter`` record; release 2.1).
    """

    sample_ids: tuple[str, ...]
    contigs: tuple[str, ...]
    informative_contigs: tuple[str, ...]
    omitted: dict[str, str]
    calls: PedigreeCalls
    edges: ParentEdges
    tier_a: PedigreeCalls
    complete: PedigreeCalls
    reach: TierReach
    calibration: dict[str, Any]
    family_solve: dict[str, Any]
    exclusion: dict[str, Any]
    eligibility: dict[str, Any]
    panel: CandidatePanel
    report: PedigreeReport = field(repr=False)
    tie_events: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------- identities
def _identity(config_part: dict, upstream: dict) -> dict:
    return stage_identity(STAGE, schema=SCHEMA_VERSION, config=config_part, upstream=upstream)


def prepared_identity(file, config, genetic_map) -> dict:
    """Prepared factors <- this contig's painting and evidence records, preparation settings, map."""
    return _identity({"preparation": config.preparation}, {
        "painting": file.info(PAINTING_KEY).identity, "evidence": file.info(EVIDENCE_KEY).identity,
        "genetic_map": json_digest(genetic_map.identity())})


def screen_identity(file, config, eligibility) -> dict:
    """Raw component screen and aggregate model <- prepared, scoring, floor, eligibility."""
    from . import m1
    return _identity({"scoring": config.scoring, "m1": m1.identity(config)},
                     {"prepared": file.info(PREPARED).identity, "eligibility": eligibility.identity()})


def scores_identity(file, panel: "CandidatePanel") -> dict:
    """Raw chromosome scores <- screen record and the genome-retained M2 panel."""
    return _identity({}, {"screen": file.info(SCREEN).identity, "panel": array_digest(panel.trios)})


def selective_screen_identity(store, contigs, config, eligibility) -> dict:
    """Whole-genome profile <- all prepared factors, original model and eligibility."""
    from . import m1
    upstream = {"eligibility": eligibility.identity()}
    upstream.update({f"prepared/{c}": store.contig(c).info(PREPARED).identity for c in contigs})
    return _identity({"scoring": config.scoring, "m1": m1.identity(config),
                      "representation": "genome-floor-csr-v1"}, upstream)


def floored_scores_identity(store, file, panel) -> dict:
    """Fixed-profile scores use a separate key from legacy raw M1 records."""
    return _identity({"m1_values": "fixed-genome-profile"}, {
        "prepared": file.info(PREPARED).identity,
        "selective_screen": store.run.info(SELECTIVE_SCREEN).identity,
        "panel": array_digest(panel.trios)})


def exclusion_identity(file, config, dyads: np.ndarray, *, certificate=None) -> dict:
    """Exclusion evidence <- evidence record, target dyads, exclusion settings."""
    settings = {"exclusion": config.exclusion}
    if certificate is not None:
        settings["certificate"] = certificate
    return _identity(settings,
                     {"evidence": file.info(EVIDENCE_KEY).identity, "dyads": array_digest(dyads)})


def product_identity(store, contigs, informative, config, eligibility, panel,
                     *, scores_key=SCORES) -> dict:
    """The pedigree <- every contig's prepared and scores records, the panel and all settings (the
    exclusion targets and evidence are a deterministic function of these)."""
    upstream = {"eligibility": eligibility.identity(), "panel": array_digest(panel.trios)}
    for contig in contigs:
        file = store.contig(contig)
        upstream[f"prepared/{contig}"] = file.info(PREPARED).identity
        if contig in informative:
            upstream[f"scores/{contig}"] = file.info(scores_key).identity
    return _identity({"pedigree": config}, upstream)


# ------------------------------------------------------------------------------- encoding
# Records are plain dicts of arrays and builtins (no package classes inside stored payloads).
def encode_prepared(value: PreparedChromosome) -> dict:
    return dict(schema=SCHEMA_VERSION, contig=value.contig, component_count=value.component_count,
                omitted_reason=value.omitted_reason,
                component_indices=[c.component_index for c in value.components],
                informative_markers=value.informative_markers, components=[dict(
                    component_index=c.component_index, named_alleles=c.named_alleles,
                    bin_sizes=c.bin_sizes, anchored_states=c.anchored_states,
                    same=c.transition.same, one_change=c.transition.one_change,
                    two_changes=c.transition.two_changes, initial=c.factors.initial,
                    right_weight=c.factors.right_weight, available=c.factors.available,
                    robustness_epsilon=c.robustness_epsilon,
                    selector_switch_probability=c.selector_switch_probability,
                    genotype_likelihoods=c.genotype_likelihoods, observed=c.observed,
                    state_classes=c.state_classes, information_exponent=c.information_exponent)
                    for c in value.components])


def decode_prepared(record: dict) -> PreparedChromosome:
    components = tuple(PreparedComponent(
        component_index=int(c["component_index"]), named_alleles=c["named_alleles"],
        bin_sizes=c["bin_sizes"], anchored_states=c["anchored_states"],
        transition=HammingTransition(c["same"], c["one_change"], c["two_changes"]),
        factors=SourceFactors(c["initial"], c["right_weight"], c["available"]),
        robustness_epsilon=float(c["robustness_epsilon"]),
        selector_switch_probability=c["selector_switch_probability"],
        genotype_likelihoods=c["genotype_likelihoods"], observed=c["observed"],
        state_classes=c["state_classes"], information_exponent=c["information_exponent"])
        for c in record["components"])
    return PreparedChromosome(record["contig"], int(record["component_count"]), components,
                              record["omitted_reason"])


def encode_screen(value: ChromosomeScreen) -> dict:
    return dict(schema=SCHEMA_VERSION, contig=value.contig,
                informative_markers=value.informative_markers,
                zero=[c.zero for c in value.components], one=[c.one for c in value.components])


def decode_screen(record: dict) -> ChromosomeScreen:
    return ChromosomeScreen(record["contig"], tuple(
        ComponentScreen(zero, one) for zero, one in zip(record["zero"], record["one"])),
        int(record["informative_markers"]))


def _fields(value) -> dict:
    """Top-level fields of a record dataclass without copying its arrays (dataclasses.asdict deep-copies)."""
    return {item.name: getattr(value, item.name) for item in fields(value)}


def encode_scores(value: ChromosomeScores) -> dict:
    return dict(schema=SCHEMA_VERSION, **_fields(value))


def decode_scores(record: dict) -> ChromosomeScores:
    record = dict(record)
    record.pop("schema")
    return ChromosomeScores(**record)


def encode_exclusion(value: ExclusionEvidence) -> dict:
    return dict(schema=SCHEMA_VERSION, **_fields(value))


def decode_exclusion(record: dict) -> ExclusionEvidence:
    record = dict(record)
    record.pop("schema")
    return ExclusionEvidence(**record)


def encode_product(value: PedigreeProduct) -> dict:
    calls = lambda c: dict(state=c.state, parents=c.parents, status=list(c.status))  # noqa: E731
    return dict(
        schema=SCHEMA_VERSION, sample_ids=list(value.sample_ids), contigs=list(value.contigs),
        informative_contigs=list(value.informative_contigs), omitted=dict(value.omitted),
        calls=calls(value.calls), tier_a=calls(value.tier_a), complete=calls(value.complete),
        edges=dict(parents=value.edges.parents, children=value.edges.children,
                   slots=value.edges.slots,
                   observed_parent_minimum=value.edges.observed_parent_minimum,
                   observed_parent_maximum=value.edges.observed_parent_maximum),
        reach=asdict(value.reach), calibration=value.calibration,
        family_solve=value.family_solve, exclusion=value.exclusion,
        eligibility=value.eligibility,
        panel=dict(trios=value.panel.trios, screen_scores=value.panel.screen_scores),
        report={field.name: getattr(value.report, field.name) for field in fields(value.report)},
        tie_events=dict(value.tie_events))


def _decode_calls(record: dict) -> PedigreeCalls:
    return PedigreeCalls(record["state"], record["parents"], tuple(record["status"]))


def decode_edges(record: dict, sample_ids) -> ParentEdges:
    return ParentEdges(tuple(sample_ids), record["parents"], record["children"], record["slots"],
                       record["observed_parent_minimum"], record["observed_parent_maximum"])


def decode_product(record: dict) -> PedigreeProduct:
    return PedigreeProduct(
        sample_ids=tuple(record["sample_ids"]), contigs=tuple(record["contigs"]),
        informative_contigs=tuple(record["informative_contigs"]), omitted=dict(record["omitted"]),
        calls=_decode_calls(record["calls"]), edges=decode_edges(record["edges"], record["sample_ids"]),
        tier_a=_decode_calls(record["tier_a"]), complete=_decode_calls(record["complete"]),
        reach=TierReach(**record["reach"]), calibration=record["calibration"],
        family_solve=record["family_solve"], exclusion=record["exclusion"],
        eligibility=record["eligibility"],
        panel=CandidatePanel(record["panel"]["trios"], record["panel"]["screen_scores"]),
        report=PedigreeReport(**record["report"]), tie_events=dict(record.get("tie_events", {})))


# ------------------------------------------------------------------------------- readers
def complete(store) -> bool:
    """Whether the run's pedigree is recorded."""
    return store.run.has(PRODUCT)


def load_product(store) -> PedigreeProduct:
    """The run's PedigreeProduct (report tables included)."""
    return decode_product(store.run.get(PRODUCT))


def load_edges(store) -> ParentEdges:
    """Only the supported Tier-B edges (family phase, recombination): no report decoding."""
    record = store.run.get(PRODUCT, items=("sample_ids", "edges"))
    return decode_edges(record["edges"], record["sample_ids"])


def load_calls(store) -> tuple[tuple[str, ...], PedigreeCalls]:
    """(sample IDs, Tier-B calls) for evaluation."""
    record = store.run.get(PRODUCT, items=("sample_ids", "calls"))
    return tuple(record["sample_ids"]), _decode_calls(record["calls"])
