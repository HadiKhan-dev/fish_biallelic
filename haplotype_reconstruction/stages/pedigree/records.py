"""Bind pedigree products to every scientific upstream derivation and explicit axis.

Owns pedigree's keys, checked dependency reads and identity construction; no scientific computation.
Method: docs/methods/pedigree.md#products-and-published-tables (PD27.
The relatedness input is the families product.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ...contracts.contigs import AnalysisContigs
from ...contracts.payload import from_payload, to_payload
from ...contracts.pedigree import CandidatePanel, ChromosomeSummaryView, ExclusionRequest, ParentEligibility, Pedigree
from ...contracts.roles import require_schema
from ...core.concurrency import threaded_each
from ...core.digest import json_digest
from ...core.errors import InputError, InvariantError
from ...core.ties import TieCounts
from ...science.pedigree.parameters import PedigreeParameters
from ...science.pedigree.run import exclusion_contigs
from ...store.container import CheckpointFile
from ...store.identity import CodeIdentity, code_dependency_identity, pin_generation
from ...store.protocol import Derivation, NumericalPolicy, read_items
from ...store.store import CheckpointStore
from ..evidence import records as evidence_records
from ..framework import load_product
from ..painting import repaint as painting_records  # the painting the pedigree reads (4.21 P)
from ..relatedness import families_records
from . import exclusion, map_scale, prepare, relationship, scores, screen
from .codec import PedigreeAdapter

__all__ = [
    "PRODUCT",
    "PREPARED",
    "SCREEN",
    "PANEL",
    "FLOORED_SCORES",
    "EXCLUSION",
    "PedigreeDependencies",
    "expected",
    "BoundInputs",
    "bound_inputs",
    "request_dependencies",
    "result_reader",
]

PRODUCT = "pedigree"
PREPARED = prepare.KEY
SCREEN = screen.SCREEN
PANEL = screen.PANEL
FLOORED_SCORES = scores.KEY
EXCLUSION = exclusion.KEY
RELATIONSHIP = relationship.KEY
REQUEST = "pedigree/exclusion_request"

# Prepared inputs depend on the selected analysis through painting and the working map.
ANALYSIS_KEYS = (
    PRODUCT, PREPARED, SCREEN, PANEL, screen.BOUNDS, FLOORED_SCORES, EXCLUSION, RELATIONSHIP, REQUEST, map_scale.KEY,
    map_scale.INPUTS, map_scale.PAIRS, map_scale.FRAMES,
)


@dataclass(frozen=True)
class PedigreeDependencies:
    """Named derivation identities, in configured/informative contig order as appropriate."""

    eligibility: str
    panel: str
    relatedness_families: str
    prepared: tuple[tuple[str, str], ...]
    floored_scores: tuple[tuple[str, str], ...]
    exclusion: tuple[tuple[str, str], ...] = ()
    reports: tuple[tuple[str, str], ...] = ()
    evidence: tuple[tuple[str, str], ...] = ()
    relationship: tuple[tuple[str, str], ...] = ()
    regime: str = "standard"
    sample_rates: tuple[float, ...] = ()  # A11 (4.22): every sample's outlier rate, the exclusion's allowances
    map_scale: str = ""  # identity of pedigree/map_scale (regimes.material_choice reads it)
    veto: str = ""  # the regime whose released calls a call must also be in (both-maps route), or ""
    reference: str = ""  # the working regime, run only to count the product against it (regimes.reference_regime)
    exclusion_sharing: tuple[bool, ...] = ()
    exclusion_rate_at_top: tuple[bool, ...] = ()  # samples whose rate sits at the calibration grid's top: withheld
    contamination: str = ""  # identity of the run's contamination decision (4.22 QC A; the exclusion reads it)


def expected(
    parameters: PedigreeParameters,
    sample_ids: tuple[str, ...],
    contigs: tuple[str, ...],
    informative_contigs: tuple[str, ...],
    inputs: PedigreeDependencies,
    *,
    numerical_policy: NumericalPolicy | None = None,
    code: CodeIdentity | None = None,
) -> str:
    """Build the real schema-10 derivation; workers/storage routes and output switches are absent.

    Prepared records cover configured C, while floored records cover informative C.
    Exclusion is empty only for an empty request; otherwise it covers the configured contigs with material
    (positive exclusion weight), in configured order.
    Code roots include input construction, adapters, codec and scientific closure.
    """
    for label, actual, expected in (
        ("prepared", inputs.prepared, contigs),
        ("floored scores", inputs.floored_scores, () if inputs.regime == "not_assessable" else informative_contigs),
        ("relationship", inputs.relationship, contigs if inputs.relationship else ()),
        ("evidence", inputs.evidence, contigs if inputs.evidence else ()),
    ):
        if tuple(name for name, _ in actual) != expected:
            raise InputError(f"pedigree {label} identities do not follow their contig axis")
    excluded = tuple(name for name, _ in inputs.exclusion)
    if tuple(c for c in contigs if c in set(excluded)) != excluded:
        raise InputError("pedigree exclusion identities do not follow their contig axis")
    upstream = {
        "eligibility": inputs.eligibility,
        "panel": inputs.panel,
        "relatedness/families": inputs.relatedness_families,
    }
    if inputs.regime == "not_assessable":
        upstream.pop("panel")
        upstream.pop("relatedness/families")
    for prefix, entries in (
        (PREPARED, inputs.prepared),
        (FLOORED_SCORES, inputs.floored_scores),
        (EXCLUSION, inputs.exclusion),
        (RELATIONSHIP, inputs.relationship),
        ("report", inputs.reports),
        ("evidence", inputs.evidence),
    ):
        for name, identity in entries:
            key = prefix + "/" + name
            if key in upstream:
                raise InputError(f"duplicate pedigree dependency {key}")
            upstream[key] = identity
    if inputs.map_scale:
        upstream[map_scale.KEY] = inputs.map_scale
    if not all(upstream.values()):
        raise InputError("pedigree derivation identities must be nonempty")
    closure = code or code_dependency_identity(
        (
            "stages.pedigree.decision",
            "stages.pedigree.execution",
            "stages.pedigree.codec",
            "stages.pedigree.records",
            "science.pedigree.run",
            "science.pedigree.views",
            "science.pedigree.stacking",
        )
    )
    return Derivation(
        closure,
        dict(
            pedigree=parameters,
            sample_ids=sample_ids,
            configured_contigs=contigs,
            informative_contigs=informative_contigs,
            product_schema=PedigreeAdapter.schema,
            regime=inputs.regime,
            veto=inputs.veto,
            reference=inputs.reference,
            partition="genome-axis-material-units-v1",
        ),
        numerical_policy or NumericalPolicy.current(),
        upstream,
        parameters.reverse_conventions,
    ).identity


@dataclass(frozen=True)
class BoundInputs:
    """Small checked descriptors; no prepared components or chromosome score arrays."""

    parameters: PedigreeParameters
    policy: NumericalPolicy
    sample_ids: tuple[str, ...]
    contigs: tuple[str, ...]
    eligibility: ParentEligibility
    panel: CandidatePanel | None
    summary: tuple[ChromosomeSummaryView, ...]
    dependencies: PedigreeDependencies
    classification: Any = None
    axis: Any = None
    material: Any = None  # MaterialChoice
    map_scale: Any = None  # MapScaleEstimate
    regime_morgans: float | None = None  # the Morgans the applied regime reads (None: U)
    veto_morgans: float | None = None  # the Morgans the veto regime reads (None: U)

    @property
    def informative_contigs(self) -> tuple[str, ...]:
        return tuple(row.contig for row in self.summary if row.omitted_reason is None)


def _header(
    file: CheckpointFile,
    key: str,
    identity: str,
    names: tuple[str, ...] = (),
    *,
    schemas: int | tuple[int, ...] = 1,
) -> Mapping[str, Any]:
    """Check identity before decoding even a scalar header."""
    found = read_items(file, key, identity, ("schema", *names))
    if found is None:
        raise InvariantError(f"pedigree requires complete upstream {file.label}:{key}")
    require_schema(found.payload, schemas)
    return found.payload


def run_parameters(manifest: Mapping[str, Any]) -> PedigreeParameters:
    """The bound run's pedigree parameters: the manifest's, with the run's tie convention (``reverse_conventions``
    from the manifest's ties; the parameters payload keeps the configured value). Every producer and cold reader of a
    pedigree identity (records, the map-scale record, the run-log reports) reads its parameters here, so that they
    agree under reversed ties."""
    return replace(from_payload(PedigreeParameters, manifest["parameters"]["pedigree"]),
                   reverse_conventions=bool(manifest["ties"]["reverse"]))


def bound_inputs(store: CheckpointStore, manifest: Mapping[str, Any], *,
                 causes: tuple[dict[str, tuple[str, str, str]], str] | None = None) -> BoundInputs:
    """Rebuild pedigree preparation–pedigree scoring causes from the bound manifest, validating every publication.

    Read omission/summary headers and the run panel. Mixed read-library status
    also needs the stored evidence counts to recover sample membership. Original
    genotype/map sources are never reopened. A cached product's own derivation is
    never an input to the expected identity. Output-only relatedness settings are
    excluded by the families owner's reader.
    """
    parameters = run_parameters(manifest)
    policy = NumericalPolicy(**manifest["numerical_policy"])
    sample_ids = tuple(manifest["sample_ids"])
    contigs = AnalysisContigs.from_manifest(manifest).included
    eligibility = ParentEligibility.unrestricted(sample_ids)
    contig_inputs, contamination = map_scale.contig_causes(store, manifest) if causes is None else causes

    def bind(c: str) -> tuple[str, Mapping[str, Any]]:
        file = store.contig(c)
        painting, evidence, working_map = contig_inputs[c]
        _header(file, evidence_records.EVIDENCE, evidence, schemas=evidence_records.SCHEMA)
        _header(file, painting_records.REFIT_REFERENCE, painting, schemas=painting_records.SCHEMA)
        prepared_identity = prepare.derivation(
            parameters.preparation,
            dict(painting=painting, evidence=evidence, genetic_map=working_map),
            reverse=parameters.reverse_conventions,
            numerical_policy=policy,
        )
        header = _header(file, PREPARED, prepared_identity, prepare.LAYOUT_ITEMS[1:],
                         schemas=prepare.PreparedAdapter.schema)
        if header["contig"] != c:
            raise InputError("prepared header contig differs from its configured axis")
        return prepared_identity, header

    # each contig's upstream chain (evidence, painting and through it assembly, selection and discovery
    # readers: header reads and decodes) is independent of the others; contig after contig, this binding took most of
    # an export check and of the run's completion step (chr3's export manifest check: 55 s on one CPU).
    bound = threaded_each(bind, contigs)
    prepared_ids = [(c, prepared) for c, (prepared, _) in zip(contigs, bound, strict=True)]
    evidence_ids = [(c, evidence) for c, (_, evidence, _) in contig_inputs.items()]
    prepared_headers = [header for _, header in bound]
    informative = tuple(
        pair for pair, h in zip(prepared_ids, prepared_headers, strict=True) if h["omitted_reason"] is None
    )
    from ...science.pedigree import regimes
    layouts, prepared_summary = [], []
    for header in prepared_headers:
        layout, prepared_row = prepare.layout_summary(header)
        prepared_summary.append(prepared_row)
        if header["omitted_reason"] is None:
            layouts.append(layout)
    # With rate_estimate = ignore nothing in the pedigree reads the map-scale record (it is reported on its own).
    scale_identity, estimate = "", None
    if parameters.map_scale.mode != "ignore":
        scale_identity = map_scale.expected(store, manifest, parameters, policy, contig_inputs=contig_inputs)
        estimate = map_scale.load(store, scale_identity)
    choice, classification, applied_morgans, axis = regimes.material_choice(tuple(layouts), parameters, estimate)
    regime_morgans, veto_morgans = material_morgans(choice, axis, applied_morgans)
    if classification.name == "not_assessable":
        dependencies = PedigreeDependencies(eligibility.identity(), "", "", tuple(prepared_ids), (),
                                           evidence=tuple(evidence_ids), regime=classification.name,
                                           map_scale=scale_identity, veto=choice.veto_regime,
                                           reference=regimes.reference_regime(choice))
        return BoundInputs(parameters, policy, sample_ids, contigs, eligibility, None,
                           tuple(prepared_summary), dependencies, classification, axis, choice, estimate,
                           regime_morgans, veto_morgans)
    screen_identity = screen.screen_derivation(parameters, eligibility.identity(), informative, numerical_policy=policy)
    _header(store.run, SCREEN, screen_identity)
    bound_ids = tuple(
        (c, screen.bound_derivation(parameters, identity, screen_identity, numerical_policy=policy))
        for c, identity in informative
    )
    for c, identity in bound_ids:
        _header(store.contig(c), screen.BOUNDS, identity)
    panel_identity = screen.panel_derivation(
        screen_identity,
        bound_ids,
        parameters.panel.max_pairs_per_child,
        reverse=parameters.reverse_conventions,
        numerical_policy=policy,
    )
    panel = load_product(store.run, PANEL, panel_identity, screen.PanelAdapter())
    if panel is None:
        raise InvariantError("pedigree requires the complete candidate panel")
    placed = prepare.placements(tuple(layouts), axis, parameters)
    score_ids = tuple(
        (c, scores.derivation(parameters, identity, screen_identity, panel, placement, numerical_policy=policy))
        for (c, identity), placement in zip(informative, placed, strict=True)
    )
    score_id_map = dict(score_ids)
    summary = []
    for c, header in zip(contigs, prepared_headers, strict=True):
        if c in score_id_map:
            value = _header(
                store.contig(c),
                FLOORED_SCORES,
                score_id_map[c],
                ("contig", "component_count", "scored_component_count", "informative_markers"),
                schemas=scores.ScoresAdapter.schema,
            )
            if value["contig"] != c or value["component_count"] != header["component_count"]:
                raise InputError("score summary disagrees with its prepared chromosome")
            summary.append(
                ChromosomeSummaryView(
                    c,
                    value["component_count"],
                    value["scored_component_count"],
                    value["informative_markers"],
                    None,
                    float(header["configured_morgans"]),
                )
            )
        else:
            summary.append(ChromosomeSummaryView(c, header["component_count"], 0, 0, header["omitted_reason"],
                                                 float(header["configured_morgans"])))
    family_reader = families_records.result_reader(store, manifest, product="families")
    if family_reader is None:
        raise InvariantError("pedigree requires relatedness/families")
    _header(store.run, family_reader.key, family_reader.identity, schemas=families_records.SCHEMA)
    rates, at_top, sharing = _exclusion_calibration(store, manifest, sample_ids, evidence_ids)
    dependencies = PedigreeDependencies(
        eligibility.identity(),
        panel_identity,
        family_reader.identity,
        tuple(prepared_ids),
        score_ids,
        evidence=tuple(evidence_ids),
        sample_rates=rates,
        exclusion_sharing=sharing,
        exclusion_rate_at_top=at_top,
        regime=classification.name,
        map_scale=scale_identity,
        veto=choice.veto_regime,
        reference=regimes.reference_regime(choice),
        contamination=contamination,
    )
    return BoundInputs(parameters, policy, sample_ids, contigs, eligibility, panel, tuple(summary), dependencies,
                       classification, axis, choice, estimate, regime_morgans, veto_morgans)


def _exclusion_calibration(store, manifest, sample_ids, evidence_ids):
    """Read the exclusion's allowances (4.22 A11) and fitted sharing libraries from the same calibration: each
    sample's outlier rate and whether it sits at the top of the rate grid (unresolved: its dyads never certify).
    A calibration with one candidate rate estimates no per-sample rates: every sample then takes the run's rate
    (robustness_epsilon), none flagged."""
    from ..painting import calibration_records

    reader = calibration_records.result_reader(store, manifest, product="calibration")
    if reader is None:
        raise InvariantError("pedigree exclusion requires the painting calibration's outlier rates")
    saved = read_items(store.run, reader.key, reader.identity,
                       ("sample_rates", "sample_rate_at_top", "robustness_epsilon", "read_libraries"))
    if saved is None:
        raise InvariantError("pedigree exclusion requires the painting calibration's outlier rates")
    calibration = saved.payload
    rates = tuple(float(x) for x in calibration.get("sample_rates", ()))
    at_top = tuple(bool(x) for x in calibration.get("sample_rate_at_top", ()))
    if not rates:
        rates, at_top = (float(calibration["robustness_epsilon"]),) * len(sample_ids), (False,) * len(sample_ids)
    if len(rates) != len(sample_ids) or len(at_top) != len(sample_ids):
        raise InputError("the calibration's per-sample rates do not cover the pedigree's samples")
    sharing = exclusion.sharing_samples(store, manifest, evidence_ids, calibration.get("read_libraries", ()))
    return rates, at_top, sharing


def material_morgans(choice: Any, axis: Any, applied_morgans: float) -> tuple[float | None, float | None]:
    """The Morgans the applied and the veto regimes read (None: the working U itself)."""
    applied = None if applied_morgans == axis.morgans else float(applied_morgans)
    veto = None
    if choice.veto_regime:
        veto = float(choice.estimate_morgans) if applied is None else None
    return applied, veto


def request_dependencies(
    inputs: PedigreeDependencies,
    request: ExclusionRequest,
    parameters: PedigreeParameters,
    policy: NumericalPolicy | None,
) -> PedigreeDependencies:
    """Bind separate count and relationship requests to their actual input records."""
    if tuple(c for c, _ in inputs.evidence) != request.contigs:
        raise InputError("pedigree exclusion/observed-count evidence must cover configured contigs in order")
    weights = dict(exclusion_contigs(request))
    identities = (
        tuple(
            (
                c,
                exclusion.derivation(
                    parameters.exclusion,
                    identity,
                    request.dyads,
                    request.sample_ids,
                    request.contigs,
                    weights[c],
                    numerical_policy=policy,
                    sample_rates=inputs.sample_rates,
                    rate_at_top=inputs.exclusion_rate_at_top,
                    sharing=inputs.exclusion_sharing,
                    contamination=inputs.contamination,
                ),
            )
            for c, identity in inputs.evidence
            if c in weights
        )
        if len(request.dyads)
        else ()
    )
    if inputs.exclusion and inputs.exclusion != identities:
        raise InputError("supplied exclusion identities disagree with the actual ordered request")
    related = (
        tuple(
            (c, relationship.derivation(parameters, identity, request.relationship, index, numerical_policy=policy))
            for index, (c, identity) in enumerate(inputs.prepared)
        )
        if len(request.relationship.dyads)
        else ()
    )
    if inputs.relationship and inputs.relationship != related:
        raise InputError("supplied relationship identities disagree with the actual ordered request")
    return replace(inputs, exclusion=identities, relationship=related)


@dataclass(frozen=True)
class Reader:
    """Results reader; no scientific producer runs while decoding a complete product."""

    identity: str
    product: str = "pedigree"
    key: str = PRODUCT
    schema: int = PedigreeAdapter.schema

    def decode(self, payload: Mapping[str, Any]) -> Pedigree | TieCounts:
        result = PedigreeAdapter().decode(payload)
        if self.product != "pedigree_ties":
            return result
        return TieCounts(()) if result.retained is None else result.retained.decision.tie_events


# Pedigree readers this process derived, by run container (path, inode), manifest digest, product and record-reuse
# pin generation (store.identity.pin_generation: the code identities the derivation binds). A derivation
# checks every contig's evidence, painting, preparation and score headers (bound_inputs), and every phase,
# recombination and export read of each contig needs it: re-deriving it per read made a report over eight contigs
# read for minutes. Those records are immutable once the pedigree is published, so one derivation per run serves the
# process; the product's own identity is still checked at every read.
_DERIVED: dict[tuple[str, int, str, str, int], Reader] = {}


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    causes=None,
) -> Reader | None:
    """Validate all independent causes before requesting the whole immutable product.

    The cached request only supplies the dyad-dependent exclusion identity. Every
    cause of that request is independently bound above; a changed cause refuses
    the product rather than repairing any reports or refitting a decision.
    """
    if product not in ("pedigree", "pedigree_ties") or contig is not None:
        raise InputError("invalid pedigree reader request")
    if not store.run.has(PRODUCT):
        return None
    derived = (store.run.path, os.stat(store.run.path).st_ino, json_digest(manifest), product, pin_generation())
    reader = _DERIVED.get(derived)
    if reader is None:
        reader = _DERIVED[derived] = _derive_reader(store, manifest, product, causes=causes)
    return reader


def _derive_reader(store: CheckpointStore, manifest: Mapping[str, Any], product: str, *, causes=None) -> Reader:
    # Report output needs the bound product and its ordered release request. The remaining contig header
    # reads run independently. Cold matched reports took 12.7 wall / 15.3 CPU s on famhs and 17.0 / 22.7
    # on TroMau after sharing refit and map inputs, down from 39.9 / 51.0 and 158.0 / 233.8.
    bound = bound_inputs(store, manifest, causes=causes)
    request_id = expected(
        bound.parameters,
        bound.sample_ids,
        bound.contigs,
        bound.informative_contigs,
        replace(bound.dependencies, exclusion=(), relationship=()),
        numerical_policy=bound.policy,
    )
    if bound.dependencies.regime == "not_assessable":
        return Reader(request_id, product)
    request = load_product(store.run, REQUEST, request_id, RequestAdapter())
    if request is None:
        raise InvariantError("pedigree publication lacks its independently bound exclusion request")
    if request.sample_ids != bound.sample_ids or request.contigs != bound.contigs:
        raise InputError("cached pedigree request axes differ from the bound run")
    inputs = request_dependencies(bound.dependencies, request, bound.parameters, bound.policy)
    for c, identity in inputs.exclusion:
        _header(store.contig(c), EXCLUSION, identity, schemas=exclusion.ExclusionAdapter.schema)
    for c, identity in inputs.relationship:
        _header(store.contig(c), RELATIONSHIP, identity)
    identity = expected(
        bound.parameters,
        bound.sample_ids,
        bound.contigs,
        bound.informative_contigs,
        inputs,
        numerical_policy=bound.policy,
    )
    return Reader(identity, product)


def published_identity(store: CheckpointStore, manifest: Mapping[str, Any], *, causes=None) -> str | None:
    """Return the current product's identity without decoding scores or the released product."""
    reader = result_reader(store, manifest, product="pedigree", causes=causes)
    if reader is None:
        return None
    _header(store.run, reader.key, reader.identity, schemas=reader.schema)
    return reader.identity


class RequestAdapter:
    """Independently checked decision request; schema 2, no scientific reconstruction."""

    schema = 2

    def encode(self, request: ExclusionRequest) -> Mapping[str, Any]:
        return {"schema": self.schema, **to_payload(request)}

    def decode(self, payload: Mapping[str, Any]) -> ExclusionRequest:
        require_schema(payload, self.schema)
        return from_payload(ExclusionRequest, {k: v for k, v in payload.items() if k != "schema"})
