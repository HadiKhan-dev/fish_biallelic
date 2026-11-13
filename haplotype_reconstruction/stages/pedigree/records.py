"""Bind pedigree products to every scientific upstream derivation and explicit axis.

Owns pedigree's keys, checked dependency reads and identity construction; no scientific computation.
Method: docs/methods/pedigree.md#products-and-published-tables (PD27.
The relatedness input is families, never the optional pairwise-IBD report selection.
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
from ...store.container import CheckpointFile
from ...store.identity import CodeIdentity, code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, read_items
from ...store.store import CheckpointStore
from ..evidence import records as evidence_records
from ..framework import load_product
from ..painting import records as painting_records
from ..relatedness import families_records
from . import exclusion, prepare, relationship, scores, screen
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

# Prepared data depends on one contig; all other pedigree records depend on the analysed genome.
SHARED_KEYS = (PREPARED,)
ANALYSIS_KEYS = (PRODUCT, SCREEN, PANEL, screen.BOUNDS, FLOORED_SCORES, EXCLUSION, RELATIONSHIP, REQUEST)


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
    """Build the real schema-8 derivation; workers/storage routes and output switches are absent.

    Prepared records cover configured C, while floored records cover informative C.
    Exclusion is empty only for an empty request; otherwise it covers configured C.
    Code roots include input construction, adapters, codec and scientific closure.
    """
    for label, actual, expected in (
        ("prepared", inputs.prepared, contigs),
        ("floored scores", inputs.floored_scores, () if inputs.regime == "not_assessable" else informative_contigs),
        ("exclusion", inputs.exclusion, contigs if inputs.exclusion else ()),
        ("relationship", inputs.relationship, contigs if inputs.relationship else ()),
        ("evidence", inputs.evidence, contigs if inputs.evidence else ()),
    ):
        if tuple(name for name, _ in actual) != expected:
            raise InputError(f"pedigree {label} identities do not follow their contig axis")
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
            partition=(1.0, 1.5, 0.5, 0.499999, "component-grid-occupied-bins"),
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


def bound_inputs(store: CheckpointStore, manifest: Mapping[str, Any]) -> BoundInputs:
    """Rebuild pedigree preparation–pedigree scoring causes from the bound manifest, validating every publication.

    Only omission/summary headers and the run panel are decoded here. Original
    genotype/map sources are never reopened. A cached product's own derivation is
    never an input to the expected identity. Output-only relatedness settings are
    excluded by the families owner's reader.
    """
    parameters = replace(
        from_payload(PedigreeParameters, manifest["parameters"]["pedigree"]),
        reverse_conventions=bool(manifest["ties"]["reverse"]),
    )
    policy = NumericalPolicy(**manifest["numerical_policy"])
    sample_ids = tuple(manifest["sample_ids"])
    contigs = AnalysisContigs.from_manifest(manifest).included
    eligibility = ParentEligibility.unrestricted(sample_ids)

    def bind(c: str) -> tuple[str, str, Mapping[str, Any]]:
        file = store.contig(c)
        evidence = evidence_records.result_reader(store, manifest, product="evidence", contig=c)
        painting = painting_records.result_reader(
            store, manifest, product="painting", contig=c, panels=False, emissions=False
        )
        if evidence is None or painting is None:
            raise InvariantError(f"pedigree requires evidence and painting for {c}")
        _header(file, evidence.key, evidence.identity, schemas=evidence_records.SUPPORTED_SCHEMAS)
        _header(file, painting.key, painting.identity)
        prepared_identity = prepare.derivation(
            parameters.preparation,
            dict(painting=painting.identity, evidence=evidence.identity, genetic_map=manifest["map_identities"][c]),
            rate_is_measured=parameters.exposure.rate_is_measured,
            reverse=parameters.reverse_conventions,
            numerical_policy=policy,
        )
        header = _header(file, PREPARED, prepared_identity, ("contig", "component_count", "omitted_reason", "layout"),
                         schemas=prepare.PreparedAdapter.schema)
        if header["contig"] != c:
            raise InputError("prepared header contig differs from its configured axis")
        return prepared_identity, evidence.identity, header

    # each contig's upstream chain (evidence, painting and through it assembly, selection and discovery
    # readers: header reads and decodes) is independent of the others; contig after contig, this binding took most of
    # an export check and of the run's completion step (chr3's export manifest check: 55 s on one CPU).
    bound = threaded_each(bind, contigs)
    prepared_ids = [(c, prepared) for c, (prepared, _, _) in zip(contigs, bound, strict=True)]
    evidence_ids = [(c, evidence) for c, (_, evidence, _) in zip(contigs, bound, strict=True)]
    prepared_headers = [header for _, _, header in bound]
    informative = tuple(
        pair for pair, h in zip(prepared_ids, prepared_headers, strict=True) if h["omitted_reason"] is None
    )
    from ...science.pedigree import regimes
    layouts, prepared_summary = [], []
    for header in prepared_headers:
        layout, prepared_row = prepare.layout_summary(header, parameters)
        prepared_summary.append(prepared_row)
        if header["omitted_reason"] is None:
            layouts.append(layout)
    classification = regimes.classify(layouts, parameters)
    if classification.name == "not_assessable":
        dependencies = PedigreeDependencies(eligibility.identity(), "", "", tuple(prepared_ids), (),
                                           evidence=tuple(evidence_ids), regime=classification.name)
        return BoundInputs(parameters, policy, sample_ids, contigs, eligibility, None,
                           tuple(prepared_summary), dependencies, classification)
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
        parameters.panel.pair_relevance_margin,
        reverse=parameters.reverse_conventions,
        numerical_policy=policy,
    )
    panel = load_product(store.run, PANEL, panel_identity, screen.PanelAdapter())
    if panel is None:
        raise InvariantError("pedigree requires the complete candidate panel")
    score_ids = tuple(
        (c, scores.derivation(parameters, identity, screen_identity, panel, numerical_policy=policy, regime=classification.name))
        for c, identity in informative
    )
    score_id_map = dict(score_ids)
    summary = []
    for c, header in zip(contigs, prepared_headers, strict=True):
        if c in score_id_map:
            value = _header(
                store.contig(c),
                FLOORED_SCORES,
                score_id_map[c],
                ("contig", "component_count", "scored_component_count", "informative_markers", "form"),
                schemas=scores.ScoresAdapter.schema,
            )
            if value.get("form", "standard") != classification.name:
                raise InputError("score record form differs from the run regime")
            if value["contig"] != c or value["component_count"] != header["component_count"]:
                raise InputError("score summary disagrees with its prepared chromosome")
            summary.append(
                ChromosomeSummaryView(
                    c,
                    value["component_count"],
                    value["scored_component_count"],
                    value["informative_markers"],
                    None,
                )
            )
        else:
            summary.append(ChromosomeSummaryView(c, header["component_count"], 0, 0, header["omitted_reason"]))
    family_reader = families_records.result_reader(store, manifest, product="families")
    if family_reader is None:
        raise InvariantError("pedigree requires relatedness/families")
    _header(store.run, family_reader.key, family_reader.identity)
    dependencies = PedigreeDependencies(
        eligibility.identity(),
        panel_identity,
        family_reader.identity,
        tuple(prepared_ids),
        score_ids,
        evidence=tuple(evidence_ids),
        regime=classification.name,
    )
    return BoundInputs(parameters, policy, sample_ids, contigs, eligibility, panel, tuple(summary), dependencies, classification)


def request_dependencies(
    inputs: PedigreeDependencies,
    request: ExclusionRequest,
    parameters: PedigreeParameters,
    policy: NumericalPolicy | None,
) -> PedigreeDependencies:
    """Bind separate count and relationship requests to their actual input records."""
    if tuple(c for c, _ in inputs.evidence) != request.contigs:
        raise InputError("pedigree exclusion/observed-count evidence must cover configured contigs in order")
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
                    numerical_policy=policy,
                ),
            )
            for c, identity in inputs.evidence
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
        return (TieCounts(()) if result.retained is None else result.retained.decision.tie_events) if self.product == "pedigree_ties" else result


# Pedigree readers this process derived, by run container (path, inode), manifest digest and product. A derivation
# checks every contig's evidence, painting, preparation and score headers (bound_inputs), and every phase,
# recombination and export read of each contig needs it: re-deriving it per read made a report over eight contigs
# read for minutes. Those records are immutable once the pedigree is published, so one derivation per run serves the
# process; the product's own identity is still checked at every read.
_DERIVED: dict[tuple[str, int, str, str], Reader] = {}


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
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
    derived = (store.run.path, os.stat(store.run.path).st_ino, json_digest(manifest), product)
    reader = _DERIVED.get(derived)
    if reader is None:
        reader = _DERIVED[derived] = _derive_reader(store, manifest, product)
    return reader


def _derive_reader(store: CheckpointStore, manifest: Mapping[str, Any], product: str) -> Reader:
    bound = bound_inputs(store, manifest)
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
        _header(store.contig(c), EXCLUSION, identity)
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


class RequestAdapter:
    """Independently checked decision request; schema 2, no scientific reconstruction."""

    schema = 2

    def encode(self, request: ExclusionRequest) -> Mapping[str, Any]:
        return {"schema": self.schema, **to_payload(request)}

    def decode(self, payload: Mapping[str, Any]) -> ExclusionRequest:
        require_schema(payload, self.schema)
        return from_payload(ExclusionRequest, {k: v for k, v in payload.items() if k != "schema"})
