"""Convert a neutral phase seam payload into frozen scientific inputs.

Owns representation conversion and the ParentEdges identity, not model choices.
The canonical shared map evaluates recorded knots at painting bin centres;
the shared Haldane formula belongs in science.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.painting import Painting
from ...contracts.pedigree import ParentEdges
from ...contracts.phase import (
    AgreedRelease,
    AlleleUncertainty,
    MeiosisChain,
    PhaseBins,
    PhaseComponentInputs,
    PhaseComponentResult,
    PhasedSamples,
    PhaseInputIdentities,
    PhaseMap,
    PhaseObservations,
    PhasePainting,
    PhaseRefit,
    PhaseSolve,
    PhaseStates,
    PhaseSummary,
    PhaseTransitions,
    SampleRelease,
)
from ...contracts.refit import FounderRefit
from ...science.phase.parameters import PhaseParameters
from ...store.container import CheckpointFile
from ...store.identity import CodeIdentity
from ...store.protocol import Derivation, NumericalPolicy
from ...store.store import CheckpointStore
from .boundaries import chromosome_prior_counts, validate_parent_edges, validate_sample_order

__all__ = ["component_inputs", "canonical_component_inputs", "ComponentResultAdapter"]


def _map_distances(record: Mapping[str, Any], centers: np.ndarray) -> np.ndarray:
    """Use the canonical shared map, including its exact fallback and tail rules."""
    genetic_map = ChromosomeGeneticMap(**record)
    return np.asarray(genetic_map.interval_morgans(centers[:-1], centers[1:]), dtype=np.float64)


def component_inputs(payload: Mapping[str, Any]) -> tuple[PhaseComponentInputs, PhaseParameters]:
    """Borrow fixture arrays and construct typed inputs plus parameters; mutate nothing.

    The payload is the normalized 3.5 component seam, with local evidence sites,
    absolute provenance, painting/refit records and all contig prior inference
    arrays. The exact integer sufficient statistics replace those prior arrays.
    Missing required fields raise KeyError at this adapter boundary.
    """
    painting = payload["painting"]
    sample_ids = tuple(payload["sample_ids"])
    observations = PhaseObservations(
        sample_ids,
        payload["positions"],
        payload["absolute_site_indices"],
        payload["likelihoods"],
        payload["observed"],
        payload["excluded"],
    )
    bins = PhaseBins(painting["bin_starts"], painting["bin_stops"], painting["bin_centers"], painting["bin_edges"])
    states = PhaseStates(
        painting["positions"],
        painting["named_alleles"],
        tuple(tuple(rows) for rows in painting["classes"]),
        painting["background_frequency"],
        len(painting["classes"]) + 1,
    )
    transitions = PhaseTransitions(painting["log_same"], painting["site_correlation"])
    painted = PhasePainting(
        states,
        bins,
        transitions,
        painting["local_keys"],
        painting["eligible"],
        painting["labels"],
        painting["track_classes"],
        painting["status"],
        painting["member_ids"],
        tuple(tuple(rows) for rows in painting["member_sets"]),
        payload["packed_emissions"],
        payload["painting_config"]["robustness_epsilon"],
        payload["painting_config"]["emission_log_floor"],
        sample_ids=tuple(painting.get("sample_ids", sample_ids)),
    )
    refit = PhaseRefit(
        payload["refit"]["positions"],
        payload["refit"]["inference"],
        *chromosome_prior_counts(payload["prior_inference"]),
    )
    edges = payload["edges"]
    pedigree = ParentEdges(
        tuple(edges.get("sample_ids", sample_ids)),
        edges["parents"],
        edges["children"],
        edges["slots"],
        np.bincount(edges["children"], minlength=len(sample_ids)).astype(np.int64),
        np.full(len(sample_ids), 2, dtype=np.int64),
    )
    validate_sample_order(
        sample_ids,
        painting=painted.sample_ids,
        pedigree=pedigree.sample_ids,
        refit=tuple(payload["refit"].get("sample_ids", sample_ids)),
    )
    validate_parent_edges(pedigree)
    config = PhaseParameters(**payload["config"])
    genetic_map = PhaseMap(
        _map_distances(payload["genetic_map"], bins.centers), tuple(config.protected_for(payload["contig"]))
    )
    return PhaseComponentInputs(
        payload["contig"], payload["component"], observations, painted, refit, pedigree, genetic_map
    ), config


class ComponentResultAdapter:
    """Encode/decode the component result through the foundation's RecordAdapter protocol.

    Axes, labels and integer encodings are preserved exactly. Only frozen records
    become named dictionaries; no field is omitted and unknown fields raise.
    """

    schema = 1

    def encode(self, product: PhaseComponentResult) -> dict[str, Any]:
        """Return the versioned codec payload; arrays are borrowed and not mutated."""
        from dataclasses import fields

        def values(record: Any) -> dict[str, Any]:
            return {field.name: getattr(record, field.name) for field in fields(record)}

        return {
            "schema": self.schema,
            "solves": [values(solve) for solve in product.solves],
            "site_indices": product.site_indices,
            "site_bins": product.site_bins,
            "site_alleles": product.site_alleles,
            "uncertainty": [values(value) for value in product.uncertainty],
            "releases": [[values(value) for value in release] for release in product.releases],
            "agreed": [values(value) for value in product.agreed],
            "chains": [values(value) for value in product.chains],
        }

    def decode(self, payload: Mapping[str, Any]) -> PhaseComponentResult:
        """Reconstruct the frozen component result; raise on an unsupported schema or field."""
        from ...contracts.roles import require_schema

        require_schema(payload, self.schema)
        record = dict(payload)
        record.pop("schema")
        record["solves"] = tuple(PhaseSolve(**value) for value in record["solves"])
        record["uncertainty"] = tuple(AlleleUncertainty(**value) for value in record["uncertainty"])
        record["releases"] = tuple(tuple(SampleRelease(**value) for value in release) for release in record["releases"])
        record["agreed"] = tuple(AgreedRelease(**value) for value in record["agreed"])
        record["chains"] = tuple(MeiosisChain(**value) for value in record["chains"])
        return PhaseComponentResult(**record)


def validate_canonical_inputs(
    evidence: ContigEvidence,
    painting: Painting,
    refit: FounderRefit,
    edges: ParentEdges,
    genetic_map: ChromosomeGeneticMap,
    identities: PhaseInputIdentities,
) -> None:
    """Validate independently supplied axes and identities before any resume or compute."""
    from dataclasses import fields

    from ...core.digest import json_digest
    from ...core.errors import InvariantError

    sample_ids = tuple(evidence.sample_ids)
    validate_sample_order(sample_ids, painting=painting.sample_ids, refit=refit.sample_ids, pedigree=edges.sample_ids)
    validate_parent_edges(edges)
    if painting.contig != evidence.contig or refit.contig != evidence.contig or genetic_map.contig != evidence.contig:
        raise InvariantError("phase upstream records refer to different contigs")
    if len(painting.components) != len(refit.components):
        raise InvariantError("phase painting/refit component counts differ")
    if any(not getattr(identities, f.name) for f in fields(identities)):
        raise InvariantError("phase requires every upstream derivation identity")
    if identities.pedigree != edges.identity or identities.genetic_map != json_digest(genetic_map.identity()):
        raise InvariantError("phase pedigree/map identity differs from its canonical input")
    n, length = len(sample_ids), len(evidence.positions)
    if evidence.likelihoods.shape != (n, length, 3) or evidence.observed.shape != (n, length):
        raise InvariantError("phase evidence arrays disagree with the sample/site axes")
    for painted, fitted in zip(painting.components, refit.components, strict=True):
        if (
            painted.panel is None
            or fitted.panel is None
            or not np.array_equal(painted.panel.positions, fitted.panel.positions)
        ):
            raise InvariantError("phase painting/refit component positions or order differ")
        if painted.panel.inference.shape != fitted.panel.inference.shape:
            raise InvariantError("phase painting/refit component row identities differ")


def canonical_component_inputs(
    evidence: ContigEvidence,
    painting: Painting,
    refit: FounderRefit,
    edges: ParentEdges,
    genetic_map: ChromosomeGeneticMap,
    parameters: PhaseParameters,
    *,
    component: int,
    identities: PhaseInputIdentities,
) -> PhaseComponentInputs:
    """Project identity-checked canonical records onto one component, borrowing arrays.

    Read each record's independent sample axis. Painting/refit must retain their component
    panels (the canonical derived views), with identical row and site identities.
    Prior counts include every refit component, even ones without painting bins.
    """
    from ...core.errors import InvariantError

    validate_canonical_inputs(evidence, painting, refit, edges, genetic_map, identities)
    sample_ids = tuple(evidence.sample_ids)
    painted = painting.components[component]
    panel = refit.components[component].panel
    if panel is None:
        raise InvariantError("phase refit component has no reconstructed panel")
    source = panel.positions
    if not len(source):
        raise InvariantError("phase component panel must have at least one site")
    left = int(np.searchsorted(evidence.positions, source[0], side="left"))
    right = int(np.searchsorted(evidence.positions, source[-1], side="right"))
    states, bins = painted.states, painted.bins
    observations = PhaseObservations(
        sample_ids,
        evidence.positions[left:right],
        np.arange(left, right, dtype=np.int64),
        evidence.likelihoods[:, left:right],
        evidence.observed[:, left:right],
        evidence.founder_excluded,
    )
    phase_painting = PhasePainting(
        PhaseStates(
            states.positions, states.named_alleles, states.classes, states.background_frequency, states.n_states
        ),
        PhaseBins(bins.starts, bins.stops, bins.centers, bins.edges),
        PhaseTransitions(painted.transitions.log_same, painted.transitions.site_correlation),
        painted.local_keys,
        painted.eligible,
        painted.labels,
        painted.classes,
        painted.status,
        painted.member_ids,
        painted.member_sets,
        painted.packed_emissions,
        painting.parameters.robustness_epsilon,
        painting.parameters.emission_log_floor,
        sample_ids=sample_ids,
    )
    fitted = PhaseRefit(source, panel.inference, *chromosome_prior_counts([c.inference for c in refit.components]))
    phase_map = PhaseMap(
        np.asarray(genetic_map.interval_morgans(bins.centers[:-1], bins.centers[1:]), dtype=np.float64),
        tuple(parameters.protected_for(evidence.contig)),
    )
    return PhaseComponentInputs(
        evidence.contig,
        component,
        observations,
        phase_painting,
        fitted,
        edges,
        phase_map,
        identities=identities,
    )


class PhasedSamplesAdapter:
    """Encode the complete schema-1 contig record with the shared frozen-record codec."""

    schema = 1

    def encode(self, product: PhasedSamples) -> dict[str, Any]:
        from ...contracts.payload import to_payload

        return to_payload(product)

    def decode(self, payload: Mapping[str, Any]) -> PhasedSamples:
        from ...contracts.payload import from_payload
        from ...contracts.phase import PhasedSamples
        from ...contracts.roles import require_schema

        require_schema(payload, self.schema)
        return from_payload(PhasedSamples, payload)


KEY = "phase"
BINDING_KEY = "identity/phase"
CODE_ROOTS = ("haplotype_reconstruction.stages.phase.tasks",)


def derivation(
    contig: str,
    sample_ids: tuple[str, ...],
    identities: PhaseInputIdentities,
    parameters: PhaseParameters,
    *,
    reverse: bool = False,
    code: CodeIdentity | None = None,
    policy: NumericalPolicy | None = None,
) -> Derivation:
    """Bind both solves and release to canonical upstream identities; buffer routes are excluded."""
    from dataclasses import asdict

    from ...core.errors import InvariantError
    from ...store.identity import code_dependency_identity
    from ...store.protocol import Derivation, NumericalPolicy

    if any(not value for value in asdict(identities).values()):
        raise InvariantError("phase requires every upstream derivation identity")
    return Derivation(
        code_dependency_identity(CODE_ROOTS) if code is None else code,
        dict(contig=contig, sample_ids=tuple(sample_ids), phase=parameters),
        NumericalPolicy.current() if policy is None else policy,
        asdict(identities),
        tie_reverse=bool(reverse),
    )


def bind(file: CheckpointFile, expected: Derivation) -> str:
    """Persist the phase binding before consulting completion or doing scientific work."""
    from ...store.protocol import publish

    publish(file, BINDING_KEY, expected.identity, dict(schema=1, stage=KEY, derivation=expected.identity))
    return expected.identity


def load(file: CheckpointFile, expected: Derivation) -> PhasedSamples | None:
    """Resume only a complete record with the expected externally derived identity."""
    from ...store.protocol import resume

    saved = resume(file, KEY, bind(file, expected))
    return None if saved is None else PhasedSamplesAdapter().decode(saved.payload)


def write(file: CheckpointFile, expected: Derivation, product: PhasedSamples) -> None:
    """Publish a complete contig product with its immutable derivation binding."""
    from ...contracts.roles import freeze_record_arrays
    from ...store.protocol import publish

    freeze_record_arrays(product)
    publish(file, KEY, bind(file, expected), PhasedSamplesAdapter().encode(product))


@dataclass(frozen=True)
class Reader:
    """Identity checked schema-1 product or diagnostics projection for Results."""

    identity: str
    product: str = "phase"
    elapsed_seconds: float | None = None
    key: str = KEY
    schema: int = 1

    @property
    def items(self) -> tuple[str, ...]:
        from dataclasses import fields

        from ...contracts.phase import PhasedSamples

        return (
            ("schema", "contig", "sample_ids", "positions", "bin_edge_offsets", "diagnostics")
            if self.product == "phase_summary"
            else tuple(f.name for f in fields(PhasedSamples))
        )

    def decode(self, payload: Mapping[str, Any]) -> PhasedSamples | PhaseSummary:
        from ...contracts.payload import from_payload
        from ...contracts.phase import PhaseDiagnostics
        from ...contracts.roles import require_schema

        require_schema(payload, self.schema)
        if self.product == "phase_summary":
            from ...contracts.phase import PhaseSummary

            d = from_payload(PhaseDiagnostics, payload["diagnostics"])
            return PhaseSummary(
                payload["contig"],
                len(payload["sample_ids"]),
                len(payload["positions"]),
                len(payload["bin_edge_offsets"]) - 1,
                int(sum(d.history_steps)),
                d.converged,
                float(sum(o[-1] for o in d.objective if o)),
                int(sum(d.junction_candidates)),
                d.withheld_joins,
                d.tie_breaks,
                d.called_alleles,
                d.phased_heterozygotes,
                d.unphased_heterozygotes,
                d.phase_sets,
                self.elapsed_seconds,
            )
        return PhasedSamplesAdapter().decode(payload)


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    elapsed_seconds: float | None = None,
) -> Reader | None:
    """Derive phase identity from checked upstream products, never the phase envelope itself."""
    from ...contracts.payload import from_payload
    from ...contracts.roles import require_schema
    from ...core.errors import InputError, InvariantError
    from ...store.protocol import NumericalPolicy, read_items
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records
    from ..pedigree import records as pedigree_records
    from ..refit import records as refit_records

    if product not in ("phase", "phase_summary") or contig is None or contig not in manifest["contigs"]:
        raise InputError("phase reader requires a declared product and bound contig")
    file = store.contig(contig)
    if not file.has(KEY) and not file.has(BINDING_KEY):
        return None
    requests = (
        evidence_records.result_reader(store, manifest, product="evidence", contig=contig),
        painting_records.result_reader(
            store, manifest, product="painting", contig=contig, panels=False, emissions=False
        ),
        refit_records.result_reader(store, manifest, product="refit", contig=contig, panels=False),
    )
    evidence_request, painting_request, refit_request = requests
    if evidence_request is None or painting_request is None or refit_request is None:
        raise InvariantError("phase upstream publication is absent")
    for request in (evidence_request, painting_request, refit_request):
        if read_items(file, request.key, request.identity, ("schema",)) is None:
            raise InvariantError("phase upstream publication is incomplete")
    pedigree_request = pedigree_records.result_reader(store, manifest, product="pedigree")
    if pedigree_request is None:
        raise InvariantError("phase pedigree publication is absent")
    # Only the edges enter phase's identity: decode that one field of the identity-checked pedigree publication
    # (its header identity is checked before any payload is decompressed) and derive the edge-content identity
    # from the decoded edges themselves, never from an envelope.
    saved = read_items(store.run, pedigree_request.key, pedigree_request.identity, ("schema", "edges"))
    if saved is None or "edges" not in saved.payload:
        raise InvariantError("phase pedigree publication is incomplete")
    require_schema(saved.payload, pedigree_request.schema)
    edges = from_payload(ParentEdges, saved.payload["edges"])
    if not isinstance(edges, ParentEdges):
        raise InvariantError("phase upstream reader did not return the parent-edges contract")
    identities = PhaseInputIdentities(
        evidence_request.identity,
        painting_request.identity,
        refit_request.identity,
        edges.identity,
        manifest["map_identities"][contig],
    )
    parameters = from_payload(PhaseParameters, manifest["parameters"]["family_phase"])
    expected = derivation(
        contig,
        tuple(manifest["sample_ids"]),
        identities,
        parameters,
        reverse=manifest["ties"]["reverse"],
        policy=NumericalPolicy(**manifest["numerical_policy"]),
    )
    return Reader(expected.identity, product, elapsed_seconds)


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Return committed product status from container metadata without decoding arrays."""
    info = file.info(KEY)
    return dict(
        stage="family_phase",
        complete=info is not None,
        **({} if info is None else dict(decoded_bytes=info.decoded_bytes)),
    )
