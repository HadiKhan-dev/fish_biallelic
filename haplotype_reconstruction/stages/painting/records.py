"""Publish painting and its optional packed-emission cache independently.

Reads verify caller-supplied derivations. Four load routes independently select
reconstructed panels and emissions. Raw paths and posteriors are never stored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ...contracts.assembly import AssemblyProduct, JoinSummaryView
from ...contracts.painting import Painting, PaintingSummary, painting_from_payload, painting_to_payload
from ...contracts.panels import FounderComponent, PreparedPanel
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.errors import InputError, InvariantError
from ...core.ties import TieCounts, TieSettings
from ...store.container import CheckpointFile
from ...store.protocol import NumericalPolicy, bound_payload_identity, publish, read_items, resume
from ...store.store import CheckpointStore

__all__ = [
    "PAINTING",
    "SUMMARY",
    "CACHE",
    "DIAGNOSTICS",
    "IDENTITY",
    "SCHEMA",
    "Adapter",
    "write",
    "load",
]
PAINTING = "painting"
SUMMARY = "painting/summary"
CACHE = "painting/cache"
DIAGNOSTICS = "painting/diagnostics"
IDENTITY = "identity/painting"
SCHEMA = 1
Assembly = tuple[tuple[PreparedPanel, ...], tuple[FounderComponent, ...]]


from ...science.painting.parameters import PaintingParameters
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation


def derivation(
    parameters: PaintingParameters,
    *,
    contig: str,
    sample_ids: tuple[str, ...],
    assembly_identity: str,
    evidence_identity: str,
    payload_identity: str,
    map_identity: str,
    ties: TieSettings = TieSettings(),
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind scientific inputs and convention; exclude cache, batches and thread counts."""
    return Derivation(
        code_dependency_identity(("science.painting", "stages.painting.tasks", "stages.painting.records")),
        dict(parameters=parameters, contig=contig, sample_ids=sample_ids),
        numerical_policy or NumericalPolicy.current(),
        dict(assembly_final=assembly_identity, evidence=evidence_identity, payload=payload_identity, map=map_identity),
        tie_reverse=ties.reverse,
        tie_keep=ties.keep,
    ).identity


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Report durable publication from the container index without decoding arrays.

    This is work status only; the result reader checks schema and compatibility
    with the independently bound request before exposing scientific products.
    """
    info = file.info(PAINTING)
    return dict(
        stage=PAINTING,
        complete=info is not None,
        **({} if info is None else dict(decoded_bytes=info.decoded_bytes)),
    )


class Adapter:
    """Product codec; cache buffers have their own publication and lifetime."""

    schema = SCHEMA

    def encode(self, product: Painting) -> dict[str, Any]:
        payload = painting_to_payload(product)
        payload.pop("packed_emissions")
        return payload

    def decode(self, payload: Mapping[str, Any], *, assembly: Assembly | None = None) -> Painting:
        return painting_from_payload(payload, assembly=assembly)


def write(
    file: CheckpointFile,
    identity: str,
    painting: Painting,
    summary: PaintingSummary,
    *,
    evidence_identity: str | None = None,
) -> None:
    """Publish cache and summary first, scientific product last as the commit record."""
    publish(
        file,
        CACHE,
        identity,
        {"schema": SCHEMA, "packed_emissions": [component.packed_emissions for component in painting.components]},
    )
    publish(file, SUMMARY, identity, to_payload(summary))
    publish(file, IDENTITY, identity, {"schema": SCHEMA, "identity": identity, "evidence_identity": evidence_identity})
    publish(file, PAINTING, identity, Adapter().encode(painting))


def load(
    file: CheckpointFile, identity: str, *, assembly: Assembly | None = None, emissions: bool = True
) -> Painting | None:
    """Read a matching product; assembly reconstructs panels, emissions adds the cache."""
    names = ("schema", "contig", "sample_ids", "parameters", "components", "label_continuity")
    record = read_items(file, PAINTING, identity, names + (("rows",) if assembly is not None else ()))
    if record is None:
        return None
    payload = dict(record.payload)
    if emissions:
        cache = resume(file, CACHE, identity)
        if cache is not None:
            require_schema(cache.payload, SCHEMA)
            payload["packed_emissions"] = cache.payload["packed_emissions"]
    return Adapter().decode(payload, assembly=assembly)


@dataclass(frozen=True)
class TieSource:
    """The explicit painting group of the shared diagnostic report protocol."""

    record_group: str
    counters: tuple[TieCounts, ...]
    discovery: tuple[object, ...] = ()
    refinement_agreement: tuple[bool | None, ...] = ()
    count_calls: tuple[object, ...] = ()


@dataclass(frozen=True)
class PaintingReport:
    """Read-only report inputs, including each recorded final/feedback search once."""

    painting: Painting
    positions: np.ndarray
    observed: np.ndarray
    searches: tuple[tuple[int, int, int, JoinSummaryView], ...]
    feedback_searches: tuple[tuple[int, int, int, JoinSummaryView], ...]


@dataclass(frozen=True)
class Reader:
    """Results request with identity derived independently from the bound manifest."""

    identity: str
    product: str
    assembly: Assembly | None = None
    packed_emissions: tuple[np.ndarray | None, ...] | None = None
    positions: np.ndarray | None = None
    observed: np.ndarray | None = None
    searches: tuple[tuple[int, int, int, JoinSummaryView], ...] = ()
    feedback_searches: tuple[tuple[int, int, int, JoinSummaryView], ...] = ()
    key: str = PAINTING
    schema: int = SCHEMA

    @property
    def items(self) -> tuple[str, ...] | None:
        if self.product == "summary":
            return None
        names = ("schema", "contig", "sample_ids", "parameters", "components", "label_continuity")
        return names + (("rows",) if self.assembly is not None else ())

    def decode(self, payload: Mapping[str, Any]) -> Painting | PaintingSummary | PaintingReport | tuple[TieSource, ...]:
        if self.product == "summary":
            return from_payload(PaintingSummary, payload)
        data = dict(payload)
        if self.packed_emissions is not None:
            data["packed_emissions"] = self.packed_emissions
        painting = painting_from_payload(data, assembly=self.assembly)
        if self.product == "painting_report":
            if self.positions is None or self.observed is None:
                raise InvariantError("painting report requires checked evidence arrays")
            return PaintingReport(painting, self.positions, self.observed, self.searches, self.feedback_searches)
        if self.product == "tie_sources":
            return (TieSource("painting", tuple(outcome.tie_events for outcome in painting.label_continuity)),)
        return painting


def expected(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
) -> Reader | None:
    """Rebuild painting identity from checked assembly/evidence requests and run settings.

    Never read an expected identity from this product's own envelope. Physical
    cache presence is independent of the scientific identity and its schema.
    """
    from ...science.painting.parameters import PaintingParameters
    from ..assembly import records as assembly_records
    from ..evidence import records as evidence_records
    from ..relatedness.working_map import map_identity

    if contig is None or contig not in manifest["contigs"]:
        raise InputError("painting reader requires a bound contig")
    if product not in ("painting", "summary", "tie_sources", "painting_report"):
        raise InputError(f"unsupported painting reader product {product!r}")
    file = store.contig(contig)
    assembled = assembly_records.result_reader(store, manifest, product="assembly", contig=contig)
    evidence = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
    if assembled is None or evidence is None:
        raise InvariantError("painting requires published assembly and evidence")
    # Identity checks read headers only; the assembly payload is decoded only when panels are wanted.
    checked_assembly = read_items(file, assembled.key, assembled.identity, ("schema",))
    checked_evidence = read_items(file, evidence.key, evidence.identity, ("schema",))
    if checked_assembly is None or checked_evidence is None:
        raise InvariantError("painting upstream publication is incomplete")
    parameters = from_payload(PaintingParameters, manifest["parameters"]["painting"])
    settings = manifest["ties"]
    expected_identity = derivation(
        parameters,
        contig=contig,
        sample_ids=tuple(manifest["sample_ids"]),
        assembly_identity=assembled.identity,
        evidence_identity=evidence.identity,
        payload_identity=bound_payload_identity(file),
        map_identity=map_identity(store, manifest, contig),
        ties=TieSettings(reverse=bool(settings["reverse"]), keep=tuple(settings["keep"])),
        numerical_policy=NumericalPolicy(**manifest["numerical_policy"]),
    )
    return Reader(expected_identity, product, key=SUMMARY if product == "summary" else PAINTING)


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    panels: bool = True,
    emissions: bool = True,
) -> Reader | None:
    """Attach requested panel/cache views to the shared expected publication."""
    from ..assembly import records as assembly_records
    from ..evidence import records as evidence_records

    file = store.contig(contig)
    if not file.has(PAINTING):
        return None
    request = expected(store, manifest, product=product, contig=contig)
    expected_identity = request.identity
    if product == "summary":
        return request
    assembled = assembly_records.result_reader(store, manifest, product="assembly", contig=contig)
    evidence = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
    assembly = None
    if (panels and product == "painting") or product == "painting_report":
        assembly_record = resume(file, assembled.key, assembled.identity)
        if assembly_record is None:
            raise InvariantError("painting upstream publication is incomplete")
        value = cast(AssemblyProduct, assembled.decode(assembly_record.payload))
        assembly = (tuple(value.prepared), tuple(value.components))
    packed = None
    if emissions and product == "painting":
        cache = resume(file, CACHE, expected_identity)
        if cache is not None:
            require_schema(cache.payload, SCHEMA)
            packed = tuple(cache.payload["packed_emissions"])
    if product == "painting_report":
        arrays = evidence_records.load_arrays(file, ("positions", "observed"), expected_identity=evidence.identity)
        arrays["observed"].flags.writeable = False
        histories: list[tuple[tuple[int, int, int, JoinSummaryView], ...]] = []
        for round_name in (None, "l1", "l2"):
            request = assembly_records.result_reader(
                store, manifest, product="join_searches", contig=contig, round_name=round_name
            )
            if request is None:
                if round_name is None:
                    raise InvariantError("painting report requires final assembly search history")
                histories.append(())
                continue
            request = cast(assembly_records.JoinSearchReader, request)
            saved = read_items(file, request.key, request.identity, request.items)
            if saved is None:
                raise InvariantError("painting report search history is incomplete")
            histories.append(cast(tuple[tuple[int, int, int, JoinSummaryView], ...], request.decode(saved.payload)))
        return Reader(
            expected_identity,
            product,
            assembly,
            positions=arrays["positions"],
            observed=arrays["observed"],
            searches=histories[0],
            feedback_searches=histories[1] + histories[2],
        )
    return Reader(expected_identity, product, assembly, packed)


__all__ += ["Reader", "TieSource", "PaintingReport", "result_reader"]


def require_evidence_binding(file: CheckpointFile, identity: str, evidence_identity: str) -> None:
    """Reject a painting made from another evidence product before conditional refit."""
    binding = resume(file, IDENTITY, identity)
    if binding is None or binding.payload.get("evidence_identity") != evidence_identity:
        raise InvariantError("painting is not bound to the current evidence identity")


__all__ += ["require_evidence_binding"]


__all__ += ["progress"]
