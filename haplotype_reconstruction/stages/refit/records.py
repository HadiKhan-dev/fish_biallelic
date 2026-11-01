"""Store schema-2 founder refits and reconstruct panels from painting views.

The envelope contains the versioned D1 product. Derived panels never enter storage;
reads check the caller's expected derivation, never a physical record identifier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ...contracts.painting import Painting
from ...contracts.payload import from_payload, to_payload
from ...contracts.refit import FounderRefit
from ...contracts.roles import require_schema
from ...core.errors import InputError, InvariantError
from ...core.ties import TieSettings
from ...science.refit.parameters import RefitParameters
from ...science.refit.run import RefitSummary
from ...store.container import CheckpointFile
from ...store.protocol import NumericalPolicy, bound_payload_identity, publish, read_items, resume
from ...store.store import CheckpointStore

__all__ = ["FOUNDER_REFIT", "SUMMARY", "IDENTITY", "SCHEMA", "Adapter", "write", "load"]
FOUNDER_REFIT = "founder_refit"
SUMMARY = "founder_refit/summary"
IDENTITY = "identity/founder_refit"
SCHEMA = 2


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Report durable publication from the container index without decoding arrays.

    This is work status only; the result reader checks schema and compatibility
    with the independently bound request before exposing scientific products.
    """
    info = file.info(FOUNDER_REFIT)
    return dict(
        stage=FOUNDER_REFIT,
        complete=info is not None,
        **({} if info is None else dict(decoded_bytes=info.decoded_bytes)),
    )


class Adapter:
    schema = SCHEMA

    def encode(self, product: FounderRefit) -> dict[str, Any]:
        return {"schema": SCHEMA, "refit": to_payload(product)}

    def decode(self, payload: Mapping[str, Any], *, painting: Painting | None = None) -> FounderRefit:
        require_schema(payload, SCHEMA)
        if set(payload) != {"schema", "refit"}:
            raise InvariantError("unexpected founder-refit envelope fields")
        result = from_payload(FounderRefit, payload["refit"])
        if painting is None:
            return result
        if result.contig != painting.contig or result.sample_ids != painting.sample_ids:
            raise InvariantError("refit and painting contig/sample axes differ")
        if len(result.components) != len(painting.components):
            raise InvariantError("refit and painting component counts differ")
        components = []
        for value, painted in zip(result.components, painting.components, strict=True):
            if painted.panel is None:
                raise InvariantError("refit reconstruction requires painted panels")
            if value.released.shape != painted.panel.released.shape:
                raise InvariantError("refit and painted panel axes differ")
            components.append(
                replace(value, panel=replace(painted.panel, released=value.released, inference=value.inference))
            )
        return replace(result, components=tuple(components))


def write(file: CheckpointFile, identity: str, product: FounderRefit, summary: RefitSummary) -> None:
    publish(file, SUMMARY, identity, {"schema": SCHEMA, **to_payload(summary)})
    publish(file, IDENTITY, identity, {"schema": SCHEMA, "identity": identity})
    publish(file, FOUNDER_REFIT, identity, Adapter().encode(product))


def load(file: CheckpointFile, identity: str, *, painting: Painting | None = None) -> FounderRefit | None:
    record = resume(file, FOUNDER_REFIT, identity)
    return None if record is None else Adapter().decode(record.payload, painting=painting)


@dataclass(frozen=True)
class Reader:
    """Manifest-derived request, with independently checked painted panels."""

    identity: str
    product: str
    painting: Painting | None = None
    key: str = FOUNDER_REFIT
    schema: int = SCHEMA

    def decode(self, payload: Mapping[str, Any]) -> FounderRefit | RefitSummary | tuple[()]:
        if self.product == "tie_sources":
            # RF1 has no shared tie census (its carrier-specific relation is a local rule).
            require_schema(payload, SCHEMA)
            return ()
        if self.product == "summary":
            require_schema(payload, SCHEMA)
            return from_payload(RefitSummary, {key: value for key, value in payload.items() if key != "schema"})
        return Adapter().decode(payload, painting=self.painting)


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    panels: bool = True,
) -> Reader | None:
    """Derive the refit request from manifest settings and checked upstream products."""
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records
    from .tasks import identity

    if contig is None or contig not in manifest["contigs"]:
        raise InputError("refit reader requires a bound contig")
    if product not in ("refit", "summary", "tie_sources"):
        raise InputError(f"unsupported refit reader product {product!r}")
    file = store.contig(contig)
    if not file.has(FOUNDER_REFIT):
        return None
    # The painting payload is decoded only to rebuild refit panels; otherwise its header is checked.
    rebuild = panels and product == "refit"
    painted = painting_records.result_reader(
        store, manifest, product="painting", contig=contig, panels=rebuild, emissions=False
    )
    evidence = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
    if painted is None or evidence is None:
        raise InvariantError("refit requires published painting and evidence")
    if painted.items is None:
        raise InvariantError("painting product reader requires explicit fields")
    painting_record = read_items(file, painted.key, painted.identity, painted.items if rebuild else ("schema",))
    checked_evidence = read_items(file, evidence.key, evidence.identity, ("schema",))
    if painting_record is None or checked_evidence is None:
        raise InvariantError("refit upstream publication is incomplete")
    painting_records.require_evidence_binding(file, painted.identity, evidence.identity)
    parameters = from_payload(RefitParameters, manifest["parameters"]["founder_refit"])
    settings = manifest["ties"]
    expected = identity(
        parameters,
        contig=contig,
        sample_ids=tuple(manifest["sample_ids"]),
        painting_identity=painted.identity,
        evidence_identity=evidence.identity,
        payload_identity=bound_payload_identity(file),
        ties=TieSettings(reverse=bool(settings["reverse"]), keep=tuple(settings.get("keep", ()))),
        numerical_policy=NumericalPolicy(**manifest["numerical_policy"]),
    )
    product_painting: Painting | None = None
    if rebuild:
        decoded = painted.decode(painting_record.payload)
        if not isinstance(decoded, Painting):
            raise InvariantError("painting reader returned the wrong product")
        product_painting = decoded
    return Reader(expected, product, product_painting, key=SUMMARY if product == "summary" else FOUNDER_REFIT)


__all__ += ["Reader", "result_reader"]


__all__ += ["progress"]
