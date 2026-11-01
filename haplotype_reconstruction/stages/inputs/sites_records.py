"""Decode ReleasedSites and derive checked read requests without running a stage.

Owns the schema-one sites codec and expected derivation calculation. Publication
and source verification remain in sites; results may depend only on this adapter.
Method: docs/methods/export.md#sites.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...core.errors import InvariantError

if TYPE_CHECKING:
    from ...contracts.release import ReleasedSites
    from ...store.store import CheckpointStore


@dataclass(frozen=True)
class SitesAdapter:
    """Schema-one storage codec for the option-free ReleasedSites publication."""

    schema: int = 1

    def encode(self, record: ReleasedSites) -> Mapping[str, object]:
        from ...contracts.payload import to_payload

        return to_payload(record)

    def decode(self, payload: Mapping[str, object]) -> ReleasedSites:
        from ...contracts.payload import from_payload
        from ...contracts.release import ReleasedSites
        from ...contracts.roles import require_schema

        require_schema(payload, self.schema)
        return from_payload(ReleasedSites, payload)


def identity(evidence_identity: str, payload_identity: str, *, synthetic: bool = False) -> str:
    """Bind upstream evidence, consumed observations and source/stage code, without export options."""
    from ...store.identity import code_dependency_identity
    from ...store.protocol import Derivation, NumericalPolicy

    roots = ("stages.inputs.sites", "stages.inputs.sources", "stages.inputs.vcf_alleles")
    return Derivation(
        code=code_dependency_identity(roots),
        parameters={"synthetic": synthetic},
        numerical_policy=NumericalPolicy.current(),
        inputs={"evidence": evidence_identity, "payload": payload_identity},
        schema=1,
    ).identity


@dataclass(frozen=True)
class SitesReader:
    """Results read request computed from verified upstream derivations, never the sites envelope."""

    identity: str
    key: str = "sites"
    schema: int = 1

    def decode(self, payload: Mapping[str, object]) -> ReleasedSites:
        return SitesAdapter().decode(payload)


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str,
    evidence_identity: str | None = None,
    payload_identity: str | None = None,
) -> SitesReader:
    """Recompute the sites request from evidence and the immutable source binding.

    Composition may pass checked upstream identities explicitly. Otherwise EV3's
    result_reader supplies its expected evidence derivation; the input binding is
    the immutable source record made at evidence construction, never this product.
    """
    if product != "released_sites":
        raise InvariantError(f"unsupported sites reader product {product!r}")
    if evidence_identity is None:
        from ..evidence import records

        request = records.result_reader(store, manifest, product="evidence", contig=contig)
        if request is None:
            raise InvariantError(f"{contig}: released sites require published evidence")
        evidence_identity = request.identity
    if payload_identity is None:
        from ...store.protocol import bound_payload_identity

        payload_identity = bound_payload_identity(store.contig(contig))
    source = manifest.get("source", {})
    synthetic = source.get("kind") == "simulation"
    return SitesReader(identity(evidence_identity, payload_identity, synthetic=synthetic))
