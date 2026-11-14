"""Bind and publish the separate relatedness/families record.

The derivation excludes output selection. It includes the science/task closure,
evidence adapter, codecs, explicit numerical policy and bound sample/contig orders.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...contracts.payload import from_payload, to_payload
from ...contracts.relatedness import RelatednessFamilies
from ...core.errors import InputError
from ...science.relatedness.parameters import AlleleFrequencyParameters, FamilyParameters, RateParameters
from ...store.container import CheckpointFile, RecordInfo
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy
from ...store.store import CheckpointStore
from ..framework import load_product, publish_product
from .evidence_pass import EvidenceInput, bound_upstream, upstream_identities

__all__ = ["KEY", "SCHEMA", "CODE_ROOTS", "Adapter", "identity", "write", "load", "Reader", "expected", "result_reader"]
KEY = "relatedness/families"
SCHEMA = 2
CODE_ROOTS = (
    "stages.relatedness.families_task",
    "science.relatedness.rate",
    "stages.relatedness.families_records",
    "stages.evidence.records",
    "contracts.payload",
    "store.codec",
)


class Adapter:
    """Versioned plain-payload adapter; decoded arrays are read-only."""

    schema = SCHEMA

    def encode(self, product: RelatednessFamilies) -> Mapping[str, Any]:
        return to_payload(product)

    def decode(self, payload: Mapping[str, Any]) -> RelatednessFamilies:
        return from_payload(RelatednessFamilies, payload)


_ADAPTER = Adapter()
_DEFAULT_RATE = RateParameters()


def identity(
    inputs: tuple[EvidenceInput, ...],
    sample_ids: tuple[str, ...],
    parameters: FamilyParameters,
    frequency: AlleleFrequencyParameters,
    rate: RateParameters = _DEFAULT_RATE,
    *,
    numerical_policy: NumericalPolicy | None = None,
    reverse: bool = False,
) -> str:
    """The deterministic scientific derivation; contigs retain their supplied order."""
    return _bound_identity(
        tuple(item.contig for item in inputs),
        sample_ids,
        parameters,
        frequency,
        rate,
        upstream_identities(inputs),
        numerical_policy=numerical_policy,
        reverse=reverse,
    )


def _bound_identity(
    contigs: tuple[str, ...],
    sample_ids: tuple[str, ...],
    parameters: FamilyParameters,
    frequency: AlleleFrequencyParameters,
    rate: RateParameters,
    upstream: Mapping[str, str],
    *,
    numerical_policy: NumericalPolicy | None = None,
    reverse: bool = False,
) -> str:
    return Derivation(
        code=code_dependency_identity(CODE_ROOTS),
        parameters=dict(
            families=parameters,
            frequency=frequency,
            rate=rate,
            sample_ids=sample_ids,
            contigs=contigs,
        ),
        numerical_policy=numerical_policy or NumericalPolicy.current(),
        inputs=upstream,
        tie_reverse=reverse,
        schema=SCHEMA,
    ).identity


def write(file: CheckpointFile, product: RelatednessFamilies, derivation: str) -> RecordInfo:
    return publish_product(file, KEY, derivation, product, _ADAPTER)


def load(file: CheckpointFile, derivation: str) -> RelatednessFamilies | None:
    return load_product(file, KEY, derivation, _ADAPTER)


@dataclass(frozen=True)
class Reader(Adapter):
    """Structural Results.ProductReader; expected identity is rebuilt from bound inputs."""

    identity: str
    key: str = KEY


def expected(
    store: CheckpointStore, manifest: Mapping[str, Any], *, product: str, contig: str | None = None, upstream: Mapping[str, str] | None = None
) -> Reader:
    """Rebuild this run record's derivation without opening original evidence or map files."""
    if product != "families" or contig is not None:
        raise InputError("invalid relatedness reader request")
    contigs = tuple(manifest["contigs"])
    upstream = bound_upstream(store, manifest) if upstream is None else upstream
    parameters = FamilyParameters(**manifest["parameters"]["relatedness"])
    # A run manifest records only configurable stage parameters; allele-frequency estimation is not configurable,
    # so a run-level request rebuilds the derivation with its defaults (task manifests record them explicitly).
    frequency = AlleleFrequencyParameters(**manifest["parameters"].get("allele_frequency", {}))
    # The rate estimator's constants are not configurable either (rl20).
    rate = from_payload(RateParameters, manifest["parameters"].get("rate", to_payload(RateParameters())))
    reverse = bool(manifest["ties"]["reverse"])
    return Reader(
        _bound_identity(
            contigs,
            tuple(manifest["sample_ids"]),
            parameters,
            frequency,
            rate,
            upstream,
            reverse=reverse,
            numerical_policy=NumericalPolicy(**manifest["numerical_policy"]),
        )
    )


def result_reader(store, manifest, *, product, contig=None):
    """Request the expected publication when its commit record exists."""
    if not store.run.has(KEY):
        return None
    return expected(store, manifest, product=product, contig=contig)
