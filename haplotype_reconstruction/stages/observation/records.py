"""Derivations and readers of the run's observation pilot (docs/methods/observation.md; ADVICE2 sections 1 and 8, with
its notes of 16 November 2026 08:45 and 09:12 on the rate):

    evidence(c) for every analysed c -> fits(c) -> observation

per analysed contig, the strict fits of its pilot blocks (discovery's block search at the strict outlier rate: no fit
tolerates the discordance it measures), then the run's outlier rate, the one-rate maximum likelihood over every
known-state cell of the strict fits. Discovery's evidence is read at face value (no evidence scale: the held-out scale
search of option (b) was removed by the owner's decision of 16 November 2026).
Every consumer (relatedness, discovery, local selection, assembly) binds the product's identity as an upstream input
and reads its rate through ``outlier_rate_blocks``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...contracts.contigs import AnalysisContigs
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.errors import InvariantError
from ...science.discovery.parameters import DiscoveryParameters
from ...science.observation.parameters import ObservationParameters
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, read_items, resume

OBSERVATION = "observation"
SCHEMA = 5
CODE_ROOTS = (
    "science.observation.run",
    "science.observation.estimate",
    "stages.observation.tasks",
    "stages.observation.records",
)


def key(product: str) -> str:
    """The record key of a contig's pilot fits or of the run's observation product."""
    if product == OBSERVATION:
        return OBSERVATION
    if product != "fits":
        raise InvariantError(f"unknown observation record: {product}")
    return "observation/fits"


@dataclass(frozen=True)
class Reader:
    identity: str
    key: str
    schema: int = SCHEMA
    items: tuple | None = None

    def decode(self, payload):
        require_schema(payload, self.schema)
        return dict(payload)


def parameters(manifest: Mapping[str, Any]) -> tuple[DiscoveryParameters, ObservationParameters]:
    """The discovery model the pilot fits and the pilot's own settings (not configurable)."""
    return from_payload(DiscoveryParameters, manifest["parameters"]["discovery"]), ObservationParameters()


def _identities(store, manifest) -> dict:
    """Every pilot derivation of the run: (product, contig) -> identity."""
    from ..evidence import records as evidence_records

    contigs = tuple(AnalysisContigs.from_manifest(manifest).included)
    inputs = {}
    for c in contigs:
        request = evidence_records.expected(store, manifest, product="evidence", contig=c)
        if request is None:
            raise InvariantError("the observation pilot requires every analysed contig's evidence")
        inputs[c + "/evidence"] = request.identity
    discovery, observation = parameters(manifest)
    policy = NumericalPolicy(**manifest["numerical_policy"])
    code = code_dependency_identity(CODE_ROOTS)

    def identity(which, scope=None, upstream=None):
        return Derivation(
            code,
            dict(
                product=which, contig=scope, contigs=contigs, discovery=to_payload(discovery),
                observation=to_payload(observation), schema=SCHEMA,
            ),
            policy,
            dict(inputs, **(upstream or {})),
            tie_reverse=manifest["ties"]["reverse"],
            tie_keep=tuple(manifest["ties"]["keep"]),
        ).identity

    out = {("fits", c): identity("fits", c) for c in contigs}
    out[(OBSERVATION, None)] = identity(OBSERVATION, upstream={f"{c}/fits": out[("fits", c)] for c in contigs})
    return out


_CACHE: dict[tuple, dict] = {}


def identities(store, manifest) -> dict:
    """_identities, cached per process (the analysed contigs' evidence bindings of a run root are immutable)."""
    from ...core.digest import json_digest, jsonable

    cache_key = (store.root, json_digest(jsonable(dict(manifest))))
    found = _CACHE.get(cache_key)
    if found is None:
        found = _CACHE[cache_key] = _identities(store, manifest)
    return found


def expected(store, manifest, *, product: str = OBSERVATION, contig: str | None = None):
    """The derivation and key of a contig's pilot fits or of the observation product."""
    found = identities(store, manifest).get((product, contig))
    if found is None:
        raise InvariantError(f"no observation derivation for {product}/{contig}")
    return Reader(found, key(product))


def bound_identity(store, manifest) -> str:
    """The observation product's derivation: the upstream input every consumer binds."""
    return expected(store, manifest).identity


def load(store, manifest) -> dict:
    """The run's published observation product; raises when missing."""
    request = expected(store, manifest)
    saved = resume(store.run, request.key, request.identity)
    if saved is None:
        raise InvariantError(f"the run's observation record {request.key} is not published")
    return request.decode(saved.payload)


def result_reader(store, manifest, *, product: str = OBSERVATION, contig: str | None = None):
    file = store.run if contig is None else store.contig(contig)
    request = expected(store, manifest, product=product, contig=contig)
    if not file.has(request.key):
        return None
    return request if read_items(file, request.key, request.identity, ("schema",)) is not None else None


def outlier_rate_blocks(reader, store) -> float:
    """The run's genotype outlier rate before the calibration barrier (``reader``: anything with ``.manifest``)."""
    return float(load(store, reader.manifest)["outlier_rate_blocks"])
