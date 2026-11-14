"""Publish recombination by derivation identity before consulting completion (RC07).

The binding survives a failed computation and an ordinary checkpoint copy.
A fresh root containing only upstream products has no recombination binding and
may choose its own downstream parameters. Arrays use the shared plain codec.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.payload import from_payload, to_payload
from ...contracts.pedigree import ParentEdges
from ...contracts.recombination import Recombination
from ...contracts.roles import require_schema
from ...core.digest import json_digest
from ...core.errors import InvariantError
from ...science.recombination.parameters import RecombinationParameters
from ...store.container import CheckpointFile
from ...store.identity import CodeIdentity, code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, publish, resume
from ...store.store import CheckpointStore

KEY = "recombination"
BINDING_KEY = "identity/recombination"
CONFIGURATION_KEY = "configuration/recombination"
ANALYSIS_KEYS = (KEY, BINDING_KEY, CONFIGURATION_KEY, "complete/" + KEY)
CODE_ROOTS = ("haplotype_reconstruction.stages.recombination.tasks",)


class RecombinationAdapter:
    """Store schema 2 as named plain fields, preserving every scientific dtype and axis."""

    schema = 2

    def encode(self, product: Recombination) -> dict[str, Any]:
        return to_payload(product)

    def decode(self, payload: Mapping[str, Any]) -> Recombination:
        require_schema(payload, self.schema)
        return from_payload(Recombination, payload)


def derivation(
    contig: str,
    sample_ids: tuple[str, ...],
    phase_identity: str,
    edges: ParentEdges,
    parameters: RecombinationParameters,
    genetic_map: ChromosomeGeneticMap,
    *,
    reverse: bool = False,
    code: CodeIdentity | None = None,
    policy: NumericalPolicy | None = None,
) -> Derivation:
    """Bind the actual phase, pedigree, map, parameters, code closure and convention."""
    if not phase_identity or contig != genetic_map.contig or sample_ids != edges.sample_ids:
        raise InvariantError("recombination upstream identity, sample order or contig is inconsistent")
    return _bound_derivation(
        contig,
        sample_ids,
        parameters,
        dict(phase=phase_identity, pedigree=edges.identity, genetic_map=json_digest(genetic_map.identity())),
        reverse=reverse,
        code=code,
        policy=policy,
    )


def _bound_derivation(
    contig: str,
    sample_ids: tuple[str, ...],
    parameters: RecombinationParameters,
    inputs: Mapping[str, str],
    *,
    reverse: bool,
    code: CodeIdentity | None = None,
    policy: NumericalPolicy | None = None,
) -> Derivation:
    """One recipe over independently checked phase, pedigree and map identities."""
    return Derivation(
        code_dependency_identity(CODE_ROOTS) if code is None else code,
        dict(contig=contig, sample_ids=sample_ids, recombination=parameters),
        NumericalPolicy.current() if policy is None else policy,
        inputs,
        tie_reverse=bool(reverse),
    )


def bind(file: CheckpointFile, expected: Derivation) -> str:
    """Bind before reading completion; a previously bound different derivation refuses."""
    identity = expected.identity
    publish(file, BINDING_KEY, identity, dict(schema=1, stage=KEY, derivation=identity))
    # Bind the derivation first: a changed choice must refuse before configuration publication.
    configuration = configuration_identity(
        expected.parameters["contig"], tuple(expected.parameters["sample_ids"]), expected.inputs
    )
    publish(
        file,
        CONFIGURATION_KEY,
        configuration,
        dict(schema=1, parameters=to_payload(expected.parameters["recombination"])),
    )
    return identity


def configuration_identity(contig: str, sample_ids: tuple[str, ...], inputs: Mapping[str, str]) -> str:
    """Name the upstream context of one immutable downstream parameter choice."""
    return json_digest(dict(schema=1, stage=KEY, contig=contig, sample_ids=sample_ids, inputs=dict(inputs)))


def bound_parameters(file: CheckpointFile, expected_configuration: str) -> RecombinationParameters:
    """Read parameters by independent upstream identity, never by the product envelope."""
    saved = resume(file, CONFIGURATION_KEY, expected_configuration)
    if saved is None:
        raise InvariantError("recombination parameter binding is incomplete; supply the requested parameters")
    require_schema(saved.payload, 1)
    return from_payload(RecombinationParameters, saved.payload["parameters"])


def load(file: CheckpointFile, expected: Derivation) -> Recombination | None:
    """Validate the persistent binding and then resume the complete scientific record."""
    identity = bind(file, expected)
    record = resume(file, KEY, identity)
    return None if record is None else RecombinationAdapter().decode(record.payload)


def write(file: CheckpointFile, expected: Derivation, product: Recombination) -> None:
    """Publish the complete record through the same immutable binding."""
    from ...contracts.roles import freeze_record_arrays

    freeze_record_arrays(product)
    identity = bind(file, expected)
    publish(file, KEY, identity, RecombinationAdapter().encode(product))


@dataclass(frozen=True)
class Reader:
    """Concrete Results request with its expected identity rebuilt from manifest and checked phase."""

    identity: str
    key: str = KEY
    schema: int = 2
    items: tuple[str, ...] | None = None

    def decode(self, payload: Mapping[str, Any]) -> Recombination:
        return RecombinationAdapter().decode(payload)


def expected(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    parameters: RecombinationParameters | None = None,
) -> Reader | None:
    """Rebuild identity from checked upstreams and the immutable RC parameter choice.

    Alias preflight passes its explicit requested parameters. Ordinary Results
    reads the stage-owned binding, allowing an independent upstream-only root to
    choose RC settings without changing its original initialization manifest.
    """
    from ...core.errors import InputError
    from ...store.protocol import read_items
    from ..phase import records as phase_records
    from ..relatedness.working_map import map_identity

    if product != "recombination" or contig is None or contig not in manifest["contigs"]:
        raise InputError("recombination reader requires a bound contig")
    file = store.contig(contig)
    phase = phase_records.result_reader(store, manifest, product="phase", contig=contig)
    if phase is None:
        raise InvariantError("recombination requires a complete phase publication")
    checked = read_items(file, phase.key, phase.identity, ("schema", "edges_identity"))
    if checked is None:
        raise InvariantError("recombination phase publication is incomplete")
    inputs = dict(
        phase=phase.identity,
        pedigree=checked.payload["edges_identity"],
        genetic_map=map_identity(store, manifest, contig),
    )
    if parameters is None:
        parameters = bound_parameters(file, configuration_identity(contig, tuple(manifest["sample_ids"]), inputs))
    expected = _bound_derivation(
        contig,
        tuple(manifest["sample_ids"]),
        parameters,
        inputs,
        reverse=bool(manifest["ties"]["reverse"]),
        policy=NumericalPolicy(**manifest["numerical_policy"]),
    )
    return expected


def result_reader(store, manifest, *, product, contig=None, parameters=None):
    """Request the same expected derivation used by the production publisher."""
    if contig is None or not (store.contig(contig).has(KEY) or store.contig(contig).has(BINDING_KEY)):
        return None
    request = expected(store, manifest, product=product, contig=contig, parameters=parameters)
    return Reader(request.identity)


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Return committed product status from container metadata without decoding arrays."""
    info = file.info(KEY)
    return dict(
        stage=KEY, complete=info is not None, **({} if info is None else dict(decoded_bytes=info.decoded_bytes))
    )
