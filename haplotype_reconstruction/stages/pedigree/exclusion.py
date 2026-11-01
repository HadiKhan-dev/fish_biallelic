"""Score requested exclusion dyads once per configured contig (PS20–PS22).

Request order is authoritative: target prefix and guard-only suffix are retained.
The configured chromosome divisor is passed to every certificate, including when
only one contig is executed in a stage cut. Empty requests schedule no task.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.payload import from_payload, to_payload
from ...contracts.pedigree import ExclusionEvidence, ExclusionRequest
from ...contracts.roles import require_schema
from ...core.digest import array_digest
from ...core.errors import InputError, InvariantError
from ...engine.runtime import thread_team
from ...science.pedigree import exclusion_bound
from ...science.pedigree.parameters import ExclusionConfig
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy
from ..framework import RecordKey, ResourceRequest, Task, TaskKey, load_product, publish_product

KEY = "pedigree/exclusion"


class ExclusionAdapter:
    """Persist both certificate endpoints; a missing upper endpoint is invalid."""

    schema = 1

    def encode(self, value: ExclusionEvidence) -> Mapping[str, Any]:
        if value.log_e_upper is None:
            raise InvariantError("exclusion publication requires its upper endpoint")
        return {"schema": self.schema, **to_payload(value)}

    def decode(self, payload: Mapping[str, Any]) -> ExclusionEvidence:
        require_schema(payload, self.schema)
        names = {f.name for f in fields(ExclusionEvidence)}
        result = from_payload(ExclusionEvidence, {k: v for k, v in payload.items() if k != "schema" or k in names})
        if result.log_e_upper is None:
            raise InvariantError("exclusion record lacks its upper endpoint")
        return result


def derivation(
    parameters: ExclusionConfig,
    evidence_identity: str,
    dyads: np.ndarray,
    sample_ids: tuple[str, ...],
    configured_contigs: tuple[str, ...],
    *,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind full evidence, ordered dyads, configured C and sample axis."""
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(exclusion=parameters, model=exclusion_bound.MODEL, samples=sample_ids, contigs=configured_contigs),
        numerical_policy or NumericalPolicy.current(),
        dict(evidence=evidence_identity, dyads=array_digest(dyads)),
    ).identity


def run(
    contig: str,
    evidence: ContigEvidence,
    dyads: np.ndarray,
    parameters: ExclusionConfig,
    configured_contigs: int,
    sample_count: int,
    *,
    threads: int = 1,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> ExclusionEvidence:
    """Return the certificate in request order; scientific failure propagates."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    rows = np.ascontiguousarray(dyads, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise InputError("exclusion dyads must have shape [D,2]")
    adapter = ExclusionAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, KEY, identity, adapter)
        if previous is not None:
            return previous
    with thread_team(threads):
        product, _ = exclusion_bound.score_contig(
            contig,
            evidence.likelihoods,
            evidence.observed,
            evidence.positions,
            rows,
            parameters,
            configured_contigs,
            sample_count,
        )
    if file is not None:
        assert identity is not None
        publish_product(file, KEY, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def task(
    contig: str,
    evidence: ContigEvidence,
    dyads: np.ndarray,
    parameters: ExclusionConfig,
    configured_contigs: int,
    sample_count: int,
    *,
    file: CheckpointFile | None = None,
    identity: str | None = None,
    cores: int = 1,
    depends_on: tuple[TaskKey, ...] = (),
) -> Task:
    """Declare a requested contig; callers omit this task when D is zero."""
    if not len(dyads):
        raise InputError("an empty exclusion request schedules no contig task")
    return Task(
        TaskKey("pedigree", "exclusion", contig),
        run,
        args=(contig, evidence, dyads, parameters, configured_contigs, sample_count),
        kwargs=dict(file=file, identity=identity),
        depends_on=depends_on,
        inputs=(RecordKey(contig, "evidence"),),
        outputs=(RecordKey(contig, KEY),),
        resources=ResourceRequest(max_cores=cores),
        affinity=contig,
        thread_argument="threads",
    )


def tasks(
    request: ExclusionRequest, evidence: Mapping[str, ContigEvidence], parameters: ExclusionConfig, *, cores: int = 1
) -> tuple[Task, ...]:
    """One task per configured contig for targets, guard-only requests, or no tasks."""
    if not len(request.dyads):
        return ()
    return tuple(
        task(
            contig,
            evidence[contig],
            request.dyads,
            parameters,
            len(request.contigs),
            len(request.sample_ids),
            cores=cores,
        )
        for contig in request.contigs
    )
