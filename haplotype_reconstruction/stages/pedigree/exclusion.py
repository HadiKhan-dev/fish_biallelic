"""Score requested exclusion dyads once per configured contig (PS20–PS22).

Request order is authoritative: target prefix and guard-only suffix are retained.
Each contig's fixed weight in the genome e-value (DESIGN_417 D9: its configured working-map Morgans over the
run's) is passed to its certificate and bound in its identity, including when only one contig is executed in a
stage cut. Empty requests and contigs without material schedule no task.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.likelihoods import gather_likelihoods
from ...contracts.observations import dense_observed
from ...contracts.payload import from_payload, to_payload
from ...contracts.pedigree import ExclusionEvidence
from ...contracts.roles import require_schema
from ...core.digest import array_digest
from ...core.errors import InputError, InvariantError
from ...engine.cpu import claim_threads, thread_team
from ...science.pedigree import exclusion_bound
from ...science.pedigree.parameters import ExclusionParameters
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy
from ..framework import RecordKey, ResourceRequest, Task, TaskKey, load_product, publish_product

KEY = "pedigree/exclusion"


class ExclusionAdapter:
    """Persist both certificate endpoints and the contig's log weight; a missing upper endpoint is invalid.
    Schema 2 (4.18): 1/C records are refused."""

    schema = 2

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
    parameters: ExclusionParameters,
    evidence_identity: str,
    dyads: np.ndarray,
    sample_ids: tuple[str, ...],
    configured_contigs: tuple[str, ...],
    log_weight: float,
    *,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind full evidence, ordered dyads, the configured contig axis, this contig's log weight and the samples."""
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(exclusion=parameters, model=exclusion_bound.MODEL, samples=sample_ids, contigs=configured_contigs,
             log_weight=float(log_weight)),
        numerical_policy or NumericalPolicy.current(),
        dict(evidence=evidence_identity, dyads=array_digest(dyads)),
    ).identity


def run(
    contig: str,
    evidence: ContigEvidence,
    dyads: np.ndarray,
    parameters: ExclusionParameters,
    log_weight: float,
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
    with thread_team(default=threads):
        product, _ = exclusion_bound.score_contig(
            contig,
            gather_likelihoods(evidence.likelihoods, slice(None)),
            dense_observed(evidence.observed),
            evidence.positions,
            rows,
            parameters,
            log_weight,
            sample_count,
            claim_threads,  # the science's kernel boundaries re-read this task's share
        )
    if file is not None:
        assert identity is not None
        publish_product(file, KEY, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def task(
    contig: str,
    evidence: ContigEvidence,
    dyads: np.ndarray,
    parameters: ExclusionParameters,
    log_weight: float,
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
        args=(contig, evidence, dyads, parameters, log_weight, sample_count),
        kwargs=dict(file=file, identity=identity),
        depends_on=depends_on,
        inputs=(RecordKey(contig, "evidence"),),
        outputs=(RecordKey(contig, KEY),),
        resources=ResourceRequest(max_cores=cores),
        affinity=contig,
        thread_argument="threads",
    )
