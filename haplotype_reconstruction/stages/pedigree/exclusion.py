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
from ...core.concurrency import threaded_each
from ...core.digest import array_digest
from ...core.errors import InputError, InvariantError
from ...engine.cpu import claim_threads, thread_team
from ...science.pedigree import exclusion_bound
from ...science.pedigree.exclusion_parameters import ExclusionParameters
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy
from ..evidence import records as evidence_records
from ..framework import RecordKey, ResourceRequest, Task, TaskKey, load_product, publish_product
from ..painting import read_library_records

KEY = "pedigree/exclusion"


def sharing_samples(store, manifest, evidence_ids, read_libraries) -> tuple[bool, ...]:
    """Samples in any fitted library on the analysed contigs, using painting's published leaves."""
    by_contig = {}
    for row in read_libraries:
        by_contig.setdefault(row["contig"], []).append(row)
    sample_count = len(manifest["sample_ids"])
    mixed = []
    for contig, identity in evidence_ids:
        rows = by_contig.get(contig, ())
        fitted = [row["status"] == "fitted" for row in rows]
        if fitted and all(fitted):
            return (True,) * sample_count
        if any(fitted):
            mixed.append((contig, identity, sorted(rows, key=lambda row: row["library"])))
    if not mixed:
        return ()

    parameters = read_library_records.read_state_parameters(manifest)

    def recover(item):
        contig, identity, rows = item
        arrays = evidence_records.load_arrays(store.contig(contig), ("counts", "positions"),
                                             expected_identity=identity)
        groups = read_library_records.published_groups(
            arrays["counts"].sum(axis=2, dtype=np.float64), arrays["positions"],
            [row["samples"] for row in rows], parameters,
        )
        return [sample for row, group in zip(rows, groups, strict=True)
                if row["status"] == "fitted" for sample in group]

    sharing = np.zeros(sample_count, dtype=np.bool_)
    for samples in threaded_each(recover, mixed):
        sharing[samples] = True
    return tuple(sharing.tolist())


class ExclusionAdapter:
    """Persist the scalar endpoints, contig weight and dyad certification eligibility."""

    schema = 3

    def encode(self, value: ExclusionEvidence) -> Mapping[str, Any]:
        if value.log_e_upper is None or value.certifiable is None:
            raise InvariantError("exclusion publication requires upper endpoints and certification eligibility")
        return {"schema": self.schema, **to_payload(value)}

    def decode(self, payload: Mapping[str, Any]) -> ExclusionEvidence:
        require_schema(payload, self.schema)
        names = {f.name for f in fields(ExclusionEvidence)}
        result = from_payload(ExclusionEvidence, {k: v for k, v in payload.items() if k != "schema" or k in names})
        if result.log_e_upper is None or result.certifiable is None:
            raise InvariantError("exclusion record lacks upper endpoints or certification eligibility")
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
    sample_rates: tuple[float, ...] = (),
    sharing: tuple[bool, ...] = (),
) -> str:
    """Bind full evidence, ordered dyads, the configured contig axis, this contig's log weight, the samples and their
    per-sample outlier rates (A11; empty: the 4.20 constant allowance)."""
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(exclusion=parameters, model=exclusion_bound.MODEL, samples=sample_ids, contigs=configured_contigs,
             log_weight=float(log_weight), sample_rates=tuple(float(x) for x in sample_rates), sharing=sharing),
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
    sample_rates: tuple[float, ...] = (),
    sharing: tuple[bool, ...] = (),
) -> ExclusionEvidence:
    """Return scalar scores and certification eligibility in request order. ``sample_rates``: A11's per-sample
    outlier rates (empty: the 4.20 constant allowance)."""
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
    certifiable = None
    if sharing:
        fitted = np.asarray(sharing, dtype=np.bool_)
        certifiable = ~(fitted[rows[:, 0]] | fitted[rows[:, 1]])
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
            sample_rates=np.asarray(sample_rates, dtype=np.float64) if len(sample_rates) else None,
            certifiable=certifiable,
        )
    if file is not None:
        assert identity is not None
        publish_product(file, KEY, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def run_stored(
    contig: str, source: CheckpointFile, evidence_identity: str, dyads: np.ndarray,
    parameters: ExclusionParameters, log_weight: float, sample_count: int, *,
    threads: int = 1, file: CheckpointFile | None = None, identity: str | None = None,
    sample_rates: tuple[float, ...] = (), sharing: tuple[bool, ...] = (),
) -> ExclusionEvidence:
    """Read the scalar likelihoods and observations for the requested contig."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    if file is not None and identity is not None:
        previous = load_product(file, KEY, identity, ExclusionAdapter())
        if previous is not None:
            return previous
    evidence = evidence_records.load_dense(source, expected_identity=evidence_identity)
    return run(contig, evidence, dyads, parameters, log_weight, sample_count, threads=threads, file=file,
               identity=identity, sample_rates=sample_rates, sharing=sharing)


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
    sample_rates: tuple[float, ...] = (),
    sharing: tuple[bool, ...] = (),
) -> Task:
    """Declare a requested contig; callers omit this task when D is zero."""
    if not len(dyads):
        raise InputError("an empty exclusion request schedules no contig task")
    return Task(
        TaskKey("pedigree", "exclusion", contig),
        run,
        args=(contig, evidence, dyads, parameters, log_weight, sample_count),
        kwargs=dict(file=file, identity=identity, sample_rates=tuple(sample_rates), sharing=sharing),
        depends_on=depends_on,
        inputs=(RecordKey(contig, "evidence"),),
        outputs=(RecordKey(contig, KEY),),
        resources=ResourceRequest(max_cores=cores),
        affinity=contig,
        thread_argument="threads",
    )
