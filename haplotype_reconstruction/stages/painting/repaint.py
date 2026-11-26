"""The reference painting decoded again on the founder refit's founders: the painting the pedigree reads.

Owns stage "repaint": its record key, identity, reader and contig action. Order of the run:
painting -> founder refit -> repaint -> pedigree. The refit keeps the ORIGINAL painting as its input and never reads
this product; nothing here feeds back upstream.

The decode is the reference painting's (stages.painting.tasks) with one input replaced: the founder panels are the
refit's (the fine painting's panels with the refit's allele changes). Parameters, evidence, working map and run
calibration are those of the first painting. The fine painting's label-continuity outcome is passed without row maps,
because the refit's panels already carry them.

The record identity binds the reference painting's identity (hence the final assembly, the evidence and its payload,
the working map, the run calibration, the painting parameters and the fine painting) and the refit's identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.painting import Painting
from ...contracts.roles import require_schema
from ...core.errors import InputError, InvariantError
from ...core.ties import TieSettings
from ...science.painting.diagnostics import summarise
from ...science.painting.parameters import LABEL_ERROR_CORRELATION
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, claim, mark_complete, resume
from ...store.store import CheckpointStore
from ..framework import RecordKey, ResourceRequest, Task
from . import records
from .compute import paint_chromosome
from .read_state import stage_builder
from .tasks import WindowEvidence, _at_batch_boundary, _Progress, read_evidence, windowed_emissions

if TYPE_CHECKING:
    from ...contracts.genetic_map import GeneticMapSet
    from ..configure import ResolvedConfiguration
    from ..context import StageRun

REFIT_REFERENCE = "painting/refit_reference"
SCHEMA = records.SCHEMA

__all__ = ["REFIT_REFERENCE", "SCHEMA", "derivation", "expected", "result_reader", "run", "task"]


def derivation(reference_identity: str, refit_identity: str, contamination_identity: str, *,
               numerical_policy: NumericalPolicy | None = None) -> str:
    """The second reference decode's identity: the reference painting's, the founder refit's and the run's
    contamination decision (4.22 QC A: flagged libraries are decoded on corrected likelihoods)."""
    return Derivation(
        code_dependency_identity(("stages.painting.repaint",)),
        dict(product=REFIT_REFERENCE), numerical_policy or NumericalPolicy.current(),
        dict(reference=reference_identity, refit=refit_identity, contamination=contamination_identity),
    ).identity


def expected(
    store: CheckpointStore, manifest: Mapping[str, Any], *, product: str, contig: str | None = None
) -> records.Reader:
    """Rebuild the identity from the checked reference painting and refit requests; never read it from this
    product's own envelope."""
    from ..refit import records as refit_records
    from . import contamination_records

    if product != "refit_reference":
        raise InputError(f"unsupported refit painting reader product {product!r}")
    reference = records.expected(store, manifest, product="reference", contig=contig)
    refit = refit_records.expected(store, manifest, product="refit", contig=contig)
    if reference is None or refit is None:
        raise InvariantError("the refit painting requires the reference painting and the founder refit")
    contamination = contamination_records.expected(store, manifest, product="contamination")
    identity = derivation(
        reference.identity, refit.identity, contamination.identity,
        numerical_policy=NumericalPolicy(**manifest["numerical_policy"]),
    )
    return records.Reader(identity, product, key=REFIT_REFERENCE)


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    panels: bool = True,
    emissions: bool = True,
) -> records.Reader | None:
    """The published second painting's request (None before it exists). It carries no founder panels: they are the
    founder refit's (stages.refit.records)."""
    del panels
    file = store.contig(contig)
    if not file.has(REFIT_REFERENCE):
        return None
    request = expected(store, manifest, product=product, contig=contig)
    packed = None
    if emissions:
        cache = resume(file, REFIT_REFERENCE + "/cache", request.identity)
        if cache is not None:
            require_schema(cache.payload, SCHEMA)
            packed = tuple(cache.payload["packed_emissions"])
    return records.Reader(request.identity, product, None, packed, key=REFIT_REFERENCE)


def run(
    root: str,
    contig: str,
    resolved: ResolvedConfiguration,
    maps: GeneticMapSet,
    ties: TieSettings,
    *,
    team: int = 1,
    cores: int = 1,
) -> dict[RecordKey, str]:
    """Paint the contig again, at the reference grid, on the founder refit's founders."""
    from ...store.protocol import bound_payload_identity
    from ..evidence import records as evidence_records
    from ..inputs.products import required_product, task_inputs
    from ..refit import records as refit_records
    from ..relatedness.working_map import working_maps
    from . import calibration_records

    reader, store = task_inputs(root)
    file = store.contig(contig)
    identity = expected(store, reader.manifest, product="refit_reference", contig=contig).identity
    refit_request = reader.request(refit_records, "refit", contig, panels=False)
    reader.require_receipt(refit_request, contig)
    complete = {name: identity for name in (REFIT_REFERENCE, REFIT_REFERENCE + "/summary", "identity/" + REFIT_REFERENCE)}
    with claim(file, REFIT_REFERENCE, identity, blocking=False) as needed:
        if needed:
            refit = required_product(reader, refit_records, "refit", contig)
            fine = records.expected(store, reader.manifest, product="painting", contig=contig)
            painted = records.load(file, fine.identity, emissions=False)
            if painted is None:
                raise InvariantError("the refit painting requires the contig's painting")
            evidence_identity = evidence_records.expected(
                store, reader.manifest, product="evidence", contig=contig
            ).identity
            payload_identity = bound_payload_identity(file)
            calibration = required_product(reader, calibration_records, "calibration", None)
            from .read_state import effective_painting

            base = effective_painting(resolved.parameters.painting, calibration)
            parameters = replace(base, snps_per_bin=base.reference_snps_per_bin)
            genetic_map = working_maps(store, reader.manifest, maps).for_contig(contig)
            continuity = tuple(replace(outcome, accepted=False, row_maps=None) for outcome in painted.label_continuity)
            batch = max(1, team, cores)
            read_setup = (
                calibration_records.read_setup(store, reader.manifest, contig) if parameters.read_classes > 1 else None
            )

            def compute(source: ContigEvidence | WindowEvidence, corrected_rows: tuple[int, ...] = ()) -> Painting:
                if source.payload_identity != payload_identity:
                    raise InvariantError("refit painting evidence payload binding differs")
                return paint_chromosome(
                    tuple(refit.panels), source, parameters, genetic_map,
                    batch_samples=batch, reverse=ties.reverse,
                    # the painting's read state (read_classes > 1) as in stages.painting.tasks: the pedigree reads this
                    # painting, so it carries the same emission model as the run's fine and reference paintings
                    emission_builder=stage_builder(
                        partial(
                            _at_batch_boundary,
                            partial(windowed_emissions, sample_tile=batch, cohort=source.cohort_frequencies),
                            batch,
                        ),
                        file, evidence_identity, source.cohort_frequencies, calibration["confidence_correlation"],
                        parameters, unit_batches=True, label_correlation=LABEL_ERROR_CORRELATION, read_setup=read_setup,
                        corrected_rows=corrected_rows,
                    ),
                    progress=_Progress(file, identity), label_continuity=continuity,
                    label_errors=True,  # as the reference painting: the pedigree's painting carries the label errors
                    robustness_epsilon=calibration["robustness_epsilon"],
                    site_correlation=calibration["path_correlation"],
                    confidence_correlation=calibration["confidence_correlation"],
                )

            from . import contamination as contamination_stage

            _, decided = contamination_stage.decision(store, reader.manifest)
            if np.any(decided["flagged"]):
                # 4.22 QC A: the flagged libraries' rows corrected; a dense projection replaces the windows
                # (the uncorrected projection is not kept alive beside the corrected one during the decode). The
                # corrected rows carry no uniform foreign class (stage_builder): their likelihoods hold QC A's model
                source, corrected_rows = contamination_stage.corrected_evidence_with_rows(
                    store, reader.manifest, contig, evidence_records.load_dense(file, expected_identity=evidence_identity),
                    evidence_identity=evidence_identity)
                result = compute(source, corrected_rows)
                del source
            else:
                with read_evidence(file, evidence_identity) as source:
                    result = compute(source)
            records.write(file, identity, result, summarise(result), evidence_identity=evidence_identity,
                          key=REFIT_REFERENCE)
    mark_complete(file, "repaint", complete)
    return {RecordKey(contig, name): identity for name in complete}


def task(key, context: StageRun) -> Task:
    """Declare the second reference decode of one contig."""
    names = (REFIT_REFERENCE, REFIT_REFERENCE + "/summary", "identity/" + REFIT_REFERENCE)
    # Saved repaints check headers independently on one worker per contig.
    # A new decode still uses the scheduler's full available team.
    existing = CheckpointStore(context.root).contig(key.scope).has(REFIT_REFERENCE)
    return Task(
        key, run, (context.root, key.scope, context.resolved, context.maps, context.ties),
        kwargs={"cores": context.cores},
        outputs=tuple(RecordKey(key.scope, name) for name in names),
        resources=ResourceRequest(1, 1 if existing else context.cores), thread_argument="team",
    )
