"""The painting's read-state builder for one contig (PaintingParameters.read_classes > 1; science.shared.reads).

Reads the contig's inputs of the read state from the evidence record: the REF/ALT counts and positions of the dense
evidence, the published read-sharing curve (the all-sample survival fit), the evidence read model's per-sample error
rates, and groups the samples into sequencing libraries (science.shared.reads.library_groups), then wraps the stage's
emission builder in science.painting.read_inputs.ReadEmissionBuilder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from ...contracts.evidence import EvidenceDecisions
from ...contracts.payload import from_payload
from ...core.concurrency import threaded_each
from ...core.errors import InvariantError
from ...engine.records import record_diagnostic
from ...science.shared.read_loss import check_zero_loss
from ...science.evidence.read_sharing import read_sharing
from ...science.painting.parameters import READ_ACTION_MEMORY_FRACTION, PaintingParameters
from ...science.painting.read_inputs import ReadEmissionBuilder, split_minor
from ...science.shared import reads
from ...store.container import CheckpointFile
from ...store.protocol import read_items
from ..evidence import records as evidence_records

__all__ = ["stage_builder", "contig_read_libraries", "detection_margins", "read_state_route", "run_detection",
           "with_detection", "effective_painting", "READ_STATE_ON", "READ_STATE_NO_SHARING"]

# The run's read-state route, recorded by the calibration (painting/calibration outlier and calibration records).
READ_STATE_ON = "on"
READ_STATE_NO_SHARING = "off: no library shares reads"
READ_STATE_ONE_CLASS = "off: read_classes = 1"


def run_detection(store, manifest, settings) -> dict:
    """The run's family-wise read-sharing detection (ReadStateParameters.sharing_detection_alpha): dict(alpha, tests
    M, z = z*) from every analysed contig's read-sharing record: M counts each contig's all-sample fit and its
    candidate library splits' tests over the contig's structural bins (reads.structural_bins, reads.candidate_splits,
    reads.family_tests; z* by reads.family_z); recorded by the calibration and carried on the paintings' parameters
    (with_detection)."""
    from ...contracts.contigs import AnalysisContigs

    bins, samples = [], 0
    for contig in AnalysisContigs.from_manifest(manifest).included:
        ev = evidence_records.expected(store, manifest, product="evidence", contig=contig)
        sharing = evidence_records.load_read_sharing(store.contig(contig), ev.identity)
        bins.append(0 if sharing is None else reads.structural_bins(sharing))
        samples = max(samples, 0 if sharing is None else int(sharing.samples))
    tests = reads.family_tests(bins, reads.candidate_splits(samples, settings))
    alpha = float(settings.sharing_detection_alpha)
    return dict(alpha=alpha, tests=tests, z=reads.family_z(alpha, tests))


def with_detection(parameters: PaintingParameters, detection) -> PaintingParameters:
    """``parameters`` with the run's detection threshold z* (run_detection's ``detection``) on its read state."""
    return replace(parameters, read_state=replace(parameters.read_state, detection_z=float(detection["z"])))


def _count_inputs(file, evidence_identity):
    arrays = evidence_records.load_arrays(file, ("counts", "positions"), expected_identity=evidence_identity)
    found = read_items(file, "evidence", evidence_identity, ("schema", "contig", "decisions"))
    if found is None:
        raise InvariantError("the painting's read state requires the evidence decisions")
    counts, positions = np.asarray(arrays["counts"]), np.asarray(arrays["positions"])
    return found.payload, counts, positions, counts.sum(axis=2)


def _library_laws(file: CheckpointFile, evidence_identity: str, settings):
    """Read the count inputs and fit their library survival laws once per contig."""
    z = settings.detection_z
    if z is None:
        raise InvariantError("the read state's detection threshold is resolved per run (the calibration's "
                             "read_detection, carried on the paintings' parameters)")
    info, counts, positions, depth = _count_inputs(file, evidence_identity)
    sharing = evidence_records.load_read_sharing(file, evidence_identity)
    survival = reads.fit_survival(sharing, settings, z) if sharing is not None else reads.no_survival(settings)
    library, survivals, splits, strengths = reads.library_groups(
        depth, positions, str(info["contig"]), settings, read_sharing, z, map_work=threaded_each
    )
    laws = [survival if law is None else law for law in survivals]
    whole = reads.bin_summary(sharing) if sharing is not None else dict(max_z=float("nan"), bin_bp=None, usable=0)
    strengths = [whole if s is None else s for s in strengths]
    if sharing is not None:
        for index, law in enumerate(laws):
            selected, check = check_zero_loss(law, depth, np.flatnonzero(library == index), positions,
                                              sharing.edges, settings, z, map_work=threaded_each)
            laws[index] = selected
            strengths[index] = dict(strengths[index], zero_loss_check=check, applied_loss=selected.loss,
                                    covariance_loss=law.fitted_loss)
            if len(laws) == 1:
                survival = selected
            else:
                survivals[index] = selected
    return info, counts, positions, depth, survival, library, survivals, splits, laws, strengths


def contig_read_libraries(file: CheckpointFile, evidence_identity: str, parameters: PaintingParameters) -> tuple:
    """Library summaries, split diagnostics and fitted setup as grouped by the painting's read state. Each library
    dict(library, samples, status: 'fitted' (shares reads) or 'none', max_z: its largest usable-bin excess / SE,
    bin_bp: that bin's lag interval, usable: its usable bins); each candidate split as reads.library_groups logs it
    (level, sizes, contrast z, survival difference, accepted, the parts' max z and fit status). ``parameters``
    carry the run's z* (with_detection). The last returned item is the fitted library setup for later calibration
    tasks, from exactly the laws used to produce those diagnostics."""
    _, _, _, _, survival, library, survivals, splits, laws, strengths = _library_laws(
        file, evidence_identity, parameters.read_state,
    )
    libraries = [dict(library=k, samples=int((library == k).sum()), status=law.status,
                      max_z=float(strengths[k]["max_z"]),
                      bin_bp=None if strengths[k]["bin_bp"] is None else list(strengths[k]["bin_bp"]),
                      usable=int(strengths[k]["usable"]), applied_loss=strengths[k].get("applied_loss"),
                      covariance_loss=strengths[k].get("covariance_loss"),
                      zero_loss_check=strengths[k].get("zero_loss_check"))
                 for k, law in enumerate(laws)]
    splits = [dict(split, sizes=list(split["sizes"]), parts_max_z=list(split["parts_max_z"]),
                   parts_status=list(split["parts_status"])) for split in splits]

    def law_payload(law):
        return dict(grid=law.grid, weights=law.weights, loss=law.loss, status=law.status, fitted_loss=law.fitted_loss)

    setup = dict(
        library=library, survival=law_payload(survival),
        survivals=tuple(None if law is None else law_payload(law) for law in survivals), splits=splits,
    )
    return libraries, splits, setup


def detection_margins(contigs, detection, route: str, tolerance: float) -> tuple[list, list, list]:
    """(libraries, splits, warnings) of the run: every analysed contig's libraries and candidate splits (``contigs``:
    [(contig, libraries, splits)] of contig_read_libraries, libraries None where not assessed) with their signed
    margin to z* (statistic - z*) and near_edge, and one run-report warning line per library or split near the edge,
    naming the run's calibration ``route``. A library is near the edge when its largest excess / SE lies within a
    factor of two of z* (reads.near_detection_edge); a candidate split when its contrast does and the contrast
    decides it (its parts' survival laws differ by more than ``tolerance``, library_survival_tolerance: otherwise
    the split fails on the material difference whatever its contrast; on a library that shares reads the
    two-means partition gives contrasts of 4-10 between parts of one law, which the tolerance rejects)."""
    z, tests, alpha = float(detection["z"]), int(detection["tests"]), float(detection["alpha"])
    libraries, splits, warnings = [], [], []
    for contig, rows, candidates in contigs:
        for row in rows or ():
            near = reads.near_detection_edge(row["max_z"], z)
            margin = float(row["max_z"]) - z
            libraries.append(dict(row, contig=contig, margin=margin, near_edge=near))
            if near:
                side = "sharing" if row["status"] == "fitted" else "no sharing"
                where = ("no usable bin" if row["bin_bp"] is None else
                         f"lag bin {row['bin_bp'][0]}-{row['bin_bp'][1]} bp")
                warnings.append(
                    f"read sharing: {contig} library {row['library']} ({row['samples']} samples): {side} detected, "
                    f"largest excess/SE {row['max_z']:.4f} at {where} vs z* {z:.4f} (margin {margin:+.4f}; "
                    f"{row['usable']} usable bins; M {tests} tests, alpha {alpha:g}; calibration route '{route}'): "
                    f"the read-state decision for this library is close to the detection edge.")
        for split in candidates or ():
            near = reads.near_detection_edge(split["z"], z) and float(split["survival_difference"]) > tolerance
            margin = float(split["z"]) - z
            splits.append(dict(split, contig=contig, margin=margin, near_edge=near))
            if near:
                verdict = "accepted" if split["accepted"] else "rejected"
                warnings.append(
                    f"read sharing: {contig} candidate library split {split['sizes'][0]} / {split['sizes'][1]} "
                    f"samples (level {split['level']}): {verdict}, contrast {split['z']:.4f} vs z* {z:.4f} (margin "
                    f"{margin:+.4f}; survival difference {split['survival_difference']:.4f}; parts' largest "
                    f"excess/SE {split['parts_max_z'][0]:.4f} / {split['parts_max_z'][1]:.4f}; calibration route "
                    f"'{route}'): the library grouping is close to the detection edge.")
    return libraries, splits, warnings


def read_state_route(parameters: PaintingParameters, sharing) -> str:
    """The run's read-state route from each analysed contig's sharing bit (any library 'fitted' in
    contig_read_libraries; ``sharing``, None where not assessed): READ_STATE_ON when read_classes > 1 and some library
    of the run shares reads; READ_STATE_NO_SHARING
    when no library of the run does (the read model's exact limit without sharing is one read class: the run takes
    the route without read state, calibration included); READ_STATE_ONE_CLASS when read_classes = 1."""
    if parameters.read_classes == 1:
        return READ_STATE_ONE_CLASS
    return READ_STATE_ON if any(bool(s) for s in sharing) else READ_STATE_NO_SHARING


def effective_painting(parameters: PaintingParameters, calibration) -> PaintingParameters:
    """The painting parameters the run paints with: read_classes = 1 when the run calibration recorded the read state
    off because no library shares reads (``calibration``: its payload; a record without the field is 'on'), else
    ``parameters`` with the run's detection threshold z* (the calibration's read_detection). The paintings carry
    them, so every consumer of a painting (the pedigree's source chain and map scale included) takes the same route
    and the same threshold."""
    if parameters.read_classes == 1:
        return parameters
    parameters = replace(parameters, foreign_read_share=calibration.get("foreign_read_share",
                                                                         parameters.foreign_read_share))
    if calibration.get("read_state", READ_STATE_ON) != READ_STATE_ON:
        return replace(parameters, read_classes=1)
    # the run's family-wise detection threshold z*, which every read-state builder of the run uses
    return with_detection(parameters, calibration["read_detection"])


def stage_builder(
    base: Callable,
    file: CheckpointFile,
    evidence_identity: str,
    cohort: np.ndarray,
    confidence_correlation: float | None,
    parameters: PaintingParameters,
    *,
    unit_batches: bool = False,
    label_correlation: float | None = None,
    read_setup=None,
    corrected_rows: tuple[int, ...] = (),
) -> Callable:
    """The emission builder of one contig's painting: ``base`` unchanged when ``parameters.read_classes`` is 1 (no read
    state), else the read-state builder with 2 minor + 3 = read_classes classes over ``base``'s emissions. Its inputs
    are the contig's dense counts and positions, the read-sharing record's survival fit, per-library survival fits
    when the samples form several libraries, the evidence read model's fitted per-sample error rates (the read-state
    fallback rate when the evidence stage applied its fixed model), and the samples' covering depths per homologue
    (mean called depth / 2 / (1 - loss)). Samples of a library without read sharing (survival status 'none') and
    samples deep at every site bypass the read state (ReadEmissionBuilder: the emission without read state exactly).
    ``unit_batches``: a batch whose every sample bypasses is carried with one class (ReadEmissionBuilder; the
    painting's and the calibration's consumers read the class count from the batch; a consumer that assumes
    read_classes classes keeps the default). ``label_correlation``: the label-error strength
    (parameters.LABEL_ERROR_CORRELATION) when a decode of this builder reads the label-error statistic (the
    reference paintings), else None. The read matrices take the bin-level tempering correction
    (science.painting.read_factor). When calibration already fitted the libraries, its robustness
    receipt supplies the same laws and memberships through ``read_setup``. ``corrected_rows``: the samples whose
    likelihood rows QC A replaced (contamination.corrected_evidence_with_rows, which every consumer of the corrected
    evidence passes here): they carry no uniform foreign class (ReadEmissionBuilder)."""
    if parameters.read_classes == 1:
        return base
    settings = parameters.read_state
    minor = split_minor(parameters.read_classes)
    if read_setup is None:
        info, counts, positions, depth, survival, library, survivals, splits, laws, _ = _library_laws(
            file, evidence_identity, settings
        )
    else:
        info, counts, positions, depth = _count_inputs(file, evidence_identity)
        survival = reads.ReadSurvival(**read_setup["survival"])
        library, splits = read_setup["library"], read_setup["splits"]
        survivals = [None if law is None else reads.ReadSurvival(**law) for law in read_setup["survivals"]]
        laws = [survival if law is None else law for law in survivals]
    model = from_payload(EvidenceDecisions, info["decisions"]).read_model
    if model.selected is None:
        # the evidence stage's fixed binomial model (science.evidence.run: fixed_model(read_model.fixed_error))
        error_rate = np.full(counts.shape[0], float(settings.fallback_error))
    else:
        fit = model.candidates[model.selected]
        error_rate = 1.0 / (1.0 + np.exp(-np.asarray(fit.error_logit, dtype=np.float64)))
    loss = np.full(depth.shape[0], survival.loss)
    if len(survivals) > 1:
        for k, law in enumerate(survivals):
            loss[library == k] = law.loss
    covering = depth.mean(axis=1) / 2.0 / np.maximum(1.0 - loss, 1.0 - settings.maximum_loss)
    deep = reads.deep_rows(depth, minor, settings.deep_tail)
    record_diagnostic(
        f"{info['contig']}: painting read state, {parameters.read_classes} classes; libraries "
        f"{[int((library == k).sum()) for k in range(len(survivals))]} (candidate splits {splits}); "
        f"{int(deep.sum())} of {len(deep)} samples deep at every site; "
        f"{int(sum((library == k).sum() for k, law in enumerate(laws) if law.status != 'fitted'))} samples in "
        f"libraries without read sharing (both bypass the read state; detection z* {settings.detection_z:.3f}); "
        f"covering depth per homologue median "
        f"{float(np.median(covering)) if len(covering) else 0.0:.2f}; "
        f"foreign share {parameters.foreign_read_share:g}"
        + (f" (none for the {len(corrected_rows)} QC A-corrected samples {list(corrected_rows)})"
           if corrected_rows and parameters.foreign_read_share else "")
    )
    # Table chunks use the ordered map. Site kernels use the ambient team.
    return ReadEmissionBuilder(base, counts, positions, survival, error_rate, covering, cohort, minor, settings,
                               confidence_correlation, library, survivals, map_work=threaded_each,
                               unit_batches=unit_batches, label_correlation=label_correlation,
                               foreign_read_share=parameters.foreign_read_share, action_bytes=_action_bytes(),
                               corrected_rows=corrected_rows)


def _action_bytes() -> int | None:
    """The compiled-action memory of this run's builders (ReadEmissionBuilder.compiles): READ_ACTION_MEMORY_FRACTION of
    the run's memory limit (the job's memory capacity, or the configured limit when lower); None (always compile)
    where no limit can be read. A capacity, not the live usage, so the route of a unit is the same at every point of
    the run on one allocation."""
    from ...engine.memory import current_governor
    from ...engine.runtime import memory_pressure

    pressure = memory_pressure()
    limit = None if pressure is None else pressure[1]
    governor = current_governor()
    if governor is not None:
        limit = governor.limit if limit is None else min(limit, governor.limit)
    return None if limit is None else int(READ_ACTION_MEMORY_FRACTION * limit)
