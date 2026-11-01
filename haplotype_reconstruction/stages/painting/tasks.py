"""Run one contig's painting, with sequential components and whole-bin windows.

The engine owns the team and scheduling. Storage windows and cache sizes affect
cost only; evidence includes every sample. A completed product resumes as a unit,
so an interrupted contig repeats model fitting, label continuity and release.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.painting import Painting
from ...contracts.panels import materialise
from ...contracts.payload import to_payload
from ...contracts.roles import require_schema
from ...core.errors import InvariantError
from ...core.ties import TieCensus, TieSettings
from ...science.painting.diagnostics import ModelTrajectory, PaintingDiagnostics, summarise
from ...science.painting.emissions import ArraySource, gather_sites, tempered_window
from ...science.painting.fidelity import ChainModel
from ...science.painting.label_continuity import relabel_bytes
from ...science.painting.parameters import EMISSION_CACHE_MAX_BYTES, WINDOW_BYTES, PaintingParameters
from ...science.painting.run import paint_chromosome
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, claim, mark_complete, open_windows, publish, read_items
from ...store.slices import WindowArray
from ..framework import RecordKey, ResourceRequest, Task, TaskKey
from . import records

_DEFAULT_TIES = TieSettings()

__all__ = ["WindowEvidence", "windowed_emissions", "read_evidence", "identity", "run_contig", "painting_task"]


@dataclass(frozen=True)
class WindowEvidence:
    """Borrowed all-sample sources valid only inside read_evidence's context."""

    contig: str
    sample_ids: tuple[str, ...]
    likelihoods: ArraySource
    observed: ArraySource
    payload_identity: str


@contextmanager
def read_evidence(file: CheckpointFile, expected: str) -> Iterator[WindowEvidence]:
    """Verify identity before opening checked evidence windows; never mask founder QC."""
    found = read_items(file, "evidence", expected, ("schema", "contig", "sample_ids", "payload_identity"))
    if found is None:
        raise InvariantError("painting requires the evidence record")
    payload = found.payload
    require_schema(payload, 1)
    with open_windows(file, "evidence", expected, ("likelihoods", "observed")) as arrays:
        if arrays is None:
            raise InvariantError("evidence disappeared before painting windows opened")
        yield WindowEvidence(
            payload["contig"],
            tuple(payload["sample_ids"]),
            arrays["likelihoods"],
            arrays["observed"],
            payload["payload_identity"],
        )


def windowed_emissions(
    likelihoods: np.ndarray | WindowArray,
    observed: np.ndarray | WindowArray,
    sites: np.ndarray,
    samples: slice,
    alt_probability: np.ndarray,
    background_frequency: np.ndarray,
    bin_starts: np.ndarray,
    bin_stops: np.ndarray,
    config: PaintingParameters,
    *,
    parts: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    sample_tile: int = 1,
    window_bytes: int = WINDOW_BYTES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather bounded whole-bin windows and use the one scientific emission formula.

    Parts never split the design-effect denominator. Windows preserve every bin's
    ascending site accumulation, including its last short bin. Arrays and storage
    readers exercise the same route in forced-route tests.
    """
    start, stop, step = samples.indices(likelihoods.shape[0])
    if step != 1:
        raise InvariantError("painting requires contiguous sample batches")
    count = stop - start
    starts, stops = (bin_starts, bin_stops) if parts is None else parts[:2]
    states = alt_probability.shape[0] + 1
    raw = np.empty((count, states * states, len(starts)), dtype=np.float64)
    flags = np.empty((count, len(sites)), dtype=np.bool_)
    informative = np.empty_like(flags)
    if not len(bin_starts):
        return raw, flags, informative
    largest = int(np.max(bin_stops - bin_starts))
    for first_sample in range(0, count, max(1, sample_tile)):
        last_sample = min(count, first_sample + max(1, sample_tile))
        width = max(1, window_bytes // max(1, 25 * (last_sample - first_sample) * largest))
        for first_bin in range(0, len(bin_starts), width):
            last_bin = min(len(bin_starts), first_bin + width)
            left, right = int(bin_starts[first_bin]), int(bin_stops[last_bin - 1])
            first_part, last_part = (
                (first_bin, last_bin)
                if parts is None
                else (int(np.searchsorted(parts[2], first_bin)), int(np.searchsorted(parts[2], last_bin)))
            )
            selection = slice(start + first_sample, start + last_sample)
            if isinstance(likelihoods, np.ndarray):
                if not isinstance(observed, np.ndarray):
                    raise InvariantError("painting evidence sources must use the same storage route")
                values, seen = gather_sites(likelihoods, observed, sites[left:right], selection)
            else:
                if isinstance(observed, np.ndarray):
                    raise InvariantError("painting evidence sources must use the same storage route")
                values = likelihoods.gather(selection, sites[left:right])
                seen = observed.gather(selection, sites[left:right])
            emissions, info = tempered_window(
                values,
                seen,
                alt_probability[:, left:right],
                background_frequency[left:right],
                starts[first_part:last_part] - left,
                stops[first_part:last_part] - left,
                bin_starts[first_bin:last_bin] - left,
                config,
                part_bins=None if parts is None else parts[2][first_part:last_part] - first_bin,
            )
            raw[first_sample:last_sample, :, first_part:last_part] = emissions
            flags[first_sample:last_sample, left:right] = seen
            informative[first_sample:last_sample, left:right] = info
    return raw, flags, informative


def identity(
    parameters: PaintingParameters,
    *,
    contig: str,
    sample_ids: tuple[str, ...],
    assembly_identity: str,
    evidence_identity: str,
    payload_identity: str,
    map_identity: str,
    ties: TieSettings = _DEFAULT_TIES,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind scientific inputs and convention; exclude cache, batches and thread counts."""
    return Derivation(
        code_dependency_identity(("science.painting", "stages.painting.tasks", "stages.painting.records")),
        dict(parameters=parameters, contig=contig, sample_ids=sample_ids),
        numerical_policy or NumericalPolicy.current(),
        dict(assembly_final=assembly_identity, evidence=evidence_identity, payload=payload_identity, map=map_identity),
        tie_reverse=ties.reverse,
        tie_keep=ties.keep,
    ).identity


def run_contig(
    file: CheckpointFile,
    expected: str,
    assembly: records.Assembly,
    genetic_map: ChromosomeGeneticMap,
    parameters: PaintingParameters,
    *,
    evidence_identity: str,
    payload_identity: str,
    evidence: ContigEvidence | None = None,
    ties: TieSettings = _DEFAULT_TIES,
    threads: int = 1,
    batch_samples: int | None = None,
    emission_cache_bytes: int = EMISSION_CACHE_MAX_BYTES,
    census: bool = False,
) -> Painting:
    """Resume or execute model -> continuity -> decode -> release; publish product last."""
    complete = {records.PAINTING: expected, records.SUMMARY: expected, records.IDENTITY: expected}
    with claim(file, records.PAINTING, expected) as needed:
        if not needed:
            result = records.load(file, expected, assembly=assembly)
            if result is None:
                raise InvariantError("painting claim completed without product")
        else:
            trajectories: list[ModelTrajectory] = []
            events: list[TieCensus] = []

            def census_sink(event: TieCensus) -> None:
                if census:
                    events.append(event)
                if ties.census is not None:
                    ties.census(event)

            def model_sink(component: int, model: ChainModel | None) -> None:
                trajectories.append(ModelTrajectory(component, () if model is None else model.trajectory))

            def compute(source: ContigEvidence | WindowEvidence) -> Painting:
                if source.payload_identity != payload_identity:
                    raise InvariantError("painting evidence payload binding differs")
                from functools import partial

                return paint_chromosome(
                    materialise(assembly[1], assembly[0]),
                    source,
                    parameters,
                    genetic_map,
                    batch_samples=max(1, threads if batch_samples is None else batch_samples),
                    reverse=ties.reverse,
                    emission_cache_bytes=emission_cache_bytes,
                    model_sink=model_sink,
                    census_sink=census_sink if census or ties.census is not None else None,
                    emission_builder=partial(windowed_emissions, sample_tile=max(1, threads)),
                )

            wall, cpu = time.perf_counter(), time.process_time()
            if evidence is None:
                with read_evidence(file, evidence_identity) as source:
                    result = compute(source)
            else:
                result = compute(evidence)
            diagnostics = to_payload(PaintingDiagnostics(trajectories=tuple(trajectories), census=tuple(events)))
            diagnostics.update(wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu)
            publish(file, records.DIAGNOSTICS, expected, diagnostics)
            records.write(file, expected, result, summarise(result), evidence_identity=evidence_identity)
        mark_complete(file, "painting", complete)
        return result


def working_bytes(assembly: records.Assembly, samples: int, parameters: PaintingParameters, cores: int) -> int:
    """Dimension-only upper bound for cache, output, per-sample E-step and search scratch.

    All panel sites and founder rows bound selected sites and trajectory classes.
    Components execute serially, so only the largest component workspace is live.
    """
    prepared, components = assembly
    peak = 0
    output = 0
    for component in components:
        sites = sum(len(p.positions) for p in prepared[component.start : component.stop])
        founders = len(component.rows)
        states = founders + 1
        tracks = 2 * states - 1
        bins = max(1, sites // parameters.snps_per_bin)
        links = max(0, component.stop - component.start - 1)
        batch = min(samples, cores)
        decoder = batch * (40 * tracks * tracks * bins + 16 * states * states * bins + 24 * sites)
        continuity = relabel_bytes(batch, states, bins, links, sites)
        statistics = 8 * samples * (max(0, bins - 1) + 9)
        panels = 24 * founders * sites + 8 * founders * (links + 1)
        peak = max(peak, max(decoder, continuity) + statistics + panels)
        output += 128 * samples * bins
    return EMISSION_CACHE_MAX_BYTES + (512 << 20) + output + peak


def painting_task(
    contig: str,
    *,
    file: CheckpointFile,
    assembly: records.Assembly,
    genetic_map: ChromosomeGeneticMap,
    parameters: PaintingParameters,
    sample_ids: tuple[str, ...],
    assembly_identity: str,
    evidence_identity: str,
    payload_identity: str,
    map_identity: str,
    ties: TieSettings = _DEFAULT_TIES,
    numerical_policy: NumericalPolicy | None = None,
    evidence: ContigEvidence | None = None,
    dependencies: tuple[TaskKey, ...] = (),
    max_cores: int = 16,
) -> Task:
    """One contig task; engine selects the team, components retain chromosome order."""
    if max_cores < 1:
        raise InvariantError("painting max_cores must be positive")
    expected = identity(
        parameters,
        contig=contig,
        sample_ids=sample_ids,
        assembly_identity=assembly_identity,
        evidence_identity=evidence_identity,
        payload_identity=payload_identity,
        map_identity=map_identity,
        ties=ties,
        numerical_policy=numerical_policy,
    )
    return Task(
        TaskKey("painting", "contig", contig),
        run_contig,
        (file, expected, assembly, genetic_map, parameters),
        kwargs=dict(
            evidence_identity=evidence_identity, payload_identity=payload_identity, evidence=evidence, ties=ties
        ),
        depends_on=dependencies,
        inputs=(RecordKey(contig, "assembly/final"), RecordKey(contig, "evidence")),
        outputs=tuple(RecordKey(contig, key) for key in (records.PAINTING, records.SUMMARY, records.CACHE)),
        resources=ResourceRequest(1, max_cores, working_bytes(assembly, len(sample_ids), parameters, max_cores)),
        affinity=contig,
        thread_argument="threads",
    )
