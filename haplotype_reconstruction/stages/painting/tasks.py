"""Run one contig's painting, with sequential components and whole-bin windows.

The engine owns the team and scheduling. Storage windows and cache sizes affect
cost only; evidence includes every sample. A completed product resumes as a unit;
partial model, continuity and sample-block publications resume within the stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configure import ResolvedConfiguration
    from ...contracts.genetic_map import GeneticMapSet

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.observations import gather_observed
from ...contracts.painting import Painting
from ...contracts.panels import materialise
from ...contracts.payload import to_payload
from ...contracts.roles import require_schema
from ...core.errors import InvariantError
from ...core.ties import TieCensus, TieSettings
from ...science.painting.diagnostics import ModelTrajectory, PaintingDiagnostics, summarise
from ...science.painting.emissions import ArraySource, gather_sites, tempered_into
from ...science.painting.fidelity import ChainModel
from ...science.painting.parameters import EMISSION_CACHE_MAX_BYTES, WINDOW_BYTES, PaintingParameters
from .compute import paint_chromosome
from ...store.container import CheckpointFile
from ...store.protocol import claim, mark_complete, publish, read_items, resume
from ...store.slices import WindowArray
from ..evidence.records import SUPPORTED_SCHEMAS
from ..evidence.records import open_arrays as evidence_windows
from ..framework import RecordKey, ResourceRequest, Task
from . import records

_DEFAULT_TIES = TieSettings()

__all__ = ["WindowEvidence", "windowed_emissions", "read_evidence", "run_contig", "task"]


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
    require_schema(payload, SUPPORTED_SCHEMAS)
    with evidence_windows(file, ("likelihoods", "observed"), expected_identity=expected) as arrays:
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
    part_bins = np.arange(len(bin_starts), dtype=np.int64) if parts is None else np.asarray(parts[2], dtype=np.int64)
    if isinstance(likelihoods, WindowArray):
        # Flags are already a returned output. Fill them once before GL work so
        # a large mask chunk cannot evict the GL chunk between every site window.
        flag_sites = max(1, window_bytes)  # One byte per flag; keep its temporary within the existing workspace.
        for sample in range(count):
            for left in range(0, len(sites), flag_sites):
                right = min(len(sites), left + flag_sites)
                flags[sample, left:right] = gather_observed(
                    observed, sites[left:right], slice(start + sample, start + sample + 1)
                )[0]
        sample_windows = likelihoods.sample_tiles(samples, sample_tile)
    else:
        sample_windows = (
            slice(start + first, start + min(count, first + max(1, sample_tile)))
            for first in range(0, count, max(1, sample_tile))
        )
    for sample_window in sample_windows:
        first_sample, last_sample = sample_window.start - start, sample_window.stop - start
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
            if isinstance(likelihoods, WindowArray):
                values = likelihoods.gather(selection, sites[left:right])
                seen = np.ascontiguousarray(flags[first_sample:last_sample, left:right])
            else:
                values, seen = gather_sites(likelihoods, observed, sites[left:right], selection)
            # One kernel tempers the window into the batch's arrays: informative sites, the design
            # effect of each whole bin and the parts' emissions, with no serial steps between them.
            tempered_into(
                values,
                seen,
                alt_probability,
                background_frequency,
                starts,
                stops,
                part_bins,
                bin_starts,
                (first_part, last_part),
                (first_bin, last_bin),
                left,
                config,
                raw,
                flags,
                informative,
                first_sample,
            )
    return raw, flags, informative


class _Progress:
    """Painting progress records in the contig's checkpoint file: the copying
    model, the label-continuity outcome and each decoded sample batch, under keys namespaced by the painting's
    derivation, so an interrupted painting resumes from its last published step and another derivation never
    reads them."""

    def __init__(self, file: CheckpointFile, identity: str) -> None:
        self.file, self.identity = file, identity
        self.prefix = f"painting/progress/{identity[:16]}"

    def load(self, key: str) -> Any | None:
        record = resume(self.file, f"{self.prefix}/{key}", self.identity)
        return None if record is None else record.payload

    def save(self, key: str, payload: Any) -> None:
        publish(self.file, f"{self.prefix}/{key}", self.identity, payload)


def _at_batch_boundary(builder: Callable[..., Any], batch: int, *args: Any, **kwargs: Any) -> Any:
    """A sample batch's emissions, after re-reading this task's CPU share (it grows into free CPUs and gives back
    when peers are short) and applying it to the Numba team of the batch's kernels."""
    from ...engine.cpu import claim_threads

    claim_threads(max_threads=batch)
    return builder(*args, **kwargs)


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
    max_threads: int | None = None,
    batch_samples: int | None = None,
    emission_cache_bytes: int = EMISSION_CACHE_MAX_BYTES,
    census: bool = False,
) -> Painting:
    """Resume or execute model -> continuity -> decode -> release; publish product last."""
    complete = {records.PAINTING: expected, records.SUMMARY: expected, records.IDENTITY: expected}
    with claim(file, records.PAINTING, expected, blocking=False) as needed:
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

                # Sample batches are sized by the most CPUs this task may use (its memory estimate assumes as much), and
                # each batch re-reads the task's share first, so painting grows into CPUs its peers free within the
                # step. Results do not depend on the batch size (per-sample statistics are summed in sample order).
                batch = max(1, max(threads, max_threads or threads) if batch_samples is None else batch_samples)
                return paint_chromosome(
                    materialise(assembly[1], assembly[0]),
                    source,
                    parameters,
                    genetic_map,
                    batch_samples=batch,
                    reverse=ties.reverse,
                    emission_cache_bytes=emission_cache_bytes,
                    model_sink=model_sink,
                    census_sink=census_sink if census or ties.census is not None else None,
                    emission_builder=partial(_at_batch_boundary, partial(windowed_emissions, sample_tile=batch), batch),
                    progress=_Progress(file, expected),
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


from ..inputs.products import task_inputs, required_product
from ..context import StageRun


def run(
    root: str,
    contig: str,
    resolved: ResolvedConfiguration,
    maps: GeneticMapSet,
    ties: TieSettings,
    *,
    team: int = 1,
    cores: int = 1,
) -> str:
    """Bind assembly's checked final assembly to painting's window-capable contig action."""
    from ..assembly import records as assembly_records
    from ..painting.tasks import run_contig
    from ..painting import records
    from ..evidence import records as evidence_records

    reader, store = task_inputs(root)
    assembly = required_product(reader, assembly_records, "assembly", contig)
    evidence_id = evidence_records.expected(store, reader.manifest, product="evidence", contig=contig).identity
    file = store.contig(contig)
    from ...store.protocol import bound_payload_identity

    payload = bound_payload_identity(file)
    expected = records.expected(store, reader.manifest, product="painting", contig=contig).identity
    run_contig(
        file,
        expected,
        (assembly.prepared, assembly.components),
        maps.for_contig(contig),
        resolved.parameters.painting,
        evidence_identity=evidence_id,
        payload_identity=payload,
        ties=ties,
        threads=team,
        max_threads=cores,
        census=ties.census is not None,
    )
    return expected


def task(key, context: StageRun) -> Task:
    """Declare painting's production window-capable contig action."""
    return Task(
        key, run, (context.root, key.scope, context.resolved, context.maps, context.ties),
        kwargs={"cores": context.cores},
        outputs=tuple(RecordKey(key.scope, name) for name in ("painting", "painting/summary", "identity/painting")),
        resources=ResourceRequest(1, context.cores), thread_argument="team",
    )
