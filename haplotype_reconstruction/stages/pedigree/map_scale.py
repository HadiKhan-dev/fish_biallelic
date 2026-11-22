"""Publish the run's painting-based fallback-rate estimate (record ``pedigree/map_scale``) for the pedigree material rule.

A genome-wide pair test and fit with per-contig read-frame tasks: after every included contig's painting and the relatedness families (the
pedigree node's own prerequisites), before the screen barrier, which reads the record to choose the material regime
(science.pedigree.regimes.material_choice). Each task reads one painting and keeps only the relevant
samples (members and sequenced slots of the families of >= 2). The frames read the painting's emissions at the path
strength: without read state, a component's packed emission cache when it is at that strength, else recomputed from
the contig's evidence for those samples (science.pedigree.map_scale_reads.ScalarFrames); with the read state, the
frames' emissions are rebuilt by the painting's read builder over the pedigree's scalar emissions (the source chain's
builder, stages.pedigree.prepare.read_state_builder) for the requested frames only, the contig's evidence read again
for each such pass (map_scale_reads.ReadFrames). The derivation binds every painting, the evidence, the families
product, the working map identities and the map-scale parameters, so resume follows each cause.

Each guard pass waits for all contigs before testing for target removal. Builder workers read only the frame bins.
The gather uses the original contig, component and family order for every reduction. When the active set stabilises,
the same frames supply the fit. The scalar route uses the painting's packed emissions directly.

Method: docs/methods/pedigree.md#map-scale.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from ...contracts.contigs import AnalysisContigs
from ...contracts.genetic_map import GeneticMapSet
from ...contracts.map_scale import MapScaleEstimate
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.concurrency import threaded_each
from ...core.errors import InputError
from ...engine.cpu import thread_team
from ...engine.graph import Expansion, ResourceRequest, ResultRef, Task, TaskKey
from ...engine.records import record_diagnostic
from ...science.pedigree import map_scale as science
from ...science.pedigree import map_scale_frames as frames_science
from ...science.pedigree import map_scale_reads as reads_science
from ...science.pedigree.parameters import PedigreeParameters
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy
from ...store.store import CheckpointStore
from ..framework import load_product, publish_product
from ..relatedness.working_map import map_identity

__all__ = ["KEY", "MapScaleAdapter", "derivation", "expected", "load", "run_stored"]

KEY = "pedigree/map_scale"


class MapScaleAdapter:
    """Schema-1 plain payload of MapScaleEstimate; decoded arrays are read-only."""

    schema = 1

    def encode(self, value: MapScaleEstimate) -> Mapping[str, Any]:
        return {"schema": self.schema, **to_payload(value)}

    def decode(self, payload: Mapping[str, Any]) -> MapScaleEstimate:
        require_schema(payload, self.schema)
        return from_payload(MapScaleEstimate, {k: v for k, v in payload.items() if k != "schema"})


def derivation(
    parameters: PedigreeParameters, upstream: Mapping[str, str], *, numerical_policy: NumericalPolicy | None = None
) -> str:
    """Bind the estimator's code, its parameters and material unit, and every upstream record identity."""
    material = parameters.material
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(map_scale=parameters.map_scale, unit=(material.unit_morgans, material.tail_morgans),
             schema=MapScaleAdapter.schema),
        numerical_policy or NumericalPolicy.current(),
        dict(upstream),
        parameters.reverse_conventions,
    ).identity


def expected(store: CheckpointStore, manifest: Mapping[str, Any], parameters: PedigreeParameters,
             numerical_policy: NumericalPolicy | None = None, *,
             contig_inputs: Mapping[str, tuple[str, str, str]] | None = None) -> str:
    """Bind the included contigs' painting, evidence and working map plus the families product.

    A caller that already checked those contig dependencies can pass them in configured order.
    """
    from ..evidence import records as evidence_records
    from ..painting import repaint as painting_records  # the painting the pedigree reads (4.21 P)
    from ..relatedness import families_records

    def bind(c: str) -> tuple[str, str, str]:
        painted = painting_records.result_reader(store, manifest, product="refit_reference", contig=c, panels=False)
        observed = evidence_records.result_reader(store, manifest, product="evidence", contig=c)
        if painted is None or observed is None:
            raise InputError(f"the map-scale estimate requires painting and evidence for {c}")
        return painted.identity, observed.identity, map_identity(store, manifest, c)

    contigs = AnalysisContigs.from_manifest(manifest).included
    upstream: dict[str, str] = {}
    if contig_inputs is None:
        contig_inputs = dict(zip(contigs, threaded_each(bind, contigs), strict=True))
    if tuple(contig_inputs) != contigs:
        raise InputError("map-scale inputs differ from the included contig axis")
    for c, (painted, observed, working_map) in contig_inputs.items():
        upstream[f"painting:{c}"] = painted
        upstream[f"evidence:{c}"] = observed
        upstream[f"map:{c}"] = working_map
    family = families_records.result_reader(store, manifest, product="families")
    if family is None:
        raise InputError("the map-scale estimate requires relatedness/families")
    upstream["relatedness/families"] = family.identity
    return derivation(parameters, upstream, numerical_policy=numerical_policy)


def load(store: CheckpointStore, identity: str) -> MapScaleEstimate:
    value = load_product(store.run, KEY, identity, MapScaleAdapter())
    if value is None:
        raise InputError("the pedigree material rule requires the published map-scale estimate")
    return value


def _component_inputs(store, manifest, contig_index, contig, rows, threads) -> tuple[list, list, tuple | None]:
    """Restricted component inputs of one contig (``rows``: relevant samples, ascending): without read state with
    their packed emissions (missing ones recomputed from the contig's evidence); with it, the components' frame sites
    for the read source and the painting's (parameters, outlier rate, path correlation)."""
    from ..evidence import records as evidence_records
    from ..painting import repaint as painting_records  # the painting the pedigree reads (4.21 P)

    from .prepare import _read_request

    file = store.contig(contig)
    # Reading one compatible painting is a store operation. Across TroMau's 22 independent input tasks,
    # 241 summed task wall s used 242 CPU s. On famhs, 77.4 summed wall s used 83.3 CPU s.
    # A contig's frame inputs need its decoded painting. Independent contig reads use their live grants.
    reader = painting_records.result_reader(store, manifest, product="refit_reference", contig=contig, panels=False)
    painting = _read_request(file, reader)
    if tuple(painting.sample_ids) != tuple(manifest["sample_ids"]):
        raise InputError("painting sample axis differs from the run's")
    evidence = None
    out, sites = [], []
    # the pedigree reads the painting at its path strength (science.pedigree.prepare.path_correlation)
    from ...science.pedigree.prepare import path_correlation

    site_correlation = path_correlation(painting)
    reads = int(painting.parameters.read_classes) > 1
    for comp in painting.components:
        if comp.bins.n_bins < 2:
            continue
        if reads:
            # the frames' emissions are rebuilt with the read state (map_scale_reads.ReadFrames)
            sites.append(reads_science.FrameSites(
                contig_index, np.asarray(comp.states.positions), np.asarray(comp.states.alt_probability()),
                np.asarray(comp.states.background_frequency), np.asarray(comp.bins.starts, dtype=np.int64),
                np.asarray(comp.bins.stops, dtype=np.int64), int(comp.states.n_states)))
            out.append(frames_science.component_input(
                contig_index, comp.bins.centers, comp.local_keys, comp.eligible, comp.classes, comp.status,
                int(comp.states.n_classes), int(comp.states.n_states), None, rows))
            continue
        packed = comp.packed_emissions if comp.transitions.site_correlation == site_correlation else None
        if packed is None:
            if evidence is None:
                observed = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
                evidence = evidence_records.load_dense(file, expected_identity=observed.identity)
            packed = frames_science.packed_emissions(
                comp, painting.parameters, painting.robustness_epsilon, evidence, rows, site_correlation,
                map_work=threaded_each,
            )
            local = np.arange(len(rows))
        else:
            local = rows
        out.append(frames_science.component_input(
            contig_index, comp.bins.centers, comp.local_keys, comp.eligible, comp.classes, comp.status,
            int(comp.states.n_classes), int(comp.states.n_states), packed, rows, packed_rows=local))
    context = (painting.parameters, float(painting.robustness_epsilon), site_correlation) if reads else None
    del painting, evidence
    return out, sites, context


class _PhaseClock:
    """Wall and process-CPU seconds of the task's phases, for the run log's diagnostic line (no value reads them)."""

    def __init__(self):
        self.marks = [("start", time.perf_counter(), time.process_time())]

    def mark(self, name: str) -> None:
        self.marks.append((name, time.perf_counter(), time.process_time()))

    def text(self) -> str:
        return ", ".join(f"{b[0]} {b[1] - a[1]:.1f} / {b[2] - a[2]:.1f}" for a, b in zip(self.marks, self.marks[1:]))


def _concurrent(functions: list) -> list:
    """Budgeted contig calls in original order for direct execution outside the graph."""
    from ...engine.cpu import current_budget

    results: list[Any] = [None] * len(functions)
    calls = [lambda share, function=function: function() for function in functions]
    for index, value in current_budget().concurrent(calls, label="map-scale read frames"):
        results[index] = value
    return results


def _contig_reads(store, manifest, contig: str, context: tuple) -> reads_science.ContigReads:
    """One contig's read-state inputs for the frames: its dense evidence and the painting's read builder over the
    pedigree's scalar emissions at the path strength (as stages.pedigree.prepare.read_state_builder builds it for
    the source chain)."""
    from ...science.pedigree.prepare import scalar_emissions
    from ..evidence import records as evidence_records
    from ..painting.read_state import stage_builder

    parameters, robustness_epsilon, site_correlation = context
    file = store.contig(contig)
    observed = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
    evidence = evidence_records.load_dense(file, expected_identity=observed.identity)
    cohort = evidence.cohort_frequencies
    builder = stage_builder(partial(scalar_emissions, cohort=cohort), file, observed.identity, cohort, None,
                            parameters, unit_batches=True)
    return reads_science.ContigReads(evidence, builder, parameters, robustness_epsilon, site_correlation)


@dataclass(frozen=True)
class _FamilyFrames:
    """One guard family's sequences on one component, packed into contiguous graph transport buffers."""

    contig: int
    component: int
    family: int
    target: np.ndarray
    offsets: np.ndarray
    positions: np.ndarray
    symbols: np.ndarray
    entry: np.ndarray


def _build_contig(root, manifest, contig, components, sites, context, samples, families, active, guard_pass,
                  batch_bytes, *, threads=1):
    store = CheckpointStore(root, nthreads=threads)
    source = reads_science.ReadFrames(sites, samples, lambda _: _contig_reads(store, manifest, contig, context),
                                     batch_bytes=batch_bytes)
    pieces = frames_science.frame_pieces(components, families, active, threaded_each)
    # The read law crosses the ordered sites between successive requested bins. Scattering contigs and
    # restricting the emission bins leaves the evidence read and label-independent builder work; on TroMau
    # these frame tasks took 455 CPU s in 159 summed wall s (22 contigs, before removing worker affinity).
    # Famhs's contig frame tasks used 548.8 CPU s over 266.5 summed wall s after rebinding (.770).
    # Independent frames share the contig's site operators and cache; the gap operator uses them in site order.
    source.prepare([(i, rows, bins, labels) for i, _, rows, _, bins, labels in pieces])
    batches = []
    for piece in pieces:
        i, family, rows, _, bins, _ = piece
        symbols, entry, offsets = frames_science.piece_emissions(piece, source, source.class_count(i, rows))
        if not guard_pass:
            symbols = np.ascontiguousarray(symbols[:, frames_science.FULL_SIB_COLUMNS])
        batches.append(_FamilyFrames(components[i].contig, i, family, rows[offsets[:-1]], offsets,
                                     components[i].centers[bins], symbols, entry))
    statistics = source.statistics()
    record_diagnostic(f"map scale: {contig} {statistics['computed']} frame requests, "
                      f"{statistics['classes']} classes, {statistics['rows_bypassed']} bypassed rows")
    return batches


def _join_frames(batches, guard_pass):
    """Join ordered sequences, embedding an entirely bypassed contig in class zero of a mixed run."""
    frames = frames_science.Frames()
    em25 = []
    classes = max((part.entry.shape[-1] for part in batches), default=1)
    for part in batches:
        symbols, entry = part.symbols, part.entry
        if entry.shape[1] != classes:
            if entry.shape[1] != 1:
                raise InputError("frame emissions with different read-class counts")
            padded = np.zeros((*symbols.shape[:2], classes, classes))
            padded[:, :, 0, 0] = symbols[:, :, 0, 0]
            symbols = padded
            entry = np.zeros((len(padded), classes))
            entry[:, 0] = 1.
        frames.contig.extend([part.contig] * len(part.target))
        frames.family.extend([part.family] * len(part.target))
        frames.target.extend(part.target.tolist())
        for a, z in zip(part.offsets[:-1], part.offsets[1:], strict=True):
            frames.positions.append(part.positions[a:z])
            frames.entry.append(entry[a:z])
            if guard_pass:
                em25.append(symbols[a:z])
            if symbols.shape[1] == reads_science.FRAME_COLUMNS:
                frames.emissions.append(np.ascontiguousarray(symbols[a:z][:, frames_science.FULL_SIB_COLUMNS]))
            else:
                frames.emissions.append(symbols[a:z])
    return frames, em25


def _frame_tasks(root, manifest, maps, parameters, identity, inputs, samples, families, state, iteration,
                 maximum_cores, retained=(), changed=None):
    active = state.active if changed is None else {f: state.active[f] for f in changed}
    contigs = AnalysisContigs.from_manifest(manifest).included
    guard_pass = iteration < parameters.map_scale.target_iterations
    # The read builder's per-batch memory budget is shared by these independent contig workers, as it was in
    # ReadFrames.prepare's concurrent pass. Each task loads its own evidence. On TroMau, pinning these
    # reads behind contig preparation spread 159 summed worker-seconds across a 197-second span.
    has_families = any(len(members) >= 2 for members in active.values())
    share = max(1, reads_science.READ_STATE_BATCH_BYTES // max(1, len(contigs)))
    tasks = tuple(
        Task(TaskKey("pedigree", "map_scale.frames", c, str(iteration)), _build_contig,
             args=(root, manifest, c, *inputs[i], samples, families, active, guard_pass, share),
             resources=ResourceRequest(max_cores=maximum_cores), thread_argument="threads")
        for i, c in enumerate(contigs) if has_families and inputs[i][0]
    )
    gather = Task(TaskKey("pedigree", "map_scale.guard", round=str(iteration)), _after_frames,
                  args=(tuple(ResultRef(t.key) for t in tasks), root, manifest, maps, parameters, identity,
                        inputs, samples, families, state, iteration, maximum_cores, retained),
                  resources=ResourceRequest(max_cores=maximum_cores), thread_argument="threads")
    return Expansion(tasks, gather)


def _after_frames(batches, root, manifest, maps, parameters, identity, inputs, samples, families, state,
                  iteration, maximum_cores, retained, *, threads=1):
    p = parameters.map_scale
    guard_pass = iteration < p.target_iterations
    ordered = sorted((*retained, *(part for contig in batches for part in contig)),
                     key=lambda part: (part.contig, part.component, min(state.active[part.family])))
    # The next guard round depends on this genome-wide decision. The likelihood grids inside the guard
    # parallelise sequences and rates. The ordered frame gather, dependent guard and fit took 38.2 CPU s
    # in 5.9 wall s on TroMau and 220.3 CPU s in 21.6 wall s on famhs in the rebind replay.
    frames, em25 = _join_frames(ordered, guard_pass)
    if guard_pass:
        updated = frames_science.target_guard(frames, em25, state, iteration, p)
        if updated.active != state.active:
            changed = {f for f in state.active if updated.active[f] != state.active[f]}
            # Target removal changes that family's carriers only. On tl, rebuilding all eight contigs' families
            # on the second pass cost 3,097 CPU-s in the two passes together; reuse other families' exact frames.
            keep = tuple(part for part in ordered if part.family not in changed)
            return _frame_tasks(root, manifest, maps, parameters, identity, inputs, samples, families,
                                updated, iteration + 1, maximum_cores, keep, changed)
        state = updated
    return _finish(CheckpointStore(root, nthreads=threads), identity, maps, parameters,
                   AnalysisContigs.from_manifest(manifest).included, frames, state)


def run_stored(
    root: str,
    manifest: Mapping[str, Any],
    maps: GeneticMapSet,
    parameters: PedigreeParameters,
    numerical_policy: NumericalPolicy,
    *,
    threads: int = 1,
    maximum_cores: int | None = None,
) -> str | Expansion:
    """Resume or compute the record; returns its identity (the barrier reads the record by it)."""
    from ..relatedness import families_records

    store = CheckpointStore(root, nthreads=threads)
    # Resume and family input depend on the complete upstream identity. The contig reads run independently.
    # Removing repeated refit binding cut this TroMau task from 114.6 wall / 166.7 CPU s to 17.7 / 23.8 s.
    # The remaining family read and the single derivation follow the checked contig results.
    identity = expected(store, manifest, parameters, numerical_policy)
    adapter = MapScaleAdapter()
    if load_product(store.run, KEY, identity, adapter) is not None:
        return identity
    family_reader = families_records.result_reader(store, manifest, product="families")
    families_product = families_records.load(store.run, family_reader.identity)
    if families_product is None or tuple(families_product.sample_ids) != tuple(manifest["sample_ids"]):
        raise InputError("relatedness/families sample axis differs from the map-scale estimate's")
    family = np.asarray(families_product.family, dtype=np.int64)
    parents = np.asarray(families_product.parents, dtype=np.int64)
    rows = frames_science.relevant_samples(family, parents)
    families = frames_science.families_of(family, parents, rows)
    contigs = AnalysisContigs.from_manifest(manifest).included
    if maximum_cores is not None:
        tasks = tuple(Task(TaskKey("pedigree", "map_scale.inputs", c), _load_contig,
                           args=(root, manifest, i, c, rows), resources=ResourceRequest(max_cores=maximum_cores),
                           thread_argument="threads") for i, c in enumerate(contigs))
        gather = Task(TaskKey("pedigree", "map_scale.pairs"), _after_inputs,
                      args=(tuple(ResultRef(t.key) for t in tasks), root, manifest, maps, parameters,
                            identity, families, rows, maximum_cores),
                      resources=ResourceRequest(max_cores=maximum_cores), thread_argument="threads")
        return Expansion(tasks, gather)
    inputs = _concurrent([partial(_component_inputs, store, manifest, k, c, rows, threads)
                          for k, c in enumerate(contigs)])
    return _after_inputs(inputs, root, manifest, maps, parameters, identity, families, rows, None, threads=threads)


def _load_contig(root, manifest, index, contig, rows, *, threads=1):
    return _component_inputs(CheckpointStore(root, nthreads=threads), manifest, index, contig, rows, threads)


def _after_inputs(inputs, root, manifest, maps, parameters, identity, families, rows, maximum_cores, *, threads=1):
    store = CheckpointStore(root, nthreads=threads)
    contigs = AnalysisContigs.from_manifest(manifest).included
    clock = _PhaseClock()
    with thread_team(default=threads):
        components, sites, contexts = [], [], {}
        for k, (part, part_sites, context) in enumerate(inputs):
            components.extend(part)
            sites.extend(part_sites)
            if context is not None:
                contexts[k] = context
        clock.mark("collect inputs")
        if contexts and len(sites) != len(components):
            raise InputError("the map-scale estimate needs every contig's painting with or without the read state")
        if contexts and maximum_cores is not None:
            # The target frame requests depend on this pair guard. Its contig/family work is already mapped
            # independently. On TroMau its gather and guard took 5.8 CPU s in 0.48 wall s after rebinding.
            state = frames_science.pair_guard(components, families, parameters.map_scale, threaded_each)
            return _frame_tasks(root, manifest, maps, parameters, identity, inputs, rows, families, state, 0,
                                maximum_cores)
        if contexts:
            source = reads_science.ReadFrames(
                sites, rows, lambda k: _contig_reads(store, manifest, contigs[k], contexts[k]), concurrent=_concurrent)
        else:
            source = reads_science.ScalarFrames(components)
        p = parameters.map_scale
        # (component, family) items of the pair test and the frames: one CPU and one Numba thread each
        guard = frames_science.guard(components, families, p, source, threaded_each)
        clock.mark("guard")
        final, _ = frames_science.build_frames(components, families, guard.active, False, source, threaded_each)
        clock.mark("final frames")
        statistics = source.statistics()
        del components, source
        if contexts:
            record_diagnostic(
                f"map scale: frames with the read state, {statistics['classes']} classes; target rows per component "
                f"with read state {statistics['rows_with_reads']}, bypassed {statistics['rows_bypassed']}; "
                f"{statistics['computed']} of {statistics['requests']} frame requests computed in "
                f"{statistics['loads']} contig loads")
        record_diagnostic(f"map scale: phases (wall s / process CPU s) {clock.text()}; {len(final)} frame sequences")
        return _finish(store, identity, maps, parameters, contigs, final, guard)


def _finish(store, identity, maps, parameters, contigs, final, guard):
    p = parameters.map_scale
    chromosome_maps = {k: maps.for_contig(c) for k, c in enumerate(contigs)}
    free = tuple(c for c in contigs if not maps.for_contig(c).has_map)
    filed = tuple(c for c in contigs if maps.for_contig(c).has_map)
    free_index = {k for k, c in enumerate(contigs) if c in free}
    decision_frames = final.subset(np.asarray([c in free_index for c in final.contig], dtype=np.bool_))
    value = science.fit(decision_frames, chromosome_maps, p, parameters.material) if free else None
    file_value = None
    if filed:
        file_frames = frames_science.file_frames(final, chromosome_maps, set(free_index))
        file_value = science.fit(file_frames, None, p, parameters.material, positions_scale=1e6)
    record = science.estimate_record(value, file_value, guard, p, working_rate=float(maps.default_rate_cm_per_mb),
                                     map_free=free, file_contigs=filed)
    publish_product(store.run, KEY, identity, record, MapScaleAdapter())
    return identity
