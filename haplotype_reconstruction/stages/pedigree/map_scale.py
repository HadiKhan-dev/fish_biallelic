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
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import numpy as np

from ...contracts.contigs import AnalysisContigs
from ...contracts.genetic_map import GeneticMapSet
from ...contracts.map_scale import MapScaleEstimate
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.concurrency import threaded_each
from ...core.errors import InputError, InvariantError
from ...engine.cpu import TASK_THREADS, thread_team
from ...engine.graph import Expansion, ResourceRequest, ResultRef, Task, TaskKey
from ...engine.records import record_diagnostic
from ...science.painting.parameters import PaintingParameters
from ...science.pedigree import map_scale as science
from ...science.pedigree import map_scale_frames as frames_science
from ...science.pedigree import map_scale_reads as reads_science
from ...science.pedigree.map_scale_operators import FrameOperators, full_sib
from ...science.pedigree.parameters import PedigreeParameters
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, claim, publish, read_items, resume
from ...store.store import CheckpointStore
from ..framework import load_product, publish_product
from ..relatedness.working_map import map_identity

__all__ = ["KEY", "MapScaleAdapter", "derivation", "expected", "load", "run_stored"]

KEY = "pedigree/map_scale"
INPUTS = "pedigree/map_scale_inputs"
PAIRS = "pedigree/map_scale_pairs"
FRAMES = "pedigree/map_scale_frames"


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
    contigs = AnalysisContigs.from_manifest(manifest).included
    if contig_inputs is None:
        contig_inputs, _ = contig_causes(store, manifest)
    if tuple(contig_inputs) != contigs:
        raise InputError("map-scale inputs differ from the included contig axis")
    return _bound_identity(parameters, numerical_policy, contig_inputs, _family_reader(store, manifest).identity)


def _bound_identity(parameters, numerical_policy, contig_inputs, family_identity):
    upstream: dict[str, str] = {}
    for c, (painted, observed, working_map) in contig_inputs.items():
        upstream[f"painting:{c}"] = painted
        upstream[f"evidence:{c}"] = observed
        upstream[f"map:{c}"] = working_map
    upstream["relatedness/families"] = family_identity
    return derivation(parameters, upstream, numerical_policy=numerical_policy)


def contig_causes(store: CheckpointStore, manifest: Mapping[str, Any]) -> tuple[dict[str, tuple[str, str, str]], str]:
    """Bind repaint, evidence and map headers with one run-wide contamination binding."""
    from ..painting import contamination_records
    from ..refit import records as refit_records

    contigs = AnalysisContigs.from_manifest(manifest).included
    # Contamination binds the original paintings of every contig. Bind that reduction once, outside
    # the contig map, and read repaint identities without their optional emission caches.
    contamination = contamination_records.expected(store, manifest, product="contamination").identity
    refit_records.expected(store, manifest, product="priors")  # warm the shared worker-local refit binding
    causes = threaded_each(partial(contig_cause, store, manifest, contamination=contamination), contigs)
    return dict(zip(contigs, causes, strict=True)), contamination


def contig_cause(store: CheckpointStore, manifest: Mapping[str, Any], contig: str, *,
                 contamination: str) -> tuple[str, str, str]:
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records
    from ..painting import repaint
    from ..refit import records as refit_records

    if not store.contig(contig).has(repaint.REFIT_REFERENCE):
        raise InputError(f"the pedigree requires repaint for {contig}")
    reference = painting_records.expected(store, manifest, product="reference", contig=contig)
    refit = refit_records.expected(store, manifest, product="refit", contig=contig)
    observed = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
    if reference is None or refit is None or observed is None:
        raise InputError(f"the pedigree requires painting, refit and evidence for {contig}")
    painted = repaint.derivation(reference.identity, refit.identity, contamination,
                                 numerical_policy=NumericalPolicy(**manifest["numerical_policy"]))
    return painted, observed.identity, map_identity(store, manifest, contig)


def load(store: CheckpointStore, identity: str) -> MapScaleEstimate:
    value = load_product(store.run, KEY, identity, MapScaleAdapter())
    if value is None:
        raise InputError("the pedigree material rule requires the published map-scale estimate")
    return value


def _component_inputs(store, manifest, contig_index, contig, rows, threads, *,
                      causes=None, contamination=None) -> tuple[list, list, tuple | None]:
    """Restricted component inputs of one contig (``rows``: relevant samples, ascending): without read state with
    their packed emissions (missing ones recomputed from the contig's evidence); with it, the components' frame sites
    for the read source and the painting's (parameters, outlier rate, path correlation)."""
    from ...science.pedigree.prepare import path_correlation
    from ..painting import contamination_records, repaint
    from ..painting import records as painting_records

    file = store.contig(contig)
    if causes is None:
        contamination = contamination_records.expected(store, manifest, product="contamination").identity
        causes = contig_cause(store, manifest, contig, contamination=contamination)
    painting_identity, evidence_identity, _ = causes
    reader = painting_records.Reader(painting_identity, "refit_reference", key=repaint.REFIT_REFERENCE)
    saved = read_items(file, reader.key, reader.identity, reader.items)
    if saved is None:
        raise InputError(f"missing pedigree prerequisite {file.label}:{reader.key}")
    require_schema(saved.payload, reader.schema)
    payload = saved.payload
    # Read-state frames do not use the painting's optional packed emissions.
    # Calibration resolves the painting's parameters, which are stored on the product.
    if int(payload["parameters"]["read_classes"]) == 1:
        cache = read_items(file, reader.key + "/cache", reader.identity, ("schema", "packed_emissions"))
        if cache is not None:
            require_schema(cache.payload, reader.schema)
            payload = dict(payload, packed_emissions=cache.payload["packed_emissions"])
    painting = reader.decode(payload)
    if tuple(painting.sample_ids) != tuple(manifest["sample_ids"]):
        raise InputError("painting sample axis differs from the run's")
    evidence = None
    out, sites = [], []
    # the pedigree reads the painting at its path strength (science.pedigree.prepare.path_correlation)
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
                evidence, _ = _frame_evidence(store, manifest, contig, evidence_identity, contamination)
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


def _frame_evidence(store, manifest, contig, identity, contamination):
    """(the contig's evidence with QC A's corrected likelihood rows, the corrected sample indices)."""
    from ..evidence import records as evidence_records
    from ..painting import contamination as contamination_stage
    from ..painting import contamination_records

    # The frame builders gather the needed samples and sites through the same evidence
    # adapter as painting. Keep coded GLs coded until those gathers.
    evidence = evidence_records.load(store.contig(contig), expected_identity=identity)
    if contamination is not None:
        # The graph already derived the decision identity. The owning stage applies its
        # correction, including the donor painting, whenever any library was flagged.
        saved = read_items(store.run, contamination_records.CONTAMINATION, contamination, ("schema", "flagged"))
        if saved is None:
            raise InvariantError("the run's contamination decision is not published")
        require_schema(saved.payload, contamination_records.SCHEMA)
        if not np.any(saved.payload["flagged"]):
            return evidence, ()
    return contamination_stage.corrected_evidence_with_rows(
        store, manifest, contig, evidence, evidence_identity=identity
    )


def _contig_reads(store, manifest, contig: str, context: tuple, *,
                  evidence_identity=None, contamination=None) -> reads_science.ContigReads:
    """One contig's read-state inputs for the frames: its evidence and the painting's read builder over the
    pedigree's scalar emissions at the path strength (as stages.pedigree.prepare.read_state_builder builds it for
    the source chain; QC A's corrected rows without the uniform foreign class)."""
    from ...science.pedigree.prepare import scalar_emissions
    from ..evidence import records as evidence_records
    from ..painting.read_state import stage_builder

    parameters, robustness_epsilon, site_correlation = context
    file = store.contig(contig)
    if evidence_identity is None:
        evidence_identity = evidence_records.result_reader(store, manifest, product="evidence", contig=contig).identity
    evidence, corrected_rows = _frame_evidence(store, manifest, contig, evidence_identity, contamination)
    cohort = evidence.cohort_frequencies
    builder = stage_builder(partial(scalar_emissions, cohort=cohort), file, evidence_identity, cohort, None,
                            parameters, unit_batches=True, corrected_rows=corrected_rows)
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
    symbols: np.ndarray | FrameOperators
    entry: np.ndarray


def _build_contig(root, manifest, contig, components, sites, context, samples, families, active, guard_pass,
                  batch_bytes, *, threads=1, evidence_identity=None, contamination=None):
    store = CheckpointStore(root, nthreads=threads)
    source = (
        reads_science.ScalarFrames(components) if context is None else
        reads_science.ReadFrames(
            sites, samples, lambda _: _contig_reads(store, manifest, contig, context,
                                                    evidence_identity=evidence_identity, contamination=contamination),
            batch_bytes=batch_bytes,
        )
    )
    pieces = frames_science.frame_pieces(components, families, active, threaded_each)
    # The read law crosses the ordered sites between successive requested bins. Scattering contigs and
    # restricting the emission bins leaves the evidence read and label-independent builder work; on TroMau
    # these frame tasks took 455 CPU s in 159 summed wall s (22 contigs, before removing worker affinity).
    # Famhs's contig frame tasks used 548.8 CPU s over 266.5 summed wall s after rebinding (.770).
    # Independent frames share the contig's site operators and cache; the gap operator uses them in site order.
    source.prepare([(i, rows, bins, labels) for i, _, rows, _, bins, labels in pieces])

    def pack(piece):
        i, family, rows, _, bins, _ = piece
        classes = 1 if context is None else source.class_count(i, rows)
        symbols, entry, offsets = frames_science.piece_emissions(piece, source, classes)
        if not guard_pass:
            symbols = full_sib(symbols)
        return _FamilyFrames(components[i].contig, i, family, rows[offsets[:-1]], offsets,
                             components[i].centers[bins], symbols, entry)

    batches = _concurrent([partial(pack, piece) for piece in pieces])
    statistics = source.statistics()
    record_diagnostic(f"map scale: {contig} {statistics['computed']} frame requests, "
                      f"{statistics['classes']} classes, {statistics['rows_bypassed']} bypassed rows")
    return batches


def _join_frames(batches, guard_pass):
    """Join ordered sequences, embedding an entirely bypassed contig in class zero of a mixed run."""
    frames = frames_science.Frames()
    em25 = []
    classes = max((part.entry.shape[-1] for part in batches), default=1)

    def fit_symbols(part):
        symbols = part.symbols
        if isinstance(symbols, FrameOperators) or symbols.shape[1] != reads_science.FRAME_COLUMNS:
            return symbols
        return full_sib(symbols)

    for part, fit in zip(batches, threaded_each(fit_symbols, batches), strict=True):
        symbols, entry = part.symbols, part.entry
        if entry.shape[1] != classes:
            padded = np.zeros((len(entry), classes))
            padded[:, :entry.shape[1]] = entry
            entry = padded
        frames.contig.extend([part.contig] * len(part.target))
        frames.family.extend([part.family] * len(part.target))
        frames.target.extend(part.target.tolist())
        for a, z in zip(part.offsets[:-1], part.offsets[1:], strict=True):
            frames.positions.append(part.positions[a:z])
            frames.entry.append(entry[a:z])
            fit_sequence = fit[a:z]
            if guard_pass:
                em25.append(fit_sequence if symbols is fit else symbols[a:z])
            frames.emissions.append(fit_sequence)
    return frames, em25


@dataclass(frozen=True)
class _Inputs:
    components: tuple[frames_science.ComponentInput, ...]
    sites: tuple[reads_science.FrameSites, ...]
    context: tuple[PaintingParameters, float, float] | None
    schema: int = 1

    def parts(self):
        return self.components, self.sites, self.context


@dataclass(frozen=True)
class _Frames:
    batches: tuple[_FamilyFrames, ...]
    schema: int = 2


def _family_reader(store, manifest):
    from ..relatedness import families_records

    reader = families_records.result_reader(store, manifest, product="families")
    if reader is None:
        raise InputError("the map-scale estimate requires relatedness/families")
    return reader


def _family_inputs(store, manifest, identity=None):
    from ..relatedness import families_records

    if identity is None:
        identity = _family_reader(store, manifest).identity
    product = families_records.load(store.run, identity)
    if product is None or tuple(product.sample_ids) != tuple(manifest["sample_ids"]):
        raise InputError("relatedness/families sample axis differs from the map-scale estimate's")
    family = np.asarray(product.family, dtype=np.int64)
    parents = np.asarray(product.parents, dtype=np.int64)
    rows = frames_science.relevant_samples(family, parents)
    return rows, frames_science.families_of(family, parents, rows), identity


def _inputs_identity(manifest, index, contig, causes, family_identity):
    painting, evidence, working_map = causes
    return Derivation(
        code_dependency_identity((__name__,)), dict(product=INPUTS, contig=contig, index=index, schema=1),
        NumericalPolicy(**manifest["numerical_policy"]),
        dict(painting=painting, evidence=evidence, map=working_map, families=family_identity),
    ).identity


def _published_inputs(store, contig, identity, *, required=False):
    saved = resume(store.contig(contig), INPUTS, identity)
    if saved is None:
        if required:
            raise InputError(f"{contig}: publish pedigree --stage map-inputs before the map-scale gather")
        return None
    require_schema(saved.payload, 1)
    return from_payload(_Inputs, saved.payload).parts()


def _pair_state(components, families, parameters):
    return frames_science.pair_guard(components, families, parameters.map_scale, threaded_each)


def _publish_pairs(store, identity, state):
    guard = dict(
        active=[[f, sorted(members)] for f, members in state.active.items()],
        excluded_pair=state.excluded_pair,
        pair_tests=list(state.pair_tests.items()),
    )
    publish(store.run, PAIRS, identity, dict(schema=1, guard=guard))


def _read_pairs(store, identity):
    saved = resume(store.run, PAIRS, identity)
    if saved is None:
        raise InputError("publish pedigree --stage map-pairs before building map-scale frames")
    require_schema(saved.payload, 1)
    guard = saved.payload["guard"]
    return frames_science.GuardResult(
        {f: set(members) for f, members in guard["active"]},
        tuple(guard["excluded_pair"]), (), (), {row: tuple(counts) for row, counts in guard["pair_tests"]},
    )


def _read_frames(store, contig, identity, *, required=False):
    saved = resume(store.contig(contig), FRAMES, identity)
    if saved is None:
        if required:
            raise InputError(f"{contig}: publish pedigree --stage map-frames before the map-scale gather")
        return None
    require_schema(saved.payload, 2)
    return from_payload(_Frames, saved.payload).batches


def scatter_inputs(root, manifest, contig, *, threads=1):
    """Publish the same restricted component data the single-node map-scale graph transfers."""
    from ..painting import contamination_records

    store = CheckpointStore(root, nthreads=threads)
    index = AnalysisContigs.from_manifest(manifest).included.index(contig)
    contamination = contamination_records.expected(store, manifest, product="contamination").identity
    causes = contig_cause(store, manifest, contig, contamination=contamination)
    rows, _, family_identity = _family_inputs(store, manifest)
    identity = _inputs_identity(manifest, index, contig, causes, family_identity)
    file = store.contig(contig)
    with claim(file, INPUTS, identity) as needed:
        if needed:
            components, sites, context = _component_inputs(store, manifest, index, contig, rows, threads,
                                                           causes=causes, contamination=contamination)
            publish(file, INPUTS, identity, to_payload(_Inputs(tuple(components), tuple(sites), context)))
    return identity


def scatter_pairs(root, manifest, parameters, policy, *, threads=1):
    """Gather the contig inputs and freeze the active families of the first frame round."""
    store = CheckpointStore(root, nthreads=threads)
    causes, _ = contig_causes(store, manifest)
    family_identity = _family_reader(store, manifest).identity
    identity = _bound_identity(parameters, policy, causes, family_identity)
    with claim(store.run, PAIRS, identity) as needed:
        if needed:
            _, families, _ = _family_inputs(store, manifest, family_identity)
            # Contig decodes share the task's team. Each map item supplies its own live codec allowance.
            readers = CheckpointStore(root, nthreads=TASK_THREADS)
            inputs = threaded_each(
                lambda row: _published_inputs(readers, row[1], _inputs_identity(
                    manifest, row[0], row[1], causes[row[1]], family_identity), required=True),
                enumerate(causes),
            )
            components = [component for part, _, _ in inputs for component in part]
            state = _pair_state(components, families, parameters)
            _publish_pairs(store, identity, state)
    return identity


def scatter_frames(root, manifest, contig, parameters, policy, *, threads=1):
    """Publish the contig's first-round frame emissions from the frozen pair guard."""
    store = CheckpointStore(root, nthreads=threads)
    causes, contamination = contig_causes(store, manifest)
    family_identity = _family_reader(store, manifest).identity
    identity = _bound_identity(parameters, policy, causes, family_identity)
    file = store.contig(contig)
    with claim(file, FRAMES, identity) as needed:
        if needed:
            guard = _read_pairs(store, identity)
            batches = ()
            if any(len(members) >= 2 for members in guard.active.values()):
                rows, families, _ = _family_inputs(store, manifest, family_identity)
                index = tuple(causes).index(contig)
                inputs = _published_inputs(store, contig, _inputs_identity(
                    manifest, index, contig, causes[contig], family_identity), required=True)
                if inputs[0]:
                    share = max(1, reads_science.READ_STATE_BATCH_BYTES // len(causes))
                    batches = tuple(_build_contig(
                        root, manifest, contig, *inputs, rows, families, guard.active,
                        parameters.map_scale.target_iterations > 0, share, threads=threads,
                        evidence_identity=causes[contig][1], contamination=contamination,
                    ))
            publish(file, FRAMES, identity, to_payload(_Frames(batches)))
    return identity


def _frames_stored(root, manifest, contig, inputs, samples, families, active, guard_pass, share, identity,
                   iteration, gather_only, *, threads=1, evidence_identity=None, contamination=None):
    if iteration == 0:
        batches = _read_frames(CheckpointStore(root, nthreads=threads), contig, identity, required=gather_only)
        if batches is not None:
            return batches
    return _build_contig(root, manifest, contig, *inputs, samples, families, active, guard_pass, share,
                         threads=threads, evidence_identity=evidence_identity, contamination=contamination)


def _frame_tasks(root, manifest, maps, parameters, identity, inputs, samples, families, state, iteration,
                 maximum_cores, retained=(), changed=None, *, gather_only=False, causes=None, contamination=None):
    active = state.active if changed is None else {f: state.active[f] for f in changed}
    contigs = AnalysisContigs.from_manifest(manifest).included
    guard_pass = iteration < parameters.map_scale.target_iterations
    # The read builder's per-batch memory budget is shared by these independent contig workers, as it was in
    # ReadFrames.prepare's concurrent pass. Each task loads its own evidence. On TroMau, pinning these
    # reads behind contig preparation spread 159 summed worker-seconds across a 197-second span.
    has_families = any(len(members) >= 2 for members in active.values())
    share = max(1, reads_science.READ_STATE_BATCH_BYTES // max(1, len(contigs)))
    tasks = tuple(
        Task(TaskKey("pedigree", "map_scale.frames", c, str(iteration)), _frames_stored,
             args=(root, manifest, c, inputs[i], samples, families, active, guard_pass, share, identity,
                   iteration, gather_only),
             kwargs=dict(evidence_identity=causes[c][1], contamination=contamination) if causes is not None else {},
             resources=ResourceRequest(max_cores=maximum_cores), thread_argument="threads")
        for i, c in enumerate(contigs) if has_families and inputs[i][0]
    )
    gather = Task(TaskKey("pedigree", "map_scale.guard", round=str(iteration)), _after_frames,
                  args=(tuple(ResultRef(t.key) for t in tasks), root, manifest, maps, parameters, identity,
                        inputs, samples, families, state, iteration, maximum_cores, retained),
                  kwargs=dict(causes=causes, contamination=contamination),
                  resources=ResourceRequest(max_cores=maximum_cores), thread_argument="threads")
    return Expansion(tasks, gather)


def _dense_parts(parts):
    """Each family's frames as dense matrices: a frame record holding structured operators (one written before
    every width was cached at build time) is densified once here with the same construction
    (map_scale_reads.dense_frames), so the guard and the fit never apply the structured action per grid cell."""
    return [replace(part, symbols=reads_science.dense_frames(part.symbols, part.entry.shape[1]))
            if isinstance(part.symbols, FrameOperators) else part for part in parts]


def _after_frames(batches, root, manifest, maps, parameters, identity, inputs, samples, families, state,
                  iteration, maximum_cores, retained, *, threads=1, causes=None, contamination=None):
    p = parameters.map_scale
    guard_pass = iteration < p.target_iterations
    ordered = sorted((*retained, *(part for contig in batches for part in contig)),
                     key=lambda part: (part.contig, part.component, min(state.active[part.family])))
    ordered = _dense_parts(ordered)
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
                                updated, iteration + 1, maximum_cores, keep, changed,
                                causes=causes, contamination=contamination)
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
    gather_only: bool = False,
) -> str | Expansion:
    """Resume or compute the record; returns its identity (the barrier reads the record by it)."""
    store = CheckpointStore(root, nthreads=threads)
    causes, contamination = contig_causes(store, manifest)
    family_identity = _family_reader(store, manifest).identity
    identity = _bound_identity(parameters, numerical_policy, causes, family_identity)
    saved = read_items(store.run, KEY, identity, ("schema",))
    if saved is not None:
        require_schema(saved.payload, MapScaleAdapter.schema)
        return identity
    rows, families, _ = _family_inputs(store, manifest, family_identity)
    contigs = tuple(causes)
    if maximum_cores is not None:
        tasks = tuple(Task(TaskKey("pedigree", "map_scale.inputs", c), _load_contig,
                           args=(root, manifest, i, c, rows,
                                 _inputs_identity(manifest, i, c, causes[c], family_identity), gather_only),
                           kwargs=dict(causes=causes[c], contamination=contamination),
                           resources=ResourceRequest(max_cores=maximum_cores),
                           thread_argument="threads") for i, c in enumerate(contigs))
        gather = Task(TaskKey("pedigree", "map_scale.pairs"), _after_inputs,
                      args=(tuple(ResultRef(t.key) for t in tasks), root, manifest, maps, parameters,
                            identity, families, rows, maximum_cores),
                      kwargs=dict(gather_only=gather_only, causes=causes, contamination=contamination),
                      resources=ResourceRequest(max_cores=maximum_cores), thread_argument="threads")
        return Expansion(tasks, gather)
    readers = store.with_threads(TASK_THREADS)
    inputs = _concurrent([partial(_component_inputs, readers, manifest, k, c, rows, threads,
                                  causes=causes[c], contamination=contamination)
                          for k, c in enumerate(contigs)])
    return _after_inputs(inputs, root, manifest, maps, parameters, identity, families, rows, None,
                         threads=threads, causes=causes, contamination=contamination)


def _load_contig(root, manifest, index, contig, rows, input_identity, gather_only, *, threads=1,
                 causes=None, contamination=None):
    store = CheckpointStore(root, nthreads=threads)
    saved = _published_inputs(store, contig, input_identity, required=gather_only)
    if saved is not None:
        return saved
    return _component_inputs(store, manifest, index, contig, rows, threads,
                             causes=causes, contamination=contamination)


def _after_inputs(inputs, root, manifest, maps, parameters, identity, families, rows, maximum_cores, *,
                  threads=1, gather_only=False, causes=None, contamination=None):
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
        if maximum_cores is not None and (contexts or gather_only or store.run.has(PAIRS)):
            # The target frame requests depend on this pair guard. Its contig/family work is already mapped
            # independently. On TroMau its gather and guard took 5.8 CPU s in 0.48 wall s after rebinding.
            if gather_only or store.run.has(PAIRS):
                state = _read_pairs(store, identity)
            else:
                state = _pair_state(components, families, parameters)
            return _frame_tasks(root, manifest, maps, parameters, identity, inputs, rows, families, state, 0,
                                maximum_cores, gather_only=gather_only, causes=causes, contamination=contamination)
        if contexts:
            source = reads_science.ReadFrames(
                sites, rows, lambda k: _contig_reads(
                    store.with_threads(TASK_THREADS), manifest, contigs[k], contexts[k],
                    evidence_identity=causes[contigs[k]][1] if causes is not None else None,
                    contamination=contamination,
                ), concurrent=_concurrent)
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
