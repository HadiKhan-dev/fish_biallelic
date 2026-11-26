"""Declare one family-phase component task with its true upstream record dependencies.

Owns the engine boundary and the explicit storage-route choice, never science.
Components run in contig order. Independent solves, sample releases and hard-cut
release intervals share the task's CPU authority. Memory estimates are diagnostics
and do not choose a reduction order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...contracts.genetic_map import GeneticMapSet
    from ...contracts.painting import ComponentPainting
    from ...science.painting.parameters import PaintingParameters
    from ...science.painting.read_decode import ReadEmissions
    from ...science.painting.read_inputs import ReadEmissionBuilder
    from ..configure import ResolvedConfiguration

from collections.abc import Callable, Iterable
from functools import cache, partial
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ...core.concurrency import threaded_each
from ...contracts.evidence import ContigEvidence
from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.painting import Painting
from ...contracts.pedigree import ParentEdges
from ...contracts.phase import PhasedSamples, PhaseInputIdentities, PhaseObservations
from ...contracts.refit import FounderRefit
from ...science.phase.component import PhaseMemo, PhaseWorkspace
from ...science.phase.parameters import PhaseParameters
from ...science.phase.read_emissions import apply_read_matrices
from ...science.selection.parameters import LANE_WIDTH
from ...store.container import CheckpointFile
from ...store.protocol import Derivation
from ..context import StageRun, default_scratch
from ..framework import RecordKey, ResourceRequest, Task, TaskKey
from ..inputs.products import required_dense_evidence, required_product, task_inputs
from .boundaries import validate_parent_edges, validate_sample_order

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["execute_contig", "publish_contig", "task"]


RESIDENT_EVIDENCE_BYTES = 8 << 30


def use_elastic_releases(observed_rows: int) -> bool:
    """Reserve elastic sample children for cohorts fitting one SIMD lane block.

    Larger cohorts already supply many independent releases; reusable map workers
    amortise their short calls without per-sample thread admission.
    """
    return observed_rows <= LANE_WIDTH


def use_windows(samples: int, sites: int, storage: str = "auto") -> bool:
    """Select the native capability threshold from decoded dimensions only (FP22)."""
    from ...core.errors import SettingsError

    if storage not in ("auto", "resident", "windowed"):
        raise SettingsError(f"invalid phase storage route {storage!r}")
    return 24 * samples * sites > RESIDENT_EVIDENCE_BYTES if storage == "auto" else storage == "windowed"


def _read_masks(likelihoods, observed, sites, samples, *_args, cohort, robustness_epsilon, **_kwargs):
    """Supply painting's read builder with its masks, reusing phase's existing scalar scores."""
    from ...science.painting.api import gather_sites, informative_sites, mix_window

    gl, flags = gather_sites(likelihoods, observed, sites, samples)
    mixed = mix_window(gl, cohort[sites], robustness_epsilon)
    return np.empty(0), flags, informative_sites(mixed, flags)


def read_state_builder(
    painting: Painting,
    evidence: ContigEvidence,
    file: CheckpointFile,
    evidence_identity: str,
    *,
    read_setup: Callable[[], dict | None] | None = None,
    corrected_rows: tuple[int, ...] = (),
) -> Callable[[], ReadEmissionBuilder] | None:
    """Make a lazy contig read builder using the effective parameters published by painting. ``corrected_rows``: QC
    A's corrected samples of ``evidence``, which carry no uniform foreign class (stages.painting.read_state)."""
    if painting.parameters.read_classes == 1:
        return None

    @cache
    def build() -> ReadEmissionBuilder:
        from ..painting.read_state import stage_builder

        cohort = evidence.cohort_frequencies
        base = partial(_read_masks, cohort=cohort)
        builder = stage_builder(
            base, file, evidence_identity, cohort, None, painting.parameters, unit_batches=True,
            read_setup=None if read_setup is None else read_setup(), corrected_rows=corrected_rows,
        )
        # The native kernels run one sample-parallel batch and write the parts once. This saves CPU by
        # removing per-sample Python dispatch and the copy into a second batch.
        builder.map_work = None
        # Phase consumes each batch once. Keep the builder's table cache and skip its reusable site-parts cache.
        builder.cache_bytes = 0
        return builder

    return build


def _calibrated_read_setup(store, manifest, contig):
    """Read the fitted library setup bound by the painting's run calibration."""
    from ...contracts.roles import require_schema
    from ...store.protocol import read_items
    from ..painting import calibration_records

    request = calibration_records.expected(store, manifest, product="robustness", contig=contig)
    saved = read_items(store.contig(contig), request.key, request.identity, ("schema", "read_setup"))
    if saved is None:
        return None
    require_schema(saved.payload, request.schema)
    return saved.payload.get("read_setup")


def _build_read_matrices(
    builder: ReadEmissionBuilder,
    evidence: ContigEvidence,
    painted: ComponentPainting,
    sites: np.ndarray,
    samples: slice,
    alt_probability: np.ndarray,
    parameters: PaintingParameters,
    robustness_epsilon: float,
) -> ReadEmissions:
    """Build once at painting's effective share and the component's confidence strength."""
    bins = painted.bins
    read, _, _ = builder(
        evidence.likelihoods, evidence.observed, sites, samples, alt_probability,
        painted.states.background_frequency, bins.starts, bins.stops, parameters,
        robustness_epsilon=robustness_epsilon, site_correlation=painted.transitions.site_correlation,
    )
    return read


def execute_contig(
    evidence: ContigEvidence,
    painting: Painting,
    refit: FounderRefit,
    edges: ParentEdges,
    genetic_map: ChromosomeGeneticMap,
    parameters: PhaseParameters,
    identities: PhaseInputIdentities,
    *,
    storage: str = "auto",
    root: str | Path | None = None,
    memo: PhaseMemo | None = None,
    read_builder: Callable[[], ReadEmissionBuilder] | None = None,
    corrected_rows: tuple[int, ...] = (),
) -> PhasedSamples:
    """Check canonical boundaries, then compute every component and assemble a complete contig."""
    import numpy as np

    from ...core.errors import InvariantError
    from ...engine.cpu import claim_threads, current_budget
    from ...science.phase.component import phase_contig
    from ...science.phase.window import site_windows
    from ...store.mapped import discard_pages, mapped_array
    from .parameters import PHASE_RELEASE_BATCH
    from .records import canonical_component_inputs, validate_canonical_inputs

    validate_canonical_inputs(evidence, painting, refit, edges, genetic_map, identities)
    windowed = use_windows(len(evidence.sample_ids), len(evidence.positions), storage)
    if windowed and root is None:
        raise InvariantError("windowed phase requires a task-owned scratch root")
    components = tuple(
        canonical_component_inputs(
            evidence, painting, refit, edges, genetic_map, parameters, component=index, identities=identities
        )
        for index in range(len(painting.components))
    )
    # Empty contigs still validate independent upstream sample axes and pedigree.
    validate_sample_order(
        evidence.sample_ids, painting=painting.sample_ids, refit=refit.sample_ids, pedigree=edges.sample_ids
    )
    validate_parent_edges(edges)
    if len(painting.components) != len(refit.components):
        raise InvariantError("phase painting/refit component counts differ")

    def allocate(shape: tuple[int, ...], fill: int, *, dtype: Any) -> np.ndarray:
        return mapped_array(shape, dtype, fill, root=root)

    budget = current_budget()

    def prepare_likelihoods(observations: PhaseObservations, indices: np.ndarray, site_bins: np.ndarray) -> np.ndarray:
        shape = (len(observations.sample_ids), len(indices), 3)
        likelihoods = observations.likelihoods
        if windowed:
            output = mapped_array(shape, np.float32, root=root)
            windows = tuple(site_windows(site_bins))

            def row(sample: int) -> None:
                for left, right in windows:
                    output[sample, left:right] = likelihoods[sample, indices[left:right]]
                discard_pages(output[sample])
        else:
            output = np.empty(shape, dtype=np.float32)

            def row(sample: int) -> None:
                output[sample] = likelihoods[sample, indices]

        budget.thread_map(row, range(shape[0]), label="family-phase likelihood rows")
        return output

    def map_work(function: Callable[[T], R], items: Iterable[T]) -> list[R]:
        return budget.thread_map(function, items, label="family-phase independent work")

    def concurrent(functions: list[Callable[[Callable[..., list[Any]]], R]]) -> list[R]:
        """Each function(map) on a live share; map runs its independent blocks or
        hard-cut release intervals with one thread each. Return results in order."""

        def on_share(function: Callable[[Callable[..., list[Any]]], R]) -> Callable[[Any], R]:
            def run(share: Any) -> R:
                return function(lambda work, items: share.thread_map(work, items, label="family-phase child work"))

            return run

        results: list[Any] = [None] * len(functions)
        for index, value in budget.concurrent([on_share(f) for f in functions], label="family-phase children"):
            results[index] = value
        return results

    def apply_read_state(component: int, emissions: np.ndarray) -> None:
        assert read_builder is not None
        builder = read_builder()
        if builder.unshared.all():
            return
        painted = painting.components[component]
        states, bins = painted.states, painted.bins
        sites = np.searchsorted(evidence.positions, states.positions)
        alt_probability = states.alt_probability()
        start = 0
        while start < len(emissions):
            # The confidence action and one-shot site parts fit painting's batch memory bound.
            # The live grant sets the team width, so smaller shares still amortise library tables.
            claim_threads()
            width = builder.batch_samples(sites, states.n_states, bins.n_bins, False, len(emissions) - start)
            stop = min(len(emissions), start + width)
            samples = slice(start, stop)
            bypass = builder._bypassed(np.arange(start, stop), int(sites[0]), int(sites[-1]) + 1)
            if not bypass.all():
                read = _build_read_matrices(
                    builder, evidence, painted, sites, samples, alt_probability, painting.parameters,
                    painting.robustness_epsilon,
                )
                claim_threads()
                apply_read_matrices(emissions[samples], read, bypass)
                del read
            start = stop
        claim_threads()

    claim_threads()
    return phase_contig(
        evidence.contig,
        evidence.sample_ids,
        evidence.positions,
        components,
        parameters,
        edges,
        PhaseWorkspace(
            windowed=windowed,
            scores=False,
            memo=memo,
            release_batch=PHASE_RELEASE_BATCH,
            prepare_likelihoods=prepare_likelihoods,
            map_work=map_work,
            apply_read_state=apply_read_state if read_builder is not None else None,
            corrected_rows=corrected_rows,
            concurrent=concurrent,
            concurrent_releases=concurrent if use_elastic_releases(evidence.observed.shape[0]) else None,
        ),
        allocate=allocate if windowed else None,
    )


def publish_contig(
    file: CheckpointFile,
    evidence: ContigEvidence,
    painting: Painting,
    refit: FounderRefit,
    edges: ParentEdges,
    genetic_map: ChromosomeGeneticMap,
    parameters: PhaseParameters,
    identities: PhaseInputIdentities,
    expected: Derivation,
    *,
    read_builder: Callable[[], ReadEmissionBuilder] | None = None,
    corrected_rows: tuple[int, ...] = (),
    storage: str = "auto",
    root: str | Path | None = None,
) -> PhasedSamples:
    """Validate external identities before resume; write completion after all scientific records."""
    from ...core.errors import InvariantError
    from ...engine.records import record_diagnostic
    from ...store.protocol import mark_complete
    from . import records

    records.validate_canonical_inputs(evidence, painting, refit, edges, genetic_map, identities)
    actual = records.derivation(
        evidence.contig,
        evidence.sample_ids,
        identities,
        parameters,
        reverse=expected.tie_reverse,
        code=expected.code,
        policy=expected.numerical_policy,
        included_contigs=expected.parameters.get("included_contigs", ()),
    )
    if actual.identity != expected.identity:
        raise InvariantError("phase task inputs differ from its supplied derivation")
    memo = records.PhaseMemo(file, expected)
    saved = records.load(file, expected)
    if saved is not None:
        mark_complete(file, records.KEY, {records.KEY: expected.identity})
        memo.remove()
        record_diagnostic(f"{evidence.contig}: family phase reused 0 intermediate units; final record resumed")
        return saved
    product = execute_contig(
        evidence, painting, refit, edges, genetic_map, parameters, identities, storage=storage, root=root, memo=memo,
        read_builder=read_builder, corrected_rows=corrected_rows,
    )
    records.write(file, expected, product)
    mark_complete(file, records.KEY, {records.KEY: expected.identity})
    if records.load(file, expected) is None:
        raise InvariantError("completed phase record is not readable")
    memo.remove()
    record_diagnostic(f"{evidence.contig}: family phase reused {memo.reused} intermediate units")
    return product


def _load_phase_inputs(loaders):
    """Decode independent records on the contig share, one codec thread per reader."""
    return threaded_each(lambda load: load(), loaders)


def run(root: str, contig: str, resolved: ResolvedConfiguration, maps: GeneticMapSet, scratch: str | None) -> str:
    """Bind the canonical typed upstream products to family phase's complete-contig publisher."""
    from ...contracts.phase import PhaseInputIdentities
    from ...contracts.roles import require_schema
    from ...core.digest import json_digest
    from ...store.protocol import is_complete, read_items
    from ..painting import records as painting_records
    from ..pedigree import records as pedigree_records
    from ..phase import records
    from ..phase.tasks import publish_contig
    from ..refit import records as refit_records
    from ..relatedness.working_map import working_maps

    reader, store = task_inputs(root)
    expected = records.expected(store, reader.manifest, product="phase", contig=contig)
    file = store.contig(contig)
    # Resume before decoding any input: a current saved record needs none of them. Decoding first cost ~2 min per
    # TroMau contig on the final run of a scattered workflow (4.22 release validation). Same branch as publish_contig.
    saved = read_items(file, records.KEY, expected.identity, ("schema",))
    if saved is not None:
        from ...engine.records import record_diagnostic
        from ...store.protocol import mark_complete

        require_schema(saved.payload, records.PhasedSamplesAdapter.schema)
        if not is_complete(file, {records.BINDING_KEY: expected.identity}):
            records.bind(file, expected)
        if not file.has("complete/" + records.KEY):
            mark_complete(file, records.KEY, {records.KEY: expected.identity})
        records.PhaseMemo(file, expected).remove()
        record_diagnostic(f"{contig}: family phase reused 0 intermediate units; final record resumed")
        return expected.identity
    evidence, painting, refit, pedigree = _load_phase_inputs((
        partial(required_dense_evidence, reader, contig),
        partial(required_product, reader, painting_records, "painting", contig),
        partial(required_product, reader, refit_records, "refit", contig),
        partial(required_product, reader, pedigree_records, "pedigree", None),
    ))
    # 4.22 QC A: flagged libraries phased on contamination-corrected likelihoods. The decision is a function of every
    # contig's painting and evidence, which the phase identity binds through the refit's run priors, and the
    # correction's code is in the phase task's code closure
    from ..painting import contamination as contamination_stage

    identities = PhaseInputIdentities(**expected.inputs)
    evidence, corrected_rows = contamination_stage.corrected_evidence_with_rows(
        store, reader.manifest, contig, evidence, evidence_identity=identities.evidence,
    )
    publish_contig(
        file,
        evidence,
        painting,
        refit,
        pedigree.edges,
        working_maps(store, reader.manifest, maps).for_contig(contig),
        resolved.parameters.family_phase,
        identities,
        expected,
        read_builder=read_state_builder(
            painting, evidence, file, identities.evidence,
            read_setup=partial(_calibrated_read_setup, store, reader.manifest, contig), corrected_rows=corrected_rows,
        ),
        corrected_rows=corrected_rows,
        root=default_scratch(Path(root)) / "phase" / json_digest(contig)[:16]
        if scratch is None
        else Path(scratch) / "phase" / json_digest(contig)[:16],
    )
    return expected.identity


def task(key: TaskKey, context: StageRun) -> Task:
    """Declare the production complete-contig phase publisher."""
    from ...store.store import CheckpointStore
    from .records import KEY

    existing = CheckpointStore(context.root).contig(key.scope).has(KEY)
    return Task(
        key,
        run,
        (context.root, key.scope, context.resolved, context.maps, context.scratch),
        outputs=(RecordKey(key.scope, KEY),),
        resources=ResourceRequest(1, 1 if existing else context.cores),
    )
