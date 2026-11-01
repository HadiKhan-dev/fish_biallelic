"""Certified M1 task phases, pair bounds and the published M2 panel (PS11–PS15).

Scientific child/contig order is explicit. Setup, bound and query tasks have
contig affinity; task transport or reconstruction is value neutral. Pair-bound
reads gather one child's canonical pair slice, never a genome-wide C-by-P stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal

import numpy as np

from ...contracts.payload import from_payload, to_payload
from ...contracts.pedigree import (
    CandidatePanel,
    PairBounds,
    ParentEligibility,
    PreparedChromosome,
    ScreenBatchCounts,
    ScreenRecord,
    ScreenWork,
)
from ...contracts.roles import require_schema
from ...core.errors import InputError, InvariantError
from ...engine.runtime import thread_team
from ...engine.scheduler import TaskBudget, TaskSession, current_budget
from ...engine.workers import ResidentRef, release_resident, resident_value, retain
from ...science.pedigree import forward, m1, pair_bound, panel, selective_m1, selective_screen
from ...science.pedigree.compact_m1 import CompactBuilder, CompactM1
from ...science.pedigree.parameters import PedigreeConfig
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, open_windows
from ..framework import RecordKey, ResourceRequest, Task, TaskKey, load_product, publish_product

SCREEN = "pedigree/selective_screen"
BOUNDS = "pedigree/pair_bounds"
PANEL = "pedigree/panel"


class ScreenAdapter:
    schema = 1

    def encode(self, value: ScreenRecord) -> Mapping[str, Any]:
        return to_payload(value)

    def decode(self, payload: Mapping[str, Any]) -> ScreenRecord:
        return from_payload(ScreenRecord, payload)


class PairBoundsAdapter:
    """Schema-1 storage uses [1,P] so ST1's bounded two-axis windows can stream P.

    The singleton is a storage axis, removed by the adapter. Values are never cast
    or recomputed. This avoids retaining every contig's large bound vector.
    """

    schema = 1

    def encode(self, value: PairBounds) -> Mapping[str, Any]:
        return dict(schema=self.schema, contig=value.contig, bounds=value.bounds.reshape(1, -1))

    def decode(self, payload: Mapping[str, Any]) -> PairBounds:
        require_schema(payload, self.schema)
        values = payload["bounds"]
        if values.ndim != 2 or values.shape[0] != 1 or values.dtype != np.float32:
            raise InputError("pair-bound payload needs float32[1,P]")
        return from_payload(PairBounds, dict(payload, bounds=values[0]))


class PanelAdapter:
    """Schema-1 fixed M2 panel; screen utilities remain report-only."""

    schema = 1

    def encode(self, value: CandidatePanel) -> Mapping[str, Any]:
        return {"schema": self.schema, **to_payload(value)}

    def decode(self, payload: Mapping[str, Any]) -> CandidatePanel:
        require_schema(payload, self.schema)
        names = {f.name for f in fields(CandidatePanel)}
        return from_payload(CandidatePanel, {k: v for k, v in payload.items() if k != "schema" or k in names})


@dataclass
class ContigState:
    """Task-local components and the neutral emission workspace batch limit."""

    components: tuple[selective_screen.Component, ...]
    zero: np.ndarray
    batch: int


def setup(
    prepared: PreparedChromosome,
    eligibility: ParentEligibility,
    parameters: PedigreeConfig,
    *,
    reverse: bool | None = None,
) -> ContigState:
    """Build a contig once; source pair masses are unnecessary for M1 geometry."""
    components = tuple(selective_screen.prepare(prepared, eligibility, parameters, reverse=reverse))
    if not components:
        raise InputError("selective screen requires an informative prepared contig")
    batch = min(
        min(
            c.resident.terms.batch,
            max(
                1,
                forward.COMMON_EMISSION_CACHE_BYTES
                // (32 * c.resident.projection.n_bins * c.resident.projection.n_states**2),
            ),
        )
        for c in components
    )
    zero = np.sum(np.stack([c.zero for c in components]), axis=0)
    return ContigState(components, zero, min(32, batch))


def bound(state: ContigState, children: np.ndarray, eligible: np.ndarray) -> tuple[ContigState, np.ndarray]:
    """Clear all exact caches before this child batch's bound phase."""
    active = np.flatnonzero(eligible.any(1))
    result = np.full(eligible.shape, -np.inf)
    # Even an empty active batch crosses the cache boundary.
    if len(active):
        result[active] = selective_m1.outward_sum(np.stack([c.bound(children[active]) for c in state.components]))
    else:
        for c in state.components:
            c.cache_range = c.products = c.powered = c.columns = None
    return state, result


def query(state: ContigState, pairs: np.ndarray, first: int, stop: int) -> tuple[ContigState, np.ndarray]:
    """Reuse products only for the exact same child range; sum component order."""
    if not len(pairs):
        return state, np.empty(0)
    return state, np.sum(np.stack([c.exact(pairs, first, stop) for c in state.components]), axis=0)


@dataclass(frozen=True)
class ResidentContig:
    """Small setup result; the large state stays in its contig's affinity worker."""

    reference: ResidentRef | ContigState
    zero: np.ndarray
    batch: int


def setup_resident(
    prepared: PreparedChromosome,
    eligibility: ParentEligibility,
    parameters: PedigreeConfig,
    *,
    reverse: bool | None = None,
) -> ResidentContig:
    state = setup(prepared, eligibility, parameters, reverse=reverse)
    return ResidentContig(retain(state, "pedigree-screen"), state.zero, state.batch)


def bound_resident(reference: ResidentRef, children: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Borrow the contig state and return only this batch's bound matrix."""
    return bound(resident_value(reference), children, eligible)[1]


def query_resident(reference: ResidentRef, pairs: np.ndarray, first: int, stop: int) -> np.ndarray:
    """Reuse the worker's current child-range cache without transporting it."""
    return query(resident_value(reference), pairs, first, stop)[1]


class _ThreadPhases:
    """Share contig state inside an executing graph task under its engine budget.

    The enclosing worker already owns an admission. Each phase uses the engine's
    nested thread map, with one Numba thread per admitted contig. No worker pool or
    second TaskSession is opened; returned batch arrays are discarded by the caller.
    """

    def __init__(self, threads: int):
        self.budget = TaskBudget(current_budget(), 1, threads)

    def __enter__(self) -> _ThreadPhases:
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> Literal[False]:
        return False

    @staticmethod
    def execute(task: Task) -> ResidentContig | np.ndarray | None:
        with thread_team(1):
            if task.function is setup_resident:
                state = setup(*task.args, **task.kwargs)
                return ResidentContig(state, state.zero, state.batch)
            if task.function is bound_resident:
                return bound(*task.args, **task.kwargs)[1]
            if task.function is query_resident:
                return query(*task.args, **task.kwargs)[1]
            if task.function is release_resident:
                return None
        raise InvariantError("unexpected nested screen phase")

    def run(self, tasks: Sequence[Task]) -> dict[TaskKey, Any]:
        values = self.budget.thread_map(self.execute, tasks, label="pedigree screen phase")
        return {task.key: value for task, value in zip(tasks, values, strict=True)}


def run(
    prepared: tuple[PreparedChromosome, ...],
    eligibility: ParentEligibility,
    parameters: PedigreeConfig,
    *,
    threads: int = 1,
    reverse: bool | None = None,
    batch_size: int | None = None,
    in_process: bool = True,
    nested: bool = False,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> ScreenRecord:
    """Execute certified queries; neither a memory estimate nor a worker count gates candidates."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    adapter = ScreenAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, SCREEN, identity, adapter)
        if previous is not None:
            return previous
    informative = tuple(p for p in prepared if p.components)
    contigs = tuple(p.contig for p in informative)
    if len(set(contigs)) != len(contigs):
        raise InputError("screen contig axis contains duplicates")
    n = len(eligibility.children)
    markers = np.asarray([p.informative_markers for p in informative], dtype=np.float64)
    builder = CompactBuilder(n, markers, eligibility.parents.sum(1))
    batch_counts = []
    tasks = tuple(
        Task(
            TaskKey("pedigree", "screen.setup", p.contig),
            setup_resident,
            args=(p, eligibility, parameters),
            kwargs=dict(reverse=reverse),
            resources=ResourceRequest(max_cores=threads),
            affinity=p.contig,
        )
        for p in informative
    )
    phases = _ThreadPhases(threads) if nested else TaskSession(threads, in_process=in_process, preload=(__name__,))
    with phases as session:
        results = session.run(tasks)
        states = [results[t.key] for t in tasks]
        zero = np.stack([s.zero for s in states]) if states else np.empty((0, n))
        batch = min((s.batch for s in states), default=max(1, n))
        if batch_size is not None:
            batch = min(batch, max(1, batch_size))
        for first in range(0, n, batch):
            stop = min(n, first + batch)
            children: np.ndarray = np.arange(first, stop, dtype=np.int64)
            allowed = eligibility.parents[first:stop]
            if not states or not np.any(allowed):
                continue
            tasks = tuple(
                Task(
                    TaskKey("pedigree", "screen.bound", c, str(first)),
                    bound_resident,
                    args=(state.reference, children, allowed),
                    resources=ResourceRequest(max_cores=threads),
                    affinity=c,
                )
                for c, state in zip(contigs, states, strict=True)
            )
            results = session.run(tasks)
            upper = np.stack([results[t.key] for t in tasks])
            round_number = 0

            def exact(requests: np.ndarray, first: int = first, stop: int = stop) -> np.ndarray:
                nonlocal round_number
                round_number += 1
                indices = [np.flatnonzero(requests[:, 0] == k) for k in range(len(contigs))]
                work = tuple(
                    Task(
                        TaskKey("pedigree", "screen.query", c, f"{first}.{round_number}"),
                        query_resident,
                        args=(state.reference, requests[ix, 1:], first, stop),
                        resources=ResourceRequest(max_cores=threads),
                        affinity=c,
                    )
                    for c, state, ix in zip(contigs, states, indices, strict=True)
                )
                answers = session.run(work)
                values = np.empty(len(requests))
                for t, ix in zip(work, indices, strict=True):
                    values[ix] = answers[t.key]
                return values

            batch_result = selective_m1.search_batch(upper, children, allowed, markers, parameters, exact)
            builder.add(batch_result)
            batch_counts.append(ScreenBatchCounts(**batch_result["counts"]))
        session.run(
            tuple(
                Task(
                    TaskKey("pedigree", "screen.release", contig),
                    release_resident,
                    args=(state.reference,),
                    affinity=contig,
                )
                for contig, state in zip(contigs, states, strict=True)
            )
        )
    product = ScreenRecord(contigs, builder.finish(), zero, ScreenWork(tuple(batch_counts)))
    if file is not None:
        assert identity is not None
        publish_product(file, SCREEN, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def bound_run(
    prepared: PreparedChromosome,
    compact: CompactM1,
    parameters: PedigreeConfig,
    *,
    threads: int = 1,
    reverse: bool | None = None,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> PairBounds:
    """Evaluate the canonical retained-pair rows and round float32 upwards once."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    adapter = PairBoundsAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, BOUNDS, identity, adapter)
        if previous is not None:
            return previous
    with thread_team(threads):
        product = PairBounds(
            prepared.contig, pair_bound.chromosome_bounds(prepared, compact, parameters, reverse=reverse)
        )
    if file is not None:
        assert identity is not None
        publish_product(file, BOUNDS, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


@dataclass(frozen=True)
class StoredBounds:
    """Identity-checked bounded reads; each request materializes one child slice."""

    files: tuple[CheckpointFile, ...]
    identities: tuple[str, ...]
    pairs: int

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.files), self.pairs

    def __getitem__(self, key: tuple[slice, slice]) -> np.ndarray:
        chromosomes, columns = key
        if chromosomes != slice(None):
            raise InputError("panel requests every configured bound contig in order")
        first, stop, step = columns.indices(self.pairs)
        if step != 1:
            raise InputError("panel bound slices must be contiguous")
        output: np.ndarray = np.empty((len(self.files), max(0, stop - first)), np.float32)
        for k, (file, identity) in enumerate(zip(self.files, self.identities, strict=True)):
            with open_windows(file, BOUNDS, identity, ("bounds",)) as windows:
                if windows is None or windows["bounds"].shape != (1, self.pairs):
                    raise InvariantError("pair-bound record missing or has the wrong canonical row count")
                output[k] = windows["bounds"].gather(slice(0, 1), np.arange(first, stop))[0]
        return output


@dataclass(frozen=True)
class ArrayBounds:
    """The same bounded child slice for already-resident or mapped contig vectors."""

    values: tuple[np.ndarray, ...]
    pairs: int

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.values), self.pairs

    def __getitem__(self, key: tuple[slice, slice]) -> np.ndarray:
        chromosomes, columns = key
        if chromosomes != slice(None):
            raise InputError("panel requires configured contig order")
        return np.stack([v[columns] for v in self.values])


def panel_run(
    screen: ScreenRecord,
    bounds: StoredBounds | ArrayBounds,
    eligibility: ParentEligibility,
    parameters: PedigreeConfig,
    *,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> CandidatePanel:
    """Build and publish the M2 panel, including every cutoff tie and indeterminate sum."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    adapter = PanelAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, PANEL, identity, adapter)
        if previous is not None:
            return previous
    c = screen.compact
    utility = panel.robust_screen_values(c.values, c.n_contigs, c.n_samples, c.markers, eligibility, parameters)
    product, _ = panel.bounded_panel(c, screen.zero, bounds, utility, eligibility, parameters)
    if file is not None:
        assert identity is not None
        publish_product(file, PANEL, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def screen_derivation(
    parameters: PedigreeConfig,
    eligibility_identity: str,
    prepared_identities: Sequence[tuple[str, str]],
    *,
    reverse: bool | None = None,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind every informative contig in order and the fixed M1 floor definition."""
    reverse = parameters.reverse_conventions if reverse is None else reverse
    inputs = {"eligibility": eligibility_identity}
    for contig, identity in prepared_identities:
        if "prepared:" + contig in inputs:
            raise InputError("duplicate contig in screen identity")
        inputs["prepared:" + contig] = identity
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(scoring=parameters.scoring, m1=m1.identity(parameters), contigs=tuple(c for c, _ in prepared_identities)),
        numerical_policy or NumericalPolicy.current(),
        inputs,
        reverse,
    ).identity


def bound_derivation(
    parameters: PedigreeConfig,
    prepared_identity: str,
    screen_identity: str,
    *,
    reverse: bool | None = None,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind the fixed screen, component scoring and one upward float32 rounding."""
    reverse = parameters.reverse_conventions if reverse is None else reverse
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(
            scoring=parameters.scoring,
            pair_bound_high_mass=parameters.panel.pair_bound_high_mass,
            representation="retained-pairs-float32-up-v1",
        ),
        numerical_policy or NumericalPolicy.current(),
        dict(prepared=prepared_identity, screen=screen_identity),
        reverse,
    ).identity


def panel_derivation(
    screen_identity: str,
    bound_identities: Sequence[tuple[str, str]],
    margin: float,
    *,
    reverse: bool = False,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind the screen and every bound identity; resource choices are absent."""
    inputs = {"screen": screen_identity}
    for contig, identity in bound_identities:
        if "bounds:" + contig in inputs:
            raise InputError("duplicate contig in panel identity")
        inputs["bounds:" + contig] = identity
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(pair_relevance_margin=float(margin), contigs=tuple(c for c, _ in bound_identities)),
        numerical_policy or NumericalPolicy.current(),
        inputs,
        reverse,
    ).identity


def task(
    prepared: tuple[PreparedChromosome, ...],
    eligibility: ParentEligibility,
    parameters: PedigreeConfig,
    *,
    cores: int = 1,
    reverse: bool | None = None,
    file: CheckpointFile | None = None,
    identity: str | None = None,
    depends_on: tuple[TaskKey, ...] = (),
) -> Task:
    """Declare the run-wide certified-screen barrier."""
    return Task(
        TaskKey("pedigree", "screen"),
        run,
        args=(prepared, eligibility, parameters),
        kwargs=dict(file=file, identity=identity, reverse=reverse, nested=True),
        depends_on=depends_on,
        inputs=tuple(RecordKey(p.contig, "pedigree/prepared") for p in prepared),
        outputs=(RecordKey("run", SCREEN),),
        resources=ResourceRequest(max_cores=cores),
        thread_argument="threads",
    )


def bound_task(
    prepared: PreparedChromosome,
    compact: CompactM1,
    parameters: PedigreeConfig,
    *,
    cores: int = 1,
    reverse: bool | None = None,
    file: CheckpointFile | None = None,
    identity: str | None = None,
    depends_on: tuple[TaskKey, ...] = (),
) -> Task:
    """One pair-bound task per informative contig."""
    c = prepared.contig
    return Task(
        TaskKey("pedigree", "pair_bounds", c),
        bound_run,
        args=(prepared, compact, parameters),
        kwargs=dict(file=file, identity=identity, reverse=reverse),
        depends_on=depends_on,
        inputs=(RecordKey(c, "pedigree/prepared"), RecordKey("run", SCREEN)),
        outputs=(RecordKey(c, BOUNDS),),
        resources=ResourceRequest(max_cores=cores),
        affinity=c,
        thread_argument="threads",
    )


def panel_task(
    screen: ScreenRecord,
    bounds: StoredBounds | ArrayBounds,
    eligibility: ParentEligibility,
    parameters: PedigreeConfig,
    *,
    file: CheckpointFile | None = None,
    identity: str | None = None,
    depends_on: tuple[TaskKey, ...] = (),
) -> Task:
    """Publish the explicit run panel before M2 score tasks are admitted."""
    return Task(
        TaskKey("pedigree", "panel"),
        panel_run,
        args=(screen, bounds, eligibility, parameters),
        kwargs=dict(file=file, identity=identity),
        depends_on=depends_on,
        inputs=(RecordKey("run", SCREEN),) + tuple(RecordKey(c, BOUNDS) for c in screen.contigs),
        outputs=(RecordKey("run", PANEL),),
    )
