"""Certified M1 task phases, pair bounds and the published M2 panel (PS11–PS15).

Scientific child/contig order is explicit. Setup, bound and query tasks have
contig affinity; task transport or reconstruction is value neutral. Pair-bound
reads gather one child's canonical pair slice, never a genome-wide C-by-P stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import partial
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
from ...engine.cpu import claim_threads, current_budget, thread_team
from ...engine.scheduler import TaskSession
from ...engine.workers import ResidentRef, release_resident, resident_value, retain
from ...science.pedigree import forward, m1, pair_bound_screen, panel, selective_m1, selective_screen
from ...science.pedigree.blocks import ScorePlacement
from ...science.pedigree.compact_m1 import CompactBuilder, CompactM1
from ...science.pedigree.parameters import PedigreeParameters
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, open_windows
from ..framework import ResourceRequest, Task, TaskKey, load_product, publish_product

SCREEN = "pedigree/selective_screen"
BOUNDS = "pedigree/pair_bounds"
PANEL = "pedigree/panel"


class ScreenAdapter:
    schema = 1

    def encode(self, value: ScreenRecord) -> Mapping[str, Any]:
        return to_payload(value)

    def decode(self, payload: Mapping[str, Any]) -> ScreenRecord:
        return from_payload(ScreenRecord, payload)


@dataclass(frozen=True)
class RankedBounds(PairBounds):
    """Optional scored pieces published in the same record as their ranking keys."""

    pieces: pair_bound_screen.RankedPieces | None = None


class PairBoundsAdapter:
    """Schema-1 storage uses [1,P] so the store protocol's bounded two-axis windows can stream P.

    The singleton is a storage axis, removed by the adapter. Values are never cast
    or recomputed. This avoids retaining every contig's large bound vector.
    """

    schema = 1

    def encode(self, value: PairBounds) -> Mapping[str, Any]:
        payload = dict(schema=self.schema, contig=value.contig, bounds=value.bounds.reshape(1, -1))
        if isinstance(value, RankedBounds) and value.pieces is not None:
            parts = value.pieces
            payload.update(ranked_trios=parts.trios, ranked_cuts=parts.cuts, ranked_values=parts.values)
        return payload

    @staticmethod
    def decode_pieces(payload: Mapping[str, Any]) -> pair_bound_screen.RankedPieces | None:
        if "ranked_values" not in payload:
            return None
        trios, cuts, values = (payload[f"ranked_{name}"] for name in ("trios", "cuts", "values"))
        if trios.ndim != 2 or trios.shape[1] != 3 or cuts.ndim != 2 or cuts.shape[1] != 3:
            raise InputError("invalid ranked trio or piece coordinates")
        if values.shape != (len(trios), len(cuts)):
            raise InputError("invalid ranked piece dimensions")
        return pair_bound_screen.RankedPieces(trios, cuts, values)

    def decode(self, payload: Mapping[str, Any]) -> PairBounds:
        require_schema(payload, self.schema)
        values = payload["bounds"]
        if values.ndim != 2 or values.shape[0] != 1 or values.dtype != np.float32:
            raise InputError("pair-bound payload needs float32[1,P]")
        pieces = self.decode_pieces(payload)
        if pieces is not None:
            return RankedBounds(payload["contig"], values[0], pieces=pieces)
        return PairBounds(payload["contig"], values[0])


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


def _reread_share() -> None:
    """The screen's kernel boundaries: re-read this task's CPU share (engine claim_threads; it grows as peers
    finish)."""
    claim_threads()


def setup(
    prepared: PreparedChromosome,
    eligibility: ParentEligibility,
    parameters: PedigreeParameters,
    *,
    reverse: bool | None = None,
) -> ContigState:
    """Build a contig once; source pair masses are unnecessary for M1 geometry."""
    components = tuple(
        selective_screen.prepare(prepared, eligibility, parameters, claim=_reread_share, reverse=reverse)
    )
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
        span = (int(children[0]), int(children[-1]) + 1)  # the batch's child range (its exact queries' range)
        result[active] = selective_m1.outward_sum(
            np.stack([c.bound(children[active], span, eligible[active]) for c in state.components])
        )
    else:
        for c in state.components:
            c.cache_range = c.products = c.classes = c.classes_range = c.scored = None
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
    parameters: PedigreeParameters,
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

    The enclosing worker already owns an admission. Each phase runs its contigs as the engine's nested concurrent
    tasks, each on its live fair CPU share, so ready contigs can use the enclosing admission. No worker pool or second
    TaskSession is opened; returned batch arrays are discarded by the caller.
    """

    def __init__(self) -> None:
        # The enclosing task's live allowance, its declared range included. (The thread count a task is called with
        # is only the team it started with: as a cap it held the whole screen at one CPU when the task began beside
        # busy peers.)
        self.budget = current_budget()

    def __enter__(self) -> _ThreadPhases:
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> Literal[False]:
        return False

    @staticmethod
    def execute(task: Task, share: Any) -> ResidentContig | np.ndarray | None:
        """One contig's phase on its share (the engine has set the calling thread's kernel team to it)."""
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
        functions = [partial(self.execute, task) for task in tasks]
        values = dict(self.budget.concurrent(functions, label="pedigree screen phase"))
        return {task.key: values[index] for index, task in enumerate(tasks)}


def run(
    prepared: tuple[PreparedChromosome, ...],
    eligibility: ParentEligibility,
    parameters: PedigreeParameters,
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
    # the retention band's units of each chromosome (m1.retention_units)
    units = m1.retention_units(informative)
    builder = CompactBuilder(n, markers, eligibility.parents.sum(1), units)
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
    phases = _ThreadPhases() if nested else TaskSession(threads, in_process=in_process, preload=(__name__,))
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

            batch_result = selective_m1.search_batch(upper, children, allowed, units, parameters, exact)
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
    parameters: PedigreeParameters,
    *,
    threads: int = 1,
    reverse: bool | None = None,
    eligibility: ParentEligibility | None = None,
    child_range: tuple[int, int] | None = None,
    placement: ScorePlacement | None = None,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> PairBounds:
    """Evaluate canonical pair rows and round upwards once.

    A child range is a compute-only task. A scatter gather joins all ranges
    in child order before publishing the ordinary complete contig record.
    """
    if child_range is not None and file is not None:
        raise InputError("publish pair bounds only after gathering the whole contig")
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    adapter = PairBoundsAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, BOUNDS, identity, adapter)
        if previous is not None:
            return previous
    with thread_team(default=threads):
        keys, pieces = pair_bound_screen.chromosome_ranking(
            prepared, compact, parameters, claim_threads, reverse=reverse, eligibility=eligibility,
            child_range=child_range, placement=placement,
        )
        product = RankedBounds(prepared.contig, keys, pieces=pieces)
    if file is not None:
        assert identity is not None
        publish_product(file, BOUNDS, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def merge_child_bounds(
    contig: str,
    compact: CompactM1,
    blocks: Sequence[tuple[tuple[int, int], PairBounds]],
) -> PairBounds:
    """Join complete, disjoint child ranges for one contig, in canonical order."""
    offsets = pair_bound_screen.retained_pair_offsets(compact)
    bounds = np.empty(offsets[-1], dtype=np.float32)
    cursor = 0
    parts = []
    for (start, stop), part in sorted(blocks, key=lambda block: block[0]):
        if start != cursor or not start < stop <= compact.n_samples or part.contig != contig:
            raise InputError(f"{contig}: invalid M2 ranking child block {(start, stop)}")
        destination = bounds[offsets[start]:offsets[stop]]
        if part.bounds.shape != destination.shape:
            raise InputError(f"{contig}: invalid number of pair keys for children {(start, stop)}")
        destination[:] = part.bounds
        if isinstance(part, RankedBounds) and part.pieces is not None:
            parts.append(part.pieces)
        cursor = stop
    if cursor != compact.n_samples:
        raise InputError(f"{contig}: missing M2 ranking child blocks from {cursor} onward")
    if parts:
        cuts = parts[0].cuts
        if not all(np.array_equal(p.cuts, cuts) for p in parts):
            raise InputError(f"{contig}: inconsistent material cuts in M2 child blocks")
        return RankedBounds(contig, bounds, pieces=pair_bound_screen.RankedPieces(
            np.concatenate([p.trios for p in parts]), cuts, np.concatenate([p.values for p in parts])))
    return PairBounds(contig, bounds)


def load_ranked_pieces(file: CheckpointFile, identity: str) -> pair_bound_screen.RankedPieces | None:
    """Read optional pieces from the rank record with its expected derivation."""
    from ...store.protocol import read_items

    record = read_items(file, BOUNDS, identity, ("schema", "ranked_trios", "ranked_cuts", "ranked_values"))
    if record is None:
        return None
    require_schema(record.payload, PairBoundsAdapter.schema)
    return PairBoundsAdapter.decode_pieces(record.payload)


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
    parameters: PedigreeParameters,
    *,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> CandidatePanel:
    """Build and publish the M2 panel: every retained eligible pair, capped per child (ties and indeterminate bounds
    kept)."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    adapter = PanelAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, PANEL, identity, adapter)
        if previous is not None:
            return previous
    c = screen.compact
    utility = panel.robust_screen_values(c.values, c.n_contigs, c.n_samples, c.markers, eligibility, parameters,
                                         c.units)
    product, _ = panel.bounded_panel(c, bounds, utility, eligibility, parameters)
    if file is not None:
        assert identity is not None
        publish_product(file, PANEL, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def screen_derivation(
    parameters: PedigreeParameters,
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
    parameters: PedigreeParameters,
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
            max_pairs_per_child=parameters.panel.max_pairs_per_child,
            representation="retained-pairs-full-likelihood-float32-up-v2",
        ),
        numerical_policy or NumericalPolicy.current(),
        dict(prepared=prepared_identity, screen=screen_identity),
        reverse,
    ).identity


def panel_derivation(
    screen_identity: str,
    bound_identities: Sequence[tuple[str, str]],
    cap: int,
    *,
    reverse: bool = False,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind the screen, every bound identity and the per-child pair cap; resource choices are absent."""
    inputs = {"screen": screen_identity}
    for contig, identity in bound_identities:
        if "bounds:" + contig in inputs:
            raise InputError("duplicate contig in panel identity")
        inputs["bounds:" + contig] = identity
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(max_pairs_per_child=int(cap), contigs=tuple(c for c, _ in bound_identities)),
        numerical_policy or NumericalPolicy.current(),
        inputs,
        reverse,
    ).identity
