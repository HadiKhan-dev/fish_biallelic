"""Prepare and publish one contig's pedigree inputs (PS01–PS05, PS22).

The task borrows painting/evidence, owns its result, and publishes only a complete
schema-1 product. Identities bind source derivations and numerical policy. Worker
counts and emission workspace sizes do not enter that identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from functools import partial
from typing import Any

from ...contracts.evidence import ContigEvidence
from ...contracts.genetic_map import ChromosomeGeneticMap, GeneticMapSet
from ...contracts.painting import Painting
from ...contracts.payload import from_payload, to_payload
from ...contracts.pedigree import ParentEligibility, PreparedChromosome
from ...contracts.roles import require_schema
from ...core.concurrency import threaded_each
from ...core.digest import json_digest
from ...core.errors import InputError
from ...engine.graph import Expansion, ResultRef
from ...engine.runtime import task_team
from ...engine.scheduler import TASK_THREADS
from ...science.pedigree import prepare as science
from ...science.pedigree.parameters import PedigreeConfig, PreparationConfig
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, read_items, resume
from ...store.store import CheckpointStore
from ..framework import RecordKey, ResourceRequest, Task, TaskKey, load_product, publish_product

KEY = "pedigree/prepared"


class PreparedAdapter:
    """Lossless schema-1 prepared record; derived views are not persisted."""

    schema = 1

    def encode(self, value: PreparedChromosome) -> Mapping[str, Any]:
        return {"schema": self.schema, **to_payload(value)}

    def decode(self, payload: Mapping[str, Any]) -> PreparedChromosome:
        require_schema(payload, self.schema)
        names = {f.name for f in fields(PreparedChromosome)}
        return from_payload(PreparedChromosome, {k: v for k, v in payload.items() if k != "schema" or k in names})


def derivation(
    parameters: PreparationConfig,
    upstream: Mapping[str, str],
    *,
    rate_is_measured: bool = False,
    reverse: bool = False,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind painting, evidence and genetic-map identities in their configured order."""
    if set(upstream) != {"painting", "evidence", "genetic_map"}:
        raise InputError("prepared identity needs painting, evidence and genetic_map derivations")
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(preparation=parameters, rate_is_measured=rate_is_measured),
        numerical_policy or NumericalPolicy.current(),
        dict(upstream),
        reverse,
    ).identity


def run(
    painting: Painting,
    evidence: ContigEvidence,
    chromosome_map: ChromosomeGeneticMap,
    parameters: PreparationConfig,
    *,
    rate_is_measured: bool = False,
    threads: int = 1,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> PreparedChromosome:
    """Return the whole prepared contig, reusing only a matching published identity."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    adapter = PreparedAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, KEY, identity, adapter)
        if previous is not None:
            return previous
    with task_team(threads):
        product = science.prepare_chromosome(
            painting, evidence, chromosome_map, parameters, rate_is_measured=rate_is_measured
        )
    if file is not None:
        assert identity is not None
        publish_product(file, KEY, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def task(
    painting: Painting,
    evidence: ContigEvidence,
    chromosome_map: ChromosomeGeneticMap,
    parameters: PreparationConfig,
    *,
    file: CheckpointFile | None = None,
    identity: str | None = None,
    rate_is_measured: bool = False,
    cores: int = 1,
    depends_on: tuple[TaskKey, ...] = (),
) -> Task:
    """Declare one contig task with painting/evidence affinity and explicit inputs."""
    contig = painting.contig
    return Task(
        TaskKey("pedigree", "prepare", contig),
        run,
        args=(painting, evidence, chromosome_map, parameters),
        kwargs=dict(file=file, identity=identity, rate_is_measured=rate_is_measured),
        depends_on=depends_on,
        inputs=(RecordKey(contig, "painting"), RecordKey(contig, "evidence")),
        outputs=(RecordKey(contig, KEY),),
        resources=ResourceRequest(max_cores=cores),
        affinity=contig,
        thread_argument="threads",
    )


@dataclass(frozen=True)
class PreparedReference:
    """Small prepared completion; graph results never retain its component arrays."""

    contig: str
    identity: str
    omitted_reason: str | None


def _read_request(file: CheckpointFile, request: Any) -> Any:
    """Decode an upstream through its owner's independently derived reader contract."""
    if request is None:
        raise InputError(f"missing pedigree prerequisite in {file.label}")
    items = getattr(request, "items", None)
    value = (
        resume(file, request.key, request.identity)
        if items is None
        else read_items(file, request.key, request.identity, tuple(dict.fromkeys(("schema", *items))))
    )
    if value is None:
        raise InputError(f"missing pedigree prerequisite {file.label}:{request.key}")
    require_schema(value.payload, request.schema)
    return request.decode(value.payload)


def prepare_stored(
    root: str,
    manifest: Mapping[str, Any],
    chromosome_map: ChromosomeGeneticMap,
    parameters: PedigreeConfig,
    numerical_policy: NumericalPolicy,
    *,
    threads: int = 1,
) -> PreparedReference:
    """Load checked contig inputs, publish preparation and return only its small cause."""
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records

    store = CheckpointStore(root, nthreads=threads)
    contig = chromosome_map.contig
    file = store.contig(contig)
    painted = painting_records.result_reader(store, manifest, product="painting", contig=contig, panels=False)
    observed = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
    if painted is None or observed is None:
        raise InputError(f"pedigree requires painting and evidence for {contig}")
    identity = derivation(
        parameters.preparation,
        dict(painting=painted.identity, evidence=observed.identity, genetic_map=manifest["map_identities"][contig]),
        rate_is_measured=parameters.exposure.rate_is_measured,
        reverse=parameters.reverse_conventions,
        numerical_policy=numerical_policy,
    )
    cached = read_items(file, KEY, identity, ("schema", "contig", "omitted_reason"))
    if cached is not None:
        require_schema(cached.payload, 1)
        if cached.payload["contig"] != contig:
            raise InputError("prepared contig differs from its configured axis")
        return PreparedReference(contig, identity, cached.payload["omitted_reason"])
    painting = _read_request(file, painted)
    evidence = _read_request(file, observed)
    if painting.sample_ids != tuple(manifest["sample_ids"]) or evidence.sample_ids != painting.sample_ids:
        raise InputError("pedigree preparation sample axes differ")
    product = run(
        painting,
        evidence,
        chromosome_map,
        parameters.preparation,
        rate_is_measured=parameters.exposure.rate_is_measured,
        threads=threads,
        file=file,
        identity=identity,
    )
    return PreparedReference(contig, identity, product.omitted_reason)


def _required_prepared(store: CheckpointStore, reference: PreparedReference) -> PreparedChromosome:
    value = load_product(store.contig(reference.contig), KEY, reference.identity, PreparedAdapter())
    if value is None:
        raise InputError(f"missing prepared pedigree contig {reference.contig}")
    return value


def _bound_stored(
    root: str,
    reference: PreparedReference,
    screen_identity: str,
    parameters: PedigreeConfig,
    numerical_policy: NumericalPolicy,
    *,
    threads: int = 1,
) -> tuple[str, str]:
    from . import screen

    store = CheckpointStore(root, nthreads=threads)
    identity = screen.bound_derivation(
        parameters, reference.identity, screen_identity, numerical_policy=numerical_policy
    )
    file = store.contig(reference.contig)
    cached = read_items(file, screen.BOUNDS, identity, ("schema", "contig"))
    if cached is not None:
        require_schema(cached.payload, 1)
        return reference.contig, identity
    record = load_product(store.run, screen.SCREEN, screen_identity, screen.ScreenAdapter())
    if record is None:
        raise InputError("pair bounds require the complete selective screen")
    screen.bound_run(
        _required_prepared(store, reference), record.compact, parameters, threads=threads, file=file, identity=identity
    )
    return reference.contig, identity


def _scores_stored(
    root: str,
    reference: PreparedReference,
    chromosome: int,
    screen_identity: str,
    panel_identity: str,
    sample_ids: tuple[str, ...],
    parameters: PedigreeConfig,
    numerical_policy: NumericalPolicy,
    *,
    threads: int = 1,
) -> tuple[str, str]:
    from . import scores, screen

    store = CheckpointStore(root, nthreads=threads)
    panel = load_product(store.run, screen.PANEL, panel_identity, screen.PanelAdapter())
    if panel is None:
        raise InputError("floored scores require the complete candidate panel")
    identity = scores.derivation(
        parameters, reference.identity, screen_identity, panel, numerical_policy=numerical_policy
    )
    file = store.contig(reference.contig)
    cached = read_items(file, scores.KEY, identity, ("schema", "contig"))
    if cached is not None:
        require_schema(cached.payload, 1)
        return reference.contig, identity
    record = load_product(store.run, screen.SCREEN, screen_identity, screen.ScreenAdapter())
    if record is None:
        raise InputError("floored scores require the complete selective screen")
    scores.run(
        _required_prepared(store, reference),
        record.compact,
        chromosome,
        panel,
        ParentEligibility.unrestricted(sample_ids),
        parameters,
        threads=threads,
        file=file,
        identity=identity,
    )
    return reference.contig, identity


def _after_bounds(
    bounds: tuple[tuple[str, str], ...],
    root: str,
    manifest: Mapping[str, Any],
    references: tuple[PreparedReference, ...],
    screen_identity: str,
    parameters: PedigreeConfig,
    numerical_policy: NumericalPolicy,
    maximum_cores: int,
    *,
    threads: int = 1,
) -> Expansion:
    from ...science.pedigree.pair_bound import retained_pair_count
    from . import decision, screen

    store = CheckpointStore(root, nthreads=threads)
    record = load_product(store.run, screen.SCREEN, screen_identity, screen.ScreenAdapter())
    if record is None:
        raise InputError("panel requires the complete selective screen")
    identity = screen.panel_derivation(
        screen_identity,
        bounds,
        parameters.panel.pair_relevance_margin,
        reverse=parameters.reverse_conventions,
        numerical_policy=numerical_policy,
    )
    values = screen.StoredBounds(
        tuple(store.contig(c) for c, _ in bounds), tuple(i for _, i in bounds), retained_pair_count(record.compact)
    )
    screen.panel_run(
        record,
        values,
        ParentEligibility.unrestricted(tuple(manifest["sample_ids"])),
        parameters,
        file=store.run,
        identity=identity,
    )
    tasks = tuple(
        Task(
            TaskKey("pedigree", "scores", reference.contig),
            _scores_stored,
            args=(
                root,
                reference,
                k,
                screen_identity,
                identity,
                tuple(manifest["sample_ids"]),
                parameters,
                numerical_policy,
            ),
            resources=ResourceRequest(max_cores=maximum_cores),
            affinity=reference.contig,
            outputs=(RecordKey(reference.contig, "pedigree/floored_scores"),),
            thread_argument="threads",
        )
        for k, reference in enumerate(references)
    )
    return Expansion(
        tasks, decision.task_stored(root, manifest, cores=maximum_cores, depends_on=tuple(t.key for t in tasks))
    )


def _after_prepared(
    references: tuple[PreparedReference, ...],
    root: str,
    manifest: Mapping[str, Any],
    parameters: PedigreeConfig,
    numerical_policy: NumericalPolicy,
    maximum_cores: int,
    *,
    threads: int = 1,
) -> Expansion:
    from ...science.pedigree.run import precheck
    from . import screen

    informative = tuple(r for r in references if r.omitted_reason is None)
    sample_ids = tuple(manifest["sample_ids"])
    precheck(len(sample_ids), len(informative), parameters, basis="informative")
    eligibility = ParentEligibility.unrestricted(sample_ids)
    identity = screen.screen_derivation(
        parameters,
        eligibility.identity(),
        tuple((r.contig, r.identity) for r in informative),
        numerical_policy=numerical_policy,
    )
    store = CheckpointStore(root, nthreads=threads)
    cached = read_items(store.run, screen.SCREEN, identity, ("schema",))
    if cached is None:
        # Release 4.10: the contigs' prepared records read on this task's share, one per thread (one after another
        # before). Release 4.11: each thread decodes on its own budget (TASK_THREADS: one CPU inside the map), with
        # Blosc's GIL release on, so the decodes overlap (a store with the task's width ran them one at a time).
        readers = CheckpointStore(root, nthreads=TASK_THREADS)
        prepared = tuple(threaded_each(partial(_required_prepared, readers), informative))
        screen.run(prepared, eligibility, parameters, threads=threads, nested=True, file=store.run, identity=identity)
        del prepared
    else:
        require_schema(cached.payload, 1)
    tasks = tuple(
        Task(
            TaskKey("pedigree", "pair_bounds", r.contig),
            _bound_stored,
            args=(root, r, identity, parameters, numerical_policy),
            resources=ResourceRequest(max_cores=maximum_cores),
            affinity=r.contig,
            outputs=(RecordKey(r.contig, screen.BOUNDS),),
            thread_argument="threads",
        )
        for r in informative
    )
    gather = Task(
        TaskKey("pedigree", "panel.barrier"),
        _after_bounds,
        args=(
            tuple(ResultRef(t.key) for t in tasks),
            root,
            manifest,
            informative,
            identity,
            parameters,
            numerical_policy,
            maximum_cores,
        ),
        depends_on=tuple(t.key for t in tasks),
        resources=ResourceRequest(max_cores=maximum_cores),
        outputs=(RecordKey("run", screen.PANEL),),
        thread_argument="threads",
    )
    return Expansion(tasks, gather)


def pipeline_start(root: str, maps: GeneticMapSet, manifest: Mapping[str, Any] | None, maximum_cores: int) -> Expansion:
    """Expand checked painting inputs through scores and the terminal PE4/PE5 action.

    Each contig task returns only a completion descriptor. Bounds are streamed per
    child at the panel barrier. Stored inputs are decoded by their owning adapters;
    no cached product supplies its own expected identity. The outer engine retains
    authority over every task and the screen's nested thread phases.
    """
    from ...science.pedigree.run import precheck

    store = CheckpointStore(root)
    store.check_layout()
    manifest = store.manifest("run") if manifest is None else manifest
    if manifest is None:
        raise InputError("pedigree requires the bound run manifest")
    parameters = replace(
        from_payload(PedigreeConfig, manifest["parameters"]["pedigree"]),
        reverse_conventions=bool(manifest["ties"]["reverse"]),
    )
    numerical_policy = NumericalPolicy(**manifest["numerical_policy"])
    contigs = tuple(manifest["contigs"])
    precheck(len(manifest["sample_ids"]), len(contigs), parameters, basis="configured")
    if any(json_digest(maps.identity(c)) != manifest["map_identities"][c] for c in contigs):
        raise InputError("pedigree maps differ from the bound run manifest")
    tasks = tuple(
        Task(
            TaskKey("pedigree", "prepare", c),
            prepare_stored,
            args=(root, manifest, maps.for_contig(c), parameters, numerical_policy),
            affinity=c,
            resources=ResourceRequest(max_cores=maximum_cores),
            inputs=(RecordKey(c, "painting"), RecordKey(c, "evidence")),
            outputs=(RecordKey(c, KEY),),
            thread_argument="threads",
        )
        for c in contigs
    )
    gather = Task(
        TaskKey("pedigree", "screen.barrier"),
        _after_prepared,
        args=(tuple(ResultRef(t.key) for t in tasks), root, manifest, parameters, numerical_policy, maximum_cores),
        depends_on=tuple(t.key for t in tasks),
        resources=ResourceRequest(max_cores=maximum_cores),
        outputs=(RecordKey("run", "pedigree/selective_screen"),),
        thread_argument="threads",
    )
    return Expansion(tasks, gather)


def pipeline_task(
    root: str,
    maps: GeneticMapSet,
    manifest: Mapping[str, Any] | None = None,
    *,
    cores: int = 1,
    depends_on: tuple[TaskKey, ...] = (),
) -> Task:
    """Construct the complete G1 pedigree/run action without reading its future inputs."""
    return Task(
        TaskKey("pedigree", "run"),
        pipeline_start,
        args=(root, maps, manifest, cores),
        depends_on=depends_on,
        resources=ResourceRequest(max_cores=cores),
    )
