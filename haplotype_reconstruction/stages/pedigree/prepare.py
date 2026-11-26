"""Prepare and publish one contig's pedigree inputs (PS01–PS05, PS22).

The task borrows painting/evidence, owns its result, and publishes only a complete
schema-2 product. Identities bind source derivations and numerical policy. Worker
counts and emission workspace sizes do not enter that identity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from functools import partial
from typing import Any

from ...contracts.contigs import AnalysisContigs
from ...contracts.evidence import ContigEvidence
from ...contracts.genetic_map import ChromosomeGeneticMap, GeneticMapSet
from ...contracts.painting import Painting
from ...contracts.payload import from_payload, to_payload
from ...contracts.pedigree import ChromosomeSummaryView, ParentEligibility, PreparedChromosome
from ...contracts.roles import require_schema
from ...core.concurrency import threaded_each
from ...core.digest import json_digest
from ...core.errors import InputError
from ...engine.cpu import TASK_THREADS, thread_team
from ...engine.graph import Expansion, ResultRef
from ...science.pedigree import prepare as science
from ...science.pedigree.blocks import ScorePlacement, cut_list
from ...science.pedigree.parameters import PedigreeParameters
from ...science.pedigree.regimes import Layout
from ...science.pedigree.scoring_parameters import PreparationParameters
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, read_items, resume
from ...store.store import CheckpointStore
from ..framework import RecordKey, ResourceRequest, Task, TaskKey, load_product, publish_product
from ..relatedness.working_map import map_identity, working_maps

KEY = "pedigree/prepared"


class PreparedAdapter:
    """Schema-8 source laws and structured child gaps, with the contig's compact layout header."""

    schema = 8

    def encode(self, value: PreparedChromosome) -> Mapping[str, Any]:
        payload = to_payload(value)
        payload["schema"] = self.schema
        return {**payload, "layout": [
            dict(selector=c.selector_switch_probability, fallback=c.selector_fallback_bp, bins=c.n_bins,
                 markers=c.informative_markers)
            for c in value.components]}

    def decode(self, payload: Mapping[str, Any]) -> PreparedChromosome:
        require_schema(payload, self.schema)
        names = {f.name for f in fields(PreparedChromosome)}
        data = {k: v for k, v in payload.items() if k in names}
        # The contract's dataclass still describes the schema-7 payload. This adapter owns
        # the on-disk version and selects the structured gap representation before decoding.
        data["schema"] = PreparedChromosome.schema
        return replace(from_payload(PreparedChromosome, data), schema=self.schema)


# Prepared header items the material axis and the chromosome summary read (no component arrays).
LAYOUT_ITEMS = ("schema", "contig", "component_count", "omitted_reason", "layout", "configured_morgans")


def layout_summary(header: Mapping[str, Any]) -> tuple[Layout, ChromosomeSummaryView]:
    """Prepared header to the material axis and the original chromosome report summary; no block arrays."""
    parts = header["layout"]
    return (Layout(header["contig"], tuple(p["selector"] for p in parts), tuple(p.get("fallback") for p in parts)),
            ChromosomeSummaryView(header["contig"], header["component_count"], len(parts),
                                  sum(p["markers"] for p in parts), header["omitted_reason"],
                                  float(header["configured_morgans"])))


def derivation(
    parameters: PreparationParameters,
    upstream: Mapping[str, str],
    *,
    reverse: bool = False,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind painting, evidence and genetic-map identities in their configured order."""
    if set(upstream) != {"painting", "evidence", "genetic_map"}:
        raise InputError("prepared identity needs painting, evidence and genetic_map derivations")
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(preparation=parameters),
        numerical_policy or NumericalPolicy.current(),
        dict(upstream),
        reverse,
    ).identity


def run(
    painting: Painting,
    evidence: ContigEvidence,
    chromosome_map: ChromosomeGeneticMap,
    parameters: PreparationParameters,
    *,
    threads: int = 1,
    file: CheckpointFile | None = None,
    identity: str | None = None,
    read_builder: Callable | None = None,
) -> PreparedChromosome:
    """Return the whole prepared contig, reusing only a matching published identity. ``read_builder``: the painting's
    read-state builder (read_state_builder; None when the painting has no read state)."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    adapter = PreparedAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, KEY, identity, adapter)
        if previous is not None:
            return previous
    with thread_team(default=threads):
        product = science.prepare_chromosome(painting, evidence, chromosome_map, parameters, read_builder=read_builder)
    if file is not None:
        assert identity is not None
        publish_product(file, KEY, identity, product, adapter)
    return adapter.decode(adapter.encode(product))


def read_state_builder(painting: Painting, evidence: ContigEvidence, file: CheckpointFile, evidence_identity: str,
                       corrected_rows: tuple[int, ...] = ()):
    """The source chain's emission builder with the painting's read state (stages.painting.read_state over
    science.pedigree.prepare.scalar_emissions, at the path strength only), or None when the painting has none.
    ``corrected_rows``: QC A's corrected samples of ``evidence``, which carry no uniform foreign class."""
    if painting.parameters.read_classes == 1:
        return None
    from ..painting.read_state import stage_builder

    cohort = evidence.cohort_frequencies
    return stage_builder(partial(science.scalar_emissions, cohort=cohort), file, evidence_identity, cohort, None,
                         painting.parameters, unit_batches=True, corrected_rows=corrected_rows)


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
    parameters: PedigreeParameters,
    numerical_policy: NumericalPolicy,
    *,
    threads: int = 1,
) -> PreparedReference:
    """Load checked contig inputs, publish preparation and return only its small cause."""
    from ...science.painting.parameters import PaintingParameters
    from ..evidence import records as evidence_records
    from ..painting import repaint as painting_records  # the painting the pedigree reads (4.21 P)

    store = CheckpointStore(root, nthreads=threads)
    contig = chromosome_map.contig
    file = store.contig(contig)
    painted = painting_records.result_reader(
        store, manifest, product="refit_reference", contig=contig, panels=False, emissions=False,
    )
    observed = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
    if painted is None or observed is None:
        raise InputError(f"pedigree requires painting and evidence for {contig}")
    identity = derivation(
        parameters.preparation,
        dict(painting=painted.identity, evidence=observed.identity, genetic_map=map_identity(store, manifest, contig)),
        reverse=parameters.reverse_conventions,
        numerical_policy=numerical_policy,
    )
    cached = read_items(file, KEY, identity, ("schema", "contig", "omitted_reason"))
    if cached is not None:
        require_schema(cached.payload, PreparedAdapter.schema)
        if cached.payload["contig"] != contig:
            raise InputError("prepared contig differs from its configured axis")
        return PreparedReference(contig, identity, cached.payload["omitted_reason"])
    # The read-state source builds its own raw emissions at path strength.
    # Decode the packed repaint cache only when the scalar source can use it.
    header = read_items(file, painted.key, painted.identity, ("schema", "parameters"))
    if header is None:
        raise InputError(f"missing pedigree prerequisite {file.label}:{painted.key}")
    require_schema(header.payload, painted.schema)
    painting_parameters = from_payload(PaintingParameters, header.payload["parameters"])
    if painting_parameters.read_classes == 1:
        painted = painting_records.result_reader(
            store, manifest, product="refit_reference", contig=contig, panels=False
        )
    painting = _read_request(file, painted)
    evidence = _read_request(file, observed)
    if painting.sample_ids != tuple(manifest["sample_ids"]) or evidence.sample_ids != painting.sample_ids:
        raise InputError("pedigree preparation sample axes differ")
    # 4.22 QC A: flagged libraries read on contamination-corrected likelihoods. The prepared identity already binds the
    # run's contamination decision through the painting it reads (the repaint's identity includes it).
    from ..painting import contamination as contamination_stage

    evidence, corrected_rows = contamination_stage.corrected_evidence_with_rows(
        store, manifest, contig, evidence, evidence_identity=observed.identity
    )
    product = run(
        painting,
        evidence,
        chromosome_map,
        parameters.preparation,
        threads=threads,
        file=file,
        identity=identity,
        read_builder=read_state_builder(painting, evidence, file, observed.identity, corrected_rows),
    )
    return PreparedReference(contig, identity, product.omitted_reason)


def prepared_reference(store, contig, parameters, policy, causes) -> PreparedReference:
    """Check the publication needed by a scatter gather."""
    painting, evidence, working_map = causes
    identity = derivation(parameters.preparation, dict(painting=painting, evidence=evidence, genetic_map=working_map),
                          reverse=parameters.reverse_conventions, numerical_policy=policy)
    saved = read_items(store.contig(contig), KEY, identity, ("schema", "contig", "omitted_reason"))
    if saved is None:
        raise InputError(f"{contig}: publish pedigree --stage prepare before the gather")
    require_schema(saved.payload, PreparedAdapter.schema)
    if saved.payload["contig"] != contig:
        raise InputError("prepared contig differs from its configured axis")
    return PreparedReference(contig, identity, saved.payload["omitted_reason"])


def _required_prepared(store: CheckpointStore, reference: PreparedReference) -> PreparedChromosome:
    value = load_product(store.contig(reference.contig), KEY, reference.identity, PreparedAdapter())
    if value is None:
        raise InputError(f"missing prepared pedigree contig {reference.contig}")
    return value


def _bound_stored(
    root: str,
    reference: PreparedReference,
    screen_identity: str,
    eligibility: ParentEligibility,
    parameters: PedigreeParameters,
    numerical_policy: NumericalPolicy,
    placement: ScorePlacement,
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
        _required_prepared(store, reference), record.compact, parameters, threads=threads, eligibility=eligibility,
        placement=placement, file=file, identity=identity,
    )
    return reference.contig, identity


def placements(layouts: tuple[Layout, ...], axis: Any, parameters: PedigreeParameters) -> tuple[ScorePlacement, ...]:
    """Every informative contig's placement from the prepared layouts (the barrier and the bound-input reader)."""
    last = axis.units - 1
    return tuple(
        ScorePlacement(offset, last, cut_list(layout.selectors, offset, last, parameters.material))
        for layout, offset in zip(layouts, axis.offsets, strict=True)
    )


def _scores_stored(
    root: str,
    reference: PreparedReference,
    chromosome: int,
    screen_identity: str,
    panel_identity: str,
    bound_identity: str,
    placement: ScorePlacement,
    sample_ids: tuple[str, ...],
    parameters: PedigreeParameters,
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
        parameters, reference.identity, screen_identity, panel, placement, numerical_policy=numerical_policy
    )
    file = store.contig(reference.contig)
    cached = read_items(file, scores.KEY, identity, ("schema", "contig"))
    if cached is not None:
        require_schema(cached.payload, scores.ScoresAdapter.schema)
        return reference.contig, identity
    record = load_product(store.run, screen.SCREEN, screen_identity, screen.ScreenAdapter())
    if record is None:
        raise InputError("floored scores require the complete selective screen")
    # The score tiles run independently, while each chain carries its previous bin's state.
    # TroMau chr19 used .828 / .792 / .803 at 8 / 19 / 38 threads (19.54 / 8.68 / 4.50 wall s).
    scores.run(
        _required_prepared(store, reference),
        record.compact,
        chromosome,
        panel,
        ParentEligibility.unrestricted(sample_ids),
        parameters,
        placement,
        threads=threads,
        ranked=screen.load_ranked_pieces(file, bound_identity),
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
    placed: tuple[ScorePlacement, ...],
    parameters: PedigreeParameters,
    numerical_policy: NumericalPolicy,
    maximum_cores: int,
    costs: tuple[int, ...],
    *,
    threads: int = 1,
) -> Expansion:
    from ...science.pedigree.pair_bound_screen import retained_pair_count
    from . import decision, screen

    store = CheckpointStore(root, nthreads=threads)
    record = load_product(store.run, screen.SCREEN, screen_identity, screen.ScreenAdapter())
    if record is None:
        raise InputError("panel requires the complete selective screen")
    identity = screen.panel_derivation(
        screen_identity,
        bounds,
        parameters.panel.max_pairs_per_child,
        reverse=parameters.reverse_conventions,
        numerical_policy=numerical_policy,
    )
    values = screen.StoredBounds(
        tuple(store.contig(c) for c, _ in bounds), tuple(i for _, i in bounds), retained_pair_count(record.compact)
    )
    # The genome panel depends on the completed screen and bounds on all contigs.
    # This serial publication and its ordered reduction took 0.15 s on famhs and 0.18 s on TroMau.
    screen.panel_run(
        record,
        values,
        ParentEligibility.unrestricted(tuple(manifest["sample_ids"])),
        parameters,
        file=store.run,
        identity=identity,
    )
    from ...science.painting.parameters import PaintingParameters
    from ..painting.records import reference_key

    reference_key(from_payload(PaintingParameters, manifest["parameters"]["painting"]))  # validates the payload
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
                bound_identity,
                placement,
                tuple(manifest["sample_ids"]),
                parameters,
                numerical_policy,
            ),
            resources=ResourceRequest(max_cores=maximum_cores, width=maximum_cores, cost=cost),
            outputs=(RecordKey(reference.contig, "pedigree/floored_scores"),),
            thread_argument="threads",
        )
        for k, (reference, (_, bound_identity), placement, cost) in enumerate(
            zip(references, bounds, placed, costs, strict=True)
        )
    )
    return Expansion(
        tasks, decision.task_stored(root, manifest, cores=maximum_cores, depends_on=tuple(t.key for t in tasks))
    )


def _after_prepared(
    references: tuple[PreparedReference, ...],
    root: str,
    manifest: Mapping[str, Any],
    parameters: PedigreeParameters,
    numerical_policy: NumericalPolicy,
    maximum_cores: int,
    *,
    threads: int = 1,
    scale_identity: str | None = None,
) -> Expansion | dict[RecordKey, str]:
    """The screen barrier: the material choice (regimes.material_choice; with rate_estimate = ignore the map-scale
    record ``scale_identity`` is not read), then the screen and the pair-bound tasks."""
    from ...science.pedigree import regimes
    from ...science.pedigree.run import precheck
    from . import decision, map_scale, screen

    informative = tuple(r for r in references if r.omitted_reason is None)
    sample_ids = tuple(manifest["sample_ids"])
    precheck(len(sample_ids), basis="informative")
    eligibility = ParentEligibility.unrestricted(sample_ids)
    store = CheckpointStore(root, nthreads=threads)
    summaries, layouts = [], []
    for r in references:
        header = read_items(store.contig(r.contig), KEY, r.identity, LAYOUT_ITEMS)
        if header is None:
            raise InputError("missing complete prepared chromosome")
        require_schema(header.payload, PreparedAdapter.schema)
        layout, summary = layout_summary(header.payload)
        summaries.append(summary)
        if r.omitted_reason is None:
            layouts.append(layout)
    estimate = None
    if parameters.map_scale.mode != "ignore":
        if scale_identity is None:
            raise InputError("the pedigree material rule requires the map-scale estimate's identity")
        estimate = map_scale.load(store, scale_identity)
    else:
        scale_identity = ""
    choice, classification, _morgans, axis = regimes.material_choice(tuple(layouts), parameters, estimate)
    if classification.name == "not_assessable":
        from .records import PedigreeDependencies
        dependencies = PedigreeDependencies(eligibility.identity(), "", "",
            tuple((r.contig, r.identity) for r in references), (), regime=classification.name,
            map_scale=scale_identity, veto=choice.veto_regime, reference=regimes.reference_regime(choice))
        # Bind the same evidence causes as bound_inputs, without requiring screen/panel/fits.
        from ..evidence import records as evidence_records
        dependencies = replace(dependencies, evidence=tuple((r.contig,
            evidence_records.result_reader(store, manifest, product="evidence", contig=r.contig).identity)
            for r in references))
        reply = decision.finish_not_assessable(sample_ids, tuple(r.contig for r in references),
            tuple(summaries), classification, parameters, dependencies,
            file=store.run, numerical_policy=numerical_policy, receipt=True, material=choice, map_scale=estimate)
        assert isinstance(reply, dict)
        return reply
    placed = placements(tuple(layouts), axis, parameters)
    identity = screen.screen_derivation(
        parameters,
        eligibility.identity(),
        tuple((r.contig, r.identity) for r in informative),
        numerical_policy=numerical_policy,
    )
    store = CheckpointStore(root, nthreads=threads)
    cached = read_items(store.run, screen.SCREEN, identity, ("schema",))
    if cached is None:
        # Read prepared contigs concurrently on the task's share. Each reader uses TASK_THREADS
        # (one CPU inside the map), with Blosc's GIL release enabled so independent decodes overlap.
        readers = CheckpointStore(root, nthreads=TASK_THREADS)
        prepared = tuple(threaded_each(partial(_required_prepared, readers), informative))
        # Per-contig phases are concurrent. Each exact query round depends on the preceding bound and
        # request mask. The famhs screen used 2,217 CPU s in 37.8 wall s (.772) in the saved baseline.
        screen.run(prepared, eligibility, parameters, threads=threads, nested=True, file=store.run, identity=identity)
        del prepared
    else:
        require_schema(cached.payload, 1)
    from ...science.painting.parameters import PaintingParameters
    from ..painting.records import reference_key

    reference_key(from_payload(PaintingParameters, manifest["parameters"]["painting"]))  # validates the payload
    # These tasks load immutable checkpoint inputs and have no resident state.
    # Independent contigs share the allocator, including when there are more than
    # the ordinary number of lanes. Publish exactly one record per contig.
    costs = tuple(summary.informative_markers for summary in summaries if summary.omitted_reason is None)
    tasks = tuple(
        Task(
            TaskKey("pedigree", "pair_bounds", r.contig),
            _bound_stored,
            args=(root, r, identity, eligibility, parameters, numerical_policy, placement),
            resources=ResourceRequest(max_cores=maximum_cores, width=maximum_cores, cost=cost),
            outputs=(RecordKey(r.contig, screen.BOUNDS),),
            thread_argument="threads",
        )
        for r, cost, placement in zip(informative, costs, placed, strict=True)
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
            placed,
            parameters,
            numerical_policy,
            maximum_cores,
            costs,
        ),
        depends_on=tuple(t.key for t in tasks),
        resources=ResourceRequest(max_cores=maximum_cores),
        outputs=(RecordKey("run", screen.PANEL),),
        thread_argument="threads",
    )
    return Expansion(tasks, gather)


def pipeline_start(
    root: str, maps: GeneticMapSet, manifest: Mapping[str, Any] | None, maximum_cores: int, *,
    threads: int = 1, gather_only: bool = False,
) -> Expansion | dict[RecordKey, str]:
    """Expand checked painting inputs through scores and the terminal pedigree decision/pedigree action.

    Each contig task returns only a completion descriptor. Bounds are streamed per
    child at the panel barrier. Stored inputs are decoded by their owning adapters;
    no cached product supplies its own expected identity. The outer engine retains
    authority over every task and the screen's nested thread phases.
    """
    from ...science.pedigree.run import precheck

    store = CheckpointStore(root, nthreads=threads)
    store.check_layout()
    manifest = store.manifest("run") if manifest is None else manifest
    if manifest is None:
        raise InputError("pedigree requires the bound run manifest")
    from . import map_scale
    from .decision import product_receipt
    from .records import PRODUCT, published_identity, run_parameters

    parameters = run_parameters(manifest)
    numerical_policy = NumericalPolicy(**manifest["numerical_policy"])
    contigs = AnalysisContigs.from_manifest(manifest).included
    precheck(len(manifest["sample_ids"]), basis="configured")
    maps = working_maps(store, manifest, maps)  # the material, units and exclusion weights read this map (rl20)
    if any(json_digest(maps.identity(c)) != map_identity(store, manifest, c) for c in contigs):
        raise InputError("pedigree maps differ from the run's working maps")
    causes = None
    if gather_only or (store.run.has(PRODUCT) and store.run.has(map_scale.KEY)):
        causes = map_scale.contig_causes(store, manifest)
        identity = published_identity(store, manifest, causes=causes)
        if identity is not None:
            if parameters.map_scale.mode == "ignore":
                # Even the ignored estimate is part of the reported stage's results.
                scale = map_scale.expected(store, manifest, parameters, numerical_policy, contig_inputs=causes[0])
                map_scale.load(store, scale)
            return product_receipt(identity)
    if gather_only:
        contig_inputs, _ = causes
        references = tuple(threaded_each(
            lambda c: prepared_reference(store, c, parameters, numerical_policy, contig_inputs[c]), contigs
        ))
    else:
        references = ()

    from ...science.painting.parameters import PaintingParameters
    from ..painting.records import reference_key

    painting_key = reference_key(from_payload(PaintingParameters, manifest["parameters"]["painting"]))
    # Preparation overlaps the map graph. Its famhs chr19 warm read took 24.3 wall / 27.0 CPU s,
    # and preparation took 85.6 wall / 347.8 CPU s (19 threads). With the table work, its matched warm
    # reads were 11.8 s, the read builder 4.6 s and preparation 30.4 s. The per-sample read recurrence depends
    # on ordered sites. On TroMau, unpinning only preparation cut its span from 819 to 327 s and raised the
    # whole command from 1,658 to 1,692 s. The current separate frame workers overlap the contig tasks.
    tasks = tuple(
        Task(
            TaskKey("pedigree", "prepare", c),
            prepare_stored,
            args=(root, manifest, maps.for_contig(c), parameters, numerical_policy),
            affinity=c,
            resources=ResourceRequest(max_cores=maximum_cores),
            inputs=(RecordKey(c, painting_key), RecordKey(c, "evidence")),
            outputs=(RecordKey(c, KEY),),
            thread_argument="threads",
        )
        for c in contigs if not gather_only
    )
    # The genome-wide map-scale estimate (record pedigree/map_scale) runs beside the contigs' preparations; the
    # screen barrier reads it to choose the material regime (regimes.material_choice).
    scale = Task(
        TaskKey("pedigree", "map_scale"),
        map_scale.run_stored,
        args=(root, manifest, maps, parameters, numerical_policy),
        kwargs=dict(maximum_cores=maximum_cores, gather_only=gather_only),
        resources=ResourceRequest(max_cores=maximum_cores),
        inputs=tuple(RecordKey(c, painting_key) for c in contigs),
        outputs=(RecordKey("run", map_scale.KEY),),
        thread_argument="threads",
    )
    gather = Task(
        TaskKey("pedigree", "screen.barrier"),
        _after_prepared,
        args=(references or tuple(ResultRef(t.key) for t in tasks), root, manifest, parameters, numerical_policy, maximum_cores),
        kwargs=dict(scale_identity=ResultRef(scale.key)),
        depends_on=tuple(t.key for t in tasks) + (scale.key,),
        resources=ResourceRequest(max_cores=maximum_cores),
        outputs=(RecordKey("run", "pedigree/selective_screen"),),
        thread_argument="threads",
    )
    return Expansion(tasks + (scale,), gather)
