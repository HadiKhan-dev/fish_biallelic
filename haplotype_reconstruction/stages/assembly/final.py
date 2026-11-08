"""Resume final and context assemblies at completion, join and refinement seams.

Owns the sequential hierarchy and publication. Stop rules and L1 context order
come from science.assembly.run. Refinement is a caller-supplied AS8 stage action,
so storage and execution never enter science (WP_AS6 §4.5/4.6).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial

from haplotype_reconstruction.contracts.assembly import (
    AssemblyProduct,
    CompletionProduct,
    JoinedLevel,
    LevelRecord,
    RefinedLevel,
    ReleaseFamily,
)
from haplotype_reconstruction.contracts.evidence import ContigEvidence
from haplotype_reconstruction.contracts.genetic_map import ChromosomeGeneticMap
from haplotype_reconstruction.contracts.panels import LocalPanel
from haplotype_reconstruction.core.errors import SettingsError
from haplotype_reconstruction.core.ties import TieSettings
from haplotype_reconstruction.engine.graph import RecordKey, ResourceRequest, ResultRef, Task, TaskKey
from haplotype_reconstruction.science.assembly import kernel_inventory
from haplotype_reconstruction.science.assembly.evidence import AssemblyEvidence
from haplotype_reconstruction.science.assembly.hierarchy.level import LevelOutput
from haplotype_reconstruction.science.assembly.parameters import AssemblyParameters
from haplotype_reconstruction.science.assembly.run import (
    AssemblyInputs,
    AssemblyResult,
    Role,
    initial_components,
    join_settings,
    level_record,
    next_context,
)
from haplotype_reconstruction.stages.framework import Expansion
from haplotype_reconstruction.store.container import CheckpointFile
from haplotype_reconstruction.store.protocol import NumericalPolicy

from .completion import SharedEvidence, complete_stage, expand_completion
from .join_team import task_cpu_ceiling
from .levels import expand_level
from .records import (
    FINAL,
    RecordCapability,
    decode_joined,
    decode_product,
    decode_refined,
    encode_joined,
    encode_product,
    encode_refined,
    level_key,
    refined_key,
)

# Importing this stage registers its full numerical closure before engine warm-up.
KERNEL_MODULES = kernel_inventory.MODULES


@dataclass(frozen=True)
class AssemblySession:
    """One role's bound inputs, record capability and AS8 orchestration callback."""

    inputs: AssemblyInputs
    parameters: AssemblyParameters
    role: Role
    levels: int | None
    records: RecordCapability | None
    ties: TieSettings
    parent: TaskKey
    max_cores: int
    refine: Callable | None = None
    storage: SharedEvidence | None = None


@dataclass(frozen=True)
class HierarchyState:
    """Reconstructible input of a level, including the ordered L1 alphabet."""

    prepared: tuple
    components: tuple
    level: int = 1
    records: tuple = ()
    alternatives: tuple = ()
    l1_context: tuple | None = None


def start_assembly(session: AssemblySession, *, cores: int = 1) -> AssemblyProduct | Expansion:
    """Resume the product or enter completion (its blocks as engine tasks) and then the first level expansion."""
    if session.role is Role.CONTEXT and session.levels not in (1, 2):
        raise SettingsError("a context assembly requires levels 1 or 2")
    if session.role is Role.FINAL and session.levels is not None:
        raise SettingsError("final assembly needs an AS8 refinement stage and no level cap")
    return expand_completion(
        session.inputs,
        session.parameters.completion,
        session.records,
        partial(completed, session),
        ties=session.ties,
        parent=session.parent,
        max_cores=session.max_cores,
        storage=session.storage,
        cores=cores,
    )


def completed(
    session: AssemblySession, completion: CompletionProduct, *, cores: int = 1
) -> AssemblyProduct | Expansion:
    """Resume the product or enter the first level expansion of a completed contig."""
    stored = None if session.records is None or session.role is Role.CONTEXT else session.records.load(FINAL)
    if stored is not None:
        return decode_product(stored, prepared=completion.prepared)
    state = HierarchyState(completion.prepared, initial_components(completion.prepared))
    return advance(session, state, cores=cores)


def advance(session: AssemblySession, state: HierarchyState, *, cores: int = 1) -> AssemblyProduct | Expansion:
    """Load a joined level or expand its missing batch tasks."""
    records = session.records
    stored = None if records is None else records.load(level_key(state.level))
    if stored is not None:
        return after_join(session, state, decode_joined(stored))
    key = replace(session.parent, unit=f"{session.parent.unit}.level{state.level}")
    return expand_level(
        state.components,
        state.prepared,
        session.inputs.evidence,
        join_settings(session.parameters, session.inputs.genetic_map, state.level, session.ties),
        batch_size=session.parameters.batch_size,
        min_informative=session.parameters.min_boundary_informative_samples,
        records=None if records is None else records.scoped(f"l{state.level}"),
        release_alternatives=state.alternatives,
        parent=key,
        callback=partial(joined, session, state),
        max_cores=session.max_cores,
        cores=cores,
        scoring_ref=None if session.storage is None else session.storage.scoring_ref,
    )


def joined(session: AssemblySession, state: HierarchyState, result: LevelOutput) -> AssemblyProduct | Expansion:
    """Persist the join decision before any final-role refinement starts."""
    record = level_record(state.level, len(state.components), result, session.parameters, session.role, session.levels)
    alternatives = tuple(
        f.updated(source_level=f.get("source_level", state.level)) for f in result.release_alternatives
    )
    product = JoinedLevel(result.components, record, alternatives)
    if session.records is not None:
        session.records.save(level_key(state.level), encode_joined(product))
    return after_join(session, state, product)


def after_join(
    session: AssemblySession, state: HierarchyState, joined_level: JoinedLevel
) -> AssemblyProduct | Expansion:
    """Release neutral evidence only after the terminal join, before AS8 runs."""
    if session.role is Role.CONTEXT:
        return finish_level(session, state, joined_level, None)
    if joined_level.record.stop_reason is not None:
        if session.storage is None:
            session.inputs.evidence.release_neutral()
        else:
            session.storage.release_neutral()
    publish_refinement_index(session, state, joined_level)
    stored = None if session.records is None else session.records.load(refined_key(state.level))
    if stored is not None:
        return finish_level(session, state, joined_level, decode_refined(stored))
    key = replace(session.parent, unit=f"{session.parent.unit}.refine{state.level}")
    task = Task(
        key,
        execute_refinement,
        args=(session, state, joined_level),
        resources=ResourceRequest(1, session.max_cores),
        affinity=session.parent.scope,
        thread_argument="cores",
    )
    gather = Task(
        replace(key, unit=key.unit + ".gather"),
        finish_level,
        args=(session, state, joined_level, ResultRef(key)),
        depends_on=(key,),
        affinity=session.parent.scope,
    )
    return Expansion((task,), gather)


def execute_refinement(
    session: AssemblySession, state: HierarchyState, joined_level: JoinedLevel, *, cores: int = 1
) -> RefinedLevel | Expansion:
    """Delegate AS8's component expansion, or run its explicit stage callback."""
    from .refinement import refine

    action = refine if session.refine is None else session.refine
    return action(session=session, state=state, joined=joined_level, cores=cores)


def finish_level(
    session: AssemblySession, state: HierarchyState, joined_level: JoinedLevel, refined: RefinedLevel | None
) -> AssemblyProduct | Expansion:
    """Publish refinement, rebuild refined-first L1 context and advance or finish."""
    record = joined_level.record
    components, alternatives = joined_level.components, joined_level.release_alternatives
    context = state.l1_context
    if refined is not None:
        if session.records is not None:
            session.records.save(refined_key(state.level), encode_refined(refined))
        components, alternatives = refined.components, refined.release_alternatives
        context = next_context(state.level, components, joined_level.components, context)
        record = replace(record, refinement=refined.summary)
    records = (*state.records, record)
    if record.stop_reason is not None:
        product = AssemblyProduct(tuple(components), records, tuple(alternatives), state.prepared)
        if session.records is not None and session.role is Role.FINAL:
            session.records.save(FINAL, encode_product(product))
        if session.role is Role.FINAL:
            from .refinement_diagnostics import report_refinement

            report_refinement(session.parent.scope, (level.refinement or {} for level in records))
        return product
    following = HierarchyState(
        state.prepared, tuple(components), state.level + 1, records, tuple(alternatives), context
    )
    key = replace(session.parent, unit=f"{session.parent.unit}.advance{following.level}")
    task = Task(
        key,
        advance,
        args=(session, following),
        affinity=session.parent.scope,
        resources=ResourceRequest(1, session.max_cores),
        thread_argument="cores",
    )
    gather = Task(
        replace(key, unit=key.unit + ".gather"),
        identity_result,
        args=(ResultRef(key),),
        depends_on=(key,),
        affinity=session.parent.scope,
    )
    return Expansion((task,), gather)


def identity_result(value: AssemblyProduct) -> AssemblyProduct:
    """Return the completed child product to its enclosing expansion."""
    return value


def assembly_task(
    session: AssemblySession,
    *,
    depends_on: Sequence[TaskKey] = (),
    inputs: Sequence[RecordKey] = (),
    outputs: Sequence[RecordKey] = (),
) -> Task:
    """Construct the application graph's assembly[c] or context assembly action."""
    return Task(
        session.parent,
        start_assembly,
        args=(session,),
        depends_on=tuple(depends_on),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        affinity=session.parent.scope,
        resources=ResourceRequest(1, session.max_cores),
        thread_argument="cores",
    )


def bind_session(
    inputs: AssemblyInputs,
    parameters: AssemblyParameters,
    *,
    file: CheckpointFile | None,
    upstream: Mapping[str, str],
    role: Role = Role.FINAL,
    levels: int | None = None,
    records_prefix: str = "assembly",
    parent: TaskKey,
    max_cores: int = 1,
    ties: TieSettings | None = None,
    numerical_policy: NumericalPolicy | None = None,
    storage: SharedEvidence | None = None,
) -> AssemblySession:
    """Bind independently checked inputs to every resumable assembly subrecord."""
    from .records import assembly_identity

    ties = TieSettings() if ties is None else ties
    expected = assembly_identity(
        parameters,
        upstream,
        role=role.value,
        levels=levels,
        reverse=ties.reverse,
        tie_keep=ties.keep,
        numerical_policy=numerical_policy,
    )
    records = None if file is None else RecordCapability(file, records_prefix, expected)
    if storage is not None:
        inputs = replace(inputs, evidence=storage.view)
    return AssemblySession(
        inputs, parameters, role, levels, records, ties, parent, task_cpu_ceiling(max_cores), storage=storage
    )


def assemble_context(
    panels: Sequence[LocalPanel],
    evidence: AssemblyEvidence,
    genetic_map: ChromosomeGeneticMap,
    parameters: AssemblyParameters,
    *,
    level: int,
    records_prefix: str,
    file: CheckpointFile | None,
    upstream: Mapping[str, str],
    cores: int = 1,
    ties: TieSettings | None = None,
    numerical_policy: NumericalPolicy | None = None,
) -> AssemblyResult:
    """SE2's bounded context callback, using its existing task's declared team.

    IN1 supplies file, checked upstream identities, ties and cores with partial.
    The caller passes panels/evidence/map/parameters, level and records_prefix.
    Each join is published as it finishes; no new scheduler or pool is created.
    """
    from .levels import run_level_stage

    if level not in (1, 2):
        raise SettingsError("a context assembly requires levels 1 or 2")
    session = bind_session(
        AssemblyInputs(tuple(panels), evidence, genetic_map),
        parameters,
        file=file,
        upstream=upstream,
        role=Role.CONTEXT,
        levels=level,
        records_prefix=records_prefix,
        parent=TaskKey("assembly", "context", records_prefix),
        max_cores=cores,
        ties=ties,
        numerical_policy=numerical_policy,
    )
    completion = complete_stage(session.inputs, parameters.completion, session.records, ties=session.ties, cores=cores)
    working = initial_components(completion.prepared)
    families: tuple[ReleaseFamily, ...] = ()
    records: list[LevelRecord] = []
    for ordinal in range(1, level + 1):
        saved = None if session.records is None else session.records.load(level_key(ordinal))
        if saved is None:
            result = run_level_stage(
                working,
                completion.prepared,
                evidence,
                join_settings(parameters, genetic_map, ordinal, session.ties),
                batch_size=parameters.batch_size,
                min_informative=parameters.min_boundary_informative_samples,
                records=None if session.records is None else session.records.scoped(f"l{ordinal}"),
                release_alternatives=families,
                cores=cores,
            )
            record = level_record(ordinal, len(working), result, parameters, Role.CONTEXT, level)
            families = tuple(
                f.updated(source_level=f.get("source_level", ordinal)) for f in result.release_alternatives
            )
            product = JoinedLevel(result.components, record, families)
            if session.records is not None:
                session.records.save(level_key(ordinal), encode_joined(product))
        else:
            product = decode_joined(saved)
        working, families = product.components, product.release_alternatives
        records.append(product.record)
        if product.record.stop_reason is not None:
            break
    return AssemblyResult(completion.prepared, tuple(working), tuple(records), tuple(families))


def publish_refinement_index(session: AssemblySession, state: HierarchyState, joined_level: JoinedLevel) -> None:
    """Bind the component scopes under the level before AS8 or a resume reads them.

    The report reader checks this parent identity and each indexed child identity;
    it never obtains expected identities from the child publication envelopes.
    """
    from haplotype_reconstruction.science.assembly.refinement.run import RefinementSettings, bin_floor, macro_context

    from .records import component_derivation, refinement_index_key

    records = session.records
    if records is None or records.has(refinement_index_key(state.level)):
        return
    config = session.parameters.refinement
    settings = RefinementSettings(
        config,
        bin_floor(state.level, state.prepared, config),
        session.parameters.cost_scale,
        joined_level.record.stop_reason is not None and config.final_additions,
        session.parameters.count_rule,
        state.level,
        session.inputs.genetic_map,
        session.ties.reverse,
    )
    scope = records.scoped(refined_key(state.level))
    entries = []
    for number, component in enumerate(joined_level.components):
        identity = component_derivation(
            scope.identity,
            component,
            level=state.level,
            context=macro_context(component, state.l1_context),
            parameters=settings,
            additions=settings.additions,
        )
        entries.append(dict(name=f"{number:05d}", prefix=scope.prefix, identity=identity))
    records.save(refinement_index_key(state.level), dict(schema=1, components=entries))


def run_contig(
    file: CheckpointFile,
    panels: Sequence[LocalPanel],
    evidence: ContigEvidence,
    genetic_map: ChromosomeGeneticMap,
    parameters: AssemblyParameters,
    *,
    upstream: Mapping[str, str],
    parent: TaskKey,
    cores: int = 1,
    ties: TieSettings | None = None,
    numerical_policy: NumericalPolicy | None = None,
    role: Role = Role.FINAL,
    levels: int | None = None,
    records_prefix: str = "assembly",
) -> AssemblyProduct | Expansion:
    """IN1's concrete contig action, returning dynamic work to its existing engine.

    evidence is the checked ContigEvidence product. This entry constructs the
    exact assembly views once; in-process tasks share those immutable arrays.
    Isolated callers may instead bind_session with an owned SharedEvidence.
    """
    from .completion import build_evidence

    if not isinstance(evidence, ContigEvidence):
        raise TypeError("assembly run_contig requires checked ContigEvidence")
    view = build_evidence(evidence, upstream["evidence"], cores=cores)
    session = bind_session(
        AssemblyInputs(tuple(panels), view, genetic_map),
        parameters,
        file=file,
        upstream=upstream,
        role=role,
        levels=levels,
        records_prefix=records_prefix,
        parent=parent,
        max_cores=cores,
        ties=ties,
        numerical_policy=numerical_policy,
    )
    return start_assembly(session, cores=cores)
