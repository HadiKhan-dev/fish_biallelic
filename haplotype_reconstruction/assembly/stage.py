"""Assembly of path-selected local panels into chromosome-scale founder components.

Science (unchanged from base 92b3daf): founder completion freezes evidence-supported local
panels (prepared panels: published calls with completion fills plus the pre-fill inference
snapshot); the hierarchy then joins batches of adjacent components level by level (a
distance-aware linker fits dense haploid transitions, an endpoint beam enumerates candidate
chromosome paths, a certified panel search chooses the panel and its founder count by the
effective-carrier count rule), never across a boundary where too few samples are informative on
both sides. In the final assembly founder refinement re-chooses every component's founders as
paths of original prepared rows after each level (founder additions only at the last level).

Two roles of the same computation:
* ``Role.FINAL``: the contig's assembly product. Levels continue while they reduce the component
  count and more than one component remains (see AssemblyConfig.minimum_levels); founder
  refinement runs after every level.
* ``Role.CONTEXT``: a local-selection feedback context, through a fixed number of levels (1 or 2),
  without founder refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import hashlib
import time

import numpy as np

from ..model.components import FounderComponent, FounderComponents, PreparedPanels
from ..store.identity import canonical_json, json_digest, jsonable, stage_identity
from . import records as assembly_records
from .config import AssemblyConfig
from .evidence import AssemblyEvidence

SCHEMA = 2


class Role(enum.Enum):
    """What an assembly is for (replaces base's ``max_level == 4`` sentinel)."""

    FINAL = "final"
    CONTEXT = "context"


@dataclass(frozen=True)
class AssemblyInputs:
    """What one assembly reads.

    panels: the path-selected local panels (``model.local_panels.LocalPanel``) in genomic order,
        one per assembled block (blocks without a panel are not assembled).
    evidence: the contig's assembly evidence view (:class:`AssemblyEvidence`).
    genetic_map: the contig's ``model.genetic_map.ChromosomeGeneticMap``; linker distances and
        the beam's maximum gap read it (its fallback rate per bp when no map file is given).
    panels_identity: identity of the panels' upstream record (the final assembly: the l2 local
        selection record's identity); None: a content digest of the panels is used (contexts).
    """

    panels: tuple
    evidence: AssemblyEvidence
    genetic_map: object
    panels_identity: str | None = None


@dataclass(frozen=True)
class LevelRecord:
    """One hierarchy level (stored in the level record; logged).

    level: 1-based; batch_size: components per batch; components_before / components_after;
    stop_reason: why the loop stopped after this level (None: it continued):
        "single_component", "component_count_not_reduced", "only_phase_breaks_remain" (final role,
        from ``minimum_levels`` on), "context_levels_reached" (context role);
    plan: per batch (first, end, digest | None) over the level's input components;
    joins: per joined batch (batch order), its search summary (``join.search.JoinSummary``; reports
        read them); log_likelihoods: per joined batch, the chosen panel's join objective (nats);
    reused_joins: joined batches whose join record already existed (any assembly of the contig);
    boundary_joint: samples informative on both sides of each input boundary;
    refinement: founder refinement's level summary as a plain record (final role), else None;
    elapsed_seconds / refinement_elapsed_seconds: wall time of the hierarchy and refinement.
    """

    level: int
    batch_size: int
    components_before: int
    components_after: int
    stop_reason: str | None
    plan: tuple
    joins: tuple
    log_likelihoods: tuple
    reused_joins: int
    boundary_joint: tuple
    refinement: dict | None = None
    elapsed_seconds: float = 0.0
    refinement_elapsed_seconds: float | None = None
    shadows: tuple = ()

    def record(self) -> dict:
        """Plain record (stored in the level and final records)."""
        return dict(level=self.level, batch_size=self.batch_size,
                    components_before=self.components_before,
                    components_after=self.components_after, stop_reason=self.stop_reason,
                    plan=[list(item) for item in self.plan],
                    joins=[summary.record() for summary in self.joins],
                    log_likelihoods=list(self.log_likelihoods), reused_joins=self.reused_joins,
                    boundary_joint=list(self.boundary_joint), refinement=self.refinement,
                    elapsed_seconds=self.elapsed_seconds,
                    refinement_elapsed_seconds=self.refinement_elapsed_seconds, shadows=self.shadows)

    @classmethod
    def from_record(cls, record: dict) -> "LevelRecord":
        from .join.search import JoinSummary
        return cls(level=record["level"], batch_size=record["batch_size"],
                   components_before=record["components_before"],
                   components_after=record["components_after"],
                   stop_reason=record["stop_reason"],
                   plan=tuple(tuple(item) for item in record["plan"]),
                   joins=tuple(JoinSummary.from_record(item) for item in record["joins"]),
                   log_likelihoods=tuple(record["log_likelihoods"]),
                   reused_joins=record["reused_joins"],
                   boundary_joint=tuple(record["boundary_joint"]),
                   refinement=record["refinement"], elapsed_seconds=record["elapsed_seconds"],
                   refinement_elapsed_seconds=record["refinement_elapsed_seconds"],
                   shadows=tuple(record.get("shadows", ())))


@dataclass(frozen=True)
class AssemblyResult:
    """prepared: the prepared panels (one per input panel, same order); components: the phase
    components (row paths over ``prepared``; count calls set in the final role); levels: one record
    per executed level; identity: the assembly's identity (bound to its records)."""

    prepared: PreparedPanels
    components: FounderComponents
    levels: tuple[LevelRecord, ...]
    identity: str
    release_alternatives: tuple = ()


def panels_digest(panels) -> str:
    """Content digest of the local panels an assembly reads (its upstream identity)."""
    digest = hashlib.sha256(b"local-panels\0")
    for panel in panels:
        digest.update(f"{int(panel.block)}:{int(panel.site_start)}:{float(panel.unknown_mass)!r}".encode())
        for array in (panel.positions, panel.kept, panel.calls, panel.alt_probability, panel.support,
                      panel.has_depth, panel.unknown_slots):
            value = np.ascontiguousarray(array)
            digest.update(f"{value.dtype.str}{value.shape}".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def assembly_identity(inputs: AssemblyInputs, config: AssemblyConfig, role: Role,
                      levels: int | None) -> dict:
    """The identity of one assembly: stage config, role, upstream identities, code identity.

    A context assembly never refines, so its identity leaves the refinement settings out (changing
    them does not invalidate the contexts' records, as in base).
    """
    settings = jsonable(config)
    if role is Role.CONTEXT:
        settings.pop("refinement")
    identity = stage_identity("assembly", schema=SCHEMA, config=settings, upstream={
        "evidence": inputs.evidence.identity,
        "local_panels": (inputs.panels_identity if inputs.panels_identity is not None
                         else panels_digest(inputs.panels)),
        "genetic_map": json_digest(inputs.genetic_map.identity())})
    identity["role"] = role.value
    identity["levels"] = levels
    return identity


def stop_reason(role: Role, levels: int | None, level: int, before: int, components,
                minimum_levels: int) -> str | None:
    """Why the level loop stops after ``level`` (None: run the next level).

    A single component, or a level (after the first) that did not reduce the count, ends any
    assembly. A context assembly stops at its level count. The final assembly runs at least
    ``minimum_levels`` levels (while they reduce the count: base ran exactly four) and continues
    beyond them only while some boundary between adjacent components is not a phase break, i.e.
    while a further level could still join something (B4).
    """
    after = len(components)
    if after == 1:
        return "single_component"
    if level >= 2 and after >= before:
        return "component_count_not_reduced"
    if role is Role.CONTEXT:
        return "context_levels_reached" if level >= levels else None
    if level >= minimum_levels and all(c.break_after is not None for c in components[:-1]):
        return "only_phase_breaks_remain"
    return None


def _phase(budget, role: Role, levels: int | None, name: str):
    prefix = "final" if role is Role.FINAL else f"feedback_l{levels}"
    return budget.phase(f"{prefix}_{name}")


def assemble(inputs: AssemblyInputs, config: AssemblyConfig, *, role: Role,
             levels: int | None = None, records=None, budget=None, queue=None) -> AssemblyResult:
    """Assemble one contig's local panels (the stage's one entry point).

    role: Role.FINAL (``levels`` must be None) or Role.CONTEXT with ``levels`` in {1, 2}.
    records: the section's ``store.RecordScope`` (final: ``RecordScope(file, "assembly")``, written by
        the driver's contig wrapper; contexts: the caller's scope) or None (nothing recorded or
        resumed). The identity is bound first; completion, every level's plan and output, every
        refined level and every joined batch (``joins/<digest>``, shared by all assemblies of the
        contig) are recorded as they complete, and a rerun with identical inputs resumes from them.
    budget: the CPU budget (``execution.CpuBudget``, or a thread count) of this contig's assembly.
    queue: the cross-node batch queue directory (``[execution] batch_queue``) or None (local pool).
    Inputs are never modified.
    """
    from ..execution.budget import as_budget
    from .completion.stage import CompletionSummary, complete
    from .hierarchy.batch import JoinSettings
    from .hierarchy.level import run_level
    if role is Role.FINAL and levels is not None:
        raise ValueError("the final assembly runs its levels to completion (levels must be None)")
    if role is Role.CONTEXT and levels not in (1, 2):
        raise ValueError("a context assembly runs through level 1 or 2")
    budget = as_budget(1 if budget is None else budget)
    evidence = inputs.evidence.ensure_shared()     # pool workers attach its arrays (no copies)
    identity = assembly_identity(inputs, config, role, levels)
    if records is not None:
        records.bind(identity)

    stored = None if records is None else records.load(assembly_records.COMPLETION)
    if stored is not None:
        prepared = assembly_records.decode_prepared(stored["prepared"])
        summary = CompletionSummary.from_record(stored["summary"])
    else:
        prepared, summary = complete(inputs.panels, evidence, config.completion,
                                     _phase(budget, role, levels, "preprocess"))
        if records is not None:
            records.save(assembly_records.COMPLETION, dict(
                schema=assembly_records.SCHEMA, prepared=assembly_records.encode_prepared(prepared),
                summary=summary.record()))
    print(f"[assembly] {role.value}: completion of {len(prepared)} blocks: {summary.describe()}",
          flush=True)

    working = tuple(FounderComponent(index, index + 1, np.arange(panel.rows, dtype=np.int32)[:, None])
                    for index, panel in enumerate(prepared))
    level_records = []
    release_alternatives = []
    l1_context = None
    level = 0
    while True:
        level += 1
        before = len(working)
        level_scope = None if records is None else records.scoped(f"l{level}")
        stored = None if records is None else records.load(assembly_records.level_key(level))
        if stored is not None:
            output = assembly_records.decode_components(stored["components"])
            record = LevelRecord.from_record(stored["record"])
            level_alternatives = tuple(stored.get("release_alternatives", ()))
        else:
            settings = JoinSettings(
                linker=config.linker, join=config.join, count_rule=config.count_rule,
                cost_scale=config.cost_scale, n_generations=config.n_generations,
                recombination_tolerance=config.recombination_tolerance, level_one=level == 1,
                genetic_map=inputs.genetic_map)
            result = run_level(working, prepared, evidence, settings, batch_size=config.batch_size,
                               min_informative=config.min_boundary_informative_samples,
                               records=level_scope,
                               budget=_phase(budget, role, levels, f"hierarchy_l{level}"),
                               queue=queue, release_alternatives=release_alternatives)
            output = result.components
            level_alternatives = tuple(dict(item, source_level=item.get("source_level", level))
                                       for item in result.release_alternatives)
            record = LevelRecord(
                level=level, batch_size=config.batch_size, components_before=before,
                components_after=len(output),
                stop_reason=stop_reason(role, levels, level, before, output, config.minimum_levels),
                plan=result.plan, joins=result.joins, log_likelihoods=result.log_likelihoods,
                reused_joins=result.reused, boundary_joint=tuple(int(v) for v in result.partition.joint),
                elapsed_seconds=result.elapsed_seconds, shadows=result.shadows)
            if records is not None:
                records.save(assembly_records.level_key(level), dict(
                    schema=assembly_records.SCHEMA,
                    components=assembly_records.encode_components(output), record=record.record(),
                    release_alternatives=level_alternatives))
        release_alternatives = list(level_alternatives)
        _log_level(role, record)
        working = output
        if role is Role.FINAL:
            working, record, refined_alternatives = _refine(
                level, record, working, prepared, evidence, l1_context, config,
                records, _phase(budget, role, levels, f"refinement_l{level}"),
                release_alternatives=release_alternatives)
            release_alternatives = list(refined_alternatives)
            if level == 1:
                # Refined paths feed the next level; both refined and unrefined L1 alternatives stay
                # available to the refinement's coarse moves, so an early local optimum cannot erase
                # a useful chromosome-wide move.
                l1_context = (*working, *output)
        level_records.append(record)
        if record.stop_reason is not None:
            break
    if role is Role.FINAL and records is not None and not records.has(assembly_records.FINAL):
        records.save(assembly_records.FINAL, dict(
            schema=assembly_records.SCHEMA, components=assembly_records.encode_components(working),
            levels=[item.record() for item in level_records],
            release_alternatives=release_alternatives))
    return AssemblyResult(prepared=prepared, components=working, levels=tuple(level_records),
                          identity=canonical_json(jsonable(identity)),
                          release_alternatives=tuple(release_alternatives))


def _refine(level, record, components, prepared, evidence, l1_context, config, records, budget,
            *, release_alternatives=()):
    """Founder refinement after a final level (resumed from ``l<k>/refined`` when recorded)."""
    from dataclasses import replace
    from .founder_refinement import LevelInput, refine_level
    stored = None if records is None else records.load(assembly_records.refined_key(level))
    if stored is not None:
        refined = assembly_records.decode_components(stored["components"])
        return refined, replace(record, refinement=stored["summary"],
                                refinement_elapsed_seconds=stored["elapsed_seconds"]), tuple(
                                    stored.get("release_alternatives", ()))
    started = time.perf_counter()
    result = refine_level(
        LevelInput(level=level, prepared=prepared, components=tuple(components),
                   evidence=evidence.scoring, l1_context=l1_context, cost_scale=config.cost_scale,
                   additions=record.stop_reason is not None, count_rule=config.count_rule,
                   release_alternatives=tuple(release_alternatives),
                   evidence_ref=None if evidence.shared is None else evidence.shared.scoring),
        config.refinement, budget,
        None if records is None else records.scoped(assembly_records.refine_scope(level), nthreads=1))
    elapsed = time.perf_counter() - started
    summary = jsonable(result.summary)
    release_alternatives = tuple(result.release_alternatives)
    if records is not None:
        records.save(assembly_records.refined_key(level), dict(
            schema=assembly_records.SCHEMA,
            components=assembly_records.encode_components(result.components), summary=summary,
            elapsed_seconds=elapsed, release_alternatives=release_alternatives))
    return tuple(result.components), replace(record, refinement=summary,
                                             refinement_elapsed_seconds=elapsed), release_alternatives


def _log_level(role: Role, record: LevelRecord) -> None:
    """One line per level, plus the join census lines (base hierarchy's)."""
    joined = sum(1 for _first, _end, digest in record.plan if digest is not None)
    print(f"[assembly] {role.value} L{record.level}: {record.components_before} -> "
          f"{record.components_after} components; {joined} joined batches "
          f"({record.reused_joins} reused), stop: {record.stop_reason}", flush=True)
    if record.joins:
        from .join.search import census_lines
        for line in census_lines(record.joins):
            print(f"  {line}", flush=True)
