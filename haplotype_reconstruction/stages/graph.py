"""Declare the model dependencies and workflow cuts without scheduling science.

Execution barriers are policy in engine.scheduler. Relatedness consumes
all evidence, refit consumes painting, and only pedigree consumes relatedness.
Stage adapters supply concrete importable actions; this module owns no science.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..core.errors import SettingsError
from ..contracts.contigs import AnalysisContigs
from .catalog import STAGES
from ..engine.graph import RecordKey, Task, TaskKey
from ..store.container import CheckpointFile
from ..store.identity import code_dependency_identity
from ..store.protocol import Derivation, NumericalPolicy

__all__ = [
    "Node",
    "RunGraph",
    "Cut",
    "build_graph",
    "workflow_cut",
    "bind_tasks",
    "derivation",
    "preload_modules",
    "run_checks",
    "execution_options",
    "complete_cut",
]


@dataclass(frozen=True)
class Node:
    """A model node, its upstream nodes, and external content identities.

    External inputs name content, never filesystem paths or completion markers.
    ``code_roots`` are supplied by the owning stage when it binds an action.
    """

    key: TaskKey
    dependencies: tuple[TaskKey, ...] = ()
    content_inputs: tuple[str, ...] = ()
    code_roots: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.key.stage + ("." + self.key.round if self.key.round else "")


@dataclass(frozen=True)
class RunGraph:
    """The bound contig order and model graph, independent of resource choices."""

    contigs: tuple[str, ...]
    nodes: tuple[Node, ...]
    analysis: AnalysisContigs | None = None

    @property
    def edges(self) -> tuple[tuple[TaskKey, TaskKey], ...]:
        return tuple((parent, node.key) for node in self.nodes for parent in node.dependencies)

    def node(self, key: TaskKey) -> Node:
        for node in self.nodes:
            if node.key == key:
                return node
        raise SettingsError(f"unknown model node: {key}")

    def closure(self, targets: Sequence[TaskKey]) -> tuple[TaskKey, ...]:
        indexed = {node.key: node for node in self.nodes}
        seen: set[TaskKey] = set()
        todo = list(targets)
        while todo:
            key = todo.pop()
            if key in seen:
                continue
            if key not in indexed:
                raise SettingsError(f"unknown model node: {key}")
            seen.add(key)
            todo.extend(indexed[key].dependencies)
        return tuple(node.key for node in self.nodes if node.key in seen)


@dataclass(frozen=True)
class Cut:
    """Selected work and identity-checked prerequisites supplied by prior invocations.

    Serial order is a scheduling constraint, never an extra derivation edge.
    A max-contigs cut ends cleanly, including when zero new contigs are selected.
    """

    step: str
    nodes: tuple[TaskKey, ...]
    prerequisites: tuple[TaskKey, ...]
    serial_order: tuple[str, ...] = ()
    clean_stop: bool = False


def _key(stage: str, contig: str = "run", round_: str = "") -> TaskKey:
    return TaskKey(stage, "run" if contig == "run" else "contig", contig, round_)


def build_graph(
    contigs: Sequence[str], *, output_nodes: Sequence[Node] = (), analysis: AnalysisContigs | None = None
) -> RunGraph:
    """Carry ENG-03/05 as model edges; context assemblies share contig affinity.

    The initial selection has no context. L1 and L2 contexts consume the previous
    round. Final assembly consumes L2 and releases its founder-view workspace.
    Caller-supplied export/report nodes carry their own declared dependencies.
    """
    order = tuple(contigs)
    if len(set(order)) != len(order) or "run" in order:
        raise SettingsError("contig identifiers must be unique and must not be 'run'")
    analysis = AnalysisContigs(order) if analysis is None else analysis
    init = _key("init")
    nodes = [Node(init, content_inputs=("resolved_configuration", "source_descriptor"))]
    for c in order:
        evidence = _key("evidence", c)
        nodes.append(Node(evidence, (init,), (f"payload:{c}", "EvidenceParameters")))
    relatedness = _key("relatedness")
    nodes.append(Node(relatedness, tuple(_key("evidence", c) for c in order), ("maps",)))
    for c in order:
        ev, discovery = _key("evidence", c), _key("discovery", c)
        initial, l1, l2 = (_key("selection", c, r) for r in ("initial", "l1", "l2"))
        context1, context2 = (_key("assembly", c, r) for r in ("context.l1", "context.l2"))
        assembly, painting, refit = (_key(s, c) for s in ("assembly", "painting", "refit"))
        maps = (f"map:{c}",)
        nodes.extend(
            (
                Node(discovery, (ev,)),
                Node(initial, (discovery, ev), maps),
                Node(context1, (initial, ev), maps),
                Node(l1, (discovery, initial, context1), maps),
                Node(context2, (l1, ev), maps),
                Node(l2, (discovery, l1, context2), maps),
                Node(assembly, (l2, ev), maps),
                Node(painting, (assembly, ev), maps),
                Node(refit, (painting, ev)),
            )
        )
    pedigree = _key("pedigree")
    nodes.append(
        Node(
            pedigree,
            (relatedness,) + tuple(_key("painting", c) for c in analysis.included)
            + tuple(_key("evidence", c) for c in analysis.included),
            ("maps", "relatedness/families"),
        )
    )
    for c in analysis.included:
        phase = _key("phase", c)
        nodes.extend(
            (
                Node(
                    phase,
                    (pedigree, _key("painting", c), _key("refit", c), _key("evidence", c)),
                    (f"map:{c}", "pedigree_edges_content"),
                ),
                Node(_key("recombination", c), (phase, pedigree), (f"map:{c}", "pedigree_edges_content")),
            )
        )
    nodes.extend(output_nodes)
    indexed = {node.key: node for node in nodes}
    if len(indexed) != len(nodes):
        raise SettingsError("duplicate graph node")
    for node in nodes:
        if set(node.dependencies) - indexed.keys():
            raise SettingsError(f"{node.key}: undeclared graph dependency")
    graph = RunGraph(order, tuple(nodes), analysis)
    # Detect cycles even in user-supplied output nodes.
    done: set[TaskKey] = set()
    while len(done) < len(nodes):
        ready = {n.key for n in nodes if n.key not in done and set(n.dependencies) <= done}
        if not ready:
            raise SettingsError("model graph contains a cycle")
        done.update(ready)
    return graph


def workflow_cut(
    graph: RunGraph,
    step: str = "run",
    *,
    contig: str | None = None,
    contigs: Sequence[str] | None = None,
    stop_after: str | None = None,
    serial: bool = False,
    max_contigs: int | None = None,
    painted: Sequence[str] = (),
) -> Cut:
    """Select model nodes; ``painted`` must come from expected record identities.

    A workflow step does not silently recompute its prerequisites. A run includes
    prerequisite closure and early relatedness/refits whose inputs are in the cut.
    """
    if contig is not None and contigs is not None:
        raise SettingsError("use contig or contigs, not both")
    wanted = set(graph.contigs if contigs is None else contigs)
    if contig is not None:
        wanted = {contig}
    if wanted - set(graph.contigs):
        raise SettingsError(f"unknown contigs: {sorted(wanted - set(graph.contigs))}")
    requested = tuple(c for c in graph.contigs if c in wanted)
    scatter = requested
    if step == "run" and graph.analysis is not None:
        # Run cuts spend reconstruction only on the included axis. Evidence is
        # retained separately because relatedness still consumes the configured genome.
        scatter = tuple(c for c in requested if c in graph.analysis.included)
    if max_contigs is not None:
        if step != "run" or max_contigs < 0:
            raise SettingsError("max_contigs is a nonnegative run limit")
        pending = [c for c in scatter if c not in set(painted)]
        scatter = tuple(pending[:max_contigs])
        serial = True
        stop_after = "painting"
    if stop_after is not None and (step != "run" or stop_after not in {"relatedness", "discovery", "painting"}):
        raise SettingsError(f"unsupported stop point: {stop_after}")
    keys = {n.key for n in graph.nodes}
    if step == "run":
        if stop_after == "relatedness":
            targets = [_key("relatedness")]
        elif stop_after:
            targets = [_key(stop_after, c) for c in scatter]
        else:
            targets = [n.key for n in graph.nodes if n.key.scope == "run" or n.key.scope in scatter]
        if graph.analysis is not None:
            targets.extend(_key("evidence", c) for c in requested if c in graph.analysis.excluded)
        selected = set(graph.closure(targets))
        # A restricted run never schedules scattered work outside its selection;
        # global gathers retain these as identity-checked prerequisites.
        selected = {
            k for k in selected
            if k.scope == "run" or k.scope in scatter or (k.stage == "evidence" and k.scope in requested)
        }
        for node in graph.nodes:
            if node.key.stage in {"relatedness", "refit"} and set(node.dependencies) <= selected:
                if node.key.scope == "run" or node.key.scope in scatter:
                    selected.add(node.key)
    else:
        stages = {name for name, stage in STAGES.items() if step in stage.steps}
        if not stages:
            raise SettingsError(f"unknown workflow step: {step}")
        if step in {"evidence", "reconstruct", "phase"} and contig is None:
            raise SettingsError(f"{step} requires one contig")
        selected = {k for k in keys if k.stage in stages and (k.scope == "run" or k.scope in scatter)}
    prerequisites = {p for n in graph.nodes if n.key in selected for p in n.dependencies} - selected
    return Cut(
        step,
        tuple(n.key for n in graph.nodes if n.key in selected),
        tuple(n.key for n in graph.nodes if n.key in prerequisites),
        scatter if serial else (),
        max_contigs is not None or stop_after is not None,
    )


def bind_tasks(
    graph: RunGraph, cut: Cut, actions: Mapping[TaskKey, Task], *, completed: Mapping[TaskKey, Any] | None = None
) -> tuple[Task, ...]:
    """Bind stage actions, requiring checked upstream results for workflow cuts.

    ``completed`` is supplied by protocol resume.
    The scheduler receives it as ``initial_results``; no marker is consulted.
    """
    completed = {} if completed is None else completed
    missing = set(cut.prerequisites) - completed.keys()
    if missing:
        raise SettingsError(f"workflow prerequisites are not published: {sorted(missing)}")
    tasks = []
    for key in cut.nodes:
        if key not in actions:
            raise SettingsError(f"no stage action for {key}")
        node, task = graph.node(key), actions[key]
        if task.key != key:
            raise SettingsError(f"stage action key {task.key} does not match logical graph key {key}")
        dependencies = tuple(dict.fromkeys((*node.dependencies, *task.depends_on)))
        # A contig's inference chain shares its pinned worker. Recombination, export and reports consume
        # published records and retain no chain state, so any free worker may run them.
        affinity = task.affinity
        if affinity is None and key.scope != "run" and key.stage not in {"report", "recombination", "export"}:
            affinity = key.scope
        tasks.append(replace(task, depends_on=dependencies, affinity=affinity))
    return tuple(tasks)


def preload_modules(tasks: Sequence[Task]) -> tuple[str, ...]:
    """Declare the actual supplied stage task modules for the runtime boundary."""
    return tuple(dict.fromkeys(task.function.__module__ for task in tasks))


def derivation(
    node: Node,
    parameters: Any,
    inputs: Mapping[str, str],
    *,
    numerical_policy: NumericalPolicy,
    tie_reverse: bool = False,
    tie_keep: tuple[str, ...] = (),
) -> Derivation:
    """Bind science entry, adapter, input construction and codec code roots."""
    if not node.code_roots:
        raise SettingsError(f"{node.key}: derivation requires declared code roots")
    return Derivation(
        code_dependency_identity(node.code_roots), parameters, numerical_policy, dict(inputs), tie_reverse, tie_keep
    )


def run_checks(
    contigs: Sequence[str],
    *,
    genome_wide: bool,
    parameters: Any = None,
    minimum_scored_contigs: int | None = None,
    tier_b_minimum_contigs: int | None = None,
) -> tuple[str, ...]:
    """Return the driver's warning text; the application decides where to print it."""
    if not genome_wide:
        return ()
    if parameters is not None:
        import math

        from ..science.pedigree.resampling import minimum_contigs_for_tier

        # The exposure rule supplies the minimum; pedigree's tier function determines reach.
        minimum_scored_contigs = math.ceil(parameters.exposure.minimum_exposed_units)
        tier_b_minimum_contigs = minimum_contigs_for_tier(parameters.release.tier_b, minimum_scored_contigs)
    if minimum_scored_contigs is None or tier_b_minimum_contigs is None:
        raise SettingsError("pedigree parameters or both explicit contig thresholds are required")
    if len(contigs) < tier_b_minimum_contigs:
        return ("configured contigs are below the pedigree tier-B minimum; informative contigs are rechecked",)
    return ()


def execution_options(
    graph: RunGraph,
    cut: Cut,
    *,
    contig_costs: Mapping[str, float] | None = None,
    phase_costs: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Translate ENG-03 early nodes and ENG-23 contig costs into executor policy.

    Chain costs are application-supplied size estimates, never measured pressure.
    Phase size estimates may be supplied separately. Early ready refits
    and the relatedness gather precede ordinary chain work. Ready recombination
    precedes export; both consume immutable phase products on any free worker.
    These priorities/leases add no model edge or scientific identity input.
    """
    from ..engine.lanes import ChainPlan

    costs = {c: 1.0 if contig_costs is None else float(contig_costs[c]) for c in graph.contigs}
    if any(value < 0 for value in costs.values()):
        raise SettingsError("contig planning costs must be nonnegative")
    scale = 1.0 + sum(costs.values())
    # Preserve the early-refit/relatedness/recombination priority classes while
    # ordering phase contigs by size within their own class.
    phase_priorities = costs
    if phase_costs is not None:
        phase_total = max(1.0, sum(phase_costs.values()))
        phase_priorities = {c: (scale - 1.0) * value / phase_total for c, value in phase_costs.items()}
    priorities = {
        key: (
            3 * scale
            if key.stage == "refit"
            else 2 * scale
            if key.stage == "relatedness"
            else scale
            if key.stage == "recombination"
            else phase_priorities.get(key.scope, 0.0)
            if key.stage == "phase"
            else costs.get(key.scope, 0.0)
        )
        for key in cut.nodes
    }
    chain_stages = {name for name, stage in STAGES.items() if "reconstruct" in stage.steps}
    chains = tuple(
        ChainPlan(c, costs[c], tuple(key for key in cut.nodes if key.scope == c and key.stage in chain_stages))
        for c in graph.contigs
        if any(key.scope == c and key.stage in chain_stages for key in cut.nodes)
    )
    return dict(priorities=priorities, chains=chains, serial_order=cut.serial_order)


def complete_cut(
    cut: Cut,
    containers: Mapping[str, CheckpointFile],
    outputs: Mapping[TaskKey, Mapping[RecordKey, str]],
    marker: Any,
    *,
    map_work: Any = None,
) -> bool:
    """Write a workflow marker last, only after checking every expected record.

    This also repairs a missing marker after process death without recomputation.
    Markers never enter the completion decision. Append-only records make the
    multi-container check stable while the final marker is atomically placed.
    ``map_work(function, items)`` (ordered results) checks independent contig containers concurrently,
    avoiding a serial tail of file reads and decodes. A mismatch propagates through the map.
    """
    from ..store.protocol import is_complete

    grouped: dict[str, dict[str, str]] = {}
    for task in cut.nodes:
        if task not in outputs or not outputs[task]:
            return False
        for key, identity in outputs[task].items():
            previous = grouped.setdefault(key.container, {}).get(key.key)
            if previous is not None and previous != identity:
                raise SettingsError(f"conflicting completion identities: {key}")
            grouped[key.container][key.key] = identity
    if not grouped:
        return False
    if map_work is None:
        for container, records in grouped.items():
            if not is_complete(containers[container], records):
                return False
    elif not all(map_work(lambda item: is_complete(containers[item[0]], item[1]), list(grouped.items()))):
        return False
    # Stage actions own complete/<step> in their permitted container. The cut
    # only writes its external marker after all checked publications exist.
    import os
    import tempfile
    from pathlib import Path

    from ..core.digest import json_digest

    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json_digest(grouped) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return True


def stage_modules(task):
    """Return the implementation owners behind a task's stage and product variant."""
    spec = STAGES.get(task.key.stage)
    if spec is None:
        return None
    modules = spec.pair_owners if task.key.stage == "relatedness" and task.key.unit == "pairs" else spec.owners
    result = tuple("haplotype_reconstruction." + module for module in modules)
    if task.key.stage == "assembly":
        from ..science.assembly.kernel_inventory import MODULES

        result += tuple(module.__name__ for module in MODULES)
    return result


def task_preloads(tasks: Sequence[Task]) -> tuple[str, ...]:
    """Name the stage implementations behind application wrappers before workers warm their kernels."""

    modules = list(preload_modules(tasks))
    for task in tasks:
        modules.extend(stage_modules(task) or ())
    return tuple(dict.fromkeys(modules))


def stage_kernel_modules(task: Task) -> tuple[str, ...]:
    """Numerical ownership within the stage's static closure, including dynamically expanded work."""
    from ..store.identity import code_dependency_identity

    spec = STAGES.get(task.key.stage)
    if spec is None:
        return ()
    roots = spec.pair_kernel_roots if task.key.stage == "relatedness" and task.key.unit == "pairs" else spec.kernel_roots
    if not roots:
        return ()
    # Tie kernels are the common numerical foundation; source identity still traverses the full task closure.
    numerical_roots = tuple("haplotype_reconstruction." + root for root in roots)
    modules = code_dependency_identity((*numerical_roots, "core.ties")).modules
    science = spec.kernels
    if task.key.stage == "selection" and task.key.round in {"initial", "l1", "l2"}:
        science = ("selection", "assembly.evidence", "shared.emissions")
        if task.key.round != "initial":
            science += ("assembly.alleles",)
        else:
            modules = tuple(m for m in modules if m != "haplotype_reconstruction.science.selection.context_refit")
    if task.key.stage == "assembly" and task.key.round.startswith("context."):
        # CONTEXT returns after the join; only FINAL expands final refinement refinement.
        modules = tuple(m for m in modules if not m.startswith("haplotype_reconstruction.science.assembly.refinement."))
    prefixes = tuple(
        "haplotype_reconstruction." + name + "."
        for name in ("core", "contracts", *("science." + name for name in science))
    )
    return tuple(
        module
        for module in modules
        if (module + ".").startswith(prefixes)
        or module == "haplotype_reconstruction.science.evidence.likelihoods"
        or module in numerical_roots
    )


def with_kernel_tasks(
    tasks: tuple[Task, ...], scheduling: dict[str, Any], state: Any
) -> tuple[tuple[Task, ...], dict[str, Any]]:
    """Gate each scientific root on its own missing compiles, leaving unrelated work runnable.

    A root's dependency also protects its dynamic expansions: no child can use a tail signature before its module
    has been warmed in the shared run cache. Only state.missing emits tasks or edges. Compilation retains the
    earliest-consumer/learned-cost order; independent stages keep their existing priority over spare warm-up work.
    """
    from ..engine.kernels import compile_tasks

    compiles: dict[tuple[str, str], list[Task]] = {}
    for task in compile_tasks(state):  # one per missing overload, grouped by function
        compiles.setdefault(task.args[0], []).append(task)
    if not compiles:
        return tasks, scheduling
    by_module: dict[str, list[tuple[str, str]]] = {}
    for name in compiles:
        by_module.setdefault(name[0], []).append(name)
    indexed = {task.key: task for task in tasks}
    levels: dict[TaskKey, int] = {}

    def level(key: TaskKey) -> int:  # the longest dependency path to ``key`` within this graph
        if key not in levels:
            levels[key] = 1 + max((level(d) for d in indexed[key].depends_on if d in indexed), default=-1)
        return levels[key]

    first_use: dict[tuple[str, str], int] = {}
    gated = []
    for task in tasks:
        dependencies = list(task.depends_on)
        for module in stage_kernel_modules(task):
            for name in by_module.get(module, ()):
                first_use[name] = min(first_use.get(name, level(task.key)), level(task.key))
                dependencies.extend(compile.key for compile in compiles[name])
        gated.append(replace(task, depends_on=tuple(dict.fromkeys(dependencies))))
    unused = 1 + max(levels.values(), default=0)
    order = sorted(
        compiles,
        key=lambda name: (
            first_use.get(name, unused),
            -state.seconds.get(name, 0.0),
            -len(state.missing[name]),
            name,
        ),
    )
    priorities = dict(scheduling.get("priorities") or {})
    floor = min(priorities.values(), default=0.0) - 1.0
    priorities.update({compile.key: floor - rank for rank, name in enumerate(order) for compile in compiles[name]})
    return (*gated, *(compile for name in order for compile in compiles[name])), dict(scheduling, priorities=priorities)
