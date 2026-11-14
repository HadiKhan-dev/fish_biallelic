"""Declare production context and final assembly entries over checked stage inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configure import ResolvedConfiguration
    from ...contracts.genetic_map import GeneticMapSet
    from ...core.ties import TieSettings

from ...core.errors import SettingsError
from ..framework import Expansion, Task, RecordKey, ResourceRequest
from ..context import StageRun
from ..inputs.products import task_inputs
from ..inputs.chain import _chain_evidence, _CHAIN_INPUTS


def _publication_result(root, contig, round_name):
    """Return the independently checked assembly publication after expansion."""
    from . import records

    reader, store = task_inputs(root)
    request = reader.request(records, "assembly", contig, round_name=round_name)
    reader.require_receipt(request, contig)
    return {RecordKey(contig, request.key): request.identity}


def run(
    root: str,
    contig: str,
    round_name: str,
    resolved: ResolvedConfiguration,
    maps: GeneticMapSet,
    ties: TieSettings,
    *,
    cores: int = 1,
) -> Expansion | dict[RecordKey, str]:
    """Expand one assembly context/final role from checked selected panels and founder evidence."""
    from ...engine.graph import Expansion, ResultRef, Task, TaskKey
    from ...science.assembly.run import AssemblyInputs, Role
    from ..assembly.final import assembly_task, AssemblySession
    from ..assembly.join_team import task_cpu_ceiling
    from ..relatedness.working_map import working_maps
    from ..selection import records as selection_records

    reader, store = task_inputs(root)
    context = None if not round_name else round_name.removeprefix("context.")
    # A role already published under the identities this run derives returns the receipt its gather
    # (_publication_result) would, after the same checks, instead of decoding the selected panels, building the
    # evidence view and decoding the stored product's diagnostics again.
    from ...engine.graph import RecordKey
    from ..assembly import records as assembly_records

    done = assembly_records.result_reader(
        store, reader.manifest, product="assembly", contig=contig, round_name=context
    )
    if reader.receipt(done, contig) is not None:
        return {RecordKey(contig, done.key): done.identity}
    capability = assembly_records.expected(store, reader.manifest, contig=contig, round_name=context)
    previous = "l2" if context is None else "initial" if context == "l1" else "l1"
    request = selection_records.result_reader(
        store, reader.manifest, product="selected", contig=contig, stage="local_selection_" + previous
    )
    selected = None if request is None else reader.read(request, contig)
    if request is None or selected is None:
        raise SettingsError(f"{contig}: assembly needs published selection round {previous}")
    evidence_request, evidence, _, source = _chain_evidence(reader, store, contig)
    from ...engine.cpu import spare_threads, thread_team
    from ...science.assembly.evidence import DEPTH_ROWS

    # The views are built in this serial chain step, which the contig's other work waits for: the task's CPUs plus
    # its lane's idle ones.
    with thread_team(spare_threads(DEPTH_ROWS)):
        view = source()  # no assembly arrays live in the cached chain tuple
    if context is None:
        from ...engine.workers import release_resident_namespace

        release_resident_namespace(_CHAIN_INPUTS)  # the final assembly is the chain inputs' last consumer
    child_key = TaskKey("assembly", "compute", contig, round_name)
    session = AssemblySession(
        AssemblyInputs(tuple(selected), view, working_maps(store, reader.manifest, maps).for_contig(contig)),
        resolved.parameters.assembly,
        Role.FINAL if context is None else Role.CONTEXT,
        None if context is None else int(context[1:]),
        capability,
        ties, child_key, task_cpu_ceiling(cores),
    )
    child = assembly_task(session)
    gather = Task(
        TaskKey("assembly", "published", contig, round_name),
        _publication_result,
        (root, contig, context),
        depends_on=(child.key,),
        affinity=contig,
    )
    return Expansion((child,), gather)


def task(key, context: StageRun) -> Task:
    """Declare one context or final assembly role on its contig worker."""
    return Task(
        key, run,
        (context.root, key.scope, key.round, context.resolved, context.maps, context.ties),
        kwargs={"cores": context.cores}, resources=ResourceRequest(1, context.cores),
    )
