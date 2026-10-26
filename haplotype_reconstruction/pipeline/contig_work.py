"""One contig's reconstruction after discovery: local selection -> final assembly -> painting.

`reconstruct_contig` is the work a contig lane (or the serial/sequential route) runs; it resumes from each stage's
records. The founder-view assembly evidence is built once per contig and serves the local-selection context
assemblies and the final assembly (as base); it and the assembly products are released before painting, which
reads every sample's evidence (release and painting pools never overlap).

The final-role assembly wrapper lives here (blueprint 11.1: assembly never imports local selection): it loads the
l2 local panels and the evidence, calls assembly.assemble(role=FINAL) with the contig's record scope, and the
assembly records itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import execution

ASSEMBLY_SCOPE = "assembly"
FINAL_ROUND = "l2"

# Simultaneous selection/context arrays, checkpoint copies and block scratch. The named array subtotals are
# 138 founder-site bytes during selection and 167 during the second context; 200 is a planning allowance,
# not a hard bound on Python memos, decoded backing groups or all search trajectories.
SELECTION_BYTES_PER_FOUNDER_SITE = 200


@dataclass(frozen=True)
class ContigContext:
    """What every contig's work needs besides the store (picklable: it travels to the lanes)."""

    configs: Any                    # config.stages.StageConfigs
    genetic_maps: Any               # model.genetic_map.GeneticMapSet
    queue: Path | None = None       # [execution] batch_queue
    output: Path | None = None      # run directory (published tables)


@dataclass(frozen=True)
class ContigSummary:
    contig: str
    resumed: bool
    painting: dict


def contig_done(store, contig) -> bool:
    """Whether the contig's per-contig work (through painting) is recorded."""
    from ..painting import records as painting_records
    return painting_records.complete(store.contig(contig))


def planning_bytes(store, contig, n_samples: int) -> int:
    """Lane-owned evidence and selection arrays, estimated without decoding scientific tensors.

    Raw/neutral float64 GLs, float32 scoring and three masks coexist (63NL). Discovery decoded bytes D
    include at least13 bytes per selected founder-site; D/13 plus the existing search-growth allowance
    estimates the founder-site width of overlapping selection generations, proposals and context arrays.
    Counts can grow beyond this planning allowance; it is neither a founder cap nor an exact peak bound.
    Pool copies, task scratch and process residence have separate owners and reservations.
    """
    from ..local_selection.rounds import SEARCH_GROWTH
    file = store.contig(contig)
    sites = len(file.get("evidence", items=("positions",))["positions"])
    discovered = sum(info.decoded_bytes for info in file.infos("discovery/"))
    founder_sites = -(-discovered // 13) + SEARCH_GROWTH * sites
    evidence = n_samples * sites * (2 * 3 * 8 + 3 * 4 + 3)
    # The completion-keep control also retains transient REF/ALT depth-key construction.
    completion_depth = 8 * n_samples * sites + 9 * sites + 32 * n_samples
    return discovered + evidence + SELECTION_BYTES_PER_FOUNDER_SITE * founder_sites + completion_depth


def check_reused(store, file, configs, genetic_map, *, painting=False) -> None:
    """Before a contig's completed records are reused without running their stages: bind its local-selection rounds,
    its final assembly and (``painting``) its painting to this code and configuration, as the stages themselves do
    (CheckpointMismatch when any was produced by another code or configuration)."""
    from ..assembly import Role
    from ..assembly.stage import identity_of
    from ..evidence import records as evidence_records
    from ..local_selection import records as selection_records
    from ..local_selection.stage import bind_rounds
    from ..painting import records as painting_records
    from ..store.store import RecordScope
    bind_rounds(file, configs.local_selection, configs.assembly, genetic_map, configs.discovery)
    RecordScope(file, ASSEMBLY_SCOPE).bind(identity_of(
        file.info(evidence_records.EVIDENCE).identity, selection_records.panels_identity(file, FINAL_ROUND),
        genetic_map, configs.assembly, Role.FINAL, None))
    if painting:
        painting_records.bind_identity(store, file, configs.painting, genetic_map)


def final_assembly(store, contig, config, genetic_map, budget, *, view=None, queue=None, panels=None):
    """The contig's final assembly (role FINAL) from its l2 local panels (``panels`` when the caller holds them,
    else decoded from the records); resumes from its records."""
    from ..assembly import AssemblyEvidence, AssemblyInputs, Role, assemble
    from ..assembly import records as assembly_records
    from ..local_selection import records as selection_records
    from ..store.store import RecordScope
    file = store.contig(contig)
    if assembly_records.complete(file):
        return
    if view is None:
        view = AssemblyEvidence.from_record(file)
    if panels is None:
        panels = selection_records.load_panels(file, FINAL_ROUND)
    inputs = AssemblyInputs(panels, view, genetic_map,
                            panels_identity=selection_records.panels_identity(file, FINAL_ROUND))
    assemble(inputs, config, role=Role.FINAL, records=RecordScope(file, ASSEMBLY_SCOPE), budget=budget,
             queue=queue)


def reconstruct_contig(store, contig, context: ContigContext, budget) -> ContigSummary:
    """Local selection, final assembly and painting of one contig (each resumed from its records)."""
    from .. import painting
    from ..local_selection.stage import run_contig as select_local_panels
    from ..painting import records as painting_records
    from .outputs import publish
    file = store.contig(contig)
    if painting_records.complete(file):
        check_reused(store, file, context.configs, context.genetic_maps.for_contig(contig), painting=True)
        if context.output is not None:
            publish("write_painting", context.output, store, contig)
        return ContigSummary(contig, True, painting_records.progress(file))
    configs = context.configs
    genetic_map = context.genetic_maps.for_contig(contig)
    with execution.stage_timer("reconstruction", contig):
        from ..assembly import AssemblyEvidence
        from ..assembly import records as assembly_records
        from ..evidence import records as evidence_records
        if assembly_records.complete(file):
            check_reused(store, file, configs, genetic_map)
        else:
            # One founder-view evidence for the context assemblies and the final assembly (as base).
            decoded = evidence_records.load(file)
            view = AssemblyEvidence.from_record(file, decoded)
            # Local selection takes the decoded evidence instead of decoding the record again when the founder
            # view is the evidence itself (no QC-flagged sample); otherwise this copy would stay referenced beside
            # local selection's founder copy during the whole selection.
            evidence = None if decoded.founder_excluded.any() else decoded
            del decoded
            with execution.stage_timer("local_selection", contig):
                # The final round's panels in block order (as recorded): final assembly need not decode them again.
                panels = select_local_panels(store, contig, configs.local_selection, configs.assembly, genetic_map,
                                             budget, discovery_config=configs.discovery, queue=context.queue,
                                             assembly_evidence=view, evidence=evidence)
            del evidence
            with execution.stage_timer("assembly", contig):
                final_assembly(store, contig, configs.assembly, genetic_map, budget, view=view,
                               queue=context.queue, panels=panels)
            del panels
            del view
        execution.release_memory()
        with execution.stage_timer("painting", contig):
            painting.run_contig(store, contig, configs.painting, genetic_map, budget)
        execution.release_memory()
        if context.output is not None:
            publish("write_painting", context.output, store, contig)
    return ContigSummary(contig, False, painting_records.progress(file))
