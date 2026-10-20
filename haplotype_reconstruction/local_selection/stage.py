"""Local selection stage of one contig: initial path fit, then the context rounds.

Round "initial" refits every informative discovery block from its raw discovery panel. Round
"l1" assembles the current selection through level 1 (assembly role CONTEXT), refits each
assembled component's founders from every covered block's own evidence (the context proposals:
candidates, never observations) and reselects every block; round "l2" does the same through
level 2. The last round's panels are the final assembly's input.

Local selection sits above assembly: it calls ``assembly.assemble(..., role=CONTEXT)``;
assembly never imports this package. Completed rounds (and a context round's proposals) resume
from their records.
"""
from __future__ import annotations

import numpy as np

from ..assembly import AssemblyInputs, Role, assemble
from ..assembly.evidence import AssemblyEvidence
from ..discovery import records as discovery_records
from ..evidence import records as evidence_records
from ..execution import release_memory
from ..model.components import materialise
from ..store import RecordScope
from . import records
from .context_refit import refine
from .rounds import BlockTask, run_round

ROUNDS = records.ROUNDS


def _tied_panels(block) -> tuple:
    """Release 2.2: discovery's NLL-tied alternative latent panels of a block (DiscoveredBlock
    ``tied_panels`` tuple after decoding; stacked rows split by ``tied_panel_k`` in older fixtures).
    These enter every round on the same footing as the latent panel."""
    rows = getattr(block, "tied_panels", None)
    sizes = getattr(block, "tied_panel_k", None)
    if rows is None:
        return ()
    if isinstance(rows, (tuple, list)):
        return tuple(np.ascontiguousarray(panel, dtype=np.int64) for panel in rows if len(panel))
    if sizes is None or not len(sizes):
        return ()
    bounds = np.cumsum(np.asarray(sizes, dtype=np.int64))[:-1]
    return tuple(np.ascontiguousarray(panel, dtype=np.int64)
                 for panel in np.split(np.asarray(rows), bounds) if len(panel))


def _block_tasks(discovered, evidence, genetic_map):
    """Per block with an incumbent (in block order): task fields shared by every round."""
    tasks = []
    for block in discovered:
        # No incumbent exists when every admissible initial class was omitted. Like an
        # uninformative block, it supplies no local panel; the discovery record retains its
        # unresolved/omission diagnostics. Incomplete searches with an incumbent stay eligible.
        if block is None or block.latent.shape[0] == 0:
            continue
        index = len(tasks)
        kept = block.kept
        positions = block.positions[kept]
        tasks.append(dict(
            index=index, block=block.block, site_start=block.site_start,
            positions=block.positions, kept=kept,
            columns=block.site_start + np.flatnonzero(kept),
            intervals=genetic_map.interval_morgans(positions[:-1], positions[1:]),
            latent=np.ascontiguousarray(block.latent, dtype=np.int64),
            alternatives=_tied_panels(block)))
    return tasks


def run_contig(store, contig, config, assembly_config, genetic_map, budget, *,
               discovery_config, queue=None, assembly_evidence=None, evidence=None):
    """Select the local panels of one contig (resuming completed rounds from their records).

    store: the run's checkpoint store; contig: name; config: LocalSelectionConfig;
    assembly_config: passed to the context assemblies (its n_generations is the path model's
    copying-hazard multiplier); genetic_map: the contig's ChromosomeGeneticMap (a constant-rate
    map when no map file is configured); budget: the contig lane's CPU budget; discovery_config:
    the DiscoveryConfig of the discovery records (the candidate bank refits with the same fixed-K
    model and cavity score); queue: the cross-node batch queue directory or None;
    assembly_evidence: the contig's ``assembly.AssemblyEvidence`` view when the caller already
    built it (one view for the context and final assemblies), else built here on first use;
    evidence: the contig's ``ContigEvidence`` (as recorded, e.g. the one the caller decoded to
    build ``assembly_evidence``; its founder view is taken here), else decoded from the evidence
    record when a round must be computed.
    Returns tuple[LocalPanel, ...] of the last round, one per informative block, in block order.
    A contig whose rounds are all recorded decodes neither the evidence nor the fits.
    """
    file = store.contig(contig)
    evidence_identity = file.info(evidence_records.EVIDENCE).identity
    discovery_identity = file.info(f"identity/{discovery_records.STAGE}").identity
    generations = float(assembly_config.n_generations)
    settings = (config, discovery_config, generations)
    upstream = dict(evidence=evidence_identity, discovery=discovery_identity)
    scopes = []
    for round_index, round_name in enumerate(ROUNDS):
        scope = records.round_scope(file, round_name)
        scope.bind(records.round_identity(round_name, config, discovery_config, generations,
                                          genetic_map, upstream,
                                          assembly_config if round_index else None))
        upstream = dict(upstream, previous_round=file.info(
            f"identity/{records.round_prefix(round_name)}").identity)
        scopes.append(scope)
    first = next((index for index, name in enumerate(ROUNDS) if not records.round_complete(file, name)),
                 len(ROUNDS))
    if first == len(ROUNDS):
        print(f"[Local selection] {contig}: resumed all rounds", flush=True)
        return records.load_panels(file, ROUNDS[-1])
    evidence = (evidence_records.load(file) if evidence is None else evidence).founder_view()
    discovered = discovery_records.load(file)
    shared = _block_tasks(discovered, evidence, genetic_map)
    del discovered
    # Earlier rounds' calls are candidates of every later round; the last completed round's
    # selections (with their memos) are the incumbents of the first round to compute.
    selected_calls = [[panel.calls for panel in records.load_panels(file, name)]
                      for name in ROUNDS[:max(first - 1, 0)]]
    previous = panels = None
    if first:
        previous = records.load_selections(file, ROUNDS[first - 1])
        panels = [selection.panel for selection in previous]
        selected_calls.append([panel.calls for panel in panels])
        print(f"[Local selection] {contig}: resumed rounds {', '.join(ROUNDS[:first])}", flush=True)
    for round_index in range(first, len(ROUNDS)):
        round_name, scope = ROUNDS[round_index], scopes[round_index]
        proposals = None
        if round_index:
            level = config.context_levels[round_index - 1]
            saved = scope.load("proposals")
            if saved is None:
                if assembly_evidence is None:
                    assembly_evidence = AssemblyEvidence.from_record(file, evidence)
                print(f"[Local selection] {contig}: assembling the selected context through "
                      f"L{level}", flush=True)
                result = assemble(AssemblyInputs(tuple(panels), assembly_evidence, genetic_map),
                                  assembly_config, role=Role.CONTEXT, levels=level,
                                  records=RecordScope(file, records.context_assembly_prefix(round_name)),
                                  budget=budget, queue=queue)
                components = materialise(result.components, result.prepared)
                proposals, summary = refine(
                    panels, components, evidence, genetic_map, config, generations,
                    budget.phase(f"feedback_l{level}_refine"), census_key=f"{contig}/{round_name}/refit")
                scope.save("proposals", records.encode_proposals(proposals, summary))
                del result, components
                release_memory()
            else:
                proposals = records.decode_proposals(saved)
        tasks = []
        for fields in shared:
            index, kept = fields["index"], fields["kept"]
            if round_index:
                proposal = proposals[index]
                candidate_panels = tuple(
                    np.ascontiguousarray(calls[index][:, kept]) for calls in selected_calls
                ) + (np.ascontiguousarray(proposal.calls[:, kept]),)
                founders = (None if proposal.founders is None
                            else np.ascontiguousarray(proposal.founders[:, kept]))
            else:
                candidate_panels, founders = (), None
            prior = previous[index] if previous is not None else None
            tasks.append(BlockTask(
                **fields, proposals=candidate_panels, founders=founders,
                incumbent=None if prior is None else prior.outcome,
                incumbent_release=None if prior is None else prior.release,
                memo=None if prior is None else prior.memo,
                keep_memo=round_index < len(ROUNDS) - 1,
                label=f"{contig}/{round_name}/i{index}"))
        previous = run_round(scope, tasks, settings, evidence,
                             budget.phase(f"feedback_round{round_index}"), queue=queue,
                             label=f"{contig} {round_name}")
        panels = [selection.panel for selection in previous]
        selected_calls.append([panel.calls for panel in panels])
    return tuple(panels)
