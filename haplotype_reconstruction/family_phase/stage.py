"""Family phase stage: pedigree-conditioned, genotype-preserving final phase of every chromosome.

Per chromosome: scaffold (painting + founder refit) -> family belief propagation (solver.py) -> family MAP frames
-> root-frame repair (frames.py) -> polish of two-parent children on the repaired scaffold (polish.py). The
released calls are the scaffold alleles in the displayed track order: genotypes and missingness are the
scaffold's. Samples with fewer than two observed parents are displayed in their (repaired) scaffold orientation
and decoded by recombination in their family MAP frame.
"""
from __future__ import annotations

import time

import numpy as np

from ..evidence import records as evidence_records
from ..execution import ChromosomeExecutor, claim_threads, memory_worker_limit, release_memory, timed_stage
from ..founder_refit import records as refit_records
from ..painting import records as painting_records
from ..store import CheckpointStore
from . import frames, polish, records, solver
from .records import FamilyPhaseDiagnostics, FamilyPhaseProduct, WORK_SAVE_SECONDS
from .scaffold import build_scaffold, tie_boundaries


@timed_stage("family_phase")
def run(store, contigs, edges, config, genetic_maps, budget):
    """Final phase of every chromosome (one executor task each, with a shared dynamic CPU budget).

    Returns the per-contig FamilyPhaseSummary rows (also stored as run record ``family_phase``). Contigs whose
    product exists are not recomputed; an interrupted chromosome resumes its belief propagation from its work
    state.
    """
    contigs = tuple(map(str, contigs))
    store.require(painting_records.KEY, contigs)
    store.require(refit_records.KEY, contigs)
    weights, process_limit = _resources(store, contigs, len(edges.sample_ids), budget)
    common = dict(root=store.root, edges=edges, config=config, genetic_maps=genetic_maps)
    with ChromosomeExecutor(contigs, _phase_contig, budget=budget, weights=weights, max_workers=process_limit,
                            label="FAMILY PHASE", task_bytes=weights) as executor:
        summaries = tuple(executor.run("family_phase", common))
    store.run.put(records.KEY, summaries)
    store.mark_stage_complete(records.STAGE)
    return summaries


class _ResumeState:
    """The chromosome's work state bound to its stage identity: load() -> FamilyMessages | None, save(messages)."""

    def __init__(self, work, identity):
        self.work, self.identity = work, identity

    def load(self):
        saved = self.work.load()
        if saved is None:
            return None
        identity, messages = saved
        if identity != self.identity:
            raise ValueError(f"{self.work.contig}: family-phase work state belongs to other inputs or settings")
        return messages

    def save(self, messages):
        self.work.save((self.identity, messages))


def _phase_contig(contig, state, phase, common, budget):
    """Own one chromosome's evidence, scaffold and product."""
    store = CheckpointStore(common["root"])
    file = store.contig(contig)
    edges, config = common["edges"], common["config"]
    chromosome_map = common["genetic_maps"].for_contig(contig)
    identity = records.stage_identity(file, edges, config, chromosome_map)
    store.bind_identity(file, records.STAGE, identity)
    if file.has(records.KEY):
        return state, records.summary(records.load(file, nthreads=claim_threads()))
    painting_records.require_evidence_binding(file)
    evidence = evidence_records.load(file, nthreads=claim_threads())
    if tuple(evidence.sample_ids) != tuple(edges.sample_ids):
        raise ValueError(f"{contig}: pedigree and evidence sample axes differ")
    painting = painting_records.load(file, panels=False, emissions=False).components
    scaffold = build_scaffold(painting, refit_records.founder_panels(file), evidence.positions,
                              len(evidence.sample_ids))
    parents = np.bincount(edges.parents, minlength=len(evidence.sample_ids))
    ties = tie_boundaries(painting, evidence.positions,
                          np.flatnonzero(parents >= config.root_frame_minimum_children))
    del painting
    work = store.work(contig, records.KEY)
    product = phase_chromosome(contig, evidence.sample_ids, evidence.positions, scaffold, evidence.likelihoods,
                               evidence.observed, edges, config, chromosome_map, ties=ties,
                               work=_ResumeState(work, identity))
    del evidence, scaffold
    records.write(file, product)
    work.clear()
    summary = records.summary(product)
    del product
    release_memory()
    print(f"[family phase {contig}] final phase released: iterations={summary.iterations} "
          f"converged={summary.converged}", flush=True)
    return state, summary


def _painting_bytes(file, samples) -> int:
    """Decoded bytes of the painting as this stage loads it: without its packed emissions (emissions=False)."""
    packed = sum(8 * samples * shape.states * (shape.states + 1) // 2 * shape.bins
                 for shape in painting_records.load_summary(file).components if shape.packed_emissions)
    return max(0, file.info(painting_records.KEY).decoded_bytes - packed)


def _resources(store, contigs, samples, budget):
    """Per-chromosome peak estimates (also each task's memory reservation, execution.memory) and the
    concurrent-chromosome limit (a scheduling estimate, not an RSS bound).

    Float64 GLs, four-state family messages, selector caches and the polish peaked at 7.7 decoded-evidence
    sizes plus 3.2 GiB for Python/JIT and up to 3.6 GiB a worker retains from earlier chromosomes (base,
    N = 320); nine evidence-sized copies (evidence = N x L x (3 float64 GLs + mask)), two painting copies and
    5 GiB per process stayed above every measured peak by 12-21%. A finished chromosome only decodes its product.
    """
    estimates = []
    for contig in contigs:
        file = store.contig(contig)
        if file.has(records.KEY):
            estimate = 2 * file.info(records.KEY).decoded_bytes
        else:
            sites = len(evidence_records.load_arrays(file, ("positions",))["positions"])
            estimate = 9 * samples * sites * 25 + 2 * _painting_bytes(file, samples)
        estimates.append(estimate + (5 << 30))
    return estimates, memory_worker_limit(estimates, budget.threads(), label="FAMILY PHASE")


def phase_chromosome(contig, sample_ids, positions, scaffold, likelihoods, observed, edges, config,
                     chromosome_map, *, ties, work=None, log=print):
    """FamilyPhaseProduct of one chromosome.

    ``scaffold`` PhaseScaffold; ``likelihoods`` float64 [N, L, 3] normalised genotype likelihoods and
    ``observed`` bool [N, L] (positive read depth) of every sample; ``edges`` ParentEdges; ``ties``
    {sample: marker boundaries} (scaffold.tie_boundaries of the parents with root_frame_minimum_children
    children); ``work`` a resume handle with load() -> FamilyMessages | None and save(messages), bound to this
    chromosome's inputs by the caller (run uses the store's work state).
    """
    started = time.perf_counter()
    samples = len(sample_ids)
    positions = np.asarray(positions, dtype=np.int64)
    graph = solver.family_graph(edges.parents, edges.children, edges.slots, samples)
    workspace = solver.prepare_workspace(likelihoods, observed, scaffold.alleles, positions, scaffold.bins,
                                         scaffold.components, graph, config, chromosome_map)
    messages = None if work is None else work.load()
    if messages is None:
        messages = solver.initial_messages(samples, len(positions), len(graph.parents), len(graph.group_child))
    last_saved = time.perf_counter()

    def after_iteration(state):
        nonlocal last_saved
        if work is not None and time.perf_counter() - last_saved >= WORK_SAVE_SECONDS:
            work.save(state)
            last_saved = time.perf_counter()

    converged = solver.iterate(workspace, messages, config, after_iteration)
    log(f"  [family phase {contig}] belief propagation: iterations={messages.iteration} "
        f"residual={messages.residuals[-1] if messages.residuals else float('nan'):.6g} converged={converged}")
    inferred, family_displayed = solver.released_frames(workspace, messages, config)
    emissions = solver.selector_emissions(workspace, messages, config)
    repair = frames.repair_root_frames(workspace, emissions, inferred, config, tie_boundaries=ties,
                                       protected=config.protected_for(contig))
    del emissions
    log(f"  [family phase {contig}] root frame repair: moves={len(repair.moves)} reverts={len(repair.reverts)} "
        f"transfers={len(repair.transfers)} rejected={dict(repair.rejected)}")
    zero = np.zeros(inferred.shape, dtype=np.int8)
    display = zero if repair.display is None else repair.display
    internal = zero if repair.internal is None else repair.internal
    polished_scaffold = scaffold.alleles if repair.display is None else frames.swap_tracks(scaffold.alleles, display)
    result = polish.polish(polished_scaffold, scaffold.bins, scaffold.components, graph.parents, graph.children,
                           graph.slots, workspace.theta, config)
    if not result.converged:
        raise RuntimeError(f"{contig}: the phase polish did not converge in {config.polish_max_sweeps} sweeps")
    incomplete = graph.parent_counts < 2
    # Frames relative to the polished scaffold; samples without two observed parents are displayed in the
    # family's displayed frame (0) and decoded in their family MAP frame with the repair's relabelling.
    emitted = result.frames.copy()
    emitted[incomplete] = family_displayed[incomplete]
    decoder = result.frames.copy()
    decoder[incomplete] = (inferred ^ internal ^ display)[incomplete]
    calls = np.take_along_axis(polished_scaffold, np.stack((emitted, 1 ^ emitted), axis=-1), axis=-1)
    heterozygous = polished_scaffold[..., 0] != polished_scaffold[..., 1]
    diagnostics = FamilyPhaseDiagnostics(
        messages.iteration, bool(converged), tuple(map(float, messages.residuals)), result.sweep_updates,
        int(np.count_nonzero(calls >= 0)), int(np.count_nonzero((emitted != family_displayed) & heterozygous)),
        frames.FrameRepair(repair.moves, repair.reverts, repair.transfers, repair.child_frames, repair.rejected,
                           repair.guard_blocks, repair.flagged_edges),
        time.perf_counter() - started)
    return FamilyPhaseProduct(
        str(contig), tuple(map(str, sample_ids)), positions, scaffold.components, scaffold.bin_edges,
        scaffold.bin_edge_offsets, calls, emitted ^ display, decoder ^ display, scaffold.coverage,
        scaffold.coverage_offsets, str(edges.identity), diagnostics)
