"""Family phase stage: pedigree-conditioned, genotype-preserving final phase of every chromosome.

Per chromosome: scaffold (painting + founder refit) -> family MAP (map_solver.py: the posterior mode of the family
model by monotone exact block ascent; release 2.1, replacing belief propagation) -> family MAP frames -> root-frame
repair (frames.py) -> polish of two-parent children on the repaired scaffold (polish.py). The
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
from . import frames, map_solver, polish, records, solver
from .records import FamilyPhaseDiagnostics, FamilyPhaseProduct
from .scaffold import build_scaffold, tie_boundaries


@timed_stage("family_phase")
def run(store, contigs, edges, config, genetic_maps, budget):
    """Final phase of every chromosome (one executor task each, with a shared dynamic CPU budget).

    Returns the per-contig FamilyPhaseSummary rows (also stored as run record ``family_phase``). Contigs whose
    product exists are not recomputed; an interrupted chromosome is recomputed (the solver keeps no work state).
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
    # Painting tie boundaries of every sample displayed by its scaffold or decided by its family (fewer than two
    # observed parents): the root-frame repair's candidates and the released phase-set breaks (release 2.2).
    observed_parents = np.bincount(edges.children, minlength=len(evidence.sample_ids))
    ties = tie_boundaries(painting, evidence.positions, np.flatnonzero(observed_parents < 2))
    del painting
    started = time.perf_counter()
    sample_ids, positions = evidence.sample_ids, evidence.positions.copy()
    workspace = prepare_chromosome(sample_ids, positions, scaffold, evidence.likelihoods, evidence.observed, edges,
                                   config, chromosome_map)
    # The genotype likelihoods enter only the potentials: release them before solving (release 2.1;
    # N x L x 25 bytes less at the stage's memory peak).
    del evidence
    product = solve_chromosome(contig, sample_ids, scaffold, workspace, edges, config, ties=ties, started=started)
    del scaffold, workspace
    records.write(file, product)
    summary = records.summary(product)
    del product
    release_memory()
    print(f"[family phase {contig}] final phase released: family MAP sweeps={summary.map_sweeps} "
          f"J={summary.map_objective:.6f}", flush=True)
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
    N = 320; release 2.0); nine evidence-sized copies (evidence = N x L x (3 float64 GLs + mask)), two painting
    copies and 5 GiB per process stayed above every measured peak by 12-21%. Release 2.1's family MAP keeps no
    messages: float64 potentials (N x L x 4), int8 states, per-block work arrays and, at the release, float64
    Gibbs conditionals and repair emissions. Its measured peaks are below 2.1's belief propagation (8 threads,
    prepare + solve + release: std_n80 chr1 3.2 GiB vs 5.5, stress_n80 chr13 2.2 vs 5.5, TroMau chr7 2.3 vs 4.8),
    so ten evidence-sized copies stay conservative.
    A finished chromosome only decodes its product.
    """
    estimates = []
    for contig in contigs:
        file = store.contig(contig)
        if file.has(records.KEY):
            estimate = 2 * file.info(records.KEY).decoded_bytes
        else:
            sites = len(evidence_records.load_arrays(file, ("positions",))["positions"])
            estimate = 10 * samples * sites * 25 + 2 * _painting_bytes(file, samples)
        estimates.append(estimate + (5 << 30))
    return estimates, memory_worker_limit(estimates, budget.threads(), label="FAMILY PHASE")


def phase_chromosome(contig, sample_ids, positions, scaffold, likelihoods, observed, edges, config,
                     chromosome_map, *, ties, log=print):
    """FamilyPhaseProduct of one chromosome.

    ``scaffold`` PhaseScaffold; ``likelihoods`` float64 [N, L, 3] normalised genotype likelihoods and
    ``observed`` bool [N, L] (positive read depth) of every sample; ``edges`` ParentEdges; ``ties``
    {sample: marker boundaries} (scaffold.tie_boundaries of the samples with fewer than two observed parents: the
    root-frame repair's flagged edges and the released phase-set breaks; samples left out get no breaks).
    """
    started = time.perf_counter()
    workspace = prepare_chromosome(sample_ids, positions, scaffold, likelihoods, observed, edges, config,
                                   chromosome_map)
    return solve_chromosome(contig, sample_ids, scaffold, workspace, edges, config, ties=ties, log=log,
                            started=started)


def prepare_chromosome(sample_ids, positions, scaffold, likelihoods, observed, edges, config, chromosome_map):
    """The chromosome's solver workspace (phase_chromosome's first step; the genotype likelihoods enter only its
    potentials, so a caller may release them before solve_chromosome)."""
    positions = np.asarray(positions, dtype=np.int64)
    graph = solver.family_graph(edges.parents, edges.children, edges.slots, len(sample_ids))
    return solver.prepare_workspace(likelihoods, observed, scaffold.alleles, positions, scaffold.bins,
                                    scaffold.components, graph, config, chromosome_map)


def solve_chromosome(contig, sample_ids, scaffold, workspace, edges, config, *, ties, log=print, started=None):
    """FamilyPhaseProduct of a prepared chromosome (phase_chromosome's second step; arguments as there)."""
    started = time.perf_counter() if started is None else started
    positions = workspace.positions
    graph = workspace.graph
    family_map = map_solver.FamilyMap(workspace, config)
    solution, merge = family_map.solve_symmetric(config.map_max_sweeps)
    log(f"  [family phase {contig}] family MAP: sweeps={solution.sweeps} accepted={list(solution.accepted)} "
        f"J={solution.objective[0]:.6f} -> {solution.objective[-1]:.6f} two schedules={merge}")
    inferred, family_displayed, map_undetermined, evidence = map_solver.release_frames(family_map, solution.state)
    emissions = map_solver.selector_emissions(family_map, solution.state)
    conflicts = tuple((str(sample_ids[int(parent)]), int(positions[int(site)]))
                      for parent, site in map_solver.conflict_sites(family_map, solution.state))
    log(f"  [family phase {contig}] Mendelian-conflict sites in the MAP: {len(conflicts)}")
    del family_map
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
    one_parent = graph.parent_counts == 1
    # Frames relative to the polished scaffold; roots are displayed in their (repaired) scaffold orientation (0) and
    # decoded in their family MAP frame with the repair's relabelling. Release 2.2: a one-parent child is displayed in
    # parental origin (the observed parent's transmitted allele in its slot track): inside each stretch between its
    # painting tie boundaries the painting's phase is kept, and the stretch's orientation is the one its family-MAP
    # phase view supports (one_parent_orientation); the decoder frame stays the family MAP's.
    emitted = result.frames.copy()
    emitted[incomplete] = family_displayed[incomplete]
    decoder = result.frames.copy()
    decoder[incomplete] = (inferred ^ internal ^ display)[incomplete]
    chunk_evidence = one_parent_orientation(graph, workspace.components, evidence, internal, ties, emitted)
    calls = np.take_along_axis(polished_scaffold, np.stack((emitted, 1 ^ emitted), axis=-1), axis=-1)
    heterozygous = polished_scaffold[..., 0] != polished_scaffold[..., 1]
    undetermined, breaks, break_offsets = released_ambiguity(
        graph, workspace.components, map_undetermined, chunk_evidence, repair, result.undetermined, ties)
    diagnostics = FamilyPhaseDiagnostics(
        solution.sweeps, True, tuple(map(int, solution.accepted)), tuple(map(float, solution.objective)),
        result.sweep_updates,
        int(np.count_nonzero(calls >= 0)),
        int(np.count_nonzero(((emitted != family_displayed) & heterozygous)[~incomplete])),
        frames.FrameRepair(repair.moves, repair.reverts, repair.transfers, repair.child_frames, repair.rejected,
                           repair.guard_blocks, repair.flagged_edges),
        time.perf_counter() - started, conflicts, merge)
    return FamilyPhaseProduct(
        str(contig), tuple(map(str, sample_ids)), positions, scaffold.components, scaffold.bin_edges,
        scaffold.bin_edge_offsets, calls, emitted ^ display, decoder ^ display, scaffold.coverage,
        scaffold.coverage_offsets, str(edges.identity), diagnostics, undetermined, breaks, break_offsets)


def one_parent_orientation(graph, components, evidence, internal, ties, emitted):
    """Set each one-parent child's displayed frame (``emitted``, in place) to the orientation of each stretch
    between its painting tie boundaries (inside a component) that the family-MAP phase view supports: 1 where the
    stretch's summed frame log-likelihood ratio (``evidence``, with the root-frame repair's following relabelling,
    ``internal``) is positive, else 0 (the painting scaffold); the painting's phase is kept inside the stretch.
    Returns {sample: [(start, stop, summed evidence)]} of the stretches (stage.released_ambiguity reads it)."""
    sites = evidence.shape[1]
    comps = np.asarray(components)
    stretches = {}
    for sample in np.flatnonzero(graph.parent_counts == 1):
        sample = int(sample)
        signed = np.where(internal[sample] == 1, -evidence[sample], evidence[sample]).astype(np.float64)
        cuts = np.flatnonzero(comps[1:] != comps[:-1]) + 1
        own = np.asarray(ties.get(sample, ()), dtype=np.int64)
        own = own[(own >= 0) & (own < sites - 1)]
        if len(own):
            own = own[(comps[own] >= 0) & (comps[own] == comps[own + 1])]
        bounds = np.unique(np.r_[0, cuts, own + 1, sites])
        rows = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            total = float(np.sum(signed[start:stop]))
            flip = 1 if total > chains_tie() else 0
            emitted[sample, start:stop] = flip
            rows.append((int(start), int(stop), total))
        stretches[sample] = rows
    return stretches


def chains_tie():
    from ..maths import chains
    return chains.CHAIN_TIE


def released_ambiguity(graph, components, map_undetermined, chunk_evidence, repair, polish_undetermined, ties):
    """(frame_undetermined int8 [N, L], phase_breaks int64 [B], phase_break_offsets int64 [N + 1]) (release 2.2).

    frame_undetermined bit 0: the decoder frame is undetermined (the two extreme tie placements of the sample's
    chain differ there: roots' frame chains and one-parent children's phase views at the family MAP, two-parent
    children's polished chains; plus the stretch of a root-frame repair decision at a flagged edge); bit 1: the same
    for the displayed frame (roots: the repair stretches only, their display being the repaired scaffold; one-parent
    children: their decoder frame; two-parent children: their polished frame). phase_breaks: per sample, the marker
    boundaries where the displayed phase of the right side relative to the left is not determined by data: a
    painting tie boundary (inside a component) of a root that the root-frame repair does not decide (every one of a
    root with fewer than root_frame_minimum_children children), and of a one-parent child when the stretch on either
    side (between its tie boundaries, one_parent_orientation) carries no frame evidence."""
    samples, sites = map_undetermined.shape
    comps = np.asarray(components)
    ambiguous = (repair.ambiguous if repair.ambiguous is not None
                 else np.zeros((samples, sites), dtype=np.bool_))
    polish_undetermined = (np.zeros((samples, sites), dtype=np.bool_) if polish_undetermined is None
                           else polish_undetermined)
    incomplete = (graph.parent_counts < 2)[:, None]
    decoder_undetermined = np.where(incomplete, map_undetermined | ambiguous, polish_undetermined)
    # Displayed: roots - the repair stretches; one-parent children - nothing (their stretches keep the painting's
    # phase; a stretch without evidence is set off by phase breaks); two-parent children - the polished chains.
    display_undetermined = np.where(graph.roots[:, None], ambiguous,
                                    np.where((graph.parent_counts == 1)[:, None], False, polish_undetermined))
    undetermined = decoder_undetermined.astype(np.int8) | (display_undetermined.astype(np.int8) << 1)
    undecided = repair.undecided_ties or {}
    breaks = []
    for sample in range(samples):
        found = np.zeros(0, dtype=np.int64)
        own = np.asarray(ties.get(sample, ()), dtype=np.int64)
        own = own[(own >= 0) & (own < sites - 1)]
        if len(own):
            own = own[(comps[own] >= 0) & (comps[own] == comps[own + 1])]
        if graph.roots[sample]:
            found = np.asarray(undecided[sample], dtype=np.int64) if sample in undecided else own
        elif graph.parent_counts[sample] == 1 and len(own):
            # A tie boundary between two stretches is undecided when either stretch carries no frame evidence.
            rows = chunk_evidence.get(sample, ())
            starts = np.asarray([row[0] for row in rows], dtype=np.int64)
            weak = np.asarray([abs(row[2]) <= chains_tie() for row in rows], dtype=np.bool_)
            keep = []
            for i in own:
                k = int(np.searchsorted(starts, i + 1))       # stretch starting right after boundary i
                if 0 < k < len(rows) and (weak[k - 1] or weak[k]):
                    keep.append(int(i))
            found = np.asarray(keep, dtype=np.int64)
        breaks.append(np.unique(found))
    offsets = np.r_[0, np.cumsum([len(b) for b in breaks])].astype(np.int64)
    flat = np.concatenate(breaks).astype(np.int64) if breaks else np.zeros(0, dtype=np.int64)
    return undetermined, flat, offsets


def _component_start(components, site):
    """First site of the run of ``site``'s component containing it."""
    start = site
    while start > 0 and components[start - 1] == components[site]:
        start -= 1
    return start


def _component_stop(components, site):
    """One past the last site of the run of ``site``'s component containing it."""
    stop = site + 1
    while stop < len(components) and components[stop] == components[site]:
        stop += 1
    return stop
