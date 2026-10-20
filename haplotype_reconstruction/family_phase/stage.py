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
from ..execution.budget import current_budget
from ..founder_refit import records as refit_records
from ..painting import records as painting_records
from ..store import CheckpointStore
from ..maths import ties as score_ties
from . import axes, frames, map_solver, polish, records, release, solver
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
    estimates, process_limit = _resources(store, contigs, len(edges.sample_ids), budget)
    weights, process_limit = scheduling_resources(store, contigs, budget.threads(), process_limit)
    common = dict(root=store.root, edges=edges, config=config, genetic_maps=genetic_maps)
    with ChromosomeExecutor(contigs, _phase_contig, budget=budget, weights=weights, max_workers=process_limit,
                            label="FAMILY PHASE", task_bytes=estimates, envelope=True, frontload=True) as executor:
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
    # Every painting tie requires either a family evidence witness or a display
    # phase-set break, including samples with two observed parents.
    ties = tie_boundaries(painting, evidence.positions, np.arange(len(evidence.sample_ids)))
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


def scheduling_resources(store, contigs, cores, memory_workers, *, fresh=False):
    """Name-free CPU work estimates, separate from byte reservations.

    A marker scan plus the minimum two-sided compound window per released
    painting switch estimates work without fitting sample or chromosome names.
    Give each initial chromosome room for its four endpoints and their numerical
    teams (three threads each), then grow its live lease as peers finish. The
    full CPU budget remains available; this limits processes, not total threads.
    """
    weights = []
    for contig in contigs:
        file = store.contig(contig)
        if not fresh and file.has(records.KEY):
            weights.append(1.0)  # cached products decode after unfinished fits
        else:
            shape = painting_records.load_summary(file)
            sites = len(evidence_records.load_arrays(file, ("positions",))["positions"])
            weights.append(float(max(1, sites + 2 * map_solver.COMPOUND_MARGIN * shape.switches_after)))
    workers = min(memory_workers, max(1, int(cores) // 12))
    return weights, workers


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
    {sample: marker boundaries} from scaffold.tie_boundaries for all samples. These supply root-frame
    repair's flagged edges and family-evidence tests for released phase-set breaks.
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


def _endpoint_bytes(workspace):
    samples, sites = workspace.potentials.shape[:2]
    groups = len(workspace.graph.group_child)
    held = np.count_nonzero(~workspace.graph.roots[np.unique(workspace.graph.parents)])
    # Private G/S/F, retained seed/results, initialization soft conditionals, and
    # per-thread block scratch. Inputs and coordinate views are shared.
    return int(sites * (23 * samples + 3 * groups + 16 * held + 128 * min(samples, claim_threads())))


def _ordered_tasks(functions, task_bytes, label):
    values = [None] * len(functions)
    for index, value in current_budget().concurrent(functions, task_bytes, label=label, envelope=True):
        values[index] = value
    return values


def _solve_four_endpoints(sample_ids, scaffold, workspace, edges, config, ties):
    """Both pedigree/data-axis directions times both schedules, closed under reversal.

    Every initial endpoint owns its continuation coordinates and schedule. Its
    transplant donor is the other schedule on that same axis. The global switch
    changes execution order only; the evaluated endpoint set stays unchanged.
    """
    started = time.perf_counter()
    ascending, coordinate_order = axes.pedigree_data_order(workspace)
    if coordinate_order['unresolved_duplicate_groups']:
        raise RuntimeError('family phase cannot distinguish coordinate keys from the data and pedigree; '
                           'joint treatment of unresolved duplicate groups is required')
    views = [axes.make_view(sample_ids, scaffold, workspace, edges, ties, order)
             for order in (ascending, ascending[::-1])]
    endpoints = {}
    provenance = [(axis, reverse) for axis in (0, 1) for reverse in (False, True)]
    if score_ties.reverse_conventions():
        provenance.reverse()
    def initial_task(key):
        def run(budget):
            axis, reverse = key
            view = views[axis]
            owner = map_solver.FamilyMap(view[2], config, reverse=reverse)
            initial = owner.solve(config.map_max_sweeps, long_tier=False)
            return owner, initial, view
        return run
    task_bytes = _endpoint_bytes(workspace)
    initial_values = _ordered_tasks([initial_task(key) for key in provenance], task_bytes,
                                   'family initial endpoints')
    endpoints.update(zip(provenance, initial_values))
    initial_seconds = time.perf_counter() - started
    # Rank in a provenance order independent of the caller's axis and switch.
    keys = sorted(endpoints)
    quantum = endpoints[keys[0]][0].quantum
    scores = [endpoints[key][1].objective[-1] for key in keys]
    ranking = score_ties.tie_classes(scores, quantum, descending=True)
    winners = np.flatnonzero(ranking.classes == 0)
    continuation_bound = 4
    # Continue every initial endpoint before comparing final objectives. A lower
    # initial J can have a better merged continuation; initial ranking is diagnostic.
    per_endpoint = continuation_bound // len(keys)
    branches, rows = [], []
    continuation_started = time.perf_counter()
    def continuation_task(key):
        def run(budget):
            axis, reverse = key
            owner, initial, _ = endpoints[key]
            donor = endpoints[axis, not reverse][1]
            return owner.continue_endpoint(config.map_max_sweeps, initial, donor, per_endpoint)
        return run
    continued = _ordered_tasks([continuation_task(key) for key in keys], task_bytes,
                               'family continued endpoints')
    for (axis, reverse), (results, own_rows) in zip(keys, continued):
        owner, initial, view = endpoints[axis, reverse]
        for result in results:
            branches.append(dict(axis=axis, reverse=reverse, owner=owner, result=result, view=view))
        rows.extend(dict(axis=axis, **row) for row in own_rows)
    final_rank = score_ties.tie_classes([branch['result'].objective[-1] for branch in branches],
                                        quantum, descending=True)
    retained = [branches[i] for i in np.flatnonzero(final_rank.classes == 0)]

    def representative_key(branch):
        state = axes.align_state(branch['result'].state, branch['view'][0], views[0][0],
                                 branch['owner'].graph, views[0][2].graph)
        return (state.frame.tobytes(), state.genotype.tobytes(), state.selector.tobytes(),
                branch['axis'], branch['reverse'])

    # This labels the stored representative only. Every tied release is projected.
    retained.sort(key=representative_key)
    import hashlib
    state_hash_cache = {}
    def state_hashes(state):
        key = id(state)
        if key not in state_hash_cache:
            hashes = {}
            for field in ('genotype', 'selector', 'frame'):
                values = np.ascontiguousarray(getattr(state, field))
                # A valid no-parent state has a selector axis of length zero.
                # Python memoryview cannot cast such multidimensional views.
                raw = memoryview(values).cast('B') if values.size else b''
                hashes[field] = hashlib.sha256(raw).hexdigest()
            state_hash_cache[key] = hashes
        return state_hash_cache[key]
    record = dict(initial_state_hashes=[dict(axis=axis, reverse=reverse,
                      hashes=state_hashes(endpoints[axis, reverse][1].state)) for axis, reverse in keys],
                  continued_state_hashes=[dict(axis=b['axis'], reverse=b['reverse'],
                      J=float(b['result'].objective[-1]), hashes=state_hashes(b['result'].state)) for b in branches],
                  initial_endpoints=[dict(axis=axis, reverse=reverse, J=float(scores[i]))
                                    for i, (axis, reverse) in enumerate(keys)],
                  parallel_layout=[dict(axis=axis, reverse=reverse,
                      colour_sizes=[len(x) for x in endpoints[axis, reverse][0].colours],
                      initial_batch_sizes=[len(x) for x in endpoints[axis, reverse][0].initial_batches],
                      compound_schedule=getattr(endpoints[axis, reverse][0], 'compound_schedule', 'serial'),
                      compound_layout=getattr(endpoints[axis, reverse][0], 'compound_layout', []))
                      for axis, reverse in keys],
                  initial_seconds=initial_seconds, coordinate_order=coordinate_order,
                  continued_endpoints=len(keys), tied_base=bool(ranking.tied_winner),
                  initial_winners=len(winners), branches=len(branches),
                  release_alternatives=len(retained), continuation_bound=continuation_bound,
                  continuation_seconds=time.perf_counter() - continuation_started,
                  bounded_out=any(row['bounded_out'] for row in rows),
                  clusters=sum(row['clusters'] for row in rows),
                  transplanted=sum(row['transplanted'] for row in rows),
                  tied_transplants=sum(row['tied_transplants'] for row in rows),
                  continuations=rows, J_release=float(retained[0]['result'].objective[-1]))
    return retained, record


def _align_release(branch, old_names, new_names, old_graph, new_graph):
    """Map one native release to the caller's axis without changing its decisions."""
    order = axes.sample_order(old_names, new_names)
    inverse = np.argsort(order)
    return (branch[0][order], branch[1][order], branch[2][order],
            [set(branch[3][i]) for i in order], axes.align_release_record(branch[4], old_names, new_names,
                                                                          old_graph=old_graph, new_graph=new_graph),
            axes.align_frame_repair(branch[5], old_names, new_names, old_graph=old_graph, new_graph=new_graph),
            branch[6], branch[7][order], {int(inverse[i]): row for i, row in branch[8].items()})


def _check_duplicate_releases(groups, emitted, decoder, undetermined, breaks):
    """A storage order among certified exchangeable inputs cannot choose phase."""
    for members in groups:
        first = members[0]
        if any(not np.array_equal(array[first], array[sample])
               for array in (emitted, decoder, undetermined) for sample in members[1:]):
            raise RuntimeError('family phase produced distinct releases for exchangeable inputs; '
                               'joint symmetry projection is required')
        if any(breaks[sample] != breaks[first] for sample in members[1:]):
            raise RuntimeError('family phase produced distinct phase sets for exchangeable inputs; '
                               'joint symmetry projection is required')


def _orientation_endpoints(contexts, branches, sample_ids, graph):
    """Caller-aligned, read-only proposals for the marginal orientation estimator.

    Build before union projection mutates the primary branch's mask or breaks.
    The tuple is an unordered candidate set; axis/schedule and J are provenance,
    not an instruction to take the first candidate or to rank by old J.
    """
    def readonly(array, *, copy=False):
        result = array.copy() if copy else array.view()
        result.flags.writeable = False
        return result

    endpoints = []
    for context, branch in zip(contexts, branches):
        state = axes.align_state(context['result'].state, context['view'][0], sample_ids,
                                 context['owner'].graph, graph)
        for array in (state.genotype, state.selector, state.frame):
            array.flags.writeable = False
        endpoints.append(dict(
            axis=context['axis'], reverse=context['reverse'], state=state,
            J=float(context['result'].objective[-1]),
            displayed_frame=readonly(branch[0]), decoder_frame=readonly(branch[1]),
            frame_undetermined=readonly(branch[2], copy=True),
            phase_breaks=tuple(readonly(np.asarray(sorted(boundaries), dtype=np.int64))
                               for boundaries in branch[3]),
            family_displayed=readonly(branch[7])))
    return tuple(endpoints)


def solve_chromosome(contig, sample_ids, scaffold, workspace, edges, config, *, ties, log=print, started=None):
    """FamilyPhaseProduct of a prepared chromosome (phase_chromosome's second step; arguments as there)."""
    started = time.perf_counter() if started is None else started
    positions = workspace.positions
    graph = workspace.graph
    contexts, merge = _solve_four_endpoints(sample_ids, scaffold, workspace, edges, config, ties)
    primary = contexts[0]
    solution = primary['result']
    log(f"  [family phase {contig}] family MAP: sweeps={solution.sweeps} accepted={list(solution.accepted)} "
        f"J={solution.objective[0]:.6f} -> {solution.objective[-1]:.6f} four endpoints={merge}")
    conflicts = tuple((str(primary['view'][0][int(parent)]), int(positions[int(site)]))
                      for parent, site in map_solver.conflict_sites(primary['owner'], solution.state))
    def release_task(context):
        def run(budget):
            names, own_scaffold, _, _, own_ties = context['view']
            return _release_branch(context['owner'], context['result'].state, own_scaffold, config, own_ties, contig)
        return run
    # Branches own their mutable release arrays. Complete each one in its own
    # coordinates, then project in the original retained-set order.
    release_bytes = len(positions) * (64 * len(sample_ids) + 32 * len(graph.group_child))
    native_branches = _ordered_tasks([release_task(c) for c in contexts], release_bytes, 'family release branches')
    branches = []
    for context, branch in zip(contexts, native_branches):
        names = context['view'][0]
        branches.append(_align_release(branch, names, sample_ids, context['owner'].graph, graph))
        _check_duplicate_releases(merge['coordinate_order']['certified_duplicate_groups'], *branches[-1][:4])
    emitted, decoder, undetermined, breaks, branch_record, repair, polish_updates, family_displayed, _ = branches[0]
    projection = []
    for branch_index, other in enumerate(branches[1:], 1):
        other_display, other_decoder, other_mask, other_breaks = other[:4]
        # Project the two evaluated releases onto observable ordered calls.
        # Equal homozygotes and fully unknown calls do not differ when swapped.
        ordered_call_can_differ = scaffold.alleles[..., 0] != scaffold.alleles[..., 1]
        raw_display_diff = emitted != other_display
        display_diff = raw_display_diff & ordered_call_can_differ
        decoder_diff = (decoder != other_decoder) & ordered_call_can_differ
        # Secondary chain/repair masks also describe frame alternatives across
        # entire stretches. Project those alternatives onto observable calls too;
        # the primary branch's existing intrinsic ambiguity policy is unchanged.
        other_mask = other_mask & (ordered_call_can_differ.astype(np.int8) * 3)
        added_display = ((other_mask & 2) != 0) & ((undetermined & 2) == 0)
        added_decoder = ((other_mask & 1) != 0) & ((undetermined & 1) == 0)
        undetermined |= other_mask | decoder_diff.astype(np.int8) | (display_diff.astype(np.int8) << 1)
        projection.append(dict(branch=branch_index, samples=[
            dict(sample=str(sample_ids[i]), displayed_cells=int(np.count_nonzero(display_diff[i])),
                 decoder_cells=int(np.count_nonzero(decoder_diff[i])),
                 intrinsic_display_cells=int(np.count_nonzero(added_display[i])),
                 intrinsic_decoder_cells=int(np.count_nonzero(added_decoder[i])))
            for i in range(len(sample_ids)) if (np.any(display_diff[i]) or np.any(decoder_diff[i])
                                                or np.any(added_display[i]) or np.any(added_decoder[i]))]))
        for sample in range(len(sample_ids)):
            breaks[sample].update(other_breaks[sample])
            for lo, hi in map_solver._runs_of(raw_display_diff[sample]):
                if not np.any(ordered_call_can_differ[sample, lo:hi]):
                    continue
                for cut in (lo, hi):
                    if (0 < cut < len(positions) and scaffold.components[cut - 1] >= 0
                            and scaffold.components[cut - 1] == scaffold.components[cut]):
                        breaks[sample].add(cut - 1)
    # Export keeps PS continuity across unphased cells. Test the final union
    # mask, including differences introduced by the retained-branch projection.
    gaps = release.uncertain_gaps(scaffold.components, (undetermined & 2) != 0, breaks)
    gap_records = []
    gap_started = time.perf_counter()
    def gap_task(context, native):
        def run(budget):
            if not gaps:
                return []
            names, own_scaffold = context['view'][:2]
            to_native = axes.sample_order(sample_ids, names)
            from_caller = np.argsort(to_native)
            native_gaps = [(int(from_caller[gap[0]]), *gap[1:]) for gap in gaps]
            witnesses = release.gap_witnesses(
                context['owner'], context['result'].state, own_scaffold.alleles, native[0],
                ((undetermined & 2) != 0)[to_native], native_gaps, native[5], native[8])
            return axes.align_release_record(dict(uncertain_joins=witnesses), names, sample_ids,
                old_graph=context['owner'].graph, new_graph=graph)['uncertain_joins']
        return run
    if gaps:
        # Each context owns its profile metadata; the nested witness jobs reserve
        # their window scratch independently through the same execution governor.
        values = _ordered_tasks([gap_task(context, native) for context, native in zip(contexts, native_branches)],
                                8 * scaffold.alleles.shape[0] * len(positions), 'family gap contexts')
        for branch, aligned in zip(branches, values):
            branch[4]['uncertain_joins'] = aligned
            gap_records.append(aligned)
    else:
        for branch in branches:
            branch[4]['uncertain_joins'] = []
    gap_breaks = 0
    for index, gap in enumerate(gaps):
        rows = [record[index] for record in gap_records]
        if not (all(row['joined'] for row in rows)
                and len({row['relative_frame'] for row in rows}) == 1):
            breaks[gap[0]].add(gap[3] - 1)
            gap_breaks += 1
    merge['uncertain_joins'] = dict(tested=len(gaps), phase_breaks_added=gap_breaks,
                                    family_profiles=4 * sum(row['kind'] == 'family_profile'
                                                            for rows in gap_records for row in rows),
                                    seconds=time.perf_counter() - gap_started)
    _check_duplicate_releases(merge['coordinate_order']['certified_duplicate_groups'],
                              emitted, decoder, undetermined, breaks)
    calls = np.take_along_axis(scaffold.alleles, np.stack((emitted, 1 ^ emitted), axis=-1), axis=-1)
    flat_breaks = [np.asarray(sorted(x), dtype=np.int64) for x in breaks]
    offsets = np.r_[0, np.cumsum([len(x) for x in flat_breaks])].astype(np.int64)
    flat = np.concatenate(flat_breaks) if flat_breaks else np.empty(0, dtype=np.int64)
    incomplete = graph.parent_counts < 2
    heterozygous = scaffold.alleles[..., 0] != scaffold.alleles[..., 1]
    merge['release'] = [branch[4] for branch in branches]
    merge['release_projection'] = projection
    log(f"  [family phase {contig}] phase sets: "
        f"{sum(len(x[4]['disagreements']) for x in branches)} MAP/polish disagreements, "
        f"{sum(len(x[4]['painting_joins']) for x in branches)} painting witnesses")
    diagnostics = FamilyPhaseDiagnostics(
        solution.sweeps, True, tuple(map(int, solution.accepted)), tuple(map(float, solution.objective)),
        polish_updates, int(np.count_nonzero(calls >= 0)),
        int(np.count_nonzero(((emitted != family_displayed) & heterozygous)[~incomplete])),
        frames.FrameRepair(repair.moves, repair.reverts, repair.transfers, repair.child_frames, repair.rejected,
                           repair.guard_blocks, repair.flagged_edges),
        time.perf_counter() - started, conflicts, merge)
    return FamilyPhaseProduct(
        str(contig), tuple(map(str, sample_ids)), positions, scaffold.components, scaffold.bin_edges,
        scaffold.bin_edge_offsets, calls, emitted, decoder, scaffold.coverage,
        scaffold.coverage_offsets, str(edges.identity), diagnostics, undetermined, flat, offsets)


def _release_branch(family_map, state, scaffold, config, ties, contig):
    """One retained MAP alternative, with its own coordinates and schedule through release.

    A unique J winner supplies the schedule, so reversing the two search calls
    cannot change this branch. Tied alternatives are all released and compared by
    the caller; their disagreement is ambiguity, not a convention winner.
    """
    workspace, graph = family_map.workspace, family_map.graph
    inferred, family_displayed, map_undetermined, evidence = map_solver.release_frames(family_map, state)
    emissions = map_solver.selector_emissions(family_map, state)
    repair = frames.repair_root_frames(workspace, emissions, inferred, config, tie_boundaries=ties,
                                       protected=config.protected_for(contig), reverse=family_map.reverse)
    del emissions
    zero = np.zeros(inferred.shape, dtype=np.int8)
    display = zero if repair.display is None else repair.display
    internal = zero if repair.internal is None else repair.internal
    polished_scaffold = scaffold.alleles if repair.display is None else frames.swap_tracks(scaffold.alleles, display)
    result = polish.polish(polished_scaffold, scaffold.bins, scaffold.components, graph.parents, graph.children,
                           graph.slots, workspace.theta, config, reverse=family_map.reverse)
    if not result.converged:
        raise RuntimeError(f"{contig}: phase polish did not converge in {config.polish_max_sweeps} sweeps")
    incomplete = graph.parent_counts < 2
    emitted = result.frames.copy()
    emitted[incomplete] = family_displayed[incomplete]
    decoder = result.frames.copy()
    decoder[incomplete] = (inferred ^ internal ^ display)[incomplete]
    chunk_evidence = one_parent_orientation(graph, workspace.components, evidence, internal, ties, emitted)
    undetermined, flat, offsets = released_ambiguity(
        graph, workspace.components, map_undetermined, chunk_evidence, repair, result.undetermined, ties)
    emitted ^= display
    decoder ^= display
    extra_breaks, record = release.phase_sets(
        family_map, state, scaffold.alleles, emitted, family_displayed, workspace.components, ties,
        reverse=family_map.reverse, undetermined=(undetermined & 2) != 0)
    breaks = [set(map(int, flat[offsets[i]:offsets[i + 1]])) | extra_breaks[i]
              for i in range(len(incomplete))]
    skipped = [row for sweep in family_map.compound_layout for row in sweep.get('skipped_neighbourhoods', ())]
    for row in skipped:
        for sample in row['samples']:
            for cut in row['cuts']:
                if (0 < cut < len(workspace.positions) and workspace.components[cut - 1] >= 0
                        and workspace.components[cut - 1] == workspace.components[cut]):
                    breaks[sample].add(cut - 1)
    record['skipped_compound_proposals'] = skipped
    # Keep only one-parent evidence needed by the post-projection continuity test.
    orientation_evidence = {
        int(sample): np.where((internal[sample] ^ display[sample]) == 1, -evidence[sample], evidence[sample])
        for sample in np.flatnonzero(graph.parent_counts == 1)}
    return emitted, decoder, undetermined, breaks, record, repair, result.sweep_updates, family_displayed, orientation_evidence


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
