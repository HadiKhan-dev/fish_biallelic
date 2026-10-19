"""Recombination stage: missing-aware maps from the final phase of every chromosome.

Per chromosome: decoder-frame calls (family phase; sites of an undetermined decoder frame that are not called
homozygotes are decoded as unknown, release 2.2) -> optional shared-family orientation fit -> per-meiosis
decoding (expected crossovers, called events, exposure) -> map aggregation -> bin-honest intervals and the
junction table. Supplied order breakpoints split the axis first (junctions.split_at_breakpoints). Results are
conditional on the final phase and the fixed inferred pedigree, not calibrated joint-pedigree uncertainty.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from ..evidence import records as evidence_records
from ..execution import ChromosomeExecutor, claim_threads, memory_worker_limit, release_memory, timed_stage
from ..family_phase import records as phase_records
from ..family_phase.breakpoints import contig_breakpoints
from ..store import CheckpointStore
from . import junctions as junction_report
from . import records
from .coverage import observed_coverage, sample_runs
from .decoder import decode_meioses
from .map import aggregate_map, genetic_coordinates, map_bins
from .records import AppliedBreakpoints, RecombinationProduct, RecombinationSummary, packed
from .shared_orientation import fit_shared_orientations


def map_chromosome(phase, edges, observed, config, chromosome_map):
    """RecombinationProduct of one chromosome.

    ``phase`` FamilyPhaseProduct; ``edges`` ParentEdges (the edges the phase was conditioned on); ``observed`` bool
    [N, L] positive read depth of the evidence behind the phase (None: read-free tracts are not removed).
    """
    started = time.monotonic()
    if phase.edges_identity != edges.identity:
        raise ValueError(f"{phase.contig}: family phase was conditioned on a different pedigree")
    names = phase.sample_ids
    positions = np.asarray(phase.positions, dtype=np.int64)
    components = np.asarray(phase.component_ids, dtype=np.int32)
    bin_edges = tuple(phase.component_bin_edges(c) for c in range(len(phase.bin_edge_offsets) - 1))
    breakpoints = contig_breakpoints(config.order_breakpoints, phase.contig)
    if breakpoints is None:
        segments, origin, masked = components, None, 0
    else:
        segments, origin, masked = junction_report.split_at_breakpoints(positions, components, breakpoints, bin_edges)
    parents, children, slots = (np.asarray(x, dtype=np.int64) for x in (edges.parents, edges.children, edges.slots))
    calls = phase.decoder_calls()
    if phase.frame_undetermined is not None:
        # Released ambiguity (release 2.2): where a sample's decoder frame is undetermined, which track holds which
        # allele is a tie convention, so a site there that is not a called homozygote (a half call; such stretches hold
        # no heterozygous call) carries no observation of the transmitted homologue: it is decoded as unknown.
        undetermined = (np.asarray(phase.frame_undetermined) & 1) > 0
        homozygous = (calls[..., 0] >= 0) & (calls[..., 0] == calls[..., 1])
        calls[undetermined & ~homozygous] = -1
    coverage, offsets, read_free_tracts, read_free_sites = observed_coverage(
        phase.coverage, phase.coverage_offsets, observed, config.read_free_tract_sites)
    fit = None
    if config.shared_family_evidence:
        runs = sample_runs(len(names), len(positions), coverage, offsets, segments, positions, config.maximum_gap_bp)
        genetic = np.asarray(chromosome_map.cumulative_morgans(positions), dtype=np.float64)
        orientation, fit = fit_shared_orientations(
            (calls, positions, runs, parents, children, slots, genetic), config)
        calls = np.take_along_axis(calls, np.stack((orientation, 1 ^ orientation), axis=-1), axis=-1)
        del runs, orientation
    bins = map_bins(positions, segments, config.bin_bp)
    genetic, knots, knot_genetic = genetic_coordinates(positions, chromosome_map)
    events, spans, marker_counts, expected, exposure, informative = decode_meioses(
        calls, positions, segments, coverage, offsets, parents, children, slots,
        chromosome_map.fallback_rate_per_bp, config.copy_error, config.minimum_origin_probability,
        config.maximum_gap_bp, config.artifact_rate, config.artifact_mean_bp, bins, genetic, knots, knot_genetic)
    crossover_offsets, crossover_rows = packed(events, 5)
    callable_offsets, callable_rows = packed(spans, 3)
    informative_offsets, informative_rows = packed(informative, 3)
    curve = aggregate_map(bins, crossover_rows, callable_rows, expected, exposure, informative_rows, knots, knot_genetic)
    applied = None
    if origin is not None:
        # Calls and spans were decoded per segment; report assembly components again.
        for rows, column in ((crossover_rows, 4), (callable_rows, 2), (informative_rows, 2)):
            if len(rows):
                rows[:, column] = origin[rows[:, column].astype(np.int64)]
        applied = AppliedBreakpoints(breakpoints[0], breakpoints[1], breakpoints[2], masked, origin)
    reported = junction_report.reported_intervals(crossover_rows, positions, segments, bin_edges, config.junctions)
    reversed_calls = junction_report.in_reversed_segment(crossover_rows, breakpoints)
    loci = junction_report.detect_junctions(
        phase.contig, names, parents, children, crossover_rows, crossover_offsets, reported, callable_rows,
        callable_offsets,
        config.junctions, chromosome_map, breakpoints)
    summary = RecombinationSummary(
        phase.contig, len(parents), int(np.count_nonzero(np.diff(callable_offsets))), len(crossover_rows),
        float(curve.expected_crossovers.sum()), int(marker_counts[:, 2].sum()),
        float(curve.exposure_meiosis_bp.sum()), float(curve.covered_bp.sum()),
        float(curve.cumulative_observed_cM[-1]), int(np.sum(curve.exposure_meiosis_bp == 0)),
        read_free_tracts, read_free_sites, bool(config.shared_family_evidence),
        None if fit is None else fit.converged, 0 if fit is None else fit.corrected_samples,
        0 if fit is None else len(fit.accepted),
        0 if breakpoints is None else len(breakpoints[0]) + len(breakpoints[1]), masked, len(loci),
        sum(locus.flagged for locus in loci), bool(phase.diagnostics.converged), time.monotonic() - started)
    return RecombinationProduct(
        phase.contig, names, parents, children, slots, crossover_offsets, crossover_rows, reported, reversed_calls,
        callable_offsets, callable_rows, informative_offsets, informative_rows, marker_counts, expected, exposure,
        curve, loci, fit, applied, summary)


@timed_stage("recombination")
def run(store, contigs, edges, config, genetic_maps, budget):
    """Recombination of every chromosome from its stored final phase; returns the RecombinationSummary rows
    (also stored as run record ``recombination``)."""
    contigs = tuple(map(str, contigs))
    store.require(phase_records.KEY, contigs)
    weights, process_limit = _resources(store, contigs, len(edges.sample_ids), config, budget)
    common = dict(root=store.root, edges=edges, config=config, genetic_maps=genetic_maps)
    with ChromosomeExecutor(contigs, _map_contig, budget=budget, weights=weights, max_workers=process_limit,
                            label="RECOMBINATION", task_bytes=weights) as executor:
        summaries = tuple(executor.run("recombination", common))
    store.run.put(records.KEY, summaries)
    store.mark_stage_complete(records.STAGE)
    return summaries


def _map_contig(contig, state, phase_name, common, budget):
    store = CheckpointStore(common["root"])
    file = store.contig(contig)
    edges, config = common["edges"], common["config"]
    chromosome_map = common["genetic_maps"].for_contig(contig)
    store.bind_identity(file, records.STAGE, records.stage_identity(file, edges, config, chromosome_map))
    if file.has(records.KEY):
        summary = records.load(file, nthreads=claim_threads()).summary
    else:
        phase = phase_records.load(file, nthreads=claim_threads())
        observed = None
        if config.read_free_tract_sites > 0:
            evidence = evidence_records.load_arrays(file, ("positions", "observed"), nthreads=claim_threads())
            if not np.array_equal(evidence["positions"], phase.positions):
                raise ValueError(f"{contig}: read-depth evidence and final phase sites differ")
            observed = np.asarray(evidence["observed"], dtype=bool)
            del evidence
        claim_threads()
        product = map_chromosome(phase, edges, observed, config, chromosome_map)
        del phase, observed
        records.write(file, product)
        summary = product.summary
        del product
        release_memory()
    if config.shared_family_evidence and not summary.shared_orientation_converged:
        warnings.warn(f"{contig}: shared-family orientation fit reached its sweep limit; the map remains "
                      "conditional on the selected, nonconverged shared paths.", RuntimeWarning)
    print(f"[recombination {contig}] calls={summary.crossovers} expected={summary.expected_crossovers:.3f} "
          f"junction loci={summary.junction_loci} (flagged {summary.junction_loci_flagged})", flush=True)
    return state, summary


def _resources(store, contigs, samples, config, budget):
    """Per-chromosome peak estimates (a scheduling estimate, not an RSS bound; also each task's memory reservation,
    execution.memory): six decoded phase products cover
    the retained phase arrays, working call/orientation/run copies (decoder-frame and corrected calls 2 N L each,
    run ids 4 N L) and per-meiosis workspaces, plus 2 GiB for reads and Python/JIT, plus the read-depth mask
    (N x L bool) when read-free tracts are removed."""
    estimates = []
    for contig in contigs:
        file = store.contig(contig)
        if file.has(records.KEY):
            estimate = 2 * file.info(records.KEY).decoded_bytes
        else:
            decoded = file.info(phase_records.KEY).decoded_bytes
            sites = len(phase_records.load_arrays(file, ("positions",))["positions"])
            estimate = 6 * decoded + (samples * sites if config.read_free_tract_sites > 0 else 0)
        estimates.append(estimate + (2 << 30))
    return estimates, memory_worker_limit(estimates, budget.threads(), label="RECOMBINATION")
