"""Recombination stage: missing-aware maps from family phase's meiosis chains of every chromosome.

Per chromosome and pedigree edge, family phase releases the child's meiosis chain on that edge (FamilyPhaseProduct
chain_*: per painting bin the parent homologue it copies, released where both tie conventions give it a posterior of
at least phase_posterior, and the resets). Here:

- a chain is followed from reset to reset (component edges, meiosis resets, the parent's and the child's released
  phase-set starts, and supplied order breakpoints, junctions.split_at_breakpoints);
- a crossover is called between consecutive released bins of one followed stretch whose parental homologues differ
  (the called interval joins the two bins' centres; the reported interval holds both bins: the chain places a change
  at bin resolution);
- exposure runs from the first to the last released bin of each followed stretch, and the expected crossovers there
  are conditional on the released endpoint states: the parity of consecutive released bins, converted to an expected
  count over their interval by map.posterior_intervals (Haldane: lambda tanh lambda for even parity, lambda / tanh
  lambda for odd). It is not the full-data posterior mean: partial evidence inside an interval is not used. The
  family model routes some changes of homologue through not-a-copy, whose re-entry forgets the parental homologue,
  so the chain's own pairwise switch posteriors would miss them; the released endpoints see every difference
  between them, whatever produced it (a crossover, or a wrong edge or an upstream phase, identity or homology
  error);
- map aggregation and the junction table follow.

Results are conditional on the final phase and the fixed inferred pedigree, not calibrated joint-pedigree uncertainty.
"""
from __future__ import annotations

import time

import numpy as np

from ..execution import ChromosomeExecutor, claim_threads, memory_worker_limit, release_memory, timed_stage
from ..family_phase import records as phase_records
from ..family_phase.breakpoints import contig_breakpoints
from ..store import CheckpointStore
from . import junctions as junction_report
from . import records
from .map import aggregate_map, genetic_coordinates, map_bins, posterior_intervals
from .records import AppliedBreakpoints, RecombinationProduct, RecombinationSummary, packed


def chain_bins(phase, segments):
    """(left, right, centre bp float64 [M], segment int64 [M], reset bool [M]) of the chain arrays' painting bins:
    each bin's half-open bp range and centre (family phase's bin_centers: the mean position of its sites, which the
    meiosis transitions use), the segment of its sites on the decoded axis ``segments`` (-1 when its sites fall in
    two segments or in an excluded breakpoint zone), and where a new segment starts."""
    edges = [phase.component_bin_edges(c) for c in range(len(phase.bin_edge_offsets) - 1)]
    left = np.concatenate([e[:-1] for e in edges]).astype(np.float64) if edges else np.zeros(0)
    right = np.concatenate([e[1:] for e in edges]).astype(np.float64) if edges else np.zeros(0)
    positions = np.asarray(phase.positions, dtype=np.int64)
    first = np.searchsorted(positions, left, side="left")
    last = np.searchsorted(positions, right, side="left") - 1
    segment = np.full(len(left), -1, dtype=np.int64)
    for b in range(len(left)):
        if last[b] >= first[b]:
            values = segments[first[b]:last[b] + 1]
            if values[0] >= 0 and np.all(values == values[0]):
                segment[b] = values[0]
    reset = np.ones(len(left), dtype=bool)
    reset[1:] = segment[1:] != segment[:-1]
    return left, right, np.asarray(phase.bin_centers, dtype=np.float64), segment, reset


def follow_chain(source, support, reset, left, right, centres, segment):
    """One meiosis from its chain arrays: (called events (n, 5): left bp, right bp, released-homologue support left
    and right, segment; reported intervals (n, 2), inclusive bp; callable spans (m, 3); marker counts (released bins,
    bins of followed stretches, of them without a released homologue); the map markers for
    map.posterior_intervals: the released bins, whether each starts a stretch, and the released parity of the
    interval ending at each)."""
    released = (source >= 0) & (segment >= 0)
    stretch = np.cumsum(reset)
    events, reported, spans = [], [], []
    followed = np.zeros(len(source), dtype=bool)
    previous = -1
    for b in np.flatnonzero(released):
        if previous >= 0 and stretch[b] == stretch[previous]:
            followed[previous + 1:b + 1] = True
            part = float(segment[b])
            if spans and spans[-1][1] == centres[previous] and spans[-1][2] == part:
                spans[-1] = (spans[-1][0], float(centres[b]), part)
            else:
                spans.append((float(centres[previous]), float(centres[b]), part))
            if source[b] != source[previous]:
                events.append((float(centres[previous]), float(centres[b]), float(support[previous]),
                               float(support[b]), part))
                reported.append((int(left[previous]), int(right[b]) - 1))
        previous = b
    inside = followed.copy()
    starts = np.flatnonzero(followed & ~np.r_[False, followed[:-1]])
    inside[np.clip(starts - 1, 0, None)] = True
    counts = (int(released.sum()), int(inside.sum()), int(np.sum(inside & ~released)))
    markers = np.flatnonzero(released)
    marker_reset = np.ones(len(markers), dtype=bool)
    marker_reset[1:] = stretch[markers[1:]] != stretch[markers[:-1]]
    parity = np.zeros(len(markers), dtype=np.float64)
    parity[1:] = source[markers[1:]] != source[markers[:-1]]
    return (np.asarray(events, dtype=np.float64).reshape(-1, 5), np.asarray(reported, dtype=np.int64).reshape(-1, 2),
            np.asarray(spans, dtype=np.float64).reshape(-1, 3), counts, (markers, marker_reset, parity))


def map_chromosome(phase, edges, config, chromosome_map):
    """RecombinationProduct of one chromosome.

    ``phase`` FamilyPhaseProduct with meiosis chains (schema 4); ``edges`` ParentEdges (the edges the phase was
    conditioned on, whose order the chain rows follow).
    """
    started = time.monotonic()
    if phase.edges_identity != edges.identity:
        raise ValueError(f"{phase.contig}: family phase was conditioned on a different pedigree")
    if phase.chain_source is None:
        raise ValueError(f"{phase.contig}: the family-phase product carries no meiosis chains; recompute family phase")
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
    left, right, centres, segment, segment_reset = chain_bins(phase, segments)
    bins = map_bins(positions, segments, config.bin_bp)
    genetic, knots, knot_genetic = genetic_coordinates(positions, chromosome_map)
    centre_genetic = None if knots is None else np.interp(centres, knots, knot_genetic)
    rate = chromosome_map.fallback_rate_per_bp
    events, reported, spans, informative = [], [], [], []
    marker_counts = np.zeros((len(parents), 3), dtype=np.int64)
    expected = np.zeros((len(parents), len(bins) - 1))
    exposure = np.zeros_like(expected)
    for edge in range(len(parents)):
        called, report, callable_spans, counts, (markers, marker_reset, parity) = follow_chain(
            np.asarray(phase.chain_source[edge]), np.asarray(phase.chain_support[edge]),
            np.asarray(phase.chain_reset[edge]) | segment_reset, left, right, centres, segment)
        if centre_genetic is None:
            soft = posterior_intervals(centres[markers], segment[markers], marker_reset, parity, bins, rate)
        else:
            soft = posterior_intervals(centres[markers], segment[markers], marker_reset, parity, bins, rate,
                                       centre_genetic[markers], knots, knot_genetic)
        events.append(called)
        reported.append(report)
        spans.append(callable_spans)
        expected[edge], exposure[edge] = soft[0], soft[1]
        informative.append(soft[2])
        marker_counts[edge] = counts
    crossover_offsets, crossover_rows = packed(events, 5)
    callable_offsets, callable_rows = packed(spans, 3)
    informative_offsets, informative_rows = packed(informative, 3)
    reported_rows = np.concatenate(reported).astype(np.int64) if reported else np.zeros((0, 2), dtype=np.int64)
    curve = aggregate_map(bins, crossover_rows, callable_rows, expected, exposure, informative_rows, knots, knot_genetic)
    applied = None
    if origin is not None:
        # Calls and spans were followed per segment; report assembly components again.
        for rows, column in ((crossover_rows, 4), (callable_rows, 2), (informative_rows, 2)):
            if len(rows):
                rows[:, column] = origin[rows[:, column].astype(np.int64)]
        applied = AppliedBreakpoints(breakpoints[0], breakpoints[1], breakpoints[2], masked, origin)
    reversed_calls = junction_report.in_reversed_segment(crossover_rows, breakpoints)
    loci = junction_report.detect_junctions(
        phase.contig, names, parents, children, crossover_rows, crossover_offsets, reported_rows, callable_rows,
        callable_offsets, config.junctions, chromosome_map, breakpoints)
    summary = RecombinationSummary(
        phase.contig, len(parents), int(np.count_nonzero(np.diff(callable_offsets))), len(crossover_rows),
        float(curve.expected_crossovers.sum()), int(marker_counts[:, 2].sum()),
        float(curve.exposure_meiosis_bp.sum()), float(curve.covered_bp.sum()),
        float(curve.cumulative_observed_cM[-1]), int(np.sum(curve.exposure_meiosis_bp == 0)),
        0 if breakpoints is None else len(breakpoints[0]) + len(breakpoints[1]), masked, len(loci),
        sum(locus.flagged for locus in loci), bool(phase.diagnostics.converged), time.monotonic() - started)
    return RecombinationProduct(
        phase.contig, names, parents, children, slots, crossover_offsets, crossover_rows, reported_rows,
        reversed_calls, callable_offsets, callable_rows, informative_offsets, informative_rows, marker_counts, expected,
        exposure, curve, loci, applied, summary)


@timed_stage("recombination")
def run(store, contigs, edges, config, genetic_maps, budget):
    """Recombination of every chromosome from its stored final phase; returns the RecombinationSummary rows
    (also stored as run record ``recombination``)."""
    contigs = tuple(map(str, contigs))
    store.require(phase_records.KEY, contigs)
    weights, process_limit = _resources(store, contigs, budget)
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
        claim_threads()
        product = map_chromosome(phase, edges, config, chromosome_map)
        del phase
        records.write(file, product)
        summary = product.summary
        del product
        release_memory()
    print(f"[recombination {contig}] calls={summary.crossovers} expected={summary.expected_crossovers:.3f} "
          f"junction loci={summary.junction_loci} (flagged {summary.junction_loci_flagged})", flush=True)
    return state, summary


def _resources(store, contigs, budget):
    """Per-chromosome peak estimates (a scheduling estimate, not an RSS bound; also each task's memory reservation,
    execution.memory): twice the decoded phase product (the product and the chain rows in use; the per-edge map
    arrays are smaller), plus 2 GiB for reads and Python/JIT."""
    estimates = []
    for contig in contigs:
        file = store.contig(contig)
        if file.has(records.KEY):
            estimate = 2 * file.info(records.KEY).decoded_bytes
        else:
            estimate = 2 * file.info(phase_records.KEY).decoded_bytes
        estimates.append(estimate + (2 << 30))
    return estimates, memory_worker_limit(estimates, budget.threads(), label="RECOMBINATION")
