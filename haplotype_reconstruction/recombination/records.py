"""Recombination product types and checkpoint records.

``identity/recombination``  per contig: config + identities of the family-phase record, the pedigree edges and the
                            genetic map
``recombination``           per contig: the RecombinationProduct (frozen dataclasses of arrays)
run ``recombination``       the per-contig RecombinationSummary rows
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..family_phase import records as phase_records
from ..store.identity import json_digest, stage_identity as build_identity
from .junctions import JunctionLocus
from .map import RecombinationMap


STAGE = "recombination"
KEY = "recombination"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class AppliedBreakpoints:
    """Supplied order breakpoints of the contig: breakpoints int64 (k, 2) and reversed segments int64 (j, 2)
    (1-based inclusive bp), their reasons, masked sites, and the assembly component of every decoded segment."""

    breakpoints: np.ndarray
    reversed_segments: np.ndarray
    reasons: tuple[str, ...]
    masked_sites: int
    segment_components: np.ndarray


@dataclass(frozen=True)
class RecombinationSummary:
    """One row of the recombination summary table."""

    contig: str
    meioses: int
    observable_meioses: int
    crossovers: int
    expected_crossovers: float
    unresolved_bins: int
    exposure_meiosis_bp: float
    covered_bp: float
    cumulative_observed_cM: float
    uncovered_bins: int
    order_breakpoints: int
    order_breakpoint_masked_sites: int
    junction_loci: int
    junction_loci_flagged: int
    family_phase_converged: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class RecombinationProduct:
    """Recombination of one chromosome, conditional on family phase's meiosis chains and the fixed pedigree.

    edge_parent/edge_child/edge_slot int64 [E]; per-edge tables packed with offsets int64 [E+1]:
    crossovers float64 [n, 5] (left bp, right bp, released-homologue support left/right, assembly component: changes
    of parental homologue between the centres of consecutive released chain bins; even crossovers within one
    interval are not identifiable) with reported_intervals int64 [n, 2] (both bins, inclusive bp) and
    in_reversed_segment bool [n]; callable_spans float64 [m, 3] and informative_spans float64 [k, 3] (the followed
    stretches between released chain bins, as (left bp, right bp, component)); marker_counts int64 [E, 3] (released
    chain bins, bins of followed stretches, of them without a released homologue); expected_crossovers / exposure
    float64 [E, B] per map bin (working expected crossovers: Haldane interpolations conditional on the called
    endpoint states of the released chain bins, not posterior means; followed meiosis-bp).
    """

    contig: str
    sample_ids: tuple[str, ...]
    edge_parent: np.ndarray
    edge_child: np.ndarray
    edge_slot: np.ndarray
    crossover_offsets: np.ndarray
    crossovers: np.ndarray
    reported_intervals: np.ndarray
    in_reversed_segment: np.ndarray
    callable_offsets: np.ndarray
    callable_spans: np.ndarray
    informative_offsets: np.ndarray
    informative_spans: np.ndarray
    marker_counts: np.ndarray
    expected_crossovers: np.ndarray
    exposure: np.ndarray
    map: RecombinationMap
    junctions: tuple[JunctionLocus, ...]
    order_breakpoints: AppliedBreakpoints | None
    summary: RecombinationSummary

    def edge_crossovers(self, edge):
        return self.crossovers[self.crossover_offsets[edge]:self.crossover_offsets[edge + 1]]

    def edge_reported_intervals(self, edge):
        return self.reported_intervals[self.crossover_offsets[edge]:self.crossover_offsets[edge + 1]]

    def edge_callable_spans(self, edge):
        return self.callable_spans[self.callable_offsets[edge]:self.callable_offsets[edge + 1]]

    def edge_informative_spans(self, edge):
        return self.informative_spans[self.informative_offsets[edge]:self.informative_offsets[edge + 1]]


def packed(rows, columns):
    """(offsets int64 [E+1], float64 [n, columns]) of per-edge row arrays."""
    offsets = np.r_[0, np.cumsum([len(item) for item in rows])].astype(np.int64)
    values = (np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1, columns) for item in rows])
              if len(rows) else np.zeros((0, columns)))
    return offsets, values.reshape(-1, columns)


def stage_identity(file, edges, config, chromosome_map) -> dict:
    """Identity of a contig's recombination (module docstring)."""
    upstream = {"family_phase": file.info(phase_records.KEY).identity, "edges": edges.identity,
                "genetic_map": json_digest(chromosome_map.identity())}
    return build_identity(STAGE, schema=SCHEMA_VERSION, config=config, upstream=upstream,
                          packages=("recombination", "family_phase"))


def complete(file) -> bool:
    return file.has(KEY)


def progress(file) -> dict:
    return {"stage": STAGE, "complete": file.has(KEY)}


def write(file, product):
    return file.put(KEY, product)


def load(file, *, nthreads=None):
    return file.get(KEY, nthreads=nthreads)
