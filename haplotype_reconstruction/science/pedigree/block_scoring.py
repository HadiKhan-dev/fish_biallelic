"""Continuous-chain block scoring with fixed genome retention and bin-share floors.

M2 fallback slots are chosen on original components, before any block aggregation.
The schedule and its traceback are likewise computed once on each whole component.
"""
from __future__ import annotations
import numpy as np
from ...contracts.pedigree import BlockScores, LowMaterialChromosomeScores
from . import blocks, exposure, forward, junctions, transmission
from .resident import resident
from .scoring import scored_morgans


def score(prepared, compact, chromosome, panel, eligibility, parameters, claim_threads=lambda: None, *, reverse=None):
    n = len(eligibility.sample_ids)
    edges = np.ascontiguousarray(np.argwhere(eligibility.parents), dtype=np.int64)
    retained_pairs = np.column_stack((np.repeat(np.arange(n), np.diff(compact.indptr)), compact.parents)).astype(np.int64)
    trios = panel.trios
    rows = np.concatenate((np.column_stack((np.arange(n), np.full((n, 2), -1))),
                           np.column_stack((retained_pairs, np.full(len(retained_pairs), -1))), trios)).astype(np.int64)
    required = eligibility.parents | eligibility.parents.T
    np.fill_diagonal(required, True)
    oriented = required.copy()
    np.fill_diagonal(oriented, False)
    dyads = np.ascontiguousarray(np.argwhere(oriented), dtype=np.int64)
    parts = {key: [] for key in ("zero", "one_raw", "two", "edge_matched", "edge_exposed", "pair_explained",
        "pair_exposed", "junctions", "callable_bins", "transmission_log_likelihood", "total_bins", "informative_markers")}
    descriptors = []
    matched_total, exposed_total = np.zeros((n, n)), np.zeros((n, n))
    reverse = parameters.reverse_conventions if reverse is None else reverse
    for component in prepared.components:
        axis, description = blocks.partition(component.selector_switch_probability, component.component_index)
        descriptors.extend(description)
        state = resident(component, parameters, claim_threads, reverse=reverse)
        claim_threads()
        values = forward.component_block_scores(state.projection, state.terms, rows, axis, len(description))
        parts["zero"].append(values[:n].T)
        parts["one_raw"].append(values[n:n + len(retained_pairs)].T)
        parts["two"].append(values[n + len(retained_pairs):].T)
        schedule = junctions.schedules(component.state_classes, component.named_alleles, component.bin_sizes, reverse=reverse)
        straight = np.count_nonzero(schedule.pairs[:, :-1] != schedule.pairs[:, 1:], axis=2)
        crossed = np.count_nonzero(schedule.pairs[:, :-1] != schedule.pairs[:, 1:, ::-1], axis=2)
        costs = np.minimum(straight, crossed)
        present = np.count_nonzero(component.state_classes >= 0, axis=2)
        for block in description:
            first, stop = block.first_bin, block.stop_bin
            matched, exposed, explained, pair_exposed, _, _ = exposure.expected_structure(
                np.ascontiguousarray(state.pair_masses[:, first:stop]), component.anchored_states,
                required, trios, component.factors.available, claim_threads, bins_only=True)
            matched_total += matched
            exposed_total += exposed
            for key, value in (("edge_matched", matched[edges[:, 0], edges[:, 1]]),
                               ("edge_exposed", exposed[edges[:, 0], edges[:, 1]]),
                               ("pair_explained", explained), ("pair_exposed", pair_exposed),
                               ("junctions", costs[:, max(first, 1) - 1:stop - 1].sum(axis=1)),
                               ("callable_bins", present[:, first:stop].sum(axis=1))):
                parts[key].append(np.asarray(value, dtype=np.float64)[None])
            parts["total_bins"].append(float(stop - first))
            parts["informative_markers"].append(int(component.bin_sizes[first:stop].sum()))
        claim_threads()
        parts["transmission_log_likelihood"].append(transmission.blocks(
            dyads, np.array(schedule.pairs, dtype=np.int64, order="C", copy=True),
            np.array(schedule.classes, dtype=np.int64, order="C", copy=True),
            np.array(schedule.known, order="C", copy=True),
            np.array(component.selector_switch_probability, copy=True), int(schedule.n_trajectories),
            float(parameters.direction.mismatch_probability), axis, len(description)).T)
        del state
    data = {key: np.ascontiguousarray(np.concatenate(value), dtype=np.float64)
            for key, value in parts.items() if key not in ("total_bins", "informative_markers")}
    data["total_bins"] = np.asarray(parts["total_bins"], dtype=np.float64)
    data["informative_markers"] = np.asarray(parts["informative_markers"], dtype=np.int64)
    zero_total = data["zero"].sum(axis=0)
    shares = data["total_bins"] / data["total_bins"].sum()
    floors = data["zero"] + shares[:, None] * (compact.floors[chromosome] - zero_total)[None]
    block_scores = BlockScores(tuple(descriptors), retained_pairs, floors=floors, **data)
    total = lambda key: data[key].sum(axis=0)
    return LowMaterialChromosomeScores(
        contig=prepared.contig, zero=zero_total, one_floored=compact.chromosome_reference(chromosome, eligibility),
        two=total("two"), edge_matched=matched_total, edge_exposed=exposed_total,
        pair_explained=total("pair_explained"), pair_exposed=total("pair_exposed"),
        total_bins=float(data["total_bins"].sum()), junctions=total("junctions").astype(np.int64),
        callable_bins=total("callable_bins").astype(np.int64), informative_markers=prepared.informative_markers,
        component_count=prepared.component_count, scored_component_count=len(prepared.components),
        junction_dyads=dyads, transmission_log_likelihood=total("transmission_log_likelihood"),
        scored_morgans=scored_morgans(prepared, parameters.exposure.unit_morgans),
        working_morgans=sum(b.working_morgans for b in descriptors),
        edge_exposed_morgans=np.zeros((n, n)), pair_exposed_morgans=np.zeros(len(trios)), blocks=block_scores)
