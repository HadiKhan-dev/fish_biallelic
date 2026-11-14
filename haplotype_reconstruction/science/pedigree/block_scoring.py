"""Continuous-chain piece scoring of one contig with fixed genome retention (DESIGN_417 I1).

Each scored component is cut into pieces at the whole-Morgan cuts of the genome axis (blocks.pieces, from the
contig's genome offset); every piece row is a sum of per-bin increments of chains computed once on the whole
component, so the stacking sums pieces into material units exactly. M2 fallback slots are chosen on original
components, before any aggregation; the junction schedule and its traceback are likewise computed once on each
whole component. Floors are not spread here: the stacking spreads the genome deficit by bins (D7).
"""

from __future__ import annotations

import numpy as np

from ...contracts.pedigree import BlockScores, ChromosomeScores
from ...core.errors import InvariantError
from . import blocks, exposure, forward, junctions, transmission
from .resident import resident


def score(prepared, compact, chromosome, panel, eligibility, parameters, offset, last, claim_threads=lambda: None, *,
          reverse=None, cuts=None):
    """The contig's pieces from genome coordinate ``offset`` (its first scored bin) with the genome's last nominal
    unit ``last`` (regimes.axis). ``cuts``: the barrier's cut list of this contig (blocks.cut_list), checked."""
    n = len(eligibility.sample_ids)
    if cuts is not None:
        mine = blocks.cut_list(tuple(c.selector_switch_probability for c in prepared.components), offset, last,
                               parameters.material)
        if not np.array_equal(mine, np.asarray(cuts, dtype=np.int64).reshape(-1, 3)):
            raise InvariantError(f"{prepared.contig}: prepared pieces differ from the genome cut list")
    material = parameters.material
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
    reverse = parameters.reverse_conventions if reverse is None else reverse
    start = float(offset)
    for component in prepared.components:
        axis, description = blocks.pieces(component.selector_switch_probability, start, last, material,
                                          component.component_index)
        start += blocks.component_length(component.selector_switch_probability, material)
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
        for piece in description:
            first, stop = piece.first_bin, piece.stop_bin
            matched, exposed, explained, pair_exposed = exposure.expected_structure(
                np.ascontiguousarray(state.pair_masses[:, first:stop]), component.anchored_states,
                required, trios, component.factors.available, claim_threads)
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
    return ChromosomeScores(
        contig=prepared.contig, offset_morgans=float(offset), end_morgans=float(start),
        informative_markers=prepared.informative_markers, component_count=prepared.component_count,
        scored_component_count=len(prepared.components), junction_dyads=dyads,
        floor=np.ascontiguousarray(compact.floors[chromosome], dtype=np.float64),
        working_morgans=float(sum(p.working_morgans for p in descriptors)),
        pieces=BlockScores(tuple(descriptors), retained_pairs, **data))
