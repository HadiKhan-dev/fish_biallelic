"""Coordinate views of the same family model; no schedule or orientation decision.

An order lists old sample rows in the requested new order. Parent slots, marker
positions, selector bits and within-chain marker order are never relabelled.
Callers explicitly choose which axes and reversal schedules to evaluate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib

import numpy as np

from . import solver



def pedigree_data_order(workspace):
    """Topological, discrete-data coordinate order with explicit duplicate certificates.

    No sample name or likelihood float enters an ordering key. Identical refined
    keys either have independent degree-zero nodes, or require exact pedigree
    twins with identical model rows.
    Their row order is a storage convention: callers must preserve exchangeability
    when releasing a member-specific answer. Unresolved groups are reported and
    must not be used as an implicitly ordered scientific search axis.
    """
    graph, scaffold = workspace.graph, workspace.scaffold
    samples = len(scaffold)
    incoming, outgoing = [[] for _ in range(samples)], [[] for _ in range(samples)]
    for parent, child, slot in zip(graph.parents, graph.children, graph.slots):
        incoming[int(child)].append((int(slot), int(parent)))
        outgoing[int(parent)].append((int(slot), int(child)))
    remaining = np.asarray([len(items) for items in incoming], dtype=np.int64)
    depth = np.zeros(samples, dtype=np.int64)
    frontier = np.flatnonzero(remaining == 0).tolist()
    visited = 0
    while frontier:
        following = []
        for parent in frontier:
            visited += 1
            for _, child in outgoing[parent]:
                depth[child] = max(depth[child], depth[parent] + 1)
                remaining[child] -= 1
                if remaining[child] == 0:
                    following.append(child)
        frontier = following
    if visited != samples:
        raise ValueError('family data order requires an acyclic pedigree')

    observed = workspace.observed
    data_keys, genotype_keys = [], []
    for sample, calls in enumerate(scaffold):
        # A summed dosage would conflate [1, -1] with [0, 0]. Encode both alleles.
        low, high = np.minimum(calls[:, 0], calls[:, 1]), np.maximum(calls[:, 0], calls[:, 1])
        unordered = ((low + 1) * 3 + high + 1).astype(np.uint8)
        ordered = ((calls[:, 0] + 1) * 3 + calls[:, 1] + 1).astype(np.uint8)
        genotype = hashlib.sha256(unordered.tobytes()).digest()
        phase = hashlib.sha256(ordered.tobytes()).digest()
        mask = b'' if observed is None else hashlib.sha256(np.packbits(observed[sample]).tobytes()).digest()
        genotype_keys.append((int(depth[sample]), genotype))
        data_keys.append((int(depth[sample]), genotype, phase, mask,
                          tuple(sorted(slot for slot, _ in incoming[sample])),
                          tuple(sorted(slot for slot, _ in outgoing[sample]))))

    def ranks(keys):
        levels = {key: rank for rank, key in enumerate(sorted(set(keys)))}
        return np.asarray([levels[key] for key in keys], dtype=np.int64), len(levels)

    colors, count = ranks(data_keys)
    initial_count = count
    from .refinement import refine
    colors, refinement = refine(colors, incoming, outgoing)
    count = refinement['refined_classes']
    order = np.argsort(colors, kind='stable').astype(np.int64)
    groups = [[] for _ in range(count)]
    for sample, color in enumerate(colors):
        groups[int(color)].append(sample)
    duplicates = [group for group in groups if len(group) > 1]
    certified, independent, unresolved = [], [], []

    def same_unary_ratios(left, right):
        # A per-site unary constant cancels from every model comparison. In
        # particular, the sole finite value at a hard genotype is irrelevant.
        finite = np.isfinite(left)
        if not np.array_equal(finite, np.isfinite(right)):
            return False
        normalized_left, normalized_right = np.zeros_like(left), np.zeros_like(right)
        np.subtract(left, np.max(left, axis=1, keepdims=True), out=normalized_left, where=finite)
        np.subtract(right, np.max(right, axis=1, keepdims=True), out=normalized_right, where=finite)
        return np.array_equal(normalized_left, normalized_right)

    for group in duplicates:
        # Degree-zero samples share no family factor, selector or root marginal.
        # Their own unary/frame-chain updates commute even when likelihood rows
        # differ. Their storage order need not imply exchangeable evidence or
        # identical member releases; no floating value is used to order them.
        if all(not incoming[i] and not outgoing[i] for i in group):
            independent.append(group)
            continue
        reference = group[0]
        same_neighbours = all(sorted(incoming[i]) == sorted(incoming[reference])
                              and sorted(outgoing[i]) == sorted(outgoing[reference]) for i in group[1:])
        # Equality can certify exchangeability, but it never orders float values.
        same_rows = all(np.array_equal(getattr(workspace, name)[reference], getattr(workspace, name)[i])
                        for name in ('scaffold', 'orientation', 'scaffold_orientation', 'anchors', 'frozen')
                        for i in group[1:])
        same_rows = same_rows and all(same_unary_ratios(workspace.potentials[reference], workspace.potentials[i])
                                      for i in group[1:])
        if observed is not None:
            same_rows = same_rows and all(np.array_equal(observed[reference], observed[i]) for i in group[1:])
        if same_neighbours and same_rows:
            certified.append(group)
        else:
            unresolved.append(dict(members=group, same_pedigree_neighbours=bool(same_neighbours),
                                   same_model_rows=bool(same_rows)))
    genotype_counts = {}
    for key in genotype_keys:
        genotype_counts[key] = genotype_counts.get(key, 0) + 1
    diagnostics = dict(policy='pedigree_depth_discrete_scaffold_splitters', samples=samples,
                       depth_counts=np.bincount(depth).tolist(), refinement_rounds=None, refinement=refinement,
                       initial_classes=initial_count, refined_classes=count,
                       genotype_duplicate_groups=sum(value > 1 for value in genotype_counts.values()),
                       refined_duplicate_groups=len(duplicates),
                       certified_duplicate_groups=certified, independent_duplicate_groups=independent,
                       unresolved_duplicate_groups=unresolved)
    return order, diagnostics


def sample_order(old_names, new_names):
    """Old sample row of each name on the new axis."""
    lookup = {str(name): i for i, name in enumerate(old_names)}
    return np.asarray([lookup[str(name)] for name in new_names], dtype=np.int64)


def _packed(values, offsets, order):
    pieces = [values[offsets[i]:offsets[i + 1]] for i in order]
    new_offsets = np.r_[0, np.cumsum([len(piece) for piece in pieces])].astype(np.int64)
    return (np.concatenate(pieces) if pieces else values[:0].copy()), new_offsets


def make_view(sample_ids, scaffold, workspace, edges, ties, order):
    """(names, scaffold, workspace, ParentEdges, ties) on an explicitly requested axis.

    Identity order returns the original five objects. Other views copy sample
    rows and packed groups, sharing only the unchanged chromosome/site arrays.
    Selector support is permuted exactly, never recomputed from floating values.
    """
    order = np.asarray(order, dtype=np.int64)
    if np.array_equal(order, np.arange(len(sample_ids))):
        return sample_ids, scaffold, workspace, edges, ties
    names = tuple(str(sample_ids[i]) for i in order)
    inverse = np.argsort(order)
    parents, children = inverse[edges.parents], inverse[edges.children]
    edge_order = np.lexsort((edges.slots, children))
    new_edges = replace(edges, sample_ids=names, parents=parents[edge_order], children=children[edge_order],
                        slots=edges.slots[edge_order],
                        observed_parent_minimum=edges.observed_parent_minimum[order],
                        observed_parent_maximum=edges.observed_parent_maximum[order])
    graph = solver.family_graph(new_edges.parents, new_edges.children, new_edges.slots, len(names))
    coverage, coverage_offsets = _packed(scaffold.coverage, scaffold.coverage_offsets, order)
    new_scaffold = replace(scaffold, alleles=scaffold.alleles[order], coverage=coverage,
                           coverage_offsets=coverage_offsets)
    groups = workspace.graph.node_group[order[graph.group_child]]
    offsets, markers, theta = workspace.selector_support
    new_markers, new_offsets = _packed(markers, offsets, groups)
    new_theta, _ = _packed(theta, offsets, groups)
    new_workspace = replace(workspace,
        scaffold=(new_scaffold.alleles if workspace.scaffold is scaffold.alleles else workspace.scaffold[order]),
        potentials=workspace.potentials[order], orientation=workspace.orientation[order],
        scaffold_orientation=workspace.scaffold_orientation[order], anchors=workspace.anchors[order],
        frozen=workspace.frozen[order], graph=graph, selector_support=(new_offsets, new_markers, new_theta),
        observed=None if workspace.observed is None else workspace.observed[order])
    new_ties = {int(inverse[int(sample)]): boundaries for sample, boundaries in ties.items()}
    return names, new_scaffold, new_workspace, new_edges, new_ties


def align_state(state, old_names, new_names, old_graph, new_graph):
    """Copy a MapState to another axis, matching selector rows by their child name."""
    from .map_solver import MapState
    order = sample_order(old_names, new_names)
    groups = old_graph.node_group[order[new_graph.group_child]]
    return MapState(state.genotype[order], state.selector[groups], state.frame[order])


def align_result(result, old_names, new_names, old_graph, new_graph):
    """MapResult with state and census row indices aligned; scores and site indices unchanged."""
    inverse = np.argsort(sample_order(old_names, new_names))
    group_inverse = new_graph.node_group[inverse[old_graph.group_child]]
    decisions = []
    for kind, row, position, gain in result.decisions:
        # Child block/compound sources are group rows; root/relabel/site moves use sample rows.
        mapped = group_inverse[int(row)] if int(kind) in (0, 3, 4, 5, 8) else inverse[int(row)]
        decisions.append((kind, int(mapped), position, gain))
    return replace(result, state=align_state(result.state, old_names, new_names, old_graph, new_graph),
                   decisions=tuple(decisions))


def _child_gain_order(parent, old_names, new_names, old_graph, new_graph, *, excluding_shared_with=None):
    """Map a root's positional vote tuple, including the guard's nonshared-child restriction."""
    old_parent = int(parent)
    new_parent = tuple(map(str, new_names)).index(str(old_names[old_parent]))
    excluded = set()
    if excluding_shared_with is not None:
        excluded = {str(old_names[c]) for p, c in zip(old_graph.parents, old_graph.children)
                    if p == excluding_shared_with}
    old_keys = [(str(old_names[c]), int(slot)) for p, c, slot in
                zip(old_graph.parents, old_graph.children, old_graph.slots)
                if p == old_parent and str(old_names[c]) not in excluded]
    new_keys = [(str(new_names[c]), int(slot)) for p, c, slot in
                zip(new_graph.parents, new_graph.children, new_graph.slots)
                if p == new_parent and str(new_names[c]) not in excluded]
    lookup = {key: i for i, key in enumerate(old_keys)}
    return [lookup[key] for key in new_keys]


def align_frame_repair(repair, old_names, new_names, *, old_graph=None, new_graph=None):
    """FrameRepair in new coordinates, including sample IDs, masks and root tie keys.

    Pass both graphs to align positional per-child gain tuples to the new edge
    order. Without graphs, those tuples keep their original provenance order;
    scalar scores and counts never masquerade as sample indices.
    """
    order = sample_order(old_names, new_names)
    inverse = np.argsort(order)
    reorder_votes = old_graph is not None and new_graph is not None

    def decision(item):
        changes = dict(sample=int(inverse[item.sample]))
        if reorder_votes:
            indices = _child_gain_order(item.sample, old_names, new_names, old_graph, new_graph)
            # Some retained diagnostic decisions do not store per-child votes.
            for name in ('child_gains', 'child_net_gains'):
                values = getattr(item, name)
                changes[name] = tuple(values[i] for i in indices) if values else ()
        return replace(item, **changes)

    guards = []
    for item in repair.guard_blocks:
        changes = dict(sample=int(inverse[item.sample]), blocking_parent=int(inverse[item.blocking_parent]))
        if reorder_votes:
            indices = _child_gain_order(item.blocking_parent, old_names, new_names, old_graph, new_graph,
                                        excluding_shared_with=item.sample)
            changes['blocking_child_gains'] = (tuple(item.blocking_child_gains[i] for i in indices)
                                               if item.blocking_child_gains else ())
        guards.append(replace(item, **changes))
    masks = {name: None if getattr(repair, name) is None else getattr(repair, name)[order]
             for name in ('internal', 'display', 'ambiguous')}
    undecided = (None if repair.undecided_ties is None else
                 {int(inverse[int(sample)]): boundaries for sample, boundaries in repair.undecided_ties.items()})
    return replace(repair, moves=tuple(decision(item) for item in repair.moves),
        reverts=tuple(decision(item) for item in repair.reverts),
        transfers=tuple(decision(item) for item in repair.transfers),
        child_frames=tuple(replace(item, sample=int(inverse[item.sample]), parent=int(inverse[item.parent]))
                           for item in repair.child_frames),
        guard_blocks=tuple(guards), flagged_edges=tuple((int(inverse[sample]), count)
                                                        for sample, count in repair.flagged_edges),
        undecided_ties=undecided, **masks)


def align_release_record(record, old_names, new_names, *, old_graph=None, new_graph=None):
    """Copy witness sample indices; optional graphs also align root per-child vote tuples."""
    inverse = np.argsort(sample_order(old_names, new_names))
    result = deepcopy(record)
    if 'disagreements' in result:
        result['disagreements'] = [(int(inverse[int(sample)]), start, stop)
                                    for sample, start, stop in result['disagreements']]
    for key in ('painting_joins', 'uncertain_joins'):
        for row in result.get(key, ()):
            sample = int(row['sample'])
            if old_graph is not None and new_graph is not None and row.get('repair_witnesses'):
                indices = _child_gain_order(sample, old_names, new_names, old_graph, new_graph)
                for witness in row['repair_witnesses']:
                    for name in ('child_gains', 'child_net_gains'):
                        values = witness.get(name, ())
                        if values:
                            witness[name] = tuple(values[i] for i in indices)
            row['sample'] = int(inverse[sample])
    return result
