"""Damped loopy belief propagation of the family model on one chromosome.

Variables: ordered genotypes (four states per sample and site), one phase chain per pedigree root (does the
sample's frame match its scaffold orientation; it may switch only at painting-bin edges, with probability 1/2 at
component edges and never inside a component for a root with a single child edge), and per child one joint
four-state selector chain (one transmission bit per observed parent; Haldane switches from the genetic map,
continuous across assembly breaks). Factors: base potentials (potentials.py), copy factors (factors.py).
One iteration updates the phase chains, sweeps the copy factors (direction alternating) and updates the
selector chains; every ``gauge_move_interval`` iterations joint root/children gauge moves (gauges.py) may
relabel a root's homologues. The release state is the one after ``max_iterations`` iterations or the first
iteration whose largest message change is below ``tolerance`` without a gauge move.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from numba import njit, prange

from ..execution import claim_threads
from ..maths import chains
from . import factors, gauges, selector_chains
from .potentials import ROOT_PRIOR, base_potentials


@dataclass
class FamilyMessages:
    """Mutable message state (the resumable part of the solve).

    parent_log float32 [E, L, 4]: copy-factor message to each edge's parent (log, normalised);
    child_log float32 [G, L, 4]: copy-factor message to each child group's child;
    phase_match float32 [N, L]: phase-chain message (probability the frame matches the scaffold orientation);
    segregation_match float32 [G, L, 4]: selector-chain cavity of each child group's four selector states;
    residuals: largest message change of every iteration so far (1 or more after a gauge move).
    """

    iteration: int
    parent_log: np.ndarray
    child_log: np.ndarray
    phase_match: np.ndarray
    segregation_match: np.ndarray
    residuals: tuple[float, ...] = ()


def initial_messages(samples, sites, edges, groups):
    return FamilyMessages(
        0, np.full((edges, sites, 4), -math.log(4), dtype=np.float32),
        np.full((groups, sites, 4), -math.log(4), dtype=np.float32),
        np.full((samples, sites), .999, dtype=np.float32),
        np.full((groups, sites, 4), .25, dtype=np.float32))


@dataclass(frozen=True)
class FamilyGraph:
    """Pedigree layout on the sample axis. Groups are children with at least one observed parent.

    parents/children/slots int64 [E] (edges); offsets int64 [N+1] + adjacency int64 [2E] (edges incident to
    each sample); group_parent/group_edge int64 [G, 2] (-1 for an unobserved slot); group_child int64 [G];
    node_group int64 [N] (-1 for roots); edge_group int64 [E]; parent_counts int64 [N] (observed parents);
    roots bool [N] (no observed parent).
    """

    parents: np.ndarray
    children: np.ndarray
    slots: np.ndarray
    offsets: np.ndarray
    adjacency: np.ndarray
    group_parent: np.ndarray
    group_edge: np.ndarray
    group_child: np.ndarray
    node_group: np.ndarray
    edge_group: np.ndarray
    parent_counts: np.ndarray
    roots: np.ndarray


def family_graph(parents, children, slots, samples):
    parents, children, slots = (np.asarray(values, dtype=np.int64) for values in (parents, children, slots))
    neighbours = [[] for _ in range(samples)]
    for edge, (parent, child) in enumerate(zip(parents, children)):
        neighbours[int(parent)].append(edge)
        neighbours[int(child)].append(edge)
    offsets = np.concatenate(([0], np.cumsum([len(value) for value in neighbours]))).astype(np.int64)
    adjacency = np.asarray([edge for values in neighbours for edge in values], dtype=np.int64)
    group_child = np.unique(children)
    node_group = np.full(samples, -1, dtype=np.int64)
    node_group[group_child] = np.arange(len(group_child))
    edge_group = node_group[children]
    group_parent = np.full((len(group_child), 2), -1, dtype=np.int64)
    group_edge = np.full_like(group_parent, -1)
    for edge, (parent, child, slot) in enumerate(zip(parents, children, slots)):
        group = node_group[child]
        group_parent[group, slot] = parent
        group_edge[group, slot] = edge
    parent_counts = np.bincount(children, minlength=samples).astype(np.int64)
    return FamilyGraph(parents, children, slots, offsets, adjacency, group_parent, group_edge, group_child,
                       node_group, edge_group, parent_counts, parent_counts == 0)


@dataclass
class FamilyWorkspace:
    """Fixed per-chromosome model inputs plus the solver's transient caches (never saved).

    scaffold int8 [N, L, 2]; potentials float32 [N, L, 4]; orientation int8 [N, L] (roots' scaffold
    orientation, -1 elsewhere: what the phase chains and beliefs see); scaffold_orientation int8 [N, L] (every
    sample's, used by the released phase view); anchors int64 [N]; phase_theta float64 [L-1] (frame switch
    probability per marker interval: 0 inside a painting bin, phase_switch_probability at a bin edge, 1/2 at
    a component edge); frozen bool [N] (roots with exactly one child edge) and frozen_theta float64 [L-1]
    (their chains switch only at component edges); theta float64 [L-1] (Haldane meiosis switch probability).
    """

    scaffold: np.ndarray
    positions: np.ndarray
    bins: np.ndarray
    components: np.ndarray
    potentials: np.ndarray
    orientation: np.ndarray
    scaffold_orientation: np.ndarray
    anchors: np.ndarray
    phase_theta: np.ndarray
    frozen: np.ndarray
    frozen_theta: np.ndarray
    theta: np.ndarray
    graph: FamilyGraph
    factor_support: tuple
    selector_support: tuple
    selector_cache: object = None
    dirty_tiles: np.ndarray = field(default=None, repr=False)


def meiosis_switch_probabilities(positions, chromosome_map):
    """Haldane switch probability of every marker interval from the chromosome's genetic map (NumPy's
    vectorised expm1, which differs in the last bit from the scalar expm1 of model.genetic_map.haldane_fraction)."""
    if not chromosome_map.has_map:
        return -.5 * np.expm1(-2 * chromosome_map.fallback_rate_per_bp * np.diff(positions))
    return -.5 * np.expm1(-2 * chromosome_map.interval_morgans(positions[:-1], positions[1:]))


def prepare_workspace(likelihoods, observed, scaffold, positions, bins, components, graph, config,
                      chromosome_map):
    samples, sites = observed.shape
    potentials, scaffold_orientation, anchors = base_potentials(
        likelihoods, observed, scaffold, graph.roots, ROOT_PRIOR)
    orientation = scaffold_orientation.copy()
    orientation[~graph.roots] = -1
    phase_theta = np.zeros(max(0, sites - 1), dtype=np.float64)
    phase_theta[bins[1:] != bins[:-1]] = config.phase_switch_probability
    phase_theta[components[1:] != components[:-1]] = .5
    frozen_theta = np.where(components[1:] != components[:-1], .5, 0.)
    # With one child, a root frame change and that child's crossover explain the data equally well.
    frozen = graph.roots & (np.bincount(graph.parents, minlength=samples) == 1)
    theta = meiosis_switch_probabilities(positions, chromosome_map)
    return FamilyWorkspace(
        scaffold, positions, bins, components, potentials, orientation, scaffold_orientation, anchors,
        phase_theta, frozen, frozen_theta, theta, graph,
        factors.factor_support(potentials, graph.group_parent, graph.group_child, config.transmission_error),
        factors.selector_support(potentials, graph.group_parent, theta))


@njit(cache=True, inline="always")
def phase_emission(belief_row, orientation_value, phase_value, phase_error):
    """Phase-chain emission (frame matches, frame swapped) of one site from the cavity of its phase message."""
    orient = orientation_value
    if orient < 0:
        return .5, .5
    p = phase_error + (1 - 2 * phase_error) * phase_value
    f1 = p if orient == 0 else 1 - p
    f2 = 1 - p if orient == 0 else p
    q0 = math.exp(float(belief_row[0])) / .5
    q1 = math.exp(float(belief_row[1])) / f1
    q2 = math.exp(float(belief_row[2])) / f2
    q3 = math.exp(float(belief_row[3])) / .5
    total = ((q0 + q1) + q2) + q3
    q0 /= total
    q1 /= total
    q2 /= total
    q3 /= total
    matching_zero = phase_error + (1 - 2 * phase_error)
    matching_one = phase_error
    e10 = matching_zero if orient == 0 else 1 - matching_zero
    e20 = 1 - matching_zero if orient == 0 else matching_zero
    e11 = matching_one if orient == 0 else 1 - matching_one
    e21 = 1 - matching_one if orient == 0 else matching_one
    e0 = ((q0 * .5 + q1 * e10) + q2 * e20) + q3 * .5
    e1 = ((q0 * .5 + q1 * e11) + q2 * e21) + q3 * .5
    return e0, e1


@njit(cache=True, parallel=True)
def update_phase_messages(belief, orientation, phase, phase_theta, anchors, phase_error, damping,
                          frozen, frozen_theta):
    """Damped phase-chain messages (in place); returns the largest change per sample.

    A sample without an anchor or any oriented site has no phase chain; its message decays to 1/2.
    """
    samples, sites, _ = belief.shape
    delta = np.zeros(samples, dtype=np.float64)
    free_starts, free_trans = chains.chain_segments(phase_theta, sites)
    frozen_starts, frozen_trans = chains.chain_segments(frozen_theta, sites)
    active = np.ones(samples, dtype=np.bool_)
    for sample in prange(samples):
        if anchors[sample] < 0:
            informative = False
            for site in range(sites):
                if orientation[sample, site] >= 0:
                    informative = True
                    break
            if not informative:
                active[sample] = False
                for site in range(sites):
                    old = phase[sample, site]
                    new = (1 - damping) * old + damping * .5
                    delta[sample] = max(delta[sample], abs(new - old))
                    phase[sample, site] = new
    active_samples = np.flatnonzero(active)
    for index in prange(len(active_samples)):
        sample = active_samples[index]
        starts, trans = (frozen_starts, frozen_trans) if frozen[sample] else (free_starts, free_trans)
        emissions = np.empty((sites, 2))
        for site in range(sites):
            emissions[site, 0], emissions[site, 1] = phase_emission(
                belief[sample, site], orientation[sample, site], phase[sample, site], phase_error)
        compact, anchor_run = chains.contracted_emissions(emissions, starts, anchors[sample])
        posterior = chains.binary_posterior(compact, trans, anchor_run)
        for run in range(len(starts) - 1):
            p0, p1 = posterior[run, 0], posterior[run, 1]
            for site in range(starts[run], starts[run + 1]):
                c0 = p0 / emissions[site, 0]
                c1 = p1 / emissions[site, 1]
                old = phase[sample, site]
                new = (1 - damping) * old + damping * c0 / (c0 + c1)
                delta[sample] = max(delta[sample], abs(new - old))
                phase[sample, site] = new
    return delta


@njit(cache=True, parallel=True)
def phase_view(belief, orientation, phase, phase_theta, anchors, parent_counts, phase_error,
               frozen, frozen_theta):
    """Family MAP frames int8 [N, L]: (inferred, displayed); displayed = inferred for two-parent samples, 0 else.

    Every sample's MAP phase path over its painting-bin runs, from its scaffold orientation; roots also use
    their phase message, other samples a neutral one.
    """
    samples, sites, _ = belief.shape
    inferred = np.empty((samples, sites), dtype=np.int8)
    displayed = np.empty_like(inferred)
    free_starts, free_trans = chains.chain_segments(phase_theta, sites)
    frozen_starts, frozen_trans = chains.chain_segments(frozen_theta, sites)
    for sample in prange(samples):
        starts, trans = (frozen_starts, frozen_trans) if frozen[sample] else (free_starts, free_trans)
        emissions = np.empty((sites, 2))
        for site in range(sites):
            alignment = phase[sample, site] if parent_counts[sample] == 0 else .5
            emissions[site, 0], emissions[site, 1] = phase_emission(
                belief[sample, site], orientation[sample, site], alignment, phase_error)
        compact, anchor_run = chains.contracted_emissions(emissions, starts, anchors[sample])
        path = chains.binary_viterbi(compact, trans, anchor_run)
        for run in range(len(starts) - 1):
            flip = path[run]
            shown = flip if parent_counts[sample] >= 2 else 0
            for site in range(starts[run], starts[run + 1]):
                inferred[sample, site] = flip
                displayed[sample, site] = shown
    return inferred, displayed


def beliefs(workspace, messages, config):
    graph = workspace.graph
    return factors.joint_beliefs(
        workspace.potentials, workspace.orientation, messages.phase_match, messages.parent_log,
        messages.child_log, graph.node_group, graph.offsets, graph.adjacency, graph.children,
        config.scaffold_phase_error)


def selector_emissions(workspace, messages, config):
    """Copy-factor selector emissions of the current messages, float32 [G, L, 4] (no message changes)."""
    graph = workspace.graph
    belief = beliefs(workspace, messages, config)
    emissions, _ = factors.copy_factor_sweep(
        belief, messages.parent_log, messages.child_log, messages.segregation_match, graph.group_parent,
        graph.group_edge, graph.group_child, config.transmission_error, 0., False, workspace.factor_support)
    return emissions


def iterate(workspace, messages, config, after_iteration=None):
    """Run belief propagation from ``messages`` (in place) to the release state; returns ``converged``.

    ``after_iteration(messages)`` is called after every iteration (resume checkpoints).
    """
    graph = workspace.graph
    cfg = config
    if workspace.selector_cache is None:
        workspace.selector_cache = selector_chains.SelectorChainCache(workspace.selector_support)
    if workspace.dirty_tiles is None:
        workspace.dirty_tiles = np.zeros(
            (len(graph.group_child), (messages.phase_match.shape[1] + factors.FACTOR_TILE_SITES - 1)
             // factors.FACTOR_TILE_SITES), dtype=np.bool_)
    belief = beliefs(workspace, messages, cfg)
    converged = bool(messages.residuals and messages.residuals[-1] < cfg.tolerance)
    root_groups = np.unique(graph.edge_group[graph.roots[graph.parents]])
    while messages.iteration < cfg.max_iterations and not converged:
        claim_threads()
        damping = cfg.damping
        phase_delta = update_phase_messages(
            belief, workspace.orientation, messages.phase_match, workspace.phase_theta, workspace.anchors,
            cfg.scaffold_phase_error, damping, workspace.frozen, workspace.frozen_theta)
        belief = beliefs(workspace, messages, cfg)
        workspace.dirty_tiles.fill(False)
        emissions, copy_delta = factors.copy_factor_sweep(
            belief, messages.parent_log, messages.child_log, messages.segregation_match, graph.group_parent,
            graph.group_edge, graph.group_child, cfg.transmission_error, damping, messages.iteration % 2 == 1,
            workspace.factor_support, workspace.dirty_tiles)
        chain_delta = selector_chains.update_selector_chains(
            emissions, messages.segregation_match, damping, workspace.selector_support,
            workspace.selector_cache, workspace.dirty_tiles)
        change = max(float(np.max(phase_delta)), float(np.max(copy_delta)),
                     float(np.max(chain_delta)) if len(chain_delta) else 0.)
        messages.iteration += 1
        messages.residuals = (*messages.residuals, change)
        converged = change < cfg.tolerance
        belief = beliefs(workspace, messages, cfg)
        if messages.iteration % cfg.gauge_move_interval == 0:
            paths = gauges.selector_paths(emissions, workspace.theta, root_groups)
            flips = gauges.root_mode_flips(
                belief, workspace.orientation, messages.phase_match, graph.parents, graph.edge_group,
                graph.slots, graph.roots, workspace.anchors, workspace.phase_theta, workspace.theta, paths,
                cfg.scaffold_phase_error, workspace.frozen, workspace.frozen_theta, cfg.minimum_gauge_support)
            if np.any(flips):
                gauges.apply_root_flips(flips, messages.phase_match, messages.parent_log,
                                        messages.segregation_match, graph.parents, graph.group_parent)
                workspace.selector_cache.invalidate()
                converged = False
                messages.residuals = (*messages.residuals[:-1], max(1., change))
                belief = beliefs(workspace, messages, cfg)
        if after_iteration is not None:
            after_iteration(messages)
    return converged


def released_frames(workspace, messages, config):
    """(inferred, displayed) family MAP frames of the current messages (phase_view)."""
    graph = workspace.graph
    belief = beliefs(workspace, messages, config)
    return phase_view(belief, workspace.scaffold_orientation, messages.phase_match, workspace.phase_theta,
                      workspace.anchors, graph.parent_counts, config.scaffold_phase_error, workspace.frozen,
                      workspace.frozen_theta)
