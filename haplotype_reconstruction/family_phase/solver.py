"""The family model of one chromosome: pedigree layout and the solver's fixed inputs (release 2.1).

Variables: ordered genotypes (four states per sample and site), one frame chain per pedigree root (does the
sample's frame match its scaffold orientation; it may switch only at painting-bin edges, with probability 1/2 at
component edges and never inside a component for a root with a single child edge), and per child one joint
four-state selector chain (one transmission bit per observed parent; Haldane switches from the genetic map,
continuous across assembly breaks). Factors: base potentials (potentials.py), copy factors and selector support
(factors.py). The family MAP (map_solver.py) is the posterior mode of this model; release 2.1 replaced the damped
loopy belief propagation of 2.0, which has no convergence guarantee on this loopy graph and cycled where a
family's data conflict (TroMau chr5/8/12/18/19), so that its capped release depended on the phase of the cycle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import factors
from .potentials import ROOT_PRIOR, base_potentials


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
    """Fixed per-chromosome model inputs (never saved).

    scaffold int8 [N, L, 2]; potentials float64 [N, L, 4]; orientation int8 [N, L] (roots' scaffold
    orientation, -1 elsewhere: what the phase chains and beliefs see); scaffold_orientation int8 [N, L] (every
    sample's, used by the released phase view); anchors int64 [N]; phase_theta float64 [L-1] (frame switch
    probability per marker interval: 0 inside a painting bin, phase_switch_probability at a bin edge, 1/2 at
    a component edge); frozen bool [N] (roots with exactly one child edge) and frozen_theta float64 [L-1]
    (their chains switch only at component edges); theta float64 [L-1] (Haldane meiosis switch probability);
    selector_support: factors.selector_support (each group's supported markers and composed switch probabilities).
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
    selector_support: tuple


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
        factors.selector_support(potentials, graph.group_parent, theta))
