"""Pedigree-conditioned family solver for the stable final-phase product."""


from __future__ import annotations


import math
from dataclasses import dataclass
import numpy as np
from . import evidence as refinement_evidence
from . import messages as refinement_messages
from . import factors, selector_chains
from .phase_chains import phase_update as _phase_update, phase_context_view


MODEL_VERSION = "pedigree-phase-focused-v1"


@dataclass
class JointFamilyMessages:
    iteration: int
    parent_log: np.ndarray       # observed edges, sites, four parental genotypes
    child_log: np.ndarray        # observed children, sites, four child genotypes
    phase_match: np.ndarray
    segregation_match: np.ndarray  # observed children, sites, four selector pairs
    deltas: tuple[float,...]=()
    branch_clusters: tuple=()
    residual_history: tuple = ()



@dataclass(frozen=True)
class FamilyRefinementConfig:
    condition_on_scaffold_genotypes: bool = True
    scaffold_scope: str = "roots"
    correct_incomplete_parent_phase: bool = False
    root_gauge_moves: bool = True
    gauge_move_interval: int = 5
    recombination_rate: float = 5e-8
    phase_switch_probability: float = 0.005
    scaffold_phase_error: float = 0.01
    transmission_error: float = 0.001
    damping: float = 0.5
    max_iterations: int = 520
    minimum_iterations: int = 20
    required_unchanged: int = 5
    phase_retry_count: int = 2
    tolerance: float = 1e-4
    phase_call_probability: float = 0.98
    checkpoint_every: int = 5
    branch_cluster_after: int = 80
    branch_cluster_interval: int = 20

    def validated(self):
        value = self.phase_retry_count
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError("phase_retry_count must be a nonnegative integer")
        if self.scaffold_scope not in ("roots", "incomplete", "all"):
            raise ValueError("scaffold_scope must be roots, incomplete, or all")
        for name in ("max_iterations", "minimum_iterations", "required_unchanged", "checkpoint_every", "gauge_move_interval", "branch_cluster_after", "branch_cluster_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.recombination_rate) or self.recombination_rate < 0:
            raise ValueError("recombination_rate must be finite and nonnegative")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        for name in ("phase_switch_probability", "scaffold_phase_error", "transmission_error"):
            if not 0 < getattr(self, name) < 0.5:
                raise ValueError(f"{name} must lie strictly between 0 and 0.5")
        if not 0 < self.damping <= 1:
            raise ValueError("damping must lie in (0,1]")
        for name in ("phase_call_probability",):
            if not 0.5 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie strictly between 0.5 and 1")
        return self



def grouped_edges(parents,children,slots,samples):
    group_child=np.unique(children)
    node_group=np.full(samples,-1,dtype=np.int64)
    node_group[group_child]=np.arange(len(group_child))
    edge_group=node_group[children]
    group_parent=np.full((len(group_child),2),-1,dtype=np.int64)
    group_edge=np.full_like(group_parent,-1)
    for edge,(parent,child,slot) in enumerate(zip(parents,children,slots)):
        group=node_group[child];group_parent[group,slot]=parent;group_edge[group,slot]=edge
    return group_parent,group_edge,group_child,node_group,edge_group



def initialise_messages(samples, sites, parents, group_edge):
    groups = len(group_edge)
    return JointFamilyMessages(
        0, np.full((len(parents), sites, 4), -math.log(4), dtype=np.float32),
        np.full((groups, sites, 4), -math.log(4), dtype=np.float32),
        np.full((samples, sites), .999, dtype=np.float32),
        np.full((groups, sites, 4), .25, dtype=np.float32))


@dataclass
class FamilyWorkspace:
    """Fixed per-chromosome evidence; reused only with the same input axes/model."""
    reference: np.ndarray
    observed: np.ndarray
    positions: np.ndarray
    bins: np.ndarray
    components: np.ndarray
    base: np.ndarray
    orientation: np.ndarray
    reference_orientation: np.ndarray
    anchors: np.ndarray
    phase_theta: np.ndarray
    theta: np.ndarray
    parents: np.ndarray
    children: np.ndarray
    slots: np.ndarray
    offsets: np.ndarray
    adjacency: np.ndarray
    gp: np.ndarray
    ge: np.ndarray
    gc: np.ndarray
    node_group: np.ndarray
    edge_group: np.ndarray
    parent_counts: np.ndarray
    roots: np.ndarray
    selector_support: object = None
    selector_cache: object = None
    factor_support: object = None
    dirty_tiles: object = None



def prepare_family_workspace(genotype_likelihoods,observed,reference_alleles,positions,
                             phase_bins,component_ids,relationships,sample_ids,*,
                             config=FamilyRefinementConfig(),chromosome_map=None):
    cfg=config.validated()
    if cfg.scaffold_scope!="roots":raise ValueError("family refinement requires root-only scaffold phase anchors")
    gl=np.asarray(genotype_likelihoods,dtype=np.float64)
    obs=np.asarray(observed,dtype=bool);reference=np.asarray(reference_alleles,dtype=np.int8)
    pos=np.asarray(positions,dtype=np.int64);bins=np.asarray(phase_bins);components=np.asarray(component_ids)
    samples,sites=gl.shape[:2]
    if gl.shape!=(samples,sites,3) or sites==0 or samples!=len(sample_ids):raise ValueError("invalid GL axes")
    if obs.shape!=(samples,sites) or reference.shape!=(samples,sites,2):raise ValueError("invalid scaffold axes")
    if pos.shape!=(sites,) or bins.shape!=pos.shape or components.shape!=pos.shape or np.any(np.diff(pos)<=0):
        raise ValueError("invalid marker coordinates")
    parents,children,slots=pedigree_edges(relationships,sample_ids)
    offsets,adjacency=_adjacency(samples,parents,children)
    gp,ge,gc,node_group,edge_group=grouped_edges(parents,children,slots,samples)
    parent_counts=np.bincount(children,minlength=samples);roots=parent_counts==0
    root_prior=np.log(np.asarray([1/3,1/6,1/6,1/3],dtype=np.float32))
    base,reference_orientation,anchors,invalid=refinement_evidence.prepare_evidence(
        gl,obs,reference,roots,cfg.condition_on_scaffold_genotypes,root_prior)
    if np.any(invalid):raise ValueError("invalid allele evidence")
    orientation=reference_orientation.copy()
    orientation[~roots]=-1
    phase_theta=np.zeros(max(0,sites-1),dtype=np.float64)
    phase_theta[bins[1:]!=bins[:-1]]=cfg.phase_switch_probability
    phase_theta[components[1:]!=components[:-1]]=.5
    if chromosome_map is None or not chromosome_map.has_map:
        rate=cfg.recombination_rate if chromosome_map is None else chromosome_map.fallback_rate_per_bp
        theta=-.5*np.expm1(-2*rate*np.diff(pos))
    else:
        theta=-.5*np.expm1(-2*chromosome_map.interval_morgans(pos[:-1],pos[1:]))
    return FamilyWorkspace(reference,obs,pos,bins,components,base,orientation,reference_orientation,
        anchors,phase_theta,theta,parents,children,slots,offsets,adjacency,
        gp,ge,gc,node_group,edge_group,parent_counts,roots)



def refine_family(genotype_likelihoods,observed,reference_alleles,positions,phase_bins,
                  component_ids,relationships,sample_ids,*,config=FamilyRefinementConfig(),
                  resume=None,checkpoint_callback=None,chromosome_map=None,workspace=None):
    cfg=config.validated()
    if cfg.scaffold_scope!="roots":raise ValueError("family refinement requires root-only scaffold phase anchors")
    if workspace is None:
        workspace=prepare_family_workspace(genotype_likelihoods,observed,reference_alleles,positions,
            phase_bins,component_ids,relationships,sample_ids,config=cfg,chromosome_map=chromosome_map)
    reference=workspace.reference
    samples,sites=reference.shape[:2]
    base,orientation=workspace.base,workspace.orientation
    reference_orientation,anchors=workspace.reference_orientation,workspace.anchors
    phase_theta,theta=workspace.phase_theta,workspace.theta
    parents,children,slots=workspace.parents,workspace.children,workspace.slots
    offsets,adjacency=workspace.offsets,workspace.adjacency
    gp,ge,gc=workspace.gp,workspace.ge,workspace.gc
    node_group,edge_group=workspace.node_group,workspace.edge_group
    parent_counts,roots=workspace.parent_counts,workspace.roots
    messages=resume or initialise_messages(samples,sites,parents,ge)
    if (messages.parent_log.shape!=(len(parents),sites,4) or
        messages.child_log.shape!=(len(gc),sites,4) or messages.segregation_match.shape!=(len(gc),sites,4)):
        raise ValueError("joint message axes differ from pedigree/site axes")
    def beliefs():
        return factors.joint_beliefs(base,orientation,messages.phase_match,messages.parent_log,
            messages.child_log,node_group,offsets,adjacency,children,cfg.scaffold_phase_error)
    belief=beliefs();converged=bool(messages.deltas and messages.deltas[-1]<cfg.tolerance)
    if workspace.factor_support is None:
        workspace.factor_support=factors.prepare_factors(base,gp,gc,cfg.transmission_error)
    factor_support=workspace.factor_support
    emissions=factor_support[3]
    if workspace.selector_support is None:
        workspace.selector_support=selector_chains.prepare_selector_support(base,gp,theta)
    selector_support=workspace.selector_support
    if not converged and messages.iteration<cfg.max_iterations and workspace.selector_cache is None:
        workspace.selector_cache=selector_chains.ChainCache(selector_support)
    selector_cache=workspace.selector_cache
    if workspace.dirty_tiles is None:
        workspace.dirty_tiles=np.zeros((len(gc),(sites+31)//32),dtype=np.bool_)
    dirty_tiles=workspace.dirty_tiles
    while messages.iteration<cfg.max_iterations and not converged:
        damping=cfg.damping
        _,phase_delta,_,_=_phase_update(belief,orientation,messages.phase_match,phase_theta,anchors,
                                       cfg.scaffold_phase_error,damping,True)
        belief=beliefs()
        emissions,copy_delta=refinement_messages.copy_messages(belief,messages,gp,ge,gc,
            cfg.transmission_error,damping,messages.iteration%2==1,factor_support,dirty_tiles)
        seg_delta=refinement_messages.chain_messages(emissions,messages,theta,damping,selector_support,selector_cache,dirty_tiles)
        phase_change=float(np.max(phase_delta))
        copy_change=float(np.max(copy_delta))
        chain_change=float(np.max(seg_delta)) if len(seg_delta) else 0.
        change=max(phase_change,copy_change,chain_change)
        messages.residual_history=(*messages.residual_history,
            (phase_change/damping,copy_change/damping,chain_change/damping,
             tuple(map(int,np.flatnonzero(seg_delta >= cfg.tolerance*damping/cfg.damping))),
             damping))
        messages.iteration+=1;messages.deltas=(*messages.deltas,change);converged=change<cfg.tolerance
        belief=beliefs()
        gauge_changed=False
        if cfg.root_gauge_moves and messages.iteration%cfg.gauge_move_interval==0:
            maps=refinement_messages.selector_maps(emissions,messages,theta)
            flips,_=refinement_messages.root_mode_flips(belief,orientation,messages.phase_match,parents,edge_group,slots,
                roots,anchors,phase_theta,theta,maps,cfg.scaffold_phase_error)
            if np.any(flips):
                refinement_messages.apply_root_flips(flips,messages,parents,gp)
                if selector_cache is not None:selector_cache.invalidate()
                gauge_changed=True
                converged=False;messages.deltas=(*messages.deltas[:-1],max(1.,change));belief=beliefs()
        if (not converged and not gauge_changed and messages.iteration>=cfg.branch_cluster_after
                and messages.iteration%cfg.branch_cluster_interval==0):
            if refinement_messages.add_unstable_clusters(messages,gp,ge,gc,node_group,seg_delta,cfg.tolerance):
                messages.deltas=(*messages.deltas[:-1],max(1.,change))
                if selector_cache is not None:selector_cache.invalidate()
                belief=beliefs()
        if checkpoint_callback is not None and (converged or messages.iteration%cfg.checkpoint_every==0
                                                or messages.iteration==cfg.max_iterations):
            checkpoint_callback(messages)
    context,displayed,inferred=phase_context_view(belief,reference,reference_orientation,messages.phase_match,
        phase_theta,anchors,parent_counts,cfg.scaffold_phase_error,cfg.phase_call_probability,
        cfg.correct_incomplete_parent_phase)
    for cluster in messages.branch_clusters:
        joint=cluster.cavity.reshape((-1,4,4));a,b=cluster.groups
        messages.segregation_match[a],messages.segregation_match[b]=joint.sum(axis=2),joint.sum(axis=1)
    return FamilyPhaseState(context,displayed,inferred,messages,converged)


@dataclass(frozen=True)
class FamilyPhaseState:
    """Internal finalizer inputs only; no posterior probability product."""
    phase_context: np.ndarray
    phase_map: np.ndarray
    inferred_phase_map: np.ndarray
    messages: JointFamilyMessages
    converged: bool



def pedigree_edges(relationships, sample_ids):
    """Read resolved scientific M1/M2 identities without assigning generations."""
    names = tuple(map(str, sample_ids))
    lookup = {name: index for index, name in enumerate(names)}
    if len(lookup) != len(names):
        raise ValueError("sample IDs must be unique")
    if tuple(map(str, relationships["Sample"])) != names:
        raise ValueError("pedigree and genotype sample order differ")
    parents, children, slots = [], [], []
    for child, (_, row) in enumerate(relationships.iterrows()):
        state = row.get("ParentState")
        if state not in ("one_observed_parent", "two_observed_parents"):
            continue
        present = [(slot, lookup[str(row[column])]) for slot, column in enumerate(("Parent1", "Parent2"))
                   if str(row.get(column)) in lookup]
        if state == "two_observed_parents" and len(present) != 2:
            continue
        if state == "one_observed_parent" and len(present) != 1:
            continue
        if len({parent for _, parent in present}) != len(present) or any(parent == child for _, parent in present):
            raise ValueError("invalid self/duplicate parental identity")
        for slot, parent in present:
            parents.append(parent); children.append(child); slots.append(slot)
    return tuple(np.asarray(values, dtype=np.int64) for values in (parents, children, slots))



def _adjacency(samples, parents, children):
    neighbours = [[] for _ in range(samples)]
    for edge, (parent, child) in enumerate(zip(parents, children)):
        neighbours[int(parent)].append(edge); neighbours[int(child)].append(edge)
    offsets = np.concatenate(([0], np.cumsum([len(value) for value in neighbours]))).astype(np.int64)
    return offsets, np.asarray([edge for values in neighbours for edge in values], dtype=np.int64)
