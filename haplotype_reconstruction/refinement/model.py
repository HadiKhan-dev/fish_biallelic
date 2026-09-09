"""refinement / model for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import dataclass, replace
from copy import deepcopy
import math, time
import numpy as np


from numba import njit, prange


@dataclass
class JointFamilyMessages:
    iteration: int
    parent_log: np.ndarray       # observed edges, sites, four parental genotypes
    child_log: np.ndarray        # observed children, sites, four child genotypes
    phase_match: np.ndarray
    segregation_match: np.ndarray  # observed children, sites, four selector pairs
    deltas: tuple[float,...]=()
    warm_start_iterations: int=0
    branch_clusters: tuple=()


MODEL_VERSION = "pedigree-paired-meiosis-branch-cavity-v2"


@njit(cache=True, parallel=True)
def complete_scaffold(corrected, phase_flip, posterior, threshold):
    samples, sites, _ = corrected.shape
    calls = corrected.copy()
    support = np.zeros(corrected.shape, dtype=np.float32)
    for sample in prange(samples):
        has_phase_frame = False
        for site in range(sites):
            a, b = corrected[sample, site, 0], corrected[sample, site, 1]
            if a != b:
                has_phase_frame = True
                break
        # With no oriented scaffold at all, H1/H2 may be defined directly by
        # the observed-parent slots. This is a label convention, not evidence.
        for site in range(sites):
            a, b = corrected[sample, site, 0], corrected[sample, site, 1]
            q = posterior[sample, site]
            genotype = np.array([q[0], q[1]+q[2], q[3]])
            if a >= 0:
                support[sample, site, 0] = 1.  # conditional accepted scaffold
            if b >= 0:
                support[sample, site, 1] = 1.
            if (a >= 0) != (b >= 0):
                known = a if a >= 0 else b
                slot = 1 if a >= 0 else 0
                denominator = genotype[known]+genotype[known+1]
                alt = genotype[known+1]/denominator
                confidence = max(alt, 1-alt)
                if confidence >= threshold:
                    calls[sample, site, slot] = int(alt >= .5)
                    support[sample, site, slot] = confidence
            elif a < 0 and b < 0:
                first, second = q[2]+q[3], q[1]+q[3]
                if phase_flip[sample, site] < 0 and has_phase_frame:
                    first = second = (first+second)/2.
                for slot, alt in enumerate((first, second)):
                    confidence = max(alt, 1-alt)
                    if confidence >= threshold:
                        calls[sample, site, slot] = int(alt >= .5)
                        support[sample, site, slot] = confidence
    return calls, support


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


LOG_FLOOR = 1e-30


def initialise_messages(samples,sites,parents,group_edge,warm_start=None):
    groups=len(group_edge)
    if warm_start is None:
        return JointFamilyMessages(0,np.full((len(parents),sites,4),-math.log(4),dtype=np.float32),
            np.full((groups,sites,4),-math.log(4),dtype=np.float32),
            np.full((samples,sites),.999,dtype=np.float32),
            np.full((groups,sites,4),.25,dtype=np.float32))
    if isinstance(warm_start,JointFamilyMessages):
        return replace(warm_start,iteration=0,deltas=(),
            parent_log=warm_start.parent_log.copy(),child_log=warm_start.child_log.copy(),
            phase_match=warm_start.phase_match.copy(),segregation_match=warm_start.segregation_match.copy(),
            branch_clusters=deepcopy(warm_start.branch_clusters),
            warm_start_iterations=warm_start.iteration+warm_start.warm_start_iterations)
    # A warm start changes initialization, not observations. Preserve the old
    # checkpoint and reset the iteration counter for the new factor grouping.
    child=np.zeros((groups,sites,4),dtype=np.float64)
    segregation=np.ones((groups,sites,4),dtype=np.float64)
    for group in range(groups):
        for slot in range(2):
            edge=group_edge[group,slot]
            if edge<0:
                segregation[group]*=.5
                continue
            p=warm_start.segregation_match[edge]
            for state in range(4):
                bit=(state>>(1-slot))&1
                child[group,:,state]+=warm_start.child_log[edge,:,bit]
                segregation[group,:,state]*=p if bit==0 else 1-p
    maximum=child.max(axis=2,keepdims=True)
    child-=maximum+np.log(np.exp(child-maximum).sum(axis=2,keepdims=True))
    return JointFamilyMessages(0,warm_start.parent_log.copy(),child.astype(np.float32),
        warm_start.phase_match.copy(),segregation.astype(np.float32),(),warm_start.iteration)


@dataclass(frozen=True)
class FamilyRefinementConfig:
    condition_on_scaffold_genotypes: bool = True
    impute_missing: bool = False
    scaffold_scope: str = "roots"
    correct_incomplete_parent_phase: bool = False
    root_gauge_moves: bool = True
    gauge_move_interval: int = 5
    recombination_rate: float = 5e-8
    phase_switch_probability: float = 0.005
    scaffold_phase_error: float = 0.01
    transmission_error: float = 0.001
    damping: float = 0.5
    max_iterations: int = 500
    tolerance: float = 1e-4
    allele_call_probability: float = 0.98
    phase_call_probability: float = 0.98
    checkpoint_every: int = 5
    branch_cluster_after: int = 80
    branch_cluster_interval: int = 20

    def validated(self):
        if self.scaffold_scope not in ("roots", "incomplete", "all"):
            raise ValueError("scaffold_scope must be roots, incomplete, or all")
        if not isinstance(self.impute_missing, bool):
            raise ValueError("impute_missing must be boolean")
        for name in ("max_iterations", "checkpoint_every", "gauge_move_interval", "branch_cluster_after", "branch_cluster_interval"):
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
        for name in ("allele_call_probability", "phase_call_probability"):
            if not 0.5 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie strictly between 0.5 and 1")
        return self


def refine_family(genotype_likelihoods,observed,reference_alleles,positions,phase_bins,
                  component_ids,relationships,sample_ids,*,config=FamilyRefinementConfig(),
                  resume=None,warm_start=None,checkpoint_callback=None,chromosome_map=None):
    started=time.perf_counter();cfg=config.validated()
    if cfg.scaffold_scope!="roots":raise ValueError("family refinement requires root-only scaffold phase anchors")
    if resume is not None and warm_start is not None:raise ValueError("choose resume or warm_start")
    gl=np.asarray(genotype_likelihoods,dtype=np.float64)
    obs=np.asarray(observed,dtype=bool);reference=np.asarray(reference_alleles,dtype=np.int8)
    pos=np.asarray(positions,dtype=np.int64);bins=np.asarray(phase_bins);components=np.asarray(component_ids)
    samples,sites=gl.shape[:2]
    if gl.shape!=(samples,sites,3) or sites==0 or samples!=len(sample_ids):raise ValueError("invalid GL axes")
    if obs.shape!=(samples,sites) or reference.shape!=(samples,sites,2):raise ValueError("invalid scaffold axes")
    if pos.shape!=(sites,) or bins.shape!=pos.shape or components.shape!=pos.shape or np.any(np.diff(pos)<=0):
        raise ValueError("invalid marker coordinates")
    if np.any(~np.isfinite(gl)) or np.any(gl<0) or np.any(~np.isin(reference,(-1,0,1))):raise ValueError("invalid allele evidence")
    total=gl.sum(axis=2,keepdims=True);normalised=np.full_like(gl,1/3)
    np.divide(gl,total,out=normalised,where=total>0);normalised[~obs]=1/3
    parents,children,slots=pedigree_edges(relationships,sample_ids)
    offsets,adjacency=_adjacency(samples,parents,children)
    gp,ge,gc,node_group,edge_group=grouped_edges(parents,children,slots,samples)
    parent_counts=np.bincount(children,minlength=samples);roots=parent_counts==0
    base=np.log(np.maximum(normalised[:,:,(0,1,1,2)],LOG_FLOOR)).astype(np.float32,order="C")
    if cfg.condition_on_scaffold_genotypes:
        k0,k1=reference[:,:,0]>=0,reference[:,:,1]>=0;dosage=reference.sum(axis=2)
        for state,count in enumerate((0,1,1,2)):
            allowed=np.ones((samples,sites),dtype=bool)
            allowed[k0&k1]=dosage[k0&k1]==count
            for slot,partial in ((0,k0&~k1),(1,~k0&k1)):
                complement=count-reference[:,:,slot][partial]
                allowed[partial]=(complement>=0)&(complement<=1)
            base[:,:,state][~allowed]=-np.inf
    base[roots]+=np.log(np.asarray([1/3,1/6,1/6,1/3],dtype=np.float32))
    orientation=np.full((samples,sites),-1,dtype=np.int8)
    het=np.all(reference>=0,axis=2)&(reference[:,:,0]!=reference[:,:,1])
    orientation[het]=reference[:,:,0][het];reference_orientation=orientation.copy()
    orientation[~roots]=-1
    phase_theta=np.zeros(max(0,sites-1),dtype=np.float64)
    phase_theta[bins[1:]!=bins[:-1]]=cfg.phase_switch_probability
    phase_theta[components[1:]!=components[:-1]]=.5
    if chromosome_map is None or not chromosome_map.has_map:
        rate=cfg.recombination_rate if chromosome_map is None else chromosome_map.fallback_rate_per_bp
        theta=-.5*np.expm1(-2*rate*np.diff(pos))
    else:
        theta=-.5*np.expm1(-2*chromosome_map.interval_morgans(pos[:-1],pos[1:]))
    anchors=np.full(samples,-1,dtype=np.int64)
    for sample in np.flatnonzero(roots):
        informative=np.flatnonzero(het[sample])
        if len(informative):anchors[sample]=informative[0]
    del gl,normalised
    messages=resume or initialise_messages(samples,sites,parents,ge,warm_start)
    if (messages.parent_log.shape!=(len(parents),sites,4) or
        messages.child_log.shape!=(len(gc),sites,4) or messages.segregation_match.shape!=(len(gc),sites,4)):
        raise ValueError("joint message axes differ from pedigree/site axes")
    def beliefs():
        return refinement_messages.joint_beliefs(base,orientation,messages.phase_match,messages.parent_log,
            messages.child_log,node_group,offsets,adjacency,children,cfg.scaffold_phase_error)
    belief=beliefs();converged=bool(messages.deltas and messages.deltas[-1]<cfg.tolerance)
    while messages.iteration<cfg.max_iterations and not converged:
        _,phase_delta,_,_=_phase_update(belief,orientation,messages.phase_match,phase_theta,anchors,
                                       cfg.scaffold_phase_error,cfg.damping,True)
        belief=beliefs()
        emissions,copy_delta=refinement_messages.copy_messages(belief,messages,gp,ge,gc,
            cfg.transmission_error,cfg.damping,messages.iteration%2==1)
        _,_,seg_delta=refinement_messages.chain_messages(emissions,messages,theta,cfg.damping,False)
        change=max(float(np.max(phase_delta)),float(np.max(copy_delta)),
                   float(np.max(seg_delta)) if len(seg_delta) else 0.)
        messages.iteration+=1;messages.deltas=(*messages.deltas,change);converged=change<cfg.tolerance
        belief=beliefs()
        gauge_changed=False
        if cfg.root_gauge_moves and messages.iteration%cfg.gauge_move_interval==0:
            maps=refinement_messages.selector_maps(emissions,messages,theta)
            flips,_=refinement_messages.root_mode_flips(belief,orientation,messages.phase_match,parents,edge_group,slots,
                roots,anchors,phase_theta,theta,maps,cfg.scaffold_phase_error)
            if np.any(flips):
                refinement_messages.apply_root_flips(flips,messages,parents,gp)
                gauge_changed=True
                converged=False;messages.deltas=(*messages.deltas[:-1],max(1.,change));belief=beliefs()
        if (not converged and not gauge_changed and messages.iteration>=cfg.branch_cluster_after
                and messages.iteration%cfg.branch_cluster_interval==0):
            if refinement_messages.add_unstable_clusters(messages,gp,ge,gc,node_group,seg_delta,cfg.tolerance):
                messages.deltas=(*messages.deltas[:-1],max(1.,change))
                belief=beliefs()
        if checkpoint_callback is not None and (converged or messages.iteration%cfg.checkpoint_every==0
                                                or messages.iteration==cfg.max_iterations):
            checkpoint_callback(messages)
    alignment=messages.phase_match.copy();alignment[~roots]=.5
    phase,_,phase_map,phase_links=_phase_update(belief,reference_orientation,alignment,phase_theta,
        anchors,cfg.scaffold_phase_error,cfg.damping,False)
    emissions,_=refinement_messages.copy_messages(belief,messages,gp,ge,gc,cfg.transmission_error,0.,False)
    joint,cross,_=refinement_messages.chain_messages(emissions,messages,theta,0.,True)
    segregation=np.empty((len(parents),sites),dtype=np.float32)
    recombinations=np.empty((len(parents),max(0,sites-1)),dtype=np.float32)
    for edge,slot in enumerate(slots):
        group=edge_group[edge]
        segregation[edge]=joint[group,:,0]+joint[group,:,1 if slot==0 else 2]
        recombinations[edge]=cross[group,:,slot]
    posterior=np.exp(belief)
    alt=np.stack((posterior[:,:,2]+posterior[:,:,3],posterior[:,:,1]+posterior[:,:,3]),axis=2)
    support=np.maximum(alt,1-alt)
    calls=np.where(support>=cfg.allele_call_probability,(alt>=.5).astype(np.int8),-1)
    flip=np.full((samples,sites),-1,dtype=np.int8)
    flip[(phase_map==0)&(phase>=cfg.phase_call_probability)]=0
    flip[(phase_map==1)&(phase<=1-cfg.phase_call_probability)]=1
    inferred_map=phase_map.copy()
    if not cfg.correct_incomplete_parent_phase:phase_map[parent_counts<2]=0
    corrected=np.where(phase_map[:,:,None],reference[:,:,::-1],reference).copy()
    completion=posterior.copy();reverse=inferred_map!=phase_map
    completion[:,:,1]=np.where(reverse,posterior[:,:,2],posterior[:,:,1])
    completion[:,:,2]=np.where(reverse,posterior[:,:,1],posterior[:,:,2])
    provenance=np.zeros(reference.shape,dtype=np.uint8)
    if cfg.impute_missing:
        if cfg.condition_on_scaffold_genotypes:
            calls,support=complete_scaffold(corrected,flip,completion,cfg.allele_call_probability)
        provenance[calls>=0]=1;provenance[(calls>=0)&(corrected<0)]=2
        provenance[(calls>=0)&(corrected>=0)&(calls!=corrected)]=3
    else:
        calls=corrected.copy();provenance[calls>=0]=1
    if not cfg.impute_missing and cfg.condition_on_scaffold_genotypes:
        support=(calls>=0).astype(np.float32)
    elif not cfg.condition_on_scaffold_genotypes:
        emitted_alt=np.stack((completion[:,:,2]+completion[:,:,3],completion[:,:,1]+completion[:,:,3]),axis=2)
        support=np.where(calls<0,0.,np.where(calls==1,emitted_alt,1-emitted_alt))
    return FamilyRefinementResult(posterior,phase,segregation,recombinations,calls,support.astype(np.float32),
        flip,inferred_map,phase_map,phase_links,corrected,provenance,messages,converged,time.perf_counter()-started)


@dataclass(frozen=True)
class FamilyRefinementResult:
    ordered_genotype_probability: np.ndarray  # samples, sites, 00/01/10/11
    phase_zero_probability: np.ndarray
    transmission_zero_probability: np.ndarray
    recombination_probability: np.ndarray
    allele_calls: np.ndarray                 # samples, sites, tracks; -1/0/1
    allele_call_support: np.ndarray
    phase_flip: np.ndarray                   # -1 uncertain, otherwise 0/1
    inferred_phase_map: np.ndarray           # internal family-frame alignment
    phase_map: np.ndarray                    # coherent path; confidence is separate
    phase_link_support: np.ndarray           # probability of chosen adjacent phase relation
    corrected_reference_alleles: np.ndarray
    provenance: np.ndarray                   # 0 missing, 1 retained, 2 filled, 3 revised
    messages: object                         # solver checkpoint state
    converged: bool
    elapsed_seconds: float


@njit(cache=True)
def binary_chain_messages(emission, switch_probability, anchor=-1):
    """Scaled binary forward/backward: marginals, site cavities and switches.

    The cavity omits the current emission, NOT the root's arbitrary phase
    gauge. At a structural phase boundary theta=.5 severs that phase chain.
    """
    length = emission.shape[0]
    alpha = np.empty((length, 2), dtype=np.float64)
    predicted = np.empty((length, 2), dtype=np.float64)
    beta = np.ones((length, 2), dtype=np.float64)
    for site in range(length):
        if site == 0:
            p0, p1 = 0.5, 0.5
        else:
            theta = switch_probability[site - 1]
            p0 = alpha[site - 1, 0] * (1-theta) + alpha[site - 1, 1] * theta
            p1 = alpha[site - 1, 1] * (1-theta) + alpha[site - 1, 0] * theta
        if site == anchor:
            p1 = 0.0
        predicted[site, 0], predicted[site, 1] = p0, p1
        a0, a1 = p0 * emission[site, 0], p1 * emission[site, 1]
        total = a0 + a1
        alpha[site, 0], alpha[site, 1] = a0 / total, a1 / total
    for site in range(length - 2, -1, -1):
        e0 = emission[site + 1, 0] * beta[site + 1, 0]
        e1 = emission[site + 1, 1] * beta[site + 1, 1]
        if site + 1 == anchor:
            e1 = 0.0
        theta = switch_probability[site]
        b0, b1 = (1-theta)*e0 + theta*e1, theta*e0 + (1-theta)*e1
        total = b0+b1
        beta[site, 0], beta[site, 1] = b0/total, b1/total
    posterior = np.empty_like(alpha)
    cavity = np.empty_like(alpha)
    switches = np.empty(max(0, length - 1), dtype=np.float64)
    for site in range(length):
        c0 = predicted[site, 0] * beta[site, 0]
        c1 = predicted[site, 1] * beta[site, 1]
        total = c0+c1
        cavity[site, 0], cavity[site, 1] = c0/total, c1/total
        p0, p1 = c0*emission[site, 0], c1*emission[site, 1]
        posterior[site, 0], posterior[site, 1] = p0/(p0+p1), p1/(p0+p1)
        if site + 1 < length:
            e0 = emission[site + 1, 0] * beta[site + 1, 0]
            e1 = emission[site + 1, 1] * beta[site + 1, 1]
            if site + 1 == anchor:
                e1 = 0.0
            theta = switch_probability[site]
            jump = theta*(alpha[site, 0]*e1 + alpha[site, 1]*e0)
            stay = (1-theta)*(alpha[site, 0]*e0 + alpha[site, 1]*e1)
            switches[site] = jump/(jump+stay)
    return posterior, cavity, switches


@njit(cache=True, inline="always")
def _phase_likelihood(state, orientation, phase_zero, error):
    if orientation < 0 or state == 0 or state == 3:
        return 0.5
    matching_state = 1 if orientation == 0 else 2
    p = error + (1-2*error)*phase_zero
    return p if state == matching_state else 1-p


@njit(cache=True, parallel=True)
def _phase_update(belief_log, orientation, phase, phase_theta, anchors,
                  phase_error, damping, write_messages):
    samples, sites, _ = belief_log.shape
    probability = np.empty((0,0) if write_messages else (samples,sites), dtype=np.float32)
    delta = np.zeros(samples, dtype=np.float64)
    paths = np.empty((0,0) if write_messages else (samples,sites), dtype=np.int8)
    links = np.empty((0,0) if write_messages else (samples,max(0,sites-1)), dtype=np.float32)
    for sample in prange(samples):
        emissions = np.empty((sites, 2), dtype=np.float64)
        for site in range(sites):
            q = np.empty(4, dtype=np.float64)
            for state in range(4):
                q[state] = math.exp(float(belief_log[sample, site, state])) / _phase_likelihood(
                    state, orientation[sample, site], phase[sample, site], phase_error)
            q /= np.sum(q)
            for flip in range(2):
                total = 0.0
                for state in range(4):
                    total += q[state]*_phase_likelihood(state, orientation[sample, site],
                                                       1.0-float(flip), phase_error)
                emissions[site, flip] = total
        posterior, cavity, switches = binary_chain_messages(emissions, phase_theta, anchors[sample])
        if not write_messages:
            paths[sample] = refinement_messages.binary_map(emissions, phase_theta, anchors[sample])
            for site in range(sites-1):
                links[sample,site] = (switches[site] if paths[sample,site]!=paths[sample,site+1]
                                      else 1-switches[site])
        for site in range(sites):
            if not write_messages:
                probability[sample, site] = posterior[site, 0]
            if write_messages:
                old = phase[sample, site]
                new = (1-damping)*old + damping*cavity[site, 0]
                delta[sample] = max(delta[sample], abs(new-old))
                phase[sample, site] = new
    return probability, delta, paths, links


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


import haplotype_reconstruction.refinement.messages as refinement_messages
