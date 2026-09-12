"""Family message scheduling, branch topology and root phase gauges."""


import math
import numpy as np
from numba import njit, prange


from dataclasses import dataclass
from . import factors, selector_chains
from .phase_chains import binary_map


@dataclass
class BranchCluster:
    groups: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    intermediate_slot: int
    cavity: np.ndarray
    emissions: np.ndarray | None = None



@njit(cache=True,parallel=True)
def root_mode_flips(belief,orientation,phase,parents,edge_group,edge_slot,
                    roots,anchors,phase_theta,theta,joint_maps,phase_error):
    samples,sites,_=belief.shape
    flips=np.zeros((samples,sites),dtype=np.int8);gains=np.zeros(samples)
    # Schedule the actual root chains: contiguous root rows otherwise occupy
    # only a few static worker chunks when most samples are offspring.
    active_roots=np.flatnonzero(roots)
    for index in prange(len(active_roots)):
        root=active_roots[index]
        outgoing=np.where(parents==root)[0]
        if len(outgoing)==0:continue
        emissions=np.empty((sites,2),dtype=np.float64)
        for site in range(sites):
            q=np.empty(4,dtype=np.float64);orient=orientation[root,site]
            current=phase_error+(1-2*phase_error)*phase[root,site]
            for state in range(4):
                if orient<0 or state==0 or state==3:factor=.5
                else:factor=current if state==(1 if orient==0 else 2) else 1-current
                q[state]=math.exp(float(belief[root,site,state]))/factor
            q/=np.sum(q)
            if orient<0:
                emissions[site,0]=.5;emissions[site,1]=.5
            else:
                matching=1 if orient==0 else 2;opposite=3-matching
                emissions[site,0]=.5*(q[0]+q[3])+(1-phase_error)*q[matching]+phase_error*q[opposite]
                emissions[site,1]=.5*(q[0]+q[3])+phase_error*q[matching]+(1-phase_error)*q[opposite]
        path=binary_map(emissions,phase_theta,anchors[root])
        child_paths=np.empty((len(outgoing),sites),dtype=np.int8)
        for row in range(len(outgoing)):
            edge=outgoing[row];group=edge_group[edge];shift=1-edge_slot[edge]
            for site in range(sites):child_paths[row,site]=(joint_maps[group,site]>>shift)&1
        local,gain=best_gauge_flips(path,child_paths,phase_theta,theta,anchors[root])
        flips[root]=local;gains[root]=gain
    return flips,gains



def unclustered(groups, clusters):
    occupied = {int(g) for cluster in clusters for g in cluster.groups}
    return np.asarray([g for g in range(groups) if g not in occupied],dtype=np.int64)



@njit(cache=True,parallel=True)
def apply_joint_root_flips(flips,phase,parent_log,joint,parents,group_parent):
    samples,sites=phase.shape
    for sample in prange(samples):
        for site in range(sites):
            if flips[sample,site]:phase[sample,site]=1-phase[sample,site]
    for edge in prange(len(parents)):
        parent=parents[edge]
        for site in range(sites):
            if flips[parent,site]:
                value=parent_log[edge,site,1]
                parent_log[edge,site,1]=parent_log[edge,site,2]
                parent_log[edge,site,2]=value
    # Both roots may parent the same child. Apply their commuting selector-bit
    # permutations once per child, never by concurrent writes from two roots.
    for group in prange(len(group_parent)):
        p0,p1=group_parent[group,0],group_parent[group,1]
        for site in range(sites):
            permutation=(2*flips[p0,site] if p0>=0 else 0)+(flips[p1,site] if p1>=0 else 0)
            if permutation:
                previous=joint[group,site].copy()
                for state in range(4):joint[group,site,state]=previous[state^permutation]



@njit(cache=True)
def best_gauge_flips(phase_path,child_paths,phase_theta,theta,anchor):
    n=len(phase_path)
    flip=np.zeros(n,dtype=np.int8)
    gain=0.
    for i in range(n-1):
        # A zero-probability phase boundary cannot be crossed by this move.
        if phase_theta[i]<=0:
            flip[i+1]=flip[i];continue
        phase_changed=phase_path[i]!=phase_path[i+1]
        phase_odds=math.log(phase_theta[i]/(1-phase_theta[i]))
        improvement=(-1. if phase_changed else 1.)*phase_odds
        meiosis_odds=math.log(theta[i]/(1-theta[i]))
        for child in range(len(child_paths)):
            changed=child_paths[child,i]!=child_paths[child,i+1]
            improvement+=(-1. if changed else 1.)*meiosis_odds
        move=improvement>1e-6
        flip[i+1]=flip[i] ^ int(move)
        if move:gain+=improvement
    gauge=flip[anchor] if anchor>=0 else flip[0]
    for i in range(n):flip[i]^=gauge
    return flip,gain



@njit(cache=True)
def branch_map(emission, theta):
    """Max-product binary-transition butterfly: O(16 log 16) per marker."""
    sites = len(emission)
    score = np.full(16,-math.log(16.))
    back = np.zeros((sites,16),dtype=np.int8)
    for site in range(sites):
        if site:
            t = theta[site-1]
            stay = math.log(1-t)
            jump = math.log(t) if t > 0 else -np.inf
            origin = np.arange(16)
            for bit in (1,2,4,8):
                for left in range(16):
                    if left & bit:
                        continue
                    right = left | bit
                    a,b = score[left],score[right]
                    ia,ib = origin[left],origin[right]
                    if b+jump > a+stay or (b+jump == a+stay and ib < ia):
                        score[left],origin[left] = b+jump,ib
                    else:
                        score[left],origin[left] = a+stay,ia
                    if a+jump > b+stay or (a+jump == b+stay and ia < ib):
                        score[right],origin[right] = a+jump,ia
                    else:
                        score[right],origin[right] = b+stay,ib
            back[site] = origin
        for state in range(16):
            score[state] += math.log(max(1e-300,emission[site,state]))
        score -= np.max(score)
    path = np.empty(sites,dtype=np.int8)
    state = int(np.argmax(score))
    for site in range(sites-1,-1,-1):
        path[site] = state
        if site:
            state = back[site,state]
    return path



def add_unstable_clusters(messages, gp, ge, gc, node_group, delta, tolerance):
    """Greedy residual-ranked matching, using topology and messages, never truth."""
    occupied = {int(g) for cluster in messages.branch_clusters for g in cluster.groups}
    candidates = []
    for downstream in range(len(gc)):
        if downstream in occupied or delta[downstream] <= tolerance:
            continue
        for slot in range(2):
            parent = gp[downstream,slot]
            upstream = node_group[parent] if parent >= 0 else -1
            if upstream < 0 or upstream in occupied or delta[upstream] <= tolerance:
                continue
            nodes = (gp[upstream,0],gp[upstream,1],gp[downstream,1-slot],gc[upstream],gc[downstream])
            # The exact five-node formula assumes no extra shared individual.
            # Missing-parent and inbred branches retain the general solver.
            if min(nodes) < 0 or len(set(nodes)) != 5:
                continue
            candidates.append((-min(delta[upstream],delta[downstream]),upstream,downstream,slot,nodes))
    added = []
    for _,a,b,slot,nodes in sorted(candidates):
        if a in occupied or b in occupied:
            continue
        bridge = ge[b,slot]
        combined = messages.child_log[a].astype(np.float64)+messages.parent_log[bridge]
        maximum = combined.max(axis=1,keepdims=True)
        combined -= maximum+np.log(np.exp(combined-maximum).sum(axis=1,keepdims=True))
        messages.child_log[a] = combined
        messages.parent_log[bridge] = -np.log(4.)
        cavity = (messages.segregation_match[a,:,:,None]*messages.segregation_match[b,:,None,:]).reshape((-1,16)).copy()
        added.append(BranchCluster(np.asarray([a,b]),np.asarray(nodes),
            np.asarray([ge[a,0],ge[a,1],ge[b,1-slot]]),slot,cavity))
        occupied.update((a,b))
    messages.branch_clusters = (*messages.branch_clusters,*added)
    return len(added)



@njit(cache=True)
def paired_map(emission,theta):
    n=len(emission);score=np.zeros(4,dtype=np.float64)
    back=np.zeros((n,4),dtype=np.int8)
    for site in range(n):
        new=np.empty(4,dtype=np.float64)
        for state in range(4):
            if site==0:
                new[state]=math.log(max(1e-300,emission[site,state]))-math.log(4.)
                continue
            best=-np.inf;arg=0;t=theta[site-1]
            for previous in range(4):
                bits=previous^state;changes=(bits>>1)+(bits&1)
                if changes and t==0:continue
                value=score[previous]+(2-changes)*math.log(1-t)
                if changes:value+=changes*math.log(t)
                if value>best:best=value;arg=previous
            new[state]=best+math.log(max(1e-300,emission[site,state]));back[site,state]=arg
        score=new
    result=np.empty(n,dtype=np.int8);state=np.argmax(score)
    for site in range(n-1,-1,-1):
        result[site]=state
        if site:state=back[site,state]
    return result



def copy_messages(belief, messages, gp, ge, gc, error, damping, reverse, factor_support, dirty_tiles=None):
    clusters = messages.branch_clusters
    keep = None if not clusters else unclustered(len(gc), clusters)
    if dirty_tiles is not None:dirty_tiles.fill(False)
    emissions, delta = factors.active_copy_sweep(
        belief, messages.parent_log, messages.child_log, messages.segregation_match,
        gp, ge, gc, error, damping, reverse, keep, factor_support[3], factor_support, dirty_tiles)
    if not clusters:
        return emissions, delta
    maximum = float(delta.max())
    for cluster in clusters:
        # Each clustered factor replaces, rather than duplicates, two ordinary factors.
        emissions[cluster.groups] = 1.
        cluster.emissions, change = factors.branch_sweep(
            belief, messages.parent_log, messages.child_log, cluster.cavity,
            cluster.nodes, cluster.edges, cluster.groups, cluster.intermediate_slot,
            error, damping)
        maximum = max(maximum, change)
    return emissions, np.asarray([maximum])


@njit(cache=True,parallel=True)
def joint_selector_maps(emissions,theta):
    result=np.empty(emissions.shape[:2],dtype=np.int8)
    for group in prange(len(emissions)):result[group]=paired_map(emissions[group],theta)
    return result



def chain_messages(emissions, messages, theta, damping, support, cache, dirty_tiles=None):
    clusters = messages.branch_clusters
    current = messages.segregation_match
    keep = unclustered(len(current), clusters)
    _, _, delta = selector_chains.update_incremental_chains(
        emissions, current, damping, keep, support, cache, dirty_tiles)
    for cluster in clusters:
        key = tuple(map(int, cluster.groups))
        branch_cache = cache.branches.get(key)
        if (branch_cache is None and cache.branch_bytes + len(theta)*578 + 4096
                <= selector_chains.MAX_BRANCH_CACHE_BYTES):
            branch_cache = selector_chains.BranchCache(len(cluster.emissions))
            cache.branches[key] = branch_cache
            cache.branch_bytes += branch_cache.bytes
        cavity = (selector_chains.branch_cavity(cluster.emissions, theta)
                  if branch_cache is None else branch_cache.update(cluster.emissions, theta))
        delta[cluster.groups] = damping*np.max(abs(cavity-cluster.cavity))
        if damping:
            cluster.cavity *= 1-damping
            cluster.cavity += damping*cavity
        a, b = cluster.groups
        joint = cluster.cavity.reshape((-1, 4, 4))
        current[a], current[b] = joint.sum(axis=2), joint.sum(axis=1)
    return delta


def selector_maps(emissions,messages,theta):
    paths = joint_selector_maps(emissions,theta)
    for cluster in messages.branch_clusters:
        path = branch_map(cluster.emissions,theta)
        a,b = cluster.groups
        paths[a],paths[b] = path >> 2,path & 3
    return paths



@njit(cache=True,parallel=True)
def permute_cluster(cavity,flips,gp,groups):
    a,b = groups
    for site in prange(len(cavity)):
        permutation = (8*flips[gp[a,0],site]+4*flips[gp[a,1],site]
                       +2*flips[gp[b,0],site]+flips[gp[b,1],site])
        if permutation:
            previous = cavity[site].copy()
            for state in range(16):
                cavity[site,state] = previous[state^permutation]



def apply_root_flips(flips,messages,parents,gp):
    apply_joint_root_flips(flips,messages.phase_match,messages.parent_log,
        messages.segregation_match,parents,gp)
    for cluster in messages.branch_clusters:
        permute_cluster(cluster.cavity,flips,gp,cluster.groups)
