"""refinement / messages for the canonical reconstruction pipeline."""
from __future__ import annotations


import numpy as np
from numba import njit
import math
from dataclasses import dataclass
from numba import prange


@njit(cache=True)
def branch_factor(q, selector, intermediate_slot, error):
    """Return selector emissions and the five factor-to-genotype messages.

q[node, ordered genotype] excludes this entire cluster. selector[16] excludes
this site's factor. State 4*sA+sB joins the two parental-selector pairs.
"""
    alt = np.empty((3, 2))
    for node in range(3):
        alt[node, 0] = q[node, 2]+q[node, 3]
        alt[node, 1] = q[node, 1]+q[node, 3]
    ca = np.zeros((4, 4))
    ca0 = np.zeros((4, 4, 4))
    ca1 = np.zeros((4, 4, 4))
    cb = np.zeros((4, 4, 4))
    lb = np.zeros((4, 4))
    lp = np.zeros((4, 4, 4))
    for state in range(4):
        s0, s1 = state >> 1, state & 1
        a = error+(1-2*error)*alt[0, s0]
        b = error+(1-2*error)*alt[1, s1]
        sa = s0 if intermediate_slot == 0 else s1
        sp = s1 if intermediate_slot == 0 else s0
        p = error+(1-2*error)*alt[2, sp]
        for genotype in range(4):
            bit0, bit1 = genotype >> 1, genotype & 1
            ca[state, genotype] = (a if bit0 else 1-a)*(b if bit1 else 1-b)
            for parent in range(4):
                x = error+(1-2*error)*((parent >> (1-s0)) & 1)
                y = error+(1-2*error)*((parent >> (1-s1)) & 1)
                ca0[state, genotype, parent] = (x if bit0 else 1-x)*(b if bit1 else 1-b)
                ca1[state, genotype, parent] = (a if bit0 else 1-a)*(y if bit1 else 1-y)
            inherited = error+(1-2*error)*((genotype >> (1-sa)) & 1)
            for child in range(4):
                child_a = (child >> (1-intermediate_slot)) & 1
                child_p = (child >> intermediate_slot) & 1
                own = inherited if child_a else 1-inherited
                chance = own*(p if child_p else 1-p)
                cb[state, genotype, child] = chance
                lb[state, genotype] += q[4, child]*chance
                for parent in range(4):
                    other = error+(1-2*error)*((parent >> (1-sp)) & 1)
                    lp[state, genotype, parent] += q[4, child]*own*(other if child_p else 1-other)
    emission = np.zeros(16)
    message = np.zeros((5, 4))
    for state in range(16):
        a, b = state >> 2, state & 3
        weight = selector[state]
        for genotype in range(4):
            upstream = ca[a, genotype]
            downstream = lb[b, genotype]
            emission[state] += q[3, genotype]*upstream*downstream
            message[3, genotype] += weight*upstream*downstream
            for other in range(4):
                mass = weight*q[3, genotype]
                message[0, other] += mass*ca0[a, genotype, other]*downstream
                message[1, other] += mass*ca1[a, genotype, other]*downstream
                message[2, other] += mass*upstream*lp[b, genotype, other]
                message[4, other] += mass*upstream*cb[b, genotype, other]
    for node in range(5):
        message[node] /= np.sum(message[node])
    return emission, message


@njit(cache=True)
def mix_except(row, theta, omitted):
    result = row.copy()
    for bit in (1, 2, 4, 8):
        if bit == omitted:
            continue
        for left in range(16):
            if left & bit:
                continue
            right = left | bit
            a, b = result[left], result[right]
            result[left] = (1-theta)*a+theta*b
            result[right] = (1-theta)*b+theta*a
    return result


@dataclass
class BranchCluster:
    groups: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    intermediate_slot: int
    cavity: np.ndarray
    emissions: np.ndarray | None = None


@njit(cache=True,inline="always")
def mix4(a,b,c,d,t):
    u=1-t
    x0,x1,x2,x3=u*a+t*c,u*b+t*d,u*c+t*a,u*d+t*b
    return u*x0+t*x1,u*x1+t*x0,u*x2+t*x3,u*x3+t*x2


@njit(cache=True,inline="always")
def _normalise_log(row):
    maximum=np.max(row)
    denominator=maximum+math.log(np.sum(np.exp(row-maximum)))
    for state in range(4):row[state]-=denominator


@njit(cache=True,parallel=True)
def root_mode_flips(belief,orientation,phase,parents,edge_group,edge_slot,
                    roots,anchors,phase_theta,theta,joint_maps,phase_error):
    samples,sites,_=belief.shape
    flips=np.zeros((samples,sites),dtype=np.int8);gains=np.zeros(samples)
    for root in prange(samples):
        if not roots[root]:continue
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


@njit(cache=True)
def binary_map(emission,theta,anchor=-1):
    n=len(emission)
    scores=np.zeros(2,dtype=np.float64)
    back=np.zeros((n,2),dtype=np.int8)
    for i in range(n):
        new=np.empty(2,dtype=np.float64)
        for state in range(2):
            if i==anchor and state==1:
                new[state]=-np.inf;continue
            if i==0:
                new[state]=math.log(max(emission[i,state],1e-300))-math.log(2.)
                continue
            stay=math.log(max(1-theta[i-1],1e-300))
            jump=math.log(theta[i-1]) if theta[i-1]>0 else -np.inf
            a=scores[state]+stay;b=scores[1-state]+jump
            if a>=b:new[state]=a;back[i,state]=state
            else:new[state]=b;back[i,state]=1-state
            new[state]+=math.log(max(emission[i,state],1e-300))
        scores=new
    path=np.empty(n,dtype=np.int8)
    state=0 if scores[0]>=scores[1] else 1
    for i in range(n-1,-1,-1):
        path[i]=state
        if i:state=back[i,state]
    return path


@njit(cache=True)
def hypercube_mix(row, theta):
    """Independent binary transitions in O(S log S), here S=16."""
    result = row.copy()
    bit = 1
    while bit < len(row):
        for left in range(len(row)):
            if left & bit:
                continue
            right = left | bit
            a, b = result[left], result[right]
            result[left] = (1-theta)*a+theta*b
            result[right] = (1-theta)*b+theta*a
        bit *= 2
    return result


@njit(cache=True)
def branch_chain_messages(emission, theta, diagnostics=True):
    sites = len(emission)
    alpha = np.empty((sites,16))
    predicted = np.empty((sites,16))
    beta = np.ones((sites,16))
    for site in range(sites):
        predicted[site] = np.full(16,1/16) if site == 0 else hypercube_mix(alpha[site-1],theta[site-1])
        alpha[site] = predicted[site]*emission[site]
        alpha[site] /= np.sum(alpha[site])
    for site in range(sites-2,-1,-1):
        beta[site] = hypercube_mix(emission[site+1]*beta[site+1],theta[site])
        beta[site] /= np.sum(beta[site])
    cavity = predicted*beta
    posterior = np.empty((sites,16) if diagnostics else (0,0))
    switches = np.empty((max(0,sites-1),4) if diagnostics else (0,0))
    for site in range(sites):
        cavity[site] /= np.sum(cavity[site])
        if diagnostics:
            posterior[site] = cavity[site]*emission[site]
            posterior[site] /= np.sum(posterior[site])
            if site+1 < sites:
                following = emission[site+1]*beta[site+1]
                denominator = np.sum(hypercube_mix(alpha[site],theta[site])*following)
                for slot in range(4):
                    bit = 1 << (3-slot)
                    other = mix_except(alpha[site],theta[site],bit)
                    numerator = 0.
                    for state in range(16):
                        numerator += theta[site]*other[state^bit]*following[state]
                    switches[site,slot] = numerator/denominator
    return posterior,cavity,switches


def unclustered(groups, clusters):
    occupied = {int(g) for cluster in clusters for g in cluster.groups}
    return np.asarray([g for g in range(groups) if g not in occupied],dtype=np.int64)


@njit(cache=True)
def paired_chain_messages(emission,theta):
    n=len(emission)
    alpha=np.empty((n,4),dtype=np.float64)
    predicted=np.empty((n,4),dtype=np.float64)
    beta=np.ones((n,4),dtype=np.float64)
    for site in range(n):
        if site==0:p=(.25,.25,.25,.25)
        else:p=mix4(alpha[site-1,0],alpha[site-1,1],alpha[site-1,2],alpha[site-1,3],theta[site-1])
        total=0.
        for state in range(4):
            predicted[site,state]=p[state]
            alpha[site,state]=p[state]*emission[site,state]
            total+=alpha[site,state]
        for state in range(4):alpha[site,state]/=total
    for site in range(n-2,-1,-1):
        values=mix4(emission[site+1,0]*beta[site+1,0],emission[site+1,1]*beta[site+1,1],
                    emission[site+1,2]*beta[site+1,2],emission[site+1,3]*beta[site+1,3],theta[site])
        total=sum(values)
        for state in range(4):beta[site,state]=values[state]/total
    posterior=np.empty_like(alpha);cavity=np.empty_like(alpha)
    switches=np.empty((max(0,n-1),2),dtype=np.float64)
    for site in range(n):
        total=0.;post_total=0.
        for state in range(4):
            cavity[site,state]=predicted[site,state]*beta[site,state]
            posterior[site,state]=cavity[site,state]*emission[site,state]
            total+=cavity[site,state];post_total+=posterior[site,state]
        for state in range(4):
            cavity[site,state]/=total;posterior[site,state]/=post_total
        if site+1<n:
            t=theta[site];u=1-t
            p=mix4(alpha[site,0],alpha[site,1],alpha[site,2],alpha[site,3],t)
            denominator=0.;jump0=0.;jump1=0.
            for state in range(4):
                value=emission[site+1,state]*beta[site+1,state]
                denominator+=p[state]*value
                jump0+=t*(u*alpha[site,state^2]+t*alpha[site,state^3])*value
                jump1+=t*(u*alpha[site,state^1]+t*alpha[site,state^3])*value
            switches[site,0]=jump0/denominator;switches[site,1]=jump1/denominator
    return posterior,cavity,switches


@njit(cache=True,inline="always")
def probabilities(log_values):
    q=np.exp(log_values-np.max(log_values))
    q/=np.sum(q)
    return q


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


@njit(cache=True,parallel=True)
def update_joint_chains(emissions,current,theta,damping,diagnostics=True):
    groups,sites,_=emissions.shape
    posterior=np.empty((groups,sites,4) if diagnostics else (0,0,0),dtype=np.float32)
    switches=np.empty((groups,max(0,sites-1),2) if diagnostics else (0,0,0),dtype=np.float32)
    delta=np.zeros(groups,dtype=np.float64)
    for group in prange(groups):
        q,cavity,cross=paired_chain_messages(emissions[group],theta)
        for site in range(sites):
            for state in range(4):
                old=current[group,site,state]
                new=(1-damping)*old+damping*cavity[site,state]
                current[group,site,state]=new
                delta[group]=max(delta[group],abs(new-old))
                if diagnostics:posterior[group,site,state]=q[site,state]
            if diagnostics and site+1<sites:
                switches[group,site,0]=cross[site,0];switches[group,site,1]=cross[site,1]
    return posterior,switches,delta


@njit(cache=True,parallel=True)
def joint_beliefs(base,orientation,phase,parent_log,child_log,node_group,
                  offsets,adjacency,edge_child,phase_error):
    samples,sites,_=base.shape;result=np.empty_like(base)
    tiles=(sites+255)//256
    for task in prange(samples*tiles):
        sample=task % samples;start=(task//samples)*256;group=node_group[sample]
        for site in range(start,min(sites,start+256)):
            row=np.empty(4,dtype=np.float64)
            for state in range(4):
                value=float(base[sample,site,state])+math.log(refinement_model._phase_likelihood(
                    state,orientation[sample,site],phase[sample,site],phase_error))
                if group>=0:value+=child_log[group,site,state]
                for index in range(offsets[sample],offsets[sample+1]):
                    edge=adjacency[index]
                    if edge_child[edge]!=sample:value+=parent_log[edge,site,state]
                row[state]=value
            maximum=np.max(row);normalizer=maximum+math.log(np.sum(np.exp(row-maximum)))
            for state in range(4):result[sample,site,state]=row[state]-normalizer
    return result


@njit(cache=True,parallel=True)
def branch_sweep(belief,parent_log,child_log,cavity,nodes,edges,groups,slot,error,damping):
    sites = belief.shape[1]
    emissions = np.empty((sites,16),dtype=np.float32)
    delta = np.zeros(sites)
    for site in prange(sites):
        q = np.empty((5,4))
        for node in range(5):
            message = parent_log[edges[node],site] if node < 3 else child_log[groups[node-3],site]
            q[node] = probabilities(belief[nodes[node],site].astype(np.float64)-message)
        emission,messages = branch_factor(q,cavity[site],slot,error)
        emissions[site] = emission
        if damping == 0:
            continue
        for node in range(5):
            old_log = parent_log[edges[node],site] if node < 3 else child_log[groups[node-3],site]
            for genotype in range(4):
                old = np.exp(old_log[genotype])
                probability = (1-damping)*old+damping*messages[node,genotype]
                value = np.log(max(1e-30,probability))
                belief[nodes[node],site,genotype] += value-old_log[genotype]
                old_log[genotype] = value
                delta[site] = max(delta[site],abs(probability-old))
            _normalise_log(belief[nodes[node],site])
    return emissions,np.max(delta)


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


@njit(cache=True,parallel=True)
def joint_copy_sweep(belief,parent_log,child_log,segregation,group_parent,
                     group_edge,group_child,error,damping,reverse=False):
    groups,sites,_=child_log.shape
    emissions=np.empty((groups,sites,4),dtype=np.float32)
    tiles=(sites+31)//32;delta=np.zeros(tiles,dtype=np.float64)
    for tile in prange(tiles):
        lo,hi=tile*32,min(sites,(tile+1)*32)
        for step in range(groups):
            group=groups-1-step if reverse else step
            child=group_child[group]
            for site in range(lo,hi):
                qchild=probabilities(belief[child,site].astype(np.float64)-child_log[group,site])
                parent_alt=np.full((2,2),.5,dtype=np.float64)
                for slot in range(2):
                    parent=group_parent[group,slot];edge=group_edge[group,slot]
                    if parent<0:continue
                    q=probabilities(belief[parent,site].astype(np.float64)-parent_log[edge,site])
                    parent_alt[slot,0]=q[2]+q[3];parent_alt[slot,1]=q[1]+q[3]
                pm=np.zeros((2,4),dtype=np.float64);cm=np.zeros(4,dtype=np.float64)
                for state in range(4):
                    s0,s1=state>>1,state&1
                    a=error+(1-2*error)*parent_alt[0,s0]
                    b=error+(1-2*error)*parent_alt[1,s1]
                    chance=np.array([(1-a)*(1-b),(1-a)*b,a*(1-b),a*b])
                    total=0.
                    for genotype in range(4):total+=qchild[genotype]*chance[genotype]
                    emissions[group,site,state]=total
                    if damping==0:continue
                    weight=segregation[group,site,state]
                    for genotype in range(4):cm[genotype]+=weight*chance[genotype]
                    u0=qchild[0]*(1-b)+qchild[1]*b;u1=qchild[2]*(1-b)+qchild[3]*b
                    v0=qchild[0]*(1-a)+qchild[2]*a;v1=qchild[1]*(1-a)+qchild[3]*a
                    for genotype in range(4):
                        allele0=(genotype>>(1-s0))&1;allele1=(genotype>>(1-s1))&1
                        x=error+(1-2*error)*allele0;y=error+(1-2*error)*allele1
                        pm[0,genotype]+=weight*(u0*(1-x)+u1*x)
                        pm[1,genotype]+=weight*(v0*(1-y)+v1*y)
                if damping==0:continue
                cm/=np.sum(cm)
                for genotype in range(4):
                    old_log=float(child_log[group,site,genotype]);old=math.exp(old_log)
                    updated=(1-damping)*old+damping*cm[genotype]
                    value=math.log(max(1e-30,updated))
                    child_log[group,site,genotype]=value
                    belief[child,site,genotype]+=value-old_log
                    delta[tile]=max(delta[tile],abs(updated-old))
                _normalise_log(belief[child,site])
                for slot in range(2):
                    parent=group_parent[group,slot];edge=group_edge[group,slot]
                    if parent<0:continue
                    denominator=np.sum(pm[slot])
                    for genotype in range(4):
                        old_log=float(parent_log[edge,site,genotype]);old=math.exp(old_log)
                        updated=(1-damping)*old+damping*pm[slot,genotype]/denominator
                        value=math.log(max(1e-30,updated))
                        parent_log[edge,site,genotype]=value
                        belief[parent,site,genotype]+=value-old_log
                        delta[tile]=max(delta[tile],abs(updated-old))
                    _normalise_log(belief[parent,site])
    return emissions,delta


def copy_messages(belief,messages,gp,ge,gc,error,damping,reverse):
    clusters = messages.branch_clusters
    if not clusters:
        return joint_copy_sweep(belief,messages.parent_log,messages.child_log,
            messages.segregation_match,gp,ge,gc,error,damping,reverse)
    keep = unclustered(len(gc),clusters)
    local = messages.child_log[keep].copy()
    ordinary,delta = joint_copy_sweep(belief,messages.parent_log,local,
        messages.segregation_match[keep],gp[keep],ge[keep],gc[keep],error,damping,reverse)
    messages.child_log[keep] = local
    emissions = np.ones_like(messages.segregation_match)
    emissions[keep] = ordinary
    maximum = float(delta.max())
    for cluster in clusters:
        cluster.emissions,change = branch_sweep(belief,messages.parent_log,messages.child_log,
            cluster.cavity,cluster.nodes,cluster.edges,cluster.groups,cluster.intermediate_slot,error,damping)
        maximum = max(maximum,change)
    return emissions,np.asarray([maximum])


@njit(cache=True,parallel=True)
def joint_selector_maps(emissions,theta):
    result=np.empty(emissions.shape[:2],dtype=np.int8)
    for group in prange(len(emissions)):result[group]=paired_map(emissions[group],theta)
    return result


def chain_messages(emissions,messages,theta,damping,diagnostics):
    clusters = messages.branch_clusters
    current = messages.segregation_match
    if not clusters:
        return update_joint_chains(emissions,current,theta,damping,diagnostics)
    keep = unclustered(len(current),clusters)
    local = current[keep].copy()
    ordinary,cross,changes = update_joint_chains(emissions[keep],local,theta,damping,diagnostics)
    current[keep] = local
    delta = np.zeros(len(current))
    delta[keep] = changes
    posterior = np.empty_like(current) if diagnostics else np.empty((0,0,0),dtype=np.float32)
    switches = np.empty((len(current),len(theta),2) if diagnostics else (0,0,0),dtype=np.float32)
    if diagnostics:
        posterior[keep],switches[keep] = ordinary,cross
    for cluster in clusters:
        q,cavity,jumps = branch_chain_messages(cluster.emissions,theta,diagnostics)
        delta[cluster.groups] = damping*np.max(abs(cavity-cluster.cavity))
        if damping:
            cluster.cavity *= 1-damping
            cluster.cavity += damping*cavity
        a,b = cluster.groups
        joint = cluster.cavity.reshape((-1,4,4))
        current[a],current[b] = joint.sum(axis=2),joint.sum(axis=1)
        if diagnostics:
            joint = q.reshape((-1,4,4))
            posterior[a],posterior[b] = joint.sum(axis=2),joint.sum(axis=1)
            switches[a],switches[b] = jumps[:,:2],jumps[:,2:]
    return posterior,switches,delta


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

import haplotype_reconstruction.refinement.model as refinement_model
