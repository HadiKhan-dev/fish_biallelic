"""Genotype beliefs and sparse ordinary/coupled family copy factors."""


import math
import numpy as np
from numba import njit, prange


@njit(cache=True,inline="always")
def _normalise_log(row):
    maximum=max(row[0],row[1],row[2],row[3])
    total=((np.exp(row[0]-maximum)+np.exp(row[1]-maximum))
           +np.exp(row[2]-maximum))+np.exp(row[3]-maximum)
    denominator=maximum+math.log(total)
    for state in range(4):row[state]-=denominator



@njit(cache=True,inline="always")
def probabilities(log_values):
    q=np.exp(log_values-np.max(log_values))
    q/=np.sum(q)
    return q



@njit(cache=True,inline="always")
def cavity4(belief, message):
    """Ordered-genotype cavity without temporary arrays.

    Impossible scaffold genotypes stay exactly zero; singleton and two-state
    supports need no separate inferred allele or altered message normalization.
    """
    a=np.float64(belief[0])-np.float64(message[0])
    b=np.float64(belief[1])-np.float64(message[1])
    c=np.float64(belief[2])-np.float64(message[2])
    d=np.float64(belief[3])-np.float64(message[3])
    maximum=max(a,b,c,d)
    a=math.exp(a-maximum) if a>-np.inf else 0.
    b=math.exp(b-maximum) if b>-np.inf else 0.
    c=math.exp(c-maximum) if c>-np.inf else 0.
    d=math.exp(d-maximum) if d>-np.inf else 0.
    total=((a+b)+c)+d
    return a/total,b/total,c/total,d/total



@njit(cache=True,inline="always")
def fixed_homozygote(row):
    return (row[1]==-np.inf and row[2]==-np.inf and
            ((row[0]==-np.inf and row[3]>-np.inf) or (row[3]==-np.inf and row[0]>-np.inf)))



@njit(cache=True,parallel=True)
def fixed_nodes(base):
    n,l,_=base.shape;fixed=np.zeros((n,l),dtype=np.bool_)
    for i in prange(n):
        for j in range(l):
            fixed[i,j]=(base[i,j,1]==-np.inf and base[i,j,2]==-np.inf and
                ((base[i,j,0]>-np.inf and base[i,j,3]==-np.inf) or
                 (base[i,j,3]>-np.inf and base[i,j,0]==-np.inf)))
    return fixed



@njit(cache=True,parallel=True)
def joint_beliefs(base,orientation,phase,parent_log,child_log,node_group,
                 offsets,adjacency,edge_child,phase_error):
    samples,sites,_=base.shape;result=np.empty_like(base)
    tiles=(sites+255)//256
    for task in prange(samples*tiles):
        sample=task%samples;start=(task//samples)*256;group=node_group[sample]
        for site in range(start,min(sites,start+256)):
            # Hard-conditioned homozygotes are exact point masses, whatever
            # the incoming factor-message normalization. No inference skipped.
            if np.isfinite(base[sample,site,0]) and not np.isfinite(base[sample,site,1]) and not np.isfinite(base[sample,site,2]) and not np.isfinite(base[sample,site,3]):
                result[sample,site,0]=0.
                for state in range(1,4):result[sample,site,state]=-np.inf
                continue
            if np.isfinite(base[sample,site,3]) and not np.isfinite(base[sample,site,0]) and not np.isfinite(base[sample,site,1]) and not np.isfinite(base[sample,site,2]):
                result[sample,site,3]=0.
                for state in range(3):result[sample,site,state]=-np.inf
                continue
            a=float(base[sample,site,0]);b=float(base[sample,site,1])
            c=float(base[sample,site,2]);d=float(base[sample,site,3])
            orient=orientation[sample,site]
            p=phase_error+(1-2*phase_error)*phase[sample,site]
            a+=math.log(.5);d+=math.log(.5)
            if orient<0:b+=math.log(.5);c+=math.log(.5)
            elif orient==0:b+=math.log(p);c+=math.log(1-p)
            else:b+=math.log(1-p);c+=math.log(p)
            if group>=0:
                a+=child_log[group,site,0];b+=child_log[group,site,1]
                c+=child_log[group,site,2];d+=child_log[group,site,3]
            for index in range(offsets[sample],offsets[sample+1]):
                edge=adjacency[index]
                if edge_child[edge]!=sample:
                    a+=parent_log[edge,site,0];b+=parent_log[edge,site,1]
                    c+=parent_log[edge,site,2];d+=parent_log[edge,site,3]
            maximum=max(a,b,c,d)
            normalizer=maximum+math.log(((math.exp(a-maximum)+math.exp(b-maximum))+math.exp(c-maximum))+math.exp(d-maximum))
            result[sample,site,0]=a-normalizer;result[sample,site,1]=b-normalizer
            result[sample,site,2]=c-normalizer;result[sample,site,3]=d-normalizer
    return result



@njit(cache=True)
def branch_factor(q,selector,intermediate_slot,error):
    """Five-node factor contracted through two transmitted-allele masses.

No support is discarded. The same 16 selector states and all five outgoing
four-genotype messages are retained, including hypothetical genotype states.
"""
    alt=np.empty((3,2))
    for node in range(3):
        alt[node,0]=q[node,2]+q[node,3];alt[node,1]=q[node,1]+q[node,3]
    emission=np.zeros(16);message=np.zeros((5,4))
    chance=np.empty(4);down=np.empty(4)
    e=error;r=1-2*e
    for state in range(16):
        upstream=state>>2;downstream=state&3
        s0=upstream>>1;s1=upstream&1
        sa=(downstream>>(1-intermediate_slot))&1
        sp=(downstream>>intermediate_slot)&1
        a=e+r*alt[0,s0];b=e+r*alt[1,s1];p=e+r*alt[2,sp]
        chance[0]=(1-a)*(1-b);chance[1]=(1-a)*b
        chance[2]=a*(1-b);chance[3]=a*b
        # Child likelihood given the transmitted intermediate allele.
        hbit=1<<(1-intermediate_slot);pbit=1<<intermediate_slot
        l0=q[4,0]*(1-p)+q[4,pbit]*p
        l1=q[4,hbit]*(1-p)+q[4,3]*p
        d0=(1-e)*l0+e*l1;d1=e*l0+(1-e)*l1
        m0=0.;m1=0.;w=selector[state]
        for genotype in range(4):
            bit=(genotype>>(1-sa))&1
            d=d1 if bit else d0;down[genotype]=d
            mass=q[3,genotype]*chance[genotype]
            if bit:m1+=mass
            else:m0+=mass
            emission[state]+=mass*d
            message[3,genotype]+=w*chance[genotype]*d
        i0=(1-e)*m0+e*m1;i1=e*m0+(1-e)*m1
        u0=q[3,0]*(1-b)*down[0]+q[3,1]*b*down[1]
        u1=q[3,2]*(1-b)*down[2]+q[3,3]*b*down[3]
        v0=q[3,0]*(1-a)*down[0]+q[3,2]*a*down[2]
        v1=q[3,1]*(1-a)*down[1]+q[3,3]*a*down[3]
        z0=i0*q[4,0]+i1*q[4,hbit]
        z1=i0*q[4,pbit]+i1*q[4,3]
        for genotype in range(4):
            x=e+r*((genotype>>(1-s0))&1)
            y=e+r*((genotype>>(1-s1))&1)
            z=e+r*((genotype>>(1-sp))&1)
            message[0,genotype]+=w*((1-x)*u0+x*u1)
            message[1,genotype]+=w*((1-y)*v0+y*v1)
            message[2,genotype]+=w*((1-z)*z0+z*z1)
            child_h=(genotype>>(1-intermediate_slot))&1
            child_p=(genotype>>intermediate_slot)&1
            message[4,genotype]+=w*(i1 if child_h else i0)*(p if child_p else 1-p)
    for node in range(5):message[node]/=np.sum(message[node])
    return emission,message



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
            if fixed_homozygote(belief[nodes[node],site]):continue
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



@njit(cache=True,parallel=True)
def count_factors(fixed,gp,gc):
    sites=fixed.shape[1];tiles=(sites+31)//32;counts=np.zeros(tiles,dtype=np.int64)
    for tile in prange(tiles):
        for group in range(len(gc)):
            p0,p1=gp[group];child=gc[group]
            for site in range(tile*32,min(sites,(tile+1)*32)):
                if (not fixed[child,site] or (p0>=0 and not fixed[p0,site]) or
                    (p1>=0 and not fixed[p1,site])):counts[tile]+=1
    return counts



@njit(cache=True,parallel=True)
def fill_factors(base,fixed,gp,gc,error,offsets):
    sites=fixed.shape[1];tiles=len(offsets)-1
    group_index=np.empty(offsets[-1],dtype=np.int32)
    site_index=np.empty(offsets[-1],dtype=np.int32)
    emissions=np.empty((len(gc),sites,4),dtype=np.float32)
    for tile in prange(tiles):
        cursor=offsets[tile]
        for group in range(len(gc)):
            p0,p1=gp[group];child=gc[group]
            for site in range(tile*32,min(sites,(tile+1)*32)):
                if (not fixed[child,site] or (p0>=0 and not fixed[p0,site]) or
                    (p1>=0 and not fixed[p1,site])):
                    group_index[cursor]=group;site_index[cursor]=site;cursor+=1
                else:
                    a=.5 if p0<0 else (1. if base[p0,site,3]>-np.inf else 0.)
                    b=.5 if p1<0 else (1. if base[p1,site,3]>-np.inf else 0.)
                    a=error+(1-2*error)*a;b=error+(1-2*error)*b
                    value=(1-a)*(1-b) if base[child,site,0]>-np.inf else a*b
                    for state in range(4):emissions[group,site,state]=value
    return offsets,group_index,site_index,emissions



def prepare_factors(base,gp,gc,error):
    fixed=fixed_nodes(base);counts=count_factors(fixed,gp,gc)
    offsets=np.r_[0,np.cumsum(counts)].astype(np.int64)
    return fill_factors(base,fixed,gp,gc,error,offsets)



@njit(cache=True,parallel=True)
def active_copy_sweep(belief,parent_log,child_log,segregation,group_parent,
                     group_edge,group_child,error,damping,reverse,
                     active_groups,emissions,factor_support,dirty_tiles=None):
    offsets,group_index,site_index,_=factor_support
    groups,sites,_=child_log.shape
    keep=np.ones(groups,dtype=np.bool_) if active_groups is None else np.zeros(groups,dtype=np.bool_)
    if active_groups is not None:
        for group in active_groups:keep[group]=True
    tiles=len(offsets)-1;delta=np.zeros(tiles,dtype=np.float64)
    for tile in prange(tiles):
        parent_alt=np.empty((2,2),dtype=np.float64)
        pm=np.empty((2,4),dtype=np.float64)
        parent_fixed=np.empty(2,dtype=np.bool_)
        cm=np.empty(4,dtype=np.float64)
        for step in range(offsets[tile],offsets[tile+1]):
            entry=offsets[tile+1]-1-(step-offsets[tile]) if reverse else step
            group=group_index[entry];site=site_index[entry]
            if not keep[group]:continue
            old0,old1,old2,old3=emissions[group,site]
            child=group_child[group]
            qchild=cavity4(belief[child,site],child_log[group,site])
            child_fixed=fixed_homozygote(belief[child,site])
            parent_alt[:,:]=.5
            parent_fixed[:]=True
            for slot in range(2):
                parent=group_parent[group,slot];edge=group_edge[group,slot]
                if parent<0:continue
                parent_fixed[slot]=fixed_homozygote(belief[parent,site])
                q=cavity4(belief[parent,site],parent_log[edge,site])
                parent_alt[slot,0]=q[2]+q[3];parent_alt[slot,1]=q[1]+q[3]
            pm[:,:]=0.;cm[:]=0.
            if parent_fixed[0] and parent_fixed[1]:
                a=error+(1-2*error)*parent_alt[0,0]
                b=error+(1-2*error)*parent_alt[1,0]
                chance=((1-a)*(1-b),(1-a)*b,a*(1-b),a*b)
                total=0.
                for genotype in range(4):total+=qchild[genotype]*chance[genotype]
                for state in range(4):emissions[group,site,state]=total
                if damping>0 and not child_fixed:
                    for genotype in range(4):cm[genotype]=chance[genotype]
            else:
                for state in range(4):
                    s0,s1=state>>1,state&1
                    a=error+(1-2*error)*parent_alt[0,s0]
                    b=error+(1-2*error)*parent_alt[1,s1]
                    chance=((1-a)*(1-b),(1-a)*b,a*(1-b),a*b)
                    total=0.
                    for genotype in range(4):total+=qchild[genotype]*chance[genotype]
                    emissions[group,site,state]=total
                    if damping==0:continue
                    weight=segregation[group,site,state]
                    if not child_fixed:
                        for genotype in range(4):cm[genotype]+=weight*chance[genotype]
                    u0=qchild[0]*(1-b)+qchild[1]*b;u1=qchild[2]*(1-b)+qchild[3]*b
                    v0=qchild[0]*(1-a)+qchild[2]*a;v1=qchild[1]*(1-a)+qchild[3]*a
                    for genotype in range(4):
                        allele0=(genotype>>(1-s0))&1;allele1=(genotype>>(1-s1))&1
                        x=error+(1-2*error)*allele0;y=error+(1-2*error)*allele1
                        if not parent_fixed[0]:pm[0,genotype]+=weight*(u0*(1-x)+u1*x)
                        if not parent_fixed[1]:pm[1,genotype]+=weight*(v0*(1-y)+v1*y)
            if dirty_tiles is not None:
                if (emissions[group,site,0]!=old0 or emissions[group,site,1]!=old1
                        or emissions[group,site,2]!=old2 or emissions[group,site,3]!=old3):
                    dirty_tiles[group,tile]=True
            if damping==0:continue
            if not child_fixed:
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
                if parent<0 or parent_fixed[slot]:continue
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
