"""Exact segment dynamic programming for all requested pair-excluded forward maxima.

The adaptive path bounds preparation by O(m^2*K^2) per sample while keeping direct
forwards when their estimated complete work is smaller. See choose_segments.
"""
import numpy as np
from numba import njit
from .objective import pair_lists, _gather_states

@njit(cache=True,inline='always')
def best_without(weights,first,second,a,b):
    best=-np.inf;which=-1
    for z in range(len(weights)):
        x,y=first[z],second[z]
        if x==a or y==a or x==b or y==b:continue
        if weights[z]>best:best=weights[z];which=z
    return which,best

@njit(cache=True)
def interval_values(weights,first,second,qa,qb,out):
    z0,v0=best_without(weights,first,second,-1,-1)
    roots=np.empty(2,np.int64);roots[0]=first[z0];roots[1]=second[z0]
    nodes=np.empty(2,np.int64);single=np.empty(2)
    children=np.empty((2,2),np.int64);double=np.empty((2,2))
    for side in range(2):
        r=roots[side]
        if side==1 and r==roots[0]:break
        z,v=best_without(weights,first,second,r,-1);nodes[side]=z;single[side]=v
        children[side,0]=first[z];children[side,1]=second[z]
        for child in range(2):
            s=children[side,child]
            if child==1 and s==children[side,0]:break
            _,double[side,child]=best_without(weights,first,second,r,s)
    for d in range(len(qa)):
        a,b=qa[d],qb[d]
        if a!=roots[0] and b!=roots[0] and a!=roots[1] and b!=roots[1]:out[d]=v0;continue
        side=0 if a==roots[0] or b==roots[0] else 1
        r=roots[side];s=b if a==r else a
        if s!=children[side,0] and s!=children[side,1]:out[d]=single[side]
        else:out[d]=double[side,0 if s==children[side,0] else 1]

@njit(cache=True,nogil=True)
def pair_segment_dp(emission,first,second,qa,qb,penalty,prune=True):
    m,Q=emission.shape;P=len(qa)
    values=np.empty((m,P));weights=np.empty((m,Q));scores=np.empty((m,P))
    alive=np.zeros((m,P),np.bool_);active=np.zeros(m,np.bool_);interval=np.empty(P)
    intervals=0;max_active=0;pair_visits=0
    for t in range(m):
        values[t,:]=-np.inf;alive[t,:]=True;active[t]=True;weights[t,:]=0.
        active_count=0
        for u in range(t+1):
            if not active[u]:continue
            active_count+=1;intervals+=1
            for z in range(Q):weights[u,z]+=emission[t,z]
            interval_values(weights[u],first,second,qa,qb,interval)
            for d in range(P):
                if not alive[u,d]:continue
                pair_visits+=1
                v=interval[d] if u==0 else values[u-1,d]-penalty+interval[d]
                scores[u,d]=v
                if v>values[t,d]:values[t,d]=v
        max_active=max(max_active,active_count)
        if prune:
            for u in range(t+1):
                if not active[u]:continue
                keep=False
                for d in range(P):
                    if alive[u,d] and scores[u,d]<=values[t,d]-penalty:alive[u,d]=False
                    if alive[u,d]:keep=True
                active[u]=keep
    return values,intervals,max_active,pair_visits


@njit(cache=True)
def choose_segments(k, m, queries):
    """Compare proportional loop-cost models, without inspecting scientific outcomes.

    Direct forwards gather, update and maximize over q2 states per bin; pair-list
    construction scans at most 2*q states. The segment model includes seven state
    scans, interval accumulation and the main score/prune query scans. The static
    oracle also scans the queries: since queries<=q, actual work stays O(model).
    Choosing the smaller estimate preserves the O(m^2*(K^2+queries)) joint bound.
    These are arithmetic-loop units, not measured CPU instructions or time guarantees.
    """
    q=k*(k+1)//2; q2=(k-1)*(k-2)//2
    direct=3.0*m*queries*q2+2.0*queries*q
    segment=(m*(m+1)/2.0)*(8.0*q+2.0*queries)+m*queries
    return queries>0 and segment<direct


def workspace_bytes(k, m, queries):
    """Peak new scratch for one active sample (including input gather and all DP arrays)."""
    if not choose_segments(k,m,queries):return int(8*k*k+1024)
    q=k*(k+1)//2
    return int(16*m*q+17*m*queries+m+24*queries+24*q+8*k*k+1024)


@njit(cache=True,nogil=True)
def regions(flat, block_start, block_haps, block_bins, rows, penalty, sample, qa, qb):
    k=rows.shape[0];first,second=pair_lists(np.ones(k,np.bool_))
    m=0
    for bins in block_bins:m+=bins
    emission=np.empty((m,len(first)));offset=0
    for block in range(len(block_bins)):
        haps=block_haps[block];bins=block_bins[block]
        base=block_start[block]+sample*(haps*(haps+1)//2)*bins
        _gather_states(flat,base,haps,bins,rows,first,second,block,emission[offset:offset+bins])
        offset+=bins
    return pair_segment_dp(emission,first,second,qa,qb,penalty)[0]
