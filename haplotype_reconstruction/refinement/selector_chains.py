"""Exact selector-chain contraction and bounded incremental HMM workspaces.

Neutral ordinary-chain markers are integrated out, not dropped as evidence.
Cache blocks are computational units, never biological boundaries. Changed
likelihoods propagate until boundary messages are exactly unchanged. MAP scans
remain on the original marker grid in messages.py. Workspaces are not saved.
"""


import math
import numpy as np
from numba import njit, prange


from .factors import fixed_nodes


BLOCK_MARKERS = 256
MAX_CACHE_BYTES = 24 * 1024**3
BLOCK = 256
MAX_BRANCH_CACHE_BYTES = 4 * 1024**3


@njit(cache=True,inline="always")
def mix4(a,b,c,d,t):
    u=1-t
    x0,x1,x2,x3=u*a+t*c,u*b+t*d,u*c+t*a,u*d+t*b
    return u*x0+t*x1,u*x1+t*x0,u*x2+t*x3,u*x3+t*x2



@njit(cache=True)
def paired_chain_messages(emission,theta,diagnostics=True):
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
    posterior=np.empty_like(alpha) if diagnostics else np.empty((0,0),dtype=np.float64)
    cavity=np.empty_like(alpha)
    switches=np.empty((max(0,n-1),2) if diagnostics else (0,0),dtype=np.float64)
    for site in range(n):
        total=0.;post_total=0.
        for state in range(4):
            cavity[site,state]=predicted[site,state]*beta[site,state]
            total+=cavity[site,state]
            if diagnostics:
                posterior[site,state]=cavity[site,state]*emission[site,state]
                post_total+=posterior[site,state]
        for state in range(4):
            cavity[site,state]/=total
            if diagnostics:posterior[site,state]/=post_total
        if diagnostics and site+1<n:
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



@njit(cache=True,parallel=True)
def count_active(fixed,gp):
    counts=np.zeros(len(gp),dtype=np.int64)
    for group in prange(len(gp)):
        p0,p1=gp[group]
        for site in range(fixed.shape[1]):
            if (p0>=0 and not fixed[p0,site]) or (p1>=0 and not fixed[p1,site]):counts[group]+=1
    return counts



@njit(cache=True,parallel=True)
def fill_support(fixed,gp,offsets,log_correlation,resets):
    indices=np.empty(offsets[-1],dtype=np.int32);theta=np.zeros(offsets[-1])
    for group in prange(len(gp)):
        p0,p1=gp[group];cursor=offsets[group];previous=-1
        for site in range(fixed.shape[1]):
            if not ((p0>=0 and not fixed[p0,site]) or (p1>=0 and not fixed[p1,site])):continue
            indices[cursor]=site
            if previous>=0:
                theta[cursor-1]=(.5 if resets[site]!=resets[previous] else
                    -.5*math.expm1(log_correlation[site]-log_correlation[previous]))
            previous=site;cursor+=1
    return offsets,indices,theta



def prepare_selector_support(base,gp,theta):
    fixed=fixed_nodes(base);counts=count_active(fixed,gp)
    offsets=np.r_[0,np.cumsum(counts)].astype(np.int64)
    resets=np.r_[0,np.cumsum(theta==.5)].astype(np.int64)
    logs=np.zeros(len(theta));np.log1p(-2*theta,out=logs,where=theta<.5)
    log_correlation=np.r_[0.,np.cumsum(logs)]
    return fill_support(fixed,gp,offsets,log_correlation,resets)



class ChainCache:
    def __init__(self,support,max_bytes=MAX_CACHE_BYTES):
        offsets=support[0]
        # Three four-state float64 streams plus four-state float32 emissions;
        # block metadata adds less than one byte per marker at this block size.
        self.groups=max(0,int(np.searchsorted(offsets,max_bytes//113,side='right')-1))
        count=int(offsets[self.groups]);lengths=np.diff(offsets[:self.groups+1])
        blocks=np.r_[0,np.cumsum((lengths+BLOCK_MARKERS-1)//BLOCK_MARKERS)].astype(np.int64)
        nblocks=int(blocks[-1])
        self.arrays=(np.empty((count,4),dtype=np.float32),
            np.empty((count,4)),np.empty((count,4)),np.empty((count,4)),blocks,
            np.empty((nblocks,4)),np.zeros(self.groups,dtype=np.bool_),
            np.zeros(nblocks,dtype=np.bool_),np.zeros(nblocks),
            np.zeros((len(offsets)-1,5),dtype=np.int64))
        self.tile_mapping=None
        self.damping=None
        self.branches={}
        self.branch_bytes=0

    def invalidate(self):
        self.arrays[6].fill(False);self.arrays[7].fill(False)
        for cache in self.branches.values():cache.invalidate()



@njit(cache=True,parallel=True)
def _physical_tile_mapping(offsets,indices,tiles):
    """Map 32-marker factor tiles onto compressed selector cache blocks."""
    first=np.full((len(offsets)-1,tiles),-1,dtype=np.int32)
    stop=np.zeros_like(first)
    for group in prange(len(offsets)-1):
        begin=offsets[group];end=offsets[group+1];cursor=begin
        for tile in range(tiles):
            left=cursor
            while cursor<end and indices[cursor]<(tile+1)*32:cursor+=1
            if cursor>left:
                first[group,tile]=(left-begin)//BLOCK_MARKERS
                stop[group,tile]=(cursor-1-begin)//BLOCK_MARKERS+1
    return first,stop


@njit(cache=True,parallel=True)
def incremental_update(emissions,current,damping,active_groups,offsets,indices,theta,
                       saved,alpha,predicted,beta,block_offsets,boundary,valid,settled,
                       block_delta,stats,producer_dirty=None,tile_mapping=None):
    delta=np.zeros(len(emissions))
    for job in prange(len(active_groups)):
        group=active_groups[job];start=offsets[group];stop=offsets[group+1];length=stop-start
        if not length:continue
        stats[group,0]=length
        if group>=len(valid):
            local=np.empty((length,4),dtype=np.float32)
            for i in range(start,stop):
                for state in range(4):local[i-start,state]=emissions[group,indices[i],state]
            _,cavity,_=paired_chain_messages(local,theta[start:stop-1],False)
            for i in range(start,stop):
                for state in range(4):
                    old=current[group,indices[i],state]
                    new=(1-damping)*old+damping*cavity[i-start,state]
                    current[group,indices[i],state]=new
                    delta[group]=max(delta[group],abs(new-old))
            for k in range(1,5):stats[group,k]=length
            continue
        was_valid=valid[group];first_block=block_offsets[group]
        blocks=block_offsets[group+1]-first_block
        dirty=np.zeros(blocks,dtype=np.bool_)
        forward=np.zeros(blocks,dtype=np.bool_);backward=np.zeros(blocks,dtype=np.bool_)
        inspect=np.ones(blocks,dtype=np.bool_)
        if was_valid and producer_dirty is not None:
            inspect[:]=False
            first,stop_block=tile_mapping
            for tile in range(producer_dirty.shape[1]):
                if producer_dirty[group,tile] and first[group,tile]>=0:
                    for block in range(first[group,tile],stop_block[group,tile]):inspect[block]=True
        for block in range(blocks):
            if not inspect[block]:continue
            lo=start+block*BLOCK_MARKERS;hi=min(stop,lo+BLOCK_MARKERS)
            for i in range(lo,hi):
                different=not was_valid
                for state in range(4):
                    value=emissions[group,indices[i],state]
                    if was_valid and value!=saved[i,state]:different=True
                    saved[i,state]=value
                if different:dirty[block]=True;stats[group,1]+=1
        carry=not was_valid
        for block in range(blocks):
            lo=start+block*BLOCK_MARKERS;hi=min(stop,lo+BLOCK_MARKERS)
            if not (dirty[block] or carry):continue
            old0=alpha[hi-1,0] if was_valid else 0.
            old1=alpha[hi-1,1] if was_valid else 0.
            old2=alpha[hi-1,2] if was_valid else 0.
            old3=alpha[hi-1,3] if was_valid else 0.
            for i in range(lo,hi):
                p=(.25,.25,.25,.25) if i==start else mix4(alpha[i-1,0],alpha[i-1,1],alpha[i-1,2],alpha[i-1,3],theta[i-1])
                total=0.
                for state in range(4):
                    predicted[i,state]=p[state];alpha[i,state]=p[state]*saved[i,state]
                    total+=alpha[i,state]
                for state in range(4):alpha[i,state]/=total
            carry=(not was_valid or alpha[hi-1,0]!=old0 or alpha[hi-1,1]!=old1 or
                   alpha[hi-1,2]!=old2 or alpha[hi-1,3]!=old3)
            forward[block]=True;stats[group,2]+=hi-lo
        carry=not was_valid
        for block in range(blocks-1,-1,-1):
            lo=start+block*BLOCK_MARKERS;hi=min(stop,lo+BLOCK_MARKERS);key=first_block+block
            if not (dirty[block] or carry):continue
            old0=boundary[key,0] if was_valid else 0.
            old1=boundary[key,1] if was_valid else 0.
            old2=boundary[key,2] if was_valid else 0.
            old3=boundary[key,3] if was_valid else 0.
            for i in range(hi-1,lo-1,-1):
                if i==stop-1:
                    for state in range(4):beta[i,state]=1.
                else:
                    values=mix4(saved[i+1,0]*beta[i+1,0],saved[i+1,1]*beta[i+1,1],
                        saved[i+1,2]*beta[i+1,2],saved[i+1,3]*beta[i+1,3],theta[i])
                    total=sum(values)
                    for state in range(4):beta[i,state]=values[state]/total
            for state in range(4):boundary[key,state]=saved[lo,state]*beta[lo,state]
            # The emission at a block's first marker affects the preceding
            # block even when this block's beta vector itself is unchanged.
            carry=(not was_valid or boundary[key,0]!=old0 or boundary[key,1]!=old1 or
                   boundary[key,2]!=old2 or boundary[key,3]!=old3)
            backward[block]=True;stats[group,3]+=hi-lo
        for block in range(blocks):
            key=first_block+block;lo=start+block*BLOCK_MARKERS;hi=min(stop,lo+BLOCK_MARKERS)
            if forward[block] or backward[block] or not settled[key]:
                same=True;change=0.
                for i in range(lo,hi):
                    total=0.
                    for state in range(4):total+=predicted[i,state]*beta[i,state]
                    for state in range(4):
                        target=predicted[i,state]*beta[i,state]/total
                        old=current[group,indices[i],state]
                        new=(1-damping)*old+damping*target
                        current[group,indices[i],state]=new
                        if np.float32(new)!=old:same=False
                        change=max(change,abs(new-old))
                settled[key]=same;block_delta[key]=change;stats[group,4]+=hi-lo
            # Retain the actual residual of an update that rounded to the
            # same stored messages; do not pretend it was exactly zero.
            delta[group]=max(delta[group],block_delta[key])
        valid[group]=True
    return np.empty((0,0,0),dtype=np.float32),np.empty((0,0,0),dtype=np.float32),delta



def update_incremental_chains(emissions,current,damping,keep,support,cache,dirty_tiles=None):
    if cache.damping!=damping:
        cache.arrays[7].fill(False);cache.damping=damping
    cache.arrays[-1].fill(0)
    if dirty_tiles is not None and cache.tile_mapping is None:
        cache.tile_mapping=_physical_tile_mapping(support[0],support[1],dirty_tiles.shape[1])
    return incremental_update(emissions,current,damping,keep,*support,*cache.arrays,
                              dirty_tiles,cache.tile_mapping)



@njit(cache=True,inline='always')
def mix16_inplace(row,theta):
    for bit in (1,2,4,8):
        for left in range(16):
            if left&bit:continue
            right=left|bit;a=row[left];b=row[right]
            row[left]=(1-theta)*a+theta*b
            row[right]=(1-theta)*b+theta*a



@njit(cache=True)
def branch_cavity(emission,theta):
    """Same scaled forward/backward, with reused fixed-width scratch."""
    sites=len(emission);predicted=np.empty((sites,16));cavity=np.empty((sites,16))
    row=np.full(16,1/16);following=np.ones(16)
    for site in range(sites):
        if site:mix16_inplace(row,theta[site-1])
        total=0.
        for state in range(16):
            predicted[site,state]=row[state]
            row[state]*=emission[site,state];total+=row[state]
        for state in range(16):row[state]/=total
    for site in range(sites-1,-1,-1):
        total=0.
        for state in range(16):
            cavity[site,state]=predicted[site,state]*following[state]
            total+=cavity[site,state]
        for state in range(16):cavity[site,state]/=total
        if site:
            for state in range(16):following[state]*=emission[site,state]
            mix16_inplace(following,theta[site-1])
            total=np.sum(following)
            for state in range(16):following[state]/=total
    return cavity



class BranchCache:
    def __init__(self,sites):
        blocks=(sites+BLOCK-1)//BLOCK
        self.arrays=(np.empty((sites,16),dtype=np.float32),np.empty((sites,16)),
            np.empty((sites,16)),np.empty((sites,16)),np.empty((sites,16)),
            np.empty((blocks,16)),np.zeros(1,dtype=np.bool_),np.zeros(4,dtype=np.int64))
        self.bytes=sum(a.nbytes for a in self.arrays)

    def invalidate(self):self.arrays[-2][0]=False

    def update(self,emissions,theta):
        self.arrays[-1].fill(0)
        update_branch(emissions,theta,*self.arrays)
        return self.arrays[4]



@njit(cache=True)
def update_branch(emissions,theta,saved,alpha,predicted,beta,cavity,boundary,valid,stats):
    sites=len(emissions);blocks=len(boundary);ready=valid[0]
    dirty=np.zeros(blocks,dtype=np.bool_)
    forward=np.zeros(blocks,dtype=np.bool_);backward=np.zeros(blocks,dtype=np.bool_)
    row=np.empty(16);previous=np.empty(16)
    for site in range(sites):
        different=not ready
        for state in range(16):
            value=emissions[site,state]
            if ready and value!=saved[site,state]:different=True
            saved[site,state]=value
        if different:dirty[site//BLOCK]=True;stats[0]+=1
    carry=not ready
    for block in range(blocks):
        lo=block*BLOCK;hi=min(sites,lo+BLOCK)
        if not (dirty[block] or carry):continue
        if ready:
            for state in range(16):previous[state]=alpha[hi-1,state]
        for state in range(16):row[state]=1/16 if lo==0 else alpha[lo-1,state]
        for site in range(lo,hi):
            if site:mix16_inplace(row,theta[site-1])
            total=0.
            for state in range(16):
                predicted[site,state]=row[state]
                row[state]*=saved[site,state];total+=row[state]
            for state in range(16):row[state]/=total;alpha[site,state]=row[state]
        carry=not ready
        if ready:
            for state in range(16):
                if alpha[hi-1,state]!=previous[state]:carry=True
        forward[block]=True;stats[1]+=hi-lo
    carry=not ready
    for block in range(blocks-1,-1,-1):
        lo=block*BLOCK;hi=min(sites,lo+BLOCK)
        if not (dirty[block] or carry):continue
        if ready:
            for state in range(16):previous[state]=boundary[block,state]
        if hi==sites:
            row[:]=1.
        else:
            for state in range(16):row[state]=saved[hi,state]*beta[hi,state]
            mix16_inplace(row,theta[hi-1]);total=np.sum(row)
            for state in range(16):row[state]/=total
        for site in range(hi-1,lo-1,-1):
            for state in range(16):beta[site,state]=row[state]
            if site>lo:
                for state in range(16):row[state]*=saved[site,state]
                mix16_inplace(row,theta[site-1]);total=np.sum(row)
                for state in range(16):row[state]/=total
        carry=not ready
        for state in range(16):
            boundary[block,state]=saved[lo,state]*beta[lo,state]
            if ready and boundary[block,state]!=previous[state]:carry=True
        backward[block]=True;stats[2]+=hi-lo
    for block in range(blocks):
        if not (forward[block] or backward[block]):continue
        lo=block*BLOCK;hi=min(sites,lo+BLOCK)
        for site in range(lo,hi):
            total=0.
            for state in range(16):
                cavity[site,state]=predicted[site,state]*beta[site,state]
                total+=cavity[site,state]
            for state in range(16):cavity[site,state]/=total
        stats[3]+=hi-lo
    valid[0]=True
