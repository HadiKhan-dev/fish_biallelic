"""Incumbent-tight sample-specific grids for founder coordinate search.

Keep each sample switch and leaf edge; sum all usable likelihoods. A changed
panel is scored on a restricted path family and still needs full acceptance.
Unsplit sample/leaf cells reuse immutable totals; split cells retain every SNP.
"""
import time
from threading import Lock
from .proxy_primary import CoarsePrimary, informative_weights
from ...painting.model import available_process_memory_bytes
import math
import numpy as np
from numba import njit,prange
from .scoring import _unordered_pairs


@njit(cache=True,parallel=True,nogil=True)
def bin_counts(painting,offsets):
    result=np.ones((len(painting),len(offsets)-1),np.int64)
    for sample in prange(len(painting)):
        for block in range(len(offsets)-1):
            for site in range(offsets[block]+1,offsets[block+1]):
                if painting[sample,site]!=painting[sample,site-1]:
                    result[sample,block]+=1
    return result


@njit(cache=True,parallel=True,nogil=True)
def fill(logs,evidence,complete,leaves,offsets,painting,bins,data_offsets,
         bin_offsets,data,weights,starts,tolerance):
    samples=len(logs);center=math.log(1./3.)
    for task in prange(samples*len(leaves)):
        block,sample=task//samples,task%samples
        leaf=leaves[block];size=len(leaf);width=bins[sample,block]
        base=data_offsets[sample,block];pbase=bin_offsets[sample,block]
        marker=0
        starts[pbase]=offsets[block]
        for position in range(leaf.shape[1]):
            site=offsets[block]+position
            if position and painting[sample,site]!=painting[sample,site-1]:
                marker+=1;starts[pbase+marker]=site
            if complete[site]:
                p=evidence[sample,site]
                if max(p[0],p[1],p[2])-min(p[0],p[1],p[2])>tolerance:
                    weights[pbase+marker]+=1
            a,b,c=logs[sample,site]
            if a==0. and b==0. and c==0.:
                continue
            for first in range(size):
                for second in range(first,size):
                    dosage=leaf[first,position]+leaf[second,position]
                    value=logs[sample,site,dosage]-center
                    data[base+(first*size+second)*width+marker]+=value
                    if first!=second:
                        data[base+(second*size+first)*width+marker]+=value


@njit(cache=True,nogil=True)
def pair_codes(selected,sizes,first,second):
    result=np.empty((selected.shape[1],len(first)),np.int64)
    for block in range(selected.shape[1]):
        for state in range(len(first)):
            result[block,state]=selected[first[state],block]*sizes[block]+selected[second[state],block]
    return result


@njit(cache=True,parallel=True,nogil=True)
def score_sparse(data,data_offsets,bins,codes,penalty):
    answer=np.empty(len(bins),np.float64)
    for sample in prange(len(bins)):
        row=np.zeros(codes.shape[1],np.float64)
        for block in range(bins.shape[1]):
            width=bins[sample,block];base=data_offsets[sample,block]
            for marker in range(width):
                switched=np.max(row)-penalty
                for state in range(len(row)):
                    row[state]=max(row[state],switched)+data[base+codes[block,state]*width+marker]
        answer[sample]=np.max(row)
    return answer


@njit(cache=True,parallel=True,nogil=True)
def paint_sparse(data,data_offsets,bins,bin_offsets,codes,first,second,founders,penalty):
    answer=np.empty(bin_offsets[-1,-1],np.int32)
    scores=np.empty(len(bins),np.float64)
    for sample in prange(len(bins)):
        sample_start=bin_offsets[sample,0]
        length=bin_offsets[sample,-1]-sample_start
        previous=np.empty(length,np.int32)
        flags=np.empty((length,codes.shape[1]),np.bool_)
        row=np.zeros(codes.shape[1],np.float64)
        for block in range(bins.shape[1]):
            width=bins[sample,block];base=data_offsets[sample,block]
            for marker in range(width):
                index=bin_offsets[sample,block]-sample_start+marker
                best=int(np.argmax(row));previous[index]=best
                switched=row[best]-penalty
                for state in range(len(row)):
                    flags[index,state]=row[state]<switched
                    row[state]=max(row[state],switched)+data[base+codes[block,state]*width+marker]
        state=int(np.argmax(row));scores[sample]=row[state]
        for index in range(length-1,-1,-1):
            answer[sample_start+index]=first[state]*founders+second[state]
            if flags[index,state]:state=previous[index]
    return answer,scores


@njit(cache=True,parallel=True,nogil=True)
def fixed_sparse(selected,painting,data,data_offsets,bin_offsets,bins,sizes):
    founders,blocks=selected.shape
    proposed=selected.copy();gains=np.zeros((founders,blocks),np.float64)
    for block in prange(blocks):
        size=sizes[block];unary=np.zeros((founders,size),np.float64)
        for sample in range(len(bins)):
            width=bins[sample,block];base=data_offsets[sample,block]
            for marker in range(width):
                state=painting[bin_offsets[sample,block]+marker]
                first,second=state//founders,state%founders
                row_a,row_b=selected[first,block],selected[second,block]
                current=data[base+(row_a*size+row_b)*width+marker]
                for side in range(1 if first==second else 2):
                    focal=first if side==0 else second
                    for candidate in range(size):
                        a=candidate if first==focal else row_a
                        b=candidate if second==focal else row_b
                        unary[focal,candidate]+=data[base+(a*size+b)*width+marker]-current
        for focal in range(founders):
            best=int(np.argmax(unary[focal]))
            if unary[focal,best]>1e-8:
                proposed[focal,block]=best;gains[focal,block]=unary[focal,best]
    return proposed,gains.sum(axis=1)


@njit(cache=True,parallel=True,nogil=True)
def occupancy_sparse(painting,weights,bin_offsets,founders):
    counts=np.zeros((len(bin_offsets),founders),np.int64)
    for sample in prange(len(bin_offsets)):
        for marker in range(bin_offsets[sample,0],bin_offsets[sample,-1]):
            state=painting[marker]
            counts[sample,state//founders]+=weights[marker]
            counts[sample,state%founders]+=weights[marker]
    return counts.sum(axis=0)


@njit(cache=True,parallel=True,nogil=True)
def expand(painting,starts,offsets,bin_offsets):
    output=np.empty((len(bin_offsets),offsets[-1]),np.int32)
    for sample in prange(len(bin_offsets)):
        for block in range(len(offsets)-1):
            for marker in range(bin_offsets[sample,block],bin_offsets[sample,block+1]):
                stop=starts[marker+1] if marker+1<bin_offsets[sample,block+1] else offsets[block+1]
                output[sample,starts[marker]:stop]=painting[marker]
    return output


class SampleSparsePrimary:
    def __init__(self,workspace,selected):
        painting=workspace.evaluate(selected,paint=True)
        self.bins=bin_counts(painting,workspace.offsets)
        self.sizes=np.array([len(leaf) for leaf in workspace.leaves],np.int64)
        self.bin_offsets=np.empty((len(painting),len(self.sizes)+1),np.int64)
        self.data_offsets=np.empty_like(self.bin_offsets)
        total_data=total_bins=0
        for sample in range(len(painting)):
            self.bin_offsets[sample,0]=total_bins
            self.data_offsets[sample,0]=total_data
            self.bin_offsets[sample,1:]=total_bins+np.cumsum(self.bins[sample])
            self.data_offsets[sample,1:]=total_data+np.cumsum(self.bins[sample]*self.sizes**2)
            total_bins=int(self.bin_offsets[sample,-1]);total_data=int(self.data_offsets[sample,-1])
        self.data=np.zeros(total_data,np.float64)
        self.weights=np.zeros(total_bins,np.int64);self.starts=np.empty(total_bins,np.int64)
        self.offsets=workspace.offsets;self.penalty=workspace.penalty
        fill(workspace.logs,workspace.evidence,workspace.complete,workspace.leaves,
            workspace.offsets,painting,self.bins,self.data_offsets,self.bin_offsets,
            self.data,self.weights,self.starts,workspace.evidence.dtype.type(1e-10))

    def _codes(self,selected):
        first,second=_unordered_pairs(len(selected))
        return pair_codes(selected,self.sizes,first,second),first,second

    def score(self,selected):
        codes,_,_=self._codes(selected)
        return float(score_sparse(self.data,self.data_offsets,self.bins,codes,self.penalty).sum())

    def paint_compact(self,selected):
        codes,first,second=self._codes(selected)
        return paint_sparse(self.data,self.data_offsets,self.bins,self.bin_offsets,
            codes,first,second,len(selected),self.penalty)

    def fixed(self,selected,painting):
        return fixed_sparse(selected,painting,self.data,self.data_offsets,
            self.bin_offsets,self.bins,self.sizes)

    def occupancy(self,painting,founders):
        return occupancy_sparse(painting,self.weights,self.bin_offsets,founders)

    def expand_for_validation(self,painting):
        return expand(painting,self.starts,self.offsets,self.bin_offsets)

@njit(cache=True, parallel=True, nogil=True)
def fill_reused(logs,evidence,complete,leaves,offsets,painting,bins,data_offsets,
         bin_offsets,data,weights,starts,tolerance,totals,whole_weights):
    samples=len(logs);center=np.log(1./3.)
    for task in prange(samples*len(leaves)):
        block,sample=task//samples,task%samples
        leaf=leaves[block];size=len(leaf);width=bins[sample,block]
        base=data_offsets[sample,block];pbase=bin_offsets[sample,block]
        if width==1:
            table=totals[block]
            starts[pbase]=offsets[block]
            weights[pbase]=whole_weights[sample,block]
            for first in range(size):
                for second in range(size):
                    data[base+first*size+second]=table[sample,first,second,0]
            continue
        marker=0;starts[pbase]=offsets[block]
        for position in range(leaf.shape[1]):
            site=offsets[block]+position
            if position and painting[sample,site]!=painting[sample,site-1]:
                marker+=1;starts[pbase+marker]=site
            if complete[site]:
                p=evidence[sample,site]
                if max(p[0],p[1],p[2])-min(p[0],p[1],p[2])>tolerance:
                    weights[pbase+marker]+=1
            a,b,c=logs[sample,site]
            if a==0. and b==0. and c==0.:continue
            for first in range(size):
                for second in range(first,size):
                    dosage=leaf[first,position]+leaf[second,position]
                    value=logs[sample,site,dosage]-center
                    data[base+(first*size+second)*width+marker]+=value
                    if first!=second:
                        data[base+(second*size+first)*width+marker]+=value

class Totals:
    def __init__(self,workspace):
        width=max(int(x.shape[1]) for x in workspace.leaves)
        primary=CoarsePrimary(workspace,width)
        self.tables=primary.models.arrays
        offsets=np.arange(len(workspace.leaves)+1,dtype=np.int64)
        self.weights=informative_weights(workspace.evidence,workspace.complete,
            workspace.offsets,offsets,width,workspace.evidence.dtype.type(1e-10))


class ReusedPrimary(SampleSparsePrimary):
    registry_lock=Lock()
    statistics=dict(preparations=0,total_cells=0,unsplit_cells=0,
                    invariant_build_seconds=0.,fill_seconds=0.,memory_fallbacks=0)

    def __init__(self,workspace,selected):
        painting=workspace.evaluate(selected,paint=True)
        self.bins=bin_counts(painting,workspace.offsets)
        self.sizes=np.array([len(leaf) for leaf in workspace.leaves],np.int64)
        self.bin_offsets=np.empty((len(painting),len(self.sizes)+1),np.int64)
        self.data_offsets=np.empty_like(self.bin_offsets)
        total_data=total_bins=0
        for sample in range(len(painting)):
            self.bin_offsets[sample,0]=total_bins
            self.data_offsets[sample,0]=total_data
            self.bin_offsets[sample,1:]=total_bins+np.cumsum(self.bins[sample])
            self.data_offsets[sample,1:]=total_data+np.cumsum(self.bins[sample]*self.sizes**2)
            total_bins=int(self.bin_offsets[sample,-1]);total_data=int(self.data_offsets[sample,-1])
        self.data=np.zeros(total_data,np.float64)
        self.weights=np.zeros(total_bins,np.int64);self.starts=np.empty(total_bins,np.int64)
        self.offsets=workspace.offsets;self.penalty=workspace.penalty
        owner=workspace
        while hasattr(owner,'base'):owner=owner.base
        with self.registry_lock:
            if not hasattr(owner,'_anchor_leaf_totals_lock'):
                owner._anchor_leaf_totals_lock=Lock()
        elapsed=0.
        required=8*len(painting)*(int((self.sizes**2).sum())+len(self.sizes))
        available=available_process_memory_bytes()
        if not hasattr(owner,'_anchor_leaf_totals') and available is not None and required>available//8:
            fill(workspace.logs,workspace.evidence,workspace.complete,workspace.leaves,
                workspace.offsets,painting,self.bins,self.data_offsets,self.bin_offsets,
                self.data,self.weights,self.starts,workspace.evidence.dtype.type(1e-10))
            with self.registry_lock:self.statistics['memory_fallbacks']+=1
            return
        with owner._anchor_leaf_totals_lock:
            if not hasattr(owner,'_anchor_leaf_totals'):
                start=time.monotonic();owner._anchor_leaf_totals=Totals(owner)
                elapsed=time.monotonic()-start
            totals=owner._anchor_leaf_totals
        start=time.monotonic()
        fill_reused(workspace.logs,workspace.evidence,workspace.complete,workspace.leaves,
            workspace.offsets,painting,self.bins,self.data_offsets,self.bin_offsets,
            self.data,self.weights,self.starts,workspace.evidence.dtype.type(1e-10),
            totals.tables,totals.weights)
        with self.registry_lock:
            self.statistics['preparations']+=1
            self.statistics['invariant_build_seconds']+=elapsed
            self.statistics['fill_seconds']+=time.monotonic()-start
            self.statistics['total_cells']+=self.bins.size
            self.statistics['unsplit_cells']+=int(np.count_nonzero(self.bins==1))
