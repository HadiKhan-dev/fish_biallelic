"""Incremental messages after an allele-preserving suffix transposition.

The right suffix is only a state relabeling. Recompute the affected forward
and backward messages outward from the swap, stopping once every state agrees
with the old message up to one common offset. Copy the remaining boundary
messages under that permutation/offset. This preserves the max-plus HMM in real
arithmetic; full canonical candidate acceptance remains the original caller's.
"""
from collections import OrderedDict
from threading import Lock
import numpy as np
from numba import njit, prange
from . import site_kernels
from . import scoring


@njit(cache=True,parallel=True,nogil=True)
def update_messages(old_forward,old_backward,dosages,logs,cuts,permutation,boundary,penalty):
    samples,boundaries,states=old_forward.shape
    forward=np.empty_like(old_forward)
    backward=np.empty_like(old_backward)
    visited=np.zeros((samples,2),np.int64)
    joined=np.zeros((samples,2),np.bool_)
    center=np.log(1./3.)
    for sample in prange(samples):
        for cut in range(boundary+1):
            forward[sample,cut]=old_forward[sample,cut]
        for cut in range(boundary,boundaries):
            for state in range(states):
                backward[sample,cut,state]=old_backward[sample,cut,permutation[state]]
        row=old_forward[sample,boundary].copy()
        cut=boundary+1
        for site in range(cuts[boundary],cuts[-1]):
            switched=np.max(row)-penalty
            neutral=np.all(logs[sample,site]==0.)
            for state in range(states):
                value=0. if neutral else logs[sample,site,dosages[site,state]]-center
                row[state]=max(row[state],switched)+value
            visited[sample,0]+=1
            if cut<boundaries and site+1==cuts[cut]:
                forward[sample,cut]=row
                shift=row[0]-old_forward[sample,cut,permutation[0]]
                equal=True
                for state in range(states):
                    if row[state] != old_forward[sample,cut,permutation[state]]+shift:
                        equal=False
                        break
                if equal:
                    for later in range(cut+1,boundaries):
                        for state in range(states):
                            forward[sample,later,state]=old_forward[sample,later,permutation[state]]+shift
                    joined[sample,0]=True
                    break
                cut+=1
        row=backward[sample,boundary].copy()
        cut=boundary-1
        for site in range(cuts[boundary]-1,cuts[0]-1,-1):
            switched=np.max(row)-penalty
            neutral=np.all(logs[sample,site]==0.)
            for state in range(states):
                value=0. if neutral else logs[sample,site,dosages[site,state]]-center
                row[state]=max(row[state],switched)+value
            visited[sample,1]+=1
            if cut>=0 and site==cuts[cut]:
                backward[sample,cut]=row
                shift=row[0]-old_backward[sample,cut,0]
                equal=True
                for state in range(states):
                    if row[state] != old_backward[sample,cut,state]+shift:
                        equal=False
                        break
                if equal:
                    for earlier in range(cut):
                        backward[sample,earlier]=old_backward[sample,earlier]+shift
                    joined[sample,1]=True
                    break
                cut-=1
    return forward,backward,visited,joined


def suffix_change(before,after,cuts):
    if before.shape!=after.shape or not len(cuts):return None
    rows=np.flatnonzero(np.any(before!=after,axis=1))
    if len(rows)!=2:return None
    a,b=map(int,rows)
    first=int(np.flatnonzero(before[a]!=after[a])[0])
    boundary=int(np.searchsorted(cuts,first,side='right')-1)
    if boundary<0:return None
    cut=int(cuts[boundary])
    if not (np.array_equal(before[:,:cut],after[:,:cut]) and
            np.array_equal(after[a,cut:],before[b,cut:]) and
            np.array_equal(after[b,cut:],before[a,cut:])):
        return None
    first,second=np.triu_indices(len(before))
    lookup=np.empty((len(before),len(before)),np.int64)
    lookup[first,second]=lookup[second,first]=np.arange(len(first))
    swap=np.arange(len(before));swap[a],swap[b]=swap[b],swap[a]
    return boundary,np.ascontiguousarray(lookup[swap[first],swap[second]])


class ExchangeCache:
    def __init__(self,native):
        self.native=native
        self.entries=OrderedDict()
        self.lock=Lock()
        self.statistics=dict(full_preparations=0,incremental_preparations=0,
            unchanged_panels=0,visited_sites=0,full_scan_sites_avoided=0,
            coalesced_directions=0)

    def __call__(self,haps,evidence,complete,cuts,penalty,prepared=None,quota=None):
        if prepared is None:return self.native(haps,evidence,complete,cuts,penalty,prepared,quota=quota)
        key=(id(prepared),float(penalty),haps.shape,cuts.tobytes())
        with self.lock:entry=self.entries.get(key)
        if entry is not None and np.array_equal(entry['haps'],haps):
            forward,backward=entry['forward'],entry['backward']
            with self.lock:self.statistics['unchanged_panels']+=1
        else:
            dosages=scoring._dosage_table(haps)
            if dosages is None:return self.native(haps,evidence,complete,cuts,penalty,prepared,quota=quota)
            changed=None if entry is None else suffix_change(entry['haps'],haps,cuts)
            if changed is None:
                forward,backward=site_kernels.exchange_messages(dosages,prepared,cuts,float(penalty))
                with self.lock:self.statistics['full_preparations']+=1
            else:
                boundary,permutation=changed
                forward,backward,visited,joined=update_messages(entry['forward'],entry['backward'],
                    dosages,prepared,cuts,permutation,boundary,float(penalty))
                with self.lock:
                    self.statistics['incremental_preparations']+=1
                    self.statistics['visited_sites']+=int(visited.sum())
                    self.statistics['full_scan_sites_avoided']+=int(2*len(prepared)*haps.shape[1]-visited.sum())
                    self.statistics['coalesced_directions']+=int(joined.sum())
            # Hold the evidence reference so its ordinary Python id cannot be
            # reused for a different workspace while this cache entry lives.
            saved=dict(logs=prepared,haps=haps.copy(),forward=forward,backward=backward)
            with self.lock:
                self.entries[key]=saved;self.entries.move_to_end(key)
                while len(self.entries)>2:self.entries.popitem(last=False)
        from .exchanges import _exchange_scores
        return _exchange_scores(forward,backward,len(haps),float(penalty),quota)
