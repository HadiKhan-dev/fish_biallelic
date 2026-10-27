"""Reusable block geometry for a one-sided bound on the unchanged M1 HMM.

Each block uses a shared founder emission envelope and a transition envelope
covering every member of a painted-word class. Taking the smaller bound from
the two aligned block partitions preserves validity. Length eight is a workspace
choice, never a biological threshold: uncertified scores are evaluated exactly.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from . import block_certificate_fast as kernel


@dataclass
class Block:
    first: int
    stop: int
    which: np.ndarray
    refs: np.ndarray
    tv: np.ndarray
    mass: np.ndarray
    reference_mass: np.ndarray


class Geometry:
    def __init__(self, component, projection, candidates, claim_threads):
        p=projection;n=len(p.available);self.partitions=[]
        self.all_unavailable=not np.any(p.available & candidates)
        if self.all_unavailable:return
        claim_threads()
        dense=kernel.dense_bridges(p.stay,p.alpha,p.beta,p.row_arm,p.column_arm,p.pivot,p.branch,p.excursion)
        masses=dense.sum(-1)
        normal=np.divide(dense,masses[:,:,:,None],out=np.zeros_like(dense),where=masses[:,:,:,None]>0)
        del dense
        labels=np.sort(component.state_classes,axis=2)
        keys=[hashlib.sha256(component.genotype_likelihoods[i].tobytes()+component.observed[i].tobytes()).digest() for i in range(n)]
        edges=list(range(0,p.n_bins,8))+[p.n_bins]
        for bounds in (edges,sorted({p.n_bins-e for e in edges})):
            blocks=[]
            for first,stop in zip(bounds[:-1],bounds[1:]):
                word=np.ascontiguousarray(labels[:,first:stop]).reshape(n,-1)
                parents=np.flatnonzero(candidates & p.available)
                _,inverse=np.unique(word[parents],axis=0,return_inverse=True)
                which=np.full(n,-1,np.int64);which[parents]=inverse
                groups=int(inverse.max()+1);refs=np.empty(groups,np.int64)
                tv=np.zeros((groups,stop-first,p.n_states));mass=np.ones_like(tv);reference=np.ones_like(tv)
                for g in range(groups):
                    members=np.flatnonzero((which==g)&p.available)
                    ref=min(members,key=lambda q:keys[q]);refs[g]=ref
                    j=max(first,1);target=slice(j-first,stop-first);source=slice(j-1,stop-1)
                    rv=normal[ref,source]
                    distance=.5*np.abs(normal[members,source]-rv).sum(-1)
                    tv[g,target]=np.nextafter(distance.max(0)+128*np.finfo(float).eps,np.inf)
                    mass[g,target]=np.nextafter(masses[members,source].max(0)*(1+128*np.finfo(float).eps),np.inf)
                    reference[g,target]=np.where(masses[ref,source]>0,masses[ref,source],1.)
                blocks.append(Block(first,stop,which,refs,tv,mass,reference))
            self.partitions.append(blocks)

    def upper(self,p,t,children,zero,claim_threads):
        if self.all_unavailable:
            return np.broadcast_to(zero[children,None],(len(children),len(p.available))).copy()
        claim_threads()
        common=kernel.global_common_logs(children,p.state_alt,t.gl,t.coefficient,t.site_index,
                    t.site_start,t.site_stop,t.mismatch)
        claim_threads();powered,scales=kernel.power_common(common,children,t.exponent)
        dummy=np.empty((0,0,0))  # global emission envelope is finite; no allele-box branch
        answers=[]
        for partition in self.partitions:
            upper=np.zeros((len(children),len(p.available)))
            for b in partition:
                claim_threads()
                values=kernel.block_upper(children,b.first,b.stop,int(p.bin_start[b.first]),b.refs,b.tv,
                    b.mass,b.reference_mass,dummy,dummy,common,p.state_alt,t.gl,t.coefficient,t.site_index,
                    t.site_start,t.site_stop,t.exponent,t.null_diagonal,t.null_off,t.null_frequency,t.mismatch,
                    p.stay,p.alpha,p.beta,p.row_arm,p.column_arm,p.pivot,p.branch,p.excursion,powered,scales)
                active=np.flatnonzero(b.which>=0)
                upper[:,active]+=values[:,b.which[active]]
            with np.errstate(divide='ignore',invalid='ignore'):
                upper+=np.log(np.sum(p.pi[:,0],axis=1))[None]
                pad=4096*np.finfo(float).eps*(t.gl.shape[1]+64*p.n_bins*p.n_states)*np.maximum(1,np.abs(upper))
                upper+=pad
            upper[~np.isfinite(upper)]=np.inf
            upper[:,~p.available]=zero[children,None]
            answers.append(upper)
        return np.minimum(*answers)
