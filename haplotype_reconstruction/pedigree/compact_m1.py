"""Exact G4 M1 values as retained raw scores plus a counted default floor.

The genome gate/profile is already fixed. Never apply genome_profile to these
values again: the exact-sum floor is not idempotent. Dense expansion is explicit
and reserved for the reference path and migration checks.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CompactM1:
    indptr: np.ndarray             # N+1, parent rows sorted within each child
    parents: np.ndarray            # retained identities; complete cutoff ties
    raw: np.ndarray                # C x retained entries, exact raw M1
    best: np.ndarray               # N, exact genome maximum
    cutoff: np.ndarray             # N, exact G4 cutoff
    chromosome_best: np.ndarray    # C x N, exact maximum over ALL eligible parents
    floors: np.ndarray             # C x N, fixed G4 exact-sum allocation
    markers: np.ndarray            # C, original information counts
    eligible_counts: np.ndarray    # N, includes every implicit floor identity

    @property
    def n_samples(self):
        return len(self.indptr)-1

    @property
    def n_contigs(self):
        return len(self.markers)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in vars(self).values())

    def retained(self, child):
        return self.parents[self.indptr[child]:self.indptr[child+1]]

    def values(self, child, parents):
        """C x requested identities. Caller supplies legitimate eligible parents."""
        parents=np.asarray(parents,dtype=np.int64)
        first,stop=self.indptr[child:child+2]
        retained=self.parents[first:stop]
        output=np.broadcast_to(self.floors[:,child,None],(self.n_contigs,len(parents))).copy()
        where=np.searchsorted(retained,parents)
        possible=np.flatnonzero(where<len(retained))
        hits=possible[retained[where[possible]]==parents[possible]]
        output[:,hits]=self.raw[:,first+where[hits]]
        return output

    def dense_reference(self, eligibility):
        """Explicit O(C N^2) reference/migration adapter, never an implicit cast."""
        output=np.full((self.n_contigs,self.n_samples,self.n_samples),-np.inf)
        for c in np.flatnonzero(eligibility.children):
            parents=np.flatnonzero(eligibility.parents[c])
            output[:,c,parents]=self.values(c,parents)
        return output

    def chromosome_reference(self, chromosome, eligibility):
        """One dense chromosome for the unchanged score/exposure migration API."""
        output = np.full((self.n_samples, self.n_samples), -np.inf)
        for child in np.flatnonzero(eligibility.children):
            parents = np.flatnonzero(eligibility.parents[child])
            output[child, parents] = self.floors[chromosome, child]
            first, stop = self.indptr[child:child+2]
            output[child, self.parents[first:stop]] = self.raw[chromosome, first:stop]
        return output

    def keep_reference(self):
        keep=np.zeros((self.n_samples,self.n_samples),bool)
        for c in range(self.n_samples):keep[c,self.retained(c)]=True
        return keep

    def encode(self):
        return dict(format='genome-floor-csr-v1',**vars(self))

    @classmethod
    def decode(cls, record):
        if record['format']!='genome-floor-csr-v1':
            raise ValueError('unsupported compact M1 record')
        return cls(**{key:record[key] for key in cls.__dataclass_fields__})


class CompactBuilder:
    def __init__(self,n,markers,eligible_counts):
        self.n=n;self.markers=np.asarray(markers);self.counts=eligible_counts
        c=len(markers);self.best=np.full(n,-np.inf);self.cutoff=np.full(n,-np.inf)
        self.local=np.full((c,n),-np.inf);self.floors=self.local.copy()
        self.parents=[np.empty(0,np.int64) for _ in range(n)]
        self.raw=[np.empty((c,0)) for _ in range(n)]

    def add(self,batch):
        for i,c in enumerate(batch['children']):
            self.best[c]=batch['best'][i];self.cutoff[c]=batch['cutoff'][i]
            self.local[:,c]=batch['chromosome_best'][:,i];self.floors[:,c]=batch['floors'][:,i]
            parents=batch['retained'][i]
            self.parents[c]=parents;self.raw[c]=batch['values'][:,i,parents].copy()

    def finish(self):
        indptr=np.r_[0,np.cumsum([len(x) for x in self.parents],dtype=np.int64)]
        return CompactM1(indptr,np.concatenate(self.parents),np.concatenate(self.raw,axis=1),self.best,self.cutoff,
                         self.local,self.floors,self.markers,np.asarray(self.counts,np.int64))
