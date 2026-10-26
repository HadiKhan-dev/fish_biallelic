"""Selective M0/M1 screen with persistent chromosome workers and compact output.

The existing forward.screen is the dense exact reference. This path reads only
PREPARED records, never saved exact SCREEN/SCORES values. A bounded child batch
shares emissions; every unproved frontier is answered by the original exact HMM.
"""
from dataclasses import dataclass
import time
import numpy as np
from .. import execution
from ..store import CheckpointStore
from . import records,scoring,stage,forward,m1
from . import selective_m1 as search
from .selective_bounds import Geometry
from .compact_m1 import CompactBuilder


def _clock():return time.perf_counter(),time.process_time()
def _elapsed(start):return dict(wall=time.perf_counter()-start[0],cpu=time.process_time()-start[1])


@dataclass
class Component:
    resident: object
    geometry: Geometry
    zero: np.ndarray
    cache_range: object=None
    products: object=None
    powered: object=None

    def bound(self,children):
        self.cache_range=self.products=self.powered=None
        return self.geometry.upper(self.resident.projection,self.resident.terms,children,self.zero,execution.claim_threads)

    def exact(self,pairs,first,stop):
        p,t=self.resident.projection,self.resident.terms
        if self.cache_range!=(first,stop):
            execution.claim_threads();self.products,self.powered=t.common(p,first,stop)
            self.cache_range=(first,stop)
        return search.exact_pairs(p,t,pairs,self.zero,first,self.products,self.powered,execution.claim_threads)


def prepare(prepared,eligibility,config):
    result=[]
    for data in prepared.components:
        resident=scoring.resident(data,config,execution.claim_threads)
        resident.pair_masses=None  # later exposure rebuilds it; not an M1 factor
        p,t=resident.projection,resident.terms
        zero=search.zero(p,t,eligibility.children,execution.claim_threads)
        geometry=Geometry(data,p,eligibility.parents.any(0),execution.claim_threads)
        result.append(Component(resident,geometry,zero))
    return result


def worker(contig,state,phase,common,budget):
    at=_clock();timings={}
    if phase=='query' and not len(common['queries'].get(contig,())):
        return state,dict(value=np.empty(0),timings={})
    if state is not None:components=state
    else:
        store=CheckpointStore(common['root'],nthreads=budget)
        prepared=records.decode_prepared(store.contig(contig).get(records.PREPARED))
        components=prepare(prepared,common['eligibility'],common['config'])
        del prepared
        timings['prepare']=_elapsed(at)
    if phase=='setup':
        batch=min(min(x.resident.terms.batch,max(1,forward.COMMON_EMISSION_CACHE_BYTES//
                  (32*x.resident.projection.n_bins*x.resident.projection.n_states**2))) for x in components)
        result=dict(zero=np.sum(np.stack([x.zero for x in components]),axis=0),batch=min(32,batch))
    elif phase=='bound':
        at=_clock();children=common['children']
        active=np.flatnonzero(common['eligibility'].parents[children].any(1))
        parts=[x.bound(children[active]) for x in components]
        result=np.full((len(children),len(common['eligibility'].children)),-np.inf)
        result[active]=search.outward_sum(np.stack(parts))
        timings['bound']=_elapsed(at)
    elif phase=='query':
        rows=common['queries'].get(contig)
        if rows is None or not len(rows):result=np.empty(0)
        else:
            at=_clock();parts=[x.exact(rows,common['first'],common['stop']) for x in components]
            result=np.sum(np.stack(parts),axis=0);timings['exact']=_elapsed(at)
    else:raise ValueError(phase)
    timings['state_bytes']=stage._array_bytes(components)
    timings['retained']=common['retain']
    return (components if common['retain'] else None),dict(value=result,timings=timings)


def run(root,contigs,markers,eligibility,config,budget,*,batch_size=None,progress=lambda value:None,method="certified"):
    """Return (CompactM1, M0[C,N], numerical work/cost records).

    Batch width is a computational workspace control; it cannot limit candidates.
    Input and original dense checkpoint schemas are unchanged/read-only.
    """
    if method=="exact":
        return reference(root,contigs,markers,eligibility,config,budget)
    if method!="certified":raise ValueError(method)
    store=CheckpointStore(root,nthreads=budget);n=len(eligibility.children)
    estimates=[2*stage._scoring_bytes(store,c) for c in contigs]
    workers=execution.memory_worker_limit(estimates,len(contigs),label='PEDIGREE SELECTIVE')
    builder=CompactBuilder(n,markers,eligibility.parents.sum(1));costs=[];counts=[]
    # Full peak reservations stay across barriers only when the complete set
    # fits. Otherwise recompute factors on demand, with no held-state deadlock.
    retain=sum(estimates)<=execution.memory_governor().limit
    common=dict(root=str(root),eligibility=eligibility,config=config,retain=retain)
    def collect(phase,result):
        costs.extend(dict(contig=c,phase=phase,**r['timings']) for c,r in zip(contigs,result))
        return [r['value'] for r in result]
    with execution.ChromosomeExecutor(contigs,worker,budget=budget,max_workers=workers,
            task_bytes=estimates,state_bytes=estimates if retain else None,label='PEDIGREE SELECTIVE') as pool:
        setup=collect('setup',pool.run('setup',common))
        batch=min(x['batch'] for x in setup)
        if batch_size is not None:batch=min(batch,max(1,batch_size))
        zero=np.stack([x['zero'] for x in setup])
        for first in range(0,n,batch):
            stop=min(n,first+batch);children=np.arange(first,stop,dtype=np.int64)
            allowed=eligibility.parents[first:stop]
            if not np.any(allowed):continue
            upper=np.stack(collect('bound',pool.run('bound',dict(common,children=children))))
            def exact(requests):
                where={c:np.flatnonzero(requests[:,0]==j) for j,c in enumerate(contigs)}
                queries={c:requests[ix,1:] for c,ix in where.items() if len(ix)}
                values=collect('query',pool.run('query',dict(common,queries=queries,first=first,stop=stop)))
                output=np.empty(len(requests))
                for c,v in zip(contigs,values):output[where[c]]=v
                return output
            value=search.search_batch(upper,children,allowed,np.asarray(markers),config,exact)
            builder.add(value);counts.append(value['counts'])
            progress(dict(first=first,stop=stop,**value['counts']))
    return builder.finish(),zero,dict(batches=counts,phases=costs)


def _reference_worker(contig,state,phase,common,budget):
    store=CheckpointStore(common['root'],nthreads=budget)
    prepared=records.decode_prepared(store.contig(contig).get(records.PREPARED))
    screen,_=scoring.screen_chromosome(prepared,common['eligibility'],common['config'],execution.claim_threads)
    return None,(screen.pair_scores(),np.sum(np.stack([x.zero for x in screen.components]),axis=0))


def reference(root,contigs,markers,eligibility,config,budget):
    """Dense exact reference behind the same API; deliberately O(C N^2) memory."""
    store=CheckpointStore(root,nthreads=budget)
    estimates=[stage._scoring_bytes(store,c) for c in contigs]
    workers=execution.memory_worker_limit(estimates,len(contigs),label='PEDIGREE EXACT REFERENCE')
    with execution.ChromosomeExecutor(contigs,_reference_worker,budget=budget,max_workers=workers,
            task_bytes=estimates,label='PEDIGREE EXACT REFERENCE') as pool:
        results=pool.run('reference',dict(root=str(root),eligibility=eligibility,config=config))
    raw=np.stack([x[0] for x in results]);zero=np.stack([x[1] for x in results])
    _,keep,floors,best,cutoff=m1.genome_profile(raw,markers,eligibility,config)
    n=len(eligibility.children);builder=CompactBuilder(n,markers,eligibility.parents.sum(1))
    local=np.max(raw,axis=2)
    builder.add(dict(children=np.arange(n),best=best,cutoff=cutoff,chromosome_best=local,floors=floors,
                     retained=[np.flatnonzero(keep[c]) for c in range(n)],values=raw))
    return builder.finish(),zero,dict(reference=True,queried_chromosome_pairs=int(eligibility.parents.sum())*len(contigs))
