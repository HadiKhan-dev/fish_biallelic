"""Selected exact M1 kernel and batched one-sided search under the G4 model."""
import numpy as np
from numba import njit,prange
from . import forward,m1


def outward_sum(values):
    total=np.sum(values,axis=0)
    with np.errstate(invalid='ignore'):
        guard=8*np.finfo(float).eps*values.shape[0]*np.sum(np.abs(values),axis=0)
        return np.where(np.isfinite(total),np.nextafter(total+guard,np.inf),total)


@njit(cache=True,parallel=True)
def _dyads(pairs,pi,candidate_alt,stay,alpha,beta,row_arm,column_arm,pivot,excursion,branch,
           available,state_alt,gl,coefficient,indices,starts,stops,exponent,null_diag,null_off,
           mismatch,zero,products,first_child,powered):
    output=np.empty(len(pairs))
    for i in prange(len(pairs)):
        c,p=pairs[i]
        if not available[p]:
            output[i]=zero[c]
        else:
            a,b=forward._score_path(c,p,0,1,pi,candidate_alt,stay,alpha,beta,row_arm,column_arm,pivot,
                excursion,branch,state_alt,gl,coefficient,indices,starts,stops,exponent,null_diag,null_off,
                mismatch,products,first_child,powered)
            output[i]=.5*(a+b)
    return output


def terms_args(p,t):
    return (p.state_alt,t.gl,t.coefficient,t.site_index,t.site_start,t.site_stop,t.exponent,
            t.null_diagonal,t.null_off,t.mismatch)


def zero(p,t,eligible_children,claim_threads):
    claim_threads()
    a,b=forward._m0_kernel(*forward._model(p),*terms_args(p,t),np.ascontiguousarray(eligible_children))
    return .5*(a+b)


def exact_pairs(p,t,pairs,zero_scores,first_child,products,powered,claim_threads):
    claim_threads()
    return _dyads(np.ascontiguousarray(pairs,dtype=np.int64),*forward._model(p),p.available,*terms_args(p,t),
                  zero_scores,products,first_child,powered)


def search_batch(upper,children,eligible,markers,config,score):
    """Certify B, the complete genome gate and all b in bulk exact-query phases.

    upper: C x batch x N; eligible: batch x N. score receives triples of
    (chromosome index, GLOBAL child, parent) and returns exact raw component sums.
    No raw score is inspected before requested. Complete bound ties are queried.
    Scratch is O(C batch N); no candidate cap or numerical ambiguity fallback.
    """
    upper=np.where(eligible[None],upper,-np.inf)
    contigs,batch,n=upper.shape;children=np.asarray(children,np.int64)
    values=np.full_like(upper,np.nan);known=np.zeros(upper.shape,bool)
    best=np.full(batch,-np.inf);cutoff=best.copy();local=np.full((contigs,batch),-np.inf)
    total_upper=outward_sum(upper);shift=m1.LOG_FLOOR*float(m1.temperature(np.sum(markers),config))
    counts=dict(global_queries=0,local_max_queries=0,rounds=0)

    def query(mask,kind):
        rows=np.argwhere(mask&~known)
        if not len(rows):return
        requests=rows.copy();requests[:,1]=children[rows[:,1]]
        exact=np.asarray(score(requests),dtype=float)
        if exact.shape!=(len(rows),) or not np.all(np.isfinite(exact)):
            raise ValueError('selected M1 query returned nonfinite or malformed scores')
        if np.any(exact>upper[tuple(rows.T)]):
            raise AssertionError('selected exact M1 exceeds its upper certificate')
        values[tuple(rows.T)]=exact;known[tuple(rows.T)]=True
        counts[kind]+=len(rows);counts['rounds']+=1

    def genome_rows(mask):
        return np.broadcast_to(mask,(contigs,batch,n))

    def update_best():
        for c in range(batch):
            completed=np.flatnonzero(eligible[c]&known[:,c].all(0))
            if len(completed):
                # This indexed layout and summation match G4 genome_profile.
                best[c]=np.max(np.sum(values[:,c,completed],axis=0))
                cutoff[c]=best[c]+shift

    # First complete maximum-upper tie class, with no index-based tie split.
    seed=np.zeros((batch,n),bool)
    for c in range(batch):
        parents=np.flatnonzero(eligible[c])
        if len(parents):seed[c,parents]=total_upper[c,parents]==np.max(total_upper[c,parents])
    query(genome_rows(seed),'global_queries');update_best()
    counts['seed_queries']=counts['global_queries']
    while True:
        guard=32*np.finfo(float).eps*contigs*np.maximum(1.,np.abs(best))
        contenders=eligible&~known.all(0)&(total_upper>=best[:,None]-guard[:,None])
        if not np.any(contenders):break
        query(genome_rows(contenders),'global_queries');update_best()
    counts['B_frontier_queries']=counts['global_queries']-counts['seed_queries']
    before_floor=counts['global_queries']
    guard=32*np.finfo(float).eps*contigs*np.maximum.reduce([np.ones(batch),np.abs(best),np.full(batch,abs(shift))])
    frontier=eligible&~known.all(0)&(total_upper>=cutoff[:,None]-guard[:,None])
    query(genome_rows(frontier),'global_queries');update_best()
    counts['floor_frontier_queries']=counts['global_queries']-before_floor
    retained=[]
    for c in range(batch):
        complete=np.flatnonzero(eligible[c]&known[:,c].all(0))
        if len(complete):
            totals=np.sum(values[:,c,complete],axis=0)
            retained.append(complete[totals>=cutoff[c]])
            for j in range(contigs):local[j,c]=np.max(values[j,c,complete])
        else:retained.append(np.empty(0,np.int64))
    # Every not-yet-known score that could reach the current maximum is exact.
    # After this query no missing score can reach the (possibly raised) maximum.
    pending=eligible[None]&~known&(upper>=local[:,:,None])
    query(pending,'local_max_queries')
    for c in range(batch):
        parents=np.flatnonzero(eligible[c])
        if len(parents):
            for j in range(contigs):
                local[j,c]=np.max(values[j,c,parents[known[j,c,parents]]])
                missing=parents[~known[j,c,parents]]
                assert np.all(upper[j,c,missing]<local[j,c])
    temp=m1.temperature(markers,config);floors=np.full_like(local,-np.inf)
    for c in range(batch):
        if np.isfinite(best[c]):
            floors[:,c]=local[:,c]-temp/np.sum(temp)*(np.sum(local[:,c])-cutoff[c])
    counts['nonfinite_upper_pairs']=int(np.count_nonzero(eligible[None]&~np.isfinite(upper)))
    counts['eligible_chromosome_pairs']=int(eligible.sum())*contigs
    counts['queried_chromosome_pairs']=int(known.sum())
    counts['retained_pairs']=sum(map(len,retained))
    return dict(children=children,best=best,cutoff=cutoff,chromosome_best=local,floors=floors,
                retained=retained,values=values,known=known,counts=counts)
