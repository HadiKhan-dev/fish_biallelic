"""Private G6 block/group induced-norm upper bound. No production callers."""
import math
import numpy as np
from numba import njit, prange
from .bounds import vector_bridge
from .forward import _site_likelihood
EPS=np.finfo(np.float64).eps

@njit(cache=True, parallel=True)
def dense_bridges(stay,alpha,beta,row_arm,column_arm,pivot,branch,excursion):
    n,bins,k=stay.shape
    out=np.empty((n,bins,k,k))
    for item in prange(n*bins):
        p,b=item//bins,item%bins
        for source in range(k):
            unit=np.zeros(k);unit[source]=1.
            out[p,b,source]=vector_bridge(unit,stay[p,b],alpha[p,b],beta[p,b],row_arm[p,b],column_arm[p,b],pivot[p,b],branch[p,b],excursion[p,b])
    return out

@njit(cache=True)
def backward(value,stay,alpha,beta,row_arm,column_arm,pivot,branch,excursion):
    k=len(value);out=np.empty(k)
    if branch==3:
        arm=0.
        for i in range(k):arm+=row_arm[i]*value[i]
        for i in range(k):
            out[i]=stay[i]*value[i]
            if i==pivot:out[i]+=arm
            else:out[i]+=alpha[i]*value[pivot]
    else:
        prefix=np.zeros(k+1);suffix=np.zeros(k+1)
        for i in range(k):prefix[i+1]=prefix[i]+beta[i]*value[i]
        for i in range(k-1,-1,-1):suffix[i]=suffix[i+1]+beta[i]*value[i]
        arm=0.
        if pivot>=0:
            for i in range(k):arm+=row_arm[i]*value[i]
        for i in range(k):
            out[i]=stay[i]*value[i]+alpha[i]*(prefix[i]+suffix[i+1])
            if pivot>=0:
                out[i]+=column_arm[i]*value[pivot]
                if i==pivot:out[i]+=arm
    if excursion>0.:
        share=excursion*np.sum(value)/k
        for i in range(k):out[i]=(1-excursion)*out[i]+share
    return out

@njit(cache=True,parallel=True)
def common_logs(children,state_alt,gl,coefficient,indices,starts,stops,mismatch):
    bins=starts.shape[1];k=len(state_alt)
    result=np.empty((len(children),bins,k,k))
    for item in prange(len(children)*bins):
        ci,b=item//bins,item%bins;c=children[ci]
        for s in range(k):
            for t in range(k):
                total=0.
                for slot in range(starts[c,b],stops[c,b]):
                    site=indices[slot];a=state_alt[s,site]
                    if a!=0. and a!=1.:
                        total=np.nan;break
                    likelihood=_site_likelihood(gl[c,site],coefficient[c,site],a,state_alt[t,site],mismatch)
                    total+=math.log(likelihood+64*EPS*max(1.,3*np.max(gl[c,site])))
                result[ci,b,s,t]=total
    return result

@njit(cache=True,parallel=True)
def block_upper(children,first,stop,site_first,refs,tv,mass,ref_mass,alt_low,alt_high,common,
                state_alt,gl,coefficient,indices,starts,stops,exponent,null_diag,null_off,mismatch,
                stay,alpha,beta,row_arm,column_arm,pivot,branch,excursion):
    groups=len(refs);k=len(state_alt);length=stop-first
    output=np.empty((len(children),groups))
    for task in prange(len(children)*groups):
        ci,g=task//groups,task%groups;c=children[ci];p=refs[g]
        logs=np.empty((length,k,k))
        for b in range(first,stop):
            for s in range(k):
                for t in range(k):
                    val=common[ci,b,s,t]
                    if not np.isfinite(val):
                        val=0.
                        for slot in range(starts[c,b],stops[c,b]):
                            site=indices[slot];v=site-site_first;other=state_alt[t,site]
                            lo=_site_likelihood(gl[c,site],coefficient[c,site],alt_low[g,v,s],other,mismatch)
                            hi=_site_likelihood(gl[c,site],coefficient[c,site],alt_high[g,v,s],other,mismatch)
                            pad=64*EPS*max(1.,3*np.max(gl[c,site]))
                            val+=math.log(max(lo,hi)+pad)
                    logs[b-first,s,t]=val
        score=0.
        for view in range(2):
            value=np.ones((k,k));work=np.empty_like(value)
            for b in range(stop-1,first-1,-1):
                local=b-first;eta=exponent[c,b] if view==0 else 1.
                mx=np.max(logs[local])*eta
                for s in range(k):
                    for t in range(k):value[s,t]*=math.exp(eta*logs[local,s,t]-mx)
                score+=.5*mx
                if b:
                    nd,no=null_diag[c,b-1],null_off[c,b-1]
                    # Positive diagonal/off decomposition (never cancellation).
                    for s in range(k):
                        for t in range(k):
                            total=nd*value[s,t]
                            for u in range(k):
                                if u!=t:total+=no*value[s,u]
                            work[s,t]=total
                    for t in range(k):
                        vec=work[:,t].copy();low=np.min(vec);high=np.max(vec)
                        ref=backward(vec,stay[p,b-1],alpha[p,b-1],beta[p,b-1],row_arm[p,b-1],column_arm[p,b-1],pivot[p,b-1],branch[p,b-1],excursion[p,b-1])
                        for s in range(k):
                            bound=ref[s]/ref_mass[g,local,s]+tv[g,local,s]*(high-low)
                            value[s,t]=mass[g,local,s]*min(high,bound)
                maximum=np.max(value)
                if maximum<=0. or not np.isfinite(maximum):
                    score=np.inf;break
                value/=maximum;score+=.5*math.log(maximum)
        # The component pad is applied after composing all blocks, also covering
        # row normalization, TV construction, libm and alternate reduction order.
        output[ci,g]=score
    return output
