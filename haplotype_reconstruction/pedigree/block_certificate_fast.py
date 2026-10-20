"""Allocation-free arithmetic for G6's unchanged block envelope, private only."""
import math
import numpy as np
from numba import njit,prange
from .block_certificate import common_logs,dense_bridges,backward,EPS
from .forward import _site_likelihood

@njit(cache=True,parallel=True)
def power_common(common,children,exponent):
    count,bins,k,_=common.shape
    powered=np.empty((count,bins,2,k,k));scales=np.zeros((count,bins,2))
    for task in prange(count*bins):
        ci,b=task//bins,task%bins
        maximum=-np.inf
        for s in range(k):
            for t in range(k):
                x=common[ci,b,s,t]
                if np.isfinite(x):maximum=max(maximum,x)
        if not np.isfinite(maximum):maximum=0.
        for view in range(2):
            eta=exponent[children[ci],b] if view==0 else 1.
            scales[ci,b,view]=eta*maximum
            for s in range(k):
                for t in range(k):
                    x=common[ci,b,s,t]
                    powered[ci,b,view,s,t]=math.exp(eta*(x-maximum)) if np.isfinite(x) else np.nan
    return powered,scales

@njit(cache=True,parallel=True)
def block_upper(children,first,stop,site_first,refs,tv,mass,ref_mass,alt_low,alt_high,common,
                state_alt,gl,coefficient,indices,starts,stops,exponent,null_diag,null_off,mismatch,
                stay,alpha,beta,row_arm,column_arm,pivot,branch,excursion,powered,scales):
    groups=len(refs);k=len(state_alt);length=stop-first
    output=np.empty((len(children),groups))
    for task in prange(len(children)*groups):
        ci,g=task//groups,task%groups;c=children[ci];p=refs[g]
        emission=np.empty((length,2,k,k))
        for b in range(first,stop):
            for s in range(k):
                for t in range(k):
                    val=common[ci,b,s,t]
                    if np.isfinite(val):
                        emission[b-first,0,s,t]=powered[ci,b,0,s,t]
                        emission[b-first,1,s,t]=powered[ci,b,1,s,t]
                    else:
                        val=0.
                        for slot in range(starts[c,b],stops[c,b]):
                            site=indices[slot];v=site-site_first;other=state_alt[t,site]
                            lo=_site_likelihood(gl[c,site],coefficient[c,site],alt_low[g,v,s],other,mismatch)
                            hi=_site_likelihood(gl[c,site],coefficient[c,site],alt_high[g,v,s],other,mismatch)
                            pad=64*EPS*max(1.,3*np.max(gl[c,site]))
                            val+=math.log(max(lo,hi)+pad)
                        emission[b-first,0,s,t]=math.exp(exponent[c,b]*val-scales[ci,b,0])
                        emission[b-first,1,s,t]=math.exp(val-scales[ci,b,1])
        score=0.
        value=np.empty((k,k));work=np.empty_like(value);ref=np.empty_like(value)
        prefix=np.empty((k+1,k));suffix=np.empty_like(prefix)
        arm=np.empty(k);lo_v=np.empty(k);hi_v=np.empty(k);sum_v=np.empty(k)
        for view in range(2):
            value[:]=1.
            for b in range(stop-1,first-1,-1):
                local=b-first
                for s in range(k):
                    for t in range(k):value[s,t]*=emission[local,view,s,t]
                score+=.5*scales[ci,b,view]
                if b:
                    nd,no=null_diag[c,b-1],null_off[c,b-1]
                    for s in range(k):
                        if nd>=no:
                            total=0.
                            for t in range(k):total+=value[s,t]
                            for t in range(k):work[s,t]=(nd-no)*value[s,t]+no*total
                        else:
                            for t in range(k):
                                total=nd*value[s,t]
                                for u in range(k):
                                    if u!=t:total+=no*value[s,u]
                                work[s,t]=total
                    boundary=b-1;piv=pivot[p,boundary];kind=branch[p,boundary]
                    for t in range(k):
                        lo_v[t]=np.inf;hi_v[t]=0.;sum_v[t]=0.;arm[t]=0.
                        for s in range(k):
                            lo_v[t]=min(lo_v[t],work[s,t]);hi_v[t]=max(hi_v[t],work[s,t]);sum_v[t]+=work[s,t]
                            if piv>=0:arm[t]+=row_arm[p,boundary,s]*work[s,t]
                    if kind==3:
                        for s in range(k):
                            for t in range(k):
                                ref[s,t]=stay[p,boundary,s]*work[s,t]
                                if s==piv:ref[s,t]+=arm[t]
                                else:ref[s,t]+=alpha[p,boundary,s]*work[piv,t]
                    else:
                        prefix[0,:]=0.;suffix[k,:]=0.
                        for s in range(k):
                            for t in range(k):prefix[s+1,t]=prefix[s,t]+beta[p,boundary,s]*work[s,t]
                        for s in range(k-1,-1,-1):
                            for t in range(k):suffix[s,t]=suffix[s+1,t]+beta[p,boundary,s]*work[s,t]
                        for s in range(k):
                            for t in range(k):
                                x=stay[p,boundary,s]*work[s,t]+alpha[p,boundary,s]*(prefix[s,t]+suffix[s+1,t])
                                if piv>=0:
                                    x+=column_arm[p,boundary,s]*work[piv,t]
                                    if s==piv:x+=arm[t]
                                ref[s,t]=x
                    ex=excursion[p,boundary]
                    for s in range(k):
                        for t in range(k):
                            r=ref[s,t]
                            if ex>0.:r=(1-ex)*r+ex*sum_v[t]/k
                            bounded=r/ref_mass[g,local,s]+tv[g,local,s]*(hi_v[t]-lo_v[t])
                            value[s,t]=mass[g,local,s]*min(hi_v[t],bounded)
                maximum=np.max(value)
                if maximum<=0. or not np.isfinite(maximum):
                    score=np.inf;break
                value/=maximum;score+=.5*math.log(maximum)
        output[ci,g]=score
    return output

@njit(cache=True,parallel=True)
def global_common_logs(children,state_alt,gl,coefficient,indices,starts,stops,mismatch):
    """Relax only uncertain parental alleles to [0,1]; retain the null chain.

    All groups can now share the complete emission operator. Exact known
    founder alleles stay fixed. This is a weaker bound, never a point model.
    """
    bins=starts.shape[1];k=len(state_alt)
    result=np.empty((len(children),bins,k,k))
    for item in prange(len(children)*bins):
        ci,b=item//bins,item%bins;c=children[ci]
        for s in range(k):
            for t in range(k):
                total=0.
                for slot in range(starts[c,b],stops[c,b]):
                    site=indices[slot];a=state_alt[s,site];other=state_alt[t,site]
                    if a==0. or a==1.:
                        likelihood=_site_likelihood(gl[c,site],coefficient[c,site],a,other,mismatch)
                    else:
                        likelihood=max(_site_likelihood(gl[c,site],coefficient[c,site],0.,other,mismatch),
                                       _site_likelihood(gl[c,site],coefficient[c,site],1.,other,mismatch))
                    total+=math.log(likelihood+64*EPS*max(1.,3*np.max(gl[c,site])))
                result[ci,b,s,t]=total
    return result
