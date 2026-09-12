"""Phase-HMM inference with exact contraction of zero-transition bins."""


import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def binary_chain_messages(emission, switch_probability, anchor=-1, diagnostics=True):
    """Scaled binary forward/backward: marginals, site cavities and switches.

    The cavity omits the current emission, NOT the root's arbitrary phase
    gauge. At a structural phase boundary theta=.5 severs that phase chain.
    """
    length = emission.shape[0]
    alpha = np.empty((length, 2), dtype=np.float64)
    predicted = np.empty((length, 2), dtype=np.float64)
    beta = np.ones((length, 2), dtype=np.float64)
    for site in range(length):
        if site == 0:
            p0, p1 = 0.5, 0.5
        else:
            theta = switch_probability[site - 1]
            p0 = alpha[site - 1, 0] * (1-theta) + alpha[site - 1, 1] * theta
            p1 = alpha[site - 1, 1] * (1-theta) + alpha[site - 1, 0] * theta
        if site == anchor:
            p1 = 0.0
        predicted[site, 0], predicted[site, 1] = p0, p1
        a0, a1 = p0 * emission[site, 0], p1 * emission[site, 1]
        total = a0 + a1
        alpha[site, 0], alpha[site, 1] = a0 / total, a1 / total
    for site in range(length - 2, -1, -1):
        e0 = emission[site + 1, 0] * beta[site + 1, 0]
        e1 = emission[site + 1, 1] * beta[site + 1, 1]
        if site + 1 == anchor:
            e1 = 0.0
        theta = switch_probability[site]
        b0, b1 = (1-theta)*e0 + theta*e1, theta*e0 + (1-theta)*e1
        total = b0+b1
        beta[site, 0], beta[site, 1] = b0/total, b1/total
    posterior = np.empty_like(alpha) if diagnostics else np.empty((0,0),dtype=np.float64)
    cavity = np.empty_like(alpha)
    switches = np.empty(max(0, length - 1) if diagnostics else 0, dtype=np.float64)
    for site in range(length):
        c0 = predicted[site, 0] * beta[site, 0]
        c1 = predicted[site, 1] * beta[site, 1]
        total = c0+c1
        cavity[site, 0], cavity[site, 1] = c0/total, c1/total
        if diagnostics:
            p0, p1 = c0*emission[site, 0], c1*emission[site, 1]
            posterior[site, 0], posterior[site, 1] = p0/(p0+p1), p1/(p0+p1)
        if diagnostics and site + 1 < length:
            e0 = emission[site + 1, 0] * beta[site + 1, 0]
            e1 = emission[site + 1, 1] * beta[site + 1, 1]
            if site + 1 == anchor:
                e1 = 0.0
            theta = switch_probability[site]
            jump = theta*(alpha[site, 0]*e1 + alpha[site, 1]*e0)
            stay = (1-theta)*(alpha[site, 0]*e0 + alpha[site, 1]*e1)
            switches[site] = jump/(jump+stay)
    return posterior, cavity, switches



@njit(cache=True)
def binary_map(emission,theta,anchor=-1):
    n=len(emission)
    scores=np.zeros(2,dtype=np.float64)
    back=np.zeros((n,2),dtype=np.int8)
    for i in range(n):
        new=np.empty(2,dtype=np.float64)
        for state in range(2):
            if i==anchor and state==1:
                new[state]=-np.inf;continue
            if i==0:
                new[state]=math.log(max(emission[i,state],1e-300))-math.log(2.)
                continue
            stay=math.log(max(1-theta[i-1],1e-300))
            jump=math.log(theta[i-1]) if theta[i-1]>0 else -np.inf
            a=scores[state]+stay;b=scores[1-state]+jump
            if a>=b:new[state]=a;back[i,state]=state
            else:new[state]=b;back[i,state]=1-state
            new[state]+=math.log(max(emission[i,state],1e-300))
        scores=new
    path=np.empty(n,dtype=np.int8)
    state=0 if scores[0]>=scores[1] else 1
    for i in range(n-1,-1,-1):
        path[i]=state
        if i:state=back[i,state]
    return path



@njit(cache=True)
def phase_segments(theta,sites):
    count=1
    for t in theta:
        if t>0:count+=1
    starts=np.empty(count+1,dtype=np.int64);trans=np.empty(count-1)
    starts[0]=0;starts[count]=sites;cursor=1
    for site in range(1,sites):
        if theta[site-1]>0:
            starts[cursor]=site;trans[cursor-1]=theta[site-1];cursor+=1
    return starts,trans



@njit(cache=True)
def compact_phase(emissions,starts,trans,anchor,paths):
    bins=len(starts)-1;compact=np.empty((bins,2));anchor_bin=-1
    for block in range(bins):
        a=0.;b=0.
        for site in range(starts[block],starts[block+1]):
            a+=math.log(emissions[site,0]);b+=math.log(emissions[site,1])
        if starts[block]<=anchor<starts[block+1]:
            # Condition before scaling: otherwise the allowed state can
            # underflow against a forbidden state in a very long interval.
            anchor_bin=block;b=-np.inf
        maximum=max(a,b)
        compact[block,0]=math.exp(a-maximum);compact[block,1]=math.exp(b-maximum)
    posterior,_,switches=binary_chain_messages(compact,trans,anchor_bin,True)
    path=binary_map(compact,trans,anchor_bin) if paths else np.empty(0,dtype=np.int8)
    return posterior,switches,path



@njit(cache=True,inline="always")
def emission_pair(belief_row,orientation_value,phase_value,phase_error):
    orient=orientation_value
    if orient<0:
        return .5,.5
    p=phase_error+(1-2*phase_error)*phase_value
    f1=p if orient==0 else 1-p;f2=1-p if orient==0 else p
    q0=math.exp(float(belief_row[0]))/.5
    q1=math.exp(float(belief_row[1]))/f1
    q2=math.exp(float(belief_row[2]))/f2
    q3=math.exp(float(belief_row[3]))/.5
    total=((q0+q1)+q2)+q3;q0/=total;q1/=total;q2/=total;q3/=total
    matching_zero=phase_error+(1-2*phase_error)
    matching_one=phase_error
    e10=matching_zero if orient==0 else 1-matching_zero
    e20=1-matching_zero if orient==0 else matching_zero
    e11=matching_one if orient==0 else 1-matching_one
    e21=1-matching_one if orient==0 else matching_one
    e0=((q0*.5+q1*e10)+q2*e20)+q3*.5
    e1=((q0*.5+q1*e11)+q2*e21)+q3*.5
    return e0,e1



@njit(cache=True,parallel=True)
def phase_update(belief_log,orientation,phase,phase_theta,anchors,phase_error,damping,write_messages):
    samples,sites,_=belief_log.shape
    probability=np.empty((0,0) if write_messages else (samples,sites),dtype=np.float32)
    delta=np.zeros(samples,dtype=np.float64)
    paths=np.empty((0,0) if write_messages else (samples,sites),dtype=np.int8)
    links=np.empty((0,0) if write_messages else (samples,max(0,sites-1)),dtype=np.float32)
    starts,trans=phase_segments(phase_theta,sites)
    active=np.ones(samples,dtype=np.bool_)
    if write_messages:
        for sample in prange(samples):
            if anchors[sample]<0:
                informative=False
                for site in range(sites):
                    if orientation[sample,site]>=0:informative=True;break
                if not informative:
                    active[sample]=False
                    for site in range(sites):
                        old=phase[sample,site];new=(1-damping)*old+damping*.5
                        delta[sample]=max(delta[sample],abs(new-old));phase[sample,site]=new
    active_samples=np.flatnonzero(active)
    for index in prange(len(active_samples)):
        sample=active_samples[index];emissions=np.empty((sites,2))
        for site in range(sites):
            emissions[site,0],emissions[site,1]=emission_pair(
                belief_log[sample,site],orientation[sample,site],phase[sample,site],phase_error)
        posterior,switches,path=compact_phase(emissions,starts,trans,anchors[sample],not write_messages)
        for block in range(len(starts)-1):
            p0,p1=posterior[block,0],posterior[block,1]
            for site in range(starts[block],starts[block+1]):
                if write_messages:
                    c0=p0/emissions[site,0];c1=p1/emissions[site,1]
                    old=phase[sample,site];new=(1-damping)*old+damping*c0/(c0+c1)
                    delta[sample]=max(delta[sample],abs(new-old));phase[sample,site]=new
                else:
                    probability[sample,site]=p0;paths[sample,site]=path[block]
                    if site+1<sites:links[sample,site]=1.
            if not write_messages and block+1<len(starts)-1:
                boundary=starts[block+1]-1
                links[sample,boundary]=switches[block] if path[block]!=path[block+1] else 1-switches[block]
    return probability,delta,paths,links



@njit(cache=True,parallel=True)
def phase_context_view(belief,reference,orientation,phase,phase_theta,anchors,parent_counts,
                       phase_error,phase_threshold,correct_incomplete):
    """Same phase MAP/context as a full family view, with no dense posterior."""
    samples,sites,_=reference.shape
    context=np.empty_like(reference)
    inferred=np.empty((samples,sites),dtype=np.int8)
    emitted=np.empty_like(inferred)
    starts,trans=phase_segments(phase_theta,sites)
    high=np.float32(phase_threshold);low=np.float32(1-phase_threshold)
    for sample in prange(samples):
        has_frame=False
        for site in range(sites):
            if reference[sample,site,0]!=reference[sample,site,1]:
                has_frame=True;break
        emissions=np.empty((sites,2))
        for site in range(sites):
            alignment=phase[sample,site] if parent_counts[sample]==0 else .5
            emissions[site,0],emissions[site,1]=emission_pair(
                belief[sample,site],orientation[sample,site],alignment,phase_error)
        posterior,_,path=compact_phase(emissions,starts,trans,anchors[sample],True)
        for block in range(len(starts)-1):
            inferred_flip=path[block]
            displayed_flip=inferred_flip if correct_incomplete or parent_counts[sample]>=2 else 0
            confidence=np.float32(posterior[block,0])
            phase_supported=((inferred_flip==0 and confidence>=high) or
                             (inferred_flip==1 and confidence<=low))
            for site in range(starts[block],starts[block+1]):
                inferred[sample,site]=inferred_flip;emitted[sample,site]=displayed_flip
                a=reference[sample,site,displayed_flip];b=reference[sample,site,1-displayed_flip]
                # Called scaffold alleles are unchanged. Only missing sites
                # require genotype probabilities for conditional context.
                if a<0 or b<0:
                    q0=np.exp(belief[sample,site,0]);q1=np.exp(belief[sample,site,1])
                    q2=np.exp(belief[sample,site,2]);q3=np.exp(belief[sample,site,3])
                    if inferred_flip!=displayed_flip:q1,q2=q2,q1
                    if (a>=0)!=(b>=0):
                        known=a if a>=0 else b;het=q1+q2
                        alt=het/(q0+het) if known==0 else q3/(het+q3)
                        if max(alt,1-alt)>=.98:
                            if a<0:a=int(alt>=.5)
                            else:b=int(alt>=.5)
                    else:
                        first,second=float(q2+q3),float(q1+q3)
                        if not phase_supported and has_frame:first=second=(first+second)/2.
                        if max(first,1-first)>=.98:a=int(first>=.5)
                        if max(second,1-second)>=.98:b=int(second>=.5)
                context[sample,site,displayed_flip]=a
                context[sample,site,1-displayed_flip]=b
    return context,emitted,inferred
