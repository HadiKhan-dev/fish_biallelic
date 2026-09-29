"""Sample-contiguous workspace and kernels for short-block wide founder beams.

The conditional-path caller owns these arrays for its entire query. Ordinary
window and macro scans retain their original kernels. Emissions are gathered
once per block and reused by score/retain; no founder states, beam candidates
or bins are removed. Candidate sums keep sample order exactly, without fastmath.

Performance evidence is the 1024-wide two-direction AulStu workload. The caller
uses a deliberately conservative width gate, not a universal crossover claim.
"""
import numpy as np
from numba import njit,prange
from .beam_kernels import ordered


def workspace(blocks, samples, states, width, max_choices):
    """Keep the sample-contiguous DP layout for the whole conditional query."""
    dp = np.empty((width, states, samples))
    dp[0] = 0.0
    alternate = np.empty_like(dp)
    values = np.empty((samples, width * max_choices))
    uppers = np.empty_like(values)
    ancestry = np.empty((blocks, width), np.int64)
    rows = np.empty_like(ancestry)
    return dp, alternate, values, uppers, ancestry, rows


@njit(cache=True, parallel=True, nogil=True)
def direct_score_sum(source):
    """Parallelize independent candidates, never their sample addition order."""
    scores = np.empty(len(source))
    for candidate in prange(len(source)):
        total = 0.0
        for sample in range(source.shape[1]):
            total += source[candidate, sample]
        scores[candidate] = total
    return scores


@njit(cache=True,parallel=True,nogil=True)
def prepare(data,base,size,bins,known,block,choices,begin,end,reverse,first,second):
    samples,founders=len(data),len(known)+1
    background=np.empty((bins,len(first),samples))
    focal=np.empty((bins,end-begin,founders,samples))
    for sample in prange(samples):
        for state in range(len(first)):
            if second[state]==founders-1:continue
            address=base+(known[first[state],block]*size+known[second[state],block])*bins
            for step in range(bins):
                marker=bins-1-step if reverse else step
                background[step,state,sample]=data[sample,address+marker]
        for branch in range(end-begin):
            choice=choices[begin+branch]
            for mate_row in range(founders):
                mate=choice if mate_row==founders-1 else known[mate_row,block]
                address=base+(mate*size+choice)*bins
                for step in range(bins):
                    marker=bins-1-step if reverse else step
                    focal[step,branch,mate_row,sample]=data[sample,address+marker]
    return background,focal

@njit(cache=True,parallel=True,nogil=True)
def score(bg0,bg1,focal0,focal1,width,known,choices,begin,end,penalty,first,second,
          dp,beams,flank,upper_flank,two_flanks,values,uppers):
    samples,focal=dp.shape[2],len(known)
    affected=np.flatnonzero(second==focal)
    branches=end-begin
    for beam in prange(beams):
        rows=np.empty((len(affected),samples))
        switched=np.full(samples,-np.inf)
        bg_best=np.full(samples,-np.inf)
        bg_stay=np.full(samples,-np.inf)
        bg_enter=np.full(samples,-np.inf)
        up_stay=np.full(samples,-np.inf)
        up_enter=np.full(samples,-np.inf)
        best=np.empty(samples)
        changed=np.empty(samples)
        value=np.empty(samples)
        upvalue=np.empty(samples)
        for state in range(len(first)):
            for sample in range(samples):
                switched[sample]=max(switched[sample],dp[beam,state,sample])
        for sample in range(samples):
            switched[sample]-=penalty
        for state in range(len(first)):
            if second[state]==focal:
                continue
            for sample in range(samples):
                v=max(dp[beam,state,sample],switched[sample])+bg0[state,sample]
                bg_best[sample]=max(bg_best[sample],v)
                if width==1:
                    bg_stay[sample]=max(bg_stay[sample],v+flank[state,sample])
                    if two_flanks:
                        up_stay[sample]=max(up_stay[sample],v+upper_flank[state,sample])
                else:
                    e=bg1[state,sample]
                    bg_stay[sample]=max(bg_stay[sample],v+e+flank[state,sample])
                    bg_enter[sample]=max(bg_enter[sample],e+flank[state,sample])
                    if two_flanks:
                        up_stay[sample]=max(up_stay[sample],v+e+upper_flank[state,sample])
                        up_enter[sample]=max(up_enter[sample],e+upper_flank[state,sample])
        for candidate in range(branches):
            for sample in range(samples):
                best[sample]=bg_best[sample]
            for a,state in enumerate(affected):
                for sample in range(samples):
                    v=max(dp[beam,state,sample],switched[sample])+focal0[candidate,a,sample]
                    rows[a,sample]=v
                    best[sample]=max(best[sample],v)
            for sample in range(samples):
                changed[sample]=best[sample]-penalty
                value[sample]=(bg_stay[sample] if width==1 else
                    max(bg_stay[sample],changed[sample]+bg_enter[sample]))
                upvalue[sample]=(up_stay[sample] if width==1 else
                    max(up_stay[sample],changed[sample]+up_enter[sample]))
            for a,state in enumerate(affected):
                for sample in range(samples):
                    row=rows[a,sample]
                    if width==2:
                        row=max(row,changed[sample])+focal1[candidate,a,sample]
                    value[sample]=max(value[sample],row+flank[state,sample])
                    if two_flanks:
                        upvalue[sample]=max(upvalue[sample],row+upper_flank[state,sample])
            index=beam*branches+candidate
            for sample in range(samples):
                values[index,sample]=value[sample]
                if two_flanks:
                    uppers[index,sample]=upvalue[sample]


@njit(cache=True,parallel=True,nogil=True)
def retain(background,focal_emission,focal,penalty,first,second,dp,order,count,output):
    bins,states,samples=background.shape
    branches=focal_emission.shape[1]
    for kept in prange(count):
        candidate=order[kept]
        parent,branch=candidate//branches,candidate%branches
        switched=np.full(samples,-np.inf)
        for state in range(states):
            for sample in range(samples):
                switched[sample]=max(switched[sample],dp[parent,state,sample])
        for sample in range(samples):switched[sample]-=penalty
        for state in range(states):
            for sample in range(samples):
                emission=(focal_emission[0,branch,first[state],sample]
                    if second[state]==focal else background[0,state,sample])
                output[kept,state,sample]=max(dp[parent,state,sample],switched[sample])+emission
        for step in range(1,bins):
            switched[:]=-np.inf
            for state in range(states):
                for sample in range(samples):
                    switched[sample]=max(switched[sample],output[kept,state,sample])
            for sample in range(samples):switched[sample]-=penalty
            for state in range(states):
                for sample in range(samples):
                    emission=(focal_emission[step,branch,first[state],sample]
                        if second[state]==focal else background[step,state,sample])
                    output[kept,state,sample]=max(output[kept,state,sample],switched[sample])+emission

@njit(cache=True, nogil=True)
def chunk(
    data,
    bases,
    sizes,
    bins,
    known,
    choices,
    offsets,
    penalty,
    reverse,
    first,
    second,
    suffix,
    upper_suffix,
    two_flanks,
    upper_start,
    best_seen,
    ranking,
    start,
    stop,
    dp,
    alternate,
    beams,
    beam_width,
    values,
    uppers,
    ancestry,
    local_rows,
    trace_upper,
    trace_equivalent
):
    transposed_values = np.empty((values.shape[1],len(data)))
    transposed_uppers = np.empty_like(transposed_values)
    last_scores = np.empty(0)
    last_order = np.empty(0, np.int64)
    equivalent = True
    for step in range(start, stop):
        block = len(bins) - 1 - step if reverse else step
        begin, end = (offsets[block], offsets[block + 1])
        branches = end - begin
        flank = suffix[step + 1]
        upper_flank = upper_suffix[step - upper_start + 1] if two_flanks else flank
        background, focal_emission = prepare(data,bases[block],sizes[block],
            bins[block],known,block,choices,begin,end,reverse,first,second)
        score(background[0],background[-1],focal_emission[0],focal_emission[-1],
            bins[block],known,choices,begin,end,penalty,first,second,dp,beams,
            np.ascontiguousarray(flank.T),np.ascontiguousarray(upper_flank.T),
            two_flanks,transposed_values,transposed_uppers)
        length = beams * branches
        if length*len(data)*8 >= 8*1024**2:
            # Each candidate retains sample0..N-1 addition order, without
            # restoring a second full matrix layout first.
            scores = direct_score_sum(transposed_values[:length])
            upper = (direct_score_sum(transposed_uppers[:length])
                     if two_flanks else np.zeros(length))
        else:
            values[:,:length] = transposed_values[:length].T
            if two_flanks:
                uppers[:,:length] = transposed_uppers[:length].T
            scores = np.zeros(length)
            upper = np.zeros(length)
            for sample in range(len(data)):
                for candidate in range(length):
                    scores[candidate] += values[sample, candidate]
                    if two_flanks:
                        upper[candidate] += uppers[sample, candidate]
        if two_flanks:
            remaining = np.max(upper)
            trace_upper[step] = remaining
            margin = 1e-10 * max(1.0, abs(remaining), abs(best_seen))
            if remaining + margin <= best_seen + 1e-06:
                return (dp, alternate, beams, scores, last_order, True, equivalent)
        order = ordered(scores, upper, ranking, beam_width)
        count = min(beam_width, length)
        if two_flanks:
            ordinary = order if ranking == 0 else ordered(scores, upper, 0, beam_width)
            tied = order if ranking == 1 else ordered(scores, upper, 1, beam_width)
            for i in range(count):
                if ordinary[i] != tied[i]:
                    equivalent = False
                    trace_equivalent[step] = False
        for i in range(count):
            ancestry[step, i] = order[i] // branches
            local_rows[step, i] = choices[begin + order[i] % branches]
        retain(background,focal_emission,len(known),penalty,
            first,second,dp,order,count,alternate)
        dp, alternate = (alternate, dp)
        beams = count
        last_scores = scores
        last_order = order[:count]
    return (dp, alternate, beams, last_scores, last_order, False, equivalent)
