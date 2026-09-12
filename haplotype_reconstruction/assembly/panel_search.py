"""Bounded-refit assembly search using exact fixed-painting edit statistics.

Conditional gains generate proposals, not reoptimized likelihoods. Every
accepted panel is checked by the existing full Viterbi/BIC objective. Unlike
the previous broader search, this heuristic can miss beneficial repainting moves.
With an O(K) input candidate pool and fixed batch width, each sweep has
O(N*m*K**2 + K**2*B*log(K)) work and a bounded number of full scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import nsmallest
import math
import numpy as np
from numba import njit, prange
from numba.typed import List

from . import panel_scoring
from . import chimera_scoring as scoring
from . import chimera_kernels as kernels
from .panel_candidates import (
    continuous_candidate_scores, prepare_candidate_scores, replacement_proposals,
    local_proposals, boundary_proposals,
)
from ..discovery.objectives import compute_outer_bic_from_log_likelihood


@dataclass(frozen=True)
class PanelSearchConfig:
    paths_per_endpoint: int = 16
    max_sweeps: int = 20
    full_scores_per_kind: int = 16
    max_bins: int = 2000
    tensor_budget_mb: int = 256

    def __post_init__(self):
        for value in (self.paths_per_endpoint,self.max_sweeps,self.full_scores_per_kind,self.max_bins,self.tensor_budget_mb):
            if isinstance(value,bool) or int(value)!=value or value<1:
                raise ValueError("panel search budgets must be positive integers")


def endpoint_select(candidates, block, quota):
    """O(M log M) selection, retaining up to quota paths per endpoint state."""
    kept=[];seen=set();counts={}
    for candidate in sorted(candidates,key=lambda x:(-float(x[1]),tuple(x[0]))):
        signature=tuple(candidate[0]);endpoint=signature[block]
        if signature in seen or counts.get(endpoint,0)>=quota:continue
        seen.add(signature);counts[endpoint]=counts.get(endpoint,0)+1;kept.append(candidate)
    return kept


@njit(cache=True,parallel=True)
def conditional_fields(emission, painted, local_paths, panel_size):
    """Per-path replacement deltas and pair correction for a suffix exchange.

    Use the same emission values as the float64 full-score tensor. Homozygous
    painting states replace BOTH copies once, rather than treating them as
    two independent replacements. Uniform/masked input has zero deltas.
    """
    samples,local_k,_,bins=emission.shape
    unary=np.zeros((panel_size,local_k))
    pair=np.zeros((panel_size,panel_size))
    for founder in prange(panel_size):
        for sample in range(samples):
            for site in range(bins):
                state=painted[sample,site];first=state//panel_size;second=state%panel_size
                if first!=founder and second!=founder:continue
                a,b=local_paths[first],local_paths[second]
                current=np.float64(emission[sample,a,b,site])
                for candidate in range(local_k):
                    aa=candidate if first==founder else a
                    bb=candidate if second==founder else b
                    unary[founder,candidate]+=np.float64(emission[sample,aa,bb,site])-current
                if first==founder and first!=second:
                    # Only this ordered painting contributes here. Each
                    # unordered pair is symmetrized after the parallel pass.
                    pair[first,second]+=2*current-np.float64(emission[sample,a,a,site])-np.float64(emission[sample,b,b,site])
    return unary,pair+pair.T


@njit(cache=True,parallel=True)
def birth_scores(emission, painted, local_paths, candidate_paths, panel_size):
    """Optimistic local residual gain, used only to screen birth proposals."""
    samples,_,_,bins=emission.shape
    result=np.zeros(len(candidate_paths))
    for candidate in prange(len(candidate_paths)):
        h=candidate_paths[candidate]
        gain=0.
        for sample in range(samples):
            for site in range(bins):
                state=painted[sample,site]
                a,b=local_paths[state//panel_size],local_paths[state%panel_size]
                current=np.float64(emission[sample,a,b,site])
                alternative=max(np.float64(emission[sample,h,b,site]),
                    np.float64(emission[sample,a,h,site]),np.float64(emission[sample,h,h,site]))
                gain+=max(0.,alternative-current)
        result[candidate]=gain
    return result


@njit(cache=True)
def fixed_paint_delta(emission,painted,old_local,new_local,panel_size):
    result=0.
    for sample in range(len(painted)):
        for site in range(painted.shape[1]):
            state=painted[sample,site];a,b=state//panel_size,state%panel_size
            result+=np.float64(emission[sample,new_local[a],new_local[b],site])
            result-=np.float64(emission[sample,old_local[a],old_local[b],site])
    return result


@njit(cache=True)
def _exclusion_scores(current):
    """All one-founder exclusions need at most three K² scans per sample."""
    samples,k,_=current.shape
    best=np.empty(samples);excluded=np.empty((samples,k))
    for sample in range(samples):
        flat=np.argmax(current[sample]);first,second=flat//k,flat%k
        best[sample]=current[sample,first,second]
        excluded[sample,:]=best[sample]
        for i in (first,second):
            maximum=-np.inf
            for a in range(k):
                for b in range(k):
                    if a!=i and b!=i:maximum=max(maximum,current[sample,a,b])
            excluded[sample,i]=maximum
    return best,excluded


@njit(cache=True,parallel=True)
def _static_gains(current,cross,diagonal):
    samples,m,k=cross.shape
    best,excluded=_exclusion_scores(current)
    gains=np.zeros((m,k))
    for candidate in prange(m):
        for sample in range(samples):
            winner=np.argmax(cross[sample,candidate])
            first=cross[sample,candidate,winner];second=-np.inf
            for partner in range(k):
                if partner!=winner:second=max(second,cross[sample,candidate,partner])
            for removed in range(k):
                retained=second if removed==winner else first
                score=max(excluded[sample,removed],retained,diagonal[sample,candidate])
                gains[candidate,removed]+=score-best[sample]
    return gains.T


@njit(cache=True, parallel=True)
def _stream_static_gains(totals, local, candidates, best, excluded):
    """Accumulate blocks before mate maxima; retain only a K-vector per task."""
    samples, k = excluded.shape
    gains = np.zeros((len(candidates), k))
    for candidate in prange(len(candidates)):
        cross = np.empty(k)
        for sample in range(samples):
            cross[:] = 0.
            diagonal = 0.
            for block in range(len(totals)):
                total = totals[block]
                c = candidates[candidate, block]
                diagonal += np.float64(total[sample, c, c])
                for partner in range(k):
                    cross[partner] += np.float64(total[sample, c, local[partner, block]])
            winner = np.argmax(cross)
            first = cross[winner]
            second = -np.inf
            for partner in range(k):
                if partner != winner:
                    second = max(second, cross[partner])
            for removed in range(k):
                retained = second if removed == winner else first
                score = max(excluded[sample, removed], retained, diagonal)
                gains[candidate, removed] += score-best[sample]
    return gains.T


def static_replacement_gains(sub,local,candidates, *, totals=None):
    """Exact all-replacement gains in the *no-switch* proposal objective.

    This allows pair reassignment, unlike fixed-painting deltas. Only two
    endpoints of the best old pair need an exclusion rescan. The best and
    second-best candidate partners handle all removed founders in O(K*M).
    No-switch scores screen proposals, not the final recombining model.
    The N*M*K candidate/mate tensor is streamed in native code, preserving
    float64 block accumulation and the sample-reduction order.
    """
    if totals is None:
        totals = [item['bin_emissions'].sum(axis=3) for item in sub]
    samples=totals[0].shape[0];k=len(local)
    current=np.zeros((samples,k,k))
    arrays=List()
    for b,total in enumerate(totals):
        a=local[:,b]
        current+=total[:,a[:,None],a[None,:]]
        arrays.append(total)
    best,excluded=_exclusion_scores(current)
    return _stream_static_gains(arrays,local,candidates,best,excluded)


def _cycle_permutation(preference):
    """Take cycles from a best-target graph; no cubic assignment solver."""
    size=len(preference);target=np.argmax(preference,axis=1)
    permutation=np.arange(size);finished=set()
    for first in range(size):
        if first in finished:continue
        path=[];offset={};current=first
        while current not in finished and current not in offset:
            offset[current]=len(path);path.append(current);current=int(target[current])
        if current in offset:
            cycle=path[offset[current]:]
            if len(cycle)>1:
                for item in cycle:permutation[item]=target[item]
        finished.update(path)
    return permutation


def _greedy_permutation(weights):
    """Quadratic-log greedy assignment plus a fixed quadratic swap search."""
    k=len(weights);permutation=np.full(k,-1,dtype=np.int64);used=np.zeros(k,bool)
    for flat in np.argsort(-weights.ravel(),kind='stable'):
        first,second=divmod(int(flat),k)
        if permutation[first]<0 and not used[second]:
            permutation[first]=second;used[second]=True
    return _improve_permutation(weights,permutation)


@njit(cache=True)
def _improve_permutation(weights,permutation):
    for _ in range(20):
        changed=False
        for i in range(len(permutation)):
            for j in range(i+1,len(permutation)):
                a,b=permutation[i],permutation[j]
                if weights[i,b]+weights[j,a]>weights[i,a]+weights[j,b]:
                    permutation[i],permutation[j]=b,a;changed=True
        if not changed:break
    return permutation


def _materialize(description,selected,candidates):
    kind,*args=description
    result=[list(path) for path in selected]
    if kind=='replace':
        founder,candidate=args;result[founder]=list(candidates[candidate])
    elif kind=='local':
        founder,block,hap=args;result[founder][block]=hap
    elif kind=='onesided':
        first,second,boundary=args;result[first][boundary:]=selected[second][boundary:]
    elif kind=='merge':
        first,second,boundary=args;result[first][boundary:]=selected[second][boundary:]
        result.pop(second)
    elif kind=='splice':
        first,second,boundary=args
        result[first][boundary:],result[second][boundary:]=result[second][boundary:],result[first][boundary:]
    elif kind=='cycle':
        boundary,permutation=args
        for first,second in enumerate(permutation):result[first][boundary:]=selected[second][boundary:]
    elif kind=='death':
        remove=set(args[0]);result=[path for index,path in enumerate(result) if index not in remove]
    elif kind=='birth':
        result.extend(list(candidates[index]) for index in args[0])
    else:raise ValueError(kind)
    return list(dict.fromkeys(tuple(path) for path in result))



def evaluate_panel(paths, sub, penalty, samples, *, paint=False, per_sample=False,
                   tensor_budget_mb=256, num_threads=None):
    """Same objective/traceback; score directly or bound traceback tensors.

    Samples factor independently. Scores stream existing block emissions;
    traceback retains bounded sample chunks. Neither path prunes states or
    changes an individual's dynamic program or the final sample reduction.
    """
    if not paint:
        kernels._resolve_threads(num_threads)
        result=panel_scoring.score_panel(paths,sub,penalty,samples)
        return result if per_sample else float(result.sum())
    bins=sum(item['n_bins'] for item in sub)
    bytes_per_sample=max(1,len(paths)**2*bins*8)
    chunk=max(1,min(samples,int(tensor_budget_mb*1024**2)//bytes_per_sample))
    result=np.empty((samples,bins),dtype=np.int32)
    for start in range(0,samples,chunk):
        stop=min(samples,start+chunk)
        kernels._resolve_threads(num_threads)
        tensor=scoring._build_tensor_from_paths(paths,sub,samples,
            sample_range=(start,stop),parallel_build=True)
        result[start:stop]=kernels._viterbi_traceback(tensor,float(penalty))
        del tensor
    return result

def select_and_resolve(beam_results,fast_mesh,batch_blocks,global_probs,global_sites,
        *,config=PanelSearchConfig(),cc_scale=.5,num_threads=None,diagnostics=None):
    """Scale the candidate search without changing the full acceptance score.

    Intermediate panels are bounded by the O(K) candidate-pool size rather
    than a fixed founder count. Search iteration and exact-score counts are
    explicit; no exhaustive refinement fallback is hidden in this route.
    """
    samples=global_probs.shape[0];blocks=len(batch_blocks)
    candidates=list(dict.fromkeys(tuple(path) for path,_ in beam_results))
    if not candidates:return []
    pen=scoring.compute_penalty(batch_blocks)
    spb=max(scoring.compute_spb(batch_blocks),
            math.ceil(sum(len(b.positions) for b in batch_blocks)/config.max_bins))
    cc=scoring.compute_cc(batch_blocks,samples,cc_scale)
    sub=scoring.compute_subblock_emissions(batch_blocks,global_probs,global_sites,spb,num_threads=num_threads)
    # Beam values are dense local indices; full-score APIs consume hap keys.
    def keys(paths):
        return [[fast_mesh.reverse_mappings[b][h] for b,h in enumerate(path)] for path in paths]
    evaluations=0
    def score(paths):
        nonlocal evaluations
        evaluations+=1
        return evaluate_panel(keys(paths),sub,pen,samples,num_threads=num_threads,
            tensor_budget_mb=config.tensor_budget_mb,per_sample=True)
    # Cover each discovered state at the most diverse input block. Low-score
    # paths remain candidates; they do not become released founders by quota.
    anchor=max(range(blocks),key=lambda b:len(batch_blocks[b].haplotypes))
    seen=set();selected=[]
    for path in candidates:
        if path[anchor] not in seen:
            selected.append(path);seen.add(path[anchor])
    baseline_scores=score(selected)
    likelihood=float(baseline_scores.sum())
    bic=compute_outer_bic_from_log_likelihood(len(selected),likelihood,cc)
    bin_offsets=np.r_[0,np.cumsum([item['n_bins'] for item in sub])]
    candidate_array=np.asarray(candidates,dtype=np.int64)
    candidate_workspace=prepare_candidate_scores(sub,candidate_array)
    static_totals=[item['bin_emissions'].sum(axis=3) for item in sub]
    capacity=len(candidates)
    for sweep in range(config.max_sweeps):
        kernels._resolve_threads(num_threads)
        k=len(selected)
        painted=evaluate_panel(keys(selected),sub,pen,samples,paint=True,
            num_threads=num_threads,tensor_budget_mb=config.tensor_budget_mb)
        local=np.asarray(selected,dtype=np.int64)
        fields=[];corrections=[];birth=np.zeros(len(candidates))
        for b,em in enumerate(sub):
            painting=np.ascontiguousarray(painted[:,bin_offsets[b]:bin_offsets[b+1]])
            unary,pair=conditional_fields(em['bin_emissions'],painting,local[:,b],k)
            fields.append(unary);corrections.append(pair)
            birth+=birth_scores(em['bin_emissions'],painting,local[:,b],candidate_array[:,b],k)
        selected_set=set(selected)
        proposals={kind:[] for kind in ('replace','static_replace','novel_replace','novel_birth','local','onesided','merge','splice','cycle','switch_splice','switch_cycle','death','birth')}
        candidate_scores=continuous_candidate_scores(sub,local,candidate_array,pen,
            workspace=candidate_workspace)
        novelty=np.maximum(candidate_scores-baseline_scores[:,None],0.).sum(axis=0)
        novel=[c for c in np.argsort(-novelty,kind='stable')
               if candidates[c] not in selected_set and novelty[c]>0]
        for c in novel[:config.full_scores_per_kind]:
            proposals['novel_birth'].append((float(novelty[c]),('birth',(int(c),))))
            similarity=np.mean(local==candidate_array[c],axis=1)
            for founder in np.argsort(-similarity,kind='stable')[:2]:
                proposals['novel_replace'].append((float(novelty[c]*(.5+similarity[founder])),
                    ('replace',int(founder),int(c))))
        eligible=np.asarray([c for c,path in enumerate(candidates)
                             if path not in selected_set],dtype=np.int64)
        static_gains=static_replacement_gains(sub,local,candidate_array,totals=static_totals)
        proposals['static_replace']=replacement_proposals(
            static_gains,eligible,config.full_scores_per_kind)
        gains=np.zeros((k,len(candidates)))
        for b in range(blocks):gains+=fields[b][:,candidate_array[:,b]]
        proposals['replace']=replacement_proposals(
            gains,eligible,config.full_scores_per_kind)
        # Report all screened edits, though only the same top-q descriptions
        # are materialized. Ranking remains before cross-kind de-duplication.
        proposal_counts={'replace':k*len(eligible),'static_replace':k*len(eligible)}
        proposals['local'],proposal_counts['local']=local_proposals(
            fields,local,config.full_scores_per_kind)
        for kind in ('onesided','merge'):
            proposal_counts[kind]=(blocks-1)*k*(k-1)
        for kind in ('splice','switch_splice'):
            proposal_counts[kind]=(blocks-1)*k*(k-1)//2
        # A lower-penalty painting reveals coordinated apparent switches.
        # It generates proposals only; acceptance still uses the same pen/BIC.
        exploratory=evaluate_panel(keys(selected),sub,min(pen,10.),samples,paint=True,
            num_threads=num_threads,tensor_budget_mb=config.tensor_budget_mb)
        preference=np.zeros((k,k));pair_gain=np.zeros((k,k))
        for boundary in range(blocks-1,0,-1):
            preference+=fields[boundary][:,local[:,boundary]]
            pair_gain+=corrections[boundary]
            edits=boundary_proposals(preference,boundary,
                config.full_scores_per_kind,'onesided')
            proposals['onesided'].extend(edits)
            # The deleted row is repainted in the full merge acceptance.
            proposals['merge'].extend((gain,('merge',*description[1:]))
                                     for gain,description in edits)
            votes=scoring._step5_build_W_at_boundary(exploratory,int(bin_offsets[boundary]),k)
            switch_gain=votes+votes.T-np.diag(votes)[:,None]-np.diag(votes)[None,:]
            proposals['switch_splice'].extend(boundary_proposals(
                switch_gain,boundary,config.full_scores_per_kind,'splice'))
            permutation=_greedy_permutation(votes)
            if np.any(permutation!=np.arange(k)):
                gain=float(votes[np.arange(k),permutation].sum()-np.trace(votes))
                proposals['switch_cycle'].append((gain,('cycle',boundary,tuple(map(int,permutation)))))
            delta=preference+preference.T+pair_gain
            proposals['splice'].extend(boundary_proposals(
                delta,boundary,config.full_scores_per_kind,'splice'))
            permutation=_cycle_permutation(preference)
            if np.any(permutation!=np.arange(k)):
                new=local.copy();new[:,boundary:]=local[permutation,boundary:]
                gain=0.
                for b in range(boundary,blocks):
                    gain+=fixed_paint_delta(sub[b]['bin_emissions'],
                        painted[:,bin_offsets[b]:bin_offsets[b+1]],local[:,b],new[:,b],k)
                proposals['cycle'].append((float(gain),('cycle',boundary,tuple(map(int,permutation)))))
        occupancy=np.bincount((painted//k).ravel(),minlength=k)+np.bincount((painted%k).ravel(),minlength=k)
        if k>1:
            for founder in np.argsort(occupancy,kind='stable')[:config.full_scores_per_kind]:
                proposals['death'].append((-float(occupancy[founder]),('death',(int(founder),))))
            absent=tuple(map(int,np.flatnonzero(occupancy==0)))
            if absent and len(absent)<k:proposals['death'].append((1.,('death',absent)))
        available=[index for index in np.argsort(-birth,kind='stable')
                   if candidates[index] not in selected_set and birth[index]>0]
        if k<capacity and available:
            batch=tuple(map(int,available[:min(k,capacity-k)]))
            proposals['birth'].append((float(sum(birth[list(batch)])),('birth',batch)))
            proposals['birth'].append((float(birth[available[0]]),('birth',(int(available[0]),))))
        start_evaluations=evaluations;best=None;scored=set()
        for kind,items in proposals.items():
            # Only q proposals were ever considered, before de-duplication.
            # Stable top-q selection preserves those exact edits and tie order.
            for gain,description in nsmallest(
                    config.full_scores_per_kind, items, key=lambda item:(-item[0],item[1])):
                trial=_materialize(description,selected,candidates)
                signature=tuple(sorted(trial))
                if not trial or len(trial)>capacity or signature in scored or signature==tuple(sorted(selected)):continue
                scored.add(signature)
                trial_scores=score(trial)
                trial_ll=float(trial_scores.sum())
                trial_bic=compute_outer_bic_from_log_likelihood(len(trial),trial_ll,cc)
                if trial_bic < bic-1e-8 and (best is None or trial_bic<best[0]):
                    best=(trial_bic,trial_ll,trial,description,gain,trial_scores)
        record=dict(sweep=sweep+1,panel_size=k,candidate_pool=len(candidates),
            proposals={kind:proposal_counts.get(kind,len(items)) for kind,items in proposals.items()},
            full_scores=evaluations-start_evaluations,bic=float(bic),
            accepted=None if best is None else str(best[3]))
        if diagnostics is not None:diagnostics.append(record)
        if best is None:break
        bic,likelihood,selected,_,_,baseline_scores=best
    if diagnostics is not None:
        diagnostics.append(dict(total_full_scores=evaluations,final_panel_size=len(selected),
            full_score_bound=1+13*config.full_scores_per_kind*config.max_sweeps))
    return [(list(path),float(likelihood)) for path in selected]
