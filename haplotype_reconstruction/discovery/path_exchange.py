"""Exact fixed-K reciprocal suffix exchange with batched screening.

For row transposition P at cut c, suffix emissions become original emissions
in coordinates u=P(s). The suffix transition prior is P*pi, NOT pi unless the
two frequencies agree. One backward pass per pair therefore scores all cuts:
 prefix_logZ[c] + suffix_log_scale[c]
 + log(sum_ab prefix_prediction[c,P(a),P(b)] * suffix_B[c,a,b]).

Prefix predictions are shared across all pairs; backward messages are shared
across cuts of one pair. Missing emissions are 1; the unknown state U stays
fixed under P. Screening evaluates every founder pair at up to 16 cut sites.
The normalized objective is exact for each screened candidate.

General cost is O(N*L*K^4) over all pairs, versus O(C*N*L*K^4) for a separate
forward pass at each cut. Working scratch is O(threads*C*K^2), output storage
is O(N*pairs*C), and C<=16. Unequal frequencies preclude a general subquartic
claim. Near the top-eight boundary, direct forward rescoring protects
floating-point ordering; it does not introduce a scientific acceptance rule.
"""
import numpy as np
from numba import njit, prange
from haplotype_reconstruction.core import parallel
from . import path_model as model, path_fitting as fitting
from .path_fitting import canonical, regularizer


def swap(panel, frequencies, i, j, cut):
    trial = panel.copy()
    trial[i, cut:], trial[j, cut:] = panel[j, cut:], panel[i, cut:]
    if len(np.unique(trial, axis=0)) != len(panel):
        return None, 'duplicate_rows'
    trial, mass = canonical(trial, frequencies)
    original, original_mass = canonical(panel, frequencies)
    if np.array_equal(trial, original) and np.array_equal(mass, original_mass):
        return None, 'noop'
    return (trial, mass), None


def proposals(panel, frequencies):
    cuts = np.unique(np.linspace(1, panel.shape[1]-1, min(16,panel.shape[1]-1), dtype=int))
    seen = set()
    for i in range(len(panel)):
        for j in range(i+1,len(panel)):
            for cut in cuts:
                state, reason = swap(panel,frequencies,i,j,int(cut))
                if state is not None:
                    key = state[0].tobytes(), state[1].tobytes()
                    if key in seen:
                        state, reason = None, 'repeated_state'
                    seen.add(key)
                yield dict(i=i,j=j,cut=int(cut)), state, reason


def protect(incumbent, fitted):
    return incumbent if fitted['objective'] < incumbent['objective'] else fitted



@njit(cache=True)
def _predict_into(filtered,pi,rate,out,row,col):
    s=len(pi); row[:]=0.; col[:]=0.
    for a in range(s):
        for b in range(s):
            row[a]+=filtered[a,b]; col[b]+=filtered[a,b]
    stay=1.-rate
    for a in range(s):
        for b in range(s):
            out[a,b]=(stay*stay*filtered[a,b]+stay*rate*(row[a]*pi[b]+pi[a]*col[b])+rate*rate*pi[a]*pi[b])


@njit(cache=True)
def _backward_into(weighted,pi,rate,out,row,col):
    s=len(pi); row[:]=0.; col[:]=0.; total=0.
    for a in range(s):
        for b in range(s):
            value=weighted[a,b]
            row[a]+=value*pi[b]; col[b]+=value*pi[a]
            total+=pi[a]*value*pi[b]
    stay=1.-rate
    for a in range(s):
        for b in range(s):
            out[a,b]=stay*stay*weighted[a,b]+stay*rate*(row[a]+col[b])+rate*rate*total


@njit(cache=True,parallel=True)
def _score_samples(panel,gl,observed,pi,rates,pairs,cuts):
    n,length=gl.shape[:2]; k=len(panel); s=k+1; ccount=len(cuts)
    result=np.full((n,len(pairs),ccount),-np.inf)
    baseline=np.full(n,-np.inf)
    for sample in prange(n):
        predictions=np.zeros((ccount,s,s)); prefix_log=np.full(ccount,-np.inf)
        prior=np.empty((s,s)); filtered=np.empty((s,s))
        row,col=np.empty(s),np.empty(s)
        for a in range(s):
            for b in range(s): filtered[a,b]=pi[a]*pi[b]
        logz=0.; ci=0
        for site in range(length):
            if site==0: prior[:,:]=filtered
            else: _predict_into(filtered,pi,rates[site-1],prior,row,col)
            if ci<ccount and site==cuts[ci]:
                predictions[ci,:,:]=prior; prefix_log[ci]=logz; ci+=1
            scale=0.
            for a in range(s):
                ha=panel[a,site] if a<k else -1
                for b in range(s):
                    hb=panel[b,site] if b<k else -1
                    emission=model._emission(ha,hb,gl[sample,site]) if observed[sample,site] else 1.
                    filtered[a,b]=prior[a,b]*emission; scale+=filtered[a,b]
            if scale<=0.: break
            logz+=np.log(scale); filtered/=scale
            if site==length-1: baseline[sample]=logz
        if not ccount: continue
        future=np.empty((s,s)); weighted=np.empty((s,s)); permuted_pi=np.empty(s)
        for pair in range(len(pairs)):
            i,j=pairs[pair]; permuted_pi[:]=pi
            permuted_pi[i]=pi[j]; permuted_pi[j]=pi[i]
            future[:,:]=1.; logscale=0.; ci=ccount-1
            for site in range(length-1,cuts[0]-1,-1):
                maximum=0.
                for a in range(s):
                    ha=panel[a,site] if a<k else -1
                    for b in range(s):
                        hb=panel[b,site] if b<k else -1
                        emission=model._emission(ha,hb,gl[sample,site]) if observed[sample,site] else 1.
                        value=future[a,b]*emission; weighted[a,b]=value
                        maximum=max(maximum,value)
                if maximum<=0.: break
                weighted/=maximum; logscale+=np.log(maximum)
                if ci>=0 and site==cuts[ci]:
                    contraction=0.
                    for a in range(s):
                        pa=j if a==i else i if a==j else a
                        for b in range(s):
                            pb=j if b==i else i if b==j else b
                            contraction+=predictions[ci,pa,pb]*weighted[a,b]
                    if contraction>0.:
                        result[sample,pair,ci]=prefix_log[ci]+logscale+np.log(contraction)
                    ci-=1
                if site>cuts[0]: _backward_into(weighted,permuted_pi,rates[site-1],future,row,col)
    return baseline,result


def score_all_suffix_swaps(panel,frequencies,gl,observed,positions,*,generations=3.,
        recombination_rate_per_bp=5e-8,wildcard_mass=.01,interval_morgans=None):
    """All pair/cut likelihoods, including states later rejected as duplicates.

    Axes are sample,pair,cut; frequencies remain associated with prefix rows.
    The caller owns parallel thread limits, as for path_model.score_panel.
    """
    panel,gl,observed,pi,rates=model._prepare(panel,gl,positions,observed,generations,
        recombination_rate_per_bp,wildcard_mass,interval_morgans,frequencies)
    pairs=np.asarray([(i,j) for i in range(len(panel)) for j in range(i+1,len(panel))],np.int64).reshape(-1,2)
    cuts=np.unique(np.linspace(1,panel.shape[1]-1,min(16,panel.shape[1]-1),dtype=np.int64))
    baseline,values=_score_samples(panel,gl,observed,pi,rates,pairs,cuts)
    return dict(pairs=pairs,cuts=cuts,sample_log_likelihood=values,
        log_likelihood=values.sum(axis=0),baseline_sample_log_likelihood=baseline,
        baseline_log_likelihood=float(baseline.sum()))


def screen_candidates(incumbent,gl,observed,positions,*,reference_boundary=True,
        generations=3., recombination_rate_per_bp=5e-8, wildcard_mass=.01,
        interval_morgans=None, learn_frequencies=True):
    """Screen all pairs at 16 cuts, preserving the normalized path objective.

    Uses duplicate/no-op filtering, paired canonicalization and the exact regularizer.
    Top candidates and candidates within a numerical guard of the top-eight boundary
    are rescored by the direct forward routine, protecting the same top-eight
    ordering when accumulation differences can matter. The guard only chooses
    evaluation implementation; it never discards a scientific candidate.
    """
    parallel.apply_dynamic_threads()
    model_options = dict(generations=generations,
        recombination_rate_per_bp=recombination_rate_per_bp,
        wildcard_mass=wildcard_mass, interval_morgans=interval_morgans)
    raw=score_all_suffix_swaps(incumbent['panel'],incumbent['frequencies'],gl,observed,positions,**model_options)
    pair_index={tuple(pair):i for i,pair in enumerate(raw['pairs'])}
    cut_index={int(cut):i for i,cut in enumerate(raw['cuts'])}
    scores=[]; skipped={}; states=[]
    for edit,state,reason in proposals(incumbent['panel'],incumbent['frequencies']):
        if state is None:
            skipped[reason]=skipped.get(reason,0)+1; continue
        panel,freq=state
        ll=float(raw['log_likelihood'][pair_index[(edit['i'],edit['j'])],cut_index[edit['cut']]])
        scores.append(dict(**edit,log_likelihood=ll,objective=ll+regularizer(panel,freq,learn_frequencies),
            delta_ll=ll-raw['baseline_log_likelihood']))
        states.append(state)
    rescored=set()
    # Conservative numerical implementation guard, not an acceptance tolerance.
    margin=1e-7
    if reference_boundary and scores:
        while True:
            ordering=sorted(range(len(scores)),key=lambda i:(-scores[i]['objective'],scores[i]['i'],scores[i]['j'],scores[i]['cut']))
            boundary=scores[ordering[min(8,len(ordering))-1]]['objective']
            pending=[i for i in ordering if i not in rescored and scores[i]['objective']>=boundary-margin]
            if not pending: break
            for index in pending:
                panel,freq=states[index]
                ll=model.score_panel(panel,gl,positions,observed,founder_frequencies=freq,**model_options)['log_likelihood']
                scores[index].update(log_likelihood=ll,objective=ll+regularizer(panel,freq,learn_frequencies),
                    delta_ll=ll-raw['baseline_log_likelihood'])
                rescored.add(index)
    scores.sort(key=lambda x:(-x['objective'],x['i'],x['j'],x['cut']))
    return dict(scores=scores,top=scores[:8],skipped=skipped,incumbent_objective=incumbent['objective'],
        diagnostic=dict(method='prefix-forward/pair-specific-permuted-prior-backward',
            candidates=len(scores),reference_rescores=len(rescored),numerical_boundary_guard=margin,
            baseline_log_likelihood=raw['baseline_log_likelihood']))


def exchange_panel(incumbent, gl, observed, positions, *, prepared=None,
                   generations=3., recombination_rate_per_bp=5e-8,
                   wildcard_mass=.01, interval_morgans=None,
                   learn_frequencies=True, max_updates=20):
    """Optional fixed-K pass; never enabled implicitly by ordinary selection.

    Refit the top eight swaps and one ordinary warm-start control under the
    same model/update budget. Each trial is protected by the original panel;
    ties within 1e-6 prefer the warm control, then the screening order. Neither
    switch count, founder count nor downstream diagnostics choose the winner.
    """
    model_options = dict(generations=generations,
        recombination_rate_per_bp=recombination_rate_per_bp,
        wildcard_mass=wildcard_mass, interval_morgans=interval_morgans)
    screen = screen_candidates(incumbent, gl, observed, positions,
                               learn_frequencies=learn_frequencies, **model_options)
    if prepared is None:
        from . import path_scoring
        prepared = path_scoring.prepare_observations(gl, observed, positions, **model_options)
    prepared = fitting.prepare_fit_observations(prepared)
    trials = []
    for rank in range(-1, len(screen["top"])):
        panel, frequencies = incumbent["panel"], incumbent["frequencies"]
        edit = None
        if rank >= 0:
            edit = screen["top"][rank]
            (panel, frequencies), reason = swap(panel, frequencies,
                edit["i"], edit["j"], edit["cut"])
            assert reason is None
        fit = fitting.fit_fixed(panel, gl, observed, positions,
            frequencies=frequencies, learn_frequencies=learn_frequencies,
            max_updates=max_updates, prepared=prepared, **model_options)
        assert len(fit["panel"]) == len(incumbent["panel"])
        retained = protect(incumbent, fit)
        trials.append(dict(rank=rank, edit=edit, fit=fit, retained=retained,
                           incumbent_retained=retained is incumbent))
    maximum = max(trial["retained"]["objective"] for trial in trials)
    winner = next(trial for trial in trials
                  if trial["retained"]["objective"] >= maximum - 1e-6)
    diagnostic = dict(winner_rank=winner["rank"],
        incumbent_retained=winner["incumbent_retained"],
        warm_control_objective=trials[0]["retained"]["objective"],
        winner_vs_warm_objective=winner["retained"]["objective"]
            - trials[0]["retained"]["objective"],
        screened=len(screen["scores"]), skipped=screen["skipped"], top=screen["top"],
        screen_diagnostic=screen["diagnostic"],
        trials=[dict(rank=row["rank"], objective=row["fit"]["objective"],
            retained_objective=row["retained"]["objective"],
            incumbent_retained=row["incumbent_retained"]) for row in trials])
    return winner["retained"], diagnostic
