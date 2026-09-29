"""Parallel sparse deletion repairs with canonical full-site endpoint choice.

All deletion starts compete; intermediate repairs are approximate. The original
unrepaired deletion remains a fallback and count acceptance retains full BIC.
"""
import resource
import time
import numpy as np
from .proxy_primary import CompactPrimary
from .workspace import component_workspace, resolve_threads
from .candidates import completed_candidates
from ...core import parallel
from ...discovery.objectives import compute_outer_bic_from_log_likelihood as bic

def fit_deletion(dropped,batch,selected,neutral,sites,config,threads,
                 cost,prepared_arrays,proxy):
    started,cpu=time.perf_counter(),time.process_time()
    original=np.delete(selected,int(dropped),axis=0)
    rows=original.copy()
    history=[]
    with parallel.numba_thread_scope(resolve_threads(threads)):
        proxy_score=proxy.score(rows)
    for iteration in range(config.count_repair_sweeps):
        with parallel.numba_thread_scope(resolve_threads(threads)):
            painting=proxy.paint_compact(rows)[0]
            replacement,gains=proxy.fixed(rows,painting)
            focal=int(np.argmax(gains))
            single=rows.copy();single[focal]=replacement[focal]
            best_rows,best_score=rows,proxy_score
            for proposal in (replacement,single):
                if np.array_equal(proposal,rows):continue
                value=proxy.score(proposal)
                if value>best_score+1e-6:
                    best_rows,best_score=proposal,value
        history.append(dict(iteration=iteration,proxy_gain=best_score-proxy_score))
        if best_score<=proxy_score+1e-6:break
        rows,proxy_score=best_rows,best_score
    workspace=component_workspace({},batch,neutral,sites,config.proposal_max_bins,
        threads,prepared_arrays,minimum_bin_size=config.proposal_min_sites_per_bin)
    original_score=workspace.canonical(original)
    full_calls=1
    final_score=original_score
    accepted=False
    if not np.array_equal(original,rows):
        endpoint=workspace.canonical(rows);full_calls+=1
        if endpoint>original_score+1e-6:
            accepted=True;final_score=endpoint
        else:rows=original
    return dict(selected=rows,score=final_score,bic=float(bic(len(rows),final_score,cost)),
        paired_changed=False,repair_history=history,
        proposal_model='sparse_endpoint_full_score_with_unrepaired_fallback',
        endpoint_accepted=accepted,full_endpoint_evaluations=full_calls,
        runtime=dict(seconds=time.perf_counter()-started,
            shared_process_cpu_seconds=time.process_time()-cpu,
            maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

def run(tasks, *, batch, selected, neutral, sites, config, threads, cost, workspaces):
    if not tasks:
        return []
    total = int(resolve_threads(threads))
    workspace = component_workspace(workspaces, batch, neutral, sites,
        config.proposal_max_bins, threads,
        minimum_bin_size=config.proposal_min_sites_per_bin)
    arrays = workspace.evidence, workspace.complete, workspace.logs
    cache = getattr(workspace, '_compact_proxies', None)
    if cache is None:
        cache = workspace._compact_proxies = {}
    if 200 not in cache:
        with parallel.numba_thread_scope(total):
            cache[200] = CompactPrimary(workspace, 200)
    proxy = cache[200]
    functions = []
    for dropped, _ in tasks:
        def compute(budget, dropped=dropped):
            return fit_deletion(dropped, batch, selected, neutral, sites, config,
                threads if budget is None else budget, cost, arrays, proxy)
        functions.append(compute)
    samples, length = workspace.evidence.shape[:2]
    founders = len(selected)-1
    states = founders*(founders+1)//2
    bins = int(proxy.bin_offsets[-1])
    # No full-site traceback or private evidence copy in this path. Full-score
    # calls still materialize alleles/dosages; compact traceback is per thread.
    per_candidate = (2*length*(founders+states) + samples*bins*4
        + min(total,samples)*bins*(states+8) + 512*1024**2)
    answers = [None]*len(tasks)
    for index, result in completed_candidates(functions, threads, per_candidate):
        answers[index] = result
        tasks[index][1].save('result',result)
    return answers
