"""Block-cavity refitting of assembled context into local founder proposals.

The current block's emission is excluded from its carrier weights. Original
reads then refit its alleles once; assembly proposals are not extra reads.
Exact shared-bit inference is capped at ten context founders. Larger or
partially covered contexts retain their input local panels.
"""
import atexit
import copy
import time
import numpy as np
from numba import njit, prange, set_num_threads
from haplotype_reconstruction.core import parallel
from haplotype_reconstruction.core.haplotypes import BlockResult, BlockResults
from haplotype_reconstruction.assembly.allele_polynomials import add_dosage_term, quadratic_values

@njit(cache=True,parallel=True)
def emission_grid(h, gl, observed, starts, stops, mix):
    n, length, _ = gl.shape
    k = len(h); s = k + 1
    emission = np.zeros((n, len(starts), s, s))
    background = np.empty(length)
    for site in range(length):
        called = 0; alt = 0
        for i in range(k):
            if h[i,site] >= 0:
                called += 1; alt += h[i,site]
        background[site] = (1.0+alt)/(2.0+called)
    for sample in prange(n):
        for b in range(len(starts)):
            for i in range(s):
                for j in range(i,s):
                    value = 0.0
                    for site in range(starts[b],stops[b]):
                        if not observed[sample,site]: continue
                        any_called = False
                        for a in range(k):
                            if h[a,site] >= 0: any_called = True
                        if not any_called: continue
                        qi = background[site] if i==k or h[i,site]<0 else float(h[i,site])
                        qj = background[site] if j==k or h[j,site]<0 else float(h[j,site])
                        if i==j and i<k:
                            p0 = 1-qi; p1 = 0.0; p2 = qi
                        else:
                            p0 = (1-qi)*(1-qj); p2=qi*qj; p1=1-p0-p2
                        e = gl[sample,site]
                        total = e.sum()
                        likelihood = (e[0]*p0+e[1]*p1+e[2]*p2)/total if total>0 else 1/3
                        value += np.log((1-mix)*likelihood+mix/3)
                    emission[sample,b,i,j] = value
                    emission[sample,b,j,i] = value
    return emission, background

@njit(cache=True)
def refresh_forward(x, pi, r):
    s=len(pi); out=np.empty_like(x)
    row=x.sum(axis=1); col=x.sum(axis=0); total=x.sum()
    for i in range(s):
        for j in range(s):
            out[i,j] = ((1-r)**2*x[i,j] + r*(1-r)*(row[i]*pi[j]+pi[i]*col[j])
                        +r*r*pi[i]*pi[j]*total)
    return out

@njit(cache=True)
def refresh_backward(x, pi, r):
    s=len(pi); out=np.empty_like(x)
    row=x@pi; col=pi@x; total=pi@row
    for i in range(s):
        for j in range(s):
            out[i,j] = (1-r)**2*x[i,j]+r*(1-r)*(row[i]+col[j])+r*r*total
    return out

@njit(cache=True,parallel=True)
def carrier_weights(log_emission, centers, rate, background_mass):
    n,bins,s,_=log_emission.shape
    pi=np.full(s,(1-background_mass)/(s-1)); pi[-1]=background_mass
    cavity=np.empty_like(log_emission)
    posterior=np.empty_like(log_emission)
    for sample in prange(n):
        predicted=np.empty((bins,s,s)); emit=np.empty((bins,s,s))
        x=np.outer(pi,pi)
        for b in range(bins):
            if b: x=refresh_forward(x,pi,-np.expm1(-rate*(centers[b]-centers[b-1])))
            predicted[b]=x
            emit[b]=np.exp(log_emission[sample,b]-np.max(log_emission[sample,b]))
            x=x*emit[b]; x/=x.sum()
        beta=np.ones((s,s))
        for b in range(bins-1,-1,-1):
            w=predicted[b]*beta; w/=w.sum()
            cavity[sample,b]=w
            w=w*emit[b]; w/=w.sum()
            posterior[sample,b]=w
            if b:
                beta=refresh_backward(beta*emit[b],pi,-np.expm1(-rate*(centers[b]-centers[b-1])))
                beta/=beta.sum()
    return cavity,posterior

@njit(cache=True)
def site_refit(gl, observed, weights, background_q):
    """Exact uniform-prior shared K-bit posterior, background integrated per copy."""
    n,s,_=weights.shape; k=s-1; count=1<<k
    fixed=np.zeros(k,np.int8); bit_index=np.arange(k)
    log_joint=np.zeros(count); support=np.zeros(k)
    for sample in range(n):
        if not observed[sample]: continue
        total=gl[sample].sum()
        if total<=0: continue
        e=.99*gl[sample]/total+.01/3
        if np.max(e)-np.min(e)<1e-12: continue
        w=weights[sample]
        unary=np.zeros(k); coupling=np.zeros((k,k))
        q=background_q
        constant=w[k,k]*((1-q)**2*e[0]+2*q*(1-q)*e[1]+q*q*e[2])
        for i in range(k):
            mass=w[i,k]+w[k,i]
            f0=(1-q)*e[0]+q*e[1]; f1=(1-q)*e[1]+q*e[2]
            constant+=mass*f0; unary[i]+=mass*(f1-f0)
            for j in range(i,k):
                mass=w[i,j] if i==j else w[i,j]+w[j,i]
                constant=add_dosage_term(constant,unary,coupling,i,j,fixed,bit_index,
                                         mass*e[0],mass*e[1],mass*e[2])
            # Effective independently observed samples carrying i, never two per homozygote.
            support[i]+=w[i,:].sum()+w[:,i].sum()-w[i,i]
        values=quadratic_values(constant,unary,coupling)
        for code in range(count):
            log_joint[code]+=np.log(max(values[code],1e-300))
    probabilities=np.exp(log_joint-log_joint.max()); probabilities/=probabilities.sum()
    q=np.zeros(k)
    for code in range(count):
        for i in range(k):
            if (code>>i)&1: q[i]+=probabilities[code]
    for i in range(k): q[i]=min(1.0,max(0.0,q[i]))
    return q,support,probabilities

@njit(cache=True,parallel=True)
def refit_sites(result,probabilities,supporters,gl,observed,cavity,background,start,stop,threshold):
    for site in prange(start,stop):
        q,support,_=site_refit(gl[:,site],observed[:,site],cavity,background[site])
        probabilities[:,site]=q; supporters[:,site]=support
        for i in range(len(result)):
            if support[i]>=2:
                if q[i]>=threshold: result[i,site]=1
                elif 1-q[i]>=threshold: result[i,site]=0

def refit_component(h,gl,observed,starts,stops,centers,threshold,background_mass,rate,projection_only=False):
    parallel.apply_dynamic_threads()
    emission,background=emission_grid(h,gl,observed,starts,stops,.01)
    parallel.apply_dynamic_threads()
    cavity,posterior=carrier_weights(emission,centers,rate,background_mass)
    result=h.copy(); probabilities=np.full(h.shape,.5); supporters=np.zeros(h.shape)
    for b in range(0 if projection_only else len(starts)):
        parallel.apply_dynamic_threads()
        refit_sites(result,probabilities,supporters,gl,observed,cavity[:,b],background,
                    starts[b],stops[b],threshold)
    return result,probabilities,supporters,posterior

_ARRAYS=None
_HANDLES=[]
def initialize(shared,cpus,active,extra,startup):
    global _ARRAYS,_HANDLES
    set_num_threads(1)
    parallel.set_dynamic_thread_state(cpus,active,extra,**startup)
    values=[parallel.attach_shared_array(m) for m in shared]
    _HANDLES=[x[0] for x in values]; _ARRAYS=[x[1] for x in values]
    atexit.register(parallel.close_shared_memory,_HANDLES)

def _worker(task):
    index,h,global_indices,starts,stops,centers,options=task
    gl,observed=_ARRAYS
    local_gl=np.ascontiguousarray(gl[:,global_indices],dtype=np.float64)
    local_observed=np.ascontiguousarray(observed[:,global_indices])
    started=time.monotonic()
    result,q,support,post=refit_component(h,local_gl,local_observed,starts,stops,centers,
                                         options["threshold"],options["background_mass"],options["rate"],options.get("projection_only",False))
    wildcard=np.empty((len(starts),len(gl)),np.int8)
    for b in range(len(starts)):
        pairs=post[:,b].reshape(len(gl),-1).argmax(axis=1)
        k=len(h); wildcard[b]=(pairs//(k+1)==k).astype(np.int8)+(pairs%(k+1)==k)
    return index,result,q,support,wildcard,time.monotonic()-started

def worker(task):
    parallel.increment_active()
    try:
        return _worker(task)
    finally:
        set_num_threads(1)
        parallel.release_dynamic_extra()
        parallel.decrement_active()

def refine(raw_blocks,super_blocks,gl,sites,observed,cpus,options,chromosome_map=None,
           n_generations=3):
    """Replace covered panels, exact-dedup; preserve unsupported template alleles."""
    originals=list(raw_blocks); output=BlockResults(list(originals))
    observed=observed.copy()
    for block in originals:
        if block.keep_flags is not None:
            indices=np.searchsorted(sites,block.positions)
            observed[:,indices]&=np.asarray(block.keep_flags)>0
    raw_first=np.array([b.positions[0] for b in originals])
    raw_last=np.array([b.positions[-1] for b in originals])
    specs={};tasks=[];skipped=[]
    for super_index,super_block in enumerate(super_blocks):
        positions=np.asarray(super_block.positions)
        block_ids=np.flatnonzero((raw_first>=positions[0])&(raw_last<=positions[-1]))
        if not len(block_ids): continue
        expected=np.concatenate([originals[b].positions for b in block_ids])
        if not np.array_equal(expected,positions):
            skipped.append((super_index,"incomplete_original_blocks"));continue
        # Exclude completion fills from carrier evidence, as the canonical painter does.
        inference=np.asarray(getattr(super_block,"missing_aware_inference_discrete_haps",super_block.discrete_haps),np.int8)
        _,representatives=np.unique(inference,axis=0,return_index=True)
        representatives=np.sort(representatives)
        h=np.ascontiguousarray(inference[representatives])
        template=np.ascontiguousarray(super_block.discrete_haps[representatives])
        if len(h)>options["max_k"]:
            skipped.append((super_index,"enumeration_cap"));continue
        starts=np.r_[0,np.cumsum([len(originals[b].positions) for b in block_ids])[:-1]].astype(np.int64)
        stops=np.r_[starts[1:],len(positions)].astype(np.int64)
        centers=np.array([np.mean(originals[b].positions) for b in block_ids])
        local_options = dict(options)
        if chromosome_map is not None:
            if chromosome_map.has_map:
                # Rate is now per Morgan, not per bp. The map is integrated
                # between block centres, including its configured tails.
                centers = chromosome_map.cumulative_morgans(centers)
                local_options["rate"] = float(n_generations)
            else:
                local_options["rate"] = chromosome_map.fallback_rate_per_bp * n_generations
        indices=np.searchsorted(sites,positions)
        assert np.array_equal(sites[indices],positions)
        specs[super_index]=(block_ids,starts,stops,template)
        tasks.append((super_index,h,indices,starts,stops,centers,local_options))
    handles=[]; metadata=[]
    try:
        for array in (gl,observed):
            handle,meta=parallel.create_shared_array(array);handles.append(handle);metadata.append(meta)
        diagnostics=[];completed=0
        workers=min(cpus,len(tasks))
        if not workers: return output,dict(components=[],skipped=skipped,options=options,changed_alleles=0,filled_alleles=0)
        context=parallel.forkserver_context
        active=context.Value("i",0);extra=context.Value("i",0)
        startup={name:context.Value("i",value) for name,value in dict(
            started_counter=0,participant_counter=0,batch_generation=0,
            batch_task_count=len(tasks),startup_target=workers,startup_ready=0).items()}
        with parallel.ForkserverPool(workers,initializer=initialize,
                initargs=(metadata,cpus,active,extra,startup)) as pool:
            for index,result,q,support,wildcard,seconds in pool.imap_unordered(worker,tasks,chunksize=1):
                block_ids,starts,stops,template=specs[index]
                # If the frozen inference template was unknown but completion had filled it,
                # retain that existing release unless the new refit supports a concrete call.
                fallback=(result<0)&(template>=0)
                result[fallback]=template[fallback]
                changes=int(((result>=0)&(template>=0)&(result!=template)).sum())
                fills=int(((result>=0)&(template<0)).sum())
                diagnostics.append(dict(component=index,founders=len(result),seconds=seconds,
                                         allele_changes=changes,filled_alleles=fills))
                for b,lo,hi,slots in zip(block_ids,starts,stops,wildcard):
                    panel=result[:,lo:hi]
                    _,unique=np.unique(panel,axis=0,return_index=True);unique=np.sort(unique)
                    panel=np.ascontiguousarray(panel[unique])
                    old=originals[b]
                    block=BlockResult(old.positions.copy(),
                        {i:np.stack((np.where(row<0,.5,1-row),np.where(row<0,.5,row)),axis=-1)
                         for i,row in enumerate(panel)},keep_flags=copy.deepcopy(old.keep_flags),
                         genotype_evidence_mode=old.genotype_evidence_mode)
                    block.discrete_haps=panel
                    block.n_directional_site_supporters=support[unique,lo:hi].copy()
                    block.feedback_posterior_alt=q[unique,lo:hi].copy()
                    block.founder_alt_pseudo_probability=q[unique,lo:hi].copy()
                    start=np.searchsorted(sites,old.positions[0]); stop=start+len(old.positions)
                    depth=np.any(observed[:,start:stop] & np.asarray(old.keep_flags if old.keep_flags is not None else np.ones(stop-start),bool),axis=1)
                    block.sample_has_observed_kept_depth=depth
                    block.wildcard_slots=np.where(depth,slots,2).astype(np.int8)
                    block.wildcard_mass=float(slots[depth].sum()/max(2*depth.sum(),1))
                    for name in ("missing_aware_break_before", "missing_aware_break_after"):
                        if hasattr(old, name):
                            setattr(block, name, getattr(old, name))
                    # Do not inherit old selected-path or immutable input identities.
                    output.blocks[b]=block
                completed+=1
                if completed%20==0 or completed==len(tasks):
                    print("FEEDBACK",completed,len(tasks),"changes",sum(x["allele_changes"] for x in diagnostics),flush=True)
        return output,dict(components=diagnostics,skipped=skipped,options=options,
                           changed_alleles=sum(x["allele_changes"] for x in diagnostics),
                           filled_alleles=sum(x["filled_alleles"] for x in diagnostics))
    finally:
        parallel.close_shared_memory(handles,unlink=True)


def select_worker(task):
    """One local selection, using the same shared GL/mask pool initializer."""
    from .candidate_selection import select_candidate_panel
    from .candidate_rescue import CandidateRescueConfig, rescue_candidate_panel
    index, indices, latent, proposals, config, selection = task
    parallel.increment_active()
    try:
        gl, observed = _ARRAYS
        evidence = np.ascontiguousarray(gl[:, indices], dtype=np.float64)
        mask = np.ascontiguousarray(observed[:, indices])
        competing = select_candidate_panel(evidence, observed_mask=mask,
            original_latent=latent, proposal_panels=proposals, config=config)
        # Source endpoints are cavity-ranked even though the competing bank
        # is BIC-ranked. Reusing their fits avoids a second full bank search.
        backbone = next(value for value in reversed(competing.cavity_source_results)
                        if value is not None)
        result = rescue_candidate_panel(evidence, observed_mask=mask,
            feedback=backbone, competing=dict(discrete_haps=competing.discrete_haps,
                latent_haps=competing.selected_mode.haplotypes),
            config=CandidateRescueConfig(discovery=config.discovery), selection=selection)
        result["diagnostic"]["candidate_selection"] = competing.diagnostic
        return index, result
    finally:
        set_num_threads(1)
        parallel.release_dynamic_extra()
        parallel.decrement_active()


def materialize_selected_block(original, result, observed):
    """Release only supported calls, preserving the original marker layout."""
    keep = (np.ones(len(original.positions), dtype=bool) if original.keep_flags is None
            else np.asarray(original.keep_flags) > 0)
    k = len(result["discrete_haps"])
    calls = np.full((k, len(keep)), -1, dtype=np.int8)
    q = np.full(calls.shape, .5)
    support = np.zeros(calls.shape, dtype=np.int64)
    calls[:, keep] = result["discrete_haps"]
    q[:, keep] = result["q"]
    support[:, keep] = result["support"]
    public_q = np.where(calls >= 0, q, .5)
    block = BlockResult(original.positions.copy(),
        {i: np.stack((1-public_q[i], public_q[i]), axis=-1) for i in range(k)},
        keep_flags=copy.deepcopy(original.keep_flags),
        genotype_evidence_mode=original.genotype_evidence_mode)
    block.discrete_haps = calls
    block.founder_alt_pseudo_probability = q
    block.n_directional_site_supporters = support
    block.founder_allele_pseudo_confidence = np.maximum(q, 1-q)
    block.pair_assignments = np.asarray(result["assignments"], dtype=np.int64)
    depth = np.any(observed[:, keep], axis=1)
    block.sample_has_observed_kept_depth = depth
    block.wildcard_slots = np.where(depth, (block.pair_assignments == k).sum(axis=1), 2)
    block.wildcard_mass = float(block.wildcard_slots[depth].sum() / max(2*depth.sum(), 1))
    for name in ("missing_aware_break_before", "missing_aware_break_after"):
        if hasattr(original, name):
            setattr(block, name, getattr(original, name))
    return block


def select_blocks(originals, proposals, gl, sites, observed, cpus, config, selection,
                  checkpoint_io):
    """Parallel local selection with resumable 128-block checkpoint batches."""
    output = BlockResults(list(originals))
    diagnostics = [None] * len(originals)
    batch_size = 128
    pending = []
    batches = {}
    for start in range(0, len(originals), batch_size):
        phase = f"batch{start:06d}"
        saved = checkpoint_io.load(phase)
        if saved is not None:
            for index, block, diagnostic in saved:
                output.blocks[index] = block
                diagnostics[index] = diagnostic
            continue
        batches[start] = []
        for index in range(start, min(start+batch_size, len(originals))):
            block = originals[index]
            keep = (np.ones(len(block.positions), dtype=bool) if block.keep_flags is None
                    else np.asarray(block.keep_flags) > 0)
            indices = np.searchsorted(sites, block.positions)
            if not np.array_equal(sites[indices], block.positions):
                raise ValueError("feedback block sites are not aligned to the raw evidence")
            mode = getattr(block, "cavity_selected_mode", None)
            if mode is None or not keep.any() or not observed[:, indices[keep]].any():
                diagnostic = dict(skipped="no_fitted_local_mode_or_observed_kept_sites")
                diagnostics[index] = diagnostic
                # Strip large raw arrays from a fallback checkpoint without
                # modifying the original Stage-1 object.
                fallback = copy.copy(block)
                fallback.reads_count_matrix = fallback.probs_array = None
                output.blocks[index] = fallback
                batches[start].append((index, fallback, diagnostic))
                continue
            panels = []
            for proposal in proposals:
                if not np.array_equal(proposal[index].positions, block.positions):
                    raise ValueError("feedback proposal sites differ from original block")
                panels.append(np.ascontiguousarray(proposal[index].discrete_haps[:, keep]))
            pending.append((index, indices[keep], mode.haplotypes, panels, config, selection))

    def save_complete(start):
        rows = batches[start]
        if len(rows) == min(batch_size, len(originals)-start):
            checkpoint_io.save(f"batch{start:06d}", sorted(rows, key=lambda row: row[0]))
            del batches[start]

    for start in list(batches):
        save_complete(start)
    if not pending:
        return output, diagnostics
    workers = min(cpus, len(pending))
    handles = []
    metadata = []
    try:
        for array in (gl, observed):
            handle, meta = parallel.create_shared_array(array)
            handles.append(handle)
            metadata.append(meta)
        context = parallel.forkserver_context
        active = context.Value("i", 0)
        extra = context.Value("i", 0)
        startup = {name: context.Value("i", value) for name, value in dict(
            started_counter=0, participant_counter=0, batch_generation=0,
            batch_task_count=len(pending), startup_target=workers, startup_ready=0).items()}
        print(f"[Feedback selection] {len(pending)} blocks; {workers} workers, "
              f"1 initial thread each, dynamic ceiling {cpus}; {selection}", flush=True)
        with parallel.ForkserverPool(workers, initializer=initialize,
                initargs=(metadata, cpus, active, extra, startup)) as pool:
            for index, result in pool.imap_unordered(select_worker, pending, chunksize=1):
                block = originals[index]
                indices = np.searchsorted(sites, block.positions)
                selected = materialize_selected_block(block, result, observed[:, indices])
                output.blocks[index] = selected
                diagnostics[index] = result["diagnostic"]
                start = index // batch_size * batch_size
                batches[start].append((index, selected, result["diagnostic"]))
                save_complete(start)
        return output, diagnostics
    finally:
        parallel.close_shared_memory(handles, unlink=True)
