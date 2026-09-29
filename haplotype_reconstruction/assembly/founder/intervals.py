"""Local-flank scoring for the same feasible paired-tract shortlist.

Swapping two founder rows inside a tract and relabelling the incumbent sample
states there preserves every emission and internal switch. Only the two tract
boundaries change cost. Their gains add, giving an O(P N B + P B W) proposal
scan rather than O(P N B W K^2). These are FEASIBLE lower bounds, not bounds for
discarding other intervals. The existing full objective/genotype-fit guard
decides acceptance. Missing-data masks, alleles and component boundaries stay.

The shortlist can miss edits requiring a different sample painting; this is a
scientific search approximation requiring real/simulation comparison.
"""
import time
import numpy as np
from numba import njit, prange, get_num_threads
from .. import founder_refinement as fr
from . import scoring
from .count import _reconstruct
from .workspace import component_workspace
from ...core.haplotypes import BlockResults


@njit(cache=True, parallel=True, nogil=True)
def boundary_gains(before, after, pairs, founders, penalty):
    result = np.zeros((len(pairs), len(before)+2), np.float64)
    for index in prange(len(pairs)):
        a, b = pairs[index]
        for boundary in range(len(before)):
            delta = 0
            for sample in range(before.shape[1]):
                old, new = before[boundary, sample], after[boundary, sample]
                first, second = new // founders, new % founders
                if first == a: first = b
                elif first == b: first = a
                if second == a: second = b
                elif second == b: second = a
                relabelled = min(first, second)*founders + max(first, second)
                delta += int(old != new) - int(old != relabelled)
            result[index, boundary+1] = penalty*delta
    return result


@njit(cache=True, parallel=True, nogil=True)
def best_ends(gains, starts, stops):
    values = np.empty((len(gains), len(starts)), np.float64)
    ends = np.empty((len(gains), len(starts)), np.int64)
    for pair in prange(len(gains)):
        for item in range(len(starts)):
            start, stop = starts[item], stops[item]
            winner = start+1
            for end in range(start+2, stop+1):
                if gains[pair,end] > gains[pair,winner]: winner=end
            values[pair,item] = gains[pair,start]+gains[pair,winner]
            ends[pair,item] = winner
    return values, ends


def _interval_ranges(blocks, window, groups=None):
    if groups is None:
        starts = np.arange(blocks, dtype=np.int64)
        stops = np.minimum(starts + window, blocks)
    else:
        assert groups[0][0] == 0 and groups[-1][1] == blocks
        assert all((a[1] == b[0] for a, b in zip(groups, groups[1:])))
        starts = np.asarray([a for a, b in groups], np.int64)
        stops = np.asarray(
            [groups[min(len(groups), i + window) - 1][1] for i in range(len(groups))],
            np.int64
        )
    return (starts, stops)



def refine_components(prepared, components, neutral, sites, *, config,
                      checkpoints=None, l1_blocks=None, workspaces=None):
    workspaces = {} if workspaces is None else workspaces
    outputs, diagnostics = [], []
    for number, component in enumerate(components):
        token = f'founder_boundary_interval_local_v1.component{number}'
        saved = fr._load(checkpoints, token)
        if saved is not None:
            outputs.append(saved['block']); diagnostics.append(saved['diagnostic'])
            continue
        batch = [b for b in prepared if b.positions[0] >= component.positions[0]
                 and b.positions[-1] <= component.positions[-1]]
        if not np.array_equal(np.concatenate([b.positions for b in batch]),component.positions):
            raise ValueError('Boundary interval component is not a prepared-block span')
        rows = fr._local_selection(component,batch)
        original = rows.copy()
        workspace = component_workspace(workspaces,batch,neutral,sites,
            config.proposal_max_bins,get_num_threads(),
            minimum_bin_size=config.proposal_min_sites_per_bin)
        def counted(panel):
            calls = scoring.selected_alleles(workspace.leaves,workspace.offsets,panel)
            values, switches = scoring.score_and_switch_count(calls,workspace.evidence,
                workspace.complete,workspace.penalty,workspace.logs)
            return float(values.sum()),int(switches.sum())
        current, switches = counted(rows)
        initial = current
        history = []
        pairs = np.asarray(list(zip(*np.triu_indices(len(rows),1))),np.int64)
        scales = [('local',None,False)]
        context = fr._macro_context(batch,l1_blocks)
        if context is not None:
            groups = context[0]
            scales += [('l1_start',groups,False),('l1_stop',
                [(len(batch)-b,len(batch)-a) for a,b in groups[::-1]],True)]
        for iteration in range(config.max_iterations if len(pairs) and len(batch)>1 else 0):
            phase = token + f'.iteration{iteration}'
            saved = fr._load(checkpoints,phase)
            if saved is not None:
                rows,current,switches,history = (saved[k] for k in ('selected','score','switches','history'))
                if saved['converged']: break
                continue
            started = time.monotonic()
            workspace.set_reference(rows)
            painting = workspace.evaluate(rows,paint=True)
            before = np.ascontiguousarray(painting[:,workspace.offsets[1:-1]-1].T)
            after = np.ascontiguousarray(painting[:,workspace.offsets[1:-1]].T)
            gains = boundary_gains(before,after,pairs,len(rows),workspace.penalty)
            candidates = {}
            for scale,groups,reverse in scales:
                starts,stops = _interval_ranges(len(batch),config.window_blocks,groups)
                values,ends = best_ends(np.ascontiguousarray(gains[:,::-1]) if reverse else gains,starts,stops)
                order = np.argsort(-values.ravel(),kind='stable')
                retained = 0
                for flat in order:
                    pair,index = divmod(int(flat),len(starts))
                    left,right = int(starts[index]),int(ends[pair,index])
                    if reverse: left,right = len(batch)-right,len(batch)-left
                    a,b = map(int,pairs[pair])
                    if np.array_equal(rows[a,left:right],rows[b,left:right]): continue
                    key=(a,b,left,right)
                    candidates.setdefault(key,(float(values[pair,index]),scale))
                    retained += 1
                    if retained >= config.branch_cap: break
            del painting,before,after
            ranked = sorted(candidates.items(),key=lambda x:(-x[1][0],x[0]))[:config.branch_cap]
            best=None; records=[]
            for (a,b,left,right),(feasible_gain,scale) in ranked:
                trial=rows.copy();trial[[a,b],left:right]=rows[[b,a],left:right]
                # Exact local-flank score where affordable, otherwise the
                # existing whole-panel fallback. Only potential winners need
                # a full canonical switch count for the genotype-fit veto.
                score=workspace.evaluate(trial)
                margin=max(1e-6,1e-10*max(abs(score),abs(current)))
                if score+margin < current+feasible_gain:
                    raise RuntimeError('Boundary relabelling score is not a feasible lower bound')
                threshold=current if best is None else best[0]
                if score+margin < threshold:
                    records.append(dict(pair=[a,b],start=left,stop=right,scale=scale,
                        feasible_gain=feasible_gain,gain=score-current,
                        eligible=False,screen='exact_flank_score_below_winner'))
                    continue
                full_score,next_switches=counted(trial)
                if abs(score-full_score)>margin:
                    raise RuntimeError('Localized interval score disagrees with full score')
                score=full_score
                fit_gain=score-current+workspace.penalty*(next_switches-switches)
                eligible=score>current+1e-6 and fit_gain>=-1e-6
                record=dict(pair=[a,b],start=left,stop=right,scale=scale,
                    feasible_gain=feasible_gain,gain=score-current,
                    genotype_fit_gain=fit_gain,eligible=bool(eligible))
                records.append(record)
                if eligible and (best is None or score>best[0]):
                    best=score,next_switches,trial,record
            accepted=None
            if best is not None: current,switches,rows,accepted=best
            history.append(dict(iteration=iteration,proposals=records,accepted=accepted,
                                seconds=time.monotonic()-started))
            fr._save(checkpoints,phase,dict(selected=rows,score=current,switches=switches,
                history=history,converged=best is None))
            if best is None: break
        result=component if np.array_equal(original,rows) else _reconstruct(rows,current,batch,component)
        if not np.array_equal(np.sort(result.discrete_haps,axis=0),np.sort(component.discrete_haps,axis=0)):
            raise RuntimeError('Paired interval changed local allele multiset')
        diagnostic=dict(component=number,initial_score=initial,final_score=current,
            changed=not np.array_equal(original,rows),history=history,
            approximation='feasible incumbent-painting boundary shortlist')
        fr._save(checkpoints,token,dict(block=result,diagnostic=diagnostic))
        outputs.append(result);diagnostics.append(diagnostic)
    return BlockResults(outputs),dict(model='painting_boundary_intervals_local_v1',components=diagnostics)
