"""Budgeted sparse coordinates alternating with chromosome-wide escapes.

Both steps spend the existing iteration budget. Every changed endpoint must
improve canonical full-site support (or predictive support at an exact primary
tie). Coarse search can miss a better path; it does not change likelihoods,
masks, count penalties or allele sources. No target founder count is imposed.
"""
import time
from dataclasses import replace
import numpy as np
from numba import njit, prange
from .. import founder_refinement as fr
from .workspace import resolve_threads
from ...core import parallel
from .sparse_primary import ReusedPrimary
from .search_ranking import Schedule as SearchBudget
from . import global_search

@njit(cache=True, parallel=True, nogil=True)
def sweep(selected, painting, data, data_offsets, bin_offsets, bins, sizes):
    """Increase the fixed-painting objective; sample labels remain unchanged."""
    founders, blocks = selected.shape
    proposed = selected.copy()
    gains = np.zeros(blocks, np.float64)
    for block in prange(blocks):
        size = sizes[block]
        for focal in range(founders):
            unary = np.zeros(size, np.float64)
            for sample in range(len(bins)):
                width, base = bins[sample, block], data_offsets[sample, block]
                for marker in range(width):
                    state = painting[bin_offsets[sample, block] + marker]
                    first, second = state // founders, state % founders
                    if first != focal and second != focal:
                        continue
                    row_a, row_b = proposed[first, block], proposed[second, block]
                    for candidate in range(size):
                        a = candidate if first == focal else row_a
                        b = candidate if second == focal else row_b
                        unary[candidate] += data[base + (a * size + b) * width + marker]
            current = proposed[focal, block]
            winner = int(np.argmax(unary))
            gain = unary[winner] - unary[current]
            # Keep the incumbent on ties, including wholly unobserved rows.
            if gain > 1e-8:
                proposed[focal, block] = winner
                gains[block] += gain
    return proposed, gains.sum()


class CoordinateSearch(SearchBudget):
    def panel(self, selected, leaves, offsets, evidence, complete, submodels,
              penalty, evaluate, config, checkpoints, token, num_threads,
              prepared_evidence=None, macro_context=None, *, dual=False,
              window=False, workspace=None):
        if workspace is None:
            raise RuntimeError('Sparse founder search requires canonical workspace')
        phase = token + '.sparse_coordinate_v1'
        saved = fr._load(checkpoints, phase)
        if saved is not None:
            return saved['selected'], saved['score'], saved['history']
        rows = selected.copy()
        score = workspace.canonical(rows)
        history = []
        # Repeated named search passes need not recompute a verified local
        # fixed point. No cache is installed for a budget-exhausted endpoint.
        final = getattr(workspace, '_coordinate_verified_fixed_point', None)
        if final is not None and np.array_equal(rows, final):
            history.append(dict(search='coordinate_verified_fixed_point_reuse', accepted_gain=0.))
            fr._save(checkpoints, phase, dict(selected=rows, score=score, history=history))
            return rows, score, history
        iteration = 0
        converged = False
        while iteration < config.max_iterations:
            started = time.monotonic()
            with parallel.numba_thread_scope(resolve_threads(num_threads)):
                proxy = ReusedPrimary(workspace, rows)
                proxy_score = proxy.score(rows)
            margin = max(1e-6, 1e-10 * abs(score))
            if abs(proxy_score - score) > margin:
                raise RuntimeError('Coordinate anchor changed the incumbent objective')
            initial = rows.copy()
            trial = rows.copy()
            sparse_steps = 0
            while iteration < config.max_iterations:
                with parallel.numba_thread_scope(resolve_threads(num_threads)):
                    painting = proxy.paint_compact(trial)[0]
                    proposed, fixed_gain = sweep(trial, painting, proxy.data,
                        proxy.data_offsets, proxy.bin_offsets, proxy.bins, proxy.sizes)
                    candidate_proxy = proxy.score(proposed)
                if candidate_proxy + margin < proxy_score:
                    raise RuntimeError('Fixed-painting coordinate sweep reduced sparse score')
                iteration += 1
                sparse_steps += 1
                changed = not np.array_equal(proposed, trial)
                trial, proxy_score = proposed, candidate_proxy
                if not changed:
                    break
            checked = workspace.canonical(trial)
            if checked + margin < proxy_score:
                raise RuntimeError('Sparse edited-panel score is not a lower bound')
            changed = not np.array_equal(trial, initial)
            tie = (workspace.predictive(trial) - workspace.predictive(rows)
                   if changed and checked == score else 0.)
            accepted = changed and (checked > score + 1e-6 or (checked == score and tie > 1e-6))
            history.append(dict(search='coordinate_sparse_epoch', sparse_steps=sparse_steps,
                accepted_gain=checked-score if accepted else 0., proxy_score=proxy_score,
                full_score=checked, accepted=bool(accepted), seconds=time.monotonic()-started))
            self.add('coordinate_sparse_steps', sparse_steps)
            self.add('coordinate_full_endpoints')
            if accepted:
                rows, score = trial, checked
            elif not changed:
                converged = True
                break
            else:
                break
        if converged:
            workspace._coordinate_verified_fixed_point = rows.copy()
        fr._save(checkpoints, phase, dict(selected=rows, score=score, history=history))
        return rows, score, history

class BudgetedSearch(CoordinateSearch):
    def panel(self,selected,leaves,offsets,evidence,complete,submodels,penalty,
              evaluate,config,checkpoints,token,num_threads,prepared_evidence=None,
              macro_context=None,*,dual=False,window=False,workspace=None):
        args=(leaves,offsets,evidence,complete,submodels,penalty)
        if not dual or window:
            return super().panel(selected,*args,evaluate,config,checkpoints,
                token,num_threads,prepared_evidence,macro_context,dual=dual,
                window=window,workspace=workspace)
        phase=token+'.sparse_global_v1'
        saved=fr._load(checkpoints,phase)
        if saved is not None:return saved['selected'],saved['score'],saved['history']
        selected=selected.copy();score=workspace.canonical(selected);history=[]
        remaining=max(0,config.max_iterations);round_number=0
        while remaining:
            prefix=phase+f'.round{round_number}'
            if remaining>1:
                # Never spend a different iteration budget because an earlier
                # named phase left a transient cache which a restart would lose.
                if hasattr(workspace,'_coordinate_verified_fixed_point'):
                    del workspace._coordinate_verified_fixed_point
                selected,score,local=super().panel(selected,*args,evaluate,
                    replace(config,max_iterations=remaining-1),checkpoints,
                    prefix+'.local',num_threads,prepared_evidence,macro_context,
                    dual=True,window=False,workspace=workspace)
                used=sum(item.get('sparse_steps',0) for item in local)
                if not 0<=used<remaining:
                    raise RuntimeError('Local sweeps exceeded alternating search budget')
                remaining-=used
                history+=local
            before=selected.copy()
            selected,score,escape=global_search.refine_panel(selected,*args,
                evaluate,replace(config,max_iterations=1),checkpoints,
                prefix+'.global',num_threads,prepared_evidence,
                macro_context=macro_context,dual=True,window=False,
                workspace=workspace,_schedule=self)
            remaining-=1
            history.append(dict(search='budgeted_global_escape',round=round_number,
                proposal_history=escape,remaining_iterations=remaining,
                accepted_gain=sum(item.get('accepted_gain',0.) for item in escape)))
            round_number+=1
            if np.array_equal(before,selected):break
        fr._save(checkpoints,phase,dict(selected=selected,score=score,history=history))
        return selected,score,history

def refine_panel(*args, **kwargs):
    """Ordinary production entry, with no process-global replacement hooks."""
    return BudgetedSearch().panel(*args, **kwargs)
