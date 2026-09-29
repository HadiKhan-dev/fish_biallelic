"""Exact conditional-query reuse and bounded fragment seeds for count-up.

Fragment grouping restricts founder decisions, not sample-HMM transition bins.
Use only for tagged seed models; ordinary global escapes retain fine choices.
All proposals expand to original leaf rows before full-site acceptance.
"""
from collections import OrderedDict
from dataclasses import replace
from threading import Lock
import numpy as np
from . import path_search
from .packing import PreparedModels, packed_emissions, score_rows
from .workspace import resolve_threads
from ...core.parallel import numba_thread_scope

def groups_and_context(offsets, context, target_sites=2000):
    """Never cross a phase component or the supplied original L1 boundaries."""
    blocks = len(offsets)-1
    if context is None:
        intervals, rows = [(0,blocks)], [np.empty((0,blocks),np.int64)]
    else:
        intervals, rows = context
    groups, pieces = [], []
    for (left,right), paths in zip(intervals,rows):
        start = left
        while start < right:
            end = start+1
            while end < right and offsets[end]-offsets[start] < target_sites:
                end += 1
            groups.append((start,end))
            pieces.append(np.ascontiguousarray(paths[:,start-left:end-left],np.int64))
            start = end
    assert groups[0][0] == 0 and groups[-1][1] == blocks
    assert all(a[1] == b[0] for a,b in zip(groups,groups[1:]))
    return np.asarray(groups,np.int64), pieces

class Catalogue:
    def __init__(self,models,panel,groups,context):
        panel=np.asarray(sorted(panel,key=lambda row:row.tobytes()),np.int64)
        self.models,self.alphabets,_=path_search.coarsen_submodels(
            models,panel,groups,context)
        self.groups=groups
        self.maps=[{row.tobytes():i for i,row in enumerate(rows)}
                   for rows in self.alphabets]

    def encode(self, panel):
        return np.asarray([[lookup[row[start:end].tobytes()]
            for (start,end),lookup in zip(self.groups,self.maps)] for row in panel],np.int64)

    def decode(self, path):
        return np.concatenate([rows[int(choice)] for rows,choice in zip(self.alphabets,path)])

def tagged_models(base, offsets, context):
    owner = PreparedModels(base)
    # Only read-only numerical tables are shared; score/search caches stay local.
    owner._packed = packed_emissions(base)
    owner._coarse_groups, owner._coarse_context = groups_and_context(offsets,context)
    owner._coarse_catalogues = OrderedDict()
    owner._coarse_lock = Lock()
    return owner


def panel_key(panel):
    return panel.shape, tuple(sorted(row.tobytes() for row in panel))


def prepare(models, known, incumbent, threads):
    panel = np.ascontiguousarray(np.vstack((known,incumbent)),np.int64)
    key = panel_key(panel)
    with models._coarse_lock:
        found = models._coarse_catalogues.get(key)
        if found is None:
            with numba_thread_scope(resolve_threads(threads)):
                found = Catalogue(models,panel,models._coarse_groups,models._coarse_context)
            models._coarse_catalogues[key] = found
            while len(models._coarse_catalogues)>2:
                models._coarse_catalogues.popitem(last=False)
        else:
            models._coarse_catalogues.move_to_end(key)
    return found

_cache_lock = Lock()


def _array_key(value):
    return value.shape, value.dtype.str, value.tobytes()


def compute(models, known, incumbent, penalty, config, width, reverse, dual,
            window, thread_budget, background, candidate_choices, solve):
    """Cache only identical numerical problems, independently of outer budgets."""
    arguments = (models, known, incumbent, penalty, config, width, reverse,
                 dual, window, thread_budget, background, candidate_choices)
    if not isinstance(models, PreparedModels):
        return solve(*arguments)
    choices = (None if candidate_choices is None else
               tuple(_array_key(value) for value in candidate_choices))
    signature = (config.branch_cap, config.beam_width, config.window_blocks,
                 config.dual_search_sweeps)
    key = (_array_key(known), _array_key(incumbent), float(penalty),
           signature, width, bool(reverse), dual, window, choices)
    with _cache_lock:
        if not hasattr(models, "_conditional_query_cache"):
            models._conditional_query_cache = OrderedDict()
        cache = models._conditional_query_cache
        if key in cache:
            answer = cache[key]
            cache.move_to_end(key)
            return dict(answer, path=answer["path"].copy(), repeated_query=True)
    # No lock across independent numerical work. Duplicate cold requests may
    # compute twice; their threads still share the caller's bounded team.
    if not hasattr(models, "_coarse_groups") or candidate_choices is not None:
        answer = solve(*arguments)
    else:
        catalogue = prepare(models, known, incumbent, thread_budget or 1)
        encoded = catalogue.encode(np.vstack((known, incumbent)))
        span = max(1, int(np.median(catalogue.groups[:, 1] - catalogue.groups[:, 0])))
        settings = replace(config, window_blocks=max(
            1, (config.window_blocks + span - 1) // span))
        proposal = solve(catalogue.models, encoded[:-1], encoded[-1], penalty,
            settings, width, reverse, dual, window, thread_budget, None, None)
        proposed = catalogue.decode(proposal["path"])
        score = score_rows(models, np.vstack((known, proposed)), penalty)
        detail = dict(proposal.get("search_diagnostic") or {})
        detail.update(coarse_catalogue=True, original_blocks=len(models),
            coarse_blocks=len(catalogue.models), equivalent_incumbent_tie=False,
            fine_binned_proposal_score=score)
        answer = dict(path=proposed, score=score, search_diagnostic=detail)
    with _cache_lock:
        cache[key] = dict(answer, path=answer["path"].copy())
        cache.move_to_end(key)
        while len(cache) > 128:
            cache.popitem(last=False)
    return answer
