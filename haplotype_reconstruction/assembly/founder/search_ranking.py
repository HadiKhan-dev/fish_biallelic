"""Rank bounded proposal batches; every winner receives full-site scoring.

A coarse score is a lower bound, not proof that a skipped candidate loses.
Check the top four in original order and exhaust a category when it stalls.
Joint conditional proposals retain all individual contenders and are rescored.
"""
from collections import OrderedDict
from threading import Lock
import time
import numpy as np
from .proxy_primary import CompactPrimary
from .packing import score_rows

class Schedule:
    def __init__(self, *, full_quota=4, bin_size=200, proxy_paint=True):
        if full_quota is not None and (int(full_quota) != full_quota or full_quota < 1):
            raise ValueError("full_quota must be a positive integer or None")
        self.full_quota = full_quota
        self.bin_size = int(bin_size)
        self.proxy_paint = bool(proxy_paint)
        self.primary_first = True
        self.stats = {}
        self.lock = Lock()

    def add(self, key, value=1):
        with self.lock:
            self.stats[key] = self.stats.get(key, 0) + value

class ProxyRanking:
    """One call's bounded score cache; immutable proxy belongs to its workspace."""
    def __init__(self, schedule, workspace):
        self.schedule, self.workspace = schedule, workspace
        self.cache = OrderedDict()

    def proxy(self):
        cache = getattr(self.workspace, "_compact_proxies", None)
        if cache is None:
            cache = self.workspace._compact_proxies = {}
        width = self.schedule.bin_size
        if width not in cache:
            started = time.perf_counter()
            cache[width] = CompactPrimary(self.workspace, width)
            self.schedule.add("proxy_prepare_seconds", time.perf_counter() - started)
        return cache[width]

    def score(self, panel):
        key = (panel.shape, panel.tobytes())
        if key in self.cache:
            self.cache.move_to_end(key)
            self.schedule.add("proxy_score_cache_hits")
            return self.cache[key]
        value = self.proxy().score(panel)
        self.cache[key] = value
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return value

    def ordered(self, candidates, panel_index, current_best, record, category):
        """Check top-ranked members in native order, then exhaust on a stall.

        A category has stalled if it did not replace its entering full-score
        winner, including a predictive-only winner at exact primary equality.
        Checking the remainder in native order retains deterministic tie rules.
        """
        count, quota = len(candidates), self.schedule.full_quota
        if quota is None or count <= quota:
            self.schedule.add("exhaustive_categories")
            self.schedule.add("full_candidate_checks", count)
            yield from candidates
            return
        started = time.perf_counter()
        scores = [self.score(item[panel_index]) for item in candidates]
        self.schedule.add("proxy_rank_seconds", time.perf_counter() - started)
        self.schedule.add("ranked_candidates", count)
        top = set(sorted(range(count), key=lambda i: (-scores[i], i))[:quota])
        before = current_best()
        audit = dict(category=category, candidates=count, quota=quota,
                     proxy_top_indices=sorted(top), full_checked=0,
                     exhausted=False, skipped=0)
        record.setdefault("delayed_acceptance", []).append(audit)
        for index, item in enumerate(candidates):
            if index in top:
                audit["full_checked"] += 1
                self.schedule.add("full_candidate_checks")
                yield item
        if current_best() is before:
            audit["exhausted"] = True
            self.schedule.add("category_stall_fallbacks")
            for index, item in enumerate(candidates):
                if index not in top:
                    audit["full_checked"] += 1
                    self.schedule.add("full_candidate_checks")
                    yield item
        else:
            audit["skipped"] = count - quota
            self.schedule.add("skipped_full_candidate_checks", count - quota)

def joint_candidate(candidates,reference,score):
    """Best positive proxy proposal per founder; no combination of alleles."""
    initial=score(reference)
    choices={}
    for item in candidates:
        focal=int(item[0])
        if focal<0:continue  # A pre-existing whole-panel proposal is not a row.
        value=score(item[2])
        prior=choices.get(focal)
        if value>initial+1e-6 and (prior is None or value>prior[0]):
            choices[focal]=(value,item[2][focal])
    if len(choices)<2:return None,[]
    trial=reference.copy()
    for focal in sorted(choices):trial[focal]=choices[focal][1]
    return trial,sorted(choices)


class Ranking(ProxyRanking):
    def ordered(self,candidates,panel_index,current_best,record,category):
        candidates=list(candidates)
        joint=None
        if category.startswith('conditional_') and panel_index==2 and self.workspace is not None:
            reference=getattr(self.workspace,'_joint_reference',None)
            if reference is None:reference=self.workspace.reference
            if reference is not None:
                joint,focals=joint_candidate(candidates,reference,self.score)
                if joint is not None:
                    predicted=score_rows(self.workspace.models(),joint,self.workspace.penalty)
                    # -1 is diagnostic shorthand for a whole-panel proposal.
                    # The caller consumes the complete trial, never indexes it
                    # with this label; native candidates remain in native order.
                    candidates.append((-1,False,joint,predicted))
                    record.setdefault('joint_conditional_candidates',[]).append(dict(
                        category=category,founders=focals,individual_gains_not_added=True))
                    self.schedule.add('joint_conditional_proposals')
        yield from super().ordered(candidates,panel_index,current_best,record,category)
        winner=current_best()
        if joint is not None and winner is not None and np.array_equal(winner[1],joint):
            self.schedule.add('joint_conditional_category_winners')
