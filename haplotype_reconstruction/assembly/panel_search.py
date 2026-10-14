"""Certified panel search: same-count moves under the join likelihood, counts by the count rule.

Tier 1 is the bounded sweep: conditional fixed-painting gains generate same-count
proposals and the best improving panel is accepted after an exact score (a proposal
replacing one path is scored only on the samples its two-region bound leaves open,
panel_certify.replacement_scores; elsewhere its value is V(P minus f) exactly). At every
tier-1 stall the count step (panel_count, the pipeline's effective-carrier rule on
the total gain and its sequence part) decides one death (after deleting every path
without sequence of its own) or else one birth. Once the count step is quiescent, tier 2
bounds every same-count move of an explicit K-scaled neighbourhood (panel_certify)
and scores the survivors exactly (one-path moves only on the samples their bound
leaves open); an improving move returns the search to tier 1. A same-count move must
leave every changed path a private cell. A pool-saturation step re-runs the endpoint
beam with doubled quotas before a panel is certified (up to 256 paths per endpoint,
up to 1024 for a join whose 256 rung accepted a move). Each sweep has
O(N*m*K**2 + K**2*B*log(K)) work (candidate scores only where a block-restart bound
leaves a candidate's novelty open); a tier-2 pass O(N*m*K*(|C| + K**2)) (structural
moves share forward prefixes and backward suffixes, panel_certify.structural_bounds); a
count step O(N*m*K*|C|) plus sparse exact scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import nsmallest
import math
import threading
import time
import numba
import numpy as np
from numba import njit, prange

from . import panel_certify
from . import panel_count
from .founder import count_criterion
from . import panel_scoring
from . import chimera_scoring as scoring
from . import chimera_kernels as kernels
from .panel_candidates import (
    candidate_novelty, prepare_candidate_scores, replacement_proposals,
    local_proposals, boundary_proposals,
)


# Reported backstops, never expected to bind: total sweeps (tier-1 sweeps plus
# accepted tier-2 moves) <= base + per_row * max_b K_b, and the pool-saturation
# ladder. Every join doubles the endpoint quota up to POOL_QUOTA_DOUBLINGS times
# (16 * 2**4 = 256 paths per endpoint); a join whose 256 rung accepted a move climbs
# on, up to POOL_EXTENDED_DOUBLINGS (16 * 2**6 = 1024). A panel is certified only when
# no rung up to its ladder top adds a move; a move accepted at 1024 is the reported
# pool backstop.
SWEEP_BACKSTOP_BASE = 100
SWEEP_BACKSTOP_PER_ROW = 20
POOL_QUOTA_DOUBLINGS = 4
POOL_EXTENDED_DOUBLINGS = 6
# Distinct-path sets whose birth bounds are kept for reuse (memory: one samples-long vector
# per pool path and set).
BIRTH_SETS = 4


@dataclass(frozen=True)
class PanelSearchConfig:
    paths_per_endpoint: int = 16
    full_scores_per_kind: int = 16
    max_bins: int = 2000
    tensor_budget_mb: int = 256

    def __post_init__(self):
        for value in (
            self.paths_per_endpoint,
            self.full_scores_per_kind,
            self.max_bins,
            self.tensor_budget_mb
        ):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError("panel search budgets must be positive integers")


@njit(cache=True, parallel=True)
def conditional_fields(emission, painted, local_paths, panel_size):
    """Per-path replacement deltas and pair correction for a suffix exchange.

    Use the same emission values as the float64 full-score tensor. Homozygous
    painting states replace BOTH copies once, rather than treating them as
    two independent replacements. Uniform/masked input has zero deltas.
    """
    samples, local_k, _, bins = emission.shape
    unary = np.zeros((panel_size, local_k))
    pair = np.zeros((panel_size, panel_size))
    for founder in prange(panel_size):
        for sample in range(samples):
            for site in range(bins):
                state = painted[sample, site]
                first = state // panel_size
                second = state % panel_size
                if first != founder and second != founder:
                    continue
                a, b = local_paths[first], local_paths[second]
                current = np.float64(emission[sample, a, b, site])
                for candidate in range(local_k):
                    aa = candidate if first == founder else a
                    bb = candidate if second == founder else b
                    unary[founder, candidate] += np.float64(emission[sample, aa, bb, site]) - current
                if first == founder and first != second:
                    # Only this ordered painting contributes here. Each
                    # unordered pair is symmetrized after the parallel pass.
                    pair[first, second] += 2 * current - np.float64(emission[sample, a, a, site]) - np.float64(emission[sample, b, b, site])
    return unary, pair + pair.T


@njit(cache=True)
def fixed_paint_delta(emission, painted, old_local, new_local, panel_size):
    result = 0.
    for sample in range(len(painted)):
        for site in range(painted.shape[1]):
            state = painted[sample, site]
            a, b = state // panel_size, state % panel_size
            result += np.float64(emission[sample, new_local[a], new_local[b], site])
            result -= np.float64(emission[sample, old_local[a], old_local[b], site])
    return result


@njit(cache=True)
def _exclusion_scores(current):
    """All one-founder exclusions need at most three K² scans per sample."""
    samples, k, _ = current.shape
    best = np.empty(samples)
    excluded = np.empty((samples, k))
    for sample in range(samples):
        flat = np.argmax(current[sample])
        first, second = flat // k, flat % k
        best[sample] = current[sample, first, second]
        excluded[sample,:] = best[sample]
        for i in (first, second):
            maximum = -np.inf
            for a in range(k):
                for b in range(k):
                    if a != i and b != i:
                        maximum = max(maximum, current[sample, a, b])
            excluded[sample, i] = maximum
    return best, excluded


@njit(cache=True, parallel=True)
def _stream_static_gains(flat, start, haps, local, candidates, order, common, best, excluded,
                         chunks):
    """Accumulate blocks before mate maxima; retain only a K-vector per task.

    Candidates are visited in sorted order (order, common = leading blocks shared with the
    previous one) in contiguous chunks, one task each: the block sums of a shared prefix are
    reused, accumulated in block order as before; per candidate the samples are summed in order.
    Each task sums into rows of its own (the same chain per cell, from zero) and copies them
    out at the end: rows of tasks interleaved in candidate order would share cache lines.
    Block b's totals are flat[start[b] + (sample * K_b + a) * K_b + c] (K_b = haps[b]).
    """
    samples, k = excluded.shape
    blocks = len(start)
    gains = np.zeros((len(candidates), k))
    count = len(order)
    for chunk in prange(chunks):
        lo = (count * chunk) // chunks
        hi = (count * (chunk + 1)) // chunks
        if lo >= hi:
            continue
        cross = np.empty((blocks + 1, k))
        diagonal = np.empty(blocks + 1)
        own = np.zeros((hi - lo, k))
        for sample in range(samples):
            for position in range(lo, hi):
                candidate = order[position]
                shared = 0 if position == lo else min(common[position], blocks)
                if shared == 0:
                    cross[0, :] = 0.
                    diagonal[0] = 0.
                for block in range(shared, blocks):
                    kb = haps[block]
                    base = start[block] + sample * kb * kb
                    c = candidates[candidate, block]
                    diagonal[block + 1] = diagonal[block] + np.float64(flat[base + c * kb + c])
                    for partner in range(k):
                        cross[block + 1, partner] = (
                            cross[block, partner]
                            + np.float64(flat[base + c * kb + local[partner, block]]))
                row = cross[blocks]
                winner = np.argmax(row)
                first = row[winner]
                second = -np.inf
                for partner in range(k):
                    if partner != winner:
                        second = max(second, row[partner])
                for removed in range(k):
                    retained = second if removed == winner else first
                    score = max(excluded[sample, removed], retained, diagonal[blocks])
                    own[position - lo, removed] += score - best[sample]
        for position in range(lo, hi):
            for removed in range(k):
                gains[order[position], removed] = own[position - lo, removed]
    return gains.T


def static_replacement_gains(local, candidates, *, totals, workspace):
    """Exact all-replacement gains in the *no-switch* proposal objective.

    This allows pair reassignment, unlike fixed-painting deltas. Only two
    endpoints of the best old pair need an exclusion rescan. The best and
    second-best candidate partners handle all removed founders in O(K*M).
    No-switch scores screen proposals, not the final recombining model.
    The N*M*K candidate/mate tensor is streamed in native code, preserving
    float64 block accumulation and the sample-reduction order; candidates
    sharing a prefix (workspace: prepare_candidate_scores' sorted order) share
    its block sums; totals: the blocks' bin-summed emissions.
    """
    samples = totals[0].shape[0]
    k = len(local)
    current = np.zeros((samples, k, k))
    for b, total in enumerate(totals):
        a = local[:, b]
        current += total[:, a[:, None], a[None,:]]
    best, excluded = _exclusion_scores(current)
    _, order, common = workspace
    chunks = max(1, min(len(order), 4 * max(1, numba.get_num_threads())))
    # The kernel reads the block totals as one flat array: taking a block's array out of a
    # typed List in its inner loop would update that array's reference count, an atomic shared
    # by all threads, which made the kernel slower with more threads than with one.
    haps = np.asarray([total.shape[1] for total in totals], dtype=np.int64)
    start = np.zeros(len(totals), dtype=np.int64)
    start[1:] = np.cumsum([total.size for total in totals])[:-1]
    flat = np.concatenate([np.ascontiguousarray(total).ravel() for total in totals])
    return _stream_static_gains(flat, start, haps, local, candidates, order, common, best,
                                excluded, chunks)


def _cycle_permutation(preference):
    """Take cycles from a best-target graph; no cubic assignment solver."""
    size = len(preference)
    target = np.argmax(preference, axis=1)
    permutation = np.arange(size)
    finished = set()
    for first in range(size):
        if first in finished:
            continue
        path = []
        offset = {}
        current = first
        while current not in finished and current not in offset:
            offset[current] = len(path)
            path.append(current)
            current = int(target[current])
        if current in offset:
            cycle = path[offset[current]:]
            if len(cycle) > 1:
                for item in cycle:
                    permutation[item] = target[item]
        finished.update(path)
    return permutation


def _greedy_permutation(weights):
    """Quadratic-log greedy assignment plus a fixed quadratic swap search."""
    k = len(weights)
    permutation = np.full(k, -1, dtype=np.int64)
    used = np.zeros(k, bool)
    for flat in np.argsort(-weights.ravel(), kind='stable'):
        first, second = divmod(int(flat), k)
        if permutation[first] < 0 and not used[second]:
            permutation[first] = second
            used[second] = True
    return _improve_permutation(weights, permutation)


@njit(cache=True)
def _improve_permutation(weights, permutation):
    for _ in range(20):
        changed = False
        for i in range(len(permutation)):
            for j in range(i + 1, len(permutation)):
                a, b = permutation[i], permutation[j]
                if weights[i, b] + weights[j, a] > weights[i, a] + weights[j, b]:
                    permutation[i], permutation[j] = b, a
                    changed = True
        if not changed:
            break
    return permutation


def _materialize(description, selected, candidates):
    """Panel after a same-count move (paths keep their slots)."""
    kind, *args = description
    result = [list(path) for path in selected]
    if kind == 'replace':
        founder, candidate = args
        result[founder] = list(candidates[candidate])
    elif kind == 'local':
        founder, block, hap = args
        result[founder][block] = hap
    elif kind == 'onesided':
        first, second, boundary = args
        result[first][boundary:] = selected[second][boundary:]
    elif kind == 'splice':
        first, second, boundary = args
        result[first][boundary:], result[second][boundary:] = result[second][boundary:], result[first][boundary:]
    elif kind == 'cycle':
        boundary, permutation = args
        for first, second in enumerate(permutation):
            result[first][boundary:] = selected[second][boundary:]
    else:
        raise ValueError(kind)
    return [tuple(path) for path in result]


def _admissible(old, new, geometry, exempt=frozenset()):
    """A same-count move keeps k distinct paths, and every changed path owns sequence.

    A changed path must keep a private cell and stay farther than the near-identity
    tolerance from every other path; otherwise it would be a pure mosaic or a
    near-duplicate (q = 0), which the count rule deletes, so the move would turn a
    justified path into a deletion the count rule never decided. ``exempt`` slots (a seed
    and its copy during their refit) may duplicate each other or lack novelty; their
    panel is settled by the count rule afterwards.
    """
    if len(new) != len(old):
        return False
    checked = [i for i, (a, b) in enumerate(zip(old, new)) if a != b and i not in exempt]
    if not exempt and len(set(new)) != len(new):
        return False
    if not checked:
        return True
    private = panel_count.private_mask(np.asarray(new, dtype=np.int64))
    for i in checked:
        if not private[i].any():
            return False
        others = [path for j, path in enumerate(new) if j != i]
        if new[i] in others or geometry.near_identical(new[i], others):
            return False
    return True


def search_summary(records_per_block):
    """Per-level counts from the panel-search diagnostics of super-blocks."""
    summary = dict(searched=0, certified=0, changed=0, tier2_accepted=0,
                   saturation_accepted=0, tier2_passes=0, backstop_reached=0,
                   pool_backstop=0, extended_ladders=0, largest_pool=0, births=0, deaths=0,
                   pure_mosaic_deaths=0,
                   ambiguous=0, frozen=0, cycles_undecided=0, most_decisions=0, start_rows=0,
                   final_rows=0,
                   seed_births=0, seed_swaps=0, seed_refits=0, ladder_births=0,
                   trap_blocks=0, failing_final=0)
    for records in records_per_block:
        for record in records or ():
            if 'certified' in record:
                summary['searched'] += 1
                summary['certified'] += bool(record['certified'])
                accepted = record['tier2_accepted'] + record['saturation_accepted']
                summary['changed'] += accepted > 0
                summary['tier2_accepted'] += record['tier2_accepted']
                summary['saturation_accepted'] += record['saturation_accepted']
                summary['tier2_passes'] += record['tier2_passes']
                summary['backstop_reached'] += bool(record['backstop_reached'])
                summary['pool_backstop'] += bool(record['pool_backstop'])
                # Records written before the targeted ladder (e.g. passthrough blocks of a level
                # replayed from older checkpoints) never extended it.
                summary['extended_ladders'] += bool(record.get('ladder_extended', False))
                summary['largest_pool'] = max(summary['largest_pool'],
                                              record['pool_sizes']['largest'])
                count = record['count']
                summary['births'] += count['births']
                summary['deaths'] += count['deaths']
                summary['pure_mosaic_deaths'] += count['pure_mosaic_deaths']
                summary['ambiguous'] += count['ambiguous']
                summary['frozen'] += bool(count['births_frozen'])
                # Records written before T28 carry no cycle classification.
                summary['cycles_undecided'] += count.get('cycles_undecided', 0)
                summary['most_decisions'] = max(summary['most_decisions'], count['decisions'])
                summary['start_rows'] += count['start_size']
                summary['final_rows'] += record['final_panel_size']
                for key in ('seed_births', 'seed_swaps', 'seed_refits', 'ladder_births'):
                    summary[key] += count[key]
                summary['trap_blocks'] += count['trap_indicators'] > 0
                summary['failing_final'] += count['final_paths_failing_death_test'] or 0
    return summary


def evaluate_panel(paths, sub, penalty, samples, *, layout, paint=False, per_sample=False,
                   tensor_budget_mb=256, num_threads=None):
    """Same objective/traceback; score directly or paint (Viterbi traceback).

    Samples factor independently. Scores stream existing block emissions (layout: the
    chimera_scoring.EmissionLayout of sub); the
    painting runs over unordered path pairs (panel_scoring.paint_emissions) in
    sample chunks whose back pointers (int32, bins x pairs per sample) fit in
    tensor_budget_mb. Neither path prunes states or changes an individual's
    dynamic program or the final sample reduction.
    """
    kernels._resolve_threads(num_threads)
    if not paint:
        result = panel_scoring.score_panel(paths, sub, penalty, samples, layout)
        return result if per_sample else float(result.sum())
    indices = panel_scoring.panel_indices(paths, sub)
    bins = sum(item['n_bins'] for item in sub)
    bytes_per_sample = max(1, len(paths) * (len(paths) + 1) // 2 * bins * 4)
    chunk = max(1, min(samples, int(tensor_budget_mb * 1024 ** 2) // bytes_per_sample))
    result = np.empty((samples, bins), dtype=np.int32)
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        kernels._resolve_threads(num_threads)
        result[start:stop] = panel_scoring.paint_emissions(*layout.arrays(), indices,
                                                           float(penalty), start, stop)
    return result

def select_and_resolve(beam_results, fast_mesh, batch_blocks, global_probs, global_sites,
        *, beam_runner, config=PanelSearchConfig(), cc_scale=.5, num_threads=None,
        diagnostics=None):
    """Certified same-count search under the join likelihood plus the pipeline's count rule.

    beam_runner(quota, overrides) re-runs the batch's endpoint beam with a base
    quota and per-endpoint quotas. Same-count moves are accepted when they raise
    the join likelihood and leave every changed path a private cell: tier 1 (the
    bounded sweep) runs until it finds none; tier 2 then bounds every replace,
    local, onesided and splice move of N(P, C) and scores the survivors exactly.
    The count step (panel_count) runs at every tier-1 stall: paths without sequence of
    their own are deleted, else one death or one birth is decided by the
    effective-carrier rule on the total gain and its sequence part; tier 2 runs only
    when it is quiescent. At a panel certified for both, the beam is re-run with the quota
    doubled and replacements (same-count) and births (count rule) by the new paths
    are tested the same way, up to quota 256, or up to 1024 once a move was accepted
    at the 256 rung. The panel size follows from the count rule; the pool,
    not a cap, bounds it. A reported backstop never decides a normal result.

    The duplicate-seed refits of one count step are independent searches from the same
    panel: they run concurrently (chimera_kernels.concurrent_map), each with a share of
    num_threads, and are evaluated in seed order as before.
    """
    # A thread running one of the concurrent refits takes its share of the join's threads
    # (refit_threads.share) wherever the search resolves its thread count.
    join_threads, refit_threads = num_threads, threading.local()
    if callable(join_threads):
        def num_threads():
            share = getattr(refit_threads, 'share', None)
            return join_threads() if share is None else share()
    samples = global_probs.shape[0]
    blocks = len(batch_blocks)
    candidates = list(dict.fromkeys(tuple(path) for path, _ in beam_results))
    if not candidates:
        return []
    pen = scoring.compute_penalty(batch_blocks)
    spb = max(scoring.compute_spb(batch_blocks),
            math.ceil(sum(len(b.positions) for b in batch_blocks) / config.max_bins))
    cc = scoring.compute_cc(batch_blocks, samples, cc_scale)
    sub, layout = scoring.compute_subblock_emissions(
        batch_blocks,
        global_probs,
        global_sites,
        spb,
        num_threads=num_threads
    )
    # Beam values are dense local indices; full-score APIs consume hap keys.
    def keys(paths):
        return [[fast_mesh.reverse_mappings[b][h] for b, h in enumerate(path)] for path in paths]
    evaluations = 0
    evaluations_lock = threading.Lock()
    def score(paths):
        nonlocal evaluations
        with evaluations_lock:
            evaluations += 1
        return evaluate_panel(keys(paths), sub, pen, samples, layout=layout, num_threads=num_threads,
            tensor_budget_mb=config.tensor_budget_mb, per_sample=True)
    # Start: the anchor-covering panel of largest likelihood. It only seeds the
    # search; the count rule decides the panel size.
    initialization = []
    initial_panels = set()
    selected = None
    for anchor in range(blocks):
        seen = set()
        trial = []
        for path in candidates:
            if path[anchor] not in seen:
                trial.append(path)
                seen.add(path[anchor])
        signature = tuple(sorted(trial))
        if signature in initial_panels:
            continue
        initial_panels.add(signature)
        trial_scores = score(trial)
        trial_ll = float(trial_scores.sum())
        initialization.append(dict(anchor=anchor, panel_size=len(trial), likelihood=trial_ll))
        if selected is None or trial_ll > likelihood + 1e-8:
            selected, baseline_scores, likelihood = list(trial), trial_scores, trial_ll
            selected_anchor = anchor
    start_size = len(selected)
    bin_offsets = np.r_[0, np.cumsum([item['n_bins'] for item in sub])]
    static_totals = [item['bin_emissions'].sum(axis=3) for item in sub]
    original_pool = candidates
    original_set = set(original_pool)

    def set_pool(pool):
        """Tier-1 workspaces of the current candidate pool."""
        nonlocal candidates, candidate_array, candidate_workspace, capacity
        candidates = pool
        candidate_array = np.asarray(candidates, dtype=np.int64)
        candidate_workspace = prepare_candidate_scores(layout, candidate_array)
        capacity = len(candidates)
    candidate_array = candidate_workspace = capacity = None
    set_pool(original_pool)
    beam_cache = {}
    beam_seconds = [0.0]

    def extension_pool(quota, panel):
        """C0 plus the beam with base quota and the panel's row multiplicity."""
        counts = {}
        for path in panel:
            for block, row in enumerate(path):
                counts[(block, row)] = counts.get((block, row), 0) + 1
        overrides = {endpoint: quota * count for endpoint, count in counts.items()
                     if count >= 2}
        if quota == config.paths_per_endpoint and not overrides:
            return original_pool
        key = (quota, tuple(sorted(overrides.items())))
        if key not in beam_cache:
            if len(beam_cache) >= 4:
                beam_cache.clear()
            started = time.perf_counter()
            paths = dict.fromkeys(tuple(path) for path, _ in beam_runner(quota, overrides))
            beam_cache[key] = original_pool + [path for path in paths if path not in original_set]
            beam_seconds[0] += time.perf_counter() - started
        return beam_cache[key]

    total_bins = int(bin_offsets[-1])
    block_offsets = np.ascontiguousarray(bin_offsets[:-1], dtype=np.int64)
    block_sizes = [len(item['hap_keys']) for item in sub]
    dense_to_local = [np.asarray([item['key_to_local_idx'][key]
                                  for key in fast_mesh.reverse_mappings[b]], dtype=np.int64)
                      for b, item in enumerate(sub)]

    def to_local(paths):
        paths = np.asarray(paths, dtype=np.int64).reshape(-1, blocks)
        return np.stack([dense_to_local[b][paths[:, b]] for b in range(blocks)], axis=1)

    def exact(trial):
        trial_scores = score(trial)
        trial_ll = float(trial_scores.sum())
        return -2.0 * trial_ll, (trial_ll, trial_scores)

    forwards_cache = {}
    births_by_set = {}
    birth_sets = [BIRTH_SETS]       # raised while the seed refits of a count step run
    cache_lock = threading.Lock()   # concurrent seed refits share both caches

    def panel_forwards(panel):
        """Exclusion forwards of a panel (row order matters), kept for the last few panels: a
        sweep, its stall, tier 2 and the saturation ladder often share one panel.

        Birth bounds depend only on the set of distinct panel paths: the forward maxima and the
        maxima over the new path's states are exact over the same values, which duplicate rows
        and row order do not change. Panels with one path set therefore share them (kept for
        the last BIRTH_SETS sets), above all the duplicate-seed refits of a count step, whose
        panels start as P plus a copy and mostly continue through the same accepted moves."""
        key = tuple(map(tuple, panel))
        with cache_lock:
            found = forwards_cache.get(key)
        if found is None:
            kernels._resolve_threads(num_threads)
            computed = panel_certify.PanelForwards(layout, to_local(np.asarray(panel)), pen,
                                                   total_bins)
            with cache_lock:
                found = forwards_cache.get(key)
                if found is None:
                    found = computed
                    if len(forwards_cache) >= 3:
                        forwards_cache.pop(next(iter(forwards_cache)))
                    forwards_cache[key] = found
        path_set = tuple(sorted(set(key)))
        with cache_lock:
            shared = births_by_set.pop(path_set, None)
            if shared is None:
                shared = found.births
            elif shared is not found.births:
                shared.update(found.births)
                found.births = shared
            births_by_set[path_set] = shared
            while len(births_by_set) > birth_sets[0]:
                births_by_set.pop(next(iter(births_by_set)))
        return found

    def certify(panel, pool, pool_offset, structural, pool_capacity, panel_ll=None, slots=None,
                refit_scores=None):
        """Bound every same-count move of N(P, pool) (or only replacements by pool paths).

        slots: a seed refit; only moves changing these rows, which are exempt from the
        novelty rule of same-count moves. refit_scores: the exact one-path scores shared by
        the seed refits of one count step."""
        started = time.perf_counter()
        panel_ll = likelihood if panel_ll is None else panel_ll
        rows = np.asarray(panel, dtype=np.int64)
        moves = panel_certify.Moves(rows, pool_capacity)
        added = moves.add_pool(np.asarray(pool, dtype=np.int64).reshape(-1, blocks),
                               pool_offset)
        if structural:
            moves.add_structural(block_sizes, slots)
        moves.finish()
        forwards = panel_forwards(panel)
        kernels._resolve_threads(num_threads)
        bounds, sample_bounds = panel_certify.lower_bounds(moves, forwards, panel_ll, to_local,
                                                           block_offsets, replaced=slots)
        # A seed refit moves the seed and its copy freely (no novelty rule for any path while the
        # copies separate); its panel is settled by the count rule afterwards.
        exempt = frozenset() if slots is None else frozenset(range(len(panel)))
        if slots is not None:
            replacing = moves.kind == panel_certify.REPLACE
            bounds[replacing & ~np.isin(moves.first_removed, list(slots))] = np.inf
        combined = candidates if not pool_offset else candidates + list(pool)
        cache = {}
        sparse = dict(scores=0, samples=0)

        def exact_move(trial, move):
            # A survivor replacing one path is scored exactly per sample where its bound exceeds
            # V_n(P minus f) (one_path_move_scores); other moves by the full score.
            row = panel_certify.replaced_row(moves, move)
            shared = None
            if refit_scores is not None and row is not None:
                # The sparse scores are a function of the retained rows and the new path alone
                # (V_n(P minus f), the move's two-region bound, and V_n of the trial at the
                # samples the bound leaves open), whatever panel and move produced them.
                shared = (tuple(sorted(trial[:row] + trial[row + 1:])), trial[row])
                found = refit_scores.get(shared)
                if found is not None:
                    return found
            if not cache:
                cache['removal'] = forwards.removal_scores()
                cache['local_new'] = np.ascontiguousarray(to_local(moves.new_paths))
            kernels._resolve_threads(num_threads)
            found = panel_certify.one_path_move_scores(
                forwards, moves, cache['local_new'], cache['removal'], move, to_local(trial),
                panel_count.score_samples, sample_bounds.get(move))
            if found is None:
                return exact(trial)
            values, scored_samples = found
            sparse['scores'] += 1
            sparse['samples'] += scored_samples
            trial_ll = float(values.sum())
            result = -2.0 * trial_ll, (trial_ll, values)
            if shared is not None:
                refit_scores[shared] = result
            return result

        best_move, survivors, exact_scores = panel_certify.best_improvement(
            bounds, moves.canonical,
            lambda move: _materialize(moves.description(move), panel, combined),
            exact_move, -2.0 * panel_ll, signature=lambda trial: tuple(sorted(trial)),
            admissible=lambda trial: _admissible(panel, trial, geometry, exempt))
        record = dict(panel_size=len(panel), pool=len(combined), pool_paths_tested=added,
            neighbourhood=moves.counts(), moves=len(moves), survivors=int(survivors),
            exact_scores=int(exact_scores), sparse_exact_scores=sparse['scores'],
            sparse_exact_samples=sparse['samples'],
            recomputed_exclusions_per_sample=float(forwards.recomputed_per_sample),
            accepted=None if best_move is None else str(moves.description(best_move[2])),
            accepted_likelihood_change=None if best_move is None else -0.5 * float(best_move[0]),
            seconds=time.perf_counter() - started)
        return best_move, record, forwards

    geometry = panel_count.RowGeometry(batch_blocks)
    counter = panel_count.CountStep(layout=layout, total_bins=total_bins,
                                    offsets=block_offsets, penalty=pen,
                                    cost=cc, samples=samples, geometry=geometry, to_local=to_local,
                                    score=score)
    sweep_forwards = None    # (panel, exclusion forwards, removal scores) of the last sweep's panel
    born = set()             # paths added by births here, followed through same-count moves
    births_frozen = False
    # Panel size at the start and after every count change, and the changes that deleted a
    # path without sequence of its own (count_criterion.pre_cycle_count; diagnostics only).
    sizes = [start_size]
    cleanup_steps = set()
    count = dict(decisions=0, births=0, pool_births=0, seed_births=0, seed_swaps=0, deaths=0,
                 pure_mosaic_deaths=0, ambiguous=0, births_frozen=False, start_size=start_size,
                 cycles=0, cycles_undecided=0,
                 trap_indicators=0, seed_refits=0, ladder_births=0, final_paths_failing_death_test=None,
                 seconds=0.0, seed_seconds=0.0)
    from .founder_refinement import COUNT_RANKED_REFITS as SEED_REFITS

    def apply_same_count(new_panel, new_scores):
        nonlocal selected, baseline_scores, likelihood
        for old_path, new_path in zip(selected, new_panel):
            if old_path in born and old_path != new_path:
                born.discard(old_path)
                born.add(new_path)
        selected = list(new_panel)
        baseline_scores = new_scores
        likelihood = float(new_scores.sum())

    def improves(new_ll, old_ll):
        # The join score is -2 LL; t = max(1e-8, 1e-9 (1 + 2|LL|)) as in panel_certify.
        return 2.0 * (new_ll - old_ll) > max(1e-8, 1e-9 * (1.0 + 2.0 * abs(old_ll)))

    def tier1_sweep(panel, panel_scores, panel_ll):
        """One bounded sweep at ``panel``: the best improving admissible move among the top
        proposals of every same-count kind."""
        nonlocal sweep_forwards
        k = len(panel)
        started = time.perf_counter()
        sparse_scores = 0

        def proposal_scores(trial):
            # A proposal that replaces one path is scored exactly per sample where its two-region
            # bound exceeds V_n(P minus f) (panel_certify.replacement_scores); others in full.
            nonlocal sparse_scores, sweep_forwards
            changed = [i for i, (old, new) in enumerate(zip(panel, trial)) if old != new]
            if len(changed) != 1:
                return score(trial)
            if sweep_forwards is None or sweep_forwards[0] != tuple(panel):
                forwards = panel_forwards(panel)
                sweep_forwards = (tuple(panel), forwards, forwards.removal_scores())
            kernels._resolve_threads(num_threads)
            values, _ = panel_certify.replacement_scores(
                sweep_forwards[1], sweep_forwards[2], changed[0], to_local([trial[changed[0]]])[0],
                to_local(trial), panel_count.score_samples)
            sparse_scores += 1
            return values

        kernels._resolve_threads(num_threads)
        painted = evaluate_panel(keys(panel), sub, pen, samples, layout=layout, paint=True,
            num_threads=num_threads, tensor_budget_mb=config.tensor_budget_mb)
        local = np.asarray(panel, dtype=np.int64)
        fields = []
        corrections = []
        for b, em in enumerate(sub):
            painting = np.ascontiguousarray(painted[:, bin_offsets[b]:bin_offsets[b + 1]])
            unary, pair = conditional_fields(em['bin_emissions'], painting, local[:, b], k)
            fields.append(unary)
            corrections.append(pair)
        panel_set = set(panel)
        proposals = {kind: [] for kind in same_count_kinds}
        eligible = np.asarray([c for c, path in enumerate(candidates)
                             if path not in panel_set], dtype=np.int64)
        proposal_counts = {}
        eligible_mask = np.zeros(len(candidates), dtype=bool)
        eligible_mask[eligible] = True
        novelty, novelty_scored = candidate_novelty(local, candidate_array, pen, panel_scores,
                                                    eligible_mask, workspace=candidate_workspace)
        novel = [c for c in np.argsort(-novelty, kind='stable')
               if candidates[c] not in panel_set and novelty[c] > 0]
        for c in novel[:config.full_scores_per_kind]:
            similarity = np.mean(local == candidate_array[c], axis=1)
            for founder in np.argsort(-similarity, kind='stable')[:2]:
                proposals['novel_replace'].append((float(novelty[c] * (.5 + similarity[founder])),
                    ('replace', int(founder), int(c))))
        static_gains = static_replacement_gains(local, candidate_array, totals=static_totals,
                                                workspace=candidate_workspace)
        gains = np.zeros((k, len(candidates)))
        for b in range(blocks):
            gains += fields[b][:, candidate_array[:, b]]
        proposals['static_replace'] = replacement_proposals(
            static_gains, eligible, config.full_scores_per_kind)
        proposals['replace'] = replacement_proposals(
            gains, eligible, config.full_scores_per_kind)
        proposal_counts['replace'] = proposal_counts['static_replace'] = k * len(eligible)
        proposals['local'], proposal_counts['local'] = local_proposals(
            fields, local, config.full_scores_per_kind)
        proposal_counts['onesided'] = (blocks - 1) * k * (k - 1)
        proposal_counts['splice'] = (blocks - 1) * k * (k - 1) // 2
        proposal_counts['switch_splice'] = proposal_counts['splice']
        # A lower-penalty painting reveals coordinated apparent switches.
        # It generates proposals only; acceptance uses the join score at pen.
        exploratory = evaluate_panel(keys(panel), sub, min(pen, 10.), samples, layout=layout,
            paint=True,
            num_threads=num_threads, tensor_budget_mb=config.tensor_budget_mb)
        preference = np.zeros((k, k))
        pair_gain = np.zeros((k, k))
        for boundary in range(blocks - 1, 0, -1):
            preference += fields[boundary][:, local[:, boundary]]
            pair_gain += corrections[boundary]
            proposals['onesided'].extend(boundary_proposals(preference, boundary,
                config.full_scores_per_kind, 'onesided'))
            delta = preference + preference.T + pair_gain
            proposals['splice'].extend(boundary_proposals(
                delta, boundary, config.full_scores_per_kind, 'splice'))
            votes = scoring.boundary_vote_matrix(exploratory, int(bin_offsets[boundary]), k)
            switch_gain = votes + votes.T - np.diag(votes)[:, None] - np.diag(votes)[None,:]
            proposals['switch_splice'].extend(boundary_proposals(
                switch_gain, boundary, config.full_scores_per_kind, 'splice'))
            permutation = _greedy_permutation(votes)
            if np.any(permutation != np.arange(k)):
                gain = float(votes[np.arange(k), permutation].sum() - np.trace(votes))
                proposals['switch_cycle'].append((gain, ('cycle', boundary, tuple(map(int, permutation)))))
            permutation = _cycle_permutation(preference)
            if np.any(permutation != np.arange(k)):
                new = local.copy()
                new[:, boundary:] = local[permutation, boundary:]
                gain = 0.
                for b in range(boundary, blocks):
                    gain += fixed_paint_delta(sub[b]['bin_emissions'],
                        painted[:, bin_offsets[b]:bin_offsets[b + 1]], local[:, b], new[:, b], k)
                proposals['cycle'].append((float(gain), ('cycle', boundary, tuple(map(int, permutation)))))
        start_evaluations = evaluations
        best = None
        evaluated = []
        evaluated_by_kind = {}
        inadmissible = 0
        boundary_acceptances = []
        scored = set()
        current_signature = tuple(sorted(panel))
        for kind, items in proposals.items():
            # Only q proposals were ever considered, before de-duplication.
            # Stable top-q selection preserves those exact edits and tie order.
            budget = config.full_scores_per_kind
            ranked = nsmallest(budget, items, key=lambda item: (-item[0], item[1]))
            evaluated_by_kind[kind] = 0
            for proposal_rank, (gain, description) in enumerate(ranked, 1):
                trial = _materialize(description, panel, candidates)
                signature = tuple(sorted(trial))
                if signature in scored or signature == current_signature:
                    continue
                scored.add(signature)
                if not _admissible(panel, trial, geometry):
                    inadmissible += 1
                    continue
                trial_scores = proposal_scores(trial)
                trial_ll = float(trial_scores.sum())
                evaluated_by_kind[kind] += 1
                evaluated.append((float(trial_ll), str(description)))
                improving = improves(trial_ll, panel_ll)
                if improving and proposal_rank == budget:
                    boundary_acceptances.append(kind)
                if improving and (best is None or trial_ll > best[0]):
                    best = (trial_ll, trial, description, gain, trial_scores)
        leading = sorted([(float(panel_ll), "current panel"), *evaluated], reverse=True)[:2]
        record = dict(panel_size=k, candidate_pool=len(candidates),
            proposals={kind: proposal_counts.get(kind, len(items)) for kind, items in proposals.items()},
            full_scores=evaluations - start_evaluations, sparse_scores=sparse_scores,
            novelty_scored=novelty_scored, likelihood=float(panel_ll),
            accepted=None if best is None else str(best[2]),
            full_scores_per_kind=config.full_scores_per_kind,
            evaluated_by_kind=evaluated_by_kind, inadmissible=inadmissible,
            truncated_kinds=[kind for kind, items in proposals.items()
                             if proposal_counts.get(kind, len(items)) > config.full_scores_per_kind],
            improving_at_budget_boundary=boundary_acceptances,
            leading_evaluated_alternatives=leading,
            scope="this sweep only; proxy-truncated and ungenerated edits are untested",
            seconds=time.perf_counter() - started)
        return best, record

    def share_threads(function):
        """function(item) run as an item of kernels.concurrent_map: the search's thread
        resolution in this thread takes the item's share."""
        def run(item, threads):
            refit_threads.share = threads if callable(join_threads) else None
            try:
                return function(item)
            finally:
                refit_threads.share = None
        return run

    def refit_seeds(seeds, refit_scores):
        """The duplicate-seed refits: for each seed, certified best-improvement from P plus a copy
        of the seed over the same-count moves that change the seed or its copy (both exempt from
        the novelty rule meanwhile). Returns (panel, scores, likelihood, passes) per seed.

        The refits are independent searches, advanced together one certify pass at a time: the
        passes of one round run concurrently (each refit with a share of the threads); before
        them, the pool's birth bounds of every path set the round meets are computed once with
        all threads (they depend on the path set alone, and the refits mostly reach the same
        sets), and the caches keep all of them through the round."""
        states = [[selected + [selected[seed]], baseline_scores, likelihood,
                   frozenset((seed, len(selected))), []] for seed in seeds]
        active = list(range(len(states)))

        def certify_pass(index):
            panel, _, panel_ll, slots, _ = states[index]
            return certify(panel, candidates, 0, True, len(candidates) + 1, panel_ll=panel_ll,
                           slots=slots, refit_scores=refit_scores)[:2]
        for _ in range(sweep_backstop):
            if not active:
                break
            panels = {}
            for index in active:
                panel = states[index][0]
                panels.setdefault(tuple(sorted(set(map(tuple, panel)))), panel)
            birth_sets[0] = max(BIRTH_SETS, len(panels))
            for path_set, panel in panels.items():
                members = set(path_set)
                outside = [path for path in candidates if path not in members]
                forwards = panel_forwards(panel)
                if outside:
                    kernels._resolve_threads(num_threads)
                    forwards.birth_values(to_local(outside), block_offsets)
            moved = []
            outcomes = kernels.concurrent_map(share_threads(certify_pass), active, join_threads)
            for index, (best_move, record) in zip(active, outcomes):
                state = states[index]
                state[4].append(dict(accepted=record['accepted'], survivors=record['survivors'],
                                     exact=record['exact_scores'],
                                     change=record['accepted_likelihood_change']))
                if best_move is not None:
                    state[0], state[2], state[1] = (list(best_move[3]), best_move[4][0],
                                                    best_move[4][1])
                    moved.append(index)
            active = moved
        birth_sets[0] = BIRTH_SETS
        return [(state[0], state[1], state[2], state[4]) for state in states]

    def deaths(forwards):
        """Deaths by the count rule; True if any path died.

        Paths without sequence of their own are deleted unconditionally, one after another
        without re-certifying in between (largest V(P minus f) first); otherwise the best-F
        candidate (largest V(P minus f)) is decided by the rule, and a deletion returns the
        search to tier 1 (the next death is decided at the re-optimized panel). Deleting a
        path that a birth here added closes a count cycle and freezes births; the deletion
        stands. It is recorded as undecided when it is ambiguous (its bootstrap band contains
        zero) and the cycle began with a decided deletion (count_criterion.pre_cycle_count;
        deleting a path without sequence of its own is not a decision)."""
        nonlocal selected, baseline_scores, likelihood, births_frozen
        changed = False
        while True:
            outcome = counter.deletion(selected, forwards, baseline_scores)
            if outcome is None:
                break
            record, rest, values = outcome
            removed = selected[record['founder']]
            record['tagged'] = bool(removed in born)
            if rest is not None and record['tagged']:
                count['cycles'] += 1
                if not record['pure_mosaic']:
                    origin = count_criterion.pre_cycle_count(sizes, len(selected), cleanup_steps)
                    record['cycle_start_size'] = origin
                    if origin >= len(selected) and record['ambiguous']:
                        record['cycle_undecided'] = True
                        count['cycles_undecided'] += 1
            if diagnostics is not None:
                diagnostics.append(dict(stage='count_death', **record))
            if rest is None:
                break
            if removed in born:
                births_frozen = True       # count cycle: no further births here
                born.discard(removed)
            count['decisions'] += 1
            count['deaths'] += 1
            count['pure_mosaic_deaths'] += record['pure_mosaic']
            count['ambiguous'] += record['ambiguous']
            selected, baseline_scores, likelihood = rest, values, float(values.sum())
            sizes.append(len(selected))
            if record['pure_mosaic']:
                cleanup_steps.add(len(sizes) - 1)
            changed = True
            if not record['pure_mosaic']:
                break
            forwards = panel_forwards(selected)
        return changed

    def count_step(forwards, pool=None, seeds=False):
        started = time.perf_counter()
        try:
            return _count_step(forwards, pool, seeds)
        finally:
            count['seconds'] += time.perf_counter() - started

    def _count_step(forwards, pool=None, seeds=False):
        """One count decision; True if the panel changed. pool None: deaths, then births from the
        current pool, then (seeds=True, at a certified panel) duplicate-seed births; otherwise
        births by the given (saturation) paths only."""
        nonlocal selected, baseline_scores, likelihood, births_frozen
        if pool is None:
            if deaths(forwards):
                return True
            pool_paths = candidates
        else:
            pool_paths = pool
        if births_frozen:
            return False
        record, larger_panel, values = counter.birth(selected, pool_paths, forwards, baseline_scores)
        count['trap_indicators'] = max(count['trap_indicators'], record['trap_indicators'])
        best = None if larger_panel is None else (float(values.sum()), larger_panel, values, record, 'pool')
        if seeds and best is None and counter.indicators:
            # Duplicate-seed births (T15's addition generator): pure-mosaic pool births that pass
            # the rule on their total gain mark a panel path covering a founder's sequence (the
            # chimera trap). Copy the main coverer of the strongest ones, refit the two copies,
            # and keep the refit panel if it survives its own death test and LL rises.
            if len(selected) <= SEED_REFITS:
                chosen = list(range(len(selected)))      # T15: every seed when k <= r
            else:
                strongest = []
                for _, path in counter.indicators[:3 * SEED_REFITS]:
                    gain = counter.total_gain(selected, path, forwards, baseline_scores)
                    if gain is not None:
                        strongest.append((-gain, path))
                strongest.sort(key=lambda item: item[0])
                chosen = []
                for _, path in strongest:
                    seed = panel_count.main_coverer(selected, path)
                    if seed not in chosen:
                        chosen.append(seed)
                    if len(chosen) >= SEED_REFITS:
                        break
            seed_started = time.perf_counter()
            # Every refit starts from P plus a copy, so the refits of one count step reach many of
            # the same one-path moves (above all replace(seed, c) = P + c): each is scored once
            # (its value does not depend on the refit that scored it).
            shared_scores = {}
            refits = refit_seeds(chosen, shared_scores)
            settles = kernels.concurrent_map(
                share_threads(lambda refit: counter.settle(refit[0], refit[1])), refits,
                join_threads)
            for seed, (_, _, _, used), (settled, settled_scores, settle_record, survives) in zip(
                    chosen, refits, settles):
                count['seed_refits'] += 1
                settled_ll = float(settled_scores.sum())
                seed_record = dict(kind='seed_birth', seed=seed, refit_passes=len(used), refit=used,
                                   settled_size=len(settled), survives=bool(survives),
                                   likelihood_change=settled_ll - likelihood, settle=settle_record)
                if diagnostics is not None:
                    diagnostics.append(dict(stage='count_seed', **seed_record))
                if (survives and len(settled) >= len(selected) and improves(settled_ll, likelihood)
                        and (best is None or settled_ll > best[0])):
                    best = (settled_ll, settled, settled_scores, seed_record, 'seed')
            count['seed_seconds'] += time.perf_counter() - seed_started
        if diagnostics is not None:
            diagnostics.append(dict(stage='count_birth', **record))
        if best is None:
            return False
        new_ll, new_panel, new_scores, new_record, source = best
        old = set(selected)
        if len(new_panel) > len(selected):
            count['decisions'] += 1
            count['births'] += 1
            count['seed_births' if source == 'seed' else 'pool_births'] += 1
            count['ambiguous'] += bool(new_record.get('ambiguous'))
            born.update(path for path in new_panel if path not in old)
        else:
            count['seed_swaps'] += 1
        selected, baseline_scores, likelihood = list(new_panel), new_scores, new_ll
        if len(selected) != sizes[-1]:
            sizes.append(len(selected))
        return True

    def final_census(forwards):
        """Diagnostics only: final paths that would fail their own death test."""
        failing = 0
        k = len(selected)
        if k < 2:
            return 0
        removal = forwards.removal_scores()
        for founder in range(k):
            rest = [path for i, path in enumerate(selected) if i != founder]
            cover = panel_count.covered_rows(np.asarray(rest, dtype=np.int64))
            if not panel_count.has_novelty(selected[founder], rest, cover, geometry):
                failing += 1
                continue
            shadow_founder, lost = panel_count.shadow_path(selected[founder], cover, geometry)
            shadow_panel = list(selected)
            shadow_panel[founder] = shadow_founder
            decision = count_criterion.decide_with_sequence(
                baseline_scores, removal[:, founder], score(shadow_panel), cc, kind='deletion',
                config=panel_count.count_config(), seed_key=('join_census', founder))
            failing += not decision.accept_larger
        return failing

    same_count_kinds = ('replace', 'static_replace', 'novel_replace', 'local', 'onesided', 'splice',
                        'cycle', 'switch_splice', 'switch_cycle')
    sweep_backstop = SWEEP_BACKSTOP_BASE + SWEEP_BACKSTOP_PER_ROW * max(block_sizes)
    quota = config.paths_per_endpoint
    ladder_top = config.paths_per_endpoint * 2 ** POOL_QUOTA_DOUBLINGS
    quota_backstop = config.paths_per_endpoint * 2 ** POOL_EXTENDED_DOUBLINGS
    sweeps = phase_sweeps = tier1_sweeps = tier2_passes = tier2_accepted = 0
    saturation_steps = saturation_accepted = 0
    certified = backstop_reached = pool_backstop = False
    largest_pool = len(candidates)
    final_forwards = None
    while True:
        if phase_sweeps >= sweep_backstop:
            backstop_reached = True
            print(f"  [panel search] sweep backstop {sweep_backstop} reached "
                  f"(panel {len(selected)}, pool {len(candidates)}): not certified", flush=True)
            break
        best, record = tier1_sweep(selected, baseline_scores, likelihood)
        record.update(sweep=tier1_sweeps + 1)
        if diagnostics is not None:
            diagnostics.append(record)
        tier1_sweeps += 1
        sweeps += 1
        phase_sweeps += 1
        if best is not None:
            apply_same_count(best[1], best[4])
            continue
        # A tier-1 stall: count decisions first (re-certification only once they are quiescent).
        stall_forwards = panel_forwards(selected)
        if count_step(stall_forwards):
            phase_sweeps = 0
            continue
        # Tier 2 at the stall, on C0 plus the beam with the panel's multiplicity.
        pool_before = len(candidates)
        beam_before = beam_seconds[0]
        pool = extension_pool(quota, selected)
        if pool is not candidates:
            set_pool(pool)
            largest_pool = max(largest_pool, len(candidates))
            # The pool grew: births by its new paths are count moves of this stall too.
            if count_step(stall_forwards):
                phase_sweeps = 0
                continue
        tier2_passes += 1
        best_move, record, forwards = certify(selected, candidates, 0, True, capacity)
        record.update(tier=2, tier2_pass=tier2_passes, quota=quota, pool_before=pool_before,
                      beam_seconds=beam_seconds[0] - beam_before)
        if diagnostics is not None:
            diagnostics.append(record)
        if best_move is not None:
            tier2_accepted += 1
            sweeps += 1
            phase_sweeps += 1
            apply_same_count(best_move[3], best_move[4][1])
            continue
        # Same-count certified and count-quiescent: duplicate-seed births (the chimera trap).
        if count_step(forwards, seeds=True):
            phase_sweeps = 0
            continue
        # Pool saturation: certify only when no doubled quota up to the ladder top adds an
        # improving replacement or accepted birth. The top is 256 for every join; once a
        # move was accepted at the 256 rung (the quota reached it) the ladder climbs on to
        # the backstop.
        if quota >= ladder_top:
            ladder_top = quota_backstop
        if 2 * quota > ladder_top:
            pool_backstop = True
            print(f"  [panel search] pool backstop: a move was accepted at quota "
                  f"{quota} = backstop (panel {len(selected)}): not certified", flush=True)
            final_forwards = forwards
            break
        checked = set(candidates)
        level = quota
        accepted = False
        while not accepted and 2 * level <= ladder_top:
            level *= 2
            saturation_steps += 1
            beam_before = beam_seconds[0]
            doubled = [path for path in extension_pool(level, selected) if path not in checked]
            best_move, record, level_forwards = certify(selected, doubled, len(candidates), False,
                                                        len(candidates) + len(doubled))
            checked.update(doubled)
            record.update(stage='pool_saturation', quota=quota, tested_quota=level,
                          new_paths=len(doubled), beam_seconds=beam_seconds[0] - beam_before)
            if diagnostics is not None:
                diagnostics.append(record)
            if best_move is not None:
                apply_same_count(best_move[3], best_move[4][1])
                accepted = True
            elif doubled and count_step(level_forwards, pool=doubled):
                count['ladder_births'] += 1
                accepted = True
        if not accepted:
            certified = True
            final_forwards = forwards
            break
        saturation_accepted += 1
        sweeps += 1
        phase_sweeps = 0
        quota = level
        set_pool(extension_pool(quota, selected))
        largest_pool = max(largest_pool, len(candidates))
    count['births_frozen'] = births_frozen
    if final_forwards is not None and diagnostics is not None:
        count['final_paths_failing_death_test'] = final_census(final_forwards)
    if diagnostics is not None:
        diagnostics.append(dict(total_full_scores=evaluations, final_panel_size=len(selected),
            final_likelihood=float(likelihood), initialization=initialization,
            selected_initial_anchor=selected_anchor, sweeps=sweeps, tier1_sweeps=tier1_sweeps,
            sweep_backstop=sweep_backstop, tier2_passes=tier2_passes,
            tier2_accepted=tier2_accepted, saturation_steps=saturation_steps,
            saturation_accepted=saturation_accepted, quota=quota, ladder_top=ladder_top,
            ladder_extended=ladder_top == quota_backstop, quota_backstop=quota_backstop,
            beam_seconds=beam_seconds[0], count=dict(count),
            count_exact_scores=counter.exact_scores,
            count_exact_sample_scores=counter.exact_sample_scores,
            pool_sizes=dict(original=len(original_pool), final=len(candidates),
                            largest=largest_pool),
            certified=certified, backstop_reached=backstop_reached,
            pool_backstop=pool_backstop))
    return [(list(path), float(likelihood)) for path in selected]
