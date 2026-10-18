"""Drop / merge / add panel search with block-scoped reuse.

Rounds continue until the best refit does not improve the selection objective by more than the
tolerance; ``round_backstop(K)`` rounds are a reported backstop. Each round screens every drop
and add by its exact initial score and every merge by an EM lower bound from the
forward-backward statistics of the two corresponding drop panels (plus at most 2K exact
"double-removal" merges), then refits the top ``refits_per_kind`` proposals of each kind:
O(N L K^3) per round. Reuse is keyed by the complete panel AND its frequencies. Fits, scores and
merge bounds are founder-independent and kept across rounds in a ``LikelihoodMemo``; a whole
search is reused when all its inputs are identical (``search_input_key``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools

import numpy as np

from ..execution import claim_threads
from .fit import PathFit, canonical, fit_fixed, fit_posteriors, unique_rows
from .objective import ContextFounders, coded_regularizer, selection_objective
from .path_model import FLIPS, PathWorkspace, WorkspaceBuffers, log_likelihood


def proposals(fit: PathFit, bank):
    """(kind, panel, frequencies, origin) of every drop, merge and add start.

    origin: the deleted row j of a drop; (i, j, consensus) of the merge of rows i < j (row j
    deleted, row i set to the occupancy-weighted consensus); None for an add.
    """
    panel, f = fit.panel, fit.frequencies

    def key(p, q):
        return (p.tobytes(), np.ascontiguousarray(q).tobytes())

    seen = {key(panel, f)}
    if len(panel) > 1:
        for i in range(len(panel)):
            p, q = canonical(np.delete(panel, i, axis=0), np.delete(f, i))
            if key(p, q) not in seen:
                seen.add(key(p, q))
                yield "drop", p, q, i
        # The consensus is only an initialisation; the same model refits it.
        for i in range(len(panel)):
            for j in range(i + 1, len(panel)):
                p, q = np.delete(panel, j, axis=0), np.delete(f, j)
                weight = fit.occupancy[i] + fit.occupancy[j]
                mean = np.divide(fit.occupancy[i] * fit.allele_probability[i]
                                 + fit.occupancy[j] * fit.allele_probability[j],
                                 weight, out=np.full(panel.shape[1], .5), where=weight > 0)
                p[i] = np.where(mean == .5, panel[i], mean > .5)
                consensus = p[i].copy()
                q[i] += f[j]
                p, q = canonical(p, q)
                if key(p, q) not in seen:
                    seen.add(key(p, q))
                    yield "merge", p, q, (i, j, consensus)
    for row in bank:
        if np.any(np.all(panel == row, axis=1)):
            continue
        p, q = canonical(np.vstack((panel, row)),
                         np.r_[f * len(panel) / (len(panel) + 1), 1 / (len(panel) + 1)])
        if key(p, q) not in seen:
            seen.add(key(p, q))
            yield "add", p, q, None


def merge_lower_bound(drop_panel, drop_frequencies, drop_order, statistics, merged_panel,
                      merged_frequencies, kept, consensus) -> float:
    """EM lower bound on a merge's log likelihood from one drop panel's statistics.

    The merge of rows i and j equals the drop-j panel with row i replaced by the consensus (and
    the drop-i panel with row j replaced), with the pair's mass on the consensus. Under the drop
    posterior, Jensen's inequality gives LL(merge) >= LL(drop) + the kept row's complete-data
    flip gains at the consensus-changed sites + sum_a counts_a log(q'_a / q''_a) (counts:
    expected initial plus redraw-destination copies). A consensus equal to another row splits
    that row's mass proportionally to counts. O(L + K) per call.
    """
    log_likelihood_value, flip_gains, counts = statistics
    row = int(drop_order[kept])
    rows = drop_panel.copy()
    rows[row] = consensus
    changed = consensus != drop_panel[row]
    unique, inverse = unique_rows(rows, return_inverse=True)
    inverse = inverse.reshape(-1)
    group_counts = np.bincount(inverse, weights=counts, minlength=len(unique))
    group_sizes = np.bincount(inverse, minlength=len(unique))
    share = np.where(group_counts[inverse] > 0,
                     counts / np.where(group_counts[inverse] > 0, group_counts[inverse], 1.),
                     1. / group_sizes[inverse])
    frequency = merged_frequencies[inverse] * share
    used = counts > 0
    frequency_term = float(np.sum(counts[used] * (np.log(frequency[used])
                                                  - np.log(drop_frequencies[used]))))
    return log_likelihood_value + float(flip_gains[row, changed].sum()) + frequency_term


def merge_bounds(panel, drops, merges, prepared, buffers=None):
    """EM lower bounds of all merges (one forward-backward per drop panel, sharing one K - 1
    workspace); each merge takes the larger of its two bounds. Returns (bounds, passes)."""
    workspace = PathWorkspace(len(panel) - 1, prepared, buffers)
    statistics, orders = {}, {}

    def bound(dropped, kept, merged_panel, merged_frequency, consensus):
        drop_panel, drop_frequency = drops[dropped]
        if dropped not in statistics:
            claim_threads()
            state = workspace.state(drop_panel, drop_frequency, FLIPS)
            counts = (state["initial_copy_counts"] + state["redraw_destination_counts"])[:-1]
            statistics[dropped] = (state["log_likelihood"], state["flip_gains"], counts)
            order = unique_rows(np.delete(panel, dropped, axis=0), return_inverse=True)[1]
            orders[dropped] = np.asarray(order).reshape(-1)
        return merge_lower_bound(drop_panel, drop_frequency, orders[dropped], statistics[dropped],
                                 merged_panel, merged_frequency, kept, consensus)

    bounds = [max(bound(j, i, merged_panel, merged_frequency, consensus),
                  bound(i, j - 1, merged_panel, merged_frequency, consensus))
              for merged_panel, merged_frequency, (i, j, consensus) in merges]
    return bounds, len(statistics)


def double_removal_merges(panel, merges) -> set[int]:
    """Merges scored exactly besides their EM bounds: for every row k, the merge of k with its
    nearest row and the merge (i, j), k not in {i, j}, whose consensus is nearest row k (Hamming;
    ties: the earlier merge). The fixed-posterior bound credits no re-routing of carriers, so it
    can rank such merges far below their exact scores. At most 2K exact forwards per round."""
    if len(panel) < 2 or not merges:
        return set()
    panel = np.asarray(panel)
    consensus = np.asarray([origin[2] for _panel, _frequency, origin in merges], dtype=np.int64)
    pairs = np.asarray([origin[:2] for _panel, _frequency, origin in merges], dtype=np.int64)
    order = {(int(i), int(j)): m for m, (i, j) in enumerate(pairs)}
    rows = (panel[:, None, :] != panel[None, :, :]).sum(axis=2)
    np.fill_diagonal(rows, np.iinfo(np.int64).max)
    distance = (consensus[:, None, :] != panel[None, :, :]).sum(axis=2)
    chosen = set()
    for k in range(len(panel)):
        nearest = int(np.argmin(rows[k]))
        pair = (min(k, nearest), max(k, nearest))
        if pair in order:
            chosen.add(order[pair])
        valid = (pairs[:, 0] != k) & (pairs[:, 1] != k)
        if valid.any():
            chosen.add(int(np.argmin(np.where(valid, distance[:, k], np.iinfo(np.int64).max))))
    return chosen


def content_key(*arrays) -> bytes:
    """Exact content key of arrays (shape, dtype and bytes); a cache key only."""
    digest = hashlib.blake2b(digest_size=20)
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.digest()


class LikelihoodMemo:
    """Founder-independent likelihood results of one block, kept across rounds.

    A fit is a deterministic function of its canonical start, a score of its (panel,
    frequencies), and the merge bounds of a fit's (panel, frequencies), given the block's data
    and model (fixed for a contig run by the round identities). The kernels are thread
    invariant, so an entry equals its recomputation bit for bit. Fits are kept without their
    two K x L posterior arrays (one E-step restores them), with bit-packed panels.
    """

    def __init__(self, state=None):
        state = state or dict(fits={}, scores={}, bounds={})
        self.fits = dict(state["fits"])
        self.scores = dict(state["scores"])
        self.bounds = dict(state["bounds"])
        self.hits = dict(fits=0, scores=0, bounds=0, completed=0)

    def state(self) -> dict:
        """Picklable entries for a later round of the same block."""
        return dict(fits=self.fits, scores=self.scores, bounds=self.bounds)

    def fit(self, key) -> PathFit | None:
        stored = self.fits.get(key)
        if stored is None:
            return None
        self.hits["fits"] += 1
        length, packed = stored["panel"]
        panel = np.ascontiguousarray(np.unpackbits(packed, axis=1, count=length), dtype=np.int64)
        return PathFit(panel=panel, frequencies=stored["frequencies"], allele_probability=None,
                       occupancy=None, log_likelihood=stored["log_likelihood"],
                       objective=stored["objective"], converged=stored["converged"],
                       termination=stored["termination"])

    def add_fit(self, key, fit: PathFit) -> None:
        self.fits[key] = dict(panel=(fit.panel.shape[1], np.packbits(fit.panel.astype(np.uint8),
                                                                     axis=1)),
                              frequencies=fit.frequencies, log_likelihood=fit.log_likelihood,
                              objective=fit.objective, converged=fit.converged,
                              termination=fit.termination)


@dataclass(frozen=True)
class RoundTrace:
    """One search round: proposals per kind, refits, K before and of the best refit, its gain,
    kind and rank within its kind."""

    proposals: tuple[tuple[str, int], ...]
    refits: int
    old_k: int
    best_k: int
    gain: float
    kind: str
    winner_rank: int


@dataclass(frozen=True)
class SearchOutcome:
    """A block search's result: the fit, its selection objective, rounds, how it started
    ("start", "incumbent" or "reused"), the round backstop, the context founders and free rows,
    the reuse key of its inputs and work counts."""

    fit: PathFit
    selection_objective: float
    trace: tuple[RoundTrace, ...]
    search_start: str
    round_backstop: bool
    context_founders: int | None
    free_rows: int
    input_key: str
    work: tuple[tuple[str, int], ...]


def search_input_key(start, bank, prepared, search_config, founders) -> str:
    """Exact deterministic-search inputs: start, bank, data, model and search settings, context
    founders (a reuse key, not a likelihood test)."""
    digest = hashlib.sha256(b"local-selection-search-v1")
    for array in (start, bank, prepared.gl, prepared.observed, prepared.rates, founders):
        if array is None:
            digest.update(b"None")
            continue
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast("B"))
    digest.update(repr((prepared.unknown_mass, search_config)).encode())
    return digest.hexdigest()


def search_panel(start, bank, prepared, search_config, *, incumbent=None, founders=None,
                 memo: LikelihoodMemo | None = None) -> SearchOutcome:
    """Local search from ``start``, or from ``incumbent`` (a previous round's selected outcome of
    the same block) when it scores better; reuses the incumbent's whole search when the inputs
    are identical. ``founders``: the assembled context's founders in this block (context rounds);
    None codes every row as free."""
    tolerance = search_config.tolerance
    input_key = search_input_key(start, bank, prepared, search_config, founders)
    if incumbent is not None and incumbent.input_key == input_key:
        return replace(incumbent, search_start="reused", work=(("reused_previous_search", 1),))
    memo = LikelihoodMemo() if memo is None else memo
    # Fits keep their posterior arrays only while they are this round's trials; the cache keeps
    # them without (one E-step restores a cached fit that becomes the incumbent fit).
    score_cache, fit_cache = {}, {}
    buffers = WorkspaceBuffers()
    work = dict(score_calls=0, score_cache_hits=0, fit_calls=0, fit_cache_hits=0, bound_passes=0)
    context = None if founders is None else ContextFounders(founders)
    incumbent_fit = None if incumbent is None else incumbent.fit

    def fit_panel(panel, frequencies=None) -> PathFit:
        key = (panel.tobytes(), None if frequencies is None else frequencies.tobytes())
        if key in fit_cache:
            work["fit_cache_hits"] += 1
            return fit_cache[key]
        # fit_fixed depends on its input only through this canonical start.
        start_panel, start_frequencies = canonical(panel, frequencies)
        memo_key = content_key(start_panel, start_frequencies)
        result = memo.fit(memo_key)
        if result is None:
            result = fit_fixed(panel, frequencies, prepared, search_config, buffers)
            work["fit_calls"] += 1
            memo.add_fit(memo_key, result)
        fit_cache[key] = replace(result, allele_probability=None, occupancy=None)
        return result

    completed: dict = {}

    def complete(value: PathFit) -> PathFit:
        """A memoised fit with its posterior arrays (functions of its (panel, frequencies): the
        incumbent's when it is the same fit, an earlier completion's, else one E-step)."""
        if value.occupancy is not None:
            return value
        key = (value.panel.tobytes(), value.frequencies.tobytes())
        if key not in completed:
            if (incumbent_fit is not None and np.array_equal(value.panel, incumbent_fit.panel)
                    and np.array_equal(value.frequencies, incumbent_fit.frequencies)):
                completed[key] = (incumbent_fit.allele_probability, incumbent_fit.occupancy)
            else:
                memo.hits["completed"] += 1
                completed[key] = fit_posteriors(value.panel, value.frequencies, prepared, buffers)
        allele_probability, occupancy = completed[key]
        return replace(value, allele_probability=allele_probability, occupancy=occupancy)

    def score(panel, frequency) -> float:
        key = (panel.tobytes(), frequency.tobytes())
        if key in score_cache:
            work["score_cache_hits"] += 1
            return score_cache[key]
        memo_key = content_key(panel, frequency)
        value = memo.scores.get(memo_key)
        if value is None:
            claim_threads()
            value = log_likelihood(panel, frequency, prepared)
            work["score_calls"] += 1
            memo.scores[memo_key] = value
        else:
            memo.hits["scores"] += 1
        score_cache[key] = value
        return value

    def objective(value: PathFit) -> float:
        return selection_objective(value, context)

    search_start = "start"
    fit = fit_panel(start)
    if incumbent_fit is not None and objective(incumbent_fit) > objective(fit) + tolerance:
        # Climbing from the context start back to a better incumbent dominated the context
        # rounds' cost; explore the incumbent's neighbourhood instead.
        trial = fit_panel(incumbent_fit.panel, incumbent_fit.frequencies)
        fit = trial if objective(trial) >= objective(incumbent_fit) else incumbent_fit
        search_start = "incumbent"
    trace = []
    round_backstop = False
    for round_index in itertools.count():
        if round_index >= search_config.round_backstop(len(fit.panel)):
            round_backstop = True
            break
        ranked, seen, drops, merges = {}, set(), {}, []
        fit = complete(fit)
        for kind, panel, frequency, origin in proposals(fit, bank):
            tagged_key = (kind, panel.tobytes(), frequency.tobytes())
            if tagged_key in seen:
                continue
            seen.add(tagged_key)
            if kind == "merge":
                # Screened below by its EM lower bound, not an exact forward.
                ranked.setdefault(kind, [])
                merges.append((panel, frequency, origin))
                continue
            value = score(panel, frequency) + coded_regularizer(panel, frequency, context)
            ranked.setdefault(kind, []).append((value, panel, frequency))
            if kind == "drop":
                drops[origin] = (panel, frequency)
        if merges:
            # The drop and merge lists, hence the bounds, are functions of the fit's (panel,
            # frequencies) and the block data only.
            bounds_key = content_key(fit.panel, fit.frequencies)
            bounds = memo.bounds.get(bounds_key)
            if bounds is None:
                bounds, passes = merge_bounds(fit.panel, drops, merges, prepared, buffers)
                work["bound_passes"] += passes
                memo.bounds[bounds_key] = bounds
            else:
                memo.hits["bounds"] += 1
            exact = double_removal_merges(fit.panel, merges)
            for index, (bound, (panel, frequency, _origin)) in enumerate(zip(bounds, merges)):
                if index in exact:
                    bound = score(panel, frequency)
                ranked["merge"].append(
                    (bound + coded_regularizer(panel, frequency, context), panel, frequency))
        trials = []
        for kind, options in ranked.items():
            options.sort(key=lambda value: (-value[0], value[1].tobytes()))
            for rank, (_score, panel, frequency) in enumerate(options[:search_config.refits_per_kind]):
                trials.append((fit_panel(panel, frequency), kind, rank))
        if not trials:
            break
        best, kind, rank = min(trials, key=lambda value: (
            -objective(value[0]), len(value[0].panel), value[0].panel.tobytes()))
        trace.append(RoundTrace(
            proposals=tuple((name, len(values)) for name, values in ranked.items()),
            refits=len(trials), old_k=len(fit.panel), best_k=len(best.panel),
            gain=objective(best) - objective(fit), kind=kind, winner_rank=rank))
        if objective(best) <= objective(fit) + tolerance:
            break
        fit = best
    fit = complete(fit)
    return SearchOutcome(
        fit=fit, selection_objective=objective(fit), trace=tuple(trace),
        search_start=search_start, round_backstop=round_backstop,
        context_founders=None if context is None else len(context),
        free_rows=len(fit.panel) if context is None else context.free(fit.panel),
        input_key=input_key, work=tuple(sorted(work.items())))
