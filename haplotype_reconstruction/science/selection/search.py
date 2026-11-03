"""Search local founder panels while preserving proposal order and block memo state.

Owns LS08-LS14: drop/merge/add starts, two-sided merge bounds, whole-class refit
cuts, winner/continuation rules and exact completion precedence. Must not release
calls, schedule work, read storage or inspect process state.
Method: docs/methods/local_selection.md#search (WP_SE2 sections 4.1-4.7).
Proposals keep first equal-frequency occurrences; only complete tie classes enter
the refit budget. The same-panel kind representative is not convention-reversed.
Accretion (LS13): whole-search reuse excludes the incumbent from its key. Packed
memos are scientific state: fresh and restored fits can carry different posterior
last bits, so hit order and completion precedence must be retained exactly.
The context-coded screening's O(K^5) cost is an approved 0c exemption.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterator
from dataclasses import replace
from typing import TypedDict

import numpy as np

from ...contracts.selection import PackedFit, PackedPanel, RoundTrace, SearchOutcome, SelectionMemo
from ...core.digest import array_digest, json_digest
from ...core.ties import TieCounter, census, census_enabled, fixed_policy, improves, tie_classes
from .diagnostics import CensusSink
from .fit import PathFit, canonical, fit_fixed, fit_posteriors, unique_rows
from .objective import ContextFounders, coded_regularizer, selection_objective
from .parameters import (
    CONSENSUS_CENSUS_WINDOW,
    CONSENSUS_TIE,
    DECISION_MOTION,
    KIND_ORDER,
    PROPOSAL_FREQUENCY_TIE,
    PathSearchConfig,
)
from .path_model import FLIPS, PathWorkspace, PreparedBlock, WorkspaceBuffers, log_likelihood

__all__ = [
    "merge_consensus",
    "proposals",
    "merge_lower_bound",
    "merge_bounds",
    "double_removal_merges",
    "content_key",
    "LikelihoodMemo",
    "search_input_key",
    "tied_cut",
    "fitted_start",
    "search_panel",
]


class _Starts:
    """Distinct (panel, frequencies) starts: the same panel bytes with frequencies equal up to
    PROPOSAL_FREQUENCY_TIE is one start (the first one generated is kept)."""

    def __init__(self):
        self.by_panel: dict[bytes, list] = {}

    def add(self, panel, frequencies) -> bool:
        """Record a start; False when an equal one was recorded before."""
        frequencies = np.asarray(frequencies, dtype=np.float64)
        known = self.by_panel.setdefault(panel.tobytes(), [])
        for other in known:
            if other.shape == frequencies.shape and np.max(np.abs(other - frequencies)) <= PROPOSAL_FREQUENCY_TIE:
                return False
        known.append(frequencies)
        return True


def merge_consensus(
    panel_i: np.ndarray, mean: np.ndarray, panel_j: np.ndarray | None = None, *, reverse: bool
) -> np.ndarray:
    """A merge's consensus row: ALT where the occupancy-weighted ALT probability exceeds 1/2, row
    i's allele within CONSENSUS_TIE of 1/2 (an exact 1/2, e.g. at a site without information, must
    not be decided by its last bits: a band at a mathematical tie, design A-7). Row i is the
    lexicographically smaller row: a content-order convention (row j's allele under
    maths.ties.reverse_conventions)."""
    kept = (
        panel_j
        if (panel_j is not None and fixed_policy("selection.merge.consensus", CONSENSUS_TIE, reverse=reverse).reverse)
        else panel_i
    )
    return np.where(np.abs(mean - 0.5) <= CONSENSUS_TIE, kept, mean > 0.5).astype(np.int64)


def proposals(
    fit: PathFit,
    bank: np.ndarray,
    ties: TieCounter | None = None,
    census_key: str | None = None,
    *,
    reverse: bool,
    census_sink: CensusSink | None = None,
) -> Iterator[tuple[str, np.ndarray, np.ndarray, int | tuple[int, int, np.ndarray] | None]]:
    """(kind, panel, frequencies, origin) of every drop, merge and add start.

    origin: the deleted row j of a drop; (i, j, consensus) of the merge of rows i < j (row j
    deleted, row i set to the occupancy-weighted consensus); None for an add. Starts equal to an
    earlier one (same panel, frequencies equal up to rounding) are skipped. ``ties`` counts the
    consensus band decisions (kind "ls.consensus.band": one per merge, inside the tie when some
    site's probability lies within the band); ``census_key`` records the sites near 1/2.
    """
    panel, f = fit.panel, fit.frequencies
    seen = _Starts()
    seen.add(panel, f)
    collect = census_key is not None and census_enabled(sink=census_sink)
    if len(panel) > 1:
        assert fit.occupancy is not None and fit.allele_probability is not None
        for i in range(len(panel)):
            p, q = canonical(np.delete(panel, i, axis=0), np.delete(f, i))
            if seen.add(p, q):
                yield "drop", p, q, i
        # The consensus is only an initialisation; the same model refits it.
        for i in range(len(panel)):
            for j in range(i + 1, len(panel)):
                p, q = np.delete(panel, j, axis=0), np.delete(f, j)
                weight = fit.occupancy[i] + fit.occupancy[j]
                mean = np.divide(
                    fit.occupancy[i] * fit.allele_probability[i] + fit.occupancy[j] * fit.allele_probability[j],
                    weight,
                    out=np.full(panel.shape[1], 0.5),
                    where=weight > 0,
                )
                p[i] = merge_consensus(panel[i], mean, panel[j], reverse=reverse)
                distance = mean - 0.5
                if ties is not None:
                    ties.note(
                        "ls.consensus.band", inside_tie=bool(np.any((np.abs(distance) <= CONSENSUS_TIE) & (weight > 0)))
                    )
                if collect:
                    for site in np.flatnonzero((np.abs(distance) <= CONSENSUS_CENSUS_WINDOW) & (weight > 0)):
                        census(
                            "ls.consensus.band",
                            f"{census_key}/m{i}_{j}/s{site}",
                            (0.0, float(distance[site])),
                            sink=census_sink,
                        )
                consensus = p[i].copy()
                q[i] += f[j]
                p, q = canonical(p, q)
                if seen.add(p, q):
                    yield "merge", p, q, (i, j, consensus)
    for row in bank:
        if np.any(np.all(panel == row, axis=1)):
            continue
        p, q = canonical(np.vstack((panel, row)), np.r_[f * len(panel) / (len(panel) + 1), 1 / (len(panel) + 1)])
        if seen.add(p, q):
            yield "add", p, q, None


def merge_lower_bound(
    drop_panel, drop_frequencies, drop_order, statistics, merged_panel, merged_frequencies, kept, consensus
) -> float:
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
    share = np.where(
        group_counts[inverse] > 0,
        counts / np.where(group_counts[inverse] > 0, group_counts[inverse], 1.0),
        1.0 / group_sizes[inverse],
    )
    frequency = merged_frequencies[inverse] * share
    used = counts > 0
    frequency_term = float(np.sum(counts[used] * (np.log(frequency[used]) - np.log(drop_frequencies[used]))))
    return log_likelihood_value + float(flip_gains[row, changed].sum()) + frequency_term


def merge_bounds(panel, drops, merges, prepared, buffers=None):
    """EM lower bounds of all merges (one forward-backward per drop panel, sharing one K - 1
    workspace); each merge takes the larger of its two bounds. Returns (bounds, passes)."""
    workspace = PathWorkspace(len(panel) - 1, prepared, buffers)
    statistics, orders = {}, {}

    def bound(dropped, kept, merged_panel, merged_frequency, consensus):
        drop_panel, drop_frequency = drops[dropped]
        if dropped not in statistics:
            state = workspace.state(drop_panel, drop_frequency, FLIPS)
            counts = (state["initial_copy_counts"] + state["redraw_destination_counts"])[:-1]
            statistics[dropped] = (state["log_likelihood"], state["flip_gains"], counts)
            order = unique_rows(np.delete(panel, dropped, axis=0), return_inverse=True)[1]
            orders[dropped] = np.asarray(order).reshape(-1)
        return merge_lower_bound(
            drop_panel,
            drop_frequency,
            orders[dropped],
            statistics[dropped],
            merged_panel,
            merged_frequency,
            kept,
            consensus,
        )

    bounds = [
        max(
            bound(j, i, merged_panel, merged_frequency, consensus),
            bound(i, j - 1, merged_panel, merged_frequency, consensus),
        )
        for merged_panel, merged_frequency, (i, j, consensus) in merges
    ]
    return bounds, len(statistics)


def double_removal_merges(
    panel, merges, bounds, tolerance, ties: TieCounter | None = None, *, reverse: bool
) -> set[int]:
    """Merges scored exactly besides their EM bounds: for every row k, the best-bounded merge of k with
    one of its nearest rows, and the best-bounded merge (i, j), k not in {i, j}, whose consensus is
    nearest row k. The fixed-posterior bound credits no re-routing of carriers, so it can rank such
    merges far below their exact scores. Release 2.17: within each nearest-distance class only its
    representative is scored exactly (the best of the class's EM bounds, tie classes of
    ``tolerance`` broken by content; a choice inside a tie class is counted as
    ``ls.rank.merge_representative``), so at most 2K merges are scored exactly per screen and the
    likelihood screening stays O(N L K^3); before, a distance tie admitted the whole class (up to
    K(K - 1)/2 merges; a K30 census of screens with incomplete context inputs found 76-277 of about 400).
    The other members keep their bounds. Tied bounds do not imply tied exact scores: the choice is
    deterministic for a fixed tie convention, not invariant to it."""
    if len(panel) < 2 or not merges:
        return set()
    panel = np.asarray(panel)
    consensus = np.asarray([origin[2] for _panel, _frequency, origin in merges], dtype=np.int64)
    pairs = np.asarray([origin[:2] for _panel, _frequency, origin in merges], dtype=np.int64)
    order = {(int(i), int(j)): m for m, (i, j) in enumerate(pairs)}
    rows = (panel[:, None, :] != panel[None, :, :]).sum(axis=2)
    np.fill_diagonal(rows, np.iinfo(np.int64).max)
    distance = (consensus[:, None, :] != panel[None, :, :]).sum(axis=2)

    def representative(members):
        members = sorted(set(members))
        if len(members) == 1:
            return members[0]
        classes = tie_classes(
            [bounds[m] for m in members],
            tolerance,
            content=[(merges[m][0].tobytes(), merges[m][1].tobytes()) for m in members],
            descending=True,
            motion=DECISION_MOTION,
            reverse=fixed_policy("selection.search.merge_representative", tolerance, reverse=reverse).reverse,
        )
        if ties is not None:
            ties.add("ls.rank.merge_representative", classes)
        return members[int(classes.order[0])]

    chosen = set()
    for k in range(len(panel)):
        nearest = [
            order[(min(k, int(n)), max(k, int(n)))]
            for n in np.flatnonzero(rows[k] == rows[k].min())
            if (min(k, int(n)), max(k, int(n))) in order
        ]
        if nearest:
            chosen.add(representative(nearest))
        valid = (pairs[:, 0] != k) & (pairs[:, 1] != k)
        if valid.any():
            minimum = distance[valid, k].min()
            chosen.add(representative(np.flatnonzero(valid & (distance[:, k] == minimum)).tolist()))
    return chosen


def content_key(*arrays: np.ndarray) -> bytes:
    """Exact content key of arrays (shape, dtype and bytes); a cache key only."""
    digest = hashlib.blake2b(digest_size=20)
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.digest()


class _PackedState(TypedDict):
    """Internal memo representation; arrays are restored on demand."""

    panel: tuple[int, np.ndarray]
    frequencies: np.ndarray
    log_likelihood: float
    objective: float
    converged: bool
    termination: str
    selected: str
    selected_update: int
    endpoint_selection: float | None
    selected_selection: float | None
    context_snaps: int
    snap_ties: int


class LikelihoodMemo:
    """Likelihood results of one block, kept across rounds (context-aware fits keyed by their context).

    A fit is a deterministic function of its canonical start, a score of its (panel,
    frequencies), and the merge bounds of a fit's (panel, frequencies), given the block's data
    and model (fixed for a contig run by the round identities). Accretion (LS13):
    fit hits are stripped, while the first fresh return retains trajectory arrays.
    Canonical-row completion may differ in their last bits; memo hit patterns and
    completion precedence are scientific state. Panels are stored bit-packed.
    """

    def __init__(self, state: SelectionMemo | None = None):
        state = SelectionMemo() if state is None else state
        self.fits: dict[bytes, _PackedState] = {
            bytes.fromhex(key): dict(
                panel=(value.panel.sites, value.panel.rows),
                frequencies=value.frequencies,
                log_likelihood=value.log_likelihood,
                objective=value.objective,
                converged=value.converged,
                termination=value.termination,
                selected=value.selected,
                selected_update=value.selected_update,
                endpoint_selection=value.endpoint_selection,
                selected_selection=value.selected_selection,
                context_snaps=value.context_snaps,
                snap_ties=value.snap_ties,
            )
            for key, value in state.fits.items()
        }
        self.scores = {bytes.fromhex(key): value for key, value in state.scores.items()}
        self.bounds = {bytes.fromhex(key): list(value) for key, value in state.bounds.items()}
        self.hits = dict(fits=0, scores=0, bounds=0, completed=0)

    def state(self) -> SelectionMemo:
        """Freeze exact memo state for the next round; hex keys are a reversible encoding."""
        fits = {
            key.hex(): PackedFit(
                panel=PackedPanel(*value["panel"]),
                frequencies=value["frequencies"],
                log_likelihood=value["log_likelihood"],
                objective=value["objective"],
                converged=value["converged"],
                termination=value["termination"],
                selected=value["selected"],
                selected_update=value["selected_update"],
                endpoint_selection=value["endpoint_selection"],
                selected_selection=value["selected_selection"],
                context_snaps=value["context_snaps"],
                snap_ties=value["snap_ties"],
            )
            for key, value in self.fits.items()
        }
        return SelectionMemo(
            fits=fits,
            scores={key.hex(): value for key, value in self.scores.items()},
            bounds={key.hex(): tuple(value) for key, value in self.bounds.items()},
        )

    def fit(self, key) -> PathFit | None:
        stored = self.fits.get(key)
        if stored is None:
            return None
        self.hits["fits"] += 1
        length, packed = stored["panel"]
        panel = np.ascontiguousarray(np.unpackbits(packed, axis=1, count=length), dtype=np.int64)
        return PathFit(
            panel=panel,
            frequencies=stored["frequencies"],
            allele_probability=None,
            occupancy=None,
            log_likelihood=stored["log_likelihood"],
            objective=stored["objective"],
            converged=stored["converged"],
            termination=stored["termination"],
            selected=stored["selected"],
            selected_update=stored["selected_update"],
            endpoint_selection=stored["endpoint_selection"],
            selected_selection=stored["selected_selection"],
            context_snaps=stored["context_snaps"],
            snap_ties=stored["snap_ties"],
        )

    def add_fit(self, key, fit: PathFit) -> None:
        self.fits[key] = dict(
            panel=(fit.panel.shape[1], np.packbits(fit.panel.astype(np.uint8), axis=1)),
            frequencies=fit.frequencies,
            log_likelihood=fit.log_likelihood,
            objective=fit.objective,
            converged=fit.converged,
            termination=fit.termination,
            selected=fit.selected,
            selected_update=fit.selected_update,
            endpoint_selection=fit.endpoint_selection,
            selected_selection=fit.selected_selection,
            context_snaps=fit.context_snaps,
            snap_ties=fit.snap_ties,
        )


def search_input_key(
    start: np.ndarray,
    bank: np.ndarray,
    prepared: PreparedBlock,
    search_config: PathSearchConfig,
    founders: np.ndarray | None,
) -> str:
    """Exact deterministic-search inputs: start, bank, data, model and search settings, context
    founders (a reuse key, not a likelihood test)."""
    arrays = (start, bank, prepared.gl, prepared.observed, prepared.rates, founders)
    return json_digest(
        {
            "version": "selection-search-v1",
            "arrays": tuple(None if value is None else array_digest(value) for value in arrays),
            "unknown_mass": prepared.unknown_mass,
            "search": search_config,
        }
    )


def tied_cut(classes, limit: int, cap: int) -> tuple[int, bool]:
    """(how many proposals of tie-class order ``classes.order`` (maths.ties.TieClasses) are
    refitted, overflow). Release 2.2: whole tie classes only, in class order, until at least
    ``limit``; a class that would take the total past ``cap`` is left out whole (overflow: its
    tied members are not refitted, and the caller counts and flags it). Order-invariant: no
    member of a class is preferred to another."""
    total = len(classes.order)
    if total <= limit:
        return total, False
    order = classes.order
    take = 0
    while take < total and take < limit:
        cut_class = classes.classes[int(order[take])]
        end = take
        while end < total and classes.classes[int(order[end])] == cut_class:
            end += 1
        if end > cap:
            return take, True
        take = end
    return take, False


def _selection(context):
    """The context-coded selection objective of a fit state (None without a context: the fit's own objective)."""
    if context is None:
        return None
    return lambda panel, frequencies, log_likelihood: log_likelihood + coded_regularizer(panel, frequencies, context)


def _context_key(context) -> bytes:
    """Memo-key suffix: a context-aware fit depends on the context (release 2.12)."""
    return b"" if context is None else content_key(context.founders, context.called)


def fitted_start(
    start: np.ndarray,
    prepared: PreparedBlock,
    search_config: PathSearchConfig,
    memo: LikelihoodMemo,
    context: ContextFounders | None = None,
    *,
    reverse: bool,
    census_sink: CensusSink | None = None,
) -> PathFit:
    """The fixed-K fit of a start (memoised under the key search_panel uses, so the search from
    this start reuses it)."""
    start_panel, start_frequencies = canonical(start, None)
    memo_key = content_key(start_panel, start_frequencies) + _context_key(context)
    result = memo.fit(memo_key)
    if result is None:
        result = fit_fixed(
            start,
            None,
            prepared,
            search_config,
            WorkspaceBuffers(),
            selection=_selection(context),
            context=context,
            reverse=reverse,
            census_sink=census_sink,
        )
        memo.add_fit(memo_key, result)
    return result


def search_panel(
    start: np.ndarray,
    bank: np.ndarray,
    prepared: PreparedBlock,
    search_config: PathSearchConfig,
    *,
    incumbent: SearchOutcome | None = None,
    founders: np.ndarray | None = None,
    memo: LikelihoodMemo | None = None,
    ties: TieCounter | None = None,
    census_key: str | None = None,
    reverse: bool,
    census_sink: CensusSink | None = None,
) -> SearchOutcome:
    """Local search from ``start``, or from ``incumbent`` (a previous round's selected outcome of
    the same block) when the start's fit reaches the incumbent's panel or the incumbent scores
    better by more than the decision tolerance; reuses the incumbent's whole search when the
    inputs are identical. ``founders``: the assembled context's founders in this block (context
    rounds); None codes every row as free. ``ties``: counts the decisions taken inside a tie class
    (and guard events); ``census_key``: the block's key for the valley census (maths.ties)."""
    tolerance = search_config.decision_tolerance
    ties = TieCounter() if ties is None else ties
    collect = census_key is not None and census_enabled(sink=census_sink)
    input_key = search_input_key(start, bank, prepared, search_config, founders)
    if incumbent is not None and incumbent.input_key == input_key:
        return replace(incumbent, search_start="reused", work=(("reused_previous_search", 1),))
    memo = LikelihoodMemo() if memo is None else memo
    # Fits keep their posterior arrays only while they are this round's trials; the cache keeps
    # them without (one E-step restores a cached fit that becomes the incumbent fit).
    score_cache: dict[tuple[bytes, bytes], float] = {}
    fit_cache: dict[tuple[bytes, bytes | None], PathFit] = {}
    buffers = WorkspaceBuffers()
    work = dict(score_calls=0, score_cache_hits=0, fit_calls=0, fit_cache_hits=0, bound_passes=0, exact_merges=0)
    context = None if founders is None else ContextFounders(founders)
    incumbent_fit = None if incumbent is None else incumbent.fit

    def fit_panel(panel, frequencies=None) -> PathFit:
        key = (panel.tobytes(), None if frequencies is None else frequencies.tobytes())
        if key in fit_cache:
            work["fit_cache_hits"] += 1
            return fit_cache[key]
        # fit_fixed depends on its input only through this canonical start.
        start_panel, start_frequencies = canonical(panel, frequencies)
        memo_key = content_key(start_panel, start_frequencies) + _context_key(context)
        result = memo.fit(memo_key)
        if result is None:
            fit_key = None
            if collect:
                # A diagnostic key stable across reruns: the start panel and its frequencies to 1e-6.
                fit_key = f"{census_key}/fit{content_key(start_panel, np.round(start_frequencies, 6)).hex()[:16]}"
            result = fit_fixed(
                panel,
                frequencies,
                prepared,
                search_config,
                buffers,
                census_key=fit_key,
                selection=_selection(context),
                context=context,
                reverse=reverse,
                census_sink=census_sink,
            )
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
            if (
                incumbent_fit is not None
                and np.array_equal(value.panel, incumbent_fit.panel)
                and np.array_equal(value.frequencies, incumbent_fit.frequencies)
            ):
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
    start_was_incumbent_panel = False
    if incumbent_fit is not None:
        if collect:
            census("ls.start", f"{census_key}/start", (-objective(incumbent_fit), -objective(fit)), sink=census_sink)
        if np.array_equal(fit.panel, incumbent_fit.panel):
            # The start's fit reached the incumbent's founders: the two fits differ only by EM
            # residuals, so continue from the incumbent fit (deterministic), unless the start's
            # frequencies are better by more than the decision tolerance.
            start_was_incumbent_panel = True
            if not improves(objective(fit), objective(incumbent_fit), tolerance):
                fit = incumbent_fit
                search_start = "incumbent"
        elif improves(objective(incumbent_fit), objective(fit), tolerance):
            # Climbing from the context start back to a better incumbent dominated the context
            # rounds' cost; explore the incumbent's neighbourhood instead (its refit only when that
            # improves it by more than the decision tolerance, e.g. an unconverged incumbent).
            trial = fit_panel(incumbent_fit.panel, incumbent_fit.frequencies)
            fit = trial if improves(objective(trial), objective(incumbent_fit), tolerance) else incumbent_fit
            search_start = "incumbent"
    trace = []
    round_backstop = False
    for round_index in itertools.count():
        if round_index >= search_config.round_backstop(len(fit.panel)):
            round_backstop = True
            break
        ranked: dict[str, list] = {}
        seen: set = set()
        drops: dict = {}
        merges: list = []
        fit = complete(fit)
        for kind, panel, frequency, origin in proposals(
            fit,
            bank,
            ties,
            None if not collect else f"{census_key}/r{round_index}",
            reverse=reverse,
            census_sink=census_sink,
        ):
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
            exact = double_removal_merges(fit.panel, merges, bounds, tolerance, ties, reverse=reverse)
            work["exact_merges"] += len(exact)
            for index, (bound, (panel, frequency, _origin)) in enumerate(zip(bounds, merges, strict=True)):
                if index in exact:
                    bound = score(panel, frequency)
                ranked["merge"].append((bound + coded_regularizer(panel, frequency, context), panel, frequency))
        trials = []
        limit, cap = search_config.refits_per_kind, search_config.refits_per_kind_cap
        for kind, options in ranked.items():
            # Tie classes of the screened scores; whole classes are refitted until at least
            # ``limit`` (a class crossing ``cap`` is left out whole and counted as an overflow).
            classes = tie_classes(
                [option[0] for option in options],
                tolerance,
                content=[(option[1].tobytes(), n) for n, option in enumerate(options)],
                descending=True,
                motion=DECISION_MOTION,
                reverse=fixed_policy("selection.search.rank", tolerance, reverse=reverse).reverse,
            )
            ordered = classes.order
            take, overflow = tied_cut(classes, limit, cap)
            if len(options) > limit:
                ties.note(f"ls.rank.{kind}.join", inside_tie=take > limit)
                ties.note(f"ls.rank.{kind}.overflow", inside_tie=overflow)
                inside = bool(
                    0 < take < len(options)
                    and classes.classes[int(ordered[take - 1])] == classes.classes[int(ordered[take])]
                )
                ties.note(
                    f"ls.rank.{kind}", inside_tie=inside, wide=classes.wide_classes, boundary=len(classes.boundary_gaps)
                )
                if collect:
                    census(
                        f"ls.rank.{kind}",
                        f"{census_key}/r{round_index}/{kind}",
                        [-options[int(n)][0] for n in ordered[max(0, limit - 4) : limit + 4]],
                        sink=census_sink,
                    )
            for rank, n in enumerate(ordered[:take]):
                _score, panel, frequency = options[int(n)]
                trials.append((fit_panel(panel, frequency), kind, rank, panel.tobytes()))
        if not trials:
            break
        objectives = [objective(trial[0]) for trial in trials]
        if collect:
            # One value per distinct panel (its best refit) for the winner's valley; the spread of the
            # refits that reached one panel (EM residuals) separately.
            per_panel: dict = {}
            for trial_item, value in zip(trials, objectives, strict=True):
                per_panel.setdefault(trial_item[0].panel.tobytes(), []).append(value)
            census(
                "ls.trials",
                f"{census_key}/r{round_index}",
                sorted(-max(values) for values in per_panel.values())[:8],
                sink=census_sink,
            )
            for number, values in enumerate(sorted(per_panel.values(), key=max, reverse=True)[:4]):
                if len(values) > 1:
                    census(
                        "ls.trials.same_panel",
                        f"{census_key}/r{round_index}/p{number}",
                        sorted(values),
                        sink=census_sink,
                    )
        classes = tie_classes(
            objectives,
            tolerance,
            secondary=[(len(trial[0].panel),) for trial in trials],
            content=[(trial[0].panel.tobytes(), n) for n, trial in enumerate(trials)],
            descending=True,
            motion=DECISION_MOTION,
            reverse=fixed_policy("selection.search.winner", tolerance, reverse=reverse).reverse,
        )
        winner = int(classes.order[0])
        # Refits that reached the winning panel from different proposals are one answer (their fits
        # differ by EM residuals only): the refit of the proposal with the smallest (kind, proposal
        # panel) label represents it (release 2.2: a label of the same answer, stable under noise
        # and not a convention reversed by maths.ties; 2.1 took the trial order inside the class).
        panel_key = trials[winner][0].panel.tobytes()
        same = [
            n
            for n in range(len(trials))
            if classes.classes[n] == classes.classes[winner] and trials[n][0].panel.tobytes() == panel_key
        ]
        if len(same) > 1:
            winner = min(same, key=lambda n: (KIND_ORDER[trials[n][1]], trials[n][3], trials[n][2]))
        # Inside a tie only when the best class holds different founder sets (refits reaching the
        # same canonical panel are one candidate).
        ties.note(
            "ls.winner",
            inside_tie=len({trials[int(n)][0].panel.tobytes() for n in classes.members(0)}) > 1,
            wide=classes.wide_classes,
            boundary=len(classes.boundary_gaps),
        )
        best, kind, rank, _proposal = trials[winner]
        trace.append(
            RoundTrace(
                proposals=tuple((name, len(values)) for name, values in ranked.items()),
                refits=len(trials),
                old_k=len(fit.panel),
                best_k=len(best.panel),
                gain=objective(best) - objective(fit),
                kind=kind,
                winner_rank=rank,
            )
        )
        if collect:
            census(
                "ls.stop", f"{census_key}/r{round_index}/stop", (-objective(best), -objective(fit)), sink=census_sink
            )
        if not improves(objective(best), objective(fit), tolerance):
            break
        fit = best
    fit = complete(fit)
    return SearchOutcome(
        fit=fit,
        selection_objective=objective(fit),
        trace=tuple(trace),
        search_start=search_start,
        round_backstop=round_backstop,
        context_founders=None if context is None else len(context),
        free_rows=len(fit.panel) if context is None else context.free(fit.panel),
        input_key=input_key,
        work=tuple(sorted(work.items())),
        start_was_incumbent_panel=start_was_incumbent_panel,
    )
