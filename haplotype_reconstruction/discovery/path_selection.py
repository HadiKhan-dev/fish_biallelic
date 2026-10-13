"""Drop/merge/add panel search with block-scoped score reuse.

Rounds continue until the best refit does not improve; 2K + 10 rounds (K of
the current panel) are a reported backstop. Each round screens every drop and
add by its exact initialisation score and every merge by an EM lower bound
from the forward-backward statistics of the two corresponding drop panels, then
refits the top eight per proposal kind: O(N L K^3) per round. Reuse is keyed
by the complete binary panel AND its frequency vector: equal alleles with
different starts must still compete. Prepared observations and memoized fits
live within one block search. A completed deterministic search can also be
reused across feedback rounds when all its inputs match exactly, and its
founder-independent fits, scores and merge bounds always are (LikelihoodMemo).
"""
import hashlib
import math
import itertools
import numpy as np
from haplotype_reconstruction.core import parallel
from . import path_scoring as scoring, path_fitting as fitting
from .path_fitting import canonical, regularizer
from .objectives import log_binary_haplotype_set_count
from .candidate_selection import CandidateSelectionConfig, select_candidate_panel


def prepare_candidate_bank(gl, observed, original_latent, proposal_panels,
                           config=None, incumbent=None):
    """Return the tested local start, binary candidate bank and diagnostics.

    Assembly calls are candidate initializations, never extra observations.
    Partial rows use the existing nearest-latent completion; known alleles are
    preserved. The latest nonempty cavity-ranked source endpoint initializes
    search; the BIC-selected endpoint augments the bank. Neither score replaces
    the normalized path objective. An incumbent contributes all its latent rows.
    """
    proposals = list(proposal_panels)
    if proposals:
        if config is None:
            config = CandidateSelectionConfig(criterion="bic")
        competing = select_candidate_panel(
            gl, observed_mask=observed, original_latent=original_latent,
            proposal_panels=proposals, config=config, defer_cavity=True)
        backbone = next(value for value in reversed(competing.cavity_source_results)
                        if value is not None)
        start = np.where(backbone["discrete_haps"] >= 0,
                         backbone["discrete_haps"], backbone["latent_haps"])
        bank = np.unique(np.vstack((original_latent, start,
                                    competing.selected_mode.haplotypes)), axis=0)
        diagnostic = competing.diagnostic
    else:
        start = np.asarray(original_latent)
        bank = np.unique(start, axis=0)
        diagnostic = dict(source="raw latent initialization only")
    if incumbent is not None:
        bank = np.unique(np.vstack((bank, incumbent["panel"])), axis=0)
    return start, bank, diagnostic


def wildcard_compatibility(pair, observed):
    """Observed-site MAP unknown-count summary, not a constant founder pair.

    With no observed site a sample has two unknown slots, but contributes no
    mass denominator. Posterior expected unknown mass is reported separately.
    """
    count_probability = np.stack((pair[:, :, :-1, :-1].sum(axis=(2, 3)),
        pair[:, :, -1, :-1].sum(axis=2) + pair[:, :, :-1, -1].sum(axis=2),
        pair[:, :, -1, -1]), axis=2)
    counts = count_probability.argmax(axis=2)
    has_depth = observed.any(axis=1)
    slots = np.where(observed, counts, 0).max(axis=1)
    slots = np.where(has_depth, slots, 2).astype(np.int8)
    mass = float(slots[has_depth].sum() / max(1, 2*has_depth.sum()))
    return dict(slots=slots, has_depth=has_depth, mass=mass)


def proposals(fit, bank):
    """Yield (kind, panel, frequencies, origin) for drop, merge and add starts.

    ``origin`` is the deleted row j of ``fit['panel']`` for a drop, the merged
    pair (i, j) with the consensus row for a merge (row j deleted, row i set
    to the occupancy-weighted consensus) and None for an add.
    """
    panel, f = fit['panel'], fit['frequencies']
    def key(p, q):
        return (p.tobytes(), np.ascontiguousarray(q).tobytes())
    seen = {key(panel, f)}
    if len(panel) > 1:
        for i in range(len(panel)):
            p, q = canonical(np.delete(panel, i, axis=0), np.delete(f, i))
            if key(p, q) not in seen:
                seen.add(key(p, q))
                yield 'drop', p, q, i
        # Consensus is only an initialization; the same read model refits it.
        # Both oriented deletion starts above compete with these merge starts.
        for i in range(len(panel)):
            for j in range(i + 1, len(panel)):
                p, q = np.delete(panel, j, axis=0), np.delete(f, j)
                weight = fit['occupancy'][i] + fit['occupancy'][j]
                mean = np.divide(fit['occupancy'][i]*fit['allele_probability'][i] +
                                 fit['occupancy'][j]*fit['allele_probability'][j],
                                 weight, out=np.full(panel.shape[1], .5), where=weight>0)
                p[i] = np.where(mean == .5, panel[i], mean > .5)
                consensus = p[i].copy()
                q[i] += f[j]
                p, q = canonical(p, q)
                if key(p, q) not in seen:
                    seen.add(key(p, q))
                    yield 'merge', p, q, (i, j, consensus)
    for row in bank:
        if np.any(np.all(panel == row, axis=1)):
            continue
        p, q = canonical(np.vstack((panel, row)), np.r_[f * len(panel)/(len(panel)+1), 1/(len(panel)+1)])
        if key(p, q) not in seen:
            seen.add(key(p, q))
            yield 'add', p, q, None


def merge_lower_bound(drop_panel, drop_frequencies, drop_order, statistics,
                      merged_panel, merged_frequencies, kept, consensus):
    """EM lower bound on a merge's log-likelihood from one drop panel's statistics.

    The merge of rows i and j equals the drop-j panel with row i replaced by
    the consensus, and also the drop-i panel with row j replaced by it, with
    frequencies q' (the pair's mass on the consensus) instead of the drop's
    q''. ``kept`` is the replaced row's index in the drop panel before
    canonical sorting (``drop_order`` maps it). Under the drop posterior
    (redraw-augmented complete data), Jensen's inequality gives
    LL(merge) >= LL(drop) + sum over consensus-changed sites t of the kept
    row's complete-data flip gain + sum_a counts_a log(q'_a / q''_a), counts
    being the expected initial plus redraw-destination copies (the unknown
    state's prior mass is unchanged). A consensus equal to another row is the
    same likelihood with that row's mass split between the copies; the split
    proportional to counts maximises the bound. O(L + K) per call.
    """
    log_likelihood, flip_gains, counts = statistics
    row = int(drop_order[kept])
    rows = drop_panel.copy()
    rows[row] = consensus
    changed = consensus != drop_panel[row]
    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    if not np.array_equal(unique, merged_panel):
        raise AssertionError('merge start does not match its drop panel')
    group_counts = np.bincount(inverse, weights=counts, minlength=len(unique))
    group_sizes = np.bincount(inverse, minlength=len(unique))
    share = np.where(group_counts[inverse] > 0,
                     counts / np.where(group_counts[inverse] > 0, group_counts[inverse], 1.),
                     1. / group_sizes[inverse])
    frequency = merged_frequencies[inverse] * share
    used = counts > 0
    frequency_term = float(np.sum(counts[used] * (np.log(frequency[used])
                                                  - np.log(drop_frequencies[used]))))
    return log_likelihood + float(flip_gains[row, changed].sum()) + frequency_term


def merge_bounds(panel, drops, merges, gl, observed, positions, prepared, model_options):
    """EM lower bounds of all merges; one forward-backward per drop panel.

    ``drops`` maps the deleted row to the scored drop panel and frequencies,
    ``merges`` lists (panel, frequencies, (i, j, consensus)), i < j. Each
    merge takes the larger of its two valid bounds, from the drop-j panel
    (row i replaced) and from the drop-i panel (row j, index j - 1 after
    deleting i, replaced): the bound is tight in the orientation whose kept
    row is close to the consensus. The drop panels have K - 1 rows and share
    one fitting workspace: O(N L K^3) in total. Returns the bounds (without
    the regularizer) and the number of forward-backward passes.
    """
    workspace = fitting._Workspace(panel[1:], gl, positions, observed, prepared=prepared,
                                   **model_options)
    statistics, orders = {}, {}

    def bound(dropped, kept, merged_panel, merged_frequency, consensus):
        drop_panel, drop_frequency = drops[dropped]
        if dropped not in statistics:
            parallel.apply_dynamic_threads()
            state = workspace.state(drop_panel, drop_frequency, full_statistics=True)
            counts = (state['initial_copy_counts'] + state['redraw_destination_counts'])[:-1]
            statistics[dropped] = (state['log_likelihood'], state['flip_gains'], counts)
            order = np.unique(np.delete(panel, dropped, axis=0), axis=0, return_inverse=True)[1]
            orders[dropped] = np.asarray(order).reshape(-1)
        return merge_lower_bound(drop_panel, drop_frequency, orders[dropped], statistics[dropped],
                                 merged_panel, merged_frequency, kept, consensus)

    bounds = [max(bound(j, i, merged_panel, merged_frequency, consensus),
                  bound(i, j - 1, merged_panel, merged_frequency, consensus))
              for merged_panel, merged_frequency, (i, j, consensus) in merges]
    return bounds, len(statistics)


def double_removal_merges(panel, merges):
    """Merges scored exactly in addition to their EM bounds.

    The fixed-posterior EM bound credits no re-routing of carriers, so it can
    rank far below its exact score a merge that changes how a group of similar
    rows explains its carriers (T19 replays: real chr10 L1 block 412, merge of
    rows 7 sites apart inside a 3-row cluster, bound 3,700 nats below exact;
    nf_s3 chr8 block 107, a merge whose consensus nearly reproduces a third row,
    1,627 nats below). For every row k two merges are selected: k with its
    nearest row, and the merge (i, j), k not in {i, j}, whose consensus is
    nearest row k (Hamming; ties: the earlier merge). At most 2K exact forwards
    per round and O(K^3 L) distances, within the round's O(N L K^3).
    """
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


class ContextFounders:
    """The assembled context's founders in one block, for the dictionary code.

    ``founders`` holds the distinct local sequences of the context founders
    (this round's feedback proposals; -1 = no call). The multi-block assembly
    decided these founders exist. A founder claims its nearest panel row on
    its called sites (ties: the lowest row); a claimed row is coded by
    reference to its founder plus its edits, log C(n_f, d) for d differing
    sites among the founder's n_f called sites (the smallest over the row's
    claimers, and never more than coding it as one more free row), so a
    claimed row equal to a proposal costs nothing and a far one pays for its
    distance. Rows no founder claims are free, coded as new arbitrary
    haplotypes. Founders without a called site claim no row. Distances are
    cached per distinct row: O(K_c L) once per row, O(K_c K) per panel.
    """

    def __init__(self, founders):
        founders = np.asarray(founders, dtype=np.int64)
        called = founders >= 0
        informative = called.any(axis=1)
        self.founders = np.ascontiguousarray(founders[informative])
        self.called = np.ascontiguousarray(called[informative])
        self.n_called = self.called.sum(axis=1)
        self.distances = {}

    def __len__(self):
        return len(self.founders)

    def _columns(self, panel):
        columns = []
        for row in np.asarray(panel, dtype=np.int64):
            key = row.tobytes()
            value = self.distances.get(key)
            if value is None:
                value = ((self.founders != row[None, :]) & self.called).sum(axis=1)
                self.distances[key] = value
            columns.append(value)
        return np.stack(columns, axis=1)

    def claims(self, panel):
        """(claimed mask, free-row count, claimed rows' edit code in nats)."""
        k, length = np.shape(panel)
        claimed = np.zeros(k, dtype=bool)
        if not len(self.founders):
            return claimed, k, 0.
        distance = self._columns(panel)
        owner = np.argmin(distance, axis=1)
        claimed[owner] = True
        free = k - int(claimed.sum())
        # One more free row would cost log C(2^L, free + 1) - log C(2^L, free).
        free_row = float(log_binary_haplotype_set_count(length, free + 1)) - (
            float(log_binary_haplotype_set_count(length, free)) if free else 0.)
        code = 0.
        for row in np.flatnonzero(claimed):
            claimers = np.flatnonzero(owner == row)
            edits = min(math.lgamma(self.n_called[f] + 1) - math.lgamma(distance[f, row] + 1)
                        - math.lgamma(self.n_called[f] - distance[f, row] + 1) for f in claimers)
            code += min(edits, free_row)
        return claimed, free, code

    def claimed(self, panel):
        return self.claims(panel)[0]

    def free(self, panel):
        return self.claims(panel)[1]


def selection_objective(fit, learn_frequencies, context):
    """Fitted log-likelihood plus the regularizer that codes claimed rows by
    reference to their founders (edit code) and only the unclaimed rows as new
    haplotypes; without context, the fit's own objective (every row free)."""
    if context is None:
        return fit['objective']
    _, free, code = context.claims(fit['panel'])
    return fit['log_likelihood'] + regularizer(fit['panel'], fit['frequencies'],
                                               learn_frequencies, free, code)


def _digest(*arrays):
    """Exact content key of arrays (shape, dtype and bytes); a cache key only."""
    digest = hashlib.blake2b(digest_size=20)
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast('B'))
    return digest.digest()


# Posterior arrays of a fit, restored by one E-step instead of being memoized.
_POSTERIOR_FIELDS = ('allele_probability', 'occupancy')


class LikelihoodMemo:
    """Founder-independent likelihood results of one block, kept across rounds.

    Context founders enter only the selection regularizer (coded rows), never
    a likelihood. A fixed-K fit is a deterministic function of its canonical
    start, a forward score of its (panel, frequencies), and the merge bounds of
    a fit of that fit's (panel, frequencies), each given the block data, copying
    model and fit settings that ``bind`` checks. The kernels are thread-
    invariant (sample-ordered sums), so an entry equals its recomputation bit
    for bit. Fits are kept without their two K x L posterior arrays (one E-step
    restores them) and with bit-packed panels, restored as C-contiguous int64
    because proposal ties break on panel bytes: O(fits K L / 8) bytes.
    """

    def __init__(self, state=None):
        state = state or dict(data_key=None, fits={}, scores={}, bounds={})
        self.data_key = state['data_key']
        self.fits = dict(state['fits'])
        self.scores = dict(state['scores'])
        self.bounds = dict(state['bounds'])
        self.hits = dict(fits=0, scores=0, bounds=0, completed=0)

    def bind(self, data_key):
        """Discard entries of other block data, model or fit settings."""
        if data_key != self.data_key:
            self.data_key, self.fits, self.scores, self.bounds = data_key, {}, {}, {}

    def state(self):
        """Picklable entries for a later round of the same block."""
        return dict(data_key=self.data_key, fits=self.fits, scores=self.scores, bounds=self.bounds)

    def fit(self, key):
        stored = self.fits.get(key)
        if stored is None:
            return None
        self.hits['fits'] += 1
        length, packed = stored['panel']
        panel = np.ascontiguousarray(np.unpackbits(packed, axis=1, count=length), dtype=np.int64)
        return dict(stored, panel=panel)

    def add_fit(self, key, result):
        # Posterior fields stay as None placeholders, keeping the key order.
        panel = result['panel']
        self.fits[key] = {name: None if name in _POSTERIOR_FIELDS else value
                          for name, value in result.items()}
        self.fits[key]['panel'] = (panel.shape[1], np.packbits(panel.astype(np.uint8), axis=1))


def _memo_data_key(gl, observed, positions, model_options, fit_options):
    """Exact identity of everything a memoized result depends on besides its key."""
    intervals = model_options['interval_morgans']
    digest = hashlib.blake2b(_digest(gl, observed, positions), digest_size=20)
    digest.update(b'None' if intervals is None else _digest(intervals))
    digest.update(repr(sorted((name, value) for name, value in {**model_options, **fit_options}.items()
                              if name != 'interval_morgans')).encode())
    return digest.hexdigest()


def _search_input_key(start, bank, gl, observed, positions, options, founders=None):
    """Exact deterministic-search inputs; reuse only within compatible fits.

    Input data are included because a retained fit can cross feedback rounds
    through a checkpoint. This is a cache key, not a likelihood or confidence
    test. Candidate order, all search budgets and copying-model settings matter.
    """
    digest = hashlib.sha256(b'normalized-local-search-reuse-v2')
    for array in (start, bank, gl, observed, positions, options.pop('interval_morgans'), founders):
        if array is None:
            digest.update(b'None')
            continue
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast('B'))
    digest.update(repr(sorted(options.items())).encode())
    return digest.hexdigest()


def search_panel(start, bank, gl, observed, positions, *, generations,
                 learn_frequencies, refits_per_kind=8, round_backstop_base=10,
                 round_backstop_per_founder=2, update_backstop_base=20,
                 update_backstop_per_founder=4, update_backstop_per_site=2, initial_fit=None,
                 bulk=False, prepared=None,
                 previous_search=None, incumbent=None, recombination_rate_per_bp=5e-8,
                 interval_morgans=None, wildcard_mass=.01, founders=None, memo=None):
    """Local search from ``start``, or from ``incumbent`` when it scores better.

    ``incumbent`` is a previous-round fit of the same block data and objective.
    When its objective exceeds the fitted start's, the search begins from its
    refit instead, as ``initial_fit`` would; otherwise the search is unchanged.
    The reuse key still describes start, bank, data and settings, so a later
    round with identical inputs reuses this result. Rounds stop when the best
    refit does not improve; ``round_backstop`` records a backstop hit.
    ``founders`` (feedback rounds) are the assembled context's founders in this
    block: the objective then codes only the rows no founder claims as new
    haplotypes (ContextFounders); None (the initial fit) codes every row.
    ``memo`` (LikelihoodMemo) carries this block's fits, scores and merge
    bounds into later rounds; it is updated in place, None keeps them local.
    """
    assert not bulk, 'This optimized route preserves the ordinary-neighbour search only'
    model_options = dict(generations=generations,
        recombination_rate_per_bp=recombination_rate_per_bp,
        interval_morgans=interval_morgans, wildcard_mass=wildcard_mass)
    input_key = (None if initial_fit is not None else _search_input_key(
        start, bank, gl, observed, positions,
        dict(model_options, learn_frequencies=learn_frequencies,
             refits_per_kind=refits_per_kind, round_backstop_base=round_backstop_base,
             round_backstop_per_founder=round_backstop_per_founder,
             update_backstop_base=update_backstop_base,
             update_backstop_per_founder=update_backstop_per_founder,
             update_backstop_per_site=update_backstop_per_site), founders))
    if (input_key is not None and previous_search is not None
            and previous_search.get('search_input_key') == input_key):
        return dict(previous_search, search_start='reused', computational_work=dict(
            score_calls=0, score_cache_hits=0, fit_calls=0, fit_cache_hits=0,
            bound_passes=0, reused_previous_search=True))
    if prepared is None:
        prepared = scoring.prepare_observations(gl, observed, positions, **model_options)
    prepared = fitting.prepare_fit_observations(prepared)
    fit_options = dict(learn_frequencies=learn_frequencies,
                       update_backstop_base=update_backstop_base,
                       update_backstop_per_founder=update_backstop_per_founder,
                       update_backstop_per_site=update_backstop_per_site)
    memo = LikelihoodMemo() if memo is None else memo
    memo.bind(_memo_data_key(gl, observed, positions, model_options, fit_options))
    score_cache, fit_cache = {}, {}
    work = dict(score_calls=0, score_cache_hits=0, fit_calls=0, fit_cache_hits=0,
                bound_passes=0)

    def fit_panel(panel, frequencies=None):
        key = (panel.tobytes(), None if frequencies is None else frequencies.tobytes())
        if key in fit_cache:
            work['fit_cache_hits'] += 1
            return fit_cache[key]
        # fit_fixed depends on its input only through this canonical start.
        start_panel, start_frequencies = canonical(panel, frequencies)
        if not learn_frequencies:
            start_frequencies = np.full(len(start_panel), 1. / len(start_panel))
        memo_key = _digest(start_panel, start_frequencies)
        result = memo.fit(memo_key)
        if result is None:
            result = fitting.fit_fixed(panel, gl, observed, positions, frequencies=frequencies,
                                       prepared=prepared, **fit_options, **model_options)
            work['fit_calls'] += 1
            memo.add_fit(memo_key, result)
        fit_cache[key] = result
        return result

    def complete(value):
        """Restore a memoized fit's posterior arrays (proposals and output read
        them): a function of its (panel, frequencies), so the incumbent's when
        it is the same fit, else one E-step."""
        if value['occupancy'] is not None:
            return
        if (incumbent is not None and np.array_equal(value['panel'], incumbent['panel'])
                and np.array_equal(value['frequencies'], incumbent['frequencies'])):
            value.update(allele_probability=incumbent['allele_probability'],
                         occupancy=incumbent['occupancy'])
        else:
            memo.hits['completed'] += 1
            value.update(fitting.fit_posteriors(value['panel'], value['frequencies'], gl, observed,
                                                positions, prepared=prepared, **model_options))

    def log_likelihood(panel, frequency):
        key = (panel.tobytes(), frequency.tobytes())
        if key in score_cache:
            work['score_cache_hits'] += 1
            return score_cache[key]
        memo_key = _digest(panel, frequency)
        value = memo.scores.get(memo_key)
        if value is None:
            parallel.apply_dynamic_threads()
            value = scoring.score_prepared(panel, frequency, prepared)['log_likelihood']
            work['score_calls'] += 1
            memo.scores[memo_key] = value
        else:
            memo.hits['scores'] += 1
        score_cache[key] = value
        return value

    context = None if founders is None else ContextFounders(founders)

    def objective(value):
        return selection_objective(value, learn_frequencies, context)

    def coded(panel, frequency):
        if context is None:
            return regularizer(panel, frequency, learn_frequencies)
        _, free, code = context.claims(panel)
        return regularizer(panel, frequency, learn_frequencies, free, code)

    search_start = 'start' if initial_fit is None else 'initial_fit'
    if initial_fit is None:
        fit = fit_panel(start)
        if incumbent is not None and objective(incumbent) > objective(fit) + 1e-6:
            # Climbing from the context start back to a better incumbent
            # dominated L1/L2 cost; explore the incumbent's neighbourhood.
            initial_fit, search_start = incumbent, 'incumbent'
    if initial_fit is not None:
        trial = fit_panel(initial_fit['panel'], initial_fit['frequencies'])
        fit = trial if objective(trial) >= objective(initial_fit) else initial_fit
    trace = []
    round_backstop = False
    for round_index in itertools.count():
        if round_index >= round_backstop_base + round_backstop_per_founder * len(fit['panel']):
            round_backstop = True
            break
        ranked, seen, drops, merges = {}, set(), {}, []
        complete(fit)
        for kind, panel, frequency, origin in proposals(fit, bank):
            if not learn_frequencies:
                frequency = np.full(len(panel), 1./len(panel))
            key = (panel.tobytes(), frequency.tobytes())
            tagged_key = (kind, *key)
            if tagged_key in seen:
                continue
            seen.add(tagged_key)
            if kind == 'merge':
                # Screened below by its EM lower bound, not an exact forward.
                ranked.setdefault(kind, [])
                merges.append((panel, frequency, origin))
                continue
            score = log_likelihood(panel, frequency) + coded(panel, frequency)
            ranked.setdefault(kind, []).append((score, panel, frequency))
            if kind == 'drop':
                drops[origin] = (panel, frequency)
        if merges:
            # The drop and merge lists, hence the bounds, are functions of the
            # fit's (panel, frequencies) and the block data only.
            bounds_key = _digest(fit['panel'], fit['frequencies'])
            bounds = memo.bounds.get(bounds_key)
            if bounds is None:
                bounds, passes = merge_bounds(fit['panel'], drops, merges, gl, observed, positions,
                                              prepared, model_options)
                work['bound_passes'] += passes
                memo.bounds[bounds_key] = bounds
            else:
                memo.hits['bounds'] += 1
            exact = double_removal_merges(fit['panel'], merges)
            for index, (bound, (panel, frequency, _merge)) in enumerate(zip(bounds, merges)):
                if index in exact:
                    bound = log_likelihood(panel, frequency)
                ranked['merge'].append(
                    (bound + coded(panel, frequency), panel, frequency))
        trials = []
        for kind, options in ranked.items():
            options.sort(key=lambda value: (-value[0], value[1].tobytes()))
            for rank, (score, panel, frequency) in enumerate(options[:refits_per_kind]):
                trials.append((fit_panel(panel, frequency), kind, rank))
        if not trials:
            break
        best, kind, rank = min(trials, key=lambda value: (-objective(value[0]),
            len(value[0]['panel']), value[0]['panel'].tobytes()))
        trace.append(dict(round=round_index, proposals={k:len(v) for k,v in ranked.items()},
            refits=len(trials), old_k=len(fit['panel']), best_k=len(best['panel']),
            gain=objective(best)-objective(fit), kind=kind, winner_rank=rank,
            winner_kind_refits=min(len(ranked[kind]), refits_per_kind),
            winner_kind_proposals=len(ranked[kind])))
        if objective(best) <= objective(fit) + 1e-6:
            break
        fit = best
    complete(fit)
    return dict(fit, search_trace=trace, search_method='reference_neighbours',
                computational_work=work, search_input_key=input_key,
                search_start=search_start, round_backstop=round_backstop,
                selection_objective=objective(fit),
                context_founders=None if context is None else len(context),
                free_rows=len(fit['panel']) if context is None else context.free(fit['panel']))


def release_and_wildcard(fit, gl, observed, positions, config, generations, *,
                         recombination_rate_per_bp=5e-8, interval_morgans=None,
                         wildcard_mass=.01):
    """One diagnostic E-step supplies allele release and spatial MAP-U counts."""
    from scipy.special import expit
    from . import path_model as model
    state = model.e_step(fit['panel'], gl, positions, observed,
        generations=generations, founder_frequencies=fit['frequencies'],
        return_pair_posteriors=True, return_sample_stats=True,
        recombination_rate_per_bp=recombination_rate_per_bp,
        interval_morgans=interval_morgans, wildcard_mass=wildcard_mass)
    probability = expit(-state['conditional_flip_gains'])
    favorable = state['sample_conditional_flip_gains'] < -config.score_tolerance
    carrier = state['sample_carrier_probability']
    # Release needs min_directional_supporters (2) by either measure over the
    # samples whose own evidence favours the allele: as many different samples
    # each more likely than not carrying the founder at the site, or that much
    # expected carrier mass. (The mass alone sits just below 2 for a founder
    # with exactly two near-certain carriers; the count alone would withhold
    # diffusely supported sites the mass releases.)
    supporters = np.sum((carrier > .5) & favorable, axis=0)
    mass = np.sum(carrier * favorable, axis=0)
    confident = ((probability >= config.min_hard_call_pseudo_probability)
                 & (np.maximum(supporters, mass) >= config.min_directional_supporters))
    release = dict(calls=np.where(confident, fit['panel'], -1).astype(np.int8),
        current_allele_probability=probability, directional_supporters=supporters,
        directional_support=np.sum(carrier * favorable, axis=0),
        positive_sample_count=np.sum(favorable, axis=0),
        unknown_copy_mass=float(state['occupancy'][-1].sum()),
        total_copy_mass=float(state['occupancy'].sum()),
        unknown_copy_mass_by_site=state['occupancy'][-1],
        known_partner_support=state['current_support_mass'],
        expected_state_changes=float(state['expected_state_changes'].sum()),
        interpretation='Exact one-bit probability conditional on other panel alleles and fitted parameters; released when the favourable samples include min_directional_supporters with posterior carrier probability above 1/2 (directional_supporters) or carry that much expected carrier mass (directional_support). Not a full allele posterior.')
    wildcard = wildcard_compatibility(state['pair_posteriors'], observed)
    return release, wildcard
