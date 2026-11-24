"""The birth-order layer: the birth-order prior over pedigrees, sampled exactly inside every fit (4.22 R4, route B).

Model. Every sample has an unknown birth order pi; all total orders of the samples are equally likely a priori, and a
parent is born before its child. For each child c,

    Z_c(pi) = sum of w_r over the configuration rows r of c whose named parents all precede c in pi,

w_r = exp(score_r + direction_r - log full multiplicity of r's state), the configured state priors acting later in the
local selection. Rows that name no parent (M0) and exposure-untestable rows are always allowed. P(pi) is proportional
to prod_c Z_c(pi), so the marginal prior of a pedigree is proportional to the number of birth orders consistent with
it: shallow explanations are favoured (a sib named as a parent costs log 2 against the true pair), and every
configuration consistent with an order is acyclic. The layer replaces 4.21's reciprocal cavity messages,
short-ancestry path factor and greedy acyclic repair (PD08-PD10, PD13).

Inference. Gibbs sampling over orders: one sample s at a time (a fresh random visiting order each sweep) is
reinserted at an insertion index drawn from its exact conditional. After burn-in every sweep adds the
Rao-Blackwellised P(X_c = r | pi) = w_r 1[r allowed in pi] / Z_c(pi); the averages P_hat(X_c = r) are returned as
configuration scores log P_hat + log full multiplicity (so the local selection's marginal state evidence is the
sampled state posterior). The kept sweep-end orders give every fit's consensus birth order (each sample's mean number of
predecessors, its mean position: the joint read, selection.select, restricts the children on a cycle of the local
choices to it) and, in full-data fits only, each pair's order counts and from them the pair orientation log odds
log P(p before c) - log P(c before p), smoothed by half a count. Monte Carlo error: about sqrt(P (1 - P) / S_eff) on
P_hat, so about 1 % of P_hat for a row near one half (S_eff the effective kept sweeps); rows the data make near certain
are far tighter. Exactly tied configurations are therefore split by the chain's realisation, not by the tie protocol.

Incremental moves (cost per sweep O(total rows + rows naming each sample + N^2), not O(N x total rows)):
  - s's own rows add their weight at 1 + the reduced index of their later-born named parent (one pass over rows_s);
  - for each child c naming s only the rows of c that name s are visited (an index built once per fit): A_cs, their
    weight whose other named parent precedes c. Z_c without s is the maintained Z_c minus A_cs when s precedes c;
    where that subtraction would keep less than OrderParameters.cancellation_guard of Z_c it is recomputed exactly
    from c's rows, so every ratio keeps full relative precision;
  - the N insertion weights are one linear scan: own(j) times the product of the factors Zw / (Zw + A) (<= 1) of the
    children already passed, relative to index 0, whose weight is >= 1 (weights are relative to each child's best
    always-allowed row); no exp or log in the chain;
  - the order is updated by shifting the entries between the old and the new position; all Z_c are recomputed
    exactly at every sweep end (no drift crosses a sweep), in the pass that adds the Rao-Blackwell terms.

Randomness. Each fit draws from its own PCG64 stream (core.random's pedigree.order.chain, seeded by the run's
resampling.bootstrap_seed), keyed by the fit's own multiplicity vector or left-out unit (fit_key): the initial order
(one permutation), then one visiting permutation per sweep (Generator.permuted along the sweep axis), then one uniform
per move (Generator.random, sweeps x N). The permutations are drawn over the samples in sorted-identifier order (as the
relationship reference stream), so the input's column order does not change the realisation; renaming samples does.
The draws, and therefore the result, do not depend on threads, processes, task partitioning or evaluation order.

Serial. One chain is inherently sequential: each reinsertion is drawn given the order the previous one left. A fit runs
on one thread (the kernels release the GIL); fits are independent and run in parallel across the pedigree stage's
resampling workers.

Reference path: lab422/r4/scripts/order_mc.py (the step-2 prototype, a full rescan per move) and
lab422/r4/sampler/order_fast2.py (this algorithm with Numba's internal generator), both lab only.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from ...core import random
from ...core.errors import InvariantError
from ...core.kernels import kernel
from .decision_parameters import DirectionParameters
from .kernels.order import OVERLOADS
from .order_parameters import OrderParameters

__all__ = ["OrderDiagnostics", "OrderFit", "adjust", "chain_draws", "fit_key", "order_stream", "run_chain"]


@dataclass(frozen=True)
class OrderDiagnostics:
    """One fit's chain: sweeps kept, guarded exact recomputes of Z_c without s, children whose weight reference was
    raised (a row more than max_weight_span above the child's best always-allowed row)."""

    kept_sweeps: int
    guarded_recomputes: int
    raised_references: int


@dataclass(frozen=True)
class OrderFit:
    """One fit's birth-order layer: configuration scores [A] for the marginal state and the identity (identity -inf
    where exposure-untestable), the consensus birth order (``rank`` float64 [N], each sample's mean number of samples
    born before it over the kept sweeps, smaller first) and the chain's diagnostics."""

    state_scores: np.ndarray
    identity_scores: np.ndarray
    rank: np.ndarray
    chain: OrderDiagnostics


# ------------------------------------------------------------------------------------------------ randomness


def fit_key(weights: np.ndarray, omitted: int | None = None) -> int:
    """The 64-bit key of one fit: SHA-256 (first eight bytes, little-endian) of its float64 unit multiplicities and
    its left-out unit (-1 for none). Bootstrap vectors and leave-one-unit-out fits are distinct exact values, so each
    fit owns its stream; equal vectors are the same fit and draw the same numbers."""
    left_out = np.int64(-1 if omitted is None else omitted).astype("<i8")
    data = np.ascontiguousarray(weights, dtype="<f8").tobytes() + left_out.tobytes()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def order_stream(key: int, seed: int) -> np.random.Generator:
    """PCG64 stream of one fit's chain (core.random's pedigree.order.chain: SeedSequence([seed, 8000000, key])); no
    draw is consumed."""
    return random.stream("pedigree.order.chain", seed=int(seed), key=int(key))


def chain_draws(rng: np.random.Generator, sample_order: np.ndarray, sweeps: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(initial order int64[N], visiting order per sweep int64[S, N], one uniform per move float64[S, N]), drawn in
    that order from ``rng``. The permutations are drawn over positions in ``sample_order`` (sample indices in
    sorted-identifier order) and mapped to sample indices, so a reordering of the input's columns, with its
    sample_order, gives the same draws per sample identifier."""
    sample_order = np.asarray(sample_order, dtype=np.int64)
    samples = len(sample_order)
    initial = sample_order[rng.permutation(samples)]
    visits = np.tile(np.arange(samples, dtype=np.int64), (sweeps, 1))
    rng.permuted(visits, axis=1, out=visits)
    uniforms = rng.random((sweeps, samples))
    return initial, sample_order[visits], uniforms


# ---------------------------------------------------------------------------------------------------- kernels


@kernel(overloads=OVERLOADS["order._index"], cache=True, nogil=True)
def _index(rows, starts, ell, constrained, n, max_span):
    """Per-fit index: linear row weights relative to each child's reference, and the constrained rows naming each
    sample grouped by child (entry weight and the row's other named parent)."""
    n_rows = rows.shape[0]
    v = np.zeros(n_rows)
    raised = 0
    no_free = 0
    for c in range(n):
        m_free = -np.inf
        m_all = -np.inf
        for r in range(starts[c], starts[c + 1]):
            if ell[r] > m_all:
                m_all = ell[r]
            if not constrained[r] and ell[r] > m_free:
                m_free = ell[r]
        ref = m_free
        if m_free == -np.inf:
            no_free += 1
            ref = m_all
        if m_all - ref > max_span:
            ref = m_all - max_span
            raised += 1
        if ref == -np.inf:
            continue
        for r in range(starts[c], starts[c + 1]):
            if ell[r] > -np.inf:
                v[r] = math.exp(ell[r] - ref)
    ent_count = np.zeros(n + 1, dtype=np.int64)
    for r in range(n_rows):
        if constrained[r]:
            for slot in range(1, 3):
                p = rows[r, slot]
                if p >= 0:
                    ent_count[p + 1] += 1
    ent_ptr = np.cumsum(ent_count)
    ent_v = np.empty(ent_ptr[n])
    ent_other = np.empty(ent_ptr[n], dtype=np.int64)
    ent_child = np.empty(ent_ptr[n], dtype=np.int64)
    fill = ent_ptr[:-1].copy()
    for r in range(n_rows):  # rows are grouped by child in increasing order: each sample's entries come out by child
        if constrained[r]:
            a, b = rows[r, 1], rows[r, 2]
            if a >= 0:
                k = fill[a]
                ent_v[k] = v[r]
                ent_other[k] = b
                ent_child[k] = rows[r, 0]
                fill[a] += 1
            if b >= 0:
                k = fill[b]
                ent_v[k] = v[r]
                ent_other[k] = a
                ent_child[k] = rows[r, 0]
                fill[b] += 1
    groups = 0
    for s in range(n):
        last = -1
        for k in range(ent_ptr[s], ent_ptr[s + 1]):
            if ent_child[k] != last:
                groups += 1
                last = ent_child[k]
    grp_ptr = np.zeros(n + 1, dtype=np.int64)
    grp_child = np.empty(groups, dtype=np.int64)
    grp_start = np.empty(groups + 1, dtype=np.int64)
    g = 0
    for s in range(n):
        last = -1
        for k in range(ent_ptr[s], ent_ptr[s + 1]):
            if ent_child[k] != last:
                grp_child[g] = ent_child[k]
                grp_start[g] = k
                g += 1
                last = ent_child[k]
        grp_ptr[s + 1] = g
    grp_start[groups] = ent_ptr[n]
    return v, grp_ptr, grp_child, grp_start, ent_v, ent_other, raised, no_free


@kernel(overloads=OVERLOADS["order._allowed"], cache=True, nogil=True, inline="always")
def _allowed(rows, constrained, pos, r, c):
    """Whether row r of child c is allowed under the order with positions ``pos``."""
    if not constrained[r]:
        return True
    a, b = rows[r, 1], rows[r, 2]
    if a >= 0 and pos[a] >= pos[c]:
        return False
    if b >= 0 and pos[b] >= pos[c]:
        return False
    return True


@kernel(overloads=OVERLOADS["order._exact_z"], cache=True, nogil=True)
def _exact_z(rows, starts, constrained, v, pos, z):
    """Z_c of every child under ``pos``, summed afresh (written into ``z``)."""
    for c in range(starts.shape[0] - 1):
        total = 0.0
        for r in range(starts[c], starts[c + 1]):
            if _allowed(rows, constrained, pos, r, c):
                total += v[r]
        z[c] = total


@kernel(overloads=OVERLOADS["order._chain"], cache=True, nogil=True)
def _chain(rows, starts, constrained, v, grp_ptr, grp_child, grp_start, ent_v, ent_other, initial, visits, uniforms,
           burn, guard, count_before):
    """The Gibbs chain over the draws: sweep t reinserts visits[t, 0], visits[t, 1], ... in turn, each at the
    insertion index whose cumulative weight first exceeds uniforms[t, k] x the total, then recomputes every Z_c
    exactly. Returns (P_hat per row, pair order counts ([1, 1] unless count_before), summed positions over the kept
    sweeps, kept sweeps, guarded recomputes, final order)."""
    n = initial.shape[0]
    sweeps = visits.shape[0]
    order = initial.copy()
    pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos[order[i]] = i
    n_rows = rows.shape[0]
    acc_rows = np.zeros(n_rows)
    before = np.zeros((n, n) if count_before else (1, 1), dtype=np.int64)
    position_sum = np.zeros(n, dtype=np.int64)
    z = np.zeros(n)
    _exact_z(rows, starts, constrained, v, pos, z)
    max_grp = 1
    for s in range(n):
        max_grp = max(max_grp, grp_ptr[s + 1] - grp_ptr[s])
    lin_add = np.zeros(n)
    mul = np.ones(n)
    own_at = np.empty(n)
    cum = np.empty(n)
    scratch = np.empty(n_rows)
    g_zw = np.empty(max_grp)
    g_a = np.empty(max_grp)
    kept = 0
    recomputes = 0
    for sweep in range(sweeps):
        for k in range(visits.shape[1]):
            s = visits[sweep, k]
            ps = pos[s]
            # Insertion weights of s (index j of the order without s; relative to index 0, whose weight is s's
            # always-allowed weight >= 1): own rows unlock at 1 + the reduced index of their later-born named parent;
            # each child c naming s contributes Zw / (Zw + A) from index rp(c) + 1 on, Zw = Z_c without the rows naming
            # s (subtracted, or recomputed exactly below the cancellation guard) and A = their weight whose other
            # parent precedes c. Written inline: as a called kernel (even IR-inlined) the move ran 5 % slower.
            base = 0.0
            for r in range(starts[s], starts[s + 1]):
                if not constrained[r]:
                    base += v[r]
                    continue
                a, b = rows[r, 1], rows[r, 2]
                u = 0
                if a >= 0:
                    u = pos[a] - 1 if pos[a] > ps else pos[a]
                if b >= 0:
                    ub = pos[b] - 1 if pos[b] > ps else pos[b]
                    if ub > u or a < 0:
                        u = ub
                lin_add[u + 1] += v[r]
            g0 = grp_ptr[s]
            for g in range(g0, grp_ptr[s + 1]):
                c = grp_child[g]
                pc = pos[c]
                a_cs = 0.0
                for e in range(grp_start[g], grp_start[g + 1]):
                    o = ent_other[e]
                    if o < 0 or pos[o] < pc:
                        a_cs += ent_v[e]
                if ps < pc:
                    zw = z[c] - a_cs
                    if zw <= guard * z[c]:
                        recomputes += 1
                        zw = 0.0
                        for r in range(starts[c], starts[c + 1]):
                            if not constrained[r]:
                                zw += v[r]
                                continue
                            a, b = rows[r, 1], rows[r, 2]
                            if a == s or b == s:
                                continue
                            if a >= 0 and pos[a] >= pc:
                                continue
                            if b >= 0 and pos[b] >= pc:
                                continue
                            zw += v[r]
                else:
                    zw = z[c]
                g_zw[g - g0] = zw
                g_a[g - g0] = a_cs
                if a_cs > 0.0:
                    rc = pc - 1 if pc > ps else pc
                    mul[rc + 1] *= zw / (zw + a_cs)
            own = base
            factor = 1.0
            total = 0.0
            for j in range(n):
                own += lin_add[j]
                factor *= mul[j]
                lin_add[j] = 0.0
                mul[j] = 1.0
                own_at[j] = own
                total += own * factor
                cum[j] = total
            u_draw = uniforms[sweep, k] * total
            lo = 0
            hi = n - 1
            while lo < hi:  # first index whose cumulative weight exceeds the draw
                mid = (lo + hi) // 2
                if cum[mid] > u_draw:
                    hi = mid
                else:
                    lo = mid + 1
            j_new = lo
            while j_new > 0 and cum[j_new] == cum[j_new - 1]:  # never an index of zero weight
                j_new -= 1
            g0 = grp_ptr[s]
            for g in range(g0, grp_ptr[s + 1]):  # children whose order relative to s flipped
                c = grp_child[g]
                pc = pos[c]
                rc = pc - 1 if pc > ps else pc
                now_before = j_new <= rc
                if (ps < pc) != now_before:
                    z[c] = g_zw[g - g0] + (g_a[g - g0] if now_before else 0.0)
            z[s] = own_at[j_new]
            if j_new < ps:
                for i in range(ps, j_new, -1):
                    order[i] = order[i - 1]
                    pos[order[i]] = i
            elif j_new > ps:
                for i in range(ps, j_new):
                    order[i] = order[i + 1]
                    pos[order[i]] = i
            order[j_new] = s
            pos[s] = j_new
        keep = sweep >= burn
        for c in range(n):  # exact Z at every sweep end; after burn-in, the Rao-Blackwell terms
            t_z = 0.0
            for r in range(starts[c], starts[c + 1]):
                x = v[r] if _allowed(rows, constrained, pos, r, c) else 0.0
                scratch[r] = x
                t_z += x
            z[c] = t_z
            if keep and t_z > 0.0:
                inv = 1.0 / t_z
                for r in range(starts[c], starts[c + 1]):
                    acc_rows[r] += scratch[r] * inv
        if keep:
            kept += 1
            for i in range(n):
                position_sum[order[i]] += i
            if count_before:
                for i in range(n):
                    p = order[i]
                    for jj in range(i + 1, n):
                        before[p, order[jj]] += 1
    if kept > 0:
        for r in range(n_rows):
            acc_rows[r] /= kept
    return acc_rows, before, position_sum, kept, recomputes, order


# ---------------------------------------------------------------------------------------------------- drivers


def run_chain(rows: np.ndarray, starts: np.ndarray, ell: np.ndarray, constrained: np.ndarray, samples: int,
              draws: tuple[np.ndarray, np.ndarray, np.ndarray], burn: int, parameters: OrderParameters,
              count_before: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, OrderDiagnostics]:
    """The chain on prepared arrays (rows sorted by child, ``starts`` their child offsets, ``ell`` the log weights,
    ``constrained`` the rows whose named parents must precede the child) and explicit draws. Returns P_hat per row,
    the pair order counts (int64[N, N] when ``count_before``, else [1, 1]), each sample's summed position over the
    kept sweeps (int64[N]), the kept sweeps and the diagnostics."""
    # Owned, C-contiguous, writable arrays (copied only when needed): one declared overload per kernel, never a second
    # specialisation for a read-only record array.
    owned = ("C_CONTIGUOUS", "WRITEABLE", "OWNDATA")
    rows = np.require(rows, dtype=np.int64, requirements=owned)
    starts = np.require(starts, dtype=np.int64, requirements=owned)
    ell = np.require(ell, dtype=np.float64, requirements=owned)
    constrained = np.require(constrained, dtype=np.bool_, requirements=owned)
    initial, visits, uniforms = (np.require(draws[0], dtype=np.int64, requirements=owned),
                                 np.require(draws[1], dtype=np.int64, requirements=owned),
                                 np.require(draws[2], dtype=np.float64, requirements=owned))
    if len(rows) and np.any(np.diff(rows[:, 0]) < 0):
        raise InvariantError("configuration rows are not grouped by child")
    v, grp_ptr, grp_child, grp_start, ent_v, ent_other, raised, no_free = _index(
        rows, starts, ell, constrained, int(samples), float(parameters.max_weight_span))
    if no_free:
        raise InvariantError(f"{no_free} children have no always-allowed configuration row (M0 or untestable)")
    probability, before, position_sum, kept, recomputes, _ = _chain(
        rows, starts, constrained, v, grp_ptr, grp_child, grp_start, ent_v, ent_other, initial, visits, uniforms,
        int(burn), float(parameters.cancellation_guard), bool(count_before))
    return probability, before, position_sum, int(kept), OrderDiagnostics(int(kept), int(recomputes), int(raised))


def adjust(
    state_scores: np.ndarray,
    identity_scores: np.ndarray,
    rows: np.ndarray,
    states: np.ndarray,
    full_counts: np.ndarray,
    orientation,
    direction: DirectionParameters,
    parameters: OrderParameters,
    key: int,
    trace: dict | None = None,
    *,
    seed: int,
    sample_order: np.ndarray,
) -> OrderFit:
    """The birth-order layer of one fit (module docstring).

    Inputs: configuration scores [A] (state, identity: -inf where exposure-untestable), rows [A, 3] grouped by child,
    states [A], full_counts [N, 3], the fit's orientation and the direction parameters, the order parameters, the
    fit's stream ``key`` (fit_key), the run's ``seed`` (resampling.bootstrap_seed) and ``sample_order`` (sample
    indices in sorted-identifier order, chain_draws). Returns OrderFit: scores log P_hat(X_c = r) + log full
    multiplicity (identity -inf where the input identity is -inf), the consensus birth order and the chain's
    diagnostics (timing belongs to the stage's task records: science imports no clock). With ``trace`` (full-data
    fits: full_sweeps, else resample_sweeps) the per-row direction term, the layer's term per row (output minus the
    row's input weight), the pair orientation log odds [c, p] and the kept sweeps are recorded in it."""
    samples = orientation.log_odds.shape[0]
    eligible = np.isfinite(identity_scores)
    # log(2 P(parent -> child | one orientation holds)) under the pair-level failure mixture, as family.adjust
    epsilon = max(direction.pair_failure_probability, np.finfo(float).eps)
    log_support = -np.logaddexp(0.0, -orientation.log_odds)
    log_direction = math.log(2.0) + np.logaddexp(math.log1p(-epsilon) + log_support, math.log(0.5 * epsilon))
    direction_term = np.zeros(len(rows))
    for slot in (1, 2):
        present = (rows[:, slot] >= 0) & eligible
        direction_term[present] += log_direction[rows[present, 0], rows[present, slot]]
    state_scores = state_scores + direction_term
    identity_scores = identity_scores + direction_term
    log_full = np.log(np.maximum(1, full_counts))
    child = rows[:, 0]
    named = (rows[:, 1] >= 0) | (rows[:, 2] >= 0)
    constrained = eligible & named
    ell = np.where(eligible, identity_scores, state_scores) - log_full[child, states]
    ell = np.where(np.isfinite(ell), ell, -np.inf)
    starts = np.searchsorted(rows[:, 0], np.arange(samples + 1)).astype(np.int64)
    full_fit = trace is not None
    burn = parameters.burn_in_sweeps
    sweeps = burn + (parameters.full_sweeps if full_fit else parameters.resample_sweeps)
    if len(sample_order) != samples:
        raise InvariantError("the chain's sample order must list every sample once")
    draws = chain_draws(order_stream(key, seed), sample_order, sweeps)
    # pair order counts (O(N^2) per kept sweep) only for the full-data fit's orientation log odds; every fit's
    # consensus order is its summed positions (O(N) per kept sweep), equal to the counts' column sums
    probability, before, position_sum, kept, chain = run_chain(rows, starts, ell, constrained, samples, draws, burn,
                                                               parameters, full_fit)
    with np.errstate(divide="ignore"):
        log_p = np.where(probability > 0, np.log(probability), -np.inf)
    score = log_p + log_full[child, states]
    state_out = score
    identity_out = np.where(eligible, score, -np.inf)
    if trace is not None:
        odds = np.log(before.T + 0.5) - np.log(before + 0.5)  # [c, p]: log P(p before c) - log P(c before p)
        trace.update(direction=direction_term, birth_order=score - np.where(eligible, identity_scores, state_scores),
                     orientation_log_odds=odds, birth_order_sweeps=kept)
    # a sample's mean number of predecessors (its mean position, an integer sum over kept sweeps) orders the samples
    rank = position_sum / max(kept, 1)
    return OrderFit(state_out, identity_out, rank, chain)
