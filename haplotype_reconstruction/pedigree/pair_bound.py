"""Valid upper bounds on the M2 likelihood of every pair of a child's genome-retained parents (release 3.4).

The M2 panel (``panel.bounded_panel``) admits a pair only when this bound can reach the child's best known
configuration, so no pair whose M2 could be best (or within the panel window of the best) is left out, while the
exact M2 forward (``forward.score_pairs``) runs only on the admitted pairs.

The bound relaxes the M2 forward of one component (``forward._score_path`` mode 2; both views) by STATE AGGREGATION.
Each parent's states at each bin are grouped: every state with transmitted-state mass pi >= ``high_mass`` is its own
group, BACKGROUND is its own group, and the remaining named states form one low group. The aggregated forward uses
(i) the exact initial mass of each group, (ii) the largest exact row mass of the parent's bridge-with-excursion
operator from a state of the source group into the destination group, and (iii) the largest joint emission bound
over the two groups. By induction over bins every aggregated value is at least the exact forward mass of the states
it covers, so the aggregated likelihood bounds M2 from above for any grouping (the grouping only sets tightness).
The two slots keep their own chains, and the emission couples them bin by bin: a pair is scored jointly, never as the
sum of its parents' separate fits.

Emission bounds, valid for every candidate pair: named founder states transmit their panel allele (or BACKGROUND's
frequency where uncalled) in every candidate (``projection._transmitted_alt`` never conditions them), so their joint
emissions are exact child-level values. BACKGROUND is the only candidate-specific state: its ALT probability lies in
the component's range over every allele a slot can transmit as BACKGROUND (the available candidates' read-conditioned
values and, when the component has an evidence-free candidate, the null chain's unconditional frequency), and the
per-site likelihood is bilinear in the two transmitted ALT probabilities (it peaks at a corner of that box); in
addition a pair whose parent q is in BACKGROUND is capped by q's own partner bound (q's actual BACKGROUND alleles
against the best partner state in that range). An unavailable candidate (no source evidence; production scores the
pair as the other parent's M1, or M0, with the child's null chain in its slot) is one group with the largest
null-kernel row sum as its transition. The tempered view raises the same bounds to the bin's information exponent
(monotone). Each bin's normaliser enters through its own logarithm, and a bin whose aggregated mass would underflow
is redone in the log domain, so a valid bound never becomes infinite or NaN through rounding.

Cost per pair and bin: a few joint groups (two to four per slot when paintings are confident), two small group
transitions and two normalisations; emission probabilities are exponentiated once per child and bin, in tables
indexed by the batch's listed children. Per child the work is linear in its number of retained pairs; there is no
per-site work per pair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .forward import _apply_bridge, _apply_excursion, _site_likelihood

# A bin whose aggregated normaliser falls below this is redone in the log domain (no emission mass lost to underflow);
# every bin's normaliser enters the log likelihood through its own logarithm (no running product to overflow).
_TINY = 1e-280
# Temporary per-batch workspace of child emission tables and partner bounds (not a setting).
WORKSPACE_BYTES = 512 << 20
# Pair rows per batch.
BATCH_ROWS = 4_000_000


@njit(cache=True, parallel=True)
def emission_bound(state_alt, background_low, background_high, child_gl, coefficient, site_index, site_start,
                   site_stop, mismatch, child_list):
    """float64 [n, B, S, S]: log upper bound on the bin emission of each ordered state pair, any candidate pair, for
    the n children of ``child_list`` (tables indexed by position in the list, never by an index span)."""
    states = state_alt.shape[0]
    bins = site_start.shape[1]
    children = len(child_list)
    out = np.zeros((children, bins, states, states))
    background = states - 1
    for task in prange(children * bins):
        local = task // bins
        block = task - local * bins
        child = child_list[local]
        for slot in range(site_start[child, block], site_stop[child, block]):
            site = site_index[slot]
            for x in range(states):
                for y in range(states):
                    best = 0.0
                    for i in range(2 if x == background else 1):
                        a = (state_alt[x, site] if x < background
                             else (background_low[site] if i == 0 else background_high[site]))
                        for j in range(2 if y == background else 1):
                            b = (state_alt[y, site] if y < background
                                 else (background_low[site] if j == 0 else background_high[site]))
                            value = _site_likelihood(child_gl[child, site], coefficient[child, site], a, b,
                                                     mismatch)
                            if value > best:
                                best = value
                    if best <= 0.0:
                        out[local, block, x, y] = -np.inf
                    else:
                        out[local, block, x, y] += math.log(best)
    return out


@njit(cache=True, parallel=True)
def partner_bound(state_alt, background_low, background_high, candidate_background, child_gl, coefficient,
                  site_index, site_start, site_stop, mismatch, children, offsets, candidates):
    """float64 [K, B]: log bound on any emission with candidate q in BACKGROUND, for the (child, candidate) entries
    children[i] x candidates[offsets[i]:offsets[i + 1]]: the best partner state against q's own BACKGROUND alleles."""
    states = state_alt.shape[0]
    bins = site_start.shape[1]
    background = states - 1
    total = offsets[-1]
    out = np.empty((total, bins))
    for task in prange(total * bins):
        entry = task // bins
        block = task - entry * bins
        low, high = 0, len(children)
        while high - low > 1:
            middle = (low + high) // 2
            if offsets[middle] <= entry:
                low = middle
            else:
                high = middle
        child = children[low]
        candidate = candidates[entry]
        best_state = -np.inf
        for x in range(states):
            value = 0.0
            for slot in range(site_start[child, block], site_stop[child, block]):
                site = site_index[slot]
                b = candidate_background[candidate, site]
                best = 0.0
                for i in range(2 if x == background else 1):
                    a = (state_alt[x, site] if x < background
                         else (background_low[site] if i == 0 else background_high[site]))
                    v = _site_likelihood(child_gl[child, site], coefficient[child, site], a, b, mismatch)
                    if v > best:
                        best = v
                if best <= 0.0:
                    value = -np.inf
                    break
                value += math.log(best)
            if value > best_state:
                best_state = value
        out[entry, block] = best_state
    return out


@njit(cache=True)
def _candidate_groups(n, pi, stay, alpha, beta, row_arm, column_arm, pivot, branch, excursion, available, high_mass,
                      null_row_max, group, count, initial, transfer):
    bins, states = pi.shape[1], pi.shape[2]
    background = states - 1
    if not available[n]:
        initial[n, 0] = 1.0
        for block in range(bins - 1):
            transfer[n, block, 0, 0] = null_row_max[block]
        return
    for block in range(bins):
        groups = 0
        for s in range(states):
            if pi[n, block, s] >= high_mass or s == background:
                group[n, block, s] = groups
                groups += 1
        low = 0
        for s in range(states):
            if pi[n, block, s] < high_mass and s != background:
                group[n, block, s] = groups
                low = 1
        count[n, block] = groups + low
    for s in range(states):
        initial[n, group[n, 0, s]] += pi[n, 0, s]
    current = np.zeros((states, states))
    output = np.zeros((states, states))
    prefix = np.zeros(states + 1)
    suffix = np.zeros(states + 1)
    mass = np.zeros(states + 1)
    for block in range(bins - 1):
        for s in range(states):
            for t in range(states):
                current[s, t] = 1.0 if s == t else 0.0
        # On the identity, output[destination, source] = T'(source -> destination) (forward._advance, axis 0).
        _apply_bridge(current, output, stay[n, block], alpha[n, block], beta[n, block], row_arm[n, block],
                      column_arm[n, block], pivot[n, block], 0, prefix, suffix, branch[n, block])
        if excursion[n, block] > 0.0:
            _apply_excursion(current, output, excursion[n, block], 0)
        for source in range(states):
            for h in range(states + 1):
                mass[h] = 0.0
            for destination in range(states):
                value = output[destination, source]
                if value > 0.0:
                    mass[group[n, block + 1, destination]] += value
            g = group[n, block, source]
            for h in range(count[n, block + 1]):
                if mass[h] > transfer[n, block, g, h]:
                    transfer[n, block, g, h] = mass[h]


@njit(cache=True, parallel=True)
def _group_counts(pi, available, high_mass):
    candidates, bins, states = pi.shape
    background = states - 1
    count = np.ones((candidates, bins), dtype=np.int8)
    for n in prange(candidates):
        if not available[n]:
            continue
        for block in range(bins):
            groups = 0
            low = 0
            for s in range(states):
                if pi[n, block, s] >= high_mass or s == background:
                    groups += 1
                else:
                    low = 1
            count[n, block] = groups + low
    return count


@njit(cache=True, parallel=True)
def _fill_groups(pi, stay, alpha, beta, row_arm, column_arm, pivot, branch, excursion, available, high_mass,
                 null_row_max, group, count, initial, transfer):
    for n in prange(pi.shape[0]):
        _candidate_groups(n, pi, stay, alpha, beta, row_arm, column_arm, pivot, branch, excursion, available,
                          high_mass, null_row_max, group, count, initial, transfer)


def slot_groups(pi, stay, alpha, beta, row_arm, column_arm, pivot, branch, excursion, available, high_mass,
                null_row_max):
    """(group int8 [N, B, S], count int8 [N, B], initial [N, G], transfer [N, B - 1, G, G]) with G the largest group
    count of the component (compact, since the pair kernel reads both parents' transfers at every bin)."""
    candidates, bins, states = pi.shape
    count = _group_counts(pi, available, high_mass)
    width = int(count.max()) if count.size else 1
    group = np.zeros((candidates, bins, states), dtype=np.int8)
    initial = np.zeros((candidates, width))
    transfer = np.zeros((candidates, max(bins - 1, 0), width, width))
    _fill_groups(pi, stay, alpha, beta, row_arm, column_arm, pivot, branch, excursion, available, high_mass,
                 null_row_max, group, count, initial, transfer)
    return group, count, initial, transfer


@njit(cache=True, parallel=True)
def scaled_emissions(log_emission, exponent, child_list):
    """Emission bounds relative to each child and bin's best state pair: (full, tempered [n, B, S, S] probabilities,
    their summed log offsets [n] per view, the bin maxima [n, B])."""
    children, bins, states, _ = log_emission.shape
    full = np.zeros(log_emission.shape)
    tempered = np.zeros(log_emission.shape)
    top = np.full((children, bins), -np.inf)
    offset_full = np.zeros(children)
    offset_tempered = np.zeros(children)
    for local in prange(children):
        child = child_list[local]
        for block in range(bins):
            maximum = -np.inf
            for x in range(states):
                for y in range(states):
                    if log_emission[local, block, x, y] > maximum:
                        maximum = log_emission[local, block, x, y]
            top[local, block] = maximum
            if not np.isfinite(maximum):
                continue
            eta = exponent[child, block]
            offset_full[local] += maximum
            if eta != 0.0:
                offset_tempered[local] += eta * maximum
            for x in range(states):
                for y in range(states):
                    d = log_emission[local, block, x, y] - maximum
                    full[local, block, x, y] = math.exp(d)
                    tempered[local, block, x, y] = 1.0 if eta == 0.0 else math.exp(eta * d)
    return full, tempered, offset_full, offset_tempered, top


@njit(cache=True)
def _step(alpha, work, n_from_p, n_from_q, n_to_p, n_to_q, tp, tq):
    """alpha[to_p, to_q] = sum alpha[from_p, from_q] tp[from_p, to_p] tq[from_q, to_q] (axis by axis)."""
    for i in range(n_from_p):
        for j in range(n_to_q):
            value = 0.0
            for k in range(n_from_q):
                value += alpha[i, k] * tq[k, j]
            work[i, j] = value
    for i in range(n_to_p):
        for j in range(n_to_q):
            value = 0.0
            for k in range(n_from_p):
                value += work[k, j] * tp[k, i]
            alpha[i, j] = value


@njit(cache=True)
def _log_absorb(alpha, log_group, n_p, n_q):
    """alpha <- alpha x exp(log_group), normalised; returns the log normaliser. The shift is the largest
    log(alpha) + log_group over groups holding mass, so no group's product underflows against an empty maximum."""
    maximum = -np.inf
    for i in range(n_p):
        for j in range(n_q):
            if alpha[i, j] > 0.0:
                value = math.log(alpha[i, j]) + log_group[i, j]
                if value > maximum:
                    maximum = value
    if not np.isfinite(maximum):
        return -np.inf
    total = 0.0
    for i in range(n_p):
        for j in range(n_q):
            if alpha[i, j] > 0.0:
                alpha[i, j] = math.exp(math.log(alpha[i, j]) + log_group[i, j] - maximum)
            total += alpha[i, j]
    if total <= 0.0 or not np.isfinite(total):
        return -np.inf
    for i in range(n_p):
        for j in range(n_q):
            alpha[i, j] /= total
    return maximum + math.log(total)


@njit(cache=True)
def _absorb_view(alpha, saved, factor, log_group, n_p, n_q, group_p, group_q, log_emission, local, block, maximum,
                 eta, cap_p, cap_q, states):
    """Multiply one view's aggregated forward by its group emissions and normalise; returns (log normaliser or
    -inf, whether the probability path was used). The log path redoes an underflowing bin from the log bounds."""
    background = states - 1
    total = 0.0
    for i in range(n_p):
        for j in range(n_q):
            saved[i, j] = alpha[i, j]
            alpha[i, j] *= factor[i, j]
            total += alpha[i, j]
    if total > _TINY and np.isfinite(total):
        for i in range(n_p):
            for j in range(n_q):
                alpha[i, j] /= total
        return total, True
    for i in range(n_p):
        for j in range(n_q):
            alpha[i, j] = saved[i, j]
            log_group[i, j] = -np.inf
    for x in range(states):
        gx = group_p[x]
        for y in range(states):
            gy = group_q[y]
            if eta == 0.0:
                v = 0.0
            else:
                v = log_emission[local, block, x, y] - maximum
                if x == background and cap_p < v:
                    v = cap_p
                if y == background and cap_q < v:
                    v = cap_q
                v *= eta
            if v > log_group[gx, gy]:
                log_group[gx, gy] = v
    return _log_absorb(alpha, log_group, n_p, n_q), False


@njit(cache=True)
def _pair_row(row, trios, row_child, entry_p, entry_q, partner, log_emission, full_e, tempered_e, top, exponent,
              group, count, initial, transfer, available, partner_full, partner_tempered):
    states = log_emission.shape[2]
    bins = log_emission.shape[1]
    background = states - 1
    groups = initial.shape[1]
    child, p, q = trios[row, 0], trios[row, 1], trios[row, 2]
    local = row_child[row]
    a_full = np.zeros((groups, groups))
    a_temp = np.zeros((groups, groups))
    work = np.empty((groups, groups))
    g_full = np.empty((groups, groups))
    g_temp = np.empty((groups, groups))
    saved = np.empty((groups, groups))
    log_group = np.empty((groups, groups))
    for i in range(count[p, 0]):
        for j in range(count[q, 0]):
            a_full[i, j] = initial[p, i] * initial[q, j]
            a_temp[i, j] = a_full[i, j]
    log_full = 0.0
    log_temp = 0.0
    for block in range(bins):
        n_p, n_q = count[p, block], count[q, block]
        if block > 0:
            a_p, a_q = count[p, block - 1], count[q, block - 1]
            _step(a_full, work, a_p, a_q, n_p, n_q, transfer[p, block - 1], transfer[q, block - 1])
            _step(a_temp, work, a_p, a_q, n_p, n_q, transfer[p, block - 1], transfer[q, block - 1])
        maximum = top[local, block]
        if not np.isfinite(maximum):
            return -np.inf
        eta = exponent[child, block]
        cap_p = partner[entry_p[row], block] - maximum if available[p] else np.inf
        cap_q = partner[entry_q[row], block] - maximum if available[q] else np.inf
        pf_p = partner_full[entry_p[row], block] if available[p] else 1.0
        pf_q = partner_full[entry_q[row], block] if available[q] else 1.0
        pt_p = partner_tempered[entry_p[row], block] if available[p] else 1.0
        pt_q = partner_tempered[entry_q[row], block] if available[q] else 1.0
        for i in range(n_p):
            for j in range(n_q):
                g_full[i, j] = 0.0
                g_temp[i, j] = 0.0
        group_p = group[p, block]
        group_q = group[q, block]
        for x in range(states):
            gx = group_p[x]
            for y in range(states):
                gy = group_q[y]
                vf = full_e[local, block, x, y]
                vt = tempered_e[local, block, x, y]
                if x == background:
                    vf = min(vf, pf_p)
                    vt = min(vt, pt_p)
                if y == background:
                    vf = min(vf, pf_q)
                    vt = min(vt, pt_q)
                if vf > g_full[gx, gy]:
                    g_full[gx, gy] = vf
                if vt > g_temp[gx, gy]:
                    g_temp[gx, gy] = vt
        value, scaled = _absorb_view(a_full, saved, g_full, log_group, n_p, n_q, group_p, group_q, log_emission,
                                     local, block, maximum, 1.0, cap_p, cap_q, states)
        if scaled:
            log_full += math.log(value)
        elif np.isfinite(value):
            log_full += value
        else:
            return -np.inf
        value, scaled = _absorb_view(a_temp, saved, g_temp, log_group, n_p, n_q, group_p, group_q, log_emission,
                                     local, block, maximum, eta, cap_p, cap_q, states)
        if scaled:
            log_temp += math.log(value)
        elif np.isfinite(value):
            log_temp += value
        else:
            return -np.inf
    return 0.5 * (log_full + log_temp)


@njit(cache=True, parallel=True)
def pair_bounds(trios, row_child, entry_p, entry_q, partner, log_emission, full_e, tempered_e, top, offset_full,
                offset_tempered, exponent, group, count, initial, transfer, available):
    """float64 [T]: the bound on 0.5 (tempered + full) M2 of each row (child, first, second) on one component;
    ``row_child`` is each row's position in the batch's child list (the emission tables' first axis)."""
    # Every retained-parent entry belongs to one child. Its cap is reused by all
    # that child's pairs, so exponentiate it once per entry and bin, not per pair.
    entry_child = np.zeros(partner.shape[0], dtype=np.int64)
    entry_local = np.zeros(partner.shape[0], dtype=np.int64)
    for row in range(trios.shape[0]):
        entry_child[entry_p[row]] = trios[row, 0]
        entry_child[entry_q[row]] = trios[row, 0]
        entry_local[entry_p[row]] = row_child[row]
        entry_local[entry_q[row]] = row_child[row]
    partner_full = np.ones(partner.shape)
    partner_tempered = np.ones(partner.shape)
    for entry in prange(partner.shape[0]):
        for block in range(partner.shape[1]):
            cap = partner[entry, block] - top[entry_local[entry], block]
            eta = exponent[entry_child[entry], block]
            if cap < 0.0:
                partner_full[entry, block] = math.exp(cap)
                if eta != 0.0:
                    partner_tempered[entry, block] = math.exp(eta * cap)
    out = np.empty(trios.shape[0])
    for row in prange(trios.shape[0]):
        value = _pair_row(row, trios, row_child, entry_p, entry_q, partner, log_emission, full_e, tempered_e, top,
                          exponent, group, count, initial, transfer, available, partner_full, partner_tempered)
        local = row_child[row]
        out[row] = value + 0.5 * (offset_full[local] + offset_tempered[local])
    return out


@dataclass(frozen=True)
class PairBatch:
    """Retained pairs of consecutive children: trios int64 [T, 3] (child, first, second) in the canonical row order
    (children ascending, then the compact profile's ascending parents, pairs i < j); the batch's (child, retained
    candidate) entries as CSR (children, offsets, candidates); each row's position in ``children`` and the entries of
    its two parents."""

    trios: np.ndarray
    children: np.ndarray
    offsets: np.ndarray
    candidates: np.ndarray
    entry_p: np.ndarray
    entry_q: np.ndarray
    row_child: np.ndarray | None = None

    def __post_init__(self):
        if self.row_child is None:              # children ascending: each row's position in the child list
            object.__setattr__(self, "row_child", np.searchsorted(self.children, self.trios[:, 0]).astype(np.int64))


def retained_pair_count(compact) -> int:
    counts = np.diff(compact.indptr)
    return int(np.sum(counts * (counts - 1) // 2))


def retained_batches(compact, bins: int, states: int):
    """PairBatch generator over every retained pair, bounded by BATCH_ROWS rows and WORKSPACE_BYTES of tables."""
    per_child = 3 * bins * states * states * 8        # emission tables of one listed child (never an index span)
    trios, kids, cands, ep, eq, local = [], [], [], [], [], []
    offsets = [0]
    rows = 0
    for child in range(compact.n_samples):
        a, b = compact.indptr[child:child + 2]
        parents = np.asarray(compact.parents[a:b], dtype=np.int64)
        r = len(parents)
        if r < 2:
            continue
        i, j = np.triu_indices(r, 1)
        base = offsets[-1]
        trios.append(np.column_stack([np.full(len(i), child, np.int64), parents[i], parents[j]]))
        ep.append(base + i)
        eq.append(base + j)
        local.append(np.full(len(i), len(kids), np.int64))
        kids.append(child)
        cands.append(parents)
        offsets.append(base + r)
        rows += len(i)
        # Partner log bounds plus the full/tempered probability factors.
        workspace = len(kids) * per_child + offsets[-1] * bins * 24
        if rows >= BATCH_ROWS or workspace >= WORKSPACE_BYTES:
            yield PairBatch(np.concatenate(trios), np.asarray(kids, np.int64), np.asarray(offsets, np.int64),
                            np.concatenate(cands), np.concatenate(ep), np.concatenate(eq), np.concatenate(local))
            trios, kids, cands, ep, eq, local, offsets, rows = [], [], [], [], [], [], [0], 0
    if trios:
        yield PairBatch(np.concatenate(trios), np.asarray(kids, np.int64), np.asarray(offsets, np.int64),
                        np.concatenate(cands), np.concatenate(ep), np.concatenate(eq), np.concatenate(local))


@dataclass(frozen=True)
class ComponentContext:
    """Per-component inputs of the pair bound shared by every batch: the slot groups and transfers of every
    candidate, and BACKGROUND's ALT range over every allele a slot can transmit as BACKGROUND (with the candidates'
    BACKGROUND ALT [N, M])."""

    projection: object
    terms: object
    group: np.ndarray
    count: np.ndarray
    initial: np.ndarray
    transfer: np.ndarray
    background: np.ndarray
    low: np.ndarray
    high: np.ndarray


def component_context(projection, terms, high_mass: float, claim_threads=lambda: None) -> ComponentContext:
    bins = projection.n_bins
    null_row = terms.null_diagonal + (projection.n_states - 1) * terms.null_off
    null_row_max = (np.max(null_row, axis=0) if null_row.size else np.zeros(max(bins - 1, 0))) * (1.0 + 1e-12)
    claim_threads()
    group, count, initial, transfer = slot_groups(
        projection.pi, projection.stay, projection.alpha, projection.beta, projection.row_arm,
        projection.column_arm, projection.pivot, projection.branch, projection.excursion, projection.available,
        float(high_mass), np.ascontiguousarray(null_row_max))
    background = np.ascontiguousarray(projection.candidate_alt[:, :, -1])
    # The range holds every BACKGROUND ALT probability a slot can transmit: each available candidate's (conditioned on
    # its reads) and, where an evidence-free candidate makes a pair fall back to M1/M0 (forward.score_pairs), the null
    # chain's unconditional frequency state_alt[-1]. Without evidence-free candidates the narrower box is kept.
    null_background = np.asarray(projection.state_alt[-1], dtype=np.float64)
    if np.any(projection.available):
        low = np.min(background[projection.available], axis=0)
        high = np.max(background[projection.available], axis=0)
        if not np.all(projection.available):
            low = np.minimum(low, null_background)
            high = np.maximum(high, null_background)
    else:
        low = high = null_background
    return ComponentContext(projection, terms, group, count, initial, transfer, background,
                            np.ascontiguousarray(low), np.ascontiguousarray(high))


def component_bounds(context: ComponentContext, batch: PairBatch, claim_threads=lambda: None) -> np.ndarray:
    """float64 [T]: the bound of each batch row on one component."""
    projection, terms = context.projection, context.terms
    claim_threads()
    log_emission = emission_bound(projection.state_alt, context.low, context.high, terms.gl, terms.coefficient,
                                  terms.site_index, terms.site_start, terms.site_stop, terms.mismatch, batch.children)
    claim_threads()
    partner = partner_bound(projection.state_alt, context.low, context.high, context.background, terms.gl,
                            terms.coefficient, terms.site_index, terms.site_start, terms.site_stop, terms.mismatch,
                            batch.children, batch.offsets, batch.candidates)
    claim_threads()
    full_e, tempered_e, offset_full, offset_tempered, top = scaled_emissions(log_emission, terms.exponent,
                                                                             batch.children)
    claim_threads()
    return pair_bounds(batch.trios, batch.row_child, batch.entry_p, batch.entry_q, partner, log_emission, full_e,
                       tempered_e, top, offset_full, offset_tempered, terms.exponent, context.group, context.count,
                       context.initial, context.transfer, projection.available)


def chromosome_bounds(prepared, compact, config, claim_threads=lambda: None) -> np.ndarray:
    """float32 [P]: per retained pair (canonical row order) the chromosome's M2 bound, the sum over its components,
    rounded up to float32 (never below the float64 bound)."""
    from . import scoring
    high_mass = config.panel.pair_bound_high_mass
    contexts = []
    for component in prepared.components:
        state = scoring.resident(component, config, claim_threads)
        contexts.append(component_context(state.projection, state.terms, high_mass, claim_threads))
        del state
    out = np.zeros(retained_pair_count(compact), dtype=np.float32)
    bins = max((component.n_bins for component in prepared.components), default=1)
    n_states = max((component.n_states for component in prepared.components), default=1)
    cursor = 0
    for batch in retained_batches(compact, bins, n_states):
        value = np.zeros(len(batch.trios))
        for context in contexts:
            value += component_bounds(context, batch, claim_threads)
        rounded = value.astype(np.float32)
        below = rounded.astype(np.float64) < value
        rounded[below] = np.nextafter(rounded[below], np.float32(np.inf))
        out[cursor:cursor + len(value)] = rounded
        cursor += len(value)
    return out
