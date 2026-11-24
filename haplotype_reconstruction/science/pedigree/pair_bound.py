"""Valid upper bounds on the M2 likelihood of every pair of a child's genome-retained parents.

The M2 panel (``panel.bounded_panel``) holds every pair of a child's retained parents; this bound only ranks a
child's pairs when they exceed the per-child cap (the largest full-data bounds are kept, the child is flagged, and
the cap can discard competitive pairs). The exact M2 forward (``forward.component_m2``) runs on the resulting panel.

The bound relaxes the M2 forward of one component (``read_forward.score_path`` mode 2) by STATE AGGREGATION.
Each parent's states at each bin are grouped: every state with transmitted-state mass pi >= ``high_mass`` is its own
group, BACKGROUND is its own group, and the remaining named states form one low group. The aggregated forward uses
(i) the exact initial mass of each group, (ii) the largest exact row mass of the parent's bridge-with-excursion
operator from a state of the source group into the destination group, and (iii) the largest joint emission bound
over the two groups. By induction over bins every aggregated value is at least the exact forward mass of the states
it covers, so the aggregated likelihood bounds M2 from above for any grouping (the grouping only sets tightness).
The two slots keep their own chains, and the emission couples them bin by bin: a pair is scored jointly, never as the
sum of its parents' separate fits.

Emission bounds, valid for every candidate pair: named founder states transmit their panel allele (or BACKGROUND's
frequency where uncalled) in every candidate (``read_source_kernels._background_alt`` never conditions them), so their
joint emissions are exact child-level values. BACKGROUND is the only candidate-specific state: its ALT probability lies
in the component's range over every allele a slot can transmit as BACKGROUND (the available candidates' read-conditioned
values and, when the component has an evidence-free candidate, the null chain's unconditional frequency), and the
per-site likelihood is bilinear in the two transmitted ALT probabilities (it peaks at a corner of that box); in addition
a pair whose parent q is in BACKGROUND is capped by q's own partner bound (q's actual BACKGROUND alleles against the
best partner state in that range). An unavailable candidate (no source evidence; production scores the pair as the other
parent's M1, or M0, with the child's null chain in its slot) is one group with the largest null-kernel row sum as its
transition. Each bin's normaliser enters through its own logarithm, and a bin whose aggregated mass would underflow is
redone in the log domain, so a valid bound never becomes infinite or NaN through rounding.

Cost per pair and bin: a few joint groups (two to four per slot when paintings are confident), one small group
transition and one normalisation; emission probabilities are exponentiated once per child and bin, in tables
indexed by the batch's listed children. Per child the work is linear in its number of retained pairs; there is no
per-site work per pair.

Method: docs/methods/pedigree.md#candidate-panel
"""

from __future__ import annotations

import math

import numpy as np
from numba import prange

from ...core.errors import InvariantError
from ...core.kernels import kernel
from .forward import apply_bridge, apply_excursion, site_likelihood
from .kernels.bounds import OVERLOADS
from .numerical_parameters import TINY as _TINY

# A bin whose aggregated normaliser falls below this is redone in the log domain (no emission mass lost to underflow);
# every bin's normaliser enters the log likelihood through its own logarithm (no running product to overflow).

# Temporary per-batch workspace of child emission tables and partner bounds (not a setting).

# Pair rows per batch.


@kernel(overloads=OVERLOADS["pair_bound.emission_bound"], cache=True, parallel=True, nogil=True)
def emission_bound(
    state_alt,
    background_low,
    background_high,
    child_gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    mismatch,
    child_list,
):
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
                        a = (
                            state_alt[x, site]
                            if x < background
                            else (background_low[site] if i == 0 else background_high[site])
                        )
                        for j in range(2 if y == background else 1):
                            b = (
                                state_alt[y, site]
                                if y < background
                                else (background_low[site] if j == 0 else background_high[site])
                            )
                            value = site_likelihood(child_gl[child, site], coefficient[child, site], a, b, mismatch)
                            if value > best:
                                best = value
                    if best <= 0.0:
                        out[local, block, x, y] = -np.inf
                    else:
                        out[local, block, x, y] += math.log(best)
    return out


@kernel(overloads=OVERLOADS["pair_bound.partner_bound"], cache=True, parallel=True, nogil=True)
def partner_bound(
    state_alt,
    background_low,
    background_high,
    candidate_background,
    child_gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    mismatch,
    children,
    offsets,
    candidates,
):
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
                    a = (
                        state_alt[x, site]
                        if x < background
                        else (background_low[site] if i == 0 else background_high[site])
                    )
                    v = site_likelihood(child_gl[child, site], coefficient[child, site], a, b, mismatch)
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


@kernel(overloads=OVERLOADS["pair_bound._candidate_groups"], cache=True, nogil=True)
def _candidate_groups(
    n,
    pi,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    branch,
    excursion,
    available,
    high_mass,
    null_row_max,
    group,
    count,
    initial,
    transfer,
):
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
        # On the identity, output[destination, source] = T'(source -> destination) (forward.advance, axis 0).
        apply_bridge(
            current,
            output,
            stay[n, block],
            alpha[n, block],
            beta[n, block],
            row_arm[n, block],
            column_arm[n, block],
            pivot[n, block],
            0,
            prefix,
            suffix,
            branch[n, block],
        )
        if excursion[n, block] > 0.0:
            apply_excursion(current, output, excursion[n, block], 0)
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


@kernel(overloads=OVERLOADS["pair_bound._group_counts"], cache=True, parallel=True, nogil=True)
def _group_counts(pi, available, high_mass):
    candidates, bins, states = pi.shape
    background = states - 1
    count = np.ones((candidates, bins), dtype=np.int64)
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


@kernel(overloads=OVERLOADS["pair_bound._fill_groups"], cache=True, parallel=True, nogil=True)
def _fill_groups(
    pi,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    branch,
    excursion,
    available,
    high_mass,
    null_row_max,
    group,
    count,
    initial,
    transfer,
):
    for n in prange(pi.shape[0]):
        _candidate_groups(
            n,
            pi,
            stay,
            alpha,
            beta,
            row_arm,
            column_arm,
            pivot,
            branch,
            excursion,
            available,
            high_mass,
            null_row_max,
            group,
            count,
            initial,
            transfer,
        )


def slot_groups(
    pi, stay, alpha, beta, row_arm, column_arm, pivot, branch, excursion, available, high_mass, null_row_max
):
    """(group int8 [N, B, S], count int8 [N, B], initial [N, G], transfer [N, B - 1, G, G]) with G the largest group
    count of the component (compact, since the pair kernel reads both parents' transfers at every bin)."""
    candidates, bins, states = pi.shape
    count = _group_counts(pi, available, high_mass)
    width = int(count.max()) if count.size else 1
    if width > np.iinfo(np.int8).max:
        raise InvariantError(f"pair-bound group count {width} exceeds int8 capacity")
    count = count.astype(np.int8)
    group = np.zeros((candidates, bins, states), dtype=np.int8)
    initial = np.zeros((candidates, width))
    transfer = np.zeros((candidates, max(bins - 1, 0), width, width))
    _fill_groups(
        pi,
        stay,
        alpha,
        beta,
        row_arm,
        column_arm,
        pivot,
        branch,
        excursion,
        available,
        high_mass,
        null_row_max,
        group,
        count,
        initial,
        transfer,
    )
    return group, count, initial, transfer


@kernel(overloads=OVERLOADS["pair_bound.scaled_emissions"], cache=True, parallel=True, nogil=True)
def scaled_emissions(log_emission):
    """Emission bounds relative to each child and bin's best state pair: (probabilities [n, B, S, S], their summed
    log offset [n], the bin maxima [n, B])."""
    children, bins, states, _ = log_emission.shape
    scaled = np.zeros(log_emission.shape)
    top = np.full((children, bins), -np.inf)
    offset = np.zeros(children)
    for local in prange(children):
        for block in range(bins):
            maximum = -np.inf
            for x in range(states):
                for y in range(states):
                    if log_emission[local, block, x, y] > maximum:
                        maximum = log_emission[local, block, x, y]
            top[local, block] = maximum
            if not np.isfinite(maximum):
                continue
            offset[local] += maximum
            for x in range(states):
                for y in range(states):
                    scaled[local, block, x, y] = math.exp(log_emission[local, block, x, y] - maximum)
    return scaled, offset, top


@kernel(overloads=OVERLOADS["pair_bound._step"], cache=True, nogil=True)
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


@kernel(overloads=OVERLOADS["pair_bound._log_absorb"], cache=True, nogil=True)
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


@kernel(overloads=OVERLOADS["pair_bound._absorb"], cache=True, nogil=True)
def _absorb(
    alpha,
    saved,
    factor,
    log_group,
    n_p,
    n_q,
    group_p,
    group_q,
    log_emission,
    local,
    block,
    maximum,
    cap_p,
    cap_q,
    states,
):
    """Multiply the aggregated forward by its group emissions and normalise; returns (normaliser, True) on the
    probability path, else (log normaliser or -inf, False): the log path redoes an underflowing bin from the log
    bounds."""
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
            v = log_emission[local, block, x, y] - maximum
            if x == background and cap_p < v:
                v = cap_p
            if y == background and cap_q < v:
                v = cap_q
            if v > log_group[gx, gy]:
                log_group[gx, gy] = v
    return _log_absorb(alpha, log_group, n_p, n_q), False


@kernel(overloads=OVERLOADS["pair_bound._pair_row"], cache=True, nogil=True)
def _pair_row(
    row,
    trios,
    row_child,
    entry_p,
    entry_q,
    partner,
    log_emission,
    scaled,
    top,
    group,
    count,
    initial,
    transfer,
    available,
    partner_scaled,
):
    states = log_emission.shape[2]
    bins = log_emission.shape[1]
    background = states - 1
    groups = initial.shape[1]
    p, q = trios[row, 1], trios[row, 2]
    local = row_child[row]
    alpha = np.zeros((groups, groups))
    work = np.empty((groups, groups))
    factor = np.empty((groups, groups))
    saved = np.empty((groups, groups))
    log_group = np.empty((groups, groups))
    for i in range(count[p, 0]):
        for j in range(count[q, 0]):
            alpha[i, j] = initial[p, i] * initial[q, j]
    log_likelihood = 0.0
    for block in range(bins):
        n_p, n_q = count[p, block], count[q, block]
        if block > 0:
            _step(alpha, work, count[p, block - 1], count[q, block - 1], n_p, n_q, transfer[p, block - 1],
                  transfer[q, block - 1])
        maximum = top[local, block]
        if not np.isfinite(maximum):
            return -np.inf
        cap_p = partner[entry_p[row], block] - maximum if available[p] else np.inf
        cap_q = partner[entry_q[row], block] - maximum if available[q] else np.inf
        ps_p = partner_scaled[entry_p[row], block] if available[p] else 1.0
        ps_q = partner_scaled[entry_q[row], block] if available[q] else 1.0
        for i in range(n_p):
            for j in range(n_q):
                factor[i, j] = 0.0
        group_p = group[p, block]
        group_q = group[q, block]
        for x in range(states):
            gx = group_p[x]
            for y in range(states):
                gy = group_q[y]
                v = scaled[local, block, x, y]
                if x == background:
                    v = min(v, ps_p)
                if y == background:
                    v = min(v, ps_q)
                if v > factor[gx, gy]:
                    factor[gx, gy] = v
        value, probability = _absorb(alpha, saved, factor, log_group, n_p, n_q, group_p, group_q, log_emission, local,
                                     block, maximum, cap_p, cap_q, states)
        if probability:
            log_likelihood += math.log(value)
        elif np.isfinite(value):
            log_likelihood += value
        else:
            return -np.inf
    return log_likelihood


@kernel(overloads=OVERLOADS["pair_bound.pair_bounds"], cache=True, parallel=True, nogil=True)
def pair_bounds(
    trios,
    row_child,
    entry_p,
    entry_q,
    partner,
    log_emission,
    scaled,
    top,
    offset,
    group,
    count,
    initial,
    transfer,
    available,
):
    """float64 [T]: the bound on the M2 log likelihood of each row (child, first, second) on one component;
    ``row_child`` is each row's position in the batch's child list (the emission tables' first axis)."""
    # Every retained-parent entry belongs to one child. Its cap is reused by all
    # that child's pairs, so exponentiate it once per entry and bin, not per pair.
    entry_local = np.zeros(partner.shape[0], dtype=np.int64)
    for row in range(trios.shape[0]):
        entry_local[entry_p[row]] = row_child[row]
        entry_local[entry_q[row]] = row_child[row]
    partner_scaled = np.ones(partner.shape)
    for entry in prange(partner.shape[0]):
        for block in range(partner.shape[1]):
            cap = partner[entry, block] - top[entry_local[entry], block]
            if cap < 0.0:
                partner_scaled[entry, block] = math.exp(cap)
    out = np.empty(trios.shape[0])
    for row in prange(trios.shape[0]):
        value = _pair_row(row, trios, row_child, entry_p, entry_q, partner, log_emission, scaled, top, group, count,
                          initial, transfer, available, partner_scaled)
        out[row] = value + offset[row_child[row]]
    return out
