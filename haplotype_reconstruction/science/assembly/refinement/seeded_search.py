"""Seeded window search with shared max-plus messages.

N denotes samples, W units, B total bins, R the largest row alphabet, Q=K(K+1)/2
pair states, P<=N+1 distinct wishes, L=1+SEED_STARTS+INTERVAL_SEED_STARTS,
and I=LOCAL_ITERATIONS. Candidate count C<=W R. All scores are nats.

The state after bin t contains focal_values[K] and bin_maxima[0:t+1].
A fixed suffix is a max-plus covector (beta, gamma, const): its value is
max(const, max_c focal_values[c]+beta[c], max_j bin_maxima[j]+gamma[j]).
Forward states and backward covectors of the incumbent score restrictions and
single-unit moves without repeating the unchanged parts of each path.

Relevant sample wishes and their restrictions provide six starts each, besides
the incumbent. Exact single-unit ascent selects the best endpoint in canonical
content order. The ordinary recurrence in proposals.path_value rescores it.
"""

from __future__ import annotations

import numpy as np

from haplotype_reconstruction.core.kernels import kernel

from .emissions import pair_index
from .overloads import overloads_for
from .parameters import INTERVAL_SEED_STARTS, LOCAL_ITERATIONS, RELEVANCE_MARGIN, SEED_STARTS


@kernel(overloads=overloads_for(__name__, "prepare"), cache=True, nogil=True)
def prepare(emissions, cols, cands, offsets, focal, first, second, fz, bg):
    """Gather focal/background emissions and bin-to-unit indices. O(N B (Q+R K))."""
    n = emissions[0].shape[0]
    k = len(fz)
    w = len(emissions)
    bins = np.zeros(w + 1, np.int64)
    rmax = 0
    for u in range(w):
        bins[u + 1] = bins[u] + emissions[u].shape[2]
        rmax = max(rmax, offsets[u + 1] - offsets[u])
    focal_emissions = np.empty((n, bins[-1], rmax, k))
    background_emissions = np.empty((n, bins[-1], len(bg)))
    unit = np.empty(bins[-1], np.int64)
    for u in range(w):
        e = emissions[u]
        for t in range(e.shape[2]):
            bn = bins[u] + t
            unit[bn] = u
            for j in range(len(bg)):
                z = bg[j]
                a, b = cols[u, first[z]], cols[u, second[z]]
                for s in range(n):
                    background_emissions[s, bn, j] = e[s, pair_index(a, b), t]
            for i in range(offsets[u], offsets[u + 1]):
                row = cands[i]
                for c in range(k):
                    other = row if c == focal else cols[u, c]
                    for s in range(n):
                        focal_emissions[s, bn, row, c] = e[s, pair_index(row, other), t]
    return focal_emissions, background_emissions, unit, bins


@kernel(overloads=overloads_for(__name__, "context"), cache=True, nogil=True)
def context(background_emissions, start, end, fz, bg, penalty):
    """Eliminate background runs for closed flanks. O(N B^2 Q). Returns runs, head, tail, full, focal_start,
    focal_end."""
    n, bins, nb = background_emissions.shape
    runs = np.full((n, bins, bins), -np.inf)
    head = np.full((n, bins), -np.inf)
    tail = np.full((n, bins), -np.inf)
    full = np.full(n, -np.inf)
    focal_start = np.empty((n, len(fz)))
    focal_end = np.empty((n, len(fz)))
    for s in range(n):
        top = -np.inf
        for z in range(start.shape[1]):
            top = max(top, start[s, z])
        for c in range(len(fz)):
            focal_start[s, c] = max(start[s, fz[c]], top - penalty)
            focal_end[s, c] = end[s, fz[c]]
        for j in range(nb):
            a = max(start[s, bg[j]], top - penalty)
            b = end[s, bg[j]]
            for i in range(bins):
                acc = 0.0
                for t in range(i, bins):
                    acc += background_emissions[s, t, j]
                    runs[s, i, t] = max(runs[s, i, t], acc)
                    if i == 0:
                        head[s, t] = max(head[s, t], a + acc)
                tail[s, i] = max(tail[s, i], acc + b)
                if i == 0:
                    full[s] = max(full[s], a + acc + b)
    return runs, head, tail, full, focal_start, focal_end


@kernel(overloads=overloads_for(__name__, "samples_value"), cache=True, nogil=True)
def samples_value(focal_emissions, unit, rows, begin, stop, ctx, penalty):
    """Per-sample path values on bins [begin, stop). O(N (B K+B^2))."""
    runs, head, tail, full, focal_start, focal_end = ctx
    n, k = focal_start.shape
    length = stop - begin
    out = np.empty(n)
    focal_values = np.empty(k)
    maxima = np.empty(length)
    for s in range(n):
        focal_values[:] = focal_start[s]
        top = -np.inf
        for c in range(k):
            top = max(top, focal_values[c])
        # A background start may exceed every focal start. Its first-bin
        # switch is already included in the closed focal_start and head arrays.
        for t in range(length):
            b = head[s, t]
            for i in range(1, t + 1):
                b = max(b, maxima[i - 1] - penalty + runs[s, i, t])
            row = rows[unit[begin + t]]
            ftop = -np.inf
            for c in range(k):
                focal_values[c] = max(focal_values[c], top - penalty) + focal_emissions[s, begin + t, row, c]
                ftop = max(ftop, focal_values[c])
            top = max(b, ftop)
            maxima[t] = top
        value = full[s]
        for i in range(1, length):
            value = max(value, maxima[i - 1] - penalty + tail[s, i])
        for c in range(k):
            value = max(value, focal_values[c] + focal_end[s, c])
        out[s] = value
    return out


@kernel(overloads=overloads_for(__name__, "value"), cache=True, nogil=True)
def value(focal_emissions, unit, rows, ctx, penalty):
    """Sum sample path values in sample order. O(N (B K+B^2))."""
    vals = samples_value(focal_emissions, unit, rows, 0, focal_emissions.shape[1], ctx, penalty)
    total = 0.0
    for s in range(len(vals)):
        total += vals[s]
    return total


@kernel(overloads=overloads_for(__name__, "advance"), cache=True, nogil=True)
def advance(focal_emissions, background_emissions, rows, start, begin, stop, fz, bg, penalty):
    """Forward message for one row per sample on a bin interval. O(N B Q)."""
    current = start.copy()
    out = np.empty_like(start)
    for t in range(begin, stop):
        for s in range(start.shape[0]):
            top = -np.inf
            for z in range(start.shape[1]):
                top = max(top, current[s, z])
            floor = top - penalty
            for c in range(len(fz)):
                z = fz[c]
                out[s, z] = max(current[s, z], floor) + focal_emissions[s, t, rows[s], c]
            for j in range(len(bg)):
                z = bg[j]
                out[s, z] = max(current[s, z], floor) + background_emissions[s, t, j]
        current[:] = out
    return current


@kernel(overloads=overloads_for(__name__, "latent"), cache=True, nogil=True)
def latent(focal_emissions, background_emissions, unit, rows, fz, bg, penalty, start, end):
    """Trace the optimal bin states for sample-specific unit rows. O(N B Q). Stay wins transition ties, then the
    first pair state."""
    n, q = start.shape
    bins = focal_emissions.shape[1]
    chosen = np.empty((n, bins), np.int64)
    previous = np.empty((bins, q), np.int64)
    state_values = np.empty(q)
    out = np.empty(q)
    for s in range(n):
        state_values[:] = start[s]
        for t in range(bins):
            arg = 0
            for z in range(1, q):
                if state_values[z] > state_values[arg]:
                    arg = z
            floor = state_values[arg] - penalty
            for z in range(q):
                previous[t, z] = z if state_values[z] >= floor else arg
            for c in range(len(fz)):
                z = fz[c]
                out[z] = max(state_values[z], floor) + focal_emissions[s, t, rows[s, unit[t]], c]
            for j in range(len(bg)):
                z = bg[j]
                out[z] = max(state_values[z], floor) + background_emissions[s, t, j]
            state_values[:] = out
        zbest = 0
        for z in range(1, q):
            if state_values[z] + end[s, z] > state_values[zbest] + end[s, zbest]:
                zbest = z
        for t in range(bins - 1, -1, -1):
            chosen[s, t] = zbest
            zbest = previous[t, zbest]
    return chosen


@kernel(overloads=overloads_for(__name__, "wishes"), cache=True, nogil=True)
def wishes(
    focal_emissions, background_emissions, unit, bins, cands, offsets, incumbent, fz, bg, penalty, start, end, upper
):
    """Jointly choose each sample's whole-unit rows and hidden path. O(N (B^2 Q+R B K+R B^2)). Units unused by its
    focal path retain the incumbent."""
    n = start.shape[0]
    w = len(incumbent)
    paths = np.empty((n, w), np.int64)
    prefix = start.copy()
    for u in range(w):
        ctx = context(background_emissions[:, bins[u] : bins[u + 1]], prefix, upper[u + 1], fz, bg, penalty)
        scores = np.empty((offsets[u + 1] - offsets[u], n))
        trial = incumbent.copy()
        for i in range(offsets[u], offsets[u + 1]):
            trial[u] = cands[i]
            scores[i - offsets[u]] = samples_value(focal_emissions, unit, trial, bins[u], bins[u + 1], ctx, penalty)
        for s in range(n):
            best = 0
            for i in range(offsets[u], offsets[u + 1]):
                if cands[i] == incumbent[u]:
                    best = i - offsets[u]
            for i in range(offsets[u], offsets[u + 1]):
                j = i - offsets[u]
                if scores[j, s] > scores[best, s]:
                    best = j
            paths[s, u] = cands[offsets[u] + best]
        prefix = advance(
            focal_emissions, background_emissions, paths[:, u], prefix, bins[u], bins[u + 1], fz, bg, penalty
        )
    chosen = latent(focal_emissions, background_emissions, unit, paths, fz, bg, penalty, start, end)
    has = np.zeros(start.shape[1], np.bool_)
    for c in range(len(fz)):
        has[fz[c]] = True
    for s in range(n):
        for u in range(w):
            uses = False
            for t in range(bins[u], bins[u + 1]):
                uses = uses or has[chosen[s, t]]
            if not uses:
                paths[s, u] = incumbent[u]
    return paths


@kernel(overloads=overloads_for(__name__, "before"), cache=True, nogil=True)
def before(a, b, incumbent, reverse):
    """Incumbent-first lexicographic content order, reversed when requested. O(W)."""
    for u in range(len(a)):
        if a[u] == b[u]:
            continue
        if a[u] == incumbent[u]:
            return True
        if b[u] == incumbent[u]:
            return False
        return a[u] > b[u] if reverse else a[u] < b[u]
    return False


@kernel(overloads=overloads_for(__name__, "unique"), cache=True, nogil=True)
def unique(paths, count, incumbent, reverse):
    """Sort paths by canonical order and remove duplicates. O(P^2 W)."""
    order = np.arange(count)
    for i in range(1, count):
        x = order[i]
        j = i
        while j and before(paths[x], paths[order[j - 1]], incumbent, reverse):
            order[j] = order[j - 1]
            j -= 1
        order[j] = x
    out = np.empty_like(paths)
    length = 0
    for i in order:
        same = length > 0
        if same:
            for u in range(paths.shape[1]):
                same = same and paths[i, u] == out[length - 1, u]
        if not same:
            out[length] = paths[i]
            length += 1
    return out[:length].copy()


@kernel(overloads=overloads_for(__name__, "background_values"), cache=True, nogil=True)
def background_values(background_emissions, bg, penalty, start, end):
    """Best path staying in background states, per sample. O(N B Q)."""
    n, bins, nb = background_emissions.shape
    out = np.empty(n)
    background_state_values = np.empty(nb)
    for s in range(n):
        top = -np.inf
        for z in range(start.shape[1]):
            top = max(top, start[s, z])
        for j in range(nb):
            background_state_values[j] = max(start[s, bg[j]], top - penalty) + background_emissions[s, 0, j]
        for t in range(1, bins):
            top = -np.inf
            for j in range(nb):
                top = max(top, background_state_values[j])
            floor = top - penalty
            for j in range(nb):
                background_state_values[j] = max(background_state_values[j], floor) + background_emissions[s, t, j]
        best = -np.inf
        for j in range(nb):
            best = max(best, background_state_values[j] + end[s, bg[j]])
        out[s] = best
    return out


@kernel(overloads=overloads_for(__name__, "relevant_samples"), cache=True, nogil=True)
def relevant_samples(background_emissions, bg, penalty, start, end, upper0):
    """Keep samples whose relaxed bound exceeds their background value by the relative arithmetic margin. O(N B Q)."""
    n = start.shape[0]
    keep = np.zeros(n, np.bool_)
    if len(bg) == 0:
        keep[:] = True
        return np.flatnonzero(keep)
    background = background_values(background_emissions, bg, penalty, start, end)
    for s in range(n):
        relaxed = -np.inf
        for z in range(start.shape[1]):
            relaxed = max(relaxed, start[s, z] + upper0[s, z])
        keep[s] = relaxed > background[s] + RELEVANCE_MARGIN * max(1.0, abs(background[s]))
    return np.flatnonzero(keep)


@kernel(overloads=overloads_for(__name__, "take_samples"), cache=True, nogil=True)
def take_samples(focal_emissions, background_emissions, start, end, upper, index):
    """Gather relevant samples in their original order. O(N (B R K+B Q+W Q))."""
    sample_count = len(index)
    fe_s = np.empty((sample_count, focal_emissions.shape[1], focal_emissions.shape[2], focal_emissions.shape[3]))
    be_s = np.empty((sample_count, background_emissions.shape[1], background_emissions.shape[2]))
    start_s = np.empty((sample_count, start.shape[1]))
    end_s = np.empty((sample_count, end.shape[1]))
    upper_s = np.empty((upper.shape[0], sample_count, upper.shape[2]))
    for i in range(sample_count):
        s = index[i]
        fe_s[i] = focal_emissions[s]
        be_s[i] = background_emissions[s]
        start_s[i] = start[s]
        end_s[i] = end[s]
        for u in range(upper.shape[0]):
            upper_s[u, i] = upper[u, s]
    return fe_s, be_s, start_s, end_s, upper_s


@kernel(overloads=overloads_for(__name__, "best_starts"), cache=True, nogil=True)
def best_starts(paths, scores, count, incumbent, reverse, tol):
    """Select six distinct whole wishes; score ties within tol use before. O(SEED_STARTS P W)."""
    taken = np.zeros(len(paths), np.bool_)
    out = np.empty((min(count, len(paths)), paths.shape[1]), np.int64)
    for j in range(len(out)):
        top = -np.inf
        for i in range(len(paths)):
            if not taken[i]:
                top = max(top, scores[i])
        best = -1
        for i in range(len(paths)):
            if taken[i] or scores[i] < top - tol:
                continue
            if best < 0 or before(paths[i], paths[best], incumbent, reverse):
                best = i
        taken[best] = True
        out[j] = paths[best]
    return out


@kernel(overloads=overloads_for(__name__, "interval_scores"), cache=True, nogil=True)
def interval_scores(focal_emissions, unit, bins, seeds, incumbent, ctx, penalty):
    """Score every valid seed restriction [a,c), with changed endpoints. O(N (B K+B^2+W^2 B+P W (B K+B^2+W K))).
    Whole paths use (0,W)."""
    runs, head, tail, full, focal_start, focal_end = ctx
    n, k = focal_start.shape
    width = len(incumbent)
    total = focal_emissions.shape[1]
    count = len(seeds)
    scores = np.zeros((count, width, width + 1))
    valid = np.zeros((count, width, width + 1), np.bool_)
    for i in range(count):
        for a in range(width):
            if seeds[i, a] == incumbent[a]:
                continue
            for c in range(a + 1, width + 1):
                valid[i, a, c] = seeds[i, c - 1] != incumbent[c - 1]
    boundary = np.empty((width + 1, k))
    incumbent_maxima = np.empty(total)
    bin_maxima = np.empty(total)
    focal_values = np.empty(k)
    beta = np.empty((width + 1, k))
    gamma = np.full((width + 1, total), -np.inf)
    const = np.empty(width + 1)
    prefix = np.empty((width + 1, width + 1))
    beta_work = np.empty(k)
    gamma_work = np.empty(total)
    for s in range(n):
        # Forward pass of the incumbent: the focal values at every unit boundary and the maxima of every bin.
        focal_values[:] = focal_start[s]
        top = -np.inf
        for c in range(k):
            top = max(top, focal_values[c])
        first_top = top
        boundary[0] = focal_values
        for t in range(total):
            b = head[s, t]
            for i in range(1, t + 1):
                b = max(b, incumbent_maxima[i - 1] - penalty + runs[s, i, t])
            row = incumbent[unit[t]]
            ftop = -np.inf
            for c in range(k):
                focal_values[c] = max(focal_values[c], top - penalty) + focal_emissions[s, t, row, c]
                ftop = max(ftop, focal_values[c])
            top = max(b, ftop)
            incumbent_maxima[t] = top
            if t + 1 == bins[unit[t] + 1]:
                boundary[unit[t] + 1] = focal_values
        # Backward pass of the incumbent: the suffix covector at every unit boundary 1 .. W.
        beta_work[:] = focal_end[s]
        gamma_work[:] = -np.inf
        for j in range(total - 1):
            gamma_work[j] = -penalty + tail[s, j + 1]
        suffix_constant = full[s]
        beta[width] = beta_work
        gamma[width] = gamma_work
        const[width] = suffix_constant
        for t in range(total - 1, bins[1] - 1, -1):
            g = gamma_work[t]
            row = incumbent[unit[t]]
            reach = -np.inf
            for c in range(k):
                beta_work[c] = focal_emissions[s, t, row, c] + max(beta_work[c], g)
                reach = max(reach, beta_work[c])
            for j in range(t):
                gamma_work[j] = max(gamma_work[j], -penalty + runs[s, j + 1, t] + g)
            gamma_work[t - 1] = max(gamma_work[t - 1], -penalty + reach)
            suffix_constant = max(suffix_constant, head[s, t] + g)
            gamma_work[t] = -np.inf
            if t == bins[unit[t]]:
                u = unit[t]
                beta[u] = beta_work
                gamma[u] = gamma_work
                const[u] = suffix_constant
        for a in range(width):
            for c in range(a + 1, width + 1):
                best = -np.inf
                for j in range(bins[a]):
                    best = max(best, incumbent_maxima[j] + gamma[c, j])
                prefix[a, c] = best
        for i in range(count):
            for a in range(width):
                if seeds[i, a] == incumbent[a]:
                    continue
                focal_values[:] = boundary[a]
                top = first_top if a == 0 else incumbent_maxima[bins[a] - 1]
                for j in range(bins[a]):
                    bin_maxima[j] = incumbent_maxima[j]
                for u in range(a, width):
                    row = seeds[i, u]
                    for t in range(bins[u], bins[u + 1]):
                        b = head[s, t]
                        for j in range(1, t + 1):
                            b = max(b, bin_maxima[j - 1] - penalty + runs[s, j, t])
                        ftop = -np.inf
                        for c in range(k):
                            focal_values[c] = max(focal_values[c], top - penalty) + focal_emissions[s, t, row, c]
                            ftop = max(ftop, focal_values[c])
                        top = max(b, ftop)
                        bin_maxima[t] = top
                    c2 = u + 1
                    if not valid[i, a, c2]:
                        continue
                    value = max(const[c2], prefix[a, c2])
                    for c in range(k):
                        value = max(value, focal_values[c] + beta[c2, c])
                    for j in range(bins[a], bins[c2]):
                        value = max(value, bin_maxima[j] + gamma[c2, j])
                    scores[i, a, c2] += value
    return scores, valid


@kernel(overloads=overloads_for(__name__, "best_intervals"), cache=True, nogil=True)
def best_intervals(seeds, scores, valid, count, incumbent, reverse, tol):
    """Select six distinct restrictions, excluding whole seeds; ties use before. O(INTERVAL_SEED_STARTS P W^3)."""
    total, width = seeds.shape
    taken = np.zeros((total, width, width + 1), np.bool_)
    out = np.empty((count, width), np.int64)
    found = 0
    trial = np.empty(width, np.int64)
    best = np.empty(width, np.int64)
    while found < count:
        top = -np.inf
        for i in range(total):
            for a in range(width):
                for c in range(a + 1, width + 1):
                    if valid[i, a, c] and not taken[i, a, c] and not (a == 0 and c == width):
                        top = max(top, scores[i, a, c])
        if top == -np.inf:
            break
        chosen = False
        for i in range(total):
            for a in range(width):
                for c in range(a + 1, width + 1):
                    if not valid[i, a, c] or taken[i, a, c] or (a == 0 and c == width) or scores[i, a, c] < top - tol:
                        continue
                    trial[:] = incumbent
                    trial[a:c] = seeds[i, a:c]
                    if not chosen or before(trial, best, incumbent, reverse):
                        best[:] = trial
                        chosen = True
        out[found] = best
        found += 1
        for i in range(total):
            for a in range(width):
                for c in range(a + 1, width + 1):
                    if not valid[i, a, c] or taken[i, a, c]:
                        continue
                    same = True
                    for u in range(width):
                        value = seeds[i, u] if a <= u < c else incumbent[u]
                        same = same and value == best[u]
                    taken[i, a, c] = same
    return out[:found].copy()


@kernel(overloads=overloads_for(__name__, "unit_scores"), cache=True, nogil=True)
def unit_scores(focal_emissions, unit, bins, rows, cands, offsets, ctx, penalty, current):
    """Score every single-unit move using shared incumbent messages. O(N (B K+B^2+W B+R (B K+B^2))). Current rows
    use current."""
    runs, head, tail, full, focal_start, focal_end = ctx
    n, k = focal_start.shape
    width = len(rows)
    total = focal_emissions.shape[1]
    scores = np.zeros(len(cands))
    boundary = np.empty((width + 1, k))
    incumbent_maxima = np.empty(total)
    bin_maxima = np.empty(total)
    focal_values = np.empty(k)
    beta = np.empty((width + 1, k))
    gamma = np.full((width + 1, total), -np.inf)
    const = np.empty(width + 1)
    beta_work = np.empty(k)
    gamma_work = np.empty(total)
    for s in range(n):
        focal_values[:] = focal_start[s]
        top = -np.inf
        for c in range(k):
            top = max(top, focal_values[c])
        first_top = top
        boundary[0] = focal_values
        for t in range(total):
            b = head[s, t]
            for i in range(1, t + 1):
                b = max(b, incumbent_maxima[i - 1] - penalty + runs[s, i, t])
            row = rows[unit[t]]
            ftop = -np.inf
            for c in range(k):
                focal_values[c] = max(focal_values[c], top - penalty) + focal_emissions[s, t, row, c]
                ftop = max(ftop, focal_values[c])
            top = max(b, ftop)
            incumbent_maxima[t] = top
            if t + 1 == bins[unit[t] + 1]:
                boundary[unit[t] + 1] = focal_values
        beta_work[:] = focal_end[s]
        gamma_work[:] = -np.inf
        for j in range(total - 1):
            gamma_work[j] = -penalty + tail[s, j + 1]
        suffix_constant = full[s]
        beta[width] = beta_work
        gamma[width] = gamma_work
        const[width] = suffix_constant
        for t in range(total - 1, bins[1] - 1, -1):
            g = gamma_work[t]
            row = rows[unit[t]]
            reach = -np.inf
            for c in range(k):
                beta_work[c] = focal_emissions[s, t, row, c] + max(beta_work[c], g)
                reach = max(reach, beta_work[c])
            for j in range(t):
                gamma_work[j] = max(gamma_work[j], -penalty + runs[s, j + 1, t] + g)
            gamma_work[t - 1] = max(gamma_work[t - 1], -penalty + reach)
            suffix_constant = max(suffix_constant, head[s, t] + g)
            gamma_work[t] = -np.inf
            if t == bins[unit[t]]:
                u = unit[t]
                beta[u] = beta_work
                gamma[u] = gamma_work
                const[u] = suffix_constant
        for u in range(width):
            c2 = u + 1
            floor = const[c2]
            for j in range(bins[u]):
                bin_maxima[j] = incumbent_maxima[j]
                floor = max(floor, incumbent_maxima[j] + gamma[c2, j])
            entry = first_top if u == 0 else incumbent_maxima[bins[u] - 1]
            for i in range(offsets[u], offsets[u + 1]):
                row = cands[i]
                if row == rows[u]:
                    continue
                focal_values[:] = boundary[u]
                top = entry
                for t in range(bins[u], bins[c2]):
                    b = head[s, t]
                    for j in range(1, t + 1):
                        b = max(b, bin_maxima[j - 1] - penalty + runs[s, j, t])
                    ftop = -np.inf
                    for c in range(k):
                        focal_values[c] = max(focal_values[c], top - penalty) + focal_emissions[s, t, row, c]
                        ftop = max(ftop, focal_values[c])
                    top = max(b, ftop)
                    bin_maxima[t] = top
                value = floor
                for c in range(k):
                    value = max(value, focal_values[c] + beta[c2, c])
                for j in range(bins[u], bins[c2]):
                    value = max(value, bin_maxima[j] + gamma[c2, j])
                scores[i] += value
    for u in range(width):
        for i in range(offsets[u], offsets[u + 1]):
            if cands[i] == rows[u]:
                scores[i] = current
    return scores


@kernel(overloads=overloads_for(__name__, "ascend"), cache=True, nogil=True)
def ascend(focal_emissions, unit, bins, cands, offsets, seed, incumbent, ctx, penalty, tol, reverse):
    """Best improving single-unit ascent with exact rechecks. O(I (N (B K+B^2+W B+R (B K+B^2))+C W)). Returns rows,
    value, evaluations, steps, capped."""
    rows = seed.copy()
    h = value(focal_emissions, unit, rows, ctx, penalty)
    evaluations = 1
    for iteration in range(LOCAL_ITERATIONS):
        scores = unit_scores(focal_emissions, unit, bins, rows, cands, offsets, ctx, penalty, h)
        evaluations += len(cands) - len(rows)
        best = h
        for i in range(len(cands)):
            best = max(best, scores[i])
        if best <= h + tol:
            return rows, h, evaluations, iteration + 1, False
        # The first complete path within tol of the best improving value, independent of the enumeration order.
        chosen = rows.copy()
        selected = False
        trial = rows.copy()
        for u in range(len(rows)):
            for i in range(offsets[u], offsets[u + 1]):
                r = cands[i]
                if r == rows[u]:
                    continue
                candidate_value = scores[i]
                if candidate_value >= best - tol and candidate_value > h + tol:
                    trial[:] = rows
                    trial[u] = r
                    if not selected or before(trial, chosen, incumbent, reverse):
                        chosen[:] = trial
                        selected = True
        hn = value(focal_emissions, unit, chosen, ctx, penalty)
        evaluations += 1
        if hn <= h + tol:
            return rows, h, evaluations, iteration + 1, False
        rows[:] = chosen
        h = hn
    return rows, h, evaluations, LOCAL_ITERATIONS, True


@kernel(overloads=overloads_for(__name__, "search"), cache=True, nogil=True)
def search(
    focal_emissions,
    background_emissions,
    unit,
    bins,
    cands,
    offsets,
    incumbent,
    fz,
    bg,
    penalty,
    start,
    end,
    upper,
    tol,
    reverse,
):
    """Relevant wishes, restrictions and bounded single-unit ascent. O(N B^2 Q+N R (B K+B^2)+P^2 W+N P W (B K+B^2+W
    K)+INTERVAL_SEED_STARTS P W^3+L I (N (B K+B^2+W B+R (B K+B^2))+C W)). Stats: evaluations, relevant samples,
    seeds, restrictions, starts, ascent steps, caps."""
    stats = np.zeros(7, np.int64)
    width = len(incumbent)
    index = relevant_samples(background_emissions, bg, penalty, start, end, upper[0])
    stats[1] = len(index)
    if len(index) == 0:
        return incumbent.copy(), 0.0, stats
    focal_emissions, background_emissions, start, end, upper = take_samples(
        focal_emissions, background_emissions, start, end, upper, index
    )
    ctx = context(background_emissions, start, end, fz, bg, penalty)
    initial = value(focal_emissions, unit, incumbent, ctx, penalty)
    stats[0] += 1
    wish = wishes(
        focal_emissions, background_emissions, unit, bins, cands, offsets, incumbent, fz, bg, penalty, start, end, upper
    )
    raw = np.empty((len(wish) + 1, width), np.int64)
    raw[0] = incumbent
    raw[1:] = wish
    seeds = unique(raw, len(raw), incumbent, reverse)
    stats[2] = len(seeds)
    scores, valid = interval_scores(focal_emissions, unit, bins, seeds, incumbent, ctx, penalty)
    whole_scores = np.empty(len(seeds))
    for i in range(len(seeds)):
        # A seed equal to the incumbent at its first or last unit is one of its own restrictions; score it there.
        whole_scores[i] = initial
        a = 0
        while a < width and seeds[i, a] == incumbent[a]:
            a += 1
        c = width
        while c > a and seeds[i, c - 1] == incumbent[c - 1]:
            c -= 1
        if c > a:
            whole_scores[i] = scores[i, a, c]
        for x in range(width):
            for y in range(x + 1, width + 1):
                stats[3] += int(valid[i, x, y])
    stats[0] += stats[3]
    whole = best_starts(seeds, whole_scores, SEED_STARTS, incumbent, reverse, tol)
    intervals = best_intervals(seeds, scores, valid, INTERVAL_SEED_STARTS, incumbent, reverse, tol)
    raw = np.empty((1 + len(whole) + len(intervals), width), np.int64)
    raw[0] = incumbent
    raw[1 : 1 + len(whole)] = whole
    raw[1 + len(whole) :] = intervals
    starts = unique(raw, len(raw), incumbent, reverse)
    stats[4] = len(starts)
    finishes = np.empty_like(starts)
    finish_values = np.empty(len(starts))
    top = initial
    for i in range(len(starts)):
        rows, h, evaluations, steps, capped = ascend(
            focal_emissions, unit, bins, cands, offsets, starts[i], incumbent, ctx, penalty, tol, reverse
        )
        finishes[i] = rows
        finish_values[i] = h
        stats[0] += evaluations
        stats[5] += steps
        stats[6] += int(capped)
        top = max(top, h)
    best = incumbent.copy()
    best_value = initial
    if initial < top - tol:
        selected = False
        for i in range(len(finishes)):
            h = finish_values[i]
            if h >= top - tol and h > initial + tol:
                if not selected or before(finishes[i], best, incumbent, reverse):
                    best[:] = finishes[i]
                    best_value = h
                    selected = True
    return best, best_value - initial, stats
