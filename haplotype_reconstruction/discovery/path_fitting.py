"""Cached-trajectory fixed-panel fitter for the normalized diploid copying model.

Prepare read-only emission/log-ratio tables once, retain predicted priors plus
filtered marginals after a score, and reuse them in the backward pass. Before
a frequency update, compute only the necessary frequency statistics. State
trajectories store only the upper triangle; homologue-exchange symmetry avoids
duplicate storage and work without changing the model or enabling fastmath.

Samples are processed in lane blocks of _LANE_WIDTH: data tables and
trajectories hold the samples of a block side by side, and every inner loop
runs over the lanes of one block, so the compiler vectorizes the per-sample
recursions across samples. Each sample's floating-point operations, operands
and their order do not depend on its lane, block or the thread count, and
totals over samples add the samples one by one in ascending order, so every
result is the same for any thread count.

Workspace memory is O(N*L*S**2), where S=K+1 includes the unknown state;
predicted priors use S*(S+1)/2 entries per sample/marker, not S*S. The
cached trajectories avoid repeated forward/prediction work within a fit.
One workspace belongs to one fit, not a shared/global cache. Dynamic threads
remain caller-controlled.
"""
import numpy as np
from numba import njit, prange, get_num_threads
import math
from . import path_model as reference
from .objectives import log_binary_haplotype_set_count

# Samples per lane block. A work split only: results do not depend on it.
# Wide blocks amortize the per-loop overhead of the lane loops (16 lanes gave
# 1.1x over one sample at a time, 32 1.5-1.9x, 64 2.0-2.2x, 128 no more).
_LANE_WIDTH = 64


def canonical(panel, frequencies=None):
    """Sort exact binary rows and sum their frequency mass when deduplicating."""
    panel = np.ascontiguousarray(panel, dtype=np.int64)
    unique, inverse = np.unique(panel, axis=0, return_inverse=True)
    if frequencies is None:
        frequencies = np.full(len(panel), 1. / len(panel))
    mass = np.bincount(inverse, weights=frequencies, minlength=len(unique))
    return unique, mass / mass.sum()


def regularizer(panel, frequencies, learn_frequencies, free=None, claimed_code=0.):
    """Dictionary-size penalty plus uniform-centered frequency shrinkage.

    The dictionary term codes the ``free`` rows (default: all K) as new
    arbitrary binary haplotypes, log C(2^L, free). A row that is a context
    founder's local sequence is coded by reference to that founder plus its
    edits, ``claimed_code`` nats in total (path_selection.ContextFounders).
    This is a regularized best-panel objective, not a Bayesian posterior over K.
    """
    k, length = panel.shape
    free = k if free is None else int(free)
    dictionary = float(log_binary_haplotype_set_count(length, free)) if free else 0.
    value = -dictionary - float(claimed_code) - math.log(k) - math.log(k + 1)
    if learn_frequencies:
        # Explicit shrinkage penalty centered to zero at uniform. Its MAP
        # extra counts total one; avoid a dimension-dependent density reward.
        value += float(np.log(k * frequencies).sum() / k)
    return value


@njit(cache=True)
def _table(gl, observed, emission):
    if not observed:
        emission[:, :] = 1.
        return
    emission[0, 0] = gl[0]
    emission[0, 1] = emission[1, 0] = gl[1]
    emission[1, 1] = gl[2]
    emission[0, 2] = emission[2, 0] = .5 * (gl[0] + gl[1])
    emission[1, 2] = emission[2, 1] = .5 * (gl[1] + gl[2])
    emission[2, 2] = .25 * gl[0] + .5 * gl[1] + .25 * gl[2]


@njit(cache=True)
def _prepare_emissions(gl, observed):
    n, length = observed.shape
    emissions = np.empty((n, length, 3, 3))

    for sample in range(n):
        for site in range(length):
            table = emissions[sample, site]
            _table(gl[sample, site], observed[sample, site], table)
    ratios, self_ratios = _prepare_ratios(emissions, observed)
    return emissions, ratios, self_ratios


@njit(cache=True)
def _prepare_ratios(emissions, observed):
    n, length = observed.shape
    ratios = np.zeros((n, length, 2, 3))
    self_ratios = np.zeros((n, length, 2))
    for sample in range(n):
        for site in range(length):
            table = emissions[sample, site]
            if observed[sample, site]:
                for h in range(2):
                    for partner in range(3):
                        if table[h, partner] > 0:
                            ratios[sample, site, h, partner] = reference._log_ratio(
                                table[1-h, partner], table[h, partner])
                    if table[h, h] > 0:
                        self_ratios[sample, site, h] = reference._log_ratio(
                            table[1-h, 1-h], table[h, h])
    return ratios, self_ratios


# ---------------------------------------------------------------------------
# Sample-lane kernels. Layouts: et[block, site, ha, hb, lane] emissions,
# rt[block, site, h, partner, lane] and st[block, site, h, lane] Q log ratios,
# ot[block, site, lane] observed; trajectories pt[block, site, pair, lane]
# (predicted priors, upper-triangle pair order) and wt[block, site, state,
# lane] (filtered row marginals). The lane loops vectorize because their trip
# counts are runtime values, no loop updates two rows of one array, no array
# views are made inside loops, and error_model='numpy' removes Python's
# ZeroDivisionError branch (an early exit). Only finite-likelihood samples are
# read; their divisors are positive except a backward normaliser that
# underflows to zero, which _Workspace.state reports as ZeroDivisionError.
# Logarithms stay scalar libm calls.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _emission_lanes(emissions, width):
    """Lane layout et of a block's emissions; padding lanes are never read."""
    n, length = emissions.shape[:2]
    blocks = (n + width - 1) // width
    et = np.ones((blocks, length, 3, 3, width))
    for sample in range(n):
        block, lane = sample // width, sample % width
        for site in range(length):
            for a in range(3):
                for b in range(3):
                    et[block, site, a, b, lane] = emissions[sample, site, a, b]
    return et


@njit(cache=True)
def _ratio_lanes(observed, ratios, self_ratios, width):
    """Lane layouts rt, st, ot of a block's Q log ratios and observed mask."""
    n, length = observed.shape
    blocks = (n + width - 1) // width
    rt = np.zeros((blocks, length, 2, 3, width))
    st = np.zeros((blocks, length, 2, width))
    ot = np.zeros((blocks, length, width), dtype=np.bool_)
    for sample in range(n):
        block, lane = sample // width, sample % width
        for site in range(length):
            for h in range(2):
                st[block, site, h, lane] = self_ratios[sample, site, h]
                for p in range(3):
                    rt[block, site, h, p, lane] = ratios[sample, site, h, p]
            ot[block, site, lane] = observed[sample, site]
    return rt, st, ot


@njit(cache=True, inline='always')
def _pair(a, b, s):
    """Upper-triangle pair index of the states (min(a, b), max(a, b))."""
    if a > b:
        a, b = b, a
    return a * s - a * (a - 1) // 2 + (b - a)


@njit(cache=True, error_model='numpy')
def _forward_block(block, lanes, panel, et, pi, rates, pt, wt, slot, filtered, scale, ll, alive):
    """Scaled forward pass of the first ``lanes`` samples of one lane block.

    Writes the triangular predicted priors P_t to pt[slot] and the common
    row/column marginal of the filtered F_t to wt[slot], site t at index
    t % (their site length): a fit keeps every site, a score keeps one prior
    and the two latest marginals (the recursion reads only the previous one).
    Each stored off-diagonal entry represents one ordered state, not their
    combined mass; normalizers count it twice. The two homologues share
    priors and unphased emissions, so their messages remain symmetric. A
    sample stops adding to its likelihood at its first non-positive scale
    (likelihood -inf); the E-step refuses such a fit, so the rest of that
    lane's trajectory is never read.
    """
    length, s = et.shape[1], len(pi)
    k = s - 1
    pairs = s * (s + 1) // 2
    prior_period, row_period = pt.shape[1], wt.shape[1]
    for j in range(lanes):
        ll[j] = 0.
        alive[j] = True
    for site in range(length):
        ps, ws, wp = site % prior_period, site % row_period, (site - 1) % row_period
        rate = rates[site - 1] if site else 0.
        stay = 1. - rate
        # Hoisted products keep their operands and order: stay*stay*x is
        # (stay*stay)*x and rate*rate*pi[a]*pi[b] is ((rate*rate)*pi[a])*pi[b].
        ss, sr, rr = stay * stay, stay * rate, rate * rate
        for j in range(lanes):
            scale[j] = 0.
        index = 0
        for a in range(s):
            ha = panel[a, site] if a < k else 2
            for b in range(a, s):
                hb = panel[b, site] if b < k else 2
                if site:
                    pa, pb = pi[a], pi[b]
                    c3 = rr * pa * pb
                    if a == b:
                        for j in range(lanes):
                            predicted = (ss * filtered[index, j]
                                         + sr * (wt[slot, wp, a, j] * pb + pa * wt[slot, wp, b, j]) + c3)
                            pt[slot, ps, index, j] = predicted
                            value = predicted * et[block, site, ha, hb, j]
                            filtered[index, j] = value
                            scale[j] += value
                    else:
                        for j in range(lanes):
                            predicted = (ss * filtered[index, j]
                                         + sr * (wt[slot, wp, a, j] * pb + pa * wt[slot, wp, b, j]) + c3)
                            pt[slot, ps, index, j] = predicted
                            value = predicted * et[block, site, ha, hb, j]
                            filtered[index, j] = value
                            scale[j] += 2. * value
                else:
                    predicted = pi[a] * pi[b]
                    if a == b:
                        for j in range(lanes):
                            pt[slot, ps, index, j] = predicted
                            value = predicted * et[block, site, ha, hb, j]
                            filtered[index, j] = value
                            scale[j] += value
                    else:
                        for j in range(lanes):
                            pt[slot, ps, index, j] = predicted
                            value = predicted * et[block, site, ha, hb, j]
                            filtered[index, j] = value
                            scale[j] += 2. * value
                index += 1
        for j in range(lanes):
            if alive[j]:
                if scale[j] <= 0:
                    ll[j] = -np.inf
                    alive[j] = False
                else:
                    ll[j] += np.log(scale[j])
        for index in range(pairs):
            for j in range(lanes):
                filtered[index, j] = filtered[index, j] / scale[j]
        # A row marginal adds its pairs in the order (0, r), ..., (r - 1, r),
        # (r, r), ..., (r, s - 1), as a sweep over the upper triangle does.
        for r in range(s):
            for j in range(lanes):
                wt[slot, ws, r, j] = 0.
            for a in range(r):
                p = _pair(a, r, s)
                for j in range(lanes):
                    wt[slot, ws, r, j] += filtered[p, j]
            for b in range(r, s):
                p = _pair(r, b, s)
                for j in range(lanes):
                    wt[slot, ws, r, j] += filtered[p, j]


@njit(cache=True, parallel=True, error_model='numpy')
def _forward_lanes(panel, et, pi, rates, pt, wt, n, workers):
    """Per-sample log likelihoods; caches the trajectories pt and wt."""
    blocks, width = et.shape[0], et.shape[4]
    s = len(pi)
    pairs = s * (s + 1) // 2
    likelihoods = np.empty(n)
    for wi in prange(workers):
        worker = np.int64(wi)
        filtered = np.empty((pairs, width))
        scale, ll = np.empty(width), np.empty(width)
        alive = np.empty(width, dtype=np.bool_)
        for block in range(worker, blocks, workers):
            lanes = min(width, n - block * width)
            _forward_block(block, lanes, panel, et, pi, rates, pt, wt, block, filtered, scale, ll, alive)
            for j in range(lanes):
                likelihoods[block * width + j] = ll[j]
    return likelihoods


@njit(cache=True, parallel=True, error_model='numpy')
def _score_lanes(panel, et, pi, rates, n, workers):
    """Per-sample log likelihoods without trajectories: the same forward
    recursion, each worker keeping one prior and two row marginals."""
    blocks, width = et.shape[0], et.shape[4]
    s = len(pi)
    pairs = s * (s + 1) // 2
    likelihoods = np.empty(n)
    for wi in prange(workers):
        worker = np.int64(wi)
        filtered = np.empty((pairs, width))
        scale, ll = np.empty(width), np.empty(width)
        alive = np.empty(width, dtype=np.bool_)
        pt, wt = np.empty((1, 1, pairs, width)), np.empty((1, 2, s, width))
        for block in range(worker, blocks, workers):
            lanes = min(width, n - block * width)
            _forward_block(block, lanes, panel, et, pi, rates, pt, wt, 0, filtered, scale, ll, alive)
            for j in range(lanes):
                likelihoods[block * width + j] = ll[j]
    return likelihoods


@njit(cache=True, error_model='numpy')
def _backward_block(block, lanes, panel, et, rt, st, ot, pi, rates, pt, wt, full_statistics, direct,
                    gains, occupancy, lane_gains, lane_occupancy, initial, destinations, work):
    """Backward pass of the first ``lanes`` samples of one lane block over
    their cached forward trajectories.

    Both homologues have the same copying prior/transition and genotype
    emissions are unordered, so all messages are symmetric even when the
    founder frequencies are unequal: each unordered cell is evaluated once,
    and the backward messages, weights, masses, posteriors and futures are
    kept as upper triangles. A sample's flip gains (observed sites only) and
    copy occupancy go, with ``direct``, into the totals ``gains`` and
    ``occupancy`` in lane (= sample) order; otherwise into
    ``lane_gains[2, k, L, lane]`` and ``lane_occupancy[s, L, lane]`` (+0 where
    a sample adds nothing) for an ordered combine. ``initial[a, lane]`` and
    ``destinations[a, lane]`` receive each sample's initial copies and its
    redraw destinations summed over sites. ``work`` holds the scratch arrays.
    """
    (backward, weights, mass, posterior, future, row, redraw, previous, copies, outside, qgain, alt,
     vec, prefix, suffix, tail, quotient, branch, log_old, have_log_old) = work
    length, k = et.shape[1], len(panel)
    s = k + 1
    pairs = s * (s + 1) // 2
    for a in range(s):
        for j in range(lanes):
            destinations[a, j] = 0.
    for index in range(pairs):
        for j in range(lanes):
            backward[index, j] = 1.
    # Defined before the site loop; every use at site > 0 follows the
    # assignment for that site.
    rate, stay = 0., 1.
    for site in range(length - 1, -1, -1):
        # vec rows: 0 denominator, 1 total, 2 maximum, 3 left, 4 right.
        for j in range(lanes):
            vec[0, j] = 0.
        if site:
            rate = rates[site - 1]
            stay = 1. - rate
            for b in range(s):
                c = rate * pi[b]
                for j in range(lanes):
                    previous[b, j] = stay * wt[block, site - 1, b, j] + c
        store_mass = full_statistics or site == 0
        index = 0
        for a in range(s):
            ha = panel[a, site] if a < k else 2
            for b in range(a, s):
                hb = panel[b, site] if b < k else 2
                twice = 1. if a == b else 2.
                if full_statistics:
                    for j in range(lanes):
                        weight = pt[block, site, index, j] * backward[index, j]
                        value = weight * et[block, site, ha, hb, j]
                        future[index, j] = et[block, site, ha, hb, j] * backward[index, j]
                        weights[index, j] = weight
                        mass[index, j] = value
                        vec[0, j] += value if a == b else twice * value
                elif store_mass:
                    for j in range(lanes):
                        value = pt[block, site, index, j] * backward[index, j] * et[block, site, ha, hb, j]
                        future[index, j] = et[block, site, ha, hb, j] * backward[index, j]
                        mass[index, j] = value
                        vec[0, j] += value if a == b else twice * value
                else:
                    for j in range(lanes):
                        value = pt[block, site, index, j] * backward[index, j] * et[block, site, ha, hb, j]
                        future[index, j] = et[block, site, ha, hb, j] * backward[index, j]
                        vec[0, j] += value if a == b else twice * value
                index += 1
        if site:
            # row[r] = sum future * pi[other] and redraw[r] = sum future *
            # previous[other] over the pairs holding r, in sweep order.
            for r in range(s):
                for j in range(lanes):
                    row[r, j] = 0.
                    redraw[r, j] = 0.
                for a in range(r):
                    p, pa = _pair(a, r, s), pi[a]
                    for j in range(lanes):
                        row[r, j] += future[p, j] * pa
                        redraw[r, j] += future[p, j] * previous[a, j]
                for b in range(r, s):
                    p, pb = _pair(r, b, s), pi[b]
                    for j in range(lanes):
                        row[r, j] += future[p, j] * pb
                        redraw[r, j] += future[p, j] * previous[b, j]
        if store_mass:
            for index in range(pairs):
                for j in range(lanes):
                    posterior[index, j] = mass[index, j] / vec[0, j]
            for r in range(s):
                for j in range(lanes):
                    copies[r, j] = 0.
                for a in range(r):
                    p = _pair(a, r, s)
                    for j in range(lanes):
                        copies[r, j] += 2. * posterior[p, j]
                for b in range(r, s):
                    p = _pair(r, b, s)
                    for j in range(lanes):
                        copies[r, j] += 2. * posterior[p, j]
            for a in range(s):
                if full_statistics:
                    if direct:
                        total = occupancy[a, site]
                        for j in range(lanes):
                            total += copies[a, j]
                        occupancy[a, site] = total
                    else:
                        for j in range(lanes):
                            lane_occupancy[a, site, j] = copies[a, j]
                if site == 0:
                    for j in range(lanes):
                        initial[a, j] = copies[a, j]
        any_observed = False
        for j in range(lanes):
            any_observed |= ot[block, site, j]
        if full_statistics and not direct:
            for a in range(k):
                for j in range(lanes):
                    lane_gains[0, a, site, j] = 0.
                    lane_gains[1, a, site, j] = 0.
        if full_statistics and any_observed:
            for j in range(lanes):
                have_log_old[j] = False
            # Carrier-excluded mass outside[a] from positive upper-triangle
            # prefix/suffix sums, which avoid subtracting a nearly complete
            # carrier mass from the total. Symmetry counts each edge twice.
            for j in range(lanes):
                tail[s, j] = 0.
            for a in range(s - 1, -1, -1):
                p = _pair(a, a, s)
                for j in range(lanes):
                    vec[3, j] = mass[p, j]
                for b in range(a + 1, s):
                    p = _pair(a, b, s)
                    for j in range(lanes):
                        prefix[a, b, j] = vec[3, j]
                        vec[3, j] += 2. * mass[p, j]
                for j in range(lanes):
                    prefix[a, s, j] = vec[3, j]
                    vec[4, j] = 0.
                    suffix[a, s, j] = 0.
                for b in range(s - 1, a, -1):
                    p = _pair(a, b, s)
                    for j in range(lanes):
                        vec[4, j] += 2. * mass[p, j]
                        suffix[a, b, j] = vec[4, j]
                for j in range(lanes):
                    tail[a, j] = tail[a + 1, j] + vec[3, j]
            for a in range(s):
                for j in range(lanes):
                    outside[a, j] = tail[a + 1, j]
                for b in range(a):
                    for j in range(lanes):
                        outside[a, j] += prefix[b, a, j] + suffix[b, a + 1, j]
            for a in range(k):
                ha = panel[a, site]
                for j in range(lanes):
                    qgain[a, j] = 0.
                    alt[a, j] = 0.
                for b in range(s):
                    hb = panel[b, site] if b < k else 2
                    p = _pair(a, b, s)
                    if a == b:
                        for j in range(lanes):
                            alt[a, j] += weights[p, j] * et[block, site, 1 - ha, 1 - ha, j]
                        for j in range(lanes):
                            prob, q = posterior[p, j], qgain[a, j]
                            qgain[a, j] = q + prob * st[block, site, ha, j] if prob > 0 else q
                    else:
                        for j in range(lanes):
                            weight = 2. * weights[p, j]
                            alt[a, j] += weight * et[block, site, 1 - ha, hb, j]
                        for j in range(lanes):
                            prob, q = 2. * posterior[p, j], qgain[a, j]
                            qgain[a, j] = q + prob * rt[block, site, ha, hb, j] if prob > 0 else q
                # The exact flip gain path_model._log_ratio(outside + alt,
                # denominator): its near-branch quotient for every lane, and
                # log(denominator) at most once per lane and site (the same
                # libm value whenever it is evaluated).
                for j in range(lanes):
                    new = outside[a, j] + alt[a, j]
                    old = vec[0, j]
                    alt[a, j] = new
                    difference = new - old
                    quotient[j] = difference / old
                    branch[j] = 0 if new == 0 else (1 if abs(difference) <= .5 * old else 2)
                if direct:
                    g0, g1 = gains[0, a, site], gains[1, a, site]
                for j in range(lanes):
                    if ot[block, site, j]:
                        kind = branch[j]
                        if kind == 0:
                            ratio = -np.inf
                        elif kind == 1:
                            ratio = np.log1p(quotient[j])
                        else:
                            if not have_log_old[j]:
                                log_old[j] = np.log(vec[0, j])
                                have_log_old[j] = True
                            ratio = np.log(alt[a, j]) - log_old[j]
                        if direct:
                            g0 += qgain[a, j]
                            g1 += ratio
                        else:
                            lane_gains[0, a, site, j] = qgain[a, j]
                            lane_gains[1, a, site, j] = ratio
                if direct:
                    gains[0, a, site], gains[1, a, site] = g0, g1
        if site > 0:
            for j in range(lanes):
                vec[1, j] = 0.
            for a in range(s):
                c, pa = 2. * rate * pi[a], pi[a]
                for j in range(lanes):
                    destinations[a, j] += c * redraw[a, j] / vec[0, j]
                    vec[1, j] += pa * row[a, j]
            ss, sr, rr = stay * stay, stay * rate, rate * rate
            for j in range(lanes):
                vec[2, j] = 0.
            index = 0
            for a in range(s):
                for b in range(a, s):
                    for j in range(lanes):
                        value = ss * future[index, j] + sr * (row[a, j] + row[b, j]) + rr * vec[1, j]
                        backward[index, j] = value
                        # numba's max(largest, value), a select.
                        largest = vec[2, j]
                        vec[2, j] = value if value > largest else largest
                    index += 1
            for index in range(pairs):
                for j in range(lanes):
                    largest = vec[2, j]
                    backward[index, j] = backward[index, j] / largest if largest > 0 else backward[index, j]


@njit(cache=True)
def _backward_work(s, width):
    """Scratch arrays of one _backward_block call: O(S**2 * width)."""
    pairs = s * (s + 1) // 2
    return (np.empty((pairs, width)), np.empty((pairs, width)), np.empty((pairs, width)),
            np.empty((pairs, width)), np.empty((pairs, width)),
            np.empty((s, width)), np.empty((s, width)), np.empty((s, width)), np.empty((s, width)),
            np.empty((s, width)), np.empty((s, width)), np.empty((s, width)),
            np.empty((5, width)), np.empty((s, s + 1, width)), np.empty((s, s + 1, width)),
            np.empty((s + 1, width)), np.empty(width), np.empty(width, dtype=np.int64),
            np.empty(width), np.empty(width, dtype=np.bool_))


@njit(cache=True, parallel=True, error_model='numpy')
def _backward_lanes(panel, et, rt, st, ot, pi, rates, pt, wt, full_statistics, n, workers):
    """Sufficient statistics summed over samples in ascending sample order.

    With one worker the blocks run in order and add their lanes to the
    totals directly, one after another. Otherwise waves of ``workers`` blocks
    run in parallel and store per-lane statistics, and every total element
    then adds the wave's lanes one by one in sample order. Every sum is
    therefore the same sequential chain over samples 0..N-1 for any thread
    count: a sample's +0 at an unobserved site adds nothing, as the totals
    start at +0 and so never hold -0.
    """
    blocks, length, width = et.shape[0], et.shape[1], et.shape[4]
    k = len(panel)
    s = k + 1
    gains = np.zeros((2, k, length))
    occupancy = np.zeros((s, length))
    initial = np.zeros(s)
    destinations = np.zeros(s)
    if workers <= 1:
        work = _backward_work(s, width)
        lane_initial, lane_destinations = np.empty((s, width)), np.empty((s, width))
        no_gains, no_occupancy = np.empty((2, k, 0, 0)), np.empty((s, 0, 0))
        for block in range(blocks):
            lanes = min(width, n - block * width)
            _backward_block(block, lanes, panel, et, rt, st, ot, pi, rates, pt, wt, full_statistics, True,
                            gains, occupancy, no_gains, no_occupancy, lane_initial, lane_destinations, work)
            for j in range(lanes):
                for a in range(s):
                    initial[a] += lane_initial[a, j]
                    destinations[a] += lane_destinations[a, j]
        return gains, occupancy, initial, destinations
    wave = min(workers, blocks)
    lane_gains = np.empty((wave, 2, k, length if full_statistics else 0, width))
    lane_occupancy = np.empty((wave, s, length if full_statistics else 0, width))
    lane_initial = np.empty((wave, s, width))
    lane_destinations = np.empty((wave, s, width))
    for first in range(0, blocks, wave):
        count = min(wave, blocks - first)
        for slot_index in prange(count):
            slot = np.int64(slot_index)
            block = first + slot
            lanes = min(width, n - block * width)
            work = _backward_work(s, width)
            _backward_block(block, lanes, panel, et, rt, st, ot, pi, rates, pt, wt, full_statistics, False,
                            gains, occupancy, lane_gains[slot], lane_occupancy[slot], lane_initial[slot],
                            lane_destinations[slot], work)
        # Fixed-order combine: each element adds this wave's lanes in sample
        # order; the founder rows only distribute elements over threads.
        if full_statistics:
            for a_index in prange(k):
                a = np.int64(a_index)
                for site in range(length):
                    g0, g1 = gains[0, a, site], gains[1, a, site]
                    for slot in range(count):
                        lanes = min(width, n - (first + slot) * width)
                        for j in range(lanes):
                            g0 += lane_gains[slot, 0, a, site, j]
                            g1 += lane_gains[slot, 1, a, site, j]
                    gains[0, a, site], gains[1, a, site] = g0, g1
            for a_index in prange(s):
                a = np.int64(a_index)
                for site in range(length):
                    total = occupancy[a, site]
                    for slot in range(count):
                        lanes = min(width, n - (first + slot) * width)
                        for j in range(lanes):
                            total += lane_occupancy[slot, a, site, j]
                    occupancy[a, site] = total
        for slot in range(count):
            lanes = min(width, n - (first + slot) * width)
            for j in range(lanes):
                for a in range(s):
                    initial[a] += lane_initial[slot, a, j]
                    destinations[a] += lane_destinations[slot, a, j]
    return gains, occupancy, initial, destinations


def prepare_fit_observations(prepared):
    """Extend a forward-scorer block preparation with data-only Q log ratios."""
    if 'fit_ratios' in prepared and 'fit_self_ratios' in prepared:
        return prepared
    ratios, self_ratios = _prepare_ratios(prepared['emissions'], prepared['observed'])
    return dict(prepared, fit_ratios=ratios, fit_self_ratios=self_ratios)


def lane_emissions(prepared):
    """Lane layout of a prepared block's emissions, built on first use and
    kept in ``prepared`` (a block search's scores and fits share it)."""
    et = prepared.get('lane_emissions')
    if et is None:
        et = prepared['lane_emissions'] = _emission_lanes(prepared['emissions'], _LANE_WIDTH)
    return et


def _fit_lanes(prepared):
    """Lane layouts (et, rt, st, ot) of a prepared block's tables, built on
    first use and kept in ``prepared``."""
    lanes = prepared.get('fit_lanes')
    if lanes is None:
        lanes = prepared['fit_lanes'] = _ratio_lanes(prepared['observed'], prepared['fit_ratios'],
                                                      prepared['fit_self_ratios'], _LANE_WIDTH)
    return (lane_emissions(prepared), *lanes)


def score_lanes(panel, pi, prepared):
    """Per-sample forward log likelihoods of a trusted panel on a prepared block."""
    et = lane_emissions(prepared)
    return _score_lanes(panel, et, pi, prepared['rates'], len(prepared['emissions']),
                        min(len(et), get_num_threads()))


class _Workspace:
    """One fixed-K fit owns data tables and the latest parameter trajectory."""

    def __init__(self, panel, gl, positions, observed, generations=3.,
                 recombination_rate_per_bp=5e-8, wildcard_mass=.01,
                 interval_morgans=None, founder_frequencies=None, prepared=None):
        if prepared is None:
            panel, gl, observed, pi, self.rates = reference._prepare(
                panel, gl, positions, observed, generations, recombination_rate_per_bp,
                wildcard_mass, interval_morgans, founder_frequencies)
            emissions, ratios, self_ratios = _prepare_emissions(gl, observed)
            self.lanes = (_emission_lanes(emissions, _LANE_WIDTH),
                          *_ratio_lanes(observed, ratios, self_ratios, _LANE_WIDTH))
        else:
            # The outer scorer validated these observations once for this exact
            # block. Read-only data tables can be shared, trajectories cannot.
            self.rates = prepared['rates']
            assert prepared['wildcard_mass'] == wildcard_mass
            assert gl.shape[:2] == prepared['observed'].shape
            assert np.array_equal(positions, prepared['positions'])
            intervals = (np.diff(np.asarray(positions, dtype=np.float64)) * recombination_rate_per_bp
                         if interval_morgans is None else np.asarray(interval_morgans))
            assert np.array_equal(self.rates, -np.expm1(-generations * intervals))
            self.lanes = _fit_lanes(prepare_fit_observations(prepared))
        self.n, length, s = len(gl), gl.shape[1], len(panel)+1
        blocks, width = self.lanes[0].shape[0], self.lanes[0].shape[4]
        self.priors = np.empty((blocks, length, s * (s + 1) // 2, width))
        self.rows = np.empty((blocks, length, s, width))
        self.wildcard_mass = wildcard_mass
        self.panel = self.frequencies = self.state_cache = None
        self.state_is_full = False

    def _workers(self):
        """Lane blocks run in parallel; at most one thread per block."""
        return min(len(self.lanes[0]), get_num_threads())

    def score(self, panel, frequencies):
        frequencies = np.asarray(frequencies, dtype=np.float64)
        if (self.panel is not None and np.array_equal(panel, self.panel)
                and np.array_equal(frequencies, self.frequencies)):
            return float(self.likelihoods.sum())
        pi = np.r_[(1-self.wildcard_mass) * (frequencies / frequencies.sum()), self.wildcard_mass]
        values = _forward_lanes(panel, self.lanes[0], pi, self.rates, self.priors, self.rows,
                                self.n, self._workers())
        self.panel, self.frequencies = panel.copy(), frequencies.copy()
        self.pi, self.likelihoods = pi, values
        self.state_cache, self.state_is_full = None, False
        return float(values.sum())

    def state(self, panel, frequencies, full_statistics=True):
        if (self.panel is None or not np.array_equal(panel, self.panel)
                or not np.array_equal(frequencies, self.frequencies)):
            self.score(panel, frequencies)
        if np.any(~np.isfinite(self.likelihoods)):
            raise ValueError('E-step undefined for a sample with zero panel likelihood')
        if self.state_cache is not None and (self.state_is_full or not full_statistics):
            return self.state_cache
        et, rt, st, ot = self.lanes
        gains, occ, initial, dest = _backward_lanes(panel, et, rt, st, ot, self.pi, self.rates,
            self.priors, self.rows, full_statistics, self.n, self._workers())
        # Every site divides by its posterior normaliser (posteriors at site 0
        # or with full statistics, redraw destinations at the other sites). A
        # zero normaliser of a finite-likelihood sample (underflow of its
        # forward-backward overlap) leaves NaN or +inf in that sample's initial
        # or destination counts, hence in their nonnegative ascending totals,
        # which are finite otherwise: fail as a per-sample division would.
        if not (np.isfinite(initial).all() and np.isfinite(dest).all()):
            raise ZeroDivisionError('E-step: zero posterior normaliser at a site of a sample '
                                    'with a finite likelihood')
        self.state_cache = dict(log_likelihood=float(self.likelihoods.sum()),
            sample_log_likelihood=self.likelihoods, state_prior=self.pi,
            redraw_probabilities=self.rates, flip_gains=gains[0],
            conditional_flip_gains=gains[1], occupancy=occ,
            initial_copy_counts=initial, redraw_destination_counts=dest)
        self.state_is_full = full_statistics
        return self.state_cache


def fit_e_step(panel, gl, positions, observed_mask, *, generations=3.,
               recombination_rate_per_bp=5e-8, wildcard_mass=.01,
               interval_morgans=None, founder_frequencies=None):
    """Fit sufficient statistics only; use path_model.e_step for release diagnostics."""
    workspace = _Workspace(panel, gl, positions, observed_mask, generations,
        recombination_rate_per_bp, wildcard_mass, interval_morgans, founder_frequencies)
    panel = np.ascontiguousarray(panel, dtype=np.int64)
    frequencies = np.ones(len(panel)) if founder_frequencies is None else np.asarray(founder_frequencies)
    return workspace.state(panel, frequencies)


def _reported_posteriors(state, panel):
    """A fit's allele probabilities and copy occupancy, from its final E-step."""
    from scipy.special import expit
    current_p = expit(-state['conditional_flip_gains'])
    return dict(allele_probability=np.where(panel == 1, current_p, 1-current_p),
                occupancy=state['occupancy'][:-1])


def fit_posteriors(panel, frequencies, gl, observed, positions, *, prepared, generations,
                   recombination_rate_per_bp, wildcard_mass, interval_morgans):
    """The posterior arrays fit_fixed reports for a fit ending at (panel, frequencies)."""
    workspace = _Workspace(panel, gl, positions, observed, generations=generations,
        recombination_rate_per_bp=recombination_rate_per_bp, wildcard_mass=wildcard_mass,
        interval_morgans=interval_morgans, prepared=prepared)
    return _reported_posteriors(workspace.state(panel, frequencies), panel)


def fit_fixed(panel, gl, observed, positions, *, generations, learn_frequencies,
              frequencies=None, update_backstop_base=20, update_backstop_per_founder=4,
              update_backstop_per_site=2,
              tolerance=1e-6, prepared=None, recombination_rate_per_bp=5e-8,
              wildcard_mass=.01, interval_morgans=None):
    """Fit binary rows and optional copying frequencies without changing K.

    Input GL has shape (samples, markers, 3), normalized at observed cells;
    positions are ordered base-pair coordinates. ``generations`` scales the
    copying hazard (default genetic distance is 5e-8 Morgans/base pair), not an
    inferred biological generation count. The unknown-state prior mass is fixed
    at ``wildcard_mass`` (default 0.01); explicit ``interval_morgans`` overrides
    position-derived genetic distances.
    ``prepared`` must describe these same data and parameters.

    Each update performs frequency EM, then one founder's positive complete-
    emission gains, or one exact conditional flip at a GEM stationary point.
    The fit stops at a conditional fixed point; a row collision is deferred to
    outer selection. ``update_backstop_base + update_backstop_per_founder * K
    + update_backstop_per_site * L`` updates are a reported backstop
    (termination "update_backstop"). Callers
    control dynamic Numba allocation; one fit owns O(N*L*(K+1)**2) trajectory
    storage.
    """
    from haplotype_reconstruction.core import parallel

    panel, frequencies = canonical(panel, frequencies)
    if not learn_frequencies:
        frequencies = np.full(len(panel), 1. / len(panel))
    workspace = _Workspace(panel, gl, positions, observed, generations=generations,
        founder_frequencies=frequencies, prepared=prepared,
        recombination_rate_per_bp=recombination_rate_per_bp,
        wildcard_mass=wildcard_mass, interval_morgans=interval_morgans)
    history = []
    score = workspace.score(panel, frequencies)
    objective = score + regularizer(panel, frequencies, learn_frequencies)
    converged = False
    frequency_history = []
    backstop_reached = False
    termination = 'update_backstop'
    for iteration in range(update_backstop_base + update_backstop_per_founder * len(panel)
                           + update_backstop_per_site * panel.shape[1]):
        parallel.apply_dynamic_threads()
        state = workspace.state(panel, frequencies, full_statistics=not learn_frequencies)
        frequency_gain = 0.
        if learn_frequencies:
            counts = (state['initial_copy_counts'] + state['redraw_destination_counts'])[:-1]
            trial_f = (counts + 1. / len(panel)) / (counts.sum() + 1.)
            trial_score = workspace.score(panel, trial_f)
            trial_objective = trial_score + regularizer(panel, trial_f, True)
            if trial_objective >= objective - tolerance:
                frequency_gain = trial_objective - objective
                frequency_history.append(float(frequency_gain))
                frequencies, score, objective = trial_f, trial_score, trial_objective
            state = workspace.state(panel, frequencies)
        gains = state['flip_gains']
        positive = np.where(gains > tolerance, gains, 0.)
        founder = int(np.argmax(positive.sum(axis=1)))
        changed = positive[founder] > 0
        if not changed.any():
            # A conditional one-bit improvement can escape a stationary GEM
            # point. It is checked using the exact marginal path likelihood.
            exact = state['conditional_flip_gains']
            founder, site = np.unravel_index(np.argmax(exact), exact.shape)
            if exact[founder, site] <= tolerance:
                if frequency_gain > tolerance:
                    continue
                converged = True
                termination = 'conditional_fixed_point'
                break
            changed = np.zeros(panel.shape[1], dtype=bool)
            changed[site] = True
        trial = panel.copy()
        trial[founder, changed] = 1 - trial[founder, changed]
        if len(np.unique(trial, axis=0)) < len(panel):
            # A K change belongs to outer selection, not a fixed-K update.
            termination = 'row_collision_requires_outer_selection'
            break
        trial_score = workspace.score(trial, frequencies)
        trial_objective = trial_score + regularizer(trial, frequencies, learn_frequencies)
        if trial_objective < objective - tolerance:
            raise AssertionError('Single-founder GEM step decreased its objective')
        history.append(dict(iteration=iteration, founder=int(founder),
                            flips=int(changed.sum()), gain=float(trial_objective-objective)))
        panel, score, objective = trial, trial_score, trial_objective
    else:
        backstop_reached = True
    return dict(panel=panel, frequencies=frequencies,
                **_reported_posteriors(workspace.state(panel, frequencies), panel),
                log_likelihood=float(score),
                objective=float(objective), history=history, converged=converged,
                frequency_history=frequency_history,
                update_backstop_reached=backstop_reached, termination=termination)
