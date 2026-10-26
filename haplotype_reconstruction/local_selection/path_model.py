"""The normalised diploid copying model of one block (fixed founder panel).

Each of a sample's two homologues copies a founder row or the *unknown* state (its alleles are
independent Bernoulli(1/2) draws, NOT one shared unknown founder). The prior state probability is
pi = ((1 - u) f, u) with founder frequencies f and fixed unknown mass u; between adjacent sites
a homologue redraws its state from pi with probability r = 1 - exp(-g x Morgans) (g = the
copying redraw intensity per Morgan), so the ordered-pair transition is ((1-r) I + r 1 pi^T) per homologue.
Emissions are the sample's normalised genotype likelihoods of the pair's dosage (an unknown copy
mixes its two alleles); unobserved cells emit 1.

Two kernels evaluate the model:
* the lane kernels (score, fit statistics): samples in lane blocks of up to 64 side by side so the
  compiler vectorises each recursion across samples; trajectories store upper triangles (the two
  homologues are exchangeable); a lane block's forward trajectory is kept only until its backward
  pass, which follows at once (``_evaluate_lanes``: storage for one lane block per thread, not
  O(N L S^2) for the whole sample);
* the release E-step: per-sample full posteriors for the release statistics (per-sample exact
  one-bit gains and carrier probabilities, copy occupancy, MAP unknown-copy counts).
Every total over samples adds the samples one by one in ascending order, so no result depends
on the thread count. Memory: forward trajectories of up to 64 samples x L x S(S+1)/2 per thread
(S = K + 1), in one buffer shared by the sequential fits of a block search
(``WorkspaceBuffers``). A backward pass computes only the statistics its caller asks for
(``COUNTS``, ``FLIPS`` or ``EXACT``): the exact one-bit gains cost about a third of a full pass
and are needed only at stationary points and for the reported allele probabilities.
"""
from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange

from ..execution import numba_threads

# Maximum samples per lane block (a work split only). Wide blocks amortise loop overhead;
# narrower blocks use a growing tail budget when every block fits in one parallel wave.
LANE_WIDTH = 64
# Lane blocks are balanced: as many blocks as the chosen width needs, each the samples' share rounded
# up to a multiple of LANE_ALIGN, so padding lanes no longer widen every array a recursion walks
# (2.10: 40 samples were laid out 64 wide).
LANE_ALIGN = 8
# Samples per parallel release E-step batch and elements per combine tile (memory bounds only).
RELEASE_SAMPLE_BATCH = 128
_COMBINE_TILE = 4096

# Statistics levels of a backward pass (each includes the lower ones).
COUNTS = 0   # expected initial copies and redraw destinations (the frequency M-step)
FLIPS = 1    # + expected complete-emission flip gains and copy occupancy (GEM flips, merge bounds)
EXACT = 2    # + exact one-bit conditional gains (stationary points, reported allele probabilities)


@njit(cache=True)
def _log_ratio(new, old):
    if new == 0:
        return -np.inf
    difference = new - old
    if abs(difference) <= .5 * old:
        return np.log1p(difference / old)
    return np.log(new) - np.log(old)


@njit(cache=True)
def _emission(a, b, gl):
    if a < 0 and b < 0:
        return .25 * gl[0] + .5 * gl[1] + .25 * gl[2]
    if a < 0:
        return .5 * (gl[b] + gl[b + 1])
    if b < 0:
        return .5 * (gl[a] + gl[a + 1])
    return gl[a + b]


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


@njit(cache=True, inline='always', error_model='numpy')
def _reciprocals(denominator, inverse, lanes):
    """One inverse per lane; subnormal divisors retain direct division.

    A ratio can be finite even when the denominator's reciprocal overflows.
    The rare fallback also preserves the existing zero-normalizer error path.
    """
    regular = True
    for j in range(lanes):
        if denominator[j] >= np.finfo(np.float64).tiny:
            inverse[j] = 1. / denominator[j]
        else:
            inverse[j] = 0.
            regular = False
    return regular


@njit(cache=True, error_model='numpy')
def _normalise_pairs(values, denominator, inverse, regular, normalized, marginal,
                     s, lanes, copies):
    """Normalize triangular pairs and sum both endpoints in one sweep.

    Off-diagonal cells represent one ordered pair and contribute to both row
    marginals. ``copies`` is one for filtered row masses, two for copy counts.
    Source and normalized arrays may be the same forward buffer.
    """
    for a in range(s):
        for j in range(lanes):
            marginal[a, j] = 0.
    index = 0
    if regular:
        for a in range(s):
            for b in range(a, s):
                if a == b:
                    for j in range(lanes):
                        value = values[index, j] * inverse[j]
                        normalized[index, j] = value
                        marginal[a, j] += copies * value
                else:
                    for j in range(lanes):
                        value = values[index, j] * inverse[j]
                        normalized[index, j] = value
                        marginal[a, j] += copies * value
                        marginal[b, j] += copies * value
                index += 1
    else:
        for a in range(s):
            for b in range(a, s):
                if a == b:
                    for j in range(lanes):
                        value = values[index, j] / denominator[j]
                        normalized[index, j] = value
                        marginal[a, j] += copies * value
                else:
                    for j in range(lanes):
                        value = values[index, j] / denominator[j]
                        normalized[index, j] = value
                        marginal[a, j] += copies * value
                        marginal[b, j] += copies * value
                index += 1


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
    inverse = np.empty_like(scale)
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
        regular = _reciprocals(scale, inverse, lanes)
        _normalise_pairs(filtered, scale, inverse, regular, filtered, wt[slot, ws],
                         s, lanes, 1.)


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
def _backward_block(block, slot, lanes, panel, et, rt, st, ot, pi, rates, pt, wt, level, direct,
                    gains, occupancy, lane_gains, lane_occupancy, initial, destinations, work):
    """Backward pass of the first ``lanes`` samples of lane block ``block`` over
    their forward trajectories in pt[slot] and wt[slot].

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
    redraw destinations summed over sites. ``work`` holds the scratch arrays. ``level``: COUNTS
    (initial copies and destinations only), FLIPS (+ flip gains in gains[0] and occupancy) or
    EXACT (+ exact one-bit gains in gains[1]); a lower level computes its statistics exactly as a
    higher one does.
    """
    (backward, weights, mass, posterior, future, row, redraw, previous, copies, outside, qgain, alt,
     vec, prefix, suffix, tail, quotient, branch, log_old, have_log_old) = work
    full_statistics = level >= FLIPS
    exact = level >= EXACT
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
        # vec rows: 0 denominator, 1 total, 2 maximum, 3 left, 4 right, 5 inverse.
        for j in range(lanes):
            vec[0, j] = 0.
        if site:
            rate = rates[site - 1]
            stay = 1. - rate
            for b in range(s):
                c = rate * pi[b]
                for j in range(lanes):
                    previous[b, j] = stay * wt[slot, site - 1, b, j] + c
        store_mass = full_statistics or site == 0
        index = 0
        for a in range(s):
            ha = panel[a, site] if a < k else 2
            for b in range(a, s):
                hb = panel[b, site] if b < k else 2
                twice = 1. if a == b else 2.
                if exact:
                    for j in range(lanes):
                        weight = pt[slot, site, index, j] * backward[index, j]
                        value = weight * et[block, site, ha, hb, j]
                        future[index, j] = et[block, site, ha, hb, j] * backward[index, j]
                        weights[index, j] = weight
                        mass[index, j] = value
                        vec[0, j] += value if a == b else twice * value
                elif store_mass:
                    for j in range(lanes):
                        value = pt[slot, site, index, j] * backward[index, j] * et[block, site, ha, hb, j]
                        future[index, j] = et[block, site, ha, hb, j] * backward[index, j]
                        mass[index, j] = value
                        vec[0, j] += value if a == b else twice * value
                else:
                    for j in range(lanes):
                        value = pt[slot, site, index, j] * backward[index, j] * et[block, site, ha, hb, j]
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
        regular = _reciprocals(vec[0], vec[5], lanes)
        if store_mass:
            _normalise_pairs(mass, vec[0], vec[5], regular, posterior, copies, s, lanes, 2.)
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
        if full_statistics and not exact and any_observed:
            # FLIPS: the expected complete-emission gains only (as in the EXACT branch below).
            for a in range(k):
                ha = panel[a, site]
                for j in range(lanes):
                    qgain[a, j] = 0.
                for b in range(s):
                    hb = panel[b, site] if b < k else 2
                    p = _pair(a, b, s)
                    if a == b:
                        for j in range(lanes):
                            prob, q = posterior[p, j], qgain[a, j]
                            qgain[a, j] = q + prob * st[block, site, ha, j] if prob > 0 else q
                    else:
                        for j in range(lanes):
                            prob, q = 2. * posterior[p, j], qgain[a, j]
                            qgain[a, j] = q + prob * rt[block, site, ha, hb, j] if prob > 0 else q
                if direct:
                    g0 = gains[0, a, site]
                    for j in range(lanes):
                        if ot[block, site, j]:
                            g0 += qgain[a, j]
                    gains[0, a, site] = g0
                else:
                    for j in range(lanes):
                        if ot[block, site, j]:
                            lane_gains[0, a, site, j] = qgain[a, j]
        if exact and any_observed:
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
            if regular:
                for a in range(s):
                    c, pa = 2. * rate * pi[a], pi[a]
                    for j in range(lanes):
                        destinations[a, j] += c * redraw[a, j] * vec[5, j]
                        vec[1, j] += pa * row[a, j]
            else:
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
            if _reciprocals(vec[2], vec[5], lanes):
                for index in range(pairs):
                    for j in range(lanes):
                        backward[index, j] *= vec[5, j]
            else:
                for index in range(pairs):
                    for j in range(lanes):
                        largest = vec[2, j]
                        if largest > 0:
                            backward[index, j] /= largest


@njit(cache=True)
def _backward_work(s, width):
    """Scratch arrays of one _backward_block call: O(S**2 * width)."""
    pairs = s * (s + 1) // 2
    return (np.empty((pairs, width)), np.empty((pairs, width)), np.empty((pairs, width)),
            np.empty((pairs, width)), np.empty((pairs, width)),
            np.empty((s, width)), np.empty((s, width)), np.empty((s, width)), np.empty((s, width)),
            np.empty((s, width)), np.empty((s, width)), np.empty((s, width)),
            np.empty((6, width)), np.empty((s, s + 1, width)), np.empty((s, s + 1, width)),
            np.empty((s + 1, width)), np.empty(width), np.empty(width, dtype=np.int64),
            np.empty(width), np.empty(width, dtype=np.bool_))


@njit(cache=True, parallel=True, error_model='numpy')
def _evaluate_lanes(panel, et, rt, st, ot, pi, rates, pt, wt, level, n, workers):
    """Per-sample log likelihoods and the sufficient statistics at ``level``, each lane block's
    forward pass followed at once by its backward pass over the trajectory just stored in its
    worker's slot of pt / wt (``workers`` slots, not one per lane block).

    Totals add the samples in ascending order: with one worker the blocks run in order and add
    their lanes to the totals directly; otherwise waves of ``workers`` blocks run in parallel and
    store per-lane statistics, and every total element then adds the wave's lanes one by one in
    sample order. Every sum is therefore the same sequential chain over samples 0..N-1 for any
    thread count: a sample's +0 at an unobserved site adds nothing, as the totals start at +0 and
    so never hold -0. The statistics of a sample with likelihood -inf are meaningless; callers
    refuse such a panel.
    """
    full_statistics = level >= FLIPS
    blocks, length, width = et.shape[0], et.shape[1], et.shape[4]
    k = len(panel)
    s = k + 1
    pairs = s * (s + 1) // 2
    likelihoods = np.empty(n)
    gains = np.zeros((2, k, length))
    occupancy = np.zeros((s, length))
    initial = np.zeros(s)
    destinations = np.zeros(s)
    if workers <= 1:
        filtered = np.empty((pairs, width))
        scale, ll = np.empty(width), np.empty(width)
        alive = np.empty(width, dtype=np.bool_)
        work = _backward_work(s, width)
        lane_initial, lane_destinations = np.empty((s, width)), np.empty((s, width))
        no_gains, no_occupancy = np.empty((2, k, 0, 0)), np.empty((s, 0, 0))
        for block in range(blocks):
            lanes = min(width, n - block * width)
            _forward_block(block, lanes, panel, et, pi, rates, pt, wt, 0, filtered, scale, ll, alive)
            for j in range(lanes):
                likelihoods[block * width + j] = ll[j]
            _backward_block(block, 0, lanes, panel, et, rt, st, ot, pi, rates, pt, wt, level, True,
                            gains, occupancy, no_gains, no_occupancy, lane_initial, lane_destinations,
                            work)
            for j in range(lanes):
                for a in range(s):
                    initial[a] += lane_initial[a, j]
                    destinations[a] += lane_destinations[a, j]
        return likelihoods, gains, occupancy, initial, destinations
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
            filtered = np.empty((pairs, width))
            scale, ll = np.empty(width), np.empty(width)
            alive = np.empty(width, dtype=np.bool_)
            _forward_block(block, lanes, panel, et, pi, rates, pt, wt, slot, filtered, scale, ll, alive)
            for j in range(lanes):
                likelihoods[block * width + j] = ll[j]
            work = _backward_work(s, width)
            _backward_block(block, slot, lanes, panel, et, rt, st, ot, pi, rates, pt, wt, level, False,
                            gains, occupancy, lane_gains[slot], lane_occupancy[slot], lane_initial[slot],
                            lane_destinations[slot], work)
        # Fixed-order combine: each element adds this wave's lanes in sample order; the founder rows
        # only distribute elements over threads.
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
    return likelihoods, gains, occupancy, initial, destinations


@njit(cache=True)
def _emission_matrix_into(panel, site, gl, observed, out):
    k = len(panel)
    if not observed:
        out[:, :] = 1.
        return
    for a in range(k + 1):
        ha = panel[a, site] if a < k else -1
        for b in range(k + 1):
            hb = panel[b, site] if b < k else -1
            out[a, b] = _emission(ha, hb, gl)


@njit(cache=True)
def _predict_into(filtered, pi, rate, row, col, out):
    s = len(pi)
    row[:] = 0.
    col[:] = 0.
    for a in range(s):
        for b in range(s):
            row[a] += filtered[a, b]
            col[b] += filtered[a, b]
    stay = 1. - rate
    for a in range(s):
        for b in range(s):
            out[a, b] = (stay * stay * filtered[a, b]
                + stay * rate * (row[a] * pi[b] + pi[a] * col[b])
                + rate * rate * pi[a] * pi[b])


@njit(cache=True)
def _backward_into(weighted_future, pi, rate, row, col, out):
    s = len(pi)
    row[:] = 0.
    col[:] = 0.
    total = 0.
    for a in range(s):
        for b in range(s):
            value = weighted_future[a, b]
            row[a] += value * pi[b]
            col[b] += value * pi[a]
            total += pi[a] * value * pi[b]
    stay = 1. - rate
    for a in range(s):
        for b in range(s):
            out[a, b] = (stay * stay * weighted_future[a, b]
                        + stay * rate * (row[a] + col[b]) + rate * rate * total)
    maximum = np.max(out)
    if maximum > 0:
        out /= maximum


@njit(cache=True)
def _forward_into(panel, gl, observed, pi, rates, history, store_path, filtered, prior,
                  emission, row, col):
    """Scaled forward log-likelihood; normalized filtered states go to history
    (every site with ``store_path``, else only the last in history[0])."""
    length, s = len(gl), len(pi)
    for a in range(s):
        for b in range(s):
            filtered[a, b] = pi[a] * pi[b]
    log_likelihood = 0.
    for site in range(length):
        if site:
            _predict_into(filtered, pi, rates[site - 1], row, col, prior)
        else:
            prior[:, :] = filtered
        _emission_matrix_into(panel, site, gl[site], observed[site], emission)
        for a in range(s):
            for b in range(s):
                filtered[a, b] = prior[a, b] * emission[a, b]
        scale = np.sum(filtered)
        if scale <= 0:
            return -np.inf
        log_likelihood += np.log(scale)
        filtered /= scale
        history[site if store_path else 0] = filtered
    return log_likelihood


@njit(cache=True)
def _outside_rows_and_columns_into(matrix, tl, tr, bl, br, out):
    """Nonnegative quadrant sums; avoid subtracting nearly all of the total."""
    s = len(matrix)
    tl[:, :] = 0.
    tr[:, :] = 0.
    bl[:, :] = 0.
    br[:, :] = 0.
    for a in range(s):
        left, right = 0., 0.
        for b in range(s):
            left += matrix[a, b]
            tl[a + 1, b + 1] = tl[a, b + 1] + left
            j = s - 1 - b
            right += matrix[a, j]
            tr[a + 1, j] = tr[a, j] + right
    for a in range(s - 1, -1, -1):
        left, right = 0., 0.
        for b in range(s):
            left += matrix[a, b]
            bl[a, b + 1] = bl[a + 1, b + 1] + left
            j = s - 1 - b
            right += matrix[a, j]
            br[a, j] = br[a + 1, j] + right
    for a in range(s):
        out[a] = tl[a, a] + tr[a, a + 1] + bl[a + 1, a] + br[a + 1, a + 1]


@njit(cache=True, parallel=True)
def _emission_tables(gl, observed):
    """(N, L, 3, 3) emission of every allele pair (0, 1, 2 = unknown copy) per sample and site;
    1 at unobserved cells."""
    n, length = observed.shape
    output = np.empty((n, length, 3, 3))
    for sample in prange(n):
        for site in range(length):
            for a in range(3):
                for b in range(3):
                    output[sample, site, a, b] = (
                        _emission(a if a < 2 else -1, b if b < 2 else -1, gl[sample, site])
                        if observed[sample, site] else 1.)
    return output


@njit(cache=True)
def _ratio_tables(emissions, observed):
    """Data-only Q log ratios of a one-allele flip: ratios[s, l, h, partner] (partner 0, 1 or
    unknown) and self_ratios[s, l, h] (both copies flipped)."""
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
                            ratios[sample, site, h, partner] = _log_ratio(
                                table[1 - h, partner], table[h, partner])
                    if table[h, h] > 0:
                        self_ratios[sample, site, h] = _log_ratio(table[1 - h, 1 - h], table[h, h])
    return ratios, self_ratios


def balanced_width(samples: int, width: int) -> int:
    """The narrowest multiple of LANE_ALIGN (at most ``width``) that lays ``samples`` out in as many lane
    blocks as ``width`` does."""
    blocks = max(1, -(-int(samples) // width))
    share = -(-int(samples) // blocks)
    return min(width, max(LANE_ALIGN, -(-share // LANE_ALIGN) * LANE_ALIGN))


class PreparedBlock:
    """Data-only tables of one block's copying model, shared by every score and fit of a block
    search: GLs and observation mask at the discovery sites, redraw probabilities between
    adjacent sites, emission tables, and (built on first use) Q log ratios and lane layouts."""

    def __init__(self, likelihoods, observed, intervals_morgans, redraw_rate, unknown_mass):
        self.gl = np.ascontiguousarray(likelihoods, dtype=np.float64)
        self.observed = np.ascontiguousarray(observed, dtype=np.bool_)
        self.unknown_mass = float(unknown_mass)
        self.rates = -np.expm1(-float(redraw_rate) * np.asarray(intervals_morgans, dtype=np.float64))
        self.emissions = _emission_tables(self.gl, self.observed)
        self._lane_width = LANE_WIDTH
        self._lane_emissions = None
        self._fit_lanes = None

    @property
    def samples(self) -> int:
        return int(self.observed.shape[0])

    def state_prior(self, frequencies):
        """pi = ((1 - u) f / sum f, u)."""
        frequencies = np.asarray(frequencies, dtype=np.float64)
        return np.r_[(1 - self.unknown_mass) * (frequencies / frequencies.sum()), self.unknown_mass]

    def lane_emissions(self):
        width = LANE_WIDTH
        if self.samples > LANE_WIDTH:
            threads = get_num_threads()
            for candidate in (16, 32):
                if (self.samples + candidate - 1) // candidate <= threads:
                    width = candidate
                    break
        width = balanced_width(self.samples, width)
        if width != self._lane_width:
            # Retain one layout. Repacking occurs only when the caller crosses
            # a lane-budget boundary, never during a numerical kernel.
            self._lane_width = width
            self._lane_emissions = self._fit_lanes = None
        if self._lane_emissions is None:
            self._lane_emissions = _emission_lanes(self.emissions, width)
        return self._lane_emissions

    def fit_lanes(self):
        """(et, rt, st, ot) lane layouts of emissions, Q log ratios and the observed mask."""
        emissions = self.lane_emissions()
        if self._fit_lanes is None:
            ratios, self_ratios = _ratio_tables(self.emissions, self.observed)
            self._fit_lanes = _ratio_lanes(self.observed, ratios, self_ratios, self._lane_width)
        return (emissions, *self._fit_lanes)


def log_likelihood(panel, frequencies, prepared: PreparedBlock) -> float:
    """Forward log likelihood of a panel (no trajectories kept)."""
    et = prepared.lane_emissions()
    workers = min(len(et), get_num_threads())
    if get_num_threads() > workers:
        # The budget claim stays held: restoring this caller's team is safe.
        with numba_threads(workers):
            values = _score_lanes(np.ascontiguousarray(panel, dtype=np.int64), et,
                                  prepared.state_prior(frequencies), prepared.rates, prepared.samples,
                                  workers)
    else:
        values = _score_lanes(np.ascontiguousarray(panel, dtype=np.int64), et,
                              prepared.state_prior(frequencies), prepared.rates, prepared.samples,
                              workers)
    return float(values.sum())


class WorkspaceBuffers:
    """Trajectory scratch shared by the sequential evaluations of one block search: one priors /
    row-marginal buffer pair grown to the largest (threads, panel) seen; an evaluation uses a
    prefix view and consumes it before returning, so evaluations of any workspaces can share it
    (one at a time)."""

    def __init__(self):
        self._priors = self._rows = None

    def views(self, priors_shape, rows_shape):
        need_priors, need_rows = int(np.prod(priors_shape)), int(np.prod(rows_shape))
        if self._priors is None or self._priors.size < need_priors:
            self._priors = None
            self._priors = np.empty(need_priors)
        if self._rows is None or self._rows.size < need_rows:
            self._rows = None
            self._rows = np.empty(need_rows)
        return (self._priors[:need_priors].reshape(priors_shape),
                self._rows[:need_rows].reshape(rows_shape))

    def nbytes(self) -> int:
        return sum(0 if array is None else array.nbytes for array in (self._priors, self._rows))


def workspace_bytes(sites: int, founders: int, threads: int = 1) -> int:
    """Bytes of the trajectory scratch of one evaluation: priors and row marginals (float64) of
    one 64-sample lane block per thread."""
    s = int(founders) + 1
    return 8 * max(1, int(threads)) * LANE_WIDTH * int(sites) * (s * (s + 1) // 2 + s)


class PathWorkspace:
    """Evaluations of one panel size over a prepared block: ``state`` returns the per-sample log
    likelihood total and the sufficient statistics of (panel, frequencies) at a level (forward and
    backward pass together), cached for the latest (panel, frequencies) at the highest level
    computed (every level computes its statistics identically). ``buffers``: the block search's
    shared trajectory scratch (a private one without)."""

    def __init__(self, k: int, prepared: PreparedBlock, buffers: WorkspaceBuffers | None = None):
        self.prepared = prepared
        self.lanes = prepared.fit_lanes()
        self.n = prepared.samples
        self.k = int(k)
        self.buffers = WorkspaceBuffers() if buffers is None else buffers
        self.panel = self.frequencies = self.cached = None
        self.cached_level = -1

    def _workers(self):
        """Lane blocks run in parallel; at most one thread per block."""
        return min(len(self.lanes[0]), get_num_threads())

    def state(self, panel, frequencies, level=EXACT) -> dict:
        """log_likelihood (sum over samples), initial_copy_counts and redraw_destination_counts
        (S) at every level; flip_gains (expected complete-emission gains, K x L) and occupancy
        (S, L) from FLIPS; conditional_flip_gains (exact one-bit gains, K x L) at EXACT."""
        frequencies = np.asarray(frequencies, dtype=np.float64)
        if (self.cached is not None and self.cached_level >= level
                and np.array_equal(panel, self.panel) and np.array_equal(frequencies, self.frequencies)):
            return self.cached
        # The pool may grant more threads after other blocks finish. Each
        # evaluation uses the current data layout; cached statistics need no repack.
        self.lanes = self.prepared.fit_lanes()
        et, rt, st, ot = self.lanes
        length, width = et.shape[1], et.shape[4]
        s = self.k + 1
        workers = self._workers()
        priors, rows = self.buffers.views((workers, length, s * (s + 1) // 2, width),
                                          (workers, length, s, width))
        pi = self.prepared.state_prior(frequencies)
        # More threads add parallel-region overhead without more lane work.
        # Keep the budget claim and restore the mask before another phase runs.
        if get_num_threads() > workers:
            with numba_threads(workers):
                likelihoods, gains, occupancy, initial, destinations = _evaluate_lanes(
                    panel, et, rt, st, ot, pi, self.prepared.rates, priors, rows, level, self.n, workers)
        else:
            likelihoods, gains, occupancy, initial, destinations = _evaluate_lanes(
                panel, et, rt, st, ot, pi, self.prepared.rates, priors, rows, level, self.n, workers)
        if np.any(~np.isfinite(likelihoods)):
            raise ValueError("E-step undefined for a sample with zero panel likelihood")
        # A zero posterior normaliser of a finite-likelihood sample (underflow) leaves NaN or inf
        # in its initial or destination counts and hence in their nonnegative totals.
        if not (np.isfinite(initial).all() and np.isfinite(destinations).all()):
            raise ZeroDivisionError("E-step: zero posterior normaliser at a site of a sample "
                                    "with a finite likelihood")
        state = dict(log_likelihood=float(likelihoods.sum()), initial_copy_counts=initial,
                     redraw_destination_counts=destinations)
        if level >= FLIPS:
            state.update(flip_gains=gains[0], occupancy=occupancy)
        if level >= EXACT:
            state.update(conditional_flip_gains=gains[1])
        self.panel, self.frequencies = np.array(panel, copy=True), frequencies.copy()
        self.cached, self.cached_level = state, level
        return state


# ---------------------------------------------------------------------------------------------
# Release E-step (per-sample full posteriors).
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def _unknown_count(none, one, two, band):
    """(count, strict count, gap) of a site's posterior masses of 0, 1 and 2 unknown copies: the
    smallest count whose mass is within ``band`` x the largest mass of it (a larger count must win
    by more than that relative margin); the first maximum (the rule without the band); the relative
    gap between the two largest masses."""
    top = max(none, max(one, two))
    floor = top - band * top
    count = 0
    if none < floor:
        count = 1 if one >= floor else 2
    strict = 0
    if one > none:
        strict = 1
    if two > (one if strict == 1 else none):
        strict = 2
    second = max(min(none, one), min(max(none, one), two))
    return count, strict, (top - second) / top


@njit(cache=True, parallel=True)
def _release_samples(panel, gl, observed, pi, rates, favourable_margin, count_band, batch, keep_gains):
    """Per-sample posteriors for the release, aggregates summed in sample order.

    Returns per-sample log likelihoods; the exact one-bit gains summed over samples (K, L) (log
    of the marginal likelihood ratio of flipping founder a's allele at the site, every other bit
    and pi fixed; 0 at unobserved cells); per founder and site the favourable carriers by count
    (samples whose own exact gain is below -favourable_margin and whose carrier probability,
    the posterior probability that a homologue copies founder a with a homozygous state counted
    once, exceeds 1/2) and by mass (their summed carrier probabilities); copy occupancy summed
    over samples (S, L); per sample the largest MAP unknown-copy count (0, 1, 2) over its
    observed sites, where a larger count must win by more than ``count_band`` relative posterior
    mass (else the smaller count, release 2.1), with per sample the number of observed sites where
    the band took a smaller count than the strict MAP and the smallest relative gap between a
    site's two largest count masses (inf without observed sites); per sample the number of
    observed carrier cells (carrier probability above
    1/2) whose exact gain lies inside the favourable band [-margin, margin] (release 2.1: the
    band around the exact zero of symmetric-evidence cells; not favourable there); and, with
    ``keep_gains`` (diagnostics only), every carrier cell's exact gain (N, K, L; NaN elsewhere),
    else a (1, 1, 1) placeholder.
    """
    n, length, k = len(gl), gl.shape[1], len(panel)
    s = k + 1
    likelihoods = np.empty(n)
    gains = np.zeros((k, length))
    supporters = np.zeros((k, length), dtype=np.int64)
    carrier_mass = np.zeros((k, length))
    occupancy = np.zeros((s, length))
    width = max(1, min(batch, n))
    gain_rows = np.empty((width, k, length))
    support_rows = np.empty((width, k, length), dtype=np.int64)
    mass_rows = np.empty((width, k, length))
    occupancy_rows = np.empty((width, s, length))
    gain_size, occupancy_size = k * length, s * length
    unknown_slots = np.zeros(n, dtype=np.int64)
    count_band_sites = np.zeros(n, dtype=np.int64)
    count_gaps = np.full(n, np.inf)
    band_cells = np.zeros(n, dtype=np.int64)
    sample_gains = (np.full((n, k, length), np.nan) if keep_gains
                    else np.full((1, 1, 1), np.nan))
    for start in range(0, n, width):
        count = min(width, n - start)
        for slot_index in prange(count):
            worker = np.int64(slot_index)
            sample = start + worker
            gain_rows[worker] = 0.
            support_rows[worker] = 0
            mass_rows[worker] = 0.
            occupancy_rows[worker] = 0.
            forward = np.empty((length, s, s))
            emission, prior = np.empty((s, s)), np.empty((s, s))
            weights, mass = np.empty((s, s)), np.empty((s, s))
            posterior, future = np.empty((s, s)), np.empty((s, s))
            backward = np.ones((s, s))
            row, col = np.empty(s), np.empty(s)
            outside = np.empty(s)
            tl, tr = np.empty((s + 1, s + 1)), np.empty((s + 1, s + 1))
            bl, br = np.empty((s + 1, s + 1)), np.empty((s + 1, s + 1))
            ll = _forward_into(panel, gl[sample], observed[sample], pi, rates, forward, True,
                               mass, prior, emission, row, col)
            likelihoods[sample] = ll
            if not np.isfinite(ll):
                continue
            slots = 0
            for site in range(length - 1, -1, -1):
                _emission_matrix_into(panel, site, gl[sample, site], observed[sample, site], emission)
                if site == 0:
                    for a in range(s):
                        for b in range(s):
                            prior[a, b] = pi[a] * pi[b]
                else:
                    _predict_into(forward[site - 1], pi, rates[site - 1], row, col, prior)
                for a in range(s):
                    for b in range(s):
                        weights[a, b] = prior[a, b] * backward[a, b]
                        mass[a, b] = weights[a, b] * emission[a, b]
                denominator = np.sum(mass)
                for a in range(s):
                    for b in range(s):
                        posterior[a, b] = mass[a, b] / denominator
                for a in range(s):
                    copies = 0.
                    for b in range(s):
                        copies += posterior[a, b] + posterior[b, a]
                    occupancy_rows[worker, a, site] = copies
                if observed[sample, site]:
                    # MAP number of unknown copies under the relative band (_unknown_count).
                    none = 0.
                    for a in range(k):
                        for b in range(k):
                            none += posterior[a, b]
                    one = 0.
                    for b in range(k):
                        one += posterior[k, b]
                    for a in range(k):
                        one += posterior[a, k]
                    two = posterior[k, k]
                    best, strict, gap = _unknown_count(none, one, two, count_band)
                    if best < strict:
                        count_band_sites[sample] += 1
                    if gap < count_gaps[sample]:
                        count_gaps[sample] = gap
                    if best > slots:
                        slots = best
                _outside_rows_and_columns_into(mass, tl, tr, bl, br, outside)
                for a in range(k):
                    carrier, alt_mass = 0., 0.
                    for b in range(s):
                        prob = posterior[a, b] if a == b else posterior[a, b] + posterior[b, a]
                        weight = weights[a, b] if a == b else weights[a, b] + weights[b, a]
                        carrier += prob
                        if not observed[sample, site]:
                            continue
                        flipped_a = 1 - panel[a, site]
                        allele_b = flipped_a if a == b else (panel[b, site] if b < k else -1)
                        alt_mass += weight * _emission(flipped_a, allele_b, gl[sample, site])
                    exact_gain = 0.
                    if observed[sample, site]:
                        exact_gain = _log_ratio(outside[a] + alt_mass, denominator)
                    gain_rows[worker, a, site] = exact_gain
                    if exact_gain < -favourable_margin:
                        support_rows[worker, a, site] = 1 if carrier > .5 else 0
                        mass_rows[worker, a, site] = carrier
                    elif carrier > .5 and observed[sample, site] and exact_gain <= favourable_margin:
                        band_cells[sample] += 1
                    if keep_gains and carrier > .5 and observed[sample, site]:
                        sample_gains[sample, a, site] = exact_gain
                if site > 0:
                    for a in range(s):
                        for b in range(s):
                            future[a, b] = emission[a, b] * backward[a, b]
                    _backward_into(future, pi, rates[site - 1], row, col, backward)
            unknown_slots[sample] = slots
        # Fixed-order combine: each element adds this batch's rows in sample order; tiles only
        # distribute elements over threads.
        gain_matrix = gain_rows.reshape((width, gain_size))
        support_matrix = support_rows.reshape((width, gain_size))
        mass_matrix = mass_rows.reshape((width, gain_size))
        gain_total = gains.reshape(gain_size)
        support_total = supporters.reshape(gain_size)
        mass_total = carrier_mass.reshape(gain_size)
        for tile in prange((gain_size + _COMBINE_TILE - 1) // _COMBINE_TILE):
            low = np.int64(tile) * _COMBINE_TILE
            high = min(gain_size, low + _COMBINE_TILE)
            for row_index in range(count):
                for element in range(low, high):
                    gain_total[element] += gain_matrix[row_index, element]
                    support_total[element] += support_matrix[row_index, element]
                    mass_total[element] += mass_matrix[row_index, element]
        occupancy_matrix = occupancy_rows.reshape((width, occupancy_size))
        occupancy_total = occupancy.reshape(occupancy_size)
        for tile in prange((occupancy_size + _COMBINE_TILE - 1) // _COMBINE_TILE):
            low = np.int64(tile) * _COMBINE_TILE
            high = min(occupancy_size, low + _COMBINE_TILE)
            for row_index in range(count):
                for cell in range(low, high):
                    occupancy_total[cell] += occupancy_matrix[row_index, cell]
    return (likelihoods, gains, supporters, carrier_mass, occupancy, unknown_slots, count_band_sites,
            count_gaps, band_cells, sample_gains)


def release_statistics(panel, frequencies, prepared: PreparedBlock, favourable_margin,
                       unknown_count_band, keep_gains: bool = False) -> dict:
    """The release E-step of a fitted (panel, frequencies) (see ``_release_samples``):
    conditional_flip_gains (K, L), favourable carriers by count (supporters, int64) and by mass
    (carrier_mass) (K, L), occupancy (S, L); has_depth and unknown_slots (2 without depth) per
    sample, the MAP counts under the relative band ``unknown_count_band``, with unknown_band_sites
    (observed sites where the band took a smaller count than the strict MAP, summed over samples)
    and unknown_gaps (per sample the smallest relative gap between a site's two largest count
    masses, inf without depth); band_cells (carrier cells inside the favourable band, summed over
    samples) and, with ``keep_gains``, sample_gains (N, K, L carrier-cell exact gains, NaN
    elsewhere)."""
    (likelihoods, gains, supporters, carrier_mass, occupancy, slots, count_band_sites, count_gaps, band,
     sample_gains) = _release_samples(
        np.ascontiguousarray(panel, dtype=np.int64), prepared.gl, prepared.observed,
        prepared.state_prior(frequencies), prepared.rates, float(favourable_margin),
        float(unknown_count_band), RELEASE_SAMPLE_BATCH, bool(keep_gains))
    if np.any(~np.isfinite(likelihoods)):
        raise ValueError("E-step undefined for a sample with zero panel likelihood")
    has_depth = prepared.observed.any(axis=1)
    statistics = dict(conditional_flip_gains=gains, supporters=supporters, carrier_mass=carrier_mass,
                      occupancy=occupancy, has_depth=has_depth,
                      unknown_slots=np.where(has_depth, slots, 2).astype(np.int8),
                      unknown_band_sites=int(count_band_sites.sum()), unknown_gaps=count_gaps,
                      band_cells=int(band.sum()))
    if keep_gains:
        statistics["sample_gains"] = sample_gains
    return statistics
