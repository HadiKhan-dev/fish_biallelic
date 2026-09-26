"""Normalized conditional diploid copying model for a fixed binary dictionary.

Positions are base pairs; interval_morgans may override their genetic distances.
The parameter ``generations`` is a dictionary-copy hazard multiplier, not an
estimated biological pedigree generation count. The default genetic rate is
5e-8 Morgans per base pair.

Ordered homologue states have T=(1-r)I+r*pi independently, with
r=-expm1(-generations*interval_Morgans). The extra state represents independent
Bernoulli(.5) unrepresented alleles, NOT one shared unknown founder. Its fixed
prior mass defaults to .01; remaining mass is uniform across dictionary rows
unless founder_frequencies is supplied. No founder-count/allele selection,
wildcard penalty, truth input, or environmental/thread-state mutation occurs.

Observed normalized GLs are emissions up to data-only normalization constants.
Missing cells have emission ONE, hence contribute zero log likelihood. Forward
messages are scaled; backward messages are rescaled by their maximum because
their arbitrary common scale cancels from all posterior/conditional ratios.

score_panel needs O(threads*S**2) working memory; e_step needs
O(threads*L*S**2) forward memory plus O(threads*K*L) aggregate statistics.
Optional sample statistics cost O(N*K*L); optional ordered pair posteriors cost
O(N*L*S**2), S=K+1. Caller controls Numba threads externally.
"""
import numpy as np
from numba import njit, prange, get_num_threads


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
def _emission_matrix(panel, site, gl, observed):
    k = len(panel)
    out = np.ones((k + 1, k + 1), np.float64)
    if observed:
        for a in range(k + 1):
            ha = panel[a, site] if a < k else -1
            for b in range(k + 1):
                hb = panel[b, site] if b < k else -1
                out[a, b] = _emission(ha, hb, gl)
    return out


@njit(cache=True)
def _predict(filtered, pi, rate):
    s = len(pi)
    row, col = np.zeros(s), np.zeros(s)
    for a in range(s):
        for b in range(s):
            row[a] += filtered[a, b]
            col[b] += filtered[a, b]
    stay = 1. - rate
    out = np.empty((s, s), np.float64)
    for a in range(s):
        for b in range(s):
            out[a, b] = (stay * stay * filtered[a, b]
                + stay * rate * (row[a] * pi[b] + pi[a] * col[b])
                + rate * rate * pi[a] * pi[b])
    return out


@njit(cache=True)
def _backward(weighted_future, pi, rate):
    s = len(pi)
    row, col = np.zeros(s), np.zeros(s)
    total = 0.
    for a in range(s):
        for b in range(s):
            value = weighted_future[a, b]
            row[a] += value * pi[b]
            col[b] += value * pi[a]
            total += pi[a] * value * pi[b]
    stay = 1. - rate
    out = np.empty((s, s), np.float64)
    for a in range(s):
        for b in range(s):
            out[a, b] = (stay * stay * weighted_future[a, b]
                        + stay * rate * (row[a] + col[b]) + rate * rate * total)
    maximum = np.max(out)
    if maximum > 0:
        out /= maximum
    return out


@njit(cache=True)
def _forward(panel, gl, observed, pi, rates, store_path):
    length, s = len(gl), len(pi)
    history = np.empty((length if store_path else 1, s, s), np.float64)
    filtered = np.outer(pi, pi)
    log_likelihood = 0.
    for site in range(length):
        prior = filtered if site == 0 else _predict(filtered, pi, rates[site - 1])
        emission = _emission_matrix(panel, site, gl[site], observed[site])
        filtered = prior * emission
        scale = np.sum(filtered)
        if scale <= 0:
            return -np.inf, history
        log_likelihood += np.log(scale)
        filtered /= scale
        history[site if store_path else 0] = filtered
    return log_likelihood, history


@njit(cache=True, parallel=True)
def _score_samples(panel, gl, observed, pi, rates):
    result = np.empty(len(gl), np.float64)
    for sample in prange(len(gl)):
        result[sample], _ = _forward(panel, gl[sample], observed[sample], pi, rates, False)
    return result


@njit(cache=True)
def _outside_rows_and_columns(matrix):
    """Nonnegative quadrant sums; avoid subtracting nearly all of the total."""
    s = len(matrix)
    tl = np.zeros((s + 1, s + 1))
    tr = np.zeros((s + 1, s + 1))
    bl = np.zeros((s + 1, s + 1))
    br = np.zeros((s + 1, s + 1))
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
    out = np.empty(s)
    for a in range(s):
        out[a] = tl[a, a] + tr[a, a + 1] + bl[a + 1, a] + br[a + 1, a + 1]
    return out


@njit(cache=True)
def _log_ratio(new, old):
    if new == 0:
        return -np.inf
    difference = new - old
    if abs(difference) <= .5 * old:
        return np.log1p(difference / old)
    return np.log(new) - np.log(old)


@njit(cache=True)
def _transition_statistics(filtered, future, pi, rate, denominator):
    """Redraw destinations include redraw-to-self; state changes exclude it."""
    s = len(pi)
    row, col = np.zeros(s), np.zeros(s)
    for a in range(s):
        for b in range(s):
            row[a] += filtered[a, b]
            col[b] += filtered[a, b]
    stay = 1. - rate
    destinations = np.zeros(s)
    same_redraw = 0.
    for a in range(s):
        first, second, same_first, same_second = 0., 0., 0., 0.
        weighted_row, weighted_col = 0., 0.
        for b in range(s):
            first += future[a, b] * (stay * col[b] + rate * pi[b])
            second += future[b, a] * (stay * row[b] + rate * pi[b])
            same_first += filtered[a, b] * future[a, b]
            same_second += filtered[b, a] * future[b, a]
            weighted_row += pi[b] * future[a, b]
            weighted_col += pi[b] * future[b, a]
        destinations[a] = rate * pi[a] * (first + second) / denominator
        same_redraw += rate * pi[a] * (stay * (same_first + same_second)
            + rate * (row[a] * weighted_row + col[a] * weighted_col)) / denominator
    # Roundoff can make a mathematically zero difference slightly negative.
    return destinations, max(0., np.sum(destinations) - same_redraw)


@njit(cache=True, parallel=True)
def _e_samples(panel, gl, observed, pi, rates, pair_output, sample_output, workers):
    n, length, k = len(gl), gl.shape[1], len(panel)
    s = k + 1
    likelihoods = np.empty(n)
    # Q flip, exact flip, known-partner Q, known-partner exact flip,
    # carrier mass, known-partner carrier mass, current/flip support mass.
    stats = np.zeros((workers, 8, k, length))
    occupancy = np.zeros((workers, s, length))
    sample_occupancy = np.zeros((n, s))
    unknown = np.zeros((n, length))
    initial = np.zeros((n, s))
    destinations = np.zeros((n, s))
    state_changes = np.zeros((n, max(0, length - 1)))
    redraws = np.zeros((n, max(0, length - 1)))
    pair = np.empty((n, length, s, s)) if pair_output else np.empty((0, 0, 0, 0))
    sample_stats = np.zeros((n, 4, k, length)) if sample_output else np.empty((0, 0, 0, 0))
    for worker_index in prange(workers):
        worker = np.int64(worker_index)
        for sample in range(worker, n, workers):
            ll, forward = _forward(panel, gl[sample], observed[sample], pi, rates, True)
            likelihoods[sample] = ll
            if not np.isfinite(ll):
                continue
            backward = np.ones((s, s))
            for site in range(length - 1, -1, -1):
                emission = _emission_matrix(panel, site, gl[sample, site], observed[sample, site])
                prior = np.outer(pi, pi) if site == 0 else _predict(forward[site - 1], pi, rates[site - 1])
                weights = prior * backward
                mass = weights * emission
                denominator = np.sum(mass)
                posterior = mass / denominator
                if pair_output:
                    pair[sample, site] = posterior
                for a in range(s):
                    copies = 0.
                    for b in range(s):
                        copies += posterior[a, b] + posterior[b, a]
                    occupancy[worker, a, site] += copies
                    sample_occupancy[sample, a] += copies
                    if site == 0:
                        initial[sample, a] = copies
                    if a == k:
                        unknown[sample, site] = copies
                outside = _outside_rows_and_columns(mass)
                known_mass = mass[:k, :k]
                known_denominator = np.sum(known_mass)
                known_outside = _outside_rows_and_columns(known_mass)
                for a in range(k):
                    qgain, known_qgain, carrier, known_carrier = 0., 0., 0., 0.
                    alt_mass, known_alt_mass, current_support, flip_support = 0., 0., 0., 0.
                    for b in range(s):
                        prob = posterior[a, b] if a == b else posterior[a, b] + posterior[b, a]
                        weight = weights[a, b] if a == b else weights[a, b] + weights[b, a]
                        carrier += prob
                        if b < k:
                            known_carrier += prob
                        if not observed[sample, site]:
                            continue
                        flipped_a = 1 - panel[a, site]
                        allele_b = flipped_a if a == b else (panel[b, site] if b < k else -1)
                        alternative = _emission(flipped_a, allele_b, gl[sample, site])
                        alt_mass += weight * alternative
                        if b < k:
                            known_alt_mass += weight * alternative
                        if prob > 0:
                            delta = _log_ratio(alternative, emission[a, b])
                            qgain += prob * delta
                            if b < k:
                                known_qgain += prob * delta
                                if alternative < emission[a, b]:
                                    current_support += prob
                                elif alternative > emission[a, b]:
                                    flip_support += prob
                    exact_gain, known_exact_gain = 0., 0.
                    if observed[sample, site]:
                        exact_gain = _log_ratio(outside[a] + alt_mass, denominator)
                        if known_denominator > 0 and known_carrier > 0:
                            known_exact_gain = _log_ratio(known_outside[a] + known_alt_mass, known_denominator)
                    stats[worker, 0, a, site] += qgain
                    stats[worker, 1, a, site] += exact_gain
                    stats[worker, 2, a, site] += known_qgain
                    stats[worker, 3, a, site] += known_exact_gain
                    stats[worker, 4, a, site] += carrier
                    stats[worker, 5, a, site] += known_carrier
                    stats[worker, 6, a, site] += current_support
                    stats[worker, 7, a, site] += flip_support
                    if sample_output:
                        sample_stats[sample, 0, a, site] = exact_gain
                        sample_stats[sample, 1, a, site] = known_exact_gain
                        sample_stats[sample, 2, a, site] = carrier
                        sample_stats[sample, 3, a, site] = known_carrier
                if site > 0:
                    future = emission * backward
                    redraw, changes = _transition_statistics(forward[site - 1], future,
                        pi, rates[site - 1], denominator)
                    destinations[sample] += redraw
                    redraws[sample, site - 1] = np.sum(redraw)
                    state_changes[sample, site - 1] = changes
                    backward = _backward(future, pi, rates[site - 1])
    return (likelihoods, stats, occupancy, sample_occupancy, unknown, initial,
            destinations, state_changes, redraws, pair, sample_stats)


def _prepare(panel, gl, positions, observed_mask, generations, recombination_rate_per_bp,
             wildcard_mass, interval_morgans, founder_frequencies):
    panel = np.asarray(panel)
    gl = np.asarray(gl, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=bool)
    if panel.ndim != 2 or min(panel.shape) < 1 or np.any((panel != 0) & (panel != 1)):
        raise ValueError('panel must contain K>=1 binary rows at L>=1 markers')
    k, length = panel.shape
    if gl.ndim != 3 or gl.shape[0] < 1 or gl.shape[1:] != (length, 3) or observed.shape != gl.shape[:2]:
        raise ValueError('GL/mask shapes must be (N,L,3)/(N,L) with N>=1')
    if positions.shape != (length,) or np.any(~np.isfinite(positions)) or np.any(np.diff(positions) < 0):
        raise ValueError('positions must be finite and nondecreasing')
    evidence = gl[observed]
    if np.any(~np.isfinite(evidence)) or np.any(evidence < 0) or not np.allclose(evidence.sum(axis=1), 1., rtol=1e-6, atol=1e-12):
        raise ValueError('observed GLs must be nonnegative finite normalized likelihoods')
    if not np.isfinite(generations) or generations < 0 or not np.isfinite(recombination_rate_per_bp) or recombination_rate_per_bp < 0:
        raise ValueError('generations and recombination rate must be finite and nonnegative')
    if not np.isfinite(wildcard_mass) or not 0 <= wildcard_mass <= 1:
        raise ValueError('wildcard_mass must lie in [0,1]')
    intervals = (np.diff(positions) * recombination_rate_per_bp if interval_morgans is None
                 else np.asarray(interval_morgans, dtype=np.float64))
    if intervals.shape != (length - 1,) or np.any(~np.isfinite(intervals)) or np.any(intervals < 0):
        raise ValueError('interval_morgans must contain L-1 finite nonnegative intervals')
    frequencies = np.ones(k) if founder_frequencies is None else np.asarray(founder_frequencies, dtype=np.float64)
    if frequencies.shape != (k,) or np.any(~np.isfinite(frequencies)) or np.any(frequencies < 0) or frequencies.sum() <= 0:
        raise ValueError('founder_frequencies must be nonnegative K-vector with positive total')
    frequencies = frequencies / frequencies.sum()
    pi = np.r_[(1 - wildcard_mass) * frequencies, wildcard_mass]
    rates = -np.expm1(-generations * intervals)
    return np.ascontiguousarray(panel, dtype=np.int64), np.ascontiguousarray(gl), np.ascontiguousarray(observed), pi, rates


def score_panel(panel, gl, positions, observed_mask, *, generations=3.,
                recombination_rate_per_bp=5e-8, wildcard_mass=.01,
                interval_morgans=None, founder_frequencies=None):
    """Marginal forward likelihood, not Viterbi score; state_prior includes U."""
    panel, gl, observed, pi, rates = _prepare(panel, gl, positions, observed_mask,
        generations, recombination_rate_per_bp, wildcard_mass, interval_morgans, founder_frequencies)
    values = _score_samples(panel, gl, observed, pi, rates)
    return dict(log_likelihood=float(values.sum()), sample_log_likelihood=values,
                state_prior=pi, redraw_probabilities=rates)


def e_step(panel, gl, positions, observed_mask, *, generations=3.,
           recombination_rate_per_bp=5e-8, wildcard_mass=.01,
           interval_morgans=None, founder_frequencies=None,
           return_pair_posteriors=False, return_sample_stats=False):
    """Fixed-panel posterior and sufficient statistics, with ordered pairs.

    flip_gains[K,L] is the expected COMPLETE-emission log-likelihood change,
    holding the current posterior fixed; an i/i state flips BOTH copies once.
    conditional_flip_gains is instead the EXACT marginal log-L change for one
    founder/site flip, with every other dictionary bit and pi fixed. It is NOT
    valid to sum these exact gains for simultaneous changes. The returned
    conditional bit probabilities assume equal prior odds for that bit only;
    they are neither a full Bayesian panel posterior nor independently tested.

    occupancy[S,L] sums expected copies over samples; sample_occupancy[N,S]
    sums them over markers. Carrier masses count a sample once per founder,
    including homozygotes. current_support_mass/flip_support_mass sum posterior
    known-partner carrier states whose OBSERVED local emission favours current
    or flipped allele. They are continuous diagnostics, not hard support counts.
    known_partner_conditional_flip_gains conditions both homologue states known;
    missing/no-known-carrier cells give zero and cannot supply support.

    initial_copy_counts and redraw_destination_counts are frequency-EM
    sufficient statistics; marker occupancy is NOT their substitute. Redraws
    include latent resets to the same state; expected_state_changes excludes
    those. No frequency update or release threshold is implemented here.
    """
    panel, gl, observed, pi, rates = _prepare(panel, gl, positions, observed_mask,
        generations, recombination_rate_per_bp, wildcard_mass, interval_morgans, founder_frequencies)
    values = _e_samples(panel, gl, observed, pi, rates, return_pair_posteriors, return_sample_stats,
                        min(len(gl), get_num_threads()))
    ll, partial, occ, sample_occ, unknown, initial, destinations, changes, redraws, pair, sample_stats = values
    if np.any(~np.isfinite(ll)):
        raise ValueError('E-step undefined for a sample with zero panel likelihood')
    stats = partial.sum(axis=0)
    gains = stats[1]
    probability = np.empty_like(gains)
    positive = gains >= 0
    exp_negative = np.exp(-gains[positive])
    probability[positive] = exp_negative / (1 + exp_negative)
    probability[~positive] = 1 / (1 + np.exp(gains[~positive]))
    result = dict(log_likelihood=float(ll.sum()), sample_log_likelihood=ll, state_prior=pi,
        redraw_probabilities=rates, flip_gains=stats[0], conditional_flip_gains=gains,
        known_partner_flip_gains=stats[2], known_partner_conditional_flip_gains=stats[3],
        carrier_mass=stats[4], known_partner_carrier_mass=stats[5],
        current_support_mass=stats[6], flip_support_mass=stats[7],
        conditional_current_allele_probability=probability,
        conditional_alt_probability=np.where(panel == 1, probability, 1 - probability),
        occupancy=occ.sum(axis=0), sample_occupancy=sample_occ,
        sample_unknown_copy_posterior=unknown,
        initial_copy_counts=initial.sum(axis=0), sample_initial_copy_counts=initial,
        redraw_destination_counts=destinations.sum(axis=0), sample_redraw_destination_counts=destinations,
        expected_state_changes=changes, expected_redraws=redraws)
    if return_pair_posteriors:
        result['pair_posteriors'] = pair
    if return_sample_stats:
        result.update(sample_conditional_flip_gains=sample_stats[:, 0],
            sample_known_partner_conditional_flip_gains=sample_stats[:, 1],
            sample_carrier_probability=sample_stats[:, 2],
            sample_known_partner_carrier_probability=sample_stats[:, 3])
    return result
