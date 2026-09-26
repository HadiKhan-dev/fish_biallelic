"""Cached-trajectory fixed-panel fitter for the normalized diploid copying model.

Prepare read-only emission/log-ratio tables once, retain predicted priors plus
filtered marginals after a score, and reuse them in the backward pass. Before
a frequency update, compute only the necessary frequency statistics. State
matrices remain ordered/full, without symmetry approximation or fastmath.

Workspace memory is O(N*L*S**2), where S=K+1 includes the unknown state. The
cached trajectories avoid repeated forward/prediction work within a fit.
One workspace belongs to one fit, not a shared/global cache. Dynamic threads
remain caller-controlled.
"""
import numpy as np
from numba import njit, prange, get_num_threads
import math
from . import path_model as reference
from .objectives import log_binary_haplotype_set_count


def canonical(panel, frequencies=None):
    """Sort exact binary rows and sum their frequency mass when deduplicating."""
    panel = np.ascontiguousarray(panel, dtype=np.int64)
    unique, inverse = np.unique(panel, axis=0, return_inverse=True)
    if frequencies is None:
        frequencies = np.full(len(panel), 1. / len(panel))
    mass = np.bincount(inverse, weights=frequencies, minlength=len(unique))
    return unique, mass / mass.sum()


def regularizer(panel, frequencies, learn_frequencies):
    """Dictionary-size penalty plus uniform-centered frequency shrinkage.

    This is a regularized best-panel objective, not a Bayesian posterior over K.
    """
    k, length = panel.shape
    value = -float(log_binary_haplotype_set_count(length, k)) - math.log(k) - math.log(k + 1)
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
def _backward_into(future, pi, rate, row, col, output):
    s = len(pi)
    row[:] = 0.
    col[:] = 0.
    total = 0.
    for a in range(s):
        for b in range(s):
            value = future[a, b]
            row[a] += value * pi[b]
            col[b] += value * pi[a]
            total += pi[a] * value * pi[b]
    stay = 1. - rate
    maximum = 0.
    for a in range(s):
        for b in range(s):
            value = (stay * stay * future[a, b]
                     + stay * rate * (row[a] + col[b]) + rate * rate * total)
            output[a, b] = value
            maximum = max(maximum, value)
    if maximum > 0:
        output /= maximum


@njit(cache=True)
def _outside_into(matrix, tl, tr, bl, br, output):
    # Scratch boundaries are zero once per worker; interiors are overwritten.
    s = len(matrix)
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
        output[a] = tl[a, a] + tr[a, a + 1] + bl[a + 1, a] + br[a + 1, a + 1]


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


@njit(cache=True, parallel=True)
def _forward_samples(panel, emissions, pi, rates, priors, rows, cols, workers):
    """Keep P_t before the marker emission and marginals of filtered F_t."""
    n, length, s = len(emissions), emissions.shape[1], len(pi)
    k = s - 1
    likelihoods = np.empty(n)
    for wi in prange(workers):
        worker = np.int64(wi)
        filtered = np.empty((s, s))
        for sample in range(worker, n, workers):
            ll = 0.
            for site in range(length):
                prior = priors[sample, site]
                if site == 0:
                    for a in range(s):
                        for b in range(s):
                            prior[a, b] = pi[a] * pi[b]
                else:
                    rate = rates[site-1]
                    stay = 1. - rate
                    for a in range(s):
                        for b in range(s):
                            prior[a, b] = (stay * stay * filtered[a, b]
                                + stay * rate * (rows[sample, site-1, a] * pi[b]
                                                + pi[a] * cols[sample, site-1, b])
                                + rate * rate * pi[a] * pi[b])
                scale = 0.
                for a in range(s):
                    ha = panel[a, site] if a < k else 2
                    for b in range(s):
                        hb = panel[b, site] if b < k else 2
                        value = prior[a, b] * emissions[sample, site, ha, hb]
                        filtered[a, b] = value
                        scale += value
                if scale <= 0:
                    ll = -np.inf
                    break
                ll += np.log(scale)
                filtered /= scale
                rows[sample, site, :] = 0.
                cols[sample, site, :] = 0.
                for a in range(s):
                    for b in range(s):
                        rows[sample, site, a] += filtered[a, b]
                        cols[sample, site, b] += filtered[a, b]
            likelihoods[sample] = ll
    return likelihoods


@njit(cache=True, parallel=True)
def _backward_samples(panel, emissions, ratios, self_ratios, observed, pi, rates,
                      priors, rows, cols, full_statistics, workers):
    n, length, k = len(emissions), emissions.shape[1], len(panel)
    s = k + 1
    gains = np.zeros((workers, 2, k, length))
    occupancy = np.zeros((workers, s, length))
    initial = np.zeros((workers, s))
    destinations = np.zeros((workers, s))
    for wi in prange(workers):
        worker = np.int64(wi)
        backward = np.empty((s, s))
        weights = np.empty((s, s))
        mass = np.empty((s, s))
        posterior = np.empty((s, s))
        future = np.empty((s, s))
        row, col = np.empty(s), np.empty(s)
        outside = np.empty(s)
        tl = np.zeros((s+1, s+1))
        tr = np.zeros((s+1, s+1))
        bl = np.zeros((s+1, s+1))
        br = np.zeros((s+1, s+1))
        for sample in range(worker, n, workers):
            backward[:, :] = 1.
            for site in range(length-1, -1, -1):
                emission = emissions[sample, site]
                denominator = 0.
                for a in range(s):
                    ha = panel[a, site] if a < k else 2
                    for b in range(s):
                        hb = panel[b, site] if b < k else 2
                        weights[a, b] = priors[sample, site, a, b] * backward[a, b]
                        mass[a, b] = weights[a, b] * emission[ha, hb]
                        future[a, b] = emission[ha, hb] * backward[a, b]
                        denominator += mass[a, b]
                if full_statistics or site == 0:
                    for a in range(s):
                        for b in range(s):
                            posterior[a, b] = mass[a, b] / denominator
                    for a in range(s):
                        copies = 0.
                        for b in range(s):
                            copies += posterior[a, b] + posterior[b, a]
                        if full_statistics:
                            occupancy[worker, a, site] += copies
                        if site == 0:
                            initial[worker, a] += copies
                if full_statistics and observed[sample, site]:
                    _outside_into(mass, tl, tr, bl, br, outside)
                    for a in range(k):
                        ha = panel[a, site]
                        qgain, alt_mass = 0., 0.
                        for b in range(s):
                            hb = panel[b, site] if b < k else 2
                            if a == b:
                                prob, weight = posterior[a, b], weights[a, b]
                                alternative = emission[1-ha, 1-ha]
                                delta = self_ratios[sample, site, ha]
                            else:
                                prob = posterior[a, b] + posterior[b, a]
                                weight = weights[a, b] + weights[b, a]
                                alternative = emission[1-ha, hb]
                                delta = ratios[sample, site, ha, hb]
                            alt_mass += weight * alternative
                            if prob > 0:
                                qgain += prob * delta
                        gains[worker, 0, a, site] += qgain
                        gains[worker, 1, a, site] += reference._log_ratio(
                            outside[a] + alt_mass, denominator)
                if site > 0:
                    rate = rates[site-1]
                    stay = 1. - rate
                    for a in range(s):
                        first, second = 0., 0.
                        for b in range(s):
                            first += future[a, b] * (stay * cols[sample, site-1, b] + rate * pi[b])
                            second += future[b, a] * (stay * rows[sample, site-1, b] + rate * pi[b])
                        destinations[worker, a] += rate * pi[a] * (first+second) / denominator
                    _backward_into(future, pi, rate, row, col, backward)
    return gains, occupancy, initial, destinations


def prepare_fit_observations(prepared):
    """Extend a forward-scorer block preparation with data-only Q log ratios."""
    if 'fit_ratios' in prepared and 'fit_self_ratios' in prepared:
        return prepared
    ratios, self_ratios = _prepare_ratios(prepared['emissions'], prepared['observed'])
    return dict(prepared, fit_ratios=ratios, fit_self_ratios=self_ratios)


class _Workspace:
    """One fixed-K fit owns data tables and the latest parameter trajectory."""

    def __init__(self, panel, gl, positions, observed, generations=3.,
                 recombination_rate_per_bp=5e-8, wildcard_mass=.01,
                 interval_morgans=None, founder_frequencies=None, prepared=None):
        if prepared is None:
            panel, gl, self.observed, pi, self.rates = reference._prepare(
                panel, gl, positions, observed, generations, recombination_rate_per_bp,
                wildcard_mass, interval_morgans, founder_frequencies)
            self.emissions, self.ratios, self.self_ratios = _prepare_emissions(gl, self.observed)
        else:
            # The outer scorer validated these observations once for this exact
            # block. Read-only data tables can be shared, trajectories cannot.
            self.observed, self.rates = prepared['observed'], prepared['rates']
            self.emissions = prepared['emissions']
            assert prepared['wildcard_mass'] == wildcard_mass
            assert gl.shape[:2] == self.observed.shape
            assert np.array_equal(positions, prepared['positions'])
            intervals = (np.diff(np.asarray(positions, dtype=np.float64)) * recombination_rate_per_bp
                         if interval_morgans is None else np.asarray(interval_morgans))
            assert np.array_equal(self.rates, -np.expm1(-generations * intervals))
            fitted_prepared = prepare_fit_observations(prepared)
            self.ratios, self.self_ratios = fitted_prepared['fit_ratios'], fitted_prepared['fit_self_ratios']
        n, length, s = len(gl), gl.shape[1], len(panel)+1
        self.priors = np.empty((n, length, s, s))
        self.rows = np.empty((n, length, s))
        self.cols = np.empty_like(self.rows)
        self.wildcard_mass = wildcard_mass
        self.panel = self.frequencies = self.state_cache = None
        self.state_is_full = False

    def score(self, panel, frequencies):
        frequencies = np.asarray(frequencies, dtype=np.float64)
        if (self.panel is not None and np.array_equal(panel, self.panel)
                and np.array_equal(frequencies, self.frequencies)):
            return float(self.likelihoods.sum())
        pi = np.r_[(1-self.wildcard_mass) * (frequencies / frequencies.sum()), self.wildcard_mass]
        values = _forward_samples(panel, self.emissions, pi, self.rates,
            self.priors, self.rows, self.cols, min(len(self.observed), get_num_threads()))
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
        gains, occ, initial, dest = _backward_samples(panel, self.emissions, self.ratios,
            self.self_ratios, self.observed, self.pi, self.rates, self.priors,
            self.rows, self.cols, full_statistics, min(len(self.observed), get_num_threads()))
        totals = gains.sum(axis=0)
        self.state_cache = dict(log_likelihood=float(self.likelihoods.sum()),
            sample_log_likelihood=self.likelihoods, state_prior=self.pi,
            redraw_probabilities=self.rates, flip_gains=totals[0],
            conditional_flip_gains=totals[1], occupancy=occ.sum(axis=0),
            initial_copy_counts=initial.sum(axis=0), redraw_destination_counts=dest.sum(axis=0))
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


def fit_fixed(panel, gl, observed, positions, *, generations, learn_frequencies,
              frequencies=None, max_updates=20, tolerance=1e-6, prepared=None,
              recombination_rate_per_bp=5e-8, wildcard_mass=.01,
              interval_morgans=None):
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
    A row collision is deferred to outer selection. Budgets, tolerances and
    conditional release-probability semantics are preserved. Callers control
    dynamic Numba allocation; one fit owns O(N*L*(K+1)**2) trajectory storage.
    """
    from scipy.special import expit
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
    budget_reached = False
    termination = 'update_budget'
    for iteration in range(max_updates):
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
        budget_reached = True
    final_state = workspace.state(panel, frequencies)
    current_p = expit(-final_state['conditional_flip_gains'])
    allele_p = np.where(panel == 1, current_p, 1-current_p)
    return dict(panel=panel, frequencies=frequencies, allele_probability=allele_p,
                occupancy=final_state['occupancy'][:-1], log_likelihood=float(score),
                objective=float(objective), history=history, converged=converged,
                frequency_history=frequency_history,
                update_budget_reached=budget_reached, termination=termination)
