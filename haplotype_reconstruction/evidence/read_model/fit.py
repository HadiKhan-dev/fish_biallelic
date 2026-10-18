"""Fits of the nested read models on training histograms (folds 0 and 1).

Candidates, from simplest (the order is the tie rule of the selection):

* ``binomial``: one shared binomial model, error e and heterozygote ALT fraction q, fitted by
  EM from several ordered starts (per-sample genotype mixtures are nuisances);
* ``sample_binomial``: per-sample logit-error and logit-balance effects pooled by an
  empirical-Bayes normal measurement-error approximation (diagonal Laplace: each sample's
  uncertainty from the observed profile Hessian of its own likelihood; hyperparameters from
  training data only), binomial genotype kernels;
* ``sample_joint``: those effects refitted jointly with one shared heterozygote concentration
  kappa (beta-binomial heterozygotes, binomial homozygotes).

Genotype mixtures are fitting nuisances only: they are never returned as genotype priors.
Optimiser stop reasons (convergence, iteration cap, line-search stop) are recorded as diagnostics
(``ReadModelFit.success``) and never decide which fit is used (class-B fix B-RM). Every L-BFGS-B fit is finished by
a Newton polish to its stationary point (release 2.1, design A-5): the optimiser's stopping point depends on its
path, which rounding-level arithmetic changes (another CPU, BLAS kernel or SIMD path) alter, while the optimum does
not; the EM start is chosen within a band above EM's convergence residual. Homozygotes are
binomial in every candidate: the shared homozygote-overdispersion extension of earlier versions
was removed by the owner's decision (17 Oct 2026; tickets/B-RM).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Mapping

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import logit

from ..config import ReadModelConfig
from .emission import (em_statistics, samples_objective_parallel, samples_objective_serial)


# Model names in candidate order (simplest first).
BINOMIAL = "binomial"
SAMPLE_BINOMIAL = "sample_binomial"
SAMPLE_JOINT = "sample_joint"

# Box constraints of the optimisers (logit error, logit balance, log kappa, mixing logits).
_ERROR_LOGIT_BOUNDS = (-16., -1e-8)
_BALANCE_LOGIT_BOUNDS = (-12., 12.)
_LOG_CONCENTRATION_BOUNDS = (-8., 16.)
_MIXING_BOUNDS = (-25., 25.)
_SAMPLE_FIT_OPTIONS = dict(maxiter=400, ftol=1e-11, gtol=1e-7)
_JOINT_FIT_OPTIONS = dict(maxiter=1200, ftol=1e-12, gtol=1e-9, maxcor=20)
# Observed-Hessian finite-difference step and the positive-definiteness floor.
_HESSIAN_STEP = 1e-4
_HESSIAN_EIGENVALUE_FLOOR = 1e-8
# Samples with an identifiable effect needed for the empirical-Bayes hyperfit.
_HYPERFIT_MINIMUM_SAMPLES = 3
# Newton polish of every L-BFGS-B fit (release 2.1, design A-5 / review R5): central-difference Hessian columns of the
# analytic gradient (relative step), at most this many Newton steps, done when a step moves no coordinate by more
# than the tolerance (relative to max(1, |x|)), or when steps below _POLISH_STAGNATION stop shrinking (a flat
# direction, e.g. log kappa near the binomial limit, locates its optimum only to the gradient's rounding divided by
# its curvature: TroMau chr15 ~1e-10 in log kappa). Measured on actm3 chr13 (sample_joint, 465 parameters): L-BFGS-B
# stops 8e-3 (logits) and 3e-4 (log kappa) short of the optimum and two paths differ by 2e-4; after 3 Newton steps
# both agree to 1e-14 (logits) and 2e-13 (log kappa); on TroMau chr15 log kappa moves 12.49 -> 15.23 in 12 steps.
# A step whose predicted decrease is above the objective's rounding (relative _POLISH_DECREASE) must decrease it
# (backtracking); smaller steps are taken as they are.
_POLISH_ITERATIONS = 40
_POLISH_STEP = 1e-5
_POLISH_TOLERANCE = 1e-13
_POLISH_STAGNATION = 1e-8
_POLISH_DECREASE = 1e-12
# A coordinate whose Hessian diagonal is below this fraction of the largest one has no curvature (it is held).
_POLISH_CURVATURE_FLOOR = 1e-12
# EM starts whose converged log likelihoods lie within this many nats per observation of the best are the same
# optimum (EM stops at a gain of em_tolerance_per_observation per observation; three starts of real contigs end
# within 1e-11 per observation of each other, measured on actm3): the first of them in start order is used.
_EM_START_BAND_PER_OBSERVATION = 1e-7
# Empirical-Bayes hyperfit: the boundary tau = 0 is kept when its profile value is within this relative band of the
# interior optimum's (the two coincide when the optimum is the boundary; a band at that exact tie, design A-7).
_HYPERFIT_BOUNDARY_BAND = 1e-10


@dataclass(frozen=True)
class ReadModelFit:
    """One fitted observation model of a contig's N samples.

    error_logit, balance_logit: float64[N]; e = expit(error_logit) is the homozygote error
    (REF/REF reads ALT with mean e, ALT/ALT reads REF with mean e), balance = expit(balance_logit)
    places the heterozygote ALT mean q = e + (1 - 2e) balance between e and 1 - e (homozygotes
    are binomial). log_concentration: shared heterozygote log kappa (beta-binomial
    heterozygotes), None for binomial heterozygotes. mixing: float64[N, 2] genotype-mixture logits
    relative to genotype 2 (training nuisances; never a genotype prior). success: the optimiser's
    own convergence flag (a diagnostic; selection does not use it). details: JSON-compatible fit
    diagnostics.
    """

    name: str
    error_logit: np.ndarray
    balance_logit: np.ndarray
    log_concentration: float | None
    mixing: np.ndarray
    success: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.error_logit)

    @property
    def heterozygote_inverse_concentration(self) -> float:
        return 0. if self.log_concentration is None else float(np.exp(-self.log_concentration))

    def local(self) -> np.ndarray:
        """float64[N, 4]: error logit, balance logit, mixing logit 0, mixing logit 1."""
        return np.column_stack((self.error_logit, self.balance_logit, self.mixing))

    def error_and_heterozygote(self) -> tuple[np.ndarray, np.ndarray]:
        """(e, q) per sample: the homozygote error and heterozygote ALT mean."""
        error = 1. / (1. + np.exp(-np.asarray(self.error_logit, float)))
        balance = 1. / (1. + np.exp(-np.asarray(self.balance_logit, float)))
        return error, error + (1. - 2. * error) * balance

    def record(self) -> dict:
        """JSON-compatible form for the evidence record and the run record."""
        return dict(name=self.name, error_logit=np.asarray(self.error_logit).tolist(),
                    balance_logit=np.asarray(self.balance_logit).tolist(),
                    log_concentration=self.log_concentration,
                    mixing=np.asarray(self.mixing).tolist(), success=self.success,
                    details=dict(self.details))

    @classmethod
    def from_record(cls, record: Mapping) -> "ReadModelFit":
        return cls(name=record["name"], error_logit=np.asarray(record["error_logit"], float),
                   balance_logit=np.asarray(record["balance_logit"], float),
                   log_concentration=record["log_concentration"],
                   mixing=np.asarray(record["mixing"], float).reshape(-1, 2),
                   success=bool(record["success"]), details=dict(record["details"]))


def fixed_model(error: float, n_samples: int) -> ReadModelFit:
    """The fallback model: binomial, error ``error`` and heterozygote ALT fraction 1/2 for all."""
    return ReadModelFit(name="fixed", error_logit=np.full(n_samples, logit(error)),
                        balance_logit=np.zeros(n_samples), log_concentration=None,
                        mixing=np.zeros((n_samples, 2)), success=True,
                        details=dict(error=float(error)))


# ---------------------------------------------------------------------------------------------
# Objective of the sample models (normalised negative log likelihood and its gradient)
# ---------------------------------------------------------------------------------------------

def _objective(local, het_inverse, hist, depth, alt, mean, variance):
    """(-sum of sample values / scale, -gradients / scale); gradients (n, 5) as sample_objective."""
    kernel = samples_objective_serial if len(hist) == 1 else samples_objective_parallel
    values, gradients = kernel(local, het_inverse, hist, depth, alt, mean, variance)
    scale = max(1., float(hist.sum()))
    return -float(values.sum()) / scale, -gradients / scale


def joint_objective(vector, hist, depth, alt, mean, variance):
    """Per-sample (logit e, logit balance, mixing 0, mixing 1) and one shared log kappa."""
    local = vector[:-1].reshape(len(hist), 4)
    value, gradients = _objective(local, np.exp(-vector[-1]), hist, depth, alt, mean, variance)
    return value, np.r_[gradients[:, :4].ravel(), gradients[:, 4].sum()]


def score(hist, depth, alt, fit: ReadModelFit) -> float:
    """Unpenalised log likelihood of a histogram under a fit (its training mixture nuisances)."""
    value, _ = _objective(fit.local(), fit.heterozygote_inverse_concentration, hist, depth, alt,
                          np.zeros(2), np.zeros(2))
    return -value * max(1., float(hist.sum()))


# ---------------------------------------------------------------------------------------------
# Newton polish (release 2.1): every fit is moved to its optimum, independently of the optimiser's path
# ---------------------------------------------------------------------------------------------

def _held(x, gradient, lower, upper):
    """Coordinates that do not move: fixed (lower == upper) or at a bound with the gradient pushing outward."""
    return (lower == upper) | ((x <= lower) & (gradient > 0)) | ((x >= upper) & (gradient < 0))


def _polish_step(evaluate, x, f, gradient, step, free, lower, upper):
    """Apply a Newton step on the ``free`` coordinates (clipped to the bounds); a step whose predicted decrease is
    above the objective's rounding must decrease it (halving up to 12 times). Returns (x, f, gradient, moved) or None
    when no acceptable step exists."""
    predicted = -0.5 * float(gradient[free] @ step)
    alpha = 1.
    for _ in range(13):
        candidate = x.copy()
        candidate[free] = np.clip(x[free] + alpha * step, lower[free], upper[free])
        value, slope = evaluate(candidate)
        if not np.isfinite(value):
            alpha /= 2.
            continue
        if predicted <= _POLISH_DECREASE * max(1., abs(f)) or value <= f:
            moved = float(np.max(np.abs(candidate - x) / np.maximum(1., np.abs(x)), initial=0.))
            return candidate, value, slope, moved
        alpha /= 2.
    return None


def _polish(evaluate, x, lower, upper):
    """Newton polish of a small fit: minimise evaluate(x) -> (f, gradient) from ``x`` (near the optimum), holding
    fixed and bound-active coordinates. Returns (x, iterations, converged); a Hessian that is not positive definite
    or a failed step keeps the last point (converged False)."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    x = np.clip(np.asarray(x, float).copy(), lower, upper)
    f, gradient = evaluate(x)
    previous = np.inf
    for iteration in range(1, _POLISH_ITERATIONS + 1):
        free = np.flatnonzero(~_held(x, gradient, lower, upper))
        if not len(free):
            return x, iteration - 1, True
        hessian = np.empty((len(free), len(free)))
        for column, j in enumerate(free):
            h = _POLISH_STEP * max(1., abs(x[j]))
            up, down = x.copy(), x.copy()
            up[j] += h
            down[j] -= h
            hessian[:, column] = (evaluate(up)[1][free] - evaluate(down)[1][free]) / (2. * h)
        hessian = (hessian + hessian.T) / 2.
        try:
            factor = cho_factor(hessian)
        except (LinAlgError, ValueError):
            # A coordinate without curvature (e.g. the balance of a sample with no heterozygote mass) does not move
            # the objective: hold it and polish the others.
            diagonal = np.diag(hessian)
            kept = diagonal > _POLISH_CURVATURE_FLOOR * max(float(diagonal.max(initial=0.)), 0.)
            if kept.all() or not kept.any():
                return x, iteration - 1, False
            free, hessian = free[kept], hessian[np.ix_(kept, kept)]
            try:
                factor = cho_factor(hessian)
            except (LinAlgError, ValueError):
                return x, iteration - 1, False
        result = _polish_step(evaluate, x, f, gradient, -cho_solve(factor, gradient[free]), free, lower, upper)
        if result is None:
            return x, iteration - 1, False
        x, f, gradient, moved = result
        if moved <= _POLISH_TOLERANCE or (moved <= _POLISH_STAGNATION and moved > previous / 4.):
            return x, iteration, True
        previous = moved
    return x, _POLISH_ITERATIONS, False


def _polish_joint(vector, hist, depth, alt, mean, variance, lower, upper):
    """Newton polish of the joint fit (N x 4 sample effects and one shared log kappa).

    The Hessian is an arrowhead: one 4 x 4 block per sample (samples are independent given kappa), each coupled to log
    kappa. Central differences of the per-sample gradient rows give every block from 4 x 2 evaluations (one effect of
    every sample moved at once) and the kappa column from 2; the step solves the arrowhead system through the Schur
    complement of log kappa. Returns (vector, iterations, converged).
    """
    n = len(hist)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    x = np.clip(np.asarray(vector, float).copy(), lower, upper)
    scale = max(1., float(hist.sum()))
    kernel = samples_objective_serial if n == 1 else samples_objective_parallel

    def evaluate(point):
        return joint_objective(point, hist, depth, alt, mean, variance)

    def rows(local, log_kappa):
        return -kernel(local, np.exp(-log_kappa), hist, depth, alt, mean, variance)[1] / scale

    f, gradient = evaluate(x)
    previous = np.inf
    for iteration in range(1, _POLISH_ITERATIONS + 1):
        held = _held(x, gradient, lower, upper)
        if held.all():
            return x, iteration - 1, True
        local, log_kappa = x[:-1].reshape(n, 4), x[-1]
        blocks = np.empty((n, 4, 4))
        coupling = np.empty((n, 4))
        for j in range(4):
            h = _POLISH_STEP * np.maximum(1., np.abs(local[:, j]))
            up, down = local.copy(), local.copy()
            up[:, j] += h
            down[:, j] -= h
            difference = (rows(up, log_kappa) - rows(down, log_kappa)) / (2. * h[:, None])
            blocks[:, :, j] = difference[:, :4]
            coupling[:, j] = difference[:, 4]
        h = _POLISH_STEP * max(1., abs(log_kappa))
        difference = (rows(local, log_kappa + h) - rows(local, log_kappa - h)) / (2. * h)
        coupling = (coupling + difference[:, :4]) / 2.
        corner = float(difference[:, 4].sum())
        blocks = (blocks + blocks.transpose(0, 2, 1)) / 2.
        # Held coordinates: identity rows/columns, zero gradient and coupling (their step is exactly 0).
        held_local = held[:-1].reshape(n, 4)
        local_gradient = np.where(held_local, 0., gradient[:-1].reshape(n, 4))
        coupling = np.where(held_local, 0., coupling)
        blocks = np.where(held_local[:, :, None] | held_local[:, None, :], 0., blocks)
        blocks[:, np.arange(4), np.arange(4)] += held_local
        try:
            np.linalg.cholesky(blocks)
            solved_gradient = np.linalg.solve(blocks, local_gradient[..., None])[..., 0]
            solved_coupling = np.linalg.solve(blocks, coupling[..., None])[..., 0]
        except np.linalg.LinAlgError:
            return x, iteration - 1, False
        kappa_step = 0.
        if not held[-1]:
            schur = corner - float(np.sum(coupling * solved_coupling))
            if not schur > 0.:
                return x, iteration - 1, False
            kappa_step = -(float(gradient[-1]) - float(np.sum(coupling * solved_gradient))) / schur
        step = np.r_[(-solved_gradient - solved_coupling * kappa_step).ravel(), kappa_step]
        free = np.flatnonzero(~held)
        result = _polish_step(evaluate, x, f, gradient, step[free], free, lower, upper)
        if result is None:
            return x, iteration - 1, False
        x, f, gradient, moved = result
        if moved <= _POLISH_TOLERANCE or (moved <= _POLISH_STAGNATION and moved > previous / 4.):
            return x, iteration, True
        previous = moved
    return x, _POLISH_ITERATIONS, False


# ---------------------------------------------------------------------------------------------
# Shared binomial model (EM)
# ---------------------------------------------------------------------------------------------

def _em(hist, depth, alt, initial, config: ReadModelConfig):
    """EM for P(ALT count | depth, sample) with ordered components e < q < 1 - e."""
    error, heterozygote = initial
    mixture = np.full((len(hist), 3), 1 / 3.)
    sample_weight = hist.sum(axis=1)
    scale = max(1., float(sample_weight.sum()))
    old = -np.inf
    last_gain = None
    for iteration in range(config.em_maximum_iterations):
        values, mass, moments = em_statistics(hist, depth, alt, error, heterozygote, mixture)
        value = float(values.sum())  # the binomial coefficient cancels in comparisons
        if not np.isfinite(value) or value < old - 1e-7 * scale:
            raise RuntimeError("read-model EM likelihood decreased or became non-finite")
        if iteration:
            last_gain = value - old
            if last_gain <= config.em_tolerance_per_observation * scale:
                break
        old = value
        mixture = np.divide(mass, sample_weight[:, None], out=np.full_like(mass, 1 / 3.),
                            where=sample_weight[:, None] > 0)
        totals = moments.sum(axis=0)
        error = float(totals[0] / max(totals[1], 1e-300))
        heterozygote = float(totals[2] / max(totals[3], 1e-300))
        if not (0 < error < heterozygote < 1 - error < 1):
            raise ValueError("EM left the ordered-component domain")
    else:
        raise RuntimeError("read-model EM did not converge")
    return dict(error=error, heterozygote_alt=heterozygote, mixture=mixture, log_likelihood=value,
                iterations=iteration + 1, last_gain=last_gain)


def fit_shared(hist, depth, alt, config: ReadModelConfig) -> ReadModelFit:
    """The shared binomial model: the first EM start (in start order) whose converged log likelihood is within
    _EM_START_BAND_PER_OBSERVATION per observation of the best (starts that reach the same optimum differ only by EM's
    convergence residual, which rounding-level arithmetic changes reorder; 2.0 took the strict maximum)."""
    candidates, failed = [], []
    for initial in config.em_starts:
        try:
            candidates.append(_em(hist, depth, alt, initial, config))
        except (ValueError, RuntimeError) as error:
            failed.append(str(error))
    if not candidates:
        raise RuntimeError(f"No shared-binomial EM fit converged: {failed}")
    top = max(value["log_likelihood"] for value in candidates)
    band = _EM_START_BAND_PER_OBSERVATION * max(1., float(hist.sum()))
    best = next(value for value in candidates if value["log_likelihood"] >= top - band)
    error, heterozygote = best["error"], best["heterozygote_alt"]
    mixture = np.maximum(best["mixture"], 1e-300)
    n = len(hist)
    return ReadModelFit(
        name=BINOMIAL, error_logit=np.full(n, logit(error)),
        balance_logit=np.full(n, logit((heterozygote - error) / (1 - 2 * error))),
        log_concentration=None,
        mixing=np.log(mixture[:, :2]) - np.log(mixture[:, 2, None]), success=True,
        details=dict(message="shared-binomial EM", iterations=best["iterations"],
                     failed_starts=failed, train_log_likelihood=best["log_likelihood"],
                     starts_in_band=sum(value["log_likelihood"] >= top - band for value in candidates)))


# ---------------------------------------------------------------------------------------------
# Pooled sample effects (empirical Bayes) and the joint concentration
# ---------------------------------------------------------------------------------------------

def _fit_sample(task, depth, alt, shared: ReadModelFit, prior=None):
    """One sample's binomial fit from the shared model; returns (vector, variances, success, polished).

    vector = (logit e, logit balance, mixing 0, mixing 1). Without a prior, the variances of the
    two logits come from the observed Hessian (genotype-mixture uncertainty profiled out); with
    a prior (mean, variance), effects of zero prior variance stay at the prior mean.
    """
    index, counts = task
    vector = shared.local()[index].copy()
    if counts.sum() == 0:
        # With no training reads the normal prior is the entire objective: use its mode,
        # including exactly pooled (zero-variance) effects.
        if prior is not None:
            vector[:2] = prior[0]
        return vector, np.full(2, np.inf), True, True
    free = np.ones(len(vector), bool)
    mean, variance = (np.zeros(2), np.zeros(2)) if prior is None else prior
    if prior is not None:
        vector[:2] = prior[0]
        free[:2] = prior[1] > 0
    bounds = [_ERROR_LOGIT_BOUNDS, _BALANCE_LOGIT_BOUNDS, _MIXING_BOUNDS, _MIXING_BOUNDS]
    histogram = counts[None]

    def reduced(x):
        current = vector.copy()
        current[free] = x
        value, gradients = _objective(current[None], 0., histogram, depth, alt, mean, variance)
        return value, gradients[0, :4][free]

    active_bounds = [bound for bound, active in zip(bounds, free) if active]
    fitted = minimize(reduced, vector[free], jac=True, method="L-BFGS-B", bounds=active_bounds,
                      options=_SAMPLE_FIT_OPTIONS)
    # Newton polish to the optimum itself (release 2.1): the result no longer depends on L-BFGS-B's path.
    polished, _, polish_converged = _polish(reduced, fitted.x, [bound[0] for bound in active_bounds],
                                            [bound[1] for bound in active_bounds])
    vector[free] = polished
    variances = np.full(2, np.inf)
    if prior is None:
        # A small observed Hessian profiles out the sample's genotype-mixture uncertainty.
        indices = np.flatnonzero(free)
        hessian = np.empty((len(indices), len(indices)))
        zeros = np.zeros(2)
        for column, j in enumerate(indices):
            upper, lower = vector.copy(), vector.copy()
            upper[j] += _HESSIAN_STEP
            lower[j] -= _HESSIAN_STEP
            gu = _objective(upper[None], 0., histogram, depth, alt, zeros, zeros)[1][0, :4]
            gl = _objective(lower[None], 0., histogram, depth, alt, zeros, zeros)[1][0, :4]
            hessian[:, column] = (gu[free] - gl[free]) / (2 * _HESSIAN_STEP) * counts.sum()
        hessian = (hessian + hessian.T) / 2
        if np.linalg.eigvalsh(hessian).min() > _HESSIAN_EIGENVALUE_FLOOR:
            variances = np.diag(np.linalg.inv(hessian))[:2]
    return vector, variances, bool(fitted.success), bool(polish_converged)


def _normal_hyperfit(values, variances, fallback):
    """Normal measurement-error hyperparameters (mean, tau) by profile likelihood."""
    valid = np.isfinite(variances) & (variances > 0) & np.isfinite(values)
    if valid.sum() < _HYPERFIT_MINIMUM_SAMPLES:
        return float(fallback), 0.
    y, v = values[valid], variances[valid]

    def profile(tau):
        weights = 1 / (v + tau)
        mu = np.sum(weights * y) / weights.sum()
        return .5 * np.sum(np.log(v + tau) + (y - mu) ** 2 / (v + tau)), mu

    def slope(tau):
        """d profile / d tau at the profiled mean (envelope theorem)."""
        weights = 1 / (v + tau)
        mu = np.sum(weights * y) / weights.sum()
        return .5 * float(np.sum(weights - (y - mu) ** 2 * weights ** 2))

    ceiling = max(1., 4 * float(np.var(y)))
    optimum = minimize_scalar(lambda t: profile(t)[0], bounds=(0., ceiling), method="bounded")
    tau = _profile_root(slope, float(optimum.x), ceiling)
    band = _HYPERFIT_BOUNDARY_BAND * max(1., abs(float(profile(tau)[0])))
    tau = 0. if profile(0.)[0] <= profile(tau)[0] + band else tau
    return float(profile(tau)[1]), tau


def _profile_root(slope, start, ceiling):
    """The stationary point of the hyperfit profile near Brent's bounded minimum ``start`` (xatol 1e-5): the root of
    its derivative, bracketed around ``start`` and solved to rounding precision (release 2.1), so tau does not depend
    on Brent's path; ``start`` when no sign change brackets it (a minimum at a bound)."""
    if slope(ceiling) < 0.:
        return float(ceiling)
    width = 1e-4 * max(1., start)
    for _ in range(40):
        low, high = max(0., start - width), min(ceiling, start + width)
        below, above = slope(low), slope(high)
        if below <= 0. <= above:
            if below == 0.:
                return float(low)
            if above == 0.:
                return float(high)
            return float(brentq(slope, low, high, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=200))
        if low == 0. and high == ceiling:
            break
        width *= 4.
    return float(start)


def fit_pooled_samples(hist, depth, alt, shared: ReadModelFit, threads: int) -> ReadModelFit:
    """Per-sample binomial effects shrunk by the empirical-Bayes prior (one sample per thread)."""
    tasks = list(enumerate(hist))
    worker = partial(_fit_sample, depth=depth, alt=alt, shared=shared)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        preliminary = list(pool.map(worker, tasks))
    estimates = np.asarray([x[0] for x in preliminary])
    variances = np.asarray([x[1] for x in preliminary])
    shared_local = shared.local()
    hyper = [_normal_hyperfit(estimates[:, j], variances[:, j], shared_local[0, j]) for j in range(2)]
    mean, variance = np.asarray(hyper).T
    worker = partial(worker, prior=(mean, variance))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        final = list(pool.map(worker, tasks))
    fitted = np.asarray([x[0] for x in final])
    return ReadModelFit(
        name=SAMPLE_BINOMIAL, error_logit=fitted[:, 0], balance_logit=fitted[:, 1],
        log_concentration=None, mixing=fitted[:, 2:],
        success=all(x[2] for x in final),
        details=dict(sample_fit_failures=sum(not x[2] for x in final),
                     polish_failures=sum(not x[3] for x in (*preliminary, *final)),
                     identifiable_sample_effects=np.isfinite(variances).sum(axis=0).tolist(),
                     hyper_mean=mean.tolist(), hyper_variance=variance.tolist(),
                     approximation="diagonal Laplace empirical Bayes"))


def _local_bounds(mean, variance, n):
    """Per-sample box constraints; an effect with zero prior variance is fixed at its mean."""
    per_sample = [_ERROR_LOGIT_BOUNDS, _BALANCE_LOGIT_BOUNDS, _MIXING_BOUNDS, _MIXING_BOUNDS]
    for j in range(2):
        if variance[j] == 0:
            per_sample[j] = (mean[j], mean[j])
    return per_sample * n


def fit_joint(hist, depth, alt, pooled: ReadModelFit, config: ReadModelConfig) -> ReadModelFit:
    """Pooled sample effects refitted with one shared heterozygote concentration.

    Starts from the pooled sample-binomial fit and kappa = ``initial_heterozygote_concentration``;
    the empirical-Bayes prior of the pooled fit is kept. Uses the caller's Numba team.
    """
    n = len(hist)
    mean = np.asarray(pooled.details["hyper_mean"])
    variance = np.asarray(pooled.details["hyper_variance"])
    initial = np.r_[pooled.local().ravel(), np.log(config.initial_heterozygote_concentration)]
    bounds = _local_bounds(mean, variance, n) + [_LOG_CONCENTRATION_BOUNDS]
    result = minimize(joint_objective, initial, args=(hist, depth, alt, mean, variance), jac=True,
                      method="L-BFGS-B", bounds=bounds, options=_JOINT_FIT_OPTIONS)
    # Newton polish to the optimum itself (release 2.1): the result no longer depends on L-BFGS-B's path.
    polished, steps, converged = _polish_joint(result.x, hist, depth, alt, mean, variance,
                                               [bound[0] for bound in bounds], [bound[1] for bound in bounds])
    local = polished[:-1].reshape(n, 4)
    return ReadModelFit(
        name=SAMPLE_JOINT, error_logit=local[:, 0], balance_logit=local[:, 1],
        log_concentration=float(polished[-1]), mixing=local[:, 2:],
        success=bool(result.success),
        details=dict(message=str(result.message), iterations=int(result.nit),
                     hyper_mean=mean.tolist(), hyper_variance=variance.tolist(),
                     polish_steps=int(steps), polish_converged=bool(converged),
                     polish_shift=float(np.max(np.abs(polished - result.x), initial=0.))))
