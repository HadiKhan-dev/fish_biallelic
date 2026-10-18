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
(``ReadModelFit.success``) and never decide which fit is used (class-B fix B-RM). Homozygotes are
binomial in every candidate: the shared homozygote-overdispersion extension of earlier versions
was removed by the owner's decision (17 Oct 2026; tickets/B-RM).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize, minimize_scalar
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
    """The shared binomial model: the best of the EM starts that converge."""
    candidates, failed = [], []
    for initial in config.em_starts:
        try:
            candidates.append(_em(hist, depth, alt, initial, config))
        except (ValueError, RuntimeError) as error:
            failed.append(str(error))
    if not candidates:
        raise RuntimeError(f"No shared-binomial EM fit converged: {failed}")
    best = max(candidates, key=lambda value: value["log_likelihood"])
    error, heterozygote = best["error"], best["heterozygote_alt"]
    mixture = np.maximum(best["mixture"], 1e-300)
    n = len(hist)
    return ReadModelFit(
        name=BINOMIAL, error_logit=np.full(n, logit(error)),
        balance_logit=np.full(n, logit((heterozygote - error) / (1 - 2 * error))),
        log_concentration=None,
        mixing=np.log(mixture[:, :2]) - np.log(mixture[:, 2, None]), success=True,
        details=dict(message="shared-binomial EM", iterations=best["iterations"],
                     failed_starts=failed, train_log_likelihood=best["log_likelihood"]))


# ---------------------------------------------------------------------------------------------
# Pooled sample effects (empirical Bayes) and the joint concentration
# ---------------------------------------------------------------------------------------------

def _fit_sample(task, depth, alt, shared: ReadModelFit, prior=None):
    """One sample's binomial fit from the shared model; returns (vector, variances, success).

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
        return vector, np.full(2, np.inf), True
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

    fitted = minimize(reduced, vector[free], jac=True, method="L-BFGS-B",
                      bounds=[bound for bound, active in zip(bounds, free) if active],
                      options=_SAMPLE_FIT_OPTIONS)
    vector[free] = fitted.x
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
    return vector, variances, bool(fitted.success)


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

    ceiling = max(1., 4 * float(np.var(y)))
    optimum = minimize_scalar(lambda t: profile(t)[0], bounds=(0., ceiling), method="bounded")
    tau = 0. if profile(0.)[0] <= optimum.fun else float(optimum.x)
    return float(profile(tau)[1]), tau


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
    result = minimize(joint_objective, initial, args=(hist, depth, alt, mean, variance), jac=True,
                      method="L-BFGS-B",
                      bounds=_local_bounds(mean, variance, n) + [_LOG_CONCENTRATION_BOUNDS],
                      options=_JOINT_FIT_OPTIONS)
    local = result.x[:-1].reshape(n, 4)
    return ReadModelFit(
        name=SAMPLE_JOINT, error_logit=local[:, 0], balance_logit=local[:, 1],
        log_concentration=float(result.x[-1]), mixing=local[:, 2:],
        success=bool(result.success),
        details=dict(message=str(result.message), iterations=int(result.nit),
                     hyper_mean=mean.tolist(), hyper_variance=variance.tolist()))
