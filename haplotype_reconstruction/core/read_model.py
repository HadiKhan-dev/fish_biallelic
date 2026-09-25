"""Raw-AD calibration with pooled sample effects.

Homozygotes remain binomial; heterozygotes may be beta-binomial. Sample
genotype mixtures are arbitrary nuisance distributions (not HWE). They are
used only for fitting/predictive scoring and are NOT returned as raw GL priors.

Sample logit-error and balance effects are pooled by an empirical-Bayes normal
measurement-error approximation. Their uncertainty comes from the observed
profile Hessian of the per-sample likelihood. Hyperparameters use only training
data. This is a Laplace/diagonal approximation, not exact Bayes.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit, logsumexp

from .read_kernels import objective


def emission(parameters, depth, alt, dispersed):
    """Log kernels and derivatives; binomial coefficient cancels in comparisons."""
    parameters = np.asarray(parameters)
    e = expit(parameters[:, 0])
    balance = expit(parameters[:, 1])
    q = e + (1-2*e)*balance
    d, a = depth[None, :], alt[None, :]
    e, q, balance = e[:, None], q[:, None], balance[:, None]
    homo0 = a*np.log(e) + (d-a)*np.log1p(-e)
    homo2 = (d-a)*np.log(e) + a*np.log1p(-e)
    if dispersed:
        kappa = np.exp(parameters[:, 2, None])
        # Integer read depths permit rising-product sums. Unlike subtracting
        # large log-gamma values, these remain accurate at the binomial limit.
        j = np.arange(int(np.max(depth)), dtype=float)[None, :]/kappa
        def prefix(values):
            return np.column_stack((np.zeros(len(parameters)), np.cumsum(values, axis=1)))
        ai, ri, di = alt.astype(int), (depth-alt).astype(int), depth.astype(int)
        hetero = (prefix(np.log(q+j))[:, ai] + prefix(np.log(1-q+j))[:, ri]
                  - prefix(np.log1p(j))[:, di])
        dq = prefix(1/(q+j))[:, ai]-prefix(1/(1-q+j))[:, ri]
        dk = (-prefix(j/(q+j))[:, ai]-prefix(j/(1-q+j))[:, ri]
              + prefix(j/(1+j))[:, di])
    else:
        hetero = a*np.log(q)+(d-a)*np.log1p(-q)
        dq = a/q-(d-a)/(1-q)
        dk = np.zeros_like(hetero)
    # e affects heterozygote mean through the ordered-component transform.
    de_hetero = dq*(1-2*balance)*e*(1-e)
    db_hetero = dq*(1-2*e)*balance*(1-balance)
    return np.stack((homo0, hetero, homo2), axis=2), (
        a-d*e, de_hetero, (d-a)-d*e, db_hetero, dk)


def fit_shared(hist, depth, alt, config):
    """Fit the nested shared-binomial model from multiple ordered EM starts."""
    from .read_calibration import fit
    candidates, failed = [], []
    for initial in config.initial_parameters:
        try:
            candidates.append(fit(hist, depth, alt, initial, config))
        except (ValueError, RuntimeError) as error:
            failed.append(str(error))
    if not candidates:
        raise RuntimeError(f"No shared-binomial EM fit converged: {failed}")
    best = max(candidates, key=lambda value: value["log_likelihood"])
    error, q = best["error"], best["heterozygote_alt"]
    parameters = [logit(error), logit((q-error)/(1-2*error))]
    mixture = np.maximum(best["mixture"], 1e-300)
    return dict(parameters=np.tile(parameters, (len(hist), 1)),
                mixing=np.log(mixture[:, :2])-np.log(mixture[:, 2, None]),
                dispersed=False, success=True, message="shared-binomial EM",
                iterations=best["iterations"], failed_starts=failed,
                train_log_likelihood=best["log_likelihood"])


def _fit_sample(task, depth, alt, shared, prior=None):
    index, counts = task
    dispersed = shared['dispersed']
    pcount = 3 if dispersed else 2
    vector = np.r_[shared['parameters'][index], shared['mixing'][index]]
    if counts.sum() == 0:
        # With no training reads the normal prior is the entire objective:
        # use its mode, including exactly pooled (zero-variance) effects.
        if prior is not None:
            vector[:2] = prior[0]
        return vector, np.full(2, np.inf), True
    # Dispersion is pooled, not freely fitted for every sample.
    free = np.ones(len(vector), bool)
    if dispersed:
        free[2] = False
    if prior is not None:
        vector[:2] = prior[0]
        free[:2] = prior[1] > 0
    bounds = [(-16., -1e-8), (-12., 12.)] + ([(-8., 16.)] if dispersed else [])
    bounds += [(-25., 25.)]*2

    def reduced(x):
        current = vector.copy()
        current[free] = x
        value, grad = objective(current, counts[None], depth, alt, dispersed, False, prior)
        return value, grad[free]

    fitted = minimize(reduced, vector[free], jac=True, method='L-BFGS-B',
        bounds=[bound for bound, active in zip(bounds, free) if active],
        options=dict(maxiter=400, ftol=1e-11, gtol=1e-7))
    vector[free] = fitted.x
    variances = np.full(2, np.inf)
    if prior is None:
        # Small observed Hessian profiles out sample genotype-mixture uncertainty.
        indices = np.flatnonzero(free)
        hessian = np.empty((len(indices), len(indices)))
        step = 1e-4
        for column, j in enumerate(indices):
            upper, lower = vector.copy(), vector.copy()
            upper[j] += step
            lower[j] -= step
            gu = objective(upper, counts[None], depth, alt, dispersed, False)[1]
            gl = objective(lower, counts[None], depth, alt, dispersed, False)[1]
            hessian[:, column] = (gu[free]-gl[free])/(2*step)*counts.sum()
        hessian = (hessian+hessian.T)/2
        if np.linalg.eigvalsh(hessian).min() > 1e-8:
            variances = np.diag(np.linalg.inv(hessian))[:2]
    return vector, variances, bool(fitted.success)


def _normal_hyperfit(values, variances, fallback):
    valid = np.isfinite(variances) & (variances > 0) & np.isfinite(values)
    if valid.sum() < 3:
        return float(fallback), 0.
    y, v = values[valid], variances[valid]

    def profile(tau):
        weights = 1/(v+tau)
        mu = np.sum(weights*y)/weights.sum()
        return .5*np.sum(np.log(v+tau)+(y-mu)**2/(v+tau)), mu

    ceiling = max(1., 4*float(np.var(y)))
    opt = minimize_scalar(lambda t: profile(t)[0], bounds=(0., ceiling), method='bounded')
    tau = 0. if profile(0.)[0] <= opt.fun else float(opt.x)
    return float(profile(tau)[1]), tau


def fit_pooled_samples(hist, depth, alt, shared, threads):
    tasks = list(enumerate(hist))
    worker = partial(_fit_sample, depth=depth, alt=alt, shared=shared)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        preliminary = list(pool.map(worker, tasks))
    estimates = np.asarray([x[0] for x in preliminary])
    variances = np.asarray([x[1] for x in preliminary])
    hyper = [_normal_hyperfit(estimates[:, j], variances[:, j], shared['parameters'][0, j])
             for j in range(2)]
    mean, variance = np.asarray(hyper).T
    worker = partial(worker, prior=(mean, variance))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        final = list(pool.map(worker, tasks))
    fitted = np.asarray([x[0] for x in final])
    pcount = shared['parameters'].shape[1]
    return dict(parameters=fitted[:, :pcount], mixing=fitted[:, pcount:],
        dispersed=shared['dispersed'], success=all(x[2] for x in final),
        sample_fit_failures=sum(not x[2] for x in final),
        identifiable_sample_effects=np.isfinite(variances).sum(axis=0).tolist(),
        hyper_mean=mean.tolist(), hyper_variance=variance.tolist(),
        approximation='diagonal Laplace empirical Bayes; pooled dispersion')


def score(hist, depth, alt, model):
    logits = np.column_stack((model['mixing'], np.zeros(len(hist))))
    lm = logits-logsumexp(logits, axis=1, keepdims=True)
    kernel, _ = emission(model['parameters'], depth, alt, model['dispersed'])
    return float(np.sum(hist*logsumexp(kernel+lm[:, None, :], axis=2)))



def joint_objective(vector, hist, depth, alt, mean, variance):
    """One concentration is shared; reduce only its samplewise derivatives."""
    n = len(hist)
    local = vector[:-1].reshape(n, 4)
    matrix = np.column_stack((local[:, :2], np.full(n, vector[-1]), local[:, 2:]))
    value, gradient = objective(
        matrix.ravel(), hist, depth, alt, True, False, (mean, variance))
    gradient = gradient.reshape(n, 5)
    return value, np.r_[gradient[:, [0, 1, 3, 4]].ravel(), gradient[:, 2].sum()]


def fit_joint(hist, depth, alt, binomial, threads):
    """Fit pooled sample effects and one shared heterozygote concentration."""
    from .parallel import numba_thread_scope

    pooled = fit_pooled_samples(hist, depth, alt, binomial, threads)
    n = len(hist)
    mean = np.asarray(pooled["hyper_mean"])
    variance = np.asarray(pooled["hyper_variance"])
    initial = np.r_[
        np.column_stack((pooled["parameters"], pooled["mixing"])).ravel(),
        np.log(40.)]
    per_sample = [(-16., -1e-8), (-12., 12.), (-25., 25.), (-25., 25.)]
    for j in range(2):
        if variance[j] == 0:
            per_sample[j] = (mean[j], mean[j])
    bounds = per_sample * n + [(-8., 16.)]
    # The sample fits above have one native thread per Python worker. Only
    # this joint objective uses a Numba team, bounded by the supplied budget.
    with numba_thread_scope(threads):
        result = minimize(
            joint_objective, initial, args=(hist, depth, alt, mean, variance),
            jac=True, method="L-BFGS-B", bounds=bounds,
            options=dict(maxiter=1200, ftol=1e-12, gtol=1e-9, maxcor=20))
    local = result.x[:-1].reshape(n, 4)
    fitted = dict(
        parameters=np.column_stack((local[:, :2], np.full(n, result.x[-1]))),
        mixing=local[:, 2:], dispersed=True, success=bool(result.success),
        message=str(result.message), iterations=int(result.nit),
        hyper_mean=mean.tolist(), hyper_variance=variance.tolist(),
        initialization="pooled sample-binomial; joint concentration starts at 40",
        approximation="diagonal empirical-Bayes hyperparameters from nested binomial model")
    return pooled, fitted
