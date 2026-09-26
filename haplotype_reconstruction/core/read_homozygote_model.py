"""Nested beta-binomial calibration of homozygote read overdispersion.

The null is the production sample-joint model (binomial homozygotes,
beta-binomial heterozygotes). The extension adds one shared homozygote
intraclass correlation rho; rho=0 recovers the null exactly. REF/ALT error
means remain symmetric and sample e/balance/mixture nuisances are refitted.
Train-only empirical-Bayes hyperparameters are shared by both fits. No founder
count, haplotype panel, genotype truth or pedigree enters this objective.
"""
import math
import time
import numpy as np
from numba import njit, prange
from scipy.optimize import minimize
from . import read_model, parallel
from scipy.special import expit


@njit(cache=True)
def _prefix(mean, inverse_concentration, maximum_depth):
    out = np.zeros((8, maximum_depth + 1))
    for j in range(maximum_depth):
        shift = j * inverse_concentration
        first, second, total = mean + shift, 1. - mean + shift, 1. + shift
        out[0, j+1] = out[0, j] + math.log(first)
        out[1, j+1] = out[1, j] + math.log(second)
        out[2, j+1] = out[2, j] + math.log1p(shift)
        out[3, j+1] = out[3, j] + 1. / first
        out[4, j+1] = out[4, j] + 1. / second
        out[5, j+1] = out[5, j] + j / first
        out[6, j+1] = out[6, j] + j / second
        out[7, j+1] = out[7, j] + j / total
    return out


@njit(cache=True, nogil=True)
def _sample(local, log_hetero_concentration, homo_rho, counts, depth, alt, mean, variance):
    e = 1. / (1. + math.exp(-local[0]))
    balance = 1. / (1. + math.exp(-local[1]))
    q = e + (1. - 2. * e) * balance
    high = max(local[2], local[3], 0.)
    normalizer = high + math.log(math.exp(local[2]-high) + math.exp(local[3]-high) + math.exp(-high))
    log_mix = np.array([local[2]-normalizer, local[3]-normalizer, -normalizer])
    het_inverse = math.exp(-log_hetero_concentration)
    hom_inverse = homo_rho / (1. - homo_rho)
    hom_derivative = 1. / ((1. - homo_rho) ** 2)
    maximum_depth = int(np.max(depth))
    het = _prefix(q, het_inverse, maximum_depth)
    hom = _prefix(e, hom_inverse, maximum_depth)
    # Gradient order is e-logit, balance-logit, mix0, mix1, log-kappa-het, rho-hom.
    gradient = np.zeros(6)
    value, weight, mass0, mass1 = 0., 0., 0., 0.
    for cell in range(len(counts)):
        count = counts[cell]
        if count == 0.:
            continue
        d, a = int(depth[cell]), int(alt[cell])
        r = d - a
        h0 = hom[0,a] + hom[1,r] - hom[2,d]
        h1 = het[0,a] + het[1,r] - het[2,d]
        h2 = hom[0,r] + hom[1,a] - hom[2,d]
        l0, l1, l2 = h0 + log_mix[0], h1 + log_mix[1], h2 + log_mix[2]
        high = max(l0, l1, l2)
        z = high + math.log(math.exp(l0-high) + math.exp(l1-high) + math.exp(l2-high))
        r0, r1, r2 = count * math.exp(l0-z), count * math.exp(l1-z), count * math.exp(l2-z)
        value += count * z
        weight += count
        mass0 += r0
        mass1 += r1
        de0 = hom[3,a] - hom[4,r]
        de2 = hom[3,r] - hom[4,a]
        dq = het[3,a] - het[4,r]
        gradient[0] += (r0 * de0 + r2 * de2 + r1 * dq * (1.-2.*balance)) * e * (1.-e)
        gradient[1] += r1 * dq * (1.-2.*e) * balance * (1.-balance)
        gradient[4] -= r1 * het_inverse * (het[5,a] + het[6,r] - het[7,d])
        gradient[5] += hom_derivative * (
            r0 * (hom[5,a] + hom[6,r] - hom[7,d])
            + r2 * (hom[5,r] + hom[6,a] - hom[7,d]))
    gradient[2] = mass0 - weight * math.exp(log_mix[0])
    gradient[3] = mass1 - weight * math.exp(log_mix[1])
    for j in range(2):
        if variance[j] > 0.:
            delta = local[j] - mean[j]
            value -= delta * delta / (2. * variance[j])
            gradient[j] -= delta / variance[j]
    return value, gradient


@njit(cache=True, parallel=True, nogil=True)
def _many(local, hetero_concentration, homo_rho, hist, depth, alt, mean, variance):
    values = np.empty(len(hist))
    gradients = np.empty((len(hist), 6))
    for sample in prange(len(hist)):
        values[sample], gradients[sample] = _sample(
            local[sample], hetero_concentration, homo_rho,
            hist[sample], depth, alt, mean, variance)
    return values, gradients


def objective(vector, hist, depth, alt, mean, variance):
    local = vector[:-2].reshape(len(hist), 4)
    values, gradients = _many(local, vector[-2], vector[-1], hist, depth, alt, mean, variance)
    scale = max(1., float(hist.sum()))
    gradient = np.r_[gradients[:, :4].ravel(), gradients[:, 4:].sum(axis=0)]
    return -float(values.sum()) / scale, -gradient / scale


def _vector(model, homo_rho=0.):
    return np.r_[np.column_stack((model['parameters'][:, :2], model['mixing'])).ravel(),
                 model['parameters'][0, 2], homo_rho]


def score(hist, depth, alt, model):
    """Unpenalized prediction using training-fitted genotype-mixture nuisances."""
    vector = _vector(model, model.get('homo_rho', 0.))
    value, _ = objective(vector, hist, depth, alt, np.zeros(2), np.zeros(2))
    return -value * max(1., float(hist.sum()))


def fit_nested(hist, depth, alt, joint, *, threads, homo_starts=(0., .02, .1)):
    """Refit the null and extension on identical training counts and EB prior.

    The exact converged null is included in the extension's nested candidate
    set. Multistart selection uses the training objective only; a separate
    caller must select model family on its selection fold and report held-out
    diagnostics without further tuning. No sample nuisance is frozen merely
    to make the extension win.
    """
    assert joint['dispersed']
    n = len(hist)
    mean = np.asarray(joint['hyper_mean'], dtype=float)
    variance = np.asarray(joint['hyper_variance'], dtype=float)
    per_sample = [(-16., -1e-8), (-12., 12.), (-25., 25.), (-25., 25.)]
    for j in range(2):
        if variance[j] == 0.:
            per_sample[j] = (mean[j], mean[j])
    null_bounds = per_sample * n + [(-8., 16.)]
    settings = dict(maxiter=1200, ftol=1e-12, gtol=1e-9, maxcor=20)
    records = []

    def model_from(vector, success, label, result=None):
        local = vector[:-2].reshape(n, 4)
        return dict(parameters=np.column_stack((local[:, :2], np.full(n, vector[-2]))),
            mixing=local[:, 2:], dispersed=True, homo_rho=float(vector[-1]),
            success=bool(success), hyper_mean=mean.tolist(), hyper_variance=variance.tolist(),
            label=label, iterations=0 if result is None else int(result.nit))

    with parallel.numba_thread_scope(threads):
        start = time.monotonic()
        null = minimize(read_model.joint_objective, _vector(joint)[:-1],
            args=(hist, depth, alt, mean, variance), jac=True, method='L-BFGS-B',
            bounds=null_bounds, options=settings)
        if not null.success:
            raise RuntimeError('null nuisance refit did not converge: ' + str(null.message))
        null_model = model_from(np.r_[null.x, 0.], True, 'binomial_homozygotes', null)
        best_vector = np.r_[null.x, 0.]
        best_value = objective(best_vector, hist, depth, alt, mean, variance)[0]
        best_model = model_from(best_vector, True, 'nested_null_endpoint', null)
        records.append(dict(label='null_refit', success=True, seconds=time.monotonic()-start,
                            iterations=int(null.nit), objective=float(null.fun), message=str(null.message)))
        for rho in homo_starts:
            initial = np.r_[null.x, float(rho)]
            start = time.monotonic()
            result = minimize(objective, initial, args=(hist, depth, alt, mean, variance),
                jac=True, method='L-BFGS-B', bounds=null_bounds + [(0., 1.-1e-6)], options=settings)
            records.append(dict(label='homo_beta_binomial', rho_start=float(rho),
                success=bool(result.success), seconds=time.monotonic()-start,
                iterations=int(result.nit), objective=float(result.fun),
                rho=float(result.x[-1]), message=str(result.message)))
            if result.success and np.isfinite(result.fun) and result.fun < best_value:
                best_value = float(result.fun)
                best_model = model_from(result.x, True, 'homo_beta_binomial', result)
    return null_model, best_model, records


def emission(parameters, depth, alt, *, homo_rho, dispersed=True):
    """Log AD kernels without mixture priors or count combinatorial factors.

    REF/REF and ALT/ALT have symmetric means and one shared homozygote
    dispersion. Rising products preserve the exact rho=0 binomial limit.
    """
    parameters = np.asarray(parameters, float)
    depth, alt = np.asarray(depth, np.int64), np.asarray(alt, np.int64)
    ref = depth - alt
    maximum = int(depth.max(initial=0))
    output = np.empty((len(parameters), len(depth), 3))
    for sample, par in enumerate(parameters):
        error, balance = expit(par[:2])
        q = error + (1. - 2. * error) * balance
        hom = _prefix(error, homo_rho / (1. - homo_rho), maximum)
        het = _prefix(q, np.exp(-par[2]) if dispersed else 0., maximum)
        output[sample, :, 0] = hom[0, alt] + hom[1, ref] - hom[2, depth]
        output[sample, :, 1] = het[0, alt] + het[1, ref] - het[2, depth]
        output[sample, :, 2] = hom[0, ref] + hom[1, alt] - hom[2, depth]
    return output
