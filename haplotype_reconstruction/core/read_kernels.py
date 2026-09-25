"""Fused observed-read likelihoods and analytic parameter gradients.

These kernels evaluate the same binomial/beta-binomial mixture as read_model.
They skip zero-count histogram cells and keep only one sample's prefix sums,
rather than materializing sample-by-count-by-genotype derivative tensors.
Integer-depth rising products avoid log-gamma cancellation at large dispersion
concentrations. No fast-math reassociation or likelihood approximation is used.

Single-sample fits release the GIL but do not start nested thread teams. Joint
fits parallelize across samples under the caller's existing Numba CPU budget.
"""
import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _logsum3(a, b, c):
    high = max(a, b, c)
    return high + math.log(math.exp(a-high) + math.exp(b-high) + math.exp(c-high))


@njit(cache=True)
def _sample(parameters, mixing, counts, depth, alt, dispersed, mean, variance):
    e = 1. / (1. + math.exp(-parameters[0]))
    balance = 1. / (1. + math.exp(-parameters[1]))
    q = e + (1.-2.*e)*balance
    log_e, log_not_e = math.log(e), math.log1p(-e)
    log_q, log_not_q = math.log(q), math.log1p(-q)
    log_norm = _logsum3(mixing[0], mixing[1], 0.)
    log_mix = np.array([mixing[0]-log_norm, mixing[1]-log_norm, -log_norm])
    # Eight cumulative sufficient sums provide the beta-binomial kernel and
    # its derivatives with respect to mean and log concentration.
    prefix = np.zeros((8, int(np.max(depth))+1 if dispersed else 0))
    if dispersed:
        kappa = math.exp(parameters[2])
        for j in range(prefix.shape[1]-1):
            shift = j/kappa
            qa, qr, total = q+shift, 1.-q+shift, 1.+shift
            prefix[0, j+1] = prefix[0, j]+math.log(qa)
            prefix[1, j+1] = prefix[1, j]+math.log(qr)
            prefix[2, j+1] = prefix[2, j]+math.log1p(shift)
            prefix[3, j+1] = prefix[3, j]+1./qa
            prefix[4, j+1] = prefix[4, j]+1./qr
            prefix[5, j+1] = prefix[5, j]+shift/qa
            prefix[6, j+1] = prefix[6, j]+shift/qr
            prefix[7, j+1] = prefix[7, j]+shift/total
    gradient = np.zeros(len(parameters)+2)
    value, weight, mass0, mass1 = 0., 0., 0., 0.
    for cell in range(len(counts)):
        count = counts[cell]
        if count == 0:
            continue
        d, a = int(depth[cell]), int(alt[cell])
        r = d-a
        if dispersed:
            hetero = prefix[0,a]+prefix[1,r]-prefix[2,d]
            dq = prefix[3,a]-prefix[4,r]
            dk = -prefix[5,a]-prefix[6,r]+prefix[7,d]
        else:
            hetero = a*log_q+r*log_not_q
            dq = a/q-r/(1.-q)
            dk = 0.
        l0 = a*log_e+r*log_not_e+log_mix[0]
        l1 = hetero+log_mix[1]
        l2 = r*log_e+a*log_not_e+log_mix[2]
        z = _logsum3(l0, l1, l2)
        r0, r1, r2 = count*math.exp(l0-z), count*math.exp(l1-z), count*math.exp(l2-z)
        value += count*z
        weight += count
        mass0 += r0
        mass1 += r1
        gradient[0] += r0*(a-d*e) + r1*dq*(1.-2.*balance)*e*(1.-e) + r2*(r-d*e)
        gradient[1] += r1*dq*(1.-2.*e)*balance*(1.-balance)
        if dispersed:
            gradient[2] += r1*dk
    gradient[-2] = mass0-weight*math.exp(log_mix[0])
    gradient[-1] = mass1-weight*math.exp(log_mix[1])
    for j in range(2):
        if variance[j] > 0:
            delta = parameters[j]-mean[j]
            value -= delta*delta/(2.*variance[j])
            gradient[j] -= delta/variance[j]
    return value, gradient


@njit(cache=True, nogil=True)
def _serial_objective(parameters, mixing, hist, depth, alt, dispersed, mean, variance):
    values = np.empty(len(hist))
    gradient = np.empty((len(hist), parameters.shape[1]+2))
    for i in range(len(hist)):
        values[i], gradient[i] = _sample(parameters[i], mixing[i], hist[i], depth, alt, dispersed, mean, variance)
    return values, gradient


@njit(cache=True, nogil=True, parallel=True)
def _parallel_objective(parameters, mixing, hist, depth, alt, dispersed, mean, variance):
    values = np.empty(len(hist))
    gradient = np.empty((len(hist), parameters.shape[1]+2))
    for i in prange(len(hist)):
        values[i], gradient[i] = _sample(parameters[i], mixing[i], hist[i], depth, alt, dispersed, mean, variance)
    return values, gradient


def objective(vector, hist, depth, alt, dispersed, shared, prior=None):
    """Return normalized negative log likelihood and its exact gradient."""
    n = len(hist)
    pcount = 3 if dispersed else 2
    if shared:
        parameters = np.broadcast_to(vector[:pcount], (n, pcount))
        mixing = vector[pcount:].reshape(n, 2)
    else:
        matrix = vector.reshape(n, pcount+2)
        parameters, mixing = matrix[:, :pcount], matrix[:, pcount:]
    mean, variance = (np.zeros(2), np.zeros(2)) if prior is None else prior
    kernel = _serial_objective if n == 1 else _parallel_objective
    values, gradient = kernel(parameters, mixing, hist, depth, alt, dispersed, mean, variance)
    if shared:
        result = np.r_[gradient[:, :pcount].sum(axis=0), gradient[:, pcount:].ravel()]
    else:
        result = gradient.ravel()
    scale = max(1., float(hist.sum()))
    return -float(values.sum())/scale, -result/scale


@njit(cache=True, nogil=True, parallel=True)
def binomial_expectation(hist, depth, alt, error, heterozygote, mixture):
    """Per-sample EM sufficient statistics without a responsibility tensor.

    The four moments are the error numerator/denominator and heterozygote
    ALT numerator/denominator. Samplewise outputs make the pooled reduction
    deterministic for a fixed input, independent of worker completion order.
    """
    values = np.zeros(len(hist))
    mass = np.zeros((len(hist), 3))
    moments = np.zeros((len(hist), 4))
    probability = np.array([error, heterozygote, 1. - error])
    log_probability = np.log(probability)
    log_complement = np.log1p(-probability)
    for sample in prange(len(hist)):
        log_mixture = np.log(np.maximum(mixture[sample], 1e-300))
        for cell in range(len(depth)):
            count = hist[sample, cell]
            if count == 0.:
                continue
            d, a = depth[cell], alt[cell]
            l0 = a * log_probability[0] + (d-a) * log_complement[0] + log_mixture[0]
            l1 = a * log_probability[1] + (d-a) * log_complement[1] + log_mixture[1]
            l2 = a * log_probability[2] + (d-a) * log_complement[2] + log_mixture[2]
            z = _logsum3(l0, l1, l2)
            r0 = count * math.exp(l0-z)
            r1 = count * math.exp(l1-z)
            r2 = count * math.exp(l2-z)
            values[sample] += count * z
            mass[sample, 0] += r0
            mass[sample, 1] += r1
            mass[sample, 2] += r2
            moments[sample, 0] += r0*a + r2*(d-a)
            moments[sample, 1] += (r0+r2)*d
            moments[sample, 2] += r1*a
            moments[sample, 3] += r1*d
    return values, mass, moments
