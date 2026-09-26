"""Exact repeated-count lookup for AD likelihood application.

Read likelihood depends on sample parameters and integer ref/alt counts, not
marker identity. Cache common count pairs; uncommon high-depth observations use
the unchanged reference calculation. The cap is a memory/performance control,
never a read-depth filter or model approximation.
"""
import numpy as np
from numba import njit, prange
from scipy.special import logsumexp

from .read_model import emission
from .read_homozygote_model import emission as homozygote_emission


def _model_emission(model, parameters, depth, alt):
    """Apply the fitted homozygote dispersion, using its binomial limit at zero."""
    rho = float(model.get("homo_rho", 0.))
    if rho > 0.:
        return homozygote_emission(parameters, depth, alt,
            homo_rho=rho, dispersed=model["dispersed"])
    return emission(parameters, depth, alt, model["dispersed"])[0]


@njit(cache=True, parallel=True)
def gather(counts, table, cap):
    """Fill independent sample/marker tiles from the common-depth lookup."""
    n, length = counts.shape[:2]
    out = np.empty((n, length, 3), np.float64)
    chunks = (length + 4095) // 4096
    for task in prange(n * chunks):
        sample, chunk = task // chunks, task % chunks
        for site in range(chunk * 4096, min(length, (chunk + 1) * 4096)):
            ref = int(counts[sample, site, 0])
            alt = int(counts[sample, site, 1])
            depth = ref + alt
            if depth <= cap:
                for g in range(3):
                    out[sample, site, g] = table[sample, depth, alt, g]
            else:
                for g in range(3):
                    out[sample, site, g] = 0.
    return out


def raw_likelihoods(counts, model, cap=64):
    """Return normalized raw GLs, without the calibration genotype-mixture prior.

    The triangle contains all REF/ALT combinations up to the lookup depth.
    High-depth cells are evaluated directly below, not truncated or omitted.
    Zero observed reads give three equal likelihoods.
    """
    counts = np.asarray(counts)
    depth = counts.sum(axis=2)
    cap = min(cap, int(depth.max(initial=0)))
    d, a = np.tril_indices(cap + 1)
    table = np.empty((len(counts), cap + 1, cap + 1, 3), np.float64)
    for sample in range(len(counts)):
        kernel = _model_emission(
            model, model['parameters'][sample:sample + 1], d, a)[0]
        table[sample, d, a] = np.exp(
            kernel - logsumexp(kernel, axis=1, keepdims=True))
        table[sample, 0, 0] = 1 / 3.
    output = gather(counts, table, cap)
    for sample in range(len(counts)):
        high = depth[sample] > cap
        if np.any(high):
            kernel = _model_emission(
                model, model['parameters'][sample:sample + 1],
                depth[sample, high], counts[sample, high, 1])
            output[sample, high] = np.exp(
                kernel[0] - logsumexp(kernel[0], axis=1, keepdims=True))
    return output
