"""Numba scoring kernels for broad panel selection and chimera proposals."""
from __future__ import annotations


import numpy as np
import math
import numba
from numba import njit, prange


def _resolve_threads(num_threads):
    """
    Resolve num_threads: if callable, call it to get the current value.

    This enables dynamic thread reallocation — hierarchical_assembly passes
    a function (e.g. _get_dynamic_threads) instead of a fixed integer.
    Each time _resolve_threads is called, the function re-checks how many
    peer workers are active and returns the appropriate thread count.

    Also updates numba's active thread count to match, so prange loops
    in scoring functions use the correct number of threads.
    """
    if callable(num_threads):
        n = num_threads()
        try:
            numba.set_num_threads(n)
        except Exception:
            pass
        return n
    return num_threads


@njit(parallel=True, fastmath=False)
def _compute_bin_emissions_numba(
        block_samples, hap_q, n_haps, n_bins, snps_per_bin, n_sites,
        complete_sites=None):
    """Hard-call counterpart of the partial-founder emission kernel.

    The caller supplies only q=0/1 at retained sites. Three genotype dosage
    likelihoods therefore cover all pairs; compute their robust/floored logs
    once per sample/site. Symmetric pair updates retain the exact accumulation
    order and the original missing/uniform-site neutrality.
    """
    result = np.zeros((len(block_samples), n_haps, n_haps, n_bins), np.float64)
    center = math.log(1.0 / 3.0)
    for sample in prange(len(block_samples)):
        logs = np.empty(3, np.float64)
        for site in range(n_sites):
            if complete_sites is not None and not complete_sites[site]:
                continue
            a, b, c = block_samples[sample, site]
            total = a + b + c
            if total <= 0.0 or (a == b and b == c):
                continue
            for dosage in range(3):
                evidence = block_samples[sample, site, dosage] / total
                robust = evidence * .99 + .01 / 3.0
                logs[dosage] = max(math.log(max(robust, 1e-300)), -2.0) - center
            for first in range(n_haps):
                allele = int(hap_q[first, site])
                for second in range(first, n_haps):
                    value = logs[allele + int(hap_q[second, site])]
                    result[sample, first, second, site // snps_per_bin] += value
                    if first != second:
                        result[sample, second, first, site // snps_per_bin] += value
    return result


@njit(parallel=True, fastmath=True)
def _gather_samples_numba(probs, indices):
    """Parallel equivalent of np.ascontiguousarray(probs[:, indices, :]).

    probs: (N, n_global, 3); indices: (n_sites,) integer positions into axis 1.
    Returns a C-contiguous (N, n_sites, 3) array with
    out[s, t, c] = probs[s, indices[t], c].

    This is a pure copy (no arithmetic), so it is bit-identical to numpy's
    advanced-index gather.  numpy runs that gather single-threaded, and at L4 the
    probs slice is ~5.4 GB, so the fancy-index copy costs ~10-20s on one core;
    spreading the copy over samples lets the inner-core pool absorb it.  The
    output layout (N, n_sites, 3) matches numpy's, so block_samples[s, :, c] has
    the same stride the emission kernels already expect.
    """
    N = probs.shape[0]
    n_sites = indices.shape[0]
    out = np.empty((N, n_sites, 3), dtype=probs.dtype)
    for s in prange(N):
        for t in range(n_sites):
            g = indices[t]
            out[s, t, 0] = probs[s, g, 0]
            out[s, t, 1] = probs[s, g, 1]
            out[s, t, 2] = probs[s, g, 2]
    return out


