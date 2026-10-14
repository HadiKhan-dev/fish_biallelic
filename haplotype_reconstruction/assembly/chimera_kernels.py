"""Numba scoring kernels for broad panel selection and chimera proposals."""
from __future__ import annotations


from concurrent.futures import ThreadPoolExecutor
import functools
import threading
import numpy as np
import math
import numba
from numba import njit, prange


def _thread_budget(num_threads):
    """Current value of a thread budget (an int, a callable, or None when unmanaged)."""
    return num_threads() if callable(num_threads) else num_threads


class _ThreadShares:
    """Equal shares of a live thread budget among the running items of one concurrent_map.

    The budget is read under the lock, so concurrent items never call it at once (a worker's
    core.parallel.get_dynamic_threads claims its remainder thread for the whole process).
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.lock = threading.Lock()
        self.running = []

    def start(self, index):
        with self.lock:
            self.running.append(index)
        return functools.partial(self.threads, index)

    def finish(self, index):
        with self.lock:
            self.running.remove(index)

    def threads(self, index):
        with self.lock:
            total = max(1, int(_thread_budget(self.num_threads)))
            running = len(self.running)
            rank = self.running.index(index)
        return max(1, total // running + (rank < total % running))


def concurrent_map(function, items, num_threads):
    """[function(item, threads) for item in items], the items running concurrently.

    For independent calls whose heavy work runs in nogil kernels. Each item runs in a thread
    with an equal share of the caller's live thread budget (num_threads: an int or a callable,
    as _resolve_threads takes); ``threads`` is the item's share as a callable, which the item
    passes on as its own budget and re-reads at its thread boundaries: floor(total / running)
    plus one for the first total % running of the running items, so the threads of finished
    items go to the rest. The first item runs in the calling thread. Results come back in item
    order, so callers combine them exactly as their sequential loop did; the calls share no
    floating-point work, so neither the concurrency nor the thread counts change a result.
    With fewer than two items or threads (or an unmanaged budget, None) the items run in order
    in the calling thread with num_threads itself.
    """
    items = list(items)
    total = _thread_budget(num_threads)
    if len(items) < 2 or total is None or total < 2:
        return [function(item, num_threads) for item in items]
    shares = _ThreadShares(num_threads)

    def run(index):
        threads = shares.start(index)
        try:
            numba.set_num_threads(threads())
            return function(items[index], threads)
        finally:
            shares.finish(index)

    caller_threads = numba.get_num_threads()
    with ThreadPoolExecutor(max_workers=min(len(items), total) - 1) as executor:
        others = [executor.submit(run, index) for index in range(1, len(items))]
        try:
            first = run(0)
        finally:
            numba.set_num_threads(caller_threads)
        return [first] + [future.result() for future in others]


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


@njit(cache=True, parallel=True, nogil=True)
def _gather_rows_numba(probs, rows, start, stop):
    """np.ascontiguousarray(probs[rows, start:stop, :]) (a pure copy) spread over the rows.

    A join's evidence slice of its informative samples over its site span: at L4 a whole
    chromosome, several GB, which numpy copies on one core.
    """
    sites = stop - start
    out = np.empty((len(rows), sites, probs.shape[2]), dtype=probs.dtype)
    for r in prange(len(rows)):
        source = rows[r]
        for t in range(sites):
            for c in range(probs.shape[2]):
                out[r, t, c] = probs[source, start + t, c]
    return out


