"""Self-calibrated site-quality filter on one contig's biallelic SNPs.

Each contig is calibrated on its own distributions, from allele depths only: no sample labels,
benchmark, pedigree or simulation truth. A site is removed when any applied rule fires:

  r1  total depth outside the clean core of total depth (both tails);
  r2  per-sample depth heterogeneity, the dispersion index of sample depths about the site's own
      depth factor, above its clean core;
  r3  wrong-allele reads in samples classified homozygous: a negative-binomial artefact class is
      more likely than the Poisson clean core;
  r4  heterozygote allele balance: a beta-binomial artefact class is more likely than the
      binomial clean core;
  r5  (informativeness, switchable) fewer than two expected carriers of a non-major genotype
      (among samples the per-sample contamination QC does not flag).

Rule r0 (not a biallelic SNP) is the loader's: such records never reach this filter. A clean
core is the histogram mode with Gaussian-equivalent left/right widths from its half-maximum
crossings; its window is where the core's local posterior (Efron's empirical-null local fdr) is
at least ``core_posterior_level``. A mixture whose artefact class does not separate from its
clean core flags nothing. Genotypes are classified under a fixed read model
(``classification_error``, ``classification_heterozygote_alt``).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import math
import ctypes
from typing import Any, Mapping

import numpy as np
from numba import njit, prange, config as numba_config
from numba.extending import get_cython_function_address
from scipy import special

from ..maths import ties
from ..execution import numba_threads
from ..execution.budget import current_budget, CpuBudget
from .config import SampleQualityConfig, SiteQualityConfig
from . import sample_quality

VERSION = "self-calibrated-site-quality-v1"
RULES = ("r1", "r2", "r3", "r4", "r5")
QUALITY_RULES = RULES[:4]          # r1-r4; r5 is informativeness

# Numerical settings of the fits, not scientific thresholds.
_SITE_EM_ITERATIONS = 60
_MIXTURE_ITERATIONS = 300
_HISTOGRAM_RANGE_PERCENTILES = (0.5, 99.5)
# Artefact M-steps: projected Newton, stopped when the full Newton step moves no coordinate by more than _M_STEP_STEP
# times max(1, |coordinate|) (log/logit scale); certified stationary when the projected gradient divided by the
# artefact class's posterior weight is then at most _M_STEP_GRADIENT, else recorded as a small-step stop with the
# gradient certificate unmet ("precision"). A step is accepted when the objective does not fall by more than
# _M_STEP_ROUNDING times max(1, |objective|), or, for a Newton step, when it reduces the projected gradient while the
# objective falls by at most _M_STEP_NOISE times that (near the maximum the summed objective's rounding exceeds the
# remaining gain: large dispersions sum cancelling log-gamma terms).
_M_STEP_GRADIENT = 1e-9
_M_STEP_STEP = 1e-8
_M_STEP_ROUNDING = 1e-13
_M_STEP_NOISE = 1e-11
_M_STEP_ITERATIONS = 100
_M_STEP_CHUNKS = 256             # fixed site chunks: the sums do not depend on the thread count
_POLYGAMMA_SHIFT = 16.0          # recurrence up to this argument before the asymptotic series
_SITE_TILE_CELLS = 1 << 20
_MAXIMUM_TILE_SITES = 4096
_MINIMUM_CHUNK_SITES = 16384   # amortize thread dispatch for vectorized special functions


@dataclass(frozen=True)
class SiteQualityResult:
    """The filter's decision on one contig's input sites (ContigReads order).

    applied: the rules ran (False when disabled or with fewer than minimum_calibration_sites
    SNPs; then every site is kept). kept: bool[L_in]. rule_bits: uint8[L_in], bit k set when rule
    rk fired (k = 1..5; bit 0 is unused: r0 is the loader's); None when not applied.
    applied_rules: the rules that remove sites (r5 only with drop_uninformative_sites).
    flagged / removed_only_by: sites per rule / removed by that rule alone. parameters: fitted
    cores and mixtures (JSON). sample_quality: the contamination QC decided on this contig (None
    when it did not run). r5_excluded_samples: flagged samples left out of the r5 carrier count.
    """

    enabled: bool
    applied: bool
    reason: str | None
    kept: np.ndarray
    rule_bits: np.ndarray | None = None
    applied_rules: tuple[str, ...] = ()
    flagged: Mapping[str, int] = field(default_factory=dict)
    removed_only_by: Mapping[str, int] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    sample_quality: "sample_quality.SampleQualityResult | None" = None
    r5_excluded_samples: int = 0

    def record(self) -> dict:
        return dict(version=VERSION, enabled=self.enabled, applied=self.applied, reason=self.reason,
                    kept=int(np.sum(self.kept)), records=int(len(self.kept)),
                    rule_bits=self.rule_bits, applied_rules=list(self.applied_rules),
                    flagged=dict(self.flagged), removed_only_by=dict(self.removed_only_by),
                    parameters=dict(self.parameters),
                    sample_quality=None if self.sample_quality is None else self.sample_quality.record(),
                    r5_excluded_samples=self.r5_excluded_samples)

    @classmethod
    def from_record(cls, record: Mapping, kept: np.ndarray) -> "SiteQualityResult":
        samples = record["sample_quality"]
        return cls(enabled=record["enabled"], applied=record["applied"], reason=record["reason"],
                   kept=np.asarray(kept, bool), rule_bits=record["rule_bits"],
                   applied_rules=tuple(record["applied_rules"]), flagged=record["flagged"],
                   removed_only_by=record["removed_only_by"], parameters=record["parameters"],
                   sample_quality=None if samples is None
                   else sample_quality.SampleQualityResult.from_record(samples),
                   r5_excluded_samples=record["r5_excluded_samples"])


# ---------------------------------------------------------------------------------------------
# Per-site genotype classification under a fixed read model
# ---------------------------------------------------------------------------------------------

@njit(cache=True)
def _site_genotype_fit(a, n, eps, h, iters, f, ll):
    """EM over one site's free genotype frequencies (no Hardy-Weinberg).

    alt reads ~ Binomial(depth, p_g), p = (eps, h, 1 - eps). Returns the expected wrong-allele
    reads and depth in homozygotes and the expected ALT reads and depth in heterozygotes from the
    last E-step; f holds the updated frequencies.
    """
    S = len(n)
    le, l1e = math.log(eps), math.log(1.0 - eps)
    lh, l1h = math.log(h), math.log(1.0 - h)
    for s in range(S):
        ll[s, 0] = a[s] * le + (n[s] - a[s]) * l1e
        ll[s, 1] = a[s] * lh + (n[s] - a[s]) * l1h
        ll[s, 2] = a[s] * l1e + (n[s] - a[s]) * le
    # These likelihoods do not depend on the EM frequencies. Evaluate the same
    # exponentials once, preserving the original operation and sample order.
    maximum = np.empty(S)
    for s in range(S):
        maximum[s] = max(ll[s, 0], ll[s, 1], ll[s, 2])
        for g in range(3):
            ll[s, g] = math.exp(ll[s, g] - maximum[s])
    f[0] = f[1] = f[2] = 1.0 / 3.0
    old = -1e300
    wrong = 0.0; homn = 0.0; heta = 0.0; hetn = 0.0
    for it in range(iters):
        value = 0.0
        c0 = c1 = c2 = 0.0
        wrong = 0.0; homn = 0.0; heta = 0.0; hetn = 0.0
        for s in range(S):
            if n[s] == 0:
                continue
            m = maximum[s]
            p0 = f[0] * ll[s, 0]
            p1 = f[1] * ll[s, 1]
            p2 = f[2] * ll[s, 2]
            t = p0 + p1 + p2
            value += m + math.log(t)
            p0 /= t; p1 /= t; p2 /= t
            c0 += p0; c1 += p1; c2 += p2
            wrong += p0 * a[s] + p2 * (n[s] - a[s])
            homn += (p0 + p2) * n[s]
            heta += p1 * a[s]
            hetn += p1 * n[s]
        tot = c0 + c1 + c2
        # A tiny floor keeps the frequencies inside the simplex.
        f[0] = max(c0 / tot, 1e-6); f[1] = max(c1 / tot, 1e-6); f[2] = max(c2 / tot, 1e-6)
        z = f[0] + f[1] + f[2]
        f[0] /= z; f[1] /= z; f[2] /= z
        if it > 3 and value - old < 1e-7:
            break
        old = value
    return wrong, homn, heta, hetn


@njit(cache=True, parallel=True)
def _genotype_expectations(alt, dep, eps, h, iters):
    """alt, dep: (sites, samples). Per-site frequencies, expected counts, covered samples."""
    L, S = dep.shape
    frequencies = np.zeros((L, 3))
    wrong = np.zeros(L); homn = np.zeros(L); heta = np.zeros(L); hetn = np.zeros(L)
    covered = np.zeros(L, np.int32)
    for i in prange(L):
        a = alt[i].astype(np.float64); n = dep[i].astype(np.float64)
        c = 0
        for s in range(S):
            if n[s] > 0:
                c += 1
        covered[i] = c
        if c == 0:
            frequencies[i, 0] = 1.0
            continue
        f = np.zeros(3); ll = np.zeros((S, 3))
        w, hn, ha, hh = _site_genotype_fit(a, n, eps, h, iters, f, ll)
        wrong[i] = w; homn[i] = hn; heta[i] = ha; hetn[i] = hh
        frequencies[i, 0] = f[0]; frequencies[i, 1] = f[1]; frequencies[i, 2] = f[2]
    return frequencies, wrong, homn, heta, hetn, covered


@njit(cache=True, nogil=True)
def _genotype_counts_into(counts, samples, eps, h, iters, start, stop,
                          frequencies, wrong, homn, heta, hetn, covered):
    S = len(samples)
    for i in range(start, stop):
        a = np.empty(S); n = np.empty(S)
        c = 0
        for j in range(S):
            sample = samples[j]
            a[j] = counts[sample, i, 1]
            n[j] = np.int32(counts[sample, i, 0] + counts[sample, i, 1])
            if n[j] > 0:
                c += 1
        covered[i] = c
        if c == 0:
            frequencies[i, 0] = 1.0
            continue
        f = np.zeros(3); ll = np.zeros((S, 3))
        w, hn, ha, hh = _site_genotype_fit(a, n, eps, h, iters, f, ll)
        wrong[i] = w; homn[i] = hn; heta[i] = ha; hetn[i] = hh
        frequencies[i, 0] = f[0]; frequencies[i, 1] = f[1]; frequencies[i, 2] = f[2]


def _genotype_expectations_counts(counts, samples, eps, h, iters):
    """One live-budget queue, with direct-count tasks and disjoint output slices."""
    L = counts.shape[1]
    values = (np.zeros((L, 3)), np.zeros(L), np.zeros(L), np.zeros(L),
              np.zeros(L), np.zeros(L, np.int32))
    if L == 0:
        return values
    # Limit the number of simultaneous O(N) workspaces to the reserved capacity.
    # Increasing this site's span adds no dense sample-by-site allocation.
    capacity = _tile_sites(counts.shape[0])
    tasks = min(capacity, 4 * numba_config.NUMBA_NUM_THREADS)
    span = max(1, (L + tasks - 1) // tasks)
    chunks = [(start, min(start + span, L)) for start in range(0, L, span)]
    chunks.sort(key=lambda bounds: bounds[1] - bounds[0], reverse=True)
    # Compile once before dispatch, preventing compilation races among workers.
    _genotype_counts_into(counts, samples, eps, h, iters, 0, 0, *values)

    def work(bounds):
        _genotype_counts_into(counts, samples, eps, h, iters, *bounds, *values)

    # The outer stage reservation covers the output vectors and all worker scratch.
    # Freed workers immediately take the next task; a live lease can admit more.
    current_budget().thread_map(work, chunks, label="site genotype queue")
    return values


# ---------------------------------------------------------------------------------------------
# Empirical-null clean cores
# ---------------------------------------------------------------------------------------------

def _peak_core(x, smooth, integer=False):
    """Mode and left/right Gaussian-equivalent sds from the half-maximum crossings of a smoothed histogram."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, _HISTOGRAM_RANGE_PERCENTILES)
    q25, q75 = np.percentile(x, [25, 75])
    width = 2 * (q75 - q25) / len(x) ** (1 / 3)          # Freedman-Diaconis
    if integer:
        width = max(1.0, float(np.round(width)))
        lo = np.floor(lo) - 0.5
        nb = int(np.ceil((hi - lo) / width)); hi = lo + nb * width
    else:
        nb = int(np.clip((hi - lo) / max(width, 1e-12), 50, 2000))
    counts, edges = np.histogram(x, bins=nb, range=(lo, hi))
    centres = 0.5 * (edges[1:] + edges[:-1])
    kernel = np.ones(smooth) / smooth
    if nb >= smooth:
        smoothed = np.convolve(counts, kernel, mode="same")
    else:
        # mode="same" returns the longer operand's length: keep the window centred on each of the nb bins.
        smoothed = np.convolve(counts, kernel, mode="full")[(smooth - 1) // 2:(smooth - 1) // 2 + nb]
    # The mode bin: the first of exactly tied maxima (the last under the convention reversal, release 2.2).
    p = ties.conv_argmax(smoothed); top = smoothed[p]

    def cross(direction):
        # Walk while the next bin exists (an interior mode stops where 2.10's walk did; a mode at an end of the
        # histogram can now walk inwards, and the previous bin is never taken from beyond the other end).
        i = p
        while 0 <= i + direction < len(smoothed) and smoothed[i] > top / 2:
            i += direction
        j = i - direction
        if not 0 <= j < len(smoothed) or smoothed[j] == smoothed[i]:
            return centres[i]
        return centres[j] + (top / 2 - smoothed[j]) * (centres[i] - centres[j]) / (smoothed[i] - smoothed[j])
    left, right = cross(-1), cross(+1)
    f = np.sqrt(2 * np.log(2))
    density = top / (len(x) * (edges[1] - edges[0]))
    sd_left, sd_right = (centres[p] - left) / f, (right - centres[p]) / f
    return dict(mode=float(centres[p]), sd_left=float(sd_left), sd_right=float(sd_right),
                peak_density=float(density), tied_mode=bool(np.count_nonzero(smoothed == top) > 1),
                core_weight=float(density * np.sqrt(2 * np.pi) * 0.5 * (sd_left + sd_right)),
                bins=nb, centres=centres, smooth=smoothed, n=len(x), width=float(edges[1] - edges[0]))


def _bounds(core, level):
    """Window around the mode inside which the core's local posterior is >= level."""
    if core["sd_left"] == 0.0 and core["sd_right"] == 0.0:
        # Every value in one histogram bin: the core has no spread to bound, and nothing lies outside it.
        return -np.inf, np.inf
    c = core["centres"]
    density = core["smooth"] / (core["n"] * core["width"])
    sd = np.where(c < core["mode"], core["sd_left"], core["sd_right"])
    # A side without spread (sd 0) has null density only at the mode itself.
    z = np.divide(c - core["mode"], sd, out=np.where(c == core["mode"], 0.0, np.inf), where=sd > 0)
    null = core["peak_density"] * np.exp(-0.5 * z ** 2)  # pi0*f0 anchored at the peak
    posterior = np.where(density > 0, np.clip(null / np.maximum(density, 1e-300), 0, 1), 1.0)
    p = int(np.argmin(np.abs(c - core["mode"])))
    i = p
    while i > 0 and posterior[i - 1] >= level:
        i -= 1
    j = p
    while j < len(c) - 1 and posterior[j + 1] >= level:
        j += 1
    lo = -np.inf if i == 0 else c[i] - 0.5 * core["width"]
    hi = np.inf if j == len(c) - 1 else c[j] + 0.5 * core["width"]
    return float(lo), float(hi)


def _core_record(core):
    return {key: core[key] for key in ("mode", "sd_left", "sd_right", "core_weight")}


# ---------------------------------------------------------------------------------------------
# Clean-core / artefact mixtures (log pmfs; the parameter-free leading terms log W! and
# log C(H, A) are evaluated once per fit)
# ---------------------------------------------------------------------------------------------

def _log_poisson_pmf(W, mu, log_w_factorial):
    return W * np.log(np.maximum(mu, 1e-300)) - mu - log_w_factorial


def _log_negative_binomial_pmf(W, mu, k, log_w_factorial):
    return (special.gammaln(W + k) - special.gammaln(k) - log_w_factorial
            + k * np.log(k / (k + mu)) + W * np.log(mu / (k + mu)))


def _log_binomial_pmf(A, H, h, log_choose):
    return log_choose + A * np.log(h) + (H - A) * np.log(1 - h)


def _log_beta_binomial_pmf(A, H, m, c, log_choose):
    a = m * c; b = (1 - m) * c
    return log_choose + special.betaln(A + a, H - A + b) - special.betaln(a, b)


def _posterior_clean(log_clean, log_artefact):
    return 1 / (1 + np.exp(np.clip(log_artefact - log_clean, -700, 700)))


# Use the same SciPy scalar special functions as its NumPy ufuncs. Passing the
# pointers as kernel arguments keeps cached machine code process-independent.
_GAMMALN = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_int)(
    get_cython_function_address("scipy.special.cython_special", "gammaln"))
_BETALN = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int)(
    get_cython_function_address("scipy.special.cython_special", "betaln"))


@njit(cache=True, parallel=True, nogil=True)
def _negative_binomial_values(W, N, lw, rate, shape, gamma_shape, function):
    result = np.empty(len(W))
    for i in prange(len(W)):
        mu = N[i] * rate
        result[i] = (function(W[i] + shape, 1) - gamma_shape - lw[i]
                     + shape * math.log(shape / (shape + mu))
                     + W[i] * math.log(mu / (shape + mu)))
    return result


@njit(cache=True, parallel=True, nogil=True)
def _beta_values(A, H, a, b, function):
    result = np.empty(len(A))
    for i in prange(len(A)):
        result[i] = function(A[i] + a, H[i] - A[i] + b, 1)
    return result


class _Chunks:
    """Mixture evaluation under the caller's live CPU allowance.

    Native sitewise special-function kernels reclaim available CPUs at every objective
    boundary. The vectorized chunk evaluator remains a reference path. Reductions retain
    the original NumPy order; r4 eligibility still waits for the completed r3 decision.
    """

    def __init__(self, pool, size, threads):
        count = max(1, min(4 * threads, size // _MINIMUM_CHUNK_SITES))
        bounds = np.linspace(0, size, count + 1).astype(np.int64)
        self.pool, self.size = pool, size
        budget = current_budget()
        self.budget = budget if budget.live else CpuBudget.fixed(min(threads, budget.threads()))
        self.chunks = [(start, stop) for start, stop in zip(bounds[:-1], bounds[1:]) if stop > start]

    def __call__(self, function, *arrays):
        out = np.empty(self.size)

        def work(bounds):
            start, stop = bounds
            out[start:stop] = function(*(array[start:stop] for array in arrays))
        for _ in self.pool.map(work, self.chunks):
            pass
        return out

    def negative_binomial(self, W, N, lw, rate, shape):
        with numba_threads(self.budget.threads()):
            return _negative_binomial_values(W, N, lw, rate, shape,
                                             special.gammaln(shape), _GAMMALN)

    def beta_binomial(self, A, H, lc, mean, concentration):
        a = mean * concentration; b = (1 - mean) * concentration
        with numba_threads(self.budget.threads()):
            beta = _beta_values(A, H, a, b, _BETALN)
        return lc + beta - special.betaln(a, b)


# ---------------------------------------------------------------------------------------------
# Exact artefact M-steps (release 2.11). Each maximises the weighted artefact log likelihood over its two
# parameters on the log/logit scale, jointly within the box of the stored dispersion clips, by projected Newton steps
# with analytic gradients and Hessians, and records whether the result is certified stationary. Releases 2.3-2.10
# instead searched the unbounded parameters with Nelder-Mead (xatol 1e-4) and then clipped the dispersion alone, so
# the stored rate or mean could be non-stationary for the stored dispersion; a different, not just more precise, M-step.
# ---------------------------------------------------------------------------------------------

@njit(cache=True)
def _digamma(x):
    out = 0.0
    while x < _POLYGAMMA_SHIFT:
        out -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    z = inv * inv
    return out + math.log(x) - 0.5 * inv - z * (1 / 12 - z * (1 / 120 - z * (1 / 252 - z * (1 / 240 - z * (
        1 / 132 - z * (691 / 32760 - z * (1 / 12 - z * 3617 / 8160)))))))


@njit(cache=True)
def _trigamma(x):
    out = 0.0
    while x < _POLYGAMMA_SHIFT:
        out += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    z = inv * inv
    # psi'(x) ~ 1/x + 1/(2x^2) + sum_k B_2k / x^(2k+1)
    return out + inv + 0.5 * z + inv * z * (1 / 6 - z * (1 / 30 - z * (1 / 42 - z * (1 / 30 - z * (
        5 / 66 - z * (691 / 2730 - z * 7 / 6))))))


@njit(cache=True, parallel=True, nogil=True)
def _negative_binomial_terms(W, N, w, rate, k):
    """Weighted sums over fixed site chunks of log NegBin(W | mean N rate, shape k) (without log W!) and its first
    and second derivatives in (log rate, log k): rows of (value, gx, gy, hxx, hyy, hxy)."""
    n = len(W)
    out = np.zeros((_M_STEP_CHUNKS, 6))
    log_k, digamma_k, trigamma_k, lgamma_k = math.log(k), _digamma(k), _trigamma(k), math.lgamma(k)
    for chunk in prange(_M_STEP_CHUNKS):
        s0 = s1 = s2 = s3 = s4 = s5 = 0.0
        for i in range(chunk * n // _M_STEP_CHUNKS, (chunk + 1) * n // _M_STEP_CHUNKS):
            if w[i] == 0.0:
                continue
            x, mu = W[i], N[i] * rate
            km = k + mu
            log_km = math.log(km)
            d_mu = x / mu - (x + k) / km
            d_k = _digamma(x + k) - digamma_k + log_k + 1.0 - log_km - (x + k) / km
            s0 += w[i] * (math.lgamma(x + k) - lgamma_k + k * (log_k - log_km) + x * (math.log(mu) - log_km))
            s1 += w[i] * d_mu * mu
            s2 += w[i] * d_k * k
            s3 += w[i] * ((-x / (mu * mu) + (x + k) / (km * km)) * mu * mu + d_mu * mu)
            s4 += w[i] * ((_trigamma(x + k) - trigamma_k + 1.0 / k - 1.0 / km - (mu - x) / (km * km)) * k * k
                          + d_k * k)
            s5 += w[i] * (x - mu) / (km * km) * mu * k
        out[chunk, 0], out[chunk, 1], out[chunk, 2] = s0, s1, s2
        out[chunk, 3], out[chunk, 4], out[chunk, 5] = s3, s4, s5
    return out


@njit(cache=True, parallel=True, nogil=True)
def _beta_binomial_terms(A, H, w, a, b):
    """Weighted sums over fixed site chunks of the site parts of log B(A + a, H - A + b) and their derivatives in
    (a, b): rows of (value, da, db, daa, dbb, dab)."""
    n = len(A)
    out = np.zeros((_M_STEP_CHUNKS, 6))
    for chunk in prange(_M_STEP_CHUNKS):
        s0 = s1 = s2 = s3 = s4 = s5 = 0.0
        for i in range(chunk * n // _M_STEP_CHUNKS, (chunk + 1) * n // _M_STEP_CHUNKS):
            if w[i] == 0.0:
                continue
            x, y, t = A[i] + a, H[i] - A[i] + b, H[i] + a + b
            digamma_t, trigamma_t = _digamma(t), _trigamma(t)
            s0 += w[i] * (math.lgamma(x) + math.lgamma(y) - math.lgamma(t))
            s1 += w[i] * (_digamma(x) - digamma_t)
            s2 += w[i] * (_digamma(y) - digamma_t)
            s3 += w[i] * (_trigamma(x) - trigamma_t)
            s4 += w[i] * (_trigamma(y) - trigamma_t)
            s5 -= w[i] * trigamma_t
        out[chunk, 0], out[chunk, 1], out[chunk, 2] = s0, s1, s2
        out[chunk, 3], out[chunk, 4], out[chunk, 5] = s3, s4, s5
    return out


def _chunk_totals(rows):
    total = np.zeros(rows.shape[1])
    for row in rows:                 # fixed order
        total += row
    return total


def _negative_binomial_objective(q, W, N, w, weight_total):
    """Weighted artefact log likelihood of r3 at q = (log eps1, log k), its gradient and Hessian."""
    total = _chunk_totals(_negative_binomial_terms(W, N, w, math.exp(q[0]), math.exp(q[1])))
    return total[0], total[1:3], np.array([[total[3], total[5]], [total[5], total[4]]])


def _beta_binomial_objective(q, A, H, w, weight_total):
    """Weighted artefact log likelihood of r4 at q = (logit mean, log concentration), its gradient and Hessian."""
    mean, concentration = 1.0 / (1.0 + math.exp(-q[0])), math.exp(q[1])
    a, b = mean * concentration, (1.0 - mean) * concentration
    total = _chunk_totals(_beta_binomial_terms(A, H, w, a, b))
    value = total[0] - weight_total * (math.lgamma(a) + math.lgamma(b) - math.lgamma(concentration))
    fa = total[1] - weight_total * (_digamma(a) - _digamma(concentration))
    fb = total[2] - weight_total * (_digamma(b) - _digamma(concentration))
    faa = total[3] - weight_total * (_trigamma(a) - _trigamma(concentration))
    fbb = total[4] - weight_total * (_trigamma(b) - _trigamma(concentration))
    fab = total[5] + weight_total * _trigamma(concentration)
    d = concentration * mean * (1.0 - mean)          # da/du = -db/du for u = logit mean; da/dv = a, db/dv = b
    gradient = np.array([(fa - fb) * d, fa * a + fb * b])
    huu = (faa - 2 * fab + fbb) * d * d + (fa - fb) * d * (1.0 - 2.0 * mean)
    huv = (faa * a + fab * (b - a) - fbb * b) * d + (fa - fb) * d
    hvv = faa * a * a + 2 * fab * a * b + fbb * b * b + fa * a + fb * b
    return value, gradient, np.array([[huu, huv], [huv, hvv]])


def _newton_maximum(objective, start, lower, upper, scale):
    """Projected Newton ascent of ``objective`` in the box [lower, upper]. A coordinate at a bound whose gradient points
    outward is held (the Karush-Kuhn-Tucker condition there); the others take a Newton step where their Hessian is
    negative definite, else a scaled gradient step, halved until the objective does not decrease beyond rounding.

    Returns (q, status, gradient): status ``stationary`` (the certificate above), ``precision`` (the step tolerance
    is met but the gradient certificate is not), ``line_search`` (no ascent along the step) or ``iterations`` (the
    cap); gradient is the infinity norm of the free coordinates' gradient divided by ``scale`` (the artefact class's
    posterior weight) at q.
    """
    def projected_gradient(q, gradient):
        free = ~(((q <= lower) & (gradient < 0)) | ((q >= upper) & (gradient > 0)))
        return free, (float(np.abs(gradient[free]).max()) / scale if free.any() else 0.0)

    q = np.clip(np.asarray(start, float), lower, upper)
    value, gradient, hessian = objective(q)
    status = "iterations"
    for _ in range(_M_STEP_ITERATIONS + 1):
        free, projected = projected_gradient(q, gradient)
        step = np.zeros(len(q))
        newton = False
        if free.any():
            block = hessian[np.ix_(free, free)]
            if np.linalg.eigvalsh(block).max() < 0:
                step[free] = -np.linalg.solve(block, gradient[free])
                newton = True
            else:
                step[free] = gradient[free] / max(1.0, float(np.abs(block).max()))
        if not free.any() or (newton and np.all(np.abs(step) <= _M_STEP_STEP * np.maximum(1.0, np.abs(q)))):
            status = "stationary" if projected <= _M_STEP_GRADIENT else "precision"
            break
        if _ == _M_STEP_ITERATIONS:
            break
        allowance = _M_STEP_ROUNDING * max(1.0, abs(value))
        alpha, accepted = 1.0, False
        for _halving in range(40):
            candidate = np.clip(q + alpha * step, lower, upper)
            candidate_value, candidate_gradient, candidate_hessian = objective(candidate)
            if not np.isfinite(candidate_value):
                alpha *= 0.5
                continue
            if candidate_value >= value - allowance or (
                    newton and candidate_value >= value - _M_STEP_NOISE * max(1.0, abs(value))
                    and projected_gradient(candidate, candidate_gradient)[1] < projected):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            status = "line_search"
            break
        q, value, gradient, hessian = candidate, candidate_value, candidate_gradient, candidate_hessian
    return q, status, projected


def _tally(statuses, status, gradient):
    """Count an M-step's outcome; keep the largest returned gradient over all of the fit's M-steps, whatever their
    status."""
    statuses[status] = statuses.get(status, 0) + 1
    statuses["worst_gradient"] = max(statuses.get("worst_gradient", 0.0), gradient)


# Search boxes of the M-steps: the shape and concentration boxes are the clips of the fitted values; the rate and mean
# boxes only keep the arithmetic finite.
# np.log as for the M-steps' starts, so that a clipped dispersion starts exactly on its bound.
_ERROR_LOWER, _ERROR_UPPER = np.log([1e-12, 1e-3]), np.log([10.0, 1e4])
_BALANCE_LOWER, _BALANCE_UPPER = np.array([-30.0, np.log(1e-2)]), np.array([30.0, np.log(1e5)])


def _error_mixture(W, N, evaluate):
    """Clean: W ~ Poisson(N eps0). Artefact: W ~ NegBin(mean N eps1, shape k). Returns parameters and P(clean)."""
    pi = 0.6; e0 = max(np.median(W / N) / 2, 1e-5); e1 = 10 * e0; k = 1.0
    lw = special.gammaln(W + 1)
    m_steps = {}

    def artefact(rate, shape):
        return evaluate.negative_binomial(W, N, lw, rate, shape)
    for it in range(_MIXTURE_ITERATIONS):
        wc = _posterior_clean(np.log(pi) + _log_poisson_pmf(W, N * e0, lw),
                              np.log1p(-pi) + artefact(e1, k))
        pi2 = float(np.clip(wc.mean(), 1e-6, 1 - 1e-6)); e02 = float((wc * W).sum() / (wc * N).sum())
        wa = 1 - wc
        if wa.sum() > 1e-6:
            weight = float(wa.sum())
            with numba_threads(evaluate.budget.threads()):
                q, status, gradient = _newton_maximum(
                    lambda q: _negative_binomial_objective(q, W, N, wa, weight),
                    [np.log(e1), np.log(k)], _ERROR_LOWER, _ERROR_UPPER, weight)
            _tally(m_steps, status, gradient)
            e12, k2 = float(np.exp(q[0])), float(np.clip(np.exp(q[1]), 1e-3, 1e4))
        else:
            e12, k2 = e1, k
        change = abs(e02 - e0) / e0 + abs(pi2 - pi) + abs(e12 - e1) / e1
        pi, e0, e1, k = pi2, e02, e12, k2
        if change < 1e-5:
            break
    wc = _posterior_clean(np.log(pi) + _log_poisson_pmf(W, N * e0, lw),
                          np.log1p(-pi) + artefact(e1, k))
    return dict(pi_clean=pi, eps0=e0, eps1=e1, shape=k, iterations=it + 1, m_steps=m_steps), wc


def _balance_mixture(A, H, evaluate):
    """Clean: A ~ Binomial(H, h0). Artefact: beta-binomial with its own mean and concentration."""
    pi = 0.6; h0 = float(np.clip(A.sum() / H.sum(), 0.2, 0.8)); m1 = 0.75 * h0; c1 = 5.0
    lc = special.gammaln(H + 1) - special.gammaln(A + 1) - special.gammaln(H - A + 1)
    m_steps = {}

    def artefact(mean, concentration):
        return evaluate.beta_binomial(A, H, lc, mean, concentration)
    for it in range(_MIXTURE_ITERATIONS):
        wc = _posterior_clean(np.log(pi) + _log_binomial_pmf(A, H, h0, lc),
                              np.log1p(-pi) + artefact(m1, c1))
        pi2 = float(np.clip(wc.mean(), 1e-6, 1 - 1e-6)); h02 = float((wc * A).sum() / (wc * H).sum())
        wa = 1 - wc
        if wa.sum() > 1e-6:
            weight = float(wa.sum())
            with numba_threads(evaluate.budget.threads()):
                q, status, gradient = _newton_maximum(
                    lambda q: _beta_binomial_objective(q, A, H, wa, weight),
                    [np.log(m1 / (1 - m1)), np.log(c1)], _BALANCE_LOWER, _BALANCE_UPPER, weight)
            _tally(m_steps, status, gradient)
            m12 = float(1 / (1 + np.exp(-q[0]))); c12 = float(np.clip(np.exp(q[1]), 1e-2, 1e5))
        else:
            m12, c12 = m1, c1
        change = abs(h02 - h0) + abs(pi2 - pi) + abs(m12 - m1)
        pi, h0, m1, c1 = pi2, h02, m12, c12
        if change < 1e-6:
            break
    wc = _posterior_clean(np.log(pi) + _log_binomial_pmf(A, H, h0, lc),
                          np.log1p(-pi) + artefact(m1, c1))
    return dict(pi_clean=pi, h0=h0, artefact_mean=m1, artefact_concentration=c1, iterations=it + 1,
                m_steps=m_steps), wc


# ---------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------

def _tile_sites(n_samples):
    """Bound dense working arrays by cells as well as sites (large cohorts use shorter tiles)."""
    return max(1, min(_MAXIMUM_TILE_SITES, _SITE_TILE_CELLS // max(1, n_samples)))


def peak_scratch_bytes(n_samples: int, n_input_sites: int) -> int:
    """Conservative peak beyond input counts, including site fits and sample QC.

    Site vectors and sparse near-private cells are O(M); dense work is at most one bounded
    tile. The direct genotype queue admits at most min(live CPUs, tile) sample workspaces,
    also covered by the tile allowance. No extra adaptive allocation is made. The fixed
    allowance covers histogram/optimizer and allocator overhead.
    This reservation does not include the input counts or the later kept-site/GL arrays.
    """
    tile = min(n_input_sites, _tile_sites(n_samples))
    return int(512 * n_input_sites + 128 * n_samples * tile + 512 * n_samples + (16 << 20))


def _count_tile(counts, start, stop, selected=None):
    counts = counts[:, start:stop, :]
    if selected is not None:
        counts = counts[selected]
    alt = np.ascontiguousarray(counts[:, :, 1].T, dtype=np.int32)
    # Two int32 alleles: addition has the same overflow semantics as sum, without
    # a length-two reduction for every cell (repeated in all three depth passes).
    dep = np.ascontiguousarray((counts[:, :, 0] + counts[:, :, 1]).T)
    return alt, dep


def _quality_flags(n_sites, n_samples, read_tile, config, *, threads, counts=None):
    """The original rule calculations with O(sites) summaries and bounded dense tiles."""
    L, S = n_sites, n_samples
    tile = _tile_sites(S)
    D = np.empty(L)
    if counts is not None:
        # One live-budget work queue with only per-worker sample vectors.
        # The task count also bounds worker scratch for unusually large N.
        with numba_threads(min(threads, tile)):
            F, W, N, A, H, covered = _genotype_expectations_counts(
                counts, np.arange(S), config.classification_error,
                config.classification_heterozygote_alt, _SITE_EM_ITERATIONS)
        for start in range(0, L, tile):
            stop = min(start + tile, L)
            alt, dep = read_tile(start, stop)
            D[start:stop] = dep.sum(axis=1, dtype=np.float64)
        del alt, dep
    else:
        F = np.empty((L, 3))
        W = np.empty(L); N = np.empty(L); A = np.empty(L); H = np.empty(L)
        covered = np.empty(L, np.int32)
        for start in range(0, L, tile):
            stop = min(start + tile, L)
            alt, dep = read_tile(start, stop)
            D[start:stop] = dep.sum(axis=1, dtype=np.float64)
            values = _genotype_expectations(alt, dep, config.classification_error,
                                           config.classification_heterozygote_alt, _SITE_EM_ITERATIONS)
            for output, value in zip((F, W, N, A, H, covered), values):
                output[start:stop] = value
        del alt, dep, values
    parameters = dict(records=int(L), samples=int(S), snp=int(L))
    # Decisions taken by a convention (tied histogram maxima), reported in OUTPUT/ties.
    counter = ties.TieCounter()
    core = _peak_core(D, config.histogram_smoothing_bins, integer=True)
    counter.note("ev.depth.mode", inside_tie=core["tied_mode"])
    lo, hi = _bounds(core, config.core_posterior_level)
    r1 = (D < lo) | (D > hi)
    parameters["depth"] = dict(_core_record(core), low=lo, high=hi,
                               poisson_sd=float(np.sqrt(core["mode"])))
    reference = ~r1
    # Integer input depths have exact float64 sums on the supported contigs. Sum before
    # dividing, as the original full-array mean does, rather than averaging tile means.
    depth_sum = np.zeros(S)
    for start in range(0, L, tile):
        stop = min(start + tile, L)
        alt, dep = read_tile(start, stop)
        depth_sum += dep[reference[start:stop]].sum(axis=0, dtype=np.float64)
    mean_depth = depth_sum / reference.sum()
    X = np.empty(L)
    for start in range(0, L, tile):
        stop = min(start + tile, L)
        alt, dep = read_tile(start, stop)
        depth = dep.astype(np.float64)
        expected = np.outer(D[start:stop] / mean_depth.sum(), mean_depth)
        X[start:stop] = ((depth - expected) ** 2 / np.maximum(expected, 1e-9)).sum(1) / (S - 1)
    del alt, dep, depth, expected
    dispersed = reference & (X > 0)
    if dispersed.any():
        core = _peak_core(np.log(X[dispersed]), config.histogram_smoothing_bins)
        counter.note("ev.heterogeneity.mode", inside_tie=core["tied_mode"])
        _, high = _bounds(core, config.core_posterior_level)
        r2 = X > np.exp(high)
        parameters["heterogeneity"] = dict(mode=float(np.exp(core["mode"])), sd_log_left=core["sd_left"],
                                           sd_log_right=core["sd_right"], high=float(np.exp(high)),
                                           core_weight=core["core_weight"])
    else:
        # Every reference site's depths are exactly proportional to the samples' mean depths: there is
        # no heterogeneity to calibrate a core on, and no site exceeds one.
        r2 = np.zeros(L, bool)
        parameters["heterogeneity"] = dict(applied=False, reason="no reference site with depth heterogeneity")
    base = ~r1 & ~r2
    separation = config.minimum_artefact_separation
    # r3 wrong-allele reads in homozygotes
    m3 = base & (N >= 1)
    r3 = np.zeros(L, bool)
    r4 = np.zeros(L, bool)
    with ThreadPoolExecutor(threads) as pool:
        if m3.any():
            error, clean = _error_mixture(W[m3], N[m3], _Chunks(pool, int(m3.sum()), threads))
            error["applied"] = bool(error["eps1"] >= separation * error["eps0"])
            if error["applied"]:
                r3[np.where(m3)[0][clean < config.clean_posterior_level]] = True
        else:
            error = dict(applied=False, reason="no site with expected homozygote reads >= 1")
        # r4 heterozygote allele balance
        m4 = base & ~r3 & (H >= 1)
        if m4.any():
            balance, clean = _balance_mixture(A[m4], H[m4], _Chunks(pool, int(m4.sum()), threads))
    if m4.any():
        typical = float(np.median(H[m4]))
        balance["binomial_sd_at_median_reads"] = float(np.sqrt(balance["h0"] * (1 - balance["h0"]) / typical))
        balance["artefact_sd"] = float(np.sqrt(balance["artefact_mean"] * (1 - balance["artefact_mean"])
                                               / (balance["artefact_concentration"] + 1)))
        balance["applied"] = bool(
            balance["artefact_sd"] >= separation * balance["binomial_sd_at_median_reads"]
            or abs(balance["artefact_mean"] - balance["h0"]) >= separation * balance["binomial_sd_at_median_reads"])
        if balance["applied"]:
            r4[np.where(m4)[0][clean < config.clean_posterior_level]] = True
    else:
        # No heterozygote evidence to fit a clean balance on: the sample QC uses the configured
        # classification balance instead, and the record says so.
        balance = dict(applied=False, reason="no site with expected heterozygote reads >= 1",
                       h0=float(config.classification_heterozygote_alt), h0_source="classification balance")
    parameters["error"] = error
    parameters["balance"] = balance
    parameters["tie_events"] = counter.record()
    # r5 expected carriers of a non-major genotype among covered samples
    r5 = covered * (1 - F.max(1)) < config.minimum_expected_carriers
    return dict(r1=r1, r2=r2, r3=r3, r4=r4, r5=r5), parameters


def site_quality_flags(alt, dep, config: SiteQualityConfig, *, threads: int):
    """Compatibility entry point for (sites, samples) integer ALT and total depths."""
    return _quality_flags(len(dep), dep.shape[1],
                          lambda start, stop: (alt[start:stop], dep[start:stop]), config,
                          threads=threads)


def uninformative_sites(alt, dep, config: SiteQualityConfig):
    """r5 alone with bounded per-site genotype fits, preserving the original sample order."""
    result = np.empty(len(dep), bool)
    tile = _tile_sites(dep.shape[1])
    for start in range(0, len(dep), tile):
        stop = min(start + tile, len(dep))
        F, _, _, _, _, covered = _genotype_expectations(
            np.ascontiguousarray(alt[start:stop], dtype=np.int32),
            np.ascontiguousarray(dep[start:stop], dtype=np.int32),
            config.classification_error, config.classification_heterozygote_alt, _SITE_EM_ITERATIONS)
        result[start:stop] = covered * (1 - F.max(1)) < config.minimum_expected_carriers
    return result


def filter_sites(positions, counts, config: SiteQualityConfig, sample_config: SampleQualityConfig, *,
                 threads: int) -> SiteQualityResult:
    """Decide the site filter (and, with it, the contamination QC) on one contig's SNPs.

    positions int64[L], counts int32[N, L, 2] of the analysed samples. With an enabled
    ``sample_config``, the per-sample QC is decided on this contig's r1-r4-passing SNPs with r4's
    clean heterozygote balance, and r5 counts carriers among unflagged samples only.
    """
    n_sites = len(positions)
    if not config.enabled:
        return SiteQualityResult(enabled=False, applied=False, reason="site filter off",
                                 kept=np.ones(n_sites, bool))
    if n_sites < config.minimum_calibration_sites:
        return SiteQualityResult(
            enabled=True, applied=False, kept=np.ones(n_sites, bool),
            reason=f"fewer than {config.minimum_calibration_sites} biallelic SNPs")
    flags, parameters = _quality_flags(
        n_sites, counts.shape[0], lambda start, stop: _count_tile(counts, start, stop),
        config, threads=threads, counts=counts)
    samples = None
    if sample_config.enabled:
        passing = ~np.any([flags[rule] for rule in QUALITY_RULES], axis=0)
        samples = sample_quality.assess_counts(
            counts, positions, passing, parameters["balance"]["h0"], sample_config,
            chunk_sites=_tile_sites(counts.shape[0]))
        counted = ~samples.flagged
        if not np.all(counted):
            tile = _tile_sites(counts.shape[0])
            with numba_threads(min(threads, tile)):
                F, _, _, _, _, covered = _genotype_expectations_counts(
                    counts, np.flatnonzero(counted), config.classification_error,
                    config.classification_heterozygote_alt, _SITE_EM_ITERATIONS)
            flags["r5"] = covered * (1 - F.max(1)) < config.minimum_expected_carriers
    applied = RULES if config.drop_uninformative_sites else QUALITY_RULES
    remove = np.zeros(n_sites, bool)
    for rule in applied:
        remove |= flags[rule]
    only = {rule: int((flags[rule] & ~np.any([flags[other] for other in applied if other != rule],
                                             axis=0)).sum())
            for rule in applied}
    bits = np.zeros(n_sites, np.uint8)
    for rule in RULES:
        bits |= flags[rule].astype(np.uint8) << int(rule[1:])
    return SiteQualityResult(
        enabled=True, applied=True, reason=None, kept=~remove, rule_bits=bits,
        applied_rules=tuple(applied), flagged={rule: int(flags[rule].sum()) for rule in RULES},
        removed_only_by=only, parameters=parameters, sample_quality=samples,
        r5_excluded_samples=0 if samples is None else int(np.sum(samples.flagged)))


def identity(config: SiteQualityConfig) -> dict:
    return dict(version=VERSION, config=asdict(config))
