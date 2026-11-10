"""Self-calibrated site-quality filter on one contig's biallelic SNPs.

Each contig is calibrated on its own distributions, from allele depths only: no sample labels,
benchmark, pedigree or simulation truth. A site is removed when any applied rule fires:

  r1  total depth outside the clean core of total depth (both tails);
  r2  per-sample depth heterogeneity, the dispersion index of sample depths about the site's own
      depth factor, above its clean core;
  r3  wrong-allele reads in samples classified homozygous: a negative-binomial artefact class is
      more likely than the Poisson clean core;
  r4  heterozygote allele balance: of two beta-binomial classes (each including its binomial
      limit), the minority artefact class is more likely than the majority clean class;
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

import ctypes
import math

import numpy as np
from numba import prange, typeof, types
from numba.extending import get_cython_function_address
from scipy import special
from scipy.optimize import minimize

from haplotype_reconstruction.contracts.evidence import (
    BalanceMixture,
    BalanceStart,
    DepthCore,
    ErrorMixture,
    ErrorMSteps,
    HeterogeneityCore,
    SampleQualityDecision,
    SiteQualityDecision,
    SiteQualityFit,
)
from haplotype_reconstruction.core import ties
from haplotype_reconstruction.core.concurrency import ThreadSource, thread_count
from haplotype_reconstruction.core.kernels import kernel

from . import sample_quality
from .parameters import (
    BALANCE_BACKTRACK_TRIALS,
    BALANCE_FIT_OPTIONS,
    BALANCE_POLISH_DIFFERENCE_STEP,
    BALANCE_POLISH_ITERATIONS,
    BALANCE_POLISH_RELATIVE_STOP,
    BALANCE_RESTART_LIMIT,
    BALANCE_RESTART_OPTIONS,
    BALANCE_SERIES_RADIUS,
    BALANCE_START_LOGIT_BOUND,
    BALANCE_STARTS,
    CLEAN_RATE_RELATIVE_STEP,
    DEPTH_MEAN_FLOOR,
    ERROR_BACKTRACK_TRIALS,
    ERROR_INITIAL_CLEAN_WEIGHT,
    ERROR_INITIAL_RATE_CEILING,
    ERROR_INITIAL_RATE_FLOOR,
    ERROR_INITIAL_RATE_RATIO,
    ERROR_MIXTURE_RELATIVE_STOP,
    ERROR_MIXTURE_SHAPE_CEILING,
    ERROR_MIXTURE_SHAPE_FLOOR,
    ERROR_MIXTURE_WEIGHT_FLOOR,
    HISTOGRAM_MAXIMUM_BINS,
    HISTOGRAM_MINIMUM_BINS,
    HISTOGRAM_WIDTH_FLOOR,
    POSITIVE_WEIGHT_FLOOR,
    POSTERIOR_LOG_ODDS_BOUND,
    SITE_EM_FREQUENCY_FLOOR,
    SITE_EM_INITIAL_LOG_LIKELIHOOD,
    SITE_EM_LOG_GAIN_TOLERANCE,
    SampleQualityParameters,
    SiteQualityParameters,
)
from .parameters import (
    SITE_BALANCE_BOUNDS as _BALANCE_BOUNDS,
)
from .parameters import (
    SITE_BALANCE_GRADIENT as _BALANCE_GRADIENT,
)
from .parameters import (
    SITE_BALANCE_STIRLING as _BALANCE_STIRLING,
)
from .parameters import (
    SITE_BALANCE_T_SCALE as _BALANCE_T_SCALE,
)
from .parameters import (
    SITE_CLEAN_RATE_BOUNDS as _CLEAN_RATE_BOUNDS,
)
from .parameters import (
    SITE_ERROR_LOWER as _ERROR_LOWER,
)
from .parameters import (
    SITE_ERROR_UPPER as _ERROR_UPPER,
)
from .parameters import (
    SITE_HISTOGRAM_RANGE_PERCENTILES as _HISTOGRAM_RANGE_PERCENTILES,
)
from .parameters import (
    SITE_LEAKAGE_SLOPE_FLOOR as _LEAKAGE_SLOPE_FLOOR,
)
from .parameters import (
    SITE_M_STEP_CHUNKS as _M_STEP_CHUNKS,
)
from .parameters import (
    SITE_M_STEP_GRADIENT as _M_STEP_GRADIENT,
)
from .parameters import (
    SITE_M_STEP_ITERATIONS as _M_STEP_ITERATIONS,
)
from .parameters import (
    SITE_M_STEP_NOISE as _M_STEP_NOISE,
)
from .parameters import (
    SITE_M_STEP_ROUNDING as _M_STEP_ROUNDING,
)
from .parameters import (
    SITE_M_STEP_STEP as _M_STEP_STEP,
)
from .parameters import (
    SITE_MAXIMUM_TILE_SITES as _MAXIMUM_TILE_SITES,
)
from .parameters import (
    SITE_MIXTURE_ITERATIONS as _MIXTURE_ITERATIONS,
)
from .parameters import (
    SITE_POLYGAMMA_SHIFT as _POLYGAMMA_SHIFT,
)
from .parameters import (
    SITE_SITE_EM_ITERATIONS as _SITE_EM_ITERATIONS,
)
from .parameters import (
    SITE_SITE_TILE_CELLS as _SITE_TILE_CELLS,
)

RULES = ("r1", "r2", "r3", "r4", "r5")
QUALITY_RULES = RULES[:4]


@kernel(
    overloads=(
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
            types.float64,
            types.int64,
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
)
def _site_genotype_fit(a, n, eps, h, iters, f, ll):
    """EM over one site's free genotype frequencies (no Hardy-Weinberg).

    alt reads ~ Binomial(depth, p_g), p = (eps, h, 1 - eps). Returns the expected wrong-allele
    reads and depth in homozygotes and the expected ALT reads and depth in heterozygotes from the
    last E-step; f holds the updated frequencies.
    """
    S = len(n)
    le, l1e = (math.log(eps), math.log(1.0 - eps))
    lh, l1h = (math.log(h), math.log(1.0 - h))
    for s in range(S):
        ll[s, 0] = a[s] * le + (n[s] - a[s]) * l1e
        ll[s, 1] = a[s] * lh + (n[s] - a[s]) * l1h
        ll[s, 2] = a[s] * l1e + (n[s] - a[s]) * le
    maximum = np.empty(S)
    for s in range(S):
        maximum[s] = max(ll[s, 0], ll[s, 1], ll[s, 2])
        for g in range(3):
            ll[s, g] = math.exp(ll[s, g] - maximum[s])
    f[0] = f[1] = f[2] = 1.0 / 3.0
    old = SITE_EM_INITIAL_LOG_LIKELIHOOD
    wrong = 0.0
    homn = 0.0
    heta = 0.0
    hetn = 0.0
    for it in range(iters):
        value = 0.0
        c0 = c1 = c2 = 0.0
        wrong = 0.0
        homn = 0.0
        heta = 0.0
        hetn = 0.0
        for s in range(S):
            if n[s] == 0:
                continue
            m = maximum[s]
            p0 = f[0] * ll[s, 0]
            p1 = f[1] * ll[s, 1]
            p2 = f[2] * ll[s, 2]
            t = p0 + p1 + p2
            value += m + math.log(t)
            p0 /= t
            p1 /= t
            p2 /= t
            c0 += p0
            c1 += p1
            c2 += p2
            wrong += p0 * a[s] + p2 * (n[s] - a[s])
            homn += (p0 + p2) * n[s]
            heta += p1 * a[s]
            hetn += p1 * n[s]
        tot = c0 + c1 + c2
        f[0] = max(c0 / tot, SITE_EM_FREQUENCY_FLOOR)
        f[1] = max(c1 / tot, SITE_EM_FREQUENCY_FLOOR)
        f[2] = max(c2 / tot, SITE_EM_FREQUENCY_FLOOR)
        z = f[0] + f[1] + f[2]
        f[0] /= z
        f[1] /= z
        f[2] /= z
        if it > 3 and value - old < SITE_EM_LOG_GAIN_TOLERANCE:
            break
        old = value
    return (wrong, homn, heta, hetn)


@kernel(
    overloads=(
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
)
def _leakage_terms(n, f, eps, h, binomial=None):
    """the expectation of a site's homozygote wrong-read statistic W under a clean site with error
    rate e, E[W] = lam + kap e, exact to first order in e. W weights each cell's reads by the classification's
    posteriors, w(a, d) = p0 a + p2 (d - a). A heterozygote's reads ~ Bin(d, h), so heterozygotes add
    f1 sum_a Bin(a; d, h) w(a, d) whatever e; an error-free homozygote adds w(0, d) (hom-ref) or w(d, d) (hom-alt),
    its own reads weighted by the posterior of the other homozygote (not small at depth 1-2); each error read
    (probability d e to first order) moves a homozygote from w(0, d) to w(1, d) (hom-ref) or from w(d, d) to
    w(d - 1, d) (hom-alt). So lam = sum over cells of f1 H(d) + f0 w(0, d) + f2 w(d, d) and
    kap = sum over cells of d (f0 (w(1, d) - w(0, d)) + f2 (w(d - 1, d) - w(d, d))). Same read model as the
    classification (eps, h) and the site's fitted frequencies f; cells cached by depth."""
    S = len(n)
    dmax = 0
    for s in range(S):
        if n[s] > dmax:
            dmax = int(n[s])
    leak = np.full(dmax + 1, -1.0)
    slope = np.zeros(dmax + 1)
    le, l1e = (math.log(eps), math.log(1.0 - eps))
    lh, l1h = (math.log(h), math.log(1.0 - h))
    lf0, lf1, lf2 = (math.log(f[0]), math.log(f[1]), math.log(f[2]))
    lam = 0.0
    kap = 0.0
    for s in range(S):
        d = int(n[s])
        if d == 0:
            continue
        if leak[d] < 0.0:
            het = 0.0
            w0 = w1 = wd1 = wd = 0.0
            lgd = math.lgamma(d + 1.0)
            for a in range(d + 1):
                l0 = lf0 + a * le + (d - a) * l1e
                l1 = lf1 + a * lh + (d - a) * l1h
                l2 = lf2 + a * l1e + (d - a) * le
                m = max(l0, l1, l2)
                e0 = math.exp(l0 - m)
                e1 = math.exp(l1 - m)
                e2 = math.exp(l2 - m)
                w = (e0 * a + e2 * (d - a)) / (e0 + e1 + e2)
                probability = (
                    binomial[d, a]
                    if binomial is not None and d < len(binomial)
                    else math.exp(lgd - math.lgamma(a + 1.0) - math.lgamma(d - a + 1.0) + a * lh + (d - a) * l1h)
                )
                het += probability * w
                if a == 0:
                    w0 = w
                if a == 1:
                    w1 = w
                if a == d - 1:
                    wd1 = w
                if a == d:
                    wd = w
            leak[d] = f[1] * het + f[0] * w0 + f[2] * wd
            slope[d] = d * (f[0] * (w1 - w0) + f[2] * (wd1 - wd))
        lam += leak[d]
        kap += slope[d]
    return (lam, kap)


@kernel(overloads=((types.float64,),), cache=True, nogil=True)
def _depth_binomial(h):
    """Read-model terms shared by every site's leakage calculation. Deeper cells use direct evaluation of the read-model terms."""
    width = 129
    out = np.zeros((width, width))
    lh, l1h = (math.log(h), math.log(1.0 - h))
    for d in range(width):
        lgd = math.lgamma(d + 1.0)
        for a in range(d + 1):
            out[d, a] = math.exp(lgd - math.lgamma(a + 1.0) - math.lgamma(d - a + 1.0) + a * lh + (d - a) * l1h)
    return out


@kernel(
    overloads=(
        (
            types.Array(types.int32, 3, "C", readonly=True),
            types.Array(types.int64, 1, "C", readonly=False),
            types.float64,
            types.float64,
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int32, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
)
def _genotype_counts_into(
    counts, samples, eps, h, iters, start, stop, frequencies, wrong, homn, heta, hetn, covered, leak, slope, binomial
):
    S = len(samples)
    for i in range(start, stop):
        a = np.empty(S)
        n = np.empty(S)
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
        f = np.zeros(3)
        ll = np.zeros((S, 3))
        w, hn, ha, hh = _site_genotype_fit(a, n, eps, h, iters, f, ll)
        wrong[i] = w
        homn[i] = hn
        heta[i] = ha
        hetn[i] = hh
        leak[i], slope[i] = _leakage_terms(n, f, eps, h, binomial)
        frequencies[i, 0] = f[0]
        frequencies[i, 1] = f[1]
        frequencies[i, 2] = f[2]


def _genotype_expectations_counts(counts, samples, eps, h, iters, team: ThreadSource = 1, *, map_work=map):
    """Direct-count chunks with disjoint output slices, evaluated by the caller's ordered map."""
    L = counts.shape[1]
    values = (
        np.zeros((L, 3)),
        np.zeros(L),
        np.zeros(L),
        np.zeros(L),
        np.zeros(L),
        np.zeros(L, np.int32),
        np.zeros(L),
        np.zeros(L),
    )
    if L == 0:
        return values
    capacity = _tile_sites(counts.shape[0])
    tasks = min(capacity, 4 * thread_count(team))
    span = max(1, (L + tasks - 1) // tasks)
    chunks = [(start, min(start + span, L)) for start in range(0, L, span)]
    chunks.sort(key=lambda bounds: bounds[1] - bounds[0], reverse=True)
    binomial = _depth_binomial(h)
    _genotype_counts_into(counts, samples, eps, h, iters, 0, 0, *values, binomial)

    def work(bounds):
        _genotype_counts_into(counts, samples, eps, h, iters, *bounds, *values, binomial)

    list(map_work(work, chunks))
    return values


def _peak_core(x, smooth, integer=False, *, reverse=False):
    """Mode and left/right Gaussian-equivalent sds from the half-maximum crossings of a smoothed histogram."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, _HISTOGRAM_RANGE_PERCENTILES)
    q25, q75 = np.percentile(x, [25, 75])
    width = 2 * (q75 - q25) / len(x) ** (1 / 3)
    if integer:
        width = max(1.0, float(np.round(width)))
        lo = np.floor(lo) - 0.5
        nb = int(np.ceil((hi - lo) / width))
        hi = lo + nb * width
    else:
        nb = int(np.clip((hi - lo) / max(width, HISTOGRAM_WIDTH_FLOOR), HISTOGRAM_MINIMUM_BINS, HISTOGRAM_MAXIMUM_BINS))
    counts, edges = np.histogram(x, bins=nb, range=(lo, hi))
    centres = 0.5 * (edges[1:] + edges[:-1])
    kernel = np.ones(smooth) / smooth
    if nb >= smooth:
        smoothed = np.convolve(counts, kernel, mode="same")
    else:
        smoothed = np.convolve(counts, kernel, mode="full")[(smooth - 1) // 2 : (smooth - 1) // 2 + nb]
    p = ties.conv_argmax(smoothed, reverse=reverse)
    top = smoothed[p]

    def cross(direction):
        i = p
        while 0 <= i + direction < len(smoothed) and smoothed[i] > top / 2:
            i += direction
        j = i - direction
        if not 0 <= j < len(smoothed) or smoothed[j] == smoothed[i]:
            return centres[i]
        return centres[j] + (top / 2 - smoothed[j]) * (centres[i] - centres[j]) / (smoothed[i] - smoothed[j])

    left, right = (cross(-1), cross(+1))
    f = np.sqrt(2 * np.log(2))
    density = top / (len(x) * (edges[1] - edges[0]))
    sd_left, sd_right = ((centres[p] - left) / f, (right - centres[p]) / f)
    return dict(
        mode=float(centres[p]),
        sd_left=float(sd_left),
        sd_right=float(sd_right),
        peak_density=float(density),
        tied_mode=bool(np.count_nonzero(smoothed == top) > 1),
        core_weight=float(density * np.sqrt(2 * np.pi) * 0.5 * (sd_left + sd_right)),
        bins=nb,
        centres=centres,
        smooth=smoothed,
        n=len(x),
        width=float(edges[1] - edges[0]),
    )


def _bounds(core, level):
    """Window around the mode inside which the core's local posterior is >= level."""
    if core["sd_left"] == 0.0 and core["sd_right"] == 0.0:
        return (-np.inf, np.inf)
    c = core["centres"]
    density = core["smooth"] / (core["n"] * core["width"])
    sd = np.where(c < core["mode"], core["sd_left"], core["sd_right"])
    z = np.divide(c - core["mode"], sd, out=np.where(c == core["mode"], 0.0, np.inf), where=sd > 0)
    null = core["peak_density"] * np.exp(-0.5 * z**2)
    posterior = np.where(density > 0, np.clip(null / np.maximum(density, POSITIVE_WEIGHT_FLOOR), 0, 1), 1.0)
    p = int(np.argmin(np.abs(c - core["mode"])))
    i = p
    while i > 0 and posterior[i - 1] >= level:
        i -= 1
    j = p
    while j < len(c) - 1 and posterior[j + 1] >= level:
        j += 1
    lo = -np.inf if i == 0 else c[i] - 0.5 * core["width"]
    hi = np.inf if j == len(c) - 1 else c[j] + 0.5 * core["width"]
    return (float(lo), float(hi))


def _core_record(core):
    return {key: core[key] for key in ("mode", "sd_left", "sd_right", "core_weight")}


def _log_poisson_pmf(W, mu, log_w_factorial):
    return W * np.log(np.maximum(mu, POSITIVE_WEIGHT_FLOOR)) - mu - log_w_factorial


def _posterior_clean(log_clean, log_artefact):
    return 1 / (1 + np.exp(np.clip(log_artefact - log_clean, -POSTERIOR_LOG_ODDS_BOUND, POSTERIOR_LOG_ODDS_BOUND)))


_GAMMALN = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_int)(
    get_cython_function_address("scipy.special.cython_special", "gammaln")
)


@kernel(
    overloads=(
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
            types.float64,
            types.float64,
            typeof(_GAMMALN),
        ),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _negative_binomial_values(W, lam, kap, lw, rate, shape, gamma_shape, function):
    result = np.empty(len(W))
    for i in prange(len(W)):
        mu = lam[i] + kap[i] * rate
        result[i] = (
            function(W[i] + shape, 1)
            - gamma_shape
            - lw[i]
            + shape * math.log(shape / (shape + mu))
            + W[i] * math.log(mu / (shape + mu))
        )
    return result


@kernel(overloads=((types.float64,),), cache=True, nogil=True)
def _digamma(x):
    out = 0.0
    while x < _POLYGAMMA_SHIFT:
        out -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    z = inv * inv
    return (
        out
        + math.log(x)
        - 0.5 * inv
        - z
        * (
            1 / 12
            - z
            * (
                1 / 120
                - z * (1 / 252 - z * (1 / 240 - z * (1 / 132 - z * (691 / 32760 - z * (1 / 12 - z * 3617 / 8160)))))
            )
        )
    )


@kernel(overloads=((types.float64,),), cache=True, nogil=True)
def _trigamma(x):
    out = 0.0
    while x < _POLYGAMMA_SHIFT:
        out += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    z = inv * inv
    return (
        out
        + inv
        + 0.5 * z
        + inv * z * (1 / 6 - z * (1 / 30 - z * (1 / 42 - z * (1 / 30 - z * (5 / 66 - z * (691 / 2730 - z * 7 / 6))))))
    )


@kernel(
    overloads=(
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
            types.float64,
        ),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _negative_binomial_terms(W, lam, kap, w, rate, k):
    """Weighted sums over fixed site chunks of log NegBin(W | mean lam + kap rate, shape k) (without log W!) and its
    first and second derivatives in (log rate, log k): rows of (value, gx, gy, hxx, hyy, hxy). The mean's derivative
    in log rate is g = kap rate (= the mean when lam = 0)."""
    n = len(W)
    out = np.zeros((_M_STEP_CHUNKS, 6))
    log_k, digamma_k, trigamma_k, lgamma_k = (math.log(k), _digamma(k), _trigamma(k), math.lgamma(k))
    for chunk in prange(_M_STEP_CHUNKS):
        s0 = s1 = s2 = s3 = s4 = s5 = 0.0
        for i in range(chunk * n // _M_STEP_CHUNKS, (chunk + 1) * n // _M_STEP_CHUNKS):
            if w[i] == 0.0:
                continue
            x, g = (W[i], kap[i] * rate)
            mu = lam[i] + g
            km = k + mu
            log_km = math.log(km)
            d_mu = x / mu - (x + k) / km
            d_k = _digamma(x + k) - digamma_k + log_k + 1.0 - log_km - (x + k) / km
            s0 += w[i] * (math.lgamma(x + k) - lgamma_k + k * (log_k - log_km) + x * (math.log(mu) - log_km))
            s1 += w[i] * d_mu * g
            s2 += w[i] * d_k * k
            s3 += w[i] * ((-x / (mu * mu) + (x + k) / (km * km)) * g * g + d_mu * g)
            s4 += w[i] * ((_trigamma(x + k) - trigamma_k + 1.0 / k - 1.0 / km - (mu - x) / (km * km)) * k * k + d_k * k)
            s5 += w[i] * (x - mu) / (km * km) * g * k
        out[chunk, 0], out[chunk, 1], out[chunk, 2] = (s0, s1, s2)
        out[chunk, 3], out[chunk, 4], out[chunk, 5] = (s3, s4, s5)
    return out


def _chunk_totals(rows):
    total = np.zeros(rows.shape[1])
    for row in rows:
        total += row
    return total


def _negative_binomial_objective(q, W, lam, kap, w):
    """Weighted artefact log likelihood of r3 at q = (log eps1, log k), its gradient and Hessian."""
    total = _chunk_totals(_negative_binomial_terms(W, lam, kap, w, math.exp(q[0]), math.exp(q[1])))
    return (total[0], total[1:3], np.array([[total[3], total[5]], [total[5], total[4]]]))


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
        free = ~((q <= lower) & (gradient < 0) | (q >= upper) & (gradient > 0))
        return (free, float(np.abs(gradient[free]).max()) / scale if free.any() else 0.0)

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
        alpha, accepted = (1.0, False)
        for _halving in range(ERROR_BACKTRACK_TRIALS):
            candidate = np.clip(q + alpha * step, lower, upper)
            candidate_value, candidate_gradient, candidate_hessian = objective(candidate)
            if not np.isfinite(candidate_value):
                alpha *= 0.5
                continue
            if candidate_value >= value - allowance or (
                newton
                and candidate_value >= value - _M_STEP_NOISE * max(1.0, abs(value))
                and (projected_gradient(candidate, candidate_gradient)[1] < projected)
            ):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            status = "line_search"
            break
        q, value, gradient, hessian = (candidate, candidate_value, candidate_gradient, candidate_hessian)
    return (q, status, projected)


def _tally(statuses, status, gradient):
    """Count an M-step's outcome; keep the largest returned gradient over all of the fit's M-steps, whatever their
    status."""
    statuses[status] = statuses.get(status, 0) + 1
    statuses["worst_gradient"] = max(statuses.get("worst_gradient", 0.0), gradient)


def _clean_rate(W, lam, kap, wc, start):
    """the clean core's rate e0 in [_CLEAN_RATE_BOUNDS] maximising
    sum wc (W log(lam + kap e) - kap e), concave in e (its gradient decreases). A boundary is returned when the
    gradient does not change sign inside; otherwise Newton steps from ``start`` (clipped into the bracket) keep a
    bracket of the root and fall back to bisection when a step leaves it. With lam = 0 the root is the closed form
    sum wc W / sum wc kap (clipped)."""
    low, high = _CLEAN_RATE_BOUNDS
    if not np.any(lam > 0.0):
        return float(np.clip((wc * W).sum() / (wc * kap).sum(), low, high))

    def gradient_and_curvature(e):
        mu = lam + kap * e
        return (float(np.sum(wc * (W * kap / mu - kap))), float(np.sum(wc * W * kap * kap / (mu * mu))))

    if gradient_and_curvature(low)[0] <= 0.0:
        return low
    if gradient_and_curvature(high)[0] >= 0.0:
        return high
    e = min(max(float(start), low), high)
    for _ in range(_MIXTURE_ITERATIONS):
        gradient, curvature = gradient_and_curvature(e)
        if gradient > 0.0:
            low = e
        else:
            high = e
        new = e + gradient / curvature if curvature > 0.0 else 0.5 * (low + high)
        if not low < new < high:
            new = 0.5 * (low + high)
        if abs(new - e) <= CLEAN_RATE_RELATIVE_STEP * e:
            return new
        e = new
    return e


def _error_mixture(W, lam, kap):
    """Clean: W ~ Poisson(lam + kap eps0). Artefact: W ~ NegBin(mean lam + kap eps1, shape k). Returns parameters
    and P(clean). lam and kap describe the clean-site expectation of W from _leakage_terms."""
    pi = ERROR_INITIAL_CLEAN_WEIGHT
    e0 = float(
        np.clip(np.median(np.maximum(W - lam, 0.0) / kap) / 2, ERROR_INITIAL_RATE_FLOOR, ERROR_INITIAL_RATE_CEILING)
    )
    e1 = ERROR_INITIAL_RATE_RATIO * e0
    k = 1.0
    lw = special.gammaln(W + 1)
    m_steps = {}

    def artefact(rate, shape):
        return _negative_binomial_values(W, lam, kap, lw, rate, shape, special.gammaln(shape), _GAMMALN)

    for _iteration in range(_MIXTURE_ITERATIONS):
        wc = _posterior_clean(np.log(pi) + _log_poisson_pmf(W, lam + kap * e0, lw), np.log1p(-pi) + artefact(e1, k))
        pi2 = float(np.clip(wc.mean(), ERROR_MIXTURE_WEIGHT_FLOOR, 1 - ERROR_MIXTURE_WEIGHT_FLOOR))
        e02 = _clean_rate(W, lam, kap, wc, e0)
        wa = 1 - wc
        if wa.sum() > ERROR_MIXTURE_WEIGHT_FLOOR:
            weight = float(wa.sum())
            q, status, gradient = _newton_maximum(
                lambda q, wa=wa: _negative_binomial_objective(q, W, lam, kap, wa),
                [np.log(e1), np.log(k)],
                _ERROR_LOWER,
                _ERROR_UPPER,
                weight,
            )
            _tally(m_steps, status, gradient)
            e12, k2 = (
                float(np.exp(q[0])),
                float(np.clip(np.exp(q[1]), ERROR_MIXTURE_SHAPE_FLOOR, ERROR_MIXTURE_SHAPE_CEILING)),
            )
        else:
            e12, k2 = (e1, k)
        change = abs(e02 - e0) / e0 + abs(pi2 - pi) + abs(e12 - e1) / e1
        pi, e0, e1, k = (pi2, e02, e12, k2)
        if change < ERROR_MIXTURE_RELATIVE_STOP:
            break
    wc = _posterior_clean(np.log(pi) + _log_poisson_pmf(W, lam + kap * e0, lw), np.log1p(-pi) + artefact(e1, k))
    clean = float((wc * (lam + kap * e0)).sum())
    return (
        dict(
            pi_clean=pi,
            eps0=e0,
            eps1=e1,
            shape=k,
            iterations=_iteration + 1,
            m_steps=m_steps,
            heterozygote_leakage=float(lam.sum() / max(W.sum(), POSITIVE_WEIGHT_FLOOR)),
            clean_leakage=float((wc * lam).sum() / clean) if clean > 0 else 0.0,
        ),
        wc,
    )


@kernel(overloads=((types.float64,),), cache=True, nogil=True)
def _balance_log1pmx_over_square(r):
    """(log(1 + r) - r) / r^2; near zero its Horner series through r^8."""
    if abs(r) < BALANCE_SERIES_RADIUS:
        return -0.5 + r * (
            1 / 3 + r * (-0.25 + r * (0.2 + r * (-1 / 6 + r * (1 / 7 + r * (-0.125 + r * (1 / 9 - r * 0.1))))))
        )
    return (math.log1p(r) - r) / (r * r)


@kernel(
    overloads=(
        (
            types.float64,
            types.float64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _balance_gamma_ratio_remainder(t, y):
    """J(t, y) = log Gamma(1/t + y) - log Gamma(1/t) + y log t and dJ/dt; J(0, y) = 0 with the exact limiting
    derivative y (y - 1) / 2 (the binomial boundary)."""
    if t == 0.0:
        return (0.0, 0.5 * y * (y - 1.0))
    if t > 0.125:
        x = 1.0 / t
        value = math.lgamma(x + y) - math.lgamma(x) + y * math.log(t)
        derivative = (_digamma(x) - _digamma(x + y) + y * t) / (t * t)
        return (value, derivative)
    r = y * t
    lr = math.log1p(r)
    q = _balance_log1pmx_over_square(r)
    value = (y - 0.5) * lr + y * y * t * q
    derivative = -0.5 * y / (1.0 + r) - y * y * q
    power = t
    for i in range(len(_BALANCE_STIRLING)):
        k = 2 * i + 1
        coefficient = _BALANCE_STIRLING[i]
        value += coefficient * power * math.expm1(-k * lr)
        derivative += k * coefficient * (power / t) * math.expm1(-(k + 1) * lr)
        power *= t * t
    return (value, derivative)


@kernel(
    overloads=(
        (
            types.float64,
            types.float64,
            types.float64,
            types.float64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _balance_beta_kernel(a, h, m, t):
    """A component's log likelihood of (A, H) = (a, h) relative to Binomial(h, 1/2), and its derivatives in m and t."""
    b = h - a
    q = 1.0 - m
    ja, da = _balance_gamma_ratio_remainder(t / m, a)
    jb, db = _balance_gamma_ratio_remainder(t / q, b)
    jh, dh = _balance_gamma_ratio_remainder(t, h)
    value = a * math.log(2 * m) + b * math.log(2 * q) + ja + jb - jh
    dm = a / m - b / q - t * da / (m * m) + t * db / (q * q)
    dt = da / m + db / q - dh
    return (value, dm, dt)


@kernel(
    overloads=(
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
            types.float64,
            types.float64,
            types.float64,
            types.float64,
        ),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _balance_mixture_terms(a, h, pi, m0, t0, m1, t1):
    """Per site: the mixture log likelihood (relative to Binomial(h, 1/2)) and its derivatives in the five
    coordinates (logit pi, logit m0, log1p(t0 / scale), logit m1, log1p(t1 / scale))."""
    out = np.empty((len(a), 6), np.float64)
    lpi = math.log(pi)
    lq = math.log1p(-pi)
    for i in prange(len(a)):
        v0, d0, e0 = _balance_beta_kernel(a[i], h[i], m0, t0)
        v1, d1, e1 = _balance_beta_kernel(a[i], h[i], m1, t1)
        x = lpi + v0
        y = lq + v1
        z = max(x, y)
        u = math.exp(x - z)
        v = math.exp(y - z)
        den = u + v
        w = u / den
        out[i, 0] = z + math.log(den)
        out[i, 1] = w - pi
        out[i, 2] = w * d0 * m0 * (1 - m0)
        out[i, 3] = w * e0 * (_BALANCE_T_SCALE + t0)
        out[i, 4] = (1 - w) * d1 * m1 * (1 - m1)
        out[i, 5] = (1 - w) * e1 * (_BALANCE_T_SCALE + t1)
    return out


@kernel(
    overloads=(
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
            types.float64,
            types.float64,
            types.float64,
            types.float64,
        ),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _balance_posterior(a, h, pi, m0, t0, m1, t1):
    """Per site: the posterior probability of the first component."""
    out = np.empty(len(a))
    for i in prange(len(a)):
        v0, _, _ = _balance_beta_kernel(a[i], h[i], m0, t0)
        v1, _, _ = _balance_beta_kernel(a[i], h[i], m1, t1)
        difference = math.log1p(-pi) + v1 - math.log(pi) - v0
        out[i] = 1.0 / (1.0 + math.exp(min(max(difference, -700.0), 700.0)))
    return out


def _balance_encode(pi, m0, t0, m1, t1):
    return np.array(
        [
            math.log(pi / (1 - pi)),
            math.log(m0 / (1 - m0)),
            math.log1p(t0 / _BALANCE_T_SCALE),
            math.log(m1 / (1 - m1)),
            math.log1p(t1 / _BALANCE_T_SCALE),
        ]
    )


def _balance_decode(p):
    return (
        1 / (1 + math.exp(-p[0])),
        1 / (1 + math.exp(-p[1])),
        _BALANCE_T_SCALE * math.expm1(p[2]),
        1 / (1 + math.exp(-p[3])),
        _BALANCE_T_SCALE * math.expm1(p[4]),
    )


def _balance_objective(p, a, h):
    """Minus the mean per-site mixture log likelihood at coordinates p, and its gradient."""
    totals = _balance_mixture_terms(a, h, *_balance_decode(p)).sum(axis=0) / len(a)
    return (-totals[0], -totals[1:])


def _balance_projected_gradient(p, g):
    out = np.array(g, copy=True)
    for i, (low, high) in enumerate(_BALANCE_BOUNDS):
        if p[i] <= low and g[i] > 0 or (p[i] >= high and g[i] < 0):
            out[i] = 0.0
    return out


def _balance_polish(p, a, h):
    """At most eight positive-curvature Newton steps on the score (finite-difference Hessian of the analytic
    gradient), active bounds held, each step accepted only when it reduces the projected gradient. Near the optimum
    the objective's improvements, of order score^2, fall below floating-point resolution, so solving the score avoids
    a false stop on rounded likelihood changes."""
    p = np.array(p, copy=True)
    value, g = _balance_objective(p, a, h)
    iterations = 0
    smallest_eigenvalue = None
    for _ in range(BALANCE_POLISH_ITERATIONS):
        pg = _balance_projected_gradient(p, g)
        if np.max(np.abs(pg)) < BALANCE_POLISH_RELATIVE_STOP:
            break
        free = np.array(
            [
                i
                for i, (low, high) in enumerate(_BALANCE_BOUNDS)
                if not (p[i] <= low and g[i] >= 0 or (p[i] >= high and g[i] <= 0))
            ]
        )
        if not len(free):
            break
        hessian = np.empty((len(free), len(free)))
        for j, index in enumerate(free):
            step = BALANCE_POLISH_DIFFERENCE_STEP * max(1.0, abs(p[index]))
            low, high = _BALANCE_BOUNDS[index]
            left = p.copy()
            right = p.copy()
            left[index] = max(low, p[index] - step)
            right[index] = min(high, p[index] + step)
            hessian[:, j] = (_balance_objective(right, a, h)[1][free] - _balance_objective(left, a, h)[1][free]) / (
                right[index] - left[index]
            )
        hessian = (hessian + hessian.T) * 0.5
        eigenvalues = np.linalg.eigvalsh(hessian)
        smallest_eigenvalue = float(eigenvalues[0])
        if eigenvalues[0] <= 0:
            break
        delta = np.linalg.solve(hessian, -g[free])
        alpha = 1.0
        for j, index in enumerate(free):
            low, high = _BALANCE_BOUNDS[index]
            if delta[j] < 0:
                alpha = min(alpha, (low - p[index]) / delta[j])
            if delta[j] > 0:
                alpha = min(alpha, (high - p[index]) / delta[j])
        accepted = False
        for _backtrack in range(BALANCE_BACKTRACK_TRIALS):
            candidate = p.copy()
            candidate[free] += alpha * delta
            for index in free:
                candidate[index] = min(max(candidate[index], _BALANCE_BOUNDS[index][0]), _BALANCE_BOUNDS[index][1])
            candidate_value, candidate_g = _balance_objective(candidate, a, h)
            if np.linalg.norm(_balance_projected_gradient(candidate, candidate_g)) < np.linalg.norm(pg):
                p, value, g = (candidate, candidate_value, candidate_g)
                accepted = True
                iterations += 1
                break
            alpha *= 0.5
        if not accepted:
            break
    return (p, float(value), g, iterations, smallest_eigenvalue)


def _balance_fit(A, H):
    """The two-component fit from fixed starts. Each start's parameters (pi, m0, t0, m1, t1) are labelled by weight:
    the majority component first (pi >= 1/2; at exactly equal weights, the smaller (latent variance, mean, t) first).
    Returns the best start's record (the highest likelihood among certified starts, else among all; exact ties by the
    parameters, then the start), all starts' records, whether any start is certified, and each site's posterior of
    the first component in input order."""
    a = np.asarray(A, float)
    h = np.asarray(H, float)
    if len(a) == 0 or not np.all((h >= 1) & (a >= 0) & (a <= h)):
        raise ValueError("expected H >= 1 and 0 <= A <= H")
    order = np.lexsort((a, h))
    aa = np.ascontiguousarray(a[order])
    hh = np.ascontiguousarray(h[order])
    mean = float(aa.sum() / hh.sum())
    low, high = (1 / (1 + math.exp(BALANCE_START_LOGIT_BOUND)), 1 / (1 + math.exp(-BALANCE_START_LOGIT_BOUND)))
    starts = sorted(
        set(
            (
                (float(pi), float(np.clip(m0, low, high)), float(t0), float(np.clip(m1, low, high)), float(t1))
                for pi, m0, t0, m1, t1 in (
                    (pi, mean + offset0, t0, mean + offset1, t1) for pi, offset0, t0, offset1, t1 in BALANCE_STARTS
                )
            )
        )
    )
    results = []
    for start in starts:
        r = minimize(
            _balance_objective,
            _balance_encode(*start),
            args=(aa, hh),
            jac=True,
            method="L-BFGS-B",
            bounds=_BALANCE_BOUNDS,
            options=BALANCE_FIT_OPTIONS,
        )
        value, g = _balance_objective(r.x, aa, hh)
        projected = float(np.max(np.abs(_balance_projected_gradient(r.x, g))))
        retries = 0
        while projected > _BALANCE_GRADIENT and retries < BALANCE_RESTART_LIMIT:
            r = minimize(
                _balance_objective,
                r.x,
                args=(aa, hh),
                jac=True,
                method="L-BFGS-B",
                bounds=_BALANCE_BOUNDS,
                options=BALANCE_RESTART_OPTIONS,
            )
            value, g = _balance_objective(r.x, aa, hh)
            projected = float(np.max(np.abs(_balance_projected_gradient(r.x, g))))
            retries += 1
        polished, value, g, polish_iterations, curvature = _balance_polish(r.x, aa, hh)
        projected = float(np.max(np.abs(_balance_projected_gradient(polished, g))))
        pi, m0, t0, m1, t1 = _balance_decode(polished)
        if pi < 0.5 or (
            pi == 0.5 and (m0 * (1 - m0) * t0 / (1 + t0), m0, t0) > (m1 * (1 - m1) * t1 / (1 + t1), m1, t1)
        ):
            pi, m0, t0, m1, t1 = (1 - pi, m1, t1, m0, t0)
        results.append(
            dict(
                start=list(start),
                parameters=[pi, m0, t0, m1, t1],
                objective=float(value),
                projected_gradient=projected,
                converged=bool(projected <= _BALANCE_GRADIENT),
                optimizer_success=bool(r.success),
                message=str(r.message),
                iterations=int(r.nit),
                retries=retries,
                polish_iterations=polish_iterations,
                minimum_curvature=curvature,
            )
        )
    certified = [result for result in results if result["converged"]]
    best = min(
        certified or results,
        key=lambda result: (result["objective"], tuple(result["parameters"]), tuple(result["start"])),
    )
    return (best, results, bool(certified), _balance_posterior(a, h, *best["parameters"]))


def _balance_mixture(A, H, separation):
    """r4's fit and gate on its domain (expected heterozygote ALT reads A and depth H, H >= 1). The clean class is the
    majority component, with mean h0 (the balance the sample QC uses). The artefact class separates when its latent SD,
    or its mean offset, is at least ``separation`` times max(clean latent SD, binomial SD at the median H). This
    contig-level working gate can miss well-resolved high-H sites. Returns the record and P(clean) per site;
    r4 abstains when no start is certified or the majority mean is farther from 1/2 than the minority mean by
    at least ``separation`` times that scale. Labels and the h0 supplied to sample QC are unchanged."""
    best, starts, certified, clean = _balance_fit(A, H)
    pi, m0, t0, m1, t1 = best["parameters"]
    clean_sd = math.sqrt(m0 * (1 - m0) * t0 / (1 + t0))
    artefact_sd = math.sqrt(m1 * (1 - m1) * t1 / (1 + t1))
    binomial_sd = math.sqrt(m0 * (1 - m0) / float(np.median(H)))
    scale = max(clean_sd, binomial_sd)
    record = dict(
        model="two-beta-binomial",
        pi_clean=pi,
        h0=m0,
        clean_inverse_concentration=t0,
        clean_sd=clean_sd,
        artefact_mean=m1,
        artefact_inverse_concentration=t1,
        artefact_sd=artefact_sd,
        binomial_sd_at_median_reads=binomial_sd,
        separation_scale=scale,
        objective=best["objective"],
        projected_gradient=best["projected_gradient"],
        starts=starts,
    )
    for component, inverse in (("clean", t0), ("artefact", t1)):
        record[component + "_binomial"] = inverse == 0.0
        record[component + "_concentration"] = None if inverse == 0.0 else 1.0 / inverse
    if abs(m0 - 0.5) - abs(m1 - 0.5) >= separation * scale:
        record.update(
            applied=False,
            reason=(
                "unresolved class labelling: the majority mean is farther from 1/2 than the minority mean "
                "by at least the minimum artefact separation times the scale"
            ),
        )
    elif certified:
        record["applied"] = bool(artefact_sd >= separation * scale or abs(m1 - m0) >= separation * scale)
    else:
        record.update(applied=False, reason="no start of the balance fit reached its stationarity certificate")
    return (record, clean)


def _tile_sites(n_samples):
    """Bound dense working arrays by cells as well as sites (large cohorts use shorter tiles)."""
    return max(1, min(_MAXIMUM_TILE_SITES, _SITE_TILE_CELLS // max(1, n_samples)))


def _count_tile(counts, start, stop, selected=None):
    counts = counts[:, start:stop, :]
    if selected is not None:
        counts = counts[selected]
    alt = np.ascontiguousarray(counts[:, :, 1].T, dtype=np.int32)
    dep = np.ascontiguousarray((counts[:, :, 0] + counts[:, :, 1]).T)
    return (alt, dep)


def _quality_flags(n_sites, n_samples, read_tile, config, *, team, map_work=map, counts=None, reverse=False):
    """Evaluate site-filter rules with O(sites) summaries and bounded dense tiles."""
    L, S = (n_sites, n_samples)
    tile = _tile_sites(S)
    D = np.empty(L)
    F, W, N, A, H, covered, LK, SL = _genotype_expectations_counts(
        counts,
        np.arange(S),
        config.classification_error,
        config.classification_heterozygote_alt,
        _SITE_EM_ITERATIONS,
        team=team,
        map_work=map_work,
    )
    for start in range(0, L, tile):
        stop = min(start + tile, L)
        alt, dep = read_tile(start, stop)
        D[start:stop] = dep.sum(axis=1, dtype=np.float64)
    del alt, dep
    parameters = dict(records=int(L), samples=int(S), snp=int(L))
    counter = ties.TieCounter()
    core = _peak_core(D, config.histogram_smoothing_bins, integer=True, reverse=reverse)
    counter.note("ev.depth.mode", inside_tie=core["tied_mode"])
    lo, hi = _bounds(core, config.core_posterior_level)
    r1 = (D < lo) | (D > hi)
    parameters["depth"] = dict(_core_record(core), low=lo, high=hi, poisson_sd=float(np.sqrt(core["mode"])))
    reference = ~r1
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
        X[start:stop] = ((depth - expected) ** 2 / np.maximum(expected, DEPTH_MEAN_FLOOR)).sum(1) / (S - 1)
    del alt, dep, depth, expected
    dispersed = reference & (X > 0)
    if dispersed.any():
        core = _peak_core(np.log(X[dispersed]), config.histogram_smoothing_bins, reverse=reverse)
        counter.note("ev.heterogeneity.mode", inside_tie=core["tied_mode"])
        _, high = _bounds(core, config.core_posterior_level)
        r2 = X > np.exp(high)
        parameters["heterogeneity"] = dict(
            mode=float(np.exp(core["mode"])),
            sd_log_left=core["sd_left"],
            sd_log_right=core["sd_right"],
            high=float(np.exp(high)),
            core_weight=core["core_weight"],
        )
    else:
        r2 = np.zeros(L, bool)
        parameters["heterogeneity"] = dict(applied=False, reason="no reference site with depth heterogeneity")
    base = ~r1 & ~r2
    separation = config.minimum_artefact_separation
    m3 = base & (N >= 1) & (SL > _LEAKAGE_SLOPE_FLOOR * N)
    r3 = np.zeros(L, bool)
    r4 = np.zeros(L, bool)
    if m3.any():
        error, clean = _error_mixture(W[m3], LK[m3], SL[m3])
        error["applied"] = bool(error["eps1"] >= separation * error["eps0"])
        if error["applied"]:
            r3[np.where(m3)[0][clean < config.clean_posterior_level]] = True
    else:
        error = dict(
            applied=False,
            reason="no site with expected homozygote reads >= 1 and a leakage slope above 1e-9 of them",
        )
    m4 = base & ~r3 & (H >= 1)
    if m4.any():
        balance, clean = _balance_mixture(A[m4], H[m4], separation)
        if balance["applied"]:
            r4[np.where(m4)[0][clean < config.clean_posterior_level]] = True
    else:
        balance = dict(
            applied=False,
            reason="no site with expected heterozygote reads >= 1",
            h0=float(config.classification_heterozygote_alt),
            h0_source="classification balance",
        )
    parameters["error"] = error
    parameters["balance"] = balance
    parameters["tie_events"] = {
        row.kind: dict(decisions=row.decisions, inside_tie=row.inside_tie, wide=row.wide, boundary=row.boundary)
        for row in counter.record().entries
    }
    r5 = covered * (1 - F.max(1)) < config.minimum_expected_carriers
    return (dict(r1=r1, r2=r2, r3=r3, r4=r4, r5=r5), parameters)


def _fitted_quality(values) -> SiteQualityFit:
    """Bind the transient fitted dictionaries to typed boundary records."""
    error = dict(values["error"])
    if "m_steps" in error:
        error["m_steps"] = ErrorMSteps(**error["m_steps"])
    balance = dict(values["balance"])
    balance["starts"] = tuple(
        BalanceStart(**{**row, "start": tuple(row["start"]), "parameters": tuple(row["parameters"])})
        for row in balance.get("starts", ())
    )
    return SiteQualityFit(
        records=values["records"],
        samples=values["samples"],
        snp=values["snp"],
        depth=DepthCore(**values["depth"]),
        heterogeneity=HeterogeneityCore(**values["heterogeneity"]),
        error=ErrorMixture(**error),
        balance=BalanceMixture(**balance),
        tie_events=ties.TieCounts(
            tuple(ties.TieCount(kind, **counts) for kind, counts in values["tie_events"].items())
        ),
    )


def filter_sites(
    positions,
    counts,
    config: SiteQualityParameters,
    sample_config: SampleQualityParameters,
    *,
    team: ThreadSource,
    map_work=map,
    reverse: bool = False,
) -> tuple[SiteQualityDecision, SampleQualityDecision | None]:
    """Decide the site filter (and, with it, the contamination QC) on one contig's SNPs.

    positions int64[L], counts int32[N, L, 2] of the analysed samples. With an enabled
    ``sample_config``, the per-sample QC is decided on this contig's r1-r4-passing SNPs with r4's
    clean heterozygote balance (its interval when r4 records an unresolved fit) and r3's clean error rate, and r5
    counts carriers among unflagged samples only.
    """
    n_sites = len(positions)
    if not config.enabled:
        return SiteQualityDecision(
            enabled=False, applied=False, reason="site filter off", kept=np.ones(n_sites, bool)
        ), None
    if n_sites < config.minimum_calibration_sites:
        return SiteQualityDecision(
            enabled=True,
            applied=False,
            kept=np.ones(n_sites, bool),
            reason=f"fewer than {config.minimum_calibration_sites} biallelic SNPs",
        ), None
    flags, parameters = _quality_flags(
        n_sites,
        counts.shape[0],
        lambda start, stop: _count_tile(counts, start, stop),
        config,
        team=team,
        map_work=map_work,
        counts=counts,
        reverse=reverse,
    )
    samples = None
    if sample_config.enabled:
        passing = ~np.any([flags[rule] for rule in QUALITY_RULES], axis=0)
        interval, why = sample_quality.balance_support(parameters["balance"])
        samples = sample_quality.assess_counts(
            counts,
            positions,
            passing,
            parameters["balance"]["h0"],
            parameters["error"].get("eps0", config.classification_error),
            sample_config,
            balance_interval=interval,
            balance_reason=why,
            chunk_sites=_tile_sites(counts.shape[0]),
            reverse=reverse,
        )
        counted = ~samples.flagged
        if not np.all(counted):
            F, _, _, _, _, covered, _, _ = _genotype_expectations_counts(
                counts,
                np.flatnonzero(counted),
                config.classification_error,
                config.classification_heterozygote_alt,
                _SITE_EM_ITERATIONS,
                team=team,
                map_work=map_work,
            )
            flags["r5"] = covered * (1 - F.max(1)) < config.minimum_expected_carriers
    applied = RULES if config.drop_uninformative_sites else QUALITY_RULES
    remove = np.zeros(n_sites, bool)
    for rule in applied:
        remove |= flags[rule]
    only = {
        rule: int((flags[rule] & ~np.any([flags[other] for other in applied if other != rule], axis=0)).sum())
        for rule in applied
    }
    bits = np.zeros(n_sites, np.uint8)
    for rule in RULES:
        bits |= flags[rule].astype(np.uint8) << int(rule[1:])
    return SiteQualityDecision(
        enabled=True,
        applied=True,
        reason=None,
        kept=~remove,
        rule_bits=bits,
        applied_rules=tuple(applied),
        flagged={rule: int(flags[rule].sum()) for rule in RULES},
        removed_only_by=only,
        parameters=_fitted_quality(parameters),
        r5_excluded_samples=0 if samples is None else int(np.sum(samples.flagged)),
    ), samples


__all__ = ["SiteQualityDecision", "filter_sites"]
