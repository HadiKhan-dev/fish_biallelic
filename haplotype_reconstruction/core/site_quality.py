"""Self-calibrated site-quality filter on loaded allele depths.

Each contig is calibrated on its own distributions, from allele depths and
record type only: no sample labels, benchmark, pedigree or simulation truth.
A record is removed when any applied rule fires:

  r0  not a biallelic SNP;
  r1  total depth outside the clean core of total depth (both tails);
  r2  per-sample depth heterogeneity, the dispersion index of sample depths
      about the site's own depth factor, above its clean core;
  r3  wrong-allele reads in samples classified homozygous: a negative-binomial
      artefact class is more likely than the Poisson clean core;
  r4  heterozygote allele balance: a beta-binomial artefact class is more
      likely than the binomial clean core;
  r5  (informativeness, switchable) fewer than two expected carriers of a
      non-major genotype (among samples the per-sample contamination QC does
      not flag).

A clean core is the histogram mode with Gaussian-equivalent left/right widths
from its half-maximum crossings; its window is where the core's local
posterior (Efron's empirical-null local fdr) is at least one half. A mixture
whose artefact class does not separate from its clean core flags nothing.

Port of the review's reference (claude_lab/review_20260929/scratch/
r5-site-filter-design: selfqc.py, robustcore.py, sitestats.py) with the
classification read model fixed at (0.01, 0.5); the statistics are unchanged.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import math
import os

import numpy as np
from numba import njit, prange
from scipy import special
from scipy.optimize import minimize

from .environment import boolean_setting
from .run_record import record_product, timed_stage
from . import sample_quality, variants


VERSION = "self-calibrated-site-quality-v1"
RULES = ("r0", "r1", "r2", "r3", "r4", "r5")

# Numerical settings of the reference fits, not scientific thresholds.
_SITE_EM_ITERATIONS = 60
_MIXTURE_ITERATIONS = 300
_HISTOGRAM_RANGE_PERCENTILES = (0.5, 99.5)
_NELDER_MEAD_OPTIONS = dict(xatol=1e-4, fatol=1e-3, maxiter=200)
_MINIMUM_CHUNK_SITES = 8192   # keeps thread scheduling negligible beside each chunk's special functions


def configured(default):
    """Route default, overridden by HAPLOTYPES_SITE_QUALITY_FILTER (CLI/TOML)."""
    value = os.environ.get("HAPLOTYPES_SITE_QUALITY_FILTER")
    if value is None:
        return default
    return replace(default, enabled=boolean_setting(value, "site_quality_filter"))


def identity(config):
    """Discovery-identity component; None keeps a filter-off identity unchanged."""
    return dict(version=VERSION, config=asdict(config)) if config.enabled else None


# ---------------------------------------------------------------------------
# Per-site genotype classification under a fixed read model
# ---------------------------------------------------------------------------

@njit(cache=True)
def _site_genotype_fit(a, n, eps, h, iters, f, ll):
    """EM over one site's free genotype frequencies (no Hardy-Weinberg).

    alt reads ~ Binomial(depth, p_g), p = (eps, h, 1 - eps). Returns the
    expected wrong-allele reads and depth in homozygotes and the expected ALT
    reads and depth in heterozygotes from the last E-step; f holds the
    updated frequencies.
    """
    S = len(n)
    le, l1e = math.log(eps), math.log(1.0 - eps)
    lh, l1h = math.log(h), math.log(1.0 - h)
    for s in range(S):
        ll[s, 0] = a[s] * le + (n[s] - a[s]) * l1e
        ll[s, 1] = a[s] * lh + (n[s] - a[s]) * l1h
        ll[s, 2] = a[s] * l1e + (n[s] - a[s]) * le
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
            m = max(ll[s, 0], ll[s, 1], ll[s, 2])
            p0 = f[0] * math.exp(ll[s, 0] - m)
            p1 = f[1] * math.exp(ll[s, 1] - m)
            p2 = f[2] * math.exp(ll[s, 2] - m)
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


@njit(parallel=True, cache=True)
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


# ---------------------------------------------------------------------------
# Empirical-null clean cores
# ---------------------------------------------------------------------------

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
    smoothed = np.convolve(counts, np.ones(smooth) / smooth, mode="same")
    p = int(np.argmax(smoothed)); top = smoothed[p]

    def cross(direction):
        i = p
        while 0 < i < len(smoothed) - 1 and smoothed[i] > top / 2:
            i += direction
        j = i - direction
        if smoothed[j] == smoothed[i]:
            return centres[i]
        return centres[j] + (top / 2 - smoothed[j]) * (centres[i] - centres[j]) / (smoothed[i] - smoothed[j])
    left, right = cross(-1), cross(+1)
    f = np.sqrt(2 * np.log(2))
    density = top / (len(x) * (edges[1] - edges[0]))
    sd_left, sd_right = (centres[p] - left) / f, (right - centres[p]) / f
    return dict(mode=float(centres[p]), sd_left=float(sd_left), sd_right=float(sd_right),
                peak_density=float(density),
                core_weight=float(density * np.sqrt(2 * np.pi) * 0.5 * (sd_left + sd_right)),
                bins=nb, centres=centres, smooth=smoothed, n=len(x), width=float(edges[1] - edges[0]))


def _bounds(core, level):
    """Window around the mode inside which the core's local posterior is >= level."""
    c = core["centres"]
    density = core["smooth"] / (core["n"] * core["width"])
    sd = np.where(c < core["mode"], core["sd_left"], core["sd_right"])
    null = core["peak_density"] * np.exp(-0.5 * ((c - core["mode"]) / sd) ** 2)  # pi0*f0 anchored at the peak
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


# ---------------------------------------------------------------------------
# Clean-core / artefact mixtures
# ---------------------------------------------------------------------------
# The parameter-free leading terms (log W! and log C(H, A)) are evaluated once
# per fit; the left-to-right evaluation order, hence every value, is unchanged.

def _log_poisson(W, mu, log_w_factorial):
    return W * np.log(np.maximum(mu, 1e-300)) - mu - log_w_factorial


def _log_negative_binomial(W, mu, k, log_w_factorial):
    return (special.gammaln(W + k) - special.gammaln(k) - log_w_factorial
            + k * np.log(k / (k + mu)) + W * np.log(mu / (k + mu)))


def _log_binomial(A, H, h, log_choose):
    return log_choose + A * np.log(h) + (H - A) * np.log(1 - h)


def _log_beta_binomial(A, H, m, c, log_choose):
    a = m * c; b = (1 - m) * c
    return log_choose + special.betaln(A + a, H - A + b) - special.betaln(a, b)


def _posterior_clean(log_clean, log_artefact):
    return 1 / (1 + np.exp(np.clip(log_artefact - log_clean, -700, 700)))


class _Chunks:
    """Evaluate one elementwise expression over contiguous chunks in threads.

    NumPy and scipy.special ufuncs release the GIL. Each element is computed by
    the same expression, so values equal a whole-array evaluation; reductions
    stay single-threaded over the whole array, preserving summation order.
    """

    def __init__(self, pool, size, threads):
        count = max(1, min(4 * threads, size // _MINIMUM_CHUNK_SITES))
        bounds = np.linspace(0, size, count + 1).astype(np.int64)
        self.pool, self.size = pool, size
        self.chunks = [(start, stop) for start, stop in zip(bounds[:-1], bounds[1:]) if stop > start]

    def __call__(self, function, *arrays):
        out = np.empty(self.size)

        def work(bounds):
            start, stop = bounds
            out[start:stop] = function(*(array[start:stop] for array in arrays))
        for _ in self.pool.map(work, self.chunks):
            pass
        return out


def _error_mixture(W, N, evaluate):
    """Clean: W ~ Poisson(N eps0). Artefact: W ~ NegBin(mean N eps1, shape k). Returns parameters and P(clean)."""
    pi = 0.6; e0 = max(np.median(W / N) / 2, 1e-5); e1 = 10 * e0; k = 1.0
    lw = special.gammaln(W + 1)

    def artefact(rate, shape):
        return evaluate(lambda W, N, lw: _log_negative_binomial(W, N * rate, shape, lw), W, N, lw)
    for it in range(_MIXTURE_ITERATIONS):
        wc = _posterior_clean(np.log(pi) + _log_poisson(W, N * e0, lw),
                              np.log1p(-pi) + artefact(e1, k))
        pi2 = float(np.clip(wc.mean(), 1e-6, 1 - 1e-6)); e02 = float((wc * W).sum() / (wc * N).sum())
        wa = 1 - wc
        if wa.sum() > 1e-6:
            o = minimize(lambda p: -(wa * artefact(np.exp(p[0]), np.exp(p[1]))).sum(),
                         [np.log(e1), np.log(k)], method="Nelder-Mead", options=_NELDER_MEAD_OPTIONS)
            e12, k2 = float(np.exp(o.x[0])), float(np.clip(np.exp(o.x[1]), 1e-3, 1e4))
        else:
            e12, k2 = e1, k
        change = abs(e02 - e0) / e0 + abs(pi2 - pi) + abs(e12 - e1) / e1
        pi, e0, e1, k = pi2, e02, e12, k2
        if change < 1e-5:
            break
    wc = _posterior_clean(np.log(pi) + _log_poisson(W, N * e0, lw),
                          np.log1p(-pi) + artefact(e1, k))
    return dict(pi_clean=pi, eps0=e0, eps1=e1, shape=k, iterations=it + 1), wc


def _balance_mixture(A, H, evaluate):
    """Clean: A ~ Binomial(H, h0). Artefact: beta-binomial with its own mean and concentration."""
    pi = 0.6; h0 = float(np.clip(A.sum() / H.sum(), 0.2, 0.8)); m1 = 0.75 * h0; c1 = 5.0
    lc = special.gammaln(H + 1) - special.gammaln(A + 1) - special.gammaln(H - A + 1)

    def artefact(mean, concentration):
        return evaluate(lambda A, H, lc: _log_beta_binomial(A, H, mean, concentration, lc), A, H, lc)
    for it in range(_MIXTURE_ITERATIONS):
        wc = _posterior_clean(np.log(pi) + _log_binomial(A, H, h0, lc),
                              np.log1p(-pi) + artefact(m1, c1))
        pi2 = float(np.clip(wc.mean(), 1e-6, 1 - 1e-6)); h02 = float((wc * A).sum() / (wc * H).sum())
        wa = 1 - wc
        if wa.sum() > 1e-6:
            o = minimize(lambda p: -(wa * artefact(1 / (1 + np.exp(-p[0])), np.exp(p[1]))).sum(),
                         [np.log(m1 / (1 - m1)), np.log(c1)], method="Nelder-Mead", options=_NELDER_MEAD_OPTIONS)
            m12 = float(1 / (1 + np.exp(-o.x[0]))); c12 = float(np.clip(np.exp(o.x[1]), 1e-2, 1e5))
        else:
            m12, c12 = m1, c1
        change = abs(h02 - h0) + abs(pi2 - pi) + abs(m12 - m1)
        pi, h0, m1, c1 = pi2, h02, m12, c12
        if change < 1e-6:
            break
    wc = _posterior_clean(np.log(pi) + _log_binomial(A, H, h0, lc),
                          np.log1p(-pi) + artefact(m1, c1))
    return dict(pi_clean=pi, h0=h0, artefact_mean=m1, artefact_concentration=c1, iterations=it + 1), wc


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def site_quality_flags(alt, dep, is_snp, config, *, threads):
    """Rule flags for one contig's records and the fitted parameters.

    alt, dep: (records, samples) ALT and total depths; is_snp: biallelic SNP
    records. Returns ({rule: bool array}, parameters).
    """
    L, S = dep.shape
    snp = np.asarray(is_snp, bool)
    depth = dep.astype(np.float64); D = depth.sum(1)
    parameters = dict(records=int(L), samples=int(S), snp=int(snp.sum()))
    # r1 total depth
    core = _peak_core(D[snp], config.histogram_smoothing_bins, integer=True)
    lo, hi = _bounds(core, config.core_posterior_level)
    r1 = (D < lo) | (D > hi)
    parameters["depth"] = dict(_core_record(core), low=lo, high=hi, poisson_sd=float(np.sqrt(core["mode"])))
    # r2 depth heterogeneity across samples given the site's own depth factor
    reference = snp & ~r1
    mean_depth = depth[reference].mean(0)
    expected = np.outer(D / mean_depth.sum(), mean_depth)
    X = ((depth - expected) ** 2 / np.maximum(expected, 1e-9)).sum(1) / (S - 1)
    del expected
    core = _peak_core(np.log(X[reference & (X > 0)]), config.histogram_smoothing_bins)
    _, high = _bounds(core, config.core_posterior_level)
    r2 = X > np.exp(high)
    parameters["heterogeneity"] = dict(mode=float(np.exp(core["mode"])), sd_log_left=core["sd_left"],
                                       sd_log_right=core["sd_right"], high=float(np.exp(high)),
                                       core_weight=core["core_weight"])
    # Genotype classification under the fixed read model; the clean rates are fitted below.
    idx = np.where(snp)[0]
    from .parallel import numba_thread_scope
    with numba_thread_scope(threads):
        frequencies, wrong, homn, heta, hetn, covered = _genotype_expectations(
            np.ascontiguousarray(alt[idx]).astype(np.int32), np.ascontiguousarray(dep[idx]).astype(np.int32),
            config.classification_error, config.classification_heterozygote_alt, _SITE_EM_ITERATIONS)
    W = np.zeros(L); N = np.zeros(L); A = np.zeros(L); H = np.zeros(L); F = np.zeros((L, 3)); ncov = np.zeros(L)
    W[idx] = wrong; N[idx] = homn; A[idx] = heta; H[idx] = hetn; F[idx] = frequencies; ncov[idx] = covered
    base = snp & ~r1 & ~r2
    separation = config.minimum_artefact_separation
    # r3 wrong-allele reads in homozygotes
    m3 = base & (N >= 1)
    with ThreadPoolExecutor(threads) as pool:
        error, clean = _error_mixture(W[m3], N[m3], _Chunks(pool, int(m3.sum()), threads))
        error["applied"] = bool(error["eps1"] >= separation * error["eps0"])
        r3 = np.zeros(L, bool)
        if error["applied"]:
            r3[np.where(m3)[0][clean < config.clean_posterior_level]] = True
        # r4 heterozygote allele balance
        m4 = base & ~r3 & (H >= 1)
        balance, clean = _balance_mixture(A[m4], H[m4], _Chunks(pool, int(m4.sum()), threads))
    typical = float(np.median(H[m4]))
    balance["binomial_sd_at_median_reads"] = float(np.sqrt(balance["h0"] * (1 - balance["h0"]) / typical))
    balance["artefact_sd"] = float(np.sqrt(balance["artefact_mean"] * (1 - balance["artefact_mean"])
                                           / (balance["artefact_concentration"] + 1)))
    balance["applied"] = bool(
        balance["artefact_sd"] >= separation * balance["binomial_sd_at_median_reads"]
        or abs(balance["artefact_mean"] - balance["h0"]) >= separation * balance["binomial_sd_at_median_reads"])
    r4 = np.zeros(L, bool)
    if balance["applied"]:
        r4[np.where(m4)[0][clean < config.clean_posterior_level]] = True
    parameters["error"] = error
    parameters["balance"] = balance
    # r5 expected carriers of a non-major genotype among covered samples
    r5 = snp & (ncov * (1 - F.max(1)) < config.minimum_expected_carriers)
    return dict(r0=~snp, r1=r1, r2=r2, r3=r3, r4=r4, r5=r5), parameters


def uninformative_sites(alt, dep, is_snp, config, *, threads):
    """r5 alone for the samples in alt/dep: the same per-site genotype fit and carrier rule."""
    L = dep.shape[0]
    snp = np.asarray(is_snp, bool)
    idx = np.where(snp)[0]
    from .parallel import numba_thread_scope
    with numba_thread_scope(threads):
        frequencies, _, _, _, _, covered = _genotype_expectations(
            np.ascontiguousarray(alt[idx], dtype=np.int32), np.ascontiguousarray(dep[idx], dtype=np.int32),
            config.classification_error, config.classification_heterozygote_alt, _SITE_EM_ITERATIONS)
    F = np.zeros((L, 3)); ncov = np.zeros(L)
    F[idx] = frequencies; ncov[idx] = covered
    return snp & (ncov * (1 - F.max(1)) < config.minimum_expected_carriers)


def _unique_block_sites(genomic_data):
    """Concatenate blocks with concatenate_unique_block_reads' first-occurrence rule, keeping site flags."""
    blocks = [index for index, positions in enumerate(genomic_data.positions) if len(positions) > 0]
    if not blocks:
        return None, None, None, None
    positions = np.concatenate([np.asarray(genomic_data.positions[index]) for index in blocks])
    _, first = np.unique(positions, return_index=True)
    first.sort()
    reads = np.concatenate([np.asarray(genomic_data.reads[index]) for index in blocks], axis=1)[:, first, :]
    keep = np.concatenate([np.asarray(genomic_data.keep_flags[index]) for index in blocks])[first]
    # Blocks without record types (simulated reads) are biallelic SNPs by construction.
    snp = (np.ones(len(first), bool) if genomic_data.biallelic_snp is None else
           np.concatenate([np.asarray(genomic_data.biallelic_snp[index], bool) for index in blocks])[first])
    return positions[first], reads, keep, snp


@timed_stage("site_quality", contig_arg="contig")
def filter_loaded_sites(genomic_data, *, contig, config, threads, snps_per_block, sample_rows=None,
                        sample_config=None):
    """Remove flagged records before read calibration and discovery.

    The kept sites are re-formed into non-overlapping blocks of
    snps_per_block consecutive sites, as the SNP-count loader forms them from
    an input restricted to those sites. sample_rows selects the analysed
    cohort for the statistics (default all rows). With an enabled
    sample_config, the per-sample contamination QC (core/sample_quality.py)
    is decided on this contig's r0-r4-passing SNPs with r4's clean
    heterozygote balance, and r5 counts carriers among unflagged samples
    only; the decision is the record's "sample_quality" entry.
    Returns (genomic_data, record); the record names every rule's fitted
    parameters and counts and carries each loaded record's rule bits
    (bit k = rule rk).
    """
    if not config.enabled:
        if sample_config is not None and sample_config.enabled:
            print(f"    [Sample quality] {contig}: not run (it uses the site filter's quality-passing sites, "
                  "and the site filter is off)", flush=True)
        return genomic_data, dict(version=VERSION, enabled=False)
    positions, reads, keep_flags, snp = _unique_block_sites(genomic_data)
    record = dict(version=VERSION, enabled=True, config=asdict(config), contig=str(contig))
    if positions is None or int(snp.sum()) < config.minimum_calibration_sites:
        record.update(applied=False, records=0 if positions is None else len(positions),
                      reason=f"fewer than {config.minimum_calibration_sites} biallelic SNPs")
        print(f"    [Site quality] {contig}: not applied ({record['reason']})", flush=True)
        record_product("site_quality", contig, record)
        return genomic_data, record
    rows = reads if sample_rows is None else reads[np.asarray(sample_rows)]
    # (records, samples) C-ordered, as the reference computed its row sums.
    alt = np.ascontiguousarray(rows[:, :, 1].T, dtype=np.int32)
    dep = np.ascontiguousarray(rows.sum(axis=2).T, dtype=np.int32)
    del rows
    flags, parameters = site_quality_flags(alt, dep, snp, config, threads=threads)
    samples = None
    if sample_config is not None and sample_config.enabled:
        passing = snp & ~np.any([flags[rule] for rule in RULES[:5]], axis=0)
        samples = sample_quality.assess(alt[passing], dep[passing], positions[passing],
                                        parameters["balance"]["h0"], sample_config)
        counted = ~np.asarray(samples["flagged"], bool)
        if not np.all(counted):
            flags["r5"] = uninformative_sites(alt[:, counted], dep[:, counted], snp, config, threads=threads)
    del alt, dep
    applied = RULES if config.drop_uninformative_sites else RULES[:5]
    remove = np.zeros(len(positions), bool)
    for rule in applied:
        remove |= flags[rule]
    only = {rule: int((flags[rule] & ~np.any([flags[other] for other in applied if other != rule], axis=0)).sum())
            for rule in applied}
    record.update(applied=True, applied_rules=list(applied), records=int(len(positions)),
                  kept=int((~remove).sum()), removed=int(remove.sum()),
                  flagged={rule: int(flags[rule].sum()) for rule in RULES}, removed_only_by=only,
                  parameters=parameters)
    excluded = 0 if samples is None else int(np.sum(samples["flagged"]))
    if excluded:
        record["r5_excluded_samples"] = excluded
    print(f"    [Site quality] {contig}: kept {record['kept']:,} of {record['records']:,} records; "
          + ", ".join(f"{rule} {count:,}" for rule, count in record["flagged"].items())
          + f"; depth window [{parameters['depth']['low']:.0f}, {parameters['depth']['high']:.0f}], "
          f"clean error {parameters['error']['eps0']:.2e} (r3 applied {parameters['error']['applied']}), "
          f"clean balance {parameters['balance']['h0']:.4f} (r4 applied {parameters['balance']['applied']})"
          + (f"; r5 carriers exclude {excluded} flagged samples" if excluded else ""),
          flush=True)
    record_product("site_quality", contig, record)
    bits = np.zeros(len(positions), np.uint8)
    for bit, rule in enumerate(RULES):
        bits |= flags[rule].astype(np.uint8) << bit
    record.update(loaded_positions=positions, rule_bits=bits)
    if samples is not None:
        record["sample_quality"] = samples
    kept = ~remove
    positions, reads, keep_flags = positions[kept], reads[:, kept], keep_flags[kept]
    starts = range(0, len(positions), snps_per_block)
    return variants.GenomicData(
        [positions[start:start + snps_per_block].copy() for start in starts],
        [keep_flags[start:start + snps_per_block].copy() for start in starts],
        [np.ascontiguousarray(reads[:, start:start + snps_per_block]) for start in starts],
        biallelic_snp=None if genomic_data.biallelic_snp is None else [
            np.ones(len(positions[start:start + snps_per_block]), bool) for start in starts]), record
