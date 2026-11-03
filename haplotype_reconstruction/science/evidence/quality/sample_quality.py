"""Per-library, per-contig quality exclusion from founder construction: a private-allele mixture test for a foreign
minor-read component, with an explicit model of local artefacts.

Foreign DNA at read fraction alpha < 1/2 in a library shows, at a site where the foreign genome carries an allele the
library's own genome lacks, a minor-read fraction a = alpha (foreign homozygote) or alpha/2 (foreign heterozygote), at
the same a wherever on the genome such sites fall. A library's own private heterozygous allele shows its heterozygote
balance instead, its errors a small rate of their own, and a local artefact (a collapsed repeat, a mismapped paralog,
a private haplotype tract with mapping bias) whatever fraction it has, in a few places.

**Eligible cells.** At the biallelic SNPs that pass the site filter's quality rules r1-r4 (whole analysed cohort),
the cohort-minor allele is ALT when the cohort's ALT reads do not exceed its REF reads. A library carries it at a site
when its k minor reads of depth d are improbable as errors at the scale of the cohort, P(K >= k | d, e) <= q / (N - 1)
with K ~ Binomial(d, e), e the contig's clean wrong-allele rate (r3's eps0) and q ``carrier_tail_level``. A cell
(library, site) is eligible when the library is observed there and no other library carries the minor allele; every
eligible cell enters, k = 0 included.

**Composite weights.** Cells of a library closer than one sequenced fragment share reads. Each eligible cell weighs
1 / (the library's eligible cells in its ``fragment_bp`` window); windows are counted from the contig's first passing
site, so translating every coordinate changes nothing.

**Cell model.** a_min = ``minimum_foreign_fraction`` / 2 is a working foreign-allele amplitude. An eligible cell in
depth class j (the library's eligible-cell depth quantiles, ``heterozygote_depth_classes`` of them) has minor reads k
with probability

    x0(k) = sum_u pi_ju Bin(k; d, eps_u) + sum_v w_jv BB(k; d, h_v^o, rho_v)          (no foreign component)
    x1(k) = the same + c Bin(k; d, a),   a in (a_min, a_hi]                               (foreign component)

* eps_u: the library's own error distribution, a free mixture over a log grid from e / 10 to a_min, per depth class;
* own private heterozygotes: beta-binomial components with ALT mean h_v (r4's h0, or the ends and middle of r4's
  balance interval when r4's labelling is unresolved) and latent dispersion rho_v on a fixed grid (latent SD up to
  0.1 at h = 1/2); a private REF allele uses 1 - h_v with the same rho_v, so both tails are modelled alike in both
  orientations (a complemented library gives the same likelihood);
* c: the foreign-component share within the non-artefact mixture, common to depth classes and regions;
  its marginal share in region R is (1 - pi_R) c;
* a: the foreign allele's read fraction on G log-spaced values over (a_min, a_hi], a_hi = f h_lo / 2
  (f ``heterozygote_floor_ratio``); grosser contamination is fitted at that ceiling.

**Local artefacts.** The contig's passing sites are cut into ``artefact_regions`` contiguous regions of equal site
counts. In region R a cell is, with probability pi_R, a local artefact (a collapsed repeat, a mismapped paralog, a
private haplotype tract with mapping or reference bias) whose minor fraction is its own, uniform on [0, 1], so that its
k is uniform on 0..d: A(k) = 1 / (d + 1). Each region's artefact share pi_R is free, under both hypotheses:

    L = prod_R max_{pi_R} prod_{b in R} [(1 - pi_R) x(b) + pi_R A(b)]^n_b,   x = x0 (H0) or x1 (H1).

The foreign fraction a and non-artefact share c are common; the marginal foreign weight (1 - pi_R) c varies by
region. A flag pools evidence across eligible cells; it does not require each region to support contamination.
Empty or ineligible regions provide no penalty. Local excesses can be absorbed by pi_R, but this is not guaranteed:
a short private tract can dominate when other regions supply little eligible information, and dispersed artefacts
with different fractions can fit the foreign component better than the uniform artefact law. The uniform law and
region count are working modelling choices, not biological guarantees.

**Statistic and working threshold.** T = max over the grid of 2 (log L(H1, a) - log L(H0)), each maximised over the
mixture weights, c and the regions' pi_R by expectation-maximisation (the M-step is the concave mixture-weight problem
with each cell weighted by its posterior of being an ordinary, non-artefact cell). A library is flagged when T >= u_N,
P(chi2_1 >= u_N) = 2 q / (G N): the boundary law (chi2_0 + chi2_1) / 2 of one foreign fraction, Bonferroni over the G
fractions and the N libraries. That level is nominal only for a regular, correctly specified likelihood of independent
windows; the composite weights, mixing weights on their boundary, the artefact components and misspecification break
it, so u_N is a working threshold whose clean false-flag rate is measured on simulated libraries, not a certified
level.

**What a flag means.** A pooled likelihood preference for the foreign-component model above a_min over the fitted
error, own-heterozygote and artefact mixture. It excludes the library from founder construction on the contig;
it is not a contamination diagnosis. Elevated own error, biased private alleles and tens of dispersed,
variable-fraction artefacts can also trigger exclusion. Physical tract length alone gives no safety boundary.
Foreign DNA shared with analysed libraries can evade private-allele QC; donors concentrated in few regions can
be absorbed by artefact shares. The reported foreign_share is c from the alternative, whereas reported artefact
shares describe the null; they are not parts of one fitted decomposition. Fitted fractions are not calibrated
contamination estimates. No labels, pedigree, benchmark or cohort-relative scale enter; the decision is per contig.

Inherited-segment diagnostic (never a flag): near-private carrier cells (the library and at most one partner carry
the allele, carriers as above) whose reads favour the heterozygous fraction over h/2 at even odds, grouped per
(sample, partner) into fragment events and into regions of events less than segment_gap_bp apart; regions with at
least segment_minimum_events such events are reported for unflagged samples, with the cell count and minor-read
fractions of all the pair's near-private cells in the region.

On the contig where it is flagged, a sample's evidence leaves founder construction
(``ContigEvidence.founder_view``) and the r5 carrier count; read calibration, painting, pedigree, family phase and
recombination use every sample's evidence. Painting and later stages keep its reads, but that does not restore
founder-panel support lost by the exclusion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np
from numba import prange, types
from scipy.stats import binom, chi2

from haplotype_reconstruction.contracts.evidence import InheritedSegment, SampleQualityDecision, SampleQualitySettings
from haplotype_reconstruction.core import ties
from haplotype_reconstruction.core.kernels import kernel
from haplotype_reconstruction.core.ties import TieCount, TieCounts

from .parameters import (
    POSITIVE_WEIGHT_FLOOR,
    REGION_SHARE_CEILING_GAP,
    WEIGHT_DAMPING_CEILING,
    WEIGHT_DAMPING_FLOOR,
    WEIGHT_DAMPING_GROWTH,
    WEIGHT_DAMPING_SHRINK,
    WEIGHT_GAIN_TOLERANCE,
    SampleQualityParameters,
)
from .parameters import (
    SAMPLE_AMBIGUOUS_BALANCE as _AMBIGUOUS_BALANCE,
)
from .parameters import (
    SAMPLE_CHUNK_SITES as _CHUNK_SITES,
)
from .parameters import (
    SAMPLE_CODE as _CODE,
)
from .parameters import (
    SAMPLE_EM_ITERATIONS as _EM_ITERATIONS,
)
from .parameters import (
    SAMPLE_EM_TOLERANCE as _EM_TOLERANCE,
)
from .parameters import (
    SAMPLE_ERROR_GRID_BOTTOM as _ERROR_GRID_BOTTOM,
)
from .parameters import (
    SAMPLE_ERROR_GRID_RATIO as _ERROR_GRID_RATIO,
)
from .parameters import (
    SAMPLE_ERROR_SEPARATION as _ERROR_SEPARATION,
)
from .parameters import (
    SAMPLE_FOREIGN_GRID as _FOREIGN_GRID,
)
from .parameters import (
    SAMPLE_HET_DISPERSION_GRID as _HET_DISPERSION_GRID,
)
from .parameters import (
    SAMPLE_M_STEP_ITERATIONS as _M_STEP_ITERATIONS,
)
from .parameters import (
    SAMPLE_MINIMUM_FOREIGN_RANGE as _MINIMUM_FOREIGN_RANGE,
)
from .parameters import (
    SAMPLE_REDUCE_ENTRIES as _REDUCE_ENTRIES,
)
from .parameters import (
    SAMPLE_SHARE_BISECTIONS as _SHARE_BISECTIONS,
)
from .parameters import (
    SAMPLE_SHARE_FLOOR as _SHARE_FLOOR,
)
from .parameters import (
    SAMPLE_WEIGHT_START as _WEIGHT_START,
)

# Numerical settings of the fit, not scientific thresholds.
# Own-heterozygote latent dispersion grid (rho; latent SD sqrt(h (1 - h) rho), at most 0.1 at h = 1/2): a bounded
# working nuisance, the same in both orientations.
# Applicability: the foreign range must reach this multiple of a_min, and the cohort's clean error must stay below
# a_min / this, so that errors (fitted up to a_min) and foreign alleles (above it) can separate.
# Reasons r4 records when its h0 is ambiguous (unresolved class labels, no certified fit): own heterozygotes then
# span both of r4's component means.


def balance_support(balance: Mapping) -> tuple[tuple[float, float], str | None]:
    """The own-heterozygote balance interval from r4's record, and r4's reason when it is widened.

    (h0, h0), unless r4 recorded an unresolved class labelling or no certified balance fit: its h0 is then one of two
    candidate clean balances, and the interval spans both component means."""
    h0 = float(balance["h0"])
    reason = balance.get("reason") or ""
    if reason.startswith(_AMBIGUOUS_BALANCE) and balance.get("artefact_mean") is not None:
        other = float(balance["artefact_mean"])
        return (min(h0, other), max(h0, other)), reason
    return (h0, h0), None


# ---------------------------------------------------------------------------------------------
# Carriers, eligible cells and the inherited-segment cells
# ---------------------------------------------------------------------------------------------


def carrier_thresholds(max_depth: int, error_rate: float, level: float) -> np.ndarray:
    """Smallest k with P(K >= k | d, e) <= level for each depth d = 0..max_depth (d + 1: never a carrier).

    Vectorised bisection on the binomial tail, which decreases in k: O(D log D)."""
    depth = np.arange(max_depth + 1)
    low = np.ones(len(depth), np.int64)
    high = depth + 1
    while np.any(low < high):
        middle = (low + high) // 2
        ok = (binom.sf(middle - 1, depth, error_rate) <= level) & (middle <= depth)
        active = low < high
        high = np.where(active & ok, middle, high)
        low = np.where(active & ~ok, middle + 1, low)
    return low


def _region_bounds(passing_positions, regions, fragment_bp):
    """Fragment-window indices (counted from the first passing site) cutting the passing sites into ``regions`` runs
    of about equal site counts; a cell's region is the number of cuts at or below its window. Cuts that coincide, or
    fall in the first window, are dropped, so a fragment window never spans two regions."""
    n = len(passing_positions)
    if n == 0 or regions <= 1:
        return np.zeros(0, np.int64)
    indices = (np.arange(1, regions) * n) // regions
    windows = (passing_positions[indices] - passing_positions[0]) // fragment_bp
    return np.unique(windows[windows > 0]).astype(np.int64)


class _Collector:
    """One pass over position-ordered tiles: per-library weighted histograms of eligible cells (sample, region,
    orientation, depth, minor reads) and the near-private carrier cells of the segment report. The eligible cells
    of a tile's last fragment window wait for the next tile, so every window's cell count is complete when its
    cells are weighted."""

    def __init__(
        self, n_samples, error_rate, level, heterozygote_balance, fragment_bp, region_bounds, origin, *, reverse=False
    ):
        self.reverse = ties.reverse_conventions(reverse)
        self.n = n_samples
        self.error_rate, self.level = error_rate, level
        self.balance, self.fragment_bp = heterozygote_balance, fragment_bp
        self.bounds = region_bounds
        self.regions = len(region_bounds) + 1
        self.origin = origin
        self.table = carrier_thresholds(255, error_rate, level)
        self.observed = np.zeros(n_samples, np.int64)
        self.eligible = np.zeros(n_samples, np.int64)
        self.minor_cells = np.zeros(n_samples, np.int64)
        self.codes, self.weights, self.entries = [], [], 0
        self.open = tuple(np.zeros(0, np.int64) for _ in range(5))  # (sample, window, orientation, depth, k)
        self.columns = [[] for _ in range(8)]
        self.passing_sites = self.contributing = self.tied = 0

    def _reduce(self):
        if len(self.codes) > 1:
            code, weight = np.concatenate(self.codes), np.concatenate(self.weights)
            unique, inverse = np.unique(code, return_inverse=True)
            self.codes, self.weights = [unique], [np.bincount(inverse, weights=weight)]
        self.entries = sum(len(code) for code in self.codes)

    def _weigh(self, sample, window, orientation, d, k):
        """Histogram the eligible cells of complete windows, each weighing 1 / (its library's cells in the window)."""
        if not len(sample):
            return
        region = np.searchsorted(self.bounds, window, side="right").astype(np.int64)
        local = window - window.min()
        key = sample * (int(local.max()) + 1) + local
        _, inverse, size = np.unique(key, return_inverse=True, return_counts=True)
        code = (((sample * self.regions + region) * 2 + orientation) * _CODE + d) * _CODE + k
        unique, where = np.unique(code, return_inverse=True)
        self.codes.append(unique)
        self.weights.append(np.bincount(where, weights=1.0 / size[inverse]))
        self.entries += len(unique)
        if self.entries > _REDUCE_ENTRIES:
            self._reduce()

    def add(self, alt, dep, positions):
        """alt, dep: int64 (sites, samples) of consecutive passing sites; positions: int64[sites], increasing."""
        n = self.n
        self.passing_sites += len(positions)
        if not len(positions):
            return
        deepest = int(dep.max())
        if deepest >= len(self.table):
            self.table = carrier_thresholds(max(deepest, 2 * len(self.table)), self.error_rate, self.level)
        # At equal read totals ALT is minor by convention (REF under convention reversal).
        alt_total, ref_total = alt.sum(axis=1), (dep - alt).sum(axis=1)
        minor_is_alt = (alt_total < ref_total) if self.reverse else (alt_total <= ref_total)
        minor = np.where(minor_is_alt[:, None], alt, dep - alt)
        observed = dep > 0
        self.observed += observed.sum(axis=0)
        threshold = self.table[dep]
        carrier = observed & (minor >= threshold)
        carriers = carrier.sum(axis=1)
        eligible = observed & (carriers[:, None] - carrier == 0)
        site, sample = np.nonzero(eligible)
        d, k = dep[site, sample], minor[site, sample]
        self.eligible += np.bincount(sample, minlength=n)
        self.minor_cells += np.bincount(sample[k > 0], minlength=n)
        # Composite weights need every cell of a window: the tile's last window stays open until the next tile.
        window = (positions - self.origin) // self.fragment_bp
        cells = (
            sample.astype(np.int64),
            window[site],
            minor_is_alt[site].astype(np.int64),
            d.astype(np.int64),
            k.astype(np.int64),
        )
        cells = tuple(np.concatenate((held, new)) for held, new in zip(self.open, cells, strict=False))
        complete = cells[1] < window[-1]
        self._weigh(*(column[complete] for column in cells))
        self.open = tuple(column[~complete] for column in cells)

        # Near-private carriers (at most two per site) for the inherited-segment report.
        near = np.flatnonzero((carriers >= 1) & (carriers <= 2))
        # At equal totals either allele could have been chosen: count the tie whenever one of them gives cells.
        has_cells = np.zeros(len(positions), bool)
        has_cells[near] = True
        has_cells[site[k > 0]] = True
        equal = (alt_total == ref_total) & (alt_total > 0)
        alt_carriers = (observed & (alt >= threshold)).sum(axis=1)
        ref_carriers = (observed & (dep - alt >= threshold)).sum(axis=1)
        self.contributing += int((has_cells | (equal & ((alt_carriers <= 2) | (ref_carriers <= 2)))).sum())
        self.tied += int(equal.sum())
        site_n, sample_n = np.nonzero(carrier[near])
        site_n = near[site_n]
        partner = np.full(len(site_n), -1, np.int64)
        first = np.flatnonzero((carriers[site_n] == 2) & np.r_[True, site_n[1:] != site_n[:-1]])
        partner[first] = sample_n[first + 1]
        partner[first + 1] = sample_n[first]
        shared = partner >= 0
        partner_k = np.zeros(len(site_n), np.int64)
        partner_d = np.zeros(len(site_n), np.int64)
        partner_k[shared] = minor[site_n[shared], partner[shared]]
        partner_d[shared] = dep[site_n[shared], partner[shared]]
        values = (
            minor[site_n, sample_n].astype(float),
            dep[site_n, sample_n].astype(float),
            np.where(minor_is_alt[site_n], self.balance, 1.0 - self.balance),
            positions[site_n],
            sample_n,
            partner,
            partner_k,
            partner_d,
        )
        for column, value in zip(self.columns, values, strict=False):
            column.append(value)

    def finish(self):
        self._weigh(*self.open)
        self._reduce()
        code = self.codes[0] if self.codes else np.zeros(0, np.int64)
        weight = self.weights[0] if self.weights else np.zeros(0)
        k = code % _CODE
        rest = code // _CODE
        d = rest % _CODE
        rest //= _CODE
        orientation = rest % 2
        rest //= 2
        # sorted by (sample, region, orientation, depth, minor reads)
        histogram = (rest // self.regions, rest % self.regions, orientation, d, k, weight)
        dtypes = (float, float, float, np.int64, np.int64, np.int64, np.int64, np.int64)
        cells = tuple(
            np.concatenate(column) if column else np.empty(0, dtype)
            for column, dtype in zip(self.columns, dtypes, strict=False)
        )
        minor_allele = {
            "ev.qc.minor_allele": dict(decisions=self.contributing, inside_tie=self.tied, wide=0, boundary=0)
        }
        return histogram, cells, minor_allele


# ---------------------------------------------------------------------------------------------
# The per-library fit
# ---------------------------------------------------------------------------------------------


@kernel(
    overloads=(
        (
            types.float64,
            types.float64,
            types.float64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _log_binomial(k, d, p):
    """log of p^k (1 - p)^(d - k) (the binomial coefficient is common to every component)."""
    return k * math.log(p) + (d - k) * math.log1p(-p)


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
def _log_beta_binomial(k, d, h, rho):
    """Beta-binomial (mean h, latent dispersion rho) without the binomial coefficient; rho = 0 is Bin(d, h)."""
    if rho == 0.0:
        return _log_binomial(k, d, h)
    concentration = 1.0 / rho - 1.0
    a, b = h * concentration, (1.0 - h) * concentration
    return (
        math.lgamma(k + a)
        - math.lgamma(a)
        + math.lgamma(d - k + b)
        - math.lgamma(b)
        + math.lgamma(concentration)
        - math.lgamma(d + concentration)
    )


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
def _log_artefact(k, d):
    """A local artefact's minor fraction, uniform on [0, 1], without the binomial coefficient: B(k + 1, d - k + 1)
    (with it, k is uniform on 0..d)."""
    return math.lgamma(k + 1.0) + math.lgamma(d - k + 1.0) - math.lgamma(d + 2.0)


@kernel(
    overloads=(
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.int64,
            types.literal(False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.int64,
            types.literal(True),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.int64,
            types.bool,
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
)
def _objective(F, FA, n, cls, Q, contamination, with_c, y):
    value = 0.0
    for b in range(len(n)):
        if n[b] == 0.0:
            continue
        m = 0.0
        base = cls[b] * Q
        for p in range(Q):
            m += y[base + p] * F[p, b]
        if with_c:
            m += y[contamination] * FA[b]
        value += n[b] * math.log(max(m, POSITIVE_WEIGHT_FLOOR))
    return value


@kernel(
    overloads=(
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
)
def _cholesky_solve(A, b, out):
    """Solve A out = b for a symmetric positive definite A (Cholesky); False when A is not numerically definite."""
    size = len(b)
    factor = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1):
            total = A[i, j]
            for q in range(j):
                total -= factor[i, q] * factor[j, q]
            if i == j:
                if not total > 0.0:
                    return False
                factor[i, i] = math.sqrt(total)
            else:
                factor[i, j] = total / factor[j, j]
    for i in range(size):
        total = b[i]
        for q in range(i):
            total -= factor[i, q] * out[q]
        out[i] = total / factor[i, i]
    for i in range(size - 1, -1, -1):
        total = out[i]
        for q in range(i + 1, size):
            total -= factor[q, i] * out[q]
        out[i] = total / factor[i, i]
    return True


@kernel(
    overloads=(
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.literal(False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.literal(300),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.literal(False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.literal(4),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.literal(True),
            types.Array(types.float64, 1, "C", readonly=False),
            types.literal(300),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.literal(True),
            types.Array(types.float64, 1, "C", readonly=False),
            types.literal(4),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.bool,
            types.Array(types.float64, 1, "C", readonly=False),
            types.literal(4),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.bool,
            types.Array(types.float64, 1, "C", readonly=False),
            types.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _weights(F, FA, n, cls, classes, with_c, y, iterations):
    """Maximise sum_b n_b log(sum_p y[cls_b Q + p] F[p, b] + y[classes Q] FA_b) over the mixture weights: y >= 0,
    and in every class the Q component weights plus the contamination share (common to all classes) sum to 1.
    y holds a feasible start and returns the improved weights; at most ``iterations`` Newton iterations.

    Projected, damped Newton on the concave objective. Each iteration eliminates every class's sum constraint
    through its largest weight (the pivot, at least 1 / (Q + 1) of the class, so far from its bound); an active set
    holds at 0 the other weights whose gradient points outward; the free block solves (-H + lambda I) s = g by
    Cholesky, lambda (Levenberg-Marquardt, relative to the largest curvature) shrinking after an accepted step and
    growing after a rejected one, so collinear components (neighbouring rates at low depth) cannot stall the search.
    A step that would make a pivot negative is rejected. Stops when an accepted step gains less than 1e-12
    (relative) or the damping exceeds 1e12. Every accepted step increases the objective."""
    Q = F.shape[0]
    P = Q - 1
    contamination = classes * Q
    size = classes * P + (1 if with_c else 0)
    c_index = classes * P
    pivot = np.empty(classes, np.int64)
    gradient = np.zeros(size)
    hessian = np.zeros((size, size))
    direction = np.empty(P + 1)
    trial = np.empty(len(y))
    index = np.empty(size, np.int64)
    value = _objective(F, FA, n, cls, Q, contamination, with_c, y)
    damping = WEIGHT_DAMPING_FLOOR
    for _ in range(iterations):
        for j in range(classes):
            best = 0
            for p in range(1, Q):
                if y[j * Q + p] > y[j * Q + best]:
                    best = p
            pivot[j] = best
        gradient[:] = 0.0
        hessian[:, :] = 0.0
        for b in range(len(n)):
            if n[b] == 0.0:
                continue
            j = cls[b]
            base = j * Q
            m = 0.0
            for p in range(Q):
                m += y[base + p] * F[p, b]
            if with_c:
                m += y[contamination] * FA[b]
            m = max(m, POSITIVE_WEIGHT_FLOOR)
            weight = n[b] / (m * m)
            f0 = F[pivot[j], b]
            for p in range(P):
                q = p if p < pivot[j] else p + 1
                direction[p] = F[q, b] - f0
            if with_c:
                direction[P] = FA[b] - f0
            row = j * P
            for p in range(P):
                gradient[row + p] += n[b] * direction[p] / m
                for u in range(p + 1):
                    hessian[row + p, row + u] -= weight * direction[p] * direction[u]
                if with_c:
                    hessian[row + p, c_index] -= weight * direction[p] * direction[P]
            if with_c:
                gradient[c_index] += n[b] * direction[P] / m
                hessian[c_index, c_index] -= weight * direction[P] * direction[P]
        for j in range(classes):
            for p in range(P):
                for u in range(p):
                    hessian[j * P + u, j * P + p] = hessian[j * P + p, j * P + u]
                if with_c:
                    hessian[c_index, j * P + p] = hessian[j * P + p, c_index]
        free = 0
        for i in range(size):
            if i == c_index and with_c:
                current = y[contamination]
            else:
                j = i // P
                p = i % P
                current = y[j * Q + (p if p < pivot[j] else p + 1)]
            if not (current <= 0.0 and gradient[i] <= 0.0):
                index[free] = i
                free += 1
        if free == 0:
            break
        A = np.empty((free, free))
        rhs = np.empty(free)
        solved = np.empty(free)
        largest = 0.0
        for i in range(free):
            rhs[i] = gradient[index[i]]
            for k in range(free):
                A[i, k] = -hessian[index[i], index[k]]
            largest = max(largest, A[i, i])
        if not largest > 0.0:
            break
        accepted = False
        new_value = value
        while damping <= WEIGHT_DAMPING_CEILING:
            for i in range(free):
                A[i, i] = -hessian[index[i], index[i]] + damping * largest
            if _cholesky_solve(A, rhs, solved):
                for i in range(len(y)):
                    trial[i] = y[i]
                for i in range(free):
                    t = index[i]
                    if with_c and t == c_index:
                        trial[contamination] = max(y[contamination] + solved[i], 0.0)
                    else:
                        j = t // P
                        p = t % P
                        q = j * Q + (p if p < pivot[j] else p + 1)
                        trial[q] = max(y[q] + solved[i], 0.0)
                feasible = True
                share = trial[contamination] if with_c else 0.0
                for j in range(classes):
                    rest = share
                    for p in range(Q):
                        if p != pivot[j]:
                            rest += trial[j * Q + p]
                    trial[j * Q + pivot[j]] = 1.0 - rest
                    if not rest <= 1.0:
                        feasible = False
                if feasible:
                    new_value = _objective(F, FA, n, cls, Q, contamination, with_c, trial)
                    if new_value >= value:
                        accepted = True
                        break
            damping *= WEIGHT_DAMPING_GROWTH
        if not accepted:
            break
        damping = max(damping * WEIGHT_DAMPING_SHRINK, WEIGHT_DAMPING_FLOOR)
        for i in range(len(y)):
            y[i] = trial[i]
        change = new_value - value
        value = new_value
        if change <= WEIGHT_GAIN_TOLERANCE * max(1.0, abs(value)):
            break
    return value


@kernel(
    overloads=(
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.int64,
            types.int64,
            types.float64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _feasible(y, classes, Q, share):
    """Scale each class's component weights so that they and the contamination share sum to 1."""
    for j in range(classes):
        total = 0.0
        for p in range(Q):
            total += y[j * Q + p]
        for p in range(Q):
            y[j * Q + p] *= (1.0 - share) / total


@kernel(
    overloads=(
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.int64,
            types.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
def _region_share(x, A, pb, pw, start, stop):
    """The artefact share pi in [0, 1) maximising sum_e w_e log((1 - pi) x_b + pi A_b) over a region's cells (concave
    in pi): 0 when the slope at 0 is not positive, else bisection of log pi on the sign of the derivative."""
    slope = 0.0
    for e in range(start, stop):
        b = pb[e]
        slope += pw[e] * (A[b] - x[b]) / x[b]
    if slope <= 0.0:
        return 0.0
    low, high = math.log(_SHARE_FLOOR), math.log(1.0 - REGION_SHARE_CEILING_GAP)
    for _ in range(_SHARE_BISECTIONS):
        middle = 0.5 * (low + high)
        pi = math.exp(middle)
        slope = 0.0
        for e in range(start, stop):
            b = pb[e]
            slope += pw[e] * (A[b] - x[b]) / ((1.0 - pi) * x[b] + pi * A[b])
        if slope > 0.0:
            low = middle
        else:
            high = middle
    return math.exp(0.5 * (low + high))


@kernel(
    overloads=(
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.literal(False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.literal(True),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.bool,
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
)
def _artefact_em(F, FA, A, cls, classes, with_c, y, region_start, pb, pw, nbar, x, share):
    """Maximise sum_R max_{pi_R} sum_{b in R} w_b log((1 - pi_R) x_b + pi_R A_b) over the mixture weights y (and c
    when with_c) by generalised EM: each E-step sets every region's pi_R to its maximum given x and gives each bin its
    posterior weight of being an ordinary (non-artefact) cell; the M-step takes damped Newton steps on the concave
    weighted mixture problem. The objective never decreases. Returns it at the final weights; share holds pi_R."""
    B = F.shape[1]
    Q = F.shape[0]
    contamination = classes * Q
    R = len(region_start) - 1
    previous = -np.inf
    total = -np.inf
    for iteration in range(_EM_ITERATIONS + 1):
        for b in range(B):
            m = 0.0
            base = cls[b] * Q
            for p in range(Q):
                m += y[base + p] * F[p, b]
            if with_c:
                m += y[contamination] * FA[b]
            x[b] = max(m, POSITIVE_WEIGHT_FLOOR)
            nbar[b] = 0.0
        total = 0.0
        for r in range(R):
            start, stop = region_start[r], region_start[r + 1]
            pi = _region_share(x, A, pb, pw, start, stop) if stop > start else 0.0
            share[r] = pi
            for e in range(start, stop):
                b = pb[e]
                mixed = (1.0 - pi) * x[b] + pi * A[b]
                total += pw[e] * math.log(mixed)
                nbar[b] += pw[e] * (1.0 - pi) * x[b] / mixed
        if iteration == _EM_ITERATIONS or total - previous <= _EM_TOLERANCE * max(1.0, abs(total)):
            break
        previous = total
        _weights(F, FA, nbar, cls, classes, with_c, y, _M_STEP_ITERATIONS)
    return total


@kernel(
    overloads=(
        (
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def _profiles(
    bin_start,
    orientation,
    depth,
    minor,
    cls,
    classes,
    region_start_all,
    pair_start,
    pb_all,
    pw_all,
    errors,
    balances,
    fractions,
    null_out,
    alt_out,
    share_out,
    mix_out,
    artefact_out,
    peak_out,
):
    """Per library (a range of distinct (orientation, depth, minor reads) bins, and its region-bin pairs): the null
    maximum, and per foreign fraction the alternative maximum and its contamination share, with the artefact
    components; mix_out holds the null's component weights (class-major; error rates, then own heterozygotes); with
    The unsupported artefact-free branch is omitted (P0 EC38)."""
    G = len(fractions)
    E = len(errors)
    V = len(balances)
    Q = E + V
    for lib in prange(len(bin_start) - 1):
        lo, hi = bin_start[lib], bin_start[lib + 1]
        B = hi - lo
        null_out[lib] = np.nan
        mix_out[lib, :] = np.nan
        artefact_out[lib] = np.nan
        peak_out[lib] = np.nan
        for g in range(G):
            alt_out[lib, g] = np.nan
            share_out[lib, g] = np.nan
        if B == 0:
            continue
        J = classes[lib]
        o, d, k = orientation[lo:hi], depth[lo:hi], minor[lo:hi]
        c = cls[lo:hi]
        p0, p1 = pair_start[lib], pair_start[lib + 1]
        pb = pb_all[p0:p1]
        pw = pw_all[p0:p1]
        region_start = region_start_all[lib]
        # Per-bin scale: the largest component log likelihood over every grid; it cancels in every comparison.
        logs = np.empty((Q, B))
        scale = np.empty(B)
        artefact = np.empty(B)
        for b in range(B):
            artefact[b] = _log_artefact(k[b], d[b])
            s = artefact[b]
            for u in range(E):
                logs[u, b] = _log_binomial(k[b], d[b], errors[u])
                s = max(s, logs[u, b])
            for v in range(V):
                h = balances[v, 0]
                logs[E + v, b] = _log_beta_binomial(k[b], d[b], h if o[b] == 1 else 1.0 - h, balances[v, 1])
                s = max(s, logs[E + v, b])
            for a in fractions:
                s = max(s, _log_binomial(k[b], d[b], a))
            scale[b] = s
        F = np.empty((Q, B))
        FA = np.zeros(B)
        A = np.empty(B)
        for b in range(B):
            A[b] = math.exp(artefact[b] - scale[b])
            for p in range(Q):
                F[p, b] = math.exp(logs[p, b] - scale[b])
        nbar = np.empty(B)
        x = np.empty(B)
        R = len(region_start) - 1
        share = np.empty(R)
        y = np.empty(J * Q + 1)
        start = np.empty(J * Q + 1)
        # With the artefact components.
        for j in range(J):
            for u in range(E):
                y[j * Q + u] = (1.0 - _WEIGHT_START) / E
            for v in range(V):
                y[j * Q + E + v] = _WEIGHT_START / V
        y[J * Q] = 0.0
        for b in range(B):
            FA[b] = 0.0
        null_out[lib] = _artefact_em(F, FA, A, c, J, False, y, region_start, pb, pw, nbar, x, share)
        for i in range(J * Q):
            mix_out[lib, i] = y[i]
        peak = 0.0
        for r in range(R):
            peak = max(peak, share[r])
        peak_out[lib] = peak
        kept = 0.0
        weight = 0.0
        for e in range(len(pw)):
            weight += pw[e]
        for b in range(B):
            kept += nbar[b]
        artefact_out[lib] = 1.0 - kept / weight if weight > 0.0 else np.nan
        y[J * Q] = _WEIGHT_START
        _feasible(y, J, Q, y[J * Q])
        best = -np.inf
        for g in range(G):
            for b in range(B):
                FA[b] = math.exp(_log_binomial(k[b], d[b], fractions[g]) - scale[b])
            if y[J * Q] < _WEIGHT_START:
                y[J * Q] = _WEIGHT_START
                _feasible(y, J, Q, y[J * Q])
            alt_out[lib, g] = _artefact_em(F, FA, A, c, J, True, y, region_start, pb, pw, nbar, x, share)
            share_out[lib, g] = y[J * Q]
            if alt_out[lib, g] > best:
                best = alt_out[lib, g]
                start[:] = y
        # The null is the alternative with no contamination share: polish it from the best alternative's weights
        # too, reducing the risk that incomplete null optimisation appears as evidence.
        start[J * Q] = 0.0
        _feasible(start, J, Q, 0.0)
        for b in range(B):
            FA[b] = 0.0
        polished = _artefact_em(F, FA, A, c, J, False, start, region_start, pb, pw, nbar, x, share)
        if polished > null_out[lib]:
            null_out[lib] = polished
            for i in range(J * Q):
                mix_out[lib, i] = start[i]
            peak = 0.0
            for r in range(R):
                peak = max(peak, share[r])
            peak_out[lib] = peak
            kept = 0.0
            for b in range(B):
                kept += nbar[b]
            artefact_out[lib] = 1.0 - kept / weight if weight > 0.0 else np.nan


def _depth_cuts(depth, weight, classes):
    """Cut depths of one library: its weighted eligible-cell depth quantiles (a bin's class is the number of cuts
    at or below its depth, relabelled over the non-empty classes)."""
    if len(depth) == 0 or classes <= 1:
        return np.zeros(0, np.int64)
    order = np.argsort(depth, kind="stable")
    cumulative = np.cumsum(weight[order]) / weight.sum()
    cuts = [
        depth[order][min(int(np.searchsorted(cumulative, q)), len(order) - 1)] for q in np.arange(1, classes) / classes
    ]
    return np.unique(cuts)


def _classes(depth, cuts):
    raw = np.searchsorted(cuts, depth, side="right")
    labels, cls = np.unique(raw, return_inverse=True)  # consecutive labels of the non-empty classes
    return cls.astype(np.int64), max(len(labels), 1)


def _fit(groups, regions, errors, balances, fractions, config):
    """Profiles of the libraries: groups holds each library's (region, orientation, depth, minor, weight) histogram
    rows, sorted by region; regions is the number of regions. Returns null[n], alternative[n, G], share[n, G], the
    null's mixture weights, the class counts, the bins (orientation, depth, minor, merged weight) and class labels of
    each library and the null's artefact share and peak."""
    n_lib = len(groups)
    bins, labels, counts, offsets, pair_offsets, pbs, pws = [], [], [], [0], [0], [], []
    region_start = np.zeros((n_lib, regions + 1), np.int64)
    for lib, (region, o, d, k, w) in enumerate(groups):
        code = (o * _CODE + d) * _CODE + k
        unique, inverse = np.unique(code, return_inverse=True)
        merged = np.bincount(inverse, weights=w, minlength=len(unique)) if len(w) else np.zeros(0)
        bo, bd, bk = unique // _CODE // _CODE, unique // _CODE % _CODE, unique % _CODE
        cls, count = _classes(bd, _depth_cuts(bd, merged, config.heterozygote_depth_classes))
        bins.append((bo, bd, bk, merged))
        labels.append(cls)
        counts.append(count)
        offsets.append(offsets[-1] + len(unique))
        region_start[lib] = np.searchsorted(region, np.arange(regions + 1))
        pbs.append(inverse.astype(np.int64))
        pws.append(np.asarray(w, np.float64))
        pair_offsets.append(pair_offsets[-1] + len(w))
    column = [np.concatenate([part[i] for part in bins]) if bins else np.zeros(0) for i in range(4)]
    cls = np.concatenate(labels) if labels else np.zeros(0, np.int64)
    classes = np.asarray(counts, np.int64)
    Q = len(errors) + len(balances)
    G = len(fractions)
    null = np.empty(n_lib)
    alternative, share = (np.empty((n_lib, G)) for _ in range(2))
    artefact, peak = np.empty(n_lib), np.empty(n_lib)
    mix = np.empty((n_lib, int(classes.max(initial=1)) * Q))
    _profiles(
        np.asarray(offsets, np.int64),
        column[0].astype(np.int64),
        column[1].astype(np.float64),
        column[2].astype(np.float64),
        cls.astype(np.int64),
        classes,
        region_start,
        np.asarray(pair_offsets, np.int64),
        np.concatenate(pbs) if pbs else np.zeros(0, np.int64),
        np.concatenate(pws) if pws else np.zeros(0),
        errors,
        balances,
        fractions,
        null,
        alternative,
        share,
        mix,
        artefact,
        peak,
    )
    return null, alternative, share, mix, classes, bins, labels, artefact, peak


# ---------------------------------------------------------------------------------------------
# Inherited-segment diagnostic
# ---------------------------------------------------------------------------------------------


def _truncated_binomial(k, d, p):
    """log P(k | d, p, k >= 1) without the binomial coefficient (common to both components)."""
    return k * np.log(p) + (d - k) * np.log1p(-p) - np.log1p(-((1.0 - p) ** d))


def _low_log_ratio(k, d, het):
    """Per cell: log likelihood of the low fraction het/2 over the heterozygous fraction het."""
    return _truncated_binomial(k, d, 0.5 * het) - _truncated_binomial(k, d, het)


def _event_weights(group, position, fragment_bp):
    """1 / size of each cell's event; cells are sorted by (group, position)."""
    new = np.ones(len(position), bool)
    new[1:] = (group[1:] != group[:-1]) | (np.diff(position) >= fragment_bp)
    event = np.cumsum(new) - 1
    return 1.0 / np.bincount(event)[event]


def _segments(n_samples, cells, flagged, config):
    k, d, het, position, sample, partner, partner_k, partner_d = cells
    hetlike = np.flatnonzero(_low_log_ratio(k, d, het) <= 0)
    hetlike = hetlike[np.lexsort((position[hetlike], partner[hetlike], sample[hetlike]))]
    pair = sample[hetlike] * (n_samples + 1) + partner[hetlike] + 1
    weight = _event_weights(pair, position[hetlike], config.fragment_bp)
    region = np.ones(len(hetlike), bool)
    region[1:] = (pair[1:] != pair[:-1]) | (np.diff(position[hetlike]) >= config.segment_gap_bp)
    bounds = np.flatnonzero(np.r_[region, True])
    segments = []
    for start, stop in zip(bounds[:-1], bounds[1:], strict=False):
        events = float(weight[start:stop].sum())
        s, p = int(sample[hetlike[start]]), int(partner[hetlike[start]])
        if events < config.segment_minimum_events or flagged[s]:
            continue
        first_position, last_position = position[hetlike[start]], position[hetlike[stop - 1]]
        # Every near-private cell of the pair in the region, whatever its reads.
        members = np.flatnonzero(
            (sample == s) & (partner == p) & (position >= first_position) & (position <= last_position)
        )
        partner_fraction = None
        if p >= 0:
            partner_fraction = float(partner_k[members].sum() / partner_d[members].sum())
        segments.append(
            InheritedSegment(
                sample=s,
                partner=p,
                start=int(first_position),
                end=int(last_position),
                events=events,
                cells=int(len(members)),
                minor_read_fraction=float(k[members].sum() / d[members].sum()),
                partner_minor_read_fraction=partner_fraction,
            )
        )
    return tuple(segments)


# ---------------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------------


def assess_counts(
    counts,
    positions,
    passing,
    heterozygote_balance,
    error_rate,
    config: SampleQualityParameters,
    *,
    balance_interval=None,
    balance_reason=None,
    chunk_sites: int = _CHUNK_SITES,
    reverse: bool = False,
) -> SampleQualityDecision:
    """Assess raw counts without allocating arrays across the full sample/site axes.

    counts: int32 (samples, input sites, 2), REF then ALT; passing: bool[input sites] selecting the r1-r4-passing
    sites; heterozygote_balance: r4's clean ALT fraction h0; error_rate: r3's clean wrong-allele rate eps0;
    balance_interval, balance_reason: the own-heterozygote balance interval and why it is wider than h0 (default
    (h0, h0); see ``balance_support``). Dense scratch is
    bounded by ``chunk_sites`` passing sites (plus the eligible cells of one fragment window); retained summaries are
    the per-library histograms and at most two near-private carriers per passing site.
    """
    selected = np.flatnonzero(passing)
    passing_positions = np.asarray(positions[selected], np.int64)

    def tiles():
        for start in range(0, len(selected), chunk_sites):
            columns = selected[start : start + chunk_sites]
            alt = np.asarray(counts[:, columns, 1].T, np.int64)
            dep = np.asarray(counts[:, columns, :].sum(axis=2, dtype=np.int32).T, np.int64)
            yield alt, dep, passing_positions[start : start + chunk_sites]

    return _assess(
        counts.shape[0],
        tiles(),
        passing_positions,
        heterozygote_balance,
        error_rate,
        config,
        balance_interval,
        balance_reason,
        reverse=reverse,
    )


def _grids(error_rate, interval, config):
    """The error, own-heterozygote and foreign grids with a_min and a_hi, or the reason the QC cannot run."""
    minimum = 0.5 * config.minimum_foreign_fraction
    floor = config.heterozygote_floor_ratio * interval[0]
    ceiling = 0.5 * floor
    if not error_rate * _ERROR_SEPARATION < minimum:
        return None, (
            f"clean error rate {error_rate:.3g} not below the minimum foreign fraction {minimum:g} / "
            f"{_ERROR_SEPARATION:g}"
        )
    if not ceiling > _MINIMUM_FOREIGN_RANGE * minimum:
        return None, (
            f"foreign ceiling {ceiling:.3g} (half the balance floor) not above {_MINIMUM_FOREIGN_RANGE:g} x "
            f"the minimum foreign fraction {minimum:g}"
        )
    bottom = _ERROR_GRID_BOTTOM * error_rate
    steps = max(int(math.ceil(math.log(minimum / bottom) / math.log(_ERROR_GRID_RATIO))), 1)
    errors = np.exp(np.linspace(math.log(bottom), math.log(minimum), steps + 1))
    means = np.unique(np.array([interval[0], 0.5 * (interval[0] + interval[1]), interval[1]]))
    balances = np.array([(h, rho) for h in means for rho in _HET_DISPERSION_GRID])
    fractions = np.exp(np.linspace(math.log(minimum), math.log(ceiling), _FOREIGN_GRID + 1))[1:]
    return (errors, balances, fractions, minimum, ceiling), None


def _assess(
    n_samples,
    tiles,
    passing_positions,
    heterozygote_balance,
    error_rate,
    config,
    balance_interval,
    balance_reason,
    *,
    reverse=False,
):
    interval = (
        tuple(float(h) for h in balance_interval)
        if balance_interval is not None
        else (float(heterozygote_balance), float(heterozygote_balance))
    )
    level = config.carrier_tail_level / max(n_samples - 1, 1)
    bounds = _region_bounds(passing_positions, config.artefact_regions, config.fragment_bp)
    origin = int(passing_positions[0]) if len(passing_positions) else 0
    collector = _Collector(
        n_samples, error_rate, level, heterozygote_balance, config.fragment_bp, bounds, origin, reverse=reverse
    )
    for alt, dep, positions in tiles:
        collector.add(alt, dep, positions)
    (sample, region, orientation, depth, minor, weight), cells, minor_allele = collector.finish()
    nan = np.full(n_samples, np.nan)
    (lrt, plain_lrt, fraction, share, balance_fit, dispersion, error_fit, artefact, peak) = (
        nan.copy() for _ in range(9)
    )
    flagged = np.zeros(n_samples, bool)
    reason = grids = None
    if int((collector.observed > 0).sum()) < config.minimum_samples:
        reason = f"fewer than {config.minimum_samples} samples with observed cells"
    elif not len(sample):
        reason = "no eligible cells"
    else:
        grids, reason = _grids(error_rate, interval, config)
    threshold = minimum = ceiling = None
    if reason is None:
        errors, balances, fractions, minimum, ceiling = grids
        threshold = float(chi2.isf(2.0 * config.carrier_tail_level / (len(fractions) * n_samples), 1))
        offsets = np.searchsorted(sample, np.arange(n_samples + 1))
        groups = [
            tuple(column[offsets[s] : offsets[s + 1]] for column in (region, orientation, depth, minor, weight))
            for s in range(n_samples)
        ]
        (null, alternative, alternative_share, mix, classes, bins, labels, artefact_share, artefact_peak) = _fit(
            groups, collector.regions, errors, balances, fractions, config
        )
        E, Q = len(errors), len(errors) + len(balances)
        for s in np.flatnonzero(np.isfinite(null)):
            # the null's fitted error and own-heterozygote distributions, pooled over the depth classes
            class_cells = np.bincount(labels[s], weights=bins[s][3], minlength=classes[s])
            mixture = mix[s, : classes[s] * Q].reshape(classes[s], Q)
            error_mass = (mixture[:, :E] * class_cells[:, None]).sum(axis=0)
            error_fit[s] = float(error_mass @ errors / error_mass.sum()) if error_mass.sum() > 0 else np.nan
            het_mass = (mixture[:, E:] * class_cells[:, None]).sum(axis=0)
            if het_mass.sum() > 0:
                balance_fit[s] = float(het_mass @ balances[:, 0] / het_mass.sum())
                dispersion[s] = float(het_mass @ balances[:, 1] / het_mass.sum())
            profile = np.maximum(alternative[s], null[s])
            best = ties.conv_argmax(profile, reverse=False)
            lrt[s] = 2.0 * (profile[best] - null[s])
            fraction[s] = fractions[best]
            share[s] = alternative_share[s, best] if profile[best] > null[s] else 0.0
            artefact[s], peak[s] = artefact_share[s], artefact_peak[s]
        flagged = np.isfinite(lrt) & (lrt >= threshold)
    return SampleQualityDecision(
        applied=reason is None,
        reason=reason,
        passing_sites=collector.passing_sites,
        heterozygote_balance=float(heterozygote_balance),
        balance_interval=interval,
        balance_reason=balance_reason,
        error_rate=float(error_rate),
        flagged=flagged,
        lrt=lrt,
        plain_lrt=plain_lrt,
        foreign_fraction=fraction,
        foreign_share=share,
        heterozygote_fit=balance_fit,
        heterozygote_dispersion=dispersion,
        error_fit=error_fit,
        artefact_share=artefact,
        artefact_peak=peak,
        observed_cells=collector.observed,
        eligible_cells=collector.eligible,
        eligible_windows=np.bincount(sample, weights=weight, minlength=n_samples),
        minor_cells=collector.minor_cells,
        near_private_cells=np.bincount(cells[4], minlength=n_samples),
        significance_threshold=threshold,
        minimum_fraction=minimum,
        fraction_ceiling=ceiling,
        carrier_level=float(level),
        artefact_regions=int(collector.regions),
        segments=_segments(n_samples, cells, flagged, config),
        config=SampleQualitySettings(**asdict(config)),
        tie_events=TieCounts(tuple(TieCount(kind, **values) for kind, values in minor_allele.items())),
    )


__all__ = ["InheritedSegment", "SampleQualityDecision", "balance_support", "carrier_thresholds", "assess_counts"]
