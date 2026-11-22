"""Read sharing between nearby sites, from read depths alone (4.21, item RS; lab421/rs/DESIGN_RS.md).

Two sites a short distance apart are covered by some of the same sequenced molecules, so their depths covary within a
sample. phi(d), the read sharing at distance d, is that covariance in Poisson units: the share of the reads at a site
that also give a base d bp away. With n[i, s] the depth (REF + ALT reads) of sample i at site s of one contig:

    w_i  = the sample's share of all the contig's reads,   a_i = N w_i  (its depth factor, mean one)
    b[s] = mean over samples of n[i, s]
    e[i, s] = n[i, s] - a_i b[s] = n[i, s] - w_i n[., s]                one factor per sample and contig
    num[i, k] = sum over the pairs (s < t) of lag bin k of e[i, s] e[i, t]     (k = 0: the sum of e[i, s]^2, lag 0)

Model. A sample's covariance has a part that grows linearly with its depth and a part that grows quadratically:

    Cov_i(n_s, n_t) = a_i sqrt(b_s b_t) phi(d) + a_i^2 Q[s, t]

The linear part is Poisson sampling of reads (shared reads; at lag 0 the Poisson dispersion, 1). The quadratic part is a
sample-specific pattern of the read intensity along the contig (regional depth, missing tracts, copy number, cell
overdispersion), with Q its covariance at unit depth factor: it is not read sharing. Telling the two apart by their
growth with depth assumes one sharing curve for all samples and an intensity pattern whose law does not depend on the
sample's depth; samples of different libraries violate the first (their difference then sits in the quadratic part).

Expectations. With b and w taken as fixed, independent samples and S = P Q P^T, P = I - (b / sum b) 1^T the centring of
the fitted factor:

    E[num_ik] = a_i (1 - w_i) sum_j A_kj phi_j  +  a_i^2 (1 - 2 w_i + sum_j w_j^2) q_k

A (``_geometry``) is the centring map of the linear part for covariance constant within a lag bin and zero beyond the
last one; q_k is the sum of S over the bin's pairs (its trace at lag 0). The map is exact for that fixed, binned
operator; b and w are fitted from the same depths, so the estimator as a whole is a plug-in.

Two fits per bin, weighted least squares over the samples with reads of y_ik = num_ik / (a_i W1_k) (weights a_i; W1_k,
W2_k the bin's sums of sqrt(b_s b_t) and b_s b_t over its pairs, of b and b^2 at lag 0):

    pooled:  y = pi (1 - w)                      every covariance read as shared reads
    split:   y = theta (1 - w) + psi x           x_ik = a_i (1 - 2 w_i + sum w^2) W2_k / W1_k

Under the model theta is unbiased for M phi (M = diag(1 / W1) A) and pi_k - theta_k estimates the pooled fit's bias,
C q_k / W1_k (C a constant of the cohort's depth factors). The split needs samples of different depth. A bin's split is
UNDETERMINED when its two regressors are collinear in the full fit or in any delete-one fit (determinant of the normal
equations below _RANK_TOLERANCE of its scale), or with fewer than three samples with reads; then the pooled value is
published for that bin with weight g = 0 and flagged (``split_determined``, ``unseparated``).

Combination (in the uncentred scale, bin by bin; phi = M^-1 of the result, one curve on every measured lag):

    h_j   = max(pooled phi_j, 0), j = 1..K      the pooled curve, pi centred by M^-1 on every measured lag
    L     = M_00 + sum over j > 0 of min(M_0j, 0) h_j       the least lag-0 value the linear part can take
    D     = max(pi_0 - L, 0)
    r_k   = (W2_k / W1_k) / (W2_0 / W1_0)                  the bin's pairs' depth weight over lag 0's
    s_k   = r_k D
    g_k   = s_k^2 / (s_k^2 + v_k)       v_k the jackknife variance of pi_k - theta_k;  0 where undetermined
    raw_k = pi_k - clip(g_k (pi_k - theta_k), -s_k, s_k)

What D is. Under the model E[pi_0] = M_00 phi_0 + sum_{j>0} M_0j phi_j + delta_0, delta_0 = C q_0 / W1_0 the pooled
bias at lag 0. (F): reads are sampled at least as dispersed as Poisson, phi_0 >= 1 (and M_00 = 1 - sum b^2 / (sum b)^2
> 0). (S): at every positive lag the sharing lies between 0 and the positive part of the pooled curve, 0 <= phi_j <=
h_j. The coefficient of a lag on lag 0 has either sign,

    M_0j = (2 W1_j / B^2) (sum b^2 / sum b  -  the mean of b_s + b_t over the bin's pairs, weights sqrt(b_s b_t)),

negative when the bin's pairs lie at sites of ordinary depth, positive when their two mean depths together are below
the contig's read-weighted mean depth (uneven site means with the close pairs at shallow sites). So the linear part is
least with phi_j = h_j where M_0j < 0 and phi_j = 0 where M_0j > 0: that is L, and E[D] >= delta_0 (the positive
parts, of h and of D, only raise it). All of it is within the binned model: no covariance beyond the largest
measured lag.
What s_k is. Under (H) the centred intensity pattern has the same relative variance at every site, S[s, s] = sigma^2
b_s^2; then |S[s, t]| <= sigma^2 b_s b_t (Cauchy-Schwarz on S), so |q_k| <= sigma^2 W2_k and the pooled bias of bin k
is at most r_k times that of lag 0: |pi_k - theta_k| <= r_k delta_0 <= s_k in expectation. The bound is for the bin's
own pair weights (r_k differs from 1 when close pairs sit at deeper or shallower sites than the average read) and is
stated before the centring inverse, which is applied afterwards and need not preserve any bound.
What g_k is, and is not. For an unbiased observation of a fixed quantity delta with |delta| <= s and error variance v,
the worst-case risk of the linear correction g x observation is (1 - g)^2 s^2 + g^2 v, least at g = s^2 / (s^2 + v).
Here s and v are estimated, the bound holds only under (F), (H), (S), and the risk of the published value also holds
-2 g Cov(pooled error, difference error): g_k is that narrower optimum, not a minimax guarantee for phi.
The limit at s_k. Under (F), (H), (S) and the binned model the pooled bias lies in [-s_k, s_k] in expectation, so a
correction beyond that interval would be error of the split, whose standard error is several times the pooled one,
and limiting it cannot make the correction worse. s_k itself is estimated (D is a fitted value, not a confidence
bound), so in a given data set the limit can also hold back part of a correct split. The price is stated, not hidden:
where an assumption fails (an intensity pattern more variable where markers cluster; samples whose sharing differs
with their depth) a correct split beyond s_k is held back however much data there is, and for a positive intensity
part the limit stops subtractions that are too large while letting through ones that are too small (an upward bias
where the split is noisy). ``clipped`` publishes what the limit held back in each bin. A value beyond the split's
standard error is evidence to examine: an assumption may fail on that contig, or the split may be off by more than
its jackknife error; ``excess`` may then keep up to that much covariance that is not read sharing.
Equal depths. The split is undetermined (exactly equal totals) or its variance is huge (totals equal up to sampling):
the pooled value is published. It is an upper estimate of read sharing only where the intensity covariance at that lag
is non-negative. ``excess_dispersion`` (D) and ``unseparated`` = (1 - g_k) s_k say how much of it could be intensity
under (F), (H), (S): nothing on Poisson depths (D = 0), up to s_k otherwise.

Lag bins: the fixed edges to 1 kb (the resolution where paired-end sharing varies), then half-octaves to a quarter of
the contig's span. These, the tested octave below and the rank tolerance are choices of resolution, none fitted to
data. Every pair of every bin is counted, by prefix sums: ``_pair_sums`` is O(samples x sites x bins); ``_site_sums``
(the geometry) is O(sites x bins^2) plus two binary searches per bin and tile, with tiles x (bins + 1)^2 floats of
scratch.

Uncertainty: delete-one-sample jackknife of the whole curve, conditional on b, w and the weights g. It does not hold the
error of those, nor the model's.

Range. A consumer weighs a cell by 1 / (1 + sum of phi over its neighbours), so the range is chosen for that sum. With
nu_k = 2 pairs_k / sites the mean number of neighbours a site has in bin k, S(R) the sum of nu_k phi_k (signed
estimates) over the bins below R and T(R) the same sum over the octave beyond it, [R, 2R): for unbiased estimates,
counting the octave changes the mean squared error of the sum by Var T + 2 Cov(S, T) - (E T)^2 - 2 (E T) U, U the mean
of the tail beyond 2R. With U >= 0 the octave is worth counting whenever (E T)^2, estimated without bias by T^2 - Var T,
exceeds Var T + 2 Cov(S, T). r_phi is the smallest bin edge R at which that sufficient condition fails:

    not ( T > 0  and  T^2 > 2 Var T + 2 Cov(S, T) )                variances by the same jackknife

With independent errors that is T <= sqrt(2) se(T); errors shared between lags raise the bar through the covariance. It
is a greedy rule on an estimated local risk: it gives a sufficient reason to extend, not the range of least total risk,
and it is not a test of absence (a first bin of pure noise is counted in about 8 % of contigs). Inside the range the
consumers' phi is max(estimate, 0) per bin; beyond it they apply 0, a working value and not an estimate that nothing is
shared there. ``omitted`` is T at the range and ``omitted_upper`` its one-sided 95 % limit: they cover that octave
only, not what lies beyond it. Lags beyond an octave with no countable sharing are not looked at (a second lobe
separated by an empty octave is not counted).

Status: none_detected (R = 0), resolved_within (the rule stops inside the measured lags: a decision about the range, not
a certificate of accuracy) or unresolved (sharing is still countable at the largest measured lag; fewer than three
samples with reads; or sites that do not determine the curve: then the whole measured curve is the consumer curve).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import prange, types

from ...core.kernels import kernel

__all__ = ["ReadSharing", "read_sharing", "lag_edges", "FINE_EDGES_BP"]

FINE_EDGES_BP = (0, 25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 850, 1000)
"""Bin edges (bp) to 1 kb, as released: the resolution of the curve where paired-end sharing varies."""
UPPER_Z = 1.6448536269514722
"""One-sided 95 % normal quantile of the published upper limit of the omitted weight."""
_TILE_SITES = 1 << 12
"""Sites per tile of the centring map's site sums: the unit of parallel work and of the fixed summation order."""
_RANK_TOLERANCE = 1e-12
"""Numerical rank tolerance of a bin's two-term fit: the determinant of its normal equations over the product of their
diagonal is 1 - r^2, r the weighted correlation of the two regressors; float64 rounding of that difference is about
1e-16 per term, and a depth spread of one part in 1e5 already gives 1e-10. Below it the split is undetermined."""
_CONDITION_LIMIT = 1e8
"""Numerical rank tolerance of the centring map: beyond this condition number (realistic contigs: 2 to 25) the sites do
not determine the curve (two sites, or every pair inside the measured lags) and the result is unresolved."""
_RISK_VARIANCE_TERMS = 2.0
"""The factor of Var T in the range rule (module docstring): once in the unbiased estimate of (E T)^2 and once as the
variance that counting the octave adds. Derived, not fitted; the lab's sensitivity check replaces it by 1 and 4."""


def lag_edges(span_bp: int) -> np.ndarray:
    """int64 bin edges (bp): FINE_EDGES_BP, then 1000 x 2^(j/2), up to a quarter of the contig's span."""
    limit = max(int(span_bp) // 4, FINE_EDGES_BP[1])
    edges = [e for e in FINE_EDGES_BP if e <= limit]
    j = 1
    while True:
        edge = int(round(1000.0 * 2.0 ** (j / 2.0)))
        if edge > limit:
            break
        edges.append(edge)
        j += 1
    return np.asarray(edges, dtype=np.int64)


@kernel(
    overloads=(
        (
            types.Array(types.int64, 2, "C", readonly=True), types.Array(types.float64, 1, "C", readonly=True),
            types.Array(types.float64, 1, "C", readonly=True), types.Array(types.int64, 1, "C", readonly=True),
            types.Array(types.int64, 1, "C", readonly=True), types.Array(types.float64, 2, "C"),
        ),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _pair_sums(depth, share, site_total, positions, edges, out):
    """out[i, 0] = sum_s e_is^2 and out[i, k + 1] = sum over the pairs s < t with edges[k] <= x_t - x_s < edges[k + 1]
    of e_is e_it, e_is = depth[i, s] - share[i] site_total[s]. Every pair is counted: the partners of s in a bin are
    consecutive sites, so their sum is a difference of the prefix sums of e, and the bin boundaries only move forward
    with s (one sweep per sample). Samples are independent (parallel over samples)."""
    n_samples, n_sites = depth.shape
    n_edges = edges.shape[0]
    for i in prange(n_samples):
        prefix = np.empty(n_sites + 1)
        prefix[0] = 0.0
        square = 0.0
        for s in range(n_sites):
            e = depth[i, s] - share[i] * site_total[s]
            prefix[s + 1] = prefix[s] + e
            square += e * e
        out[i, 0] = square
        pointer = np.zeros(n_edges, dtype=np.int64)
        sums = np.zeros(n_edges - 1)
        for s in range(n_sites):
            e = depth[i, s] - share[i] * site_total[s]
            x = positions[s]
            for k in range(n_edges):
                p = pointer[k]
                if p <= s:
                    p = s + 1
                target = x + edges[k]
                while p < n_sites and positions[p] < target:
                    p += 1
                pointer[k] = p
            for k in range(n_edges - 1):
                sums[k] += e * (prefix[pointer[k + 1]] - prefix[pointer[k]])
        for k in range(n_edges - 1):
            out[i, k + 1] = sums[k]


@kernel(
    overloads=(
        (
            types.Array(types.int64, 1, "C", readonly=True), types.Array(types.float64, 1, "C", readonly=True),
            types.Array(types.float64, 1, "C", readonly=True), types.Array(types.int64, 1, "C", readonly=True),
            types.int64, types.Array(types.float64, 3, "C"), types.Array(types.float64, 2, "C"),
            types.Array(types.float64, 2, "C"), types.Array(types.int64, 2, "C"),
        ),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _site_sums(positions, root, b, edges, tile, cross, w1, w2, pairs):
    """Per tile of ``tile`` consecutive sites (independent, parallel): cross[t, k, j] = sum_s B_k(s) sqrt(b_s) H_j(s),
    w1[t, k], w2[t, k] the sums over the pairs s < t of bin k of sqrt(b_s b_t) and b_s b_t, pairs[t, k] their count
    (``_geometry``; index 0 of cross, w1, w2 is lag 0). H_j(s), B_k(s): the sums of sqrt(b) and b over the sites at a
    lag in bin j (k) of s on either side, from prefix sums and bin boundaries that only move forward with s."""
    n_sites = positions.shape[0]
    n_edges = edges.shape[0]
    n_tiles = cross.shape[0]
    prefix_root = np.empty(n_sites + 1)
    prefix_b = np.empty(n_sites + 1)
    prefix_root[0] = 0.0
    prefix_b[0] = 0.0
    for s in range(n_sites):
        prefix_root[s + 1] = prefix_root[s] + root[s]
        prefix_b[s + 1] = prefix_b[s] + b[s]
    for t in prange(n_tiles):
        start = t * tile
        stop = min(n_sites, start + tile)
        # the boundaries of the tile's first site by binary search (each tile then only sweeps its own sites)
        forward = np.empty(n_edges, dtype=np.int64)  # first site after s with x - x_s >= edge
        backward = np.empty(n_edges, dtype=np.int64)  # sites before s with x_s - x >= edge
        for k in range(n_edges):
            forward[k] = max(np.searchsorted(positions, positions[start] + edges[k]), start + 1)
            backward[k] = min(np.searchsorted(positions, positions[start] - edges[k], side="right"), start)
        near_root = np.empty(n_edges)
        near_b = np.empty(n_edges)
        for s in range(start, stop):
            x = positions[s]
            for k in range(n_edges):
                p = forward[k]
                if p <= s:
                    p = s + 1
                target = x + edges[k]
                while p < n_sites and positions[p] < target:
                    p += 1
                forward[k] = p
                q = backward[k]
                target = x - edges[k]
                while q < s and positions[q] <= target:
                    q += 1
                backward[k] = q
            near_root[0] = root[s]
            near_b[0] = b[s]
            w1[t, 0] += b[s]
            w2[t, 0] += b[s] * b[s]
            for k in range(n_edges - 1):
                ahead_root = prefix_root[forward[k + 1]] - prefix_root[forward[k]]
                ahead_b = prefix_b[forward[k + 1]] - prefix_b[forward[k]]
                near_root[k + 1] = ahead_root + prefix_root[backward[k]] - prefix_root[backward[k + 1]]
                near_b[k + 1] = ahead_b + prefix_b[backward[k]] - prefix_b[backward[k + 1]]
                w1[t, k + 1] += root[s] * ahead_root
                w2[t, k + 1] += b[s] * ahead_b
                pairs[t, k] += forward[k + 1] - forward[k]
            for k in range(n_edges):
                scaled = near_b[k] * root[s]
                for j in range(n_edges):
                    cross[t, k, j] += scaled * near_root[j]


def _geometry(positions, b, edges):
    """(W1 [K + 1], W2 [K + 1], pairs int64 [K], M [K + 1, K + 1]) of a contig's sites: W1_k, W2_k the sums over the
    pairs of lag bin k of sqrt(b_s b_t) and b_s b_t (index 0: lag 0, the sums of b and b^2), and the centring map M.

    The factor a_i is fitted from the contig's own reads, so e_i = (I - beta 1^T) e0_i with beta = b / sum b and e0 the
    residual at the true factor. For a covariance sum_j phi_j G_j, G_j[s, t] = sqrt(b_s b_t) on the pairs of bin j (the
    diagonal b for j = 0), the expected pair sums of bin k are sum_j A_kj phi_j with

        A_kj = [k = j] W1_k - c_k X_kj / B + (W2_k / B^2) T_j,      B = sum b,  c_0 = 2,  c_k = 1

    X_kj = sum_s sqrt(b_s) H_j(s) B_k(s), H_j(s) and B_k(s) the sums of sqrt(b) and of b over the sites at a lag in bin
    j (k) of s on either side (j = 0: the site itself), and T_j = sum_s sqrt(b_s) H_j(s). M = diag(1 / W1) A. The site
    sums are accumulated per tile of _TILE_SITES sites and added in tile order (the same result at any thread count)."""
    n_sites, n_bins = len(positions), len(edges) - 1
    tiles = max(1, -(-n_sites // _TILE_SITES))
    cross_tiles = np.zeros((tiles, n_bins + 1, n_bins + 1))
    w1_tiles, w2_tiles = np.zeros((tiles, n_bins + 1)), np.zeros((tiles, n_bins + 1))
    pair_tiles = np.zeros((tiles, n_bins), dtype=np.int64)
    frozen = []
    for array in (positions, np.sqrt(b), b, edges):
        view = np.ascontiguousarray(array).view()
        view.flags.writeable = False
        frozen.append(view)
    _site_sums(frozen[0], frozen[1], frozen[2], frozen[3], _TILE_SITES, cross_tiles, w1_tiles, w2_tiles, pair_tiles)
    cross, w1, w2, pairs = cross_tiles.sum(axis=0), w1_tiles.sum(axis=0), w2_tiles.sum(axis=0), pair_tiles.sum(axis=0)
    total = float(w1[0])
    reach = np.concatenate(([total], 2.0 * w1[1:]))
    a = -cross / total + np.outer(w2, reach) / (total * total)
    a[0] = -2.0 * cross[0] / total + w2[0] * reach / (total * total)
    a[np.arange(n_bins + 1), np.arange(n_bins + 1)] += w1
    occupied = w1 > 0
    m = np.eye(n_bins + 1)
    m[np.ix_(occupied, occupied)] = a[np.ix_(occupied, occupied)] / w1[occupied][:, None]
    return w1, w2, pairs, m


def _fits(suu, suy, suv, svv, svy):
    """(pooled, split, determined): the one-term and two-term weighted least squares per bin from the weighted sums
    (broadcast). ``determined`` is False where the two regressors are collinear (_RANK_TOLERANCE); the split then
    returns the pooled value and must not be read as an estimate."""
    pooled = suy / suu
    determinant = suu * svv - suv * suv
    determined = determinant > _RANK_TOLERANCE * np.abs(suu * svv)
    safe = np.where(determined, determinant, 1.0)
    return pooled, np.where(determined, (svv * suy - suv * svy) / safe, pooled), determined


def _jackknife(replicates):
    """Standard error over delete-one replicates [N, ...]."""
    n = replicates.shape[0]
    return np.sqrt((n - 1) / n * ((replicates - replicates.mean(axis=0)) ** 2).sum(axis=0))


@dataclass(frozen=True)
class _Curves:
    """The fitted curves of one contig, index 0 = lag 0 (``_curves``)."""

    estimate: np.ndarray  # float64[K + 1] the combined curve, centred
    replicates: np.ndarray  # float64[N', K + 1] its delete-one-sample replicates (weights fixed)
    pooled: np.ndarray  # float64[K + 1] centred pooled fit
    split: np.ndarray  # float64[K + 1] centred split fit (the pooled value where undetermined)
    standard_error: np.ndarray  # float64[K + 1] of estimate, pooled, split
    pooled_se: np.ndarray
    split_se: np.ndarray
    weight: np.ndarray  # float64[K + 1] g
    determined: np.ndarray  # bool[K + 1]
    scale: np.ndarray  # float64[K + 1] s = r D
    ratio: np.ndarray  # float64[K + 1] r
    bound: float  # D
    samples: int  # samples with reads
    clipped: np.ndarray  # float64[K + 1] the part of the weighted split difference beyond s (not applied)


def _curves(sums, share, w1, w2, centring, inverse) -> _Curves:
    """The pooled and split fits of the per-sample pair sums [N, K + 1] and their combination (module docstring).
    share: the samples' depth shares [N]; w1, w2: the bins' pair weights; centring, inverse: M and its inverse. The
    combination is made before the centring."""
    n_bins = sums.shape[1]
    occupied = w1 > 0
    used = share > 0  # a sample without a read has no covariance to give
    n_used = int(used.sum())
    a = share[used] * len(share)
    f1 = 1.0 - share[used]
    f2 = 1.0 - 2.0 * share[used] + float((share**2).sum())
    y = np.divide(sums[used], a[:, None] * w1[None, :], out=np.zeros((n_used, n_bins)), where=occupied[None, :])
    x = (a * f2)[:, None] * np.divide(w2, w1, out=np.zeros(n_bins), where=occupied)[None, :]
    wu = a * f1
    suu = float((wu * f1).sum())
    suy, suv = (wu[:, None] * y).sum(axis=0), (wu[:, None] * x).sum(axis=0)
    svv, svy = (a[:, None] * x * x).sum(axis=0), (a[:, None] * x * y).sum(axis=0)
    pooled, split, full = _fits(suu, suy, suv, svv, svy)
    # delete-one-sample replicates of both fits [N', K + 1] (b, w and the factors held fixed)
    rep1, rep2, each = _fits(
        (suu - wu * f1)[:, None], suy[None, :] - wu[:, None] * y, suv[None, :] - wu[:, None] * x,
        svv[None, :] - a[:, None] * x * x, svy[None, :] - a[:, None] * x * y,
    )
    # a bin's split counts only when the full fit and every delete-one fit determine it
    determined = occupied & full & each.all(axis=0) & (n_used >= 3)
    pooled = np.where(occupied, pooled, 0.0)
    rep1 = np.where(occupied[None, :], rep1, 0.0)
    split = np.where(determined, split, pooled)
    rep2 = np.where(determined[None, :], rep2, rep1)
    if n_used >= 3:
        variance = _jackknife(rep1 - rep2) ** 2
    else:
        variance = np.full(n_bins, np.inf)
    ratio = np.zeros(n_bins)
    ratio[occupied] = (w2[occupied] / w1[occupied]) / (w2[0] / w1[0])
    # D (module docstring): the lag-0 statistic over the least value its linear part can take with the sharing of each
    # positive lag between 0 and the positive part of the pooled curve (S). A lag whose coefficient on lag 0 is
    # negative counts at that upper value, one whose coefficient is positive at 0.
    reference = np.maximum(inverse @ pooled, 0.0)
    lower = float(centring[0, 0] + np.minimum(centring[0, 1:], 0.0) @ reference[1:])
    bound = max(float(pooled[0]) - lower, 0.0)
    scale = ratio * bound
    total = scale * scale + variance
    g = np.where(determined & (total > 0), np.divide(scale * scale, total, out=np.zeros(n_bins), where=total > 0), 0.0)
    # the correction never exceeds the bound s_k (valid under F, H, S; module docstring); what it holds back is kept
    weighted = g * (pooled - split)
    correction = np.clip(weighted, -scale, scale)
    clipped = np.abs(weighted - correction)
    estimate = inverse @ (pooled - correction)
    rep_scale = np.maximum(rep1[:, 0] - lower, 0.0)[:, None] * ratio[None, :]
    replicates = (rep1 - np.clip(g[None, :] * (rep1 - rep2), -rep_scale, rep_scale)) @ inverse.T
    if n_used >= 3:
        se, se1 = _jackknife(replicates), _jackknife(rep1 @ inverse.T)
        se2 = np.where(determined, _jackknife(rep2 @ inverse.T), np.inf)
    else:
        se = se1 = se2 = np.full(n_bins, np.inf)
    return _Curves(
        estimate=estimate, replicates=replicates, pooled=inverse @ pooled, split=inverse @ split, standard_error=se,
        pooled_se=se1, split_se=se2, weight=g, determined=determined, scale=scale, ratio=ratio, bound=bound,
        samples=n_used, clipped=clipped,
    )


def _band(k, lower, upper, occupied):
    """bool[K]: the bins tested at edge k: those with a lower edge in [R, 2R) (the bin itself at least), the next
    occupied bin when none of them has a pair."""
    index = np.arange(len(lower))
    members = (index >= k) & (lower < max(2 * lower[k], upper[k])) & occupied
    if not members.any():
        later = np.flatnonzero(occupied & (index >= k))
        if len(later):
            members = index == later[0]
    return members


def _range(estimate, replicates, nu, edges, occupied):
    """(first bin beyond the range, status, (T, se T) of the octave that stopped it or None, replicates of S): the
    range rule of the module docstring on the lag bins' signed estimates [K] and their jackknife replicates [N', K]."""
    n_bins = len(estimate)
    lower, upper = edges[:-1].astype(np.float64), edges[1:].astype(np.float64)
    scale = (replicates.shape[0] - 1) / replicates.shape[0]
    inside = np.zeros(replicates.shape[0])  # replicates of the sum over the bins inside so far
    for k in range(n_bins):
        members = _band(k, lower, upper, occupied)
        status = "none_detected" if k == 0 else "resolved_within"
        if not members.any():
            return k, status, None, inside
        band = float((nu[members] * estimate[members]).sum())
        band_reps = replicates[:, members] @ nu[members]
        centred = band_reps - band_reps.mean()
        variance = scale * float((centred * centred).sum())
        covariance = scale * float(((inside - inside.mean()) * centred).sum())
        if not (band > 0.0 and band * band > _RISK_VARIANCE_TERMS * variance + 2.0 * covariance):
            return k, status, (band, math.sqrt(variance)), inside
        inside = inside + nu[k] * replicates[:, k]
    return n_bins, "unresolved", None, inside


@dataclass(frozen=True)
class SharingJackknife:
    """Delete-one estimates on occupied positive-lag bins with a finite estimate and positive finite error."""

    bins: np.ndarray  # bool[K], selects the columns from excess, pairs and the lag intervals
    centered: np.ndarray  # float32[N, bins.sum()], centred over samples with reads; absent rows are zero
    active: np.ndarray  # bool[N], samples with reads, in canonical order


def _sharing_jackknife(curves, pairs, active):
    estimate = curves.estimate[1:]
    error = curves.standard_error[1:]
    bins = (pairs > 0) & np.isfinite(estimate) & np.isfinite(error) & (error > 0)
    if not np.any(bins):
        return None
    deleted = curves.replicates[:, 1:][:, bins]
    centered = np.zeros((len(active), deleted.shape[1]), dtype=np.float32)
    centered[active] = deleted - deleted.mean(axis=0)
    return SharingJackknife(bins, centered, active)


@dataclass(frozen=True)
class ReadSharing:
    """The read-sharing curve of one contig (module docstring)."""

    edges: np.ndarray  # int64[K + 1] lag bin edges (bp)
    excess: np.ndarray  # float64[K] the estimate per bin (standardised excess depth covariance), signed
    standard_error: np.ndarray  # float64[K] delete-one-sample jackknife, conditional on b, w and g
    retained: np.ndarray  # bool[K] the bins inside the range
    phi_bins: np.ndarray  # float64[K] fitted working weights: max(excess, 0) inside the range, zero outside
    range_bp: int  # r_phi (0: none detected)
    status: str  # none_detected, resolved_within or unresolved
    pairs: np.ndarray  # int64[K] pairs per sample in each bin (every pair)
    samples: int
    sites: int
    dispersion: float  # pooled value at lag 0: variance over the Poisson mean (1 for Poisson depths)
    excess_dispersion: float  # D: the lag-0 statistic over the least value of its linear part (uncentred scale)
    pooled: np.ndarray  # float64[K] every covariance read as shared reads
    pooled_se: np.ndarray  # float64[K]
    split: np.ndarray  # float64[K] the part linear in the sample's depth (the pooled value where undetermined)
    split_se: np.ndarray  # float64[K] (inf where undetermined)
    split_share: np.ndarray  # float64[K] g: the weight of the split in ``excess`` (0 where undetermined)
    split_determined: np.ndarray  # bool[K] the full fit and every delete-one fit determine the split
    pair_depth_ratio: np.ndarray  # float64[K] r: the bin's pairs' depth weight over lag 0's
    # the next three are in the uncentred scale (pair sums over W1, before M^-1), conditional on (F), (H), (S)
    intensity_scale: np.ndarray  # float64[K] s = r D: the pooled value's largest intensity part
    unseparated: np.ndarray  # float64[K] (1 - g) s: the intensity part ``excess`` may still hold
    clipped: np.ndarray  # float64[K] the part of the weighted split difference beyond s, held back (0: bound respected)
    depth_spread: float  # coefficient of variation of the depth factors of the samples with reads
    omitted: float = 0.0  # fitted weight (neighbours x phi) of the first omitted octave before the QC addition
    omitted_upper: float = 0.0  # its one-sided 95 % upper limit (that octave only)
    correction: float = 1.0  # 1 - sum of squared depth shares
    horizon_bp: int = 0  # the largest lag measured (the last edge)
    neighbour_weight: float = 0.0  # mean neighbour mass of the composed working curve over all sites
    neighbour_weight_se: float = 0.0  # signed fitted estimate's jackknife standard error at the fixed fitted range
    poisson_dispersion: float = 1.0  # the split's linear part at lag 0 (1 when reads are sampled as Poisson)
    poisson_dispersion_se: float = 0.0
    jackknife: SharingJackknife | None = None
    applied_tail: np.ndarray | None = None  # float64[K] additional QC working weights outside the fitted range

    @property
    def range_unresolved(self) -> bool:
        return self.status == "unresolved"

    @property
    def detectable(self) -> bool:
        return self.status != "none_detected"

    def resolved(self) -> bool:
        """The range rule stopped inside the measured lags (none detected, or resolved within them)."""
        return self.status != "unresolved"

    def consumer_curve(self) -> tuple[np.ndarray, np.ndarray, int]:
        """(edges, composed working weights, reach in bp) for consumer-specific neighbour counts."""
        reach = int(self.edges[-1]) if self.status == "unresolved" else int(self.range_bp)
        if self.applied_tail is None:
            return self.edges, self.phi_bins, reach
        last = int(np.flatnonzero(self.applied_tail)[-1])
        return self.edges, self.phi_bins + self.applied_tail, int(self.edges[last + 1])

    def phi(self, distance: np.ndarray) -> np.ndarray:
        """float64: the consumers' working phi at the given distances (bp), zero beyond the working reach."""
        edges, bins, _ = self.consumer_curve()
        k = np.searchsorted(edges, distance, side="right") - 1
        inside = (k >= 0) & (k < len(bins))
        value = np.zeros(np.shape(distance))
        value[inside] = bins[k[inside]]
        return value


def _undetermined(edges, n_samples, n_sites, pairs=None) -> ReadSharing:
    """The record of depths that do not determine a curve (no reads, one site, fewer than two samples with reads, or a
    singular centring map): unresolved, every value finite, nothing counted."""
    n_bins = len(edges) - 1
    zero = np.zeros(n_bins)
    return ReadSharing(
        edges=edges, excess=zero, standard_error=np.full(n_bins, np.inf), retained=np.ones(n_bins, bool), phi_bins=zero,
        range_bp=int(edges[-1]), status="unresolved", pairs=np.zeros(n_bins, np.int64) if pairs is None else pairs,
        samples=int(n_samples), sites=int(n_sites), dispersion=1.0, excess_dispersion=0.0, pooled=zero,
        pooled_se=np.full(n_bins, np.inf), split=zero, split_se=np.full(n_bins, np.inf), split_share=zero,
        split_determined=np.zeros(n_bins, bool), pair_depth_ratio=zero, intensity_scale=zero, unseparated=zero,
        clipped=zero,
        depth_spread=0.0, horizon_bp=int(edges[-1]), neighbour_weight_se=math.inf, poisson_dispersion_se=math.inf,
    )


def read_sharing(depth: np.ndarray, positions: np.ndarray, contig: str = "") -> ReadSharing:
    """The read-sharing curve (module docstring) of int depth[N, L] (REF + ALT reads; 0 where a sample has no read) at
    ascending positions int64[L] of one contig."""
    depth = np.ascontiguousarray(depth, dtype=np.int64)
    positions = np.ascontiguousarray(positions, dtype=np.int64)
    n_samples, n_sites = depth.shape
    span = int(positions[-1] - positions[0] + 1) if n_sites else 0
    edges = lag_edges(span)
    n_bins = len(edges) - 1
    site_total = depth.sum(axis=0, dtype=np.float64)
    sample_total = depth.sum(axis=1, dtype=np.float64)
    grand = float(sample_total.sum())
    if grand <= 0 or n_sites < 2 or int((sample_total > 0).sum()) < 2:
        return _undetermined(edges, n_samples, n_sites)
    share = sample_total / grand
    b = site_total / n_samples
    sums = np.zeros((n_samples, n_bins + 1))
    frozen = []
    for array in (depth, share, site_total, positions, edges):
        view = np.ascontiguousarray(array).view()
        view.flags.writeable = False
        frozen.append(view)
    _pair_sums(*frozen, sums)
    w1, w2, pairs, centring = _geometry(positions, b, edges)
    return _from_sums(sums, share, w1, w2, pairs, centring, edges, n_sites)


def _from_sums(sums, share, w1, w2, pairs, centring, edges, n_sites) -> ReadSharing:
    """The record from the two kernels' outputs: the per-sample pair sums [N, K + 1] (``_pair_sums``), the samples'
    depth shares [N] and the sites' geometry (``_geometry``). Everything statistical happens here, on K + 1 numbers per
    sample."""
    n_samples, n_bins = len(share), len(edges) - 1
    # the sites must determine the curve: a centring map of full numerical rank (two sites, or every pair inside the
    # measured lags, give a singular one: the residuals of a sample sum to zero)
    if not np.all(np.isfinite(centring)) or not np.linalg.cond(centring) < _CONDITION_LIMIT:
        return _undetermined(edges, n_samples, n_sites, pairs)
    inverse = np.linalg.inv(centring)
    curves = _curves(sums, share, w1, w2, centring, inverse)
    occupied = pairs > 0
    nu = 2.0 * pairs.astype(np.float64) / n_sites
    if curves.samples >= 3:
        beyond, status, last, _ = _range(curves.estimate[1:], curves.replicates[:, 1:], nu, edges, occupied)
    else:
        beyond, status, last = n_bins, "unresolved", None
    estimate = curves.estimate[1:]
    inside = np.arange(n_bins) < beyond
    phi_bins = np.where(inside, np.maximum(estimate, 0.0), 0.0)
    omitted = omitted_upper = 0.0
    if status != "unresolved" and last is not None:
        omitted = max(last[0], 0.0)
        omitted_upper = max(last[0] + UPPER_Z * last[1], 0.0)
    neighbour_weight = float((nu * phi_bins).sum())
    if curves.samples >= 3:
        neighbour_weight_se = float(_jackknife(curves.replicates[:, 1:][:, inside] @ nu[inside]))
    else:
        neighbour_weight_se = math.inf
    a = share[share > 0] * n_samples
    g = curves.weight[1:]
    return ReadSharing(
        edges=edges,
        excess=estimate,
        jackknife=_sharing_jackknife(curves, pairs, share > 0),
        standard_error=curves.standard_error[1:],
        retained=inside,
        phi_bins=phi_bins,
        range_bp=int(edges[beyond]),
        status=status,
        pairs=pairs,
        samples=int(n_samples),
        sites=int(n_sites),
        dispersion=float(curves.pooled[0]),
        excess_dispersion=float(curves.bound),
        pooled=curves.pooled[1:],
        pooled_se=curves.pooled_se[1:],
        split=curves.split[1:],
        split_se=curves.split_se[1:],
        split_share=g,
        split_determined=curves.determined[1:],
        pair_depth_ratio=curves.ratio[1:],
        intensity_scale=curves.scale[1:],
        unseparated=(1.0 - g) * curves.scale[1:],
        clipped=curves.clipped[1:],
        depth_spread=float(a.std() / a.mean()),
        omitted=omitted,
        omitted_upper=omitted_upper,
        correction=1.0 - float((share**2).sum()),
        horizon_bp=int(edges[-1]),
        neighbour_weight=neighbour_weight,
        neighbour_weight_se=neighbour_weight_se,
        poisson_dispersion=float(curves.split[0]),
        poisson_dispersion_se=float(curves.split_se[0]),
    )
